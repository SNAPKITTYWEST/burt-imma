#!/usr/bin/env python3
"""
BURT-IMMA Benchmark Script
License: BSL-1.1
Contact: jessica@collectivekitty.com

Benchmarks forward pass, backward pass, memory update, and routing
for different batch sizes and hidden dimensions.
"""

import argparse
import time
import sys
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Benchmarks require PyTorch.")

import numpy as np


# ---------------------------------------------------------------------------
# Minimal PyTorch models for benchmarking
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class BenchSmoothLeaky(nn.Module):
        def __init__(self, alpha: float = 0.01):
            super().__init__()
            self.alpha = alpha

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.alpha * x + (1.0 - self.alpha) * x * torch.sigmoid(x)

    class BenchCIFGMemory(nn.Module):
        def __init__(self, d_mem: int):
            super().__init__()
            self.d_mem = d_mem
            self.C = nn.Parameter(torch.zeros(d_mem, d_mem), requires_grad=False)

        def update(self, key: torch.Tensor, value: torch.Tensor,
                   forget_gate: float = 0.9) -> torch.Tensor:
            candidate = torch.outer(key, value)
            candidate = candidate / (candidate.norm() + 1e-8)
            new_C = forget_gate * self.C + (1.0 - forget_gate) * candidate
            self.C.data.copy_(new_C)
            return new_C

        def read(self, query: torch.Tensor) -> torch.Tensor:
            return self.C @ query

    class BenchGatesRouter(nn.Module):
        def __init__(self, input_dim: int, num_experts: int, top_k: int = 1):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.gate = nn.Linear(input_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Linear(input_dim, input_dim) for _ in range(num_experts)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, dim)
            scores = self.gate(x)  # (batch, num_experts)
            probs = torch.softmax(scores, dim=-1)
            top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
            top_k_vals = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

            output = torch.zeros_like(x)
            for k in range(self.top_k):
                for e in range(self.num_experts):
                    mask = (top_k_idx[:, k] == e)
                    if mask.any():
                        expert_out = self.experts[e](x[mask])
                        output[mask] += top_k_vals[mask, k:k+1] * expert_out
            return output

    class BenchBURTIMMA(nn.Module):
        def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 1):
            super().__init__()
            self.activation = BenchSmoothLeaky()
            self.memory = BenchCIFGMemory(hidden_dim)
            self.router = BenchGatesRouter(hidden_dim, num_experts, top_k)
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, hidden_dim)
            batch, seq_len, dim = x.shape
            h = self.activation(x)
            # Flatten for routing
            h_flat = h.reshape(batch * seq_len, dim)
            routed = self.router(h_flat)
            routed = routed.reshape(batch, seq_len, dim)
            out = self.norm(self.proj(routed) + x)
            return out


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, name: str, batch_size: int, hidden_dim: int,
                 mean_time_ms: float, std_time_ms: float,
                 throughput: float, peak_memory_mb: float):
        self.name = name
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.mean_time_ms = mean_time_ms
        self.std_time_ms = std_time_ms
        self.throughput = throughput
        self.peak_memory_mb = peak_memory_mb


def benchmark_forward(model: "nn.Module", x: "torch.Tensor",
                      num_warmup: int, num_runs: int,
                      device: str) -> Tuple[float, float, float]:
    """Benchmark forward pass. Returns (mean_ms, std_ms, peak_memory_mb)."""
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    peak_mem = 0.0
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    return np.mean(times), np.std(times), peak_mem


def benchmark_backward(model: "nn.Module", x: "torch.Tensor",
                       num_warmup: int, num_runs: int,
                       device: str) -> Tuple[float, float, float]:
    """Benchmark backward pass. Returns (mean_ms, std_ms, peak_memory_mb)."""
    # Warmup
    for _ in range(num_warmup):
        out = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(num_runs):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(x)
        loss = out.sum()
        loss.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()
        model.zero_grad()
        times.append((end - start) * 1000)

    peak_mem = 0.0
    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return np.mean(times), np.std(times), peak_mem


def benchmark_memory_update(hidden_dim: int, num_warmup: int, num_runs: int,
                            device: str) -> Tuple[float, float]:
    """Benchmark CIFG memory update."""
    mem = BenchCIFGMemory(hidden_dim).to(device)
    key = torch.randn(hidden_dim, device=device)
    value = torch.randn(hidden_dim, device=device)

    for _ in range(num_warmup):
        mem.update(key, value)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        mem.update(key, value)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def benchmark_routing(hidden_dim: int, batch_size: int, num_experts: int,
                      top_k: int, num_warmup: int, num_runs: int,
                      device: str) -> Tuple[float, float]:
    """Benchmark routing dispatch."""
    router = BenchGatesRouter(hidden_dim, num_experts, top_k).to(device)
    x = torch.randn(batch_size, hidden_dim, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = router(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = router(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results: List[BenchmarkResult]):
    """Print benchmark results as formatted table."""
    header = f"{'Benchmark':<25} {'Batch':<7} {'Hidden':<7} {'Mean(ms)':<10} {'Std(ms)':<9} {'Throughput':<14} {'Memory(MB)':<10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        throughput_str = f"{r.throughput:.1f} samp/s"
        mem_str = f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb > 0 else "N/A"
        print(f"{r.name:<25} {r.batch_size:<7} {r.hidden_dim:<7} "
              f"{r.mean_time_ms:<10.3f} {r.std_time_ms:<9.3f} "
              f"{throughput_str:<14} {mem_str:<10}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="BURT-IMMA Benchmark Suite")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to benchmark on (cpu or cuda)")
    parser.add_argument("--num-warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch is required for benchmarks.")
        sys.exit(1)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"BURT-IMMA Benchmark Suite")
    print(f"Device: {device}")
    print(f"Warmup: {args.num_warmup}, Runs: {args.num_runs}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch_sizes = [1, 8, 32, 64]
    hidden_dims = [64, 128, 256, 512]
    seq_len = 16
    num_experts = 4
    top_k = 1

    results: List[BenchmarkResult] = []

    # Forward pass benchmarks
    print("\n--- Forward Pass ---")
    for bs in batch_sizes:
        for hd in hidden_dims:
            model = BenchBURTIMMA(hd, num_experts, top_k).to(device)
            model.eval()
            x = torch.randn(bs, seq_len, hd, device=device)

            mean_ms, std_ms, peak_mem = benchmark_forward(
                model, x, args.num_warmup, args.num_runs, device
            )
            throughput = (bs * seq_len) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                "forward", bs, hd, mean_ms, std_ms, throughput, peak_mem
            ))

    # Backward pass benchmarks
    print("--- Backward Pass ---")
    for bs in batch_sizes[:2]:  # Fewer configs for backward
        for hd in hidden_dims[:2]:
            model = BenchBURTIMMA(hd, num_experts, top_k).to(device)
            model.train()
            x = torch.randn(bs, seq_len, hd, device=device, requires_grad=True)

            mean_ms, std_ms, peak_mem = benchmark_backward(
                model, x, args.num_warmup, args.num_runs, device
            )
            throughput = (bs * seq_len) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                "backward", bs, hd, mean_ms, std_ms, throughput, peak_mem
            ))

    # Memory update benchmarks
    print("--- Memory Update ---")
    for hd in hidden_dims:
        mean_ms, std_ms = benchmark_memory_update(
            hd, args.num_warmup, args.num_runs, device
        )
        throughput = 1.0 / (mean_ms / 1000.0)
        results.append(BenchmarkResult(
            "memory_update", 1, hd, mean_ms, std_ms, throughput, 0.0
        ))

    # Routing benchmarks
    print("--- Routing Dispatch ---")
    for bs in batch_sizes:
        for hd in hidden_dims[:2]:
            mean_ms, std_ms = benchmark_routing(
                hd, bs, num_experts, top_k,
                args.num_warmup, args.num_runs, device
            )
            throughput = bs / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                "routing", bs, hd, mean_ms, std_ms, throughput, 0.0
            ))

    # Print results
    print_table(results)


if __name__ == "__main__":
    main()
