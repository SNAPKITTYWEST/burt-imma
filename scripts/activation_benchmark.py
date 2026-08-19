#!/usr/bin/env python3
"""
Benchmark SmoothLeaky activation against standard activations.

License: BSL-1.1
Contact: jessica@collectivekitty.com

SmoothLeaky activation:
    f(x) = alpha * x + (1 - alpha) * x * sigmoid(x)
    where alpha = 0.01 (default)

This combines the non-dying gradient property of LeakyReLU with the
smooth, differentiable nature of SiLU/Swish. Benchmarks forward pass,
backward pass, and peak memory usage against standard activations.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothLeaky(nn.Module):
    """SmoothLeaky activation: alpha * x + (1-alpha) * x * sigmoid(x).

    Combines LeakyReLU's non-dying gradients with SiLU's smoothness.
    At alpha=0, reduces to SiLU/Swish.
    At alpha=1, reduces to identity.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * x * torch.sigmoid(x)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single activation function."""
    name: str
    forward_time_ms: float
    backward_time_ms: float
    peak_memory_mb: float
    forward_std_ms: float
    backward_std_ms: float


def benchmark_activation(
    activation: nn.Module,
    name: str,
    batch_size: int,
    dim: int,
    num_iters: int,
    device: torch.device,
    warmup_iters: int = 50,
) -> BenchmarkResult:
    """Benchmark a single activation function."""

    # Warmup
    for _ in range(warmup_iters):
        x = torch.randn(batch_size, dim, device=device, requires_grad=True)
        y = activation(x)
        loss = y.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iters):
        x = torch.randn(batch_size, dim, device=device, requires_grad=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        y = activation(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        forward_times.append((end - start) * 1000.0)  # ms

        # Clean up
        del y
        del x

    # Benchmark backward pass
    backward_times = []
    for _ in range(num_iters):
        x = torch.randn(batch_size, dim, device=device, requires_grad=True)
        y = activation(x)
        loss = y.sum()

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        backward_times.append((end - start) * 1000.0)  # ms

        del x, y, loss

    # Measure peak memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, dim, device=device, requires_grad=True)
        y = activation(x)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        del x, y, loss
    else:
        # On CPU, we estimate memory based on tensor sizes
        peak_memory_mb = (batch_size * dim * 4 * 3) / (1024 * 1024)  # input + output + grad (float32)

    forward_times_t = torch.tensor(forward_times)
    backward_times_t = torch.tensor(backward_times)

    return BenchmarkResult(
        name=name,
        forward_time_ms=forward_times_t.mean().item(),
        backward_time_ms=backward_times_t.mean().item(),
        peak_memory_mb=peak_memory_mb,
        forward_std_ms=forward_times_t.std().item(),
        backward_std_ms=backward_times_t.std().item(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SmoothLeaky activation against standard activations."
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension (default: 256)")
    parser.add_argument("--num-iters", type=int, default=1000, help="Number of iterations (default: 1000)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--alpha", type=float, default=0.01, help="SmoothLeaky alpha (default: 0.01)")
    args = parser.parse_args()

    # Check device availability
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    print("BURT-IMMA Activation Benchmark")
    print("=" * 80)
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Dimension:    {args.dim}")
    print(f"  Iterations:   {args.num_iters}")
    print(f"  Device:       {device}")
    print(f"  Alpha:        {args.alpha}")
    if device.type == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name(device)}")
    print("=" * 80)
    print()

    # Define activations to benchmark
    activations = [
        (SmoothLeaky(alpha=args.alpha), "SmoothLeaky (ours)"),
        (nn.ReLU(), "ReLU"),
        (nn.LeakyReLU(negative_slope=0.01), "LeakyReLU"),
        (nn.GELU(), "GELU"),
        (nn.SiLU(), "SiLU/Swish"),
        (Mish(), "Mish"),
    ]

    results: List[BenchmarkResult] = []

    for activation, name in activations:
        activation = activation.to(device)
        print(f"  Benchmarking {name}...", end="", flush=True)
        result = benchmark_activation(
            activation=activation,
            name=name,
            batch_size=args.batch_size,
            dim=args.dim,
            num_iters=args.num_iters,
            device=device,
        )
        results.append(result)
        print(f" done ({result.forward_time_ms:.4f} ms fwd)")

    print()
    print("=" * 80)
    print(f"{'Activation':<20} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Mem (MB)':<12} {'Fwd Speedup':<12}")
    print("-" * 80)

    # Use SmoothLeaky as baseline for speedup comparison
    baseline_fwd = results[0].forward_time_ms

    for r in results:
        speedup = baseline_fwd / r.forward_time_ms if r.forward_time_ms > 0 else float("inf")
        print(
            f"{r.name:<20} "
            f"{r.forward_time_ms:<12.4f} "
            f"{r.backward_time_ms:<12.4f} "
            f"{r.peak_memory_mb:<12.2f} "
            f"{speedup:<12.2f}x"
        )

    print("-" * 80)
    print()

    # Detailed statistics
    print("Detailed Statistics (mean +/- std):")
    print(f"{'Activation':<20} {'Forward':<24} {'Backward':<24}")
    print("-" * 68)
    for r in results:
        fwd_str = f"{r.forward_time_ms:.4f} +/- {r.forward_std_ms:.4f} ms"
        bwd_str = f"{r.backward_time_ms:.4f} +/- {r.backward_std_ms:.4f} ms"
        print(f"{r.name:<20} {fwd_str:<24} {bwd_str:<24}")

    print()
    print("Notes:")
    print("  - Speedup > 1.0 means faster than SmoothLeaky")
    print("  - Speedup < 1.0 means slower than SmoothLeaky")
    print(f"  - SmoothLeaky(alpha={args.alpha}): f(x) = {args.alpha}*x + {1-args.alpha:.2f}*x*sigmoid(x)")


if __name__ == "__main__":
    main()
