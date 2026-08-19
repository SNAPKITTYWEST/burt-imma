#!/usr/bin/env python3
"""
CUDA kernel profiling for BURT-IMMA custom kernels.

License: BSL-1.1
Contact: jessica@collectivekitty.com

Profiles the following custom CUDA kernels against PyTorch native implementations:
  - constrained_softmax: Softmax with Boolean mask constraints
  - cifg_update: Coupled Input-Forget Gate (LSTM variant)
  - sparse_moe_dispatch: Sparse Mixture-of-Experts routing
  - biencoder_attention: Bi-encoder cross-attention mechanism

Reports execution time, memory bandwidth utilization, occupancy estimates,
and speedup factors relative to PyTorch baselines.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KernelProfile:
    """Profiling results for a single kernel."""
    name: str
    custom_time_ms: float
    baseline_time_ms: float
    speedup: float
    memory_bandwidth_gbps: float
    estimated_occupancy: float
    custom_std_ms: float = 0.0
    baseline_std_ms: float = 0.0


def time_kernel(
    fn: Callable,
    num_runs: int,
    device: torch.device,
    warmup: int = 20,
) -> Tuple[float, float]:
    """Time a kernel function, returning (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    times_t = torch.tensor(times)
    return times_t.mean().item(), times_t.std().item()


def estimate_memory_bandwidth(
    bytes_accessed: int, time_ms: float
) -> float:
    """Estimate memory bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    time_s = time_ms / 1000.0
    return (bytes_accessed / 1e9) / time_s


def estimate_occupancy(
    shared_mem_per_block: int = 0,
    registers_per_thread: int = 32,
    threads_per_block: int = 256,
    max_threads_per_sm: int = 2048,
    max_blocks_per_sm: int = 32,
    max_shared_mem_per_sm: int = 49152,
    max_registers_per_sm: int = 65536,
) -> float:
    """Estimate kernel occupancy based on resource usage."""
    # Blocks limited by threads
    blocks_by_threads = max_threads_per_sm // threads_per_block

    # Blocks limited by shared memory
    if shared_mem_per_block > 0:
        blocks_by_shmem = max_shared_mem_per_sm // shared_mem_per_block
    else:
        blocks_by_shmem = max_blocks_per_sm

    # Blocks limited by registers
    regs_per_block = registers_per_thread * threads_per_block
    if regs_per_block > 0:
        blocks_by_regs = max_registers_per_sm // regs_per_block
    else:
        blocks_by_regs = max_blocks_per_sm

    active_blocks = min(blocks_by_threads, blocks_by_shmem, blocks_by_regs, max_blocks_per_sm)
    active_threads = active_blocks * threads_per_block
    occupancy = active_threads / max_threads_per_sm
    return min(occupancy, 1.0)


# --- Kernel implementations (PyTorch-based simulations of custom CUDA kernels) ---


def constrained_softmax_custom(
    x: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Custom constrained softmax with Boolean mask.

    Applies mask before softmax to zero out invalid positions,
    then renormalizes. Fused implementation (simulated).
    """
    # Simulate fused kernel: mask + scale + softmax in one pass
    masked = x.masked_fill(~mask, float("-inf"))
    scaled = masked / temperature
    return F.softmax(scaled, dim=-1)


def constrained_softmax_baseline(
    x: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Baseline: separate mask, scale, softmax operations."""
    masked = x.clone()
    masked[~mask] = float("-inf")
    scaled = masked / temperature
    return F.softmax(scaled, dim=-1)


def cifg_update_custom(
    x: torch.Tensor, h_prev: torch.Tensor, W_f: torch.Tensor, W_c: torch.Tensor, b_f: torch.Tensor, b_c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom CIFG (Coupled Input-Forget Gate) update.

    In CIFG, the input gate = 1 - forget gate, reducing parameters.
    Fused implementation computes both gates and cell update in one kernel.
    """
    # Fused: compute forget gate and cell update together
    combined = torch.cat([x, h_prev], dim=-1)
    f_gate = torch.sigmoid(F.linear(combined, W_f, b_f))
    candidate = torch.tanh(F.linear(combined, W_c, b_c))
    # CIFG: i_gate = 1 - f_gate
    c_new = f_gate * h_prev + (1.0 - f_gate) * candidate
    h_new = torch.tanh(c_new)
    return h_new, c_new


def cifg_update_baseline(
    x: torch.Tensor, h_prev: torch.Tensor, W_f: torch.Tensor, W_c: torch.Tensor, b_f: torch.Tensor, b_c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Baseline CIFG using separate operations."""
    combined = torch.cat([x, h_prev], dim=-1)
    # Separate matmuls
    f_pre = torch.mm(combined, W_f.t()) + b_f
    f_gate = torch.sigmoid(f_pre)
    c_pre = torch.mm(combined, W_c.t()) + b_c
    candidate = torch.tanh(c_pre)
    i_gate = 1.0 - f_gate
    c_new = f_gate * h_prev + i_gate * candidate
    h_new = torch.tanh(c_new)
    return h_new, c_new


def sparse_moe_dispatch_custom(
    x: torch.Tensor, gate_logits: torch.Tensor, num_experts: int, top_k: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom sparse MoE dispatch kernel.

    Computes top-k expert selection and dispatches tokens to experts
    in a single fused operation.
    """
    # Top-k gating
    gate_probs = F.softmax(gate_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
    # Renormalize
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    return top_k_probs, top_k_indices, gate_probs


def sparse_moe_dispatch_baseline(
    x: torch.Tensor, gate_logits: torch.Tensor, num_experts: int, top_k: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baseline sparse MoE dispatch using separate operations."""
    gate_probs = F.softmax(gate_logits, dim=-1)
    # Sort all experts
    sorted_probs, sorted_indices = torch.sort(gate_probs, dim=-1, descending=True)
    # Select top-k
    top_k_probs = sorted_probs[:, :top_k]
    top_k_indices = sorted_indices[:, :top_k]
    # Renormalize
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    return top_k_probs, top_k_indices, gate_probs


def biencoder_attention_custom(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Custom bi-encoder attention kernel.

    Fused scaled dot-product attention for bi-encoder architecture.
    q from encoder A, k/v from encoder B.
    """
    # Fused attention: Q @ K^T / scale -> softmax -> @ V
    attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.bmm(attn_weights, v)


def biencoder_attention_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Baseline bi-encoder attention using separate operations."""
    # Separate: matmul, scale, softmax, matmul
    attn_scores = torch.bmm(q, k.transpose(1, 2))
    attn_scores = attn_scores * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.bmm(attn_weights, v)
    return output


def profile_constrained_softmax(
    batch_size: int, seq_len: int, hidden_dim: int, num_runs: int, device: torch.device
) -> KernelProfile:
    """Profile constrained_softmax kernel."""
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    mask = torch.randint(0, 2, (batch_size, seq_len, hidden_dim), dtype=torch.bool, device=device)

    custom_fn = lambda: constrained_softmax_custom(x, mask)
    baseline_fn = lambda: constrained_softmax_baseline(x, mask)

    custom_time, custom_std = time_kernel(custom_fn, num_runs, device)
    baseline_time, baseline_std = time_kernel(baseline_fn, num_runs, device)

    # Memory: read x + mask, write output
    bytes_accessed = batch_size * seq_len * hidden_dim * (4 + 1 + 4)  # float32 + bool + float32
    bandwidth = estimate_memory_bandwidth(bytes_accessed, custom_time)
    occupancy = estimate_occupancy(shared_mem_per_block=hidden_dim * 4)

    return KernelProfile(
        name="constrained_softmax",
        custom_time_ms=custom_time,
        baseline_time_ms=baseline_time,
        speedup=baseline_time / custom_time if custom_time > 0 else 0,
        memory_bandwidth_gbps=bandwidth,
        estimated_occupancy=occupancy,
        custom_std_ms=custom_std,
        baseline_std_ms=baseline_std,
    )


def profile_cifg_update(
    batch_size: int, seq_len: int, hidden_dim: int, num_runs: int, device: torch.device
) -> KernelProfile:
    """Profile cifg_update kernel."""
    input_dim = hidden_dim
    x = torch.randn(batch_size, input_dim, device=device)
    h_prev = torch.randn(batch_size, hidden_dim, device=device)
    W_f = torch.randn(hidden_dim, input_dim + hidden_dim, device=device)
    W_c = torch.randn(hidden_dim, input_dim + hidden_dim, device=device)
    b_f = torch.randn(hidden_dim, device=device)
    b_c = torch.randn(hidden_dim, device=device)

    custom_fn = lambda: cifg_update_custom(x, h_prev, W_f, W_c, b_f, b_c)
    baseline_fn = lambda: cifg_update_baseline(x, h_prev, W_f, W_c, b_f, b_c)

    custom_time, custom_std = time_kernel(custom_fn, num_runs, device)
    baseline_time, baseline_std = time_kernel(baseline_fn, num_runs, device)

    # Memory: weights + inputs + outputs
    bytes_accessed = (2 * hidden_dim * (input_dim + hidden_dim) + batch_size * (input_dim + 3 * hidden_dim)) * 4
    bandwidth = estimate_memory_bandwidth(bytes_accessed, custom_time)
    occupancy = estimate_occupancy(registers_per_thread=48)

    return KernelProfile(
        name="cifg_update",
        custom_time_ms=custom_time,
        baseline_time_ms=baseline_time,
        speedup=baseline_time / custom_time if custom_time > 0 else 0,
        memory_bandwidth_gbps=bandwidth,
        estimated_occupancy=occupancy,
        custom_std_ms=custom_std,
        baseline_std_ms=baseline_std,
    )


def profile_sparse_moe_dispatch(
    batch_size: int, seq_len: int, hidden_dim: int, num_runs: int, device: torch.device
) -> KernelProfile:
    """Profile sparse_moe_dispatch kernel."""
    num_experts = 8
    top_k = 2
    x = torch.randn(batch_size * seq_len, hidden_dim, device=device)
    gate_logits = torch.randn(batch_size * seq_len, num_experts, device=device)

    custom_fn = lambda: sparse_moe_dispatch_custom(x, gate_logits, num_experts, top_k)
    baseline_fn = lambda: sparse_moe_dispatch_baseline(x, gate_logits, num_experts, top_k)

    custom_time, custom_std = time_kernel(custom_fn, num_runs, device)
    baseline_time, baseline_std = time_kernel(baseline_fn, num_runs, device)

    bytes_accessed = batch_size * seq_len * (num_experts * 4 + top_k * 8)  # logits + indices
    bandwidth = estimate_memory_bandwidth(bytes_accessed, custom_time)
    occupancy = estimate_occupancy(shared_mem_per_block=num_experts * 4 * 32)

    return KernelProfile(
        name="sparse_moe_dispatch",
        custom_time_ms=custom_time,
        baseline_time_ms=baseline_time,
        speedup=baseline_time / custom_time if custom_time > 0 else 0,
        memory_bandwidth_gbps=bandwidth,
        estimated_occupancy=occupancy,
        custom_std_ms=custom_std,
        baseline_std_ms=baseline_std,
    )


def profile_biencoder_attention(
    batch_size: int, seq_len: int, hidden_dim: int, num_runs: int, device: torch.device
) -> KernelProfile:
    """Profile biencoder_attention kernel."""
    num_heads = 8
    head_dim = hidden_dim // num_heads
    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(batch_size * num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size * num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size * num_heads, seq_len, head_dim, device=device)

    custom_fn = lambda: biencoder_attention_custom(q, k, v, scale)
    baseline_fn = lambda: biencoder_attention_baseline(q, k, v, scale)

    custom_time, custom_std = time_kernel(custom_fn, num_runs, device)
    baseline_time, baseline_std = time_kernel(baseline_fn, num_runs, device)

    # Memory: Q, K, V reads + attention matrix + output
    bytes_accessed = batch_size * num_heads * (3 * seq_len * head_dim + seq_len * seq_len + seq_len * head_dim) * 4
    bandwidth = estimate_memory_bandwidth(bytes_accessed, custom_time)
    occupancy = estimate_occupancy(shared_mem_per_block=seq_len * head_dim * 4)

    return KernelProfile(
        name="biencoder_attention",
        custom_time_ms=custom_time,
        baseline_time_ms=baseline_time,
        speedup=baseline_time / custom_time if custom_time > 0 else 0,
        memory_bandwidth_gbps=bandwidth,
        estimated_occupancy=occupancy,
        custom_std_ms=custom_std,
        baseline_std_ms=baseline_std,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Profile BURT-IMMA CUDA kernels against PyTorch baselines."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="all",
        choices=["all", "softmax", "cifg", "moe", "attention"],
        help="Which kernel to profile (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension (default: 256)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of profiling runs (default: 100)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    # Check device
    if "cuda" in args.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        print("Note: Profiling on CPU will not reflect true kernel performance.")
        args.device = "cpu"

    device = torch.device(args.device)

    print("BURT-IMMA CUDA Kernel Profiler")
    print("=" * 80)
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Seq length:    {args.seq_len}")
    print(f"  Hidden dim:    {args.hidden_dim}")
    print(f"  Num runs:      {args.num_runs}")
    print(f"  Device:        {device}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(device)}")
    print("=" * 80)
    print()

    results: List[KernelProfile] = []

    kernel_map = {
        "softmax": ("constrained_softmax", profile_constrained_softmax),
        "cifg": ("cifg_update", profile_cifg_update),
        "moe": ("sparse_moe_dispatch", profile_sparse_moe_dispatch),
        "attention": ("biencoder_attention", profile_biencoder_attention),
    }

    kernels_to_run = list(kernel_map.keys()) if args.kernel == "all" else [args.kernel]

    for kernel_key in kernels_to_run:
        name, profile_fn = kernel_map[kernel_key]
        print(f"  Profiling {name}...", end="", flush=True)
        result = profile_fn(args.batch_size, args.seq_len, args.hidden_dim, args.num_runs, device)
        results.append(result)
        print(f" done (speedup: {result.speedup:.2f}x)")

    # Results table
    print()
    print("=" * 80)
    print(f"{'Kernel':<24} {'Custom (ms)':<14} {'Baseline (ms)':<14} {'Speedup':<10} {'BW (GB/s)':<12} {'Occupancy':<10}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<24} "
            f"{r.custom_time_ms:<14.4f} "
            f"{r.baseline_time_ms:<14.4f} "
            f"{r.speedup:<10.2f}x "
            f"{r.memory_bandwidth_gbps:<12.1f} "
            f"{r.estimated_occupancy:<10.1%}"
        )

    print("-" * 80)
    print()

    # Detailed timing with standard deviations
    print("Detailed Timing (mean +/- std ms):")
    print(f"{'Kernel':<24} {'Custom':<24} {'Baseline':<24}")
    print("-" * 72)
    for r in results:
        custom_str = f"{r.custom_time_ms:.4f} +/- {r.custom_std_ms:.4f}"
        baseline_str = f"{r.baseline_time_ms:.4f} +/- {r.baseline_std_ms:.4f}"
        print(f"{r.name:<24} {custom_str:<24} {baseline_str:<24}")

    print()

    # Summary
    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        max_speedup = max(r.speedup for r in results)
        min_speedup = min(r.speedup for r in results)
        print(f"Speedup Summary:")
        print(f"  Average: {avg_speedup:.2f}x")
        print(f"  Best:    {max_speedup:.2f}x ({max((r for r in results), key=lambda r: r.speedup).name})")
        print(f"  Worst:   {min_speedup:.2f}x ({min((r for r in results), key=lambda r: r.speedup).name})")

    print()
    print("Notes:")
    print("  - Custom kernels are PyTorch-simulated (actual CUDA kernels in src/kernels/)")
    print("  - Speedup > 1.0 means custom kernel is faster than PyTorch baseline")
    print("  - Memory bandwidth is estimated from bytes accessed / kernel time")
    print("  - Occupancy is estimated from resource usage (actual may vary)")


if __name__ == "__main__":
    main()
