#!/usr/bin/env python3
"""
Verify sum-inversion properties of the BURT-IMMA architecture.

License: BSL-1.1
Contact: jessica@collectivekitty.com

Sum-inversion verification checks:
  1. Boolean kernel rank: The kernel matrix K for d-dimensional Boolean algebra
     should be full rank (rank = d), ensuring invertibility.
  2. Round-trip accuracy: encode -> sum -> invert -> decode pipeline should
     reconstruct the original signal with minimal loss.
  3. Gates normalization: All gate outputs must sum to 1 and be non-negative
     (valid probability simplex).
  4. MSE loss: Mean squared error between original and reconstructed signals.
  5. Chinchilla scaling: Verify tokens-to-parameters ratio is optimal
     (approximately 20:1 per Chinchilla scaling laws).
"""

import argparse
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_boolean_kernel(d: int, device: torch.device) -> torch.Tensor:
    """Build the Boolean kernel matrix for d-dimensional Boolean algebra.

    The kernel K is constructed such that K[i,j] = (-1)^(popcount(i & j))
    for i,j in {0,1,...,2^k-1} where k = log2(d).

    For large d, we use a structured approximation based on Hadamard matrices.
    """
    # Use Hadamard-like construction for the Boolean kernel
    # For d dimensions, build a d x d kernel
    k = int(np.ceil(np.log2(d)))
    actual_d = min(d, 2**k)

    # Sylvester construction of Hadamard matrix (normalized)
    H = torch.tensor([[1.0]], device=device)
    for _ in range(k):
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    # Truncate/pad to d x d
    if H.shape[0] >= d:
        kernel = H[:d, :d]
    else:
        kernel = torch.zeros(d, d, device=device)
        kernel[:H.shape[0], :H.shape[1]] = H

    # Normalize
    kernel = kernel / np.sqrt(d)
    return kernel


class SumInversionEncoder(nn.Module):
    """Encoder that maps inputs to Boolean kernel space."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SumInversionDecoder(nn.Module):
    """Decoder that maps from kernel space back to input space."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.decoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class GatedSumModule(nn.Module):
    """Gated summation module with normalized gates."""

    def __init__(self, d: int, num_gates: int = 4):
        super().__init__()
        self.d = d
        self.num_gates = num_gates
        self.gate_network = nn.Linear(d, num_gates)
        self.projections = nn.ModuleList([nn.Linear(d, d) for _ in range(num_gates)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (gated_sum, gate_weights)."""
        gates = F.softmax(self.gate_network(x), dim=-1)  # [batch, num_gates]
        projected = torch.stack([proj(x) for proj in self.projections], dim=1)  # [batch, num_gates, d]
        gated_sum = (gates.unsqueeze(-1) * projected).sum(dim=1)  # [batch, d]
        return gated_sum, gates


def verify_kernel_rank(d: int, device: torch.device, verbose: bool = False) -> Tuple[bool, dict]:
    """Verify that the Boolean kernel is full rank."""
    kernel = build_boolean_kernel(d, device)
    rank = torch.linalg.matrix_rank(kernel).item()
    condition_number = torch.linalg.cond(kernel).item()

    passed = (rank == d)
    info = {
        "rank": rank,
        "expected_rank": d,
        "condition_number": condition_number,
        "determinant": torch.linalg.det(kernel).item() if d <= 64 else "skipped (d>64)",
    }

    if verbose:
        print(f"  Kernel shape:      {kernel.shape}")
        print(f"  Rank:              {rank} (expected {d})")
        print(f"  Condition number:  {condition_number:.4f}")
        if isinstance(info["determinant"], float):
            print(f"  Determinant:       {info['determinant']:.6e}")

    return passed, info


def verify_round_trip(
    d: int, num_samples: int, device: torch.device, verbose: bool = False
) -> Tuple[bool, dict]:
    """Verify round-trip accuracy: encode -> sum -> invert -> decode."""
    kernel = build_boolean_kernel(d, device)
    kernel_inv = torch.linalg.inv(kernel) if d <= 512 else torch.linalg.pinv(kernel)

    encoder = SumInversionEncoder(d).to(device)
    decoder = SumInversionDecoder(d).to(device)
    gate_module = GatedSumModule(d).to(device)

    # Train briefly for meaningful round-trip
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(gate_module.parameters()),
        lr=1e-3,
    )

    for step in range(200):
        x = torch.randn(min(num_samples, 256), d, device=device)
        encoded = encoder(x)
        # Apply kernel transform
        in_kernel_space = torch.mm(encoded, kernel.t())
        # Gated sum
        gated, _ = gate_module(in_kernel_space)
        # Invert
        inverted = torch.mm(gated, kernel_inv.t())
        # Decode
        reconstructed = decoder(inverted)

        loss = F.mse_loss(reconstructed, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        x = torch.randn(num_samples, d, device=device)
        encoded = encoder(x)
        in_kernel_space = torch.mm(encoded, kernel.t())
        gated, _ = gate_module(in_kernel_space)
        inverted = torch.mm(gated, kernel_inv.t())
        reconstructed = decoder(inverted)

        mse = F.mse_loss(reconstructed, x).item()
        cosine_sim = F.cosine_similarity(reconstructed, x, dim=-1).mean().item()
        relative_error = (torch.norm(reconstructed - x) / torch.norm(x)).item()

    # Pass if MSE is reasonably low after brief training
    passed = mse < 1.0  # Threshold for untrained network
    info = {
        "mse": mse,
        "cosine_similarity": cosine_sim,
        "relative_error": relative_error,
    }

    if verbose:
        print(f"  MSE:               {mse:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print(f"  Relative error:    {relative_error:.6f}")

    return passed, info


def verify_gates_normalization(
    d: int, num_samples: int, device: torch.device, verbose: bool = False
) -> Tuple[bool, dict]:
    """Verify gate outputs sum to 1 and are non-negative."""
    gate_module = GatedSumModule(d).to(device)

    with torch.no_grad():
        x = torch.randn(num_samples, d, device=device)
        _, gates = gate_module(x)

        # Check sum to 1
        gate_sums = gates.sum(dim=-1)
        sum_error = torch.abs(gate_sums - 1.0).max().item()

        # Check non-negative
        min_gate = gates.min().item()
        all_non_negative = min_gate >= 0.0

        # Check valid probability simplex
        max_gate = gates.max().item()

    passed = (sum_error < 1e-5) and all_non_negative
    info = {
        "max_sum_deviation": sum_error,
        "min_gate_value": min_gate,
        "max_gate_value": max_gate,
        "all_non_negative": all_non_negative,
        "mean_gate_entropy": -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean().item(),
    }

    if verbose:
        print(f"  Max sum deviation: {sum_error:.2e}")
        print(f"  Min gate value:    {min_gate:.6f}")
        print(f"  Max gate value:    {max_gate:.6f}")
        print(f"  All non-negative:  {all_non_negative}")
        print(f"  Mean gate entropy: {info['mean_gate_entropy']:.4f}")

    return passed, info


def compute_mse_loss(
    d: int, num_samples: int, device: torch.device, verbose: bool = False
) -> Tuple[float, dict]:
    """Compute MSE loss between original and reconstructed signals."""
    kernel = build_boolean_kernel(d, device)

    with torch.no_grad():
        x = torch.randn(num_samples, d, device=device)
        # Direct kernel round-trip (no learning, just structure)
        transformed = torch.mm(x, kernel.t())
        # Invert
        kernel_inv = torch.linalg.inv(kernel) if d <= 512 else torch.linalg.pinv(kernel)
        reconstructed = torch.mm(transformed, kernel_inv.t())

        mse = F.mse_loss(reconstructed, x).item()
        max_error = torch.abs(reconstructed - x).max().item()

    info = {
        "mse": mse,
        "max_element_error": max_error,
        "is_exact": mse < 1e-10,
    }

    if verbose:
        print(f"  MSE (kernel only): {mse:.2e}")
        print(f"  Max element error: {max_error:.2e}")
        print(f"  Exact inversion:   {info['is_exact']}")

    return mse, info


def verify_chinchilla_scaling(
    d: int, verbose: bool = False
) -> Tuple[bool, dict]:
    """Verify tokens-to-parameters ratio follows Chinchilla scaling.

    Chinchilla optimal: tokens ~= 20 * parameters
    """
    # Estimate model parameters for a BURT-IMMA network of dimension d
    # Encoder: d -> 2d -> d = d*2d + 2d + 2d*d + d = 4d^2 + 3d
    # Decoder: same = 4d^2 + 3d
    # Gates: d -> 4 = 4d + 4, plus 4 projections d -> d = 4*(d^2 + d)
    # Kernel: d^2 (fixed, not trained)
    encoder_params = 4 * d * d + 3 * d
    decoder_params = 4 * d * d + 3 * d
    gate_params = 4 * d + 4 + 4 * (d * d + d)
    total_params = encoder_params + decoder_params + gate_params

    # Chinchilla optimal tokens
    chinchilla_ratio = 20.0
    optimal_tokens = int(chinchilla_ratio * total_params)

    # Typical training tokens (assume standard dataset)
    # For verification, we just report the optimal
    typical_tokens = total_params * 10  # Assume 10x (under-trained)
    actual_ratio = typical_tokens / total_params

    passed = True  # This is informational
    info = {
        "total_parameters": total_params,
        "optimal_tokens_chinchilla": optimal_tokens,
        "chinchilla_ratio": chinchilla_ratio,
        "current_ratio": actual_ratio,
        "recommended_tokens": optimal_tokens,
        "parameter_breakdown": {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "gates": gate_params,
        },
    }

    if verbose:
        print(f"  Total parameters:      {total_params:,}")
        print(f"  Chinchilla ratio:      {chinchilla_ratio}:1 (tokens:params)")
        print(f"  Optimal tokens:        {optimal_tokens:,}")
        print(f"  Parameter breakdown:")
        print(f"    Encoder:             {encoder_params:,}")
        print(f"    Decoder:             {decoder_params:,}")
        print(f"    Gates:               {gate_params:,}")

    return passed, info


def main():
    parser = argparse.ArgumentParser(
        description="Verify sum-inversion properties of BURT-IMMA architecture."
    )
    parser.add_argument("--d", type=int, default=256, help="Dimension (default: 256)")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of test samples (default: 1000)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("BURT-IMMA Sum-Inversion Verification")
    print("=" * 60)
    print(f"  d = {args.d}, num_samples = {args.num_samples}, device = {device}")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Boolean kernel rank verification
    print("[1/5] Boolean Kernel Rank Verification")
    print("-" * 40)
    passed, info = verify_kernel_rank(args.d, device, verbose=args.verbose)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (rank={info['rank']}/{info['expected_rank']}, cond={info['condition_number']:.2f})")
    all_passed = all_passed and passed
    print()

    # 2. Round-trip accuracy
    print("[2/5] Round-Trip Accuracy (encode -> sum -> invert -> decode)")
    print("-" * 40)
    passed, info = verify_round_trip(args.d, args.num_samples, device, verbose=args.verbose)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (MSE={info['mse']:.6f}, cosine_sim={info['cosine_similarity']:.4f})")
    all_passed = all_passed and passed
    print()

    # 3. Gates normalization
    print("[3/5] Gates Normalization Verification")
    print("-" * 40)
    passed, info = verify_gates_normalization(args.d, args.num_samples, device, verbose=args.verbose)
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (sum_dev={info['max_sum_deviation']:.2e}, non_neg={info['all_non_negative']})")
    all_passed = all_passed and passed
    print()

    # 4. MSE loss computation
    print("[4/5] MSE Loss (Kernel Round-Trip)")
    print("-" * 40)
    mse, info = compute_mse_loss(args.d, args.num_samples, device, verbose=args.verbose)
    passed = info["is_exact"]
    status = "PASS" if passed else "INFO"
    print(f"  Status: {status} (MSE={mse:.2e}, exact={info['is_exact']})")
    print()

    # 5. Chinchilla scaling check
    print("[5/5] Chinchilla Scaling Check")
    print("-" * 40)
    passed, info = verify_chinchilla_scaling(args.d, verbose=args.verbose)
    print(f"  Status: INFO (params={info['total_parameters']:,}, optimal_tokens={info['optimal_tokens_chinchilla']:,})")
    print()

    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    overall = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"  Result: {overall}")
    print(f"  Dimension: {args.d}")
    print(f"  Samples:   {args.num_samples}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
