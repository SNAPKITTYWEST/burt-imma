#!/usr/bin/env python3
"""
MMEP Gradient vs Backprop Gradient Comparison

Computes gradients via both MMEP (free/nudged phase difference) and
standard backpropagation, then reports cosine similarity and L2 distance.

Usage:
  python scripts/gradient_comparison.py --hidden-dim 256 --num-layers 4

Contact: jessica@collectivekitty.com
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mmep_gradient(model, x, target, T_free=20, T_nudge=4,
                          alpha=0.5, beta=0.1):
    """Compute EP gradient via free/nudged phase difference."""
    # Free phase
    h = x.clone()
    for _ in range(T_free):
        h = (1 - alpha) * h + alpha * torch.tanh(model(h))
    h_free = h.detach().clone()

    # Nudged phase
    h = h_free.clone()
    for _ in range(T_nudge):
        h = (1 - alpha) * h + alpha * torch.tanh(model(h))
        h = h + beta * (target - h)
    h_nudged = h.detach().clone()

    # EP gradient: (1/beta) * (correlation_nudged - correlation_free)
    # For weight gradient: dW ~ (1/beta) * (h_nudged @ x^T - h_free @ x^T)
    grad_ep = (1.0 / beta) * (h_nudged - h_free).mean(dim=0)
    return grad_ep


def compute_backprop_gradient(model, x, target):
    """Compute standard backprop gradient."""
    model.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    # Return gradient of first parameter as representative
    for p in model.parameters():
        if p.grad is not None:
            return p.grad.flatten()[:target.shape[-1]]
    return torch.zeros(target.shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--T-free", type=int, default=20)
    parser.add_argument("--T-nudge", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    d = args.hidden_dim
    model = nn.Linear(d, d, bias=False)

    torch.manual_seed(42)
    cos_sims = []

    for _ in range(args.num_samples):
        x = torch.randn(8, d)
        target = torch.randn(8, d)

        grad_ep = compute_mmep_gradient(model, x, target,
                                         T_free=args.T_free, T_nudge=args.T_nudge,
                                         beta=args.beta)
        grad_bp = compute_backprop_gradient(model, x, target)

        # Align dimensions
        min_dim = min(grad_ep.shape[0], grad_bp.shape[0])
        cos_sim = F.cosine_similarity(
            grad_ep[:min_dim].unsqueeze(0),
            grad_bp[:min_dim].unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

    avg_cos = sum(cos_sims) / len(cos_sims)
    print(f"Gradient Comparison (d={d}, L={args.num_layers})")
    print(f"  Mean cosine similarity: {avg_cos:.4f}")
    print(f"  Min: {min(cos_sims):.4f}, Max: {max(cos_sims):.4f}")
    print(f"  (1.0 = perfect alignment, expected to improve with more T_free)")


if __name__ == "__main__":
    main()
