"""
Meta-Inverted Sum Softmax with Huntington Postulates

Alternative to standard softmax using Boolean ring operations:
  Standard: sigma(z)_i = exp(z_i) / sum_j exp(z_j)
  Meta-inverted: p_i = w_i / (1 + sum_j w_j exp(-z_j))

The meta-inverted form satisfies:
  1. Outputs in [0, 1] (Boolean compatible)
  2. Monotonic in z_i (preserves ordering)
  3. Connected to Boolean ring via: w1 XOR w2 = w1 + w2 - 2*w1*w2

Boolean ring operations:
  Addition (XOR): a + b = a + b - 2ab
  Multiplication (AND): a * b = ab
  Complement (NOT): ~a = 1 - a
  Identity (add): 0
  Identity (mul): 1

Connection to Huntington postulates:
  H1-H6 satisfied by the Boolean ring structure
  H7 (idempotence): a + a = a (saturation) - gives convergence

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def bool_ring_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Boolean ring addition (XOR): a + b - 2ab."""
    return a + b - 2.0 * a * b


def bool_ring_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Boolean ring multiplication (AND): ab."""
    return a * b


def bool_ring_not(a: torch.Tensor) -> torch.Tensor:
    """Boolean ring complement (NOT): 1 - a."""
    return 1.0 - a


def meta_inverted_sum_softmax(
    logits: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Meta-inverted sum softmax.

    p_i = w_i / (1 + sum_j w_j * exp(-z_j / tau))

    Args:
        logits: [batch, n] raw scores
        weights: [batch, n] or [n] trait weights in [0, 1] (default: uniform)
        temperature: scaling parameter

    Returns:
        probs: [batch, n] meta-inverted probabilities
    """
    if weights is None:
        weights = torch.ones_like(logits)
    weights = weights.clamp(0, 1)

    # Compute inverted sum: 1 + sum_j w_j * exp(-z_j / tau)
    scaled = logits / temperature
    inv_terms = weights * torch.exp(-scaled)
    inverted_sum = 1.0 + inv_terms.sum(dim=-1, keepdim=True)

    # p_i = w_i / inverted_sum
    probs = weights / inverted_sum

    return probs


def huntington_softmax(
    logits: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    idempotent_steps: int = 3
) -> torch.Tensor:
    """
    Huntington softmax: iterative application of Boolean ring operations
    until idempotence (fixed point).

    Uses H7 (idempotence: a + a = a) as convergence criterion.
    After k applications, the output satisfies a + a = a (saturation).

    Args:
        logits: [batch, n] raw scores
        weights: [batch, n] trait weights
        idempotent_steps: iterations toward idempotence

    Returns:
        probs: [batch, n] idempotent probability-like values
    """
    # Initialize with standard sigmoid (maps to [0, 1])
    p = torch.sigmoid(logits)

    if weights is not None:
        p = bool_ring_mul(p, weights.clamp(0, 1))

    # Iterate toward idempotence: p_{k+1} = p_k XOR p_k should equal p_k
    for _ in range(idempotent_steps):
        # Apply ring addition with self (should converge to fixed point)
        p_xor = bool_ring_add(p, p)
        # Move toward fixed point: p = lerp(p, p_xor, alpha)
        # At fixed point: p_xor = p (idempotence H7)
        residual = (p_xor - p).abs().mean()
        if residual < 1e-6:
            break
        # Gradient step toward idempotence
        p = p - 0.5 * (p_xor - p)
        p = p.clamp(0, 1)

    # Normalize to sum to 1 (probability simplex)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-10)

    return p


class MetaInvertedSumLayer(nn.Module):
    """
    Neural network layer using meta-inverted sum instead of softmax.

    Learns trait weights that satisfy Huntington postulates.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        # Trait weights (learnable, constrained to [0, 1])
        self.trait_weights = nn.Parameter(torch.rand(d_out) * 0.5 + 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with meta-inverted sum activation.

        Args:
            x: [batch, d_in]
        Returns:
            out: [batch, d_out]
        """
        logits = self.linear(x)
        weights = torch.sigmoid(self.trait_weights)  # project to [0, 1]
        return meta_inverted_sum_softmax(logits, weights.unsqueeze(0))

    def huntington_loss(self) -> torch.Tensor:
        """
        Loss term encouraging Huntington postulate satisfaction.
        Primarily targets H7 (idempotence).
        """
        w = torch.sigmoid(self.trait_weights)
        # H7: w XOR w should equal w
        w_xor_w = bool_ring_add(w, w)
        idempotence_loss = (w_xor_w - w).pow(2).mean()

        # H5: w AND (NOT w) should be 0
        complement_loss = bool_ring_mul(w, bool_ring_not(w)).pow(2).mean()

        return idempotence_loss + complement_loss
