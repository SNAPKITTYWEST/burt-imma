"""
Perceptron Expert and Router

Integrates PerceptronActor with IMMA expert routing.
The router uses Boolean ring operations for expert selection,
and each expert is a PerceptronNetwork with CIFG memory.

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .perceptron_actor import PerceptronActor, PerceptronNetwork, Signal, SignalType
from .meta_inverted_sum import meta_inverted_sum_softmax, bool_ring_add, bool_ring_mul


class PerceptronExpert(nn.Module):
    """
    Expert module built from perceptron actors.

    Each expert is a PerceptronNetwork with:
      - Boolean ring routing between actors
      - CIFG memory for state retention
      - Huntington-compatible weight updates

    Args:
        d: hidden dimension
        num_actors: actors per expert
        d_mem: memory dimension
    """

    def __init__(self, d: int, num_actors: int = 4, d_mem: int = None):
        super().__init__()
        self.d = d
        self.d_mem = d_mem or d
        self.num_actors = num_actors

        # Perceptron network core
        self.network = PerceptronNetwork(d, num_actors=num_actors)

        # CIFG memory components
        self.W_f = nn.Linear(d, d)  # forget gate
        self.W_v = nn.Linear(d, self.d_mem)  # value projection
        self.W_k = nn.Linear(d, self.d_mem)  # key projection

        # Output projection
        self.output_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        C_prev: Optional[torch.Tensor] = None,
        steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expert forward pass.

        Args:
            x: [batch, d] input
            C_prev: [batch, d_mem, d_mem] previous memory (or None)
            steps: perceptron propagation steps

        Returns:
            output: [batch, d]
            C: [batch, d_mem, d_mem] updated memory
        """
        batch = x.shape[0]

        # Initialize memory if needed
        if C_prev is None:
            C_prev = torch.zeros(batch, self.d_mem, self.d_mem, device=x.device)

        # Run perceptron network
        h = self.network(x, steps=steps)

        # CIFG memory update
        f = torch.sigmoid(self.W_f(h))  # [batch, d]
        i = 1.0 - f  # CIFG constraint

        v = self.W_v(h)  # [batch, d_mem]
        k = self.W_k(h)  # [batch, d_mem]

        # Outer product update
        outer = v.unsqueeze(-1) @ k.unsqueeze(-2)  # [batch, d_mem, d_mem]
        f_scale = f.mean(dim=-1, keepdim=True).unsqueeze(-1)
        i_scale = i.mean(dim=-1, keepdim=True).unsqueeze(-1)
        C = f_scale * C_prev + i_scale * outer

        # Output
        output = self.ln(self.output_proj(h))

        return output, C


class PerceptronRouter(nn.Module):
    """
    Expert router using Boolean ring operations.

    Routes tokens to experts using meta-inverted sum softmax
    with entropy constraint. The routing weights are trait vectors
    in the Boolean ring, satisfying Huntington postulates.

    Args:
        d: hidden dimension
        num_experts: number of experts
        top_k: experts per token
        entropy_bound: max entropy constraint (default 0.20)
    """

    def __init__(self, d: int, num_experts: int = 4, top_k: int = 1,
                 entropy_bound: float = 0.20):
        super().__init__()
        self.d = d
        self.num_experts = num_experts
        self.top_k = top_k
        self.entropy_bound = entropy_bound

        # Gate projection
        self.W_gate = nn.Linear(d, num_experts)

        # Trait weights (one per expert, learnable Boolean ring elements)
        self.trait_weights = nn.Parameter(torch.rand(num_experts) * 0.5 + 0.25)

        # Perceptron actor for routing decisions
        self.routing_actor = PerceptronActor(num_experts, actor_id=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to experts.

        Args:
            x: [batch, d] input

        Returns:
            indices: [batch, top_k] selected expert indices
            weights: [batch, top_k] gating weights (normalized)
            aux: dict with entropy, balance_loss, trait_weights
        """
        # Compute gate logits
        logits = self.W_gate(x)  # [batch, num_experts]

        # Apply meta-inverted sum with trait weights
        trait_w = torch.sigmoid(self.trait_weights)  # project to [0, 1]
        probs = meta_inverted_sum_softmax(logits, trait_w.unsqueeze(0))

        # Entropy check
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch]

        # If entropy exceeds bound, apply Boolean ring sharpening
        violations = entropy > self.entropy_bound
        if violations.any():
            # Apply idempotent sharpening: p XOR p = p (at convergence)
            sharp_probs = probs.clone()
            for _ in range(5):
                # Each iteration moves toward idempotent fixed point
                sharp_probs = bool_ring_mul(sharp_probs, sharp_probs)
                sharp_probs = sharp_probs / (sharp_probs.sum(dim=-1, keepdim=True) + 1e-10)
            probs = torch.where(violations.unsqueeze(-1), sharp_probs, probs)

        # Top-k selection
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        # Renormalize
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute auxiliary losses
        # Balance loss
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for t in range(self.top_k):
            for k in range(self.num_experts):
                expert_usage[k] += (topk_idx[:, t] == k).float().sum()
        expert_usage = expert_usage / (x.shape[0] * self.top_k)
        balance_loss = self.num_experts * (expert_usage * probs.mean(dim=0)).sum()

        # Huntington loss (idempotence of trait weights)
        w_xor_w = bool_ring_add(trait_w, trait_w)
        huntington_loss = (w_xor_w - trait_w).pow(2).mean()

        aux = {
            "entropy": entropy.mean(),
            "balance_loss": balance_loss,
            "huntington_loss": huntington_loss,
            "trait_weights": trait_w.detach(),
        }

        return topk_idx, topk_vals, aux
