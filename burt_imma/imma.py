"""
IMMA: Instruct-MoE Matrix-Memory Architecture

Generation phase of BURT-IMMA unified architecture.

Architecture:
  - IMMAExpert: Single expert with CIFG matrix memory
  - IMMALayer: Full MoE layer with top-k routing
  - Complexity: T=1 O(L*d^2) time, O(L*K*d^2) memory

Key equations:
  r = W_r @ h_prev                              [recurrence projection]
  gates = LayerNorm(W_g @ [x, h_prev, r])       [5-way gating]
  f, i, o, v, k = chunk(gates, 5)               [split gates]
  i = 1 - f                                     [CIFG constraint]
  C = f * C_prev + i * (v @ k^T)                [outer-product memory]
  h = o * LayerNorm(C @ W_h^T)                  [readout]

Training protocol:
  Phase 1: Dense pretrain (no MoE routing)
  Phase 2: Expert splitting (k-means on W_g rows)
  Phase 3: Instruct-MoE fine-tune (L_instruct + lambda_ent*L_router + lambda_bal*L_balance)

Falsification criteria:
  - T=1 latency <= 1.15x dense
  - H(alpha) <= 0.20 constraint
  - Expert splitting preserves zero-shot acc >= 0.9x

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IMMAExpert(nn.Module):
    """
    Single IMMA expert with CIFG matrix memory.

    Implements coupled input-forget gate (CIFG) with outer-product
    memory matrix. The expert maintains a d x d memory matrix C
    that accumulates key-value associations.

    Args:
        d: hidden dimension
        d_r: recurrence projection dimension
    """

    def __init__(self, d: int, d_r: int):
        super().__init__()
        self.d = d
        self.d_r = d_r

        # Gate projection: [x, h_prev, r] -> 5*d (f, i, o, v, k)
        self.W_g = nn.Linear(2 * d + d_r, 5 * d, bias=False)
        # Recurrence projection
        self.W_r = nn.Linear(d, d_r, bias=False)
        # Readout projection
        self.W_h = nn.Linear(d, d, bias=False)
        # Layer norm for gate stabilization
        self.ln = nn.LayerNorm(5 * d)
        # Layer norm for readout
        self.ln_read = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        C_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single expert forward pass.

        Args:
            x: [batch, d] input
            h_prev: [batch, d] previous hidden state
            C_prev: [batch, d, d] previous memory matrix

        Returns:
            C: [batch, d, d] updated memory matrix
            h: [batch, d] new hidden state
        """
        # Recurrence projection
        r = self.W_r(h_prev)  # [batch, d_r]

        # Concatenate inputs for gating
        gate_input = torch.cat([x, h_prev, r], dim=-1)  # [batch, 2*d + d_r]

        # Compute all gates in one projection
        gates = self.ln(self.W_g(gate_input))  # [batch, 5*d]

        # Split into 5 gates
        f, i, o, v, k = gates.chunk(5, dim=-1)  # each [batch, d]

        # Sigmoid gates
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        i = 1.0 - f  # CIFG constraint: i = 1 - f

        # Memory update: C = f * C_prev + i * (v @ k^T)
        # v: [batch, d, 1], k: [batch, 1, d]
        outer = v.unsqueeze(-1) @ k.unsqueeze(-2)  # [batch, d, d]
        C = f.unsqueeze(-1) * C_prev + i.unsqueeze(-1) * outer

        # Readout: h = o * LayerNorm(C @ W_h^T)
        # C @ W_h^T: [batch, d, d] @ [d, d] -> [batch, d, d]
        # We take the diagonal or reduce - here we do C @ W_h.weight.T and take mean
        Ch = torch.bmm(C, self.W_h.weight.T.unsqueeze(0).expand(x.shape[0], -1, -1))
        # Reduce to [batch, d] via diagonal
        h = o * self.ln_read(Ch.diagonal(dim1=-2, dim2=-1))

        return C, h


class IMMALayer(nn.Module):
    """
    Full IMMA layer with Mixture-of-Experts routing.

    Routes tokens to top-k experts with entropy-constrained gating.
    Each expert maintains its own memory matrix.

    Args:
        d: hidden dimension
        d_r: recurrence dimension
        num_experts: number of experts (K)
        top_k: number of experts per token
        entropy_bound: max router entropy (default 0.20)
    """

    def __init__(self, d: int, d_r: int, num_experts: int = 4,
                 top_k: int = 1, entropy_bound: float = 0.20):
        super().__init__()
        self.d = d
        self.num_experts = num_experts
        self.top_k = top_k
        self.entropy_bound = entropy_bound

        # Expert modules
        self.experts = nn.ModuleList([
            IMMAExpert(d, d_r) for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(d, num_experts)

        # Balance loss coefficient
        self.balance_coeff = 0.01

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        C_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Full MoE layer forward.

        Args:
            x: [batch, d] input
            h_prev: [batch, d] previous hidden
            C_prev: [batch, num_experts, d, d] per-expert memories

        Returns:
            C: [batch, num_experts, d, d] updated memories
            h: [batch, d] combined hidden state
            aux: dict with router_entropy, balance_loss, assignments
        """
        batch = x.shape[0]

        # Router logits
        logits = self.router(x)  # [batch, num_experts]

        # Entropy-constrained softmax
        from .burt import constrained_softmax
        alpha = constrained_softmax(logits, self.entropy_bound)  # [batch, num_experts]

        # Top-k selection
        topk_vals, topk_idx = alpha.topk(self.top_k, dim=-1)  # [batch, top_k]
        # Renormalize top-k weights
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Run selected experts
        C_new = C_prev.clone()
        h_combined = torch.zeros(batch, self.d, device=x.device)

        for t in range(self.top_k):
            expert_idx = topk_idx[:, t]  # [batch]
            weight = topk_vals[:, t]     # [batch]

            for k in range(self.num_experts):
                mask = expert_idx == k
                if not mask.any():
                    continue

                x_k = x[mask]
                h_k = h_prev[mask]
                C_k = C_prev[mask, k]

                C_out, h_out = self.experts[k](x_k, h_k, C_k)

                C_new[mask, k] = C_out
                h_combined[mask] += weight[mask].unsqueeze(-1) * h_out

        # Compute auxiliary losses
        router_entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=-1).mean()

        # Balance loss: encourage uniform expert usage
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for t in range(self.top_k):
            for k in range(self.num_experts):
                expert_usage[k] += (topk_idx[:, t] == k).float().sum()
        expert_usage = expert_usage / (batch * self.top_k)
        balance_loss = self.num_experts * (expert_usage * alpha.mean(dim=0)).sum()

        aux = {
            "router_entropy": router_entropy,
            "balance_loss": balance_loss * self.balance_coeff,
            "assignments": topk_idx,
            "gate_values": topk_vals,
        }

        return C_new, h_combined, aux
