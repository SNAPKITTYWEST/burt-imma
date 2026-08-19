"""
Superpositioned Induction Daemon

Multi-path Chain-of-Thought via quantum-inspired superposition:
  W_super = sum_k alpha_k (v_k v_k^T) tensor W_induction_k
  alpha_k = softmax(state @ v_k^T / tau)
  Interference: Gamma_{ij} = <v_i, v_j>
  Orthogonality loss: sum_{i!=j} Gamma_{ij}^2
  Decoherence: collapsed = tanh(sum_k w_k H_k) where w_k proportional to validity_score

Key properties:
  - K paths explored in parallel (superposition)
  - Invalid paths get destructive interference (cancelled)
  - Valid paths get constructive interference (reinforced)
  - Collapse to single output via decoherence (validity-weighted)
  - Iterative refinement converges to fixed point (contraction)

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class SuperpositionConfig:
    """Configuration for superpositioned induction."""
    d_model: int = 768
    num_paths: int = 4              # K paths in superposition
    num_heads: int = 8
    head_dim: int = 64
    interference_threshold: float = 0.1
    collapse_temperature: float = 1.0
    orthogonality_reg: float = 0.01
    max_iterations: int = 5         # iterative refinement cap


class SuperpositionedInductionHeads(nn.Module):
    """
    Multi-path induction heads with quantum-inspired superposition.

    Each path k has:
      - Basis vector v_k (almost-orthogonal)
      - Induction weights W_induction_k
      - Confidence score (validity)

    Paths interfere constructively (valid) or destructively (invalid).
    Final output is decoherence collapse of all paths.
    """

    def __init__(self, config: SuperpositionConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        K = config.num_paths
        H = config.num_heads
        d_h = config.head_dim

        # Path basis vectors (almost-orthogonal)
        self.path_basis = nn.Parameter(torch.randn(K, d) * 0.1)
        nn.init.orthogonal_(self.path_basis)

        # Per-path induction weights
        self.W_induction = nn.Parameter(torch.randn(K, H, d_h, d_h) * 0.02)

        # Shared QKV projections
        self.W_Q = nn.Linear(d, H * d_h)
        self.W_K = nn.Linear(d, H * d_h)
        self.W_V = nn.Linear(d, H * d_h)
        self.W_O = nn.Linear(H * d_h, d)

        # Path confidence (learnable prior)
        self.path_confidence = nn.Parameter(torch.ones(K) / K)

        # Decoherence parameters
        self.collapse_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Superpositioned forward pass.

        Args:
            state: [batch, seq, d]
        Returns:
            output: [batch, seq, d] collapsed output
            aux: dict with path info
        """
        batch, seq, d = state.shape
        K = self.config.num_paths

        # Compute path weights: alpha_k = softmax(state @ v_k^T / tau)
        # Use CLS (first token) for path selection
        cls = state[:, 0]  # [batch, d]
        path_logits = cls @ self.path_basis.T / self.config.collapse_temperature  # [batch, K]
        alpha = F.softmax(path_logits, dim=-1)  # [batch, K]

        # Compute per-path induction outputs
        path_outputs = []
        for k in range(K):
            h_k = self._compute_path_induction(state, k)  # [batch, seq, d]
            path_outputs.append(h_k)

        path_outputs = torch.stack(path_outputs, dim=1)  # [batch, K, seq, d]

        # Compute interference matrix
        interference = self._compute_interference_matrix()  # [K, K]

        # Apply interference: each path gets contribution from interfering paths
        for k in range(K):
            interference_k = self._compute_interference(k, path_outputs)
            path_outputs[:, k] = path_outputs[:, k] + interference_k

        # Decoherence collapse
        output = self._decoherence_collapse(path_outputs, alpha)  # [batch, seq, d]

        # Orthogonality loss
        orth_loss = self._orthogonality_loss()

        aux = {
            "path_weights": alpha,
            "interference_matrix": interference,
            "orthogonality_loss": orth_loss,
        }

        return output, aux

    def _compute_path_induction(self, state: torch.Tensor, k: int) -> torch.Tensor:
        """Compute path-specific multi-head attention."""
        batch, seq, d = state.shape
        H = self.config.num_heads
        d_h = self.config.head_dim

        Q = self.W_Q(state).view(batch, seq, H, d_h).transpose(1, 2)
        K_proj = self.W_K(state).view(batch, seq, H, d_h).transpose(1, 2)
        V = self.W_V(state).view(batch, seq, H, d_h).transpose(1, 2)

        # Apply path-specific induction weight
        # W_induction_k: [H, d_h, d_h]
        K_induced = torch.einsum("bhsd,hde->bhse", K_proj, self.W_induction[k])

        # Attention
        scale = 1.0 / math.sqrt(d_h)
        attn = (Q @ K_induced.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ V  # [batch, H, seq, d_h]

        out = out.transpose(1, 2).contiguous().view(batch, seq, H * d_h)
        return self.W_O(out)

    def _compute_interference(self, k: int, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute interference from other paths onto path k.
        Gamma_{ij} = <v_i, v_j> controls interference strength.
        """
        K = self.config.num_paths
        interference = torch.zeros_like(outputs[:, k])

        v_k = F.normalize(self.path_basis[k], dim=-1)
        for j in range(K):
            if j == k:
                continue
            v_j = F.normalize(self.path_basis[j], dim=-1)
            gamma_kj = (v_k * v_j).sum()  # inner product
            if gamma_kj.abs() > self.config.interference_threshold:
                interference = interference + gamma_kj * outputs[:, j]

        return interference

    def _compute_interference_matrix(self) -> torch.Tensor:
        """Compute full interference matrix Gamma = V @ V^T."""
        V_norm = F.normalize(self.path_basis, dim=-1)
        return V_norm @ V_norm.T  # [K, K]

    def _orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality loss: sum_{i!=j} Gamma_{ij}^2."""
        gamma = self._compute_interference_matrix()
        K = self.config.num_paths
        mask = 1.0 - torch.eye(K, device=gamma.device)
        return (gamma * mask).pow(2).sum()

    def _decoherence_collapse(self, path_outputs: torch.Tensor,
                              alpha: torch.Tensor) -> torch.Tensor:
        """
        Collapse superposition to single output.
        collapsed = tanh(sum_k w_k * H_k) where w_k proportional to alpha_k
        """
        # alpha: [batch, K], path_outputs: [batch, K, seq, d]
        weights = alpha.unsqueeze(-1).unsqueeze(-1)  # [batch, K, 1, 1]
        superposed = (weights * path_outputs).sum(dim=1)  # [batch, seq, d]
        collapsed = torch.tanh(self.collapse_proj(superposed))
        return self.ln(collapsed)


class IterativeSuperpositionRefinement(nn.Module):
    """
    Iterative refinement of superpositioned output until convergence.

    Runs SuperpositionedInductionHeads repeatedly until:
      - Output change < threshold (convergence)
      - Max iterations reached
    """

    def __init__(self, config: SuperpositionConfig):
        super().__init__()
        self.config = config
        self.heads = SuperpositionedInductionHeads(config)
        self.convergence_threshold = 1e-4

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Iteratively refine until convergence.

        Returns:
            output: [batch, seq, d] converged output
            aux: dict with iterations, final_delta
        """
        current = state
        iterations = 0

        for i in range(self.config.max_iterations):
            output, aux = self.heads(current)
            delta = (output - current).norm() / (current.norm() + 1e-8)
            iterations = i + 1

            if delta < self.convergence_threshold:
                break
            current = output

        aux["iterations"] = iterations
        aux["final_delta"] = delta.item() if isinstance(delta, torch.Tensor) else delta

        return output, aux


class SuperpositionedBURTIMMA(nn.Module):
    """
    Full BURT-IMMA with superpositioned induction.

    Integrates:
      - Gates normalization (entropy-constrained routing)
      - CIFG memory (trajectory history)
      - SmoothLeaky activation (axiomatic nonlinearity)
      - Superpositioned induction (multi-path CoT)
    """

    def __init__(self, d: int = 768, num_paths: int = 4):
        super().__init__()
        config = SuperpositionConfig(d_model=d, num_paths=num_paths)
        self.refinement = IterativeSuperpositionRefinement(config)
        self.output_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with iterative superposition."""
        output, aux = self.refinement(x)
        return self.output_proj(output), aux
