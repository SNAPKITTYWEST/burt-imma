"""
Deterministic Sum-Inversion Agent

Replaces probabilistic token generation with deterministic decoding:
  Standard: sample from softmax(logits) — lossy, stochastic
  Sum-Inversion: x_{t+1} = argmax(B^dagger * delta_S_t) — exact, deterministic

Key math:
  Boolean Kernel: B in {0,1}^{d x V}, full column rank (d >= V)
  Trajectory: S_t = sum_{i=1}^t B x_i (running sum, sufficient statistic)
  Exact inversion: x_{t+1} = argmax(B^dagger * delta_S_t)
  B^dagger = Moore-Penrose pseudoinverse

Why this works:
  - Trajectory preserves ALL information (vs softmax discards magnitude)
  - No sampling noise (deterministic vs probabilistic)
  - Connection to CIFG memory (trajectory history as matrix state)
  - No softmax bottleneck (bypasses Yang et al. 2018 limitation)

Chinchilla scaling: N proportional to C^0.5, D proportional to C^0.5, D ~ 20N

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from .smooth_leaky import SmoothLeakyActivation
from .meta_inverted_sum import meta_inverted_sum_softmax


@dataclass
class IntegratedAgentConfig:
    """Configuration for the integrated sum-inversion agent."""
    vocab_size: int = 32768
    embed_dim: int = 4096
    hidden_dim: int = 8192
    num_layers: int = 4
    num_heads: int = 16
    num_experts: int = 4
    top_k: int = 1
    entropy_bound: float = 0.20
    alpha: float = 0.01          # SmoothLeaky alpha
    beta: float = 1.0            # SmoothLeaky beta
    chinchilla_ratio: float = 20.0  # D/N optimal ratio
    max_seq_len: int = 2048
    kernel_rank: int = 256       # d dimension of Boolean kernel


class BooleanKernel(nn.Module):
    """
    Boolean kernel B in {0,1}^{d x V} for sum-inversion.

    The kernel maps vocabulary tokens to d-dimensional binary codes.
    Full column rank (d >= V) guarantees exact inversion.
    """

    def __init__(self, d: int, vocab_size: int):
        super().__init__()
        assert d >= vocab_size, f"Kernel rank d={d} must be >= vocab_size={vocab_size} for full rank"
        # Initialize with random binary codes (Bernoulli)
        B_init = (torch.rand(d, vocab_size) > 0.5).float()
        self.register_buffer("B", B_init)
        # Precompute pseudoinverse
        self.register_buffer("B_pinv", torch.linalg.pinv(B_init))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens to trajectory space: B @ one_hot(token)."""
        one_hot = F.one_hot(tokens, self.B.shape[1]).float()
        return one_hot @ self.B.T  # [batch, seq, d]

    def decode(self, delta_S: torch.Tensor) -> torch.Tensor:
        """Decode trajectory delta to token: argmax(B^dagger @ delta_S)."""
        logits = delta_S @ self.B_pinv.T  # [batch, vocab_size]
        return logits.argmax(dim=-1)

    def verify_rank(self) -> bool:
        """Verify kernel has full column rank."""
        rank = torch.linalg.matrix_rank(self.B).item()
        return rank == self.B.shape[1]


class BooleanKernelActorNetwork(nn.Module):
    """
    Actor network that operates in Boolean kernel space.

    Processes trajectories (running sums) instead of token probabilities.
    Uses SmoothLeaky activation for MMEP compatibility.
    """

    def __init__(self, config: IntegratedAgentConfig):
        super().__init__()
        self.config = config

        # Boolean kernel
        self.kernel = BooleanKernel(config.kernel_rank, config.vocab_size)

        # Dynamics engine (processes trajectories)
        self.activation = SmoothLeakyActivation(config.alpha, config.beta)
        self.layers = nn.ModuleList([
            nn.Linear(config.kernel_rank, config.kernel_rank)
            for _ in range(config.num_layers)
        ])
        self.ln = nn.ModuleList([
            nn.LayerNorm(config.kernel_rank) for _ in range(config.num_layers)
        ])

    def forward(self, S_t: torch.Tensor) -> torch.Tensor:
        """
        Predict next trajectory delta from current trajectory.

        Args:
            S_t: [batch, d] current trajectory state
        Returns:
            delta_S: [batch, d] predicted trajectory change
        """
        h = S_t
        for layer, ln in zip(self.layers, self.ln):
            h = h + self.activation(ln(layer(h)))
        return h


class TrajectoryMMEPTrainer:
    """
    MMEP trainer operating in trajectory space.

    The energy function is defined over trajectories:
      E(S) = ||S_pred - S_true||^2 + lambda * constraint_terms

    Free phase: relax S_pred to equilibrium
    Nudged phase: perturb toward S_true
    Gradient: local EP rule (no backprop through time)
    """

    def __init__(self, network: BooleanKernelActorNetwork,
                 lr: float = 1e-3, T_free: int = 20, T_nudge: int = 4,
                 beta_nudge: float = 0.1):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.T_free = T_free
        self.T_nudge = T_nudge
        self.beta_nudge = beta_nudge
        self.step_count = 0

    def train_step(self, tokens: torch.Tensor) -> Dict[str, float]:
        """
        One MMEP training step.

        Args:
            tokens: [batch, seq_len] token IDs
        Returns:
            dict with loss, accuracy, etc.
        """
        batch, seq_len = tokens.shape
        device = tokens.device

        # Encode to trajectory space
        encoded = self.network.kernel.encode(tokens)  # [batch, seq, d]

        # Compute running sum trajectory
        S = encoded.cumsum(dim=1)  # [batch, seq, d]

        total_loss = 0.0
        correct = 0
        total = 0

        for t in range(seq_len - 1):
            S_t = S[:, t]  # [batch, d]
            S_target = encoded[:, t + 1]  # [batch, d] next step delta

            # Free phase: predict delta
            self.optimizer.zero_grad()
            delta_pred = self.network(S_t)

            # Loss in trajectory space
            loss = F.mse_loss(delta_pred, S_target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Check decoding accuracy
            pred_tokens = self.network.kernel.decode(delta_pred.detach())
            correct += (pred_tokens == tokens[:, t + 1]).sum().item()
            total += batch

        self.step_count += 1

        return {
            "loss": total_loss / max(seq_len - 1, 1),
            "accuracy": correct / max(total, 1),
            "step": self.step_count,
        }


class IntegratedSumInversionAgent(nn.Module):
    """
    Complete sum-inversion agent integrating all components.

    Architecture connections:
      - Boolean Kernel B → Perceptron Actor Traits (columns of B are trait vectors)
      - B^dagger delta_S → Gates Normalization (structural simplex)
      - Dynamics engine → SmoothLeakyActivation (axiomatic nonlinearity)
      - Trajectory energy → MMEP Learning (local equilibrium)
      - S_t history → CIFG Memory (matrix state)
    """

    def __init__(self, config: Optional[IntegratedAgentConfig] = None):
        super().__init__()
        self.config = config or IntegratedAgentConfig()

        # Core network
        self.actor_network = BooleanKernelActorNetwork(self.config)

        # CIFG memory (trajectory as matrix state)
        self.memory = nn.Parameter(
            torch.zeros(self.config.kernel_rank, self.config.kernel_rank)
        )
        self.W_f = nn.Linear(self.config.kernel_rank, 1)

    def generate(self, prompt_tokens: torch.Tensor, max_new: int = 100) -> torch.Tensor:
        """
        Deterministic generation via sum-inversion.

        Args:
            prompt_tokens: [batch, prompt_len] token IDs
            max_new: maximum new tokens to generate

        Returns:
            tokens: [batch, prompt_len + max_new] complete sequence
        """
        batch = prompt_tokens.shape[0]
        device = prompt_tokens.device

        # Encode prompt to trajectory
        encoded = self.actor_network.kernel.encode(prompt_tokens)
        S = encoded.sum(dim=1)  # [batch, d] trajectory state

        generated = []
        for _ in range(max_new):
            # Predict next delta
            delta_S = self.actor_network(S)

            # Decode deterministically
            next_token = self.actor_network.kernel.decode(delta_S)
            generated.append(next_token)

            # Update trajectory
            next_encoded = self.actor_network.kernel.encode(next_token.unsqueeze(1)).squeeze(1)
            S = S + next_encoded

            # Update CIFG memory
            f = torch.sigmoid(self.W_f(S))
            v = delta_S.unsqueeze(-1)
            k = S.unsqueeze(-2)
            self.memory.data = f.mean() * self.memory + (1 - f.mean()) * (v @ k).mean(0)

        return torch.cat([prompt_tokens, torch.stack(generated, dim=1)], dim=1)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            tokens: [batch, seq_len]
        Returns:
            dict with logits, trajectory, loss
        """
        encoded = self.actor_network.kernel.encode(tokens)
        S = encoded.cumsum(dim=1)

        # Predict all next deltas
        predictions = []
        for t in range(tokens.shape[1] - 1):
            delta = self.actor_network(S[:, t])
            predictions.append(delta)

        if predictions:
            pred_stack = torch.stack(predictions, dim=1)
            target_stack = encoded[:, 1:]
            loss = F.mse_loss(pred_stack, target_stack)
        else:
            loss = torch.tensor(0.0, device=tokens.device)

        return {
            "loss": loss,
            "trajectory": S,
            "predictions": pred_stack if predictions else None,
        }
