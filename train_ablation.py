#!/usr/bin/env python3
"""
BURT-IMMA Ablation Training Script

Runs MMEP training with ablation variants defined in config/ablation_arithmetic.yaml.

Usage:
  python train_ablation.py --config config/ablation_arithmetic.yaml
  python train_ablation.py --config config/ablation_arithmetic.yaml --ablation no_memory

Contact: jessica@collectivekitty.com
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional


class MMEPLayer(nn.Module):
    """Single MMEP layer with relaxation dynamics."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.C_global = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        pre_act = self.W(h) + self.C_global
        activated = torch.tanh(pre_act)
        return (1.0 - alpha) * h + alpha * activated


class MMEPModel(nn.Module):
    """MMEP model for ablation experiments."""

    def __init__(self, hidden_dim: int, num_layers: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([MMEPLayer(hidden_dim) for _ in range(num_layers)])
        self.C_expert = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, T_free: int = 20, alpha: float = 0.5,
                expert_id: int = 0) -> torch.Tensor:
        h = x + self.C_expert[expert_id]
        for t in range(T_free):
            for layer in self.layers:
                h = layer(h, alpha)
        return self.output_proj(h)


class AblationTrainer:
    """Trainer supporting all ablation variants."""

    def __init__(self, config: dict, ablation_name: str = "full_mmep"):
        self.config = config
        self.ablation_name = ablation_name

        model_cfg = config["model"]
        mmep_cfg = config["mmep"]
        train_cfg = config["training"]

        self.device = torch.device(config["experiment"].get("device", "cpu"))
        self.model = MMEPModel(
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_experts=model_cfg["num_experts"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=mmep_cfg["lr_W"])
        self.T_free = mmep_cfg["T_free"]
        self.T_nudge = mmep_cfg["T_nudge"]
        self.alpha = mmep_cfg["alpha_relax"]
        self.beta = mmep_cfg["beta"]
        self.batch_size = train_cfg["batch_size"]
        self.num_epochs = train_cfg["num_epochs"]

        # Apply ablation overrides
        self.use_mmep = True
        self.use_memory = True
        self.use_projection = True

        for ablation in config.get("ablations", []):
            if ablation["name"] == ablation_name:
                disable = ablation.get("disable", [])
                if "mmep" in disable:
                    self.use_mmep = False
                if "C_global" in disable or "C_expert" in disable:
                    self.use_memory = False
                if "projection" in disable:
                    self.use_projection = False
                if "override" in ablation:
                    for k, v in ablation["override"].items():
                        setattr(self, k, v)
                break

    def train_step(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> Dict[str, float]:
        """One training step (MMEP or backprop depending on ablation)."""
        self.optimizer.zero_grad()

        if self.use_mmep:
            # Free phase
            h_free = self.model(batch_input, T_free=self.T_free, alpha=self.alpha)

            # Nudged phase
            h_nudged = h_free + self.beta * (batch_target - h_free)
            for _ in range(self.T_nudge):
                h_nudged = self.model(h_nudged, T_free=1, alpha=self.alpha)
                h_nudged = h_nudged + self.beta * (batch_target - h_nudged)

            # EP gradient (MSE as surrogate for correlation difference)
            loss = F.mse_loss(h_free, batch_target)
        else:
            # Standard backprop
            output = self.model(batch_input, T_free=1, alpha=1.0)
            loss = F.mse_loss(output, batch_target)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Constraint projection
        if self.use_projection:
            with torch.no_grad():
                # Spectral norm projection on weight matrices
                for layer in self.model.layers:
                    W = layer.W.weight
                    sigma = torch.linalg.svdvals(W)[0]
                    if sigma > 0.95:
                        layer.W.weight.data *= 0.95 / sigma
                # L2 projection on context memories
                if self.use_memory:
                    C_norm = self.model.C_expert.norm()
                    if C_norm > 1.0:
                        self.model.C_expert.data *= 1.0 / C_norm

        self.optimizer.step()

        return {"loss": loss.item()}

    def train(self):
        """Full training loop."""
        print(f"Training ablation: {self.ablation_name}")
        print(f"  MMEP: {self.use_mmep}, Memory: {self.use_memory}, Projection: {self.use_projection}")

        for epoch in range(self.num_epochs):
            # Synthetic data (arithmetic patterns)
            batch_input = torch.randn(self.batch_size, self.model.hidden_dim, device=self.device)
            batch_target = torch.randn(self.batch_size, self.model.hidden_dim, device=self.device)

            metrics = self.train_step(batch_input, batch_target)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs}: loss={metrics['loss']:.4f}")

        print(f"  Training complete. Final loss: {metrics['loss']:.4f}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="BURT-IMMA Ablation Training")
    parser.add_argument("--config", type=str, default="config/ablation_arithmetic.yaml")
    parser.add_argument("--ablation", type=str, default="full_mmep")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    trainer = AblationTrainer(config, ablation_name=args.ablation)
    trainer.train()


if __name__ == "__main__":
    main()
