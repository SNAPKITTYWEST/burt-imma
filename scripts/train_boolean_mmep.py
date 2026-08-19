#!/usr/bin/env python3
"""
Train Boolean actor network using MMEP (Min-Max Equilibrium Propagation).

License: BSL-1.1
Contact: jessica@collectivekitty.com

MMEP training loop:
  1. Free phase: run network forward to equilibrium (energy minimization)
  2. Nudge phase: clamp output toward target, re-equilibrate with nudge factor beta
  3. Local update: compute weight updates from difference in activations between phases
  4. Huntington regularization: penalize violations of Boolean algebra postulates

The Boolean perceptron actors enforce algebraic structure while MMEP provides
biologically plausible gradient-free learning.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class BooleanPerceptronActor(nn.Module):
    """Single Boolean perceptron actor with Huntington-compatible operations."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        # Boolean operation parameters (learnable)
        self.or_weight = nn.Parameter(torch.randn(output_dim, 2 * output_dim) * 0.01)
        self.and_weight = nn.Parameter(torch.randn(output_dim, 2 * output_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sigmoid activation (Boolean-compatible)."""
        return torch.sigmoid(F.linear(x, self.weight, self.bias))

    def boolean_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Approximate Boolean OR via learned gate."""
        combined = torch.cat([a, b], dim=-1)
        return torch.sigmoid(F.linear(combined, self.or_weight))

    def boolean_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Approximate Boolean AND via learned gate."""
        combined = torch.cat([a, b], dim=-1)
        return torch.sigmoid(F.linear(combined, self.and_weight))


class BooleanActorNetwork(nn.Module):
    """Network of Boolean perceptron actors trained with MMEP."""

    def __init__(self, layer_dims: list, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(BooleanPerceptronActor(layer_dims[i], layer_dims[i + 1]))
        self.layer_dims = layer_dims

    def free_phase(self, x: torch.Tensor, num_steps: int = 10) -> list:
        """Run free phase to equilibrium. Returns activations at each layer."""
        activations = [x]
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        # Iterate to equilibrium
        for _ in range(num_steps - 1):
            h = activations[0]
            for i, layer in enumerate(self.layers):
                h = layer(h)
                activations[i + 1] = h
        return activations

    def nudge_phase(self, x: torch.Tensor, target: torch.Tensor, num_steps: int = 10) -> list:
        """Run nudge phase with target clamping. Returns activations at each layer."""
        activations = [x]
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        # Nudge final layer toward target
        for _ in range(num_steps - 1):
            # Clamp output toward target
            activations[-1] = (1 - self.beta) * activations[-1] + self.beta * target
            h = activations[0]
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    activations[i + 1] = h
                else:
                    activations[i + 1] = (1 - self.beta) * h + self.beta * target
        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference."""
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def huntington_regularization(network: BooleanActorNetwork, x: torch.Tensor) -> torch.Tensor:
    """Compute Huntington postulate violation as regularization loss.

    Penalizes violations of:
      - Commutativity: OR(a,b) = OR(b,a), AND(a,b) = AND(b,a)
      - Idempotence: OR(a,a) = a, AND(a,a) = a
      - Complement: OR(a, 1-a) ~ 1, AND(a, 1-a) ~ 0
    """
    loss = torch.tensor(0.0, device=x.device)
    count = 0

    for layer in network.layers:
        # Generate test activations
        a = torch.sigmoid(torch.randn(x.shape[0], layer.output_dim, device=x.device))
        b = torch.sigmoid(torch.randn(x.shape[0], layer.output_dim, device=x.device))

        # Commutativity: OR(a,b) = OR(b,a)
        or_ab = layer.boolean_or(a, b)
        or_ba = layer.boolean_or(b, a)
        loss = loss + F.mse_loss(or_ab, or_ba)

        # Commutativity: AND(a,b) = AND(b,a)
        and_ab = layer.boolean_and(a, b)
        and_ba = layer.boolean_and(b, a)
        loss = loss + F.mse_loss(and_ab, and_ba)

        # Idempotence: OR(a,a) = a
        or_aa = layer.boolean_or(a, a)
        loss = loss + F.mse_loss(or_aa, a)

        # Idempotence: AND(a,a) = a
        and_aa = layer.boolean_and(a, a)
        loss = loss + F.mse_loss(and_aa, a)

        # Complement: OR(a, 1-a) ~ 1
        comp_a = 1.0 - a
        or_comp = layer.boolean_or(a, comp_a)
        loss = loss + F.mse_loss(or_comp, torch.ones_like(or_comp))

        # Complement: AND(a, 1-a) ~ 0
        and_comp = layer.boolean_and(a, comp_a)
        loss = loss + F.mse_loss(and_comp, torch.zeros_like(and_comp))

        count += 6

    return loss / max(count, 1)


def compute_mmep_update(
    free_activations: list,
    nudge_activations: list,
    network: BooleanActorNetwork,
    lr: float,
):
    """Compute and apply MMEP local weight updates.

    The update rule is:
      dW_i = (1/beta) * (s_i^nudge * s_{i-1}^nudge^T - s_i^free * s_{i-1}^free^T)
    """
    beta = network.beta
    for i, layer in enumerate(network.layers):
        # Pre-synaptic and post-synaptic activations
        pre_free = free_activations[i]
        post_free = free_activations[i + 1]
        pre_nudge = nudge_activations[i]
        post_nudge = nudge_activations[i + 1]

        # Compute correlation differences
        # dW = (1/beta) * (post_nudge @ pre_nudge^T - post_free @ pre_free^T) / batch_size
        batch_size = pre_free.shape[0]
        corr_nudge = torch.mm(post_nudge.t(), pre_nudge) / batch_size
        corr_free = torch.mm(post_free.t(), pre_free) / batch_size

        dW = (1.0 / beta) * (corr_nudge - corr_free)
        db = (1.0 / beta) * (post_nudge - post_free).mean(dim=0)

        # Apply updates
        with torch.no_grad():
            layer.weight.add_(lr * dW)
            layer.bias.add_(lr * db)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def generate_boolean_data(batch_size: int, input_dim: int, device: torch.device):
    """Generate synthetic Boolean data for training."""
    x = torch.randint(0, 2, (batch_size, input_dim), dtype=torch.float32, device=device)
    # Target: XOR of first half and second half (non-trivial Boolean function)
    half = input_dim // 2
    target_bits = ((x[:, :half] + x[:, half:2*half]) % 2)
    # Pad or truncate to match output dim
    return x, target_bits


def main():
    parser = argparse.ArgumentParser(
        description="Train Boolean actor network with MMEP (Min-Max Equilibrium Propagation)."
    )
    parser.add_argument("--config", type=str, default="config/boolean_actor.yaml",
                        help="Path to config YAML (default: config/boolean_actor.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (overrides config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract parameters with defaults
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    mmep_cfg = config.get("mmep", {})

    input_dim = model_cfg.get("input_dim", 256)
    hidden_dims = model_cfg.get("hidden_dims", [512, 256, 128])
    output_dim = model_cfg.get("output_dim", 128)
    layer_dims = [input_dim] + hidden_dims + [output_dim]

    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 256)
    lr = train_cfg.get("learning_rate", 0.01)
    huntington_weight = train_cfg.get("huntington_weight", 0.1)
    log_interval = train_cfg.get("log_interval", 10)

    beta = mmep_cfg.get("beta", 0.1)
    free_steps = mmep_cfg.get("free_steps", 10)
    nudge_steps = mmep_cfg.get("nudge_steps", 10)

    device_str = args.device or train_cfg.get("device", "cpu")
    device = torch.device(device_str)

    print("BURT-IMMA Boolean MMEP Training")
    print("=" * 60)
    print(f"  Config:           {args.config}")
    print(f"  Layer dims:       {layer_dims}")
    print(f"  Epochs:           {epochs}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Learning rate:    {lr}")
    print(f"  Beta (nudge):     {beta}")
    print(f"  Free steps:       {free_steps}")
    print(f"  Nudge steps:      {nudge_steps}")
    print(f"  Huntington wt:    {huntington_weight}")
    print(f"  Device:           {device}")
    print("=" * 60)
    print()

    # Create network
    network = BooleanActorNetwork(layer_dims, beta=beta).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Training loop
    metrics_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_huntington = 0.0
        epoch_entropy = 0.0
        num_batches = 0

        # Generate training data
        steps_per_epoch = max(1, 1000 // batch_size)
        for step in range(steps_per_epoch):
            x, target = generate_boolean_data(batch_size, input_dim, device)

            # Ensure target matches output dim
            if target.shape[1] != output_dim:
                # Tile or truncate
                if target.shape[1] < output_dim:
                    repeats = output_dim // target.shape[1] + 1
                    target = target.repeat(1, repeats)[:, :output_dim]
                else:
                    target = target[:, :output_dim]

            # === MMEP Training ===
            # Phase 1: Free phase
            free_acts = network.free_phase(x, num_steps=free_steps)

            # Phase 2: Nudge phase
            nudge_acts = network.nudge_phase(x, target, num_steps=nudge_steps)

            # Phase 3: Local update (MMEP rule)
            compute_mmep_update(free_acts, nudge_acts, network, lr)

            # Compute task loss for logging
            with torch.no_grad():
                output = network(x)
                task_loss = F.mse_loss(output, target)

            # Phase 4: Huntington regularization (uses gradient)
            optimizer.zero_grad()
            h_loss = huntington_regularization(network, x)
            reg_loss = huntington_weight * h_loss
            reg_loss.backward()
            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                output = network(x)
                predictions = (output > 0.5).float()
                accuracy = (predictions == target).float().mean().item()
                # Binary entropy of outputs
                p = torch.clamp(output, 1e-7, 1 - 1e-7)
                entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean().item()

            epoch_loss += task_loss.item()
            epoch_accuracy += accuracy
            epoch_huntington += h_loss.item()
            epoch_entropy += entropy
            num_batches += 1

        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_huntington = epoch_huntington / num_batches
        avg_entropy = epoch_entropy / num_batches

        metrics = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "huntington_violation": avg_huntington,
            "entropy": avg_entropy,
        }
        metrics_history.append(metrics)

        if epoch % log_interval == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"loss: {avg_loss:.6f} | "
                f"acc: {avg_accuracy:.4f} | "
                f"hunt_viol: {avg_huntington:.6f} | "
                f"entropy: {avg_entropy:.4f} | "
                f"time: {elapsed:.1f}s"
            )

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Total time:              {elapsed:.1f}s")
    print(f"  Final loss:              {metrics_history[-1]['loss']:.6f}")
    print(f"  Final accuracy:          {metrics_history[-1]['accuracy']:.4f}")
    print(f"  Final Huntington viol.:  {metrics_history[-1]['huntington_violation']:.6f}")
    print(f"  Final entropy:           {metrics_history[-1]['entropy']:.4f}")

    # Save model
    save_path = Path("checkpoints/boolean_mmep_latest.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": network.state_dict(),
        "config": config,
        "metrics": metrics_history,
        "layer_dims": layer_dims,
    }, save_path)
    print(f"  Model saved to:          {save_path}")


if __name__ == "__main__":
    main()
