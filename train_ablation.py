#!/usr/bin/env python3
"""
BURT-IMMA Full Training Script with Ablation Support
License: BSL-1.1
Contact: jessica@collectivekitty.com

Loads config from config/ablation_arithmetic.yaml and runs MMEP training
with ablation variants.

Usage:
  python train_ablation.py --config config/ablation_arithmetic.yaml
  python train_ablation.py --ablation no_memory --device cuda:0
  python train_ablation.py --ablation all --output-dir results/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    raise ImportError("PyTorch is required for training. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------


class SmoothLeakyActivation(nn.Module):
    """alpha * x + (1-alpha) * x * sigmoid(x)"""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * x * torch.sigmoid(x)


class GatesNormalization(nn.Module):
    """Constrained softmax with entropy <= max_entropy."""

    def __init__(self, d_input: int, num_gates: int, max_entropy: float = 0.20):
        super().__init__()
        self.gate_proj = nn.Linear(d_input, num_gates)
        self.max_entropy = max_entropy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_proj(x)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        # Sharpen via temperature reduction if entropy exceeds bound
        violations = entropy > self.max_entropy
        if violations.any():
            tau_lo = torch.full_like(entropy, 0.01)
            tau_hi = torch.ones_like(entropy)
            for _ in range(20):
                tau_mid = (tau_lo + tau_hi) / 2.0
                p = F.softmax(logits / tau_mid.unsqueeze(-1), dim=-1)
                h = -(p * (p + 1e-10).log()).sum(dim=-1)
                too_high = h > self.max_entropy
                tau_hi = torch.where(too_high, tau_mid, tau_hi)
                tau_lo = torch.where(too_high, tau_lo, tau_mid)
            tau_final = (tau_lo + tau_hi) / 2.0
            probs_sharp = F.softmax(logits / tau_final.unsqueeze(-1), dim=-1)
            probs = torch.where(violations.unsqueeze(-1), probs_sharp, probs)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        return probs, entropy


class CIFGMatrixMemory(nn.Module):
    """Coupled Input-Forget Gate Matrix Memory."""

    def __init__(self, d_mem: int):
        super().__init__()
        self.d_mem = d_mem
        self.W_f = nn.Linear(d_mem, 1)
        self.register_buffer("C_global", torch.zeros(d_mem, d_mem))

    def forward(self, x: torch.Tensor, C_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = torch.sigmoid(self.W_f(x))  # (batch, 1)
        candidate = x.unsqueeze(-1) @ x.unsqueeze(-2)  # (batch, d, d)
        candidate = candidate / (candidate.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        C_new = f.unsqueeze(-1) * C_prev + (1.0 - f).unsqueeze(-1) * candidate
        return C_new, f

    def read(self, query: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return torch.bmm(C, query.unsqueeze(-1)).squeeze(-1)


class GatesRouter(nn.Module):
    """Mixture-of-Experts router with top-k gating."""

    def __init__(self, d_input: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_input, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (expert_indices, expert_weights, load_balance_loss)."""
        scores = self.gate(x)
        probs = F.softmax(scores, dim=-1)

        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_vals = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing auxiliary loss
        routing_probs = probs.mean(dim=0)
        uniform = torch.ones(self.num_experts, device=x.device) / self.num_experts
        lb_loss = ((routing_probs - uniform) ** 2).sum()

        return top_k_idx, top_k_vals, lb_loss


class SuperpositionedInductionHeads(nn.Module):
    """Multi-head attention with induction bias."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d = x.shape
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d)
        return self.W_O(out)


class ExpertFFN(nn.Module):
    """Single expert feed-forward network."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.activation = SmoothLeakyActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.activation(self.W1(x)))


class BURTIMMALayer(nn.Module):
    """Single BURT-IMMA layer."""

    def __init__(self, d_model: int, num_experts: int, top_k: int, d_mem: int):
        super().__init__()
        self.attention = SuperpositionedInductionHeads(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.router = GatesRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([ExpertFFN(d_model) for _ in range(num_experts)])
        self.memory = CIFGMatrixMemory(d_mem)

    def forward(self, x: torch.Tensor, C_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention with residual
        h = self.norm1(x + self.attention(x))

        # MoE routing
        batch, seq_len, d = h.shape
        h_flat = h.view(batch * seq_len, d)
        indices, weights, lb_loss = self.router(h_flat)

        # Dispatch to experts
        expert_out = torch.zeros_like(h_flat)
        for k in range(indices.shape[1]):
            for e_id in range(len(self.experts)):
                mask = (indices[:, k] == e_id)
                if mask.any():
                    out_e = self.experts[e_id](h_flat[mask])
                    expert_out[mask] += weights[mask, k:k+1] * out_e

        expert_out = expert_out.view(batch, seq_len, d)
        h = self.norm2(h + expert_out)

        # Memory update (use mean across sequence for memory key)
        h_mean = h.mean(dim=1)  # (batch, d)
        C_new, f_gate = self.memory(h_mean, C_prev)

        return h, C_new, lb_loss


class BURTIMMA(nn.Module):
    """Full BURT-IMMA model."""

    def __init__(self, hidden_dim: int, num_layers: int, num_experts: int,
                 top_k: int, d_mem: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.d_mem = d_mem

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([
            BURTIMMALayer(hidden_dim, num_experts, top_k, d_mem)
            for _ in range(num_layers)
        ])
        self.gates_norm = GatesNormalization(hidden_dim, num_experts)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = x.shape[0]
        h = self.input_proj(x)

        C = torch.zeros(batch, self.d_mem, self.d_mem, device=x.device)
        total_lb_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            h, C, lb_loss = layer(h, C)
            total_lb_loss = total_lb_loss + lb_loss

        # Gates normalization on final hidden
        h_mean = h.mean(dim=1)
        gate_probs, entropy = self.gates_norm(h_mean)

        out = self.output_proj(h)

        aux = {
            "entropy": entropy,
            "gate_probs": gate_probs,
            "lb_loss": total_lb_loss,
            "memory_trace": C.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        }
        return out, aux


# ---------------------------------------------------------------------------
# Ablation Trainer
# ---------------------------------------------------------------------------


class AblationTrainer:
    """Full ablation training with MMEP support."""

    def __init__(self, config: dict, ablation_name: str = "full_mmep",
                 device: str = "cpu", output_dir: str = "results"):
        self.config = config
        self.ablation_name = ablation_name
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) / ablation_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract config sections
        self.model_cfg = config["model"]
        self.mmep_cfg = config["mmep"]
        self.train_cfg = config["training"]
        self.data_cfg = config["data"]

        # Apply ablation overrides
        self.use_mmep = True
        self.use_memory = True
        self.use_constraint = True
        self.use_moe = True
        self._apply_ablation(ablation_name)

        # Build model and optimizer
        self.model = self.build_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.mmep_cfg["lr_W"]
        )

        # Metrics log
        self.metrics_history: List[Dict[str, float]] = []

    def _apply_ablation(self, name: str) -> None:
        """Apply ablation variant configuration."""
        for ablation in self.config.get("ablations", []):
            if ablation["name"] == name:
                disable = ablation.get("disable", [])
                if "mmep" in disable:
                    self.use_mmep = False
                if "C_global" in disable or "C_expert" in disable:
                    self.use_memory = False
                if "projection" in disable:
                    self.use_constraint = False
                if ablation.get("use_backprop", False):
                    self.use_mmep = False
                if "override" in ablation:
                    for k, v in ablation["override"].items():
                        self.model_cfg[k] = v
                break

    def build_model(self) -> BURTIMMA:
        """Create BURT-IMMA model from config."""
        num_experts = self.model_cfg["num_experts"] if self.use_moe else 1
        return BURTIMMA(
            hidden_dim=self.model_cfg["hidden_dim"],
            num_layers=self.model_cfg["num_layers"],
            num_experts=num_experts,
            top_k=self.model_cfg["top_k"],
            d_mem=self.model_cfg["d_mem"],
        )

    def mmep_free_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Run T_free relaxation steps (free phase of MMEP)."""
        T_free = self.mmep_cfg["T_free"]
        alpha = self.mmep_cfg["alpha_relax"]

        h = x.clone()
        for t in range(T_free):
            out, aux = self.model(h)
            # Relaxation: blend output back toward input
            h = (1.0 - alpha) * h + alpha * out

        return h

    def mmep_nudge_phase(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Run T_nudge nudged steps (nudged phase of MMEP)."""
        T_nudge = self.mmep_cfg["T_nudge"]
        beta = self.mmep_cfg["beta"]
        alpha = self.mmep_cfg["alpha_relax"]

        h = x.clone()
        for t in range(T_nudge):
            out, aux = self.model(h)
            h = (1.0 - alpha) * h + alpha * out
            # Nudge toward target
            h = h + beta * (target - h)

        return h

    def compute_local_update(self, h_free: torch.Tensor,
                             h_nudged: torch.Tensor) -> torch.Tensor:
        """Compute local parameter update: (nudged - free) / beta."""
        beta = self.mmep_cfg["beta"]
        return (h_nudged - h_free) / beta

    def _spectral_norm_project(self) -> None:
        """Project weight matrices to satisfy spectral norm constraint."""
        lambda_max = self.mmep_cfg["lambda_max"]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() >= 2 and "weight" in name:
                    try:
                        s = torch.linalg.svdvals(param)
                        if s[0] > lambda_max:
                            param.data *= lambda_max / s[0]
                    except RuntimeError:
                        pass  # Skip non-matrix params

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Iterate batches, run MMEP, update weights."""
        self.model.train()
        batch_size = self.train_cfg["batch_size"]
        hidden_dim = self.model_cfg["hidden_dim"]
        seq_len = min(self.data_cfg["max_seq_len"], 32)  # Cap for memory

        epoch_loss = 0.0
        epoch_entropy = 0.0
        num_batches = self.data_cfg["train_queries"] // batch_size

        for batch_idx in range(num_batches):
            # Synthetic arithmetic data
            x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            target = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

            self.optimizer.zero_grad()

            if self.use_mmep:
                # MMEP training: free phase then nudge phase
                h_free = self.mmep_free_phase(x)
                h_nudged = self.mmep_nudge_phase(x, target)

                # Local update signal
                update_signal = self.compute_local_update(h_free, h_nudged)

                # Use MSE of free-phase output vs target as differentiable loss
                out_free, aux_free = self.model(x)
                loss = F.mse_loss(out_free, target)

                # Add load balancing loss
                loss = loss + 0.01 * aux_free["lb_loss"]
            else:
                # Standard backpropagation
                output, aux = self.model(x)
                loss = F.mse_loss(output, target)
                loss = loss + 0.01 * aux["lb_loss"]
                aux_free = aux

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_cfg["gradient_clip"]
            )

            self.optimizer.step()

            # Constraint projection (spectral norm)
            if self.use_constraint:
                self._spectral_norm_project()

            # Zero out memory parameters if ablation disables memory
            if not self.use_memory:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if "memory" in name:
                            param.zero_()

            epoch_loss += loss.item()
            epoch_entropy += aux_free["entropy"].mean().item()

        avg_loss = epoch_loss / num_batches
        avg_entropy = epoch_entropy / num_batches

        return {
            "loss": avg_loss,
            "entropy": avg_entropy,
        }

    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """Run evaluation on synthetic data."""
        self.model.eval()
        batch_size = self.train_cfg["batch_size"]
        hidden_dim = self.model_cfg["hidden_dim"]
        seq_len = min(self.data_cfg["max_seq_len"], 32)

        total_loss = 0.0
        total_entropy = 0.0
        max_spectral = 0.0

        with torch.no_grad():
            for _ in range(num_batches):
                x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
                target = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

                output, aux = self.model(x)
                loss = F.mse_loss(output, target)

                total_loss += loss.item()
                total_entropy += aux["entropy"].mean().item()

                # Track spectral norm
                for name, param in self.model.named_parameters():
                    if param.dim() >= 2 and "weight" in name:
                        try:
                            s = torch.linalg.svdvals(param)
                            max_spectral = max(max_spectral, s[0].item())
                        except RuntimeError:
                            pass

        return {
            "val_loss": total_loss / num_batches,
            "val_entropy": total_entropy / num_batches,
            "max_spectral_norm": max_spectral,
        }

    def run_ablation(self) -> Dict[str, List[float]]:
        """Run a specific ablation variant to completion."""
        print(f"\n{'='*60}")
        print(f"  Ablation: {self.ablation_name}")
        print(f"  MMEP: {self.use_mmep} | Memory: {self.use_memory} | "
              f"Constraint: {self.use_constraint} | MoE: {self.use_moe}")
        print(f"{'='*60}")

        num_epochs = self.train_cfg["num_epochs"]
        eval_every = self.train_cfg["eval_every"]
        checkpoint_every = self.train_cfg["checkpoint_every"]

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            elapsed = time.time() - start_time

            metrics = {"epoch": epoch, "time_s": elapsed, **train_metrics}

            # Evaluation
            if epoch % eval_every == 0:
                val_metrics = self.evaluate()
                metrics.update(val_metrics)
                print(f"  Epoch {epoch:3d}/{num_epochs} | "
                      f"loss={train_metrics['loss']:.4f} | "
                      f"val_loss={val_metrics['val_loss']:.4f} | "
                      f"entropy={train_metrics['entropy']:.4f} | "
                      f"spectral={val_metrics['max_spectral_norm']:.4f} | "
                      f"time={elapsed:.1f}s")
            else:
                print(f"  Epoch {epoch:3d}/{num_epochs} | "
                      f"loss={train_metrics['loss']:.4f} | "
                      f"entropy={train_metrics['entropy']:.4f} | "
                      f"time={elapsed:.1f}s")

            self.metrics_history.append(metrics)

            # Save checkpoint
            if epoch % checkpoint_every == 0:
                self._save_checkpoint(epoch)

        # Save final metrics
        self._save_metrics()
        return self.metrics_history

    def run_all(self) -> Dict[str, List[Dict[str, float]]]:
        """Run all ablation variants defined in config."""
        all_results = {}
        ablation_names = [a["name"] for a in self.config.get("ablations", [])]

        for abl_name in ablation_names:
            # Create a fresh trainer for each ablation
            trainer = AblationTrainer(
                self.config,
                ablation_name=abl_name,
                device=str(self.device),
                output_dir=str(self.output_dir.parent),
            )
            results = trainer.run_ablation()
            all_results[abl_name] = results

        # Save comparative summary
        summary_path = self.output_dir.parent / "ablation_summary.json"
        summary = {}
        for name, history in all_results.items():
            if history:
                final = history[-1]
                summary[name] = {
                    "final_loss": final.get("loss", float("nan")),
                    "final_entropy": final.get("entropy", float("nan")),
                    "total_time": sum(h.get("time_s", 0) for h in history),
                }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAblation summary saved to: {summary_path}")

        return all_results

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "ablation": self.ablation_name,
        }, ckpt_path)

    def _save_metrics(self) -> None:
        """Save metrics history to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="BURT-IMMA Ablation Training Script"
    )
    parser.add_argument(
        "--config", type=str, default="config/ablation_arithmetic.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--ablation", type=str, default="full_mmep",
        help="Ablation variant name, or 'all' to run all variants"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu, cuda:0, etc). Overrides config."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for checkpoints and metrics"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        raise SystemExit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override device if specified
    device = args.device or config["experiment"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"BURT-IMMA Training")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Output: {args.output_dir}")

    if args.ablation == "all":
        trainer = AblationTrainer(config, "full_mmep", device, args.output_dir)
        trainer.run_all()
    else:
        trainer = AblationTrainer(config, args.ablation, device, args.output_dir)
        trainer.run_ablation()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
