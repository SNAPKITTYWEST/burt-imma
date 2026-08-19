#!/usr/bin/env python3
"""
Verify Huntington postulates on Boolean perceptron actors.

License: BSL-1.1
Contact: jessica@collectivekitty.com

Huntington's postulates define a Boolean algebra (B, +, *, ', 0, 1):
  H1. Commutativity:  x + y = y + x,  x * y = y * x
  H2. Associativity:  x + (y + z) = (x + y) + z,  x * (y * z) = (x * y) * z
  H3. Distributivity: x * (y + z) = (x*y) + (x*z),  x + (y*z) = (x+y) * (x+z)
  H4. Identity:       x + 0 = x,  x * 1 = x
  H5. Complement:     x + x' = 1,  x * x' = 0
  H6. Idempotence:    x + x = x,  x * x = x

This script verifies these postulates hold (within tolerance) for
BooleanPerceptronActor networks operating over d-dimensional Boolean vectors.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BooleanPerceptronActor(nn.Module):
    """A perceptron actor that approximates Boolean algebra operations.

    The actor learns to implement Boolean OR (+) and AND (*) operations
    over d-dimensional vectors in [0, 1]^d, along with complement (').
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # OR gate approximation: trained to output x + y (Boolean OR)
        self.or_gate = nn.Sequential(
            nn.Linear(2 * d, 4 * d),
            nn.Sigmoid(),
            nn.Linear(4 * d, d),
            nn.Sigmoid(),
        )
        # AND gate approximation: trained to output x * y (Boolean AND)
        self.and_gate = nn.Sequential(
            nn.Linear(2 * d, 4 * d),
            nn.Sigmoid(),
            nn.Linear(4 * d, d),
            nn.Sigmoid(),
        )
        # Complement gate: trained to output x' (Boolean NOT)
        self.complement_gate = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.Sigmoid(),
            nn.Linear(2 * d, d),
            nn.Sigmoid(),
        )
        self._initialize_near_boolean()

    def _initialize_near_boolean(self):
        """Initialize weights to approximate Boolean operations."""
        with torch.no_grad():
            # Initialize OR gate close to max(x, y)
            for layer in self.or_gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
            # Initialize AND gate close to min(x, y)
            for layer in self.and_gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
            # Initialize complement gate close to 1 - x
            for layer in self.complement_gate:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    def boolean_or(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Boolean OR: x + y."""
        combined = torch.cat([x, y], dim=-1)
        return self.or_gate(combined)

    def boolean_and(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Boolean AND: x * y."""
        combined = torch.cat([x, y], dim=-1)
        return self.and_gate(combined)

    def boolean_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Boolean complement: x'."""
        return self.complement_gate(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor, op: str = "or") -> torch.Tensor:
        if op == "or":
            return self.boolean_or(x, y)
        elif op == "and":
            return self.boolean_and(x, y)
        else:
            raise ValueError(f"Unknown op: {op}")


@dataclass
class PostulateResult:
    """Result of verifying a single Huntington postulate."""
    name: str
    passed: bool
    max_violation: float
    mean_violation: float


def generate_boolean_vectors(num: int, d: int, device: torch.device) -> torch.Tensor:
    """Generate random Boolean-like vectors in {0, 1}^d."""
    return torch.randint(0, 2, (num, d), dtype=torch.float32, device=device)


def verify_commutativity(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    y: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H1: x + y = y + x and x * y = y * x."""
    with torch.no_grad():
        # OR commutativity
        or_xy = actor.boolean_or(x, y)
        or_yx = actor.boolean_or(y, x)
        or_diff = torch.abs(or_xy - or_yx)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

        # AND commutativity
        and_xy = actor.boolean_and(x, y)
        and_yx = actor.boolean_and(y, x)
        and_diff = torch.abs(and_xy - and_yx)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

    return (
        PostulateResult("H1a: OR commutativity (x+y=y+x)", or_max < tolerance, or_max, or_mean),
        PostulateResult("H1b: AND commutativity (x*y=y*x)", and_max < tolerance, and_max, and_mean),
    )


def verify_associativity(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H2: x + (y + z) = (x + y) + z and x * (y * z) = (x * y) * z."""
    with torch.no_grad():
        # OR associativity
        yz_or = actor.boolean_or(y, z)
        x_yz_or = actor.boolean_or(x, yz_or)
        xy_or = actor.boolean_or(x, y)
        xy_z_or = actor.boolean_or(xy_or, z)
        or_diff = torch.abs(x_yz_or - xy_z_or)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

        # AND associativity
        yz_and = actor.boolean_and(y, z)
        x_yz_and = actor.boolean_and(x, yz_and)
        xy_and = actor.boolean_and(x, y)
        xy_z_and = actor.boolean_and(xy_and, z)
        and_diff = torch.abs(x_yz_and - xy_z_and)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

    return (
        PostulateResult("H2a: OR associativity (x+(y+z)=(x+y)+z)", or_max < tolerance, or_max, or_mean),
        PostulateResult("H2b: AND associativity (x*(y*z)=(x*y)*z)", and_max < tolerance, and_max, and_mean),
    )


def verify_distributivity(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H3: x*(y+z) = (x*y)+(x*z) and x+(y*z) = (x+y)*(x+z)."""
    with torch.no_grad():
        # AND distributes over OR
        yz_or = actor.boolean_or(y, z)
        lhs_and = actor.boolean_and(x, yz_or)
        xy_and = actor.boolean_and(x, y)
        xz_and = actor.boolean_and(x, z)
        rhs_and = actor.boolean_or(xy_and, xz_and)
        and_diff = torch.abs(lhs_and - rhs_and)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

        # OR distributes over AND
        yz_and = actor.boolean_and(y, z)
        lhs_or = actor.boolean_or(x, yz_and)
        xy_or = actor.boolean_or(x, y)
        xz_or = actor.boolean_or(x, z)
        rhs_or = actor.boolean_and(xy_or, xz_or)
        or_diff = torch.abs(lhs_or - rhs_or)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

    return (
        PostulateResult("H3a: AND over OR (x*(y+z)=(x*y)+(x*z))", and_max < tolerance, and_max, and_mean),
        PostulateResult("H3b: OR over AND (x+(y*z)=(x+y)*(x+z))", or_max < tolerance, or_max, or_mean),
    )


def verify_identity(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H4: x + 0 = x and x * 1 = x."""
    d = x.shape[-1]
    device = x.device
    with torch.no_grad():
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        # OR identity: x + 0 = x
        x_or_0 = actor.boolean_or(x, zeros)
        or_diff = torch.abs(x_or_0 - x)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

        # AND identity: x * 1 = x
        x_and_1 = actor.boolean_and(x, ones)
        and_diff = torch.abs(x_and_1 - x)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

    return (
        PostulateResult("H4a: OR identity (x+0=x)", or_max < tolerance, or_max, or_mean),
        PostulateResult("H4b: AND identity (x*1=x)", and_max < tolerance, and_max, and_mean),
    )


def verify_complement(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H5: x + x' = 1 and x * x' = 0."""
    with torch.no_grad():
        x_comp = actor.boolean_complement(x)
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)

        # x + x' = 1
        x_or_comp = actor.boolean_or(x, x_comp)
        or_diff = torch.abs(x_or_comp - ones)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

        # x * x' = 0
        x_and_comp = actor.boolean_and(x, x_comp)
        and_diff = torch.abs(x_and_comp - zeros)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

    return (
        PostulateResult("H5a: OR complement (x+x'=1)", or_max < tolerance, or_max, or_mean),
        PostulateResult("H5b: AND complement (x*x'=0)", and_max < tolerance, and_max, and_mean),
    )


def verify_idempotence(
    actor: BooleanPerceptronActor,
    x: torch.Tensor,
    tolerance: float,
) -> Tuple[PostulateResult, PostulateResult]:
    """H6: x + x = x and x * x = x."""
    with torch.no_grad():
        # OR idempotence: x + x = x
        x_or_x = actor.boolean_or(x, x)
        or_diff = torch.abs(x_or_x - x)
        or_max = or_diff.max().item()
        or_mean = or_diff.mean().item()

        # AND idempotence: x * x = x
        x_and_x = actor.boolean_and(x, x)
        and_diff = torch.abs(x_and_x - x)
        and_max = and_diff.max().item()
        and_mean = and_diff.mean().item()

    return (
        PostulateResult("H6a: OR idempotence (x+x=x)", or_max < tolerance, or_max, or_mean),
        PostulateResult("H6b: AND idempotence (x*x=x)", and_max < tolerance, and_max, and_mean),
    )


def train_actor_to_boolean(actor: BooleanPerceptronActor, d: int, device: torch.device, steps: int = 2000):
    """Pre-train the actor to approximate Boolean operations."""
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for step in range(steps):
        x = generate_boolean_vectors(256, d, device)
        y = generate_boolean_vectors(256, d, device)

        # Target: element-wise Boolean operations
        target_or = torch.clamp(x + y, 0, 1)  # Boolean OR
        target_and = x * y  # Boolean AND
        target_complement = 1.0 - x  # Boolean NOT

        pred_or = actor.boolean_or(x, y)
        pred_and = actor.boolean_and(x, y)
        pred_comp = actor.boolean_complement(x)

        loss = (
            F.mse_loss(pred_or, target_or)
            + F.mse_loss(pred_and, target_and)
            + F.mse_loss(pred_comp, target_complement)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Training step {step+1}/{steps}, loss: {loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Huntington postulates on Boolean perceptron actors."
    )
    parser.add_argument("--num-actors", type=int, default=8, help="Number of actors to verify (default: 8)")
    parser.add_argument("--d", type=int, default=256, help="Dimension of Boolean vectors (default: 256)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for verification (default: 1e-6)")
    parser.add_argument("--num-tests", type=int, default=1000, help="Number of test vectors (default: 1000)")
    parser.add_argument("--train-steps", type=int, default=2000, help="Pre-training steps per actor (default: 2000)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Verifying Huntington postulates on {args.num_actors} actors")
    print(f"  d={args.d}, tolerance={args.tolerance}, num_tests={args.num_tests}")
    print(f"  device={device}")
    print()

    all_results = []

    for actor_idx in range(args.num_actors):
        print(f"--- Actor {actor_idx + 1}/{args.num_actors} ---")
        actor = BooleanPerceptronActor(args.d).to(device)

        print("  Pre-training to approximate Boolean operations...")
        train_actor_to_boolean(actor, args.d, device, steps=args.train_steps)
        actor.eval()

        # Generate test vectors
        x = generate_boolean_vectors(args.num_tests, args.d, device)
        y = generate_boolean_vectors(args.num_tests, args.d, device)
        z = generate_boolean_vectors(args.num_tests, args.d, device)

        # Verify all postulates
        results = []
        results.extend(verify_commutativity(actor, x, y, args.tolerance))
        results.extend(verify_associativity(actor, x, y, z, args.tolerance))
        results.extend(verify_distributivity(actor, x, y, z, args.tolerance))
        results.extend(verify_identity(actor, x, args.tolerance))
        results.extend(verify_complement(actor, x, args.tolerance))
        results.extend(verify_idempotence(actor, x, args.tolerance))

        print(f"  {'Postulate':<50} {'Status':<8} {'Max Viol.':<12} {'Mean Viol.':<12}")
        print(f"  {'-'*82}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.name:<50} {status:<8} {r.max_violation:<12.8f} {r.mean_violation:<12.8f}")

        all_results.append(results)
        print()

    # Summary
    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    total_checks = sum(len(r) for r in all_results)
    total_pass = sum(1 for results in all_results for r in results if r.passed)
    total_fail = total_checks - total_pass
    print(f"  Total checks: {total_checks}")
    print(f"  Passed:       {total_pass}")
    print(f"  Failed:       {total_fail}")
    print(f"  Pass rate:    {100.0 * total_pass / total_checks:.1f}%")

    if total_fail > 0:
        print("\n  Failed postulates (worst violations across actors):")
        postulate_names = [r.name for r in all_results[0]]
        for i, name in enumerate(postulate_names):
            worst = max(all_results[a][i].max_violation for a in range(args.num_actors))
            if any(not all_results[a][i].passed for a in range(args.num_actors)):
                print(f"    {name}: max_violation={worst:.8f}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
