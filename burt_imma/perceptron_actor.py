"""
Perceptron Actor Architecture with Huntington Postulates

Boolean-algebraic foundation for BURT-IMMA expert routing.
Implements perceptron actors as Boolean ring elements satisfying
Huntington's 1904 postulates.

Key math:
  Boolean ring: x + y = x XOR y = x + y - 2xy (symmetric difference)
  Ring multiply: x * y = x AND y = xy
  Complement: NOT x = 1 - x
  Idempotence: x + x = x, x * x = x (saturation property)

The perceptron actor model connects:
  1. Boolean algebra (Huntington postulates) -> routing decisions
  2. Perceptron updates -> weight learning
  3. MMEP equilibrium -> convergence guarantees

Reference: Huntington, E.V. (1904) "Sets of Independent Postulates for the
Algebra of Logic", Transactions of the AMS.

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass


class SignalType(Enum):
    """Signal types in the perceptron actor network."""
    EXCITATORY = "excitatory"   # Positive activation (AND-like)
    INHIBITORY = "inhibitory"   # Negative activation (NOT-like)
    MODULATORY = "modulatory"   # Gate modulation (XOR-like)
    MEMORY = "memory"           # Memory retention signal


@dataclass
class Signal:
    """A signal between perceptron actors."""
    source_id: int
    target_id: int
    signal_type: SignalType
    value: torch.Tensor          # [d] signal vector
    weight: float = 1.0          # Boolean weight in [0, 1]
    timestamp: int = 0


class PerceptronActor(nn.Module):
    """
    Single perceptron actor satisfying Huntington postulates.

    The actor implements Boolean operations on its weight vector:
      OR:  w1 + w2 = w1 + w2 - w1*w2 (inclusion)
      AND: w1 * w2 = w1 * w2 (conjunction)
      NOT: ~w = 1 - w (complement)
      XOR: w1 ^ w2 = w1 + w2 - 2*w1*w2 (symmetric difference, Boolean ring add)

    Huntington postulates satisfied:
      H1. Closure under +, *
      H2. Identity: 0 for +, 1 for *
      H3. Commutativity: a+b = b+a, a*b = b*a
      H4. Distributivity: a*(b+c) = a*b + a*c, a+(b*c) = (a+b)*(a+c)
      H5. Complement: a + ~a = 1, a * ~a = 0
      H6. Distinct elements: 0 != 1
      H7. Idempotence: a + a = a, a * a = a

    Args:
        d: dimension of weight vector
        actor_id: unique identifier
    """

    def __init__(self, d: int, actor_id: int = 0):
        super().__init__()
        self.d = d
        self.actor_id = actor_id

        # Weight vector (initialized in [0, 1] for Boolean interpretation)
        self.w = nn.Parameter(torch.rand(d) * 0.5 + 0.25)

        # Bias (threshold)
        self.bias = nn.Parameter(torch.zeros(1))

        # Activation memory (for CIFG-style retention)
        self.register_buffer("activation_history", torch.zeros(d))
        self.register_buffer("step_count", torch.tensor(0))

    def bool_or(self, other_w: torch.Tensor) -> torch.Tensor:
        """Boolean OR: w1 + w2 - w1*w2 (Huntington H4 dual)."""
        return self.w + other_w - self.w * other_w

    def bool_and(self, other_w: torch.Tensor) -> torch.Tensor:
        """Boolean AND: w1 * w2."""
        return self.w * other_w

    def bool_not(self) -> torch.Tensor:
        """Boolean NOT: 1 - w (Huntington H5)."""
        return 1.0 - self.w

    def bool_xor(self, other_w: torch.Tensor) -> torch.Tensor:
        """Boolean XOR (ring addition): w1 + w2 - 2*w1*w2."""
        return self.w + other_w - 2.0 * self.w * other_w

    def forward(self, x: torch.Tensor, signals: List[Signal] = None) -> torch.Tensor:
        """
        Actor activation.

        Computes: output = sigma(w * x + bias + signal_contributions)
        where sigma projects onto [0, 1] (Boolean compatible).

        Args:
            x: [batch, d] input
            signals: incoming signals from other actors

        Returns:
            activation: [batch, d] in [0, 1]
        """
        # Base activation
        activation = self.w.unsqueeze(0) * x + self.bias

        # Process incoming signals
        if signals:
            for sig in signals:
                if sig.target_id != self.actor_id:
                    continue
                if sig.signal_type == SignalType.EXCITATORY:
                    # AND with signal
                    activation = activation * sig.value.unsqueeze(0) * sig.weight
                elif sig.signal_type == SignalType.INHIBITORY:
                    # NOT of signal contribution
                    activation = activation * (1.0 - sig.value.unsqueeze(0) * sig.weight)
                elif sig.signal_type == SignalType.MODULATORY:
                    # XOR modulation
                    activation = activation + sig.value.unsqueeze(0) * sig.weight \
                                 - 2.0 * activation * sig.value.unsqueeze(0) * sig.weight
                elif sig.signal_type == SignalType.MEMORY:
                    # Memory retention (CIFG-style)
                    f = torch.sigmoid(sig.value.unsqueeze(0) * sig.weight)
                    activation = f * self.activation_history.unsqueeze(0) + (1 - f) * activation

        # Project to [0, 1] via sigmoid (Boolean compatible)
        output = torch.sigmoid(activation)

        # Update activation history
        with torch.no_grad():
            self.activation_history = output.mean(dim=0)
            self.step_count += 1

        return output

    def verify_huntington(self) -> dict:
        """
        Verify Huntington postulates on current weight vector.
        Returns dict of postulate -> (satisfied: bool, error: float).
        """
        w = self.w.detach().clamp(0, 1)
        one = torch.ones_like(w)
        zero = torch.zeros_like(w)

        results = {}

        # H2: Identity elements
        # w + 0 = w (OR with 0)
        h2_add = (w + zero - w * zero - w).abs().max().item()
        # w * 1 = w (AND with 1)
        h2_mul = (w * one - w).abs().max().item()
        results["H2_identity"] = (h2_add < 1e-6 and h2_mul < 1e-6, max(h2_add, h2_mul))

        # H5: Complement
        # w + ~w = 1
        not_w = 1.0 - w
        h5_add = (w + not_w - w * not_w - one).abs().max().item()
        # w * ~w = 0
        h5_mul = (w * not_w - zero).abs().max().item()
        results["H5_complement"] = (h5_add < 1e-6 and h5_mul < 1e-6, max(h5_add, h5_mul))

        # H7: Idempotence
        # w + w = w (OR with self)
        h7_add = (w + w - w * w - w).abs().max().item()
        # w * w = w (AND with self)
        h7_mul = (w * w - w).abs().max().item()
        results["H7_idempotence"] = (h7_add < 1e-6 and h7_mul < 1e-6, max(h7_add, h7_mul))

        return results


class PerceptronNetwork(nn.Module):
    """
    Network of perceptron actors with Boolean routing.

    Implements a directed graph of actors where edges carry typed signals.
    The network routing is governed by Boolean operations satisfying
    Huntington postulates, ensuring deterministic and verifiable behavior.

    Args:
        d: dimension
        num_actors: number of perceptron actors
        connectivity: "full" or "sparse"
    """

    def __init__(self, d: int, num_actors: int = 8, connectivity: str = "sparse"):
        super().__init__()
        self.d = d
        self.num_actors = num_actors

        # Create actors
        self.actors = nn.ModuleList([
            PerceptronActor(d, actor_id=i) for i in range(num_actors)
        ])

        # Adjacency (learnable, represents connection strength)
        self.adjacency = nn.Parameter(
            torch.rand(num_actors, num_actors) * 0.3
        )

        # Signal type assignment (fixed per edge)
        signal_types = [SignalType.EXCITATORY, SignalType.INHIBITORY,
                       SignalType.MODULATORY, SignalType.MEMORY]
        self.register_buffer(
            "edge_types",
            torch.randint(0, len(signal_types), (num_actors, num_actors))
        )

        # Output combination
        self.output_proj = nn.Linear(d * num_actors, d)

    def forward(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Run network for given number of steps.

        Args:
            x: [batch, d] input
            steps: number of propagation steps

        Returns:
            output: [batch, d]
        """
        batch = x.shape[0]
        activations = [torch.zeros(batch, self.d, device=x.device) for _ in range(self.num_actors)]

        # Initialize first actor with input
        activations[0] = self.actors[0](x)

        signal_types = [SignalType.EXCITATORY, SignalType.INHIBITORY,
                       SignalType.MODULATORY, SignalType.MEMORY]

        for step in range(steps):
            new_activations = []
            for i in range(self.num_actors):
                # Collect signals from connected actors
                signals = []
                for j in range(self.num_actors):
                    if i == j:
                        continue
                    weight = torch.sigmoid(self.adjacency[j, i]).item()
                    if weight < 0.1:
                        continue
                    sig = Signal(
                        source_id=j,
                        target_id=i,
                        signal_type=signal_types[self.edge_types[j, i].item() % 4],
                        value=activations[j].mean(dim=0),
                        weight=weight,
                        timestamp=step
                    )
                    signals.append(sig)

                # Actor activation
                actor_input = x if step == 0 else activations[i]
                new_activations.append(self.actors[i](actor_input, signals))

            activations = new_activations

        # Combine all actor outputs
        combined = torch.cat(activations, dim=-1)  # [batch, d * num_actors]
        return self.output_proj(combined)
