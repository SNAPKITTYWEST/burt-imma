"""
SmoothLeakyActivation — First-Principles Validated Activation Function

f(x) = alpha*x + ((1-alpha)/beta) * [softplus(beta*x) - ln(2)]
f'(x) = alpha + (1-alpha) * sigmoid(beta*x)

Properties (all formally proven in lean4/SmoothLeakyActivation.lean):
  1. f'(x) in (alpha, 1) for all x → NO DYING NEURONS
  2. f'(x) → 1 as x → +inf → NO VANISHING GRADIENT
  3. f(0) = 0 → zero-centered
  4. C^inf → smooth for equilibrium dynamics (MMEP requirement)
  5. f(x) < 0 for x < 0 → negative range (better than ReLU)

Parameters:
  alpha in (0, 1): minimum gradient (leaky slope). Default 0.01.
  beta > 0: sharpness of transition. Default 1.0.

Bridge to AlexNet:
  - ReLU (alpha=0, beta→inf) is the degenerate case
  - SmoothLeaky (alpha=0.01, beta=1.0) is the correct generalization
  - Removes dying neuron problem while preserving sparsity
  - C^inf smoothness enables MMEP equilibrium dynamics

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SmoothLeakyActivation(nn.Module):
    """
    SmoothLeaky activation function.

    f(x) = alpha*x + ((1-alpha)/beta) * (softplus(beta*x) - ln(2))

    This satisfies all 4 axioms required for MMEP convergence:
      A1. Gradient bounded: f'(x) in (alpha, 1)
      A2. Asymptotically linear: f'(x) → 1 as x → +inf
      A3. Zero-centered: f(0) = 0
      A4. Smooth: C^inf (all derivatives exist)

    Args:
        alpha: minimum gradient / leaky slope (default 0.01)
        beta: transition sharpness (default 1.0)
        learnable: if True, alpha and beta are learnable parameters
    """

    def __init__(self, alpha: float = 0.01, beta: float = 1.0,
                 learnable: bool = False):
        super().__init__()

        if learnable:
            # Store in unconstrained space, apply constraints in forward
            self._alpha_raw = nn.Parameter(torch.tensor(math.log(alpha / (1 - alpha))))
            self._beta_raw = nn.Parameter(torch.tensor(math.log(beta)))
        else:
            self.register_buffer("_alpha", torch.tensor(alpha))
            self.register_buffer("_beta", torch.tensor(beta))

        self.learnable = learnable
        self._ln2 = math.log(2.0)

    @property
    def alpha(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self._alpha_raw)
        return self._alpha

    @property
    def beta(self) -> torch.Tensor:
        if self.learnable:
            return torch.exp(self._beta_raw).clamp(min=0.01, max=100.0)
        return self._beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        f(x) = alpha*x + ((1-alpha)/beta) * (softplus(beta*x) - ln(2))
        """
        a = self.alpha
        b = self.beta

        # Numerically stable softplus: log(1 + exp(z))
        # For large |z|: softplus(z) ≈ z if z > 20, ≈ exp(z) if z < -20
        bx = b * x
        softplus_bx = F.softplus(bx)

        return a * x + ((1.0 - a) / b) * (softplus_bx - self._ln2)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        f'(x) = alpha + (1-alpha) * sigmoid(beta*x)

        Always in (alpha, 1) — no dying neurons, no vanishing gradient.
        """
        a = self.alpha
        b = self.beta
        return a + (1.0 - a) * torch.sigmoid(b * x)

    def verify_properties(self, x: Optional[torch.Tensor] = None) -> dict:
        """
        Verify all 4 axioms on given input tensor.

        Returns dict of property -> (satisfied, value).
        """
        if x is None:
            x = torch.linspace(-10, 10, 1000)

        results = {}
        a = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha

        # A1: gradient bounded in (alpha, 1)
        grad = self.derivative(x)
        results["A1_gradient_bounded"] = (
            grad.min().item() > a - 1e-6 and grad.max().item() < 1.0 + 1e-6,
            (grad.min().item(), grad.max().item())
        )

        # A2: asymptotically linear (f'(x) → 1 as x → +inf)
        grad_large = self.derivative(torch.tensor([100.0]))
        results["A2_asymptotic_linear"] = (
            abs(grad_large.item() - 1.0) < 1e-4,
            grad_large.item()
        )

        # A3: zero-centered (f(0) = 0)
        f_zero = self.forward(torch.tensor([0.0]))
        results["A3_zero_centered"] = (
            abs(f_zero.item()) < 1e-6,
            f_zero.item()
        )

        # A4: smooth (check that second derivative exists and is continuous)
        # Approximation: finite difference of derivative should be smooth
        dx = 0.001
        x_test = torch.linspace(-5, 5, 100)
        grad1 = self.derivative(x_test)
        grad2 = self.derivative(x_test + dx)
        second_deriv = (grad2 - grad1) / dx
        results["A4_smooth"] = (
            not torch.isnan(second_deriv).any().item(),
            second_deriv.abs().max().item()
        )

        # Bonus: negative range
        f_neg = self.forward(torch.tensor([-1.0]))
        results["negative_range"] = (
            f_neg.item() < 0,
            f_neg.item()
        )

        return results


class SmoothLeakyPerceptronActor(nn.Module):
    """
    Perceptron actor using SmoothLeaky activation instead of sigmoid.

    This gives better gradient flow while maintaining Boolean compatibility
    (outputs are rescaled to [0, 1] after activation).
    """

    def __init__(self, d: int, alpha: float = 0.01, beta: float = 1.0):
        super().__init__()
        self.d = d
        self.activation = SmoothLeakyActivation(alpha, beta)
        self.weight = nn.Parameter(torch.randn(d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SmoothLeaky + sigmoid normalization."""
        pre_act = self.weight.unsqueeze(0) * x + self.bias
        activated = self.activation(pre_act)
        # Project to [0, 1] for Boolean compatibility
        return torch.sigmoid(activated)


class SmoothLeakyMMEP(nn.Module):
    """
    MMEP layer using SmoothLeaky as the nonlinearity.

    The relaxation dynamics become:
      h_{t+1} = (1-alpha_relax) * h_t + alpha_relax * SmoothLeaky(W @ h_{t-1} + C)

    Since SmoothLeaky is C^inf with bounded gradient, this guarantees:
      - Existence of equilibrium (Banach fixed point when ||W|| < 1)
      - Convergence of free phase (energy decrease)
      - Valid EP gradients (smooth implicit function theorem)
    """

    def __init__(self, d: int, alpha_activation: float = 0.01):
        super().__init__()
        self.d = d
        self.activation = SmoothLeakyActivation(alpha_activation)
        self.W = nn.Linear(d, d, bias=False)
        self.C = nn.Parameter(torch.zeros(d))

    def relaxation_step(self, h: torch.Tensor, alpha_relax: float = 0.5) -> torch.Tensor:
        """Single relaxation step."""
        pre_act = self.W(h) + self.C
        activated = self.activation(pre_act)
        return (1.0 - alpha_relax) * h + alpha_relax * activated

    def forward(self, x: torch.Tensor, T_free: int = 20,
                alpha_relax: float = 0.5) -> torch.Tensor:
        """Run free phase to equilibrium."""
        h = x
        for _ in range(T_free):
            h = self.relaxation_step(h, alpha_relax)
        return h
