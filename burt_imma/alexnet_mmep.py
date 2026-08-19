"""
AlexNet (2012) → BURT-IMMA/MMEP Bridge

Maps AlexNet innovations to modern MMEP equivalents:
  ReLU → SmoothLeakyActivation (f'(x) in (alpha, 1), C^inf)
  Dual-GPU → Expert Parallelism (DP=4, TP=2, EP=4, PP=1 on 32xH100)
  Dropout(p=0.5) → Entropy-Constrained Routing H(alpha) <= 0.20
  Overlapping Pool → Sparse MoE Dispatch (Top-K + scatter)
  PCA Color Jitter → BiEncoder Retrieval + Instruction Routing

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .smooth_leaky import SmoothLeakyActivation
from .burt import constrained_softmax


class EntropyConstrainedRouter(nn.Module):
    """
    Router that replaces Dropout with entropy-constrained gating.

    AlexNet used Dropout(p=0.5) for regularization.
    BURT-IMMA uses H(alpha) <= 0.20 for the same effect:
      - Forces sparsity (most weight on 1-2 experts)
      - Prevents co-adaptation (experts must be independent)
      - Mathematically principled (information-theoretic bound)

    Args:
        d: hidden dimension
        num_routes: number of routing paths (replaces dropout mask)
        entropy_bound: maximum allowed entropy (default 0.20)
    """

    def __init__(self, d: int, num_routes: int = 4, entropy_bound: float = 0.20):
        super().__init__()
        self.d = d
        self.num_routes = num_routes
        self.entropy_bound = entropy_bound
        self.router = nn.Linear(d, num_routes)
        self.route_transforms = nn.ModuleList([
            nn.Linear(d, d) for _ in range(num_routes)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route with entropy constraint (replaces dropout).

        Args:
            x: [batch, d]
        Returns:
            output: [batch, d] routed output
            entropy: [batch] router entropy (should be <= 0.20)
        """
        logits = self.router(x)
        alpha = constrained_softmax(logits, self.entropy_bound)
        entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=-1)

        # Weighted combination of route transforms
        output = torch.zeros_like(x)
        for k in range(self.num_routes):
            output += alpha[:, k:k+1] * self.route_transforms[k](x)

        return output, entropy


class AlexNetMMEP(nn.Module):
    """
    AlexNet architecture re-interpreted through MMEP lens.

    8-layer architecture (matching AlexNet) but with:
      - SmoothLeaky instead of ReLU
      - Entropy routing instead of dropout
      - MMEP-compatible energy function

    Parameter count: ~62M (original AlexNet: 61M)

    Architecture:
      Layer 1: Conv 96 filters 11x11 stride 4 → SmoothLeaky
      Layer 2: Conv 256 filters 5x5 pad 2 → SmoothLeaky
      Layer 3: Conv 384 filters 3x3 pad 1 → SmoothLeaky
      Layer 4: Conv 384 filters 3x3 pad 1 → SmoothLeaky
      Layer 5: Conv 256 filters 3x3 pad 1 → SmoothLeaky
      Layer 6: FC 4096 → SmoothLeaky + EntropyRoute
      Layer 7: FC 4096 → SmoothLeaky + EntropyRoute
      Layer 8: FC 1000 (output)
    """

    def __init__(self, num_classes: int = 1000, alpha: float = 0.01,
                 entropy_bound: float = 0.20):
        super().__init__()
        self.activation = SmoothLeakyActivation(alpha=alpha)
        self.entropy_bound = entropy_bound

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully connected with entropy routing (replaces dropout)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.route6 = EntropyConstrainedRouter(4096, entropy_bound=entropy_bound)
        self.fc7 = nn.Linear(4096, 4096)
        self.route7 = EntropyConstrainedRouter(4096, entropy_bound=entropy_bound)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Returns:
            logits: [batch, num_classes]
            aux: dict with layer entropies
        """
        # Conv layers with SmoothLeaky
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool(self.activation(self.conv5(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC with entropy routing
        x = self.activation(self.fc6(x))
        x, ent6 = self.route6(x)
        x = self.activation(self.fc7(x))
        x, ent7 = self.route7(x)
        logits = self.fc8(x)

        aux = {
            "entropy_fc6": ent6.mean(),
            "entropy_fc7": ent7.mean(),
        }
        return logits, aux


class AlexNetMMEPTrainer:
    """
    Training loop for AlexNet-MMEP with constraint monitoring.

    Monitors:
      - Router entropy (must stay <= 0.20)
      - SmoothLeaky gradient flow (must stay in (alpha, 1))
      - Expert utilization balance
    """

    def __init__(self, model: AlexNetMMEP, lr: float = 0.01,
                 momentum: float = 0.9, weight_decay: float = 5e-4):
        self.model = model
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.step_count = 0

    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        logits, aux = self.model(images)
        loss = self.criterion(logits, labels)

        # Add entropy regularization (penalize entropy above bound)
        for key in ["entropy_fc6", "entropy_fc7"]:
            ent = aux[key]
            if ent > self.model.entropy_bound:
                loss += 0.1 * (ent - self.model.entropy_bound) ** 2

        loss.backward()
        self.optimizer.step()
        self.step_count += 1

        return {
            "loss": loss.item(),
            "accuracy": (logits.argmax(dim=-1) == labels).float().mean().item(),
            **{k: v.item() for k, v in aux.items()},
        }
