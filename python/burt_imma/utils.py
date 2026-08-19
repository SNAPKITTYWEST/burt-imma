"""
BURT-IMMA Utilities
License: BSL-1.1
Contact: jessica@collectivekitty.com
"""

import os
import time
import random
from typing import Any, Dict, Optional
from contextlib import contextmanager

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def spectral_norm(W) -> float:
    """Compute the largest singular value (spectral norm) of a matrix.

    Args:
        W: numpy array or torch Tensor of shape (m, n)

    Returns:
        Largest singular value as a float
    """
    if HAS_TORCH and isinstance(W, torch.Tensor):
        s = torch.linalg.svdvals(W)
        return s[0].item()
    else:
        W_np = np.asarray(W)
        if W_np.ndim < 2:
            return float(np.abs(W_np).max())
        s = np.linalg.svd(W_np, compute_uv=False)
        return float(s[0])


def entropy(p) -> float:
    """Compute the Shannon entropy of a probability distribution.

    Args:
        p: numpy array or torch Tensor representing a probability distribution
           (must sum to 1, all elements >= 0)

    Returns:
        Entropy in nats (natural log base)
    """
    if HAS_TORCH and isinstance(p, torch.Tensor):
        p_clamped = p.clamp(min=1e-10)
        return -(p_clamped * p_clamped.log()).sum().item()
    else:
        p_np = np.asarray(p, dtype=np.float64)
        p_clipped = np.clip(p_np, 1e-10, 1.0)
        return float(-np.sum(p_clipped * np.log(p_clipped)))


def check_huntington(actor, x, y) -> Dict[str, bool]:
    """Verify Huntington postulates for a Boolean algebra actor.

    The Huntington postulates define a Boolean algebra (B, +, *, ', 0, 1):
      H1 (Commutativity): x + y = y + x, x * y = y * x
      H2 (Distributivity): x * (y + z) = (x*y) + (x*z),
                            x + (y * z) = (x+y) * (x+z)
      H3 (Identity): x + 0 = x, x * 1 = x
      H4 (Complement): x + x' = 1, x * x' = 0

    Args:
        actor: Object with methods `join(a, b)`, `meet(a, b)`, `complement(a)`,
               and attributes `zero` and `one`.
        x: First element
        y: Second element

    Returns:
        Dict mapping postulate name to whether it holds
    """
    results = {}

    # H1: Commutativity
    try:
        h1_join = np.allclose(
            np.asarray(actor.join(x, y)),
            np.asarray(actor.join(y, x)),
            atol=1e-6
        )
        h1_meet = np.allclose(
            np.asarray(actor.meet(x, y)),
            np.asarray(actor.meet(y, x)),
            atol=1e-6
        )
        results["H1_commutativity"] = bool(h1_join and h1_meet)
    except (AttributeError, TypeError):
        results["H1_commutativity"] = False

    # H3: Identity
    try:
        h3_join = np.allclose(
            np.asarray(actor.join(x, actor.zero)),
            np.asarray(x),
            atol=1e-6
        )
        h3_meet = np.allclose(
            np.asarray(actor.meet(x, actor.one)),
            np.asarray(x),
            atol=1e-6
        )
        results["H3_identity"] = bool(h3_join and h3_meet)
    except (AttributeError, TypeError):
        results["H3_identity"] = False

    # H4: Complement
    try:
        x_comp = actor.complement(x)
        h4_join = np.allclose(
            np.asarray(actor.join(x, x_comp)),
            np.asarray(actor.one),
            atol=1e-6
        )
        h4_meet = np.allclose(
            np.asarray(actor.meet(x, x_comp)),
            np.asarray(actor.zero),
            atol=1e-6
        )
        results["H4_complement"] = bool(h4_join and h4_meet)
    except (AttributeError, TypeError):
        results["H4_complement"] = False

    return results


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        ImportError: If PyYAML is not installed
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: Python random, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Timer:
    """Context manager for profiling code blocks.

    Usage:
        with Timer("forward pass") as t:
            output = model(x)
        print(t.elapsed)  # seconds

        # Or accumulate multiple measurements:
        timer = Timer("training")
        for batch in data:
            with timer:
                train_step(batch)
        print(timer.total, timer.count, timer.mean)
    """

    def __init__(self, name: str = "timer", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.elapsed: float = 0.0
        self.total: float = 0.0
        self.count: int = 0
        self._start: Optional[float] = None

    @property
    def mean(self) -> float:
        """Mean elapsed time across all measurements."""
        return self.total / max(self.count, 1)

    def __enter__(self) -> "Timer":
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        self.total += self.elapsed
        self.count += 1
        self._start = None

        if self.verbose:
            print(f"[{self.name}] {self.elapsed*1000:.2f} ms")

    def reset(self) -> None:
        """Reset all accumulated measurements."""
        self.elapsed = 0.0
        self.total = 0.0
        self.count = 0
        self._start = None

    def __repr__(self) -> str:
        return (f"Timer(name={self.name!r}, count={self.count}, "
                f"total={self.total:.4f}s, mean={self.mean*1000:.2f}ms)")
