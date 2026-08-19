"""
BURT-IMMA pytest configuration
License: BSL-1.1
Contact: jessica@collectivekitty.com
"""

import os
import pytest
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


# ---------------------------------------------------------------------------
# Custom marks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cuda: marks tests requiring CUDA GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if no GPU available."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            if not HAS_TORCH or not torch.cuda.is_available():
                item.add_marker(skip_cuda)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Return device string: 'cuda' if available, else 'cpu'."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture
def config():
    """Load ablation_arithmetic.yaml config."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "ablation_arithmetic.yaml"
    )
    if HAS_YAML and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Fallback default config
        return {
            "experiment": {"name": "mmep_arithmetic_ablation", "seed": 42, "device": "cpu"},
            "model": {
                "hidden_dim": 256,
                "num_layers": 4,
                "num_experts": 4,
                "top_k": 1,
                "d_mem": 256,
            },
            "mmep": {
                "T_free": 20,
                "T_nudge": 4,
                "alpha_relax": 0.5,
                "beta": 0.1,
                "rho_ret": 1.0,
                "rho_inst": 0.5,
                "lambda_max": 0.95,
                "lr_W": 0.001,
                "lr_C_global": 0.01,
                "lr_C_expert": 0.005,
            },
            "data": {
                "corpus_size": 10000,
                "train_queries": 5000,
                "val_queries": 500,
                "test_queries": 1000,
                "max_seq_len": 128,
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 50,
                "eval_every": 5,
                "checkpoint_every": 10,
                "gradient_clip": 1.0,
            },
            "ablations": [
                {"name": "full_mmep", "description": "Complete MMEP", "disable": []},
                {"name": "no_memory", "disable": ["C_global", "C_expert"]},
                {"name": "no_constraint", "disable": ["projection"]},
                {"name": "no_moe", "override": {"num_experts": 1}},
                {"name": "standard_bp", "disable": ["mmep"], "use_backprop": True},
            ],
            "logging": {
                "wandb_project": "burt-imma-ablation",
                "log_entropy": True,
                "log_spectral_norm": True,
                "log_memory_trace": True,
            },
        }


@pytest.fixture
def random_seed():
    """Set deterministic random seed (42) for reproducibility."""
    seed = 42
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def sample_batch(device, config, random_seed):
    """Generate a random tensor batch for testing."""
    batch_size = config["training"]["batch_size"]
    hidden_dim = config["model"]["hidden_dim"]
    seq_len = 16  # Short sequence for fast tests

    if HAS_TORCH:
        x = torch.randn(batch_size, seq_len, hidden_dim)
        if device == "cuda":
            x = x.cuda()
        return x
    else:
        return np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
