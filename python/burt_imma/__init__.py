"""
BURT-IMMA: Bi-encoder Unified Retrieval Transformer with
Interferometric Matrix Memory Architecture

License: BSL-1.1
Contact: jessica@collectivekitty.com
"""

__version__ = "0.1.0"

# Try to import CUDA-accelerated kernels, fall back to pure PyTorch
_CUDA_AVAILABLE = False

try:
    import _burt_imma_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _burt_imma_cuda = None

# Import kernel wrappers (these handle CUDA/CPU fallback internally)
from .kernels import (
    constrained_softmax,
    cifg_update,
    batched_cifg_update,
    sparse_moe_dispatch,
    biencoder_attention,
    attention_softmax,
)

from .utils import (
    spectral_norm,
    entropy,
    check_huntington,
    load_config,
    set_seed,
    Timer,
)

__all__ = [
    # Kernel functions
    "constrained_softmax",
    "cifg_update",
    "batched_cifg_update",
    "sparse_moe_dispatch",
    "biencoder_attention",
    "attention_softmax",
    # Utilities
    "spectral_norm",
    "entropy",
    "check_huntington",
    "load_config",
    "set_seed",
    "Timer",
    # Meta
    "__version__",
    "_CUDA_AVAILABLE",
]
