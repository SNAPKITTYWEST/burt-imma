"""
BURT-IMMA CUDA Extension Build Script
License: BSL-1.1
Contact: jessica@collectivekitty.com

Build with:
  python setup.py install
  # or for development:
  pip install -e .
"""

import os
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    HAS_CUDA_EXT = True
except ImportError:
    from setuptools import Extension
    HAS_CUDA_EXT = False


def get_cuda_extensions():
    """Build CUDA extension if torch and CUDA are available."""
    if not HAS_CUDA_EXT:
        print("WARNING: torch.utils.cpp_extension not available. "
              "Building without CUDA acceleration.")
        return []

    # Source files
    sources = [
        "bindings.cpp",
    ]

    # Check for CUDA kernel source files
    cuda_src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "cuda")
    if os.path.isdir(cuda_src_dir):
        for fname in os.listdir(cuda_src_dir):
            if fname.endswith(".cu"):
                sources.append(os.path.join(cuda_src_dir, fname))

    # Include directories
    include_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "include"),
    ]

    # CUDA include path
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    if cuda_home:
        include_dirs.append(os.path.join(cuda_home, "include"))

    # Compiler flags
    extra_compile_args = {
        "cxx": ["-std=c++17", "-O3"],
        "nvcc": [
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # T4 / RTX 2080
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 3080/3090
            "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        ],
    }

    ext = CUDAExtension(
        name="_burt_imma_cuda",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )

    return [ext]


setup(
    name="burt-imma",
    version="0.1.0",
    description="BURT-IMMA: Bi-encoder Unified Retrieval Transformer with "
                "Interferometric Matrix Memory Architecture",
    author="SnapKitty",
    author_email="jessica@collectivekitty.com",
    license="BSL-1.1",
    url="https://github.com/SNAPKITTYWEST/burt-imma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "train": [
            "pyyaml>=6.0",
            "tqdm>=4.64",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-benchmark>=4.0",
        ],
    },
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension} if HAS_CUDA_EXT else {},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
