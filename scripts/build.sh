#!/bin/bash
# BURT-IMMA Build Script
# License: BSL-1.1
# Contact: jessica@collectivekitty.com
set -e
echo "Building BURT-IMMA CUDA kernels..."
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;90"
make -j$(nproc)
echo "Build complete."
