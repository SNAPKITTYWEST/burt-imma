#!/bin/bash
# BURT-IMMA Test Runner
# License: BSL-1.1
# Contact: jessica@collectivekitty.com
set -e

echo "============================================"
echo "  BURT-IMMA Test Suite"
echo "============================================"

RUN_CUDA=0

for arg in "$@"; do
    case $arg in
        --cuda)
            RUN_CUDA=1
            shift
            ;;
    esac
done

echo ""
echo "[1/2] Running Python test suite..."
pytest tests/ -v --tb=short

if [ $RUN_CUDA -eq 1 ]; then
    echo ""
    echo "[2/2] Running CUDA kernel tests..."
    if [ -f build/test_kernels ]; then
        ./build/test_kernels
    else
        echo "  CUDA test binary not found. Run ./scripts/build.sh first."
        exit 1
    fi
else
    echo ""
    echo "[2/2] Skipping CUDA kernel tests (pass --cuda to enable)"
fi

echo ""
echo "============================================"
echo "  All tests passed."
echo "============================================"
