cmake_minimum_required(VERSION 3.18)
project(burt_imma LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target architectures: sm_86 (RTX 3080) + sm_90 (H100)
set(CMAKE_CUDA_ARCHITECTURES "86;90" CACHE STRING "CUDA architectures")

find_package(CUDAToolkit 12.0 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA kernel sources
set(KERNEL_SOURCES
    src/mmep_step.cu
)

# Shared library
add_library(burt_imma SHARED ${KERNEL_SOURCES})
target_link_libraries(burt_imma CUDA::cudart)
target_compile_options(burt_imma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
)

# Test binary
add_executable(burt_imma_test src/mmep_step.cu)
target_link_libraries(burt_imma_test CUDA::cudart)
target_compile_definitions(burt_imma_test PRIVATE BURT_IMMA_TEST_MAIN)
target_compile_options(burt_imma_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
    >
)

# Optional: pybind11 Python bindings
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_burt_imma_cuda python/bindings.cpp ${KERNEL_SOURCES})
    target_link_libraries(_burt_imma_cuda PRIVATE CUDA::cudart)
    target_compile_options(_burt_imma_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -O3
            --use_fast_math
        >
    )
endif()

# Install
install(TARGETS burt_imma DESTINATION lib)
install(DIRECTORY include/burt_imma DESTINATION include)
