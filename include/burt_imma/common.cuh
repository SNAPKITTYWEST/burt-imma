#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define WARP_SIZE 32
#define BURT_IMMA_KERNEL_CHECK() \
    { cudaError_t err = cudaGetLastError(); \
      if (err != cudaSuccess) return err; }
#define BURT_IMMA_CHECK(call) \
    { cudaError_t err = (call); \
      if (err != cudaSuccess) return err; }

// =============================================================================
// WARP PRIMITIVES
// =============================================================================

/**
 * Warp-level sum reduction.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level max reduction.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Block-level sum reduction using shared memory.
 * Assumes blockDim.x is a multiple of WARP_SIZE.
 */
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

/**
 * Block-level max reduction using shared memory.
 */
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -CUDART_INF_F;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

/**
 * Safe log to prevent NaN (clamps to minimum epsilon).
 */
__device__ __forceinline__ float safe_log(float x) {
    return logf(fmaxf(x, 1e-10f));
}

/**
 * Softmax entropy: H = -sum_i p_i * log(p_i)
 * Assumes p is already a valid probability distribution.
 */
__device__ __forceinline__ float softmax_entropy(const float* p, int n) {
    float h = 0.0f;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-10f) {
            h -= p[i] * safe_log(p[i]);
        }
    }
    return h;
}

/**
 * Hardtanh activation (bounded for energy stability).
 */
__device__ __forceinline__ float hardtanh(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}

/**
 * GELU approximation.
 */
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
