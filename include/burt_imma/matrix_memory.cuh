#pragma once
#include "common.cuh"

/**
 * Matrix Memory Kernel — CIFG Outer Product Update
 *
 * Implements the Coupled Input-Forget Gate (CIFG) matrix memory:
 *   C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
 *   i_t = 1 - f_t  (CIFG constraint)
 *
 * This is a matrix-valued associative memory where:
 *   - f_t: forget gate (sigmoid output, controls retention)
 *   - v_t: value vector (what to remember)
 *   - k_t: key vector (how to index it)
 *   - C_t: memory matrix (outer product accumulator)
 *
 * Conservation property: Tr(C_t) is bounded by initial + writes.
 */

/**
 * Single-step CIFG outer product memory update.
 * One thread block per memory head.
 */
__global__ void cifg_outer_product_kernel(
    float*       __restrict__ C,          // [num_heads, d_mem, d_mem] (in-place)
    const float* __restrict__ f_gate,     // [batch, num_heads] forget gates
    const float* __restrict__ v,          // [batch, num_heads, d_mem] values
    const float* __restrict__ k,          // [batch, num_heads, d_mem] keys
    const int    d_mem,
    const int    num_heads,
    const int    batch_size
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    float* C_head = C + head * d_mem * d_mem;

    // Process each batch element sequentially (memory is shared across batch)
    for (int b = 0; b < batch_size; b++) {
        float f = f_gate[b * num_heads + head];
        float i = 1.0f - f;  // CIFG: i = 1 - f

        const float* v_b = v + b * num_heads * d_mem + head * d_mem;
        const float* k_b = k + b * num_heads * d_mem + head * d_mem;

        // Update C: C = f * C + i * (v @ k^T)
        for (int row = tid; row < d_mem; row += blockDim.x) {
            float v_val = v_b[row];
            for (int col = 0; col < d_mem; col++) {
                int idx = row * d_mem + col;
                C_head[idx] = f * C_head[idx] + i * v_val * k_b[col];
            }
        }
        __syncthreads();
    }
}

/**
 * Batched CIFG update — each batch element has its own memory.
 */
__global__ void batched_cifg_kernel(
    float*       __restrict__ C,          // [batch, num_heads, d_mem, d_mem]
    const float* __restrict__ f_gate,     // [batch, num_heads]
    const float* __restrict__ v,          // [batch, num_heads, d_mem]
    const float* __restrict__ k,          // [batch, num_heads, d_mem]
    const int    d_mem,
    const int    num_heads,
    const int    batch_size
) {
    int b = blockIdx.x;
    int head = blockIdx.y;
    if (b >= batch_size || head >= num_heads) return;

    int tid = threadIdx.x;
    float* C_bh = C + (b * num_heads + head) * d_mem * d_mem;
    float f = f_gate[b * num_heads + head];
    float i_gate = 1.0f - f;

    const float* v_bh = v + (b * num_heads + head) * d_mem;
    const float* k_bh = k + (b * num_heads + head) * d_mem;

    for (int row = tid; row < d_mem; row += blockDim.x) {
        float v_val = v_bh[row];
        for (int col = 0; col < d_mem; col++) {
            int idx = row * d_mem + col;
            C_bh[idx] = f * C_bh[idx] + i_gate * v_val * k_bh[col];
        }
    }
}

/**
 * Compute trace of memory matrix (diagnostic).
 */
__global__ void compute_trace_kernel(
    const float* __restrict__ C,          // [batch, num_heads, d_mem, d_mem]
    float*       __restrict__ traces,     // [batch, num_heads]
    const int    d_mem,
    const int    num_heads,
    const int    batch_size
) {
    int b = blockIdx.x;
    int head = blockIdx.y;
    if (b >= batch_size || head >= num_heads) return;

    const float* C_bh = C + (b * num_heads + head) * d_mem * d_mem;
    float trace = 0.0f;
    for (int d = threadIdx.x; d < d_mem; d += blockDim.x) {
        trace += C_bh[d * d_mem + d];
    }
    trace = block_reduce_sum(trace);
    if (threadIdx.x == 0) {
        traces[b * num_heads + head] = trace;
    }
}

/**
 * Check trace drift — flags if any memory head has drifted beyond threshold.
 */
__global__ void check_trace_drift_kernel(
    const float* __restrict__ traces,     // [batch, num_heads]
    const float* __restrict__ init_traces,// [batch, num_heads]
    int*         __restrict__ violations, // [1]
    const float  threshold,
    const int    num_heads,
    const int    batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads) return;

    float drift = fabsf(traces[idx] - init_traces[idx]);
    if (drift > threshold) {
        atomicAdd(violations, 1);
    }
}

/**
 * Memory read: output = C @ query
 */
__global__ void matrix_memory_read_kernel(
    const float* __restrict__ C,         // [batch, num_heads, d_mem, d_mem]
    const float* __restrict__ queries,   // [batch, num_heads, d_mem]
    float*       __restrict__ output,    // [batch, num_heads, d_mem]
    const int    d_mem,
    const int    num_heads,
    const int    batch_size
) {
    int b = blockIdx.x;
    int head = blockIdx.y;
    if (b >= batch_size || head >= num_heads) return;

    const float* C_bh = C + (b * num_heads + head) * d_mem * d_mem;
    const float* q = queries + (b * num_heads + head) * d_mem;
    float* out = output + (b * num_heads + head) * d_mem;

    int tid = threadIdx.x;
    for (int row = tid; row < d_mem; row += blockDim.x) {
        float val = 0.0f;
        for (int col = 0; col < d_mem; col++) {
            val += C_bh[row * d_mem + col] * q[col];
        }
        out[row] = val;
    }
}

/**
 * Launch wrappers.
 */
inline cudaError_t launch_cifg_update(
    float* C, const float* f_gate, const float* v, const float* k,
    int d_mem, int num_heads, int batch_size, cudaStream_t stream = 0
) {
    cifg_outer_product_kernel<<<num_heads, min(d_mem, 256), 0, stream>>>(
        C, f_gate, v, k, d_mem, num_heads, batch_size);
    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}

inline cudaError_t launch_batched_cifg(
    float* C, const float* f_gate, const float* v, const float* k,
    int d_mem, int num_heads, int batch_size, cudaStream_t stream = 0
) {
    dim3 grid(batch_size, num_heads);
    batched_cifg_kernel<<<grid, min(d_mem, 256), 0, stream>>>(
        C, f_gate, v, k, d_mem, num_heads, batch_size);
    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}

inline cudaError_t launch_compute_trace(
    const float* C, float* traces, int d_mem, int num_heads, int batch_size,
    cudaStream_t stream = 0
) {
    dim3 grid(batch_size, num_heads);
    compute_trace_kernel<<<grid, min(d_mem, 256), 0, stream>>>(
        C, traces, d_mem, num_heads, batch_size);
    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}
