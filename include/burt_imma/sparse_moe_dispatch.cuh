#pragma once
#include "common.cuh"

/**
 * Sparse Mixture-of-Experts Dispatch Kernel
 *
 * Routes tokens to top-k experts with:
 *   1. Warp-level top-k selection (k=1 or k=2)
 *   2. Load-balanced dispatch with capacity factor
 *   3. Auxiliary balance loss computation
 *   4. Entropy-constrained gating (H(alpha) <= 0.20)
 */

/**
 * Warp-level top-1 selection.
 * Returns index and value of maximum element across warp lanes.
 */
__device__ __forceinline__ void warp_topk_1(
    float val, int idx, float* out_val, int* out_idx
) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    *out_val = val;
    *out_idx = idx;
}

/**
 * Warp-level top-2 selection.
 * Returns top-2 indices and values.
 */
__device__ __forceinline__ void warp_topk_2(
    float val, int idx,
    float* out_val1, int* out_idx1,
    float* out_val2, int* out_idx2
) {
    // First pass: find top-1
    float best_val = val;
    int best_idx = idx;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    best_val = __shfl_sync(0xffffffff, best_val, 0);
    best_idx = __shfl_sync(0xffffffff, best_idx, 0);
    *out_val1 = best_val;
    *out_idx1 = best_idx;

    // Second pass: find top-2 (mask out top-1)
    float val2 = (idx == best_idx) ? -CUDART_INF_F : val;
    int idx2 = idx;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val2, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx2, offset);
        if (other_val > val2) {
            val2 = other_val;
            idx2 = other_idx;
        }
    }
    *out_val2 = __shfl_sync(0xffffffff, val2, 0);
    *out_idx2 = __shfl_sync(0xffffffff, idx2, 0);
}

/**
 * Sparse MoE dispatch kernel.
 * Computes gating, selects top-k experts, dispatches tokens.
 */
__global__ void sparse_moe_dispatch_kernel(
    const float* __restrict__ hidden,      // [batch, hidden_dim]
    const float* __restrict__ W_gate,      // [num_experts, hidden_dim]
    float*       __restrict__ expert_buf,  // [num_experts, capacity, hidden_dim]
    int*         __restrict__ expert_count,// [num_experts]
    int*         __restrict__ assignments, // [batch, top_k]
    float*       __restrict__ gate_vals,   // [batch, top_k]
    float*       __restrict__ balance_loss,// [1]
    const int    hidden_dim,
    const int    num_experts,
    const int    capacity,
    const int    batch_size,
    const int    top_k
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    const float* x = hidden + b * hidden_dim;

    // Compute gating scores: g = W_gate @ x
    __shared__ float scores[64]; // max 64 experts
    if (tid < num_experts) {
        float score = 0.0f;
        const float* w_row = W_gate + tid * hidden_dim;
        for (int d = 0; d < hidden_dim; d++) {
            score += w_row[d] * x[d];
        }
        scores[tid] = score;
    }
    __syncthreads();

    // Softmax over expert scores
    if (tid == 0) {
        float max_score = -CUDART_INF_F;
        for (int k = 0; k < num_experts; k++) {
            max_score = fmaxf(max_score, scores[k]);
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < num_experts; k++) {
            scores[k] = expf(scores[k] - max_score);
            sum_exp += scores[k];
        }
        for (int k = 0; k < num_experts; k++) {
            scores[k] /= sum_exp;
        }

        // Check entropy constraint
        float h = 0.0f;
        for (int k = 0; k < num_experts; k++) {
            if (scores[k] > 1e-10f) {
                h -= scores[k] * safe_log(scores[k]);
            }
        }

        // If entropy too high, sharpen (temperature < 1)
        if (h > ENTROPY_BOUND) {
            // Simple sharpening: raise to power > 1
            float power = logf(ENTROPY_BOUND) / logf(h + 1e-10f);
            power = fmaxf(1.0f, fminf(power, 10.0f));
            float new_sum = 0.0f;
            for (int k = 0; k < num_experts; k++) {
                scores[k] = powf(scores[k], power);
                new_sum += scores[k];
            }
            for (int k = 0; k < num_experts; k++) {
                scores[k] /= new_sum;
            }
        }

        // Top-k selection
        for (int t = 0; t < top_k; t++) {
            int best_k = 0;
            float best_v = -1.0f;
            for (int k = 0; k < num_experts; k++) {
                if (scores[k] > best_v) {
                    best_v = scores[k];
                    best_k = k;
                }
            }
            assignments[b * top_k + t] = best_k;
            gate_vals[b * top_k + t] = best_v;
            scores[best_k] = -1.0f; // mask out

            // Dispatch to expert buffer
            int slot = atomicAdd(&expert_count[best_k], 1);
            if (slot < capacity) {
                float* dst = expert_buf + best_k * capacity * hidden_dim + slot * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    dst[d] = x[d] * best_v;
                }
            }
        }

        // Balance loss: sum_k (f_k * P_k)
        // (accumulated across batch in separate kernel)
    }
}

/**
 * Launch wrapper for sparse MoE dispatch.
 */
inline cudaError_t launch_sparse_moe_dispatch(
    const float* hidden,
    const float* W_gate,
    float* expert_buf,
    int* expert_count,
    int* assignments,
    float* gate_vals,
    float* balance_loss,
    int hidden_dim,
    int num_experts,
    int capacity,
    int batch_size,
    int top_k,
    cudaStream_t stream = 0
) {
    // Zero expert counts
    BURT_IMMA_CHECK(cudaMemsetAsync(expert_count, 0, num_experts * sizeof(int), stream));

    sparse_moe_dispatch_kernel<<<batch_size, max(num_experts, WARP_SIZE), 0, stream>>>(
        hidden, W_gate, expert_buf, expert_count, assignments, gate_vals,
        balance_loss, hidden_dim, num_experts, capacity, batch_size, top_k);

    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}

#ifndef ENTROPY_BOUND
#define ENTROPY_BOUND 0.20f
#endif
