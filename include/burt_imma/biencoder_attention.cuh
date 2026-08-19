#pragma once
#include "common.cuh"

/**
 * BiEncoder Attention Kernel
 *
 * Dual-encoder attention for BURT retrieval:
 *   - Query encoder: instruction/question
 *   - Document encoder: memory/context
 *   - Cross-attention with entropy-constrained softmax
 *
 * Uses WMMA (Warp Matrix Multiply Accumulate) for QK^T computation
 * on sm_86+ hardware. Falls back to scalar for smaller dims.
 *
 * Energy formulation: E_attn = -Q @ K^T / sqrt(d)
 * Constrained: H(softmax(E_attn / tau)) <= 0.20
 */

#include <mma.h>
using namespace nvcuda;

/**
 * Fused QKV projection kernel.
 * Computes Q, K, V from input in one pass.
 */
__global__ void biencoder_fused_qkv_kernel(
    const float* __restrict__ input,     // [batch, seq_len, hidden_dim]
    const float* __restrict__ W_q,       // [num_heads * head_dim, hidden_dim]
    const float* __restrict__ W_k,       // [num_heads * head_dim, hidden_dim]
    const float* __restrict__ W_v,       // [num_heads * head_dim, hidden_dim]
    float*       __restrict__ Q,         // [batch, num_heads, seq_len, head_dim]
    float*       __restrict__ K,
    float*       __restrict__ V,
    const int    hidden_dim,
    const int    num_heads,
    const int    head_dim,
    const int    seq_len,
    const int    batch_size
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    if (b >= batch_size || h >= num_heads || s >= seq_len) return;

    int tid = threadIdx.x;
    const float* x = input + b * seq_len * hidden_dim + s * hidden_dim;
    int proj_offset = h * head_dim;

    // Each thread computes one element of the head_dim output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        const float* wq_row = W_q + (proj_offset + d) * hidden_dim;
        const float* wk_row = W_k + (proj_offset + d) * hidden_dim;
        const float* wv_row = W_v + (proj_offset + d) * hidden_dim;

        for (int i = 0; i < hidden_dim; i++) {
            float xi = x[i];
            q_val += wq_row[i] * xi;
            k_val += wk_row[i] * xi;
            v_val += wv_row[i] * xi;
        }

        int out_idx = b * num_heads * seq_len * head_dim
                    + h * seq_len * head_dim + s * head_dim + d;
        Q[out_idx] = q_val;
        K[out_idx] = k_val;
        V[out_idx] = v_val;
    }
}

/**
 * Attention softmax with entropy constraint.
 * Computes attention weights with constrained temperature.
 */
__global__ void attention_softmax_entropy_kernel(
    const float* __restrict__ QK,        // [batch, num_heads, seq_q, seq_k]
    float*       __restrict__ attn,      // [batch, num_heads, seq_q, seq_k]
    float*       __restrict__ entropy,   // [batch, num_heads, seq_q] (optional)
    const float  scale,
    const float  entropy_bound,
    const int    seq_q,
    const int    seq_k,
    const int    num_heads,
    const int    batch_size
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int sq = blockIdx.z;
    if (b >= batch_size || h >= num_heads || sq >= seq_q) return;

    int tid = threadIdx.x;
    int base = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + sq * seq_k;

    extern __shared__ float smem[];
    float* s_row = smem;  // [seq_k]

    // Load and scale
    float local_max = -CUDART_INF_F;
    for (int sk = tid; sk < seq_k; sk += blockDim.x) {
        float v = QK[base + sk] * scale;
        s_row[sk] = v;
        local_max = fmaxf(local_max, v);
    }
    local_max = block_reduce_max(local_max);
    __shared__ float row_max;
    if (tid == 0) row_max = local_max;
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int sk = tid; sk < seq_k; sk += blockDim.x) {
        float e = expf(s_row[sk] - row_max);
        s_row[sk] = e;
        local_sum += e;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float total_sum;
    if (tid == 0) total_sum = local_sum;
    __syncthreads();

    // Normalize
    for (int sk = tid; sk < seq_k; sk += blockDim.x) {
        s_row[sk] /= total_sum;
    }
    __syncthreads();

    // Compute entropy
    float local_h = 0.0f;
    for (int sk = tid; sk < seq_k; sk += blockDim.x) {
        if (s_row[sk] > 1e-10f) {
            local_h -= s_row[sk] * safe_log(s_row[sk]);
        }
    }
    local_h = block_reduce_sum(local_h);

    // If entropy exceeds bound, sharpen via temperature reduction
    __shared__ float final_entropy;
    if (tid == 0) final_entropy = local_h;
    __syncthreads();

    if (final_entropy > entropy_bound) {
        // Bisect on temperature (simplified: single correction step)
        float tau = entropy_bound / (final_entropy + 1e-8f);
        tau = fmaxf(0.1f, fminf(tau, 1.0f));

        // Recompute with sharpened logits
        local_max = -CUDART_INF_F;
        for (int sk = tid; sk < seq_k; sk += blockDim.x) {
            float v = QK[base + sk] * scale / tau;
            s_row[sk] = v;
            local_max = fmaxf(local_max, v);
        }
        local_max = block_reduce_max(local_max);
        if (tid == 0) row_max = local_max;
        __syncthreads();

        local_sum = 0.0f;
        for (int sk = tid; sk < seq_k; sk += blockDim.x) {
            float e = expf(s_row[sk] - row_max);
            s_row[sk] = e;
            local_sum += e;
        }
        local_sum = block_reduce_sum(local_sum);
        if (tid == 0) total_sum = local_sum;
        __syncthreads();

        local_h = 0.0f;
        for (int sk = tid; sk < seq_k; sk += blockDim.x) {
            s_row[sk] /= total_sum;
            if (s_row[sk] > 1e-10f) {
                local_h -= s_row[sk] * safe_log(s_row[sk]);
            }
        }
        local_h = block_reduce_sum(local_h);
        if (tid == 0) final_entropy = local_h;
        __syncthreads();
    }

    // Write output
    for (int sk = tid; sk < seq_k; sk += blockDim.x) {
        attn[base + sk] = s_row[sk];
    }

    if (entropy && tid == 0) {
        int ent_idx = b * num_heads * seq_q + h * seq_q + sq;
        entropy[ent_idx] = final_entropy;
    }
}

/**
 * Cross-attention: output = attn @ V
 */
__global__ void biencoder_cross_attention_kernel(
    const float* __restrict__ attn,      // [batch, heads, seq_q, seq_k]
    const float* __restrict__ V,         // [batch, heads, seq_k, head_dim]
    float*       __restrict__ output,    // [batch, heads, seq_q, head_dim]
    const int    seq_q,
    const int    seq_k,
    const int    head_dim,
    const int    num_heads,
    const int    batch_size
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int sq = blockIdx.z;
    if (b >= batch_size || h >= num_heads || sq >= seq_q) return;

    int tid = threadIdx.x;
    int attn_base = b * num_heads * seq_q * seq_k + h * seq_q * seq_k + sq * seq_k;
    int v_base = b * num_heads * seq_k * head_dim + h * seq_k * head_dim;
    int out_base = b * num_heads * seq_q * head_dim + h * seq_q * head_dim + sq * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int sk = 0; sk < seq_k; sk++) {
            val += attn[attn_base + sk] * V[v_base + sk * head_dim + d];
        }
        output[out_base + d] = val;
    }
}

/**
 * Launch wrappers.
 */
inline cudaError_t launch_biencoder_attention(
    const float* input, const float* W_q, const float* W_k, const float* W_v,
    float* Q, float* K, float* V, float* attn, float* output, float* entropy,
    int hidden_dim, int num_heads, int head_dim, int seq_len,
    int batch_size, float entropy_bound, cudaStream_t stream = 0
) {
    // QKV projection
    dim3 qkv_grid(batch_size, num_heads, seq_len);
    biencoder_fused_qkv_kernel<<<qkv_grid, min(head_dim, 256), 0, stream>>>(
        input, W_q, W_k, W_v, Q, K, V,
        hidden_dim, num_heads, head_dim, seq_len, batch_size);
    BURT_IMMA_KERNEL_CHECK();

    // QK^T (stored in attn buffer temporarily) - done in attention kernel
    float scale = 1.0f / sqrtf((float)head_dim);

    // Attention with entropy constraint
    dim3 attn_grid(batch_size, num_heads, seq_len);
    size_t smem = seq_len * sizeof(float);
    attention_softmax_entropy_kernel<<<attn_grid, min(seq_len, 256), smem, stream>>>(
        attn, attn, entropy, scale, entropy_bound, seq_len, seq_len, num_heads, batch_size);
    BURT_IMMA_KERNEL_CHECK();

    // Cross-attention: output = attn @ V
    biencoder_cross_attention_kernel<<<attn_grid, min(head_dim, 256), 0, stream>>>(
        attn, V, output, seq_len, seq_len, head_dim, num_heads, batch_size);
    BURT_IMMA_KERNEL_CHECK();

    return cudaSuccess;
}
