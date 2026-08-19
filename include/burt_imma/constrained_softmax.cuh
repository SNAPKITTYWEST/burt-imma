#pragma once
#include "common.cuh"

/**
 * Constrained Softmax Kernel
 *
 * Computes softmax with entropy constraint via temperature bisection:
 *   p_i = softmax(logits / tau)
 *   subject to: H(p) <= ENTROPY_BOUND (0.20)
 *
 * If unconstrained softmax has H(p) > bound, we bisect on temperature
 * tau to find the sharpening that satisfies the constraint.
 *
 * Bisection is O(log(1/epsilon)) iterations, bounded at 32 steps.
 */

#define ENTROPY_BOUND 0.20f
#define BISECTION_MAX_ITER 32
#define BISECTION_TOL 1e-6f

/**
 * Bisection kernel: finds temperature tau such that H(softmax(logits/tau)) <= ENTROPY_BOUND.
 * One thread block per row (sequence position).
 */
__global__ void constrained_softmax_bisection_kernel(
    const float* __restrict__ logits,    // [batch, seq_len]
    float*       __restrict__ output,    // [batch, seq_len]
    float*       __restrict__ tau_out,   // [batch] - computed temperatures
    float*       __restrict__ entropy_out,// [batch] - final entropies
    const int    seq_len,
    const int    batch_size
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const float* row = logits + b * seq_len;
    float* out_row = output + b * seq_len;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* s_probs = smem;                    // [seq_len]
    float* s_scratch = smem + seq_len;        // [blockDim.x]

    int tid = threadIdx.x;

    // Step 1: Find max for numerical stability
    float local_max = -CUDART_INF_F;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    local_max = block_reduce_max(local_max);
    __shared__ float row_max;
    if (tid == 0) row_max = local_max;
    __syncthreads();

    // Step 2: Try tau = 1.0 first (standard softmax)
    float tau_lo = 0.01f;
    float tau_hi = 10.0f;
    float tau = 1.0f;

    // Lambda: compute softmax and entropy for given tau
    auto compute_entropy = [&](float t) -> float {
        // Compute exp((logit - max) / t)
        float sum_exp = 0.0f;
        for (int i = tid; i < seq_len; i += blockDim.x) {
            float e = expf((row[i] - row_max) / t);
            s_probs[i] = e;
            sum_exp += e;
        }
        // Block reduce sum_exp
        s_scratch[tid] = sum_exp;
        __syncthreads();
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; i++) total += s_scratch[i];
        __syncthreads();

        // Normalize and compute entropy
        float h = 0.0f;
        for (int i = tid; i < seq_len; i += blockDim.x) {
            s_probs[i] /= total;
            if (s_probs[i] > 1e-10f) {
                h -= s_probs[i] * safe_log(s_probs[i]);
            }
        }
        s_scratch[tid] = h;
        __syncthreads();
        float total_h = 0.0f;
        for (int i = 0; i < blockDim.x; i++) total_h += s_scratch[i];
        __syncthreads();
        return total_h;
    };

    // Step 3: Check if constraint is already satisfied at tau=1
    float h = compute_entropy(1.0f);

    if (h <= ENTROPY_BOUND) {
        // Standard softmax satisfies constraint
        tau = 1.0f;
    } else {
        // Bisection on tau: lower tau -> sharper distribution -> lower entropy
        for (int iter = 0; iter < BISECTION_MAX_ITER; iter++) {
            tau = (tau_lo + tau_hi) / 2.0f;
            h = compute_entropy(tau);

            if (h > ENTROPY_BOUND) {
                tau_hi = tau;  // need sharper (lower tau)
            } else {
                tau_lo = tau;  // can afford softer (higher tau)
            }

            if (tau_hi - tau_lo < BISECTION_TOL) break;
        }
        // Final computation at converged tau
        h = compute_entropy(tau);
    }

    // Step 4: Write output
    for (int i = tid; i < seq_len; i += blockDim.x) {
        out_row[i] = s_probs[i];
    }

    if (tid == 0) {
        if (tau_out) tau_out[b] = tau;
        if (entropy_out) entropy_out[b] = h;
    }
}

/**
 * Validate that all rows satisfy entropy bound.
 */
__global__ void validate_entropy_kernel(
    const float* __restrict__ probs,     // [batch, seq_len]
    int*         __restrict__ violations, // [1] - count of violations
    const int    seq_len,
    const int    batch_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const float* row = probs + b * seq_len;
    float h = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        if (row[i] > 1e-10f) {
            h -= row[i] * safe_log(row[i]);
        }
    }

    if (h > ENTROPY_BOUND + 1e-5f) {
        atomicAdd(violations, 1);
    }
}

/**
 * Launch wrapper for constrained softmax.
 */
inline cudaError_t launch_constrained_softmax(
    const float* logits,
    float* output,
    float* tau_out,
    float* entropy_out,
    int seq_len,
    int batch_size,
    cudaStream_t stream = 0
) {
    int threads = min(seq_len, 256);
    size_t smem_size = (seq_len + threads) * sizeof(float);

    constrained_softmax_bisection_kernel<<<batch_size, threads, smem_size, stream>>>(
        logits, output, tau_out, entropy_out, seq_len, batch_size);

    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}
