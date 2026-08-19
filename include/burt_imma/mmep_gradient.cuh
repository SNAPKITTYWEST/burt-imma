#pragma once
#include "common.cuh"
#include "mmep_buffers.cuh"

/**
 * MMEP Gradient Accumulation Kernel
 *
 * Computes equilibrium propagation gradients:
 *   dW_l = (1/beta) * (h_l^nudged @ h_{l-1}^nudged^T - h_l^free @ h_{l-1}^free^T)
 *   dC_global = (1/beta) * sum_l(h_l^nudged - h_l^free)
 *   dC_expert_k = (1/beta) * sum_{l, b in expert_k}(h_l^nudged - h_l^free)
 *
 * Two-phase EP: free phase finds equilibrium, nudged phase perturbs toward target.
 * Gradient is the difference in correlations between phases.
 */
__global__ void mmep_accumulate_gradients_kernel(
    const float* __restrict__ H_free,       // [batch, layers, hidden_dim]
    const float* __restrict__ H_nudged,     // [batch, layers, hidden_dim]
    float*       __restrict__ dW,           // [layers, hidden_dim, hidden_dim]
    float*       __restrict__ dC_global,    // [hidden_dim]
    float*       __restrict__ dC_expert,    // [num_experts, hidden_dim]
    const int*   __restrict__ expert_assign,// [batch]
    const float  inv_beta,                  // 1.0 / beta
    const int    hidden_dim,
    const int    num_layers,
    const int    batch_size
) {
    const int l = blockIdx.x;              // layer
    const int d_out = blockIdx.y;          // output neuron
    const int d_in = threadIdx.x;          // input neuron (within warp)

    if (l >= num_layers || d_out >= hidden_dim) return;

    float dw_accum = 0.0f;
    float dc_global_accum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        const int idx_l = b * num_layers * hidden_dim + l * hidden_dim;
        const int idx_prev = (l > 0) ?
            b * num_layers * hidden_dim + (l - 1) * hidden_dim : idx_l;

        float h_free_l = H_free[idx_l + d_out];
        float h_nudged_l = H_nudged[idx_l + d_out];
        float h_free_prev = H_free[idx_prev + d_in];
        float h_nudged_prev = H_nudged[idx_prev + d_in];

        // Hebbian correlation difference
        dw_accum += inv_beta * (h_nudged_l * h_nudged_prev - h_free_l * h_free_prev);

        // Context memory gradient (only accumulate for first d_in thread)
        if (d_in == 0) {
            float diff = inv_beta * (h_nudged_l - h_free_l);
            dc_global_accum += diff;

            // Expert-specific gradient
            int expert_k = expert_assign[b];
            atomicAdd(&dC_expert[expert_k * hidden_dim + d_out], diff);
        }
    }

    // Write weight gradient
    const int w_idx = l * hidden_dim * hidden_dim + d_out * hidden_dim + d_in;
    atomicAdd(&dW[w_idx], dw_accum);

    // Write global context gradient
    if (d_in == 0) {
        atomicAdd(&dC_global[d_out], dc_global_accum);
    }
}

/**
 * Launch wrapper for mmep_accumulate_gradients_kernel.
 */
inline cudaError_t launch_mmep_accumulate_gradients(
    const MMEPStateBuffers& state,
    float inv_beta,
    int hidden_dim,
    int num_layers,
    int batch_size,
    cudaStream_t stream = 0
) {
    dim3 block(WARP_SIZE);
    dim3 grid(num_layers, hidden_dim);

    mmep_accumulate_gradients_kernel<<<grid, block, 0, stream>>>(
        state.H_free,
        state.H_nudged,
        state.dW,
        state.dC_global,
        state.dC_expert,
        state.expert_assignments,
        inv_beta,
        hidden_dim,
        num_layers,
        batch_size
    );

    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}
