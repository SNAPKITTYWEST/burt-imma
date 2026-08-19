#pragma once
#include "common.cuh"
#include "mmep_buffers.cuh"

/**
 * MMEP Relaxation Step Kernel
 *
 * Implements the free-phase energy relaxation:
 *   h_l^{t+1} = (1 - alpha) * h_l^t + alpha * sigma(W_l @ h_{l-1}^t + C_global + C_expert_k)
 *
 * Each thread handles one neuron in one layer.
 * Block dimension: (WARP_SIZE, layers_per_block)
 * Grid dimension: (batch_size, num_layers / layers_per_block)
 */
__global__ void mmep_relaxation_step_kernel(
    const float* __restrict__ H_prev,       // [batch, layers, hidden_dim]
    float*       __restrict__ H_next,       // [batch, layers, hidden_dim]
    const float* __restrict__ W,            // [layers, hidden_dim, hidden_dim]
    const float* __restrict__ C_global,     // [hidden_dim]
    const float* __restrict__ C_expert,     // [num_experts, hidden_dim]
    const int*   __restrict__ expert_assign,// [batch] -> expert index
    const float  alpha_relax,               // relaxation rate
    const int    hidden_dim,
    const int    num_layers,
    const int    batch_size
) {
    const int b = blockIdx.x;                           // batch index
    const int l = blockIdx.y * blockDim.y + threadIdx.y;// layer index
    const int d = threadIdx.x;                          // hidden dim (warp lane)

    if (b >= batch_size || l >= num_layers) return;

    const int h_idx = b * num_layers * hidden_dim + l * hidden_dim + d;
    const int expert_k = expert_assign[b];

    // Compute W_l @ h_{l-1} for this neuron
    float activation = 0.0f;
    if (l > 0) {
        const int prev_base = b * num_layers * hidden_dim + (l - 1) * hidden_dim;
        const int w_base = l * hidden_dim * hidden_dim + d * hidden_dim;
        for (int k = 0; k < hidden_dim; k += WARP_SIZE) {
            int kk = k + threadIdx.x;
            if (kk < hidden_dim) {
                // Each warp lane accumulates partial dot product
                activation += W[w_base + kk] * H_prev[prev_base + kk];
            }
        }
        // Warp-level reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            activation += __shfl_down_sync(0xffffffff, activation, offset);
        }
    } else {
        // Layer 0: use input directly (H_prev stores input in layer 0 slot)
        activation = H_prev[h_idx];
    }

    // Add context memories
    activation += C_global[d];
    activation += C_expert[expert_k * hidden_dim + d];

    // Apply nonlinearity (hardtanh for bounded energy)
    activation = fmaxf(-1.0f, fminf(1.0f, activation));

    // Relaxation update
    float h_old = H_prev[h_idx];
    H_next[h_idx] = (1.0f - alpha_relax) * h_old + alpha_relax * activation;
}

/**
 * Launch wrapper for mmep_relaxation_step_kernel.
 * Handles grid/block computation and stream assignment.
 */
inline cudaError_t launch_mmep_relaxation_step(
    const MMEPStateBuffers& state,
    const float* W,
    float alpha_relax,
    int hidden_dim,
    int num_layers,
    int batch_size,
    cudaStream_t stream = 0
) {
    const int layers_per_block = 4;
    dim3 block(WARP_SIZE, layers_per_block);
    dim3 grid(batch_size, (num_layers + layers_per_block - 1) / layers_per_block);

    mmep_relaxation_step_kernel<<<grid, block, 0, stream>>>(
        state.H_free,
        state.H_free_next,
        W,
        state.C_global,
        state.C_expert,
        state.expert_assignments,
        alpha_relax,
        hidden_dim,
        num_layers,
        batch_size
    );

    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}
