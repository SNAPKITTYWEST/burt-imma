#pragma once
#include "common.cuh"
#include "mmep_buffers.cuh"

/**
 * MMEP Constraint Projection Kernel
 *
 * Projects parameters back onto constraint manifold after gradient step:
 *   1. Memory retention constraint: ||C_global||_F <= rho_ret
 *   2. Instruction constraint: ||C_expert_k||_F <= rho_inst for all k
 *   3. Spectral norm constraint: sigma_max(W_l) <= lambda_max for all l
 *
 * The spectral norm is approximated via power iteration (1 step per call,
 * converges over training iterations). Full SVD is too expensive per step.
 */
__global__ void mmep_project_constraints_kernel(
    float*       __restrict__ C_global,     // [hidden_dim]
    float*       __restrict__ C_expert,     // [num_experts, hidden_dim]
    float*       __restrict__ W,            // [layers, hidden_dim, hidden_dim]
    float*       __restrict__ sigma_buffer, // [layers] - running spectral norm estimates
    float*       __restrict__ v_buffer,     // [layers, hidden_dim] - power iteration vectors
    const float  rho_ret,                   // memory retention bound
    const float  rho_inst,                  // instruction bound
    const float  lambda_max,               // spectral norm bound
    const int    hidden_dim,
    const int    num_layers,
    const int    num_experts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Phase 1: Project C_global onto L2 ball of radius rho_ret ---
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            norm_sq += C_global[d] * C_global[d];
        }
        float norm = sqrtf(norm_sq);
        if (norm > rho_ret) {
            float scale = rho_ret / norm;
            for (int d = 0; d < hidden_dim; d++) {
                C_global[d] *= scale;
            }
        }
    }

    // --- Phase 2: Project each C_expert_k onto L2 ball of radius rho_inst ---
    if (tid < num_experts) {
        int k = tid;
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            float val = C_expert[k * hidden_dim + d];
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);
        if (norm > rho_inst) {
            float scale = rho_inst / norm;
            for (int d = 0; d < hidden_dim; d++) {
                C_expert[k * hidden_dim + d] *= scale;
            }
        }
    }

    // --- Phase 3: One step of power iteration for spectral norm of W_l ---
    if (tid < num_layers) {
        int l = tid;
        float* v = v_buffer + l * hidden_dim;
        float* Wl = W + l * hidden_dim * hidden_dim;

        // u = W_l @ v
        float u[1024]; // max hidden_dim (stack allocated, safe for sm_86)
        float u_norm = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            u[i] = 0.0f;
            for (int j = 0; j < hidden_dim; j++) {
                u[i] += Wl[i * hidden_dim + j] * v[j];
            }
            u_norm += u[i] * u[i];
        }
        u_norm = sqrtf(u_norm);

        // v = W_l^T @ u (normalized)
        float v_norm = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            float vj = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                vj += Wl[i * hidden_dim + j] * (u[i] / (u_norm + 1e-8f));
            }
            v[j] = vj;
            v_norm += vj * vj;
        }
        v_norm = sqrtf(v_norm);
        for (int j = 0; j < hidden_dim; j++) {
            v[j] /= (v_norm + 1e-8f);
        }

        // sigma estimate = u_norm (after normalization of v)
        sigma_buffer[l] = u_norm;

        // Project: if sigma > lambda_max, scale W_l
        if (u_norm > lambda_max) {
            float scale = lambda_max / u_norm;
            for (int i = 0; i < hidden_dim * hidden_dim; i++) {
                Wl[i] *= scale;
            }
        }
    }
}

/**
 * Launch wrapper for mmep_project_constraints_kernel.
 */
inline cudaError_t launch_mmep_project_constraints(
    MMEPStateBuffers& state,
    const MMEPConfig& config,
    cudaStream_t stream = 0
) {
    int max_threads = max(config.num_experts, config.num_layers);
    max_threads = max(max_threads, 1);
    dim3 block(min(max_threads, 256));
    dim3 grid((max_threads + block.x - 1) / block.x);

    mmep_project_constraints_kernel<<<grid, block, 0, stream>>>(
        state.C_global,
        state.C_expert,
        state.W,
        state.sigma_buffer,
        state.v_buffer,
        config.rho_ret,
        config.rho_inst,
        config.lambda_max,
        config.hidden_dim,
        config.num_layers,
        config.num_experts
    );

    BURT_IMMA_KERNEL_CHECK();
    return cudaSuccess;
}
