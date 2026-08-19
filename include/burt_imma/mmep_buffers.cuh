#pragma once
#include "common.cuh"

/**
 * MMEPConfig — All hyperparameters for MMEP training.
 *
 * These correspond to the ablation config (config/ablation_arithmetic.yaml).
 */
struct MMEPConfig {
    // Architecture
    int hidden_dim;          // Hidden dimension per layer
    int num_layers;          // Number of equilibrium layers
    int num_experts;         // Number of MoE experts
    int batch_size;          // Training batch size

    // Relaxation phase
    float alpha_relax;       // Relaxation rate (free phase)
    int T_free;              // Free-phase relaxation steps
    int T_nudge;             // Nudged-phase relaxation steps

    // Nudging
    float beta;              // Nudge strength (small positive)

    // Constraint bounds
    float rho_ret;           // Memory retention L2 bound (C_global)
    float rho_inst;          // Instruction L2 bound (C_expert)
    float lambda_max;        // Spectral norm bound on W_l

    // Learning rates
    float lr_W;              // Learning rate for weight matrices
    float lr_C_global;       // Learning rate for global context
    float lr_C_expert;       // Learning rate for expert contexts

    // Retention/instruction balance
    float alpha_ret;         // Retention loss weight
    float alpha_inst;        // Instruction loss weight
};

/**
 * MMEPStateBuffers — GPU memory layout for MMEP computation.
 *
 * Holds both free-phase and nudged-phase copies of all state,
 * plus gradient accumulators and power-iteration buffers.
 */
struct MMEPStateBuffers {
    // Hidden states: [batch, num_layers, hidden_dim]
    float* H_free;           // Free-phase hidden states (current)
    float* H_free_next;      // Free-phase hidden states (next step, double-buffer)
    float* H_nudged;         // Nudged-phase hidden states

    // Context memories
    float* C_global;         // Global context memory [hidden_dim]
    float* C_expert;         // Expert-specific memories [num_experts, hidden_dim]

    // Weight matrices
    float* W;                // Layer weights [num_layers, hidden_dim, hidden_dim]

    // Expert assignments
    int* expert_assignments; // [batch] -> expert index

    // Gradient accumulators
    float* dW;               // Weight gradients [num_layers, hidden_dim, hidden_dim]
    float* dC_global;        // Global context gradient [hidden_dim]
    float* dC_expert;        // Expert context gradients [num_experts, hidden_dim]

    // Spectral norm buffers (power iteration)
    float* sigma_buffer;     // Running sigma estimates [num_layers]
    float* v_buffer;         // Power iteration vectors [num_layers, hidden_dim]

    // Retention / instruction targets
    float* alpha_ret_buffer; // Retention targets [batch, hidden_dim]
    float* alpha_inst_buffer;// Instruction targets [batch, hidden_dim]
};

/**
 * Allocate all MMEP state buffers on device.
 */
inline cudaError_t mmep_allocate_buffers(
    MMEPStateBuffers& state,
    const MMEPConfig& config
) {
    size_t H_size = (size_t)config.batch_size * config.num_layers * config.hidden_dim * sizeof(float);
    size_t W_size = (size_t)config.num_layers * config.hidden_dim * config.hidden_dim * sizeof(float);
    size_t C_global_size = (size_t)config.hidden_dim * sizeof(float);
    size_t C_expert_size = (size_t)config.num_experts * config.hidden_dim * sizeof(float);
    size_t sigma_size = (size_t)config.num_layers * sizeof(float);
    size_t v_size = (size_t)config.num_layers * config.hidden_dim * sizeof(float);
    size_t assign_size = (size_t)config.batch_size * sizeof(int);
    size_t ret_size = (size_t)config.batch_size * config.hidden_dim * sizeof(float);

    BURT_IMMA_CHECK(cudaMalloc(&state.H_free, H_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.H_free_next, H_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.H_nudged, H_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.C_global, C_global_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.C_expert, C_expert_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.W, W_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.expert_assignments, assign_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.dW, W_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.dC_global, C_global_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.dC_expert, C_expert_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.sigma_buffer, sigma_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.v_buffer, v_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.alpha_ret_buffer, ret_size));
    BURT_IMMA_CHECK(cudaMalloc(&state.alpha_inst_buffer, ret_size));

    // Initialize power iteration vectors to random (caller should seed)
    BURT_IMMA_CHECK(cudaMemset(state.dW, 0, W_size));
    BURT_IMMA_CHECK(cudaMemset(state.dC_global, 0, C_global_size));
    BURT_IMMA_CHECK(cudaMemset(state.dC_expert, 0, C_expert_size));

    return cudaSuccess;
}

/**
 * Free all MMEP state buffers.
 */
inline cudaError_t mmep_free_buffers(MMEPStateBuffers& state) {
    cudaFree(state.H_free);
    cudaFree(state.H_free_next);
    cudaFree(state.H_nudged);
    cudaFree(state.C_global);
    cudaFree(state.C_expert);
    cudaFree(state.W);
    cudaFree(state.expert_assignments);
    cudaFree(state.dW);
    cudaFree(state.dC_global);
    cudaFree(state.dC_expert);
    cudaFree(state.sigma_buffer);
    cudaFree(state.v_buffer);
    cudaFree(state.alpha_ret_buffer);
    cudaFree(state.alpha_inst_buffer);
    return cudaSuccess;
}
