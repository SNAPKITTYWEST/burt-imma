#include "burt_imma/common.cuh"
#include "burt_imma/mmep_buffers.cuh"
#include "burt_imma/mmep_relaxation.cuh"
#include "burt_imma/mmep_gradient.cuh"
#include "burt_imma/mmep_project.cuh"

#include <cstdio>
#include <cmath>

/**
 * MMETrainer — Host-side orchestrator for MMEP training steps.
 *
 * Implements the full MMEP training loop:
 *   1. Free phase: relax to equilibrium (T_free steps)
 *   2. Nudged phase: perturb toward target (T_nudge steps)
 *   3. Gradient accumulation: EP gradient from correlation difference
 *   4. Parameter update: SGD with constraint projection
 *
 * Target hardware: NVIDIA RTX 3080 (sm_86, 10GB VRAM)
 */
struct MMETrainer {
    MMEPConfig config;
    MMEPStateBuffers state;
    cudaStream_t stream_free;
    cudaStream_t stream_nudge;
    cudaStream_t stream_grad;

    int step_count;
    float running_loss;

    /**
     * Initialize trainer with config. Allocates GPU memory.
     */
    cudaError_t init(const MMEPConfig& cfg) {
        config = cfg;
        step_count = 0;
        running_loss = 0.0f;

        BURT_IMMA_CHECK(cudaStreamCreate(&stream_free));
        BURT_IMMA_CHECK(cudaStreamCreate(&stream_nudge));
        BURT_IMMA_CHECK(cudaStreamCreate(&stream_grad));

        BURT_IMMA_CHECK(mmep_allocate_buffers(state, config));

        // Initialize power iteration vectors to uniform
        size_t v_size = (size_t)config.num_layers * config.hidden_dim * sizeof(float);
        float* v_host = new float[config.num_layers * config.hidden_dim];
        float inv_sqrt_d = 1.0f / sqrtf((float)config.hidden_dim);
        for (int i = 0; i < config.num_layers * config.hidden_dim; i++) {
            v_host[i] = inv_sqrt_d;
        }
        BURT_IMMA_CHECK(cudaMemcpy(state.v_buffer, v_host, v_size, cudaMemcpyHostToDevice));
        delete[] v_host;

        printf("[MMETrainer] Initialized: %d layers, %d hidden, %d experts, batch %d\n",
               config.num_layers, config.hidden_dim, config.num_experts, config.batch_size);
        printf("[MMETrainer] VRAM estimate: %.1f MB\n", estimate_vram_mb());

        return cudaSuccess;
    }

    /**
     * Estimate VRAM usage in MB.
     */
    float estimate_vram_mb() const {
        size_t total = 0;
        size_t H_size = (size_t)config.batch_size * config.num_layers * config.hidden_dim * sizeof(float);
        size_t W_size = (size_t)config.num_layers * config.hidden_dim * config.hidden_dim * sizeof(float);

        total += H_size * 3;  // H_free, H_free_next, H_nudged
        total += W_size * 2;  // W, dW
        total += (size_t)config.hidden_dim * sizeof(float) * 2;  // C_global, dC_global
        total += (size_t)config.num_experts * config.hidden_dim * sizeof(float) * 2; // C_expert, dC_expert
        total += (size_t)config.num_layers * sizeof(float);  // sigma_buffer
        total += (size_t)config.num_layers * config.hidden_dim * sizeof(float); // v_buffer
        total += (size_t)config.batch_size * config.hidden_dim * sizeof(float) * 2; // ret/inst buffers
        total += (size_t)config.batch_size * sizeof(int);  // expert_assignments

        return (float)total / (1024.0f * 1024.0f);
    }

    /**
     * Execute one full MMEP training step.
     *
     * Returns the loss for this step (MSE between output and target).
     */
    cudaError_t mmep_step(
        const float* input,      // [batch, hidden_dim] on device
        const float* target,     // [batch, hidden_dim] on device
        float* loss_out          // scalar output
    ) {
        // Zero gradient accumulators
        size_t W_size = (size_t)config.num_layers * config.hidden_dim * config.hidden_dim * sizeof(float);
        size_t C_global_size = (size_t)config.hidden_dim * sizeof(float);
        size_t C_expert_size = (size_t)config.num_experts * config.hidden_dim * sizeof(float);

        BURT_IMMA_CHECK(cudaMemsetAsync(state.dW, 0, W_size, stream_grad));
        BURT_IMMA_CHECK(cudaMemsetAsync(state.dC_global, 0, C_global_size, stream_grad));
        BURT_IMMA_CHECK(cudaMemsetAsync(state.dC_expert, 0, C_expert_size, stream_grad));
        BURT_IMMA_CHECK(cudaStreamSynchronize(stream_grad));

        // Initialize H_free layer 0 with input
        size_t input_size = (size_t)config.batch_size * config.hidden_dim * sizeof(float);
        BURT_IMMA_CHECK(cudaMemcpyAsync(
            state.H_free, input, input_size, cudaMemcpyDeviceToDevice, stream_free));

        // =====================================================================
        // FREE PHASE: Relax to equilibrium
        // =====================================================================
        for (int t = 0; t < config.T_free; t++) {
            BURT_IMMA_CHECK(launch_mmep_relaxation_step(
                state, state.W, config.alpha_relax,
                config.hidden_dim, config.num_layers, config.batch_size,
                stream_free));

            // Double-buffer swap
            float* tmp = state.H_free;
            state.H_free = state.H_free_next;
            state.H_free_next = tmp;
        }
        BURT_IMMA_CHECK(cudaStreamSynchronize(stream_free));

        // =====================================================================
        // NUDGED PHASE: Perturb toward target
        // =====================================================================
        // Copy free-phase equilibrium as starting point for nudged phase
        size_t H_size = (size_t)config.batch_size * config.num_layers * config.hidden_dim * sizeof(float);
        BURT_IMMA_CHECK(cudaMemcpyAsync(
            state.H_nudged, state.H_free, H_size, cudaMemcpyDeviceToDevice, stream_nudge));

        // Add nudge to output layer: h_L += beta * (target - h_L)
        // This is done inline via a simple kernel
        {
            int output_offset = (config.num_layers - 1) * config.hidden_dim;
            int total_output = config.batch_size * config.hidden_dim;
            // Inline nudge application
            auto nudge_kernel = [&]() __device__ {};  // placeholder, actual below
        }
        // Launch nudge application (separate small kernel)
        apply_nudge<<<(config.batch_size * config.hidden_dim + 255) / 256, 256, 0, stream_nudge>>>(
            state.H_nudged, target,
            config.beta, config.hidden_dim, config.num_layers, config.batch_size);
        BURT_IMMA_KERNEL_CHECK();

        // Relax with nudge active
        for (int t = 0; t < config.T_nudge; t++) {
            // Reuse relaxation kernel but on nudged state
            // Temporarily swap pointers for the kernel
            float* save_free = state.H_free;
            float* save_next = state.H_free_next;
            state.H_free = state.H_nudged;
            state.H_free_next = state.H_nudged;  // in-place for nudged

            BURT_IMMA_CHECK(launch_mmep_relaxation_step(
                state, state.W, config.alpha_relax,
                config.hidden_dim, config.num_layers, config.batch_size,
                stream_nudge));

            state.H_free = save_free;
            state.H_free_next = save_next;

            // Re-apply nudge each step to maintain target pressure
            apply_nudge<<<(config.batch_size * config.hidden_dim + 255) / 256, 256, 0, stream_nudge>>>(
                state.H_nudged, target,
                config.beta, config.hidden_dim, config.num_layers, config.batch_size);
            BURT_IMMA_KERNEL_CHECK();
        }
        BURT_IMMA_CHECK(cudaStreamSynchronize(stream_nudge));

        // =====================================================================
        // GRADIENT ACCUMULATION
        // =====================================================================
        float inv_beta = 1.0f / config.beta;
        BURT_IMMA_CHECK(launch_mmep_accumulate_gradients(
            state, inv_beta, config.hidden_dim, config.num_layers, config.batch_size,
            stream_grad));
        BURT_IMMA_CHECK(cudaStreamSynchronize(stream_grad));

        // =====================================================================
        // PARAMETER UPDATE (SGD)
        // =====================================================================
        apply_sgd_update<<<(config.num_layers * config.hidden_dim * config.hidden_dim + 255) / 256, 256>>>(
            state.W, state.dW, config.lr_W,
            config.num_layers * config.hidden_dim * config.hidden_dim);
        BURT_IMMA_KERNEL_CHECK();

        apply_sgd_update<<<(config.hidden_dim + 255) / 256, 256>>>(
            state.C_global, state.dC_global, config.lr_C_global,
            config.hidden_dim);
        BURT_IMMA_KERNEL_CHECK();

        apply_sgd_update<<<(config.num_experts * config.hidden_dim + 255) / 256, 256>>>(
            state.C_expert, state.dC_expert, config.lr_C_expert,
            config.num_experts * config.hidden_dim);
        BURT_IMMA_KERNEL_CHECK();

        // =====================================================================
        // CONSTRAINT PROJECTION
        // =====================================================================
        BURT_IMMA_CHECK(launch_mmep_project_constraints(state, config));

        // =====================================================================
        // COMPUTE LOSS (MSE at output layer)
        // =====================================================================
        float loss = 0.0f;
        compute_output_mse<<<1, 256>>>(
            state.H_free, target,
            config.hidden_dim, config.num_layers, config.batch_size,
            state.sigma_buffer);  // reuse sigma_buffer[0] for loss
        BURT_IMMA_KERNEL_CHECK();
        BURT_IMMA_CHECK(cudaMemcpy(&loss, state.sigma_buffer, sizeof(float), cudaMemcpyDeviceToHost));

        *loss_out = loss;
        running_loss = 0.99f * running_loss + 0.01f * loss;
        step_count++;

        return cudaSuccess;
    }

    /**
     * Cleanup all resources.
     */
    cudaError_t destroy() {
        BURT_IMMA_CHECK(mmep_free_buffers(state));
        cudaStreamDestroy(stream_free);
        cudaStreamDestroy(stream_nudge);
        cudaStreamDestroy(stream_grad);
        return cudaSuccess;
    }
};

// =============================================================================
// HELPER KERNELS
// =============================================================================

/**
 * Apply nudge: h_L += beta * (target - h_L) for output layer only.
 */
__global__ void apply_nudge(
    float* __restrict__ H,
    const float* __restrict__ target,
    float beta,
    int hidden_dim,
    int num_layers,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_dim) return;

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;
    int h_idx = b * num_layers * hidden_dim + (num_layers - 1) * hidden_dim + d;
    int t_idx = b * hidden_dim + d;

    H[h_idx] += beta * (target[t_idx] - H[h_idx]);
}

/**
 * SGD update: param -= lr * grad
 */
__global__ void apply_sgd_update(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    params[idx] -= lr * grads[idx];
}

/**
 * Compute MSE loss at output layer.
 */
__global__ void compute_output_mse(
    const float* __restrict__ H,
    const float* __restrict__ target,
    int hidden_dim,
    int num_layers,
    int batch_size,
    float* __restrict__ loss_out
) {
    __shared__ float partial[256];
    int tid = threadIdx.x;
    partial[tid] = 0.0f;

    int total = batch_size * hidden_dim;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int b = idx / hidden_dim;
        int d = idx % hidden_dim;
        int h_idx = b * num_layers * hidden_dim + (num_layers - 1) * hidden_dim + d;
        float diff = H[h_idx] - target[idx];
        partial[tid] += diff * diff;
    }
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        loss_out[0] = partial[0] / (float)total;
    }
}
