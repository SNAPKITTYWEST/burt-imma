// BURT-IMMA Python Bindings
// License: BSL-1.1
// Contact: jessica@collectivekitty.com

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;


// ---------------------------------------------------------------------------
// Kernel declarations (implementations in CUDA source files)
// ---------------------------------------------------------------------------

// Constrained softmax with spectral norm bound on attention weights.
// Applies softmax along last dimension with constraint that resulting
// distribution has entropy <= max_entropy.
// Args:
//   logits: (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
//   max_entropy: maximum allowed entropy (default 0.20)
//   temperature: initial temperature for sharpening
// Returns:
//   Tensor of same shape with constrained softmax applied
torch::Tensor constrained_softmax(
    torch::Tensor logits,
    double max_entropy,
    double temperature
) {
    // Placeholder: calls CUDA kernel when available
    // For now, compute on CPU/GPU via PyTorch ops
    auto scaled = logits / temperature;
    auto probs = torch::softmax(scaled, -1);

    // Iteratively sharpen if entropy exceeds bound
    auto entropy = -(probs * (probs + 1e-10).log()).sum(-1);
    auto violations = entropy > max_entropy;

    if (violations.any().item<bool>()) {
        double temp = temperature;
        for (int i = 0; i < 50; i++) {
            temp *= 0.8;
            auto p = torch::softmax(logits / temp, -1);
            auto h = -(p * (p + 1e-10).log()).sum(-1);
            if ((h <= max_entropy).all().item<bool>()) {
                probs = p;
                break;
            }
            probs = p;
        }
    }

    return probs;
}


// Coupled Input-Forget Gate memory cell update.
// C_new = f * C_old + (1-f) * candidate
// where f is computed from the forget gate weight and input.
// Args:
//   C_old: (batch, d_mem, d_mem) - current memory matrix
//   key: (batch, d_mem) - key vector for write
//   value: (batch, d_mem) - value vector for write
//   forget_bias: scalar bias for forget gate
// Returns:
//   C_new: (batch, d_mem, d_mem) - updated memory matrix
torch::Tensor cifg_update(
    torch::Tensor C_old,
    torch::Tensor key,
    torch::Tensor value,
    double forget_bias
) {
    // Compute forget gate from key norm
    auto f = torch::sigmoid(key.norm(-1, true) + forget_bias);  // (batch, 1)

    // Outer product candidate
    auto candidate = torch::bmm(
        key.unsqueeze(-1),      // (batch, d_mem, 1)
        value.unsqueeze(-2)     // (batch, 1, d_mem)
    );  // (batch, d_mem, d_mem)

    // Normalize candidate
    auto cand_norm = candidate.norm() + 1e-8;
    candidate = candidate / cand_norm;

    // CIFG update: C_new = f * C_old + (1-f) * candidate
    auto f_expanded = f.unsqueeze(-1);  // (batch, 1, 1)
    auto C_new = f_expanded * C_old + (1.0 - f_expanded) * candidate;

    return C_new;
}


// Batched CIFG update for multiple memory slots.
// Args:
//   C_old: (batch, num_slots, d_mem, d_mem)
//   keys: (batch, num_slots, d_mem)
//   values: (batch, num_slots, d_mem)
//   forget_biases: (num_slots,)
// Returns:
//   C_new: (batch, num_slots, d_mem, d_mem)
torch::Tensor batched_cifg_update(
    torch::Tensor C_old,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor forget_biases
) {
    auto batch = C_old.size(0);
    auto num_slots = C_old.size(1);
    auto d_mem = C_old.size(2);

    auto C_new = torch::zeros_like(C_old);

    for (int64_t s = 0; s < num_slots; s++) {
        auto C_slot = C_old.select(1, s);           // (batch, d_mem, d_mem)
        auto k_slot = keys.select(1, s);            // (batch, d_mem)
        auto v_slot = values.select(1, s);          // (batch, d_mem)
        auto bias = forget_biases[s].item<double>();

        auto updated = cifg_update(C_slot, k_slot, v_slot, bias);
        C_new.select(1, s).copy_(updated);
    }

    return C_new;
}


// Sparse Mixture-of-Experts dispatch.
// Routes each token to top-k experts based on gating scores.
// Args:
//   x: (batch, seq_len, d_model) - input tokens
//   gate_weights: (d_model, num_experts) - gating weight matrix
//   expert_weights: list of (d_model, d_model) - expert weight matrices
//   top_k: number of experts per token
// Returns:
//   output: (batch, seq_len, d_model) - routed output
torch::Tensor sparse_moe_dispatch(
    torch::Tensor x,
    torch::Tensor gate_weights,
    torch::Tensor expert_weights,
    int64_t top_k
) {
    auto batch = x.size(0);
    auto seq_len = x.size(1);
    auto d_model = x.size(2);
    auto num_experts = gate_weights.size(1);

    // Flatten batch and sequence
    auto x_flat = x.reshape({batch * seq_len, d_model});

    // Compute gating scores
    auto scores = torch::mm(x_flat, gate_weights);  // (B*S, num_experts)
    auto probs = torch::softmax(scores, -1);

    // Top-k selection
    auto [top_vals, top_idx] = torch::topk(probs, top_k, -1);
    top_vals = top_vals / (top_vals.sum(-1, true) + 1e-8);

    // Dispatch to experts
    auto output = torch::zeros_like(x_flat);
    for (int64_t k = 0; k < top_k; k++) {
        for (int64_t e = 0; e < num_experts; e++) {
            auto mask = (top_idx.select(1, k) == e);
            if (mask.any().item<bool>()) {
                auto x_masked = x_flat.index({mask});
                // expert_weights shape: (num_experts, d_model, d_model)
                auto W_e = expert_weights.select(0, e);
                auto out_e = torch::mm(x_masked, W_e);
                auto w_k = top_vals.select(1, k).index({mask}).unsqueeze(-1);
                output.index_put_({mask}, output.index({mask}) + w_k * out_e);
            }
        }
    }

    return output.reshape({batch, seq_len, d_model});
}


// Bi-encoder cross-attention.
// Computes cross-attention between two encoded sequences.
// Args:
//   query: (batch, q_len, d_model) - query sequence
//   key: (batch, kv_len, d_model) - key sequence
//   value: (batch, kv_len, d_model) - value sequence
//   W_Q: (d_model, d_model) - query projection
//   W_K: (d_model, d_model) - key projection
//   W_V: (d_model, d_model) - value projection
// Returns:
//   output: (batch, q_len, d_model)
torch::Tensor biencoder_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor W_Q,
    torch::Tensor W_K,
    torch::Tensor W_V
) {
    auto d_model = query.size(-1);
    auto scale = std::sqrt(static_cast<double>(d_model));

    // Project
    auto Q = torch::mm(
        query.reshape({-1, d_model}), W_Q
    ).reshape_as(query);
    auto K = torch::mm(
        key.reshape({-1, d_model}), W_K
    ).reshape_as(key);
    auto V = torch::mm(
        value.reshape({-1, d_model}), W_V
    ).reshape_as(value);

    // Attention scores
    auto scores = torch::bmm(Q, K.transpose(-2, -1)) / scale;
    auto attn = torch::softmax(scores, -1);
    auto output = torch::bmm(attn, V);

    return output;
}


// Fused attention + softmax kernel.
// Computes scaled dot-product attention with optional causal mask.
// Args:
//   Q: (batch, heads, seq_len, d_head)
//   K: (batch, heads, seq_len, d_head)
//   V: (batch, heads, seq_len, d_head)
//   causal: whether to apply causal mask
// Returns:
//   output: (batch, heads, seq_len, d_head)
torch::Tensor attention_softmax(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    auto d_head = Q.size(-1);
    auto seq_len = Q.size(-2);
    auto scale = std::sqrt(static_cast<double>(d_head));

    // Compute attention scores
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / scale;

    // Apply causal mask if requested
    if (causal) {
        auto mask = torch::triu(
            torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                        Q.options()),
            1
        );
        scores = scores + mask;
    }

    // Softmax
    auto attn = torch::softmax(scores, -1);

    // Weighted sum
    return torch::matmul(attn, V);
}


// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_burt_imma_cuda, m) {
    m.doc() = "BURT-IMMA CUDA-accelerated kernels";

    m.def("constrained_softmax", &constrained_softmax,
          "Constrained softmax with entropy bound",
          py::arg("logits"),
          py::arg("max_entropy") = 0.20,
          py::arg("temperature") = 1.0);

    m.def("cifg_update", &cifg_update,
          "Coupled Input-Forget Gate memory update",
          py::arg("C_old"),
          py::arg("key"),
          py::arg("value"),
          py::arg("forget_bias") = 0.0);

    m.def("batched_cifg_update", &batched_cifg_update,
          "Batched CIFG memory update for multiple slots",
          py::arg("C_old"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("forget_biases"));

    m.def("sparse_moe_dispatch", &sparse_moe_dispatch,
          "Sparse Mixture-of-Experts dispatch with top-k routing",
          py::arg("x"),
          py::arg("gate_weights"),
          py::arg("expert_weights"),
          py::arg("top_k") = 1);

    m.def("biencoder_attention", &biencoder_attention,
          "Bi-encoder cross-attention",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("W_Q"),
          py::arg("W_K"),
          py::arg("W_V"));

    m.def("attention_softmax", &attention_softmax,
          "Fused attention + softmax with optional causal mask",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("causal") = true);
}
