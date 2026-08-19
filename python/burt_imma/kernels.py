"""
BURT-IMMA Kernel Functions with CPU Fallbacks
License: BSL-1.1
Contact: jessica@collectivekitty.com

For each kernel function: try CUDA version first, fall back to pure PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Optional

try:
    import _burt_imma_cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


def constrained_softmax(
    logits: torch.Tensor,
    max_entropy: float = 0.20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Softmax with spectral norm constraint (entropy bounded).

    Applies softmax along the last dimension, iteratively sharpening
    (reducing temperature) until entropy <= max_entropy.

    Args:
        logits: Input logits of any shape (..., N)
        max_entropy: Maximum allowed entropy of output distribution
        temperature: Initial temperature for scaling

    Returns:
        Probability distribution with entropy <= max_entropy
    """
    if _HAS_CUDA and logits.is_cuda:
        return _burt_imma_cuda.constrained_softmax(logits, max_entropy, temperature)

    # Pure PyTorch fallback
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)

    # Check entropy and sharpen if needed
    ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    violations = ent > max_entropy

    if violations.any():
        temp = temperature
        for _ in range(50):
            temp *= 0.8
            p = F.softmax(logits / temp, dim=-1)
            h = -(p * (p + 1e-10).log()).sum(dim=-1)
            if (h <= max_entropy).all():
                return p
            # Update only violating entries
            probs = torch.where(
                violations.unsqueeze(-1).expand_as(probs),
                p, probs
            )
            ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            violations = ent > max_entropy
            if not violations.any():
                break

    return probs


def cifg_update(
    C_old: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forget_bias: float = 0.0,
) -> torch.Tensor:
    """Coupled Input-Forget Gate memory update.

    Implements: C_new = f * C_old + (1-f) * candidate
    where candidate = normalized outer product of key and value,
    and f = sigmoid(||key|| + forget_bias).

    Args:
        C_old: (batch, d_mem, d_mem) - current memory matrix
        key: (batch, d_mem) - write key
        value: (batch, d_mem) - write value
        forget_bias: bias term for forget gate

    Returns:
        C_new: (batch, d_mem, d_mem) - updated memory matrix
    """
    if _HAS_CUDA and C_old.is_cuda:
        return _burt_imma_cuda.cifg_update(C_old, key, value, forget_bias)

    # Pure PyTorch fallback
    # Forget gate from key norm
    f = torch.sigmoid(key.norm(dim=-1, keepdim=True) + forget_bias)  # (batch, 1)

    # Outer product candidate
    candidate = torch.bmm(
        key.unsqueeze(-1),   # (batch, d_mem, 1)
        value.unsqueeze(-2)  # (batch, 1, d_mem)
    )  # (batch, d_mem, d_mem)

    # Normalize candidate
    cand_norm = candidate.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-8
    candidate = candidate / cand_norm

    # CIFG update
    f_exp = f.unsqueeze(-1)  # (batch, 1, 1)
    C_new = f_exp * C_old + (1.0 - f_exp) * candidate

    return C_new


def batched_cifg_update(
    C_old: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    forget_biases: torch.Tensor,
) -> torch.Tensor:
    """Batched CIFG update for multiple memory slots.

    Args:
        C_old: (batch, num_slots, d_mem, d_mem)
        keys: (batch, num_slots, d_mem)
        values: (batch, num_slots, d_mem)
        forget_biases: (num_slots,)

    Returns:
        C_new: (batch, num_slots, d_mem, d_mem)
    """
    if _HAS_CUDA and C_old.is_cuda:
        return _burt_imma_cuda.batched_cifg_update(C_old, keys, values, forget_biases)

    # Pure PyTorch fallback
    batch, num_slots, d_mem, _ = C_old.shape
    C_new = torch.zeros_like(C_old)

    for s in range(num_slots):
        C_slot = C_old[:, s]           # (batch, d_mem, d_mem)
        k_slot = keys[:, s]            # (batch, d_mem)
        v_slot = values[:, s]          # (batch, d_mem)
        bias = forget_biases[s].item()

        C_new[:, s] = cifg_update(C_slot, k_slot, v_slot, bias)

    return C_new


def sparse_moe_dispatch(
    x: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    top_k: int = 1,
) -> torch.Tensor:
    """Sparse Mixture-of-Experts dispatch with top-k routing.

    Routes each token to its top-k experts based on gating scores.

    Args:
        x: (batch, seq_len, d_model) - input tokens
        gate_weights: (d_model, num_experts) - gating projection
        expert_weights: (num_experts, d_model, d_model) - expert matrices
        top_k: number of experts per token

    Returns:
        output: (batch, seq_len, d_model) - routed output
    """
    if _HAS_CUDA and x.is_cuda:
        return _burt_imma_cuda.sparse_moe_dispatch(x, gate_weights, expert_weights, top_k)

    # Pure PyTorch fallback
    batch, seq_len, d_model = x.shape
    num_experts = gate_weights.shape[1]

    # Flatten
    x_flat = x.reshape(batch * seq_len, d_model)

    # Gating scores
    scores = torch.mm(x_flat, gate_weights)  # (B*S, num_experts)
    probs = F.softmax(scores, dim=-1)

    # Top-k
    top_vals, top_idx = torch.topk(probs, top_k, dim=-1)
    top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

    # Dispatch
    output = torch.zeros_like(x_flat)
    for k in range(top_k):
        for e in range(num_experts):
            mask = (top_idx[:, k] == e)
            if mask.any():
                x_masked = x_flat[mask]
                W_e = expert_weights[e]
                out_e = torch.mm(x_masked, W_e)
                output[mask] = output[mask] + top_vals[mask, k:k+1] * out_e

    return output.reshape(batch, seq_len, d_model)


def biencoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
) -> torch.Tensor:
    """Bi-encoder cross-attention between two sequences.

    Args:
        query: (batch, q_len, d_model) - query sequence
        key: (batch, kv_len, d_model) - key sequence
        value: (batch, kv_len, d_model) - value sequence
        W_Q: (d_model, d_model) - query projection
        W_K: (d_model, d_model) - key projection
        W_V: (d_model, d_model) - value projection

    Returns:
        output: (batch, q_len, d_model)
    """
    if _HAS_CUDA and query.is_cuda:
        return _burt_imma_cuda.biencoder_attention(query, key, value, W_Q, W_K, W_V)

    # Pure PyTorch fallback
    batch, q_len, d_model = query.shape
    kv_len = key.shape[1]
    scale = d_model ** 0.5

    # Project
    Q = torch.matmul(query, W_Q)   # (batch, q_len, d_model)
    K = torch.matmul(key, W_K)     # (batch, kv_len, d_model)
    V = torch.matmul(value, W_V)   # (batch, kv_len, d_model)

    # Attention scores
    scores = torch.bmm(Q, K.transpose(-2, -1)) / scale  # (batch, q_len, kv_len)
    attn = F.softmax(scores, dim=-1)
    output = torch.bmm(attn, V)  # (batch, q_len, d_model)

    return output


def attention_softmax(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Fused attention + softmax with optional causal mask.

    Computes scaled dot-product attention efficiently.

    Args:
        Q: (batch, heads, seq_len, d_head)
        K: (batch, heads, seq_len, d_head)
        V: (batch, heads, seq_len, d_head)
        causal: whether to apply causal (autoregressive) mask

    Returns:
        output: (batch, heads, seq_len, d_head)
    """
    if _HAS_CUDA and Q.is_cuda:
        return _burt_imma_cuda.attention_softmax(Q, K, V, causal)

    # Pure PyTorch fallback
    d_head = Q.shape[-1]
    seq_len = Q.shape[-2]
    scale = d_head ** 0.5

    # Compute scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    # Apply causal mask
    if causal:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=Q.device, dtype=Q.dtype),
            diagonal=1
        )
        scores = scores + mask

    # Softmax + weighted sum
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)
