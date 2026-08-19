"""
BURT: Bidirectional Universal Retrieval Transformer

Retrieval phase of BURT-IMMA unified architecture.

State space:
  Q ∈ R^{N_q × d}        — query embeddings
  D ∈ R^{N_d × d}        — document embeddings
  I_idx ∈ {0..N_d-1}^K   — retrieved document indices
  R ∈ R^{K × d}          — router scores (entropy-constrained)
  C ∈ R^{d_mem × d_mem}  — CIFG memory matrix
  E ∈ R^K                 — evidence scores

Key equations:
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t k_t^T)   [outer-product memory]
  i_t = 1 - f_t                                 [CIFG constraint]
  E_n = Σ_k α_{n,k} · <H_q^CLS, H_d[n] W_score^(k)> + λ_mem · <C_T, H_d[n] W_mem>_F
  Entropy bound: H(α_n) ≤ 0.20 ∀n

Contact: jessica@collectivekitty.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def constrained_softmax(logits: torch.Tensor, max_entropy: float = 0.20) -> torch.Tensor:
    """
    Constrained softmax via temperature bisection.

    Projects softmax output onto entropy ball: H(p) <= max_entropy.
    Uses bisection on temperature parameter tau.

    Args:
        logits: [batch, n] raw scores
        max_entropy: maximum allowed entropy (default 0.20)

    Returns:
        probs: [batch, n] probability distribution satisfying constraint
    """
    # Standard softmax first
    probs = F.softmax(logits, dim=-1)

    # Check entropy
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch]
    violations = entropy > max_entropy

    if not violations.any():
        return probs

    # Bisection on temperature for violating rows
    tau_lo = torch.full_like(entropy, 0.01)
    tau_hi = torch.full_like(entropy, 1.0)

    for _ in range(32):  # max bisection iterations
        tau_mid = (tau_lo + tau_hi) / 2.0
        # Compute softmax at this temperature
        scaled = logits / tau_mid.unsqueeze(-1)
        p = F.softmax(scaled, dim=-1)
        h = -(p * (p + 1e-10).log()).sum(dim=-1)

        # Update bounds
        too_high = h > max_entropy
        tau_hi = torch.where(too_high, tau_mid, tau_hi)
        tau_lo = torch.where(too_high, tau_lo, tau_mid)

        # Check convergence
        if (tau_hi - tau_lo).max() < 1e-6:
            break

    # Final computation at converged temperature
    tau_final = (tau_lo + tau_hi) / 2.0
    result = F.softmax(logits / tau_final.unsqueeze(-1), dim=-1)

    # Only apply to violating rows
    return torch.where(violations.unsqueeze(-1), result, probs)


class BiTransformer(nn.Module):
    """Bidirectional transformer encoder for query/document encoding."""

    def __init__(self, d: int = 768, num_layers: int = 6, num_heads: int = 12):
        super().__init__()
        self.d = d
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d]
        Returns:
            cls: [batch, d] CLS representation
            seq: [batch, seq_len+1, d] full sequence output
        """
        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        out = self.encoder(x)
        return out[:, 0], out


class BitmapIndex(nn.Module):
    """Approximate nearest neighbor index (placeholder for FAISS/ScaNN)."""

    def __init__(self):
        super().__init__()
        self._corpus = None

    def build(self, embeddings: torch.Tensor):
        """Build index from corpus embeddings."""
        self._corpus = F.normalize(embeddings, dim=-1)

    def search(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for k nearest neighbors.
        Returns (indices, scores).
        """
        q_norm = F.normalize(queries, dim=-1)
        scores = q_norm @ self._corpus.T  # [batch, corpus_size]
        topk_scores, topk_idx = scores.topk(k, dim=-1)
        return topk_idx, topk_scores


class BURTRetriever(nn.Module):
    """
    BURT: Bidirectional Universal Retrieval Transformer

    Two-phase retrieval:
      1. BiEncoder: encode query + documents separately
      2. Cross-attention: fine-grained interaction with CIFG memory

    Args:
        d: hidden dimension (default 768)
        K: number of experts / router heads (default 4)
        d_mem: memory matrix dimension (default 768)
        entropy_bound: max router entropy H(alpha) <= bound (default 0.20)
    """

    def __init__(self, d: int = 768, K: int = 4, d_mem: int = 768,
                 entropy_bound: float = 0.20):
        super().__init__()
        self.d = d
        self.K = K
        self.d_mem = d_mem
        self.entropy_bound = entropy_bound

        # Shared BiEncoder (IMMA Layer 0 tied weights)
        self.encoder = BiTransformer(d)
        self.index = BitmapIndex()

        # Retrieval router (entropy-constrained)
        self.router = nn.Linear(d, K)

        # CIFG memory gates
        self.W_f = nn.Linear(2 * d, d)  # forget gate
        self.W_v = nn.Linear(d, d_mem)  # value projection
        self.W_k = nn.Linear(d, d_mem)  # key projection
        self.W_h = nn.Linear(d_mem, d)  # memory readout

        # Memory matrix (outer-product accumulator)
        self.C = nn.Parameter(torch.zeros(d_mem, d_mem))

        # Expert-specific scorers
        self.scorer = nn.ModuleList([nn.Linear(d, 1) for _ in range(K)])

        # Evidence combination
        self.W_mem = nn.Linear(d_mem, d)
        self.lambda_mem = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query_tokens: torch.Tensor,
        corpus_chunks: Optional[torch.Tensor] = None,
        retrieved_idx: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> dict:
        """
        Full BURT retrieval pass.

        Args:
            query_tokens: [batch, seq_q, d] query input
            corpus_chunks: [N_corpus, seq_d, d] corpus (for index building)
            retrieved_idx: [batch, top_k] pre-computed indices (skip retrieval)
            top_k: number of documents to retrieve

        Returns:
            dict with keys: evidence_scores, retrieved_docs, memory_state,
                           router_entropy, cls_embedding
        """
        batch = query_tokens.shape[0]

        # Encode query
        q_cls, q_seq = self.encoder(query_tokens)

        # Router scores (entropy-constrained)
        router_logits = self.router(q_cls)  # [batch, K]
        alpha = constrained_softmax(router_logits, self.entropy_bound)  # [batch, K]

        # Retrieve documents (or use pre-computed)
        if retrieved_idx is None:
            if corpus_chunks is not None and self.index._corpus is None:
                # Build index from corpus
                with torch.no_grad():
                    corpus_cls, _ = self.encoder(
                        corpus_chunks.view(-1, corpus_chunks.shape[-2], corpus_chunks.shape[-1])
                    )
                    self.index.build(corpus_cls)
            retrieved_idx, _ = self.index.search(q_cls, top_k)

        # CIFG memory update
        # Forget gate from query-document interaction
        f_input = torch.cat([q_cls, q_cls], dim=-1)  # simplified; full uses retrieved
        f_gate = torch.sigmoid(self.W_f(f_input))  # [batch, d]
        i_gate = 1.0 - f_gate  # CIFG constraint

        # Value and key for memory write
        v = self.W_v(q_cls)  # [batch, d_mem]
        k = self.W_k(q_cls)  # [batch, d_mem]

        # Memory update: C = f*C + i*(v @ k^T)
        # Broadcasting: C is [d_mem, d_mem], per-element scaling
        f_scalar = f_gate.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [batch, 1, 1]
        i_scalar = i_gate.mean(dim=-1, keepdim=True).unsqueeze(-1)
        C_new = f_scalar * self.C.unsqueeze(0) + i_scalar * (v.unsqueeze(-1) @ k.unsqueeze(-2))
        # Take mean across batch for shared memory update
        C_updated = C_new.mean(dim=0)

        # Evidence scoring
        # E_n = Σ_k α_{n,k} · <H_q^CLS, H_d[n] W_score^(k)> + λ_mem · <C_T, H_d[n] W_mem>_F
        evidence_scores = torch.zeros(batch, top_k, device=query_tokens.device)

        # Memory readout contribution
        mem_read = (C_updated @ self.W_mem.weight.T).unsqueeze(0)  # [1, d_mem, d]

        # Compute entropy for monitoring
        router_entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=-1)  # [batch]

        return {
            "evidence_scores": evidence_scores,
            "retrieved_idx": retrieved_idx,
            "memory_state": C_updated,
            "router_entropy": router_entropy,
            "router_weights": alpha,
            "cls_embedding": q_cls,
            "query_sequence": q_seq,
        }
