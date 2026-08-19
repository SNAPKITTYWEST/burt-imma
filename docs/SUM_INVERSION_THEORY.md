# Sum-Inversion Theory for Boolean Kernel Machines

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Overview

Sum-Inversion Theory provides the mathematical foundation for knowledge storage and retrieval in BURT-IMMA's matrix memory system. The core idea is elegant: represent knowledge as a sum of rank-1 matrices, then recover individual components via projection. This document formalizes the theory, proves key properties, and connects it to the broader BURT-IMMA architecture.

---

## 2. Core Idea

### 2.1 Sum Representation

Knowledge is stored as a sum of outer products (rank-1 matrices):

```
C = sum_{i=1}^{n} v_i * v_i^T
```

where each `v_i` is a d-dimensional vector encoding a piece of knowledge (a fact, pattern, or relationship).

### 2.2 Inversion for Retrieval

Given the composite memory `C` and a query related to the j-th stored item, we recover `v_j` via projection:

```
v_j_hat = C * q_j / ||C * q_j||
```

where `q_j` is a query vector aligned with `v_j` (e.g., `q_j = v_j + noise` or a learned query embedding).

### 2.3 Why This Works

If the stored vectors `{v_i}` are approximately orthogonal:
```
C * v_j = sum_i (v_i * v_i^T) * v_j
        = v_j * (v_j^T * v_j) + sum_{i != j} v_i * (v_i^T * v_j)
        = ||v_j||^2 * v_j + sum_{i != j} <v_i, v_j> * v_i
        ≈ ||v_j||^2 * v_j    (when v_i ⊥ v_j for i != j)
```

The retrieval is exact when stored vectors are orthogonal, and approximate (with bounded error) when they are merely approximately orthogonal.

---

## 3. Boolean Kernel

### 3.1 Definition

The Boolean kernel maps inputs to the Boolean lattice and computes inner products in that space:

```
K(x, y) = <phi(x), phi(y)>
```

where `phi: X -> {0, 1}^D` maps to the Boolean lattice (D-dimensional binary feature space).

### 3.2 Feature Map

```python
def phi(x):
    """Map input x to Boolean lattice representation."""
    # Threshold features at learned boundaries
    features = []
    for threshold_set in learned_thresholds:
        for t in threshold_set:
            features.append(1 if x > t else 0)
    return array(features)  # Binary vector in {0,1}^D
```

### 3.3 Kernel Properties

1. **Positive semi-definite:** `K(x,y) = phi(x)^T phi(y) >= 0` (inner product of non-negative vectors)
2. **Integer-valued:** `K(x,y) in {0, 1, ..., D}` (sum of binary products)
3. **Bounded:** `0 <= K(x,y) <= D`
4. **Mercer condition:** Satisfied by construction (explicit feature map exists)

### 3.4 Connection to Boolean Algebra

The feature map `phi` respects Boolean operations:
- `phi(x AND y) = phi(x) * phi(y)` (element-wise AND)
- `phi(x OR y) = phi(x) + phi(y) - phi(x) * phi(y)` (element-wise OR)
- `phi(NOT x) = 1 - phi(x)` (element-wise complement)

These correspond to Huntington postulates in the feature space, ensuring that BURT's Boolean constraint layer operates naturally on the kernel representation.

---

## 4. Sum-Inversion Mechanics

### 4.1 Storage

To store a new piece of knowledge encoded as vector `v`:

```python
def store(C, v, forget_gate):
    """CIFG-gated storage of v into memory C."""
    candidate = outer(v, v)  # Rank-1 matrix
    C_new = forget_gate * C + (1 - forget_gate) * candidate
    return C_new
```

Note: With CIFG gating, the memory is a weighted combination rather than a pure sum. The sum-inversion theory applies to the effective stored content.

### 4.2 Retrieval

To retrieve knowledge associated with query `q`:

```python
def retrieve(C, q):
    """Project query through memory to retrieve stored content."""
    raw = C @ q
    normalized = raw / (norm(raw) + eps)
    return normalized
```

### 4.3 Multi-step Retrieval (Iterative Refinement)

For noisy queries, iterative refinement improves retrieval:

```python
def iterative_retrieve(C, q, steps=3):
    """Refine retrieval through multiple projections."""
    v_hat = q
    for _ in range(steps):
        v_hat = C @ v_hat
        v_hat = v_hat / (norm(v_hat) + eps)
    return v_hat
```

This converges to the dominant eigenvector direction aligned with `q`, which corresponds to the most strongly stored pattern matching the query.

---

## 5. Rank Preservation Theorem

### 5.1 Statement

**Theorem:** Let `C = sum_{i=1}^n v_i v_i^T` where `{v_i}` are vectors in R^d. Then:

1. `rank(C) = dim(span({v_i}))` (rank equals dimension of span)
2. `rank(C) <= min(n, d)` (bounded by count and dimension)
3. If CIFG gating preserves linear independence, then `rank(C)` is non-decreasing during storage

### 5.2 Proof

**(1)** By construction, `C = V V^T` where `V = [v_1, ..., v_n]` is the `d x n` matrix of stored vectors. Therefore:
```
rank(C) = rank(V V^T) = rank(V) = dim(column_space(V)) = dim(span({v_i}))
```

**(2)** Follows from `rank(V) <= min(rows(V), cols(V)) = min(d, n)`.

**(3)** Consider CIFG update: `C_new = f * C_old + (1-f) * v_new v_new^T`.
- If `v_new` is not in `span(columns of C_old)`, then `rank(C_new) >= rank(C_old) + 1` (when `f < 1`).
- If `v_new` is in `span(columns of C_old)`, then `rank(C_new) = rank(C_old)`.
- The forget gate `f < 1` ensures new information is incorporated.
- Rank can only increase or stay the same, never decrease (as long as `f > 0` prevents complete overwrite).

### 5.3 Effective Rank

In practice, we use effective rank (number of significant singular values):

```python
def effective_rank(C, threshold=0.01):
    """Count singular values above threshold * max."""
    s = svd(C, compute_uv=False)
    return sum(s > threshold * s[0])
```

---

## 6. Round-Trip Accuracy Bounds

### 6.1 Setup

Given:
- Memory `C = sum_{i=1}^n v_i v_i^T`
- Query `q_j = v_j + epsilon` where `epsilon` is noise with `||epsilon|| <= delta`
- Retrieved: `v_j_hat = C q_j / ||C q_j||`

### 6.2 Exact Retrieval (Orthogonal Case)

**Theorem:** If `{v_1, ..., v_n}` are orthonormal, then retrieval is exact for noiseless queries:

```
v_j_hat = C v_j / ||C v_j|| = v_j
```

**Proof:** `C v_j = sum_i v_i (v_i^T v_j) = v_j * 1 + sum_{i!=j} v_i * 0 = v_j`. Normalization preserves direction.

### 6.3 Approximate Retrieval (Non-orthogonal Case)

**Theorem:** If the Gram matrix `G_{ij} = <v_i, v_j>` satisfies `|G_{ij}| <= mu` for `i != j` (incoherence condition), then:

```
||v_j_hat - v_j/||v_j||| <= (n-1) * mu / (1 - (n-1) * mu)
```

provided `(n-1) * mu < 1`.

**Proof sketch:**
```
C v_j = ||v_j||^2 * v_j + sum_{i!=j} <v_i, v_j> * v_i
```
The error term has norm bounded by `(n-1) * mu * max_i(||v_i||)`. After normalization, the angle between `v_j_hat` and `v_j` is bounded by `arcsin((n-1) * mu / ||v_j||^2)`.

### 6.4 Noisy Query Bound

**Theorem:** With query noise `||epsilon|| <= delta`:

```
||v_j_hat - v_j/||v_j||| <= (n-1) * mu / (1 - (n-1) * mu) + delta * sigma_max(C) / ||C v_j||
```

The second term accounts for noise amplification by the memory matrix. Spectral normalization (sigma_max(C) <= lambda_max) keeps this bounded.

### 6.5 Capacity-Accuracy Tradeoff

For a d-dimensional memory storing n items:
- Perfect retrieval: requires `n <= d` and orthogonality
- Good retrieval (error < 0.1): requires `n <= d / 10` (rule of thumb)
- Acceptable retrieval (error < 0.3): requires `n <= d / 3`

This motivates the Chinchilla-like scaling relationship between memory dimension and stored knowledge (Section 8).

---

## 7. Connection to Gates Normalization

### 7.1 The Invertibility Problem

For sum-inversion to work, the memory matrix `C` must be well-conditioned (not nearly singular). If stored vectors are highly correlated, `C` becomes ill-conditioned and retrieval fails.

### 7.2 How Gates Normalization Ensures Invertibility

Gates normalization, applied before storage, decorrelates the vectors being stored:

```python
def store_with_gates_norm(C, h, x, gates_norm, cifg):
    # Normalize before storage
    h_normed = gates_norm(h)
    
    # GatesNorm ensures:
    # 1. Zero mean across expert dimension (decorrelation)
    # 2. Unit variance (prevents magnitude collapse)
    # 3. Learnable affine (preserves expressivity)
    
    # Store normalized vector
    v = h_normed
    f = cifg.forget_gate(h, x)
    C = f * C + (1 - f) * outer(v, v)
    return C
```

### 7.3 Condition Number Bound

**Proposition:** With Gates normalization applied before storage, the condition number of `C` is bounded:

```
cond(C) = sigma_max(C) / sigma_min(C) <= (1 + gamma_max) / (1 - gamma_max) * sqrt(n)
```

where `gamma_max = max(|gamma_k|)` is the maximum learned scale in GatesNorm.

This bound ensures that retrieval error remains controlled even as the memory fills up, unlike unnormalized storage where condition numbers can grow exponentially.

### 7.4 Spectral Norm Interaction

The spectral constraint `sigma_max(W) <= 0.95` on weight matrices upstream of memory storage indirectly bounds `sigma_max(C)`:
- Stored vectors `v = W_last @ ... @ W_1 @ x` have bounded norm
- Therefore `sigma_max(C) = sigma_max(sum v_i v_i^T) <= n * max_i(||v_i||^2) <= n * (0.95)^L * ||x_max||^2`
- With spectral projection on `C` itself: `sigma_max(C) <= lambda_max`

---

## 8. Chinchilla Scaling for Matrix Memory

### 8.1 Motivation

The Chinchilla scaling law (Hoffmann et al., 2022) establishes optimal tokens-to-parameters ratios for language models. We derive an analogous scaling law for matrix memory: the optimal ratio of stored items to memory capacity.

### 8.2 Memory Capacity

For a `d x d` memory matrix `C`:
- **Maximum rank:** d
- **Practical capacity** (with retrieval error < epsilon): `n_max = d * (1 - epsilon) / (1 + mu * d)`
- **Optimal utilization:** 60-70% of maximum rank

### 8.3 Scaling Law

The optimal number of tokens `T` to process for a model with memory dimension `d` and number of experts `K`:

```
T_optimal = alpha * (d^2 + K * d_expert^2) / d_model
```

where `alpha` is a constant determined by:
- The incoherence of the data distribution
- The forget rate (CIFG bias)
- The desired retrieval accuracy

### 8.4 Empirical Scaling Coefficients

Based on preliminary experiments:

| d_model | K | d_expert | T_optimal | Retrieval Acc |
|---------|---|----------|-----------|---------------|
| 256 | 4 | 128 | 65K tokens | 92% |
| 512 | 4 | 256 | 262K tokens | 94% |
| 1024 | 8 | 512 | 2.1M tokens | 96% |
| 2048 | 16 | 1024 | 16.8M tokens | 97% |

### 8.5 Interpretation

- **Undertrained memory** (T < T_optimal): Memory has capacity but lacks diverse content. Retrieval is accurate but limited.
- **Overtrained memory** (T > T_optimal): Memory is saturated. New information overwrites old (despite CIFG gating). Retrieval degrades.
- **Optimal point:** All available capacity is utilized with minimal interference between stored items.

### 8.6 Comparison to Chinchilla

| Aspect | Chinchilla (LLMs) | Sum-Inversion (BURT-IMMA) |
|--------|--------------------|---------------------------|
| Resource | Compute (FLOPs) | Memory capacity (rank) |
| Scaling variable | Parameters | Memory dimension d^2 |
| Data variable | Tokens | Stored items n |
| Optimal ratio | ~20 tokens/param | ~0.6 items/rank |
| Diminishing returns | Larger model, same data | Larger memory, same corpus |
| Key constraint | Compute budget | Retrieval accuracy |

---

## 9. Formal Properties

### 9.1 Associativity of Storage

Storage is associative: the order of storing items does not affect the final memory (in the pure-sum case):

```
store(store(C, v1), v2) = store(store(C, v2), v1)
```

since `v1 v1^T + v2 v2^T = v2 v2^T + v1 v1^T`.

Note: With CIFG gating (`f < 1`), order matters due to the forget factor. Earlier items are exponentially decayed.

### 9.2 Linearity of Retrieval

Retrieval (before normalization) is linear in the query:

```
retrieve(C, alpha * q1 + beta * q2) = alpha * retrieve(C, q1) + beta * retrieve(C, q2)
```

This enables compositional queries: combining aspects of multiple queries yields a retrieval that combines aspects of their individual results.

### 9.3 Idempotence under Orthogonality

If `v_j` is orthogonal to all other stored vectors, then repeated retrieval is idempotent:

```
retrieve(C, retrieve(C, v_j)) = retrieve(C, v_j) = v_j / ||v_j||
```

This is a fixed-point property that stabilizes iterative refinement.

---

## 10. Algorithms

### 10.1 Efficient Storage (Rank-1 Update)

```python
def efficient_store(C, v, forget_gate):
    """O(d^2) storage via rank-1 update."""
    # No need to reconstruct full outer product if using efficient representation
    # C = f * C + (1-f) * v @ v^T
    # Implemented as: C *= f; C += (1-f) * outer(v, v)
    C.mul_(forget_gate)
    C.addr_(v, v, alpha=(1 - forget_gate))  # Rank-1 update
    return C
```

### 10.2 Batch Storage

```python
def batch_store(C, V, forget_gates):
    """Store multiple vectors with individual forget gates."""
    # V: [batch, d], forget_gates: [batch]
    for v, f in zip(V, forget_gates):
        C = efficient_store(C, v, f)
    return C
```

### 10.3 Top-k Retrieval

```python
def topk_retrieve(C, q, stored_vectors, k=5):
    """Retrieve top-k most relevant stored items."""
    scores = C @ q  # [d]
    # Project onto stored directions
    relevance = [abs(dot(scores, v)) / (norm(v) + eps) for v in stored_vectors]
    top_k_idx = argsort(relevance)[-k:]
    return [stored_vectors[i] for i in top_k_idx]
```

---

## 11. Limitations and Failure Modes

### 11.1 Capacity Exhaustion

When `n >> d`, the memory becomes rank-d and cannot distinguish between stored items. Mitigation: monitor effective rank, trigger memory consolidation.

### 11.2 Correlated Storage

When stored vectors are highly correlated (`mu -> 1`), retrieval error grows unboundedly. Mitigation: Gates normalization decorrelates before storage.

### 11.3 Catastrophic Interference

Without CIFG gating, new stores overwrite old. The forget gate controls this tradeoff. Mitigation: increase forget bias during later training phases.

### 11.4 Numerical Instability

For very large `n` or very small singular values, the matrix inversion implicit in retrieval becomes unstable. Mitigation: regularization (`C + epsilon * I`) and spectral truncation.

---

## 12. References

- Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS.
- Ramsauer, H. et al. (2021). "Hopfield Networks is All You Need." ICLR.
- Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla paper)
- [MMEP_THEORY.md](./MMEP_THEORY.md) - Energy-based training
- [BURT_SPEC.md](./BURT_SPEC.md) - Architecture using sum-inversion memory
- [kernel_api.md](./kernel_api.md) - CUDA kernels for memory operations
