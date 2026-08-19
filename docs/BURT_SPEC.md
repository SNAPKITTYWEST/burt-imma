# BURT: Boolean Universal Reasoning Transformer

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Overview

BURT (Boolean Universal Reasoning Transformer) is a transformer-based architecture augmented with a Boolean algebra constraint layer and trained via Matrix-Memory Equilibrium Propagation (MMEP). It combines the representational power of attention mechanisms with the logical rigor of Boolean lattice operations, enforced through Huntington postulates at the architectural level.

BURT is designed for tasks requiring provably correct logical reasoning while maintaining the flexibility and scalability of modern transformer architectures.

---

## 2. Architecture Overview

```
Input -> Embedding -> [BURT Block x N] -> Output Head -> Output
                          |
                          v
                   +-------------+
                   | Transformer  |
                   | Backbone     |
                   +------+------+
                          |
                   +------v------+
                   | Boolean      |
                   | Constraint   |
                   | Layer        |
                   +------+------+
                          |
                   +------v------+
                   | MMEP         |
                   | Training     |
                   +-------------+
```

The architecture consists of three integrated systems:

1. **Transformer Backbone:** Multi-head attention with SmoothLeaky activations and GatesNormalization
2. **Boolean Algebra Constraint Layer:** Huntington postulate enforcement on intermediate representations
3. **MMEP Training:** Equilibrium propagation with matrix memory (see [MMEP_THEORY.md](./MMEP_THEORY.md))

---

## 3. Components

### 3.1 SmoothLeaky Activation

```python
SmoothLeaky(x) = x * sigmoid(k * x)    for x >= 0
               = alpha * x * sigmoid(k * x)    for x < 0
```

**Parameters:**
- `k`: smoothness parameter (default: 1.0, learnable)
- `alpha`: negative slope (default: 0.01, fixed)

**Properties:**
- Smooth everywhere (infinitely differentiable)
- Monotonically increasing
- Bounded gradient: `|dSmoothLeaky/dx| <= 1 + alpha`
- Converges to LeakyReLU as `k -> infinity`
- Required for MMEP convergence (energy function must be smooth)

### 3.2 GatesNormalization

```python
GatesNorm(x) = (x - mu) / sqrt(sigma^2 + eps) * gamma + beta
```

where `mu` and `sigma^2` are computed over the expert dimension, ensuring that the gating distribution is well-conditioned for sparse routing.

**Key difference from LayerNorm:** GatesNormalization operates on the gating logits before softmax, ensuring that:
1. No single expert dominates (prevents collapse)
2. The L1 norm of the resulting distribution is bounded
3. The entropy constraint (0.20) is satisfiable

**Parameters:**
- `gamma`: scale (learnable, per-expert)
- `beta`: bias (learnable, per-expert)
- `eps`: numerical stability (default: 1e-5)

### 3.3 CIFGMatrixMemory

The Coupled Input-Forget Gate Matrix Memory module:

```python
class CIFGMatrixMemory:
    C_global: Tensor[d_model, d_model]    # Shared memory
    C_expert: List[Tensor[d_expert, d_expert]]    # Per-expert memory
    W_f: Tensor[d_model, 2 * d_model]    # Forget gate weights
    b_f: Tensor[d_model]    # Forget gate bias

    def forward(self, h, x, expert_id=None):
        f = sigmoid(W_f @ concat(h, x) + b_f)
        candidate = outer(h, h)  # Rank-1 update
        C_global = f * C_global + (1 - f) * candidate
        if expert_id is not None:
            C_expert[expert_id] = f * C_expert[expert_id] + (1 - f) * candidate
        return C_global @ h  # Memory retrieval
```

**Properties:**
- Coupled gates: input gate = `1 - forget gate`
- Rank-1 updates preserve invertibility
- Forget bias schedule enables consolidation
- See [MMEP_THEORY.md](./MMEP_THEORY.md) Section 5 for full dynamics

### 3.4 GatesRouter

Sparse Mixture-of-Experts router with Gates normalization:

```python
class GatesRouter:
    W_gate: Tensor[num_experts, d_model]
    
    def forward(self, x):
        logits = W_gate @ x
        logits = gates_norm(logits)
        alpha = constrained_softmax(logits, entropy_bound=0.20)
        top_k_indices = topk(alpha, k=2)
        return top_k_indices, alpha[top_k_indices]
```

**Routing strategy:**
- Top-2 expert selection (default)
- Load balancing via auxiliary loss
- Entropy bounded at 0.20 nats
- Sparsity enforced via L1 penalty on `alpha`

### 3.5 SuperpositionedInductionHeads

Multi-head attention with superposition-aware induction:

```python
class SuperpositionedInductionHeads:
    W_Q, W_K, W_V: Tensor[num_heads, d_head, d_model]
    
    def forward(self, x, memory_context):
        Q = W_Q @ x
        K = W_K @ x
        V = W_V @ x
        
        # Standard attention
        attn = softmax(Q @ K^T / sqrt(d_head))
        
        # Induction: detect repeated patterns
        induction_score = detect_copy_pattern(Q, K, offset=1)
        
        # Superposition: combine with memory retrieval
        memory_V = memory_context @ V
        output = attn @ V + induction_score * memory_V
        
        return output
```

**Properties:**
- Detects and exploits repeated subsequences (induction)
- Superpositions memory-retrieved values with attention output
- Enables in-context learning through pattern completion
- Spectral norm constraint applied to all projection matrices

### 3.6 QuantumInterferenceResolver

Resolves conflicting expert outputs through interference-inspired combination:

```python
class QuantumInterferenceResolver:
    def forward(self, expert_outputs, routing_weights):
        # Represent each expert output as amplitude
        amplitudes = [sqrt(w) * exp(1j * phase(out)) for w, out in zip(routing_weights, expert_outputs)]
        
        # Interfere: sum amplitudes, then measure (take magnitude squared)
        superposition = sum(amplitudes)
        resolved = |superposition|^2 * sign(real(superposition))
        
        return resolved
```

**Purpose:** When multiple experts produce conflicting outputs, simple weighted averaging loses information. The interference resolver preserves phase relationships between expert outputs, allowing constructive and destructive interference to naturally select the most coherent answer.

---

## 4. Input/Output Specification

### 4.1 Input

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `input_ids` | int64 | `[batch, seq_len]` | Tokenized input |
| `attention_mask` | float32 | `[batch, seq_len]` | Padding mask |
| `memory_state` | float32 | `[batch, d_model, d_model]` | Optional: pre-loaded memory |

### 4.2 Output

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `logits` | float32 | `[batch, seq_len, vocab_size]` | Token predictions |
| `memory_state` | float32 | `[batch, d_model, d_model]` | Updated memory |
| `routing_decisions` | float32 | `[batch, seq_len, num_experts]` | Expert routing weights |
| `entropy` | float32 | `[batch]` | Routing entropy per sample |
| `spectral_norms` | float32 | `[num_layers]` | Max singular value per layer |

---

## 5. Layer-by-Layer Description

### Layer 0: Token Embedding + Positional Encoding

```
x_0 = Embed(input_ids) + PosEncode(positions)
```
- Embedding dimension: `d_model`
- Positional encoding: Rotary (RoPE)

### Layers 1 to N: BURT Blocks

Each BURT block consists of:

```
# Sub-layer 1: Attention with memory
h_attn = SuperpositionedInductionHeads(GatesNorm(x), memory_context=C_global @ x)
x = x + h_attn  # Residual

# Sub-layer 2: Boolean constraint
h_bool = BooleanConstraintLayer(GatesNorm(x))
x = x + h_bool  # Residual

# Sub-layer 3: MoE Feed-Forward
expert_ids, weights = GatesRouter(x)
h_ff = sum(weights[k] * Expert_k(x) for k in expert_ids)
h_ff = QuantumInterferenceResolver(expert_outputs, weights)
x = x + h_ff  # Residual

# Sub-layer 4: Memory update
C_global, C_expert = CIFGMatrixMemory.update(x, input)
```

### Layer N+1: Output Head

```
logits = Linear(GatesNorm(x_N))
```
- Entropy constraint enforced on softmax(logits) during training
- Spectral normalization on output projection

---

## 6. Inference Pipeline

```
1. Tokenize input
2. Load or initialize memory state (C_global, C_expert)
3. Forward pass through all BURT blocks
4. For each block:
   a. Compute attention with induction heads
   b. Apply Boolean constraints (Huntington verification)
   c. Route through sparse MoE (top-2)
   d. Resolve expert conflicts via interference
   e. Update memory via CIFG
5. Project to vocabulary via output head
6. Apply constrained decoding (entropy <= 0.20 on output distribution)
7. Return: tokens, updated memory, diagnostics
```

**Autoregressive generation:**
```
for each new token:
    logits, memory = BURT(context, memory)
    next_token = sample(constrained_softmax(logits, entropy_bound=0.20))
    context = append(context, next_token)
```

---

## 7. Training via MMEP

BURT is trained using Matrix-Memory Equilibrium Propagation as defined in [MMEP_THEORY.md](./MMEP_THEORY.md).

### 7.1 Free Phase

The network processes the input and relaxes to equilibrium:
- All layers compute forward pass
- Memory retrieval and routing decisions settle
- Hidden states converge: `||dh/dt|| < 1e-5`
- Duration: `T_free = 20` steps

### 7.2 Nudged Phase

The output is clamped toward the target:
- Nudging strength: `beta = 0.1` (annealed)
- Only output layer receives direct target signal
- Internal representations adjust via energy minimization
- Duration: `T_nudge = 4` steps

### 7.3 Parameter Update

```
dW = (1/beta) * (h_nudged @ h_nudged^T - h_free @ h_free^T)
W <- W - lr * dW
W <- project_spectral(W, lambda_max=0.95)
alpha <- project_entropy(alpha, bound=0.20)
```

---

## 8. Huntington Postulate Enforcement

The Boolean constraint layer enforces Huntington's postulates on intermediate representations:

### 8.1 Postulates

For any two elements `a`, `b` in the representation space:

1. **Commutativity:** `a + b = b + a` and `a * b = b * a`
2. **Distributivity:** `a * (b + c) = (a * b) + (a * c)` and `a + (b * c) = (a + b) * (a + c)`
3. **Identity:** There exist `0` and `1` such that `a + 0 = a` and `a * 1 = a`
4. **Complement:** For each `a`, there exists `a'` such that `a + a' = 1` and `a * a' = 0`

### 8.2 Enforcement Mechanism

```python
class BooleanConstraintLayer:
    def forward(self, x):
        # Project to Boolean lattice
        x_bool = hard_sigmoid(x)  # Approximate Boolean
        
        # Verify Huntington postulates
        complement = 1 - x_bool
        assert_approx(x_bool + complement, 1, tol=1e-3)
        assert_approx(x_bool * complement, 0, tol=1e-3)
        
        # Soft constraint loss (added to energy)
        huntington_loss = ||x_bool + complement - 1||^2 + ||x_bool * complement||^2
        
        # Return constrained representation
        return x + (x_bool - x).detach()  # Straight-through estimator
```

### 8.3 Constraint Satisfaction During Training

The Huntington constraint is enforced as a soft penalty during early training, gradually hardened:
```
lambda_huntington(epoch) = min(1.0, epoch / warmup_epochs)
```

After warmup, violations are treated as hard constraints and projected away.

---

## 9. Memory Management (CIFG Gating)

### 9.1 Memory Lifecycle

```
Phase 1 (Memorization): C_global absorbs corpus knowledge
    - Forget gate bias: low (high plasticity)
    - Update frequency: every batch

Phase 2 (Query Training): C_expert learns task-specific patterns
    - Forget gate bias: medium
    - Update frequency: when expert is selected

Phase 3 (Convergence): All memories consolidate
    - Forget gate bias: high (freeze)
    - Update frequency: decreasing schedule
```

### 9.2 Memory Capacity

- `C_global`: `d_model^2` parameters = effective capacity of `d_model` memories
- `C_expert_k`: `d_expert^2` parameters each, `K` experts total
- Total memory: `d_model^2 + K * d_expert^2`
- For `d_model=512`, `K=4`, `d_expert=256`: 262K + 262K = 524K memory parameters

### 9.3 Memory Overflow Handling

When memory approaches capacity (determined by condition number of C):
1. Compute SVD of memory matrix
2. Truncate smallest singular values (keep top `r` rank)
3. Re-normalize to maintain spectral bound
4. Log memory utilization metric

---

## 10. Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `d_model` | 512 | [256, 2048] | Model dimension |
| `num_layers` | 6 | [4, 24] | Number of BURT blocks |
| `num_heads` | 8 | [4, 32] | Attention heads |
| `num_experts` | 4 | [2, 16] | MoE experts |
| `top_k` | 2 | [1, 4] | Experts per token |
| `d_expert` | 256 | [128, 1024] | Expert dimension |
| `lambda_max` | 0.95 | [0.9, 0.99] | Spectral bound |
| `entropy_bound` | 0.20 | [0.1, 0.5] | Routing entropy limit |
| `beta` | 0.1 | [0.01, 1.0] | Nudging strength |
| `T_free` | 20 | [10, 100] | Free phase steps |
| `T_nudge` | 4 | [2, 20] | Nudged phase steps |
| `smoothleaky_k` | 1.0 | [0.1, 10.0] | Activation smoothness |

---

## 11. Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Attention | O(n^2 * d) | Standard quadratic |
| Boolean constraint | O(n * d) | Element-wise |
| MoE routing | O(n * K) | K = num_experts |
| Expert forward | O(n * d_expert^2 * top_k) | Only active experts |
| Memory update | O(d^2) | Per-step CIFG |
| Memory retrieval | O(d^2) | Matrix-vector product |
| EP free phase | O(T_free * forward_pass) | Iterative |
| EP nudged phase | O(T_nudge * forward_pass) | Iterative |

**Total training cost:** Approximately `(T_free + T_nudge)` times standard backprop cost per sample.

---

## 12. References

- [MMEP_THEORY.md](./MMEP_THEORY.md) - Mathematical foundation for training
- [IMMA_TRAINING.md](./IMMA_TRAINING.md) - Training protocol
- [SUM_INVERSION_THEORY.md](./SUM_INVERSION_THEORY.md) - Memory retrieval theory
- [kernel_api.md](./kernel_api.md) - CUDA kernel implementations
