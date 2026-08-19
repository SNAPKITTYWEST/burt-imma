# Matrix-Memory Equilibrium Propagation: Mathematical Foundation

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Overview

Matrix-Memory Equilibrium Propagation (MMEP) is a biologically plausible training algorithm that combines equilibrium propagation with structured matrix memory. Unlike backpropagation, MMEP relies on local Hebbian-style learning rules derived from the difference between free and nudged equilibrium states, while maintaining hard constraints on spectral norm, entropy, and sparsity.

MMEP replaces the implicit differentiation of standard EP with explicit matrix-memory dynamics governed by CIFG (Coupled Input-Forget Gate) cells, enabling the network to store and retrieve knowledge through invertible matrix operations.

---

## 2. Energy Function

The total energy of the system is defined as:

```
E(C, H, alpha; theta) = E_pred + E_constraint + E_entropy + E_sparsity
```

where:
- `C` is the set of memory matrices (global + per-expert)
- `H` is the set of hidden state activations
- `alpha` is the expert gating distribution
- `theta` is the set of all learnable parameters

### 2.1 Prediction Energy

```
E_pred = ||y - f(x; theta)||^2
```

Standard squared prediction error. The function `f(x; theta)` represents the full forward pass through the BURT architecture, including Boolean constraint enforcement and memory retrieval.

### 2.2 Spectral Constraint Energy

```
E_constraint = lambda * max(0, sigma_max(W) - lambda_max)
```

where:
- `sigma_max(W)` is the largest singular value of weight matrix `W`
- `lambda_max` is the spectral radius bound (default: 0.95)
- `lambda` is the constraint penalty strength

This term ensures all weight matrices remain within a bounded spectral radius, preventing gradient explosion and guaranteeing convergence of the equilibrium dynamics.

### 2.3 Entropy Energy

```
E_entropy = -sum(alpha_k * log(alpha_k))    for k = 1, ..., K
```

**Bounded by 0.20**: The entropy of the expert gating distribution is constrained to not exceed 0.20 nats. This prevents the router from distributing load uniformly across all experts (which would defeat the purpose of sparse mixture-of-experts) while ensuring at least some diversity in expert selection.

The bound is enforced via projection after each update:
```
if H(alpha) > 0.20:
    alpha <- project_entropy(alpha, bound=0.20)
```

### 2.4 Sparsity Energy

```
E_sparsity = ||alpha||_1
```

The L1 norm of the gating vector encourages sparse expert activation. Combined with Gates normalization, this ensures that only a small subset of experts are active for any given input, maintaining computational efficiency and interpretability.

---

## 3. Two-Phase Dynamics

MMEP operates in two distinct phases that together define the learning signal.

### 3.1 Free Phase

In the free phase, the network relaxes to its natural equilibrium without external forcing:

```
dh/dt = -dE/dh
```

The hidden states `h` evolve according to the negative gradient of the energy function with respect to themselves. This is iterated for `T_free` steps (default: 20) until approximate convergence:

```
for t in range(T_free):
    h <- h - eta_h * dE/dh
```

At convergence, the network reaches a fixed point `h_free` that represents its best prediction given the current parameters.

**Convergence criterion:** `||dh/dt|| < epsilon` where `epsilon = 1e-5`.

### 3.2 Nudged Phase

In the nudged phase, the output layer is clamped toward the target with strength `beta`:

```
dh/dt = -dE/dh - beta * (h_output - y_target)
```

Only the output layer receives the nudging signal. The rest of the network adjusts to accommodate this external forcing, reaching a new equilibrium `h_nudged` after `T_nudge` steps (default: 4):

```
for t in range(T_nudge):
    h <- h - eta_h * (dE/dh + beta * clamp_signal)
```

The nudging strength `beta` controls the trade-off between biological plausibility (small beta) and learning speed (large beta). In the limit `beta -> 0`, the learning rule becomes exactly the gradient of the energy function.

---

## 4. Local Learning Rule

The weight update is computed from the difference between nudged and free equilibrium states:

```
dW = (1/beta) * (h_nudged * h_nudged^T - h_free * h_free^T)
```

This is a purely local, Hebbian-style rule: each synapse only needs access to the pre- and post-synaptic activities in both phases. No backpropagation of error signals through the network is required.

**Properties:**
- In the limit `beta -> 0`: `dW` converges to the true gradient `dE/dtheta`
- For finite `beta`: `dW` is a biased but consistent estimator of the gradient
- The rule is symmetric: `dW = dW^T` for symmetric connections
- Memory matrices are updated by the same rule applied to their input/output pairs

### 4.1 Bias Updates

```
db = (1/beta) * (h_nudged - h_free)
```

### 4.2 Memory Matrix Updates

For each memory matrix `C`:
```
dC = (1/beta) * (h_nudged * h_nudged^T - h_free * h_free^T) projected onto rank-1 update
```

---

## 5. CIFG Freeze Dynamics

The Coupled Input-Forget Gate (CIFG) controls how memory matrices are updated, preventing catastrophic forgetting:

```
C_new = f * C_old + (1 - f) * candidate
```

where:
```
f = sigma(W_f * [h, x])
```

- `f` is the forget gate (values in [0, 1])
- `sigma` is the sigmoid function
- `[h, x]` is the concatenation of hidden state and input
- `candidate` is the proposed new memory content

**Key insight:** When `f -> 1`, the memory is frozen (old content preserved). When `f -> 0`, the memory is fully overwritten. The CIFG couples the input and forget gates (input gate = `1 - f`), halving the parameter count compared to standard LSTM gates while maintaining expressivity.

### 5.1 Freeze Schedule

During training, the forget gate bias is gradually increased:
```
bias_f(epoch) = bias_f_init + freeze_rate * epoch
```

This encourages early plasticity followed by later consolidation, mimicking biological memory formation.

---

## 6. Memory Matrices

### 6.1 Global Memory: C_global

A single shared memory matrix accessible by all experts and all layers:
- Shape: `[d_model, d_model]`
- Stores corpus-level knowledge (facts, patterns, invariants)
- Updated during Phase 1 (memorization) and Phase 3 (convergence)
- Protected by high forget gate bias after initial training

### 6.2 Expert Memory: C_expert_k

Per-expert memory matrices (one per expert in the MoE layer):
- Shape: `[d_expert, d_expert]` for each expert `k = 1, ..., K`
- Stores expert-specific knowledge
- Updated only when the expert is selected by the router
- Lower forget gate bias allows more rapid adaptation

### 6.3 Memory Retrieval

Given a query vector `q`, retrieval from memory `C` is:
```
v = C * q / ||C * q||
```

Normalized retrieval ensures bounded activations regardless of memory content magnitude.

---

## 7. Convergence Guarantee

**Theorem (Lyapunov Convergence):** Under the MMEP dynamics with spectral constraint `sigma_max(W) <= lambda_max < 1`, the energy function `E(C, H, alpha; theta)` is a Lyapunov function for the free-phase dynamics.

**Proof sketch:**

Define `V(h) = E(C, H, alpha; theta)` evaluated at the current state.

1. **Bounded below:** `V(h) >= 0` since all terms are non-negative (squared error, max with 0, bounded entropy, L1 norm).

2. **Monotonically decreasing:** Along trajectories of `dh/dt = -dE/dh`:
   ```
   dV/dt = (dE/dh)^T * (dh/dt) = -||dE/dh||^2 <= 0
   ```

3. **Strict decrease:** `dV/dt = 0` only at fixed points where `dE/dh = 0`.

4. **Bounded trajectories:** The spectral constraint ensures `||h(t)|| <= M` for some finite `M`, since:
   ```
   ||Wh|| <= sigma_max(W) * ||h|| <= lambda_max * ||h|| < ||h||
   ```
   for `lambda_max < 1`, guaranteeing contraction.

By LaSalle's invariance principle, all trajectories converge to the set of equilibria. The spectral constraint guarantees uniqueness of the equilibrium for each input, completing the proof.

**Convergence rate:** Exponential with rate `1 - lambda_max`. For the default `lambda_max = 0.95`, convergence to machine precision requires approximately `T_free = ceil(log(epsilon) / log(0.95)) = 98` steps, though in practice `T_free = 20` suffices for the learning rule.

---

## 8. Constraint Projection

After each parameter update, constraints are enforced by projection:

### 8.1 Spectral Projection

```python
def project_spectral(W, lambda_max=0.95):
    U, S, V = svd(W)
    S_clipped = minimum(S, lambda_max)
    return U @ diag(S_clipped) @ V^T
```

### 8.2 Entropy Projection

```python
def project_entropy(alpha, bound=0.20):
    if entropy(alpha) <= bound:
        return alpha
    # Binary search for temperature tau such that
    # H(softmax(log(alpha) / tau)) = bound
    tau = binary_search_temperature(alpha, bound)
    return softmax(log(alpha) / tau)
```

### 8.3 Combined Projection

```python
def project_all(params):
    for W in params.weight_matrices:
        W <- project_spectral(W)
    for alpha in params.gating_vectors:
        alpha <- project_entropy(alpha)
    # Sparsity is enforced via L1 penalty, not projection
```

---

## 9. Relationship to Standard EP

| Property | Standard EP | MMEP |
|----------|------------|------|
| Energy function | `E = ||y - f(x)||^2` | `E = E_pred + E_constraint + E_entropy + E_sparsity` |
| Memory | Implicit in weights | Explicit matrix memory (C_global, C_expert) |
| Constraints | None | Spectral, entropy, sparsity |
| Convergence | Assumed | Guaranteed via Lyapunov + spectral bound |
| Gating | None | CIFG freeze dynamics |
| Expert routing | None | Sparse MoE with Gates normalization |
| Learning rule | Same | Same form: `(1/beta)(nudged - free)` |

---

## 10. Notation Summary

| Symbol | Meaning |
|--------|---------|
| `E` | Total energy |
| `C` | Memory matrix (global or expert) |
| `H` | Hidden states |
| `alpha` | Expert gating distribution |
| `theta` | All learnable parameters |
| `sigma_max(W)` | Largest singular value of W |
| `lambda_max` | Spectral radius bound (default 0.95) |
| `beta` | Nudging strength |
| `f` | Forget gate value |
| `T_free` | Free phase steps (default 20) |
| `T_nudge` | Nudged phase steps (default 4) |
| `K` | Number of experts |
| `d_model` | Model dimension |
| `d_expert` | Expert dimension |

---

## References

1. Scellier, B. & Bengio, Y. (2017). "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation." Frontiers in Computational Neuroscience.
2. Greff, K. et al. (2017). "LSTM: A Search Space Odyssey." IEEE TNNLS.
3. Fedus, W. et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity."
4. Miyato, T. et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.
