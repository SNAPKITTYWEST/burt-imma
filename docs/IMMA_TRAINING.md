# IMMA: Inductive Matrix-Memory Architecture Training Protocol

**Project:** BURT-IMMA  
**Contact:** jessica@collectivekitty.com  
**License:** BSL-1.1

---

## 1. Overview

IMMA (Inductive Matrix-Memory Architecture) defines the complete training protocol for BURT models. Training proceeds in four sequential phases, each building on the previous phase's outputs. The protocol is designed to be reproducible, falsifiable, and hardware-efficient (single GPU).

---

## 2. Phase 1: Corpus Memorization

**Goal:** Store factual knowledge into `C_global` via CIFG gating.

### 2.1 Objective

Minimize reconstruction loss while filling the global memory matrix:

```
L_phase1 = ||x_reconstructed - x_original||^2 + lambda_spectral * E_constraint
```

### 2.2 Procedure

```
for each document d in corpus:
    x = tokenize(d)
    for each chunk in sliding_window(x, window_size=512, stride=256):
        # Forward pass (no routing, no Boolean constraints)
        h = embed(chunk)
        h = transformer_forward(h)  # Attention only, no MoE
        
        # CIFG update to global memory
        f = sigmoid(W_f @ [h_mean, chunk_embed] + b_f)
        candidate = outer(h_mean, h_mean)  # Rank-1
        C_global = f * C_global + (1 - f) * candidate
        
        # Reconstruction objective
        x_hat = decode(C_global @ h_mean)
        loss = mse(x_hat, chunk)
        
        # Update only: embedding, attention, W_f, b_f, decoder
        optimizer.step(loss)
```

### 2.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | High LR for rapid memorization |
| Forget gate bias init | -2.0 | Low bias = high plasticity |
| Batch size | 32 | Memory-efficient |
| Window size | 512 | Context length |
| Stride | 256 | 50% overlap for continuity |
| Epochs | 1 | Single pass (memorization, not fitting) |
| Spectral lambda | 0.1 | Light constraint during memorization |

### 2.4 Completion Criteria

- Memory utilization > 60% (measured by effective rank of C_global)
- Reconstruction loss < 0.5
- Spectral norm of C_global < lambda_max

### 2.5 Outputs

- Checkpoint: `phase1_memorized.pt`
- Contains: C_global (filled), embedding weights, attention weights
- Frozen for Phase 2: C_global (read-only during query training)

---

## 3. Phase 2: Query Training

**Goal:** Train expert routing and per-expert memories using question-answer pairs.

### 3.1 Objective

```
L_phase2 = L_qa + lambda_entropy * E_entropy + lambda_sparsity * E_sparsity + lambda_balance * L_load_balance
```

where:
- `L_qa = cross_entropy(predicted_answer, true_answer)`
- `L_load_balance = K * sum(fraction_k * routing_prob_k)` (auxiliary load balancing)

### 3.2 Procedure

```
# Load Phase 1 checkpoint
model.load("phase1_memorized.pt")
model.C_global.requires_grad = False  # Freeze global memory

for each (query, answer) in qa_dataset:
    x = tokenize(query)
    h = model.forward_through_attention(x)
    
    # Route through experts
    expert_ids, weights = GatesRouter(h)
    
    # Expert forward (updates C_expert)
    for k in expert_ids:
        h_k = Expert_k(h)
        f_k = sigmoid(W_f_k @ [h_k, h] + b_f_k)
        candidate_k = outer(h_k, h_k)
        C_expert[k] = f_k * C_expert[k] + (1 - f_k) * candidate_k
    
    # Combine expert outputs
    output = QuantumInterferenceResolver(expert_outputs, weights)
    
    # Compute loss
    logits = output_head(output)
    loss = cross_entropy(logits, answer_tokens) + auxiliary_losses
    
    # Update: experts, router, W_f_k, output head
    optimizer.step(loss)
    
    # Project constraints
    project_entropy(router.alpha, bound=0.20)
    for W in all_weight_matrices:
        project_spectral(W, lambda_max=0.95)
```

### 3.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 3e-4 | Moderate for stable routing |
| Forget gate bias init (expert) | 0.0 | Neutral plasticity |
| Batch size | 16 | Larger effective batch via accumulation |
| Gradient accumulation | 4 | Effective batch = 64 |
| Entropy bound | 0.20 | Sparse routing |
| Entropy lambda | 0.01 | Soft enforcement |
| Sparsity lambda | 0.001 | Encourage L1 sparsity |
| Balance lambda | 0.01 | Prevent expert collapse |
| Epochs | 10 | Until routing stabilizes |

### 3.4 Completion Criteria

- QA accuracy > 70% on validation set
- Router entropy consistently < 0.20
- No expert receives < 10% of traffic (no collapse)
- Expert memory utilization > 30% for each active expert

### 3.5 Outputs

- Checkpoint: `phase2_query_trained.pt`
- Contains: All Phase 1 weights + expert weights + router + C_expert matrices
- Routing statistics saved for analysis

---

## 4. Phase 3: Equilibrium Convergence

**Goal:** Full MMEP training with all constraints active simultaneously.

### 4.1 Objective

Full MMEP energy minimization:
```
E(C, H, alpha; theta) = E_pred + E_constraint + E_entropy + E_sparsity
```

All components active. Training proceeds via equilibrium propagation (not backpropagation).

### 4.2 Procedure

```
# Load Phase 2 checkpoint
model.load("phase2_query_trained.pt")
model.C_global.requires_grad = True  # Unfreeze global memory

for each (x, y) in training_data:
    # === FREE PHASE ===
    h = model.embed(x)
    for t in range(T_free):
        h = h - eta_h * dE_dh(h, model.params)
        if norm(dE_dh) < 1e-5:
            break
    h_free = h.clone()
    
    # === NUDGED PHASE ===
    h = h_free.clone()
    for t in range(T_nudge):
        h = h - eta_h * (dE_dh(h, model.params) + beta * nudge_signal(h, y))
    h_nudged = h.clone()
    
    # === LOCAL LEARNING RULE ===
    for W in model.weight_matrices:
        dW = (1/beta) * (h_nudged @ h_nudged.T - h_free @ h_free.T)
        W = W - lr * dW
    
    # === MEMORY UPDATE ===
    # CIFG update using equilibrium states
    f = sigmoid(W_f @ [h_nudged, x_embed] + b_f)
    C_global = f * C_global + (1 - f) * outer(h_nudged_mean, h_nudged_mean)
    
    # === CONSTRAINT PROJECTION ===
    for W in model.weight_matrices:
        project_spectral(W, lambda_max=0.95)
    project_entropy(model.router.alpha, bound=0.20)
    
    # === HUNTINGTON ENFORCEMENT ===
    huntington_loss = verify_huntington(model.boolean_layer.output)
    if huntington_loss > tolerance:
        project_boolean(model.boolean_layer)
```

### 4.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Low for stable convergence |
| Beta (nudging) | 0.1 -> 0.01 | Annealed for precision |
| T_free | 20 | Sufficient for convergence |
| T_nudge | 4 | Short nudge for local signal |
| eta_h (state LR) | 0.5 | Fast state dynamics |
| Spectral lambda_max | 0.95 | Hard bound |
| Entropy bound | 0.20 | Hard bound |
| Batch size | 8 | EP is memory-intensive |
| Epochs | 50 | Until energy stabilizes |
| Huntington tolerance | 1e-3 | Tight Boolean compliance |

### 4.4 Beta Annealing Schedule

```
beta(epoch) = beta_max * (1 - epoch / total_epochs) + beta_min
```
- `beta_max = 0.1` (early: strong learning signal)
- `beta_min = 0.01` (late: precise gradient approximation)

### 4.5 Completion Criteria

- Energy function monotonically decreasing for 5 consecutive epochs
- `||h_free(t+1) - h_free(t)|| < 1e-5` (equilibrium reached)
- Spectral norm of all W < lambda_max (no violations)
- Entropy < 0.20 (no violations)
- Huntington postulates satisfied within tolerance
- Validation loss not increasing (no overfitting)

### 4.6 Outputs

- Checkpoint: `phase3_converged.pt`
- Contains: Full model with converged parameters
- Energy trajectory saved for analysis
- Convergence diagnostics

---

## 5. Phase 4: Ablation Validation

**Goal:** Verify that each component contributes to performance; check falsification criteria.

### 5.1 Ablation Configurations

| Config | Description | What is removed |
|--------|-------------|-----------------|
| `full_mmep` | Complete model | Nothing (baseline) |
| `no_memory` | Remove C_global and C_expert | Matrix memory |
| `no_constraint` | Remove spectral + entropy bounds | All constraints |
| `no_moe` | Single expert (no routing) | Mixture of experts |
| `no_boolean` | Remove Huntington enforcement | Boolean layer |
| `no_interference` | Replace QIR with weighted avg | Quantum resolver |
| `standard_bp` | Train with backpropagation | EP training |
| `no_cifg` | Replace CIFG with simple EMA | Gated memory |

### 5.2 Procedure

```
for each config in ablation_configs:
    # Initialize from Phase 2 checkpoint (pre-EP)
    model = load_with_ablation("phase2_query_trained.pt", config)
    
    # Train (Phase 3 or backprop depending on config)
    if config == "standard_bp":
        train_backprop(model, training_data, epochs=50)
    else:
        train_mmep(model, training_data, epochs=50)
    
    # Evaluate
    metrics = evaluate(model, test_data)
    save_results(config, metrics)
```

### 5.3 Evaluation Metrics

For each configuration, measure:

1. **Accuracy:** Task-specific accuracy on held-out test set
2. **Convergence speed:** Epochs to reach 90% of final performance
3. **Entropy:** Average routing entropy (for MoE configs)
4. **Spectral norm:** Maximum singular value across all layers
5. **Memory utilization:** Effective rank of memory matrices / max rank
6. **Huntington compliance:** Fraction of inputs satisfying all postulates
7. **Energy trajectory:** E(t) over training (for EP configs)
8. **Inference latency:** Wall-clock time per sample

### 5.4 Falsification Criteria

The MMEP theory is falsified if ANY of the following hold:

1. **`standard_bp` > `full_mmep` on accuracy:** EP training provides no benefit
2. **`no_memory` >= `full_mmep` on accuracy:** Matrix memory is unnecessary
3. **`no_constraint` > `full_mmep` on accuracy:** Constraints hurt performance
4. **Energy is non-monotonic in Phase 3:** Lyapunov guarantee violated
5. **Spectral norm exceeds lambda_max post-projection:** Projection is broken
6. **Entropy exceeds 0.20 post-projection:** Entropy projection is broken

If any falsification criterion is triggered, the corresponding theoretical claim must be revised or abandoned.

### 5.5 Statistical Significance

- Each configuration run 3 times with different random seeds
- Report mean and standard deviation
- Use paired t-test (p < 0.05) for comparisons against `full_mmep`
- Bonferroni correction for multiple comparisons

---

## 6. Hyperparameter Schedule (Complete)

### 6.1 Learning Rate Schedule

```
Phase 1: constant 1e-3
Phase 2: cosine decay from 3e-4 to 1e-5 over 10 epochs
Phase 3: linear warmup (1 epoch) then cosine decay from 1e-4 to 1e-6 over 49 epochs
Phase 4: same as Phase 3 for each ablation run
```

### 6.2 Constraint Schedule

```
Epoch 0-5:   lambda_spectral = 0.01 (soft)
Epoch 5-10:  lambda_spectral = 0.1 (moderate)
Epoch 10+:   lambda_spectral = 1.0 (hard) + projection

Epoch 0-5:   entropy_bound = 0.50 (relaxed)
Epoch 5-10:  entropy_bound = 0.35 (tightening)
Epoch 10+:   entropy_bound = 0.20 (final)

Epoch 0-20:  huntington_tolerance = 0.1 (soft)
Epoch 20-40: huntington_tolerance = 0.01 (moderate)
Epoch 40+:   huntington_tolerance = 0.001 (hard)
```

### 6.3 Memory Freeze Schedule

```
C_global:
  Phase 1: fully plastic (forget bias = -2.0)
  Phase 2: frozen (not updated)
  Phase 3: slowly plastic (forget bias = +2.0, rarely updates)

C_expert:
  Phase 1: not used
  Phase 2: plastic (forget bias = 0.0)
  Phase 3: consolidating (forget bias increasing by 0.1/epoch)
```

---

## 7. Checkpoint Strategy

### 7.1 Checkpoint Contents

Each checkpoint saves:
```
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "C_global": ...,
    "C_expert": [...],
    "epoch": ...,
    "phase": ...,
    "energy_history": [...],
    "metrics": {...},
    "config": {...},
    "random_state": ...,
}
```

### 7.2 Checkpoint Schedule

| Event | Filename Pattern | Retention |
|-------|-----------------|-----------|
| Phase completion | `phase{N}_complete.pt` | Permanent |
| Best validation | `best_val_{metric}.pt` | Keep top 3 |
| Periodic | `checkpoint_epoch_{E}.pt` | Every 5 epochs, keep last 3 |
| Pre-ablation | `pre_ablation.pt` | Permanent |
| Ablation result | `ablation_{config}.pt` | Permanent |

### 7.3 Checkpoint Validation

Before saving:
```python
def validate_checkpoint(model):
    assert all(spectral_norm(W) <= lambda_max for W in model.weights)
    assert entropy(model.router.alpha) <= 0.20 + 1e-5
    assert effective_rank(model.C_global) > 0  # Memory not degenerate
    assert model.C_global.isfinite().all()  # No NaN/Inf
```

---

## 8. Evaluation Metrics (Detailed)

### 8.1 Primary Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | correct / total | > 85% (task-dependent) |
| Energy | E(C, H, alpha; theta) | Monotonically decreasing |
| Convergence | epochs to 90% accuracy | < 30 epochs |

### 8.2 Constraint Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Max spectral norm | max_W(sigma_max(W)) | <= 0.95 |
| Routing entropy | -sum(alpha * log(alpha)) | <= 0.20 |
| Huntington violation | mean(huntington_loss) | < 0.001 |
| Sparsity | mean(||alpha||_1) | Decreasing |

### 8.3 Memory Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Memory utilization | effective_rank(C) / d | > 0.6 after Phase 1 |
| Memory stability | ||C(t) - C(t-1)||_F | Decreasing in Phase 3 |
| Condition number | sigma_max(C) / sigma_min(C) | < 100 |
| Retrieval accuracy | cosine(C@q, expected) | > 0.9 |

### 8.4 Diagnostic Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Gradient norm (EP) | ||(1/beta)(nudged - free)|| | Bounded, decreasing |
| Expert load balance | std(expert_usage) / mean(expert_usage) | < 0.3 |
| Free phase convergence | steps to ||dh/dt|| < 1e-5 | < T_free |
| Nudge sensitivity | ||h_nudged - h_free|| / beta | Bounded |

---

## 9. Hardware and Runtime

### 9.1 Minimum Requirements

- GPU: NVIDIA RTX 3080 (10 GB VRAM) or better
- RAM: 32 GB
- Storage: 50 GB (checkpoints + data)
- CUDA: 11.8+

### 9.2 Expected Runtime

| Phase | Duration (RTX 3080) | VRAM Usage |
|-------|---------------------|------------|
| Phase 1 | 30 min | 4 GB |
| Phase 2 | 2 hours | 6 GB |
| Phase 3 | 8 hours | 8 GB |
| Phase 4 | 4 hours (all ablations) | 8 GB |
| **Total** | **~15 hours** | **8 GB peak** |

### 9.3 Memory Optimization

- Gradient checkpointing in Phase 3 (EP requires storing both equilibria)
- Mixed precision (fp16) for forward pass, fp32 for EP dynamics
- Memory matrices always fp32 (numerical stability critical)
- Batch size auto-tuned to available VRAM

---

## 10. Reproducibility

### 10.1 Random Seeds

```python
SEEDS = [42, 137, 256]  # Three runs per configuration
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

### 10.2 Version Pinning

All dependencies pinned in `requirements.txt`. Key versions:
- PyTorch >= 2.0
- CUDA >= 11.8
- NumPy >= 1.24

### 10.3 Data Determinism

- Training data order fixed by seed
- No non-deterministic augmentation in Phases 2-4
- Validation set is static (no shuffling)
