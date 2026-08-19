# BURT-IMMA

**BiEncoder Unified Retrieval-Transformer with Instruction, Memory, and Mixture of Experts Agents**

Matrix-Memory Equilibrium Propagation (MMEP) implementation targeting NVIDIA RTX 3080 (sm_86).

## What is MMEP?

Matrix-Memory Equilibrium Propagation is a biologically-plausible alternative to backpropagation:

1. **Free phase**: Network relaxes to energy equilibrium (T_free steps)
2. **Nudged phase**: Target signal perturbs the equilibrium (T_nudge steps)
3. **EP gradient**: Difference in correlations between phases gives the true gradient

Key equations:
```
h_{t+1} = (1-alpha) * h_t + alpha * sigma(W @ h_{t-1} + C_global + C_expert_k)
dW = (1/beta) * (h_nudged @ h_prev_nudged^T - h_free @ h_prev_free^T)
C_t = f_t * C_{t-1} + (1-f_t) * (v_t @ k_t^T)   [CIFG memory]
```

## Architecture

BURT-IMMA is a 13-layer unified architecture:

| Layer | Component | Function |
|-------|-----------|----------|
| 1 | Entropy | ANU quantum / CSPRNG seeding |
| 2 | Superposition | Multi-path candidate generation |
| 3 | Oracle | Invariant validation (Z3/SPARK/Lean) |
| 4 | Interference | Phase mask (constructive/destructive) |
| 5 | Collapse | Decoherence to single state |
| 6 | Memory | CIFG matrix update (outer-product) |
| 7 | Constraints | Entropy H(a)<=0.20, spectral norm, L2 |
| 8 | Activation | SmoothLeaky (4 axioms, C^inf) |
| 9 | Learning | MMEP (free + nudged + EP gradient) |
| 10 | Actors | Boolean Perceptron (Huntington postulates) |
| 11 | Generation | Sum-Inversion (deterministic decoding) |
| 12 | Runtime | SPARK executor (contract validation) |
| 13 | Harness | PyTorch persistent session |

### BURT (Retrieval Phase)
- BiEncoder: encode query + documents separately
- Entropy-constrained routing: H(alpha) <= 0.20
- CIFG memory: C_t = f*C + (1-f)*(v @ k^T)
- Evidence scoring with expert-specific weights

### IMMA (Generation Phase)
- MoE with CIFG matrix memory per expert
- Top-k routing with entropy constraint
- Complexity: T=1 O(L*d^2) time, O(L*K*d^2) memory
- CIFG constraint: i = 1 - f (coupled input-forget)

## CUDA Kernels

Target: NVIDIA sm_86 (RTX 3080, 10GB VRAM) + sm_90 (H100)

- `mmep_relaxation.cuh` — Free/nudged phase relaxation step
- `mmep_gradient.cuh` — EP gradient accumulation (Hebbian correlation difference)
- `mmep_project.cuh` — Constraint projection (L2 balls + spectral norm via power iteration)
- `constrained_softmax.cuh` — Bisection on temperature for entropy bound
- `matrix_memory.cuh` — CIFG outer-product memory (batched + shared)
- `sparse_moe_dispatch.cuh` — Warp-level top-k expert dispatch
- `biencoder_attention.cuh` — Fused QKV + entropy-constrained attention

## Lean 4 Convergence Proofs

8 core theorems (sorry-pending, structurally complete):

1. Energy bounded below (quadratic form argument)
2. Free phase decreases energy (gradient descent on E)
3. Equilibrium unique (Banach fixed-point, lambda_max < 1)
4. EP gradient = backprop gradient (implicit function theorem, beta->0)
5. Constraint projection non-expansive (convex set projection)
6. Memory retention stable (triangle inequality + spectral response)
7. Spectral norm ensures contraction (composition of contractions)
8. Full training converges (Robbins-Monro + projected SGD)

Additional formalizations: BURT state, IMMA state, Boolean perceptron,
MetaInvertedSum, SmoothLeaky activation, AlexNet bridge, Sum-Inversion,
SPARK executor, Superpositioned induction, Quantum interference.

## Build

### CUDA Kernels
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;90"
make -j$(nproc)
```

### Python Package
```bash
pip install -r requirements.txt
pip install -e .
```

### Lean 4 Proofs
```bash
cd lean4
lake build
```

## Run Ablation Experiment

```bash
# Generate arithmetic dataset
python scripts/generate_arithmetic.py --corpus-size 10000 --output data/

# Run ablation training
python train_ablation.py --config config/ablation_arithmetic.yaml

# Check gradient comparison (MMEP vs backprop)
python scripts/gradient_comparison.py --hidden-dim 256 --num-layers 4
```

## IMMA Complexity

| T (steps) | Time | Memory |
|-----------|------|--------|
| T=1 | O(L*d^2) | O(L*K*d^2) |
| T=2 | O(2L*d^2) | O(L*K*d^2) |

Key: T=1 inference latency matches dense MMRU.

## Authorization Gate

Protected operations require authorization via Ed25519 capability tokens.

```bash
# Check authorization status
./scripts/burt-imma-gate

# Verify clone integrity
./scripts/verify-clone
```

Contact for authorization: jessica@collectivekitty.com

## License

BSL-1.1 / AGPL-3.0 / MPL-2.0 (Tri-License). See LICENSE.tri.

Copyright (C) 2026 SnapKitty West / Bel Esprit D'Accord Irrevocable Trust.
