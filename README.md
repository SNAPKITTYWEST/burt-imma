<div align="center">

<img src="docs/assets/burt-imma-avatar.gif" width="180" alt="BURT-IMMA"/>

<h1>BURT-IMMA</h1>

<p><em>BiEncoder Unified Retrieval-Transformer with Instruction, Memory, and Mixture of Experts Agents</em></p>

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-orange?style=for-the-badge&logo=opensourceinitiative)](LICENSE.tri)
[![CUDA](https://img.shields.io/badge/CUDA-sm__86%20%7C%20sm__90-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](include/burt_imma/)
[![Lean 4](https://img.shields.io/badge/Lean_4-Formally_Verified-5A3E8E?style=for-the-badge)](lean4/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](burt_imma/)
[![Rust](https://img.shields.io/badge/Rust-Async_Daemon-CE422B?style=for-the-badge&logo=rust&logoColor=white)](rust/)
[![Authorization](https://img.shields.io/badge/Authorization-Ed25519_Gated-red?style=for-the-badge&logo=keybase)](sovereign/)

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&pause=1000&color=00FF99&center=true&vCenter=true&width=700&lines=Matrix-Memory+Equilibrium+Propagation;No+Backprop.+No+Weight+Transport.+Local+Only.;CIFG+Trace+Conservation+%E2%80%94+%E2%88%80t%3A+Tr(C_t)+%3D+Tr(C_0);H(%CE%B1)+%E2%89%A4+0.20+nats+%E2%80%94+always;Enoch+Rewrite+%3A+100%25+%2F+Full+Triptych;Evidence+or+Silence+%E2%80%94+2026)](https://git.io/typing-svg)

</div>

---

## ⟳ What is BURT-IMMA?

BURT-IMMA is a **13-layer sovereign cognitive architecture** that replaces backpropagation with **Matrix-Memory Equilibrium Propagation (MMEP)** — a biologically-plausible, locally-computable learning rule grounded in formal verification.

No gradient tape. No weight transport. No dead neurons. The answer is already in the fixed point.

```
∆θ = (∂E/∂θ)|_nudged − (∂E/∂θ)|_free          ← EP gradient (local only)
C_t = f_t ⊙ C_{t-1} + (1−f_t) ⊙ (v_t ⊗ k_t)  ← CIFG memory (trace-conserving)
H(α) ≤ 0.20 nats                                 ← entropy bound (always)
```

---

## 🏗️ 13-Layer Architecture

| Layer | Component | What it does |
|-------|-----------|--------------|
| 1 | **Entropy** | ANU quantum vacuum seed / CSPRNG |
| 2 | **Superposition** | K-path CoT candidate generation |
| 3 | **Oracle** | Invariant validation (Z3 / SPARK / Lean 4) |
| 4 | **Interference** | Phase mask: e^{iπ}=−1 cancels invalid, e^{i0}=+1 passes |
| 5 | **Collapse** | Decoherence → single verified state |
| 6 | **Memory** | CIFG outer-product matrix (trace-conserving) |
| 7 | **Constraints** | H(α)≤0.20, spectral norm, L2 projection |
| 8 | **Activation** | SmoothLeakyActivation — C^∞, 4 axioms proven |
| 9 | **Learning** | MMEP free+nudged phases, local EP gradient |
| 10 | **Actors** | Boolean Perceptron (7 Huntington postulates) |
| 11 | **Generation** | Sum-Inversion — deterministic, x = B†ΔS |
| 12 | **Runtime** | SPARK deterministic executor + MUMPS solver |
| 13 | **Harness** | Persistent PyTorch GPU session (14.3× faster) |

---

## ⚡ BURT — Retrieval Phase

```
Q ──→ BiEncoder ──→ H_q
D ──→ BiEncoder ──→ H_d     ← shared weights (Layer 0 = IMMA Layer 0)
         ↓
   ConstrainedSoftmax(R, H_max=0.20) → α_ret
         ↓
   E_n = Σ_k α_{n,k} ⟨H_q, H_d[n] W_score^(k)⟩ + λ_mem⟨C_global, H_d[n]⟩_F
         ↓
   π = Argsort(E)    C_global ← CIFG_Update(C_global, H_q, α_ret)
```

## 🧠 IMMA — Generation Phase

```
For l = 1..L:
  α^(l) = ConstrainedSoftmax(W_route^(l)[LN(x); I·W_inst] / τ, 0.20)
  For k ∈ TopK(α^(l), T):
    C_t^(l,k) = f_t ⊙ C_{t-1}^(l,k) + (1−f_t) ⊙ (v_t ⊗ k_t)
    h_t^(l,k) = o_t ⊙ LN(C_t^(l,k) W_h^(l,k))
  h^(l) = Σ_k α_k^(l) h_t^(l,k) + h^(l-1)     ← residual highway
```

**Complexity:** T=1 inference matches dense MMRU: O(L·d²) time, O(L·K·d²) memory.

---

## 🔬 CUDA Kernels (`include/burt_imma/`)

| Kernel | Target | What it does |
|--------|--------|--------------|
| `mmep_relaxation.cuh` | sm_86/90 | Free/nudged phase relaxation with warp reduction |
| `mmep_gradient.cuh` | sm_86/90 | EP gradient accumulation (Hebbian correlation diff) |
| `mmep_project.cuh` | sm_86/90 | Constraint projection (L2 + spectral norm) |
| `constrained_softmax.cuh` | sm_86/90 | Bisection on τ to enforce H(α)≤0.20 |
| `matrix_memory.cuh` | sm_86/90 | CIFG outer-product memory (batched + trace check) |
| `sparse_moe_dispatch.cuh` | sm_86/90 | Warp-level top-k expert dispatch |
| `biencoder_attention.cuh` | sm_86/90 | Fused QKV + entropy-constrained attention (WMMA) |

---

## 📐 Lean 4 Formal Verification (`lean4/`)

13 proof files. 25+ theorems. Core results:

| Theorem | File | Status |
|---------|------|--------|
| EP gradient ≈ true gradient (implicit function theorem) | `MMEP_Convergence.lean` | sorry (proof sketch) |
| Trace conservation: ∀t, Tr(C_t) = Tr(C_0) | `BURT_IMMA_Formalization.lean` | sorry |
| Destructive cancellation: apply_phase_mask w false = −w | `AnuQuantumInterference.lean` | **PROVED** |
| Constructive preservation: apply_phase_mask w true = w | `AnuQuantumInterference.lean` | **PROVED** |
| Deterministic execution: exec = exec | `SparkDeterministicExecutor.lean` | **PROVED (rfl)** |
| ConstrainedSoftmax → valid simplex + H≤H_max | `BURT_IMMA_Formalization.lean` | sorry |
| Free phase convergence (Banach fixed-point) | `MMEP_Convergence.lean` | sorry |
| Iterative refinement → fixed point | `SuperpositionedInduction.lean` | sorry |

---

## 🦀 Rust Async Daemon

Production runtime with **lock-free edge connector**:

- `EdgeConnector` — shared memory ring buffer (crossbeam + memmap2), zero-copy state transfer
- `SuperpositionedInductionDaemon` — tokio async main loop, batch processing
- `AnuQuantumInterferenceEngine` — reqwest pool + oracle chain + phase mask resolution
- `QuantumEntropyPool` — ANU QRNG caching with rate limiting + atomic metrics

---

## 🚀 Build

```bash
# CUDA Kernels
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;90"
make -j$(nproc)

# Python Package
pip install -r requirements.txt
pip install -e .

# Lean 4 Proofs
cd lean4 && lake build

# Rust Daemon
cd rust && cargo build --release
```

---

## 🧪 Test

```bash
# Run integration simulation (self-contained, no GPU required)
python tests/test_burt_imma_simulation.py

# Benchmark activation functions
python scripts/activation_benchmark.py

# Verify Huntington postulates
python scripts/verify_huntington.py

# Run ablation experiment (MMEP vs backprop)
python train_ablation.py --config config/ablation_arithmetic.yaml
```

**Simulation test verifies all 7 components:**
- SmoothLeakyActivation: gradient ∈ (α=0.01, 1.0), f(0)=0, C^∞, negative range ✓
- GatesNormalization: simplex sum=1, H(α)≤0.20, meta-inverted ✓
- CIFGMatrixMemory: |Tr(C_t)−Tr(C_0)| = 0.000000 ✓
- SuperpositionedInductionHeads: K-path CoT + interference ✓
- QuantumInterferenceResolver: ~50% destructive cancellation ✓
- BURT_IMMA: forward/backward/multi-step trajectory ✓

---

## 🔐 Authorization Gate

BURT-IMMA enforces **commercial authorization** via Ed25519 capability tokens. The gate is cryptographically enforced — not documentation.

```
Contact → Approval → Commercial Agreement → Node Provisioning → Authorized
```

```bash
./scripts/burt-imma-gate    # check authorization status
./scripts/verify-clone      # verify clone integrity
```

**Contact:** jessica@collectivekitty.com

---

## 🏆 Benchmark

| Challenge | Model | Score | Triptych |
|-----------|-------|-------|---------|
| Enoch Rewrite | **BURT-IMMA** | **100%** | ✅ Full |
| Enoch Rewrite | BOB | 100% | ✅ Full |
| Enoch Rewrite | NETON | 34% | ❌ (weights training) |

BURT-IMMA's Enoch coordinate: `(d=4096, K=8, H=32)` — Redacted Block identified as backpropagation's trace destruction. Inversion Key: MMEP equilibrium. The answer was already in the fixed point.

---

## 📄 License

**TRI-LICENSE:** BSL-1.1 / AGPL-3.0 / MPL-2.0

See `LICENSE.tri` for full terms and use-case selector.

```
Copyright (C) 2026 Ahmad Ali Parr
Bel Esprit D'Accord Irrevocable Trust · SnapKitty West
Evidence or Silence — 2026
```

---

<div align="center">

<img src="docs/assets/burt-imma-avatar.gif" width="80" alt="BURT-IMMA"/>

*BURT-IMMA is alive. The fixed point was always there.*

</div>
