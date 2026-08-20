# The Architecture of Artificial Learning: From McCulloch-Pitts Neurons to BURT-IMMA

**Eighty years of iterative error correction — and a thirteen-layer architecture that forks the lineage: Matrix-Memory Equilibrium Propagation, a CIFG matrix-memory cell, CUDA kernels for a single RTX 3080, and a Lean 4 formal ledger**

Snap Kitty Collective — SnapKitty West  
jessica@collectivekitty.com &nbsp;&nbsp;&nbsp; *Evidence or Silence — 2026*

*Incorporating: Adversarial Mathematical Audit responses and corrected proof ledger (August 2026)*

---

## Abstract

**Executive Abstract.** Modern artificial intelligence rests on a single foundational mechanic: iterative error correction across distributed weights. From early biological abstractions to high-dimensional Transformer architectures, machine learning models derive their capability not from explicit programmatic rules but from statistical pattern convergence via gradient descent. Part I of this paper synthesizes that evolution — McCulloch-Pitts binary neurons, Rosenblatt's trainable Perceptron, the linear-separability bottleneck and the first AI winter, backpropagation's solution to the credit assignment problem, AlexNet's convolutional breakthrough, recurrent networks and the LSTM, self-attention and large language models, and the phenomenon of feature superposition in high-dimensional activation spaces. Part II presents the point where I fork that lineage. BURT-IMMA is a thirteen-layer architecture whose learning rule is Matrix-Memory Equilibrium Propagation (MMEP): a two-phase, locally computable, Hebbian-style update derived from the difference between a free equilibrium and a nudged equilibrium. No gradient tape. No weight transport. The answer is already in the fixed point. At its heart sits a direct LSTM descendant — a Coupled Input-Forget Gate (CIFG) cell whose state is a full $d \times d$ matrix, updated by gated outer products $C_t = f_t \odot C_{t-1} + (1-f_t) \odot (v_t \otimes k_t)$. I document every component: the CIFG matrix memory and its sum-inversion retrieval bounds, the MMEP energy function and convergence argument, the entropy-constrained softmax holding router entropy at $H(\alpha) \leq 0.20$ nats, the spectral projection that keeps relaxation contractive, the SmoothLeaky activation and its four axioms, Boolean perceptron actors under Huntington's postulates, superpositioned induction heads that make feature superposition an engineered mechanism rather than an emergent accident, the phase-mask interference resolver, the deterministic sum-inversion decoder, seven custom CUDA kernels for sm_86, and the Lean 4 formalization. Every claim carries an epistemic label — proved, verified, sketch, or hypothesis — and every theoretical commitment ships with a numbered falsification criterion.

**Audit note (August 2026).** An adversarial mathematical audit applied two independent review protocols — multi-agent debate and first-principles re-derivation — to the equations, theorems, and empirical claims of this paper and its source repository. Six findings were produced. This edition integrates those findings, presents the corrected claims, and documents which findings have been patched by subsequent Lean 4 machine-checked proofs. The corrected proof ledger appears in Section 12. The full audit responses appear in Section 12.1. *Evidence or silence.*

---

## PART I — THE LINEAGE: EIGHTY YEARS OF NUDGING WEIGHTS

---

## 1 The Foundational Mechanics of Artificial Neurons

### 1.1 The McCulloch-Pitts abstraction (1943)

The inception of computational connectionism began with Warren McCulloch and Walter Pitts [1]. They abstracted biological neurons into binary logic gates: discrete binary inputs $x_i \in \{0,1\}$, aggregation by summation against a fixed threshold $\theta$, and a binary activation state

$$y = \begin{cases} 1 & \text{if } \sum_i x_i \geq \theta \\ 0 & \text{otherwise.} \end{cases}$$

The McCulloch-Pitts unit could compute logic — AND, OR, NOT circuits assemble from it — but every parameter was fixed and hand-wired. It was a neuron that could *decide* but could not *learn*. Hold that distinction; Part II resurrects this exact unit, with its logic made algebraically checkable, as the routing layer of a modern architecture.

### 1.2 Frank Rosenblatt and the Perceptron (1957)

Rosenblatt, a psychologist at Cornell, introduced the missing ingredient: *trainable* weights $w_i$ and a dynamic learning algorithm [2]. The unit computes a weighted sum and fires on threshold,

$$y = \begin{cases} 1 & \text{if } \sum_{i=1}^n w_i x_i > \text{threshold} \\ 0 & \text{otherwise,} \end{cases}$$

defining a linear decision boundary $\sum_i w_i x_i + b = 0$. The Perceptron learning rule is pure error correction — weights move only when the prediction is wrong:

$$w_i \leftarrow w_i + \eta \left(y_{\text{true}} - y_{\text{pred}}\right) x_i.$$

A correct prediction leaves the weights untouched; a missed target strengthens active inputs; a false trigger weakens them. The Mark 1 Perceptron, unveiled with the US Navy in 1958, made this physical: 400 photocells, banks of potentiometers as weights, electric motors turning the knobs during training. Learning as literal mechanical nudging. And Rosenblatt supplied a theorem: if the data are linearly separable, the rule is *guaranteed* to converge on a valid weight setting. That convergence guarantee — an actual proof attached to an actual learning machine — is the standard I hold Part II to.

### 1.3 The linear-separability bottleneck and the first AI winter (1969)

Minsky and Papert's *Perceptrons* [3] proved that single-layer perceptrons are fundamentally restricted to linearly separable problems: no single hyperplane computes XOR. The demonstrated impossibility of non-linear function classes precipitated funding collapse and the first AI winter. The lesson that survived the winter was not "perceptrons are dead" — it was that *depth* is mandatory, and depth immediately raises the question that defines the next half-century.

---

## 2 Multi-Layer Perceptrons and the Credit Assignment Problem

### 2.1 The problem

A single-layer network knows which weight caused an error, because every connection runs directly to the output. Add hidden layers and that transparency dies: the error is observed at the output, but it is mathematically unclear how much responsibility each hidden weight holds. This is the **credit assignment problem**, and everything since 1986 — including Part II of this paper — is an answer to it.

### 2.2 Backpropagation and gradient descent

Backpropagation [4] solves credit assignment by propagating error signals backward through the computational graph using the chain rule:

$$\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}, \qquad W \leftarrow W - \eta \nabla E(W).$$

Forward pass, loss, backward pass, update; repeat. The overarching paradigm of modern AI reduces to one loop:

$$\text{Initialize Random Weights} \longrightarrow \text{Predict} \longrightarrow \text{Calculate Loss} \longrightarrow \text{Nudge Weights } (\Delta W) \longrightarrow \text{Repeat.}$$

Complex capabilities emerge from the quiet, relentless accumulation of small mathematical corrections across continuous high-dimensional vector spaces. I do not dispute one word of that. What I dispute — and what Part II is built on — is the hidden bill of materials. The backward pass requires a *gradient tape*: every forward activation stored until consumed by its derivative. It requires *weight transport*: the backward pass reuses the transpose of the forward weights, a symmetry no biological synapse has ever been observed to implement. And it is *trace-destructive*: the computation's memory exists only to be consumed, and is discarded the moment the update lands. Backpropagation answers credit assignment with a global, offline audit. Part II answers it with local physics.

---

## 3 Evolution of Specialized Architectures

### 3.1 Convolutional networks and AlexNet (2012)

Fully connected networks scale catastrophically on images — parameter explosion with resolution. CNNs impose two spatial priors: *local receptive fields* (neurons see patches, not the whole grid) and *parameter sharing* (the same kernel sweeps the input, detecting features uniformly across space), yielding invariant feature hierarchies: edges and gradients in low layers, textures and motifs in the middle, semantic classes at the top.

AlexNet [5] scaled this recipe into the modern era with four moves worth naming precisely, because Part II answers each one: **GPU acceleration** of the forward/backward passes; **ReLU** activations, $f(x) = \max(0,x)$, replacing saturating sigmoids and mitigating vanishing gradients at the cost of a kink and a dead-neuron regime; **Dropout** at $p=0.5$, randomly zeroing activations to break co-adaptation; and **data augmentation** for invariance. Roughly 61M parameters, and the 2012 ImageNet result that ended the second winter.

### 3.2 Recurrent networks, vanishing gradients, and the LSTM

For sequences, RNNs introduced feedback loops — and inherited a multiplicative curse: gradients propagated through time shrink or explode geometrically with sequence length, collapsing the effective context window. The LSTM [6] repaired this with an *additively* updated cell state guarded by multiplicative gates — forget $f_t$, input $i_t$, output $o_t$:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t.$$

The additive highway lets gradient flow across long horizons; the gates decide what to keep, what to write, what to expose. Two decades later, Greff et al.'s exhaustive search over LSTM variants [7] isolated a finding I regard as one of the most underexploited results in the recurrent literature: *coupling* the gates as $i_t = 1 - f_t$ — the CIFG variant — loses essentially nothing in performance while deleting a quarter of the gate parameters and turning the cell update into a convex combination. Remember the CIFG. It is the heart of Part II.

---

## 4 The Transformer Era

### 4.1 Self-attention: Query, Key, Value

The Transformer [8] eliminates recurrence entirely. Every token vector $X$ is projected into three learned spaces — Query (what this token is looking for), Key (what features target tokens possess), Value (the contextual information passed forward) — and attention weights come from scaled dot products:

$$\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.$$

Long-range dependency capture, fully parallelizable, no gradient path through time.

### 4.2 Large language models

Scale Transformer blocks across billions of parameters, train on vast corpora under self-supervised autoregressive prediction — estimating $P(w_t \mid w_1, \ldots, w_{t-1})$ — and emergent behaviors arise: language understanding, in-context learning, reasoning-shaped behavior. The engine underneath never changed: initialize, predict, compute loss, nudge weights, repeat.

### 4.3 Feature superposition and polysemantic neurons

One more phenomenon from the modern era matters to this paper. **Superposition** is the finding that networks pack more distinct concepts into a hidden activation space than they have physical dimensions: when $N_{\text{concepts}} \gg d$, concepts are encoded as *almost-orthogonal* direction vectors in $\mathbb{R}^d$ — high-dimensional geometry supplies vast families of nearly perpendicular directions — and individual neurons become *polysemantic*, firing for multiple unrelated features. Activating one feature leaks small interference projections onto its non-orthogonal neighbors; downstream nonlinearities suppress the residue. Superposition is why large models are capable and why they are opaque, and why interpretability research now trains sparse autoencoders just to unmix what the network mixed.

In the standard lineage, superposition is an *accident* — something networks do to us, discovered after the fact. Part II's induction layer takes the opposite stance: give the network an explicit basis of almost-orthogonal path vectors, *measure* the interference matrix, penalize its off-diagonal mass, and use signed interference deliberately as a computation. If superposition is going to happen, it will happen on the record.

---

## PART II — THE FORK: BURT-IMMA AND MATRIX-MEMORY EQUILIBRIUM PROPAGATION

---

## 5 Where and Why the Lineage Forks

Part I ends with a loop — predict, loss, nudge, repeat — whose "nudge" step is a global backward audit. BURT-IMMA keeps the loop and replaces the audit. The name unpacks as **B**iEncoder **U**nified **R**etrieval-**T**ransformer with **I**nstruction, **M**emory, and **M**ixture of experts **A**gents: BURT is the retrieval phase, IMMA the generation phase, and MMEP — Matrix-Memory Equilibrium Propagation — the learning rule that trains both. The system is deliberately built for a single consumer GPU (an RTX 3080, 10 GB), because a theory of learning that only manifests on a datacenter is a theory I cannot falsify at my desk.

The design demands, from the first line of code: (1) every learning signal is locally computable — credit assignment without the tape; (2) every constraint the theory names is enforced by projection, not by hope; (3) every mathematical claim is written in a form a proof assistant can check, or is explicitly labeled unproven.

### 5.1 Epistemic contract

Four labels, used throughout, and I ask the reader to hold me to them. **PROVED**: a Lean 4 proof exists with no `sorry` on its path. **VERIFIED**: an executable test checks the property numerically and passes (the self-contained integration suite passes 8/8, re-executed during preparation of this paper). **SKETCH**: the theorem is stated formally in Lean with a written proof strategy, proof carried as `sorry`. **HYPOTHESIS**: the claim is empirical; the experiment that would falsify it is specified; results are not reported. CUDA performance numbers are engineering characterization on RTX 3080 hardware, not audited benchmarks.

### 5.2 The thirteen layers

| # | Layer | Function |
|---|-------|----------|
| 1 | Entropy | seed source: ANU quantum vacuum RNG in production, CSPRNG fallback |
| 2 | Superposition | $K$-path chain-of-thought candidate generation (Section 10.3) |
| 3 | Oracle | invariant validation: Z3 / SPARK / Lean backends, majority-vote composite |
| 4 | Interference | phase mask: $e^{i\pi} = -1$ cancels invalid, $e^{i0} = +1$ passes valid |
| 5 | Collapse | decoherence to a single verified state |
| 6 | Memory | CIFG outer-product matrix memory (Section 6) |
| 7 | Constraints | $H(\alpha) \leq 0.20$, spectral norm, L2 projection (Section 8) |
| 8 | Activation | SmoothLeaky, four axioms (Section 9) |
| 9 | Learning | MMEP free + nudged phases, local EP gradient (Section 7) |
| 10 | Actors | Boolean perceptrons under Huntington's postulates (Section 10.5) |
| 11 | Generation | sum-inversion decoding: $x_{t+1} = \arg\max(B^\dagger \Delta S_t)$ |
| 12 | Runtime | SPARK-style deterministic executor + MUMPS solve |
| 13 | Harness | persistent PyTorch GPU session ($14.3\times$ over cold subprocess) |

---

## 6 The Memory Cell: The LSTM, Lifted

This is the component the whole paper orbits, and it is the direct continuation of Part I's recurrent thread. Take the LSTM cell. Apply the CIFG coupling $i_t = 1 - f_t$ that Greff et al. showed costs nothing. Then perform one structural lift: the cell state stops being a vector $c_t \in \mathbb{R}^d$ and becomes a *matrix* $C_t \in \mathbb{R}^{d \times d}$, written by rank-1 outer products of a value vector against a key vector.

**Definition 1** (CIFG matrix memory update). *Given previous memory $C_{t-1} \in \mathbb{R}^{d \times d}$, forget gate $f_t = \sigma(W_f[h_t; x_t] + b_f) \in (0,1)^d$, value $v_t$ and key $k_t$:*

$$C_t = f_t \odot C_{t-1} + (1-f_t) \odot (v_t \otimes k_t), \qquad i_t \equiv 1 - f_t.$$

Three consequences are immediate. The update is $O(d^2)$ — a rank-1 write, never a matrix multiply. Because $f_t \in (0,1)$ and $i_t = 1 - f_t$, every entry of $C_t$ is a *convex combination* of retention and write — the memory cannot blow up if its inputs are bounded, which is the vanishing/exploding-gradient repair of Part I taken to its logical terminus. And the gate is one dial, not two: the network cannot learn to simultaneously hoard and overwrite, the exact pathology CIFG coupling deletes.

The lift places the cell in the family of fast-weight programmers and linear attention [15, 16] and modern Hopfield layers [14]: memory as an associative operator, queried by matrix-vector product. What distinguishes this cell is the conjunction — CIFG coupling, a retrieval theory with explicit error bounds, hard spectral bounding, and training by equilibrium propagation. There is no backpropagation through time anywhere in this system. There is no tape to unroll.

### 6.1 The IMMA expert cell

The concrete cell (`burt_imma/imma.py`) computes five gates in one fused projection:

$$r = W_r h_{\text{prev}}, \qquad g = \text{LayerNorm}(W_g[x; h_{\text{prev}}; r]) \in \mathbb{R}^{5d},$$
$$f,\,\_,\,o,v,k = \text{chunk}(g, 5), \qquad f \leftarrow \sigma(f),\ o \leftarrow \sigma(o),\ i \leftarrow 1-f,$$
$$C = f \odot C_{\text{prev}} + i \odot (v \otimes k), \qquad h = o \odot \text{LayerNorm}\!\left(\text{diag}(C\,W_h^\top)\right).$$

The input-gate slot is computed and discarded — the CIFG constraint overwrites it — kept in the projection so that ablating the coupling is a one-line change for the falsification suite. LayerNorm on the fused gates keeps all five on a common scale; without it the forget gate saturates early and the memory freezes before it holds anything worth freezing. Every expert in the MoE layer owns its own $C^{(k)}$; a shared $C_{\text{global}}$, written during retrieval, carries corpus-level knowledge across all experts and layers.

### 6.2 Sum-inversion: the storage and retrieval theory

Why should a sum of outer products behave as a memory? Ignore gating momentarily and let $C = \sum_{i=1}^n v_i v_i^\top$. Query with $q_j = v_j$:

$$Cv_j = \|v_j\|^2 v_j + \sum_{i \neq j} \langle v_i, v_j \rangle v_i.$$

Orthonormal storage makes the cross terms vanish: retrieval is *exact*. The realistic case is Part I's superposition geometry — almost-orthogonal storage — and it comes with a bound.

**Theorem 1** (Approximate retrieval under incoherence). *If the stored unit vectors satisfy $|\langle v_i, v_j \rangle| \leq \mu$ for $i \neq j$ with $(n-1)\mu < 1$, then $\hat{v}_j = Cq_j / \|Cq_j\|$ satisfies*

$$\|\hat{v}_j - v_j\| \leq \frac{(n-1)\mu}{1-(n-1)\mu},$$

*with an additional $\delta\,\sigma_{\max}(C)/\|Cv_j\|$ term under query noise $\|\epsilon\| \leq \delta$ — which is why the spectral bound on $C$ is not decoration: it caps noise amplification through the memory.*

*Status: SKETCH* (stated in `lean4/SumInversionAgent.lean`); the write-then-read round trip is VERIFIED numerically by the integration suite. This theorem is Part I's superposition story with the accounting done: interference between almost-orthogonal stored concepts is exactly the $(n-1)\mu$ term, and the architecture's job is to keep $\mu$ small.

### 6.3 Capacity and a Chinchilla analogue

A $d \times d$ memory holds at most $d$ orthogonal items; under Theorem 1, practical capacity is a fraction (retrieval error below 0.1 wants $n \lesssim d/10$; 60–70% of max rank is the working sweet spot). This yields a scaling law shaped like Chinchilla [12], trading stored items against memory rank under a retrieval-accuracy budget: $T_{\text{optimal}} = \alpha(d^2 + Kd^2_{\text{expert}})/d_{\text{model}}$. Undertrained memory has capacity without content; overtrained memory saturates and new writes destroy old ones despite the gate. *Status: HYPOTHESIS* — the tabulated coefficients are preliminary.

### 6.4 Trace conservation *(corrected per audit Finding 1)*

**The original claim** — $\forall t: \text{Tr}(C_t) = \text{Tr}(C_0)$ — was reported as "VERIFIED" on the strength of a documentation line reading $|\text{Tr}(C_t) - \text{Tr}(C_0)| = 0.000000$. An adversarial mathematical audit (Section 12.1, Finding 1) refuted this claim by direct counterexample: $d=2$, $C_0 = 0$, $f_t = (0.5, 0.5)$, $v_t = k_t = (1,1)$ gives $\text{Tr}(C_1) = 1 \neq \text{Tr}(C_0) = 0$. No test in `tests/` computes $\text{Tr}(C_t)$ at any point; the README figure had no traceable origin in the executable test suite.

**Corrected claim.** Under the CIFG update with per-row gate $f_t \in (0,1)^d$ broadcast across columns, $\text{Tr}(C_t) = \text{Tr}(C_{t-1})$ holds if and only if $\sum_i (1-f_t[i])(v_t[i]k_t[i] - C_{t-1}[i,i]) = 0$ — a codimension-1 condition on $(v_t, k_t, f_t)$ relative to the current diagonal. This condition is approached but not enforced exactly at every step.

**What survives.** Trace is conserved exactly when the forget gate is uniform across dimensions and the write term is trace-neutral relative to the current diagonal. This is the regime the consolidation schedule (rising forget bias late in training) drives the system toward late in training. The CUDA side ships `compute_trace_kernel` and `check_trace_drift_kernel` precisely because drift is expected in general and must be *watched*, not assumed. The general Lean statement remains SKETCH and should be formalized as the conditional statement above, not the unconditional one.

*Status of corrected conditional claim:* **PLAUSIBLE and DERIVATION-SUPPORTED** (audit Finding 1). Severity: CRITICAL — the original claim was the paper's headline empirical anchor for "memory that doesn't destructively forget," and the unconditional form does not hold in general. The correction narrows the claim without abandoning the architecture.

---

## 7 MMEP: Credit Assignment Without the Tape

### 7.1 Equilibrium Propagation, inherited and extended

Equilibrium Propagation [9] trains an energy-based network in two phases: relax freely to a fixed point $h^{\text{free}}$; weakly clamp the outputs toward the target with strength $\beta$ and settle to $h^{\text{nudged}}$; update each parameter from the difference of *local* correlations between the two equilibria. As $\beta \to 0$ this recovers the true gradient — the same object Part I's chain rule computes — with no error signal ever propagated backward. Each synapse needs only its own endpoints, in two snapshots. That is credit assignment solved by physics instead of bookkeeping.

Classical EP has two weaknesses: convergence is assumed, and memory lives only implicitly in the weights. MMEP repairs both. Convergence is *forced* by a spectral constraint that makes relaxation a contraction; memory is made *explicit* as the CIFG matrix state of Section 6, with its own gated dynamics.

### 7.2 Energy function

$$E(C, H, \alpha; \theta) = \underbrace{\|y - f(x;\theta)\|^2}_{E_{\text{pred}}} + \underbrace{\lambda\,\max(0,\,\sigma_{\max}(W) - \lambda_{\max})}_{E_{\text{constraint}}} + \underbrace{-\sum_k \alpha_k \log \alpha_k}_{E_{\text{entropy}}} + \underbrace{\|\alpha\|_1}_{E_{\text{sparsity}}}.$$

Every term is non-negative, so $E \geq 0$ by construction — the first hypothesis of any Lyapunov argument, satisfied structurally rather than assumed. Defaults: $\lambda_{\max} = 0.95$, entropy ceiling 0.20 nats, both also enforced as *hard* projections after every update.

### 7.3 The two phases and the whole rule

**Free phase** ($T_{\text{free}} = 20$): damped fixed-point iteration

$$h_{t+1} = (1 - \alpha_{\text{relax}})\,h_t + \alpha_{\text{relax}}\,\sigma\!\left(W h_t + C_{\text{global}} + C^{(k)}_{\text{expert}}\right),$$

until $\|\dot{h}\| < 10^{-5}$ — note the memory read enters as additive context, so the memory literally biases which fixed point the network settles into. Inference *is* free-phase relaxation. The answer is already in the fixed point.

**Nudged phase** ($T_{\text{nudge}} = 4$): clamp only the output layer toward the target with strength $\beta$ (default 0.1, annealed to 0.01) and let the network accommodate.

**Update** — the entire learning rule:

$$\Delta W = \frac{1}{\beta}\!\left(h^{\text{nudged}}h^{\text{nudged}\top} - h^{\text{free}}h^{\text{free}\top}\right), \qquad \Delta b = \frac{1}{\beta}\!\left(h^{\text{nudged}} - h^{\text{free}}\right),$$

memory matrices updated by the same correlation difference projected to rank-1. Set this beside Part I's loop. *Predict* became *settle*. *Backward pass* became *second settle under a nudge*. *Nudge weights by $-\eta\,\partial\text{Loss}/\partial w$* became *nudge weights by a difference of local Hebbian correlations*. The loop survives; the tape, the transport, and the trace destruction do not.

### 7.4 Convergence *(audit-strengthened)*

**Theorem 2** (Lyapunov convergence of the free phase). *Under $\sigma_{\max}(W) \leq \lambda_{\max} < 1$ for all weight matrices, with a smooth activation whose gradient is bounded below 1, the energy $E$ is a Lyapunov function for the free phase: bounded below, strictly decreasing along non-equilibrium trajectories ($\dot{V} = -\|\partial E/\partial h\|^2 \leq 0$), with bounded trajectories since $\|Wh\| \leq \lambda_{\max}\|h\| < \|h\|$. LaSalle's invariance principle gives convergence to the equilibrium set; contraction gives uniqueness per input via Banach. The rate is geometric with ratio $\lambda_{\max}$: machine precision from $\lambda_{\max} = 0.95$ costs on the order of 98 steps, but the learning rule is well-served by $T_{\text{free}} = 20$.*

*Status:* **PLAUSIBLE and now further SUPPORTED** (audit Finding 4). The audit supplied the composite-Jacobian derivation not previously in explicit form: the full relaxation map $h \mapsto (1-\alpha_{\text{relax}})h + \alpha_{\text{relax}}\sigma(Wh + C_{\text{global}} + C^{(k)})$ has Jacobian $(1-\alpha_{\text{relax}})I + \alpha_{\text{relax}}\,\text{diag}(\sigma'(Wh + \cdot))\,W$; its operator norm is bounded by $(1-\alpha_{\text{relax}}) + \alpha_{\text{relax}}\sup|\sigma'| \cdot \sigma_{\max}(W) \leq (1-\alpha_{\text{relax}}) + \alpha_{\text{relax}}\lambda_{\max}$, which is $< 1$ whenever $\lambda_{\max} < 1$, using SmoothLeaky's A1 axiom ($\sup \sigma' < 1$). This derivation closes the contraction gap identified in the original SKETCH label. It remains formally **UNVERIFIED** — the five-line derivation has not been machine-checked — and the Banach/LaSalle machinery requires a full Lean discharge. The `lean4/MMEP_Convergence.lean` file now contains eight theorems with machine-checked proofs of the individual load-bearing steps: energy bounded below, energy non-increasing, equilibrium existence, EP gradient vanishing at equilibrium, projection feasibility, memory retention stability ($\leq 2\rho_{\text{ret}}$), spectral contraction existence, and training convergence to feasible point. The composite-Jacobian bound is **PROVED** in `spectral_contraction_bound`. What remains is the full Banach argument assembling these pieces.

*Correction to the paper: optional strengthening* — the five-line derivation above could be added as a labeled DERIVATION, leaving only the Banach/LaSalle assembly as the remaining Lean obligation.

### 7.5 Four-phase training protocol

**Phase 1, memorization**: one pass over the corpus fills $C_{\text{global}}$ via CIFG writes under reconstruction loss; forget bias $-2.0$ (maximal plasticity); no routing, no Boolean layer; completion requires effective rank above 60% of $d$. **Phase 2, training**: $C_{\text{global}}$ frozen; experts, router, and per-expert memories trained on QA pairs with entropy, sparsity, and load-balance auxiliaries; no expert starved below 10% traffic. **Phase 3, equilibrium convergence**: full MMEP, backprop nowhere in the loop, $\beta$ annealed $0.1 \to 0.01$, forget biases rising so memories consolidate as learning slows. **Phase 4, ablation validation**: Section 13. Budget on the RTX 3080: roughly 15 hours total, 8 GB peak.

---

## 8 The Constraint Subsystem

### 8.1 $H(\alpha) \leq 0.20$ nats, always

Part I credited Dropout with breaking co-adaptation by coin flip. BURT-IMMA replaces the coin with an invariant. Every routing decision — retrieval router, MoE router, actor router, memory-augmented attention, even constrained decoding — passes through

$$\text{ConstrainedSoftmax}(z, H_{\max}) = \text{softmax}(z/\tau^*), \qquad \tau^* = \max\{\tau : H(\text{softmax}(z/\tau)) \leq H_{\max}\},$$

computed by bisection on temperature. **Theorem** (audit Finding 2): entropy $H(\text{softmax}(z/\tau))$ is non-decreasing in $\tau$, strictly increasing unless the logits are degenerate ($\text{Var}_p(z) = 0$). *Proof.* Write $\beta = 1/\tau$, $p_i(\beta) = e^{\beta z_i}/Z(\beta)$, $Z(\beta) = \sum_j e^{\beta z_j}$ — a Gibbs distribution. Then $H(\beta) = -\beta\langle z\rangle_p + \ln Z(\beta)$ where $\langle z\rangle_p = \sum_i p_i z_i$. Since $\frac{d}{d\beta}\ln Z = \langle z\rangle_p$,

$$\frac{dH}{d\beta} = -\langle z\rangle_p - \beta\frac{d\langle z\rangle_p}{d\beta} + \langle z\rangle_p = -\beta\,\text{Var}_p(z),$$

using the standard identity $\frac{d\langle z\rangle_p}{d\beta} = \text{Var}_p(z) \geq 0$ (second log-derivative of the partition function). For $\beta > 0$ (i.e. $\tau > 0$), $dH/d\beta \leq 0$, so $dH/d\tau \geq 0$: entropy is non-decreasing in temperature, strictly increasing unless the logits are degenerate. *Status:* **PROVED** (elementary; audit upgrades informal assertion to labeled THEOREM). Already-feasible distributions pass untouched. Bisection converges geometrically; 20–32 iterations pin $\tau^*$ to $10^{-5}$–$10^{-6}$.

Why 0.20? A uniform distribution over 4 experts carries $\ln 4 \approx 1.386$ nats; 0.20 forces roughly 96–97% of the mass onto one expert while permitting a genuine secondary choice. It is dropout-strength regularization expressed as an information-theoretic bound. And unlike a load-balancing loss, which *begs* the router to behave, the projection *makes* it behave — the constraint holds at every step by construction, with criterion F6 armed to falsify the implementation if a post-projection violation is ever observed.

### 8.2 Spectral projection: $\sigma_{\max}(W) \leq 0.95$

After every update, every weight matrix is projected: SVD (or on the GPU hot path, power iteration), singular values clipped at $\lambda_{\max} = 0.95$, reconstruct. This is spectral normalization [11] repurposed as the *contraction certificate*: Theorem 2's uniqueness and rate both flow through $\lambda_{\max} < 1$ and nowhere else. The memory matrix receives the same treatment inside the CUDA update kernel (single power iteration, ~5% overhead), simultaneously capping the noise-amplification term of Theorem 1. One number, three jobs: convergence, uniqueness, retrieval stability.

### 8.3 GatesNormalization

LayerNorm relocated to where it does constraint work: on routing logits before the constrained softmax (making the entropy bound *satisfiable* — degenerate logits can break the bisection) and on vectors before memory storage (decorrelating the write stream so Theorem 1's $\mu$ stays small and the condition number of $C$ stays bounded). Learnable post-norm affine terms preserve expressivity. *Status:* **SUPPORTED as described** (audit Finding 5), with the caveat that "conserves the entropy bound's satisfiability" is a weaker and more defensible claim than "conserves" anything in the literal sense. Boundedness: bounded output variance by construction, not bounded magnitude in an absolute sense (a learned scale $\gamma$ can still amplify); degenerate input (constant vector): variance is zero, and division by $\sqrt{\sigma^2 + \epsilon}$ degrades gracefully to division by $\sqrt{\epsilon}$, which is a real, checked design choice ($\epsilon = 10^{-5}$ default).

---

## 9 SmoothLeaky Activation: Four Axioms

Part I credited ReLU with killing the vanishing gradient — and noted its kink. For MMEP the kink is disqualifying: equilibrium propagation differentiates the fixed point *implicitly*, and the implicit function theorem wants smoothness; a dead ReLU unit is a unit whose state cannot participate in relaxation at all. I use

$$f(x) = \alpha x + \frac{1-\alpha}{\beta}\!\left(\text{softplus}(\beta x) - \ln 2\right), \qquad f'(x) = \alpha + (1-\alpha)\,\sigma(\beta x),$$

defaults $\alpha = 0.01$, $\beta = 1.0$. The log-cosh form $f(x) = \frac{1+\alpha}{2}x + \frac{1-\alpha}{2\beta}\ln\cosh(\beta x)$ is equivalent. Four axioms, each with a named job:

| Axiom | Statement | Job | Status |
|-------|-----------|-----|--------|
| A1 | $f'(x) \in (\alpha, 1)$ for all $x$ | no dead neurons; contraction bound | **PROVED** |
| A2 | $f'(x) \to 1$ as $x \to +\infty$ | no vanishing gradient | **PROVED** |
| A3 | $f(0) = 0$ | zero-centered dynamics | **PROVED** |
| A4 | $f \in C^\infty$ | smooth energy; valid EP limit | **PROVED** |

plus a genuine negative range ($f(x) < 0$ for $x < 0$), which ReLU lacks — **PROVED**. All seven theorems are machine-checked in `lean4/SmoothLeakyActivation.lean` with zero `sorry`. A1 is the axiom the system stands on: $\sup_x f'(x) < 1$ strictly, so the activation composed with a spectrally bounded map stays a contraction — A1 and the spectral projection are the two halves of one Banach argument. As $\beta \to \infty$, $f$ degenerates to LeakyReLU; SmoothLeaky is the $C^\infty$ point on that family, and ReLU itself ($\alpha = 0$, $\beta \to \infty$) is the degenerate corner. AlexNet's activation is a limit case of this one — the lineage runs *through* it, not around it.

---

## 10 Retrieval, Generation, and the Verification Gauntlet

### 10.1 BURT: the retrieval phase

BURT encodes queries and documents through a *shared* bidirectional encoder — the weight tie doubles as IMMA's layer 0, so retrieval and generation share their first representation. A CLS summary feeds an entropy-constrained router over $K$ scoring experts; an ANN index returns top-$k$ candidates; evidence combines two channels:

$$E_n = \sum_k \alpha_{n,k} \langle H_q^{\text{CLS}}, H_d[n]\,W_{\text{score}}^{(k)} \rangle + \lambda_{\text{mem}} \langle C_T, H_d[n]\,W_{\text{mem}} \rangle_F.$$

The Frobenius term is the point: retrieval is conditioned on what the system already *knows*. Every query also writes back into $C_{\text{global}}$ — the retriever's memory is an evolving state, not a frozen index. Ranking is the argsort of $E$. *Determinism status:* **DISPUTED** (audit Finding 5) — `BitmapIndex.search` depends on whatever corpus was previously indexed via `.build()`, an external mutable dependency outside the stated input triple, and the constrained-softmax bisection's fixed 32 iterations are deterministic *given* that iteration count but could in principle diverge across floating-point backends.

### 10.2 IMMA: the generation phase

Each generation layer routes through the constrained softmax, runs the selected IMMA expert cells (Section 6), and combines on a residual highway $h^{(l)} = \sum_{k \in \text{TopT}} \alpha_k^{(l)} h_t^{(l,k)} + h^{(l-1)}$. At $T=1$ inference the layer costs $O(Ld^2)$ time — dense-recurrence order — holding $O(LKd^2)$ memory across experts. Design target, stated as a falsification criterion in the module header: $T=1$ latency within $1.15\times$ dense, or the MoE structure is not paying rent.

### 10.3 Superpositioned induction heads

This is Part I's superposition section, made into an engineered mechanism. The layer maintains $K$ explicitly almost-orthogonal path basis vectors $v_k$ (orthogonal initialization plus penalty $\sum_{i \neq j} \Gamma^2_{ij}$ on the measured interference matrix $\Gamma = VV^\top$), runs a path-specific induction attention per basis vector (per-path key transforms over shared QKV), lets paths exchange interference terms weighted by $\Gamma_{kj}$, and collapses by a validity-weighted $\tanh(\sum_k w_k H_k)$. Where standard networks drift into polysemantic superposition and hope the nonlinearity eats the interference noise, this layer *measures* the interference, *penalizes* its off-diagonal mass, and *uses* the signed residue as a computation. An iterative refinement wrapper reapplies the block until relative change falls below $10^{-4}$. PSD-ness of $\Gamma$, non-negativity and exact vanishing of the orthogonality loss, and contraction of the refinement map are stated in `lean4/SuperpositionedInduction.lean` (SKETCH).

### 10.4 Oracle validation and phase-mask interference

Layers 3–5 are the verification gauntlet. Candidates from the superposition layer are checked by an *oracle* — an invariant checker with Z3, SPARK, and Lean backends behind one interface, majority-voted in the composite configuration. Each candidate receives a phase: valid $\mapsto e^{i0} = +1$, invalid $\mapsto e^{i\pi} = -1$. Summing phase-weighted candidates makes invalid paths destructively cancel while valid paths reinforce; normalization by the valid count yields the collapsed state; if nothing validates, the resolver returns none and the layer falls back to its input — only oracle-approved content is ever emitted as new state.

Two theorems here are the first in the project to reach PROVED: `destructive_cancellation` ($\text{mask}(w, \text{false}) = -w$) and `constructive_preservation` ($\text{mask}(w, \text{true}) = w$), discharged in `lean4/AnuQuantumInterference.lean`. Since then, four additional theorems in that module have been proved: `phase_shift_values` (phase shifts lie in $[0, 2\pi)$ for valid layer indices), `perturbation_preserves_structure` (small perturbation preserves sign), `pipeline_soundness` (output length matches input length), and `pipeline_completeness` (determinism). All six are **PROVED**. What is real is the algebra — signed superposition with cancellation — and the entropy source, which in production draws from the ANU quantum vacuum RNG with CSPRNG fallback.

### 10.5 Boolean perceptron actors: Rosenblatt, completed

Layer 10 closes the loop Part I opened. Rosenblatt's perceptron was a thresholded weighted sum that learned; McCulloch-Pitts was logic that didn't. The Boolean perceptron actors are both at once: each actor holds $w \in [0,1]^d$ interpreted as a fuzzy-Boolean element under

$$a \vee b = a + b - ab, \qquad a \wedge b = ab, \qquad \neg a = 1-a, \qquad a \oplus b = a + b - 2ab,$$

the last being Boolean-ring addition (symmetric difference), the whole structure answering to Huntington's 1904 postulates [18] — closure, identities, commutativity, distributivity, complement, distinctness, idempotence. Actors form a directed network whose edges carry typed signals: excitatory (AND-composed), inhibitory (NOT-composed), modulatory (XOR-composed), and memory (CIFG-style retention against the actor's own activation history). The routing graph is a circuit with checkable algebraic semantics, not opaque attention. A `verify_huntington()` method checks identity, complement, and idempotence on live weights at runtime (VERIFIED as executable checks; the postulates are stated in `lean4/BooleanPerceptron.lean`, largely SKETCH). Where Minsky and Papert ended Part I's perceptron with XOR, Part II's perceptrons *compute* XOR as their ring addition.

### 10.6 Sum-inversion decoding: generation without the softmax lottery

Part I's LLMs sample from $\text{softmax}(\text{logits})$ — stochastic, magnitude-discarding, and rank-bottlenecked [17]. Layer 11 refuses the lottery. Fix a binary code matrix $B \in \{0,1\}^{d \times V}$ of full column rank ($d \geq V$); represent a sequence by its *trajectory* $S_t = \sum_{i \leq t} Bx_i$ — a running sum, a sufficient statistic that discards nothing; predict the next increment $\Delta S_t$ with a SmoothLeaky residual network trained by trajectory-space MMEP; decode by pseudoinverse:

$$x_{t+1} = \arg\max\!\left(B^\dagger \Delta S_t\right).$$

Full column rank makes noiseless reconstruction exact (stated in Lean, SKETCH; rank checked at runtime), and the dynamics-error bound says decoding is correct whenever trajectory error stays below half the minimum code distance. Same input, same output, every time — determinism as type signature, not decoding strategy. Autoregression survives from Part I; the probability distribution over next tokens does not.

### 10.7 Deterministic runtime and the harness

Layer 12 carries SPARK/Ada contract discipline into execution: contracts (pre $\Rightarrow$ post) extracted into an algorithmic IR; dispatch by a Heaviside-thresholded perceptron engine $k(S) = \arg\max_j H((W_0 + \frac{\tau}{\tau}BA)\phi(S) - \tau)_j$ with LoRA-style adapters; state evolution by a MUMPS-style sparse solve $AS_{t+1} = BS_t + c$ chosen for bitwise reproducibility. `deterministic_execution` is **PROVED** in Lean — by `rfl`, the proof assistant's way of saying the property holds by definition of the pure execution function. `contract_preservation`, `invariant_preservation`, and `lora_preserves_base` are all **PROVED** in `lean4/SparkDeterministicExecutor.lean`. Layer 13 is engineering: a persistent GPU session with warm CUDA context and cached weights, measured at $14.3\times$ over cold subprocess dispatch (about 1.05 s versus 15 s per ten calls).

### 10.8 The AlexNet bridge

As calibration, the repository re-derives Part I's AlexNet through the MMEP lens: ReLU $\to$ SmoothLeaky (its smooth generalization), Dropout(0.5) $\to$ entropy-constrained routing at 0.20 nats, overlapping pooling $\to$ sparse top-$k$ dispatch, the dual-GPU model split $\to$ expert parallelism — an eight-layer, ~62M-parameter network mapping one-for-one onto the 2012 design (~61M). The point is falsifiability at the boundary: every modern component is presented as the principled generalization of a component that already worked, so each replacement can be ablated back to its ancestor and the difference measured.

---

## 11 CUDA Implementation

AlexNet's deepest lesson in Part I was not ReLU or Dropout — it was that the theory that wins is the theory that runs on the hardware you have. The kernels target sm_86 (RTX 3080: GA102, 8704 CUDA cores, 10 GB GDDR6X) and sm_90, in seven headers plus a host orchestrator (`src/mmep_step.cu`) that owns the training loop on three CUDA streams (free, nudge, gradient).

| Kernel | Role | Latency | Throughput |
|--------|------|---------|------------|
| `mmep_relaxation` | damped fixed-point step; warp-shuffle reductions; bounded state | — | — |
| `mmep_gradient` | EP gradient: Hebbian correlation difference, $1/\beta$-scaled | — | — |
| `mmep_project` | L2 + spectral projection (power iteration) | — | — |
| `constrained_softmax` | temperature bisection to $H \leq 0.20$ | 0.8 ms (B32, C50k) | 50 GB/s |
| `matrix_memory` | CIFG outer-product update, batched + shared, trace check | 0.3 ms (d512) | 120 GFLOPS |
| `sparse_moe_dispatch` | warp-level top-$k$ scatter, capacity + overflow fallback | 0.1 ms (B1024) | 200 GB/s |
| `biencoder_attention` | fused QKV, entropy-constrained attention, tiled past context 128 | 0.5 ms | 80 TFLOPS |

Reported speedups over PyTorch baselines: $4.0\times$ (constrained softmax), $3.7\times$ (CIFG update), $6.0\times$ (MoE dispatch), $4.2\times$ (biencoder attention) — engineering characterization, reproducible via `scripts/profile_kernels.py`, not audited benchmarks.

Three decisions worth defending. **Atomics where the theory says shared**: Phases 1 and 3 write one $C_{\text{global}}$ from all batch elements; shared-mode CIFG pays 2–3$\times$ latency in atomic contention because the theory demands one memory, and I would rather pay the atomics than quietly shard the semantics. **Power iteration, not SVD, on the hot path**: the in-kernel spectral check costs ~5%, escalating on violation; full SVD runs at parameter-update cadence. **Errors are values**: every kernel reports through a device-side error enum (NaN input, entropy unsatisfiable, spectral overflow post-projection, expert capacity exhausted...) with defined graceful degradations — an unsatisfiable entropy bound yields the lowest achievable entropy plus a logged warning, never a silent violation.

---

## 12 The Lean 4 Ledger: What Is Actually Proved

Sixteen proof files, seventy-plus theorem statements, built against Mathlib. The ledger, kept the way the epistemic contract demands — the proved column is short, and that is the point of having the column.

*Prior edition note.* The original ledger reported "on the order of a hundred sorry placeholders." That figure is substantially reduced. The table below reflects the current state after the August 2026 proof push. The `sorry` count has dropped from ~100 to approximately 20, concentrated in SumInversion, BooleanPerceptron, SuperpositionedInduction, and the Banach assembly for Theorem 2.

| Statement | File | Status |
|-----------|------|--------|
| Destructive cancellation: $\text{mask}(w, \text{false}) = -w$ | AnuQuantumInterference | **PROVED** (simp) |
| Constructive preservation: $\text{mask}(w, \text{true}) = w$ | AnuQuantumInterference | **PROVED** (simp) |
| Phase shift bounds: $0 \leq \phi_l < 2\pi$ | AnuQuantumInterference | **PROVED** |
| Perturbation preserves sign structure | AnuQuantumInterference | **PROVED** |
| Pipeline soundness and completeness | AnuQuantumInterference | **PROVED** (×2) |
| Deterministic execution: exec = exec | SparkDeterministicExecutor | **PROVED** (rfl) |
| Contract preservation | SparkDeterministicExecutor | **PROVED** |
| LoRA preserves base weights where adapter is zero | SparkDeterministicExecutor | **PROVED** |
| SmoothLeaky A1: $f'(x) \in (\alpha,1)$ | SmoothLeakyActivation | **PROVED** |
| SmoothLeaky A2: $f'(x) \to 1$ as $x \to +\infty$ | SmoothLeakyActivation | **PROVED** |
| SmoothLeaky A2$'$: $f'(x) \to \alpha$ as $x \to -\infty$ | SmoothLeakyActivation | **PROVED** |
| SmoothLeaky A3: $f(0) = 0$ | SmoothLeakyActivation | **PROVED** |
| SmoothLeaky A4: $f \in C^\infty$ | SmoothLeakyActivation | **PROVED** |
| SmoothLeaky negative range: $x < 0 \Rightarrow f(x) < 0$ | SmoothLeakyActivation | **PROVED** |
| MMEP energy bounded below ($\geq 0$ on manifold) | MMEP_Convergence | **PROVED** |
| Free phase energy non-increasing | MMEP_Convergence | **PROVED** |
| Equilibrium exists (zero state is feasible) | MMEP_Convergence | **PROVED** |
| EP gradient vanishes at equilibrium (zero loss gradient) | MMEP_Convergence | **PROVED** |
| Projection feasibility | MMEP_Convergence | **PROVED** |
| Memory retention stability: $\|C_1 - C_2\| \leq 2\rho_{\text{ret}}$ | MMEP_Convergence | **PROVED** |
| Spectral contraction bound ($\exists c < 1$) | MMEP_Convergence | **PROVED** |
| Training reaches feasible point | MMEP_Convergence | **PROVED** |
| Constrained-softmax temperature bisection converges | BURT_IMMA_Formalization | **PROVED** |
| Router entropy bound (conditional) | BURT_IMMA_Formalization | **PROVED** |
| Temperature positive invariant | BURT_IMMA_Formalization | **PROVED** |
| Entropy monotone in temperature (Finding 2 THEOREM) | MMEP_Convergence / paper | **PROVED** (elementarily) |
| CIFG entrywise boundedness (Finding 3 THEOREM) | BURT_IMMA_Formalization | **PROVED** (elementary induction) |
| RBM sigmoid strictly positive: $0 < \sigma(x)$ | RBM | **PROVED** |
| RBM sigmoid strictly less than 1: $\sigma(x) < 1$ | RBM | **PROVED** |
| RBM conditional probabilities valid | RBM | **PROVED** (×2) |
| RBM free energy log argument positive | RBM | **PROVED** |
| RBM bipartite energy decomposition | RBM | **PROVED** |
| RBM detailed balance ratio | RBM | **PROVED** |
| RBM free energy tractability | RBM | **PROVED** |
| RBM energy bounded by norms | RBM | **PROVED** |
| RBM sigmoid monotone increasing | RBM | **PROVED** |
| BlackHoleGravity: 30 theorems (horizon, singularity, no-hair, tidal, photon sphere, ISCO, Kerr, RN, ER=EPR, …) | BlackHoleGravity | **PROVED** (×30) |
| Lyapunov convergence of free phase (Theorem 2 assembly) | MMEP_Convergence | **SKETCH** (Banach/LaSalle assembly remains) |
| Trace conservation $\forall t$: $\text{Tr}(C_t) = \text{Tr}(C_0)$ | — | **REFUTED** (unconditional); conditional version **PLAUSIBLE** |
| Huntington postulates for Boolean ring | BooleanPerceptron | **SKETCH** |
| Sum-inversion exact reconstruction; trajectory sufficiency | SumInversionAgent | **SKETCH** |
| Interference matrix PSD; refinement contraction | SuperpositionedInduction | **SKETCH** |

The three discharged theorems in the original ledger were trivial; I report them not as trophies but as the calibration line between "the machine agrees" and "I believe." The current ledger has approximately 70 proved statements and ~20 remaining sketches. The proved column has grown; the sketch column has shrunk; the trace-conservation row has changed from a false VERIFIED to a correctly labeled REFUTED (unconditional) and PLAUSIBLE (conditional). That is progress measured honestly.

### 12.1 Adversarial Audit Responses

An adversarial mathematical audit applied two review protocols — multi-agent debate (Proponent, Adversary, Independent Verifier, Implementation Auditor, Red Team) and first-principles mathematical re-derivation — to the equations, theorems, and empirical claims of this paper and its source repository. Six findings were produced, resolved, and incorporated. The following is the complete response record.

---

**Finding 1 — Trace conservation: REFUTED as stated, CORRECTED and patched**

*Original claim:* $\forall t: \text{Tr}(C_t) = \text{Tr}(C_0)$, reported as VERIFIED.

*Audit verdict:* **REFUTED** (unconditional). Counterexample: $d=2$, $C_0=0$, $f_t=(0.5,0.5)$, $v_t=k_t=(1,1)$ gives $\text{Tr}(C_1) = 1 \neq 0$. No file in `tests/` computed $\text{Tr}(C_t)$; the README figure had no traceable executable origin. The CUDA side ships `compute_trace_kernel` and `check_trace_drift_kernel`, confirming drift is expected and monitored, not assumed absent. DEFUNCT status attaches to the *trace-conservation theorem*, not to the memory cell itself.

*Correction applied:* Section 6.4 of this paper has been rewritten with the conditional claim (uniform forget gate, trace-neutral write relative to diagonal). The unconditional row in the ledger above is labeled REFUTED. The conditional version is labeled PLAUSIBLE.

*Patch status:* The corrected conditional claim has no machine-checked Lean proof yet. Two unit tests are specified (Finding 1): one asserting invariance in the consolidation regime (uniform $f_t$, trace-neutral synthetic writes), one asserting it generically does *not* hold. Both tests are currently absent and constitute the primary open implementation obligation from this audit. **Severity: CRITICAL.**

---

**Finding 2 — Constrained softmax entropy monotonicity: SUPPORTED, now THEOREM**

*Original claim:* entropy is monotone in temperature (asserted informally).

*Audit verdict:* **SUPPORTED** — classification THEOREM (elementary). The Gibbs-distribution proof appears in Section 8.1 of this paper. $dH/d\tau \geq 0$ follows from $dH/d\beta = -\beta\,\text{Var}_p(z) \leq 0$ where $\beta = 1/\tau$; strictly increasing unless logits are degenerate.

*Patch status:* The entropy monotonicity argument is **PROVED** elementarily (hand-checkable derivation, reproduced in full in Section 8.1). Lean machine-check of the bisection convergence is **PROVED** (`temperature_bisection_converges` in `lean4/BURT_IMMA_Formalization.lean`). The full monotonicity theorem as a Lean statement remains not yet committed but the derivation is complete and correct. **No correction to paper required** — this finding upgrades an informal assertion to a labeled THEOREM. **Severity: CRITICAL if this had failed** (the entire entropy-bound mechanism depends on it) — it does not fail.

---

**Finding 3 — CIFG entrywise boundedness: PROVED by elementary induction**

*Original claim:* because $f_t \in (0,1)$ and $i_t = 1-f_t$, every entry of $C_t$ is a convex combination and therefore bounded.

*Audit verdict:* **PROVED** (elementary induction). Per entry $(i,j)$: $C_t[i,j] = f_t[i]\,C_{t-1}[i,j] + (1-f_t[i])\,v_t[i]k_t[j]$. Since $f_t[i] \in (0,1)$ strictly (sigmoid never attains 0 or 1 for finite input), this is a strict convex combination, hence $\min(C_{t-1}[i,j], v_t[i]k_t[j]) < C_t[i,j] < \max(C_{t-1}[i,j], v_t[i]k_t[j])$. Boundedness follows by induction from boundedness of the write stream. Red Team attempted counterexample (unbounded repeated writes) — refuted: the bound $\max_i(|C_{t-1}[i,j]|, |v_t[i]k_t[j]|)$ is non-increasing in the sense that it cannot exceed $\max(|C_0[i,j]|, \sup_t|v_t[i]k_t[j]|)$.

*Patch status:* **PROVED** in `lean4/BURT_IMMA_Formalization.lean` (`unified_energy_bounded` and the CIFG boundedness argument). **Severity: N/A** — claim holds as stated; this finding is flagged precisely because Finding 1 shows "bounded" and "trace-conserving" are not the same property, and the original text's rhetorical proximity of the two invited conflation.

---

**Finding 4 — Lyapunov contraction: PLAUSIBLE/SUPPORTED, Jacobian gap closed**

*Original claim:* under $\sigma_{\max}(W) \leq \lambda_{\max} < 1$ with smooth bounded-gradient activation, $E$ is a Lyapunov function with unique equilibrium at geometric rate $\lambda_{\max}$.

*Audit verdict:* **PLAUSIBLE and now further SUPPORTED**. The individual algebraic steps re-derive cleanly. The actual gap: the contraction constant of the *composite* relaxation map $(1-\alpha_{\text{relax}})I + \alpha_{\text{relax}}\,\text{diag}(\sigma'(Wh+\cdot))\,W$ was asserted rather than derived. The audit derives it explicitly: operator norm bounded by $(1-\alpha_{\text{relax}}) + \alpha_{\text{relax}}\sup|\sigma'|\cdot\sigma_{\max}(W) \leq (1-\alpha_{\text{relax}}) + \alpha_{\text{relax}}\lambda_{\max}$, which is $< 1$ using SmoothLeaky A1.

*Patch status:* The composite-Jacobian derivation is reproduced in Section 7.4. `lean4/MMEP_Convergence.lean` now contains eight machine-checked theorems covering the individual load-bearing steps: energy bounded, non-increasing, equilibrium existence, EP gradient at equilibrium, projection feasibility, memory stability, spectral contraction existence, training feasibility. The `spectral_contraction_bound` theorem is **PROVED**. The full Banach/LaSalle assembly is **SKETCH** (one remaining open obligation). **Severity: MAJOR, not CRITICAL** — this claim was never marked VERIFIED by the manuscript; it was already labeled SKETCH, and this audit narrows which specific step remains open.

---

**Finding 5 — BURT/IMMA comparative formalization: DISPUTED and HYBRID**

*BURT retrieval determinism:* `BitmapIndex.search` depends on previously indexed corpus via `.build()` — an external mutable dependency outside the stated input triple. The constrained-softmax bisection is deterministic *given* fixed iteration count but could diverge across floating-point backends. *Verdict:* **DISPUTED** — external index state is a real, previously unflagged determinism gap. **Severity: MINOR.**

*IMMA Boolean reduction:* the top-$k$ expert selection and CIFG forget/input coupling are expressible as deterministic threshold logic over real-valued gates (Heaviside-style dispatch) *after* the continuous gate values are computed; the continuous gate computation itself (a sigmoid over a learned linear projection) is not reducible to Boolean primitives without discretization. *Verdict:* IMMA is **HYBRID** — Boolean-reducible in its control flow, not in its arithmetic core. This matches what Section 10.5 does explicitly for a different subsystem; IMMA's routing *could* receive the same explicit Boolean-algebra treatment but currently does not.

*Softmax necessity:* within `constrained_softmax`, softmax is necessary *given the proof strategy actually used* (Gibbs-distribution structure). A non-exponential-family normalization would not carry the same one-line monotonicity argument. *Verdict:* **SUPPORTED, not PROVEN necessary in the absolute sense.**

*GatesNormalization:* SUPPORTED as described; conservation caveat applies (Section 8.3 above).

*Patch status:* The BitmapIndex determinism gap is flagged in the code comments and in this paper. A determinism test across two floating-point backends for BURTRetriever is specified as a required experiment (Section 7 of the audit). **No Lean patch required** for this finding; the issue is architectural documentation.

---

**Finding 6 — DEFUNCT test applied to Finding 1**

The audit applied an eight-step DEFUNCT protocol to trace conservation: (1) claimed behavior is memory trace invariant under all CIFG updates for all time; (2) required invariant is $\text{Tr}(C_t) = \text{Tr}(C_{t-1})$ every step; (3) smallest valid input is $d=1$ — trivially satisfied; (4) boundary/adversarial input is $d=2$, generic $(v,k,f)$ as in Finding 1's counterexample — fails there; (5) predicted 0, observed 1; (6) counterexample exhibited in closed form; (7) independent reproduction attempted against actual test suite: no reproduction exists because no test computes this quantity at all; (8) failure classification: *specification failure*, not implementation failure — the CIFG update code does exactly what `docs/MMEP_THEORY.md` and `imma.py` describe; the failure is that the described update does not entail the claimed invariant.

*Verdict:* **DEFUNCT** in the protocol's precise sense — the mechanism's claimed defining invariant fails and cannot currently be independently reproduced — while the underlying CIFG update mechanism (Finding 3, boundedness) is **not** DEFUNCT and continues to function exactly as specified. DEFUNCT status attaches to the trace-conservation theorem, not to the memory cell.

*Patch status:* Two required unit tests are specified and absent. This is the single highest-priority open obligation from this audit. **Severity: CRITICAL.**

---

**Audit summary table**

| Finding | Prior status | Severity | Current status | Patched? |
|---------|-------------|----------|----------------|----------|
| Trace conservation $\forall t$ | VERIFIED (in error) | CRITICAL | REFUTED unconditional; conditional PLAUSIBLE | Partial — Lean SKETCH; unit tests absent |
| Constrained-softmax entropy monotonicity | asserted informally | CRITICAL if false | **PROVED** (elementary THEOREM) | **Yes** — derivation complete, `temperature_bisection_converges` PROVED |
| CIFG entrywise boundedness | asserted | N/A | **PROVED** (elementary induction) | **Yes** — PROVED in Lean |
| Lyapunov contraction of damped relaxation | SKETCH | MAJOR | PLAUSIBLE/SUPPORTED; Jacobian gap closed | Partial — 8 sub-theorems PROVED; Banach assembly SKETCH |
| BURT retrieval determinism | implied by Layer 12 framing | MINOR | DISPUTED — external index state is a real gap | No — test specified, not yet written |
| Gate normalization boundedness | asserted | MODERATE | SUPPORTED with conservation caveat | **Yes** — documented in Section 8.3 |

**Required experiments to resolve remaining disputes.** Two unit tests for trace conservation (consolidation-regime holds, generic-regime fails); a machine-checked Lean discharge of the composite contraction bound in `lean4/MMEP_Convergence.lean`; an explicit determinism test for BURTRetriever across two floating-point backends; and, unchanged from every prior audit of this project, the three-seed ablation suite (Experiment Protocol 001) already specified in the manuscript, which remains the only path to empirical rather than derivational evidence for the architecture's central performance claims.

---

## 13 Falsification Protocol

Rosenblatt shipped a convergence theorem with his machine; Minsky and Papert shipped the counterexample that ended an era. I intend to be on the right side of that transaction by publishing my counterexamples' job descriptions in advance. Experiment Protocol 001 pre-registers the arithmetic-reasoning ablation: 100K generated expressions (addition through nested PEMDAS, five difficulty tiers, character-level tokenization over 20 symbols), eight configurations — `full_mmep`, `no_memory`, `no_constraint`, `no_moe`, `standard_bp`, `high_T_free`, `low_beta`, `large_memory` — three seeds each (42, 137, 256), paired $t$-tests with Bonferroni correction, Cohen's $d$, deterministic CUDA. Seven numbered criteria:

| # | Trigger | What dies |
|---|---------|-----------|
| F1 | `standard_bp` beats `full_mmep` by $> 2\%$ ($p < .05$) | the EP training advantage claim |
| F2 | `no_memory` $\geq$ `full_mmep` | the necessity of matrix memory |
| F3 | `no_constraint` beats by $> 2\%$ | the benefit of hard constraints |
| F4 | energy non-monotone 10+ consecutive steps | the Lyapunov guarantee (theory or bug) |
| F5 | $\sigma_{\max} > \lambda_{\max}$ after projection | spectral projection correctness |
| F6 | $H(\alpha) > 0.20 + 10^{-3}$ after projection | entropy projection correctness |
| F7 | `no_moe` $\geq$ `full_mmep` | the MoE specialization benefit |

*Status: HYPOTHESIS.* The protocol, data generator, configs, and runner ship in the repository (`train_ablation.py`, `config/ablation_arithmetic.yaml`); this paper reports no ablation accuracies because none have completed the three-seed protocol, and I will not launder a design document into a results section. What *is* reported: the self-contained integration suite — SmoothLeaky axioms, CIFG write/read round trip and trace check, router behavior, induction heads, ~50% interference cancellation, spectral bound, full pipeline forward — passes 8/8, re-executed during preparation of this paper. Note the design asymmetry: F4, F5, F6 are *always-on runtime monitors*, not one-shot experiments — the system carries its own falsifiers in production.

---

## 14 Limitations

I list these myself, because a limitations section written by reviewers is written too late. **The convergence theorem is a sketch.** Theorem 2 is the keystone and it is not machine-checked; the numerics agree everywhere I have looked, and that is evidence, not proof. **EP pays a wall-clock tax**: roughly $(T_{\text{free}} + T_{\text{nudge}}) \approx 24\times$ a backprop step at defaults; F1 is the settlement mechanism for whether the purchase (locality, no transport, explicit consolidating memory, runtime-checkable invariants) was worth it. **The exact-decoding regime is expensive**: full column rank wants $d \geq V$; at realistic vocabularies the practical regime is error-bounded, not exact. **Small scale**: every number here lives at $d \leq 4096$ on one consumer GPU; I claim nothing about three more orders of magnitude, and the memory scaling law is explicitly preliminary. **Internal benchmarks are internal**: the Enoch result and kernel speedups await external replication, and the repository's Ed25519 commercial gate is acknowledged friction against exactly the replication I am inviting; the integration suite, proofs, and protocol documents are the audit surface. **The trace-conservation claim was wrong**: the audit found it; this paper corrects it; the correction is not fully verified. **The ablation suite has not run.**

---

## 15 Conclusion and Core Insight

Part I ends where the field stands: initialize, predict, calculate loss, nudge weights, repeat — capability as the quiet, relentless accumulation of small corrections in high-dimensional space. Part II keeps every word of that except the mechanism of the nudge. *Learning should settle, not propagate.* The LSTM-descended CIFG cell gives the system a memory whose writes are convex combinations and whose reads carry error bounds. Equilibrium propagation gives it a learning rule a synapse could implement — two snapshots, one difference, no tape, no transported weights. The constrained softmax and spectral projection give it invariants that hold by construction and are watched at runtime by kernels that treat violation as an error code. The SmoothLeaky activation gives the fixed-point theorem its smoothness. The Boolean actors give Rosenblatt's unit the algebra Minsky said it lacked. The superposition layer takes the field's strangest emergent accident and puts it on the payroll.

And the adversarial audit gave the paper something equally valuable: a refuted claim (trace conservation), a proved theorem (entropy monotonicity), a derived bound (composite Jacobian), and a clearly labeled disputed gap (BitmapIndex determinism). The ledger now has approximately 70 proved statements, roughly 20 sketches, one REFUTED claim corrected, and one DEFUNCT test specified and awaiting implementation. That is the honest distance between what has been built and what has been proved.

The falsification suite is armed, the criteria are numbered, the fixed point is waiting.

*Evidence or silence.*

---

## References

[1] W. S. McCulloch and W. Pitts. A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5:115–133, 1943.

[2] F. Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6):386–408, 1958.

[3] M. Minsky and S. Papert. *Perceptrons: An Introduction to Computational Geometry*. MIT Press, 1969.

[4] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. *Nature*, 323:533–536, 1986.

[5] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In *NeurIPS*, 2012.

[6] S. Hochreiter and J. Schmidhuber. Long short-term memory. *Neural Computation*, 9(8):1735–1780, 1997.

[7] K. Greff, R. K. Srivastava, J. Koutník, B. R. Steunebrink, and J. Schmidhuber. LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10):2222–2232, 2017.

[8] A. Vaswani et al. Attention is all you need. In *NeurIPS*, 2017.

[9] B. Scellier and Y. Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in Computational Neuroscience*, 11:24, 2017.

[10] W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.

[11] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In *ICLR*, 2018.

[12] J. Hoffmann et al. Training compute-optimal large language models. *arXiv:2203.15556*, 2022.

[13] J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8):2554–2558, 1982.

[14] H. Ramsauer et al. Hopfield networks is all you need. In *ICLR*, 2021.

[15] I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In *ICML*, 2021.

[16] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In *ICML*, 2020.

[17] Z. Yang, Z. Dai, R. Salakhutdinov, and W. W. Cohen. Breaking the softmax bottleneck: A high-rank language model. In *ICLR*, 2018.

[18] E. V. Huntington. Sets of independent postulates for the algebra of logic. *Transactions of the American Mathematical Society*, 5(3):288–309, 1904.

[19] Adversarial Mathematical Audit of BURT-IMMA: Multi-agent debate and first-principles re-derivation of the central claims. *Internal audit document*, SnapKitty West, August 2026.
