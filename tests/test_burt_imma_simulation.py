#!/usr/bin/env python3
"""
BURT-IMMA Self-Contained Integration Test
License: BSL-1.1
Contact: jessica@collectivekitty.com

Self-contained minimal implementations for testing.
Does NOT import from burt_imma package.
"""

import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Minimal implementations (numpy only, no torch dependency)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class SmoothLeakyActivation:
    """Smooth leaky activation: alpha * x + (1-alpha) * x * sigmoid(x), alpha=0.01"""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * x + (1.0 - self.alpha) * x * _sigmoid(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        sig = _sigmoid(x)
        # d/dx [alpha*x + (1-alpha)*x*sig(x)]
        # = alpha + (1-alpha)*(sig(x) + x*sig(x)*(1-sig(x)))
        return self.alpha + (1.0 - self.alpha) * (sig + x * sig * (1.0 - sig))


class GatesNormalization:
    """Softmax normalization that ensures sum=1 and entropy <= 0.20"""

    def __init__(self, max_entropy: float = 0.20):
        self.max_entropy = max_entropy

    def forward(self, logits: np.ndarray) -> np.ndarray:
        # Start with softmax
        probs = _softmax(logits, axis=-1)
        # Iteratively sharpen until entropy bound is satisfied
        temperature = 1.0
        for _ in range(100):
            entropy = self._entropy(probs)
            if entropy <= self.max_entropy:
                break
            temperature *= 0.8
            probs = _softmax(logits / temperature, axis=-1)
        return probs

    def _entropy(self, p: np.ndarray) -> float:
        p_clipped = np.clip(p, 1e-10, 1.0)
        return float(-np.sum(p_clipped * np.log(p_clipped), axis=-1).mean())


class CIFGMatrixMemory:
    """Coupled Input-Forget Gate Matrix Memory.
    C_new = f * C_old + (1-f) * candidate
    """

    def __init__(self, d_mem: int):
        self.d_mem = d_mem
        self.C = np.zeros((d_mem, d_mem), dtype=np.float64)

    def write(self, key: np.ndarray, value: np.ndarray, forget_gate: float = 0.9) -> None:
        # Outer product as candidate
        k = key.flatten()[:self.d_mem]
        v = value.flatten()[:self.d_mem]
        candidate = np.outer(k, v)
        candidate = candidate / (np.linalg.norm(candidate) + 1e-8)
        # CIFG update: C_new = f * C_old + (1-f) * candidate
        self.C = forget_gate * self.C + (1.0 - forget_gate) * candidate

    def read(self, query: np.ndarray) -> np.ndarray:
        q = query.flatten()[:self.d_mem]
        return self.C @ q

    def reset(self) -> None:
        self.C = np.zeros((self.d_mem, self.d_mem), dtype=np.float64)


class GatesRouter:
    """Routes input to top-k experts based on gating scores."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 1):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Random gating weights
        self.W_gate = np.random.randn(input_dim, num_experts) * 0.01

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (expert_indices, expert_weights) for top-k experts."""
        # x: (batch, input_dim)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        scores = x @ self.W_gate  # (batch, num_experts)
        probs = _softmax(scores, axis=-1)

        # Top-k selection
        top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
        top_k_weights = np.take_along_axis(probs, top_k_indices, axis=-1)
        # Renormalize weights
        top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-8)

        return top_k_indices, top_k_weights

    def load_balance_loss(self, x: np.ndarray) -> float:
        """Compute load balancing auxiliary loss."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        scores = x @ self.W_gate
        probs = _softmax(scores, axis=-1)
        # Fraction of tokens routed to each expert
        routing_probs = probs.mean(axis=0)
        # Ideal uniform distribution
        uniform = np.ones(self.num_experts) / self.num_experts
        # L2 deviation from uniform
        return float(np.sum((routing_probs - uniform) ** 2))


class SuperpositionedInductionHeads:
    """Attention pattern that detects repeated subsequences."""

    def __init__(self, d_model: int, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Random projections for Q, K, V
        self.W_Q = np.random.randn(n_heads, d_model, self.d_head) * 0.02
        self.W_K = np.random.randn(n_heads, d_model, self.d_head) * 0.02
        self.W_V = np.random.randn(n_heads, d_model, self.d_head) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (seq_len, d_model) -> output: (seq_len, d_model)"""
        seq_len = x.shape[0]
        outputs = []

        for h in range(self.n_heads):
            Q = x @ self.W_Q[h]  # (seq_len, d_head)
            K = x @ self.W_K[h]
            V = x @ self.W_V[h]

            # Scaled dot-product attention
            scores = Q @ K.T / np.sqrt(self.d_head)  # (seq_len, seq_len)

            # Causal mask
            mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
            scores = scores + mask

            # Induction bias: boost attention to positions where previous token matches
            for i in range(1, seq_len):
                for j in range(1, i):
                    # If token at j-1 is similar to token at i-1, boost score at (i, j)
                    similarity = np.dot(x[j - 1], x[i - 1]) / (
                        np.linalg.norm(x[j - 1]) * np.linalg.norm(x[i - 1]) + 1e-8
                    )
                    if similarity > 0.8:
                        scores[i, j] += 2.0

            attn = _softmax(scores, axis=-1)
            out = attn @ V  # (seq_len, d_head)
            outputs.append(out)

        # Concatenate heads
        return np.concatenate(outputs, axis=-1)[:, :self.d_model]

    def detect_repeats(self, x: np.ndarray) -> List[Tuple[int, int]]:
        """Detect positions where repeated subsequences occur."""
        seq_len = x.shape[0]
        repeats = []
        for i in range(2, seq_len):
            for j in range(1, i):
                sim = np.dot(x[j - 1], x[i - 1]) / (
                    np.linalg.norm(x[j - 1]) * np.linalg.norm(x[i - 1]) + 1e-8
                )
                if sim > 0.9:
                    repeats.append((j, i))
        return repeats


class QuantumInterferenceResolver:
    """Resolves superposed states via Born rule (|psi|^2 normalization)."""

    def __init__(self, d_state: int):
        self.d_state = d_state

    def resolve(self, psi: np.ndarray) -> np.ndarray:
        """Apply Born rule: probability = |psi|^2 / sum(|psi|^2)"""
        amplitudes_sq = np.abs(psi) ** 2
        total = np.sum(amplitudes_sq)
        if total < 1e-12:
            return np.ones_like(amplitudes_sq) / len(amplitudes_sq)
        return amplitudes_sq / total

    def interfere(self, psi1: np.ndarray, psi2: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """Combine two states with interference: psi1 + e^(i*phase) * psi2"""
        # Treat as complex amplitudes
        psi1_c = psi1.astype(np.complex128)
        psi2_c = psi2.astype(np.complex128)
        combined = psi1_c + np.exp(1j * phase) * psi2_c
        return self.resolve(np.abs(combined))

    def measure(self, psi: np.ndarray) -> int:
        """Collapse to a definite state via sampling from Born distribution."""
        probs = self.resolve(psi)
        return int(np.random.choice(len(probs), p=probs))


class BURT_IMMA:
    """Full BURT-IMMA pipeline connecting all components."""

    def __init__(self, d_model: int = 64, num_experts: int = 4, top_k: int = 1,
                 d_mem: int = 64, n_heads: int = 4):
        self.d_model = d_model
        self.activation = SmoothLeakyActivation(alpha=0.01)
        self.gates_norm = GatesNormalization(max_entropy=0.20)
        self.memory = CIFGMatrixMemory(d_mem=d_mem)
        self.router = GatesRouter(input_dim=d_model, num_experts=num_experts, top_k=top_k)
        self.induction_heads = SuperpositionedInductionHeads(d_model=d_model, n_heads=n_heads)
        self.quantum_resolver = QuantumInterferenceResolver(d_state=d_model)

        # Expert weight matrices
        self.experts = [
            np.random.randn(d_model, d_model) * 0.02
            for _ in range(num_experts)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Full forward pass.
        x: (seq_len, d_model)
        returns: (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # 1. Activation
        h = self.activation.forward(x)

        # 2. Induction heads (attention)
        h = self.induction_heads.forward(h)

        # 3. Route through experts (per-token)
        output = np.zeros_like(h)
        for t in range(seq_len):
            indices, weights = self.router.route(h[t:t+1])
            token_out = np.zeros(self.d_model)
            for k_idx in range(indices.shape[1]):
                expert_id = indices[0, k_idx]
                weight = weights[0, k_idx]
                expert_out = h[t] @ self.experts[expert_id]
                token_out += weight * expert_out
            output[t] = token_out

        # 4. Memory interaction
        # Write final representation
        if seq_len > 0:
            self.memory.write(output[-1], output[0], forget_gate=0.9)

        # 5. Activation again
        output = self.activation.forward(output)

        # 6. Quantum interference resolution on gating logits
        for t in range(seq_len):
            resolved = self.quantum_resolver.resolve(np.abs(output[t]))
            output[t] = output[t] * resolved

        # 7. Gates normalization on output distribution
        output_norm = self.gates_norm.forward(np.abs(output) + 1e-8)
        output = output * output_norm

        return output


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------


def test_smooth_leaky():
    """Verify output shape, alpha blending, and gradient existence."""
    act = SmoothLeakyActivation(alpha=0.01)
    x = np.random.randn(32, 64)

    # Test output shape
    out = act.forward(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    # Test alpha blending: for large positive x, sigmoid(x) -> 1, so output ~ x
    x_large = np.array([10.0, 20.0, 50.0])
    out_large = act.forward(x_large)
    np.testing.assert_allclose(out_large, x_large, rtol=0.05,
                               err_msg="Alpha blending failed for large positive inputs")

    # Test gradient exists and is positive for positive inputs
    grad = act.gradient(x)
    assert grad.shape == x.shape, f"Gradient shape mismatch: {grad.shape}"
    # For positive x, gradient should be positive
    x_pos = np.abs(x) + 0.1
    grad_pos = act.gradient(x_pos)
    assert np.all(grad_pos > 0), "Gradient should be positive for positive inputs"

    print("  [PASS] test_smooth_leaky")


def test_gates_normalization():
    """Verify sum=1 and entropy bound."""
    gn = GatesNormalization(max_entropy=0.20)

    logits = np.random.randn(8)
    probs = gn.forward(logits)

    # Verify sum = 1
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-6,
                               err_msg="Probabilities do not sum to 1")

    # Verify entropy bound
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    assert entropy <= 0.20 + 1e-6, f"Entropy {entropy:.4f} exceeds bound 0.20"

    # Test batch
    logits_batch = np.random.randn(4, 8)
    probs_batch = gn.forward(logits_batch)
    row_sums = np.sum(probs_batch, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6,
                               err_msg="Batch probabilities do not sum to 1")

    print("  [PASS] test_gates_normalization")


def test_cifg_memory():
    """Verify write then read recovers content, verify forget gate."""
    d_mem = 32
    mem = CIFGMatrixMemory(d_mem=d_mem)

    # Write a known pattern
    key = np.random.randn(d_mem)
    value = np.random.randn(d_mem)
    key_norm = key / np.linalg.norm(key)
    value_norm = value / np.linalg.norm(value)

    mem.write(key_norm, value_norm, forget_gate=0.0)  # forget_gate=0 means full write

    # Read back with the same key
    retrieved = mem.read(key_norm)
    # Memory stores normalized outer(key, value), read is C @ query.
    # With query=key_norm: result = (key_norm^T key_norm) * value_norm / norm_factor
    # Since key_norm is unit vector, key_norm^T key_norm = 1, so retrieved ~ value_norm / norm
    # The retrieved vector should point in same direction as value_norm
    correlation = np.dot(retrieved, value_norm) / (
        np.linalg.norm(retrieved) * np.linalg.norm(value_norm) + 1e-8
    )
    assert correlation > 0.1, f"Retrieval correlation too low: {correlation:.4f}"

    # Verify stronger property: reading with the exact key used for write
    # gives a result aligned with value (dot product > 0)
    assert np.dot(retrieved, value_norm) > 0, "Retrieved vector should align with stored value"

    # Test forget gate: writing with forget_gate=1.0 should not change memory
    mem2 = CIFGMatrixMemory(d_mem=d_mem)
    mem2.C = np.eye(d_mem)
    old_C = mem2.C.copy()
    mem2.write(key_norm, value_norm, forget_gate=1.0)
    np.testing.assert_allclose(mem2.C, old_C, atol=1e-10,
                               err_msg="Forget gate=1.0 should preserve memory")

    print("  [PASS] test_cifg_memory")


def test_gates_router():
    """Verify top-k routing and load balancing."""
    input_dim = 64
    num_experts = 4
    top_k = 2
    router = GatesRouter(input_dim=input_dim, num_experts=num_experts, top_k=top_k)

    x = np.random.randn(16, input_dim)
    indices, weights = router.route(x)

    # Verify top-k shape
    assert indices.shape == (16, top_k), f"Indices shape: {indices.shape}"
    assert weights.shape == (16, top_k), f"Weights shape: {weights.shape}"

    # Verify weights sum to 1 per sample
    weight_sums = weights.sum(axis=-1)
    np.testing.assert_allclose(weight_sums, np.ones(16), atol=1e-6,
                               err_msg="Router weights don't sum to 1")

    # Verify all indices are valid expert IDs
    assert np.all(indices >= 0) and np.all(indices < num_experts), \
        "Invalid expert indices"

    # Verify load balancing loss is computable
    lb_loss = router.load_balance_loss(x)
    assert lb_loss >= 0, "Load balance loss should be non-negative"
    assert np.isfinite(lb_loss), "Load balance loss should be finite"

    print("  [PASS] test_gates_router")


def test_induction_heads():
    """Verify pattern detection on repeated sequence."""
    d_model = 32
    heads = SuperpositionedInductionHeads(d_model=d_model, n_heads=4)

    # Create a sequence with repeated pattern: [A, B, C, A, B, ?]
    # The induction head should detect that after A,B previously came C
    np.random.seed(42)
    A = np.random.randn(d_model)
    B = np.random.randn(d_model)
    C = np.random.randn(d_model)
    D = np.random.randn(d_model)

    # Normalize
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    C = C / np.linalg.norm(C)
    D = D / np.linalg.norm(D)

    # Sequence with exact repeat
    seq = np.stack([A, B, C, A, B, D])  # (6, d_model)

    # Detect repeats
    repeats = heads.detect_repeats(seq)
    # Position 3 (second A) should detect similarity with position 0 (first A) via prev token
    # x[j-1] similar to x[i-1]: when j=4,i=1 => x[3]=A sim x[0]=A => repeat (4,1)
    found_repeat = any(
        np.dot(seq[j-1], seq[i-1]) / (np.linalg.norm(seq[j-1]) * np.linalg.norm(seq[i-1])) > 0.9
        for j, i in repeats
    ) if repeats else False

    # Forward pass should produce valid output
    output = heads.forward(seq)
    assert output.shape == (6, d_model), f"Output shape: {output.shape}"
    assert np.all(np.isfinite(output)), "Output contains non-finite values"

    # At minimum, verify the mechanism finds the A->A repeat
    assert found_repeat or len(repeats) > 0, "Induction head should detect repeated patterns"

    print("  [PASS] test_induction_heads")


def test_quantum_interference():
    """Verify Born rule normalization (sum of |psi|^2 = 1)."""
    d_state = 16
    resolver = QuantumInterferenceResolver(d_state=d_state)

    # Random complex-like amplitudes
    psi = np.random.randn(d_state) + 0.5

    # Resolve via Born rule
    probs = resolver.resolve(psi)

    # Sum must equal 1
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-10,
                               err_msg="Born rule: probabilities must sum to 1")

    # All probabilities must be non-negative
    assert np.all(probs >= 0), "Born rule: probabilities must be non-negative"

    # Test interference
    psi1 = np.random.randn(d_state)
    psi2 = np.random.randn(d_state)
    probs_interfered = resolver.interfere(psi1, psi2, phase=np.pi)
    np.testing.assert_allclose(np.sum(probs_interfered), 1.0, atol=1e-10,
                               err_msg="Interference: probabilities must sum to 1")

    # Test measurement
    np.random.seed(123)
    state = resolver.measure(psi)
    assert 0 <= state < d_state, f"Measured state {state} out of range"

    # Test zero vector handling
    psi_zero = np.zeros(d_state)
    probs_zero = resolver.resolve(psi_zero)
    np.testing.assert_allclose(np.sum(probs_zero), 1.0, atol=1e-10,
                               err_msg="Zero vector should produce uniform distribution")

    print("  [PASS] test_quantum_interference")


def test_full_pipeline():
    """Verify end-to-end forward pass produces valid output."""
    np.random.seed(42)

    model = BURT_IMMA(d_model=32, num_experts=4, top_k=1, d_mem=32, n_heads=4)

    # Input sequence
    seq_len = 8
    x = np.random.randn(seq_len, 32)

    # Forward pass
    output = model.forward(x)

    # Check shape
    assert output.shape == (seq_len, 32), f"Output shape: {output.shape}"

    # Check finiteness
    assert np.all(np.isfinite(output)), "Output contains non-finite values"

    # Check non-trivial (not all zeros)
    assert np.any(np.abs(output) > 1e-10), "Output is trivially zero"

    # Check memory was written
    assert np.any(np.abs(model.memory.C) > 1e-10), "Memory should have been written"

    # Second forward pass should also work (tests statefulness)
    x2 = np.random.randn(seq_len, 32)
    output2 = model.forward(x2)
    assert output2.shape == (seq_len, 32), "Second pass shape incorrect"
    assert np.all(np.isfinite(output2)), "Second pass contains non-finite values"

    print("  [PASS] test_full_pipeline")


def test_spectral_norm():
    """Verify weight matrices have bounded spectral norm."""
    np.random.seed(42)

    model = BURT_IMMA(d_model=32, num_experts=4, top_k=1, d_mem=32, n_heads=4)

    # Check spectral norm of expert matrices
    for i, W in enumerate(model.experts):
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        spectral_norm = s[0]
        # Randomly initialized with scale 0.02, so spectral norm should be bounded
        assert spectral_norm < 5.0, \
            f"Expert {i} spectral norm {spectral_norm:.4f} exceeds bound"
        assert spectral_norm > 0.0, \
            f"Expert {i} spectral norm should be positive"

    # Check induction head projection matrices
    for h in range(model.induction_heads.n_heads):
        for name, W in [("Q", model.induction_heads.W_Q[h]),
                        ("K", model.induction_heads.W_K[h]),
                        ("V", model.induction_heads.W_V[h])]:
            _, s, _ = np.linalg.svd(W, full_matrices=False)
            spectral_norm = s[0]
            assert spectral_norm < 5.0, \
                f"Head {h} {name} spectral norm {spectral_norm:.4f} exceeds bound"

    # Verify spectral norm computation is correct
    W_test = np.array([[3.0, 0.0], [0.0, 2.0]])
    _, s_test, _ = np.linalg.svd(W_test, full_matrices=False)
    np.testing.assert_allclose(s_test[0], 3.0, atol=1e-10,
                               err_msg="SVD spectral norm computation is wrong")

    print("  [PASS] test_spectral_norm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("BURT-IMMA Self-Contained Integration Tests")
    print("=" * 50)

    tests = [
        test_smooth_leaky,
        test_gates_normalization,
        test_cifg_memory,
        test_gates_router,
        test_induction_heads,
        test_quantum_interference,
        test_full_pipeline,
        test_spectral_norm,
    ]

    failures = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            failures.append((test_fn.__name__, str(e)))
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print("=" * 50)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)} tests failed")
        for name, err in failures:
            print(f"  - {name}: {err}")
        raise SystemExit(1)
    else:
        print("ALL TESTS PASSED")
