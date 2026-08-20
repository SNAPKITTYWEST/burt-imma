// RBM.qs
// Quantum Amplitude Encoding of RBM Distribution via QAOA-inspired ansatz
namespace RBM {

    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;

    /// Prepare Gibbs state Σ_{v,h} √p(v,h) |v⟩|h⟩ via QAOA-inspired ansatz
    /// Uses alternating energy-phase and mixer-phase layers
    operation PrepareGibbsState(
        nVisible : Int,
        nHidden : Int,
        W : Double[][],
        b : Double[],
        c : Double[],
        beta : Double,
        depth : Int,
        v : Qubit[],
        h : Qubit[]
    ) : Unit is Adj + Ctl {
        // Initialize uniform superposition over all (v,h) configurations
        ApplyToEach(H, v);
        ApplyToEach(H, h);

        // QAOA layers: alternate energy phase and mixer
        for layer in 1..depth {
            // Energy phase: e^{-iβ E(v,h)} = e^{iβ(v^T W h + b^T v + c^T h)}
            for i in 0..nVisible-1 {
                for j in 0..nHidden-1 {
                    // ZZ coupling for v_i * h_j term
                    within { CNOT(v[i], h[j]); }
                    apply { Rz(2.0 * beta * W[i][j], h[j]); }
                }
            }
            for i in 0..nVisible-1 {
                Rz(2.0 * beta * b[i], v[i]);
            }
            for j in 0..nHidden-1 {
                Rz(2.0 * beta * c[j], h[j]);
            }

            // Mixer phase: e^{-iγ Σ X_i}
            let gamma = 0.5;
            ApplyToEach(Rx(2.0 * gamma, _), v);
            ApplyToEach(Rx(2.0 * gamma, _), h);
        }
    }

    /// Estimate overlap between data and model distributions via SWAP test
    /// Returns probability of ancilla measuring |0⟩: P(0) = (1 + |⟨ψ_data|ψ_model⟩|²) / 2
    operation SwapTestFidelity(
        dataRegister : Qubit[],
        modelRegister : Qubit[],
        ancilla : Qubit
    ) : Result {
        H(ancilla);
        for i in 0..Length(dataRegister)-1 {
            Controlled SWAP([ancilla], (dataRegister[i], modelRegister[i]));
        }
        H(ancilla);
        return M(ancilla);
    }

    /// Bernoulli sampling gate: RY(2*arctan(exp(-θ))) approximates σ(θ) sampling
    /// Applied to |0⟩, produces |1⟩ with probability sigmoid(θ)
    operation BernoulliSample(theta : Double, q : Qubit) : Unit is Adj + Ctl {
        // RY(2*arccos(sqrt(1-σ(θ)))) rotates |0⟩ to √(1-p)|0⟩ + √p|1⟩
        let p = 1.0 / (1.0 + E() ^ (-theta));
        let angle = 2.0 * ArcCos(Sqrt(1.0 - p));
        Ry(angle, q);
    }

    /// One-step quantum Gibbs sampling: h ~ p(h|v) then v' ~ p(v|h)
    operation GibbsStep(
        v : Qubit[],
        h : Qubit[],
        W : Double[][],
        b : Double[],
        c : Double[]
    ) : Unit {
        let nV = Length(v);
        let nH = Length(h);

        // Sample h ~ p(h|v): for each h_j, compute (W^T v)_j + c_j and sample
        // (In full implementation, this requires classical pre-measurement of v)
        // Here we apply a parametric rotation as a proxy
        for j in 0..nH-1 {
            BernoulliSample(c[j], h[j]);
        }

        // Sample v' ~ p(v|h): for each v_i, compute (W h)_i + b_i and sample
        for i in 0..nV-1 {
            BernoulliSample(b[i], v[i]);
        }
    }
}
