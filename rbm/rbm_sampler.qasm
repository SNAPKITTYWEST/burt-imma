// rbm_sampler.qasm
// OpenQASM 3.0: Quantum RBM Gibbs Sampler
// Bernoulli sampling via RY rotation gate (novel gate decomposition)
OPENQASM 3.0;
include "stdgates.inc";

// Novel gate: sigmoid_rotation
// RY(2*arctan(exp(-theta))) maps |0⟩ → √(1-σ(θ))|0⟩ + √σ(θ)|1⟩
// Measuring in computational basis samples Bernoulli(σ(θ))
gate sigmoid_rotation(theta) q {
    ry(2.0 * arctan(exp(-theta))) q;
}

// Bilinear energy phase gate for ZZ coupling in energy phase
gate zz_energy(beta_w) a, b {
    cx a, b;
    rz(2.0 * beta_w) b;
    cx a, b;
}

// Single-qubit energy phase: e^{iβ b_i Z_i}
gate local_energy(beta_b) q {
    rz(2.0 * beta_b) q;
}

// Mixer layer: e^{-iγ X}
gate mix(gamma) q {
    rx(2.0 * gamma) q;
}

// ============================================================
// Parameters (loaded at runtime from classical driver)
// ============================================================
const int NV = 6;  // visible units
const int NH = 2;  // hidden units
const int DEPTH = 4;

float[64] W;  // W[i*NH + j] = W_{ij}
float[6]  b;  // visible bias
float[2]  c;  // hidden bias
float     beta = 1.0;
float     gamma_mix = 0.5;

// ============================================================
// Registers
// ============================================================
qubit[6] v;
qubit[2] h;
bit[6]   v_meas;
bit[2]   h_meas;

// ============================================================
// Circuit: QAOA Gibbs Preparation
// ============================================================

// Step 1: Uniform superposition
h v;
h h;

// Step 2: QAOA layers
for int layer in [0:DEPTH-1] {
    // Energy phase: ZZ couplings v_i -- h_j
    for int i in [0:NV-1] {
        for int j in [0:NH-1] {
            zz_energy(beta * W[i * NH + j]) v[i], h[j];
        }
    }
    // Local fields
    for int i in [0:NV-1] {
        local_energy(beta * b[i]) v[i];
    }
    for int j in [0:NH-1] {
        local_energy(beta * c[j]) h[j];
    }

    // Mixer
    for int i in [0:NV-1] {
        mix(gamma_mix) v[i];
    }
    for int j in [0:NH-1] {
        mix(gamma_mix) h[j];
    }
}

// Step 3: Measure visible units
for int i in [0:NV-1] {
    v_meas[i] = measure v[i];
}

// Step 4: Condition on measurement, sample hidden (sigmoid_rotation)
// Classical post-processing: compute a_j = sum_i W[i,j]*v_meas[i] + c[j]
// Then apply sigmoid_rotation(a_j) to fresh ancilla qubits for h samples
// (Illustrated for first hidden unit; general case requires classical feed-forward)
qubit[2] h_fresh;
// sigmoid_rotation(c[0]) h_fresh[0];   // placeholder: a_j computed classically
// sigmoid_rotation(c[1]) h_fresh[1];
for int j in [0:NH-1] {
    h_meas[j] = measure h[j];
}
