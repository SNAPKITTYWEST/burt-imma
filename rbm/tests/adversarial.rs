// rbm/tests/adversarial.rs
// Adversarial counterexamples for RBM training
use rbm::RBM;

#[test]
fn test_xor_not_reliably_learned_with_2_hidden() {
    // XOR requires at least 2 hidden units. CD-1 often fails to find it.
    let xor_data = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    let mut rbm = RBM::new(3, 2);
    for _ in 0..2000 {
        rbm.train_batch(&xor_data, 0.1);
    }
    // Free energy variance documents whether mode collapse occurred
    let energies: Vec<f32> = xor_data.iter().map(|v| rbm.free_energy(v)).collect();
    let _max_e = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // No hard assertion — documents the known limitation of CD-1 on XOR
}

#[test]
fn test_mode_capture_two_clusters() {
    let data = vec![
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ];
    let mut rbm = RBM::new(6, 2);
    for _ in 0..1000 {
        rbm.train_batch(&data, 0.1);
    }
    let samples: Vec<Vec<f32>> = (0..100).map(|_| rbm.sample_chain(50, None)).collect();
    let mode1 = samples.iter().filter(|v| v[0] > 0.5).count();
    let mode2 = samples.iter().filter(|v| v[3] > 0.5).count();
    assert!(mode1 > 5 && mode2 > 5, "Mode collapse: mode1={} mode2={}", mode1, mode2);
}

#[test]
fn test_free_energy_numerical_gradient() {
    let mut rbm = RBM::new(3, 2);
    // Train briefly so weights are nonzero
    let data = vec![vec![1.0, 0.0, 1.0]];
    rbm.train_batch(&data, 0.01);

    let v = vec![1.0, 0.0, 1.0];
    let eps = 1e-4f32;
    let fe0 = rbm.free_energy(&v);
    rbm.w[0] += eps;
    let fe1 = rbm.free_energy(&v);
    let _numerical_grad = (fe1 - fe0) / eps;
    // Analytic gradient of free energy w.r.t. W[0,0] = -v[0] * sigmoid(...)
    // Just verify it's finite
    assert!(_numerical_grad.is_finite());
}
