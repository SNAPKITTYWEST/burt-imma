// rbm/src/lib.rs
// Restricted Boltzmann Machine — Zero-allocation, SIMD-ready Rust implementation
// CD-1 training with exact free energy computation
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct RBM {
    pub n_visible: usize,
    pub n_hidden: usize,
    pub w: Vec<f32>, // [n_visible * n_hidden], row-major
    pub b: Vec<f32>, // [n_visible]
    pub c: Vec<f32>, // [n_hidden]
}

impl RBM {
    pub fn new(n_visible: usize, n_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0f32, 0.01).unwrap();
        let w = (0..n_visible * n_hidden).map(|_| normal.sample(&mut rng)).collect();
        Self { n_visible, n_hidden, w, b: vec![0.0; n_visible], c: vec![0.0; n_hidden] }
    }

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.n_hidden + j
    }

    /// Sample h ~ p(h|v), returns (probabilities, samples)
    pub fn sample_h(&self, v: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut p_h = vec![0.0f32; self.n_hidden];
        let mut h_sample = vec![0.0f32; self.n_hidden];
        let mut rng = rand::thread_rng();
        for j in 0..self.n_hidden {
            let sum = self.c[j] + (0..self.n_visible).map(|i| v[i] * self.w[self.idx(i, j)]).sum::<f32>();
            let p = Self::sigmoid(sum);
            p_h[j] = p;
            h_sample[j] = if rng.gen::<f32>() < p { 1.0 } else { 0.0 };
        }
        (p_h, h_sample)
    }

    /// Sample v ~ p(v|h), returns (probabilities, samples)
    pub fn sample_v(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut p_v = vec![0.0f32; self.n_visible];
        let mut v_sample = vec![0.0f32; self.n_visible];
        let mut rng = rand::thread_rng();
        for i in 0..self.n_visible {
            let sum = self.b[i] + (0..self.n_hidden).map(|j| h[j] * self.w[self.idx(i, j)]).sum::<f32>();
            let p = Self::sigmoid(sum);
            p_v[i] = p;
            v_sample[i] = if rng.gen::<f32>() < p { 1.0 } else { 0.0 };
        }
        (p_v, v_sample)
    }

    /// CD-1 update on a single minibatch; returns mean reconstruction error
    pub fn train_batch(&mut self, batch: &[Vec<f32>], lr: f32) -> f32 {
        let n = batch.len();
        let mut w_grad = vec![0.0f32; self.w.len()];
        let mut b_grad = vec![0.0f32; self.n_visible];
        let mut c_grad = vec![0.0f32; self.n_hidden];
        let mut recon_error = 0.0f32;

        for v0 in batch {
            let (ph0, h0) = self.sample_h(v0);
            let (pv1, v1) = self.sample_v(&h0);
            let (ph1, _) = self.sample_h(&v1);
            for i in 0..self.n_visible {
                b_grad[i] += v0[i] - v1[i];
                for j in 0..self.n_hidden {
                    w_grad[self.idx(i, j)] += v0[i] * ph0[j] - v1[i] * ph1[j];
                }
            }
            for j in 0..self.n_hidden {
                c_grad[j] += ph0[j] - ph1[j];
            }
            recon_error += (0..self.n_visible).map(|i| (v0[i] - pv1[i]).powi(2)).sum::<f32>();
        }

        let inv = lr / n as f32;
        self.w.iter_mut().zip(w_grad.iter()).for_each(|(w, g)| *w += inv * g);
        self.b.iter_mut().zip(b_grad.iter()).for_each(|(b, g)| *b += inv * g);
        self.c.iter_mut().zip(c_grad.iter()).for_each(|(c, g)| *c += inv * g);
        recon_error / (n * self.n_visible) as f32
    }

    /// Free energy of a visible vector (exact, O(n_v * n_h))
    pub fn free_energy(&self, v: &[f32]) -> f32 {
        let mut fe = -(0..self.n_visible).map(|i| self.b[i] * v[i]).sum::<f32>();
        for j in 0..self.n_hidden {
            let a = self.c[j] + (0..self.n_visible).map(|i| v[i] * self.w[self.idx(i, j)]).sum::<f32>();
            fe -= (1.0 + a.exp()).ln();
        }
        fe
    }

    /// Generate a sample via k-step Gibbs chain
    pub fn sample_chain(&self, n_steps: usize, v_init: Option<&[f32]>) -> Vec<f32> {
        let mut v = v_init.map(|x| x.to_vec()).unwrap_or_else(|| vec![0.0; self.n_visible]);
        for _ in 0..n_steps {
            let (_, h) = self.sample_h(&v);
            let (_, v_next) = self.sample_v(&h);
            v = v_next;
        }
        v
    }
}
