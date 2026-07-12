//! # SpikeGrad bridge — learnable spike-to-drive projection with BPTT through T3.
//!
//! The bridge solves the core problem of spike-based computation:
//! **how to backpropagate through discrete spikes**.
//!
//! ## Architecture
//!
//! ```text
//! spikes[t] (T × spike_dim)
//!     │
//!     ▼  W × spikes[t] + b  (linear projection)
//! drives[t] (T × 3·n_osc)
//!     │
//!     ▼  T3 oscillator dynamics (differentiable physics)
//! T3 trajectory
//!     │
//!     ▼  PST features (sin, cos, velocity, order parameters)
//! PST[t] (T × pst_dim)
//!     │
//!     ▼  decoder → behavior
//! loss
//! ```
//!
//! Gradient flows back through ALL layers via BPTT:
//! `dLoss/dW = Σ_t dLoss/d(drives[t]) · d(drives[t])/dW`
//!
//! This is **safe backprop for spikes** — no surrogate gradients,
//! no spike-cutting, through smooth physical dynamics (T3 oscillators).

use crate::spike::{T3SpikeTrajectory, t3spike_forward, t3spike_backward_per_step};

/// Learnable linear bridge: spike features → T3 drives.
///
/// Holds projection weights W (spike_dim × drive_dim) and bias b (drive_dim),
/// plus T3 configuration parameters.
pub struct SpikeGradBridge {
    // ── Learnable bridge weights ──
    /// Projection matrix: [spike_dim × drive_dim], row-major.
    /// drive[t, j] = Σ_i spikes[t, i] × W[i, j] + b[j]
    pub w: Vec<f32>,
    /// Bias: [drive_dim]
    pub b: Vec<f32>,

    // ── T3 configuration ──
    pub n_osc: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub coupling_w12: f32,
    pub coupling_w13: f32,
    pub coupling_w23: f32,

    // ── Derived dimensions ──
    pub spike_dim: usize,
    pub drive_dim: usize,
}

impl SpikeGradBridge {
    /// Create new bridge with random weights.
    ///
    /// `spike_dim`: number of spike input features (per timestep)
    /// `n_osc`: number of T3 oscillators (drive_dim = 3 × n_osc)
    /// Weights are initialized from Uniform(-0.01, 0.01), bias to zero.
    pub fn new(
        spike_dim: usize,
        n_osc: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
    ) -> Self {
        let drive_dim = 3 * n_osc;
        let n_weights = spike_dim * drive_dim;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut w = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            w.push((rng.gen::<f32>() - 0.5) * 0.02); // ~Uniform(-0.01, 0.01)
        }
        let b = vec![0.0; drive_dim];

        Self {
            w, b,
            n_osc, dt, drive_gain, alpha, freq_min, freq_max,
            coupling_w12, coupling_w13, coupling_w23,
            spike_dim, drive_dim,
        }
    }

    /// Create bridge from existing weights (load from saved state).
    ///
    /// `weights`: `[spike_dim × drive_dim]` flattened row-major.
    /// `bias`: `[drive_dim]`.
    pub fn from_weights(
        spike_dim: usize,
        n_osc: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
        weights: &[f32],
        bias: &[f32],
    ) -> Self {
        let drive_dim = 3 * n_osc;
        assert_eq!(weights.len(), spike_dim * drive_dim,
            "from_weights: weights.len()={} != spike_dim*drive_dim={}*{}",
            weights.len(), spike_dim, drive_dim);
        assert_eq!(bias.len(), drive_dim,
            "from_weights: bias.len()={} != drive_dim={}", bias.len(), drive_dim);
        Self {
            w: weights.to_vec(),
            b: bias.to_vec(),
            n_osc, dt, drive_gain, alpha, freq_min, freq_max,
            coupling_w12, coupling_w13, coupling_w23,
            spike_dim, drive_dim,
        }
    }

    /// Create bridge with zero weights (for baseline/debug).
    pub fn new_zero(
        spike_dim: usize,
        n_osc: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
    ) -> Self {
        let drive_dim = 3 * n_osc;
        Self {
            w: vec![0.0; spike_dim * drive_dim],
            b: vec![0.0; drive_dim],
            n_osc, dt, drive_gain, alpha, freq_min, freq_max,
            coupling_w12, coupling_w13, coupling_w23,
            spike_dim, drive_dim,
        }
    }

    /// ── Forward pass ──
    ///
    /// `spikes`: [T × spike_dim] flattened, row-major.
    /// Returns `(pst_features[T × pst_dim], trajectory)`.
    ///
    /// Internally computes:
    /// ```text
    /// for t in 0..T:
    ///     drives[t, j] = Σ_i spikes[t, i] × W[i, j] + b[j]
    /// ```
    /// then runs T3 forward with these drives.
    pub fn forward(&self, spikes: &[f32], n_steps: usize) -> (Vec<f32>, T3SpikeTrajectory) {
        let sd = self.spike_dim;
        let dd = self.drive_dim;
        let len = n_steps * dd;

        let mut drives = vec![0.0; len];
        for t in 0..n_steps {
            let base_t = t * dd;
            let spike_t = &spikes[t * sd..(t + 1) * sd];
            for j in 0..dd {
                let mut acc = self.b[j];
                for i in 0..sd {
                    acc += spike_t[i] * self.w[i * dd + j];
                }
                drives[base_t + j] = acc;
            }
        }

        t3spike_forward(
            self.n_osc, n_steps, self.dt, self.drive_gain, self.alpha,
            self.freq_min, self.freq_max,
            self.coupling_w12, self.coupling_w13, self.coupling_w23,
            &drives,
        )
    }

    /// ── Backward pass ──
    ///
    /// Computes gradients of bridge weights w.r.t. loss through BPTT.
    ///
    /// `grad_pst`: [pst_dim] — dLoss/d(PST) at final step
    /// `traj`: trajectory from `forward()`
    /// `spikes`: [T × spike_dim] — same as forward input
    /// `per_step_grad_pst`: optional [T × pst_dim] — per-step PST gradients for SPIGRAD
    ///
    /// Returns `(grad_w, grad_b)` — gradients w.r.t. bridge weights.
    pub fn backward(
        &self,
        grad_pst: &[f32],
        traj: &T3SpikeTrajectory,
        spikes: &[f32],
        per_step_grad_pst: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let dd = self.drive_dim;
        let sd = self.spike_dim;
        let n_steps = traj.n_steps;

        // BPTT through T3 → grad_drives [n_steps × drive_dim]
        let grad_drives = t3spike_backward_per_step(grad_pst, traj, per_step_grad_pst);

        // Project grad_drives back to bridge weights:
        // dL/dW[i, j] = Σ_t grad_drives[t, j] × spikes[t, i]
        // dL/db[j]    = Σ_t grad_drives[t, j]
        let mut grad_w = vec![0.0; sd * dd];
        let mut grad_b = vec![0.0; dd];

        for t in 0..n_steps {
            let base_d = t * dd;
            let base_s = t * sd;
            for j in 0..dd {
                let g = grad_drives[base_d + j];
                grad_b[j] += g;
                for i in 0..sd {
                    grad_w[i * dd + j] += g * spikes[base_s + i];
                }
            }
        }

        (grad_w, grad_b)
    }

    /// Apply gradient update to bridge weights.
    ///
    /// `grad_w`: [spike_dim × drive_dim] — gradient w.r.t. W
    /// `grad_b`: [drive_dim] — gradient w.r.t. b
    /// `lr`: learning rate
    pub fn update(&mut self, grad_w: &[f32], grad_b: &[f32], lr: f32) {
        let nw = self.spike_dim * self.drive_dim;
        for i in 0..nw {
            self.w[i] -= lr * grad_w[i];
        }
        for i in 0..self.drive_dim {
            self.b[i] -= lr * grad_b[i];
        }
    }

    /// Apply gradient update with per-parameter ADAM.
    ///
    /// State vectors `m` and `v` must be same shape as (w, b).
    /// Mutates state in place.
    pub fn update_adam(
        &mut self,
        grad_w: &[f32],
        grad_b: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u32,
        m_w: &mut [f32],
        v_w: &mut [f32],
        m_b: &mut [f32],
        v_b: &mut [f32],
    ) {
        let nw = self.spike_dim * self.drive_dim;
        for i in 0..nw {
            m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * grad_w[i];
            v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * grad_w[i] * grad_w[i];
            let m_hat = m_w[i] / (1.0 - beta1.powi(t as i32));
            let v_hat = v_w[i] / (1.0 - beta2.powi(t as i32));
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        for i in 0..self.drive_dim {
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grad_b[i];
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * grad_b[i] * grad_b[i];
            let m_hat = m_b[i] / (1.0 - beta1.powi(t as i32));
            let v_hat = v_b[i] / (1.0 - beta2.powi(t as i32));
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// Bridge output: compute drives from spikes without running T3.
    pub fn compute_drives(&self, spikes: &[f32], n_steps: usize) -> Vec<f32> {
        let sd = self.spike_dim;
        let dd = self.drive_dim;
        let mut drives = vec![0.0; n_steps * dd];
        for t in 0..n_steps {
            let base = t * dd;
            let s = &spikes[t * sd..(t + 1) * sd];
            for j in 0..dd {
                let mut acc = self.b[j];
                for i in 0..sd {
                    acc += s[i] * self.w[i * dd + j];
                }
                drives[base + j] = acc;
            }
        }
        drives
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_new() {
        let bridge = SpikeGradBridge::new(1, 4, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        assert_eq!(bridge.spike_dim, 1);
        assert_eq!(bridge.n_osc, 4);
        assert_eq!(bridge.drive_dim, 12);
        assert_eq!(bridge.w.len(), 12);
        assert_eq!(bridge.b.len(), 12);
        // Weights should be non-zero (random init)
        assert!(bridge.w.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_bridge_forward_backward() {
        let mut bridge = SpikeGradBridge::new(2, 3, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let t = 10;

        let mut rng = rand::thread_rng();
        use rand::Rng;
        // Random spikes [T × 2]
        let mut spikes = Vec::with_capacity(t * 2);
        for _ in 0..t * 2 {
            spikes.push((rng.gen::<f32>() - 0.5) * 2.0);
        }

        // Forward
        let (pst, traj) = bridge.forward(&spikes, t);
        let pst_dim = pst.len();
        assert!(pst_dim > 0);
        assert_eq!(traj.n_steps, t);

        // Backward with dummy gradient
        let grad_pst = vec![1.0; pst_dim];
        let (gw, gb) = bridge.backward(&grad_pst, &traj, &spikes, None);
        assert_eq!(gw.len(), 2 * 9); // spike_dim=2, drive_dim=9
        assert_eq!(gb.len(), 9);

        // Gradient norms should be finite
        assert!(gw.iter().all(|&x| x.is_finite()));
        assert!(gb.iter().all(|&x| x.is_finite()));

        // Update should change weights
        let w_before = bridge.w.clone();
        bridge.update(&gw, &gb, 0.01);
        let mut changed = false;
        for (a, b) in w_before.iter().zip(bridge.w.iter()) {
            if (a - b).abs() > 1e-8 {
                changed = true;
                break;
            }
        }
        assert!(changed, "weights should change after update");
    }

    #[test]
    fn test_bridge_per_step_gradient() {
        let bridge = SpikeGradBridge::new(1, 2, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let t = 5;
        let pst_dim = crate::t3::pst_linear_dim(2);

        let spikes = vec![0.5; t]; // constant spike rate

        // Forward
        let (_, traj) = bridge.forward(&spikes, t);

        // Per-step PST gradient (nonzero only at the last feature for testing)
        let mut per_step = vec![0.0; t * pst_dim];
        for step in 0..t {
            per_step[step * pst_dim + 0] = 1.0; // gradient on sin(θ₁) at every step
        }

        let grad_pst = vec![0.0; pst_dim];
        let (gw, gb) = bridge.backward(&grad_pst, &traj, &spikes, Some(&per_step));

        assert!(gw.iter().all(|&x| x.is_finite()));
        assert!(gb.iter().all(|&x| x.is_finite()));
        assert!(gw.iter().any(|&x| x.abs() > 1e-6), "per-step gradient should be non-zero");
    }

    #[test]
    fn test_bridge_adam_update() {
        let mut bridge = SpikeGradBridge::new(1, 4, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let nw = bridge.spike_dim * bridge.drive_dim;
        let nd = bridge.drive_dim;

        let mut m_w = vec![0.0; nw];
        let mut v_w = vec![0.0; nw];
        let mut m_b = vec![0.0; nd];
        let mut v_b = vec![0.0; nd];

        let gw = vec![0.1; nw];
        let gb = vec![0.05; nd];

        let w_before = bridge.w.clone();
        bridge.update_adam(&gw, &gb, 0.01, 0.9, 0.999, 1e-8, 1, &mut m_w, &mut v_w, &mut m_b, &mut v_b);

        let changed = w_before.iter().zip(bridge.w.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "ADAM should update weights");
    }
}
