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

use crate::spike::{t3spike_backward_per_step, t3spike_forward, T3SpikeTrajectory};

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
    #[inline]
    fn assert_config_values(
        n_osc: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
    ) {
        assert!(n_osc > 0, "SpikeGradBridge: n_osc must be > 0");
        assert!(
            dt.is_finite() && dt > 0.0,
            "SpikeGradBridge: dt must be finite and > 0"
        );
        assert!(
            drive_gain.is_finite(),
            "SpikeGradBridge: drive_gain must be finite"
        );
        assert!(alpha.is_finite(), "SpikeGradBridge: alpha must be finite");
        assert!(
            freq_min.is_finite() && freq_max.is_finite() && freq_min <= freq_max,
            "SpikeGradBridge: frequency bounds must be finite and ordered"
        );
        assert!(
            coupling_w12.is_finite() && coupling_w13.is_finite() && coupling_w23.is_finite(),
            "SpikeGradBridge: coupling weights must be finite"
        );
    }

    #[inline]
    fn assert_config(&self) {
        Self::assert_config_values(
            self.n_osc,
            self.dt,
            self.drive_gain,
            self.alpha,
            self.freq_min,
            self.freq_max,
            self.coupling_w12,
            self.coupling_w13,
            self.coupling_w23,
        );
    }

    #[inline]
    fn checked_len(n_steps: usize, width: usize, label: &str) -> usize {
        n_steps
            .checked_mul(width)
            .unwrap_or_else(|| panic!("{label}: n_steps * width overflowed"))
    }

    #[inline]
    fn expected_pst_len(&self, n_steps: usize) -> usize {
        Self::checked_len(n_steps, crate::t3::pst_linear_dim(self.n_osc), "PST shape")
    }

    #[inline]
    fn assert_layout(&self) {
        self.assert_config();
        assert!(self.spike_dim > 0, "SpikeGradBridge: spike_dim must be > 0");
        assert_eq!(
            self.drive_dim,
            self.n_osc
                .checked_mul(3)
                .expect("SpikeGradBridge: drive shape overflowed"),
            "SpikeGradBridge: drive_dim does not match n_osc"
        );
        assert_eq!(
            self.w.len(),
            self.spike_dim
                .checked_mul(self.drive_dim)
                .expect("SpikeGradBridge: weight shape overflowed"),
            "SpikeGradBridge: weight storage has the wrong length"
        );
        assert_eq!(
            self.b.len(),
            self.drive_dim,
            "SpikeGradBridge: bias storage has the wrong length"
        );
    }

    #[inline]
    fn assert_finite(values: &[f32], label: &str) {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "SpikeGradBridge: {label} contains a non-finite value"
        );
    }

    #[inline]
    fn assert_trajectory_shape_config(&self, traj: &T3SpikeTrajectory) {
        let n = traj.n_osc;
        let steps = traj.n_steps;
        let state_len = Self::checked_len(steps, n, "SpikeGradBridge: trajectory state");
        let flat_len = state_len
            .checked_mul(9)
            .expect("SpikeGradBridge: trajectory flat shape overflowed");
        assert_eq!(
            traj.flat.len(),
            flat_len,
            "SpikeGradBridge::backward: trajectory flat storage has the wrong length"
        );
        for (label, state) in [
            ("th1_final", &traj.th1_final),
            ("th2_final", &traj.th2_final),
            ("th3_final", &traj.th3_final),
            ("r1_final", &traj.r1_final),
            ("r2_final", &traj.r2_final),
            ("r3_final", &traj.r3_final),
        ] {
            assert_eq!(
                state.len(),
                n,
                "SpikeGradBridge::backward: trajectory {label} has the wrong length"
            );
        }
        for (label, actual, expected) in [
            ("dt", traj.dt, self.dt),
            ("drive_gain", traj.drive_gain, self.drive_gain),
            ("alpha", traj.alpha, self.alpha),
            ("freq_min", traj.freq_min, self.freq_min),
            ("freq_max", traj.freq_max, self.freq_max),
            ("w12", traj.w12, self.coupling_w12),
            ("w13", traj.w13, self.coupling_w13),
            ("w23", traj.w23, self.coupling_w23),
        ] {
            assert_eq!(
                actual, expected,
                "SpikeGradBridge::backward: trajectory {label} does not match bridge configuration"
            );
        }
    }

    #[inline]
    fn assert_trajectory_layout(&self, traj: &T3SpikeTrajectory) {
        self.assert_trajectory_shape_config(traj);
        for (label, state) in [
            ("th1_final", &traj.th1_final),
            ("th2_final", &traj.th2_final),
            ("th3_final", &traj.th3_final),
            ("r1_final", &traj.r1_final),
            ("r2_final", &traj.r2_final),
            ("r3_final", &traj.r3_final),
        ] {
            Self::assert_finite(state, label);
        }
        Self::assert_finite(&traj.flat, "trajectory storage");
    }

    #[inline]
    fn assert_backward_shapes(
        &self,
        grad_pst: &[f32],
        traj: &T3SpikeTrajectory,
        spikes: &[f32],
        per_step_grad_pst: Option<&[f32]>,
    ) {
        assert_eq!(
            traj.n_osc, self.n_osc,
            "SpikeGradBridge::backward: trajectory has a different oscillator count"
        );
        let n_steps = traj.n_steps;
        assert_eq!(
            grad_pst.len(),
            crate::t3::pst_linear_dim(self.n_osc),
            "SpikeGradBridge::backward: final PST gradient has the wrong length"
        );
        assert_eq!(
            spikes.len(),
            Self::checked_len(n_steps, self.spike_dim, "SpikeGradBridge::backward input"),
            "SpikeGradBridge::backward: spikes must match trajectory length"
        );
        if let Some(per_step) = per_step_grad_pst {
            assert_eq!(
                per_step.len(),
                self.expected_pst_len(n_steps),
                "SpikeGradBridge::backward: per-step PST gradient has the wrong length"
            );
        }
    }

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
        Self::assert_config_values(
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
        );
        assert!(spike_dim > 0, "SpikeGradBridge: spike_dim must be > 0");
        let drive_dim = 3usize
            .checked_mul(n_osc)
            .expect("SpikeGradBridge::new: drive shape overflowed");
        let n_weights = spike_dim
            .checked_mul(drive_dim)
            .expect("SpikeGradBridge::new: weight shape overflowed");
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut w = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            w.push((rng.gen::<f32>() - 0.5) * 0.02); // ~Uniform(-0.01, 0.01)
        }
        let b = vec![0.0; drive_dim];

        Self {
            w,
            b,
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            spike_dim,
            drive_dim,
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
        Self::assert_config_values(
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
        );
        assert!(spike_dim > 0, "SpikeGradBridge: spike_dim must be > 0");
        Self::assert_finite(weights, "weights");
        Self::assert_finite(bias, "bias");
        let drive_dim = 3usize
            .checked_mul(n_osc)
            .expect("SpikeGradBridge::from_weights: drive shape overflowed");
        let expected_weights = spike_dim
            .checked_mul(drive_dim)
            .expect("SpikeGradBridge::from_weights: weight shape overflowed");
        assert_eq!(
            weights.len(),
            expected_weights,
            "from_weights: weights.len()={} != spike_dim*drive_dim={}*{}",
            weights.len(),
            spike_dim,
            drive_dim
        );
        assert_eq!(
            bias.len(),
            drive_dim,
            "from_weights: bias.len()={} != drive_dim={}",
            bias.len(),
            drive_dim
        );
        Self {
            w: weights.to_vec(),
            b: bias.to_vec(),
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            spike_dim,
            drive_dim,
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
        Self::assert_config_values(
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
        );
        assert!(spike_dim > 0, "SpikeGradBridge: spike_dim must be > 0");
        let drive_dim = 3usize
            .checked_mul(n_osc)
            .expect("SpikeGradBridge::new_zero: drive shape overflowed");
        let n_weights = spike_dim
            .checked_mul(drive_dim)
            .expect("SpikeGradBridge::new_zero: weight shape overflowed");
        Self {
            w: vec![0.0; n_weights],
            b: vec![0.0; drive_dim],
            n_osc,
            dt,
            drive_gain,
            alpha,
            freq_min,
            freq_max,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            spike_dim,
            drive_dim,
        }
    }

    /// ── Forward pass ──
    ///
    /// `spikes`: [T × spike_dim] flattened, row-major.
    /// Returns `(final_step_pst_features[pst_dim], trajectory)`.
    ///
    /// Internally computes:
    /// ```text
    /// for t in 0..T:
    ///     drives[t, j] = Σ_i spikes[t, i] × W[i, j] + b[j]
    /// ```
    /// then runs T3 forward with these drives.
    pub fn forward(&self, spikes: &[f32], n_steps: usize) -> (Vec<f32>, T3SpikeTrajectory) {
        self.assert_layout();
        Self::assert_finite(&self.w, "weights");
        Self::assert_finite(&self.b, "bias");
        Self::assert_finite(spikes, "spikes");
        let sd = self.spike_dim;
        let dd = self.drive_dim;
        assert_eq!(
            spikes.len(),
            Self::checked_len(n_steps, sd, "SpikeGradBridge::forward input"),
            "SpikeGradBridge::forward: spikes must have n_steps * spike_dim elements"
        );
        let len = Self::checked_len(n_steps, dd, "SpikeGradBridge::forward drives");

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
            self.n_osc,
            n_steps,
            self.dt,
            self.drive_gain,
            self.alpha,
            self.freq_min,
            self.freq_max,
            self.coupling_w12,
            self.coupling_w13,
            self.coupling_w23,
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
        self.assert_layout();
        Self::assert_finite(&self.w, "weights");
        Self::assert_finite(&self.b, "bias");
        self.assert_trajectory_layout(traj);
        self.assert_backward_shapes(grad_pst, traj, spikes, per_step_grad_pst);
        Self::assert_finite(grad_pst, "final PST gradient");
        Self::assert_finite(spikes, "spikes");
        if let Some(per_step) = per_step_grad_pst {
            Self::assert_finite(per_step, "per-step PST gradient");
        }
        self.backward_impl(grad_pst, traj, spikes, per_step_grad_pst)
    }

    /// Fast BPTT path for trusted, already-validated trajectories.
    ///
    /// This keeps shape/configuration checks but skips the O(T×N) finite-value
    /// scan over trajectory buffers. Callers must guarantee that the bridge,
    /// trajectory, gradients, and spikes contain only finite values. The
    /// regular [`Self::backward`] method remains the safe default.
    pub fn backward_fast(
        &self,
        grad_pst: &[f32],
        traj: &T3SpikeTrajectory,
        spikes: &[f32],
        per_step_grad_pst: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        self.assert_layout();
        self.assert_trajectory_shape_config(traj);
        self.assert_backward_shapes(grad_pst, traj, spikes, per_step_grad_pst);
        self.backward_impl(grad_pst, traj, spikes, per_step_grad_pst)
    }

    #[inline]
    fn backward_impl(
        &self,
        grad_pst: &[f32],
        traj: &T3SpikeTrajectory,
        spikes: &[f32],
        per_step_grad_pst: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let dd = self.drive_dim;
        let sd = self.spike_dim;
        let n_steps = traj.n_steps;
        let grad_drives = t3spike_backward_per_step(grad_pst, traj, per_step_grad_pst);
        let mut grad_w = vec![
            0.0;
            sd.checked_mul(dd)
                .expect("SpikeGradBridge::backward: gradient shape overflowed")
        ];
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
        self.assert_layout();
        Self::assert_finite(&self.w, "weights");
        Self::assert_finite(&self.b, "bias");
        Self::assert_finite(grad_w, "weight gradient");
        Self::assert_finite(grad_b, "bias gradient");
        let nw = self
            .spike_dim
            .checked_mul(self.drive_dim)
            .expect("SpikeGradBridge::update: gradient shape overflowed");
        assert_eq!(
            grad_w.len(),
            nw,
            "SpikeGradBridge::update: grad_w has the wrong length"
        );
        assert_eq!(
            grad_b.len(),
            self.drive_dim,
            "SpikeGradBridge::update: grad_b has the wrong length"
        );
        assert!(
            lr.is_finite() && lr >= 0.0,
            "SpikeGradBridge::update: lr must be finite and >= 0"
        );
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
        self.assert_layout();
        Self::assert_finite(&self.w, "weights");
        Self::assert_finite(&self.b, "bias");
        Self::assert_finite(grad_w, "weight gradient");
        Self::assert_finite(grad_b, "bias gradient");
        Self::assert_finite(m_w, "Adam m_w state");
        Self::assert_finite(v_w, "Adam v_w state");
        Self::assert_finite(m_b, "Adam m_b state");
        Self::assert_finite(v_b, "Adam v_b state");
        let nw = self
            .spike_dim
            .checked_mul(self.drive_dim)
            .expect("SpikeGradBridge::update_adam: gradient shape overflowed");
        assert_eq!(
            grad_w.len(),
            nw,
            "SpikeGradBridge::update_adam: grad_w has the wrong length"
        );
        assert_eq!(
            grad_b.len(),
            self.drive_dim,
            "SpikeGradBridge::update_adam: grad_b has the wrong length"
        );
        assert_eq!(
            m_w.len(),
            nw,
            "SpikeGradBridge::update_adam: m_w has the wrong length"
        );
        assert_eq!(
            v_w.len(),
            nw,
            "SpikeGradBridge::update_adam: v_w has the wrong length"
        );
        assert_eq!(
            m_b.len(),
            self.drive_dim,
            "SpikeGradBridge::update_adam: m_b has the wrong length"
        );
        assert_eq!(
            v_b.len(),
            self.drive_dim,
            "SpikeGradBridge::update_adam: v_b has the wrong length"
        );
        assert!(
            lr.is_finite()
                && lr >= 0.0
                && beta1.is_finite()
                && (0.0..1.0).contains(&beta1)
                && beta2.is_finite()
                && (0.0..1.0).contains(&beta2)
                && eps.is_finite()
                && eps > 0.0,
            "SpikeGradBridge::update_adam: invalid optimizer parameters"
        );
        assert!(t > 0, "SpikeGradBridge::update_adam: t must be > 0");
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
        self.assert_layout();
        Self::assert_finite(&self.w, "weights");
        Self::assert_finite(&self.b, "bias");
        Self::assert_finite(spikes, "spikes");
        let sd = self.spike_dim;
        let dd = self.drive_dim;
        assert_eq!(
            spikes.len(),
            Self::checked_len(n_steps, sd, "SpikeGradBridge::compute_drives input"),
            "SpikeGradBridge::compute_drives: spikes must match n_steps * spike_dim"
        );
        let mut drives =
            vec![0.0; Self::checked_len(n_steps, dd, "SpikeGradBridge::compute_drives output")];
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
            per_step[step * pst_dim] = 1.0; // gradient on sin(θ₁) at every step
        }

        let grad_pst = vec![0.0; pst_dim];
        let (gw, gb) = bridge.backward(&grad_pst, &traj, &spikes, Some(&per_step));

        assert!(gw.iter().all(|&x| x.is_finite()));
        assert!(gb.iter().all(|&x| x.is_finite()));
        assert!(
            gw.iter().any(|&x| x.abs() > 1e-6),
            "per-step gradient should be non-zero"
        );
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
        bridge.update_adam(
            &gw, &gb, 0.01, 0.9, 0.999, 1e-8, 1, &mut m_w, &mut v_w, &mut m_b, &mut v_b,
        );

        let changed = w_before
            .iter()
            .zip(bridge.w.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "ADAM should update weights");
    }

    #[test]
    fn test_bridge_backward_fast_matches_checked() {
        let bridge = SpikeGradBridge::new_zero(1, 2, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let spikes = vec![0.2; 5];
        let (pst, traj) = bridge.forward(&spikes, 5);
        let grad_pst = vec![1.0; pst.len()];
        let checked = bridge.backward(&grad_pst, &traj, &spikes, None);
        let fast = bridge.backward_fast(&grad_pst, &traj, &spikes, None);
        assert_eq!(checked, fast);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge: spike_dim must be > 0")]
    fn test_bridge_rejects_zero_spike_dim() {
        let _ = SpikeGradBridge::new_zero(0, 1, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge: dt must be finite and > 0")]
    fn test_bridge_rejects_invalid_dt() {
        let _ = SpikeGradBridge::new_zero(1, 1, 0.0, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge: frequency bounds must be finite and ordered")]
    fn test_bridge_rejects_invalid_frequency_bounds() {
        let _ = SpikeGradBridge::new_zero(1, 1, 0.05, 10.0, 1.0, 3.0, 0.5, 0.3, 0.15, 0.08);
    }

    #[test]
    #[should_panic(
        expected = "SpikeGradBridge::backward: trajectory dt does not match bridge configuration"
    )]
    fn test_bridge_rejects_mismatched_trajectory_config() {
        let bridge = SpikeGradBridge::new_zero(1, 1, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let spikes = [0.0; 1];
        let (pst, mut traj) = bridge.forward(&spikes, 1);
        traj.dt = 0.1;
        let _ = bridge.backward(&pst, &traj, &spikes, None);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge: weights contains a non-finite value")]
    fn test_bridge_rejects_non_finite_weights() {
        let _ = SpikeGradBridge::from_weights(
            1,
            1,
            0.05,
            10.0,
            1.0,
            0.5,
            3.0,
            0.3,
            0.15,
            0.08,
            &[f32::NAN; 3],
            &[0.0; 3],
        );
    }

    #[test]
    #[should_panic(expected = "spikes must have n_steps * spike_dim elements")]
    fn test_bridge_rejects_wrong_forward_shape() {
        let bridge = SpikeGradBridge::new_zero(2, 1, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let _ = bridge.forward(&[0.0; 3], 2);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge::backward: trajectory flat storage")]
    fn test_bridge_rejects_malformed_trajectory() {
        let bridge = SpikeGradBridge::new_zero(1, 1, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let spikes = [0.0; 1];
        let (pst, mut traj) = bridge.forward(&spikes, 1);
        traj.flat.pop();
        let _ = bridge.backward(&pst, &traj, &spikes, None);
    }

    #[test]
    #[should_panic(expected = "SpikeGradBridge::update_adam: m_w")]
    fn test_bridge_rejects_wrong_adam_state_shape() {
        let mut bridge =
            SpikeGradBridge::new_zero(1, 1, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08);
        let mut empty_m_w = Vec::new();
        let mut empty_v_w = Vec::new();
        let mut empty_m_b = Vec::new();
        let mut empty_v_b = Vec::new();
        bridge.update_adam(
            &[0.0; 3],
            &[0.0; 3],
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
            &mut empty_m_w,
            &mut empty_v_w,
            &mut empty_m_b,
            &mut empty_v_b,
        );
    }
}
