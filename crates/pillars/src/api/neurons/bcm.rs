//! BCM Rule (Bienenstock-Cooper-Munro 1982) + Homeostatic Synaptic Scaling.
//!
//! ## BCM Rule
//! Rate-based Hebbian plasticity with a sliding modification threshold:
//!
//! ```text
//! dw/dt = η · y · (y − θ_M) · x  −  decay · w
//! dθ_M/dt = (y² − θ_M) / τ_θ
//! ```
//!
//! - `y > θ_M` → LTP (weight ↑)
//! - `y < θ_M` → LTD (weight ↓)
//! - θ_M tracks `E[y²]` — high activity raises the threshold (metaplasticity)
//!
//! ## Homeostatic Synaptic Scaling
//! Multiplicative normalization of weights toward a target firing rate:
//!
//! ```text
//! Δw = η_h · (r_target − r_mean) · w · dt
//! ```
//!
//! ## Usage
//! ```ignore
//! use pillars::api::neurons::bcm::{BcmSynapse, HomeostaticScale};
//!
//! let n = 64;
//! let mut bcm = BcmSynapse::new(n);
//! bcm.step(&pre_rates, &post_rates, 1.0);
//!
//! // Apply homeostatic scaling separately
//! let mut hscale = HomeostaticScale::new(0.01, 5.0);
//! hscale.step(&mut bcm.weights, &bcm.y, 1.0);
//! ```
//!
//! Перенесено из `core::neuron::bcm` (Фаза 8) — ядра
//! `kernels::learning::{bcm_step, homeostatic_scale}` (волна 8, транш 1b).

use crate::kernels::learning::{bcm_step, homeostatic_scale};

/// BCM synapse with traces, sliding threshold, and weights.
#[derive(Debug, Clone)]
pub struct BcmSynapse {
    pub n: usize,
    /// Presynaptic trace (running average of pre activity)
    pub x: Vec<f32>,
    /// Postsynaptic trace (running average of post activity)
    pub y: Vec<f32>,
    /// Modification threshold θ_M
    pub theta_m: Vec<f32>,
    /// Synaptic weights
    pub weights: Vec<f32>,
    /// Time constants (ms)
    pub tau_x: f32,
    pub tau_y: f32,
    pub tau_theta: f32,
    /// BCM learning rate
    pub eta: f32,
    /// Weight decay
    pub decay: f32,
    /// Weight bounds
    pub w_min: f32,
    pub w_max: f32,
}

impl BcmSynapse {
    /// Create new BCM synapse population of `n` synapses (padded to %8).
    pub fn new(n: usize) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            n: n_pad,
            x: vec![0.0; n_pad],
            y: vec![0.0; n_pad],
            theta_m: vec![0.0; n_pad],
            weights: vec![0.5; n_pad],
            tau_x: 20.0,
            tau_y: 20.0,
            tau_theta: 100.0,
            eta: 0.01,
            decay: 0.001,
            w_min: 0.0,
            w_max: 10.0,
        }
    }

    /// One step of BCM plasticity.
    ///
    /// `pre_rates` / `post_rates`: firing rates or spike indicators `[n]`.
    ///
    /// Канон — `kernels::learning::bcm_step` (SIMD + scalar tail внутри канона).
    pub fn step(&mut self, pre_rates: &[f32], post_rates: &[f32], dt: f32) {
        let n = self.n.min(pre_rates.len()).min(post_rates.len());
        if n == 0 {
            return;
        }
        bcm_step(
            &mut self.x[..n],
            &mut self.y[..n],
            &mut self.theta_m[..n],
            &mut self.weights[..n],
            &pre_rates[..n],
            &post_rates[..n],
            dt,
            self.tau_x,
            self.tau_y,
            self.tau_theta,
            self.eta,
            self.decay,
            self.w_min,
            self.w_max,
        );
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.x.fill(0.0);
        self.y.fill(0.0);
        self.theta_m.fill(0.0);
        self.weights.fill(0.5);
    }

    /// Configure BCM parameters.
    pub fn configure(&mut self, eta: f32, tau_theta: f32, tau_x: f32, tau_y: f32) {
        self.eta = eta;
        self.tau_theta = tau_theta;
        self.tau_x = tau_x;
        self.tau_y = tau_y;
    }

    /// Set weight bounds.
    pub fn set_bounds(&mut self, w_min: f32, w_max: f32) {
        self.w_min = w_min;
        self.w_max = w_max;
    }
}

/// Homeostatic synaptic scaling controller.
#[derive(Debug, Clone)]
pub struct HomeostaticScale {
    /// Learning rate for homeostatic scaling
    pub eta_h: f32,
    /// Target firing rate (Hz)
    pub target_rate: f32,
}

impl HomeostaticScale {
    pub fn new(eta_h: f32, target_rate: f32) -> Self {
        Self { eta_h, target_rate }
    }

    /// Apply homeostatic scaling to weights.
    ///
    /// `weights` and `post_rates` must have the same length.
    pub fn step(&self, weights: &mut [f32], post_rates: &[f32], dt: f32) {
        let n = weights.len().min(post_rates.len());
        let n8 = n & !7;
        if n8 > 0 {
            homeostatic_scale(
                &mut weights[..n8],
                &post_rates[..n8],
                dt,
                self.eta_h,
                self.target_rate,
            );
        }
        if n8 < n {
            for i in n8..n {
                let scale = self.eta_h * (self.target_rate - post_rates[i]) * dt;
                weights[i] = weights[i] + scale * weights[i];
                weights[i] = weights[i].max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcm_state_basic() {
        let n = 16;
        let mut bcm = BcmSynapse::new(n);
        let pre = vec![0.5f32; n];
        let post = vec![8.0f32; n];

        // Need multiple steps for traces to ramp up (τ=20ms, so ~40 steps for y to reach ~8)
        for _ in 0..200 {
            bcm.step(&pre, &post, 1.0);
        }

        // With y≈8, θ_M ~ y²/2 ≈ 32: y < θ_M initially → LTD initially
        // But as θ_M catches up to y², the system balances
        // At steady state: θ_M → y², and BCM drive → 0 (balanced)
        // Weights should have moved from 0.5 — check they changed
        let mean_w = bcm.weights.iter().sum::<f32>() / n as f32;
        assert!(
            (mean_w - 0.5).abs() > 0.001,
            "BCM: weights should change from 0.5, got {mean_w:.6}"
        );
        // θ_M should be significant
        assert!(
            bcm.theta_m[0] > 1.0,
            "BCM: θ_M should accumulate, got {:.4}",
            bcm.theta_m[0]
        );
    }

    #[test]
    fn bcm_state_configure() {
        let mut bcm = BcmSynapse::new(16);
        bcm.configure(0.001, 200.0, 10.0, 10.0);
        assert_eq!(bcm.eta, 0.001);
        assert_eq!(bcm.tau_theta, 200.0);
    }

    #[test]
    fn homeostatic_state_basic() {
        let n = 16;
        let hs = HomeostaticScale::new(0.01, 5.0);
        let mut weights = vec![1.0f32; n];
        let rates = vec![10.0f32; n]; // above target → decrease

        hs.step(&mut weights, &rates, 1.0);
        assert!(
            weights[0] < 1.0,
            "Homeostatic: high rate should decrease weights"
        );
    }
}
