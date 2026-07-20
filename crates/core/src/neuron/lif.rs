//! LIF (Leaky Integrate-and-Fire) + LIF-AT + E-Prop LIF.
//!
//! Safe Rust wrappers over SIMD kernels (`kernels::lif_step`,
//! `kernels::lif_at_step`, `kernels::eprop_lif_step`).

use crate::kernels;
use rayon::prelude::*;

use super::eprop::surrogate_gradient;

// ─── LIF-AT ──────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Step `n` LIF-AT (LIF with Adaptive Threshold) neurons. Parallel, n must be multiple of 8.
///
/// LIF-AT добавляет adaptive threshold offset θ:
/// dv/dt = (v_rest - v + R*I) / tau_m
/// v >= V_th + θ → spike, v = v_reset, θ += beta
/// dθ/dt = -θ / tau_theta
pub fn step_lif_at(
    v: &mut [f32], theta: &mut [f32], I: &mut [f32], spikes: &mut [i32],
    dt: f32, tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32],
    v_reset: &[f32], tau_theta: &[f32], beta: &[f32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == theta.len() && n == I.len() && n == spikes.len()
        && n == tau_m.len() && n == v_rest.len() && n == v_thresh.len()
        && n == v_reset.len() && n == tau_theta.len() && n == beta.len() && n == refractory.len());

    // Build params on stack per chunk — avoids heap allocation on every step.
    v.par_chunks_exact_mut(8)
        .zip(theta.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((v8, th8), I8), sp8), refr8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 56];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = tau_m[i + j];
                params[16 + j] = v_rest[i + j];
                params[24 + j] = v_thresh[i + j];
                params[32 + j] = v_reset[i + j];
                params[40 + j] = tau_theta[i + j];
                params[48 + j] = beta[i + j];
            }
            kernels::lif_at_step(v8, th8, I8, sp8, &params, refr8, tau_ref, None);
        });
}

// ─── LIF ─────────────────────────────────────────────────

/// Step `n` LIF neurons. Parallel. n must be multiple of 8.
pub fn step_lif(
    v: &mut [f32], I: &mut [f32], spikes: &mut [i32],
    dt: f32, tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32], v_reset: &[f32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == I.len() && n == spikes.len() && n == tau_m.len()
        && n == v_rest.len() && n == v_thresh.len() && n == v_reset.len() && n == refractory.len());

    kernels::safety_clamp(I, -super::I_CLAMP, super::I_CLAMP);

    // Build params on stack per chunk — avoids heap allocation on every step.
    v.par_chunks_exact_mut(8)
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, (((v8, I8), sp8), refr8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 40];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = tau_m[i + j];
                params[16 + j] = v_rest[i + j];
                params[24 + j] = v_thresh[i + j];
                params[32 + j] = v_reset[i + j];
            }
            kernels::lif_step(v8, I8, sp8, &params, refr8, tau_ref, None);
        });
}

// ═══════════════════════════════════════════════════════════
//  E-Prop LIF — Eligibility Propagation (online SNN learning)
// ═══════════════════════════════════════════════════════════

/// E-Prop LIF neuron with eligibility traces + refractory period.
///
/// Based on: Bellec et al. \"A solution to the learning dilemma for
/// recurrent networks of spiking neurons\" (Nature Comms 2020)
///
/// Refractory period matches `lif_step` semantics: during refractory,
/// v/eligibility/psi are frozen, refractory counter decrements.
pub struct EPropLIF {
    pub v: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub refractory: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_m: f32,
    pub tau_e: f32,
    pub tau_ref: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub tau_surrogate: f32,
}

impl EPropLIF {
    pub fn new(n: usize, dt: f32, tau_m: f32, tau_e: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            v: vec![-65.0; n_pad],
            eligibility: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            refractory: vec![0.0; n_pad],
            n: n_pad, dt, tau_m, tau_e,
            tau_ref: 3.0,
            v_rest: -65.0, v_thresh: -50.0, v_reset: -70.0,
            tau_surrogate: 2.0,
        }
    }

    /// One step of LIF dynamics + eligibility trace update + refractory.
    /// Uses SIMD `f32x8` kernel.
    ///
    /// Returns: number of output spikes
    pub fn step(&mut self, I: &[f32], spikes: &mut [i32], pre_spikes: &[i32]) -> usize {
        let n = self.n.min(I.len()).min(spikes.len()).min(pre_spikes.len());
        let n8 = n & !7;
        if n8 == 0 { return 0; }

        let total: usize = self.v[..n8].par_chunks_exact_mut(8)
            .zip(self.eligibility[..n8].par_chunks_exact_mut(8))
            .zip(self.psi[..n8].par_chunks_exact_mut(8))
            .zip(I[..n8].par_chunks_exact(8))
            .zip(spikes[..n8].par_chunks_exact_mut(8))
            .zip(pre_spikes[..n8].par_chunks_exact(8))
            .zip(self.refractory[..n8].par_chunks_exact_mut(8))
            .map(|((((((v8, e8), ps8), I8), sp8), pre8), refr8)| {
                kernels::eprop_lif_step(
                    v8, e8, ps8,
                    I8, sp8, pre8,
                    refr8,
                    self.dt, self.tau_m, self.tau_e, self.tau_ref,
                    self.v_rest, self.v_thresh, self.v_reset, self.tau_surrogate,
                    None,
                )
            })
            .sum();

        // Scalar tail (n & 7)
        let mut tail_count = 0;
        if n8 < n {
            let dt = self.dt;
            let tau_m = self.tau_m;
            let tau_e = self.tau_e;
            let tau_ref = self.tau_ref;
            let v_rest = self.v_rest;
            let v_thresh = self.v_thresh;
            let v_reset = self.v_reset;
            let tau_sg = self.tau_surrogate;
            for i in n8..n {
                if self.refractory[i] > 0.0 {
                    self.refractory[i] = (self.refractory[i] - dt).max(0.0);
                    spikes[i] = 0;
                    continue;
                }
                let decay_m = (-dt / tau_m).exp();
                self.v[i] = v_rest + (self.v[i] - v_rest) * decay_m + I[i] * (1.0 - decay_m);
                self.psi[i] = surrogate_gradient(self.v[i], v_thresh, tau_sg);
                let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
                let e_decay = (-dt / tau_e).exp();
                self.eligibility[i] = self.eligibility[i] * e_decay + self.psi[i] * z_pre * tau_e * (1.0 - e_decay);
                if self.v[i] >= v_thresh {
                    spikes[i] = 1;
                    tail_count += 1;
                    self.v[i] = v_reset;
                    self.refractory[i] = tau_ref;
                } else {
                    spikes[i] = 0;
                }
            }
        }

        total + tail_count
    }

    /// Apply E-Prop weight update: Δw = η · L_j · e_ji
    pub fn update_weights(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        let n = self.n.min(weights.len()).min(learning_signals.len());
        let chunks = n / 8;
        if chunks > 0 {
            weights[..chunks * 8].par_chunks_exact_mut(8)
                .zip(self.eligibility[..chunks * 8].par_chunks_exact(8))
                .zip(learning_signals[..chunks * 8].par_chunks_exact(8))
                .for_each(|((w8, e8), l8)| {
                    for i in 0..8 {
                        w8[i] += lr * l8[i] * e8[i];
                    }
                });
        }
        for i in chunks * 8..n {
            weights[i] += lr * learning_signals[i] * self.eligibility[i];
        }
    }

    pub fn reset(&mut self) {
        self.v.fill(-65.0);
        self.eligibility.fill(0.0);
        self.psi.fill(0.0);
        self.refractory.fill(0.0);
    }
}
