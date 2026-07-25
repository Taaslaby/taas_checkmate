//! AdEx (Adaptive Exponential) neuron + E-Prop AdEx Multi.
//!
//! Safe Rust wrappers over SIMD kernels (`kernels::adex_step`,
//! `kernels::eprop_adex_step_multi`).

use crate::kernels;
use rayon::prelude::*;

// ─── AdEx ────────────────────────────────────────────────

/// Step `n` AdEx neurons. Parallel. n must be multiple of 8.
///
/// Adaptive Exponential (AdEx) model dynamics:
///   dv/dt = (v_rest - v + Δ_T·exp((v - v_T)/Δ_T) - w + I) / τ_m
///   dw/dt = (a·(v - v_rest) - w) / τ_w
///   v ≥ v_thresh → spike, v = v_reset, w += b
pub fn step_adex(
    v: &mut [f32],
    u: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    tau_w: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    delta_t: &[f32],
    a: &[f32],
    b: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == u.len()
            && n == I.len()
            && n == spikes.len()
            && n == tau_m.len()
            && n == tau_w.len()
            && n == v_rest.len()
            && n == v_thresh.len()
            && n == v_reset.len()
            && n == delta_t.len()
            && n == a.len()
            && n == b.len()
            && n == refractory.len()
    );

    // Build params on stack per chunk — avoids heap allocation on every step.
    // Each chunk builds [72]f32: [dt×8, tau_m×8, tau_w×8, v_rest×8, v_thresh×8,
    // v_reset×8, delta_t×8, a×8, b×8].
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((v8, u8), I8), sp8), refr8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 72];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = tau_m[i + j];
                params[16 + j] = tau_w[i + j];
                params[24 + j] = v_rest[i + j];
                params[32 + j] = v_thresh[i + j];
                params[40 + j] = v_reset[i + j];
                params[48 + j] = delta_t[i + j];
                params[56 + j] = a[i + j];
                params[64 + j] = b[i + j];
            }
            kernels::adex_step(v8, u8, I8, sp8, &params, refr8, tau_ref, None);
        });
}

// ═══════════════════════════════════════════════════════════
//  E-Prop AdEx Multi — per-synapse eligibility for Adaptive Exponential
// ═══════════════════════════════════════════════════════════

/// E-Prop AdEx with per-synapse eligibility traces.
///
/// Column-major layout: `e[i * n_post + j]` = eligibility for synapse (i→j).
pub struct EPropAdExMulti {
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub psi: Vec<f32>,
    pub e: Vec<f32>, // [n_pre × n_post] column-major
    pub n_pre: usize,
    pub n_post: usize,
    pub dt: f32,
    pub tau_m: f32,
    pub tau_w: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub delta_t: f32,
    pub a: f32,
    pub b: f32,
    pub tau_e: f32,
    pub tau_surrogate: f32,
}

impl EPropAdExMulti {
    pub fn new(
        n_pre: usize,
        n_post: usize,
        dt: f32,
        tau_m: f32,
        tau_w: f32,
        tau_e: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        delta_t: f32,
        a: f32,
        b: f32,
    ) -> Self {
        let n_pad = n_post.div_ceil(8) * 8;
        Self {
            v: vec![v_rest; n_pad],
            w: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            e: vec![0.0; n_pre * n_pad],
            n_pre,
            n_post: n_pad,
            dt,
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_e,
            tau_surrogate: 2.0,
        }
    }

    /// Default AdEx params (regular spiking, adapted from Brette & Gerstner 2005).
    pub fn new_default(n_pre: usize, n_post: usize, dt: f32, tau_e: f32) -> Self {
        Self::new(
            n_pre, n_post, dt, 5.0,   // tau_m (ms)
            100.0, // tau_w (ms)
            tau_e, -70.0, // v_rest
            -50.0, // v_thresh
            -70.0, // v_reset
            2.0,   // delta_t
            0.02,  // a (subthreshold adaptation)
            0.05,  // b (spike-triggered adaptation)
        )
    }

    /// One step of AdEx dynamics + per-synapse eligibility update.
    pub fn step(
        &mut self,
        weights: &[f32],
        z_pre: &[i32],
        I_ext: &[f32],
        spikes: &mut [i32],
    ) -> usize {
        let n = self.n_post.min(I_ext.len()).min(spikes.len());
        if n == 0 {
            return 0;
        }

        let n_pre = self.n_pre;
        let dt = self.dt;
        let tau_m = self.tau_m;
        let tau_w = self.tau_w;
        let v_rest = self.v_rest;
        let v_thresh = self.v_thresh;
        let v_reset = self.v_reset;
        let delta_t = self.delta_t;
        let a = self.a;
        let b = self.b;
        let tau_e = self.tau_e;
        let tau_sg = self.tau_surrogate;

        kernels::eprop_adex_step_multi(
            &mut self.v,
            &mut self.w,
            &mut self.psi,
            &mut self.e,
            z_pre,
            I_ext,
            weights,
            n_pre,
            dt,
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_e,
            tau_sg,
            spikes,
            None,
        )
    }

    /// Apply E-Prop weight update for column-major weight matrix.
    pub fn update_weights_multi(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        let n_post = self.n_post;
        let n8 = n_post & !7;
        if n8 == 0 {
            return;
        }

        self.e
            .par_chunks_exact(n_post)
            .zip(weights.par_chunks_exact_mut(n_post))
            .for_each(|(e_row, w_row)| {
                w_row[..n8]
                    .chunks_exact_mut(8)
                    .zip(e_row[..n8].chunks_exact(8))
                    .zip(learning_signals[..n8].chunks_exact(8))
                    .for_each(|((w8, e8), l8)| {
                        for i in 0..8 {
                            w8[i] += lr * l8[i] * e8[i];
                        }
                    });
                for i in n8..n_post {
                    w_row[i] += lr * learning_signals[i] * e_row[i];
                }
            });
    }

    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
        self.w.fill(0.0);
        self.psi.fill(0.0);
        self.e.fill(0.0);
    }
}
