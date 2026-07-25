//! SAW (Spike-Adaptive-Width) neuron + ψ-wave + E-Prop SAW.
//!
//! Safe Rust wrappers over SIMD kernels (`kernels::saw_step`,
//! `kernels::psi_wave_step`, `kernels::eprop_saw_step`).

#![allow(non_snake_case)]

use crate::kernels;
use rayon::prelude::*;

use super::eprop::surrogate_gradient;


// ─── SAW ─────────────────────────────────────────────────

/// Step `n` SAW neurons. Parallel. n must be multiple of 8.
pub fn step_saw(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], I: &[f32],
    dt: f32, trace_decay: f32, noise_amp: f32,
    spikes: &mut [i32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == u.len() && n == trace.len() && n == a.len() && n == b.len()
        && n == I.len() && n == spikes.len() && n == refractory.len());

    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(trace.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .zip(a.par_chunks_exact(8)
            .zip(b.par_chunks_exact(8))
            .zip(I.par_chunks_exact(8)))
        .for_each(|(((((v8, u8), tr8), sp8), refr8), ((a8, b8), I8))| {
            kernels::saw_step(
                v8, u8, tr8,
                a8, b8, I8,
                dt, trace_decay, noise_amp,
                sp8,
                refr8, tau_ref,
                None,
            );
        });
}

// ─── PSI-Wave ────────────────────────────────────────────

/// Step `n` ψ-wave neurons. Parallel. n must be multiple of 8.
pub fn step_psi_wave(
    v: &mut [f32], u: &mut [f32],
    h_real: &mut [f32], h_imag: &mut [f32],
    I_ext: &mut [f32], speed: &mut [f32],
    dt: f32, hbar: f32,
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == u.len() && n == h_real.len() && n == h_imag.len()
        && n == I_ext.len() && n == speed.len());

    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(h_real.par_chunks_exact_mut(8))
        .zip(h_imag.par_chunks_exact_mut(8))
        .zip(I_ext.par_chunks_exact_mut(8))
        .zip(speed.par_chunks_exact_mut(8))
        .for_each(|(((((v8, u8), hr8), hi8), ie8), sp8)| {
            kernels::psi_wave_step(
                v8, u8,
                hr8, hi8,
                ie8, sp8,
                dt, hbar,
            );
        });
}

// ═══════════════════════════════════════════════════════════
//  E-Prop SAW — Eligibility Propagation for SAW neurons
// ═══════════════════════════════════════════════════════════

/// E-Prop SAW (Spike-Adaptive-Width) neuron with eligibility traces.
pub struct EPropSaw {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_e: f32,
    pub a: f32,
    pub b: f32,
    pub tau_surrogate: f32,
}

impl EPropSaw {
    pub fn new(n: usize, dt: f32, tau_e: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            v: vec![-65.0; n_pad],
            u: vec![-13.0; n_pad],
            eligibility: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            n: n_pad, dt, tau_e,
            a: 0.02, b: 0.2,
            tau_surrogate: 2.0,
        }
    }

    /// One step of SAW dynamics + eligibility trace update.
    /// Uses SIMD `f32x8` kernel.
    pub fn step(&mut self, I: &[f32], spikes: &mut [i32], pre_spikes: &[i32]) -> usize {
        let n = self.n.min(I.len()).min(spikes.len()).min(pre_spikes.len());
        let n8 = n & !7;
        if n8 == 0 { return 0; }

        let total: usize = self.v[..n8].par_chunks_exact_mut(8)
            .zip(self.u[..n8].par_chunks_exact_mut(8))
            .zip(self.eligibility[..n8].par_chunks_exact_mut(8))
            .zip(self.psi[..n8].par_chunks_exact_mut(8))
            .zip(I[..n8].par_chunks_exact(8))
            .zip(spikes[..n8].par_chunks_exact_mut(8))
            .zip(pre_spikes[..n8].par_chunks_exact(8))
            .map(|((((((v8, u8), e8), ps8), I8), sp8), pre8)| {
                kernels::eprop_saw_step(
                    v8, u8, e8, ps8,
                    I8, sp8, pre8,
                    self.dt, self.tau_e, self.a, self.b, self.tau_surrogate,
                    None,
                )
            })
            .sum();

        // Scalar tail (n & 7) — exact exponential decay, matching SIMD kernel
        let mut tail_count = 0;
        if n8 < n {
            let dt = self.dt;
            let tau_e = self.tau_e.max(1e-6);
            let a = self.a;
            let b = self.b;
            let tau_sg = self.tau_surrogate;
            let e_decay = (-dt / tau_e).exp();
            for i in n8..n {
                let vv = self.v[i];
                let dv = (0.04 * vv + 5.0) * vv + 140.0 - self.u[i] + I[i];
                let du = a * (b * vv - self.u[i]);
                self.v[i] += dt * dv;
                self.u[i] += dt * du;
                self.psi[i] = surrogate_gradient(self.v[i], 30.0, tau_sg);
                let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
                self.eligibility[i] = self.eligibility[i] * e_decay + self.psi[i] * z_pre * tau_e * (1.0 - e_decay);
                if self.v[i] >= 30.0 {
                    spikes[i] = 1;
                    tail_count += 1;
                    self.v[i] = -65.0;
                    self.u[i] += 8.0;
                } else {
                    spikes[i] = 0;
                }
            }
        }

        total + tail_count
    }

    /// Apply E-Prop weight update.
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
        self.u.fill(-13.0);
        self.eligibility.fill(0.0);
        self.psi.fill(0.0);
    }
}
