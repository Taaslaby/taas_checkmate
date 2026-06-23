//! Izhikevich neuron model — step + E-Prop variants.
//!
//! Safe Rust wrappers over SIMD kernels (`kernels::izhikevich_step`,
//! `kernels::eprop_izhikevich_step`, `kernels::eprop_izhikevich_step_multi`).

use crate::kernels;
use crate::io;
use rayon::prelude::*;

use super::eprop::surrogate_gradient;
use super::I_CLAMP;

// ─── step_izhikevich ──────────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Step `n` Izhikevich neurons by `dt`. Parallel. n must be multiple of 8.
///
/// Supports refractory period via `refractory` buffer (0 = not refractory).
pub fn step_izhikevich(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], I: &mut [f32],
    dt: f32, trace_decay: f32,
    spikes: &mut [i32], c: &[f32], d: &[f32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == u.len() && n == trace.len() && n == a.len() && n == b.len()
        && n == I.len() && n == spikes.len() && n == c.len() && n == d.len() && n == refractory.len());

    io::safety_clamp(I, -I_CLAMP, I_CLAMP);

    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(trace.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .zip(a.par_chunks_exact(8)
            .zip(b.par_chunks_exact(8))
            .zip(c.par_chunks_exact(8))
            .zip(d.par_chunks_exact(8)))
        .for_each(|((((((v8, u8), tr8), I8), sp8), refr8), (((a8, b8), c8), d8))| {
            unsafe {
                kernels::izhikevich_step(
                    v8.as_mut_ptr(), u8.as_mut_ptr(), tr8.as_mut_ptr(),
                    a8.as_ptr(), b8.as_ptr(), I8.as_mut_ptr(),
                    dt, trace_decay,
                    sp8.as_mut_ptr(), c8.as_ptr(), d8.as_ptr(),
                    refr8.as_mut_ptr(), tau_ref, 8,
                );
            }
        });
}

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich — Eligibility Propagation for Izhikevich neurons
// ═══════════════════════════════════════════════════════════

/// E-Prop Izhikevich neuron with eligibility traces.
pub struct EPropIzhikevich {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_e: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tau_surrogate: f32,
}

impl EPropIzhikevich {
    pub fn new(n: usize, dt: f32, tau_e: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            v: vec![-65.0; n_pad],
            u: vec![-13.0; n_pad],
            eligibility: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            n: n_pad, dt, tau_e,
            a: 0.02, b: 0.2, c: -65.0, d: 8.0,
            tau_surrogate: 2.0,
        }
    }

    /// One step of Izhikevich dynamics + eligibility trace update.
    /// Uses SIMD `f32x8` kernel — ~8× faster than scalar loop.
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
                unsafe {
                    kernels::eprop_izhikevich_step(
                        v8.as_mut_ptr(), u8.as_mut_ptr(), e8.as_mut_ptr(), ps8.as_mut_ptr(),
                        I8.as_ptr(), sp8.as_mut_ptr(), pre8.as_ptr(),
                        self.dt, self.tau_e, self.a, self.b, self.c, self.d, self.tau_surrogate,
                        8,
                    )
                }
            })
            .sum();

        // Scalar tail (n & 7)
        let mut tail_count = 0;
        if n8 < n {
            let dt = self.dt;
            let tau_e = self.tau_e;
            let a = self.a;
            let b = self.b;
            let c = self.c;
            let d = self.d;
            let tau_sg = self.tau_surrogate;
            for i in n8..n {
                let vv = self.v[i];
                let dv = (0.04 * vv + 5.0) * vv + 140.0 - self.u[i] + I[i];
                let du = a * (b * vv - self.u[i]);
                self.v[i] += dt * dv;
                self.u[i] += dt * du;
                self.psi[i] = surrogate_gradient(self.v[i], 30.0, tau_sg);
                let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
                self.eligibility[i] += dt * (-self.eligibility[i] / tau_e + self.psi[i] * z_pre);
                if self.v[i] >= 30.0 {
                    spikes[i] = 1;
                    tail_count += 1;
                    self.v[i] = c;
                    self.u[i] += d;
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

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich Multi — per-synapse eligibility
// ═══════════════════════════════════════════════════════════

/// E-Prop Izhikevich with per-synapse eligibility traces.
///
/// Each post-neuron maintains `n_pre` eligibility traces, one per input.
/// Column-major layout: `e[i * n_post + j]` = eligibility for synapse (i→j).
pub struct EPropIzhikevichMulti {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub psi: Vec<f32>,
    pub e: Vec<f32>,       // [n_pre × n_post] column-major
    pub n_pre: usize,
    pub n_post: usize,
    pub dt: f32,
    pub tau_e: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tau_surrogate: f32,
}

impl EPropIzhikevichMulti {
    pub fn new(n_pre: usize, n_post: usize, dt: f32, tau_e: f32) -> Self {
        let n_pad = n_post.div_ceil(8) * 8;
        Self {
            v: vec![-65.0; n_pad],
            u: vec![-13.0; n_pad],
            psi: vec![0.0; n_pad],
            e: vec![0.0; n_pre * n_pad],
            n_pre,
            n_post: n_pad,
            dt, tau_e,
            a: 0.02, b: 0.2, c: -65.0, d: 2.0,
            tau_surrogate: 5.0,
        }
    }

    /// One step of Izhikevich dynamics + per-synapse eligibility update.
    pub fn step(
        &mut self,
        weights: &[f32],
        z_pre: &[i32],
        I_ext: &[f32],
        spikes: &mut [i32],
    ) -> usize {
        let n = self.n_post.min(I_ext.len()).min(spikes.len());
        if n == 0 { return 0; }

        let n_pre = self.n_pre;
        let dt = self.dt;
        let tau_e = self.tau_e;
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;
        let tau_sg = self.tau_surrogate;

        unsafe {
            kernels::eprop_izhikevich_step_multi(
                self.v.as_mut_ptr(),
                self.u.as_mut_ptr(),
                self.psi.as_mut_ptr(),
                self.e.as_mut_ptr(),
                z_pre.as_ptr(),
                I_ext.as_ptr(),
                weights.as_ptr(),
                n,        // column stride = n_post
                n_pre,
                dt, tau_e, a, b, c, d, tau_sg,
                spikes.as_mut_ptr(),
            )
        }
    }

    /// Apply E-Prop weight update for column-major weight matrix.
    pub fn update_weights_multi(
        &self,
        weights: &mut [f32],
        learning_signals: &[f32],
        lr: f32,
    ) {
        let n_post = self.n_post;
        let n8 = n_post & !7;
        if n8 == 0 { return; }

        self.e.par_chunks_exact(n_post)
            .zip(weights.par_chunks_exact_mut(n_post))
            .for_each(|(e_row, w_row)| {
                w_row[..n8].chunks_exact_mut(8)
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
        self.v.fill(-65.0);
        self.u.fill(-13.0);
        self.psi.fill(0.0);
        self.e.fill(0.0);
    }
}
