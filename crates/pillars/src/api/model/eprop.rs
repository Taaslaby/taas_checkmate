//! # E-Prop Stateful Populations — канон pillars::api::neurons
//!
//! E-Prop (Eligibility Propagation, Bellec et al. 2020) обучаемые популяции
//! поверх `crates::kernels::eprop` lane-ядер:
//!
//! | Класс | Ядро | Описание |
//! |-------|------|----------|
//! | [`EPropIzhikevich`] | `eprop_izhikevich_step` | Izhikevich + eligibility traces |
//! | [`EPropIzhikevichMulti`] | `eprop_izhikevich_step_multi` | per-synapse eligibility `[n_pre × n_post]` |
//! | [`EPropSaw`] | `eprop_saw_step` | SAW + eligibility traces |
//! | [`EPropAdExMulti`] | `eprop_adex_step_multi` | AdEx + per-synapse eligibility |
//!
//! rayon используется при `feature = "rayon"` (default), иначе — serial.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::learning::surrogate_gradient;

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich
// ═══════════════════════════════════════════════════════════

/// E-Prop Izhikevich neuron with eligibility traces.
pub struct EPropIzhikevich {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub n: usize,
    /// Logical (unpadded) neuron count — the buffer-length contract for `step`
    /// and `update_weights` (callers pass `n`-length buffers, not padded).
    n_logical: usize,
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
            n: n_pad,
            n_logical: n,
            dt,
            tau_e,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            tau_surrogate: 2.0,
        }
    }

    /// One step of Izhikevich dynamics + eligibility trace update.
    /// Uses SIMD `f32x8` kernel — ~8× faster than scalar loop.
    ///
    /// # Panics
    /// If any of `I`/`spikes`/`pre_spikes` differs from the logical neuron
    /// count — mismatched buffers are a caller error and were previously
    /// silently truncated.
    pub fn step(&mut self, I: &[f32], spikes: &mut [i32], pre_spikes: &[i32]) -> usize {
        assert_eq!(
            I.len(),
            self.n_logical,
            "E-Prop step: I length must equal neuron count"
        );
        assert_eq!(
            spikes.len(),
            self.n_logical,
            "E-Prop step: spikes length must equal neuron count"
        );
        assert_eq!(
            pre_spikes.len(),
            self.n_logical,
            "E-Prop step: pre_spikes length must equal neuron count"
        );
        let n = self.n_logical;
        let n8 = n & !7;
        if n8 == 0 {
            return 0;
        }

        #[cfg(feature = "rayon")]
        let total: usize = self.v[..n8]
            .par_chunks_exact_mut(8)
            .zip(self.u[..n8].par_chunks_exact_mut(8))
            .zip(self.eligibility[..n8].par_chunks_exact_mut(8))
            .zip(self.psi[..n8].par_chunks_exact_mut(8))
            .zip(I[..n8].par_chunks_exact(8))
            .zip(spikes[..n8].par_chunks_exact_mut(8))
            .zip(pre_spikes[..n8].par_chunks_exact(8))
            .map(|((((((v8, u8), e8), ps8), I8), sp8), pre8)| {
                crate::kernels::eprop::eprop_izhikevich_step(
                    v8,
                    u8,
                    e8,
                    ps8,
                    I8,
                    sp8,
                    pre8,
                    self.dt,
                    self.tau_e,
                    self.a,
                    self.b,
                    self.c,
                    self.d,
                    self.tau_surrogate,
                )
            })
            .sum();
        #[cfg(not(feature = "rayon"))]
        let total: usize = {
            let mut total = 0usize;
            for i in (0..n8).step_by(8) {
                total += crate::kernels::eprop::eprop_izhikevich_step(
                    &mut self.v[i..i + 8],
                    &mut self.u[i..i + 8],
                    &mut self.eligibility[i..i + 8],
                    &mut self.psi[i..i + 8],
                    &I[i..i + 8],
                    &mut spikes[i..i + 8],
                    &pre_spikes[i..i + 8],
                    self.dt,
                    self.tau_e,
                    self.a,
                    self.b,
                    self.c,
                    self.d,
                    self.tau_surrogate,
                );
            }
            total
        };

        // Scalar tail (n & 7) — exact exponential decay, matching SIMD kernel
        let mut tail_count = 0;
        if n8 < n {
            let dt = self.dt;
            let tau_e = self.tau_e.max(1e-6);
            let a = self.a;
            let b = self.b;
            let c = self.c;
            let d = self.d;
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
                self.eligibility[i] =
                    self.eligibility[i] * e_decay + self.psi[i] * z_pre * tau_e * (1.0 - e_decay);
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
    ///
    /// # Panics
    /// If `weights`/`learning_signals` differs from the logical neuron count.
    pub fn update_weights(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        assert_eq!(
            weights.len(),
            self.n_logical,
            "E-Prop update: weights length must equal neuron count"
        );
        assert_eq!(
            learning_signals.len(),
            self.n_logical,
            "E-Prop update: learning_signals length must equal neuron count"
        );
        let n = self.n_logical;
        let chunks = n / 8;
        if chunks > 0 {
            #[cfg(feature = "rayon")]
            weights[..chunks * 8]
                .par_chunks_exact_mut(8)
                .zip(self.eligibility[..chunks * 8].par_chunks_exact(8))
                .zip(learning_signals[..chunks * 8].par_chunks_exact(8))
                .for_each(|((w8, e8), l8)| {
                    for i in 0..8 {
                        w8[i] += lr * l8[i] * e8[i];
                    }
                });
            #[cfg(not(feature = "rayon"))]
            for i in 0..chunks * 8 {
                weights[i] += lr * learning_signals[i] * self.eligibility[i];
            }
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
    pub e: Vec<f32>, // [n_pre × n_post] column-major
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
        assert!(n_pre > 0, "EPropIzhikevichMulti: n_pre must be > 0");
        assert!(n_post > 0, "EPropIzhikevichMulti: n_post must be > 0");
        let n_pad = n_post
            .checked_add(7)
            .and_then(|value| value.checked_div(8))
            .and_then(|lanes| lanes.checked_mul(8))
            .expect("EPropIzhikevichMulti: n_post overflows SIMD padding");
        let eligibility_len = n_pre
            .checked_mul(n_pad)
            .expect("EPropIzhikevichMulti: n_pre * n_post overflows");
        Self {
            v: vec![-65.0; n_pad],
            u: vec![-13.0; n_pad],
            psi: vec![0.0; n_pad],
            e: vec![0.0; eligibility_len],
            n_pre,
            n_post: n_pad,
            dt,
            tau_e,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 2.0,
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
        assert_eq!(
            I_ext.len(),
            self.n_post,
            "multi E-Prop step: I_ext length must equal padded n_post"
        );
        assert_eq!(
            spikes.len(),
            self.n_post,
            "multi E-Prop step: spikes length must equal padded n_post"
        );
        let expected_synapses = self
            .n_pre
            .checked_mul(self.n_post)
            .expect("multi E-Prop step: synapse shape overflow");
        assert_eq!(
            weights.len(),
            expected_synapses,
            "multi E-Prop step: weights length mismatch"
        );
        assert_eq!(
            z_pre.len(),
            self.n_pre,
            "multi E-Prop step: z_pre length mismatch"
        );

        let n_pre = self.n_pre;
        let dt = self.dt;
        let tau_e = self.tau_e;
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;
        let tau_sg = self.tau_surrogate;

        crate::kernels::eprop::eprop_izhikevich_step_multi(
            &mut self.v,
            &mut self.u,
            &mut self.psi,
            &mut self.e,
            z_pre,
            I_ext,
            weights,
            n_pre,
            dt,
            tau_e,
            a,
            b,
            c,
            d,
            tau_sg,
            spikes,
        )
    }

    /// Apply E-Prop weight update for column-major weight matrix.
    pub fn update_weights_multi(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        let n_post = self.n_post;
        let expected_synapses = self
            .n_pre
            .checked_mul(n_post)
            .expect("multi E-Prop update: synapse shape overflow");
        assert_eq!(
            self.e.len(),
            expected_synapses,
            "multi E-Prop update: eligibility shape mismatch"
        );
        assert_eq!(
            weights.len(),
            expected_synapses,
            "multi E-Prop update: weights length mismatch"
        );
        assert_eq!(
            learning_signals.len(),
            n_post,
            "multi E-Prop update: learning_signals length mismatch"
        );
        let n8 = n_post & !7;
        if n8 == 0 {
            return;
        }

        #[cfg(feature = "rayon")]
        self.e
            .par_chunks_exact(n_post)
            .zip(weights.par_chunks_exact_mut(n_post))
            .for_each(|(e_row, w_row)| {
                for i in 0..n8 {
                    w_row[i] += lr * learning_signals[i] * e_row[i];
                }
            });
        #[cfg(not(feature = "rayon"))]
        for row in 0..self.e.len() / n_post {
            let e_row = &self.e[row * n_post..(row + 1) * n_post];
            let w_row = &mut weights[row * n_post..(row + 1) * n_post];
            for i in 0..n8 {
                w_row[i] += lr * learning_signals[i] * e_row[i];
            }
        }
        // Scalar tail per row. НЕ cfg-gated сознательно: rayon-ветка обрабатывает
        // только [0..n8), tail [n8..n_post) выполняется в обоих конфигах —
        // двойного счёта нет.
        let n_rows = self.e.len() / n_post;
        for row in 0..n_rows {
            let e_row = &self.e[row * n_post..(row + 1) * n_post];
            let w_row = &mut weights[row * n_post..(row + 1) * n_post];
            for i in n8..n_post {
                w_row[i] += lr * learning_signals[i] * e_row[i];
            }
        }
    }

    pub fn reset(&mut self) {
        self.v.fill(-65.0);
        self.u.fill(-13.0);
        self.psi.fill(0.0);
        self.e.fill(0.0);
    }
}

// ═══════════════════════════════════════════════════════════
//  E-Prop SAW
// ═══════════════════════════════════════════════════════════

/// E-Prop SAW (Spike-Adaptive-Width) neuron with eligibility traces.
pub struct EPropSaw {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub n: usize,
    /// Logical (unpadded) neuron count — the buffer-length contract for `step`
    /// and `update_weights` (callers pass `n`-length buffers, not padded).
    n_logical: usize,
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
            n: n_pad,
            n_logical: n,
            dt,
            tau_e,
            a: 0.02,
            b: 0.2,
            tau_surrogate: 2.0,
        }
    }

    /// One step of SAW dynamics + eligibility trace update.
    /// Uses SIMD `f32x8` kernel.
    ///
    /// # Panics
    /// If any of `I`/`spikes`/`pre_spikes` differs from the logical neuron
    /// count — mismatched buffers are a caller error and were previously
    /// silently truncated.
    pub fn step(&mut self, I: &[f32], spikes: &mut [i32], pre_spikes: &[i32]) -> usize {
        assert_eq!(
            I.len(),
            self.n_logical,
            "E-Prop step: I length must equal neuron count"
        );
        assert_eq!(
            spikes.len(),
            self.n_logical,
            "E-Prop step: spikes length must equal neuron count"
        );
        assert_eq!(
            pre_spikes.len(),
            self.n_logical,
            "E-Prop step: pre_spikes length must equal neuron count"
        );
        let n = self.n_logical;
        let n8 = n & !7;
        if n8 == 0 {
            return 0;
        }

        #[cfg(feature = "rayon")]
        let total: usize = self.v[..n8]
            .par_chunks_exact_mut(8)
            .zip(self.u[..n8].par_chunks_exact_mut(8))
            .zip(self.eligibility[..n8].par_chunks_exact_mut(8))
            .zip(self.psi[..n8].par_chunks_exact_mut(8))
            .zip(I[..n8].par_chunks_exact(8))
            .zip(spikes[..n8].par_chunks_exact_mut(8))
            .zip(pre_spikes[..n8].par_chunks_exact(8))
            .map(|((((((v8, u8), e8), ps8), I8), sp8), pre8)| {
                crate::kernels::eprop::eprop_saw_step(
                    v8,
                    u8,
                    e8,
                    ps8,
                    I8,
                    sp8,
                    pre8,
                    self.dt,
                    self.tau_e,
                    self.a,
                    self.b,
                    self.tau_surrogate,
                )
            })
            .sum();
        #[cfg(not(feature = "rayon"))]
        let total: usize = {
            let mut total = 0usize;
            for i in (0..n8).step_by(8) {
                total += crate::kernels::eprop::eprop_saw_step(
                    &mut self.v[i..i + 8],
                    &mut self.u[i..i + 8],
                    &mut self.eligibility[i..i + 8],
                    &mut self.psi[i..i + 8],
                    &I[i..i + 8],
                    &mut spikes[i..i + 8],
                    &pre_spikes[i..i + 8],
                    self.dt,
                    self.tau_e,
                    self.a,
                    self.b,
                    self.tau_surrogate,
                );
            }
            total
        };

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
                self.eligibility[i] =
                    self.eligibility[i] * e_decay + self.psi[i] * z_pre * tau_e * (1.0 - e_decay);
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
    ///
    /// # Panics
    /// If `weights`/`learning_signals` differs from the logical neuron count.
    pub fn update_weights(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        assert_eq!(
            weights.len(),
            self.n_logical,
            "E-Prop update: weights length must equal neuron count"
        );
        assert_eq!(
            learning_signals.len(),
            self.n_logical,
            "E-Prop update: learning_signals length must equal neuron count"
        );
        let n = self.n_logical;
        let chunks = n / 8;
        if chunks > 0 {
            #[cfg(feature = "rayon")]
            weights[..chunks * 8]
                .par_chunks_exact_mut(8)
                .zip(self.eligibility[..chunks * 8].par_chunks_exact(8))
                .zip(learning_signals[..chunks * 8].par_chunks_exact(8))
                .for_each(|((w8, e8), l8)| {
                    for i in 0..8 {
                        w8[i] += lr * l8[i] * e8[i];
                    }
                });
            #[cfg(not(feature = "rayon"))]
            for i in 0..chunks * 8 {
                weights[i] += lr * learning_signals[i] * self.eligibility[i];
            }
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
//  E-Prop AdEx Multi — per-synapse eligibility for AdEx
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
        assert!(n_pre > 0, "EPropAdExMulti: n_pre must be > 0");
        assert!(n_post > 0, "EPropAdExMulti: n_post must be > 0");
        let n_pad = n_post
            .checked_add(7)
            .and_then(|value| value.checked_div(8))
            .and_then(|lanes| lanes.checked_mul(8))
            .expect("EPropAdExMulti: n_post overflows SIMD padding");
        let eligibility_len = n_pre
            .checked_mul(n_pad)
            .expect("EPropAdExMulti: n_pre * n_post overflows");
        Self {
            v: vec![v_rest; n_pad],
            w: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            e: vec![0.0; eligibility_len],
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
        assert_eq!(
            I_ext.len(),
            self.n_post,
            "multi E-Prop step: I_ext length must equal padded n_post"
        );
        assert_eq!(
            spikes.len(),
            self.n_post,
            "multi E-Prop step: spikes length must equal padded n_post"
        );
        let expected_synapses = self
            .n_pre
            .checked_mul(self.n_post)
            .expect("multi E-Prop step: synapse shape overflow");
        assert_eq!(
            weights.len(),
            expected_synapses,
            "multi E-Prop step: weights length mismatch"
        );
        assert_eq!(
            z_pre.len(),
            self.n_pre,
            "multi E-Prop step: z_pre length mismatch"
        );

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

        crate::kernels::eprop::eprop_adex_step_multi(
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
        )
    }

    /// Apply E-Prop weight update for column-major weight matrix.
    pub fn update_weights_multi(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        let n_post = self.n_post;
        let expected_synapses = self
            .n_pre
            .checked_mul(n_post)
            .expect("multi E-Prop update: synapse shape overflow");
        assert_eq!(
            self.e.len(),
            expected_synapses,
            "multi E-Prop update: eligibility shape mismatch"
        );
        assert_eq!(
            weights.len(),
            expected_synapses,
            "multi E-Prop update: weights length mismatch"
        );
        assert_eq!(
            learning_signals.len(),
            n_post,
            "multi E-Prop update: learning_signals length mismatch"
        );
        let n8 = n_post & !7;
        if n8 == 0 {
            return;
        }

        #[cfg(feature = "rayon")]
        self.e
            .par_chunks_exact(n_post)
            .zip(weights.par_chunks_exact_mut(n_post))
            .for_each(|(e_row, w_row)| {
                for i in 0..n8 {
                    w_row[i] += lr * learning_signals[i] * e_row[i];
                }
            });
        #[cfg(not(feature = "rayon"))]
        for row in 0..self.e.len() / n_post {
            let e_row = &self.e[row * n_post..(row + 1) * n_post];
            let w_row = &mut weights[row * n_post..(row + 1) * n_post];
            for i in 0..n8 {
                w_row[i] += lr * learning_signals[i] * e_row[i];
            }
        }
        // Scalar tail per row. НЕ cfg-gated сознательно: rayon-ветка обрабатывает
        // только [0..n8), tail [n8..n_post) выполняется в обоих конфигах —
        // двойного счёта нет.
        let n_rows = self.e.len() / n_post;
        for row in 0..n_rows {
            let e_row = &self.e[row * n_post..(row + 1) * n_post];
            let w_row = &mut weights[row * n_post..(row + 1) * n_post];
            for i in n8..n_post {
                w_row[i] += lr * learning_signals[i] * e_row[i];
            }
        }
    }

    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
        self.w.fill(0.0);
        self.psi.fill(0.0);
        self.e.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eprop_izhikevich_fires_with_input() {
        let mut pop = EPropIzhikevich::new(8, 1.0, 20.0);
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let pre = vec![1i32; 8];
        let mut total = 0usize;
        for _ in 0..300 {
            total += pop.step(&i_ext, &mut spikes, &pre);
        }
        assert!(total > 0, "EPropIzhikevich should fire with I=30");
    }

    #[test]
    fn eprop_saw_fires_with_input() {
        let mut pop = EPropSaw::new(8, 1.0, 20.0);
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let pre = vec![1i32; 8];
        let mut total = 0usize;
        for _ in 0..300 {
            total += pop.step(&i_ext, &mut spikes, &pre);
        }
        assert!(total > 0, "EPropSaw should fire with I=30");
    }

    #[test]
    fn eprop_multi_step_runs() {
        let mut pop = EPropIzhikevichMulti::new(4, 8, 1.0, 20.0);
        let weights = vec![0.5f32; 4 * 8];
        let z_pre = vec![1i32; 4];
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0usize;
        for _ in 0..200 {
            total += pop.step(&weights, &z_pre, &i_ext, &mut spikes);
        }
        assert!(total > 0, "EPropIzhikevichMulti should fire");
    }

    #[test]
    fn eprop_adex_multi_step_runs() {
        let mut pop = EPropAdExMulti::new_default(4, 8, 1.0, 20.0);
        let weights = vec![0.5f32; 4 * 8];
        let z_pre = vec![1i32; 4];
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0usize;
        for _ in 0..200 {
            total += pop.step(&weights, &z_pre, &i_ext, &mut spikes);
        }
        assert!(total > 0, "EPropAdExMulti should fire");
    }

    #[test]
    #[should_panic(expected = "multi E-Prop step: weights length mismatch")]
    fn eprop_izh_multi_rejects_mismatched_weights() {
        let mut pop = EPropIzhikevichMulti::new(2, 8, 1.0, 20.0);
        let mut spikes = vec![0i32; 8];
        pop.step(&[0.0; 15], &[1, 0], &[0.0; 8], &mut spikes);
    }

    #[test]
    #[should_panic(expected = "multi E-Prop update: learning_signals length mismatch")]
    fn eprop_adex_multi_rejects_mismatched_learning_signals() {
        let pop = EPropAdExMulti::new_default(2, 8, 1.0, 20.0);
        let mut weights = vec![0.0f32; 2 * 8];
        pop.update_weights_multi(&mut weights, &[1.0; 7], 0.1);
    }

    #[test]
    #[should_panic(expected = "n_pre must be > 0")]
    fn eprop_multi_rejects_zero_pre_dimension() {
        let _ = EPropIzhikevichMulti::new(0, 8, 1.0, 20.0);
    }

    #[test]
    #[should_panic(expected = "n_post must be > 0")]
    fn eprop_multi_rejects_zero_post_dimension() {
        let _ = EPropAdExMulti::new_default(2, 0, 1.0, 20.0);
    }

    #[test]
    #[should_panic(expected = "overflows SIMD padding")]
    fn eprop_multi_rejects_padding_overflow() {
        let _ = EPropIzhikevichMulti::new(1, usize::MAX, 1.0, 20.0);
    }

    #[test]
    #[should_panic(expected = "I length must equal neuron count")]
    fn eprop_izhikevich_rejects_mismatched_I() {
        let mut pop = EPropIzhikevich::new(8, 1.0, 20.0);
        let mut spikes = vec![0i32; 8];
        pop.step(&[0.0f32; 7], &mut spikes, &[0i32; 8]);
    }

    #[test]
    #[should_panic(expected = "spikes length must equal neuron count")]
    fn eprop_izhikevich_rejects_mismatched_spikes() {
        let mut pop = EPropIzhikevich::new(8, 1.0, 20.0);
        let mut spikes = vec![0i32; 7];
        pop.step(&[0.0f32; 8], &mut spikes, &[0i32; 8]);
    }

    #[test]
    #[should_panic(expected = "pre_spikes length must equal neuron count")]
    fn eprop_izhikevich_rejects_mismatched_pre_spikes() {
        let mut pop = EPropIzhikevich::new(8, 1.0, 20.0);
        let mut spikes = vec![0i32; 8];
        pop.step(&[0.0f32; 8], &mut spikes, &[0i32; 7]);
    }

    #[test]
    #[should_panic(expected = "weights length must equal neuron count")]
    fn eprop_izhikevich_rejects_mismatched_weights() {
        let pop = EPropIzhikevich::new(8, 1.0, 20.0);
        let mut weights = vec![0.0f32; 7];
        pop.update_weights(&mut weights, &[1.0f32; 8], 0.1);
    }

    #[test]
    #[should_panic(expected = "learning_signals length must equal neuron count")]
    fn eprop_saw_rejects_mismatched_learning_signals() {
        let pop = EPropSaw::new(8, 1.0, 20.0);
        let mut weights = vec![0.0f32; 8];
        pop.update_weights(&mut weights, &[1.0f32; 7], 0.1);
    }

    #[test]
    #[should_panic(expected = "I length must equal neuron count")]
    fn eprop_saw_rejects_mismatched_I() {
        let mut pop = EPropSaw::new(8, 1.0, 20.0);
        let mut spikes = vec![0i32; 8];
        pop.step(&[0.0f32; 7], &mut spikes, &[0i32; 8]);
    }
}
