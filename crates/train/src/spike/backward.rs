//! Backward pass: BPTT through T3 Stuart-Landau + Kuramoto dynamics.
//!
//! SIMD-оптимизированная версия (f32x8):
//! - **ноль f64** в горячем цикле — авто-векторизация не блокируется
//! - **f32x8** для sin/cos (через wide) и всей арифметики
//! - **BpttWorkspace** — преаллоцированные буферы, новая аллокация только при
//!   изменении размеров (n_osc, n_steps)
//! - **PstJacobianBuf + PstGradients** — ноль аллокаций на шаг в SPIGRAD
//!
//! ## Trig mismatch (forward vs backward)
//!
//! Forward pass использует `fast_sincos_simd` (minimax polynomial, ~1e-4 accuracy),
//! backward pass использует `f32x8::sin_cos()` (libm, ~1 ULP).
//! Это означает, что синусы/косинусы forward и backward отличаются на ~1e-4,
//! что даёт ~0.03% систематической ошибки в градиентах.
//! Для <1000 шагов BPTT это несущественно (тесты проходят с tolerance 15-25%).
//! Для длинных rollouts (>10000 шагов) стоит переключить backward на `fast_sincos_simd`.
//!
//! ## Forward dynamics (per oscillator i, per population):
//!
//! ```text
//! r' = r + dt · ((α - r²)·r + drive · gain)
//! θ' = θ + dt · (ω + coupling + drive · gain / (r + ε))
//! ```
use wide::f32x8;
use wide::{CmpGt, CmpLt};

use crate::spike::pst_jacobian::{pst_jacobian_v_jit, PstGradients};
use crate::spike::trajectory::{
    off_drive1, off_drive2, off_drive3, off_r1, off_r2, off_r3, off_th2, off_th3,
    T3SpikeTrajectory, OFF_TH1,
};
use crate::t3::pst_linear_dim;

const EPS: f32 = 0.1;
const PG_GAIN: f32 = 4.0;

// ─── Phase gate (via t3::phase_gate) ─────────────────────

/// Compute R from sin/cos (f64 accumulator) and apply phase gate.
#[inline]
fn phase_gate_from_sincos(sin: &[f32], cos: &[f32], n: usize, gamma: f32) -> f32 {
    let (ss, sc): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
        (s + sin[i] as f64, c + cos[i] as f64)
    });
    let r = ((ss * ss + sc * sc).sqrt() / n as f64) as f32;
    crate::t3::phase_gate(r, gamma, n, PG_GAIN)
}

// ─── r-factor gate (branchless SIMD) ─────────────────────

fn r_clamp_effect(r: f32, drive: f32, dt: f32, gain: f32, alpha: f32) -> f32 {
    let rp = r as f64
        + (dt as f64)
            * ((alpha as f64 - r as f64 * r as f64) * r as f64 + drive as f64 * gain as f64);
    if !(0.001..=10.0).contains(&rp) {
        0.0
    } else {
        1.0
    }
}

#[inline]
fn r_factor_gate(r: f32, drive: f32, dt: f32, gain: f32, alpha: f32) -> f32 {
    r_clamp_effect(r, drive, dt, gain, alpha)
}

#[inline]
fn r_factor_gate_f32x8(r: f32x8, drive: f32x8, dt: f32, gain: f32, alpha: f32) -> f32x8 {
    let rp =
        r + f32x8::splat(dt) * ((f32x8::splat(alpha) - r * r) * r + drive * f32x8::splat(gain));
    let mask_lo = rp.cmp_lt(f32x8::splat(0.001));
    let mask_hi = rp.cmp_gt(f32x8::splat(10.0));
    (mask_lo | mask_hi).blend(f32x8::splat(0.0), f32x8::splat(1.0))
}

// ─── BpttWorkspace ───────────────────────────────────────

/// Pre-allocated workspace for BPTT — **ноль аллокаций** на backward pass.
///
/// Все буферы растут только при необходимости и никогда не сжимаются.
pub struct BpttWorkspace {
    /// Recurrent gradient buffers (size n, 3 pops each)
    pub v_th1: Vec<f32>,
    pub v_th2: Vec<f32>,
    pub v_th3: Vec<f32>,
    pub v_r1: Vec<f32>,
    pub v_r2: Vec<f32>,
    pub v_r3: Vec<f32>,
    /// Temp sin/cos buffers (n per pop)
    sin1: Vec<f32>,
    cos1: Vec<f32>,
    sin2: Vec<f32>,
    cos2: Vec<f32>,
    sin3: Vec<f32>,
    cos3: Vec<f32>,
    /// Grad drives [n_steps × 3n]
    pub grad_drives: Vec<f32>,
    /// Pre-allocated PST gradient buffers (for SPIGRAD K1/K1')
    pub pst_v: PstGradients,
}

impl BpttWorkspace {
    /// Create workspace pre-sized for `n` oscillators and `n_steps` time steps.
    pub fn new(n: usize, n_steps: usize) -> Self {
        Self {
            v_th1: vec![0.0; n],
            v_th2: vec![0.0; n],
            v_th3: vec![0.0; n],
            v_r1: vec![0.0; n],
            v_r2: vec![0.0; n],
            v_r3: vec![0.0; n],
            sin1: vec![0.0; n],
            cos1: vec![0.0; n],
            sin2: vec![0.0; n],
            cos2: vec![0.0; n],
            sin3: vec![0.0; n],
            cos3: vec![0.0; n],
            grad_drives: vec![0.0; n_steps * 3 * n],
            pst_v: PstGradients::new(n),
        }
    }

    /// Grow buffers if needed — keep existing contents, only zero-extend.
    pub fn resize(&mut self, n: usize, n_steps: usize) {
        let grow = |v: &mut Vec<f32>, len: usize| {
            if v.len() < len {
                v.resize(len, 0.0);
            }
        };
        grow(&mut self.v_th1, n);
        grow(&mut self.v_th2, n);
        grow(&mut self.v_th3, n);
        grow(&mut self.v_r1, n);
        grow(&mut self.v_r2, n);
        grow(&mut self.v_r3, n);
        grow(&mut self.sin1, n);
        grow(&mut self.cos1, n);
        grow(&mut self.sin2, n);
        grow(&mut self.cos2, n);
        grow(&mut self.sin3, n);
        grow(&mut self.cos3, n);
        grow(&mut self.grad_drives, n_steps * 3 * n);
        self.pst_v.resize(n);
    }
}

// ─── SIMD load/store helpers ─────────────────────────────

#[inline]
fn ld(slice: &[f32], i: usize) -> f32x8 {
    f32x8::new([
        slice[i],
        slice[i + 1],
        slice[i + 2],
        slice[i + 3],
        slice[i + 4],
        slice[i + 5],
        slice[i + 6],
        slice[i + 7],
    ])
}

#[inline]
fn st(slice: &mut [f32], i: usize, v: f32x8) {
    let arr = v.as_array_ref();
    slice[i..i + 8].copy_from_slice(arr);
}

/// Helper: get step slice from flat trajectory
fn step_slice(data: &[f32], off: usize, n: usize, step: usize) -> &[f32] {
    &data[off + step * n..off + (step + 1) * n]
}

/// Helper: mutable step slice from flat trajectory
fn step_slice_mut(data: &mut [f32], off: usize, n: usize, step: usize) -> &mut [f32] {
    let start = off + step * n;
    &mut data[start..start + n]
}

// ─── SIMD per-step helpers ───────────────────────────────

/// Pass 1a: sin/cos + reductions for ONE population.
/// Returns (r_acc_sin, r_acc_cos, sin_acc, cos_acc, v_sin_acc, v_cos_acc).
fn simd_pass1_pop(
    n: usize,
    pre_th: &[f32],
    pre_r: &[f32],
    v_th: &[f32],
    sin: &mut [f32],
    cos: &mut [f32],
) -> (f32, f32, f32, f32, f32, f32) {
    let n_chunks = n / 8;
    let mut r_ss_v = f32x8::splat(0.0);
    let mut r_cc_v = f32x8::splat(0.0);
    let mut s_acc_v = f32x8::splat(0.0);
    let mut c_acc_v = f32x8::splat(0.0);
    let mut v_ss_v = f32x8::splat(0.0);
    let mut v_cc_v = f32x8::splat(0.0);

    for ch in 0..n_chunks {
        let i = ch * 8;
        let (s, c) = ld(pre_th, i).sin_cos();
        st(sin, i, s);
        st(cos, i, c);
        let r = ld(pre_r, i);
        let v = ld(v_th, i);
        r_ss_v += r * s;
        r_cc_v += r * c;
        s_acc_v += s;
        c_acc_v += c;
        v_ss_v += v * s;
        v_cc_v += v * c;
    }

    let mut r_ss_s = 0.0;
    let mut r_cc_s = 0.0;
    let mut s_acc_s = 0.0;
    let mut c_acc_s = 0.0;
    let mut v_ss_s = 0.0;
    let mut v_cc_s = 0.0;
    let rem = n % 8;
    if rem > 0 {
        let base = n_chunks * 8;
        for j in 0..rem {
            let idx = base + j;
            let s = pre_th[idx].sin();
            sin[idx] = s;
            let c = pre_th[idx].cos();
            cos[idx] = c;
            let r = pre_r[idx];
            let v = v_th[idx];
            r_ss_s += r * s;
            r_cc_s += r * c;
            s_acc_s += s;
            c_acc_s += c;
            v_ss_s += v * s;
            v_cc_s += v * c;
        }
    }

    (
        r_ss_v.reduce_add() + r_ss_s,
        r_cc_v.reduce_add() + r_cc_s,
        s_acc_v.reduce_add() + s_acc_s,
        c_acc_v.reduce_add() + c_acc_s,
        v_ss_v.reduce_add() + v_ss_s,
        v_cc_v.reduce_add() + v_cc_s,
    )
}

/// Pass 2: SIMD BPTT arithmetic — update v_th/v_r + write grad_drives.
#[allow(clippy::too_many_arguments)]
fn simd_pass2(
    n: usize,
    dt: f32,
    gain: f32,
    alpha: f32,
    pre_r1: &[f32],
    d1: &[f32],
    pre_r2: &[f32],
    d2: &[f32],
    pre_r3: &[f32],
    d3: &[f32],
    sin1: &[f32],
    cos1: &[f32],
    sin2: &[f32],
    cos2: &[f32],
    sin3: &[f32],
    cos3: &[f32],
    r1s1: f32,
    r1c1: f32,
    r2s2: f32,
    r2c2: f32,
    r3s3: f32,
    r3c3: f32,
    svs1: f32,
    svc1: f32,
    svs2: f32,
    svc2: f32,
    svs3: f32,
    svc3: f32,
    sin1_sum: f32,
    cos1_sum: f32,
    sin2_sum: f32,
    cos2_sum: f32,
    sin3_sum: f32,
    cos3_sum: f32,
    g1: f32,
    g2: f32,
    g3: f32,
    pg_pre1: f32,
    pg_pre2: f32,
    pg_pre3: f32,
    cross_gate1: f32,
    cross_gate2: f32,
    cross_gate3: f32,
    w12: f32,
    w13: f32,
    w23: f32,
    grad_drives: &mut [f32],
    d_off: usize,
    v_th1: &mut [f32],
    v_th2: &mut [f32],
    v_th3: &mut [f32],
    v_r1: &mut [f32],
    v_r2: &mut [f32],
    v_r3: &mut [f32],
    step: usize,
    n_steps: usize,
    grad_pst: &[f32],
) {
    let s_dt = f32x8::splat(dt);
    let _s_gain = f32x8::splat(gain);
    let s_alpha = f32x8::splat(alpha);
    let s_w12g2 = f32x8::splat(w12 * g2);
    let s_w13g3 = f32x8::splat(w13 * g3);
    let s_w12g1 = f32x8::splat(w12 * g1);
    let s_w13g1 = f32x8::splat(w13 * g1);
    let _s_w23g2 = f32x8::splat(w23 * g2);
    let s_w12_2 = f32x8::splat(w12 * g2);
    let s_w23_2 = f32x8::splat(w23 * g3);
    let s_w13_3 = f32x8::splat(w13 * g3);
    let s_w23_3 = f32x8::splat(w23 * g2);
    let s_r2c2 = f32x8::splat(r2c2);
    let s_r2s2 = f32x8::splat(r2s2);
    let s_r3c3 = f32x8::splat(r3c3);
    let s_r3s3 = f32x8::splat(r3s3);
    let s_r1c1 = f32x8::splat(r1c1);
    let s_r1s1 = f32x8::splat(r1s1);
    let s_svc2 = f32x8::splat(svc2);
    let s_svs2 = f32x8::splat(svs2);
    let s_svc3 = f32x8::splat(svc3);
    let s_svs3 = f32x8::splat(svs3);
    let s_svc1 = f32x8::splat(svc1);
    let s_svs1 = f32x8::splat(svs1);
    let s_cross1 = f32x8::splat(cross_gate1);
    let s_cross2 = f32x8::splat(cross_gate2);
    let s_cross3 = f32x8::splat(cross_gate3);
    let s_w12g1_2 = f32x8::splat(w12 * g1);
    let s_w23g3 = f32x8::splat(w23 * g3);
    let s_w13g1_2 = f32x8::splat(w13 * g1);
    let s_w23g2 = f32x8::splat(w23 * g2);
    let s_pg1_sc = f32x8::splat(pg_pre1 * sin1_sum);
    let s_pg1_cs = f32x8::splat(pg_pre1 * cos1_sum);
    let s_pg2_sc = f32x8::splat(pg_pre2 * sin2_sum);
    let s_pg2_cs = f32x8::splat(pg_pre2 * cos2_sum);
    let s_pg3_sc = f32x8::splat(pg_pre3 * sin3_sum);
    let s_pg3_cs = f32x8::splat(pg_pre3 * cos3_sum);
    let s_eps = f32x8::splat(EPS);
    let s_one = f32x8::splat(1.0);
    let s_three = f32x8::splat(3.0);
    let s_dt_gain = f32x8::splat(dt * gain);

    let n_chunks = n / 8;
    for c in 0..n_chunks {
        let i = c * 8;
        let s1 = ld(sin1, i);
        let c1 = ld(cos1, i);
        let s2 = ld(sin2, i);
        let c2 = ld(cos2, i);
        let s3 = ld(sin3, i);
        let c3 = ld(cos3, i);
        let r1 = ld(pre_r1, i);
        let d1v = ld(d1, i);
        let r2 = ld(pre_r2, i);
        let d2v = ld(d2, i);
        let r3 = ld(pre_r3, i);
        let d3v = ld(d3, i);
        let vth1 = ld(v_th1, i);
        let vr1 = ld(v_r1, i);
        let vth2 = ld(v_th2, i);
        let vr2 = ld(v_r2, i);
        let vth3 = ld(v_th3, i);
        let vr3 = ld(v_r3, i);

        let dcpl1 =
            -(s_w12g2 * (c1 * s_r2c2 + s1 * s_r2s2) + s_w13g3 * (c1 * s_r3c3 + s1 * s_r3s3));
        let dcpl2 =
            -(s_w12_2 * (c2 * s_r1c1 + s2 * s_r1s1) + s_w23_2 * (c2 * s_r3c3 + s2 * s_r3s3));
        let dcpl3 =
            -(s_w13_3 * (c3 * s_r1c1 + s3 * s_r1s1) + s_w23_3 * (c3 * s_r2c2 + s3 * s_r2s2));

        let vt1_self = vth1 * (s_one + s_dt * dcpl1);
        let vt2_self = vth2 * (s_one + s_dt * dcpl2);
        let vt3_self = vth3 * (s_one + s_dt * dcpl3);

        let dg1 = s_pg1_sc * c1 - s_pg1_cs * s1;
        let dg2 = s_pg2_sc * c2 - s_pg2_cs * s2;
        let dg3 = s_pg3_sc * c3 - s_pg3_cs * s3;

        let vt1_cross = s_dt
            * (r1
                * (s_w12g1 * (c1 * s_svc2 + s1 * s_svs2) + s_w13g1 * (c1 * s_svc3 + s1 * s_svs3))
                + dg1 * s_cross1);
        let vt2_cross = s_dt
            * (r2
                * (s_w12g1_2 * (c2 * s_svc1 + s2 * s_svs1)
                    + s_w23g3 * (c2 * s_svc3 + s2 * s_svs3))
                + dg2 * s_cross2);
        let vt3_cross = s_dt
            * (r3
                * (s_w13g1_2 * (c3 * s_svc1 + s3 * s_svs1)
                    + s_w23g2 * (c3 * s_svc2 + s3 * s_svs2))
                + dg3 * s_cross3);

        let r_factor1 = s_one + s_dt * (s_alpha - s_three * r1 * r1);
        let rg1 = r_factor_gate_f32x8(r1, d1v, dt, gain, alpha);
        let vr1_out =
            vr1 * r_factor1 * rg1 + vth1 * (-s_dt_gain * d1v / ((r1 + s_eps) * (r1 + s_eps)));

        let r_factor2 = s_one + s_dt * (s_alpha - s_three * r2 * r2);
        let rg2 = r_factor_gate_f32x8(r2, d2v, dt, gain, alpha);
        let vr2_out =
            vr2 * r_factor2 * rg2 + vth2 * (-s_dt_gain * d2v / ((r2 + s_eps) * (r2 + s_eps)));

        let r_factor3 = s_one + s_dt * (s_alpha - s_three * r3 * r3);
        let rg3 = r_factor_gate_f32x8(r3, d3v, dt, gain, alpha);
        let vr3_out =
            vr3 * r_factor3 * rg3 + vth3 * (-s_dt_gain * d3v / ((r3 + s_eps) * (r3 + s_eps)));

        st(v_th1, i, vt1_self + vt1_cross);
        st(v_th2, i, vt2_self + vt2_cross);
        st(v_th3, i, vt3_self + vt3_cross);
        st(v_r1, i, vr1_out);
        st(v_r2, i, vr2_out);
        st(v_r3, i, vr3_out);

        let dth_dd1 = s_dt_gain / (r1 + s_eps);
        let dth_dd2 = s_dt_gain / (r2 + s_eps);
        let dth_dd3 = s_dt_gain / (r3 + s_eps);
        st(
            grad_drives,
            d_off + i,
            vth1 * dth_dd1 + vr1 * s_dt_gain * rg1,
        );
        st(
            grad_drives,
            d_off + n + i,
            vth2 * dth_dd2 + vr2 * s_dt_gain * rg2,
        );
        st(
            grad_drives,
            d_off + 2 * n + i,
            vth3 * dth_dd3 + vr3 * s_dt_gain * rg3,
        );
    }

    let rem = n % 8;
    if rem > 0 {
        let start = n_chunks * 8;
        let dt_f = dt as f64;
        let gain_f = gain as f64;
        let alpha_f = alpha as f64;
        let eps_f = EPS as f64;
        let w12_f = w12 as f64;
        let w13_f = w13 as f64;
        let w23_f = w23 as f64;
        let g1_f = g1 as f64;
        let g2_f = g2 as f64;
        let g3_f = g3 as f64;
        let r1c1_f = r1c1 as f64;
        let r1s1_f = r1s1 as f64;
        let r2c2_f = r2c2 as f64;
        let r2s2_f = r2s2 as f64;
        let r3c3_f = r3c3 as f64;
        let r3s3_f = r3s3 as f64;
        let svc1_f = svc1 as f64;
        let svs1_f = svs1 as f64;
        let svc2_f = svc2 as f64;
        let svs2_f = svs2 as f64;
        let svc3_f = svc3 as f64;
        let svs3_f = svs3 as f64;
        let sc1_f = sin1_sum as f64;
        let cc1_f = cos1_sum as f64;
        let sc2_f = sin2_sum as f64;
        let cc2_f = cos2_sum as f64;
        let sc3_f = sin3_sum as f64;
        let cc3_f = cos3_sum as f64;
        let cg1_f = cross_gate1 as f64;
        let cg2_f = cross_gate2 as f64;
        let cg3_f = cross_gate3 as f64;
        let pp1_f = pg_pre1 as f64;
        let pp2_f = pg_pre2 as f64;
        let pp3_f = pg_pre3 as f64;

        for j in 0..rem {
            let i = start + j;
            let s1_f = sin1[i] as f64;
            let c1_f = cos1[i] as f64;
            let s2_f = sin2[i] as f64;
            let c2_f = cos2[i] as f64;
            let s3_f = sin3[i] as f64;
            let c3_f = cos3[i] as f64;
            let r1_f = pre_r1[i] as f64;
            let d1v_f = d1[i] as f64;
            let r2_f = pre_r2[i] as f64;
            let d2v_f = d2[i] as f64;
            let r3_f = pre_r3[i] as f64;
            let d3v_f = d3[i] as f64;
            let vth1_f = v_th1[i] as f64;
            let vr1_f = v_r1[i] as f64;
            let vth2_f = v_th2[i] as f64;
            let vr2_f = v_r2[i] as f64;
            let vth3_f = v_th3[i] as f64;
            let vr3_f = v_r3[i] as f64;

            let dth_dd1_f = dt_f * gain_f / (r1_f + eps_f);
            let dth_dd2_f = dt_f * gain_f / (r2_f + eps_f);
            let dth_dd3_f = dt_f * gain_f / (r3_f + eps_f);
            let rce1 = r_clamp_effect(pre_r1[i], d1[i], dt, gain, alpha) as f64;
            let rce2 = r_clamp_effect(pre_r2[i], d2[i], dt, gain, alpha) as f64;
            let rce3 = r_clamp_effect(pre_r3[i], d3[i], dt, gain, alpha) as f64;
            grad_drives[d_off + i] = (vth1_f * dth_dd1_f + vr1_f * dt_f * gain_f * rce1) as f32;
            grad_drives[d_off + n + i] = (vth2_f * dth_dd2_f + vr2_f * dt_f * gain_f * rce2) as f32;
            grad_drives[d_off + 2 * n + i] =
                (vth3_f * dth_dd3_f + vr3_f * dt_f * gain_f * rce3) as f32;

            let dcpl1 = -(w12_f * g2_f * (c1_f * r2c2_f + s1_f * r2s2_f)
                + w13_f * g3_f * (c1_f * r3c3_f + s1_f * r3s3_f));
            let dcpl2 = -(w12_f * g1_f * (c2_f * r1c1_f + s2_f * r1s1_f)
                + w23_f * g3_f * (c2_f * r3c3_f + s2_f * r3s3_f));
            let dcpl3 = -(w13_f * g1_f * (c3_f * r1c1_f + s3_f * r1s1_f)
                + w23_f * g2_f * (c3_f * r2c2_f + s3_f * r2s2_f));

            let vt1_self = vth1_f * (1.0 + dt_f * dcpl1);
            let vt2_self = vth2_f * (1.0 + dt_f * dcpl2);
            let vt3_self = vth3_f * (1.0 + dt_f * dcpl3);

            let dg1 = pp1_f * (sc1_f * c1_f - cc1_f * s1_f);
            let vt1_cross = dt_f
                * (r1_f
                    * (w12_f * g1_f * (c1_f * svc2_f + s1_f * svs2_f)
                        + w13_f * g1_f * (c1_f * svc3_f + s1_f * svs3_f))
                    + dg1 * cg1_f);
            let dg2 = pp2_f * (sc2_f * c2_f - cc2_f * s2_f);
            let vt2_cross = dt_f
                * (r2_f
                    * (w12_f * g2_f * (c2_f * svc1_f + s2_f * svs1_f)
                        + w23_f * g2_f * (c2_f * svc3_f + s2_f * svs3_f))
                    + dg2 * cg2_f);
            let dg3 = pp3_f * (sc3_f * c3_f - cc3_f * s3_f);
            let vt3_cross = dt_f
                * (r3_f
                    * (w13_f * g3_f * (c3_f * svc1_f + s3_f * svs1_f)
                        + w23_f * g3_f * (c3_f * svc2_f + s3_f * svs2_f))
                    + dg3 * cg3_f);

            let rg1 = r_factor_gate(pre_r1[i], d1[i], dt, gain, alpha) as f64;
            let r_factor1 = 1.0 + dt_f * (alpha_f - 3.0 * r1_f * r1_f);
            let vr1_out = vr1_f * r_factor1 * rg1
                + vth1_f * (-dt_f * gain_f * d1v_f / (r1_f + eps_f).powi(2));

            let rg2 = r_factor_gate(pre_r2[i], d2[i], dt, gain, alpha) as f64;
            let r_factor2 = 1.0 + dt_f * (alpha_f - 3.0 * r2_f * r2_f);
            let vr2_out = vr2_f * r_factor2 * rg2
                + vth2_f * (-dt_f * gain_f * d2v_f / (r2_f + eps_f).powi(2));

            let rg3 = r_factor_gate(pre_r3[i], d3[i], dt, gain, alpha) as f64;
            let r_factor3 = 1.0 + dt_f * (alpha_f - 3.0 * r3_f * r3_f);
            let vr3_out = vr3_f * r_factor3 * rg3
                + vth3_f * (-dt_f * gain_f * d3v_f / (r3_f + eps_f).powi(2));

            v_th1[i] = (vt1_self + vt1_cross) as f32;
            v_th2[i] = (vt2_self + vt2_cross) as f32;
            v_th3[i] = (vt3_self + vt3_cross) as f32;
            v_r1[i] = vr1_out as f32;
            v_r2[i] = vr2_out as f32;
            v_r3[i] = vr3_out as f32;
        }
    }

    // ── Velocity correction ──
    if step == n_steps - 1 {
        let n_chunks = n / 8;
        for c in 0..n_chunks {
            let i = c * 8;
            st(v_th1, i, ld(v_th1, i) - ld(grad_pst, 6 * n + i));
            st(v_th2, i, ld(v_th2, i) - ld(grad_pst, 7 * n + i));
            st(v_th3, i, ld(v_th3, i) - ld(grad_pst, 8 * n + i));
        }
        for j in 0..(n % 8) {
            let i = n_chunks * 8 + j;
            v_th1[i] -= grad_pst[6 * n + i];
            v_th2[i] -= grad_pst[7 * n + i];
            v_th3[i] -= grad_pst[8 * n + i];
        }
    }
}

// ─── Core BPTT loop ──────────────────────────────────────

/// Zero-allocation BPTT loop with SIMD.
///
/// `ws` must have sin/cos + grad_drives buffers sized appropriately.
/// `v_th*`/`v_r*` are **in/out**: pre-initialized with PST Jacobian v
/// (from `compute_pst_jacobian_v_into`), mutated in-place.
pub fn bptt_loop_into(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    per_step_grad_pst: Option<&[f32]>,
    ws: &mut BpttWorkspace,
) {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let dt = traj.dt;
    let gain = traj.drive_gain;
    let alpha = traj.alpha;
    let pst_dim = pst_linear_dim(n);
    let nf = n as f32;

    let v_th1 = &mut ws.v_th1[..n];
    let v_th2 = &mut ws.v_th2[..n];
    let v_th3 = &mut ws.v_th3[..n];
    let v_r1 = &mut ws.v_r1[..n];
    let v_r2 = &mut ws.v_r2[..n];
    let v_r3 = &mut ws.v_r3[..n];

    let sin1 = &mut ws.sin1[..n];
    let cos1 = &mut ws.cos1[..n];
    let sin2 = &mut ws.sin2[..n];
    let cos2 = &mut ws.cos2[..n];
    let sin3 = &mut ws.sin3[..n];
    let cos3 = &mut ws.cos3[..n];

    for step in (0..n_steps).rev() {
        // Read from flat trajectory
        let pre_th1 = step_slice(&traj.flat, OFF_TH1, n, step);
        let pre_th2 = step_slice(&traj.flat, off_th2(n_steps, n), n, step);
        let pre_th3 = step_slice(&traj.flat, off_th3(n_steps, n), n, step);
        let pre_r1 = step_slice(&traj.flat, off_r1(n_steps, n), n, step);
        let pre_r2 = step_slice(&traj.flat, off_r2(n_steps, n), n, step);
        let pre_r3 = step_slice(&traj.flat, off_r3(n_steps, n), n, step);
        let d1 = step_slice(&traj.flat, off_drive1(n_steps, n), n, step);
        let d2 = step_slice(&traj.flat, off_drive2(n_steps, n), n, step);
        let d3 = step_slice(&traj.flat, off_drive3(n_steps, n), n, step);

        let d_off = step * 3 * n;

        // ── Pass 1: sin/cos + reductions ──
        let (r1s1, r1c1, sin1_sum, cos1_sum, svs1, svc1) =
            simd_pass1_pop(n, pre_th1, pre_r1, v_th1, sin1, cos1);
        let (r2s2, r2c2, sin2_sum, cos2_sum, svs2, svc2) =
            simd_pass1_pop(n, pre_th2, pre_r2, v_th2, sin2, cos2);
        let (r3s3, r3c3, sin3_sum, cos3_sum, svs3, svc3) =
            simd_pass1_pop(n, pre_th3, pre_r3, v_th3, sin3, cos3);

        // Phase gates (via t3::phase_gate — f64 accumulator, no Kahan)
        let g1 = phase_gate_from_sincos(sin1, cos1, n, 0.5);
        let g2 = phase_gate_from_sincos(sin2, cos2, n, 1.0);
        let g3 = phase_gate_from_sincos(sin3, cos3, n, 1.12);

        let w12 = traj.w12 / nf;
        let w13 = traj.w13 / nf;
        let w23 = traj.w23 / nf;

        // ── Per-step PST gradient (SPIGRAD) — zero-alloc via PstJacobianBuf ──
        if let Some(ps) = per_step_grad_pst {
            let base = step * pst_dim;
            if (base..base + pst_dim).any(|k| ps[k] != 0.0) {
                let (pth1, pr1, pth2, pr2, pth3, pr3) = if step == n_steps - 1 {
                    (
                        &traj.th1_final[..],
                        &traj.r1_final[..],
                        &traj.th2_final[..],
                        &traj.r2_final[..],
                        &traj.th3_final[..],
                        &traj.r3_final[..],
                    )
                } else {
                    let nxt = step + 1;
                    (
                        step_slice(&traj.flat, OFF_TH1, n, nxt),
                        step_slice(&traj.flat, off_r1(n_steps, n), n, nxt),
                        step_slice(&traj.flat, off_th2(n_steps, n), n, nxt),
                        step_slice(&traj.flat, off_r2(n_steps, n), n, nxt),
                        step_slice(&traj.flat, off_th3(n_steps, n), n, nxt),
                        step_slice(&traj.flat, off_r3(n_steps, n), n, nxt),
                    )
                };
                pst_jacobian_v_jit(
                    pth1,
                    pr1,
                    pth2,
                    pr2,
                    pth3,
                    pr3,
                    n,
                    &ps[base..base + pst_dim],
                    &mut ws.pst_v,
                );
                for i in 0..n {
                    v_th1[i] += ws.pst_v.v_th1[i];
                    v_th2[i] += ws.pst_v.v_th2[i];
                    v_th3[i] += ws.pst_v.v_th3[i];
                    v_r1[i] += ws.pst_v.v_r1[i];
                    v_r2[i] += ws.pst_v.v_r2[i];
                    v_r3[i] += ws.pst_v.v_r3[i];
                }
            }
        }

        // Phase gate derivative pre-factors
        let r1_op = (sin1_sum * sin1_sum + cos1_sum * cos1_sum).sqrt() / nf;
        let r2_op = (sin2_sum * sin2_sum + cos2_sum * cos2_sum).sqrt() / nf;
        let r3_op = (sin3_sum * sin3_sum + cos3_sum * cos3_sum).sqrt() / nf;
        let pg_pre1 = if r1_op > 1e-10 {
            PG_GAIN * g1 * (1.0 - g1) / (nf * nf * r1_op)
        } else {
            0.0
        };
        let pg_pre2 = if r2_op > 1e-10 {
            PG_GAIN * g2 * (1.0 - g2) / (nf * nf * r2_op)
        } else {
            0.0
        };
        let pg_pre3 = if r3_op > 1e-10 {
            PG_GAIN * g3 * (1.0 - g3) / (nf * nf * r3_op)
        } else {
            0.0
        };
        let cross_gate1 = w12 * (r1s1 * svc2 - r1c1 * svs2) + w13 * (r1s1 * svc3 - r1c1 * svs3);
        let cross_gate2 = w12 * (r2s2 * svc1 - r2c2 * svs1) + w23 * (r2s2 * svc3 - r2c2 * svs3);
        let cross_gate3 = w13 * (r3s3 * svc1 - r3c3 * svs1) + w23 * (r3s3 * svc2 - r3c3 * svs2);

        // ── Pass 2: SIMD arithmetic ──
        simd_pass2(
            n,
            dt,
            gain,
            alpha,
            pre_r1,
            d1,
            pre_r2,
            d2,
            pre_r3,
            d3,
            sin1,
            cos1,
            sin2,
            cos2,
            sin3,
            cos3,
            r1s1,
            r1c1,
            r2s2,
            r2c2,
            r3s3,
            r3c3,
            svs1,
            svc1,
            svs2,
            svc2,
            svs3,
            svc3,
            sin1_sum,
            cos1_sum,
            sin2_sum,
            cos2_sum,
            sin3_sum,
            cos3_sum,
            g1,
            g2,
            g3,
            pg_pre1,
            pg_pre2,
            pg_pre3,
            cross_gate1,
            cross_gate2,
            cross_gate3,
            w12,
            w13,
            w23,
            &mut ws.grad_drives[..],
            d_off,
            v_th1,
            v_th2,
            v_th3,
            v_r1,
            v_r2,
            v_r3,
            step,
            n_steps,
            grad_pst,
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  Online BPTT (Truncated) — sliding window с hot-start
// ═══════════════════════════════════════════════════════════

use crate::spike::trajectory::copy_slice;

/// Truncated BPTT — онлайн-обучение через sliding window.
///
/// ## Usage
///
/// ```ignore
/// let mut tbptt = TruncatedBptt::new(
///     n_osc, window_size, dt, drive_gain, alpha, 0.1,
///     0.5, 3.0, 0.3, 0.15, 0.08,
/// );
/// for step in 0..n_total {
///     let drive = &drives[step * 3 * n..(step+1) * 3 * n];
///     tbptt.step(drive);
///     if tbptt.ready() {
///         tbptt.backward(&grad_pst);
///         let grad = tbptt.gradients();
///         // apply gradient to model weights
///     }
/// }
/// ```
pub struct TruncatedBptt {
    pub n: usize,
    pub window_size: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,

    /// Internal TriOscillator for forward stepping
    pub tri: crate::t3::TriOscillator<crate::t3::backend::Avx2Backend>,
    /// Sliding window trajectory (circular — всегда window_size шагов)
    pub traj: T3SpikeTrajectory,
    /// BPTT workspace
    pub ws: BpttWorkspace,
    /// Hot-start recurrent state (наследуется между windows)
    pub v_th1: Vec<f32>,
    pub v_th2: Vec<f32>,
    pub v_th3: Vec<f32>,
    pub v_r1: Vec<f32>,
    pub v_r2: Vec<f32>,
    pub v_r3: Vec<f32>,
    /// Accumulated PST Jacobian v
    pub pst_v: PstGradients,
    /// Счётчик обработанных шагов
    pub n_processed: usize,
    /// Последний backward output ([window_size × 3n] grad_drives)
    pub grad_buf: Vec<f32>,
}

impl TruncatedBptt {
    /// Create new TruncatedBptt with window_size шагов.
    ///
    /// Память: O(window_size · n) для trajectory + O(n) для workspace.
    pub fn new(
        n: usize,
        window_size: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        epsilon: f32,
        freq_min: f32,
        freq_max: f32,
        w12: f32,
        w13: f32,
        w23: f32,
    ) -> Self {
        let mut tri =
            crate::t3::TriOscillator::new(n, freq_min, freq_max, crate::t3::backend::Avx2Backend);
        tri.drive_gain = drive_gain;
        tri.alpha = alpha;
        tri.epsilon = epsilon;
        tri.set_uniform_coupling(w12, w13, w23);

        let traj = T3SpikeTrajectory::new(
            n,
            window_size,
            dt,
            drive_gain,
            alpha,
            epsilon,
            freq_min,
            freq_max,
            w12,
            w13,
            w23,
        );

        Self {
            n,
            window_size,
            dt,
            drive_gain,
            alpha,
            epsilon,
            freq_min,
            freq_max,
            w12,
            w13,
            w23,
            tri,
            traj,
            ws: BpttWorkspace::new(n, window_size),
            v_th1: vec![0.0; n],
            v_th2: vec![0.0; n],
            v_th3: vec![0.0; n],
            v_r1: vec![0.0; n],
            v_r2: vec![0.0; n],
            v_r3: vec![0.0; n],
            pst_v: PstGradients::new(n),
            n_processed: 0,
            grad_buf: vec![0.0; window_size * 3 * n],
        }
    }

    /// Forward one step: drive → oscillator, сохранить pre-state в траекторию.
    ///
    /// Траектория заполняется циклически: после window_size шагов новые данные
    /// перезаписывают старые.
    pub fn step(&mut self, d1: &[f32], d2: &[f32], d3: &[f32]) {
        let step = self.n_processed % self.window_size;
        let n = self.n;
        let n_steps = self.window_size;

        // Save pre-state
        copy_slice(
            step_slice_mut(&mut self.traj.flat, OFF_TH1, n, step),
            &self.tri.th1[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_th2(n_steps, n), n, step),
            &self.tri.th2[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_th3(n_steps, n), n, step),
            &self.tri.th3[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_r1(n_steps, n), n, step),
            &self.tri.r1[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_r2(n_steps, n), n, step),
            &self.tri.r2[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_r3(n_steps, n), n, step),
            &self.tri.r3[..n],
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_drive1(n_steps, n), n, step),
            d1,
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_drive2(n_steps, n), n, step),
            d2,
        );
        copy_slice(
            step_slice_mut(&mut self.traj.flat, off_drive3(n_steps, n), n, step),
            d3,
        );

        // Apply drives and step
        self.tri.drive1[..n].copy_from_slice(d1);
        self.tri.drive2[..n].copy_from_slice(d2);
        self.tri.drive3[..n].copy_from_slice(d3);
        self.tri.step(self.dt);

        self.n_processed += 1;
    }

    /// Достаточно ли шагов накоплено для backward?
    pub fn ready(&self) -> bool {
        self.n_processed >= self.window_size
    }

    /// Backward pass на текущем окне с hot-start.
    ///
    /// 1. Копирует финальное состояние триа в траекторию
    /// 2. PST Jacobian на последнем шаге + hot-start
    /// 3. BPTT loop + сохранение recurrent state на границе
    pub fn backward(&mut self, grad_pst: &[f32]) {
        let n = self.n;
        let n_steps = self.window_size;

        // Копируем финальное состояние
        self.traj.th1_final.copy_from_slice(&self.tri.th1[..n]);
        self.traj.th2_final.copy_from_slice(&self.tri.th2[..n]);
        self.traj.th3_final.copy_from_slice(&self.tri.th3[..n]);
        self.traj.r1_final.copy_from_slice(&self.tri.r1[..n]);
        self.traj.r2_final.copy_from_slice(&self.tri.r2[..n]);
        self.traj.r3_final.copy_from_slice(&self.tri.r3[..n]);

        // PST Jacobian (JIT) на последнем шаге
        crate::spike::pst_jacobian::pst_jacobian_v_jit(
            &self.traj.th1_final,
            &self.traj.r1_final,
            &self.traj.th2_final,
            &self.traj.r2_final,
            &self.traj.th3_final,
            &self.traj.r3_final,
            n,
            grad_pst,
            &mut self.pst_v,
        );

        // Hot-start: v_th/v_r = pst_v (TRUNCATED BPTT: нет state propagation между окнами)
        // При full BPTT здесь нужно + previous recurrent state, но TBPTT обрезает
        // градиент на границе окна для streaming-сценариев.
        self.ws.v_th1[..n].copy_from_slice(&self.pst_v.v_th1[..n]);
        self.ws.v_th2[..n].copy_from_slice(&self.pst_v.v_th2[..n]);
        self.ws.v_th3[..n].copy_from_slice(&self.pst_v.v_th3[..n]);
        self.ws.v_r1[..n].copy_from_slice(&self.pst_v.v_r1[..n]);
        self.ws.v_r2[..n].copy_from_slice(&self.pst_v.v_r2[..n]);
        self.ws.v_r3[..n].copy_from_slice(&self.pst_v.v_r3[..n]);

        // BPTT
        bptt_loop_into(grad_pst, &self.traj, None, &mut self.ws);

        // Копируем результат
        self.grad_buf
            .copy_from_slice(&self.ws.grad_drives[..n_steps * 3 * n]);
    }

    /// Последний backward результат: [window_size × 3n] grad_drives.
    pub fn gradients(&self) -> &[f32] {
        &self.grad_buf
    }

    /// Сбросить состояние (обнулить recurrent state, начать заново).
    pub fn reset(&mut self) {
        self.n_processed = 0;
        self.v_th1.fill(0.0);
        self.v_th2.fill(0.0);
        self.v_th3.fill(0.0);
        self.v_r1.fill(0.0);
        self.v_r2.fill(0.0);
        self.v_r3.fill(0.0);
        self.tri.reset();
    }
}

// ─── Backward-compatible wrappers ────────────────────────

/// Allocating wrapper — use `t3spike_backward_per_step_with_ws` for zero-alloc.
#[deprecated(since = "0.2.0", note = "use t3spike_backward_per_step_with_ws")]
pub fn t3spike_backward(grad_pst: &[f32], traj: &T3SpikeTrajectory) -> Vec<f32> {
    t3spike_backward_per_step(grad_pst, traj, None)
}

/// Allocating wrapper — use `t3spike_backward_per_step_with_ws` for zero-alloc.
#[deprecated(since = "0.2.0", note = "use t3spike_backward_per_step_with_ws")]
pub fn t3spike_backward_per_step(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    per_step_grad_pst: Option<&[f32]>,
) -> Vec<f32> {
    let mut ws = BpttWorkspace::new(traj.n_osc, traj.n_steps);
    t3spike_backward_per_step_with_ws(grad_pst, traj, per_step_grad_pst, &mut ws);
    ws.grad_drives
}

/// Zero-alloc BPTT with external workspace.
///
/// `ws.v_th*`/`ws.v_r*` — будут перезаписаны (используются как временные буферы).
/// `ws.grad_drives` — результат.
pub fn t3spike_backward_per_step_with_ws(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    per_step_grad_pst: Option<&[f32]>,
    ws: &mut BpttWorkspace,
) {
    let n = traj.n_osc;
    let _pst_dim = pst_linear_dim(n);

    // K1: v = J^T · grad_pst — JIT, без матрицы
    pst_jacobian_v_jit(
        &traj.th1_final,
        &traj.r1_final,
        &traj.th2_final,
        &traj.r2_final,
        &traj.th3_final,
        &traj.r3_final,
        n,
        grad_pst,
        &mut ws.pst_v,
    );

    ws.resize(n, traj.n_steps);
    // pst_v already filled by pst_jacobian_v_jit
    ws.pst_v.copy_to_slices(
        &mut ws.v_th1,
        &mut ws.v_th2,
        &mut ws.v_th3,
        &mut ws.v_r1,
        &mut ws.v_r2,
        &mut ws.v_r3,
    );

    bptt_loop_into(grad_pst, traj, per_step_grad_pst, ws);
}
