//! # 🔄 Phase Dynamics — Kuramoto & T3 Oscillators
//!
//! `f32x8` kernels for coupled phase oscillators. Both standard Kuramoto
//! and Stuart-Landau amplitude dynamics (T3 Triple-Threshold Theorem).
//!
//! ## Models
//!
//! | Kernel | Equation | Type |
//! |--------|----------|------|
//! | [`kuramoto_prepare`] | `φ ← wrap(φ)` + sin/cos | Phase wrap + trig in one pass |
//! | [`kuramoto_update`] | `φ ← φ + dt·(ω + C)` | Standard Kuramoto |
//! | [`kuramoto_update_viscous`] | `φ ← φ + dt·(ω + C + η·(⟨φ⟩−φ))` | Viscous damping |
//! | [`t3_update`] | `r ← r + dt·((α−r²)r + G·D − δ·r)` | Stuart-Landau amplitude |
//! | | `θ ← θ + dt·(ω + C + G·D/(r+ε))` | Kuramoto phase (with drive) |
//!
//! ## Kuramoto vs T3
//!
//! ```text
//! Standard Kuramoto:    φ̇ = ω + Σ W·sin(φⱼ − φᵢ)
//! T3 (Stuart-Landau):  ṙ = (α − r²)r + G·D − δ·r   (amplitude envelope)
//!                      θ̇ = ω + C + G·D/(r+ε)       (phase with gain)
//! ```

use wide::f32x8;

use crate::core::util::{load8, store8};
use crate::kernels::{fast_sincos_simd, wrap_phase};
use core::f32::consts::TAU;

/// Wrap phases to [−π, π) and compute sin/cos in one pass.
/// (The SIMD wrap `φ − 2π·round(φ/2π)` and the scalar `wrap_phase` both land
/// in [−π, π) — the same range the Kuramoto steppers below document.)
///
/// SIMD path uses the minimax polynomial `fast_sincos_simd` (~1e-4 accuracy),
/// scalar tail uses libm — both on wrapped phases.
pub fn kuramoto_prepare(phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
    let n = phases.len();
    assert_eq!(
        cos_out.len(),
        n,
        "kuramoto_prepare: cos_out length mismatch"
    );
    assert_eq!(
        sin_out.len(),
        n,
        "kuramoto_prepare: sin_out length mismatch"
    );
    let tau_s = f32x8::splat(TAU);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let pv = load8(phases.as_ptr(), off);
            let wrapped = pv - tau_s * (pv / tau_s).round();
            let (si, co) = fast_sincos_simd(wrapped);
            store8(phases.as_mut_ptr(), off, wrapped);
            store8(cos_out.as_mut_ptr(), off, co);
            store8(sin_out.as_mut_ptr(), off, si);
        }
    }

    for i in n8..n {
        let phi = wrap_phase(phases[i]);
        phases[i] = phi;
        cos_out[i] = phi.cos();
        sin_out[i] = phi.sin();
    }
}

/// Euler-step Kuramoto phases: `φ += dt·(ω + coupling)`, wrapped to [-π, π).
pub fn kuramoto_update(phases: &mut [f32], omegas: &[f32], coupling: &[f32], dt: f32) {
    let n = phases.len();
    assert_eq!(omegas.len(), n, "kuramoto_update: omegas length mismatch");
    assert_eq!(
        coupling.len(),
        n,
        "kuramoto_update: coupling length mismatch"
    );
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut pv = load8(phases.as_ptr(), off);
            let ov = load8(omegas.as_ptr(), off);
            let cv = load8(coupling.as_ptr(), off);
            pv += dt_s * (ov + cv);
            pv = pv - tau_s * (pv / tau_s).round();
            store8(phases.as_mut_ptr(), off, pv);
        }
    }

    for i in n8..n {
        phases[i] = wrap_phase(phases[i] + dt * (omegas[i] + coupling[i]));
    }
}

/// Viscous Kuramoto update: `φ += dt·(ω + coupling + η·(mean(φ) − φ))`.
///
/// The viscous term η·∇²φ is approximated as η·(mean(φ) − φ) for fully
/// connected coupling. It smooths phase differences and suppresses chaotic
/// switching between synchronized modes. `η = 0` degrades to standard
/// [`kuramoto_update`].
///
/// Reference: Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence".
pub fn kuramoto_update_viscous(
    phases: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    eta: f32,
    dt: f32,
) {
    let n = phases.len();
    assert_eq!(omegas.len(), n, "kuramoto_viscous: omegas length mismatch");
    assert_eq!(
        coupling.len(),
        n,
        "kuramoto_viscous: coupling length mismatch"
    );

    if eta <= 0.0 {
        return kuramoto_update(phases, omegas, coupling, dt);
    }

    // 1. Mean phase (scalar, one pass)
    let mean_phase = phases.iter().sum::<f32>() / n as f32;

    // 2. SIMD viscous update
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);
    let eta_s = f32x8::splat(eta);
    let mean_s = f32x8::splat(mean_phase);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut pv = load8(phases.as_ptr(), off);
            let ov = load8(omegas.as_ptr(), off);
            let cv = load8(coupling.as_ptr(), off);
            let viscous = eta_s * (mean_s - pv);
            pv += dt_s * (ov + cv + viscous);
            pv = pv - tau_s * (pv / tau_s).round();
            store8(phases.as_mut_ptr(), off, pv);
        }
    }

    for i in n8..n {
        let viscous = eta * (mean_phase - phases[i]);
        phases[i] = wrap_phase(phases[i] + dt * (omegas[i] + coupling[i] + viscous));
    }
}

/// Euler-step phase (Kuramoto) + amplitude (Stuart-Landau).
///
/// ```text
/// r += dt · ((α − r²)·r + gain·drive − decay·r)   ← Stuart-Landau
/// θ += dt · (ω + coupling + gain·drive / (r + ε)) ← Kuramoto + drive
/// ```
///
/// Phase is updated FIRST using the OLD amplitude (matches ScalarBackend),
/// then amplitude. Amplitude clamped to [0.001, 10.0].
#[allow(clippy::too_many_arguments)]
pub fn t3_update(
    phases: &mut [f32],
    amplitudes: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    drives: &[f32],
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
) {
    let n = phases.len();
    assert_eq!(amplitudes.len(), n, "t3_update: amplitudes length mismatch");
    assert_eq!(omegas.len(), n, "t3_update: omegas length mismatch");
    assert_eq!(coupling.len(), n, "t3_update: coupling length mismatch");
    assert_eq!(drives.len(), n, "t3_update: drives length mismatch");
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);
    let decay_s = f32x8::splat(decay);
    let alpha_s = f32x8::splat(alpha);
    let gain_s = f32x8::splat(gain);
    let eps_s = f32x8::splat(0.1);
    let min_amp = f32x8::splat(0.001);
    let max_amp = f32x8::splat(10.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut pv = load8(phases.as_ptr(), off);
            let mut av = load8(amplitudes.as_ptr(), off);
            let ov = load8(omegas.as_ptr(), off);
            let cv = load8(coupling.as_ptr(), off);
            let dv = load8(drives.as_ptr(), off);

            // Kuramoto phase FIRST (uses OLD r)
            let drive_phase = gain_s * dv / (av + eps_s);
            pv += dt_s * (ov + cv + drive_phase);
            pv = pv - tau_s * (pv / tau_s).round();

            // Stuart-Landau amplitude SECOND
            let r2 = av * av;
            let r_dot = (alpha_s - r2) * av + gain_s * dv - decay_s * av;
            av += dt_s * r_dot;
            av = av.max(min_amp).min(max_amp);

            store8(phases.as_mut_ptr(), off, pv);
            store8(amplitudes.as_mut_ptr(), off, av);
        }
    }

    for i in n8..n {
        let r = amplitudes[i];
        let drive_phase = gain * drives[i] / (r + 0.1);
        phases[i] = wrap_phase(phases[i] + dt * (omegas[i] + coupling[i] + drive_phase));
        let r_dot = (alpha - r * r) * r + gain * drives[i] - decay * r;
        amplitudes[i] = (r + dt * r_dot).clamp(0.001, 10.0);
    }
}

/// Euler-step graph-coupled oscillator (GCO) — phase-only dynamics on the unit circle.
///
/// ```text
/// dv/dt = H_real − H_imag·speed + I_ext     (real part)
/// du/dt = H_imag + H_real·speed             (imag part)
/// (v, u) ← normalize to |v + iu| = 1
/// ```
///
/// `h_real` / `h_imag` are the Hamiltonian images (H·u, −H·v) computed
/// externally; `i_ext` is clamped to ±100. Canonical port of the legacy
/// `kernels::gco_step` — same operation order, bit-identical (guarded by
/// `crates/kernels/tests/gco_parity.rs`).
#[allow(clippy::too_many_arguments)]
pub fn gco_step(
    v: &mut [f32],
    u: &mut [f32],
    h_real: &[f32],
    h_imag: &[f32],
    i_ext: &[f32],
    speed: &[f32],
    dt: f32,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "gco_step: u length mismatch");
    assert_eq!(h_real.len(), n, "gco_step: h_real length mismatch");
    assert_eq!(h_imag.len(), n, "gco_step: h_imag length mismatch");
    assert_eq!(i_ext.len(), n, "gco_step: i_ext length mismatch");
    assert_eq!(speed.len(), n, "gco_step: speed length mismatch");

    let dt_s = f32x8::splat(dt);
    let min = f32x8::splat(-100.0);
    let max = f32x8::splat(100.0);
    let one = f32x8::splat(1.0);
    let eps = f32x8::splat(1e-9);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let hr = load8(h_real.as_ptr(), off);
            let hi = load8(h_imag.as_ptr(), off);
            let ie = load8(i_ext.as_ptr(), off).max(min).min(max);
            let sp = load8(speed.as_ptr(), off);

            // Same grouping as legacy: dv = (hr − hi·sp) + ie
            let dv = hr - hi * sp + ie;
            let du = hi + hr * sp;
            vv += dt_s * dv;
            uu += dt_s * du;

            // Normalize: keep on unit circle
            let nrm = (vv * vv + uu * uu + eps).sqrt();
            let inv = one / nrm;
            vv *= inv;
            uu *= inv;

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
        }
    }

    for i in n8..n {
        let dv = h_real[i] - h_imag[i] * speed[i] + i_ext[i].clamp(-100.0, 100.0);
        let du = h_imag[i] + h_real[i] * speed[i];
        v[i] += dt * dv;
        u[i] += dt * du;
        let nrm = (v[i] * v[i] + u[i] * u[i] + 1e-9).sqrt();
        let inv = 1.0 / nrm;
        v[i] *= inv;
        u[i] *= inv;
    }
}

// ─── Sparse/dense CSR Kuramoto & T3 steps───────
//
// Перенесено из `core::kuramoto`— rayon-параллельные шаги
// поверх `kuramoto_prepare` / `kuramoto_update` / `t3_update`.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[inline(always)]
fn safe_row(row_ptr: &[i32], idx: usize) -> usize {
    if idx >= row_ptr.len() {
        return 0;
    }
    let r = row_ptr[idx];
    if r < 0 {
        0
    } else {
        r as usize
    }
}

#[inline(always)]
fn safe_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() {
        return 0;
    }
    let c = col_indices[idx];
    if c < 0 || c as usize >= n {
        0
    } else {
        c as usize
    }
}

/// Full sparse Kuramoto step with rayon parallelization.
///
/// Computes coupling via CSR sparse matrix: `coupling[i] = Σⱼ w[i,j]·sin(θⱼ−θᵢ)`,
/// then updates phases via `θ += dt·(ω + coupling)`.
pub fn kuramoto_step_sparse(
    phases: &mut [f32],
    omegas: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    weights: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(
        n == omegas.len()
            && n + 1 == row_ptr.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        coupling_buf[i] = coupling_sum;
    }

    kuramoto_update(phases, omegas, coupling_buf, dt);
}

/// Dense Kuramoto step with rayon parallelization.
///
/// Same as `kuramoto_step_sparse` but with a full N×N coupling matrix.
pub fn kuramoto_step_dense(
    phases: &mut [f32],
    omegas: &[f32],
    coupling_matrix: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(
        n == omegas.len()
            && n * n == coupling_matrix.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let row_start = i * n;
        let mut coupling_sum = 0.0f32;
        for j in 0..n {
            let w = coupling_matrix[row_start + j];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let row_start = i * n;
        let mut coupling_sum = 0.0f32;
        for j in 0..n {
            let w = coupling_matrix[row_start + j];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        coupling_buf[i] = coupling_sum;
    }

    kuramoto_update(phases, omegas, coupling_buf, dt);
}

/// Full sparse T3 (Temporal Triplet) step with rayon parallelization.
///
/// Combines Kuramoto phase dynamics with Stuart-Landau amplitude evolution:
/// `r += dt·((α−r²)·r + gain·drive − decay·r)`
/// `θ += dt·(ω + coupling + gain·drive/(r+ε))`
pub fn t3_step_sparse(
    phases: &mut [f32],
    amplitudes: &mut [f32],
    omegas: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    weights: &[f32],
    drives: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
) {
    let n = phases.len();
    assert!(
        n == amplitudes.len()
            && n == omegas.len()
            && n + 1 == row_ptr.len()
            && n == drives.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            let a_j = amplitudes[j];
            coupling_sum += w * a_j * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            let a_j = amplitudes[j];
            coupling_sum += w * a_j * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        coupling_buf[i] = coupling_sum;
    }

    t3_update(
        phases,
        amplitudes,
        omegas,
        coupling_buf,
        drives,
        dt,
        decay,
        alpha,
        gain,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gco_step_normalizes_on_unit_circle() {
        let n = 16;
        let mut v: Vec<f32> = (0..n).map(|i| 0.3 + i as f32 * 0.1).collect();
        let mut u: Vec<f32> = (0..n).map(|i| -0.2 + i as f32 * 0.05).collect();
        let h_real = vec![1.0f32; n];
        let h_imag = vec![0.5f32; n];
        let i_ext = vec![0.2f32; n];
        let speed = vec![0.8f32; n];

        for _ in 0..1000 {
            gco_step(&mut v, &mut u, &h_real, &h_imag, &i_ext, &speed, 0.01);
        }
        for i in 0..n {
            let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
            assert!(
                (nrm - 1.0).abs() < 1e-4,
                "GCO[{i}] left the unit circle: |v+iu|={nrm}"
            );
            assert!(v[i].is_finite() && u[i].is_finite(), "GCO[{i}] non-finite");
        }
    }

    #[test]
    fn gco_step_clamps_input() {
        let n = 8;
        let mut v = vec![0.5f32; n];
        let mut u = vec![0.0f32; n];
        let h_real = vec![0.0f32; n];
        let h_imag = vec![0.0f32; n];
        // Хьюг — без клампа осциллятор улетел бы; проверяем конечность
        let i_ext = vec![1e6f32; n];
        let speed = vec![0.0f32; n];
        for _ in 0..10 {
            gco_step(&mut v, &mut u, &h_real, &h_imag, &i_ext, &speed, 0.01);
        }
        for i in 0..n {
            assert!(v[i].is_finite() && u[i].is_finite(), "GCO[{i}] non-finite");
            let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
            assert!((nrm - 1.0).abs() < 1e-3, "GCO[{i}] off-circle: {nrm}");
        }
    }

    #[test]
    fn kuramoto_prepare_basic() {
        let n = 16;
        let mut phases: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut cos_out = vec![0.0f32; n];
        let mut sin_out = vec![0.0f32; n];

        kuramoto_prepare(&mut phases, &mut cos_out, &mut sin_out);
        for i in 0..n {
            let expected_cos = phases[i].cos();
            let expected_sin = phases[i].sin();
            assert!((cos_out[i] - expected_cos).abs() < 5e-4, "cos[{i}]");
            assert!((sin_out[i] - expected_sin).abs() < 5e-4, "sin[{i}]");
        }
    }

    #[test]
    fn kuramoto_update_basic() {
        let n = 8;
        let mut phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        kuramoto_update(&mut phases, &omegas, &coupling, 0.01);
        for i in 0..n {
            assert!(
                (phases[i] - 0.01).abs() < 1e-4,
                "kuramoto_update[{i}]: {}",
                phases[i]
            );
        }
    }

    #[test]
    fn t3_update_basic() {
        let n = 16;
        let mut phases = vec![0.0f32; n];
        let mut amps = vec![1.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        let drives = vec![0.0f32; n];
        t3_update(
            &mut phases,
            &mut amps,
            &omegas,
            &coupling,
            &drives,
            0.01,
            0.0,
            1.0,
            10.0,
        );
        assert!(phases[0].abs() > 0.001, "t3_update should advance phases");
        for &r in &amps {
            assert!(
                (0.001..=10.0).contains(&r),
                "t3_update amp out of range: {r}"
            );
        }
    }

    #[test]
    fn kuramoto_viscous_equals_plain_when_eta_zero() {
        let n = 16;
        let mut p1: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut p2 = p1.clone();
        let omegas = vec![0.5f32; n];
        let coupling = vec![0.1f32; n];
        kuramoto_update(&mut p1, &omegas, &coupling, 0.01);
        kuramoto_update_viscous(&mut p2, &omegas, &coupling, 0.0, 0.01);
        for i in 0..n {
            assert!(
                (p1[i] - p2[i]).abs() < 1e-6,
                "eta=0 should match plain update: [{i}]"
            );
        }
    }
}
