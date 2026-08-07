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

#[cfg(test)]
mod tests {
    use super::*;

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
