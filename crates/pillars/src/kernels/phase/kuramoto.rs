use core::f32::consts::TAU;

use crate::core::util::{load8, store8};
use crate::kernels::{fast_sincos_simd, wrap_phase};
use wide::f32x8;

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
