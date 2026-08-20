use core::f32::consts::TAU;

use crate::core::util::{load8, store8};
use crate::kernels::wrap_phase;
use wide::f32x8;

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
