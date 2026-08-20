use crate::core::util::{load8, store8};
use wide::f32x8;

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
