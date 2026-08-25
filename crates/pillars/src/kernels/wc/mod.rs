//! # Wilson-Cowan (WC) batch kernel — canonical port of `kernels::wc_step`
//!
//! Population-level rate dynamics (Wilson & Cowan 1972):
//!
//! ```text
//! τ_E·dE/dt = −E + σ(w_EE·E − w_EI·I + I_E)     σ(x) = 1/(1 + exp(−β·x))
//! τ_I·dI/dt = −I + σ(w_IE·E − w_II·I + I_I)
//! ```
//!
//! β is a **gain** (input multiplied by β), NOT a width — the legacy batch
//! convention. (Zoo `WCModel` deliberately uses the width form `exp(−x/β)`;
//! the two are bridged by passing `1/β` as width.) The batch kernel writes the
//! continuous sigmoid rates (0..1) into the "spikes" output buffers.
//!
//! Bit-identical to the legacy `kernels::wc_step` (guarded by
//! `crates/kernels/tests/wc_parity.rs`): SIMD lanes clamp the sigmoid argument
//! to ±10 and use the minimax polynomial `fast_exp_poly_simd`; the scalar tail
//! uses plain `f32::exp` **without** clamping — the documented legacy
//! SIMD/tail divergence is preserved.

use wide::f32x8;
use wide::CmpGt;

use crate::core::util::{load8, store8};
use crate::math::fast_exp::fast_exp_poly_simd;

/// Step `n` Wilson-Cowan populations (SoA, column-major params `8 × n`:
/// `[dt, τ_E, τ_I, w_EE, w_EI, w_IE, w_II, β]`).
///
/// `spikes_e` / `spikes_i` receive the continuous sigmoid rates (0..1).
#[allow(clippy::too_many_arguments)]
pub fn wc_step(
    e: &mut [f32],
    i: &mut [f32],
    i_e: &[f32],
    i_i: &[f32],
    spikes_e: &mut [f32],
    spikes_i: &mut [f32],
    params: &[f32],
) {
    let n = e.len();
    assert_eq!(i.len(), n, "wc_step: i length mismatch");
    assert_eq!(i_e.len(), n, "wc_step: i_e length mismatch");
    assert_eq!(i_i.len(), n, "wc_step: i_i length mismatch");
    assert_eq!(spikes_e.len(), n, "wc_step: spikes_e length mismatch");
    assert_eq!(spikes_i.len(), n, "wc_step: spikes_i length mismatch");
    assert_eq!(params.len(), n * 8, "wc_step: params length mismatch");

    let dt_val = params[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut ee = load8(e.as_ptr(), off);
            let mut ii = load8(i.as_ptr(), off);
            let ie_ext = load8(i_e.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));
            let ii_ext = load8(i_i.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));

            let tau_e = load8(params.as_ptr().add(n), off);
            let tau_i = load8(params.as_ptr().add(2 * n), off);
            let w_ee = load8(params.as_ptr().add(3 * n), off);
            let w_ei = load8(params.as_ptr().add(4 * n), off);
            let w_ie = load8(params.as_ptr().add(5 * n), off);
            let w_ii = load8(params.as_ptr().add(6 * n), off);
            let beta = load8(params.as_ptr().add(7 * n), off);

            let input_e = w_ee * ee - w_ei * ii + ie_ext;
            let input_i = w_ie * ee - w_ii * ii + ii_ext;

            let f_e = sigmoid_simd(input_e, beta);
            let f_i = sigmoid_simd(input_i, beta);

            // τ·dE/dt = −E + f(input) → E += (−E + f)/τ · dt (Forward Euler)
            let mask_te = tau_e.cmp_gt(zero);
            let inv_tau_e = mask_te.blend(f32x8::splat(1.0) / tau_e.max(f32x8::splat(1e-10)), zero);
            let de = (-ee + f_e) * inv_tau_e * dt_s;
            ee = mask_te.blend(ee + de, ee);

            let mask_ti = tau_i.cmp_gt(zero);
            let inv_tau_i = mask_ti.blend(f32x8::splat(1.0) / tau_i.max(f32x8::splat(1e-10)), zero);
            let di = (-ii + f_i) * inv_tau_i * dt_s;
            ii = mask_ti.blend(ii + di, ii);

            store8(e.as_mut_ptr(), off, ee);
            store8(i.as_mut_ptr(), off, ii);
            store8(spikes_e.as_mut_ptr(), off, f_e);
            store8(spikes_i.as_mut_ptr(), off, f_i);
        }
    }

    // Scalar tail — legacy tail: plain f32::exp, NO ±10 clamp.
    for idx in n8..n {
        let tau_e = params[n + idx];
        let tau_i = params[2 * n + idx];
        let w_ee = params[3 * n + idx];
        let w_ei = params[4 * n + idx];
        let w_ie = params[5 * n + idx];
        let w_ii = params[6 * n + idx];
        let beta = params[7 * n + idx];

        let input_e = w_ee * e[idx] - w_ei * i[idx] + i_e[idx].clamp(-100.0, 100.0);
        let input_i = w_ie * e[idx] - w_ii * i[idx] + i_i[idx].clamp(-100.0, 100.0);

        let f_e = 1.0 / (1.0 + (-beta * input_e).exp());
        let f_i = 1.0 / (1.0 + (-beta * input_i).exp());

        if tau_e > 0.0 {
            e[idx] += (-e[idx] + f_e) / tau_e * dt_val;
        }
        if tau_i > 0.0 {
            i[idx] += (-i[idx] + f_i) / tau_i * dt_val;
        }

        spikes_e[idx] = f_e;
        spikes_i[idx] = f_i;
    }
}

/// σ(x) = 1/(1 + exp(−β·x)) with the legacy SIMD ±10 argument clamp.
#[inline(always)]
fn sigmoid_simd(x: f32x8, beta: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let neg = (-x * beta).max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + fast_exp_poly_simd(neg))
}

#[cfg(test)]
mod tests;
