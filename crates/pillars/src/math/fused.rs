//! Fused operations — combine multiple neural primitives into a single SIMD pass.
//!
//! Instead of:
//! ```text
//! u_inf = sigmoid(v)          // 1 fast_exp
//! u     = gate_integrate(u)   // 1 fast_exp ← can be reused!
//! i_M   = ohmic(v) * u        // 1 multiply
//! ```
//!
//! One fused call:
//! ```text
//! (u_new, i_M) = sigmoid_gate_ohmic(v, u)  // 1 fast_exp, fused
//! ```

use wide::f32x8;
use crate::math::gate::gate_integrate;
use crate::math::ohmic::ohmic;
use crate::math::sigmoid::sigmoid;

/// Fused: sigmoid → gate_integrate → ohmic (HAAP M-current).
///
/// Computes in one pass — no division, uses `inv_tau`.
/// - `u_inf = sigmoid(v, v_half_u, k_u)`
/// - `u_new = gate_integrate(u, u_inf, inv_tau_u, dt)`
/// - `i_M = ohmic(g_M, E_K, v) * u_new`
#[allow(non_snake_case)]
#[inline(always)]
pub fn sigmoid_gate_ohmic(
    v: f32x8, u: f32x8,
    g_M: f32, E_K: f32,
    v_half_u: f32, k_u: f32, inv_tau_u: f32, dt: f32,
) -> (f32x8, f32x8) {
    let u_inf = sigmoid(v, v_half_u, k_u);
    let u_new = gate_integrate(u, u_inf, inv_tau_u, dt);
    let i_M = ohmic(g_M, E_K, v) * u_new;
    (u_new, i_M)
}

/// Full HAAP step: u + h + I_ion in a single SIMD pass.
///
/// 1. u_inf → u_new → i_M (sigmoid + gate + ohmic)
/// 2. h_inf → h_new → i_h (sigmoid + gate + ohmic)
/// 3. i_L + i_sum
#[allow(non_snake_case)]
#[inline(always)]
pub fn haap_step_fused(
    v: f32x8, u: f32x8, h: f32x8,
    g_M: f32, E_K: f32, v_half_u: f32, k_u: f32, inv_tau_u: f32,
    g_h: f32, E_h: f32, v_half_h: f32, k_h: f32, inv_tau_h: f32,
    g_L: f32, E_L: f32, dt: f32,
) -> (f32x8, f32x8, f32x8) {
    let (u_new, i_M) = sigmoid_gate_ohmic(v, u, g_M, E_K, v_half_u, k_u, inv_tau_u, dt);
    let h_inf = sigmoid(v, v_half_h, k_h);
    let h_new = gate_integrate(h, h_inf, inv_tau_h, dt);
    let i_h = ohmic(g_h, E_h, v) * h_new;
    let i_L = ohmic(g_L, E_L, v);
    (u_new, h_new, i_L + i_M + i_h)
}

/// Total synaptic current from four receptor types.
#[inline(always)]
pub fn synaptic_sum(
    v: f32x8,
    g_ampa: f32x8, g_nmda: f32x8, g_gaba: f32x8, g_glucl: f32x8,
) -> f32x8 {
    fn ohmic_simd(g: f32x8, e: f32, v: f32x8) -> f32x8 { g * (f32x8::splat(e) - v) }
    ohmic_simd(g_ampa, crate::E_NA, v)
        + ohmic_simd(g_nmda, crate::E_CA, v)
        + ohmic_simd(g_gaba, crate::E_CL, v)
        + ohmic_simd(g_glucl, crate::E_CL, v)
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use wide::f32x8;
    use crate::prelude::*;

    #[test]
    fn test_sigmoid_gate_ohmic_fused() {
        let v = f32x8::splat(-65.0);
        let u = f32x8::splat(0.5);
        let inv_tau_u = 1.0 / 100.0;
        let (_u_new, i_M) = sigmoid_gate_ohmic(v, u, 0.5, -80.0, -35.0, 5.0, inv_tau_u, 0.05);
        let a = i_M.to_array();
        // i_M should be negative (K+ efflux hyperpolarizes)
        assert!(a[0] < 0.0, "i_M should be negative, got {}", a[0]);
    }

    #[test]
    fn test_haap_step_fused() {
        let v = f32x8::splat(-65.0);
        let u = f32x8::splat(0.0);
        let h = f32x8::splat(0.0);
        let inv_tau_u = 1.0 / 100.0;
        let inv_tau_h = 1.0 / 200.0;
        let (_u2, _h2, i_sum) = haap_step_fused(
            v, u, h,
            0.5, -80.0, -35.0, 5.0, inv_tau_u,
            0.3, -30.0, -75.0, 6.0, inv_tau_h,
            0.1, -65.0, 0.05,
        );
        for i in 0..8 { assert!(i_sum.to_array()[i].is_finite()); }
    }

    #[test]
    fn test_synaptic_sum_fused() {
        let v = f32x8::splat(-65.0);
        let g_ampa = f32x8::splat(0.5);
        let g_nmda = f32x8::splat(0.3);
        let g_gaba = f32x8::splat(0.2);
        let g_glucl = f32x8::splat(0.1);
        let i = synaptic_sum(v, g_ampa, g_nmda, g_gaba, g_glucl);
        assert!(i.to_array()[0].is_finite());
    }
}
