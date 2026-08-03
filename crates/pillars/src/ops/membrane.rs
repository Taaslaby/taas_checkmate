//! # Membrane — Membrane Potential Integration
//!
//! ## `leak_integrate`
//! ```text
//! v += (v_rest - v + I_syn) · inv_tau · dt
//! ```
//! Forward Euler for LIF-like membrane dynamics.
//! All tau values are `inv_tau = 1/τ` — zero divisions in hot path.
//!
//! ## `membrane_depolarize`
//! ```text
//! dv = (I_inj - ΣI_ion) · inv_Cm · dt    (NEURON convention)
//! v += dv
//! ```
//! All ionic currents subtracted from I_inj.
//! Uses FMA: `g.mul_add(v, -g · e_rev)` — one instruction, not two.

use crate::core::lane::SimdLane;

/// Forward Euler leak integration.
///
/// ```text
/// dv = (v_rest - v + I_syn) · inv_tau · dt
/// ```
#[inline(always)]
pub fn leak_integrate<S: SimdLane>(v: S, v_rest: S, i_syn: S, inv_tau: S, dt: S) -> S {
    v + (v_rest - v + i_syn) * inv_tau * dt
}

/// Membrane depolarization with ionic currents.
///
/// ```text
/// dv = (I_inj - I_ion) · inv_Cm · dt
/// ```
/// Subtracts total ionic current from injected current (NEURON convention).
#[inline(always)]
pub fn membrane_depolarize<S: SimdLane>(v: S, i_inj: S, i_ion: S, inv_cm: S, dt: S) -> S {
    v + (i_inj - i_ion) * inv_cm * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_leak_integrate_resting() {
        // v = v_rest → no change
        let v = f32x8::splat(-70.0);
        let vr = f32x8::splat(-70.0);
        let i = f32x8::splat(0.0);
        let inv_tau = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let v_new = leak_integrate(v, vr, i, inv_tau, dt);
        for vi in v_new.to_array() {
            assert!((vi - (-70.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_leak_integrate_depolarize() {
        let v = f32x8::splat(-70.0);
        let vr = f32x8::splat(-70.0);
        let i = f32x8::splat(10.0);
        let inv_tau = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let v_new = leak_integrate(v, vr, i, inv_tau, dt);
        // v_new = -70 + (70-70+10) * 0.1 * 1 = -69
        assert!((v_new.to_array()[0] - (-69.0)).abs() < 1e-6);
    }

    #[test]
    fn test_membrane_depolarize() {
        let v = f32x8::splat(-65.0);
        let i_inj = f32x8::splat(10.0);
        let i_ion = f32x8::splat(5.0);
        let inv_cm = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let v_new = membrane_depolarize(v, i_inj, i_ion, inv_cm, dt);
        // v_new = -65 + (10-5) * 0.1 * 1 = -64.5
        assert!((v_new.to_array()[0] - (-64.5)).abs() < 1e-6);
    }
}
