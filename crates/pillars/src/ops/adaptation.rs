//! # Adaptation — Spike-Frequency Adaptation (M-type, AHP)
//!
//! ## `integrate_u`
//! ```text
//! u' = u + dt · (a · (b · v - u)) · inv_tau
//! ```
//! Forward Euler for adaptation variable. Used in Izhikevich, HAAP, SAW.
//!
//! ## `integrate_ahp`
//! ```text
//! g' = g + dt · (-g) · inv_tau    (unbounded — no .min(1.0)!)
//! ```
//! Used in GLIF3, GLIF4 AHP conductance.
//!
//! ## `adex_exp`
//! ```text
//! exp_term = delta_t · exp((v - v_thresh) · inv_delta_t)
//! ```
//! AdEx exponential activation. Clamps exp arg to [-10, 10] for numerical safety.
//!
//! ## `ahp_current`
//! ```text
//! I_AHP = g · (V - E_AHP)
//! ```
//! Combined AHP current from conductance and voltage.

use crate::core::lane::SimdLane;

/// Forward Euler adaptation variable (Izhikevich-style): `u += dt · a · (b·v - u)`
#[inline(always)]
pub fn integrate_u<S: SimdLane>(u: S, v: S, a: S, b: S, dt: S) -> S {
    u + dt * a * (b * v - u)
}

/// Forward Euler after-hyperpolarization conductance: `g += dt · (-g) · inv_tau`
/// Unbounded — allows deep AHP in bursting regimes.
#[inline(always)]
pub fn integrate_ahp<S: SimdLane>(g: S, inv_tau: S, dt: S) -> S {
    g + dt * (-g) * inv_tau
}

/// AdEx exponential activation with numerical safety clamp.
///
/// ```text
/// exp_term = delta_t · exp(clamp((v - v_thresh) · inv_delta_t, [-10, 10]))
/// ```
#[inline(always)]
pub fn adex_exp<S: SimdLane>(v: S, v_thresh: S, inv_delta_t: S, delta_t: S) -> S {
    let exp_arg = ((v - v_thresh) * inv_delta_t).max(S::splat(-10.0)).min(S::splat(10.0));
    delta_t * exp_arg.exp()
}

/// AHP current: `I = g · (V - E_ahp)` (subtracted from I_inj)
#[inline(always)]
pub fn ahp_current<S: SimdLane>(g: S, v: S, e_ahp: S) -> S {
    g.mul_add(v, -g * e_ahp)
}

/// Threshold adaptation: `θ' = θ + dt · (-θ) · inv_tau` (GLIF4)
/// After spike: `θ += Δθ`
#[inline(always)]
pub fn integrate_theta<S: SimdLane>(theta: S, inv_tau: S, dt: S) -> S {
    theta + dt * (-theta) * inv_tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_integrate_u_izhi() {
        let u = f32x8::splat(0.0);
        let v = f32x8::splat(-65.0);
        let a = f32x8::splat(0.02);
        let b = f32x8::splat(0.2);
        let dt = f32x8::splat(0.5);
        let u_new = integrate_u(u, v, a, b, dt);
        // u' = 0 + 0.5 * 0.02 * (0.2*(-65) - 0) = 0.5*0.02*(-13) = -0.13
        assert!((u_new.to_array()[0] - (-0.13)).abs() < 1e-6);
    }

    #[test]
    fn test_adex_exp_zero() {
        let v = f32x8::splat(-70.0);
        let v_th = f32x8::splat(-50.0);
        let inv_dt = f32x8::splat(0.1); // 1/10
        let dt_val = f32x8::splat(10.0); // delta_t
        let exp_term = adex_exp(v, v_th, inv_dt, dt_val);
        // arg = (-70 - (-50)) * 0.1 = -2, exp = 0.135, *10 ≈ 1.35
        let expected = 10.0 * (-2.0_f32).exp();
        assert!((exp_term.to_array()[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_ahp_current() {
        let g = f32x8::splat(0.01);
        let v = f32x8::splat(-60.0);
        let e = f32x8::splat(-80.0);
        let i = ahp_current(g, v, e);
        // 0.01 * (-60 - (-80)) = 0.01 * 20 = 0.2 (positive → hyperpolarizing)
        assert!((i.to_array()[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_integrate_theta_decay() {
        let th = f32x8::splat(10.0);
        let inv_tau = f32x8::splat(1.0 / 30.0);
        let dt = f32x8::splat(1.0);
        let th_new = integrate_theta(th, inv_tau, dt);
        let expected = 10.0 + 1.0 * (-10.0 / 30.0);
        assert!((th_new.to_array()[0] - expected).abs() < 0.01);
    }
}
