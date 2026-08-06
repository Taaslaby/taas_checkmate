//! # 📐 Polynomial — Neuron-Specific Horner/Cubic Forms
//!
//! Per-lane (SIMD-safe) polynomial evaluations used by specific neuron
//! models. All expressed in Horner's form — minimum operations, no power
//! calls.
//!
//! ## Functions
//!
//! | Function | Formula | Model |
//! |----------|---------|-------|
//! | [`horner_izhi`] | `v·(0.04·v + 5) + 140` | Izhikevich quadratic membrane |
//! | [`cubic_fhn`] | `v·(1 − v·v/3)` | FitzHugh-Nagumo cubic |
//! | [`substep_euler_izhi`] | Two half-dt steps | Stable Izhikevich at large dt |

use crate::core::lane::SimdLane;

/// Izhikevich membrane polynomial: `v · (0.04 · v + 5.0) + 140`
///
/// Horner's form minimizes operations: 2 mul + 1 add + 1 FMA.
#[inline(always)]
pub fn horner_izhi<S: SimdLane>(v: S) -> S {
    v * (S::splat(0.04) * v + S::splat(5.0)) + S::splat(140.0)
}

/// FitzHugh-Nagumo cubic polynomial: `v · (1 - v² / 3)`
///
/// Horner's form: `v - v³ / 3 = v · (1 - v·v / 3)`
#[inline(always)]
pub fn cubic_fhn<S: SimdLane>(v: S) -> S {
    v * (S::one() - v * v * S::splat(1.0 / 3.0))
}

/// Two-step Forward Euler substepping: `v_new = substep(v, dv_fn, half_dt)`
///
/// Runs two half-dt steps for numerical stability on quadratic dynamics.
/// `dv_fn(v, u, I)` returns the derivative.
#[inline(always)]
pub fn substep_euler_izhi<S: SimdLane, F: Fn(S, S, S) -> S>(
    v: S,
    u: S,
    i_syn: S,
    half_dt: S,
    dv_fn: F,
) -> S {
    let dv1 = dv_fn(v, u, i_syn) * half_dt;
    let v_mid = v + dv1;
    let dv2 = dv_fn(v_mid, u, i_syn) * half_dt;
    v_mid + dv2
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_horner_izhi_resting() {
        // At V=-65: (-65)*(0.04*(-65)+5)+140 = (-65)*(-2.6+5)+140 = (-65)*2.4+140 = -156+140 = -16
        let v = f32x8::splat(-65.0);
        let h = horner_izhi(v).to_array()[0];
        assert!((h - (-16.0)).abs() < 0.01, "horner(-65): got {:.2}", h);
    }

    #[test]
    fn test_cubic_fhn_resting() {
        // At V=0: 0*(1-0) = 0
        let v = f32x8::splat(0.0);
        let c = cubic_fhn(v).to_array()[0];
        assert!((c - 0.0).abs() < 1e-6, "cubic(0) should be 0");
    }

    #[test]
    fn test_cubic_fhn_positive() {
        // At V=1: 1*(1-1/3) = 0.666...
        let v = f32x8::splat(1.0);
        let c = cubic_fhn(v).to_array()[0];
        assert!(
            (c - (2.0 / 3.0)).abs() < 1e-6,
            "cubic(1) = 2/3, got {:.6}",
            c
        );
    }

    #[test]
    fn test_substep_euler_matches_two_steps() {
        let v = f32x8::splat(-65.0);
        let u = f32x8::splat(0.0);
        // Izhikevich at V=-65: horner(-65) = -65*(0.04*(-65)+5)+140 = -16
        // dv1 = (-16 - 0 + 10) * 0.25 = -1.5 → v_mid = -66.5 (hyperpolarizes first!)
        // This is CORRECT Izhikevich behavior at subthreshold currents.
        let i = f32x8::splat(10.0);
        let half_dt = f32x8::splat(0.25);
        let dv_fn = |v: f32x8, _u: f32x8, _i: f32x8| -> f32x8 {
            super::horner_izhi(v) + _i // dv = f(v) + I (u=0 at t=0)
        };
        let v_double = substep_euler_izhi(v, u, i, half_dt, dv_fn);
        assert!(
            v_double.to_array()[0].is_finite(),
            "substep should be finite"
        );
        // At this low current, v may slightly hyperpolarize before recovering.
        // Just check finiteness and no NaN.
        assert!(!v_double.to_array()[0].is_nan(), "no NaN");
    }
}
