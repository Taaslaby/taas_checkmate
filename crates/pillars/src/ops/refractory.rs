//! # Refractory — Clamp and Timer
//!
//! ## `refractory_clamp`
//! ```text
//! v' = (refr > 0) ? v_reset : v
//! ```
//! Clamp at INPUT — before any integration. Prevents integrating from
//! spike-peak during the refractory period.
//!
//! ## `refractory_timer`
//! ```text
//! refr' = max(spike ? tau_ref : refr - dt, 0)
//! ```
//! Set to tau_ref on spike, otherwise decrement by dt, clamped to [0, ∞).

use crate::core::lane::SimdLane;

/// Clamp membrane potential to v_reset during refractory.
#[inline(always)]
pub fn refractory_clamp<S: SimdLane>(v: S, refr: S, v_reset: S) -> S {
    let is_refr = refr.cmp_gt(S::zero()).cond();
    is_refr.then(v_reset).else_(v)
}

/// Decrement refractory timer or set to tau_ref on spike.
#[inline(always)]
pub fn refractory_timer<S: SimdLane>(refr: S, spike: S, tau_ref: S, dt: S) -> S {
    let refr_dec = refr - dt;
    let refr_set = spike.cond().then(tau_ref).else_(refr_dec);
    refr_set.max(S::zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_refractory_clamp_active() {
        let v = f32x8::splat(30.0); // spike peak
        let refr = f32x8::splat(5.0); // still refractory
        let reset = f32x8::splat(-70.0);
        let clamped = refractory_clamp(v, refr, reset);
        assert_eq!(clamped.to_array()[0], -70.0, "should clamp to v_reset");
    }

    #[test]
    fn test_refractory_clamp_inactive() {
        let v = f32x8::splat(-65.0);
        let refr = f32x8::splat(0.0); // not refractory
        let reset = f32x8::splat(-70.0);
        let clamped = refractory_clamp(v, refr, reset);
        assert_eq!(clamped.to_array()[0], -65.0, "should NOT clamp when refr=0");
    }

    #[test]
    fn test_refractory_timer_decay() {
        let refr = f32x8::splat(10.0);
        let spike = f32x8::splat(0.0);
        let tau = f32x8::splat(5.0);
        let dt = f32x8::splat(2.0);
        let new = refractory_timer(refr, spike, tau, dt);
        assert_eq!(new.to_array()[0], 8.0, "should decrement by dt");
    }

    #[test]
    fn test_refractory_timer_set() {
        let refr = f32x8::splat(0.0);
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // true mask
        let tau = f32x8::splat(5.0);
        let dt = f32x8::splat(2.0);
        let new = refractory_timer(refr, spike, tau, dt);
        assert_eq!(new.to_array()[0], 5.0, "should set to tau_ref on spike");
    }

    #[test]
    fn test_refractory_timer_non_negative() {
        let refr = f32x8::splat(1.0);
        let spike = f32x8::splat(0.0);
        let tau = f32x8::splat(5.0);
        let dt = f32x8::splat(2.0);
        let new = refractory_timer(refr, spike, tau, dt);
        assert_eq!(new.to_array()[0], 0.0, "should clamp to 0");
    }
}
