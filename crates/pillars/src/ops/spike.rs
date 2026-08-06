//! # Spike — Detection, Reset, Post-Spike Bump & Format Conversion
//!
//! ## `fire_reset`
//! ```text
//! spike = v ≥ v_thresh
//! v'    = spike ? v_reset : v
//! ```
//!
//! ## `edge_spike`
//! ```text
//! spike = (v ≥ v_thresh) AND (v_prev < v_thresh)
//! ```
//! Rising edge only — prevents sustained above-threshold from re-firing.
//!
//! ## `spike_bump`
//! ```text
//! state' = spike ? state + bump : state
//! ```
//! For u += d (Izhikevich), g_ahp += Δg (GLIF3), θ += Δθ (GLIF4).
//!
//! ## `spike_to_float`
//! ```text
//! active = (spike is NaN) OR (|spike| > 0) ? 1.0 : 0.0
//! ```
//! Universal converter via IEEE 754 NaN-detection (`x >= x` is false only for
//! NaN): SIMD NaN mask → `1.0`, scalar `-1.0` mask → `1.0`, float `1.0` → `1.0`,
//! `0.0` → `0.0`. (The historical `spike² > 0` scheme was dropped because
//! NaN·NaN = NaN, which made `NaN > 0` false.)

use crate::core::lane::SimdLane;

/// Universal spike format converter: any active spike → 1.0, silent → 0.0.
///
/// 🛡️ BUGFIX: `spike² > 0` НЕ работает для SIMD-масок — NaN·NaN = NaN, NaN > 0 = false.
/// Новая логика использует свойство IEEE 754: NaN — единственный float, где `x >= x` false.
///
/// Detects all three formats in one branchless chain:
/// - **NaN** (f32x8 SIMD mask from `cmp_ge`): `!(spike >= spike)` = true (только для NaN)
/// - **-1.0** (f32 scalar mask from `cmp_ge`): `spike.abs() > 0` = true
/// - **1.0** (arithmetic active): `spike.abs() > 0` = true
/// - **0.0** (silent): оба false → inactive
///
/// Комбинация: `is_nan | is_nonzero` через De Morgan `!(!is_nan & !is_nonzero)`.
/// Для NaN битовый OR с маской заведомо даёт sign_bit=1; для -1.0 | 1.0 тоже.
#[inline(always)]
pub fn spike_to_float<S: SimdLane>(spike: S) -> S {
    let is_nan = spike.cmp_ge(spike).not(); // true only for NaN
    let is_nonzero = spike.abs().cmp_gt(S::zero()); // true for any non-zero float except NaN
                                                    // Bitwise OR: is_nan | is_nonzero  (via De Morgan: !(!is_nan & !is_nonzero))
    let active = is_nan.not().and(is_nonzero.not()).not();
    active.cond().then(S::one()).else_(S::zero())
}

/// Fire-and-reset: spike if v ≥ v_thresh, reset to v_reset.
/// Returns spike mask and new voltage.
#[inline(always)]
pub fn fire_reset<S: SimdLane>(v: S, v_thresh: S, v_reset: S) -> (S, S) {
    let spike = v.cmp_ge(v_thresh);
    let v_new = spike.cond().then(v_reset).else_(v);
    (v_new, spike)
}

/// Edge-triggered spike detection (rising edge only).
///
/// `spike = is_above AND was_below`
///
/// Returns spike mask and updated was_below flag.
/// `was_below_next = is_above ? 0 : 1` — inverted for next step.
#[inline(always)]
pub fn edge_spike<S: SimdLane>(is_above: S, was_below: S) -> (S, S) {
    let spike = is_above.and(was_below.cmp_gt(S::zero()));
    let was_below_next = is_above.cond().then(S::zero()).else_(S::one());
    (spike, was_below_next)
}

/// Post-spike bump: `state' = spike ? state + bump : state`
#[inline(always)]
pub fn spike_bump<S: SimdLane>(state: S, bump: S, spike: S) -> S {
    spike.cond().then(state + bump).else_(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_fire_reset_spike() {
        let v = f32x8::splat(0.0);
        let th = f32x8::splat(-10.0);
        let reset = f32x8::splat(-70.0);
        let (v_new, spike) = fire_reset(v, th, reset);
        assert_eq!(v_new.to_array()[0], -70.0, "should reset");
        // Convert spike mask to bool via sign bit
        assert!(spike.move_mask() != 0, "should spike");
    }

    #[test]
    fn test_fire_reset_no_spike() {
        let v = f32x8::splat(-80.0);
        let th = f32x8::splat(-50.0);
        let reset = f32x8::splat(-70.0);
        let (v_new, spike) = fire_reset(v, th, reset);
        assert_eq!(v_new.to_array()[0], -80.0, "should not reset");
        assert_eq!(spike.move_mask(), 0, "should not spike");
    }

    #[test]
    fn test_edge_spike_rising() {
        let one = f32x8::splat(1.0);
        let zero = f32x8::splat(0.0);
        let is_above = one.cmp_ge(zero); // true mask (-1.0, all bits set)
                                         // was_below MUST be 1.0 (not -1.0!) — edge_spike uses cmp_gt(zero) internally
                                         // to convert flag to mask. A flag of 1.0 passes, -1.0 does NOT.
        let was_below = f32x8::splat(1.0); // was below → edge!
        let (spike, flag) = edge_spike(is_above, was_below);
        assert!(spike.move_mask() != 0, "rising edge should spike");
        assert_eq!(flag.to_array()[0], 0.0, "flag should be 0 after spike");
    }

    #[test]
    fn test_edge_spike_sustained() {
        let is_above = f32x8::splat(1.0); // still above
        let was_below = f32x8::splat(0.0); // was NOT below → not an edge
        let (spike, flag) = edge_spike(is_above, was_below);
        assert_eq!(spike.move_mask(), 0, "sustained above should NOT spike");
        assert_eq!(flag.to_array()[0], 0.0, "flag stays 0");
    }

    #[test]
    fn test_spike_to_float_nan_mask() {
        // f32x8 SIMD mask: cmp_ge returns NaN for true
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0));
        let result = spike_to_float(spike);
        assert!(
            (result.to_array()[0] - 1.0).abs() < 1e-6,
            "NaN mask → 1.0, got {:.4}",
            result.to_array()[0]
        );
    }

    #[test]
    fn test_spike_to_float_neg_one_mask() {
        // f32 scalar mask: cmp_ge returns -1.0 for true
        let spike: f32 = -1.0;
        let result = spike_to_float(spike);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "f32 -1.0 mask → 1.0, got {:.4}",
            result
        );
    }

    #[test]
    fn test_spike_to_float_float_one() {
        // Arithmetic float 1.0 (positive active)
        assert!((spike_to_float(1.0f32) - 1.0).abs() < 1e-6, "f32 1.0 → 1.0");
        assert!((spike_to_float(0.0f32) - 0.0).abs() < 1e-6, "f32 0.0 → 0.0");
    }

    #[test]
    fn test_spike_to_float_f32x8_float_one() {
        let spike = f32x8::splat(1.0);
        let result = spike_to_float(spike);
        assert!(
            (result.to_array()[0] - 1.0).abs() < 1e-6,
            "f32x8 1.0 → 1.0, got {:.4}",
            result.to_array()[0]
        );
    }

    #[test]
    fn test_spike_to_float_zero() {
        assert!((spike_to_float(0.0f32) - 0.0).abs() < 1e-6, "0.0 → 0.0");
        let spike = f32x8::splat(0.0);
        let result = spike_to_float(spike);
        assert!(
            (result.to_array()[0] - 0.0).abs() < 1e-6,
            "f32x8 0.0 → 0.0, got {:.4}",
            result.to_array()[0]
        );
    }

    #[test]
    fn test_spike_bump() {
        let state = f32x8::splat(0.0);
        let bump = f32x8::splat(8.0);
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // true mask
        let bumped = spike_bump(state, bump, spike);
        assert_eq!(bumped.to_array()[0], 8.0, "should bump on spike");
    }
}
