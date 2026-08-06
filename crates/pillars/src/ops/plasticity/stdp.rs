//! # STDP — Spike-Timing Dependent Plasticity
//!
//! Three-factor STDP (dopamine-modulated) for synaptic weight updates.
//! Calcium-based eligibility traces track pre/post spike history.
//!
//! ## Traces (eligibility)
//! ```text
//! tr_pre'  = tr_pre  · exp(-dt/τ₊) + pre_spike_float
//! tr_post' = tr_post · exp(-dt/τ₋) + post_spike_float
//! ```
//!
//! ## Weight update
//! ```text
//! Δw = η · δ · (tr_pre · post_spike + tr_post · pre_spike)
//! ```
//! where δ is dopamine concentration (reward-prediction error).
//!
//! # Spike convention
//! Accepts SIMD NaN masks (from `cmp_ge`), scalar `-1.0` masks, or arithmetic
//! flags (`1.0`/`0.0`). Converted via
//! [`spike_to_float`](crate::ops::spike::spike_to_float) (IEEE 754
//! NaN-detection).

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

/// One step of STDP eligibility traces.
///
/// Both traces decay exponentially and increment by 1 on their respective
/// spike events.
///
/// Returns `(tr_pre_new, tr_post_new)`.
#[inline(always)]
pub fn stdp_trace<S: SimdLane>(
    tr_pre: S,
    tr_post: S,
    pre_spike: S,
    post_spike: S,
    inv_tau_plus: S,
    inv_tau_minus: S,
    dt: S,
) -> (S, S) {
    let pre = spike_to_float(pre_spike);
    let post = spike_to_float(post_spike);
    let decay_pre = tr_pre * (-dt * inv_tau_plus).exp();
    let decay_post = tr_post * (-dt * inv_tau_minus).exp();
    (decay_pre + pre, decay_post + post)
}

/// Standard STDP weight update (no dopamine modulation).
///
/// ```text
/// w' = max(w + η · (tr_pre · post_spike + tr_post · pre_spike), 0)
/// ```
///
/// LTP when pre precedes post (tr_pre is high at post-spike time).
/// LTD when post precedes pre (tr_post is high at pre-spike time).
#[inline(always)]
pub fn stdp_weight_update<S: SimdLane>(
    w: S,
    tr_pre: S,
    tr_post: S,
    pre_spike: S,
    post_spike: S,
    lr: S,
) -> S {
    let pre = spike_to_float(pre_spike);
    let post = spike_to_float(post_spike);
    let dw = tr_pre * post + tr_post * pre;
    (w + lr * dw).max(S::zero())
}

/// Dopamine-modulated (3-factor) weight update.
///
/// ```text
/// w' = clamp(w + η · δ · (tr_pre·post_spike + tr_post·pre_spike), 0, 1)
/// ```
///
/// - `dopamine` (δ): reward-prediction error signal.
///   δ > 0 → coincident pre-post pairs potentiate (reward).
///   δ < 0 → coincident pairs depress (punishment).
/// - Synaptic weights soft-clamped to [0, 1].
#[inline(always)]
pub fn dopamine_weight_update<S: SimdLane>(
    w: S,
    tr_pre: S,
    tr_post: S,
    pre_spike: S,
    post_spike: S,
    dopamine: S,
    lr: S,
) -> S {
    let pre = spike_to_float(pre_spike);
    let post = spike_to_float(post_spike);
    let dw = dopamine * (tr_pre * post + tr_post * pre);
    (w + lr * dw).clamp01()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_stdp_trace_decay() {
        let tr = f32x8::splat(1.0);
        let inv_tau = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);
        let (new, _) = stdp_trace(
            tr,
            f32x8::splat(0.0),
            f32x8::splat(0.0),
            f32x8::splat(0.0),
            inv_tau,
            inv_tau,
            dt,
        );
        assert!(new.to_array()[0] < 1.0, "trace should decay");
        assert!(new.to_array()[0] > 0.0, "trace should stay positive");
    }

    #[test]
    fn test_stdp_trace_spike_increment() {
        let tr = f32x8::splat(0.5);
        let inv_tau = f32x8::splat(1.0 / 1000.0);
        let dt = f32x8::splat(0.1);

        // Test with mask (-1.0) and float (1.0)
        let (m, _) = stdp_trace(
            tr,
            f32x8::splat(0.0),
            f32x8::splat(-1.0),
            f32x8::splat(0.0),
            inv_tau,
            inv_tau,
            dt,
        );
        let (f, _) = stdp_trace(
            tr,
            f32x8::splat(0.0),
            f32x8::splat(1.0),
            f32x8::splat(0.0),
            inv_tau,
            inv_tau,
            dt,
        );

        assert!(
            (m.to_array()[0] - 1.5).abs() < 0.01,
            "mask spike: expected ~1.5, got {:.4}",
            m.to_array()[0]
        );
        assert!(
            (f.to_array()[0] - 1.5).abs() < 0.01,
            "float spike: expected ~1.5, got {:.4}",
            f.to_array()[0]
        );
    }

    #[test]
    fn test_dopamine_weight_bounds() {
        let w = f32x8::splat(0.5);
        let tr_pre = f32x8::splat(0.1);
        let tr_post = f32x8::splat(0.05);
        let pre = f32x8::splat(-1.0); // mask
        let post = f32x8::splat(-1.0); // mask
        let dopamine = f32x8::splat(0.5);
        let lr = f32x8::splat(0.01);
        let new_w = dopamine_weight_update(w, tr_pre, tr_post, pre, post, dopamine, lr);
        assert!(new_w.to_array()[0] > 0.5, "should increase w");
        assert!(new_w.to_array()[0] <= 1.0, "should be clamped ≤ 1");
    }

    #[test]
    fn test_weight_update_both_spike_formats() {
        let w = f32x8::splat(0.5);
        let tr_pre = f32x8::splat(0.2);
        let tr_post = f32x8::splat(0.1);
        let lr = f32x8::splat(0.1);

        // Mask format: pre=-1.0 (active), post=-1.0 (active)
        let wm = stdp_weight_update(
            w,
            tr_pre,
            tr_post,
            f32x8::splat(-1.0),
            f32x8::splat(-1.0),
            lr,
        );

        // Float format: pre=1.0 (active), post=1.0 (active)
        let wf = stdp_weight_update(w, tr_pre, tr_post, f32x8::splat(1.0), f32x8::splat(1.0), lr);

        assert!(
            (wm.to_array()[0] - wf.to_array()[0]).abs() < 1e-6,
            "mask and float should give same w: {:.6} vs {:.6}",
            wm.to_array()[0],
            wf.to_array()[0]
        );
    }
}
