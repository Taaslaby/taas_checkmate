//! # Homeo — Homeostatic Synaptic Scaling
//!
//! Homeostatic plasticity maintains network activity within a target firing
//! range. Unlike Hebbian (positive feedback), homeostatic scaling is
//! **negative feedback**: high firing → weights decrease, low firing →
//! weights increase.
//!
//! ## `homeostatic_scale`
//! ```text
//! w' = max(w - (r - r₀) · w · τ⁻¹ · dt, 0)
//! ```
//! Multiplicative normalization. When `rate > target_rate`, weights scale down.
//!
//! ## `rate_estimate`
//! ```text
//! rate' = EMA(rate, spike_hz, τ)
//!       = spike_hz + (rate - spike_hz) · exp(-dt/τ)
//! where spike_hz = spike_float · 1000 / dt   (converts spikes to Hz)
//! ```
//! Exponential moving average invariant to simulation timestep.
//! At dt = 1ms: single spike → 1000 Hz instantaneous rate.
//! At dt = 0.1ms: same spike → 10000 Hz.
//! Both converge to the same EMA value.

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

/// Homeostatic weight scaling: decrease weight when rate exceeds target.
///
/// ```text
/// w' = max(w - (rate - target) · w · τ⁻¹ · dt, 0)
/// ```
///
/// - `w` — current synaptic weight
/// - `rate` — estimated firing rate in Hz (from `rate_estimate`)
/// - `target_rate` — target firing rate in Hz
/// - `inv_tau` — inverse homeostatic time constant (1/τ_homeo)
/// - `dt` — simulation timestep
#[inline(always)]
pub fn homeostatic_scale<S: SimdLane>(w: S, rate: S, target_rate: S, inv_tau: S, dt: S) -> S {
    let error = rate - target_rate;
    (w - error * w * inv_tau * dt).max(S::zero())
}

/// Running rate estimate in Hz (exponential moving average of spike events).
///
/// Converts discrete spike events to Hz, making the estimate **invariant**
/// to simulation timestep. Without this, changing `dt` would shift the
/// rate and break homeostatic set-points.
///
/// # Arguments
/// - `rate` — previous rate estimate (Hz)
/// - `spike` — spike event from `cmp_ge()` (mask -1.0/0.0) or arithmetic (1.0/0.0)
/// - `inv_tau` — inverse time constant (1/τ_rate) for the EMA filter
/// - `dt` — simulation timestep
#[inline(always)]
pub fn rate_estimate<S: SimdLane>(rate: S, spike: S, inv_tau: S, dt: S) -> S {
    let spike_f = spike_to_float(spike);
    let decay = (-dt * inv_tau).exp();
    let spike_hz = spike_f * S::splat(1000.0) / dt;
    // EMA: rate' = decay * rate + (1 - decay) * spike_hz
    spike_hz + (rate - spike_hz) * decay
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_homeo_high_rate_decreases_weight() {
        let w = f32x8::splat(1.0);
        let rate = f32x8::splat(50.0);
        let target = f32x8::splat(10.0);
        let inv_tau = f32x8::splat(1.0 / 1000.0);
        let dt = f32x8::splat(1.0);
        let new_w = homeostatic_scale(w, rate, target, inv_tau, dt);
        assert!(new_w.to_array()[0] < 1.0,
            "high rate → decrease w: {:.6}", new_w.to_array()[0]);
    }

    #[test]
    fn test_homeo_low_rate_increases_weight() {
        let w = f32x8::splat(0.5);
        let rate = f32x8::splat(2.0);
        let target = f32x8::splat(10.0);
        let inv_tau = f32x8::splat(1.0 / 1000.0);
        let dt = f32x8::splat(1.0);
        let new_w = homeostatic_scale(w, rate, target, inv_tau, dt);
        assert!(new_w.to_array()[0] > 0.5,
            "low rate → increase w: {:.6}", new_w.to_array()[0]);
    }

    #[test]
    fn test_rate_estimate_decay() {
        let rate = f32x8::splat(10.0);
        let spike = f32x8::splat(0.0);
        let inv_tau = f32x8::splat(1.0 / 100.0);
        let dt = f32x8::splat(1.0);
        let new = rate_estimate(rate, spike, inv_tau, dt);
        assert!(new.to_array()[0] < 10.0,
            "rate should decay: {:.4} < 10", new.to_array()[0]);
    }

    #[test]
    fn test_rate_estimate_spike_both_formats() {
        let rate = f32x8::splat(0.0);
        let inv_tau = f32x8::splat(1.0 / 1000.0);
        let dt = f32x8::splat(1.0);

        let r_mask = rate_estimate(rate, f32x8::splat(-1.0), inv_tau, dt);
        let r_float = rate_estimate(rate, f32x8::splat(1.0), inv_tau, dt);

        // At dt=1ms, spike_hz = 1.0·1000/1.0 = 1000 Hz
        // decay = exp(-1/1000) ≈ 0.999
        // rate' = 1000 * (1 - 0.999) ≈ 1.0
        assert!(r_mask.to_array()[0] > 0.5,
            "mask spike → rate ~1 Hz: {:.4}", r_mask.to_array()[0]);
        assert!(r_float.to_array()[0] > 0.5,
            "float spike → rate ~1 Hz: {:.4}", r_float.to_array()[0]);
    }

    #[test]
    fn test_rate_invariant_to_dt() {
        // Same firing rate should give same estimate regardless of dt
        let inv_tau = f32x8::splat(1.0 / 100.0);

        // dt=1ms: 1 spike every 10 steps = 100 Hz
        let mut rate_1 = f32x8::splat(0.0);
        for step in 0..100 {
            let spike = if step % 10 == 0 { f32x8::splat(-1.0) } else { f32x8::splat(0.0) };
            rate_1 = rate_estimate(rate_1, spike, inv_tau, f32x8::splat(1.0));
        }

        // dt=0.5ms: same 100 Hz, 1 spike every 20 steps
        let mut rate_05 = f32x8::splat(0.0);
        for step in 0..200 {
            let spike = if step % 20 == 0 { f32x8::splat(-1.0) } else { f32x8::splat(0.0) };
            rate_05 = rate_estimate(rate_05, spike, inv_tau, f32x8::splat(0.5));
        }

        let diff = (rate_1.to_array()[0] - rate_05.to_array()[0]).abs();
        assert!(diff < 2.0,
            "dt invariance: dt=1 → {:.1}, dt=0.5 → {:.1}, diff={:.1}",
            rate_1.to_array()[0], rate_05.to_array()[0], diff);
    }
}
