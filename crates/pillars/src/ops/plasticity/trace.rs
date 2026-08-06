//! # Trace — Spike Train Traces for STDP & Eligibility
//!
//! Exponential decay traces track pre- and post-synaptic spike history,
//! forming the eligibility signal for STDP and 3-factor learning rules.
//!
//! ## Functions
//!
//! | Operation | Equation | Use |
//! |-----------|----------|-----|
//! | `trace_decay` | `tr' = tr · exp(-dt · τ⁻¹)` | Pure decay between spikes |
//! | `trace_add` | `tr' = tr + spike_float` | Increment on spike |
//! | `trace_step` | `tr' = tr · exp(-dt·τ⁻¹) + spike_float` | Decay + add in one pass |
//!
//! # Spike convention
//! Accepts SIMD NaN masks (from `cmp_ge`), scalar `-1.0` masks, or arithmetic
//! flags (`1.0`/`0.0`). Converted via
//! [`spike_to_float`](crate::ops::spike::spike_to_float) (IEEE 754
//! NaN-detection).

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

/// Exponential trace decay: `tr' = tr · exp(-dt · τ⁻¹)`
#[inline(always)]
pub fn trace_decay<S: SimdLane>(trace: S, inv_tau: S, dt: S) -> S {
    trace * (-dt * inv_tau).exp()
}

/// Increment trace on spike: `tr' = tr + spike_float`
///
/// Uses `spike_to_float` — accepts both mask (-1.0) and arithmetic (1.0).
#[inline(always)]
pub fn trace_add<S: SimdLane>(trace: S, spike: S) -> S {
    trace + spike_to_float(spike)
}

/// Combined decay + spike increment: `tr' = tr · exp(-dt·τ⁻¹) + spike_float`
#[inline(always)]
pub fn trace_step<S: SimdLane>(trace: S, spike: S, inv_tau: S, dt: S) -> S {
    trace * (-dt * inv_tau).exp() + spike_to_float(spike)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_trace_decay() {
        let tr = f32x8::splat(1.0);
        let inv_tau = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);
        let new = trace_decay(tr, inv_tau, dt);
        assert!(new.to_array()[0] < 1.0, "trace should decay");
        assert!(new.to_array()[0] > 0.0, "trace should stay positive");
    }

    #[test]
    fn test_trace_step_both_formats() {
        let tr = f32x8::splat(0.0);
        let inv_tau = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);

        // Mask -1.0
        let m = trace_step(tr, f32x8::splat(-1.0), inv_tau, dt);
        // Float 1.0
        let f = trace_step(tr, f32x8::splat(1.0), inv_tau, dt);

        assert!(
            (m.to_array()[0] - 1.0).abs() < 0.01,
            "mask → ~1.0, got {:.4}",
            m.to_array()[0]
        );
        assert!(
            (f.to_array()[0] - 1.0).abs() < 0.01,
            "float → ~1.0, got {:.4}",
            f.to_array()[0]
        );
    }

    #[test]
    fn test_trace_add_no_spike() {
        let tr = f32x8::splat(0.5);
        let spike = f32x8::splat(0.0);
        let new = trace_add(tr, spike);
        assert!((new.to_array()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_trace_add_mask_and_float_match() {
        let tr = f32x8::splat(0.5);
        let m = trace_add(tr, f32x8::splat(-1.0));
        let f = trace_add(tr, f32x8::splat(1.0));
        assert!((m.to_array()[0] - 1.5).abs() < 1e-6);
        assert!(
            (m.to_array()[0] - f.to_array()[0]).abs() < 1e-6,
            "mask({:.4}) ≠ float({:.4})",
            m.to_array()[0],
            f.to_array()[0]
        );
    }
}
