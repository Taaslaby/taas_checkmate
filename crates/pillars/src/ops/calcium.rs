//! # Calcium — Ca²⁺ Concentration Dynamics
//!
//! Calcium ions (Ca²⁺) are a ubiquitous second messenger in neurons and glia.
//! They gate short-term plasticity (STP), gene expression, and astrocytic
//! signaling. All operations are **SIMD-safe** and avoid divisions.
//!
//! ## Functions
//!
//! | Operation | Equation | Use | `alpha` meaning |
//! |-----------|----------|-----|----------------|
//! | `calcium_step` | `Ca' = Ca + dt · ((Ca₀ - Ca)·τ⁻¹ + α·I)` | General Ca²⁺ dynamics | input-current **gain** (scales external drive) |
//! | `calcium_bump` | `Ca' = clamp(Ca + α · spike_float, 0, 1)` | Spike-triggered Ca²⁺ influx | fixed Ca²⁺ **increment per spike** |
//! | `decay_toward` | `x' = x + dt · (x₀ - x)·τ⁻¹` | Exponential relaxation | — |
//!
//! > ⚠️ `alpha` is deliberately overloaded: a *gain* in `calcium_step` (units of
//! > concentration per current) vs a *per-spike increment* in `calcium_bump`
//! > (units of concentration). They have different dimensions — don't reuse
//! > one value for both calls.
//!
//! # FMA note
//! The `a + b·c` patterns here compile to FMA instructions on FMA-capable
//! targets via LLVM contraction (`wide` lowers to native SIMD; explicit
//! `.mul_add` is used in the hot-path ops like `ohmic_gated` where the FMA
//! fusion matters for bit-exact parity with the CUDA backend).
//!
//! # Spike convention
//! All `spike` parameters accept a SIMD **mask** (`NaN`/`0.0` from `cmp_ge`) OR
//! an **arithmetic** flag (`1.0`/`0.0`) OR a scalar `-1.0` mask. Converted via
//! [`spike_to_float`](crate::ops::spike::spike_to_float), which uses IEEE 754
//! NaN-detection (`x >= x` is false only for NaN) — see that function for the
//! full contract.

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

/// Convert any spike representation (NaN mask, -1.0 mask, or 1.0 float) to 1.0/0.0.
///
/// Thin wrapper over [`spike_to_float`](crate::ops::spike::spike_to_float) — the
/// single source of truth for spike-format conversion (NaN-detection via the
/// IEEE 754 property that `x >= x` is false only for NaN; see spike.rs).
#[inline(always)]
fn spike_active<S: SimdLane>(spike: S) -> S {
    spike_to_float(spike)
}

/// Single Forward Euler step for calcium concentration.
///
/// ```text
/// Ca' = Ca + dt · ((Ca₀ - Ca) · τ⁻¹ + α · I)
/// ```
///
/// - `ca` — current Ca²⁺ concentration
/// - `input` — external current / presynaptic input
/// - `inv_tau` — inverse time constant (1/τ_ca), precomputed
/// - `alpha` — input scale factor
/// - `target` — steady-state Ca²⁺ concentration (Ca₀)
/// - `dt` — simulation timestep
///
/// Result clamped to [0, 1] for biological plausibility.
#[inline(always)]
pub fn calcium_step<S: SimdLane>(ca: S, input: S, inv_tau: S, alpha: S, target: S, dt: S) -> S {
    let new = ca + dt * ((target - ca) * inv_tau + alpha * input);
    new.clamp01()
}

/// Spike-triggered Ca²⁺ influx: adds `alpha` on each spike.
///
/// ```text
/// Ca' = clamp(Ca + spike_float · α, 0, 1)
/// ```
///
/// Accepts mask (-1.0/0.0) or arithmetic (1.0/0.0) spike representation.
#[inline(always)]
pub fn calcium_bump<S: SimdLane>(ca: S, alpha: S, spike: S) -> S {
    (ca + spike_active(spike) * alpha).clamp01()
}

/// Exponential relaxation toward a target.
///
/// ```text
/// x' = x + dt · (x₀ - x) · τ⁻¹
/// ```
///
/// Pure first-order decay (no input, no clamp). Used for ion concentrations,
/// adaptation variables, and other slow dynamics.
#[inline(always)]
pub fn decay_toward<S: SimdLane>(state: S, target: S, inv_tau: S, dt: S) -> S {
    state + dt * (target - state) * inv_tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    fn test_bump(spike: f32x8, label: &str) {
        let ca = f32x8::splat(0.2);
        let alpha = f32x8::splat(0.05);
        let bumped = calcium_bump(ca, alpha, spike);
        assert!(
            (bumped.to_array()[0] - 0.25).abs() < 1e-6,
            "{}: expected 0.25, got {:.4}",
            label,
            bumped.to_array()[0]
        );
    }

    #[test]
    fn test_calcium_bump_with_mask() {
        test_bump(f32x8::splat(-1.0), "mask -1.0");
    }

    #[test]
    fn test_calcium_bump_with_float() {
        test_bump(f32x8::splat(1.0), "float 1.0");
    }

    #[test]
    fn test_calcium_bump_no_spike() {
        let ca = f32x8::splat(0.2);
        let alpha = f32x8::splat(0.05);
        let bumped = calcium_bump(ca, alpha, f32x8::splat(0.0));
        assert!(
            (bumped.to_array()[0] - 0.20).abs() < 1e-6,
            "no spike: expected 0.20, got {:.4}",
            bumped.to_array()[0]
        );
    }

    #[test]
    fn test_calcium_step_with_input() {
        let ca = f32x8::splat(0.0);
        let input = f32x8::splat(1.0);
        let inv_tau = f32x8::splat(1.0 / 1000.0);
        let alpha = f32x8::splat(0.1);
        let target = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = calcium_step(ca, input, inv_tau, alpha, target, dt);
        let expected = (0.1f32 - 0.0) * 0.001 + 0.1;
        assert!(
            (new.to_array()[0] - expected).abs() < 1e-6,
            "got {:.6}, expected {:.6}",
            new.to_array()[0],
            expected
        );
    }

    #[test]
    fn test_calcium_clamp_upper() {
        let ca = f32x8::splat(0.9);
        let input = f32x8::splat(1000.0);
        let inv_tau = f32x8::splat(1.0);
        let alpha = f32x8::splat(1.0);
        let target = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = calcium_step(ca, input, inv_tau, alpha, target, dt);
        assert!(
            new.to_array()[0] <= 1.0,
            "Ca should be clamped to ≤1, got {:.4}",
            new.to_array()[0]
        );
    }

    /// Lower clamp: huge negative input must floor Ca at 0, not go negative.
    #[test]
    fn test_calcium_clamp_lower() {
        let ca = f32x8::splat(0.5);
        let input = f32x8::splat(-1000.0);
        let inv_tau = f32x8::splat(1.0);
        let alpha = f32x8::splat(1.0);
        let target = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = calcium_step(ca, input, inv_tau, alpha, target, dt);
        assert!(
            new.to_array()[0] >= 0.0,
            "Ca should be clamped to ≥0, got {:.4}",
            new.to_array()[0]
        );
        assert!((new.to_array()[0] - 0.0).abs() < 1e-6);
    }

    /// Calcium bump must detect a NaN SIMD mask (what `cmp_ge` actually
    /// produces) — the claimed spike format that was never explicitly tested.
    #[test]
    fn test_calcium_bump_nan_mask() {
        let ca = f32x8::splat(0.2);
        let alpha = f32x8::splat(0.05);
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // true → NaN mask
        let bumped = calcium_bump(ca, alpha, spike);
        assert!(
            (bumped.to_array()[0] - 0.25).abs() < 1e-6,
            "NaN mask → +α, got {:.4}",
            bumped.to_array()[0]
        );
    }

    /// decay_toward: pure first-order relaxation, no clamp, no input.
    #[test]
    fn test_decay_toward_approaches_target() {
        let state = f32x8::splat(0.0);
        let target = f32x8::splat(1.0);
        let inv_tau = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = decay_toward(state, target, inv_tau, dt);
        let expected = 0.0 + (1.0 - 0.0) * 0.1; // x + dt·(x₀−x)·τ⁻¹
        assert!(
            (new.to_array()[0] - expected).abs() < 1e-6,
            "decay step: got {:.6}, expected {:.6}",
            new.to_array()[0],
            expected
        );

        // Repeated steps converge monotonically toward the target.
        let mut x = new;
        for _ in 0..100 {
            x = decay_toward(x, target, inv_tau, dt);
        }
        assert!(x.to_array()[0] < 1.0, "converges from below");
        assert!(
            x.to_array()[0] > 0.99,
            "after 100 steps should be near 1.0, got {:.4}",
            x.to_array()[0]
        );
    }

    /// All 8 SIMD lanes must behave identically — guards against any lane
    /// asymmetry sneaking into the ops (e.g. from a bad shuffle/permute).
    #[test]
    fn test_all_lanes_identical() {
        let vals = [0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let v = f32x8::from(vals);

        // calcium_step: all lanes see the same params → outputs must match
        // the per-lane scalar math exactly.
        let step = calcium_step(
            v,
            f32x8::splat(0.5),
            f32x8::splat(0.01),
            f32x8::splat(0.1),
            f32x8::splat(0.3),
            f32x8::splat(0.5),
        );
        for (i, &vi) in vals.iter().enumerate() {
            let expected = (vi + 0.5 * ((0.3 - vi) * 0.01 + 0.1 * 0.5)).clamp(0.0, 1.0);
            assert!(
                (step.to_array()[i] - expected).abs() < 1e-5,
                "lane {i}: got {:.6}, expected {:.6}",
                step.to_array()[i],
                expected
            );
        }

        // calcium_bump: lanes with a spike (odd index) bump, others don't.
        let mask = f32x8::from([-1.0f32, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0]);
        let bumped = calcium_bump(f32x8::splat(0.2), f32x8::splat(0.05), mask);
        for i in 0..8 {
            let expected = if i % 2 == 0 { 0.25 } else { 0.20 };
            assert!(
                (bumped.to_array()[i] - expected).abs() < 1e-6,
                "bump lane {i}: got {:.4}, expected {expected}",
                bumped.to_array()[i]
            );
        }
    }
}
