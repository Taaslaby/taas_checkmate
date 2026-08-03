//! # Calcium — Ca²⁺ Concentration Dynamics
//!
//! Calcium ions (Ca²⁺) are a ubiquitous second messenger in neurons and glia.
//! They gate short-term plasticity (STP), gene expression, and astrocytic
//! signaling. All operations are **SIMD-safe** and avoid divisions.
//!
//! ## Functions
//!
//! | Operation | Equation | Use |
//! |-----------|----------|-----|
//! | `calcium_step` | `Ca' = Ca + dt · ((Ca₀ - Ca)·τ⁻¹ + α·I)` | General Ca²⁺ dynamics |
//! | `calcium_bump` | `Ca' = clamp(Ca + α · spike_float, 0, 1)` | Spike-triggered Ca²⁺ influx |
//! | `decay_toward` | `x' = x + dt · (x₀ - x)·τ⁻¹` | Exponential relaxation |
//!
//! # Spike convention
//! All `spike` parameters accept a SIMD **mask** (`-1.0`/`0.0` from `cmp_ge`) OR
//! an **arithmetic** flag (`1.0`/`0.0`). Internally converted by squaring:
//! `spike² > 0 → 1.0`, `spike² = 0 → 0.0`. Bulletproof for both formats.

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

/// Convert any spike representation (NaN mask, -1.0 mask, or 1.0 float) to 1.0/0.0.
///
/// Делегирует `spike_to_float` — тот использует NaN-детекцию через
/// свойство IEEE 754 (только NaN не проходит `x >= x`).
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
        assert!((bumped.to_array()[0] - 0.25).abs() < 1e-6,
            "{}: expected 0.25, got {:.4}", label, bumped.to_array()[0]);
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
        assert!((bumped.to_array()[0] - 0.20).abs() < 1e-6,
            "no spike: expected 0.20, got {:.4}", bumped.to_array()[0]);
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
        assert!((new.to_array()[0] - expected).abs() < 1e-6,
            "got {:.6}, expected {:.6}", new.to_array()[0], expected);
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
        assert!(new.to_array()[0] <= 1.0,
            "Ca should be clamped to ≤1, got {:.4}", new.to_array()[0]);
    }
}
