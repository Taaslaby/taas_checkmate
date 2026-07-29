//! Gate integration — Exponential Euler for gating variables.
//!
//! `x_new = x_∞ + (x - x_∞) · exp(-dt · τ⁻¹)`
//!
//! Takes **inv_tau** (1/τ) — no division in hot loop, pure FMA.
//! Matches NEURON's `cnexp` method — exact for linear ODEs.
//!
//! ## Zero-division guarantee
//!
//! All params MUST be precomputed as `inv_tau = 1.0 / τ` at init time.
//! The hot path uses only `-dt * inv_tau` — one multiply, one exp.

use wide::f32x8;

/// SIMD: `x = x_inf + (x - x_inf) · exp(-dt · inv_tau)`
///
/// No division — `inv_tau` must be precomputed as `1.0 / τ`.
#[inline(always)]
pub fn gate_integrate(gate: f32x8, x_inf: f32x8, inv_tau: f32, dt: f32) -> f32x8 {
    let decay = (-f32x8::splat(dt * inv_tau)).exp();
    x_inf + (gate - x_inf) * decay
}

/// Scalar version — no division, `inv_tau` must be precomputed.
#[inline(always)]
pub fn gate_integrate_s(gate: f32, x_inf: f32, inv_tau: f32, dt: f32) -> f32 {
    let decay = (-dt * inv_tau).exp();
    x_inf + (gate - x_inf) * decay
}

/// With Q10 temperature scaling: inv_tau_eff = inv_tau · Q10(T)
/// Uses multiplication only — no division.
#[inline(always)]
pub fn gate_integrate_q10(gate: f32x8, x_inf: f32x8, inv_tau: f32, dt: f32, celsius: f32) -> f32x8 {
    let q10 = crate::q10_from_celsius_s(celsius);
    let inv_tau_eff = f32x8::splat(inv_tau * q10);
    let decay = (-f32x8::splat(dt) * inv_tau_eff).exp();
    x_inf + (gate - x_inf) * decay
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_gate_integrate_basic() {
        let inv_tau = 1.0 / 10.0;
        let g = super::gate_integrate(f32x8::splat(0.0), f32x8::splat(1.0), inv_tau, 1.0);
        let expected = 1.0 - (-1.0 / 10.0_f32).exp();
        assert!((g.to_array()[0] - expected).abs() < 0.001, "gate: {}", g.to_array()[0]);
    }

    #[test]
    fn test_gate_integrate_steady_state() {
        let inv_tau = 1.0 / 10.0;
        let g = super::gate_integrate(f32x8::splat(0.5), f32x8::splat(0.5), inv_tau, 1.0);
        assert!((g.to_array()[0] - 0.5).abs() < 0.001, "steady: {}", g.to_array()[0]);
    }

    #[test]
    fn test_gate_integrate_scalar_matches_simd() {
        let inv_tau = 1.0 / 10.0;
        let s = super::gate_integrate_s(0.0, 1.0, inv_tau, 1.0);
        let simd = super::gate_integrate(f32x8::splat(0.0), f32x8::splat(1.0), inv_tau, 1.0);
        assert!((s - simd.to_array()[0]).abs() < 0.001);
    }
}
