//! # vtrap — Hodgkin-Huxley Voltage Trap Function
//!
//! Computes `x / (exp(x/y) - 1)` while avoiding the singularity at `x = 0`.
//!
//! ## Mathematical Form
//! ```text
//! vtrap(x, y) = x / (exp(x/y) - 1)        when |x/y| > ε
//!             = y · (1 - x/(2y))          when |x/y| ≤ ε  (Taylor expansion)
//! ```
//! where `ε = 1e-6` (numerical stability threshold).
//!
//! The formula lives in **one** place: [`SimdLane::vtrap`] (canonical trait
//! default — branchless `blend()`, compiles to both SIMD and scalar). These
//! free functions are thin concrete-type conveniences.
//!
//! ## Usage in Hodgkin-Huxley Models
//! HH α/β rate functions contain terms like:
//! ```text
//! α_m = 0.1 · vtrap(-(V+40), 10)
//!     = 0.1 · [-(V+40)] / [exp(-(V+40)/10) - 1]
//! ```
//! At `V = -40 mV`, the numerator and denominator both approach 0, causing
//! numerical instability. `vtrap` handles this via Taylor expansion.
//!
//! ## Reference
//! - NEURON's `vtrap` in `hoc.cpp` (Hines & Carnevale 1997)
//! - Hodgkin & Huxley (1952), J. Physiology 117:500-544
//!
//! ## Performance
//! Branchless `blend()` on both SIMD and scalar — no data-dependent branching.
//! Under `fast-exp` feature: uses polynomial exp approximation.

use crate::core::lane::SimdLane;
use wide::f32x8;

/// SIMD vtrap: `x / (exp(x/y) - 1)` for 8 lanes simultaneously — thin wrapper
/// over [`SimdLane::vtrap`].
///
/// # Arguments
/// - `x`: numerator (typically `-(V + V₀)`)
/// - `y`: denominator scale (typically voltage slope factor)
///
/// # Example (HH m-gate α)
/// ```ignore
/// let x = -(v + f32x8::splat(40.0));
/// let y = f32x8::splat(10.0);
/// let alpha_m = f32x8::splat(0.1) * vtrap(x, y);
/// ```
#[inline(always)]
pub fn vtrap(x: f32x8, y: f32x8) -> f32x8 {
    <f32x8 as SimdLane>::vtrap(x, y)
}

/// Scalar vtrap: `x / (exp(x/y) - 1)` for single value — thin wrapper over
/// [`SimdLane::vtrap`].
///
/// Tail-processing version for non-aligned neuron counts.
/// Uses same numerical stability threshold as SIMD version.
///
/// # Arguments
/// - `x`: numerator
/// - `y`: denominator scale (must be non-zero)
#[inline(always)]
pub fn vtrap_s(x: f32, y: f32) -> f32 {
    <f32 as SimdLane>::vtrap(x, y)
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_vtrap_at_zero() {
        let x = f32x8::splat(0.0);
        let y = f32x8::splat(10.0);
        let r = super::vtrap(x, y);
        assert!(
            (r.to_array()[0] - 10.0).abs() < 0.01,
            "vtrap(0,10) = {}",
            r.to_array()[0]
        );
    }

    #[test]
    fn test_vtrap_large() {
        let x = f32x8::splat(10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = 10.0 / (10.0_f32.exp() - 1.0);
        assert!(
            (r.to_array()[0] - expected).abs() < 0.001,
            "vtrap(10,1) = {} vs {}",
            r.to_array()[0],
            expected
        );
    }

    #[test]
    fn test_vtrap_negative() {
        let x = f32x8::splat(-10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = -10.0 / ((-10.0_f32).exp() - 1.0);
        assert!(
            (r.to_array()[0] - expected).abs() < 0.001,
            "vtrap(-10,1) = {} vs {}",
            r.to_array()[0],
            expected
        );
    }

    #[test]
    fn test_vtrap_numerical_stability() {
        for x in [-0.01, -0.001, 0.0, 0.001, 0.01] {
            let r = super::vtrap_s(x, 10.0);
            assert!(r.is_finite(), "vtrap({}, 10) = {} is not finite", x, r);
        }
    }
}
