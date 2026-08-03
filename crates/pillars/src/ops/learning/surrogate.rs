//! # Surrogate — Surrogate Gradient Functions
//!
//! For backpropagation through spiking neurons (SLAYER, SuperSpike).
//! Provides smooth approximations to the spike step function.
//!
//! ## `fast_sigmoid_surrogate`
//! ```text
//! dσ/dv ≈ 1 / (1 + |v - v_th|)²
//! ```
//! Fast sigmoid surrogate from SuperSpike (Zenke & Ganguli 2018).
//!
//! ## `rectangular_surrogate`
//! ```text
//! dσ/dv = 1 if |v - v_th| < width, else 0
//! ```
//! Rectangular pulse surrogate (Bellec et al. 2018).

use crate::core::lane::SimdLane;

/// Fast sigmoid surrogate gradient: `1 / (1 + |v - v_th|)²`
#[inline(always)]
pub fn fast_sigmoid_surrogate<S: SimdLane>(v: S, v_th: S) -> S {
    let diff = (v - v_th).abs();
    S::one() / ((S::one() + diff) * (S::one() + diff))
}

/// Rectangular surrogate: `width` around threshold.
#[inline(always)]
pub fn rectangular_surrogate<S: SimdLane>(v: S, v_th: S, width: S) -> S {
    let diff = (v - v_th).abs();
    diff.cmp_gt(width).cond().then(S::zero()).else_(S::one())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_surrogate_at_threshold() {
        let v = f32x8::splat(-54.0);
        let th = f32x8::splat(-54.0);
        let g = fast_sigmoid_surrogate(v, th);
        assert!((g.to_array()[0] - 1.0).abs() < 1e-6, "at threshold: {:.6}", g.to_array()[0]);
    }

    #[test]
    fn test_surrogate_far_from_threshold() {
        let v = f32x8::splat(-65.0);
        let th = f32x8::splat(-54.0);
        let g = fast_sigmoid_surrogate(v, th);
        assert!(g.to_array()[0] < 0.5, "far from threshold: {:.6}", g.to_array()[0]);
    }
}
