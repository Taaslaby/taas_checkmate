//! # Hebbian — Hebbian Learning Rules
//!
//! ## `hebbian_correlation`
//! ```text
//! Δw = η · pre · post
//! ```
//! Classical Hebb: cells that fire together, wire together.
//!
//! ## `oja_rule`
//! ```text
//! Δw = η · post · (pre - post · w)
//! ```
//! Oja's rule: Hebbian with weight normalization (PCA).

use crate::core::lane::SimdLane;

/// Classical Hebbian: `w' = w + η · pre · post`
#[inline(always)]
pub fn hebbian_correlation<S: SimdLane>(w: S, pre: S, post: S, eta: S) -> S {
    (w + eta * pre * post).max(S::zero())
}

/// Oja's rule: `w' = w + η · post · (pre - post · w)`
#[inline(always)]
pub fn oja_rule<S: SimdLane>(w: S, pre: S, post: S, eta: S) -> S {
    w + eta * post * (pre - post * w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_hebbian_correlation() {
        let w = f32x8::splat(0.5);
        let pre = f32x8::splat(1.0);
        let post = f32x8::splat(1.0);
        let eta = f32x8::splat(0.01);
        let new = hebbian_correlation(w, pre, post, eta);
        assert!((new.to_array()[0] - 0.51).abs() < 1e-6, "Hebbian increase: {:.6}", new.to_array()[0]);
    }
}
