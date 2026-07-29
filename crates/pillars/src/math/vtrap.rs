//! HH vtrap — the gold-standard auxiliary function for Hodgkin-Huxley rate equations.
//!
//! ```text
//! vtrap(x, y) = x / (exp(x/y) - 1)   when |x/y| > 1e-6
//!               y * (1 - x/(2*y))     when |x/y| ≤ 1e-6  (Taylor expansion)
//! ```
//!
//! See NEURON's `vtrap` in `hoc.cpp`. Maintains numerical stability at V ≈ 0.

use wide::{f32x8, CmpGt};

const VTAP_EPS: f32 = 1e-6;

/// SIMD vtrap: 8 neurons at once.
/// Taylor expansion at |x/y| < 1e-6 prevents 0/0 division.
#[inline(always)]
pub fn vtrap(x: f32x8, y: f32x8) -> f32x8 {
    let eps = f32x8::splat(VTAP_EPS);
    let ratio = x / y;
    let small = eps.cmp_gt(ratio.abs());

    let taylor = y * (f32x8::splat(1.0) - ratio * f32x8::splat(0.5));
    let exact = x / (ratio.exp() - f32x8::splat(1.0));

    small.blend(taylor, exact)
}

/// Scalar vtrap: safe for all x, y ≠ 0.
#[inline(always)]
pub fn vtrap_s(x: f32, y: f32) -> f32 {
    if x.abs() / y.abs() < VTAP_EPS {
        y * (1.0 - x / (2.0 * y))
    } else {
        x / ((x / y).exp() - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_vtrap_at_zero() {
        let x = f32x8::splat(0.0);
        let y = f32x8::splat(10.0);
        let r = super::vtrap(x, y);
        assert!((r.to_array()[0] - 10.0).abs() < 0.01, "vtrap(0,10) = {}", r.to_array()[0]);
    }

    #[test]
    fn test_vtrap_large() {
        let x = f32x8::splat(10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = 10.0 / (10.0_f32.exp() - 1.0);
        assert!((r.to_array()[0] - expected).abs() < 0.001, "vtrap(10,1) = {} vs {}", r.to_array()[0], expected);
    }

    #[test]
    fn test_vtrap_negative() {
        let x = f32x8::splat(-10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = -10.0 / ((-10.0_f32).exp() - 1.0);
        assert!((r.to_array()[0] - expected).abs() < 0.001, "vtrap(-10,1) = {} vs {}", r.to_array()[0], expected);
    }

    #[test]
    fn test_vtrap_numerical_stability() {
        for x in [-0.01, -0.001, 0.0, 0.001, 0.01] {
            let r = super::vtrap_s(x, 10.0);
            assert!(r.is_finite(), "vtrap({}, 10) = {} is not finite", x, r);
        }
    }
}
