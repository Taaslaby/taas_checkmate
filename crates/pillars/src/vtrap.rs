//! HH vtrap — the gold-standard auxiliary function for Hodgkin-Huxley rate equations.
//!
//! ```text
//! vtrap(x, y) = x / (exp(x/y) - 1)   when |x/y| > 1e-6
//!               y * (1 - x/(2*y))     when |x/y| ≤ 1e-6  (Taylor expansion)
//! ```
//!
//! Used in HH α(V)/β(V) rate functions to avoid numerical instability at V ≈ 0.
//! See NEURON's `vtrap` in `hoc.cpp` for the canonical implementation.

use wide::f32x8;

const VTAP_EPS: f32 = 1e-6;

/// SIMD vtrap: `x / (exp(x/y) - 1)`, 8 neurons at once.
///
/// Uses Taylor expansion `y * (1 - x/(2*y))` when `|x/y| < 1e-6`
/// to avoid 0/0 division at the origin.
#[inline(always)]
pub fn vtrap(x: f32x8, y: f32x8) -> f32x8 {
    let x_div_y = x / y;

    // Taylor expansion for small |x/y| to avoid 0/0
    let taylor = y * (f32x8::splat(1.0) - x / (f32x8::splat(2.0) * y));

    // Large argument: use exact formula
    let exact = x / (crate::fast_exp_simd(x_div_y) - f32x8::splat(1.0));

    // Blend per-element: Taylor when |x/y| < eps, exact otherwise
    let x_div_y_arr: [f32; 8] = x_div_y.to_array();
    let t_arr: [f32; 8] = taylor.to_array();
    let e_arr: [f32; 8] = exact.to_array();
    let out: [f32; 8] = core::array::from_fn(|i| {
        if x_div_y_arr[i].abs() < VTAP_EPS { t_arr[i] } else { e_arr[i] }
    });
    f32x8::from(out)
}

/// Scalar vtrap: `x / (exp(x/y) - 1)`
#[inline(always)]
pub fn vtrap_s(x: f32, y: f32) -> f32 {
    if x.abs() / y.abs() < VTAP_EPS {
        y * (1.0 - x / (2.0 * y))
    } else {
        x / (crate::fast_exp(x / y) - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_vtrap_at_zero() {
        // vtrap(0, 10) = y = 10 (Taylor limit)
        let x = f32x8::splat(0.0);
        let y = f32x8::splat(10.0);
        let r = super::vtrap(x, y);
        assert!((r.to_array()[0] - 10.0).abs() < 0.01, "vtrap(0,10) = {}", r.to_array()[0]);
    }

    #[test]
    fn test_vtrap_large() {
        // vtrap(10, 1) = 10 / (exp(10) - 1) ≈ 0.00045
        let x = f32x8::splat(10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = 10.0 / (10.0_f32.exp() - 1.0);
        assert!((r.to_array()[0] - expected).abs() < 0.001, "vtrap(10,1) = {} vs {}", r.to_array()[0], expected);
    }

    #[test]
    fn test_vtrap_negative() {
        // vtrap(-10, 1) = -10 / (exp(-10) - 1) ≈ 10.0004
        let x = f32x8::splat(-10.0);
        let y = f32x8::splat(1.0);
        let r = super::vtrap(x, y);
        let expected = -10.0 / ((-10.0_f32).exp() - 1.0);
        assert!((r.to_array()[0] - expected).abs() < 0.001, "vtrap(-10,1) = {} vs {}", r.to_array()[0], expected);
    }

    #[test]
    fn test_vtrap_scalar_matches_simd() {
        let x = 0.1_f32;
        let y = 10.0_f32;
        let s = super::vtrap_s(x, y);
        let simd = super::vtrap(f32x8::splat(x), f32x8::splat(y)).to_array()[0];
        assert!((s - simd).abs() < 0.001, "scalar {} vs simd {}", s, simd);
    }

    #[test]
    fn test_vtrap_numerical_stability() {
        // Ensure no NaN at exact zero for a range of values
        for x in [-0.01, -0.001, 0.0, 0.001, 0.01] {
            let r = super::vtrap_s(x, 10.0);
            assert!(r.is_finite(), "vtrap({}, 10) = {} is not finite", x, r);
        }
    }


}
