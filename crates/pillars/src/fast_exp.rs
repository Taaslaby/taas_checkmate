//! Fast exponential approximation — polynomial via range reduction.
//!
//! Uses the same algorithm as NEURON's `exp(x)` approximation:
//! 1. Range reduction: `x → t = x * log2(e)`
//! 2. Polynomial on fractional part: `exp2(f) ≈ 1 + c1*f + c2*f² + ...`
//! 3. Scale by `exp2(n) = 2^n`
//!
//! Relative error < 0.001 for |x| < 80. Independent implementation,
//! no dependencies on other crates.

use wide::f32x8;
use wide::i32x8;

/// Scalar fast exponential: `exp(x)` via range reduction + polynomial.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = std::f32::consts::LOG2_E;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    // Polynomial coefficients for exp2(f) on [-1, 1]
    let c1 = std::f32::consts::LN_2;
    let c2 = 0.240_226_5_f32;
    let c3 = 0.055_504_11_f32;
    let c4 = 0.009_618_129_f32;
    let c5 = 0.001_333_356_f32;
    let c6 = 0.000_154_035_3_f32;

    let poly = (((c6 * f + c5) * f + c4) * f + c3) * f + c2;
    let poly = (poly * f + c1) * f + 1.0;
    let exp2f = poly;

    // Scale by 2^n via IEEE 754 exponent manipulation
    let exp2n = if n < -126 {
        0.0
    } else if n > 127 {
        f32::INFINITY
    } else {
        let biased = (n + 127) as u32;
        f32::from_bits(biased << 23)
    };

    exp2n * exp2f
}

/// SIMD fast exponential: 8 neurons at once, clamped to [-87, 87] for overflow safety.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    // Clamp to prevent overflow (exp(87) ≈ 1.6e38 < f32::MAX)
    let x = x.max(f32x8::splat(-87.0)).min(f32x8::splat(87.0));

    let log2e = f32x8::splat(std::f32::consts::LOG2_E);
    let t = x * log2e;

    let n = t.floor();
    let ff = t - n;

    // Horner's method for exp2(f) polynomial
    let c1 = f32x8::splat(std::f32::consts::LN_2);
    let c2 = f32x8::splat(0.240_226_5_f32);
    let c3 = f32x8::splat(0.055_504_11_f32);
    let c4 = f32x8::splat(0.009_618_129_f32);
    let c5 = f32x8::splat(0.001_333_356_f32);
    let c6 = f32x8::splat(0.000_154_035_3_f32);

    let poly = ff.mul_add(c6, c5);
    let poly = ff.mul_add(poly, c4);
    let poly = ff.mul_add(poly, c3);
    let poly = ff.mul_add(poly, c2);
    let poly = ff.mul_add(poly, c1);
    let poly = ff.mul_add(poly, f32x8::splat(1.0));

    // Convert n to integer for IEEE 754 exponent manipulation
    let n_i32: i32x8 = n.fast_round_int();
    let biased = n_i32 + i32x8::splat(127);
    let exp2n: f32x8 = bytemuck::cast(biased << 23_u32);

    exp2n * poly
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_fast_exp_scalar_accuracy() {
        for &x in &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 5.0, -5.0] {
            let approx = super::fast_exp(x);
            let exact = x.exp();
            let rel_err = ((approx - exact) / exact).abs();
            assert!(rel_err < 0.01, "fast_exp({}) = {} vs {} (err={})", x, approx, exact, rel_err);
        }
    }

    #[test]
    fn test_fast_exp_simd_accuracy() {
        let inputs: [f32; 8] = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let x = f32x8::from(inputs);
        let result = super::fast_exp_simd(x);
        let arr = result.to_array();
        for i in 0..8 {
            let exact = inputs[i].exp();
            let rel_err = ((arr[i] - exact) / exact).abs();
            assert!(rel_err < 0.01, "fast_exp_simd[{}] = {} vs {} (err={})",
                    i, arr[i], exact, rel_err);
        }
    }

    #[test]
    fn test_fast_exp_exp_zero() {
        let approx = super::fast_exp(0.0);
        assert!((approx - 1.0).abs() < 0.001, "exp(0) = {}", approx);
    }

    #[test]
    fn test_fast_exp_overflow_clamp() {
        // exp(100) should not overflow — clamped to exp(87)
        let x = f32x8::splat(100.0);
        let result = super::fast_exp_simd(x);
        assert!(result.to_array()[0].is_finite(), "exp_simd(100) overflow: {}", result.to_array()[0]);
    }

    #[test]
    fn test_fast_exp_scalar_matches_simd() {
        let test_vals = [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &v in &test_vals {
            let s = super::fast_exp(v);
            let simd = super::fast_exp_simd(f32x8::splat(v)).to_array()[0];
            let diff = (s - simd).abs();
            assert!(diff < 0.001, "fast_exp({}): scalar {} vs simd {} (diff={})", v, s, simd, diff);
        }
    }
}
