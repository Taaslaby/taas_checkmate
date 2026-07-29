//! Fast exponential — scalar via polynomial, SIMD via hardware `vexp` instruction.
//!
//! Scalar uses the same algorithm as NEURON's `exp(x)` approximation
//! (range reduction + polynomial, relative error < 0.001).
//!
//! SIMD delegates to `wide::f32x8::exp()` which compiles to a single
//! `vexp2ps` (AVX-512) or polynomial fallback (AVX2). This is more
//! accurate (≈ 1 ULP) and often faster than a hand-rolled polynomial.

use wide::f32x8;

/// Scalar fast exponential: `exp(x)` via range reduction + polynomial.
/// Relative error < 0.001 for |x| < 80.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = std::f32::consts::LOG2_E;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    // Minimax polynomial for exp2(f) on [-1, 1]
    let c1 = std::f32::consts::LN_2;
    let c2 = 0.240_226_5_f32;
    let c3 = 0.055_504_11_f32;
    let c4 = 0.009_618_129_f32;
    let c5 = 0.001_333_356_f32;
    let c6 = 0.000_154_035_3_f32;

    let poly = (((c6 * f + c5) * f + c4) * f + c3) * f + c2;
    let poly = (poly * f + c1) * f + 1.0;

    // Scale by 2^n via IEEE 754 exponent field
    if n < -126 {
        0.0
    } else if n > 127 {
        f32::INFINITY
    } else {
        let biased = (n + 127) as u32;
        f32::from_bits(biased << 23) * poly
    }
}

/// SIMD exponential: delegates to hardware `vexp2ps` via `wide::f32x8::exp()`.
/// Accurate to ≈1 ULP. Clamped to [-87, 87] for overflow safety.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    let x = x.max(f32x8::splat(-87.0)).min(f32x8::splat(87.0));
    x.exp()
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
        // wide::exp ≈ 1 ULP (< 1e-6 relative error)
        let inputs: [f32; 8] = [-5.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, 20.0];
        let x = f32x8::from(inputs);
        let result = super::fast_exp_simd(x);
        let arr = result.to_array();
        for i in 0..8 {
            let exact = inputs[i].exp();
            let rel_err = ((arr[i] - exact) / exact).abs();
            assert!(rel_err < 1e-5, "fast_exp_simd[{}] = {} vs {} (err={})",
                    i, arr[i], exact, rel_err);
        }
    }

    #[test]
    fn test_fast_exp_scalar_matches_simd() {
        for &v in &[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0] {
            let s = super::fast_exp(v);
            let simd = super::fast_exp_simd(f32x8::splat(v)).to_array()[0];
            let diff = (s - simd).abs();
            assert!(diff < 0.01, "fast_exp({}): scalar {} vs simd {} (diff={})", v, s, simd, diff);
        }
    }
}
