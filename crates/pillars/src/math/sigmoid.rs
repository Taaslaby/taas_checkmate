//! Sigmoid — voltage-dependent gating variable.
//!
//! `1 / (1 + exp(-(V - V_half) / k))`
//!
//! Standard form for m_∞, h_∞, n_∞ in conductance-based models.
//! Uses `wide::f32x8::exp()` (hardware) for SIMD, `std::f32::exp()` for scalar.

use wide::f32x8;

/// SIMD sigmoid: `1 / (1 + exp(-(V - V_half) / k))`
#[inline(always)]
pub fn sigmoid(v: f32x8, v_half: f32, k: f32) -> f32x8 {
    let one = f32x8::splat(1.0);
    let arg = -(v - f32x8::splat(v_half)) / f32x8::splat(k);
    one / (one + arg.exp())
}

/// Scalar sigmoid.
#[inline(always)]
pub fn sigmoid_s(v: f32, v_half: f32, k: f32) -> f32 {
    1.0 / (1.0 + (-(v - v_half) / k).exp())
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_sigmoid_at_midpoint() {
        let s = super::sigmoid(f32x8::splat(-35.0), -35.0, 5.0);
        assert!((s.to_array()[0] - 0.5).abs() < 0.01, "mid: {}", s.to_array()[0]);
    }

    #[test]
    fn test_sigmoid_saturation() {
        let s1 = super::sigmoid(f32x8::splat(50.0), -35.0, 5.0);
        assert!((s1.to_array()[0] - 1.0).abs() < 0.01, "high: {}", s1.to_array()[0]);

        let s2 = super::sigmoid(f32x8::splat(-80.0), -35.0, 5.0);
        assert!(s2.to_array()[0] < 0.01, "low: {}", s2.to_array()[0]);
    }

    #[test]
    fn test_sigmoid_scalar_matches_simd() {
        let s = super::sigmoid_s(-35.0, -35.0, 5.0);
        let simd = super::sigmoid(f32x8::splat(-35.0), -35.0, 5.0).to_array()[0];
        assert!((s - simd).abs() < 0.001);
    }
}
