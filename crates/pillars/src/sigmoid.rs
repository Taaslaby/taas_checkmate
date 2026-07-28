//! Sigmoid — voltage-dependent gating variable.
//!
//! Common form: `1 / (1 + exp(-(V - V_half) / k))`
//!
//! Used for steady-state activation/inactivation of ion channels
//! (h_inf, m_inf, n_inf, u_inf, etc.)

use wide::f32x8;
use crate::fast_exp_simd;

/// SIMD sigmoid: `1 / (1 + exp(-(V - V_half) / k))`
/// Processes 8 neurons at once.
#[inline(always)]
pub fn sigmoid(v: f32x8, v_half: f32, k: f32) -> f32x8 {
    let vh = f32x8::splat(v_half);
    let kv = f32x8::splat(k);
    let one = f32x8::splat(1.0);
    let exp_arg = -(v - vh) / kv;
    one / (one + fast_exp_simd(exp_arg))
}

/// Scalar sigmoid: `1 / (1 + exp(-(V - V_half) / k))`
#[inline(always)]
pub fn sigmoid_s(v: f32, v_half: f32, k: f32) -> f32 {
    let exp_arg = -(v - v_half) / k;
    1.0 / (1.0 + crate::fast_exp(exp_arg))
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_sigmoid_at_vhalf() {
        let v = f32x8::splat(-35.0);
        let s = super::sigmoid(v, -35.0, 5.0);
        let a = s.to_array();
        assert!((a[0] - 0.5).abs() < 0.01, "sigmoid at Vh: {}", a[0]);
    }

    #[test]
    fn test_sigmoid_saturation() {
        // At V >> V_half: sigmoid → 1
        let v = f32x8::splat(50.0);
        let s = super::sigmoid(v, -35.0, 5.0);
        assert!((s.to_array()[0] - 1.0).abs() < 0.01, "sigmoid sat: {}", s.to_array()[0]);

        // At V << V_half: sigmoid → 0
        let v2 = f32x8::splat(-80.0);
        let s2 = super::sigmoid(v2, -35.0, 5.0);
        assert!(s2.to_array()[0] < 0.01, "sigmoid low: {}", s2.to_array()[0]);
    }

    #[test]
    fn test_sigmoid_scalar_matches_simd() {
        let v = -35.0_f32;
        let s_s = super::sigmoid_s(v, -35.0, 5.0);
        let s_simd = super::sigmoid(f32x8::splat(v), -35.0, 5.0).to_array()[0];
        assert!((s_s - s_simd).abs() < 0.001, "scalar vs SIMD: {} vs {}", s_s, s_simd);
    }
}
