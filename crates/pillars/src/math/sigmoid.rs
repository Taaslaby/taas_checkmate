//! # 📈 Sigmoid — Voltage-Dependent Gating Variable
//!
//! ```text
//!          1
//!  s(V) = ────────────────────   (m_∞, h_∞, n_∞, w_∞ in conductance models)
//!         1 + exp(-(V - V_half) / k)
//! ```
//!
//! The **single source of truth** for the sigmoid formula, served in two
//! canonical flavours:
//!
//! | Variant | Input | exp engine | Used by |
//! |---------|-------|-----------|---------|
//! | [`sigmoid`] / [`sigmoid_s`] | lane (`S: SimdLane`) | [`SimdLane::exp`] (exact; polynomial under the `fast-exp` feature) | `ops::gate::sigmoid_gate`, zoo-модели |
//! | [`sigmoid_poly_simd`] / [`sigmoid_poly_s`] | flat `f32x8` / `f32` | [`fast_exp_poly_simd`] / [`fast_exp`] — **bit-identical** | batch-микроядра (`kernels::signal`) |
//!
//! The polynomial pair exists for batch microkernels, where the SIMD lane
//! and the scalar tail must produce **bit-identical** results.

use wide::f32x8;
use crate::core::lane::SimdLane;
use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd};

/// SIMD sigmoid: `1 / (1 + exp(-(V - V_half) / k))`
///
/// `exp` идёт через [`SimdLane::exp`] (а не инхерентный `wide::exp`), чтобы
/// переключатель точности `fast-exp` покрывал и сигмоиды.
#[inline(always)]
pub fn sigmoid(v: f32x8, v_half: f32, k: f32) -> f32x8 {
    let one = f32x8::splat(1.0);
    let arg = -(v - f32x8::splat(v_half)) / f32x8::splat(k);
    one / (one + <f32x8 as SimdLane>::exp(arg))
}

/// Scalar sigmoid.
///
/// `exp` через [`SimdLane::exp`] — чтобы под фичей `fast-exp` скалярный путь
/// использовал ту же аппроксимацию, что и SIMD (иначе консистентность ломается).
#[inline(always)]
pub fn sigmoid_s(v: f32, v_half: f32, k: f32) -> f32 {
    1.0 / (1.0 + <f32 as SimdLane>::exp(-(v - v_half) / k))
}

// ═══════════════════════════════════════════════════════════════
//  Batch-poly flavour (bit-identical SIMD lane / scalar tail)
// ═══════════════════════════════════════════════════════════════
//
//  Эти две функции — ЕДИНСТВЕННОЕ место, где batch-сигмоида живёт как
//  формула. `kernels::signal` (и любые другие batch-кернели) обязаны
//  вызывать их, а не дублировать `1/(1+exp(-x))` руками.

/// SIMD batch sigmoid: `σ(x) = 1 / (1 + exp(−x))` через полиномиальный exp.
///
/// Использует [`fast_exp_poly_simd`] —
/// побитово идентичен скалярному хвосту [`sigmoid_poly_s`].
#[inline(always)]
pub fn sigmoid_poly_simd(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + fast_exp_poly_simd(-x))
}

/// Scalar batch sigmoid: `σ(x) = 1 / (1 + exp(−x))` через [`fast_exp`].
///
/// Побитово идентичен SIMD-лейну [`sigmoid_poly_simd`].
#[inline(always)]
pub fn sigmoid_poly_s(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
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

    #[test]
    fn test_sigmoid_poly_simd_scalar_identical() {
        // Полиномиальная пара должна быть побитово идентична SIMD/скаляр.
        let xs = [-10.0f32, -3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0];
        let simd = super::sigmoid_poly_simd(f32x8::from(xs));
        for (i, &x) in xs.iter().enumerate() {
            let scalar = super::sigmoid_poly_s(x);
            assert!(
                simd.to_array()[i].to_bits() == scalar.to_bits(),
                "poly sigmoid lane {i} (x={x}): SIMD={:e} != scalar={:e}",
                simd.to_array()[i], scalar
            );
        }
    }

    #[test]
    fn test_sigmoid_poly_midpoint() {
        assert!((super::sigmoid_poly_s(0.0) - 0.5).abs() < 1e-6, "σ(0)=0.5");
    }
}
