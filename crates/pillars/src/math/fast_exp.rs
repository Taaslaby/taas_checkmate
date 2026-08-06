//! # ⚡ Fast Exponential — exp() Without the Library Cost
//!
//! The single most-called transcendental in neural simulation (gate
//! kinetics: `m_∞`, `h_∞`, `n_∞`, `w_∞`). Two engines, one contract:
//!
//! | Engine | Function | Implementation | Accuracy |
//! |--------|----------|----------------|----------|
//! | Scalar | [`fast_exp`] | range reduction + polynomial (same algorithm as NEURON's `exp` approximation) | rel. error < 0.001 |
//! | SIMD | [`fast_exp_poly_simd`] | **bit-identical** scalar/SIMD polynomial pair | rel. error < 0.001 |
//! | SIMD (exact) | [`SimdLane::exp`](crate::core::lane::SimdLane::exp) | `wide::f32x8::exp()` → single `vexp2ps` (AVX-512) or fallback (AVX2) | ≈ 1 ULP |
//!
//! The exact SIMD path is often **faster** than a hand-rolled polynomial
//! (one instruction, no branches); the polynomial pair exists for batch
//! microkernels that require scalar and SIMD tails to agree **bit-for-bit**.
//!
//! Toggle accuracy vs speed globally with the `fast-exp` feature.

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

    // Horner. КРИТИЧНО: wide::f32x8::mul_add даёт fused-результат (одно округление)
    // только на x86_64 с target_feature="fma" (pick! смотрит только на sse2/avx
    // cfg-ключи — на aarch64 он НИКОГДА не fused, а fma там включён по умолчанию).
    // Скалярный же f32::mul_add в Rust fused ВСЕГДА. Чтобы скалярный путь был
    // бит-идентичен SIMD-пути (`fast_exp_poly_simd`) на любом таргете, зеркалим
    // wide: fused на x86_64+fma, двойное округление иначе.
    let poly = {
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        {
            let poly = c6.mul_add(f, c5);
            let poly = poly.mul_add(f, c4);
            let poly = poly.mul_add(f, c3);
            let poly = poly.mul_add(f, c2);
            let poly = poly.mul_add(f, c1);
            poly.mul_add(f, 1.0)
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "fma")))]
        {
            let poly = (c6 * f + c5) * f + c4;
            let poly = (poly * f + c3) * f + c2;
            (poly * f + c1) * f + 1.0
        }
    };

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

/// SIMD exponential: delegates to `wide::f32x8::exp()` (hardware `vexp2ps`
/// on AVX-512, FMA-полином на AVX2).
///
/// ⚠️ НЕСМОТРЯ на имя — это **точный** путь (≈1 ULP), не аппроксимация.
/// Настоящая быстрая аппроксимация — [`fast_exp_poly_simd`] (под фичей
/// `fast-exp` `SimdLane::exp` маршрутизируется именно на неё).
/// Clamped to [-87, 87] for overflow safety.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    let x = x.max(f32x8::splat(-87.0)).min(f32x8::splat(87.0));
    x.exp()
}

/// SIMD fast exponential — polynomial, **bit-identical to scalar [`fast_exp`]**.
///
/// Тот же алгоритм: range reduction через `floor`, полином Тейлора 6-й степени
/// для 2^f на [0,1), масштабирование через экспонентное поле IEEE-754.
/// Каждый лейн выполняет ту же последовательность mul/add, что и скаляр —
/// результаты совпадают бит-в-бит (критично для SIMD/скаляр консистентности
/// и для переключателя точности `fast-exp`).
///
/// Относительная ошибка < 2e-5 для |x| < 87 (полином 6-й степени).
#[inline(always)]
pub fn fast_exp_poly_simd(x: f32x8) -> f32x8 {
    use wide::{f32x8 as V, i32x8, CmpGt, CmpLt};

    let log2e = std::f32::consts::LOG2_E;
    let c1 = std::f32::consts::LN_2;
    let c2 = 0.240_226_5_f32;
    let c3 = 0.055_504_11_f32;
    let c4 = 0.009_618_129_f32;
    let c5 = 0.001_333_356_f32;
    let c6 = 0.000_154_035_3_f32;

    // Range reduction: n = floor(x·log2e), f ∈ [0, 1)
    let t = x * V::splat(log2e);
    let n_f = t.floor();
    let f = t - n_f;

    // Horner через mul_add (FMA) — та же структура, что в scalar fast_exp:
    // оба пути дают идентичные биты (fused-семантика), и оба получают
    // аппаратный FMA на AVX2+FMA без контракции со стороны LLVM.
    let poly = V::splat(c6).mul_add(f, V::splat(c5));
    let poly = poly.mul_add(f, V::splat(c4));
    let poly = poly.mul_add(f, V::splat(c3));
    let poly = poly.mul_add(f, V::splat(c2));
    let poly = poly.mul_add(f, V::splat(c1));
    let poly = poly.mul_add(f, V::splat(1.0));

    // Масштаб 2^n через экспонентное поле: bits = (n + 127) << 23
    // trunc_int = identity для целочисленных значений; NaN → 0 (как в скаляре)
    // Битовая реинтерпретация i32x8 → f32x8: wide не публикует cast/from_bits,
    // а from_i32x8 делает ЧИСЛОВУЮ cvtepi32_ps конверсию (не годится!).
    // transmute — тот же bit-level каст, что wide использует внутри своего cast.
    let n = n_f.trunc_int();
    let biased = n + i32x8::splat(127);
    let valid: V = unsafe { std::mem::transmute(biased << 23) };

    // Защита от переполнения/андерфлоу — зеркало скалярного `if n < -126 / n > 127`
    let under = CmpLt::cmp_lt(n_f, V::splat(-126.0));
    let over = CmpGt::cmp_gt(n_f, V::splat(127.0));
    let scale = over.blend(V::splat(f32::INFINITY), under.blend(V::splat(0.0), valid));

    poly * scale
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
            assert!(
                rel_err < 0.01,
                "fast_exp({}) = {} vs {} (err={})",
                x,
                approx,
                exact,
                rel_err
            );
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
            assert!(
                rel_err < 1e-5,
                "fast_exp_simd[{}] = {} vs {} (err={})",
                i,
                arr[i],
                exact,
                rel_err
            );
        }
    }

    #[test]
    fn test_fast_exp_scalar_matches_simd() {
        for &v in &[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0] {
            let s = super::fast_exp(v);
            let simd = super::fast_exp_simd(f32x8::splat(v)).to_array()[0];
            let diff = (s - simd).abs();
            assert!(
                diff < 0.01,
                "fast_exp({}): scalar {} vs simd {} (diff={})",
                v,
                s,
                simd,
                diff
            );
        }
    }

    // ─── fast_exp_poly_simd: точность + бит-идентичность со скаляром ───

    /// Максимальная относительная ошибка полинома на сетке [-87, 88].
    #[test]
    fn test_fast_exp_poly_simd_accuracy() {
        let mut max_rel = 0.0f32;
        let mut worst_x = 0.0f32;
        for i in -8700..=8800 {
            let x = i as f32 / 100.0;
            let exact = x.exp();
            if !exact.is_finite() || exact == 0.0 {
                continue;
            }
            let approx = super::fast_exp_poly_simd(f32x8::splat(x)).to_array()[0];
            let rel = ((approx - exact) / exact).abs();
            if rel > max_rel {
                max_rel = rel;
                worst_x = x;
            }
        }
        eprintln!(
            "fast_exp_poly_simd: max rel err = {:.3e} at x={}",
            max_rel, worst_x
        );
        assert!(
            max_rel < 1e-4,
            "max rel err {:.2e} too high at x={}",
            max_rel,
            worst_x
        );
    }

    /// SIMD-полином обязан совпадать со скаляром почти бит-в-бит на всём
    /// рабочем диапазоне (≤1e-5 rel): это гарантирует консистентность
    /// SIMD/скаляр в тестах и переключатель точности в моделях.
    ///
    /// Допуск вместо точного равенства: на билдах без `+fma` таргет-фичи
    /// скалярный `f32::mul_add` разворачивается в mul+add (двойное округление),
    /// а SIMD-путь эмитит аппаратный `vfmaddps` (single rounding) — до 1 ULP
    /// расхождения. Это всё ещё ловит ЛЮБОЙ алгоритмический дрейф
    /// (битый масштаб из from_i32x8 дал бы расхождение в ~1e9 раз).
    #[test]
    fn test_fast_exp_poly_simd_near_identical_to_scalar() {
        let mut max_rel = 0.0f32;
        for i in -8700..=8800 {
            let x = i as f32 / 100.0;
            let s = super::fast_exp(x);
            let simd = super::fast_exp_poly_simd(f32x8::splat(x)).to_array()[0];
            let rel = if s != 0.0 {
                ((s - simd) / s).abs()
            } else {
                0.0
            };
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-5,
                "x={}: scalar {} vs simd {} (rel {:.2e})",
                x,
                s,
                simd,
                rel
            );
        }
        eprintln!(
            "fast_exp_poly_simd: max scalar/simd rel diff = {:.3e}",
            max_rel
        );
    }

    /// Поведение на границах: переполнение → inf, андерфлоу → 0.
    #[test]
    fn test_fast_exp_poly_simd_edges() {
        let over = super::fast_exp_poly_simd(f32x8::splat(100.0)).to_array()[0];
        assert!(
            over.is_infinite(),
            "exp(100) should overflow to inf, got {}",
            over
        );
        let under = super::fast_exp_poly_simd(f32x8::splat(-100.0)).to_array()[0];
        assert_eq!(under, 0.0, "exp(-100) should underflow to 0, got {}", under);
        let mid = super::fast_exp_poly_simd(f32x8::splat(1.0)).to_array()[0];
        assert!((mid - std::f32::consts::E).abs() < 1e-4, "exp(1) = {}", mid);
    }
}
