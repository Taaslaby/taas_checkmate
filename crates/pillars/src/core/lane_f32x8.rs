//! # SimdLane for f32x8 — SIMD (AVX2 / NEON / WASM SIMD128)
//!
//! Each method maps to a single hardware instruction via `wide::f32x8`.

use super::lane::SimdLane;
use wide::{f32x8, CmpGe, CmpGt, CmpLt};

impl SimdLane for f32x8 {
    const SIZE: usize = 8;
    #[inline(always)]
    fn splat(v: f32) -> Self {
        f32x8::splat(v)
    }
    #[inline(always)]
    fn zero() -> Self {
        f32x8::splat(0.0)
    }
    #[inline(always)]
    fn one() -> Self {
        f32x8::splat(1.0)
    }

    #[inline(always)]
    fn sigmoid(v: Self, v_half: f32, k: f32) -> Self {
        crate::math::sigmoid::sigmoid(v, v_half, k)
    }
    #[inline(always)]
    fn vtrap(x: Self, y: Self) -> Self {
        crate::math::vtrap::vtrap(x, y)
    }
    #[inline(always)]
    fn ohmic(g: Self, e: f32, v: Self) -> Self {
        g * (v - Self::splat(e))
    }

    #[inline(always)]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }

    // Точный exp (wide) — только в дефолтной конфигурации; под фичей
    // `fast-exp` trait-дефолт маршрутизирует exp() на fast_exp() (полином).
    #[cfg(not(feature = "fast-exp"))]
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }
    /// Полиномиальная аппроксимация exp — бит-идентична скалярному fast_exp.
    #[inline(always)]
    fn fast_exp(self) -> Self {
        crate::math::fast_exp::fast_exp_poly_simd(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn powf(self, e: f32) -> Self {
        self.powf(e)
    }
    #[inline(always)]
    fn powi_impl<const N: usize>(self) -> Self {
        if N == 0 {
            Self::splat(1.0)
        } else if N == 1 {
            self
        } else if N == 2 {
            self * self
        } else if N == 3 {
            self * self * self
        } else if N == 4 {
            let s = self * self;
            s * s
        } else {
            let mut result = Self::splat(1.0);
            let mut base = self;
            let mut exp = N;
            while exp > 0 {
                if (exp & 1) == 1 {
                    result *= base;
                }
                base = base * base;
                exp >>= 1;
            }
            result
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn signum(self) -> Self {
        let zero = Self::splat(0.0);
        let one = Self::splat(1.0);
        let neg = Self::splat(-1.0);
        let ge_zero = CmpGe::cmp_ge(self, zero);
        let lt = CmpLt::cmp_lt(self, zero);
        lt.blend(neg, ge_zero.blend(one, zero))
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    #[inline(always)]
    fn recip(self) -> Self {
        Self::splat(1.0) / self
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        let abs_val = self.abs().powf(1.0 / 3.0);
        let is_pos = SimdLane::cmp_ge(self, Self::zero());
        SimdLane::blend(is_pos, abs_val, -abs_val)
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        let a = self.abs();
        let b = other.abs();
        let mx = a.max(b);
        let mn = a.min(b);
        // 🛡️ BUGFIX: ratio*ratio instead of powf(2.0) — transcendental function is
        // 10-100× slower than a single multiply on AVX2.
        // EPS=1e-30 prevents 0/0 NaN when both inputs are zero.
        let eps = Self::splat(1e-30);
        let ratio = mn / (mx + eps);
        mx * (Self::one() + ratio * ratio).sqrt()
    }

    #[inline(always)]
    fn round(self) -> Self {
        let pos = (self + Self::splat(0.5)).floor();
        let neg = (self - Self::splat(0.5)).ceil();
        let ge_zero = SimdLane::cmp_ge(self, Self::zero());
        SimdLane::blend(ge_zero, pos, neg)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self.ceil()
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        let ge_zero = CmpGe::cmp_ge(self, Self::zero());
        ge_zero.blend(self.floor(), self.ceil())
    }
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    #[inline(always)]
    fn blend(cond: Self, t: Self, f: Self) -> Self {
        cond.blend(t, f)
    }
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        self & other
    }
    #[inline(always)]
    fn not(self) -> Self {
        !self
    }
    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        CmpGe::cmp_ge(self, other)
    }
    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        CmpGt::cmp_gt(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[inline(always)]
    fn clamp01(self) -> Self {
        self.max(Self::zero()).min(Self::one())
    }
    #[inline(always)]
    fn move_mask(self) -> u32 {
        self.move_mask() as u32
    }

    #[inline(always)]
    unsafe fn load_param(p: *const f32, i: usize, n: usize, off: usize) -> Self {
        (p.add(i * n + off) as *const f32x8).read_unaligned()
    }
    #[inline(always)]
    fn write_to_slice(self, slice: &mut [f32]) {
        let arr = self.to_array();
        slice.copy_from_slice(&arr);
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 8] {
        self.to_array()
    }
    #[inline(always)]
    fn from_array(arr: [f32; 8]) -> Self {
        f32x8::from(arr)
    }
}
