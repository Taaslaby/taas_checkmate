//! # SimdLane for f32 — Scalar Fallback (non-aligned tails)
//!
//! Each method delegates to `std::f32` operations.
//! Masks use IEEE float bit patterns (NaN = true, 0.0 = false).

use super::lane::SimdLane;

impl SimdLane for f32 {
    const SIZE: usize = 1;
    #[inline(always)] fn splat(v: f32) -> Self { v }
    #[inline(always)] fn zero() -> Self { 0.0 }
    #[inline(always)] fn one() -> Self { 1.0 }

    #[inline(always)] fn sigmoid(v: Self, v_half: f32, k: f32) -> Self {
        crate::math::sigmoid::sigmoid_s(v, v_half, k)
    }
    #[inline(always)] fn vtrap(x: Self, y: Self) -> Self { crate::math::vtrap::vtrap_s(x, y) }
    #[inline(always)] fn ohmic(g: Self, e: f32, v: Self) -> Self { g * (v - e) }

    #[inline(always)] fn sin(self) -> Self { self.sin() }
    #[inline(always)] fn cos(self) -> Self { self.cos() }
    #[inline(always)] fn tan(self) -> Self { self.tan() }
    #[inline(always)] fn asin(self) -> Self { self.asin() }
    #[inline(always)] fn acos(self) -> Self { self.acos() }
    #[inline(always)] fn atan(self) -> Self { self.atan() }
    #[inline(always)] fn atan2(self, other: Self) -> Self { self.atan2(other) }

    // Точный exp (libm) — только в дефолтной конфигурации; под фичей
    // `fast-exp` trait-дефолт маршрутизирует exp() на fast_exp() (полином).
    #[cfg(not(feature = "fast-exp"))]
    #[inline(always)] fn exp(self) -> Self { self.exp() }
    #[inline(always)] fn fast_exp(self) -> Self { crate::math::fast_exp::fast_exp(self) }
    #[inline(always)] fn ln(self) -> Self { self.ln() }
    #[inline(always)] fn log2(self) -> Self { self.log2() }
    #[inline(always)] fn log10(self) -> Self { self.log10() }
    #[inline(always)] fn powf(self, e: f32) -> Self { self.powf(e) }
    #[inline(always)] fn powi<const N: usize>(self) -> Self {
        if N == 0 { 1.0 }
        else if N == 1 { self }
        else if N == 2 { self * self }
        else if N == 3 { self * self * self }
        else if N == 4 { let s = self * self; s * s }
        else {
            let mut result = 1.0;
            let mut base = self;
            let mut exp = N;
            while exp > 0 {
                if (exp & 1) == 1 { result *= base; }
                base = base * base;
                exp >>= 1;
            }
            result
        }
    }

    #[inline(always)] fn abs(self) -> Self { self.abs() }
    #[inline(always)] fn signum(self) -> Self { self.signum() }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { self.mul_add(a, b) }
    #[inline(always)] fn recip(self) -> Self { Self::splat(1.0) / self }
    #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)] fn cbrt(self) -> Self { self.cbrt() }
    #[inline(always)] fn hypot(self, other: Self) -> Self { self.hypot(other) }

    #[inline(always)] fn round(self) -> Self {
        let pos = (self + Self::splat(0.5)).floor();
        let neg = (self - Self::splat(0.5)).ceil();
        let ge_zero = SimdLane::cmp_ge(self, Self::zero());
        SimdLane::blend(ge_zero, pos, neg)
    }
    #[inline(always)] fn floor(self) -> Self { self.floor() }
    #[inline(always)] fn ceil(self) -> Self { self.ceil() }
    #[inline(always)] fn trunc(self) -> Self { self.trunc() }
    #[inline(always)] fn fract(self) -> Self { self.fract() }

    #[inline(always)]
    fn blend(cond: Self, t: Self, f: Self) -> Self {
        if cond.to_bits() != 0 { t } else { f }
    }
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        f32::from_bits(self.to_bits() & other.to_bits())
    }
    #[inline(always)]
    fn not(self) -> Self { f32::from_bits(!self.to_bits()) }
    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self {
        if self >= other { f32::from_bits(u32::MAX) } else { f32::from_bits(0) }
    }
    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self {
        if self > other { f32::from_bits(u32::MAX) } else { f32::from_bits(0) }
    }
    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn clamp01(self) -> Self { self.clamp(0.0, 1.0) }
    #[inline(always)]
    fn move_mask(self) -> u32 {
        if self.to_bits() != 0 { 1 } else { 0 }
    }

    #[inline(always)]
    unsafe fn load_param(p: *const f32, i: usize, n: usize, off: usize) -> Self {
        *p.add(i * n + off)
    }
    #[inline(always)]
    fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self;
    }

    #[inline(always)] fn to_array(self) -> [f32; 8] {
        let mut a = [0.0f32; 8]; a[0] = self; a
    }
    #[inline(always)] fn from_array(arr: [f32; 8]) -> Self { arr[0] }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splat_zero_one() {
        assert_eq!(f32::splat(std::f32::consts::PI), std::f32::consts::PI);
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
    }

    #[test]
    fn test_cmp_ge() {
        let t = f32::cmp_ge(5.0, 3.0);
        assert_ne!(t.to_bits(), 0, "5>=3 should be true");
        let f = f32::cmp_ge(2.0, 3.0);
        assert_eq!(f.to_bits(), 0, "2>=3 should be false");
    }

    #[test]
    fn test_cmp_gt() {
        let t = f32::cmp_gt(5.0, 3.0);
        assert_ne!(t.to_bits(), 0);
        let f = f32::cmp_gt(3.0, 3.0);
        assert_eq!(f.to_bits(), 0);
    }

    #[test]
    fn test_blend() {
        let t = f32::cmp_ge(5.0, 3.0);
        let result = f32::blend(t, 10.0, -10.0);
        assert!((result - 10.0).abs() < 1e-6, "blend true should pick 10: {result}");

        let f = f32::cmp_ge(2.0, 3.0);
        let result2 = f32::blend(f, 10.0, -10.0);
        assert!((result2 - (-10.0)).abs() < 1e-6, "blend false should pick -10: {result2}");
    }

    #[test]
    fn test_and_not() {
        let mask = f32::cmp_ge(5.0, 3.0); // all true
        let other = f32::splat(2.0);
        let masked = f32::and(mask, other);
        assert_ne!(masked.to_bits(), 0, "and with true should keep bits");

        let notted = f32::not(mask);
        assert_eq!(notted.to_bits(), 0, "not true should be false");
    }

    #[test]
    fn test_max_min_clamp01() {
        assert_eq!(f32::max(5.0, 3.0), 5.0);
        assert_eq!(f32::min(5.0, 3.0), 3.0);
        assert!((f32::clamp01(1.5) - 1.0).abs() < 1e-6);
        assert!((f32::clamp01(-0.5) - 0.0).abs() < 1e-6);
        assert!((f32::clamp01(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_trig_functions() {
        let x: f32 = 0.5;
        assert!((f32::sin(x) - x.sin()).abs() < 1e-6);
        assert!((f32::cos(x) - x.cos()).abs() < 1e-6);
        assert!((f32::tan(x) - x.tan()).abs() < 1e-6);
    }

    #[test]
    fn test_transcendental() {
        let x: f32 = 2.0;
        assert!((f32::exp(x) - x.exp()).abs() < 1e-5);
        assert!((f32::ln(x) - x.ln()).abs() < 1e-6);
        assert!((f32::log2(x) - x.log2()).abs() < 1e-6);
        assert!((f32::log10(x) - x.log10()).abs() < 1e-6);
        assert!((f32::powf(x, 3.0) - x.powf(3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_powi() {
        let x: f32 = 2.0;
        // Call through trait to avoid conflict with f32::powi(i32)
        use super::SimdLane;
        assert!((SimdLane::powi::<0>(x) - 1.0).abs() < 1e-6, "powi<0>");
        assert!((SimdLane::powi::<1>(x) - 2.0).abs() < 1e-6, "powi<1>");
        assert!((SimdLane::powi::<2>(x) - 4.0).abs() < 1e-6, "powi<2>");
        assert!((SimdLane::powi::<3>(x) - 8.0).abs() < 1e-6, "powi<3>");
        assert!((SimdLane::powi::<4>(x) - 16.0).abs() < 1e-6, "powi<4>");
        assert!((SimdLane::powi::<10>(x) - 1024.0).abs() < 1e-4, "powi<10>");
    }

    #[test]
    fn test_round_floor_ceil() {
        assert!((f32::round(2.5) - 3.0).abs() < 1e-6, "round 2.5");
        assert!((f32::round(-2.5) - (-3.0)).abs() < 1e-6, "round -2.5");
        assert!((f32::floor(2.7) - 2.0).abs() < 1e-6);
        assert!((f32::ceil(2.3) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_abs_signum() {
        assert!((f32::abs(-5.0) - 5.0).abs() < 1e-6);
        assert!((f32::abs(5.0) - 5.0).abs() < 1e-6);
        assert!(((-5.0f32).signum() - (-1.0)).abs() < 1e-6);
        assert!(((5.0f32).signum() - 1.0).abs() < 1e-6);
        // signum(0) = 1.0 (std::f32::signum returns 1.0 for +0.0)
        assert!(((0.0f32).signum() - 1.0).abs() < 1e-6, "signum(0)");
    }

    #[test]
    fn test_mul_add() {
        // 2.0 * 3.0 + 4.0 = 10.0
        assert!((f32::mul_add(2.0, 3.0, 4.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_cbrt_hypot() {
        assert!((f32::sqrt(4.0) - 2.0).abs() < 1e-6);
        assert!((f32::cbrt(27.0) - 3.0).abs() < 1e-5, "cbrt(27)");
        assert!((f32::hypot(3.0, 4.0) - 5.0).abs() < 1e-5, "hypot(3,4)");
    }

    #[test]
    fn test_move_mask() {
        let t = f32::cmp_ge(5.0, 3.0);
        assert_eq!(f32::move_mask(t), 1);
        let f = f32::cmp_ge(2.0, 3.0);
        assert_eq!(f32::move_mask(f), 0);
    }

    #[test]
    fn test_recip() {
        assert!((f32::recip(2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ohmic_sigmoid_vtrap() {
        // ohmic: g * (v - e)
        assert!((f32::ohmic(0.3, -65.0, -50.0) - 4.5).abs() < 1e-6);
        // sigmoid at v_half with k=1 → should be 0.5
        let sig = f32::sigmoid(-60.0, -60.0, 1.0);
        assert!((sig - 0.5).abs() < 1e-5, "sigmoid at v_half should be 0.5: {sig}");
        // vtrap: x/(exp(x/y)-1) with x=0 → 0/(exp(0)-1) → needs vtrap to diverge
        let vt = f32::vtrap(1e-10, 1.0);
        assert!(vt.is_finite(), "vtrap small x should be finite: {vt}");
        // vtrap with normal values
        let vt2 = f32::vtrap(10.0, 5.0);
        assert!(vt2.is_finite());
    }

    #[test]
    fn test_trunc_fract() {
        assert!((f32::trunc(3.7) - 3.0).abs() < 1e-6);
        assert!((f32::trunc(-3.7) - (-3.0)).abs() < 1e-6);
        assert!((f32::fract(3.7) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_load_param_write_to_slice() {
        // Simulate SoA layout: n=5 neurons, 3 param columns → 5*3=15 elements
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, // col0: dt
                        0.1, 0.2, 0.3, 0.4, 0.5, // col1: inv_tau
                        1.0, 2.0, 3.0, 4.0, 5.0]; // col2: v_rest
        unsafe {
            // load_param(col=2, n=5, off=0) → reads data[2*5+0]=data[10]=1.0
            let v = f32::load_param(data.as_ptr(), 2, 5, 0);
            assert!((v - 1.0).abs() < 1e-6, "load_param col=2: expected 1.0, got {v}");
            // off=3 → data[2*5+3]=data[13]=4.0
            let v3 = f32::load_param(data.as_ptr(), 2, 5, 3);
            assert!((v3 - 4.0).abs() < 1e-6, "load_param col=2 off=3: expected 4.0, got {v3}");
        }
        let mut out = [0.0f32; 8];
        f32::write_to_slice(42.0, &mut out);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn test_to_from_array() {
        let a = f32::to_array(std::f32::consts::PI);
        assert_eq!(a[0], std::f32::consts::PI);
        assert_eq!(a[1], 0.0);
        assert_eq!(f32::from_array(a), std::f32::consts::PI);
    }
}

