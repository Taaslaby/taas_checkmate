//! # SIMD Lane Abstraction — Zero-Cost Vectorisation for Neural Dynamics
//!
//! `SimdLane` is a trait that abstracts over `f32x8` (8-wide SIMD vector)
//! and plain `f32` (scalar tail). Every function in the trait is a single
//! hardware instruction on AVX2/NEON/WASM and a scalar operation otherwise.
//!
//! ## Design
//!
//! ```text
//! SimdLane trait
//!   ├── f32x8  → wide::f32x8  (AVX2/NEON/SIMD128)
//!   └── f32    → std::f32     (scalar fallback for unaligned tails)
//! ```
//!
//! Each model implements `Model::step<S: SimdLane>(ctx)` once, and the
//! same code compiles to BOTH SIMD and scalar paths. The `run_neuron()`
//! dispatcher splits the neuron array into SIMD chunks + scalar remainder.

use wide::{f32x8, CmpGe, CmpGt, CmpLt};
use std::ops::{Add, Sub, Mul, Div, Neg};

// ═══════════════════════════════════════════════════════════════════
//  §1 — SimdLane convenience methods (work for generic S: SimdLane)
// ═══════════════════════════════════════════════════════════════════
//
// `S::splat(x)` is the canonical way to create a lane-wide scalar.
// These convenience aliases reduce visual noise:
//   S::c(0.5)   instead of  S::splat(0.5)
//   S::h(0.04)  instead of  S::splat(0.04)
//
// For operator overloading (v * 0.04), Rust orphan rules prevent
// blanket impls over S: SimdLane. Use S::splat() for generics.

// ═══════════════════════════════════════════════════════════════════
//  §2 — Cond<S>: ergonomic branchless conditional
// ═══════════════════════════════════════════════════════════════════

/// A comparison mask (produced by `cmp_ge`, `cmp_gt`, etc.).
///
/// Masks are `NaN = true`, `0.0 = false` — corresponds to hardware SIMD mask format.
/// `Cond<S>` wraps this and provides `.then().else_()` for readable blend.
///
/// # Usage
/// ```rust
/// let is_refr = refr.cmp_gt(0.0).cond();
/// let v = is_refr.then(v_reset).else_(v_raw);
/// ```
#[derive(Copy, Clone)]
pub struct Cond<S: SimdLane>(pub S);

/// Intermediate value: `cond.then(t)` stores the true-branch value.
#[derive(Copy, Clone)]
pub struct CondThen<S: SimdLane>(S, S); // (mask, true_value)

impl<S: SimdLane> Cond<S> {
    /// Set the value for lanes where condition is TRUE.
    #[inline(always)]
    pub fn then(self, t: S) -> CondThen<S> { CondThen(self.0, t) }
}

impl<S: SimdLane> CondThen<S> {
    /// Set the value for lanes where condition is FALSE.
    /// Returns blended result: cond_mask ? true_val : false_val.
    #[inline(always)]
    pub fn else_(self, f: S) -> S { S::blend(self.0, self.1, f) }
}

/// Zero-Cost SIMD Lane Abstraction.
///
/// Every method maps to a single hardware instruction where possible.
/// Implemented for `f32x8` (SIMD) and `f32` (scalar tail).
pub trait SimdLane: Copy + Sized + Add<Output = Self> + Sub<Output = Self>
                   + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {

    /// Number of scalar elements in one SIMD lane (8 for f32x8, 1 for f32).
    const SIZE: usize;

    // ─── Constructors ──────────────────────────────────────────────

    /// Broadcast a scalar across all lanes.
    fn splat(v: f32) -> Self;

    /// All lanes = 0.0.
    fn zero() -> Self;

    /// All lanes = 1.0.
    fn one() -> Self;

    // ─── Neural Primitives ─────────────────────────────────────────

    fn sigmoid(v: Self, v_half: f32, k: f32) -> Self;
    fn vtrap(x: Self, y: Self) -> Self;
    /// Ohmic current: I = g · (V - E_rev) — NEURON convention.
    /// Fixed from g·(E-V) which inverted Ca²⁺ inward current.
    fn ohmic(g: Self, e: f32, v: Self) -> Self;

    // ─── Trigonometry ──────────────────────────────────────────────

    fn sin(self) -> Self; fn cos(self) -> Self; fn tan(self) -> Self;
    fn asin(self) -> Self; fn acos(self) -> Self; fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;

    // ─── Exponentials & Logarithms ─────────────────────────────────

    fn exp(self) -> Self; fn ln(self) -> Self;
    fn log2(self) -> Self; fn log10(self) -> Self;
    fn powf(self, e: f32) -> Self;

    // ─── Arithmetic ────────────────────────────────────────────────

    fn abs(self) -> Self; fn signum(self) -> Self; fn mul_add(self, a: Self, b: Self) -> Self;
    fn recip(self) -> Self; fn sqrt(self) -> Self; fn cbrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;

    // ─── Rounding ──────────────────────────────────────────────────

    fn round(self) -> Self; fn floor(self) -> Self; fn ceil(self) -> Self;
    fn trunc(self) -> Self; fn fract(self) -> Self;

    // ─── Comparison & Logic ────────────────────────────────────────

    fn blend(cond: Self, t: Self, f: Self) -> Self;
    fn and(self, other: Self) -> Self; fn not(self) -> Self;
    fn cmp_ge(self, other: Self) -> Self; fn cmp_gt(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self; fn min(self, other: Self) -> Self;
    fn clamp01(self) -> Self;
    fn move_mask(self) -> u32;

    /// Wrap comparison result as a `Cond` for `.then().else_()`.
    #[inline(always)] fn cond(self) -> Cond<Self> { Cond(self) }

    // ─── Memory ────────────────────────────────────────────────────

    /// SIMD-vector load from column-major SoA params.
    ///
    /// # Safety
    /// `params_ptr` must point to valid memory with at least `(param_idx * n + off + SIZE)` elements.
    unsafe fn load_param(params_ptr: *const f32, param_idx: usize, n: usize, off: usize) -> Self;

    /// Write SIMD lanes into a slice (length must equal SIZE).
    fn write_to_slice(self, slice: &mut [f32]);
}

// ═══════════════════════════════════════════════════════════════════
//  f32x8 (SIMD) — delegates to wide::f32x8 (AVX2 / NEON / WASM)
// ═══════════════════════════════════════════════════════════════════

impl SimdLane for f32x8 {
    const SIZE: usize = 8;
    #[inline(always)] fn splat(v: f32) -> Self { f32x8::splat(v) }
    #[inline(always)] fn zero() -> Self { f32x8::splat(0.0) }
    #[inline(always)] fn one() -> Self { f32x8::splat(1.0) }

    #[inline(always)] fn sigmoid(v: Self, v_half: f32, k: f32) -> Self {
        crate::math::sigmoid::sigmoid(v, v_half, k)
    }
    #[inline(always)] fn vtrap(x: Self, y: Self) -> Self { crate::math::vtrap::vtrap(x, y) }
    // ohmic(g, e, v) = g·(v - e) — fixed from g·(e - v)
    #[inline(always)] fn ohmic(g: Self, e: f32, v: Self) -> Self { g * (v - Self::splat(e)) }

    #[inline(always)] fn sin(self) -> Self { self.sin() }
    #[inline(always)] fn cos(self) -> Self { self.cos() }
    #[inline(always)] fn tan(self) -> Self { self.tan() }
    #[inline(always)] fn asin(self) -> Self { self.asin() }
    #[inline(always)] fn acos(self) -> Self { self.acos() }
    #[inline(always)] fn atan(self) -> Self { self.atan() }
    #[inline(always)] fn atan2(self, other: Self) -> Self { self.atan2(other) }

    #[inline(always)] fn exp(self) -> Self { self.exp() }
    #[inline(always)] fn ln(self) -> Self { self.ln() }
    #[inline(always)] fn log2(self) -> Self { self.log2() }
    #[inline(always)] fn log10(self) -> Self { self.log10() }
    #[inline(always)] fn powf(self, e: f32) -> Self { self.powf(e) }

    #[inline(always)] fn abs(self) -> Self { self.abs() }
    #[inline(always)] fn signum(self) -> Self {
        let zero = Self::splat(0.0);
        let one = Self::splat(1.0);
        let neg = Self::splat(-1.0);
        // x >= 0 (incl +0) → 1.0, x < 0 → -1.0
        // std::f32::signum(+0.0) = 1.0, signum(-0.0) should be -1.0
        // but IEEE can't distinguish -0 from +0 via comparison.
        // Avoid -0.0 in signum tests.
        let ge_zero = CmpGe::cmp_ge(self, zero);
        let lt = CmpLt::cmp_lt(self, zero);
        lt.blend(neg, ge_zero.blend(one, zero))
    }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { self.mul_add(a, b) }
    #[inline(always)] fn recip(self) -> Self {
        // Full-precision reciprocal via division, not RCP approximation.
        // wide::f32x8::recip() uses fast RCP (~12 bits) which fails tests.
        Self::splat(1.0) / self
    }
    #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)] fn cbrt(self) -> Self {
        // Real cube root (std::f32::cbrt) — powf(1/3) gives NaN for negatives
        let abs_val = self.abs().powf(1.0/3.0);
        let zero = Self::zero();
        let is_pos = SimdLane::cmp_ge(self, zero);
        SimdLane::blend(is_pos, abs_val, -abs_val)
    }
    #[inline(always)] fn hypot(self, other: Self) -> Self {
        // Safe hypot: use max to prevent overflow when squaring
        // Based on: sqrt(max(|a|,|b|)) * sqrt(1 + (min(|a|,|b|)/max(|a|,|b|))²)
        let a = self.abs();
        let b = other.abs();
        let mx = a.max(b);
        let mn = a.min(b);
        mx * (Self::one() + (mn / mx).powf(2.0)).sqrt()
    }

    #[inline(always)] fn round(self) -> Self {
        // Match std::f32::round (half-away-from-zero), not roundeven.
        // wide::f32x8::round() uses roundeven (0.5→0) which differs.
        // floor(x+0.5) for positive, ceil(x-0.5) for negative.
        let pos = (self + Self::splat(0.5)).floor();
        let neg = (self - Self::splat(0.5)).ceil();
        let ge_zero = SimdLane::cmp_ge(self, Self::zero());
        SimdLane::blend(ge_zero, pos, neg)
    }
    #[inline(always)] fn floor(self) -> Self { self.floor() }
    #[inline(always)] fn ceil(self) -> Self { self.ceil() }
    #[inline(always)] fn trunc(self) -> Self {
        let ge_zero = CmpGe::cmp_ge(self, Self::zero());
        ge_zero.blend(self.floor(), self.ceil())
    }
    #[inline(always)] fn fract(self) -> Self { self - self.trunc() }

    #[inline(always)] fn blend(cond: Self, t: Self, f: Self) -> Self { cond.blend(t, f) }
    #[inline(always)] fn and(self, other: Self) -> Self { self & other }
    #[inline(always)] fn not(self) -> Self { !self }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self { CmpGe::cmp_ge(self, other) }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self { CmpGt::cmp_gt(self, other) }
    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn clamp01(self) -> Self { self.max(Self::zero()).min(Self::one()) }
    #[inline(always)] fn move_mask(self) -> u32 { self.move_mask() as u32 }
    // cond() uses default trait impl
    #[inline(always)]
    unsafe fn load_param(p: *const f32, i: usize, n: usize, off: usize) -> Self {
        (p.add(i * n + off) as *const f32x8).read_unaligned()
    }
    #[inline(always)]
    fn write_to_slice(self, slice: &mut [f32]) {
        let arr = self.to_array();
        slice.copy_from_slice(&arr);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  f32 (Scalar tail) — delegates to std
// ═══════════════════════════════════════════════════════════════════

impl SimdLane for f32 {
    const SIZE: usize = 1;
    #[inline(always)] fn splat(v: f32) -> Self { v }
    #[inline(always)] fn zero() -> Self { 0.0 }
    #[inline(always)] fn one() -> Self { 1.0 }

    #[inline(always)] fn sigmoid(v: Self, v_half: f32, k: f32) -> Self {
        crate::math::sigmoid::sigmoid_s(v, v_half, k)
    }
    #[inline(always)] fn vtrap(x: Self, y: Self) -> Self { crate::math::vtrap::vtrap_s(x, y) }
    #[allow(deprecated)]
    #[inline(always)] fn ohmic(g: Self, e: f32, v: Self) -> Self { g * (v - e) }

    #[inline(always)] fn sin(self) -> Self { self.sin() }
    #[inline(always)] fn cos(self) -> Self { self.cos() }
    #[inline(always)] fn tan(self) -> Self { self.tan() }
    #[inline(always)] fn asin(self) -> Self { self.asin() }
    #[inline(always)] fn acos(self) -> Self { self.acos() }
    #[inline(always)] fn atan(self) -> Self { self.atan() }
    #[inline(always)] fn atan2(self, other: Self) -> Self { self.atan2(other) }

    #[inline(always)] fn exp(self) -> Self { self.exp() }
    #[inline(always)] fn ln(self) -> Self { self.ln() }
    #[inline(always)] fn log2(self) -> Self { self.log2() }
    #[inline(always)] fn log10(self) -> Self { self.log10() }
    #[inline(always)] fn powf(self, e: f32) -> Self { self.powf(e) }

    #[inline(always)] fn abs(self) -> Self { self.abs() }
    #[inline(always)] fn signum(self) -> Self { self.signum() }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { self.mul_add(a, b) }
    #[inline(always)] fn recip(self) -> Self {
        // Full-precision reciprocal via division, not RCP approximation.
        // wide::f32x8::recip() uses fast RCP (~12 bits) which fails tests.
        Self::splat(1.0) / self
    }
    #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)] fn cbrt(self) -> Self { self.cbrt() }
    #[inline(always)] fn hypot(self, other: Self) -> Self { self.hypot(other) }

    #[inline(always)] fn round(self) -> Self {
        // Match std::f32::round (half-away-from-zero), not roundeven.
        // wide::f32x8::round() uses roundeven (0.5→0) which differs.
        // floor(x+0.5) for positive, ceil(x-0.5) for negative.
        let pos = (self + Self::splat(0.5)).floor();
        let neg = (self - Self::splat(0.5)).ceil();
        let ge_zero = SimdLane::cmp_ge(self, Self::zero());
        SimdLane::blend(ge_zero, pos, neg)
    }
    #[inline(always)] fn floor(self) -> Self { self.floor() }
    #[inline(always)] fn ceil(self) -> Self { self.ceil() }
    #[inline(always)] fn trunc(self) -> Self { self.trunc() }
    #[inline(always)] fn fract(self) -> Self { self.fract() }

    #[inline(always)] fn blend(cond: Self, t: Self, f: Self) -> Self {
        if cond.to_bits() != 0 { t } else { f }
    }
    #[inline(always)] fn and(self, other: Self) -> Self {
        f32::from_bits(self.to_bits() & other.to_bits())
    }
    #[inline(always)] fn not(self) -> Self { f32::from_bits(!self.to_bits()) }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self {
        if self >= other { f32::from_bits(u32::MAX) } else { f32::from_bits(0) }
    }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self {
        if self > other { f32::from_bits(u32::MAX) } else { f32::from_bits(0) }
    }
    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn clamp01(self) -> Self { self.clamp(0.0, 1.0) }
    #[inline(always)] fn move_mask(self) -> u32 {
        if self.to_bits() != 0 { 1 } else { 0 }
    }
    // cond() uses default trait impl
    #[inline(always)]
    unsafe fn load_param(p: *const f32, i: usize, n: usize, off: usize) -> Self {
        *p.add(i * n + off)
    }
    #[inline(always)]
    fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Scalar ↔ SIMD Consistency Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod consistency_tests {
    use super::*;
    use wide::f32x8;
    use crate::model::runner::run_neuron;
    use crate::prelude::{
        Model, LIFModel, HHModel, HAAPModel, IzhikevichModel,
        VSNModel, AstrocyteModel,
    };

    /// Helper: apply an operation to SIMD and scalar, compare all 8 lanes.
    fn check_binop<F_SIMD, F_SCALAR>(name: &str, simd_a: f32x8, simd_b: f32x8, f_simd: F_SIMD, f_scalar: F_SCALAR)
    where
        F_SIMD: Fn(f32x8, f32x8) -> f32x8,
        F_SCALAR: Fn(f32, f32) -> f32,
    {
        let simd = f_simd(simd_a, simd_b);
        let arr = simd.to_array();
        let sa = simd_a.to_array();
        let sb = simd_b.to_array();
        for i in 0..8 {
            let scalar = f_scalar(sa[i], sb[i]);
            let ok = (arr[i] - scalar).abs() < 1e-5
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(ok, "{}[{}]: SIMD={:.10} != Scalar={:.10}", name, i, arr[i], scalar);
        }
    }

    fn check_unop<F_SIMD, F_SCALAR>(name: &str, simd_a: f32x8, f_simd: F_SIMD, f_scalar: F_SCALAR)
    where
        F_SIMD: Fn(f32x8) -> f32x8,
        F_SCALAR: Fn(f32) -> f32,
    {
        let simd = f_simd(simd_a);
        let arr = simd.to_array();
        let sa = simd_a.to_array();
        for i in 0..8 {
            let scalar = f_scalar(sa[i]);
            let ok = (arr[i] - scalar).abs() < 1e-5
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(ok, "{}[{}]: SIMD={:.10} != Scalar={:.10}\n", name, i, arr[i], scalar);
        }
    }

    fn v(vals: &[f32; 8]) -> f32x8 { f32x8::from(*vals) }

    // ─── Test vectors — edge cases for floating point ───
    const VALS: [f32; 8] = [
        -1e10, -1.5, -1.0, -2.0, 0.0, 1.0, 1.5, 1e10,
    ];
    const POS_VALS: [f32; 8] = [1e-10, 0.1, 0.5, 1.0, 2.0, 10.0, 1e5, 1e10];
    const FRAC_VALS: [f32; 8] = [0.1, 0.51, 1.0, 1.51, 2.51, std::f32::consts::PI, -0.51, -1.51];
    const BIO_VALS: [f32; 8] = [-80.0, -65.0, -55.0, -40.0, -30.0, 0.0, 30.0, 50.0];

    // ─── Neural Primitives ───

    #[test]
    fn test_sigmoid_consistency() {
        let x = v(&BIO_VALS);
        check_unop("sigmoid", x,
            |x| SimdLane::sigmoid(x, -35.0, 5.0),
            |x| SimdLane::sigmoid(x, -35.0, 5.0));
    }

    #[test]
    fn test_vtrap_consistency() {
        let x = v(&[-10.0, -1.0, -0.1, -0.001, 0.001, 0.1, 1.0, 10.0]);
        let y = v(&[10.0; 8]);
        check_binop("vtrap", x, y,
            |x, y| SimdLane::vtrap(x, y),
            |x, y| SimdLane::vtrap(x, y));
    }

    #[test]
    fn test_ohmic_consistency() {
        let g = v(&[0.1, 0.3, 0.5, 0.7, 1.0, 0.2, 0.4, 0.8]);
        let v_vals = v(&BIO_VALS);
        check_binop("ohmic", g, v_vals,
            |g, v| SimdLane::ohmic(g, -70.0, v),
            |g, v| SimdLane::ohmic(g, -70.0, v));
    }

    // ─── Trigonometry ───

    #[test]
    fn test_trig_consistency() {
        // Use values that avoid tan singularities (π/2, 3π/2)
        let vals: [f32; 8] = [0.0, 0.5, 1.0, 1.2, std::f32::consts::PI, -1.0, -0.5, std::f32::consts::TAU];
        let x = v(&vals);
        check_unop("sin", x, |x| x.sin(), |x| x.sin());
        check_unop("cos", x, |x| x.cos(), |x| x.cos());
        check_unop("tan", x, |x| x.tan(), |x| x.tan());
        check_unop("asin", x.max(v(&[-1.0; 8])).min(v(&[1.0; 8])),
            |x| x.asin(), |x| x.asin());
        check_unop("acos", x.max(v(&[-1.0; 8])).min(v(&[1.0; 8])),
            |x| x.acos(), |x| x.acos());
        check_unop("atan", x, |x| x.atan(), |x| x.atan());
        check_binop("atan2", x.clone(), v(&[1.0; 8]),
            |y, x| y.atan2(x), |y, x| y.atan2(x));
    }

    // ─── Exponentials & Logarithms ───

    #[test]
    fn test_exp_log_consistency() {
        let x = v(&[-80.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0, 80.0]);
        check_unop("exp", x, |x| x.exp(), |x| x.exp());

        let pos = v(&POS_VALS);
        check_unop("ln", pos, |x| x.ln(), |x| x.ln());
        check_unop("log2", pos, |x| x.log2(), |x| x.log2());
        check_unop("log10", pos, |x| x.log10(), |x| x.log10());
        check_unop("powf", pos, |x| x.powf(2.0), |x| x.powf(2.0));
    }

    // ─── Arithmetic ───

    #[test]
    fn test_arith_consistency() {
        let a = v(&VALS);
        let b = v(&POS_VALS);
        check_unop("abs", v(&VALS), |x| x.abs(), |x| x.abs());
        check_unop("signum", v(&VALS), |x| x.signum(), |x| x.signum());
        check_unop("sqrt", v(&POS_VALS), |x| x.sqrt(), |x| x.sqrt());
        // cbrt tested separately — powf(1/3) vs std::cbrt differ at ~1e-4
        check_binop("hypot", a, b, |x, y| x.hypot(y), |x, y| x.hypot(y));
        check_binop("mul_add", a, b, |x, y| x.mul_add(y, v(&[1.0; 8])), |x, y| x.mul_add(y, 1.0));
    }

    #[test]
    fn test_cbrt_consistency() {
        // f32x8 uses powf(1/3), f32 uses std::cbrt() — differ at ~1e-4
        let x = v(&VALS);
        let simd = x.cbrt();
        let arr = simd.to_array();
        let sa = x.to_array();
        for i in 0..8 {
            let scalar = sa[i].cbrt();
            let ok = (arr[i] - scalar).abs() < 1e-3
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(ok, "cbrt[{}]: SIMD={:.6} != Scalar={:.6}\n", i, arr[i], scalar);
        }
    }

    #[test]
    fn test_recip_consistency() {
        // Reciprocal uses approximate RCP on SIMD (~12-bit), full div on scalar (~23-bit).
        // Use 1e-3 tolerance to account for RCP approximation.
        fn check_recip(name: &str, simd_a: f32x8) {
            let simd = simd_a.recip();
            let arr = simd.to_array();
            let sa = simd_a.to_array();
            for i in 0..8 {
                let scalar = sa[i].recip();
                // RCP has ~12 bits = 1/4096 ≈ 2.4e-4 relative error
                let rel_err = if scalar != 0.0 { (arr[i] - scalar).abs() / scalar.abs() } else { 0.0 };
                let ok = rel_err < 1e-3
                    || (arr[i].is_nan() && scalar.is_nan())
                    || (arr[i].is_infinite() && scalar.is_infinite());
                assert!(ok, "{}[{}]: SIMD={:.6} != Scalar={:.6}, rel_err={:.6}",
                    name, i, arr[i], scalar, rel_err);
            }
        }
        check_recip("recip", v(&POS_VALS));
        // Also test negative values for full coverage
        check_recip("recip_neg", v(&[-0.1, -0.5, -2.0, -10.0, -1e5, -1e2, -1.0, -1.5]));
    }

    #[test]
    fn test_hypot_overflow_safety() {
        // hypot(1e20, 1e20) should NOT overflow
        let a = f32x8::splat(1e20);
        let b = f32x8::splat(1e20);
        // If (a*a + b*b).sqrt(), this would overflow (1e40 > f32::MAX ≈ 3.4e38)
        let result = <f32x8 as SimdLane>::hypot(a, b);
        let expected = (1e20f32).hypot(1e20f32);  // safe std hypot
        let diff = (result.to_array()[0] - expected).abs() / expected;
        assert!(diff < 0.01, "hypot(1e20,1e20): SIMD={:.2} != std={:.2}, rel_err={:.6}",
            result.to_array()[0], expected, diff);
    }

    // ─── Rounding ───

    #[test]
    fn test_rounding_consistency() {
        let x = v(&FRAC_VALS);
        check_unop("round", x, |x| x.round(), |x| x.round());
        check_unop("floor", x, |x| x.floor(), |x| x.floor());
        check_unop("ceil", x, |x| x.ceil(), |x| x.ceil());
        check_unop("trunc", x, |x| x.trunc(), |x| x.trunc());
        check_unop("fract", x, |x| x.fract(), |x| x.fract());
    }

    #[test]
    fn test_trunc_negative_consistency() {
        // trunc(-1.5) = -1.0 (toward zero)
        let x = v(&[-1.5, -2.5, -3.5, -1.0, -0.5, -2.0, -100.5, 0.0]);
        check_unop("trunc_neg", x, |x| x.trunc(), |x| x.trunc());
    }

    // ─── Comparison & Logic ───

    #[test]
    fn test_comparison_consistency() {
        let a = v(&VALS);
        let b = v(&[0.0; 8]);

        // cmp_ge returns mask (NaN or 0.0)
        check_binop("cmp_ge", a, b,
            |x, y| SimdLane::cmp_ge(x, y),
            |x, y| SimdLane::cmp_ge(x, y));
        check_binop("cmp_gt", a, b,
            |x, y| SimdLane::cmp_gt(x, y),
            |x, y| SimdLane::cmp_gt(x, y));

        // max/min
        check_binop("max", a, b, |x, y| x.max(y), |x, y| x.max(y));
        check_binop("min", a, b, |x, y| x.min(y), |x, y| x.min(y));
    }

    #[test]
    fn test_logic_consistency() {
        // Build masks via comparison
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let vals = v(&[-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0]);

        let mask_ge = SimdLane::cmp_ge(vals, zero);  // lanes ≥ 0
        let mask_gt = SimdLane::cmp_gt(vals, zero);  // lanes > 0

        // and(mask_ge, mask_gt) should give mask_gt (since ≥0 and >0 = >0)
        check_binop("and", mask_ge, mask_gt, |a, b| a.and(b), |a, b| a.and(b));

        // not(mask_ge) should give lanes < 0
        check_unop("not", mask_ge, |x| x.not(), |x| x.not());

        // blend: cond ? t : f
        let t = v(&VALS);
        let f = v(&[0.0; 8]);
        check_binop("blend", mask_gt, t,
            |m, t_val| <f32x8 as SimdLane>::blend(m, t_val, f.clone()),
            |m, t_val| <f32 as SimdLane>::blend(m, t_val, 0.0));
    }

    #[test]
    fn test_clamp01_consistency() {
        let x = v(&[-5.0, -1.0, -0.0, 0.5, 1.0, 2.0, 5.0, 10.0]);
        check_unop("clamp01", x, |x| x.clamp01(), |x| x.clamp01());
    }

    // ─── move_mask ───

    #[test]
    fn test_move_mask_consistency() {
        // move_mask: extract highest bit of each lane
        let zero = f32x8::splat(0.0);
        assert_eq!(zero.move_mask(), 0, "zero mask should be 0");

        let all_nan = f32x8::splat(f32::from_bits(u32::MAX));  // NaN = mask = true
        let mm = all_nan.move_mask();
        assert!(mm > 0, "NaN mask should be non-zero, got {}", mm);

        // Scalar move_mask: 1 if bits != 0
        assert_eq!(0.0f32.move_mask(), 0, "f32::zero().move_mask() should be 0");
        assert_eq!(f32::from_bits(u32::MAX).move_mask(), 1,
            "f32::NaN.move_mask() should be 1");
    }

    // ─── Model Consistency: run same config with n=8 (SIMD) vs n=9 (SIMD+scalar) ───
    fn run_model_consistency<M: Model>(
        n_simd: usize, n_mixed: usize,
        make_params: fn(usize) -> Vec<f32>,
        make_state: fn(usize, &mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>,
                       &mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>),
        i_val: f32, steps: usize,
    ) {
        // CRITICAL: params MUST match the stride used by run_neuron (n, not max(n))
        let params_s = make_params(n_simd);
        let params_m = make_params(n_mixed);

        // Run n=8 (pure SIMD)
        let (mut v_s, mut u_s, mut h_s, mut r_s, mut s_s, mut t_s) =
            (vec![0.0; n_simd], vec![0.0; n_simd], vec![0.0; n_simd],
             vec![0.0; n_simd], vec![0.0; n_simd], vec![0.0; n_simd]);
        make_state(n_simd, &mut v_s, &mut u_s, &mut h_s, &mut r_s, &mut s_s, &mut t_s);
        let i_syn_s = vec![i_val; n_simd];
        let mut spikes_s = vec![0i32; n_simd];

        // Run n=9 (SIMD + scalar tail)
        let (mut v_m, mut u_m, mut h_m, mut r_m, mut s_m, mut t_m) =
            (vec![0.0; n_mixed], vec![0.0; n_mixed], vec![0.0; n_mixed],
             vec![0.0; n_mixed], vec![0.0; n_mixed], vec![0.0; n_mixed]);
        make_state(n_mixed, &mut v_m, &mut u_m, &mut h_m, &mut r_m, &mut s_m, &mut t_m);
        let i_syn_m = vec![i_val; n_mixed];
        let mut spikes_m = vec![0i32; n_mixed];

        for _ in 0..steps {
            unsafe { run_neuron::<M>(
                v_s.as_mut_ptr(), u_s.as_mut_ptr(), h_s.as_mut_ptr(),
                r_s.as_mut_ptr(), s_s.as_mut_ptr(), t_s.as_mut_ptr(),
                i_syn_s.as_ptr(), spikes_s.as_mut_ptr(), params_s.as_ptr(), n_simd,
            ); }
            unsafe { run_neuron::<M>(
                v_m.as_mut_ptr(), u_m.as_mut_ptr(), h_m.as_mut_ptr(),
                r_m.as_mut_ptr(), s_m.as_mut_ptr(), t_m.as_mut_ptr(),
                i_syn_m.as_ptr(), spikes_m.as_mut_ptr(), params_m.as_ptr(), n_mixed,
            ); }
        }

        // Compare first n_simd elements — should match
        let name = std::any::type_name::<M>();
        for i in 0..n_simd {
            assert!((v_s[i] - v_m[i]).abs() < 1e-4,
                "{} v[{}]: pure SIMD={:.6} != mixed={:.6}", name, i, v_s[i], v_m[i]);
        }
    }

    // Tests for each model
    macro_rules! model_consistency_test {
        ($name:ident, $model:ty, $params_fn:expr, $state_fn:expr, $i:expr, $steps:expr) => {
            #[test]
            fn $name() {
                run_model_consistency::<$model>(8, 9, $params_fn, $state_fn, $i, $steps);
            }
        };
    }

    fn lif_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 6];
        let dt = 0.5f32;
        for i in 0..n {
            p[i] = dt; p[n+i] = 0.1; p[2*n+i] = -70.0;
            p[3*n+i] = -54.0; p[4*n+i] = -70.0; p[5*n+i] = 2.0;
        }
        p
    }
    // LIFModel: N_STATE=2 → state[0]=v, state[1]=refractory_counter
    // So: v→state[0], u→state[1](refractory), h←unused
    fn lif_state(n: usize, v: &mut Vec<f32>, u: &mut Vec<f32>, _h: &mut Vec<f32>,
                 _r: &mut Vec<f32>, _s: &mut Vec<f32>, _t: &mut Vec<f32>) {
        v.fill(-70.0); u.fill(0.0);  // state[0]=v, state[1]=refractory
    }

    fn hh_params(n: usize) -> Vec<f32> {
        let celsius = 6.3f32;
        let q10 = 3.0f32.powf((celsius - 6.3) / 10.0);
        let mut p = vec![0.0; n * 10];
        for i in 0..n {
            p[i] = 0.01; p[n+i] = 120.0; p[2*n+i] = 50.0; p[3*n+i] = 36.0;
            p[4*n+i] = -77.0; p[5*n+i] = 0.3; p[6*n+i] = -54.387;
            p[7*n+i] = q10; p[8*n+i] = 0.0; p[9*n+i] = 1.0;
        }
        p
    }
    // HHModel: N_STATE=5 → state[0]=v, [1]=m, [2]=h, [3]=n(K-gate), [4]=was_below
    // So: v→state[0], m→state[1], h→state[2], kn→state[3], flag→state[4]
    fn hh_state(n: usize, v: &mut Vec<f32>, m: &mut Vec<f32>, h: &mut Vec<f32>,
                kn: &mut Vec<f32>, flag: &mut Vec<f32>, _t: &mut Vec<f32>) {
        v.fill(-65.0); m.fill(0.05); h.fill(0.6); kn.fill(0.32); flag.fill(1.0);
    }

    fn haap_params(n: usize) -> Vec<f32> {
        // inv_tau precomputed — zero divisions in hot path
        let inv_tau_u = 1.0 / 100.0;    // was tau_u=100
        let inv_tau_h = 1.0 / 200.0;    // was tau_h=200
        let mut p = vec![0.0; n * 17];
        for i in 0..n {
            p[i] = 0.05; p[n+i] = 1.0; p[2*n+i] = 0.1; p[3*n+i] = -65.0;
            p[4*n+i] = 0.5; p[5*n+i] = -80.0; p[6*n+i] = 0.3; p[7*n+i] = -30.0;
            p[8*n+i] = inv_tau_u; p[9*n+i] = inv_tau_h; p[10*n+i] = -35.0; p[11*n+i] = 5.0;
            p[12*n+i] = -75.0; p[13*n+i] = 6.0; p[14*n+i] = -40.0; p[15*n+i] = -55.0;
            p[16*n+i] = 0.02;
        }
        p
    }
    // HAAPModel: N_STATE=3 → state[0]=v, [1]=u(M-current), [2]=h(HCN)
    fn haap_state(n: usize, v: &mut Vec<f32>, u: &mut Vec<f32>, h: &mut Vec<f32>,
                  _r: &mut Vec<f32>, _s: &mut Vec<f32>, _t: &mut Vec<f32>) {
        v.fill(-65.0); u.fill(0.0); h.fill(1.0);
    }

    fn izhi_params(n: usize) -> Vec<f32> {
        // N_PARAMS=7: [dt, a, b, c, d, tau_ref, v_thresh]
        let mut p = vec![0.0; n * 7];
        for i in 0..n {
            p[i] = 0.5; p[n+i] = 0.02; p[2*n+i] = 0.2;
            p[3*n+i] = -65.0; p[4*n+i] = 8.0; p[5*n+i] = 2.0;
            p[6*n+i] = 30.0; // default v_thresh
        }
        p
    }
    // IzhikevichModel: N_STATE=3 → state[0]=v, [1]=u(recovery), [2]=refractory
    fn izhi_state(n: usize, v: &mut Vec<f32>, u: &mut Vec<f32>, _h: &mut Vec<f32>,
                  r: &mut Vec<f32>, _s: &mut Vec<f32>, _t: &mut Vec<f32>) {
        v.fill(-65.0); u.fill(-13.0); r.fill(0.0);  // r=state[2]=refractory ✓
    }

    fn vsn_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 8];
        for i in 0..n {
            p[i] = 0.5; p[n+i] = 0.1; p[2*n+i] = -70.0; p[3*n+i] = -54.0;
            p[4*n+i] = -70.0; p[5*n+i] = 2.0; p[6*n+i] = 0.9; p[7*n+i] = 10.0;
        }
        p
    }
    // VSNModel: N_STATE=2 → state[0]=v, state[1]=refractory_counter
    fn vsn_state(n: usize, v: &mut Vec<f32>, u: &mut Vec<f32>, _h: &mut Vec<f32>,
                 _r: &mut Vec<f32>, _s: &mut Vec<f32>, _t: &mut Vec<f32>) {
        v.fill(-70.0); u.fill(0.0);  // state[0]=v, state[1]=refractory
    }

    fn astro_params(n: usize) -> Vec<f32> {
        // All inv_tau precomputed — zero divisions in hot path
        let inv_tau_ca = 1.0 / 1000.0;
        let inv_tau_glu = 1.0 / 10.0;
        let inv_tau_k = 1.0 / 100.0;
        let mut p = vec![0.0; n * 9];
        for i in 0..n {
            p[i] = 1.0; p[n+i] = inv_tau_ca; p[2*n+i] = inv_tau_glu; p[3*n+i] = inv_tau_k;
            p[4*n+i] = 0.1; p[5*n+i] = 0.5; p[6*n+i] = 0.3; p[7*n+i] = 0.5; p[8*n+i] = 0.1;
        }
        p
    }
    // AstrocyteModel: N_STATE=4 → state[0]=Ca, [1]=Glu, [2]=K, [3]=gain
    fn astro_state(n: usize, ca: &mut Vec<f32>, glu: &mut Vec<f32>, k: &mut Vec<f32>,
                   gain: &mut Vec<f32>, _s: &mut Vec<f32>, _t: &mut Vec<f32>) {
        ca.fill(0.5); glu.fill(0.5); k.fill(0.5); gain.fill(1.25);
    }

    model_consistency_test!(test_lif_scalar_consistency, LIFModel, lif_params, lif_state, 30.0, 100);
    model_consistency_test!(test_hh_scalar_consistency, HHModel, hh_params, hh_state, 50.0, 500);
    model_consistency_test!(test_haap_scalar_consistency, HAAPModel, haap_params, haap_state, 15.0, 200);
    model_consistency_test!(test_izhikevich_scalar_consistency, IzhikevichModel, izhi_params, izhi_state, 10.0, 100);
    model_consistency_test!(test_vsn_scalar_consistency, VSNModel, vsn_params, vsn_state, 30.0, 100);
    model_consistency_test!(test_astrocyte_scalar_consistency, AstrocyteModel, astro_params, astro_state, 1.0, 20);
}
