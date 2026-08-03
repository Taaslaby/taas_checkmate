//! # 🏗️ SimdLane Trait — The Compiler Bridge
//!
//! **This is the critical boundary between Rust and hardware.**
//!
//! Every method maps to **exactly one hardware instruction**.
//! Writing a model against `SimdLane` means writing code that compiles
//! to both AVX2 and scalar from the same source — without runtime checks.
//!
//! ## The Contract
//!
//! ```text
//! trait SimdLane {
//!     fn cmp_ge(self, other: Self) -> Self;
//!     //     ↑                      ↑
//!     //  1 SIMD call          1 instruction in assembly
//! }
//! ```
//!
//! A `SimdLane` method is not "fast" or "slow" — it IS the instruction.
//! Benchmarking `SimdLane::exp()` in isolation tells you the exact cost
//! of `_mm256_exp_ps` on your CPU.
//!
//! ## Implementations
//!
//! | File | Target | Instruction set |
//! |------|--------|-----------------|
//! | `lane_f32x8.rs` | AVX2 (x86) | `_mm256_*`, `wide::f32x8` |
//! | `lane_f32x8.rs` | NEON (ARM) | `wide::f32x8` (emulated 2×128-bit) |
//! | `lane_f32.rs` | Any CPU | `f32` scalar ops (no vectorization) |
//!
//! ## Safety
//!
//! The trait itself is safe. Only the `load_param` method is `unsafe` —
//! it dereferences raw pointers into SoA memory. All higher layers
//! (`ops/`, `zoo/`, `api/`) use `load_param` behind safe wrappers.

use std::ops::{Add, Sub, Mul, Div, Neg};

use super::cond::Cond;

// ═══════════════════════════════════════════════════════════════════
//  SimdLane Trait — Every method maps to one hardware instruction
// ═══════════════════════════════════════════════════════════════════

/// Zero-Cost SIMD Lane Abstraction.
///
/// Implemented for `f32x8` (SIMD) and `f32` (scalar tail).
/// Every method maps to a single hardware instruction where possible.
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

    /// Sigmoid activation: `1 / (1 + exp(-k·(V - V_half)))`
    ///
    /// Standard Boltzmann sigmoid for voltage-dependent gating.
    /// - `v_half`: half-activation voltage (mV)
    /// - `k`: slope factor (mV⁻¹)
    fn sigmoid(v: Self, v_half: f32, k: f32) -> Self;

    /// Hodgkin-Huxley voltage trap: `x / (exp(x) - 1)`, avoiding singularity at x=0.
    ///
    /// Used in HH α/β rate functions to handle the `vtrap(-(V+V₀)/k, k)` form.
    /// When `|x| < ε`, returns `y` (the limit value) to avoid `0/0`.
    ///
    /// # Example (m-gate α)
    /// ```ignore
    /// let x = -(v + 40.0) / 10.0;
    /// let alpha = 0.1 * vtrap(x, 10.0);  // → 0.1·x/(exp(x)-1)
    /// ```
    ///
    /// # Performance
    /// - SIMD: uses `blend()` for branchless singularity handling
    /// - Scalar: `if |x| < 1e-6 then y else x/(exp(x)-1)`
    fn vtrap(x: Self, y: Self) -> Self;

    /// Ohmic (conductance-based) current: `I = g · (V - E_rev)`
    ///
    /// NEURON convention for ionic currents. Positive current = depolarizing.
    /// - `g`: conductance (mS/cm² or nS)
    /// - `e`: reversal potential (mV)
    /// - `v`: membrane potential (mV)
    ///
    /// # Example
    /// ```ignore
    /// let i_na = ohmic(g_na, E_NA, v);  // Na⁺ current
    /// let i_k  = ohmic(g_k,  E_K,  v);  // K⁺ current
    /// ```
    ///
    /// # Performance
    /// Single FMA on AVX2: `g * (v - e)` → `vfmadd213ps`
    fn ohmic(g: Self, e: f32, v: Self) -> Self;

    // ─── Trigonometry ──────────────────────────────────────────────

    fn sin(self) -> Self; fn cos(self) -> Self; fn tan(self) -> Self;
    fn asin(self) -> Self; fn acos(self) -> Self; fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;

    // ─── Exponentials & Logarithms ─────────────────────────────────

    /// Precise exponential (≈1 ULP, libm / `wide` hardware path).
    ///
    /// # Precision switch
    /// Под фичей `fast-exp` `exp()` автоматически маршрутизируется на
    /// [`SimdLane::fast_exp`] — полиномиальную аппроксимацию (~1e-5 rel err).
    /// Модели продолжают вызывать `.exp()` без изменений: переключатель
    /// точности/скорости срабатывает на этапе компиляции.
    #[cfg(not(feature = "fast-exp"))]
    fn exp(self) -> Self;

    #[cfg(feature = "fast-exp")]
    #[inline(always)]
    fn exp(self) -> Self { self.fast_exp() }

    /// Fast exponential — polynomial approximation (~1e-5 relative error).
    ///
    /// Для моделей, где `exp()` — бутылочное горлышко (HH, conductance-based).
    /// Под фичей `fast-exp` становится реализацией `exp()`;
    /// в противном случае по умолчанию делегирует в точный `exp()`.
    ///
    /// ⚠️ Внешним имплементаторам: реализуйте и `exp()`, и `fast_exp()` —
    /// под разными фичами обязательный метод меняется местами.
    #[cfg(feature = "fast-exp")]
    fn fast_exp(self) -> Self;

    #[cfg(not(feature = "fast-exp"))]
    #[inline(always)]
    fn fast_exp(self) -> Self { self.exp() }

    fn ln(self) -> Self;
    fn log2(self) -> Self; fn log10(self) -> Self;
    fn powf(self, e: f32) -> Self;
    /// Integer power (compile-time constant N) — unrolled, no branches.
    ///
    /// N=0 → 1.0, N=1 → self, N=2 → square(), N=3 → cube(),
    /// N=4 → square().square(), N>4 → binary exponentiation.
    /// Due to `const N: usize`, LLVM resolves the match at compile time.
    fn powi<const N: usize>(self) -> Self;

    /// Square: `self * self` — 1 FMA-able multiply on AVX2.
    #[inline(always)] fn square(self) -> Self { self * self }
    /// Cube: `self * self * self` — 2 linked FMAs on AVX2.
    #[inline(always)] fn cube(self) -> Self { self * self * self }

    // ─── Arithmetic ────────────────────────────────────────────────

    fn abs(self) -> Self; fn signum(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
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

    /// Wrap comparison result as a `Cond` for `.then().else_()` chaining.
    #[inline(always)] fn cond(self) -> Cond<Self> { Cond(self) }

    // ─── Memory ────────────────────────────────────────────────────

    /// SIMD-vector load from column-major SoA parameters.
    ///
    /// # Safety
    /// `params_ptr` must point to valid memory with at least
    /// `(param_idx * n + off + SIZE)` elements.
    unsafe fn load_param(params_ptr: *const f32, param_idx: usize, n: usize, off: usize) -> Self;

    /// Write SIMD lanes into a slice (length must equal SIZE).
    fn write_to_slice(self, slice: &mut [f32]);

    /// Convert to a fixed-size array (lanes beyond SIZE are zero).
    fn to_array(self) -> [f32; 8];
    /// Build from a fixed-size array (uses only lane 0 for scalar types).
    fn from_array(arr: [f32; 8]) -> Self;
}

// ═══════════════════════════════════════════════════════════════════
//  Scalar ↔ SIMD Consistency Tests (см. core/lane_tests.rs)
// ═══════════════════════════════════════════════════════════════════
