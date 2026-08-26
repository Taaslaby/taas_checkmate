//! # Cond — Ergonomic Branchless Conditional for SIMD Masks
//!
//! Generic over `S: SimdLane`.
//!
//! Wraps a hardware SIMD mask (NaN = true, 0.0 = false) and provides
//! `.then().else_()` for readable blend expressions.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let is_refr = refr.cmp_gt(0.0).cond();
//! let v = is_refr.then(v_reset).else_(v_raw);
//! ```
//!
//! Equivalent to `S::blend(mask, t, f)` but reads left-to-right.

use super::SimdLane;

/// A comparison mask — produced by `cmp_ge`, `cmp_gt`, etc.
///
/// Masks use hardware convention: **sign bit set** = true, 0.0 = false
/// (NaN and -1.0 are both valid true-masks; 1.0 is NOT — `wide::blend`
/// selects by sign bit on the AVX path (`blendvps`) but by any non-zero
/// bit on the SSE fallback (`generic_bit_blend`), so only sign-bit masks
/// are portable across targets).
#[derive(Copy, Clone)]
pub struct Cond<S: SimdLane>(pub S);

/// Intermediate value: `cond.then(t)` stores the true-branch value.
#[derive(Copy, Clone)]
pub struct CondThen<S: SimdLane>(S, S); // (mask, true_value)

impl<S: SimdLane> Cond<S> {
    /// Set the value for lanes where condition is TRUE.
    #[inline(always)]
    pub fn then(self, t: S) -> CondThen<S> {
        CondThen(self.0, t)
    }
}

impl<S: SimdLane> CondThen<S> {
    /// Set the value for lanes where condition is FALSE.
    /// Returns blended result: cond_mask ? true_val : false_val.
    #[inline(always)]
    pub fn else_(self, f: S) -> S {
        S::blend(self.0, self.1, f)
    }
}
