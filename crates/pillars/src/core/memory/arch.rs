//! Architecture detection — compile-time SIMD width & alignment.
//!
//! The whole crate drives SIMD through **one** type, [`SimdType`] = `f32x8`
//! (from the `wide` crate). All widths/alignments below are *derived* from
//! that type (`size_of`/`align_of`), so the constants can never drift apart
//! from the type they describe — see [`LANES`].
//!
//! | Hardware | `SimdType` | `LANES` | `ALIGN` | Notes |
//! |:---------|:-----------|:--------|:--------|:------|
//! | x86_64 AVX2 | f32x8 | 8 | 32 | Native 256-bit, 8 f32/cycle |
//! | x86_64 SSE / ARM NEON / WASM simd128 | f32x8 | 8 | 32 | `wide` emulates 2×128-bit; SIMD code is identical |
//!
//! There is intentionally **no per-architecture SIMD type** (no f32x4/NEON
//! variant): the chunking logic in `population.rs`, `AlignedVec` padding and
//! `LANES` are all hardcoded against 8-wide `f32x8`. `LANES` is kept as a
//! symbolic name for `size_of::<SimdType>() / 4` so call sites read
//! `pad_lanes(n)` instead of magic `n.div_ceil(8) * 8`.

use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  1. SIMD TYPE & WIDTH — single source of truth
// ═══════════════════════════════════════════════════════════

/// SIMD type: always `f32x8` (8-wide). Native 256-bit on AVX2,
/// emulated as 2× f32x4 on NEON/SSE.
pub type SimdType = f32x8;

/// SIMD width in f32 elements — **derived from [`SimdType`]** so it can
/// never disagree with the type it describes (a `cfg!`-based constant
/// previously reported 4 on NEON/SSE while the actual type was 8-wide,
/// which would silently mis-chunk buffers).
pub const LANES: usize = std::mem::size_of::<SimdType>() / std::mem::size_of::<f32>();

/// Alignment for [`SimdType`] (`wide::f32x8` guarantees `align(32)`).
pub const ALIGN: usize = std::mem::align_of::<SimdType>();

/// SIMD register size in bytes (= [`LANES`] * 4 = `size_of::<SimdType>()`).
pub const SIMD_SIZE_BYTES: usize = std::mem::size_of::<SimdType>();

// ═══════════════════════════════════════════════════════════
//  2. ALIGNMENT HELPERS
// ═══════════════════════════════════════════════════════════

/// Round `n` up to the nearest multiple of SIMD `LANES`.
#[inline(always)]
pub fn pad_lanes(n: usize) -> usize {
    n.div_ceil(LANES)
        .checked_mul(LANES)
        .expect("pad_lanes: size overflows usize")
}

/// Number of complete SIMD chunks (n / LANES).
#[inline(always)]
pub fn num_chunks(n: usize) -> usize {
    n / LANES
}

/// Scalar tail length (remainder after SIMD chunks).
#[inline(always)]
pub fn tail_len(n: usize) -> usize {
    n % LANES
}

/// Check if pointer is aligned for SIMD access (align=ALIGN).
#[inline(always)]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(ALIGN)
}

/// Check if a slice is aligned.
#[inline(always)]
pub fn is_slice_aligned<T>(slice: &[T]) -> bool {
    is_aligned(slice.as_ptr())
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanes_is_power_of_two() {
        // LANES must be 1, 2, 4, or 8
        const _: () = assert!(LANES == 1 || LANES == 2 || LANES == 4 || LANES == 8);
    }

    #[test]
    fn test_lanes_consistent_with_simd_type() {
        // THE invariant: LANES is derived from the SIMD type, so this can
        // never fail unless SimdType itself changes (in which case every
        // constant here must change with it). Previously LANES was a
        // cfg!-guessed constant that reported 4 on NEON/SSE while the real
        // type was 8-wide — this test would have caught that.
        const _: () =
            assert!(LANES == std::mem::size_of::<SimdType>() / std::mem::size_of::<f32>());
        const _: () = assert!(SIMD_SIZE_BYTES == std::mem::size_of::<SimdType>());
        const _: () = assert!(SIMD_SIZE_BYTES == LANES * 4);
    }

    #[test]
    fn test_pad_lanes_basic() {
        // pad_lanes rounds UP to multiple of LANES
        assert_eq!(pad_lanes(0), 0);
        let padded = pad_lanes(1);
        assert!(padded >= 1);
        assert!(padded.is_multiple_of(LANES));
        assert_eq!(pad_lanes(LANES), LANES);
        assert_eq!(pad_lanes(LANES * 2), LANES * 2);
        assert_eq!(pad_lanes(LANES + 1), LANES * 2);
    }

    #[test]
    fn test_num_chunks_basic() {
        assert_eq!(num_chunks(0), 0);
        assert_eq!(num_chunks(LANES), 1);
        assert_eq!(num_chunks(LANES * 3), 3);
        // n < LANES → 0 chunks
        assert_eq!(num_chunks(LANES - 1), 0);
    }

    #[test]
    fn test_tail_len_basic() {
        assert_eq!(tail_len(0), 0);
        assert_eq!(tail_len(LANES), 0);
        assert_eq!(tail_len(LANES - 1), LANES - 1);
        assert_eq!(tail_len(LANES + 1), 1);
    }

    #[test]
    fn test_is_aligned() {
        // Stack arrays are NOT guaranteed 32-byte aligned — use AlignedVec.
        let av = crate::core::buf::AlignedVec::new(8, 0.0);
        assert!(
            is_slice_aligned(av.as_slice()),
            "AlignedVec should be SIMD-aligned"
        );
        assert!(
            is_aligned(av.as_ptr()),
            "AlignedVec pointer should be SIMD-aligned"
        );
    }

    #[test]
    fn test_simd_size_bytes() {
        assert_eq!(SIMD_SIZE_BYTES, LANES * 4);
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn test_pad_lanes_overflow_is_rejected() {
        let _ = pad_lanes(usize::MAX);
    }

    #[test]
    fn test_pad_lanes_consistency() {
        // pad_lanes(n) >= n for all n
        for n in [0, 1, 3, 7, 13, 100, 1000] {
            let p = pad_lanes(n);
            assert!(p >= n, "pad_lanes({}) = {} < {}", n, p, n);
            assert!(
                p.is_multiple_of(LANES),
                "pad_lanes({}) = {} not multiple of LANES={}",
                n,
                p,
                LANES
            );
        }
    }
}
