//! Architecture detection — compile-time SIMD width & alignment.
//!
//! Selects the optimal SIMD width for the current target architecture
//! using `#[cfg]` compile-time dispatch (not `cfg!()` runtime).
//!
//! | Architecture | Type | LANES | ALIGN | Notes |
//! |:------------|:----|:-----|:------|:------|
//! | x86_64 AVX2 | f32x8 | 8 | 32 | 256-bit, 8 f32/cycle |
//! | x86_64 SSE  | f32x4 | 4 | 16 | Legacy CPU fallback |
//! | ARM NEON    | f32x4 | 4 | 16 | aarch64, 128-bit SIMD |
//! | WASM simd128| f32x4 | 4 | 16 | 128-bit SIMD |
//! | Fallback    | f32   | 1 | 4  | Any unknown arch |

use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  1. SIMD WIDTH — compile-time via #[cfg]
// ═══════════════════════════════════════════════════════════

// `LANES` = hardware SIMD width (8 on AVX2, 4 on NEON/SSE, 1 on scalar).
// `wide::f32x8` is ALWAYS 8-wide regardless of hardware — NEON emulates 2×128-bit.
#[cfg(target_feature = "avx2")]
mod simd_width { pub const LANES: usize = 8; }

#[cfg(not(target_feature = "avx2"))]
mod simd_width {
    pub const LANES: usize = if cfg!(any(target_feature = "neon", target_feature = "sse")) { 4 } else { 1 };
}

pub use simd_width::*;

/// Alignment for `f32x8` — always 32 bytes (`wide::f32x8` guarantees `align(32)`).
pub const ALIGN: usize = std::mem::align_of::<f32x8>();

/// SIMD register size in bytes (= LANES * 4).
pub const SIMD_SIZE_BYTES: usize = LANES * 4;

// ═══════════════════════════════════════════════════════════
//  2. SIMD TYPE (always f32x8)
// ═══════════════════════════════════════════════════════════

/// SIMD type: always `f32x8` (8-wide). Native 256-bit on AVX2,
/// emulated as 2× f32x4 on NEON/SSE.
pub type SimdType = f32x8;

// ═══════════════════════════════════════════════════════════
//  3. ALIGNMENT HELPERS
// ═══════════════════════════════════════════════════════════

/// Round `n` up to the nearest multiple of SIMD `LANES`.
#[inline(always)]
pub fn pad_lanes(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
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
        assert!(is_slice_aligned(av.as_slice()), "AlignedVec should be SIMD-aligned");
        assert!(is_aligned(av.as_ptr()), "AlignedVec pointer should be SIMD-aligned");
    }

    #[test]
    fn test_simd_size_bytes() {
        assert_eq!(SIMD_SIZE_BYTES, LANES * 4);
    }

    #[test]
    fn test_pad_lanes_consistency() {
        // pad_lanes(n) >= n for all n
        for n in [0, 1, 3, 7, 13, 100, 1000] {
            let p = pad_lanes(n);
            assert!(p >= n, "pad_lanes({}) = {} < {}", n, p, n);
            assert!(p.is_multiple_of(LANES), "pad_lanes({}) = {} not multiple of LANES={}", n, p, LANES);
        }
    }
}
