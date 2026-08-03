//! # Compact Spikes — SIMD move_mask + tzcnt compaction
//!
//! Extracts indices of active neurons from a dense spike array.
//!
//! ## How it works
//! Spikes arrive as `i32` 0/1. Each chunk of 8 is converted — branchlessly —
//! into an `f32x8` where an active spike is `-0.0` (sign bit set) and a
//! silent one is `+0.0`: `((spike != 0) as u32) << 31` is the sign bit.
//! One `move_mask()` then extracts the sign bits of all 8 lanes in a single
//! `vmovmskps` (AVX2), and `trailing_zeros` (`tzcnt`) walks only the set bits.
//!
//! For 1–5% sparsity (typical SNN) this avoids up to 8 data-dependent
//! branches per chunk — the previous implementation paid them on every
//! non-empty chunk.

use wide::f32x8;

/// Extract indices of active (non-zero) neurons into a reusable buffer.
///
/// SIMD fast path: 8 spikes per chunk via `move_mask` + `trailing_zeros`
/// (branchless, one `vmovmskps` per chunk on AVX2). Scalar tail for the
/// remainder. Semantics are identical to the scalar reference:
/// `out = [i for (i, s) in spikes.iter().enumerate() if *s != 0]`.
#[inline]
pub fn compact_spikes_into(spikes: &[i32], out: &mut Vec<u32>) {
    let n = spikes.len();
    let n8 = n & !7;
    let mut i = 0;

    // SIMD path: sign-bit trick, 8 lanes at a time.
    while i < n8 {
        let mut bits = [0u32; 8];
        for k in 0..8 {
            bits[k] = ((spikes[i + k] != 0) as u32) << 31;
        }
        let lanes = f32x8::from(bits.map(f32::from_bits));
        let mask = lanes.move_mask();
        if mask != 0 {
            let mut m = mask as u32;
            while m != 0 {
                let tz = m.trailing_zeros() as usize;
                out.push((i + tz) as u32);
                m &= m - 1; // clear lowest set bit
            }
        }
        i += 8;
    }

    // Scalar tail (n % 8 remainder).
    while i < n {
        if spikes[i] != 0 {
            out.push(i as u32);
        }
        i += 1;
    }
}

/// Legacy: allocates a new Vec each call. Prefer [`compact_spikes_into`].
#[inline]
pub fn compact_spikes(spikes: &[i32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(spikes.len() / 20 + 8);
    compact_spikes_into(spikes, &mut out);
    out
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_empty() {
        let spikes = vec![0i32; 100];
        let active = compact_spikes(&spikes);
        assert!(active.is_empty());
    }

    #[test]
    fn test_compact_some() {
        let spikes: Vec<i32> = (0..20).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let active = compact_spikes(&spikes);
        assert_eq!(active, vec![0, 3, 6, 9, 12, 15, 18]);
    }

    #[test]
    fn test_compact_all() {
        let spikes = vec![1i32; 10];
        let active = compact_spikes(&spikes);
        assert_eq!(active.len(), 10);
        assert_eq!(active, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_compact_into_reuse() {
        let spikes = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let mut out = Vec::with_capacity(10);
        compact_spikes_into(&spikes, &mut out);
        assert_eq!(out, vec![1, 3, 5, 7, 9]);
        // Reuse
        out.clear();
        compact_spikes_into(&spikes, &mut out);
        assert_eq!(out, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_compact_simd_matches_scalar_reference() {
        // Pseudo-random spikes over a non-multiple-of-8 length: the SIMD
        // fast path (first n8) plus the scalar tail must exactly match a
        // scalar reference implementation.
        let n = 1003usize; // 125 SIMD chunks + 3 tail elements
        let spikes: Vec<i32> = (0..n)
            .map(|i| {
                if (i as u64).wrapping_mul(2654435761).wrapping_rem(997) < 97 {
                    1
                } else {
                    0
                }
            })
            .collect();

        let mut out = Vec::new();
        compact_spikes_into(&spikes, &mut out);

        let reference: Vec<u32> = spikes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(out, reference);
        assert!(out.len() > 80, "expected sparse activity, got {}", out.len());
    }

    #[test]
    fn test_compact_dense_64() {
        // Dense case: every chunk active — tzcnt loop must emit all 64.
        let spikes = vec![1i32; 64];
        let mut out = Vec::new();
        compact_spikes_into(&spikes, &mut out);
        assert_eq!(out, (0..64).collect::<Vec<u32>>());
    }
}
