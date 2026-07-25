//! Unique koit ASM kernels: XOR attention + hardware RNG.
//!
//! These functions use Win64 calling convention (`-mabi=ms` in build.rs).
//!
//! ## Functions
//!
//! - `xor_attention` — binary attention: XOR + popcnt, 0 float multiplications
//! - `rdrand_fill` — hardware RNG via CPU rdrand instruction
//!
//! это тупо побитовое сложение по модулю 2 с последующим подсчетом количества единиц
//! зато у меня даже работает быстро без всяких ваших векторизаций
//! а что если квантовать откроется магия?
//! а если сделать нейронку от Koit нейронку?   

use crate::kernels;

/// XOR attention: weighted average of values by XOR similarity between binary query and keys.
///
/// Formula per key:
///   dist = popcnt(query XOR key)
///   sim  = max(0, 1 - dist / n_bits)
///   out  = Σ (sim_i × value_i) / Σ sim_i    (normalized by sum of similarities)
///
/// If Σ sim_i = 0 (no matches), returns 0.
/// Zero floating-point multiplications until the final weighted average.
/// n_bytes must be a multiple of 8 (64-bit chunks for popcnt).
pub fn xor_attention(
    query: &[u8],
    keys: &[u8],
    values: &[f32],
    out: &mut [f32],
    n_keys: usize,
    n_bytes: usize,
) {
    assert!(n_bytes.is_multiple_of(8), "n_bytes must be multiple of 8");
    assert!(query.len() >= n_bytes);
    assert!(keys.len() >= n_keys * n_bytes);
    assert!(values.len() >= n_keys);
    assert!(!out.is_empty());
    kernels::xor_attention(
        &query[..n_bytes],
        &keys[..n_keys * n_bytes],
        &values[..n_keys],
        &mut out[0],
    );
}

/// Fill buffer with pseudo-random floats in [0, 1) via xorshift32.
pub fn rdrand_fill(buf: &mut [f32]) {
    kernels::rdrand_fill(buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_attention_basic() {
        let n_keys = 4;
        let n_bytes = 8;
        // Query and keys: first key matches exactly, others differ
        let query = vec![0xFFu8; n_bytes];
        let keys = vec![
            0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // key 0: exact match
            0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // key 1: all flipped
            0xFFu8, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, // key 2: half match
            0x00u8, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, // key 3: half match
        ];
        let values = vec![1.0f32, 0.0, 0.5, 0.5];
        let mut out = vec![0.0f32; 1];

        xor_attention(&query, &keys, &values, &mut out, n_keys, n_bytes);

        // XOR attention normalizes by sum of similarities:
        //   sim = max(0, 1 - popcnt/dist)
        //   Key 0 (exact match) → sim=1.0
        //   Key 1 (all flipped) → sim=0.0
        //   Keys 2,3 (half match) → sim=0.5
        //   Σsim = 1.0 + 0.0 + 0.5 + 0.5 = 2.0
        //   out = (1.0*1.0 + 0.0*0.0 + 0.5*0.5 + 0.5*0.5) / Σsim = 1.5 / 2.0 = 0.75
        let expected = 0.75;
        assert!(
            (out[0] - expected).abs() < 0.1,
            "xor_attention: expected ~{:.3}, got {:.3}",
            expected,
            out[0]
        );
    }

    #[test]
    fn test_xor_attention_all_same() {
        let n_keys = 3;
        let n_bytes = 8;
        let query = vec![0xAAu8; n_bytes];
        // All keys identical to query
        let keys = vec![0xAAu8; n_keys * n_bytes];
        let values = vec![2.0f32, 3.0, 4.0];
        let mut out = vec![0.0f32; 1];

        xor_attention(&query, &keys, &values, &mut out, n_keys, n_bytes);

        // All match perfectly → sim = 1.0 for all → mean of values
        let expected = (2.0 + 3.0 + 4.0) / 3.0;
        assert!(
            (out[0] - expected).abs() < 0.1,
            "xor_attention: all same, expected ~{:.1}, got {:.3}",
            expected,
            out[0]
        );
    }

    #[test]
    fn test_xor_attention_no_match() {
        let n_keys = 2;
        let n_bytes = 8;
        let query = vec![0xFFu8; n_bytes];
        // All keys opposite
        let keys = vec![0x00u8; n_keys * n_bytes];
        let values = vec![10.0f32, 20.0];
        let mut out = vec![0.0f32; 1];

        xor_attention(&query, &keys, &values, &mut out, n_keys, n_bytes);

        // All flipped → sim = 0 for all
        // out = 0
        assert!(
            (out[0] - 0.0).abs() < 0.1,
            "xor_attention: no match, expected ~0.0, got {:.3}",
            out[0]
        );
    }

    #[test]
    fn test_rdrand_fill_no_nan() {
        let mut buf = vec![0.0f32; 16];
        rdrand_fill(&mut buf);

        // All values should be in [0, 1) and not NaN
        for &v in &buf {
            assert!(
                v >= 0.0 && v < 1.0,
                "rdrand_fill: value {:.6} out of range [0, 1)",
                v
            );
        }

        // Not all the same (at least some randomness)
        let unique: std::collections::HashSet<u32> = buf
            .iter()
            .map(|&v| v.to_bits() >> 12) // coarse buckets
            .collect();
        assert!(
            unique.len() > 1,
            "rdrand_fill: should produce varied values"
        );
    }

    #[test]
    fn test_rdrand_fill_large_buffer() {
        let mut buf = vec![0.0f32; 128];
        rdrand_fill(&mut buf);

        // All valid
        for &v in &buf {
            assert!(!v.is_nan(), "rdrand_fill should not produce NaN");
            assert!(
                v >= 0.0 && v < 1.0,
                "rdrand_fill value out of range: {:.6}",
                v
            );
        }
    }
}
