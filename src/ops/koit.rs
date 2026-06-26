//! Unique koit ASM kernels: XOR attention + hardware RNG.
//!
//! These functions use Win64 calling convention (`-mabi=ms` in build.rs).
//!
//! ## Functions
//!
//! - `xor_attention` — binary attention: XOR + popcnt, 0 float multiplications
//! - `rdrand_fill` — hardware RNG via CPU rdrand instruction

use crate::kernels;

/// XOR attention: weighted sum of values by XOR similarity between binary query and keys.
///
/// Formula per key:
///   dist = popcnt(query XOR key)
///   sim  = max(0, 1 - dist / (n_bits))
///   out  = Σ sim_i × value_i
///
/// Zero floating-point multiplications until the final weighted sum.
/// n_bytes must be a multiple of 8 (64-bit chunks for popcnt).
pub fn xor_attention(
    query: &[u8], keys: &[u8], values: &[f32],
    out: &mut [f32], n_keys: usize, n_bytes: usize,
) {
    assert!(n_bytes.is_multiple_of(8), "n_bytes must be multiple of 8");
    assert!(query.len() >= n_bytes);
    assert!(keys.len() >= n_keys * n_bytes);
    assert!(values.len() >= n_keys);
    assert!(!out.is_empty());
    kernels::xor_attention(&query[..n_bytes], &keys[..n_keys * n_bytes], &values[..n_keys], &mut out[0]);
}

/// Fill buffer with pseudo-random floats in [0, 1) via xorshift32.
pub fn rdrand_fill(buf: &mut [f32]) {
    kernels::rdrand_fill(buf);
}
