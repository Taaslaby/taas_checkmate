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
    unsafe {
        kernels::xor_attention(
            query.as_ptr(), keys.as_ptr(), values.as_ptr(),
            out.as_mut_ptr(), n_keys, n_bytes,
        );
    }
}

/// Fill buffer with pseudo-random floats in [0, 1) via xorshift32.
/// `n` must be a multiple of 8.
pub fn rdrand_fill(buf: &mut [f32]) {
    let n = buf.len();
    assert!(n.is_multiple_of(8), "n must be multiple of 8 for AVX");
    unsafe { kernels::rdrand_fill(buf.as_mut_ptr(), n); }
}
