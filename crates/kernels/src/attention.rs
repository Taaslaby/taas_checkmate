//! Pure Rust XOR attention + RNG (replaces ASM).
//! Uses CPU popcnt intrinsic for binary attention.
//!
//! XOR attention: sim = max(0, 1 - popcnt(Q⊕K) / n_bits)
//! Returns weighted average of values (or 0 if no weight).

const RNG_SCALE_32: f32 = 2.3283064e-10;

/// XOR attention with popcnt similarity.
///
/// # Safety
/// `query[n_bytes]`, `keys[n_keys·n_bytes]`, `values[n_keys]` valid; `out` non-null.
pub(crate) unsafe fn xor_attention(
    query: *const u8, keys: *const u8, values: *const f32,
    out: *mut f32, n_keys: usize, n_bytes: usize,
) {
    let q = std::slice::from_raw_parts(query, n_bytes);
    let k = std::slice::from_raw_parts(keys, n_keys * n_bytes);
    let v = std::slice::from_raw_parts(values, n_keys);
    let n_bits = (n_bytes * 8) as f32;
    let mut total_val = 0.0f32;
    let mut total_w = 0.0f32;

    for i in 0..n_keys {
        let mut diff = 0u32;
        for b in 0..n_bytes {
            diff += (q[b] ^ k[i * n_bytes + b]).count_ones();
        }
        let sim = 1.0 - diff as f32 / n_bits;
        if sim > 0.0 {
            total_w += sim;
            total_val += sim * v[i];
        }
    }
    *out = if total_w > 0.0 { total_val / total_w } else { 0.0 };
}

/// Fill buffer with pseudo-random floats in [0, 1) using xorshift32.
///
/// # Safety
/// `buf` must be non-null, aligned, valid for `n` elements.
pub(crate) unsafe fn rdrand_fill(buf: *mut f32, n: usize) {
    let b = std::slice::from_raw_parts_mut(buf, n);
    let mut state: u32 = 42;
    for v in b.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *v = state as f32 * RNG_SCALE_32;
    }
}
