#![allow(non_snake_case)]

// ═══════════════════════════════════════════════════════
//  Delay Buffer
// ═══════════════════════════════════════════════════════

// Delay buffer functions are memory-bound — keep scalar (no SIMD benefit)

/// Circular delay buffer: shift copy from position `pos`.
///
/// # Safety
/// `output[n]`, `buffer[n]` must be valid, non-overlapping.
pub(crate) unsafe fn delay_buffer_circular(
    output: *mut f32, buffer: *const f32,
    n: usize, pos: usize,
) {
    if n == 0 { return; }
    let buf = std::slice::from_raw_parts(buffer, n);
    let out = std::slice::from_raw_parts_mut(output, n);
    let len = n - pos;
    out[..len].copy_from_slice(&buf[pos..]);
    out[len..].copy_from_slice(&buf[..pos]);
}

/// Multi-tap delay: read buffer at 8 different delay indices.
///
/// # Safety
/// `output[n_taps]`, `buffer[n]`, `delays[n_taps]` valid, non-overlapping.
pub(crate) unsafe fn delay_buffer_multi_tap(
    output: *mut f32, buffer: *const f32,
    delays: *const i32, n: usize, n_taps: usize,
) {
    let buf = std::slice::from_raw_parts(buffer, n);
    let out = std::slice::from_raw_parts_mut(output, n_taps);
    let d = std::slice::from_raw_parts(delays, n_taps);
    for (i, tap) in d.iter().enumerate() {
        let idx = (*tap as usize).min(n.saturating_sub(1));
        out[i] = buf[idx];
    }
}
