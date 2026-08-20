//! # Delay Buffer Microkernels
//!
//! Delay buffers are memory-bound — scalar copies, no SIMD benefit.

/// Circular delay buffer: shift copy from position `pos`.
///
/// `output = [buffer[pos..], buffer[..pos]]` — the classic ring-buffer readout.
pub fn delay_buffer_circular(output: &mut [f32], buffer: &[f32], pos: usize) {
    let n = output.len().min(buffer.len());
    if n == 0 {
        return;
    }
    let pos = pos % n;
    let len = n - pos;
    output[..len].copy_from_slice(&buffer[pos..n]);
    output[len..n].copy_from_slice(&buffer[..pos]);
}

/// Multi-tap delay: read `buffer` at several delay indices.
///
/// Each tap is clamped to `[0, n-1]`; `delays.len() == output.len()`.
pub fn delay_buffer_multi_tap(output: &mut [f32], buffer: &[f32], delays: &[i32]) {
    let n = buffer.len();
    if n == 0 {
        output.fill(0.0);
        return;
    }
    let n_taps = output.len().min(delays.len());
    for (i, &tap) in delays[..n_taps].iter().enumerate() {
        let idx = (tap as usize).min(n.saturating_sub(1));
        output[i] = buffer[idx];
    }
}

#[cfg(test)]
mod tests;
