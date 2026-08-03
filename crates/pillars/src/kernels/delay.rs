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
mod tests {
    use super::*;

    #[test]
    fn delay_buffer_circular_basic() {
        let buffer: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 8];
        delay_buffer_circular(&mut output, &buffer, 3);
        assert_eq!(output, vec![3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn delay_buffer_multi_tap_basic() {
        let buffer: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let delays = vec![0i32, 3, 7, 9];
        let mut output = vec![0.0f32; 4];
        delay_buffer_multi_tap(&mut output, &buffer, &delays);
        assert_eq!(output, vec![0.0, 3.0, 7.0, 9.0]);
    }

    #[test]
    fn delay_buffer_tap_clamped() {
        let buffer: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let delays = vec![100i32];
        let mut output = vec![0.0f32; 1];
        delay_buffer_multi_tap(&mut output, &buffer, &delays);
        assert_eq!(output[0], 3.0, "tap beyond end clamps to last element");
    }
}
