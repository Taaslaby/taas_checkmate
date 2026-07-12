//! Circular delay buffer with AVX2 read.

use crate::kernels;

/// Circular delay buffer with SIMD read.
pub struct DelayBuffer {
    buf: Vec<f32>,
    n: usize,
    pos: usize,
}

impl DelayBuffer {
    pub fn new(n: usize) -> Self {
        let padded = n.div_ceil(8) * 8;
        Self { buf: vec![0.0f32; padded], n, pos: 0 }
    }

    pub fn push(&mut self, data: &[f32]) {
        for (i, &v) in data.iter().enumerate() {
            let idx = (self.pos + i) % self.n;
            self.buf[idx] = v;
        }
        self.pos = (self.pos + data.len()) % self.n;
    }

    pub fn read(&self, delay: usize) -> Vec<f32> {
        let n_out = self.n.min(8);
        let mut out = vec![0.0f32; n_out];
        let read_pos = (self.pos + self.n - delay % self.n) % self.n;
        // Manual copy with wrap-around.
        // Note: delay_buffer_circular requires output.len() == buffer.len(),
        // but we want a smaller output. So we copy manually.
        for i in 0..n_out {
            let idx = (read_pos + i) % self.n;
            out[i] = self.buf[idx];
        }
        out
    }

    pub fn read_multi(&self, delays: &[i32]) -> Vec<f32> {
        let n_taps = delays.len().min(self.n).min(8);
        let mut out = vec![0.0f32; n_taps];
        let wrapped: Vec<i32> = delays.iter().map(|&d| {
            let abs_d = d.unsigned_abs() as usize % self.n;
            if d < 0 { -(abs_d as i32) } else { abs_d as i32 }
        }).collect();
        kernels::delay_buffer_multi_tap(&mut out, &self.buf[..self.n], &wrapped);
        out
    }
}
