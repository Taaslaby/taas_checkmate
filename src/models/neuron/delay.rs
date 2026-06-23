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

    pub fn read(&self, delay: usize) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        let read_pos = (self.pos + self.n - delay % self.n) % self.n;
        unsafe { kernels::delay_buffer_circular(out.as_mut_ptr(), self.buf.as_ptr(), self.n, read_pos); }
        out
    }

    pub fn read_multi(&self, delays: &[i32; 8]) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { kernels::delay_buffer_multi_tap(out.as_mut_ptr(), self.buf.as_ptr(), delays.as_ptr(), self.n, 8); }
        out
    }
}
