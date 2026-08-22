//! Circular delay buffer for NMJ tap reads.

// ═══════════════════════════════════════════════════════════
//  Circular delay buffer
// ═══════════════════════════════════════════════════════════

/// Circular delay buffer with SIMD read.
///
/// Отдельный от `api::delay::DelayBuffer` (сетевой ring buffer) — этот
/// класс обслуживает NMJ-задержки и tap-чтение (push/read/read_multi).
#[derive(Debug, Clone)]
pub struct DelayBuffer {
    buf: Vec<f32>,
    n: usize,
    pos: usize,
}

impl DelayBuffer {
    pub fn new(n: usize) -> Self {
        // n = 0 would make push/read compute `% 0` (divide-by-zero panic on
        // first use) — reject at construction instead.
        assert!(n > 0, "DelayBuffer::new: capacity must be > 0, got 0");
        let padded = n.div_ceil(8) * 8;
        Self {
            buf: vec![0.0f32; padded],
            n,
            pos: 0,
        }
    }

    pub fn push(&mut self, data: &[f32]) {
        for (i, &v) in data.iter().enumerate() {
            let idx = (self.pos + i) % self.n;
            self.buf[idx] = v;
        }
        self.pos = (self.pos + data.len()) % self.n;
    }

    /// The most recently pushed value, without allocating (unlike
    /// [`read`](Self::read), which returns a fresh `Vec`).
    pub fn latest(&self) -> f32 {
        self.buf[(self.pos + self.n - 1) % self.n]
    }

    pub fn read(&self, delay: usize) -> Vec<f32> {
        let n_out = self.n.min(8);
        let mut out = vec![0.0f32; n_out];
        let read_pos = (self.pos + self.n - delay % self.n) % self.n;
        for (i, o) in out.iter_mut().enumerate() {
            let idx = (read_pos + i) % self.n;
            *o = self.buf[idx];
        }
        out
    }

    pub fn read_multi(&self, delays: &[i32]) -> Vec<f32> {
        let n_taps = delays.len().min(self.n).min(8);
        let mut out = vec![0.0f32; n_taps];
        let wrapped: Vec<i32> = delays
            .iter()
            .map(|&d| {
                let abs_d = d.unsigned_abs() as usize % self.n;
                if d < 0 {
                    -(abs_d as i32)
                } else {
                    abs_d as i32
                }
            })
            .collect();
        crate::kernels::delay::delay_buffer_multi_tap(&mut out, &self.buf[..self.n], &wrapped);
        out
    }
}
