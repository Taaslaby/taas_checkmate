//! # TrajectoryBuffer — ring buffer for L-frame trajectories.
//!
//! Из оригинального Nakajima NeuroBrain trajectory.rs.
//! Хранит per-frame features в кольцевом буфере и строит
//! L-кадровые траектории для PstDecoder/Ridge.

/// Ring buffer for L-frame PST trajectories.
pub struct TrajectoryBuffer {
    pub buf: Vec<f32>,
    pub capacity: usize,
    pub feat_dim: usize,
    pub pos: usize,
    pub count: usize,
}

impl TrajectoryBuffer {
    pub fn new(capacity: usize, feat_dim: usize) -> Self {
        Self {
            buf: vec![0.0; capacity * feat_dim],
            capacity,
            feat_dim,
            pos: 0,
            count: 0,
        }
    }

    /// Push one frame of features into the ring buffer.
    pub fn push(&mut self, features: &[f32]) {
        let start = self.pos * self.feat_dim;
        let end = start + self.feat_dim.min(features.len());
        self.buf[start..end].copy_from_slice(&features[..self.feat_dim.min(features.len())]);
        self.pos = (self.pos + 1) % self.capacity;
        self.count = (self.count + 1).min(self.capacity);
    }

    /// Build trajectory by concatenating L consecutive frames.
    pub fn trajectory(&self, L: usize) -> Vec<f32> {
        if self.count < L || L == 0 { return vec![]; }
        let mut traj = Vec::with_capacity(L * self.feat_dim);
        let start = if self.pos >= L { self.pos - L } else { self.capacity - (L - self.pos) };
        for k in 0..L {
            let idx = (start + k) % self.capacity;
            let off = idx * self.feat_dim;
            traj.extend_from_slice(&self.buf[off..off + self.feat_dim]);
        }
        traj
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool { self.count >= self.capacity }

    /// Clear buffer.
    pub fn clear(&mut self) { self.pos = 0; self.count = 0; self.buf.fill(0.0); }

    /// Current number of frames in buffer.
    pub fn len(&self) -> usize { self.count }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize { self.capacity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_buffer_push() {
        let mut buf = TrajectoryBuffer::new(4, 3);
        assert!(!buf.is_full());
        buf.push(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.len(), 1);
        buf.push(&[4.0, 5.0, 6.0]);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_trajectory_buffer_trajectory() {
        let mut buf = TrajectoryBuffer::new(4, 3);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.push(&[4.0, 5.0, 6.0]);
        buf.push(&[7.0, 8.0, 9.0]);
        let t = buf.trajectory(2);
        assert_eq!(t.len(), 6);
        assert!((t[0] - 4.0).abs() < 1e-6, "t[0]={}", t[0]);
        assert!((t[3] - 7.0).abs() < 1e-6, "t[3]={}", t[3]);
    }

    #[test]
    fn test_trajectory_buffer_wrap_around() {
        let mut buf = TrajectoryBuffer::new(3, 2);
        for i in 0..5 {
            buf.push(&[i as f32, (i + 1) as f32]);
        }
        assert_eq!(buf.len(), 3);
        let t = buf.trajectory(3);
        assert_eq!(t.len(), 6);
        // After 5 pushes, buf has [push3, push4, push5] at positions [0,1,2]
        // trajectory(3) should return them in order
        assert!((t[0] - 4.0).abs() < 1e-6 || (t[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_buffer_clear() {
        let mut buf = TrajectoryBuffer::new(4, 3);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.trajectory(2).is_empty());
    }
}
