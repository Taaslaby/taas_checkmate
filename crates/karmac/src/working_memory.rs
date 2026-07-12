//! Working Memory — Prefrontal Cortex-like state retention.
//!
//! Inspired by SPAUN's prefrontal cortex: maintains active representations
//! across time steps through recurrent excitation + leaky decay.
//!
//! # Architecture
//!
//! ```text
//! Input → [W_in] → Working Memory Buffer (n_slots × dim)
//!                  ↺ recurrent excitation
//!                  ↓ leaky decay
//!                  ↓ noise gating
//!        Output = buffer[slot]
//! ```
//!
//! Each slot holds a high-dimensional vector (semantic pointer).
//! Slots are updated via:
//! 1. **New input** → overwrite slot 0 (or least-recently-used slot)
//! 2. **Recurrent** → buffer = decay * buffer + W_rec @ buffer
//! 3. **Gate** → noise threshold determines if update happens
//!
//! # Usage
//! ```rust
//! use karmac::working_memory::WorkingMemory;
//!
//! let mut wm = WorkingMemory::new(4, 128); // 4 slots, dim=128
//! wm.store(&vec![0.1; 128]);  // store in slot 0
//! wm.store(&vec![0.2; 128]);  // store in slot 1
//! let retrieved = wm.recall(0); // retrieve slot 0
//! assert_eq!(retrieved.len(), 128);
//! ```

use rand::Rng;

/// Working Memory buffer.
pub struct WorkingMemory {
    /// Number of memory slots.
    pub n_slots: usize,
    /// Dimensionality of each slot (semantic pointer size).
    pub dim: usize,
    /// Memory buffer [n_slots × dim], row-major.
    buffer: Vec<f32>,
    /// Recurrent weight matrix [dim × dim].
    w_rec: Vec<f32>,
    /// Input projection weights [dim × dim].
    w_in: Vec<f32>,
    /// Slot ages (for LRU eviction).
    ages: Vec<u32>,
    /// Current write pointer (round-robin).
    write_ptr: usize,
    /// Leak decay factor per step.
    pub decay: f32,
    /// Noise gate threshold (0 = no gate, 1 = always gate).
    pub gate_threshold: f32,
    /// Recurrent gain factor (controls attractor strength).
    pub recurrent_gain: f32,
    /// Step counter.
    step: u64,
    /// Pre-allocated delta buffer — avoids heap alloc in hot path.
    delta_buf: Vec<f32>,
}

impl WorkingMemory {
    /// Create working memory with Xavier-initialized recurrent weights.
    pub fn new(n_slots: usize, dim: usize) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let scale = (2.0 / dim as f32).sqrt();

        let w_rec: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();

        let w_in: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();

        Self {
            n_slots,
            dim,
            buffer: vec![0.0; n_slots * dim],
            w_rec,
            w_in,
            ages: vec![0; n_slots],
            write_ptr: 0,
            decay: 0.9,
            gate_threshold: 0.1,
            recurrent_gain: 0.01,
            step: 0,
            delta_buf: vec![0.0; n_slots * dim],
        }
    }

    /// Store a vector into working memory (round-robin slots).
    ///
    /// Input is projected through W_in before storage.
    pub fn store(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.dim);
        self.step += 1;

        // Gate: only store if input is above threshold
        let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < self.gate_threshold {
            return;
        }

        // Project input through W_in
        let mut projected = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let row_start = j * self.dim;
            projected[j] = self.w_in[row_start..row_start + self.dim]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum();
        }

        // Write to current slot
        let slot = self.write_ptr;
        let off = slot * self.dim;
        self.buffer[off..off + self.dim].copy_from_slice(&projected);
        self.ages[slot] = 0;

        // Advance write pointer (round-robin)
        self.write_ptr = (self.write_ptr + 1) % self.n_slots;

        // Age all other slots
        for i in 0..self.n_slots {
            if i != slot {
                self.ages[i] += 1;
            }
        }
    }

    /// Recall a specific slot by index.
    pub fn recall(&self, slot: usize) -> Vec<f32> {
        assert!(slot < self.n_slots, "slot {} out of range (n={})", slot, self.n_slots);
        let off = slot * self.dim;
        self.buffer[off..off + self.dim].to_vec()
    }

    /// Recall all slots as a flat vector [n_slots × dim].
    pub fn recall_all(&self) -> &[f32] {
        &self.buffer
    }

    /// Get the least-recently-used slot index.
    pub fn lru_slot(&self) -> usize {
        self.ages.iter()
            .enumerate()
            .max_by_key(|(_, &age)| age)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Run one step of recurrent dynamics (without new input).
    ///
    /// buffer = decay * buffer + W_rec @ buffer (modulated)
    pub fn tick(&mut self) {
        self.step += 1;
        let dim = self.dim;
        let n = self.n_slots;

        // Apply decay
        for x in self.buffer.iter_mut() {
            *x *= self.decay;
        }

        // Recurrent excitation: for each slot, apply W_rec
        // This maintains attractor dynamics (Hopfield-like)
        self.delta_buf.fill(0.0);
        for slot in 0..n {
            let off = slot * dim;
            let slot_state = &self.buffer[off..off + dim];
            for j in 0..dim {
                let row_start = j * dim;
                self.delta_buf[off + j] = self.w_rec[row_start..row_start + dim]
                    .iter()
                    .zip(slot_state.iter())
                    .map(|(w, s)| w * s)
                    .sum::<f32>()
                    * self.recurrent_gain;
            }
        }

        // Add recurrent delta
        for (x, d) in self.buffer.iter_mut().zip(self.delta_buf.iter()) {
            *x += d;
        }
    }

    /// Clear all slots.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.ages.fill(0);
        self.write_ptr = 0;
    }

    /// Get current step count.
    pub fn step_count(&self) -> u64 { self.step }

    pub fn n_slots(&self) -> usize { self.n_slots }
    pub fn dim(&self) -> usize { self.dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wm_store_and_recall() {
        let mut wm = WorkingMemory::new(4, 8);
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        wm.store(&v);
        let recalled = wm.recall(0);
        // After projection, values should differ from input
        assert_eq!(recalled.len(), 8);
        // But should not be zero (projection is non-trivial)
        assert!(recalled.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn wm_round_robin() {
        let mut wm = WorkingMemory::new(3, 4);
        let v1 = vec![1.0; 4];
        let v2 = vec![2.0; 4];
        let v3 = vec![3.0; 4];
        wm.store(&v1);
        wm.store(&v2);
        wm.store(&v3);
        // All 3 slots should be written
        assert_eq!(wm.write_ptr, 0); // wrapped around
    }

    #[test]
    fn wm_lru() {
        let mut wm = WorkingMemory::new(3, 4);
        let v = vec![1.0; 4];
        wm.store(&v); // slot 0, age[0]=0, others=1
        wm.store(&v); // slot 1, age[1]=0, others age
        // Slot 2 is oldest (never written)
        assert_eq!(wm.lru_slot(), 2);
    }

    #[test]
    fn wm_gate() {
        let mut wm = WorkingMemory::new(2, 4);
        wm.gate_threshold = 10.0; // high threshold
        let tiny = vec![0.001; 4]; // below threshold
        wm.store(&tiny);
        // Should not have written (norm < threshold)
        let recalled = wm.recall(0);
        assert!(recalled.iter().all(|&x| x == 0.0), "should not store below gate threshold");
    }

    #[test]
    fn wm_tick_no_nan() {
        let mut wm = WorkingMemory::new(4, 16);
        let v: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        wm.store(&v);
        for _ in 0..100 {
            wm.tick();
        }
        let recalled = wm.recall(0);
        for &x in &recalled {
            assert!(x.is_finite(), "no NaN/Inf after tick");
        }
    }

    #[test]
    fn wm_clear() {
        let mut wm = WorkingMemory::new(2, 4);
        wm.store(&vec![1.0; 4]);
        wm.clear();
        assert!(wm.recall(0).iter().all(|&x| x == 0.0));
        assert_eq!(wm.write_ptr, 0);
    }

    #[test]
    fn wm_recall_all_shape() {
        let wm = WorkingMemory::new(3, 8);
        let all = wm.recall_all();
        assert_eq!(all.len(), 3 * 8);
    }
}
