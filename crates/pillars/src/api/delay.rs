//! # DelayBuffer — Ring Buffer of Synaptic Currents
//!
//! === BOUNDARY: Safe Rust → unsafe SoA ===
//! Owns `MAX_DELAY_SLOTS` aligned current vectors, one per future
//! simulation step. On spike, weights are scattered into the slot
//! `(step + delay) % MAX_DELAY_SLOTS`; each step reads and clears the
//! current slot.
//!
//! ## Ring-buffer ordering (read-before-write)
//!
//! ```text
//! step s:  read  slot[s % 32]  → i_in
//!          clear slot[s % 32]
//!          step neurons
//!          write spikes into slot[(s + delay) % 32]  (delay ≥ 1)
//! ```
//!
//! A spike fired at step `s` is delivered at step `s + max(delay, 1)` —
//! it can never land in the slot that was already read and cleared.

use crate::api::graph::MAX_DELAY_SLOTS;
use crate::core::buf::AlignedVec;

/// Ring buffer of per-neuron synaptic currents.
///
/// Layout: `[MAX_DELAY_SLOTS][n_neurons]` (SoA, SIMD-aligned columns).
#[derive(Debug, Clone)]
pub struct DelayBuffer {
    /// One current vector per delay slot.
    slots: Vec<AlignedVec>,
    /// Number of neurons per slot (logical length, not padded capacity).
    n_neurons: usize,
}

impl DelayBuffer {
    /// Create a buffer for `n_neurons` with all slots zeroed.
    pub fn new(n_neurons: usize) -> Self {
        let slots = (0..MAX_DELAY_SLOTS)
            .map(|_| AlignedVec::new(n_neurons, 0.0))
            .collect();
        Self { slots, n_neurons }
    }

    /// Number of delay slots.
    #[inline(always)]
    pub fn num_slots(&self) -> usize {
        MAX_DELAY_SLOTS
    }

    /// Number of neurons per slot.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n_neurons
    }

    /// Returns `true` if the buffer holds no neurons.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n_neurons == 0
    }

    /// Read the current arriving at each neuron from slot `slot`.
    ///
    /// Copies into `out` (length must be >= [`DelayBuffer::len`]).
    #[inline(always)]
    pub fn read_slot(&self, slot: usize, out: &mut [f32]) {
        let src = &self.slots[slot % MAX_DELAY_SLOTS].as_slice()[..self.n_neurons];
        out[..self.n_neurons].copy_from_slice(src);
    }

    /// Zero the current slot after reading it (frees it for future writes).
    #[inline(always)]
    pub fn clear_slot(&mut self, slot: usize) {
        self.slots[slot % MAX_DELAY_SLOTS].as_mut_slice()[..self.n_neurons].fill(0.0);
    }

    /// Add `weight` to neuron `post` in the slot for step `slot`.
    #[inline(always)]
    pub fn add(&mut self, slot: usize, post: usize, weight: f32) {
        self.slots[slot % MAX_DELAY_SLOTS].as_mut_slice()[post] += weight;
    }

    /// Read-only access to one slot (for tests / diagnostics).
    #[inline(always)]
    pub fn slot(&self, slot: usize) -> &[f32] {
        &self.slots[slot % MAX_DELAY_SLOTS].as_slice()[..self.n_neurons]
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_read_clear_add() {
        let mut buf = DelayBuffer::new(4);
        assert_eq!(buf.num_slots(), MAX_DELAY_SLOTS);
        assert_eq!(buf.len(), 4);

        // Write into slot 1, then read it back.
        buf.add(1, 2, 5.0);
        let mut out = [0.0f32; 4];
        buf.read_slot(1, &mut out);
        assert_eq!(out, [0.0, 0.0, 5.0, 0.0]);

        // Clear frees the slot.
        buf.clear_slot(1);
        buf.read_slot(1, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_delay_wraparound() {
        let mut buf = DelayBuffer::new(2);
        // slot index beyond MAX_DELAY_SLOTS wraps modulo.
        buf.add(MAX_DELAY_SLOTS + 3, 0, 7.0);
        assert_eq!(buf.slot(3), &[7.0, 0.0]);
        assert_eq!(buf.slot(MAX_DELAY_SLOTS + 3), &[7.0, 0.0]);
    }

    #[test]
    fn test_delay_accumulate() {
        let mut buf = DelayBuffer::new(2);
        buf.add(2, 1, 3.0);
        buf.add(2, 1, 4.0);
        assert_eq!(buf.slot(2), &[0.0, 7.0]);
    }
}
