//! -------------------------------------------------------------------------
//! Filename:    src/delay.rs
//! Class:       buffer
//! Purpose:     DelayBuffer — Ring Buffer of Synaptic Currents
//! Description: Owns MAX_DELAY_SLOTS aligned current vectors, one per
//!              future simulation step. On spike, weights are scattered
//!              into slot (step + delay) % MAX_DELAY_SLOTS; each step
//!              reads and clears the current slot.
//!
//! Ring-buffer ordering (read-before-write):
//! ```text
//! step s:  read  slot[s % 32]  → i_in
//!          clear slot[s % 32]
//!          step neurons
//!          write spikes into slot[(s + delay) % 32]  (delay ≥ 1)
//! ```
//! A spike fired at step s is delivered at step s + max(delay, 1) —
//! it can never land in the slot that was already read and cleared.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

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
    ///
    /// # Panics
    /// If `post >= len()`.
    #[inline(always)]
    pub fn add(&mut self, slot: usize, post: usize, weight: f32) {
        // Явная проверка вместо голой слайс-паники: logical_len буфера
        // равен n_neurons, но сообщение «index out of bounds» не говорит,
        // что именно post-индекс сети невалиден.
        assert!(
            post < self.n_neurons,
            "DelayBuffer::add: post index {post} >= n_neurons {}",
            self.n_neurons
        );
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

    /// post за пределами сети — явная ошибка с понятным сообщением
    /// вместо голой slice-паники «index out of bounds».
    #[test]
    #[should_panic(expected = "post index")]
    fn test_delay_post_out_of_bounds() {
        let mut buf = DelayBuffer::new(4);
        buf.add(0, 5, 1.0);
    }

    /// Контракт `effective_delay = delay.max(1)`: спайк с delay=0 должен
    /// быть доставлен на *следующем* шаге (слот (step+1) % MAX), а не в
    /// слот, который уже прочитан и очищен на текущем шаге. push.rs
    /// применяет max(1) на обоих сайтах записи — этот тест фиксирует
    /// кольцевую часть контракта.
    #[test]
    fn test_delay_zero_delivers_next_step() {
        let mut buf = DelayBuffer::new(2);
        // Эмулируем сайт записи из push.rs: slot = (step + delay.max(1)) % MAX
        let step = 0usize;
        let delay = 0u8;
        buf.add((step + delay.max(1) as usize) % MAX_DELAY_SLOTS, 0, 1.0);

        let mut out = [0.0f32; 2];
        buf.read_slot(0, &mut out);
        assert_eq!(out, [0.0, 0.0], "слот текущего шага должен остаться пустым");
        buf.read_slot(1, &mut out);
        assert_eq!(out, [1.0, 0.0], "спайк ложится в слот следующего шага");
    }
}
