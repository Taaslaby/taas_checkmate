//! # Spike Push — Event-Driven Synaptic Current Routing
//!
//! For every **active** neuron (1–5% of population in typical SNN),
//! scatters its weight into future ring buffer slots based on axonal delay.
//!
//! ## Principle: Push, not Pull
//!
//! | Approach | Cost | When |
//! |:---------|:-----|:-----|
//! | ❌ Pull | O(N × avg_degree) | Every step, even silent neurons |
//! | ✅ Push | O(N_active × avg_degree) | Scales with sparsity |
//!
//! With 1–5% activity and avg_degree ~100, push is 20–100× cheaper.

use crate::api::graph::{SparseGraph, MAX_DELAY_SLOTS};
use crate::core::buf::AlignedVec;

/// Push active spike indices through the graph into ring buffer slots.
///
/// Each active neuron scatters its weight into
/// `ring_buf[(step + delay.max(1)) % 32]` for each outgoing synapse — the
/// current arrives at the post-synaptic neuron `max(delay, 1)` simulation
/// steps later (delay 0 = delivered on the very next step, since a spike
/// cannot reach the post neuron in the same step it fires).
///
/// # Arguments
/// * `active` — indices of neurons that spiked this step
/// * `graph` — CSR adjacency (outgoing synapses per neuron)
/// * `ring_buf` — ring buffer of currents, `[MAX_DELAY_SLOTS][N]`
/// * `step_counter` — current simulation step (for delay computation)
#[inline]
pub fn push_spikes(
    active: &[u32],
    graph: &SparseGraph,
    ring_buf: &mut [AlignedVec],
    step_counter: usize,
) {
    for &pre_idx in active {
        let pre = pre_idx as usize;
        if !graph.has_outgoing(pre) {
            continue;
        }
        let row = graph.row(pre);
        for syn_i in 0..row.len() {
            let post = row.col[syn_i] as usize;
            let weight = row.weights[syn_i];
            let delay = row.delays[syn_i] as usize;

            // delay 0 = next step: a spike fired at step `s` is delivered at
            // step `s + max(delay, 1)`. It cannot land in slot `s % 32` — that
            // slot was already read and cleared at the start of step `s`
            // (read-before-write ring-buffer ordering).
            let target_slot = (step_counter + delay.max(1)) % MAX_DELAY_SLOTS;
            ring_buf[target_slot].as_mut_slice()[post] += weight;
        }
    }
}

/// Collect `(post, slot, weight)` triples for active spikes — race-free
/// thread-local accumulation for parallel dispatch.
///
/// Each thread processes its chunk of active neurons and returns triples;
/// the caller scatters them into the ring buffer on the main thread after
/// join. Delays ARE respected: `slot = (step + delay) % MAX_DELAY_SLOTS`.
#[cfg(feature = "rayon")]
pub(crate) fn collect_push_triples(
    active: &[u32],
    graph: &SparseGraph,
    step_counter: usize,
) -> Vec<(u32, usize, f32)> {
    let mut triples = Vec::with_capacity(active.len() * 4);
    for &pre_idx in active {
        let pre = pre_idx as usize;
        if !graph.has_outgoing(pre) {
            continue;
        }
        let row = graph.row(pre);
        for syn_i in 0..row.len() {
            let post = row.col[syn_i];
            let w = row.weights[syn_i];
            let delay = row.delays[syn_i] as usize;
            // Same minimum-1-step rule as push_spikes (delay 0 = next step).
            let slot = (step_counter + delay.max(1)) % MAX_DELAY_SLOTS;
            triples.push((post, slot, w));
        }
    }
    triples
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::SparseGraph;

    /// Build a simple graph: N neurons, one-to-one connections
    fn make_chain_graph(n: usize) -> SparseGraph {
        let from: Vec<u32> = (0..n as u32).collect();
        let to: Vec<u32> = from.clone();
        let weights = vec![1.0f32; n];
        let delays = vec![0u8; n];
        SparseGraph::from_triplets(n, &from, &to, &weights, &delays)
    }

    #[test]
    fn test_push_spikes_empty() {
        let graph = make_chain_graph(8);
        let mut ring = vec![AlignedVec::new(8, 0.0); MAX_DELAY_SLOTS];
        push_spikes(&[], &graph, &mut ring, 0);
        for slot in &ring {
            for i in 0..8 {
                assert_eq!(slot.as_slice()[i], 0.0);
            }
        }
    }

    #[test]
    fn test_push_spikes_no_outgoing() {
        // Neuron 2 has no outgoing edges: graph has 3 neurons but only 0→0 and 1→1
        let from = vec![0u32, 1u32];
        let to = vec![0u32, 1u32];
        let weights = vec![1.0f32, 1.0f32];
        let delays = vec![0u8, 0u8];
        let graph = SparseGraph::from_triplets(3, &from, &to, &weights, &delays);
        let mut ring = vec![AlignedVec::new(3, 0.0); MAX_DELAY_SLOTS];
        push_spikes(&[2], &graph, &mut ring, 0);
        // Ring should remain zero since neuron 2 has no outgoing edges
        for slot in &ring {
            assert_eq!(slot.as_slice()[2], 0.0);
        }
    }

    #[test]
    fn test_push_spikes_basic() {
        let graph = make_chain_graph(8);
        let mut ring = vec![AlignedVec::new(8, 0.0); MAX_DELAY_SLOTS];
        push_spikes(&[0, 3, 7], &graph, &mut ring, 0);
        // Delay 0 → delivered at the NEXT step (slot 1).
        let slot1 = &ring[1];
        assert_eq!(slot1.as_slice()[0], 1.0);
        assert_eq!(slot1.as_slice()[3], 1.0);
        assert_eq!(slot1.as_slice()[7], 1.0);
        // Other slots should be zero
        for (i, s) in ring.iter().enumerate().take(MAX_DELAY_SLOTS) {
            if i == 1 {
                continue;
            }
            for j in 0..8 {
                assert_eq!(s.as_slice()[j], 0.0, "slot {i}[{j}] should be 0");
            }
        }
    }

    #[test]
    fn test_push_spikes_with_delays() {
        // Build graph with delays
        let from = vec![0u32, 1u32, 2u32];
        let to = vec![3u32, 4u32, 5u32];
        let w = vec![1.0f32, 2.0f32, 0.5f32];
        let d = vec![0u8, 1u8, 5u8];
        let graph = SparseGraph::from_triplets(6, &from, &to, &w, &d);
        let mut ring = vec![AlignedVec::new(6, 0.0); MAX_DELAY_SLOTS];

        // Push at step 0
        push_spikes(&[0, 1, 2], &graph, &mut ring, 0);

        // Neuron 0 → 3 with delay 0: delivered next step → slot (0+1)%32 = 1
        assert_eq!(ring[1].as_slice()[3], 1.0);
        // Neuron 1 → 4 with delay 1: also next step → slot 1
        assert_eq!(ring[1].as_slice()[4], 2.0);
        // Neuron 2 → 5 with delay 5: in slot (0+5)%32 = 5
        assert_eq!(ring[5].as_slice()[5], 0.5);
        // Slot 0 stays empty — zero-delay current goes to slot 1, not 0
        assert_eq!(ring[0].as_slice()[3], 0.0);
        assert_eq!(ring[0].as_slice()[4], 0.0);
    }

    #[test]
    fn test_push_spikes_delay_wraparound() {
        let from = vec![0u32];
        let to = vec![1u32];
        let w = vec![1.0f32];
        let d = vec![(MAX_DELAY_SLOTS - 1) as u8]; // max valid delay = 31
        let graph = SparseGraph::from_triplets(2, &from, &to, &w, &d);
        let mut ring = vec![AlignedVec::new(2, 0.0); MAX_DELAY_SLOTS];

        push_spikes(&[0], &graph, &mut ring, 1); // step=1, target = (1+31)%32 = 0
        assert_eq!(ring[0].as_slice()[1], 1.0);
    }

    #[test]
    fn test_push_spikes_step_counter() {
        let from = vec![0u32];
        let to = vec![1u32];
        let w = vec![1.0f32];
        let d = vec![2u8];
        let graph = SparseGraph::from_triplets(2, &from, &to, &w, &d);
        let mut ring = vec![AlignedVec::new(2, 0.0); MAX_DELAY_SLOTS];

        // Push at step 5: target = (5+2)%32 = 7
        push_spikes(&[0], &graph, &mut ring, 5);
        assert_eq!(ring[7].as_slice()[1], 1.0);
        // Slot 0 should be zero
        assert_eq!(ring[0].as_slice()[1], 0.0);
    }

    #[test]
    fn test_push_spikes_accumulate() {
        let from = vec![0u32, 1u32];
        let to = vec![2u32, 2u32]; // both converge to neuron 2
        let w = vec![3.0f32, 4.0f32];
        let d = vec![0u8, 0u8];
        let graph = SparseGraph::from_triplets(3, &from, &to, &w, &d);
        let mut ring = vec![AlignedVec::new(3, 0.0); MAX_DELAY_SLOTS];

        push_spikes(&[0, 1], &graph, &mut ring, 0);
        // Both weights accumulate in slot 1 (delay 0 = next step): 3.0 + 4.0 = 7.0
        assert!((ring[1].as_slice()[2] - 7.0).abs() < 1e-6);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_collect_push_triples_respects_delays() {
        // Regression: the parallel push path previously dropped axonal delays.
        // Triples must carry (post, slot = (step + delay) % 32, weight).
        let from = vec![0u32, 1u32];
        let to = vec![2u32, 3u32];
        let w = vec![1.0f32, 2.0f32];
        let d = vec![0u8, 3u8];
        let graph = SparseGraph::from_triplets(4, &from, &to, &w, &d);

        // step 7: delay 0 → next step slot 8; delay 3 → slot 10
        let triples = collect_push_triples(&[0, 1], &graph, 7);
        assert_eq!(triples.len(), 2);
        assert!(triples.contains(&(2, 8, 1.0)), "delay 0 → slot 8: {triples:?}");
        assert!(triples.contains(&(3, 10, 2.0)), "delay 3 → slot 10: {triples:?}");

        // Same result as the sequential push_spikes: scatter triples → compare.
        let mut ring_par = vec![AlignedVec::new(4, 0.0); MAX_DELAY_SLOTS];
        for (post, slot, weight) in triples {
            ring_par[slot].as_mut_slice()[post as usize] += weight;
        }
        let mut ring_seq = vec![AlignedVec::new(4, 0.0); MAX_DELAY_SLOTS];
        push_spikes(&[0, 1], &graph, &mut ring_seq, 7);
        for slot in 0..MAX_DELAY_SLOTS {
            for i in 0..4 {
                assert_eq!(
                    ring_par[slot].as_slice()[i],
                    ring_seq[slot].as_slice()[i],
                    "slot {slot}[{i}] mismatch"
                );
            }
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_collect_push_triples_skip_no_outgoing() {
        // Neuron 3 has no outgoing edges — must produce no triples.
        let from = vec![0u32, 1u32];
        let to = vec![2u32, 2u32];
        let w = vec![0.5f32, 0.5f32];
        let d = vec![0u8, 0u8];
        let graph = SparseGraph::from_triplets(4, &from, &to, &w, &d);
        let triples = collect_push_triples(&[0, 3], &graph, 0);
        assert_eq!(triples.len(), 1, "only neuron 0 fires: {triples:?}");
        assert_eq!(triples[0], (2, 1, 0.5)); // delay 0 → next step (slot 1)
    }
}
