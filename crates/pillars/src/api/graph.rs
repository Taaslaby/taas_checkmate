//! # SparseGraph — CSR Adjacency for Dendritic Spike Routing
//!
//! A minimal **Compressed Sparse Row** graph built specifically for
//! event-driven spike propagation. Contains weight + axonal delay per synapse.
//!
//! ## Example
//!
//! ```rust,ignore
//! use pillars::api::SparseGraph;
//!
//! let mut g = SparseGraph::from_triplets(
//!     4,                          // 4 neurons
//!     &[0, 1, 2, 0, 1],          // pre indices
//!     &[1, 2, 3, 2, 3],          // post indices
//!     &[1.0, 0.5, 2.0, 0.1, 3.0],// weights
//!     &[0, 1, 1, 2, 0],          // delays (steps)
//! );
//! assert_eq!(g.num_synapses(), 5);
//! assert_eq!(g.row(0).len(), 2); // neuron 0 → 1, 2
//! ```

use crate::core::buf::AlignedVec;

/// Maximum supported axonal delay in simulation steps.
/// 32 steps at dt=0.1ms → 3.2ms, covers most monosynaptic delays.
pub const MAX_DELAY_SLOTS: usize = 32;

/// A row in the sparse graph — the outgoing synapses of one neuron.
#[derive(Debug, Clone, Copy)]
pub struct SynapseRow<'a> {
    /// Target neuron global indices
    pub col: &'a [u32],
    /// Synaptic weights
    pub weights: &'a [f32],
    /// Axonal delays in steps (0..MAX_DELAY_SLOTS)
    pub delays: &'a [u8],
}

impl<'a> SynapseRow<'a> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.col.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }
}

/// Sparse directed graph in Compressed Sparse Row (CSR) format.
///
/// Each neuron has outgoing synapses stored as contiguous ranges in `col_idx`.
#[derive(Debug, Clone)]
pub struct SparseGraph {
    /// CSR row-pointer: start indices of each neuron's outgoing synapses
    pub row_ptr: Vec<u32>,
    /// Target neuron global index per synapse
    pub col_idx: Vec<u32>,
    /// Synaptic weight per synapse
    pub weights: AlignedVec,
    /// Axonal delay (steps, 0..MAX_DELAY_SLOTS) per synapse
    pub delays: Vec<u8>,
    n_neurons: usize,
}

impl SparseGraph {
    /// Build a CSR graph from triplets (sorted by `from` automatically).
    ///
    /// # Panics
    /// - If any `from` / `to` >= `n_neurons`
    /// - If `delays` contains values >= `MAX_DELAY_SLOTS`
    /// - If input slices differ in length
    pub fn from_triplets(
        n_neurons: usize,
        from: &[u32],
        to: &[u32],
        weights: &[f32],
        delays: &[u8],
    ) -> Self {
        assert_eq!(from.len(), to.len(), "from/to length mismatch");
        assert_eq!(from.len(), weights.len(), "from/weights length mismatch");
        assert_eq!(from.len(), delays.len(), "from/delays length mismatch");
        let n_syn = from.len();

        // Validate bounds
        if let Some(&f) = from.iter().max() {
            assert!((f as usize) < n_neurons, "from index {f} >= {n_neurons}");
        }
        if let Some(&t) = to.iter().max() {
            assert!((t as usize) < n_neurons, "to index {t} >= {n_neurons}");
        }
        for &d in delays.iter() {
            assert!(
                (d as usize) < MAX_DELAY_SLOTS,
                "delay {d} >= MAX_DELAY_SLOTS ({MAX_DELAY_SLOTS})"
            );
        }

        // Build (from, to, weight, delay) and sort by from
        let mut triplets: Vec<(u32, u32, f32, u8)> = Vec::with_capacity(n_syn);
        for i in 0..n_syn {
            triplets.push((from[i], to[i], weights[i], delays[i]));
        }
        triplets.sort_unstable_by_key(|t| t.0);

        // Count outgoing synapses per neuron
        let mut counts = vec![0u32; n_neurons];
        for &(f, _, _, _) in &triplets {
            counts[f as usize] += 1;
        }

        // Build CSR row_ptr: row_ptr[i] = start index of neuron i's edges
        let mut row_ptr = vec![0u32; n_neurons + 1];
        let mut cum = 0u32;
        for i in 0..n_neurons {
            row_ptr[i] = cum;
            cum += counts[i];
        }
        row_ptr[n_neurons] = cum;
        let total = cum as usize;

        let mut col_idx = vec![0u32; total];
        let mut weights_vec = vec![0.0f32; total];
        let mut delays_vec = vec![0u8; total];

        // Second pass: place each edge using cursor = row_ptr copy
        let mut cursor = row_ptr.clone(); // cursor[i] = next write pos for neuron i
        for (f, t, w, d) in triplets {
            let pos = cursor[f as usize] as usize;
            col_idx[pos] = t;
            weights_vec[pos] = w;
            delays_vec[pos] = d;
            cursor[f as usize] += 1;
        }

        Self {
            row_ptr,
            col_idx,
            weights: AlignedVec::from_slice(&weights_vec),
            delays: delays_vec,
            n_neurons,
        }
    }

    /// Build a CSR graph directly from precomputed CSR arrays (e.g. loaded
    /// from a `.taas` snapshot). `row_ptr` must be length `n_neurons + 1`
    /// and end at `col_idx.len()`. `weights` / `delays` must match `col_idx`.
    ///
    /// # Panics
    /// - If lengths mismatch
    /// - If any `col_idx` index >= `n_neurons`
    /// - If `delays` contains values >= `MAX_DELAY_SLOTS`
    pub fn from_csr(row_ptr: Vec<u32>, col_idx: Vec<u32>, weights: &[f32], delays: &[u8]) -> Self {
        assert!(!row_ptr.is_empty(), "row_ptr must have at least one entry");
        let n_neurons = row_ptr.len() - 1;
        assert_eq!(
            row_ptr[n_neurons] as usize,
            col_idx.len(),
            "CSR row_ptr must end at col_idx.len()"
        );
        assert_eq!(col_idx.len(), weights.len(), "col_idx/weights length mismatch");
        assert_eq!(col_idx.len(), delays.len(), "col_idx/delays length mismatch");
        for &t in &col_idx {
            assert!((t as usize) < n_neurons, "col index {t} >= {n_neurons}");
        }
        for &d in delays.iter() {
            assert!(
                (d as usize) < MAX_DELAY_SLOTS,
                "delay {d} >= MAX_DELAY_SLOTS ({MAX_DELAY_SLOTS})"
            );
        }
        Self {
            row_ptr,
            col_idx,
            weights: AlignedVec::from_slice(weights),
            delays: delays.to_vec(),
            n_neurons,
        }
    }

    /// Number of neurons in the graph.
    #[inline(always)]
    pub fn num_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Number of synapses.
    #[inline(always)]
    pub fn num_synapses(&self) -> usize {
        self.col_idx.len()
    }

    /// Get outgoing synapses of a neuron by its global index.
    #[inline(always)]
    pub fn row(&self, i: usize) -> SynapseRow<'_> {
        let start = self.row_ptr[i] as usize;
        let end = self.row_ptr[i + 1] as usize;
        SynapseRow {
            col: &self.col_idx[start..end],
            weights: &self.weights.as_slice()[start..end],
            delays: &self.delays[start..end],
        }
    }

    /// Check if the neuron has any outgoing synapses.
    #[inline(always)]
    pub fn has_outgoing(&self, i: usize) -> bool {
        self.row_ptr[i] < self.row_ptr[i + 1]
    }

    /// Outgoing synapse count (degree) of the neuron.
    #[inline(always)]
    pub fn degree(&self, i: usize) -> usize {
        (self.row_ptr[i + 1] - self.row_ptr[i]) as usize
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_simple() {
        // 4 neurons, 5 synapses
        let g = SparseGraph::from_triplets(
            4,
            &[0, 1, 2, 0, 1],
            &[1, 2, 3, 2, 3],
            &[1.0, 0.5, 2.0, 0.1, 3.0],
            &[0, 1, 1, 2, 0],
        );
        assert_eq!(g.num_neurons(), 4);
        assert_eq!(g.num_synapses(), 5);

        // Neuron 0 → 1, 2
        let r0 = g.row(0);
        assert_eq!(r0.len(), 2);
        assert_eq!(r0.col, &[1, 2]);
        assert!((r0.weights[0] - 1.0).abs() < 1e-6);
        assert!((r0.weights[1] - 0.1).abs() < 1e-6);
        assert_eq!(r0.delays, &[0, 2]);

        // Neuron 1 → 2, 3
        let r1 = g.row(1);
        assert_eq!(r1.len(), 2);
        assert_eq!(r1.col, &[2, 3]);
        assert!((r1.weights[0] - 0.5).abs() < 1e-6);
        assert_eq!(r1.delays, &[1, 0]);

        // Neuron 2 → 3
        let r2 = g.row(2);
        assert_eq!(r2.len(), 1);
        assert_eq!(r2.col[0], 3);
        assert!((r2.weights[0] - 2.0).abs() < 1e-6);

        // Neuron 3 — no outgoing
        assert!(g.row(3).is_empty());
        assert!(!g.has_outgoing(3));
    }

    #[test]
    fn test_graph_empty() {
        let g = SparseGraph::from_triplets(5, &[], &[], &[], &[]);
        assert_eq!(g.num_synapses(), 0);
        assert_eq!(g.num_neurons(), 5);
        for i in 0..5 {
            assert!(g.row(i).is_empty());
        }
    }

    #[test]
    fn test_graph_single() {
        let g = SparseGraph::from_triplets(2, &[0], &[1], &[2.5], &[0]);
        assert_eq!(g.num_synapses(), 1);
        let r0 = g.row(0);
        assert_eq!(r0.len(), 1);
        assert_eq!(r0.col[0], 1);
        assert!((r0.weights[0] - 2.5).abs() < 1e-6);
        assert_eq!(r0.delays[0], 0);
    }

    #[test]
    fn test_graph_degree() {
        let g = SparseGraph::from_triplets(
            3,
            &[0, 0, 0, 1],
            &[1, 2, 0, 2],
            &[1.0, 1.0, 1.0, 1.0],
            &[0, 0, 0, 0],
        );
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.degree(1), 1);
        assert_eq!(g.degree(2), 0);
    }

    #[test]
    #[should_panic(expected = "from index")]
    fn test_graph_oob_from() {
        SparseGraph::from_triplets(3, &[5], &[0], &[1.0], &[0]);
    }

    #[test]
    #[should_panic(expected = "to index")]
    fn test_graph_oob_to() {
        SparseGraph::from_triplets(3, &[0], &[5], &[1.0], &[0]);
    }

    #[test]
    #[should_panic(expected = "delay")]
    fn test_graph_delay_oob() {
        SparseGraph::from_triplets(3, &[0], &[1], &[1.0], &[99]);
    }
}
