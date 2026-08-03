//! # Gap — Gap Junction (Electrical Synapse)
//!
//! ```text
//! I_gap = g_gap · (V_pre - V_post)
//! ```
//!
//! Unlike chemical synapses, gap junctions directly couple membrane
//! potentials via connexin pores. Current flows bidirectionally.
//!
//! In Drosophila: ~25% of lamina synapses are electrical.
//! g_gap ≈ 0.05-0.5 nS.

use crate::core::lane::SimdLane;

/// Gap junction current: positive when pre is depolarized relative to post.
#[inline(always)]
pub fn gap_current<S: SimdLane>(v_pre: S, v_post: S, g_gap: S) -> S {
    g_gap * (v_pre - v_post)
}

/// Accumulate gap junction current into a neuron from all neighbors.
/// `v_self` is this neuron's V, `v_neighbors` is summed V of neighbors.
#[inline(always)]
pub fn gap_network_current<S: SimdLane>(v_self: S, sum_v_neighbors: S, g_gap: S, n_connections: S) -> S {
    g_gap * (sum_v_neighbors - n_connections * v_self)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_gap_basic() {
        let pre = f32x8::splat(-60.0);
        let post = f32x8::splat(-65.0);
        let g = f32x8::splat(0.1);
        let i = gap_current(pre, post, g);
        // I = 0.1 * (-60 - (-65)) = 0.5 (positive → depolarizing for post)
        assert!((i.to_array()[0] - 0.5).abs() < 1e-6);
    }
}
