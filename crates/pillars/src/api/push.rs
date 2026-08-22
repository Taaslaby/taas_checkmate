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

use crate::api::delay::DelayBuffer;
use crate::api::graph::{SparseGraph, MAX_DELAY_SLOTS};

/// Precomputed per-profile Tsodyks–Markram constants (from
/// [`crate::api::PlasticityProfile`] + the simulation `dt`).
#[derive(Clone, Copy)]
pub(crate) struct PlastRtProfile {
    pub u0: f32,
    /// `e^(−dt/tau_rec)` — per-step recovery factor.
    pub a_rec: f32,
    /// `e^(−dt/tau_fac)`; unused (`0.0`) for depression-only.
    pub a_fac: f32,
    pub is_fac: bool,
}

/// Push-time view of per-edge short-term plasticity state.
pub(crate) struct PlasticityState<'a> {
    pub profiles: &'a [PlastRtProfile],
    /// Per-edge profile id (`0` = none), sized `n_edges` (CSR edge order).
    pub plast_types: &'a [u8],
    /// Per-edge recovery variable `R` (1 = fully recovered).
    pub r: &'a mut [f32],
    /// Per-edge release probability `U`.
    pub u: &'a mut [f32],
    /// Per-edge last-spike step (for the lazy exponential recovery advance).
    pub last: &'a mut [u32],
}

/// Apply one spike's Tsodyks–Markram update to a single synapse's state and
/// return the scaled weight (`weight · R·U`).
///
/// The scale is the **pre-spike** factor `R·U`, exactly per NeuroML2 LEMS
/// ordering: `plasticityFactor = R·U` is a derived variable, so the parent
/// `blockingPlasticSynapse`'s OnEvent (`A += weight · plasticityFactor · …`)
/// evaluates it **before** the relayed child event depletes R (`R = R·(1 − U)`)
/// — the current spike therefore sees the resource it is about to consume.
/// Recovery is lazy: the ODEs relax exponentially between spikes, so a single
/// `powi(n)` over the elapsed steps is exact.
pub(crate) fn tm_spike(
    p: &PlastRtProfile,
    r: &mut f32,
    u: &mut f32,
    last: &mut u32,
    weight: f32,
    step: usize,
) -> f32 {
    let n = (step.saturating_sub(*last as usize)).min(1_000_000) as i32;
    if n > 0 {
        let ar = p.a_rec.powi(n);
        *r = 1.0 + (*r - 1.0) * ar;
        if p.is_fac {
            let af = p.a_fac.powi(n);
            *u = p.u0 + (*u - p.u0) * af;
        }
    }
    *last = step as u32;
    // Spike update (NeuroML OnEvent ordering): deliver with the pre-spike R·U,
    // then deplete the resource / facilitate the release probability.
    let scale = *r * *u;
    *r *= 1.0 - *u;
    if p.is_fac {
        *u += p.u0 * (1.0 - *u);
    }
    weight * scale
}

impl PlasticityState<'_> {
    /// Deliver the plasticity-scaled weight for edge `e` firing at `step`,
    /// advancing its TM state. Returns `weight` unchanged for non-plastic
    /// edges.
    pub fn scale(&mut self, e: usize, weight: f32, step: usize) -> f32 {
        let pt = self.plast_types[e] as usize;
        if pt == 0 {
            return weight;
        }
        let p = &self.profiles[pt - 1];
        tm_spike(
            p,
            &mut self.r[e],
            &mut self.u[e],
            &mut self.last[e],
            weight,
            step,
        )
    }
}

/// Push active spike indices through the graph into delay buffer slots.
///
/// Each active neuron scatters its weight into
/// `delay[(step + delay.max(1)) % 32]` for each outgoing synapse — the
/// current arrives at the post-synaptic neuron `max(delay, 1)` simulation
/// steps later (delay 0 = delivered on the very next step, since a spike
/// cannot reach the post neuron in the same step it fires).
///
/// # Arguments
/// * `active` — indices of neurons that spiked this step
/// * `graph` — CSR adjacency (outgoing synapses per neuron)
/// * `delay` — ring buffer of currents, `[MAX_DELAY_SLOTS][N]` (static synapses)
/// * `kinetic_delay` — one ring buffer per kinetic profile (synapse type `1..`)
/// * `step_counter` — current simulation step (for delay computation)
/// * `plast` — per-edge short-term plasticity state (`None` = no plasticity);
///   the delivered weight is scaled by the synapse's current `R·U`.
#[inline]
pub(crate) fn push_spikes(
    active: &[u32],
    graph: &SparseGraph,
    delay: &mut DelayBuffer,
    kinetic_delay: &mut [DelayBuffer],
    step_counter: usize,
    mut plast: Option<&mut PlasticityState<'_>>,
) {
    // Предвалидация до любого scatter: иначе меньший буфер падал бы на первом
    // вне-диапазонном `post` уже после частичного изменения состояния.
    assert_eq!(
        delay.len(),
        graph.num_neurons(),
        "push_spikes: DelayBuffer length {} does not match graph neuron count {}",
        delay.len(),
        graph.num_neurons()
    );
    for &pre_idx in active {
        let pre = pre_idx as usize;
        if !graph.has_outgoing(pre) {
            continue;
        }
        let row = graph.row(pre);
        for syn_i in 0..row.len() {
            let post = row.col[syn_i] as usize;
            let mut weight = row.weights[syn_i];
            let d = row.delays[syn_i] as usize;
            let syn_type = row.syn_types[syn_i] as usize;

            // Short-term plasticity: scale the delivered weight by the edge's
            // current R·U and update its state (lazily recovered to now).
            if let Some(p) = plast.as_deref_mut() {
                weight = p.scale(row.start + syn_i, weight, step_counter);
            }

            // delay 0 = next step: a spike fired at step `s` is delivered at
            // step `s + max(delay, 1)`. It cannot land in slot `s % 32` — that
            // slot was already read and cleared at the start of step `s`
            // (read-before-write ring-buffer ordering).
            let target_slot = (step_counter + d.max(1)) % MAX_DELAY_SLOTS;
            if syn_type == 0 {
                delay.add(target_slot, post, weight);
            } else {
                kinetic_delay[syn_type - 1].add(target_slot, post, weight);
            }
        }
    }
}

/// Parallel push — thread-local accumulation with a race-free scatter.
///
/// Each worker thread collects `(post, slot, weight)` triples for its chunk
/// of active neurons; the caller scatters them into the delay buffer after
/// join. Delays ARE respected: `slot = (step + max(delay, 1)) % MAX_DELAY_SLOTS` (delay 0 = next step).
///
/// Falls back to the sequential [`push_spikes`] when there is no parallelism
/// to exploit (`active` too small or single-threaded machine).
#[cfg(feature = "rayon")]
pub(crate) fn push_spikes_par(
    active: &[u32],
    graph: &SparseGraph,
    delay: &mut DelayBuffer,
    kinetic_delay: &mut [DelayBuffer],
    step_counter: usize,
    plast: Option<&mut PlasticityState<'_>>,
) {
    assert_eq!(
        delay.len(),
        graph.num_neurons(),
        "push_spikes_par: DelayBuffer length {} does not match graph neuron count {}",
        delay.len(),
        graph.num_neurons()
    );
    // Plasticity mutates per-edge state, which is inherently sequential —
    // fall back to the sequential push for correctness (the common case with
    // no plasticity keeps the parallel path).
    if plast.is_some() {
        push_spikes(active, graph, delay, kinetic_delay, step_counter, plast);
        return;
    }
    let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    if num_threads <= 1 || active.len() <= 64 {
        push_spikes(active, graph, delay, kinetic_delay, step_counter, None);
        return;
    }

    let active_len = active.len();
    let chunk_size = active_len.div_ceil(num_threads).max(8);

    // Use rayon::scope to reuse thread pool instead of spawning OS threads.
    // SparseGraph is Sync, so &SparseGraph is Send across the scope.
    use rayon::prelude::*;
    let triples: Vec<_> = (0..num_threads)
        .into_par_iter()
        .filter(|tid| tid * chunk_size < active_len)
        .flat_map(|tid| {
            let chunk_start = tid * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(active_len);
            let chunk: Vec<u32> = active[chunk_start..chunk_end].to_vec();
            collect_push_triples(&chunk, graph, step_counter)
        })
        .collect();

    // Scatter: each triple is (post, slot, weight, syn_type).
    for (post, slot, w, syn_type) in triples {
        if syn_type == 0 {
            delay.add(slot, post as usize, w);
        } else {
            kinetic_delay[(syn_type as usize) - 1].add(slot, post as usize, w);
        }
    }
}

/// Collect `(post, slot, weight, syn_type)` quadruples for active spikes —
/// race-free thread-local accumulation for parallel dispatch.
///
/// Each thread processes its chunk of active neurons and returns quadruples;
/// the caller scatters them into the ring buffer on the main thread after
/// join. Delays ARE respected: `slot = (step + max(delay, 1)) % MAX_DELAY_SLOTS` (delay 0 = next step).
#[cfg(feature = "rayon")]
pub(crate) fn collect_push_triples(
    active: &[u32],
    graph: &SparseGraph,
    step_counter: usize,
) -> Vec<(u32, usize, f32, u8)> {
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
            let d = row.delays[syn_i] as usize;
            let syn_type = row.syn_types[syn_i];
            // Same minimum-1-step rule as push_spikes (delay 0 = next step).
            let slot = (step_counter + d.max(1)) % MAX_DELAY_SLOTS;
            triples.push((post, slot, w, syn_type));
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

    /// Меньший DelayBuffer против графа должен отвергаться до любого scatter
    /// (атомарность), а не падать на первом вне-диапазонном `post` после
    /// частичного изменения буфера.
    #[test]
    #[should_panic(expected = "does not match graph neuron count")]
    fn test_push_spikes_rejects_mismatched_delay_buffer() {
        let graph = make_chain_graph(8);
        let mut delay = DelayBuffer::new(4); // 4 != 8
        push_spikes(&[0, 3], &graph, &mut delay, &mut [], 0, None);
    }

    #[test]
    fn test_push_spikes_empty() {
        let graph = make_chain_graph(8);
        let mut delay = DelayBuffer::new(8);
        push_spikes(&[], &graph, &mut delay, &mut [], 0, None);
        for slot in 0..MAX_DELAY_SLOTS {
            for i in 0..8 {
                assert_eq!(delay.slot(slot)[i], 0.0);
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
        let mut delay = DelayBuffer::new(3);
        push_spikes(&[2], &graph, &mut delay, &mut [], 0, None);
        // Delay buffer should remain zero since neuron 2 has no outgoing edges
        for slot in 0..MAX_DELAY_SLOTS {
            assert_eq!(delay.slot(slot)[2], 0.0);
        }
    }

    #[test]
    fn test_push_spikes_basic() {
        let graph = make_chain_graph(8);
        let mut delay = DelayBuffer::new(8);
        push_spikes(&[0, 3, 7], &graph, &mut delay, &mut [], 0, None);
        // Delay 0 → delivered at the NEXT step (slot 1).
        assert_eq!(delay.slot(1)[0], 1.0);
        assert_eq!(delay.slot(1)[3], 1.0);
        assert_eq!(delay.slot(1)[7], 1.0);
        // Other slots should be zero
        for slot in 0..MAX_DELAY_SLOTS {
            if slot == 1 {
                continue;
            }
            for j in 0..8 {
                assert_eq!(delay.slot(slot)[j], 0.0, "slot {slot}[{j}] should be 0");
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
        let mut delay = DelayBuffer::new(6);

        // Push at step 0
        push_spikes(&[0, 1, 2], &graph, &mut delay, &mut [], 0, None);

        // Neuron 0 → 3 with delay 0: delivered next step → slot (0+1)%32 = 1
        assert_eq!(delay.slot(1)[3], 1.0);
        // Neuron 1 → 4 with delay 1: also next step → slot 1
        assert_eq!(delay.slot(1)[4], 2.0);
        // Neuron 2 → 5 with delay 5: in slot (0+5)%32 = 5
        assert_eq!(delay.slot(5)[5], 0.5);
        // Slot 0 stays empty — zero-delay current goes to slot 1, not 0
        assert_eq!(delay.slot(0)[3], 0.0);
        assert_eq!(delay.slot(0)[4], 0.0);
    }

    #[test]
    fn test_push_spikes_delay_wraparound() {
        let from = vec![0u32];
        let to = vec![1u32];
        let w = vec![1.0f32];
        let d = vec![(MAX_DELAY_SLOTS - 1) as u8]; // max valid delay = 31
        let graph = SparseGraph::from_triplets(2, &from, &to, &w, &d);
        let mut delay = DelayBuffer::new(2);

        push_spikes(&[0], &graph, &mut delay, &mut [], 1, None); // step=1, target = (1+31)%32 = 0
        assert_eq!(delay.slot(0)[1], 1.0);
    }

    #[test]
    fn test_push_spikes_step_counter() {
        let from = vec![0u32];
        let to = vec![1u32];
        let w = vec![1.0f32];
        let d = vec![2u8];
        let graph = SparseGraph::from_triplets(2, &from, &to, &w, &d);
        let mut delay = DelayBuffer::new(2);

        // Push at step 5: target = (5+2)%32 = 7
        push_spikes(&[0], &graph, &mut delay, &mut [], 5, None);
        assert_eq!(delay.slot(7)[1], 1.0);
        // Slot 0 should be zero
        assert_eq!(delay.slot(0)[1], 0.0);
    }

    #[test]
    fn test_push_spikes_accumulate() {
        let from = vec![0u32, 1u32];
        let to = vec![2u32, 2u32]; // both converge to neuron 2
        let w = vec![3.0f32, 4.0f32];
        let d = vec![0u8, 0u8];
        let graph = SparseGraph::from_triplets(3, &from, &to, &w, &d);
        let mut delay = DelayBuffer::new(3);

        push_spikes(&[0, 1], &graph, &mut delay, &mut [], 0, None);
        // Both weights accumulate in slot 1 (delay 0 = next step): 3.0 + 4.0 = 7.0
        assert!((delay.slot(1)[2] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_push_spikes_routes_kinetic() {
        // Synapse type 1 → kinetic profile 0's delay ring; type 0 → static.
        let from = vec![0u32, 1u32];
        let to = vec![2u32, 2u32];
        let w = vec![3.0f32, 4.0f32];
        let d = vec![0u8, 0u8];
        let syn_types = vec![1u8, 0u8];
        let graph = SparseGraph::from_triplets_with_types(3, &from, &to, &w, &d, &syn_types);
        let mut delay = DelayBuffer::new(3);
        let mut kinetic = vec![DelayBuffer::new(3)];
        push_spikes(&[0, 1], &graph, &mut delay, &mut kinetic, 0, None);
        // Neuron 0 → type 1 → kinetic ring slot 1.
        assert_eq!(kinetic[0].slot(1)[2], 3.0);
        // Neuron 1 → type 0 → static ring slot 1.
        assert_eq!(delay.slot(1)[2], 4.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_push_spikes_par_matches_sequential() {
        // The parallel dispatcher must produce byte-identical ring buffers.
        let from = vec![0u32, 1u32, 2u32];
        let to = vec![3u32, 4u32, 5u32];
        let w = vec![1.0f32, 2.0f32, 0.5f32];
        let d = vec![0u8, 3u8, 7u8];
        let graph = SparseGraph::from_triplets(6, &from, &to, &w, &d);

        let mut seq = DelayBuffer::new(6);
        push_spikes(&[0, 1, 2], &graph, &mut seq, &mut [], 7, None);

        let mut par = DelayBuffer::new(6);
        push_spikes_par(&[0, 1, 2], &graph, &mut par, &mut [], 7, None);

        for slot in 0..MAX_DELAY_SLOTS {
            for i in 0..6 {
                assert_eq!(
                    par.slot(slot)[i],
                    seq.slot(slot)[i],
                    "slot {slot}[{i}] mismatch"
                );
            }
        }
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
        assert!(
            triples.contains(&(2, 8, 1.0, 0)),
            "delay 0 → slot 8: {triples:?}"
        );
        assert!(
            triples.contains(&(3, 10, 2.0, 0)),
            "delay 3 → slot 10: {triples:?}"
        );

        // Same result as the sequential push_spikes: scatter triples → compare.
        let mut par = DelayBuffer::new(4);
        for (post, slot, weight, _) in triples {
            par.add(slot, post as usize, weight);
        }
        let mut seq = DelayBuffer::new(4);
        push_spikes(&[0, 1], &graph, &mut seq, &mut [], 7, None);
        for slot in 0..MAX_DELAY_SLOTS {
            for i in 0..4 {
                assert_eq!(
                    par.slot(slot)[i],
                    seq.slot(slot)[i],
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
        assert_eq!(triples[0], (2, 1, 0.5, 0)); // delay 0 → next step (slot 1)
    }

    /// Depression-only TM: with U0 = 0.5 and negligible recovery, each spike
    /// delivers `R·U0` and depletes R by the factor (1 − U0) — delivered
    /// weights decay geometrically 0.5, 0.25, 0.125, …
    #[test]
    fn test_tm_depression_geometric_decay() {
        let p = PlastRtProfile {
            u0: 0.5,
            a_rec: (-0.5e-6f32 / 1e6).exp(), // τ_rec = 1e6 ms, dt = 0.5 ms → ≈ 1.0
            a_fac: 0.0,
            is_fac: false,
        };
        let mut r = 1.0f32;
        let mut u = 0.5f32;
        let mut last = 0u32;
        let mut got = Vec::new();
        for step in [0usize, 10, 20, 30] {
            got.push(tm_spike(&p, &mut r, &mut u, &mut last, 1.0, step));
        }
        for (i, w) in got.iter().enumerate() {
            let expect = 0.5f32.powi(i as i32 + 1);
            assert!(
                (w - expect).abs() < 1e-3,
                "spike {i}: got {w}, expected {expect}"
            );
        }
    }

    /// Recovery: with a short τ_rec the resource R relaxes back towards 1
    /// between spikes, so a later spike delivers more than the pure geometric
    /// decay would allow.
    #[test]
    fn test_tm_depression_recovers_between_spikes() {
        // dt = 1 ms, τ_rec = 1 ms → a_rec = e^(−1) ≈ 0.3679.
        let p = PlastRtProfile {
            u0: 0.5,
            a_rec: (-1.0f32).exp(),
            a_fac: 0.0,
            is_fac: false,
        };
        let mut r = 1.0f32;
        let mut u = 0.5f32;
        let mut last = 0u32;
        // Back-to-back spikes at steps 0,1,2 (n=1 recovery each).
        let mut tight = Vec::new();
        for step in 0usize..4 {
            tight.push(tm_spike(&p, &mut r, &mut u, &mut last, 1.0, step));
        }
        // Same profile, spikes 1s apart (n=1000 → recovery ≈ complete).
        let mut r2 = 1.0f32;
        let mut u2 = 0.5f32;
        let mut last2 = 0u32;
        let mut spaced = Vec::new();
        for step in [0usize, 1000, 2000, 3000] {
            spaced.push(tm_spike(&p, &mut r2, &mut u2, &mut last2, 1.0, step));
        }
        // First spike is identical; after that, spaced recovers to R≈1 → the
        // delivered weight stabilises at ≈ U0, while tight keeps decaying.
        assert!((tight[0] - spaced[0]).abs() < 1e-4);
        for i in 1..4 {
            assert!(
                spaced[i] > tight[i],
                "spike {i}: spaced {} should exceed tight {}",
                spaced[i],
                tight[i]
            );
            assert!(
                (spaced[i] - 0.5).abs() < 0.05,
                "spike {i}: spaced {:.4} ≈ U0 = 0.5",
                spaced[i]
            );
        }
    }

    /// Facilitation: with a small τ_fac, U accumulates across rapid spikes and
    /// the second spike delivers **more** than the first (U rises 0.1 → 0.19).
    #[test]
    fn test_tm_facilitation_second_spike_stronger() {
        // dt = 1 ms, τ_rec huge (no depression), τ_fac = 1 ms.
        let p = PlastRtProfile {
            u0: 0.1,
            a_rec: (-1e-6f32).exp(),
            a_fac: (-1.0f32).exp(),
            is_fac: true,
        };
        let mut r = 1.0f32;
        let mut u = 0.1f32;
        let mut last = 0u32;
        let w0 = tm_spike(&p, &mut r, &mut u, &mut last, 1.0, 0);
        let w1 = tm_spike(&p, &mut r, &mut u, &mut last, 1.0, 1);
        assert!(w1 > w0, "facilitation: {w1} must exceed {w0}");
        // U after two spikes: spike0 → 0.1 + 0.1·0.9 = 0.19, then 1-step
        // recovery toward 0.1 (e^(−1)) and spike1 → U + 0.1·(1 − U) ≈ 0.22.
        assert!(
            (u - 0.2198).abs() < 1e-3,
            "U after two spikes: {u} (expected ≈ 0.2198)"
        );
    }

    /// `PlasticityState::scale` passes non-plastic edges (id 0) through
    /// untouched and scales plastic edges by the current R·U.
    #[test]
    fn test_plasticity_scale_passthrough_and_scaling() {
        let profiles = [PlastRtProfile {
            u0: 0.5,
            a_rec: 1.0,
            a_fac: 0.0,
            is_fac: false,
        }];
        let plast_types = [0u8, 1u8];
        let mut r = vec![1.0f32, 1.0f32];
        let mut u = vec![0.0f32, 0.5f32];
        let mut last = vec![0u32, 0u32];
        let mut st = PlasticityState {
            profiles: &profiles,
            plast_types: &plast_types,
            r: &mut r,
            u: &mut u,
            last: &mut last,
        };
        assert_eq!(st.scale(0, 2.0, 5), 2.0, "id 0 → passthrough");
        assert_eq!(st.scale(1, 2.0, 5), 1.0, "2.0 · R(1) · U(0.5)");
    }

    /// Tsodyks–Markram против NEURON `tm.mod`: та же последовательность
    /// доставленных весов для фиксированного спайк-трейна. Эталоны
    /// `REF_TM_DEPRESSION`/`REF_TM_FACILITATION` генерится `gen_tm_ref.py`
    /// (NetStim → `TsodyksMarkram` synapse, `delivered = weight·R·U`). Сверяем
    /// `tm_spike` на тех же шагах (dt=0.5 мс, спайки на шагах 20+40k):
    /// депрессия сходится к геометрическому фиксированному весу, фасилитация
    /// сначала растёт, затем депрессия доминирует — 1:1 с NEURON.
    #[test]
    fn test_tm_matches_neuron_delivered_weights() {
        const REF_DEP: [f32; 12] = [
            0.500000, 0.295317, 0.211527, 0.177227, 0.163185, 0.157437, 0.155084, 0.154121,
            0.153726, 0.153565, 0.153499, 0.153472,
        ];
        const REF_FAC: [f32; 12] = [
            0.100000, 0.144296, 0.148667, 0.132751, 0.110006, 0.087407, 0.067775, 0.051800,
            0.039252, 0.029592, 0.022241, 0.016687,
        ];

        // Depression: U0=0.5, τ_rec=100 мс, dt=0.5 → a_rec = e^(−0.005).
        let dep = PlastRtProfile {
            u0: 0.5,
            a_rec: (-0.5f32 / 100.0).exp(),
            a_fac: 0.0,
            is_fac: false,
        };
        let mut r = 1.0f32;
        let mut u = 0.5f32;
        let mut last = 0u32;
        for (k, &ref_w) in REF_DEP.iter().enumerate() {
            let step = 20 + 40 * k;
            let w = tm_spike(&dep, &mut r, &mut u, &mut last, 1.0, step);
            assert!(
                (w - ref_w).abs() < 1e-4,
                "depression spike {}: got {w}, NEURON {ref_w}",
                k + 1
            );
        }

        // Facilitation: U0=0.1, τ_rec=1e6, τ_facil=50 мс.
        let fac = PlastRtProfile {
            u0: 0.1,
            a_rec: (-0.5f32 / 1e6).exp(),
            a_fac: (-0.5f32 / 50.0).exp(),
            is_fac: true,
        };
        let mut r = 1.0f32;
        let mut u = 0.1f32;
        let mut last = 0u32;
        for (k, &ref_w) in REF_FAC.iter().enumerate() {
            let step = 20 + 40 * k;
            let w = tm_spike(&fac, &mut r, &mut u, &mut last, 1.0, step);
            assert!(
                (w - ref_w).abs() < 1e-4,
                "facilitation spike {}: got {w}, NEURON {ref_w}",
                k + 1
            );
        }
    }
}
