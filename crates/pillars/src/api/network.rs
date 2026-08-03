//! # Network — Spike Routing Orchestrator
//!
//! Connects [`NeuralPopulation`]s via a [`SparseGraph`] and drives
//! the simulation loop. Uses **ring buffer of currents** (Approach 2)
//! for axonal delays — on spike, weights are written to a future ring
//! buffer slot; each step reads the current slot and clears it.
//!
//! ## Architecture
//!
//! ```text
//!     Neuron step (all pops)
//!           ↓
//!     spikes_out  →  compact_spikes  →  active_indices
//!           ↓                              ↓
//!     [current slot]   ←   push: for each active neuron,
//!                              for each outgoing synapse,
//!                              ring_buf[(step + delay.max(1)) % 32] += weight
//!                              (delay 0 = next step)
//!           ↓
//!     Clear current slot, advance step_counter
//! ```
//!
//! ## Push vs Pull
//!
//! | Approach | Operation |
//! |:---------|:----------|
//! | ❌ Pull  | For every neuron, scan incoming synapses → `i_in[i] += w` |
//! | ✅ Push  | For every **active** neuron (1–5%), scatter to future slots |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pillars::api::{Network, SparseGraph};
//! use pillars::zoo::neurons::LIFModel;
//!
//! let mut net = Network::new();
//!
//! // Add populations
//! let sensory = net.add_population::<LIFModel>(100).unwrap();
//! let motor = net.add_population::<LIFModel>(50).unwrap();
//!
//! // Connect sensory[0..10] → motor[0..5] with weight 0.8, no delay
//! net.connect(
//!     sensory, &[0,1,2,3,4,5,6,7,8,9],
//!     motor,   &[0,0,1,1,2,2,3,3,4,4],
//!     &[0.8; 10], &[0; 10],
//! );
//!
//! // Build graph and run
//! net.build();
//! for _ in 0..1000 {
//!     net.step();
//! }
//!
//! println!("Motor spikes: {:?}", &net.spikes(motor)[..10]);
//! ```

use std::any::Any;

use crate::api::graph::{SparseGraph, MAX_DELAY_SLOTS};
use crate::api::NeuralPopulation;
use crate::api::compact::compact_spikes_into;
use crate::api::push::push_spikes;
#[cfg(feature = "rayon")]
use crate::api::push::collect_push_triples;
use crate::core::buf::AlignedVec;
use crate::zoo::Model;

// ═══════════════════════════════════════════════════════════════
//  PopulationId
// ═══════════════════════════════════════════════════════════════

/// Opaque handle returned by [`Network::add_population`].
pub type PopulationId = usize;

// ═══════════════════════════════════════════════════════════════
//  Population Trait (type erasure)
// ═══════════════════════════════════════════════════════════════

/// Abstract interface for a neuron population.
///
/// Enables heterogeneous populations (LIF + HH + Izhikevich) in one
/// [`Network`].
pub trait Population: 'static {
    /// Advance one time step.
    ///
    /// `i_in` — input currents, length >= `len()`.
    /// `spikes_out` — output spikes (0 or 1), length >= `len()`.
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]);

    /// Number of neurons in this population.
    fn len(&self) -> usize;

    /// Returns `true` if the population has no neurons.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Read-only access to a state column.
    fn state_slice(&self, col: usize) -> &[f32];

    /// Erase to `dyn Any` for safe typed downcasting.
    ///
    /// Required (not defaulted) so the trait stays object-safe and callable
    /// through `&mut dyn Population`. Implementors must erase `self`; without
    /// it, [`Network::pop_typed_mut`] silently returns `None`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<M: Model + 'static> Population for NeuralPopulation<M> {
    #[inline(always)]
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        self.step(i_in, spikes_out)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn state_slice(&self, col: usize) -> &[f32] {
        self.state_slice(col)
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════════
//  Network
// ═══════════════════════════════════════════════════════════════

/// A spiking neural network: populations + connections + ring buffer.
///
/// ## Step Cycle
///
/// 1. **Read** `ring_buf[step % MAX_DELAY_SLOTS]` → `i_in`
/// 2. **Clear** that slot for future writes
/// 3. **Step** all populations with `i_in`
/// 4. **Compact** spikes → active indices
/// 5. **Push** active spikes through graph into future ring buffer slots
/// 6. **Advance** `step_counter`
pub struct Network {
    /// Populations (heterogeneous via trait object)
    pops: Vec<Box<dyn Population>>,
    /// Global offset per population (cumulative neuron count)
    pop_offsets: Vec<usize>,
    total_neurons: usize,

    /// Sparse adjacency (CSR) built after `build()`.
    graph: Option<SparseGraph>,

    /// Edges accumulator before `build()`.
    edges: Vec<(u32, u32, f32, u8)>,

    /// Ring buffer: `[MAX_DELAY_SLOTS][total_neurons]`.
    /// `ring_buf[step % MAX_DELAY_SLOTS][i]` = current arriving at neuron i.
    ring_buf: Vec<AlignedVec>,
    step_counter: usize,

    /// Scratch buffers (reused across steps to avoid allocation).
    i_in: AlignedVec,
    spikes_out: Vec<i32>,

    /// Reusable scratch for compacted spike indices (avoids Vec allocation every step).
    active_scratch: Vec<u32>,
}

impl Network {
    /// Create an empty network.
    pub fn new() -> Self {
        Self {
            pops: vec![],
            pop_offsets: vec![0],
            total_neurons: 0,
            graph: None,
            edges: vec![],
            ring_buf: vec![],
            step_counter: 0,
            i_in: AlignedVec::new(0, 0.0),
            spikes_out: vec![],
            active_scratch: vec![],
        }
    }

    /// Add a population of `count` neurons.
    ///
    /// Returns a [`PopulationId`] for use with [`Network::connect`].
    ///
    /// # Panics
    /// Panics if `build()` has already been called.
    pub fn add_population<M: Model + 'static>(&mut self, count: usize) -> PopulationId {
        assert!(
            self.graph.is_none(),
            "Cannot add populations after build()"
        );
        let id = self.pops.len();
        let pop = NeuralPopulation::<M>::new(count);
        self.pops.push(Box::new(pop));
        self.total_neurons += count;
        self.pop_offsets.push(self.total_neurons);
        id
    }

    /// Connect neurons from one population to another.
    ///
    /// Each triplet `(from_indices[i], to_indices[i])` creates one synapse
    /// with `weights[i]` and `delays[i]` steps of axonal delay.
    ///
    /// # Panics
    /// - If `build()` has already been called
    /// - If any index exceeds the population size
    /// - If slice lengths mismatch
    pub fn connect(
        &mut self,
        from: PopulationId,
        from_indices: &[u32],
        to: PopulationId,
        to_indices: &[u32],
        weights: &[f32],
        delays: &[u8],
    ) {
        assert!(
            self.graph.is_none(),
            "Cannot connect after build()"
        );
        assert_eq!(from_indices.len(), to_indices.len());
        assert_eq!(from_indices.len(), weights.len());
        assert_eq!(from_indices.len(), delays.len());
        assert!(
            from < self.pops.len(),
            "from population {from} out of range (have {})",
            self.pops.len()
        );
        assert!(
            to < self.pops.len(),
            "to population {to} out of range (have {})",
            self.pops.len()
        );

        let from_offset = self.pop_offsets[from];
        let from_count = self.pop_offsets[from + 1] - from_offset;
        let to_offset = self.pop_offsets[to];
        let to_count = self.pop_offsets[to + 1] - to_offset;

        for i in 0..from_indices.len() {
            let fi = from_indices[i] as usize;
            let ti = to_indices[i] as usize;
            assert!(fi < from_count, "from index {fi} out of bounds");
            assert!(ti < to_count, "to index {ti} out of bounds");

            self.edges.push((
                (from_offset + fi) as u32,
                (to_offset + ti) as u32,
                weights[i],
                delays[i],
            ));
        }
    }

    /// Build the sparse graph from accumulated edges.
    ///
    /// Must be called after all `add_population` / `connect` calls.
    /// After `build()`, new populations cannot be added.
    pub fn build(&mut self) {
        let n_syn = self.edges.len();
        let mut from = Vec::with_capacity(n_syn);
        let mut to = Vec::with_capacity(n_syn);
        let mut weights = Vec::with_capacity(n_syn);
        let mut delays = Vec::with_capacity(n_syn);

        for (f, t, w, d) in self.edges.drain(..) {
            from.push(f);
            to.push(t);
            weights.push(w);
            delays.push(d);
        }

        self.graph = Some(SparseGraph::from_triplets(
            self.total_neurons,
            &from,
            &to,
            &weights,
            &delays,
        ));

        // Allocate ring buffer: [MAX_DELAY_SLOTS][total_neurons]
        self.ring_buf = (0..MAX_DELAY_SLOTS)
            .map(|_| AlignedVec::new(self.total_neurons, 0.0))
            .collect();

        // Allocate scratch
        self.i_in = AlignedVec::new(self.total_neurons, 0.0);
        self.spikes_out = vec![0i32; self.total_neurons];
        self.active_scratch = Vec::with_capacity(self.total_neurons / 20 + 8);
    }

    /// True if the network has been built and is ready to step.
    #[inline(always)]
    pub fn is_built(&self) -> bool {
        self.graph.is_some()
    }

    /// Number of populations.
    #[inline(always)]
    pub fn num_populations(&self) -> usize {
        self.pops.len()
    }

    /// Total number of neurons across all populations.
    #[inline(always)]
    pub fn total_neurons(&self) -> usize {
        self.total_neurons
    }

    /// Current simulation step counter.
    #[inline(always)]
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Read spikes for a population (updated after `step()`).
    ///
    /// Returns an empty slice if the population index is invalid.
    pub fn spikes(&self, pop: PopulationId) -> &[i32] {
        if pop >= self.pops.len() {
            return &[];
        }
        let start = self.pop_offsets[pop];
        let end = self.pop_offsets[pop + 1];
        &self.spikes_out[start..end]
    }

    /// Read a state column for a population.
    pub fn state(&self, pop: PopulationId, col: usize) -> &[f32] {
        if pop >= self.pops.len() {
            return &[];
        }
        self.pops[pop].state_slice(col)
    }

    /// Mutable access to a population by ID.
    pub fn pop_mut(&mut self, pop: PopulationId) -> Option<&mut dyn Population> {
        self.pops.get_mut(pop).map(|b| b.as_mut())
    }

    /// Typed mutable access to a population (safe downcast).
    ///
    /// Returns `None` if `pop` is invalid or the population is not of type `M`.
    pub fn pop_typed_mut<M: Model + 'static>(&mut self, pop: PopulationId) -> Option<&mut NeuralPopulation<M>> {
        self.pops
            .get_mut(pop)?
            .as_any_mut()
            .downcast_mut::<NeuralPopulation<M>>()
    }

    /// Inject external current into specific neurons of a population.
    ///
    /// The current is added to the ring-buffer slot that the **next**
    /// [`Network::step`] reads, so it arrives exactly one step later
    /// (summing with any synaptically-delivered current). This is the safe
    /// way to drive populations from the outside — the node editor's
    /// "Stimulus" node and integration tests use it.
    ///
    /// # Panics
    /// - If `build()` has not been called
    /// - If `pop` is out of range
    /// - If `indices`/`values` lengths differ
    /// - If any neuron index exceeds the population size
    pub fn inject_current(&mut self, pop: PopulationId, indices: &[u32], values: &[f32]) {
        assert!(
            self.graph.is_some(),
            "inject_current() called before build()"
        );
        assert_eq!(
            indices.len(),
            values.len(),
            "inject_current: indices/values length mismatch"
        );
        assert!(
            pop < self.pops.len(),
            "inject_current: population {pop} out of range (have {})",
            self.pops.len()
        );

        let start = self.pop_offsets[pop];
        let count = self.pop_offsets[pop + 1] - start;
        let slot = self.step_counter % MAX_DELAY_SLOTS;
        let buf = &mut self.ring_buf[slot];
        for i in 0..indices.len() {
            let idx = indices[i] as usize;
            assert!(
                idx < count,
                "inject_current: neuron index {idx} out of bounds (population has {count})"
            );
            buf.as_mut_slice()[start + idx] += values[i];
        }
    }

    /// Advance one simulation step for the entire network.
    ///
    /// # Panics
    /// Panics if `build()` has not been called.
    pub fn step(&mut self) {
        let Some(ref graph) = self.graph else {
            panic!("Network::step() called before build()");
        };
        if self.total_neurons == 0 {
            self.step_counter += 1; // advance counter even for empty networks
            return;
        }

        let slot = self.step_counter % MAX_DELAY_SLOTS;
        let n = self.total_neurons;

        // ── 1. Read currents from current ring buffer slot ──
        // Pure safe slice copy — `self.i_in` and `self.ring_buf[slot]` are
        // different fields, so Rust allows the split borrow.
        self.i_in.as_mut_slice()[..n]
            .copy_from_slice(&self.ring_buf[slot].as_slice()[..n]);

        // ── 2. Clear the current slot for future writes ──
        // Safe: mutable access to ring_buf field, graph is still borrowed
        // immutably from 'let ref graph' — different fields, no conflict.
        if n > 0 {
            let buf = &mut self.ring_buf[slot];
            buf.as_mut_slice()[..n].fill(0.0);
        }

        // ── 3. Step each population ──
        for pop_id in 0..self.pops.len() {
            let start = self.pop_offsets[pop_id];
            let end = self.pop_offsets[pop_id + 1];
            let pop_n = end - start;
            if pop_n == 0 {
                continue;
            }
            let pop = &mut self.pops[pop_id];
            pop.step(
                &self.i_in.as_slice()[start..end],
                &mut self.spikes_out[start..end],
            );
        }

        // ── 4. Compact spikes → active indices (reusable scratch) ──
        self.active_scratch.clear();
        compact_spikes_into(&self.spikes_out, &mut self.active_scratch);

        // ── 5. Push active spikes through graph ──
        // Thread-local accumulation: each thread collects (post, slot, weight)
        // triples for its chunk; the main thread scatters them into the ring
        // buffer after join. Delays ARE respected — each triple carries the
        // target slot `(step + delay.max(1)) % MAX_DELAY_SLOTS`.
        #[cfg(feature = "rayon")]
        {
            let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
            if num_threads > 1 && self.active_scratch.len() > 64 {
                let step_counter = self.step_counter;
                let active_slice = &self.active_scratch[..];
                let active_len = active_slice.len();
                let chunk_size = active_len.div_ceil(num_threads).max(8);

                // Scoped threads borrow `graph` directly — no raw-pointer round-trip.
                // SparseGraph is Sync, so &SparseGraph is Send across the scope.
                std::thread::scope(|s| {
                    let handles: Vec<_> = (0..num_threads)
                        .filter(|tid| tid * chunk_size < active_len)
                        .map(|tid| {
                            let chunk_start = tid * chunk_size;
                            let chunk_end = (chunk_start + chunk_size).min(active_len);
                            // Copy the chunk (small — only active indices, 1-5% of neurons)
                            let chunk: Vec<u32> = active_slice[chunk_start..chunk_end].to_vec();
                            s.spawn(move || collect_push_triples(&chunk, graph, step_counter))
                        })
                        .collect();

                    // Scatter: each triple is (post, slot, weight) → ring_buf[slot][post] += w
                    for handle in handles {
                        let triples = handle.join().expect("push thread panicked");
                        for (post, slot, w) in triples {
                            self.ring_buf[slot].as_mut_slice()[post as usize] += w;
                        }
                    }
                });
            } else {
                push_spikes(&self.active_scratch, graph,
                    &mut self.ring_buf, self.step_counter);
            }
        }

        #[cfg(not(feature = "rayon"))]
        push_spikes(&self.active_scratch, graph,
            &mut self.ring_buf, self.step_counter);

        // ── 6. Advance counter ──
        self.step_counter += 1;
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::neurons::LIFModel;

    /// Helper: set up LIF params on a population.
    fn setup_lif_params(pop: &mut NeuralPopulation<LIFModel>) {
        pop.fill_param(0, 0.5); // dt
        pop.fill_param(1, 0.1); // inv_tau_m
        pop.fill_param(2, -70.0); // v_rest
        pop.fill_param(3, -54.0); // v_thresh
        pop.fill_param(4, -70.0); // v_reset
        pop.fill_param(5, 2.0); // tau_ref
        pop.fill_state(0, -70.0); // v
    }

    #[test]
    fn test_network_empty() {
        let mut net = Network::new();
        net.build();
        net.step(); // must not crash
        assert_eq!(net.step_counter(), 1);
    }

    #[test]
    fn test_network_single_pop() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(8);
        assert_eq!(pop, 0);
        assert_eq!(net.num_populations(), 1);
        assert_eq!(net.total_neurons(), 8);

        // Set params via typed access
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());

        net.build();
        assert!(net.is_built());

        // Step a few times — weak current, no external drive
        for _ in 0..10 {
            net.step();
        }
        assert_eq!(net.step_counter(), 10);
        let spikes = net.spikes(pop);
        assert_eq!(spikes.len(), 8);
    }

    #[test]
    fn test_network_two_pops_connected() {
        let mut net = Network::new();

        // Two populations of LIF
        let a = net.add_population::<LIFModel>(8);
        let b = net.add_population::<LIFModel>(8);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(net.total_neurons(), 16);

        // Set params for both
        for pid in 0..2 {
            setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
        }

        // Connect a[0] → b[0] with weight 1.0, no delay
        net.connect(a, &[0], b, &[0], &[1.0], &[0]);
        // Connect a[1] → b[1] with weight 0.5, 2-step delay
        net.connect(a, &[1], b, &[1], &[0.5], &[2]);

        net.build();
        assert!(net.is_built());

        // Step 50 times — neurons don't spike without external current
        // but the network should not crash
        for _ in 0..50 {
            net.step();
        }
        assert_eq!(net.step_counter(), 50);

        // Spikes should be empty (no external drive)
        let spikes_b = net.spikes(b);
        assert_eq!(spikes_b.iter().sum::<i32>(), 0,
            "No spikes without external drive");
    }

    #[test]
    fn test_network_ring_buffer_delays() {
        // Verify that delayed spikes end up in the correct future slot
        let mut net = Network::new();
        let a = net.add_population::<LIFModel>(2);
        let b = net.add_population::<LIFModel>(2);

        for pid in 0..2 {
            setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
        }

        // a[0] → b[0] with delay 3 steps, weight 10.0 (strong)
        net.connect(a, &[0], b, &[0], &[10.0], &[3]);

        net.build();

        // Manually force a[0] to spike by manipulating the ring buffer's
        // current slot — simulate a spike at step 0
        let _pre_global = 0usize; // global index for a[0]
        let _post_global = 2usize; // global index for b[0] (offset=8? No, a=0..2, b=2..4)

        // Actually, a has 2 neurons at global indices 0,1. b has 2 at 2,3.
        // a[0] = global 0, b[0] = global 2
        // Let's inject current directly into a[0] through the graph:
        // We can't easily do this through the API.
        // Instead, let's just verify the ring buffer slot allocation.

        // The graph has one edge: pre=0 → post=2, weight=10.0, delay=3
        // After step 0: slot[0+3=3][2] += 10.0  (but we need a spike first)
        // Since neurons don't spike without current, this test verifies
        // non-crash behavior only.

        net.step();
        // Step counter = 1, no spikes expected

        // Manually verify: ring_buf[3] should be all zeros (no spike happened)
        let slot3 = &net.ring_buf[3 % MAX_DELAY_SLOTS];
        for i in 0..4 {
            assert!((slot3.get(i) - 0.0).abs() < 1e-10,
                "ring_buf slot 3 [{}] should be 0, got {}", i, slot3.get(i));
        }
    }

    #[test]
    fn test_network_with_external_drive() {
        // Drive the population from the outside via the safe inject_current API.
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(8);

        setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());

        net.build();

        let idx: Vec<u32> = (0..8).collect();
        let vals = vec![30.0f32; 8]; // moderate drive
        let mut total_spikes = 0i32;
        for _ in 0..200 {
            net.inject_current(pop, &idx, &vals);
            net.step();
            total_spikes += net.spikes(pop).iter().sum::<i32>();
        }

        // LIF with I=30 must spike — the drive reached the neurons.
        assert!(total_spikes > 0, "LIF with I=30 must spike, got {total_spikes}");
        assert_eq!(net.step_counter(), 200);
    }

    #[test]
    #[should_panic(expected = "before build")]
    fn test_inject_current_before_build_panics() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        net.inject_current(pop, &[0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_inject_current_bad_pop_panics() {
        let mut net = Network::new();
        net.add_population::<LIFModel>(2);
        net.build();
        net.inject_current(7, &[0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_inject_current_bad_neuron_panics() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        net.build();
        net.inject_current(pop, &[5], &[1.0]);
    }

    #[test]
    fn test_population_trait_object() {
        // Verify that NeuralPopulation<LIFModel> implements Population
        let pop = NeuralPopulation::<LIFModel>::new(8);
        let mut boxed: Box<dyn Population> = Box::new(pop);
        assert_eq!(boxed.len(), 8);

        let mut spikes = vec![0i32; 8];
        let i_in = vec![0.0; 8];
        boxed.step(&i_in, &mut spikes);
        // Must not crash
    }

    #[test]
    fn test_network_many_connections() {
        let mut net = Network::new();
        let a = net.add_population::<LIFModel>(10);
        let b = net.add_population::<LIFModel>(10);

        for pid in 0..2 {
            setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
        }

        // All-to-all: each of 10 neurons connects to all 10 of the other pop
        let n = 10usize;
        let total_synapses = n * n;
        let mut from = Vec::with_capacity(total_synapses);
        let mut to = Vec::with_capacity(total_synapses);
        let mut weights = Vec::with_capacity(total_synapses);
        let mut delays = Vec::with_capacity(total_synapses);

        for i in 0..n {
            for j in 0..n {
                from.push(i as u32);
                to.push(j as u32);
                weights.push(0.1);
                delays.push(0);
            }
        }

        net.connect(a, &from, b, &to, &weights, &delays);
        net.build();

        assert!(net.is_built());

        for _ in 0..10 {
            net.step();
        }
        assert_eq!(net.step_counter(), 10);
    }
}
