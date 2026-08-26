use super::*;
use super::super::delay::DelayBuffer;
use super::super::graph::SparseGraph;
use super::super::population::{Population, PopulationId};
use super::super::push::PlastRtProfile;
use super::super::synapse::{PlasticityProfile, SynapseKinetics};
use crate::core::buf::AlignedVec;
use crate::zoo::Model;

/// A spiking neural network: populations + connections + delay buffer.
///
/// ## Step Cycle
///
/// 1. **Read** `delay[step % MAX_DELAY_SLOTS]` → `i_in`
/// 2. **Clear** that slot for future writes
/// 3. **Step** all populations with `i_in`
/// 4. **Compact** spikes → active indices
/// 5. **Push** active spikes through graph into future delay slots
/// 6. **Advance** `step_counter`
pub struct Network {
    /// Populations (heterogeneous via trait object)
    pub(crate) pops: Vec<Box<dyn Population>>,
    /// Global offset per population (cumulative neuron count)
    pub(crate) pop_offsets: Vec<usize>,
    pub(crate) total_neurons: usize,

    /// Sparse adjacency (CSR) built after `build()`.
    pub(crate) graph: Option<SparseGraph>,

    /// Edges accumulator before `build()`:
    /// `(pre, post, weight, delay, syn_type, plast_type)`.
    pub(crate) edges: Vec<(u32, u32, f32, u8, u8, u8)>,

    /// Ring buffer of currents: `[MAX_DELAY_SLOTS][total_neurons]`.
    /// `delay[step % MAX_DELAY_SLOTS][i]` = current arriving at neuron i.
    pub(crate) delay: DelayBuffer,
    /// Kinetic-synapse profiles (one per `register_kinetics` call): gating
    /// state (`g`/`x`) integrated each step, indexed by `syn_type - 1`.
    pub(crate) kinetics: Vec<KineticRuntime>,
    /// Short-term plasticity profiles (one per `register_plasticity` call),
    /// precomputed per-step recovery factors, indexed by `plast_type - 1`.
    pub(crate) plast_profiles: Vec<PlastRtProfile>,
    /// Per-edge plasticity profile id (`0` = none), sized `n_edges` at build.
    pub(crate) plast_types: Vec<u8>,
    /// Per-edge Tsodyks–Markram recovery `R`.
    pub(crate) plast_r: Vec<f32>,
    /// Per-edge release probability `U`.
    pub(crate) plast_u: Vec<f32>,
    /// Per-edge last-spike step (lazy recovery advance).
    pub(crate) plast_last: Vec<u32>,
    /// Delay rings for kinetic jumps, one per profile, `[slot][total_neurons]`.
    pub(crate) kinetic_delay: Vec<DelayBuffer>,
    /// Scratch for reading one kinetic delay slot per step (reused).
    pub(crate) kin_scratch: AlignedVec,
    /// Scratch for the per-neuron membrane voltage (kinetic driving force).
    pub(crate) v_scratch: AlignedVec,
    pub(crate) step_counter: usize,

    /// Scratch buffers (reused across steps to avoid allocation).
    pub(crate) i_in: AlignedVec,
    pub(crate) spikes_out: Vec<i32>,

    /// Reusable scratch for compacted spike indices (avoids Vec allocation every step).
    pub(crate) active_scratch: Vec<u32>,

    /// External Poisson spike sources (e.g. `poissonFiringSynapse` as a
    /// presynaptic population). Their spikes are rolled each step and pushed
    /// through their outgoing synapses like any real neuron's.
    pub(crate) poisson_sources: Vec<PoissonSourceRuntime>,

    /// Electrical synapses (gap junctions): `(pre_global, post_global, g)` —
    /// symmetric ohmic coupling `I = g·(V_j − V_i)` applied each step.
    pub(crate) gap_junctions: Vec<(u32, u32, f32)>,

    /// Analog (graded) synapses: `(pre_global, post_global, g)` — **one-way**
    /// ohmic coupling `I_post = g·(V_pre − V_post)` applied each step
    /// (`linearGradedSynapse` — "behaves like a one-way gap junction").
    pub(crate) graded_synapses: Vec<(u32, u32, f32)>,

    /// Sigmoid graded-synapse profiles ([`SigmoidGradedSynapse`]), integrated
    /// each step from the presynaptic membrane potential.
    pub(crate) sigmoid_graded: Vec<SigmoidGradedRuntime>,
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
            delay: DelayBuffer::new(0),
            kinetics: vec![],
            plast_profiles: vec![],
            plast_types: vec![],
            plast_r: vec![],
            plast_u: vec![],
            plast_last: vec![],
            kinetic_delay: vec![],
            kin_scratch: AlignedVec::new(0, 0.0),
            v_scratch: AlignedVec::new(0, 0.0),
            step_counter: 0,
            i_in: AlignedVec::new(0, 0.0),
            spikes_out: vec![],
            active_scratch: vec![],
            poisson_sources: vec![],
            gap_junctions: vec![],
            graded_synapses: vec![],
            sigmoid_graded: vec![],
        }
    }

    /// Add a pre-built population (boxed trait object).
    ///
    /// This is the type-erased entry point behind [`Network::add_population`];
    /// it is also how callers add populations whose dynamics are **not** a
    /// fixed [`Model`] type — e.g. [`crate::api::ConductancePopulation`], which
    /// runs an arbitrary data-driven channel set.
    ///
    /// # Panics
    /// Panics if `build()` has already been called.
    pub fn add_boxed_population(&mut self, pop: Box<dyn Population>) -> PopulationId {
        assert!(self.graph.is_none(), "Cannot add populations after build()");
        let count = pop.len();
        let new_total = self
            .total_neurons
            .checked_add(count)
            .expect("Network: total neuron count overflow");
        assert!(
            new_total <= u32::MAX as usize,
            "Network: total neuron count {} exceeds u32::MAX",
            new_total
        );
        let id = self.pops.len();
        self.pops.push(pop);
        self.total_neurons = new_total;
        self.pop_offsets.push(new_total);
        id
    }

    /// Add a population of `count` neurons.
    ///
    /// Returns a [`PopulationId`] for use with [`Network::connect`].
    ///
    /// # Panics
    /// Panics if `build()` has already been called.
    pub fn add_population<M: Model + 'static + Send + Sync>(
        &mut self,
        count: usize,
    ) -> PopulationId {
        let pop = crate::api::NeuralPopulation::<M>::new(count);
        self.add_boxed_population(Box::new(pop))
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
        let syn_types = vec![0u8; from_indices.len()];
        let plast_types = vec![0u8; from_indices.len()];
        self.connect_inner(
            from,
            from_indices,
            to,
            to_indices,
            weights,
            delays,
            &syn_types,
            &plast_types,
        );
    }

    /// Shared implementation: [`connect`] with explicit per-synapse types
    /// (`0` = static current, `1..` = kinetic profile id; `0` = no plasticity,
    /// `1..` = short-term plasticity profile id).
    fn connect_inner(
        &mut self,
        from: PopulationId,
        from_indices: &[u32],
        to: PopulationId,
        to_indices: &[u32],
        weights: &[f32],
        delays: &[u8],
        syn_types: &[u8],
        plast_types: &[u8],
    ) {
        assert!(self.graph.is_none(), "Cannot connect after build()");
        assert_eq!(from_indices.len(), to_indices.len());
        assert_eq!(from_indices.len(), weights.len());
        assert_eq!(from_indices.len(), delays.len());
        assert_eq!(from_indices.len(), syn_types.len());
        assert_eq!(from_indices.len(), plast_types.len());
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
            assert!(
                weights[i].is_finite(),
                "Network::connect: weight {i} must be finite, got {}",
                weights[i]
            );

            let pre = u32::try_from(from_offset + fi)
                .expect("Network::connect: global pre index exceeds u32::MAX");
            let post = u32::try_from(to_offset + ti)
                .expect("Network::connect: global post index exceeds u32::MAX");
            self.edges.push((
                pre,
                post,
                weights[i],
                delays[i],
                syn_types[i],
                plast_types[i],
            ));
        }
    }

    /// Connect `from` → `to` with a slice of edges — avoids juggling four
    /// parallel arrays. Each edge's `pre`/`post` are indices **within the
    /// respective populations**. Must be called before [`Network::build`].
    pub fn connect_edges(&mut self, from: PopulationId, to: PopulationId, edges: &[Edge]) {
        let mut from_indices = Vec::with_capacity(edges.len());
        let mut to_indices = Vec::with_capacity(edges.len());
        let mut weights = Vec::with_capacity(edges.len());
        let mut delays = Vec::with_capacity(edges.len());
        let mut syn_types = Vec::with_capacity(edges.len());
        let mut plast_types = Vec::with_capacity(edges.len());
        for e in edges {
            from_indices.push(e.pre);
            to_indices.push(e.post);
            weights.push(e.weight);
            delays.push(e.delay);
            syn_types.push(e.syn_type);
            plast_types.push(e.plast_type);
        }
        self.connect_inner(
            from,
            &from_indices,
            to,
            &to_indices,
            &weights,
            &delays,
            &syn_types,
            &plast_types,
        );
    }

    /// Add an external Poisson spike source (fires at `rate_hz` Hz, seeded)
    /// and return its id for [`Network::connect_poisson_source`]. `dt_ms` must
    /// match the simulation step. Must be called before [`Network::build`].
    ///
    /// # Panics
    /// If `build()` has already been called, `rate_hz` is non-finite/negative,
    /// or `dt_ms` is non-finite/≤ 0.
    pub fn add_poisson_source(&mut self, rate_hz: f32, dt_ms: f32, seed: u32) -> usize {
        assert!(
            self.graph.is_none(),
            "add_poisson_source must be called before build()"
        );
        assert!(
            rate_hz.is_finite() && rate_hz >= 0.0,
            "add_poisson_source: rate_hz must be finite and >= 0, got {rate_hz}"
        );
        assert!(
            dt_ms.is_finite() && dt_ms > 0.0,
            "add_poisson_source: dt_ms must be finite and > 0, got {dt_ms}"
        );
        let p = (rate_hz * dt_ms / 1000.0).clamp(0.0, 1.0);
        let id = self.poisson_sources.len();
        self.poisson_sources.push(PoissonSourceRuntime {
            p,
            seed,
            outgoing: Vec::new(),
            r: Vec::new(),
            u: Vec::new(),
            last: Vec::new(),
        });
        id
    }

    /// Connect a Poisson source's spikes to a target population through
    /// synapses. `posts` are indices **within `to`**; `weights[i]` and
    /// `delays[i]` mirror [`Network::connect`], and `syn_types[i]` selects
    /// static (`0`) vs kinetic (`1..`, from [`Network::register_kinetics`])
    /// delivery. Must be called before [`Network::build`].
    ///
    /// # Panics
    /// If `build()` has already been called, `source`/`to` are out of range,
    /// slice lengths mismatch, or a post index is out of bounds.
    pub fn connect_poisson_source(
        &mut self,
        source: usize,
        to: PopulationId,
        posts: &[u32],
        weights: &[f32],
        delays: &[u8],
        syn_types: &[u8],
        plast_types: &[u8],
    ) {
        assert!(self.graph.is_none(), "Cannot connect after build()");
        assert_eq!(posts.len(), weights.len());
        assert_eq!(posts.len(), delays.len());
        assert_eq!(posts.len(), syn_types.len());
        assert_eq!(posts.len(), plast_types.len());
        assert!(
            source < self.poisson_sources.len(),
            "poisson source {source} out of range (have {})",
            self.poisson_sources.len()
        );
        assert!(
            to < self.pops.len(),
            "to population {to} out of range (have {})",
            self.pops.len()
        );

        let to_offset = self.pop_offsets[to];
        let to_count = self.pop_offsets[to + 1] - to_offset;
        let src = &mut self.poisson_sources[source];
        for i in 0..posts.len() {
            let ti = posts[i] as usize;
            assert!(ti < to_count, "post index {ti} out of bounds");
            assert!(
                weights[i].is_finite(),
                "Network::connect_poisson_source: weight {i} must be finite, got {}",
                weights[i]
            );
            // Same contract as SparseGraph::from_triplets: a delay >=
            // MAX_DELAY_SLOTS silently aliases modulo 32 at delivery time.
            assert!(
                (delays[i] as usize) < super::super::graph::MAX_DELAY_SLOTS,
                "Network::connect_poisson_source: delay {} >= MAX_DELAY_SLOTS ({})",
                delays[i],
                super::super::graph::MAX_DELAY_SLOTS
            );
            let post = u32::try_from(to_offset + ti)
                .expect("Network::connect_poisson_source: global post index exceeds u32::MAX");
            src.outgoing
                .push((post, weights[i], delays[i], syn_types[i], plast_types[i]));
            let u0 = if plast_types[i] == 0 {
                0.0
            } else {
                self.plast_profiles[(plast_types[i] - 1) as usize].u0
            };
            src.r.push(1.0);
            src.u.push(u0);
            src.last.push(0u32);
        }
    }

    /// Add a symmetric ohmic **gap junction** (electrical synapse) between two
    /// neurons. `conductance` is raw (nS in the NeuroML convention); the
    /// coupling current is `I_i += g·(V_j − V_i)` / `I_j += g·(V_i − V_j)` each
    /// step. Must be called before [`Network::build`].
    ///
    /// # Panics
    /// If `build()` has already been called, populations/indices are out of
    /// range, or `conductance` is non-finite/negative.
    pub fn connect_gap_junction(
        &mut self,
        from: PopulationId,
        from_idx: u32,
        to: PopulationId,
        to_idx: u32,
        conductance: f32,
    ) {
        assert!(self.graph.is_none(), "Cannot connect after build()");
        assert!(
            conductance.is_finite() && conductance >= 0.0,
            "connect_gap_junction: conductance must be finite and >= 0, got {conductance}"
        );
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
        let fi = from_idx as usize;
        let ti = to_idx as usize;
        assert!(fi < from_count, "from index {fi} out of bounds");
        assert!(ti < to_count, "to index {ti} out of bounds");
        let a = u32::try_from(from_offset + fi)
            .expect("connect_gap_junction: global pre index exceeds u32::MAX");
        let b = u32::try_from(to_offset + ti)
            .expect("connect_gap_junction: global post index exceeds u32::MAX");
        self.gap_junctions.push((a, b, conductance));
    }

    /// Add a **one-way** graded (analog) synapse between two neurons — the
    /// postsynaptic neuron receives `I = g·(V_pre − V_post)` each step (the
    /// presynaptic neuron is unaffected), like a `linearGradedSynapse`.
    /// `conductance` is raw (nS in the NeuroML convention). Must be called
    /// before [`Network::build`].
    ///
    /// # Panics
    /// If `build()` has already been called, populations/indices are out of
    /// range, or `conductance` is non-finite/negative.
    pub fn connect_graded_synapse(
        &mut self,
        from: PopulationId,
        from_idx: u32,
        to: PopulationId,
        to_idx: u32,
        conductance: f32,
    ) {
        assert!(self.graph.is_none(), "Cannot connect after build()");
        assert!(
            conductance.is_finite() && conductance >= 0.0,
            "connect_graded_synapse: conductance must be finite and >= 0, got {conductance}"
        );
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
        let fi = from_idx as usize;
        let ti = to_idx as usize;
        assert!(fi < from_count, "from index {fi} out of bounds");
        assert!(ti < to_count, "to index {ti} out of bounds");
        let a = u32::try_from(from_offset + fi)
            .expect("connect_graded_synapse: global pre index exceeds u32::MAX");
        let b = u32::try_from(to_offset + ti)
            .expect("connect_graded_synapse: global post index exceeds u32::MAX");
        self.graded_synapses.push((a, b, conductance));
    }

    /// Register a sigmoid [`SigmoidGradedSynapse`] profile and return its id
    /// for [`Network::connect_sigmoid_graded`]. `dt_ms` must match the
    /// simulation step. Must be called before [`Network::build`].
    ///
    /// # Errors
    /// If the profile is invalid (non-finite, `delta == 0`, `k <= 0`, negative
    /// conductance).
    pub fn register_sigmoid_graded(
        &mut self,
        spec: SigmoidGradedSynapse,
        dt_ms: f32,
    ) -> Result<usize, String> {
        assert!(
            self.graph.is_none(),
            "register_sigmoid_graded must be called before build()"
        );
        assert!(
            dt_ms.is_finite() && dt_ms > 0.0,
            "register_sigmoid_graded: dt_ms must be finite and > 0, got {dt_ms}"
        );
        spec.validate()?;
        let id = self.sigmoid_graded.len();
        self.sigmoid_graded
            .push(SigmoidGradedRuntime::new(spec, dt_ms));
        Ok(id)
    }

    /// Connect a registered sigmoid graded-synapse profile from one neuron to
    /// another (one-way: only the post neuron receives current). `weight` is
    /// the per-connection multiplier. Must be called before [`Network::build`].
    ///
    /// # Panics
    /// If `build()` has already been called, `profile`/populations/indices are
    /// out of range, or `weight` is non-finite.
    pub fn connect_sigmoid_graded(
        &mut self,
        profile: usize,
        from: PopulationId,
        from_idx: u32,
        to: PopulationId,
        to_idx: u32,
        weight: f32,
    ) {
        assert!(self.graph.is_none(), "Cannot connect after build()");
        assert!(
            profile < self.sigmoid_graded.len(),
            "sigmoid graded profile {profile} out of range (have {})",
            self.sigmoid_graded.len()
        );
        assert!(
            weight.is_finite(),
            "connect_sigmoid_graded: weight must be finite, got {weight}"
        );
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
        let fi = from_idx as usize;
        let ti = to_idx as usize;
        assert!(fi < from_count, "from index {fi} out of bounds");
        assert!(ti < to_count, "to index {ti} out of bounds");
        let a = u32::try_from(from_offset + fi)
            .expect("connect_sigmoid_graded: global pre index exceeds u32::MAX");
        let b = u32::try_from(to_offset + ti)
            .expect("connect_sigmoid_graded: global post index exceeds u32::MAX");
        self.sigmoid_graded[profile].conns.push((a, b, weight, 0.0));
    }

    /// Register a kinetic-synapse waveform ([`SynapseKinetics`]) and return
    /// its 1-based profile id for use in [`Edge::with_synapse`].
    ///
    /// `dt` must match the simulation step (`ms`) used to drive the network —
    /// it precomputes the per-step decay factors. Must be called before
    /// [`Network::build`].
    ///
    /// # Errors
    /// If the profile is invalid (non-positive/non-finite time constants, or
    /// `tau_rise >= tau_decay`).
    pub fn register_kinetics(&mut self, k: SynapseKinetics, dt: f32) -> Result<u8, String> {
        assert!(
            self.graph.is_none(),
            "register_kinetics must be called before build()"
        );
        assert!(
            dt.is_finite() && dt > 0.0,
            "register_kinetics: dt must be finite and > 0, got {dt}"
        );
        k.validate()?;
        let id = self.kinetics.len() + 1;
        // Profile ids are u8 (0 = "no profile"); a 256th registration would
        // silently wrap to 0 and read as "no kinetics" downstream.
        assert!(
            id <= u8::MAX as usize,
            "Network::register_kinetics: profile id overflow (max {} profiles)",
            u8::MAX
        );
        self.kinetics.push(KineticRuntime::new(k, dt));
        Ok(id as u8)
    }

    /// Register a short-term plasticity profile ([`PlasticityProfile`],
    /// Tsodyks–Markram) and return its 1-based profile id for use in
    /// [`Edge::with_synapse_plasticity`].
    ///
    /// `dt` must match the simulation step (`ms`) used to drive the network —
    /// it precomputes the per-step recovery factors. Must be called before
    /// [`Network::build`].
    ///
    /// # Errors
    /// If the profile is invalid (non-finite/out-of-range `init_release_prob`,
    /// non-positive time constants).
    pub fn register_plasticity(&mut self, p: PlasticityProfile, dt: f32) -> Result<u8, String> {
        assert!(
            self.graph.is_none(),
            "register_plasticity must be called before build()"
        );
        assert!(
            dt.is_finite() && dt > 0.0,
            "register_plasticity: dt must be finite and > 0, got {dt}"
        );
        p.validate()?;
        let id = self.plast_profiles.len() + 1;
        // Same u8-id contract as register_kinetics.
        assert!(
            id <= u8::MAX as usize,
            "Network::register_plasticity: profile id overflow (max {} profiles)",
            u8::MAX
        );
        self.plast_profiles.push(PlastRtProfile {
            u0: p.init_release_prob,
            a_rec: (-dt / p.tau_rec).exp(),
            a_fac: p.tau_fac.map(|tf| (-dt / tf).exp()).unwrap_or(0.0),
            is_fac: p.tau_fac.is_some(),
        });
        Ok(id as u8)
    }
}
