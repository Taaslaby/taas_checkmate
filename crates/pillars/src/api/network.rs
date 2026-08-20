//! # Network — Spike Routing Orchestrator
//!
//! Connects [`NeuralPopulation`](crate::api::NeuralPopulation)s via a [`SparseGraph`] and drives
//! the simulation loop. Uses a **ring buffer of currents** (Approach 2)
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
//!     [current slot]   ←  push: for each active neuron,
//!                              for each outgoing synapse,
//!                              delay[(step + delay.max(1)) % 32] += weight
//!                              (delay 0 = next step)
//!           ↓
//!     Clear current slot, advance step_counter
//! ```
//!
//! ## Module boundaries (anti-monolith)
//!
//! | Concern | Module |
//! |---------|--------|
//! | Type erasure (`Population`, `PopulationId`) | [`population`](crate::api::population) |
//! | CSR adjacency | [`graph`](crate::api::graph) |
//! | Delay ring buffer | [`delay`](crate::api::delay) |
//! | Event-driven scatter (seq + par) | [`push`](crate::api::push) |
//! | **Orchestration (this file)** | [`Network`] |
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

use crate::api::compact::compact_spikes_into;
use crate::api::delay::DelayBuffer;
use crate::api::graph::{SparseGraph, MAX_DELAY_SLOTS};
use crate::api::population::{CableCa, CableVoltage, Population, PopulationId};
#[cfg(not(feature = "rayon"))]
use crate::api::push::push_spikes;
#[cfg(feature = "rayon")]
use crate::api::push::push_spikes_par;
use crate::api::push::{tm_spike, PlastRtProfile, PlasticityState};
use crate::api::synapse::{exp_waveform_factor, PlasticityProfile, SynapseKinetics};
use crate::core::buf::AlignedVec;
use crate::zoo::Model;

// ═══════════════════════════════════════════════════════════════
//  Network
// ═══════════════════════════════════════════════════════════════

/// A single directed edge: `pre` (index in the source population), `post`
/// (index in the target population), `weight` and axonal `delay` (steps,
/// `0` = delivered on the next step).
///
/// See [`Network::connect_edges`] for ergonomic connection.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub pre: u32,
    pub post: u32,
    pub weight: f32,
    pub delay: u8,
    /// Synapse type: `0` = static current (default), `1..` = a kinetic
    /// profile id from [`Network::register_kinetics`].
    pub syn_type: u8,
    /// Short-term plasticity profile: `0` = none (default), `1..` = a profile
    /// id from [`Network::register_plasticity`].
    pub plast_type: u8,
}

impl Edge {
    #[inline]
    pub fn new(pre: u32, post: u32, weight: f32, delay: u8) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type: 0,
            plast_type: 0,
        }
    }

    /// An edge with a kinetic synapse type (see [`Network::register_kinetics`]).
    #[inline]
    pub fn with_synapse(pre: u32, post: u32, weight: f32, delay: u8, syn_type: u8) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type,
            plast_type: 0,
        }
    }

    /// An edge with a kinetic synapse type **and** a short-term plasticity
    /// profile (see [`Network::register_kinetics`] / [`Network::register_plasticity`]).
    #[inline]
    pub fn with_synapse_plasticity(
        pre: u32,
        post: u32,
        weight: f32,
        delay: u8,
        syn_type: u8,
        plast_type: u8,
    ) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            syn_type,
            plast_type,
        }
    }
}

/// Deterministic, stateless pseudo-Poisson roll: a uniform `[0, 1)` value
/// hashed from `(seed, stream, step)`, so the same seed reproduces the same
/// spike train every run without mutable RNG state (CPU/GPU/repeated runs).
/// SplitMix64 finaliser over a mixed key.
pub(crate) fn poisson_spike(seed: u32, stream: u32, step: u64, p: f32) -> bool {
    let mut z = ((seed as u64) << 32) | (stream as u64);
    z ^= step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = (z >> 40) as f32 / (1u64 << 24) as f32;
    u < p
}

/// Runtime state of one external Poisson spike source (e.g. NeuroML
/// `poissonFiringSynapse` used as a presynaptic population). Fires at per-step
/// probability `p = rate·dt` and scatters its outgoing synapses through the
/// same delay/kinetics machinery as a real neuron's spikes.
struct PoissonSourceRuntime {
    /// Per-step spike probability (`rate_hz · dt_s`), clamped to `[0, 1]`.
    p: f32,
    seed: u32,
    /// Outgoing synapses: `(post_global, weight, delay, syn_type, plast_type)`.
    outgoing: Vec<(u32, f32, u8, u8, u8)>,
    /// Per-outgoing Tsodyks–Markram state (parallel to `outgoing`).
    r: Vec<f32>,
    u: Vec<f32>,
    last: Vec<u32>,
}

/// A sigmoid `gradedSynapse` profile — a **stateful** analog conductance whose
/// activation follows the presynaptic membrane potential:
///
/// ```text
/// inf(vpre) = 1 / (1 + exp(-(vpre − Vth)/delta))
/// tau(vpre) = 1 / (k · (1 + exp((vpre − Vth)/delta)))
/// ds/dt     = (inf − s) / tau
/// I_post    = weight · conductance · s · (erev − V_post)
/// ```
///
/// This is the NeuroML2 `gradedSynapse` (a one-way, continuously-transmitting
/// analogue synapse; the presynaptic neuron is unaffected).
#[derive(Debug, Clone, Copy)]
pub struct SigmoidGradedSynapse {
    /// Half-activation voltage (mV).
    pub vth: f32,
    /// Activation-curve slope (mV); must be non-zero.
    pub delta: f32,
    /// Transmitter–receptor dissociation rate (per ms); must be `> 0`.
    pub k: f32,
    /// Reversal potential (mV).
    pub erev: f32,
    /// Conductance (raw nS, as the other synapse weights).
    pub conductance: f32,
}

impl SigmoidGradedSynapse {
    pub fn new(vth: f32, delta: f32, k: f32, erev: f32, conductance: f32) -> Self {
        Self {
            vth,
            delta,
            k,
            erev,
            conductance,
        }
    }

    /// Validate the profile: finite parameters, `delta != 0`, `k > 0`,
    /// non-negative `conductance`.
    pub fn validate(&self) -> Result<(), String> {
        let p = [self.vth, self.delta, self.k, self.erev, self.conductance];
        if p.iter().any(|x| !x.is_finite()) {
            return Err("SigmoidGradedSynapse: parameters must be finite".into());
        }
        if self.delta == 0.0 {
            return Err("SigmoidGradedSynapse: delta must be non-zero".into());
        }
        if self.k <= 0.0 {
            return Err("SigmoidGradedSynapse: k must be > 0".into());
        }
        if self.conductance < 0.0 {
            return Err("SigmoidGradedSynapse: conductance must be >= 0".into());
        }
        Ok(())
    }
}

/// Runtime state of one registered [`SigmoidGradedSynapse`] profile: shared
/// parameters plus a per-connection activation state `s`.
struct SigmoidGradedRuntime {
    vth: f32,
    inv_delta: f32,
    k: f32,
    erev: f32,
    conductance: f32,
    dt_ms: f32,
    /// `(pre_global, post_global, weight, s)` — `s` is the activation state.
    conns: Vec<(u32, u32, f32, f32)>,
}

impl SigmoidGradedRuntime {
    fn new(spec: SigmoidGradedSynapse, dt_ms: f32) -> Self {
        Self {
            vth: spec.vth,
            inv_delta: 1.0 / spec.delta,
            k: spec.k,
            erev: spec.erev,
            conductance: spec.conductance,
            dt_ms,
            conns: Vec::new(),
        }
    }

    /// Integrate every connection one step (exponential Euler on `s`) and add
    /// the resulting current `weight·conductance·s·(erev − V_post)` to `i_in`.
    fn step(&mut self, v: &[f32], i_in: &mut [f32]) {
        for (pre, post, weight, s) in &mut self.conns {
            let vpre = v[*pre as usize];
            let vpost = v[*post as usize];
            let arg = (vpre - self.vth) * self.inv_delta;
            let inf = 1.0 / (1.0 + (-arg).exp());
            let tau = 1.0 / (self.k * (1.0 + arg.exp()));
            let decay = (-self.dt_ms / tau).exp();
            *s += (inf - *s) * (1.0 - decay);
            let current = *weight * self.conductance * *s * (self.erev - vpost);
            i_in[*post as usize] += current;
        }
    }
}

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
    pops: Vec<Box<dyn Population>>,
    /// Global offset per population (cumulative neuron count)
    pop_offsets: Vec<usize>,
    total_neurons: usize,

    /// Sparse adjacency (CSR) built after `build()`.
    graph: Option<SparseGraph>,

    /// Edges accumulator before `build()`:
    /// `(pre, post, weight, delay, syn_type, plast_type)`.
    edges: Vec<(u32, u32, f32, u8, u8, u8)>,

    /// Ring buffer of currents: `[MAX_DELAY_SLOTS][total_neurons]`.
    /// `delay[step % MAX_DELAY_SLOTS][i]` = current arriving at neuron i.
    delay: DelayBuffer,
    /// Kinetic-synapse profiles (one per `register_kinetics` call): gating
    /// state (`g`/`x`) integrated each step, indexed by `syn_type - 1`.
    kinetics: Vec<KineticRuntime>,
    /// Short-term plasticity profiles (one per `register_plasticity` call),
    /// precomputed per-step recovery factors, indexed by `plast_type - 1`.
    plast_profiles: Vec<PlastRtProfile>,
    /// Per-edge plasticity profile id (`0` = none), sized `n_edges` at build.
    plast_types: Vec<u8>,
    /// Per-edge Tsodyks–Markram recovery `R`.
    plast_r: Vec<f32>,
    /// Per-edge release probability `U`.
    plast_u: Vec<f32>,
    /// Per-edge last-spike step (lazy recovery advance).
    plast_last: Vec<u32>,
    /// Delay rings for kinetic jumps, one per profile, `[slot][total_neurons]`.
    kinetic_delay: Vec<DelayBuffer>,
    /// Scratch for reading one kinetic delay slot per step (reused).
    kin_scratch: AlignedVec,
    /// Scratch for the per-neuron membrane voltage (kinetic driving force).
    v_scratch: AlignedVec,
    step_counter: usize,

    /// Scratch buffers (reused across steps to avoid allocation).
    i_in: AlignedVec,
    spikes_out: Vec<i32>,

    /// Reusable scratch for compacted spike indices (avoids Vec allocation every step).
    active_scratch: Vec<u32>,

    /// External Poisson spike sources (e.g. `poissonFiringSynapse` as a
    /// presynaptic population). Their spikes are rolled each step and pushed
    /// through their outgoing synapses like any real neuron's.
    poisson_sources: Vec<PoissonSourceRuntime>,

    /// Electrical synapses (gap junctions): `(pre_global, post_global, g)` —
    /// symmetric ohmic coupling `I = g·(V_j − V_i)` applied each step.
    gap_junctions: Vec<(u32, u32, f32)>,

    /// Analog (graded) synapses: `(pre_global, post_global, g)` — **one-way**
    /// ohmic coupling `I_post = g·(V_pre − V_post)` applied each step
    /// (`linearGradedSynapse` — "behaves like a one-way gap junction").
    graded_synapses: Vec<(u32, u32, f32)>,

    /// Sigmoid graded-synapse profiles ([`SigmoidGradedSynapse`]), integrated
    /// each step from the presynaptic membrane potential.
    sigmoid_graded: Vec<SigmoidGradedRuntime>,
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
        self.plast_profiles.push(PlastRtProfile {
            u0: p.init_release_prob,
            a_rec: (-dt / p.tau_rec).exp(),
            a_fac: p.tau_fac.map(|tf| (-dt / tf).exp()).unwrap_or(0.0),
            is_fac: p.tau_fac.is_some(),
        });
        Ok(id as u8)
    }

    /// Build the sparse graph from accumulated edges.
    ///
    /// Must be called after all `add_population` / `connect` calls.
    /// After `build()`, new populations cannot be added.
    ///
    /// # Panics
    /// If called twice: `build()` **drains** [`Self::edges`], so a second call
    /// would silently rebuild the graph with no connections. Rebuilding is
    /// rejected explicitly rather than corrupting the graph.
    pub fn build(&mut self) {
        assert!(
            self.graph.is_none(),
            "Network::build() called twice — the graph is already built; connections would be lost"
        );
        let n_syn = self.edges.len();
        let mut from = Vec::with_capacity(n_syn);
        let mut to = Vec::with_capacity(n_syn);
        let mut weights = Vec::with_capacity(n_syn);
        let mut delays = Vec::with_capacity(n_syn);
        let mut syn_types = Vec::with_capacity(n_syn);
        let mut plast_types = Vec::with_capacity(n_syn);

        for (f, t, w, d, st, pt) in self.edges.drain(..) {
            from.push(f);
            to.push(t);
            weights.push(w);
            delays.push(d);
            syn_types.push(st);
            plast_types.push(pt);
        }

        for &st in &syn_types {
            assert!(
                (st as usize) <= self.kinetics.len(),
                "Network::build: syn_type {st} exceeds registered kinetics ({})",
                self.kinetics.len()
            );
        }
        for &pt in &plast_types {
            assert!(
                (pt as usize) <= self.plast_profiles.len(),
                "Network::build: plast_type {pt} exceeds registered plasticity profiles ({})",
                self.plast_profiles.len()
            );
        }
        for src in &self.poisson_sources {
            for &(_, _, _, st, pt) in &src.outgoing {
                assert!(
                    (st as usize) <= self.kinetics.len(),
                    "Network::build: Poisson-source syn_type {st} exceeds registered kinetics ({})",
                    self.kinetics.len()
                );
                assert!(
                    (pt as usize) <= self.plast_profiles.len(),
                    "Network::build: Poisson-source plast_type {pt} exceeds registered plasticity ({})",
                    self.plast_profiles.len()
                );
            }
        }

        // Per-edge short-term plasticity state (R, U, last-spike step).
        self.plast_types = plast_types;
        let profiles = &self.plast_profiles;
        self.plast_r = vec![1.0; n_syn];
        self.plast_u = self
            .plast_types
            .iter()
            .map(|&pt| {
                if pt == 0 {
                    0.0
                } else {
                    profiles[(pt - 1) as usize].u0
                }
            })
            .collect();
        self.plast_last = vec![0u32; n_syn];

        self.graph = Some(SparseGraph::from_triplets_with_types(
            self.total_neurons,
            &from,
            &to,
            &weights,
            &delays,
            &syn_types,
        ));

        // Allocate delay ring buffer + scratch
        self.delay = DelayBuffer::new(self.total_neurons);
        // Allocate kinetic runtime state + per-profile delay rings.
        for k in &mut self.kinetics {
            k.allocate(self.total_neurons);
        }
        self.kinetic_delay = (0..self.kinetics.len())
            .map(|_| DelayBuffer::new(self.total_neurons))
            .collect();
        self.kin_scratch = AlignedVec::new(self.total_neurons, 0.0);
        self.v_scratch = AlignedVec::new(self.total_neurons, 0.0);
        self.i_in = AlignedVec::new(self.total_neurons, 0.0);
        self.spikes_out = vec![0i32; self.total_neurons];
        self.active_scratch = Vec::with_capacity(self.total_neurons / 20 + 8);
    }

    /// Per-edge short-term plasticity profile ids (`0` = none, `1..` = a
    /// profile from [`Network::register_plasticity`]). Sized `n_edges` after
    /// [`Network::build`] — lets callers verify plasticity routing.
    pub fn plast_types(&self) -> &[u8] {
        &self.plast_types
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

    /// Per-compartment membrane potentials of a **cable** population (a
    /// morphologically-detailed cell), if `pop` is one. Returns `None` for
    /// point-neuron populations and out-of-range ids.
    pub fn cable_voltage(&self, pop: PopulationId) -> Option<CableVoltage<'_>> {
        self.pops.get(pop).and_then(|p| p.cable_voltage())
    }

    /// Per-compartment intracellular Ca²⁺ concentrations of a **cable**
    /// population that tracks a Ca²⁺ `species`, if `pop` is one. Returns `None`
    /// for point-neuron populations, Ca-less cables and out-of-range ids.
    pub fn cable_ca(&self, pop: PopulationId) -> Option<CableCa<'_>> {
        self.pops.get(pop).and_then(|p| p.cable_ca())
    }

    /// Read a parameter column for a population.
    pub fn param_slice(&self, pop: PopulationId, col: usize) -> &[f32] {
        if pop >= self.pops.len() {
            return &[];
        }
        self.pops[pop].param_slice(col)
    }

    /// Read-only access to the built sparse graph (CSR), if `build()` was
    /// called. Used by GPU backends to mirror the projection graph.
    pub fn graph(&self) -> Option<&SparseGraph> {
        self.graph.as_ref()
    }

    /// Mutable access to a population by ID.
    pub fn pop_mut(&mut self, pop: PopulationId) -> Option<&mut dyn Population> {
        self.pops.get_mut(pop).map(|b| b.as_mut())
    }

    /// Typed mutable access to a population (safe downcast).
    ///
    /// Returns `None` if `pop` is invalid or the population is not of type `M`.
    pub fn pop_typed_mut<M: Model + 'static + Send + Sync>(
        &mut self,
        pop: PopulationId,
    ) -> Option<&mut crate::api::NeuralPopulation<M>> {
        self.pops
            .get_mut(pop)?
            .as_any_mut()
            .downcast_mut::<crate::api::NeuralPopulation<M>>()
    }

    /// Configure a population's model parameters / state via its typed model.
    ///
    /// The ergonomic replacement for `pop_typed_mut` — the closure gets the
    /// typed [`crate::api::NeuralPopulation<M>`] directly, so you can `fill_param`/
    /// `fill_state` by the model's column indices.
    ///
    /// # Panics
    /// - If `pop` is out of range
    /// - If the population is **not** of model type `M` (create it with
    ///   [`Network::add_population`] using the same `M`)
    pub fn configure<M: Model + 'static + Send + Sync>(
        &mut self,
        pop: PopulationId,
        f: impl FnOnce(&mut crate::api::NeuralPopulation<M>),
    ) {
        let p = self
            .pop_typed_mut::<M>(pop)
            .expect("Network::configure: population is not of the requested model type M");
        f(p);
    }

    /// Inject external current into specific neurons of a population.
    ///
    /// The current is added to the delay-buffer slot `step_counter % MAX_DELAY`
    /// — the **same** slot the next [`Network::step`] reads, so it drives the
    /// population in that step (same-step arrival, not one step later),
    /// summing with any synaptically-delivered current. This is the safe
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
        for i in 0..indices.len() {
            let idx = indices[i] as usize;
            assert!(
                idx < count,
                "inject_current: neuron index {idx} out of bounds (population has {count})"
            );
            assert!(
                values[i].is_finite(),
                "inject_current: value {i} must be finite, got {}",
                values[i]
            );
            self.delay.add(slot, start + idx, values[i]);
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

        // ── 1. Read currents from current delay slot ──
        // Safe: `i_in` and `delay` are different fields → split borrow.
        self.delay.read_slot(slot, self.i_in.as_mut_slice());

        // ── 2. Clear the current slot for future writes ──
        self.delay.clear_slot(slot);

        // ── 2b. Copy membrane voltage for the kinetic driving force and gap
        // junctions (semi-implicit: V at the start of the step, like the HH
        // derivimplicit path). ──
        let need_v = !self.kinetics.is_empty()
            || !self.gap_junctions.is_empty()
            || !self.graded_synapses.is_empty()
            || !self.sigmoid_graded.is_empty();
        if need_v {
            let v = self.v_scratch.as_mut_slice();
            for pop_id in 0..self.pops.len() {
                let start = self.pop_offsets[pop_id];
                let end = self.pop_offsets[pop_id + 1];
                if end == start {
                    continue;
                }
                let state0 = self.pops[pop_id].state_slice(0);
                v[start..end].copy_from_slice(&state0[..end - start]);
            }
        }

        // Kinetic synapses: decay state, apply arriving jumps, and add the
        // resulting conductance current `g·(erev − V)` on top of the static
        // input for this step.
        if !self.kinetics.is_empty() {
            for (k, d) in self.kinetics.iter_mut().zip(self.kinetic_delay.iter_mut()) {
                k.step(
                    d,
                    slot,
                    self.kin_scratch.as_mut_slice(),
                    self.v_scratch.as_slice(),
                    self.i_in.as_mut_slice(),
                    self.total_neurons,
                );
            }
        }

        // Gap junctions (electrical synapses): symmetric ohmic coupling
        // `I_i += g·(V_j − V_i)`, `I_j += g·(V_i − V_j)`, from the start-of-step
        // V (explicit in V — stable for g·dt ≪ 1).
        if !self.gap_junctions.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for &(a, b, g) in &self.gap_junctions {
                let dv = v[b as usize] - v[a as usize];
                i_in[a as usize] += g * dv;
                i_in[b as usize] -= g * dv;
            }
        }

        // Graded (analog) synapses: one-way ohmic coupling
        // `I_post += g·(V_pre − V_post)`.
        if !self.graded_synapses.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for &(pre, post, g) in &self.graded_synapses {
                let dv = v[pre as usize] - v[post as usize];
                i_in[post as usize] += g * dv;
            }
        }

        // Sigmoid graded synapses: stateful activation `s` driven by V_pre,
        // current `weight·g·s·(erev − V_post)` into the post neuron.
        if !self.sigmoid_graded.is_empty() {
            let v = self.v_scratch.as_slice();
            let i_in = self.i_in.as_mut_slice();
            for rt in &mut self.sigmoid_graded {
                rt.step(v, i_in);
            }
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
        // Thread-local accumulation lives in `push_spikes_par` (push.rs):
        // Short-term plasticity state for this push: per-edge R/U/last. The
        // push scales each delivered weight by the edge's current R·U.
        let mut plast = if self.plast_profiles.is_empty() && self.plast_types.is_empty() {
            None
        } else {
            Some(PlasticityState {
                profiles: &self.plast_profiles,
                plast_types: &self.plast_types,
                r: &mut self.plast_r,
                u: &mut self.plast_u,
                last: &mut self.plast_last,
            })
        };
        // each thread collects (post, slot, weight) triples for its chunk;
        // the caller scatters them into the delay buffer after join.
        // Delays ARE respected — each triple carries the target slot.
        #[cfg(feature = "rayon")]
        push_spikes_par(
            &self.active_scratch,
            graph,
            &mut self.delay,
            &mut self.kinetic_delay,
            self.step_counter,
            plast.as_mut(),
        );

        #[cfg(not(feature = "rayon"))]
        push_spikes(
            &self.active_scratch,
            graph,
            &mut self.delay,
            &mut self.kinetic_delay,
            self.step_counter,
            plast.as_mut(),
        );

        // ── 5b. Roll Poisson sources and scatter their spikes ──
        // Same minimum-1-step delay rule as push_spikes (a spike fired at step
        // `s` is delivered at step `s + max(delay, 1)`).
        for src in &mut self.poisson_sources {
            if src.p <= 0.0 || !poisson_spike(src.seed, 0, self.step_counter as u64, src.p) {
                continue;
            }
            let step = self.step_counter;
            for i in 0..src.outgoing.len() {
                let (post, w, d, st, pt) = src.outgoing[i];
                let mut w = w;
                if pt > 0 {
                    let p = &self.plast_profiles[(pt - 1) as usize];
                    w = tm_spike(p, &mut src.r[i], &mut src.u[i], &mut src.last[i], w, step);
                }
                let target_slot = (step + (d as usize).max(1)) % MAX_DELAY_SLOTS;
                if st == 0 {
                    self.delay.add(target_slot, post as usize, w);
                } else {
                    self.kinetic_delay[(st as usize) - 1].add(target_slot, post as usize, w);
                }
            }
        }

        // ── 6. Advance counter ──
        self.step_counter += 1;
    }
}

/// Per-neuron kinetic-synapse state for one registered profile.
///
/// The ODE systems match the NeuroML/LEMS component definitions and are
/// integrated exactly (exponential Euler on linear ODEs, jumps applied at the
/// step boundary):
///
/// * expOne: `g` decays; the jump is added to `g`.
/// * expTwo: `g` (decay) and `x` (rise); the jump is added to both;
///   current = `norm·(g − x)`.
/// * alpha: `g` (conductance) and `a` (auxiliary); the jump is added to `a`
///   and `g` integrates `dg/dt = (e·a − g)/τ`.
/// * expThree: `a` (shared rise), `b` (decay1), `c` (decay2); the jump is
///   distributed per-state and the conductance is
///   `gbase1·(b−a) + gbase2·(c−a)`.
struct KineticRuntime {
    erev: f32,
    kind: KinKind,
    /// expOne/expTwo decay accumulator; alpha conductance.
    g: AlignedVec,
    /// expTwo rise accumulator.
    x: AlignedVec,
    /// alpha auxiliary state.
    a: AlignedVec,
    /// expThree decay-1 accumulator.
    b: AlignedVec,
    /// expThree decay-2 accumulator.
    c: AlignedVec,
}

/// Precomputed per-step integration constants for one waveform kind.
#[derive(Clone, Copy)]
enum KinKind {
    ExpOne {
        a_decay: f32,
    },
    ExpTwo {
        a_rise: f32,
        a_decay: f32,
        norm: f32,
    },
    Alpha {
        r: f32,
        e_r_dt_tau: f32,
    },
    ExpThree {
        a_rise: f32,
        a_decay1: f32,
        a_decay2: f32,
        wf1: f32,
        wf2: f32,
        k_a: f32,
        gbase1: f32,
        gbase2: f32,
    },
}

impl KineticRuntime {
    fn new(k: SynapseKinetics, dt: f32) -> Self {
        let erev = k.erev();
        let kind = match k {
            SynapseKinetics::ExpOne { tau_decay, .. } => KinKind::ExpOne {
                a_decay: (-dt / tau_decay).exp(),
            },
            SynapseKinetics::ExpTwo {
                tau_rise,
                tau_decay,
                ..
            } => KinKind::ExpTwo {
                a_rise: (-dt / tau_rise).exp(),
                a_decay: (-dt / tau_decay).exp(),
                norm: exp_waveform_factor(tau_rise, tau_decay),
            },
            SynapseKinetics::Alpha { tau, .. } => {
                let r = (-dt / tau).exp();
                KinKind::Alpha {
                    r,
                    e_r_dt_tau: std::f32::consts::E * (dt / tau) * r,
                }
            }
            SynapseKinetics::ExpThree {
                tau_rise,
                tau_decay1,
                tau_decay2,
                gbase1,
                gbase2,
                ..
            } => {
                let wf1 = exp_waveform_factor(tau_rise, tau_decay1);
                let wf2 = exp_waveform_factor(tau_rise, tau_decay2);
                KinKind::ExpThree {
                    a_rise: (-dt / tau_rise).exp(),
                    a_decay1: (-dt / tau_decay1).exp(),
                    a_decay2: (-dt / tau_decay2).exp(),
                    wf1,
                    wf2,
                    k_a: (gbase1 * wf1 + gbase2 * wf2) / (gbase1 + gbase2),
                    gbase1,
                    gbase2,
                }
            }
        };
        Self {
            erev,
            kind,
            g: AlignedVec::new(0, 0.0),
            x: AlignedVec::new(0, 0.0),
            a: AlignedVec::new(0, 0.0),
            b: AlignedVec::new(0, 0.0),
            c: AlignedVec::new(0, 0.0),
        }
    }

    fn allocate(&mut self, n: usize) {
        match self.kind {
            KinKind::ExpOne { .. } => self.g = AlignedVec::new(n, 0.0),
            KinKind::ExpTwo { .. } => {
                self.g = AlignedVec::new(n, 0.0);
                self.x = AlignedVec::new(n, 0.0);
            }
            KinKind::Alpha { .. } => {
                self.g = AlignedVec::new(n, 0.0);
                self.a = AlignedVec::new(n, 0.0);
            }
            KinKind::ExpThree { .. } => {
                self.a = AlignedVec::new(n, 0.0);
                self.b = AlignedVec::new(n, 0.0);
                self.c = AlignedVec::new(n, 0.0);
            }
        }
    }

    /// Advance one step: read arriving jumps from `delay` slot, decay the
    /// state, and add the conductance current `g·(erev − V)` into `i_in`.
    fn step(
        &mut self,
        delay: &mut DelayBuffer,
        slot: usize,
        scratch: &mut [f32],
        v: &[f32],
        i_in: &mut [f32],
        n: usize,
    ) {
        delay.read_slot(slot, &mut scratch[..n]);
        delay.clear_slot(slot);
        let erev = self.erev;
        match self.kind {
            KinKind::ExpOne { a_decay } => {
                let g = self.g.as_mut_slice();
                for i in 0..n {
                    let gn = g[i] * a_decay + scratch[i];
                    g[i] = gn;
                    i_in[i] += gn * (erev - v[i]);
                }
            }
            KinKind::ExpTwo {
                a_rise,
                a_decay,
                norm,
            } => {
                let g = self.g.as_mut_slice();
                let x = self.x.as_mut_slice();
                for i in 0..n {
                    let gn = g[i] * a_decay + scratch[i];
                    let xn = x[i] * a_rise + scratch[i];
                    g[i] = gn;
                    x[i] = xn;
                    i_in[i] += norm * (gn - xn) * (erev - v[i]);
                }
            }
            KinKind::Alpha { r, e_r_dt_tau } => {
                let g = self.g.as_mut_slice();
                let a = self.a.as_mut_slice();
                for i in 0..n {
                    let a_old = a[i];
                    let an = a_old * r + scratch[i];
                    let gn = g[i] * r + e_r_dt_tau * a_old;
                    g[i] = gn;
                    a[i] = an;
                    i_in[i] += gn * (erev - v[i]);
                }
            }
            KinKind::ExpThree {
                a_rise,
                a_decay1,
                a_decay2,
                wf1,
                wf2,
                k_a,
                gbase1,
                gbase2,
            } => {
                let a = self.a.as_mut_slice();
                let b = self.b.as_mut_slice();
                let c = self.c.as_mut_slice();
                for i in 0..n {
                    let j = scratch[i];
                    let an = a[i] * a_rise + j * k_a;
                    let bn = b[i] * a_decay1 + j * wf1;
                    let cn = c[i] * a_decay2 + j * wf2;
                    a[i] = an;
                    b[i] = bn;
                    c[i] = cn;
                    let cur = gbase1 * (bn - an) + gbase2 * (cn - an);
                    i_in[i] += cur * (erev - v[i]);
                }
            }
        }
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
    fn setup_lif_params(pop: &mut crate::api::NeuralPopulation<LIFModel>) {
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
        assert_eq!(
            spikes_b.iter().sum::<i32>(),
            0,
            "No spikes without external drive"
        );
    }

    /// A second `build()` drains the (now-empty) edge list, so it would rebuild
    /// the graph with no connections. It must be rejected, not silently corrupt.
    #[test]
    #[should_panic(expected = "build() called twice")]
    fn test_network_build_twice_panics() {
        let mut net = Network::new();
        let a = net.add_population::<LIFModel>(2);
        let b = net.add_population::<LIFModel>(2);
        net.connect(a, &[0], b, &[0], &[1.0], &[0]);
        net.build();
        net.build();
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

        // a has 2 neurons at global indices 0,1. b has 2 at 2,3.
        // a[0] = global 0, b[0] = global 2
        // The graph has one edge: pre=0 → post=2, weight=10.0, delay=3
        // After step 0: slot[0+3=3][2] += 10.0  (but we need a spike first)
        // Since neurons don't spike without current, this test verifies
        // non-crash behavior only.

        net.step();
        // Step counter = 1, no spikes expected

        // Manually verify: slot 3 should be all zeros (no spike happened)
        let slot3 = net.delay.slot(3 % MAX_DELAY_SLOTS);
        for (i, &val) in slot3.iter().enumerate() {
            assert!(
                val.abs() < 1e-10,
                "delay slot 3 [{i}] should be 0, got {val}"
            );
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
        assert!(
            total_spikes > 0,
            "LIF with I=30 must spike, got {total_spikes}"
        );
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
    #[should_panic(expected = "must be finite")]
    fn test_network_rejects_non_finite_weight() {
        let mut net = Network::new();
        let a = net.add_population::<LIFModel>(1);
        let b = net.add_population::<LIFModel>(1);
        net.connect(a, &[0], b, &[0], &[f32::NAN], &[0]);
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn test_network_rejects_non_finite_current() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(1);
        net.build();
        net.inject_current(pop, &[0], &[f32::INFINITY]);
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

    /// expOneSynapse: after a unit jump, the injected current must decay as
    /// `e^(−t/τ)` — peak immediately, then exponential tail (not a delta).
    #[test]
    fn kinetic_exp_one_decays_exponentially() {
        let tau = 10.0f32;
        let mut rt = KineticRuntime::new(SynapseKinetics::exp_decay(tau), 1.0);
        rt.allocate(1);
        let mut delay = DelayBuffer::new(1);
        let mut scratch = vec![0.0f32; 1];
        let mut i_in = vec![0.0f32; 1];
        // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
        let v = vec![-1.0f32; 1];

        // One arriving spike (weight = 1.0) in slot 0.
        delay.add(0, 0, 1.0);
        let a = (-1.0f32 / tau).exp();
        for step in 0..5 {
            i_in[0] = 0.0;
            rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
            let expect = a.powi(step);
            assert!(
                (i_in[0] - expect).abs() < 1e-5,
                "step {step}: {} vs {expect}",
                i_in[0]
            );
        }
    }

    /// expTwoSynapse: the double-exponential starts at 0, rises to a
    /// normalised peak near t_peak, then decays — and the peak ≈ the jump.
    #[test]
    fn kinetic_exp_two_rises_then_decays() {
        let k = SynapseKinetics::exp_rise_decay(1.0, 3.0);
        let mut rt = KineticRuntime::new(k, 0.1);
        rt.allocate(1);
        let mut delay = DelayBuffer::new(1);
        let mut scratch = vec![0.0f32; 1];
        let mut i_in = vec![0.0f32; 1];
        // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
        let v = vec![-1.0f32; 1];

        delay.add(0, 0, 1.0);
        let mut trace = Vec::with_capacity(150);
        for _ in 0..150 {
            i_in[0] = 0.0;
            rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
            trace.push(i_in[0]);
        }

        // Starts at 0 (rise phase), peaks ≈ 1.0 (normalised), then decays.
        assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
        let peak = trace.iter().cloned().fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.05, "peak ≈ 1.0: {peak}");
        // Peak near t_peak = 1·3/(3−1)·ln(3) ≈ 1.648 ms → index ≈ 16.
        let peak_i = trace
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            ((peak_i as f32) * 0.1 - 1.648).abs() < 0.25,
            "peak near t_peak: index {peak_i}"
        );
        // Monotone rise to the peak, decay after, tail ≈ 0 (≈5×τ_decay).
        assert!(trace[5] < trace[peak_i]);
        assert!(trace[40] < trace[peak_i]);
        assert!(trace[149].abs() < 0.05, "tail decays: {}", trace[149]);
    }

    /// Conductance synapses respect the driving force `erev − V`: zero at the
    /// reversal potential (shunting), depolarising below it, hyperpolarising
    /// above it.
    #[test]
    fn kinetic_conductance_respects_driving_force() {
        let k = SynapseKinetics::exp_decay(10.0).with_erev(-65.0);
        let mut rt = KineticRuntime::new(k, 1.0);
        rt.allocate(3);
        let mut delay = DelayBuffer::new(3);
        let mut scratch = vec![0.0f32; 3];
        let mut i_in = vec![0.0f32; 3];
        // One unit jump to all three neurons.
        delay.add(0, 0, 1.0);
        delay.add(0, 1, 1.0);
        delay.add(0, 2, 1.0);
        let v = vec![-65.0f32, -70.0, -60.0];
        rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 3);
        // At V = erev: pure shunt, no current.
        assert!(i_in[0].abs() < 1e-6, "shunt at erev: {}", i_in[0]);
        // V below erev → positive (depolarising toward erev).
        assert!((i_in[1] - 5.0).abs() < 1e-6, "depolarising: {}", i_in[1]);
        // V above erev → negative (hyperpolarising toward erev).
        assert!((i_in[2] + 5.0).abs() < 1e-6, "hyperpolarising: {}", i_in[2]);
    }

    /// alphaSynapse: the alpha function `g(t) = A·(t/τ)·e^(1−t/τ)` starts at 0,
    /// peaks at `A` near `t = τ`, and matches the closed form at every step
    /// (exponential Euler is exact for this linear ODE).
    #[test]
    fn kinetic_alpha_peaks_at_tau() {
        let tau = 3.0f32;
        let dt = 0.1f32;
        let mut rt = KineticRuntime::new(SynapseKinetics::alpha(tau), dt);
        rt.allocate(1);
        let mut delay = DelayBuffer::new(1);
        let mut scratch = vec![0.0f32; 1];
        let mut i_in = vec![0.0f32; 1];
        // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
        let v = vec![-1.0f32; 1];

        delay.add(0, 0, 1.0); // unit jump A = 1
        let mut trace = Vec::with_capacity(150);
        for _ in 0..150 {
            i_in[0] = 0.0;
            rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
            trace.push(i_in[0]);
        }

        // g(t) = A·(t/τ)·e^(1−t/τ) = (t/τ)·e·e^(−t/τ) at every sampled step.
        for (step, &got) in trace.iter().enumerate().take(20) {
            let t = step as f32 * dt;
            let expect = (t / tau) * std::f32::consts::E * (-t / tau).exp();
            assert!(
                (got - expect).abs() < 1e-5,
                "step {step}: {got} vs {expect}"
            );
        }

        // Starts at 0, peaks ≈ the jump (1.0) at t ≈ tau = 3 ms.
        assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
        let (peak_i, peak) =
            trace
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |(bi, bv), (i, &x)| {
                        if x > bv {
                            (i, x)
                        } else {
                            (bi, bv)
                        }
                    },
                );
        assert!((peak - 1.0).abs() < 0.05, "peak ≈ 1.0: {peak}");
        assert!(((peak_i as f32) * dt - tau).abs() < 0.5, "peak near t=tau");
        assert!(trace[149].abs() < 0.1, "tail decays: {}", trace[149]);
    }

    /// expThreeSynapse: the two-component conductance matches its closed form
    /// `gbase1·(B−A) + gbase2·(C−A)` at every step and starts at 0 (the shared
    /// rise term cancels the initial per-component jumps).
    #[test]
    fn kinetic_exp_three_matches_closed_form() {
        let (tr, td1, td2, g1, g2) = (1.0f32, 3.0f32, 10.0f32, 1.0f32, 0.5f32);
        let dt = 0.1f32;
        let mut rt = KineticRuntime::new(SynapseKinetics::exp_three(tr, td1, td2, g1, g2), dt);
        rt.allocate(1);
        let mut delay = DelayBuffer::new(1);
        let mut scratch = vec![0.0f32; 1];
        let mut i_in = vec![0.0f32; 1];
        let v = vec![-1.0f32; 1];

        delay.add(0, 0, 1.0);
        let wf1 = exp_waveform_factor(tr, td1);
        let wf2 = exp_waveform_factor(tr, td2);
        let k_a = (g1 * wf1 + g2 * wf2) / (g1 + g2);

        // 300 steps = 30 ms ≈ 3× the slow decay (τ2 = 10 ms) so the tail has
        // fully relaxed.
        let mut trace = Vec::with_capacity(300);
        for _ in 0..300 {
            i_in[0] = 0.0;
            rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
            trace.push(i_in[0]);
        }

        for (step, &got) in trace.iter().enumerate() {
            let t = step as f32 * dt;
            let expect = g1 * (wf1 * (-t / td1).exp() - k_a * (-t / tr).exp())
                + g2 * (wf2 * (-t / td2).exp() - k_a * (-t / tr).exp());
            assert!(
                (got - expect).abs() < 1e-5,
                "step {step}: {got} vs {expect}"
            );
        }
        assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
        assert!(trace.iter().all(|x| x.is_finite() && *x >= -1e-6));
        assert!(trace[299].abs() < 0.1, "tail decays: {}", trace[299]);
    }

    /// expThreeSynapse with a zero second component must reduce exactly to
    /// `expTwoSynapse` with `gbase = gbase1` (the ring carries the connection
    /// weight for expThree, and `weight·gbase` for expTwo).
    #[test]
    fn kinetic_exp_three_zero_second_component_matches_exp_two() {
        let (tr, td) = (1.0f32, 3.0f32);
        let gbase = 2.0f32;
        let dt = 0.1f32;
        let mut a = KineticRuntime::new(SynapseKinetics::exp_three(tr, td, 10.0, gbase, 0.0), dt);
        let mut b = KineticRuntime::new(SynapseKinetics::exp_rise_decay(tr, td), dt);
        a.allocate(1);
        b.allocate(1);
        let mut da = DelayBuffer::new(1);
        let mut db = DelayBuffer::new(1);
        let mut sa = vec![0.0f32; 1];
        let mut sb = vec![0.0f32; 1];
        let mut ia = vec![0.0f32; 1];
        let mut ib = vec![0.0f32; 1];
        let v = vec![-1.0f32; 1];

        da.add(0, 0, 1.0); // expThree: connection weight
        db.add(0, 0, gbase); // expTwo: weight · gbase
        for _ in 0..100 {
            ia[0] = 0.0;
            ib[0] = 0.0;
            a.step(&mut da, 0, &mut sa, &v, &mut ia, 1);
            b.step(&mut db, 0, &mut sb, &v, &mut ib, 1);
            assert!((ia[0] - ib[0]).abs() < 1e-5, "{} vs {}", ia[0], ib[0]);
        }
    }

    /// Registering an invalid profile must be rejected before build.
    #[test]
    fn register_kinetics_rejects_invalid_profile() {
        let mut net = Network::new();
        assert!(net
            .register_kinetics(SynapseKinetics::exp_decay(0.0), 1.0)
            .is_err());
        assert!(net
            .register_kinetics(SynapseKinetics::exp_rise_decay(3.0, 1.0), 1.0)
            .is_err());
    }

    /// A Poisson source at `p = rate·dt = 1` fires every step and routes its
    /// weight through the delay ring to the post neuron (like a presynaptic
    /// spike); at 0 Hz it never fires, leaving the neuron at rest.
    #[test]
    fn poisson_source_routes_spikes_through_synapses() {
        // rate=2000 Hz, dt=0.5 ms → p=1.0 → fires every step.
        let mut net = Network::new();
        let post = net.add_population::<LIFModel>(1);
        setup_lif_params(net.pop_typed_mut::<LIFModel>(post).unwrap());
        let src = net.add_poisson_source(2000.0, 0.5, 42);
        net.connect_poisson_source(src, post, &[0], &[30.0], &[0], &[0], &[0]);
        net.build();

        let mut fired = 0i32;
        for _ in 0..200 {
            net.step();
            fired += net.spikes(post)[0];
        }
        assert!(
            fired > 0,
            "Poisson drive (p=1) must make the LIF fire, got {fired}"
        );

        // rate=0 → never fires → the neuron stays silent.
        let mut net0 = Network::new();
        let post0 = net0.add_population::<LIFModel>(1);
        setup_lif_params(net0.pop_typed_mut::<LIFModel>(post0).unwrap());
        let src0 = net0.add_poisson_source(0.0, 0.5, 42);
        net0.connect_poisson_source(src0, post0, &[0], &[30.0], &[0], &[0], &[0]);
        net0.build();
        let mut fired0 = 0i32;
        for _ in 0..200 {
            net0.step();
            fired0 += net0.spikes(post0)[0];
        }
        assert_eq!(fired0, 0, "0 Hz Poisson must never fire");
    }

    /// Adding a Poisson source or connecting one after `build()` is rejected.
    #[test]
    #[should_panic(expected = "before build")]
    fn poisson_source_after_build_panics() {
        let mut net = Network::new();
        net.add_population::<LIFModel>(1);
        net.build();
        net.add_poisson_source(100.0, 0.5, 0);
    }

    /// Tsodyks–Markram depression end-to-end: pre (driven by strong external
    /// current) fires repeatedly through a static edge into post 1 and through
    /// a plastic edge (U0 = 0.5, τ_rec = 10 ms) into post 2. The plastic edge
    /// delivers R·U of every spike — each pre-spike depletes R by (1 − U0), so
    /// the sustained drive reaches post 2 only weakly and it stays silent,
    /// while post 1 keeps firing at full weight.
    #[test]
    fn plastic_synapse_depresses_repeated_drive() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(3); // pre + 2 posts
        let lif = net.pop_typed_mut::<LIFModel>(pop).unwrap();
        setup_lif_params(lif);
        // Short refractory + near-rest threshold → pre fires every ~3 steps,
        // giving the static post a clear drive margin and the depressed post a
        // clear silence margin (steady-state delivered ~11 vs full 100).
        lif.fill_param(5, 1.0); // tau_ref
        lif.fill_param(3, -65.0); // v_thresh
        let plast = net
            .register_plasticity(PlasticityProfile::depression(0.5, 10.0), 0.5)
            .unwrap();
        assert_eq!(plast, 1, "first profile gets id 1");
        // pre(0) → post(1) static; pre(0) → post(2) plastic.
        net.connect_edges(
            pop,
            pop,
            &[
                Edge::new(0, 1, 100.0, 0),
                Edge::with_synapse_plasticity(0, 2, 100.0, 0, 0, plast),
            ],
        );
        net.build();
        assert_eq!(net.plast_types(), &[0, 1], "edge 1 carries the TM profile");

        let mut fired_static = 0i32;
        let mut fired_plastic = 0i32;
        for _ in 0..500 {
            net.inject_current(pop, &[0], &[100.0]);
            net.step();
            fired_static += net.spikes(pop)[1];
            fired_plastic += net.spikes(pop)[2];
        }
        assert!(
            fired_static > 20,
            "static edge keeps driving post 1: {fired_static}"
        );
        assert_eq!(
            fired_plastic, 0,
            "depressed edge must silence post 2 (got {fired_plastic})"
        );
    }

    /// Registering an invalid plasticity profile is rejected before build.
    #[test]
    fn register_plasticity_rejects_invalid_profile() {
        let mut net = Network::new();
        assert!(net
            .register_plasticity(PlasticityProfile::depression(0.0, 1.0), 0.5)
            .is_err());
        assert!(net
            .register_plasticity(PlasticityProfile::depression(1.5, 1.0), 0.5)
            .is_err());
        assert!(net
            .register_plasticity(PlasticityProfile::depression(0.5, 0.0), 0.5)
            .is_err());
        assert!(net
            .register_plasticity(PlasticityProfile::depression(0.5, f32::NAN), 0.5)
            .is_err());
    }

    /// A gap junction is a symmetric ohmic coupling: neuron 0 (depolarised) and
    /// neuron 1 (at rest) exchange `g·ΔV`, pulling them toward each other.
    #[test]
    fn gap_junction_couples_two_neurons() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());
        // Neuron 0 depolarised to -55; neuron 1 stays at rest -70.
        net.pop_typed_mut::<LIFModel>(pop)
            .unwrap()
            .state_slice_mut(0)[0] = -55.0;
        net.connect_gap_junction(pop, 0, pop, 1, 0.5);
        net.build();

        net.step();
        let v0 = net.state(pop, 0)[0];
        let v1 = net.state(pop, 0)[1];
        // Neuron 1 receives g·(V0 − V1) > 0 → rises above rest.
        assert!(v1 > -70.0, "coupled neuron should rise, got {v1}");
        // Neuron 0 receives g·(V1 − V0) < 0 → falls below its start.
        assert!(v0 < -55.0, "driven neuron should fall, got {v0}");
    }

    /// Gap-junction conductance must be finite and non-negative at connect time.
    #[test]
    #[should_panic(expected = "conductance must be finite")]
    fn gap_junction_rejects_non_finite_conductance() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        net.connect_gap_junction(pop, 0, pop, 1, f32::NAN);
    }

    /// A graded (analog) synapse is a **one-way** ohmic coupling: the post
    /// neuron receives `g·(V_pre − V_post)`, the pre neuron is unaffected (only
    /// its leak acts on it).
    #[test]
    fn graded_synapse_is_one_way_coupling() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());
        net.pop_typed_mut::<LIFModel>(pop)
            .unwrap()
            .state_slice_mut(0)[0] = -55.0;
        net.connect_graded_synapse(pop, 0, pop, 1, 0.5);
        net.build();

        net.step();
        let v0 = net.state(pop, 0)[0];
        let v1 = net.state(pop, 0)[1];
        // Post neuron receives g·(V0 − V1) = 7.5 → rises above rest.
        assert!(v1 > -70.0, "post neuron should rise, got {v1}");
        // Pre neuron only relaxes toward rest (leak), NOT pulled by coupling:
        // leak-only dv = (v_rest − v)·inv_tau·dt = (−15)·0.05 = −0.75.
        assert!(
            (v0 - (-55.75)).abs() < 1e-3,
            "pre neuron must be unaffected by the graded coupling, got {v0}"
        );
    }

    /// Graded-synapse conductance must be finite and non-negative at connect time.
    #[test]
    #[should_panic(expected = "conductance must be finite")]
    fn graded_synapse_rejects_non_finite_conductance() {
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        net.connect_graded_synapse(pop, 0, pop, 1, f32::NAN);
    }

    /// The sigmoid `gradedSynapse` activation follows V_pre: `s` rises toward
    /// `inf`, and the post neuron receives `weight·g·s·(erev − V_post)`.
    #[test]
    fn sigmoid_graded_runtime_activates_and_drives_current() {
        let spec = SigmoidGradedSynapse::new(-50.0, 5.0, 0.5, 0.0, 1.0);
        let mut rt = SigmoidGradedRuntime::new(spec, 0.5);
        rt.conns.push((0, 1, 1.0, 0.0)); // pre=0, post=1, weight=1, s=0
        let v = vec![-40.0f32, -70.0];
        let mut i_in = vec![0.0f32; 2];
        rt.step(&v, &mut i_in);
        // arg = (−40 −(−50))/5 = 2 → inf ≈ 0.8808, tau ≈ 0.2384 ms.
        // s = inf·(1 − e^(−0.5/0.2384)) ≈ 0.8808·0.8772 ≈ 0.7726.
        // i = 1·1·0.7726·(0 −(−70)) ≈ 54.08.
        assert!(
            i_in[1] > 40.0 && i_in[1] < 65.0,
            "driving current: {}",
            i_in[1]
        );
        assert_eq!(i_in[0], 0.0, "pre receives nothing (one-way)");
        let s = rt.conns[0].3;
        assert!(s > 0.0 && s < 0.8808, "s rises toward inf: {s}");
    }

    /// Invalid sigmoid graded profiles (`delta == 0`, `k <= 0`) are rejected.
    #[test]
    fn register_sigmoid_graded_rejects_invalid() {
        let mut net = Network::new();
        assert!(net
            .register_sigmoid_graded(SigmoidGradedSynapse::new(-50.0, 0.0, 0.5, 0.0, 1.0), 0.5)
            .is_err()); // delta = 0
        assert!(net
            .register_sigmoid_graded(SigmoidGradedSynapse::new(-50.0, 5.0, 0.0, 0.0, 1.0), 0.5)
            .is_err()); // k = 0
    }
}
