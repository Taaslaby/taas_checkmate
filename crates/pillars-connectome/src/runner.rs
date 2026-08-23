//! # BrainRunner — Universal Brain Simulation Engine
//!
//! A single unified Rust struct that replaces the multi-file Python simulation loop.
//! All neuron models, learning rules, and stimulation types in one place.
//!
//! ## Why BrainRunner?
//!
//! **Python `TaasNetwork.run()`** calls `step_lif()` + `propagate_spikes()` 50× per
//! decision via PyO3 → **200ms overhead**. BrainRunner calls the entire loop in one
//! Rust function → **~26ms** (7.7× faster for 138K FlyWire neurons).
//!
//! ## Supported Models
//!
//! | Model | Variant | Learning | Conductance |
//! |-------|---------|----------|-------------|
//! | LIF   | `Lif`, `LifSynaptic` | — | optional |
//! | Izhikevich | `Izh`, `IzhSynaptic` | E-Prop, Hebbian | optional |
//! | HH (Hodgkin-Huxley) | `Hh`, `HhSynaptic` | — | optional |
//! | Rate | `Rate` | — | — |
//! | GCO | `Gco` | — | — |
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use pillars_connectome::{ConnectomeData, BrainRunner};
//!
//! // 1. Create connectome from CSR arrays
//! let mut conn = ConnectomeData::new(row_ptr, cols, vals, nt_types, groups);
//!
//! // 2. Build runner with desired neuron model
//! let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5, 2.0));
//!
//! // 3. Add Poisson stimulation (e.g., ascending neurons in fly)
//! runner.add_poisson_group(&ascending_indices, 150.0, 10.0);
//!
//! // 4. Run 50 steps in ONE Rust call (was 100 Python→Rust calls)
//! let spike_buf = runner.run(50, &external_I);
//!
//! // 5. Read spike trains per neuron
//! let trains = runner.spike_trains(0.5);  // {neuron: [time_ms, ...]}
//!
//! // 6. Get firing rates for a group
//! let dn_rates = runner.firing_rates_group(&dn_indices, 50, 0.5);
//! ```
//!
//! ## Python
//!
//! ```python
//! runner = tcm.BrainRunner(connectome, model="lif", dt_ms=0.5)
//! runner.add_poisson_group(indices, rate=150.0, weight=10.0)
//! spike_buf = runner.run(n_steps=50, external_I=ext_I)
//! ```

use std::collections::HashMap;

use crate::data::ConnectomeData;
use crate::propagate::*;
use crate::types::*;
use pillars::kernels::csr::ValidatedCsrGraph;
use pillars::kernels::synapse::{DecayFactors, SynapticKinetics};
use rand::{Rng, SeedableRng};

use crate::sim::{haap_population, hh_population_pacemaking, hh_population_standard};
use pillars::core::arch::SimdType;
use pillars::core::lane::SimdLane;
use pillars::core::NeuralPopulation;
use pillars::zoo::{
    AlphaLifModel, AlphaLifModelParamIdx, AlphaLifModelStateIdx, HAAPModel, HAAPModelParamIdx,
    HHModel, HHModelParamIdx,
};

/// Build the alpha-LIF SIMD population (Shiu fly brain model).
///
/// Maps the config scalars to the 7 per-neuron params of [`AlphaLifModel`]:
/// `[dt, time_factor, v_rest, v_thresh, v_reset, refr_steps, syn_decay]` and
/// initialises state `[v=v_rest, g=0, refr_cnt=refr_steps]` (start out of
/// refractory so the first step accepts input, matching the reference model).
pub(crate) fn alpha_lif_population(
    n: usize,
    dt_ms: f32,
    tau_ref: f32,
    tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_syn: f32,
) -> NeuralPopulation<AlphaLifModel> {
    let mut pop = NeuralPopulation::<AlphaLifModel>::new(n);
    let time_factor = dt_ms / tau_m.max(1e-6);
    let syn_decay = 1.0 - dt_ms / tau_syn.max(1e-6);
    let refr_steps = (tau_ref / dt_ms).round().max(1.0);
    pop.fill_param(AlphaLifModelParamIdx::dt.into(), dt_ms);
    pop.fill_param(AlphaLifModelParamIdx::time_factor.into(), time_factor);
    pop.fill_param(AlphaLifModelParamIdx::v_rest.into(), v_rest);
    pop.fill_param(AlphaLifModelParamIdx::v_thresh.into(), v_thresh);
    pop.fill_param(AlphaLifModelParamIdx::v_reset.into(), v_reset);
    pop.fill_param(AlphaLifModelParamIdx::refr_steps.into(), refr_steps);
    pop.fill_param(AlphaLifModelParamIdx::syn_decay.into(), syn_decay);
    pop.fill_state(AlphaLifModelStateIdx::v.into(), v_rest);
    pop.fill_state(AlphaLifModelStateIdx::g.into(), 0.0);
    pop.fill_state(AlphaLifModelStateIdx::refr_cnt.into(), refr_steps);
    pop
}

// ═══════════════════════════════════════════════════════════
//  NeuronConfig — all supported neuron models
// ═══════════════════════════════════════════════════════════

/// Configuration for a specific neuron model with all its parameters.
///
/// Each variant holds the minimal set of parameters needed for its model.
/// Use [`BrainConfig`] helpers to construct common configurations.
#[derive(Debug, Clone)]
pub enum NeuronConfig {
    /// Leaky Integrate-and-Fire (current-based propagation)
    Lif {
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
    },
    /// LIF with conductance-based synaptic kinetics (requires receptor_types)
    LifSynaptic {
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
    },
    /// LIF + alpha-synapse (Shiu et al. 2024 fly brain model).
    ///
    /// Each pre-spike *jumps* a per-post-synaptic conductance `g` by the edge
    /// weight (mV), `g` decays with τ_syn, and the resulting `g` is added
    /// directly to `v` — the exact model of `Drosophila_brain_model`
    /// (brian2 default_params / fly-brain-main run_pytorch.py).
    LifAlpha {
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_syn: f32,
        t_delay: f32,
    },
    /// Izhikevich model (current-based)
    Izh {
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    },
    /// Izhikevich with conductance-based synapses
    IzhSynaptic {
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    },
    /// Izhikevich + slow memory + Hebbian plasticity
    IzhDual {
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        tau_m: f32,
        w_m: f32,
        hebb_lr: f32,
        hebb_decay: f32,
    },
    /// Izhikevich + E-Prop eligibility traces
    IzhEprop {
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        tau_e: f32,
        tau_surrogate: f32,
    },
    /// Hodgkin-Huxley (current-based, polynomial rates)
    Hh { dt_ms: f32, use_pacemaking: bool },
    /// Hodgkin-Huxley with conductance-based synapses
    HhSynaptic { dt_ms: f32, use_pacemaking: bool },
    /// Rate model (continuous firing rates)
    Rate { dt_ms: f32, tau: f32 },
    /// Graph-coupled oscillator (ψ-wave dynamics)
    Gco { dt_ms: f32 },
    /// HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting) — current-based.
    /// M-current (`u`) adaptation + h-current (`h`) rebound bound the firing rate
    /// (unlike HH, which has no adaptation and can run away).
    Haap { dt_ms: f32 },
    /// HAAP with a decaying synaptic current (τ_syn). A pre-spike adds `w` to a
    /// persistent per-neuron synaptic current that decays with τ_syn, instead of
    /// a single one-step pulse. Required for rebound half-center oscillators:
    /// an inhibitory GABAergic input must hyperpolarize the target for several
    /// ms (τ_syn ~ 5–10 ms) so the h-current can de-inactivate and rebound.
    HaapSynaptic { dt_ms: f32, tau_syn: f32 },
}

// ═══════════════════════════════════════════════════════════
//  BrainConfig — convenience constructors
// ═══════════════════════════════════════════════════════════

/// Convenience constructors for common brain configurations.
pub struct BrainConfig;

impl BrainConfig {
    /// Default LIF (FlyWire / NeuroFly parameters).
    /// τ_m=20ms, v_rest=-70mV, v_thresh=-58mV, v_reset=-70mV, τ_ref=2ms
    pub fn lif(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Lif {
            dt_ms,
            tau_ref: 2.0,
            tau_m: 20.0,
            v_rest: -70.0,
            v_thresh: -58.0,
            v_reset: -70.0,
        }
    }

    /// LIF with conductance-based synapses.
    pub fn lif_synaptic(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::LifSynaptic {
            dt_ms,
            tau_ref: 2.0,
            tau_m: 20.0,
            v_rest: -70.0,
            v_thresh: -58.0,
            v_reset: -70.0,
        }
    }

    /// LIF + alpha-synapse — exact Shiu et al. 2024 fly brain model
    /// (brian2 default_params / fly-brain-main run_pytorch.py MODEL_PARAMS).
    ///
    /// v_rest=v_reset=-52mV, v_thresh=-45mV, τ_mem=20ms, τ_ref=2.2ms,
    /// τ_syn=5ms, axonal delay 1.8ms. Weights in mV (already ×0.275 in .taas).
    pub fn lif_alpha(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::LifAlpha {
            dt_ms,
            tau_ref: 2.2,
            tau_m: 20.0,
            v_rest: -52.0,
            v_thresh: -45.0,
            v_reset: -52.0,
            tau_syn: 5.0,
            t_delay: 1.8,
        }
    }

    /// Default Izhikevich (regular spiking).
    pub fn izh(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Izh {
            dt_ms,
            tau_ref: 2.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }

    /// Izhikevich + Hebbian dual memory.
    pub fn izh_dual(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::IzhDual {
            dt_ms,
            tau_ref: 2.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            tau_m: 0.999,
            w_m: 5.0,
            hebb_lr: 0.001,
            hebb_decay: 0.0001,
        }
    }

    /// Izhikevich + E-Prop learning.
    pub fn izh_eprop(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::IzhEprop {
            dt_ms,
            tau_ref: 2.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            tau_e: 20.0,
            tau_surrogate: 2.0,
        }
    }

    /// Default Hodgkin-Huxley (NaV + KV channels).
    pub fn hh(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Hh {
            dt_ms,
            use_pacemaking: false,
        }
    }

    /// HH with pacemaking channels (NaV + KV + CaT, for oscillatory neurons).
    pub fn hh_pacemaking(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Hh {
            dt_ms,
            use_pacemaking: true,
        }
    }

    /// HH with conductance-based synapses.
    pub fn hh_synaptic(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::HhSynaptic {
            dt_ms,
            use_pacemaking: false,
        }
    }

    /// Rate model.
    pub fn rate(dt_ms: f32, tau: f32) -> NeuronConfig {
        NeuronConfig::Rate { dt_ms, tau }
    }

    /// Graph-coupled oscillator.
    pub fn gco(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Gco { dt_ms }
    }

    /// HAAP (M-current adaptation + h-current rebound), current-based.
    pub fn haap(dt_ms: f32) -> NeuronConfig {
        NeuronConfig::Haap { dt_ms }
    }

    /// HAAP with a decaying synaptic current (τ_syn). Use for rebound
    /// half-center / CPG circuits where inhibition must persist across steps.
    pub fn haap_synaptic(dt_ms: f32, tau_syn: f32) -> NeuronConfig {
        NeuronConfig::HaapSynaptic { dt_ms, tau_syn }
    }
}

// ═══════════════════════════════════════════════════════════
//  PoissonGroup — external stimulation
// ═══════════════════════════════════════════════════════════

/// A Poisson stimulation group: a set of neurons that receive stochastic input.
#[derive(Debug, Clone)]
pub struct PoissonGroup {
    /// Neuron indices that receive Poisson input.
    pub indices: Vec<usize>,
    /// Firing rate in Hz.
    pub rate_hz: f32,
    /// Current weight (injected per Poisson event).
    pub weight: f32,
}

/// Inject Poisson-group events into an arbitrary current buffer.
///
/// Shared by the HH runners (pillars `i_in` and legacy `cfg.Ii`) so the
/// stochastic logic lives in exactly one place. `p = rate·dt/1000` is the
/// per-step spike probability; if `p ≥ 1` every member fires every step.
fn inject_poisson_into(
    groups: &[PoissonGroup],
    rng: &mut rand::rngs::StdRng,
    buf: &mut [f32],
    dt_ms: f32,
) {
    for pg in groups {
        if pg.rate_hz <= 0.0 || pg.weight == 0.0 {
            continue;
        }
        let p = pg.rate_hz * dt_ms / 1000.0;
        if p <= 0.0 {
            continue;
        }
        // Optimization: if p ≥ 1, inject all neurons every step
        if p >= 1.0 {
            for &idx in &pg.indices {
                if idx < buf.len() {
                    buf[idx] += pg.weight;
                }
            }
        } else {
            for &idx in &pg.indices {
                if idx < buf.len() && rng.gen::<f32>() < p {
                    buf[idx] += pg.weight;
                }
            }
        }
    }
}

/// CSR-propagate the spike queue after an HH step.
///
/// Conductance models route through `csr_propagate_conductance_from_queue`
/// (jumps `g` on `kinetics`); current models add `W·nt_types` into `buf`.
fn csr_propagate_after_hh(
    connectome: &ConnectomeData,
    csr_graph: &ValidatedCsrGraph,
    queue: &mut SpikeQueue,
    kinetics: &mut Option<SynapticKinetics>,
    buf: &mut [f32],
    synaptic: bool,
) {
    if synaptic {
        csr_propagate_conductance_from_queue_with_topology(
            csr_graph.as_topology(),
            &connectome.values,
            &connectome.receptor_types,
            queue,
            kinetics.as_mut().unwrap(),
            1.0,
        );
    } else {
        csr_propagate_from_queue_with_topology(
            csr_graph.as_topology(),
            &connectome.values,
            &connectome.nt_types,
            queue,
            buf,
            1.0,
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  Learning config
// ═══════════════════════════════════════════════════════════

/// Configuration for online learning during simulation.
#[derive(Debug, Clone, Default)]
pub enum LearningRule {
    /// Hebbian plasticity (used with IzhDual).
    Hebbian { lr: f32, decay: f32 },
    /// E-Prop eligibility traces (used with IzhEprop).
    Eprop { tau_e: f32, surrogate_tau: f32 },
    /// No learning.
    #[default]
    None,
}

// ═══════════════════════════════════════════════════════════
//  BrainRunner — the main struct
// ═══════════════════════════════════════════════════════════

/// Universal brain runner — ONE Rust call replaces the slow Python loop.
///
/// Holds all state internally: connectome, neuron parameters, brain state,
/// Poisson groups, synaptic kinetics, and learning rules.
/// Call `run(n_steps, external_I)` to run the simulation.
///
/// # Lifecycle per `run()` call
///
/// For each of `n_steps`:
/// 1. Inject Poisson spikes into `I_buf`
/// 2. Add `external_I` to `I_buf`
/// 3. Step all neurons (SIMD: LIF/Izhikevich/HH/...)
/// 4. Push spikes to queue → record to spike buffer
/// 5. CSR-propagate spikes through connectome (current or conductance)
/// 6. Reset `I_buf`
///
/// State persists between `run()` calls — you can chain them for a
/// continuous simulation timeline.
pub struct BrainRunner {
    /// Neuron model configuration.
    pub config: NeuronConfig,
    /// Connectome data (CSR matrix + NT types + groups).
    pub connectome: ConnectomeData,
    /// Immutable validated CSR topology snapshot used by timestep hot paths.
    /// Rebuild with [`BrainRunner::refresh_csr_graph`] after mutating public CSR arrays.
    csr_graph: ValidatedCsrGraph,
    /// Brain state (v, u, I_buf, spikes, refractory, queue).
    pub state: BrainState,
    /// Poisson stimulation groups.
    pub poisson_groups: Vec<PoissonGroup>,
    // Internal buffers (persistent across calls to avoid per-run allocation)
    spike_buffer: Vec<i32>,
    /// Reusable work buffer (n_neurons f32) for Poisson injection + external I.
    work_buf: Vec<f32>,
    /// Reusable external current buffer (n_neurons f32).
    ext_buf: Vec<f32>,
    /// Precomputed [0, 1, ..., n_neurons-1] — passed to `brain.inject()`.
    all_indices: Vec<u32>,
    n_neurons: usize,
    step_counter: usize,

    // HH population (pillars SIMD engine) — standard or pacemaking
    // (NaV+KV+CaV-T) via NeuralPopulation<HHModel>.
    hh_pop: Option<NeuralPopulation<HHModel>>,

    // HAAP population (pillars SIMD engine) — M-current adaptation + h-current.
    haap_pop: Option<NeuralPopulation<HAAPModel>>,

    // Persistent decaying synaptic current for HAAP-synaptic (τ_syn). Lives
    // across `run()` calls so a chain of `run()` calls is one continuous timeline.
    haap_syn: Vec<f32>,

    // Synaptic kinetics (for conductance-based models)
    kinetics: Option<SynapticKinetics>,

    // ═══ Alpha-synapse state (Shiu et al. 2024 fly brain model) ═══
    // Per-neuron dynamics (v, g, refr) — pillars SIMD population (f32x8).
    alpha_pop: Option<NeuralPopulation<AlphaLifModel>>,
    // Ring buffer (steps_delay+1) × n — delayed axonal input (delivered as
    // ctx.i_syn to the SIMD population; refractory mask applied inside model).
    alpha_delay: Vec<f32>,
    // Write position into `alpha_delay` (read at (pos+1) % size).
    alpha_delay_pos: usize,
    // Compact list of pre-neurons that fired on the previous step (recurrent
    // input source). Avoids a full 138K-neuron scan per step in the scatter.
    alpha_fired: Vec<usize>,

    // E-Prop eligibility traces
    eligibility: Vec<f32>,

    // RNG for Poisson stimulation
    rng: rand::rngs::StdRng,
}

impl BrainRunner {
    /// Create a new BrainRunner with the given connectome and neuron config.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the connectome state is invalid (e.g., empty).
    pub fn new(connectome: ConnectomeData, config: NeuronConfig) -> Result<Self, String> {
        let n = connectome.n_neurons;
        if n == 0 {
            return Err("BrainRunner: connectome has 0 neurons".into());
        }
        if connectome.row_ptr.len() != n + 1 {
            return Err(format!(
                "BrainRunner: row_ptr length {} != n_neurons + 1 ({})",
                connectome.row_ptr.len(),
                n + 1
            ));
        }
        if connectome.col_indices.len() != connectome.values.len() {
            return Err(format!(
                "BrainRunner: col_indices length {} != values length {}",
                connectome.col_indices.len(),
                connectome.values.len()
            ));
        }
        let csr_graph =
            ValidatedCsrGraph::from_slices(&connectome.row_ptr, &connectome.col_indices, n);

        // Determine model enum from config
        let model = match &config {
            NeuronConfig::Lif { .. } | NeuronConfig::LifSynaptic { .. } => NeuronModel::LIF,
            NeuronConfig::Hh { .. } | NeuronConfig::HhSynaptic { .. } => NeuronModel::Hh,
            NeuronConfig::Izh { .. }
            | NeuronConfig::IzhSynaptic { .. }
            | NeuronConfig::IzhDual { .. }
            | NeuronConfig::IzhEprop { .. } => NeuronModel::Izhikevich,
            NeuronConfig::Rate { .. } => NeuronModel::Rate,
            NeuronConfig::Gco { .. } => NeuronModel::Gco,
            NeuronConfig::LifAlpha { .. } => NeuronModel::LIF,
            NeuronConfig::Haap { .. } | NeuronConfig::HaapSynaptic { .. } => NeuronModel::Haap,
        };

        let state = BrainState::new(n, model);
        // Initialize kinetics for synaptic models
        let kinetics = match &config {
            NeuronConfig::LifSynaptic { .. }
            | NeuronConfig::IzhSynaptic { .. }
            | NeuronConfig::HhSynaptic { .. } => Some(SynapticKinetics::new(n)),
            _ => None,
        };

        // Initialize HH engine: pillars NeuralPopulation for BOTH standard
        // and pacemaking (NaV+KV+CaV-T — CaT channel is now in pillars HHModel).
        let hh_pop = match &config {
            NeuronConfig::Hh {
                dt_ms,
                use_pacemaking,
            }
            | NeuronConfig::HhSynaptic {
                dt_ms,
                use_pacemaking,
            } => {
                if *use_pacemaking {
                    Some(hh_population_pacemaking(n, *dt_ms))
                } else {
                    Some(hh_population_standard(n, *dt_ms))
                }
            }
            _ => None,
        };

        // HAAP engine: pillars population (M-current + h-current).
        let haap_pop = match &config {
            NeuronConfig::Haap { dt_ms } | NeuronConfig::HaapSynaptic { dt_ms, .. } => {
                Some(haap_population(n, *dt_ms))
            }
            _ => None,
        };

        // Initialize E-Prop eligibility buffer
        let eligibility = match &config {
            NeuronConfig::IzhEprop { .. } => vec![0.0f32; connectome.values.len()],
            _ => Vec::new(),
        };

        // Alpha-synapse: SIMD population (v, g, refr) built from config;
        // delay ring buffer sized lazily in run_lif_alpha.
        let alpha_pop = match &config {
            NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                ..
            } => Some(alpha_lif_population(
                n, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, *tau_syn,
            )),
            _ => None,
        };
        let alpha_delay = Vec::new();
        let alpha_delay_pos = 0;
        let alpha_fired = Vec::new();

        let spike_buf_len = n * 64; // initial allocation (grows on demand)
        let rng = rand::rngs::StdRng::from_seed([0u8; 32]);
        let all_indices: Vec<u32> = (0..n as u32).collect();

        Ok(Self {
            config,
            connectome,
            csr_graph,
            state,
            poisson_groups: Vec::new(),
            spike_buffer: vec![0i32; spike_buf_len],
            work_buf: vec![0.0f32; n],
            ext_buf: vec![0.0f32; n],
            all_indices,
            n_neurons: n,
            step_counter: 0,
            hh_pop,
            haap_pop,
            haap_syn: vec![0.0f32; n],
            kinetics,
            eligibility,
            alpha_pop,
            alpha_delay,
            alpha_delay_pos,
            alpha_fired,
            rng,
        })
    }

    /// Get the learning rule extracted from the current config.
    pub fn learning_rule(&self) -> LearningRule {
        match &self.config {
            NeuronConfig::IzhDual {
                hebb_lr,
                hebb_decay,
                ..
            } if *hebb_lr > 0.0 => LearningRule::Hebbian {
                lr: *hebb_lr,
                decay: *hebb_decay,
            },
            NeuronConfig::IzhEprop {
                tau_e,
                tau_surrogate,
                ..
            } => LearningRule::Eprop {
                tau_e: *tau_e,
                surrogate_tau: *tau_surrogate,
            },
            _ => LearningRule::None,
        }
    }

    /// Add a Poisson stimulation group.
    ///
    /// Each step, every neuron in `indices` has `P = rate_hz × dt_ms / 1000`
    /// probability of receiving `weight` current injection.
    ///
    /// Multiple groups are independent (each neuron can receive from multiple groups).
    pub fn add_poisson_group(&mut self, indices: &[usize], rate_hz: f32, weight: f32) {
        let indices = indices
            .iter()
            .filter(|&&i| i < self.n_neurons)
            .copied()
            .collect();
        self.poisson_groups.push(PoissonGroup {
            indices,
            rate_hz,
            weight,
        });
    }

    /// Clear all Poisson groups.
    pub fn clear_poisson_groups(&mut self) {
        self.poisson_groups.clear();
    }

    /// Revalidate and replace the topology snapshot after mutating public CSR arrays.
    ///
    /// `ConnectomeData::values` may be changed freely by learning without calling
    /// this method; only `row_ptr` or `col_indices` mutations require a refresh.
    pub fn refresh_csr_graph(&mut self) {
        self.csr_graph = ValidatedCsrGraph::from_slices(
            &self.connectome.row_ptr,
            &self.connectome.col_indices,
            self.n_neurons,
        );
    }

    /// Borrow the validated topology for diagnostics or custom kernels.
    #[inline]
    pub fn csr_graph(&self) -> &ValidatedCsrGraph {
        &self.csr_graph
    }

    // ─── Run ────────────────────────────────────────────

    /// Run `n_steps` simulation steps.
    ///
    /// `external_I` — per-neuron background current `[n_neurons]` (zeros if empty).
    ///
    /// Returns the spike buffer as a flat `[n_steps × n_neurons]` array (0/1).
    /// State persists — call `run()` again to continue the simulation.
    pub fn run(&mut self, n_steps: usize, external_I: &[f32]) -> &[i32] {
        // Ensure spike buffer is large enough
        let needed = n_steps * self.n_neurons;
        if self.spike_buffer.len() < needed {
            self.spike_buffer.resize(needed, 0);
        }
        // `record_spikes` writes only 1s (never clears 0s), and this buffer is
        // reused across `run()` calls — clear the returned window so repeated
        // calls don't accumulate stale spikes from earlier chunks.
        self.spike_buffer[..needed].fill(0);

        match &self.config {
            NeuronConfig::Lif {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
            } => self.run_lif(
                n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, false,
            ),
            NeuronConfig::LifSynaptic {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
            } => self.run_lif(
                n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, true,
            ),
            NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                t_delay,
            } => {
                let _ = self.run_lif_alpha(
                    n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset,
                    *tau_syn, *t_delay, true,
                );
            }
            NeuronConfig::Izh {
                dt_ms,
                tau_ref,
                a,
                b,
                c,
                d,
            } => self.run_izhikevich(n_steps, external_I, *dt_ms, *tau_ref, *a, *b, *c, *d, false),
            NeuronConfig::IzhSynaptic {
                dt_ms,
                tau_ref,
                a,
                b,
                c,
                d,
            } => self.run_izhikevich(n_steps, external_I, *dt_ms, *tau_ref, *a, *b, *c, *d, true),
            NeuronConfig::IzhDual {
                dt_ms,
                tau_ref,
                a,
                b,
                c,
                d,
                tau_m,
                w_m,
                hebb_lr,
                hebb_decay,
            } => self.run_izhikevich_dual(
                n_steps,
                external_I,
                *dt_ms,
                *tau_ref,
                *a,
                *b,
                *c,
                *d,
                *tau_m,
                *w_m,
                *hebb_lr,
                *hebb_decay,
            ),
            NeuronConfig::IzhEprop {
                dt_ms,
                tau_ref,
                a,
                b,
                c,
                d,
                tau_e,
                tau_surrogate,
            } => self.run_izhikevich_eprop(
                n_steps,
                external_I,
                *dt_ms,
                *tau_ref,
                *a,
                *b,
                *c,
                *d,
                *tau_e,
                *tau_surrogate,
            ),
            NeuronConfig::Hh { dt_ms, .. } => self.run_hh(n_steps, external_I, *dt_ms, false),
            NeuronConfig::HhSynaptic { dt_ms, .. } => {
                self.run_hh(n_steps, external_I, *dt_ms, true)
            }
            NeuronConfig::Haap { dt_ms } => self.run_haap(n_steps, external_I, *dt_ms, 0.0),
            NeuronConfig::HaapSynaptic { dt_ms, tau_syn } => {
                self.run_haap(n_steps, external_I, *dt_ms, *tau_syn)
            }
            NeuronConfig::Rate { dt_ms, tau } => self.run_rate(n_steps, external_I, *dt_ms, *tau),
            NeuronConfig::Gco { dt_ms } => self.run_gco(n_steps, external_I, *dt_ms),
        }

        &self.spike_buffer[..needed]
    }

    /// Run without materializing the spike raster — count on the fly.
    ///
    /// Returns `(total_spikes, active_set)` over `n_steps`: `active_set` is a
    /// length-`n_neurons` bitmap of neurons that fired at least once.
    ///
    /// This is the hot path for [`crate::AlphaLifWorker`]: the raster path
    /// (`run`) zeroes `n_steps × n` i32 (~554 MB per 0.1 s chunk) and the
    /// worker then scans the same buffer — two full passes that metrics don't
    /// need. Counting here skips both, bit-identically (same neuron math).
    pub fn run_count(&mut self, n_steps: usize, external_I: &[f32]) -> (usize, Vec<bool>) {
        match &self.config {
            NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                t_delay,
            } => self.run_lif_alpha(
                n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset,
                *tau_syn, *t_delay, false,
            ),
            // Other models: fall back to the raster path and scan it. Only the
            // alpha worker uses run_count today; this keeps the API total.
            _ => {
                let n = self.n_neurons;
                let buf = self.run(n_steps, external_I);
                let mut total = 0usize;
                let mut active = vec![false; n];
                for (idx, &s) in buf.iter().enumerate() {
                    if s != 0 {
                        total += 1;
                        active[idx % n] = true;
                    }
                }
                (total, active)
            }
        }
    }

    // ─── Spike trains ──────────────────────────────────

    /// Build spike trains from the spike buffer.
    ///
    /// Returns `HashMap<neuron_idx, Vec<spike_time_ms>>` — the time (in ms)
    /// of each spike for each neuron that fired.
    ///
    /// `dt_ms` — the timestep used in `run()` (needed to convert step indices to ms).
    pub fn spike_trains(&self, dt_ms: f32) -> HashMap<usize, Vec<f32>> {
        let mut trains: HashMap<usize, Vec<f32>> = HashMap::new();
        let n_neurons = self.n_neurons;
        let n_steps = self.spike_buffer.len() / n_neurons;

        for step in 0..n_steps {
            let offset = step * n_neurons;
            let t = step as f32 * dt_ms;
            for i in 0..n_neurons {
                let idx = offset + i;
                if idx < self.spike_buffer.len() && self.spike_buffer[idx] != 0 {
                    trains.entry(i).or_default().push(t);
                }
            }
        }

        trains
    }

    /// Compute firing rates (Hz) for all neurons from the spike buffer.
    ///
    /// `dt_ms` — the timestep used in `run()`.
    /// Returns `[n_neurons]` firing rates in Hz.
    pub fn firing_rates(&self, dt_ms: f32) -> Vec<f32> {
        let n_neurons = self.n_neurons;
        let n_steps = self.spike_buffer.len() / n_neurons;
        let duration_s = n_steps as f32 * dt_ms / 1000.0;
        if duration_s <= 0.0 {
            return vec![0.0; n_neurons];
        }

        let mut counts = vec![0u32; n_neurons];
        for step in 0..n_steps {
            let offset = step * n_neurons;
            for i in 0..n_neurons {
                let idx = offset + i;
                if idx < self.spike_buffer.len() && self.spike_buffer[idx] != 0 {
                    counts[i] += 1;
                }
            }
        }

        counts.iter().map(|&c| c as f32 / duration_s).collect()
    }

    /// Compute firing rates for a named neuron group.
    ///
    /// Returns `[n_indices]` firing rates in Hz.
    pub fn firing_rates_group(&self, indices: &[usize], n_steps: usize, dt_ms: f32) -> Vec<f32> {
        let duration_s = n_steps as f32 * dt_ms / 1000.0;
        if duration_s <= 0.0 {
            return vec![0.0; indices.len()];
        }

        indices
            .iter()
            .map(|&idx| {
                let mut count = 0u32;
                for step in 0..n_steps {
                    let buf_idx = step * self.n_neurons + idx;
                    if buf_idx < self.spike_buffer.len() && self.spike_buffer[buf_idx] != 0 {
                        count += 1;
                    }
                }
                count as f32 / duration_s
            })
            .collect()
    }

    /// Get current membrane potentials (v).
    pub fn get_v(&self) -> &[f32] {
        if let Some(ref pop) = self.hh_pop {
            pop.state_slice(0)
        } else if let Some(ref pop) = self.haap_pop {
            pop.state_slice(0)
        } else if let Some(ref pop) = self.alpha_pop {
            pop.state_slice(AlphaLifModelStateIdx::v.into())
        } else {
            &self.state.v[..self.n_neurons]
        }
    }

    /// Set membrane potentials from a slice.
    pub fn set_v(&mut self, v: &[f32]) {
        let n = v.len().min(self.n_neurons);
        if let Some(ref mut pop) = self.hh_pop {
            // pillars state column 0 = membrane potential
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), pop.state_slice_mut(0).as_mut_ptr(), n);
            }
        } else if let Some(ref mut pop) = self.haap_pop {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), pop.state_slice_mut(0).as_mut_ptr(), n);
            }
        } else if let Some(ref mut pop) = self.alpha_pop {
            pop.state_slice_mut(AlphaLifModelStateIdx::v.into())[..n].copy_from_slice(&v[..n]);
        } else {
            self.state.v[..n].copy_from_slice(&v[..n]);
        }
    }

    /// Get current recovery variables (u) — relevant for Izhikevich.
    pub fn get_u(&self) -> &[f32] {
        &self.state.u[..self.n_neurons]
    }

    /// Set recovery variables.
    pub fn set_u(&mut self, u: &[f32]) {
        let n = u.len().min(self.n_neurons);
        self.state.u[..n].copy_from_slice(&u[..n]);
    }

    /// Set per-neuron HH parameter column (e.g. `HHModelParamIdx::g_Na`) from a slice.
    ///
    /// Enables literature-based per-class physiology: load a distinct
    /// g_Na / g_K / g_l / E_Leak / cm for each neuron instead of one
    /// population-wide default. Returns the number of neurons written.
    ///
    /// Only valid when the runner was created with an `Hh` / `HhSynaptic` config;
    /// otherwise it's a no-op returning 0.
    ///
    /// ⚠️ `reset()` re-creates the HH population from the config defaults and
    /// **wipes** any per-neuron params set here — re-apply after every `reset()`.
    pub fn set_hh_param_column(&mut self, col: usize, values: &[f32]) -> usize {
        let Some(pop) = self.hh_pop.as_mut() else {
            return 0;
        };
        let n = values.len().min(self.n_neurons);
        pop.param_slice_mut(col)[..n].copy_from_slice(&values[..n]);
        n
    }

    /// Set per-neuron HAAP parameter column (e.g. `HAAPModelParamIdx::g_m`) from a
    /// slice — enables literature-based per-class physiology (M-current/h-current
    /// atlas params) instead of one population-wide default. Returns the number
    /// of neurons written.
    ///
    /// Only valid when the runner was created with a `Haap` config; otherwise a
    /// no-op returning 0. ⚠️ `reset()` re-creates the HAAP population from the
    /// config defaults and **wipes** any per-neuron params — re-apply after reset.
    pub fn set_haap_param_column(&mut self, col: usize, values: &[f32]) -> usize {
        let Some(pop) = self.haap_pop.as_mut() else {
            return 0;
        };
        let n = values.len().min(self.n_neurons);
        pop.param_slice_mut(col)[..n].copy_from_slice(&values[..n]);
        n
    }

    /// Read one per-neuron HH parameter column (e.g. `HHModelParamIdx::g_Na`)
    /// — pairs with [`Self::set_hh_param_column`] for read-modify-write
    /// (e.g. pharmacology baselines). Empty for non-HH runners.
    pub fn get_hh_param_column(&self, col: usize) -> Vec<f32> {
        match &self.hh_pop {
            Some(pop) => pop.param_slice(col).to_vec(),
            None => Vec::new(),
        }
    }

    /// Read one per-neuron HAAP parameter column (e.g. `HAAPModelParamIdx::g_m`)
    /// — pairs with [`Self::set_haap_param_column`]. Empty for non-HAAP runners.
    pub fn get_haap_param_column(&self, col: usize) -> Vec<f32> {
        match &self.haap_pop {
            Some(pop) => pop.param_slice(col).to_vec(),
            None => Vec::new(),
        }
    }

    /// Read one live HAAP state column (`v`=0, `u`=1, `h`=2) — membrane
    /// diagnostics for GUIs/protocols. Empty for non-HAAP runners.
    pub fn get_haap_state_column(&self, col: usize) -> Vec<f32> {
        match &self.haap_pop {
            Some(pop) => pop.state_slice(col).to_vec(),
            None => Vec::new(),
        }
    }

    /// Write one HAAP state column (`v`=0, `u`=1, `h`=2) — e.g. initialize
    /// gates at a custom profile's resting equilibrium instead of the
    /// canonical defaults. Returns neurons written (0 for non-HAAP runners).
    pub fn set_haap_state_column(&mut self, col: usize, values: &[f32]) -> usize {
        let Some(pop) = self.haap_pop.as_mut() else {
            return 0;
        };
        let n = values.len().min(self.n_neurons);
        pop.state_slice_mut(col)[..n].copy_from_slice(&values[..n]);
        n
    }

    /// Scale one per-neuron HH parameter column (e.g. `HHModelParamIdx::g_Na`)
    /// by `factor` in place — pharmacology (channel blockers/openers) applied
    /// to a whole population at once. Returns neurons written (0 for non-HH
    /// runners). ⚠️ Same `reset()` caveat as [`Self::set_hh_param_column`].
    pub fn scale_hh_param_column(&mut self, col: usize, factor: f32) -> usize {
        let Some(pop) = self.hh_pop.as_mut() else {
            return 0;
        };
        let n = self.n_neurons;
        for v in pop.param_slice_mut(col)[..n].iter_mut() {
            *v *= factor;
        }
        n
    }

    /// Scale one per-neuron HAAP parameter column (e.g. `HAAPModelParamIdx::g_m`)
    /// by `factor` in place. Returns neurons written (0 for non-HAAP runners).
    /// ⚠️ Same `reset()` caveat as [`Self::set_haap_param_column`].
    pub fn scale_haap_param_column(&mut self, col: usize, factor: f32) -> usize {
        let Some(pop) = self.haap_pop.as_mut() else {
            return 0;
        };
        let n = self.n_neurons;
        for v in pop.param_slice_mut(col)[..n].iter_mut() {
            *v *= factor;
        }
        n
    }

    /// Get the accumulated spike buffer.
    pub fn spike_buffer(&self) -> &[i32] {
        &self.spike_buffer
    }

    /// Number of neurons.
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Total steps simulated so far.
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Reset simulator state (v, u, I_buf, spikes, kinetics).
    pub fn reset(&mut self) {
        let model = match &self.config {
            NeuronConfig::Lif { .. }
            | NeuronConfig::LifSynaptic { .. }
            | NeuronConfig::LifAlpha { .. } => NeuronModel::LIF,
            NeuronConfig::Hh { .. } | NeuronConfig::HhSynaptic { .. } => NeuronModel::Hh,
            NeuronConfig::Izh { .. }
            | NeuronConfig::IzhSynaptic { .. }
            | NeuronConfig::IzhDual { .. }
            | NeuronConfig::IzhEprop { .. } => NeuronModel::Izhikevich,
            NeuronConfig::Rate { .. } => NeuronModel::Rate,
            NeuronConfig::Gco { .. } => NeuronModel::Gco,
            NeuronConfig::Haap { .. } | NeuronConfig::HaapSynaptic { .. } => NeuronModel::Haap,
        };
        self.state = BrainState::new(self.n_neurons, model);
        self.state.reset_with_noise(self.n_neurons);
        self.spike_buffer.fill(0);
        self.step_counter = 0;
        if let Some(ref mut k) = self.kinetics {
            k.reset();
        }
        // Re-init HH engine: pillars population (standard or pacemaking)
        if self.hh_pop.is_some() {
            let dt = match &self.config {
                NeuronConfig::Hh { dt_ms, .. } | NeuronConfig::HhSynaptic { dt_ms, .. } => *dt_ms,
                _ => 0.5,
            };
            let use_pm = match &self.config {
                NeuronConfig::Hh { use_pacemaking, .. }
                | NeuronConfig::HhSynaptic { use_pacemaking, .. } => *use_pacemaking,
                _ => false,
            };
            self.hh_pop = if use_pm {
                Some(hh_population_pacemaking(self.n_neurons, dt))
            } else {
                Some(hh_population_standard(self.n_neurons, dt))
            };
        }

        // Re-init HAAP engine: pillars population (M-current + h-current).
        if self.haap_pop.is_some() {
            let dt = match &self.config {
                NeuronConfig::Haap { dt_ms } | NeuronConfig::HaapSynaptic { dt_ms, .. } => *dt_ms,
                _ => 0.05,
            };
            self.haap_pop = Some(haap_population(self.n_neurons, dt));
        }
        self.haap_syn.fill(0.0);

        // Reset alpha-synapse state (Shiu fly brain model): rebuild the SIMD
        // population (v→v_rest, g=0, refr at full duration), zero the delay
        // ring buffer, clear spikes_prev.
        if let NeuronConfig::LifAlpha { t_delay, dt_ms, .. } = &self.config {
            let n = self.n_neurons;
            self.state.v[..n].fill(0.0); // will be synced from pop on first run
                                         // Rebuild from config — preserves per-neuron refr_steps overrides
                                         // ONLY if the caller re-applies set_alpha_refrac_ms after reset
                                         // (same contract as the HH population).
            let new_pop = self.make_alpha_pop();
            if let Some(pop) = self.alpha_pop.as_mut() {
                *pop = new_pop;
            }
            let size = (t_delay / dt_ms).round().max(0.0) as usize + 1;
            self.alpha_delay.resize(size * n, 0.0);
            self.alpha_delay.fill(0.0);
            self.alpha_delay_pos = size - 1; // read at (pos+1) % size == 0
            self.alpha_fired.clear();
        }
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: Poisson injection (shared by all models)
    // ═══════════════════════════════════════════════════════

    fn inject_poisson(&mut self, dt_ms: f32) {
        // Shared stochastic logic — single source of truth in `inject_poisson_into`.
        inject_poisson_into(
            &self.poisson_groups,
            &mut self.rng,
            &mut self.state.I_buf,
            dt_ms,
        );
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: LIF runners
    // ═══════════════════════════════════════════════════════

    fn run_lif(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        synaptic: bool,
    ) {
        let n_neurons = self.connectome.row_ptr.len() - 1;

        // Reuse persistent buffers (was allocated every run() call).
        let work = &mut self.work_buf;
        let ext = &mut self.ext_buf;

        if synaptic {
            // Conductance path — SynapticBrain owns the kinetics loop.
            let mut brain = crate::SynapticBrain::<pillars::zoo::LIFModel>::from_connectome(
                &self.connectome,
                dt_ms,
            );
            if let Some(k) = &self.kinetics {
                brain.set_kinetics(k); // carry-over
            }
            brain.configure(|p| {
                for i in 0..n_neurons {
                    p.param_slice_mut(0)[i] = dt_ms;
                    p.param_slice_mut(1)[i] = 1.0 / tau_m.max(1e-6);
                    p.param_slice_mut(2)[i] = v_rest;
                    p.param_slice_mut(3)[i] = v_thresh;
                    p.param_slice_mut(4)[i] = v_reset;
                    p.param_slice_mut(5)[i] = tau_ref;
                    p.state_slice_mut(0)[i] = self.state.v[i];
                    p.state_slice_mut(1)[i] = self.state.refractory[i];
                }
            });
            for step in 0..n_steps {
                work.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, work, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + work[i];
                }
                let s = brain.step(ext);
                record_spikes(&s, n_neurons, &mut self.spike_buffer, step);
            }
            self.kinetics = Some(brain.kinetics().clone());
            let v = brain.state(0);
            let refr = brain.state(1);
            for i in 0..n_neurons {
                self.state.v[i] = v[i];
                self.state.refractory[i] = refr[i];
            }
        } else {
            // Instant-current path — Brain (api::Network) does the routing.
            let mut brain =
                crate::Brain::<pillars::zoo::LIFModel>::from_connectome(&self.connectome);
            brain.configure(|p| {
                for i in 0..n_neurons {
                    p.param_slice_mut(0)[i] = dt_ms;
                    p.param_slice_mut(1)[i] = 1.0 / tau_m.max(1e-6);
                    p.param_slice_mut(2)[i] = v_rest;
                    p.param_slice_mut(3)[i] = v_thresh;
                    p.param_slice_mut(4)[i] = v_reset;
                    p.param_slice_mut(5)[i] = tau_ref;
                    p.state_slice_mut(0)[i] = self.state.v[i];
                    p.state_slice_mut(1)[i] = self.state.refractory[i];
                }
            });
            let idx = &self.all_indices;
            for step in 0..n_steps {
                work.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, work, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + work[i];
                }
                brain.inject(idx, ext);
                brain.step();
                record_spikes(brain.spikes(), n_neurons, &mut self.spike_buffer, step);
            }
            let v = brain.state(0);
            let refr = brain.state(1);
            for i in 0..n_neurons {
                self.state.v[i] = v[i];
                self.state.refractory[i] = refr[i];
            }
        }

        self.step_counter += n_steps;
    }

    /// LIF + alpha-synapse — exact Shiu et al. 2024 fly brain model.
    ///
    /// Mirrors `fly-brain-main/code/run_pytorch.py` (AlphaLIF) 1:1:
    /// - pre-spike (with 1.8 ms axonal delay) jumps `g` by the edge weight (mV)
    /// - `g` decays with τ_syn: `g *= (1 - dt/τ_syn)`
    /// - Poisson/voltage stimuli are added **directly to `v`**
    /// - `v += dt/τ_mem · (g − (v − v_rest))`; spike when `v > v_thresh`;
    ///   `v = v_reset` and `g = 0` on spike
    /// - refractory: incoming `g` is gated while `refrac_cnt < refrac_steps`
    ///
    /// The edge weights in `ConnectomeData::values` are the full recurrent
    /// input in mV (already × wScale=0.275), matching the baked .taas.
    /// Build the alpha-LIF SIMD population from config params.
    ///
    /// The population owns the per-neuron `v`, `g`, `refr_cnt` state and the
    /// scalar params (dt, time_factor, v_rest/thresh/reset, refr_steps,
    /// syn_decay). The delay ring buffer + CSR scatter stay in the runner.
    fn make_alpha_pop(&self) -> NeuralPopulation<AlphaLifModel> {
        let (dt_ms, tau_ref, tau_m, v_rest, v_thresh, v_reset, tau_syn) = match &self.config {
            NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                ..
            } => (
                *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, *tau_syn,
            ),
            _ => (0.1, 2.2, 20.0, -52.0, -45.0, -52.0, 5.0),
        };
        alpha_lif_population(
            self.n_neurons,
            dt_ms,
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_syn,
        )
    }

    fn run_lif_alpha(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_syn: f32,
        t_delay: f32,
        record: bool,
    ) -> (usize, Vec<bool>) {
        let n = self.n_neurons;
        let steps_delay = (t_delay / dt_ms).round().max(0.0) as usize;
        let size = steps_delay + 1;
        if self.alpha_delay.len() < size * n {
            self.alpha_delay = vec![0.0; size * n];
            self.alpha_delay_pos = size - 1;
        }
        let _ = (tau_ref, tau_m, v_rest, v_thresh, v_reset, tau_syn);

        let rp = self.csr_graph.row_ptr();
        let ci = self.csr_graph.col_indices();
        let vals = &self.connectome.values;

        let pop = self.alpha_pop.as_mut().expect("alpha_pop must be built");
        let mut spikes = vec![0i32; n];
        let mut total_spikes = 0usize;
        let mut active = vec![false; n];

        for step in 0..n_steps {
            // 1. Poisson/voltage stimuli → direct v jumps (target_var='v'),
            //    then external_I on top. Poisson only touches the 21 sugar GRN,
            //    so injecting straight into v avoids a full `work`-buffer pass.
            {
                let v_slice = pop.state_slice_mut(AlphaLifModelStateIdx::v.into());
                inject_poisson_into(&self.poisson_groups, &mut self.rng, v_slice, dt_ms);
                if !external_I.is_empty() {
                    for i in 0..n {
                        v_slice[i] += external_I.get(i).copied().unwrap_or(0.0);
                    }
                }
            }

            // 2. Feed this step's delayed recurrent input straight from the
            //    ring slot (no copy) to the SIMD neuron step.
            let read_pos = self.alpha_delay_pos;
            // SIMD-шаг (sequential; `step_par` был WIP и удалён — см. git log).
            pop.step(
                &self.alpha_delay[read_pos * n..(read_pos + 1) * n],
                &mut spikes,
            );

            // 3. Zero the slot, scatter the recurrent input of the *previous*
            //    step's firing neurons into it, then advance the ring pointer.
            {
                let slot = &mut self.alpha_delay[read_pos * n..(read_pos + 1) * n];
                slot.fill(0.0);
                for &pre in &self.alpha_fired {
                    let start = rp[pre] as usize;
                    let end = rp[pre + 1] as usize;
                    for k in start..end {
                        let post = ci[k] as usize;
                        slot[post] += vals[k];
                    }
                }
            }
            self.alpha_delay_pos = (read_pos + 1) % size;

            // 4. Build the next step's firing list + record spikes in one pass
            //    over `spikes` (replaces the old `alpha_spikes_prev` memcpy +
            //    a separate `record_spikes` scan).
            self.alpha_fired.clear();
            let base = step * n;
            // SIMD-скан: 8 спайков за раз через move_mask. Спайки бывают 0/1
            // (i32), поэтому reinterpret в f32 и `> 0.0` даёт тот же non-zero
            // маск (i32 1 → denormal 1.4e-45 > 0). Порядок — строго по
            // возрастанию i → bit-exact с прежним скалярным циклом.
            let spike_ptr = spikes.as_ptr();
            let n8 = n & !7;
            let mut i = 0usize;
            while i < n8 {
                let lane = unsafe { (spike_ptr.add(i) as *const SimdType).read_unaligned() };
                let mut m = lane.cmp_gt(SimdType::splat(0.0)).move_mask();
                while m != 0 {
                    let idx = i + (m.trailing_zeros() as usize);
                    self.alpha_fired.push(idx);
                    total_spikes += 1;
                    active[idx] = true;
                    if record {
                        self.spike_buffer[base + idx] = 1;
                    }
                    m &= m - 1;
                }
                i += 8;
            }
            for idx in n8..n {
                if spikes[idx] != 0 {
                    self.alpha_fired.push(idx);
                    total_spikes += 1;
                    active[idx] = true;
                    if record {
                        self.spike_buffer[base + idx] = 1;
                    }
                }
            }
        }

        // Sync membrane state back to BrainState (get_v/set_v read it).
        {
            let v_slice = pop.state_slice(AlphaLifModelStateIdx::v.into());
            self.state.v[..n].copy_from_slice(v_slice);
        }

        self.step_counter += n_steps;
        (total_spikes, active)
    }

    /// Override the refractory duration (in ms) for specific neurons.
    ///
    /// Used e.g. for Poisson-stimulated neurons in the Shiu model, which have
    /// zero refractory so they can fire on every Poisson event.
    pub fn set_alpha_refrac_ms(&mut self, indices: &[usize], tau_ref_ms: f32) {
        let dt = match &self.config {
            NeuronConfig::LifAlpha { dt_ms, .. } => *dt_ms,
            _ => 0.1,
        };
        let steps = (tau_ref_ms / dt).round().max(0.0);
        if let Some(pop) = self.alpha_pop.as_mut() {
            let col = pop.param_slice_mut(AlphaLifModelParamIdx::refr_steps.into());
            for &i in indices {
                if i < self.n_neurons {
                    col[i] = steps;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: Izhikevich runners
    // ═══════════════════════════════════════════════════════

    fn run_izhikevich(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        synaptic: bool,
    ) {
        let n_neurons = self.connectome.row_ptr.len() - 1;

        // Reuse persistent buffers (was allocated every run() call).
        let work = &mut self.work_buf;
        let ext = &mut self.ext_buf;

        if synaptic {
            let mut brain = crate::SynapticBrain::<pillars::zoo::IzhikevichModel>::from_connectome(
                &self.connectome,
                dt_ms,
            );
            if let Some(k) = &self.kinetics {
                brain.set_kinetics(k);
            }
            brain.configure(|p| {
                for i in 0..n_neurons {
                    p.param_slice_mut(0)[i] = dt_ms;
                    p.param_slice_mut(1)[i] = a;
                    p.param_slice_mut(2)[i] = b;
                    p.param_slice_mut(3)[i] = c;
                    p.param_slice_mut(4)[i] = d;
                    p.param_slice_mut(5)[i] = tau_ref;
                    p.param_slice_mut(6)[i] = 30.0; // v_thresh
                    p.state_slice_mut(0)[i] = self.state.v[i];
                    p.state_slice_mut(1)[i] = self.state.u[i];
                    p.state_slice_mut(2)[i] = self.state.refractory[i];
                }
            });
            for step in 0..n_steps {
                work.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, work, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + work[i];
                }
                let s = brain.step(ext);
                record_spikes(&s, n_neurons, &mut self.spike_buffer, step);
            }
            self.kinetics = Some(brain.kinetics().clone());
            let v = brain.state(0);
            let u = brain.state(1);
            let refr = brain.state(2);
            for i in 0..n_neurons {
                self.state.v[i] = v[i];
                self.state.u[i] = u[i];
                self.state.refractory[i] = refr[i];
            }
        } else {
            let mut brain =
                crate::Brain::<pillars::zoo::IzhikevichModel>::from_connectome(&self.connectome);
            brain.configure(|p| {
                for i in 0..n_neurons {
                    p.param_slice_mut(0)[i] = dt_ms;
                    p.param_slice_mut(1)[i] = a;
                    p.param_slice_mut(2)[i] = b;
                    p.param_slice_mut(3)[i] = c;
                    p.param_slice_mut(4)[i] = d;
                    p.param_slice_mut(5)[i] = tau_ref;
                    p.param_slice_mut(6)[i] = 30.0; // v_thresh
                    p.state_slice_mut(0)[i] = self.state.v[i];
                    p.state_slice_mut(1)[i] = self.state.u[i];
                    p.state_slice_mut(2)[i] = self.state.refractory[i];
                }
            });
            let idx = &self.all_indices;
            for step in 0..n_steps {
                work.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, work, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + work[i];
                }
                brain.inject(idx, ext);
                brain.step();
                record_spikes(brain.spikes(), n_neurons, &mut self.spike_buffer, step);
            }
            let v = brain.state(0);
            let u = brain.state(1);
            let refr = brain.state(2);
            for i in 0..n_neurons {
                self.state.v[i] = v[i];
                self.state.u[i] = u[i];
                self.state.refractory[i] = refr[i];
            }
        }

        self.step_counter += n_steps;
    }

    fn run_izhikevich_dual(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        tau_m: f32,
        w_m: f32,
        hebb_lr: f32,
        hebb_decay: f32,
    ) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let n_synapses = self.connectome.values.len();
        let n_padded = self.state.n_padded;

        let padded_a = crate::sim::pad_to(&vec![a; n], n_padded);
        let padded_b = crate::sim::pad_to(&vec![b; n], n_padded);
        let padded_c = crate::sim::pad_to(&vec![c; n], n_padded);
        let padded_d = crate::sim::pad_to(&vec![d; n], n_padded);
        let trace_decay = 0.99;

        for step in 0..n_steps {
            // 1. Add external_I + Poisson
            for i in 0..n.min(external_I.len()) {
                self.state.I_buf[i] += external_I[i];
            }
            self.inject_poisson(dt_ms);

            // 2. Izhikevich dual step (slow state m modulates I)
            crate::sim::step_izhikevich_dual_pillars(
                &mut self.state.v,
                &mut self.state.u,
                &mut self.state.trace,
                &padded_a,
                &padded_b,
                &mut self.state.I_buf,
                dt_ms,
                trace_decay,
                &mut self.state.spikes,
                &padded_c,
                &padded_d,
                &mut self.state.refractory,
                tau_ref,
                &mut self.state.slow_state,
                tau_m,
                w_m,
            );

            // 3. Push spikes → record
            push_spikes_to_queue(&self.state.spikes, n_neurons, &mut self.state.queue);
            record_spikes(&self.state.spikes, n_neurons, &mut self.spike_buffer, step);

            // 4. Hebbian plasticity
            if hebb_lr > 0.0 && n_synapses > 0 {
                pillars::kernels::learning::hebbian_csr_step(
                    &mut self.connectome.values,
                    &self.connectome.col_indices,
                    &self.connectome.row_ptr,
                    &self.state.slow_state[..n_neurons],
                    n_neurons,
                    hebb_lr,
                    hebb_decay,
                    dt_ms,
                );
            }

            // 5. Clear I_buf
            self.state.I_buf[..n].fill(0.0);

            // 6. CSR propagate
            csr_propagate_from_queue_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.state.queue,
                &mut self.state.I_buf,
                1.0,
            );
        }

        self.step_counter += n_steps;
    }

    fn run_izhikevich_eprop(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau_ref: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        tau_e: f32,
        tau_surrogate: f32,
    ) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let n_syn = self.connectome.values.len();
        let n_padded = self.state.n_padded;

        let padded_a = crate::sim::pad_to(&vec![a; n], n_padded);
        let padded_b = crate::sim::pad_to(&vec![b; n], n_padded);
        let padded_c = crate::sim::pad_to(&vec![c; n], n_padded);
        let padded_d = crate::sim::pad_to(&vec![d; n], n_padded);
        let beta_e = (-dt_ms / tau_e.max(dt_ms)).exp();
        let trace_decay = 0.99;

        let mut pre_trace = vec![0.0f32; n];
        let mut psi = vec![0.0f32; n];
        let v_th = 30.0;
        let surrogate_scale = 1.0 / tau_surrogate;
        let max_syn = self.connectome.col_indices.len().min(n_syn);

        for step in 0..n_steps {
            // 1. Add external_I + Poisson
            for i in 0..n.min(external_I.len()) {
                self.state.I_buf[i] += external_I[i];
            }
            self.inject_poisson(dt_ms);

            // 2. Izhikevich step
            crate::sim::step_izhikevich_pillars(
                &mut self.state.v,
                &mut self.state.u,
                &mut self.state.trace,
                &padded_a,
                &padded_b,
                &mut self.state.I_buf,
                dt_ms,
                trace_decay,
                &mut self.state.spikes,
                &padded_c,
                &padded_d,
                &mut self.state.refractory,
                tau_ref,
            );

            // 3. Surrogate gradient
            for i in 0..n {
                let dv = (self.state.v[i] - v_th).abs();
                psi[i] = 1.0 / (1.0 + dv * surrogate_scale).powi(2);
            }

            // 4. Pre-spike trace
            for i in 0..n {
                pre_trace[i] = pre_trace[i] * trace_decay
                    + if i < self.state.spikes.len() && self.state.spikes[i] != 0 {
                        1.0
                    } else {
                        0.0
                    };
            }

            // 5. Eligibility update for active pre-neurons
            for i in 0..n {
                let tr = pre_trace[i];
                if tr < 1e-6 {
                    continue;
                }
                let start = self.connectome.row_ptr[i].max(0) as usize;
                let end = (self.connectome.row_ptr[i + 1] as usize).min(max_syn);
                for idx in start..end {
                    let j = self.connectome.col_indices[idx] as usize;
                    if j < n {
                        self.eligibility[idx] =
                            self.eligibility[idx] * beta_e + tr * psi[j] * dt_ms;
                    }
                }
            }

            // 6. Push spikes → record
            push_spikes_to_queue(&self.state.spikes, n_neurons, &mut self.state.queue);
            record_spikes(&self.state.spikes, n_neurons, &mut self.spike_buffer, step);

            // 7. Clear I_buf
            self.state.I_buf[..n].fill(0.0);

            // 8. CSR propagate
            csr_propagate_from_queue_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.state.queue,
                &mut self.state.I_buf,
                1.0,
            );
        }

        self.step_counter += n_steps;
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: HH runners
    // ═══════════════════════════════════════════════════════

    fn run_hh(&mut self, n_steps: usize, external_I: &[f32], dt_ms: f32, synaptic: bool) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        // `record_spikes` strides the buffer by n_neurons — must equal n.
        debug_assert_eq!(n, n_neurons, "spike buffer stride invariant");

        // ─── pillars engine: NeuralPopulation<HHModel> (SIMD) — standard AND pacemaking ───
        let pop = self.hh_pop.as_mut().unwrap();

        // Recreate params if dt changed (state preserved — better than legacy rebuild)
        if (pop.param_slice(HHModelParamIdx::dt.into())[0] - dt_ms).abs() > 1e-6 {
            pop.fill_param(HHModelParamIdx::dt.into(), dt_ms);
        }

        let mut spikes = vec![0i32; n];
        let mut i_in = vec![0.0f32; n]; // carry-over current buffer (like old cfg.Ii)

        for step in 0..n_steps {
            // 1. Add external_I + Poisson
            for i in 0..n.min(external_I.len()) {
                i_in[i] += external_I[i];
            }
            inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut i_in, dt_ms);

            if synaptic {
                let kinetics = self.kinetics.as_mut().unwrap();
                kinetics.decay_all(&DecayFactors::new(dt_ms));
                kinetics.compute_currents(pop.state_slice(0), &mut i_in);
            }

            // 2. pillars HH step (Ii clamp ±100 happens inside NeuralPopulation)
            pop.step(&i_in, &mut spikes);

            // 3. Push spikes → record
            push_spikes_to_queue(&spikes, n_neurons, &mut self.state.queue);
            record_spikes(&spikes, n_neurons, &mut self.spike_buffer, step);

            // 4. Clear i_in
            i_in[..n].fill(0.0);

            // 5. CSR propagate
            csr_propagate_after_hh(
                &self.connectome,
                &self.csr_graph,
                &mut self.state.queue,
                &mut self.kinetics,
                &mut i_in,
                synaptic,
            );
        }

        self.step_counter += n_steps;
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: Rate runner
    // ═══════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════
    //  Internal: HAAP runner
    // ═══════════════════════════════════════════════════════════

    fn run_haap(&mut self, n_steps: usize, external_I: &[f32], dt_ms: f32, tau_syn: f32) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        debug_assert_eq!(n, n_neurons, "spike buffer stride invariant");

        let pop = self.haap_pop.as_mut().unwrap();

        // Recreate params if dt changed (state preserved).
        if (pop.param_slice(HAAPModelParamIdx::dt.into())[0] - dt_ms).abs() > 1e-6 {
            pop.fill_param(HAAPModelParamIdx::dt.into(), dt_ms);
        }

        let mut spikes = vec![0i32; n];
        let i_syn = &mut self.haap_syn; // persistent synaptic current (decays)
        let mut i_in = vec![0.0f32; n]; // per-step input to the population

        // τ_syn = 0 → instant current (old behaviour, bit-identical).
        // τ_syn > 0 → decaying GABA/AMPA-like synaptic current.
        let decay = if tau_syn > 0.0 {
            (-dt_ms / tau_syn).exp()
        } else {
            0.0
        };

        for step in 0..n_steps {
            // 1. Combine external_I + decaying synaptic current + Poisson
            for i in 0..n {
                i_in[i] = external_I.get(i).copied().unwrap_or(0.0) + i_syn[i];
            }
            inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut i_in, dt_ms);

            // 2. pillars HAAP step (current-based — M-current + h-current)
            pop.step(&i_in, &mut spikes);

            // 3. Push spikes → record
            push_spikes_to_queue(&spikes, n_neurons, &mut self.state.queue);
            record_spikes(&spikes, n_neurons, &mut self.spike_buffer, step);

            // 4. Decay (or clear) the synaptic current buffer
            for s in i_syn.iter_mut() {
                *s *= decay;
            }

            // 5. CSR propagate (current-based: I_syn[post] += W·nt_types[pre])
            csr_propagate_after_hh(
                &self.connectome,
                &self.csr_graph,
                &mut self.state.queue,
                &mut self.kinetics,
                &mut *i_syn,
                false,
            );
        }

        self.step_counter += n_steps;
    }

    fn run_rate(&mut self, n_steps: usize, external_I: &[f32], dt_ms: f32, tau: f32) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let dt = dt_ms / 1000.0;
        let decay = (-dt / tau.max(1e-6)).exp();
        let inv_decay = 1.0 - decay;

        for step in 0..n_steps {
            // 1. Set I = external + Poisson (fresh each step — no stale tail)
            self.state.I_buf[..n].fill(0.0);
            for (i, &ext) in external_I.iter().enumerate().take(n) {
                self.state.I_buf[i] += ext;
            }
            self.inject_poisson(dt_ms);

            // 2. Build spike proxy for CSR: v > 0.5 → "spike"
            self.state.queue.drain();
            for i in 0..n_neurons {
                if i < self.state.v.len() && self.state.v[i] > 0.5 {
                    self.state.spikes[i] = 1;
                    self.state.queue.push(i as u32);
                } else {
                    self.state.spikes[i] = 0;
                }
            }

            // 3. CSR propagate
            csr_propagate_from_queue_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.state.queue,
                &mut self.state.I_buf,
                1.0,
            );

            // 4. Exponential Euler (external already in I_buf — do not add again)
            for i in 0..n {
                let input = self.state.I_buf[i].clamp(0.0, 1.0);
                self.state.v[i] = self.state.v[i] * decay + input * inv_decay;
            }

            // 5. Record "spikes"
            for i in 0..n_neurons {
                if i < self.state.v.len() && self.state.v[i] > 0.5 {
                    let idx = step * n + i;
                    if idx < self.spike_buffer.len() {
                        self.spike_buffer[idx] = 1;
                    }
                }
            }
        }

        self.step_counter += n_steps;
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: GCO runner
    // ═══════════════════════════════════════════════════════

    fn run_gco(&mut self, n_steps: usize, external_I: &[f32], dt_ms: f32) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let dt = dt_ms / 1000.0;

        let mut h_real = vec![0.0; self.state.n_padded];
        let mut h_imag = vec![0.0; self.state.n_padded];
        let speed = vec![1.0; self.state.n_padded];

        for step in 0..n_steps {
            // 1. Build Hamiltonian from the topology validated at construction.
            h_real.fill(0.0);
            h_imag.fill(0.0);
            pillars::kernels::csr::compute_hamiltonian_scatter_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &self.state.v,
                &self.state.u,
                &mut h_real,
                &mut h_imag,
            );

            // 2. Set external input
            for i in 0..n.min(external_I.len()) {
                self.state.I_buf[i] = external_I[i];
            }
            self.inject_poisson(dt_ms);

            // 3. GCO step
            pillars::kernels::phase::gco_step(
                &mut self.state.v,
                &mut self.state.u,
                &h_real,
                &h_imag,
                &self.state.I_buf,
                &speed,
                dt,
            );

            // 4. "Spike" if |v+iu|² > 0.5
            for i in 0..n_neurons {
                let prob = self.state.v[i] * self.state.v[i] + self.state.u[i] * self.state.u[i];
                if prob > 0.5 {
                    let idx = step * n + i;
                    if idx < self.spike_buffer.len() {
                        self.spike_buffer[idx] = 1;
                    }
                }
            }
        }

        self.step_counter += n_steps;
    }

    /// Get the accumulated eligibility traces (from E-Prop).
    /// Useful for weight updates after simulation.
    pub fn eligibility_traces(&self) -> &[f32] {
        &self.eligibility
    }

    /// Apply E-Prop weight updates using learning signals.
    /// Only valid after running with `IzhEprop` config.
    ///
    /// `learning_signals` — `[n_neurons]` error signal for each post-synaptic neuron.
    /// `lr` — learning rate.
    pub fn apply_eprop_update(&mut self, learning_signals: &[f32], lr: f32) {
        if self.eligibility.is_empty() {
            return;
        }
        let n_syn = self.connectome.values.len().min(self.eligibility.len());
        for idx in 0..n_syn {
            let post = self.connectome.col_indices[idx] as usize;
            if post < learning_signals.len() {
                self.connectome.values[idx] -= lr * learning_signals[post] * self.eligibility[idx];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConnectomeData;

    fn make_test_connectome() -> ConnectomeData {
        // 4 neurons, ring: 0→1, 1→2, 2→3, 3→0
        ConnectomeData::new(
            vec![0i32, 1, 2, 3, 4],
            vec![1, 2, 3, 0],
            vec![0.8f32, 0.5, 0.3, 0.7],
            vec![1.0, 1.0, -1.0, 1.0],
            vec![("dn", vec![0, 1]), ("mn", vec![2, 3])],
        )
    }

    #[test]
    fn test_brain_runner_refreshes_topology_after_public_csr_edit() {
        let mut conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn.clone(), BrainConfig::lif(0.5)).unwrap();
        assert_eq!(runner.csr_graph().n_neurons(), 4);

        // Public CSR fields remain mutable for ABI compatibility. A topology
        // mutation is accepted only after an explicit refresh.
        conn.row_ptr[1] = 2;
        runner.connectome.row_ptr = conn.row_ptr;
        runner.refresh_csr_graph();
        assert_eq!(runner.csr_graph().row_ptr()[1], 2);
    }

    #[test]
    fn test_brain_runner_lif_basic() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();

        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "LIF should fire with I=30");
        assert_eq!(spikes.len(), 50 * 4);
    }

    #[test]
    fn test_brain_runner_izhikevich_basic() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::izh(0.5)).unwrap();
        let ext = vec![10.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();

        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "Izhikevich should fire with I=10");
        assert_eq!(spikes.len(), 50 * 4);
    }

    #[test]
    fn test_brain_runner_hh_basic() {
        let conn = make_test_connectome();
        // dt=0.01 ms — HH exponential-Euler is numerically unstable at dt=0.5
        // (the NaN safety check in ohmic_gated catches it). 5000 steps = 50 ms.
        let mut runner = BrainRunner::new(conn, BrainConfig::hh(0.01)).unwrap();
        // Use `use_pacemaking` mode which has CaT channel for oscillation
        // Standard HH needs higher current
        let ext = vec![15.0f32; runner.n_neurons()];

        let spikes = runner.run(5000, &ext).to_vec();

        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "HH should fire with I=15");
        assert_eq!(spikes.len(), 5000 * 4);
    }

    #[test]
    fn test_brain_runner_rate_basic() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::rate(0.5, 10.0)).unwrap();
        let ext = vec![0.5f32; runner.n_neurons()]; // rate input (0..1)

        let spikes = runner.run(50, &ext).to_vec();

        assert_eq!(spikes.len(), 50 * 4);
    }

    #[test]
    fn test_brain_runner_poisson_stimulation() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(1.0)).unwrap();
        runner.add_poisson_group(&[0, 1, 2], 500.0, 50.0); // high rate and weight
        let ext = vec![0.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();

        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "Poisson-only stimulation should fire neurons");
    }

    #[test]
    fn test_brain_runner_izhikevich_dual() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn.clone(), BrainConfig::izh_dual(0.5)).unwrap();
        let ext = vec![10.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "Izh dual should fire");
    }

    #[test]
    fn test_brain_runner_synaptic() {
        let mut conn = make_test_connectome();
        conn.build_receptor_types();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif_synaptic(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "LIF synaptic should fire");
    }

    #[test]
    fn test_brain_runner_spike_trains() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];
        runner.run(50, &ext);

        let trains = runner.spike_trains(0.5);
        assert!(!trains.is_empty(), "At least some neurons should spike");
    }

    #[test]
    fn test_brain_runner_firing_rates() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];
        runner.run(50, &ext);

        let rates = runner.firing_rates(0.5);
        assert_eq!(rates.len(), 4);
        let total_rate: f32 = rates.iter().sum();
        assert!(total_rate > 0.0, "Total firing rate should be > 0");
    }

    #[test]
    fn test_brain_runner_firing_rates_group() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];
        runner.run(50, &ext);

        let dn_rates = runner.firing_rates_group(&[0, 1], 50, 0.5);
        assert_eq!(dn_rates.len(), 2);
    }

    #[test]
    fn test_brain_runner_reset() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];
        runner.run(50, &ext);
        assert!(runner.step_counter() > 0);

        runner.reset();
        assert_eq!(runner.step_counter(), 0);
    }

    #[test]
    fn test_brain_runner_persistence() {
        // State should persist between run() calls
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();
        let ext = vec![30.0f32; runner.n_neurons()];

        runner.run(25, &ext);
        assert_eq!(runner.step_counter(), 25);

        runner.run(25, &ext);
        assert_eq!(runner.step_counter(), 50);
    }

    #[test]
    fn test_brain_runner_set_get_v() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::lif(0.5)).unwrap();

        runner.set_v(&[-60.0, -60.0, -60.0, -60.0]);
        let v = runner.get_v();
        assert!((v[0] - (-60.0)).abs() < 1.0);
    }

    #[test]
    fn test_brain_runner_izhikevich_eprop() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::izh_eprop(0.5)).unwrap();
        let ext = vec![10.0f32; runner.n_neurons()];

        let spikes = runner.run(50, &ext).to_vec();
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "Izh E-Prop should fire");

        // Eligibility traces should be non-empty
        let traces = runner.eligibility_traces();
        assert_eq!(traces.len(), 4); // 4 synapses in test connectome
    }

    #[test]
    fn test_brain_runner_apply_eprop_update() {
        let conn = make_test_connectome();
        let mut runner = BrainRunner::new(conn, BrainConfig::izh_eprop(0.5)).unwrap();
        let ext = vec![10.0f32; runner.n_neurons()];
        runner.run(50, &ext);

        let old_weights = runner.connectome.values.clone();
        let learning_signals = vec![0.1f32; runner.n_neurons()];
        runner.apply_eprop_update(&learning_signals, 0.01);

        // Weights should have changed
        let changed: bool = runner
            .connectome
            .values
            .iter()
            .zip(old_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "E-Prop update should modify weights");
    }
}
