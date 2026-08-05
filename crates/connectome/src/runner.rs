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
//! use taas_connectome::{ConnectomeData, BrainRunner};
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
use rand::{Rng, SeedableRng};
use taas_kernels::neuron::synapse::{DecayFactors, SynapticKinetics};

use crate::sim::{
    hh_population_pacemaking, hh_population_standard, kinetics_legacy_to_pillars,
    kinetics_pillars_to_legacy,
};
use pillars::core::NeuralPopulation;
use pillars::zoo::{HHModel, HHParamIdx};

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
    Hh {
        dt_ms: f32,
        use_pacemaking: bool,
    },
    /// Hodgkin-Huxley with conductance-based synapses
    HhSynaptic {
        dt_ms: f32,
        use_pacemaking: bool,
    },
    /// Rate model (continuous firing rates)
    Rate {
        dt_ms: f32,
        tau: f32,
    },
    /// Graph-coupled oscillator (ψ-wave dynamics)
    Gco {
        dt_ms: f32,
    },
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
    queue: &mut SpikeQueue,
    kinetics: &mut Option<SynapticKinetics>,
    buf: &mut [f32],
    synaptic: bool,
) {
    if synaptic {
        csr_propagate_conductance_from_queue(
            &connectome.row_ptr,
            &connectome.col_indices,
            &connectome.values,
            &connectome.receptor_types,
            queue,
            kinetics.as_mut().unwrap(),
            1.0,
        );
    } else {
        csr_propagate_from_queue(
            &connectome.row_ptr,
            &connectome.col_indices,
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
#[derive(Debug, Clone)]
#[derive(Default)]
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
    /// Brain state (v, u, I_buf, spikes, refractory, queue).
    pub state: BrainState,
    /// Poisson stimulation groups.
    pub poisson_groups: Vec<PoissonGroup>,
    // Internal buffers (persistent across calls to avoid reallocation)
    spike_buffer: Vec<i32>,
    n_neurons: usize,
    step_counter: usize,

    // HH population (pillars SIMD engine) — standard or pacemaking
    // (NaV+KV+CaV-T) via NeuralPopulation<HHModel>.
    hh_pop: Option<NeuralPopulation<HHModel>>,

    // Synaptic kinetics (for conductance-based models)
    kinetics: Option<SynapticKinetics>,

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

        // Determine model enum from config
        let model = match &config {
            NeuronConfig::Lif { .. }
            | NeuronConfig::LifSynaptic { .. }
            | NeuronConfig::Hh { .. }
            | NeuronConfig::HhSynaptic { .. } => NeuronModel::LIF,
            NeuronConfig::Izh { .. }
            | NeuronConfig::IzhSynaptic { .. }
            | NeuronConfig::IzhDual { .. }
            | NeuronConfig::IzhEprop { .. } => NeuronModel::Izhikevich,
            NeuronConfig::Rate { .. } => NeuronModel::Rate,
            NeuronConfig::Gco { .. } => NeuronModel::Gco,
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
            NeuronConfig::Hh { dt_ms, use_pacemaking }
            | NeuronConfig::HhSynaptic { dt_ms, use_pacemaking } => {
                if *use_pacemaking {
                    Some(hh_population_pacemaking(n, *dt_ms))
                } else {
                    Some(hh_population_standard(n, *dt_ms))
                }
            }
            _ => None,
        };

        // Initialize E-Prop eligibility buffer
        let eligibility = match &config {
            NeuronConfig::IzhEprop { .. } => vec![0.0f32; connectome.values.len()],
            _ => Vec::new(),
        };

        let spike_buf_len = n * 64; // initial allocation (grows on demand)
        let rng = rand::rngs::StdRng::from_seed([0u8; 32]);

        Ok(Self {
            config,
            connectome,
            state,
            poisson_groups: Vec::new(),
            spike_buffer: vec![0i32; spike_buf_len],
            n_neurons: n,
            step_counter: 0,
            hh_pop,
            kinetics,
            eligibility,
            rng,
        })
    }

    /// Get the learning rule extracted from the current config.
    pub fn learning_rule(&self) -> LearningRule {
        match &self.config {
            NeuronConfig::IzhDual {
                hebb_lr, hebb_decay, ..
            } if *hebb_lr > 0.0 => LearningRule::Hebbian {
                lr: *hebb_lr,
                decay: *hebb_decay,
            },
            NeuronConfig::IzhEprop {
                tau_e, tau_surrogate, ..
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

        match &self.config {
            NeuronConfig::Lif {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
            } => self.run_lif(n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, false),
            NeuronConfig::LifSynaptic {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
            } => self.run_lif(n_steps, external_I, *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, true),
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
                n_steps, external_I, *dt_ms, *tau_ref,
                *a, *b, *c, *d, *tau_m, *w_m, *hebb_lr, *hebb_decay,
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
                n_steps, external_I, *dt_ms, *tau_ref,
                *a, *b, *c, *d, *tau_e, *tau_surrogate,
            ),
            NeuronConfig::Hh { dt_ms, .. } => {
                self.run_hh(n_steps, external_I, *dt_ms, false)
            }
            NeuronConfig::HhSynaptic { dt_ms, .. } => {
                self.run_hh(n_steps, external_I, *dt_ms, true)
            }
            NeuronConfig::Rate { dt_ms, tau } => {
                self.run_rate(n_steps, external_I, *dt_ms, *tau)
            }
            NeuronConfig::Gco { dt_ms } => {
                self.run_gco(n_steps, external_I, *dt_ms)
            }
        }

        &self.spike_buffer[..needed]
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

        counts
            .iter()
            .map(|&c| c as f32 / duration_s)
            .collect()
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
            | NeuronConfig::Hh { .. }
            | NeuronConfig::HhSynaptic { .. } => NeuronModel::LIF,
            NeuronConfig::Izh { .. }
            | NeuronConfig::IzhSynaptic { .. }
            | NeuronConfig::IzhDual { .. }
            | NeuronConfig::IzhEprop { .. } => NeuronModel::Izhikevich,
            NeuronConfig::Rate { .. } => NeuronModel::Rate,
            NeuronConfig::Gco { .. } => NeuronModel::Gco,
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
    }

    // ═══════════════════════════════════════════════════════
    //  Internal: Poisson injection (shared by all models)
    // ═══════════════════════════════════════════════════════

    fn inject_poisson(&mut self, dt_ms: f32) {
        // Shared stochastic logic — single source of truth in `inject_poisson_into`.
        inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut self.state.I_buf, dt_ms);
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
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;

        // Delegate to the pillars-native orchestrator. `scale` is folded in as
        // weight=1.0 (instant current) — the legacy queue/CSR loop disappears.
        let new_conn = pillars_connectome::ConnectomeData::from_csr(
            self.connectome.row_ptr.iter().map(|&v| v as u32).collect(),
            self.connectome.col_indices.iter().map(|&v| v as u32).collect(),
            self.connectome.values.clone(),
            vec![0u8; self.connectome.n_synapses],
            if synaptic {
                self.connectome.receptor_types.clone()
            } else {
                vec![0u8; self.connectome.n_synapses]
            },
        );

        let mut poisson_buf = vec![0.0f32; n];
        let mut ext = vec![0.0f32; n_neurons];

        if synaptic {
            // Conductance path — SynapticBrain owns the kinetics loop.
            let mut brain =
                pillars_connectome::SynapticBrain::<pillars::zoo::LIFModel>::from_connectome(
                    &new_conn,
                    dt_ms,
                );
            if let Some(k) = &self.kinetics {
                brain.set_kinetics(&kinetics_legacy_to_pillars(k)); // carry-over
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
                poisson_buf.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut poisson_buf, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + poisson_buf[i];
                }
                let s = brain.step(&ext);
                record_spikes(&s, n_neurons, &mut self.spike_buffer, step);
            }
            self.kinetics = Some(kinetics_pillars_to_legacy(brain.kinetics()));
            let v = brain.state(0);
            let refr = brain.state(1);
            for i in 0..n_neurons {
                self.state.v[i] = v[i];
                self.state.refractory[i] = refr[i];
            }
        } else {
            // Instant-current path — Brain (api::Network) does the routing.
            let mut brain = pillars_connectome::Brain::<pillars::zoo::LIFModel>::from_connectome(
                &new_conn,
            );
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
            let idx: Vec<u32> = (0..n_neurons as u32).collect();
            for step in 0..n_steps {
                poisson_buf.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut poisson_buf, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + poisson_buf[i];
                }
                brain.inject(&idx, &ext);
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
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;

        // Delegate to pillars-connectome (Brain instant / SynapticBrain
        // conductance). v_thresh = 30 matches the legacy hardcoded value.
        let new_conn = pillars_connectome::ConnectomeData::from_csr(
            self.connectome.row_ptr.iter().map(|&v| v as u32).collect(),
            self.connectome.col_indices.iter().map(|&v| v as u32).collect(),
            self.connectome.values.clone(),
            vec![0u8; self.connectome.n_synapses],
            if synaptic {
                self.connectome.receptor_types.clone()
            } else {
                vec![0u8; self.connectome.n_synapses]
            },
        );

        let mut poisson_buf = vec![0.0f32; n];
        let mut ext = vec![0.0f32; n_neurons];

        if synaptic {
            let mut brain =
                pillars_connectome::SynapticBrain::<pillars::zoo::IzhikevichModel>::from_connectome(
                    &new_conn,
                    dt_ms,
                );
            if let Some(k) = &self.kinetics {
                brain.set_kinetics(&kinetics_legacy_to_pillars(k));
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
                poisson_buf.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut poisson_buf, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + poisson_buf[i];
                }
                let s = brain.step(&ext);
                record_spikes(&s, n_neurons, &mut self.spike_buffer, step);
            }
            self.kinetics = Some(kinetics_pillars_to_legacy(brain.kinetics()));
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
                pillars_connectome::Brain::<pillars::zoo::IzhikevichModel>::from_connectome(
                    &new_conn,
                );
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
            let idx: Vec<u32> = (0..n_neurons as u32).collect();
            for step in 0..n_steps {
                poisson_buf.fill(0.0);
                inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut poisson_buf, dt_ms);
                for i in 0..n_neurons {
                    ext[i] = external_I.get(i).copied().unwrap_or(0.0) + poisson_buf[i];
                }
                brain.inject(&idx, &ext);
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
            taas_kernels::safe::izhikevich_dual_step(
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
                None,
            );

            // 3. Push spikes → record
            push_spikes_to_queue(&self.state.spikes, n_neurons, &mut self.state.queue);
            record_spikes(&self.state.spikes, n_neurons, &mut self.spike_buffer, step);

            // 4. Hebbian plasticity
            if hebb_lr > 0.0 && n_synapses > 0 {
                taas_kernels::safe::hebbian_csr_step(
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
            csr_propagate_from_queue(
                &self.connectome.row_ptr,
                &self.connectome.col_indices,
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
            taas_kernels::safe::izhikevich_step(
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
                None,
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
            csr_propagate_from_queue(
                &self.connectome.row_ptr,
                &self.connectome.col_indices,
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

    fn run_hh(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        synaptic: bool,
    ) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        // `record_spikes` strides the buffer by n_neurons — must equal n.
        debug_assert_eq!(n, n_neurons, "spike buffer stride invariant");

        // ─── pillars engine: NeuralPopulation<HHModel> (SIMD) — standard AND pacemaking ───
        let pop = self.hh_pop.as_mut().unwrap();

        // Recreate params if dt changed (state preserved — better than legacy rebuild)
        if (pop.param_slice(HHParamIdx::DT.into())[0] - dt_ms).abs() > 1e-6 {
            pop.fill_param(HHParamIdx::DT.into(), dt_ms);
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

    fn run_rate(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
        tau: f32,
    ) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let dt = dt_ms / 1000.0;
        let decay = (-dt / tau.max(1e-6)).exp();
        let inv_decay = 1.0 - decay;

        for step in 0..n_steps {
            // 1. Set I = external + Poisson
            self.state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
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
            csr_propagate_from_queue(
                &self.connectome.row_ptr,
                &self.connectome.col_indices,
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.state.queue,
                &mut self.state.I_buf,
                1.0,
            );

            // 4. Exponential Euler
            for i in 0..n {
                let ext = if i < external_I.len() { external_I[i] } else { 0.0 };
                let input = (self.state.I_buf[i] + ext).clamp(0.0, 1.0);
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

    fn run_gco(
        &mut self,
        n_steps: usize,
        external_I: &[f32],
        dt_ms: f32,
    ) {
        let n = self.n_neurons;
        let n_neurons = self.connectome.row_ptr.len() - 1;
        let dt = dt_ms / 1000.0;

        let mut h_real = vec![0.0; self.state.n_padded];
        let mut h_imag = vec![0.0; self.state.n_padded];
        let mut speed = vec![1.0; self.state.n_padded];

        for step in 0..n_steps {
            // 1. Build Hamiltonian from CSR
            h_real.fill(0.0);
            h_imag.fill(0.0);
            let max_syn = self.connectome.col_indices.len().min(self.connectome.values.len());
            for pre in 0..n_neurons {
                let start = self.connectome.row_ptr[pre].max(0) as usize;
                let end = (self.connectome.row_ptr[pre + 1] as usize).min(max_syn);
                let mult = if pre < self.connectome.nt_types.len() {
                    self.connectome.nt_types[pre]
                } else {
                    1.0
                };
                for j in start..end {
                    let post = self.connectome.col_indices[j] as usize;
                    let w = self.connectome.values[j] * mult;
                    if post < n {
                        h_real[post] += w * self.state.u[pre];
                        h_imag[post] -= w * self.state.v[pre];
                    }
                }
            }

            // 2. Set external input
            for i in 0..n.min(external_I.len()) {
                self.state.I_buf[i] = external_I[i];
            }
            self.inject_poisson(dt_ms);

            // 3. GCO step
            taas_kernels::safe::gco_step(
                &mut self.state.v,
                &mut self.state.u,
                &mut h_real,
                &mut h_imag,
                &mut self.state.I_buf,
                &mut speed,
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
        let changed: bool = runner.connectome.values.iter().zip(old_weights.iter()).any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "E-Prop update should modify weights");
    }
}
