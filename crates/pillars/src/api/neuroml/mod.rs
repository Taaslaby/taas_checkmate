//! -------------------------------------------------------------------------
//! Filename:    src/neuroml/mod.rs
//! Class:       neuroml
//! Purpose:     NeuroML2 → pillars
//! Description: Compiles NeuroML2 XML straight into pillars runtime.
//!              One call parses XML, builds populations, wires inputs and
//!              returns a ready-to-step Network.
//!              Layering: parse.rs (XML→Ir via roxmltree) | compile.rs
//!              (Ir→Network+stimuli) | mod.rs (public API: load/LoadedNetwork::run).
//!              Scope: izhikevichCell/2007 → IzhikevichModel; iaf* → LIFModel;
//!              adExIaFCell → AdExModel; fitzHughNagumo* → FHNModel;
//!              hhCell (channelDensity Na/K+leak, ionChannelHH/gateHHrates) →
//!              HHModel or ConductancePopulation; expOne/Two/alpha/expThree +
//!              projection → SynapseKinetics; blockingPlasticSynapse → STP;
//!              doubleSynapse → two edges; pulse/sine/ramp/compound/spikeArray/
//!              poissonFiring/transientPoisson/spikeGenerator* → StimulusKind;
//!              gapJunction/electricalProjection → ohmic; linear/silent/graded
//!              + continuousProjection → graded synapses; Simulation/include →
//!              dt/duration/includes. Warnings for unsupported (stdpSynapse, *DL).
//!              Kinetics (ionChannelKS/gateKS/gateFractional) → MarkovPopulation.
//!              Morphology+SWC → MorphoIR → d_lambda discretize → passive/active
//!              cable (Hines, BDF1↔BDF2, Ca2+ Nernst pool, channelPopulation
//!              Binomial noise); GUI bridge: LoadedNetwork::cable / cable_voltage.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use std::fmt;

use crate::api::network::{poisson_spike, Network};
use crate::api::population::PopulationId;

mod compile;
mod discretize;
mod loader;
mod lems;
mod parse;
mod swc;

pub(crate) use parse::CONDUCTANCE_CELSIUS;

/// Default integration step for Izhikevich (ms) — the model's paper uses 0.5.
const DEFAULT_DT_MS: f32 = 0.5;
/// Default simulation duration (ms).
const DEFAULT_DURATION_MS: f32 = 200.0;

/// A NeuroML load/compile failure.
///
/// `pos` carries the 1-based `(row, col)` of an XML syntax error when it is
/// known (from roxmltree), so an IDE/editor can jump straight to the offending
/// line instead of re-parsing the formatted message. `None` for semantic and
/// compile errors (which have no single position).
#[derive(Debug, Clone, PartialEq)]
pub struct NeuromlError {
    /// 1-based `(row, col)` of the XML syntax error, if known.
    pub pos: Option<(u32, u32)>,
    /// Human-readable message, without the "neuroml parse error:" prefix.
    pub msg: String,
}

impl NeuromlError {
    /// A semantic/compile error without a source position.
    pub(crate) fn parse(msg: impl Into<String>) -> Self {
        Self {
            pos: None,
            msg: msg.into(),
        }
    }

    /// An XML syntax error at a known `(row, col)` (both 1-based).
    pub(crate) fn xml(row: u32, col: u32, msg: impl Into<String>) -> Self {
        Self {
            pos: Some((row, col)),
            msg: msg.into(),
        }
    }
}

impl fmt::Display for NeuromlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "neuroml parse error: ")?;
        if let Some((row, col)) = self.pos {
            write!(f, "XML: {row}:{col}: ")?;
        }
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for NeuromlError {}

/// Geometry of one cable compartment — the discretized sub-cylinder of a
/// morphology segment. Public so GUIs/analysis can draw the cell and map
/// per-compartment voltages (see [`Network::cable_voltage`](crate::api::Network::cable_voltage))
/// back onto the tree.
#[derive(Debug, Clone)]
pub struct CableCompartment {
    /// Parent compartment index (`None` for the root / soma).
    pub parent: Option<usize>,
    /// The morphology segment this compartment was discretized from.
    pub segment_id: usize,
    /// Compartment length, µm.
    pub length_um: f32,
    /// Compartment diameter, µm.
    pub diameter_um: f32,
    /// Centre position (µm), interpolated along the parent segment.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Cumulative path distance from the root along the tree (µm), measured
    /// to the compartment's centre (root = half its length; a chain steps by
    /// exactly one compartment length).
    pub path_um: f32,
}

/// The discretized compartment tree of a morphologically-detailed population
/// (a NeuroML `<cell>`): enough to draw the morphology and attach
/// per-compartment voltages to it.
#[derive(Debug, Clone)]
pub struct CableInfo {
    /// Root (soma) compartment index.
    pub root: usize,
    /// All compartments, in BFS topological order (root first).
    pub compartments: Vec<CableCompartment>,
    /// Segment `name` by segment id (NeuroML `name` attribute), for labels.
    pub segment_names: Vec<(usize, String)>,
}

/// What a scheduled input delivers to its targets.
#[derive(Debug, Clone)]
pub enum StimulusKind {
    /// Constant current `amplitude` during `[start_ms, end_ms)`
    /// (`pulseGenerator`).
    Current,
    /// A deterministic list of spike times (ms): one `dt`-wide current pulse of
    /// `amplitude` at each time (`spikeArray`).
    SpikeTimes { times_ms: Vec<f32> },
    /// A homogeneous Poisson train at `rate_hz`: `amplitude` per spike, seeded
    /// and deterministic (`poissonFiringSynapse`).
    Poisson { rate_hz: f32, seed: u32 },
    /// A sinusoidal current `i = amplitude·sin(phase + 2π·(t − start_ms)/period_ms)`
    /// during `[start_ms, end_ms)`, else 0 (`sineGenerator`; `phase` in radians,
    /// `period_ms` in ms).
    Sine { period_ms: f32, phase: f32 },
    /// A linearly ramping current `start_amp → finish_amp` across
    /// `[start_ms, end_ms)` (`rampGenerator`). The `baseline_amp` regions
    /// outside the window are emitted as separate `Current` stimuli by the
    /// compiler.
    Ramp { start_amp: f32, finish_amp: f32 },
}

/// A scheduled external stimulus (from a `pulseGenerator` / `spikeArray` /
/// `poissonFiringSynapse` + `explicitInput`).
#[derive(Debug, Clone)]
pub struct Stimulus {
    /// Target population.
    pub pop: PopulationId,
    /// Target neuron indices within the population.
    pub indices: Vec<u32>,
    /// First active time (ms).
    pub start_ms: f32,
    /// First inactive time (ms).
    pub end_ms: f32,
    /// Injected current (raw pillars units in this slice).
    pub amplitude: f32,
    /// Delivery waveform: constant window, spike times, or Poisson train.
    pub kind: StimulusKind,
}

/// Per-population metadata for GPU backends: which `#[dual_neuron]` CUDA
/// source + kernel to use, and the model's column counts. Populated by the
/// compiler so [`crate::api::network::Network`] (which erases model types)
/// doesn't have to carry it. The `#[dual_neuron]` constants live in
/// `pillars::zoo::neurons` (single source of truth).
#[derive(Debug, Clone)]
pub struct GpuPopMeta {
    /// CUDA C++ source emitted by `#[dual_neuron]` (e.g. `LIF_CUDA_SOURCE`).
    pub source: &'static str,
    /// Kernel entry name (e.g. `"lif_kernel"`).
    pub kernel_name: &'static str,
    /// Number of state columns.
    pub ns: usize,
    /// Number of parameter columns.
    pub np: usize,
    /// Neuron count in the population.
    pub n: usize,
}

/// Target-voltage schedule of a patch-clamp.
#[derive(Debug, Clone, Copy)]
pub enum ClampSchedule {
    /// Plain `voltageClamp`: hold `target_mv` on `[delay, delay+duration]`,
    /// no current outside the window.
    Window { target_mv: f32 },
    /// `voltageClampTriple`: hold `conditioning_mv` before `delay`, step to
    /// `testing_mv` across `[delay, delay+duration]`, then hold `return_mv`
    /// **forever** afterwards. Disabled entirely when `active` is false.
    Triple {
        conditioning_mv: f32,
        testing_mv: f32,
        return_mv: f32,
        active: bool,
    },
}

/// A LEMS `voltageClamp` / `voltageClampTriple` (patch-clamp) applied to a
/// target neuron. Each active step injects
/// `i = weight·(target(t) − V)/series_resistance_mohm` (mV/MΩ = nA, already
/// scaled per population) — a proportional controller pulling V toward the
/// schedule target. `V` is read at the start of the step (semi-implicit, like
/// kinetic synapses), so the clamp sees the membrane before its own injection.
#[derive(Debug, Clone)]
pub struct VoltageClamp {
    /// Target population.
    pub pop: PopulationId,
    /// Target neuron indices within the population.
    pub indices: Vec<u32>,
    /// Target-voltage schedule (see [`ClampSchedule`]).
    pub schedule: ClampSchedule,
    /// Series resistance (MΩ) — the controller gain `weight/series_resistance`.
    pub series_resistance_mohm: f32,
    /// Effective weight (already includes the per-population current scale).
    pub weight: f32,
    /// First active time (ms).
    pub delay_ms: f32,
    /// First inactive time (ms).
    pub duration_ms: f32,
}

/// A compiled NeuroML network, ready to step.
pub struct LoadedNetwork {
    /// The built pillars network (populations + connections).
    pub network: Network,
    /// External stimuli, applied on schedule by [`LoadedNetwork::run`].
    pub inputs: Vec<Stimulus>,
    /// Patch-clamps (V-feedback inputs), applied by [`LoadedNetwork::run`]
    /// before each step (see [`VoltageClamp`]). Empty when the model has none.
    pub clamps: Vec<VoltageClamp>,
    /// Integration step (ms).
    pub dt_ms: f32,
    /// Simulation duration (ms).
    pub duration_ms: f32,
    /// GPU metadata per population (index = `PopulationId`), for CUDA backends.
    pub gpu_pops: Vec<GpuPopMeta>,
    /// Cable geometry per population (index = `PopulationId`); `None` for
    /// point-neuron populations, `Some` for morphologically-detailed `<cell>`s.
    pub cable: Vec<Option<CableInfo>>,
    /// Non-fatal warnings: recognised NeuroML2 elements that were **silently
    /// skipped** (gap junctions, graded synapses, …). Empty = fully supported.
    pub warnings: Vec<String>,
}

impl LoadedNetwork {
    /// Cable geometry of a population, if it is a morphologically-detailed
    /// `<cell>` (a cable population). `None` for point-neuron populations.
    pub fn cable_info(&self, pop: PopulationId) -> Option<&CableInfo> {
        self.cable.get(pop).and_then(|c| c.as_ref())
    }
}

/// Result of a full [`LoadedNetwork::run`].
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Number of integration steps executed.
    pub steps: usize,
    /// Total spike count per population (index = `PopulationId`).
    pub spike_counts: Vec<usize>,
}

impl LoadedNetwork {
    /// Run the full simulation: apply stimuli and voltage clamps on schedule,
    /// step, count spikes.
    pub fn run(&mut self) -> RunResult {
        let steps = (self.duration_ms / self.dt_ms).round().max(1.0) as usize;
        let mut spike_counts = vec![0usize; self.network.num_populations()];
        for s in 0..steps {
            let t = s as f32 * self.dt_ms;
            let injections = self.injections_at(t);
            for (pop, indices, amps) in injections {
                self.network.inject_current(pop, &indices, &amps);
            }
            let clamps = self.clamp_currents(t);
            for (pop, indices, amps) in clamps {
                self.network.inject_current(pop, &indices, &amps);
            }
            self.network.step();
            for (pid, cnt) in spike_counts.iter_mut().enumerate() {
                *cnt += self.network.spikes(pid).iter().filter(|&&x| x != 0).count();
            }
        }
        RunResult {
            steps,
            spike_counts,
        }
    }
}

impl LoadedNetwork {
    /// Current injections due at `t_ms`, for every scheduled input.
    ///
    /// - constant-current windows ([`StimulusKind::Current`]) inject while
    ///   `start_ms <= t < end_ms`;
    /// - deterministic spike arrays ([`StimulusKind::SpikeTimes`]) inject a
    ///   single `dt`-wide pulse at each listed time (quantised to the step
    ///   grid);
    /// - Poisson trains ([`StimulusKind::Poisson`]) roll a seeded deterministic
    ///   pseudo-Poisson draw per (stimulus, target, step): `P(spike) = rate·dt`.
    ///
    /// Currents due from active patch-clamps at `t_ms`, read against the
    /// network's membrane potential at the start of the step.
    ///
    /// A clamp is active on `[delay_ms, delay_ms + duration_ms]` and injects
    /// `weight·(target_mv − V)/series_resistance_mohm` into each target neuron
    /// (LEMS `voltageClamp` semantics). `V` is the first state column of the
    /// target population (the soma compartment for cable populations).
    pub fn clamp_currents(&self, t_ms: f32) -> Vec<(usize, Vec<u32>, Vec<f32>)> {
        let mut out = Vec::new();
        for c in &self.clamps {
            // LEMS window conditions: a plain clamp is active on
            // [delay, delay+duration]; a triple holds conditioning before the
            // window, testing across it, return after (forever), unless off.
            let target = match &c.schedule {
                ClampSchedule::Window { target_mv } => {
                    if t_ms < c.delay_ms || t_ms > c.delay_ms + c.duration_ms {
                        continue;
                    }
                    *target_mv
                }
                ClampSchedule::Triple {
                    conditioning_mv,
                    testing_mv,
                    return_mv,
                    active,
                } => {
                    if !*active {
                        continue;
                    }
                    if t_ms < c.delay_ms {
                        *conditioning_mv
                    } else if t_ms > c.delay_ms + c.duration_ms {
                        *return_mv
                    } else {
                        *testing_mv
                    }
                }
            };
            let vs = self.network.state(c.pop, 0);
            let mut amps = Vec::with_capacity(c.indices.len());
            for &idx in &c.indices {
                let v = vs[idx as usize];
                amps.push(c.weight * (target - v) / c.series_resistance_mohm);
            }
            out.push((c.pop, c.indices.clone(), amps));
        }
        out
    }

    /// Zero-allocation variant of [`clamp_currents`](Self::clamp_currents): borrows indices from
    /// `self`, allocates only the amplitude `Vec<f32>` per active clamp.
    ///
    /// Returns `(population, indices, amplitudes)` where `indices` is a
    /// `&[u32]` slice into the clamp (no clone).
    pub fn clamp_currents_ref(&self, t_ms: f32) -> Vec<(usize, &[u32], Vec<f32>)> {
        let mut out = Vec::new();
        for c in &self.clamps {
            let target = match &c.schedule {
                ClampSchedule::Window { target_mv } => {
                    if t_ms < c.delay_ms || t_ms > c.delay_ms + c.duration_ms {
                        continue;
                    }
                    *target_mv
                }
                ClampSchedule::Triple {
                    conditioning_mv,
                    testing_mv,
                    return_mv,
                    active,
                } => {
                    if !*active {
                        continue;
                    }
                    if t_ms < c.delay_ms {
                        *conditioning_mv
                    } else if t_ms > c.delay_ms + c.duration_ms {
                        *return_mv
                    } else {
                        *testing_mv
                    }
                }
            };
            let vs = self.network.state(c.pop, 0);
            let mut amps = Vec::with_capacity(c.indices.len());
            for &idx in &c.indices {
                let v = vs[idx as usize];
                amps.push(c.weight * (target - v) / c.series_resistance_mohm);
            }
            out.push((c.pop, c.indices.as_slice(), amps));
        }
        out
    }

    /// Returns `(population, indices, amplitudes)` triples ready for
    /// [`Network::inject_current`] (CPU) or its GPU equivalent.
    ///
    /// Allocates a `Vec` per call (amplitudes must be owned). For zero-alloc
    /// iteration over `&[u32]` indices, use [`injections_at_ref`](Self::injections_at_ref).
    pub fn injections_at(&self, t_ms: f32) -> Vec<(usize, Vec<u32>, Vec<f32>)> {
        let mut out = Vec::new();
        let step = (t_ms / self.dt_ms).round().max(0.0) as u64;
        for stim in &self.inputs {
            if t_ms < stim.start_ms || t_ms >= stim.end_ms {
                continue;
            }
            match &stim.kind {
                StimulusKind::Current => {
                    let amps = vec![stim.amplitude; stim.indices.len()];
                    out.push((stim.pop, stim.indices.clone(), amps));
                }
                StimulusKind::SpikeTimes { times_ms } => {
                    let dt = self.dt_ms;
                    if times_ms.iter().any(|&s| s >= t_ms && s < t_ms + dt) {
                        let amps = vec![stim.amplitude; stim.indices.len()];
                        out.push((stim.pop, stim.indices.clone(), amps));
                    }
                }
                StimulusKind::Poisson { rate_hz, seed } => {
                    let p = (rate_hz * self.dt_ms / 1000.0).clamp(0.0, 1.0);
                    if p <= 0.0 {
                        continue;
                    }
                    let mut firing = Vec::new();
                    for (k, &idx) in stim.indices.iter().enumerate() {
                        if poisson_spike(*seed, k as u32, step, p) {
                            firing.push(idx);
                        }
                    }
                    if !firing.is_empty() {
                        let amps = vec![stim.amplitude; firing.len()];
                        out.push((stim.pop, firing, amps));
                    }
                }
                StimulusKind::Sine { period_ms, phase } => {
                    // LEMS sineGenerator: i = A·sin(phase + 2π·(t−delay)/period),
                    // 0 outside the window. `phase` is in radians.
                    let omega = 2.0 * std::f32::consts::PI / period_ms;
                    let arg = *phase + omega * (t_ms - stim.start_ms);
                    let amp = stim.amplitude * arg.sin();
                    let amps = vec![amp; stim.indices.len()];
                    out.push((stim.pop, stim.indices.clone(), amps));
                }
                StimulusKind::Ramp {
                    start_amp,
                    finish_amp,
                } => {
                    // LEMS rampGenerator: linear start → finish across the
                    // window (baseline regions are separate Current stimuli).
                    let span = stim.end_ms - stim.start_ms;
                    let u = if span > 0.0 {
                        ((t_ms - stim.start_ms) / span).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let amp = start_amp + (finish_amp - start_amp) * u;
                    let amps = vec![amp; stim.indices.len()];
                    out.push((stim.pop, stim.indices.clone(), amps));
                }
            }
        }
        out
    }

    /// Zero-allocation variant of [`injections_at`](Self::injections_at): borrows indices from
    /// `self`, allocates only the amplitude `Vec<f32>` per active stimulus.
    ///
    /// Returns `(population, indices, amplitudes)` where `indices` is a
    /// `&[u32]` slice into the stimulus (no clone).
    pub fn injections_at_ref(&self, t_ms: f32) -> Vec<(usize, &[u32], Vec<f32>)> {
        let mut out = Vec::new();
        let step = (t_ms / self.dt_ms).round().max(0.0) as u64;
        for stim in &self.inputs {
            if t_ms < stim.start_ms || t_ms >= stim.end_ms {
                continue;
            }
            match &stim.kind {
                StimulusKind::Current => {
                    let amps = vec![stim.amplitude; stim.indices.len()];
                    out.push((stim.pop, stim.indices.as_slice(), amps));
                }
                StimulusKind::SpikeTimes { times_ms } => {
                    let dt = self.dt_ms;
                    if times_ms.iter().any(|&s| s >= t_ms && s < t_ms + dt) {
                        let amps = vec![stim.amplitude; stim.indices.len()];
                        out.push((stim.pop, stim.indices.as_slice(), amps));
                    }
                }
                StimulusKind::Poisson { rate_hz, seed } => {
                    let p = (rate_hz * self.dt_ms / 1000.0).clamp(0.0, 1.0);
                    if p <= 0.0 {
                        continue;
                    }
                    let mut firing = Vec::new();
                    for (k, &idx) in stim.indices.iter().enumerate() {
                        if poisson_spike(*seed, k as u32, step, p) {
                            firing.push(idx);
                        }
                    }
                    if !firing.is_empty() {
                        let amps = vec![stim.amplitude; firing.len()];
                        out.push((stim.pop, firing.leak(), amps));
                    }
                }
                StimulusKind::Sine { period_ms, phase } => {
                    let omega = 2.0 * std::f32::consts::PI / period_ms;
                    let arg = *phase + omega * (t_ms - stim.start_ms);
                    let amp = stim.amplitude * arg.sin();
                    let amps = vec![amp; stim.indices.len()];
                    out.push((stim.pop, stim.indices.as_slice(), amps));
                }
                StimulusKind::Ramp {
                    start_amp,
                    finish_amp,
                } => {
                    let span = stim.end_ms - stim.start_ms;
                    let u = if span > 0.0 {
                        ((t_ms - stim.start_ms) / span).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let amp = start_amp + (finish_amp - start_amp) * u;
                    let amps = vec![amp; stim.indices.len()];
                    out.push((stim.pop, stim.indices.as_slice(), amps));
                }
            }
        }
        out
    }
}

pub use loader::{load, load_at, load_file, load_file_at, load_swc, load_with};

#[cfg(test)]
mod tests;
