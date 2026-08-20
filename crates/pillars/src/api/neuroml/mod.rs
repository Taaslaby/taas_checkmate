//! # NeuroML2 → pillars (`api::neuroml`)
//!
//! Compiles a **NeuroML2** network description straight into the pillars
//! runtime. One call parses the XML, builds populations, wires inputs and
//! returns a ready-to-step [`Network`]:
//!
//! ```rust,ignore
//! use pillars::api::neuroml;
//! let mut ln = neuroml::load(include_str!("my.nml"))?;
//! let res = ln.run();
//! ```
//!
//! ## Layering
//!
//! ```text
//! parse.rs    NeuroML2 XML → Ir (intermediate representation, roxmltree)
//! compile.rs  Ir → Network + stimulus schedule (Izhikevich 2003)
//! mod.rs      public API: load / LoadedNetwork::run
//! ```
//!
//! ## Scope (documented honestly)
//!
//! * `izhikevichCell` (2003) → `IzhikevichModel`;
//! * `izhikevich2007Cell` (DSN 2007) → `Izhikevich2007Model`;
//! * `iafCell` / `iafTauCell` / `iafTauRefCell` / `iafRefCell` → `LIFModel`;
//! * `adExIaFCell` → `AdExModel`;
//! * `fitzHughNagumoCell` / `fitzHughNagumo1969Cell` → `FHNModel` (bias `I`);
//! * `hhCell` (классический HH через `channelDensity` Na/K + leak) → `HHModel`;
//!   каналы можно объявлять полным NeuroML2 (`ionChannelHH` + `gateHHrates`/
//!   `gateHHtauInf`) — стандартные NaV m³h / KV n⁴ резолвятся бит-в-бит против
//!   движка и запускаются как `HHModel`;
//! * `expOneSynapse` / `expTwoSynapse` / `alphaSynapse` / `expThreeSynapse` +
//!   `<projection>` → synapses; with time constants the waveform is a kinetic
//!   conductance ([`SynapseKinetics`](crate::api::SynapseKinetics), current
//!   `g(t)·(erev − V)`: expOne = single-exponential decay, expTwo =
//!   peak-normalised double-exponential, alpha = alpha function, expThree = two
//!   decay components sharing one rise time), without one — a static delta
//!   synapse (weight · gbase as edge weight, `delay` as axonal delay);
//! * `blockingPlasticSynapse` (с `tsodyksMarkramDepMechanism` /
//!   `tsodyksMarkramDepFacMechanism`) → **short-term plasticity**: the
//!   delivered weight of every spike is scaled by the edge's current
//!   Tsodyks–Markram `R·U` (depression / depression+facilitation, per-edge
//!   state in the [`Network`], see
//!   [`Network::register_plasticity`](crate::api::Network::register_plasticity));
//! * `doubleSynapse` (`synapse1`/`synapse2`) → one connection carrying two
//!   independent synaptic mechanisms (e.g. AMPA + NMDA), lowered to **two
//!   edges** per connection — each child synapse keeps its own gbase, kinetics
//!   and plasticity;
//! * `pulseGenerator` + `explicitInput` → external drive;
//! * `sineGenerator` / `rampGenerator` / `compoundInput` + `explicitInput` →
//!   dynamic currents (LEMS formulas: `i = A·sin(phase + 2π(t−delay)/period)`
//!   for sine — phase in radians; linear `start→finish` ramp, baseline held
//!   outside the window; compound = sum of its child generators);
//! * `timedSynapticInput` + `explicitInput` → deterministic spike-time input
//!   scaled by the referenced synapse's `gbase` (current pulses, like
//!   `spikeArray`);
//! * `voltageClamp` / `voltageClampTriple` + `explicitInput` → **patch clamp**:
//!   each active step injects `weight·(target(t) − V)/seriesResistance` against
//!   the neuron's live membrane potential (proportional controller, see
//!   [`VoltageClamp`] / [`ClampSchedule`] — window clamp, or the three-step
//!   conditioning/testing/return protocol with the return voltage held
//!   indefinitely);
//! * `spikeArray` / `spikeArrayRef` + `explicitInput` → deterministic spike-time
//!   input (one `dt`-wide current pulse per listed time);
//! * `poissonFiringSynapse` + `explicitInput` → seeded deterministic
//!   pseudo-Poisson input (`P(spike) = rate·dt` per step, see
//!   [`StimulusKind::Poisson`]); as the **presynaptic population** of a
//!   `<projection>` its spikes route through synapses (weight × gbase, axonal
//!   delay, optional kinetics) to the postsynaptic population;
//! * `transientPoissonFiringSynapse` + `explicitInput` → the same seeded
//!   pseudo-Poisson input, active only within `[delay, delay+duration]`;
//! * `spikeGenerator` / `spikeGeneratorPoisson` + `explicitInput` → a regular
//!   spike train (first spike at `period`, then every `period`) / a seeded
//!   pseudo-Poisson train;
//! * `spikeGeneratorRandom` / `spikeGeneratorRefPoisson` + `explicitInput` →
//!   seeded ISI trains: uniform `[minISI, maxISI]` intervals, respectively the
//!   maximum-entropy `[minimumISI, ∞)` intervals with mean `1/rate` (the whole
//!   schedule is precomputed from the seed at compile time → deterministic,
//!   stateless runtime);
//! * `gapJunction` + `electricalProjection` → electrical synapses: symmetric
//!   ohmic coupling `I = g·(V_j − V_i)` per step (see
//!   [`Network::connect_gap_junction`](crate::api::Network::connect_gap_junction));
//! * `linearGradedSynapse` / `silentSynapse` / `gradedSynapse` +
//!   `continuousProjection` → analog (graded) synapses: `linearGradedSynapse`
//!   is a **one-way** ohmic coupling `I_post = g·(V_pre − V_post)` per step (see
//!   [`Network::connect_graded_synapse`](crate::api::Network::connect_graded_synapse));
//!   `gradedSynapse` is a **stateful** sigmoid conductance `s` driven by V_pre
//!   (`I = w·g·s·(erev − V)`, see
//!   [`SigmoidGradedSynapse`](crate::api::SigmoidGradedSynapse));
//! * LEMS `<Simulation length step>` → dt/duration (else [`load_with`]);
//! * LEMS `<include href>` (мультифайловые модели) — резолвится в
//!   [`load_file`] относительно пути файла, рекурсивно.
//!
//! Recognised-but-unsupported elements are **not silently dropped**: they
//! surface in [`LoadedNetwork::warnings`] (`stdpSynapse` — в самом NeuroML
//! помечен «EXAMPLE NOT YET WORKING»; dimensionless
//! `pulseGeneratorDL`/`sineGeneratorDL`/`rampGeneratorDL`/`compoundInputDL`).
//! Кинетические схемы
//! (`ionChannelKS`/`gateKS`) **поддерживаются**: парсятся в марковские схемы и
//! запускаются скалярным master-equation движком
//! ([`MarkovPopulation`](crate::api::MarkovPopulation)); `gateFractional`
//! (взвешенная сумма tau-inf субгейтов) опускается до марковской схемы над
//! декартовым произведением состояний субгейтов.
//!
//! Морфология (`morphology`/`segment`/`segmentGroup`/`biophysicalProperties`/
//! `cell`) и SWC-импорт **парсятся** в единый `MorphoIR`, а модуль `discretize`
//! умеет d_λ-дискретизацию, пассивный кабельный солвер (implicit Euler /
//! Hines; топология кэширована, ноль аллокаций на шаг — 1000-компартментный
//! кабель сверен с NEURON на машинную точность), активные HH-каналы,
//! адаптивный шаг с **переменным порядком** (CVODE-аналог: BDF1 ↔ BDF2 через
//! Richardson-экстраполяцию, рестарт порядка на отказе шага) и мост
//! пассивной биофизики
//! (`resistivity`/`specificCapacitance`/`pas` → кабельные параметры).
//! Морфологическая клетка (`<cell>`) **запускается** как кабельная популяция в
//! `load()` → [`Network`]: пассивный кабель + активные каналы (HH и Марков),
//! привязанные к компартментам по `segmentGroup`; ток в сому, спайк на 0 мВ.
//! Плотности бывают и неравномерными (`channelDensityNonUniform` +
//! `<variableParameter>` с `<proximal>`/`<distal>`): плотность линейно
//! интерполируется вдоль пути группы (proximal у корня → distal на конце).
//! Марков-каналы (`ionChannelKS`/`gateKS`) в кабеле сверены с NEURON
//! (`gen_cable_ks_ref.py`, max |ΔV| ≈ 0.014 мВ на ball-and-stick).
//!
//! **Дискретные популяции каналов** (`channelPopulation` — `number` каналов
//! вместо плотности) поддерживаются как **стохастические**: master equation
//! даёт открытую вероятность `p_open`, а открытая доля в каждом шаге —
//! биномиальная выборка `n_open ~ Binomial(number, p_open)`, так что
//! проводимость = `single_channel_conductance · n_open`. Это канонический
//! канальный шум (∝ 1/√N) для мелких нейронов/аксонов; каждый нейрон
//! получает независимый (но воспроизводимый по сиду) поток шума.
//!
//! **Ca²⁺-подсистема** (`channelDensityNernst`/`channelPopulationNernst` +
//! `species` + `decayingPoolConcentrationModel`) поддерживается: обращение
//! потенциала Ca-канала — Nernst от внутриклеточного [Ca²⁺] (пересчитывается
//! каждый шаг, `ca_ext` из `<species>`), а [Ca²⁺] интегрируется пулом-ODE
//! `d[Ca]/dt = iCa/(2F·V_shell) − (Ca−rest)/τ` (сферическая оболочка, точная
//! экспонента). `channelPopulationNernst` — та же семантика как **дискретная
//! популяция** (биномиальный канальный шум `n_open ~ Binomial(N, p_open)`,
//! mean-field ≡ `channelDensityNernst`). Сверено с NEURON (`gen_ca_ref.py`:
//! `cachan_simple.mod` + `cacum.mod`): max |ΔV| ≈ 0.08 мВ, max rel |ΔCa| ≈
//! 0.9% (единственная расходимость — тонкая оболочка NEURON vs сферическая
//! LEMS, ~1%); mean-field стохастики — до <0.001 мВ / 0.3% от детерминистики.
//!
//! **Публичный мост в GUI/анализ:** геометрия дерева доступна через
//! [`LoadedNetwork::cable`]/[`LoadedNetwork::cable_info`] ([`CableInfo`] —
//! компартменты с родителями, позициями и path-дистанцией, имена сегментов),
//! а живой покомпартментный потенциал — через
//! [`Network::cable_voltage`](crate::api::Network::cable_voltage) (зеркало
//! `v[neuron][comp]`, обновляется каждый шаг). Connscope (neoMyelin) рисует по
//! нему окно «Морфология»: дерево + клик + V(t) по компартментам.
//!
//! Произвольные нестандартные каналы **запускаются**: `ionChannelHH`/
//! `gateHHrates`/`gateHHtauInf` парсятся в data-driven
//! [`ChannelSpec`]-подобный IR (round-trip бит-в-бит против `na_m_alpha` и
//! т.д.); стандартные m³h/n⁴ идут быстрым путём через `HHModel`, а
//! нестандартные наборы — через
//! [`ConductancePopulation`](crate::api::ConductancePopulation) (generic-движок).
//! Кастомные LEMS-`ComponentType` для гейтов (реальные OSB-модели, например
//! Ih Poirazi 2003 `Poirazi_h_tau`) тоже парсятся: `timeCourse`/`steadyState`
//! с нестандартным `type` нижются в [`LemsExpr`] (арифметика/`exp`/piecewise
//! с `Case`-условиями), параметры биндятся из атрибутов гейта.
//! Биофизическая калибровка единиц (time→ms, voltage→mV, `nA`→внутренний
//! ток) выполнена.
//!
//! [`LemsExpr`]: crate::kernels::hh_channels::LemsExpr
//!
//! [`ChannelSpec`]: crate::kernels::hh_channels::ChannelSpec
//!
//! [`Network`]: crate::api::network::Network

use std::fmt;
use std::path::Path;

use crate::api::network::{poisson_spike, Network};
use crate::api::population::PopulationId;

mod compile;
mod discretize;
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
            for (pop, indices, amps) in self.injections_at(t) {
                self.network.inject_current(pop, &indices, &amps);
            }
            for (pop, indices, amps) in self.clamp_currents(t) {
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
            let mut firing = Vec::with_capacity(c.indices.len());
            let mut amps = Vec::with_capacity(c.indices.len());
            for &idx in &c.indices {
                let v = vs[idx as usize];
                firing.push(idx);
                amps.push(c.weight * (target - v) / c.series_resistance_mohm);
            }
            out.push((c.pop, firing, amps));
        }
        out
    }

    /// Returns `(population, indices, amplitudes)` triples ready for
    /// [`Network::inject_current`] (CPU) or its GPU equivalent.
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
}

/// Parse and compile a NeuroML2 document.
///
/// `dt`/`duration` come from a LEMS `<Simulation>` if present, otherwise from
/// the Izhikevich defaults (`0.5 ms` / `200 ms`).
pub fn load(xml: &str) -> Result<LoadedNetwork, NeuromlError> {
    load_at(xml, DEFAULT_DT_MS, DEFAULT_DURATION_MS, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document with an explicit `dt`/`duration` (ms)
/// and simulation temperature `celsius` (°C, Q10 anchor).
pub fn load_at(
    xml: &str,
    dt_ms: f32,
    duration_ms: f32,
    celsius: f32,
) -> Result<LoadedNetwork, NeuromlError> {
    let ir = parse::parse(xml)?;
    compile::compile(ir, dt_ms, duration_ms, celsius)
}

/// Parse and compile a NeuroML2 document with an explicit `dt`/`duration` (ms).
/// Temperature defaults to [`CONDUCTANCE_CELSIUS`] (6.3 °C, Q10 = 1).
pub fn load_with(xml: &str, dt_ms: f32, duration_ms: f32) -> Result<LoadedNetwork, NeuromlError> {
    load_at(xml, dt_ms, duration_ms, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document **read from disk**, resolving LEMS
/// `<include href="…"/>` elements relative to the file's directory
/// (recursively, depth-limited). This lets split/multi-file models load — the
/// common case for real NeuroML files that pull in shared component types.
///
/// `dt`/`duration` come from a LEMS `<Simulation>` if present, otherwise from
/// the Izhikevich defaults (`0.5 ms` / `200 ms`).
pub fn load_file(path: &Path) -> Result<LoadedNetwork, NeuromlError> {
    load_file_at(path, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document **read from disk**, at the given
/// simulation temperature `celsius` (°C). See [`load_file`] for the include
/// resolution semantics.
pub fn load_file_at(path: &Path, celsius: f32) -> Result<LoadedNetwork, NeuromlError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| NeuromlError::parse(format!("cannot read {}: {e}", path.display())))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let merged = resolve_includes(&xml, base, 0)?;
    let ir = parse::parse(&merged)?;
    let (dt, duration) = match &ir.sim {
        Some(s) => (s.dt_ms, s.duration_ms),
        None => (DEFAULT_DT_MS, DEFAULT_DURATION_MS),
    };
    compile::compile(ir, dt, duration, celsius)
}

/// Inline LEMS `<include href="…"/>` elements, resolving each `href` relative
/// to `base` and recursing into the included file (depth-limited). XML
/// declarations in included files are stripped so concatenation stays valid.
/// Our parser walks all descendants, so nested `<neuroml>` roots from includes
/// are harmless — matching tags are still found.
fn resolve_includes(xml: &str, base: &Path, depth: usize) -> Result<String, NeuromlError> {
    if depth > 8 {
        return Err(NeuromlError::parse("LEMS include depth exceeded 8"));
    }
    let mut out = String::with_capacity(xml.len());
    let rest = xml;
    let mut search_from = 0usize;
    loop {
        // Case-insensitive `<include` (NeuroML uses lowercase; be lenient).
        // NOTE: ищем по оригиналу байт-за-байтом (eq_ignore_ascii_case), а не
        // через to_lowercase().find() — lowercasing меняет длину строки для
        // не-ASCII (кириллица в комментариях), и байтовые индексы разъезжаются.
        let start = find_ascii_case_insensitive(rest, b"<include", search_from);
        let Some(start) = start else { break };
        out.push_str(&rest[search_from..start]);
        let after = &rest[start..];
        let end = after
            .find('>')
            .ok_or_else(|| NeuromlError::parse("unterminated <include> element"))?;
        let elem = &after[..=end];
        let Some(href) = extract_href(elem) else {
            // `<include segmentGroup="…"/>` — наследование сегмент-групп в
            // морфологии, а не файловый инклюд: пропускаем как есть.
            out.push_str(elem);
            search_from = start + end + 1;
            continue;
        };
        // jNeuroML-экспорт пишет абсолютные пути вида `/channels/foo.nml` —
        // это «относительно каталога модели», а не корня диска. Срезаем
        // ведущий слэш, чтобы `base.join` дал правильный путь.
        let href = href.trim_start_matches(['/', '\\']);
        let full = base.join(href);
        let included = std::fs::read_to_string(&full).map_err(|e| {
            NeuromlError::parse(format!("cannot read include {}: {e}", full.display()))
        })?;
        let inc_base = full.parent().unwrap_or(base);
        let inc_merged = resolve_includes(&included, inc_base, depth + 1)?;
        // Стрём XML-объявление у включаемого файла (иначе конкатенация битая).
        let inc_merged = strip_xml_decl(&inc_merged);
        out.push_str(&inc_merged);
        out.push('\n');
        search_from = start + end + 1;
    }
    out.push_str(&rest[search_from..]);
    Ok(out)
}

/// Find `needle` (ASCII) in `haystack` starting at `from`, case-insensitively,
/// returning a **byte index into the original** `haystack` (no re-encoding).
fn find_ascii_case_insensitive(haystack: &str, needle: &[u8], from: usize) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.len();
    if n == 0 || from > h.len() {
        return None;
    }
    let mut i = from;
    while i + n <= h.len() {
        if h[i..i + n].eq_ignore_ascii_case(needle) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Pull `href="…"` out of an `<include …/>` element.
fn extract_href(elem: &str) -> Option<&str> {
    let s = elem.find("href=\"")? + "href=\"".len();
    let e = elem[s..].find('"')? + s;
    Some(&elem[s..e])
}

/// Drop a leading `<?xml …?>` declaration if present.
fn strip_xml_decl(xml: &str) -> String {
    let t = xml.trim_start();
    if let Some(rest) = t.strip_prefix("<?xml") {
        if let Some(end) = rest.find("?>") {
            return rest[end + 2..].trim_start().to_string();
        }
    }
    xml.to_string()
}

#[cfg(test)]
mod tests;
