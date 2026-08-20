//! # NeuroML2 parser (`api::neuroml::parse`)
//!
//! Turns a NeuroML2 (and/or LEMS) XML document into a small **intermediate
//! representation** ([`Ir`]) that is independent of both XML and the pillars
//! runtime. The compiler ([`super::compile`]) lowers this IR into a [`Network`].
//!
//! ## Supported subset
//!
//! ```text
//! <neuroml>
//!   <izhikevichCell .../> <iafCell/> <iafTauCell/> <iafTauRefCell/> <adExIaFCell/>
//!   <expOneSynapse/> <expTwoSynapse/> <pulseGenerator/>
//!   <poissonFiringSynapse/> <spikeArray/> <spikeArrayRef/>
//!   <gapJunction/> <electricalProjection> <electricalConnection/> </electricalProjection>
//!   <linearGradedSynapse/> <silentSynapse/> <gradedSynapse/>
//!   <continuousProjection/> <continuousConnection/>
//!   <network>
//!     <population .../> <projection> <connection/> </projection>
//!     <explicitInput .../>
//!   </network>
//! </neuroml>
//! <Lems> <Simulation length step/> </Lems>          ← optional dt/duration
//! ```
//!
//! ## Units (honest contract)
//!
//! * **time** (`delay`, `duration`, `tau`, `refract`, `tauw`, `length`, `step`)
//!   → **ms** (`s` ×1000);
//! * **voltage** (`v0`, `thresh`, `reset`, `leakReversal`, `EL`, `VT`, `c`) →
//!   **mV** (`V` ×1000);
//! * **current** (`amplitude`, `b`, synapse `gbase`) and **conductance** (`a`,
//!   `gL`, `leakConductance`) are kept **raw** (unit stripped). This matches the
//!   common NeuroML convention where `nA`/`nS` read as pillars' own `nA`/`nS`
//!   units; a true SI→internal current calibration is a later step.
//! * `iafCell`/`adExIaFCell` `tau_m = C / gL` is computed from the raw `pF`/`nS`
//!   numbers, which yields ms directly (`1 pF / 1 nS = 1 ms`).
//!
//! [`Network`]: crate::api::network::Network

use std::collections::{HashMap, HashSet};

use super::lems::{parse_component_type, ComponentType};
use super::NeuromlError;
use crate::kernels::hh_channels::{
    fractional_gate_to_markov, gate_kinetics_to_markov, ClosedForm, FractionalGateSpec,
    FractionalSubgate, GateKinetics, LemsExpr, MarkovGateSpec, MarkovRate, MarkovTransition,
    RateForm, TauForm, VarForm,
};

// ─── Intermediate representation ───────────────────────────────
/// The parsed NeuroML2/LEMS document.
pub(crate) struct Ir {
    pub cells: Vec<Cell>,
    pub synapses: Vec<Synapse>,
    pub double_synapses: Vec<DoubleSynapseSpec>,
    pub pulse_gens: Vec<PulseGen>,
    pub poisson_synapses: Vec<PoissonSpec>,
    pub transient_poisson_synapses: Vec<TransientPoissonSpec>,
    pub spike_gens: Vec<SpikeGenSpec>,
    pub spike_arrays: Vec<SpikeArray>,
    pub spike_array_refs: Vec<SpikeArrayRef>,
    pub sine_gens: Vec<SineGenSpec>,
    pub ramp_gens: Vec<RampGenSpec>,
    pub compounds: Vec<CompoundSpec>,
    pub timed_syn_inputs: Vec<TimedSynInputSpec>,
    pub voltage_clamps: Vec<VoltageClampSpec>,
    pub voltage_clamp_triples: Vec<VoltageClampTripleSpec>,
    pub pops: Vec<PopSpec>,
    pub projections: Vec<ProjectionSpec>,
    pub inputs: Vec<InputSpec>,
    pub gap_junctions: Vec<GapJunctionSpec>,
    pub electrical_projections: Vec<ElectricalProjectionSpec>,
    pub graded_synapses: Vec<GradedSynapseSpec>,
    pub sigmoid_graded_synapses: Vec<SigmoidGradedSynapseSpec>,
    pub continuous_projections: Vec<ContinuousProjectionSpec>,
    /// Parsed `ionChannel` / `ionChannelHH` components (arbitrary HH gates).
    pub channels: Vec<IrChannel>,
    /// Parsed `ionChannelKS` components (kinetic-scheme / Markov gates).
    pub ks_channels: Vec<IrKsChannel>,
    /// Parsed `morphology` components (segments, groups).
    pub morphologies: Vec<MorphoIR>,
    /// Parsed `biophysicalProperties` (keyed by their `id`).
    pub biophysics: HashMap<String, BiophysicalProperties>,
    /// Parsed `concentrationModel` components (keyed by their `id`).
    pub concentration_models: HashMap<String, IrConcentrationModel>,
    /// Parsed `<cell>` elements (morphologically-detailed).
    pub morpho_cells: Vec<MorphoCell>,
    /// Optional LEMS `<Simulation>` config.
    pub sim: Option<SimConfig>,
    /// Non-fatal warnings: recognised NeuroML2 elements that we **silently
    /// skip** (gap junctions, graded synapses, …). Surfaced to callers so a
    /// model that "loads" but lost connectivity is never silently wrong.
    pub warnings: Vec<String>,
}

/// A cell (one of the supported point-neuron types).
pub(crate) struct Cell {
    pub id: String,
    pub spec: CellSpec,
}

/// The concrete cell model + its NeuroML parameters.
pub(crate) enum CellSpec {
    /// `izhikevichCell` (2003): `a, b, c, d, thresh, v0`.
    Izh {
        v0: f32,
        thresh: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    },
    /// `izhikevich2007Cell` (DSN 2007, eq. 5.2): `C, k, vr, vt, vpeak,
    /// a, b, c, d, v0`.
    Izh2007 {
        v0: f32,
        cap: f32,
        k: f32,
        vr: f32,
        vt: f32,
        vpeak: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    },
    /// `iafCell` / `iafTauCell` / `iafTauRefCell` / `iafRefCell` — unified
    /// into one LIF. Pillars LIF is voltage-normalised (`i_syn` in mV = I/gL),
    /// so `current_scale` (mV per nA) rescales injected currents: `1000/gL` for
    /// `iafCell` (has C/gL), `1.0` for `iafTau*` (current already in mV units).
    Iaf {
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ref: f32,
        current_scale: f32,
    },
    /// `fitzHughNagumoCell` / `fitzHughNagumo1969Cell` — unified FHN.
    /// `bias` = встроенный стимул `I` (постоянный ток, не из pulseGenerator).
    Fhn {
        a: f32,
        b: f32,
        inv_tau_w: f32,
        v_thresh: f32,
        bias: f32,
        v0: f32,
        w0: f32,
    },
    /// `adExIaFCell` (Brette–Gerstner 2005).
    ///
    /// Pillars' AdEx is **voltage-normalised** (everything divided by the leak
    /// conductance `gL`): `dv = (v_rest - v + ΔT·exp(...) - w + I)·inv_τm·dt`.
    /// So `a` (nS→ratio) and `b` (nA→mV) are converted **at parse time**, and
    /// `current_scale` (mV per nA = 1000/gL) rescales injected nA→mV.
    AdEx {
        tau_m: f32,
        tau_w: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        delta_t: f32,
        /// Normalised adaptation conductance `a/gL` (dimensionless).
        a: f32,
        /// Normalised spike-triggered jump `b/gL` in **mV**.
        b: f32,
        tau_ref: f32,
        v_peak: f32,
        /// mV per nA (`1000/gL`) — multiply injected currents by this.
        current_scale: f32,
    },
    /// `hhCell` — классический Hodgkin–Huxley через `channelDensity`
    /// (стандартные гейты m³h / n⁴, как в pillars `HHModel`): Na/K/leak.
    Hh {
        v0: f32,
        g_na: f32,
        e_na: f32,
        g_k: f32,
        e_k: f32,
        g_leak: f32,
        e_leak: f32,
        /// 1 / C (мембранная ёмкость, uF).
        inv_cm: f32,
    },
    /// `hhCell` с произвольным набором каналов (не только стандартные m³h/n⁴).
    /// Запускается generic-движком [`crate::api::ConductancePopulation`]; если
    /// есть хотя бы один кинетический (`ionChannelKS`) канал — скалярным
    /// [`crate::api::MarkovPopulation`], который умеет гонять и HH-гейты
    /// (опущенные до 2-состоянийного Маркова).
    ConductanceBased {
        v0: f32,
        /// Мембранная ёмкость (uF) — generic-движок делит ток на `C_m`.
        c_m: f32,
        g_leak: f32,
        e_leak: f32,
        channels: Vec<ResolvedChannel>,
        /// Кинетические (`ionChannelKS`) каналы; пусто = чистый HH-набор.
        markov_channels: Vec<ResolvedMarkovChannel>,
    },
}

/// A synaptic component (`expOneSynapse` / `expTwoSynapse` / `alphaSynapse` /
/// `expThreeSynapse` / `blockingPlasticSynapse`).
pub(crate) struct Synapse {
    pub id: String,
    /// Baseline conductance (`gbase`) — used by expOne/expTwo/alpha; `None` →
    /// 1.0. expThree carries `gbase1`/`gbase2` inside [`SynapseKind`].
    pub gbase: Option<f32>,
    /// `erev` (mV) — reversal potential; `None` → 0 mV (NeuroML default).
    pub erev: Option<f32>,
    pub kind: SynapseKind,
    /// Short-term plasticity (`blockingPlasticSynapse` child mechanism).
    pub plast: Option<PlasticitySpec>,
}

/// A `doubleSynapse` — one connection carrying two independent synaptic
/// mechanisms (e.g. AMPA + NMDA). Lowered by the compiler to **two edges** per
/// connection, one per child synapse (each with its own gbase/kinetics/
/// plasticity).
pub(crate) struct DoubleSynapseSpec {
    pub id: String,
    pub synapse1: String,
    pub synapse2: String,
}

/// A `blockingPlasticSynapse` child plasticity mechanism.
pub(crate) enum PlasticitySpec {
    /// `tsodyksMarkramDepMechanism` — depression-only.
    TsodyksDep {
        init_release_prob: f32,
        tau_rec: f32,
    },
    /// `tsodyksMarkramDepFacMechanism` — depression + facilitation.
    TsodyksDepFac {
        init_release_prob: f32,
        tau_rec: f32,
        tau_fac: f32,
    },
}

/// The parsed waveform kind of a [`Synapse`].
pub(crate) enum SynapseKind {
    /// `expOneSynapse`: instantaneous rise + exponential decay. `tau_decay =
    /// None` → a static delta synapse (no kinetics).
    ExpOne { tau_decay: Option<f32> },
    /// `expTwoSynapse`: exponential rise + decay. Leniently, a missing
    /// `tauRise`/`tauDecay` degrades to expOne / static (legacy models).
    ExpTwo {
        tau_rise: Option<f32>,
        tau_decay: Option<f32>,
    },
    /// `alphaSynapse`: alpha function, rise = decay = `tau`.
    Alpha { tau: f32 },
    /// `expThreeSynapse`: two decay components sharing one rise time.
    ExpThree {
        tau_rise: f32,
        tau_decay1: f32,
        tau_decay2: f32,
        gbase1: f32,
        gbase2: f32,
    },
}

impl Synapse {
    /// Per-connection weight scale (`gbase`). expThree scales `weight` inside
    /// the kinetic profile (its `gbase1`/`gbase2` are baked in), so the
    /// connection weight is passed through unscaled.
    pub fn weight_scale(&self) -> f32 {
        match self.kind {
            SynapseKind::ExpThree { .. } => 1.0,
            _ => self.gbase.unwrap_or(1.0),
        }
    }
}

/// A pulse generator (`pulseGenerator`).
pub(crate) struct PulseGen {
    pub id: String,
    pub delay_ms: f32,
    pub duration_ms: f32,
    pub amplitude: f32,
}

/// A Poisson spike source (`poissonFiringSynapse`) — homogeneous Poisson train
/// at `rate_hz` (Hz). Lowered to a seeded, deterministic pseudo-Poisson spike
/// schedule (see [`super::mod::StimulusKind::Poisson`]).
pub(crate) struct PoissonSpec {
    pub id: String,
    pub rate_hz: f32,
}

/// `transientPoissonFiringSynapse` — a homogeneous Poisson train at `rate_hz`
/// (Hz), active only within `[delay_ms, delay_ms + duration_ms]`.
pub(crate) struct TransientPoissonSpec {
    pub id: String,
    pub rate_hz: f32,
    pub delay_ms: f32,
    pub duration_ms: f32,
}

/// A LEMS `sineGenerator`: `i = weight·amplitude·sin(phase + 2π·(t−delay)/period)`
/// within `[delay_ms, delay_ms + duration_ms)` (phase in radians), else 0.
pub(crate) struct SineGenSpec {
    pub id: String,
    pub phase: f32,
    pub delay_ms: f32,
    pub duration_ms: f32,
    pub amplitude: f32,
    pub period_ms: f32,
    pub weight: f32,
}

/// A LEMS `rampGenerator`: `i = weight·baselineAmplitude` before `delay_ms`;
/// linearly `startAmplitude → finishAmplitude` within the window; back to
/// `weight·baselineAmplitude` afterwards.
pub(crate) struct RampGenSpec {
    pub id: String,
    pub delay_ms: f32,
    pub duration_ms: f32,
    pub start_amp: f32,
    pub finish_amp: f32,
    pub baseline_amp: f32,
    pub weight: f32,
}

/// A LEMS `compoundInput`: `i = weight · Σ children[i]` — children are other
/// `basePointCurrent` generators (pulse/sine/ramp), referenced by id.
pub(crate) struct CompoundSpec {
    pub id: String,
    pub weight: f32,
    /// Child generator ids, in document order.
    pub children: Vec<String>,
}

/// A LEMS `timedSynapticInput`: a spike array (`<spike time>` children) that
/// fires the referenced `synapse` at each listed time. Lowered to current
/// pulses of `weight·gbase` at each time (see the compiler).
pub(crate) struct TimedSynInputSpec {
    pub id: String,
    pub synapse: String,
    pub weight: f32,
    pub times_ms: Vec<f32>,
}

/// A LEMS `voltageClamp` (patch-clamp): while `[delay_ms, delay_ms +
/// duration_ms]` it injects `i = weight·(targetVoltage − v)/simpleSeriesResistance`
/// — a proportional controller that pulls the membrane toward the target.
pub(crate) struct VoltageClampSpec {
    pub id: String,
    pub target_mv: f32,
    /// `simpleSeriesResistance` in **MΩ** (mV/MΩ = nA — pillars raw current
    /// units, scaled per population by the compiler).
    pub series_resistance_mohm: f32,
    pub delay_ms: f32,
    pub duration_ms: f32,
    pub weight: f32,
}

/// A LEMS `voltageClampTriple` — three clamp steps: `conditioning_mv` before
/// `delay_ms`, `testing_mv` across `[delay_ms, delay_ms + duration_ms]`, and
/// `return_mv` held afterwards (forever). `active = 0` disables the clamp.
pub(crate) struct VoltageClampTripleSpec {
    pub id: String,
    pub active: bool,
    pub conditioning_mv: f32,
    pub testing_mv: f32,
    pub return_mv: f32,
    pub series_resistance_mohm: f32,
    pub delay_ms: f32,
    pub duration_ms: f32,
    pub weight: f32,
}

/// A LEMS spike source (`spikeGenerator` / `spikeGeneratorPoisson` /
/// `spikeGeneratorRandom` / `spikeGeneratorRefPoisson`).
pub(crate) enum SpikeGenKind {
    /// `spikeGenerator`: a regular train, first spike at `period_ms`, then
    /// every `period_ms`.
    Regular { period_ms: f32 },
    /// `spikeGeneratorPoisson`: exponential-ISI Poisson at `rate_hz`, lowered
    /// to the same seeded Bernoulli-per-step schedule as
    /// [`super::mod::StimulusKind::Poisson`].
    Poisson { rate_hz: f32 },
    /// `spikeGeneratorRandom`: uniform interspike interval in
    /// `[min_isi_ms, max_isi_ms]` (LEMS: `isi = minISI + MSEC·random((maxISI−
    /// minISI)/MSEC)`, each next spike `isi` after the last planned one).
    Random { min_isi_ms: f32, max_isi_ms: f32 },
    /// `spikeGeneratorRefPoisson`: maximum-entropy ISI over
    /// `[min_isi_ms, ∞)` with mean `1/rate_hz` (LEMS: `isi = minimumISI −
    /// (averageIsi−minimumISI)·log(random(1))`).
    RefPoisson { rate_hz: f32, min_isi_ms: f32 },
}

pub(crate) struct SpikeGenSpec {
    pub id: String,
    pub kind: SpikeGenKind,
}

/// A deterministic spike-time list (`spikeArray`) — a set of `<spike>` events
/// at explicit times (ms).
pub(crate) struct SpikeArray {
    pub id: String,
    pub times_ms: Vec<f32>,
}

/// A `spikeArrayRef` — an indirection to an existing [`SpikeArray`] by id.
pub(crate) struct SpikeArrayRef {
    pub id: String,
    pub array: String,
}

/// A population of cells (`population`).
pub(crate) struct PopSpec {
    pub id: String,
    /// The `component` attribute — id of the referenced cell.
    pub component: String,
    pub size: usize,
}

/// A projection of synapses between two populations.
pub(crate) struct ProjectionSpec {
    pub pre_pop: String,
    pub post_pop: String,
    /// Synapse id (weight scaling / kinetics), if declared.
    pub synapse: Option<String>,
    pub connections: Vec<Connection>,
}

/// A single synaptic connection.
pub(crate) struct Connection {
    pub pre_idx: usize,
    pub post_idx: usize,
    pub weight: f32,
    pub delay_ms: f32,
}

/// An explicit input binding a pulse generator to a single neuron.
pub(crate) struct InputSpec {
    pub target_pop: String,
    pub target_idx: usize,
    pub generator: String,
}

/// A gap junction component (`gapJunction`).
pub(crate) struct GapJunctionSpec {
    pub id: String,
    /// `conductance` (nS) — `None` → 1.0 (unit coupling, raw).
    pub conductance: Option<f32>,
}

/// An electrical projection (`electricalProjection`) of gap junctions.
pub(crate) struct ElectricalProjectionSpec {
    pub pre_pop: String,
    pub post_pop: String,
    /// Referenced `gapJunction` id (the conductance source), if declared.
    pub gap_junction: Option<String>,
    pub connections: Vec<ElectricalConnection>,
}

/// A single electrical connection (a pre ↔ post cell pair).
pub(crate) struct ElectricalConnection {
    pub pre_idx: usize,
    pub post_idx: usize,
    pub weight: f32,
}

/// A graded (analog) synapse component (`linearGradedSynapse` / `silentSynapse`).
pub(crate) struct GradedSynapseSpec {
    pub id: String,
    /// `conductance` (nS) — `None` → 1.0 (unit coupling); `silentSynapse` → 0.
    pub conductance: Option<f32>,
}

/// A sigmoid `gradedSynapse` component (stateful analog conductance).
pub(crate) struct SigmoidGradedSynapseSpec {
    pub id: String,
    pub conductance: f32,
    /// `delta` (mV) — slope of the activation curve.
    pub delta: f32,
    /// `k` (per ms) — transmitter–receptor dissociation rate.
    pub k: f32,
    /// `Vth` (mV) — half-activation voltage.
    pub vth: f32,
    /// `erev` (mV) — reversal potential.
    pub erev: f32,
}

/// A continuous projection (`continuousProjection`) of graded synapses.
pub(crate) struct ContinuousProjectionSpec {
    pub pre_pop: String,
    pub post_pop: String,
    /// Referenced graded-synapse id (the conductance source), if declared.
    pub synapse: Option<String>,
    pub connections: Vec<ContinuousConnection>,
}

/// A single continuous (analog) connection (a pre → post cell pair).
pub(crate) struct ContinuousConnection {
    pub pre_idx: usize,
    pub post_idx: usize,
    pub weight: f32,
}

/// LEMS simulation configuration (dt + duration).
pub(crate) struct SimConfig {
    pub dt_ms: f32,
    pub duration_ms: f32,
}

// ─── Ion channels (ionChannel / ionChannelHH) ─────────────────
/// A parsed conductance-based ion channel. Gates are stored in the data-driven
/// [`GateKinetics`] representation from `kernels::hh_channels` (the same types
/// the generic engine `hh_poly_step` consumes).
pub(crate) struct IrChannel {
    pub id: String,
    /// `conductance` attribute (single-channel conductance, µS), if present.
    /// Used by `channelPopulation` (a population of discrete channels);
    /// `channelDensity` ignores it (density comes from `condDensity`).
    pub conductance: Option<f32>,
    /// HH-гейты (`gateHHrates`/`gateHHtauInf`/…).
    pub gates: Vec<IrGate>,
    /// Дробные гейты (`gateFractional`) — не влезают в [`GateKinetics`]
    /// (взвешенная сумма субгейтов), поэтому отдельным списком.
    pub fractional_gates: Vec<FractionalGateSpec>,
}

/// A single gating variable (e.g. Na-m, Na-h, K-n) inside a channel.
pub(crate) struct IrGate {
    pub id: String,
    /// Exponent in the conductance term (`instances` attribute).
    pub instances: u32,
    /// Runtime kinetics for `gateHHrates`/`gateHHtauInf`; `None` for recognised
    /// but not-yet-compilable gate kinds (`gateHHInstantaneous`, `gateHHratesTau`,
    /// `gateHHratesInf`, `gateHHratesTauInf`).
    pub kinetics: Option<GateKinetics>,
    /// NeuroML gate tag for honest reporting.
    pub kind: &'static str,
    /// `q10Settings` — parsed but not yet applied by the runtime.
    pub q10: Option<IrQ10>,
}

/// Temperature scaling of gate time courses (`q10Settings`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum IrQ10 {
    /// `q10Fixed`: fixed Q10 — rate multiplier `fixedQ10^((T - 6.3)/10)`
    /// (6.3 °C is the NEURON squid-axon reference, same as
    /// [`CONDUCTANCE_CELSIUS`](crate::api::neuroml::compile::CONDUCTANCE_CELSIUS)).
    Fixed(f32),
    /// `q10ExpTemp`: `q10 = q10Factor^((T - experimentalTemp)/10)`.
    ExpTemp { factor: f32, experimental_temp: f32 },
}

/// Default simulation temperature for NeuroML compilation. 6.3 °C is the
/// squid-axon reference at which Q10 = 1 (NEURON convention); per-gate
/// `q10Settings` are anchored to it (`q10Fixed`) or to their own
/// `experimentalTemp` (`q10ExpTemp`).
pub(crate) const CONDUCTANCE_CELSIUS: f32 = 6.3;

impl IrQ10 {
    /// The rate multiplier at simulation temperature `celsius` (°C).
    ///
    /// Matches jNeuroML's NEURON export: `q10Fixed` anchors at 6.3 °C,
    /// `q10ExpTemp` at its `experimentalTemp`. Both give 1.0 at the anchor.
    pub(crate) fn factor_at(self, celsius: f32) -> f32 {
        match self {
            IrQ10::Fixed(q) => q.powf((celsius - 6.3) / 10.0),
            IrQ10::ExpTemp {
                factor,
                experimental_temp,
            } => factor.powf((celsius - experimental_temp) / 10.0),
        }
    }
}

/// A fully-resolved conductance-based channel — gates + reversal + density —
/// ready for the generic engine. Produced when an `hhCell`'s `channelDensity`
/// references an `ionChannelHH` (or a standard Na/K that must join a mixed
/// cell on the generic engine).
pub(crate) struct ResolvedChannel {
    /// `(kinetics, power)` per gating variable, in declaration order.
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// Reversal potential (mV) — from `channelDensity` `erev`.
    pub e_rev: f32,
    /// Max conductance (mS/cm²) — from `channelDensity` `condDensity`.
    pub g_max: f32,
}

/// A fully-resolved `ionChannelKS` (kinetic-scheme) channel — its Markov gates
/// + reversal + density — ready for [`crate::api::MarkovPopulation`].
pub(crate) struct ResolvedMarkovChannel {
    /// Markov gates in declaration order.
    pub gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// Reversal potential (mV) — from `channelDensity` `erev`.
    pub e_rev: f32,
    /// Max conductance (mS/cm²) — from `channelDensity` `condDensity`.
    pub g_max: f32,
}

/// A parsed `ionChannelKS` — kinetic-scheme channel with `gateKS` children.
pub(crate) struct IrKsChannel {
    pub id: String,
    /// `conductance` attribute (single-channel conductance, µS), if present.
    /// Used by `channelPopulation` (a population of discrete channels).
    pub conductance: Option<f32>,
    /// Markov gates (`gateKS`), in declaration order.
    pub gates: Vec<MarkovGateSpec>,
}

// ─── Morphology (cell / morphology / segment / segmentGroup) ──
//
// Parsed into the IR (Фаза 5a) and consumed by the cable-equation lowering
// (`discretize`/`compile`). Every field is read by production code or by the
// module tests; only the `fractionAlong` attachment fraction is validated and
// dropped (the lowering attaches children at the parent's distal end).

/// A 3-D point with diameter (μm).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Point3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub diameter: f32,
}

/// Parent of a segment: index of the parent segment. The `fractionAlong`
/// attachment point is parsed (and validated as 0..=1) but not carried — the
/// cable lowering attaches children at the parent's distal end, so the
/// fraction is informational only.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentParent {
    pub segment: usize,
}

/// A single morphological segment (frustum between proximal and distal).
pub(crate) struct Segment {
    pub id: usize,
    pub name: Option<String>,
    /// `None` means the proximal point coincides with the parent's distal.
    pub proximal: Option<Point3d>,
    pub distal: Point3d,
    /// `None` for the root segment (soma).
    pub parent: Option<SegmentParent>,
}

/// Named collection of segments (e.g. `soma_group`, `dendrite_group`).
pub(crate) struct SegmentGroup {
    pub id: String,
    pub members: Vec<usize>,
    pub includes: Vec<String>,
}

/// Parsed `<morphology>`.
pub(crate) struct MorphoIR {
    pub id: String,
    pub segments: Vec<Segment>,
    pub segment_groups: Vec<SegmentGroup>,
}

impl MorphoIR {
    /// Resolve a `segmentGroup` id to the set of segment ids it covers,
    /// following `include` chains transitively (cycle-safe). The special id
    /// `"all"` covers every segment — the NeuroML default when a
    /// `channelDensity` omits `segmentGroup`.
    ///
    /// # Errors
    /// If the group id is unknown, or an `include` references an unknown group.
    pub fn resolve_segments(&self, group_id: &str) -> Result<Vec<usize>, NeuromlError> {
        if group_id == "all" {
            return Ok(self.segments.iter().map(|s| s.id).collect());
        }
        let groups: HashMap<&str, &SegmentGroup> = self
            .segment_groups
            .iter()
            .map(|g| (g.id.as_str(), g))
            .collect();
        let mut out: Vec<usize> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack = vec![group_id];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let g = groups.get(id).ok_or_else(|| {
                NeuromlError::parse(format!(
                    "morphology `{}`: unknown segmentGroup `{id}`",
                    self.id
                ))
            })?;
            out.extend(g.members.iter().copied());
            for inc in &g.includes {
                stack.push(inc.as_str());
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// Resolve a single segment id (a `channelPopulation`'s `segment`
    /// attribute) to a `Vec` containing just that id, validating it exists.
    /// `"all"` is accepted as a convenience for every segment.
    ///
    /// # Errors
    /// If the id is not a non-negative integer, or no segment has that id.
    pub fn resolve_segment(&self, segment: &str) -> Result<Vec<usize>, NeuromlError> {
        if segment == "all" {
            return Ok(self.segments.iter().map(|s| s.id).collect());
        }
        let id: usize = segment.parse().map_err(|_| {
            NeuromlError::parse(format!(
                "morphology `{}`: bad `segment` id `{segment}`",
                self.id
            ))
        })?;
        if !self.segments.iter().any(|s| s.id == id) {
            return Err(NeuromlError::parse(format!(
                "morphology `{}`: unknown segment id `{id}`",
                self.id
            )));
        }
        Ok(vec![id])
    }
}

/// A `channelDensity` element (ohmic conductance on a segmentGroup).
pub(crate) struct IrChannelDensity {
    pub id: String,
    pub ion_channel: String,
    /// Conductance density (S/cm²) — unit-aware parsed from `condDensity`.
    pub cond_density: f32,
    pub erev: f32,
    pub ion: String,
    pub segment_group: String,
}

/// A `channelDensityNonUniform` element — an ohmic conductance whose density
/// varies along the path within each declared segment group. Two NeuroML
/// forms:
/// * `<proximal>`/`<distal>` — linear interpolation along the group's path;
/// * `<inhomogeneousValue … value="expr"/>` — an explicit expression of the
///   path length (e.g. `1e4 * H(p-300) * 7e-3` in real Poirazi 2003 exports,
///   where `p` is the `inhomogeneousParameter` — `PathLengthOverApicalDendrites`).
pub(crate) struct NonUniformChannelDensity {
    pub id: String,
    pub ion_channel: String,
    pub erev: f32,
    /// `ion` attribute — read by the compiler for leak filtering
    /// (`non_specific`/`pas`/`leak`).
    pub ion: String,
    /// Per segment group: linear (`proximal`/`distal`) or inhomogeneous
    /// (path-length expression).
    pub variable_params: Vec<NonUniformParam>,
}

/// One `<variableParameter>` of a `channelDensityNonUniform`.
pub(crate) enum NonUniformParam {
    /// Linear interpolation: `proximal` at the group's root-near end, `distal`
    /// at the far end (S/cm²).
    Linear {
        segment_group: String,
        proximal: f32,
        distal: f32,
    },
    /// Path-length expression (e.g. `1e4 * H(p-300) * 7e-3`).
    Inhomogeneous {
        segment_group: String,
        value: InhomogeneousValue,
    },
}

/// An `inhomogeneousValue` — an expression of the path length from the soma
/// (`inhomogeneousParameter`, e.g. `PathLengthOverApicalDendrites`) evaluated
/// per compartment.
pub(crate) struct InhomogeneousValue {
    /// The density expression in **S/m²** (LEMS SI base unit — the `1e4 *`
    /// prefix in real Poirazi 2003 exports converts an S/cm² density), with `p`
    /// standing for the path length (µm) — supports numbers, `H(...)`
    /// (Heaviside), `exp(...)`, `*`, `/`, `+`, `-`. The engine always evaluates
    /// it against the compartment's path length from the soma, regardless of
    /// the declared `inhomogeneousParameter` name.
    pub expr: String,
}

/// A `channelPopulation` element — a population of `number` **discrete** ion
/// channels (as opposed to `channelDensity`, a continuous density). Each
/// channel has the single-channel conductance of the referenced ion channel;
/// the total conductance is `single_channel_conductance × number`, and — for a
/// kinetic-scheme (`ionChannelKS`) channel — the open fraction is a **binomial
/// sample** over the discrete channel count, i.e. the canonical source of
/// channel noise in small neurons/axons.
pub(crate) struct IrChannelPopulation {
    pub id: String,
    pub ion_channel: String,
    /// Number of discrete channels (`number` attribute, a non-negative int).
    pub number: usize,
    pub erev: f32,
    /// The single segment this population is placed on (`segment` attribute,
    /// a segment id as a string; default `"0"` — the soma). Unlike
    /// `channelDensity`'s `segmentGroup`, a discrete population is a **point**
    /// current on one segment, not a density spread over a group.
    pub segment: String,
}

/// A `channelPopulationNernst` element — a population of `number` **discrete**
/// Ca²⁺ channels whose combined reversal is the **Nernst potential** of the
/// compartment's [Ca²⁺] (no fixed `erev`). Same discrete-channel noise
/// semantics as [`IrChannelPopulation`], but the reversal is dynamic and the
/// Ca²⁺ current feeds the compartment's Ca²⁺ pool.
pub(crate) struct IrChannelPopulationNernst {
    pub id: String,
    pub ion_channel: String,
    /// Number of discrete channels (`number` attribute, a non-negative int).
    pub number: usize,
    /// The single segment this population is placed on (`segment` attribute,
    /// a segment id as a string; default `"0"` — the soma).
    pub segment: String,
}

/// A `channelDensityNernst` — an ohmic Ca²⁺ conductance density whose reversal
/// potential is the **Nernst potential** computed from the compartment's
/// internal/external [Ca²⁺] (no fixed `erev`). Only HH-gated channels are
/// lowered for now (`ionChannelHH`); the reversal is evaluated each step.
pub(crate) struct IrChannelDensityNernst {
    pub id: String,
    pub ion_channel: String,
    /// Conductance density (S/cm²) — unit-aware parsed from `condDensity`.
    pub cond_density: f32,
    pub segment_group: String,
}

/// A `channelDensityGHK2` — a Ca²⁺ density whose current is the **Goldman-
/// Hodgkin–Katz** flux equation (not ohmic): `I = g·ghk(v, ci, co)`, with the
/// compartment's dynamic internal [Ca²⁺] and the species' external [Ca²⁺].
/// Used by real Ca channels (Poirazi 2003 `cal`/`cat`).
pub(crate) struct IrChannelDensityGhk {
    pub id: String,
    pub ion_channel: String,
    /// Conductance density (S/cm²) — unit-aware parsed from `condDensity`.
    /// 0 for the non-uniform form (`channelDensityGHK2NonUniform`).
    pub cond_density: f32,
    /// Segment group for the uniform form; empty for the non-uniform form
    /// (the groups live in `variable_params`).
    pub segment_group: String,
    /// Non-uniform form (`channelDensityGHK2NonUniform`): one variable
    /// parameter per segment group; empty for the uniform form.
    pub variable_params: Vec<NonUniformParam>,
}

/// A `concentrationModel` (Ca²⁺ buffering) — the canonical LEMS forms. Ion
/// concentration is stored in **mM** (= mol/m³, the LEMS base unit for the
/// `concentration` dimension).
pub(crate) enum IrConcentrationModel {
    /// `decayingPoolConcentrationModel`: the ion is assumed to occupy a shell
    /// of thickness `shell_thickness` inside the membrane; the Ca current fills
    /// that shell volume and the concentration relaxes to `resting_conc` with
    /// `decay_constant`.
    DecayingPool {
        id: String,
        resting_conc_mm: f32,
        decay_constant_ms: f32,
        shell_thickness_um: f32,
    },
    /// `fixedFactorConcentrationModel`: parsed but **not lowered** — the
    /// compiler reports it as unsupported (kept only for the honest error).
    FixedFactor { id: String },
    /// A custom LEMS `ComponentType extends="concentrationModel"` whose
    /// `TimeDerivative` is the canonical decaying-pool ODE with modifiers:
    /// `dC/dt = iCa/(2·F·A·d·flux_div) − (C−rest)/τ·decay_mul` (e.g. the real
    /// Poirazi 2003 `PoiraziCaConcentrationModel`: `flux_div = 18` — the
    /// Golding sharp-repolarisation factor — and `decay_mul = 7` — 7× faster
    /// Ca removal).
    CustomDecayingPool {
        id: String,
        resting_conc_mm: f32,
        decay_constant_ms: f32,
        shell_thickness_um: f32,
        flux_divisor: f32,
        decay_multiplier: f32,
    },
}

impl IrConcentrationModel {
    /// The component id.
    pub fn id(&self) -> &str {
        match self {
            IrConcentrationModel::DecayingPool { id, .. }
            | IrConcentrationModel::FixedFactor { id, .. }
            | IrConcentrationModel::CustomDecayingPool { id, .. } => id,
        }
    }
}

/// A `<species>` — an ion whose internal/external concentrations are tracked by
/// a referenced `concentrationModel`. Declared inside `biophysicalProperties`.
pub(crate) struct IrSpecies {
    pub id: String,
    pub ion: String,
    pub concentration_model: String,
    pub initial_conc_mm: f32,
    pub initial_ext_conc_mm: f32,
    /// Segment group the species applies to (`segmentGroup`, default `all`).
    pub segment_group: String,
}

/// `<membraneProperties>`: channel densities + passive electrical constants.
pub(crate) struct MembraneProperties {
    pub channel_densities: Vec<IrChannelDensity>,
    pub non_uniform_densities: Vec<NonUniformChannelDensity>,
    /// Discrete channel populations (`channelPopulation`).
    pub channel_populations: Vec<IrChannelPopulation>,
    /// Nernst (Ca²⁺) discrete channel populations (`channelPopulationNernst`).
    pub nernst_populations: Vec<IrChannelPopulationNernst>,
    /// Nernst (Ca²⁺) channel densities (`channelDensityNernst`).
    pub nernst_densities: Vec<IrChannelDensityNernst>,
    /// GHK (Ca²⁺) channel densities (`channelDensityGHK2`) — Goldman–Hodgkin–
    /// Katz current with the compartment's [Ca²⁺].
    pub ghk_densities: Vec<IrChannelDensityGhk>,
    /// `<initMembPotential>` — resting / starting V (mV).
    pub init_memb_potential: f32,
    /// `<specificCapacitance>` — specific membrane capacitance (µF/cm²),
    /// default 1.0 (the canonical NeuroML/NEURON value).
    pub specific_capacitance: f32,
}

/// `<intracellularProperties>`: axial resistivity.
pub(crate) struct IntracellularProperties {
    pub resistivity: f32,
}

/// `<biophysicalProperties>`: membrane + intracellular + ion species.
pub(crate) struct BiophysicalProperties {
    pub membrane: MembraneProperties,
    pub intracellular: IntracellularProperties,
    /// Tracked ion species (`<species>`, e.g. Ca²⁺) with concentration models.
    pub species: Vec<IrSpecies>,
}

/// A `channelDensity` is the passive leak if its ion is `non_specific`
/// (NeuroML2 canonical) or the legacy `pas`/`leak` alias. The alias is also
/// matched on the referenced channel id, since converted models often name the
/// passive channel `pas`/`leak`.
pub(crate) fn is_passive_leak(cd: &IrChannelDensity) -> bool {
    let ion = cd.ion.to_ascii_lowercase();
    let ch = cd.ion_channel.to_ascii_lowercase();
    matches!(ion.as_str(), "non_specific" | "pas" | "leak") || matches!(ch.as_str(), "pas" | "leak")
}

impl BiophysicalProperties {
    /// The passive (leak) `channelDensity`, if declared.
    pub fn passive_leak(&self) -> Option<&IrChannelDensity> {
        self.membrane
            .channel_densities
            .iter()
            .find(|cd| is_passive_leak(cd))
    }
}

/// A `<cell>` element — a morphologically-detailed cell that references a
/// `<morphology>` (by id) and optionally a `<biophysicalProperties>`. Parsed
/// into the IR now; lowering to a cable-equation population is a later phase.
pub(crate) struct MorphoCell {
    pub id: String,
    pub morphology: String,
    pub biophysics: Option<String>,
}

// ─── Quantity parsing ──────────────────────────────────────────

/// Milliseconds per second — unit factor for `s`→`ms` and `per_s`→`per_ms`.
const MS_PER_S: f32 = 1000.0;
/// Millivolts per volt — unit factor for `V`→`mV`.
const MV_PER_V: f32 = 1000.0;

/// Split `"200ms"` → `(200.0, "ms")`; a bare number → `(n, "")`.
fn split_num_unit(s: &str) -> Result<(f32, &str), NeuromlError> {
    let s = s.trim();
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(s.len());
    let num: f32 = s[..end]
        .parse()
        .map_err(|_| NeuromlError::parse(format!("cannot parse quantity `{s}`")))?;
    Ok((num, s[end..].trim()))
}

/// Parse a **time** quantity to milliseconds.
pub(crate) fn to_ms(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "s" => n * MS_PER_S,
        _ => n, // "ms" or bare → already ms
    })
}

/// Parse a **voltage** quantity to millivolts.
pub(crate) fn to_mv(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "V" => n * MV_PER_V,
        _ => n, // "mV" or bare → already mV
    })
}

/// Parse a **rate** quantity (`per_time`) to per-millisecond. NeuroML rates are
/// commonly `per_ms` (e.g. `1per_ms`), but the schema allows `per_s` too.
pub(crate) fn to_per_ms(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "per_s" => n * MS_PER_S,
        _ => n, // "per_ms" or bare → already per ms
    })
}

/// Parse a **firing rate** quantity to Hz (`per_s`). `averageRate` on
/// `poissonFiringSynapse` is canonical `per_s` (Hz); accept explicit `Hz`,
/// `kHz` and bare (Hz) too.
pub(crate) fn to_hz(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "Hz" | "per_s" | "" => n,
        "kHz" => n * 1000.0,
        "per_ms" => n * MS_PER_S,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported rate unit `{other}` (expected Hz/per_s)"
            )))
        }
    })
}

/// Parse a **dimensionless / raw** quantity — the unit suffix is discarded.
pub(crate) fn raw(s: &str) -> Result<f32, NeuromlError> {
    Ok(split_num_unit(s)?.0)
}

/// Parse a **resistance** to MΩ: `mV/MΩ = nA` gives the pillars raw current
/// units directly (nA → per-population scale applied by the compiler).
pub(crate) fn to_mohm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "Mohm" | "" => n,
        "kohm" => n * 1e-3,
        "ohm" => n * 1e-6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported resistance unit `{other}` (expected ohm/kohm/Mohm)"
            )))
        }
    })
}

/// Parse an **axial resistivity** (`intracellularProperties/resistivity`) to
/// Ω·cm. NeuroML2 writes `kohm_cm` (e.g. Poirazi `0.05kohm_cm` = 50 Ω·cm); a
/// bare number falls back to Ω·cm.
pub(crate) fn to_ohm_cm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "ohm_cm" | "" => n,
        "kohm_cm" => n * 1e3,
        "Mohm_cm" => n * 1e6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported resistivity unit `{other}` (expected ohm_cm/kohm_cm)"
            )))
        }
    })
}

/// Parse a **conductance density** (`condDensity`) to S/cm². NeuroML2's
/// canonical unit is `S_per_m2`; legacy models (and the HH `channelDensity`
/// path) write `mS_per_cm2`; a bare number falls back to the legacy `mS/cm²`
/// convention. `rm (Ω·cm²) = 1/g (S/cm²)` follows directly.
pub(crate) fn to_s_per_cm2(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "S_per_m2" => n * 1e-4,
        "S_per_cm2" => n,
        "mS_per_cm2" => n * 1e-3,
        "mS_per_m2" => n * 1e-7,
        "uS_per_cm2" => n * 1e-6,
        "" => n * 1e-3, // legacy: bare = mS/cm²
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported conductance density unit `{other}`"
            )))
        }
    })
}

/// Parse a **single-channel conductance** (`conductance`) to µS. NeuroML2's
/// canonical unit is `S`; real channel conductances are usually written in
/// `pS`/`nS`/`uS`; a bare number falls back to µS (the codebase's legacy raw
/// convention). The cable solver's absolute-conductance contract is µS, so the
/// value comes out directly in µS (`mV/MΩ = nA`, `g[µS]·(E−V)[mV] = nA`).
pub(crate) fn to_us(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "S" => n * 1e6,
        "mS" => n * 1e3,
        "uS" => n,
        "nS" => n * 1e-3,
        "pS" => n * 1e-6,
        "" => n, // legacy: bare = µS
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported conductance unit `{other}` (expected S/mS/uS/nS/pS)"
            )))
        }
    })
}

/// Parse a **specific capacitance** (`specificCapacitance`) to µF/cm².
/// NeuroML2's canonical unit is `F_per_m2` (1 F/m² = 100 µF/cm²); legacy models
/// write `uF_per_cm2`; a bare number falls back to µF/cm² (default 1.0).
pub(crate) fn to_uf_per_cm2(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "F_per_m2" => n * 100.0,
        "F_per_cm2" => n * 1e6,
        "mF_per_cm2" => n * 1000.0,
        "uF_per_cm2" => n,
        "uF_per_m2" => n * 1e-4,
        "" => n, // legacy: bare = µF/cm²
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported specific capacitance unit `{other}`"
            )))
        }
    })
}

/// Parse an ion **concentration** to **mM** (= mol/m³, the LEMS base unit for
/// the `concentration` dimension). NeuroML writes `mM`/`M`/`uM`/`nM` (and the
/// SI `mol_per_m3`/`mol_per_cm3`); a bare number falls back to mM.
pub(crate) fn to_mm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "mM" | "mol_per_m3" | "" => n,
        "M" => n * 1000.0,
        "uM" => n * 1e-3,
        "nM" => n * 1e-6,
        "mol_per_cm3" => n * 1e6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported concentration unit `{other}` (expected mM/M/uM/nM)"
            )))
        }
    })
}

/// Attribute lookup on a roxmltree node that reports a clear parse error.
fn attr<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Result<&'a str, NeuromlError> {
    node.attribute(name).ok_or_else(|| {
        NeuromlError::parse(format!("<{}> missing `{name}`", node.tag_name().name()))
    })
}

fn attr_opt<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<&'a str> {
    node.attribute(name)
}

fn time_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    to_ms(attr(node, name)?)
}

fn voltage_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    to_mv(attr(node, name)?)
}

fn raw_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    raw(attr(node, name)?)
}

fn raw_attr_opt(node: roxmltree::Node<'_, '_>, name: &str) -> Result<Option<f32>, NeuromlError> {
    attr_opt(node, name).map(raw).transpose()
}

// ─── Ion-channel / gate parsing ────────────────────────────────

/// Direct element child with the given tag name (error if absent).
fn child<'a>(
    node: roxmltree::Node<'a, 'a>,
    name: &str,
) -> Result<roxmltree::Node<'a, 'a>, NeuromlError> {
    node.children()
        .find(|c| c.is_element() && c.tag_name().name() == name)
        .ok_or_else(|| {
            NeuromlError::parse(format!(
                "<{}> missing child <{name}>",
                node.tag_name().name()
            ))
        })
}

/// Convert a NeuroML2 HH rate/variable node (`forwardRate`, `reverseRate`,
/// `steadyState`) into a gate steady-state [`VarForm`]. The standard NeuroML
/// `type` attributes map onto the fast [`ClosedForm`]; a custom LEMS
/// `ComponentType` (real OSB models) is lowered to a [`LemsExpr`].
///
/// Mapping (NeuroML → pillars convention `arg = −(v + midpoint)/scale`):
///
/// | NeuroML type                       | NeuroML formula                 | ClosedForm                |
/// |------------------------------------|---------------------------------|---------------------------|
/// | `HHExpRate`/`HHExpVariable`        | `r·exp((v−m)/s)`                | `Exp { r, −m, −s }`       |
/// | `HHSigmoidRate`/`HHSigmoidVariable`| `r/(1+exp(−(v−m)/s))`           | `Sigmoid { r, −m, s }`    |
/// | `HHExpLinearRate`/`…Variable`      | `r·x/(1−exp(−x)), x=(v−m)/s`    | `ExpLinear { r/s, −m, s }`|
fn parse_closed_form(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<VarForm, NeuromlError> {
    let ty = attr(node, "type")?;
    // Custom LEMS ComponentType (e.g. `Poirazi_h_ss`)?
    if !matches!(
        ty,
        "HHExpRate"
            | "HHExpVariable"
            | "HHSigmoidRate"
            | "HHSigmoidVariable"
            | "HHExpLinearRate"
            | "HHExpLinearVariable"
    ) {
        if let Some(ct) = comp_types.get(ty) {
            let expr = ct.bind(node, &format!("steadyState `{ty}`"))?;
            return Ok(VarForm::Expr(expr));
        }
        return Err(NeuromlError::parse(format!(
            "unknown HH rate/variable type `{ty}` (and no matching <ComponentType> in the document)"
        )));
    }
    let is_variable = ty.ends_with("Variable");
    // `rate` is per_time for rate forms, dimensionless for variable forms.
    let rate = if is_variable {
        raw_attr(node, "rate")?
    } else {
        to_per_ms(attr(node, "rate")?)?
    };
    let midpoint = to_mv(attr(node, "midpoint")?)?;
    let scale = to_mv(attr(node, "scale")?)?;
    // `scale` sits in the denominator of every closed form (the exponent for
    // Exp/Sigmoid, both the exponent and the `rate/scale` prefactor for
    // ExpLinear). A zero scale would yield ±inf/NaN silently — reject it here.
    if scale == 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{ty}>: `scale` must be non-zero (got 0)"
        )));
    }
    let cf = match ty {
        "HHExpRate" | "HHExpVariable" => ClosedForm::Exp {
            rate,
            midpoint: -midpoint,
            scale: -scale,
        },
        "HHSigmoidRate" | "HHSigmoidVariable" => ClosedForm::Sigmoid {
            rate,
            midpoint: -midpoint,
            scale,
        },
        "HHExpLinearRate" | "HHExpLinearVariable" => ClosedForm::ExpLinear {
            rate: rate / scale,
            midpoint: -midpoint,
            scale,
        },
        _ => unreachable!("checked above"),
    };
    Ok(VarForm::Closed(cf))
}

/// Parse a `timeCourse` node into a gate [`TauForm`]. `fixedTimeCourse` →
/// fixed tau; a custom LEMS `ComponentType` (real Ih/HCN models) → V-dependent
/// expression.
fn parse_time_course(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<TauForm, NeuromlError> {
    let ty = attr(node, "type")?;
    if ty == "fixedTimeCourse" {
        return Ok(TauForm::Fixed(to_ms(attr(node, "tau")?)?));
    }
    if let Some(ct) = comp_types.get(ty) {
        let expr = ct.bind(node, &format!("timeCourse `{ty}`"))?;
        return Ok(TauForm::Expr(expr));
    }
    Err(NeuromlError::parse(format!(
        "unsupported timeCourse type `{ty}` (only fixedTimeCourse or a custom <ComponentType>)"
    )))
}

/// Parse a `q10Settings` node (temperature scaling of gate time courses).
fn parse_q10(node: roxmltree::Node<'_, '_>) -> Result<IrQ10, NeuromlError> {
    Ok(match attr(node, "type")? {
        "q10Fixed" => IrQ10::Fixed(raw_attr(node, "fixedQ10")?),
        "q10ExpTemp" => IrQ10::ExpTemp {
            factor: raw_attr(node, "q10Factor")?,
            experimental_temp: raw_attr(node, "experimentalTemp")?,
        },
        other => {
            return Err(NeuromlError::parse(format!(
                "unknown q10Settings type `{other}`"
            )))
        }
    })
}

/// Parse a gate child of an ion channel into [`IrGate`]. `gateHHrates` and
/// `gateHHtauInf` map onto the runtime [`GateKinetics`]; the remaining HH gate
/// kinds are recognised but not yet compilable (`kinetics = None`).
fn parse_gate(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrGate, NeuromlError> {
    let tag = node.tag_name().name();
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1). Reject zero/negative/
    // fractional before the `as u32` cast (which would silently truncate or
    // wrap) so a malformed model fails loudly instead of with a wrong exponent.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }
    let instances = instances_f as u32;
    let kind: &'static str = match tag {
        "gateHHrates" => "gateHHrates",
        "gateHHtauInf" => "gateHHtauInf",
        "gateHHInstantaneous" => "gateHHInstantaneous",
        "gateHHratesTau" => "gateHHratesTau",
        "gateHHratesInf" => "gateHHratesInf",
        "gateHHratesTauInf" => "gateHHratesTauInf",
        _ => return Err(NeuromlError::parse(format!("<{tag}> is not an HH gate"))),
    };

    let kinetics = match tag {
        "gateHHrates" => {
            let forward = parse_closed_form(child(node, "forwardRate")?, comp_types)?;
            let reverse = parse_closed_form(child(node, "reverseRate")?, comp_types)?;
            Some(GateKinetics::Rates {
                forward: RateForm::from_var_form(forward),
                reverse: RateForm::from_var_form(reverse),
            })
        }
        "gateHHtauInf" => {
            let steady_state = parse_closed_form(child(node, "steadyState")?, comp_types)?;
            let tau = parse_time_course(child(node, "timeCourse")?, comp_types)?;
            Some(GateKinetics::TauInf { steady_state, tau })
        }
        // Algebraic gate: q = steady_state(v[, caConc]) every step, no ODE.
        // Used by real Ca channels (Poirazi cal/cat `h2 = ki/(ki+caConc)`).
        "gateHHInstantaneous" => {
            let steady_state = parse_closed_form(child(node, "steadyState")?, comp_types)?;
            Some(GateKinetics::Instantaneous { steady_state })
        }
        _ => None, // recognised but not yet compilable
    };

    let q10 = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "q10Settings")
        .map(parse_q10)
        .transpose()?;

    Ok(IrGate {
        id,
        instances,
        kinetics,
        kind,
        q10,
    })
}

/// Parse an `ionChannel` / `ionChannelHH` element.
fn parse_ion_channel(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrChannel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let conductance = attr_opt(node, "conductance").map(to_us).transpose()?;
    let mut gates = Vec::new();
    let mut fractional_gates = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "gateHHrates"
            | "gateHHtauInf"
            | "gateHHInstantaneous"
            | "gateHHratesTau"
            | "gateHHratesInf"
            | "gateHHratesTauInf" => gates.push(parse_gate(child, comp_types)?),
            "gateFractional" => fractional_gates.push(parse_fractional_gate(child, comp_types)?),
            _ => {}
        }
    }
    Ok(IrChannel {
        id,
        conductance,
        gates,
        fractional_gates,
    })
}

/// Parse a `gateFractional` into a [`FractionalGateSpec`]. Each `subGate` is an
/// independent tau-inf gate (`dq/dt = (inf − q)/tau`) weighted by its
/// `fractionalConductance`; the gate's open fraction is `q = Σ frac_i·q_i` and
/// `fcond = q^instances`.
fn parse_fractional_gate(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<FractionalGateSpec, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1) — validate before cast.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<gateFractional id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }
    let mut subgates = Vec::new();
    for sub in node.children() {
        if !sub.is_element() || sub.tag_name().name() != "subGate" {
            continue;
        }
        let fractional_conductance = raw_attr(sub, "fractionalConductance")?;
        let steady_state = parse_closed_form(child(sub, "steadyState")?, comp_types)?;
        let tau = parse_time_course(child(sub, "timeCourse")?, comp_types)?;
        subgates.push(FractionalSubgate {
            steady_state: match steady_state {
                VarForm::Closed(cf) => cf,
                VarForm::Expr(_) => {
                    return Err(NeuromlError::parse(format!(
                        "<gateFractional id=\"{id}\">: subGate steadyState uses a custom ComponentType (only closed forms are supported in fractional gates)"
                    )))
                }
            },
            tau: match tau {
                TauForm::Fixed(t) => t,
                TauForm::Expr(_) => {
                    return Err(NeuromlError::parse(format!(
                        "<gateFractional id=\"{id}\">: subGate timeCourse uses a custom ComponentType (only fixedTimeCourse is supported in fractional gates)"
                    )))
                }
            },
            fractional_conductance,
        });
    }
    if subgates.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<gateFractional id=\"{id}\">: no subGate declared"
        )));
    }
    Ok(FractionalGateSpec {
        subgates,
        power: instances_f as u32,
    })
}

/// Parse a `gateKS` (kinetic scheme) into a [`MarkovGateSpec`]. States are
/// declared as `closedState` (relativeConductance 0) / `openState` (1);
/// transitions are `forwardTransition`+`reverseTransition` (each a directed
/// HH rate, `from`→`to` forward / `to`→`from` reverse) or `tauInfTransition`
/// (forward `inf/tau`, reverse `(1−inf)/tau`).
///
/// State ids are resolved to indices in declaration order; a transition that
/// references an unknown state is a hard parse error (never a silent drop).
fn parse_gate_ks(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<MarkovGateSpec, NeuromlError> {
    let tag = node.tag_name().name();
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1) — validate before cast.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }

    // States in declaration order (closed before open, per schema).
    let mut state_ids: Vec<String> = Vec::new();
    let mut open: Vec<f32> = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "closedState" => {
                state_ids.push(attr(child, "id")?.to_string());
                open.push(0.0);
            }
            "openState" => {
                state_ids.push(attr(child, "id")?.to_string());
                open.push(1.0);
            }
            _ => {}
        }
    }
    if state_ids.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: no closedState/openState declared"
        )));
    }
    let index: HashMap<&str, usize> = state_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut transitions = Vec::new();
    for t_node in node.children() {
        match t_node.tag_name().name() {
            "forwardTransition" | "reverseTransition" => {
                let from = resolve_state(&index, attr(t_node, "from")?, &id)?;
                let to = resolve_state(&index, attr(t_node, "to")?, &id)?;
                // `rate` child — любой элемент (xs:any в схеме), но по конвенции
                // `<rate type="HH…Rate" …/>` с параметрами baseHHRate.
                let rate_node = t_node.children().find(|c| c.is_element()).ok_or_else(|| {
                    NeuromlError::parse(format!(
                        "<{} id=\"{id}\">: transition missing rate child",
                        t_node.tag_name().name()
                    ))
                })?;
                let rate = match parse_closed_form(rate_node, comp_types)? {
                    VarForm::Closed(cf) => cf,
                    VarForm::Expr(_) => {
                        return Err(NeuromlError::parse(format!(
                            "<{} id=\"{id}\">: transition rate uses a custom ComponentType (only closed HH rate forms are supported in gateKS)",
                            t_node.tag_name().name()
                        )))
                    }
                };
                let (f, t) = if t_node.tag_name().name() == "forwardTransition" {
                    (from, to)
                } else {
                    // reverseTransition: rate applies to `to`→`from`.
                    (to, from)
                };
                transitions.push(MarkovTransition {
                    from: f,
                    to: t,
                    rate: MarkovRate::Directed { rate },
                });
            }
            "tauInfTransition" => {
                let from = resolve_state(&index, attr(t_node, "from")?, &id)?;
                let to = resolve_state(&index, attr(t_node, "to")?, &id)?;
                let steady_state = parse_closed_form(child(t_node, "steadyState")?, comp_types)?;
                let tau = parse_time_course(child(t_node, "timeCourse")?, comp_types)?;
                transitions.push(MarkovTransition {
                    from,
                    to,
                    rate: MarkovRate::TauInfForward {
                        steady_state: steady_state.clone(),
                        tau: tau.clone(),
                    },
                });
                transitions.push(MarkovTransition {
                    from: to,
                    to: from,
                    rate: MarkovRate::TauInfReverse { steady_state, tau },
                });
            }
            _ => {}
        }
    }
    if transitions.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: no transitions declared"
        )));
    }

    Ok(MarkovGateSpec {
        open,
        transitions,
        power: instances_f as u32,
    })
}

/// Resolve a transition `from`/`to` state id to its index, or fail clearly.
fn resolve_state(
    index: &HashMap<&str, usize>,
    state_id: &str,
    gate_id: &str,
) -> Result<usize, NeuromlError> {
    index.get(state_id).copied().ok_or_else(|| {
        NeuromlError::parse(format!(
            "gateKS `{gate_id}`: transition references unknown state `{state_id}`"
        ))
    })
}

/// Parse an `ionChannelKS` element into [`IrKsChannel`].
fn parse_ion_channel_ks(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrKsChannel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let conductance = attr_opt(node, "conductance").map(to_us).transpose()?;
    let mut gates = Vec::new();
    for child in node.children() {
        if child.is_element() && child.tag_name().name() == "gateKS" {
            gates.push(parse_gate_ks(child, comp_types)?);
        }
    }
    Ok(IrKsChannel {
        id,
        conductance,
        gates,
    })
}

// ─── Morphology parsing ────────────────────────────────────────

/// Parse a `<point>` child (`<proximal>` or `<distal>`) into `Point3d`.
/// Units are NeuroML SI (`1e-6 m` = μm).
fn parse_point(node: roxmltree::Node<'_, '_>) -> Result<Point3d, NeuromlError> {
    Ok(Point3d {
        x: raw(node.attribute("x").unwrap_or("0"))?,
        y: raw(node.attribute("y").unwrap_or("0"))?,
        z: raw(node.attribute("z").unwrap_or("0"))?,
        diameter: raw(node.attribute("diameter").unwrap_or("0"))?,
    })
}

/// Parse a `<segment>` element.
fn parse_segment(node: roxmltree::Node<'_, '_>) -> Result<Segment, NeuromlError> {
    let id: usize = raw_attr(node, "id")? as usize;
    let name = attr_opt(node, "name").map(str::to_string);
    let proximal = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "proximal")
        .map(parse_point)
        .transpose()?;
    let distal = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "distal")
        .map(parse_point)
        .ok_or_else(|| NeuromlError::parse(format!("<segment id={id}>: missing <distal>")))??;
    // NeuroML2: `<parent segment="…" fractionAlong="…"/>`. `segmentParent`/
    // `fractionalLength` accepted as a lenient alias (older drafts / our
    // historical fixtures) — the canonical names are `parent`/`fractionAlong`.
    // The fraction is validated as 0..=1 but not stored (the cable lowering
    // attaches children at the parent's distal end).
    let parent = node
        .children()
        .find(|c| c.is_element() && matches!(c.tag_name().name(), "parent" | "segmentParent"))
        .map(|sp| {
            let segment: usize = raw_attr(sp, "segment")? as usize;
            if let Some(frac) = attr_opt(sp, "fractionAlong")
                .or_else(|| attr_opt(sp, "fractionalLength"))
                .map(raw)
                .transpose()?
            {
                if !(0.0..=1.0).contains(&frac) {
                    return Err(NeuromlError::parse(format!(
                        "segment parent `fractionAlong` {frac} out of [0, 1]"
                    )));
                }
            }
            Ok(SegmentParent { segment })
        })
        .transpose()?;
    Ok(Segment {
        id,
        name,
        proximal,
        distal,
        parent,
    })
}

/// Parse a `<segmentGroup>` element.
fn parse_segment_group(node: roxmltree::Node<'_, '_>) -> Result<SegmentGroup, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let mut members = Vec::new();
    let mut includes = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "member" => {
                if let Some(seg) = attr_opt(child, "segment") {
                    members.push(seg.parse::<usize>().map_err(|_| {
                        NeuromlError::parse(format!(
                            "<segmentGroup id=\"{id}\">: bad member segment `{seg}`"
                        ))
                    })?);
                }
            }
            "include" => {
                if let Some(seg_group) = attr_opt(child, "segmentGroup") {
                    includes.push(seg_group.to_string());
                }
            }
            _ => {}
        }
    }
    Ok(SegmentGroup {
        id,
        members,
        includes,
    })
}

/// Parse a `<morphology>` element.
fn parse_morphology(node: roxmltree::Node<'_, '_>) -> Result<MorphoIR, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let mut segments = Vec::new();
    let mut segment_groups = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "segment" => segments.push(parse_segment(child)?),
            "segmentGroup" => segment_groups.push(parse_segment_group(child)?),
            _ => {}
        }
    }
    Ok(MorphoIR {
        id,
        segments,
        segment_groups,
    })
}

/// Parse a `<channelDensity>` element inside `<membraneProperties>`.
fn parse_channel_density(node: roxmltree::Node<'_, '_>) -> Result<IrChannelDensity, NeuromlError> {
    Ok(IrChannelDensity {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        // Unit-aware: `S_per_m2`/`mS_per_cm2`/… → S/cm² (not a raw strip).
        cond_density: to_s_per_cm2(attr(node, "condDensity")?)?,
        erev: voltage_attr(node, "erev")?,
        ion: attr(node, "ion")?.to_string(),
        segment_group: attr_opt(node, "segmentGroup").unwrap_or("all").to_string(),
    })
}

/// Parse a `<channelPopulation>` element inside `<membraneProperties>`: a
/// population of `number` discrete ion channels (single-channel conductance
/// comes from the referenced ion channel).
fn parse_channel_population(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrChannelPopulation, NeuromlError> {
    let number_f = raw_attr(node, "number")?;
    if number_f < 0.0 || number_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<channelPopulation>: `number` must be a non-negative integer, got {number_f}"
        )));
    }
    Ok(IrChannelPopulation {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        number: number_f as usize,
        erev: voltage_attr(node, "erev")?,
        segment: attr_opt(node, "segment").unwrap_or("0").to_string(),
    })
}

/// Parse a `<channelPopulationNernst>` element: a population of `number`
/// discrete Ca²⁺ channels with Nernst (dynamic) reversal.
fn parse_channel_population_nernst(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrChannelPopulationNernst, NeuromlError> {
    let number_f = raw_attr(node, "number")?;
    if number_f < 0.0 || number_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<channelPopulationNernst>: `number` must be a non-negative integer, got {number_f}"
        )));
    }
    Ok(IrChannelPopulationNernst {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        number: number_f as usize,
        segment: attr_opt(node, "segment").unwrap_or("0").to_string(),
    })
}

/// Parse a `<channelDensityNernst>` element: an ohmic Ca²⁺ density whose
/// reversal potential is the Nernst potential of the compartment's [Ca²⁺]
/// (no fixed `erev`). Only HH-gated channels are lowered.
fn parse_channel_density_nernst(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrChannelDensityNernst, NeuromlError> {
    Ok(IrChannelDensityNernst {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        cond_density: to_s_per_cm2(attr(node, "condDensity")?)?,
        segment_group: attr_opt(node, "segmentGroup").unwrap_or("all").to_string(),
    })
}

/// Parse a `<channelDensityGHK2>` / `<channelDensityGHK2NonUniform>` element: a
/// Ca²⁺ density whose current is the Goldman–Hodgkin–Katz flux equation (not
/// ohmic), driven by the compartment's dynamic internal [Ca²⁺] (no fixed
/// `erev`). The non-uniform form carries `<variableParameter>`s (linear or
/// path-length-expression) instead of a fixed `condDensity`.
fn parse_channel_density_ghk(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrChannelDensityGhk, NeuromlError> {
    let mut variable_params = Vec::new();
    for child in node.children() {
        if child.is_element() && child.tag_name().name() == "variableParameter" {
            variable_params.push(parse_variable_parameter(child)?);
        }
    }
    Ok(IrChannelDensityGhk {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        cond_density: attr_opt(node, "condDensity")
            .map(to_s_per_cm2)
            .transpose()?
            .unwrap_or(0.0),
        segment_group: attr_opt(node, "segmentGroup").unwrap_or("all").to_string(),
        variable_params,
    })
}

/// Parse one `<variableParameter>` into a [`NonUniformParam`] (linear
/// proximal/distal form or `inhomogeneousValue` path-expression form).
fn parse_variable_parameter(
    node: roxmltree::Node<'_, '_>,
) -> Result<NonUniformParam, NeuromlError> {
    let segment_group = attr(node, "segmentGroup")?.to_string();
    let mut proximal = None;
    let mut distal = None;
    let mut inhomogeneous = None;
    for sub in node.children() {
        match sub.tag_name().name() {
            "proximal" => proximal = Some(to_s_per_cm2(attr(sub, "value")?)?),
            "distal" => distal = Some(to_s_per_cm2(attr(sub, "value")?)?),
            "inhomogeneousValue" => {
                // `inhomogeneousParameter` (e.g. `PathLengthOverApicalDendrites`)
                // is not stored — the engine always evaluates `expr` against
                // the compartment's path length from the soma.
                let _param = attr(sub, "inhomogeneousParameter")?;
                inhomogeneous = Some(InhomogeneousValue {
                    expr: attr(sub, "value")?.to_string(),
                });
            }
            _ => {}
        }
    }
    if let Some(inh) = inhomogeneous {
        Ok(NonUniformParam::Inhomogeneous {
            segment_group,
            value: inh,
        })
    } else {
        Ok(NonUniformParam::Linear {
            segment_group,
            proximal: proximal.ok_or_else(|| {
                NeuromlError::parse("<variableParameter> missing <proximal> value")
            })?,
            distal: distal
                .ok_or_else(|| NeuromlError::parse("<variableParameter> missing <distal> value"))?,
        })
    }
}

/// Parse a `<decayingPoolConcentrationModel>` / `<fixedFactorConcentrationModel>`
/// / `<fixedFactorConcentrationModelTraub>` into [`IrConcentrationModel`]. The
/// Traub form (fixed `phi` + `beta` rate) is lowered onto [`FixedFactor`] with
/// `rho = 1e-9·phi`, `decay_constant = 1/beta`.
fn parse_concentration_model(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrConcentrationModel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    Ok(match node.tag_name().name() {
        "decayingPoolConcentrationModel" => IrConcentrationModel::DecayingPool {
            id,
            resting_conc_mm: to_mm(attr(node, "restingConc")?)?,
            decay_constant_ms: time_attr(node, "decayConstant")?,
            shell_thickness_um: raw(attr(node, "shellThickness")?)?,
        },
        "fixedFactorConcentrationModel" | "fixedFactorConcentrationModelTraub" => {
            // LEMS: dC/dt = (iCa/surfaceArea)·1e-9·phi − (C−rest)·beta.
            IrConcentrationModel::FixedFactor { id }
        }
        _ => {
            return Err(NeuromlError::parse(format!(
                "<{}> is not a concentration model",
                node.tag_name().name()
            )))
        }
    })
}

/// Parse a custom LEMS `ComponentType extends="concentrationModel"` instance
/// (e.g. `<PoiraziCaConcentrationModel id="capool" …/>`). The `TimeDerivative`
/// must be the canonical decaying-pool ODE with optional modifiers:
/// `iCa/(2·Faraday·surfaceArea·shellThickness·flux_div)
///  − (concentration−restingConc)/decayConstant·decay_mul`.
fn parse_custom_concentration_model(
    node: roxmltree::Node<'_, '_>,
) -> Result<IrConcentrationModel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let resting_conc_mm = to_mm(attr(node, "restingConc")?)?;
    let decay_constant_ms = time_attr(node, "decayConstant")?;
    let shell_thickness_um = raw(attr(node, "shellThickness")?)?;

    // TimeDerivative string → extract the flux divisor and decay multiplier.
    // The canonical LEMS form is
    //   iCa / (2 * Faraday * surfaceArea * shellThickness * FLUX_DIV)
    //   - ((concentration - restingConc) / decayConstant) * DECAY_MUL
    // (modifiers default to 1 — a plain decaying pool).
    let mut flux_divisor = 1.0f32;
    let mut decay_multiplier = 1.0f32;
    for d in node
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "TimeDerivative")
    {
        let value = d.attribute("value").unwrap_or("");
        // flux divisor: the last factor inside `iCa/(…*shellThickness*N)`.
        if let Some(idx) = value.find("shellThickness") {
            let after = &value[idx + "shellThickness".len()..];
            if let Some(star) = after.find('*') {
                let num: String = after[star + 1..]
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if !num.is_empty() {
                    flux_divisor = num.parse().map_err(|e: std::num::ParseFloatError| {
                        NeuromlError::parse(e.to_string())
                    })?;
                }
            }
        }
        // decay multiplier: the factor after `decayConstant)…)*N`.
        if let Some(idx) = value.find("decayConstant") {
            let after = &value[idx + "decayConstant".len()..];
            if let Some(star) = after.rfind('*') {
                let num: String = after[star + 1..]
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if !num.is_empty() {
                    decay_multiplier = num.parse().map_err(|e: std::num::ParseFloatError| {
                        NeuromlError::parse(e.to_string())
                    })?;
                }
            }
        }
    }
    Ok(IrConcentrationModel::CustomDecayingPool {
        id,
        resting_conc_mm,
        decay_constant_ms,
        shell_thickness_um,
        flux_divisor,
        decay_multiplier,
    })
}

/// Parse a `<species>` element: an ion whose concentration is tracked by a
/// referenced `concentrationModel`.
fn parse_species(node: roxmltree::Node<'_, '_>) -> Result<IrSpecies, NeuromlError> {
    Ok(IrSpecies {
        id: attr(node, "id")?.to_string(),
        ion: attr(node, "ion")?.to_string(),
        concentration_model: attr(node, "concentrationModel")?.to_string(),
        initial_conc_mm: to_mm(attr(node, "initialConcentration")?)?,
        initial_ext_conc_mm: to_mm(attr(node, "initialExtConcentration")?)?,
        segment_group: attr_opt(node, "segmentGroup").unwrap_or("all").to_string(),
    })
}

/// Parse a `<channelDensityNonUniform>` element: one `<variableParameter>` per
/// segment group, each with `<proximal>`/`<distal>` densities (S/cm²).
fn parse_non_uniform_channel_density(
    node: roxmltree::Node<'_, '_>,
) -> Result<NonUniformChannelDensity, NeuromlError> {
    let mut variable_params = Vec::new();
    for child in node.children() {
        if child.is_element() && child.tag_name().name() == "variableParameter" {
            variable_params.push(parse_variable_parameter(child)?);
        }
    }
    Ok(NonUniformChannelDensity {
        id: attr(node, "id")?.to_string(),
        ion_channel: attr(node, "ionChannel")?.to_string(),
        erev: voltage_attr(node, "erev")?,
        ion: attr(node, "ion")?.to_string(),
        variable_params,
    })
}

/// Parse `<membraneProperties>`.
fn parse_membrane_properties(
    node: roxmltree::Node<'_, '_>,
) -> Result<MembraneProperties, NeuromlError> {
    let mut channel_densities = Vec::new();
    let mut non_uniform_densities = Vec::new();
    let mut channel_populations = Vec::new();
    let mut nernst_populations = Vec::new();
    let mut nernst_densities = Vec::new();
    let mut ghk_densities = Vec::new();
    let mut init_memb_potential = -65.0; // default
    let mut specific_capacitance = 1.0; // µF/cm² (NeuroML default)
    for child in node.children() {
        match child.tag_name().name() {
            "channelDensity" => channel_densities.push(parse_channel_density(child)?),
            "channelDensityNonUniform" => {
                non_uniform_densities.push(parse_non_uniform_channel_density(child)?)
            }
            "channelDensityNernst" => nernst_densities.push(parse_channel_density_nernst(child)?),
            "channelDensityGHK2" | "channelDensityGHK2NonUniform" => {
                ghk_densities.push(parse_channel_density_ghk(child)?)
            }
            "channelPopulation" => channel_populations.push(parse_channel_population(child)?),
            "channelPopulationNernst" => {
                nernst_populations.push(parse_channel_population_nernst(child)?)
            }
            "initMembPotential" => {
                init_memb_potential = voltage_attr(child, "value")?;
            }
            "specificCapacitance" => {
                // Unit-aware: `F_per_m2`/`uF_per_cm2`/… → µF/cm².
                specific_capacitance = to_uf_per_cm2(attr(child, "value")?)?;
            }
            _ => {}
        }
    }
    Ok(MembraneProperties {
        channel_densities,
        non_uniform_densities,
        channel_populations,
        nernst_populations,
        nernst_densities,
        ghk_densities,
        init_memb_potential,
        specific_capacitance,
    })
}

/// Parse `<intracellularProperties>`.
fn parse_intracellular_properties(
    node: roxmltree::Node<'_, '_>,
) -> Result<IntracellularProperties, NeuromlError> {
    let mut resistivity = 100.0; // default Ω·cm
    for child in node.children() {
        if child.tag_name().name() == "resistivity" {
            resistivity = to_ohm_cm(attr(child, "value")?)?;
        }
    }
    Ok(IntracellularProperties { resistivity })
}

/// Parse `<biophysicalProperties>`.
fn parse_biophysical_properties(
    node: roxmltree::Node<'_, '_>,
) -> Result<BiophysicalProperties, NeuromlError> {
    let membrane = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "membraneProperties")
        .map(parse_membrane_properties)
        .transpose()?
        .unwrap_or(MembraneProperties {
            channel_densities: Vec::new(),
            non_uniform_densities: Vec::new(),
            channel_populations: Vec::new(),
            nernst_populations: Vec::new(),
            nernst_densities: Vec::new(),
            ghk_densities: Vec::new(),
            init_memb_potential: -65.0,
            specific_capacitance: 1.0,
        });
    let intracellular = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "intracellularProperties")
        .map(parse_intracellular_properties)
        .transpose()?
        .unwrap_or(IntracellularProperties { resistivity: 100.0 });
    // `<species>` may be direct children of `biophysicalProperties` or of
    // `intracellularProperties` (real exports put them next to `resistivity`).
    let mut species: Vec<IrSpecies> = Vec::new();
    for child in node.children() {
        if child.tag_name().name() == "species" {
            species.push(parse_species(child)?);
        } else if child.tag_name().name() == "intracellularProperties" {
            for sc in child.children() {
                if sc.is_element() && sc.tag_name().name() == "species" {
                    species.push(parse_species(sc)?);
                }
            }
        }
    }
    Ok(BiophysicalProperties {
        membrane,
        intracellular,
        species,
    })
}

/// Standard HH channel families — the two layouts in
/// [`crate::kernels::hh_channels::standard_channels`] (NaV: m³h, KV: n⁴). A
/// parsed channel must match one of these **bit-exactly** to be lowered onto
/// the canonical `HHModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StandardChannel {
    Nav,
    Kv,
}

/// If a channel's gates (kinetics + exponents) bit-match a standard channel,
/// return which family it belongs to. Bit-exact matching is deliberate: only
/// channels numerically identical to the NEURON-validated NaV/KV layouts may
/// be mapped onto `HHModel`, otherwise their dynamics would silently diverge.
fn standard_role(ch: &IrChannel) -> Option<StandardChannel> {
    // Дробные гейты не вписываются в канонический HHModel (взвешенная сумма
    // субгейтов), поэтому канал с gateFractional никогда не считается стандартным.
    if !ch.fractional_gates.is_empty() {
        return None;
    }
    let std = crate::kernels::hh_channels::standard_channels();
    'outer: for sc in &std {
        if sc.gates.len() != ch.gates.len() {
            continue;
        }
        for (sg, g) in sc.gates.iter().zip(&ch.gates) {
            if sg.power != g.instances {
                continue 'outer;
            }
            match &g.kinetics {
                Some(k) if *k == sg.kinetics => {}
                _ => continue 'outer,
            }
        }
        // All gates bit-match: classify by the channel's canonical name.
        return match sc.name {
            "NaV" => Some(StandardChannel::Nav),
            "KV" => Some(StandardChannel::Kv),
            // A future standard channel we don't recognise as NaV/KV — treat it
            // as non-standard rather than mis-mapping it onto HHModel.
            _ => None,
        };
    }
    None
}

/// Resolve a parsed `ionChannelHH` declaration into a runnable channel.
///
/// Fails with a clear error if any gate is recognised but not yet compilable
/// (e.g. `gateHHInstantaneous`) — a referenced channel must not be silently
/// dropped.
/// Bind the `rateScale` (→ Q10 multiplier) and `temperature` (→ Kelvin)
/// `<Requirement>`s inside a gate's custom LEMS expressions. `rateScale` is 1.0
/// when the gate has no `q10Settings`.
fn bind_kinetics_requirements(kinetics: GateKinetics, q10: f32, temp_k: f32) -> GateKinetics {
    let bind_expr = |e: LemsExpr| e.bind_rate_scale(q10).bind_temperature(temp_k);
    match kinetics {
        GateKinetics::Rates { forward, reverse } => GateKinetics::Rates {
            forward: match forward {
                RateForm::Closed(cf) => RateForm::Closed(cf),
                RateForm::Expr(e) => RateForm::Expr(bind_expr(e)),
            },
            reverse: match reverse {
                RateForm::Closed(cf) => RateForm::Closed(cf),
                RateForm::Expr(e) => RateForm::Expr(bind_expr(e)),
            },
        },
        GateKinetics::TauInf { steady_state, tau } => GateKinetics::TauInf {
            steady_state: match steady_state {
                VarForm::Closed(cf) => VarForm::Closed(cf),
                VarForm::Expr(e) => VarForm::Expr(bind_expr(e)),
            },
            tau: match tau {
                TauForm::Fixed(t) => TauForm::Fixed(t),
                TauForm::Expr(e) => TauForm::Expr(bind_expr(e)),
            },
        },
        GateKinetics::Instantaneous { steady_state } => GateKinetics::Instantaneous {
            steady_state: match steady_state {
                VarForm::Closed(cf) => VarForm::Closed(cf),
                VarForm::Expr(e) => VarForm::Expr(bind_expr(e)),
            },
        },
    }
}

pub(crate) fn resolve_channel(
    ch: &IrChannel,
    e_rev: f32,
    g_max: f32,
    celsius: f32,
) -> Result<ResolvedChannel, NeuromlError> {
    let mut gates = Vec::with_capacity(ch.gates.len());
    let mut q10s = Vec::with_capacity(ch.gates.len());
    for g in &ch.gates {
        let kinetics = g.kinetics.clone().ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel `{}`: gate `{}` ({}) not yet supported by the runtime",
                ch.id, g.id, g.kind
            ))
        })?;
        // Custom taus with a `rateScale` Requirement bind it to the gate's Q10
        // multiplier (jNeuroML semantics; 1.0 when no q10Settings); custom
        // rates/steady-states with a `temperature` Requirement bind the
        // simulation temperature in Kelvin.
        let q10 = g.q10.map(|q| q.factor_at(celsius)).unwrap_or(1.0);
        let temp_k = celsius + 273.15;
        let kinetics = bind_kinetics_requirements(kinetics, q10, temp_k);
        gates.push((kinetics, g.instances));
        q10s.push(q10);
    }
    Ok(ResolvedChannel {
        gates,
        q10s,
        e_rev,
        g_max,
    })
}

/// Resolve a parsed `ionChannelHH` whose gates include `gateFractional` into a
/// runnable kinetic-scheme channel: HH gates drop to 2-state Markov, fractional
/// gates expand to their product-space Markov scheme. Runs on
/// [`crate::api::MarkovPopulation`] (scalar) — the SIMD HH engine cannot carry
/// a weighted-sum gate.
pub(crate) fn resolve_markov_channel(
    ch: &IrChannel,
    e_rev: f32,
    g_max: f32,
    celsius: f32,
) -> Result<ResolvedMarkovChannel, NeuromlError> {
    let mut gates = Vec::with_capacity(ch.gates.len() + ch.fractional_gates.len());
    let mut q10s = Vec::with_capacity(ch.gates.len() + ch.fractional_gates.len());
    for g in &ch.gates {
        let kinetics = g.kinetics.clone().ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel `{}`: gate `{}` ({}) not yet supported by the runtime",
                ch.id, g.id, g.kind
            ))
        })?;
        // Same `rateScale` → Q10 / `temperature` → K binding as
        // `resolve_channel`.
        let q10 = g.q10.map(|q| q.factor_at(celsius)).unwrap_or(1.0);
        let temp_k = celsius + 273.15;
        let kinetics = bind_kinetics_requirements(kinetics, q10, temp_k);
        gates.push(gate_kinetics_to_markov(kinetics, g.instances));
        q10s.push(q10);
    }
    for fg in &ch.fractional_gates {
        gates.push(fractional_gate_to_markov(fg));
        // gateFractional has no q10Settings in the NeuroML spec.
        q10s.push(1.0);
    }
    Ok(ResolvedMarkovChannel {
        gates,
        q10s,
        e_rev,
        g_max,
    })
}

/// The standard NaV/KV gate layout, expressed as a runnable channel for a
/// mixed cell that must run on the generic engine (reversal/density come from
/// the `channelDensity`, not the hardcoded standard defaults).
fn standard_resolved(role: StandardChannel, e_rev: f32, g_max: f32) -> ResolvedChannel {
    let idx = match role {
        StandardChannel::Nav => 0,
        StandardChannel::Kv => 1,
    };
    let sc = &crate::kernels::hh_channels::standard_channels()[idx];
    ResolvedChannel {
        gates: sc
            .gates
            .iter()
            .map(|g| (g.kinetics.clone(), g.power))
            .collect(),
        // Standard zookeeper channels carry no q10Settings — temperature
        // scaling stays on the engine's global Q10 (celsius → q10).
        q10s: vec![1.0; sc.gates.len()],
        e_rev,
        g_max,
    }
}

// ─── Parser ────────────────────────────────────────────────────
/// Parse a NeuroML2 (and/or LEMS) XML document into the IR.
pub(crate) fn parse(xml: &str) -> Result<Ir, NeuromlError> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| {
        // Позиция XML-ошибки (row:col, 1-based) — структурированно, для
        // редактора/IDE: без неё пользователь ищет синтаксическую ошибку по
        // всему файлу. `Display` сам собирает «XML: R:C: …» из `pos`.
        let pos = e.pos();
        if pos.row == 1 && pos.col == 1 {
            // roxmltree не смог локализовать ошибку (напр. пустой документ) —
            // оставляем пометку «XML:», но без позиции.
            NeuromlError::parse(format!("XML: {e}"))
        } else {
            NeuromlError::xml(pos.row, pos.col, e.to_string())
        }
    })?;

    // Pre-pass: collect custom LEMS `ComponentType`s (gate time courses /
    // steady states) FIRST — ion channels reference them by `type`, and a
    // `ComponentType` may be declared after the channel that uses it.
    // `extends="concentrationModel"` types are Ca-pool ODEs, parsed separately
    // (their `Parameter` dimensions — concentration/length — are not gate
    // parameters), so they are excluded here.
    let mut comp_types: HashMap<String, ComponentType> = HashMap::new();
    let mut pool_type_names: Vec<String> = Vec::new();
    for node in doc.root_element().descendants() {
        if node.tag_name().name() == "ComponentType" {
            if let Some(name) = node.attribute("name") {
                let is_pool = node
                    .attribute("extends")
                    .map(|e| e == "concentrationModel")
                    .unwrap_or(false);
                if is_pool {
                    pool_type_names.push(name.to_string());
                } else {
                    comp_types.insert(name.to_string(), parse_component_type(node)?);
                }
            }
        }
    }

    // Pre-pass: collect ion channels (HH and KS) so `hhCell` can resolve
    // `channelDensity` references regardless of document order, and
    // morphologies/biophysics so `cell` can reference them in any order.
    let mut channels: HashMap<String, IrChannel> = HashMap::new();
    let mut ks_channels: HashMap<String, IrKsChannel> = HashMap::new();
    let mut morphologies: Vec<MorphoIR> = Vec::new();
    let mut biophysics: HashMap<String, BiophysicalProperties> = HashMap::new();
    let mut concentration_models: HashMap<String, IrConcentrationModel> = HashMap::new();
    for node in doc.root_element().descendants() {
        match node.tag_name().name() {
            "ionChannel" | "ionChannelHH" => {
                let ch = parse_ion_channel(node, &comp_types)?;
                channels.insert(ch.id.clone(), ch);
            }
            "ionChannelKS" => {
                let ch = parse_ion_channel_ks(node, &comp_types)?;
                ks_channels.insert(ch.id.clone(), ch);
            }
            "morphology" => morphologies.push(parse_morphology(node)?),
            "biophysicalProperties" => {
                let bp = parse_biophysical_properties(node)?;
                biophysics.insert(attr(node, "id")?.to_string(), bp);
            }
            "decayingPoolConcentrationModel"
            | "fixedFactorConcentrationModel"
            | "fixedFactorConcentrationModelTraub" => {
                let cm = parse_concentration_model(node)?;
                concentration_models.insert(cm.id().to_string(), cm);
            }
            // Custom LEMS pool instances: `<PoiraziCaConcentrationModel id="…"
            // restingConc=… decayConstant=… shellThickness=…/>`.
            _ if pool_type_names.iter().any(|t| t == node.tag_name().name()) => {
                let cm = parse_custom_concentration_model(node)?;
                concentration_models.insert(cm.id().to_string(), cm);
            }
            _ => {}
        }
    }

    let mut ir = Ir {
        cells: Vec::new(),
        synapses: Vec::new(),
        double_synapses: Vec::new(),
        pulse_gens: Vec::new(),
        poisson_synapses: Vec::new(),
        transient_poisson_synapses: Vec::new(),
        spike_gens: Vec::new(),
        spike_arrays: Vec::new(),
        spike_array_refs: Vec::new(),
        sine_gens: Vec::new(),
        ramp_gens: Vec::new(),
        compounds: Vec::new(),
        timed_syn_inputs: Vec::new(),
        voltage_clamps: Vec::new(),
        voltage_clamp_triples: Vec::new(),
        pops: Vec::new(),
        projections: Vec::new(),
        inputs: Vec::new(),
        gap_junctions: Vec::new(),
        electrical_projections: Vec::new(),
        graded_synapses: Vec::new(),
        sigmoid_graded_synapses: Vec::new(),
        continuous_projections: Vec::new(),
        channels: Vec::new(),
        ks_channels: Vec::new(),
        morphologies: Vec::new(),
        biophysics: HashMap::new(),
        concentration_models: HashMap::new(),
        morpho_cells: Vec::new(),
        sim: None,
        warnings: Vec::new(),
    };

    for node in doc.root_element().descendants() {
        match node.tag_name().name() {
            // ── cells ──
            "izhikevichCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Izh {
                        v0: voltage_attr(node, "v0")?,
                        thresh: voltage_attr(node, "thresh")?,
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        c: voltage_attr(node, "c")?,
                        d: raw_attr(node, "d")?,
                    },
                };
                ir.cells.push(cell);
            }
            "izhikevich2007Cell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Izh2007 {
                        v0: voltage_attr(node, "v0")?,
                        cap: raw_attr(node, "C")?,
                        k: raw_attr(node, "k")?,
                        vr: voltage_attr(node, "vr")?,
                        vt: voltage_attr(node, "vt")?,
                        vpeak: voltage_attr(node, "vpeak")?,
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        c: voltage_attr(node, "c")?,
                        d: raw_attr(node, "d")?,
                    },
                };
                ir.cells.push(cell);
            }
            "iafTauRefCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: time_attr(node, "tau")?,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref: time_attr(node, "refract")?,
                        // iafTau*Cell: ток уже в mV-эквиваленте (нет gL).
                        current_scale: 1.0,
                    },
                };
                ir.cells.push(cell);
            }
            "iafTauCell" => {
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: time_attr(node, "tau")?,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref: 0.0,
                        current_scale: 1.0,
                    },
                };
                ir.cells.push(cell);
            }
            "iafCell" | "iafRefCell" => {
                // tau_m = C / gL — raw pF / nS gives ms directly.
                let c = raw_attr(node, "C")?;
                let gl = raw_attr(node, "leakConductance")?;
                if gl <= 0.0 {
                    return Err(NeuromlError::parse(
                        "iafCell: leakConductance должно быть > 0",
                    ));
                }
                // iafRefCell добавляет refract; у iafCell его нет (0).
                let tau_ref = attr_opt(node, "refract")
                    .map(to_ms)
                    .transpose()?
                    .unwrap_or(0.0);
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Iaf {
                        tau_m: c / gl,
                        v_rest: voltage_attr(node, "leakReversal")?,
                        v_thresh: voltage_attr(node, "thresh")?,
                        v_reset: voltage_attr(node, "reset")?,
                        tau_ref,
                        // LIF voltage-normalised: ток nA→мВ = ·1000/gL.
                        current_scale: 1000.0 / gl,
                    },
                };
                ir.cells.push(cell);
            }
            "adExIaFCell" => {
                let c = raw_attr(node, "C")?;
                let gl = raw_attr(node, "gL")?;
                if gl <= 0.0 {
                    return Err(NeuromlError::parse("adExIaFCell: gL должно быть > 0"));
                }
                let cell = Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::AdEx {
                        tau_m: c / gl,
                        tau_w: time_attr(node, "tauw")?,
                        v_rest: voltage_attr(node, "EL")?,
                        // VT = exponential spike threshold (pillars v_thresh),
                        // thresh = spike *detection* threshold (pillars v_peak).
                        v_thresh: voltage_attr(node, "VT")?,
                        v_reset: voltage_attr(node, "reset")?,
                        delta_t: voltage_attr(node, "delT")?,
                        // Pillars AdEx — voltage-normalised (÷gL). nA→mV:
                        // 1 nA / 1 nS = 1 V = 1000 mV, так что b/gL·1000.
                        a: raw_attr(node, "a")? / gl,
                        b: raw_attr(node, "b")? / gl * 1000.0,
                        tau_ref: time_attr(node, "refract")?,
                        v_peak: voltage_attr(node, "thresh")?,
                        current_scale: 1000.0 / gl,
                    },
                };
                ir.cells.push(cell);
            }
            // Классический FitzHugh–Nagumo (безразмерный, единственный
            // параметр — встроенный ток I):
            //   dV/dt = V − V³/3 − W + I ; dW/dt = 0.08·(V + 0.7 − 0.8·W)
            // → pillars FHNModel с a=0.7, b=0.8, inv_tau_w=0.08.
            "fitzHughNagumoCell" => {
                ir.cells.push(Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Fhn {
                        a: 0.7,
                        b: 0.8,
                        inv_tau_w: 0.08,
                        v_thresh: 0.5,
                        bias: raw_attr(node, "I")?,
                        v0: 0.0,
                        w0: 0.0,
                    },
                });
            }
            // FitzHugh–Nagumo 1969: явные a, b, phi (масштаб времени) и I.
            //   dV/dt = V − V³/3 − W + I ; dW/dt = φ·(V + a − b·W)
            "fitzHughNagumo1969Cell" => {
                ir.cells.push(Cell {
                    id: attr(node, "id")?.to_string(),
                    spec: CellSpec::Fhn {
                        a: raw_attr(node, "a")?,
                        b: raw_attr(node, "b")?,
                        inv_tau_w: raw_attr(node, "phi")?,
                        v_thresh: 0.5,
                        bias: raw_attr(node, "I")?,
                        v0: raw_attr_opt(node, "V0")?.unwrap_or(0.0),
                        w0: raw_attr_opt(node, "W0")?.unwrap_or(0.0),
                    },
                });
            }
            // Классический Hodgkin–Huxley через channelDensity + leak. Гейты
            // резолвятся по объявленным `ionChannelHH` (data-driven), с
            // фолбэком на подстроки id/ionChannel для легаси-моделей.
            // Стандартные m³h/n⁴ → быстрый путь HHModel; иначе — generic-движок.
            "hhCell" => {
                let v0 = voltage_attr(node, "v0")?;
                let c = raw_attr(node, "C")?;
                let g_leak = raw_attr(node, "leakConductance")?;
                let e_leak = voltage_attr(node, "leakReversal")?;
                if c <= 0.0 {
                    return Err(NeuromlError::parse("hhCell: C должно быть > 0"));
                }
                // Na/K — для быстрого пути HHModel, если каналы стандартные.
                let mut g_na: Option<(f32, f32)> = None; // (condDensity, erev)
                let mut g_k: Option<(f32, f32)> = None;
                // Нестандартные каналы — для generic-движка.
                let mut custom: Vec<ResolvedChannel> = Vec::new();
                let mut markov_custom: Vec<ResolvedMarkovChannel> = Vec::new();

                for child in node.children() {
                    if child.tag_name().name() != "channelDensity" {
                        continue;
                    }
                    let cond = raw_attr(child, "condDensity")?;
                    let erev = voltage_attr(child, "erev")?;
                    let ion = attr(child, "ionChannel")?;
                    if let Some(ch) = channels.get(ion) {
                        if !ch.fractional_gates.is_empty() {
                            // Есть gateFractional — SIMD HH-движок не умеет
                            // взвешенную сумму субгейтов; опускаем весь канал до
                            // марковской схемы (HH → 2-состояния, fractional →
                            // произведение субгейтов).
                            markov_custom.push(resolve_markov_channel(
                                ch,
                                erev,
                                cond,
                                CONDUCTANCE_CELSIUS,
                            )?);
                        } else {
                            match standard_role(ch) {
                                Some(StandardChannel::Nav) => g_na = Some((cond, erev)),
                                Some(StandardChannel::Kv) => g_k = Some((cond, erev)),
                                None => custom.push(resolve_channel(
                                    ch,
                                    erev,
                                    cond,
                                    CONDUCTANCE_CELSIUS,
                                )?),
                            }
                        }
                    } else if let Some(ch) = ks_channels.get(ion) {
                        markov_custom.push(ResolvedMarkovChannel {
                            gates: ch.gates.clone(),
                            // ionChannelKS gates carry no q10Settings.
                            q10s: vec![1.0; ch.gates.len()],
                            e_rev: erev,
                            g_max: cond,
                        });
                    } else {
                        // Нет объявления — легаси-эвристика по подстрокам id/ionChannel.
                        let id = attr(child, "id")?.to_lowercase();
                        let ion_l = ion.to_lowercase();
                        if id.contains("na") || ion_l.contains("na") {
                            g_na = Some((cond, erev));
                        } else if id.contains("k") || ion_l.contains("k") {
                            g_k = Some((cond, erev));
                        }
                    }
                }

                if custom.is_empty() && markov_custom.is_empty() {
                    // Только стандартный Na/K — быстрый путь HHModel.
                    let (g_na, e_na) = g_na.ok_or_else(|| {
                        NeuromlError::parse(
                            "hhCell: не найден Na-канал (channelDensity с id/ionChannel, содержащим «na»)",
                        )
                    })?;
                    let (g_k, e_k) = g_k.ok_or_else(|| {
                        NeuromlError::parse(
                            "hhCell: не найден K-канал (channelDensity с id/ionChannel, содержащим «k»)",
                        )
                    })?;
                    ir.cells.push(Cell {
                        id: attr(node, "id")?.to_string(),
                        spec: CellSpec::Hh {
                            v0,
                            g_na,
                            e_na,
                            g_k,
                            e_k,
                            g_leak,
                            e_leak,
                            inv_cm: 1.0 / c,
                        },
                    });
                } else {
                    // Смешанный/нестандартный набор — generic-движок. Порядок:
                    // стандартные Na/K (если есть), затем нестандартные по
                    // объявлению. Кинетические (KS) каналы — отдельным списком:
                    // если их нет, работает SIMD-движок, иначе скалярный Markov.
                    let mut all: Vec<ResolvedChannel> = Vec::new();
                    if let Some((cond, erev)) = g_na {
                        all.push(standard_resolved(StandardChannel::Nav, erev, cond));
                    }
                    if let Some((cond, erev)) = g_k {
                        all.push(standard_resolved(StandardChannel::Kv, erev, cond));
                    }
                    all.extend(custom);
                    ir.cells.push(Cell {
                        id: attr(node, "id")?.to_string(),
                        spec: CellSpec::ConductanceBased {
                            v0,
                            c_m: c,
                            g_leak,
                            e_leak,
                            channels: all,
                            markov_channels: markov_custom,
                        },
                    });
                }
            }
            // ── synapses ──
            "expOneSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpOne {
                        tau_decay: attr_opt(node, "tauDecay").map(to_ms).transpose()?,
                    },
                    plast: None,
                });
            }
            "expTwoSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpTwo {
                        tau_rise: attr_opt(node, "tauRise").map(to_ms).transpose()?,
                        tau_decay: attr_opt(node, "tauDecay").map(to_ms).transpose()?,
                    },
                    plast: None,
                });
            }
            "alphaSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::Alpha {
                        tau: to_ms(attr(node, "tau")?)?,
                    },
                    plast: None,
                });
            }
            "expThreeSynapse" => {
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: None,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpThree {
                        tau_rise: to_ms(attr(node, "tauRise")?)?,
                        tau_decay1: to_ms(attr(node, "tauDecay1")?)?,
                        tau_decay2: to_ms(attr(node, "tauDecay2")?)?,
                        gbase1: raw_attr(node, "gbase1")?,
                        gbase2: raw_attr(node, "gbase2")?,
                    },
                    plast: None,
                });
            }
            "doubleSynapse" => {
                ir.double_synapses.push(DoubleSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    synapse1: attr(node, "synapse1")?.to_string(),
                    synapse2: attr(node, "synapse2")?.to_string(),
                });
            }
            "blockingPlasticSynapse" => {
                let mut plast = None;
                for child in node.children() {
                    match child.tag_name().name() {
                        "tsodyksMarkramDepMechanism" => {
                            plast = Some(PlasticitySpec::TsodyksDep {
                                init_release_prob: raw_attr(child, "initReleaseProb")?,
                                tau_rec: to_ms(attr(child, "tauRec")?)?,
                            });
                        }
                        "tsodyksMarkramDepFacMechanism" => {
                            plast = Some(PlasticitySpec::TsodyksDepFac {
                                init_release_prob: raw_attr(child, "initReleaseProb")?,
                                tau_rec: to_ms(attr(child, "tauRec")?)?,
                                tau_fac: to_ms(attr(child, "tauFac")?)?,
                            });
                        }
                        _ => {}
                    }
                }
                ir.synapses.push(Synapse {
                    id: attr(node, "id")?.to_string(),
                    gbase: raw_attr_opt(node, "gbase")?,
                    erev: attr_opt(node, "erev").map(to_mv).transpose()?,
                    kind: SynapseKind::ExpTwo {
                        tau_rise: Some(to_ms(attr(node, "tauRise")?)?),
                        tau_decay: Some(to_ms(attr(node, "tauDecay")?)?),
                    },
                    plast,
                });
            }
            // ── input ──
            "pulseGenerator" => {
                ir.pulse_gens.push(PulseGen {
                    id: attr(node, "id")?.to_string(),
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    amplitude: raw_attr(node, "amplitude")?,
                });
            }
            "poissonFiringSynapse" => {
                ir.poisson_synapses.push(PoissonSpec {
                    id: attr(node, "id")?.to_string(),
                    rate_hz: to_hz(attr(node, "averageRate")?)?,
                });
            }
            "transientPoissonFiringSynapse" => {
                ir.transient_poisson_synapses.push(TransientPoissonSpec {
                    id: attr(node, "id")?.to_string(),
                    rate_hz: to_hz(attr(node, "averageRate")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                });
            }
            "spikeGenerator" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Regular {
                        period_ms: time_attr(node, "period")?,
                    },
                });
            }
            "spikeGeneratorPoisson" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Poisson {
                        rate_hz: to_hz(attr(node, "averageRate")?)?,
                    },
                });
            }
            "spikeGeneratorRandom" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::Random {
                        min_isi_ms: time_attr(node, "minISI")?,
                        max_isi_ms: time_attr(node, "maxISI")?,
                    },
                });
            }
            "spikeGeneratorRefPoisson" => {
                ir.spike_gens.push(SpikeGenSpec {
                    id: attr(node, "id")?.to_string(),
                    kind: SpikeGenKind::RefPoisson {
                        rate_hz: to_hz(attr(node, "averageRate")?)?,
                        min_isi_ms: time_attr(node, "minimumISI")?,
                    },
                });
            }
            "spikeArray" => {
                let mut times = Vec::new();
                for sp in node.children() {
                    if sp.is_element() && sp.tag_name().name() == "spike" {
                        times.push(to_ms(attr(sp, "time")?)?);
                    }
                }
                times.sort_by(f32::total_cmp);
                ir.spike_arrays.push(SpikeArray {
                    id: attr(node, "id")?.to_string(),
                    times_ms: times,
                });
            }
            "spikeArrayRef" => {
                ir.spike_array_refs.push(SpikeArrayRef {
                    id: attr(node, "id")?.to_string(),
                    array: attr(node, "ref")?.to_string(),
                });
            }
            "sineGenerator" => {
                ir.sine_gens.push(SineGenSpec {
                    id: attr(node, "id")?.to_string(),
                    phase: raw_attr(node, "phase")?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    amplitude: raw_attr(node, "amplitude")?,
                    period_ms: time_attr(node, "period")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "rampGenerator" => {
                ir.ramp_gens.push(RampGenSpec {
                    id: attr(node, "id")?.to_string(),
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    start_amp: raw_attr(node, "startAmplitude")?,
                    finish_amp: raw_attr(node, "finishAmplitude")?,
                    baseline_amp: raw_attr(node, "baselineAmplitude")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "compoundInput" => {
                let mut children = Vec::new();
                for child in node.children() {
                    if child.is_element() && child.tag_name().name() == "input" {
                        children.push(attr(child, "component")?.to_string());
                    }
                }
                ir.compounds.push(CompoundSpec {
                    id: attr(node, "id")?.to_string(),
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                    children,
                });
            }
            "timedSynapticInput" => {
                let mut times = Vec::new();
                for sp in node.children() {
                    if sp.is_element() && sp.tag_name().name() == "spike" {
                        times.push(to_ms(attr(sp, "time")?)?);
                    }
                }
                times.sort_by(f32::total_cmp);
                ir.timed_syn_inputs.push(TimedSynInputSpec {
                    id: attr(node, "id")?.to_string(),
                    synapse: attr(node, "synapse")?.to_string(),
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                    times_ms: times,
                });
            }
            "voltageClamp" => {
                ir.voltage_clamps.push(VoltageClampSpec {
                    id: attr(node, "id")?.to_string(),
                    target_mv: voltage_attr(node, "targetVoltage")?,
                    series_resistance_mohm: to_mohm(attr(node, "simpleSeriesResistance")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            "voltageClampTriple" => {
                ir.voltage_clamp_triples.push(VoltageClampTripleSpec {
                    id: attr(node, "id")?.to_string(),
                    active: raw_attr(node, "active")? != 0.0,
                    conditioning_mv: voltage_attr(node, "conditioningVoltage")?,
                    testing_mv: voltage_attr(node, "testingVoltage")?,
                    return_mv: voltage_attr(node, "returnVoltage")?,
                    series_resistance_mohm: to_mohm(attr(node, "simpleSeriesResistance")?)?,
                    delay_ms: time_attr(node, "delay")?,
                    duration_ms: time_attr(node, "duration")?,
                    weight: raw_attr_opt(node, "weight")?.unwrap_or(1.0),
                });
            }
            // ── structure ──
            "population" => {
                let size: usize = raw_attr(node, "size")? as usize;
                ir.pops.push(PopSpec {
                    id: attr(node, "id")?.to_string(),
                    component: attr(node, "component")?.to_string(),
                    size,
                });
            }
            "projection" => {
                let mut proj = ProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    synapse: attr_opt(node, "synapse").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "connection" | "connectionWD" => {
                            let (_, pre_idx) = parse_cell_ref(attr(child, "preCellId")?)?;
                            let (_, post_idx) = parse_cell_ref(attr(child, "postCellId")?)?;
                            proj.connections.push(Connection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                                delay_ms: attr_opt(child, "delay")
                                    .map(to_ms)
                                    .transpose()?
                                    .unwrap_or(0.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.projections.push(proj);
            }
            "explicitInput" => {
                let (pop, idx) = parse_cell_ref(attr(node, "target")?)?;
                ir.inputs.push(InputSpec {
                    target_pop: pop.to_string(),
                    target_idx: idx,
                    generator: attr(node, "input")?.to_string(),
                });
            }
            "gapJunction" => {
                ir.gap_junctions.push(GapJunctionSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr_opt(node, "conductance")?,
                });
            }
            "electricalProjection" => {
                let mut proj = ElectricalProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    gap_junction: attr_opt(node, "gapJunction").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "electricalConnection"
                        | "electricalConnectionInstance"
                        | "electricalConnectionInstanceW" => {
                            // id-form spells the refs `preCellId`/`postCellId`,
                            // instance-form spells them `preCell`/`postCell`.
                            let pre = attr_opt(child, "preCellId")
                                .or_else(|| attr_opt(child, "preCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing preCellId/preCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let post = attr_opt(child, "postCellId")
                                .or_else(|| attr_opt(child, "postCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing postCellId/postCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let (_, pre_idx) = parse_cell_ref(pre)?;
                            let (_, post_idx) = parse_cell_ref(post)?;
                            proj.connections.push(ElectricalConnection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.electrical_projections.push(proj);
            }
            "linearGradedSynapse" => {
                ir.graded_synapses.push(GradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr_opt(node, "conductance")?,
                });
            }
            "silentSynapse" => {
                ir.graded_synapses.push(GradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: Some(0.0),
                });
            }
            "gradedSynapse" => {
                ir.sigmoid_graded_synapses.push(SigmoidGradedSynapseSpec {
                    id: attr(node, "id")?.to_string(),
                    conductance: raw_attr(node, "conductance")?,
                    delta: voltage_attr(node, "delta")?,
                    k: to_per_ms(attr(node, "k")?)?,
                    vth: voltage_attr(node, "Vth")?,
                    erev: voltage_attr(node, "erev")?,
                });
            }
            "continuousProjection" => {
                let mut proj = ContinuousProjectionSpec {
                    pre_pop: attr(node, "presynapticPopulation")?.to_string(),
                    post_pop: attr(node, "postsynapticPopulation")?.to_string(),
                    synapse: attr_opt(node, "synapse").map(str::to_string),
                    connections: Vec::new(),
                };
                for child in node.children() {
                    match child.tag_name().name() {
                        "continuousConnection" | "continuousConnectionInstance" => {
                            let pre = attr_opt(child, "preCellId")
                                .or_else(|| attr_opt(child, "preCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing preCellId/preCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let post = attr_opt(child, "postCellId")
                                .or_else(|| attr_opt(child, "postCell"))
                                .ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<{}> missing postCellId/postCell",
                                        child.tag_name().name()
                                    ))
                                })?;
                            let (_, pre_idx) = parse_cell_ref(pre)?;
                            let (_, post_idx) = parse_cell_ref(post)?;
                            proj.connections.push(ContinuousConnection {
                                pre_idx,
                                post_idx,
                                weight: raw_attr_opt(child, "weight")?.unwrap_or(1.0),
                            });
                        }
                        _ => {}
                    }
                }
                ir.continuous_projections.push(proj);
            }
            // ── LEMS simulation ──
            "Simulation" => {
                ir.sim = Some(SimConfig {
                    dt_ms: time_attr(node, "step")?,
                    duration_ms: time_attr(node, "length")?,
                });
            }
            // ── morphologically-detailed cell ──
            "cell" => {
                // Reference form: `<cell id="…" morphology="…" biophysicalProperties="…"/>`.
                // Inline form (also valid NeuroML2):
                // `<cell id="…"><morphology id="…">…</morphology><biophysicalProperties id="…">…</biophysicalProperties></cell>`
                // — the pre-pass already registered the inline elements, so we
                // just resolve the id from the inline child when the attribute
                // is absent.
                let id = attr(node, "id")?.to_string();
                let morphology = match attr_opt(node, "morphology") {
                    Some(m) => m.to_string(),
                    None => node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "morphology")
                        .map(|m| attr(m, "id").map(str::to_string))
                        .transpose()?
                        .ok_or_else(|| {
                            NeuromlError::parse(format!(
                                "<cell id=\"{id}\">: missing `morphology` (attribute or inline child)"
                            ))
                        })?,
                };
                let biophysics = match attr_opt(node, "biophysicalProperties") {
                    Some(b) => Some(b.to_string()),
                    None => node
                        .children()
                        .find(|c| c.is_element() && c.tag_name().name() == "biophysicalProperties")
                        .map(|b| attr(b, "id").map(str::to_string))
                        .transpose()?,
                };
                ir.morpho_cells.push(MorphoCell {
                    id,
                    morphology,
                    biophysics,
                });
            }
            // Распознанные, но неподдерживаемые элементы — предупреждаем,
            // а не молчим (модель «загрузилась», но потеряла связность/вход).
            other => {
                if let Some(note) = skipped_element_note(other) {
                    let msg = format!("<{other}> — не поддерживается, пропущено{note}");
                    if !ir.warnings.contains(&msg) {
                        ir.warnings.push(msg);
                    }
                }
            }
        }
    }

    ir.channels = channels.into_values().collect();
    ir.ks_channels = ks_channels.into_values().collect();
    ir.morphologies = morphologies;
    ir.biophysics = biophysics;
    ir.concentration_models = concentration_models;
    Ok(ir)
}

/// Short note for recognised-but-unsupported elements (silently skipped).
/// Returns `None` for harmless structural/LEMS tags we intentionally ignore
/// (e.g. `neuroml`, `Lems`, comments, `include` — handled elsewhere).
fn skipped_element_note(tag: &str) -> Option<&'static str> {
    Some(match tag {
        // STDP (в самом NeuroML помечен как «EXAMPLE NOT YET WORKING») — не
        // поддержан; blockingPlasticSynapse уже парсится выше.
        "stdpSynapse" => " — STDP-пластичность",
        // Dimensionless *DL-генераторы (ток задан в безразмерных единицах —
        // нельзя перевести в физический ток без клеточного масштаба).
        "pulseGeneratorDL" | "sineGeneratorDL" | "rampGeneratorDL" | "compoundInputDL" => {
            " — dimensionless (DL) input-генератор"
        }
        "doubleSynapse" => " — двойной синапс (два профиля на связи)",
        _ => return None,
    })
}

/// Parse a cell reference like `"pop[0]"` or `"../pop[0]"` into `("pop", 0)`.
pub(crate) fn parse_cell_ref(target: &str) -> Result<(&str, usize), NeuromlError> {
    let open = target
        .find('[')
        .ok_or_else(|| NeuromlError::parse(format!("bad cell reference `{target}`")))?;
    let close = target
        .find(']')
        .ok_or_else(|| NeuromlError::parse(format!("bad cell reference `{target}`")))?;
    // Strip a leading path ("../") and keep the population id.
    let pop = target[..open].rsplit('/').next().unwrap_or(&target[..open]);
    let idx: usize = target[open + 1..close]
        .parse()
        .map_err(|_| NeuromlError::parse(format!("bad cell index in `{target}`")))?;
    Ok((pop, idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::hh_channels::{
        k_n_alpha, k_n_beta, na_h_alpha, na_h_beta, na_m_alpha, na_m_beta,
    };

    /// Реальный Ih-канал Poirazi 2003 (`h.channel.nml` из OSB): кастомные
    /// LEMS-`ComponentType` `Poirazi_h_tau` (piecewise double-exponential tau)
    /// и `Poirazi_h_ss` (Boltzmann-подобный) — нижются в [`LemsExpr`].
    const POIRAZI_H_CHANNEL: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="NeuroML_ionChannel">
  <ionChannel id="hs" conductance="1pS" type="ionChannelHH" species="na">
    <gateHHtauInf id="n" instances="1">
      <timeCourse type="Poirazi_h_tau"/>
      <steadyState type="Poirazi_h_ss" midpoint="-82mV" scale="8.8mV"/>
    </gateHHtauInf>
  </ionChannel>
  <ComponentType name="Poirazi_h_tau" extends="baseVoltageDepTime">
    <Constant name="TIME_SCALE" dimension="time" value="1 ms"/>
    <Constant name="VOLT_SCALE" dimension="voltage" value="1 mV"/>
    <Dynamics>
      <DerivedVariable name="V" dimension="none" value="v / VOLT_SCALE"/>
      <DerivedVariable name="tmp" value="2 * (1 / (exp((V+145)/-17.5) + exp((V+16.8)/16.5)) + 5)" dimension="none"/>
      <ConditionalDerivedVariable name="t" dimension="time" exposure="t" >
        <Case condition="V .lt. -30" value="tmp * TIME_SCALE"/>
        <Case value="1 * TIME_SCALE"/>
      </ConditionalDerivedVariable>
    </Dynamics>
  </ComponentType>
  <ComponentType name="Poirazi_h_ss" extends="baseVoltageDepVariable">
    <Parameter name="midpoint" dimension="voltage"/>
    <Parameter name="scale" dimension="voltage"/>
    <Constant name="VOLT_SCALE" dimension="voltage" value="1 mV"/>
    <Dynamics>
      <DerivedVariable name="V" dimension="none" value="v / VOLT_SCALE"/>
      <DerivedVariable name="x" exposure="x" dimension="none" value="1 - (1 / (1 + exp(0 - (V-midpoint/VOLT_SCALE)/(scale/VOLT_SCALE))))"/>
    </Dynamics>
  </ComponentType>
</neuroml>
"#;

    /// Кастомный LEMS-`ComponentType` в `gateHHtauInf` (реальный Ih Poirazi
    /// 2003): `timeCourse`/`steadyState` с нестандартным `type` резолвятся в
    /// [`TauForm::Expr`]/[`VarForm::Expr`], и выражения совпадают с аналитикой
    /// (в т.ч. piecewise-ветка V≥−30 → tau = 1).
    #[test]
    fn custom_component_type_gate_lowers_to_lems_expr() {
        let ir = parse(POIRAZI_H_CHANNEL).unwrap();
        assert!(ir.warnings.is_empty(), "warnings: {:?}", ir.warnings);
        let ch = ir.channels.iter().find(|c| c.id == "hs").unwrap();
        let g = ch.gates[0].kinetics.clone().unwrap();
        let GateKinetics::TauInf { steady_state, tau } = g else {
            panic!("hs gate должен быть TauInf")
        };

        // steadyState: Poirazi_h_ss = 1/(1+exp((v-midpoint)/scale)), midpoint -82, scale 8.8.
        let VarForm::Expr(ss) = steady_state else {
            panic!("Poirazi_h_ss должен быть кастомным выражением")
        };
        let ss90 = ss.eval(-90.0);
        let want = 1.0f32 / (1.0f32 + ((-90.0f32 + 82.0) / 8.8).exp());
        assert!((ss90 - want).abs() < 1e-5, "ss(-90) = {ss90}, want {want}");

        // timeCourse: Poirazi_h_tau — piecewise: V < -30 → double-exp, else 1.
        let TauForm::Expr(tau) = tau else {
            panic!("Poirazi_h_tau должен быть кастомным выражением")
        };
        let tau90 = tau.eval(-90.0);
        let want_tau = 2.0f32
            * (1.0f32 / (((-90.0f32 + 145.0) / -17.5).exp() + ((-90.0f32 + 16.8) / 16.5).exp())
                + 5.0f32);
        assert!(
            (tau90 - want_tau).abs() < 1e-4,
            "tau(-90) = {tau90}, want {want_tau}"
        );
        assert!(
            (tau.eval(-20.0) - 1.0).abs() < 1e-6,
            "tau(-20) должен быть 1"
        );
    }

    /// Канонический кальмаровый аксон в NeuroML2: Na (m³h) + K (n⁴) через
    /// `ionChannelHH`/`gateHHrates`. Значения rate/midpoint/scale — точная
    /// NeuroML-кодировка `na_m_alpha` и т.д. (якорь round-trip из gap-анализа).
    const SQUID_CHANNELS: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="squid_channels">
  <ionChannelHH id="naChans" conductance="10pS" species="na">
    <gateHHrates id="m" instances="3">
      <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
    </gateHHrates>
    <gateHHrates id="h" instances="1">
      <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
      <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
    </gateHHrates>
  </ionChannelHH>
  <ionChannelHH id="kChans" conductance="10pS" species="k">
    <gateHHrates id="n" instances="4">
      <forwardRate type="HHExpLinearRate" rate="0.1per_ms" midpoint="-55mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="0.125per_ms" midpoint="-65mV" scale="-80mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;

    fn channel<'a>(ir: &'a Ir, id: &str) -> &'a IrChannel {
        ir.channels
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("channel `{id}` not parsed"))
    }

    fn gate_kinetics(ch: &IrChannel, id: &str) -> GateKinetics {
        ch.gates
            .iter()
            .find(|g| g.id == id)
            .and_then(|g| g.kinetics.clone())
            .unwrap_or_else(|| panic!("gate `{id}` not parsed (or not compilable)"))
    }

    /// m/h/n rate-функции round-trip бит-в-бит против канонических скалярных
    /// функций движка (na_m_alpha и т.д.) — якорь из gap-анализа §4.4.
    #[test]
    fn squid_ion_channel_rates_roundtrip_bit_exact() {
        let ir = parse(SQUID_CHANNELS).unwrap();
        assert_eq!(ir.channels.len(), 2);
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );

        let m = gate_kinetics(channel(&ir, "naChans"), "m");
        let h = gate_kinetics(channel(&ir, "naChans"), "h");
        let n = gate_kinetics(channel(&ir, "kChans"), "n");

        let vs = [
            -120.0f32, -100.0, -80.0, -65.0, -55.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0,
        ];
        for &v in &vs {
            let (a, b) = m.alpha_beta(v);
            assert_eq!(a.to_bits(), na_m_alpha(v).to_bits(), "m alpha @ {v}");
            assert_eq!(b.to_bits(), na_m_beta(v).to_bits(), "m beta @ {v}");
            let (a, b) = h.alpha_beta(v);
            assert_eq!(a.to_bits(), na_h_alpha(v).to_bits(), "h alpha @ {v}");
            assert_eq!(b.to_bits(), na_h_beta(v).to_bits(), "h beta @ {v}");
            let (a, b) = n.alpha_beta(v);
            assert_eq!(a.to_bits(), k_n_alpha(v).to_bits(), "n alpha @ {v}");
            assert_eq!(b.to_bits(), k_n_beta(v).to_bits(), "n beta @ {v}");
        }
    }

    /// `gateHHtauInf` (km/HCN-стиль) парсится без warning в каноническое
    /// `GateKinetics::TauInf` (steadyState + фиксированный tau).
    #[test]
    fn gate_hh_tau_inf_parses_without_warning() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hcn_test">
  <ionChannelHH id="hcn" conductance="10pS" species="hcn">
    <gateHHtauInf id="m" instances="1">
      <timeCourse type="fixedTimeCourse" tau="10ms"/>
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
    </gateHHtauInf>
  </ionChannelHH>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        assert_eq!(ir.channels.len(), 1);
        let g = gate_kinetics(&ir.channels[0], "m");
        assert_eq!(
            g,
            GateKinetics::TauInf {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 80.0,
                    scale: 10.0
                }),
                tau: TauForm::Fixed(10.0),
            }
        );
    }

    /// `gateHHInstantaneous` распознаётся (не «молча скипается»), но его
    /// кинетика ещё не компилируется движком → `kinetics = None` + честный kind.
    #[test]
    fn gate_hh_instantaneous_recognised_but_not_compilable() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="inst_test">
  <ionChannelHH id="ca" conductance="10pS" species="ca">
    <gateHHInstantaneous id="m" instances="2">
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-20mV" scale="5mV"/>
    </gateHHInstantaneous>
  </ionChannelHH>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        let ch = &ir.channels[0];
        assert_eq!(ch.gates[0].kind, "gateHHInstantaneous");
        // Algebraic gate — compiles to `Instantaneous` kinetics (no ODE).
        let kinetics = ch.gates[0].kinetics.clone().expect("compiles now");
        assert!(matches!(kinetics, GateKinetics::Instantaneous { .. }));
    }

    /// q10Settings (q10ExpTemp) парсится без ошибок.
    #[test]
    fn q10_settings_parse() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="q10_test">
  <ionChannelHH id="hcn" conductance="10pS" species="hcn">
    <gateHHrates id="m" instances="1">
      <q10Settings type="q10ExpTemp" q10Factor="3" experimentalTemp="6.3degC"/>
      <forwardRate type="HHExpRate" rate="0.01per_ms" midpoint="-70mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="0.01per_ms" midpoint="-70mV" scale="-10mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        let ch = &ir.channels[0];
        assert_eq!(ch.gates[0].kind, "gateHHrates");
        assert!(matches!(
            ch.gates[0].kinetics,
            Some(GateKinetics::Rates { .. })
        ));
        // q10 хранится на гейте (парсится, но движком пока не применяется).
        assert!(matches!(ch.gates[0].q10, Some(IrQ10::ExpTemp { .. })));
    }

    /// Гварды: нулевой `scale` и неположительные `instances` должны падать с
    /// понятной ошибкой, а не молча порождать inf/NaN или обёрнутый экспонент.
    #[test]
    fn rejects_zero_scale_and_bad_instances() {
        // scale = 0 → деление на ноль в замкнутой форме.
        let zero_scale = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">
  <ionChannelHH id="ch" conductance="10pS" species="na">
    <gateHHrates id="m" instances="3">
      <forwardRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="0mV"/>
      <reverseRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-1mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
        assert!(parse(zero_scale).is_err(), "scale=0 должен давать ошибку");

        // instances = 0 → не PositiveInteger.
        let zero_instances = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">
  <ionChannelHH id="ch" conductance="10pS" species="na">
    <gateHHrates id="m" instances="0">
      <forwardRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-10mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
        assert!(
            parse(zero_instances).is_err(),
            "instances=0 должен давать ошибку"
        );
    }

    /// `ionChannelKS`/`gateKS` (2-состояния closed/open + forward/reverse
    /// transitions) парсится в марковскую схему без warning, со скоростями
    /// `Directed`, совпадающими по битам с HH-формами.
    #[test]
    fn gate_ks_two_state_parses_without_warning() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_test">
  <ionChannelKS id="na_ks" conductance="10pS" species="na">
    <gateKS id="m" instances="3">
      <closedState id="c1"/>
      <openState id="o1"/>
      <forwardTransition from="c1" to="o1">
        <rate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        assert_eq!(ir.ks_channels.len(), 1);
        let gate = &ir.ks_channels[0].gates[0];
        assert_eq!(gate.n_states(), 2);
        assert_eq!(gate.open, vec![0.0, 1.0]);
        assert_eq!(gate.power, 3);
        assert_eq!(gate.transitions.len(), 2);
        // forward: closed(0) → open(1) со скоростью, бит-в-бит равной
        // `ClosedForm::ExpLinear { rate: 0.1, midpoint: 40, scale: 10 }`
        // (та же кодировка, что у `forwardRate` в `gateHHrates`).
        let fwd = &gate.transitions[0];
        assert_eq!(fwd.from, 0);
        assert_eq!(fwd.to, 1);
        assert_eq!(
            fwd.rate,
            MarkovRate::Directed {
                rate: ClosedForm::ExpLinear {
                    rate: 0.1,
                    midpoint: 40.0,
                    scale: 10.0
                }
            }
        );
        // reverse: open(1) → closed(0).
        let rev = &gate.transitions[1];
        assert_eq!(rev.from, 1);
        assert_eq!(rev.to, 0);
        assert_eq!(
            rev.rate,
            MarkovRate::Directed {
                rate: ClosedForm::Exp {
                    rate: 4.0,
                    midpoint: 65.0,
                    scale: 18.0
                }
            }
        );
    }

    /// `tauInfTransition` раскрывается в две направленные ноги
    /// (forward `inf/tau`, reverse `(1−inf)/tau`).
    #[test]
    fn gate_ks_tau_inf_expands_to_forward_reverse() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_tau">
  <ionChannelKS id="hcn_ks" conductance="10pS" species="hcn">
    <gateKS id="h" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="10ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        let gate = &ir.ks_channels[0].gates[0];
        assert_eq!(gate.transitions.len(), 2);
        let fwd = &gate.transitions[0];
        assert_eq!((fwd.from, fwd.to), (0, 1));
        assert_eq!(
            fwd.rate,
            MarkovRate::TauInfForward {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 80.0,
                    scale: 10.0
                }),
                tau: TauForm::Fixed(10.0),
            }
        );
        let rev = &gate.transitions[1];
        assert_eq!((rev.from, rev.to), (1, 0));
        assert_eq!(
            rev.rate,
            MarkovRate::TauInfReverse {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 80.0,
                    scale: 10.0
                }),
                tau: TauForm::Fixed(10.0),
            }
        );
    }

    /// Переход на неизвестное состояние — честная ошибка, а не молчаливый
    /// дроп ребра (который тихо сломал бы master equation).
    #[test]
    fn gate_ks_unknown_state_errors() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_bad">
  <ionChannelKS id="ch" conductance="10pS" species="na">
    <gateKS id="m" instances="1">
      <closedState id="c1"/>
      <openState id="o1"/>
      <forwardTransition from="c1" to="nope">
        <rate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-10mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
        assert!(
            parse(xml).is_err(),
            "неизвестное состояние должно давать ошибку"
        );
    }

    /// `gateFractional` с двумя субгейтами парсится без warning в
    /// [`FractionalGateSpec`] (веса + tau-inf субгейты), не как HH-гейт.
    #[test]
    fn gate_fractional_parses_without_warning() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="frac_test">
  <ionChannelHH id="ca" conductance="10pS" species="ca">
    <gateFractional id="m" instances="2">
      <subGate id="sg1" fractionalConductance="0.6">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-20mV" scale="5mV"/>
        <timeCourse type="fixedTimeCourse" tau="10ms"/>
      </subGate>
      <subGate id="sg2" fractionalConductance="0.4">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-40mV" scale="5mV"/>
        <timeCourse type="fixedTimeCourse" tau="20ms"/>
      </subGate>
    </gateFractional>
  </ionChannelHH>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        let ch = &ir.channels[0];
        assert!(
            ch.gates.is_empty(),
            "gateFractional не должен быть HH-гейтом"
        );
        assert_eq!(ch.fractional_gates.len(), 1);
        let fg = &ch.fractional_gates[0];
        assert_eq!(fg.power, 2);
        assert_eq!(fg.subgates.len(), 2);
        assert_eq!(fg.subgates[0].fractional_conductance, 0.6);
        assert_eq!(fg.subgates[0].tau, 10.0);
        assert_eq!(
            fg.subgates[0].steady_state,
            ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: 20.0,
                scale: 5.0
            }
        );
        assert_eq!(fg.subgates[1].fractional_conductance, 0.4);
        assert_eq!(fg.subgates[1].tau, 20.0);
    }

    /// Трёхсостоянийная кинетическая схема (closed ⇌ open ⇌ inactivated)
    /// парсится без warning: 3 состояния, open-веса [0,1,0], 4 перехода.
    #[test]
    fn gate_ks_three_state_parses_without_warning() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks3_test">
  <ionChannelKS id="na3" conductance="10pS" species="na">
    <gateKS id="m" instances="3">
      <closedState id="c1"/>
      <openState id="o1"/>
      <closedState id="i1"/>
      <forwardTransition from="c1" to="o1">
        <rate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
      </reverseTransition>
      <forwardTransition from="o1" to="i1">
        <rate type="HHExpRate" rate="0.2per_ms" midpoint="-65mV" scale="-20mV"/>
      </forwardTransition>
      <reverseTransition from="o1" to="i1">
        <rate type="HHExpRate" rate="0.01per_ms" midpoint="-35mV" scale="-10mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        let gate = &ir.ks_channels[0].gates[0];
        assert_eq!(gate.n_states(), 3);
        assert_eq!(gate.open, vec![0.0, 1.0, 0.0]);
        assert_eq!(gate.power, 3);
        assert_eq!(gate.transitions.len(), 4);
        // c1(0) ⇌ o1(1) ⇌ i1(2).
        assert_eq!((gate.transitions[0].from, gate.transitions[0].to), (0, 1));
        assert_eq!((gate.transitions[1].from, gate.transitions[1].to), (1, 0));
        assert_eq!((gate.transitions[2].from, gate.transitions[2].to), (1, 2));
        assert_eq!((gate.transitions[3].from, gate.transitions[3].to), (2, 1));
    }

    // ─── Фаза 5a: морфология ─────────────────────────────────

    /// Минимальная морфология: сома (корень) + дендрит (дочка), группы.
    /// Парсится без warning; segmentParent резолвится в индекс родителя.
    #[test]
    fn morphology_segments_and_groups_parse_without_warning() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morph_test">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="10" z="0" diameter="2"/>
      <distal x="0" y="100" z="0" diameter="2"/>
    </segment>
    <segmentGroup id="soma_group">
      <member segment="0"/>
    </segmentGroup>
    <segmentGroup id="dend_group">
      <member segment="1"/>
      <include segmentGroup="soma_group"/>
    </segmentGroup>
  </morphology>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        assert_eq!(ir.morphologies.len(), 1);
        let m = &ir.morphologies[0];
        assert_eq!(m.id, "m1");
        assert_eq!(m.segments.len(), 2);
        assert_eq!(m.segment_groups.len(), 2);

        // Сома — корень (нет родителя).
        let soma = &m.segments[0];
        assert_eq!(soma.id, 0);
        assert_eq!(soma.name.as_deref(), Some("soma"));
        assert!(soma.parent.is_none(), "сома должна быть корнем");
        assert_eq!(soma.distal.diameter, 20.0);

        // Дендрит — дочка сомы (fractionalLength по умолчанию 1.0).
        let dend = &m.segments[1];
        assert_eq!(dend.id, 1);
        let parent = dend.parent.as_ref().expect("дендрит должен иметь родителя");
        assert_eq!(parent.segment, 0);

        // Группы: soma_group = {0}, dend_group = {1} ∪ include soma_group.
        let soma_group = &m.segment_groups[0];
        assert_eq!(soma_group.id, "soma_group");
        assert_eq!(soma_group.members, vec![0]);
        let dend_group = &m.segment_groups[1];
        assert_eq!(dend_group.id, "dend_group");
        assert_eq!(dend_group.members, vec![1]);
        assert_eq!(dend_group.includes, vec!["soma_group"]);
    }

    /// `<segment>` без `<proximal>` наследует точку от родительского distal —
    /// этот случай должен парситься (proximal = None), а не падать.
    #[test]
    fn segment_without_proximal_parses_with_none() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morph_prox">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <distal x="0" y="100" z="0" diameter="2"/>
    </segment>
  </morphology>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        let m = &ir.morphologies[0];
        assert!(m.segments[0].proximal.is_some(), "сома имеет proximal");
        assert!(
            m.segments[1].proximal.is_none(),
            "дендрит без proximal → None"
        );
    }

    /// `<cell>` + `<biophysicalProperties>` резолвятся по id: cell ссылается на
    /// морфологию и биофизику; channelDensity/initMembPotential/resistivity
    /// парсятся в IR без warning.
    #[test]
    fn cell_with_biophysical_properties_parses() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="cell_test">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group">
      <member segment="0"/>
    </segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific" segmentGroup="soma_group"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        assert!(
            ir.warnings.is_empty(),
            "unexpected warnings: {:?}",
            ir.warnings
        );
        assert_eq!(ir.morpho_cells.len(), 1);
        let cell = &ir.morpho_cells[0];
        assert_eq!(cell.id, "c1");
        assert_eq!(cell.morphology, "m1");
        assert_eq!(cell.biophysics.as_deref(), Some("bp1"));

        let bp = ir.biophysics.get("bp1").expect("bp1 должен быть распарсен");
        assert_eq!(bp.membrane.channel_densities.len(), 1);
        let cd = &bp.membrane.channel_densities[0];
        assert_eq!(cd.ion_channel, "pas");
        assert_eq!(cd.erev, -70.0);
        assert_eq!(cd.segment_group, "soma_group");
        // 3.0 S_per_m2 → 3e-4 S/cm² (unit-aware, не raw strip).
        assert!(
            (cd.cond_density - 3e-4).abs() < 1e-9,
            "condDensity: {}",
            cd.cond_density
        );
        assert_eq!(bp.membrane.init_memb_potential, -65.0);
        assert_eq!(bp.membrane.specific_capacitance, 1.0);
        assert_eq!(bp.intracellular.resistivity, 100.0);
        // pas (ion non_specific) распознаётся как пассивный лик.
        assert_eq!(bp.passive_leak().map(|l| l.id.as_str()), Some("pasChans"));
    }

    /// `condDensity`/`erev` — обязательные атрибуты `channelDensity`;
    /// отсутствие любого из них должно давать ошибку, а не NaN.
    #[test]
    fn channel_density_missing_erev_errors() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad_cd">
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" ion="non_specific"/>
    </membraneProperties>
    <intracellularProperties/>
  </biophysicalProperties>
</neuroml>
"#;
        assert!(
            parse(xml).is_err(),
            "channelDensity без erev должен давать ошибку"
        );
    }

    /// `channelPopulation` и `channelPopulationNernst` парсятся
    /// (number/ionChannel; у обычного — ещё `erev`, у Nernst — динамическое
    /// обращение); single-channel conductance pS → µS.
    #[test]
    fn channel_population_and_nernst_parse() {
        let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="cp_test">
  <ionChannelKS id="kchan" conductance="625pS" species="k">
    <gateKS id="g" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-65mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="1ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelPopulation id="noisyK" ionChannel="kchan" number="64" erev="-40mV" ion="k"/>
      <channelPopulationNernst id="caPop" ionChannel="kchan" number="100" ion="ca"/>
    </membraneProperties>
    <intracellularProperties/>
  </biophysicalProperties>
</neuroml>
"#;
        let ir = parse(xml).unwrap();
        let bp = ir.biophysics.get("bp1").unwrap();
        assert_eq!(bp.membrane.channel_populations.len(), 1);
        let cp = &bp.membrane.channel_populations[0];
        assert_eq!(cp.id, "noisyK");
        assert_eq!(cp.ion_channel, "kchan");
        assert_eq!(cp.number, 64);
        assert_eq!(cp.erev, -40.0);
        // Nernst-популяция парсится отдельно (без `erev`, динамическое обращение).
        assert_eq!(bp.membrane.nernst_populations.len(), 1);
        let cpn = &bp.membrane.nernst_populations[0];
        assert_eq!(cpn.id, "caPop");
        assert_eq!(cpn.ion_channel, "kchan");
        assert_eq!(cpn.number, 100);
        // 625pS → 0.625 nS → 6.25e-4 µS.
        let ks = ir
            .ks_channels
            .iter()
            .find(|c| c.id == "kchan")
            .expect("kchan должен быть распарсен");
        assert!((ks.conductance.unwrap() - 6.25e-4).abs() < 1e-12);
        // Nernst-популяция больше НЕ warning (теперь поддерживается).
        assert!(
            !ir.warnings
                .iter()
                .any(|w| w.contains("channelPopulationNernst")),
            "channelPopulationNernst не должен warning: {:?}",
            ir.warnings
        );
    }

    /// `MorphoIR::resolve_segments`: `include`-цепи (dend → apical), `all`,
    /// и ошибка на неизвестную группу.
    #[test]
    fn resolve_segments_resolves_includes_and_all() {
        let m = MorphoIR {
            id: "m".into(),
            segments: (0..3)
                .map(|i| Segment {
                    id: i,
                    name: None,
                    proximal: None,
                    distal: Point3d {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        diameter: 2.0,
                    },
                    parent: None,
                })
                .collect(),
            segment_groups: vec![
                SegmentGroup {
                    id: "soma".into(),
                    members: vec![0],
                    includes: vec![],
                },
                SegmentGroup {
                    id: "dend".into(),
                    members: vec![1],
                    includes: vec!["apical".into()],
                },
                SegmentGroup {
                    id: "apical".into(),
                    members: vec![2],
                    includes: vec![],
                },
            ],
        };
        assert_eq!(m.resolve_segments("soma").unwrap(), vec![0]);
        assert_eq!(m.resolve_segments("dend").unwrap(), vec![1, 2]);
        assert_eq!(m.resolve_segments("all").unwrap(), vec![0, 1, 2]);
        assert!(m.resolve_segments("nope").is_err());
    }

    /// Конвенция Q10 (jNeuroML-семантика): `q10Fixed` якорится в 6.3 °C,
    /// `q10ExpTemp` — в свой `experimentalTemp`; оба дают 1.0 в якоре, а
    /// множитель — `q10^((T−T0)/10)`. Это ровно то, что NEURON-экспорт
    /// jNeuroML вставляет в .mod (множитель в `rates()`), поэтому сверка
    /// держится при любой температуре.
    #[test]
    fn ir_q10_factor_at_convention() {
        // q10Fixed = 2 при 6.3 °C → 1.0 (якорь); при 16.3 °C → 2.0.
        assert!((IrQ10::Fixed(2.0).factor_at(6.3) - 1.0).abs() < 1e-6);
        assert!((IrQ10::Fixed(2.0).factor_at(16.3) - 2.0).abs() < 1e-6);
        assert!(
            (IrQ10::Fixed(2.0).factor_at(35.0) - 2.0f32.powf((35.0 - 6.3) / 10.0)).abs() < 1e-6
        );

        // q10ExpTemp: factor 2.3 @ 23 °C → 1.0 при 23 °C; при 35 °C →
        // 2.3^((35−23)/10) ≈ 2.71 (это M-ток Poirazi при физиологической
        // температуре, применяется в cable-пути).
        let km = IrQ10::ExpTemp {
            factor: 2.3,
            experimental_temp: 23.0,
        };
        assert!((km.factor_at(23.0) - 1.0).abs() < 1e-6);
        assert!((km.factor_at(35.0) - 2.3f32.powf(1.2)).abs() < 1e-5);
        assert!((km.factor_at(6.3) - 2.3f32.powf((6.3 - 23.0) / 10.0)).abs() < 1e-6);

        // q10Fixed = 1 (na Poirazi) — температурно-независимый канал.
        assert!((IrQ10::Fixed(1.0).factor_at(35.0) - 1.0).abs() < 1e-6);
        assert!((IrQ10::Fixed(1.0).factor_at(6.3) - 1.0).abs() < 1e-6);
    }
}
