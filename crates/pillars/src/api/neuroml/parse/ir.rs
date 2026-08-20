// ─── Intermediate representation ───────────────────────────────
use super::*;

use std::collections::{HashMap, HashSet};

use crate::kernels::hh_channels::{FractionalGateSpec, GateKinetics, MarkovGateSpec};

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
