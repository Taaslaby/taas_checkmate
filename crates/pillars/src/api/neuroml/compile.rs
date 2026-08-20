//! # NeuroML IR → pillars compiler (`api::neuroml::compile`)
//!
//! Lowers the parsed [`Ir`] into a built [`Network`] plus a schedule of
//! external stimuli. This is where NeuroML's declarations (cells, populations,
//! projections, inputs) become live pillars populations and connections.
//!
//! Cell mapping:
//!
//! | NeuroML            | pillars model            |
//! |--------------------|--------------------------|
//! | `izhikevichCell`   | `IzhikevichModel` (2003) |
//! | `iaf*Cell`         | `LIFModel`               |
//! | `adExIaFCell`      | `AdExModel`              |
//!
//! Synaptic **kinetics** are applied: `expOneSynapse` / `expTwoSynapse` /
//! `alphaSynapse` / `expThreeSynapse` with time constants are lowered to a
//! kinetic conductance waveform ([`SynapseKinetics`], current
//! `g(t)·(erev − V)`) registered on the [`Network`]; synapses without kinetics
//! remain static delta synapses (`weight · gbase` as edge weight, `delay` as
//! axonal delay). **Short-term plasticity** (`blockingPlasticSynapse` with a
//! `tsodyksMarkramDepMechanism` / `tsodyksMarkramDepFacMechanism` child) is
//! registered as a [`PlasticityProfile`](crate::api::PlasticityProfile) and
//! assigned to edges via `Edge::with_synapse_plasticity`; the runtime scales
//! every delivered weight by the edge's current `R·U` (Tsodyks–Markram).

use std::collections::HashMap;

use crate::api::graph::MAX_DELAY_SLOTS;
use crate::api::network::{Edge, Network, SigmoidGradedSynapse};
use crate::api::population::PopulationId;
use crate::zoo::neurons::{
    AdExModel, FHNModel, HHModel, HHModelParamIdx, HHModelStateIdx, Izhikevich2007Model,
    Izhikevich2007ModelParamIdx, IzhikevichModel, IzhikevichModelParamIdx, LIFModel,
    ADEX_CUDA_SOURCE, FHN_CUDA_SOURCE, HH_CUDA_SOURCE, IZHIKEVICH_2007_CUDA_SOURCE,
    IZHIKEVICH_CUDA_SOURCE, LIF_CUDA_SOURCE,
};
use crate::zoo::Model;

use crate::api::{
    ConductancePopulation, MarkovChannel, MarkovPopulation, PlasticityProfile, SynapseKinetics,
};
use crate::kernels::hh_channels::{
    gate_kinetics_to_markov, ChannelMeta, GateKinetics, GateType, MarkovGateSpec,
};

use super::discretize::{
    discretize, passive_leak_reversal, thin_shell_pool_flux_scale, ActiveChannel, CaPool,
    CableParams, CablePopulation, GhkActiveChannel, MarkovActiveChannel, NernstActiveChannel,
    StochasticMarkovActiveChannel, StochasticNernstActiveChannel,
};
use super::parse::{
    is_passive_leak, resolve_channel, resolve_markov_channel, BiophysicalProperties, Cell,
    CellSpec, DoubleSynapseSpec, GapJunctionSpec, GradedSynapseSpec, Ir, IrChannel,
    IrChannelDensity, IrChannelPopulation, IrConcentrationModel, IrKsChannel, MorphoCell, MorphoIR,
    NonUniformParam, PlasticitySpec, PoissonSpec, ResolvedChannel, SigmoidGradedSynapseSpec,
    SpikeGenKind, Synapse, SynapseKind,
};

use super::{
    CableCompartment, CableInfo, ClampSchedule, GpuPopMeta, LoadedNetwork, NeuromlError, Stimulus,
    StimulusKind, VoltageClamp,
};

/// Regular spike train for `spikeGenerator`: first spike at `period_ms`, then
/// every `period_ms` until `duration_ms` (the NeuroML2 semantics).
fn regular_spike_times(period_ms: f32, duration_ms: f32) -> Vec<f32> {
    if !period_ms.is_finite() || period_ms <= 0.0 || !duration_ms.is_finite() {
        return Vec::new();
    }
    let mut times = Vec::new();
    let mut t = period_ms;
    while t < duration_ms {
        times.push(t);
        t += period_ms;
    }
    times
}

/// Deterministic SplitMix64 generator — the whole ISI schedule is precomputed
/// at compile time from a seed, so the runtime stays stateless (no interior
/// mutability in `injections_at`), the same seed reproduces the same train,
/// and CPU/GPU/reruns agree.
struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Spike train for `spikeGeneratorRandom`: uniform ISI in `[min_isi, max_isi]`
/// (LEMS `isi = minISI + MSEC·random((maxISI−minISI)/MSEC)`), each next spike
/// `isi` after the **last planned** time, not the current one. Seeded → the
/// same model + seed always yields the same deterministic train.
fn random_isi_spike_times(
    min_isi_ms: f32,
    max_isi_ms: f32,
    duration_ms: f32,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if !min_isi_ms.is_finite()
        || !max_isi_ms.is_finite()
        || min_isi_ms < 0.0
        || max_isi_ms < min_isi_ms
    {
        return Err(format!(
            "spikeGeneratorRandom: need 0 <= minISI <= maxISI, got [{min_isi_ms}, {max_isi_ms}]"
        ));
    }
    let span = max_isi_ms - min_isi_ms;
    let mut rng = SeededRng::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0f32;
    for _ in 0..10_000_000 {
        let isi = min_isi_ms + span * rng.unit();
        if isi <= 0.0 || !isi.is_finite() {
            break; // pathological zero-ISI guard (minISI == 0 draws)
        }
        t += isi;
        if t >= duration_ms {
            break;
        }
        times.push(t);
    }
    Ok(times)
}

/// Spike train for `spikeGeneratorRefPoisson`: maximum-entropy ISI over
/// `[minimum_isi, ∞)` with mean `1/rate_hz` (LEMS `isi = minimumISI −
/// (averageIsi−minimumISI)·log(random(1))`). Requires `averageIsi > minimumISI`
/// (the mean must exceed the support bound). Seeded → deterministic.
fn ref_poisson_spike_times(
    rate_hz: f32,
    min_isi_ms: f32,
    duration_ms: f32,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if !rate_hz.is_finite() || rate_hz <= 0.0 || !min_isi_ms.is_finite() || min_isi_ms < 0.0 {
        return Err(format!(
            "spikeGeneratorRefPoisson: need rate > 0 and minimumISI >= 0, got {rate_hz} Hz, {min_isi_ms} ms"
        ));
    }
    let avg_isi_ms = 1000.0 / rate_hz;
    if avg_isi_ms <= min_isi_ms {
        return Err(format!(
            "spikeGeneratorRefPoisson: average ISI ({avg_isi_ms:.2} ms) must exceed minimumISI ({min_isi_ms} ms)"
        ));
    }
    let mut rng = SeededRng::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0f32;
    for _ in 0..10_000_000 {
        // ln(1−u) with u ∈ [0,1) avoids log(0) = +∞.
        let isi = min_isi_ms - (avg_isi_ms - min_isi_ms) * (1.0 - rng.unit()).ln();
        if isi <= 0.0 || !isi.is_finite() {
            break;
        }
        t += isi;
        if t >= duration_ms {
            break;
        }
        times.push(t);
    }
    Ok(times)
}

/// Resolve one input-generator id into `(kind, start_ms, end_ms, amplitude)`
/// stimuli (LEMS formulas from `NeuroML2CoreTypes/Inputs.xml`).
///
/// * `pulseGenerator` → one `Current` window;
/// * `sineGenerator` → `[Sine]` window, `i = w·A·sin(phase + 2π(t−delay)/period)`;
/// * `rampGenerator` → the `[Ramp]` window plus, when `baselineAmplitude ≠ 0`,
///   constant-current stimuli for `[0, delay)` and `[delay+duration, ∞)` (LEMS
///   holds the baseline outside the window);
/// * `compoundInput` → recurses into its children; each child becomes its own
///   stimulus, scaled by the compound's `weight`;
/// * `timedSynapticInput` → `SpikeTimes` pulses of `weight·gbase` at each
///   listed spike time (the synapse's kinetics are not simulated for direct
///   inputs — delta-pulse lowering, like `spikeArray`).
///
/// Returns an empty `Vec` for unknown ids (the caller turns it into an error).
fn resolve_input_gen(
    gen_id: &str,
    ir: &Ir,
    synapses: &HashMap<&str, &Synapse>,
    duration_ms: f32,
    scale: f32,
    seed: &mut usize,
    depth: usize,
) -> Result<Vec<(StimulusKind, f32, f32, f32)>, String> {
    if depth > 8 {
        return Err("compoundInput nesting too deep (cycle?)".to_string());
    }
    let mut next_seed = || {
        let s = *seed;
        *seed += 1;
        s as u32
    };
    if let Some(pg) = ir.pulse_gens.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::Current,
            pg.delay_ms,
            pg.delay_ms + pg.duration_ms,
            pg.amplitude * scale,
        )])
    } else if let Some(ps) = ir.poisson_synapses.iter().find(|p| p.id == gen_id) {
        Ok(vec![(
            StimulusKind::Poisson {
                rate_hz: ps.rate_hz,
                seed: next_seed(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(tp) = ir
        .transient_poisson_synapses
        .iter()
        .find(|p| p.id == gen_id)
    {
        Ok(vec![(
            StimulusKind::Poisson {
                rate_hz: tp.rate_hz,
                seed: next_seed(),
            },
            tp.delay_ms,
            tp.delay_ms + tp.duration_ms,
            scale,
        )])
    } else if let Some(sa) = ir.spike_arrays.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: sa.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(r) = ir.spike_array_refs.iter().find(|g| g.id == gen_id) {
        let arr = ir
            .spike_arrays
            .iter()
            .find(|g| g.id == r.array)
            .ok_or_else(|| {
                format!(
                    "spikeArrayRef `{}` references unknown spikeArray `{}`",
                    r.id, r.array
                )
            })?;
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: arr.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(sg) = ir.spike_gens.iter().find(|g| g.id == gen_id) {
        match sg.kind {
            SpikeGenKind::Regular { period_ms } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: regular_spike_times(period_ms, duration_ms),
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            SpikeGenKind::Poisson { rate_hz } => Ok(vec![(
                StimulusKind::Poisson {
                    rate_hz,
                    seed: next_seed(),
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            // Random/refractory-ISI generators: the full schedule is
            // precomputed from the seed at compile time → deterministic
            // `SpikeTimes` (runtime stays stateless).
            SpikeGenKind::Random {
                min_isi_ms,
                max_isi_ms,
            } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: random_isi_spike_times(
                        min_isi_ms,
                        max_isi_ms,
                        duration_ms,
                        next_seed() as u64,
                    )?,
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            SpikeGenKind::RefPoisson {
                rate_hz,
                min_isi_ms,
            } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: ref_poisson_spike_times(
                        rate_hz,
                        min_isi_ms,
                        duration_ms,
                        next_seed() as u64,
                    )?,
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
        }
    } else if let Some(s) = ir.sine_gens.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::Sine {
                period_ms: s.period_ms,
                phase: s.phase,
            },
            s.delay_ms,
            s.delay_ms + s.duration_ms,
            s.amplitude * s.weight * scale,
        )])
    } else if let Some(r) = ir.ramp_gens.iter().find(|g| g.id == gen_id) {
        let w = r.weight * scale;
        let mut v = Vec::with_capacity(3);
        if r.baseline_amp != 0.0 {
            v.push((StimulusKind::Current, 0.0, r.delay_ms, r.baseline_amp * w));
            v.push((
                StimulusKind::Current,
                r.delay_ms + r.duration_ms,
                f32::INFINITY,
                r.baseline_amp * w,
            ));
        }
        v.push((
            StimulusKind::Ramp {
                start_amp: r.start_amp * w,
                finish_amp: r.finish_amp * w,
            },
            r.delay_ms,
            r.delay_ms + r.duration_ms,
            0.0,
        ));
        Ok(v)
    } else if let Some(c) = ir.compounds.iter().find(|g| g.id == gen_id) {
        let mut v = Vec::new();
        for child in &c.children {
            for (kind, start, end, amp) in
                resolve_input_gen(child, ir, synapses, duration_ms, scale, seed, depth + 1)?
            {
                v.push((kind, start, end, amp * c.weight));
            }
        }
        Ok(v)
    } else if let Some(t) = ir.timed_syn_inputs.iter().find(|g| g.id == gen_id) {
        let gbase = synapses
            .get(t.synapse.as_str())
            .and_then(|s| s.gbase)
            .unwrap_or(1.0);
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: t.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            t.weight * gbase * scale,
        )])
    } else {
        Ok(Vec::new())
    }
}

/// Lower the IR into a built network + stimulus schedule.
pub(crate) fn compile(
    mut ir: Ir,
    dt_ms: f32,
    duration_ms: f32,
    celsius: f32,
) -> Result<LoadedNetwork, NeuromlError> {
    if !dt_ms.is_finite() || dt_ms <= 0.0 {
        return Err(NeuromlError::parse("dt must be finite and > 0"));
    }
    if !duration_ms.is_finite() || duration_ms <= 0.0 {
        return Err(NeuromlError::parse("duration must be finite and > 0"));
    }
    if !celsius.is_finite() {
        return Err(NeuromlError::parse("celsius must be finite"));
    }

    let cells: HashMap<&str, &Cell> = ir.cells.iter().map(|c| (c.id.as_str(), c)).collect();
    let synapses: HashMap<&str, &Synapse> =
        ir.synapses.iter().map(|s| (s.id.as_str(), s)).collect();
    let double_synapses: HashMap<&str, &DoubleSynapseSpec> = ir
        .double_synapses
        .iter()
        .map(|d| (d.id.as_str(), d))
        .collect();
    let morphologies: HashMap<&str, &MorphoIR> =
        ir.morphologies.iter().map(|m| (m.id.as_str(), m)).collect();
    let biophysics: HashMap<&str, &BiophysicalProperties> =
        ir.biophysics.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let morpho_cells: HashMap<&str, &MorphoCell> =
        ir.morpho_cells.iter().map(|c| (c.id.as_str(), c)).collect();
    let hh_channels: HashMap<&str, &IrChannel> =
        ir.channels.iter().map(|c| (c.id.as_str(), c)).collect();
    let ks_channels: HashMap<&str, &IrKsChannel> =
        ir.ks_channels.iter().map(|c| (c.id.as_str(), c)).collect();
    let concentration_models: HashMap<&str, &IrConcentrationModel> = ir
        .concentration_models
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let poisson_synapses: HashMap<&str, &PoissonSpec> = ir
        .poisson_synapses
        .iter()
        .map(|p| (p.id.as_str(), p))
        .collect();
    let gap_junctions: HashMap<&str, &GapJunctionSpec> = ir
        .gap_junctions
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();
    let graded_synapses: HashMap<&str, &GradedSynapseSpec> = ir
        .graded_synapses
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();
    let sigmoid_graded_synapses: HashMap<&str, &SigmoidGradedSynapseSpec> = ir
        .sigmoid_graded_synapses
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();

    let mut net = Network::new();
    // Kinetic synapses: register each expOne/expTwo/alpha/expThree profile
    // once; edges carry the 1-based profile id (`0` = static delta synapse).
    let mut kin_ids: HashMap<&str, u8> = HashMap::new();
    for s in &ir.synapses {
        let k = match &s.kind {
            SynapseKind::ExpOne { tau_decay } => match tau_decay {
                Some(td) => SynapseKinetics::exp_decay(*td),
                None => continue, // static delta synapse
            },
            SynapseKind::ExpTwo {
                tau_rise,
                tau_decay,
            } => match (tau_rise, tau_decay) {
                (Some(tr), Some(td)) if *tr > 0.0 => SynapseKinetics::exp_rise_decay(*tr, *td),
                (_, Some(td)) => SynapseKinetics::exp_decay(*td),
                _ => continue, // static delta synapse
            },
            SynapseKind::Alpha { tau } => SynapseKinetics::alpha(*tau),
            SynapseKind::ExpThree {
                tau_rise,
                tau_decay1,
                tau_decay2,
                gbase1,
                gbase2,
            } => SynapseKinetics::exp_three(*tau_rise, *tau_decay1, *tau_decay2, *gbase1, *gbase2),
        }
        .with_erev(s.erev.unwrap_or(0.0));
        let id = net
            .register_kinetics(k, dt_ms)
            .map_err(|e| NeuromlError::parse(format!("synapse `{}`: {e}", s.id)))?;
        kin_ids.insert(s.id.as_str(), id);
    }
    // Short-term plasticity (`blockingPlasticSynapse`): register each
    // Tsodyks–Markram profile once; edges carry the 1-based profile id
    // (`0` = no plasticity). Independent of kinetics — a plastic synapse may
    // be static delta or expOne/expTwo under the hood.
    let mut plast_ids: HashMap<&str, u8> = HashMap::new();
    for s in &ir.synapses {
        let p = match &s.plast {
            Some(PlasticitySpec::TsodyksDep {
                init_release_prob,
                tau_rec,
            }) => PlasticityProfile::depression(*init_release_prob, *tau_rec),
            Some(PlasticitySpec::TsodyksDepFac {
                init_release_prob,
                tau_rec,
                tau_fac,
            }) => {
                PlasticityProfile::depression_facilitation(*init_release_prob, *tau_rec, *tau_fac)
            }
            None => continue,
        };
        let id = net
            .register_plasticity(p, dt_ms)
            .map_err(|e| NeuromlError::parse(format!("synapse `{}` plasticity: {e}", s.id)))?;
        plast_ids.insert(s.id.as_str(), id);
    }
    // Cable geometry per population (index = PopulationId); `None` for
    // point-neuron populations — filled together with the populations below.
    let mut cable: Vec<Option<CableInfo>> = Vec::new();
    // Warnings emitted during lowering (recognised-but-unsupported Ca²⁺ forms).
    let mut extra_warnings: Vec<String> = Vec::new();
    // (population name) → (PopulationId, size).
    let mut pop_map: HashMap<&str, (PopulationId, usize)> = HashMap::new();
    // (population name) → injected-current scale (mV per nA). 1.0 for models
    // whose native current unit is already mV; 1000/gL for voltage-normalised
    // AdEx. Applied to pulse amplitudes when building stimuli.
    let mut pop_scale: HashMap<&str, f32> = HashMap::new();
    // (population name) → constant bias current (FHN `I` — встроенный стимул,
    // не из pulseGenerator). Синтезируется в вечный стимул ниже.
    let mut pop_bias: HashMap<&str, f32> = HashMap::new();
    // GPU metadata, indexed by PopulationId (populations are added in order).
    let mut gpu_pops: Vec<GpuPopMeta> = Vec::with_capacity(ir.pops.len());

    for pop in &ir.pops {
        let (pid, meta, scale, cell_cable) = if let Some(cell) = cells.get(pop.component.as_str()) {
            let (pid, meta) = add_population(&mut net, cell, pop.size, dt_ms, celsius);
            let scale = current_scale(cell);
            if let CellSpec::Fhn { bias, .. } = &cell.spec {
                if *bias != 0.0 {
                    pop_bias.insert(pop.id.as_str(), *bias);
                }
            }
            (pid, meta, scale, None)
        } else if let Some(mc) = morpho_cells.get(pop.component.as_str()) {
            let (pid, meta, info) = add_morpho_population(
                &mut net,
                mc,
                &morphologies,
                &biophysics,
                &hh_channels,
                &ks_channels,
                &concentration_models,
                &mut extra_warnings,
                pop.size,
                dt_ms,
                celsius,
            )?;
            // Кабель принимает ток в нативных nA (без масштаба mV/nA).
            (pid, meta, 1.0, Some(info))
        } else {
            return Err(NeuromlError::parse(format!(
                "population `{}` references unknown cell `{}`",
                pop.id, pop.component
            )));
        };
        pop_map.insert(pop.id.as_str(), (pid, pop.size));
        pop_scale.insert(pop.id.as_str(), scale);
        gpu_pops.push(meta);
        cable.push(cell_cable);
    }

    // ── projections → synapses ──
    // `poissonFiringSynapse` can be the presynaptic population: its spikes are
    // an external Poisson source routed through synapses (weight × gbase,
    // axonal delay, optional kinetics) to the postsynaptic population.
    let mut poisson_map: HashMap<&str, usize> = HashMap::new();
    for proj in &ir.projections {
        let (post_pid, post_size) = *pop_map.get(proj.post_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "projection references unknown population `{}`",
                proj.post_pop
            ))
        })?;
        let syn = proj
            .synapse
            .as_deref()
            .and_then(|id| synapses.get(id).copied());
        let gbase = syn.map(|s| s.weight_scale()).unwrap_or(1.0);
        let syn_type = proj
            .synapse
            .as_deref()
            .and_then(|id| kin_ids.get(id).copied())
            .unwrap_or(0);
        let plast_type = proj
            .synapse
            .as_deref()
            .and_then(|id| plast_ids.get(id).copied())
            .unwrap_or(0);
        // `doubleSynapse` — одна связь несёт два независимых синапса (AMPA+
        // NMDA): опускается в ДВА ребра на connection (по одному на каждый
        // дочерний синапс, со своим gbase/кинетикой/пластикой).
        let double = proj
            .synapse
            .as_deref()
            .and_then(|id| double_synapses.get(id));

        if let Some(ps) = poisson_synapses.get(proj.pre_pop.as_str()) {
            if double.is_some() {
                return Err(NeuromlError::parse(format!(
                    "doubleSynapse `{}` не может быть синапсом poisson-проекции",
                    proj.pre_pop
                )));
            }
            let source = match poisson_map.get(proj.pre_pop.as_str()) {
                Some(&s) => s,
                None => {
                    let seed = poisson_map.len() as u32;
                    let s = net.add_poisson_source(ps.rate_hz, dt_ms, seed);
                    poisson_map.insert(proj.pre_pop.as_str(), s);
                    s
                }
            };
            let mut posts = Vec::with_capacity(proj.connections.len());
            let mut weights = Vec::with_capacity(proj.connections.len());
            let mut delays = Vec::with_capacity(proj.connections.len());
            let mut syn_types = Vec::with_capacity(proj.connections.len());
            let mut plast_types = Vec::with_capacity(proj.connections.len());
            for c in &proj.connections {
                if c.pre_idx != 0 {
                    return Err(NeuromlError::parse(format!(
                        "poissonFiringSynapse `{}` is a single source: preCellId index must be 0, got {}",
                        proj.pre_pop, c.pre_idx
                    )));
                }
                if c.post_idx >= post_size {
                    return Err(NeuromlError::parse(format!(
                        "connection {}[{}] → {}[{}] out of range",
                        proj.pre_pop, c.pre_idx, proj.post_pop, c.post_idx
                    )));
                }
                let delay_steps =
                    ((c.delay_ms / dt_ms).round() as i64).clamp(0, (MAX_DELAY_SLOTS - 1) as i64);
                posts.push(c.post_idx as u32);
                weights.push(c.weight * gbase);
                delays.push(delay_steps as u8);
                syn_types.push(syn_type);
                plast_types.push(plast_type);
            }
            if !posts.is_empty() {
                net.connect_poisson_source(
                    source,
                    post_pid,
                    &posts,
                    &weights,
                    &delays,
                    &syn_types,
                    &plast_types,
                );
            }
            continue;
        }

        let (pre_pid, pre_size) = *pop_map.get(proj.pre_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "projection references unknown population `{}`",
                proj.pre_pop
            ))
        })?;

        let mut edges = Vec::with_capacity(proj.connections.len());
        for c in &proj.connections {
            if c.pre_idx >= pre_size || c.post_idx >= post_size {
                return Err(NeuromlError::parse(format!(
                    "connection {}[{}] → {}[{}] out of range",
                    proj.pre_pop, c.pre_idx, proj.post_pop, c.post_idx
                )));
            }
            let delay_steps =
                ((c.delay_ms / dt_ms).round() as i64).clamp(0, (MAX_DELAY_SLOTS - 1) as i64);
            // Double synapse: one edge per child mechanism (same pre/post,
            // delay; per-child gbase/kinetics/plasticity).
            if let Some(ds) = double {
                for child in [&ds.synapse1, &ds.synapse2] {
                    let child_syn = synapses.get(child.as_str());
                    let gb = child_syn.map(|s| s.weight_scale()).unwrap_or(1.0);
                    let st = kin_ids.get(child.as_str()).copied().unwrap_or(0);
                    let pt = plast_ids.get(child.as_str()).copied().unwrap_or(0);
                    let e = if pt > 0 {
                        Edge::with_synapse_plasticity(
                            c.pre_idx as u32,
                            c.post_idx as u32,
                            c.weight * gb,
                            delay_steps as u8,
                            st,
                            pt,
                        )
                    } else {
                        Edge::with_synapse(
                            c.pre_idx as u32,
                            c.post_idx as u32,
                            c.weight * gb,
                            delay_steps as u8,
                            st,
                        )
                    };
                    edges.push(e);
                }
                continue;
            }
            let edge = if plast_type > 0 {
                Edge::with_synapse_plasticity(
                    c.pre_idx as u32,
                    c.post_idx as u32,
                    c.weight * gbase,
                    delay_steps as u8,
                    syn_type,
                    plast_type,
                )
            } else {
                Edge::with_synapse(
                    c.pre_idx as u32,
                    c.post_idx as u32,
                    c.weight * gbase,
                    delay_steps as u8,
                    syn_type,
                )
            };
            edges.push(edge);
        }
        if !edges.is_empty() {
            net.connect_edges(pre_pid, post_pid, &edges);
        }
    }

    // ── electrical projections → gap junctions ──
    for proj in &ir.electrical_projections {
        let (pre_pid, pre_size) = *pop_map.get(proj.pre_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "electricalProjection references unknown population `{}`",
                proj.pre_pop
            ))
        })?;
        let (post_pid, post_size) = *pop_map.get(proj.post_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "electricalProjection references unknown population `{}`",
                proj.post_pop
            ))
        })?;
        // Gap-junction conductance from the referenced `gapJunction`, else 1.0.
        let g = proj
            .gap_junction
            .as_deref()
            .and_then(|id| gap_junctions.get(id))
            .and_then(|g| g.conductance)
            .unwrap_or(1.0);
        for c in &proj.connections {
            if c.pre_idx >= pre_size || c.post_idx >= post_size {
                return Err(NeuromlError::parse(format!(
                    "electrical connection {}[{}] ↔ {}[{}] out of range",
                    proj.pre_pop, c.pre_idx, proj.post_pop, c.post_idx
                )));
            }
            net.connect_gap_junction(
                pre_pid,
                c.pre_idx as u32,
                post_pid,
                c.post_idx as u32,
                c.weight * g,
            );
        }
    }

    // ── continuous projections → graded (analog) synapses ──
    // A `continuousProjection` may reference a `linearGradedSynapse`/
    // `silentSynapse` (one-way ohmic) or a sigmoid `gradedSynapse` (stateful).
    let mut sigmoid_profile_map: HashMap<&str, usize> = HashMap::new();
    for proj in &ir.continuous_projections {
        let (pre_pid, pre_size) = *pop_map.get(proj.pre_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "continuousProjection references unknown population `{}`",
                proj.pre_pop
            ))
        })?;
        let (post_pid, post_size) = *pop_map.get(proj.post_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "continuousProjection references unknown population `{}`",
                proj.post_pop
            ))
        })?;

        if let Some(sg) = proj
            .synapse
            .as_deref()
            .and_then(|id| sigmoid_graded_synapses.get(id))
        {
            let profile = match sigmoid_profile_map.get(sg.id.as_str()) {
                Some(&p) => p,
                None => {
                    let spec =
                        SigmoidGradedSynapse::new(sg.vth, sg.delta, sg.k, sg.erev, sg.conductance);
                    let p = net.register_sigmoid_graded(spec, dt_ms).map_err(|e| {
                        NeuromlError::parse(format!("gradedSynapse `{}`: {e}", sg.id))
                    })?;
                    sigmoid_profile_map.insert(sg.id.as_str(), p);
                    p
                }
            };
            for c in &proj.connections {
                if c.pre_idx >= pre_size || c.post_idx >= post_size {
                    return Err(NeuromlError::parse(format!(
                        "continuous connection {}[{}] → {}[{}] out of range",
                        proj.pre_pop, c.pre_idx, proj.post_pop, c.post_idx
                    )));
                }
                net.connect_sigmoid_graded(
                    profile,
                    pre_pid,
                    c.pre_idx as u32,
                    post_pid,
                    c.post_idx as u32,
                    c.weight,
                );
            }
            continue;
        }

        // Graded-synapse conductance from the referenced `linearGradedSynapse`
        // (or `silentSynapse` → 0), else 1.0.
        let g = proj
            .synapse
            .as_deref()
            .and_then(|id| graded_synapses.get(id))
            .and_then(|g| g.conductance)
            .unwrap_or(1.0);
        for c in &proj.connections {
            if c.pre_idx >= pre_size || c.post_idx >= post_size {
                return Err(NeuromlError::parse(format!(
                    "continuous connection {}[{}] → {}[{}] out of range",
                    proj.pre_pop, c.pre_idx, proj.post_pop, c.post_idx
                )));
            }
            net.connect_graded_synapse(
                pre_pid,
                c.pre_idx as u32,
                post_pid,
                c.post_idx as u32,
                c.weight * g,
            );
        }
    }

    net.build();

    // ── explicit inputs → stimuli ──
    let mut inputs = Vec::with_capacity(ir.inputs.len());
    let mut clamps = Vec::new();
    for inp in ir.inputs.iter() {
        let (pid, size) = pop_map.get(inp.target_pop.as_str()).ok_or_else(|| {
            NeuromlError::parse(format!(
                "input targets unknown population `{}`",
                inp.target_pop
            ))
        })?;
        if inp.target_idx >= *size {
            return Err(NeuromlError::parse(format!(
                "input target {}[{}] out of range (size {size})",
                inp.target_pop, inp.target_idx
            )));
        }
        // nA → internal current (mV) for the target population's model.
        let scale = pop_scale
            .get(inp.target_pop.as_str())
            .copied()
            .unwrap_or(1.0);

        // A `voltageClamp`(Triple) is not a scheduled current — it is a
        // V-feedback input: each step injects weight·(target(t) − V)/R against
        // the neuron's live membrane potential (see
        // `LoadedNetwork::clamp_currents`).
        if let Some(vc) = ir.voltage_clamps.iter().find(|c| c.id == inp.generator) {
            clamps.push(VoltageClamp {
                pop: *pid,
                indices: vec![inp.target_idx as u32],
                schedule: ClampSchedule::Window {
                    target_mv: vc.target_mv,
                },
                series_resistance_mohm: vc.series_resistance_mohm,
                weight: vc.weight * scale,
                delay_ms: vc.delay_ms,
                duration_ms: vc.duration_ms,
            });
            continue;
        }
        if let Some(vt) = ir
            .voltage_clamp_triples
            .iter()
            .find(|c| c.id == inp.generator)
        {
            clamps.push(VoltageClamp {
                pop: *pid,
                indices: vec![inp.target_idx as u32],
                schedule: ClampSchedule::Triple {
                    conditioning_mv: vt.conditioning_mv,
                    testing_mv: vt.testing_mv,
                    return_mv: vt.return_mv,
                    active: vt.active,
                },
                series_resistance_mohm: vt.series_resistance_mohm,
                weight: vt.weight * scale,
                delay_ms: vt.delay_ms,
                duration_ms: vt.duration_ms,
            });
            continue;
        }

        // Resolve the input generator across the supported kinds. Spike-based
        // inputs (`spikeArray`/`poissonFiringSynapse`) deliver a single `dt`-wide
        // pulse of 1.0 raw nA per spike (`scale` converts it, like a current).
        // `compoundInput` expands to one stimulus per child; a `rampGenerator`
        // with a non-zero baseline additionally emits constant-current stimuli
        // for the regions outside its window (LEMS semantics).
        let gen_id = inp.generator.as_str();
        let mut seed = 0usize;
        let resolved = resolve_input_gen(gen_id, &ir, &synapses, duration_ms, scale, &mut seed, 0)
            .map_err(|e| NeuromlError::parse(format!("input `{}`: {e}", inp.generator)))?;
        if resolved.is_empty() {
            return Err(NeuromlError::parse(format!(
                "input references unknown input generator `{}`",
                inp.generator
            )));
        }

        for (kind, start_ms, end_ms, amplitude) in resolved {
            inputs.push(Stimulus {
                pop: *pid,
                indices: vec![inp.target_idx as u32],
                start_ms,
                end_ms,
                amplitude,
                kind,
            });
        }
    }

    // ── constant bias (FHN `I`) → вечный стимул на все нейроны ──
    for pop in &ir.pops {
        let Some(bias) = pop_bias.get(pop.id.as_str()) else {
            continue;
        };
        let (pid, size) = *pop_map
            .get(pop.id.as_str())
            .expect("bias pop must exist in pop_map");
        inputs.push(Stimulus {
            pop: pid,
            indices: (0..size as u32).collect(),
            start_ms: 0.0,
            end_ms: f32::INFINITY,
            amplitude: *bias,
            kind: StimulusKind::Current,
        });
    }

    ir.warnings.extend(extra_warnings);
    Ok(LoadedNetwork {
        network: net,
        inputs,
        clamps,
        dt_ms,
        duration_ms,
        gpu_pops,
        cable,
        warnings: ir.warnings,
    })
}

/// Injected-current scale (mV per nA) for a cell's model. Pillars' LIF and
/// AdEx are voltage-normalised (÷gL), so their `current_scale` is 1000/gL;
/// every other model takes current in its native units (scale 1.0).
fn current_scale(cell: &Cell) -> f32 {
    match &cell.spec {
        CellSpec::AdEx { current_scale, .. } => *current_scale,
        CellSpec::Iaf { current_scale, .. } => *current_scale,
        _ => 1.0,
    }
}

/// Add a population of the right model type and configure it from `cell`.
///
/// Returns the `PopulationId` plus the [`GpuPopMeta`] (which `#[dual_neuron]`
/// CUDA source/kernel + column counts) so GPU backends can compile this
/// population without re-deriving the erased model type.
fn add_population(
    net: &mut Network,
    cell: &Cell,
    size: usize,
    dt_ms: f32,
    celsius: f32,
) -> (PopulationId, GpuPopMeta) {
    match &cell.spec {
        CellSpec::Izh {
            v0,
            thresh,
            a,
            b,
            c,
            d,
        } => {
            let pid = net.add_population::<IzhikevichModel>(size);
            net.configure::<IzhikevichModel>(pid, |p| {
                p.fill_param(IzhikevichModelParamIdx::dt as usize, dt_ms);
                p.fill_param(IzhikevichModelParamIdx::a as usize, *a);
                p.fill_param(IzhikevichModelParamIdx::b as usize, *b);
                p.fill_param(IzhikevichModelParamIdx::c as usize, *c);
                p.fill_param(IzhikevichModelParamIdx::d as usize, *d);
                p.fill_param(IzhikevichModelParamIdx::v_thresh as usize, *thresh);
                p.fill_state(0, *v0);
                p.fill_state(1, b * v0);
                p.fill_state(2, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: IZHIKEVICH_CUDA_SOURCE,
                    kernel_name: "izhikevich_kernel",
                    ns: IzhikevichModel::N_STATE,
                    np: IzhikevichModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Izh2007 {
            v0,
            cap,
            k,
            vr,
            vt,
            vpeak,
            a,
            b,
            c,
            d,
        } => {
            let pid = net.add_population::<Izhikevich2007Model>(size);
            net.configure::<Izhikevich2007Model>(pid, |p| {
                p.fill_param(Izhikevich2007ModelParamIdx::dt as usize, dt_ms);
                p.fill_param(Izhikevich2007ModelParamIdx::cap as usize, *cap);
                p.fill_param(Izhikevich2007ModelParamIdx::k as usize, *k);
                p.fill_param(Izhikevich2007ModelParamIdx::vr as usize, *vr);
                p.fill_param(Izhikevich2007ModelParamIdx::vt as usize, *vt);
                p.fill_param(Izhikevich2007ModelParamIdx::vpeak as usize, *vpeak);
                p.fill_param(Izhikevich2007ModelParamIdx::a as usize, *a);
                p.fill_param(Izhikevich2007ModelParamIdx::b as usize, *b);
                p.fill_param(Izhikevich2007ModelParamIdx::c as usize, *c);
                p.fill_param(Izhikevich2007ModelParamIdx::d as usize, *d);
                p.fill_state(0, *v0);
                // Steady-state recovery: u = b·(v0 − vr).
                p.fill_state(1, *b * (*v0 - *vr));
            });
            (
                pid,
                GpuPopMeta {
                    source: IZHIKEVICH_2007_CUDA_SOURCE,
                    kernel_name: "izhikevich2007_kernel",
                    ns: Izhikevich2007Model::N_STATE,
                    np: Izhikevich2007Model::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Iaf {
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_ref,
            current_scale: _,
        } => {
            let pid = net.add_population::<LIFModel>(size);
            net.configure::<LIFModel>(pid, |p| {
                // LIF layout (see LIFBuilder): dt, inv_tau_m, v_rest, v_thresh,
                // v_reset, tau_ref; state v, refr.
                p.fill_param(0, dt_ms);
                p.fill_param(1, 1.0 / tau_m);
                p.fill_param(2, *v_rest);
                p.fill_param(3, *v_thresh);
                p.fill_param(4, *v_reset);
                p.fill_param(5, *tau_ref);
                p.fill_state(0, *v_rest);
                p.fill_state(1, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: LIF_CUDA_SOURCE,
                    kernel_name: "lif_kernel",
                    ns: LIFModel::N_STATE,
                    np: LIFModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::Fhn {
            a,
            b,
            inv_tau_w,
            v_thresh,
            bias: _,
            v0,
            w0,
        } => {
            let pid = net.add_population::<FHNModel>(size);
            net.configure::<FHNModel>(pid, |p| {
                // FHN layout: dt, inv_tau_w, a, b, v_thresh; state v, w, was_below.
                p.fill_param(0, dt_ms);
                p.fill_param(1, *inv_tau_w);
                p.fill_param(2, *a);
                p.fill_param(3, *b);
                p.fill_param(4, *v_thresh);
                p.fill_state(0, *v0);
                p.fill_state(1, *w0);
                p.fill_state(2, 1.0); // was_below = true (edge detection ready)
            });
            (
                pid,
                GpuPopMeta {
                    source: FHN_CUDA_SOURCE,
                    kernel_name: "fhn_kernel",
                    ns: FHNModel::N_STATE,
                    np: FHNModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::AdEx {
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_ref,
            v_peak,
            current_scale: _,
        } => {
            let pid = net.add_population::<AdExModel>(size);
            net.configure::<AdExModel>(pid, |p| {
                // AdEx layout (see AdExBuilder): dt, inv_tau_m, inv_tau_w,
                // v_rest, v_thresh, v_reset, delta_t, inv_delta_t, a, b,
                // tau_ref, v_peak; state v, w, refr.
                p.fill_param(0, dt_ms);
                p.fill_param(1, 1.0 / tau_m);
                p.fill_param(2, 1.0 / tau_w);
                p.fill_param(3, *v_rest);
                p.fill_param(4, *v_thresh);
                p.fill_param(5, *v_reset);
                p.fill_param(6, *delta_t);
                p.fill_param(7, 1.0 / delta_t);
                p.fill_param(8, *a);
                p.fill_param(9, *b);
                p.fill_param(10, *tau_ref);
                p.fill_param(11, *v_peak);
                p.fill_state(0, *v_rest);
                p.fill_state(1, 0.0);
                p.fill_state(2, 0.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: ADEX_CUDA_SOURCE,
                    kernel_name: "adex_kernel",
                    ns: AdExModel::N_STATE,
                    np: AdExModel::N_PARAMS,
                    n: size,
                },
            )
        }
        // Классический Hodgkin–Huxley (hhCell): стандартные гейты m³h/n⁴.
        // q10 = 1 (без температурной шкалы), v_thresh = 0 (детекция спайка),
        // CaT выключен — ровно классический кальмаровый аксон.
        CellSpec::Hh {
            v0,
            g_na,
            e_na,
            g_k,
            e_k,
            g_leak,
            e_leak,
            inv_cm,
        } => {
            let pid = net.add_population::<HHModel>(size);
            net.configure::<HHModel>(pid, |p| {
                p.fill_param(HHModelParamIdx::dt as usize, dt_ms);
                p.fill_param(HHModelParamIdx::g_Na as usize, *g_na);
                p.fill_param(HHModelParamIdx::E_Na as usize, *e_na);
                p.fill_param(HHModelParamIdx::g_K as usize, *g_k);
                p.fill_param(HHModelParamIdx::E_K as usize, *e_k);
                p.fill_param(HHModelParamIdx::g_Leak as usize, *g_leak);
                p.fill_param(HHModelParamIdx::E_Leak as usize, *e_leak);
                p.fill_param(HHModelParamIdx::q10 as usize, 1.0);
                p.fill_param(HHModelParamIdx::v_thresh as usize, 0.0);
                p.fill_param(HHModelParamIdx::inv_cm as usize, *inv_cm);
                p.fill_param(HHModelParamIdx::g_CaT as usize, 0.0);
                p.fill_param(HHModelParamIdx::E_CaT as usize, 120.0);
                p.fill_state(HHModelStateIdx::v as usize, *v0);
                p.fill_state(HHModelStateIdx::m as usize, 0.05);
                p.fill_state(HHModelStateIdx::h as usize, 0.6);
                p.fill_state(HHModelStateIdx::n as usize, 0.32);
                p.fill_state(HHModelStateIdx::c as usize, 0.19);
                p.fill_state(HHModelStateIdx::was_below as usize, 1.0);
            });
            (
                pid,
                GpuPopMeta {
                    source: HH_CUDA_SOURCE,
                    kernel_name: "hh_kernel",
                    ns: HHModel::N_STATE,
                    np: HHModel::N_PARAMS,
                    n: size,
                },
            )
        }
        CellSpec::ConductanceBased {
            v0,
            c_m,
            g_leak,
            e_leak,
            channels,
            markov_channels,
        } => {
            if markov_channels.is_empty() {
                // Произвольный HH-набор — SIMD generic-движок (не HHModel).
                let meta: Vec<ChannelMeta> = channels.iter().map(channel_to_meta).collect();
                let n_gates: usize = meta.iter().map(|m| m.n_gates).sum();
                let mut pop = ConductancePopulation::new(meta, size, dt_ms, celsius);
                pop.set_cm(*c_m);
                pop.set_leak(*g_leak, *e_leak);
                for (i, ch) in channels.iter().enumerate() {
                    pop.set_g_max(i, ch.g_max);
                }
                pop.set_resting_state(*v0);
                let pid = net.add_boxed_population(Box::new(pop));
                // Для data-driven каналов нет `#[dual_neuron]` CUDA-источника —
                // CPU-only популяция; пустой `source` честно помечает это.
                (
                    pid,
                    GpuPopMeta {
                        source: "",
                        kernel_name: "",
                        ns: 1 + n_gates,
                        np: 0,
                        n: size,
                    },
                )
            } else {
                // Есть кинетические (KS) каналы — скалярный master-equation
                // движок. HH-каналы опускаются до 2-состоянийного Маркова,
                // поэтому один `MarkovPopulation` гоняет всю смесь.
                let mut mc: Vec<MarkovChannel> = channels
                    .iter()
                    .map(|ch| MarkovChannel {
                        gates: ch
                            .gates
                            .iter()
                            .map(|(k, pow)| gate_kinetics_to_markov(k.clone(), *pow))
                            .collect(),
                        e_rev: ch.e_rev,
                        g_max: ch.g_max,
                    })
                    .collect();
                mc.extend(markov_channels.iter().map(|ch| MarkovChannel {
                    gates: ch.gates.clone(),
                    e_rev: ch.e_rev,
                    g_max: ch.g_max,
                }));
                let n_states: usize = mc.iter().flat_map(|c| &c.gates).map(|g| g.n_states()).sum();
                let mut pop = MarkovPopulation::new(mc, size, dt_ms, celsius);
                pop.set_cm(*c_m);
                pop.set_leak(*g_leak, *e_leak);
                // Установить g_max по порядку: сначала HH-каналы, затем KS.
                for (i, ch) in channels.iter().enumerate() {
                    pop.set_g_max(i, ch.g_max);
                }
                for (i, ch) in markov_channels.iter().enumerate() {
                    pop.set_g_max(channels.len() + i, ch.g_max);
                }
                pop.set_resting_state(*v0);
                let pid = net.add_boxed_population(Box::new(pop));
                (
                    pid,
                    GpuPopMeta {
                        source: "",
                        kernel_name: "",
                        ns: 1 + n_states,
                        np: 0,
                        n: size,
                    },
                )
            }
        }
    }
}

/// A resolved active cable channel (before area scaling): HH gates (exponential
/// Euler) or Markov gates (master equation) + reversal potential.
enum CableChannel {
    /// `(gates, per-gate q10, e_rev)`.
    Hh(Vec<(GateKinetics, u32)>, Vec<f32>, f32),
    /// `(gates, per-gate q10, e_rev)`.
    Markov(Vec<MarkovGateSpec>, Vec<f32>, f32),
}

/// Resolve a non-passive `channelDensity` into a cable channel: `ionChannelHH` →
/// HH gates (or Markov if it carries `gateFractional`), `ionChannelKS` → Markov
/// gates. Reuses the parser's resolution logic so the cable and the
/// point-neuron engines agree bit-for-bit on gate lowering.
fn resolve_cable_channel(
    cd: &IrChannelDensity,
    hh: &HashMap<&str, &IrChannel>,
    ks: &HashMap<&str, &IrKsChannel>,
    celsius: f32,
) -> Result<CableChannel, NeuromlError> {
    resolve_cable_channel_ion(&cd.id, &cd.ion_channel, cd.erev, hh, ks, celsius)
}

/// The same resolution for a non-uniform density (the density varies per
/// compartment, so the channel is resolved without a fixed conductance).
fn resolve_cable_channel_ion(
    id: &str,
    ion_channel: &str,
    erev: f32,
    hh: &HashMap<&str, &IrChannel>,
    ks: &HashMap<&str, &IrKsChannel>,
    celsius: f32,
) -> Result<CableChannel, NeuromlError> {
    if let Some(ch) = hh.get(ion_channel) {
        if ch.fractional_gates.is_empty() {
            let r = resolve_channel(ch, erev, 0.0, celsius)?;
            Ok(CableChannel::Hh(r.gates, r.q10s, r.e_rev))
        } else {
            let r = resolve_markov_channel(ch, erev, 0.0, celsius)?;
            Ok(CableChannel::Markov(r.gates, r.q10s, r.e_rev))
        }
    } else if let Some(ch) = ks.get(ion_channel) {
        Ok(CableChannel::Markov(
            ch.gates.clone(),
            // ionChannelKS gates carry no q10Settings.
            vec![1.0; ch.gates.len()],
            erev,
        ))
    } else {
        Err(NeuromlError::parse(format!(
            "channelDensity `{id}` references unknown ionChannel `{ion_channel}`"
        )))
    }
}

/// Resolve a channel reference (`ionChannel` or `ionChannelKS`) into its
/// Markov gates + single-channel conductance (µS). `ionChannelKS` is the
/// canonical stochastic channel (its `gateKS` is the master equation whose
/// open probability is binomially sampled); `ionChannelHH` is also accepted
/// (lowered to Markov gates) for permissiveness, but a discrete population is
/// physically meaningful only for a kinetic scheme.
fn resolve_channel_gates(
    id: &str,
    ion_channel: &str,
    hh: &HashMap<&str, &IrChannel>,
    ks: &HashMap<&str, &IrKsChannel>,
    celsius: f32,
) -> Result<(Vec<MarkovGateSpec>, Vec<f32>, f32), NeuromlError> {
    if let Some(ch) = ks.get(ion_channel) {
        let single = ch.conductance.ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel population `{id}` references ionChannelKS `{ion_channel}` without a `conductance`"
            ))
        })?;
        return Ok((
            ch.gates.clone(),
            // ionChannelKS gates carry no q10Settings.
            vec![1.0; ch.gates.len()],
            single,
        ));
    }
    if let Some(ch) = hh.get(ion_channel) {
        let single = ch.conductance.ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel population `{id}` references ionChannel `{ion_channel}` without a `conductance`"
            ))
        })?;
        let r = resolve_markov_channel(ch, 0.0, 0.0, celsius)?;
        return Ok((r.gates, r.q10s, single));
    }
    Err(NeuromlError::parse(format!(
        "channel population `{id}` references unknown ionChannel `{ion_channel}`"
    )))
}

/// Resolve a `channelPopulation` into its Markov gates + single-channel
/// conductance (µS) — see [`resolve_channel_gates`].
fn resolve_channel_population(
    cp: &IrChannelPopulation,
    hh: &HashMap<&str, &IrChannel>,
    ks: &HashMap<&str, &IrKsChannel>,
    celsius: f32,
) -> Result<(Vec<MarkovGateSpec>, Vec<f32>, f32), NeuromlError> {
    resolve_channel_gates(&cp.id, &cp.ion_channel, hh, ks, celsius)
}

/// FNV-1a 64-bit — a stable hash of a channel population id, used to seed the
/// stochastic channel RNG so a given model reproduces the same noise stream.
fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Add a morphologically-detailed cell population: discretize the morphology
/// (d_λ with the cell's passive biophysical constants), build one cable neuron
/// per cell, attach active channels by segmentGroup (density × membrane area),
/// and box it as a [`CablePopulation`].
fn add_morpho_population(
    net: &mut Network,
    cell: &MorphoCell,
    morphologies: &HashMap<&str, &MorphoIR>,
    biophysics: &HashMap<&str, &BiophysicalProperties>,
    hh_channels: &HashMap<&str, &IrChannel>,
    ks_channels: &HashMap<&str, &IrKsChannel>,
    concentration_models: &HashMap<&str, &IrConcentrationModel>,
    warnings: &mut Vec<String>,
    size: usize,
    dt_ms: f32,
    celsius: f32,
) -> Result<(PopulationId, GpuPopMeta, CableInfo), NeuromlError> {
    let morpho = morphologies.get(cell.morphology.as_str()).ok_or_else(|| {
        NeuromlError::parse(format!(
            "cell `{}` references unknown morphology `{}`",
            cell.id, cell.morphology
        ))
    })?;
    let bp: Option<&BiophysicalProperties> = match cell.biophysics.as_deref() {
        Some(id) => Some(biophysics.get(id).ok_or_else(|| {
            NeuromlError::parse(format!(
                "cell `{}` references unknown biophysicalProperties `{id}`",
                cell.id
            ))
        })?),
        None => None,
    };
    let (params, e_leak) = match bp {
        Some(bp) => (CableParams::from_biophysics(bp), passive_leak_reversal(bp)),
        None => (CableParams::default(), -65.0),
    };
    let mut tree = discretize(morpho, &params)?;

    // Per-group passive leak: the model may declare different leak
    // `channelDensity`s per segmentGroup (Poirazi: soma/axon 5e-3, basal
    // 5.2453e-3, apical 6.938e-3 mS/cm²). Collect the per-compartment leak
    // conductance (density × area) and zero the tree's uniform `g_leak_us`
    // (from `CableParams::rm_ohm_cm2` — used for the d_λ split only) so the
    // solver doesn't double-count; the per-group leak is attached per
    // compartment below.
    let mut leak_by_comp: HashMap<usize, (f32, f32)> = HashMap::new();
    if let Some(bp) = bp {
        for cd in &bp.membrane.channel_densities {
            if is_passive_leak(cd) {
                let segments = morpho.resolve_segments(&cd.segment_group)?;
                for comp in tree.group_compartments(&segments) {
                    let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                    let g_us = cd.cond_density * area_cm2 * 1e6; // S/cm² · cm² · 1e6 = µS
                    leak_by_comp.insert(comp, (g_us, cd.erev));
                }
            }
        }
    }
    for comp in leak_by_comp.keys() {
        tree.compartments[*comp].g_leak_us = 0.0;
    }
    let mut pop = CablePopulation::new(&tree, size, e_leak, dt_ms, celsius);
    for (comp, (g_us, e_rev)) in &leak_by_comp {
        pop.set_leak(*comp, *g_us, *e_rev);
    }

    // Attach active channels by segmentGroup (density × membrane area → µS).
    if let Some(bp) = bp {
        let mut hh_by_comp: HashMap<usize, Vec<ActiveChannel>> = HashMap::new();
        let mut mk_by_comp: HashMap<usize, Vec<MarkovActiveChannel>> = HashMap::new();
        for cd in &bp.membrane.channel_densities {
            if is_passive_leak(cd) {
                continue; // pas/leak уже учтён в CableParams::from_biophysics
            }
            let segments = morpho.resolve_segments(&cd.segment_group)?;
            let channel = resolve_cable_channel(cd, hh_channels, ks_channels, celsius)?;
            for comp in tree.group_compartments(&segments) {
                let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                let g_max_us = cd.cond_density * area_cm2 * 1e6; // S/cm² · cm² · 1e6 = µS
                match &channel {
                    CableChannel::Hh(gates, q10s, e_rev) => {
                        hh_by_comp.entry(comp).or_default().push(ActiveChannel::new(
                            gates.clone(),
                            q10s.clone(),
                            *e_rev,
                            g_max_us,
                            e_leak,
                        ))
                    }
                    CableChannel::Markov(gates, q10s, e_rev) => mk_by_comp
                        .entry(comp)
                        .or_default()
                        .push(MarkovActiveChannel::new(
                            gates.clone(),
                            q10s.clone(),
                            *e_rev,
                            g_max_us,
                            e_leak,
                        )),
                }
            }
        }
        // Неравномерные плотности: density линейно интерполируется вдоль пути
        // группы (proximal у корня → distal на конце), затем × площадь.
        for cd in &bp.membrane.non_uniform_densities {
            let ion = cd.ion.to_ascii_lowercase();
            let ch_name = cd.ion_channel.to_ascii_lowercase();
            if matches!(ion.as_str(), "non_specific" | "pas" | "leak")
                || matches!(ch_name.as_str(), "pas" | "leak")
            {
                continue;
            }
            let channel = resolve_cable_channel_ion(
                &cd.id,
                &cd.ion_channel,
                cd.erev,
                hh_channels,
                ks_channels,
                celsius,
            )?;
            for vp in &cd.variable_params {
                // Linear vs path-length-expression form.
                let (segment_group, linear): (&str, Option<(&f32, &f32)>) = match vp {
                    NonUniformParam::Linear {
                        segment_group,
                        proximal,
                        distal,
                    } => (segment_group, Some((proximal, distal))),
                    NonUniformParam::Inhomogeneous {
                        segment_group,
                        value: _,
                    } => (segment_group, None),
                };
                let segments = morpho.resolve_segments(segment_group)?;
                let comps = tree.group_compartments(&segments);
                if comps.is_empty() {
                    continue;
                }
                let (mut min_path, mut max_path) = (f32::INFINITY, f32::NEG_INFINITY);
                for &c in &comps {
                    let p = tree.compartments[c].path_um;
                    min_path = min_path.min(p);
                    max_path = max_path.max(p);
                }
                let span = max_path - min_path;
                for &comp in &comps {
                    let path_um = tree.compartments[comp].path_um;
                    let density = match (linear, vp) {
                        (Some((proximal, distal)), _) => {
                            let u = if span > 0.0 {
                                ((path_um - min_path) / span).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            proximal + (distal - proximal) * u
                        }
                        (None, NonUniformParam::Inhomogeneous { value, .. }) => {
                            eval_inhomogeneous_expr(&value.expr, path_um)
                        }
                        _ => unreachable!(),
                    };
                    let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                    let g_max_us = density * area_cm2 * 1e6; // S/cm² · cm² · 1e6 = µS
                    match &channel {
                        CableChannel::Hh(gates, q10s, e_rev) => {
                            hh_by_comp.entry(comp).or_default().push(ActiveChannel::new(
                                gates.clone(),
                                q10s.clone(),
                                *e_rev,
                                g_max_us,
                                e_leak,
                            ))
                        }
                        CableChannel::Markov(gates, q10s, e_rev) => mk_by_comp
                            .entry(comp)
                            .or_default()
                            .push(MarkovActiveChannel::new(
                                gates.clone(),
                                q10s.clone(),
                                *e_rev,
                                g_max_us,
                                e_leak,
                            )),
                    }
                }
            }
        }
        // Discrete channel populations (`channelPopulation`) are **point**
        // currents on the single segment named by `segment` (default `"0"` =
        // soma). Each is attached to that segment's compartment (its
        // proximal-most compartment when the d_λ rule split a long segment
        // into several), matching jNeuroML/NEURON's placement of a point
        // population on the segment.
        let mut stoch_by_comp: HashMap<usize, Vec<StochasticMarkovActiveChannel>> = HashMap::new();
        for cp in &bp.membrane.channel_populations {
            let (gates, q10s, single_cond_us) =
                resolve_channel_population(cp, hh_channels, ks_channels, celsius)?;
            let g_max_us = single_cond_us * cp.number as f32;
            let comps = tree.group_compartments(&morpho.resolve_segment(&cp.segment)?);
            let comp = *comps.first().ok_or_else(|| {
                NeuromlError::parse(format!(
                    "cell `{}`: channelPopulation `{}` segment `{}` produced no compartments",
                    cell.id, cp.id, cp.segment
                ))
            })?;
            stoch_by_comp
                .entry(comp)
                .or_default()
                .push(StochasticMarkovActiveChannel::new(
                    gates,
                    q10s,
                    cp.erev,
                    g_max_us,
                    cp.number,
                    e_leak,
                    fnv1a(&cp.id),
                ));
        }

        // Ca²⁺ species + Nernst channels. Each species' concentration model
        // becomes a per-compartment pool on its segment group (the flux scale
        // depends on the compartment's membrane area); each Nernst/GHK channel
        // reads that compartment's [Ca²⁺] and feeds its current back into the
        // pool. `decayingPoolConcentrationModel` and custom LEMS pools with the
        // canonical ODE (`PoiraziCaConcentrationModel`: flux÷18, decay×7) are
        // lowered; fixedFactor is not.
        let ca_species: Vec<&crate::api::neuroml::parse::IrSpecies> = bp
            .species
            .iter()
            .filter(|s| s.ion.eq_ignore_ascii_case("ca"))
            .collect();
        for sp in &ca_species {
            let Some(pool) = concentration_models.get(sp.concentration_model.as_str()) else {
                warnings.push(format!(
                    "species `{}` references unknown concentrationModel `{}`",
                    sp.id, sp.concentration_model
                ));
                continue;
            };
            let (resting, tau, shell_um, flux_div, decay_mul) = match pool {
                IrConcentrationModel::DecayingPool {
                    resting_conc_mm,
                    decay_constant_ms,
                    shell_thickness_um,
                    ..
                } => (
                    *resting_conc_mm,
                    *decay_constant_ms,
                    *shell_thickness_um,
                    1.0,
                    1.0,
                ),
                IrConcentrationModel::CustomDecayingPool {
                    resting_conc_mm,
                    decay_constant_ms,
                    shell_thickness_um,
                    flux_divisor,
                    decay_multiplier,
                    ..
                } => (
                    *resting_conc_mm,
                    *decay_constant_ms,
                    *shell_thickness_um,
                    *flux_divisor,
                    *decay_multiplier,
                ),
                IrConcentrationModel::FixedFactor { .. } => {
                    warnings.push(format!(
                        "species `{}` uses a fixedFactor/Traub concentration model — not yet lowered",
                        sp.id
                    ));
                    continue;
                }
            };
            let segments = morpho.resolve_segments(&sp.segment_group)?;
            for comp in tree.group_compartments(&segments) {
                let area_um2 = tree.compartments[comp].area_um2;
                pop.set_ca_pool(
                    comp,
                    CaPool {
                        resting_mm: resting,
                        // Custom pools decay `decay_multiplier`× faster.
                        tau_ms: tau / decay_mul,
                        // Custom pools (Poirazi capool) divide the flux by
                        // `flux_div` and use the **thin-shell** volume
                        // (SA·thickness — the model's own formula), not the
                        // exact spherical shell of the standard LEMS pool.
                        flux_scale: thin_shell_pool_flux_scale(area_um2, shell_um) / flux_div,
                    },
                    sp.initial_conc_mm,
                );
            }
        }
        // External [Ca²⁺] for Nernst/GHK channels (mM) — the first Ca species.
        let ca_ext_mm = ca_species
            .first()
            .map(|s| s.initial_ext_conc_mm)
            .unwrap_or(2.0);
        let mut nernst_by_comp: HashMap<usize, Vec<NernstActiveChannel>> = HashMap::new();
        for cd in &bp.membrane.nernst_densities {
            if ca_species.is_empty() {
                warnings.push(format!(
                    "channelDensityNernst `{}` has no Ca species — skipped",
                    cd.id
                ));
                continue;
            }
            let channel = resolve_cable_channel_ion(
                &cd.id,
                &cd.ion_channel,
                0.0,
                hh_channels,
                ks_channels,
                celsius,
            )?;
            let (gates, q10s) = match channel {
                CableChannel::Hh(gates, q10s, _) => (gates, q10s),
                CableChannel::Markov(..) => {
                    warnings.push(format!(
                        "channelDensityNernst `{}` uses a kinetic-scheme channel — not yet lowered",
                        cd.id
                    ));
                    continue;
                }
            };
            let segments = morpho.resolve_segments(&cd.segment_group)?;
            for comp in tree.group_compartments(&segments) {
                let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                let g_max_us = cd.cond_density * area_cm2 * 1e6;
                nernst_by_comp
                    .entry(comp)
                    .or_default()
                    .push(NernstActiveChannel::new(
                        gates.clone(),
                        q10s.clone(),
                        ca_ext_mm,
                        g_max_us,
                        e_leak,
                    ));
            }
        }

        // GHK (Ca²⁺) channel densities (`channelDensityGHK2` /
        // `channelDensityGHK2NonUniform`) — current is the Goldman–Hodgkin–Katz
        // flux equation, driven by the compartment's dynamic [Ca²⁺] (no
        // reversal). The non-uniform form evaluates the density per
        // compartment from the variable parameter (linear along the group's
        // path or a `PathLength*` expression) — same convention as the ohmic
        // non-uniform densities.
        let mut ghk_by_comp: HashMap<usize, Vec<GhkActiveChannel>> = HashMap::new();
        for cd in &bp.membrane.ghk_densities {
            if ca_species.is_empty() {
                warnings.push(format!(
                    "channelDensityGHK2 `{}` has no Ca species — skipped",
                    cd.id
                ));
                continue;
            }
            let channel = resolve_cable_channel_ion(
                &cd.id,
                &cd.ion_channel,
                0.0,
                hh_channels,
                ks_channels,
                celsius,
            )?;
            let (gates, q10s) = match channel {
                CableChannel::Hh(gates, q10s, _) => (gates, q10s),
                CableChannel::Markov(..) => {
                    warnings.push(format!(
                        "channelDensityGHK2 `{}` uses a kinetic-scheme channel — not yet lowered",
                        cd.id
                    ));
                    continue;
                }
            };
            if cd.variable_params.is_empty() {
                let segments = morpho.resolve_segments(&cd.segment_group)?;
                for comp in tree.group_compartments(&segments) {
                    let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                    let g_max_us = cd.cond_density * area_cm2 * 1e6;
                    ghk_by_comp
                        .entry(comp)
                        .or_default()
                        .push(GhkActiveChannel::new(
                            gates.clone(),
                            q10s.clone(),
                            ca_ext_mm,
                            g_max_us,
                            e_leak,
                        ));
                }
            } else {
                for vp in &cd.variable_params {
                    let (segment_group, linear): (&str, Option<(&f32, &f32)>) = match vp {
                        NonUniformParam::Linear {
                            segment_group,
                            proximal,
                            distal,
                        } => (segment_group, Some((proximal, distal))),
                        NonUniformParam::Inhomogeneous {
                            segment_group,
                            value: _,
                        } => (segment_group, None),
                    };
                    let segments = morpho.resolve_segments(segment_group)?;
                    let comps = tree.group_compartments(&segments);
                    if comps.is_empty() {
                        continue;
                    }
                    let (mut min_path, mut max_path) = (f32::INFINITY, f32::NEG_INFINITY);
                    for &c in &comps {
                        let p = tree.compartments[c].path_um;
                        min_path = min_path.min(p);
                        max_path = max_path.max(p);
                    }
                    let span = max_path - min_path;
                    for &comp in &comps {
                        let path_um = tree.compartments[comp].path_um;
                        let density = match (linear, vp) {
                            (Some((proximal, distal)), _) => {
                                let u = if span > 0.0 {
                                    ((path_um - min_path) / span).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                proximal + (distal - proximal) * u
                            }
                            (None, NonUniformParam::Inhomogeneous { value, .. }) => {
                                eval_inhomogeneous_expr(&value.expr, path_um)
                            }
                            _ => unreachable!(),
                        };
                        let area_cm2 = tree.compartments[comp].area_um2 * 1e-8;
                        let g_max_us = density * area_cm2 * 1e6;
                        ghk_by_comp
                            .entry(comp)
                            .or_default()
                            .push(GhkActiveChannel::new(
                                gates.clone(),
                                q10s.clone(),
                                ca_ext_mm,
                                g_max_us,
                                e_leak,
                            ));
                    }
                }
            }
        }

        // Discrete Nernst (Ca²⁺) channel populations (`channelPopulationNernst`)
        // — the same point-placement-on-soma convention as `channelPopulation`,
        // but with a dynamic Nernst reversal fed back into the Ca²⁺ pool.
        let mut stoch_nernst_by_comp: HashMap<usize, Vec<StochasticNernstActiveChannel>> =
            HashMap::new();
        for cpn in &bp.membrane.nernst_populations {
            if ca_species.is_empty() {
                warnings.push(format!(
                    "channelPopulationNernst `{}` has no Ca species — skipped",
                    cpn.id
                ));
                continue;
            };
            let (gates, q10s, single_cond_us) = resolve_channel_gates(
                &cpn.id,
                &cpn.ion_channel,
                hh_channels,
                ks_channels,
                celsius,
            )?;
            let g_max_us = single_cond_us * cpn.number as f32;
            let comps = tree.group_compartments(&morpho.resolve_segment(&cpn.segment)?);
            let comp = *comps.first().ok_or_else(|| {
                NeuromlError::parse(format!(
                    "cell `{}`: channelPopulationNernst `{}` segment `{}` produced no compartments",
                    cell.id, cpn.id, cpn.segment
                ))
            })?;
            stoch_nernst_by_comp
                .entry(comp)
                .or_default()
                .push(StochasticNernstActiveChannel::new(
                    gates,
                    q10s,
                    ca_ext_mm,
                    g_max_us,
                    cpn.number,
                    e_leak,
                    fnv1a(&cpn.id),
                ));
        }

        for (comp, chans) in hh_by_comp {
            pop.set_channels(comp, chans);
        }
        for (comp, chans) in mk_by_comp {
            pop.set_markov_channels(comp, chans);
        }
        for (comp, chans) in nernst_by_comp {
            pop.set_nernst_channels(comp, chans);
        }
        for (comp, chans) in ghk_by_comp {
            pop.set_ghk_channels(comp, chans);
        }
        for (comp, chans) in stoch_by_comp {
            pop.set_stochastic_channels(comp, chans);
        }
        for (comp, chans) in stoch_nernst_by_comp {
            pop.set_stochastic_nernst_channels(comp, chans);
        }
    }

    let pid = net.add_boxed_population(Box::new(pop));
    // Public cable geometry: compartments (positions/path for drawing) +
    // segment names for labels. Index 0 of the mirror = this compartment, so
    // GUI voltages map 1:1 onto `compartments`.
    let info = CableInfo {
        root: tree.root,
        compartments: tree
            .compartments
            .iter()
            .map(|c| CableCompartment {
                parent: c.parent,
                segment_id: c.segment_id,
                length_um: c.length_um,
                diameter_um: c.diameter_um,
                x: c.x,
                y: c.y,
                z: c.z,
                path_um: c.path_um,
            })
            .collect(),
        segment_names: morpho
            .segments
            .iter()
            .filter_map(|s| s.name.clone().map(|n| (s.id, n)))
            .collect(),
    };
    Ok((
        pid,
        GpuPopMeta {
            source: "",
            kernel_name: "",
            ns: 1, // soma V — единственная экспонируемая колонка состояния
            np: 0,
            n: size,
        },
        info,
    ))
}

/// Сборка [`ChannelMeta`] из резолвленного канала. `gate_type` заполняется
/// placeholder'ом (`NaM`) — non-fused движок `hh_poly_step` использует только
/// `kinetics`, поэтому fused fast-path (завязанный на 4 канонических гейта)
/// для произвольных каналов не задействуется.
fn channel_to_meta(ch: &ResolvedChannel) -> ChannelMeta {
    ChannelMeta {
        n_gates: ch.gates.len(),
        e_rev: ch.e_rev,
        powers: ch.gates.iter().map(|g| g.1).collect(),
        gate_type: vec![GateType::NaM; ch.gates.len()],
        kinetics: ch.gates.iter().map(|g| g.0.clone()).collect(),
    }
}

/// Evaluate a NeuroML `inhomogeneousValue` density expression at path length
/// `p` (µm from the soma) and return the density in **S/cm²**.
///
/// The expression is a LEMS expression in the SI base unit of conductance
/// density, **S/m²** (`1 S/cm² = 1e4 S/m²`), so the raw result is divided by
/// `1e4`. Real Poirazi 2003 exports write e.g.
/// `1e4 * H(p_apical_trunk-300) * 7e-3` for a `7e-3 S/cm²` density. Supports
/// numbers, the path-length symbol (any bare identifier, e.g. `p_apical_trunk`),
/// `H(...)` (Heaviside step: 1 when the argument is ≥ 0, else 0 — jNeuroML
/// convention), `exp(...)` (natural exponential, for sigmoid densities), and
/// `+ - * /` with parentheses.
pub(crate) fn eval_inhomogeneous_expr(expr: &str, p: f32) -> f32 {
    let toks = inho_tokens(expr);
    let mut pos = 0usize;
    let val = inho_expr(&toks, &mut pos, p);
    if pos != toks.len() {
        return 0.0;
    }
    val * 1e-4 // S/m² → S/cm²
}

#[derive(Debug, Clone, PartialEq)]
enum InhoTok {
    Num(f32),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Ident(String),
}

fn inho_tokens(s: &str) -> Vec<InhoTok> {
    let b = s.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < b.len() && (b[i] as char).is_ascii_digit()
                || (i < b.len() && b[i] as char == '.')
            {
                if (b[i] as char).is_ascii_digit() || b[i] as char == '.' {
                    i += 1;
                } else {
                    break;
                }
            }
            if i < b.len() && (b[i] as char == 'e' || b[i] as char == 'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] as char == '+' || b[j] as char == '-') {
                    j += 1;
                }
                if j < b.len() && (b[j] as char).is_ascii_digit() {
                    i = j + 1;
                    while i < b.len() && (b[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            toks.push(InhoTok::Num(s[start..i].parse().unwrap_or(0.0)));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_') {
                i += 1;
            }
            toks.push(InhoTok::Ident(s[start..i].to_string()));
        } else {
            match c {
                '+' => toks.push(InhoTok::Plus),
                '-' => toks.push(InhoTok::Minus),
                '*' => toks.push(InhoTok::Star),
                '/' => toks.push(InhoTok::Slash),
                '(' => toks.push(InhoTok::LParen),
                ')' => toks.push(InhoTok::RParen),
                ',' => toks.push(InhoTok::Comma),
                _ => {}
            }
            i += 1;
        }
    }
    toks
}

fn inho_expr(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    let mut lhs = inho_term(toks, pos, p);
    loop {
        match toks.get(*pos) {
            Some(InhoTok::Plus) => {
                *pos += 1;
                lhs += inho_term(toks, pos, p);
            }
            Some(InhoTok::Minus) => {
                *pos += 1;
                lhs -= inho_term(toks, pos, p);
            }
            _ => return lhs,
        }
    }
}

fn inho_term(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    let mut lhs = inho_factor(toks, pos, p);
    loop {
        match toks.get(*pos) {
            Some(InhoTok::Star) => {
                *pos += 1;
                lhs *= inho_factor(toks, pos, p);
            }
            Some(InhoTok::Slash) => {
                *pos += 1;
                let d = inho_factor(toks, pos, p);
                lhs /= if d != 0.0 { d } else { 1.0 };
            }
            _ => return lhs,
        }
    }
}

fn inho_factor(toks: &[InhoTok], pos: &mut usize, p: f32) -> f32 {
    match toks.get(*pos) {
        Some(InhoTok::Num(n)) => {
            *pos += 1;
            *n
        }
        Some(InhoTok::Minus) => {
            *pos += 1;
            -inho_factor(toks, pos, p)
        }
        Some(InhoTok::LParen) => {
            *pos += 1;
            let v = inho_expr(toks, pos, p);
            if toks.get(*pos) == Some(&InhoTok::RParen) {
                *pos += 1;
            }
            v
        }
        // Identifiers: `H(...)` — Heaviside; `exp(...)` — natural exponential;
        // a bare path-length symbol — p.
        Some(InhoTok::Ident(name)) => {
            *pos += 1;
            if toks.get(*pos) == Some(&InhoTok::LParen) {
                *pos += 1;
                let arg = inho_expr(toks, pos, p);
                if toks.get(*pos) == Some(&InhoTok::RParen) {
                    *pos += 1;
                }
                match name.as_str() {
                    // H(x): 1 when x ≥ 0, else 0 (jNeuroML step convention).
                    "H" => {
                        if arg >= 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    // exp(x): natural exponential (sigmoid densities, e.g. hdend).
                    "exp" => arg.exp(),
                    // Unknown function: fall back to the path length (lenient).
                    _ => p,
                }
            } else {
                p
            }
        }
        _ => 0.0,
    }
}
