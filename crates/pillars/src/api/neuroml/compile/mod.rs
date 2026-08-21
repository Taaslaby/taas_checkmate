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
//!
//! ## Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`stimulus`] | spike-train generation + `resolve_input_gen` (pulse/sine/ramp/spike sources) |
//! | [`population`] | `add_population` — point-neuron cells → populations + `current_scale` |
//! | [`cable`] | cable-channel resolution + `add_morpho_population` + `channel_to_meta` |
//! | [`inhomogeneous`] | NeuroML `inhomogeneousValue` density-expression evaluator |

use std::collections::HashMap;

use crate::api::graph::MAX_DELAY_SLOTS;
use crate::api::network::{Edge, Network, SigmoidGradedSynapse};
use crate::api::population::PopulationId;
use crate::api::{PlasticityProfile, SynapseKinetics};

use super::parse::{
    BiophysicalProperties, Cell, CellSpec, DoubleSynapseSpec, GapJunctionSpec, GradedSynapseSpec,
    Ir, IrChannel, IrConcentrationModel, IrKsChannel, MorphoCell, MorphoIR, PlasticitySpec,
    PoissonSpec, SigmoidGradedSynapseSpec, Synapse, SynapseKind,
};
use super::{
    CableInfo, ClampSchedule, GpuPopMeta, LoadedNetwork, NeuromlError, Stimulus, StimulusKind,
    VoltageClamp,
};

mod cable;
mod inhomogeneous;
mod population;
mod stimulus;

pub(crate) use cable::*;
pub(crate) use inhomogeneous::*;
pub(crate) use population::*;
pub(crate) use stimulus::*;

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
        warnings: ir.warnings_sorted(),
    })
}
