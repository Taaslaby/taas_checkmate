use super::*;

use std::collections::HashMap;
use std::sync::Arc as Rc;

use crate::api::network::Network;
use crate::api::neuroml::discretize::{
    discretize, passive_leak_reversal, thin_shell_pool_flux_scale, ActiveChannel, CaPool,
    CableParams, CablePopulation, GhkActiveChannel, MarkovActiveChannel, NernstActiveChannel,
    StochasticMarkovActiveChannel, StochasticNernstActiveChannel,
};
use crate::api::neuroml::parse::{
    is_passive_leak, resolve_channel, resolve_markov_channel, BiophysicalProperties, IrChannel,
    IrChannelDensity, IrChannelPopulation, IrConcentrationModel, IrKsChannel, MorphoCell, MorphoIR,
    NonUniformParam, ResolvedChannel,
};
use crate::api::neuroml::{CableCompartment, CableInfo, GpuPopMeta, NeuromlError};
use crate::api::population::PopulationId;
use crate::kernels::hh_channels::{ChannelMeta, GateKinetics, GateType, MarkovGateSpec};

/// A resolved active cable channel (before area scaling): HH gates (exponential
/// Euler) or Markov gates (master equation) + reversal potential. The gate
/// spec is `Arc`-shared — every compartment of a segment group attaches the
/// same resolved channel (only `g_max_us` differs), so the spec is copied
/// once per channel *type*, not per compartment.
enum CableChannel {
    /// `(gates, per-gate q10, e_rev)`.
    Hh(Rc<[(GateKinetics, u32)]>, Rc<[f32]>, f32),
    /// `(gates, per-gate q10, e_rev)`.
    Markov(Rc<[MarkovGateSpec]>, Rc<[f32]>, f32),
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
            Ok(CableChannel::Hh(r.gates.into(), r.q10s.into(), r.e_rev))
        } else {
            let r = resolve_markov_channel(ch, erev, 0.0, celsius)?;
            Ok(CableChannel::Markov(r.gates.into(), r.q10s.into(), r.e_rev))
        }
    } else if let Some(ch) = ks.get(ion_channel) {
        Ok(CableChannel::Markov(
            ch.gates.clone().into(),
            // ionChannelKS gates carry no q10Settings.
            vec![1.0; ch.gates.len()].into(),
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
pub(crate) fn add_morpho_population(
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
                            Rc::clone(gates),
                            Rc::clone(q10s),
                            *e_rev,
                            g_max_us,
                            e_leak,
                        ))
                    }
                    CableChannel::Markov(gates, q10s, e_rev) => mk_by_comp
                        .entry(comp)
                        .or_default()
                        .push(MarkovActiveChannel::new(
                            Rc::clone(gates),
                            Rc::clone(q10s),
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
                                Rc::clone(gates),
                                Rc::clone(q10s),
                                *e_rev,
                                g_max_us,
                                e_leak,
                            ))
                        }
                        CableChannel::Markov(gates, q10s, e_rev) => mk_by_comp
                            .entry(comp)
                            .or_default()
                            .push(MarkovActiveChannel::new(
                                Rc::clone(gates),
                                Rc::clone(q10s),
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
pub(crate) fn channel_to_meta(ch: &ResolvedChannel) -> ChannelMeta {
    ChannelMeta {
        n_gates: ch.gates.len(),
        e_rev: ch.e_rev,
        powers: ch.gates.iter().map(|g| g.1).collect(),
        gate_type: vec![GateType::NaM; ch.gates.len()],
        kinetics: ch.gates.iter().map(|g| g.0.clone()).collect(),
    }
}
