use super::*;

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
pub(crate) fn parse_concentration_model(
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
pub(crate) fn parse_custom_concentration_model(
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
pub(crate) fn parse_biophysical_properties(
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
