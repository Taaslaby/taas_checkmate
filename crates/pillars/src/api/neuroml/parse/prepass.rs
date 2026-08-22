//! Pre-pass-коллекции парсера: LEMS ComponentType и каналы/морфологии.
//! Выделяются до основного матча `parse()` — элементы NeuroML могут
//! объявляться в любом порядке, а ссылки резолвятся по собранным картам.

use std::collections::HashMap;

use super::super::NeuromlError;
use super::super::parse::{BiophysicalProperties, ComponentType, IrChannel, IrConcentrationModel, IrKsChannel, MorphoIR};
use super::biophysics::{parse_biophysical_properties, parse_concentration_model, parse_custom_concentration_model};
use super::channels::{parse_ion_channel, parse_ion_channel_ks};
use super::super::lems::parse_component_type;
use super::units::attr;
use super::morphology::parse_morphology;

/// Собрать кастомные LEMS `ComponentType` (gate time courses / steady
/// states) и имена Ca-pool типов (`extends="concentrationModel"`).
pub(super) fn collect_component_types(
    doc: &roxmltree::Document,
) -> Result<(HashMap<String, ComponentType>, Vec<String>), NeuromlError> {
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
    Ok((comp_types, pool_type_names))
}

/// Собранные pre-pass'ом каналы/морфологии/biophysics (см.
/// [`collect_channels_morphologies`]).
pub(super) type ChannelMorphoPrepass = (
    HashMap<String, IrChannel>,
    HashMap<String, IrKsChannel>,
    Vec<MorphoIR>,
    HashMap<String, BiophysicalProperties>,
    HashMap<String, IrConcentrationModel>,
);

/// Собрать ионные каналы (HH/KS), морфологии, biophysics и concentration
/// models — независимо от порядка объявления в документе.
pub(super) fn collect_channels_morphologies(
    doc: &roxmltree::Document,
    comp_types: &HashMap<String, ComponentType>,
    pool_type_names: &[String],
) -> Result<ChannelMorphoPrepass, NeuromlError> {
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
                let ch = parse_ion_channel(node, comp_types)?;
                channels.insert(ch.id.clone(), ch);
            }
            "ionChannelKS" => {
                let ch = parse_ion_channel_ks(node, comp_types)?;
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
    Ok((channels, ks_channels, morphologies, biophysics, concentration_models))
}
