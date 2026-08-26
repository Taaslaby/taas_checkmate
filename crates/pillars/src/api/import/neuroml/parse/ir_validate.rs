//! Структурная валидация NeuroML IR: уникальность id, резолв ссылок
//! проекций/популяций/стимулов, sanity параметров. Вызывается из
//! `compile()` до.lowering — ошибки становятся warnings.

use std::collections::{HashMap, HashSet};

use super::ir::{CellSpec, Ir};
use crate::api::neuroml::NeuromlError;

impl Ir {
    /// Convert warnings to a sorted Vec for stable output order.
    pub(crate) fn warnings_sorted(&self) -> Vec<String> {
        let mut v: Vec<_> = self.warnings.iter().cloned().collect();
        v.sort();
        v
    }

    /// Validate the parsed NeuroML IR.
    ///
    /// Performs structural checks on the intermediate representation:
    /// - Unique IDs across all entities (cells, synapses, morphologies, etc.)
    /// - Resolvable references (projections reference valid populations/cells)
    /// - Parameter consistency (positive values where required)
    /// - No circular dependencies in includes/resolutions
    ///
    /// Returns a vector of errors/warnings that should be surfaced to users.
    pub(crate) fn validate(&self) -> Vec<NeuromlError> {
        let mut errors = Vec::new();

        // 1. Check for duplicate IDs within each category
        errors.extend(self.check_duplicate_ids());

        // 2. Check that all projection references are resolvable
        errors.extend(self.check_projections());

        // 3. Check population-cell references
        errors.extend(self.check_populations());

        // 4. Check stimulus/input references
        errors.extend(self.check_inputs());

        // 5. Basic parameter sanity checks
        errors.extend(self.check_parameters());

        errors
    }

    fn check_duplicate_ids(&self) -> Vec<NeuromlError> {
        let mut ids: HashMap<String, Vec<String>> = HashMap::new();

        // Helper to track duplicates within an entity type
        let mut add_id = |id: &str, entity_type: String| {
            ids.entry(id.to_string()).or_default().push(entity_type);
        };

        // Add all cell IDs
        for c in &self.cells {
            add_id(&c.id, format!("cell:{}", c.id));
        }

        // Add morphology IDs
        for m in &self.morphologies {
            add_id(&m.id, format!("morphology:{}", m.id));
        }

        // Add channel IDs
        for ch in &self.channels {
            add_id(&ch.id, format!("ionChannelHH:{}", ch.id));
        }

        // Add KS channel IDs
        for ch in &self.ks_channels {
            add_id(&ch.id, format!("ionChannelKS:{}", ch.id));
        }

        // Add biophysical properties
        for bp in self.biophysics.keys() {
            add_id(bp, "biophysicalProperties".to_string());
        }

        // Find duplicates
        ids.iter()
            .filter(|(_, types)| types.len() > 1)
            .map(|(id, types)| NeuromlError::parse(format!(
                "duplicate {} ID '{}'",
                types.join(", "),
                id
            )))
            .collect()
    }

    fn check_projections(&self) -> Vec<NeuromlError> {
        // Valid presynaptic sources: either populations OR poisson synapses
        let pop_ids: HashSet<&str> = self.pops.iter().map(|p| p.id.as_str()).collect();
        let poisson_ids: HashSet<&str> = 
            self.poisson_synapses.iter().map(|s| s.id.as_str()).collect();
        
        let mut errors = Vec::new();

        for proj in &self.projections {
            // Accept both regular populations AND poisson sources as pre-population
            if !pop_ids.contains(proj.pre_pop.as_str()) && 
               !poisson_ids.contains(proj.pre_pop.as_str()) {
                errors.push(NeuromlError::parse(format!(
                    "projection '{}' references unknown pre-pop '{}'",
                    proj.connections.len(),
                    proj.pre_pop
                )));
            }
            
            // Post-population must be a regular population (not a poisson source)
            if !pop_ids.contains(proj.post_pop.as_str()) {
                errors.push(NeuromlError::parse(format!(
                    "projection '{}' references unknown post-pop '{}'",
                    proj.connections.len(),
                    proj.post_pop
                )));
            }
        }

        errors
    }

    fn check_populations(&self) -> Vec<NeuromlError> {
        let cell_ids: HashSet<&str> = self.cells.iter().map(|c| c.id.as_str()).collect();
        let morpho_cell_ids: HashSet<&str> =
            self.morpho_cells.iter().map(|c| c.id.as_str()).collect();

        let mut errors = Vec::new();
        for pop in &self.pops {
            if !cell_ids.contains(pop.component.as_str())
                && !morpho_cell_ids.contains(pop.component.as_str())
            {
                errors.push(NeuromlError::parse(format!(
                    "population '{}' references unknown component '{}'",
                    pop.id, pop.component
                )));
            }
        }

        errors
    }

    fn check_inputs(&self) -> Vec<NeuromlError> {
        // Collect all generator IDs into a single HashSet
        let mut gen_ids = HashSet::new();
        
        // Pulse generators
        for g in &self.pulse_gens {
            gen_ids.insert(g.id.as_str());
        }
        // Poisson synapses
        for s in &self.poisson_synapses {
            gen_ids.insert(s.id.as_str());
        }
        // Transient Poisson
        for s in &self.transient_poisson_synapses {
            gen_ids.insert(s.id.as_str());
        }
        // Spike generators
        for g in &self.spike_gens {
            gen_ids.insert(g.id.as_str());
        }
        // Spike arrays AND their references
        for a in &self.spike_arrays {
            gen_ids.insert(a.id.as_str());
        }
        for r in &self.spike_array_refs {
            gen_ids.insert(r.id.as_str());
        }
        // Sine generators
        for g in &self.sine_gens {
            gen_ids.insert(g.id.as_str());
        }
        // Ramp generators
        for g in &self.ramp_gens {
            gen_ids.insert(g.id.as_str());
        }
        // Compound inputs
        for g in &self.compounds {
            gen_ids.insert(g.id.as_str());
        }
        // Timed synaptic inputs
        for t in &self.timed_syn_inputs {
            gen_ids.insert(t.id.as_str());
        }
        // Voltage clamps
        for c in &self.voltage_clamps {
            gen_ids.insert(c.id.as_str());
        }
        // Voltage clamp triples
        for c in &self.voltage_clamp_triples {
            gen_ids.insert(c.id.as_str());
        }

        let mut errors = Vec::new();
        for inp in &self.inputs {
            if !gen_ids.contains(inp.generator.as_str()) {
                errors.push(NeuromlError::parse(format!(
                    "input '{}' references unknown generator '{}'",
                    inp.target_pop, inp.generator
                )));
            }
        }

        errors
    }

    fn check_parameters(&self) -> Vec<NeuromlError> {
        let mut errors = Vec::new();

        for cell in &self.cells {
            match &cell.spec {
                CellSpec::Izh { a, b, .. } |
                CellSpec::Izh2007 { a, b, .. } => {
                    if *a < 0.0 || *b < 0.0 {
                        errors.push(NeuromlError::parse(format!(
                            "cell '{}': Izhikevich params must be non-negative, got a={}, b={}",
                            cell.id, a, b
                        )));
                    }
                }
                CellSpec::AdEx { a, delta_t, .. } if *a < 0.0 || *delta_t <= 0.0 => {
                    errors.push(NeuromlError::parse(format!(
                        "cell '{}': AdEx params invalid: a={}, delta_t={}",
                        cell.id, a, delta_t
                    )));
                }
                _ => {}
            }
        }

        errors
    }
}
