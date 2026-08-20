use crate::api::neuroml::parse::BiophysicalProperties;

pub(crate) const PI: f32 = std::f32::consts::PI;

/// Faraday constant, C/mol (LEMS `charge_per_mole`).
pub(crate) const FARADAY: f32 = 96485.3;
/// Ideal gas constant, J/(K·mol) (LEMS `idealGasConstantDims`).
pub(crate) const R_GAS: f32 = 8.314_462;
/// Ca²⁺ valence (LEMS `channelPopulationNernst`/`channelDensityNernst`).
pub(crate) const Z_CA: f32 = 2.0;

/// Passive biophysical constants for the cable discretization.
pub(crate) struct CableParams {
    /// Axial (intracellular) resistivity, Ω·cm.
    pub ra_ohm_cm: f32,
    /// Specific membrane capacitance, µF/cm².
    pub cm_uf_per_cm2: f32,
    /// Specific membrane resistance, Ω·cm² (leak conductance = 1/Rm).
    pub rm_ohm_cm2: f32,
    /// Target compartment size as a fraction of the space constant (0.1).
    pub d_lambda: f32,
}

impl Default for CableParams {
    fn default() -> Self {
        Self {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        }
    }
}

impl CableParams {
    /// Resolve the passive cable constants from parsed `<biophysicalProperties>`:
    ///
    /// * `intracellularProperties/resistivity` (Ω·cm) → `ra_ohm_cm`;
    /// * the passive leak `channelDensity` (ion `non_specific`/`pas`/`leak`)
    ///   → `rm_ohm_cm2 = 1/condDensity` (its `condDensity` is already S/cm²);
    /// * `membraneProperties/specificCapacitance` (µF/cm²) → `cm_uf_per_cm2`.
    ///
    /// Anything the model didn't declare keeps its [`Default`] value
    /// (`d_lambda` has no NeuroML source and always stays at 0.1).
    pub fn from_biophysics(bp: &BiophysicalProperties) -> Self {
        let default = Self::default();
        // rm (Ω·cm²) = 1 / g_leak (S/cm²); a non-positive density would give
        // a non-physical negative resistance, so it keeps the default.
        let rm_ohm_cm2 = bp
            .passive_leak()
            .filter(|leak| leak.cond_density > 0.0)
            .map(|leak| 1.0 / leak.cond_density)
            .unwrap_or(default.rm_ohm_cm2);
        Self {
            ra_ohm_cm: bp.intracellular.resistivity,
            cm_uf_per_cm2: bp.membrane.specific_capacitance,
            rm_ohm_cm2,
            ..default
        }
    }
}

/// Leak reversal / resting potential (mV) for the passive cable: the passive
/// leak `channelDensity`'s `erev`, falling back to `initMembPotential` when no
/// passive channel is declared.
pub(crate) fn passive_leak_reversal(bp: &BiophysicalProperties) -> f32 {
    bp.passive_leak()
        .map(|leak| leak.erev)
        .unwrap_or(bp.membrane.init_memb_potential)
}

/// A single cable compartment (a sub-cylinder of one morphology segment).
#[derive(Clone)]
pub(crate) struct Compartment {
    /// Parent compartment index (`None` for the root / soma).
    pub parent: Option<usize>,
    /// The morphology segment this compartment was discretized from.
    pub segment_id: usize,
    /// Compartment length, µm.
    pub length_um: f32,
    /// Compartment diameter, µm.
    pub diameter_um: f32,
    /// Membrane surface area, µm² (`π·d·L`).
    pub area_um2: f32,
    /// Membrane capacitance, nF (`cm·area`).
    pub cm_nf: f32,
    /// Leak conductance, µS (`area/Rm`).
    pub g_leak_us: f32,
    /// Full-length axial resistance of this compartment, MΩ
    /// (`4·Ra·L/(π·d²)`); the junction to the parent is `(R_parent+R_this)/2`.
    pub r_axial_mohm: f32,
    /// Centre position (µm) — interpolated along the parent segment
    /// (`proximal + (j+0.5)/n·(distal−proximal)`); the branch node sits at the
    /// segment's distal point. Kept for GUI/analysis views of the morphology.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Cumulative path distance from the root along the tree (µm), measured
    /// to this compartment's **centre**. Root = `length/2`; each child adds
    /// `parent.length/2 + own.length/2` (a chain of equal compartments steps
    /// by exactly one compartment length).
    pub path_um: f32,
}

/// A discretized morphology: a flat compartment list + the root index.
#[derive(Clone)]
pub(crate) struct CompartmentTree {
    pub compartments: Vec<Compartment>,
    pub root: usize,
}

impl CompartmentTree {
    /// Junction axial resistance between a compartment and its parent (MΩ),
    /// i.e. `(R_parent + R_this)/2`. `0.0` for the root.
    pub fn junction_resistance(&self, idx: usize) -> f32 {
        let c = &self.compartments[idx];
        match c.parent {
            Some(p) => (self.compartments[p].r_axial_mohm + c.r_axial_mohm) / 2.0,
            None => 0.0,
        }
    }

    /// Number of compartments.
    pub fn len(&self) -> usize {
        self.compartments.len()
    }

    /// Test-only convenience (production code never checks the tree emptiness
    /// directly — the discretizer validates a non-empty morphology).
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.compartments.is_empty()
    }

    /// Indices of compartments whose `segment_id` is in `segment_ids` (in
    /// compartment order).
    pub fn group_compartments(&self, segment_ids: &[usize]) -> Vec<usize> {
        self.compartments
            .iter()
            .enumerate()
            .filter(|(_, c)| segment_ids.contains(&c.segment_id))
            .map(|(i, _)| i)
            .collect()
    }
}
