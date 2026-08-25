//! Cells: soma + neurites + attached instruction modules.
//
// References:
//   - Zubler & Douglas (2009) on CX3D: Simulating cortical development in 3D
//   - Bauer et al. (2014) on developmental simulation schedulers and discrete-time loops

use crate::module::{CellModule, Module};
use crate::Vec3;

/// Presynaptic identity (drives synapse sign, as in CX3D `NeuroML` type).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellType {
    Excitatory,
    Inhibitory,
    Glia, // astrocyte — tripartite, no synapses itself
}

impl CellType {
    /// Index for recognition matrices (Excitatory = 0, Inhibitory = 1, Glia = 2).
    pub fn as_usize(self) -> usize {
        match self {
            CellType::Excitatory => 0,
            CellType::Inhibitory => 1,
            CellType::Glia => 2,
        }
    }
}

/// A soma: a sphere at a position, with its local modules.
pub struct Soma {
    pub pos: Vec3,
    pub radius: f64,
    pub modules: Vec<Box<dyn Module>>,
}

/// Kind of a synaptic excrescence (CX3D `Excrescence`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExcrescenceKind {
    /// Presynaptic swelling on an axon (`PhysicalBouton`).
    Bouton,
    /// Postsynaptic swelling on a dendrite (`PhysicalSpine`).
    Spine,
}

/// A synaptic excrescence protruding from a neurite: a bouton (pre) or spine
/// (post) at a position along the neurite, pointing outward (`dir`).
#[derive(Clone, Copy, Debug)]
pub struct Excrescence {
    pub kind: ExcrescenceKind,
    pub pos: Vec3,
    /// Unit protrusion direction (radial from the neurite axis).
    pub dir: Vec3,
    pub length: f64,
}

/// A neurite (axon or dendrite): a polyline whose last point is the growth-cone tip.
pub struct Neurite {
    pub is_axon: bool,
    pub diameter: f64,
    pub points: Vec<Vec3>,
    pub modules: Vec<Box<dyn Module>>,
    /// Intracellular substance amounts (CX3D `IntracellularSubstance`, e.g.
    /// tubulin): diffuses between connected segments, read by growth cones.
    pub intracellular: std::collections::HashMap<String, f64>,
    /// Synaptic excrescences (boutons on axons, spines on dendrites).
    pub excrescences: Vec<Excrescence>,
    /// Branch depth (0 = primary, 1 = daughter, …; CX3D `branchOrder`).
    pub branch_order: u32,
    /// Index of the parent neurite within the same cell (`None` for the
    /// primary neurite). Enables NETMORPH-style bottom-up Rall diameter
    /// assignment.
    pub parent: Option<usize>,
    /// Per-terminal relative elongation speed (NETMORPH terminal-segment
    /// elongation model state). `1.0` = even share; terminals with higher
    /// values elongate a larger fraction of the arbor's resources. Kept on
    /// the neurite so sibling terminals can normalize against it.
    pub elongation_speed: f64,
    /// Whether this neurite is still a growing terminal (growth cone). After
    /// bifurcation the parent becomes an intermediate segment (`false`) and
    /// only the daughters count as terminals — NETMORPH terminal-segment
    /// competition scope.
    pub is_terminal: bool,
}

impl Neurite {
    /// Total polyline length.
    pub fn length_um(&self) -> f64 {
        self.points.windows(2).map(|w| w[0].dist(w[1])).sum()
    }

    /// The growth-cone tip (last point of the polyline).
    pub fn tip(&self) -> Vec3 {
        *self.points.last().expect("neurite has at least one point")
    }

    /// Direction of the last segment (the growth direction).
    pub fn axis(&self) -> Vec3 {
        let n = self.points.len();
        if n >= 2 {
            (self.points[n - 1] - self.points[n - 2]).normalized()
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        }
    }
}

/// A cell: soma + neurites + cell-level modules.
pub struct Cell {
    pub id: usize,
    pub cell_type: CellType,
    /// Developmental fate (positional identity read from a morphogen gradient,
    /// e.g. segment index / stance-vs-swing). Assigned by [`Differentiate`].
    pub role: i32,
    /// Secondary fate (e.g. stance=0 / swing=1), assigned by asymmetric division.
    pub sub_role: i32,
    /// Per-neuron identity tag for partner recognition (e.g. MANC `class` or
    /// `hemilineage`). Empty = untyped. Used by [`TagRecognition`].
    pub tag: String,
    pub soma: Soma,
    pub neurites: Vec<Neurite>,
    pub modules: Vec<Box<dyn CellModule>>,
    /// Membrane-bound substance amounts (CX3D `modifyMembraneQuantity`, e.g.
    /// adhesion molecules visible to contacting partners).
    pub membrane: std::collections::HashMap<String, f64>,
    /// Intracellular substance amounts at the soma (source compartment for
    /// intracellular diffusion along the neurite tree).
    pub intracellular: std::collections::HashMap<String, f64>,
    /// `false` after apoptosis ([`crate::CellApoptosis`]): the cell stops
    /// growing, forms no synapses and is excluded from the extracted graph.
    /// It stays in the list (index stability), it just stops participating.
    pub alive: bool,
    /// Delta–Notch signalling state (lateral inhibition, Collier et al. 1996).
    /// [`crate::LateralInhibition`] integrates these ODEs; fate decisions are
    /// read off `notch`/`delta`. Meaningless without that module.
    pub delta: f64,
    pub notch: f64,
}
