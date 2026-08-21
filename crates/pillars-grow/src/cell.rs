//! Cells: soma + neurites + attached instruction modules.

use crate::module::{CellModule, Module};
use crate::Vec3;

/// Presynaptic identity (drives synapse sign, as in CX3D `NeuroML` type).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellType {
    Excitatory,
    Inhibitory,
}

/// A soma: a sphere at a position, with its local modules.
pub struct Soma {
    pub pos: Vec3,
    pub radius: f64,
    pub modules: Vec<Box<dyn Module>>,
}

/// A neurite (axon or dendrite): a polyline whose last point is the growth-cone tip.
pub struct Neurite {
    pub is_axon: bool,
    pub diameter: f64,
    pub points: Vec<Vec3>,
    pub modules: Vec<Box<dyn Module>>,
}

impl Neurite {
    /// The growth-cone tip (last point of the polyline).
    pub fn tip(&self) -> Vec3 {
        *self.points.last().expect("neurite has at least one point")
    }

    /// Total path length (sum of segment lengths), world units.
    pub fn length_um(&self) -> f64 {
        self.points.windows(2).map(|w| w[0].dist(w[1])).sum()
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
    pub soma: Soma,
    pub neurites: Vec<Neurite>,
    pub modules: Vec<Box<dyn CellModule>>,
}
