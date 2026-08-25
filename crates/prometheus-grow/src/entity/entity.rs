//! An entity = any living unit of the simulation: cell, neurite, glia.

use crate::{CellType, Vec3};

/// Base entity — like `bdm::Agent` (`core/agent/agent.h`).
#[derive(Clone, Debug)]
pub struct Entity {
    pub id: usize,
    pub pos: Vec3,
    pub kind: EntityKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    Cell(CellType),
    Neurite { is_axon: bool },
    Glia,
}

impl Entity {
    pub fn cell(id: usize, pos: Vec3, ct: CellType) -> Self {
        Entity {
            id,
            pos,
            kind: EntityKind::Cell(ct),
        }
    }
    pub fn glia(id: usize, pos: Vec3) -> Self {
        Entity {
            id,
            pos,
            kind: EntityKind::Glia,
        }
    }
}
