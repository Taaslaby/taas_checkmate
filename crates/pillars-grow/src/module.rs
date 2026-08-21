//! The instruction language: traits for the "genome" rules.
//!
//! A genome is a set of [`Module`] / [`CellModule`] instances attached to cell
//! elements. Each rule has a `run()` executed once per time step, and a set of
//! `copy_on_*` flags deciding whether it is copied into daughter elements at
//! division/branching/elongation (the CX3D self-construction mechanism).

use crate::world::World;

/// Which cell element a [`Module`] is attached to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Host {
    /// A soma (cell index).
    Soma(usize),
    /// A neurite (cell index + neurite index).
    Neurite { cell: usize, idx: usize },
}

/// A local instruction living inside a cell element (CX3D `LocalBiologyModule`).
pub trait Module {
    /// Perform this rule's action for one time step.
    fn run(&mut self, world: &mut World, host: Host);

    /// Copy this rule into each daughter soma at cell division.
    fn copy_on_divide(&self) -> bool {
        false
    }
    /// Copy this rule into a new branch at neurite bifurcation.
    fn copy_on_branch(&self) -> bool {
        false
    }
    /// Copy this rule when the neurite merely elongates (not a new branch).
    fn copy_on_elongate(&self) -> bool {
        false
    }

    /// Clone a fresh instance of this rule.
    fn boxed_copy(&self) -> Box<dyn Module>;
}

/// A cell-level instruction (CX3D `CellModule`), e.g. the division cycle.
pub trait CellModule {
    /// Perform this rule's action for one time step.
    fn run(&mut self, world: &mut World, cell: usize);

    /// Copy this rule into daughter cells at division.
    fn copy_on_divide(&self) -> bool {
        true
    }

    /// Clone a fresh instance of this rule.
    fn boxed_copy(&self) -> Box<dyn CellModule>;
}
