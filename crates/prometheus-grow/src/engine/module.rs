//! The instruction language: traits for the "genome" rules.
//!
//! A genome is a set of [`Module`] / [`CellModule`] instances attached to cell
//! elements. Each rule has a `run()` executed once per time step, and a set of
//! `copy_on_*` flags deciding whether it is copied into daughter elements at
//! division/branching/elongation (the CX3D self-construction mechanism).

use rand::rngs::StdRng;

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

    /// Second phase of the step, run once per module after every module's
    /// `run` — for two-pass passes like elongation allocation, where a
    /// normalization must see all terminal speeds in their final state.
    fn finish_step(&mut self, _world: &mut World, _host: Host) {}

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

/// Deferred side effect produced by a [`ParCellModule`] during the parallel
/// module loop — applied serially after the loop so the module only ever
/// touches its own [`Cell`] and reads the world (no data races by
/// construction). This is the BioDynaMo `Behavior` contract: a behavior sees
/// its own agent plus a thread-safe view of the shared state.
pub enum DeferredOp {
    /// `add_substance_at(id_idx, grid_idx, amount)` — secretion.
    Secretion { substance_idx: usize, grid_idx: usize, amount: f64 },
    /// `request_division(cell)` — queue for post-loop application.
    Division(usize),
}

/// Read-only view of the world a [`ParCellModule`] may consult.
pub struct ParWorldView<'a> {
    /// Substance id → index map (built once per step).
    pub substance_ids: &'a [&'a str],
    /// All substance concentration fields (index-aligned with `substance_ids`).
    pub fields: &'a [&'a [f64]],
    pub grid: crate::substance::Grid,
    pub bounds: crate::world::Bounds,
    pub dt: f64,
    pub time: f64,
}

impl<'a> ParWorldView<'a> {
    /// Trilinear concentration of `substance_idx` at `p` (grid field only;
    /// artificial gradients are pre-baked by the caller via `add_analytic`).
    pub fn concentration(&self, substance_idx: usize, p: crate::Vec3) -> f64 {
        self.grid.trilinear(&self.fields[substance_idx], p)
    }

    /// Analytic gradient of the grid field of `substance_idx` at `p`.
    pub fn gradient(&self, substance_idx: usize, p: crate::Vec3) -> crate::Vec3 {
        self.grid.trilinear_gradient(&self.fields[substance_idx], p)
    }
}

/// A **parallel-safe** cell-level instruction (BioDynaMo `Behavior`): gets its
/// own [`Cell`] mutably plus a read-only [`ParWorldView`]. Anything that would
/// mutate shared state (secretion, division) is pushed to `ops` and applied
/// serially after the rayon loop — so the loop is data-race-free by
/// construction and can run over all cells in parallel.
///
/// Modules implementing this trait are invoked by
/// [`World::run_cell_modules_par`]; classic [`CellModule`]s keep the serial
/// loop unchanged.
pub trait ParCellModule: Send + Sync {
    // `Box<dyn ParCellModule>` is Send+Sync because the trait requires it;
    // `Cell` is Send+Sync (all fields are), so `par_iter_mut` over cells works.

    /// Perform this rule's action for one time step.
    fn run(
        &self,
        cell: &mut crate::cell::Cell,
        view: &ParWorldView,
        ops: &mut Vec<DeferredOp>,
        rng: &mut StdRng,
    );

    /// Copy this rule into daughter cells at division.
    fn copy_on_divide(&self) -> bool {
        true
    }

    /// Clone a fresh instance of this rule.
    fn boxed_copy(&self) -> Box<dyn ParCellModule>;
}
