//! Built-in instruction modules (the "genome fragments"):
//! division cycle, growth-cone chemoattraction, morphogen secretion.

use crate::module::{CellModule, Host, Module};
use crate::world::World;
use crate::Vec3;

/// Grow the soma until a minimum diameter, then divide (CX3D `SimpleCellCycle`).
pub struct SimpleCellCycle {
    /// Volume added per step (world units³).
    pub d_vol_dt: f64,
    /// Divide once the soma diameter reaches this.
    pub min_diameter: f64,
}

impl SimpleCellCycle {
    pub fn new(d_vol_dt: f64, min_diameter: f64) -> Self {
        SimpleCellCycle {
            d_vol_dt,
            min_diameter,
        }
    }
}

impl CellModule for SimpleCellCycle {
    fn run(&mut self, world: &mut World, cell: usize) {
        let diameter = world.cells[cell].soma.radius * 2.0;
        if diameter < self.min_diameter {
            world.grow_soma(cell, self.d_vol_dt);
        } else {
            world.request_division(cell);
        }
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(SimpleCellCycle::new(self.d_vol_dt, self.min_diameter))
    }
}

/// Divide exactly once, then stop (deterministic 1 → 2 cells).
///
/// Unlike [`SimpleCellCycle`] (which keeps cycling and grows a colony), this
/// models a single mitotic event: the progenitor splits once and the two
/// daughters do not divide further. This is the controlled-division building
/// block for growing precise small motifs (e.g. a half-center).
pub struct DivideOnce {
    pub d_vol_dt: f64,
    pub min_diameter: f64,
    divided: bool,
}

impl DivideOnce {
    pub fn new(d_vol_dt: f64, min_diameter: f64) -> Self {
        DivideOnce {
            d_vol_dt,
            min_diameter,
            divided: false,
        }
    }
}

impl CellModule for DivideOnce {
    fn run(&mut self, world: &mut World, cell: usize) {
        if self.divided {
            return;
        }
        let diameter = world.cells[cell].soma.radius * 2.0;
        if diameter < self.min_diameter {
            world.grow_soma(cell, self.d_vol_dt);
        } else {
            world.request_division(cell);
            self.divided = true;
        }
    }

    fn copy_on_divide(&self) -> bool {
        false
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(DivideOnce::new(self.d_vol_dt, self.min_diameter))
    }
}

/// Assign a cell's developmental fate (role) from the local concentration of
/// a morphogen: `role = floor(concentration / bin_width)`.
///
/// This is positional information in the Wolpert / dpp-wg-hh sense: a cell
/// "reads" where it is in the gradient and becomes a different type accordingly
/// (segment identity, stance-vs-swing, …). The role is stored on the cell and
/// can drive wiring rules downstream.
pub struct Differentiate {
    /// Morphogen whose gradient sets the fate (e.g. `"dpp"`).
    pub substance: String,
    /// Concentration width per fate level.
    pub bin_width: f64,
}

impl Differentiate {
    pub fn new(substance: &str, bin_width: f64) -> Self {
        Differentiate {
            substance: substance.to_string(),
            bin_width,
        }
    }
}

impl CellModule for Differentiate {
    fn run(&mut self, world: &mut World, cell: usize) {
        let c = world.concentration(&self.substance, world.cells[cell].soma.pos);
        world.cells[cell].role = (c / self.bin_width).floor() as i32;
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(Differentiate::new(&self.substance, self.bin_width))
    }
}

/// Growth cone that follows a morphogen gradient with persistence + noise
/// (CX3D tutorial `NeuriteChemoAttraction`).
pub struct ChemoAttraction {
    pub substance: String,
    /// Probability per step of branching, scaled by local concentration.
    pub branching_factor: f64,
    /// Step length per time step.
    pub speed: f64,
    /// Weight of the previous direction (persistence).
    pub old_dir_weight: f64,
    /// Weight of the concentration gradient.
    pub gradient_weight: f64,
    /// Weight of the random noise component.
    pub noise_weight: f64,
    /// Concentration above which the growth cone stops.
    pub stop_concentration: f64,
    direction: Option<Vec3>,
}

impl ChemoAttraction {
    pub fn new(substance: &str) -> Self {
        ChemoAttraction {
            substance: substance.to_string(),
            branching_factor: 0.0,
            speed: 1.0,
            old_dir_weight: 1.0,
            gradient_weight: 0.2,
            noise_weight: 0.6,
            stop_concentration: 0.3,
            direction: None,
        }
    }

    pub fn branching(mut self, f: f64) -> Self {
        self.branching_factor = f;
        self
    }
}

impl Module for ChemoAttraction {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let tip = world.cells[cell].neurites[idx].tip();
        let concentration = world.concentration(&self.substance, tip);
        let mut grad = world.gradient(&self.substance, tip);

        // Stop steering once inside the source (CX3D behaviour).
        if concentration > self.stop_concentration {
            grad = Vec3::ZERO;
        }

        let old = self.direction.unwrap_or(grad);
        let noise = world.random_vec(self.noise_weight);
        let step_dir = old * self.old_dir_weight + grad.normalized() * self.gradient_weight + noise;
        let step_dir = step_dir.normalized();

        world.extend_neurite(cell, idx, step_dir * self.speed);
        self.direction = Some(step_dir);

        // Clamp the branching probability so runaway morphogen accumulation
        // cannot make every neurite bifurcate every step.
        let branch_prob = (concentration * self.branching_factor).clamp(0.0, 0.05);
        if world.rand() < branch_prob {
            world.branch_neurite(cell, idx);
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }
    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(ChemoAttraction {
            substance: self.substance.clone(),
            branching_factor: self.branching_factor,
            speed: self.speed,
            old_dir_weight: self.old_dir_weight,
            gradient_weight: self.gradient_weight,
            noise_weight: self.noise_weight,
            stop_concentration: self.stop_concentration,
            direction: None,
        })
    }
}

/// Secretes a morphogen into the nearest grid cell each step (a soma module).
pub struct Secretor {
    pub substance: String,
    pub amount: f64,
}

impl Secretor {
    pub fn new(substance: &str, amount: f64) -> Self {
        Secretor {
            substance: substance.to_string(),
            amount,
        }
    }
}

impl Module for Secretor {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else {
            return;
        };
        let pos = world.cells[cell].soma.pos;
        world.add_substance_at(&self.substance, pos, self.amount);
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(Secretor::new(&self.substance, self.amount))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, CellType, Grid, Substance};

    fn world() -> World {
        let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
        let bounds = Bounds {
            min: Vec3::new(-80.0, -80.0, -80.0),
            max: Vec3::new(80.0, 80.0, 80.0),
        };
        World::new(grid, 1.0, bounds, 0)
    }

    #[test]
    fn divide_once_stops_after_one_division() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
        for _ in 0..30 {
            w.step();
        }
        assert_eq!(
            w.cells.len(),
            2,
            "DivideOnce must split exactly once (progenitor + one daughter)"
        );
    }

    #[test]
    fn divide_once_daughter_is_module_free() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
        for _ in 0..30 {
            w.step();
        }
        // `copy_on_divide` is false → the daughter carries no division module.
        assert!(
            w.cells[1].modules.is_empty(),
            "DivideOnce must not copy itself into the daughter"
        );
    }

    #[test]
    fn simple_cell_cycle_copies_into_daughters() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_cell_module(c, Box::new(SimpleCellCycle::new(300.0, 12.0)));
        for _ in 0..30 {
            w.step();
            if w.cells.len() >= 4 {
                break;
            }
        }
        assert!(
            w.cells.len() >= 4,
            "SimpleCellCycle daughters must keep dividing, got {} cells",
            w.cells.len()
        );
    }

    #[test]
    fn differentiate_reads_morphogen_concentration() {
        let mut w = world();
        w.add_substance(Substance::new("morph", 0.1, 0.001, &w.grid, vec![]));
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        // 10 units at the exact grid node under the cell → trilinear is exact.
        w.add_substance_at("morph", Vec3::ZERO, 10.0);
        Differentiate::new("morph", 3.0).run(&mut w, c);
        assert_eq!(w.cells[c].role, 3, "role = floor(10 / 3)");
        // Zero concentration → role 0.
        let far = w.add_cell(Vec3::new(-60.0, -60.0, -60.0), CellType::Inhibitory, 4.0);
        Differentiate::new("morph", 3.0).run(&mut w, far);
        assert_eq!(w.cells[far].role, 0);
    }

    #[test]
    fn secretor_ignores_neurite_host() {
        let mut w = world();
        w.add_substance(Substance::new("morph", 0.1, 0.001, &w.grid, vec![]));
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, false, 0.5);
        let before = w.concentration("morph", Vec3::ZERO);
        // A soma-only module hosted on a neurite must no-op.
        Secretor::new("morph", 5.0).run(&mut w, Host::Neurite { cell: c, idx: n });
        assert_eq!(w.concentration("morph", Vec3::ZERO), before);
    }

    #[test]
    fn chemo_ignores_soma_host() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.new_neurite(c, true, 0.5);
        w.add_soma_module(c, Box::new(ChemoAttraction::new("x")));
        for _ in 0..10 {
            w.step();
        }
        // The neurite module never fires (host is Soma) → no elongation points.
        assert_eq!(w.cells[c].neurites[0].points.len(), 1);
    }

    #[test]
    fn copy_flags_follow_division_contract() {
        assert!(SimpleCellCycle::new(1.0, 2.0).copy_on_divide());
        assert!(!DivideOnce::new(1.0, 2.0).copy_on_divide());
        assert!(Secretor::new("x", 1.0).copy_on_divide());
        let chemo = ChemoAttraction::new("x");
        assert!(chemo.copy_on_branch());
        assert!(chemo.copy_on_elongate());
    }
}
