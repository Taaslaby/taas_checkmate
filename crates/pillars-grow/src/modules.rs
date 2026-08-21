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
    /// Hard cap on neurite length (world units); `None` = unlimited.
    pub max_len_um: Option<f64>,
    /// Max turn angle per step, degrees (CX3D `NeuriteElongation.turnRate`);
    /// `None` = unlimited. Limits how sharply a cone can bend.
    pub turn_rate_deg: Option<f64>,
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
            max_len_um: None,
            turn_rate_deg: None,
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
        // Max-length cap: the neurite stops elongating (CX3D `MaxLength`).
        if let Some(max) = self.max_len_um {
            if world.cells[cell].neurites[idx].length_um() >= max {
                return;
            }
        }
        let tip = world.cells[cell].neurites[idx].tip();
        let concentration = world.concentration(&self.substance, tip);
        let grad = world.gradient(&self.substance, tip);

        // Arrived: once inside the source the growth cone stops entirely
        // (CX3D behaviour) — otherwise persistence would carry it past the
        // target and it would keep cruising through the far side.
        if concentration > self.stop_concentration {
            return;
        }

        // Normalize the initial direction: the raw gradient is a tiny vector,
        // so without this the first step would be noise-dominated and seed a
        // persistent drift off the gradient axis.
        let old = self.direction.unwrap_or(grad.normalized());
        let noise = world.random_vec(self.noise_weight);
        let step_dir = old * self.old_dir_weight + grad.normalized() * self.gradient_weight + noise;
        let step_dir = clamp_turn(old, step_dir.normalized(), self.turn_rate_deg);

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
            max_len_um: self.max_len_um,
            turn_rate_deg: self.turn_rate_deg,
            direction: None,
        })
    }
}

/// Clamp a candidate step direction to within `max_deg` of the previous
/// direction (CX3D `NeuriteElongation.turnRate`). `None` = no limit.
fn clamp_turn(prev: Vec3, dir: Vec3, max_deg: Option<f64>) -> Vec3 {
    let Some(max) = max_deg else {
        return dir;
    };
    let dot = prev.dot(dir).clamp(-1.0, 1.0);
    let ang = dot.acos();
    let limit = max.to_radians();
    if ang <= limit {
        return dir;
    }
    let axis = prev.cross(dir);
    if axis.len() <= 1e-12 {
        return prev;
    }
    // Rotate `dir` back toward `prev` by the excess angle.
    crate::math::rotate_vec(dir, axis.normalized(), -(ang - limit))
}

/// Growth cone that steers **away** from a morphogen source (CX3D
/// `NeuriteChemoRepulsion`): guidance by avoidance, e.g. crossing a midline
/// flanked by repulsive cues. Mirror of [`ChemoAttraction`] — the gradient is
/// flipped and the stop condition is inverted (no steering far from the cue).
pub struct ChemoRepulsion {
    pub substance: String,
    /// Step length per time step.
    pub speed: f64,
    /// Weight of the previous direction (persistence).
    pub old_dir_weight: f64,
    /// Weight of the (negated) concentration gradient.
    pub gradient_weight: f64,
    /// Weight of the random noise component.
    pub noise_weight: f64,
    /// Below this concentration the cue is too weak to steer (straight ahead).
    pub min_concentration: f64,
    /// Hard cap on neurite length; `None` = unlimited.
    pub max_len_um: Option<f64>,
    /// Max turn angle per step, degrees; `None` = unlimited.
    pub turn_rate_deg: Option<f64>,
    direction: Option<Vec3>,
}

impl ChemoRepulsion {
    pub fn new(substance: &str) -> Self {
        ChemoRepulsion {
            substance: substance.to_string(),
            speed: 1.0,
            old_dir_weight: 1.0,
            gradient_weight: 0.2,
            noise_weight: 0.6,
            min_concentration: 0.01,
            max_len_um: None,
            turn_rate_deg: None,
            direction: None,
        }
    }
}

impl Module for ChemoRepulsion {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        if let Some(max) = self.max_len_um {
            if world.cells[cell].neurites[idx].length_um() >= max {
                return;
            }
        }
        let tip = world.cells[cell].neurites[idx].tip();
        let concentration = world.concentration(&self.substance, tip);
        let mut grad = world.gradient(&self.substance, tip);

        // Far from the cue the signal is too weak to steer.
        if concentration < self.min_concentration {
            grad = Vec3::ZERO;
        }
        let old = self.direction.unwrap_or(grad.normalized());
        let noise = world.random_vec(self.noise_weight);
        // Away from the source: negate the gradient.
        let step_dir = old * self.old_dir_weight - grad.normalized() * self.gradient_weight + noise;
        let step_dir = clamp_turn(old, step_dir.normalized(), self.turn_rate_deg);

        world.extend_neurite(cell, idx, step_dir * self.speed);
        self.direction = Some(step_dir);
    }

    fn copy_on_branch(&self) -> bool {
        true
    }
    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(ChemoRepulsion {
            substance: self.substance.clone(),
            speed: self.speed,
            old_dir_weight: self.old_dir_weight,
            gradient_weight: self.gradient_weight,
            noise_weight: self.noise_weight,
            min_concentration: self.min_concentration,
            max_len_um: self.max_len_um,
            turn_rate_deg: self.turn_rate_deg,
            direction: None,
        })
    }
}

/// Retract the growth cone when the local concentration drops below
/// `threshold` — the neurite "fails to find its target" and pulls back
/// (CX3D `RetractNeurite`). Once fully retracted to the soma the neurite is
/// pruned (removed), and its geometric synapses disappear on the next step.
///
/// This is the pruning half of the classic growth→pruning story: exuberant
/// growth is followed by elimination of connections that missed their target.
pub struct RetractNeurite {
    pub substance: String,
    /// Below this concentration the neurite retracts.
    pub threshold: f64,
    /// Retraction speed, length units per step.
    pub rate: f64,
}

impl RetractNeurite {
    pub fn new(substance: &str, threshold: f64, rate: f64) -> Self {
        RetractNeurite {
            substance: substance.to_string(),
            threshold,
            rate,
        }
    }
}

impl Module for RetractNeurite {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let tip = world.cells[cell].neurites[idx].tip();
        if world.concentration(&self.substance, tip) < self.threshold {
            world.retract_neurite(cell, idx, self.rate);
        }
    }

    fn copy_on_divide(&self) -> bool {
        true
    }
    fn copy_on_branch(&self) -> bool {
        true
    }
    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(RetractNeurite::new(
            &self.substance,
            self.threshold,
            self.rate,
        ))
    }
}

/// Migrate the soma along a morphogen gradient (CX3D `Migration`): cells
/// "walk" toward (`attract`) or away from (`repel`) a source, building
/// layered structures. Movement is clamped to the simulation volume.
pub struct SomaMigration {
    pub substance: String,
    /// Migration speed, world units per step.
    pub speed: f64,
    /// `true` = toward higher concentration, `false` = away.
    pub attract: bool,
}

impl SomaMigration {
    pub fn new(substance: &str, speed: f64, attract: bool) -> Self {
        SomaMigration {
            substance: substance.to_string(),
            speed,
            attract,
        }
    }
}

impl Module for SomaMigration {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else {
            return;
        };
        let grad = world.gradient(&self.substance, world.cells[cell].soma.pos);
        // No movement in a flat field (avoid drift on numerical noise).
        if grad.len() <= 1e-12 {
            return;
        }
        let dir = if self.attract {
            grad.normalized()
        } else {
            Vec3::ZERO - grad.normalized()
        };
        world.move_soma(cell, dir * self.speed);
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(SomaMigration::new(
            &self.substance,
            self.speed,
            self.attract,
        ))
    }
}

/// Growth cone that bundles with nearby neurites (CX3D `NeuriteFasciculation`):
/// when the tip passes close to a foreign neurite segment, it steers along that
/// segment's direction instead of its own — axons gather into bundles, as in
/// real tracts. A standalone steering module (use it *instead of* a chemo
/// module for the bundled phase of growth).
pub struct Fasciculation {
    /// Search radius for a nearby foreign segment, world units.
    pub radius: f64,
    /// Weight of the bundle direction in the step.
    pub weight: f64,
    /// Step length per time step.
    pub speed: f64,
    /// Weight of the previous direction (persistence).
    pub old_dir_weight: f64,
    /// Weight of the random noise component.
    pub noise_weight: f64,
    /// Max turn angle per step, degrees; `None` = unlimited.
    pub turn_rate_deg: Option<f64>,
    /// Hard cap on neurite length; `None` = unlimited.
    pub max_len_um: Option<f64>,
    direction: Option<Vec3>,
}

impl Fasciculation {
    pub fn new(radius: f64) -> Self {
        Fasciculation {
            radius,
            weight: 0.6,
            speed: 1.0,
            old_dir_weight: 1.0,
            noise_weight: 0.1,
            turn_rate_deg: None,
            max_len_um: None,
            direction: None,
        }
    }
}

impl Module for Fasciculation {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        if let Some(max) = self.max_len_um {
            if world.cells[cell].neurites[idx].length_um() >= max {
                return;
            }
        }
        let tip = world.cells[cell].neurites[idx].tip();
        // Follow the nearest foreign segment's direction when in reach.
        let bundle = world
            .nearby_neurite_dir(cell, tip, self.radius)
            .map(|(d, _)| d);
        let old = self.direction.unwrap_or(bundle.unwrap_or(Vec3::ZERO));
        let noise = world.random_vec(self.noise_weight);
        let steer = bundle.unwrap_or(old);
        let step_dir = old * self.old_dir_weight + steer * self.weight + noise;
        let step_dir = clamp_turn(old, step_dir.normalized(), self.turn_rate_deg);

        world.extend_neurite(cell, idx, step_dir * self.speed);
        self.direction = Some(step_dir);
    }

    fn copy_on_branch(&self) -> bool {
        true
    }
    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(Fasciculation {
            radius: self.radius,
            weight: self.weight,
            speed: self.speed,
            old_dir_weight: self.old_dir_weight,
            noise_weight: self.noise_weight,
            turn_rate_deg: self.turn_rate_deg,
            max_len_um: self.max_len_um,
            direction: None,
        })
    }
}

/// Cell death when the local concentration drops below `threshold`
/// (CX3D `CellApoptosis`): during development roughly half of generated
/// neurons die — the cell-level counterpart of neurite pruning. Dead cells are
/// excluded from synapse formation and the extracted graph.
pub struct CellApoptosis {
    pub substance: String,
    /// Below this concentration the cell dies.
    pub threshold: f64,
}

impl CellApoptosis {
    pub fn new(substance: &str, threshold: f64) -> Self {
        CellApoptosis {
            substance: substance.to_string(),
            threshold,
        }
    }
}

impl Module for CellApoptosis {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else {
            return;
        };
        if world.cells[cell].alive
            && world.concentration(&self.substance, world.cells[cell].soma.pos) < self.threshold
        {
            world.kill_cell(cell);
        }
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(CellApoptosis::new(&self.substance, self.threshold))
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
    use crate::{ArtificialGradient, Bounds, CellType, Grid, Substance};

    fn world() -> World {
        let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
        let bounds = Bounds {
            min: Vec3::new(-80.0, -80.0, -80.0),
            max: Vec3::new(80.0, 80.0, 80.0),
        };
        World::new(grid, 1.0, bounds, 0)
    }

    /// A world with a Gaussian gradient rising toward +z (mean 80, sigma 40).
    fn gradient_world() -> World {
        let mut w = world();
        w.add_substance(Substance::new(
            "A",
            0.1,
            0.001,
            &w.grid,
            vec![ArtificialGradient::Gaussian {
                axis: 2,
                max: 1.0,
                mean: 80.0,
                sigma: 40.0,
            }],
        ));
        w
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

    #[test]
    fn chemo_repulsion_grows_away_from_source() {
        let mut w = gradient_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        let mut rep = ChemoRepulsion::new("A");
        rep.noise_weight = 0.05; // gradient dominates
        w.add_neurite_module(c, n, Box::new(rep));
        let start = w.cells[c].neurites[n].tip();
        for _ in 0..100 {
            w.step();
        }
        let tip = w.cells[c].neurites[n].tip();
        assert!(
            tip.z < start.z,
            "repulsion must steer away from +z source (start {start:?}, end {tip:?})"
        );
    }

    #[test]
    fn retract_neurite_prunes_fully_retracted() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        for _ in 0..5 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        assert_eq!(w.cells[c].neurites[n].points.len(), 6);
        // No substance "x" → concentration 0 < threshold → retract every step.
        w.add_neurite_module(c, n, Box::new(RetractNeurite::new("x", 1.0, 100.0)));
        for _ in 0..20 {
            w.step();
        }
        assert!(
            w.cells[c].neurites.is_empty(),
            "a fully retracted neurite must be pruned"
        );
    }

    #[test]
    fn retract_neurite_ignores_high_concentration() {
        let mut w = gradient_world();
        let c = w.add_cell(Vec3::new(0.0, 0.0, 75.0), CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        for _ in 0..3 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        let before = w.cells[c].neurites[n].points.len();
        // Near the source concentration is high (≫ threshold) → no retraction.
        w.add_neurite_module(c, n, Box::new(RetractNeurite::new("A", 0.3, 100.0)));
        for _ in 0..10 {
            w.step();
        }
        assert_eq!(w.cells[c].neurites[n].points.len(), before);
    }

    #[test]
    fn soma_migrates_toward_and_away_from_source() {
        let mut w = gradient_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_soma_module(c, Box::new(SomaMigration::new("A", 2.0, true)));
        for _ in 0..10 {
            w.step();
        }
        assert!(
            w.cells[c].soma.pos.z > 0.0,
            "soma must migrate toward +z source"
        );

        let d = w.add_cell(Vec3::new(0.0, 0.0, 20.0), CellType::Inhibitory, 4.0);
        w.add_soma_module(d, Box::new(SomaMigration::new("A", 2.0, false)));
        for _ in 0..10 {
            w.step();
        }
        assert!(
            w.cells[d].soma.pos.z < 20.0,
            "repelled soma must migrate away from +z source"
        );
    }

    #[test]
    fn migration_clamps_to_bounds() {
        let mut w = gradient_world();
        let c = w.add_cell(Vec3::new(0.0, 0.0, 79.0), CellType::Excitatory, 4.0);
        w.add_soma_module(c, Box::new(SomaMigration::new("A", 10.0, true)));
        for _ in 0..20 {
            w.step();
        }
        assert!(
            w.cells[c].soma.pos.z <= 80.0,
            "soma must stay inside bounds"
        );
    }

    #[test]
    fn turn_rate_clamps_step_direction() {
        let prev = Vec3::new(1.0, 0.0, 0.0);
        let target = Vec3::new(0.0, 1.0, 0.0); // 90° away
        let stepped = clamp_turn(prev, target, Some(30.0));
        let ang = prev.dot(stepped).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            (ang - 30.0).abs() < 1e-9,
            "step must be capped at 30°, got {ang}°"
        );
        // Within the limit → unchanged.
        let ok = Vec3::new(1.0, 0.5, 0.0).normalized(); // ~26.6° from +x
        assert_eq!(clamp_turn(prev, ok, Some(30.0)), ok);
        // No limit → unchanged.
        assert_eq!(clamp_turn(prev, target, None), target);
    }

    #[test]
    fn fasciculation_follows_nearby_path() {
        let mut w = world();
        // Pioneer axon laid straight along +x.
        let pioneer = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let pn = w.new_neurite(pioneer, true, 0.5);
        w.extend_neurite(pioneer, pn, Vec3::new(5.0, 0.0, 0.0));
        w.extend_neurite(pioneer, pn, Vec3::new(10.0, 0.0, 0.0));
        // Follower whose tip sits 3 units beside the path (within radius 5).
        let f = w.add_cell(Vec3::new(0.0, 3.0, 0.0), CellType::Excitatory, 4.0);
        let n = w.new_neurite(f, true, 0.5);
        let mut fas = Fasciculation::new(5.0);
        fas.noise_weight = 0.0;
        fas.old_dir_weight = 0.0; // pure bundling
        fas.weight = 1.0;
        w.add_neurite_module(f, n, Box::new(fas));
        let start = w.cells[f].neurites[n].tip();
        w.step();
        let tip = w.cells[f].neurites[n].tip();
        let moved = tip - start;
        assert!(
            moved.x > 0.0,
            "fasciculated cone must follow the +x path (moved {moved:?})"
        );
        assert!(
            moved.y.abs() < 1e-9,
            "pure bundling must keep the step on the path (moved {moved:?})"
        );
    }

    #[test]
    fn fasciculation_ignores_own_cell() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        w.extend_neurite(c, n, Vec3::new(5.0, 0.0, 0.0));
        // The only nearby segment is our own → no bundling target.
        assert!(w
            .nearby_neurite_dir(c, Vec3::new(2.0, 1.0, 0.0), 5.0)
            .is_none());
    }

    #[test]
    fn apoptosis_kills_cells_in_low_concentration() {
        let mut w = world(); // no substance → concentration 0
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_soma_module(c, Box::new(CellApoptosis::new("ntf", 0.5)));
        for _ in 0..5 {
            w.step();
        }
        assert!(!w.cells[c].alive, "cell in zero concentration must die");
        // Dead cells are excluded from the extracted graph.
        let g = w.to_graph();
        assert!(g.nodes.is_empty());
    }

    #[test]
    fn chemo_stops_at_max_length() {
        let mut w = gradient_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        let mut chemo = ChemoAttraction::new("A");
        chemo.noise_weight = 0.05;
        chemo.speed = 1.0;
        chemo.max_len_um = Some(5.0);
        w.add_neurite_module(c, n, Box::new(chemo));
        for _ in 0..100 {
            w.step();
        }
        let len = w.cells[c].neurites[n].length_um();
        assert!(
            (4.0..=6.0).contains(&len),
            "neurite must stop near max_len (len={len})"
        );
    }
}
