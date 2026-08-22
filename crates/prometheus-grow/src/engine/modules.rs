//! Built-in instruction modules (the "genome fragments"):
//! division cycle, growth-cone chemoattraction, morphogen secretion.

use crate::branch_model::BalancedForces;
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
    /// Mechanical repulsion from other cells' elements (CX3D physics lite; 0 = off).
    pub repulsion: f64,
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
            repulsion: 0.0,
            max_len_um: None,
            turn_rate_deg: None,
            direction: None,
        }
    }

    pub fn branching(mut self, f: f64) -> Self {
        self.branching_factor = f;
        self
    }

    /// Enable mechanical repulsion from other cells' elements (CX3D physics).
    pub fn repulsion(mut self, f: f64) -> Self {
        self.repulsion = f;
        self
    }
}

impl Module for ChemoAttraction {
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
        let crowd = if self.repulsion > 0.0 {
            world.crowding_force(tip, 10.0, self.repulsion, cell)
        } else {
            Vec3::ZERO
        };
        let step_dir = old * self.old_dir_weight + grad.normalized() * self.gradient_weight + noise + crowd;
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
            repulsion: self.repulsion,
            max_len_um: self.max_len_um,
            turn_rate_deg: self.turn_rate_deg,
            direction: None,
        })
    }
}

/// Growth cone that steers toward a fixed target point (identity-directed
/// guidance). Complements [`ChemoAttraction`] (which follows a diffused
/// morphogen) — used when the target is a learned per-type projection region
/// rather than a secreted gradient, e.g. growing each cell type toward where
/// its partner types live.
pub struct PointAttraction {
    /// The world position this cone grows toward.
    pub target: Vec3,
    /// Step length per time step.
    pub speed: f64,
    /// Weight of the previous direction (persistence).
    pub old_dir_weight: f64,
    /// Weight of the random noise component.
    pub noise_weight: f64,
    /// Distance to the target below which the cone stops steering.
    pub stop_distance: f64,
    direction: Option<Vec3>,
}

impl PointAttraction {
    pub fn new(target: Vec3, speed: f64) -> Self {
        PointAttraction {
            target,
            speed,
            old_dir_weight: 0.4,
            noise_weight: 0.1,
            stop_distance: 0.0,
            direction: None,
        }
    }

    pub fn stop_distance(mut self, d: f64) -> Self {
        self.stop_distance = d;
        self
    }
}

impl Module for PointAttraction {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let tip = world.cells[cell].neurites[idx].tip();
        let to = self.target - tip;
        let d = to.len();
        if d <= self.stop_distance {
            return;
        }
        let grad = to * (1.0 / d.max(1e-12));
        let old = self.direction.unwrap_or(grad);
        let noise = if self.noise_weight > 0.0 {
            world.random_vec(self.noise_weight)
        } else {
            Vec3::ZERO
        };
        let step_dir = (old * self.old_dir_weight + grad + noise).normalized();
        world.extend_neurite(cell, idx, step_dir * self.speed);
        self.direction = Some(step_dir);
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(PointAttraction {
            target: self.target,
            speed: self.speed,
            old_dir_weight: self.old_dir_weight,
            noise_weight: self.noise_weight,
            stop_distance: self.stop_distance,
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

/// Astrocyte (tripartite synapse): glia soma that secretes TSP/Sparc and
/// modulates nearby synapse weights. Non-neuronal, does not synapse itself.
pub struct Astrocyte {
    pub substance: String,
    pub amount: f64,
    pub radius: f64, // influence radius
    pub weight_boost: f64,
}

impl Astrocyte {
    pub fn new(substance: &str, amount: f64) -> Self {
        Astrocyte { substance: substance.to_string(), amount, radius: 15.0, weight_boost: 0.1 }
    }
}

impl Module for Astrocyte {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return; };
        if world.cells[cell].cell_type != crate::cell::CellType::Glia { return; }
        let pos = world.cells[cell].soma.pos;
        world.add_substance_at(&self.substance, pos, self.amount);
        // tripartite boost: nearby synapses get +weight_boost * local astro concentration
        let conc = world.concentration(&self.substance, pos);
        let boost = (conc * self.weight_boost).min(2.0);
        // apply to synapses within radius
        for s in world.synapses.iter_mut() {
            // find post position approx: use post cell soma
            let post_pos = world.cells[s.post].soma.pos;
            if pos.dist(post_pos) < self.radius {
                s.weight = (s.weight + boost).min(100.0);
            }
        }
    }
    fn copy_on_divide(&self) -> bool { false }
    fn boxed_copy(&self) -> Box<dyn Module> { Box::new(Astrocyte { substance: self.substance.clone(), amount: self.amount, radius: self.radius, weight_boost: self.weight_boost }) }
}

/// Hillman / L-Neuron diameter-based branching: elongate + taper the neurite;
/// while the diameter stays above `bifurcation_threshold` it bifurcates into
/// two daughters sized by Rall's power law, otherwise it becomes a terminal
/// segment and stops growing.
pub struct BranchOnDiameter {
    /// Diameter lost per unit length (taper).
    pub taper_rate: f64,
    /// Bifurcate while the diameter is above this.
    pub bifurcation_threshold: f64,
    /// Rall's power `p` (d_parent^p = d1^p + d2^p).
    pub rall_power: f64,
    /// Step length per time step.
    pub speed: f64,
    /// Angle between the two daughters (used when `balanced` is `None`).
    pub branch_angle: f64,
    /// NETMORPH balanced-forces branch angles (overrides `branch_angle`).
    pub balanced: Option<BalancedForces>,
    stopped: bool,
}

impl BranchOnDiameter {
    pub fn new(taper_rate: f64, bifurcation_threshold: f64) -> Self {
        BranchOnDiameter {
            taper_rate,
            bifurcation_threshold,
            rall_power: 1.5,
            speed: 1.0,
            branch_angle: std::f64::consts::FRAC_PI_3,
            balanced: None,
            stopped: false,
        }
    }

    pub fn rall_power(mut self, p: f64) -> Self {
        self.rall_power = p;
        self
    }

    pub fn speed(mut self, s: f64) -> Self {
        self.speed = s;
        self
    }

    pub fn balanced_forces(mut self, m: BalancedForces) -> Self {
        self.balanced = Some(m);
        self
    }
}

impl Module for BranchOnDiameter {
    fn run(&mut self, world: &mut World, host: Host) {
        if self.stopped {
            return;
        }
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let dir = world.cells[cell].neurites[idx].axis();
        world.extend_neurite(cell, idx, dir * self.speed);
        let d = world.cells[cell].neurites[idx].diameter;
        let new_d = (d - self.taper_rate * self.speed).max(0.01);
        world.cells[cell].neurites[idx].diameter = new_d;
        if new_d > self.bifurcation_threshold {
            if let Some(model) = &self.balanced {
                world.bifurcate_neurite_balanced(cell, idx, self.rall_power, 0.0, 0.0, model);
            } else {
                world.bifurcate_neurite(cell, idx, self.branch_angle, self.rall_power);
            }
        }
        // Hillman: the parent segment ends here — daughters (if any) continue
        // the same rule, this one stops growing.
        self.stopped = true;
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(BranchOnDiameter {
            taper_rate: self.taper_rate,
            bifurcation_threshold: self.bifurcation_threshold,
            rall_power: self.rall_power,
            speed: self.speed,
            branch_angle: self.branch_angle,
            balanced: self.balanced,
            stopped: self.stopped,
        })
    }
}

/// NETMORPH `van_Pelt_branching_model` (van Pelt & Uylings 2003): a terminal
/// segment bifurcates with probability
///
/// ```text
/// p = n_i^(-E) · B∞ · exp(-t/τ) · (exp(Δt/τ) − 1) · 2^(−S·γ_j) · C_n,
///     C_n = n_i / Σ_j 2^(−S·γ_j)
/// ```
///
/// where `n_i` is the number of terminal segments in the arbor (competition),
/// `E` the competition exponent, `γ_j` the centrifugal (branch) order, `S` the
/// order-dependence, `B∞` the asymptotic branching amplitude and `τ` the
/// branching time constant. As the arbor grows, `n_i` and `t` both rise, so
/// the probability decays and the tree self-limits — in contrast to
/// [`BranchOnDiameter`]'s deterministic diameter threshold.
pub struct VanPeltBranching {
    /// Asymptotic branching amplitude `B∞`.
    pub b_inf: f64,
    /// Branching time constant `τ`.
    pub tau: f64,
    /// Competition exponent `E` (number of terminals suppresses branching).
    pub e: f64,
    /// Order-dependence `S` (centrifugal order suppresses branching).
    pub s: f64,
    /// Angle between the two daughters (used when `balanced` is `None`).
    pub branch_angle: f64,
    /// Rall's power `p` (d_parent^p = d1^p + d2^p) for the top-down split.
    pub rall_power: f64,
    /// Step length per time step.
    pub speed: f64,
    /// Per-step probability clamp.
    pub max_prob: f64,
    /// NETMORPH balanced-forces branch angles (overrides `branch_angle`).
    pub balanced: Option<BalancedForces>,
    stopped: bool,
}

impl VanPeltBranching {
    pub fn new(b_inf: f64, tau: f64, e: f64, s: f64) -> Self {
        VanPeltBranching {
            b_inf,
            tau,
            e,
            s,
            branch_angle: std::f64::consts::FRAC_PI_3,
            rall_power: 1.5,
            speed: 1.0,
            max_prob: 0.05,
            balanced: None,
            stopped: false,
        }
    }

    pub fn speed(mut self, v: f64) -> Self {
        self.speed = v;
        self
    }

    pub fn max_prob(mut self, p: f64) -> Self {
        self.max_prob = p;
        self
    }

    pub fn balanced_forces(mut self, m: BalancedForces) -> Self {
        self.balanced = Some(m);
        self
    }
}

impl Module for VanPeltBranching {
    fn run(&mut self, world: &mut World, host: Host) {
        if self.stopped {
            return;
        }
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let dir = world.cells[cell].neurites[idx].axis();
        world.extend_neurite(cell, idx, dir * self.speed);

        // Arbor-level terminal count: the number of neurites sharing this
        // neurite's axon/dendrite identity (every neurite is a terminal
        // segment here, so this is the van Pelt competition term n_i).
        let is_axon = world.cells[cell].neurites[idx].is_axon;
        let n_i = world.cells[cell]
            .neurites
            .iter()
            .filter(|n| n.is_axon == is_axon && n.is_terminal)
            .count() as f64;
        let gamma = world.cells[cell].neurites[idx].branch_order as f64;
        let t = world.time;
        let dt = world.dt;
        // TSBM C_n normalization: n_i / Σ_j 2^(−S·γ_j) over the arbor's
        // terminal segments (exactly 1 when S = 0).
        let cn_sum: f64 = world.cells[cell]
            .neurites
            .iter()
            .filter(|n| n.is_axon == is_axon && n.is_terminal)
            .map(|n| 2f64.powf(-self.s * n.branch_order as f64))
            .sum();
        let p = van_pelt_probability(
            n_i, gamma, t, dt, cn_sum, self.b_inf, self.tau, self.e, self.s,
        )
        .clamp(0.0, self.max_prob);
        if world.rand() < p {
            if let Some(model) = &self.balanced {
                world.bifurcate_neurite_balanced(cell, idx, self.rall_power, 0.0, 0.0, model);
            } else {
                world.bifurcate_neurite(cell, idx, self.branch_angle, self.rall_power);
            }
            // this terminal has become a bifurcation node → stop growing
            self.stopped = true;
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(VanPeltBranching {
            b_inf: self.b_inf,
            tau: self.tau,
            e: self.e,
            s: self.s,
            branch_angle: self.branch_angle,
            rall_power: self.rall_power,
            speed: self.speed,
            max_prob: self.max_prob,
            balanced: self.balanced,
            stopped: false,
        })
    }
}

/// NETMORPH `van_Pelt_arbor_elongation_model` + `simple_terminal_segment_elongation_model`:
/// the arbor's total elongation resource is `N^(-F) · dt · ν0 · n` and, with
/// the simple even-split terminal model, each growth cone elongates by
/// `ν0 · N^(-F) · dt` per step — where `N` is the number of competing growth
/// cones. As the arbor branches, `N` rises and every cone grows slower, so
/// the tree self-limits in length (van Pelt & Uylings growth function).
///
/// `N` is approximated as the number of neurites sharing this neurite's
/// axon/dendrite identity (the same proxy [`VanPeltBranching`] uses).
pub struct VanPeltElongation {
    /// Initial elongation rate `ν0` (world units per time step; NETMORPH
    /// default 15.5 µm/day rescaled to the simulation's time units).
    pub nu0: f64,
    /// Competition exponent `F` (NETMORPH default 0.74).
    pub f: f64,
}

impl VanPeltElongation {
    pub fn new(nu0: f64, f: f64) -> Self {
        VanPeltElongation { nu0, f }
    }
}

impl Module for VanPeltElongation {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let is_axon = world.cells[cell].neurites[idx].is_axon;
        let n = world.cells[cell]
            .neurites
            .iter()
            .filter(|x| x.is_axon == is_axon && x.is_terminal)
            .count() as f64;
        let rate = self.nu0 * n.powf(-self.f) * world.dt;
        let dir = world.cells[cell].neurites[idx].axis();
        world.extend_neurite(cell, idx, dir * rate);
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(VanPeltElongation::new(self.nu0, self.f))
    }
}

/// NETMORPH `constrained_second_order_terminal_segment_elongation_model`
/// combined with `van_Pelt_arbor_elongation_model`. Each terminal keeps a
/// relative elongation *speed* that drifts as a second-order process
/// (`v = u + a·dt`, `a` clamped to ±1 and perturbed by
/// `N(0,0.3)·dt/inertia`, the NETMORPH TSEM PDF), then the arbor's van-Pelt
/// resource
/// `ν0 · N^(1−F) · dt` is split among terminals proportionally to their
/// speeds. Distinct persistent speeds give asymmetric growth — the terminal
/// speeds are exactly the "forces" that feed [`BalancedForces`] branch angles.
pub struct SecondOrderElongation {
    /// Arbor elongation rate `ν0` (world units per time step).
    pub nu0: f64,
    /// Competition exponent `F` (NETMORPH default 0.74).
    pub f: f64,
    /// Time constant of the speed perturbation (NETMORPH `tsem_inertia`).
    pub inertia: f64,
    /// Optional cap on the relative speed (constrained CSO).
    pub speed_constraint: Option<f64>,
    /// Per-terminal acceleration state (NETMORPH `l_i_acceleration`).
    acceleration: f64,
}

impl SecondOrderElongation {
    pub fn new(nu0: f64, f: f64, inertia: f64) -> Self {
        SecondOrderElongation {
            nu0,
            f,
            inertia,
            speed_constraint: None,
            acceleration: 0.5, // NETMORPH default
        }
    }

    /// NETMORPH `tsem_initial_acceleration` (0.0 simulates delayed elongation).
    pub fn initial_acceleration(mut self, a: f64) -> Self {
        self.acceleration = a;
        self
    }

    /// Cap the relative speed (NETMORPH constrained CSO).
    pub fn speed_constraint(mut self, c: f64) -> Self {
        self.speed_constraint = Some(c);
        self
    }
}

impl Module for SecondOrderElongation {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let dt = world.dt;

        // Phase 1: second-order speed update (perturb, clamp, integrate).
        self.acceleration += (dt / self.inertia) * TSEM_PERTURBATION_STD * normal01(world);
        self.acceleration = self.acceleration.clamp(-1.0, 1.0);
        let mut speed = world.cells[cell].neurites[idx].elongation_speed + self.acceleration * dt;
        if speed < 0.0 {
            speed = 0.0;
        }
        if let Some(c) = self.speed_constraint {
            speed = speed.min(c);
        }
        world.cells[cell].neurites[idx].elongation_speed = speed;
    }

    /// Phase 2: split the arbor resource among sibling terminals once all
    /// speeds are final. Order-independent; conserves `ν0 · N^(1−F) · dt`.
    fn finish_step(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        if !world.cells[cell].neurites[idx].is_terminal {
            return;
        }
        let dt = world.dt;
        let speed = world.cells[cell].neurites[idx].elongation_speed;
        let is_axon = world.cells[cell].neurites[idx].is_axon;
        let siblings: Vec<usize> = world.cells[cell]
            .neurites
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_axon == is_axon && n.is_terminal)
            .map(|(i, _)| i)
            .collect();
        let sum: f64 = siblings
            .iter()
            .map(|&i| world.cells[cell].neurites[i].elongation_speed)
            .sum();
        let n = siblings.len() as f64;
        let share = if sum > 0.0 { speed / sum } else { 0.0 };
        let resource = self.nu0 * n.powf(1.0 - self.f) * dt;
        let len = share * resource;
        let dir = world.cells[cell].neurites[idx].axis();
        world.extend_neurite(cell, idx, dir * len);
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(SecondOrderElongation {
            nu0: self.nu0,
            f: self.f,
            inertia: self.inertia,
            speed_constraint: self.speed_constraint,
            acceleration: self.acceleration,
        })
    }
}

/// The whole NETMORPH growth loop as one terminal instruction: second-order
/// elongation + van Pelt branching + ERI speed split + balanced-forces branch
/// angles. After bifurcating, the parent terminal stops (its `is_terminal` is
/// cleared) and only the daughters keep growing — the terminal-segment
/// competition that makes the arbor self-limit.
pub struct NetmorphGrowth {
    // elongation (second-order + van Pelt arbor)
    pub nu0: f64,
    pub f: f64,
    pub inertia: f64,
    pub speed_constraint: Option<f64>,
    // branching (van Pelt)
    pub b_inf: f64,
    pub tau: f64,
    pub e: f64,
    pub s: f64,
    pub max_prob: f64,
    // bifurcation (ERI + balanced-forces + Rall)
    pub rall_power: f64,
    pub step: f64,
    pub eri_std: f64,
    pub branch: BalancedForces,
    acceleration: f64,
    stopped: bool,
}

impl NetmorphGrowth {
    pub fn new(nu0: f64, f: f64, inertia: f64, b_inf: f64, tau: f64, e: f64, s: f64) -> Self {
        NetmorphGrowth {
            nu0,
            f,
            inertia,
            speed_constraint: None,
            b_inf,
            tau,
            e,
            s,
            max_prob: 0.05,
            rall_power: 1.5,
            step: 1.0,
            eri_std: 0.0,
            branch: BalancedForces::netmorph_defaults(),
            acceleration: 0.5,
            stopped: false,
        }
    }

    pub fn speed_constraint(mut self, c: f64) -> Self {
        self.speed_constraint = Some(c);
        self
    }

    pub fn max_prob(mut self, p: f64) -> Self {
        self.max_prob = p;
        self
    }

    pub fn rall_power(mut self, p: f64) -> Self {
        self.rall_power = p;
        self
    }

    pub fn step(mut self, s: f64) -> Self {
        self.step = s;
        self
    }

    pub fn eri_std(mut self, s: f64) -> Self {
        self.eri_std = s;
        self
    }

    pub fn branch(mut self, m: BalancedForces) -> Self {
        self.branch = m;
        self
    }

    pub fn initial_acceleration(mut self, a: f64) -> Self {
        self.acceleration = a;
        self
    }
}

impl Module for NetmorphGrowth {
    fn run(&mut self, world: &mut World, host: Host) {
        if self.stopped {
            return;
        }
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        let dt = world.dt;

        // Phase 1: second-order speed update.
        self.acceleration += (dt / self.inertia) * TSEM_PERTURBATION_STD * normal01(world);
        self.acceleration = self.acceleration.clamp(-1.0, 1.0);
        let mut speed = world.cells[cell].neurites[idx].elongation_speed + self.acceleration * dt;
        if speed < 0.0 {
            speed = 0.0;
        }
        if let Some(c) = self.speed_constraint {
            speed = speed.min(c);
        }
        world.cells[cell].neurites[idx].elongation_speed = speed;

        // van Pelt branching: decision uses the current terminal set; the
        // actual elongation is allocated in finish_step once all speeds final.
        let is_axon = world.cells[cell].neurites[idx].is_axon;
        let siblings: Vec<usize> = world.cells[cell]
            .neurites
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_axon == is_axon && n.is_terminal)
            .map(|(i, _)| i)
            .collect();
        let n = siblings.len() as f64;
        let gamma = world.cells[cell].neurites[idx].branch_order as f64;
        let t = world.time;
        let cn_sum: f64 = siblings
            .iter()
            .map(|&i| 2f64.powf(-self.s * world.cells[cell].neurites[i].branch_order as f64))
            .sum();
        let p = van_pelt_probability(
            n, gamma, t, dt, cn_sum, self.b_inf, self.tau, self.e, self.s,
        )
        .clamp(0.0, self.max_prob);
        if world.rand() < p {
            world.bifurcate_neurite_growth(
                cell,
                idx,
                self.rall_power,
                self.step,
                self.eri_std,
                &self.branch,
            );
            self.stopped = true;
        }
    }

    /// Phase 2: allocate the arbor resource among surviving terminals.
    fn finish_step(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else {
            return;
        };
        if !world.cells[cell].neurites[idx].is_terminal {
            return;
        }
        let dt = world.dt;
        let speed = world.cells[cell].neurites[idx].elongation_speed;
        let is_axon = world.cells[cell].neurites[idx].is_axon;
        let siblings: Vec<usize> = world.cells[cell]
            .neurites
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_axon == is_axon && n.is_terminal)
            .map(|(i, _)| i)
            .collect();
        let sum: f64 = siblings
            .iter()
            .map(|&i| world.cells[cell].neurites[i].elongation_speed)
            .sum();
        let n = siblings.len() as f64;
        let share = if sum > 0.0 { speed / sum } else { 0.0 };
        let resource = self.nu0 * n.powf(1.0 - self.f) * dt;
        let dir = world.cells[cell].neurites[idx].axis();
        world.extend_neurite(cell, idx, dir * (share * resource));
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn copy_on_elongate(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(NetmorphGrowth {
            nu0: self.nu0,
            f: self.f,
            inertia: self.inertia,
            speed_constraint: self.speed_constraint,
            b_inf: self.b_inf,
            tau: self.tau,
            e: self.e,
            s: self.s,
            max_prob: self.max_prob,
            rall_power: self.rall_power,
            step: self.step,
            eri_std: self.eri_std,
            branch: self.branch,
            acceleration: self.acceleration,
            stopped: false,
        })
    }
}

/// NETMORPH TSEM perturbation std: `normal_pdf(X_elongate, 0.0, 0.3, 1.0)`.
/// The second-order elongation models perturb acceleration by
/// `TSEM_PERTURBATION_STD · dt / inertia`.
const TSEM_PERTURBATION_STD: f64 = 0.3;

/// NETMORPH van Pelt per-terminal branching probability (Koene et al. Eq.6):
///
/// ```text
/// p = n_i^(-E) · B∞ · exp(-t/τ) · (exp(dt/τ) − 1) · 2^(−S·γ) · C_n,
///     C_n = n_i / Σ_j 2^(−S·γ_j)
/// ```
///
/// `C_n` is the TSBM normalization over the arbor's terminal segments and is
/// exactly 1 when `S = 0`. `cn_sum` is the denominator `Σ_j 2^(−S·γ_j)`.
fn van_pelt_probability(
    n_i: f64,
    gamma: f64,
    t: f64,
    dt: f64,
    cn_sum: f64,
    b_inf: f64,
    tau: f64,
    e: f64,
    s: f64,
) -> f64 {
    let cn = n_i / cn_sum;
    n_i.powf(-e)
        * b_inf
        * (-t / tau).exp()
        * ((dt / tau).exp() - 1.0)
        * 2f64.powf(-s * gamma)
        * cn
}

/// Standard normal draw via Box–Muller from the world's uniform RNG.
fn normal01(world: &mut World) -> f64 {
    let u1 = world.rand().max(1e-12);
    let u2 = world.rand();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Clamp a step direction to a maximum turn angle from the previous direction.
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
/// `ChemoRepulsion`): the mirror image of [`ChemoAttraction`]. Below
/// `min_concentration` the cue is too weak to steer and the cone goes straight.
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

/// Delta–Notch lateral inhibition — cells in a field pick their own fates
/// (Collier, Monk, Maini & Lewis, 1996, *Bull. Math. Biol.* 58(6)).
///
/// Each cell carries `delta` (a secreted ligand) and `notch` (its receptor
/// activity). Delta production is repressed by the cell's own Notch, Notch is
/// activated by the mean Delta of its neighbours:
///
/// ```text
/// dD/dt = v / (1 + b·N^h) − D
/// dN/dt = ⟨D⟩^h / (α + ⟨D⟩^h) − N
/// ```
///
/// The instability amplifies the initial jitter: the first cell to raise its
/// Delta drives its neighbours' Notch up, repressing *their* Delta — a
/// regular, self-organized spacing of primary cells emerges (the classic
/// proneural pattern). A cell whose Notch stays low becomes primary
/// (`role = primary_role`).
pub struct LateralInhibition {
    /// Neighbourhood radius (mean Delta over cells within this distance).
    pub radius: f64,
    /// Delta production rate.
    pub v: f64,
    /// Notch repression strength on Delta production.
    pub b: f64,
    /// Hill coefficient.
    pub hill: f64,
    /// Half-saturation of Notch activation.
    pub alpha: f64,
    /// Below this Notch activity the cell becomes primary.
    pub notch_threshold: f64,
    /// The fate assigned to primary cells.
    pub primary_role: i32,
}

impl LateralInhibition {
    pub fn new(radius: f64) -> Self {
        LateralInhibition {
            radius,
            v: 10.0,
            b: 100.0,
            hill: 2.0,
            alpha: 0.1,
            notch_threshold: 0.3,
            primary_role: 1,
        }
    }
}

impl CellModule for LateralInhibition {
    fn run(&mut self, world: &mut World, cell: usize) {
        // The Hill terms are fast; clamp the integration step for stability.
        let dt = world.dt.min(0.1);
        let neighbours = world.neighbors_within(cell, self.radius);
        let mean_d = if neighbours.is_empty() {
            0.0
        } else {
            neighbours
                .iter()
                .map(|&n| world.cells[n].delta)
                .sum::<f64>()
                / neighbours.len() as f64
        };
        let n = world.cells[cell].notch;
        let d = world.cells[cell].delta;
        let d_prod = self.v / (1.0 + self.b * n.powf(self.hill));
        let n_prod = mean_d.powf(self.hill) / (self.alpha + mean_d.powf(self.hill));
        world.cells[cell].delta = (d + dt * (d_prod - d)).max(0.0);
        world.cells[cell].notch = (n + dt * (n_prod - n)).max(0.0);

        // Fate readout from the *current* state: low Notch + high Delta =
        // primary. Bidirectional — during the early transient every cell has
        // low Notch, so a monotonic assignment would mark them all primary;
        // only the converged pattern is meaningful.
        world.cells[cell].role =
            if world.cells[cell].notch < self.notch_threshold && world.cells[cell].delta > 2.0 {
                self.primary_role
            } else {
                0
            };
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(LateralInhibition {
            radius: self.radius,
            v: self.v,
            b: self.b,
            hill: self.hill,
            alpha: self.alpha,
            notch_threshold: self.notch_threshold,
            primary_role: self.primary_role,
        })
    }
}

/// Neurotrophic competition (Oppenheim, 1991; Barde, 1989): neurons compete
/// for a shared, depletable trophic factor secreted by the target. Each cell
/// consumes the substance; a cell whose per-step uptake stays below
/// `min_uptake` past the grace period undergoes apoptosis. When the trophic
/// supply is limited this kills the losers and spares the winners — the
/// developmental mechanism that prunes ~half of generated neurons.
pub struct TrophicCompetition {
    pub substance: String,
    /// Intake = `consume_rate` · local concentration · dt.
    pub consume_rate: f64,
    /// Die if per-step uptake is below this after the grace period.
    pub min_uptake: f64,
    /// Simulation time before death checks start.
    pub grace: f64,
    age: f64,
}

impl TrophicCompetition {
    pub fn new(substance: &str) -> Self {
        TrophicCompetition {
            substance: substance.to_string(),
            consume_rate: 0.02,
            min_uptake: 1e-4,
            grace: 20.0,
            age: 0.0,
        }
    }
}

impl Module for TrophicCompetition {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else {
            return;
        };
        self.age += world.dt;
        let pos = world.cells[cell].soma.pos;
        let c = world.concentration(&self.substance, pos);
        let take = self.consume_rate * c * world.dt;
        world.consume_substance_at(&self.substance, pos, take);
        if self.age > self.grace && take < self.min_uptake {
            world.kill_cell(cell);
        }
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(TrophicCompetition {
            substance: self.substance.clone(),
            consume_rate: self.consume_rate,
            min_uptake: self.min_uptake,
            grace: self.grace,
            age: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, CellType, Grid, Vec3};

    fn world() -> World {
        let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
        World::new(
            grid,
            1.0,
            Bounds {
                min: Vec3::ZERO,
                max: Vec3::new(400.0, 400.0, 400.0),
            },
            7,
        )
    }

    #[test]
    fn elongation_slows_as_competitors_grow() {
        let mut w = world();
        let c = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
        let n0 = w.new_neurite(c, true, 1.0);
        w.extend_neurite(c, n0, Vec3::new(0.0, 0.0, 10.0));
        // ν0 = 1, F = 1 → rate = 1/N.
        let mut el = VanPeltElongation::new(1.0, 1.0);

        let tip = w.cells[c].neurites[n0].tip();
        el.run(&mut w, Host::Neurite { cell: c, idx: n0 });
        let grown_one = tip.dist(w.cells[c].neurites[n0].tip());
        assert!((grown_one - 1.0).abs() < 1e-9, "1 конкурент → rate 1.0");

        // Second axon terminal → N = 2 → rate halves.
        w.new_neurite(c, true, 1.0);
        let tip = w.cells[c].neurites[n0].tip();
        el.run(&mut w, Host::Neurite { cell: c, idx: n0 });
        let grown_two = tip.dist(w.cells[c].neurites[n0].tip());
        assert!((grown_two - 0.5).abs() < 1e-9, "2 конкурента → rate 0.5");
    }

    #[test]
    fn second_order_splits_resource_by_speed() {
        let mut w = world();
        let c = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
        let n0 = w.new_neurite(c, true, 1.0);
        let n1 = w.new_neurite(c, true, 1.0);
        w.extend_neurite(c, n0, Vec3::new(0.0, 0.0, 10.0));
        w.extend_neurite(c, n1, Vec3::new(0.0, 0.0, 10.0));
        // Distinct terminal speeds; disable acceleration drift.
        w.cells[c].neurites[n0].elongation_speed = 3.0;
        w.cells[c].neurites[n1].elongation_speed = 1.0;
        let mut el = SecondOrderElongation::new(4.0, 1.0, 1e12).initial_acceleration(0.0);
        // resource = ν0 · N^(1−F) · dt = 4 · 2^0 · 1 = 4; share = 3/4 → len 3.0.
        let tip = w.cells[c].neurites[n0].tip();
        el.run(&mut w, Host::Neurite { cell: c, idx: n0 });
        el.finish_step(&mut w, Host::Neurite { cell: c, idx: n0 });
        let grown = tip.dist(w.cells[c].neurites[n0].tip());
        assert!((grown - 3.0).abs() < 1e-9, "got {grown}");
    }

    #[test]
    fn elongation_allocation_conserves_resource() {
        let mut w = world();
        let c = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
        let n0 = w.new_neurite(c, true, 1.0);
        let n1 = w.new_neurite(c, true, 1.0);
        w.extend_neurite(c, n0, Vec3::new(0.0, 0.0, 10.0));
        w.extend_neurite(c, n1, Vec3::new(0.0, 0.0, 10.0));
        w.cells[c].neurites[n0].elongation_speed = 3.0;
        w.cells[c].neurites[n1].elongation_speed = 1.0;
        let make = || SecondOrderElongation::new(4.0, 1.0, 1e12).initial_acceleration(0.0);
        w.add_neurite_module(c, n0, Box::new(make()));
        w.add_neurite_module(c, n1, Box::new(make()));
        let t0 = w.cells[c].neurites[n0].tip();
        let t1 = w.cells[c].neurites[n1].tip();
        w.step();
        let g0 = t0.dist(w.cells[c].neurites[n0].tip());
        let g1 = t1.dist(w.cells[c].neurites[n1].tip());
        assert!((g0 - 3.0).abs() < 1e-9, "быстрый терминал растёт на 3.0, got {g0}");
        assert!((g1 - 1.0).abs() < 1e-9, "медленный терминал растёт на 1.0, got {g1}");
        assert!((g0 + g1 - 4.0).abs() < 1e-9, "сумма сохраняет ресурс 4.0, got {}", g0 + g1);
    }

    #[test]
    fn point_attraction_steers_toward_target() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 0.5);
        let mut pa = PointAttraction::new(Vec3::new(0.0, 0.0, 100.0), 10.0);
        pa.noise_weight = 0.0;
        pa.old_dir_weight = 0.0; // pure target following, no persistence
        w.add_neurite_module(c, n, Box::new(pa));
        let start = w.cells[c].neurites[n].tip();
        for _ in 0..20 {
            w.step();
        }
        let tip = w.cells[c].neurites[n].tip();
        assert!(
            tip.z > start.z + 50.0,
            "конус должен двигаться к цели: start {start:?}, tip {tip:?}"
        );
    }

    #[test]
    fn netmorph_growth_grows_arbor_and_marks_terminals() {
        let mut w = world();
        let c = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
        let n = w.new_neurite(c, true, 2.0);
        w.extend_neurite(c, n, Vec3::new(0.0, 0.0, 10.0));
        w.add_neurite_module(
            c,
            n,
            Box::new(
                NetmorphGrowth::new(2.0, 0.74, 5.0, 100.0, 1.0, 0.0, 0.0)
                    .max_prob(1.0)
                    .eri_std(0.2),
            ),
        );
        for _ in 0..3 {
            w.step();
        }
        let neurites = &w.cells[c].neurites;
        assert!(neurites.len() > 1, "арбор должен разветвиться");
        let terminals = neurites.iter().filter(|x| x.is_terminal).count();
        assert!(terminals >= 2);
        assert!(
            terminals < neurites.len(),
            "после бифуркаций родители не должны оставаться терминалами"
        );
    }

    #[test]
    fn van_pelt_cn_cancels_order_term_for_single_terminal() {
        // One terminal: C_n = 2^(S·γ) exactly cancels 2^(−S·γ), so S must not
        // change the branching probability.
        let dt = 1.0;
        let p_s0 = van_pelt_probability(1.0, 0.0, 0.0, dt, 1.0, 100.0, 1.0, 0.0, 0.0);
        let cn_sum = 2f64.powf(-2.0); // γ = 2, S = 1
        let p_s1 = van_pelt_probability(1.0, 2.0, 0.0, dt, cn_sum, 100.0, 1.0, 0.0, 1.0);
        assert!((p_s0 - p_s1).abs() < 1e-9);
    }

    #[test]
    fn van_pelt_cn_normalizes_total_across_orders() {
        // Two terminals (orders γ=1,2) with S=1: per-terminal probabilities
        // differ, but their sum equals the S=0 arbor total (2·p0).
        let dt = 1.0;
        let p0 = van_pelt_probability(2.0, 1.0, 0.0, dt, 2.0, 100.0, 1.0, 0.0, 0.0);
        let cn_sum = 2f64.powf(-1.0) + 2f64.powf(-2.0); // = 0.75
        let p1 = van_pelt_probability(2.0, 1.0, 0.0, dt, cn_sum, 100.0, 1.0, 0.0, 1.0);
        let p2 = van_pelt_probability(2.0, 2.0, 0.0, dt, cn_sum, 100.0, 1.0, 0.0, 1.0);
        assert!((p1 + p2 - 2.0 * p0).abs() < 1e-9);
    }
}
