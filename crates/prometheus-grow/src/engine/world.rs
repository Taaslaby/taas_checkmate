//! The simulation world (CX3D `ECM` + `Scheduler`): owns cells, substances,
//! synapses, time, and runs the per-step loop.
//!
//! ```text
//! step():
//!   1. diffuse substances (morphogens)
//!   2. run cell modules   (division)
//!   3. process divisions
//!   4. run soma modules   (secretion, migration)
//!   5. run neurite modules (growth cones)
//!   6. form synapses
//! ```

use std::collections::{HashMap, HashSet};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::branch_model::BalancedForces;
use crate::cell::{Cell, CellType, Excrescence, ExcrescenceKind, Neurite, Soma};
use crate::module::{CellModule, Host, Module};
use crate::substance::{BoundaryCondition, Grid, Substance};
use crate::synapse_model::SynapseFormationModel;
use crate::uniform_grid::UniformGrid;
use crate::Vec3;

/// Axis-aligned simulation volume.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

/// Sign of a synapse (set by the presynaptic cell type).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SynapseKind {
    Excitatory,
    Inhibitory,
}

/// Cell-type recognition (Dscam1/CadN analog): a synapse forms only when the
/// presynaptic type recognizes the postsynaptic type. `allowed[pre][post]`
/// indexes [`CellType::as_usize`] (0 = Excitatory, 1 = Inhibitory).
#[derive(Clone, Copy, Debug)]
pub struct Recognition {
    pub allowed: [[bool; 3]; 3],
    pub prob: [[f64; 3]; 3],
    pub max_dist: [[f64; 3]; 3],
}

impl Recognition {
    /// No recognition — every neuron type synapses with every neuron type; glia never synapses.
    pub fn all() -> Self {
        Recognition {
            allowed: [
                [true, true, false],
                [true, true, false],
                [false, false, false],
            ],
            prob: [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            max_dist: [[1.0; 3]; 3],
        }
    }

    /// Same-type recognition: excitatory↔excitatory, inhibitory↔inhibitory,
    /// no cross-type synapses, glia isolated.
    pub fn same_type() -> Self {
        Recognition {
            allowed: [
                [true, false, false],
                [false, true, false],
                [false, false, false],
            ],
            prob: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            max_dist: [[1.0; 3]; 3],
        }
    }

    /// Inhibitory-only recognition: only inhibitory→inhibitory (the reciprocal
    /// half-center motif), nothing else.
    pub fn inhibitory_only() -> Self {
        Recognition {
            allowed: [
                [false, false, false],
                [false, true, false],
                [false, false, false],
            ],
            prob: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            max_dist: [[1.0; 3]; 3],
        }
    }

    /// Learn a partner-recognition rule from real wiring: `prob[t1][t2]` is
    /// the empirical directed connect rate between type `t1` and `t2`
    /// (edge count / possible ordered pairs). Pairs with zero edges are
    /// disallowed. This is the data-driven "transition rule" — who wires to
    /// whom — replacing the hard-coded same-type gate.
    pub fn learned_from_edges(types: &[CellType], edges: &[(usize, usize)]) -> Self {
        let mut n = [0usize; 3];
        for &t in types {
            n[t.as_usize()] += 1;
        }
        let mut cnt = [[0usize; 3]; 3];
        for &(a, b) in edges {
            cnt[types[a].as_usize()][types[b].as_usize()] += 1;
        }
        let mut allowed = [[false; 3]; 3];
        let mut prob = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                if i == 2 || j == 2 {
                    continue;
                } // glia never synapses
                let opp = (n[i] * n[j]).max(1) as f64;
                prob[i][j] = cnt[i][j] as f64 / opp;
                allowed[i][j] = cnt[i][j] > 0;
            }
        }
        Recognition {
            allowed,
            prob,
            max_dist: [[1.0; 3]; 3],
        }
    }

    /// Override the type-specific maximum contact distance (NETMORPH
    /// `D_synmax.**.**`), e.g. shorter bridges for non-spiny pairs.
    pub fn with_max_dist(mut self, max_dist: [[f64; 3]; 3]) -> Self {
        self.max_dist = max_dist;
        self
    }
}

/// Partner recognition keyed by an arbitrary per-neuron tag (e.g. MANC `class`
/// or `hemilineage`) instead of the 2–3 [`CellType`]s. `prob[(a,b)]` is the
/// directed connect rate between tag `a` and tag `b` (edge count / ordered
/// pairs); pairs with zero edges are disallowed.
#[derive(Clone, Debug)]
pub struct TagRecognition {
    pub allowed: HashMap<(String, String), bool>,
    pub prob: HashMap<(String, String), f64>,
}

impl TagRecognition {
    /// Learn the tag→tag transition rule from real wiring (the same math as
    /// [`Recognition::learned_from_edges`], but over string tags).
    pub fn learned_from_edges(tags: &[String], edges: &[(usize, usize)]) -> Self {
        let mut n: HashMap<String, usize> = HashMap::new();
        for t in tags {
            *n.entry(t.clone()).or_default() += 1;
        }
        let mut cnt: HashMap<(String, String), usize> = HashMap::new();
        for &(a, b) in edges {
            *cnt.entry((tags[a].clone(), tags[b].clone())).or_default() += 1;
        }
        let mut allowed = HashMap::new();
        let mut prob = HashMap::new();
        for (ta, na) in &n {
            for (tb, nb) in &n {
                let key = (ta.clone(), tb.clone());
                let opp = (na * nb).max(1) as f64;
                let c = cnt.get(&key).copied().unwrap_or(0);
                prob.insert(key.clone(), c as f64 / opp);
                allowed.insert(key, c > 0);
            }
        }
        TagRecognition { allowed, prob }
    }
}

/// A directed synapse (pre → post).
#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub kind: SynapseKind,
    /// Synaptic strength, derived from the growth geometry (see `form_synapses`).
    pub weight: f64,
}

/// A grown neuron (cell) in the extracted graph.
#[derive(Clone, Debug)]
pub struct GrownNode {
    pub id: usize,
    pub cell_type: CellType,
    pub pos: Vec3,
}

/// A grown directed edge in the extracted graph.
#[derive(Clone, Copy, Debug)]
pub struct GrownEdge {
    pub pre: usize,
    pub post: usize,
    pub kind: SynapseKind,
    pub weight: f64,
}

/// The grown network: nodes + edges (pre/post/kind/weight).
#[derive(Clone, Debug)]
pub struct GrownGraph {
    pub nodes: Vec<GrownNode>,
    pub edges: Vec<GrownEdge>,
}

impl CellType {
    pub fn synapse_kind(self) -> SynapseKind {
        match self {
            CellType::Excitatory => SynapseKind::Excitatory,
            CellType::Inhibitory => SynapseKind::Inhibitory,
            CellType::Glia => SynapseKind::Excitatory, // glia never synapses, fallback
        }
    }
}

pub struct World {
    pub cells: Vec<Cell>,
    pub substances: Vec<Substance>,
    pub synapses: Vec<Synapse>,
    pub grid: Grid,
    pub time: f64,
    pub dt: f64,
    pub bounds: Bounds,
    /// Distance below which a growth cone forms a synapse.
    pub synapse_dist: f64,
    /// Global cap on neurites to stop runaway branching.
    pub max_neurites: usize,
    /// Global cap on cells to stop runaway division.
    pub max_cells: usize,
    /// Baseline synaptic strength; weight = strength / (1 + contact distance).
    pub synapse_strength: f64,
    /// Type-recognition gate for synapse formation (Dscam1/CadN analog).
    pub recognition: Option<Recognition>,
    rng: StdRng,
    /// Parallel-safe cell modules (BioDynaMo `Behavior` style), indexed by
    /// cell. Run by [`World::run_cell_modules_par`] instead of the serial
    /// [`World::run_cell_modules`]. Kept out of [`Cell`] so the parallel loop
    /// can hand `&mut Cell` to a module while it reads the world.
    par_cell_modules: Vec<Vec<Box<dyn crate::module::ParCellModule>>>,
    pending_divisions: Vec<usize>,
    /// Neurites scheduled for removal at the end of the step (fully retracted
    /// growth cones — CX3D `RetractNeurite` pruning).
    pending_prunes: Vec<(usize, usize)>,

    /// Snapshot of `copy_on_branch` modules per neurite, captured before the
    /// step's module loop (so self-branching during `run` can copy them even
    /// though the live list is temporarily emptied by `std::mem::take`).
    branch_snapshot: HashMap<(usize, usize), Vec<Box<dyn Module>>>,
    /// Geometry mutation counter: bumped whenever soma/neurite geometry or the
    /// cell set changes, so `form_synapses` can reuse its contact grid across
    /// steps that leave geometry untouched (huge win for cell-only sims).
    geom_epoch: u64,
    contact_cache: Option<(f64, UniformGrid, Vec<(Vec3, usize)>)>,
}

impl World {
    pub fn new(grid: Grid, dt: f64, bounds: Bounds, seed: u64) -> Self {
        World {
            cells: vec![],
            substances: vec![],
            synapses: vec![],
            grid,
            time: 0.0,
            dt,
            bounds,
            synapse_dist: 80.0,
            max_neurites: 512,
            max_cells: 64,
            synapse_strength: 7.0,
            recognition: None,
            rng: StdRng::seed_from_u64(seed),
            par_cell_modules: vec![],
            pending_divisions: vec![],
            pending_prunes: vec![],

            branch_snapshot: HashMap::new(),
            geom_epoch: 0,
            contact_cache: None,
        }
    }

    /// Invalidate the `form_synapses` contact-grid cache (geometry changed).
    fn touch_geometry(&mut self) {
        self.geom_epoch += 1;
    }

    // ─── construction ────────────────────────────────────────────

    pub fn add_substance(&mut self, s: Substance) {
        self.substances.push(s);
    }

    /// Enable type-recognition gating for synapse formation.
    pub fn set_recognition(&mut self, r: Recognition) {
        self.recognition = Some(r);
    }

    /// Add a cell (soma sphere) and return its id (= index).
    /// Add a cell (soma sphere) and return its id (= index).
    pub fn add_cell(&mut self, pos: Vec3, cell_type: CellType, radius: f64) -> usize {
        self.touch_geometry();
        let id = self.cells.len();
        self.cells.push(Cell {
            id,
            cell_type,
            role: 0,
            sub_role: 0,
            tag: String::new(),
            soma: Soma {
                pos,
                radius,
                modules: vec![],
            },
            neurites: vec![],
            modules: vec![],
            membrane: std::collections::HashMap::new(),
            intracellular: std::collections::HashMap::new(),
            alive: true,
            delta: 0.0,
            notch: 0.0,
        });
        self.par_cell_modules.push(vec![]);
        id
    }

    pub fn add_cell_module(&mut self, cell: usize, m: Box<dyn CellModule>) {
        self.cells[cell].modules.push(m);
    }

    pub fn add_soma_module(&mut self, cell: usize, m: Box<dyn Module>) {
        self.cells[cell].soma.modules.push(m);
    }

    /// Grow a new neurite out of a soma, return its index.
    pub fn new_neurite(&mut self, cell: usize, is_axon: bool, diameter: f64) -> usize {
        self.touch_geometry();
        let base = self.cells[cell].soma.pos;
        let idx = self.cells[cell].neurites.len();
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter,
            points: vec![base],
            modules: vec![],
            excrescences: vec![],
            branch_order: 0,
            parent: None,
            elongation_speed: 1.0,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
        idx
    }

    pub fn add_neurite_module(&mut self, cell: usize, idx: usize, m: Box<dyn Module>) {
        self.cells[cell].neurites[idx].modules.push(m);
    }

    // ─── growth operations (called by modules) ───────────────────

    /// Grow the soma by a volume increment (recomputes radius).
    pub fn grow_soma(&mut self, cell: usize, d_vol: f64) {
        let r = self.cells[cell].soma.radius;
        let vol = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) + d_vol;
        let new_r = (vol / ((4.0 / 3.0) * std::f64::consts::PI)).powf(1.0 / 3.0);
        self.cells[cell].soma.radius = new_r.max(0.1);
    }

    /// Request division; applied after the module loop (so daughter cells see
    /// an intact module list to copy).
    pub fn request_division(&mut self, cell: usize) {
        if !self.pending_divisions.contains(&cell) {
            self.pending_divisions.push(cell);
        }
    }

    /// Advance the growth-cone tip of a neurite by `step`.
    pub fn extend_neurite(&mut self, cell: usize, idx: usize, step: Vec3) {
        self.touch_geometry();
        let tip = self.cells[cell].neurites[idx].tip();
        self.cells[cell].neurites[idx].points.push(tip + step);
    }

    /// Total neurite count across all cells.
    pub fn total_neurites(&self) -> usize {
        self.cells.iter().map(|c| c.neurites.len()).sum()
    }

    /// Bifurcate a neurite: a new neurite starts at the current tip, carrying
    /// the modules flagged `copy_on_branch`. No-ops once the global cap is hit.
    pub fn branch_neurite(&mut self, cell: usize, idx: usize) {
        self.touch_geometry();
        if self.total_neurites() >= self.max_neurites {
            return;
        }
        let tip = self.cells[cell].neurites[idx].tip();
        let (is_axon, diameter, branch_order, elongation_speed) = {
            let n = &self.cells[cell].neurites[idx];
            (
                n.is_axon,
                n.diameter,
                n.branch_order + 1,
                n.elongation_speed,
            )
        };
        let modules = self.branch_copies(cell, idx);
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter,
            points: vec![tip],
            modules,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
    }

    /// Bifurcate a terminal neurite into two daughters (CX3D `bifurcate`).
    /// Daughter diameters follow Rall's power law `d1^p + d2^p = d_parent^p`
    /// (symmetric split), branch order increments, and `copy_on_branch`
    /// modules are copied into both daughters. `angle` is the angle between
    /// the two daughters (default π/3 = 60°, CX3D).
    pub fn bifurcate_neurite(&mut self, cell: usize, idx: usize, angle: f64, rall_power: f64) {
        self.touch_geometry();
        if self.total_neurites() + 2 > self.max_neurites {
            return;
        }
        let tip = self.cells[cell].neurites[idx].tip();
        let axis = self.cells[cell].neurites[idx].axis();
        let d_parent = self.cells[cell].neurites[idx].diameter;
        let branch_order = self.cells[cell].neurites[idx].branch_order + 1;
        let is_axon = self.cells[cell].neurites[idx].is_axon;
        let elongation_speed = self.cells[cell].neurites[idx].elongation_speed;
        self.cells[cell].neurites[idx].is_terminal = false;
        let mods1 = self.branch_copies(cell, idx);
        let mods2 = self.branch_copies(cell, idx);
        let d_daughter = d_parent * 2f64.powf(-1.0 / rall_power);
        let (dir1, dir2) = bifurcate_directions(axis, angle);
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir1],
            modules: mods1,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir2],
            modules: mods2,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
    }

    /// Bifurcate with asymmetric daughter angles `angle1` / `angle2` (radians)
    /// on opposite sides of the parent axis (NETMORPH branch-angle models).
    /// Otherwise identical to [`Self::bifurcate_neurite`].
    pub fn bifurcate_neurite_asym(
        &mut self,
        cell: usize,
        idx: usize,
        angle1: f64,
        angle2: f64,
        rall_power: f64,
    ) {
        self.touch_geometry();
        if self.total_neurites() + 2 > self.max_neurites {
            return;
        }
        let tip = self.cells[cell].neurites[idx].tip();
        let axis = self.cells[cell].neurites[idx].axis();
        let d_parent = self.cells[cell].neurites[idx].diameter;
        let branch_order = self.cells[cell].neurites[idx].branch_order + 1;
        let is_axon = self.cells[cell].neurites[idx].is_axon;
        let elongation_speed = self.cells[cell].neurites[idx].elongation_speed;
        self.cells[cell].neurites[idx].is_terminal = false;
        let mods1 = self.branch_copies(cell, idx);
        let mods2 = self.branch_copies(cell, idx);
        let d_daughter = d_parent * 2f64.powf(-1.0 / rall_power);
        let (dir1, dir2) = bifurcate_directions_asym(axis, angle1, angle2);
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir1],
            modules: mods1,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir2],
            modules: mods2,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
    }

    /// NETMORPH `balanced_forces_branch_angle_model`: bifurcate with daughter
    /// angles derived from the parallelogram of their elongation forces.
    pub fn bifurcate_neurite_balanced(
        &mut self,
        cell: usize,
        idx: usize,
        rall_power: f64,
        force1: f64,
        force2: f64,
        model: &BalancedForces,
    ) {
        self.touch_geometry();
        let (angle1, angle2) = model.angles(force1, force2, &mut self.rng);
        self.bifurcate_neurite_asym(cell, idx, angle1, angle2, rall_power);
    }

    /// The NETMORPH growth loop closed: an elongation-rate-initialization
    /// (ERI) split + balanced-forces angles. A bifurcating terminal splits its
    /// elongation speed between the two daughters (sum-preserving, noisy),
    /// gives each an initial length proportional to its speed, and uses those
    /// speeds as the balanced-forces "forces" — so a faster daughter continues
    /// straighter and starts longer, exactly as in NETMORPH.
    pub fn bifurcate_neurite_growth(
        &mut self,
        cell: usize,
        idx: usize,
        rall_power: f64,
        step: f64,
        eri_std: f64,
        model: &BalancedForces,
    ) {
        self.touch_geometry();
        if self.total_neurites() + 2 > self.max_neurites {
            return;
        }
        let tip = self.cells[cell].neurites[idx].tip();
        let axis = self.cells[cell].neurites[idx].axis();
        let d_parent = self.cells[cell].neurites[idx].diameter;
        let branch_order = self.cells[cell].neurites[idx].branch_order + 1;
        let is_axon = self.cells[cell].neurites[idx].is_axon;
        let v_parent = self.cells[cell].neurites[idx].elongation_speed;
        self.cells[cell].neurites[idx].is_terminal = false;
        let mods1 = self.branch_copies(cell, idx);
        let mods2 = self.branch_copies(cell, idx);
        let d_daughter = d_parent * 2f64.powf(-1.0 / rall_power);

        // ERI: daughters split the parent's speed (sum-preserving) with noise.
        let delta = sample_normal(&mut self.rng, 0.0, eri_std).clamp(-0.5, 0.5);
        let v1 = v_parent * (0.5 + delta);
        let v2 = v_parent * (0.5 - delta);

        // The daughter speeds are the balanced-forces "forces".
        let (angle1, angle2) = model.angles(v1, v2, &mut self.rng);
        let (dir1, dir2) = bifurcate_directions_asym(axis, angle1, angle2);
        let len1 = (step * v1).max(1e-6);
        let len2 = (step * v2).max(1e-6);

        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir1 * len1],
            modules: mods1,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed: v1,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter: d_daughter,
            points: vec![tip, tip + dir2 * len2],
            modules: mods2,
            excrescences: vec![],
            branch_order,
            parent: Some(idx),
            elongation_speed: v2,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        });
    }

    /// NETMORPH `rall_power_law_ndm`: assign diameters bottom-up. Terminal
    /// segments get `d_term` sampled from a normal distribution; at each
    /// bifurcation `d_parent^e = d1^e + d2^e` with `e` sampled from a normal
    /// distribution (NETMORPH defaults e~N(1.47,0.3), d_term~N(0.7,0.3)).
    /// Parents are always visited after their children because children are
    /// pushed with strictly higher indices (reverse index order = post-order).
    pub fn apply_rall_diameters(
        &mut self,
        e_mean: f64,
        e_std: f64,
        d_term_mean: f64,
        d_term_std: f64,
    ) {
        // Continue the world's RNG stream (see make_excrescences).
        let mut rng = self.rng.clone();
        for ci in 0..self.cells.len() {
            let n = self.cells[ci].neurites.len();
            let mut children = vec![Vec::<usize>::new(); n];
            for i in 0..n {
                if let Some(p) = self.cells[ci].neurites[i].parent {
                    children[p].push(i);
                }
            }
            for i in (0..n).rev() {
                let d = if children[i].is_empty() {
                    sample_normal(&mut rng, d_term_mean, d_term_std)
                        .abs()
                        .max(0.01)
                } else {
                    let e = sample_normal(&mut rng, e_mean, e_std).abs().max(0.1);
                    let sum: f64 = children[i]
                        .iter()
                        .map(|&c| self.cells[ci].neurites[c].diameter.powf(e))
                        .sum();
                    sum.powf(1.0 / e)
                };
                self.cells[ci].neurites[i].diameter = d;
            }
        }
        self.rng = rng;
    }

    // ─── substances ──────────────────────────────────────────────

    pub fn concentration(&self, id: &str, p: Vec3) -> f64 {
        let mut c = 0.0;
        for s in &self.substances {
            if s.id == id {
                c += self.grid.trilinear(&s.field, p);
                for g in &s.gradients {
                    c += g.value(p);
                }
            }
        }
        c
    }

    /// Gradient of the total concentration of `id` at `p`.
    ///
    /// Analytic: the grid part uses [`Grid::trilinear_gradient`] (8 corner
    /// loads instead of the old 6 × `trilinear` = 48), and artificial
    /// profiles contribute their closed-form derivative. This is the hot
    /// path of every chemo module (called per cell per step).
    pub fn gradient(&self, id: &str, p: Vec3) -> Vec3 {
        let mut g = Vec3::ZERO;
        for s in &self.substances {
            if s.id == id {
                if !s.field.is_empty() {
                    g = g + self.grid.trilinear_gradient(&s.field, p);
                }
                for gr in &s.gradients {
                    g = g + gr.gradient(p);
                }
            }
        }
        g
    }

    /// Add `amount` of substance `id` at the nearest grid cell.
    pub fn add_substance_at(&mut self, id: &str, p: Vec3, amount: f64) {
        let (i, j, k) = self.grid.nearest(p);
        let idx = self.grid.flat(i, j, k);
        for s in self.substances.iter_mut() {
            if s.id == id {
                s.field[idx] += amount;
            }
        }
    }

    /// BioDynaMo `DiffusionGrid::Diffuse` port: explicit Euler with c1/c2 double buffer,
    /// thresholds and Neumann boundary (copy inner neighbor).
    pub fn diffuse_substances(&mut self) {
        let g = self.grid;
        let dt = self.dt;
        for s in self.substances.iter_mut() {
            if s.field.len() != s.c2.len() {
                s.c2.resize(s.field.len(), 0.0);
            }
            let bc = s.boundary;
            // Interior: explicit Euler with the Laplacian normalized by dx²
            // (the physical diffusion equation; without the dx² the scheme is
            // unstable whenever D·dt > dx²/6 and blows up to the clamp).
            // Computed as a pure parallel map (reads `field`, writes only the
            // fresh `vals` vector), so it is deterministic and race-free.
            let nx = g.nx;
            let ny = g.ny;
            let nz = g.nz;
            let inner = (nx.saturating_sub(2)) * (ny.saturating_sub(2)) * (nz.saturating_sub(2));
            if inner > 0 {
                let dx2 = g.cell * g.cell;
                let field = &s.field;
                let diff = s.diffusion;
                let deg = s.degradation;
                let lo = s.lower_threshold;
                let hi = s.upper_threshold;
                let vals: Vec<f64> = (0..inner)
                    .into_par_iter()
                    .map(|lin| {
                        let nz2 = nz - 2;
                        let ny2 = ny - 2;
                        let i = lin / (ny2 * nz2) + 1;
                        let r = lin % (ny2 * nz2);
                        let j = r / nz2 + 1;
                        let k = r % nz2 + 1;
                        let idx = g.flat(i, j, k);
                        let lap = (field[g.flat(i - 1, j, k)]
                            + field[g.flat(i + 1, j, k)]
                            + field[g.flat(i, j - 1, k)]
                            + field[g.flat(i, j + 1, k)]
                            + field[g.flat(i, j, k - 1)]
                            + field[g.flat(i, j, k + 1)]
                            - 6.0 * field[idx])
                            / dx2;
                        (field[idx] + dt * diff * lap - dt * deg * field[idx]).clamp(lo, hi)
                    })
                    .collect();
                let mut lin = 0usize;
                for i in 1..nx.saturating_sub(1) {
                    for j in 1..ny.saturating_sub(1) {
                        for k in 1..nz.saturating_sub(1) {
                            let idx = g.flat(i, j, k);
                            s.c2[idx] = vals[lin];
                            lin += 1;
                        }
                    }
                }
            }
            // Boundary: honor the configured condition (BioDynaMo DiffusionGrid).
            for i in 0..g.nx {
                for j in 0..g.ny {
                    for k in 0..g.nz {
                        if !(i == 0
                            || i == g.nx - 1
                            || j == 0
                            || j == g.ny - 1
                            || k == 0
                            || k == g.nz - 1)
                        {
                            continue;
                        }
                        let idx = g.flat(i, j, k);
                        s.c2[idx] = match bc {
                            BoundaryCondition::Dirichlet(v) => v,
                            // Zero-flux / closed: copy the nearest interior cell.
                            BoundaryCondition::Neumann(_)
                            | BoundaryCondition::Open
                            | BoundaryCondition::Closed => {
                                match (
                                    interior_index(i, g.nx),
                                    interior_index(j, g.ny),
                                    interior_index(k, g.nz),
                                ) {
                                    (Some(ii), Some(jj), Some(kk)) => s.c2[g.flat(ii, jj, kk)],
                                    // Grid too thin for an interior cell — keep value.
                                    _ => s.field[idx],
                                }
                            }
                        };
                    }
                }
            }
            std::mem::swap(&mut s.field, &mut s.c2);
        }
    }

    // ─── CX3D cell biology: soma movement, membrane, intracellular ────

    /// Translate a soma by `step` (CX3D `PhysicalSphere.movePointMass`),
    /// clamped inside the world bounds.
    pub fn move_soma(&mut self, cell: usize, step: Vec3) {
        self.touch_geometry();
        let r = self.cells[cell].soma.radius;
        let mut p = self.cells[cell].soma.pos + step;
        p.x = p.x.clamp(self.bounds.min.x + r, self.bounds.max.x - r);
        p.y = p.y.clamp(self.bounds.min.y + r, self.bounds.max.y - r);
        p.z = p.z.clamp(self.bounds.min.z + r, self.bounds.max.z - r);
        self.cells[cell].soma.pos = p;
        // The neurite proximal ends follow the soma (CX3D: the first cylinder
        // node is rigidly attached to the sphere).
        for n in self.cells[cell].neurites.iter_mut() {
            if let Some(first) = n.points.first_mut() {
                *first = p;
            }
        }
    }

    /// Mark a cell dead (apoptosis). It stops growing, forms no synapses and
    /// is excluded from the extracted graph; its index stays valid.
    pub fn kill_cell(&mut self, cell: usize) {
        if let Some(c) = self.cells.get_mut(cell) {
            c.alive = false;
        }
    }

    /// Retract the growth-cone tip by `amount` length units: the tip walks
    /// back along the polyline, dropping exhausted segments. Once the tip is
    /// back at the soma (single point), the neurite is scheduled for pruning
    /// (removed after the module loop — CX3D `RetractNeurite`).
    pub fn retract_neurite(&mut self, cell: usize, idx: usize, amount: f64) {
        let pts = &mut self.cells[cell].neurites[idx].points;
        if pts.len() <= 1 {
            self.request_prune(cell, idx);
            return;
        }
        let mut remaining = amount;
        while remaining > 0.0 && pts.len() > 1 {
            let a = pts[pts.len() - 2];
            let b = pts[pts.len() - 1];
            let seg = a.dist(b);
            if seg <= remaining + 1e-12 {
                pts.pop();
                remaining -= seg;
            } else {
                // Partial: the tip stops inside the last segment.
                let t = (seg - remaining) / seg;
                *pts.last_mut().unwrap() = a + (b - a) * t;
                remaining = 0.0;
            }
        }
        if pts.len() <= 1 {
            self.request_prune(cell, idx);
        }
    }

    /// Schedule a neurite for removal at the end of the step (so in-flight
    /// module iterations never see a shifted neurite list).
    pub fn request_prune(&mut self, cell: usize, idx: usize) {
        if !self.pending_prunes.contains(&(cell, idx)) {
            self.pending_prunes.push((cell, idx));
        }
    }

    /// Indices of alive cells (excluding `cell`) whose soma is within `radius`.
    /// Used by lateral inhibition and other neighbourhood couplings.
    pub fn neighbors_within(&self, cell: usize, radius: f64) -> Vec<usize> {
        let p = self.cells[cell].soma.pos;
        self.cells
            .iter()
            .enumerate()
            .filter(|(i, c)| *i != cell && c.alive && c.soma.pos.dist(p) <= radius)
            .map(|(i, _)| i)
            .collect()
    }

    /// Subtract `amount` of substance `id` from the nearest grid cell
    /// (trophic consumption); never goes below zero.
    pub fn consume_substance_at(&mut self, id: &str, p: Vec3, amount: f64) {
        let (i, j, k) = self.grid.nearest(p);
        let idx = self.grid.flat(i, j, k);
        for s in self.substances.iter_mut() {
            if s.id == id {
                s.field[idx] = (s.field[idx] - amount).max(0.0);
            }
        }
    }

    /// Direction of the nearest foreign neurite segment within `radius` of `p`
    /// (for fasciculation), plus the contact distance. `None` if nothing is
    /// close. Own cell and dead cells are ignored.
    pub fn nearby_neurite_dir(&self, cell: usize, p: Vec3, radius: f64) -> Option<(Vec3, f64)> {
        let mut best: Option<(Vec3, f64)> = None;
        for (ci, c) in self.cells.iter().enumerate() {
            if ci == cell || !c.alive {
                continue;
            }
            for neur in &c.neurites {
                for w in neur.points.windows(2) {
                    let d = crate::math::dist_point_segment(p, w[0], w[1]);
                    if d < radius && best.is_none_or(|(_, bd)| d < bd) {
                        let dir = (w[1] - w[0]).normalized();
                        best = Some((dir, d));
                    }
                }
            }
        }
        best
    }

    /// Cells whose soma is within `radius` of `pos` (CX3D
    /// `PhysicalObject.getPhysicalObjectsInContact` — soma-level contact).
    pub fn objects_near(&self, pos: Vec3, radius: f64) -> Vec<usize> {
        self.cells
            .iter()
            .filter(|c| (c.soma.pos - pos).len() <= radius)
            .map(|c| c.id)
            .collect()
    }

    /// Add `amount` to a cell's membrane substance (CX3D
    /// `modifyMembraneQuantity`, e.g. surface adhesion markers).
    pub fn modify_membrane(&mut self, cell: usize, id: &str, amount: f64) {
        *self.cells[cell]
            .membrane
            .entry(id.to_string())
            .or_insert(0.0) += amount;
    }

    /// Membrane substance amount of a cell (CX3D `getMembraneConcentration`).
    pub fn membrane_concentration(&self, cell: usize, id: &str) -> f64 {
        self.cells[cell].membrane.get(id).copied().unwrap_or(0.0)
    }

    /// Add `amount` to a cell's intracellular soma pool (CX3D
    /// `modifyIntracellularQuantity` on the soma).
    pub fn modify_intracellular(&mut self, cell: usize, id: &str, amount: f64) {
        *self.cells[cell]
            .intracellular
            .entry(id.to_string())
            .or_insert(0.0) += amount;
    }

    /// Intracellular substance amount at a cell's soma.
    pub fn intracellular(&self, cell: usize, id: &str) -> f64 {
        self.cells[cell]
            .intracellular
            .get(id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Add `amount` to a neurite's intracellular pool (CX3D
    /// `modifyIntracellularQuantity` on a cylinder).
    pub fn modify_neurite_intracellular(&mut self, cell: usize, idx: usize, id: &str, amount: f64) {
        *self.cells[cell].neurites[idx]
            .intracellular
            .entry(id.to_string())
            .or_insert(0.0) += amount;
    }

    /// Intracellular substance amount of a neurite (concentration proxy:
    /// amount / cylinder volume, so a short thick neurite reads higher).
    pub fn neurite_intracellular(&self, cell: usize, idx: usize, id: &str) -> f64 {
        let n = &self.cells[cell].neurites[idx];
        let len = neurite_length(n);
        let vol = std::f64::consts::PI * (n.diameter / 2.0).powi(2) * len.max(0.1);
        n.intracellular.get(id).copied().unwrap_or(0.0) / vol
    }

    /// 1D compartment diffusion of intracellular substances along the neurite
    /// tree (CX3D `IntracellularSubstance`): soma ↔ primary neurites,
    /// parent ↔ daughter, plus degradation. Fick flux between connected
    /// compartments, scaled by the compartment distance; amounts never go
    /// negative. Constants match the CX3D tutorial (`IntracellularDiffusion`).
    pub fn diffuse_intracellular(&mut self) {
        const D: f64 = 1000.0; // diffusion constant
        const DEGRAD: f64 = 0.01; // degradation constant
        let dt = self.dt;

        for ci in 0..self.cells.len() {
            let n_n = self.cells[ci].neurites.len();
            if n_n == 0 {
                continue;
            }
            // All substance ids present in this cell's tree.
            let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            for k in self.cells[ci].intracellular.keys() {
                ids.insert(k.clone());
            }
            for n in &self.cells[ci].neurites {
                for k in n.intracellular.keys() {
                    ids.insert(k.clone());
                }
            }

            for id in &ids {
                // Compartments: 0 = soma, 1..=n_n = neurites.
                let mut amount = vec![0.0; n_n + 1];
                let mut vol = vec![0.0; n_n + 1];
                amount[0] = *self.cells[ci].intracellular.get(id).unwrap_or(&0.0);
                let r = self.cells[ci].soma.radius;
                vol[0] = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
                for ni in 0..n_n {
                    let n = &self.cells[ci].neurites[ni];
                    amount[ni + 1] = *n.intracellular.get(id).unwrap_or(&0.0);
                    let len = neurite_length(n);
                    vol[ni + 1] = std::f64::consts::PI * (n.diameter / 2.0).powi(2) * len.max(0.1);
                }

                // Tree edges: soma ↔ primary, parent ↔ daughter. Edge distance
                // = the child compartment's length.
                let mut edges: Vec<(usize, usize, f64)> = Vec::new();
                for ni in 0..n_n {
                    let len = neurite_length(&self.cells[ci].neurites[ni]);
                    match self.cells[ci].neurites[ni].parent {
                        None => edges.push((0, ni + 1, len.max(1.0))),
                        Some(par) => edges.push((par + 1, ni + 1, len.max(1.0))),
                    }
                }

                // Conservative explicit flux: each edge's flux is capped by the
                // donor's amount (so amounts can never go negative and the total
                // is conserved — the naive clamp below would break conservation
                // and let concentrations run away at coarse dt).
                let mut edge_flux: Vec<(usize, usize, f64)> = edges
                    .iter()
                    .map(|&(a, b, dist)| {
                        let ca = amount[a] / vol[a].max(0.1);
                        let cb = amount[b] / vol[b].max(0.1);
                        let mut f = D * (ca - cb) * dt / dist;
                        if f > 0.0 {
                            f = f.min(amount[a].max(0.0));
                        } else {
                            f = f.max(-amount[b].max(0.0));
                        }
                        (a, b, f)
                    })
                    .collect();

                // A compartment may feed several edges; rescale its outgoing
                // fluxes so its amount stays ≥ 0 (a few passes suffice for a
                // tree — the only multi-edge donor is the soma).
                for _ in 0..4 {
                    let mut delta = vec![0.0; n_n + 1];
                    for &(a, b, f) in &edge_flux {
                        delta[a] -= f;
                        delta[b] += f;
                    }
                    let mut scaled = false;
                    for k in 0..=n_n {
                        if amount[k] + delta[k] < 0.0 {
                            let out: f64 = edge_flux
                                .iter()
                                .filter(|(a, _, _)| *a == k)
                                .map(|(_, _, f)| f)
                                .sum();
                            if out > 1e-12 {
                                let s = amount[k].max(0.0) / out;
                                for e in edge_flux.iter_mut() {
                                    if e.0 == k {
                                        e.2 *= s;
                                    }
                                }
                                scaled = true;
                            }
                        }
                    }
                    if !scaled {
                        break;
                    }
                }

                let mut delta = vec![0.0; n_n + 1];
                for &(a, b, f) in &edge_flux {
                    delta[a] -= f;
                    delta[b] += f;
                }
                for k in 0..=n_n {
                    amount[k] += delta[k];
                }

                *self.cells[ci]
                    .intracellular
                    .entry(id.clone())
                    .or_insert(0.0) = amount[0];
                for ni in 0..n_n {
                    *self.cells[ci].neurites[ni]
                        .intracellular
                        .entry(id.clone())
                        .or_insert(0.0) = amount[ni + 1];
                }
            }
        }

        // degradation
        for c in self.cells.iter_mut() {
            for v in c.intracellular.values_mut() {
                *v *= 1.0 - DEGRAD * dt;
            }
            for n in c.neurites.iter_mut() {
                for v in n.intracellular.values_mut() {
                    *v *= 1.0 - DEGRAD * dt;
                }
            }
        }
    }

    // ─── synapses ────────────────────────────────────────────────

    /// Form a synapse whenever an axon tip is within `dist` of another cell
    /// (soma or any neurite point). Target points are gathered once per step
    /// and synapses are deduplicated per (pre, post) pair.
    /// BioDynaMo `core/interaction_force.h` port (Verlet-free, linear falloff):
    /// mechanical repulsion pushing `p` away from other somas/neurites to spread
    /// the rind — same formula as BioDynaMo `InteractionForce`:
    pub fn crowding_force(&self, p: Vec3, cutoff: f64, strength: f64, skip_cell: usize) -> Vec3 {
        // Spatial-hash lookup (O(local density)) instead of O(N²).
        let mut grid = UniformGrid::new(cutoff.max(1.0));
        let mut points: Vec<Vec3> = Vec::new();
        for (i, c) in self.cells.iter().enumerate() {
            if i == skip_cell {
                continue;
            }
            let idx = points.len();
            points.push(c.soma.pos);
            grid.insert(idx, c.soma.pos);
            for neur in &c.neurites {
                for &pt in &neur.points {
                    let idx = points.len();
                    points.push(pt);
                    grid.insert(idx, pt);
                }
            }
        }
        let mut f = Vec3::ZERO;
        for i in grid.query(p, cutoff) {
            let q = points[i];
            let d = q.dist(p);
            if d > 1e-9 && d < cutoff {
                f = f + (p - q).normalized() * (strength * (1.0 - d / cutoff));
            }
        }
        f
    }

    pub fn form_synapses(&mut self, dist: f64) {
        // Fast path: no axon tips anywhere → nothing can be wired. This skips
        // the per-step contact-grid rebuild that dominated cell-only sims
        // (e.g. the BioDynaMo soma-clustering benchmark, which never forms
        // a synapse yet spent ~65% of each step here).
        if !self
            .cells
            .iter()
            .any(|c| c.neurites.iter().any(|n| n.is_axon))
        {
            return;
        }
        // Cache the contact grid across steps whose geometry is unchanged.
        let stale = self
            .contact_cache
            .as_ref()
            .map(|(d, _, _)| *d != dist)
            .unwrap_or(true);
        if stale || self.contact_cache.is_none() {
            let (grid, points) = build_contact_grid(&self.cells, dist.max(1.0));
            self.contact_cache = Some((dist, grid, points));
        }
        let n = self.cells.len();
        let (_, grid, points) = self.contact_cache.as_ref().unwrap();
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for a in 0..n {
            let a_id = self.cells[a].id;
            let kind = self.cells[a].cell_type.synapse_kind();
            let pre_t = self.cells[a].cell_type.as_usize();
            let tips: Vec<Vec3> = self.cells[a]
                .neurites
                .iter()
                .filter(|neur| neur.is_axon)
                .map(|neur| neur.tip())
                .collect();
            let mut nearest: Vec<(usize, f64)> = nearest_posts(&grid, &points, &tips, dist)
                .into_iter()
                .collect();
            nearest.sort_by_key(|&(b, _)| b); // deterministic RNG order
            for (b, min_d) in nearest {
                if b == a {
                    continue;
                }
                let post_t = self.cells[b].cell_type.as_usize();
                let conn_p = match &self.recognition {
                    Some(rec) => {
                        if !rec.allowed[pre_t][post_t] {
                            continue;
                        }
                        rec.prob[pre_t][post_t]
                    }
                    None => 1.0,
                };
                let b_id = self.cells[b].id;
                if seen.contains(&(a_id, b_id)) {
                    continue;
                }
                if min_d < dist {
                    // Learned partner gating: with probability < 1, wire up
                    // only a fraction of the close-by allowed pairs (the
                    // data-driven transition rule from `learned_from_edges`).
                    if conn_p < 1.0 && self.rng.gen::<f64>() >= conn_p {
                        continue;
                    }
                    // Weight is decoupled from geometry — real VNC corr(dist,w)=0.008
                    // Sample heavy-tailed lognormal (median 2, mean ~7) to match real.
                    let weight = sample_lognormal(&mut self.rng, 0.693, 1.59);
                    seen.insert((a_id, b_id));
                    self.synapses.push(Synapse {
                        pre: a_id,
                        post: b_id,
                        kind,
                        weight,
                    });
                }
            }
        }
    }

    /// Partner-recognition wiring keyed by an arbitrary per-neuron tag
    /// ([`TagRecognition`]): candidates are gated by the tag→tag connect rate
    /// and distance, while the sign still comes from the presynaptic
    /// [`CellType`] (DNA → neurotransmitter → sign).
    pub fn form_synapses_tagged(&mut self, dist: f64, rec: &TagRecognition) {
        let n = self.cells.len();
        let (grid, points) = build_contact_grid(&self.cells, dist.max(1.0));
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for a in 0..n {
            let a_id = self.cells[a].id;
            let kind = self.cells[a].cell_type.synapse_kind();
            let pre_tag = self.cells[a].tag.clone();
            let tips: Vec<Vec3> = self.cells[a]
                .neurites
                .iter()
                .filter(|neur| neur.is_axon)
                .map(|neur| neur.tip())
                .collect();
            let mut nearest: Vec<(usize, f64)> = nearest_posts(&grid, &points, &tips, dist)
                .into_iter()
                .collect();
            nearest.sort_by_key(|&(b, _)| b); // deterministic RNG order
            for (b, min_d) in nearest {
                if b == a {
                    continue;
                }
                let post_tag = &self.cells[b].tag;
                let key = (pre_tag.clone(), post_tag.clone());
                if !rec.allowed.get(&key).copied().unwrap_or(false) {
                    continue;
                }
                let conn_p = rec.prob.get(&key).copied().unwrap_or(0.0);
                let b_id = self.cells[b].id;
                if seen.contains(&(a_id, b_id)) {
                    continue;
                }
                if min_d < dist {
                    if conn_p < 1.0 && self.rng.gen::<f64>() >= conn_p {
                        continue;
                    }
                    let weight = sample_lognormal(&mut self.rng, 0.693, 1.59);
                    seen.insert((a_id, b_id));
                    self.synapses.push(Synapse {
                        pre: a_id,
                        post: b_id,
                        kind,
                        weight,
                    });
                }
            }
        }
    }

    /// NETMORPH synapse-formation model (`synapse_formation_model.cc`):
    /// candidates are gated by a per-(pre,post)-type maximum fibre distance
    /// (`D_synmax`), and each synapse's sign is drawn from the pre×post
    /// receptor compatibility matrix (`P_receptor`) — so the sign is a property
    /// of the synapse, not of the presynaptic cell alone. Ignores
    /// [`Self::recognition`] (the model carries its own compatibility).
    pub fn form_synapses_netmorph(&mut self, model: &SynapseFormationModel) {
        let n = self.cells.len();
        let max_dist = model
            .max_fibre_distance
            .iter()
            .flatten()
            .copied()
            .fold(0.0f64, f64::max)
            .max(1.0);
        let (grid, points) = build_contact_grid(&self.cells, max_dist);
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for a in 0..n {
            let a_id = self.cells[a].id;
            let pre_t = self.cells[a].cell_type.as_usize();
            let tips: Vec<Vec3> = self.cells[a]
                .neurites
                .iter()
                .filter(|neur| neur.is_axon)
                .map(|neur| neur.tip())
                .collect();
            let mut nearest: Vec<(usize, f64)> = nearest_posts(&grid, &points, &tips, max_dist)
                .into_iter()
                .collect();
            nearest.sort_by_key(|&(b, _)| b); // deterministic RNG order
            for (b, min_d) in nearest {
                if model.no_autapses && b == a {
                    continue;
                }
                let post_t = self.cells[b].cell_type.as_usize();
                let max_d = model.max_fibre_distance[pre_t][post_t];
                if max_d <= 0.0 {
                    continue;
                }
                let b_id = self.cells[b].id;
                if seen.contains(&(a_id, b_id)) {
                    continue;
                }
                if min_d <= max_d {
                    let kind = match model.synapse_kind(pre_t, post_t, &mut self.rng) {
                        Some(k) => k,
                        None => continue,
                    };
                    let weight = sample_lognormal(&mut self.rng, 0.693, 1.59);
                    seen.insert((a_id, b_id));
                    self.synapses.push(Synapse {
                        pre: a_id,
                        post: b_id,
                        kind,
                        weight,
                    });
                }
            }
        }
    }

    /// Place boutons on axons and spines on dendrites, one every `interval`
    /// units along each neurite (CX3D `makeBoutons` / `makeSpines`), each
    /// protruding `length` outward in a random radial direction.
    pub fn make_excrescences(&mut self, interval: f64, length: f64) {
        // Continue the world's RNG stream — a fixed seed would make every
        // simulation place identical boutons/spines regardless of `seed`.
        let mut rng = self.rng.clone();
        for c in self.cells.iter_mut() {
            for neur in c.neurites.iter_mut() {
                let kind = if neur.is_axon {
                    ExcrescenceKind::Bouton
                } else {
                    ExcrescenceKind::Spine
                };
                place_excrescences(neur, kind, interval, length, &mut rng);
            }
        }
        self.rng = rng;
    }

    /// CX3D `synapseBetweenExistingBS`: a free bouton on an axon (pre) connects
    /// to a free spine on a dendrite (post) when they are close enough
    /// (distance < bouton.length + spine.length + tol) **and** pointing toward
    /// each other, subject to the recognition/probability rule. This replaces
    /// the soma-distance heuristic with real contact geometry.
    pub fn form_synapses_by_contact(&mut self, tol: f64) {
        #[derive(Clone, Copy)]
        struct Ex {
            cell: usize,
            pos: Vec3,
            dir: Vec3,
            length: f64,
        }
        let mut boutons: Vec<Ex> = vec![];
        let mut spines: Vec<Ex> = vec![];
        for (ci, c) in self.cells.iter().enumerate() {
            for neur in &c.neurites {
                for e in &neur.excrescences {
                    let ex = Ex {
                        cell: ci,
                        pos: e.pos,
                        dir: e.dir,
                        length: e.length,
                    };
                    match (neur.is_axon, e.kind) {
                        (true, ExcrescenceKind::Bouton) => boutons.push(ex),
                        (false, ExcrescenceKind::Spine) => spines.push(ex),
                        _ => {}
                    }
                }
            }
        }
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for b in &boutons {
            let pre_t = self.cells[b.cell].cell_type.as_usize();
            let kind = self.cells[b.cell].cell_type.synapse_kind();
            for s in &spines {
                if b.cell == s.cell {
                    continue;
                }
                let post_t = self.cells[s.cell].cell_type.as_usize();
                let (conn_p, mfd, soft) = match &self.recognition {
                    Some(rec) => {
                        if !rec.allowed[pre_t][post_t] {
                            continue;
                        }
                        (rec.prob[pre_t][post_t], rec.max_dist[pre_t][post_t], true)
                    }
                    None => (1.0, tol, false),
                };
                if seen.contains(&(b.cell, s.cell)) {
                    continue;
                }
                let oo = s.pos - b.pos;
                let d = oo.len();
                // NETMORPH: the type-specific maximum fibre distance replaces a
                // single global contact tolerance.
                let reach = b.length + s.length + mfd;
                if d > reach {
                    continue;
                }
                let oo_hat = oo * (1.0 / d.max(1e-12));
                // co-directional: bouton protrudes toward the spine, and the
                // spine protrudes toward the bouton.
                if oo_hat.dot(b.dir) <= 0.0 || oo_hat.dot(s.dir) >= 0.0 {
                    continue;
                }
                // NETMORPH "crossing line-piece" candidate likelihood: closer
                // contacts are more likely to wire up (only for stochastic
                // rules; built-in rules with prob == 1.0 stay deterministic).
                let mut p = conn_p;
                if soft && p < 1.0 {
                    p *= (1.0 - d / reach).clamp(0.0, 1.0);
                }
                if p < 1.0 && self.rng.gen::<f64>() >= p {
                    continue;
                }
                let weight = sample_lognormal(&mut self.rng, 0.693, 1.59);
                seen.insert((b.cell, s.cell));
                self.synapses.push(Synapse {
                    pre: b.cell,
                    post: s.cell,
                    kind,
                    weight,
                });
            }
        }
    }

    /// [`form_synapses_by_contact`] but with the partner gate keyed by a
    /// per-neuron tag ([`TagRecognition`]) instead of the 2–3 [`CellType`]s.
    /// A free bouton on an axon connects to a free spine on a dendrite when
    /// they are close enough and pointing toward each other, subject to the
    /// learned tag→tag connect rate.
    pub fn form_synapses_by_contact_tagged(&mut self, tol: f64, rec: &TagRecognition) {
        #[derive(Clone, Copy)]
        struct Ex {
            cell: usize,
            pos: Vec3,
            dir: Vec3,
            length: f64,
        }
        let mut boutons: Vec<Ex> = vec![];
        let mut spines: Vec<Ex> = vec![];
        for (ci, c) in self.cells.iter().enumerate() {
            for neur in &c.neurites {
                for e in &neur.excrescences {
                    let ex = Ex {
                        cell: ci,
                        pos: e.pos,
                        dir: e.dir,
                        length: e.length,
                    };
                    match (neur.is_axon, e.kind) {
                        (true, ExcrescenceKind::Bouton) => boutons.push(ex),
                        (false, ExcrescenceKind::Spine) => spines.push(ex),
                        _ => {}
                    }
                }
            }
        }
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for b in &boutons {
            let kind = self.cells[b.cell].cell_type.synapse_kind();
            let pre_tag = self.cells[b.cell].tag.clone();
            for s in &spines {
                if b.cell == s.cell {
                    continue;
                }
                let post_tag = self.cells[s.cell].tag.clone();
                let key = (pre_tag.clone(), post_tag.clone());
                if !rec.allowed.get(&key).copied().unwrap_or(false) {
                    continue;
                }
                let conn_p = rec.prob.get(&key).copied().unwrap_or(0.0);
                if seen.contains(&(b.cell, s.cell)) {
                    continue;
                }
                let oo = s.pos - b.pos;
                let d = oo.len();
                let reach = b.length + s.length + tol;
                if d > reach {
                    continue;
                }
                let oo_hat = oo * (1.0 / d.max(1e-12));
                // co-directional: bouton protrudes toward the spine, and the
                // spine protrudes toward the bouton.
                if oo_hat.dot(b.dir) <= 0.0 || oo_hat.dot(s.dir) >= 0.0 {
                    continue;
                }
                // closer contacts are more likely to wire up
                let mut p = conn_p;
                if p < 1.0 {
                    p *= (1.0 - d / reach).clamp(0.0, 1.0);
                }
                if p < 1.0 && self.rng.gen::<f64>() >= p {
                    continue;
                }
                let weight = sample_lognormal(&mut self.rng, 0.693, 1.59);
                seen.insert((b.cell, s.cell));
                self.synapses.push(Synapse {
                    pre: b.cell,
                    post: s.cell,
                    kind,
                    weight,
                });
            }
        }
    }

    /// Directed wiring rule: connect every cell of fate `(from_role, from_sub)`
    /// to every cell of fate `(to_role, to_sub)` (e.g. ring coupling
    /// stance_i → stance_{i+1}). Sign comes from the presynaptic cell type;
    /// `weight` is the rule's synaptic strength.
    pub fn connect_fates(
        &mut self,
        from_role: i32,
        from_sub: i32,
        to_role: i32,
        to_sub: i32,
        weight: f64,
    ) -> usize {
        let src: Vec<usize> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == from_role && c.sub_role == from_sub)
            .map(|(i, _)| i)
            .collect();
        let dst: Vec<usize> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == to_role && c.sub_role == to_sub)
            .map(|(i, _)| i)
            .collect();
        let mut n = 0;
        for &a in &src {
            let kind = self.cells[a].cell_type.synapse_kind();
            for &b in &dst {
                if a == b {
                    continue;
                }
                self.synapses.push(Synapse {
                    pre: a,
                    post: b,
                    kind,
                    weight,
                });
                n += 1;
            }
        }
        n
    }

    /// Extract the grown network as nodes + directed edges. This is the
    /// bridge from development (this crate) to the spiking simulators
    /// (`pillars` / `pillars-connectome`): topology, sign and strength all
    /// come from the growth process, not from hand-set weights.
    pub fn to_graph(&self) -> GrownGraph {
        GrownGraph {
            nodes: self
                .cells
                .iter()
                .filter(|c| c.alive)
                .map(|c| GrownNode {
                    id: c.id,
                    cell_type: c.cell_type,
                    pos: c.soma.pos,
                })
                .collect(),
            edges: self
                .synapses
                .iter()
                .map(|s| GrownEdge {
                    pre: s.pre,
                    post: s.post,
                    kind: s.kind,
                    weight: s.weight,
                })
                .collect(),
        }
    }

    // ─── rng ─────────────────────────────────────────────────────

    pub fn rand(&mut self) -> f64 {
        self.rng.gen()
    }

    pub fn random_vec(&mut self, scale: f64) -> Vec3 {
        Vec3::new(
            self.rng.gen_range(-scale..scale),
            self.rng.gen_range(-scale..scale),
            self.rng.gen_range(-scale..scale),
        )
    }

    // ─── main loop ───────────────────────────────────────────────

    pub fn step(&mut self) {
        self.diffuse_substances();
        self.run_cell_modules();
        self.process_divisions();
        self.run_soma_modules();
        self.run_neurite_modules();
        self.process_prunes();
        self.diffuse_intracellular();
        let d = self.synapse_dist;
        self.form_synapses(d);
        self.time += self.dt;
    }

    pub fn run_cell_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            let mut mods = std::mem::take(&mut self.cells[ci].modules);
            for mut m in mods.drain(..) {
                m.run(self, ci);
                self.cells[ci].modules.push(m);
            }
        }
    }

    pub fn run_soma_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            let mut mods = std::mem::take(&mut self.cells[ci].soma.modules);
            for mut m in mods.drain(..) {
                m.run(self, Host::Soma(ci));
                self.cells[ci].soma.modules.push(m);
            }
        }
    }

    pub fn run_neurite_modules(&mut self) {
        // Snapshot copy_on_branch modules so branching during `run` can copy
        // them (the live list is emptied by `std::mem::take` below).
        let mut snap: HashMap<(usize, usize), Vec<Box<dyn Module>>> = HashMap::new();
        for (ci, c) in self.cells.iter().enumerate() {
            for (ni, neur) in c.neurites.iter().enumerate() {
                snap.insert(
                    (ci, ni),
                    neur.modules
                        .iter()
                        .filter(|m| m.copy_on_branch())
                        .map(|m| m.boxed_copy())
                        .collect(),
                );
            }
        }
        self.branch_snapshot = snap;

        // Neurites born during this step (bifurcation daughters) must not run
        // their phase-2 allocation until the next step — NETMORPH gives a new
        // terminal its initial length at branch time and elongates it later.
        let base_counts: Vec<usize> = self.cells.iter().map(|c| c.neurites.len()).collect();

        // Phase 1: run every pre-existing module (speed updates, branching).
        let n = self.cells.len();
        for ci in 0..n {
            let nneur = base_counts[ci];
            for ni in 0..nneur {
                let mut mods = std::mem::take(&mut self.cells[ci].neurites[ni].modules);
                for mut m in mods.drain(..) {
                    m.run(self, Host::Neurite { cell: ci, idx: ni });
                    self.cells[ci].neurites[ni].modules.push(m);
                }
            }
        }

        // Phase 2: finish_step (allocation) with all speeds final, so the
        // resource split is order-independent and conserves the arbor total.
        for ci in 0..n {
            let nneur = base_counts[ci];
            for ni in 0..nneur {
                let mut mods = std::mem::take(&mut self.cells[ci].neurites[ni].modules);
                for mut m in mods.drain(..) {
                    m.finish_step(self, Host::Neurite { cell: ci, idx: ni });
                    self.cells[ci].neurites[ni].modules.push(m);
                }
            }
        }
        self.branch_snapshot.clear();
    }

    /// Fresh copies of the `copy_on_branch` modules for a neurite, preferring
    /// the step snapshot (which is the only place they exist during `run`).
    fn branch_copies(&self, cell: usize, idx: usize) -> Vec<Box<dyn Module>> {
        match self.branch_snapshot.get(&(cell, idx)) {
            Some(snap) => snap.iter().map(|m| m.boxed_copy()).collect(),
            None => self.cells[cell].neurites[idx]
                .modules
                .iter()
                .filter(|m| m.copy_on_branch())
                .map(|m| m.boxed_copy())
                .collect(),
        }
    }

    pub fn process_divisions(&mut self) {
        self.touch_geometry();
        let ids = std::mem::take(&mut self.pending_divisions);
        for id in ids {
            self.divide_cell(id);
        }
    }

    /// Remove neurites scheduled by [`World::request_prune`] (fully retracted
    /// growth cones). Run after the module loop so in-flight iterations never
    /// see a shifted neurite list.
    fn process_prunes(&mut self) {
        let prunes = std::mem::take(&mut self.pending_prunes);
        for (ci, ni) in prunes {
            if ci < self.cells.len() && ni < self.cells[ci].neurites.len() {
                self.cells[ci].neurites.remove(ni);
            }
        }
    }

    /// Split a cell into two: daughter gets a copy of every module flagged
    /// `copy_on_divide` (the self-construction instruction copy).
    fn divide_cell(&mut self, id: usize) {
        if self.cells.len() >= self.max_cells {
            return;
        }
        let offset = self.random_vec(self.grid.cell * 0.5);
        let (
            cell_type,
            role,
            tag,
            soma_pos,
            soma_radius,
            soma_mods,
            neurites,
            membrane,
            intracellular,
            cell_mods,
        ) = {
            let c = &self.cells[id];
            (
                c.cell_type,
                c.role,
                c.tag.clone(),
                c.soma.pos,
                c.soma.radius,
                c.soma
                    .modules
                    .iter()
                    .filter(|m| m.copy_on_divide())
                    .map(|m| m.boxed_copy())
                    .collect::<Vec<Box<dyn Module>>>(),
                c.neurites
                    .iter()
                    .map(|n| {
                        (
                            n.is_axon,
                            n.diameter,
                            n.points.clone(),
                            n.modules
                                .iter()
                                .filter(|m| m.copy_on_divide())
                                .map(|m| m.boxed_copy())
                                .collect::<Vec<Box<dyn Module>>>(),
                            n.elongation_speed,
                            n.is_terminal,
                            n.intracellular.clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
                c.membrane.clone(),
                c.intracellular.clone(),
                c.modules
                    .iter()
                    .filter(|m| m.copy_on_divide())
                    .map(|m| m.boxed_copy())
                    .collect::<Vec<Box<dyn CellModule>>>(),
            )
        };
        // Halve the parent volume (mitosis).
        let r = self.cells[id].soma.radius;
        self.cells[id].soma.radius = (r.powi(3) * 0.5).powf(1.0 / 3.0);

        let daughter = Cell {
            id: self.cells.len(),
            cell_type,
            role,
            // Asymmetric division: parent keeps sub_role 0 (stance), the
            // daughter becomes sub_role 1 (swing).
            sub_role: 1,
            tag,
            soma: Soma {
                pos: soma_pos + offset,
                radius: (soma_radius.powi(3) * 0.5).powf(1.0 / 3.0),
                modules: soma_mods,
            },
            neurites: neurites
                .into_iter()
                .map(
                    |(
                        is_axon,
                        diameter,
                        points,
                        modules,
                        elongation_speed,
                        is_terminal,
                        intracellular,
                    )| {
                        Neurite {
                            is_axon,
                            diameter,
                            points,
                            modules,
                            excrescences: vec![],
                            branch_order: 0,
                            parent: None,
                            elongation_speed,
                            is_terminal,
                            intracellular,
                        }
                    },
                )
                .collect(),
            modules: cell_mods,
            membrane,
            intracellular,
            alive: true,
            delta: 0.0,
            notch: 0.0,
        };
        self.cells.push(daughter);
        self.par_cell_modules.push(vec![]);
    }

    /// Attach a parallel-safe cell module (BioDynaMo `Behavior`). Run via
    /// [`World::run_cell_modules_par`].
    pub fn add_cell_par_module(&mut self, cell: usize, m: Box<dyn crate::module::ParCellModule>) {
        self.par_cell_modules[cell].push(m);
    }

    /// Parallel cell-module loop (BioDynaMo `Behavior` semantics): cells are
    /// split across rayon threads, each module gets its own `&mut Cell` plus a
    /// read-only [`ParWorldView`]; any shared-state mutation (secretion,
    /// division) is deferred via [`DeferredOp`] and applied serially after the
    /// loop — race-free by construction.
    pub fn run_cell_modules_par(&mut self) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }
        // Detach cells + modules so rayon can own the mutable slices while the
        // world (substances, grid, bounds) is only read.
        let mut cells = std::mem::take(&mut self.cells);
        let mut par = std::mem::take(&mut self.par_cell_modules);
        for _ in par.len()..n {
            par.push(vec![]);
        }
        par.truncate(n);

        // The parallel loop addresses cells through raw pointers derived from
        // the base address (each rayon block owns a disjoint `lo..hi` range),
        // so `Cell`'s non-Send `Box<dyn Module>` fields are never touched by
        // rayon — no detach/restore needed.
        let substances = std::mem::take(&mut self.substances);
        let ids: Vec<&str> = substances.iter().map(|s| s.id.as_str()).collect();
        let fields: Vec<&[f64]> = substances.iter().map(|s| s.field.as_slice()).collect();
        let view = crate::module::ParWorldView {
            substance_ids: &ids,
            fields: &fields,
            grid: self.grid,
            bounds: self.bounds,
            dt: self.dt,
            time: self.time,
        };

        let mut ops_flat: Vec<Vec<crate::module::DeferredOp>> = Vec::with_capacity(n);
        ops_flat.resize_with(n, Vec::new);

        // Cells hold `Box<dyn Module>` (not Send), so rayon cannot own the
        // slices directly. Index-parallelism with unique per-index &mut gives
        // the same guarantees: rayon gives each closure a distinct `ci`, and
        // `ptr.add(ci)` is the only live mutable reference to that cell (the
        // `Held` detach above removed the non-Send module vectors, so the
        // cells carry only plain data while the modules run).
        // `*mut` is not Send; wrap in a newtype with a documented unsafe
        // impl — the loop only ever dereferences index `ci` exactly once.
        // `*mut T` is neither Send nor Sync; carry the base address as a plain
        // `usize` (Send+Sync) and re-derive the pointer inside each task.
        // Process cells in coarse blocks (one per rayon worker) instead of
        // one task per cell — per-cell work is ~100 ns, so fine-grained
        // splitting pays only synchronization overhead (BDM does the same:
        // threads own contiguous agent blocks).
        let n_threads = rayon::current_num_threads().max(1);
        let block = (n + n_threads - 1) / n_threads;
        let cell_base = cells.as_mut_ptr() as usize;
        let par_base = par.as_mut_ptr() as usize;
        let ops_base = ops_flat.as_mut_ptr() as usize;
        // One RNG per rayon worker, seeded from the world RNG stream, owned
        // for the whole block (no per-cell thread_local overhead).
        let rng_seed = self.rng.gen::<u64>();
        (0..n_threads).into_par_iter().for_each(move |tb| {
            let lo = tb * block;
            let hi = ((tb + 1) * block).min(n);
            let mut rng =
                StdRng::seed_from_u64(rng_seed ^ (tb as u64).wrapping_mul(0x9E3779B97F4A7C15));
            for ci in lo..hi {
                // SAFETY: blocks are disjoint; cell/par/ops are not
                // aliased elsewhere during the loop.
                let cell = unsafe { &mut *(cell_base as *mut crate::cell::Cell).add(ci) };
                let mods = unsafe {
                    &mut *(par_base as *mut Vec<Box<dyn crate::module::ParCellModule>>).add(ci)
                };
                let ops_slot =
                    unsafe { &mut *(ops_base as *mut Vec<crate::module::DeferredOp>).add(ci) };
                let mut ops = std::mem::take(ops_slot);
                for m in mods.iter() {
                    m.run(cell, &view, &mut ops, &mut rng);
                }
                *ops_slot = ops;
            }
        });

        self.substances = substances;
        self.cells = cells;
        self.par_cell_modules = par;

        // Apply deferred side effects serially (deterministic order).
        for ops in ops_flat {
            for op in ops {
                match op {
                    crate::module::DeferredOp::Secretion {
                        substance_idx,
                        grid_idx,
                        amount,
                    } => {
                        if let Some(s) = self.substances.get_mut(substance_idx) {
                            s.field[grid_idx] += amount;
                        }
                    }
                    crate::module::DeferredOp::Division(cell) => {
                        if !self.pending_divisions.contains(&cell) {
                            self.pending_divisions.push(cell);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtificialGradient, ChemoAttraction, Secretor, SimpleCellCycle, Substance};

    fn world() -> World {
        let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
        let bounds = Bounds {
            min: Vec3::new(-80.0, -80.0, -80.0),
            max: Vec3::new(80.0, 80.0, 80.0),
        };
        let mut w = World::new(grid, 1.0, bounds, 0);
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
    fn division_produces_many_cells() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_cell_module(c, Box::new(SimpleCellCycle::new(300.0, 12.0)));
        for _ in 0..30 {
            w.step();
            if w.cells.len() >= 16 {
                break;
            }
        }
        assert!(
            w.cells.len() > 1,
            "expected division, got {} cells",
            w.cells.len()
        );
    }

    #[test]
    fn neurite_grows_toward_gradient() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 0.5);
        let mut chemo = ChemoAttraction::new("A");
        chemo.noise_weight = 0.05; // deterministic-ish: gradient dominates
        w.add_neurite_module(c, n, Box::new(chemo));
        let start = w.cells[c].neurites[n].tip();
        for _ in 0..100 {
            w.step();
        }
        let tip = w.cells[c].neurites[n].tip();
        assert!(
            tip.z > start.z,
            "growth cone should move toward +z gradient (start {start:?}, end {tip:?})"
        );
    }

    #[test]
    fn secretion_raises_field() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        w.add_soma_module(c, Box::new(Secretor::new("A", 1.0)));
        let before = w.concentration("A", Vec3::ZERO);
        for _ in 0..10 {
            w.step();
        }
        let after = w.concentration("A", Vec3::ZERO);
        assert!(
            after > before,
            "secretion should raise field: {before} -> {after}"
        );
    }

    #[test]
    fn recognition_gates_specific_wiring() {
        let mut w = world();
        let g0 = w.add_cell(Vec3::new(0.0, 0.0, 0.0), CellType::Inhibitory, 4.0);
        let g1 = w.add_cell(Vec3::new(5.0, 0.0, 0.0), CellType::Inhibitory, 4.0);
        let e0 = w.add_cell(Vec3::new(0.0, 5.0, 0.0), CellType::Excitatory, 4.0);
        for c in [g0, g1, e0] {
            w.new_neurite(c, true, 0.5);
        }
        w.set_recognition(Recognition::inhibitory_only());
        w.form_synapses(20.0);
        let g = w.to_graph();
        let pair = g.edges.iter().any(|e| e.pre == g0 && e.post == g1)
            && g.edges.iter().any(|e| e.pre == g1 && e.post == g0);
        let cross = g
            .edges
            .iter()
            .filter(|e| e.pre == e0 || e.post == e0)
            .count();
        assert!(pair, "ожидалась взаимная GABA-пара");
        assert_eq!(
            cross, 0,
            "при inhibitory_only не должно быть связей с возбуждающим"
        );
    }

    #[test]
    fn crowding_force_repels() {
        let mut w = world();
        w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let f = w.crowding_force(Vec3::new(2.0, 0.0, 0.0), 10.0, 1.0, usize::MAX);
        assert!(f.x > 0.0, "сила должна толкать от сомы: {f:?}");
    }

    #[test]
    fn learned_recognition_reproduces_type_rates() {
        let types = [
            CellType::Excitatory,
            CellType::Excitatory,
            CellType::Excitatory,
            CellType::Inhibitory,
            CellType::Inhibitory,
        ];
        let edges = [(0usize, 1usize), (3, 4), (4, 3)];
        let rec = Recognition::learned_from_edges(&types, &edges);
        // exc→exc: 1 edge / (3*3 = 9) pairs
        assert!((rec.prob[0][0] - 1.0 / 9.0).abs() < 1e-9);
        // inh→inh: 2 edges / (2*2 = 4) pairs
        assert!((rec.prob[1][1] - 0.5).abs() < 1e-9);
        // cross-type pairs had no edges → disallowed, zero probability
        assert!(!rec.allowed[0][1] && !rec.allowed[1][0]);
        assert_eq!(rec.prob[0][1], 0.0);
        assert_eq!(rec.prob[1][0], 0.0);
    }

    #[test]
    fn contact_synapse_requires_meeting_and_orientation() {
        let mut w = world();
        let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let b = w.add_cell(Vec3::new(0.0, 2.0, 0.0), CellType::Excitatory, 4.0);
        let na = w.new_neurite(a, true, 1.0); // axon
        let nb = w.new_neurite(b, false, 1.0); // dendrite
                                               // parallel neurites, 2 units apart
        for _ in 0..10 {
            w.extend_neurite(a, na, Vec3::new(1.0, 0.0, 0.0));
            w.extend_neurite(b, nb, Vec3::new(1.0, 0.0, 0.0));
        }
        // facing excrescences → synapse
        w.cells[a].neurites[na].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Bouton,
            pos: Vec3::new(5.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 1.0, 0.0),
            length: 2.0,
        });
        w.cells[b].neurites[nb].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Spine,
            pos: Vec3::new(5.0, 2.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
            length: 2.0,
        });
        w.form_synapses_by_contact(0.5);
        let g = w.to_graph();
        assert!(
            g.edges.iter().any(|e| e.pre == a && e.post == b),
            "facing bouton/spine должны образовать синапс"
        );

        // spine pointed away → no synapse
        w.synapses.clear();
        w.cells[b].neurites[nb].excrescences[0].dir = Vec3::new(0.0, 1.0, 0.0);
        w.form_synapses_by_contact(0.5);
        let g2 = w.to_graph();
        assert!(
            !g2.edges.iter().any(|e| e.pre == a && e.post == b),
            "spine, смотрящий в сторону, не должен дать синапс"
        );
    }

    #[test]
    fn bifurcate_neurite_splits_diameter_by_rall() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 2.0);
        for _ in 0..5 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        let d_parent = w.cells[c].neurites[n].diameter;
        w.bifurcate_neurite(c, n, std::f64::consts::FRAC_PI_3, 1.5);
        let neurites = &w.cells[c].neurites;
        assert_eq!(neurites.len(), 3, "родитель + 2 дочки");
        let d1 = neurites[1].diameter;
        let d2 = neurites[2].diameter;
        let lhs = d1.powf(1.5) + d2.powf(1.5);
        let rhs = d_parent.powf(1.5);
        assert!((lhs - rhs).abs() < 1e-9, "Rall: {lhs} ≈ {rhs}");
        assert_eq!(neurites[1].branch_order, 1);
        assert_eq!(neurites[2].branch_order, 1);
        assert!(
            d1 < d_parent && d2 < d_parent,
            "дочки тоньше родителя: {d1},{d2} < {d_parent}"
        );
    }

    #[test]
    fn growth_bifurcation_splits_speed_into_daughters() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 2.0);
        w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        w.cells[c].neurites[n].elongation_speed = 2.0;
        let mut model = crate::BalancedForces::netmorph_defaults();
        model.parallelogram_std = 0.0; // fixed 90° for determinism
                                       // ERI noise off → daughters split speed 2.0 → 1.0 each.
        w.bifurcate_neurite_growth(c, n, 1.5, 2.0, 0.0, &model);
        let neurites = &w.cells[c].neurites;
        assert_eq!(neurites.len(), 3, "родитель + 2 дочки");
        assert!((neurites[1].elongation_speed - 1.0).abs() < 1e-9);
        assert!((neurites[2].elongation_speed - 1.0).abs() < 1e-9);
        // initial length = step * speed = 2.0 * 1.0 = 2.0
        let l1 = neurites[1].points[0].dist(neurites[1].points[1]);
        let l2 = neurites[2].points[0].dist(neurites[2].points[1]);
        assert!((l1 - 2.0).abs() < 1e-9 && (l2 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn branch_on_diameter_grows_arbor() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 4.0);
        w.add_neurite_module(
            c,
            n,
            Box::new(
                crate::BranchOnDiameter::new(0.5, 2.0)
                    .rall_power(1.5)
                    .speed(1.0),
            ),
        );
        for _ in 0..30 {
            w.step();
        }
        assert!(
            w.cells[c].neurites.len() > 1,
            "должен разветвиться: {} нейритов",
            w.cells[c].neurites.len()
        );
        // diameters only shrink (taper + Rall)
        for neur in &w.cells[c].neurites {
            assert!(neur.diameter <= 4.0, "диаметр не растёт");
        }
    }

    #[test]
    fn van_pelt_branching_grows_and_self_limits() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 2.0);
        w.add_neurite_module(
            c,
            n,
            Box::new(
                crate::VanPeltBranching::new(100.0, 1.0, 0.0, 0.0)
                    .max_prob(1.0)
                    .speed(1.0),
            ),
        );
        w.step();
        // p is clamped to 1.0 → the growth cone always bifurcates once.
        let neurites = &w.cells[c].neurites;
        assert_eq!(neurites.len(), 3, "один шаг → родитель + 2 дочки");
        assert_eq!(neurites[1].branch_order, 1);
        assert_eq!(neurites[2].branch_order, 1);
        assert!(neurites[1].diameter > 0.0 && neurites[2].diameter > 0.0);
    }

    #[test]
    fn rall_diameters_parent_thicker_than_children() {
        let mut w = world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 6.0);
        let n = w.new_neurite(c, true, 3.0);
        for _ in 0..5 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        w.bifurcate_neurite(c, n, std::f64::consts::FRAC_PI_3, 1.5);
        // NETMORPH bottom-up: reassign diameters from sampled terminal
        // diameters + sampled Rall exponents.
        w.apply_rall_diameters(1.47, 0.3, 0.7, 0.3);
        let neurites = &w.cells[c].neurites;
        let dp = neurites[0].diameter;
        let d1 = neurites[1].diameter;
        let d2 = neurites[2].diameter;
        assert!(
            dp >= d1 && dp >= d2,
            "родитель не тоньше дочек: {dp} vs {d1},{d2}"
        );
        assert!(d1 > 0.0 && d2 > 0.0, "диаметры положительные");
    }

    #[test]
    fn tagged_contact_gates_by_tag() {
        let mut w = world();
        let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let b = w.add_cell(Vec3::new(0.0, 2.0, 0.0), CellType::Excitatory, 4.0);
        w.cells[a].tag = "T1".to_string();
        w.cells[b].tag = "T2".to_string();
        let na = w.new_neurite(a, true, 1.0);
        let nb = w.new_neurite(b, false, 1.0);
        w.cells[a].neurites[na].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Bouton,
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 1.0, 0.0),
            length: 2.0,
        });
        w.cells[b].neurites[nb].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Spine,
            pos: Vec3::new(0.0, 2.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
            length: 2.0,
        });

        // T1→T2 disallowed (no edges learned) → no synapse.
        let tags = vec!["T1".to_string(), "T2".to_string()];
        let off = TagRecognition::learned_from_edges(&tags, &[]);
        w.form_synapses_by_contact_tagged(0.5, &off);
        assert!(
            w.to_graph().edges.is_empty(),
            "запрещённый tag-переход не даёт синапс"
        );

        // T1→T2 allowed (one learned edge) → synapse forms.
        let on = TagRecognition::learned_from_edges(&tags, &[(0usize, 1usize)]);
        w.form_synapses_by_contact_tagged(0.5, &on);
        let g = w.to_graph();
        assert!(
            g.edges.iter().any(|e| e.pre == a && e.post == b),
            "разрешённый tag-переход на контакте даёт синапс"
        );
    }

    #[test]
    fn max_distance_is_type_specific() {
        let mut w = world();
        let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let b = w.add_cell(Vec3::new(0.0, 2.0, 0.0), CellType::Excitatory, 4.0);
        let na = w.new_neurite(a, true, 1.0);
        let nb = w.new_neurite(b, false, 1.0);
        w.cells[a].neurites[na].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Bouton,
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 1.0, 0.0),
            length: 0.1,
        });
        w.cells[b].neurites[nb].excrescences.push(Excrescence {
            kind: ExcrescenceKind::Spine,
            pos: Vec3::new(0.0, 2.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
            length: 0.1,
        });
        // exc→exc max distance too small → no synapse (gap 2.0 > 0.1+0.1+0.5)
        w.set_recognition(Recognition::all().with_max_dist([
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]));
        w.form_synapses_by_contact(10.0);
        let g = w.to_graph();
        assert!(g.edges.is_empty(), "короткий max_dist не даёт синапс");
        // widen the exc→exc bridge → synapse forms
        w.synapses.clear();
        w.set_recognition(Recognition::all().with_max_dist([
            [2.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]));
        w.form_synapses_by_contact(10.0);
        let g2 = w.to_graph();
        assert!(
            g2.edges.iter().any(|e| e.pre == a && e.post == b),
            "длинный max_dist даёт синапс"
        );
    }

    #[test]
    fn excrescence_placement_depends_on_world_seed() {
        let mk = |seed: u64| -> World {
            let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
            let bounds = Bounds {
                min: Vec3::ZERO,
                max: Vec3::new(400.0, 400.0, 400.0),
            };
            let mut w = World::new(grid, 1.0, bounds, seed);
            let c = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 6.0);
            let n = w.new_neurite(c, true, 2.0);
            for _ in 0..8 {
                w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
            }
            w.make_excrescences(1.0, 1.0);
            w
        };
        let w1 = mk(1);
        let w2 = mk(2);
        let pos = |w: &World| -> Vec<Vec3> {
            w.cells[0].neurites[0]
                .excrescences
                .iter()
                .map(|e| e.pos)
                .collect()
        };
        assert_ne!(
            pos(&w1),
            pos(&w2),
            "разные seed → разное размещение бутонов/шипиков"
        );
    }

    #[test]
    fn diffusion_respects_dirichlet_boundary() {
        let grid = Grid::new(4, 4, 4, 5.0, Vec3::ZERO);
        let bounds = Bounds {
            min: Vec3::ZERO,
            max: Vec3::new(20.0, 20.0, 20.0),
        };
        let mut w = World::new(grid, 1.0, bounds, 0);
        w.add_substance(
            Substance::new("D", 0.1, 0.0, &w.grid, vec![])
                .with_boundary(BoundaryCondition::Dirichlet(5.0)),
        );
        w.step();
        let s = w.substances.iter().find(|s| s.id == "D").unwrap();
        assert!(
            (s.field[w.grid.flat(0, 1, 1)] - 5.0).abs() < 1e-9,
            "граница зафиксирована"
        );
        assert!(
            (s.field[w.grid.flat(1, 1, 1)]).abs() < 1e-9,
            "внутренность остаётся нулевой"
        );
    }
}

/// Two unit directions ±angle/2 around `axis` (CX3D bifurcation angles).
fn bifurcate_directions(axis: Vec3, angle: f64) -> (Vec3, Vec3) {
    let perp = any_perpendicular(axis);
    let half = angle * 0.5;
    let (c, s) = (half.cos(), half.sin());
    let d1 = (axis * c + perp * s).normalized();
    let d2 = (axis * c - perp * s).normalized();
    (d1, d2)
}

/// Two unit directions deviating by `angle1` / `angle2` from `axis` on
/// opposite sides (NETMORPH balanced-forces branch angles).
fn bifurcate_directions_asym(axis: Vec3, angle1: f64, angle2: f64) -> (Vec3, Vec3) {
    let perp = any_perpendicular(axis);
    let d1 = (axis * angle1.cos() + perp * angle1.sin()).normalized();
    let d2 = (axis * angle2.cos() - perp * angle2.sin()).normalized();
    (d1, d2)
}

/// Any unit vector perpendicular to `axis`.
fn any_perpendicular(axis: Vec3) -> Vec3 {
    let v = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    (v - axis * v.dot(axis)).normalized()
}

/// Nearest interior index along one axis, or `None` when the grid is too thin
/// (dimension < 3) to have an interior cell.
fn interior_index(v: usize, n: usize) -> Option<usize> {
    if n < 3 {
        None
    } else {
        Some(v.clamp(1, n - 2))
    }
}

/// Build a spatial hash of every postsynaptic contact point (soma + all
/// neurite points), tagged by owning cell. Returns `(grid, points)` where
/// `points[i] = (position, cell_index)`.
fn build_contact_grid(cells: &[Cell], cell_size: f64) -> (UniformGrid, Vec<(Vec3, usize)>) {
    let mut grid = UniformGrid::new(cell_size);
    let mut points = Vec::new();
    for (b, c) in cells.iter().enumerate() {
        let idx = points.len();
        points.push((c.soma.pos, b));
        grid.insert(idx, c.soma.pos);
        for neur in &c.neurites {
            for &p in &neur.points {
                let idx = points.len();
                points.push((p, b));
                grid.insert(idx, p);
            }
        }
    }
    (grid, points)
}

/// Nearest contact distance from the pre cell's axon tips to each post cell
/// (spatial-hash lookup: O(tips · local density) instead of O(N²)).
fn nearest_posts(
    grid: &UniformGrid,
    points: &[(Vec3, usize)],
    tips: &[Vec3],
    dist: f64,
) -> HashMap<usize, f64> {
    let mut nearest: HashMap<usize, f64> = HashMap::new();
    for t in tips {
        for i in grid.query(*t, dist) {
            let (p, b) = points[i];
            let d = t.dist(p);
            let e = nearest.entry(b).or_insert(f64::INFINITY);
            if d < *e {
                *e = d;
            }
        }
    }
    nearest
}

/// Place excrescences every `interval` along a neurite polyline at random
/// radial directions (CX3D bouton/spine placement).
fn place_excrescences(
    neur: &mut Neurite,
    kind: ExcrescenceKind,
    interval: f64,
    length: f64,
    rng: &mut StdRng,
) {
    let total: f64 = neur.points.windows(2).map(|w| w[0].dist(w[1])).sum();
    let count = if total <= 1e-9 {
        if neur.points.is_empty() {
            0
        } else {
            1
        }
    } else {
        (total / interval).round().max(1.0) as usize
    };
    for _ in 0..count {
        let t = rng.gen::<f64>() * total;
        let (pos, axis) = point_along(&neur.points, t);
        let dir = random_radial(axis, rng);
        neur.excrescences.push(Excrescence {
            kind,
            pos,
            dir,
            length,
        });
    }
}

/// Point and local axis direction at arc-length `t` along a polyline.
fn point_along(points: &[Vec3], t: f64) -> (Vec3, Vec3) {
    if points.len() == 1 {
        return (points[0], Vec3::new(0.0, 0.0, 1.0));
    }
    let mut acc = 0.0;
    for w in points.windows(2) {
        let l = w[0].dist(w[1]);
        if l <= 1e-12 || t <= acc + l {
            let f = if l > 1e-12 {
                ((t - acc) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let axis = if l > 1e-12 {
                (w[1] - w[0]).normalized()
            } else {
                Vec3::new(0.0, 0.0, 1.0)
            };
            return (w[0] + (w[1] - w[0]) * f, axis);
        }
        acc += l;
    }
    let last = *points.last().unwrap();
    let axis = (last - points[points.len() - 2]).normalized();
    (last, axis)
}

/// Sample from a normal distribution via the Box–Muller transform (avoids
/// a `rand_distr` dependency).
/// Total polyline length of a neurite (sum of segment lengths).
fn neurite_length(n: &Neurite) -> f64 {
    n.points.windows(2).map(|w| (w[1] - w[0]).len()).sum()
}

fn sample_normal(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    mean + std * z
}

fn sample_lognormal(rng: &mut StdRng, mu: f64, sigma: f64) -> f64 {
    sample_normal(rng, mu, sigma).exp().max(0.1)
}

/// Random unit vector perpendicular to `axis` (a radial direction).
fn random_radial(axis: Vec3, rng: &mut StdRng) -> Vec3 {
    let v = Vec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );
    let p = v - axis * v.dot(axis);
    if p.len() < 1e-9 {
        let perp = if axis.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        (perp - axis * perp.dot(axis)).normalized()
    } else {
        p.normalized()
    }
}

#[cfg(test)]
mod cx3d_tests {
    use super::*;

    /// Conservative intracellular diffusion: total amount is conserved (up to
    /// degradation) and never goes negative — no runaway/NaN at coarse dt.
    #[test]
    fn intracellular_diffusion_conserves_and_stays_finite() {
        let grid = Grid::new(8, 8, 8, 50.0, Vec3::new(-200.0, -200.0, -200.0));
        let bounds = Bounds {
            min: Vec3::new(-200.0, -200.0, -200.0),
            max: Vec3::new(200.0, 200.0, 200.0),
        };
        let mut w = World::new(grid, 0.1, bounds, 0);
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        w.modify_intracellular(c, "tubulin", 10_000.0);
        let n = w.new_neurite(c, true, 1.0);
        w.extend_neurite(c, n, Vec3::new(0.0, 0.0, 10.0));

        // One step: tubulin should flow soma → neurite without loss or NaN.
        w.diffuse_intracellular();
        let soma = w.intracellular(c, "tubulin");
        let n_amt = *w.cells[c].neurites[n].intracellular.get("tubulin").unwrap();
        assert!(
            soma.is_finite() && n_amt.is_finite(),
            "NaN появился: {soma} {n_amt}"
        );
        assert!(soma < 10_000.0, "сома должна отдать тубулин");
        assert!(n_amt > 0.0, "нейрит должен получить тубулин");
        assert!(soma >= 0.0 && n_amt >= 0.0, "нет отрицательных количеств");

        // Long run with a growth cone consuming: stays finite, soma keeps
        // its pool (secretion) — the fuel-limited demo must not explode.
        for _ in 0..2000 {
            w.modify_intracellular(c, "tubulin", 60.0);
            let conc = w.neurite_intracellular(c, n, "tubulin");
            let speed = (conc * 5000.0).min(100.0);
            if speed > 0.0 {
                w.extend_neurite(c, n, Vec3::new(0.0, 0.0, speed * w.dt));
            }
            w.modify_neurite_intracellular(c, n, "tubulin", -conc * 1.0);
            w.diffuse_intracellular();
        }
        let soma = w.intracellular(c, "tubulin");
        assert!(soma.is_finite() && soma >= 0.0, "сома: {soma}");
    }
}
