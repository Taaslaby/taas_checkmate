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

use std::collections::HashSet;

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::cell::{Cell, CellType, Neurite, Soma};
use crate::module::{CellModule, Host, Module};
use crate::substance::{Grid, Substance};
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
    rng: StdRng,
    pending_divisions: Vec<usize>,
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
            synapse_dist: 8.0,
            max_neurites: 512,
            max_cells: 64,
            synapse_strength: 20.0,
            rng: StdRng::seed_from_u64(seed),
            pending_divisions: vec![],
        }
    }

    // ─── construction ────────────────────────────────────────────

    pub fn add_substance(&mut self, s: Substance) {
        self.substances.push(s);
    }

    /// Add a cell (soma sphere) and return its id (= index).
    pub fn add_cell(&mut self, pos: Vec3, cell_type: CellType, radius: f64) -> usize {
        let id = self.cells.len();
        self.cells.push(Cell {
            id,
            cell_type,
            role: 0,
            sub_role: 0,
            soma: Soma {
                pos,
                radius,
                modules: vec![],
            },
            neurites: vec![],
            modules: vec![],
        });
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
        let base = self.cells[cell].soma.pos;
        let idx = self.cells[cell].neurites.len();
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter,
            points: vec![base],
            modules: vec![],
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
        if self.total_neurites() >= self.max_neurites {
            return;
        }
        let tip = self.cells[cell].neurites[idx].tip();
        let (is_axon, diameter) = {
            let n = &self.cells[cell].neurites[idx];
            (n.is_axon, n.diameter)
        };
        let modules = self.cells[cell].neurites[idx]
            .modules
            .iter()
            .filter(|m| m.copy_on_branch())
            .map(|m| m.boxed_copy())
            .collect();
        self.cells[cell].neurites.push(Neurite {
            is_axon,
            diameter,
            points: vec![tip],
            modules,
        });
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

    /// Numeric gradient of the total concentration of `id`.
    pub fn gradient(&self, id: &str, p: Vec3) -> Vec3 {
        let h = self.grid.cell;
        let dx = (self.concentration(id, p + Vec3::new(h, 0.0, 0.0))
            - self.concentration(id, p - Vec3::new(h, 0.0, 0.0)))
            / (2.0 * h);
        let dy = (self.concentration(id, p + Vec3::new(0.0, h, 0.0))
            - self.concentration(id, p - Vec3::new(0.0, h, 0.0)))
            / (2.0 * h);
        let dz = (self.concentration(id, p + Vec3::new(0.0, 0.0, h))
            - self.concentration(id, p - Vec3::new(0.0, 0.0, h)))
            / (2.0 * h);
        Vec3::new(dx, dy, dz)
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

    /// Explicit-Euler diffusion + degradation on each substance grid.
    pub fn diffuse_substances(&mut self) {
        let g = self.grid;
        let dt = self.dt;
        for s in self.substances.iter_mut() {
            let mut next = s.field.clone();
            for i in 1..g.nx.saturating_sub(1) {
                for j in 1..g.ny.saturating_sub(1) {
                    for k in 1..g.nz.saturating_sub(1) {
                        let idx = g.flat(i, j, k);
                        let lap = g.laplacian(&s.field, i, j, k);
                        let v = s.field[idx]
                            + dt * s.diffusion * lap
                            - dt * s.degradation * s.field[idx];
                        next[idx] = v.max(0.0);
                    }
                }
            }
            s.field = next;
        }
    }

    // ─── synapses ────────────────────────────────────────────────

    /// Form a synapse whenever an axon tip is within `dist` of another cell
    /// (soma or any neurite point). Target points are gathered once per step
    /// and synapses are deduplicated per (pre, post) pair.
    pub fn form_synapses(&mut self, dist: f64) {
        let n = self.cells.len();
        // Gather each cell's axon tips and all contact points once per step.
        let mut tips: Vec<Vec<Vec3>> = Vec::with_capacity(n);
        let mut cell_points: Vec<Vec<Vec3>> = Vec::with_capacity(n);
        for c in &self.cells {
            tips.push(
                c.neurites
                    .iter()
                    .filter(|neur| neur.is_axon)
                    .map(|neur| neur.tip())
                    .collect(),
            );
            let mut pts = Vec::with_capacity(1 + c.neurites.len() * 8);
            pts.push(c.soma.pos);
            for neur in &c.neurites {
                pts.extend(neur.points.iter().copied());
            }
            cell_points.push(pts);
        }
        let mut seen: HashSet<(usize, usize)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        for (a, cell_a) in self.cells.iter().enumerate() {
            let a_id = cell_a.id;
            let kind = cell_a.cell_type.synapse_kind();
            for (b, cell_b) in self.cells.iter().enumerate() {
                if a == b {
                    continue;
                }
                let b_id = cell_b.id;
                if seen.contains(&(a_id, b_id)) {
                    continue;
                }
                let mut min_d = f64::INFINITY;
                for t in &tips[a] {
                    min_d = min_d.min(t.dist(cell_b.soma.pos));
                    for p in &cell_points[b] {
                        min_d = min_d.min(t.dist(*p));
                    }
                }
                if min_d < dist {
                    // Weights emerge from the growth geometry: closer contact →
                    // stronger synapse. Sign comes from the presynaptic cell type.
                    let weight = self.synapse_strength / (1.0 + min_d);
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
        let d = self.synapse_dist;
        self.form_synapses(d);
        self.time += self.dt;
    }

    fn run_cell_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            let mut mods = std::mem::take(&mut self.cells[ci].modules);
            for mut m in mods.drain(..) {
                m.run(self, ci);
                self.cells[ci].modules.push(m);
            }
        }
    }

    fn run_soma_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            let mut mods = std::mem::take(&mut self.cells[ci].soma.modules);
            for mut m in mods.drain(..) {
                m.run(self, Host::Soma(ci));
                self.cells[ci].soma.modules.push(m);
            }
        }
    }

    fn run_neurite_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            let nneur = self.cells[ci].neurites.len();
            for ni in 0..nneur {
                let mut mods = std::mem::take(&mut self.cells[ci].neurites[ni].modules);
                for mut m in mods.drain(..) {
                    m.run(self, Host::Neurite { cell: ci, idx: ni });
                    self.cells[ci].neurites[ni].modules.push(m);
                }
            }
        }
    }

    fn process_divisions(&mut self) {
        let ids = std::mem::take(&mut self.pending_divisions);
        for id in ids {
            self.divide_cell(id);
        }
    }

    /// Split a cell into two: daughter gets a copy of every module flagged
    /// `copy_on_divide` (the self-construction instruction copy).
    fn divide_cell(&mut self, id: usize) {
        if self.cells.len() >= self.max_cells {
            return;
        }
        let offset = self.random_vec(self.grid.cell * 0.5);
        let (cell_type, role, soma_pos, soma_radius, soma_mods, neurites, cell_mods) = {
            let c = &self.cells[id];
            (
                c.cell_type,
                c.role,
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
                        )
                    })
                    .collect::<Vec<_>>(),
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
            soma: Soma {
                pos: soma_pos + offset,
                radius: (soma_radius.powi(3) * 0.5).powf(1.0 / 3.0),
                modules: soma_mods,
            },
            neurites: neurites
                .into_iter()
                .map(|(is_axon, diameter, points, modules)| Neurite {
                    is_axon,
                    diameter,
                    points,
                    modules,
                })
                .collect(),
            modules: cell_mods,
        };
        self.cells.push(daughter);
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
        assert!(w.cells.len() > 1, "expected division, got {} cells", w.cells.len());
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
        assert!(after > before, "secretion should raise field: {before} -> {after}");
    }
}
