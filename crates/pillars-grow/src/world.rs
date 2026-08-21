//! The simulation world (CX3D `ECM` + `Scheduler`): owns cells, substances,
//! synapses, time, and runs the per-step loop.
//!
//! ```text
//! step():
//!   1. diffuse substances (morphogens)
//!   2. run cell modules   (division)
//!   3. process divisions
//!   4. run soma modules   (secretion, migration)
//!   5. run neurite modules (growth cones, retraction)
//!   6. prune fully retracted neurites
//!   7. form synapses (geometry recomputed from live tips)
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
    /// True if formed by contact geometry ([`World::form_synapses`]); such
    /// synapses are recomputed every step, so a retracted/pruned neurite
    /// automatically drops its synapses. Rule-based synapses
    /// ([`World::connect_fates`]) are `false` and persist.
    pub geometric: bool,
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
    pending_prunes: Vec<(usize, usize)>,
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
            pending_prunes: vec![],
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
            alive: true,
            // Slightly jittered Delta breaks the lateral-inhibition symmetry
            // (who becomes primary first); deterministic per seed.
            delta: 1.0 + self.rng.gen_range(-0.2..0.2),
            notch: 0.0,
        });
        id
    }

    /// Mark a cell dead (apoptosis). It stops growing, forms no synapses and
    /// is excluded from the extracted graph; its index stays valid.
    pub fn kill_cell(&mut self, cell: usize) {
        if let Some(c) = self.cells.get_mut(cell) {
            c.alive = false;
        }
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

    /// Advance the growth-cone tip of a neurite by `step`, clamped to the
    /// simulation volume (CX3D `ECM` behaviour: neurites stay inside).
    pub fn extend_neurite(&mut self, cell: usize, idx: usize, step: Vec3) {
        let tip = self.cells[cell].neurites[idx].tip() + step;
        let b = self.bounds;
        let clamped = Vec3::new(
            tip.x.clamp(b.min.x, b.max.x),
            tip.y.clamp(b.min.y, b.max.y),
            tip.z.clamp(b.min.z, b.max.z),
        );
        self.cells[cell].neurites[idx].points.push(clamped);
    }

    /// Retract the growth-cone tip by `amount` length units: the tip walks
    /// back along the polyline, dropping exhausted segments. Once the tip is
    /// back at the soma (single point), the neurite is scheduled for pruning
    /// (removed after the module loop — CX3D `RetractNeurite`: a fully
    /// retracted neurite disappears).
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

    /// Move the soma by `delta`, clamped to the simulation volume.
    pub fn move_soma(&mut self, cell: usize, delta: Vec3) {
        let p = self.cells[cell].soma.pos + delta;
        let b = self.bounds;
        self.cells[cell].soma.pos = Vec3::new(
            p.x.clamp(b.min.x, b.max.x),
            p.y.clamp(b.min.y, b.max.y),
            p.z.clamp(b.min.z, b.max.z),
        );
    }

    /// Total neurite count across all cells.
    pub fn total_neurites(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| c.alive)
            .map(|c| c.neurites.len())
            .sum()
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

    /// Explicit-Euler diffusion + degradation on each substance grid. The
    /// Laplacian is normalized by `cell²` (grid units → world units), so the
    /// diffusion rate is physical: `dt·D/h²` per step (stable for ≤ 1/6).
    pub fn diffuse_substances(&mut self) {
        let inv_h2 = 1.0 / (self.grid.cell * self.grid.cell);
        let g = self.grid;
        let dt = self.dt;
        for s in self.substances.iter_mut() {
            let mut next = s.field.clone();
            for i in 1..g.nx.saturating_sub(1) {
                for j in 1..g.ny.saturating_sub(1) {
                    for k in 1..g.nz.saturating_sub(1) {
                        let idx = g.flat(i, j, k);
                        let lap = g.laplacian(&s.field, i, j, k) * inv_h2;
                        let v = s.field[idx] + dt * s.diffusion * lap
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
    ///
    /// Geometric synapses are **recomputed from scratch every step**: they are
    /// dropped first, then re-formed from the current axon tips. This makes
    /// retraction/pruning actually rewire the network — a pruned neurite
    /// immediately loses its contacts. Rule-based synapses ([`World::connect_fates`])
    /// are untouched.
    pub fn form_synapses(&mut self, dist: f64) {
        // Drop all geometric synapses and any synapse touching a dead cell;
        // keep live rule-based ones (they dedupe).
        self.synapses
            .retain(|s| !s.geometric && self.cells[s.pre].alive && self.cells[s.post].alive);
        let n = self.cells.len();
        // Gather each cell's axon tips and all contact points once per step.
        let mut tips: Vec<Vec<Vec3>> = Vec::with_capacity(n);
        let mut cell_points: Vec<Vec<Vec3>> = Vec::with_capacity(n);
        for c in &self.cells {
            if !c.alive {
                tips.push(vec![]);
                cell_points.push(vec![]);
                continue;
            }
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
                        geometric: true,
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
            .filter(|(_, c)| c.alive && c.role == from_role && c.sub_role == from_sub)
            .map(|(i, _)| i)
            .collect();
        let dst: Vec<usize> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.alive && c.role == to_role && c.sub_role == to_sub)
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
                    geometric: false,
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
        if scale <= 0.0 {
            return Vec3::ZERO;
        }
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
        let d = self.synapse_dist;
        self.form_synapses(d);
        self.time += self.dt;
    }

    fn run_cell_modules(&mut self) {
        let n = self.cells.len();
        for ci in 0..n {
            if !self.cells[ci].alive {
                continue;
            }
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
            if !self.cells[ci].alive {
                continue;
            }
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
            if !self.cells[ci].alive {
                continue;
            }
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
            if self.cells[id].alive {
                self.divide_cell(id);
            }
        }
    }

    /// Apply pending neurite prunes (after the module loop, so in-flight
    /// module iteration never sees a shifted neurite list).
    fn process_prunes(&mut self) {
        let prunes = std::mem::take(&mut self.pending_prunes);
        for (ci, ni) in prunes {
            if ni < self.cells[ci].neurites.len() {
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
        let (cell_type, role, soma_pos, soma_radius, soma_mods, neurites, cell_mods, delta, notch) = {
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
                c.delta,
                c.notch,
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
            alive: true,
            delta,
            notch,
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
    fn geometric_synapse_dropped_when_tip_retracts() {
        let mut w = world();
        let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        w.add_cell(Vec3::new(0.0, 0.0, 10.0), CellType::Inhibitory, 4.0);
        let n = w.new_neurite(a, true, 0.5);
        // Extend the axon toward B until its tip is within synapse_dist (8).
        w.extend_neurite(a, n, Vec3::new(0.0, 0.0, 3.0)); // tip at z=3 → dist 7 < 8
        w.form_synapses(8.0);
        assert_eq!(w.synapses.len(), 1);
        assert!(w.synapses[0].geometric);
        // Retract the tip back to the soma → contact gone → synapse recomputed away.
        w.retract_neurite(a, n, 100.0);
        w.form_synapses(8.0);
        assert!(
            w.synapses.is_empty(),
            "a retracted contact must drop its geometric synapse"
        );
    }

    #[test]
    fn rule_synapses_survive_geometry_recompute() {
        let mut w = world();
        let a1 = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let a2 = w.add_cell(Vec3::new(0.0, 0.0, 5.0), CellType::Excitatory, 4.0);
        let b1 = w.add_cell(Vec3::new(0.0, 0.0, 50.0), CellType::Inhibitory, 4.0);
        w.cells[a1].role = 1;
        w.cells[a2].role = 1;
        w.cells[b1].role = 2;
        let made = w.connect_fates(1, 0, 2, 0, 3.0);
        assert_eq!(made, 2);
        // Cells are far apart, so geometry forms nothing — and must not eat
        // the rule-based wiring either.
        w.form_synapses(8.0);
        assert_eq!(
            w.synapses.len(),
            2,
            "rule synapses must survive form_synapses"
        );
        assert!(w.synapses.iter().all(|s| !s.geometric));
    }

    #[test]
    fn rule_synapses_to_dead_cell_dropped() {
        let mut w = world();
        let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
        let b = w.add_cell(Vec3::new(0.0, 0.0, 50.0), CellType::Inhibitory, 4.0);
        w.cells[a].role = 1;
        w.cells[b].role = 2;
        w.connect_fates(1, 0, 2, 0, 3.0);
        assert_eq!(w.synapses.len(), 1);
        w.kill_cell(b);
        w.form_synapses(8.0);
        assert!(
            w.synapses.is_empty(),
            "synapses to a dead cell must be dropped"
        );
    }

    #[test]
    fn extend_neurite_clamps_to_bounds() {
        let mut w = world();
        let c = w.add_cell(Vec3::new(0.0, 0.0, 79.0), CellType::Excitatory, 4.0);
        let n = w.new_neurite(c, true, 0.5);
        for _ in 0..10 {
            w.extend_neurite(c, n, Vec3::new(0.0, 0.0, 10.0));
        }
        let tip = w.cells[c].neurites[n].tip();
        assert!(
            tip.z <= 80.0,
            "neurite tip must stay inside bounds ({tip:?})"
        );
    }
}
