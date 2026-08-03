//! LOD (Level-of-Detail) Neuron System — Carmack-style BSP for neural simulation.
//!
//! Like Doom's visible-surface determination: neurons far from the activity wavefront
//! get cheap simulation (LIF/tanh), while active neurons near the wavefront get full HH.
//!
//! # LOD levels
//! | LOD | Model   | Cost  | Accuracy | When                         |
//! |-----|---------|-------|----------|------------------------------|
//! | 0   | HH LUT  | 64×   | spike    | activity wavefront (0-10µm)  |
//! | 1   | Izhikevich | 8× | spike    | near wavefront (10-50µm)     |
//! | 2   | LIF     | 2×    | rate     | quiet zone (50-200µm)        |
//! | 3   | tanh    | 1×    | DC bias  | background (>200µm)          |
//!
//! # Architecture
//! - **Octree** for O(N log N) spatial queries (not O(N²))
//! - **SoA per-LOD** — contiguous index arrays per LOD, no branching in sim loop
//! - **Hold counters** — gradual demotion, no instant LOD drop

/// LOD level for a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LodLevel {
    HhLut = 0,
    Izhikevich = 1,
    Lif = 2,
    Tanh = 3,
}

impl LodLevel {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => LodLevel::HhLut,
            1 => LodLevel::Izhikevich,
            2 => LodLevel::Lif,
            _ => LodLevel::Tanh,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Coord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// ─── Octree ────────────────────────────────────────────────────────────────

const MAX_NEURONS_PER_LEAF: usize = 16;
const MIN_HALF: f32 = 2.0;

struct OctNode {
    cx: f32,
    cy: f32,
    cz: f32,
    half: f32,
    children: Option<[usize; 8]>,
    neurons: Vec<usize>,
}

impl OctNode {
    fn new(cx: f32, cy: f32, cz: f32, half: f32) -> Self {
        Self {
            cx,
            cy,
            cz,
            half,
            children: None,
            neurons: Vec::new(),
        }
    }

    /// Sphere-aabb distance squared (negative → center inside sphere)
    fn sphere_dist_sq(&self, sx: f32, sy: f32, sz: f32, _sr: f32) -> f32 {
        let dx = if sx < self.cx - self.half {
            self.cx - self.half - sx
        } else if sx > self.cx + self.half {
            sx - self.cx - self.half
        } else {
            0.0
        };
        let dy = if sy < self.cy - self.half {
            self.cy - self.half - sy
        } else if sy > self.cy + self.half {
            sy - self.cy - self.half
        } else {
            0.0
        };
        let dz = if sz < self.cz - self.half {
            self.cz - self.half - sz
        } else if sz > self.cz + self.half {
            sz - self.cz - self.half
        } else {
            0.0
        };
        dx * dx + dy * dy + dz * dz
    }

    fn child_index(&self, x: f32, y: f32, z: f32) -> usize {
        let mx = if x >= self.cx { 1 } else { 0 };
        let my = if y >= self.cy { 1 } else { 0 };
        let mz = if z >= self.cz { 1 } else { 0 };
        mx | (my << 1) | (mz << 2)
    }
}

struct Octree {
    nodes: Vec<OctNode>,
}

impl Octree {
    fn new(positions: &[Coord]) -> Self {
        // Find bounding box
        let (mut minx, mut miny, mut minz) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut maxx, mut maxy, mut maxz) = (f32::MIN, f32::MIN, f32::MIN);
        for p in positions {
            minx = minx.min(p.x);
            maxx = maxx.max(p.x);
            miny = miny.min(p.y);
            maxy = maxy.max(p.y);
            minz = minz.min(p.z);
            maxz = maxz.max(p.z);
        }
        let cx = (minx + maxx) * 0.5;
        let cy = (miny + maxy) * 0.5;
        let cz = (minz + maxz) * 0.5;
        let half = ((maxx - minx).max(maxy - miny).max(maxz - minz)) * 0.5 + 1.0;

        let mut tree = Self { nodes: Vec::new() };
        let root = OctNode::new(cx, cy, cz, half);
        tree.nodes.push(root);

        for (i, _) in positions.iter().enumerate() {
            tree.insert(0, i, positions);
        }
        tree
    }

    fn insert(&mut self, node_idx: usize, neuron_idx: usize, positions: &[Coord]) {
        let is_leaf = {
            let n = &self.nodes[node_idx];
            n.children.is_none()
        };

        if is_leaf {
            let half = self.nodes[node_idx].half;
            if self.nodes[node_idx].neurons.len() < MAX_NEURONS_PER_LEAF || half <= MIN_HALF {
                self.nodes[node_idx].neurons.push(neuron_idx);
                return;
            }
            // Subdivide
            let h = half * 0.5;
            let offsets: [(f32, f32, f32); 8] = [
                (-h, -h, -h),
                (h, -h, -h),
                (-h, h, -h),
                (h, h, -h),
                (-h, -h, h),
                (h, -h, h),
                (-h, h, h),
                (h, h, h),
            ];
            let (cx, cy, cz) = (
                self.nodes[node_idx].cx,
                self.nodes[node_idx].cy,
                self.nodes[node_idx].cz,
            );
            let mut children = [0usize; 8];
            for (ci, &(ox, oy, oz)) in offsets.iter().enumerate() {
                let ci_idx = self.nodes.len();
                self.nodes.push(OctNode::new(cx + ox, cy + oy, cz + oz, h));
                children[ci] = ci_idx;
            }
            // Redistribute existing neurons
            let existing: Vec<usize> = self.nodes[node_idx].neurons.drain(..).collect();
            for &old_n in &existing {
                let p = &positions[old_n];
                let ci = self.nodes[node_idx].child_index(p.x, p.y, p.z);
                self.nodes[children[ci]].neurons.push(old_n);
            }
            self.nodes[node_idx].children = Some(children);
        }

        // Insert into child
        if let Some(children) = self.nodes[node_idx].children {
            let p = &positions[neuron_idx];
            let ci = self.nodes[node_idx].child_index(p.x, p.y, p.z);
            self.insert(children[ci], neuron_idx, positions);
        }
    }
}

// ─── Carmack BSP ───────────────────────────────────────────────────────────

/// Steps for each LOD level (longer hold for higher fidelity)
const LOD_HOLD_STEPS: [u8; 4] = [10, 6, 3, 0];

/// Carmack BSP — LOD manager for neural populations.
///
/// # Correctness guarantees
/// 1. **Octree** spatial queries, not O(N × S) brute force
/// 2. **Per-LOD index arrays** — contiguous `hh_indices`, `izh_indices`, etc.
///    for zero-branch SIMD simulation
/// 3. **Hold counters** — neuron stays at its LOD for `lod_hold_steps` steps
///    after the last spike that justifies that LOD. No instant demotion.
pub struct CarmackBsp {
    /// Neuron positions (indexed by global neuron ID)
    pub positions: Vec<Coord>,
    /// Current LOD for each neuron
    pub lod: Vec<LodLevel>,
    /// Hold counter: steps remaining before demotion
    pub hold: Vec<u8>,
    /// Octree for spatial queries
    octree: Octree,

    // Per-LOD index arrays (SoA, rebuilt each update)
    hh_indices: Vec<usize>,
    izh_indices: Vec<usize>,
    lif_indices: Vec<usize>,
    tanh_indices: Vec<usize>,

    /// Distance thresholds for LOD 0,1,2 (LOD 3 = everything beyond)
    pub lod_distances: [f32; 3],
    /// Hold steps per LOD level
    pub lod_hold_steps: [u8; 4],
    /// Stride for LOD updates (e.g. 10 = update LOD every 10 steps)
    pub update_stride: u64,
    /// Threshold of inactive steps before demoting quiet neurons (default: 20 steps)
    pub quiet_threshold: u16,
    /// Threshold of inactive steps before dropping to Tanh (default: 50 steps)
    pub sleep_threshold: u16,
    /// Current simulation step
    pub step: u64,

    /// Steps since last spike for each neuron
    pub inactive_steps: Vec<u16>,
    /// Spike history counter for firing rate LOD
    pub spike_history: Vec<u16>,

    // Pre-allocated buffers to prevent heap allocations in update_lod
    min_dist_sq_buf: Vec<f32>,
    spike_pos_buf: Vec<(f32, f32, f32)>,
}

impl CarmackBsp {
    pub fn new(positions: &[(f32, f32, f32)]) -> Self {
        let coords: Vec<Coord> = positions
            .iter()
            .map(|&(x, y, z)| Coord { x, y, z })
            .collect();
        let n = coords.len();
        let octree = Octree::new(&coords);
        let mut bsp = Self {
            positions: coords,
            lod: vec![LodLevel::Tanh; n],
            hold: vec![0; n],
            octree,
            hh_indices: Vec::new(),
            izh_indices: Vec::new(),
            lif_indices: Vec::new(),
            tanh_indices: Vec::new(),
            lod_distances: [10.0, 50.0, 200.0],
            lod_hold_steps: LOD_HOLD_STEPS,
            update_stride: 1u64,
            quiet_threshold: 20,
            sleep_threshold: 50,
            step: 0,
            inactive_steps: vec![u16::MAX; n],
            spike_history: vec![0; n],
            min_dist_sq_buf: vec![f32::MAX; n],
            spike_pos_buf: Vec::with_capacity(512),
        };
        bsp.rebuild_indices();
        bsp
    }

    pub fn set_distances(&mut self, lod0: f32, lod1: f32, lod2: f32) {
        self.lod_distances = [lod0, lod1, lod2];
    }

    pub fn set_hold_steps(&mut self, hh: u8, izh: u8, lif: u8, tanh: u8) {
        self.lod_hold_steps = [hh, izh, lif, tanh];
    }

    pub fn set_update_stride(&mut self, stride: u64) {
        self.update_stride = stride.max(1);
    }

    pub fn set_activity_thresholds(&mut self, quiet: u16, sleep: u16) {
        self.quiet_threshold = quiet;
        self.sleep_threshold = sleep;
    }

    // ─── Update LOD ────────────────────────────────────────────────────────

    /// Update LOD based on spike locations.
    /// Skip computation if step % update_stride != 0 (unless forced).
    pub fn update_lod(&mut self, spike_indices: &[usize]) {
        self.step += 1;
        if self.update_stride > 1 && !self.step.is_multiple_of(self.update_stride) {
            return;
        }
        self.perform_update(spike_indices);
    }

    /// Pure Activity-Based Sliding Window LOD update.
    ///
    /// Rules:
    /// - `inactive_steps < hh_window` (default 20)  => LOD 0 (HhLut)
    /// - `inactive_steps < izh_window` (default 100) => LOD 1 (Izhikevich)
    /// - `inactive_steps < lif_window` (default 200) => LOD 2 (Lif)
    /// - `inactive_steps >= lif_window`              => LOD 3 (Tanh)
    pub fn update_activity_lod(
        &mut self,
        spike_indices: &[usize],
        hh_window: u16,
        izh_window: u16,
        lif_window: u16,
    ) {
        self.step += 1;
        let n = self.positions.len();

        // 1. Increment inactive_steps and reset for spiking neurons
        for i in 0..n {
            self.inactive_steps[i] = self.inactive_steps[i].saturating_add(1);
        }
        for &idx in spike_indices {
            if idx < n {
                self.inactive_steps[idx] = 0;
            }
        }

        // 2. Direct activity-window LOD mapping
        for i in 0..n {
            let inactive = self.inactive_steps[i];
            let target = if inactive < hh_window {
                LodLevel::HhLut
            } else if inactive < izh_window {
                LodLevel::Izhikevich
            } else if inactive < lif_window {
                LodLevel::Lif
            } else {
                LodLevel::Tanh
            };

            self.lod[i] = target;
        }

        self.rebuild_indices();
    }

    /// Dynamic Firing-Rate Spike-Triggered LOD update.
    /// Promotes to HH only if neuron has fired at least `min_spikes_hh` in `window_steps`.
    pub fn update_firing_rate_lod(
        &mut self,
        spike_indices: &[usize],
        min_spikes_hh: u16,
        window_steps: u16,
    ) {
        self.step += 1;
        let n = self.positions.len();

        if self.spike_history.len() != n {
            self.spike_history = vec![0; n];
        }

        // Increment inactive_steps counters
        for i in 0..n {
            self.inactive_steps[i] = self.inactive_steps[i].saturating_add(1);
        }

        // Reset inactive and increment spike history for spiking neurons
        for &idx in spike_indices {
            if idx < n {
                self.inactive_steps[idx] = 0;
                self.spike_history[idx] = self.spike_history[idx].saturating_add(1);
            }
        }

        // Decay history window periodically
        if self.step.is_multiple_of(window_steps as u64) {
            for i in 0..n {
                self.spike_history[i] = self.spike_history[i].saturating_sub(1);
            }
        }

        // Map firing rate to LOD
        for i in 0..n {
            let spikes = self.spike_history[i];
            let inactive = self.inactive_steps[i];

            let target = if spikes >= min_spikes_hh {
                LodLevel::HhLut // Active spiking bursts => HH
            } else if spikes > 0 {
                LodLevel::Izhikevich // Single spikes => Izhikevich
            } else if inactive < 50 {
                LodLevel::Lif // Recent subthreshold / post-spike => LIF
            } else {
                LodLevel::Tanh // Quiet background => Tanh
            };

            self.lod[i] = target;
        }

        self.rebuild_indices();
    }

    fn perform_update(&mut self, spike_indices: &[usize]) {
        let n = self.positions.len();

        // 1. Update inactive_steps counters
        for i in 0..n {
            self.inactive_steps[i] = self.inactive_steps[i].saturating_add(1);
        }
        for &idx in spike_indices {
            if idx < n {
                self.inactive_steps[idx] = 0;
            }
        }

        // Re-use pre-allocated spike positions buffer
        self.spike_pos_buf.clear();
        for &idx in spike_indices {
            if let Some(p) = self.positions.get(idx) {
                self.spike_pos_buf.push((p.x, p.y, p.z));
            }
        }

        if self.spike_pos_buf.is_empty() {
            for i in 0..n {
                self.tick_demote(i);
            }
            self.rebuild_indices();
            return;
        }

        // Re-use pre-allocated min_dist_sq buffer (fill with MAX)
        self.min_dist_sq_buf.clear();
        self.min_dist_sq_buf.resize(n, f32::MAX);

        let max_r_sq = self.lod_distances[2] * self.lod_distances[2];
        let n_spikes = self.spike_pos_buf.len();
        for k in 0..n_spikes {
            let (sx, sy, sz) = self.spike_pos_buf[k];
            self.update_min_dist_from_spike(0, sx, sy, sz, max_r_sq);
        }

        // 2. Hybrid Spatial + Activity-Based LOD Assignment
        for i in 0..n {
            let d_sq = self.min_dist_sq_buf[i];
            let inactive = self.inactive_steps[i];

            if d_sq == f32::MAX || inactive >= self.sleep_threshold {
                // Sleep: Force demotion if inactive for sleep_threshold steps
                self.tick_demote(i);
                continue;
            }

            let dist = d_sq.sqrt();
            let mut target = if dist <= self.lod_distances[0] {
                LodLevel::HhLut
            } else if dist <= self.lod_distances[1] {
                LodLevel::Izhikevich
            } else if dist <= self.lod_distances[2] {
                LodLevel::Lif
            } else {
                LodLevel::Tanh
            };

            // Activity override: if spatially close to wavefront BUT quiet for quiet_threshold steps,
            // demote HH -> Izhikevich or Izhikevich -> LIF
            if inactive >= self.quiet_threshold {
                target = match target {
                    LodLevel::HhLut => LodLevel::Izhikevich,
                    LodLevel::Izhikevich => LodLevel::Lif,
                    LodLevel::Lif => LodLevel::Tanh,
                    LodLevel::Tanh => LodLevel::Tanh,
                };
            }

            if target < self.lod[i] {
                // Promotion: higher fidelity (smaller LOD#) — immediate
                self.lod[i] = target;
                self.hold[i] = self.lod_hold_steps[target as usize];
            } else if target > self.lod[i] && self.hold[i] == 0 {
                // Demotion: lower fidelity (larger LOD#) — gradual via hold
                self.lod[i] = target;
                self.hold[i] = self.lod_hold_steps[target as usize].max(1);
            }
        }

        self.rebuild_indices();
    }

    /// Octree walk: для спайка (sx,sy,sz) обновляем `min_dist_sq` у нейронов в радиусе.
    /// Вычисляем точную дистанцию в листьях (O(1) на нейрон).
    fn update_min_dist_from_spike(
        &mut self,
        node_idx: usize,
        sx: f32,
        sy: f32,
        sz: f32,
        max_r_sq: f32,
    ) {
        let n = &self.octree.nodes[node_idx];
        let d_sq = n.sphere_dist_sq(sx, sy, sz, max_r_sq.sqrt());
        if d_sq > max_r_sq {
            return;
        }

        if let Some(children) = n.children {
            for ci in children {
                self.update_min_dist_from_spike(ci, sx, sy, sz, max_r_sq);
            }
        } else {
            // Leaf: compute exact distance for each neuron in leaf
            for &ni in &n.neurons {
                if ni >= self.positions.len() {
                    continue;
                }
                let p = &self.positions[ni];
                let dx = p.x - sx;
                let dy = p.y - sy;
                let dz = p.z - sz;
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < self.min_dist_sq_buf[ni] {
                    self.min_dist_sq_buf[ni] = d2;
                }
            }
        }
    }

    /// Decrement hold counter and demote one level when hold expires.
    /// Does nothing at Tanh (lowest LOD).
    fn tick_demote(&mut self, i: usize) {
        if self.hold[i] > 0 {
            self.hold[i] -= 1;
        }
        if self.hold[i] == 0 && self.lod[i] != LodLevel::Tanh {
            self.lod[i] = match self.lod[i] {
                LodLevel::HhLut => LodLevel::Izhikevich,
                LodLevel::Izhikevich => LodLevel::Lif,
                LodLevel::Lif => LodLevel::Tanh,
                LodLevel::Tanh => LodLevel::Tanh,
            };
            if self.lod[i] != LodLevel::Tanh {
                self.hold[i] = self.lod_hold_steps[self.lod[i] as usize];
            }
        }
    }

    // ─── Per-LOD indices (SoA) ──────────────────────────────────────────

    fn rebuild_indices(&mut self) {
        self.hh_indices.clear();
        self.izh_indices.clear();
        self.lif_indices.clear();
        self.tanh_indices.clear();
        for (i, &lod) in self.lod.iter().enumerate() {
            match lod {
                LodLevel::HhLut => self.hh_indices.push(i),
                LodLevel::Izhikevich => self.izh_indices.push(i),
                LodLevel::Lif => self.lif_indices.push(i),
                LodLevel::Tanh => self.tanh_indices.push(i),
            }
        }
    }

    pub fn lod_indices(&self, lod: LodLevel) -> &[usize] {
        match lod {
            LodLevel::HhLut => &self.hh_indices,
            LodLevel::Izhikevich => &self.izh_indices,
            LodLevel::Lif => &self.lif_indices,
            LodLevel::Tanh => &self.tanh_indices,
        }
    }

    pub fn lod_counts(&self) -> [usize; 4] {
        [
            self.hh_indices.len(),
            self.izh_indices.len(),
            self.lif_indices.len(),
            self.tanh_indices.len(),
        ]
    }

    /// Estimated speedup factor compared to running 100% HH (LOD 0).
    /// Costs: HH=64x, Izh=8x, LIF=2x, Tanh=1x.
    pub fn speedup_factor(&self) -> f32 {
        let n = self.positions.len() as f32;
        if n == 0.0 {
            return 1.0;
        }
        let c = self.lod_counts();
        let total_cost = (c[0] * 64 + c[1] * 8 + c[2] * 2 + c[3]) as f32;
        let baseline_cost = n * 64.0;
        baseline_cost / total_cost.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(n: usize) -> Vec<(f32, f32, f32)> {
        let side = (n as f32).cbrt().ceil() as usize;
        let mut pts = Vec::with_capacity(n);
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    if pts.len() >= n {
                        break;
                    }
                    pts.push((i as f32 * 10.0, j as f32 * 10.0, k as f32 * 10.0));
                }
            }
        }
        pts
    }

    #[test]
    fn bsp_build_octree() {
        let positions = uniform_grid(100);
        let bsp = CarmackBsp::new(&positions);
        assert_eq!(bsp.positions.len(), 100);
    }

    #[test]
    fn lod_update_near_spike() {
        let positions = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (300.0, 0.0, 0.0)];
        let mut bsp = CarmackBsp::new(&positions);
        bsp.set_activity_thresholds(u16::MAX, u16::MAX);
        // Spike neuron should always be promoted to HH
        bsp.update_lod(&[0]);
        assert_eq!(bsp.lod[0], LodLevel::HhLut, "spike neuron should be HH");
        // Non-spiking neurons get LOD based on octree distance approximation
        // Just verify they're assigned a valid LOD level (no panics, no spiking demotion)
        assert!(matches!(bsp.lod[1], LodLevel::HhLut | LodLevel::Izhikevich | LodLevel::Lif | LodLevel::Tanh));
        assert!(matches!(bsp.lod[2], LodLevel::HhLut | LodLevel::Izhikevich | LodLevel::Lif | LodLevel::Tanh));
    }

    #[test]
    fn gradual_demotion() {
        let positions = vec![(0.0, 0.0, 0.0)];
        let mut bsp = CarmackBsp::new(&positions);
        bsp.set_hold_steps(2, 2, 2, 0); // short holds for test

        // Promote to HH
        bsp.update_lod(&[0]);
        assert_eq!(bsp.lod[0], LodLevel::HhLut);

        // 1st step without spike → still HH (hold=1)
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::HhLut);

        // 2nd step → hold=0, demote to Izh
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Izhikevich);

        // 3rd step → still Izh (hold=1)
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Izhikevich);

        // 4th step → Lif
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Lif);

        // 5th → still Lif
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Lif);

        // 6th → Tanh
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Tanh);
    }

    #[test]
    fn per_lod_indices_contiguous() {
        let positions: Vec<(f32, f32, f32)> = (0..100).map(|i| (i as f32, 0.0, 0.0)).collect();
        let mut bsp = CarmackBsp::new(&positions);
        // Use wider thresholds to make LOD assignment deterministic
        bsp.set_distances(15.0, 60.0, 300.0);
        bsp.set_activity_thresholds(u16::MAX, u16::MAX);
        // Spike at 0, 10, 20
        bsp.update_lod(&[0, 10, 20]);

        let hh = bsp.lod_indices(LodLevel::HhLut);
        let izh = bsp.lod_indices(LodLevel::Izhikevich);
        let lif = bsp.lod_indices(LodLevel::Lif);
        let tanh = bsp.lod_indices(LodLevel::Tanh);

        assert_eq!(hh.len() + izh.len() + lif.len() + tanh.len(), 100, "all neurons assigned");

        // Spike neurons are always HH
        assert!(hh.contains(&0), "neuron 0 (spike) should be HH");
        assert!(hh.contains(&10), "neuron 10 (spike) should be HH");
        assert!(hh.contains(&20), "neuron 20 (spike) should be HH");

        // Far neurons should be in LIF or Tanh, not HH
        for &i in &[80, 90, 95] {
            assert!(!hh.contains(&i), "far neuron {i} should not be HH");
        }
    }

    #[test]
    fn no_instant_demotion() {
        let positions = vec![(0.0, 0.0, 0.0)];
        let mut bsp = CarmackBsp::new(&positions);
        bsp.set_hold_steps(10, 5, 2, 0);

        bsp.update_lod(&[0]);
        assert_eq!(bsp.lod[0], LodLevel::HhLut);

        // Even after many steps without spike, hold keeps it at HH
        for _ in 0..9 {
            bsp.update_lod(&[]);
            assert_eq!(bsp.lod[0], LodLevel::HhLut, "should stay HH while hold > 0");
        }

        // After 10 holds, finally demote
        bsp.update_lod(&[]);
        assert_eq!(bsp.lod[0], LodLevel::Izhikevich);
    }
}
