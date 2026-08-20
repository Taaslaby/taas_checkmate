//! # d_λ discretization + passive cable solver (Фаза 5d) — MorphoIR → cable
//!
//! Bridges real morphology ([`MorphoIR`](super::parse::MorphoIR), from NeuroML
//! `<morphology>` or SWC) to a flat compartment tree and runs the **passive**
//! cable equation over it (implicit Euler via the Hines algorithm).
//!
//! ## d_λ rule
//!
//! Each segment is split into `n = max(1, ⌈L/(d_λ·λ)⌉)` compartments of length
//! `L/n` and the segment's average diameter, where `λ = √(d·Rm/4·Ra)` is the DC
//! space constant. A segment's first compartment connects to the parent
//! segment's last compartment; the rest chain linearly. The root (soma) is one
//! compartment — a zero-length single-point soma gets effective length = its
//! diameter (cylinder convention, matching NEURON's `L = diam` for SWC soma).
//!
//! When a segment feeds **more than one** child, a zero-volume **branch node**
//! is inserted at its distal end (`r_axial=0`, `area=cm=g_leak=0`); the children
//! connect to it. This keeps the parent's distal half-resistance a single shared
//! resistor (NEURON's branch-point convention) instead of duplicating it per
//! child — duplicating would over-couple the daughters (a real ~1% error).
//!
//! ## Unit contract (NEURON convention, same as `kernels::dendrite::cable`)
//!
//! * length / diameter — **µm**; axial resistivity `Ra` — **Ω·cm**;
//! * specific membrane capacitance `cm` — **µF/cm²**; specific membrane
//!   resistance `Rm` — **Ω·cm²**;
//! * derived per compartment: area `A = π·d·L` (µm²), capacitance
//!   `C = cm·A` in **nF**, leak conductance `g_leak = A/Rm` in **µS**, axial
//!   resistance `R = 4·Ra·L/(π·d²)` in **MΩ**.
//!
//! The solver uses V in **mV**, t in **ms**, I in **nA** — the set where
//! `C[nF]·dV/dt[ms] = g[µS]·(E−V)[mV] = (ΔV)[mV]/R[MΩ] = I[nA]` are all
//! dimensionally identical, so the membrane time constant is simply
//! `τ = C[nF]/g[µS]` (ms).

use std::collections::HashMap;

use super::parse::{BiophysicalProperties, MorphoIR, Point3d};
use super::NeuromlError;
use crate::api::population::{CableCa, CableVoltage};
use crate::api::Population;
use crate::core::population::I_SYN_CLAMP_RANGE;
use crate::kernels::hh_channels::{
    markov_open_fraction, markov_steady_state, markov_step, GateKinetics, MarkovGateSpec,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const PI: f32 = std::f32::consts::PI;

/// Faraday constant, C/mol (LEMS `charge_per_mole`).
const FARADAY: f32 = 96485.3;
/// Ideal gas constant, J/(K·mol) (LEMS `idealGasConstantDims`).
const R_GAS: f32 = 8.314_462;
/// Ca²⁺ valence (LEMS `channelPopulationNernst`/`channelDensityNernst`).
const Z_CA: f32 = 2.0;

/// Passive biophysical constants for the cable discretization.
pub(crate) struct CableParams {
    /// Axial (intracellular) resistivity, Ω·cm.
    pub ra_ohm_cm: f32,
    /// Specific membrane capacitance, µF/cm².
    pub cm_uf_per_cm2: f32,
    /// Specific membrane resistance, Ω·cm² (leak conductance = 1/Rm).
    pub rm_ohm_cm2: f32,
    /// Target compartment size as a fraction of the space constant (0.1).
    pub d_lambda: f32,
}

impl Default for CableParams {
    fn default() -> Self {
        Self {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        }
    }
}

impl CableParams {
    /// Resolve the passive cable constants from parsed `<biophysicalProperties>`:
    ///
    /// * `intracellularProperties/resistivity` (Ω·cm) → `ra_ohm_cm`;
    /// * the passive leak `channelDensity` (ion `non_specific`/`pas`/`leak`)
    ///   → `rm_ohm_cm2 = 1/condDensity` (its `condDensity` is already S/cm²);
    /// * `membraneProperties/specificCapacitance` (µF/cm²) → `cm_uf_per_cm2`.
    ///
    /// Anything the model didn't declare keeps its [`Default`] value
    /// (`d_lambda` has no NeuroML source and always stays at 0.1).
    pub fn from_biophysics(bp: &BiophysicalProperties) -> Self {
        let default = Self::default();
        // rm (Ω·cm²) = 1 / g_leak (S/cm²); a non-positive density would give
        // a non-physical negative resistance, so it keeps the default.
        let rm_ohm_cm2 = bp
            .passive_leak()
            .filter(|leak| leak.cond_density > 0.0)
            .map(|leak| 1.0 / leak.cond_density)
            .unwrap_or(default.rm_ohm_cm2);
        Self {
            ra_ohm_cm: bp.intracellular.resistivity,
            cm_uf_per_cm2: bp.membrane.specific_capacitance,
            rm_ohm_cm2,
            ..default
        }
    }
}

/// Leak reversal / resting potential (mV) for the passive cable: the passive
/// leak `channelDensity`'s `erev`, falling back to `initMembPotential` when no
/// passive channel is declared.
pub(crate) fn passive_leak_reversal(bp: &BiophysicalProperties) -> f32 {
    bp.passive_leak()
        .map(|leak| leak.erev)
        .unwrap_or(bp.membrane.init_memb_potential)
}

/// A single cable compartment (a sub-cylinder of one morphology segment).
#[derive(Clone)]
pub(crate) struct Compartment {
    /// Parent compartment index (`None` for the root / soma).
    pub parent: Option<usize>,
    /// The morphology segment this compartment was discretized from.
    pub segment_id: usize,
    /// Compartment length, µm.
    pub length_um: f32,
    /// Compartment diameter, µm.
    pub diameter_um: f32,
    /// Membrane surface area, µm² (`π·d·L`).
    pub area_um2: f32,
    /// Membrane capacitance, nF (`cm·area`).
    pub cm_nf: f32,
    /// Leak conductance, µS (`area/Rm`).
    pub g_leak_us: f32,
    /// Full-length axial resistance of this compartment, MΩ
    /// (`4·Ra·L/(π·d²)`); the junction to the parent is `(R_parent+R_this)/2`.
    pub r_axial_mohm: f32,
    /// Centre position (µm) — interpolated along the parent segment
    /// (`proximal + (j+0.5)/n·(distal−proximal)`); the branch node sits at the
    /// segment's distal point. Kept for GUI/analysis views of the morphology.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Cumulative path distance from the root along the tree (µm), measured
    /// to this compartment's **centre**. Root = `length/2`; each child adds
    /// `parent.length/2 + own.length/2` (a chain of equal compartments steps
    /// by exactly one compartment length).
    pub path_um: f32,
}

/// A discretized morphology: a flat compartment list + the root index.
#[derive(Clone)]
pub(crate) struct CompartmentTree {
    pub compartments: Vec<Compartment>,
    pub root: usize,
}

impl CompartmentTree {
    /// Junction axial resistance between a compartment and its parent (MΩ),
    /// i.e. `(R_parent + R_this)/2`. `0.0` for the root.
    pub fn junction_resistance(&self, idx: usize) -> f32 {
        let c = &self.compartments[idx];
        match c.parent {
            Some(p) => (self.compartments[p].r_axial_mohm + c.r_axial_mohm) / 2.0,
            None => 0.0,
        }
    }

    /// Number of compartments.
    pub fn len(&self) -> usize {
        self.compartments.len()
    }

    /// Test-only convenience (production code never checks the tree emptiness
    /// directly — the discretizer validates a non-empty morphology).
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.compartments.is_empty()
    }

    /// Indices of compartments whose `segment_id` is in `segment_ids` (in
    /// compartment order).
    pub fn group_compartments(&self, segment_ids: &[usize]) -> Vec<usize> {
        self.compartments
            .iter()
            .enumerate()
            .filter(|(_, c)| segment_ids.contains(&c.segment_id))
            .map(|(i, _)| i)
            .collect()
    }
}

/// DC space constant (µm) of a cylinder of diameter `d_um`.
fn space_constant_um(d_um: f32, rm_ohm_cm2: f32, ra_ohm_cm: f32) -> f32 {
    // λ(cm) = √(d(cm)·Rm/(4·Ra)); ×1e4 → µm. Folds to `√(d·Rm/(4·Ra))·1e2`.
    (d_um * rm_ohm_cm2 / (4.0 * ra_ohm_cm)).sqrt() * 100.0
}

/// Euclidean distance between two points (µm).
fn dist(p: &Point3d, q: &Point3d) -> f32 {
    let dx = p.x - q.x;
    let dy = p.y - q.y;
    let dz = p.z - q.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Geometry (length + average diameter + axis endpoints) of one morphology
/// segment, resolving an omitted `proximal` against the parent's `distal`.
struct SegmentGeom {
    length_um: f32,
    diameter_um: f32,
    prox_x: f32,
    prox_y: f32,
    prox_z: f32,
    distal_x: f32,
    distal_y: f32,
    distal_z: f32,
}

/// Discretize a morphology into a compartment tree using the d_λ rule.
///
/// # Errors
/// Fails if the morphology has no root (`parent = None` segment), more than one
/// root, a segment referencing an unknown parent, or a zero-diameter segment.
pub(crate) fn discretize(
    morpho: &MorphoIR,
    params: &CableParams,
) -> Result<CompartmentTree, NeuromlError> {
    let segs = &morpho.segments;
    if segs.is_empty() {
        return Err(NeuromlError::parse(
            "discretize: morphology has no segments",
        ));
    }

    // Segment id → index into `segs`.
    let mut index: HashMap<usize, usize> = HashMap::with_capacity(segs.len());
    for (i, s) in segs.iter().enumerate() {
        if index.insert(s.id, i).is_some() {
            return Err(NeuromlError::parse(format!(
                "discretize: duplicate segment id {}",
                s.id
            )));
        }
    }

    // Root(s): segments with `parent = None`. A neuron is a single tree — one
    // root. Validate every parent reference and count roots.
    let mut root: Option<usize> = None;
    for s in segs {
        match &s.parent {
            None => {
                if root.replace(s.id).is_some() {
                    return Err(NeuromlError::parse(
                        "discretize: morphology has multiple roots (expected one soma)",
                    ));
                }
            }
            Some(p) => {
                if !index.contains_key(&p.segment) {
                    return Err(NeuromlError::parse(format!(
                        "discretize: segment {} references unknown parent {}",
                        s.id, p.segment
                    )));
                }
            }
        }
    }
    let root_id = root.ok_or_else(|| NeuromlError::parse("discretize: no root segment"))?;

    // Children map: parent segment id → child segment ids (declaration order).
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in segs {
        if let Some(p) = &s.parent {
            children.entry(p.segment).or_default().push(s.id);
        }
    }

    // Per-segment geometry (needs parent's distal for omitted proximal).
    let geom_of = |s: &super::parse::Segment| -> Result<SegmentGeom, NeuromlError> {
        // `Point3d` is `Copy` — the parent's distal / the segment's own distal
        // is copied verbatim (no manual field rebuild).
        let distal = &s.distal;
        let proximal: Point3d = match &s.proximal {
            Some(p) => *p,
            None => match &s.parent {
                Some(p) => segs[index[&p.segment]].distal,
                // Root with no proximal (SWC single point): zero length → use
                // the distal point itself (effective length = diameter below).
                None => *distal,
            },
        };
        let mut length_um = dist(&proximal, distal);
        let diameter_um = (proximal.diameter + distal.diameter) / 2.0;
        if diameter_um <= 0.0 {
            return Err(NeuromlError::parse(format!(
                "discretize: segment {} has non-positive diameter",
                s.id
            )));
        }
        if length_um <= 0.0 {
            // Single-point soma → treat as a cylinder of length = diameter.
            length_um = diameter_um;
        }
        Ok(SegmentGeom {
            length_um,
            diameter_um,
            prox_x: proximal.x,
            prox_y: proximal.y,
            prox_z: proximal.z,
            distal_x: distal.x,
            distal_y: distal.y,
            distal_z: distal.z,
        })
    };

    // Breadth-first traversal from the root, splitting each segment.
    let mut compartments: Vec<Compartment> = Vec::new();
    // segment id → global index of its **last** compartment.
    let mut last_comp: HashMap<usize, usize> = HashMap::new();

    let mut queue: Vec<usize> = vec![root_id];
    let mut head = 0usize;
    while head < queue.len() {
        let seg_id = queue[head];
        head += 1;
        let seg = &segs[index[&seg_id]];
        let geom = geom_of(seg)?;

        let parent_comp = seg.parent.as_ref().map(|p| {
            // Parent segment must already be processed (BFS from root).
            last_comp[&p.segment]
        });

        let lambda = space_constant_um(geom.diameter_um, params.rm_ohm_cm2, params.ra_ohm_cm);
        let target = params.d_lambda * lambda;
        let n = ((geom.length_um / target).ceil() as usize).max(1);

        let comp_len = geom.length_um / n as f32;
        let d = geom.diameter_um;
        let area = PI * d * comp_len;
        // area(µm²) → cm²: ×1e-8.
        let area_cm2 = area * 1e-8;
        // C[nF] = cm[µF/cm²]·A[cm²]·1e3, g[µS] = A/Rm[Ω·cm²]·1e6.
        let cm_nf = params.cm_uf_per_cm2 * area_cm2 * 1e3;
        let g_leak_us = area_cm2 / params.rm_ohm_cm2 * 1e6;
        // R[MΩ] = 4·Ra[Ω·cm]·L[µm]/(π·d²[µm²]) · 1e-2.
        let r_axial_mohm = 4.0 * params.ra_ohm_cm * comp_len / (PI * d * d) * 1e-2;

        // Centre position of compartment j: `proximal + (j+0.5)/n·(distal−proximal)`.
        let axis = (
            geom.distal_x - geom.prox_x,
            geom.distal_y - geom.prox_y,
            geom.distal_z - geom.prox_z,
        );

        for j in 0..n {
            let parent = if j == 0 {
                parent_comp
            } else {
                Some(compartments.len() - 1)
            };
            let f = (j as f32 + 0.5) / n as f32;
            let (cx, cy, cz) = (
                geom.prox_x + f * axis.0,
                geom.prox_y + f * axis.1,
                geom.prox_z + f * axis.2,
            );
            // Path distance to the centre: parent centre + half parent +
            // half own length; within a chain the step is exactly comp_len.
            let path_um = match parent {
                Some(p) => {
                    let pc = &compartments[p];
                    pc.path_um + pc.length_um * 0.5 + comp_len * 0.5
                }
                None => comp_len * 0.5,
            };
            compartments.push(Compartment {
                parent,
                segment_id: seg_id,
                length_um: comp_len,
                diameter_um: d,
                area_um2: area,
                cm_nf,
                g_leak_us,
                r_axial_mohm,
                x: cx,
                y: cy,
                z: cz,
                path_um,
            });
        }
        last_comp.insert(seg_id, compartments.len() - 1);

        // Branch point: if this segment feeds more than one child, insert a
        // zero-volume node at its distal end. This keeps the parent's distal
        // half-resistance a **single shared** resistor (each child connects to
        // the branch node via its own proximal half), exactly as NEURON models
        // a `connect(child, parent(1.0))` branch point. Flattening two children
        // straight onto the parent's last compartment would duplicate the
        // parent half and over-couple the children (a real ~1% error).
        if children.get(&seg_id).is_some_and(|kids| kids.len() >= 2) {
            let bp = compartments.len();
            let parent_comp_idx = last_comp[&seg_id];
            let parent_c = &compartments[parent_comp_idx];
            compartments.push(Compartment {
                parent: Some(parent_comp_idx),
                segment_id: seg_id,
                length_um: 0.0,
                diameter_um: geom.diameter_um,
                area_um2: 0.0,
                cm_nf: 0.0,
                g_leak_us: 0.0,
                r_axial_mohm: 0.0,
                // Branch node sits at the segment's distal end.
                x: geom.distal_x,
                y: geom.distal_y,
                z: geom.distal_z,
                path_um: parent_c.path_um + parent_c.length_um * 0.5,
            });
            last_comp.insert(seg_id, bp);
        }

        if let Some(kids) = children.get(&seg_id) {
            queue.extend_from_slice(kids);
        }
    }

    // BFS starts with the root segment, so its first compartment is index 0.
    Ok(CompartmentTree {
        compartments,
        root: 0,
    })
}

/// Passive cable solver over a [`CompartmentTree`] — implicit Euler (Hines).
///
/// Solves `C_i·dV_i/dt = -g_leak,i·(V_i − E) + Σ_j (V_j − V_i)/R_ij + I_i` with
/// backward Euler for the linear passive system, so it is unconditionally
/// stable. The tree-structured linear system is solved in O(n) by the Hines
/// algorithm (upward elimination + downward back-substitution).
#[derive(Clone)]
pub(crate) struct CableSolver {
    tree: CompartmentTree,
    /// Membrane potential per compartment (mV).
    v: Vec<f32>,
    e_leak: f32,
    /// Fixed step (ms). Only read by the test-only [`step`](Self::step);
    /// production stepping goes through [`step_with_extra`](Self::step_with_extra).
    #[allow(dead_code)]
    dt: f32,
    /// Children adjacency: `children[i]` = compartments whose parent is `i`.
    children: Vec<Vec<usize>>,
    /// Topological order (root first) for the back-substitution pass.
    order: Vec<usize>,
    /// Axial conductance to parent (µS); 0 for the root. **Geometry-only** —
    /// built once at construction, reused every step (no divisions per step).
    g_parent: Vec<f32>,
    /// Σ g_parent[child] over direct children (µS). Geometry-only.
    g_children_sum: Vec<f32>,
    /// Reused implicit-solve buffers (sized `n` — no per-step allocations).
    diag: Vec<f32>,
    rhs: Vec<f32>,
    /// Scratch for the back-substitution; swapped with `v` each step (zero
    /// copy).
    v_new: Vec<f32>,
}

impl CableSolver {
    /// Build a solver at rest (`V = e_leak` everywhere).
    pub fn new(tree: CompartmentTree, e_leak: f32, dt: f32) -> Self {
        let n = tree.len();
        let mut children = vec![Vec::new(); n];
        for (i, c) in tree.compartments.iter().enumerate() {
            if let Some(p) = c.parent {
                children[p].push(i);
            }
        }
        // BFS topological order from the root.
        let mut order = Vec::with_capacity(n);
        let mut queue = vec![tree.root];
        let mut head = 0usize;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            order.push(i);
            queue.extend_from_slice(&children[i]);
        }
        // Cache the axial conductances (they never change with the state).
        let mut g_parent = vec![0.0f32; n];
        for (i, c) in tree.compartments.iter().enumerate() {
            if c.parent.is_some() {
                g_parent[i] = 1.0 / tree.junction_resistance(i);
            }
        }
        let mut g_children_sum = vec![0.0f32; n];
        for (i, childs) in children.iter().enumerate() {
            g_children_sum[i] = childs.iter().map(|&c| g_parent[c]).sum();
        }
        Self {
            tree,
            v: vec![e_leak; n],
            e_leak,
            dt,
            children,
            order,
            g_parent,
            g_children_sum,
            diag: vec![0.0; n],
            rhs: vec![0.0; n],
            v_new: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Index of the root compartment (the soma). Reached from production via
    /// [`ActiveCableSolver::root`]; kept un-gated so the method survives when
    /// `ActiveCableSolver` is compiled without its own gating.
    #[allow(dead_code)]
    pub fn root(&self) -> usize {
        self.tree.root
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.v[idx]
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.v[idx] = value;
    }

    /// One implicit-Euler (Hines) step of the passive cable. Test-only —
    /// production stepping goes through [`step_with_extra`](Self::step_with_extra)
    /// (which carries the extra conductance and an explicit `dt`).
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    #[cfg(test)]
    pub fn step(&mut self, i_inj: &[f32]) {
        self.step_with_extra(i_inj, &[], &[], self.dt);
    }

    /// Like [`step`](Self::step) but adds per-compartment extra conductances
    /// `g_extra` (µS) and `g_extra·E_rev` (`g_e_extra`, nA) — the semi-implicit
    /// active-channel contribution (gates evaluated at the old voltage). Empty
    /// slices mean no extra conductance.
    ///
    /// # Panics
    /// If `i_inj` (or, when non-empty, `g_extra`/`g_e_extra`) length mismatch.
    pub fn step_with_extra(&mut self, i_inj: &[f32], g_extra: &[f32], g_e_extra: &[f32], dt: f32) {
        assert_eq!(i_inj.len(), self.tree.len(), "i_inj length mismatch");
        assert!(g_extra.is_empty() || g_extra.len() == self.tree.len());
        assert!(g_e_extra.is_empty() || g_e_extra.len() == self.tree.len());
        let n = self.tree.len();
        if n == 0 {
            return;
        }
        let e_leak = self.e_leak;
        let has_ge = !g_extra.is_empty();
        let has_gee = !g_e_extra.is_empty();

        // Diagonal (µS) and RHS (nA) of the implicit system A·V = b. All
        // buffers are reused across steps (zero per-step allocations); the
        // geometry-only conductances come from the cached `g_parent` /
        // `g_children_sum`.
        for i in 0..n {
            let c = &self.tree.compartments[i];
            let inv_dt = c.cm_nf / dt; // nF/ms = µS
            let ge = if has_ge { g_extra[i] } else { 0.0 };
            let gee = if has_gee { g_e_extra[i] } else { 0.0 };
            self.diag[i] = inv_dt + c.g_leak_us + self.g_parent[i] + self.g_children_sum[i] + ge;
            self.rhs[i] = inv_dt * self.v[i] + c.g_leak_us * e_leak + i_inj[i] + gee;
        }

        // Hines elimination: children before parents (reverse BFS order).
        for &i in self.order.iter().rev() {
            for &child in &self.children[i] {
                let factor = self.g_parent[child] / self.diag[child];
                self.diag[i] -= self.g_parent[child] * factor;
                self.rhs[i] += factor * self.rhs[child];
            }
        }

        // Back-substitution: parents before children (BFS order), into the
        // scratch `v_new`, then zero-copy swap with `v`.
        for &i in &self.order {
            let up = match self.tree.compartments[i].parent {
                Some(p) => self.g_parent[i] * self.v_new[p],
                None => 0.0,
            };
            self.v_new[i] = (self.rhs[i] + up) / self.diag[i];
        }
        std::mem::swap(&mut self.v, &mut self.v_new);
    }
}

/// An active (voltage-gated) channel attached to one compartment: HH gates
/// (data-driven [`GateKinetics`]) + reversal potential + absolute max
/// conductance (µS). Gate state is integrated with **exponential Euler** at the
/// old voltage, then the resulting conductance enters the Hines solve
/// (semi-implicit).
#[derive(Clone)]
pub(crate) struct ActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// Reversal potential, mV.
    pub e_rev: f32,
    /// Absolute max conductance, µS (already density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    gate_state: Vec<f32>,
}

impl ActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates` (1.0 = no temperature
    /// scaling, the NeuroML default without `q10Settings`).
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "ActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            gate_state,
        }
    }
}

/// Steady-state of a gate at `v`: `α/(α+β)` (neutral fallback 0.5).
///
/// Ca-dependent gates (Poirazi `cagk`/`kca`) are evaluated at the initial
/// Ca²⁺ concentration `1e-4 mM` (the resting `decayingPoolConcentrationModel`
/// value) rather than `ca = 0`, so their opening gate starts at the physical
/// resting occupancy. V-only gates ignore `ca`, so this is a no-op for them
/// (and for `gateHHInstantaneous`, whose value is recomputed algebraically
/// every step).
fn gate_steady_state(kinetics: GateKinetics, v: f32) -> f32 {
    let (a, b) = kinetics.alpha_beta_at(v, 1e-4);
    let sum = a + b;
    if sum > 1e-12 {
        a / sum
    } else {
        0.5
    }
}

/// One exponential-Euler step of a gate: `q' = q_inf + (q−q_inf)·exp(−dt·(α+β)·q10)`.
/// `gateHHInstantaneous` is algebraic: `q = steady_state(v, ca)` (no ODE).
fn gate_step(kinetics: GateKinetics, q: f32, v: f32, ca: f32, dt: f32, q10: f32) -> f32 {
    match kinetics {
        GateKinetics::Instantaneous { steady_state } => steady_state.eval_at(v, ca),
        kinetics => {
            // `alpha_beta_at` (not `alpha_beta`): Ca-dependent `Rates`/`TauInf`
            // gates (Poirazi cagk/kca) must read the compartment's actual
            // [Ca²⁺] — `alpha_beta(v)` would pin them to ca = 0.
            let (a, b) = kinetics.alpha_beta_at(v, ca);
            let sum = a + b;
            let q_inf = if sum > 1e-12 { a / sum } else { 0.5 };
            let decay = (-dt * sum.max(1e-12) * q10).exp();
            q_inf + (q - q_inf) * decay
        }
    }
}

/// A Markov (kinetic-scheme, `gateKS`/`gateFractional`) channel attached to one
/// compartment: gates integrated as a master equation (matrix exponential) +
/// reversal + absolute max conductance (µS). Parallel to [`ActiveChannel`]
/// (HH gates) — the cable solver carries both lists.
#[derive(Clone)]
pub(crate) struct MarkovActiveChannel {
    pub gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    pub e_rev: f32,
    pub g_max_us: f32,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
}

impl MarkovActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "MarkovActiveChannel: q10s length");
        let mut occ = Vec::new();
        let mut gate_off = Vec::with_capacity(gates.len());
        for gate in &gates {
            gate_off.push(occ.len());
            occ.extend(markov_steady_state(gate, v_rest));
        }
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            occ,
            gate_off,
        }
    }

    /// Advance every gate one step at `v` and return the open fraction
    /// `Π q_gate^power`. Each gate uses its own Q10 (per-gate `q10Settings`).
    fn step_open(&mut self, v: f32, dt: f32) -> f32 {
        let mut fopen = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            fopen *= q.powi(gate.power as i32);
        }
        fopen
    }
}

/// Exact [`sample_binomial`] draw — the **second waiting time** (geometric)
/// method: count successes via the number of failures before each success
/// (`geom(p)` waiting times), which is exact for all `n`/`p` and runs in
/// O(successes) for small `p` (the open probability at rest is tiny, so this
/// is the fast path) / O(failures) for large `p` via the swap.
///
/// `rng.gen::<f64>()` is uniform in `[0, 1)`; we map to `(0, 1]` so `ln` is
/// never `-inf` (a `0.0` draw would produce an infinite waiting time).
fn sample_binomial(rng: &mut StdRng, n: usize, p: f32) -> usize {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    // Sample failures when p > 1/2 (fewer geometric draws), then complement.
    if p > 0.5 {
        return n - sample_binomial(rng, n, 1.0 - p);
    }
    let ln_q = (1.0 - p as f64).ln();
    let mut successes = 0usize;
    let mut trials = 0i64;
    loop {
        // Failures before the next success: `floor(ln(u) / ln(1-p))`.
        let u = 1.0 - rng.gen::<f64>(); // (0, 1]
        let k = (u.ln() / ln_q).floor() as i64;
        trials += k + 1;
        if trials > n as i64 {
            break;
        }
        successes += 1;
        if successes == n {
            break;
        }
    }
    successes
}

/// A **stochastic** kinetic-scheme channel: a population of `number` discrete
/// ion channels (`channelPopulation`), each described by the same Markov gates.
/// The master equation still yields the per-channel occupancy (mean-field), but
/// the open fraction used for the conductance is a **binomial sample** over the
/// discrete channel count — `n_open ~ Binomial(number, p_open)`, so the
/// delivered fraction is `n_open / number`. This is the canonical source of
/// channel noise (∝ 1/√N) in small neurons/axons.
///
/// The mean of the sample equals the deterministic open fraction (so the
/// `channelDensity` and `channelPopulation` models agree on average); the
/// variance `p_open·(1−p_open)/number` is the channel noise. A seeded
/// [`StdRng`] makes the noise reproducible; `CablePopulation` re-seeds each
/// neuron from a per-neuron salt so the `n` copies of a cell diverge.
#[derive(Clone)]
pub(crate) struct StochasticMarkovActiveChannel {
    gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    q10s: Vec<f32>,
    e_rev: f32,
    /// Total conductance `single_channel_conductance × number` (µS) — the
    /// deterministic upper bound; the stochastic fraction scales it each step.
    g_max_us: f32,
    /// Number of discrete channels.
    n_channels: usize,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
    rng: StdRng,
    /// Per-channel base seed (from the compiler) — [`CablePopulation`] mixes
    /// the neuron index into it so each neuron's stream is independent.
    base_seed: u64,
}

impl StochasticMarkovActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV), and its RNG seeded with `seed`. `q10s` is
    /// parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        n_channels: usize,
        v_rest: f32,
        seed: u64,
    ) -> Self {
        debug_assert_eq!(
            gates.len(),
            q10s.len(),
            "StochasticMarkovActiveChannel: q10s"
        );
        let mut occ = Vec::new();
        let mut gate_off = Vec::with_capacity(gates.len());
        for gate in &gates {
            gate_off.push(occ.len());
            occ.extend(markov_steady_state(gate, v_rest));
        }
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            n_channels,
            occ,
            gate_off,
            rng: StdRng::seed_from_u64(seed),
            base_seed: seed,
        }
    }

    /// Re-seed the RNG — used by [`CablePopulation`] to give every neuron an
    /// independent noise stream from one shared channel template.
    pub fn reseed_for_neuron(&mut self, neuron: usize) {
        self.rng = StdRng::seed_from_u64(mix_seed(self.base_seed, neuron as u64));
    }

    /// Advance every gate one step at `v` and return the **stochastic** open
    /// fraction `n_open / n_channels`, where `n_open ~ Binomial(n_channels,
    /// p_open)` and `p_open = Π q_gate^power` is the deterministic open
    /// probability (exact for a single `gateKS`, a well-defined approximation
    /// for a multi-gate channel).
    fn step_open(&mut self, v: f32, dt: f32) -> f32 {
        let mut p_open = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            p_open *= q.powi(gate.power as i32);
        }
        let n_open = sample_binomial(&mut self.rng, self.n_channels, p_open);
        n_open as f32 / self.n_channels as f32
    }
}

/// A **stochastic** Nernst (Ca²⁺) channel population
/// (`channelPopulationNernst`): `number` discrete channels whose combined
/// reversal is the **Nernst potential** of the compartment's [Ca²⁺] (dynamic)
/// and whose Ca²⁺ current feeds the compartment's pool. This is the
/// discrete-channel analogue of [`NernstActiveChannel`] (just as
/// [`StochasticMarkovActiveChannel`] is of [`ActiveChannel`]): the same
/// binomial open-fraction sampling, but `erev` is recomputed from [Ca²⁺] each
/// step.
#[derive(Clone)]
pub(crate) struct StochasticNernstActiveChannel {
    gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    ca_ext_mm: f32,
    /// Total conductance `single_channel_conductance × number` (µS).
    g_max_us: f32,
    /// Number of discrete channels.
    n_channels: usize,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
    rng: StdRng,
    base_seed: u64,
}

impl StochasticNernstActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV), and its RNG seeded with `seed`. `q10s` is
    /// parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        n_channels: usize,
        v_rest: f32,
        seed: u64,
    ) -> Self {
        debug_assert_eq!(
            gates.len(),
            q10s.len(),
            "StochasticNernstActiveChannel: q10s"
        );
        let mut occ = Vec::new();
        let mut gate_off = Vec::with_capacity(gates.len());
        for gate in &gates {
            gate_off.push(occ.len());
            occ.extend(markov_steady_state(gate, v_rest));
        }
        Self {
            gates,
            q10s,
            ca_ext_mm,
            g_max_us,
            n_channels,
            occ,
            gate_off,
            rng: StdRng::seed_from_u64(seed),
            base_seed: seed,
        }
    }

    /// Re-seed the RNG — used by [`CablePopulation`] to give every neuron an
    /// independent noise stream from one shared channel template.
    pub fn reseed_for_neuron(&mut self, neuron: usize) {
        self.rng = StdRng::seed_from_u64(mix_seed(self.base_seed, neuron as u64));
    }

    /// Advance every gate one step and return `(fopen, e_rev_mv)`: the
    /// **stochastic** open fraction `n_open / n_channels` and the Nernst
    /// reversal of `ca_int_mm` against `ca_ext_mm`.
    fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32, temp_k: f32) -> (f32, f32) {
        let mut p_open = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            p_open *= q.powi(gate.power as i32);
        }
        let n_open = sample_binomial(&mut self.rng, self.n_channels, p_open);
        (
            n_open as f32 / self.n_channels as f32,
            nernst_reversal_mv(ca_int_mm, self.ca_ext_mm, temp_k),
        )
    }
}

/// Ca²⁺ reversal potential from the Nernst equation (LEMS
/// `channelPopulationNernst`/`channelDensityNernst`):
/// `E = (R·T/(z·F))·ln(ca_ext/ca_int)` in mV. `ca_ext <= 0` → 0 (LEMS guard).
fn nernst_reversal_mv(ca_int_mm: f32, ca_ext_mm: f32, temp_k: f32) -> f32 {
    if ca_ext_mm <= 0.0 {
        return 0.0;
    }
    let ca_int = ca_int_mm.max(1e-12);
    let rt_over_zf_mv = R_GAS * temp_k / (Z_CA * FARADAY) * 1000.0;
    rt_over_zf_mv * (ca_ext_mm / ca_int).ln()
}

/// Per-compartment Ca²⁺ pool: the ODE `dC/dt = iCa·flux_scale − (C−rest)/tau`,
/// integrated with exact exponential decay over the step (linear ODE, constant
/// `iCa`). Concentrations are **mM** (= mol/m³), `iCa` is **nA**, `flux_scale`
/// is precomputed from the compartment geometry (mM/ms per nA).
#[derive(Clone, Copy)]
pub(crate) struct CaPool {
    pub resting_mm: f32,
    pub tau_ms: f32,
    pub flux_scale: f32,
}

impl CaPool {
    /// One exact step of the pool ODE at constant `i_ca_nA` over `dt` (ms).
    fn step(&self, ca: f32, i_ca_na: f32, dt: f32) -> f32 {
        let tau = self.tau_ms.max(1e-6);
        let decay = (-dt / tau).exp();
        let steady = self.resting_mm + i_ca_na * self.flux_scale * tau;
        // LEMS clamps concentration at 0.
        (steady + (ca - steady) * decay).max(0.0)
    }
}

/// Precompute the decaying-pool flux scale `1/(2·F·V_shell)` for a compartment
/// of membrane area `area_um2` (µm²) and shell thickness `shell_thickness_um`
/// (µm). LEMS computes the shell as a spherical shell whose outer radius is
/// `sqrt(surfaceArea/4π)`. Result is **mM/ms per nA**. Test-only (the compiler
/// lowers NeuroML pools via [`thin_shell_pool_flux_scale`]).
#[cfg(test)]
pub(crate) fn decaying_pool_flux_scale(area_um2: f32, shell_thickness_um: f32) -> f32 {
    let surface_m2 = area_um2 * 1e-12;
    let r_eff = (surface_m2 / (4.0 * PI)).sqrt();
    let r_in = (r_eff - shell_thickness_um * 1e-6).max(0.0);
    let shell_vol_m3 = (4.0 * PI / 3.0) * (r_eff.powi(3) - r_in.powi(3));
    // dC/dt [mM/ms] = iCa[nA]·1e-12 / (2·F·V_shell[m³])
    1e-12 / (2.0 * FARADAY * shell_vol_m3.max(f32::MIN_POSITIVE))
}

/// Precompute the **thin-shell** pool flux scale `1/(2·F·(SA·thickness))` —
/// the exact form of custom LEMS concentration models like
/// `PoiraziCaConcentrationModel` (`iCa/(2·F·surfaceArea·shellThickness)`, the
/// published Poirazi capool), whose shell volume is `membrane area × shell
/// thickness`, **not** the exact spherical shell of the standard
/// `decayingPoolConcentrationModel`. Result is mM/ms per nA.
pub(crate) fn thin_shell_pool_flux_scale(area_um2: f32, shell_thickness_um: f32) -> f32 {
    // V[m³] = SA[m²]·thickness[m]; dC/dt [mM/ms] = iCa[nA]·1e-12/(2·F·V[m³]).
    let vol_m3 = (area_um2 * 1e-12) * (shell_thickness_um * 1e-6);
    1e-12 / (2.0 * FARADAY * vol_m3.max(f32::MIN_POSITIVE))
}

/// An HH-gated Ca²⁺ channel whose reversal potential is the **Nernst
/// potential** of the compartment's [Ca²⁺] (recomputed each step) rather than a
/// fixed `erev`. Parallel to [`ActiveChannel`] (fixed `erev`) — the cable solver
/// carries both lists, and the Nernst channel also feeds its current back into
/// the compartment's Ca²⁺ pool (`iCa`).
#[derive(Clone)]
pub(crate) struct NernstActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    pub ca_ext_mm: f32,
    /// Absolute max conductance, µS (density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    gate_state: Vec<f32>,
}

impl NernstActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "NernstActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            ca_ext_mm,
            g_max_us,
            gate_state,
        }
    }

    /// Advance every gate one step at `v` and return `(fopen, e_rev_mv)`, where
    /// the reversal is the Nernst potential of `ca_int_mm` (the compartment's
    /// current [Ca²⁺]) against `ca_ext_mm`. Each gate uses its own Q10.
    fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32, temp_k: f32) -> (f32, f32) {
        let mut fopen = 1.0f32;
        for (((kinetics, power), q10), q) in self
            .gates
            .iter()
            .zip(self.q10s.iter())
            .zip(self.gate_state.iter_mut())
        {
            // Ca-dependent gates (Instantaneous h2, kca tau/ss) read the
            // compartment's [Ca²⁺].
            *q = gate_step(kinetics.clone(), *q, v, ca_int_mm, dt, *q10);
            fopen *= q.powi(*power as i32);
        }
        (fopen, nernst_reversal_mv(ca_int_mm, self.ca_ext_mm, temp_k))
    }
}

/// A Ca²⁺ channel whose current is the **Goldman–Hodgkin–Katz** flux equation
/// (`channelDensityGHK2`, e.g. Poirazi 2003 `cal`/`cat`): `I = g·ghk(v, ci, co)`
/// — not ohmic, so there is no reversal potential; the current is computed
/// from the compartment's dynamic [Ca²⁺] each step and fed back into the pool.
#[derive(Clone)]
pub(crate) struct GhkActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    pub ca_ext_mm: f32,
    /// Absolute max conductance, µS (density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    gate_state: Vec<f32>,
}

impl GhkActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "GhkActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            ca_ext_mm,
            g_max_us,
            gate_state,
        }
    }

    /// Advance every gate one step at `v` and return the open fraction. The
    /// GHK current is computed separately from `v` and the compartment's
    /// [Ca²⁺] (see [`ghk_current_na`]).
    fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32) -> f32 {
        let mut fopen = 1.0f32;
        for (((kinetics, power), q10), q) in self
            .gates
            .iter()
            .zip(self.q10s.iter())
            .zip(self.gate_state.iter_mut())
        {
            // Ca-dependent gates (Instantaneous h2, kca tau/ss) read the
            // compartment's [Ca²⁺].
            *q = gate_step(kinetics.clone(), *q, v, ca_int_mm, dt, *q10);
            fopen *= q.powi(*power as i32);
        }
        fopen
    }
}

/// Goldman–Hodgkin–Katz flux current (nA per µS of conductance) — the exact
/// form in Poirazi's `cal.mod`/`cat.mod` (KTF/efun convention):
/// `f = (25/293.15)·T/2`, `nu = v/f`, `ghk = −f·(1 − (ci/co)·e^nu)·efun(nu)`,
/// `efun(z) = 1−z/2` for `|z|<1e-4` else `z/(e^z−1)`. Returns the **inward-
/// positive** current (the NHN sign convention used by the pool and the
/// membrane RHS: a negative `ghk` = inward Ca → positive flux).
pub(crate) fn ghk_current_na(v_mv: f32, ci_mm: f32, co_mm: f32, temp_k: f32) -> f32 {
    if co_mm <= 0.0 {
        return 0.0;
    }
    let f = (25.0 / 293.15) * temp_k / 2.0;
    let nu = v_mv / f;
    let efun = if nu.abs() < 1e-4 {
        1.0 - nu / 2.0
    } else {
        nu / (nu.exp() - 1.0)
    };
    // Inward-positive: −ghk = f·(1 − (ci/co)·e^nu)·efun(nu).
    f * (1.0 - (ci_mm / co_mm) * nu.exp()) * efun
}

/// Passive cable + active HH channels, integrated semi-implicitly: gates at
/// the old voltage (exponential Euler), then the Hines solve over the linear
/// system with the active conductances added to the diagonal.
#[derive(Clone)]
pub(crate) struct ActiveCableSolver {
    passive: CableSolver,
    /// Active HH channels per compartment.
    channels: Vec<Vec<ActiveChannel>>,
    /// Active Markov (kinetic-scheme) channels per compartment.
    markov_channels: Vec<Vec<MarkovActiveChannel>>,
    /// Stochastic (discrete-population) channels per compartment.
    stochastic_channels: Vec<Vec<StochasticMarkovActiveChannel>>,
    /// Nernst (Ca²⁺) HH channels per compartment — reversal computed from
    /// the compartment's [Ca²⁺] each step.
    nernst_channels: Vec<Vec<NernstActiveChannel>>,
    /// Stochastic Nernst (Ca²⁺) channel populations per compartment.
    stochastic_nernst_channels: Vec<Vec<StochasticNernstActiveChannel>>,
    /// GHK (Ca²⁺, `channelDensityGHK2`) channels per compartment — current is
    /// the Goldman–Hodgkin–Katz flux (not ohmic), computed from the
    /// compartment's [Ca²⁺] each step.
    ghk_channels: Vec<Vec<GhkActiveChannel>>,
    /// Internal [Ca²⁺] per compartment (mM).
    ca: Vec<f32>,
    /// Ca²⁺ pool (ODE params) per compartment; `None` = no tracked Ca.
    ca_pools: Vec<Option<CaPool>>,
    dt: f32,
    /// Temperature for the Nernst reversal (K).
    temp_k: f32,
    /// Per-compartment passive leak override (µS) — from per-group
    /// `channelDensity` leak densities (the model's declared leak, which may
    /// differ between segment groups). `0.0` = use the tree's uniform
    /// `g_leak_us` (from `CableParams::rm_ohm_cm2`). Parallel to `leak_rev`.
    leak_us: Vec<f32>,
    /// Per-compartment leak reversal (mV), parallel to `leak_us`.
    leak_rev: Vec<f32>,
    /// Reused per-step buffers (sized `n`, zeroed each step — no per-step
    /// allocations for active cables).
    g_extra: Vec<f32>,
    g_e_extra: Vec<f32>,
    /// Reused Ca-current buffer (nA per compartment, zeroed each step).
    i_ca: Vec<f32>,
}

impl ActiveCableSolver {
    /// Build a solver at rest (`V = e_leak`), with no active channels.
    pub fn new(tree: CompartmentTree, e_leak: f32, dt: f32, celsius: f32) -> Self {
        let n = tree.len();
        let passive = CableSolver::new(tree, e_leak, dt);
        Self {
            passive,
            channels: (0..n).map(|_| Vec::new()).collect(),
            markov_channels: (0..n).map(|_| Vec::new()).collect(),
            stochastic_channels: (0..n).map(|_| Vec::new()).collect(),
            nernst_channels: (0..n).map(|_| Vec::new()).collect(),
            stochastic_nernst_channels: (0..n).map(|_| Vec::new()).collect(),
            ghk_channels: (0..n).map(|_| Vec::new()).collect(),
            ca: vec![0.0; n],
            ca_pools: (0..n).map(|_| None).collect(),
            dt,
            temp_k: celsius + 273.15,
            leak_us: vec![0.0; n],
            leak_rev: vec![0.0; n],
            g_extra: vec![0.0; n],
            g_e_extra: vec![0.0; n],
            i_ca: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.passive.len()
    }

    /// Index of the root compartment (the soma). Test-only — production reads
    /// the soma index from the shared tree (`CablePopulation::new`).
    #[cfg(test)]
    pub fn root(&self) -> usize {
        self.passive.root()
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.passive.v(idx)
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.passive.set_v(idx, value);
    }

    /// Replace the active channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        assert!(comp < self.len(), "set_channels: {comp} out of range");
        self.channels[comp] = channels;
    }

    /// Replace the active Markov channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        assert!(
            comp < self.len(),
            "set_markov_channels: {comp} out of range"
        );
        self.markov_channels[comp] = channels;
    }

    /// Replace the stochastic (discrete-population) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_stochastic_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticMarkovActiveChannel>,
    ) {
        assert!(
            comp < self.len(),
            "set_stochastic_channels: {comp} out of range"
        );
        self.stochastic_channels[comp] = channels;
    }

    /// Replace the Nernst (Ca²⁺) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_nernst_channels(&mut self, comp: usize, channels: Vec<NernstActiveChannel>) {
        assert!(
            comp < self.len(),
            "set_nernst_channels: {comp} out of range"
        );
        self.nernst_channels[comp] = channels;
    }
    /// Replace the stochastic Nernst (Ca²⁺) channel populations of one
    /// compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_stochastic_nernst_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticNernstActiveChannel>,
    ) {
        assert!(
            comp < self.len(),
            "set_stochastic_nernst_channels: {comp} out of range"
        );
        self.stochastic_nernst_channels[comp] = channels;
    }

    /// Replace the GHK (Ca²⁺) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_ghk_channels(&mut self, comp: usize, channels: Vec<GhkActiveChannel>) {
        assert!(comp < self.len(), "set_ghk_channels: {comp} out of range");
        self.ghk_channels[comp] = channels;
    }

    /// Override the passive leak of one compartment with an explicit
    /// conductance `g_us` (µS) at reversal `e_rev` (mV) — the per-group leak
    /// densities of the model. The tree's uniform `g_leak_us` must have been
    /// zeroed for this compartment (compile.rs) so the two don't double-count.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_leak(&mut self, comp: usize, g_us: f32, e_rev: f32) {
        assert!(comp < self.len(), "set_leak: {comp} out of range");
        self.leak_us[comp] = g_us;
        self.leak_rev[comp] = e_rev;
    }

    /// Attach a Ca²⁺ pool to one compartment and initialise its [Ca²⁺] to
    /// `resting` (the species `initialConcentration`).
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_ca_pool(&mut self, comp: usize, pool: CaPool, initial_mm: f32) {
        assert!(comp < self.len(), "set_ca_pool: {comp} out of range");
        self.ca_pools[comp] = Some(pool);
        self.ca[comp] = initial_mm;
    }

    /// Current internal [Ca²⁺] (mM) of a compartment.
    pub fn ca(&self, comp: usize) -> f32 {
        self.ca[comp]
    }

    /// One semi-implicit step with the solver's fixed `dt`.
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step(&mut self, i_inj: &[f32]) {
        self.step_dt(i_inj, self.dt);
    }

    /// One semi-implicit step with an explicit `dt` (for adaptive stepping).
    ///
    /// Gates are integrated at the **old** voltage (exponential Euler / exact
    /// matrix exponential), then the voltage solve is implicit — the same
    /// state convention as NEURON fixed-step (`secondorder=0`): the
    /// `markov_cable_matches_neuron_ks3` experiment showed that a
    /// fully-implicit (Picard) gate solve *worsens* parity (0.155 vs
    /// 0.014 mV), i.e. NEURON evaluates rates at the step-start voltage.
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step_dt(&mut self, i_inj: &[f32], dt: f32) {
        let n = self.len();
        self.g_extra.fill(0.0);
        self.g_e_extra.fill(0.0);
        self.i_ca.fill(0.0);
        for i in 0..n {
            // Per-group passive leak (constant conductance, semi-implicit like
            // the active channels).
            if self.leak_us[i] > 0.0 {
                self.g_extra[i] += self.leak_us[i];
                self.g_e_extra[i] += self.leak_us[i] * self.leak_rev[i];
            }
            let v = self.passive.v(i);
            for ch in &mut self.channels[i] {
                let mut fopen = 1.0f32;
                for (((kinetics, power), q10), q) in ch
                    .gates
                    .iter()
                    .zip(ch.q10s.iter())
                    .zip(ch.gate_state.iter_mut())
                {
                    // Ca-dependent gates (Instantaneous h2, kca tau/ss) read
                    // the compartment's [Ca²⁺].
                    *q = gate_step(kinetics.clone(), *q, v, self.ca[i], dt, *q10);
                    fopen *= q.powi(*power as i32);
                }
                let g_ch = ch.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * ch.e_rev;
            }
            for mc in &mut self.markov_channels[i] {
                let fopen = mc.step_open(v, dt);
                let g_ch = mc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * mc.e_rev;
            }
            for sc in &mut self.stochastic_channels[i] {
                // Stochastic: `fopen` is `n_open / n_channels`, so
                // `g_max_us · fopen = single_channel_conductance · n_open`.
                let fopen = sc.step_open(v, dt);
                let g_ch = sc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * sc.e_rev;
            }
            for nc in &mut self.nernst_channels[i] {
                // Reversal from the current [Ca²⁺] (Nernst); the Ca current also
                // feeds the compartment's Ca²⁺ pool.
                let (fopen, e_rev) = nc.step_open(v, dt, self.ca[i], self.temp_k);
                let g_ch = nc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * e_rev;
                self.i_ca[i] += g_ch * (e_rev - v);
            }
            for snc in &mut self.stochastic_nernst_channels[i] {
                // Same as `nernst_channels`, but `fopen` is the binomial sample
                // `n_open / n_channels` (channel noise).
                let (fopen, e_rev) = snc.step_open(v, dt, self.ca[i], self.temp_k);
                let g_ch = snc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * e_rev;
                self.i_ca[i] += g_ch * (e_rev - v);
            }
            for gh in &mut self.ghk_channels[i] {
                // GHK flux current (not ohmic): explicit at the old V/[Ca],
                // like NEURON's cnexp `BREAKPOINT`. The inward-positive
                // current feeds both the membrane RHS and the Ca²⁺ pool.
                let fopen = gh.step_open(v, dt, self.ca[i]);
                let g_ch = gh.g_max_us * fopen;
                let i_in = g_ch * ghk_current_na(v, self.ca[i], gh.ca_ext_mm, self.temp_k);
                self.g_e_extra[i] += i_in;
                self.i_ca[i] += i_in;
            }
        }
        self.passive
            .step_with_extra(i_inj, &self.g_extra, &self.g_e_extra, dt);
        // Update [Ca²⁺] after the membrane solve (semi-implicit: iCa at the old
        // V/[Ca], exact exponential decay — same convention as the gates).
        for i in 0..n {
            if let Some(pool) = self.ca_pools[i] {
                self.ca[i] = pool.step(self.ca[i], self.i_ca[i], dt);
            }
        }
    }
}

/// Adaptive-step driver over [`ActiveCableSolver`] — the CVODE-analogue for the
/// cable, with **variable order** (1 ↔ 2) and step-doubling error control.
///
/// Order 1: plain implicit-Euler (backward Euler, BDF1) step-doubling — one
/// full `dt` step vs two `dt/2` steps estimate the local truncation error.
///
/// Order 2: the accepted voltage is the **Richardson extrapolation**
/// `v = 2·v_half − v_full`, which cancels the leading `h²` error term of the
/// half-step trajectory — the one-step equivalent of BDF2 (for the linear
/// cable system the two coincide). Gates/kinetics keep the half-step state
/// (semi-implicit, as in the fixed-dt solver).
///
/// Controller: `factor = 0.9·err^(−1/(p+1))` (p = 1 → −1/2, p = 2 → −1/3),
/// clamped to `[0.1, 2.0]`; `dt` stays in `[dt_min, dt_max]`. On rejection the
/// step is retried smaller at **order 1** (CVODE restarts the low order after
/// a failed step); after an accepted step with margin the order rises back to
/// 2. Stiff channels (Ca/HCN) and spikes force small `dt` automatically, while
/// slow phases take large `dt` — the same idea as NEURON's CVODE.
/// Adaptive stepping is exercised only by the module tests (the NeuroML
/// compiler runs the fixed-dt [`ActiveCableSolver`]); the whole type is
/// test-only, so production builds see an impl-less struct.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) struct AdaptiveCableSolver {
    solver: ActiveCableSolver,
    dt: f32,
    dt_min: f32,
    dt_max: f32,
    rtol: f32,
    atol: f32,
    /// Current method order: 1 (implicit Euler) or 2 (extrapolated ≈ BDF2).
    order: u8,
}

#[cfg(test)]
#[allow(dead_code)]
impl AdaptiveCableSolver {
    /// Wrap `solver` with adaptive stepping: initial `dt0`, clamped to
    /// `[dt_min, dt_max]`, with relative/absolute tolerances (mV).
    pub fn new(
        solver: ActiveCableSolver,
        dt0: f32,
        dt_min: f32,
        dt_max: f32,
        rtol: f32,
        atol: f32,
    ) -> Self {
        Self {
            solver,
            dt: dt0.clamp(dt_min, dt_max),
            dt_min,
            dt_max,
            rtol,
            atol,
            // CVODE starts at order 1; promotion happens after accepted steps.
            order: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.solver.len()
    }

    pub fn root(&self) -> usize {
        self.solver.root()
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.solver.v(idx)
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.solver.set_v(idx, value);
    }

    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        self.solver.set_channels(comp, channels);
    }

    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        self.solver.set_markov_channels(comp, channels);
    }

    /// Current step size (ms).
    pub fn current_dt(&self) -> f32 {
        self.dt
    }

    /// Current method order: 1 (implicit Euler) or 2 (extrapolated ≈ BDF2).
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Advance one adaptive step and return the `dt` actually used (ms).
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step(&mut self, i_inj: &[f32]) -> f32 {
        loop {
            let mut full = self.solver.clone();
            full.step_dt(i_inj, self.dt);
            let mut half = self.solver.clone();
            half.step_dt(i_inj, self.dt * 0.5);
            half.step_dt(i_inj, self.dt * 0.5);

            let err = cable_error_norm(&half, &full, self.atol, self.rtol);
            // Order p: local error ~ dt^(p+1) → dt scale = err^(-1/(p+1)).
            let exponent = if self.order == 2 { -1.0 / 3.0 } else { -0.5 };
            let factor = (0.9 * err.powf(exponent)).clamp(0.1, 2.0);
            if err <= 1.0 {
                if self.order == 2 {
                    // Richardson extrapolation cancels the leading h² error
                    // term of the half-step path → second order (BDF2-eq.).
                    for i in 0..self.solver.len() {
                        half.set_v(i, 2.0 * half.v(i) - full.v(i));
                    }
                }
                self.solver = half;
                self.dt = (self.dt * factor).clamp(self.dt_min, self.dt_max);
                // Raise the order when there is margin (CVODE-style); stay at
                // 2 otherwise. The extrapolation is one-step, so promotion is
                // safe at any time.
                if self.order == 1 && err < 0.5 {
                    self.order = 2;
                }
                return self.dt;
            }
            self.dt = (self.dt * factor).clamp(self.dt_min, self.dt_max);
            // CVODE restarts at the low order after a rejected step.
            self.order = 1;
            if self.dt <= self.dt_min {
                // Can't shrink further — accept the min-dt result rather than
                // loop forever (only reachable with pathological tolerances).
                self.solver = half;
                return self.dt;
            }
        }
    }
}

/// Max-norm of the per-compartment step-doubling error, scaled by the
/// tolerance: `max_i |V_half − V_full| / (atol + rtol·max|V|)`. `≤ 1` ⇒ accept.
/// Test-only (used by [`AdaptiveCableSolver`]).
#[cfg(test)]
fn cable_error_norm(a: &ActiveCableSolver, b: &ActiveCableSolver, atol: f32, rtol: f32) -> f32 {
    let n = a.len();
    let mut err = 0.0f32;
    for i in 0..n {
        let va = a.v(i);
        let vb = b.v(i);
        let scale = atol + rtol * va.abs().max(vb.abs());
        err = err.max((va - vb).abs() / scale);
    }
    err
}

/// A population of `n` identical cable neurons, each discretized from the same
/// morphology. Injected current (nA) is delivered to the soma (root
/// compartment); the observable state is the soma voltage (column 0); a spike
/// is emitted when the soma voltage rises through 0 mV.
///
/// This is the bridge from the cable engine (Фаза 5d) into
/// [`crate::api::Network`]: a NeuroML `<cell>` with morphology/biophysics lowers
/// to this boxed population, so a morphologically-detailed cell actually runs
/// in the network loop.
pub(crate) struct CablePopulation {
    n: usize,
    /// Root (soma) compartment index — identical across all neurons (same tree).
    soma_idx: usize,
    /// One cable solver per neuron.
    neurons: Vec<ActiveCableSolver>,
    /// Soma voltage mirror (contiguous, for `state_slice(0)`).
    soma_v: Vec<f32>,
    /// **Per-compartment** voltage mirror, neuron-major flat
    /// (`v[neuron*n_comps + comp]`) — the cable view for the GUI/analysis
    /// ([`Population::cable_voltage`](crate::api::population::Population)).
    comp_v: Vec<f32>,
    /// **Per-compartment** intracellular Ca²⁺ mirror, neuron-major flat
    /// (`ca[neuron*n_comps + comp]`, mM) — the Ca view for the GUI/analysis
    /// ([`Population::cable_ca`](crate::api::population::Population)). Always
    /// present (compartments without a pool stay 0), cheap to mirror.
    comp_ca: Vec<f32>,
    /// Per-neuron spike-detection flag (was-below-threshold last step).
    was_below: Vec<bool>,
    /// Reused soma-only injection buffer (length = compartments per neuron).
    soma_inj: Vec<f32>,
}

impl CablePopulation {
    /// Build `n` identical cable neurons from a discretized tree, at rest
    /// (`V = e_leak`), with no active channels. Attach channels per compartment
    /// afterwards via [`set_channels`](Self::set_channels).
    pub fn new(tree: &CompartmentTree, n: usize, e_leak: f32, dt: f32, celsius: f32) -> Self {
        let soma_idx = tree.root;
        let n_comps = tree.len();
        let neurons = (0..n)
            .map(|_| ActiveCableSolver::new(tree.clone(), e_leak, dt, celsius))
            .collect();
        Self {
            n,
            soma_idx,
            neurons,
            soma_v: vec![e_leak; n],
            comp_v: vec![e_leak; n * n_comps],
            comp_ca: vec![0.0; n * n_comps],
            was_below: vec![true; n],
            soma_inj: vec![0.0; n_comps],
        }
    }

    /// Number of neurons. Test-only — production sizes the population from
    /// the NeuroML `<population size>` (the `Population` trait `len`).
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Attach active channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_channels(comp, channels.clone());
        }
    }

    /// Attach active Markov channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_markov_channels(comp, channels.clone());
        }
    }

    /// Attach stochastic (discrete-population) channels to compartment `comp`
    /// of every neuron, re-seeding each neuron's RNG with the neuron index so
    /// the `n` copies of the cell get **independent** (but reproducible) noise
    /// streams — otherwise every neuron would follow the identical noisy
    /// trajectory.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_stochastic_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticMarkovActiveChannel>,
    ) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut chans = channels.clone();
            for ch in &mut chans {
                ch.reseed_for_neuron(i);
            }
            neuron.set_stochastic_channels(comp, chans);
        }
    }

    /// Attach stochastic Nernst (Ca²⁺) channel populations to compartment
    /// `comp` of every neuron, re-seeding each neuron's RNG with the neuron
    /// index (independent, reproducible noise streams).
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_stochastic_nernst_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticNernstActiveChannel>,
    ) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut chans = channels.clone();
            for ch in &mut chans {
                ch.reseed_for_neuron(i);
            }
            neuron.set_stochastic_nernst_channels(comp, chans);
        }
    }

    /// Attach Nernst (Ca²⁺) channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_nernst_channels(&mut self, comp: usize, channels: Vec<NernstActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_nernst_channels(comp, channels.clone());
        }
    }

    /// Attach GHK (Ca²⁺, `channelDensityGHK2`) channels to compartment `comp`
    /// of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_ghk_channels(&mut self, comp: usize, channels: Vec<GhkActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_ghk_channels(comp, channels.clone());
        }
    }

    /// Override the passive leak of compartment `comp` of every neuron with an
    /// explicit conductance `g_us` (µS) at reversal `e_rev` (mV) — per-group
    /// leak densities. The tree's uniform `g_leak_us` must be zeroed for the
    /// compartment (compile.rs) so the two don't double-count.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_leak(&mut self, comp: usize, g_us: f32, e_rev: f32) {
        for neuron in &mut self.neurons {
            neuron.set_leak(comp, g_us, e_rev);
        }
    }

    /// Attach a Ca²⁺ pool to compartment `comp` of every neuron, initialised to
    /// `initial_mm`.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_ca_pool(&mut self, comp: usize, pool: CaPool, initial_mm: f32) {
        for neuron in &mut self.neurons {
            neuron.set_ca_pool(comp, pool, initial_mm);
        }
        // Seed the Ca mirror with the initial concentration (the per-step
        // mirror in `step` only reflects concentrations *after* a step).
        let n_comps = self.soma_inj.len();
        for i in 0..self.n {
            self.comp_ca[i * n_comps + comp] = initial_mm;
        }
    }

    /// Soma (root) [Ca²⁺] (mM) of neuron `neuron`.
    /// Test-only convenience (the GUI reads Ca via [`Population::cable_ca`]).
    #[cfg(test)]
    pub fn ca_soma(&self, neuron: usize) -> f32 {
        self.neurons[neuron].ca(self.soma_idx)
    }
}

/// SplitMix64-style finaliser — mixes a per-neuron salt into a deterministic
/// 64-bit value so each neuron's noise stream is independent yet reproducible.
fn mix_seed(base: u64, salt: u64) -> u64 {
    let mut z = base ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Population for CablePopulation {
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        assert_eq!(i_in.len(), self.n, "i_in length mismatch");
        assert_eq!(spikes_out.len(), self.n, "spikes_out length mismatch");
        if self.n == 0 {
            return;
        }
        let soma = self.soma_idx;
        let n_comps = self.soma_inj.len();
        for i in 0..self.n {
            self.soma_inj[soma] = i_in[i].clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
            self.neurons[i].step(&self.soma_inj);
            let v = self.neurons[i].v(soma);
            self.soma_v[i] = v;
            // Mirror the whole compartment vector (neuron-major flat) for the
            // cable view — cheap relative to the Hines solve.
            let base = i * n_comps;
            for (j, cv) in self.comp_v[base..base + n_comps].iter_mut().enumerate() {
                *cv = self.neurons[i].v(j);
            }
            for (j, cc) in self.comp_ca[base..base + n_comps].iter_mut().enumerate() {
                *cc = self.neurons[i].ca(j);
            }
            let was = self.was_below[i];
            let is_above = v >= 0.0;
            spikes_out[i] = i32::from(is_above && was);
            self.was_below[i] = !is_above;
        }
        self.soma_inj[soma] = 0.0;
    }

    fn len(&self) -> usize {
        self.n
    }

    fn state_slice(&self, col: usize) -> &[f32] {
        if col == 0 {
            &self.soma_v
        } else {
            &[]
        }
    }

    fn param_slice(&self, _col: usize) -> &[f32] {
        &[]
    }

    fn param_slice_mut(&mut self, _col: usize) -> &mut [f32] {
        &mut []
    }

    fn fill_param(&mut self, _col: usize, _value: f32) {
        // No per-neuron parameter columns: the cable constants live on the
        // (shared) compartment tree, not as per-neuron columns.
    }

    fn fill_state(&mut self, col: usize, value: f32) {
        assert_eq!(
            col, 0,
            "CablePopulation: only state column 0 (soma V) is exposed"
        );
        self.soma_v.fill(value);
        for neuron in &mut self.neurons {
            neuron.set_v(self.soma_idx, value);
        }
        // Keep the cable mirror consistent with the soma override.
        for i in 0..self.n {
            self.comp_v[i * self.soma_inj.len() + self.soma_idx] = value;
        }
    }

    fn cable_voltage(&self) -> Option<CableVoltage<'_>> {
        let n_comps = self.soma_inj.len();
        Some(CableVoltage {
            n_neurons: self.n,
            n_comps,
            v: &self.comp_v,
        })
    }

    fn cable_ca(&self) -> Option<CableCa<'_>> {
        let n_comps = self.soma_inj.len();
        Some(CableCa {
            n_neurons: self.n,
            n_comps,
            ca: &self.comp_ca,
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    // NEURON reference tables are f64 output rounded to 6 decimals; the
    // f32 comparison target is identical either way, so the extra digits are
    // intentional (kept verbatim from the generator for auditability).
    #![allow(clippy::excessive_precision)]
    use super::*;
    use crate::api::neuroml::parse::{
        BiophysicalProperties, IntracellularProperties, IrChannelDensity, MembraneProperties,
        MorphoIR, Point3d, Segment, SegmentParent,
    };
    use crate::kernels::hh_channels::{
        gate_kinetics_to_markov, standard_channels, ChannelSpec, ClosedForm, MarkovRate,
        MarkovTransition, TauForm, VarForm,
    };

    fn seg(
        id: usize,
        parent: Option<usize>,
        px: f32,
        py: f32,
        pz: f32,
        pd: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        dd: f32,
    ) -> Segment {
        Segment {
            id,
            name: None,
            proximal: Some(Point3d {
                x: px,
                y: py,
                z: pz,
                diameter: pd,
            }),
            distal: Point3d {
                x: dx,
                y: dy,
                z: dz,
                diameter: dd,
            },
            parent: parent.map(|p| SegmentParent { segment: p }),
        }
    }

    /// Сома (корень, короткая) + дендрит длиной 1000 µm, d = 2 µm.
    fn ball_and_stick() -> MorphoIR {
        MorphoIR {
            id: "bs".to_string(),
            segments: vec![
                // Сома: (0,0,0)→(0,10,0), d = 20 µm → L = 10 µm.
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 10.0, 0.0, 20.0),
                // Дендрит: (0,10,0)→(0,1010,0), d = 2 µm → L = 1000 µm.
                seg(1, Some(0), 0.0, 10.0, 0.0, 2.0, 0.0, 1010.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        }
    }

    fn default_params() -> CableParams {
        CableParams::default()
    }

    /// Сома — один компартмент; дендрит 1000 µm при λ=1000 µm и d_λ=0.1 →
    /// ровно 10 компартментов. Топология: корень → 10 детей цепочкой.
    #[test]
    fn d_lambda_splits_long_segment_into_10() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();

        assert_eq!(tree.len(), 11);
        assert_eq!(tree.root, 0);

        let soma = &tree.compartments[0];
        assert_eq!(soma.parent, None);
        assert_eq!(soma.segment_id, 0);
        assert_eq!(soma.length_um, 10.0);
        assert_eq!(soma.diameter_um, 20.0);

        let first = &tree.compartments[1];
        assert_eq!(first.parent, Some(0));
        assert_eq!(first.segment_id, 1);
        assert_eq!(first.length_um, 100.0);
        assert_eq!(first.diameter_um, 2.0);
        for i in 2..11 {
            assert_eq!(tree.compartments[i].parent, Some(i - 1), "comp {i}");
            assert_eq!(tree.compartments[i].segment_id, 1);
        }
    }

    /// Позиции и path-дистанции: сома (центр y=5, path=5), дендритные
    /// компартменты — центры 60/160/…/960, path шагает ровно на длину
    /// компартмента (100 µm).
    #[test]
    fn compartment_positions_and_path() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();

        let soma = &tree.compartments[0];
        assert_eq!((soma.x, soma.y, soma.z), (0.0, 5.0, 0.0));
        assert!((soma.path_um - 5.0).abs() < 1e-5);

        // Дендрит (0,10)→(0,1010) → 10 компартментов по 100 µm: центры
        // 60..960, path шагает на 100 µm от первого центра 60.
        for (i, y_expected, path_expected) in [
            (1, 60.0, 60.0),
            (2, 160.0, 160.0),
            (9, 860.0, 860.0),
            (10, 960.0, 960.0),
        ] {
            let c = &tree.compartments[i];
            assert!(
                (c.y - y_expected).abs() < 1e-3,
                "comp {i}: y = {} vs {y_expected}",
                c.y
            );
            assert!(
                (c.path_um - path_expected).abs() < 1e-3,
                "comp {i}: path = {} vs {path_expected}",
                c.path_um
            );
            assert_eq!(c.x, 0.0);
            assert_eq!(c.z, 0.0);
        }
    }

    /// Аксиальное сопротивление 100 µm-компартмента (d=2 µm, Ra=100 Ω·cm) —
    /// аналитически 4·Ra·L/(π·d²)·1e-2 ≈ 31.83 MΩ.
    #[test]
    fn axial_resistance_matches_analytic() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();
        let first = &tree.compartments[1];
        let expected = 4.0 * 100.0 * 100.0 / (PI * 2.0 * 2.0) * 1e-2;
        assert!(
            (first.r_axial_mohm - expected).abs() / expected < 1e-4,
            "R_a: {} vs {}",
            first.r_axial_mohm,
            expected
        );
    }

    /// Мембранные параметры: C = cm·π·d·L·1e-5 [nF], g = π·d·L·1e-2/Rm [µS].
    #[test]
    fn capacitance_and_leak_match_analytic() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();
        let first = &tree.compartments[1]; // L=100 µm, d=2 µm
        let area_cm2 = PI * 2.0 * 100.0 * 1e-8;
        let expected_cm = 1.0 * area_cm2 * 1e3;
        let expected_g = area_cm2 / 20_000.0 * 1e6;
        assert!((first.cm_nf - expected_cm).abs() / expected_cm < 1e-4);
        assert!((first.g_leak_us - expected_g).abs() / expected_g < 1e-4);
        assert!((first.area_um2 - PI * 2.0 * 100.0).abs() < 1e-3);
    }

    /// Короткий сегмент (L ≪ 0.1λ) → ровно один компартмент (не ноль).
    #[test]
    fn short_segment_is_single_compartment() {
        let m = MorphoIR {
            id: "s".to_string(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 2.0, 0.0, 10.0, 0.0, 2.0)],
            segment_groups: vec![],
        };
        let tree = discretize(&m, &default_params()).unwrap();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.compartments[0].parent, None);
    }

    /// Несколько корней — ошибка (нейрон — одно дерево с одной сомой).
    #[test]
    fn multiple_roots_error() {
        let m = MorphoIR {
            id: "multi".to_string(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
                seg(1, None, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
            ],
            segment_groups: vec![],
        };
        assert!(discretize(&m, &default_params()).is_err());
    }

    // ─── Cable solver ─────────────────────────────────────────

    /// Один компартмент (сома d=20, L=20): τ = C/g = 20 мс. Покой — точная
    /// неподвижная точка: без входа V остаётся ровно E_leak.
    #[test]
    fn cable_solver_rest_is_fixed_point() {
        let m = MorphoIR {
            id: "s".to_string(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        let tree = discretize(&m, &default_params()).unwrap();
        let mut solver = CableSolver::new(tree, -65.0, 0.05);
        let i_inj = [0.0f32; 1];
        for _ in 0..100 {
            solver.step(&i_inj);
        }
        assert!((solver.v(0) - (-65.0)).abs() < 1e-6);
    }

    /// Один компартмент: под током садится на steady state V = E + I/g, а при
    /// t = τ значение ≈ E + 0.632·ΔV (экспонента, τ = 20 мс).
    #[test]
    fn cable_solver_single_compartment_time_constant() {
        let m = MorphoIR {
            id: "s".to_string(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        let tree = discretize(&m, &default_params()).unwrap();
        let g = tree.compartments[0].g_leak_us;
        let i_inj = [0.006_283_185f32]; // ≈ g → ΔV ≈ 1 mV
        let dt = 0.05;
        let mut solver = CableSolver::new(tree, -65.0, dt);

        // t = τ = 20 мс (400 шагов × 0.05).
        for _ in 0..400 {
            solver.step(&i_inj);
        }
        let dv_ss = i_inj[0] / g; // mV
        let expected = -65.0 + dv_ss * (1.0 - (-1.0f32).exp());
        assert!(
            (solver.v(0) - expected).abs() < 0.05,
            "V(τ) = {} vs {}",
            solver.v(0),
            expected
        );

        // t → ∞ (ещё 4000 шагов = 200 мс = 10τ).
        for _ in 0..4_000 {
            solver.step(&i_inj);
        }
        let v_ss = -65.0 + dv_ss;
        assert!(
            (solver.v(0) - v_ss).abs() < 0.01,
            "V_ss = {} vs {}",
            solver.v(0),
            v_ss
        );
    }

    /// Ball-and-stick: ток в сому → деполяризация затухает монотонно с
    /// расстоянием (сома сильнее всего, кончик дендрита — меньше всего).
    #[test]
    fn cable_solver_ball_and_stick_attenuates() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();
        let mut solver = CableSolver::new(tree, -65.0, 0.05);

        let n = solver.len();
        let mut i_inj = vec![0.0f32; n];
        i_inj[0] = 0.01; // ток в сому (nA)

        // 4000 шагов × 0.05 = 200 мс ≫ τ=20 мс и кабельное уравновешивание.
        for _ in 0..4_000 {
            solver.step(&i_inj);
        }

        assert!(solver.v(0) > -65.0, "сома должна деполяризоваться");
        for i in 0..n - 1 {
            assert!(
                solver.v(i) > solver.v(i + 1),
                "затухание: V[{}]={:.4} <= V[{}]={:.4}",
                i,
                solver.v(i),
                i + 1,
                solver.v(i + 1)
            );
        }
        // Кончик дендрита существенно ближе к покою, чем сома.
        assert!(
            solver.v(n - 1) > -65.0,
            "кончик тоже деполяризован, но слабо: {}",
            solver.v(n - 1)
        );
        assert!(
            (solver.v(0) - (-65.0)) > (solver.v(n - 1) - (-65.0)),
            "сома должна быть сильнее кончика"
        );
    }

    /// `cable_voltage` зеркалит покомпартментный V: длина = n·comps,
    /// сома-записи совпадают с `state_slice(0)`, дендрит затухает.
    #[test]
    fn cable_population_exposes_per_compartment_voltage() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();
        let mut pop = CablePopulation::new(&tree, 2, -65.0, 0.05, 6.3);
        let n_comps = tree.len();

        // До шагов: покой везде.
        let cv = pop.cable_voltage().expect("кабельная популяция даёт V");
        assert_eq!((cv.n_neurons, cv.n_comps), (2, n_comps));
        assert_eq!(cv.v.len(), 2 * n_comps);
        assert!(cv.v.iter().all(|&v| (v + 65.0).abs() < 1e-4));

        // Шагаем с током в сомы обоих нейронов.
        let mut i_in = vec![0.01f32; 2];
        let mut spikes = vec![0i32; 2];
        for _ in 0..4_000 {
            pop.step(&i_in, &mut spikes);
        }
        i_in.fill(0.0);

        let cv = pop.cable_voltage().unwrap();
        let soma = tree.root;
        for neuron in 0..2 {
            let base = neuron * n_comps;
            // Сома-запись == state_slice(0) (публичная колонка).
            assert_eq!(cv.v[base + soma], pop.state_slice(0)[neuron]);
            // Дендрит затухает: кончик ближе к покою, чем сома.
            assert!(cv.v[base + soma] > cv.v[base + n_comps - 1]);
            assert!(cv.v[base + n_comps - 1] > -65.0);
        }
    }

    /// Implicit Euler безусловно устойчив: один компартмент при dt=100 мс
    /// (5τ) под током монотонно движется к steady state, не пересекая его.
    #[test]
    fn cable_solver_implicit_stable_at_large_dt() {
        let m = MorphoIR {
            id: "s".to_string(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        let tree = discretize(&m, &default_params()).unwrap();
        let g = tree.compartments[0].g_leak_us;
        let i_inj = [0.006_283_185f32];
        let v_ss = -65.0 + i_inj[0] / g;

        let mut solver = CableSolver::new(tree, -65.0, 100.0); // dt = 5τ
        let mut v = -65.0f32;
        for _ in 0..20 {
            solver.step(&i_inj);
            v = solver.v(0);
            assert!(
                v <= v_ss + 1e-4,
                "overshoot past steady state: {v} vs {v_ss}"
            );
            assert!(v.is_finite());
        }
        assert!((v - v_ss).abs() < 0.01, "converged: {v} vs {v_ss}");
    }

    // ─── Active channels ─────────────────────────────────────

    /// Преобразовать `ChannelSpec` (g_max в mS/cm²) в `ActiveChannel` с
    /// абсолютной проводимостью (µS) для компартмента площадью `area_cm2`.
    fn active_from_spec(spec: &ChannelSpec, area_cm2: f32, v_rest: f32) -> ActiveChannel {
        let gates: Vec<(GateKinetics, u32)> = spec
            .gates
            .iter()
            .map(|g| (g.kinetics.clone(), g.power))
            .collect();
        let g_max_us = spec.g_max * area_cm2 * 1e3; // mS/cm² · cm² · 1e3 = µS
        let q10s = vec![1.0; gates.len()];
        ActiveChannel::new(gates, q10s, spec.E_rev, g_max_us, v_rest)
    }

    /// Один компартмент со стандартными NaV/KV + HH-лик должен спайкать под
    /// надпороговым током — активные каналы реально входят в кабельный солвер.
    #[test]
    fn active_cable_single_compartment_hh_fires() {
        let std = standard_channels(); // [0] = NaV, [1] = KV
        let m = MorphoIR {
            id: "s".to_string(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² → HH-лик (0.3 mS/cm², E=-54.3).
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
        let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
        solver.set_channels(
            0,
            vec![
                active_from_spec(&std[0], area_cm2, -65.0),
                active_from_spec(&std[1], area_cm2, -65.0),
            ],
        );
        solver.set_v(0, -65.0);

        let mut i_inj = vec![0.0f32; solver.len()];
        let mut spikes = 0;
        let mut was_below = true;
        for _ in 0..8_000 {
            i_inj[0] = 0.3; // nA, надпороговый
            solver.step(&i_inj);
            let v = solver.v(0);
            if v >= 0.0 && was_below {
                spikes += 1;
            }
            was_below = v < 0.0;
        }
        assert!(
            spikes >= 1,
            "HH-компартмент должен спайкать, получено {spikes} спайков"
        );
    }

    // ─── Biophysics → CableParams bridge ─────────────────────

    /// `resistivity`→Ra, `specificCapacitance`→cm, пассивный лик→rm, и
    /// `erev` лика как e_leak. condDensity 3e-4 S/cm² → rm = 3333 Ω·cm².
    #[test]
    fn cable_params_from_biophysics_maps_units() {
        let bp = BiophysicalProperties {
            membrane: MembraneProperties {
                channel_densities: vec![IrChannelDensity {
                    id: "pas".into(),
                    ion_channel: "pas".into(),
                    cond_density: 3e-4, // S/cm²
                    erev: -70.0,
                    ion: "non_specific".into(),
                    segment_group: "soma_group".into(),
                }],
                non_uniform_densities: vec![],
                channel_populations: vec![],
                nernst_populations: vec![],
                nernst_densities: vec![],
                ghk_densities: vec![],
                init_memb_potential: -65.0,
                specific_capacitance: 1.5,
            },
            intracellular: IntracellularProperties { resistivity: 150.0 },
            species: vec![],
        };

        let params = CableParams::from_biophysics(&bp);
        assert_eq!(params.ra_ohm_cm, 150.0);
        assert_eq!(params.cm_uf_per_cm2, 1.5);
        assert!(
            (params.rm_ohm_cm2 - 3333.333).abs() < 1.0,
            "rm: {}",
            params.rm_ohm_cm2
        );
        assert_eq!(passive_leak_reversal(&bp), -70.0);
    }

    /// Без пассивного канала: rm остаётся дефолтом, e_leak = initMembPotential.
    #[test]
    fn cable_params_fallback_without_passive_channel() {
        let bp = BiophysicalProperties {
            membrane: MembraneProperties {
                channel_densities: vec![],
                non_uniform_densities: vec![],
                channel_populations: vec![],
                nernst_populations: vec![],
                nernst_densities: vec![],
                ghk_densities: vec![],
                init_memb_potential: -58.0,
                specific_capacitance: 1.0,
            },
            intracellular: IntracellularProperties { resistivity: 100.0 },
            species: vec![],
        };
        let params = CableParams::from_biophysics(&bp);
        assert_eq!(params.rm_ohm_cm2, CableParams::default().rm_ohm_cm2);
        assert_eq!(passive_leak_reversal(&bp), -58.0);
    }

    // ─── SegmentGroup → compartments ─────────────────────────

    /// `group_compartments` мапит id сегментов на индексы компартментов:
    /// сома (segment 0) → [0], дендрит (segment 1) → 10 компартментов.
    #[test]
    fn group_compartments_maps_segments() {
        let m = ball_and_stick();
        let tree = discretize(&m, &default_params()).unwrap();
        assert_eq!(tree.group_compartments(&[0]), vec![0]);
        assert_eq!(tree.group_compartments(&[1]), (1..11).collect::<Vec<_>>());
        assert_eq!(
            tree.group_compartments(&[0, 1]),
            (0..11).collect::<Vec<_>>()
        );
    }

    // ─── Markov channels in the cable ────────────────────────

    /// Стандартный NaV (m³h), опущенный до 2-состоянийного Маркова, работает
    /// как `MarkovActiveChannel`: открытая доля конечна и в [0, 1].
    #[test]
    fn markov_active_channel_runs() {
        let std = standard_channels();
        let gates: Vec<MarkovGateSpec> = std[0]
            .gates
            .iter()
            .map(|g| gate_kinetics_to_markov(g.kinetics.clone(), g.power))
            .collect();
        let mut ch = MarkovActiveChannel::new(
            gates.clone(),
            vec![1.0; gates.len()],
            std[0].E_rev,
            0.001,
            -65.0,
        );
        for _ in 0..100 {
            let f = ch.step_open(-65.0, 0.05);
            assert!(
                f.is_finite() && (0.0..=1.0).contains(&f),
                "open fraction {f}"
            );
        }
    }

    // ─── NEURON parity (passive cable) ───────────────────────

    /// Эталон NEURON 9.0.2 для пассивного ball-and-stick (сома + дендрит),
    /// сгенерированный `gen_cable_ref.py`: Ra=100, cm=1, Rm=20000, E=-65,
    /// I=+0.01 nA в сому (10–60 мс), dt=0.025, secondorder=0 (backward Euler).
    const REF_V_SOMA: [f32; 21] = [
        -65.0, -65.0, -65.0, -63.824051, -63.304875, -62.940604, -62.662314, -62.446288,
        -62.278122, -62.147147, -62.045129, -61.965666, -61.903770, -63.031507, -63.513130,
        -63.848150, -64.103655, -64.301933, -64.456276, -64.576483, -64.670114,
    ];
    const REF_V_TIP: [f32; 21] = [
        -65.0, -65.0, -65.0, -64.836370, -64.449930, -64.103704, -63.827867, -63.612175,
        -63.444054, -63.313085, -63.211068, -63.131605, -63.069709, -63.185127, -63.534014,
        -63.850988, -64.104041, -64.301986, -64.456283, -64.576484, -64.670114,
    ];

    #[test]
    fn passive_ball_and_stick_matches_neuron() {
        // Сома L=20/d=20 + дендрит L=1000/d=2 — ровно как в gen_cable_ref.py.
        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        assert_eq!(tree.len(), 11, "1 сома + 10 дендритных компартментов");
        let mut solver = CableSolver::new(tree, -65.0, 0.025);

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 2400usize; // t = 60 мс
        let mut i_inj = vec![0.0f32; solver.len()];
        let mut v_soma = Vec::with_capacity(n_steps + 1);
        let mut v_tip = Vec::with_capacity(n_steps + 1);
        v_soma.push(solver.v(0));
        v_tip.push(solver.v(10));
        for step in 0..n_steps {
            i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
            solver.step(&i_inj);
            v_soma.push(solver.v(0));
            v_tip.push(solver.v(10));
        }

        let grid = 200usize; // 5 мс / 0.025 мс
        let mut d_soma = 0.0f32;
        let mut d_tip = 0.0f32;
        for (k, (&rs, &rt)) in REF_V_SOMA.iter().zip(REF_V_TIP.iter()).enumerate() {
            d_soma = d_soma.max((v_soma[k * grid] - rs).abs());
            d_tip = d_tip.max((v_tip[k * grid] - rt).abs());
        }
        println!("cable vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV");
        // Оба — backward Euler (secondorder=0), один Hines-солвер; остаток —
        // f32 (pillars) против f64 (NEURON) за 4000 шагов (~4e-4 мВ).
        assert!(
            d_soma < 0.01 && d_tip < 0.01,
            "soma={d_soma:.6} tip={d_tip:.6} mV"
        );
    }

    /// Эталон NEURON 9.0.2 для **ветвящегося** пассивного кабеля (сома →
    /// ствол → два асимметричных дендрита), сгенерированный
    /// `gen_cable_branch_ref.py`: Ra=100, cm=1, Rm=20000, E=-65, I=+0.01 nA в
    /// сому (10–60 мс), dt=0.025, secondorder=0 (backward Euler). Проверяет
    /// Hines-солвер на нетривиальной топологии (одна ветвь → две дочери).
    const REF_BRANCH_SOMA: [f32; 21] = [
        -65.000000, -65.000000, -65.000000, -63.825576, -63.333325, -63.023504, -62.804708,
        -62.641781, -62.517441, -62.421476, -62.347035, -62.289158, -62.244113, -63.383464,
        -63.848399, -64.136945, -64.339171, -64.489190, -64.603477, -64.691610, -64.759951,
    ];
    const REF_BRANCH_BRANCH: [f32; 21] = [
        -65.000000, -65.000000, -65.000000, -64.916861, -64.686765, -64.458712, -64.267275,
        -64.113731, -63.992635, -63.897795, -63.823744, -63.766002, -63.721004, -63.769086,
        -63.971872, -64.178653, -64.353520, -64.494157, -64.605199, -64.692208, -64.760159,
    ];
    const REF_BRANCH_TIP_A: [f32; 21] = [
        -65.000000, -65.000000, -65.000000, -64.989517, -64.887070, -64.720039, -64.551670,
        -64.406343, -64.288123, -64.194284, -64.120580, -64.062959, -64.018003, -63.993444,
        -64.068586, -64.214346, -64.366145, -64.498565, -64.606732, -64.692740, -64.760344,
    ];
    const REF_BRANCH_TIP_B: [f32; 21] = [
        -65.000000, -65.000000, -65.000000, -64.932539, -64.711562, -64.485878, -64.295165,
        -64.141862, -64.020848, -63.926037, -63.851996, -63.794257, -63.749261, -63.781665,
        -63.975333, -64.179744, -64.353886, -64.494282, -64.605242, -64.692223, -64.760164,
    ];

    // ─── NEURON parity (1000-compartment cable) ─────────────────

    /// Эталон NEURON 9.0.2 для **1001-компартментного** пассивного кабеля —
    /// сгенерирован `gen_cable_long_ref.py`: сома L=20/d=20 (nseg=1) +
    /// дендрит L=100000/d=2 (nseg=1000, d_λ=0.1 → сегменты 100 мкм), Ra=100,
    /// cm=1, Rm=20000, E=-65, I=+0.01 nA в сому (10–60 мс), dt=0.025,
    /// secondorder=0. Проверяет, что f32-элиминация не деградирует на длинной
    /// цепочке: сома должна совпасть (~4e-4), а середина/кончик (50–100 λ,
    /// куда импульс не доходит) — не дрейфовать от −65.
    ///
    /// Замечание: слишком мелкая сетка (d_λ=0.001 → сегменты 1 мкм) численно
    /// патологична — осевая проводимость в ~550 раз больше мембранной, и
    /// f32-элиминация дрейфует (~2e-3 мВ/шаг). Сетка λ/10 — хорошо
    /// обусловлена.
    const REF_V_LONG_SOMA: [f32; 21] = [
        -65.0, -65.0, -65.0, -63.825486, -63.335048, -63.036762, -62.839418, -62.703421,
        -62.607393, -62.538478, -62.488445, -62.451803, -62.424785, -63.579269, -64.054791,
        -64.341927, -64.530912, -64.660623, -64.751915, -64.817253, -64.864580,
    ];
    const REF_V_LONG_MID: [f32; 21] = [-65.0; 21];
    const REF_V_LONG_TIP: [f32; 21] = [-65.0; 21];

    #[test]
    fn passive_long_cable_matches_neuron() {
        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                // Дендрит L=100000 мкм: d_λ=0.1 → 1000 сегментов по 100 мкм.
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 100020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let n = tree.len();
        assert_eq!(n, 1001, "сома 1 + дендрит 1000");
        let mid = 501; // dend(0.5): сегмент 500 (0-indexed), глобально 1+500
        let tip = n - 1;
        let mut solver = CableSolver::new(tree, -65.0, 0.025);

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 2400usize; // t = 60 мс
        let mut i_inj = vec![0.0f32; n];
        // Сэмплинг на границах шага (как NEURON Vector.record).
        let mut v_soma = vec![solver.v(0)];
        let mut v_mid = vec![solver.v(mid)];
        let mut v_tip = vec![solver.v(tip)];
        for step in 0..n_steps {
            i_inj.fill(0.0);
            i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
            solver.step(&i_inj);
            if (step + 1) % 200 == 0 {
                v_soma.push(solver.v(0));
                v_mid.push(solver.v(mid));
                v_tip.push(solver.v(tip));
            }
        }
        let mut d_soma = 0.0f32;
        let mut d_mid = 0.0f32;
        let mut d_tip = 0.0f32;
        for k in 0..21 {
            d_soma = d_soma.max((v_soma[k] - REF_V_LONG_SOMA[k]).abs());
            d_mid = d_mid.max((v_mid[k] - REF_V_LONG_MID[k]).abs());
            d_tip = d_tip.max((v_tip[k] - REF_V_LONG_TIP[k]).abs());
        }
        println!(
            "long cable vs NEURON: max |ΔV| soma={d_soma:.6} mid={d_mid:.6} tip={d_tip:.6} mV"
        );
        assert!(d_soma < 0.005, "soma: max |ΔV| = {d_soma:.4} mV");
        assert!(d_mid < 0.001, "mid: max |ΔV| = {d_mid:.4} mV");
        assert!(d_tip < 0.001, "tip: max |ΔV| = {d_tip:.4} mV");
    }

    #[test]
    fn passive_branching_cable_matches_neuron() {
        // Сома L=20/d=20, ствол L=1000/d=2, дочери da L=500/d=1 и db L=200/d=2
        // (обе от кончика ствола) — ровно как в gen_cable_branch_ref.py.
        let m = MorphoIR {
            id: "branch".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
                seg(2, Some(1), 0.0, 1020.0, 0.0, 1.0, 0.0, 1520.0, 0.0, 1.0),
                seg(3, Some(1), 0.0, 1020.0, 0.0, 2.0, 200.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        // 1 сома + 10 ствол + 1 branch-узел + 8 da + 2 db.
        assert_eq!(tree.len(), 22, "1+10+1+8+2 компартмента");
        let mut solver = CableSolver::new(tree, -65.0, 0.025);

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 2400usize; // t = 60 мс
        let mut i_inj = vec![0.0f32; solver.len()];
        // Компартменты: сома=0, кончик ствола (центр)=10, branch-узел=11,
        // кончик da=19, кончик db=21. NEURON `trunk(1.0)` = branch-узел (x=1.0).
        let (i_soma, i_branch, i_tip_a, i_tip_b) = (0usize, 11usize, 19usize, 21usize);
        let mut v_soma = Vec::with_capacity(n_steps + 1);
        let mut v_branch = Vec::with_capacity(n_steps + 1);
        let mut v_tip_a = Vec::with_capacity(n_steps + 1);
        let mut v_tip_b = Vec::with_capacity(n_steps + 1);
        v_soma.push(solver.v(i_soma));
        v_branch.push(solver.v(i_branch));
        v_tip_a.push(solver.v(i_tip_a));
        v_tip_b.push(solver.v(i_tip_b));
        for step in 0..n_steps {
            i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
            solver.step(&i_inj);
            v_soma.push(solver.v(i_soma));
            v_branch.push(solver.v(i_branch));
            v_tip_a.push(solver.v(i_tip_a));
            v_tip_b.push(solver.v(i_tip_b));
        }

        let grid = 200usize; // 5 мс / 0.025 мс
        let (mut d_soma, mut d_branch, mut d_a, mut d_b) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for k in 0..REF_BRANCH_SOMA.len() {
            d_soma = d_soma.max((v_soma[k * grid] - REF_BRANCH_SOMA[k]).abs());
            d_branch = d_branch.max((v_branch[k * grid] - REF_BRANCH_BRANCH[k]).abs());
            d_a = d_a.max((v_tip_a[k * grid] - REF_BRANCH_TIP_A[k]).abs());
            d_b = d_b.max((v_tip_b[k * grid] - REF_BRANCH_TIP_B[k]).abs());
        }
        println!(
            "branching cable vs NEURON: max |ΔV| soma={d_soma:.6} branch={d_branch:.6} tipA={d_a:.6} tipB={d_b:.6} mV"
        );
        // Один backward Euler (secondorder=0) Hines-солвер; остаток — f32 vs f64.
        assert!(
            d_soma < 0.01 && d_branch < 0.01 && d_a < 0.01 && d_b < 0.01,
            "soma={d_soma:.6} branch={d_branch:.6} tipA={d_a:.6} tipB={d_b:.6} mV"
        );
    }

    /// Активный HH на соме кабеля против NEURON `hh` (эталон `gen_cable_hh_ref.py`:
    /// сома L=20/d=20, NaV/KV + лик, I=+0.3 nA, dt=0.025, t=10–90 мс).
    /// Гейты pillars — exponential-Euler, NEURON — backward-Euler, поэтому
    /// сверяется **число спайков** (не трасса): 8 ± 2.
    #[test]
    fn active_hh_soma_matches_neuron_spike_count() {
        let std = standard_channels();
        let m = MorphoIR {
            id: "s".into(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² → HH-лик (0.3 mS/cm², E=-54.3).
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
        let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
        solver.set_channels(
            0,
            vec![
                active_from_spec(&std[0], area_cm2, -65.0),
                active_from_spec(&std[1], area_cm2, -65.0),
            ],
        );
        solver.set_v(0, -65.0);

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 3600usize; // t = 90 мс
        let mut i_inj = vec![0.0f32; solver.len()];
        let mut spikes = 0u32;
        let mut was_below = true;
        for step in 0..n_steps {
            i_inj[0] = if step >= on && step < off { 0.3 } else { 0.0 };
            solver.step(&i_inj);
            let v = solver.v(0);
            if v >= 0.0 && was_below {
                spikes += 1;
            }
            was_below = v < 0.0;
        }
        println!("active cable vs NEURON: spikes={spikes} (NEURON=8)");
        assert!((spikes as i32 - 8).abs() <= 2, "spikes={spikes} (NEURON=8)");
    }

    // ─── NEURON parity (Markov/kschan in the cable) ──────────

    /// Эталон NEURON 9.0.2 для ball-and-stick с 3-состоянием ks3
    /// (Na-m/Na-h rates) на соме — сгенерирован `gen_cable_ks_ref.py`.
    /// pas 0.3 mS/cm² / E=-54.3 везде, ks3 gbar=36 mS/cm² / ek=-77 на соме,
    /// V0=-65, I=+0.125664 nA (10 µA/cm²) в сому (10–60 мс), dt=0.025,
    /// secondorder=0. K-шунт сомы держит её у покоя, пассивный дендрит
    /// (e_pas=-54.3) заряжается — канал реально работает.
    const REF_V_KS3_SOMA: [f32; 21] = [
        -65.0, -67.796423, -67.671610, -65.322792, -65.312031, -65.311104, -65.311307, -65.311586,
        -65.311819, -65.312004, -65.312149, -65.312263, -65.312351, -67.635761, -67.649498,
        -67.650045, -67.649160, -67.648254, -67.647486, -67.646852, -67.646331,
    ];
    const REF_V_KS3_TIP: [f32; 21] = [
        -65.0, -57.698422, -56.749023, -56.339287, -56.216609, -56.200489, -56.198458, -56.198246,
        -56.198261, -56.198295, -56.198325, -56.198349, -56.198368, -56.474851, -56.584485,
        -56.599277, -56.601068, -56.601149, -56.601029, -56.600905, -56.600800,
    ];

    /// 3-state gate (closed ⇌ open ⇌ inactivated), rates = Na-m (0↔1) + Na-h
    /// (1↔2) — зеркало `ks3.mod`, как в `tests/ks_vs_neuron.rs::ks3_gate`.
    fn ks3_gate() -> MarkovGateSpec {
        MarkovGateSpec {
            open: vec![0.0, 1.0, 0.0],
            transitions: vec![
                MarkovTransition {
                    from: 0,
                    to: 1,
                    rate: MarkovRate::Directed {
                        rate: ClosedForm::ExpLinear {
                            rate: 0.1,
                            midpoint: 40.0,
                            scale: 10.0,
                        },
                    },
                },
                MarkovTransition {
                    from: 1,
                    to: 0,
                    rate: MarkovRate::Directed {
                        rate: ClosedForm::Exp {
                            rate: 4.0,
                            midpoint: 65.0,
                            scale: 18.0,
                        },
                    },
                },
                MarkovTransition {
                    from: 1,
                    to: 2,
                    rate: MarkovRate::Directed {
                        rate: ClosedForm::Exp {
                            rate: 0.07,
                            midpoint: 65.0,
                            scale: 20.0,
                        },
                    },
                },
                MarkovTransition {
                    from: 2,
                    to: 1,
                    rate: MarkovRate::Directed {
                        rate: ClosedForm::Sigmoid {
                            rate: 1.0,
                            midpoint: 35.0,
                            scale: 10.0,
                        },
                    },
                },
            ],
            power: 1,
        }
    }

    /// Марков-канал (kschan) в кабеле против NEURON: 3-состояние ks3 на соме
    /// ball-and-stick, субпороговый протокол, сверка V(сома)/V(кончик) на
    /// сетке 5 мс. NEURON интегрирует KINETIC backward-Euler (METHOD sparse),
    /// pillars — точной матричной экспонентой → согласие с точностью
    /// интегратора: max |ΔV| ≈ 0.014 мВ (замер; допуск 0.03), против ~6e-4 у
    /// точечного `ks_vs_neuron.rs` — добавка от аксиального пути 1000 мкм.
    #[test]
    fn markov_cable_matches_neuron_ks3() {
        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² (pas, как в ks_vs_neuron).
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let n = tree.len();
        let soma = tree.root;
        let tip = n - 1;
        let area_cm2 = tree.compartments[soma].area_um2 * 1e-8;
        let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
        // V0 = -65 (покой), хотя e_leak (pas reversal) = -54.3 — как в
        // NEURON `finitialize(-65)` при e_pas=-54.3.
        for i in 0..n {
            solver.set_v(i, -65.0);
        }
        // gbar 36 mS/cm² → абсолютная µS по площади сомы.
        let g_max_us = 0.036 * area_cm2 * 1e6;
        solver.set_markov_channels(
            soma,
            vec![MarkovActiveChannel::new(
                vec![ks3_gate()],
                vec![1.0],
                -77.0,
                g_max_us,
                -65.0,
            )],
        );

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 2400usize; // t = 60 мс
        let mut i_inj = vec![0.0f32; n];
        // Сэмплинг на границах шага (как NEURON Vector.record и ks_vs_neuron):
        // t=0 до первого шага, далее после шага `s` → t = (s+1)·dt.
        let mut v_soma = vec![solver.v(soma)];
        let mut v_tip = vec![solver.v(tip)];
        for step in 0..n_steps {
            i_inj.fill(0.0);
            i_inj[soma] = if step >= on && step < off {
                0.125664
            } else {
                0.0
            };
            solver.step(&i_inj);
            if (step + 1) % 200 == 0 {
                v_soma.push(solver.v(soma));
                v_tip.push(solver.v(tip));
            }
        }
        let mut d_soma = 0.0f32;
        let mut d_tip = 0.0f32;
        for (k, (&rs, &rt)) in REF_V_KS3_SOMA.iter().zip(REF_V_KS3_TIP.iter()).enumerate() {
            d_soma = d_soma.max((v_soma[k] - rs).abs());
            d_tip = d_tip.max((v_tip[k] - rt).abs());
        }
        println!("markov cable vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV");
        assert!(d_soma < 0.03, "soma: max |ΔV| = {d_soma:.4} mV");
        assert!(d_tip < 0.03, "tip: max |ΔV| = {d_tip:.4} mV");
    }

    /// Стохастический `channelPopulation` против NEURON: **среднее по ансамблю**
    /// сходится к детерминистическому следу ks3.
    ///
    /// Детерминистический предел дискретной популяции совпадает с `channelDensity`:
    /// та же master equation (тот же `p_open`), та же суммарная проводимость
    /// `single_cond·N = gbar·area`. Поэтому эталон — те же `REF_V_KS3_*`, что и в
    /// `markov_cable_matches_neuron_ks3` (NEURON `KINETIC` backward-Euler). Канальный
    /// шум `∝ 1/√N` усредняется по `N_SEEDS` независимым seed'ам: матожидание
    /// V(t) сходится к детерминистическому следу с точностью `≈ σ_ансамбля/√N_SEEDS`.
    ///
    /// jNeuroML экспортирует `channelPopulation` **детерминистически** (mean-field,
    /// общая проводимость `number × single_cond`); биномиальная выборка `n_open` —
    /// наше расширение, валидированное здесь по среднему и отдельно по дисперсии
    /// (`sample_binomial` mean/var-тест).
    #[test]
    fn channel_population_mean_matches_neuron() {
        // Быстрый режим: биномиальная выборка стоит ~N·p RNG-вызовов/шаг, а
        // шум среднего убывает как 1/√(N·seed'ов) — работа на заданную точность
        // инвариантна к разбиению N/seed. Умеренное N и несколько seed'ов дают
        // σ_mean ~ 0.02 мВ при <2 с (не 8192/128 → 70 с).
        const N_CHANNELS: usize = 2048;
        const N_SEEDS: usize = 16;

        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let n = tree.len();
        let soma = tree.root;
        let tip = n - 1;
        let area_cm2 = tree.compartments[soma].area_um2 * 1e-8;
        // Суммарная проводимость: `single_cond · N = 0.036 S/cm² · area` — тот
        // же gbar, что и в детерминистическом ks3-тесте, т.е. один и тот же
        // mean-field ток. Разложение на single_cond × N живёт в compile.rs.
        let g_max_us = 0.036 * area_cm2 * 1e6;

        let n_steps = 4000usize;
        let on = 400usize; // t = 10 мс
        let off = 2400usize; // t = 60 мс
        let grid = 200usize; // семплинг каждые 5 мс, как в REF_V_KS3_*
        let n_samples = 1 + n_steps / grid;

        let base = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
        let base_seed = 0x5EED_CAFE_0BAD_F00Du64;
        let mut acc_soma = vec![0.0f64; n_samples];
        let mut acc_tip = vec![0.0f64; n_samples];

        for seed in 0..N_SEEDS {
            let mut solver = base.clone();
            // V0 = -65 (покой), e_leak = -54.3 — как в NEURON `finitialize(-65)`.
            for i in 0..n {
                solver.set_v(i, -65.0);
            }
            // Свежий канал: occupancy в steady-state при v_rest=-65, независимый
            // seed (потоки независимы → усреднение сходится по ЦПТ).
            solver.set_stochastic_channels(
                soma,
                vec![StochasticMarkovActiveChannel::new(
                    vec![ks3_gate()],
                    vec![1.0],
                    -77.0,
                    g_max_us,
                    N_CHANNELS,
                    -65.0,
                    base_seed.wrapping_add(seed as u64),
                )],
            );

            let mut i_inj = vec![0.0f32; n];
            acc_soma[0] += solver.v(soma) as f64;
            acc_tip[0] += solver.v(tip) as f64;
            let mut k = 0usize;
            for step in 0..n_steps {
                i_inj.fill(0.0);
                i_inj[soma] = if step >= on && step < off {
                    0.125664
                } else {
                    0.0
                };
                solver.step(&i_inj);
                if (step + 1) % grid == 0 {
                    k += 1;
                    acc_soma[k] += solver.v(soma) as f64;
                    acc_tip[k] += solver.v(tip) as f64;
                }
            }
        }

        let mut d_soma = 0.0f32;
        let mut d_tip = 0.0f32;
        for (idx, (&rs, &rt)) in REF_V_KS3_SOMA.iter().zip(REF_V_KS3_TIP.iter()).enumerate() {
            let mean_soma = (acc_soma[idx] / N_SEEDS as f64) as f32;
            let mean_tip = (acc_tip[idx] / N_SEEDS as f64) as f32;
            d_soma = d_soma.max((mean_soma - rs).abs());
            d_tip = d_tip.max((mean_tip - rt).abs());
        }
        println!(
            "channelPopulation mean vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV (N={N_CHANNELS}, seeds={N_SEEDS})"
        );
        // Замер: soma≈0.041, tip≈0.015 мВ (детерминированный по фиксированным
        // seed'ам — не флакает). Допуск 0.1 = 2.4× запаса над замером.
        assert!(d_soma < 0.1, "soma: max |ΔV| = {d_soma:.4} mV");
        assert!(d_tip < 0.1, "tip: max |ΔV| = {d_tip:.4} mV");
    }

    // ─── Adaptive step (CVODE-analogue) ──────────────────────

    /// Прогнать `AdaptiveCableSolver` до `tstop`, подавая импульс в сому.
    /// Возвращает (min_dt, max_dt, число шагов).
    fn run_adaptive(
        solver: &mut AdaptiveCableSolver,
        pulse_on: f32,
        pulse_off: f32,
        i_amp: f32,
        tstop: f32,
    ) -> (f32, f32, u32) {
        let soma = solver.root();
        let n = solver.len();
        let mut i_inj = vec![0.0f32; n];
        let mut t = 0.0f32;
        let mut min_dt = f32::INFINITY;
        let mut max_dt = 0.0f32;
        let mut steps = 0u32;
        while t < tstop - 1e-6 {
            i_inj.fill(0.0);
            i_inj[soma] = if t >= pulse_on && t < pulse_off {
                i_amp
            } else {
                0.0
            };
            let dt = solver.step(&i_inj);
            t += dt;
            min_dt = min_dt.min(dt);
            max_dt = max_dt.max(dt);
            steps += 1;
        }
        (min_dt, max_dt, steps)
    }

    /// Пассивный ball-and-stick с адаптивным шагом: финальные V(сома)/V(кончик)
    /// совпадают с фиксированным эталоном NEURON (t=100 → -64.6701 мВ), а dt
    /// реально меняется (сжатие на фронте импульса, рост на пологих участках).
    #[test]
    fn adaptive_passive_cable_matches_neuron() {
        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let base = ActiveCableSolver::new(tree, -65.0, 0.025, 6.3);
        let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 1.0, 1e-3, 1e-3);

        let (min_dt, max_dt, steps) = run_adaptive(&mut solver, 10.0, 60.0, 0.01, 100.0);
        let v_soma = solver.v(0);
        let v_tip = solver.v(10);
        println!(
            "adaptive passive: min_dt={min_dt:.4} max_dt={max_dt:.4} steps={steps} v_soma={v_soma:.6} v_tip={v_tip:.6}"
        );
        assert!((v_soma + 64.670114).abs() < 0.01, "v_soma={v_soma}");
        assert!((v_tip + 64.670114).abs() < 0.01, "v_tip={v_tip}");
        assert!(
            min_dt < max_dt,
            "dt должен адаптироваться: {min_dt} vs {max_dt}"
        );
    }

    /// Переменный порядок (CVODE-стиль): солвер стартует на BDF1, после
    /// принятых шагов с запасом поднимается до BDF2 (Richardson-экстраполяция),
    /// а на фронте импульса (отказ шага) сбрасывается обратно на BDF1.
    #[test]
    fn adaptive_variable_order_transitions() {
        // Прогон протокола с фиксированным допуском; возвращает (v_soma,
        // min_order, max_order, была ли демоция порядка после отказа шага).
        fn run(tol: f32) -> (f32, u8, u8, bool) {
            let m = MorphoIR {
                id: "bs".into(),
                segments: vec![
                    seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                    seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
                ],
                segment_groups: vec![],
            };
            let params = CableParams {
                ra_ohm_cm: 100.0,
                cm_uf_per_cm2: 1.0,
                rm_ohm_cm2: 20_000.0,
                d_lambda: 0.1,
            };
            let tree = discretize(&m, &params).unwrap();
            let base = ActiveCableSolver::new(tree, -65.0, 0.025, 6.3);
            let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 1.0, tol, tol);

            let soma = solver.root();
            let n = solver.len();
            let mut i_inj = vec![0.0f32; n];
            let mut t = 0.0f32;
            let mut min_order = u8::MAX;
            let mut max_order = 0u8;
            let mut saw_rejection_restart = false;
            let mut prev_order = 1u8;
            while t < 100.0 - 1e-6 {
                i_inj.fill(0.0);
                // Резкий скачок 0 → 0.5 nA: жёсткий фронт, который должен
                // отклонить BDF2-шаг и сбросить порядок на BDF1.
                i_inj[soma] = if (10.0..60.0).contains(&t) { 0.5 } else { 0.0 };
                let dt = solver.step(&i_inj);
                t += dt;
                let order = solver.order();
                // Демоция: порядок упал 2 → 1 после шага (отказ на фронте).
                if prev_order == 2 && order == 1 {
                    saw_rejection_restart = true;
                }
                prev_order = order;
                min_order = min_order.min(order);
                max_order = max_order.max(order);
            }
            (solver.v(0), min_order, max_order, saw_rejection_restart)
        }

        // Основной прогон: рабочий допуск 1e-3.
        let (v_soma, min_order, max_order, saw_rejection_restart) = run(1e-3);
        // Эталон: жёсткий допуск 1e-5 — практически сошёлся.
        let (v_ref, _, _, _) = run(1e-5);
        println!(
            "adaptive order: min={min_order} max={max_order} rejection_restart={saw_rejection_restart} v_soma={v_soma:.6} v_ref={v_ref:.6}"
        );
        // Порядок реально поднимался до 2 (и стартовал с 1 — до первого
        // принятого шага промоция невозможна).
        assert_eq!(max_order, 2, "солвер должен достигать BDF2");
        // На фронте импульса (жёсткий скачок) шаг отклоняется → рестарт BDF1.
        assert!(
            saw_rejection_restart,
            "фронт импульса должен вызвать отказ шага и рестарт порядка"
        );
        // Переключение порядка не теряет точность: на сильном импульсе
        // (размах ~100 мВ) локальный контроль ошибки даёт накопление
        // ~tol-масштаба; против сошедшегося эталона (1e-5) — < 2 мВ.
        assert!(
            (v_soma - v_ref).abs() < 2.0,
            "v_soma={v_soma} vs v_ref={v_ref}"
        );
    }

    /// Активный HH с адаптивным шагом: то же число спайков (8), что у фикс.
    /// dt, и dt сжимается на спайках (min_dt < max_dt).
    #[test]
    fn adaptive_active_hh_spike_count() {
        let std = standard_channels();
        let m = MorphoIR {
            id: "s".into(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
        let mut base = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
        base.set_channels(
            0,
            vec![
                active_from_spec(&std[0], area_cm2, -65.0),
                active_from_spec(&std[1], area_cm2, -65.0),
            ],
        );
        base.set_v(0, -65.0);
        let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 0.1, 1e-3, 1e-3);

        let soma = solver.root();
        let n = solver.len();
        let mut i_inj = vec![0.0f32; n];
        let mut t = 0.0f32;
        let mut min_dt = f32::INFINITY;
        let mut max_dt = 0.0f32;
        let mut spikes = 0u32;
        let mut was_below = true;
        while t < 100.0 - 1e-6 {
            i_inj.fill(0.0);
            i_inj[soma] = if (10.0..90.0).contains(&t) { 0.3 } else { 0.0 };
            let dt = solver.step(&i_inj);
            t += dt;
            min_dt = min_dt.min(dt);
            max_dt = max_dt.max(dt);
            let v = solver.v(soma);
            if v >= 0.0 && was_below {
                spikes += 1;
            }
            was_below = v < 0.0;
        }
        println!("adaptive active: spikes={spikes} min_dt={min_dt:.4} max_dt={max_dt:.4}");
        assert!((spikes as i32 - 8).abs() <= 2, "spikes={spikes}");
        assert!(
            min_dt < max_dt,
            "dt должен адаптироваться: {min_dt} vs {max_dt}"
        );
    }

    // ─── Stochastic channels (`channelPopulation`) ────────────

    /// `sample_binomial` — точный сэмплер: среднее ≈ n·p, дисперсия ≈ n·p·(1−p),
    /// и детерминирован по сиду (тот же сид → та же последовательность).
    #[test]
    fn sample_binomial_mean_variance_and_determinism() {
        let n = 64usize;
        let p = 0.5f32;
        let draws = 100_000usize;

        let mut rng = StdRng::seed_from_u64(0xB00B5);
        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        for _ in 0..draws {
            let k = sample_binomial(&mut rng, n, p) as u64;
            sum += k;
            sum_sq += k * k;
        }
        let mean = sum as f64 / draws as f64;
        let var = sum_sq as f64 / draws as f64 - mean * mean;
        let expect_mean = n as f64 * f64::from(p);
        let expect_var = n as f64 * f64::from(p) * (1.0 - f64::from(p));
        assert!((mean - expect_mean).abs() < 0.2, "mean = {mean}");
        assert!((var - expect_var).abs() < 0.5, "var = {var}");

        // Детерминизм: два генератора с одним сидом дают одну последовательность.
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let seq_a: Vec<usize> = (0..64).map(|_| sample_binomial(&mut a, n, p)).collect();
        let seq_b: Vec<usize> = (0..64).map(|_| sample_binomial(&mut b, n, p)).collect();
        assert_eq!(seq_a, seq_b);
    }

    /// Stochastic-канал с p_open=0.5 и N=64: среднее открытой доли ≈ 0.5 (шум
    /// реально сэмплируется), разные сиды дают разные трейны.
    #[test]
    fn stochastic_channel_sampling_is_noisy_and_reproducible() {
        // 2-состоянийный gate с steadyState=0.5 в v_rest. Конвенция движка —
        // аргумент `-(v + midpoint)/scale`, поэтому half-activation при v=-65
        // даёт midpoint = +65 (НЕ -65).
        let gate = gate_kinetics_to_markov(
            GateKinetics::TauInf {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 65.0,
                    scale: 10.0,
                }),
                tau: TauForm::Fixed(1.0),
            },
            1,
        );
        // v_rest = -65 → inf = 1/(1+exp(0)) = 0.5.
        let mut ch = StochasticMarkovActiveChannel::new(
            vec![gate.clone()],
            vec![1.0],
            0.0,
            64.0,
            64,
            -65.0,
            1234,
        );
        let steps = 100_000usize;
        let mut sum = 0.0f64;
        for _ in 0..steps {
            sum += f64::from(ch.step_open(-65.0, 0.01));
        }
        let mean = sum / steps as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean open fraction = {mean}");

        // Два канала с разными сидами — разные последовательности.
        let mut c1 = StochasticMarkovActiveChannel::new(
            vec![gate.clone()],
            vec![1.0],
            0.0,
            64.0,
            64,
            -65.0,
            1,
        );
        let mut c2 =
            StochasticMarkovActiveChannel::new(vec![gate], vec![1.0], 0.0, 64.0, 64, -65.0, 2);
        let mut diff = 0;
        for _ in 0..64 {
            let f1 = c1.step_open(-65.0, 0.01);
            let f2 = c2.step_open(-65.0, 0.01);
            if (f1 - f2).abs() > 1e-6 {
                diff += 1;
            }
        }
        assert!(diff > 0, "разные сиды должны давать разный шум");
    }

    /// Nernst-обращение: [Ca]int=0.1 µM / [Ca]ext=2 mM при 6.3°C → ≈ +119 мВ
    /// (канонический Ca-реверс). `ca_ext <= 0` → 0 (LEMS guard).
    #[test]
    fn nernst_reversal_sanity() {
        let e = nernst_reversal_mv(1e-4, 2.0, 6.3 + 273.15);
        assert!((e - 119.2).abs() < 1.0, "nernst e_rev = {e}");
        assert_eq!(nernst_reversal_mv(1e-4, 0.0, 300.0), 0.0);
    }

    /// decayingPool: flux-масштаб сферической оболочки сходится к аналитике, а
    /// ODE выходит на steady state `rest + iCa·flux·tau` (постоянный 1 nA).
    #[test]
    fn decaying_pool_steady_state_and_relaxation() {
        // Сферическая сома R=10 мкм (площадь 4πR²), оболочка 0.1 мкм.
        let area_um2 = 4.0 * PI * 10.0 * 10.0;
        let flux = decaying_pool_flux_scale(area_um2, 0.1);
        // V_shell = (4π/3)(R³−(R−d)³) → flux ≈ 0.0417 мМ/мс на нА (аналитика).
        assert!((flux - 0.0417).abs() < 1e-3, "flux = {flux}");

        let pool = CaPool {
            resting_mm: 1e-4,
            tau_ms: 100.0,
            flux_scale: flux,
        };
        let i_ca = 1.0f32; // нА
        let steady = 1e-4 + flux * 100.0 * i_ca;
        let mut ca = 1e-4f32;
        for _ in 0..5000 {
            ca = pool.step(ca, i_ca, 0.1); // 500 мс суммарно = 5τ
        }
        // 5τ → e^-5 ≈ 0.007, т.е. в пределах ~0.03 мМ от steady.
        assert!((ca - steady).abs() < 0.05, "ca = {ca}, steady = {steady}");

        // Без тока [Ca] возвращается к покою (релаксация), и не уходит < 0.
        let mut ca2 = 4.0f32;
        for _ in 0..5000 {
            ca2 = pool.step(ca2, 0.0, 0.1);
        }
        assert!((ca2 - 1e-4).abs() < 0.05, "decay to rest: ca = {ca2}");
    }

    /// Ca²⁺-подсистема (Nernst + decaying pool) против NEURON `cacum.mod` +
    /// ohmic Ca-канал (`cachan_simple.mod`).
    ///
    /// Единственная сферическая сома (diam = L = 20 µm, area = 1256.6 µm²) с
    /// пассивной утечкой (0.3 mS/cm² → −65 mV), всегда-открытым ohmic Ca-каналом
    /// (`gbar = 1 mS/cm²`, `e_rev = Nernst(ca_ext/ca_int)`) и тонкой оболочкой
    /// (`decayingPool`, depth = 0.1 µm, tau = 100 ms, cai0 = 1e-4 mM).
    ///
    /// Эталон `REF_V_CA`/`REF_CAI_CA` генерится `gen_ca_ref.py` (NEURON
    /// `secondorder=0` — backward Euler, как Hines-солвер). Единственная
    /// осознанная расходимость — геометрия оболочки: NEURON `cacum.mod`
    /// считает объём как `area·depth` (тонкая оболочка), а LEMS
    /// `decayingPoolConcentrationModel` — точную сферическую оболочку
    /// `(4π/3)(R³−(R−d)³)`. При depth/R = 0.1/10 = 1e-2 объёмы различаются на
    /// ~1% (flux-масштаб pillars на ~1% больше) — отсюда допуск по [Ca] 1.5%
    /// и по V 0.5 мВ (измерено: max rel |ΔCa| ≈ 0.9%, max |ΔV| ≈ 0.08 мВ).
    #[test]
    fn ca_subsystem_matches_neuron_cacum() {
        const REF_V_CA: [f32; 8] = [
            -65.000_000,
            0.379_119,
            15.061_532,
            14.028_261,
            9.739_405,
            5.033_044,
            1.313_686,
            -0.889_314,
        ];
        const REF_CAI_CA: [f32; 8] = [
            0.000_100, 0.042_215, 0.067_031, 0.095_532, 0.146_739, 0.236_989, 0.349_230, 0.440_467,
        ];

        let m = MorphoIR {
            id: "ca_soma".into(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        // pas: 0.3 mS/cm² → Rm = 3333.33 Ω·cm², e_pas = −65 мВ.
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let soma = tree.root;
        let area_um2 = tree.compartments[soma].area_um2;
        let area_cm2 = area_um2 * 1e-8;

        let mut solver = ActiveCableSolver::new(tree, -65.0, 0.05, 6.3);
        solver.set_v(soma, -65.0);
        // Всегда-открытый ohmic Ca-канал (пустые gates → fopen = 1).
        let g_max_us = 1.0e-3 * area_cm2 * 1e6;
        solver.set_nernst_channels(
            soma,
            vec![NernstActiveChannel::new(
                vec![],
                vec![],
                2.0,
                g_max_us,
                -65.0,
            )],
        );
        let pool = CaPool {
            resting_mm: 1e-4,
            tau_ms: 100.0,
            flux_scale: decaying_pool_flux_scale(area_um2, 0.1),
        };
        solver.set_ca_pool(soma, pool, 1e-4);

        let dt = 0.05f32;
        let times = [0.0f32, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0];
        let steps_at: Vec<usize> = times.iter().map(|t| (t / dt).round() as usize).collect();
        let mut v_ours = vec![solver.v(soma)];
        let mut ca_ours = vec![solver.ca(soma)];
        let zero = [0.0f32; 1];
        for step in 1..=1000usize {
            solver.step(&zero);
            if steps_at.contains(&step) {
                v_ours.push(solver.v(soma));
                ca_ours.push(solver.ca(soma));
            }
        }

        let mut dv = 0.0f32;
        let mut dca_rel = 0.0f32;
        for k in 0..8 {
            dv = dv.max((v_ours[k] - REF_V_CA[k]).abs());
            dca_rel = dca_rel.max((ca_ours[k] - REF_CAI_CA[k]).abs() / REF_CAI_CA[k].max(1e-6));
        }
        println!("ca subsystem vs NEURON: max |ΔV| = {dv:.6} mV, max rel |ΔCa| = {dca_rel:.4}");
        assert!(dv < 0.5, "V: max |ΔV| = {dv:.4} mV");
        assert!(dca_rel < 0.015, "[Ca]: max rel err = {dca_rel:.4}");
    }

    /// `channelPopulationNernst` mean-field: среднее по ансамблю стохастических
    /// Ca-каналов сходится к детерминистическому `NernstActiveChannel` (тот же
    /// p_open, та же суммарная проводимость `single_cond·N`). Транзитивно это
    /// и NEURON-parity: детерминистический предел — тот самый
    /// `channelDensityNernst`, что сверен в `ca_subsystem_matches_neuron_cacum`.
    /// Канальный шум ∝ 1/√N усредняется по N_SEEDS независимым seed'ам.
    #[test]
    fn channel_population_nernst_mean_matches_density() {
        const N_CHANNELS: usize = 2048;
        const N_SEEDS: usize = 16;

        let m = MorphoIR {
            id: "ca_soma".into(),
            segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
            segment_groups: vec![],
        };
        let params = CableParams {
            rm_ohm_cm2: 3333.333,
            cm_uf_per_cm2: 1.0,
            ra_ohm_cm: 100.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let soma = tree.root;
        let area_um2 = tree.compartments[soma].area_um2;
        let area_cm2 = area_um2 * 1e-8;
        let g_max_us = 1.0e-3 * area_cm2 * 1e6;

        // Один и тот же сигмоидный gate в двух представлениях (HH-кинетика для
        // детерминистического канала, Markov-схема для стохастического).
        let kinetics = GateKinetics::TauInf {
            steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: -100.0,
                scale: 10.0,
            }),
            tau: TauForm::Fixed(0.1),
        };
        let markov_gate = gate_kinetics_to_markov(kinetics.clone(), 1);
        let pool = CaPool {
            resting_mm: 1e-4,
            tau_ms: 100.0,
            flux_scale: decaying_pool_flux_scale(area_um2, 0.1),
        };

        let dt = 0.05f32;
        let times = [0.0f32, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0];
        let steps_at: Vec<usize> = times.iter().map(|t| (t / dt).round() as usize).collect();
        let zero = [0.0f32; 1];

        // Детерминистический эталон (NernstActiveChannel, тот же gate).
        let mut det = ActiveCableSolver::new(tree.clone(), -65.0, 0.05, 6.3);
        det.set_v(soma, -65.0);
        det.set_nernst_channels(
            soma,
            vec![NernstActiveChannel::new(
                vec![(kinetics, 1)],
                vec![1.0],
                2.0,
                g_max_us,
                -65.0,
            )],
        );
        det.set_ca_pool(soma, pool, 1e-4);
        let mut det_v = vec![det.v(soma)];
        let mut det_ca = vec![det.ca(soma)];
        for step in 1..=1000usize {
            det.step(&zero);
            if steps_at.contains(&step) {
                det_v.push(det.v(soma));
                det_ca.push(det.ca(soma));
            }
        }

        // Среднее по независимым seed'ам стохастического канала.
        let base_seed = 0xCA11_CAF0_0DBA_D00Du64;
        let mut acc_v = [0.0f64; 8];
        let mut acc_ca = [0.0f64; 8];
        for seed in 0..N_SEEDS {
            let mut solver = ActiveCableSolver::new(tree.clone(), -65.0, 0.05, 6.3);
            solver.set_v(soma, -65.0);
            solver.set_stochastic_nernst_channels(
                soma,
                vec![StochasticNernstActiveChannel::new(
                    vec![markov_gate.clone()],
                    vec![1.0],
                    2.0,
                    g_max_us,
                    N_CHANNELS,
                    -65.0,
                    base_seed.wrapping_add(seed as u64),
                )],
            );
            solver.set_ca_pool(soma, pool, 1e-4);
            acc_v[0] += solver.v(soma) as f64;
            acc_ca[0] += solver.ca(soma) as f64;
            let mut k = 0usize;
            for step in 1..=1000usize {
                solver.step(&zero);
                if steps_at.contains(&step) {
                    k += 1;
                    acc_v[k] += solver.v(soma) as f64;
                    acc_ca[k] += solver.ca(soma) as f64;
                }
            }
        }

        let nf = N_SEEDS as f64;
        let mut dv = 0.0f64;
        let mut dca_rel = 0.0f64;
        for k in 0..8 {
            dv = dv.max((acc_v[k] / nf - det_v[k] as f64).abs());
            dca_rel =
                dca_rel.max((acc_ca[k] / nf - det_ca[k] as f64).abs() / det_ca[k].max(1e-6) as f64);
        }
        println!(
            "channelPopulationNernst mean vs density: max |ΔV| = {dv:.6} mV, max rel |ΔCa| = {dca_rel:.4}"
        );
        assert!(dv < 0.1, "V: mean-field max |ΔV| = {dv:.4} mV");
        assert!(
            dca_rel < 0.01,
            "[Ca]: mean-field max rel err = {dca_rel:.4}"
        );
    }
}
