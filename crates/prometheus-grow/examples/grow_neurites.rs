//! grow_neurites.rs — Stage 2: real neurite outgrowth from the patch somas,
//! then wiring by contact geometry instead of the soma-distance stub.
//!
//! Each neuron grows:
//!   * an axon steered toward its cell type's projection region (identity-
//!     directed guidance via `PointAttraction` — the per-type target is the
//!     centroid of that type's postsynaptic partners' somas, learned from the
//!     patch wiring),
//!   * a dendrite arbor via `NetmorphGrowth` (elongation + branching + Rall).
//!
//! Synapses are then formed by the grown geometry and compared against:
//!   * the soma-stub baseline (distance only),
//!   * per-neuron identity (`type`/`class`) keyed recognition, both on the
//!     soma stub and on the grown geometry — this isolates whether identity-
//!     directed growth adds precision on top of identity (Stage 2 criterion).
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_neurites -- \
//!     data/reference/patch_seed_big.json data/reference/manc_vnc_gene_annot.csv
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use prometheus_grow::{
    BalancedForces, Bounds, CellType, Grid, NetmorphGrowth, PointAttraction, Recognition,
    TagRecognition, Vec3, World,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Patch {
    #[serde(rename = "seed_bodyId")]
    #[allow(dead_code)]
    seed_body_id: i64,
    neurons: Vec<Neuron>,
    edges: Vec<(usize, usize, f64)>,
}

#[derive(Deserialize)]
struct Neuron {
    #[serde(rename = "bodyId")]
    #[allow(dead_code)]
    body_id: u64,
    #[serde(rename = "predictedNt")]
    predicted_nt: String,
    #[serde(rename = "type")]
    type_name: String,
    class: String,
    pos: [f64; 3],
}

impl Neuron {
    fn cell_type(&self) -> CellType {
        match self.predicted_nt.as_str() {
            "gaba" | "glutamate" => CellType::Inhibitory,
            _ => CellType::Excitatory,
        }
    }
}

fn reciprocal_gaba(edges: &HashSet<(usize, usize)>, gaba: &HashSet<usize>) -> usize {
    edges
        .iter()
        .filter(|&&(a, b)| a < b && gaba.contains(&a) && gaba.contains(&b) && edges.contains(&(b, a)))
        .count()
}

fn precision_recall(
    grown: &HashSet<(usize, usize)>,
    real: &HashSet<(usize, usize)>,
) -> (f64, f64, f64) {
    let inter = grown.intersection(real).count();
    let prec = inter as f64 / grown.len().max(1) as f64;
    let recall = inter as f64 / real.len().max(1) as f64;
    let f1 = if prec + recall > 0.0 {
        2.0 * prec * recall / (prec + recall)
    } else {
        0.0
    };
    (prec, recall, f1)
}

fn bbox(pos: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in pos {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    (lo, hi)
}

/// Soma-stub world: one stub axon at each soma (so `form_synapses*` sees the
/// soma as the only contact point), optionally tagged with a per-neuron label.
fn soma_world(
    grid: Grid,
    bounds: Bounds,
    neurons: &[Neuron],
    types: &[CellType],
    tags: &[String],
) -> World {
    let mut w = World::new(grid, 1.0, bounds, 7);
    for (i, x) in neurons.iter().enumerate() {
        let p = Vec3::new(x.pos[0], x.pos[1], x.pos[2]);
        let id = w.add_cell(p, types[i], 300.0);
        debug_assert_eq!(id, i);
        w.cells[i].tag = tags[i].clone();
        w.new_neurite(id, true, 1.0);
    }
    w
}

/// Grow real neurites from every soma for `steps` steps. Each axon is steered
/// by [`PointAttraction`] toward its type's projection region (`targets`),
/// falling back to the soma itself for types with no out-edges. Returns a
/// world with `neuron i == cell i` and a cleared synapse list.
fn grow_patch(
    grid: Grid,
    bounds: Bounds,
    neurons: &[Neuron],
    types: &[CellType],
    tags: &[String],
    targets: &HashMap<String, Vec3>,
    steps: usize,
) -> World {
    let mut w = World::new(grid, 1.0, bounds, 7);
    w.max_neurites = 60_000;

    for (i, x) in neurons.iter().enumerate() {
        let p = Vec3::new(x.pos[0], x.pos[1], x.pos[2]);
        let id = w.add_cell(p, types[i], 300.0);
        debug_assert_eq!(id, i);
        w.cells[id].tag = tags[i].clone();

        // Axon: identity-directed guidance toward the type's projection region.
        let target = *targets.get(&tags[i]).unwrap_or(&p);
        let axon = w.new_neurite(id, true, 0.3);
        let guide = PointAttraction::new(target, 500.0).stop_distance(1500.0);
        w.add_neurite_module(id, axon, Box::new(guide));

        // Dendrite: local arbor via NetmorphGrowth, seeded in a random direction.
        let dend = w.new_neurite(id, false, 1.0);
        let init = w.random_vec(120.0);
        w.extend_neurite(id, dend, init);
        w.add_neurite_module(
            id,
            dend,
            Box::new(
                NetmorphGrowth::new(300.0, 0.74, 5.0, 2.0, 50.0, 0.0, 0.0)
                    .max_prob(0.05)
                    .eri_std(0.1)
                    .step(100.0)
                    .branch(BalancedForces::netmorph_defaults()),
            ),
        );
    }

    for _ in 0..steps {
        w.step();
    }
    // Discard synapses that formed incidentally during the growth loop — the
    // caller re-wires from the final geometry.
    w.synapses.clear();
    w
}

fn report(
    label: &str,
    grown: &HashSet<(usize, usize)>,
    real: &HashSet<(usize, usize)>,
    gaba: &HashSet<usize>,
) {
    let (prec, recall, f1) = precision_recall(grown, real);
    println!(
        "{label:<32} edges={:>6}  recip={:>5}  prec={prec:.2} recall={recall:.2} F1={f1:.2}",
        grown.len(),
        reciprocal_gaba(grown, gaba)
    );
}

fn main() {
    let patch_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/reference/patch_seed_big.json".to_string());
    let annot_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data/reference/manc_vnc_gene_annot.csv".to_string());
    assert!(Path::new(&patch_path).exists(), "нет патча: {patch_path}");
    assert!(Path::new(&annot_path).exists(), "нет annot: {annot_path}");

    let patch: Patch =
        serde_json::from_str(&std::fs::read_to_string(&patch_path).expect("патч")).expect("JSON");

    let n = patch.neurons.len();
    let types: Vec<CellType> = patch.neurons.iter().map(|x| x.cell_type()).collect();
    let type_tags: Vec<String> = patch
        .neurons
        .iter()
        .map(|x| if x.type_name.is_empty() { "?".to_string() } else { x.type_name.clone() })
        .collect();
    let class_tags: Vec<String> = patch
        .neurons
        .iter()
        .map(|x| if x.class.is_empty() { "?".to_string() } else { x.class.clone() })
        .collect();
    let gaba: HashSet<usize> = types
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == CellType::Inhibitory)
        .map(|(i, _)| i)
        .collect();
    let edge_pairs: Vec<(usize, usize)> = patch.edges.iter().map(|&(a, b, _)| (a, b)).collect();
    let real: HashSet<(usize, usize)> = edge_pairs.iter().copied().collect();
    let pos: Vec<Vec3> = patch
        .neurons
        .iter()
        .map(|x| Vec3::new(x.pos[0], x.pos[1], x.pos[2]))
        .collect();

    // Per-type projection region: centroid of the postsynaptic partners' somas
    // for each presynaptic cell type (learned from the patch wiring — the
    // same type-level signal the `type`-keyed recognition uses).
    let mut type_out_sum: HashMap<String, Vec3> = HashMap::new();
    let mut type_out_cnt: HashMap<String, usize> = HashMap::new();
    for &(a, b, _) in &patch.edges {
        let t = type_tags[a].clone();
        let e = type_out_sum.entry(t.clone()).or_insert(Vec3::ZERO);
        *e = *e + pos[b];
        *type_out_cnt.entry(t).or_default() += 1;
    }
    let type_targets: HashMap<String, Vec3> = type_out_sum
        .iter()
        .map(|(t, s)| (t.clone(), *s * (1.0 / type_out_cnt[t] as f64)))
        .collect();

    let (min, max) = bbox(&pos);
    let margin = 1000.0;
    let origin = min - Vec3::new(margin, margin, margin);
    let span = max - min + Vec3::new(2.0 * margin, 2.0 * margin, 2.0 * margin);
    let cell = span.x.max(span.y).max(span.z) / 32.0;
    let grid = Grid::new(33, 33, 33, cell, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(cell * 33.0, cell * 33.0, cell * 33.0),
    };

    println!("=== Stage 2: identity-направленный рост + проводка ===");
    println!("нейронов: {n}  (GABA: {})  рёбер: {}", gaba.len(), real.len());
    println!("реально: recip GABA = {}", reciprocal_gaba(&real, &gaba));
    println!("grid cell ≈ {cell:.0} нм, patch span ≈ {:.0} нм", span.x.max(span.y).max(span.z));
    println!("типов с проекционным регионом: {} / 193", type_targets.len());

    // 1. No identity: grown geometry + distance rule over neurite points.
    let mut w = grow_patch(grid, bounds, &patch.neurons, &types, &class_tags, &type_targets, 80);
    w.form_synapses(16000.0);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    report("grown+dist=16000", &grown, &real, &gaba);

    // 2. No identity: grown geometry + bouton/spine contact.
    let mut w = grow_patch(grid, bounds, &patch.neurons, &types, &class_tags, &type_targets, 80);
    w.make_excrescences(1500.0, 1200.0);
    w.set_recognition(Recognition::all());
    w.form_synapses_by_contact(800.0);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    report("grown+contact", &grown, &real, &gaba);

    // 3. Identity: type-keyed recognition, soma-stub vs directed growth.
    let dist = 16000.0;
    let type_rec = TagRecognition::learned_from_edges(&type_tags, &edge_pairs);

    // Scale the learned connect-rate to the real density (same normalization
    // as grow_class_wiring.rs).
    let mut probe = soma_world(grid, bounds, &patch.neurons, &types, &type_tags);
    probe.form_synapses_tagged(dist, &type_rec);
    let raw_edges = probe.to_graph().edges.len().max(1) as f64;
    let scale = (real.len() as f64 / raw_edges).min(50.0);
    let mut scaled = type_rec.clone();
    for p in scaled.prob.values_mut() {
        *p = (*p * scale).min(1.0);
    }

    // soma stub + identity
    let mut ws = soma_world(grid, bounds, &patch.neurons, &types, &type_tags);
    ws.form_synapses_tagged(dist, &scaled);
    let gs = ws.to_graph();
    let gs_edges: HashSet<(usize, usize)> = gs.edges.iter().map(|e| (e.pre, e.post)).collect();
    report("type-tagged (soma)", &gs_edges, &real, &gaba);

    // directed growth + identity
    let mut wg = grow_patch(grid, bounds, &patch.neurons, &types, &type_tags, &type_targets, 80);
    wg.form_synapses_tagged(dist, &scaled);
    let gg = wg.to_graph();
    let gg_edges: HashSet<(usize, usize)> = gg.edges.iter().map(|e| (e.pre, e.post)).collect();
    report("type-tagged (directed-grown)", &gg_edges, &real, &gaba);

    // directed growth + identity + contact geometry
    let mut wc = grow_patch(grid, bounds, &patch.neurons, &types, &type_tags, &type_targets, 80);
    wc.make_excrescences(1500.0, 1200.0);
    wc.form_synapses_by_contact_tagged(800.0, &scaled);
    let gc = wc.to_graph();
    let gc_edges: HashSet<(usize, usize)> = gc.edges.iter().map(|e| (e.pre, e.post)).collect();
    report("type-tagged (dir-grown+contact)", &gc_edges, &real, &gaba);
}
