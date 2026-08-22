//! grow_class_wiring.rs — diagnose whether the per-neuron `class` identity
//! carries more wiring signal than the 2-type (exc/inh) gate.
//!
//! Same patch as `grow_vs_real`, but the partner-recognition rule is keyed by
//! the MANC `class` tag (motor/intrinsic/sensory/…) instead of `CellType`.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_class_wiring -- \
//!     data/reference/patch_seed_big.json data/reference/manc_vnc_gene_annot.csv
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use prometheus_grow::{Bounds, CellType, Grid, Recognition, TagRecognition, Vec3, World};
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
    body_id: u64,
    #[serde(rename = "predictedNt")]
    predicted_nt: String,
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

fn seed_world(
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

    // bodyId → class
    let mut class_of: HashMap<String, String> = HashMap::new();
    for line in std::fs::read_to_string(&annot_path).expect("annot").lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() >= 6 {
            class_of.insert(f[0].trim().to_string(), f[4].trim().to_string());
        }
    }

    let n = patch.neurons.len();
    let types: Vec<CellType> = patch.neurons.iter().map(|x| x.cell_type()).collect();
    let tags: Vec<String> = patch
        .neurons
        .iter()
        .map(|x| {
            let c = class_of.get(&x.body_id.to_string()).cloned().unwrap_or_default();
            if c.is_empty() {
                "?".to_string()
            } else {
                c
            }
        })
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

    println!("=== class-keyed recognition vs type-based rules ===");
    println!("нейронов: {n}  (GABA: {})  рёбер: {}", gaba.len(), real.len());
    println!("real recip GABA: {}", reciprocal_gaba(&real, &gaba));

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

    let class_rec = TagRecognition::learned_from_edges(&tags, &edge_pairs);

    let dist = 16000.0;
    println!("\n{:<26} {:>6} {:>6} {:>5} {:>6} {:>5}", "правило", "edges", "recip", "prec", "recall", "F1");

    // distance only
    let mut w = seed_world(grid, bounds, &patch.neurons, &types, &tags);
    w.set_recognition(Recognition::all());
    w.form_synapses(dist);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    let (prec, recall, f1) = precision_recall(&grown, &real);
    println!(
        "{:<26} {:>6} {:>6} {:>5.2} {:>6.2} {:>5.2}",
        "distance only",
        grown.len(),
        reciprocal_gaba(&grown, &gaba),
        prec, recall, f1
    );

    // same type
    let mut w = seed_world(grid, bounds, &patch.neurons, &types, &tags);
    w.set_recognition(Recognition::same_type());
    w.form_synapses(dist);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    let (prec, recall, f1) = precision_recall(&grown, &real);
    println!(
        "{:<26} {:>6} {:>6} {:>5.2} {:>6.2} {:>5.2}",
        "same_type",
        grown.len(),
        reciprocal_gaba(&grown, &gaba),
        prec, recall, f1
    );

    // class-learned, scaled to the real density
    let mut raw = seed_world(grid, bounds, &patch.neurons, &types, &tags);
    raw.form_synapses_tagged(dist, &class_rec);
    let raw_edges = raw.to_graph().edges.len().max(1) as f64;
    let scale = (real.len() as f64 / raw_edges).min(50.0);
    let mut scaled = class_rec.clone();
    for p in scaled.prob.values_mut() {
        *p = (*p * scale).min(1.0);
    }
    let mut w = seed_world(grid, bounds, &patch.neurons, &types, &tags);
    w.form_synapses_tagged(dist, &scaled);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    let (prec, recall, f1) = precision_recall(&grown, &real);
    println!(
        "{:<26} {:>6} {:>6} {:>5.2} {:>6.2} {:>5.2}",
        "class-learned (scaled)",
        grown.len(),
        reciprocal_gaba(&grown, &gaba),
        prec, recall, f1
    );

    // class-learned WITHOUT scaling (raw connect rate)
    let mut w = seed_world(grid, bounds, &patch.neurons, &types, &tags);
    w.form_synapses_tagged(dist, &class_rec);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    let (prec, recall, f1) = precision_recall(&grown, &real);
    println!(
        "{:<26} {:>6} {:>6} {:>5.2} {:>6.2} {:>5.2}",
        "class-learned (raw)",
        grown.len(),
        reciprocal_gaba(&grown, &gaba),
        prec, recall, f1
    );
}
