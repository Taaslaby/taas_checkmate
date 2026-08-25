//! grow_vs_real.rs — drive the growth engine on a REAL connectome patch.
//!
//! Loads `patch_seed_big.json` (513 real neurons with positions + predicted
//! neurotransmitter, 34621 real edges), seeds the [`World`] with the real
//! somas/types, and forms synapses with the engine's connectivity rule
//! (distance + type recognition). It then compares the grown wiring against the
//! real edges **at the edge level** (precision / recall / reciprocal-GABA).
//!
//! On top of the built-in rules it learns the **partner-recognition transition
//! rule** from the real edges ([`Recognition::learned_from_edges`]) — the
//! "who wires to whom" table that replaces the hard-coded `same_type` gate.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_vs_real -- \
//!     ../data/reference/patch_seed_big.json
//! ```

use std::collections::HashSet;
use std::path::Path;

use prometheus_grow::{Bounds, CellType, Grid, Recognition, Vec3, World};
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
    #[serde(rename = "predictedNt")]
    predicted_nt: String,
    pos: [f64; 3],
}

impl Neuron {
    /// NT → sign (same convention as gene_physiology.NT_GENE):
    ///   gaba/glutamate → inhibitory (in Drosophila glutamate is inhibitory, via GluCl).
    /// Modulators/`unknown` fall back to excitatory in this 2-type model.
    fn cell_type(&self) -> CellType {
        match self.predicted_nt.as_str() {
            "gaba" | "glutamate" => CellType::Inhibitory,
            _ => CellType::Excitatory,
        }
    }
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../data/reference/patch_seed_big.json".to_string());
    assert!(Path::new(&path).exists(), "нет патча: {path}");

    let patch: Patch =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("не читается патч"))
            .expect("не парсится JSON");
    let n = patch.neurons.len();

    // ── real stats ────────────────────────────────────────────────
    let types: Vec<CellType> = patch.neurons.iter().map(|x| x.cell_type()).collect();
    let gaba: HashSet<usize> = patch
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, x)| x.cell_type() == CellType::Inhibitory)
        .map(|(i, _)| i)
        .collect();
    let edge_pairs: Vec<(usize, usize)> = patch.edges.iter().map(|&(a, b, _)| (a, b)).collect();
    let real: HashSet<(usize, usize)> = edge_pairs.iter().copied().collect();
    let pos: Vec<Vec3> = patch
        .neurons
        .iter()
        .map(|x| Vec3::new(x.pos[0], x.pos[1], x.pos[2]))
        .collect();

    let n_gaba = gaba.len();
    let real_recip = reciprocal_gaba(&real, &gaba);
    let real_w: Vec<f64> = patch.edges.iter().map(|&(_, _, w)| w).collect();
    let real_deg: Vec<f64> = out_degree(&real, n);
    let edge_dists: Vec<f64> = patch
        .edges
        .iter()
        .map(|&(a, b, _)| pos[a].dist(pos[b]))
        .collect();
    let non_edge_dists = sample_non_edge_dists(&pos, &real, 5000);

    println!("=== реальный участок ({path}) ===");
    println!(
        "  нейронов: {n}  (GABA/inhibitory: {n_gaba}, excitatory: {})",
        n - n_gaba
    );
    println!("  рёбер:    {}", real.len());
    println!("  recip GABA-пар: {real_recip}");
    println!("  вес (медиана):  {:.2}", median(&real_w));
    println!(
        "  степень (mean±std): {:.1}±{:.1}",
        real_deg.iter().sum::<f64>() / n as f64,
        std_dev(&real_deg)
    );
    println!(
        "  дистанция сома-сома: рёбра={:.0} нм, случайные пары={:.0} нм",
        median(&edge_dists),
        median(&non_edge_dists)
    );

    // ── seed the engine with the real patch ──────────────────────
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

    let seed_world = |rec: Option<Recognition>| -> World {
        let mut w = World::new(grid, 1.0, bounds, 7);
        if let Some(r) = rec {
            w.set_recognition(r);
        }
        for (i, x) in patch.neurons.iter().enumerate() {
            let p = Vec3::new(x.pos[0], x.pos[1], x.pos[2]);
            let id = w.add_cell(p, x.cell_type(), 300.0);
            debug_assert_eq!(id, i);
            w.new_neurite(id, true, 1.0); // axon tip at the soma
        }
        w
    };

    println!("\n=== сверка: правило движка vs реальные рёбра ===");
    println!(
        "{:<24} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "правило", "edges", "recip", "prec", "recall", "F1", "w_med"
    );
    for dist in [2000.0f64, 4000.0, 8000.0, 16000.0, 32000.0] {
        for (name, rec) in [
            ("distance only", Recognition::all()),
            ("distance + same_type", Recognition::same_type()),
        ] {
            let mut w = seed_world(Some(rec));
            w.form_synapses(dist);
            let g = w.to_graph();
            let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
            let recip = reciprocal_gaba(&grown, &gaba);
            let grown_w: Vec<f64> = g.edges.iter().map(|e| e.weight).collect();
            let (prec, recall, f1) = precision_recall(&grown, &real);
            println!(
                "{:<24} {:>6} {:>6} {:>5.2} {:>5.2} {:>5.2} {:>6.2}",
                format!("dist={dist:.0} {name}"),
                grown.len(),
                recip,
                prec,
                recall,
                f1,
                median(&grown_w)
            );
        }
    }

    // ── the learned transition rule (partner recognition) ────────
    let learned = Recognition::learned_from_edges(&types, &edge_pairs);
    let ex = CellType::Excitatory.as_usize();
    let inh = CellType::Inhibitory.as_usize();
    println!(
        "\n=== правило перехода (выучено из {0} реальных рёбер) ===",
        real.len()
    );
    println!(
        "  exc→exc  p={:.4}   exc→inh  p={:.4}",
        learned.prob[ex][ex], learned.prob[ex][inh]
    );
    println!(
        "  inh→exc  p={:.4}   inh→inh  p={:.4}   (same_type принуждал inh→inh = 1.0)",
        learned.prob[inh][ex], learned.prob[inh][inh]
    );

    // Learned rule, rescaled to the real edge count so the comparison is on
    // density, not on the (distance-confounded) absolute rate.
    let dist = 16000.0;
    let mut raw = seed_world(Some(learned));
    raw.form_synapses(dist);
    let raw_edges = raw.to_graph().edges.len().max(1) as f64;
    let scale = (real.len() as f64 / raw_edges).min(50.0);
    let mut scaled_prob = learned.prob;
    for r in scaled_prob.iter_mut() {
        for p in r.iter_mut() {
            *p = (*p * scale).min(1.0);
        }
    }
    let scaled_rec = Recognition {
        allowed: learned.allowed,
        prob: scaled_prob,
        max_dist: learned.max_dist,
    };
    let mut w = seed_world(Some(scaled_rec));
    w.form_synapses(dist);
    let g = w.to_graph();
    let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
    let (prec, recall, f1) = precision_recall(&grown, &real);
    println!(
        "\n=== learned vs same_type (dist=16000, scaled к {}) ===",
        real.len()
    );
    println!(
        "  learned rule:  edges={}  recip GABA={}  prec={prec:.2} recall={recall:.2} F1={f1:.2}",
        grown.len(),
        reciprocal_gaba(&grown, &gaba)
    );
    let mut st = seed_world(Some(Recognition::same_type()));
    st.form_synapses(dist);
    let st_g = st.to_graph();
    let st_grown: HashSet<(usize, usize)> = st_g.edges.iter().map(|e| (e.pre, e.post)).collect();
    println!(
        "  same_type:     edges={}  recip GABA={}  (реально recip={real_recip})",
        st_g.edges.len(),
        reciprocal_gaba(&st_grown, &gaba)
    );
}

/// Directed reciprocal GABA pairs: count a<b with both (a,b) and (b,a) in E.
fn reciprocal_gaba(edges: &HashSet<(usize, usize)>, gaba: &HashSet<usize>) -> usize {
    edges
        .iter()
        .filter(|&&(a, b)| {
            a < b && gaba.contains(&a) && gaba.contains(&b) && edges.contains(&(b, a))
        })
        .count()
}

fn out_degree(edges: &HashSet<(usize, usize)>, n: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n];
    for &(a, _) in edges {
        d[a] += 1.0;
    }
    d
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
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

/// Median soma distance of `k` random non-edge pairs (the connectivity baseline).
fn sample_non_edge_dists(pos: &[Vec3], real: &HashSet<(usize, usize)>, k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    let grid = Grid::new(2, 2, 2, 1.0, Vec3::ZERO);
    let mut w = World::new(
        grid,
        1.0,
        Bounds {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
        },
        7,
    );
    let n = pos.len();
    while out.len() < k {
        let a = (w.rand() * n as f64) as usize % n;
        let b = (w.rand() * n as f64) as usize % n;
        if a == b || real.contains(&(a, b)) {
            continue;
        }
        out.push(pos[a].dist(pos[b]));
    }
    out
}
