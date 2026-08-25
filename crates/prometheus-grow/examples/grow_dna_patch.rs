//! grow_dna_patch.rs — attach the DNA profile to the real connectome patch and
//! grow the skeleton *from the DNA*.
//!
//! ```text
//! manc_vnc_gene_annot.csv   bodyId → gene (Gad1/ChAT/VGlut/?) → sign + channels
//! patch_seed_big.json       513 neurons (bodyId, pos, predictedNt) + edges
//!                              │
//!                              ▼  (join on bodyId)
//!                DNA profile:  gene → sign → CellType   (who I am)
//!                dna_channel_map: Dscam1 → recognition  (who I wire to)
//!                              │
//!                              ▼
//!                        grown skeleton → F1/recip vs 34621 real edges
//! ```
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_dna_patch
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use prometheus_grow::{parse_channel_map, Bounds, CellType, FlyGenome, Grid, Vec3, World};
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
    fn predicted_type(&self) -> CellType {
        if self.predicted_nt == "gaba" || self.predicted_nt == "glutamate" {
            CellType::Inhibitory
        } else {
            CellType::Excitatory
        }
    }
}

/// One neuron's DNA-derived profile from `manc_vnc_gene_annot.csv`.
#[derive(Clone)]
struct GeneProfile {
    gene: String,
    sign: f64,
    g_h: f64,
}

impl GeneProfile {
    /// DNA gene → sign → type. `sign < 0` (Gad1/GABA, VGlut/GluCl in fly) →
    /// inhibitory; `sign > 0` (ChAT) → excitatory; unknown/modulator (`?`,
    /// sign == 0) falls back to the connectome's predicted neurotransmitter.
    fn cell_type(&self, predicted_nt: &str) -> CellType {
        if self.sign < 0.0 {
            CellType::Inhibitory
        } else if self.sign > 0.0 {
            CellType::Excitatory
        } else if predicted_nt == "gaba" || predicted_nt == "glutamate" {
            CellType::Inhibitory
        } else {
            CellType::Excitatory
        }
    }
}

fn parse_annot(csv: &str) -> HashMap<u64, GeneProfile> {
    let mut out = HashMap::new();
    for line in csv.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 10 {
            continue;
        }
        let body: u64 = f[0].trim().parse().unwrap_or(0);
        out.insert(
            body,
            GeneProfile {
                gene: f[2].trim().to_string(),
                sign: f[3].trim().parse().unwrap_or(0.0),
                g_h: f[7].trim().parse().unwrap_or(0.0),
            },
        );
    }
    out
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

fn reciprocal_gaba(edges: &HashSet<(usize, usize)>, gaba: &HashSet<usize>) -> usize {
    edges
        .iter()
        .filter(|&&(a, b)| {
            a < b && gaba.contains(&a) && gaba.contains(&b) && edges.contains(&(b, a))
        })
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

fn main() {
    let patch_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../data/reference/patch_seed_big.json".to_string());
    let annot_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "../data/reference/manc_vnc_gene_annot.csv".to_string());
    let ch_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "../genes/dna_channel_map.csv".to_string());
    assert!(Path::new(&patch_path).exists(), "нет патча: {patch_path}");
    assert!(Path::new(&annot_path).exists(), "нет annot: {annot_path}");
    assert!(Path::new(&ch_path).exists(), "нет channel map: {ch_path}");

    let patch: Patch =
        serde_json::from_str(&std::fs::read_to_string(&patch_path).expect("патч")).expect("JSON");
    let annot = parse_annot(&std::fs::read_to_string(&annot_path).expect("annot"));
    let ch_map = parse_channel_map(&std::fs::read_to_string(&ch_path).expect("channel map"));

    let n = patch.neurons.len();
    let real: HashSet<(usize, usize)> = patch.edges.iter().map(|&(a, b, _)| (a, b)).collect();

    // ── attach the DNA profile ──────────────────────────────────────
    let mut gene_counts: HashMap<String, usize> = HashMap::new();
    let mut g_h_by_gene: HashMap<String, Vec<f64>> = HashMap::new();
    let mut covered = 0usize;
    let mut agree = 0usize;
    let mut disagree = 0usize;
    let mut types: Vec<CellType> = Vec::with_capacity(n);

    for x in &patch.neurons {
        let (dna_type, gene, g_h) = match annot.get(&x.body_id) {
            Some(p) => {
                covered += 1;
                (p.cell_type(&x.predicted_nt), p.gene.clone(), p.g_h)
            }
            None => (x.predicted_type(), "?".to_string(), 0.0),
        };
        if dna_type == x.predicted_type() {
            agree += 1;
        } else {
            disagree += 1;
        }
        *gene_counts.entry(gene.clone()).or_default() += 1;
        g_h_by_gene.entry(gene).or_default().push(g_h);
        types.push(dna_type);
    }

    let gaba: HashSet<usize> = types
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == CellType::Inhibitory)
        .map(|(i, _)| i)
        .collect();

    println!("=== ДНК-профиль на реальном патче ===");
    println!("  нейронов: {n}");
    println!("  покрытие annot (bodyId → gene): {covered}/{n}");
    println!("  ген → знак (число нейронов, медиана g_h):");
    let mut genes: Vec<&String> = gene_counts.keys().collect();
    genes.sort();
    for gene in genes {
        let count = gene_counts[gene];
        let gh = median(g_h_by_gene.get(gene).map(|v| v.as_slice()).unwrap_or(&[]));
        println!("    {gene:<8} {count:>4}  g_h={gh:.3}");
    }
    println!("  согласие ДНК-знак vs predictedNt: {agree}/{n} (рассогласований: {disagree})");

    // DNA → wiring rule (adhesion Dscam1 → same-type recognition)
    let genome = FlyGenome::half_center();
    let rec = genome.recognition(&ch_map);
    println!(
        "  проводка из ДНК: {:?} → same-type recognition",
        genome.adhesion_gene
    );

    // ── grow the skeleton from the DNA ──────────────────────────────
    let pos: Vec<Vec3> = patch
        .neurons
        .iter()
        .map(|x| Vec3::new(x.pos[0], x.pos[1], x.pos[2]))
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

    println!("\n=== сверка: ДНК-выращенная проводка vs реальные рёбра ===");
    println!(
        "{:<30} {:>6} {:>6} {:>5} {:>6} {:>5}",
        "правило", "edges", "recip", "prec", "recall", "F1"
    );
    for dist in [8000.0f64, 16000.0, 32000.0] {
        let mut w = World::new(grid, 1.0, bounds, 7);
        w.set_recognition(rec);
        for (i, x) in patch.neurons.iter().enumerate() {
            let p = Vec3::new(x.pos[0], x.pos[1], x.pos[2]);
            let id = w.add_cell(p, types[i], 300.0);
            debug_assert_eq!(id, i);
            w.new_neurite(id, true, 1.0); // axon tip at the soma
        }
        w.form_synapses(dist);
        let g = w.to_graph();
        let grown: HashSet<(usize, usize)> = g.edges.iter().map(|e| (e.pre, e.post)).collect();
        let (prec, recall, f1) = precision_recall(&grown, &real);
        println!(
            "{:<30} {:>6} {:>6} {:>5.2} {:>6.2} {:>5.2}",
            format!("ДНК-тип + Dscam1 (dist={dist:.0})"),
            grown.len(),
            reciprocal_gaba(&grown, &gaba),
            prec,
            recall,
            f1
        );
    }

    let real_recip = reciprocal_gaba(&real, &gaba);
    println!("\nреально: edges={} recip GABA={real_recip}", real.len());
    println!("\nитог: ДНК (gene → знак, Dscam1 → проводка) навешена на {covered}/{n} нейронов;");
    println!("      согласие с коннектомом {agree}/{n}; F1 ограничен сома-дистанцией.");
}
