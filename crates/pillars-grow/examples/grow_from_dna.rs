//! Load the real fly DNA (PROMETHEUS `dna_nt_map.csv`) and grow from it.
//!
//! ```text
//! dna_nt_map.csv  ──► FlyGenome { nt_gene: Gad1, morphogen: dpp, guidance: fra }
//!                          │
//!                          ▼
//!                Gad1 → sign -1 → CellType::Inhibitory (GABA)
//!                          │
//!                          ▼
//!         1 progenitor (Gad1) ──divide──► 2 GABA cells ──grow axons──►
//!                          │
//!                          ▼
//!            mutual inhibition → adaptive LIF → anti-phase
//! ```
//!
//! ```bash
//! cargo run -p pillars-grow --example grow_from_dna -- C:/Users/anton/Desktop/Проекты/PROMETHEUS/dna_nt_map.csv
//! ```

use std::fs;
use std::path::Path;

use pillars_grow::{
    parse_dna_map, Bounds, ChemoAttraction, DivideOnce, FlyGenome, Grid, Secretor, Substance,
    SynapseKind, Vec3, World,
};

/// Minimal adaptive LIF (aEIF-lite) with mutual inhibition — the classic
/// half-center oscillator. Returns spike times of the two neurons.
fn run_lif(w01: f64, w10: f64, dt: f64, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let tau_v = 10.0;
    let tau_a = 80.0;
    let tau_syn = 5.0;
    let v_th = 1.0;
    let drive = 1.6;
    let delta = 0.7;

    // Asymmetric initial conditions break the in-phase symmetry so the pair
    // falls into anti-phase, not a locked pair.
    let mut v = [0.2f64, 0.0];
    let mut a = [0.0f64; 2];
    let mut i_syn = [0.0f64; 2];
    let mut t0 = Vec::new();
    let mut t1 = Vec::new();

    let syn_exp = (-dt / tau_syn).exp();
    let w = [w01, w10];

    for step in 0..steps {
        i_syn[0] *= syn_exp;
        i_syn[1] *= syn_exp;
        for k in 0..2 {
            v[k] += dt * ((drive - v[k]) / tau_v + i_syn[k] - a[k]);
            a[k] += dt * (-a[k] / tau_a);
        }
        for k in 0..2 {
            if v[k] >= v_th {
                v[k] = 0.0;
                a[k] += delta;
                if k == 0 {
                    t0.push(step as f64 * dt);
                } else {
                    t1.push(step as f64 * dt);
                }
                i_syn[1 - k] -= w[k];
            }
        }
    }
    (t0, t1)
}

/// Dominant period (ms) via autocorrelation of a binary spike train.
fn autocorr_period(s: &[f64], dt: f64) -> f64 {
    if s.len() < 2 {
        return 0.0;
    }
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let n = s.len();
    let mut best_lag = 1usize;
    let mut best = f64::NEG_INFINITY;
    for lag in (n / 20).max(1)..(n / 2) {
        let mut acc = 0.0;
        for i in 0..(n - lag) {
            acc += (s[i] - mean) * (s[i + lag] - mean);
        }
        if acc > best {
            best = acc;
            best_lag = lag;
        }
    }
    best_lag as f64 * dt
}

/// Phase of a binary spike train at frequency `f` via Goertzel.
fn goertzel_phase(s: &[f64], f: f64, dt: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f * dt;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (t, &v) in s.iter().enumerate() {
        let theta = w * t as f64;
        re += v * theta.cos();
        im += v * theta.sin();
    }
    im.atan2(re).to_degrees().rem_euclid(360.0)
}

/// Phase offset (deg) of `b` relative to `a` at the shared firing frequency.
fn phase_offset(a: &[f64], b: &[f64], dt: f64) -> f64 {
    let period = autocorr_period(a, dt);
    if period <= 0.0 {
        return 0.0;
    }
    let f = 1.0 / period;
    let pa = goertzel_phase(a, f, dt);
    let pb = goertzel_phase(b, f, dt);
    (pb - pa).rem_euclid(360.0)
}

fn main() {
    // 1. Load the real DNA map (path from argv, with sensible fallbacks).
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let candidates = [
            "C:/Users/anton/Desktop/Проекты/PROMETHEUS/dna_nt_map.csv",
            "../../PROMETHEUS/dna_nt_map.csv",
            "PROMETHEUS/dna_nt_map.csv",
        ];
        candidates
            .iter()
            .find(|p| Path::new(p).exists())
            .unwrap_or(&candidates[0])
            .to_string()
    });
    let csv = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("не могу прочитать DNA-карту {path}: {e}"));
    let map = parse_dna_map(&csv);
    println!("=== ДНК загружена: {} генов из {path} ===", map.len());
    for g in &map {
        println!(
            "  {} → {} → {}",
            g.gene,
            g.neurotransmitter,
            if g.sign < 0 { "тормозной (−)" } else { "возбуждающий (+)" }
        );
    }

    // 2. Instantiate the half-center genome from real genes.
    let genome = FlyGenome::half_center();
    let cell_type = genome
        .cell_type(&map)
        .unwrap_or_else(|| panic!("ген {} не найден в DNA-карте", genome.nt_gene));
    let nt = genome.neurotransmitter(&map).unwrap_or_default();
    println!(
        "\ngenome: {} (→ {nt} → {cell_type:?}), morphogen={}, guidance={}",
        genome.nt_gene, genome.morphogen_gene, genome.guidance_gene
    );

    // 3. Grow: 1 Gad1 progenitor divides once → 2 GABA cells, axons chase dpp.
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    world.add_substance(Substance::new(
        &genome.morphogen_gene,
        0.1,
        0.001,
        &world.grid,
        vec![],
    ));

    let c = world.add_cell(Vec3::ZERO, cell_type, 4.0);
    world.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
    world.add_soma_module(c, Box::new(Secretor::new(&genome.morphogen_gene, 0.05)));

    for _ in 0..30 {
        world.step();
        if world.cells.len() >= 2 {
            break;
        }
    }
    world.max_cells = world.cells.len();
    for ci in 0..world.cells.len() {
        let n = world.new_neurite(ci, true, 0.5);
        let mut chemo = ChemoAttraction::new(&genome.morphogen_gene);
        chemo.noise_weight = 0.1;
        world.add_neurite_module(ci, n, Box::new(chemo));
    }
    for _ in 0..120 {
        world.step();
    }

    let g = world.to_graph();
    println!(
        "\ngrown graph: {} nodes (все {}), {} edges",
        g.nodes.len(),
        genome.nt_gene,
        g.edges.len()
    );
    for e in &g.edges {
        println!(
            "  {} -> {}  kind={:?}  w={:.3}",
            e.pre, e.post, e.kind, e.weight
        );
    }

    let w01 = g.edges.iter().find(|e| e.pre == 0 && e.post == 1).map(|e| e.weight).unwrap_or(0.0);
    let w10 = g.edges.iter().find(|e| e.pre == 1 && e.post == 0).map(|e| e.weight).unwrap_or(0.0);
    let both_inhibitory = g.edges.iter().all(|e| e.kind == SynapseKind::Inhibitory);
    let mutual = w01 > 0.0 && w10 > 0.0;
    println!(
        "half-center motif: mutual={mutual}  both_inhibitory={both_inhibitory}  w01={w01:.3}  w10={w10:.3}"
    );
    if !mutual {
        println!("FAIL: growth did not produce a mutual pair");
        return;
    }

    // 4. Functional check: the grown GABA pair oscillates in anti-phase.
    let dt = 0.1;
    let steps = 30_000;
    let (t0, t1) = run_lif(w01, w10, dt, steps);
    let train0: Vec<f64> = {
        let mut s = vec![0.0f64; steps];
        for &t in &t0 {
            let i = (t / dt) as usize;
            if i < steps {
                s[i] = 1.0;
            }
        }
        s
    };
    let train1: Vec<f64> = {
        let mut s = vec![0.0f64; steps];
        for &t in &t1 {
            let i = (t / dt) as usize;
            if i < steps {
                s[i] = 1.0;
            }
        }
        s
    };
    let phase = phase_offset(&train0, &train1, dt);
    let period = autocorr_period(&train0, dt);

    let mut events: Vec<(f64, usize)> = t0
        .iter()
        .map(|&t| (t, 0usize))
        .chain(t1.iter().map(|&t| (t, 1usize)))
        .collect();
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let switches = events.windows(2).filter(|w| w[0].1 != w[1].1).count();
    let total = events.len().saturating_sub(1);
    let alternation = if total > 0 {
        switches as f64 / total as f64
    } else {
        0.0
    };

    println!(
        "\nLIF result: spikes0={} spikes1={} period={:.1}ms phase={phase:.0}° alternation={:.0}%",
        t0.len(),
        t1.len(),
        period,
        alternation * 100.0
    );
    let verdict = if alternation >= 0.95 {
        "ANTI-PHASE (strict alternation) ✓"
    } else {
        "not anti-phase"
    };
    println!("  → {verdict}");
    println!(
        "\nитог: ДНК ({}) → рост → 2×{}(GABA) → взаимное торможение → антифаза",
        path,
        genome.nt_gene
    );
}
