//! Evolve the growth genome so the grown half-center hits a target rhythm.
//!
//! ```text
//! genome (synapse_strength, chemo_speed, diffusion)
//!      │ mutate
//!      ▼
//! prometheus-grow ──► CSR ──► pillars HAAP ──► (phase, period)
//!      ▲                                      │
//!      └──────────── (1+1)-ES keeps the best ─┘
//! ```
//!
//! This is the reverse of "read the weights from the scan": instead we let
//! evolution **find the developmental rules** that produce the behaviour we
//! want — the "морф-параметры мухи" loop.
//!
//! ```bash
//! cargo run -p pillars-connectome --example evolve_halfcenter --release
//! ```

use rand::Rng;

use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use prometheus_grow::{
    Bounds, CellType, ChemoAttraction, DivideOnce, Grid, Secretor, Substance, Vec3, World,
};

const DT: f32 = 0.05; // ms
const TAU_SYN: f32 = 12.0;
const STEPS: usize = 30_000; // 1500 ms (≈ 8 cycles at ~5 Hz)
const DRIVE: f32 = 2.0;

/// Target behaviour: anti-phase (180°) at a 200 ms rhythm (~5 Hz).
const TARGET_PHASE: f32 = 180.0;
const TARGET_PERIOD: f32 = 200.0; // ms

/// The evolvable growth genome.
#[derive(Clone, Copy, Debug)]
struct GrowthParams {
    synapse_strength: f64,
    chemo_speed: f64,
    diffusion: f64,
}

fn grow(p: GrowthParams) -> (usize, Vec<(usize, usize, f32)>) {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    world.synapse_strength = p.synapse_strength;
    world.add_substance(Substance::new("A", p.diffusion, 0.001, &world.grid, vec![]));

    let c = world.add_cell(Vec3::ZERO, CellType::Inhibitory, 4.0);
    world.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
    world.add_soma_module(c, Box::new(Secretor::new("A", 0.05)));

    for _ in 0..30 {
        world.step();
        if world.cells.len() >= 2 {
            break;
        }
    }
    world.max_cells = world.cells.len();
    for ci in 0..world.cells.len() {
        let n = world.new_neurite(ci, true, 0.5);
        let mut chemo = ChemoAttraction::new("A");
        chemo.noise_weight = 0.1;
        chemo.speed = p.chemo_speed;
        world.add_neurite_module(ci, n, Box::new(chemo));
    }
    for _ in 0..120 {
        world.step();
    }

    let g = world.to_graph();
    let n = g.nodes.len();
    let edges: Vec<(usize, usize, f32)> = g
        .edges
        .iter()
        .map(|e| (e.pre, e.post, e.weight as f32))
        .collect();
    (n, edges)
}

fn build_runner(n: usize, edges: &[(usize, usize, f32)]) -> BrainRunner {
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    let mut row_ptr = vec![0i32];
    for i in 0..n {
        for (pre, post, w) in edges {
            if *pre == i {
                ci.push(*post as i32);
                vals.push(*w);
            }
        }
        row_ptr.push(ci.len() as i32);
    }
    let nt_types = vec![-1.0f32; n];
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt_types, vec![]);
    let mut runner =
        BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN)).expect("runner");

    let params: [(usize, f32); 17] = [
        (0, DT),
        (1, 1.0),
        (2, 0.10),
        (3, -60.0),
        (4, 0.4),
        (5, -80.0),
        (6, 0.8),
        (7, -30.0),
        (8, 1.0f32 / 100.0),
        (9, 1.0f32 / 120.0),
        (10, -35.0),
        (11, 5.0),
        (12, -75.0),
        (13, 6.0),
        (14, -40.0),
        (15, -55.0),
        (16, 0.02),
    ];
    let mut all = vec![0.0f32; n];
    for (col, val) in params {
        all.fill(val);
        runner.set_haap_param_column(col, &all);
    }
    runner.set_v(&[-40.0, -80.0]);
    runner
}

fn autocorr_period(s: &[f32], dt: f32) -> f32 {
    let step = (s.len() / 2000).max(1);
    let ds: Vec<f32> = s.iter().step_by(step).copied().collect();
    let n = ds.len();
    let mean = ds.iter().sum::<f32>() / n as f32;
    let mut best_lag = 1usize;
    let mut best = f32::NEG_INFINITY;
    for lag in (n / 20)..(n / 2) {
        let mut acc = 0.0;
        for i in 0..(n - lag) {
            acc += (ds[i] - mean) * (ds[i + lag] - mean);
        }
        if acc > best {
            best = acc;
            best_lag = lag;
        }
    }
    best_lag as f32 * step as f32 * dt
}

fn goertzel_phase(s: &[f32], f_hz: f32, dt_ms: f32) -> f32 {
    let w = 2.0 * std::f32::consts::PI * f_hz * dt_ms / 1000.0;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (t, &v) in s.iter().enumerate() {
        let theta = w * t as f32;
        re += v * theta.cos();
        im += v * theta.sin();
    }
    im.atan2(re).to_degrees().rem_euclid(360.0)
}

fn phase_offset(a: &[f32], b: &[f32], period_ms: f32, dt: f32) -> f32 {
    if period_ms <= 0.0 {
        return 0.0;
    }
    let f_hz = 1000.0 / period_ms;
    let pa = goertzel_phase(a, f_hz, dt);
    let pb = goertzel_phase(b, f_hz, dt);
    (pb - pa).rem_euclid(360.0)
}

fn convolve(s: &[f32], win: &[f32]) -> Vec<f32> {
    let n = s.len();
    let hw = win.len() / 2;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0;
        for (k, &w) in win.iter().enumerate() {
            let j = i as isize + k as isize - hw as isize;
            if j >= 0 && j < n as isize {
                acc += s[j as usize] * w;
            }
        }
        out[i] = acc;
    }
    out
}

/// Grow → run HAAP → measure (phase, period). Returns (f32::MAX, 0.0) if the
/// growth did not produce a mutual pair.
fn evaluate(p: GrowthParams) -> (f32, f32) {
    let (n, edges) = grow(p);
    let mutual =
        edges.iter().any(|e| e.0 == 0 && e.1 == 1) && edges.iter().any(|e| e.0 == 1 && e.1 == 0);
    if n < 2 || !mutual {
        return (0.0, 0.0);
    }

    let mut runner = build_runner(n, &edges);
    let ext = vec![DRIVE; n];
    let buf = runner.run(STEPS, &ext);

    let mut t0 = vec![0.0f32; STEPS];
    let mut t1 = vec![0.0f32; STEPS];
    for step in 0..STEPS {
        t0[step] = buf[step * n] as f32;
        t1[step] = buf[step * n + 1] as f32;
    }
    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let s0 = convolve(&t0, &win);
    let s1 = convolve(&t1, &win);
    let period = autocorr_period(&s0, DT);
    let phase = phase_offset(&s0, &s1, period, DT);
    (phase, period)
}

/// Fitness: higher is better. Penalizes phase error (want 180°) and period
/// error (want TARGET_PERIOD).
fn fitness(phase: f32, period: f32) -> f32 {
    if period <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let phase_err = (phase - TARGET_PHASE).abs();
    let period_err = (period - TARGET_PERIOD).abs();
    -(phase_err + period_err)
}

fn mutate(p: &GrowthParams, rng: &mut impl Rng) -> GrowthParams {
    let mut jitter = |v: f64, lo: f64, hi: f64, sigma: f64| -> f64 {
        let v = v + rng.gen_range(-sigma..sigma);
        v.clamp(lo, hi)
    };
    GrowthParams {
        synapse_strength: jitter(p.synapse_strength, 10.0, 200.0, 15.0),
        chemo_speed: jitter(p.chemo_speed, 0.05, 3.0, 0.15),
        diffusion: jitter(p.diffusion, 0.01, 1.0, 0.05),
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    // Seed: a genome that produces a working (but not tuned) half-center.
    let mut best = GrowthParams {
        synapse_strength: 60.0,
        chemo_speed: 1.0,
        diffusion: 0.1,
    };
    let (p0, t0) = evaluate(best);
    let mut best_fit = fitness(p0, t0);
    println!(
        "gen 0   {:?}  phase={p0:.0}° period={t0:.0}ms  fit={best_fit:.1}",
        best
    );

    const GENERATIONS: usize = 150;
    for gen in 1..=GENERATIONS {
        let cand = mutate(&best, &mut rng);
        let (pc, tc) = evaluate(cand);
        let fc = fitness(pc, tc);
        if fc > best_fit {
            best = cand;
            best_fit = fc;
        }
        if gen % 25 == 0 || gen == GENERATIONS {
            let (pb, tb) = evaluate(best);
            println!(
                "gen {gen:<3} {:?}  phase={pb:.0}° period={tb:.0}ms  fit={best_fit:.1}",
                best
            );
        }
    }

    let (pf, tf) = evaluate(best);
    println!("\n=== evolved ===");
    println!("  genome: {:?}", best);
    println!(
        "  phase={pf:.0}° (target {TARGET_PHASE:.0}°)  period={tf:.0}ms (target {TARGET_PERIOD:.0}ms)"
    );
    let phase_ok = (pf - TARGET_PHASE).abs() < 20.0;
    let period_ok = (tf - TARGET_PERIOD).abs() < 25.0;
    println!(
        "  → {}",
        if phase_ok && period_ok {
            "эволюция нашла правила роста под целевой ритм ✓"
        } else {
            "близко, но не дотянула"
        }
    );
}
