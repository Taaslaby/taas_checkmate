//! Grow a half-center in `pillars-grow`, then run it through the **real**
//! pillars engine (HAAP with rebound, the same model as `tripod_cpg`).
//!
//! ```text
//! pillars-grow: 1 GABA progenitor ──divide──► 2 cells ──axons──► mutual inhibition
//!                                                              │  (weights from geometry)
//!                                                              ▼
//! CSR (ConnectomeData) ──► BrainRunner (HAAP synaptic) ──► anti-phase
//! ```
//!
//! ```bash
//! cargo run -p pillars-connectome --example grow_halfcenter --release
//! ```

use pillars_grow::{
    Bounds, CellType, ChemoAttraction, DivideOnce, Grid, Secretor, Substance, Vec3, World,
};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

const DT: f32 = 0.05; // ms
const TAU_SYN: f32 = 12.0;
const STEPS: usize = 60_000; // 3000 ms
const DRIVE: f32 = 2.0;

/// Grow the 2-GABA mutual-inhibition motif; return (n, edges) where each edge
/// is (pre, post, weight). Sign is set by the cell type (GABA ⇒ inhibitory).
fn grow_halfcenter() -> (usize, Vec<(usize, usize, f32)>) {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    // HAAP mutual inhibition needs ~18 mV weights (cf. tripod_cpg W_INH).
    world.synapse_strength = 60.0;

    world.add_substance(Substance::new("A", 0.1, 0.001, &world.grid, vec![]));
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

/// Build a `BrainRunner` (HAAP synaptic) from grown edges.
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
    let nt_types = vec![-1.0f32; n]; // all GABA ⇒ inhibitory
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt_types, vec![]);
    let mut runner = BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN))
        .expect("half-center runner");

    // Same rebound + adaptation profile as tripod_cpg's single half-center.
    let params: [(usize, f32); 17] = [
        (0, DT), (1, 1.0), (2, 0.10), (3, -60.0), (4, 0.4), (5, -80.0),
        (6, 0.8), (7, -30.0), (8, 1.0f32 / 100.0), (9, 1.0f32 / 120.0),
        (10, -35.0), (11, 5.0), (12, -75.0), (13, 6.0), (14, -40.0),
        (15, -55.0), (16, 0.02),
    ];
    let mut all = vec![0.0f32; n];
    for (col, val) in params {
        all.fill(val);
        runner.set_haap_param_column(col, &all);
    }

    // Break the in-phase symmetry.
    runner.set_v(&[-40.0, -80.0]);
    runner
}

/// Dominant period (ms) via autocorrelation of a smoothed spike train.
fn autocorr_period(s: &[f32], dt: f32) -> f32 {
    let step = (s.len() / 4000).max(1);
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

/// Phase (deg) of a spike train at frequency `f` via Goertzel.
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

/// Phase offset (deg) of `b` relative to `a` at the shared firing frequency.
fn phase_offset(a: &[f32], b: &[f32], period_ms: f32, dt: f32) -> f32 {
    if period_ms <= 0.0 {
        return 0.0;
    }
    let f_hz = 1000.0 / period_ms;
    let pa = goertzel_phase(a, f_hz, dt);
    let pb = goertzel_phase(b, f_hz, dt);
    (pb - pa).rem_euclid(360.0)
}

/// Box-window smoothing (same as tripod_cpg phase analysis).
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

fn main() {
    let (n, edges) = grow_halfcenter();
    println!("grown: {n} neurons, {} edges", edges.len());
    for (pre, post, w) in &edges {
        println!("  {pre} -> {post}  w={w:.3} (inhibitory)");
    }

    let mut runner = build_runner(n, &edges);
    let ext = vec![DRIVE; n];
    let buf = runner.run(STEPS, &ext); // flat [STEPS × n]

    // Extract binary spike trains per neuron.
    let mut t0 = vec![0.0f32; STEPS];
    let mut t1 = vec![0.0f32; STEPS];
    let (mut c0, mut c1) = (0usize, 0usize);
    for step in 0..STEPS {
        let s0 = buf[step * n];
        let s1 = buf[step * n + 1];
        t0[step] = s0 as f32;
        t1[step] = s1 as f32;
        c0 += (s0 != 0) as usize;
        c1 += (s1 != 0) as usize;
    }

    let rates = runner.firing_rates(DT);
    println!(
        "HAAP: spikes0={c0} spikes1={c1} rates=[{:.1} {:.1}]Hz",
        rates[0], rates[1]
    );

    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let s0 = convolve(&t0, &win);
    let s1 = convolve(&t1, &win);
    let period = autocorr_period(&s0, DT);
    let phase = phase_offset(&s0, &s1, period, DT);
    println!("period={period:.0}ms  phase={phase:.0}°");

    let anti = (phase - 180.0).abs() < 35.0;
    println!(
        "  → {}",
        if anti {
            "ANTI-PHASE ✓ (рост → pillars HAAP)"
        } else {
            "not anti-phase"
        }
    );
}
