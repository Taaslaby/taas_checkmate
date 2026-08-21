//! Grow a tripod CPG (6 GABA neurons) and run it through pillars HAAP.
//!
//! ```text
//! dpp gradient (z) ──► role 0,1,2 (segment, via Differentiate)
//! asymmetric division ──► sub_role 0/1 (stance/swing)
//!                            │
//!   mutual: connect_fates(i,0 ↔ i,1)      (half-center within a segment)
//!   ring:   connect_fates(i,0 → i+1,0)    (stance_i inhibits stance_{i+1})
//!                            │
//!               CSR ──► pillars HAAP ──► 3-phase (tripod)
//! ```
//!
//! ```bash
//! cargo run -p pillars-connectome --example grow_tripod --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::api::neurons::{ChordotonalBuilder, ChordotonalProfile};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use pillars_grow::{
    ArtificialGradient, Bounds, CellType, Differentiate, DivideOnce, Grid, Substance, Vec3, World,
};

const DT: f32 = 0.05; // ms
const TAU_SYN: f32 = 12.0;
const STEPS: usize = 60_000; // 3000 ms
const DRIVE: f32 = 2.0;
const W_INH: f64 = 18.0; // mutual inhibition within a half-center
const W_SEG: f64 = 8.0; // ring: stance_i → stance_{i+1}

const N_LEGS: usize = 3;
const RESET_GAIN: f32 = 0.08;
const RESET_THRESHOLD: f32 = 8.0;
const EMBODIED_STEPS: usize = 20_000; // 1000 ms

fn load_profile() -> ChordotonalProfile {
    ChordotonalProfile {
        tau_sens: 8.0,
        tau_adapt: 300.0,
        b_adapt: 0.08,
        load_gain: 10.0,
        stretch_gain: 1.0,
        baseline: -1.0,
        threshold: 0.0,
        max_rate: 200.0,
    }
}

/// Grow 3 segments × 2 cells (stance+swing), wire mutual + ring by fate.
fn grow_tripod() -> (usize, Vec<(usize, usize, f32)>) {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    world.add_substance(Substance::new(
        "dpp",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Linear {
            axis: 2,
            max: 3.0,
            peak: 40.0,
            zero: -40.0,
        }],
    ));

    // 3 progenitors (one per segment): read dpp → segment role, then divide once.
    for z in [-30.0, 0.0, 30.0] {
        let c = world.add_cell(Vec3::new(0.0, 0.0, z), CellType::Inhibitory, 4.0);
        world.add_cell_module(c, Box::new(Differentiate::new("dpp", 1.0)));
        world.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
    }
    for _ in 0..30 {
        world.step();
        if world.cells.len() >= 6 {
            break;
        }
    }

    // Print the fates: segment (role) × stance/swing (sub_role).
    for c in &world.cells {
        println!(
            "  cell {} z={:+.0} role={} sub_role={} ({} )",
            c.id,
            c.soma.pos.z,
            c.role,
            c.sub_role,
            if c.sub_role == 0 { "stance" } else { "swing" }
        );
    }

    // Mutual inhibition within each half-center: stance ↔ swing.
    for i in 0..3 {
        world.connect_fates(i, 0, i, 1, W_INH);
        world.connect_fates(i, 1, i, 0, W_INH);
    }
    // Ring: stance_i → stance_{i+1}.
    world.connect_fates(0, 0, 1, 0, W_SEG);
    world.connect_fates(1, 0, 2, 0, W_SEG);
    world.connect_fates(2, 0, 0, 0, W_SEG);

    let g = world.to_graph();
    let n = g.nodes.len();
    let edges: Vec<(usize, usize, f32)> = g
        .edges
        .iter()
        .map(|e| (e.pre, e.post, e.weight as f32))
        .collect();
    println!("wired: {n} neurons, {} edges", edges.len());
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
        BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN)).expect("tripod runner");

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

    // Break symmetry: stance (ids 0,1,2) + swing (ids 3,4,5) staggered.
    let mut init = vec![0.0f32; n];
    for i in 0..3 {
        init[i] = -40.0 - 3.0 * i as f32;
        init[3 + i] = -80.0 - 2.0 * i as f32;
    }
    runner.set_v(&init);
    runner
}

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

fn main() {
    let (n, edges) = grow_tripod();
    let mut runner = build_runner(n, &edges);
    let ext = vec![DRIVE; n];
    let buf = runner.run(STEPS, &ext);

    let mut trains: Vec<Vec<f32>> = vec![vec![0.0f32; STEPS]; n];
    for step in 0..STEPS {
        for i in 0..n {
            trains[i][step] = buf[step * n + i] as f32;
        }
    }
    let rates = runner.firing_rates(DT);
    println!(
        "\nstance rates: [{}] Hz",
        rates[0..3]
            .iter()
            .map(|r| format!("{r:.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Tripod phase analysis on the 3 stance neurons (ids 0,1,2).
    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let s: Vec<Vec<f32>> = trains[..3].iter().map(|t| convolve(t, &win)).collect();
    let period = autocorr_period(&s[0], DT);
    let phase01 = phase_offset(&s[0], &s[1], period, DT);
    let phase02 = phase_offset(&s[0], &s[2], period, DT);
    println!("period={period:.0}ms  phase01={phase01:.0}°  phase02={phase02:.0}°");

    let near120 = |deg: f32| (deg - 120.0).abs() < 30.0 || (deg - 240.0).abs() < 30.0;
    let tripod = near120(phase01) && near120(phase02);
    println!(
        "  → {}",
        if tripod {
            "TRIPOD 3-фаза ✓ (ДНК → рост → роль → кольцо → pillars HAAP)"
        } else {
            "не 3-фаза"
        }
    );

    // ── Embodied: drive legs with the grown CPG ──
    println!("\n=== embodied (CPG → мышцы → суставы) ===");
    let mut body = FlyEmbodiedState::new(N_LEGS, DT);
    let mut sensor = ChordotonalBuilder::new(N_LEGS)
        .with_profile(load_profile())
        .with_dt(DT)
        .build();
    let resets = vec![0i32; body.n_muscles];
    let mut mn = vec![0i32; body.n_muscles];
    let mut reset_cur = vec![0.0f32; N_LEGS];

    let mut stance: [Vec<f32>; N_LEGS] = std::array::from_fn(|_| vec![0.0f32; EMBODIED_STEPS]);
    let mut all_stance = 0u64;
    let mut all_swing = 0u64;
    let mut q_min = [f32::INFINITY; N_LEGS];
    let mut q_max = [f32::NEG_INFINITY; N_LEGS];

    for step in 0..EMBODIED_STEPS {
        // 1. CPG step: swing neurons (3+i) get the load-reset current.
        let mut ext = vec![DRIVE; n];
        for i in 0..N_LEGS {
            ext[3 + i] += reset_cur[i];
        }
        let spikes = runner.run(1, &ext);

        // 2. CPG → muscles. leg i: flexor ← swing, extensor ← stance.
        for i in 0..N_LEGS {
            let n_stance = spikes[i]; // stance = neuron i
            let n_swing = spikes[3 + i]; // swing = neuron 3+i
            mn[2 * i] = n_swing;
            mn[2 * i + 1] = n_stance;
            stance[i][step] = n_stance as f32;
        }
        body.step(&mn, &resets);

        // 3. Joint → load → sensor → reset (load ⇒ swing switch).
        let mut loads = [0.0f32; N_LEGS];
        let mut stance_now = [false; N_LEGS];
        for i in 0..N_LEGS {
            let q = body.simulated_joint_qpos[i];
            stance_now[i] = q > 0.0;
            loads[i] = q.clamp(0.0, 1.0);
            q_min[i] = q_min[i].min(q);
            q_max[i] = q_max[i].max(q);
        }
        sensor.step(&loads, &[0.0; N_LEGS], None);
        let rates_now = sensor.firing_rates();
        for i in 0..N_LEGS {
            let r = rates_now[i];
            reset_cur[i] = if r > RESET_THRESHOLD {
                RESET_GAIN * (r - RESET_THRESHOLD)
            } else {
                0.0
            };
        }
        if stance_now.iter().all(|&s| s) {
            all_stance += 1;
        }
        if stance_now.iter().all(|&s| !s) {
            all_swing += 1;
        }
    }

    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let sm: Vec<Vec<f32>> = stance.iter().map(|s| convolve(s, &win)).collect();
    let period = autocorr_period(&sm[0], DT);
    let phase01 = phase_offset(&sm[0], &sm[1], period, DT);
    let phase02 = phase_offset(&sm[0], &sm[2], period, DT);
    let excursion: Vec<String> = (0..N_LEGS)
        .map(|i| format!("{:.2}", q_max[i] - q_min[i]))
        .collect();
    println!(
        "joint excursion: [{}]  all_stance={:.2}  all_swing={:.2}",
        excursion.join(" "),
        all_stance as f32 / EMBODIED_STEPS as f32,
        all_swing as f32 / EMBODIED_STEPS as f32
    );
    println!("stance phases: phase01={phase01:.0}°  phase02={phase02:.0}°  period={period:.0}ms");
    let legs_move = (0..N_LEGS).all(|i| q_max[i] - q_min[i] > 0.1);
    let walking = near120(phase01) && near120(phase02) && legs_move;
    println!(
        "  → {}",
        if walking {
            "WALKING 3-на-3 ✓ (выращенный CPG двигает ноги)"
        } else {
            "не ходит"
        }
    );
}
