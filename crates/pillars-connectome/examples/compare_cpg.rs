//! Side-by-side: grown CPG (from DNA) vs hand-written `tripod_cpg`.
//!
//! Both are structurally the same 3-half-center ring:
//!   - mutual inhibition W_INH within each segment (stance ↔ swing)
//!   - ring coupling W_SEG between stances (stance_i → stance_{i+1})
//!
//! The only difference is *how* they were built:
//!
//!   - grown: DNA → dpp gradient → segments → asymmetric division → connect_fates
//!   - hand:  explicit CSR edges in tripod_cpg.rs
//!
//! To isolate the structural comparison, both use identical HAAP params and
//! identical symmetry-breaking initial conditions. If the grown pipeline is
//! faithful, the two rows must match.
//!
//! ```text
//! cargo run -p pillars-connectome --example compare_cpg --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::api::neurons::{ChordotonalBuilder, ChordotonalProfile};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, Differentiate, DivideOnce, Grid, Substance, Vec3, World,
};

const N_LEGS: usize = 3;
const DT: f32 = 0.05; // ms
const STEPS: usize = 60_000; // 3000 ms closed loop
const DRIVE: f32 = 2.0;
const W_INH: f64 = 18.0; // mutual inhibition within a half-center
const W_SEG: f64 = 8.0; // ring: stance_i → stance_{i+1}
const TAU_SYN: f32 = 12.0;
const RESET_GAIN: f32 = 0.08;
const RESET_THRESHOLD: f32 = 8.0;

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
    (n, edges)
}

fn set_haap_params(runner: &mut BrainRunner, n: usize) {
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
}

fn build_grown_runner(n: usize, edges: &[(usize, usize, f32)]) -> BrainRunner {
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
        BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN)).expect("grown runner");
    set_haap_params(&mut runner, n);
    // stance = ids 0,1,2; swing = ids 3,4,5 (grown node order).
    let mut init = vec![0.0f32; n];
    for i in 0..3 {
        init[i] = -40.0 + 3.0 * i as f32;
        init[3 + i] = -80.0 + 2.0 * i as f32;
    }
    runner.set_v(&init);
    runner
}

/// Hand-written CPG (identical edges, interleaved ids: stance=2i, swing=2i+1).
fn build_hand_cpg(w_seg: f32) -> BrainRunner {
    let mut ci: Vec<i32> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    let mut row_ptr: Vec<i32> = vec![0];

    for i in 0..N_LEGS {
        let stance = 2 * i;
        let swing = 2 * i + 1;
        let next_stance = 2 * ((i + 1) % N_LEGS);

        ci.push(swing as i32);
        vals.push(W_INH as f32);
        ci.push(next_stance as i32);
        vals.push(w_seg);
        row_ptr.push(ci.len() as i32);

        ci.push(stance as i32);
        vals.push(W_INH as f32);
        row_ptr.push(ci.len() as i32);
    }

    let nt_types = vec![-1.0f32; 2 * N_LEGS];
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt_types, vec![]);
    let mut cpg =
        BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN)).expect("hand CPG build");
    set_haap_params(&mut cpg, 2 * N_LEGS);

    let mut init = vec![0.0f32; 2 * N_LEGS];
    for i in 0..N_LEGS {
        init[2 * i] = -40.0 + 3.0 * i as f32;
        init[2 * i + 1] = -80.0 + 2.0 * i as f32;
    }
    cpg.set_v(&init);
    cpg
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

struct CpgResult {
    name: &'static str,
    stance_rates: [f32; N_LEGS],
    period_ms: f32,
    phase01: f32,
    phase02: f32,
    all_stance_frac: f32,
    all_swing_frac: f32,
    joint_excursion: [f32; N_LEGS],
}

/// Closed-loop run: CPG → muscles/joint → load sensor → reset. Identical for
/// both CPGs; only the neuron id layout differs (`stance_ids` / `swing_ids`).
fn run_embodied(
    name: &'static str,
    runner: &mut BrainRunner,
    stance_ids: [usize; N_LEGS],
    swing_ids: [usize; N_LEGS],
) -> CpgResult {
    let n_neurons = stance_ids.len() + swing_ids.len();
    let mut body = FlyEmbodiedState::new(N_LEGS, DT);
    let mut sensor = ChordotonalBuilder::new(N_LEGS)
        .with_profile(load_profile())
        .with_dt(DT)
        .build();

    let resets = vec![0i32; body.n_muscles];
    let mut mn = vec![0i32; body.n_muscles];
    let mut reset_cur = vec![0.0f32; N_LEGS];

    let mut stance: [Vec<f32>; N_LEGS] = std::array::from_fn(|_| vec![0.0f32; STEPS]);
    let mut all_stance = 0u64;
    let mut all_swing = 0u64;
    let mut rates = [0u64; N_LEGS];
    let mut q_min = [f32::INFINITY; N_LEGS];
    let mut q_max = [f32::NEG_INFINITY; N_LEGS];

    for step in 0..STEPS {
        let mut ext = vec![DRIVE; n_neurons];
        for i in 0..N_LEGS {
            ext[swing_ids[i]] += reset_cur[i];
        }
        let spikes = runner.run(1, &ext);

        for i in 0..N_LEGS {
            let n_stance = spikes[stance_ids[i]];
            let n_swing = spikes[swing_ids[i]];
            rates[i] += n_stance as u64;
            mn[2 * i] = n_swing; // flexor ← swing
            mn[2 * i + 1] = n_stance; // extensor ← stance
            stance[i][step] = n_stance as f32;
        }
        body.step(&mn, &resets);

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
    let smooth: Vec<Vec<f32>> = stance.iter().map(|s| convolve(s, &win)).collect();
    let period_ms = autocorr_period(&smooth[0], DT);
    let phase01 = phase_offset(&smooth[0], &smooth[1], period_ms, DT);
    let phase02 = phase_offset(&smooth[0], &smooth[2], period_ms, DT);

    CpgResult {
        name,
        stance_rates: [
            rates[0] as f32 / (STEPS as f32 * DT) * 1000.0,
            rates[1] as f32 / (STEPS as f32 * DT) * 1000.0,
            rates[2] as f32 / (STEPS as f32 * DT) * 1000.0,
        ],
        period_ms,
        phase01,
        phase02,
        all_stance_frac: all_stance as f32 / STEPS as f32,
        all_swing_frac: all_swing as f32 / STEPS as f32,
        joint_excursion: [
            q_max[0] - q_min[0],
            q_max[1] - q_min[1],
            q_max[2] - q_min[2],
        ],
    }
}

fn print_row(r: &CpgResult) {
    let near120 = |deg: f32| (deg - 120.0).abs() < 30.0 || (deg - 240.0).abs() < 30.0;
    let tripod = near120(r.phase01) && near120(r.phase02);
    let legs_move = r.joint_excursion.iter().all(|&e| e > 0.1);
    let verdict = if tripod && legs_move {
        "WALKING 3-на-3"
    } else if tripod {
        "трипод-фаза"
    } else {
        "нет"
    };
    println!(
        "{:<11} | {:>5.1} {:>5.1} {:>5.1} Hz | {:>4.0}ms | {:>4.0}° {:>4.0}° | {:>5.2} {:>5.2} | {:>5.2} {:>5.2} {:>5.2} | {}",
        r.name,
        r.stance_rates[0],
        r.stance_rates[1],
        r.stance_rates[2],
        r.period_ms,
        r.phase01,
        r.phase02,
        r.all_stance_frac,
        r.all_swing_frac,
        r.joint_excursion[0],
        r.joint_excursion[1],
        r.joint_excursion[2],
        verdict,
    );
}

fn main() {
    println!(
        "compare_cpg: выращенный (ДНК) vs ручной (tripod_cpg), W_SEG=8, закрытая петля 3000 мс\n"
    );

    let (n, edges) = grow_tripod();
    let mut grown = build_grown_runner(n, &edges);
    let mut hand = build_hand_cpg(W_SEG as f32);

    println!(
        "CPG         | stance rates       | period | phase01 phase02 | all_st all_sw | joint excursion   | verdict"
    );
    println!("{}", "-".repeat(92));

    let r_grown = run_embodied("выращенный", &mut grown, [0, 1, 2], [3, 4, 5]);
    print_row(&r_grown);
    let r_hand = run_embodied("ручной", &mut hand, [0, 2, 4], [1, 3, 5]);
    print_row(&r_hand);
}
