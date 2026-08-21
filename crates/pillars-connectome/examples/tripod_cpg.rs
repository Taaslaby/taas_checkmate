//! 🦗 Tripod CPG — 3 coupled half-centers + 3 embodied legs.
//!
//! Three HAAP-synaptic half-centers (one per leg), each a stance/swing pair,
//! coupled by a **ring of inter-segmental inhibition** (`stance_i` inhibits
//! `stance_{i+1}`). Each half-center drives an Atlasov muscle pair → joint,
//! and a chordotonal load sensor closes the loop per leg (load ⇒ swing switch).
//!
//! ```text
//!   segment 0 ──inhibit──► segment 1 ──inhibit──► segment 2 ──inhibit──► segment 0
//!      │                       │                       │
//!      ▼                       ▼                       ▼
//!   leg 0 (muscle+joint+sensor)  leg 1 (...)           leg 2 (...)
//! ```
//!
//! The question: does the ring coupling phase-lock the 3 legs into a tripod
//! rhythm (stance phases ~120° apart), or do they collapse into synchronous
//! hopping? Compare `w_seg = 0` (open, decoupled) vs `w_seg > 0` (coupled).
//!
//! Run:
//! ```text
//! cargo run -p pillars-connectome --example tripod_cpg --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::api::neurons::{ChordotonalBuilder, ChordotonalProfile};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

const N_LEGS: usize = 3;
const N_NEURONS: usize = 2 * N_LEGS; // stance=2i, swing=2i+1
const DT: f32 = 0.05; // ms
const STEPS: usize = 60_000; // 3000 ms
const DRIVE: f32 = 2.0;
const W_INH: f32 = 18.0; // mutual inhibition within a half-center
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

/// Build the 3-half-center CPG with a ring of inter-segmental inhibition.
///
/// Neurons: `2i` = stance(extensor) of leg i, `2i+1` = swing(flexor) of leg i.
/// Edges:
///   - within each half-center: stance ↔ swing mutual inhibition (W_INH)
///   - ring: stance_i → stance_{i+1} inhibition (w_seg)
fn build_cpg(w_seg: f32) -> BrainRunner {
    let mut ci: Vec<i32> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    let mut row_ptr: Vec<i32> = vec![0];

    for i in 0..N_LEGS {
        let stance = 2 * i;
        let swing = 2 * i + 1;
        let next_stance = 2 * ((i + 1) % N_LEGS);

        // stance → swing (mutual)
        ci.push(swing as i32);
        vals.push(W_INH);
        // ring: stance → next stance
        ci.push(next_stance as i32);
        vals.push(w_seg);
        row_ptr.push(ci.len() as i32);

        // swing → stance (mutual)
        ci.push(stance as i32);
        vals.push(W_INH);
        row_ptr.push(ci.len() as i32);
    }

    let nt_types = vec![-1.0f32; N_NEURONS]; // all GABA
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt_types, vec![]);
    let mut cpg =
        BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN)).expect("tripod CPG build");

    // HAAPModelParamIdx columns (same rebound+adaptation profile as the single leg).
    let params = [
        (0, DT),
        (1, 1.0),
        (2, 0.10),
        (3, -60.0),
        (4, 0.4),
        (5, -80.0),
        (6, 0.8),
        (7, -30.0),
        (8, 1.0 / 100.0),
        (9, 1.0 / 120.0),
        (10, -35.0),
        (11, 5.0),
        (12, -75.0),
        (13, 6.0),
        (14, -40.0),
        (15, -55.0),
        (16, 0.02),
    ];
    let mut all = vec![0.0f32; N_NEURONS];
    for (col, val) in params {
        all.fill(val);
        cpg.set_haap_param_column(col, &all);
    }

    // Break symmetry: each segment slightly offset (seeds the ring's phase spread).
    let mut init = vec![0.0f32; N_NEURONS];
    for i in 0..N_LEGS {
        init[2 * i] = -40.0 + 3.0 * i as f32; // stance
        init[2 * i + 1] = -80.0 + 2.0 * i as f32; // swing
    }
    cpg.set_v(&init);
    cpg
}

struct TripodResult {
    stance_rates: [f32; N_LEGS],
    all_stance_frac: f32, // all 3 legs in stance simultaneously (sync collapse)
    all_swing_frac: f32,  // all 3 legs in swing simultaneously
    phase01_deg: f32,     // phase offset leg0 → leg1
    phase02_deg: f32,     // phase offset leg0 → leg2
    period_ms: f32,
}

fn run_tripod(w_seg: f32) -> TripodResult {
    let mut cpg = build_cpg(w_seg);
    let mut body = FlyEmbodiedState::new(N_LEGS, DT); // 3 joints, 6 muscles
    let mut sensor = ChordotonalBuilder::new(N_LEGS)
        .with_profile(load_profile())
        .with_dt(DT)
        .build();

    let resets = vec![0i32; body.n_muscles];
    let mut mn = vec![0i32; body.n_muscles];
    let mut reset_cur = vec![0.0f32; N_LEGS];

    // stance spike trains (smoothed) for phase analysis
    let mut stance: [Vec<f32>; N_LEGS] = std::array::from_fn(|_| vec![0.0f32; STEPS]);

    let mut all_stance = 0u64;
    let mut all_swing = 0u64;
    let mut rates = [0u64; N_LEGS];

    for step in 0..STEPS {
        // 1. CPG step. ext[swing_i] += reset current (load ⇒ swing).
        let mut ext = vec![DRIVE; N_NEURONS];
        for i in 0..N_LEGS {
            ext[2 * i + 1] += reset_cur[i];
        }
        let spikes = cpg.run(1, &ext);

        // 2. CPG → muscles. leg i: flexor=2i, extensor=2i+1.
        for i in 0..N_LEGS {
            let n_stance = spikes[2 * i];
            let n_swing = spikes[2 * i + 1];
            rates[i] += n_stance as u64;
            mn[2 * i] = n_swing; // flexor ← swing
            mn[2 * i + 1] = n_stance; // extensor ← stance
            stance[i][step] = n_stance as f32;
        }
        body.step(&mn, &resets);

        // 3. Joint angle → load → sensor → reset.
        let mut loads = [0.0f32; N_LEGS];
        let mut stance_now = [false; N_LEGS];
        for i in 0..N_LEGS {
            let q = body.simulated_joint_qpos[i];
            stance_now[i] = q > 0.0;
            loads[i] = q.clamp(0.0, 1.0);
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

    // Smooth stance trains for phase analysis.
    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let smooth: Vec<Vec<f32>> = stance.iter().map(|s| convolve(s, &win)).collect();

    let period_ms = autocorr_period(&smooth[0], DT);
    let phase01_deg = phase_offset(&smooth[0], &smooth[1], period_ms, DT);
    let phase02_deg = phase_offset(&smooth[0], &smooth[2], period_ms, DT);

    TripodResult {
        stance_rates: [
            rates[0] as f32 / (STEPS as f32 * DT) * 1000.0,
            rates[1] as f32 / (STEPS as f32 * DT) * 1000.0,
            rates[2] as f32 / (STEPS as f32 * DT) * 1000.0,
        ],
        all_stance_frac: all_stance as f32 / STEPS as f32,
        all_swing_frac: all_swing as f32 / STEPS as f32,
        phase01_deg,
        phase02_deg,
        period_ms,
    }
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

/// Dominant period via autocorrelation of the smoothed stance signal.
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

/// Phase (degrees) of a smoothed stance signal at frequency `f_hz` via Goertzel.
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

/// Phase offset (degrees) of `b` relative to `a` at the fundamental frequency.
fn phase_offset(a: &[f32], b: &[f32], period_ms: f32, dt: f32) -> f32 {
    if period_ms <= 0.0 {
        return 0.0;
    }
    let f_hz = 1000.0 / period_ms;
    let pa = goertzel_phase(a, f_hz, dt);
    let pb = goertzel_phase(b, f_hz, dt);
    (pb - pa).rem_euclid(360.0)
}

fn report(name: &str, r: &TripodResult) {
    // Tripod verdict: leg phases evenly spaced ~120° apart (0°, 120°, 240°).
    let near120 = |deg: f32| (deg - 120.0).abs() < 25.0 || (deg - 240.0).abs() < 25.0;
    let verdict = if near120(r.phase01_deg) && near120(r.phase02_deg) {
        "TRIPOD 120°"
    } else if r.all_swing_frac > 0.2 {
        "sync-hop"
    } else {
        "spread"
    };
    println!(
        "  {name:<12} stance=[{:>5.1} {:>5.1} {:>5.1}]Hz  phase01={:>5.0}°  phase02={:>5.0}°  \
         all_stance={:.2}  all_swing={:.2}  period={:.0}ms  → {verdict}",
        r.stance_rates[0],
        r.stance_rates[1],
        r.stance_rates[2],
        r.phase01_deg,
        r.phase02_deg,
        r.all_stance_frac,
        r.all_swing_frac,
        r.period_ms,
    );
}

fn main() {
    println!("tripod_cpg: 3 half-centers + ring inhibition + 3 legs + load sensors");
    for w_seg in [0.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0] {
        let r = run_tripod(w_seg);
        println!("w_seg={w_seg:.0}:");
        report("", &r);
    }
}
