//! 🦗 Hexapod gait — 3 half-centers × 2 legs (anti-phase pairs) = 6 legs.
//!
//! Each half-center (stance `S_i` ↔ swing `W_i`, mutual GABA inhibition) drives
//! a left+right leg pair in anti-phase. The three half-centers are coupled by a
//! ring of inter-segmental inhibition (`S_i → S_{i+1}`), as in `tripod_cpg`.
//!
//! Leg mapping (classic alternating-tripod assignment, middle pair flipped):
//!
//! ```text
//!   L0 = S0    R0 = W0
//!   L1 = W1    R1 = S1      ← middle pair flipped
//!   L2 = S2    R2 = W2
//! ```
//!
//! Because each pair is anti-phase, exactly one leg per pair is in stance at
//! any time → the hexapod must show **3 stance / 3 swing**. This example
//! measures the stance-count distribution and the 6 leg phases to verify it.
//!
//! Run:
//! ```text
//! cargo run -p pillars-connectome --example hexapod_gait --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::api::neurons::{ChordotonalBuilder, ChordotonalProfile};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

const N_SEG: usize = 3; // half-centers
const N_LEGS: usize = 2 * N_SEG; // 6 legs
const N_NEURONS: usize = 2 * N_SEG; // S_i=2i, W_i=2i+1
const DT: f32 = 0.05;
const STEPS: usize = 60_000; // 3000 ms
const DRIVE: f32 = 2.0;
const W_INH: f32 = 18.0;
const TAU_SYN: f32 = 12.0;
const RESET_GAIN: f32 = 0.08;
const RESET_THRESHOLD: f32 = 8.0;

// Leg j → CPG neuron driving its stance (classic tripod: middle pair flipped).
// leg 0=L0(S0) 1=R0(W0) 2=L1(W1) 3=R1(S1) 4=L2(S2) 5=R2(W2)
const LEG_STANCE_NEURON: [usize; N_LEGS] = [0, 1, 3, 2, 4, 5];
const LEG_SWING_NEURON: [usize; N_LEGS] = [1, 0, 2, 3, 5, 4];

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

/// 3 half-centers with inter-segmental coupling.
///
/// `sync=false` — ring inhibition `S_i → S_{i+1}` (metachronal wave, 120°).
/// `sync=true`  — cross inhibition `S_i → W_j` (j≠i): the disynaptic path
///                `S_i → W_j → S_j` makes the half-centers phase-LOCK in phase,
///                yielding the alternating-tripod (tripod A moves as a unit).
fn build_cpg(w_seg: f32, sync: bool) -> BrainRunner {
    let mut ci: Vec<i32> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    let mut row_ptr: Vec<i32> = vec![0];

    for i in 0..N_SEG {
        let stance = 2 * i;
        let swing = 2 * i + 1;
        // S_i → W_i (mutual)
        ci.push(swing as i32);
        vals.push(W_INH);
        if sync {
            // S_i → W_j for all j≠i (sync via disynaptic release)
            for j in 0..N_SEG {
                if j != i {
                    ci.push((2 * j + 1) as i32);
                    vals.push(w_seg);
                }
            }
        } else {
            // S_i → S_{i+1} (ring)
            ci.push(2 * ((i + 1) % N_SEG) as i32);
            vals.push(w_seg);
        }
        row_ptr.push(ci.len() as i32);
        // W_i → S_i (mutual)
        ci.push(stance as i32);
        vals.push(W_INH);
        row_ptr.push(ci.len() as i32);
    }

    let nt_types = vec![-1.0f32; N_NEURONS];
    let conn = ConnectomeData::new(row_ptr, ci, vals, nt_types, vec![]);
    let mut cpg = BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN))
        .expect("hexapod CPG build");

    let params = [
        (0, DT), (1, 1.0), (2, 0.10), (3, -60.0), (4, 0.4), (5, -80.0),
        (6, 0.8), (7, -30.0), (8, 1.0 / 100.0), (9, 1.0 / 120.0),
        (10, -35.0), (11, 5.0), (12, -75.0), (13, 6.0), (14, -40.0),
        (15, -55.0), (16, 0.02),
    ];
    let mut all = vec![0.0f32; N_NEURONS];
    for (col, val) in params {
        all.fill(val);
        cpg.set_haap_param_column(col, &all);
    }

    let mut init = vec![0.0f32; N_NEURONS];
    for i in 0..N_SEG {
        init[2 * i] = -40.0 + 3.0 * i as f32;
        init[2 * i + 1] = -80.0 + 2.0 * i as f32;
    }
    cpg.set_v(&init);
    cpg
}

struct HexapodResult {
    stance_count: [f32; N_LEGS + 1], // fraction of steps with 0..6 legs in stance
    mean_stance: f32,
    phases: [f32; N_LEGS], // stance phase (deg) of each leg
    period_ms: f32,
}

fn run_hexapod(w_seg: f32, sync: bool) -> HexapodResult {
    let mut cpg = build_cpg(w_seg, sync);
    let mut body = FlyEmbodiedState::new(N_LEGS, DT); // 6 joints, 12 muscles
    let mut sensor = ChordotonalBuilder::new(N_LEGS)
        .with_profile(load_profile())
        .with_dt(DT)
        .build();

    let resets = vec![0i32; body.n_muscles];
    let mut mn = vec![0i32; body.n_muscles];
    let mut reset_cur = vec![0.0f32; N_LEGS];

    let mut stance: [Vec<f32>; N_LEGS] = std::array::from_fn(|_| vec![0.0f32; STEPS]);
    let mut count_hist = [0u64; N_LEGS + 1];

    for step in 0..STEPS {
        // 1. CPG step.
        let mut ext = vec![DRIVE; N_NEURONS];
        for j in 0..N_LEGS {
            ext[LEG_SWING_NEURON[j]] += reset_cur[j];
        }
        let spikes = cpg.run(1, &ext);

        // 2. CPG → muscles. leg j: flexor=2j (swing), extensor=2j+1 (stance).
        for j in 0..N_LEGS {
            let n_stance = spikes[LEG_STANCE_NEURON[j]];
            let n_swing = spikes[LEG_SWING_NEURON[j]];
            mn[2 * j] = n_swing; // flexor ← swing
            mn[2 * j + 1] = n_stance; // extensor ← stance
            stance[j][step] = n_stance as f32;
        }
        body.step(&mn, &resets);

        // 3. Joint → load → sensor → reset.
        let mut loads = [0.0f32; N_LEGS];
        let mut n_stance_now = 0usize;
        for j in 0..N_LEGS {
            let q = body.simulated_joint_qpos[j];
            if q > 0.0 {
                n_stance_now += 1;
            }
            loads[j] = q.clamp(0.0, 1.0);
        }
        count_hist[n_stance_now] += 1;

        sensor.step(&loads, &[0.0; N_LEGS], None);
        let rates = sensor.firing_rates();
        for j in 0..N_LEGS {
            let r = rates[j];
            reset_cur[j] = if r > RESET_THRESHOLD {
                RESET_GAIN * (r - RESET_THRESHOLD)
            } else {
                0.0
            };
        }
    }

    // Smooth stance trains + phase.
    let win: Vec<f32> = (0..400).map(|_| 1.0 / 400.0).collect();
    let smooth: Vec<Vec<f32>> = stance.iter().map(|s| convolve(s, &win)).collect();
    let period_ms = autocorr_period(&smooth[0], DT);
    let f_hz = if period_ms > 0.0 { 1000.0 / period_ms } else { 0.0 };
    let phases: [f32; N_LEGS] = std::array::from_fn(|j| {
        if f_hz > 0.0 {
            goertzel_phase(&smooth[j], f_hz, DT)
        } else {
            0.0
        }
    });
    // Normalize phases relative to leg 0.
    let phases: [f32; N_LEGS] =
        std::array::from_fn(|j| (phases[j] - phases[0]).rem_euclid(360.0));

    let mut mean_stance = 0.0;
    for (k, &c) in count_hist.iter().enumerate() {
        mean_stance += k as f32 * c as f32;
    }
    mean_stance /= STEPS as f32;

    let mut stance_count = [0.0f32; N_LEGS + 1];
    for (k, &c) in count_hist.iter().enumerate() {
        stance_count[k] = c as f32 / STEPS as f32;
    }

    HexapodResult {
        stance_count,
        mean_stance,
        phases,
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

fn report(name: &str, r: &HexapodResult) {
    // 3-stance/3-swing quality: fraction of steps with exactly 3 stance.
    let tripod_quality = r.stance_count[3];

    // Alternating-tripod check: tripod A = {L0, R1, L2} (legs 0,3,4) in phase,
    // tripod B = {R0, L1, R2} (legs 1,2,5) in phase, the two groups ~180° apart.
    let spread = |idxs: &[usize]| {
        let ph: Vec<f32> = idxs.iter().map(|&j| r.phases[j]).collect();
        let (mut mn, mut mx) = (ph[0], ph[0]);
        for &p in &ph {
            mn = mn.min(p);
            mx = mx.max(p);
        }
        // circular-aware spread near 0° (wrap 359→-1)
        let ph_wrapped: Vec<f32> = ph.iter().map(|&p| if p > 180.0 { p - 360.0 } else { p }).collect();
        let (mut mn2, mut mx2) = (ph_wrapped[0], ph_wrapped[0]);
        for &p in &ph_wrapped {
            mn2 = mn2.min(p);
            mx2 = mx2.max(p);
        }
        (mx - mn).min(mx2 - mn2)
    };
    let spread_a = spread(&[0, 3, 4]);
    let spread_b = spread(&[1, 2, 5]);
    let verdict = if spread_a < 30.0 && spread_b < 30.0 {
        "ALTERNATING TRIPOD"
    } else {
        "wave"
    };

    print!(
        "  {name:<10} mean_stance={:.2}  P(k=3)={:.2}  period={:.0}ms  tripod_A_spread={:.0}°  tripod_B_spread={:.0}°  phases=[",
        r.mean_stance, tripod_quality, r.period_ms, spread_a, spread_b,
    );
    for j in 0..N_LEGS {
        print!("{:>5.0}°", r.phases[j]);
    }
    println!("]  → {verdict}");
}

fn main() {
    println!("hexapod_gait: 3 half-centers × anti-phase leg pairs (6 legs)");
    println!("leg order: L0 R0 L1 R1 L2 R2");
    println!("\n--- decoupled ---");
    report("w=0", &run_hexapod(0.0, false));
    println!("\n--- ring S_i→S{{i+1}} (metachronal wave) ---");
    report("w=8", &run_hexapod(8.0, false));
    println!("\n--- cross S_i→W_j (alternating tripod) ---");
    for w in [4.0f32, 8.0, 12.0] {
        report(&format!("w={w:.0}"), &run_hexapod(w, true));
    }
}
