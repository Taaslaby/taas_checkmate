//! 🦵 Closed-loop leg — half-center CPG ↔ Atlasov muscles ↔ chordotonal load sensor.
//!
//! This is the kernel of the insect tripod-gait circuit, closed end-to-end:
//!
//! ```text
//!   CPG (2 HAAP-synaptic GABA neurons, anti-phase)
//!     n0 = extensor (stance)   n1 = flexor (swing)
//!        │                            │
//!        ▼                            ▼
//!   Atlasov agonist-antagonist muscles → joint torque → joint angle q
//!        │
//!        ▼
//!   load = max(0, q)          (q > 0 = leg extended = stance = loaded)
//!        │
//!        ▼
//!   Chordotonal sensor → firing rate
//!        │
//!        ▼
//!   phase-reset: load ⇒ excite the swing neuron n1 ⇒ switch to swing
//!        └──────────────► back into the CPG
//! ```
//!
//! The demo runs the SAME circuit twice — open-loop (sensor feedback off) and
//! closed-loop (feedback on) — and reports how the sensory phase-reset changes
//! the gait (entrainment).
//!
//! Run:
//! ```text
//! cargo run -p pillars-connectome --example closed_loop_leg --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::api::neurons::{ChordotonalBuilder, ChordotonalProfile};
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

const DT: f32 = 0.05; // ms
const STEPS: usize = 40_000; // 2000 ms
const DRIVE: f32 = 2.0;
const W_INH: f32 = 18.0;
const TAU_SYN: f32 = 12.0;
const RESET_GAIN: f32 = 0.08; // (rate - threshold) → current into swing neuron
const RESET_THRESHOLD: f32 = 8.0; // sensor rate (Hz) above which load triggers reset

/// Load-sensing chordotonal profile (campaniform sensilla): strong, weakly
/// adapting tonic response so sustained stance load drives a clear rate.
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

fn build_cpg() -> BrainRunner {
    let row_ptr = vec![0, 1, 2];
    let col_indices = vec![1, 0]; // 0→1, 1→0
    let values = vec![W_INH, W_INH];
    let nt_types = vec![-1.0, -1.0]; // both pre-synaptic GABA
    let conn = ConnectomeData::new(row_ptr, col_indices, values, nt_types, vec![]);
    let mut cpg = BrainRunner::new(conn, BrainConfig::haap_synaptic(DT, TAU_SYN))
        .expect("half-center CPG build");

    // HAAPModelParamIdx columns: 0=DT 1=Inv_Cm 2=G_L 3=E_L 4=G_M 5=E_K 6=G_H 7=E_H
    // 8=Inv_Tau_U 9=Inv_Tau_H 10=V_Half_U 11=K_U 12=V_Half_H 13=K_H 14=V_Thresh
    // 15=V_Reset 16=Delta_U
    let params = [
        (0, DT), (1, 1.0), (2, 0.10), (3, -60.0), (4, 0.4), (5, -80.0),
        (6, 0.8), (7, -30.0), (8, 1.0 / 100.0), (9, 1.0 / 120.0),
        (10, -35.0), (11, 5.0), (12, -75.0), (13, 6.0), (14, -40.0),
        (15, -55.0), (16, 0.02),
    ];
    for (col, val) in params {
        cpg.set_haap_param_column(col, &[val, val]);
    }
    cpg.set_v(&[-40.0, -80.0]); // break symmetry → anti-phase
    cpg
}

struct Gait {
    n0: u64,
    n1: u64,
    q_min: f32,
    q_max: f32,
    stance_frac: f32,
    sensor_frac: f32,
    reset_events: u64,
    crossings: u64, // stance↔swing transitions
}

/// Run the leg loop. `feedback` toggles the sensory phase-reset.
fn run_leg(feedback: bool) -> Gait {
    let mut cpg = build_cpg();
    let mut body = FlyEmbodiedState::new(1, DT);
    let mut sensor = ChordotonalBuilder::new(1)
        .with_profile(load_profile())
        .with_dt(DT)
        .build();

    let resets = vec![0i32; body.n_muscles];
    let mut mn = vec![0i32; body.n_muscles];

    let mut g = Gait {
        n0: 0, n1: 0, q_min: f32::INFINITY, q_max: f32::NEG_INFINITY,
        stance_frac: 0.0, sensor_frac: 0.0, reset_events: 0, crossings: 0,
    };

    let mut reset_current = 0.0f32;
    let mut was_stance = false;

    for _ in 0..STEPS {
        // 1. CPG step (sensor phase-reset into swing neuron n1).
        let ext = [DRIVE, DRIVE + reset_current];
        let spikes = cpg.run(1, &ext);
        let n0 = spikes[0];
        let n1 = spikes[1];
        g.n0 += n0 as u64;
        g.n1 += n1 as u64;

        // 2. CPG → muscles. muscle 0 = flexor (n1), muscle 1 = extensor (n0).
        mn[0] = n1;
        mn[1] = n0;
        body.step(&mn, &resets);

        // 3. Joint angle → load.
        let q = body.simulated_joint_qpos[0];
        g.q_min = g.q_min.min(q);
        g.q_max = g.q_max.max(q);
        let stance = q > 0.0;
        if stance {
            g.stance_frac += 1.0;
        }
        if stance != was_stance {
            g.crossings += 1;
            was_stance = stance;
        }
        let load = q.clamp(0.0, 1.0);

        // 4. Load → sensor rate.
        sensor.step(&[load], &[0.0], None);
        let rate = sensor.firing_rates()[0];

        // 5. Rate → phase-reset current (load ⇒ lift off).
        if feedback {
            reset_current = if rate > RESET_THRESHOLD {
                g.reset_events += 1;
                g.sensor_frac += 1.0;
                RESET_GAIN * (rate - RESET_THRESHOLD)
            } else {
                0.0
            };
        } else {
            reset_current = 0.0;
            if rate > RESET_THRESHOLD {
                g.sensor_frac += 1.0;
            }
        }
    }

    g.stance_frac /= STEPS as f32;
    g.sensor_frac /= STEPS as f32;
    g
}

fn report(name: &str, g: &Gait) {
    let total_ms = STEPS as f32 * DT;
    println!(
        "  {name:<12} n0={:>5.1}Hz  n1={:>5.1}Hz  q=[{:>6.2},{:>6.2}]  \
         stance={:>4.2}  sensor={:>4.2}  resets={:>6}  crossings={:>4}",
        g.n0 as f32 / total_ms * 1000.0,
        g.n1 as f32 / total_ms * 1000.0,
        g.q_min,
        g.q_max,
        g.stance_frac,
        g.sensor_frac,
        g.reset_events,
        g.crossings,
    );
}

fn main() {
    println!("closed_loop_leg: CPG(2 GABA) → Atlasov joint → chordotonal → phase-reset");
    let open = run_leg(false);
    let closed = run_leg(true);
    report("open-loop", &open);
    report("closed-loop", &closed);

    let period_ms = STEPS as f32 * DT / closed.crossings.max(1) as f32 * 2.0;
    println!(
        "\n  gait period ≈ {period_ms:.0} ms  |  closed-loop crossings {} vs open {}",
        closed.crossings, open.crossings
    );
    println!(
        "  sensory reset injected {}× (closed-loop) — load ⇒ swing switch",
        closed.reset_events
    );
}
