//! Integration test: LIF pop A → B with axonal delays through `Network::step`.
//!
//! A spike is forced via `fill_state(v = 100)` (v ≫ v_thresh), the network is
//! stepped, and we verify that the post-synaptic current arrives in population
//! B **exactly `delay` steps** after the pre-synaptic spike — in BOTH routing
//! modes:
//!
//! | Mode | Branch condition | Active set |
//! |------|------------------|------------|
//! | Sequential push | `active ≤ 64` | 4 forced neurons |
//! | Parallel push (rayon) | `active > 64` | 80 forced neurons |
//!
//! ## Delay semantics
//!
//! A spike fired at step `s` delivers current at step `s + max(delay, 1)`:
//! delay 0 cannot be delivered in the *same* step (the post neuron has already
//! stepped — ring buffer is read-before-write), so it arrives on the very next
//! step. Delays 2, 5, 31 arrive exactly 2, 5, 31 steps later.
//!
//! Delivery is observed as a B-population spike. Note that
//! [`NeuralPopulation::step`] clamps injected current to ±100, so the weight
//! (500) is capped to an effective 100. We therefore use a fast LIF
//! (τ = 2 ms → inv_tau = 0.5, plausible for insect neurons): a single pulse
//! yields Δv = 100 · 0.5 · 0.5 = 25 mV → v_new = -45 ≥ v_thresh (-54) → B
//! fires exactly on the arrival step, with a healthy 9 mV margin.

use pillars::api::{Network, PopulationId};
use pillars::zoo::neurons::LIFModel;

// LIF param indices (see zoo/neurons/lif.rs)
const P_DT: usize = 0;
const P_INV_TAU: usize = 1;
const P_V_REST: usize = 2;
const P_V_THRESH: usize = 3;
const P_V_RESET: usize = 4;
const P_TAU_REF: usize = 5;

// LIF state indices
const S_V: usize = 0;

/// Forced spike voltage — well above v_thresh (-54 mV).
const FORCE_V: f32 = 100.0;

/// Synaptic weight — exceeds the ±100 `i_in` clamp in `NeuralPopulation::step`,
/// so B effectively receives 100.0 (with inv_tau = 0.5, dt = 0.5 that is
/// Δv = 25 mV — enough to cross v_thresh from rest).
const W_STRONG: f32 = 500.0;

/// Run length — covers the largest delay (31) plus wraparound margin.
const RUN_STEPS: usize = 70;

/// The four delays under test (0 must deliver next-step, 2/5/31 exact).
const TEST_DELAYS: [u8; 4] = [0, 2, 5, 31];

/// Configure a LIF population with standard parameters.
fn setup_lif(net: &mut Network, pop: PopulationId) {
    let lif = net.pop_typed_mut::<LIFModel>(pop).expect("pop is LIF");
    lif.fill_param(P_DT, 0.5);
    // Fast LIF (τ = 2 ms): with the ±100 i_in clamp, a single arriving pulse
    // gives Δv = 100 · inv_tau · dt = 100 · 0.5 · 0.5 = 25 mV — enough to fire
    // from rest (-45 ≥ v_thresh -54) exactly on the arrival step.
    lif.fill_param(P_INV_TAU, 0.5);
    lif.fill_param(P_V_REST, -70.0);
    lif.fill_param(P_V_THRESH, -54.0);
    lif.fill_param(P_V_RESET, -70.0);
    lif.fill_param(P_TAU_REF, 2.0);
    lif.fill_state(S_V, -70.0);
}

/// Build `A → B`: one synapse per neuron, `A[i] → B[i]` with `delays[i]`.
fn build_network(n_neurons: usize, delays: &[u8]) -> (Network, PopulationId, PopulationId) {
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(n_neurons);
    let b = net.add_population::<LIFModel>(n_neurons);
    setup_lif(&mut net, a);
    setup_lif(&mut net, b);

    let from: Vec<u32> = (0..n_neurons as u32).collect();
    let to: Vec<u32> = from.clone();
    let weights = vec![W_STRONG; n_neurons];
    net.connect(a, &from, b, &to, &weights, delays);
    net.build();
    (net, a, b)
}

/// Expected delivery step for a spike fired at step 0.
/// Delay 0 → step 1 (a spike cannot reach the post neuron in the same step).
fn expected_arrival(delay: u8) -> usize {
    (delay as usize).max(1)
}

/// Force every A neuron to spike at step 0, record B's spike raster per step.
fn run_and_record(net: &mut Network, a: PopulationId, b: PopulationId) -> Vec<Vec<i32>> {
    // Force spike: v = 100 ≫ v_thresh = -54 → all A neurons fire on step 0.
    net.pop_typed_mut::<LIFModel>(a)
        .expect("pop A is LIF")
        .fill_state(S_V, FORCE_V);

    let mut raster: Vec<Vec<i32>> = Vec::with_capacity(RUN_STEPS);
    for step in 0..RUN_STEPS {
        net.step();
        if step == 0 {
            let a_spikes = net.spikes(a);
            assert!(
                a_spikes.iter().all(|&s| s == 1),
                "forced A neurons must fire at step 0, got {a_spikes:?}"
            );
        }
        raster.push(net.spikes(b).to_vec());
    }
    raster
}

/// Assert each `A[i] → B[i]` synapse delivers current at the expected step
/// and nowhere else; every connected B neuron spikes exactly once.
fn assert_exact_delivery(raster: &[Vec<i32>], n_neurons: usize, delays: &[u8], mode: &str) {
    for (i, &delay) in delays.iter().enumerate() {
        let arrival = expected_arrival(delay);
        for (step, spikes) in raster.iter().enumerate() {
            assert_eq!(
                spikes[i],
                if step == arrival { 1 } else { 0 },
                "[{mode}] delay={delay}: A[{i}]→B[{i}] current must arrive at step \
                 {arrival}, got spike={} at step {step}",
                spikes[i],
            );
        }
    }

    // Every connected B neuron must have fired exactly once over the run.
    let total: i32 = raster.iter().map(|s| s.iter().sum::<i32>()).sum();
    assert_eq!(
        total as usize, n_neurons,
        "[{mode}] each of {n_neurons} B neurons must spike exactly once, total={total}"
    );
}

#[test]
fn test_delays_sequential_mode() {
    // 4 active neurons (≤ 64) → sequential push path.
    let n = TEST_DELAYS.len();
    let (mut net, a, b) = build_network(n, &TEST_DELAYS);
    let raster = run_and_record(&mut net, a, b);
    assert_exact_delivery(&raster, n, &TEST_DELAYS, "sequential");
}

#[test]
fn test_delays_rayon_branch() {
    // 80 active neurons (> 64) → parallel push branch on multi-core machines.
    let n = 80usize;
    let mut delays = vec![0u8; n];
    for (i, &d) in TEST_DELAYS.iter().enumerate() {
        delays[i] = d;
    }
    let (mut net, a, b) = build_network(n, &delays);
    let raster = run_and_record(&mut net, a, b);
    assert_exact_delivery(&raster, n, &delays, "rayon>64");
}
