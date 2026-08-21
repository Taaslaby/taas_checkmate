use super::*;
use crate::api::delay::DelayBuffer;
use crate::api::graph::MAX_DELAY_SLOTS;
use crate::api::synapse::{exp_waveform_factor, PlasticityProfile, SynapseKinetics};
use crate::api::NeuralPopulation;
use crate::zoo::neurons::LIFModel;

/// Helper: set up LIF params on a population.
fn setup_lif_params(pop: &mut crate::api::NeuralPopulation<LIFModel>) {
    pop.fill_param(0, 0.5); // dt
    pop.fill_param(1, 0.1); // inv_tau_m
    pop.fill_param(2, -70.0); // v_rest
    pop.fill_param(3, -54.0); // v_thresh
    pop.fill_param(4, -70.0); // v_reset
    pop.fill_param(5, 2.0); // tau_ref
    pop.fill_state(0, -70.0); // v
}

#[test]
fn test_network_empty() {
    let mut net = Network::new();
    net.build();
    net.step(); // must not crash
    assert_eq!(net.step_counter(), 1);
}

#[test]
fn test_network_single_pop() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(8);
    assert_eq!(pop, 0);
    assert_eq!(net.num_populations(), 1);
    assert_eq!(net.total_neurons(), 8);

    // Set params via typed access
    setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());

    net.build();
    assert!(net.is_built());

    // Step a few times — weak current, no external drive
    for _ in 0..10 {
        net.step();
    }
    assert_eq!(net.step_counter(), 10);
    let spikes = net.spikes(pop);
    assert_eq!(spikes.len(), 8);
}

#[test]
fn test_network_two_pops_connected() {
    let mut net = Network::new();

    // Two populations of LIF
    let a = net.add_population::<LIFModel>(8);
    let b = net.add_population::<LIFModel>(8);
    assert_eq!(a, 0);
    assert_eq!(b, 1);
    assert_eq!(net.total_neurons(), 16);

    // Set params for both
    for pid in 0..2 {
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
    }

    // Connect a[0] → b[0] with weight 1.0, no delay
    net.connect(a, &[0], b, &[0], &[1.0], &[0]);
    // Connect a[1] → b[1] with weight 0.5, 2-step delay
    net.connect(a, &[1], b, &[1], &[0.5], &[2]);

    net.build();
    assert!(net.is_built());

    // Step 50 times — neurons don't spike without external current
    // but the network should not crash
    for _ in 0..50 {
        net.step();
    }
    assert_eq!(net.step_counter(), 50);

    // Spikes should be empty (no external drive)
    let spikes_b = net.spikes(b);
    assert_eq!(
        spikes_b.iter().sum::<i32>(),
        0,
        "No spikes without external drive"
    );
}

/// A second `build()` drains the (now-empty) edge list, so it would rebuild
/// the graph with no connections. It must be rejected, not silently corrupt.
#[test]
#[should_panic(expected = "build() called twice")]
fn test_network_build_twice_panics() {
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(2);
    let b = net.add_population::<LIFModel>(2);
    net.connect(a, &[0], b, &[0], &[1.0], &[0]);
    net.build();
    net.build();
}

#[test]
fn test_network_ring_buffer_delays() {
    // Verify that delayed spikes end up in the correct future slot
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(2);
    let b = net.add_population::<LIFModel>(2);

    for pid in 0..2 {
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
    }

    // a[0] → b[0] with delay 3 steps, weight 10.0 (strong)
    net.connect(a, &[0], b, &[0], &[10.0], &[3]);

    net.build();

    // a has 2 neurons at global indices 0,1. b has 2 at 2,3.
    // a[0] = global 0, b[0] = global 2
    // The graph has one edge: pre=0 → post=2, weight=10.0, delay=3
    // After step 0: slot[0+3=3][2] += 10.0  (but we need a spike first)
    // Since neurons don't spike without current, this test verifies
    // non-crash behavior only.

    net.step();
    // Step counter = 1, no spikes expected

    // Manually verify: slot 3 should be all zeros (no spike happened)
    let slot3 = net.delay.slot(3 % MAX_DELAY_SLOTS);
    for (i, &val) in slot3.iter().enumerate() {
        assert!(
            val.abs() < 1e-10,
            "delay slot 3 [{i}] should be 0, got {val}"
        );
    }
}

#[test]
fn test_network_with_external_drive() {
    // Drive the population from the outside via the safe inject_current API.
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(8);

    setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());

    net.build();

    let idx: Vec<u32> = (0..8).collect();
    let vals = vec![30.0f32; 8]; // moderate drive
    let mut total_spikes = 0i32;
    for _ in 0..200 {
        net.inject_current(pop, &idx, &vals);
        net.step();
        total_spikes += net.spikes(pop).iter().sum::<i32>();
    }

    // LIF with I=30 must spike — the drive reached the neurons.
    assert!(
        total_spikes > 0,
        "LIF with I=30 must spike, got {total_spikes}"
    );
    assert_eq!(net.step_counter(), 200);
}

#[test]
#[should_panic(expected = "before build")]
fn test_inject_current_before_build_panics() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    net.inject_current(pop, &[0], &[1.0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn test_inject_current_bad_pop_panics() {
    let mut net = Network::new();
    net.add_population::<LIFModel>(2);
    net.build();
    net.inject_current(7, &[0], &[1.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_inject_current_bad_neuron_panics() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    net.build();
    net.inject_current(pop, &[5], &[1.0]);
}

#[test]
#[should_panic(expected = "must be finite")]
fn test_network_rejects_non_finite_weight() {
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(1);
    let b = net.add_population::<LIFModel>(1);
    net.connect(a, &[0], b, &[0], &[f32::NAN], &[0]);
}

#[test]
#[should_panic(expected = "must be finite")]
fn test_network_rejects_non_finite_current() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(1);
    net.build();
    net.inject_current(pop, &[0], &[f32::INFINITY]);
}

#[test]
fn test_network_many_connections() {
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(10);
    let b = net.add_population::<LIFModel>(10);

    for pid in 0..2 {
        setup_lif_params(net.pop_typed_mut::<LIFModel>(pid).unwrap());
    }

    // All-to-all: each of 10 neurons connects to all 10 of the other pop
    let n = 10usize;
    let total_synapses = n * n;
    let mut from = Vec::with_capacity(total_synapses);
    let mut to = Vec::with_capacity(total_synapses);
    let mut weights = Vec::with_capacity(total_synapses);
    let mut delays = Vec::with_capacity(total_synapses);

    for i in 0..n {
        for j in 0..n {
            from.push(i as u32);
            to.push(j as u32);
            weights.push(0.1);
            delays.push(0);
        }
    }

    net.connect(a, &from, b, &to, &weights, &delays);
    net.build();

    assert!(net.is_built());

    for _ in 0..10 {
        net.step();
    }
    assert_eq!(net.step_counter(), 10);
}

/// expOneSynapse: after a unit jump, the injected current must decay as
/// `e^(−t/τ)` — peak immediately, then exponential tail (not a delta).
#[test]
fn kinetic_exp_one_decays_exponentially() {
    let tau = 10.0f32;
    let mut rt = KineticRuntime::new(SynapseKinetics::exp_decay(tau), 1.0);
    rt.allocate(1);
    let mut delay = DelayBuffer::new(1);
    let mut scratch = vec![0.0f32; 1];
    let mut i_in = vec![0.0f32; 1];
    // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
    let v = vec![-1.0f32; 1];

    // One arriving spike (weight = 1.0) in slot 0.
    delay.add(0, 0, 1.0);
    let a = (-1.0f32 / tau).exp();
    for step in 0..5 {
        i_in[0] = 0.0;
        rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
        let expect = a.powi(step);
        assert!(
            (i_in[0] - expect).abs() < 1e-5,
            "step {step}: {} vs {expect}",
            i_in[0]
        );
    }
}

/// expTwoSynapse: the double-exponential starts at 0, rises to a
/// normalised peak near t_peak, then decays — and the peak ≈ the jump.
#[test]
fn kinetic_exp_two_rises_then_decays() {
    let k = SynapseKinetics::exp_rise_decay(1.0, 3.0);
    let mut rt = KineticRuntime::new(k, 0.1);
    rt.allocate(1);
    let mut delay = DelayBuffer::new(1);
    let mut scratch = vec![0.0f32; 1];
    let mut i_in = vec![0.0f32; 1];
    // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
    let v = vec![-1.0f32; 1];

    delay.add(0, 0, 1.0);
    let mut trace = Vec::with_capacity(150);
    for _ in 0..150 {
        i_in[0] = 0.0;
        rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
        trace.push(i_in[0]);
    }

    // Starts at 0 (rise phase), peaks ≈ 1.0 (normalised), then decays.
    assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
    let peak = trace.iter().cloned().fold(0.0f32, f32::max);
    assert!((peak - 1.0).abs() < 0.05, "peak ≈ 1.0: {peak}");
    // Peak near t_peak = 1·3/(3−1)·ln(3) ≈ 1.648 ms → index ≈ 16.
    let peak_i = trace
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        ((peak_i as f32) * 0.1 - 1.648).abs() < 0.25,
        "peak near t_peak: index {peak_i}"
    );
    // Monotone rise to the peak, decay after, tail ≈ 0 (≈5×τ_decay).
    assert!(trace[5] < trace[peak_i]);
    assert!(trace[40] < trace[peak_i]);
    assert!(trace[149].abs() < 0.05, "tail decays: {}", trace[149]);
}

/// Conductance synapses respect the driving force `erev − V`: zero at the
/// reversal potential (shunting), depolarising below it, hyperpolarising
/// above it.
#[test]
fn kinetic_conductance_respects_driving_force() {
    let k = SynapseKinetics::exp_decay(10.0).with_erev(-65.0);
    let mut rt = KineticRuntime::new(k, 1.0);
    rt.allocate(3);
    let mut delay = DelayBuffer::new(3);
    let mut scratch = vec![0.0f32; 3];
    let mut i_in = vec![0.0f32; 3];
    // One unit jump to all three neurons.
    delay.add(0, 0, 1.0);
    delay.add(0, 1, 1.0);
    delay.add(0, 2, 1.0);
    let v = vec![-65.0f32, -70.0, -60.0];
    rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 3);
    // At V = erev: pure shunt, no current.
    assert!(i_in[0].abs() < 1e-6, "shunt at erev: {}", i_in[0]);
    // V below erev → positive (depolarising toward erev).
    assert!((i_in[1] - 5.0).abs() < 1e-6, "depolarising: {}", i_in[1]);
    // V above erev → negative (hyperpolarising toward erev).
    assert!((i_in[2] + 5.0).abs() < 1e-6, "hyperpolarising: {}", i_in[2]);
}

/// alphaSynapse: the alpha function `g(t) = A·(t/τ)·e^(1−t/τ)` starts at 0,
/// peaks at `A` near `t = τ`, and matches the closed form at every step
/// (exponential Euler is exact for this linear ODE).
#[test]
fn kinetic_alpha_peaks_at_tau() {
    let tau = 3.0f32;
    let dt = 0.1f32;
    let mut rt = KineticRuntime::new(SynapseKinetics::alpha(tau), dt);
    rt.allocate(1);
    let mut delay = DelayBuffer::new(1);
    let mut scratch = vec![0.0f32; 1];
    let mut i_in = vec![0.0f32; 1];
    // erev(0) − v(−1) = 1 → the current equals the conductance waveform.
    let v = vec![-1.0f32; 1];

    delay.add(0, 0, 1.0); // unit jump A = 1
    let mut trace = Vec::with_capacity(150);
    for _ in 0..150 {
        i_in[0] = 0.0;
        rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
        trace.push(i_in[0]);
    }

    // g(t) = A·(t/τ)·e^(1−t/τ) = (t/τ)·e·e^(−t/τ) at every sampled step.
    for (step, &got) in trace.iter().enumerate().take(20) {
        let t = step as f32 * dt;
        let expect = (t / tau) * std::f32::consts::E * (-t / tau).exp();
        assert!(
            (got - expect).abs() < 1e-5,
            "step {step}: {got} vs {expect}"
        );
    }

    // Starts at 0, peaks ≈ the jump (1.0) at t ≈ tau = 3 ms.
    assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
    let (peak_i, peak) =
        trace.iter().enumerate().fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            },
        );
    assert!((peak - 1.0).abs() < 0.05, "peak ≈ 1.0: {peak}");
    assert!(((peak_i as f32) * dt - tau).abs() < 0.5, "peak near t=tau");
    assert!(trace[149].abs() < 0.1, "tail decays: {}", trace[149]);
}

/// expThreeSynapse: the two-component conductance matches its closed form
/// `gbase1·(B−A) + gbase2·(C−A)` at every step and starts at 0 (the shared
/// rise term cancels the initial per-component jumps).
#[test]
fn kinetic_exp_three_matches_closed_form() {
    let (tr, td1, td2, g1, g2) = (1.0f32, 3.0f32, 10.0f32, 1.0f32, 0.5f32);
    let dt = 0.1f32;
    let mut rt = KineticRuntime::new(SynapseKinetics::exp_three(tr, td1, td2, g1, g2), dt);
    rt.allocate(1);
    let mut delay = DelayBuffer::new(1);
    let mut scratch = vec![0.0f32; 1];
    let mut i_in = vec![0.0f32; 1];
    let v = vec![-1.0f32; 1];

    delay.add(0, 0, 1.0);
    let wf1 = exp_waveform_factor(tr, td1);
    let wf2 = exp_waveform_factor(tr, td2);
    let k_a = (g1 * wf1 + g2 * wf2) / (g1 + g2);

    // 300 steps = 30 ms ≈ 3× the slow decay (τ2 = 10 ms) so the tail has
    // fully relaxed.
    let mut trace = Vec::with_capacity(300);
    for _ in 0..300 {
        i_in[0] = 0.0;
        rt.step(&mut delay, 0, &mut scratch, &v, &mut i_in, 1);
        trace.push(i_in[0]);
    }

    for (step, &got) in trace.iter().enumerate() {
        let t = step as f32 * dt;
        let expect = g1 * (wf1 * (-t / td1).exp() - k_a * (-t / tr).exp())
            + g2 * (wf2 * (-t / td2).exp() - k_a * (-t / tr).exp());
        assert!(
            (got - expect).abs() < 1e-5,
            "step {step}: {got} vs {expect}"
        );
    }
    assert!(trace[0].abs() < 1e-6, "starts at 0: {}", trace[0]);
    assert!(trace.iter().all(|x| x.is_finite() && *x >= -1e-6));
    assert!(trace[299].abs() < 0.1, "tail decays: {}", trace[299]);
}

/// expThreeSynapse with a zero second component must reduce exactly to
/// `expTwoSynapse` with `gbase = gbase1` (the ring carries the connection
/// weight for expThree, and `weight·gbase` for expTwo).
#[test]
fn kinetic_exp_three_zero_second_component_matches_exp_two() {
    let (tr, td) = (1.0f32, 3.0f32);
    let gbase = 2.0f32;
    let dt = 0.1f32;
    let mut a = KineticRuntime::new(SynapseKinetics::exp_three(tr, td, 10.0, gbase, 0.0), dt);
    let mut b = KineticRuntime::new(SynapseKinetics::exp_rise_decay(tr, td), dt);
    a.allocate(1);
    b.allocate(1);
    let mut da = DelayBuffer::new(1);
    let mut db = DelayBuffer::new(1);
    let mut sa = vec![0.0f32; 1];
    let mut sb = vec![0.0f32; 1];
    let mut ia = vec![0.0f32; 1];
    let mut ib = vec![0.0f32; 1];
    let v = vec![-1.0f32; 1];

    da.add(0, 0, 1.0); // expThree: connection weight
    db.add(0, 0, gbase); // expTwo: weight · gbase
    for _ in 0..100 {
        ia[0] = 0.0;
        ib[0] = 0.0;
        a.step(&mut da, 0, &mut sa, &v, &mut ia, 1);
        b.step(&mut db, 0, &mut sb, &v, &mut ib, 1);
        assert!((ia[0] - ib[0]).abs() < 1e-5, "{} vs {}", ia[0], ib[0]);
    }
}

/// Registering an invalid profile must be rejected before build.
#[test]
fn register_kinetics_rejects_invalid_profile() {
    let mut net = Network::new();
    assert!(net
        .register_kinetics(SynapseKinetics::exp_decay(0.0), 1.0)
        .is_err());
    assert!(net
        .register_kinetics(SynapseKinetics::exp_rise_decay(3.0, 1.0), 1.0)
        .is_err());
}

/// A Poisson source at `p = rate·dt = 1` fires every step and routes its
/// weight through the delay ring to the post neuron (like a presynaptic
/// spike); at 0 Hz it never fires, leaving the neuron at rest.
#[test]
fn poisson_source_routes_spikes_through_synapses() {
    // rate=2000 Hz, dt=0.5 ms → p=1.0 → fires every step.
    let mut net = Network::new();
    let post = net.add_population::<LIFModel>(1);
    setup_lif_params(net.pop_typed_mut::<LIFModel>(post).unwrap());
    let src = net.add_poisson_source(2000.0, 0.5, 42);
    net.connect_poisson_source(src, post, &[0], &[30.0], &[0], &[0], &[0]);
    net.build();

    let mut fired = 0i32;
    for _ in 0..200 {
        net.step();
        fired += net.spikes(post)[0];
    }
    assert!(
        fired > 0,
        "Poisson drive (p=1) must make the LIF fire, got {fired}"
    );

    // rate=0 → never fires → the neuron stays silent.
    let mut net0 = Network::new();
    let post0 = net0.add_population::<LIFModel>(1);
    setup_lif_params(net0.pop_typed_mut::<LIFModel>(post0).unwrap());
    let src0 = net0.add_poisson_source(0.0, 0.5, 42);
    net0.connect_poisson_source(src0, post0, &[0], &[30.0], &[0], &[0], &[0]);
    net0.build();
    let mut fired0 = 0i32;
    for _ in 0..200 {
        net0.step();
        fired0 += net0.spikes(post0)[0];
    }
    assert_eq!(fired0, 0, "0 Hz Poisson must never fire");
}

/// Adding a Poisson source or connecting one after `build()` is rejected.
#[test]
#[should_panic(expected = "before build")]
fn poisson_source_after_build_panics() {
    let mut net = Network::new();
    net.add_population::<LIFModel>(1);
    net.build();
    net.add_poisson_source(100.0, 0.5, 0);
}

/// Tsodyks–Markram depression end-to-end: pre (driven by strong external
/// current) fires repeatedly through a static edge into post 1 and through
/// a plastic edge (U0 = 0.5, τ_rec = 10 ms) into post 2. The plastic edge
/// delivers R·U of every spike — each pre-spike depletes R by (1 − U0), so
/// the sustained drive reaches post 2 only weakly and it stays silent,
/// while post 1 keeps firing at full weight.
#[test]
fn plastic_synapse_depresses_repeated_drive() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(3); // pre + 2 posts
    let lif = net.pop_typed_mut::<LIFModel>(pop).unwrap();
    setup_lif_params(lif);
    // Short refractory + near-rest threshold → pre fires every ~3 steps,
    // giving the static post a clear drive margin and the depressed post a
    // clear silence margin (steady-state delivered ~11 vs full 100).
    lif.fill_param(5, 1.0); // tau_ref
    lif.fill_param(3, -65.0); // v_thresh
    let plast = net
        .register_plasticity(PlasticityProfile::depression(0.5, 10.0), 0.5)
        .unwrap();
    assert_eq!(plast, 1, "first profile gets id 1");
    // pre(0) → post(1) static; pre(0) → post(2) plastic.
    net.connect_edges(
        pop,
        pop,
        &[
            Edge::new(0, 1, 100.0, 0),
            Edge::with_synapse_plasticity(0, 2, 100.0, 0, 0, plast),
        ],
    );
    net.build();
    assert_eq!(net.plast_types(), &[0, 1], "edge 1 carries the TM profile");

    let mut fired_static = 0i32;
    let mut fired_plastic = 0i32;
    for _ in 0..500 {
        net.inject_current(pop, &[0], &[100.0]);
        net.step();
        fired_static += net.spikes(pop)[1];
        fired_plastic += net.spikes(pop)[2];
    }
    assert!(
        fired_static > 20,
        "static edge keeps driving post 1: {fired_static}"
    );
    assert_eq!(
        fired_plastic, 0,
        "depressed edge must silence post 2 (got {fired_plastic})"
    );
}

/// Registering an invalid plasticity profile is rejected before build.
#[test]
fn register_plasticity_rejects_invalid_profile() {
    let mut net = Network::new();
    assert!(net
        .register_plasticity(PlasticityProfile::depression(0.0, 1.0), 0.5)
        .is_err());
    assert!(net
        .register_plasticity(PlasticityProfile::depression(1.5, 1.0), 0.5)
        .is_err());
    assert!(net
        .register_plasticity(PlasticityProfile::depression(0.5, 0.0), 0.5)
        .is_err());
    assert!(net
        .register_plasticity(PlasticityProfile::depression(0.5, f32::NAN), 0.5)
        .is_err());
}

/// A gap junction is a symmetric ohmic coupling: neuron 0 (depolarised) and
/// neuron 1 (at rest) exchange `g·ΔV`, pulling them toward each other.
#[test]
fn gap_junction_couples_two_neurons() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());
    // Neuron 0 depolarised to -55; neuron 1 stays at rest -70.
    net.pop_typed_mut::<LIFModel>(pop)
        .unwrap()
        .state_slice_mut(0)[0] = -55.0;
    net.connect_gap_junction(pop, 0, pop, 1, 0.5);
    net.build();

    net.step();
    let v0 = net.state(pop, 0)[0];
    let v1 = net.state(pop, 0)[1];
    // Neuron 1 receives g·(V0 − V1) > 0 → rises above rest.
    assert!(v1 > -70.0, "coupled neuron should rise, got {v1}");
    // Neuron 0 receives g·(V1 − V0) < 0 → falls below its start.
    assert!(v0 < -55.0, "driven neuron should fall, got {v0}");
}

/// Gap-junction conductance must be finite and non-negative at connect time.
#[test]
#[should_panic(expected = "conductance must be finite")]
fn gap_junction_rejects_non_finite_conductance() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    net.connect_gap_junction(pop, 0, pop, 1, f32::NAN);
}

/// A graded (analog) synapse is a **one-way** ohmic coupling: the post
/// neuron receives `g·(V_pre − V_post)`, the pre neuron is unaffected (only
/// its leak acts on it).
#[test]
fn graded_synapse_is_one_way_coupling() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    setup_lif_params(net.pop_typed_mut::<LIFModel>(pop).unwrap());
    net.pop_typed_mut::<LIFModel>(pop)
        .unwrap()
        .state_slice_mut(0)[0] = -55.0;
    net.connect_graded_synapse(pop, 0, pop, 1, 0.5);
    net.build();

    net.step();
    let v0 = net.state(pop, 0)[0];
    let v1 = net.state(pop, 0)[1];
    // Post neuron receives g·(V0 − V1) = 7.5 → rises above rest.
    assert!(v1 > -70.0, "post neuron should rise, got {v1}");
    // Pre neuron only relaxes toward rest (leak), NOT pulled by coupling:
    // leak-only dv = (v_rest − v)·inv_tau·dt = (−15)·0.05 = −0.75.
    assert!(
        (v0 - (-55.75)).abs() < 1e-3,
        "pre neuron must be unaffected by the graded coupling, got {v0}"
    );
}

/// Graded-synapse conductance must be finite and non-negative at connect time.
#[test]
#[should_panic(expected = "conductance must be finite")]
fn graded_synapse_rejects_non_finite_conductance() {
    let mut net = Network::new();
    let pop = net.add_population::<LIFModel>(2);
    net.connect_graded_synapse(pop, 0, pop, 1, f32::NAN);
}

/// The sigmoid `gradedSynapse` activation follows V_pre: `s` rises toward
/// `inf`, and the post neuron receives `weight·g·s·(erev − V_post)`.
#[test]
fn sigmoid_graded_runtime_activates_and_drives_current() {
    let spec = SigmoidGradedSynapse::new(-50.0, 5.0, 0.5, 0.0, 1.0);
    let mut rt = SigmoidGradedRuntime::new(spec, 0.5);
    rt.conns.push((0, 1, 1.0, 0.0)); // pre=0, post=1, weight=1, s=0
    let v = vec![-40.0f32, -70.0];
    let mut i_in = vec![0.0f32; 2];
    rt.step(&v, &mut i_in);
    // arg = (−40 −(−50))/5 = 2 → inf ≈ 0.8808, tau ≈ 0.2384 ms.
    // s = inf·(1 − e^(−0.5/0.2384)) ≈ 0.8808·0.8772 ≈ 0.7726.
    // i = 1·1·0.7726·(0 −(−70)) ≈ 54.08.
    assert!(
        i_in[1] > 40.0 && i_in[1] < 65.0,
        "driving current: {}",
        i_in[1]
    );
    assert_eq!(i_in[0], 0.0, "pre receives nothing (one-way)");
    let s = rt.conns[0].3;
    assert!(s > 0.0 && s < 0.8808, "s rises toward inf: {s}");
}

/// Invalid sigmoid graded profiles (`delta == 0`, `k <= 0`) are rejected.
#[test]
fn register_sigmoid_graded_rejects_invalid() {
    let mut net = Network::new();
    assert!(net
        .register_sigmoid_graded(SigmoidGradedSynapse::new(-50.0, 0.0, 0.5, 0.0, 1.0), 0.5)
        .is_err()); // delta = 0
    assert!(net
        .register_sigmoid_graded(SigmoidGradedSynapse::new(-50.0, 5.0, 0.0, 0.0, 1.0), 0.5)
        .is_err()); // k = 0
}
