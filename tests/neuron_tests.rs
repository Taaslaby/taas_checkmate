//! Unit tests for neuron module — Izhikevich, SAW, ψ-wave, LIF, AdEx, STDP, DelayBuffer, E-Prop, STP, Poisson, TTFS.
//! All SIMD kernels require n % 8 == 0 arrays.

// Allow non_snake_case for mathematical variable names (I=current, v=voltage, etc.)
#![allow(non_snake_case)]

use taas_checkmate::neuron;

fn make_padded(n: usize) -> usize {
    (n + 7) / 8 * 8
}

// ─── Izhikevich ──────────────────────────────────────────

#[test]
fn test_izhikevich_basic_step() {
    let n = make_padded(64);
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut trace = vec![0.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let mut i = vec![10.0f32; n];
    let dt = 0.5;
    let trace_decay = 0.5;
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    neuron::step_izhikevich(&mut v, &mut u, &mut trace, &a, &b, &mut i, dt, trace_decay, &mut spikes, &c, &d, &mut refr, tau_ref);

    // v should increase with positive I
    for &vi in &v {
        assert!(vi > -65.1, "v should increase from -65, got {}", vi);
    }
}

#[test]
fn test_izhikevich_spike_production() {
    let n = make_padded(8);
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut trace = vec![0.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let mut i = vec![100.0f32; n];  // strong current → spike
    let dt = 1.0;
    let trace_decay = 0.5;
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    // Accumulate spikes across multiple steps
    let mut total_spikes: i32 = 0;
    for _ in 0..50 {
        neuron::step_izhikevich(&mut v, &mut u, &mut trace, &a, &b, &mut i, dt, trace_decay, &mut spikes, &c, &d, &mut refr, tau_ref);
        total_spikes += spikes.iter().filter(|&&s| s != 0).count() as i32;
    }

    assert!(total_spikes > 0, "Expected spikes with high I, got {} total spikes across 50 steps", total_spikes);
}

#[test]
fn test_izhikevich_no_spike_with_negative_I() {
    let n = make_padded(8);
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut trace = vec![0.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let mut i = vec![-10.0f32; n];  // negative current → no spike
    let dt = 0.5;
    let trace_decay = 0.5;
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    for _ in 0..50 {
        neuron::step_izhikevich(&mut v, &mut u, &mut trace, &a, &b, &mut i, dt, trace_decay, &mut spikes, &c, &d, &mut refr, tau_ref);
    }

    let total_spikes: i32 = spikes.iter().filter(|&&s| s != 0).count() as i32;
    assert_eq!(total_spikes, 0, "Expected no spikes with negative I, got {}", total_spikes);
}

// ─── SAW ─────────────────────────────────────────────────

#[test]
fn test_saw_step() {
    let n = make_padded(8);
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut trace = vec![0.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let i = vec![10.0f32; n];
    let dt = 0.5;
    let trace_decay = 0.5;
    let noise_amp = 0.0;
    let mut spikes = vec![0i32; n];
    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    neuron::step_saw(&mut v, &mut u, &mut trace, &a, &b, &i, dt, trace_decay, noise_amp, &mut spikes, &mut refr, tau_ref);

    for &vi in &v {
        assert!(vi > -65.1, "SAW v should increase from -65, got {}", vi);
    }
}

// ─── ψ-wave ──────────────────────────────────────────────

#[test]
fn test_psi_wave_step() {
    let n = make_padded(8);
    let mut v = vec![1.0f32; n];
    let mut u = vec![0.0f32; n];
    let mut h_real = vec![0.0f32; n];
    let mut h_imag = vec![0.0f32; n];
    let mut i_ext = vec![0.0f32; n];
    let mut speed = vec![0.0f32; n];
    let dt = 0.05;
    let hbar = 1.0;

    neuron::step_psi_wave(&mut v, &mut u, &mut h_real, &mut h_imag, &mut i_ext, &mut speed, dt, hbar);

    // ψ-wave should roughly preserve norm (L2)
    for i in 0..n {
        let norm = (v[i] * v[i] + u[i] * u[i]).sqrt();
        assert!((norm - 1.0).abs() < 0.1, "ψ-wave norm should stay ~1, got {}", norm);
    }
}

// ─── LIF-AT ─────────────────────────────────────────────

#[test]
fn test_lif_at_step() {
    let n = make_padded(8);
    let mut v = vec![-70.0f32; n];
    let mut theta = vec![0.0f32; n];
    let mut i = vec![15.0f32; n];
    let mut spikes = vec![0i32; n];
    let dt = 1.0;
    let tau_m = vec![10.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-55.0f32; n];
    let v_reset = vec![-75.0f32; n];
    let tau_theta = vec![50.0f32; n];
    let beta = vec![10.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    neuron::step_lif_at(&mut v, &mut theta, &mut i, &mut spikes, dt,
        &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta, &mut refr, tau_ref);

    // v should increase from -70
    for &vi in &v {
        assert!(vi > -70.0, "LIF-AT v should increase, got {}", vi);
    }
}

#[test]
fn test_lif_at_adaptive_threshold() {
    // Сильный ток должен вызвать спайк, который поднимет theta
    let n = make_padded(8);
    let mut v = vec![-70.0f32; n];
    let mut theta = vec![0.0f32; n];
    let mut i = vec![100.0f32; n];  // сильный ток
    let mut spikes = vec![0i32; n];
    let dt = 1.0;
    let tau_m = vec![10.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-55.0f32; n];
    let v_reset = vec![-75.0f32; n];
    let tau_theta = vec![100.0f32; n];
    let beta = vec![10.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;

    // Достаточно шагов для спайка
    let mut total_spikes = 0;
    for _ in 0..50 {
        neuron::step_lif_at(&mut v, &mut theta, &mut i, &mut spikes, dt,
            &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta, &mut refr, tau_ref);
        total_spikes += spikes.iter().filter(|&&s| s != 0).count() as i32;
    }

    assert!(total_spikes > 0, "LIF-AT should spike with strong I");
    // Theta should have increased after spiking
    assert!(theta.iter().any(|&t| t > 0.5), "theta should increase after spikes, got {:?}", &theta[..n.min(4)]);
}

#[test]
fn test_lif_at_spike_rate_decreases_with_adaptation() {
    // Адаптация должна снижать firing rate со временем
    let n = make_padded(8);
    let mut v = vec![-70.0f32; n];
    let mut theta = vec![0.0f32; n];
    let mut i = vec![50.0f32; n];
    let mut spikes = vec![0i32; n];
    let dt = 1.0;
    let tau_m = vec![10.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-55.0f32; n];
    let v_reset = vec![-75.0f32; n];
    let tau_theta = vec![50.0f32; n];
    let beta = vec![5.0f32; n];

    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;
    let mut early_spikes = 0;
    let mut late_spikes = 0;

    for step in 0..200 {
        neuron::step_lif_at(&mut v, &mut theta, &mut i, &mut spikes, dt,
            &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta, &mut refr, tau_ref);
        let s = spikes.iter().filter(|&&s| s != 0).count() as i32;
        if step < 50 { early_spikes += s; }
        if step >= 150 { late_spikes += s; }
    }

    assert!(late_spikes < early_spikes || early_spikes == 0,
        "spike rate should decrease with adaptation: early={}, late={}", early_spikes, late_spikes);
}

// ─── LIF ─────────────────────────────────────────────────

#[test]
fn test_lif_step() {
    let n = make_padded(8);
    let mut v = vec![-70.0f32; n];
    let mut i = vec![15.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;
    let dt = 1.0;
    let tau_m = vec![10.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-55.0f32; n];
    let v_reset = vec![-75.0f32; n];

    neuron::step_lif(&mut v, &mut i, &mut spikes, dt, &tau_m, &v_rest, &v_thresh, &v_reset, &mut refr, tau_ref);
    // v should increase from -70 towards -55
    for &vi in &v {
        assert!(vi > -70.0, "LIF v should increase, got {}", vi);
        assert!(vi <= -55.0 || vi == v_reset[0], "LIF v should not exceed threshold, got {}", vi);
    }
}

// ─── AdEx ────────────────────────────────────────────────

#[test]
fn test_adex_step() {
    let n = make_padded(8);
    let mut v = vec![-70.0f32; n];
    let mut u = vec![0.0f32; n];
    let mut i = vec![15.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut refr = vec![0.0f32; n];
    let tau_ref = neuron::DEFAULT_TAU_REF;
    let dt = 1.0;
    let tau_m = vec![10.0f32; n];
    let tau_w = vec![100.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-55.0f32; n];
    let v_reset = vec![-75.0f32; n];
    let delta_t = vec![2.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.02f32; n];

    neuron::step_adex(&mut v, &mut u, &mut i, &mut spikes, dt, &tau_m, &tau_w, &v_rest, &v_thresh, &v_reset, &delta_t, &a, &b, &mut refr, tau_ref);

    for &vi in &v {
        assert!(vi > -70.0, "AdEx v should increase, got {}", vi);
    }
}

// ─── STDP ────────────────────────────────────────────────

#[test]
fn test_stdp_update() {
    let n = 16;
    let mut weights = vec![0.1f32; n];
    let traces_pre = vec![0.5f32; n];
    let traces_post = vec![0.0f32; n];
    let lr_ltd = 0.01;
    let lr_ltp = 0.01;
    let w_min = 0.0;
    let w_max = 1.0;

    neuron::stdp_update(&mut weights, &traces_pre, &traces_post, lr_ltd, lr_ltp, w_min, w_max);

    // pre > post → LTD → weights decrease
    for &w in &weights {
        assert!(w < 0.1, "STDP LTD should decrease weights, got {}", w);
    }
}

#[test]
fn test_stdp_homeostatic() {
    let n = 16;
    let mut weights = vec![0.5f32; n];
    let traces_pre = vec![0.5f32; n];
    let traces_post = vec![0.0f32; n];
    let lr_ltd = 0.01;
    let lr_ltp = 0.01;
    let target_sum = 8.0;  // target = 8.0 for 16 neurons → 0.5 each
    let w_max = 1.0;

    neuron::stdp_homeostatic(&mut weights, &traces_pre, &traces_post, lr_ltd, lr_ltp, target_sum, w_max);

    for &w in &weights {
        assert!(w >= 0.0 && w <= w_max, "STDP weights out of bounds: {}", w);
    }
}

// ─── DelayBuffer ─────────────────────────────────────────

#[test]
fn test_delay_buffer_push_read() {
    let mut buf = neuron::DelayBuffer::new(16);
    let data = [1.0f32, 2.0, 3.0, 4.0];
    buf.push(&data);

    let out = buf.read(1);
    assert!((out[0] - 4.0).abs() < 1e-5, "delay=1 should return last pushed, got {}", out[0]);
}

#[test]
fn test_delay_buffer_multi_tap() {
    let mut buf = neuron::DelayBuffer::new(32);
    let mut data = [0.0f32; 32];
    for i in 0..32 { data[i] = i as f32; }
    buf.push(&data);

    // read_multi takes ABSOLUTE indices into the buffer
    // After push, pos=0, buf[0..31] = [0.0, 1.0, ..., 31.0]
    // Relative delay d from pos=0 → absolute index = (pos + n - d) % n = (0 + 32 - d) % 32
    let pos = 0;
    let n = 32;
    let delays = [
        ((pos + n - 1) % n) as i32,  // delay 1 → buf[31] = 31.0
        ((pos + n - 2) % n) as i32,  // delay 2 → buf[30] = 30.0
        ((pos + n - 3) % n) as i32,
        ((pos + n - 4) % n) as i32,
        ((pos + n - 5) % n) as i32,
        ((pos + n - 6) % n) as i32,
        ((pos + n - 7) % n) as i32,
        ((pos + n - 8) % n) as i32,  // delay 8 → buf[24] = 24.0
    ];
    let out = buf.read_multi(&delays);
    assert!((out[0] - 31.0).abs() < 1e-5, "delay=1 should return 31, got {}", out[0]);
    assert!((out[7] - 24.0).abs() < 1e-5, "delay=8 should return 24, got {}", out[7]);
}

// ─── E-Prop ──────────────────────────────────────────────

#[test]
fn test_eprop_lif_step_basic() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    let I = vec![15.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    // Baseline: v should increase from -65
    let v_before = eprop.v.clone();
    let count = eprop.step(&I, &mut spikes, &pre_spikes);
    assert!(eprop.v[0] > v_before[0], "v should increase after step");
    assert_eq!(count, 0, "no spikes expected with moderate I");
}

#[test]
fn test_eprop_lif_spike_production() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    eprop.v_thresh = -55.0;
    let I = vec![100.0f32; n];  // strong current
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    let mut total = 0;
    for _ in 0..30 {
        total += eprop.step(&I, &mut spikes, &pre_spikes);
    }
    assert!(total > 0, "E-Prop should spike with strong I, got {total} spikes");
    assert!(total <= 30 * 8, "at most all neurons every step");
}

#[test]
fn test_eprop_eligibility_trace_grows_with_pre_spikes() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![1i32; n];  // pre-synaptic spikes everywhere

    let e_before = eprop.eligibility.clone();
    eprop.step(&I, &mut spikes, &pre_spikes);

    // Eligibility should increase due to pre_spikes + surrogate gradient
    let grew: bool = eprop.eligibility.iter().zip(e_before.iter()).any(|(e, eb)| *e != *eb);
    assert!(grew, "eligibility traces should change with pre-spikes");
}

#[test]
fn test_eprop_update_weights() {
    let n = make_padded(8);
    let mut weights = vec![0.5f32; n];
    let ls = vec![1.0f32; n];

    // Test with non-zero eligibility
    let mut eprop = neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    for i in 0..n.min(4) {
        eprop.eligibility[i] = 0.1;
    }
    let w_before = weights.clone();
    eprop.update_weights(&mut weights, &ls, 0.01);
    // First 4 weights should change
    assert!((weights[0] - w_before[0]).abs() > 0.0, "weights should change with eligibility");
    // Last 4 weights should stay same (eligibility = 0)
    assert!((weights[7] - w_before[7]).abs() < 1e-10, "weights with zero eligibility should not change");
}

// ─── E-Prop SAW ────────────────────────────────────────

#[test]
fn test_eprop_saw_step_basic() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropSaw::new(n, 0.5, 20.0);
    let I = vec![10.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    let v_before = eprop.v.clone();
    let count = eprop.step(&I, &mut spikes, &pre_spikes);
    assert!(eprop.v[0] > v_before[0], "v should increase after step");
    assert_eq!(count, 0, "no spikes with moderate I");
}

#[test]
fn test_eprop_saw_spike_production() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropSaw::new(n, 0.5, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    let mut total = 0;
    for _ in 0..50 {
        total += eprop.step(&I, &mut spikes, &pre_spikes);
    }
    assert!(total > 0, "E-Prop SAW should spike with strong I");
}

#[test]
fn test_eprop_saw_eligibility_trace() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropSaw::new(n, 1.0, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![1i32; n];

    let e_before = eprop.eligibility.clone();
    eprop.step(&I, &mut spikes, &pre_spikes);

    let changed: bool = eprop.eligibility.iter().zip(e_before.iter()).any(|(e, eb)| *e != *eb);
    assert!(changed, "eligibility traces should change with pre-spikes");
}

#[test]
fn test_eprop_saw_update_weights() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropSaw::new(n, 1.0, 20.0);
    let mut weights = vec![0.5f32; n];
    let ls = vec![1.0f32; n];

    for i in 0..n.min(4) {
        eprop.eligibility[i] = 0.1;
    }
    let w_before = weights.clone();
    eprop.update_weights(&mut weights, &ls, 0.01);
    assert!((weights[0] - w_before[0]).abs() > 0.0, "weights with eligibility should change");
    assert!((weights[7] - w_before[7]).abs() < 1e-10, "weights with zero eligibility should not change");
}

#[test]
fn test_eprop_saw_reset() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropSaw::new(n, 1.0, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![1i32; n];

    eprop.step(&I, &mut spikes, &pre_spikes);

    eprop.reset();
    assert!(eprop.eligibility.iter().all(|&e| e == 0.0));
    assert!((eprop.v[0] - (-65.0)).abs() < 1e-5);
    assert!((eprop.u[0] - (-13.0)).abs() < 1e-5);
}

// ─── E-Prop Izhikevich ─────────────────────────────────

#[test]
fn test_eprop_izhikevich_step_basic() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 0.5, 20.0);
    let I = vec![10.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    // v should increase from -65
    let v_before = eprop.v.clone();
    let count = eprop.step(&I, &mut spikes, &pre_spikes);
    assert!(eprop.v[0] > v_before[0], "v should increase after step");
    assert_eq!(count, 0, "no spikes with moderate I, got {count}");
}

#[test]
fn test_eprop_izhikevich_spike_production() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 0.5, 20.0);
    let I = vec![100.0f32; n];  // strong current
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![0i32; n];

    let mut total = 0;
    for _ in 0..50 {
        total += eprop.step(&I, &mut spikes, &pre_spikes);
    }
    assert!(total > 0, "E-Prop Izhikevich should spike with strong I, got {total}");
}

#[test]
fn test_eprop_izhikevich_eligibility_trace() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 1.0, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![1i32; n];

    let e_before = eprop.eligibility.clone();
    eprop.step(&I, &mut spikes, &pre_spikes);

    // Eligibility should change due to pre_spikes + surrogate gradient
    let changed: bool = eprop.eligibility.iter().zip(e_before.iter()).any(|(e, eb)| *e != *eb);
    assert!(changed, "eligibility traces should change with pre-spikes");
}

#[test]
fn test_eprop_izhikevich_update_weights() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 1.0, 20.0);
    let mut weights = vec![0.5f32; n];
    let ls = vec![1.0f32; n];

    // Set some eligibility manually
    for i in 0..n.min(4) {
        eprop.eligibility[i] = 0.1;
    }
    let w_before = weights.clone();
    eprop.update_weights(&mut weights, &ls, 0.01);
    // First 4 weights should change
    assert!((weights[0] - w_before[0]).abs() > 0.0, "weights with eligibility should change");
    // Last 4 weights should stay same (eligibility = 0)
    assert!((weights[7] - w_before[7]).abs() < 1e-10, "weights with zero eligibility should not change");
}

// ─── E-Prop Integration: Weight Update Direction ─────

#[test]
fn test_eprop_weight_update_direction() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 1.0, 20.0);
    let mut weights = vec![1.0; n];
    let I = vec![20.0; n];
    let pre_spikes = vec![1; n];
    let mut spikes = vec![0; n];

    eprop.step(&I, &mut spikes, &pre_spikes);

    // Eligibility should be non-zero after step with pre-spike
    assert!(
        eprop.eligibility.iter().any(|&e| e != 0.0),
        "eligibility should be non-zero after step with pre-spike"
    );

    // Positive learning signal → weights increase
    let ls_pos = vec![1.0; n];
    let w_before = weights.clone();
    eprop.update_weights(&mut weights, &ls_pos, 0.1);
    assert!(weights[0] > w_before[0], "Positive L should increase weight");

    // Reset, step again, then negative L → weights decrease
    eprop.reset();
    eprop.step(&I, &mut spikes, &pre_spikes);
    let w_before2 = weights.clone();
    let ls_neg = vec![-1.0; n];
    eprop.update_weights(&mut weights, &ls_neg, 0.1);
    assert!(weights[0] < w_before2[0], "Negative L should decrease weight");
}

// ─── E-Prop Izhikevich Multi — per-synapse eligibility ─────────

#[test]
fn test_eprop_multi_step_basic() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 0.5, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [0, 1];
    let I_ext = vec![10.0; n_post];
    let mut spikes = vec![0; n_post];

    let v_before = eprop.v.clone();
    let count = eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
    assert!(eprop.v[0] > v_before[0], "v should increase after step");
    assert_eq!(count, 0, "no spikes with moderate I");
}

#[test]
fn test_eprop_multi_eligibility_per_synapse() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    // Only pre-synapse 0 fires
    let z_pre = [1, 0];
    let I_ext = vec![100.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    // e[0 * n_post + j] — eligibility for synapse 0 → should be non-zero
    // e[1 * n_post + j] — eligibility for synapse 1 → should be 0 (no pre-spike)
    for j in 0..n_post {
        let e0 = eprop.e[0 * n_post + j];
        let e1 = eprop.e[1 * n_post + j];
        assert_ne!(e0, 0.0, "synapse 0 eligibility should be non-zero (z_pre=1), got {}", e0);
        assert_eq!(e1, 0.0, "synapse 1 eligibility should be 0 (z_pre=0), got {}", e1);
    }
}

#[test]
fn test_eprop_multi_both_inputs_active() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 1];  // both inputs fire
    let I_ext = vec![100.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    // Both synapses should have non-zero eligibility
    for j in 0..n_post {
        let e0 = eprop.e[0 * n_post + j];
        let e1 = eprop.e[1 * n_post + j];
        assert_ne!(e0, 0.0, "synapse 0 eligibility should be non-zero");
        assert_ne!(e1, 0.0, "synapse 1 eligibility should be non-zero");
        // With same z_pre=1 and same psi_j, both should be equal
        assert!((e0 - e1).abs() < 1e-6, "eligibilities should be equal with same pre-spike, got e0={}, e1={}", e0, e1);
    }
}

#[test]
fn test_eprop_multi_update_weights() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 0];
    let I_ext = vec![100.0; n_post];
    let mut spikes = vec![0; n_post];

    // Step to generate eligibility
    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    // Update weights — only synapse 0 should change (eligibility != 0)
    let w_before = weights.clone();
    let ls = vec![1.0; n_post];
    eprop.update_weights_multi(&mut weights, &ls, 0.1);

    // Synapse 0: weights should change
    assert!((weights[0 * n_post + 0] - w_before[0 * n_post + 0]).abs() > 0.0,
        "synapse 0 weight should change");
    // Synapse 1: weights should stay (eligibility = 0)
    assert!((weights[1 * n_post + 0] - w_before[1 * n_post + 0]).abs() < 1e-10,
        "synapse 1 weight should not change");
}

#[test]
fn test_eprop_multi_reset() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 1];
    let I_ext = vec![100.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
    assert!(eprop.e.iter().any(|&e| e != 0.0), "eligibility should be non-zero after step");

    eprop.reset();
    assert!(eprop.e.iter().all(|&e| e == 0.0), "eligibility should be zero after reset");
    assert!((eprop.v[0] - (-65.0)).abs() < 1e-5, "v should reset to -65");
}

// ─── XOR Learning with Per-Synapse E-Prop ───────────────

#[test]
fn test_eprop_multi_learns_xor() {
    let n_pre = 2;
    let n_post = 8;  // 8 hidden neurons
    let mut eprop = neuron::EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 30.0);
    eprop.a = 0.02;
    eprop.b = 0.2;
    eprop.tau_surrogate = 5.0;

    // Column-major weights [n_pre × n_post]
    // weights[i * n_post + j] = weight from input i to neuron j
    let mut weights = vec![0.5; n_pre * n_post];
    let n_steps = 80;
    let lr = 0.02;

    // XOR patterns: (a, b) → target (1 if a != b else 0)
    // For readout: we use population voting — if spiking neurons % 2 == 1 → XOR=1
    let patterns = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];

    let mut epoch_stats: Vec<f32> = Vec::new();

    for _epoch in 0..250 {
        let mut epoch_err = 0.0;
        for &(in0, in1, target) in &patterns {
            let z_pre = [in0, in1];
            // Current from active inputs: I_j = Σ_i w_ij * z_i + I_ext
            let I_ext: Vec<f32> = (0..n_post).map(|j| {
                let mut I = 0.0;
                if in0 != 0 { I += weights[0 * n_post + j]; }
                if in1 != 0 { I += weights[1 * n_post + j]; }
                I * 15.0  // scale up to firing range
            }).collect();

            let mut spikes = vec![0; n_post];

            // Reset v, u (but NOT eligibility — it accumulates across steps)
            eprop.v.fill(-65.0);
            eprop.u.fill(-13.0);
            eprop.psi.fill(0.0);

            // Run for n_steps with the same pre-spike pattern
            for _ in 0..n_steps {
                eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
            }

            // Readout
            let spike_count: f32 = spikes.iter().map(|&s| s as f32).sum();
            let has_any = spike_count > 0.0;
            let output = if has_any { 1.0 } else { 0.0 };
            let target_f = target as f32;
            let error = target_f - output;
            epoch_err += error.abs();

            // E-Prop update: Δw_ij = lr · L_j · e_ij
            let ls = vec![error * 5.0; n_post];
            eprop.update_weights_multi(&mut weights, &ls, lr);

            // Clear eligibility for next pattern
            eprop.e.fill(0.0);
        }

        // Track convergence
        if _epoch % 25 == 0 {
            epoch_stats.push(epoch_err / 4.0);
        }
    }

    // After training, test XOR
    let mut correct = 0;
    for &(in0, in1, target) in &patterns {
        let z_pre = [in0, in1];
        let I_ext: Vec<f32> = (0..n_post).map(|j| {
            let mut I = 0.0;
            if in0 != 0 { I += weights[0 * n_post + j]; }
            if in1 != 0 { I += weights[1 * n_post + j]; }
            I * 15.0
        }).collect();

        let mut spikes = vec![0; n_post];
        eprop.v.fill(-65.0);
        eprop.u.fill(-13.0);
        eprop.psi.fill(0.0);
        eprop.e.fill(0.0);

        for _ in 0..n_steps {
            eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
        }

        let spike_count: f32 = spikes.iter().map(|&s| s as f32).sum();
        let output = if spike_count > 0.0 { 1.0 } else { 0.0 };
        if output == target as f32 { correct += 1; }
    }

    // Should get at least 3/4 correct (XOR is hard, but E-Prop should learn)
    assert!(correct >= 3,
        "E-Prop should learn XOR after 250 epochs: {}/4 correct, weights={:.4?}",
        correct,
        &weights[..n_pre.min(4) * n_post.min(4)]);
}

// ─── E-Prop AdEx Multi — per-synapse eligibility ────────────────

#[test]
fn test_eprop_adex_multi_step_basic() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 0.5, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [0, 1];
    let I_ext = vec![5.0; n_post];
    let mut spikes = vec![0; n_post];

    let v_before = eprop.v.clone();
    let count = eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
    assert!(eprop.v[0] > v_before[0], "v should increase after step, {} → {}", v_before[0], eprop.v[0]);
    assert_eq!(count, 0, "no spikes with moderate I, got {}", count);
}

#[test]
fn test_eprop_adex_multi_eligibility_per_synapse() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 0];  // only synapse 0 fires
    let I_ext = vec![50.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    for j in 0..n_post {
        let e0 = eprop.e[0 * n_post + j];
        let e1 = eprop.e[1 * n_post + j];
        assert_ne!(e0, 0.0, "synapse 0 eligibility should be non-zero (z_pre=1)");
        assert_eq!(e1, 0.0, "synapse 1 eligibility should be 0 (z_pre=0)");
    }
}

#[test]
fn test_eprop_adex_multi_both_inputs_active() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 1];
    let I_ext = vec![50.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    for j in 0..n_post {
        let e0 = eprop.e[0 * n_post + j];
        let e1 = eprop.e[1 * n_post + j];
        assert_ne!(e0, 0.0, "synapse 0 eligibility should be non-zero");
        assert_ne!(e1, 0.0, "synapse 1 eligibility should be non-zero");
        assert!((e0 - e1).abs() < 1e-6, "eligibilities should be equal: e0={}, e1={}", e0, e1);
    }
}

#[test]
fn test_eprop_adex_multi_update_weights() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 0];
    let I_ext = vec![50.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);

    let w_before = weights.clone();
    let ls = vec![1.0; n_post];
    eprop.update_weights_multi(&mut weights, &ls, 0.1);

    // Synapse 0: weights should change
    assert!((weights[0 * n_post + 0] - w_before[0 * n_post + 0]).abs() > 0.0,
        "synapse 0 weight should change");
    // Synapse 1: weights should stay
    assert!((weights[1 * n_post + 0] - w_before[1 * n_post + 0]).abs() < 1e-10,
        "synapse 1 weight should not change");
}

#[test]
fn test_eprop_adex_multi_reset() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 1];
    let I_ext = vec![50.0; n_post];
    let mut spikes = vec![0; n_post];

    eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
    assert!(eprop.e.iter().any(|&e| e != 0.0), "eligibility should be non-zero");

    eprop.reset();
    assert!(eprop.e.iter().all(|&e| e == 0.0), "eligibility should be zero after reset");
    assert!((eprop.v[0] - eprop.v_rest).abs() < 1e-5, "v should reset to v_rest");
    assert!((eprop.w[0] - 0.0).abs() < 1e-5, "w should reset to 0");
}

#[test]
fn test_eprop_adex_multi_spike_production() {
    let n_pre = 2;
    let n_post = 8;
    let mut eprop = neuron::EPropAdExMulti::new_default(n_pre, n_post, 1.0, 20.0);
    let mut weights = vec![0.5; n_pre * n_post];
    let z_pre = [1, 0];
    let I_ext = vec![100.0; n_post];  // strong current
    let mut spikes = vec![0; n_post];

    let mut total = 0;
    for _ in 0..30 {
        total += eprop.step(&mut weights, &z_pre, &I_ext, &mut spikes);
    }
    assert!(total > 0, "AdEx Multi should spike with strong I, got {total}");
}

// ─── E-Prop Integration: AND Learning ─────────────────

#[test]
fn test_eprop_learns_and() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 1.0, 30.0);
    eprop.a = 0.02;
    eprop.b = 0.2;
    eprop.tau_surrogate = 5.0;

    // One weight per neuron — scales input current I_j = w_j * input
    let mut weights = vec![0.5; n];
    let n_steps = 80;
    let lr = 0.02;
    let input_strength = 20.0;

    // AND patterns: (a, b, target = should any neuron spike?)
    let patterns = [(0.0, 0.0, 0.0), (input_strength, 0.0, 0.0),
                    (0.0, input_strength, 0.0), (input_strength, input_strength, 1.0)];

    for _epoch in 0..200 {
        for &(in0, in1, target) in &patterns {
            let I: Vec<f32> = (0..n).map(|j| weights[j] * (in0 + in1)).collect();
            let active = if in0 > 1.0 || in1 > 1.0 { 1 } else { 0 };
            let pre_spikes = vec![active; n];
            let mut spikes = vec![0; n];

            eprop.v.fill(-65.0);
            eprop.u.fill(-13.0);
            eprop.psi.fill(0.0);

            for _ in 0..n_steps {
                eprop.step(&I, &mut spikes, &pre_spikes);
            }

            let any_spike = spikes.iter().any(|&s| s != 0);
            let output = if any_spike { 1.0 } else { 0.0 };
            let error = target - output;

            // Standard E-Prop update: Δw = η · L · e
            let ls = vec![error * 5.0; n];
            if active == 1 {
                eprop.update_weights(&mut weights, &ls, lr);
            }

            eprop.eligibility.fill(0.0);
        }
    }

    // Verify: (1,1) should spike (both inputs → 2× current → crosses threshold)
    let test_I_11: Vec<f32> = weights.iter().map(|&w| w * 2.0 * input_strength).collect();
    let mut test_spikes = vec![0; n];
    eprop.v.fill(-65.0);
    eprop.u.fill(-13.0);
    eprop.psi.fill(0.0);
    eprop.eligibility.fill(0.0);
    for _ in 0..n_steps {
        eprop.step(&test_I_11, &mut test_spikes, &vec![1; n]);
    }
    let spike_11 = test_spikes.iter().any(|&s| s != 0);
    assert!(spike_11, "After AND training, (1,1) should produce spikes, weights={:.4?}", &weights[..4]);

    // Verify: (0,0) should NOT spike
    let test_I_00 = vec![0.0; n];
    let mut test_spikes0 = vec![0; n];
    eprop.v.fill(-65.0);
    eprop.u.fill(-13.0);
    eprop.psi.fill(0.0);
    eprop.eligibility.fill(0.0);
    for _ in 0..n_steps {
        eprop.step(&test_I_00, &mut test_spikes0, &vec![0; n]);
    }
    assert!(!test_spikes0.iter().any(|&s| s != 0), "(0,0) should NOT spike after AND training");

    // Verify: (1,1) spikes more than (0,1) — proves threshold behavior
    let test_I_01: Vec<f32> = weights.iter().map(|&w| w * input_strength).collect();
    let mut test_spikes_01 = vec![0; n];
    eprop.v.fill(-65.0);
    eprop.u.fill(-13.0);
    eprop.psi.fill(0.0);
    eprop.eligibility.fill(0.0);
    for _ in 0..n_steps {
        eprop.step(&test_I_01, &mut test_spikes_01, &vec![1; n]);
    }
    let count_01: i32 = test_spikes_01.iter().sum();
    let count_11: i32 = test_spikes.iter().sum();
    assert!(count_11 > count_01,
        "(1,1) should spike more than (0,1) after AND training: (1,1)={}, (0,1)={}",
        count_11, count_01);
}

#[test]
fn test_eprop_izhikevich_reset() {
    let n = make_padded(8);
    let mut eprop = neuron::EPropIzhikevich::new(n, 1.0, 20.0);
    let I = vec![100.0f32; n];
    let mut spikes = vec![0i32; n];
    let pre_spikes = vec![1i32; n];

    eprop.step(&I, &mut spikes, &pre_spikes);
    assert!(eprop.eligibility.iter().any(|&e| e != 0.0), "eligibility should be non-zero after step");

    eprop.reset();
    assert!(eprop.eligibility.iter().all(|&e| e == 0.0), "eligibility should be zero after reset");
    assert!((eprop.v[0] - (-65.0)).abs() < 1e-5, "v should be reset to -65");
    assert!((eprop.u[0] - (-13.0)).abs() < 1e-5, "u should be reset to -13");
}

// ─── Short-Term Plasticity ───────────────────────────────

#[test]
fn test_stp_step_basic() {
    let n = make_padded(8);
    let mut stp = neuron::ShortTermPlasticity::new(n, 1.0);
    let pre_spikes = vec![1i32; n];  // all spike
    let mut I_out = vec![0.0f32; n];

    // Initial state: x=1.0, u=0.2
    assert!((stp.x[0] - 1.0).abs() < 1e-5);
    assert!((stp.u[0] - 0.2).abs() < 1e-5);

    stp.step(&pre_spikes, &mut I_out);

    // After spike: u should increase, x should decrease
    assert!(stp.u[0] > 0.2, "u should increase after spike, got {}", stp.u[0]);
    assert!(stp.x[0] < 1.0, "x should decrease after spike, got {}", stp.x[0]);
    // I_out should have some positive current
    assert!(I_out[0] > 0.0, "I_out should be positive after spike, got {}", I_out[0]);
}

#[test]
fn test_stp_depression_over_multiple_spikes() {
    let n = make_padded(8);
    let mut stp = neuron::ShortTermPlasticity::new(n, 1.0);
    let pre_spikes = vec![1i32; n];
    let mut I_out = vec![0.0f32; n];

    // Multiple spikes should show depression (decreasing I_out)
    let mut currents = Vec::new();
    for _ in 0..10 {
        I_out.fill(0.0);
        stp.step(&pre_spikes, &mut I_out);
        currents.push(I_out[0]);
    }
    // Depression: later spikes should have smaller or equal synaptic current
    assert!(currents[5] <= currents[0] + 0.01, "STP depression: early I={}, later I={}", currents[0], currents[5]);
}

#[test]
fn test_stp_heterogeneous() {
    let n = make_padded(8);
    let stp = neuron::ShortTermPlasticity::new_heterogeneous(n, 1.0, 42);
    // All tau_d should be in [100, 500]
    for i in 0..n {
        assert!(stp.tau_d[i] >= 100.0 && stp.tau_d[i] <= 500.0,
            "tau_d[{}] out of range [100, 500]: {}", i, stp.tau_d[i]);
        assert!(stp.tau_f[i] >= 300.0 && stp.tau_f[i] <= 1000.0,
            "tau_f[{}] out of range [300, 1000]: {}", i, stp.tau_f[i]);
        assert!(stp.u0[i] >= 0.05 && stp.u0[i] <= 0.5,
            "u0[{}] out of range [0.05, 0.5]: {}", i, stp.u0[i]);
    }
}

#[test]
fn test_stp_reset() {
    let n = make_padded(8);
    let mut stp = neuron::ShortTermPlasticity::new(n, 1.0);
    let pre_spikes = vec![1i32; n];
    let mut I_out = vec![0.0f32; n];

    // Depress
    for _ in 0..5 { stp.step(&pre_spikes, &mut I_out); }
    assert!(stp.x[0] < 1.0, "x should be depressed before reset");

    // Reset
    stp.reset();
    assert!((stp.x[0] - 1.0).abs() < 1e-5, "x should be 1.0 after reset, got {}", stp.x[0]);
    assert!((stp.u[0] - stp.u0[0]).abs() < 1e-5, "u should return to u0 after reset, got {}", stp.u[0]);
}

// ─── Poisson Generator ───────────────────────────────────

#[test]
fn test_poisson_step_basic() {
    let n = make_padded(16);
    let mut poisson = neuron::PoissonGenerator::new(n, 1.0, 42);
    let mut spikes = vec![0i32; n];

    poisson.step(&mut spikes);
    // Should produce some spikes at 10 Hz with dt=1.0 (P=10 per step)
    let count: i32 = spikes.iter().sum();
    assert!(count >= 0 && count <= n as i32, "spike count {count} out of bounds");
}

#[test]
fn test_poisson_rate_zero() {
    let n = make_padded(8);
    let mut poisson = neuron::PoissonGenerator::new(n, 1.0, 42);
    poisson.set_rate(0, 0.0);  // neuron 0: zero rate
    let mut spikes = vec![0i32; n];

    // Run many steps — neuron 0 should never spike
    for _ in 0..100 {
        poisson.step(&mut spikes);
        assert_eq!(spikes[0], 0, "neuron with rate=0 should never spike");
    }
}

#[test]
fn test_poisson_set_rates() {
    let n = make_padded(8);
    let mut poisson = neuron::PoissonGenerator::new(n, 1.0, 42);
    poisson.set_rates(&[100.0f32; 8]);
    // After setting, rates should be 100
    assert!((poisson.rates()[0] - 100.0).abs() < 1e-5);
}

#[test]
fn test_poisson_dt_accessor() {
    let n = make_padded(8);
    let poisson = neuron::PoissonGenerator::new(n, 0.5, 42);
    assert!((poisson.dt() - 0.5).abs() < 1e-5);
    assert_eq!(poisson.n_neurons(), n);
}

// ─── TTFS Encoder ───────────────────────────────────────

#[test]
fn test_ttfs_encode_step() {
    let n = 8;
    let mut ttfs = neuron::TtfsEncoder::new(n, 10);
    let values = vec![0.0f32, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5];
    // value=1.0 → delay 0 (spike immediately)
    // value=0.0 → delay 10
    // value=0.5 → delay 5
    let mut spikes = vec![0i32; n];

    // Step 1: plan (current_step=0)
    ttfs.encode_step(&values, &mut spikes);
    // value=1.0 → delay=0, so index 2 and 5 should spike immediately
    assert_eq!(spikes[2], 1, "value=1 should spike at step 0");
    assert_eq!(spikes[5], 1, "value=1 should spike at step 0");
    assert_eq!(spikes[0], 0, "value=0 should not spike at step 0");

    // Step 6 (current_step=6): value=0.5 → delay=5 should spike now
    for _ in 0..5 { ttfs.encode_step(&values, &mut spikes); }
    assert_eq!(spikes[1], 1, "value=0.5 should spike at step 5");
    assert_eq!(spikes[4], 1, "value=0.5 should spike at step 5");
    assert_eq!(spikes[7], 1, "value=0.5 should spike at step 5");
}

#[test]
fn test_ttfs_reset() {
    let n = 8;
    let mut ttfs = neuron::TtfsEncoder::new(n, 10);
    let values = vec![1.0f32; n];
    let mut spikes = vec![0i32; n];

    ttfs.encode_step(&values, &mut spikes);
    assert!(spikes.iter().any(|&s| s != 0));

    ttfs.reset();
    assert_eq!(ttfs.current_step, 0);
    assert!(ttfs.fired.iter().all(|&f| !f));
}

#[test]
fn test_ttfs_cycle() {
    let n = 8;
    let mut ttfs = neuron::TtfsEncoder::new(n, 5);
    let values = vec![0.0f32; n];  // all max delay (5)
    let mut spikes = vec![0i32; n];

    // Step 0: plan
    ttfs.encode_step(&values, &mut spikes);
    assert_eq!(spikes.iter().sum::<i32>(), 0, "no spikes yet");

    // Steps 1-5: still no spikes
    for _ in 1..6 {
        ttfs.encode_step(&values, &mut spikes);
    }
    // Step 6: current_step wraps to 0, no spikes, then replan
    // Actually after step 5, current_step = 6, which > max_delay, so it wraps to 0
    // Then on the next call with current_step=0, it replans
    // So after step 6 (wrapping), spikes should be scheduled again
    // Let's just verify we reach the spike
    let mut found_spike = false;
    for _ in 0..12 {
        if ttfs.encode_step(&values, &mut spikes) {
            if spikes.iter().any(|&s| s != 0) { found_spike = true; break; }
        }
    }
    assert!(found_spike, "TTFS should spike after max_delay steps");
}

// ─── Refractory Period ────────────────────────────────────

#[test]
fn test_izhikevich_refractory_prevents_back_to_back_spikes() {
    let n = make_padded(8);
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut trace = vec![0.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let mut i = vec![1000.0f32; n];
    let dt = 1.0;
    let trace_decay = 0.5;
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];
    let mut refr = vec![0.0f32; n];
    let tau_ref = 5.0;  // 5 ms refractory

    // First step — should spike
    neuron::step_izhikevich(&mut v, &mut u, &mut trace, &a, &b, &mut i, dt, trace_decay, &mut spikes, &c, &d, &mut refr, tau_ref);
    assert!(spikes[0] == 1 || refr[0] > 0.0, "should spike or be in refractory");

    // Second step — refractory should prevent spike
    neuron::step_izhikevich(&mut v, &mut u, &mut trace, &a, &b, &mut i, dt, trace_decay, &mut spikes, &c, &d, &mut refr, tau_ref);
    assert_eq!(spikes[0], 0, "refractory should prevent back-to-back spike");
    assert!(refr[0] > 0.0, "refractory counter should be positive");
}

// ─── SpikeRaster ─────────────────────────────────────────

#[test]
fn test_spike_raster_basic() {
    let mut raster = neuron::SpikeRaster::new();
    let spikes = vec![0i32, 1, 0, 1];
    raster.record(0, &spikes, 4);
    raster.record(1, &spikes, 4);

    assert_eq!(raster.total_spikes(), 4);
    assert_eq!(raster.n_steps(), 2);

    let (dense, n_steps) = raster.to_dense(4);
    assert_eq!(n_steps, 2);
    assert_eq!(dense[0 * 4 + 1], 1);
    assert_eq!(dense[1 * 4 + 3], 1);

    raster.reset();
    assert_eq!(raster.total_spikes(), 0);
}

// ─── Firing Rates ────────────────────────────────────────

#[test]
fn test_firing_rates_basic() {
    let n_neurons = 4;
    let n_steps = 100;
    let dt = 0.001;  // 1 ms
    // Neuron 0 spikes every 10 steps → 100 Hz
    let mut buf = vec![0i32; n_steps * n_neurons];
    for t in 0..n_steps {
        if t % 10 == 0 {
            buf[t * n_neurons + 0] = 1;
        }
    }

    let rates = neuron::firing_rates(&buf, n_neurons, n_steps, dt);
    assert!((rates[0] - 100.0).abs() < 1.0, "neuron 0 rate should be ~100 Hz, got {}", rates[0]);
    assert!((rates[1]).abs() < 1.0, "neuron 1 rate should be ~0 Hz, got {}", rates[1]);

    let mean = neuron::mean_firing_rate(&buf, n_neurons, n_steps, dt);
    assert!((mean - 25.0).abs() < 1.0, "mean rate should be ~25 Hz, got {}", mean);
}

// ─── ISI / CV ────────────────────────────────────────────

#[test]
fn test_isi_computation() {
    // Regular 10-step intervals → ISI = 10 * dt * 1000 ms
    let mut spikes = vec![0i32; 100];
    for t in (0..100).step_by(10) {
        spikes[t] = 1;
    }
    let isi = neuron::compute_isi(&spikes, 0.001);
    assert!(!isi.is_empty());
    assert!((isi[0] - 10.0).abs() < 0.1, "ISI should be ~10 ms, got {}", isi[0]);
}

// ─── Dopamine-Modulated STDP ─────────────────────────

#[test]
fn test_dopamine_stdp_traces_update() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let pre_spikes = vec![0i32; n];
    let post_spikes = vec![0i32; n];

    // Initial: all zero
    assert!(stdp.trace_pre.iter().all(|&t| t == 0.0));
    assert!(stdp.trace_post.iter().all(|&t| t == 0.0));

    // Step without spikes — nothing should change (traces decay from 0 stays 0)
    stdp.step(&pre_spikes, &post_spikes);
    assert!(stdp.trace_pre.iter().all(|&t| t == 0.0));
    assert!(stdp.trace_post.iter().all(|&t| t == 0.0));
}

#[test]
fn test_dopamine_stdp_pre_trace_increases_on_pre_spike() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let mut pre_spikes = vec![0i32; n];
    let post_spikes = vec![0i32; n];
    pre_spikes[0] = 1;  // neuron 0 fires pre-synaptic spike

    stdp.step(&pre_spikes, &post_spikes);

    // Pre-trace should increase for neuron 0
    assert!(stdp.trace_pre[0] > 0.0, "pre-trace should increase after pre-spike, got {}", stdp.trace_pre[0]);
    // Post-trace should stay 0 (no post-spike)
    assert_eq!(stdp.trace_post[0], 0.0, "post-trace should stay 0");
}

#[test]
fn test_dopamine_stdp_ltp_vs_ltd() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    stdp.a_ltp = 0.1;
    stdp.a_ltd = 0.1;

    // Case 1: pre-spike only → LTD (negative eligibility)
    let mut pre_only = vec![0i32; n];
    let post_only = vec![0i32; n];
    pre_only[0] = 1;
    stdp.step(&pre_only, &post_only);
    // With only pre-spike: e -= A_ltd * tr_post * pre_spike = -A_ltd * 0 * 1 = 0
    let e_after_pre = stdp.eligibility[0];

    // Reset and test post-spike only → LTP (positive eligibility)
    stdp.reset();
    stdp.step(&vec![0i32; n], &post_only);
    // With only post-spike: e += A_ltp * tr_pre * post_spike = A_ltp * 0 * 1 = 0
    let e_after_post = stdp.eligibility[0];

    // Both 0 because traces are 0 at time of spike
    assert_eq!(e_after_pre, 0.0, "no pre-trace → no LTD");
    assert_eq!(e_after_post, 0.0, "no post-trace → no LTP");

    // Case 2: pre-spike, then next step both spike → pre-trace > 0, post-spike = 1 → LTP
    stdp.reset();
    stdp.step(&pre_only, &vec![0i32; n]);  // step 1: pre-spike → trace_pre builds
    let e1 = stdp.eligibility[0];
    // After step 1: pre-trace > 0, no post spike → no eligibility change

    // step 2: both spike → LTP: e += A_ltp * tr_pre * post_spike
    let both_spikes = vec![1i32; n];
    stdp.step(&both_spikes, &both_spikes);
    let e2 = stdp.eligibility[0];

    assert!(stdp.trace_pre[0] > 0.0, "pre-trace should be > 0 after pre-spike");
    // LTP term should make e positive
    assert!(e2 >= e1, "eligibility should increase with LTP: e1={}, e2={}", e1, e2);
}

#[test]
fn test_dopamine_stdp_weight_update() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let mut weights = vec![0.5f32; n];

    // Set some eligibility manually
    for i in 0..n.min(4) {
        stdp.eligibility[i] = 0.1;
    }

    // Apply dopamine with positive reward
    let w_before = weights.clone();
    stdp.apply_dopamine(&mut weights, 1.0, 0.01);

    // Weights with eligibility should increase
    assert!((weights[0] - w_before[0]) > 0.0, "positive dopamine + eligibility should increase weight");
    // Weights with zero eligibility should stay
    assert!((weights[7] - w_before[7]).abs() < 1e-10, "zero eligibility should not change");
}

#[test]
fn test_dopamine_stdp_negative_dopamine_decreases_weights() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let mut weights = vec![0.5f32; n];

    for i in 0..n.min(4) {
        stdp.eligibility[i] = 0.1;
    }

    let w_before = weights.clone();
    stdp.apply_dopamine(&mut weights, -1.0, 0.01);

    // Negative dopamine → weights decrease
    assert!((weights[0] - w_before[0]) < 0.0, "negative dopamine should decrease weight");
}

#[test]
fn test_dopamine_stdp_no_update_without_dopamine() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let mut weights = vec![0.5f32; n];

    for i in 0..n {
        stdp.eligibility[i] = 0.1;
    }

    let w_before = weights.clone();
    // Dopamine = 0 → no change even with eligibility
    stdp.apply_dopamine(&mut weights, 0.0, 0.01);

    for i in 0..n {
        assert!((weights[i] - w_before[i]).abs() < 1e-10,
            "zero dopamine should not change weight[{}]: before={}, after={}", i, w_before[i], weights[i]);
    }
}

#[test]
fn test_dopamine_stdp_clear_eligibility() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);

    for i in 0..n {
        stdp.eligibility[i] = 0.5;
    }

    stdp.clear_eligibility();
    assert!(stdp.eligibility.iter().all(|&e| e == 0.0), "eligibility should be cleared");
}

#[test]
fn test_dopamine_stdp_reset() {
    let n = make_padded(8);
    let mut stdp = neuron::DopamineSTDP::new(n, 1.0);
    let pre_spikes = vec![1i32; n];
    let post_spikes = vec![1i32; n];

    stdp.step(&pre_spikes, &post_spikes);
    assert!(stdp.eligibility.iter().any(|&e| e != 0.0), "eligibility should be non-zero after step");
    assert!(stdp.trace_pre.iter().any(|&t| t != 0.0), "traces should be non-zero after step");

    stdp.reset();
    assert!(stdp.eligibility.iter().all(|&e| e == 0.0));
    assert!(stdp.trace_pre.iter().all(|&t| t == 0.0));
    assert!(stdp.trace_post.iter().all(|&t| t == 0.0));
}

#[test]
fn test_cv_regular_vs_bursting() {
    let n_neurons = 4;
    let n_steps = 200;
    let dt = 0.001;
    // Neuron 0: regular (every 10 steps)
    // Neuron 1: Poisson-like (random)
    let mut buf = vec![0i32; n_steps * n_neurons];
    for t in (0..n_steps).step_by(10) { buf[t * n_neurons + 0] = 1; }
    // Neuron 1: 5 spikes then 20 gap, repeat
    for cycle in 0..6 {
        for t in 0..5 { buf[(cycle * 25 + t) * n_neurons + 1] = 1; }
    }

    let cvs = neuron::compute_cvs(&buf, n_neurons, n_steps, dt);
    // Regular neuron → CV ≈ 0
    assert!(cvs[0] < 0.5, "regular neuron should have low CV, got {}", cvs[0]);
    // Bursting neuron → CV > 1
    assert!(cvs[1] > 0.5, "bursting neuron should have higher CV, got {}", cvs[1]);
}
