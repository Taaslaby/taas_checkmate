//! Kernel Progonator 🧪 — comprehensive test runner for all neural kernels.
//!
//! Tests each model with:
//! - Known inputs → expected outputs
//! - Edge cases: n=0, n=1, n=7, n=8, n=13, n=100
//! - SIMD/scalar alignment consistency
//! - Numerical stability (long runs)
//! - NaN/inf protection
//! - Boundary conditions (clamping)

#![allow(non_snake_case)]

use taas_kernels::*;
use taas_kernels::neuron::hh::{standard_channels, steady_state_gates};
use taas_kernels::build_hh_channel_meta;


// ═══════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════

fn assert_all_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(diff < tol,
            "{label}[{i}]: got {a:.6}, expected {e:.6}, diff={diff:.6}");
    }
}

fn assert_no_nan(data: &[f32], label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(!v.is_nan(), "{label}[{i}] = NaN");
        assert!(!v.is_infinite(), "{label}[{i}] = Inf ({v})");
    }
}

// ═══════════════════════════════════════════════════════
//  1. SIGNAL KERNELS
// ═══════════════════════════════════════════════════════

#[test]
fn test_safety_clamp_correctness() {
    for &n in &[0, 1, 3, 7, 8, 13, 64, 100] {
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 50.0) * 0.1).collect();
        safety_clamp(&mut data, -1.0, 1.0);
        for &v in &data {
            assert!(v >= -1.0 && v <= 1.0, "n={n}: clamp failed, v={v}");
        }
    }
}

#[test]
fn test_safety_clamp_alignment_consistency() {
    let mut d7: Vec<f32> = vec![-5.0, -2.0, -1.0, 0.0, 0.5, 3.0, 10.0];
    let mut d8: Vec<f32> = d7.iter().copied().chain(std::iter::once(7.0)).collect();
    let mut d13: Vec<f32> = d8.iter().copied().chain(vec![0.0; 5]).collect();

    safety_clamp(&mut d7, -1.0, 1.0);
    safety_clamp(&mut d8, -1.0, 1.0);
    safety_clamp(&mut d13, -1.0, 1.0);

    for i in 0..7 {
        assert!((d7[i] - d8[i]).abs() < 1e-6, "clamp mismatch at [{i}]: d7={}, d8={}", d7[i], d8[i]);
        assert!((d7[i] - d13[i]).abs() < 1e-6, "clamp mismatch at [{i}]: d7={}, d13={}", d7[i], d13[i]);
    }
}

#[test]
fn test_safety_clamp_no_nan() {
    let mut data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0, -1.0, 42.0, -42.0];
    safety_clamp(&mut data, -1.0, 1.0);
    assert_no_nan(&data, "safety_clamp(nan/inf)");
    for &v in &data {
        assert!(v >= -1.0 && v <= 1.0, "clamp(nan/inf) failed: {v}");
    }
}

#[test]
fn test_relu_simd_correctness() {
    for &n in &[0, 1, 3, 7, 8, 13, 100] {
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 50.0) * 0.5).collect();
        relu_simd(&mut data);
        for (i, &v) in data.iter().enumerate() {
            let expected = ((i as f32 - 50.0) * 0.5).max(0.0);
            assert!((v - expected).abs() < 1e-6, "relu n={n}[{i}]: {v} != {expected}");
        }
    }
}

#[test]
fn test_add_simd_correctness() {
    let mut a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| 10.0 - i as f32).collect();
    add_simd(&mut a, &b);
    for i in 0..16 {
        assert!((a[i] - 10.0).abs() < 1e-6, "add[{i}]: {}", a[i]);
    }
}

#[test]
fn test_sigmoid_gate_correctness() {
    for &n in &[0, 1, 7, 8, 13, 64] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 3.0) * 2.0).collect();
        let mut output = vec![0.0; n];
        sigmoid_gate(&mut output, &input, 1.0, 0.0);
        for (i, &v) in output.iter().enumerate() {
            let x = (i as f32 - 3.0) * 2.0;
            let expected = 1.0 / (1.0 + (-x).exp());
            let diff = (v - expected).abs();
            assert!(diff < 0.01, "sigmoid n={n}[{i}]: {v:.4} != {expected:.4}");
            assert!(v >= 0.0 && v <= 1.0, "sigmoid n={n}[{i}] out of [0,1]: {v}");
        }
    }
}

#[test]
fn test_phase_gate_apply_correctness() {
    let n = 16;
    let input = vec![1.0; n];
    let r_high = vec![10.0; n];
    let r_low = vec![-10.0; n];
    let mut out_high = vec![0.0; n];
    let mut out_low = vec![0.0; n];
    phase_gate_apply(&mut out_high, &input, &r_high, 4.0, 0.0);
    phase_gate_apply(&mut out_low, &input, &r_low, 4.0, 0.0);
    assert!(out_high[0] > 0.99, "phase_gate high: {}", out_high[0]);
    assert!(out_low[0] < 0.01, "phase_gate low: {}", out_low[0]);
}

#[test]
fn test_t3_tact_generator() {
    let n = 8;
    // All above threshold
    let ph1 = vec![1.0; n];
    let ph2 = vec![1.0; n];
    let ph3 = vec![1.0; n];
    let mut pulses = vec![0.0; n];
    t3_tact_generator(&mut pulses, &ph1, &ph2, &ph3, 0.5);
    for &p in &pulses { assert!((p - 1.0).abs() < 1e-6, "TACT: {p}"); }

    // All below threshold
    let ph1 = vec![-1.0; n];
    let ph2 = vec![-1.0; n];
    let ph3 = vec![-1.0; n];
    t3_tact_generator(&mut pulses, &ph1, &ph2, &ph3, 0.5);
    for &p in &pulses { assert!(p.abs() < 1e-6, "TACT off: {p}"); }
}

#[test]
fn test_sgd_step_simd() {
    let n = 16;
    let mut velocity = vec![0.0; n];
    let mut weight = vec![1.0; n];
    let grad = vec![-0.1; n];
    sgd_step_simd(&mut velocity, &mut weight, &grad, 0.01, 0.9);
    for i in 0..n {
        assert!((velocity[i] + 0.01).abs() < 1e-6, "sgd vel[{i}]");
        assert!((weight[i] - 1.0001).abs() < 1e-6, "sgd weight[{i}]");
    }
}

// ═══════════════════════════════════════════════════════
//  2. PHASE KERNELS
// ═══════════════════════════════════════════════════════

#[test]
fn test_kuramoto_prepare_correctness() {
    for &n in &[0, 1, 7, 8, 13, 64] {
        let mut phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut cos_out = vec![0.0; n];
        let mut sin_out = vec![0.0; n];
        kuramoto_prepare(&mut phases, &mut cos_out, &mut sin_out);
        for i in 0..n {
            let phi = ((i as f32) * 0.5) - (std::f32::consts::TAU * (((i as f32) * 0.5) / std::f32::consts::TAU).round());
            let expected_cos = phi.cos();
            let expected_sin = phi.sin();
            assert!((cos_out[i] - expected_cos).abs() < 1e-4, "cos n={n}[{i}]: {:.4} != {:.4}", cos_out[i], expected_cos);
            assert!((sin_out[i] - expected_sin).abs() < 1e-4, "sin n={n}[{i}]: {:.4} != {:.4}", sin_out[i], expected_sin);
        }
    }
}

#[test]
fn test_kuramoto_update_correctness() {
    let n = 8;
    let mut phases = vec![0.0; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.0; n];
    kuramoto_update(&mut phases, &omegas, &coupling, 0.01);
    for i in 0..n { assert!((phases[i] - 0.01).abs() < 1e-6, "kuramoto_update[{i}]"); }
}

#[test]
fn test_kuramoto_update_viscous_delegation() {
    let n = 16;
    let mut phases = vec![0.0; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.0; n];
    kuramoto_update_viscous(&mut phases, &omegas, &coupling, 0.0, 0.01);
    for i in 0..n {
        assert!((phases[i] - 0.01).abs() < 1e-6, "viscous eta=0 [{i}]");
    }
}

#[test]
fn test_kuramoto_update_viscous_synchronizing() {
    let n = 16;
    let mut phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
    let omegas = vec![0.1; n];
    let coupling = vec![0.0; n];
    for _ in 0..100 {
        kuramoto_update_viscous(&mut phases, &omegas, &coupling, 5.0, 0.1);
    }
    let mean = phases.iter().sum::<f32>() / n as f32;
    for &p in &phases {
        let diff = (p - mean).abs();
        assert!(diff < 0.5, "viscous not synchronized: diff={diff}, p={p}, mean={mean}");
    }
}

#[test]
fn test_t3_update_basic() {
    let n = 16;
    let mut phases = vec![0.0; n];
    let mut amps = vec![1.0; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.0; n];
    let drives = vec![0.0; n];
    t3_update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.01, 0.1, 1.0, 1.0);
    assert!(phases.iter().any(|&p| p.abs() > 0.001), "t3_update should advance phases");
    for &r in &amps {
        assert!(r >= 0.001 && r <= 10.0, "t3 amp out of range: {r}");
    }
    assert_no_nan(&phases, "t3_update phases");
    assert_no_nan(&amps, "t3_update amps");
}

#[test]
fn test_t3_update_stability() {
    let n = 64;
    let mut phases = vec![0.0; n];
    let mut amps = vec![1.0; n];
    let omegas: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let coupling = vec![0.0; n];
    let drives: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
    for _ in 0..10000 {
        t3_update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.01, 0.1, 1.0, 1.0);
    }
    assert_no_nan(&phases, "t3_update stability phases");
    assert_no_nan(&amps, "t3_update stability amps");
}

// ═══════════════════════════════════════════════════════
//  3. DECODE KERNELS
// ═══════════════════════════════════════════════════════

#[test]
fn test_nlms_train_convergence() {
    let n = 16;
    let mut weights = vec![0.0; n];
    let mut bias = vec![0.0; 1];
    for _ in 0..200 {
        let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let pred: f32 = weights.iter().zip(states.iter()).map(|(w, s)| w * s).sum::<f32>() + bias[0];
        let err = 1.0 - pred;
        nlms_train(&mut weights, &mut bias, &states, err, 0.1, 1e-8);
    }
    assert!(weights.iter().any(|&w| w.abs() > 0.01), "NLMS weights should adapt");
}

#[test]
fn test_kalman_update_convergence() {
    let n = 16;
    let mut weights = vec![0.0; n];
    let mut cov = vec![1.0; n];
    for _ in 0..200 {
        let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        let pred: f32 = weights.iter().zip(states.iter()).map(|(w, s)| w * s).sum();
        let err = 1.0 - pred;
        let s: f32 = states.iter().zip(cov.iter()).map(|(x, c)| c * x * x).sum::<f32>() + 0.1;
        kalman_update(&mut weights, &mut cov, &states, err, s);
    }
    assert!(weights.iter().any(|&w| w.abs() > 0.001), "Kalman weights should adapt");
    assert!(cov.iter().all(|&c| c < 1.0), "Cov should decrease");
}

#[test]
fn test_t3_complex_decoder() {
    let n_in = 8;
    let n_out = 2;
    let amps = vec![1.0; n_in];
    let cos_buf = vec![1.0; n_in];
    let weights: Vec<f32> = (0..(n_in * n_out)).map(|i| (i + 1) as f32 * 0.01).collect();
    let biases = vec![0.0; n_out];
    let mut output = vec![0.0; n_out];
    t3_complex_decoder(&mut output, &amps, &cos_buf, &weights, &biases);
    for o in 0..n_out {
        let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum();
        assert!((output[o] - expected).abs() < 1e-4, "complex_decoder[{o}]");
    }
}

// ═══════════════════════════════════════════════════════
//  4. MEASURE KERNELS
// ═══════════════════════════════════════════════════════

#[test]
fn test_phase_coherence_perfect_sync() {
    let n = 16;
    let sin_buf = vec![0.0; n];
    let cos_buf = vec![1.0; n];
    let amplitudes = vec![1.0; n];
    let mut coh = 0.0;
    phase_coherence(&mut coh, &sin_buf, &cos_buf, &amplitudes);
    assert!((coh - 1.0).abs() < 1e-4, "Perfect sync coherence = {coh}, expected 1.0");
}

#[test]
fn test_phase_coherence_random() {
    let n = 64;
    let phases: Vec<f32> = (0..n).map(|i| (i as f32) * std::f32::consts::TAU / n as f32).collect();
    let sin_buf: Vec<f32> = phases.iter().map(|p| p.sin()).collect();
    let cos_buf: Vec<f32> = phases.iter().map(|p| p.cos()).collect();
    let amplitudes = vec![1.0; n];
    let mut coh = 0.0;
    phase_coherence(&mut coh, &sin_buf, &cos_buf, &amplitudes);
    assert!(coh < 0.1, "Random phases coherence = {coh}, expected ~0");
}

#[test]
fn test_interference_identical_phases() {
    let n = 8;
    let phases = vec![0.0; n];
    let mut out = vec![0.0; n * n];
    interference(&mut out, &phases, &phases);
    for &v in &out { assert!((v - 1.0).abs() < 1e-4, "interference: {v}"); }
}

#[test]
fn test_band_energy() {
    let n = 32;
    let amplitudes = vec![1.0; n];
    let omegas: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5).collect();
    let band_edges = vec![0.0, 5.0, 10.0, 15.0, 100.0];
    let mut energies = vec![0.0; 4];
    band_energy(&mut energies, &amplitudes, &omegas, &band_edges);
    assert!(energies.iter().all(|&e| e >= 0.0), "band_energy negative");
    assert!((energies.iter().sum::<f32>() - n as f32).abs() < 1e-4, "band_energy sum mismatch");
}

#[test]
fn test_lazy_sincos_update() {
    let n = 16;
    let mut sin_out = vec![0.0; n];
    let mut cos_out = vec![1.0; n];
    let phases = vec![0.1; n];
    let mut prev = vec![0.0; n];
    lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut prev, 0.01);
    for i in 0..n {
        assert!((sin_out[i] - 0.1f32.sin()).abs() < 1e-4, "lazy_sincos[{i}] sin");
        assert!((cos_out[i] - 0.1f32.cos()).abs() < 1e-4, "lazy_sincos[{i}] cos");
    }
    let sin_before = sin_out.clone();
    let cos_before = cos_out.clone();
    lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut prev, 0.01);
    assert_all_close(&sin_out, &sin_before, 1e-6, "lazy_sincos unchanged sin");
    assert_all_close(&cos_out, &cos_before, 1e-6, "lazy_sincos unchanged cos");
}

#[test]
fn test_spike_entropy_uniform() {
    let counts = vec![10.0; 8];
    let (h, hp) = spike_entropy(&counts, 80.0, 3.0);
    assert!((h - 3.0).abs() < 0.01, "uniform entropy: {h}");
    assert!((hp - 0.0).abs() < 0.01, "uniform entropy penalty: {hp}");
}

#[test]
fn test_spike_entropy_sparse() {
    let counts = vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (h, _) = spike_entropy(&counts, 100.0, 3.0);
    assert!(h.abs() < 0.01, "sparse entropy: {h}, expected 0");
}

// ═══════════════════════════════════════════════════════
//  5. ATTENTION KERNELS
// ═══════════════════════════════════════════════════════

#[test]
fn test_xor_attention_identical() {
    let query = vec![0u8; 8];
    let keys = vec![0u8; 8 * 4];
    let values = vec![1.0, 0.0, 0.0, 0.0];
    let mut out = 0.0;
    xor_attention(&query, &keys, &values, &mut out);
    // All 4 keys are identical to query (sim=1.0), so output = average of values
    // values = [1.0, 0.0, 0.0, 0.0] → avg = 0.25
    assert!((out - 0.25).abs() < 0.01, "identical keys attention: {out}");
}

#[test]
fn test_rdrand_fill_range() {
    let mut buf = vec![0.0; 1024];
    rdrand_fill(&mut buf);
    for &v in &buf {
        assert!(v >= 0.0 && v < 1.0, "rdrand: {v} out of [0,1)");
    }
    assert!(buf.iter().any(|&v| v > 0.01), "rdrand: all zeros");
}

// ═══════════════════════════════════════════════════════
//  6. IZHKEVICH REFRACTORY
// ═══════════════════════════════════════════════════════

#[test]
fn test_izhikevich_refractory_works() {
    let n = 8;
    let mut v = vec![-65.0; n];
    let mut u = vec![-13.0; n];
    let mut trace = vec![0.0; n];
    let a = vec![0.02; n];
    let b = vec![0.2; n];
    let mut I = vec![30.0; n];
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0; n];
    let d = vec![8.0; n];
    let mut refractory = vec![0.0; n];

    izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5,
                    &mut spikes, &c, &d, &mut refractory, 3.0, None);

    if spikes.iter().any(|&s| s > 0) {
        for i in 0..n {
            if spikes[i] > 0 {
                assert!(refractory[i] > 0.0, "refractory not set after spike");
            }
        }
        izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5,
                        &mut spikes, &c, &d, &mut refractory, 3.0, None);
        assert_eq!(spikes.iter().sum::<i32>(), 0, "Spiked during refractory!");
    }
}

// ═══════════════════════════════════════════════════════
//  7. LIF
// ═══════════════════════════════════════════════════════

fn make_lif_params(n: usize, dt: f32, tau_m: f32, v_rest: f32, v_thresh: f32, v_reset: f32) -> Vec<f32> {
    let mut p = vec![0.0; n * 5];
    for i in 0..n {
        p[i] = dt;
        p[n + i] = tau_m;
        p[2*n + i] = v_rest;
        p[3*n + i] = v_thresh;
        p[4*n + i] = v_reset;
    }
    p
}

#[test]
fn test_lif_spike_rate() {
    for &n in &[1, 7, 8, 13] {
        let mut v = vec![-70.0; n];
        let mut I = vec![25.0; n];
        let mut spikes = vec![0i32; n];
        let params = make_lif_params(n, 0.5, 10.0, -70.0, -54.0, -70.0);
        let mut refractory = vec![0.0; n];
        let mut total = 0i32;
        for _ in 0..200 {
            lif_step(&mut v, &mut I, &mut spikes, &params, &mut refractory, 2.0, None);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "LIF n={n}: no spikes with I=25");
    }
}

#[test]
fn test_lif_analytical_accuracy() {
    let n = 8;
    let dt = 0.5;
    let tau_m = 10.0;
    let v_rest = -70.0;
    let v_thresh = -54.0;
    let v_reset = -70.0;
    let I_const = 15.0;
    let params = make_lif_params(n, dt, tau_m, v_rest, v_thresh, v_reset);
    let mut v = vec![v_rest; n];
    let mut Ii = vec![I_const; n];
    let mut spikes = vec![0i32; n];
    let mut refractory = vec![0.0; n];

    for _ in 0..200 {            lif_step(&mut v, &mut Ii, &mut spikes, &params, &mut refractory, 0.0, None);
    }
    for i in 0..n {
        assert!((v[i] - (-55.0)).abs() < 0.1, "LIF steady state: {:.4}", v[i]);
    }
}

// ═══════════════════════════════════════════════════════
//  8. STDP
// ═══════════════════════════════════════════════════════

#[test]
fn test_stdp_update_ltp() {
    let n = 16;
    let mut weights = vec![0.5; n];
    let traces_pre = vec![0.0; n];
    let traces_post = vec![1.0; n];
    stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, 0.0, 1.0);
    assert_all_close(&weights, &vec![0.6; n], 0.01, "stdp LTP");
}

#[test]
fn test_stdp_clamping() {
    let n = 8;
    let mut weights = vec![5.0; n];
    let traces_pre = vec![0.0; n];
    let traces_post = vec![0.0; n];
    stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, 0.0, 1.0);
    for &w in &weights { assert!(w <= 1.0, "stdp clamp max: {w}"); }

    let mut weights = vec![-5.0; n];
    stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, -0.5, 1.0);
    for &w in &weights { assert!(w >= -0.5, "stdp clamp min: {w}"); }
}

// ═══════════════════════════════════════════════════════
//  9. STP
// ═══════════════════════════════════════════════════════

#[test]
fn test_stp_facilitation() {
    let n = 8;
    let mut x = vec![1.0; n];
    let mut u = vec![0.2; n];
    let mut i_out = vec![0.0; n];
    let pre_spikes = vec![1i32; n];
    let tau_d = vec![200.0; n];
    let tau_f = vec![200.0; n];
    let u0 = vec![0.2; n];
    let a = vec![1.0; n];
    stp_step(&mut x, &mut u, &pre_spikes, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0);
    assert!(u[0] > 0.2, "u should increase: {}", u[0]);
    assert!(i_out[0] > 0.0, "i_out should be positive: {}", i_out[0]);
}

#[test]
fn test_stp_depression_no_spike() {
    let n = 8;
    let mut x = vec![0.5; n];
    let mut u = vec![0.2; n];
    let mut i_out = vec![0.0; n];
    let pre_spikes = vec![0i32; n];
    let tau_d = vec![50.0; n];
    let tau_f = vec![50.0; n];
    let u0 = vec![0.2; n];
    let a = vec![1.0; n];
    stp_step(&mut x, &mut u, &pre_spikes, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0);
    assert!(x[0] > 0.5, "x should recover toward 1: {}", x[0]);
    assert!((i_out[0] - 0.0).abs() < 1e-6, "i_out should stay 0: {}", i_out[0]);
}

// ═══════════════════════════════════════════════════════
//  10. DOPAMINE
// ═══════════════════════════════════════════════════════

#[test]
fn test_dopamine_traces_step() {
    let n = 8;
    let mut trace_pre = vec![0.0; n];
    let mut trace_post = vec![0.0; n];
    let mut eligibility = vec![0.0; n];
    let pre_spikes = vec![1i32; n];
    let post_spikes = vec![1i32; n];
    dopamine_traces_step(&mut trace_pre, &mut trace_post, &mut eligibility,
                         &pre_spikes, &post_spikes, 1.0, 20.0, 20.0, 200.0, 0.01, 0.01);
    assert!(trace_pre[0] > 0.0, "trace_pre should increase");
    assert!(trace_post[0] > 0.0, "trace_post should increase");
}

#[test]
fn test_dopamine_weight_update() {
    let n = 8;
    let mut weights = vec![0.5; n];
    let eligibility = vec![1.0; n];
    dopamine_weight_update(&mut weights, &eligibility, 0.5, 0.1, 0.0, 1.0);
    for &w in &weights {
        assert!((w - 0.55).abs() < 0.01, "dopamine weight: {w}");
    }
}

// ═══════════════════════════════════════════════════════
//  11. HH (Hodgkin-Huxley polynomial)
// ═══════════════════════════════════════════════════════

#[test]
fn test_hh_poly_step_basic() {
    let n = 8;
    let channels = standard_channels();
    let meta = build_hh_channel_meta(&channels);
    let n_gates: usize = meta.iter().map(|ch| ch.n_gates).sum();

    let mut v = vec![-65.0; n];
    let mut gates = vec![0.0; n_gates * n];
    let ii = vec![10.0; n];
    let g_leak = vec![0.3; n];
    let e_leak = vec![-54.0; n];
    let c_m = vec![1.0; n];

    let ss = steady_state_gates(-65.0, &channels);
    for gi in 0..n_gates {
        for ni in 0..n {
            gates[gi * n + ni] = ss[gi];
        }
    }

    hh_poly_step(&mut v, &mut gates, &ii, &g_leak, &e_leak, &c_m, &meta, 0.01, None);
    assert_no_nan(&v, "hh_poly v");
    assert_no_nan(&gates, "hh_poly gates");
}

#[test]
fn test_hh_poly_step_no_spike_small_current() {
    let n = 8;
    let channels = standard_channels();
    let meta = build_hh_channel_meta(&channels);
    let n_gates: usize = meta.iter().map(|ch| ch.n_gates).sum();

    let mut v = vec![-65.0; n];
    let mut gates = vec![0.0; n_gates * n];
    let ii = vec![0.0; n];
    let g_leak = vec![0.3; n];
    let e_leak = vec![-54.0; n];
    let c_m = vec![1.0; n];

    let ss = steady_state_gates(-65.0, &channels);
    for gi in 0..n_gates {
        for ni in 0..n { gates[gi * n + ni] = ss[gi]; }
    }

    for _ in 0..100 {
        hh_poly_step(&mut v, &mut gates, &ii, &g_leak, &e_leak, &c_m, &meta, 0.01, None);
    }
    for &vv in &v {
        assert!(vv < -40.0, "HH without input spiked: {vv}");
    }
}

// ═══════════════════════════════════════════════════════
//  12. FHN (FitzHugh-Nagumo)
// ═══════════════════════════════════════════════════════

#[test]
fn test_fhn_spikes_with_strong_current() {
    let n = 8;
    let mut v = vec![0.0; n];
    let mut w = vec![0.0; n];
    let ii = vec![1.0; n];
    let mut spikes = vec![0i32; n];
    let mut params = vec![0.0; n * 4];
    for i in 0..n {
        params[i] = 0.1;
        params[n + i] = 10.0;
        params[2*n + i] = 0.7;
        params[3*n + i] = 0.8;
    }
    let mut total = 0i32;
    for _ in 0..200 {
        fhn_step(&mut v, &mut w, &ii, &mut spikes, &params, None);
        total += spikes.iter().sum::<i32>();
    }
    assert!(total > 0, "FHN no spikes with I=1.0");
}

// ═══════════════════════════════════════════════════════
//  13. WILSON-COWAN
// ═══════════════════════════════════════════════════════

#[test]
fn test_wc_oscillation() {
    let n = 8;
    let mut e = vec![0.1; n];
    let mut i = vec![0.1; n];
    let i_e = vec![0.5; n];
    let i_i = vec![0.3; n];
    let mut se = vec![0.0; n];
    let mut si = vec![0.0; n];
    let mut params = vec![0.0; n * 8];
    for j in 0..n {
        params[j] = 0.1;
        params[n + j] = 5.0;
        params[2*n + j] = 10.0;
        params[3*n + j] = 12.0; // strong excitation
        params[4*n + j] = 8.0;
        params[5*n + j] = 8.0;
        params[6*n + j] = 2.0;
        params[7*n + j] = 1.0;
    }
    let e0 = e.clone();
    for _ in 0..200 {
        wc_step(&mut e, &mut i, &i_e, &i_i, &mut se, &mut si, &params);
    }
    let changed = e.iter().zip(e0.iter()).any(|(e_new, e_old)| (e_new - e_old).abs() > 0.01);
    assert!(changed, "WC should oscillate");
}

// ═══════════════════════════════════════════════════════
//  14. ADEX
// ═══════════════════════════════════════════════════════

#[test]
fn test_adex_step_basic() {
    for &n in &[1, 7, 8, 13] {
        let mut v = vec![-65.0; n];
        let mut u = vec![-13.0; n];
        let mut I = vec![20.0; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0; n * 9];
        for i in 0..n {
            params[i] = 0.5;
            params[n + i] = 10.0;
            params[2*n + i] = -65.0;
            params[3*n + i] = -50.0;
            params[4*n + i] = -65.0;
            params[5*n + i] = 2.0;
            params[6*n + i] = 100.0;
            params[7*n + i] = 0.0;
            params[8*n + i] = 10.0;
        }
        let mut refractory = vec![0.0; n];
        let mut total = 0i32;
        for _ in 0..200 {
            adex_step(&mut v, &mut u, &mut I, &mut spikes, &params, &mut refractory, 2.0, None);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx n={n}: no spikes");
    }
}

// ═══════════════════════════════════════════════════════
//  15. SAW (psi-wave)
// ═══════════════════════════════════════════════════════

#[test]
fn test_psi_wave_step_basic() {
    let n = 8;
    let mut v = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut h_real = vec![1.0; n];
    let mut h_imag = vec![0.0; n];
    let mut i_ext = vec![0.1; n];
    let mut speed = vec![0.1; n];

    for _ in 0..100 {
        psi_wave_step(&mut v, &mut u, &mut h_real, &mut h_imag, &mut i_ext, &mut speed, 0.5, 0.1);
    }
    assert_no_nan(&h_real, "psi_wave real");
    assert_no_nan(&h_imag, "psi_wave imag");
    assert_no_nan(&speed, "psi_wave speed");
}

// ═══════════════════════════════════════════════════════
//  16. EDGE CASES
// ═══════════════════════════════════════════════════════

#[test]
fn test_empty_inputs() {
    let mut empty1: Vec<f32> = vec![];
    let mut empty2: Vec<f32> = vec![];
    let mut empty3: Vec<f32> = vec![];
    let empty4: Vec<f32> = vec![];
    let mut empty5: Vec<f32> = vec![];
    let _empty_spikes = vec![0i32; 0];
    // n=0 izhikevich isn't tested separately (handled by scalar tail logic)

    safety_clamp(&mut empty1, -1.0, 1.0);
    relu_simd(&mut empty2);
    add_simd(&mut empty3, &empty4);
    sigmoid_gate(&mut empty5, &empty4, 1.0, 0.0);

    let mut empty_p1: Vec<f32> = vec![];
    let mut empty_cos: Vec<f32> = vec![];
    let mut empty_sin: Vec<f32> = vec![];
    kuramoto_prepare(&mut empty_p1, &mut empty_cos, &mut empty_sin);
}

#[test]
fn test_single_neuron() {
    let mut v = vec![-65.0];
    let mut u = vec![-13.0];
    let mut trace = vec![0.0];
    let a = vec![0.02];
    let b = vec![0.2];
    let mut I = vec![30.0];
    let mut spikes = vec![0i32];
    let c = vec![-65.0];
    let d = vec![8.0];
    let mut refractory = vec![0.0];
    let mut total = 0i32;
    for _ in 0..100 {        izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5,
                    &mut spikes, &c, &d, &mut refractory, 3.0, None);
        total += spikes.iter().sum::<i32>();
    }
    assert!(total > 0, "Single Izhikevich: no spikes");
}

#[test]
fn test_nan_input_safety() {
    let n = 8;
    let mut v = vec![f32::NAN; n];
    let mut u = vec![-13.0; n];
    let mut trace = vec![0.0; n];
    let a = vec![0.02; n];
    let b = vec![0.2; n];
    let mut I = vec![0.0; n];
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0; n];
    let d = vec![8.0; n];
    let mut refractory = vec![0.0; n];        izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5,
                        &mut spikes, &c, &d, &mut refractory, 3.0, None);
    assert!(!spikes.iter().any(|&s| s < 0), "negative spike count");
}

// ═══════════════════════════════════════════════════════
//  17. LONG-RUN STABILITY
// ═══════════════════════════════════════════════════════

#[test]
fn test_izhikevich_long_run_stability() {
    let n = 32;
    let mut v = vec![-65.0; n];
    let mut u = vec![-13.0; n];
    let mut trace = vec![0.0; n];
    let a = vec![0.02; n];
    let b = vec![0.2; n];
    let mut I = vec![20.0; n];
    let mut spikes = vec![0i32; n];
    let c = vec![-65.0; n];
    let d = vec![8.0; n];
    let mut refractory = vec![0.0; n];

    for _ in 0..100000 {
        izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 0.5, 0.5,
                        &mut spikes, &c, &d, &mut refractory, 3.0, None);
    }
    assert_no_nan(&v, "izhikevich long-run v");
    assert_no_nan(&u, "izhikevich long-run u");
    for &vv in &v {
        assert!(vv > -200.0 && vv < 100.0, "izhikevich long-run v={vv}");
    }
}

#[test]
fn test_lif_long_run_stability() {
    let n = 32;
    let mut v = vec![-70.0; n];
    let mut I = vec![20.0; n];
    let mut spikes = vec![0i32; n];
    let params = make_lif_params(n, 0.5, 10.0, -70.0, -54.0, -70.0);
    let mut refractory = vec![0.0; n];

    for _ in 0..100000 {
        lif_step(&mut v, &mut I, &mut spikes, &params, &mut refractory, 2.0, None);
    }
    assert_no_nan(&v, "lif long-run v");
    for &vv in &v { assert!(vv > -200.0 && vv < 100.0, "lif long-run v={vv}"); }
}
