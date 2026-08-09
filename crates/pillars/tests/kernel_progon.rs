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

use pillars::kernels::hh_channels::build_channel_meta;
use pillars::kernels::hh_channels::{standard_channels, steady_state_gates};
// Flat compat surface бывшего крейта `taas-kernels` (Фаза 11) — канон pillars.
use pillars::kernels::measure::lazy_sincos_update;
use pillars::kernels::signal::sgd_step_simd;
use pillars::kernels::{
    add_simd, band_energy, dopamine_traces_step, dopamine_weight_update, interference,
    kalman_update, kuramoto_update_viscous, nlms_train, phase_coherence, phase_gate_apply,
    relu_simd, safety_clamp, sigmoid_gate, spike_entropy, stdp_update, t3_complex_decoder,
    t3_tact_generator, t3_update, wc_step,
};

// ═══════════════════════════════════════════════════════
//  HELPERS (pillars-канон, транш 3b/3c)
//  Локальные обёртки сохраняют легаси-сигнатуры теста, делегируя в
//  run_neuron::<IzhikevichModel>/<LIFModel> (мосты safe:: удалены).
// ═══════════════════════════════════════════════════════

fn izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    _trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &mut [f32],
    dt: f32,
    _trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    _I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let mut p = vec![0.0f32; n * 7];
    for i in 0..n {
        p[i] = dt;
        p[n + i] = a[i];
        p[2 * n + i] = b[i];
        p[3 * n + i] = c[i];
        p[4 * n + i] = d[i];
        p[5 * n + i] = tau_ref;
        p[6 * n + i] = 30.0; // legacy v_th
    }
    unsafe {
        pillars::core::runner::run_neuron::<pillars::zoo::neurons::IzhikevichModel>(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            refractory.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            spikes.as_mut_ptr(),
            p.as_ptr(),
            n,
        );
    }
}

fn lif_step(
    v: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    params: &[f32], // legacy 5-col [dt, tau_m, v_rest, v_thresh, v_reset]
    refractory: &mut [f32],
    tau_ref: f32,
    _I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let mut p = vec![0.0f32; n * 6];
    for i in 0..n {
        p[i] = params[i];
        let tm = params[n + i];
        p[n + i] = if tm > 0.0 { 1.0 / tm } else { 0.0 };
        p[2 * n + i] = params[2 * n + i];
        p[3 * n + i] = params[3 * n + i];
        p[4 * n + i] = params[4 * n + i];
        p[5 * n + i] = tau_ref;
    }
    unsafe {
        pillars::core::runner::run_neuron::<pillars::zoo::neurons::LIFModel>(
            v.as_mut_ptr(),
            refractory.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            spikes.as_mut_ptr(),
            p.as_ptr(),
            n,
        );
    }
}

// ═══════════════════════════════════════════════════════
//  Local bridges: legacy signature → pillars канон (транш 3e, safe.rs удалён)
// ═══════════════════════════════════════════════════════

/// STP: legacy 9-аргументная сигнатура → pillars `ops::plasticity::stp` (скаляр).
fn stp_step(
    x: &mut [f32],
    u: &mut [f32],
    pre_spikes: &[i32],
    tau_d: &[f32],
    tau_f: &[f32],
    u0: &[f32],
    a: &[f32],
    i_out: &mut [f32],
    dt: f32,
) {
    let n = x.len();
    for i in 0..n {
        let s = if pre_spikes[i] != 0 { 1.0f32 } else { 0.0f32 };
        let (un, xn) = pillars::ops::plasticity::stp::stp_step(
            u[i],
            x[i],
            s,
            u0[i],
            1.0 / tau_d[i].max(1e-10),
            1.0 / tau_f[i].max(1e-10),
            dt,
        );
        u[i] = un;
        x[i] = xn;
        i_out[i] += s * a[i] * un * xn;
    }
}

/// HH poly: legacy SoA лейаут → `run_neuron::<HHModel>` (стандартные каналы).
fn hh_poly_step(
    v: &mut [f32],
    gates: &mut [f32],
    ii: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    channel_meta: &[pillars::kernels::hh_channels::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32,
    celsius: f32,
    _I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let n_channels = channel_meta.len();
    // Локальный мост покрывает только standard/pacemaking лейауты (как new_fused_hh);
    // для экзотических лейаутов легаси-fallback не воспроизводится — assert защищает.
    assert!(
        n_channels == 2 || n_channels == 3,
        "hh_poly_step: только standard/pacemaking лейауты"
    );
    let n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();
    assert_eq!(gates.len(), n_gates * n, "hh_poly_step: gates length");
    assert_eq!(ii.len(), n, "hh_poly_step: Ii length");
    assert_eq!(g_leak.len(), n, "hh_poly_step: g_leak length");
    assert_eq!(e_leak.len(), n, "hh_poly_step: E_leak length");
    assert_eq!(c_m.len(), n, "hh_poly_step: C_m length");
    assert_eq!(
        g_max_per_neuron.len(),
        n_channels * n,
        "hh_poly_step: g_max length"
    );
    let q10 = 3.0f32.powf((celsius - 6.3) / 10.0);
    let has_cat = channel_meta.len() == 3;
    let mut p = vec![0.0f32; n * 12];
    for i in 0..n {
        p[i] = dt;
        p[n + i] = g_max_per_neuron[i]; // g_Na
        p[2 * n + i] = channel_meta[0].e_rev; // E_Na
        p[3 * n + i] = g_max_per_neuron[n + i]; // g_K
        p[4 * n + i] = channel_meta[1].e_rev; // E_K
        p[5 * n + i] = g_leak[i]; // g_Leak
        p[6 * n + i] = e_leak[i]; // E_Leak
        p[7 * n + i] = q10;
        p[8 * n + i] = 0.0; // v_thresh (legacy @ 0)
        p[9 * n + i] = 1.0 / c_m[i].max(1e-6); // inv_cm
        p[10 * n + i] = if has_cat {
            g_max_per_neuron[2 * n + i]
        } else {
            0.0
        };
        p[11 * n + i] = if has_cat {
            channel_meta[2].e_rev
        } else {
            120.0
        };
    }
    let mut spikes = vec![0i32; n];
    let mut c_scratch = vec![0.0f32; n];
    let mut was_below = vec![0.0f32; n];
    unsafe {
        pillars::core::runner::run_neuron::<pillars::zoo::neurons::HHModel>(
            v.as_mut_ptr(),
            gates.as_mut_ptr(),
            gates.as_mut_ptr().add(n),     // h
            gates.as_mut_ptr().add(2 * n), // n_gate
            c_scratch.as_mut_ptr(),
            was_below.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            p.as_ptr(),
            n,
        );
    }
}

/// FHN: legacy 4-col → `run_neuron::<FHNModel>` (5-col + v_thresh=0.5).
fn fhn_step(
    v: &mut [f32],
    w: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    params: &[f32],
    _I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let mut p = vec![0.0f32; n * 5];
    let mut was_below = vec![0.0f32; n];
    for i in 0..n {
        p[i] = params[i];
        p[n + i] = if params[n + i] > 0.0 {
            1.0 / params[n + i]
        } else {
            0.0
        };
        p[2 * n + i] = params[2 * n + i];
        p[3 * n + i] = params[3 * n + i];
        p[4 * n + i] = 0.5; // legacy v_thresh
        was_below[i] = if v[i] < 0.5 { 1.0 } else { 0.0 };
    }
    unsafe {
        pillars::core::runner::run_neuron::<pillars::zoo::neurons::FHNModel>(
            v.as_mut_ptr(),
            w.as_mut_ptr(),
            was_below.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            p.as_ptr(),
            n,
        );
    }
}

/// AdEx: legacy 9-col → `run_neuron::<AdExModel>` (12-col, v_peak := v_thresh).
fn adex_step(
    v: &mut [f32],
    u: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    _I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let mut p = vec![0.0f32; n * 12];
    for i in 0..n {
        p[i] = params[i]; // dt
        p[n + i] = 1.0 / params[n + i].max(1e-6); // inv_tau_m
        p[2 * n + i] = 1.0 / params[2 * n + i].max(1e-6); // inv_tau_w
        p[3 * n + i] = params[3 * n + i]; // v_rest
        p[4 * n + i] = params[4 * n + i]; // v_thresh
        p[5 * n + i] = params[5 * n + i]; // v_reset
        p[6 * n + i] = params[6 * n + i]; // delta_t
        p[7 * n + i] = 1.0 / params[6 * n + i].max(1e-6); // inv_delta_t
        p[8 * n + i] = params[7 * n + i]; // a
        p[9 * n + i] = params[8 * n + i]; // b
        p[10 * n + i] = tau_ref;
        p[11 * n + i] = params[4 * n + i]; // v_peak := v_thresh
    }
    unsafe {
        pillars::core::runner::run_neuron::<pillars::zoo::neurons::AdExModel>(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            refractory.as_mut_ptr(), // state 2
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            p.as_ptr(),
            n,
        );
    }
}

/// Eprop AdEx multi: прямая делегация в pillars (I_dend=None → без изменений).
fn eprop_adex_step_multi(
    v: &mut [f32],
    w: &mut [f32],
    psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    i_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32,
    tau_m: f32,
    tau_w: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    delta_t: f32,
    a: f32,
    b: f32,
    tau_e: f32,
    tau_sg: f32,
    spikes: &mut [i32],
    _I_dend: Option<&[f32]>,
) -> usize {
    pillars::kernels::eprop::eprop_adex_step_multi(
        v, w, psi, e, z_pre, i_ext, weights, n_pre, dt, tau_m, tau_w, v_rest, v_thresh, v_reset,
        delta_t, a, b, tau_e, tau_sg, spikes,
    )
}

// ═══════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════

fn assert_all_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: got {a:.6}, expected {e:.6}, diff={diff:.6}"
        );
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
            assert!((-1.0..=1.0).contains(&v), "n={n}: clamp failed, v={v}");
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
        assert!(
            (d7[i] - d8[i]).abs() < 1e-6,
            "clamp mismatch at [{i}]: d7={}, d8={}",
            d7[i],
            d8[i]
        );
        assert!(
            (d7[i] - d13[i]).abs() < 1e-6,
            "clamp mismatch at [{i}]: d7={}, d13={}",
            d7[i],
            d13[i]
        );
    }
}

#[test]
fn test_safety_clamp_no_nan() {
    let mut data = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        -1.0,
        42.0,
        -42.0,
    ];
    safety_clamp(&mut data, -1.0, 1.0);
    assert_no_nan(&data, "safety_clamp(nan/inf)");
    for &v in &data {
        assert!((-1.0..=1.0).contains(&v), "clamp(nan/inf) failed: {v}");
    }
}

#[test]
fn test_relu_simd_correctness() {
    for &n in &[0, 1, 3, 7, 8, 13, 100] {
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 50.0) * 0.5).collect();
        relu_simd(&mut data);
        for (i, &v) in data.iter().enumerate() {
            let expected = ((i as f32 - 50.0) * 0.5).max(0.0);
            assert!(
                (v - expected).abs() < 1e-6,
                "relu n={n}[{i}]: {v} != {expected}"
            );
        }
    }
}

#[test]
fn test_add_simd_correctness() {
    let mut a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| 10.0 - i as f32).collect();
    add_simd(&mut a, &b);
    for (i, &ai) in a.iter().enumerate() {
        assert!((ai - 10.0).abs() < 1e-6, "add[{i}]: {}", ai);
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
            assert!(
                (0.0..=1.0).contains(&v),
                "sigmoid n={n}[{i}] out of [0,1]: {v}"
            );
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
    for &p in &pulses {
        assert!((p - 1.0).abs() < 1e-6, "TACT: {p}");
    }

    // All below threshold
    let ph1 = vec![-1.0; n];
    let ph2 = vec![-1.0; n];
    let ph3 = vec![-1.0; n];
    t3_tact_generator(&mut pulses, &ph1, &ph2, &ph3, 0.5);
    for &p in &pulses {
        assert!(p.abs() < 1e-6, "TACT off: {p}");
    }
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
        pillars::kernels::phase::kuramoto_prepare(&mut phases, &mut cos_out, &mut sin_out);
        for i in 0..n {
            let phi = ((i as f32) * 0.5)
                - (std::f32::consts::TAU * (((i as f32) * 0.5) / std::f32::consts::TAU).round());
            let expected_cos = phi.cos();
            let expected_sin = phi.sin();
            assert!(
                (cos_out[i] - expected_cos).abs() < 1e-4,
                "cos n={n}[{i}]: {:.4} != {:.4}",
                cos_out[i],
                expected_cos
            );
            assert!(
                (sin_out[i] - expected_sin).abs() < 1e-4,
                "sin n={n}[{i}]: {:.4} != {:.4}",
                sin_out[i],
                expected_sin
            );
        }
    }
}

#[test]
fn test_kuramoto_update_correctness() {
    let n = 8;
    let mut phases = vec![0.0; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.0; n];
    pillars::kernels::phase::kuramoto_update(&mut phases, &omegas, &coupling, 0.01);
    for (i, &p) in phases.iter().enumerate() {
        assert!((p - 0.01).abs() < 1e-6, "kuramoto_update[{i}]");
    }
}

#[test]
fn test_kuramoto_update_viscous_delegation() {
    let n = 16;
    let mut phases = vec![0.0; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.0; n];
    kuramoto_update_viscous(&mut phases, &omegas, &coupling, 0.0, 0.01);
    for (i, &p) in phases.iter().enumerate() {
        assert!((p - 0.01).abs() < 1e-6, "viscous eta=0 [{i}]");
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
        assert!(
            diff < 0.5,
            "viscous not synchronized: diff={diff}, p={p}, mean={mean}"
        );
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
    t3_update(
        &mut phases,
        &mut amps,
        &omegas,
        &coupling,
        &drives,
        0.01,
        0.1,
        1.0,
        1.0,
    );
    assert!(
        phases.iter().any(|&p| p.abs() > 0.001),
        "t3_update should advance phases"
    );
    for &r in &amps {
        assert!((0.001..=10.0).contains(&r), "t3 amp out of range: {r}");
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
        t3_update(
            &mut phases,
            &mut amps,
            &omegas,
            &coupling,
            &drives,
            0.01,
            0.1,
            1.0,
            1.0,
        );
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
        let pred: f32 = weights
            .iter()
            .zip(states.iter())
            .map(|(w, s)| w * s)
            .sum::<f32>()
            + bias[0];
        let err = 1.0 - pred;
        nlms_train(&mut weights, &mut bias, &states, err, 0.1, 1e-8);
    }
    assert!(
        weights.iter().any(|&w| w.abs() > 0.01),
        "NLMS weights should adapt"
    );
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
        let s: f32 = states
            .iter()
            .zip(cov.iter())
            .map(|(x, c)| c * x * x)
            .sum::<f32>()
            + 0.1;
        kalman_update(&mut weights, &mut cov, &states, err, s);
    }
    assert!(
        weights.iter().any(|&w| w.abs() > 0.001),
        "Kalman weights should adapt"
    );
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
    assert!(
        (coh - 1.0).abs() < 1e-4,
        "Perfect sync coherence = {coh}, expected 1.0"
    );
}

#[test]
fn test_phase_coherence_random() {
    let n = 64;
    let phases: Vec<f32> = (0..n)
        .map(|i| (i as f32) * std::f32::consts::TAU / n as f32)
        .collect();
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
    for &v in &out {
        assert!((v - 1.0).abs() < 1e-4, "interference: {v}");
    }
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
    assert!(
        (energies.iter().sum::<f32>() - n as f32).abs() < 1e-4,
        "band_energy sum mismatch"
    );
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
        assert!(
            (sin_out[i] - 0.1f32.sin()).abs() < 1e-4,
            "lazy_sincos[{i}] sin"
        );
        assert!(
            (cos_out[i] - 0.1f32.cos()).abs() < 1e-4,
            "lazy_sincos[{i}] cos"
        );
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

    izhikevich_step(
        &mut v,
        &mut u,
        &mut trace,
        &a,
        &b,
        &mut I,
        1.0,
        0.5,
        &mut spikes,
        &c,
        &d,
        &mut refractory,
        3.0,
        None,
    );

    if spikes.iter().any(|&s| s > 0) {
        for i in 0..n {
            if spikes[i] > 0 {
                assert!(refractory[i] > 0.0, "refractory not set after spike");
            }
        }
        izhikevich_step(
            &mut v,
            &mut u,
            &mut trace,
            &a,
            &b,
            &mut I,
            1.0,
            0.5,
            &mut spikes,
            &c,
            &d,
            &mut refractory,
            3.0,
            None,
        );
        assert_eq!(spikes.iter().sum::<i32>(), 0, "Spiked during refractory!");
    }
}

// ═══════════════════════════════════════════════════════
//  7. LIF
// ═══════════════════════════════════════════════════════

fn make_lif_params(
    n: usize,
    dt: f32,
    tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
) -> Vec<f32> {
    let mut p = vec![0.0; n * 5];
    for i in 0..n {
        p[i] = dt;
        p[n + i] = tau_m;
        p[2 * n + i] = v_rest;
        p[3 * n + i] = v_thresh;
        p[4 * n + i] = v_reset;
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
            lif_step(
                &mut v,
                &mut I,
                &mut spikes,
                &params,
                &mut refractory,
                2.0,
                None,
            );
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

    for _ in 0..200 {
        lif_step(
            &mut v,
            &mut Ii,
            &mut spikes,
            &params,
            &mut refractory,
            0.0,
            None,
        );
    }
    for &vi in &v {
        assert!((vi - (-55.0)).abs() < 0.1, "LIF steady state: {:.4}", vi);
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
    for &w in &weights {
        assert!(w <= 1.0, "stdp clamp max: {w}");
    }

    let mut weights = vec![-5.0; n];
    stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, -0.5, 1.0);
    for &w in &weights {
        assert!(w >= -0.5, "stdp clamp min: {w}");
    }
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
    stp_step(
        &mut x,
        &mut u,
        &pre_spikes,
        &tau_d,
        &tau_f,
        &u0,
        &a,
        &mut i_out,
        1.0,
    );
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
    stp_step(
        &mut x,
        &mut u,
        &pre_spikes,
        &tau_d,
        &tau_f,
        &u0,
        &a,
        &mut i_out,
        1.0,
    );
    assert!(x[0] > 0.5, "x should recover toward 1: {}", x[0]);
    assert!(
        (i_out[0] - 0.0).abs() < 1e-6,
        "i_out should stay 0: {}",
        i_out[0]
    );
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
    dopamine_traces_step(
        &mut trace_pre,
        &mut trace_post,
        &mut eligibility,
        &pre_spikes,
        &post_spikes,
        1.0,
        20.0,
        20.0,
        200.0,
        0.01,
        0.01,
    );
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
    let meta = build_channel_meta(&channels);
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

    let g_max_per_neuron = vec![1.0; meta.len() * n];
    hh_poly_step(
        &mut v,
        &mut gates,
        &ii,
        &g_leak,
        &e_leak,
        &c_m,
        &meta,
        &g_max_per_neuron,
        0.01,
        6.3,
        None,
    );
    assert_no_nan(&v, "hh_poly v");
    assert_no_nan(&gates, "hh_poly gates");
}

#[test]
fn test_hh_poly_step_no_spike_small_current() {
    let n = 8;
    let channels = standard_channels();
    let meta = build_channel_meta(&channels);
    let n_gates: usize = meta.iter().map(|ch| ch.n_gates).sum();

    let mut v = vec![-65.0; n];
    let mut gates = vec![0.0; n_gates * n];
    let ii = vec![0.0; n];
    let g_leak = vec![0.3; n];
    let e_leak = vec![-54.0; n];
    let c_m = vec![1.0; n];
    let g_max_per_neuron = vec![1.0; meta.len() * n];

    let ss = steady_state_gates(-65.0, &channels);
    for gi in 0..n_gates {
        for ni in 0..n {
            gates[gi * n + ni] = ss[gi];
        }
    }

    for _ in 0..100 {
        hh_poly_step(
            &mut v,
            &mut gates,
            &ii,
            &g_leak,
            &e_leak,
            &c_m,
            &meta,
            &g_max_per_neuron,
            0.01,
            6.3,
            None,
        );
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
        params[2 * n + i] = 0.7;
        params[3 * n + i] = 0.8;
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
        params[2 * n + j] = 10.0;
        params[3 * n + j] = 12.0; // strong excitation
        params[4 * n + j] = 8.0;
        params[5 * n + j] = 8.0;
        params[6 * n + j] = 2.0;
        params[7 * n + j] = 1.0;
    }
    let e0 = e.clone();
    for _ in 0..200 {
        wc_step(&mut e, &mut i, &i_e, &i_i, &mut se, &mut si, &params);
    }
    let changed = e
        .iter()
        .zip(e0.iter())
        .any(|(e_new, e_old)| (e_new - e_old).abs() > 0.01);
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
            params[2 * n + i] = -65.0;
            params[3 * n + i] = -50.0;
            params[4 * n + i] = -65.0;
            params[5 * n + i] = 2.0;
            params[6 * n + i] = 100.0;
            params[7 * n + i] = 0.0;
            params[8 * n + i] = 10.0;
        }
        let mut refractory = vec![0.0; n];
        let mut total = 0i32;
        for _ in 0..200 {
            adex_step(
                &mut v,
                &mut u,
                &mut I,
                &mut spikes,
                &params,
                &mut refractory,
                2.0,
                None,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx n={n}: no spikes");
    }
}

#[test]
fn test_eprop_adex_step_multi_smoke() {
    for &n in &[1, 7, 8, 13] {
        let n_pre = 3;
        let mut v = vec![-65.0; n];
        let mut w = vec![-13.0; n];
        let mut psi = vec![0.0; n];
        let mut e = vec![0.0; n_pre * n]; // [n_pre × n] column-major
        let z_pre = vec![1i32; n_pre];
        let i_ext = vec![25.0; n];
        let weights = vec![1.0; n_pre * n];
        let mut spikes = vec![0i32; n];
        let mut total = 0usize;
        for _ in 0..200 {
            total += eprop_adex_step_multi(
                &mut v,
                &mut w,
                &mut psi,
                &mut e,
                &z_pre,
                &i_ext,
                &weights,
                n_pre,
                0.5,   // dt
                10.0,  // tau_m
                100.0, // tau_w
                -65.0, // v_rest
                -50.0, // v_thresh
                -65.0, // v_reset
                2.0,   // delta_t
                0.0,   // a
                10.0,  // b
                20.0,  // tau_e
                2.0,   // tau_sg
                &mut spikes,
                None,
            );
        }
        assert!(total > 0, "eprop AdEx n={n}: no spikes with I=25");
        assert_no_nan(&v, "eprop adex v");
        assert_no_nan(&w, "eprop adex w");
        assert_no_nan(&psi, "eprop adex psi");
        assert_no_nan(&e, "eprop adex e");
    }
}

// ═══════════════════════════════════════════════════════
//  15. GCO (Graph-Coupled Oscillator)
// ═══════════════════════════════════════════════════════

#[test]
fn test_gco_step_basic() {
    let n = 8;
    let mut v = vec![0.0; n];
    let mut u = vec![0.0; n];
    let h_real = vec![1.0; n];
    let h_imag = vec![0.0; n];
    let i_ext = vec![0.1; n];
    let speed = vec![0.1; n];

    for _ in 0..100 {
        pillars::kernels::phase::gco_step(&mut v, &mut u, &h_real, &h_imag, &i_ext, &speed, 0.5);
    }
    assert_no_nan(&h_real, "gco real");
    assert_no_nan(&h_imag, "gco imag");
    assert_no_nan(&speed, "gco speed");
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
    let _empty_spikes = [0i32; 0];
    // n=0 izhikevich isn't tested separately (handled by scalar tail logic)

    safety_clamp(&mut empty1, -1.0, 1.0);
    relu_simd(&mut empty2);
    add_simd(&mut empty3, &empty4);
    sigmoid_gate(&mut empty5, &empty4, 1.0, 0.0);

    let mut empty_p1: Vec<f32> = vec![];
    let mut empty_cos: Vec<f32> = vec![];
    let mut empty_sin: Vec<f32> = vec![];
    pillars::kernels::phase::kuramoto_prepare(&mut empty_p1, &mut empty_cos, &mut empty_sin);
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
    for _ in 0..100 {
        izhikevich_step(
            &mut v,
            &mut u,
            &mut trace,
            &a,
            &b,
            &mut I,
            1.0,
            0.5,
            &mut spikes,
            &c,
            &d,
            &mut refractory,
            3.0,
            None,
        );
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
    let mut refractory = vec![0.0; n];
    izhikevich_step(
        &mut v,
        &mut u,
        &mut trace,
        &a,
        &b,
        &mut I,
        1.0,
        0.5,
        &mut spikes,
        &c,
        &d,
        &mut refractory,
        3.0,
        None,
    );
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
        izhikevich_step(
            &mut v,
            &mut u,
            &mut trace,
            &a,
            &b,
            &mut I,
            0.5,
            0.5,
            &mut spikes,
            &c,
            &d,
            &mut refractory,
            3.0,
            None,
        );
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
        lif_step(
            &mut v,
            &mut I,
            &mut spikes,
            &params,
            &mut refractory,
            2.0,
            None,
        );
    }
    assert_no_nan(&v, "lif long-run v");
    for &vv in &v {
        assert!(vv > -200.0 && vv < 100.0, "lif long-run v={vv}");
    }
}

#[test]
fn test_morris_lecar_simd_step() {
    let mut v = vec![-60.0; 8];
    let mut w = vec![0.0; 8];
    let i_in = [50.0; 8];
    let mut spikes = vec![0; 8];
    let mut params = vec![0.0f32; 8 * 16];
    for c in 0..8 {
        params[c * 16] = 20.0; // C
        params[c * 16 + 1] = 2.0; // g_L
        params[c * 16 + 2] = -60.0; // E_L
        params[c * 16 + 3] = 4.0; // g_Ca
        params[c * 16 + 4] = 120.0; // E_Ca
        params[c * 16 + 5] = 8.0; // g_K
        params[c * 16 + 6] = -84.0; // E_K
        params[c * 16 + 7] = -1.2; // V1
        params[c * 16 + 8] = 18.0; // V2
        params[c * 16 + 9] = 12.0; // V3
        params[c * 16 + 10] = 17.4; // V4
        params[c * 16 + 11] = 0.066; // phi
        params[c * 16 + 12] = 20.0; // v_thresh
    }

    for _ in 0..100 {
        unsafe {
            // MorrisLecarModel 14-col repacking из interleaved 16-col (транш 3e).
            let n = v.len();
            let mut p = vec![0.0f32; n * 14];
            let mut was_below = vec![0.0f32; n];
            for i in 0..n {
                let base = i * 16;
                p[i] = 0.1;
                p[n + i] = 1.0 / params[base].max(1e-6);
                for c in 1..12 {
                    p[(c + 1) * n + i] = params[base + c];
                }
                p[13 * n + i] = params[base + 12];
                was_below[i] = if v[i] < params[base + 12] { 1.0 } else { 0.0 };
            }
            pillars::core::runner::run_neuron::<pillars::zoo::neurons::MorrisLecarModel>(
                v.as_mut_ptr(),
                w.as_mut_ptr(),
                was_below.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                i_in.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    }
    assert_no_nan(&v, "morris_lecar v");
    assert_no_nan(&w, "morris_lecar w");
}

#[test]
fn test_nmda_mg_block_simd_step() {
    let v = [-70.0, -40.0, 0.0, 20.0, -70.0, -40.0, 0.0, 20.0];
    let g_act = [1.0; 8];
    let mut i_out = vec![0.0; 8];
    let mut params = [0.0; 8 * 4];
    for c in 0..8 {
        params[c * 4] = 1.0; // g_nmda
        params[c * 4 + 1] = 0.0; // E_syn
        params[c * 4 + 2] = 1.0; // mg_conc 1mM
    }

    unsafe {
        pillars::zoo::synapses::nmda_mg_block_step(
            v.as_ptr(),
            g_act.as_ptr(),
            i_out.as_mut_ptr(),
            params.as_ptr(),
            v.len(),
        );
    }
    assert_no_nan(&i_out, "nmda i_out");
    // At -70mV (index 0), Mg2+ block is strong (current is small). At +20mV (index 3), Mg2+ unblocks (current is large).
    assert!(i_out[0].abs() < i_out[3].abs());
}

#[test]
fn test_tsodyks_markram_stp_simd_step() {
    let mut x = vec![1.0; 8];
    let mut u = vec![0.1; 8];
    let spikes = [1, 0, 0, 0, 1, 0, 0, 0];
    let mut w_eff = vec![0.0; 8];
    let mut params = [0.0; 8 * 4];
    for c in 0..8 {
        params[c * 4] = 0.2; // U
        params[c * 4 + 1] = 200.0; // tau_d
        params[c * 4 + 2] = 50.0; // tau_f
        params[c * 4 + 3] = 1.0; // W_base
    }

    unsafe {
        pillars::zoo::synapses::tsodyks_markram_step(
            x.as_mut_ptr(),
            u.as_mut_ptr(),
            spikes.as_ptr(),
            w_eff.as_mut_ptr(),
            params.as_ptr(),
            x.len(),
            0.1,
        );
    }
    assert_no_nan(&x, "tsodyks_markram x");
    assert_no_nan(&u, "tsodyks_markram u");
    assert_no_nan(&w_eff, "tsodyks_markram w_eff");
}

#[test]
fn test_analytical_accuracy_nmda_mg_block() {
    let v_vals = [-100.0, -70.0, -50.0, -20.0, 0.0, 20.0, 40.0, 60.0];
    let g_act = [1.0; 8];
    let mut i_simd = vec![0.0; 8];
    let mut params = [0.0; 8 * 4];
    for c in 0..8 {
        params[c * 4] = 1.0; // g_nmda
        params[c * 4 + 1] = 0.0; // E_syn
        params[c * 4 + 2] = 1.0; // mg_conc 1mM
    }

    unsafe {
        pillars::zoo::synapses::nmda_mg_block_step(
            v_vals.as_ptr(),
            g_act.as_ptr(),
            i_simd.as_mut_ptr(),
            params.as_ptr(),
            v_vals.len(),
        );
    }

    for i in 0..8 {
        let v64 = v_vals[i] as f64;
        let block_exact = 1.0 / (1.0 + (1.0 / 3.57) * (-0.062 * v64).exp());
        let i_exact = 1.0 * block_exact * (v64 - 0.0);
        let diff = (i_simd[i] as f64 - i_exact).abs();
        assert!(
            diff < 1e-2,
            "NMDA precision error at V={}: SIMD={}, exact={}, diff={}",
            v_vals[i],
            i_simd[i],
            i_exact,
            diff
        );
    }
}

#[test]
fn test_analytical_accuracy_tsodyks_markram_stp() {
    let mut x_simd = vec![1.0; 8];
    let mut u_simd = vec![0.1; 8];
    let spikes = [1, 0, 1, 0, 0, 1, 0, 0];
    let mut w_simd = vec![0.0; 8];
    let mut params = [0.0; 8 * 4];
    let U = 0.2f64;
    let tau_d = 200.0f64;
    let tau_f = 50.0f64;
    let W_base = 1.5f64;
    let dt = 0.5f32;

    for c in 0..8 {
        params[c * 4] = U as f32;
        params[c * 4 + 1] = tau_d as f32;
        params[c * 4 + 2] = tau_f as f32;
        params[c * 4 + 3] = W_base as f32;
    }

    let mut x64 = [1.0f64; 8];
    let mut u64 = [0.1f64; 8];

    for step in 0..20 {
        unsafe {
            pillars::zoo::synapses::tsodyks_markram_step(
                x_simd.as_mut_ptr(),
                u_simd.as_mut_ptr(),
                spikes.as_ptr(),
                w_simd.as_mut_ptr(),
                params.as_ptr(),
                x_simd.len(),
                dt,
            );
        }

        for i in 0..8 {
            // Точное аналитическое решение dx/dt=(1-x)/τ — Exponential Euler
            // (канон pillars с волны 8 транша 1a; легаси использовал Forward
            // Euler, отклонение ~dt/2τ).
            x64[i] += (1.0 - x64[i]) * (1.0 - (-dt as f64 / tau_d).exp());
            u64[i] += (U - u64[i]) * (1.0 - (-dt as f64 / tau_f).exp());
            if spikes[i] != 0 {
                u64[i] += U * (1.0 - u64[i]);
                x64[i] -= u64[i] * x64[i];
            }
            let w_exact = W_base * u64[i] * x64[i];
            let diff_w = (w_simd[i] as f64 - w_exact).abs();
            assert!(
                diff_w < 1e-4,
                "STP precision error at step {step}, synapse {i}: SIMD={}, exact={}, diff={}",
                w_simd[i],
                w_exact,
                diff_w
            );
        }
    }
}
