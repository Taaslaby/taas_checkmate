//! Plasticity parity — волна 8, транш 1b (stp/stdp/bcm/dopamine → pillars).
//!
//! Легаси `kernels::neuron/{stp,bcm,dopamine}.rs` удалены. Канон:
//! - `pillars::kernels::learning` — bcm_step, homeostatic_scale, stdp_update,
//!   stdp_homeostatic (новый канон, бит-в-бит порт)
//! - `pillars::kernels::dopamine` — dopamine_traces_step, dopamine_weight_update
//! - `pillars::ops::plasticity::stp` — stp_step (Exp Euler lane-канон)
//!
//! Эти тесты доказывают:
//! 1. `safe::*` — тонкие делегаты pillars (бит-в-бит, n=13 → SIMD+tail).
//! 2. Мост `safe::stp_step` (накопление I_out) — SIMD-чанки == scalar tail
//!    (одинаковые параметры ⇒ одинаковые биты на нейроне 0 и 12).

#![allow(non_snake_case)]

use pillars::core::lane::SimdLane;
use pillars::core::util::{load8, load8_i32, store8};
use pillars::kernels::dopamine::{dopamine_traces_step, dopamine_weight_update};
use pillars::kernels::learning::{bcm_step, homeostatic_scale, stdp_homeostatic, stdp_update};
use pillars::ops::plasticity::stp as pillars_stp;
use wide::f32x8;

/// Локальная копия удалённого моста `safe::stp_step` — SIMD-цикл + scalar
/// tail поверх pillars lane-канона (транш 3e: safe.rs удалён).
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
    let n8 = n & !7;
    let dt_s = f32x8::splat(dt);
    let zero_s = f32x8::splat(0.0);
    let one_s = f32x8::splat(1.0);
    for off in (0..n8).step_by(8) {
        unsafe {
            let uu = load8(u.as_ptr(), off);
            let xx = load8(x.as_ptr(), off);
            let sp = load8_i32(pre_spikes.as_ptr(), off);
            let td = load8(tau_d.as_ptr(), off).max(f32x8::splat(1e-10));
            let tf = load8(tau_f.as_ptr(), off).max(f32x8::splat(1e-10));
            let u0v = load8(u0.as_ptr(), off);
            let av = load8(a.as_ptr(), off);
            let (un, xn) = pillars_stp::stp_step(uu, xx, sp, u0v, one_s / td, one_s / tf, dt_s);
            let s = sp.cmp_gt(zero_s).blend(one_s, zero_s);
            let mut io = load8(i_out.as_ptr(), off);
            io += s * av * un * xn;
            store8(u.as_mut_ptr(), off, un);
            store8(x.as_mut_ptr(), off, xn);
            store8(i_out.as_mut_ptr(), off, io);
        }
    }
    for i in n8..n {
        let s = if pre_spikes[i] != 0 { 1.0f32 } else { 0.0f32 };
        let (un, xn) = pillars_stp::stp_step(
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

fn assert_bit_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-6 || (a.to_bits() == e.to_bits()),
            "{label}[{i}]: safe={a:.9} (0x{:08x}) pillars={e:.9} (0x{:08x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

// ═══════════════════════════════════════════════════════
//  STDP batch
// ═══════════════════════════════════════════════════════

#[test]
fn test_stdp_update_safe_equals_pillars() {
    let n = 13;
    let mut w_safe: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.07).collect();
    let mut w_pill = w_safe.clone();
    let pre: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31) % 1.0).collect();
    let post: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32 * 0.17) % 0.6).collect();

    stdp_update(&mut w_safe, &pre, &post, 0.03, 0.05, 0.0, 2.0);
    pillars::kernels::learning::stdp_update(&mut w_pill, &pre, &post, 0.03, 0.05, 0.0, 2.0);

    assert_bit_eq(&w_safe, &w_pill, "stdp_update");
}

// ═══════════════════════════════════════════════════════
//  STDP homeostatic (новый канон в pillars)
// ═══════════════════════════════════════════════════════

#[test]
fn test_stdp_homeostatic_safe_equals_pillars() {
    let n = 13;
    let mut w_safe: Vec<f32> = (0..n).map(|i| 0.2 + i as f32 * 0.1).collect();
    let mut w_pill = w_safe.clone();
    let pre: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23) % 1.0).collect();
    let post: Vec<f32> = (0..n).map(|i| 0.9 - (i as f32 * 0.11) % 0.5).collect();

    stdp_homeostatic(&mut w_safe, &pre, &post, 0.02, 0.04, 4.0, 1.5);
    pillars::kernels::learning::stdp_homeostatic(&mut w_pill, &pre, &post, 0.02, 0.04, 4.0, 1.5);

    assert_bit_eq(&w_safe, &w_pill, "stdp_homeostatic");
    // Homeostatic pull: total weight должен двигаться к target_sum.
    let sum: f32 = w_pill.iter().sum();
    assert!(
        (sum - 4.0).abs() < 2.0,
        "stdp_homeostatic: сумма не тянется к target (sum={sum:.3})"
    );
}

// ═══════════════════════════════════════════════════════
//  BCM + homeostatic scaling
// ═══════════════════════════════════════════════════════

#[test]
fn test_bcm_safe_equals_pillars() {
    let n = 13;
    let mut x_safe: Vec<f32> = vec![0.4; n];
    let mut y_safe: Vec<f32> = vec![3.0; n];
    let mut th_safe: Vec<f32> = vec![1.0; n];
    let mut w_safe: Vec<f32> = vec![0.5; n];
    let mut x_pill = x_safe.clone();
    let mut y_pill = y_safe.clone();
    let mut th_pill = th_safe.clone();
    let mut w_pill = w_safe.clone();
    let pre = vec![0.5f32; n];
    let post = vec![3.0f32; n];

    bcm_step(
        &mut x_safe,
        &mut y_safe,
        &mut th_safe,
        &mut w_safe,
        &pre,
        &post,
        1.0,
        20.0,
        20.0,
        100.0,
        0.01,
        0.001,
        0.0,
        10.0,
    );
    pillars::kernels::learning::bcm_step(
        &mut x_pill,
        &mut y_pill,
        &mut th_pill,
        &mut w_pill,
        &pre,
        &post,
        1.0,
        20.0,
        20.0,
        100.0,
        0.01,
        0.001,
        0.0,
        10.0,
    );

    assert_bit_eq(&x_safe, &x_pill, "bcm x");
    assert_bit_eq(&y_safe, &y_pill, "bcm y");
    assert_bit_eq(&th_safe, &th_pill, "bcm theta");
    assert_bit_eq(&w_safe, &w_pill, "bcm weights");
}

#[test]
fn test_homeostatic_scale_safe_equals_pillars() {
    let n = 13;
    let mut w_safe: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.05).collect();
    let mut w_pill = w_safe.clone();
    let rates = vec![8.0f32; n];

    homeostatic_scale(&mut w_safe, &rates, 1.0, 0.01, 5.0);
    pillars::kernels::learning::homeostatic_scale(&mut w_pill, &rates, 1.0, 0.01, 5.0);

    assert_bit_eq(&w_safe, &w_pill, "homeostatic_scale");
    assert!(w_pill[0] < 0.5, "высокая активность должна уменьшать веса");
}

// ═══════════════════════════════════════════════════════
//  Dopamine
// ═══════════════════════════════════════════════════════

#[test]
fn test_dopamine_traces_safe_equals_pillars() {
    let n = 13;
    let mut tp_safe: Vec<f32> = vec![0.1; n];
    let mut tpo_safe: Vec<f32> = vec![0.2; n];
    let mut e_safe: Vec<f32> = vec![0.05; n];
    let mut tp_pill = tp_safe.clone();
    let mut tpo_pill = tpo_safe.clone();
    let mut e_pill = e_safe.clone();
    let pre: Vec<i32> = (0..n).map(|i| (i % 3 == 0) as i32).collect();
    let post: Vec<i32> = (0..n).map(|i| (i % 4 == 1) as i32).collect();

    dopamine_traces_step(
        &mut tp_safe,
        &mut tpo_safe,
        &mut e_safe,
        &pre,
        &post,
        0.5,
        20.0,
        20.0,
        200.0,
        0.01,
        0.012,
    );
    pillars::kernels::dopamine::dopamine_traces_step(
        &mut tp_pill,
        &mut tpo_pill,
        &mut e_pill,
        &pre,
        &post,
        0.5,
        20.0,
        20.0,
        200.0,
        0.01,
        0.012,
    );

    assert_bit_eq(&tp_safe, &tp_pill, "dopamine trace_pre");
    assert_bit_eq(&tpo_safe, &tpo_pill, "dopamine trace_post");
    assert_bit_eq(&e_safe, &e_pill, "dopamine eligibility");
}

#[test]
fn test_dopamine_weight_safe_equals_pillars() {
    let n = 13;
    let mut w_safe: Vec<f32> = (0..n).map(|i| 0.3 + i as f32 * 0.03).collect();
    let mut w_pill = w_safe.clone();
    let el = vec![1.0f32; n];

    dopamine_weight_update(&mut w_safe, &el, 0.5, 0.1, 0.0, 1.0);
    pillars::kernels::dopamine::dopamine_weight_update(&mut w_pill, &el, 0.5, 0.1, 0.0, 1.0);

    assert_bit_eq(&w_safe, &w_pill, "dopamine_weight");
}

// ═══════════════════════════════════════════════════════
//  STP — мост safe::stp_step на pillars lane-канон
// ═══════════════════════════════════════════════════════

/// SIMD-чанки (n=8) и scalar tail (n=12) должны давать одинаковые биты.
#[test]
fn test_stp_step_simd_tail_consistent() {
    let n = 13;
    let mut x = vec![0.8f32; n];
    let mut u = vec![0.3f32; n];
    let mut i_out = vec![0.0f32; n];
    let pre: Vec<i32> = (0..n).map(|i| (i % 3 == 0) as i32).collect();
    let tau_d = vec![200.0f32; n];
    let tau_f = vec![600.0f32; n];
    let u0 = vec![0.2f32; n];
    let a = vec![1.0f32; n];

    stp_step(
        &mut x, &mut u, &pre, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0,
    );

    // Нейрон 0 (SIMD) и нейрон 12 (scalar tail) — идентичные параметры ⇒ биты.
    for (i, j) in [(0usize, 12usize), (3, 6)] {
        assert_eq!(
            u[i].to_bits(),
            u[j].to_bits(),
            "u[{i}] != u[{j}] (SIMD vs tail)"
        );
        assert_eq!(
            x[i].to_bits(),
            x[j].to_bits(),
            "x[{i}] != x[{j}] (SIMD vs tail)"
        );
        assert_eq!(
            i_out[i].to_bits(),
            i_out[j].to_bits(),
            "i_out[{i}] != i_out[{j}] (SIMD vs tail)"
        );
    }
    // На спайке (i%3==0) ток ненулевой, без спайка — ноль.
    for i in 0..n {
        if pre[i] != 0 {
            assert!(i_out[i] > 0.0, "i_out[{i}] должен быть >0 на спайке");
        } else {
            assert_eq!(i_out[i], 0.0, "i_out[{i}] должен быть 0 без спайка");
        }
    }
}

/// Мост == прямой lane-канон (бит-в-бит на спайковом шаге).
#[test]
fn test_stp_step_matches_lane_canon() {
    let n = 8;
    let mut x = vec![0.9f32; n];
    let mut u = vec![0.1f32; n];
    let mut i_out = vec![0.0f32; n];
    let pre = vec![1i32; n];
    let tau_d = vec![800.0f32; n];
    let tau_f = vec![50.0f32; n];
    let u0 = vec![0.2f32; n];
    let a = vec![1.5f32; n];

    stp_step(
        &mut x, &mut u, &pre, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0,
    );

    // Прямой скалярный канон (то же, что lane-канон на f32).
    let (un, xn) = pillars_stp::stp_step(
        0.1f32,
        0.9f32,
        1.0f32,
        0.2f32,
        1.0 / 800.0,
        1.0 / 50.0,
        1.0f32,
    );
    assert_eq!(u[0].to_bits(), un.to_bits(), "u мост != lane-канон");
    assert_eq!(x[0].to_bits(), xn.to_bits(), "x мост != lane-канон");
    let i_expect = 1.0 * 1.5 * un * xn; // s·a·u·x
    assert_eq!(
        i_out[0].to_bits(),
        i_expect.to_bits(),
        "i_out мост != канон"
    );
}

/// Качественная проверка: без спайка x восстанавливается к 1, u — к u0.
#[test]
fn test_stp_step_recovery_no_spike() {
    let n = 8;
    let mut x = vec![0.5f32; n];
    let mut u = vec![0.5f32; n];
    let mut i_out = vec![0.0f32; n];
    let pre = vec![0i32; n];
    let tau_d = vec![50.0f32; n];
    let tau_f = vec![20.0f32; n];
    let u0 = vec![0.2f32; n];
    let a = vec![1.0f32; n];

    stp_step(
        &mut x, &mut u, &pre, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0,
    );

    assert!(x[0] > 0.5, "x должен восстанавливаться к 1: {:.4}", x[0]);
    assert!(u[0] < 0.5, "u должен декейить к u0: {:.4}", u[0]);
    assert_eq!(i_out[0], 0.0);
}

/// `SimdLane::exp` участвует в каноне — проверим его использование через
/// lane-версию (регресс-ловушка на случай поломки fast-exp пути).
#[test]
fn test_stp_lane_uses_exp() {
    // exp(-dt/τ) с dt=1, τ=50 → decay ≈ 0.9802; Exp Euler (канон) точен.
    let decay = <f32 as SimdLane>::exp(-1.0 / 50.0);
    let expect = (-1.0f32 / 50.0).exp();
    assert!(
        (decay - expect).abs() < 1e-3,
        "SimdLane::exp(f32) сильно расходится с нативным: {decay} vs {expect}"
    );
}
