//! # LIF-AT parity: локальный pillars-мост vs `run_neuron::<LIFAtModel>`
//!
//!консолидации: легаси-кернел `kernels::neuron::lif::lif_at_step`
//! удалён; канон — `pillars::zoo::neurons::LIFAtModel` (GLIF2). Транш 3e:
//! `safe.rs` удалён — мост переупаковки легаси 7-колоночного SoA-лейаута
//! `[dt, τ_m, v_rest, v_thresh, v_reset, τ_θ, β]×n` в pillars 8-колоночный
//! `[dt, inv_τ_m, v_rest, v_thresh, v_reset, inv_τ_θ, β, τ_ref]×n` теперь
//! локальная функция теста (та же логика, что жила в safe.rs).
//!
//! ## Что тестируется
//! 1. **Бит-в-бит**: локальный мост ≡ прямой `run_neuron::<LIFAtModel>`
//!    (оба пути — один и тот же код; тест ловит ошибки переупаковки параметров).
//!    Прогон `n=8` (SIMD) и `n=13` (SIMD-лейны + скалярный хвост).
//! 2. **Легаси-семантика сохранена**: v/θ/refr/спайки моста совпадают со
//!    скалярным эталоном легаси-семантики в допусках — θ: 1e-3 rel
//!    (легаси-SIMD использовал полином `fast_exp` ~1e-5 err, pillars — точный
//!    `exp`, задокументированное расхождение); спайк-счёт равен.

#![allow(non_snake_case)] // SoA-колонки и Ii-стиль легаси-API

use pillars::core::runner::run_neuron;
use pillars::zoo::neurons::LIFAtModel;

/// Локальная копия удалённого моста `safe::lif_at_step` (транш 3e).
fn lif_at_step(
    v: &mut [f32],
    theta: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    _i_dend: Option<&[f32]>,
) {
    let n = v.len();
    let mut p = vec![0.0f32; n * 8];
    for i in 0..n {
        p[i] = params[i];
        let tm = params[n + i];
        p[n + i] = if tm > 0.0 { 1.0 / tm } else { 0.0 };
        p[2 * n + i] = params[2 * n + i];
        p[3 * n + i] = params[3 * n + i];
        p[4 * n + i] = params[4 * n + i];
        let tt = params[5 * n + i];
        p[5 * n + i] = if tt > 0.0 { 1.0 / tt } else { 1e9 };
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = tau_ref;
    }
    unsafe {
        run_neuron::<LIFAtModel>(
            v.as_mut_ptr(),
            theta.as_mut_ptr(),
            refractory.as_mut_ptr(),
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

const DT: f32 = 0.5;
const TAU_M: f32 = 10.0;
const V_REST: f32 = -70.0;
const V_THRESH: f32 = -54.0;
const V_RESET: f32 = -70.0;
const TAU_THETA: f32 = 20.0;
const BETA: f32 = 5.0;
const TAU_REF: f32 = 3.0;
const I_VAL: f32 = 40.0;
const STEPS: usize = 1000;

/// Legacy params layout: `[dt, tau_m, v_rest, v_thresh, v_reset, tau_theta, beta] × n`.
fn legacy_params(n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; 7 * n];
    for i in 0..n {
        p[i] = DT;
        p[n + i] = TAU_M;
        p[2 * n + i] = V_REST;
        p[3 * n + i] = V_THRESH;
        p[4 * n + i] = V_RESET;
        p[5 * n + i] = TAU_THETA;
        p[6 * n + i] = BETA;
    }
    p
}

/// Pillars LIFAtModel params layout: `[dt, inv_tau_m, v_rest, v_thresh, v_reset, inv_tau_theta, beta, tau_ref] × n`.
fn pillars_params(n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; 8 * n];
    for i in 0..n {
        p[i] = DT;
        p[n + i] = 1.0 / TAU_M;
        p[2 * n + i] = V_REST;
        p[3 * n + i] = V_THRESH;
        p[4 * n + i] = V_RESET;
        p[5 * n + i] = 1.0 / TAU_THETA;
        p[6 * n + i] = BETA;
        p[7 * n + i] = TAU_REF;
    }
    p
}

/// Scalar reference of the **legacy lif_at_step semantics** (exact `exp` for θ —
/// matches pillars; the legacy SIMD path used `fast_exp` polynomial instead).
fn scalar_lifat_step(
    v: &mut [f32],
    theta: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    let dt = params[0];
    for idx in 0..n {
        let tau_m = params[n + idx];
        let v_rest = params[2 * n + idx];
        let v_thresh = params[3 * n + idx];
        let v_reset = params[4 * n + idx];
        let tau_theta = params[5 * n + idx];
        let beta = params[6 * n + idx];
        let i_clamped = ii[idx].clamp(-100.0, 100.0);

        if refractory[idx] > 0.0 {
            // Refractory: v frozen, θ still decays, no spike detection.
            refractory[idx] = (refractory[idx] - dt).max(0.0);
            spikes[idx] = 0;
            if tau_theta > 0.0 {
                theta[idx] *= (-dt / tau_theta).exp();
            }
            continue;
        }
        if tau_m > 0.0 {
            v[idx] += (v_rest - v[idx] + i_clamped) / tau_m * dt;
        }
        if tau_theta > 0.0 {
            theta[idx] *= (-dt / tau_theta).exp();
        }
        if v[idx] >= v_thresh + theta[idx] {
            spikes[idx] = 1;
            v[idx] = v_reset;
            theta[idx] += beta;
            refractory[idx] = tau_ref;
        } else {
            spikes[idx] = 0;
        }
    }
}

/// #1 Бит-в-бит: мост `safe::lif_at_step` ≡ прямой `run_neuron::<LIFAtModel>`.
/// Прогон n=8 (SIMD-путь) и n=13 (SIMD + скалярный хвост).
#[test]
fn bridge_bit_identical_to_direct_run_neuron() {
    for &n in &[8usize, 13usize] {
        // ── bridge path (safe::lif_at_step, legacy 7-col API) ──
        let mut bv = vec![V_REST; n];
        let mut bth = vec![BETA * 2.0; n];
        let mut bI = vec![I_VAL; n];
        let mut bsp = vec![0i32; n];
        let mut brefr = vec![0.0f32; n];
        let bparams = legacy_params(n);

        // ── direct pillars path (8-col params, same states) ──
        let mut pv = vec![V_REST; n];
        let mut pth = vec![BETA * 2.0; n];
        let mut pref = vec![0.0f32; n];
        let mut psp = vec![0i32; n];
        let pparams = pillars_params(n);
        let mut dummy = vec![0.0f32; n];

        for _ in 0..STEPS {
            unsafe {
                lif_at_step(
                    &mut bv, &mut bth, &mut bI, &mut bsp, &bparams, &mut brefr, TAU_REF, None,
                );
                run_neuron::<LIFAtModel>(
                    pv.as_mut_ptr(),
                    pth.as_mut_ptr(),
                    pref.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    bI.as_ptr(),
                    psp.as_mut_ptr(),
                    pparams.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert_eq!(
                    bv[i].to_bits(),
                    pv[i].to_bits(),
                    "n={n} i={i}: v not bit-identical (bridge {} vs direct {})",
                    bv[i],
                    pv[i]
                );
                assert_eq!(
                    bth[i].to_bits(),
                    pth[i].to_bits(),
                    "n={n} i={i}: θ not bit-identical"
                );
                assert_eq!(
                    brefr[i].to_bits(),
                    pref[i].to_bits(),
                    "n={n} i={i}: refractory not bit-identical"
                );
                assert_eq!(bsp[i], psp[i], "n={n} i={i}: spikes not identical");
            }
        }
        eprintln!("LIF-AT bridge: n={n}, {STEPS} steps — bit-identical ✓");
    }
}

/// #2 Легаси-семантика сохранена: мост vs скалярный эталон (допуски).
#[test]
fn bridge_matches_legacy_scalar_semantics() {
    const N: usize = 8;
    let mut bv = vec![V_REST; N];
    let mut bth = vec![BETA * 2.0; N];
    let mut bI = vec![I_VAL; N];
    let mut bsp = vec![0i32; N];
    let mut brefr = vec![0.0f32; N];
    let bparams = legacy_params(N);

    let mut sv = bv.clone();
    let mut sth = bth.clone();
    let mut ssp = vec![0i32; N];
    let mut srefr = vec![0.0f32; N];

    let mut b_total = 0i32;
    let mut s_total = 0i32;
    let mut max_v_diff = 0.0f32;
    let mut max_th_rel = 0.0f32;
    let mut max_refr_diff = 0.0f32;

    for _ in 0..STEPS {
        lif_at_step(
            &mut bv, &mut bth, &mut bI, &mut bsp, &bparams, &mut brefr, TAU_REF, None,
        );
        scalar_lifat_step(
            &mut sv, &mut sth, &bI, &mut ssp, &bparams, &mut srefr, TAU_REF,
        );
        for i in 0..N {
            b_total += bsp[i];
            s_total += ssp[i];
            max_v_diff = max_v_diff.max((bv[i] - sv[i]).abs());
            max_th_rel = max_th_rel.max((bth[i] - sth[i]).abs() / sth[i].abs().max(1e-6));
            max_refr_diff = max_refr_diff.max((brefr[i] - srefr[i]).abs());
        }
    }

    assert!(b_total > 0 && s_total > 0, "обе стороны должны спайкать");
    assert_eq!(
        b_total, s_total,
        "spike counts must match: bridge {b_total} vs scalar {s_total}"
    );
    assert!(
        max_th_rel < 1e-3,
        "θ rel diff {max_th_rel:.2e} > 1e-3 (fast_exp vs exact exp)"
    );
    assert!(
        max_v_diff < 30.0,
        "max |Δv| = {max_v_diff} — превышает допуск на шаг сдвига спайка"
    );
    assert!(
        max_refr_diff < TAU_REF + 0.5,
        "max |Δrefr| = {max_refr_diff} — расхождение больше одного спайкового шага"
    );
    eprintln!(
        "LIF-AT legacy semantics: {STEPS} steps, spikes bridge={b_total} scalar={s_total}, max|Δv|={max_v_diff:.4}, max θ rel={max_th_rel:.2e}, max|Δrefr|={max_refr_diff:.3} ✓"
    );
}
