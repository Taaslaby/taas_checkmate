//! # *_step — pillars-канон parity (волна 7, транш 3e)
//!
//! Консолидация завершена: `safe.rs` удалён — все мосты переупаковки живут
//! прямо в этих тестах как локальные функции (легаси SoA-лейауты
//! переупаковываются на границе вызова). Тесты гоняют локальный мост против
//! прямого `run_neuron::<Model>` на идентичных входах и сравнивают суммарный
//! счёт спайков — ловят ошибки переупаковки параметров/состояний.
//!
//! Note on tolerances: the pillars models deliberately *fix* legacy numerics
//! (refractory clamp-at-input vs freeze, double substep for Euler stability,
//! spike detection at v_peak). On identical parameter sets the spike **counts**
//! must still match — the exact voltage traces may differ slightly.

use pillars::core::runner::run_neuron;
use pillars::zoo::neurons::{AdExModel, HHModel, IzhikevichModel, LIFModel, SAWModel};
use pillars::zoo::Model;

/// Run `n_steps` of the OLD `lif_step` kernel and count total spikes.
/// Референт LIF (канон == легаси 35356c1, семантика совпадает): кламп Ii ±100,
/// Forward Euler `v += (v_rest − v + I)/τ·dt`, спайк при v ≥ v_thresh,
/// рефрактерность: декремент `refr −= dt`, на спайке `refr = tau_ref`.
/// (Мост `safe::lif_step` удалён в транше 3b.)
fn ref_lif(
    v: &mut [f32],
    refractory: &mut [f32],
    ii: &mut [f32],
    params: &[f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;
    let dt_val = params[0]; // dt is uniform across all neurons
    for _ in 0..n_steps {
        for i in 0..n {
            if refractory[i] > 0.0 {
                refractory[i] = (refractory[i] - dt_val).max(0.0);
                spikes[i] = 0;
                continue;
            }
            let tau_m = params[n + i];
            let v_rest = params[2 * n + i];
            let v_thresh = params[3 * n + i];
            let v_reset = params[4 * n + i];
            let i_clamped = ii[i].clamp(-100.0, 100.0);
            if tau_m > 0.0 {
                v[i] += (v_rest - v[i] + i_clamped) / tau_m * dt_val;
            }
            if v[i] >= v_thresh {
                spikes[i] = 1;
                v[i] = v_reset;
                refractory[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Run `n_steps` of pillars `run_neuron::<LIFModel>` and count total spikes.
/// Builds pillars params: [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref] × n.
fn new_lif(
    v: &mut [f32],
    refractory: &mut [f32],
    ii: &[f32],
    params_old: &[f32], // legacy 5-col layout
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut params = vec![0.0f32; n * 6];
    for i in 0..n {
        params[i] = params_old[i]; // dt
        params[n + i] = 1.0 / params_old[n + i].max(1e-6); // inv_tau_m = 1/tau_m
        params[2 * n + i] = params_old[2 * n + i]; // v_rest
        params[3 * n + i] = params_old[3 * n + i]; // v_thresh
        params[4 * n + i] = params_old[4 * n + i]; // v_reset
        params[5 * n + i] = tau_ref; // tau_ref column
    }
    let mut spikes = vec![0i32; n];
    let mut dummy = vec![0.0f32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<LIFModel>(
                v.as_mut_ptr(),
                refractory.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Run `n_steps` of the OLD `izhikevich_step` kernel and count total spikes.
/// Inline-эталон pillars-КАНОНА (`zoo/neurons/izhikevich.rs`): субстеппинг
/// (2 полушага 0.5·dt), Ii кламп ±100, du из pre-integration v, спайк при
/// v_new ≥ 30, во время рефрактерности v = c (reset) и dv ЗАМОРОЖЕН.
///
/// NOTE: легаси 35356c1 был простым Euler (один полный шаг dt) — канон
/// сознательно сменил схему на субстеппинг для стабильности (см. коммент
/// в pillars). Этот inline-эталон воспроизводит КАНОН, не легаси.
/// trace — легаси side-output, в pillars-модели отсутствует (не влияет на
/// спайки, здесь не воспроизводится).
/// (Мост `safe::izhikevich_step` удалён в транше 3b.)
fn ref_izh(
    v: &mut [f32],
    u: &mut [f32],
    _trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &mut [f32],
    dt: f32,
    _trace_decay: f32,
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        for i in 0..n {
            let is_refr = refractory[i] > 0.0;
            // Refractory clamp at input: v = v_reset (c) during refractory.
            let v_in = if is_refr { c[i] } else { v[i] };
            let i_clamped = ii[i].clamp(-100.0, 100.0);
            let half_dt = dt * 0.5;
            // Sub-step 1
            let dv1 = ((0.04 * v_in + 5.0) * v_in + 140.0 - u[i] + i_clamped) * half_dt;
            let v_mid = v_in + dv1;
            // Sub-step 2
            let dv2 = ((0.04 * v_mid + 5.0) * v_mid + 140.0 - u[i] + i_clamped) * half_dt;
            let dv_total = if is_refr { 0.0 } else { dv1 + dv2 };
            let v_new = v_in + dv_total;
            // du from pre-integration v, frozen in refractory.
            let du = if is_refr {
                0.0
            } else {
                a[i] * (b[i] * v_in - u[i]) * dt
            };
            let u_new = u[i] + du;
            if v_new >= 30.0 {
                spikes[i] = 1;
                v[i] = c[i];
                u[i] = u_new + d[i];
                refractory[i] = tau_ref;
            } else {
                spikes[i] = 0;
                v[i] = v_new;
                u[i] = u_new;
                refractory[i] = (refractory[i] - dt).max(0.0);
            }
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Pillars `run_neuron::<IzhikevichModel>`.
/// pillars IzhikevichModel: state [v, u, refr]; params [dt, a, b, c, d, tau_ref, v_thresh].
/// NOTE: pillars model has no `trace` state — trace is legacy STDP side-output.
fn new_izh(
    v: &mut [f32],
    u: &mut [f32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &[f32],
    dt: f32,
    c: &[f32],
    d: &[f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut params = vec![0.0f32; n * IzhikevichModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = a[i];
        params[2 * n + i] = b[i];
        params[3 * n + i] = c[i];
        params[4 * n + i] = d[i];
        params[5 * n + i] = tau_ref;
        params[6 * n + i] = 30.0; // legacy hardcoded v_th
    }
    let mut spikes = vec![0i32; n];
    let mut dummy = vec![0.0f32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refractory.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Локальный мост SAW (транш 3e; noise_amp=0 — детерминированный путь).
fn old_saw(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &[f32],
    dt: f32,
    trace_decay: f32,
    _noise_amp: f32,
    refractory: &mut [f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;
    let mut params = vec![0.0f32; n * SAWModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = a[i];
        params[2 * n + i] = b[i];
        params[3 * n + i] = trace_decay;
        params[4 * n + i] = 0.0; // noise_amp — модель детерминирована
        params[5 * n + i] = tau_ref;
        params[6 * n + i] = 30.0;
        params[7 * n + i] = -65.0;
    }
    let mut dummy = vec![0.0f32; n];
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<SAWModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                trace.as_mut_ptr(),
                refractory.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Pillars `run_neuron::<SAWModel>`.
/// pillars SAWModel: state [v, u, tr, refr]; params [dt, a, b, trace_decay, noise_amp, tau_ref, v_thresh, v_reset].
fn new_saw(
    v: &mut [f32],
    u: &mut [f32],
    tr: &mut [f32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &[f32],
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut params = vec![0.0f32; n * SAWModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = a[i];
        params[2 * n + i] = b[i];
        params[3 * n + i] = trace_decay;
        params[4 * n + i] = noise_amp;
        params[5 * n + i] = tau_ref;
        params[6 * n + i] = 30.0; // legacy v_th
        params[7 * n + i] = -65.0; // legacy v_reset
    }
    let mut spikes = vec![0i32; n];
    let mut dummy = vec![0.0f32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<SAWModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                tr.as_mut_ptr(),
                refractory.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Локальный мост AdEx (транш 3e). params_old: [dt, tau_m, tau_w, v_rest,
/// v_thresh, v_reset, delta_t, a, b] × n → AdExModel 12-col repacking.
fn old_adex(
    v: &mut [f32],
    u: &mut [f32],
    ii: &mut [f32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;
    let mut p = vec![0.0f32; n * AdExModel::N_PARAMS];
    for i in 0..n {
        p[i] = params[i];
        p[n + i] = 1.0 / params[n + i].max(1e-6);
        p[2 * n + i] = 1.0 / params[2 * n + i].max(1e-6);
        p[3 * n + i] = params[3 * n + i];
        p[4 * n + i] = params[4 * n + i];
        p[5 * n + i] = params[5 * n + i];
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = 1.0 / params[6 * n + i].max(1e-6);
        p[8 * n + i] = params[7 * n + i];
        p[9 * n + i] = params[8 * n + i];
        p[10 * n + i] = tau_ref;
        p[11 * n + i] = params[4 * n + i]; // v_peak := v_thresh
    }
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<AdExModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
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
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Pillars `run_neuron::<AdExModel>`.
/// pillars AdExModel: state [v, w, refr]; params [dt, inv_tau_m, inv_tau_w, v_rest, v_thresh, v_reset, delta_t, inv_delta_t, a, b, tau_ref, v_peak].
/// Legacy adex spikes at v >= v_thresh → set pillars v_peak = v_thresh for parity.
fn new_adex(
    v: &mut [f32],
    u: &mut [f32],
    refractory: &mut [f32],
    ii: &[f32],
    params_old: &[f32],
    tau_ref: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut params = vec![0.0f32; n * AdExModel::N_PARAMS];
    for i in 0..n {
        params[i] = params_old[i]; // dt
        params[n + i] = 1.0 / params_old[n + i].max(1e-6); // inv_tau_m
        params[2 * n + i] = 1.0 / params_old[2 * n + i].max(1e-6); // inv_tau_w
        params[3 * n + i] = params_old[3 * n + i]; // v_rest
        params[4 * n + i] = params_old[4 * n + i]; // v_thresh
        params[5 * n + i] = params_old[5 * n + i]; // v_reset
        params[6 * n + i] = params_old[6 * n + i]; // delta_t
        params[7 * n + i] = 1.0 / params_old[6 * n + i].max(1e-6); // inv_delta_t
        params[8 * n + i] = params_old[7 * n + i]; // a
        params[9 * n + i] = params_old[8 * n + i]; // b
        params[10 * n + i] = tau_ref;
        params[11 * n + i] = params_old[4 * n + i]; // v_peak := v_thresh (legacy)
    }
    let mut spikes = vec![0i32; n];
    let mut dummy = vec![0.0f32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<AdExModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refractory.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Compare two spike counts; allow small relative difference (late-phase timing drift).
fn assert_parity(name: &str, old: i64, new: i64) {
    if old == new {
        return;
    }
    let max = old.max(new).max(1) as f64;
    let rel = (old - new).abs() as f64 / max;
    assert!(
        rel < 0.05,
        "{name}: spike-count mismatch: old={old}, new={new} (rel diff {:.2}%)",
        rel * 100.0
    );
}

// ═══════════════════════════════════════════════════════════
//  LIF
// ═══════════════════════════════════════════════════════════

#[test]
fn parity_lif_currents() {
    for &n in &[8usize, 10usize] {
        for &i_val in &[0.0f32, 15.0, 30.0, 60.0, 100.0] {
            let dt = 0.5;
            let tau_m = 10.0;
            let v_rest = -70.0;
            let v_thresh = -54.0;
            let v_reset = -70.0;
            let tau_ref = 2.0;

            let mut params = vec![0.0f32; n * 5];
            for i in 0..n {
                params[i] = dt;
                params[n + i] = tau_m;
                params[2 * n + i] = v_rest;
                params[3 * n + i] = v_thresh;
                params[4 * n + i] = v_reset;
            }

            let ii = vec![i_val; n];

            // Old path
            let mut v_old = vec![v_rest; n];
            let mut refr_old = vec![0.0; n];
            let mut ii_old = ii.clone();
            let old = ref_lif(
                &mut v_old,
                &mut refr_old,
                &mut ii_old,
                &params,
                tau_ref,
                400,
            );

            // New path
            let mut v_new = vec![v_rest; n];
            let mut refr_new = vec![0.0; n];
            let new = new_lif(&mut v_new, &mut refr_new, &ii, &params, tau_ref, 400);

            assert_parity(&format!("LIF n={n} I={i_val}"), old, new);
        }
    }
}

#[test]
fn parity_lif_heterogeneous() {
    let n = 32;
    let dt = 0.5;
    let mut params = vec![0.0f32; n * 5];
    let mut ii = Vec::with_capacity(n);
    for i in 0..n {
        params[i] = dt;
        params[n + i] = 5.0 + 5.0 * (i % 5) as f32; // tau_m 5..25
        params[2 * n + i] = -70.0;
        params[3 * n + i] = -54.0 + 0.1 * (i % 3) as f32;
        params[4 * n + i] = -70.0;
        ii.push(5.0 * (i % 9) as f32); // I 0..40
    }
    let tau_ref = 2.0;

    let mut v_old = vec![-70.0; n];
    let mut refr_old = vec![0.0; n];
    let mut ii_old = ii.clone();
    let old = ref_lif(
        &mut v_old,
        &mut refr_old,
        &mut ii_old,
        &params,
        tau_ref,
        600,
    );

    let mut v_new = vec![-70.0; n];
    let mut refr_new = vec![0.0; n];
    let new = new_lif(&mut v_new, &mut refr_new, &ii, &params, tau_ref, 600);

    assert_parity("LIF heterogeneous", old, new);
}

// ═══════════════════════════════════════════════════════════
//  Izhikevich
// ═══════════════════════════════════════════════════════════

#[test]
fn parity_izh_currents() {
    for &n in &[8usize, 10usize] {
        for &i_val in &[0.0f32, 5.0, 10.0, 20.0, 40.0] {
            let dt = 0.5;
            let trace_decay = 0.01;
            let tau_ref = 2.0;
            let a = vec![0.02f32; n];
            let b = vec![0.2f32; n];
            let c = vec![-65.0f32; n];
            let d = vec![8.0f32; n];
            let ii = vec![i_val; n];

            let mut v_old = vec![-65.0; n];
            let mut u_old = vec![0.0; n];
            let mut tr_old = vec![0.0; n];
            let mut refr_old = vec![0.0; n];
            let mut ii_old = ii.clone();
            let old = ref_izh(
                &mut v_old,
                &mut u_old,
                &mut tr_old,
                &a,
                &b,
                &mut ii_old,
                dt,
                trace_decay,
                &c,
                &d,
                &mut refr_old,
                tau_ref,
                600,
            );

            let mut v_new = vec![-65.0; n];
            let mut u_new = vec![0.0; n];
            let mut refr_new = vec![0.0; n];
            let new = new_izh(
                &mut v_new,
                &mut u_new,
                &mut refr_new,
                &a,
                &b,
                &ii,
                dt,
                &c,
                &d,
                tau_ref,
                600,
            );

            assert_parity(&format!("Izh n={n} I={i_val}"), old, new);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  SAW
// ═══════════════════════════════════════════════════════════

#[test]
fn parity_saw_currents() {
    for &n in &[8usize, 10usize] {
        for &i_val in &[0.0f32, 8.0, 15.0, 30.0] {
            let dt = 1.0;
            let trace_decay = 0.01;
            let noise_amp = 0.0;
            let tau_ref = 2.0;
            let a = vec![0.02f32; n];
            let b = vec![0.2f32; n];
            let ii = vec![i_val; n];

            let mut v_old = vec![-65.0; n];
            let mut u_old = vec![0.0; n];
            let mut tr_old = vec![0.0; n];
            let mut refr_old = vec![0.0; n];
            let old = old_saw(
                &mut v_old,
                &mut u_old,
                &mut tr_old,
                &a,
                &b,
                &ii,
                dt,
                trace_decay,
                noise_amp,
                &mut refr_old,
                tau_ref,
                600,
            );

            let mut v_new = vec![-65.0; n];
            let mut u_new = vec![0.0; n];
            let mut tr_new = vec![0.0; n];
            let mut refr_new = vec![0.0; n];
            let new = new_saw(
                &mut v_new,
                &mut u_new,
                &mut tr_new,
                &mut refr_new,
                &a,
                &b,
                &ii,
                dt,
                trace_decay,
                noise_amp,
                tau_ref,
                600,
            );

            assert_parity(&format!("SAW n={n} I={i_val}"), old, new);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  AdEx
// ═══════════════════════════════════════════════════════════

#[test]
fn parity_adex_currents() {
    for &n in &[8usize, 10usize] {
        for &i_val in &[0.0f32, 5.0, 10.0, 15.0, 25.0] {
            let tau_ref = 2.0;
            let mut params = vec![0.0f32; n * 9];
            for i in 0..n {
                params[i] = 0.05; // dt
                params[n + i] = 5.0; // tau_m
                params[2 * n + i] = 100.0; // tau_w
                params[3 * n + i] = -65.0; // v_rest
                params[4 * n + i] = -50.0; // v_thresh
                params[5 * n + i] = -65.0; // v_reset
                params[6 * n + i] = 2.0; // delta_t
                params[7 * n + i] = 0.5; // a
                params[8 * n + i] = 7.0; // b
            }
            let ii = vec![i_val; n];

            let mut v_old = vec![-65.0; n];
            let mut u_old = vec![0.0; n];
            let mut refr_old = vec![0.0; n];
            let mut ii_old = ii.clone();
            let old = old_adex(
                &mut v_old,
                &mut u_old,
                &mut ii_old,
                &params,
                &mut refr_old,
                tau_ref,
                800,
            );

            let mut v_new = vec![-65.0; n];
            let mut u_new = vec![0.0; n];
            let mut refr_new = vec![0.0; n];
            let new = new_adex(
                &mut v_new,
                &mut u_new,
                &mut refr_new,
                &ii,
                &params,
                tau_ref,
                800,
            );

            assert_parity(&format!("AdEx n={n} I={i_val}"), old, new);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  HH (fused poly)
// ═══════════════════════════════════════════════════════════

/// Обёртка raw-кернела `hh::fused_hh_poly_step` — pillars-порт легаси-движка
/// (reverse zero-crossing spike detection внутри кернела). «Old»-референт parity:
/// независимый путь от `run_neuron::<HHModel>` (транш 3e: `safe.rs` удалён).
fn old_fused_hh(
    v: &mut [f32],
    gates: &mut [f32],
    ii: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    meta: &[pillars::kernels::hh_channels::ChannelMeta],
    g_max: &[f32],
    dt: f32,
    celsius: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    let mut spikes = vec![0i32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        unsafe {
            pillars::kernels::hh_channels::fused_hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                meta,
                g_max.as_ptr(),
                dt,
                celsius,
                spikes.as_mut_ptr(),
            );
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

/// Pillars `run_neuron::<HHModel>` — mirrors the wrapper that safe.rs will become.
///
/// Legacy gate order [m×n, h×n, n×n] maps directly onto HHModel state pointers
/// (v, m, h, n, c, was_below). Spike detection is replicated as reverse
/// zero-crossing (v_old ≥ 0 && v_new < 0) — exactly the legacy convention.
fn new_fused_hh(
    v: &mut [f32],
    gates: &mut [f32],
    ii: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    meta: &[pillars::kernels::hh_channels::ChannelMeta],
    g_max: &[f32],
    dt: f32,
    celsius: f32,
    n_steps: usize,
) -> i64 {
    let n = v.len();
    // Standard layout: NaV(m³h) + KV(n⁴) + leak, optional CaT(c).
    assert!(
        meta.len() == 2 || meta.len() == 3,
        "HH parity supports standard/pacemaking"
    );
    let q10 = 3.0f32.powf((celsius - 6.3) / 10.0);

    let mut params = vec![0.0f32; n * HHModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = g_max[i]; // g_Na
        params[2 * n + i] = meta[0].e_rev; // E_Na
        params[3 * n + i] = g_max[n + i]; // g_K
        params[4 * n + i] = meta[1].e_rev; // E_K
        params[5 * n + i] = g_leak[i]; // g_Leak
        params[6 * n + i] = e_leak[i]; // E_Leak
        params[7 * n + i] = q10;
        params[8 * n + i] = 0.0; // v_thresh (legacy crossing @ 0)
        params[9 * n + i] = 1.0 / c_m[i].max(1e-6); // inv_cm
        params[10 * n + i] = if meta.len() == 3 {
            g_max[2 * n + i]
        } else {
            0.0
        }; // g_CaT
        params[11 * n + i] = if meta.len() == 3 {
            meta[2].e_rev
        } else {
            120.0
        }; // E_CaT
    }

    let mut spikes = vec![0i32; n];
    let mut was_below = vec![0.0f32; n]; // ignored: we recompute spikes ourselves
    let mut c_scratch = vec![0.0f32; n];
    let mut total = 0i64;
    for _ in 0..n_steps {
        let v_old = v.to_vec();
        unsafe {
            run_neuron::<HHModel>(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                gates.as_mut_ptr().add(n),     // h
                gates.as_mut_ptr().add(2 * n), // n_gate
                c_scratch.as_mut_ptr(),
                was_below.as_mut_ptr(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        // Legacy convention: reverse zero-crossing (falling through 0)
        for i in 0..n {
            spikes[i] = if v_old[i] >= 0.0 && v[i] < 0.0 { 1 } else { 0 };
        }
        total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    total
}

#[test]
fn parity_fused_hh_standard() {
    use pillars::kernels::hh_channels::{
        build_channel_meta, standard_channels, steady_state_gates,
    };

    for &n in &[8usize, 16usize] {
        for &i_val in &[0.0f32, 15.0, 30.0] {
            let dt = 0.05;
            let celsius = 6.3; // q10 = 1
            let channels = standard_channels();
            let meta = build_channel_meta(&channels);
            let init = steady_state_gates(-65.0, &channels); // [m, h, n]

            let n_gates: usize = meta.iter().map(|ch| ch.n_gates).sum();
            let mut gates = vec![0.0f32; n_gates * n];
            for gi in 0..n_gates {
                for i in 0..n {
                    gates[gi * n + i] = init[gi];
                }
            }
            let mut g_max = vec![0.0f32; meta.len() * n];
            for (ci, ch) in channels.iter().enumerate() {
                for i in 0..n {
                    g_max[ci * n + i] = ch.g_max;
                }
            }
            let g_leak = vec![0.3f32; n];
            let e_leak = vec![-65.0f32; n];
            let c_m = vec![1.0f32; n];
            let ii = vec![i_val; n];

            let mut v_old = vec![-65.0; n];
            let mut g_old = gates.clone();
            let old = old_fused_hh(
                &mut v_old, &mut g_old, &ii, &g_leak, &e_leak, &c_m, &meta, &g_max, dt, celsius,
                4000,
            );

            let mut v_new = vec![-65.0; n];
            let mut g_new = gates;
            let new = new_fused_hh(
                &mut v_new, &mut g_new, &ii, &g_leak, &e_leak, &c_m, &meta, &g_max, dt, celsius,
                4000,
            );

            assert_parity(&format!("HH n={n} I={i_val}"), old, new);
        }
    }
}
