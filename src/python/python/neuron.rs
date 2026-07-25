//! PyO3 bindings for neuron step functions.
//! Все array параметры — numpy arrays (zero-copy).

#![allow(non_snake_case)]

use std::cell::RefCell;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::core::neuron;

thread_local! {
    static REFRACTORY_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn with_refractory(n: usize) -> Vec<f32> {
    REFRACTORY_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(n, 0.0);
        // Take ownership temporarily — will be dropped after the call
        std::mem::take(&mut *buf)
    })
}

fn return_refractory(mut buf: Vec<f32>) {
    REFRACTORY_BUF.with(|cell| {
        buf.clear();
        *cell.borrow_mut() = buf;
    });
}

// ─── Neuron bindings ─────────────────────────────────────

/// Izhikevich neuron — один шаг симуляции.
///
/// Модифицирует входящие numpy массивы in-place.
///
/// Args:
///     v: мембранный потенциал [n]
///     u: recovery variable [n]
///     trace: след для STDP [n]
///     a: timescale recovery [n]
///     b: sensitivity recovery [n]
///     I: ток [n] (будет изменён!)
///     dt: шаг по времени
///     trace_decay: коэффициент затухания следа
///     spikes: спайки (int32, будет заполнен 0/1) [n]
///     c: потенциал после спайка [n]
///     d: recovery после спайка [n]
///
/// Returns: None. Все изменения in-place.
#[pyfunction]
#[pyo3(name = "step_izhikevich")]
pub fn py_step_izhikevich(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    trace_decay: f32,
    spikes: Bound<'_, PyArray1<i32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let a = a.readonly();
    let b = b.readonly();
    let mut I = I.readwrite();
    let mut spikes = spikes.readwrite();
    let c = c.readonly();
    let d = d.readonly();

    let n = v.len()?;
    let mut refractory = with_refractory(n);

    neuron::step_izhikevich(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        trace.as_slice_mut()?,
        a.as_slice()?,
        b.as_slice()?,
        I.as_slice_mut()?,
        dt,
        trace_decay,
        spikes.as_slice_mut()?,
        c.as_slice()?,
        d.as_slice()?,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    Ok(())
}

/// SAW (Sine-Adaptive-Wave) neuron — один шаг симуляции.
///
/// Args:
///     v: мембранный потенциал [n]
///     u: recovery variable [n]
///     trace: след для STDP [n]
///     a: timescale [n]
///     b: sensitivity [n]
///     I: ток [n] (readonly)
///     dt: шаг по времени
///     trace_decay: затухание следа
///     noise_amp: амплитуда шума
///     spikes: спайки (int32, будет заполнен) [n]
#[pyfunction]
#[pyo3(name = "step_saw")]
pub fn py_step_saw(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    spikes: Bound<'_, PyArray1<i32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let a = a.readonly();
    let b = b.readonly();
    let I = I.readonly();
    let mut spikes = spikes.readwrite();

    let n = v.len()?;
    let mut refractory = with_refractory(n);

    neuron::step_saw(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        trace.as_slice_mut()?,
        a.as_slice()?,
        b.as_slice()?,
        I.as_slice()?,
        dt,
        trace_decay,
        noise_amp,
        spikes.as_slice_mut()?,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    Ok(())
}

/// ψ-wave (Schrödinger neuron) — один шаг эволюции волновой функции.
///
/// Уравнение: ħ·dψ/dt = -i·H·ψ + I_ext · ψ · speed
/// где ψ = v + i·u, H = гамильтониан (вычисляется отдельно)
///
/// Args:
///     v: действительная часть ψ [n]
///     u: мнимая часть ψ [n]
///     h_real: H·u (гамильтониан × мнимая часть) [n]
///     h_imag: -H·v (гамильтониан × действительная часть) [n]
///     I_ext: внешний ток [n]
///     speed: скорость (модулирует фазу) [n]
///     dt: шаг по времени
///     hbar: квант действия (ħ), контролирует частоту
#[pyfunction]
#[pyo3(name = "step_psi_wave")]
pub fn py_step_psi_wave(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    h_real: Bound<'_, PyArray1<f32>>,
    h_imag: Bound<'_, PyArray1<f32>>,
    I_ext: Bound<'_, PyArray1<f32>>,
    speed: Bound<'_, PyArray1<f32>>,
    dt: f32,
    hbar: f32,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut h_real = h_real.readwrite();
    let mut h_imag = h_imag.readwrite();
    let mut I_ext = I_ext.readwrite();
    let mut speed = speed.readwrite();

    neuron::step_psi_wave(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        h_real.as_slice_mut()?,
        h_imag.as_slice_mut()?,
        I_ext.as_slice_mut()?,
        speed.as_slice_mut()?,
        dt,
        hbar,
    );
    Ok(())
}

/// LIF (Leaky Integrate-and-Fire) — один шаг симуляции.
///
/// Простейшая модель нейрона: τ·dv/dt = -v + I
///
/// Args:
///     v: мембранный потенциал [n]
///     I: ток [n]
///     spikes: спайки (int32) [n]
///     dt: шаг по времени
///     tau_m: мембранная постоянная [n]
///     v_rest: потенциал покоя [n]
///     v_thresh: порог спайка [n]
///     v_reset: потенциал после спайка [n]
#[pyfunction]
#[pyo3(name = "step_lif")]
pub fn py_step_lif(
    v: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    dt: f32,
    tau_m: Bound<'_, PyArray1<f32>>,
    v_rest: Bound<'_, PyArray1<f32>>,
    v_thresh: Bound<'_, PyArray1<f32>>,
    v_reset: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut I = I.readwrite();
    let mut spikes = spikes.readwrite();
    let tau_m = tau_m.readonly();
    let v_rest = v_rest.readonly();
    let v_thresh = v_thresh.readonly();
    let v_reset = v_reset.readonly();

    let n = v.len()?;
    let mut refractory = with_refractory(n);

    neuron::step_lif(
        v.as_slice_mut()?,
        I.as_slice_mut()?,
        spikes.as_slice_mut()?,
        dt,
        tau_m.as_slice()?,
        v_rest.as_slice()?,
        v_thresh.as_slice()?,
        v_reset.as_slice()?,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    Ok(())
}

/// AdEx (Adaptive Exponential) — один шаг симуляции.
///
/// Более богатая модель: добаляет экспоненциальный активационный член
/// и адаптацию через u.
///
/// Args:
///     v: мембранный потенциал [n]
///     u: адаптационная переменная [n]
///     I: ток [n]
///     spikes: спайки (int32) [n]
///     dt: шаг по времени
///     tau_m: мембранная постоянная [n]
///     tau_w: постоянная адаптации [n]
///     v_rest: потенциал покоя [n]
///     v_thresh: порог [n]
///     v_reset: сброс после спайка [n]
///     delta_t: крутизна экспоненты [n]
///     a: ионная проводимость [n]
///     b: сила адаптации [n]
#[pyfunction]
#[pyo3(name = "step_adex")]
pub fn py_step_adex(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    dt: f32,
    tau_m: Bound<'_, PyArray1<f32>>,
    tau_w: Bound<'_, PyArray1<f32>>,
    v_rest: Bound<'_, PyArray1<f32>>,
    v_thresh: Bound<'_, PyArray1<f32>>,
    v_reset: Bound<'_, PyArray1<f32>>,
    delta_t: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut I = I.readwrite();
    let mut spikes = spikes.readwrite();
    let tau_m = tau_m.readonly();
    let tau_w = tau_w.readonly();
    let v_rest = v_rest.readonly();
    let v_thresh = v_thresh.readonly();
    let v_reset = v_reset.readonly();
    let delta_t = delta_t.readonly();
    let a = a.readonly();
    let b = b.readonly();

    let n = v.len()?;
    let mut refractory = with_refractory(n);

    neuron::step_adex(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        I.as_slice_mut()?,
        spikes.as_slice_mut()?,
        dt,
        tau_m.as_slice()?,
        tau_w.as_slice()?,
        v_rest.as_slice()?,
        v_thresh.as_slice()?,
        v_reset.as_slice()?,
        delta_t.as_slice()?,
        a.as_slice()?,
        b.as_slice()?,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    Ok(())
}

/// STDP (Spike-Timing-Dependent Plasticity) — обновление весов.
///
/// Δw = -lr_ltd · trace_pre (для пост-спайка)
/// Δw = +lr_ltp · trace_post (для пре-спайка)
///
/// Args:
///     weights: синаптические веса [n] (будет изменён)
///     traces_pre: следы пре-синаптических нейронов [n]
///     traces_post: следы пост-синаптических нейронов [n]
///     lr_ltd: learning rate для LTD
///     lr_ltp: learning rate для LTP
///     w_min: минимальный вес
///     w_max: максимальный вес
#[pyfunction]
#[pyo3(name = "stdp_update")]
pub fn py_stdp_update(
    weights: Bound<'_, PyArray1<f32>>,
    traces_pre: Bound<'_, PyArray1<f32>>,
    traces_post: Bound<'_, PyArray1<f32>>,
    lr_ltd: f32,
    lr_ltp: f32,
    w_min: f32,
    w_max: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let traces_pre = traces_pre.readonly();
    let traces_post = traces_post.readonly();

    neuron::stdp_update(
        weights.as_slice_mut()?,
        traces_pre.as_slice()?,
        traces_post.as_slice()?,
        lr_ltd,
        lr_ltp,
        w_min,
        w_max,
    );
    Ok(())
}

/// Homeostatic STDP — поддерживает сумму весов около target_sum.
///
/// Args:
///     weights: синаптические веса [n] (будет изменён)
///     traces_pre: следы пре-нейронов [n]
///     traces_post: следы пост-нейронов [n]
///     lr_ltd: learning rate LTD
///     lr_ltp: learning rate LTP
///     target_sum: целевая сумма весов
///     w_max: максимальный вес
#[pyfunction]
#[pyo3(name = "stdp_homeostatic")]
pub fn py_stdp_homeostatic(
    weights: Bound<'_, PyArray1<f32>>,
    traces_pre: Bound<'_, PyArray1<f32>>,
    traces_post: Bound<'_, PyArray1<f32>>,
    lr_ltd: f32,
    lr_ltp: f32,
    target_sum: f32,
    w_max: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let traces_pre = traces_pre.readonly();
    let traces_post = traces_post.readonly();

    neuron::stdp_homeostatic(
        weights.as_slice_mut()?,
        traces_pre.as_slice()?,
        traces_post.as_slice()?,
        lr_ltd,
        lr_ltp,
        target_sum,
        w_max,
    );
    Ok(())
}

// ─── FitzHugh-Nagumo ─────────────────────────────────────

/// FitzHugh-Nagumo neuron — упрощённая версия Hodgkin-Huxley.
///
/// Уравнения:
///   dv/dt = v - v³/3 - w + I_ext
///   dw/dt = (v + a - b·w) / τ
///
/// Args:
///     v: мембранный потенциал [n]
///     w: recovery variable [n]
///     I: внешний ток [n]
///     dt: шаг по времени (мс)
///     tau_w: временная константа recovery [n]
///     a: параметр nullcline [n]
///     b: параметр nullcline [n]
///     spikes: спайки (int32, будет заполнен 0/1) [n]
///
/// Returns: None. Все изменения in-place.
#[pyfunction]
#[pyo3(name = "step_fhn")]
pub fn py_step_fhn(
    v: Bound<'_, PyArray1<f32>>,
    w: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    tau_w: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut w = w.readwrite();
    let I = I.readonly();
    let tau_w = tau_w.readonly();
    let a = a.readonly();
    let b = b.readonly();
    let mut spikes = spikes.readwrite();

    neuron::step_fhn(
        v.as_slice_mut()?,
        w.as_slice_mut()?,
        I.as_slice()?,
        spikes.as_slice_mut()?,
        dt,
        tau_w.as_slice()?,
        a.as_slice()?,
        b.as_slice()?,
    );
    Ok(())
}

// ─── Wilson-Cowan ─────────────────────────────────────────

/// Wilson-Cowan population model — модель возбуждения-торможения.
///
/// Уравнения:
///   τ_E · dE/dt = -E + sigmoid(w_EE·E - w_EI·I + I_E)
///   τ_I · dI/dt = -I + sigmoid(w_IE·E - w_II·I + I_I)
///
/// Это rate-based модель (непрерывные скорости, без генерации спайков).
///
/// Args:
///     e: скорость возбуждающей популяции [n]
///     i: скорость тормозящей популяции [n]
///     i_e: внешний вход для E [n]
///     i_i: внешний вход для I [n]
///     dt: шаг по времени (мс)
///     tau_E: временная константа E [n]
///     tau_I: временная константа I [n]
///     w_EE: вес E→E [n]
///     w_EI: вес I→E [n]
///     w_IE: вес E→I [n]
///     w_II: вес I→I [n]
///     beta: крутизна сигмоиды [n]
///     spikes_e: выходная скорость E (float32) [n]
///     spikes_i: выходная скорость I (float32) [n]
///
/// Returns: None. Все изменения in-place.
#[pyfunction]
#[pyo3(name = "step_wc")]
pub fn py_step_wc(
    e: Bound<'_, PyArray1<f32>>,
    i: Bound<'_, PyArray1<f32>>,
    i_e: Bound<'_, PyArray1<f32>>,
    i_i: Bound<'_, PyArray1<f32>>,
    dt: f32,
    tau_E: Bound<'_, PyArray1<f32>>,
    tau_I: Bound<'_, PyArray1<f32>>,
    w_EE: Bound<'_, PyArray1<f32>>,
    w_EI: Bound<'_, PyArray1<f32>>,
    w_IE: Bound<'_, PyArray1<f32>>,
    w_II: Bound<'_, PyArray1<f32>>,
    beta: Bound<'_, PyArray1<f32>>,
    spikes_e: Bound<'_, PyArray1<f32>>,
    spikes_i: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut e = e.readwrite();
    let mut i = i.readwrite();
    let i_e = i_e.readonly();
    let i_i = i_i.readonly();
    let tau_E = tau_E.readonly();
    let tau_I = tau_I.readonly();
    let w_EE = w_EE.readonly();
    let w_EI = w_EI.readonly();
    let w_IE = w_IE.readonly();
    let w_II = w_II.readonly();
    let beta = beta.readonly();
    let mut spikes_e = spikes_e.readwrite();
    let mut spikes_i = spikes_i.readwrite();

    neuron::step_wc(
        e.as_slice_mut()?,
        i.as_slice_mut()?,
        i_e.as_slice()?,
        i_i.as_slice()?,
        spikes_e.as_slice_mut()?,
        spikes_i.as_slice_mut()?,
        dt,
        tau_E.as_slice()?,
        tau_I.as_slice()?,
        w_EE.as_slice()?,
        w_EI.as_slice()?,
        w_IE.as_slice()?,
        w_II.as_slice()?,
        beta.as_slice()?,
    );
    Ok(())
}

/// Atlasov Neuromotor Engine — один шаг симуляции мышц и суставов.
///
/// Modifies ca, tension, joint_delta_f, joint_stiffness, joint_damping in-place.
#[pyfunction]
#[pyo3(name = "step_atlasov_neuromotor")]
pub fn py_step_atlasov_neuromotor(
    ca: Bound<'_, PyArray1<f32>>,
    tension: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    resets: Bound<'_, PyArray1<i32>>,
    dt: f32,
    d_ca: Bound<'_, PyArray1<f32>>,
    tau_pump: Bound<'_, PyArray1<f32>>,
    eta_reset: Bound<'_, PyArray1<f32>>,
    t_max: Bound<'_, PyArray1<f32>>,
    kd: Bound<'_, PyArray1<f32>>,
    joint_delta_f: Option<Bound<'_, PyArray1<f32>>>,
    joint_stiffness: Option<Bound<'_, PyArray1<f32>>>,
    joint_damping: Option<Bound<'_, PyArray1<f32>>>,
    k0: Option<Bound<'_, PyArray1<f32>>>,
    beta: Option<Bound<'_, PyArray1<f32>>>,
    b0: Option<Bound<'_, PyArray1<f32>>>,
    alpha: Option<Bound<'_, PyArray1<f32>>>,
) -> PyResult<()> {
    let mut ca_rw = ca.readwrite();
    let mut tension_rw = tension.readwrite();
    let spikes_ro = spikes.readonly();
    let resets_ro = resets.readonly();
    let d_ca_ro = d_ca.readonly();
    let tau_pump_ro = tau_pump.readonly();
    let eta_reset_ro = eta_reset.readonly();
    let t_max_ro = t_max.readonly();
    let kd_ro = kd.readonly();

    let mut j_df_storage = joint_delta_f.as_ref().map(|a| a.readwrite());
    let mut j_stf_storage = joint_stiffness.as_ref().map(|a| a.readwrite());
    let mut j_dmp_storage = joint_damping.as_ref().map(|a| a.readwrite());

    let k0_ro = k0.as_ref().map(|a| a.readonly());
    let beta_ro = beta.as_ref().map(|a| a.readonly());
    let b0_ro = b0.as_ref().map(|a| a.readonly());
    let alpha_ro = alpha.as_ref().map(|a| a.readonly());

    neuron::step_atlasov_neuromotor(
        ca_rw.as_slice_mut()?,
        tension_rw.as_slice_mut()?,
        spikes_ro.as_slice()?,
        resets_ro.as_slice()?,
        dt,
        d_ca_ro.as_slice()?,
        tau_pump_ro.as_slice()?,
        eta_reset_ro.as_slice()?,
        t_max_ro.as_slice()?,
        kd_ro.as_slice()?,
        j_df_storage.as_mut().and_then(|a| a.as_slice_mut().ok()),
        j_stf_storage.as_mut().and_then(|a| a.as_slice_mut().ok()),
        j_dmp_storage.as_mut().and_then(|a| a.as_slice_mut().ok()),
        k0_ro.as_ref().and_then(|a| a.as_slice().ok()),
        beta_ro.as_ref().and_then(|a| a.as_slice().ok()),
        b0_ro.as_ref().and_then(|a| a.as_slice().ok()),
        alpha_ro.as_ref().and_then(|a| a.as_slice().ok()),
    );

    Ok(())
}
