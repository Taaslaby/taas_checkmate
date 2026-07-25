//! PyO3 bindings for neuron step functions.
//! Все array параметры — numpy arrays (zero-copy).

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::cell::RefCell;

use crate::core::neuron;
use crate::core::neuron::soma_dendrite::{DendriteConfig, SomatoDendriticNeuron};

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

/// GCO (Graph-Coupled Oscillator) — один шаг.
///
/// Уравнение: dv/dt = H_real + I_ext · speed, du/dt = H_imag
/// где ψ = v + i·u, H = гамильтониан (вычисляется отдельно)
///
/// Args:
///     v: действительная часть [n]
///     u: мнимая часть [n]
///     h_real: H·u (гамильтониан × мнимая часть) [n]
///     h_imag: -H·v (гамильтониан × действительная часть) [n]
///     I_ext: внешний ток [n]
///     speed: скорость (модулирует фазу) [n]
///     dt: шаг по времени
#[pyfunction]
#[pyo3(name = "step_gco")]
pub fn py_step_gco(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    h_real: Bound<'_, PyArray1<f32>>,
    h_imag: Bound<'_, PyArray1<f32>>,
    I_ext: Bound<'_, PyArray1<f32>>,
    speed: Bound<'_, PyArray1<f32>>,
    dt: f32,
    _tau: f32,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut h_real = h_real.readwrite();
    let mut h_imag = h_imag.readwrite();
    let mut I_ext = I_ext.readwrite();
    let mut speed = speed.readwrite();

    neuron::step_gco(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        h_real.as_slice_mut()?,
        h_imag.as_slice_mut()?,
        I_ext.as_slice_mut()?,
        speed.as_slice_mut()?,
        dt,
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
#[pyo3(signature = (ca, tension, spikes, resets, dt, d_ca, tau_pump, eta_reset, t_max, kd, l0, v_max, lengths=None, velocities=None,
    joint_delta_f=None, joint_stiffness=None, joint_damping=None,
    k0=None, beta=None, b0=None, alpha=None))]
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
    l0: Bound<'_, PyArray1<f32>>,
    v_max: Bound<'_, PyArray1<f32>>,
    lengths: Option<Bound<'_, PyArray1<f32>>>,
    velocities: Option<Bound<'_, PyArray1<f32>>>,
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
    let l0_ro = l0.readonly();
    let v_max_ro = v_max.readonly();

    let lengths_ro = lengths.as_ref().map(|a| a.readonly());
    let velocities_ro = velocities.as_ref().map(|a| a.readonly());

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
        l0_ro.as_slice()?,
        v_max_ro.as_slice()?,
        lengths_ro.as_ref().and_then(|a| a.as_slice().ok()),
        velocities_ro.as_ref().and_then(|a| a.as_slice().ok()),
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

/// PyO3 класс сомато-дендритного нейрона с активными HH + Ca2+ дендритами.
#[pyclass(name = "SomatoDendriticNeuron", module = "taas_checkmate.neurons")]
pub struct PySomatoDendriticNeuron {
    pub(crate) inner: SomatoDendriticNeuron,
}

#[pymethods]
impl PySomatoDendriticNeuron {
    #[new]
    #[pyo3(signature = (n_neurons, n_compartments=4, r_a=200.0, r_m=100.0, c_m=0.1, dt=0.5, tau_m=20.0, v_rest=-65.0, v_thresh=-54.0, v_reset=-65.0, tau_ref=3.0))]
    pub fn new(
        n_neurons: usize,
        n_compartments: usize,
        r_a: f32,
        r_m: f32,
        c_m: f32,
        dt: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ref: f32,
    ) -> Self {
        let config = DendriteConfig {
            n_compartments,
            R_a: r_a,
            R_m: r_m,
            C_m: c_m,
        };
        let inner = SomatoDendriticNeuron::new(
            n_neurons, &config, dt, tau_m, v_rest, v_thresh, v_reset, tau_ref,
        );
        Self { inner }
    }

    /// Включить активные HH + Ca2+ ионные каналы в дендритах.
    #[pyo3(signature = (g_na=10.0, g_kv=5.0, g_ca=2.0))]
    pub fn enable_active_channels(&mut self, g_na: f32, g_kv: f32, g_ca: f32) {
        self.inner.dendrite.enable_active_channels(g_na, g_kv, g_ca);
    }

    /// Выполнить шаг симуляции.
    pub fn step<'py>(
        &mut self,
        _py: Python<'py>,
        I_ext: Bound<'py, PyArray1<f32>>,
        dt: f32,
    ) -> PyResult<()> {
        let i_ext_ro = I_ext.readonly();
        self.inner.step(i_ext_ro.as_slice()?, dt);
        Ok(())
    }

    /// Скачок AMPA проводимости на компартменте comp нейрона nrn
    pub fn jump_ampa(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_ampa(nrn, comp, g);
    }

    /// Скачок NMDA проводимости на компартменте comp нейрона nrn
    pub fn jump_nmda(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_nmda(nrn, comp, g);
    }

    /// Скачок GABA_A проводимости на компартменте comp нейрона nrn
    pub fn jump_gaba_a(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_gaba_a(nrn, comp, g);
    }

    /// Скачок GABA_B проводимости на компартменте comp нейрона nrn
    pub fn jump_gaba_b(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_gaba_b(nrn, comp, g);
    }

    /// Скачок nAChR (ацетилхолин — главное возбуждение у насекомых) проводимости
    pub fn jump_nachr(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_nachr(nrn, comp, g);
    }

    /// Скачок GluCl (глутамат — хлорное ТОРМОЖЕНИЕ у насекомых!) проводимости
    pub fn jump_glucl(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_glucl(nrn, comp, g);
    }

    /// Скачок mAChR проводимости
    pub fn jump_machr(&mut self, nrn: usize, comp: usize, g: f32) {
        self.inner.jump_machr(nrn, comp, g);
    }

    /// Соматический мембранный потенциал [n]
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, &self.inner.v))
    }

    /// Выходные спайки [n] (0 эбэтэр 1)
    #[getter]
    pub fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice(py, &self.inner.spikes))
    }

    /// Потенциал компартмента `comp` нейрона `nrn`
    pub fn v_dend(&self, nrn: usize, comp: usize) -> f32 {
        self.inner.v_dend(nrn, comp)
    }

    /// Дендритный ток в сому нейрона `nrn`
    pub fn dend_current(&self, nrn: usize) -> f32 {
        self.inner.dend_current(nrn)
    }
}

/// Eligibility Propagation (e-prop) SNN learning step.
///
/// Updates eligibility traces and synaptic weights based on local surrogate gradient
/// and target output errors (Bellec et al., Nature 2020).
#[pyfunction]
pub fn step_eprop(
    weights: &Bound<'_, PyArray1<f32>>,
    eligibility: &Bound<'_, PyArray1<f32>>,
    v_post: &Bound<'_, PyArray1<f32>>,
    spikes_pre: &Bound<'_, PyArray1<i32>>,
    errors_post: &Bound<'_, PyArray1<f32>>,
    n_post: usize,
    n_pre: usize,
    alpha_e: f32,
    gamma: f32,
    v_thresh: f32,
    eta: f32,
) -> PyResult<()> {
    let weights_arr = weights.readwrite().as_slice_mut()?;
    let eligibility_arr = eligibility.readwrite().as_slice_mut()?;
    let v_post_arr = v_post.readonly().as_slice()?;
    let spikes_pre_arr = spikes_pre.readonly().as_slice()?;
    let errors_post_arr = errors_post.readonly().as_slice()?;

    assert_eq!(
        weights_arr.len(),
        n_post * n_pre,
        "step_eprop: weights len mismatch"
    );
    assert_eq!(
        eligibility_arr.len(),
        n_post * n_pre,
        "step_eprop: eligibility len mismatch"
    );
    assert_eq!(v_post_arr.len(), n_post, "step_eprop: v_post len mismatch");
    assert_eq!(
        spikes_pre_arr.len(),
        n_pre,
        "step_eprop: spikes_pre len mismatch"
    );
    assert_eq!(
        errors_post_arr.len(),
        n_post,
        "step_eprop: errors_post len mismatch"
    );

    for i in 0..n_post {
        let v_diff = (v_post_arr[i] - v_thresh).abs();
        let surrogate_grad_i = gamma / (1.0 + v_diff);
        let err_i = errors_post_arr[i];

        let row_off = i * n_pre;
        for j in 0..n_pre {
            let idx = row_off + j;
            let s_j = spikes_pre_arr[j] as f32;

            let e_val = alpha_e * eligibility_arr[idx] + surrogate_grad_i * s_j;
            eligibility_arr[idx] = e_val;

            let delta_w = -eta * err_i * e_val;
            weights_arr[idx] += delta_w;
        }
    }

    Ok(())
}
