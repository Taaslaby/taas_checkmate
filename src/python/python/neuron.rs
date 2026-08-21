//! PyO3 bindings for neuron step functions.
//!
//! Параметры нейронов (`tau_m`, `v_rest`, `a`, `b`, `c`, …) принимают
//! **скаляр или массив**: float → бродкаст на всех нейронов, np.ndarray[n]
//! → per-neuron значения, np.ndarray[1] → бродкаст. Все state-массивы
//! (v, u, I, spikes, …) — numpy arrays (zero-copy).
//! step_* возвращают массив спайков (int32) для удобного chaining.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::cell::RefCell;

use pillars::api::neurons as neuron;
use pillars::api::neurons::{DendriteConfig, SomatoDendriticNeuron};

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

/// Параметр «скаляр или массив» — принимает numpy-массив/список длины `n`,
/// длины 1 (бродкаст на всех нейронов) или обычный скаляр (бродкаст).
/// Принимает и float32, и float64 (numpy-дефолт) — f64 кастуется в f32.
/// `None` → `default` для всех нейронов (дефолт сигнатуры).
/// Возвращает owned `Vec<f32>` длины `n`.
fn param_f32(
    obj: Option<&Bound<'_, PyAny>>,
    n: usize,
    name: &str,
    default: f32,
) -> PyResult<Vec<f32>> {
    let obj = match obj {
        Some(o) => o,
        None => return Ok(vec![default; n]),
    };
    // Точный float32 массив/список.
    if let Ok(v) = obj.extract::<Vec<f32>>() {
        if v.len() == n {
            return Ok(v);
        }
        if v.len() == 1 {
            return Ok(vec![v[0]; n]);
        }
        return Err(len_err(name, n, v.len()));
    }
    // float64 массив/список (numpy-дефолт) — кастуем в f32.
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() == n {
            return Ok(v.into_iter().map(|x| x as f32).collect());
        }
        if v.len() == 1 {
            return Ok(vec![v[0] as f32; n]);
        }
        return Err(len_err(name, n, v.len()));
    }
    if let Ok(s) = obj.extract::<f32>() {
        return Ok(vec![s; n]);
    }
    if let Ok(s) = obj.extract::<f64>() {
        return Ok(vec![s as f32; n]);
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "{name}: expected np.ndarray of length {n} (float32/float64), a length-1 array, or a scalar"
    )))
}

fn len_err(name: &str, n: usize, got: usize) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{name}: expected array of length {n} (or 1 to broadcast, or a scalar), got {got}"
    ))
}

/// SIMD-кернелы pillars требуют длину популяции, кратную 8 (LANES).
/// Вместо паники в assert — понятная PyValueError.
/// Legacy check — the SIMD runner internally handles non-multiple-of-8 n
/// via scalar tail (n8 = n & !7 then tail loop). NeuralPopulation also
/// auto-pads. Kept as a no-op for backward compat; callers that used it
/// continue to compile without changes.
#[inline(always)]
fn check_lanes(_n: usize, _fname: &str) -> PyResult<()> {
    Ok(())
}

// ─── Neuron bindings ─────────────────────────────────────

/// Izhikevich neuron — один шаг симуляции.
///
/// Модифицирует входящие numpy массивы in-place и возвращает спайки.
///
/// Args:
///     v: мембранный потенциал [n]
///     u: recovery variable [n]
///     trace: след для STDP [n]
///     a: timescale recovery — скаляр или массив [n]
///     b: sensitivity recovery — скаляр или массив [n]
///     I: ток [n] (будет изменён!)
///     dt: шаг по времени
///     trace_decay: коэффициент затухания следа
///     spikes: спайки (int32, будет заполнен 0/1) [n]
///     c: потенциал после спайка — скаляр или массив [n]
///     d: recovery после спайка — скаляр или массив [n]
///
/// Returns: спайки (int32) [n]. Все изменения state-массивов in-place.
#[pyfunction]
#[pyo3(name = "step_izhikevich")]
#[pyo3(signature = (v, u, trace, a, b, I, dt, trace_decay, spikes, c=None, d=None))]
pub fn py_step_izhikevich<'py>(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Option<Bound<'_, PyAny>>,
    b: Option<Bound<'_, PyAny>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    trace_decay: f32,
    spikes: Bound<'py, PyArray1<i32>>,
    c: Option<Bound<'_, PyAny>>,
    d: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let mut I = I.readwrite();
    let mut spikes_rw = spikes.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_izhikevich")?;
    let a = param_f32(a.as_ref(), n, "a", 0.02)?;
    let b = param_f32(b.as_ref(), n, "b", 0.2)?;
    let c = param_f32(c.as_ref(), n, "c", -65.0)?;
    let d = param_f32(d.as_ref(), n, "d", 8.0)?;
    let mut refractory = with_refractory(n);

    neuron::step_izhikevich(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        trace.as_slice_mut()?,
        &a,
        &b,
        I.as_slice_mut()?,
        dt,
        trace_decay,
        spikes_rw.as_slice_mut()?,
        &c,
        &d,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    drop(spikes_rw);
    Ok(spikes)
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
#[pyo3(signature = (v, u, trace, a, b, I, dt, trace_decay, noise_amp, spikes))]
pub fn py_step_saw<'py>(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Option<Bound<'_, PyAny>>,
    b: Option<Bound<'_, PyAny>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    spikes: Bound<'py, PyArray1<i32>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let I = I.readonly();
    let mut spikes_rw = spikes.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_saw")?;
    let a = param_f32(a.as_ref(), n, "a", 0.02)?;
    let b = param_f32(b.as_ref(), n, "b", 0.2)?;
    let mut refractory = with_refractory(n);

    neuron::step_saw(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        trace.as_slice_mut()?,
        &a,
        &b,
        I.as_slice()?,
        dt,
        trace_decay,
        noise_amp,
        spikes_rw.as_slice_mut()?,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    drop(spikes_rw);
    Ok(spikes)
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
#[pyo3(signature = (v, u, h_real, h_imag, I_ext, speed, dt, _tau))]
pub fn py_step_gco(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    h_real: Option<Bound<'_, PyAny>>,
    h_imag: Option<Bound<'_, PyAny>>,
    I_ext: Option<Bound<'_, PyAny>>,
    speed: Option<Bound<'_, PyAny>>,
    dt: f32,
    _tau: f32,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_gco")?;
    let h_real = param_f32(h_real.as_ref(), n, "h_real", 0.0)?;
    let h_imag = param_f32(h_imag.as_ref(), n, "h_imag", 0.0)?;
    let I_ext = param_f32(I_ext.as_ref(), n, "I_ext", 0.0)?;
    let speed = param_f32(speed.as_ref(), n, "speed", 0.0)?;

    pillars::kernels::phase::gco_step(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        &h_real,
        &h_imag,
        &I_ext,
        &speed,
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
///     tau_m: мембранная постоянная — скаляр или массив [n] (default 10.0)
///     v_rest: потенциал покоя — скаляр или массив [n] (default -70.0)
///     v_thresh: порог спайка — скаляр или массив [n] (default -55.0)
///     v_reset: потенциал после спайка — скаляр или массив [n] (default -75.0)
///
/// Returns: спайки (int32) [n]. Все изменения state-массивов in-place.
///
/// Пример: `step_lif(v, I, spikes, dt=1.0, tau_m=10.0)` — tau_m бродкастится.
#[pyfunction]
#[pyo3(name = "step_lif")]
#[pyo3(signature = (v, I, spikes, dt=1.0, tau_m=None, v_rest=None, v_thresh=None, v_reset=None))]
pub fn py_step_lif<'py>(
    v: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'py, PyArray1<i32>>,
    dt: f32,
    tau_m: Option<Bound<'_, PyAny>>,
    v_rest: Option<Bound<'_, PyAny>>,
    v_thresh: Option<Bound<'_, PyAny>>,
    v_reset: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut v = v.readwrite();
    let mut I = I.readwrite();
    let mut spikes_rw = spikes.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_lif")?;
    let tau_m = param_f32(tau_m.as_ref(), n, "tau_m", 10.0)?;
    let v_rest = param_f32(v_rest.as_ref(), n, "v_rest", -70.0)?;
    let v_thresh = param_f32(v_thresh.as_ref(), n, "v_thresh", -55.0)?;
    let v_reset = param_f32(v_reset.as_ref(), n, "v_reset", -75.0)?;
    let mut refractory = with_refractory(n);

    neuron::step_lif(
        v.as_slice_mut()?,
        I.as_slice_mut()?,
        spikes_rw.as_slice_mut()?,
        dt,
        &tau_m,
        &v_rest,
        &v_thresh,
        &v_reset,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    drop(spikes_rw);
    Ok(spikes)
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
#[pyo3(signature = (v, u, I, spikes, dt=1.0, tau_m=None, tau_w=None, v_rest=None, v_thresh=None, v_reset=None, delta_t=None, a=None, b=None))]
pub fn py_step_adex<'py>(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'py, PyArray1<i32>>,
    dt: f32,
    tau_m: Option<Bound<'_, PyAny>>,
    tau_w: Option<Bound<'_, PyAny>>,
    v_rest: Option<Bound<'_, PyAny>>,
    v_thresh: Option<Bound<'_, PyAny>>,
    v_reset: Option<Bound<'_, PyAny>>,
    delta_t: Option<Bound<'_, PyAny>>,
    a: Option<Bound<'_, PyAny>>,
    b: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut I = I.readwrite();
    let mut spikes_rw = spikes.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_adex")?;
    let tau_m = param_f32(tau_m.as_ref(), n, "tau_m", 20.0)?;
    let tau_w = param_f32(tau_w.as_ref(), n, "tau_w", 100.0)?;
    let v_rest = param_f32(v_rest.as_ref(), n, "v_rest", -65.0)?;
    let v_thresh = param_f32(v_thresh.as_ref(), n, "v_thresh", -50.0)?;
    let v_reset = param_f32(v_reset.as_ref(), n, "v_reset", -65.0)?;
    let delta_t = param_f32(delta_t.as_ref(), n, "delta_t", 2.0)?;
    let a = param_f32(a.as_ref(), n, "a", 0.002)?;
    let b = param_f32(b.as_ref(), n, "b", 0.05)?;
    let mut refractory = with_refractory(n);

    neuron::step_adex(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        I.as_slice_mut()?,
        spikes_rw.as_slice_mut()?,
        dt,
        &tau_m,
        &tau_w,
        &v_rest,
        &v_thresh,
        &v_reset,
        &delta_t,
        &a,
        &b,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );
    return_refractory(refractory);
    drop(spikes_rw);
    Ok(spikes)
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

    pillars::kernels::learning::stdp_update(
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

    pillars::kernels::learning::stdp_homeostatic(
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
#[pyo3(signature = (v, w, I, dt, tau_w, a, b, spikes))]
pub fn py_step_fhn<'py>(
    v: Bound<'_, PyArray1<f32>>,
    w: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    dt: f32,
    tau_w: Option<Bound<'_, PyAny>>,
    a: Option<Bound<'_, PyAny>>,
    b: Option<Bound<'_, PyAny>>,
    spikes: Bound<'py, PyArray1<i32>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut v = v.readwrite();
    let mut w = w.readwrite();
    let I = I.readonly();
    let mut spikes_rw = spikes.readwrite();

    let n = v.len()?;
    check_lanes(n, "step_fhn")?;
    let tau_w = param_f32(tau_w.as_ref(), n, "tau_w", 14.0)?;
    let a = param_f32(a.as_ref(), n, "a", 0.7)?;
    let b = param_f32(b.as_ref(), n, "b", 0.8)?;

    neuron::step_fhn(
        v.as_slice_mut()?,
        w.as_slice_mut()?,
        I.as_slice()?,
        spikes_rw.as_slice_mut()?,
        dt,
        &tau_w,
        &a,
        &b,
    );
    drop(spikes_rw);
    Ok(spikes)
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
#[pyo3(signature = (e, i, i_e, i_i, dt, tau_E, tau_I, w_EE, w_EI, w_IE, w_II, beta, spikes_e, spikes_i))]
pub fn py_step_wc(
    e: Bound<'_, PyArray1<f32>>,
    i: Bound<'_, PyArray1<f32>>,
    i_e: Option<Bound<'_, PyAny>>,
    i_i: Option<Bound<'_, PyAny>>,
    dt: f32,
    tau_E: Option<Bound<'_, PyAny>>,
    tau_I: Option<Bound<'_, PyAny>>,
    w_EE: Option<Bound<'_, PyAny>>,
    w_EI: Option<Bound<'_, PyAny>>,
    w_IE: Option<Bound<'_, PyAny>>,
    w_II: Option<Bound<'_, PyAny>>,
    beta: Option<Bound<'_, PyAny>>,
    spikes_e: Bound<'_, PyArray1<f32>>,
    spikes_i: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut e = e.readwrite();
    let mut i = i.readwrite();
    let mut spikes_e = spikes_e.readwrite();
    let mut spikes_i = spikes_i.readwrite();

    let n = e.len()?;
    check_lanes(n, "step_wc")?;
    let i_e = param_f32(i_e.as_ref(), n, "i_e", 0.0)?;
    let i_i = param_f32(i_i.as_ref(), n, "i_i", 0.0)?;
    let tau_E = param_f32(tau_E.as_ref(), n, "tau_E", 10.0)?;
    let tau_I = param_f32(tau_I.as_ref(), n, "tau_I", 10.0)?;
    let w_EE = param_f32(w_EE.as_ref(), n, "w_EE", 12.0)?;
    let w_EI = param_f32(w_EI.as_ref(), n, "w_EI", 4.0)?;
    let w_IE = param_f32(w_IE.as_ref(), n, "w_IE", 13.0)?;
    let w_II = param_f32(w_II.as_ref(), n, "w_II", 11.0)?;
    let beta = param_f32(beta.as_ref(), n, "beta", 1.0)?;

    neuron::step_wc(
        e.as_slice_mut()?,
        i.as_slice_mut()?,
        &i_e,
        &i_i,
        spikes_e.as_slice_mut()?,
        spikes_i.as_slice_mut()?,
        dt,
        &tau_E,
        &tau_I,
        &w_EE,
        &w_EI,
        &w_IE,
        &w_II,
        &beta,
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

// ═══════════════════════════════════════════════════════════
//  HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting)
// ═══════════════════════════════════════════════════════════

/// HAAP neuron — один шаг VNC модели с M-current + HCN адаптацией.
///
/// Уравнения:
///   C_m·dV/dt = -g_L(V-E_L) - g_M·u·(V-E_K) - g_h·h·(V-E_h) + I
///   τ_u·du/dt = (u_∞(V) - u)
///   τ_h·dh/dt = (h_∞(V) - h)
///
/// u_∞ = sigmoid((V - V_half_u) / k_u) — M-current, активируется деполяризацией
/// h_∞ = sigmoid((V_half_h - V) / k_h) — HCN, активируется гиперполяризацией
///
/// Args:
///     v: мембранный потенциал [n]
///     u_state: M-current гейтинг (0..1) [n]
///     h_state: HCN гейтинг (0..1) [n]
///     I: ток [n]
///     spikes: спайки (int32, будет заполнен) [n]
///     params: упакованные параметры (17 × n)
///         dt, C_m, g_L, E_L, g_M, E_K, g_h, E_h,
///         tau_u, tau_h, V_half_u, k_u, V_half_h, k_h,
///         V_thresh, V_reset, delta_u
///
/// Returns: None. Все изменения in-place.
#[pyfunction]
#[pyo3(name = "step_haap")]
pub fn py_step_haap(
    v: Bound<'_, PyArray1<f32>>,
    u_state: Bound<'_, PyArray1<f32>>,
    h_state: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    params: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u_state = u_state.readwrite();
    let mut h_state = h_state.readwrite();
    let I = I.readonly();
    let mut spikes = spikes.readwrite();
    let params = params.readonly();

    let n = v.len()?;
    assert_eq!(u_state.len()?, n, "u_state length mismatch");
    assert_eq!(h_state.len()?, n, "h_state length mismatch");
    assert_eq!(I.len()?, n, "I length mismatch");
    assert_eq!(spikes.len()?, n, "spikes length mismatch");
    assert_eq!(
        params.len()?,
        n * 17,
        "params length mismatch: expected {n}*17"
    );

    neuron::step_haap(
        v.as_slice_mut()?,
        u_state.as_slice_mut()?,
        h_state.as_slice_mut()?,
        I.as_slice()?,
        spikes.as_slice_mut()?,
        params.as_slice()?,
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════
//  ChordotonalSensor (Drosophila mechanoreceptor)
// ═══════════════════════════════════════════════════════════

/// PyO3 класс сенсорного нейрона (Chordotonal organ) — нагрузка → частота.
#[pyclass(name = "ChordotonalSensor", module = "taas_checkmate.neurons")]
pub struct PyChordotonalSensor {
    inner: neuron::ChordotonalState,
}

#[pymethods]
impl PyChordotonalSensor {
    /// Создать новый ChordotonalSensor.
    ///
    /// Args:
    ///     n: количество сенсоров (будет padded до ×8)
    ///     profile: "mixed" (default), "phasic", или "tonic"
    ///     dt: шаг по времени (мс)
    #[new]
    #[pyo3(signature = (n, profile="mixed".to_string(), dt=1.0))]
    pub fn new(n: usize, profile: String, dt: f32) -> Self {
        let prof = match profile.as_str() {
            "phasic" => neuron::ChordotonalProfile::phasic(),
            "tonic" => neuron::ChordotonalProfile::tonic(),
            _ => neuron::ChordotonalProfile::mixed(),
        };
        let inner = neuron::ChordotonalBuilder::new(n)
            .with_profile(prof)
            .with_dt(dt)
            .build();
        Self { inner }
    }

    /// Один шаг симуляции: нагрузка → скорость стрельбы.
    ///
    /// Args:
    ///     loads: механическая нагрузка/сила контакта, нормализованная 0..1 [n]
    ///     stretches: растяжение сустава, нормализованное -1..1 [n]
    ///     d_loads: опционально — производная нагрузки (скорость). None или массив [n]
    #[pyo3(signature = (loads, stretches, d_loads=None))]
    pub fn step(
        &mut self,
        loads: Bound<'_, PyArray1<f32>>,
        stretches: Bound<'_, PyArray1<f32>>,
        d_loads: Option<Bound<'_, PyArray1<f32>>>,
    ) -> PyResult<()> {
        let loads_ro = loads.readonly();
        let stretches_ro = stretches.readonly();

        let loads_slice = loads_ro.as_slice()?;
        let stretches_slice = stretches_ro.as_slice()?;

        // Copy d_loads to owned Vec to avoid borrow issues with PyReadonlyArray1
        let d_loads_owned: Option<Vec<f32>> = match &d_loads {
            Some(arr) => {
                let ro = arr.readonly();
                Some(ro.as_slice()?.to_vec())
            }
            None => None,
        };
        let d_slice: Option<&[f32]> = d_loads_owned.as_deref();

        self.inner.step(loads_slice, stretches_slice, d_slice);
        // d_loads_owned (and its borrow d_slice) dropped here
        Ok(())
    }

    /// Получить текущие firing rates (Hz) для всех сенсоров.
    #[getter]
    pub fn rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.firing_rates()))
    }

    /// Получить рецепторный потенциал V_sens.
    #[getter]
    pub fn v_sens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, &self.inner.v_sens[..self.inner.n]))
    }

    /// Количество сенсоров.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n
    }
}

// ═══════════════════════════════════════════════════════════
//  Spike Metrics — firing rates, ISI, CV
// ═══════════════════════════════════════════════════════════

/// Вычислить firing rates для каждого нейрона по буферу спайков.
///
/// Буфер: `[time × neuron]` — `spike_buffer[t * n_neurons + i]`.
/// Скорость: `rate[i] = spike_count[i] / (n_steps * dt)` в Hz.
///
/// Args:
///     spike_buffer: буфер спайков (int32) [n_neurons * n_steps]
///     n_neurons: количество нейронов
///     n_steps: количество временных шагов
///     dt: шаг по времени (ms)
///
/// Returns:
///     np.ndarray[float32] — firing rate в Hz для каждого нейрона
#[pyfunction]
#[pyo3(name = "firing_rates")]
pub fn py_firing_rates<'py>(
    py: Python<'py>,
    spike_buffer: Bound<'py, PyArray1<i32>>,
    n_neurons: usize,
    n_steps: usize,
    dt: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let spike_buffer = spike_buffer.readonly();
    let result = neuron::firing_rates(spike_buffer.as_slice()?, n_neurons, n_steps, dt);
    Ok(PyArray1::from_vec(py, result))
}

/// Средний firing rate по всем нейронам.
///
/// Args:
///     spike_buffer: буфер спайков (int32) [n_neurons * n_steps]
///     n_neurons: количество нейронов
///     n_steps: количество временных шагов
///     dt: шаг по времени (ms)
///
/// Returns:
///     float — средний firing rate в Hz
#[pyfunction]
#[pyo3(name = "mean_firing_rate")]
pub fn py_mean_firing_rate(
    spike_buffer: Bound<'_, PyArray1<i32>>,
    n_neurons: usize,
    n_steps: usize,
    dt: f32,
) -> PyResult<f32> {
    let spike_buffer = spike_buffer.readonly();
    Ok(neuron::mean_firing_rate(
        spike_buffer.as_slice()?,
        n_neurons,
        n_steps,
        dt,
    ))
}

/// Вычислить межспайковые интервалы (ISI) для одного нейрона.
///
/// Args:
///     spikes: последовательность спайков (int32) [n_steps] — 0/1
///     dt: шаг по времени (ms)
///
/// Returns:
///     np.ndarray[float32] — ISI в ms
#[pyfunction]
#[pyo3(name = "compute_isi")]
pub fn py_compute_isi<'py>(
    py: Python<'py>,
    spikes: Bound<'py, PyArray1<i32>>,
    dt: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let spikes = spikes.readonly();
    let result = neuron::compute_isi(spikes.as_slice()?, dt);
    Ok(PyArray1::from_vec(py, result))
}

/// Вычислить Coefficient of Variation (CV) ISIs для каждого нейрона.
///
/// CV ≈ 1 → Poisson, CV < 1 → regular, CV > 1 → bursting.
///
/// Args:
///     spike_buffer: буфер спайков (int32) [n_neurons * n_steps]
///     n_neurons: количество нейронов
///     n_steps: количество временных шагов
///     dt: шаг по времени (ms)
///
/// Returns:
///     np.ndarray[float32] — CV для каждого нейрона
#[pyfunction]
#[pyo3(name = "compute_cvs")]
pub fn py_compute_cvs<'py>(
    py: Python<'py>,
    spike_buffer: Bound<'py, PyArray1<i32>>,
    n_neurons: usize,
    n_steps: usize,
    dt: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let spike_buffer = spike_buffer.readonly();
    let result = neuron::compute_cvs(spike_buffer.as_slice()?, n_neurons, n_steps, dt);
    Ok(PyArray1::from_vec(py, result))
}

// ═══════════════════════════════════════════════════════════
//  🧫 AstrocyteState — glial modulation
// ═══════════════════════════════════════════════════════════

/// PyO3 класс астроцита: медленная глиальная модуляция синаптического усиления,
/// EAAT очистка глутамата, K⁺ буферизация, Ca²⁺ волны через gap junctions.
///
/// The full model runs in Rust SIMD via `crates/kernels/src/neuron/astrocyte.rs`.
#[pyclass(name = "AstrocyteState", module = "taas_checkmate.neurons")]
pub struct PyAstrocyteState {
    inner: std::sync::Mutex<neuron::AstrocyteState>,
}

#[pymethods]
impl PyAstrocyteState {
    /// Создать новую сеть астроцитов.
    ///
    /// Args:
    ///     n: количество астроцитов (обычно = n_neurons)
    ///     tau_astro: постоянная времени медленной активности (ms, default 1000)
    ///     alpha: сила модуляции синаптического усиления (default 0.1)
    #[new]
    #[pyo3(signature = (n, tau_astro=1000.0, alpha=0.1))]
    pub fn new(n: usize, tau_astro: f32, alpha: f32) -> Self {
        Self {
            inner: std::sync::Mutex::new(neuron::AstrocyteState::new(n, tau_astro, alpha)),
        }
    }

    /// Один шаг симуляции: обновить A(t), Ca²⁺, K_ext, EAAT_occ, V_gap.
    ///
    /// Args:
    ///     spikes: спайки (int32, 0/1) [n]
    ///     gap_pairs: список пар (i, j) для gap junction связей (list of 2-tuples)
    ///     dt: шаг по времени (ms)
    pub fn step(
        &self,
        spikes: Bound<'_, PyArray1<i32>>,
        gap_pairs: Vec<(usize, usize)>,
        dt: f32,
    ) -> PyResult<()> {
        let spikes_ro = spikes.readonly();
        let mut inner = self.inner.lock().unwrap();
        inner.step(spikes_ro.as_slice()?, &gap_pairs, dt);
        Ok(())
    }

    /// Вычислить модуляционное усиление для всех нейронов:
    /// ``gain[i] = (1 + α·A[i]) · (1 + μ·Ca[i])``
    ///
    /// Returns:
    ///     np.ndarray[float32] — gain для каждого астроцита
    pub fn compute_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.lock().unwrap();
        let gain = inner.compute_gain();
        PyArray1::from_vec(py, gain[..inner.n].to_vec())
    }

    /// Применить K⁺ усталость к порогам: ``V_thresh += sensitivity · (K_ext − K_baseline)``
    ///
    /// Args:
    ///     v_thresh: массив порогов [n] (будет изменён in-place)
    pub fn modulate_v_thresh(&self, v_thresh: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let mut v_thresh_rw = v_thresh.readwrite();
        let inner = self.inner.lock().unwrap();
        inner.modulate_v_thresh(v_thresh_rw.as_slice_mut()?);
        Ok(())
    }

    /// Применить EAAT очистку + синаптическое усиление к SynapticKinetics.
    ///
    /// Args:
    ///     g_ampa, g_nmda, g_gaba_a, g_gaba_b, g_nachr, g_glucl, g_machr:
    ///         проводимости [n] (будут изменены in-place)
    ///     dt: шаг по времени (ms)
    #[allow(clippy::too_many_arguments)]
    pub fn apply_to_kinetics(
        &self,
        g_ampa: Bound<'_, PyArray1<f32>>,
        g_nmda: Bound<'_, PyArray1<f32>>,
        g_gaba_a: Bound<'_, PyArray1<f32>>,
        g_gaba_b: Bound<'_, PyArray1<f32>>,
        g_nachr: Bound<'_, PyArray1<f32>>,
        g_glucl: Bound<'_, PyArray1<f32>>,
        g_machr: Bound<'_, PyArray1<f32>>,
        dt: f32,
    ) -> PyResult<()> {
        let mut a_ampa = g_ampa.readwrite();
        let mut a_nmda = g_nmda.readwrite();
        let mut a_gaba_a = g_gaba_a.readwrite();
        let mut a_gaba_b = g_gaba_b.readwrite();
        let mut a_nachr = g_nachr.readwrite();
        let mut a_glucl = g_glucl.readwrite();
        let mut a_machr = g_machr.readwrite();

        let inner = self.inner.lock().unwrap();
        let kin = &mut pillars::kernels::synapse::SynapticKinetics::new(inner.n);

        // Copy Python arrays into kinetics buffers
        let n = inner.n.min(a_ampa.len()?);
        kin.g_ampa[..n].copy_from_slice(&a_ampa.as_slice()?[..n]);
        kin.g_nmda[..n].copy_from_slice(&a_nmda.as_slice()?[..n]);
        kin.g_gaba_a[..n].copy_from_slice(&a_gaba_a.as_slice()?[..n]);
        kin.g_gaba_b[..n].copy_from_slice(&a_gaba_b.as_slice()?[..n]);
        kin.g_nachr[..n].copy_from_slice(&a_nachr.as_slice()?[..n]);
        kin.g_glucl[..n].copy_from_slice(&a_glucl.as_slice()?[..n]);
        kin.g_machr[..n].copy_from_slice(&a_machr.as_slice()?[..n]);

        // Apply astrocyte modulation
        inner.apply_to_kinetics(kin, dt);

        // Copy back
        a_ampa.as_slice_mut()?[..n].copy_from_slice(&kin.g_ampa[..n]);
        a_nmda.as_slice_mut()?[..n].copy_from_slice(&kin.g_nmda[..n]);
        a_gaba_a.as_slice_mut()?[..n].copy_from_slice(&kin.g_gaba_a[..n]);
        a_gaba_b.as_slice_mut()?[..n].copy_from_slice(&kin.g_gaba_b[..n]);
        a_nachr.as_slice_mut()?[..n].copy_from_slice(&kin.g_nachr[..n]);
        a_glucl.as_slice_mut()?[..n].copy_from_slice(&kin.g_glucl[..n]);
        a_machr.as_slice_mut()?[..n].copy_from_slice(&kin.g_machr[..n]);

        Ok(())
    }

    /// Применить volume transmission модуляцию к Ii: ``Ii[i] *= (1 + μ·Ca[i])⁻¹``
    /// или ``Ii[i] *= (1 + μ·Ca[i])`` если excitatory=True.
    ///
    /// Args:
    ///     Ii: массив токов [n] (будет изменён in-place)
    ///     excitatory: если True — усиливает Ii, если False — ослабляет
    pub fn modulate_Ii(&self, Ii: Bound<'_, PyArray1<f32>>, excitatory: bool) -> PyResult<()> {
        let mut Ii_rw = Ii.readwrite();
        let inner = self.inner.lock().unwrap();
        inner.modulate_Ii(Ii_rw.as_slice_mut()?, excitatory);
        Ok(())
    }

    /// Установить параметр k_per_spike.
    pub fn set_k_per_spike(&self, val: f32) {
        self.inner.lock().unwrap().k_per_spike = val;
    }
    /// Установить параметр k_sensitivity.
    pub fn set_k_sensitivity(&self, val: f32) {
        self.inner.lock().unwrap().k_sensitivity = val;
    }
    /// Установить параметр k_baseline.
    pub fn set_k_baseline(&self, val: f32) {
        self.inner.lock().unwrap().k_baseline = val;
    }
    /// Установить параметр eaat_strength.
    pub fn set_eaat_strength(&self, val: f32) {
        self.inner.lock().unwrap().eaat_strength = val;
    }
    /// Установить параметр mu (volume transmission gain).
    pub fn set_mu(&self, val: f32) {
        self.inner.lock().unwrap().mu = val;
    }
    /// Установить параметр g_gap (gap junction conductance).
    pub fn set_g_gap(&self, val: f32) {
        self.inner.lock().unwrap().g_gap = val;
    }

    /// Сбросить все состояния астроцита.
    pub fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }

    /// Текущие значения A(t) (медленная активность) для всех нейронов.
    #[getter]
    pub fn a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.lock().unwrap();
        PyArray1::from_vec(py, inner.a[..inner.n].to_vec())
    }

    /// Текущие значения Ca²⁺.
    #[getter]
    pub fn ca<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.lock().unwrap();
        PyArray1::from_vec(py, inner.ca[..inner.n].to_vec())
    }

    /// Текущие значения K_ext (внеклеточный K⁺).
    #[getter]
    pub fn k_ext<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.lock().unwrap();
        PyArray1::from_vec(py, inner.k_ext[..inner.n].to_vec())
    }

    /// Текущие значения EAAT_occ (занятость транспортера).
    #[getter]
    pub fn eaat_occ<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inner = self.inner.lock().unwrap();
        PyArray1::from_vec(py, inner.eaat_occ[..inner.n].to_vec())
    }

    /// Количество астроцитов.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.lock().unwrap().n
    }
}

// ═══════════════════════════════════════════════════════════
//  NMDA — Voltage-Dependent Mg²⁺ Block (Destexhe et al. 1998)
// ═══════════════════════════════════════════════════════════

/// NMDA Mg²⁺ block — вольт-зависимый магниевый блок для NMDA рецепторов.
///
/// I_NMDA = g_NMDA · B(V) · (V - E_syn)
/// B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062 · V))
///
/// Args:
///     v: мембранный потенциал [n]
///     g_act: NMDA проводимость [n]
///     i_out: выходной ток [n] (ДОБАВЛЯЕТСЯ in-place)
///     g_nmda: максимальная проводимость [n]
///     E_syn: потенциал реверсии (mV, default 0.0) [n]
///     mg_conc: концентрация Mg²⁺ (mM, default 1.0) [n]
#[pyfunction]
#[pyo3(name = "step_nmda")]
#[pyo3(signature = (v, g_act, i_out, g_nmda=None, E_syn=None, mg_conc=None))]
pub fn py_step_nmda<'py>(
    v: Bound<'_, PyArray1<f32>>,
    g_act: Bound<'_, PyArray1<f32>>,
    i_out: Bound<'py, PyArray1<f32>>,
    g_nmda: Option<Bound<'_, PyAny>>,
    E_syn: Option<Bound<'_, PyAny>>,
    mg_conc: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let v_ro = v.readonly();
    let g_act_ro = g_act.readonly();
    let mut i_out_rw = i_out.readwrite();

    let n = v.len()?;
    let v_slice = v_ro.as_slice()?;
    let g_act_slice = g_act_ro.as_slice()?;
    let i_out_slice = i_out_rw.as_slice_mut()?;
    let g_nmda = param_f32(g_nmda.as_ref(), n, "g_nmda", 1.0)?;
    let e_syn = param_f32(E_syn.as_ref(), n, "E_syn", 0.0)?;
    let mg_conc = param_f32(mg_conc.as_ref(), n, "mg_conc", 1.0)?;

    // Params layout [4 × n]: [g_nmda, E_syn, mg_conc, _pad]
    let mut params = vec![0.0f32; n * 4];
    for i in 0..n {
        params[i * 4] = g_nmda[i];
        params[i * 4 + 1] = e_syn[i];
        params[i * 4 + 2] = mg_conc[i];
    }

    unsafe {
        pillars::zoo::synapses::nmda_mg_block_step(
            v_slice.as_ptr(),
            g_act_slice.as_ptr(),
            i_out_slice.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
    drop(i_out_rw);
    Ok(i_out)
}

// ═══════════════════════════════════════════════════════════
//  GLIF3 — LIF + After-Spike Current (Allen Institute)
// ═══════════════════════════════════════════════════════════

/// GLIF3 — LIF с after-spike current (I_ahp).
///
/// Allen Institute — Teeter et al. 2018, Nature Communications.
///
/// Уравнения:
///   dv/dt = -g_leak·(v - E_leak) + I_syn + g_ahp·(v - E_ahp)
///   dg_ahp/dt = -g_ahp / τ_ahp  (exponential Euler)
///   On spike: v = v_reset, g_ahp += Δg_ahp
///
/// Использование:
///     glif = Glif3State(100, dt=1.0)
///     I_ext = np.zeros(100, dtype=np.float32)
///     glif.step(I_ext)
///     spikes = glif.spikes
#[pyclass(name = "Glif3State", module = "taas_checkmate.neurons")]
pub struct PyGlif3State {
    inner: std::sync::Mutex<neuron::Glif3State>,
}

#[pymethods]
impl PyGlif3State {
    #[new]
    #[pyo3(signature = (n, dt=1.0, g_leak=0.1, E_leak=-65.0, v_thresh=-50.0, v_reset=-65.0,
        tau_ahp=30.0, delta_g_ahp=0.01, E_ahp=-80.0, tau_ref=2.0))]
    fn new(
        n: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_ref: f32,
    ) -> Self {
        Self {
            inner: std::sync::Mutex::new(neuron::Glif3State::new(
                n,
                dt,
                g_leak,
                E_leak,
                v_thresh,
                v_reset,
                tau_ahp,
                delta_g_ahp,
                E_ahp,
                tau_ref,
            )),
        }
    }

    /// Один шаг симуляции с внешним током I_ext.
    fn step(&self, I_ext: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let I_ext_ro = I_ext.readonly();
        let mut inner = self.inner.lock().unwrap();
        inner.step(Some(I_ext_ro.as_slice()?));
        Ok(())
    }

    /// Сброс всех состояний.
    fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }

    /// Настройка параметров конкретного нейрона.
    #[pyo3(signature = (idx, g_leak, E_leak, v_thresh, v_reset, tau_ahp, delta_g_ahp, E_ahp))]
    fn set_params(
        &self,
        idx: usize,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
    ) {
        self.inner.lock().unwrap().set_params(
            idx,
            g_leak,
            E_leak,
            v_thresh,
            v_reset,
            tau_ahp,
            delta_g_ahp,
            E_ahp,
        );
    }

    /// Мембранный потенциал [n].
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.v[..inner.n].to_vec()))
    }

    /// Спайки (0/1) [n].
    #[getter]
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.spikes[..inner.n].to_vec()))
    }

    /// After-spike conductance g_ahp [n].
    #[getter]
    fn g_ahp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.g_ahp[..inner.n].to_vec()))
    }

    fn __repr__(&self) -> String {
        format!("Glif3State(n={})", self.inner.lock().unwrap().n)
    }

    fn n_neurons(&self) -> usize {
        self.inner.lock().unwrap().n
    }
}

// ═══════════════════════════════════════════════════════════
//  GLIF4 — Threshold Adaptation + After-Spike Current
// ═══════════════════════════════════════════════════════════

/// GLIF4 — GLIF3 + адаптация порога (θ).
///
/// Allen Institute — Teeter et al. 2018.
///
/// Уравнения:
///   dv/dt = -g_leak·(v - E_leak) + I_syn + g_ahp·(v - E_ahp)
///   dg_ahp/dt = -g_ahp / τ_ahp
///   dθ/dt = -θ / τ_θ
///   Effective threshold = v_thresh_0 + θ
///   On spike: v = v_reset, g_ahp += Δg_ahp, θ += Δθ
#[pyclass(name = "Glif4State", module = "taas_checkmate.neurons")]
pub struct PyGlif4State {
    inner: std::sync::Mutex<neuron::Glif4State>,
}

#[pymethods]
impl PyGlif4State {
    #[new]
    #[pyo3(signature = (n, dt=1.0, g_leak=0.1, E_leak=-65.0, v_thresh_0=-50.0, v_reset=-65.0,
        tau_ahp=30.0, delta_g_ahp=0.01, E_ahp=-80.0, tau_theta=20.0, delta_theta=3.0, tau_ref=2.0))]
    fn new(
        n: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
        tau_ref: f32,
    ) -> Self {
        Self {
            inner: std::sync::Mutex::new(neuron::Glif4State::new(
                n,
                dt,
                g_leak,
                E_leak,
                v_thresh_0,
                v_reset,
                tau_ahp,
                delta_g_ahp,
                E_ahp,
                tau_theta,
                delta_theta,
                tau_ref,
            )),
        }
    }

    /// Один шаг симуляции с внешним током I_ext.
    fn step(&self, I_ext: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let I_ext_ro = I_ext.readonly();
        let mut inner = self.inner.lock().unwrap();
        inner.step(Some(I_ext_ro.as_slice()?));
        Ok(())
    }

    /// Сброс всех состояний.
    fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }

    /// Мембранный потенциал [n].
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.v[..inner.n].to_vec()))
    }

    /// Спайки (0/1) [n].
    #[getter]
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.spikes[..inner.n].to_vec()))
    }

    /// Threshold offset θ [n].
    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.theta[..inner.n].to_vec()))
    }

    /// After-spike conductance g_ahp [n].
    #[getter]
    fn g_ahp<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.lock().unwrap();
        Ok(PyArray1::from_vec(py, inner.g_ahp[..inner.n].to_vec()))
    }

    fn __repr__(&self) -> String {
        format!("Glif4State(n={})", self.inner.lock().unwrap().n)
    }

    fn n_neurons(&self) -> usize {
        self.inner.lock().unwrap().n
    }
}

// ═══════════════════════════════════════════════════════════
//  Hindmarsh-Rose (HR) — 3-variable spiking & bursting
// ═══════════════════════════════════════════════════════════

/// Hindmarsh-Rose (HR) — один шаг 3-переменной модели спайкинга/бёрстинга.
///
/// Уравнения:
///   dx/dt = y - a·x³ + b·x² - z + I_ext
///   dy/dt = c - d·x² - y
///   dz/dt = r·(s·(x - x_R) - z)
///
/// Параметры (по умолчанию): a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, xr=-1.6
///
/// Args:
///     x: быстрая переменная (мембранный потенциал) [n]
///     y: recovery переменная [n]
///     z: медленная адаптация [n]
///     I: внешний ток [n]
///     spikes: спайки (int32) [n]
///     dt: шаг по времени (ms)
///     a, b, c, d, r, s, xr: параметры модели [n]
#[pyfunction]
#[pyo3(name = "step_hr")]
#[pyo3(signature = (x, y, z, I, spikes, dt=0.01, a=None, b=None, c=None, d=None, r=None, s=None, xr=None))]
pub fn py_step_hr<'py>(
    x: Bound<'_, PyArray1<f32>>,
    y: Bound<'_, PyArray1<f32>>,
    z: Bound<'_, PyArray1<f32>>,
    I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'py, PyArray1<i32>>,
    dt: f32,
    a: Option<Bound<'_, PyAny>>,
    b: Option<Bound<'_, PyAny>>,
    c: Option<Bound<'_, PyAny>>,
    d: Option<Bound<'_, PyAny>>,
    r: Option<Bound<'_, PyAny>>,
    s: Option<Bound<'_, PyAny>>,
    xr: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let mut x_rw = x.readwrite();
    let mut y_rw = y.readwrite();
    let mut z_rw = z.readwrite();
    let I_ro = I.readonly();
    let mut spikes_rw = spikes.readwrite();

    let n = x.len()?;
    check_lanes(n, "step_hr")?;
    let a = param_f32(a.as_ref(), n, "a", 1.0)?;
    let b = param_f32(b.as_ref(), n, "b", 3.0)?;
    let c = param_f32(c.as_ref(), n, "c", 1.0)?;
    let d = param_f32(d.as_ref(), n, "d", 5.0)?;
    let r = param_f32(r.as_ref(), n, "r", 0.006)?;
    let s = param_f32(s.as_ref(), n, "s", 4.0)?;
    let xr = param_f32(xr.as_ref(), n, "xr", -1.6)?;

    neuron::step_hr(
        x_rw.as_slice_mut()?,
        y_rw.as_slice_mut()?,
        z_rw.as_slice_mut()?,
        I_ro.as_slice()?,
        spikes_rw.as_slice_mut()?,
        dt,
        &a,
        &b,
        &c,
        &d,
        &r,
        &s,
        &xr,
    );
    drop(spikes_rw);
    Ok(spikes)
}

// ═══════════════════════════════════════════════════════════
//  e-prop
// ═══════════════════════════════════════════════════════════

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
    let mut weights_rw = weights.readwrite();
    let weights_arr = weights_rw.as_slice_mut()?;
    let mut elig_rw = eligibility.readwrite();
    let eligibility_arr = elig_rw.as_slice_mut()?;
    let v_post_ro = v_post.readonly();
    let v_post_arr = v_post_ro.as_slice()?;
    let spikes_pre_ro = spikes_pre.readonly();
    let spikes_pre_arr = spikes_pre_ro.as_slice()?;
    let errors_post_ro = errors_post.readonly();
    let errors_post_arr = errors_post_ro.as_slice()?;

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
