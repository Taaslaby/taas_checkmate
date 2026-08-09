//! PyO3 bindings for sparse/CSR operations + fused sim_step.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use pillars::api::neurons as neuron;
use pillars::kernels::csr as sparse;

// ─── Sparse bindings ────────────────────────────────────

/// Вычислить гамильтониан GCO: H_real = W·u, H_imag = -W·v.
///
/// Для разреженной матрицы связности (CSR).
/// H·ψ = W·(u + i·v) — гамильтониан для coupled oscillator на графе.
///
/// Args:
///     row_ptr, col_indices, values: CSR матрица связности
///     v, u: действительная и мнимая части
///     h_real, h_imag: выход (будет заполнен)
#[pyfunction]
#[pyo3(name = "compute_hamiltonian")]
pub fn py_compute_hamiltonian(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    h_real: Bound<'_, PyArray1<f32>>,
    h_imag: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();
    let v = v.readonly();
    let u = u.readonly();
    let mut h_real = h_real.readwrite();
    let mut h_imag = h_imag.readwrite();
    sparse::compute_hamiltonian(
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        values.as_slice()?,
        v.as_slice()?,
        u.as_slice()?,
        h_real.as_slice_mut()?,
        h_imag.as_slice_mut()?,
    );
    Ok(())
}

/// Полный GCO шаг: Hψ + L2 нормализация, один проход.
///
/// Включает вычисление гамильтониана (CSR × ψ) и L2 нормализацию в одном вызове.
///
/// Args:
///     v, u: состояние осциллятора (будет изменён)
///     row_ptr, col_indices, values: CSR матрица
///     dt: шаг по времени
///     scale: фактор масштабирования времени-энергии
#[pyfunction]
#[pyo3(name = "step_gco_full")]
pub fn py_step_gco_full(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    dt: f32,
    tau: f32,
) -> PyResult<()> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();
    sparse::step_gco_full(
        v.as_slice_mut()?,
        u.as_slice_mut()?,
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        values.as_slice()?,
        dt,
        tau,
    );
    Ok(())
}

/// Распространить спайки через CSR синаптическую матрицу.
///
/// Для каждого спайка: I_buf[post] += W * nt_type.
/// Поддерживает тормозные/возбуждающие нейроны через nt_types.
///
/// Args:
///     row_ptr, col_indices, values: CSR матрица весов
///     spiked_indices: индексы заспайкавших нейронов
///     nt_types: тип нейромедиатора (+1 возб., -1 торм.)
///     input_buffer: буфер тока (будет увеличен)
///     weight_scale: масштаб весов
#[pyfunction]
#[pyo3(name = "propagate_spikes")]
pub fn py_propagate_spikes(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    spiked_indices: Bound<'_, PyArray1<i32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    input_buffer: Bound<'_, PyArray1<f32>>,
    weight_scale: f32,
) -> PyResult<()> {
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();
    let spiked_indices = spiked_indices.readonly();
    let nt_types = nt_types.readonly();
    let mut input_buffer = input_buffer.readwrite();
    sparse::propagate_spikes(
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        values.as_slice()?,
        spiked_indices.as_slice()?,
        nt_types.as_slice()?,
        input_buffer.as_slice_mut()?,
        weight_scale,
    );
    Ok(())
}

/// Применить STDP LTD к разреженной матрице весов.
///
/// Для каждого спайка: weights -= lr * traces[post].
#[pyfunction]
#[pyo3(name = "apply_stdp_ltd")]
pub fn py_apply_stdp_ltd(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    weights: Bound<'_, PyArray1<f32>>,
    spiked_indices: Bound<'_, PyArray1<i32>>,
    traces: Bound<'_, PyArray1<f32>>,
    learning_rate: f32,
) -> PyResult<()> {
    sparse::apply_stdp_ltd(
        row_ptr.readonly().as_slice()?,
        col_indices.readonly().as_slice()?,
        weights.readwrite().as_slice_mut()?,
        spiked_indices.readonly().as_slice()?,
        traces.readonly().as_slice()?,
        learning_rate,
    );
    Ok(())
}

/// Rate-based step: scatter SpMV (W^T @ r) + clamp + exponential Euler.
///
/// Один шаг rate-based динамики на полном коннектоме.
/// Заменяет scipy SpMV в FlyWireRateBrain на быстрый Rust.
///
/// Использует pre-allocated I_buffer для избегания аллокаций. I_buffer будет занулён внутри.
///
/// Args:
///     r: rates [n] (будет изменён)
///     I_buffer: scratch буфер [n] (user-managed, будет занулён)
///     row_ptr, col_indices, values: CSR матрица весов
///     nt_types: тип нейромедиатора (+1 возб, -1 торм) per pre-neuron [n]
///     external_I: внешний ток [n]
///     decay: exp(-dt/tau)
///     inv_decay: 1 - decay
///
/// Формула:
///     I[post] += W[pre,post] * r[pre] * nt_types[pre]  (scatter SpMV)
///     r = r * decay + clamp(I + I_ext, 0, 1) * (1 - decay)
#[pyfunction]
#[pyo3(name = "rate_step")]
pub fn py_rate_step(
    r: Bound<'_, PyArray1<f32>>,
    I_buffer: Bound<'_, PyArray1<f32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    decay: f32,
    inv_decay: f32,
) -> PyResult<()> {
    let mut r = r.readwrite();
    let mut I_buffer = I_buffer.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();
    let nt_types = nt_types.readonly();
    let external_I = external_I.readonly();
    sparse::rate_step(
        r.as_slice_mut()?,
        I_buffer.as_slice_mut()?,
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        values.as_slice()?,
        nt_types.as_slice()?,
        external_I.as_slice()?,
        decay,
        inv_decay,
    );
    Ok(())
}

// ─── Combined sim_step ────────────────────────────────

/// Полный шаг симуляции: inject + Izhikevich + propagate spikes.
///
/// Объединяет: установка внешнего тока → Izhikevich → сбор спайков →
/// распространение через CSR → подготовка к следующему шагу.
///
/// Returns: количество спайков на этом шаге.
#[pyfunction]
#[pyo3(name = "sim_step")]
pub fn py_sim_step(
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    I_buf: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    weights: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    step_idx: usize,
    dt: f32,
    trace_decay: f32,
    n: usize,
) -> PyResult<usize> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let a = a.readonly();
    let b = b.readonly();
    let c = c.readonly();
    let d = d.readonly();
    let mut I_buf = I_buf.readwrite();
    let mut spikes = spikes.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let weights = weights.readonly();
    let nt_types = nt_types.readonly();
    let external_I = external_I.readonly();

    let v = v.as_slice_mut()?;
    let u = u.as_slice_mut()?;
    let trace = trace.as_slice_mut()?;
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    let c = c.as_slice()?;
    let d = d.as_slice()?;
    let I_buf = I_buf.as_slice_mut()?;
    let spikes = spikes.as_slice_mut()?;
    let row_ptr = row_ptr.as_slice()?;
    let col_indices = col_indices.as_slice()?;
    let weights = weights.as_slice()?;
    let nt_types = nt_types.as_slice()?;
    let external_I = external_I.as_slice()?;

    // 1. Izhikevich step (on I_buf which includes propagated current from previous step)
    let mut refractory = vec![0.0f32; n];
    neuron::step_izhikevich(
        v,
        u,
        trace,
        a,
        b,
        I_buf,
        dt,
        trace_decay,
        spikes,
        c,
        d,
        &mut refractory,
        neuron::DEFAULT_TAU_REF,
    );

    // 2. Extract spikes and count
    let mut total_spikes: usize = 0;
    let n_neurons = row_ptr.len() - 1;
    let max_syn = col_indices.len().min(weights.len());
    for i in 0..n.min(spikes.len()) {
        if spikes[i] != 0 {
            total_spikes += 1;
            // Direct row_ptr indexing — row_ptr values are SYNAPSE indices (up to 5.3M), not neuron indices
            let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
            let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);
            let mult = nt_types[i];
            let mut j = start;
            while j + 4 <= end {
                let t0 = sparse::bound_col(col_indices, j, n_neurons);
                let t1 = sparse::bound_col(col_indices, j + 1, n_neurons);
                let t2 = sparse::bound_col(col_indices, j + 2, n_neurons);
                let t3 = sparse::bound_col(col_indices, j + 3, n_neurons);
                I_buf[t0] += weights[j] * mult;
                I_buf[t1] += weights[j + 1] * mult;
                I_buf[t2] += weights[j + 2] * mult;
                I_buf[t3] += weights[j + 3] * mult;
                j += 4;
            }
            for j in j..end {
                I_buf[sparse::bound_col(col_indices, j, n_neurons)] += weights[j] * mult;
            }
        }
    }

    // 3. Reset I_buf to external_I * modulation (for next step's izhikevich)
    let modulation = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * step_idx as f32 / 100.0).sin();
    for i in 0..n.min(external_I.len()) {
        I_buf[i] = external_I[i] * modulation;
    }
    for i in n..I_buf.len() {
        I_buf[i] = 0.0;
    }

    Ok(total_spikes)
}
