//! PyO3 bindings for Kuramoto / T3 step functions and decoders.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::models::kuramoto;

// ─── Kuramoto bindings ──────────────────────────────────

/// Разреженный шаг Курамото. Модифицирует фазы in-place.
///
/// dθ/dt = ω + Σ_j W_ij·sin(θ_j - θ_i)
/// для разреженной матрицы связности (CSR).
#[pyfunction]
#[pyo3(name = "kuramoto_step_sparse")]
pub fn py_kuramoto_step_sparse(
    phases: Bound<'_, PyArray1<f32>>,
    omegas: Bound<'_, PyArray1<f32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    weights: Bound<'_, PyArray1<f32>>,
    dt: f32,
) -> PyResult<()> {
    let n = phases.len()?;
    let mut phases = phases.readwrite();
    let omegas = omegas.readonly();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let weights = weights.readonly();

    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];
    let mut coupling_buf = vec![0.0f32; n];

    kuramoto::kuramoto_step_sparse(
        phases.as_slice_mut()?,
        omegas.as_slice()?,
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        weights.as_slice()?,
        &mut sin_buf,
        &mut cos_buf,
        &mut coupling_buf,
        dt,
    );
    Ok(())
}

/// Разреженный шаг T3 (Kuramoto + Stuart-Landau).
///
/// Фазовая динамика + амплитудная динамика на разреженной матрице.
#[pyfunction]
#[pyo3(name = "t3_step_sparse")]
#[allow(clippy::too_many_arguments)]
pub fn py_t3_step_sparse(
    phases: Bound<'_, PyArray1<f32>>,
    amplitudes: Bound<'_, PyArray1<f32>>,
    omegas: Bound<'_, PyArray1<f32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    weights: Bound<'_, PyArray1<f32>>,
    drives: Bound<'_, PyArray1<f32>>,
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
) -> PyResult<()> {
    let n = phases.len()?;
    let mut phases = phases.readwrite();
    let mut amplitudes = amplitudes.readwrite();
    let omegas = omegas.readonly();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let weights = weights.readonly();
    let drives = drives.readonly();

    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];
    let mut coupling_buf = vec![0.0f32; n];

    kuramoto::t3_step_sparse(
        phases.as_slice_mut()?,
        amplitudes.as_slice_mut()?,
        omegas.as_slice()?,
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        weights.as_slice()?,
        drives.as_slice()?,
        &mut sin_buf,
        &mut cos_buf,
        &mut coupling_buf,
        dt,
        decay,
        alpha,
        gain,
    );
    Ok(())
}

/// Интерференция между двумя массивами фаз.
///
/// output[i] = sin(ph_a[i]) + sin(ph_b[i])
#[pyfunction]
#[pyo3(name = "interference")]
pub fn py_interference(
    output: Bound<'_, PyArray1<f32>>,
    phases_a: Bound<'_, PyArray1<f32>>,
    phases_b: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let phases_a = phases_a.readonly();
    let phases_b = phases_b.readonly();
    kuramoto::interference(output.as_slice_mut()?, phases_a.as_slice()?, phases_b.as_slice()?);
    Ok(())
}

/// Средняя интерференция — сглаженная версия interference.
#[pyfunction]
#[pyo3(name = "interference_mean")]
pub fn py_interference_mean(
    output: Bound<'_, PyArray1<f32>>,
    phases_a: Bound<'_, PyArray1<f32>>,
    phases_b: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let phases_a = phases_a.readonly();
    let phases_b = phases_b.readonly();
    kuramoto::interference_mean(output.as_slice_mut()?, phases_a.as_slice()?, phases_b.as_slice()?);
    Ok(())
}

/// Пакетный sweep когерентности по областям.
#[pyfunction]
#[pyo3(name = "batch_coherence_sweep")]
pub fn py_batch_coherence_sweep(
    output: Bound<'_, PyArray1<f32>>,
    areas: Bound<'_, PyArray1<f32>>,
    areas_sin: Bound<'_, PyArray1<f32>>,
    areas_cos: Bound<'_, PyArray1<f32>>,
    n_areas: usize,
    n_elem: usize,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let areas = areas.readonly();
    let areas_sin = areas_sin.readonly();
    let areas_cos = areas_cos.readonly();
    kuramoto::batch_coherence_sweep(
        output.as_slice_mut()?,
        areas.as_slice()?,
        areas_sin.as_slice()?,
        areas_cos.as_slice()?,
        n_areas, n_elem,
    );
    Ok(())
}

/// Фазовая когерентность: |⟨e^{iθ}⟩|.
#[pyfunction]
#[pyo3(name = "phase_coherence")]
pub fn py_phase_coherence(
    output: Bound<'_, PyArray1<f32>>,
    sin_buf: Bound<'_, PyArray1<f32>>,
    cos_buf: Bound<'_, PyArray1<f32>>,
    amplitudes: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let sin_buf = sin_buf.readonly();
    let cos_buf = cos_buf.readonly();
    let amplitudes = amplitudes.readonly();
    kuramoto::phase_coherence(
        output.as_slice_mut()?,
        sin_buf.as_slice()?,
        cos_buf.as_slice()?,
        amplitudes.as_slice()?,
    );
    Ok(())
}

/// Энергия в частотных полосах (декомпозиция).
#[pyfunction]
#[pyo3(name = "band_energy")]
pub fn py_band_energy(
    band_energies: Bound<'_, PyArray1<f32>>,
    amplitudes: Bound<'_, PyArray1<f32>>,
    omegas: Bound<'_, PyArray1<f32>>,
    band_edges: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut band_energies = band_energies.readwrite();
    let amplitudes = amplitudes.readonly();
    let omegas = omegas.readonly();
    let band_edges = band_edges.readonly();
    kuramoto::band_energy(
        band_energies.as_slice_mut()?,
        amplitudes.as_slice()?,
        omegas.as_slice()?,
        band_edges.as_slice()?,
    );
    Ok(())
}

/// Ленивое обновление sin/cos — пересчитывает только при изменении фазы.
#[pyfunction]
#[pyo3(name = "lazy_sincos_update")]
pub fn py_lazy_sincos_update(
    sin_out: Bound<'_, PyArray1<f32>>,
    cos_out: Bound<'_, PyArray1<f32>>,
    phases: Bound<'_, PyArray1<f32>>,
    phases_prev: Bound<'_, PyArray1<f32>>,
    eps: f32,
) -> PyResult<()> {
    let mut sin_out = sin_out.readwrite();
    let mut cos_out = cos_out.readwrite();
    let phases = phases.readonly();
    let mut phases_prev = phases_prev.readwrite();
    kuramoto::lazy_sincos_update(
        sin_out.as_slice_mut()?,
        cos_out.as_slice_mut()?,
        phases.as_slice()?,
        phases_prev.as_slice_mut()?,
        eps,
    );
    Ok(())
}

/// Обновление фильтра Калмана: веса + ковариация по ошибке.
#[pyfunction]
#[pyo3(name = "kalman_update")]
pub fn py_kalman_update(
    weights: Bound<'_, PyArray1<f32>>,
    cov: Bound<'_, PyArray1<f32>>,
    states: Bound<'_, PyArray1<f32>>,
    error: f32,
    s: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let mut cov = cov.readwrite();
    let states = states.readonly();
    kuramoto::kalman_update(weights.as_slice_mut()?, cov.as_slice_mut()?, states.as_slice()?, error, s);
    Ok(())
}

/// Adam оптимизатор: обновление весов с адаптивным моментом.
///
/// Стандартный Adam с beta1, beta2, bias correction.
#[pyfunction]
#[pyo3(name = "adam_train")]
pub fn py_adam_train(
    weights: Bound<'_, PyArray1<f32>>,
    m: Bound<'_, PyArray1<f32>>,
    v: Bound<'_, PyArray1<f32>>,
    states: Bound<'_, PyArray1<f32>>,
    error: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bc1: f32,
    bc2: f32,
    eps: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let mut m = m.readwrite();
    let mut v = v.readwrite();
    let states = states.readonly();
    kuramoto::adam_train(
        weights.as_slice_mut()?, m.as_slice_mut()?, v.as_slice_mut()?,
        states.as_slice()?, error, lr, beta1, beta2, bc1, bc2, eps,
    );
    Ok(())
}

/// NLMS (Normalized Least Mean Squares) обучение.
///
/// Адаптивный фильтр: w += lr * error * states / (||states||² + eps).
#[pyfunction]
#[pyo3(name = "nlms_train")]
pub fn py_nlms_train(
    weights: Bound<'_, PyArray1<f32>>,
    bias: Bound<'_, PyArray1<f32>>,
    states: Bound<'_, PyArray1<f32>>,
    error: f32,
    lr: f32,
    eps: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let mut bias = bias.readwrite();
    let states = states.readonly();
    kuramoto::nlms_train(weights.as_slice_mut()?, bias.as_slice_mut()?, states.as_slice()?, error, lr, eps);
    Ok(())
}

/// Декодирование PST: output = W·buffer + bias.
#[pyfunction]
#[pyo3(name = "pst_decode")]
pub fn py_pst_decode(
    output: Bound<'_, PyArray1<f32>>,
    buffer: Bound<'_, PyArray1<f32>>,
    weights: Bound<'_, PyArray1<f32>>,
    feat_dim: usize,
    L: usize,
    pos: usize,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let buffer = buffer.readonly();
    let weights = weights.readonly();
    kuramoto::pst_decode(
        output.as_slice_mut()?, buffer.as_slice()?, weights.as_slice()?,
        feat_dim, L, pos,
    );
    Ok(())
}

/// Кольцевой синтез: реконструкция сигнала из амплитуд и тригонометрии.
#[pyfunction]
#[pyo3(name = "ring_synthesis")]
pub fn py_ring_synthesis(
    output: Bound<'_, PyArray1<f32>>,
    amplitudes: Bound<'_, PyArray1<f32>>,
    trig_buf: Bound<'_, PyArray1<f32>>,
    weights: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let amplitudes = amplitudes.readonly();
    let trig_buf = trig_buf.readonly();
    let weights = weights.readonly();
    kuramoto::ring_synthesis(output.as_slice_mut()?, amplitudes.as_slice()?, trig_buf.as_slice()?, weights.as_slice()?);
    Ok(())
}

/// Матрица намотки: разность фаз между двумя популяциями.
#[pyfunction]
#[pyo3(name = "winding_matrix")]
pub fn py_winding_matrix(
    output: Bound<'_, PyArray1<f32>>,
    phases_a: Bound<'_, PyArray1<f32>>,
    phases_b: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let phases_a = phases_a.readonly();
    let phases_b = phases_b.readonly();
    kuramoto::winding_matrix(output.as_slice_mut()?, phases_a.as_slice()?, phases_b.as_slice()?);
    Ok(())
}
