//! PyO3 bindings for IO gates — encoder, decoder, gate functions.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::io;

// ─── IO gate bindings ────────────────────────────────────

/// Безопасная клипса: обрезает data до [min_val, max_val] in-place.
#[pyfunction]
#[pyo3(name = "safety_clamp")]
pub fn py_safety_clamp(
    data: Bound<'_, PyArray1<f32>>,
    min_val: f32,
    max_val: f32,
) -> PyResult<()> {
    let mut data = data.readwrite();
    io::safety_clamp(data.as_slice_mut()?, min_val, max_val);
    Ok(())
}

/// Сигмоидный гейт: output = sigmoid(gain * (input - threshold)).
#[pyfunction]
#[pyo3(name = "sigmoid_gate")]
pub fn py_sigmoid_gate(
    output: Bound<'_, PyArray1<f32>>,
    input: Bound<'_, PyArray1<f32>>,
    gain: f32,
    threshold: f32,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let input = input.readonly();
    io::sigmoid_gate(output.as_slice_mut()?, input.as_slice()?, gain, threshold);
    Ok(())
}

/// Фазовый гейт: output = input * sigmoid(gain * (r - threshold)).
///
/// Модулирует вход амплитудой: чем выше r, тем больше сигнал пропускается.
#[pyfunction]
#[pyo3(name = "phase_gate_apply")]
pub fn py_phase_gate_apply(
    output: Bound<'_, PyArray1<f32>>,
    input: Bound<'_, PyArray1<f32>>,
    r: Bound<'_, PyArray1<f32>>,
    gain: f32,
    threshold: f32,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let input = input.readonly();
    let r = r.readonly();
    io::phase_gate_apply(output.as_slice_mut()?, input.as_slice()?, r.as_slice()?, gain, threshold);
    Ok(())
}

/// T3 drive encoder: drives = gain * inputs.
///
/// Кодирует входной сигнал в драйв T3 осциллятора.
#[pyfunction]
#[pyo3(name = "t3_drive_encoder")]
pub fn py_t3_drive_encoder(
    drives: Bound<'_, PyArray1<f32>>,
    inputs: Bound<'_, PyArray1<f32>>,
    gain: f32,
) -> PyResult<()> {
    let mut drives = drives.readwrite();
    let inputs = inputs.readonly();
    io::t3_drive_encoder(drives.as_slice_mut()?, inputs.as_slice()?, gain);
    Ok(())
}

/// T3 complex decoder: outputs = W·[A*cos(θ); A*sin(θ)] + bias.
///
/// Декодирует комплексное состояние осциллятора (амплитуда+фаза) в выход.
#[pyfunction]
#[pyo3(name = "t3_complex_decoder")]
pub fn py_t3_complex_decoder(
    outputs: Bound<'_, PyArray1<f32>>,
    amplitudes: Bound<'_, PyArray1<f32>>,
    cos_buf: Bound<'_, PyArray1<f32>>,
    weights: Bound<'_, PyArray1<f32>>,
    biases: Bound<'_, PyArray1<f32>>,
    n_out: usize,
) -> PyResult<()> {
    let mut outputs = outputs.readwrite();
    let amplitudes = amplitudes.readonly();
    let cos_buf = cos_buf.readonly();
    let weights = weights.readonly();
    let biases = biases.readonly();
    io::t3_complex_decoder(
        outputs.as_slice_mut()?, amplitudes.as_slice()?, cos_buf.as_slice()?,
        weights.as_slice()?, biases.as_slice()?, n_out,
    );
    Ok(())
}

/// Echo state tap: читает линию задержки из буфера осциллятора.
#[pyfunction]
#[pyo3(name = "echo_tap")]
pub fn py_echo_tap(
    output: Bound<'_, PyArray1<f32>>,
    buffer: Bound<'_, PyArray1<f32>>,
    n_osc: usize,
    delay_len: usize,
    delay_idx: usize,
) -> PyResult<()> {
    let mut output = output.readwrite();
    let buffer = buffer.readonly();
    io::echo_tap(output.as_slice_mut()?, buffer.as_slice()?, n_osc, delay_len, delay_idx);
    Ok(())
}

/// Хеббианское обновление: w += lr·cos(Δθ) - decay·w·dt.
///
/// Классическое правило Хебба: нейроны, возбуждающиеся вместе,
/// укрепляют свои связи.
#[pyfunction]
#[pyo3(name = "hebbian_update")]
pub fn py_hebbian_update(
    weights: Bound<'_, PyArray1<f32>>,
    cos_diff: Bound<'_, PyArray1<f32>>,
    lr: f32,
    decay: f32,
    dt: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let cos_diff = cos_diff.readonly();
    io::hebbian_update(weights.as_slice_mut()?, cos_diff.as_slice()?, lr, decay, dt);
    Ok(())
}

/// Стохастический энкодер: спайки при превышении порога + шум.
#[pyfunction]
#[pyo3(name = "stochastic_encoder")]
pub fn py_stochastic_encoder(
    target_I: Bound<'_, PyArray1<f32>>,
    inputs: Bound<'_, PyArray1<f32>>,
    noise: Bound<'_, PyArray1<f32>>,
    threshold: f32,
    spike_val: f32,
) -> PyResult<()> {
    io::stochastic_encoder(
        target_I.readwrite().as_slice_mut()?,
        inputs.readonly().as_slice()?,
        noise.readonly().as_slice()?,
        threshold, spike_val,
    );
    Ok(())
}

/// Таламический гейт: пропускает source ток через мембранный потенциал gate.
///
/// Имитирует таламо-кортикальные ворота.
#[pyfunction]
#[pyo3(name = "thalamic_gate")]
pub fn py_thalamic_gate(
    target_I: Bound<'_, PyArray1<f32>>,
    source_v: Bound<'_, PyArray1<f32>>,
    gate_v: Bound<'_, PyArray1<f32>>,
    threshold: f32,
    gain: f32,
) -> PyResult<()> {
    io::thalamic_gate(
        target_I.readwrite().as_slice_mut()?,
        source_v.readonly().as_slice()?,
        gate_v.readonly().as_slice()?,
        threshold, gain,
    );
    Ok(())
}

/// T3 inverse Kuramoto: вычисляет драйв, обратный к динамике Курамото.
///
/// Используется для управления осцилляторами в обратную сторону.
#[pyfunction]
#[pyo3(name = "t3_inverse_kuramoto")]
pub fn py_t3_inverse_kuramoto(
    drive_out: Bound<'_, PyArray1<f32>>,
    r: Bound<'_, PyArray1<f32>>,
    r_prev: Bound<'_, PyArray1<f32>>,
    alpha: f32,
    gain: f32,
    eps: f32,
    decay: f32,
) -> PyResult<()> {
    io::t3_inverse_kuramoto(
        drive_out.readwrite().as_slice_mut()?,
        r.readonly().as_slice()?,
        r_prev.readonly().as_slice()?,
        alpha, gain, eps, decay,
    );
    Ok(())
}

/// T3 PST features: sin/cos разности фаз между двумя популяциями.
#[pyfunction]
#[pyo3(name = "t3_pst_features")]
pub fn py_t3_pst_features(
    out_sin: Bound<'_, PyArray1<f32>>,
    out_cos: Bound<'_, PyArray1<f32>>,
    ph1: Bound<'_, PyArray1<f32>>,
    ph2: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    io::t3_pst_features(
        out_sin.readwrite().as_slice_mut()?,
        out_cos.readwrite().as_slice_mut()?,
        ph1.readonly().as_slice()?,
        ph2.readonly().as_slice()?,
    );
    Ok(())
}

/// T3 TACT генератор: генерирует импульсы из 3 фазовых популяций.
#[pyfunction]
#[pyo3(name = "t3_tact_generator")]
pub fn py_t3_tact_generator(
    pulses: Bound<'_, PyArray1<f32>>,
    ph1: Bound<'_, PyArray1<f32>>,
    ph2: Bound<'_, PyArray1<f32>>,
    ph3: Bound<'_, PyArray1<f32>>,
    threshold: f32,
) -> PyResult<()> {
    io::t3_tact_generator(
        pulses.readwrite().as_slice_mut()?,
        ph1.readonly().as_slice()?,
        ph2.readonly().as_slice()?,
        ph3.readonly().as_slice()?,
        threshold,
    );
    Ok(())
}
