//! PyO3 bindings for SNN classes — LIF-AT, SORN, TTFS, plasticity, encoding.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::models::neuron;

// ═══════════════════════════════════════════════════════════
//  PyLifNeuron — LIF-AT with adaptive threshold
// ═══════════════════════════════════════════════════════════

/// LIF-AT neuron — LIF с адаптивным порогом (stateful).
///
/// Использование:
///     n = LifNeuron(1000, dt=0.5)
///     I = np.random.randn(1000).astype(np.float32)
///     spikes = np.zeros(1000, dtype=np.int32)
///     n.step(I, spikes)
///     v = n.v        # свойство — мембранный потенциал
///     n.v = arr      # свойство — установить
#[pyclass(name = "LifNeuron")]
pub struct PyLifNeuron {
    v: Vec<f32>,
    theta: Vec<f32>,
    n: usize,
    tau_m: Vec<f32>,
    v_rest: Vec<f32>,
    v_thresh: Vec<f32>,
    v_reset: Vec<f32>,
    tau_theta: Vec<f32>,
    beta: Vec<f32>,
    dt: f32,
}

#[pymethods]
impl PyLifNeuron {
    #[new]
    #[pyo3(signature = (n, dt=0.5, tau_m=20.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0, tau_theta=10.0, beta=0.1))]
    fn new(n: usize, dt: f32, tau_m: f32, v_rest: f32, v_thresh: f32, v_reset: f32, tau_theta: f32, beta: f32) -> Self {
        let n_padded = n.div_ceil(8) * 8;
        Self {
            v: vec![v_rest; n_padded], theta: vec![0.0; n_padded], n,
            tau_m: vec![tau_m; n_padded], v_rest: vec![v_rest; n_padded],
            v_thresh: vec![v_thresh; n_padded], v_reset: vec![v_reset; n_padded],
            tau_theta: vec![tau_theta; n_padded], beta: vec![beta; n_padded], dt,
        }
    }

    fn step(&mut self, I: Bound<'_, PyArray1<f32>>, spikes: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let mut I = I.readwrite();
        let mut spikes = spikes.readwrite();
        let n = self.n.div_ceil(8) * 8;
        let mut refractory = vec![0.0f32; n];  // class-owned, will be field later
        neuron::step_lif_at(
            &mut self.v, &mut self.theta,
            I.as_slice_mut()?, spikes.as_slice_mut()?,
            self.dt, &self.tau_m, &self.v_rest, &self.v_thresh,
            &self.v_reset, &self.tau_theta, &self.beta,
            &mut refractory, neuron::DEFAULT_TAU_REF,
        );
        Ok(())
    }

    /// Мембранный потенциал как np.ndarray.
    #[getter]
    fn v(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.v[..self.n].to_vec()).into()
    }

    #[setter]
    fn set_v(&mut self, src: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let src = src.readonly();
        let n = self.n.min(src.len()?);
        self.v[..n].copy_from_slice(&src.as_slice()?[..n]);
        Ok(())
    }

    /// Адаптивный порог как np.ndarray.
    #[getter]
    fn theta(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.theta[..self.n].to_vec()).into()
    }

    #[setter]
    fn set_theta(&mut self, src: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let src = src.readonly();
        let n = self.n.min(src.len()?);
        self.theta[..n].copy_from_slice(&src.as_slice()?[..n]);
        Ok(())
    }

    fn n_neurons(&self) -> usize { self.n }

    fn __repr__(&self) -> String {
        format!("LifNeuron(n={}, dt={})", self.n, self.dt)
    }
}

// ═══════════════════════════════════════════════════════════
//  Plasticity functions
// ═══════════════════════════════════════════════════════════

/// Intrinsic Plasticity: адаптирует порог под целевой firing rate.
///
/// θ += lr · (spike - target_rate) - decay · θ
#[pyfunction]
#[pyo3(name = "intrinsic_plasticity")]
pub fn py_intrinsic_plasticity(
    theta: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    lr_ip: f32,
    target_rate: f32,
    decay: f32,
) -> PyResult<()> {
    let mut theta = theta.readwrite();
    let spikes = spikes.readonly();
    neuron::intrinsic_plasticity(theta.as_slice_mut()?, spikes.as_slice()?, lr_ip, target_rate, decay);
    Ok(())
}

/// Синаптическая нормализация: нормирует входящие веса к target_sum.
///
/// w_i = w_i · target_sum / (Σ w_i + eps)
#[pyfunction]
#[pyo3(name = "synaptic_normalization")]
pub fn py_synaptic_normalization(
    weights: Bound<'_, PyArray1<f32>>,
    n_pre: usize,
    n_post: usize,
    target_sum: f32,
    eps: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    neuron::synaptic_normalization(weights.as_slice_mut()?, n_pre, n_post, target_sum, eps);
    Ok(())
}

// ═══════════════════════════════════════════════════════════
//  SORN network
// ═══════════════════════════════════════════════════════════

#[pyclass(name = "SornNetwork")]
pub struct PySornNetwork {
    pub inner: neuron::SornNetwork,
}

#[pymethods]
impl PySornNetwork {
    #[new]
    fn new(n: usize, w_init: f32, w_target_sum: f32) -> Self {
        Self { inner: neuron::SornNetwork::new(n, w_init, w_target_sum) }
    }

    fn step(&mut self, spikes: Bound<'_, PyArray1<i32>>, traces_pre: Bound<'_, PyArray1<f32>>, traces_post: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let spikes = spikes.readonly();
        let traces_pre = traces_pre.readonly();
        let traces_post = traces_post.readonly();
        self.inner.step(spikes.as_slice()?, traces_pre.as_slice()?, traces_post.as_slice()?);
        Ok(())
    }

    fn reset(&mut self) { self.inner.reset(); }

    /// Веса синапсов как np.ndarray.
    #[getter]
    fn weights(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.weights.clone()).into()
    }

    /// Адаптивные пороги как np.ndarray.
    #[getter]
    fn theta(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.theta.clone()).into()
    }

    fn __repr__(&self) -> String {
        format!("SornNetwork(n={}, target_sum={})", self.inner.n, self.inner.w_target_sum)
    }

    fn n_neurons(&self) -> usize { self.inner.n }
}

// ═══════════════════════════════════════════════════════════
//  Spike Encoding — BSA, Phase, TTFS
// ═══════════════════════════════════════════════════════════

/// BSA (Bens Spike Algorithm) кодирование сигнала в спайки.
///
/// Пороговое кодирование с ядром. Возвращает количество спайков.
#[pyfunction]
#[pyo3(name = "bsa_encode")]
pub fn py_bsa_encode(
    spikes_out: Bound<'_, PyArray1<i32>>,
    signal: Bound<'_, PyArray1<f32>>,
    kernel: Bound<'_, PyArray1<f32>>,
    reconstructed: Bound<'_, PyArray1<f32>>,
    threshold: f32,
) -> PyResult<usize> {
    let mut spikes_out = spikes_out.readwrite();
    let signal = signal.readonly();
    let kernel = kernel.readonly();
    let mut reconstructed = reconstructed.readwrite();
    Ok(neuron::bsa_encode(spikes_out.as_slice_mut()?, signal.as_slice()?, kernel.as_slice()?, reconstructed.as_slice_mut()?, threshold))
}

/// Фазовое кодирование: сигнал → спайки через фазы осцилляторов.
///
/// Возвращает количество спайков.
#[pyfunction]
#[pyo3(name = "phase_encode")]
pub fn py_phase_encode(
    spikes_out: Bound<'_, PyArray1<i32>>,
    phases: Bound<'_, PyArray1<f32>>,
    cos_phases: Bound<'_, PyArray1<f32>>,
    signal: Bound<'_, PyArray1<f32>>,
    threshold: f32,
    gain: f32,
) -> PyResult<usize> {
    let mut spikes_out = spikes_out.readwrite();
    let mut phases = phases.readwrite();
    let cos_phases = cos_phases.readonly();
    let signal = signal.readonly();
    Ok(neuron::phase_encode(spikes_out.as_slice_mut()?, phases.as_slice_mut()?, cos_phases.as_slice()?, signal.as_slice()?, threshold, gain))
}

/// TTFS (Time-To-First-Spike) энкодер.
///
/// Кодирует значение временем до первого спайка.
#[pyclass]
pub struct PyTtfsEncoder {
    inner: neuron::TtfsEncoder,
}

#[pymethods]
impl PyTtfsEncoder {
    #[new]
    fn new(n: usize, max_delay: usize) -> Self { Self { inner: neuron::TtfsEncoder::new(n, max_delay) } }

    fn encode_step(&mut self, values: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<bool> {
        let values = values.readonly();
        let mut spikes_out = spikes_out.readwrite();
        Ok(self.inner.encode_step(values.as_slice()?, spikes_out.as_slice_mut()?))
    }

    fn reset(&mut self) { self.inner.reset(); }

    fn __repr__(&self) -> String { format!("TtfsEncoder(max_delay={}, n_neurons={})", self.inner.max_delay, self.inner.fire_times.len()) }

    fn n_neurons(&self) -> usize { self.inner.fire_times.len() }
}

// ═══════════════════════════════════════════════════════════
//  SpikeRaster → CSV export
// ═══════════════════════════════════════════════════════════

/// Экспорт spike raster из dense буфера в CSV строку.
///
/// Принимает dense буфер `[n_steps × n_neurons]` и dt (ms).
/// Возвращает CSV с колонками `time_ms,neuron_id`.
///
/// Args:
///     spike_buffer: np.ndarray[int32, shape=(n_steps * n_neurons,)]
///                   Spike events: 0/1, layout [time × neuron]
///     n_neurons: int — количество нейронов
///     dt: float — временной шаг в ms
///
/// Returns:
///     str — CSV строка с заголовком "time_ms,neuron_id"
#[pyfunction]
#[pyo3(name = "spike_raster_to_csv")]
pub fn py_spike_raster_to_csv(
    spike_buffer: Bound<'_, PyArray1<i32>>,
    n_neurons: usize,
    dt: f32,
) -> PyResult<String> {
    let spike_buffer = spike_buffer.readonly();
    let buf = spike_buffer.as_slice()?;
    let n_steps = if n_neurons > 0 { buf.len() / n_neurons } else { 0 };

    // Build raster from dense buffer, then export
    let mut raster = neuron::SpikeRaster::new();
    let mut step_buf = vec![0i32; n_neurons];
    for t in 0..n_steps {
        step_buf.fill(0);
        let offset = t * n_neurons;
        for i in 0..n_neurons {
            if buf[offset + i] != 0 {
                step_buf[i] = 1;
            }
        }
        raster.record(t, &step_buf, n_neurons);
    }

    Ok(raster.to_csv(dt))
}

/// Hamming окно для BSA энкодера.
///
/// Возвращает np.ndarray длины len.
#[pyfunction]
#[pyo3(name = "bsa_kernel_hamming")]
pub fn py_bsa_kernel_hamming(len: usize) -> Vec<f32> { neuron::bsa_kernel_hamming(len) }

/// Декодирование population rate из буфера спайков.
///
/// Вычисляет скользящее среднее частоты спайков по окну.
#[pyfunction]
#[pyo3(name = "population_rate_decode")]
pub fn py_population_rate_decode(
    rates_out: Bound<'_, PyArray1<f32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
    n_neurons: usize,
    window: usize,
) -> PyResult<()> {
    let mut rates_out = rates_out.readwrite();
    let spike_buffer = spike_buffer.readonly();
    neuron::population_rate_decode(rates_out.as_slice_mut()?, spike_buffer.as_slice()?, n_neurons, window);
    Ok(())
}
