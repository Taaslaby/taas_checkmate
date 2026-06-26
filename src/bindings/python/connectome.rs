//! PyO3 bindings for connectome benchmark — CPU (SIMD+rayon) and GPU (CUDA).
//!
//! CPU version: init I = external_I once, then per step:
//!   Izhikevich (on I with propagated current) → reset I = external_I → CSR propagate
//! Features: SIMD f32x8, rayon parallel, refractory period.
//!
//! GPU version: persistent GpuContext, no per-step H2D/D2H transfers:
//!   Izhikevich (CUDA) → count spikes (device-side) → CSR propagate (CUDA reset_I + csr_propagate)
//! Features: on-device CSR propagation, no refractory period

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::models::neuron;

// ─── CPU: full connectome simulation in tight Rust loop ──

/// Бенчмарк коннектома на CPU (SIMD + rayon).
///
/// Args: CSR матрица, N шагов, dt, массивы состояния нейронов.
/// Returns: (wall_time_sec, total_spikes).
#[pyfunction]
#[pyo3(name = "bench_connectome_cpu")]
#[allow(clippy::too_many_arguments)]
pub fn py_bench_connectome_cpu(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    I_buf: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let a = a.readonly();
    let b = b.readonly();
    let c = c.readonly();
    let d = d.readonly();
    let nt_types = nt_types.readonly();
    let external_I = external_I.readonly();
    let mut I_buf = I_buf.readwrite();
    let mut spikes = spikes.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();

    let v = v.as_slice_mut()?;
    let u = u.as_slice_mut()?;
    let trace = trace.as_slice_mut()?;
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    let c = c.as_slice()?;
    let d = d.as_slice()?;
    let nt_types = nt_types.as_slice()?;
    let external_I = external_I.as_slice()?;
    let I_buf = I_buf.as_slice_mut()?;
    let spikes = spikes.as_slice_mut()?;
    let row_ptr = row_ptr.as_slice()?;
    let col_indices = col_indices.as_slice()?;
    let values = values.as_slice()?;

    let n = v.len();
    let n_neurons = row_ptr.len() - 1;
    let mut refractory = vec![0.0f32; n];
    let mut total_spikes: usize = 0;

    // Initialize I = external_I (same as GPU: upload_I before loop)
    for i in 0..n.min(external_I.len()) {
        I_buf[i] = external_I[i];
    }
    for i in n..I_buf.len() {
        I_buf[i] = 0.0;
    }

    let t0 = std::time::Instant::now();

    for _step in 0..n_steps {
        // 1. Izhikevich step (SIMD + rayon) on current I
        //    I already has external + propagated from previous step
        neuron::step_izhikevich(
            v, u, trace, a, b, I_buf, dt, 0.99,
            spikes, c, d, &mut refractory, neuron::DEFAULT_TAU_REF,
        );

        // 2. Count spikes + prepare I for next step
        //    Reset I = external_I, then CSR propagate onto it
        for i in 0..n.min(external_I.len()) {
            I_buf[i] = external_I[i];
        }
        for i in n..I_buf.len() {
            I_buf[i] = 0.0;
        }

        // 3. CSR propagate spikes → I_buf (adds to reset I for next step)
        // NOTE: use direct row_ptr indexing, NOT sparse::bound_row (which clamps to n_neurons).
        // row_ptr values are synapse indices (up to 5.3M), not neuron indices!
        let max_syn = col_indices.len().min(values.len());
        for i in 0..n_neurons.min(spikes.len()) {
            if spikes[i] != 0 {
                total_spikes += 1;
                let start = row_ptr[i].max(0) as usize;
                let end = (row_ptr[i + 1] as usize).min(max_syn);
                let mult = if i < nt_types.len() { nt_types[i] } else { 1.0 };
                let mut j = start;
                while j + 4 <= end {
                    let t0 = col_indices[j] as usize;
                    let t1 = col_indices[j + 1] as usize;
                    let t2 = col_indices[j + 2] as usize;
                    let t3 = col_indices[j + 3] as usize;
                    if t0 < n { I_buf[t0] += values[j] * mult; }
                    if t1 < n { I_buf[t1] += values[j + 1] * mult; }
                    if t2 < n { I_buf[t2] += values[j + 2] * mult; }
                    if t3 < n { I_buf[t3] += values[j + 3] * mult; }
                    j += 4;
                }
                for j in j..end {
                    let col = col_indices[j] as usize;
                    if col < n { I_buf[col] += values[j] * mult; }
                }
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, total_spikes))
}

// ─── Connectome with per-step spike recording ───────────

/// Как bench_connectome_cpu, но записывает per-step спайки в spike_buffer.
///
/// spike_buffer [n_steps × n] — заполняется 0/1 для raster plot.
/// Возвращает (wall_time_sec, total_spikes).
#[pyfunction]
#[pyo3(name = "bench_connectome_cpu_recording")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn py_bench_connectome_cpu_recording(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    I_buf: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    let mut v = v.readwrite();
    let mut u = u.readwrite();
    let mut trace = trace.readwrite();
    let a = a.readonly();
    let b = b.readonly();
    let c = c.readonly();
    let d = d.readonly();
    let nt_types = nt_types.readonly();
    let external_I = external_I.readonly();
    let mut I_buf = I_buf.readwrite();
    let mut spikes = spikes.readwrite();
    let mut spike_buffer = spike_buffer.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();

    let v = v.as_slice_mut()?;
    let u = u.as_slice_mut()?;
    let trace = trace.as_slice_mut()?;
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    let c = c.as_slice()?;
    let d = d.as_slice()?;
    let nt_types = nt_types.as_slice()?;
    let external_I = external_I.as_slice()?;
    let I_buf = I_buf.as_slice_mut()?;
    let spikes = spikes.as_slice_mut()?;
    let spike_buffer = spike_buffer.as_slice_mut()?;
    let row_ptr = row_ptr.as_slice()?;
    let col_indices = col_indices.as_slice()?;
    let values = values.as_slice()?;

    let n = v.len();
    let n_neurons = row_ptr.len() - 1;
    let mut refractory = vec![0.0f32; n];
    let mut total_spikes: usize = 0;

    // Initialize I = external_I
    for i in 0..n.min(external_I.len()) {
        I_buf[i] = external_I[i];
    }
    for i in n..I_buf.len() {
        I_buf[i] = 0.0;
    }

    let t0 = std::time::Instant::now();

    for step in 0..n_steps {
        // 1. Izhikevich step
        neuron::step_izhikevich(
            v, u, trace, a, b, I_buf, dt, 0.99,
            spikes, c, d, &mut refractory, neuron::DEFAULT_TAU_REF,
        );

        // 2. Record to spike_buffer before reset
        let buf_offset = step * n_neurons;
        for i in 0..n_neurons.min(spikes.len()) {
            if spikes[i] != 0 {
                let idx = buf_offset + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
                total_spikes += 1;
            }
        }

        // 3. Reset I = external_I
        for i in 0..n.min(external_I.len()) {
            I_buf[i] = external_I[i];
        }
        for i in n..I_buf.len() {
            I_buf[i] = 0.0;
        }

        // 4. CSR propagate
        let max_syn = col_indices.len().min(values.len());
        for i in 0..n_neurons.min(spikes.len()) {
            if spikes[i] != 0 {
                let start = row_ptr[i].max(0) as usize;
                let end = (row_ptr[i + 1] as usize).min(max_syn);
                let mult = if i < nt_types.len() { nt_types[i] } else { 1.0 };
                let mut j = start;
                while j + 4 <= end {
                    let t0 = col_indices[j] as usize;
                    let t1 = col_indices[j + 1] as usize;
                    let t2 = col_indices[j + 2] as usize;
                    let t3 = col_indices[j + 3] as usize;
                    if t0 < n { I_buf[t0] += values[j] * mult; }
                    if t1 < n { I_buf[t1] += values[j + 1] * mult; }
                    if t2 < n { I_buf[t2] += values[j + 2] * mult; }
                    if t3 < n { I_buf[t3] += values[j + 3] * mult; }
                    j += 4;
                }
                for j in j..end {
                    let col = col_indices[j] as usize;
                    if col < n { I_buf[col] += values[j] * mult; }
                }
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, total_spikes))
}

// ─── GPU: connectome simulation using GpuContext ────────

/// Бенчмарк коннектома на GPU (CUDA).
///
/// Требует сборки с --features cuda.
/// Returns: (wall_time_sec, total_spikes).
#[pyfunction]
#[pyo3(name = "bench_connectome_gpu")]
#[allow(clippy::too_many_arguments)]
pub fn py_bench_connectome_gpu(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (row_ptr, col_indices, values, n_steps, dt, v, u, a, b, c, d, nt_types, external_I, spikes);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "CUDA not available — rebuild with --features cuda".to_string()
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use cuda_kernels::GpuContext;

        let row_ptr_data = row_ptr.readonly();
        let col_indices_data = col_indices.readonly();
        let values_data = values.readonly();
        let v_data = v.readwrite();
        let u_data = u.readwrite();
        let a_data = a.readonly();
        let b_data = b.readonly();
        let c_data = c.readonly();
        let d_data = d.readonly();
        let nt_types_data = nt_types.readonly();
        let external_I_data = external_I.readonly();
        let mut spikes_data = spikes.readwrite();

        let row_ptr_sl = row_ptr_data.as_slice()?;
        let col_indices_sl = col_indices_data.as_slice()?;
        let values_sl = values_data.as_slice()?;
        let a_sl = a_data.as_slice()?;
        let b_sl = b_data.as_slice()?;
        let c_sl = c_data.as_slice()?;
        let d_sl = d_data.as_slice()?;
        let nt_types_sl = nt_types_data.as_slice()?;
        let external_I_sl = external_I_data.as_slice()?;
        let spikes_sl = spikes_data.as_slice_mut()?;
        let _n = v_data.len()?;

        let n_neurons = row_ptr_sl.len() - 1;
        let n_synapses = values_sl.len();

        // Copy initial state to host buffers
        let v_host = v_data.as_slice()?.to_vec();
        let u_host = u_data.as_slice()?.to_vec();

        // Create GPU context with enough synapses
        let gpu_ctx = match GpuContext::new(n_neurons, n_synapses) {
            Ok(ctx) => ctx,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("GpuContext::new failed: {e}")
            )),
        };

        // Upload neuron params to GPU
        if let Err(e) = gpu_ctx.upload_neurons(&v_host, &u_host, a_sl, b_sl, c_sl, d_sl) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_neurons: {e}")
            ));
        }

        // Upload CSR connectome arrays to GPU (one-time)
        if let Err(e) = gpu_ctx.upload_connectome(row_ptr_sl, col_indices_sl, values_sl,
            nt_types_sl, external_I_sl)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_connectome: {e}")
            ));
        }

        // Allocate device-side spike counter
        let d_spike_counter = match gpu_ctx.alloc_spike_counter() {
            Ok(ptr) => ptr,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("alloc_spike_counter: {e}")
            )),
        };

        // Initialize I = external_I (needed for first step's izhikevich)
        if let Err(e) = gpu_ctx.upload_I(external_I_sl) {
            gpu_ctx.free_spike_counter(d_spike_counter);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_I initial: {e}")
            ));
        }

        let t0 = std::time::Instant::now();

        // All GPU: no per-step H2D/D2H transfers!
        for _step in 0..n_steps {
            // CSR propagate + Izhikevich — all on GPU, counts spikes device-side
            if let Err(e) = gpu_ctx.connectome_step(dt, 0.99, Some(d_spike_counter)) {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("connectome_step: {e}")
                ));
            }
        }

        // Sync, read spike counter, cleanup
        let _ = gpu_ctx.sync();
        let total_spikes_i32 = gpu_ctx.read_spike_counter(d_spike_counter)
            .unwrap_or(-1);
        gpu_ctx.free_spike_counter(d_spike_counter);
        let total_spikes = total_spikes_i32.max(0) as usize;

        // Download final spikes for caller reference
        let _ = gpu_ctx.download_spikes(spikes_sl);

        let elapsed = t0.elapsed().as_secs_f64();
        Ok((elapsed, total_spikes))
    }
}

// ─── GPU: connectome STDP initialization ─────────────

/// Запускает GPU симуляцию коннектома со STDP на пластичных весах.
///
/// Порядок на каждом шаге:
///   1. Izhikevich — динамика нейронов
///   2. Count spikes
///   3. CSR STDP — обновление d_weights из pre/post traces
///   4. CSR propagate — использует d_weights (пластичные)
///
/// Инициализирует пластичные веса из статического коннектома.
/// После STDP можно скачать финальные веса и сравнить с исходными.
///
/// spike_buffer [n_steps × n_neurons] — 0/1 для raster plot.
/// weights_out [n_synapses] — финальные веса после STDP (опционально, None = не скачивать).
/// Returns: (wall_time_sec, total_spikes, mean_weight_change).
#[pyfunction]
#[pyo3(name = "bench_connectome_gpu_stdp", signature = (
    row_ptr, col_indices, values, n_steps, dt,
    v, u, _trace, a, b, c, d, nt_types, external_I,
    _I_buf, _spikes, spike_buffer,
    lr_ltp, lr_ltd, w_min, w_max,
    weights_out=None))]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn py_bench_connectome_gpu_stdp(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    _trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    _I_buf: Bound<'_, PyArray1<f32>>,
    _spikes: Bound<'_, PyArray1<i32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
    lr_ltp: f32,
    lr_ltd: f32,
    w_min: f32,
    w_max: f32,
    weights_out: Option<Bound<'_, PyArray1<f32>>>,
) -> PyResult<(f64, usize, f32)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (row_ptr, col_indices, values, n_steps, dt,
                 v, u, _trace, a, b, c, d, nt_types, external_I,
                 _I_buf, _spikes, spike_buffer, weights_out,
                 lr_ltp, lr_ltd, w_min, w_max);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "CUDA not available — rebuild with --features cuda".to_string()
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use cuda_kernels::GpuContext;

        let row_ptr_data = row_ptr.readonly();
        let col_indices_data = col_indices.readonly();
        let values_data = values.readonly();
        let mut v_data = v.readwrite();
        let mut u_data = u.readwrite();
        let a_data = a.readonly();
        let b_data = b.readonly();
        let c_data = c.readonly();
        let d_data = d.readonly();
        let nt_types_data = nt_types.readonly();
        let external_I_data = external_I.readonly();
        let mut spike_buffer_data = spike_buffer.readwrite();

        let row_ptr_sl = row_ptr_data.as_slice()?;
        let col_indices_sl = col_indices_data.as_slice()?;
        let values_sl = values_data.as_slice()?;
        let a_sl = a_data.as_slice()?;
        let b_sl = b_data.as_slice()?;
        let c_sl = c_data.as_slice()?;
        let d_sl = d_data.as_slice()?;
        let nt_types_sl = nt_types_data.as_slice()?;
        let external_I_sl = external_I_data.as_slice()?;
        let spike_buffer_sl = spike_buffer_data.as_slice_mut()?;
        let _n = v_data.len()?;

        let n_neurons = row_ptr_sl.len() - 1;
        let n_synapses = values_sl.len();

        // Copy initial state
        let v_host = v_data.as_slice()?.to_vec();
        let u_host = u_data.as_slice()?.to_vec();

        // Create GPU context
        let gpu_ctx = match GpuContext::new(n_neurons, n_synapses) {
            Ok(ctx) => ctx,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("GpuContext::new failed: {e}")
            )),
        };

        // Upload neuron params
        if let Err(e) = gpu_ctx.upload_neurons(&v_host, &u_host, a_sl, b_sl, c_sl, d_sl) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_neurons: {e}")
            ));
        }

        // Upload CSR connectome + external_I
        if let Err(e) = gpu_ctx.upload_connectome(row_ptr_sl, col_indices_sl, values_sl,
            nt_types_sl, external_I_sl)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_connectome: {e}")
            ));
        }

        // Initialize plastic weights from static connectome
        if let Err(e) = gpu_ctx.init_plastic_weights() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("init_plastic_weights: {e}")
            ));
        }

        // Upload initial I = external_I
        if let Err(e) = gpu_ctx.upload_I(external_I_sl) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_I: {e}")
            ));
        }

        // Allocate device-side spike counter
        let d_spike_counter = match gpu_ctx.alloc_spike_counter() {
            Ok(ptr) => ptr,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("alloc_spike_counter: {e}")
            )),
        };

        let mut total_spikes: usize = 0;
        let mut step_spikes = vec![0i32; n_neurons];
        let t0 = std::time::Instant::now();

        for step in 0..n_steps {
            // One STDP step: Izhikevich → count → CSR STDP → CSR propagate (plastic)
            // Uses connectome_stdp_step from GpuContext
            if let Err(e) = gpu_ctx.connectome_stdp_step(dt, 0.99,
                lr_ltp, lr_ltd, w_min, w_max, Some(d_spike_counter))
            {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("connectome_stdp_step: {e}")
                ));
            }

            // Download spikes to host (per-step D2H for recording)
            if let Err(e) = gpu_ctx.download_spikes(&mut step_spikes) {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("download_spikes: {e}")
                ));
            }

            // Record to spike_buffer
            let buf_offset = step * n_neurons;
            let mut step_count = 0usize;
            for i in 0..n_neurons {
                if step_spikes[i] != 0 {
                    let idx = buf_offset + i;
                    if idx < spike_buffer_sl.len() {
                        spike_buffer_sl[idx] = 1;
                    }
                    step_count += 1;
                }
            }
            total_spikes += step_count;
        }

        let elapsed = t0.elapsed().as_secs_f64();

        // Download final spike counter
        let total_spikes_i32 = gpu_ctx.read_spike_counter(d_spike_counter).unwrap_or(-1);
        gpu_ctx.free_spike_counter(d_spike_counter);
        let total_spikes = total_spikes_i32.max(0) as usize;

        // Download final weights and compute mean change
        let mut weights_host = vec![0.0f32; n_synapses];
        let mean_weight_change = if let Ok(()) = gpu_ctx.download_weights(&mut weights_host) {
            let mut sum_diff = 0.0f64;
            let mut count = 0usize;
            for i in 0..n_synapses.min(values_sl.len()) {
                let diff = (weights_host[i] - values_sl[i]).abs() as f64;
                sum_diff += diff;
                count += 1;
            }
            if count > 0 { (sum_diff / count as f64) as f32 } else { 0.0 }
        } else {
            0.0
        };

        // Write weights back to Python array if requested
        if let Some(wo) = weights_out {
            let mut wo_data = wo.readwrite();
            let wo_sl = wo_data.as_slice_mut()?;
            let write_n = wo_sl.len().min(weights_host.len());
            for i in 0..write_n {
                wo_sl[i] = weights_host[i];
            }
        }

        // State persistence: download v,u back to host
        let mut v_back = vec![0.0f32; n_neurons];
        let mut u_back = vec![0.0f32; n_neurons];
        let _ = gpu_ctx.download_v(&mut v_back);
        let _ = gpu_ctx.download_u(&mut u_back);

        let v_slice = v_data.as_slice_mut()?;
        let u_slice = u_data.as_slice_mut()?;
        let write_n = n_neurons.min(v_slice.len()).min(v_back.len())
            .min(u_slice.len()).min(u_back.len());
        for i in 0..write_n {
            v_slice[i] = v_back[i];
            u_slice[i] = u_back[i];
        }

        Ok((elapsed, total_spikes, mean_weight_change))
    }
}

// ─── GPU: connectome with per-step spike recording ────

/// Как bench_connectome_gpu, но записывает per-step спайки в spike_buffer.
///
/// После каждого GPU шага скачивает spikes на host (D2H, ~552KB для 138K).
/// Скачивает v, u обратно на host после симуляции (state persistence).
///
/// spike_buffer [n_steps × n_neurons] — 0/1 для raster plot.
/// Returns: (wall_time_sec, total_spikes).
#[pyfunction]
#[pyo3(name = "bench_connectome_gpu_recording")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn py_bench_connectome_gpu_recording(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    _trace: Bound<'_, PyArray1<f32>>,
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
    c: Bound<'_, PyArray1<f32>>,
    d: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    _I_buf: Bound<'_, PyArray1<f32>>,
    _spikes: Bound<'_, PyArray1<i32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (row_ptr, col_indices, values, n_steps, dt,
                 v, u, _trace, a, b, c, d, nt_types, external_I,
                 _I_buf, _spikes, spike_buffer);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "CUDA not available — rebuild with --features cuda".to_string()
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use cuda_kernels::GpuContext;

        let row_ptr_data = row_ptr.readonly();
        let col_indices_data = col_indices.readonly();
        let values_data = values.readonly();
        let mut v_data = v.readwrite();
        let mut u_data = u.readwrite();
        let a_data = a.readonly();
        let b_data = b.readonly();
        let c_data = c.readonly();
        let d_data = d.readonly();
        let nt_types_data = nt_types.readonly();
        let external_I_data = external_I.readonly();
        let mut spike_buffer_data = spike_buffer.readwrite();

        let row_ptr_sl = row_ptr_data.as_slice()?;
        let col_indices_sl = col_indices_data.as_slice()?;
        let values_sl = values_data.as_slice()?;
        let a_sl = a_data.as_slice()?;
        let b_sl = b_data.as_slice()?;
        let c_sl = c_data.as_slice()?;
        let d_sl = d_data.as_slice()?;
        let nt_types_sl = nt_types_data.as_slice()?;
        let external_I_sl = external_I_data.as_slice()?;
        let spike_buffer_sl = spike_buffer_data.as_slice_mut()?;
        let _n = v_data.len()?;

        let n_neurons = row_ptr_sl.len() - 1;
        let n_synapses = values_sl.len();

        // Copy initial state from host (v, u are persistent across batch calls)
        let v_host = v_data.as_slice()?.to_vec();
        let u_host = u_data.as_slice()?.to_vec();

        // Create GPU context
        let gpu_ctx = match GpuContext::new(n_neurons, n_synapses) {
            Ok(ctx) => ctx,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("GpuContext::new failed: {e}")
            )),
        };

        // Upload neuron params (v, u from previous batch's end state)
        if let Err(e) = gpu_ctx.upload_neurons(&v_host, &u_host, a_sl, b_sl, c_sl, d_sl) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_neurons: {e}")
            ));
        }

        // Upload CSR connectome (one-time per GpuContext lifetime)
        if let Err(e) = gpu_ctx.upload_connectome(row_ptr_sl, col_indices_sl, values_sl,
            nt_types_sl, external_I_sl)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_connectome: {e}")
            ));
        }

        // Initialize I = external_I
        if let Err(e) = gpu_ctx.upload_I(external_I_sl) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_I: {e}")
            ));
        }

        // Allocate device-side spike counter
        let d_spike_counter = match gpu_ctx.alloc_spike_counter() {
            Ok(ptr) => ptr,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("alloc_spike_counter: {e}")
            )),
        };

        let mut total_spikes: usize = 0;
        let mut step_spikes = vec![0i32; n_neurons];
        let t0 = std::time::Instant::now();

        for step in 0..n_steps {
            // One GPU step: Izhikevich → count spikes → CSR propagate
            // All on-device, no H2D/D2H (except d_spike_counter increment)
            if let Err(e) = gpu_ctx.connectome_step(dt, 0.99, Some(d_spike_counter)) {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("connectome_step: {e}")
                ));
            }

            // Download spikes to host (per-step D2H, needed for recording)
            if let Err(e) = gpu_ctx.download_spikes(&mut step_spikes) {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("download_spikes: {e}")
                ));
            }

            // Record to spike_buffer
            let buf_offset = step * n_neurons;
            let mut step_count = 0usize;
            for i in 0..n_neurons {
                if step_spikes[i] != 0 {
                    let idx = buf_offset + i;
                    if idx < spike_buffer_sl.len() {
                        spike_buffer_sl[idx] = 1;
                    }
                    step_count += 1;
                }
            }
            total_spikes += step_count;
        }

        // State persistence: download v, u back to host
        // Это ключевое исправление — без него каждый batch стартует с начального v,u
        let mut v_back = vec![0.0f32; n_neurons];
        let mut u_back = vec![0.0f32; n_neurons];
        let _ = gpu_ctx.download_v(&mut v_back);
        let _ = gpu_ctx.download_u(&mut u_back);

        // Write v, u back to Python arrays for persistence across calls
        let v_slice = v_data.as_slice_mut()?;
        let u_slice = u_data.as_slice_mut()?;
        let write_n = n_neurons.min(v_slice.len()).min(v_back.len())
            .min(u_slice.len()).min(u_back.len());
        for i in 0..write_n {
            v_slice[i] = v_back[i];
            u_slice[i] = u_back[i];
        }

        gpu_ctx.free_spike_counter(d_spike_counter);
        let elapsed = t0.elapsed().as_secs_f64();
        Ok((elapsed, total_spikes))
    }
}

// ─── GPU: ψ-wave helpers ─────────────────────────────────

#[cfg(feature = "cuda")]
fn gpu_setup<'a>(
    row_ptr: &Bound<'a, PyArray1<i32>>,
    col_indices: &Bound<'a, PyArray1<i32>>,
    values: &Bound<'a, PyArray1<f32>>,
    v: &Bound<'a, PyArray1<f32>>,
    u: &Bound<'a, PyArray1<f32>>,
    i_ext: Option<&Bound<'a, PyArray1<f32>>>,
) -> PyResult<(cuda_kernels::GpuContext, Vec<f32>, Vec<f32>)> {
    use cuda_kernels::GpuContext;

    let rp = row_ptr.readonly();
    let ci = col_indices.readonly();
    let vl = values.readonly();
    let vd = v.readwrite();
    let ud = u.readwrite();

    let rp_sl = rp.as_slice()?;
    let ci_sl = ci.as_slice()?;
    let vl_sl = vl.as_slice()?;
    let v_sl = vd.as_slice()?;
    let u_sl = ud.as_slice()?;

    let n = rp_sl.len() - 1;
    let n_syn = vl_sl.len();

    let v_host = v_sl.to_vec();
    let u_host = u_sl.to_vec();

    let ctx = GpuContext::new(n, n_syn)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("GpuContext::new: {e}")))?;

    ctx.upload_connectome(rp_sl, ci_sl, vl_sl, &vec![1.0f32; n], &vec![0.0f32; n])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("upload_connectome: {e}")))?;

    ctx.psi_upload(&v_host, &u_host)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("psi_upload: {e}")))?;

    let i_ext_sl = match i_ext {
        Some(arr) => arr.readonly().as_slice()?.to_vec(),
        None => vec![0.0f32; n],
    };
    ctx.psi_upload_drive(&i_ext_sl)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("psi_upload_drive: {e}")))?;

    Ok((ctx, v_host, u_host))
}

/// GPU ψ-wave на полном коннектоме.
///
/// Загружает CSR + ψ-state на GPU, прогоняет n_steps,
/// скачивает финал. Returns: (сек, v_final, u_final).
#[pyfunction]
#[pyo3(name = "bench_psi_wave_gpu", signature = (row_ptr, col_indices, values, v, u, dt, hbar, n_steps, i_ext=None))]
pub fn py_bench_psi_wave_gpu(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    dt: f32,
    hbar: f32,
    n_steps: usize,
    i_ext: Option<Bound<'_, PyArray1<f32>>>,
) -> PyResult<(f64, Vec<f32>, Vec<f32>)> {
    #[cfg(not(feature = "cuda"))]
    { let _ = (row_ptr, col_indices, values, v, u, dt, hbar, n_steps, i_ext);
      return Err(pyo3::exceptions::PyRuntimeError::new_err(
          "CUDA not available — rebuild with --features cuda")); }

    #[cfg(feature = "cuda")]
    {
        let (ctx, mut v_host, mut u_host) = gpu_setup(
            &row_ptr, &col_indices, &values, &v, &u, i_ext.as_ref())?;

        let t0 = std::time::Instant::now();
        for _ in 0..n_steps {
            ctx.psi_step(dt, hbar)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("psi_step: {e}")))?;
        }
        ctx.psi_download(&mut v_host, &mut u_host)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("psi_download: {e}")))?;

        Ok((t0.elapsed().as_secs_f64(), v_host, u_host))
    }
}

/// GPU ψ-wave + Aberton collapses на полном коннектоме.
///
/// Returns: (сек, v_final, u_final, collapse_count).
#[pyfunction]
#[pyo3(name = "bench_psi_aberton_gpu", signature = (
    row_ptr, col_indices, values, v, u, dt, hbar, n_steps,
    aber_gamma=0.1, aber_omega=1.0, aber_gain=1.0,
    aber_threshold=2.0, aber_collapse_scale=0.5,
    use_synaptic_field=false, i_ext=None, aber_real=None, aber_imag=None))]
pub fn py_bench_psi_aberton_gpu(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    v: Bound<'_, PyArray1<f32>>,
    u: Bound<'_, PyArray1<f32>>,
    dt: f32, hbar: f32, n_steps: usize,
    aber_gamma: f32, aber_omega: f32, aber_gain: f32,
    aber_threshold: f32, aber_collapse_scale: f32,
    use_synaptic_field: bool,
    i_ext: Option<Bound<'_, PyArray1<f32>>>,
    aber_real: Option<Bound<'_, PyArray1<f32>>>,
    aber_imag: Option<Bound<'_, PyArray1<f32>>>,
) -> PyResult<(f64, Vec<f32>, Vec<f32>, Vec<i32>)> {
    #[cfg(not(feature = "cuda"))]
    { let _ = (row_ptr, col_indices, values, v, u, dt, hbar, n_steps,
               aber_gamma, aber_omega, aber_gain, aber_threshold, aber_collapse_scale,
               use_synaptic_field, i_ext, aber_real, aber_imag);
      return Err(pyo3::exceptions::PyRuntimeError::new_err(
          "CUDA not available — rebuild with --features cuda")); }

    #[cfg(feature = "cuda")]
    {
        let (ctx, mut v_host, mut u_host) = gpu_setup(
            &row_ptr, &col_indices, &values, &v, &u, i_ext.as_ref())?;

        let n = row_ptr.readonly().as_slice()?.len() - 1;
        let (aber_real_sl, aber_imag_sl) = match (&aber_real, &aber_imag) {
            (Some(ar), Some(ai)) =>
                (ar.readonly().as_slice()?.to_vec(),
                 ai.readonly().as_slice()?.to_vec()),
            _ => (vec![0.0f32; n], vec![0.0f32; n]),
        };
        ctx.aber_upload(&aber_real_sl, &aber_imag_sl)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("aber_upload: {e}")))?;

        let mut collapse_counts = vec![0i32; n_steps];
        let t0 = std::time::Instant::now();

        for _ in 0..n_steps {
            ctx.psi_aberton_combined_async(dt, hbar,
                aber_gamma, aber_omega, aber_gain,
                aber_threshold, aber_collapse_scale, use_synaptic_field)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("psi_aberton: {e}")))?;
        }
        let _ = ctx.sync();

        ctx.psi_download(&mut v_host, &mut u_host)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("psi_download: {e}")))?;

        let mut aber_coll = vec![0i32; n];
        let mut aber_rh = vec![0.0f32; n];
        let mut aber_ih = vec![0.0f32; n];
        ctx.aber_download(&mut aber_rh, &mut aber_ih, &mut aber_coll)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("aber_download: {e}")))?;

        collapse_counts[n_steps - 1] = aber_coll.iter().sum();
        Ok((t0.elapsed().as_secs_f64(), v_host, u_host, collapse_counts))
    }
}
