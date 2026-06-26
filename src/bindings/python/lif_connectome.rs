//! PyO3 bindings for LIF connectome benchmark — CPU (SIMD+rayon).
//!
//! Точная копия bench_connectome_cpu_recording, но с LIF вместо Izhikevich.
//! LIF params: v, tau_m, v_rest, v_thresh, v_reset (вместо v,u,trace,a,b,c,d).
//!
//! Поддерживает alpha-синапс (τ_syn) и синаптическую задержку (delay_steps)
//! из fly-brain (Shiu et al. 2024, Nature).
//!
//! Returns: (wall_time_sec, total_spikes).

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use cuda_kernels::GpuContext;

use crate::models::neuron;

/// LIF connectome simulation with per-step spike recording.
///
/// Как bench_connectome_cpu_recording, но LIF вместо Izhikevich.
///
/// Supports alpha-synapse (τ_syn) and synaptic delay (delay_steps)
/// from fly-brain (Shiu et al. 2024, Nature 634, 124–138).
///
/// Alpha-synapse:
///   g_syn[i] *= (1 - dt / τ_syn)  (exponential decay)
///   CSR propagate → g_syn[post] += w * nt_type
///   I_buf[i] = external_I[i] + g_syn[i]
///
/// Synaptic delay:
///   delay_buf: [n × delay_steps] circular buffer
///   Each step: write propagation to delay_buf[delay_pos], advance delay_pos
///   Read delay_buf[delay_pos] into g_syn before writing (delay_steps delay)
///
/// Args:
///     row_ptr, col_indices, values: CSR connectome
///     n_steps: number of simulation steps
///     dt: timestep (ms)
///     v: membrane potential [n] (padded to multiple of 8)
///     tau_m: membrane time constant [n]
///     v_rest: resting potential [n]
///     v_thresh: threshold potential [n]
///     v_reset: reset potential [n]
///     nt_types: neurotransmitter types (+1 exc, -1 inh) [n]
///     external_I: external input current [n]
///     I_buf: current buffer [n] (will be modified)
///     spikes: spike buffer [n] (will be filled 0/1)
///     spike_buffer: [n_steps × n] — per-step spike recording
///     g_syn: synaptic conductance [n] (persistent, can be zeros(len 0) to disable)
///     tau_syn: synaptic time constant (ms). 0 = instant (old behavior)
///     delay_buf: delay buffer [n × delay_steps] (can be zeros(len 0) to disable)
///     delay_steps: number of delay timesteps. 0 = no delay
///     delay_pos: current write position in delay_buf [1]
///
/// Returns: (wall_time_sec, total_spikes)
#[pyfunction]
#[pyo3(name = "lif_sim_cpu_recording")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn py_lif_sim_cpu_recording(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    tau_m: Bound<'_, PyArray1<f32>>,
    v_rest: Bound<'_, PyArray1<f32>>,
    v_thresh: Bound<'_, PyArray1<f32>>,
    v_reset: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    I_buf: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
    // Alpha-synapse + delay parameters
    g_syn: Bound<'_, PyArray1<f32>>,
    tau_syn: f32,
    delay_buf: Bound<'_, PyArray1<f32>>,
    delay_steps: usize,
    delay_pos: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    let mut v = v.readwrite();
    let tau_m = tau_m.readonly();
    let v_rest = v_rest.readonly();
    let v_thresh = v_thresh.readonly();
    let v_reset = v_reset.readonly();
    let nt_types = nt_types.readonly();
    let external_I = external_I.readonly();
    let mut I_buf = I_buf.readwrite();
    let mut spikes = spikes.readwrite();
    let mut spike_buffer = spike_buffer.readwrite();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();
    let mut g_syn = g_syn.readwrite();
    let mut delay_buf = delay_buf.readwrite();
    let mut delay_pos = delay_pos.readwrite();

    let v = v.as_slice_mut()?;
    let tau_m = tau_m.as_slice()?;
    let v_rest = v_rest.as_slice()?;
    let v_thresh = v_thresh.as_slice()?;
    let v_reset = v_reset.as_slice()?;
    let nt_types = nt_types.as_slice()?;
    let external_I = external_I.as_slice()?;
    let I_buf = I_buf.as_slice_mut()?;
    let spikes = spikes.as_slice_mut()?;
    let spike_buffer = spike_buffer.as_slice_mut()?;
    let row_ptr = row_ptr.as_slice()?;
    let col_indices = col_indices.as_slice()?;
    let values = values.as_slice()?;
    let g_syn = g_syn.as_slice_mut()?;
    let delay_buf = delay_buf.as_slice_mut()?;
    let delay_pos = delay_pos.as_slice_mut()?;

    let n = v.len();
    let n_neurons = row_ptr.len() - 1;
    let mut refractory = vec![0.0f32; n];

    // Features enabled?
    let use_alpha = tau_syn > 0.0 && g_syn.len() >= n;
    let use_delay = delay_steps > 0 && delay_buf.len() >= n * delay_steps;

    let mut total_spikes: usize = 0;

    let t0 = std::time::Instant::now();

    for step in 0..n_steps {
        // ─── 0. Set I = external_I + g_syn (if alpha-synapse) ───
        if use_alpha {
            for i in 0..n.min(external_I.len()) {
                I_buf[i] = external_I[i] + g_syn[i];
            }
            for i in n.min(external_I.len())..n {
                I_buf[i] = 0.0;
            }
        } else {
            for i in 0..n.min(external_I.len()) {
                I_buf[i] = external_I[i];
            }
            for i in n.min(external_I.len())..n {
                I_buf[i] = 0.0;
            }
        }

        // ─── 1. LIF step (SIMD + rayon) ───
        neuron::step_lif(
            v, I_buf, spikes, dt,
            tau_m, v_rest, v_thresh, v_reset,
            &mut refractory, neuron::DEFAULT_TAU_REF,
        );

        // ─── 2. Record to spike_buffer + reset g_syn на spike ───
        //   Brian2 reference: eq_rst = 'v = v_rst; w = 0; g = 0 * mV'
        //   Очищаем g_syn[i] для каждого спайка, чтобы избежать
        //   остаточного drive после рефрактерности.
        let buf_offset = step * n_neurons;
        for i in 0..n_neurons.min(spikes.len()) {
            if spikes[i] != 0 {
                if use_alpha && i < g_syn.len() {
                    g_syn[i] = 0.0;  // Brian2: g = 0 on spike
                }
                let idx = buf_offset + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
                total_spikes += 1;
            }
        }

        if use_alpha {
            // ─── 3. Decay synaptic conductance: g_syn *= (1 - dt/τ_syn) ───
            let decay = 1.0 - dt / tau_syn;
            for i in 0..n {
                g_syn[i] *= decay;
            }

            // Alpha-synapse scale factor: нормализуем заряд
            // g_syn[i] *= (1 - dt/τ_syn) — распад. Каждый спайк даёт заряд w,
            // который распределяется по τ_syn/dt шагам. Без scale factor общий
            // заряд был бы w / (1 - (1 - dt/τ_syn)) = w × τ_syn/dt = w × 10.
            // Scale: dt/τ_syn → общий заряд = w × (dt/τ_syn) / (dt/τ_syn) = w. ✓
            let syn_scale = dt / tau_syn;

            if use_delay {
                // ─── 4. Read delayed input from delay_buf into g_syn ───
                // dp читается ВНУТРИ цикла для каждого шага (персистентен между batch calls)
                if !delay_pos.is_empty() {
                    let dp = delay_pos[0] as usize % delay_steps;
                    let offset = dp * n;
                    for i in 0..n {
                        g_syn[i] += delay_buf[offset + i];
                    }
                    // Clear the read position (will be overwritten by propagate)
                    delay_buf[offset..offset + n].fill(0.0);

                    // ─── 5. Propagate spikes into delay_buf ───
                    let write_offset = dp * n;
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
                                if t0 < n { delay_buf[write_offset + t0] += values[j] * mult * syn_scale; }
                                if t1 < n { delay_buf[write_offset + t1] += values[j + 1] * mult * syn_scale; }
                                if t2 < n { delay_buf[write_offset + t2] += values[j + 2] * mult * syn_scale; }
                                if t3 < n { delay_buf[write_offset + t3] += values[j + 3] * mult * syn_scale; }
                                j += 4;
                            }
                            for j in j..end {
                                let col = col_indices[j] as usize;
                                if col < n { delay_buf[write_offset + col] += values[j] * mult * syn_scale; }
                            }
                        }
                    }

                    // Advance delay position
                    let new_dp = (dp + 1) % delay_steps;
                    delay_pos[0] = new_dp as i32;
                }
            } else {
                // No delay: propagate directly to g_syn
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
                            if t0 < n { g_syn[t0] += values[j] * mult * syn_scale; }
                            if t1 < n { g_syn[t1] += values[j + 1] * mult * syn_scale; }
                            if t2 < n { g_syn[t2] += values[j + 2] * mult * syn_scale; }
                            if t3 < n { g_syn[t3] += values[j + 3] * mult * syn_scale; }
                            j += 4;
                        }
                        for j in j..end {
                            let col = col_indices[j] as usize;
                            if col < n { g_syn[col] += values[j] * mult * syn_scale; }
                        }
                    }
                }
            }
        } else {
            // ─── Old behavior (no alpha-synapse): ───
            // 3. Reset I = external_I
            for i in 0..n.min(external_I.len()) {
                I_buf[i] = external_I[i];
            }
            for i in n.min(external_I.len())..n {
                I_buf[i] = 0.0;
            }

            // 4. CSR propagate spikes directly onto I_buf
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
    }

    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, total_spikes))
}


// ─── GPU: LIF connectome with per-step spike recording ─────

/// LIF connectome simulation на GPU (CUDA) с записью per-step спайков.
///
/// Все данные на GPU между шагами — без H2D/D2H (кроме D2H спайков для записи).
///
/// Args: как py_lif_sim_cpu_recording, но без I_buf и spikes (GPU управляет сам).
/// CPU v, tau_m, v_rest, v_thresh, v_reset передаются для инициализации GPU.
/// v in/out: начальное/конечное v (persistent между batch calls).
///
/// Returns: (wall_time_sec, total_spikes).
#[pyfunction]
#[pyo3(name = "lif_sim_gpu_recording")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn py_lif_sim_gpu_recording(
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    dt: f32,
    v: Bound<'_, PyArray1<f32>>,
    tau_m: Bound<'_, PyArray1<f32>>,
    v_rest: Bound<'_, PyArray1<f32>>,
    v_thresh: Bound<'_, PyArray1<f32>>,
    v_reset: Bound<'_, PyArray1<f32>>,
    nt_types: Bound<'_, PyArray1<f32>>,
    external_I: Bound<'_, PyArray1<f32>>,
    spike_buffer: Bound<'_, PyArray1<i32>>,
) -> PyResult<(f64, usize)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (row_ptr, col_indices, values, n_steps, dt,
                 v, tau_m, v_rest, v_thresh, v_reset,
                 nt_types, external_I, spike_buffer);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "CUDA not available — rebuild with --features cuda".to_string()
        ));
    }

    #[cfg(feature = "cuda")]
    {
        let row_ptr_data = row_ptr.readonly();
        let col_indices_data = col_indices.readonly();
        let values_data = values.readonly();
        let mut v_data = v.readwrite();
        let tau_m_data = tau_m.readonly();
        let v_rest_data = v_rest.readonly();
        let v_thresh_data = v_thresh.readonly();
        let v_reset_data = v_reset.readonly();
        let nt_types_data = nt_types.readonly();
        let external_I_data = external_I.readonly();
        let mut spike_buffer_data = spike_buffer.readwrite();

        let row_ptr_sl = row_ptr_data.as_slice()?;
        let col_indices_sl = col_indices_data.as_slice()?;
        let values_sl = values_data.as_slice()?;
        let tau_m_sl = tau_m_data.as_slice()?;
        let v_rest_sl = v_rest_data.as_slice()?;
        let v_thresh_sl = v_thresh_data.as_slice()?;
        let v_reset_sl = v_reset_data.as_slice()?;
        let nt_types_sl = nt_types_data.as_slice()?;
        let external_I_sl = external_I_data.as_slice()?;
        let spike_buffer_sl = spike_buffer_data.as_slice_mut()?;

        let n_neurons = row_ptr_sl.len() - 1;
        let n_synapses = values_sl.len();

        // Copy initial v from host (state persistence across batch calls)
        let v_host = v_data.as_slice()?.to_vec();

        // Create GPU context
        let gpu_ctx = match GpuContext::new(n_neurons, n_synapses) {
            Ok(ctx) => ctx,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("GpuContext::new failed: {e}")
            )),
        };

        // Upload v to GPU (d_v). LIF doesn't use u,a,b,c,d but upload_neurons needs them.
        let dummy_u = vec![0.0f32; n_neurons];
        let dummy_a = vec![0.02f32; n_neurons];
        let dummy_b = vec![0.2f32; n_neurons];
        let dummy_c = vec![-70.0f32; n_neurons];
        let dummy_d = vec![2.0f32; n_neurons];
        if let Err(e) = gpu_ctx.upload_neurons(&v_host, &dummy_u,
            &dummy_a, &dummy_b, &dummy_c, &dummy_d)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_neurons: {e}")
            ));
        }

        // Upload LIF params to d_lif_params (one-time)
        if let Err(e) = gpu_ctx.upload_lif_params(dt,
            tau_m_sl, v_rest_sl, v_thresh_sl, v_reset_sl)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_lif_params: {e}")
            ));
        }

        // Upload CSR connectome + nt_types + external_I (one-time)
        if let Err(e) = gpu_ctx.upload_connectome(row_ptr_sl, col_indices_sl, values_sl,
            nt_types_sl, external_I_sl)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("upload_connectome: {e}")
            ));
        }

        // Initialize I = external_I (needed for first LIF step)
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
            // One GPU step: LIF → count spikes → CSR propagate
            // All on-device, no H2D/D2H (except d_spike_counter increment)
            if let Err(e) = gpu_ctx.connectome_lif_step(Some(d_spike_counter)) {
                gpu_ctx.free_spike_counter(d_spike_counter);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("connectome_lif_step: {e}")
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
            for i in 0..n_neurons {
                if step_spikes[i] != 0 {
                    let idx = buf_offset + i;
                    if idx < spike_buffer_sl.len() {
                        spike_buffer_sl[idx] = 1;
                    }
                    total_spikes += 1;
                }
            }
        }

        // State persistence: download v back to host
        let mut v_back = vec![0.0f32; n_neurons];
        let _ = gpu_ctx.download_v(&mut v_back);

        let v_slice = v_data.as_slice_mut()?;
        let write_n = n_neurons.min(v_slice.len()).min(v_back.len());
        for i in 0..write_n {
            v_slice[i] = v_back[i];
        }

        gpu_ctx.free_spike_counter(d_spike_counter);
        let elapsed = t0.elapsed().as_secs_f64();
        Ok((elapsed, total_spikes))
    }
}
