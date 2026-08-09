//! PyO3 bindings for ChaosPatternEngine (CPE).
//!
//! Exposes CPE's core functions: kuramoto_step, cpe_phase_coherence,
//! dopamine_delta, cpe_stdp_update, eligibility_trace.
//!
//! NOTE: `phase_coherence`/`stdp_update` are prefixed `cpe_` because
//! `export_fn` adds every function to BOTH the submodule and the root
//! module — the cpe variants (registered last) would otherwise clobber
//! the kuramoto/neuron versions at root.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Kuramoto step on sparse CSR connectome (Python wrapper).
///
/// Args:
///     phases: phase array [n] (modified in-place)
///     omegas: natural frequencies [n]
///     row_ptr: CSR row pointer [n+1]
///     col_indices: CSR column indices [n_syn]
///     values: CSR weight values [n_syn]
///     dt: timestep
#[pyfunction]
#[pyo3(name = "kuramoto_step")]
pub fn py_cpe_kuramoto_step(
    phases: Bound<'_, PyArray1<f32>>,
    omegas: Bound<'_, PyArray1<f32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    col_indices: Bound<'_, PyArray1<i32>>,
    values: Bound<'_, PyArray1<f32>>,
    dt: f32,
) -> PyResult<()> {
    let mut phases = phases.readwrite();
    let omegas = omegas.readonly();
    let row_ptr = row_ptr.readonly();
    let col_indices = col_indices.readonly();
    let values = values.readonly();

    pillars::api::cpe::kuramoto_step(
        phases.as_slice_mut()?,
        omegas.as_slice()?,
        row_ptr.as_slice()?,
        col_indices.as_slice()?,
        values.as_slice()?,
        None,
        dt,
    );
    Ok(())
}

/// Phase coherence R ∈ [0, 1].
///
/// NOTE: named `cpe_phase_coherence` to avoid clobbering the 4-arg
/// `kuramoto.phase_coherence` in the root module (export_fn adds every
/// function to both root and submodule).
#[pyfunction]
#[pyo3(name = "cpe_phase_coherence")]
pub fn py_cpe_phase_coherence(phases: Bound<'_, PyArray1<f32>>) -> f32 {
    let phases = phases.readonly();
    let phases_slice = match phases.as_slice() {
        Ok(s) => s,
        Err(_) => return 0.0,
    };
    pillars::api::cpe::phase_coherence(phases_slice)
}

/// Spike entropy H and its deviation from target.
#[pyfunction]
#[pyo3(name = "spike_entropy")]
pub fn py_cpe_spike_entropy(
    counts: Bound<'_, PyArray1<f32>>,
    total_steps: usize,
    h_target: f32,
) -> (f32, f32) {
    let counts = counts.readonly();
    let counts_slice = match counts.as_slice() {
        Ok(s) => s,
        Err(_) => return (0.0, 0.0),
    };
    pillars::api::cpe::spike_entropy(counts_slice, total_steps, h_target)
}

/// Detect spikes: rising edge through threshold.
///
/// Writes spike indicators [n] (0/1) into spikes_out buffer.
#[pyfunction]
#[pyo3(name = "detect_spikes")]
pub fn py_cpe_detect_spikes(
    spikes_out: Bound<'_, PyArray1<i32>>,
    phases: Bound<'_, PyArray1<f32>>,
    phases_prev: Bound<'_, PyArray1<f32>>,
    threshold: f32,
) -> PyResult<()> {
    let mut spikes_out = spikes_out.readwrite();
    let phases = phases.readonly();
    let phases_prev = phases_prev.readonly();
    if let (Ok(sp), Ok(p), Ok(pp)) = (
        spikes_out.as_slice_mut(),
        phases.as_slice(),
        phases_prev.as_slice(),
    ) {
        let result = pillars::api::cpe::detect_spikes(p, pp, threshold);
        let n = sp.len().min(result.len());
        sp[..n].copy_from_slice(&result[..n]);
    }
    Ok(())
}

/// Compute dopamine δ = reward + β · [(R_pat − α·|H_pat−H₀|) − (R_chaos − α·|H_chaos−H₀|)]
#[pyfunction]
#[pyo3(name = "dopamine_delta")]
pub fn py_cpe_dopamine_delta(
    reward: f32,
    pattern_r: f32,
    pattern_h_err: f32,
    chaos_r: f32,
    chaos_h_err: f32,
    alpha: f32,
    beta: f32,
) -> f32 {
    let config = pillars::api::cpe::CpeConfig {
        alpha,
        beta,
        ..pillars::api::cpe::CpeConfig::default()
    };
    pillars::api::cpe::dopamine_delta(
        &config,
        reward,
        pattern_r,
        pattern_h_err,
        chaos_r,
        chaos_h_err,
    )
}

/// Dopamine timing gate: scales [0, 1] based on critical steps ratio.
#[pyfunction]
#[pyo3(name = "dopamine_timing_gate")]
pub fn py_cpe_dopamine_timing_gate(critical_steps: usize, total_steps: usize) -> f32 {
    pillars::api::cpe::dopamine_timing_gate(critical_steps, total_steps)
}

/// Eligibility trace update: decay + add on spike.
#[pyfunction]
#[pyo3(name = "eligibility_update")]
pub fn py_cpe_eligibility_update(
    trace: Bound<'_, PyArray1<f32>>,
    spikes: Bound<'_, PyArray1<i32>>,
    row_ptr: Bound<'_, PyArray1<i32>>,
    elig_add: f32,
    elig_decay: f32,
) -> PyResult<()> {
    let mut trace = trace.readwrite();
    let spikes = spikes.readonly();
    let row_ptr = row_ptr.readonly();

    let trace_slice = trace.as_slice_mut()?;
    let spikes_slice = spikes.as_slice()?;
    let row_ptr_slice = row_ptr.as_slice()?;

    let n = row_ptr_slice.len().saturating_sub(1);
    for pre in 0..n {
        if pre >= spikes_slice.len() || spikes_slice[pre] == 0 {
            continue;
        }
        let start = row_ptr_slice[pre] as usize;
        let end = row_ptr_slice[pre + 1] as usize;
        for idx in start..end {
            if idx < trace_slice.len() {
                trace_slice[idx] += elig_add;
            }
        }
    }
    for t in trace_slice.iter_mut() {
        *t *= elig_decay;
    }
    Ok(())
}

/// Dopamine-STDP weight update with sign-preserving clip.
#[pyfunction]
/// NOTE: named `cpe_stdp_update` — see `cpe_phase_coherence` for the
/// root-module clobbering rationale (neuron.stdp_update wins at root).
#[pyo3(name = "cpe_stdp_update")]
pub fn py_cpe_stdp_update(
    w: Bound<'_, PyArray1<f32>>,
    delta: f32,
    elig: Bound<'_, PyArray1<f32>>,
    gate: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
) -> PyResult<()> {
    let mut w = w.readwrite();
    let elig = elig.readonly();
    let w_slice = w.as_slice_mut()?;
    let elig_slice = elig.as_slice()?;

    let n = w_slice.len().min(elig_slice.len());
    for i in 0..n {
        w_slice[i] += lr * delta * elig_slice[i] * gate;
        if w_slice[i] >= 0.0 {
            w_slice[i] = w_slice[i].clamp(w_min, w_max);
        } else {
            w_slice[i] = w_slice[i].clamp(-w_max, -w_min);
        }
    }
    Ok(())
}
