//! PyO3 bindings for pillars — zero-copy numpy SIMD neural populations.
//!
//! API (Python):
//! ```python
//! from taas_checkmate.pillars import LIFPopulation, IzhikevichPopulation, HHPopulation
//!
//! # Create population
//! pop = LIFPopulation(1000)
//! pop.dt = 0.5          # scalar → all neurons
//! pop.inv_tau_m = 0.1
//! pop.v_rest = -70.0
//! pop.v_thresh = -54.0
//! pop.v_reset = -70.0
//! pop.tau_ref = 2.0
//! pop.v = -70.0          # init state
//!
//! # Per-neuron params with numpy arrays
//! import numpy as np
//! pop.set_param_array(2, np.linspace(-75, -65, 1000))
//!
//! # Simulate
//! current = np.ones(1000, dtype=np.float32) * 30.0
//! spikes = np.zeros(1000, dtype=np.int32)
//! for _ in range(1000):
//!     pop.step(current, spikes)
//!     print(f"Spikes: {spikes.sum()}")
//!
//! # Read states
//! v_all = pop.v   # numpy array
//! ```

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pillars::prelude::*;

// ═══════════════════════════════════════════════════════════════
//  LIFPopulation
// ═══════════════════════════════════════════════════════════════

/// Leaky Integrate-and-Fire (LIF) population — SIMD-accelerated.
///
/// Params table:
/// | idx | Name      | Default | Description |
/// |:---:|-----------|:-------:|-------------|
/// | 0   | dt        | 1.0     | Timestep (ms) |
/// | 1   | inv_tau_m | 0.1     | 1/τ_m — inverse membrane time constant |
/// | 2   | v_rest    | -70.0   | Resting potential (mV) |
/// | 3   | v_thresh  | -54.0   | Spike threshold (mV) |
/// | 4   | v_reset   | -70.0   | Reset potential (mV) |
/// | 5   | tau_ref   | 2.0     | Refractory period (ms) |
///
/// States:
/// | idx | Name | Description |
/// |:---:|------|-------------|
/// | 0   | v    | Membrane potential (mV) |
/// | 1   | refr | Refractory counter (ms) |
#[pyclass(name = "LIFPopulation", module = "taas_checkmate.pillars")]
pub struct PyLIFPopulation {
    inner: NeuralPopulation<LIFModel>,
}

#[pymethods]
impl PyLIFPopulation {
    /// Create LIF population with `n` neurons.
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<LIFModel>::new(n);
        // Set defaults
        pop.fill_param(0, 1.0);     // dt
        pop.fill_param(1, 0.1);     // inv_tau_m
        pop.fill_param(2, -70.0);   // v_rest
        pop.fill_param(3, -54.0);   // v_thresh
        pop.fill_param(4, -70.0);   // v_reset
        pop.fill_param(5, 2.0);     // tau_ref
        pop.fill_state(0, -70.0);   // v
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter]
    fn set_dt(&mut self, val: f32) { self.inner.fill_param(0, val); }
    #[setter]
    fn set_inv_tau_m(&mut self, val: f32) { self.inner.fill_param(1, val); }
    #[setter]
    fn set_v_rest(&mut self, val: f32) { self.inner.fill_param(2, val); }
    #[setter]
    fn set_v_thresh(&mut self, val: f32) { self.inner.fill_param(3, val); }
    #[setter]
    fn set_v_reset(&mut self, val: f32) { self.inner.fill_param(4, val); }
    #[setter]
    fn set_tau_ref(&mut self, val: f32) { self.inner.fill_param(5, val); }

    // ─── State getters ──────────────────────────────────────

    /// Membrane potential as numpy array (copy-on-read).
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = self.inner.state_slice(0);
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Refractory counter as numpy array.
    #[getter]
    fn refr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = self.inner.state_slice(1);
        Ok(PyArray1::from_slice(py, slice))
    }

    // ─── State setter (scalar → all neurons) ────────────────

    #[setter]
    fn set_v(&mut self, val: f32) { self.inner.fill_state(0, val); }
    #[setter]
    fn set_refr(&mut self, val: f32) { self.inner.fill_state(1, val); }

    // ─── Param getters (for introspection) ──────────────────

    #[getter]
    fn dt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.param_slice(0))
    }
    #[getter]
    fn inv_tau_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.param_slice(1))
    }
    #[getter]
    fn v_rest_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.param_slice(2))
    }

    // ─── Per-neuron param/state arrays ──────────────────────

    /// Set a param column from a numpy array (per-neuron values).
    ///
    /// Args:
    ///     col: param index (0=dt, 1=inv_tau_m, 2=v_rest, 3=v_thresh, 4=v_reset, 5=tau_ref)
    ///     array: float32 numpy array of length n
    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.param_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    /// Set a state column from a numpy array (per-neuron values).
    ///
    /// Args:
    ///     col: state index (0=v, 1=refr)
    ///     array: float32 numpy array of length n
    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.state_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    // ─── Step ───────────────────────────────────────────────

    /// Run one simulation step.
    ///
    /// Args:
    ///     i_in: input current (float32 numpy array, length n)
    ///     spikes_out: output spikes (int32 numpy array, length n, modified in-place)
    fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        self.inner.step(i_ro.as_slice()?, sp_rw.as_slice_mut()?);
        Ok(())
    }

    /// Number of neurons.
    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("LIFPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  IzhikevichPopulation
// ═══════════════════════════════════════════════════════════════

/// Izhikevich neuron population — regular/fast/bursting spiking.
///
/// Params table:
/// | idx | Name     | Default | Description |
/// |:---:|----------|:-------:|-------------|
/// | 0   | dt       | 0.5     | Timestep (ms) |
/// | 1   | a        | 0.02    | Recovery time scale |
/// | 2   | b        | 0.2     | Recovery sensitivity |
/// | 3   | c        | -65.0   | Reset potential (mV) |
/// | 4   | d        | 8.0     | After-spike recovery jump |
/// | 5   | tau_ref  | 2.0     | Refractory period (ms) |
/// | 6   | v_thresh | 30.0    | Spike threshold (mV) |
///
/// States:
/// | idx | Name | Description |
/// |:---:|------|-------------|
/// | 0   | v    | Membrane potential (mV) |
/// | 1   | u    | Recovery variable |
/// | 2   | refr | Refractory counter (ms) |
#[pyclass(name = "IzhikevichPopulation", module = "taas_checkmate.pillars")]
pub struct PyIzhikevichPopulation {
    inner: NeuralPopulation<IzhikevichModel>,
}

#[pymethods]
impl PyIzhikevichPopulation {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<IzhikevichModel>::new(n);
        pop.fill_param(0, 0.5);     // dt
        pop.fill_param(1, 0.02);    // a
        pop.fill_param(2, 0.2);     // b
        pop.fill_param(3, -65.0);   // c
        pop.fill_param(4, 8.0);     // d
        pop.fill_param(5, 2.0);     // tau_ref
        pop.fill_param(6, 30.0);    // v_thresh
        pop.fill_state(0, -65.0);   // v
        pop.fill_state(1, 0.2 * -65.0); // u ≈ b*v
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter] fn set_dt(&mut self, val: f32) { self.inner.fill_param(0, val); }
    #[setter] fn set_a(&mut self, val: f32) { self.inner.fill_param(1, val); }
    #[setter] fn set_b(&mut self, val: f32) { self.inner.fill_param(2, val); }
    #[setter] fn set_c(&mut self, val: f32) { self.inner.fill_param(3, val); }
    #[setter] fn set_d(&mut self, val: f32) { self.inner.fill_param(4, val); }
    #[setter] fn set_tau_ref(&mut self, val: f32) { self.inner.fill_param(5, val); }
    #[setter] fn set_v_thresh(&mut self, val: f32) { self.inner.fill_param(6, val); }

    // ─── State getters ──────────────────────────────────────

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(0))
    }
    #[getter]
    fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(1))
    }
    #[getter]
    fn refr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(2))
    }

    // ─── State setters ──────────────────────────────────────

    #[setter] fn set_v(&mut self, val: f32) { self.inner.fill_state(0, val); }
    #[setter] fn set_u(&mut self, val: f32) { self.inner.fill_state(1, val); }
    #[setter] fn set_refr(&mut self, val: f32) { self.inner.fill_state(2, val); }

    // ─── Per-neuron arrays ──────────────────────────────────

    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.param_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.state_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    // ─── Step ───────────────────────────────────────────────

    fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        self.inner.step(i_ro.as_slice()?, sp_rw.as_slice_mut()?);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("IzhikevichPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  HHPopulation
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley population — full Na⁺/K⁺ ion channel dynamics.
///
/// Params:
/// | idx | Name       | Default | Description |
/// |:---:|------------|:-------:|-------------|
/// | 0   | dt         | 0.01    | Timestep (ms) |
/// | 1   | g_na       | 120.0   | Na⁺ conductance (mS/cm²) |
/// | 2   | e_na       | 50.0    | Na⁺ reversal (mV) |
/// | 3   | g_k        | 36.0    | K⁺ conductance (mS/cm²) |
/// | 4   | e_k        | -77.0   | K⁺ reversal (mV) |
/// | 5   | g_leak     | 0.3     | Leak conductance (mS/cm²) |
/// | 6   | e_leak     | -54.387 | Leak reversal (mV) |
/// | 7   | q10        | 1.0     | Temperature factor (precomputed) |
/// | 8   | v_thresh   | 0.0     | Spike threshold (mV) |
/// | 9   | inv_cm     | 1.0     | 1/C_m — inverse membrane capacitance |
///
/// States:
/// | idx | Name       | Description |
/// |:---:|------------|-------------|
/// | 0   | v          | Membrane potential (mV) |
/// | 1   | m          | Na⁺ activation gate |
/// | 2   | h          | Na⁺ inactivation gate |
/// | 3   | n          | K⁺ activation gate |
/// | 4   | was_below  | Edge-triggered spike flag |
#[pyclass(name = "HHPopulation", module = "taas_checkmate.pillars")]
pub struct PyHHPopulation {
    inner: NeuralPopulation<HHModel>,
}

#[pymethods]
impl PyHHPopulation {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<HHModel>::new(n);
        pop.fill_param(0, 0.01);       // dt
        pop.fill_param(1, 120.0);      // g_na
        pop.fill_param(2, 50.0);       // e_na
        pop.fill_param(3, 36.0);       // g_k
        pop.fill_param(4, -77.0);      // e_k
        pop.fill_param(5, 0.3);        // g_leak
        pop.fill_param(6, -54.387);    // e_leak
        pop.fill_param(7, 1.0);        // q10 (6.3°C → 1.0)
        pop.fill_param(8, 0.0);        // v_thresh
        pop.fill_param(9, 1.0);        // inv_cm (C_m=1.0)
        pop.fill_state(0, -65.0);      // v
        pop.fill_state(1, 0.05);       // m (steady-state at -65mV)
        pop.fill_state(2, 0.6);        // h
        pop.fill_state(3, 0.32);       // n
        pop.fill_state(4, 1.0);        // was_below (below threshold)
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter] fn set_dt(&mut self, val: f32) { self.inner.fill_param(0, val); }
    #[setter] fn set_g_na(&mut self, val: f32) { self.inner.fill_param(1, val); }
    #[setter] fn set_e_na(&mut self, val: f32) { self.inner.fill_param(2, val); }
    #[setter] fn set_g_k(&mut self, val: f32) { self.inner.fill_param(3, val); }
    #[setter] fn set_e_k(&mut self, val: f32) { self.inner.fill_param(4, val); }
    #[setter] fn set_g_leak(&mut self, val: f32) { self.inner.fill_param(5, val); }
    #[setter] fn set_e_leak(&mut self, val: f32) { self.inner.fill_param(6, val); }
    #[setter] fn set_q10(&mut self, val: f32) { self.inner.fill_param(7, val); }
    #[setter] fn set_v_thresh(&mut self, val: f32) { self.inner.fill_param(8, val); }
    #[setter] fn set_inv_cm(&mut self, val: f32) { self.inner.fill_param(9, val); }

    // ─── State getters ──────────────────────────────────────

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(0))
    }
    #[getter]
    fn m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(1))
    }
    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(2))
    }
    #[getter]
    fn n_gate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.state_slice(3))
    }

    // ─── State setters ──────────────────────────────────────

    #[setter] fn set_v(&mut self, val: f32) { self.inner.fill_state(0, val); }
    #[setter] fn set_m(&mut self, val: f32) { self.inner.fill_state(1, val); }
    #[setter] fn set_h(&mut self, val: f32) { self.inner.fill_state(2, val); }
    #[setter] fn set_n_gate(&mut self, val: f32) { self.inner.fill_state(3, val); }

    // ─── Per-neuron arrays ──────────────────────────────────

    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.param_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.state_slice_mut(col),
                self.inner.len(),
            )
        };
        let n = dest.len().min(src.len());
        dest[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    // ─── Step ───────────────────────────────────────────────

    fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        self.inner.step(i_ro.as_slice()?, sp_rw.as_slice_mut()?);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("HHPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════

/// Register the pillars sub-module in the Python module tree.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "pillars")?;
    sub.add_class::<PyLIFPopulation>()?;
    sub.add_class::<PyIzhikevichPopulation>()?;
    sub.add_class::<PyHHPopulation>()?;
    m.add_submodule(&sub)?;
    // Also register at root for convenience
    m.add_class::<PyLIFPopulation>()?;
    m.add_class::<PyIzhikevichPopulation>()?;
    m.add_class::<PyHHPopulation>()?;
    Ok(())
}
