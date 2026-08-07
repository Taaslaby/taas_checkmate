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
use pillars::zoo::TriangulatedModel;

// ═══════════════════════════════════════════════════════════════
//  define_population! — shared pyo3 boilerplate for the point-neuron
//  zoo models. Each model gets the same surface: scalar param setters
//  (all neurons) + numpy param getters, scalar state setters + numpy
//  state getters, set_param_array/set_state_array, step(i, spikes),
//  n and __repr__. Only the param/state column layout and init defaults
//  differ per model, so they are declared once per model below, keeping
//  each new wrapper a few lines instead of ~150 lines of near-identical
//  boilerplate.
// ═══════════════════════════════════════════════════════════════

macro_rules! define_population {
    (
        docs: $docs:literal,
        class: $class:ident,
        py_name: $pyname:literal,
        model: $model:ty,
        // Each entry: (property_name_as_ident, setter_fn_name, column_idx, default).
        // The setter name is spelled out because macro_rules cannot splice an
        // ident into `set_$p` — and pyo3 infers the property name from the
        // method name verbatim (strips `set_`), so setter must match getter.
        // Defaults are passed as data (not as a call-site block) because a
        // `$init:block` metavariable is hygienic: identifiers inside it cannot
        // see `pop` defined in this macro's expansion.
        params: [ $( ( $p:ident , $pset:ident , $p_idx:literal , $p_default:expr ) ),* $(,)? ],
        states: [ $( ( $s:ident , $sset:ident , $s_idx:literal , $s_default:expr ) ),* $(,)? ],
        // Optional extra methods spliced into the same #[pymethods] block.
        // pyo3 rejects a second #[pymethods] impl for one class, so anything
        // model-specific (e.g. Triangulated::calibrate_lif) goes here.
        extra: { $( $m:item )* }
    ) => {
        #[doc = $docs]
        #[pyclass(name = $pyname, module = "taas_checkmate.pillars")]
        pub struct $class {
            inner: NeuralPopulation<$model>,
        }

        #[pymethods]
        impl $class {
            #[new]
            fn new(n: usize) -> Self {
                let mut pop = NeuralPopulation::<$model>::new(n);
                $(
                    pop.fill_param($p_idx, $p_default);
                )*
                $(
                    pop.fill_state($s_idx, $s_default);
                )*
                Self { inner: pop }
            }

            // ─── Params: getter (numpy) + scalar setter ───
            $(
                #[getter]
                fn $p<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
                    PyArray1::from_slice(py, self.inner.param_slice($p_idx))
                }
                #[setter]
                fn $pset(&mut self, val: f32) { self.inner.fill_param($p_idx, val); }
            )*

            // ─── States: getter (numpy) + scalar setter ───
            $(
                #[getter]
                fn $s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
                    PyArray1::from_slice(py, self.inner.state_slice($s_idx))
                }
                #[setter]
                fn $sset(&mut self, val: f32) { self.inner.fill_state($s_idx, val); }
            )*

            /// Set a param column from a numpy array (per-neuron values).
            fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
                let ro = array.readonly();
                let src = ro.as_slice()?;
                let dest = self.inner.param_slice_mut(col);
                if src.len() != dest.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "set_param_array: expected {} elements (n), got {}",
                        dest.len(),
                        src.len()
                    )));
                }
                dest.copy_from_slice(src);
                Ok(())
            }

            /// Set a state column from a numpy array (per-neuron values).
            fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
                let ro = array.readonly();
                let src = ro.as_slice()?;
                let dest = self.inner.state_slice_mut(col);
                if src.len() != dest.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "set_state_array: expected {} elements (n), got {}",
                        dest.len(),
                        src.len()
                    )));
                }
                dest.copy_from_slice(src);
                Ok(())
            }

            /// Run one simulation step.
            fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
                let i_ro = i_in.readonly();
                let mut sp_rw = spikes_out.readwrite();
                let i = i_ro.as_slice()?;
                let sp = sp_rw.as_slice_mut()?;
                let n = self.inner.len();
                if i.len() != n || sp.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "step: expected i_in and spikes_out of length {n}, got {} and {}",
                        i.len(),
                        sp.len()
                    )));
                }
                self.inner.step(i, sp);
                Ok(())
            }

            /// Number of neurons.
            #[getter]
            fn n(&self) -> usize { self.inner.len() }

            fn __repr__(&self) -> String {
                format!("{}(n={})", $pyname, self.inner.len())
            }

            // ─── Model-specific extras ───
            $( $m )*
        }
    };
}

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
        let dest = self.inner.param_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_param_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
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
        let dest = self.inner.state_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_state_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
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
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
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
        let dest = self.inner.param_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_param_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
        Ok(())
    }

    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = self.inner.state_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_state_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
        Ok(())
    }

    // ─── Step ───────────────────────────────────────────────

    fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
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
        let dest = self.inner.param_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_param_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
        Ok(())
    }

    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = self.inner.state_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_state_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
        Ok(())
    }

    // ─── Step ───────────────────────────────────────────────

    fn step(&mut self, i_in: Bound<'_, PyArray1<f32>>, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("HHPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  VSNPopulation (Variable Spiking Neuron — Surrogate Gradient)
// ═══════════════════════════════════════════════════════════════

/// Variable Spiking Neuron (VSN) population — SIMD-accelerated with surrogate gradient.
///
/// This is the core neuron model used in Spiking-GPT and all modern SNN architectures.
///
/// Params table:
/// | idx | Name           | Default | Description |
/// |:---:|----------------|:-------:|-------------|
/// | 0   | dt             | 0.5     | Timestep (ms) |
/// | 1   | inv_tau_m      | 0.1     | 1/τ_m — inverse membrane time constant |
/// | 2   | v_rest         | -70.0   | Resting potential (mV) |
/// | 3   | v_thresh       | -54.0   | Spike threshold (mV) |
/// | 4   | v_reset        | -70.0   | Reset potential (mV) |
/// | 5   | tau_ref        | 2.0     | Refractory period (ms) |
/// | 6   | beta           | 0.9     | Leakage mixing factor (0..1) |
/// | 7   | surrogate_gain | 10.0    | Surrogate gradient gain |
///
/// States:
/// | idx | Name      | Description |
/// |:---:|-----------|-------------|
/// | 0   | v         | Membrane potential (mV) |
/// | 1   | refr      | Refractory counter (ms) |
/// | 2   | surrogate | Surrogate gradient value |
#[pyclass(name = "VSNPopulation", module = "taas_checkmate.pillars")]
pub struct PyVSNPopulation {
    inner: NeuralPopulation<VSNModel>,
}

#[pymethods]
impl PyVSNPopulation {
    /// Create VSN population with `n` neurons.
    ///
    /// Args:
    ///     n: number of neurons
    ///     dt: timestep in ms (default 0.5)
    ///     tau_m: membrane time constant in ms (default 10.0, stored as 1/tau_m)
    ///     v_rest: resting potential in mV (default -70.0)
    ///     v_thresh: spike threshold in mV (default -54.0)
    ///     v_reset: reset potential in mV (default -70.0)
    ///     beta: leakage mixing factor 0..1 (default 0.9)
    ///     surrogate_gain: surrogate gradient gain (default 10.0)
    #[new]
    #[pyo3(signature = (n, dt=0.5, tau_m=10.0, v_rest=-70.0, v_thresh=-54.0, v_reset=-70.0, beta=0.9, surrogate_gain=10.0))]
    fn new(
        n: usize,
        dt: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        beta: f32,
        surrogate_gain: f32,
    ) -> Self {
        let mut pop = NeuralPopulation::<VSNModel>::new(n);
        pop.fill_param(0, dt);
        pop.fill_param(1, 1.0 / tau_m);  // Pre-inverted: store 1/τ_m
        pop.fill_param(2, v_rest);
        pop.fill_param(3, v_thresh);
        pop.fill_param(4, v_reset);
        pop.fill_param(5, 2.0);          // tau_ref
        pop.fill_param(6, beta);
        pop.fill_param(7, surrogate_gain);
        pop.fill_state(0, v_rest);       // v = v_rest
        pop.fill_state(1, 0.0);          // refr
        pop.fill_state(2, 0.0);          // surrogate
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter] fn set_dt(&mut self, val: f32) { self.inner.fill_param(0, val); }
    #[setter] fn set_inv_tau_m(&mut self, val: f32) { self.inner.fill_param(1, val); }
    /// Set tau_m in ms — automatically stores as 1/tau_m (pre-inverted).
    #[setter] fn set_tau_m(&mut self, val: f32) { self.inner.fill_param(1, 1.0 / val); }
    #[setter] fn set_v_rest(&mut self, val: f32) { self.inner.fill_param(2, val); }
    #[setter] fn set_v_thresh(&mut self, val: f32) { self.inner.fill_param(3, val); }
    #[setter] fn set_v_reset(&mut self, val: f32) { self.inner.fill_param(4, val); }
    #[setter] fn set_tau_ref(&mut self, val: f32) { self.inner.fill_param(5, val); }
    #[setter] fn set_beta(&mut self, val: f32) { self.inner.fill_param(6, val); }
    #[setter] fn set_surrogate_gain(&mut self, val: f32) { self.inner.fill_param(7, val); }

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

    /// Surrogate gradient as numpy array.
    #[getter]
    fn surrogate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = self.inner.state_slice(2);
        Ok(PyArray1::from_slice(py, slice))
    }

    // ─── State setters ──────────────────────────────────────

    #[setter] fn set_v(&mut self, val: f32) { self.inner.fill_state(0, val); }
    #[setter] fn set_refr(&mut self, val: f32) { self.inner.fill_state(1, val); }
    #[setter] fn set_surrogate(&mut self, val: f32) { self.inner.fill_state(2, val); }

    // ─── Per-neuron arrays ──────────────────────────────────

    /// Set a param column from a numpy array (per-neuron values).
    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = self.inner.param_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_param_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
        Ok(())
    }

    /// Set a state column from a numpy array (per-neuron values).
    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = array.readonly();
        let src = ro.as_slice()?;
        let dest = self.inner.state_slice_mut(col);
        if src.len() != dest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "set_state_array: expected {} elements (n), got {}",
                dest.len(),
                src.len()
            )));
        }
        dest.copy_from_slice(src);
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
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    /// Number of neurons.
    #[getter]
    fn n(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("VSNPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════

/// Firing-rate neuron population (exponential Euler toward clamped input).
///
/// Params:
/// | idx | Name     | Default | Description |
/// |:---:|----------|:-------:|-------------|
/// | 0   | dt       | 0.001   | Timestep (seconds) |
/// | 1   | inv_tau  | 1.0     | 1/τ — inverse rate time constant (s⁻¹) |
///
/// States:
/// | idx | Name | Description |
/// |:---:|------|-------------|
/// | 0   | r    | Firing rate (0..1) |
#[pyclass(name = "RatePopulation", module = "taas_checkmate.pillars")]
pub struct PyRatePopulation {
    inner: NeuralPopulation<pillars::zoo::RateModel>,
}

#[pymethods]
impl PyRatePopulation {
    /// Create rate population with `n` neurons.
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<pillars::zoo::RateModel>::new(n);
        pop.fill_param(0, 0.001); // dt (seconds)
        pop.fill_param(1, 1.0); // inv_tau
        pop.fill_state(0, 0.0); // r
        Self { inner: pop }
    }

    #[setter]
    fn set_dt(&mut self, val: f32) {
        self.inner.fill_param(0, val);
    }
    #[setter]
    fn set_inv_tau(&mut self, val: f32) {
        self.inner.fill_param(1, val);
    }
    /// Convenience: `tau` in seconds (stores `inv_tau = 1/tau`).
    #[setter]
    fn set_tau(&mut self, val: f32) {
        self.inner.fill_param(1, 1.0 / val.max(1e-6));
    }

    /// Firing rate as numpy array (copy-on-read).
    #[getter]
    fn r<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let slice = self.inner.state_slice(0);
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Run one simulation step (i_in = input rate, spikes = r > 0.5).
    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("RatePopulation(n={})", self.inner.len())
    }
}

/// Adaptive Exponential (AdEx) neuron population.
///
/// Params: `dt`, `inv_tau_m`, `inv_tau_w`, `v_rest`, `v_thresh`, `v_reset`,
/// `delta_t`, `inv_delta_t`, `a`, `b`, `tau_ref`, `v_peak`.
///
/// States: `v`, `w` (adaptation), `refr`.
#[pyclass(name = "AdExPopulation", module = "taas_checkmate.pillars")]
pub struct PyAdExPopulation {
    inner: NeuralPopulation<pillars::zoo::AdExModel>,
}

#[pymethods]
impl PyAdExPopulation {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<pillars::zoo::AdExModel>::new(n);
        let m = pillars::zoo::AdExModelParamIdx::dt;
        for col in 0..pillars::zoo::AdExModel::N_PARAMS {
            pop.fill_param(col, 0.0);
        }
        // Sensible defaults (all neurons uniform).
        pop.fill_param(m.into(), 0.5); // dt
        pop.fill_param(pillars::zoo::AdExModelParamIdx::inv_tau_m.into(), 1.0 / 20.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::inv_tau_w.into(), 1.0 / 100.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::v_rest.into(), -65.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::v_thresh.into(), -50.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::v_reset.into(), -70.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::delta_t.into(), 2.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::inv_delta_t.into(), 0.5);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::a.into(), 0.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::b.into(), 2.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::tau_ref.into(), 2.0);
        pop.fill_param(pillars::zoo::AdExModelParamIdx::v_peak.into(), 20.0);
        pop.fill_state(pillars::zoo::AdExModelStateIdx::v.into(), -65.0);
        pop.fill_state(pillars::zoo::AdExModelStateIdx::w.into(), 0.0);
        Self { inner: pop }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::dt.into(), v);
    }
    #[setter]
    fn set_v_rest(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::v_rest.into(), v);
    }
    #[setter]
    fn set_v_thresh(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::v_thresh.into(), v);
    }
    #[setter]
    fn set_v_reset(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::v_reset.into(), v);
    }
    #[setter]
    fn set_tau_m(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::inv_tau_m.into(), 1.0 / v.max(1e-6));
    }
    #[setter]
    fn set_b(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::AdExModelParamIdx::b.into(), v);
    }

    /// Membrane potential as numpy array.
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(0)))
    }

    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("AdExPopulation(n={})", self.inner.len())
    }
}

/// GLIF4 neuron population (v, adaptive threshold th, AHP conductance g).
///
/// Params: `dt`, `g_leak`, `e_leak`, `v_thresh_0`, `v_reset`, `inv_tau_ahp`,
/// `delta_g`, `e_ahp`, `inv_tau_theta`, `delta_theta`, `tau_ref`.
/// States: `v`, `th`, `g`, `refr`.
#[pyclass(name = "GLIF4Population", module = "taas_checkmate.pillars")]
pub struct PyGLIF4Population {
    inner: NeuralPopulation<pillars::zoo::GLIF4Model>,
}

#[pymethods]
impl PyGLIF4Population {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<pillars::zoo::GLIF4Model>::new(n);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::dt.into(), 0.5);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::g_leak.into(), 0.1);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::e_leak.into(), -65.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::v_thresh_0.into(), -50.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::v_reset.into(), -70.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::inv_tau_ahp.into(), 1.0 / 100.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::delta_g.into(), 2.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::e_ahp.into(), -70.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::inv_tau_theta.into(), 1.0 / 10.0);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::delta_theta.into(), 0.5);
        pop.fill_param(pillars::zoo::GLIF4ModelParamIdx::tau_ref.into(), 2.0);
        pop.fill_state(pillars::zoo::GLIF4ModelStateIdx::v.into(), -65.0);
        pop.fill_state(pillars::zoo::GLIF4ModelStateIdx::th.into(), -50.0);
        pop.fill_state(pillars::zoo::GLIF4ModelStateIdx::g.into(), 0.0);
        Self { inner: pop }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::GLIF4ModelParamIdx::dt.into(), v);
    }
    #[setter]
    fn set_v_thresh(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::GLIF4ModelParamIdx::v_thresh_0.into(), v);
    }
    #[setter]
    fn set_v_reset(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::GLIF4ModelParamIdx::v_reset.into(), v);
    }

    /// Membrane potential as numpy array.
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(0)))
    }

    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("GLIF4Population(n={})", self.inner.len())
    }
}

/// FitzHugh–Nagumo population (2D relaxation oscillator).
///
/// Params: `dt`, `inv_tau_w`, `a`, `b`, `v_thresh`. States: `v`, `w`, `was_below`.
#[pyclass(name = "FHNPopulation", module = "taas_checkmate.pillars")]
pub struct PyFHNPopulation {
    inner: NeuralPopulation<pillars::zoo::FHNModel>,
}

#[pymethods]
impl PyFHNPopulation {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<pillars::zoo::FHNModel>::new(n);
        // dt=0.01 ms — FHN relaxation is numerically unstable at larger dt.
        pop.fill_param(pillars::zoo::FHNModelParamIdx::dt.into(), 0.01);
        pop.fill_param(pillars::zoo::FHNModelParamIdx::inv_tau_w.into(), 1.0 / 10.0);
        pop.fill_param(pillars::zoo::FHNModelParamIdx::a.into(), 0.7);
        pop.fill_param(pillars::zoo::FHNModelParamIdx::b.into(), 0.8);
        pop.fill_param(pillars::zoo::FHNModelParamIdx::v_thresh.into(), 0.5);
        pop.fill_state(pillars::zoo::FHNModelStateIdx::v.into(), 0.0);
        pop.fill_state(pillars::zoo::FHNModelStateIdx::w.into(), 0.0);
        Self { inner: pop }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::FHNModelParamIdx::dt.into(), v);
    }

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(0)))
    }

    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("FHNPopulation(n={})", self.inner.len())
    }
}

/// Hindmarsh–Rose population (bursting 3D model).
///
/// Params: `dt`, `a`, `b`, `c`, `d`, `r`, `s`, `x_r`, `v_thresh`.
/// States: `x`, `y`, `z`.
#[pyclass(name = "HRPopulation", module = "taas_checkmate.pillars")]
pub struct PyHRPopulation {
    inner: NeuralPopulation<pillars::zoo::HRModel>,
}

#[pymethods]
impl PyHRPopulation {
    #[new]
    fn new(n: usize) -> Self {
        let mut pop = NeuralPopulation::<pillars::zoo::HRModel>::new(n);
        pop.fill_param(pillars::zoo::HRModelParamIdx::dt.into(), 0.5);
        pop.fill_param(pillars::zoo::HRModelParamIdx::a.into(), 1.0);
        pop.fill_param(pillars::zoo::HRModelParamIdx::b.into(), 3.0);
        pop.fill_param(pillars::zoo::HRModelParamIdx::c.into(), 1.0);
        pop.fill_param(pillars::zoo::HRModelParamIdx::d.into(), 5.0);
        pop.fill_param(pillars::zoo::HRModelParamIdx::r.into(), 0.001);
        pop.fill_param(pillars::zoo::HRModelParamIdx::s.into(), 4.0);
        pop.fill_param(pillars::zoo::HRModelParamIdx::x_r.into(), -1.6);
        pop.fill_param(pillars::zoo::HRModelParamIdx::v_thresh.into(), 0.0);
        pop.fill_state(pillars::zoo::HRModelStateIdx::x.into(), -1.0);
        pop.fill_state(pillars::zoo::HRModelStateIdx::y.into(), 0.0);
        pop.fill_state(pillars::zoo::HRModelStateIdx::z.into(), 0.0);
        Self { inner: pop }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .fill_param(pillars::zoo::HRModelParamIdx::dt.into(), v);
    }

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(0)))
    }

    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.inner.len();
        if i.len() != n || sp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        self.inner.step(i, sp);
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("HRPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Remaining zoo models — each is a one-line define_population!
//  (param/state column layout + init defaults). 7 models: GLIF3, HAAP,
//  Morris-Lecar, SAW, Wilson-Cowan, Triangulated (LUT), Atlas neuromotor.
// ═══════════════════════════════════════════════════════════════

define_population! {
    docs: "Generalized LIF with after-spike hyperpolarizing conductance (GLIF3).\n\nParams: dt, g_leak, e_leak, v_thresh, v_reset, inv_tau_ahp, delta_g, e_ahp, tau_ref.\nStates: v, g (ahp conductance), refr.",
    class: PyGLIF3Population,
    py_name: "GLIF3Population",
    model: GLIF3Model,
    params: [
        (dt, set_dt, 0, 1.0), (g_leak, set_g_leak, 1, 0.1), (e_leak, set_e_leak, 2, -65.0),
        (v_thresh, set_v_thresh, 3, -50.0), (v_reset, set_v_reset, 4, -65.0),
        (inv_tau_ahp, set_inv_tau_ahp, 5, 0.02), (delta_g, set_delta_g, 6, 0.01),
        (e_ahp, set_e_ahp, 7, -80.0), (tau_ref, set_tau_ref, 8, 2.0)
    ],
    states: [(v, set_v, 0, -65.0), (g, set_g, 1, 0.0), (refr, set_refr, 2, 0.0)],
    extra: {}
}

define_population! {
    docs: "Hawkes-adapted heater-like pacemaker (HAAP) — v, u, h states with spike-frequency adaptation and h-current.",
    class: PyHAAPPopulation,
    py_name: "HAAPPopulation",
    model: HAAPModel,
    params: [
        (dt, set_dt, 0, 0.05), (inv_cm, set_inv_cm, 1, 1.0), (g_l, set_g_l, 2, 0.1), (e_l, set_e_l, 3, -65.0),
        (g_m, set_g_m, 4, 0.5), (e_k, set_e_k, 5, -80.0), (g_h, set_g_h, 6, 0.3), (e_h, set_e_h, 7, -30.0),
        (inv_tau_u, set_inv_tau_u, 8, 0.01), (inv_tau_h, set_inv_tau_h, 9, 0.005),
        (v_half_u, set_v_half_u, 10, -35.0), (k_u, set_k_u, 11, 5.0), (v_half_h, set_v_half_h, 12, -75.0),
        (k_h, set_k_h, 13, 6.0), (v_thresh, set_v_thresh, 14, -40.0), (v_reset, set_v_reset, 15, -55.0),
        (delta_u, set_delta_u, 16, 0.02)
    ],
    states: [(v, set_v, 0, -65.0), (u, set_u, 1, 0.0), (h, set_h, 2, 1.0)],
    extra: {}
}

define_population! {
    docs: "Morris-Lecar two-dimensional conductance model (v, w) with edge-triggered spiking.",
    class: PyMorrisLecarPopulation,
    py_name: "MorrisLecarPopulation",
    model: MorrisLecarModel,
    params: [
        (dt, set_dt, 0, 1.0), (inv_C, set_inv_C, 1, 0.05), (g_L, set_g_L, 2, 2.0), (E_L, set_E_L, 3, -60.0),
        (g_Ca, set_g_Ca, 4, 4.4), (E_Ca, set_E_Ca, 5, 120.0), (g_K, set_g_K, 6, 8.0), (E_K, set_E_K, 7, -84.0),
        (V1, set_V1, 8, -1.2), (V2, set_V2, 9, 18.0), (V3, set_V3, 10, 2.0), (V4, set_V4, 11, 30.0),
        (phi, set_phi, 12, 0.04), (v_thresh, set_v_thresh, 13, 0.0)
    ],
    states: [(v, set_v, 0, -60.0), (w, set_w, 1, 0.0), (was_below, set_was_below, 2, 1.0)],
    extra: {}
}

define_population! {
    docs: "Subthreshold-adaptive with spike trace (SAW) — Izhikevich-like with a synaptic trace.",
    class: PySAWPopulation,
    py_name: "SAWPopulation",
    model: SAWModel,
    params: [
        (dt, set_dt, 0, 1.0), (a, set_a, 1, 0.02), (b, set_b, 2, 0.2), (trace_decay, set_trace_decay, 3, 0.01),
        (noise_amp, set_noise_amp, 4, 0.0), (tau_ref, set_tau_ref, 5, 2.0), (v_thresh, set_v_thresh, 6, 30.0),
        (v_reset, set_v_reset, 7, -65.0)
    ],
    states: [(v, set_v, 0, -65.0), (u, set_u, 1, 0.0), (tr, set_tr, 2, 0.0), (refr, set_refr, 3, 0.0)],
    extra: {}
}

define_population! {
    docs: "Wilson-Cowan excitatory/inhibitory rate pair (e, i) — no spiking output.",
    class: PyWCPopulation,
    py_name: "WCPopulation",
    model: WCModel,
    params: [
        (dt, set_dt, 0, 0.1), (inv_tau_e, set_inv_tau_e, 1, 0.1), (inv_tau_i, set_inv_tau_i, 2, 0.1),
        (w_ee, set_w_ee, 3, 1.2), (w_ei, set_w_ei, 4, 0.4), (w_ie, set_w_ie, 5, 0.6),
        (w_ii, set_w_ii, 6, 0.8), (beta, set_beta, 7, 4.0)
    ],
    states: [(e, set_e, 0, 0.0), (i, set_i, 1, 0.0)],
    extra: {}
}

define_population! {
    docs: "Triangulated LUT-based neuron — a lookup table replaces ODE integration. Call TriangulatedPopulation.calibrate_lif() once before step().",
    class: PyTriangulatedPopulation,
    py_name: "TriangulatedPopulation",
    model: TriangulatedModel,
    params: [(dt, set_dt, 0, 0.5), (v_thresh, set_v_thresh, 1, -54.0), (input_thresh, set_input_thresh, 2, 0.5)],
    states: [(v, set_v, 0, -70.0), (history, set_history, 1, 0.0)],
    extra: {
        /// Calibrate a LIF-referenced LUT and install it as the process-wide
        /// table. Required before `step()` — without it stepping panics.
        /// Defaults match the Rust calibration tests (always-on 30.0 input,
        /// 500k steps) so a plain call produces a spiking table.
        ///
        /// Args:
        ///     v_min, v_max: membrane range for quantization (mV, default -100..60)
        ///     n_steps: calibration simulation steps (default 500_000)
        ///     seed: RNG seed for reproducibility (default 42)
        #[staticmethod]
        #[pyo3(signature = (v_min=-100.0, v_max=60.0, n_steps=500_000, seed=42))]
        fn calibrate_lif(v_min: f32, v_max: f32, n_steps: usize, seed: u64) -> PyResult<()> {
            use pillars::ops::triangulated::{LutBuilder, LutConfig};
            let cfg = LutConfig {
                n_steps,
                input_modes: vec![(1.0, 30.0, 1)],
                v_min,
                v_max,
                dt: 0.5,
                seed,
            };
            // LIF reference params (SoA, n=1): dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref
            let params = [0.5f32, 0.1, -70.0, -54.0, -70.0, 2.0];
            let lut = LutBuilder::new(cfg).build::<LIFModel>(&params, &[-70.0, 0.0]);
            TriangulatedModel::set_lut(lut);
            Ok(())
        }
    }
}

define_population! {
    docs: "Atlasov neuromotor model — muscle calcium, tension, length and velocity dynamics driven by spikes.",
    class: PyAtlasovNeuromotorPopulation,
    py_name: "AtlasovNeuromotorPopulation",
    model: AtlasovNeuromotorModel,
    params: [
        (dt, set_dt, 0, 1.0), (d_ca, set_d_ca, 1, 0.1), (ca_decay_const, set_ca_decay_const, 2, 0.9),
        (eta_reset, set_eta_reset, 3, 0.5), (t_max, set_t_max, 4, 1.0), (kd, set_kd, 5, 0.2),
        (l0, set_l0, 6, 1.0), (inv_vmax, set_inv_vmax, 7, 1.0), (inv_width, set_inv_width, 8, 1.0)
    ],
    states: [
        (ca, set_ca, 0, 0.0), (tension, set_tension, 1, 0.0), (reset_flag, set_reset_flag, 2, 0.0),
        (length, set_length, 3, 1.0), (velocity, set_velocity, 4, 0.0)
    ],
    extra: {}
}

/// Connectome brain on the pillars engine — LIF population wired to a CSR
/// graph, spike-routed with axonal delays.
///
/// Build from CSR arrays (the connectome format):
/// ```python
/// from taas_checkmate.pillars import ConnectomeBrain
/// brain = ConnectomeBrain(row_ptr, col_idx, weights, delays)
/// brain.dt = 0.5
/// brain.inject([0], [30.0])
/// brain.step()
/// spikes = brain.spikes  # numpy (n,)
/// ```
#[pyclass(name = "ConnectomeBrain", module = "taas_checkmate.pillars")]
pub struct PyConnectomeBrain {
    inner: pillars_connectome::Brain<pillars::zoo::LIFModel>,
    n: usize,
}

#[pymethods]
impl PyConnectomeBrain {
    /// Build from CSR arrays: row_ptr (u32), col_idx (u32), weights (f32),
    /// delays (u8, steps). `n_neurons = len(row_ptr) - 1`.
    #[new]
    fn new(row_ptr: Vec<u32>, col_idx: Vec<u32>, weights: Vec<f32>, delays: Vec<u8>) -> Self {
        let n_syn = weights.len();
        let conn = pillars_connectome::ConnectomeData::from_csr(
            row_ptr,
            col_idx,
            weights,
            delays,
            vec![0u8; n_syn], // receptor types: instant current path
        );
        let n = conn.n_neurons();
        let mut brain =
            pillars_connectome::Brain::<pillars::zoo::LIFModel>::from_connectome(&conn);
        brain.configure(|p| {
            p.fill_param(pillars::zoo::LIFModelParamIdx::dt.into(), 1.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::inv_tau_m.into(), 0.1);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_rest.into(), -70.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_thresh.into(), -54.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_reset.into(), -70.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::tau_ref.into(), 2.0);
            p.fill_state(pillars::zoo::LIFModelStateIdx::v.into(), -70.0);
        });
        Self { inner: brain, n }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::dt.into(), v));
    }
    #[setter]
    fn set_inv_tau_m(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::inv_tau_m.into(), v));
    }
    #[setter]
    fn set_v_rest(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::v_rest.into(), v));
    }
    #[setter]
    fn set_v_thresh(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::v_thresh.into(), v));
    }
    #[setter]
    fn set_v_reset(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::v_reset.into(), v));
    }
    #[setter]
    fn set_tau_ref(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::tau_ref.into(), v));
    }

    /// Inject external current into specific neurons (arrives next step).
    fn inject(&mut self, indices: Vec<u32>, values: Vec<f32>) -> PyResult<()> {
        self.inner.inject(&indices, &values);
        Ok(())
    }

    /// Advance one simulation step (all neurons + spike routing).
    fn step(&mut self) {
        self.inner.step();
    }

    /// Current spike mask (0/1 per neuron) after the last step.
    #[getter]
    fn spikes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice(py, self.inner.spikes()))
    }

    /// Membrane potential as numpy array (copy-on-read).
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state(0)))
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    fn __repr__(&self) -> String {
        format!("ConnectomeBrain(n={})", self.n)
    }
}

/// Conductance-based connectome brain — LIF population with receptor kinetics
/// (AMPA/NMDA/GABA) via pillars-connectome `SynapticBrain`.
///
/// Receptors per synapse: `0=AMPA, 1=NMDA, 2=GABA_A, 3=GABA_B, 4=nAChR,
/// 5=GluCl, 6=mAChR` (parallel to `col_idx`).
///
/// ```python
/// from taas_checkmate.pillars import SynapticConnectomeBrain
/// brain = SynapticConnectomeBrain(
///     row_ptr, col_idx, weights, delays,
///     receptor_types, dt=0.5)
/// brain.step(i_in, spikes)   # i_in = external current, spikes written in-place
/// ```
#[pyclass(name = "SynapticConnectomeBrain", module = "taas_checkmate.pillars")]
pub struct PySynapticConnectomeBrain {
    inner: pillars_connectome::SynapticBrain<pillars::zoo::LIFModel>,
    n: usize,
}

#[pymethods]
impl PySynapticConnectomeBrain {
    /// Build from CSR arrays + per-synapse receptor types + dt (ms).
    #[new]
    fn new(
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        weights: Vec<f32>,
        delays: Vec<u8>,
        receptor_types: Vec<u8>,
        dt: f32,
    ) -> Self {
        let conn = pillars_connectome::ConnectomeData::from_csr(
            row_ptr, col_idx, weights, delays, receptor_types,
        );
        let n = conn.n_neurons();
        let mut brain =
            pillars_connectome::SynapticBrain::<pillars::zoo::LIFModel>::from_connectome(&conn, dt);
        brain.configure(|p| {
            p.fill_param(pillars::zoo::LIFModelParamIdx::dt.into(), dt);
            p.fill_param(pillars::zoo::LIFModelParamIdx::inv_tau_m.into(), 0.1);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_rest.into(), -70.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_thresh.into(), -54.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::v_reset.into(), -70.0);
            p.fill_param(pillars::zoo::LIFModelParamIdx::tau_ref.into(), 2.0);
            p.fill_state(pillars::zoo::LIFModelStateIdx::v.into(), -70.0);
        });
        Self { inner: brain, n }
    }

    #[setter]
    fn set_dt(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::dt.into(), v));
    }
    #[setter]
    fn set_inv_tau_m(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::inv_tau_m.into(), v));
    }
    #[setter]
    fn set_v_thresh(&mut self, v: f32) {
        self.inner
            .configure(|p| p.fill_param(pillars::zoo::LIFModelParamIdx::v_thresh.into(), v));
    }

    /// Run one step: decay conductances, compute I_syn = Σ g·(E_rev−V), step
    /// neurons, jump post-synaptic conductances on spikes. `i_in` = external
    /// current, `spikes_out` written in-place.
    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let out = sp_rw.as_slice_mut()?;
        if i.len() != self.n || out.len() != self.n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step: expected i_in and spikes_out of length {} (n), got {} and {}",
                self.n,
                i.len(),
                out.len()
            )));
        }
        let spikes = self.inner.step(i);
        out.copy_from_slice(&spikes);
        Ok(())
    }

    /// Membrane potential as numpy array (copy-on-read).
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state(0)))
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    fn __repr__(&self) -> String {
        format!("SynapticConnectomeBrain(n={})", self.n)
    }
}

/// Register the pillars sub-module in the Python module tree.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "pillars")?;
    sub.add_class::<PyLIFPopulation>()?;
    sub.add_class::<PyIzhikevichPopulation>()?;
    sub.add_class::<PyHHPopulation>()?;
    sub.add_class::<PyVSNPopulation>()?;
    sub.add_class::<PyRatePopulation>()?;
    sub.add_class::<PyAdExPopulation>()?;
    sub.add_class::<PyGLIF4Population>()?;
    sub.add_class::<PyFHNPopulation>()?;
    sub.add_class::<PyHRPopulation>()?;
    sub.add_class::<PyGLIF3Population>()?;
    sub.add_class::<PyHAAPPopulation>()?;
    sub.add_class::<PyMorrisLecarPopulation>()?;
    sub.add_class::<PySAWPopulation>()?;
    sub.add_class::<PyWCPopulation>()?;
    sub.add_class::<PyTriangulatedPopulation>()?;
    sub.add_class::<PyAtlasovNeuromotorPopulation>()?;
    sub.add_class::<PyConnectomeBrain>()?;
    sub.add_class::<PySynapticConnectomeBrain>()?;
    m.add_submodule(&sub)?;
    // Register in sys.modules under the public name so
    // `from taas_checkmate.pillars import X` works (not just attribute access).
    // Strip the <pkg>.<ext> suffix from the extension's own __name__ first.
    let parent_name = m.name()?;
    let parent_name = parent_name.to_str()?;
    let public_root = match parent_name.rsplit_once('.') {
        Some((root, last)) if last == "taas_checkmate" => root,
        _ => parent_name,
    };
    let fq = format!("{public_root}.pillars");
    sub.setattr("__name__", fq.clone())?;
    m.py().import("sys")?.getattr("modules")?.set_item(&fq, sub.clone())?;
    // Also register at root for convenience
    m.add_class::<PyLIFPopulation>()?;
    m.add_class::<PyIzhikevichPopulation>()?;
    m.add_class::<PyHHPopulation>()?;
    m.add_class::<PyVSNPopulation>()?;
    m.add_class::<PyRatePopulation>()?;
    m.add_class::<PyAdExPopulation>()?;
    m.add_class::<PyGLIF4Population>()?;
    m.add_class::<PyFHNPopulation>()?;
    m.add_class::<PyHRPopulation>()?;
    m.add_class::<PyGLIF3Population>()?;
    m.add_class::<PyHAAPPopulation>()?;
    m.add_class::<PyMorrisLecarPopulation>()?;
    m.add_class::<PySAWPopulation>()?;
    m.add_class::<PyWCPopulation>()?;
    m.add_class::<PyTriangulatedPopulation>()?;
    m.add_class::<PyAtlasovNeuromotorPopulation>()?;
    m.add_class::<PyConnectomeBrain>()?;
    m.add_class::<PySynapticConnectomeBrain>()?;
    Ok(())
}
