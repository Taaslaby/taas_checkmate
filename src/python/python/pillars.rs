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

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pillars::prelude::*;
use pillars::zoo::{IzhikevichDualModel, LIFAtModel, TransformerModel, TriangulatedModel};
use pyo3::prelude::*;

/// Copy one NumPy column into the canonical population storage.
///
/// The Rust population API intentionally panics on an invalid column because
/// its callers are compile-time model code. Python is a different boundary:
/// user-provided indices must become `ValueError`, never a Rust panic.
fn set_param_column<M: Model>(
    pop: &mut NeuralPopulation<M>,
    col: usize,
    array: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    if col >= M::N_PARAMS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "set_param_array: column {col} out of range (model has {} params)",
            M::N_PARAMS
        )));
    }
    let arr = array.readonly();
    let src = arr.as_slice()?; // zero-copy view — no intermediate Vec
    let dest = pop.param_slice_mut(col);
    if src.len() != dest.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "set_param_array: expected {} elements (n), got {}",
            dest.len(),
            src.len()
        )));
    }
    dest.copy_from_slice(&src);
    Ok(())
}

/// Copy one NumPy state column into the canonical population storage.
fn set_state_column<M: Model>(
    pop: &mut NeuralPopulation<M>,
    col: usize,
    array: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    if col >= M::N_STATE {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "set_state_array: column {col} out of range (model has {} states)",
            M::N_STATE
        )));
    }
    let arr = array.readonly();
    let src = arr.as_slice()?; // zero-copy view — no intermediate Vec
    let dest = pop.state_slice_mut(col);
    if src.len() != dest.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "set_state_array: expected {} elements (n), got {}",
            dest.len(),
            src.len()
        )));
    }
    dest.copy_from_slice(&src);
    Ok(())
}

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
        // Column indices are expressions (literals for zoo models with a
        // documented layout, `<Enum as usize>` for enum-indexed models).
        params: [ $( ( $p:ident , $pset:ident , $p_idx:expr , $p_default:expr ) ),* $(,)? ],
        states: [ $( ( $s:ident , $sset:ident , $s_idx:expr , $s_default:expr ) ),* $(,)? ],
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
                set_param_column(&mut self.inner, col, array)
            }

            /// Set a state column from a numpy array (per-neuron values).
            fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
                set_state_column(&mut self.inner, col, array)
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

define_population! {
    docs: "Leaky Integrate-and-Fire (LIF) population — SIMD-accelerated.\n\nParams: dt=1.0, inv_tau_m=0.1, v_rest=-70.0, v_thresh=-54.0, v_reset=-70.0, tau_ref=2.0.\nStates: v (membrane potential, mV), refr (refractory counter, ms).",
    class: PyLIFPopulation,
    py_name: "LIFPopulation",
    model: LIFModel,
    params: [
        (dt, set_dt, 0, 1.0), (inv_tau_m, set_inv_tau_m, 1, 0.1), (v_rest, set_v_rest, 2, -70.0),
        (v_thresh, set_v_thresh, 3, -54.0), (v_reset, set_v_reset, 4, -70.0), (tau_ref, set_tau_ref, 5, 2.0)
    ],
    states: [(v, set_v, 0, -70.0), (refr, set_refr, 1, 0.0)],
    extra: {
        /// `v_rest` param column as numpy array (legacy alias kept for
        /// backwards compatibility — predates the generated `v_rest` getter).
        #[getter]
        fn v_rest_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
            PyArray1::from_slice(py, self.inner.param_slice(2))
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  IzhikevichPopulation
// ═══════════════════════════════════════════════════════════════

define_population! {
    docs: "Izhikevich neuron population — regular/fast/bursting spiking.\n\nParams: dt=0.5, a=0.02, b=0.2, c=-65.0, d=8.0, tau_ref=2.0, v_thresh=30.0.\nStates: v, u (recovery), refr.",
    class: PyIzhikevichPopulation,
    py_name: "IzhikevichPopulation",
    model: IzhikevichModel,
    params: [
        (dt, set_dt, 0, 0.5), (a, set_a, 1, 0.02), (b, set_b, 2, 0.2), (c, set_c, 3, -65.0),
        (d, set_d, 4, 8.0), (tau_ref, set_tau_ref, 5, 2.0), (v_thresh, set_v_thresh, 6, 30.0)
    ],
    states: [(v, set_v, 0, -65.0), (u, set_u, 1, 0.2 * -65.0), (refr, set_refr, 2, 0.0)],
    extra: {}
}

// ═══════════════════════════════════════════════════════════════
//  HHPopulation
// ═══════════════════════════════════════════════════════════════

define_population! {
    docs: "Hodgkin-Huxley population — full Na+/K+ ion channel dynamics (with optional CaV-T).\n\nParams: dt=0.01, g_na=120.0, e_na=50.0, g_k=36.0, e_k=-77.0, g_leak=0.3, e_leak=-54.387,\nq10=1.0, v_thresh=0.0, inv_cm=1.0, g_caT=0.0 (standard HH: no Ca2+ current), e_caT=120.0\n(canonical, inert while g_caT=0).\nStates: v, m, h, n_gate (K+ gate; legacy name), c (CaV-T gate), was_below.",
    class: PyHHPopulation,
    py_name: "HHPopulation",
    model: HHModel,
    params: [
        (dt, set_dt, 0, 0.01), (g_na, set_g_na, 1, 120.0), (e_na, set_e_na, 2, 50.0),
        (g_k, set_g_k, 3, 36.0), (e_k, set_e_k, 4, -77.0), (g_leak, set_g_leak, 5, 0.3),
        (e_leak, set_e_leak, 6, -54.387), (q10, set_q10, 7, 1.0), (v_thresh, set_v_thresh, 8, 0.0),
        (inv_cm, set_inv_cm, 9, 1.0), (g_caT, set_g_caT, 10, 0.0), (e_caT, set_e_caT, 11, 120.0)
    ],
    states: [
        (v, set_v, 0, -65.0), (m, set_m, 1, 0.05), (h, set_h, 2, 0.6), (n_gate, set_n_gate, 3, 0.32),
        (c, set_c, 4, 0.19), (was_below, set_was_below, 5, 1.0)
    ],
    extra: {}
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
        pop.fill_param(1, 1.0 / tau_m); // Pre-inverted: store 1/τ_m
        pop.fill_param(2, v_rest);
        pop.fill_param(3, v_thresh);
        pop.fill_param(4, v_reset);
        pop.fill_param(5, 2.0); // tau_ref
        pop.fill_param(6, beta);
        pop.fill_param(7, surrogate_gain);
        pop.fill_state(0, v_rest); // v = v_rest
        pop.fill_state(1, 0.0); // refr
        pop.fill_state(2, 0.0); // surrogate
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter]
    fn set_dt(&mut self, val: f32) {
        self.inner.fill_param(0, val);
    }
    #[setter]
    fn set_inv_tau_m(&mut self, val: f32) {
        self.inner.fill_param(1, val);
    }
    /// Set tau_m in ms — automatically stores as 1/tau_m (pre-inverted).
    #[setter]
    fn set_tau_m(&mut self, val: f32) {
        self.inner.fill_param(1, 1.0 / val);
    }
    #[setter]
    fn set_v_rest(&mut self, val: f32) {
        self.inner.fill_param(2, val);
    }
    #[setter]
    fn set_v_thresh(&mut self, val: f32) {
        self.inner.fill_param(3, val);
    }
    #[setter]
    fn set_v_reset(&mut self, val: f32) {
        self.inner.fill_param(4, val);
    }
    #[setter]
    fn set_tau_ref(&mut self, val: f32) {
        self.inner.fill_param(5, val);
    }
    #[setter]
    fn set_beta(&mut self, val: f32) {
        self.inner.fill_param(6, val);
    }
    #[setter]
    fn set_surrogate_gain(&mut self, val: f32) {
        self.inner.fill_param(7, val);
    }

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

    #[setter]
    fn set_v(&mut self, val: f32) {
        self.inner.fill_state(0, val);
    }
    #[setter]
    fn set_refr(&mut self, val: f32) {
        self.inner.fill_state(1, val);
    }
    #[setter]
    fn set_surrogate(&mut self, val: f32) {
        self.inner.fill_state(2, val);
    }

    // ─── Per-neuron arrays ──────────────────────────────────

    /// Set a param column from a numpy array (per-neuron values).
    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        set_param_column(&mut self.inner, col, array)
    }

    /// Set a state column from a numpy array (per-neuron values).
    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        set_state_column(&mut self.inner, col, array)
    }

    // ─── Step ───────────────────────────────────────────────

    /// Run one simulation step.
    ///
    /// Args:
    ///     i_in: input current (float32 numpy array, length n)
    ///     spikes_out: output spikes (int32 numpy array, length n, modified in-place)
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

    /// Number of neurons.
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("VSNPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  TransformerPopulation — нейрон-трансформер с STDP
// ═══════════════════════════════════════════════════════════════

/// Transformer-like neuron with learnable QKV weights (STDP-style Hebbian +
/// homeostatic pull). 8 memory slots, leaky integrator, learnable keys
/// (w₀..w₇) and values (v₀..v₇).
///
/// Params (12):
/// | idx | Name        | Default | Description |
/// |:---:|-------------|:-------:|-------------|
/// | 0   | dt          | 0.5     | Timestep (ms) |
/// | 1   | decay       | 0.95    | Memory window decay (0..1) |
/// | 2   | gain        | 1.0     | Input gain for mem[0] |
/// | 3   | inv_tau_acc | 0.5     | 1/τ — inverse accumulator time constant |
/// | 4   | thresh      | 0.9     | Spike threshold |
/// | 5   | k_scale     | 0.5     | Key projection scale |
/// | 6   | eta_k       | 0.0     | STDP learning rate — keys |
/// | 7   | eta_v       | 0.0     | STDP learning rate — values |
/// | 8   | w_base      | 1.0     | Homeostatic baseline — w |
/// | 9   | w_span      | 2.0     | Clamp span — w ∈ [base±span] |
/// | 10  | v_base      | 1.0     | Homeostatic baseline — v |
/// | 11  | v_span      | 2.0     | Clamp span — v ∈ [base±span] |
///
/// States (25):
/// | idx | Name | Description |
/// |:---:|------|-------------|
/// | 0–7 | mem  | Rolling input memory (mem0 = latest, mem7 = oldest) |
/// | 8   | acc  | Leaky integrator |
/// | 9–16| w    | Learnable key weights |
/// | 17–24| v   | Learnable value weights |
#[pyclass(name = "TransformerPopulation", module = "taas_checkmate.pillars")]
pub struct PyTransformerPopulation {
    inner: NeuralPopulation<TransformerModel>,
}

#[pymethods]
impl PyTransformerPopulation {
    /// Create Transformer population with `n` neurons.
    ///
    /// Args:
    ///     n: number of neurons
    ///     dt: timestep in ms (default 0.5)
    ///     k_scale: key projection scale (default 0.5)
    ///     thresh: spike threshold (default 0.9)
    ///     eta_k: STDP learning rate for keys, 0 = frozen (default 0.0)
    ///     eta_v: STDP learning rate for values, 0 = frozen (default 0.0)
    #[new]
    #[pyo3(signature = (n, dt=0.5, k_scale=0.5, thresh=0.9, eta_k=0.0, eta_v=0.0))]
    fn new(
        n: usize,
        dt: f32,
        k_scale: f32,
        thresh: f32,
        eta_k: f32,
        eta_v: f32,
    ) -> Self {
        let mut pop = NeuralPopulation::<TransformerModel>::new(n);
        pop.fill_param(0, dt);              // DT
        pop.fill_param(1, 0.95);            // Decay
        pop.fill_param(2, 1.0);             // Gain
        pop.fill_param(3, 0.5);             // Inv_Tau_Acc
        pop.fill_param(4, thresh);          // Thresh
        pop.fill_param(5, k_scale);         // K_Scale
        pop.fill_param(6, eta_k);           // Eta_K
        pop.fill_param(7, eta_v);           // Eta_V
        pop.fill_param(8, 1.0);             // W_Base
        pop.fill_param(9, 2.0);             // W_Span
        pop.fill_param(10, 1.0);            // V_Base
        pop.fill_param(11, 2.0);            // V_Span
        // Init w/v to 1.0 (states 9..17 = w0..w7, 17..25 = v0..v7)
        for i in 0..8 {
            pop.fill_state(9 + i, 1.0);
            pop.fill_state(17 + i, 1.0);
        }
        Self { inner: pop }
    }

    // ─── Params ─────────────────────────────────────────────

    #[setter] fn set_dt(&mut self, v: f32) { self.inner.fill_param(0, v); }
    #[setter] fn set_decay(&mut self, v: f32) { self.inner.fill_param(1, v); }
    #[setter] fn set_gain(&mut self, v: f32) { self.inner.fill_param(2, v); }
    #[setter] fn set_inv_tau_acc(&mut self, v: f32) { self.inner.fill_param(3, v); }
    #[setter] fn set_thresh(&mut self, v: f32) { self.inner.fill_param(4, v); }
    #[setter] fn set_k_scale(&mut self, v: f32) { self.inner.fill_param(5, v); }
    #[setter] fn set_eta_k(&mut self, v: f32) { self.inner.fill_param(6, v); }
    #[setter] fn set_eta_v(&mut self, v: f32) { self.inner.fill_param(7, v); }
    #[setter] fn set_w_base(&mut self, v: f32) { self.inner.fill_param(8, v); }
    #[setter] fn set_w_span(&mut self, v: f32) { self.inner.fill_param(9, v); }
    #[setter] fn set_v_base(&mut self, v: f32) { self.inner.fill_param(10, v); }
    #[setter] fn set_v_span(&mut self, v: f32) { self.inner.fill_param(11, v); }

    // ─── State getters ──────────────────────────────────────

    /// Accumulator (leaky integrator) — numpy array.
    #[getter]
    fn acc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(8)))
    }

    /// Raw attention output BEFORE LIF integration (state 25).
    /// `out = Σ a_i · (v_i · mem_i)` — matches PyTorch transformer_neurons_v2
    /// for bit-identical LM forward pass.
    fn read_raw_out<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, self.inner.state_slice(25)))
    }

    /// Key weights w[0..7] stacked as (n, 8) — numpy array for inspection.
    fn read_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let n = self.inner.len();
        let mut rows: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; 8]).collect();
        for i in 0..8 {
            let src = self.inner.state_slice(9 + i);
            for (row, &v) in rows.iter_mut().zip(src.iter()) {
                row[i] = v;
            }
        }
        Ok(numpy::PyArray2::from_vec2(py, &rows)?)
    }

    /// Value weights v[0..7] stacked as (n, 8) — numpy array for inspection.
    fn read_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let n = self.inner.len();
        let mut rows: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; 8]).collect();
        for i in 0..8 {
            let src = self.inner.state_slice(17 + i);
            for (row, &v) in rows.iter_mut().zip(src.iter()) {
                row[i] = v;
            }
        }
        Ok(numpy::PyArray2::from_vec2(py, &rows)?)
    }

    // ─── Batch step (sequence) ──────────────────────────────

    /// Process a sequence of query vectors through the population,
    /// returning raw attention outputs for each timestep.
    ///
    /// Memory (mem0..mem7, acc) accumulates across timesteps.
    ///
    /// Args:
    ///     queries: (seq_len, n) float32 array
    ///
    /// Returns:
    ///     features: (n, seq_len) float32 array — raw_out per timestep
    fn batch_step<'py>(
        &mut self,
        py: Python<'py>,
        queries: Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let q_ro = queries.readonly();
        let q_slice = q_ro.as_slice()?;
        let dims = q_ro.dims();
        let seq_len = dims[0];
        let n_dim = dims[1];
        let n = self.inner.len();

        if n_dim != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch_step: queries.shape[1]={n_dim}, expected n={n}"
            )));
        }

        let mut spikes = vec![0i32; n];
        let mut rows: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; seq_len]).collect();

        for t in 0..seq_len {
            let i_in = &q_slice[t * n..(t + 1) * n];
            self.inner.step(i_in, &mut spikes);
            let out = self.inner.state_slice(25); // Out state
            for j in 0..n {
                rows[j][t] = out[j];
            }
        }

        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    // ─── Per-neuron arrays ──────────────────────────────────

    /// Set a param column from a numpy array (per-neuron values).
    fn set_param_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        set_param_column(&mut self.inner, col, array)
    }

    /// Set a state column from a numpy array (per-neuron values).
    fn set_state_array(&mut self, col: usize, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        set_state_column(&mut self.inner, col, array)
    }

    // ─── Step ───────────────────────────────────────────────

    /// Run one simulation step.
    ///
    /// On each step the neuron:
    /// 1. Computes softmax attention over 8 memory slots
    /// 2. Integrates the output → spike if above threshold
    /// 3. Updates w/v via Hebbian STDP (if eta>0)
    /// 4. Shifts the memory window
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
        format!("TransformerPopulation(n={})", self.inner.len())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Module registration
// ═══════════════════════════════════════════════════════════════

define_population! {
    docs: "Firing-rate neuron population (exponential Euler toward clamped input).\n\nParams: dt=0.001 (seconds), inv_tau=1.0.\nStates: r (firing rate, 0..1). step(): i_in = input rate, spikes = r > 0.5.",
    class: PyRatePopulation,
    py_name: "RatePopulation",
    model: pillars::zoo::RateModel,
    params: [(dt, set_dt, 0, 0.001), (inv_tau, set_inv_tau, 1, 1.0)],
    states: [(r, set_r, 0, 0.0)],
    extra: {
        /// Convenience: `tau` in seconds (stores `inv_tau = 1/tau`).
        #[setter]
        fn set_tau(&mut self, val: f32) {
            self.inner.fill_param(1, 1.0 / val.max(1e-6));
        }
    }
}

define_population! {
    docs: "Adaptive Exponential (AdEx) neuron population.\n\nParams: dt=0.5, inv_tau_m=1/20, inv_tau_w=1/100, v_rest=-65, v_thresh=-50, v_reset=-70,\ndelta_t=2.0, inv_delta_t=0.5, a=0.0, b=2.0, tau_ref=2.0, v_peak=20.0.\nStates: v, w (adaptation).",
    class: PyAdExPopulation,
    py_name: "AdExPopulation",
    model: pillars::zoo::AdExModel,
    params: [
        (dt, set_dt, pillars::zoo::AdExModelParamIdx::dt as usize, 0.5),
        (inv_tau_m, set_inv_tau_m, pillars::zoo::AdExModelParamIdx::inv_tau_m as usize, 1.0 / 20.0),
        (inv_tau_w, set_inv_tau_w, pillars::zoo::AdExModelParamIdx::inv_tau_w as usize, 1.0 / 100.0),
        (v_rest, set_v_rest, pillars::zoo::AdExModelParamIdx::v_rest as usize, -65.0),
        (v_thresh, set_v_thresh, pillars::zoo::AdExModelParamIdx::v_thresh as usize, -50.0),
        (v_reset, set_v_reset, pillars::zoo::AdExModelParamIdx::v_reset as usize, -70.0),
        (delta_t, set_delta_t, pillars::zoo::AdExModelParamIdx::delta_t as usize, 2.0),
        (inv_delta_t, set_inv_delta_t, pillars::zoo::AdExModelParamIdx::inv_delta_t as usize, 0.5),
        (a, set_a, pillars::zoo::AdExModelParamIdx::a as usize, 0.0),
        (b, set_b, pillars::zoo::AdExModelParamIdx::b as usize, 2.0),
        (tau_ref, set_tau_ref, pillars::zoo::AdExModelParamIdx::tau_ref as usize, 2.0),
        (v_peak, set_v_peak, pillars::zoo::AdExModelParamIdx::v_peak as usize, 20.0)
    ],
    states: [
        (v, set_v, pillars::zoo::AdExModelStateIdx::v as usize, -65.0),
        (w, set_w, pillars::zoo::AdExModelStateIdx::w as usize, 0.0)
    ],
    extra: {
        /// Set tau_m in ms — automatically stores inv_tau_m (pre-inverted).
        #[setter]
        fn set_tau_m(&mut self, v: f32) {
            self.inner.fill_param(
                pillars::zoo::AdExModelParamIdx::inv_tau_m as usize,
                1.0 / v.max(1e-6),
            );
        }
    }
}

define_population! {
    docs: "GLIF4 neuron population (v, adaptive threshold th, AHP conductance g).\n\nParams: dt=0.5, g_leak=0.1, e_leak=-65, v_thresh (v_thresh_0)=-50, v_reset=-70,\ninv_tau_ahp=1/100, delta_g=2.0, e_ahp=-70, inv_tau_theta=1/10, delta_theta=0.5, tau_ref=2.0.\nStates: v, th, g, refr.",
    class: PyGLIF4Population,
    py_name: "GLIF4Population",
    model: pillars::zoo::GLIF4Model,
    params: [
        (dt, set_dt, pillars::zoo::GLIF4ModelParamIdx::dt as usize, 0.5),
        (g_leak, set_g_leak, pillars::zoo::GLIF4ModelParamIdx::g_leak as usize, 0.1),
        (e_leak, set_e_leak, pillars::zoo::GLIF4ModelParamIdx::e_leak as usize, -65.0),
        (v_thresh, set_v_thresh, pillars::zoo::GLIF4ModelParamIdx::v_thresh_0 as usize, -50.0),
        (v_reset, set_v_reset, pillars::zoo::GLIF4ModelParamIdx::v_reset as usize, -70.0),
        (inv_tau_ahp, set_inv_tau_ahp, pillars::zoo::GLIF4ModelParamIdx::inv_tau_ahp as usize, 1.0 / 100.0),
        (delta_g, set_delta_g, pillars::zoo::GLIF4ModelParamIdx::delta_g as usize, 2.0),
        (e_ahp, set_e_ahp, pillars::zoo::GLIF4ModelParamIdx::e_ahp as usize, -70.0),
        (inv_tau_theta, set_inv_tau_theta, pillars::zoo::GLIF4ModelParamIdx::inv_tau_theta as usize, 1.0 / 10.0),
        (delta_theta, set_delta_theta, pillars::zoo::GLIF4ModelParamIdx::delta_theta as usize, 0.5),
        (tau_ref, set_tau_ref, pillars::zoo::GLIF4ModelParamIdx::tau_ref as usize, 2.0)
    ],
    states: [
        (v, set_v, pillars::zoo::GLIF4ModelStateIdx::v as usize, -65.0),
        (th, set_th, pillars::zoo::GLIF4ModelStateIdx::th as usize, -50.0),
        (g, set_g, pillars::zoo::GLIF4ModelStateIdx::g as usize, 0.0),
        (refr, set_refr, pillars::zoo::GLIF4ModelStateIdx::refr as usize, 0.0)
    ],
    extra: {}
}

define_population! {
    docs: "FitzHugh–Nagumo population (2D relaxation oscillator).\n\nParams: dt=0.01 (FHN relaxation is numerically unstable at larger dt), inv_tau_w=1/10,\na=0.7, b=0.8, v_thresh=0.5.\nStates: v, w, was_below.",
    class: PyFHNPopulation,
    py_name: "FHNPopulation",
    model: pillars::zoo::FHNModel,
    params: [
        (dt, set_dt, pillars::zoo::FHNModelParamIdx::dt as usize, 0.01),
        (inv_tau_w, set_inv_tau_w, pillars::zoo::FHNModelParamIdx::inv_tau_w as usize, 1.0 / 10.0),
        (a, set_a, pillars::zoo::FHNModelParamIdx::a as usize, 0.7),
        (b, set_b, pillars::zoo::FHNModelParamIdx::b as usize, 0.8),
        (v_thresh, set_v_thresh, pillars::zoo::FHNModelParamIdx::v_thresh as usize, 0.5)
    ],
    states: [
        (v, set_v, pillars::zoo::FHNModelStateIdx::v as usize, 0.0),
        (w, set_w, pillars::zoo::FHNModelStateIdx::w as usize, 0.0),
        (was_below, set_was_below, pillars::zoo::FHNModelStateIdx::was_below as usize, 1.0)
    ],
    extra: {}
}

define_population! {
    docs: "Hindmarsh–Rose population (bursting 3D model).\n\nParams: dt=0.01, a=1.0, b=3.0, c=1.0, d=5.0, r=0.006, s=4.0, x_r=-1.6, v_thresh=1.0.\nStates: v (=x), y, z.",
    class: PyHRPopulation,
    py_name: "HRPopulation",
    model: pillars::zoo::HRModel,
    params: [
        (dt, set_dt, pillars::zoo::HRModelParamIdx::dt as usize, 0.01),
        (a, set_a, pillars::zoo::HRModelParamIdx::a as usize, 1.0),
        (b, set_b, pillars::zoo::HRModelParamIdx::b as usize, 3.0),
        (c, set_c, pillars::zoo::HRModelParamIdx::c as usize, 1.0),
        (d, set_d, pillars::zoo::HRModelParamIdx::d as usize, 5.0),
        (r, set_r, pillars::zoo::HRModelParamIdx::r as usize, 0.006),
        (s, set_s, pillars::zoo::HRModelParamIdx::s as usize, 4.0),
        (x_r, set_x_r, pillars::zoo::HRModelParamIdx::x_r as usize, -1.6),
        (v_thresh, set_v_thresh, pillars::zoo::HRModelParamIdx::v_thresh as usize, 1.0)
    ],
    states: [
        (v, set_v, pillars::zoo::HRModelStateIdx::x as usize, -1.6),
        (y, set_y, pillars::zoo::HRModelStateIdx::y as usize, -12.0),
        (z, set_z, pillars::zoo::HRModelStateIdx::z as usize, 0.0)
    ],
    extra: {}
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
    docs: "LIF with spike-triggered threshold adaptation (LIFAt): θ decays toward 0 and jumps by beta on each spike — effective threshold v_thresh + θ.",
    class: PyLIFAtPopulation,
    py_name: "LIFAtPopulation",
    model: LIFAtModel,
    params: [
        (dt, set_dt, 0, 0.5), (inv_tau_m, set_inv_tau_m, 1, 0.1), (v_rest, set_v_rest, 2, -70.0),
        (v_thresh, set_v_thresh, 3, -54.0), (v_reset, set_v_reset, 4, -70.0),
        (inv_tau_theta, set_inv_tau_theta, 5, 0.05), (beta, set_beta, 6, 5.0),
        (tau_ref, set_tau_ref, 7, 3.0)
    ],
    states: [(v, set_v, 0, -70.0), (theta, set_theta, 1, 0.0), (refr, set_refr, 2, 0.0)],
    extra: {}
}

define_population! {
    docs: "Izhikevich with a slow memory state m (Sun et al. 2026) — trace dynamics plus a memory-to-current feedback w_m·m.",
    class: PyIzhikevichDualPopulation,
    py_name: "IzhikevichDualPopulation",
    model: IzhikevichDualModel,
    params: [
        (dt, set_dt, 0, 1.0), (a, set_a, 1, 0.02), (b, set_b, 2, 0.2), (c, set_c, 3, -65.0),
        (d, set_d, 4, 8.0), (tau_ref, set_tau_ref, 5, 3.0), (v_thresh, set_v_thresh, 6, 30.0),
        (trace_decay, set_trace_decay, 7, 0.5), (tau_m, set_tau_m, 8, 0.9), (w_m, set_w_m, 9, 1.0)
    ],
    states: [
        (v, set_v, 0, -65.0), (u, set_u, 1, -13.0), (refr, set_refr, 2, 0.0),
        (trace, set_trace, 3, 0.0), (m, set_m, 4, 0.0)
    ],
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
    docs: "Neuromotor model — muscle calcium, tension, length and velocity dynamics driven by spikes.",
    class: PyNeuromotorPopulation,
    py_name: "NeuromotorPopulation",
    model: NeuromotorModel,
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
        let graph = pillars::api::SparseGraph::from_csr(row_ptr, col_idx, &weights, &delays);
        let n = graph.num_neurons();
        let mut brain =
            pillars_connectome::Brain::<pillars::zoo::LIFModel>::from_sparse_graph(graph);
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
        let graph = pillars::api::SparseGraph::from_csr(row_ptr, col_idx, &weights, &delays);
        let n = graph.num_neurons();
        let mut brain =
            pillars_connectome::SynapticBrain::<pillars::zoo::LIFModel>::from_sparse_graph(
                graph,
                receptor_types,
                dt,
            );
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

/// Human hand/forearm reference runtime backed by the Neuromotor muscle model.
///
/// Eight muscles (FCR, ECRL, FCU, ECU, FDS-II, FDP-II, ECRB, FDI) with
/// forces in newtons and angle-dependent wrist moment arms.
#[pyclass(name = "HumanHandState", module = "taas_checkmate.pillars")]
pub struct PyHumanHandState {
    inner: HumanHandState,
}

#[pymethods]
impl PyHumanHandState {
    /// Create the reference human hand runtime (all forces/params in SI).
    #[new]
    fn new() -> Self {
        Self {
            inner: HumanHandState::new(),
        }
    }

    /// Advance one step in isometric mode with motor-neuron spikes and resets
    /// (int32 arrays of length 8). `dt` is in seconds.
    fn step(
        &mut self,
        spikes: Bound<'_, PyArray1<i32>>,
        resets: Bound<'_, PyArray1<i32>>,
        dt: f32,
    ) -> PyResult<()> {
        let sp = spikes.readonly();
        let rs = resets.readonly();
        let sp_s = sp.as_slice()?;
        let rs_s = rs.as_slice()?;
        if sp_s.len() != 8 || rs_s.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spikes/resets must be length 8",
            ));
        }
        self.inner.step(sp_s, rs_s, dt, None, None);
        Ok(())
    }

    /// Advance one step with kinematics: `lengths` (m) and `velocities` (m/s)
    /// arrays of length 8, plus motor spikes/resets. `dt` is in seconds.
    fn step_kinematic(
        &mut self,
        spikes: Bound<'_, PyArray1<i32>>,
        resets: Bound<'_, PyArray1<i32>>,
        dt: f32,
        lengths: Bound<'_, PyArray1<f32>>,
        velocities: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let sp = spikes.readonly();
        let rs = resets.readonly();
        let lr = lengths.readonly();
        let vr = velocities.readonly();
        let sp_s = sp.as_slice()?;
        let rs_s = rs.as_slice()?;
        let l_s = lr.as_slice()?;
        let v_s = vr.as_slice()?;
        if sp_s.len() != 8 || rs_s.len() != 8 || l_s.len() != 8 || v_s.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spikes/resets/lengths/velocities must all be length 8",
            ));
        }
        self.inner.step(sp_s, rs_s, dt, Some(l_s), Some(v_s));
        Ok(())
    }

    /// Set the wrist flexion angle in degrees (positive = flexion). Re-evaluates
    /// the angle-dependent wrist moment arms immediately.
    fn set_wrist_angle_deg(&mut self, angle_deg: f32) {
        self.inner.set_wrist_angle_deg(angle_deg);
    }

    /// Current wrist flexion angle in degrees.
    #[getter]
    fn wrist_angle_deg(&self) -> f32 {
        self.inner.wrist_angle_deg
    }

    /// Current muscle forces in newtons (length-8 float32 array).
    #[getter]
    fn force_n<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.force_n)
    }

    /// Net wrist moment in N·m at the current angle.
    #[getter]
    fn wrist_moment_nm(&self) -> f32 {
        self.inner.wrist_moment_nm
    }

    /// Muscle channel names (FCR, ECRL, FCU, ECU, FDS-II, FDP-II, ECRB, FDI).
    #[getter]
    fn muscle_names(&self) -> Vec<String> {
        HumanMuscle::ALL
            .iter()
            .map(|m| m.name().to_string())
            .collect()
    }

    /// Maximum isometric force capacity per muscle (newtons, length-8).
    #[getter]
    fn max_isometric_force_n<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let p = self.inner.profile;
        let caps: Vec<f32> = p
            .muscles
            .iter()
            .map(|s| s.max_isometric_force_n())
            .collect();
        PyArray1::from_slice(py, &caps)
    }

    /// Sum of all muscle force capacities (newtons).
    #[getter]
    fn peak_force_n(&self) -> f32 {
        self.inner.profile.peak_force_n()
    }

    /// Peak wrist moment in N·m at a given wrist angle (all muscles at
    /// maximum isometric force).
    fn peak_wrist_moment_nm_at_angle(&self, angle_deg: f32) -> f32 {
        self.inner.profile.peak_wrist_moment_nm_at_angle(angle_deg)
    }

    fn __repr__(&self) -> String {
        format!("HumanHandState(angle={}°)", self.inner.wrist_angle_deg)
    }
}

/// Runtime neuron factory: create a population of `n` neurons for a named
/// zoo model.
///
/// Names are case-insensitive and accept both short forms ("hh") and
/// class-style forms ("HHPopulation", "hh_population", "morris-lecar").
/// Returns the concrete population class — all population classes share the
/// duck-typed surface: `step(i_in, spikes_out)`, `n`, per-parameter/state
/// getters/setters, and `set_param_array`/`set_state_array`.
///
/// Raises `ValueError` for unknown model names.
/// camelCase → snake_case: insert '_' before an uppercase letter only when it
/// follows a lowercase letter or digit (so "HHPopulation" and "VSNPopulation"
/// stay intact as "hh"/"vsn", while "IzhikevichDualPopulation" → "izhikevich_dual").
fn camel_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_lower_or_digit = false;
    for ch in s.chars() {
        if ch.is_ascii_uppercase() {
            if prev_lower_or_digit {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_lower_or_digit = false;
        } else {
            out.push(ch);
            prev_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (name, n))]
pub fn make_population<'py>(py: Python<'py>, name: &str, n: usize) -> PyResult<Py<PyAny>> {
    let mut key = camel_to_snake(name.trim());
    key = key.replace('-', "_");
    let key = key
        .strip_suffix("population")
        .unwrap_or(&key)
        .trim_end_matches('_');
    let pop: Py<PyAny> = match key {
        "lif" => Py::new(py, PyLIFPopulation::new(n))?.into_any(),
        "izhikevich" | "izh" => Py::new(py, PyIzhikevichPopulation::new(n))?.into_any(),
        "izhikevich_dual" | "dual" => Py::new(py, PyIzhikevichDualPopulation::new(n))?.into_any(),
        "hh" => Py::new(py, PyHHPopulation::new(n))?.into_any(),
        // VSN is the only manual class whose Rust `new` takes the full param
        // list; mirror the pyo3-signature defaults here (dt, tau_m, v_rest,
        // v_thresh, v_reset, beta, surrogate_gain).
        "vsn" => Py::new(
            py,
            PyVSNPopulation::new(n, 0.5, 10.0, -70.0, -54.0, -70.0, 0.9, 10.0),
        )?
        .into_any(),
        "rate" => Py::new(py, PyRatePopulation::new(n))?.into_any(),
        "adex" | "ad_ex" => Py::new(py, PyAdExPopulation::new(n))?.into_any(),
        "glif3" => Py::new(py, PyGLIF3Population::new(n))?.into_any(),
        "glif4" => Py::new(py, PyGLIF4Population::new(n))?.into_any(),
        "fhn" => Py::new(py, PyFHNPopulation::new(n))?.into_any(),
        "hr" => Py::new(py, PyHRPopulation::new(n))?.into_any(),
        "haap" => Py::new(py, PyHAAPPopulation::new(n))?.into_any(),
        "morris_lecar" | "ml" => Py::new(py, PyMorrisLecarPopulation::new(n))?.into_any(),
        "saw" => Py::new(py, PySAWPopulation::new(n))?.into_any(),
        "wc" | "wilson_cowan" => Py::new(py, PyWCPopulation::new(n))?.into_any(),
        "triangulated" | "lut" => Py::new(py, PyTriangulatedPopulation::new(n))?.into_any(),
        "neuromotor" => Py::new(py, PyNeuromotorPopulation::new(n))?.into_any(),
        "lifat" | "lif_at" => Py::new(py, PyLIFAtPopulation::new(n))?.into_any(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown neuron model '{name}' (normalized '{other}'); available: \
                 lif, izhikevich, izhikevich_dual, hh, vsn, rate, adex, glif3, glif4, \
                 fhn, hr, haap, morris_lecar, saw, wc, triangulated, neuromotor, lifat"
            )));
        }
    };
    Ok(pop)
}

/// Register the pillars sub-module in the Python module tree.
///
/// The identical class list is registered twice: on the `pillars` facade
/// sub-module and at the root module (backward compat) — the macro keeps the
/// two lists from drifting apart.
pub fn register<'py>(m: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    macro_rules! register_pillars_classes {
        ($mod:expr) => {{
            let mod_: &Bound<'_, PyModule> = &$mod;
            mod_.add_class::<PyLIFPopulation>()?;
            mod_.add_class::<PyIzhikevichPopulation>()?;
            mod_.add_class::<PyHHPopulation>()?;
            mod_.add_class::<PyVSNPopulation>()?;
            mod_.add_class::<PyRatePopulation>()?;
            mod_.add_class::<PyAdExPopulation>()?;
            mod_.add_class::<PyGLIF4Population>()?;
            mod_.add_class::<PyFHNPopulation>()?;
            mod_.add_class::<PyHRPopulation>()?;
            mod_.add_class::<PyIzhikevichDualPopulation>()?;
            mod_.add_class::<PyLIFAtPopulation>()?;
            mod_.add_class::<PyGLIF3Population>()?;
            mod_.add_class::<PyHAAPPopulation>()?;
            mod_.add_class::<PyMorrisLecarPopulation>()?;
            mod_.add_class::<PySAWPopulation>()?;
            mod_.add_class::<PyWCPopulation>()?;
            mod_.add_class::<PyTriangulatedPopulation>()?;
            mod_.add_class::<PyNeuromotorPopulation>()?;
            mod_.add_class::<PyTransformerPopulation>()?;
            mod_.add_class::<PyHumanHandState>()?;
            mod_.add_class::<PyConnectomeBrain>()?;
            mod_.add_class::<PySynapticConnectomeBrain>()?;
        }};
    }

    let sub = PyModule::new(m.py(), "pillars")?;
    register_pillars_classes!(sub);
    sub.add_function(pyo3::wrap_pyfunction!(make_population, &sub)?)?;
    m.add_submodule(&sub)?;
    // Register in sys.modules under the public name so
    // `from taas_checkmate.pillars import X` works (not just attribute access).
    // Strip the <pkg>.<ext> suffix from the extension's own __name__ first.
    let parent_name = m.name()?;
    let parent_name = parent_name.to_str()?;
    let public_root = match parent_name.rsplit_once('.') {
        Some((root, "taas_checkmate")) => root,
        _ => parent_name,
    };
    let fq = format!("{public_root}.pillars");
    sub.setattr("__name__", fq.clone())?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(&fq, sub.clone())?;
    // Also register at root for convenience
    register_pillars_classes!(m);
    m.add_function(pyo3::wrap_pyfunction!(make_population, m)?)?;
    Ok(sub)
}
