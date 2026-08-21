//! NeuroML2 → Python bindings: load a `.nml` document and run it.
//!
//! ```python
//! from taas_checkmate.neuroml import load, load_file, NeuromlNetwork
//!
//! net = load(xml, dt_ms=0.5, duration_ms=200.0)   # or load_file("model.nml")
//! net.warnings               # recognised-but-skipped elements
//! spikes = net.run()         # per-population spike counts (full sim)
//! v = net.run_trace(0)       # V(t) trace of population 0: [steps+1, n] f32
//! ca = net.ca(0)             # soma [Ca2+] (mM) per neuron of cable pop 0
//! ca_t = net.run_ca_trace(0) # soma [Ca2+](t): [steps+1, n] f32
//! v0 = net.state(0)          # final state column of population 0 (f32)
//! ```

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pillars::api::neuroml::{self, LoadedNetwork};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A compiled NeuroML2 network, ready to step.
#[pyclass(name = "NeuromlNetwork", module = "taas_checkmate.neuroml")]
pub struct NeuromlNetwork {
    inner: LoadedNetwork,
}

impl NeuromlNetwork {
    fn check_pop(&self, pop: usize) -> PyResult<()> {
        let n = self.inner.network.num_populations();
        if pop >= n {
            return Err(PyValueError::new_err(format!(
                "population {pop} out of range ({n} populations)"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl NeuromlNetwork {
    /// Run the full simulation; returns the per-population spike counts.
    fn run(&mut self) -> Vec<usize> {
        self.inner.run().spike_counts
    }

    /// Advance one integration step (manual stepping, e.g. for sampling).
    fn step(&mut self) {
        self.inner.network.step();
    }

    /// Run the full simulation while recording the state trace of population
    /// `pop` (column `col`, default V). Returns a `[steps+1, n_neurons]` float32
    /// array: row 0 is the initial state, row k the state after k steps.
    /// Inputs and clamps are applied exactly as in [`run`](Self::run).
    #[pyo3(signature = (pop, col=0))]
    fn run_trace<'py>(
        &mut self,
        py: Python<'py>,
        pop: usize,
        col: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.check_pop(pop)?;
        let steps = (self.inner.duration_ms / self.inner.dt_ms).round().max(1.0) as usize;
        let n = self.inner.network.state(pop, col).len();
        let mut trace: Vec<f32> = Vec::with_capacity((steps + 1) * n);
        trace.extend_from_slice(self.inner.network.state(pop, col));
        for s in 0..steps {
            let t = s as f32 * self.inner.dt_ms;
            for (p, indices, amps) in self.inner.injections_at(t) {
                self.inner.network.inject_current(p, &indices, &amps);
            }
            for (p, indices, amps) in self.inner.clamp_currents(t) {
                self.inner.network.inject_current(p, &indices, &amps);
            }
            self.inner.network.step();
            trace.extend_from_slice(self.inner.network.state(pop, col));
        }
        PyArray1::from_vec(py, trace).reshape([steps + 1, n])
    }

    /// Soma (root) intracellular Ca²⁺ concentration of every neuron in cable
    /// population `pop` (mM), or an error if the population has no Ca²⁺ `species`.
    fn ca<'py>(&self, py: Python<'py>, pop: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.check_pop(pop)?;
        let cable = self.inner.network.cable_ca(pop).ok_or_else(|| {
            PyValueError::new_err(format!("population {pop} has no Ca2+ species"))
        })?;
        let root = self.inner.cable_info(pop).map(|c| c.root).unwrap_or(0);
        let mut out = Vec::with_capacity(cable.n_neurons);
        for i in 0..cable.n_neurons {
            out.push(cable.ca[i * cable.n_comps + root]);
        }
        Ok(out.into_pyarray(py))
    }

    /// Run the full simulation while recording the soma [Ca²⁺] trace of cable
    /// population `pop` (mM). Returns a `[steps+1, n_neurons]` float32 array:
    /// row 0 is the initial concentration, row k after k steps. Inputs and
    /// clamps are applied exactly as in [`run`](Self::run).
    fn run_ca_trace<'py>(
        &mut self,
        py: Python<'py>,
        pop: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.check_pop(pop)?;
        if self.inner.network.cable_ca(pop).is_none() {
            return Err(PyValueError::new_err(format!(
                "population {pop} has no Ca2+ species"
            )));
        }
        let root = self.inner.cable_info(pop).map(|c| c.root).unwrap_or(0);
        let steps = (self.inner.duration_ms / self.inner.dt_ms).round().max(1.0) as usize;
        let n = self.inner.network.state(pop, 0).len();
        let mut trace: Vec<f32> = Vec::with_capacity((steps + 1) * n);
        // Initial concentrations (row 0).
        let cable = self.inner.network.cable_ca(pop).unwrap();
        for i in 0..cable.n_neurons {
            trace.push(cable.ca[i * cable.n_comps + root]);
        }
        for s in 0..steps {
            let t = s as f32 * self.inner.dt_ms;
            for (p, indices, amps) in self.inner.injections_at(t) {
                self.inner.network.inject_current(p, &indices, &amps);
            }
            for (p, indices, amps) in self.inner.clamp_currents(t) {
                self.inner.network.inject_current(p, &indices, &amps);
            }
            self.inner.network.step();
            let cable = self.inner.network.cable_ca(pop).unwrap();
            for i in 0..cable.n_neurons {
                trace.push(cable.ca[i * cable.n_comps + root]);
            }
        }
        PyArray1::from_vec(py, trace).reshape([steps + 1, n])
    }

    /// Current state column `col` of population `pop` (col 0 = membrane V).
    #[pyo3(signature = (pop, col=0))]
    fn state<'py>(
        &self,
        py: Python<'py>,
        pop: usize,
        col: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.check_pop(pop)?;
        Ok(self.inner.network.state(pop, col).to_vec().into_pyarray(py))
    }

    /// Spike flags of population `pop` after the last step.
    fn spikes<'py>(&self, py: Python<'py>, pop: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
        self.check_pop(pop)?;
        Ok(self.inner.network.spikes(pop).to_vec().into_pyarray(py))
    }

    /// Number of populations.
    fn num_populations(&self) -> usize {
        self.inner.network.num_populations()
    }

    /// Integration step (ms).
    #[getter]
    fn dt(&self) -> f32 {
        self.inner.dt_ms
    }

    /// Simulation duration (ms).
    #[getter]
    fn duration(&self) -> f32 {
        self.inner.duration_ms
    }

    /// Non-fatal warnings (recognised-but-skipped NeuroML2 elements).
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
}

/// Parse and compile a NeuroML2 document (XML string) with explicit
/// `dt_ms`/`duration_ms` (ms).
#[pyfunction]
#[pyo3(name = "load", signature = (xml, dt_ms = 0.5, duration_ms = 200.0))]
pub fn py_load_neuroml(xml: &str, dt_ms: f32, duration_ms: f32) -> PyResult<NeuromlNetwork> {
    let ln = neuroml::load_with(xml, dt_ms, duration_ms)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NeuromlNetwork { inner: ln })
}

/// Parse and compile a NeuroML2 document read from disk, resolving LEMS
/// `<include href="…"/>` elements relative to the file's directory.
#[pyfunction]
#[pyo3(name = "load_file")]
pub fn py_load_neuroml_file(path: &str) -> PyResult<NeuromlNetwork> {
    let ln = neuroml::load_file(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NeuromlNetwork { inner: ln })
}
