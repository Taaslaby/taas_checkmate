//! PyO3 bindings for LUT-based Hodgkin-Huxley simulation.
//!
//! Exposes `HHLUT` Python class that manages HH state + LUTs internally.
//! Call `.step()` to advance all neurons by one timestep.
//! Access `.v`, `.Ii`, `.gates` as numpy arrays.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::core::neuron::hh::{HhLutConfig, step_hh_lut, standard_config};

/// LUT-based Hodgkin-Huxley simulator.
///
/// Uses precomputed lookup tables for zero-exp() simulation.
/// Standard channels: NaV (m³h) + KV (n⁴).
/// Optional: CaV-T (T-type calcium) for pacemaking.
///
/// Usage:
/// ```python
/// sim = HHLUT(128, 0.01)       # 128 neurons, dt=0.01ms
/// sim.Ii[:] = 10.0             # inject current
/// for _ in range(1000):
///     spikes = sim.step()      # returns spike count
///     print(sim.v[:3])         # first 3 membrane potentials
/// ```
#[pyclass(name = "HHLUT")]
pub struct PyHHLUT {
    cfg: HhLutConfig,
}

#[pymethods]
impl PyHHLUT {
    /// Create HH simulator.
    ///
    /// Args:
    ///     n: number of neurons
    ///     dt: timestep in ms (default 0.01)
    ///     pacemaking: if True, adds CaV-T for spontaneous spiking
    #[new]
    pub fn new(n: usize, dt: f32, pacemaking: Option<bool>) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("n must be > 0"));
        }

        let cfg = if pacemaking.unwrap_or(false) {
            let channels = crate::kernels::neuron::hh::pacemaking_channels();
            crate::core::neuron::hh::build_config_from_channels(
                &channels, n, dt, -100.0, 60.0, 1024)
        } else {
            standard_config(n, dt)
        };

        Ok(PyHHLUT { cfg })
    }

    /// Step all neurons by one timestep.
    /// Returns: total spike count (int)
    pub fn step(&mut self) -> usize {
        let v_before = self.cfg.v.clone();
        step_hh_lut(&mut self.cfg);

        let mut count = 0;
        for i in 0..self.cfg.n {
            if self.cfg.v[i] > 0.0 && v_before[i] <= 0.0 {
                count += 1;
            }
        }
        count
    }

    /// Membrane potentials.
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.v.clone())
    }

    /// Input currents.
    #[getter]
    pub fn Ii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.Ii.clone())
    }

    /// Gate variables, row-major [n_neurons * n_gates].
    #[getter]
    pub fn gates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.gates.clone())
    }

    /// Leak conductances.
    #[getter]
    pub fn g_leak<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.g_leak.clone())
    }

    /// Leak reversal potentials.
    #[getter]
    pub fn E_leak<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.E_leak.clone())
    }

    /// Membrane capacitances.
    #[getter]
    pub fn C_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.C_m.clone())
    }

    /// Number of neurons.
    #[getter]
    pub fn n(&self) -> usize { self.cfg.n }

    /// Total gates across all channels.
    #[getter]
    pub fn n_gates(&self) -> usize { self.cfg.n_gates }

    /// Number of ion channel types.
    #[getter]
    pub fn n_channels(&self) -> usize { self.cfg.n_channels }

    /// Timestep in ms.
    #[getter]
    pub fn dt(&self) -> f32 { self.cfg.dt }

    /// Reset all neurons to resting state (correct steady-state gates).
    pub fn reset(&mut self) {
        self.cfg.v.fill(-65.0);
        for i in 0..self.cfg.n {
            let s = i * self.cfg.n_gates;
            let g = &self.cfg.init_gates;
            for j in 0..self.cfg.n_gates {
                self.cfg.gates[s + j] = g[j];
            }
        }
        self.cfg.Ii.fill(0.0);
    }

    /// Set input current for a single neuron.
    pub fn set_I(&mut self, idx: usize, value: f32) -> PyResult<()> {
        if idx >= self.cfg.n {
            return Err(PyValueError::new_err(format!("index {idx} out of range (n={})", self.cfg.n)));
        }
        self.cfg.Ii[idx] = value;
        Ok(())
    }

    /// Set input current for all neurons at once.
    pub fn set_I_all(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice()
            .map_err(|e| PyValueError::new_err(format!("I array: {e}")))?;
        if slice.len() != self.cfg.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}", self.cfg.n, slice.len()
            )));
        }
        self.cfg.Ii.copy_from_slice(slice);
        Ok(())
    }

    /// Set membrane potentials for all neurons.
    pub fn set_v(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice()
            .map_err(|e| PyValueError::new_err(format!("v array: {e}")))?;
        if slice.len() != self.cfg.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}", self.cfg.n, slice.len()
            )));
        }
        self.cfg.v.copy_from_slice(slice);
        Ok(())
    }

    /// Run simulation for multiple steps.
    /// Returns: list of spike counts per step
    pub fn run(&mut self, n_steps: usize) -> Vec<usize> {
        let mut spike_counts = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            spike_counts.push(self.step());
        }
        spike_counts
    }
}
