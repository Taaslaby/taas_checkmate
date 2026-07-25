//! PyO3 bindings for LUT-based Hodgkin-Huxley simulation.
//!
//! Exposes `HHLUT` Python class that manages HH state + LUTs internally.
//! Call `.step()` to advance all neurons by one timestep.
//! Access `.v`, `.Ii`, `.gates` as numpy arrays.

#![allow(non_snake_case)]
#![allow(deprecated)]

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
/// sim = HhLut(128, 0.01)       # 128 neurons, dt=0.01ms
/// sim.Ii[:] = 10.0             # inject current
/// for _ in range(1000):
///     spikes = sim.step()      # returns spike count
///     print(sim.v[:3])         # first 3 membrane potentials
/// ```
#[pyclass(name = "HhLut", module = "taas_checkmate.hh")]
pub struct PyHhLut {
    cfg: HhLutConfig,
}

#[pymethods]
impl PyHhLut {
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

        Ok(PyHhLut { cfg })
    }

    /// Step all neurons by one timestep.
    /// Returns: total spike count (int), via reverse zero-crossing detection
    pub fn step(&mut self) -> usize {
        let mut spikes = vec![0i32; self.cfg.n];
        step_hh_lut(&mut self.cfg, &mut spikes);
        spikes.iter().map(|&s| s as usize).sum()
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

    /// Set leak conductances.
    pub fn set_g_leak(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("g_leak: {e}")))?;
        if slice.len() != self.cfg.n {
            return Err(PyValueError::new_err(format!("expected {} values, got {}", self.cfg.n, slice.len())));
        }
        self.cfg.g_leak.copy_from_slice(slice);
        Ok(())
    }

    /// Leak reversal potentials.
    #[getter]
    pub fn E_leak<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.E_leak.clone())
    }

    /// Set leak reversal potentials.
    pub fn set_E_leak(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("E_leak: {e}")))?;
        if slice.len() != self.cfg.n {
            return Err(PyValueError::new_err(format!("expected {} values, got {}", self.cfg.n, slice.len())));
        }
        self.cfg.E_leak.copy_from_slice(slice);
        Ok(())
    }

    /// Membrane capacitances.
    #[getter]
    pub fn C_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.C_m.clone())
    }

    /// Maximum conductances per channel per neuron [n_channels * n].
    /// Column-major: [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
    /// Standard: [0..n]=NaV (120 mS/cm²), [n..2n]=KV (36 mS/cm²).
    #[getter]
    pub fn g_max_arr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.g_max_arr.clone())
    }

    /// Set maximum conductances per channel per neuron.
    ///
    /// Args:
    ///     values: [n_channels] (broadcast to all neurons) or [n_channels * n] (per-neuron).
    pub fn set_g_max_arr(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("g_max_arr: {e}")))?;
        let n = self.cfg.n;
        if slice.len() == self.cfg.n_channels {
            // Broadcast per-channel to all neurons
            for (ch_i, &val) in slice.iter().enumerate() {
                for ni in 0..n {
                    self.cfg.g_max_arr[ch_i * n + ni] = val;
                }
            }
        } else if slice.len() == self.cfg.n_channels * n {
            self.cfg.g_max_arr.copy_from_slice(slice);
        } else {
            return Err(PyValueError::new_err(format!(
                "g_max_arr length {}: expected n_channels={} (broadcast) or n_channels*n={}",
                slice.len(), self.cfg.n_channels, self.cfg.n_channels * n
            )));
        }
        Ok(())
    }

    /// Reversal potentials for each ion channel type [n_channels] (mV).
    /// Standard: [0]=NaV (+50 mV), [1]=KV (-77 mV).
    #[getter]
    pub fn E_rev_arr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.cfg.E_rev_arr.clone())
    }

    /// Set reversal potentials for each ion channel type.
    pub fn set_E_rev_arr(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("E_rev_arr: {e}")))?;
        if slice.len() != self.cfg.n_channels {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}", self.cfg.n_channels, slice.len()
            )));
        }
        self.cfg.E_rev_arr.copy_from_slice(slice);
        Ok(())
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
        self.cfg.v_prev.fill(-65.0);
        self.cfg.Ii.fill(0.0);
        // Restore steady-state gates
        let n_gates = self.cfg.n_gates;
        for i in 0..self.cfg.n {
            let s = i * n_gates;
            self.cfg.gates[s..s + n_gates].copy_from_slice(&self.cfg.init_gates);
        }
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
