//! PyO3 bindings for pillars-based Hodgkin-Huxley simulation.
//!
//! Exposes `HHLUT` Python class that manages HH state internally via
//! [`NeuralPopulation<HHModel>`](pillars) (SIMD engine, no legacy LUT config).
//! Call `.step()` to advance all neurons by one timestep.
//! Access `.v`, `.Ii`, `.gates` as numpy arrays.
//!
//! Legacy `HhLutConfig` / `step_hh_lut` were removed from `taas-core` — this
//! class is the direct pillars replacement with the same Python API surface.

#![allow(non_snake_case)]
#![allow(deprecated)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::connectome::{hh_population_pacemaking, hh_population_standard};
use crate::pillars::zoo::HHModel;
use crate::pillars::zoo::HHModelParamIdx;
use crate::pillars::NeuralPopulation;

/// Pillars-based Hodgkin-Huxley simulator.
///
/// Uses the SIMD `NeuralPopulation<HHModel>` engine (no LUTs, no legacy config).
/// Standard channels: NaV (m³h) + KV (n⁴) + leak.
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
    pop: NeuralPopulation<HHModel>,
    i_in: Vec<f32>,   // input current buffer (fed to pop.step each call)
    spikes: Vec<i32>, // spike output buffer
    n: usize,
    n_gates: usize,    // 3 = m,h,n (standard), 4 = +c (pacemaking)
    n_channels: usize, // 2 = NaV,KV (standard), 3 = +CaV-T (pacemaking)
    dt: f32,
    pacemaking: bool,
}

impl PyHhLut {
    fn from_pop(pop: NeuralPopulation<HHModel>, dt: f32, pacemaking: bool) -> Self {
        let n = pop.len();
        Self {
            n_gates: if pacemaking { 4 } else { 3 },
            n_channels: if pacemaking { 3 } else { 2 },
            i_in: vec![0.0; n],
            spikes: vec![0; n],
            pop,
            n,
            dt,
            pacemaking,
        }
    }
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
        let pm = pacemaking.unwrap_or(false);
        let pop = if pm {
            hh_population_pacemaking(n, dt)
        } else {
            hh_population_standard(n, dt)
        };
        Ok(Self::from_pop(pop, dt, pm))
    }

    /// Step all neurons by one timestep.
    /// Returns: total spike count (int)
    pub fn step(&mut self) -> usize {
        self.pop.step(&self.i_in, &mut self.spikes);
        self.spikes.iter().map(|&s| s as usize).sum()
    }

    /// Membrane potentials.
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.pop.state_slice(0).to_vec())
    }

    /// Input currents (buffer passed to the SIMD step).
    #[getter]
    pub fn Ii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.i_in.clone())
    }

    /// Gate variables, column-major [n_gates * n] (m,h,n[,c] rows).
    #[getter]
    pub fn gates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut gates = Vec::with_capacity(self.n_gates * self.n);
        for g in 0..self.n_gates {
            gates.extend_from_slice(self.pop.state_slice(g + 1));
        }
        PyArray1::from_vec(py, gates)
    }

    /// Leak conductances.
    #[getter]
    pub fn g_leak<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(
            py,
            self.pop
                .param_slice(HHModelParamIdx::g_Leak.into())
                .to_vec(),
        )
    }

    /// Set leak conductances.
    pub fn set_g_leak(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("g_leak: {e}")))?;
        if slice.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.n,
                slice.len()
            )));
        }
        let p = self.pop.param_slice_mut(HHModelParamIdx::g_Leak.into());
        for i in 0..self.n {
            p[i] = slice[i];
        }
        Ok(())
    }

    /// Leak reversal potentials.
    #[getter]
    pub fn E_leak<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(
            py,
            self.pop
                .param_slice(HHModelParamIdx::E_Leak.into())
                .to_vec(),
        )
    }

    /// Set leak reversal potentials.
    pub fn set_E_leak(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("E_leak: {e}")))?;
        if slice.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.n,
                slice.len()
            )));
        }
        let p = self.pop.param_slice_mut(HHModelParamIdx::E_Leak.into());
        for i in 0..self.n {
            p[i] = slice[i];
        }
        Ok(())
    }

    /// Membrane capacitances (derived: C_m = 1/inv_cm).
    #[getter]
    pub fn C_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let inv = self.pop.param_slice(HHModelParamIdx::inv_cm.into());
        PyArray1::from_vec(py, inv.iter().map(|&v| 1.0 / v).collect())
    }

    /// Maximum conductances per channel per neuron [n_channels * n].
    /// Column-major: [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
    /// Standard: [0..n]=NaV, [n..2n]=KV.
    #[getter]
    pub fn g_max_arr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut arr = Vec::with_capacity(self.n_channels * self.n);
        let chans = [
            HHModelParamIdx::g_Na,
            HHModelParamIdx::g_K,
            HHModelParamIdx::g_CaT,
        ];
        for ch in chans.iter().take(self.n_channels) {
            arr.extend_from_slice(self.pop.param_slice((*ch).into()));
        }
        PyArray1::from_vec(py, arr)
    }

    /// Set maximum conductances per channel per neuron.
    ///
    /// Args:
    ///     values: [n_channels] (broadcast to all neurons) or [n_channels * n] (per-neuron).
    pub fn set_g_max_arr(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("g_max_arr: {e}")))?;
        let chans = [
            HHModelParamIdx::g_Na,
            HHModelParamIdx::g_K,
            HHModelParamIdx::g_CaT,
        ];
        if slice.len() == self.n_channels {
            // Broadcast per-channel to all neurons
            for (ch_i, &val) in slice.iter().enumerate() {
                self.pop.fill_param(chans[ch_i].into(), val);
            }
        } else if slice.len() == self.n_channels * self.n {
            for (ch_i, ch) in chans.iter().enumerate().take(self.n_channels) {
                let p = self.pop.param_slice_mut((*ch).into());
                for ni in 0..self.n {
                    p[ni] = slice[ch_i * self.n + ni];
                }
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "g_max_arr length {}: expected n_channels={} (broadcast) or n_channels*n={}",
                slice.len(),
                self.n_channels,
                self.n_channels * self.n
            )));
        }
        Ok(())
    }

    /// Reversal potentials for each ion channel type [n_channels] (mV).
    /// Standard: [0]=NaV (+50 mV), [1]=KV (-77 mV).
    #[getter]
    pub fn E_rev_arr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let chans = [
            HHModelParamIdx::E_Na,
            HHModelParamIdx::E_K,
            HHModelParamIdx::E_CaT,
        ];
        let arr: Vec<f32> = chans
            .iter()
            .take(self.n_channels)
            .map(|ch| self.pop.param_slice((*ch).into())[0])
            .collect();
        PyArray1::from_vec(py, arr)
    }

    /// Set reversal potentials for each ion channel type.
    pub fn set_E_rev_arr(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("E_rev_arr: {e}")))?;
        if slice.len() != self.n_channels {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.n_channels,
                slice.len()
            )));
        }
        let chans = [
            HHModelParamIdx::E_Na,
            HHModelParamIdx::E_K,
            HHModelParamIdx::E_CaT,
        ];
        for (ch_i, &val) in slice.iter().enumerate() {
            self.pop.fill_param(chans[ch_i].into(), val);
        }
        Ok(())
    }

    /// Number of neurons.
    #[getter]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total gates across all channels.
    #[getter]
    pub fn n_gates(&self) -> usize {
        self.n_gates
    }

    /// Number of ion channel types.
    #[getter]
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }

    /// Timestep in ms.
    #[getter]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Reset all neurons to resting state (correct steady-state gates).
    pub fn reset(&mut self) {
        // Rebuild a fresh population — same init as the constructor.
        self.pop = if self.pacemaking {
            hh_population_pacemaking(self.n, self.dt)
        } else {
            hh_population_standard(self.n, self.dt)
        };
        self.i_in.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set input current for a single neuron.
    pub fn set_I(&mut self, idx: usize, value: f32) -> PyResult<()> {
        if idx >= self.n {
            return Err(PyValueError::new_err(format!(
                "index {idx} out of range (n={})",
                self.n
            )));
        }
        self.i_in[idx] = value;
        Ok(())
    }

    /// Set input current for all neurons at once.
    pub fn set_I_all(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("I array: {e}")))?;
        if slice.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.n,
                slice.len()
            )));
        }
        self.i_in.copy_from_slice(slice);
        Ok(())
    }

    /// Set membrane potentials for all neurons.
    pub fn set_v(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("v array: {e}")))?;
        if slice.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "expected {} values, got {}",
                self.n,
                slice.len()
            )));
        }
        let p = self.pop.state_slice_mut(0);
        for i in 0..self.n {
            p[i] = slice[i];
        }
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
