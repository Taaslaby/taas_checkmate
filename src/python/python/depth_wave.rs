//! PyO3 bindings for DepthGCO (depth-modulated graph-coupled oscillator).
//!
//! Exposes `DepthWave` Python class for ultra-fast CSR oscillator simulation in Rust.

#![allow(non_snake_case)]
#![allow(deprecated)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use taas_kernels::neuron::depth_wave::DepthWaveState;

/// Depth-modulated graph-coupled oscillator on CSR connectome graph.
///
/// speed[i] = 1.0 + depth[i] * mod_range
///
/// Usage:
/// ```python
/// dw = DepthWave(n_neurons, depth_arr, mod_range=2.0, dt=0.5)
/// dw.set_csr(row_ptr, col_idx, vals, nt_signs)
/// coherence, motor_amps = dw.step(n_steps=1)
/// ```
#[pyclass(name = "DepthWave", module = "taas_checkmate.neurons")]
pub struct PyDepthWave {
    state: GcoState,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    vals: Vec<f32>,
    nt_signs: Vec<f32>,
    has_csr: bool,
}

#[pymethods]
impl PyDepthWave {
    /// Create DepthWave simulator.
    ///
    /// Args:
    ///     n: number of neurons
    ///     depth: topological depth array [n] in [0, 1]
    ///     mod_range: strength of depth modulation (default 2.0)
    ///     dt: evolution timestep (default 0.5)
    ///     seed: random seed for initial phases (default 42)
    #[new]
    pub fn new(
        n: usize,
        depth: Bound<'_, PyArray1<f32>>,
        mod_range: Option<f32>,
        dt: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("n must be > 0"));
        }
        let ro_depth = depth.readonly();
        let depth_slice = ro_depth.as_slice().map_err(|e| PyValueError::new_err(format!("depth: {e}")))?;
        if depth_slice.len() != n {
            return Err(PyValueError::new_err(format!("depth length mismatch: expected {n}, got {}", depth_slice.len())));
        }

        let state = GcoState::new(
            n,
            depth_slice.to_vec(),
            mod_range.unwrap_or(2.0),
            dt.unwrap_or(0.5),
            seed.unwrap_or(42),
        );

        Ok(PyDepthWave {
            state,
            row_ptr: Vec::new(),
            col_idx: Vec::new(),
            vals: Vec::new(),
            nt_signs: vec![1.0; n],
            has_csr: false,
        })
    }

    /// Set CSR connectome graph.
    ///
    /// Args:
    ///     row_ptr: CSR row pointers [n + 1] uint32
    ///     col_idx: CSR column indices [n_synapses] uint32
    ///     vals: CSR weights [n_synapses] float32
    ///     nt_signs: Neurotransmitter signs per neuron [n] (+1 = exc, -1 = inh)
    pub fn set_csr(
        &mut self,
        row_ptr: Bound<'_, PyArray1<u32>>,
        col_idx: Bound<'_, PyArray1<u32>>,
        vals: Bound<'_, PyArray1<f32>>,
        nt_signs: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let r_ptr = row_ptr.readonly();
        let c_idx = col_idx.readonly();
        let v_arr = vals.readonly();
        let s_arr = nt_signs.readonly();

        let r_slice = r_ptr.as_slice().map_err(|e| PyValueError::new_err(format!("row_ptr: {e}")))?;
        let c_slice = c_idx.as_slice().map_err(|e| PyValueError::new_err(format!("col_idx: {e}")))?;
        let v_slice = v_arr.as_slice().map_err(|e| PyValueError::new_err(format!("vals: {e}")))?;
        let s_slice = s_arr.as_slice().map_err(|e| PyValueError::new_err(format!("nt_signs: {e}")))?;

        if r_slice.len() != self.state.n + 1 {
            return Err(PyValueError::new_err(format!("row_ptr length mismatch: expected {}, got {}", self.state.n + 1, r_slice.len())));
        }
        if c_slice.len() != v_slice.len() {
            return Err(PyValueError::new_err(format!("col_idx and vals length mismatch: {} vs {}", c_slice.len(), v_slice.len())));
        }
        if s_slice.len() != self.state.n {
            return Err(PyValueError::new_err(format!("nt_signs length mismatch: expected {}, got {}", self.state.n, s_slice.len())));
        }

        self.row_ptr = r_slice.to_vec();
        self.col_idx = c_slice.to_vec();
        self.vals = v_slice.to_vec();
        self.nt_signs = s_slice.to_vec();
        self.has_csr = true;
        Ok(())
    }

    /// Step simulation for `n_steps` timesteps.
    ///
    /// Returns: (coherence: float, motor_amplitudes: numpy array)
    pub fn step(
        &mut self,
        i_ext: Option<Bound<'_, PyArray1<f32>>>,
        n_steps: Option<usize>,
    ) -> PyResult<(f32, Vec<f32>)> {
        if !self.has_csr {
            return Err(PyValueError::new_err("CSR graph not set. Call set_csr() first."));
        }

        let steps = n_steps.unwrap_or(1);
        let ext_vec = if let Some(ext) = i_ext {
            let ro = ext.readonly();
            let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("i_ext: {e}")))?;
            Some(slice.to_vec())
        } else {
            None
        };

        let ext_ref = ext_vec.as_deref();

        for _ in 0..steps {
            self.state.step(&self.row_ptr, &self.col_idx, &self.vals, &self.nt_signs, ext_ref);
        }

        let coh = self.state.coherence();
        let motor_amps = self.state.motor_amplitudes();
        Ok((coh, motor_amps))
    }

    /// Fast batch execution for `n_steps` timesteps entirely in Rust.
    /// Returns: final coherence (float).
    pub fn run(&mut self, n_steps: usize) -> PyResult<f32> {
        if !self.has_csr {
            return Err(PyValueError::new_err("CSR graph not set. Call set_csr() first."));
        }
        for _ in 0..n_steps {
            self.state.step(&self.row_ptr, &self.col_idx, &self.vals, &self.nt_signs, None);
        }
        Ok(self.state.coherence())
    }

    /// Re(ψ) array [n].
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.state.v.clone())
    }

    /// Set Re(ψ) array [n].
    #[setter]
    pub fn set_v(&mut self, arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = arr.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("set_v: {e}")))?;
        if slice.len() == self.state.n {
            self.state.v.copy_from_slice(slice);
        }
        Ok(())
    }

    /// Im(ψ) array [n].
    #[getter]
    pub fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.state.u.clone())
    }

    /// Set Im(ψ) array [n].
    #[setter]
    pub fn set_u(&mut self, arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = arr.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("set_u: {e}")))?;
        if slice.len() == self.state.n {
            self.state.u.copy_from_slice(slice);
        }
        Ok(())
    }

    /// Topological depth array [n].
    #[getter]
    pub fn depth<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.state.depth.clone())
    }

    /// Speed array [n].
    #[getter]
    pub fn speed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.state.speed.clone())
    }

    /// Coherence |mean(ψ)|.
    #[getter]
    pub fn coherence(&self) -> f32 {
        self.state.coherence()
    }

    /// Motor amplitudes |ψ| for depth < 0.3.
    #[getter]
    pub fn motor_amplitudes(&self) -> Vec<f32> {
        self.state.motor_amplitudes()
    }

    /// Reset state.
    pub fn reset(&mut self, seed: Option<u64>) {
        self.state.reset(seed.unwrap_or(42));
    }
}
