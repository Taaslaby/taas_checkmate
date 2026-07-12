//! PyO3 bindings for `taas-connectome` — universal connectome brain simulator.
//!
//! Exposes:
//! - `taas_checkmate.connectome_brain.ConnectomeData` — build from CSR arrays
//! - `taas_checkmate.connectome_brain.BrainSimulator` — LIF/Izhikevich/Rate/Psi
//! - `taas_checkmate.connectome_brain.SpikeRecorder` — spike counting for named groups
//!
//! Usage:
//! ```python
//! import taas_checkmate as tcm
//!
//! conn = tcm.ConnectomeData(rp, ci, vals, nt_types, [("dn", dn_ids), ("mn", mn_ids)])
//! sim = tcm.BrainSimulator(conn.n_neurons, "lif")
//! spike_buf = sim.step_batch(conn, n_steps=50, external_I=...)
//! rates = tcm.SpikeRecorder(conn.n_neurons).record_group(
//!     spike_buf, 50, 0.5, conn.group("mn"), max_rate=100.0)
//!
//! # Conductance-based (requires receptor_types):
//! conn.build_receptor_types()
//! spike_buf = sim.step_batch_synaptic(conn, n_steps=50, external_I=...)
//! ```

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArray2, PyArrayMethods, IntoPyArray};
use pyo3::prelude::*;

use crate::connectome::{
    self as conn,
    ConnectomeData, NeuronModel,
    LifParams, IzhikevichParams, RateParams, PsiParams,
    SynapticKinetics,
};

// ─── ConnectomeData ──────────────────────────────────

/// Universal connectome: CSR matrix + NT types + named neuron groups.
///
/// Can be built from any source (FlyWire, MANC, synthetic, etc.)
/// and then passed to ``BrainSimulator.step_batch()``.
#[pyclass(name = "ConnectomeData", module = "taas_checkmate.connectome_brain")]
pub struct PyConnectomeData {
    pub inner: ConnectomeData,
}

#[pymethods]
impl PyConnectomeData {
    #[new]
    #[pyo3(signature = (row_ptr, col_indices, values, nt_types, groups=None))]
    pub fn new(
        row_ptr: Bound<'_, PyArray1<i32>>,
        col_indices: Bound<'_, PyArray1<i32>>,
        values: Bound<'_, PyArray1<f32>>,
        nt_types: Bound<'_, PyArray1<f32>>,
        groups: Option<Vec<(String, Vec<usize>)>>,
    ) -> PyResult<Self> {
        let rp = row_ptr.readonly();
        let ci = col_indices.readonly();
        let vl = values.readonly();
        let nt = nt_types.readonly();

        let groups = groups.unwrap_or_default();
        let groups_refs: Vec<(&str, Vec<usize>)> = groups.iter()
            .map(|(name, indices)| (name.as_str(), indices.clone()))
            .collect();

        let inner = ConnectomeData::new(
            rp.as_slice()?.to_vec(),
            ci.as_slice()?.to_vec(),
            vl.as_slice()?.to_vec(),
            nt.as_slice()?.to_vec(),
            groups_refs,
        );

        Ok(Self { inner })
    }

    /// Number of neurons in the connectome.
    #[getter]
    pub fn n_neurons(&self) -> usize {
        self.inner.n_neurons
    }

    /// Number of synapses (non-zero CSR entries).
    #[getter]
    pub fn n_synapses(&self) -> usize {
        self.inner.n_synapses
    }

    /// Get neuron indices for a named group.
    /// Returns None if group not found.
    pub fn group(&self, name: &str) -> Option<Vec<usize>> {
        self.inner.group(name).map(|s| s.to_vec())
    }

    /// List all group names registered in this connectome.
    pub fn group_names(&self) -> Vec<String> {
        self.inner.groups.iter().map(|g| g.name.clone()).collect()
    }

    /// Raw row_ptr (read-only copy).
    pub fn row_ptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.inner.row_ptr.clone().into_pyarray(py)
    }

    /// Raw col_indices (read-only copy).
    pub fn col_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.inner.col_indices.clone().into_pyarray(py)
    }

    /// Raw values (read-only copy).
    pub fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.values.clone().into_pyarray(py)
    }

    /// Raw nt_types (read-only copy).
    pub fn nt_types<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.nt_types.clone().into_pyarray(py)
    }

    /// Build per-synapse receptor_types from nt_types (sign-based mapping).
    /// Must be called once before using ``step_batch_synaptic``.
    pub fn build_receptor_types(&mut self) {
        self.inner.build_receptor_types();
    }

    /// Raw receptor_types (read-only copy) — per synapse, maps to ReceptorType.
    pub fn receptor_types<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u8>>> {
        if self.inner.receptor_types.is_empty() {
            None
        } else {
            Some(self.inner.receptor_types.clone().into_pyarray(py))
        }
    }

    /// Check if receptor_types have been built.
    pub fn has_receptor_types(&self) -> bool {
        !self.inner.receptor_types.is_empty()
    }
}

// ─── BrainSimulator ──────────────────────────────────

/// Universal brain simulator for any CSR connectome.
///
/// Supports models: ``"lif"``, ``"izhikevich"``, ``"rate"``, ``"psi"``.
///
/// Usage:
/// ```python
/// sim = tcm.BrainSimulator(conn.n_neurons, "lif")
/// spike_buf = sim.step_batch(conn, n_steps=50, dt_ms=0.5, external_I=...)
/// ```
#[pyclass(name = "BrainSimulator", module = "taas_checkmate.connectome_brain")]
pub struct PyBrainSimulator {
    state: conn::BrainState,
    model: NeuronModel,
    n_neurons: usize,
    kinetics: Option<SynapticKinetics>,
}

#[pymethods]
impl PyBrainSimulator {
    #[new]
    #[pyo3(signature = (n_neurons, model="lif".to_string()))]
    pub fn new(n_neurons: usize, model: String) -> PyResult<Self> {
        let nm = match model.as_str() {
            "lif" => NeuronModel::LIF,
            "izhikevich" => NeuronModel::Izhikevich,
            "rate" => NeuronModel::Rate,
            "psi" => NeuronModel::Psi,
            other => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown model '{other}'. Use: lif, izhikevich, rate, psi")
            )),
        };
        let state = conn::BrainState::new(n_neurons, nm);
        Ok(Self { state, model: nm, n_neurons, kinetics: None })
    }

    /// Getter for n_neurons.
    #[getter]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Run a batch of simulation steps on the connectome.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT types.
    ///     n_steps: number of simulation steps.
    ///     dt_ms: timestep in milliseconds (default 0.5).
    ///     external_I: external input current ``[n_neurons]`` (optional, zeros if None).
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[pyo3(signature = (connectome, n_steps, dt_ms=0.5, external_I=None, scale=1.0))]
    pub fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
        scale: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let conn_data = &connectome.inner;
        let n = self.n_neurons;
        let ext_I: Vec<f32> = match external_I {
            Some(arr) => {
                let ro = arr.readonly();
                let sl = ro.as_slice()?;
                let mut v = vec![0.0f32; n];
                let copy_n = sl.len().min(n);
                v[..copy_n].copy_from_slice(&sl[..copy_n]);
                v
            }
            None => vec![0.0f32; n],
        };

        let spike_buf = match self.model {
            NeuronModel::LIF => {
                let tau_m = vec![20.0f32; n];
                let v_rest = vec![-70.0f32; n];
                let v_thresh = vec![-58.0f32; n];
                let v_reset = vec![-70.0f32; n];
                let params = LifParams::new(&tau_m, &v_rest, &v_thresh, &v_reset, dt_ms, 2.0);
                conn::sim_lif(conn_data, &mut self.state, &params, &ext_I, n_steps, scale)
            }
            NeuronModel::Izhikevich => {
                let params = IzhikevichParams {
                    a: vec![0.02f32; n],
                    b: vec![0.2f32; n],
                    c: vec![-65.0f32; n],
                    d: vec![8.0f32; n],
                    dt_ms,
                    tau_ref: 2.0,
                };
                conn::sim_izhikevich(conn_data, &mut self.state, &params, &ext_I, n_steps)
            }
            NeuronModel::Rate => {
                let params = RateParams { tau: 10.0, dt_ms };
                conn::sim_rate(conn_data, &mut self.state, &params, &ext_I, n_steps)
            }
            NeuronModel::Psi => {
                let params = PsiParams { hbar: 1.0, dt_ms };
                conn::sim_psi(conn_data, &mut self.state, &params, &ext_I, n_steps)
            }
        };

        Ok(spike_buf.into_pyarray(py))
    }

    /// Run a batch of conductance-based LIF steps.
    ///
    /// Requires ``connectome.build_receptor_types()`` to have been called first.
    /// Conductance-based synapses give realistic PSC decay (AMPA τ≈2ms, GABA_A τ≈10ms).
    /// Synaptic kinetics persist across calls — conductances carry over between batches.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT + receptor types.
    ///     n_steps: number of simulation steps.
    ///     dt_ms: timestep in milliseconds (default 0.5).
    ///     external_I: external input current ``[n_neurons]`` (optional, zeros if None).
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[pyo3(signature = (connectome, n_steps, dt_ms=0.5, external_I=None))]
    pub fn step_batch_synaptic<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if connectome.inner.receptor_types.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "receptor_types empty — call connectome.build_receptor_types() first"
            ));
        }

        let conn_data = &connectome.inner;
        let n = self.n_neurons;

        let ext_I: Vec<f32> = match external_I {
            Some(arr) => {
                let ro = arr.readonly();
                let sl = ro.as_slice()?;
                let mut v = vec![0.0f32; n];
                let copy_n = sl.len().min(n);
                v[..copy_n].copy_from_slice(&sl[..copy_n]);
                v
            }
            None => vec![0.0f32; n],
        };

        // Initialize kinetics on first call
        if self.kinetics.is_none() {
            self.kinetics = Some(SynapticKinetics::new(n));
        }
        let kinetics = self.kinetics.as_mut().unwrap();

        let spike_buf = match self.model {
            NeuronModel::LIF => {
                let tau_m = vec![20.0f32; n];
                let v_rest = vec![-70.0f32; n];
                let v_thresh = vec![-58.0f32; n];
                let v_reset = vec![-70.0f32; n];
                let params = LifParams::new(&tau_m, &v_rest, &v_thresh, &v_reset, dt_ms, 2.0);
                conn::sim_lif_synaptic(conn_data, &mut self.state, &params, &ext_I, n_steps, kinetics)
            }
            NeuronModel::Izhikevich => {
                let params = IzhikevichParams {
                    a: vec![0.02f32; n],
                    b: vec![0.2f32; n],
                    c: vec![-65.0f32; n],
                    d: vec![8.0f32; n],
                    dt_ms,
                    tau_ref: 2.0,
                };
                conn::sim_izhikevich_synaptic(conn_data, &mut self.state, &params, &ext_I, n_steps, kinetics)
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Synaptic mode not supported for model {:?}", self.model)
            )),
        };

        Ok(spike_buf.into_pyarray(py))
    }

    /// Reset synaptic conductances to zero (keeps membrane potentials intact).
    pub fn reset_kinetics(&mut self) {
        if let Some(k) = &mut self.kinetics {
            k.reset();
        }
    }

    /// Reset brain state with small noise. Also resets synaptic conductances.
    pub fn reset(&mut self) {
        self.state = conn::BrainState::new(self.n_neurons, self.model);
        self.state.reset_with_noise(self.n_neurons);
        if let Some(k) = &mut self.kinetics {
            k.reset();
        }
    }

    /// Set membrane potentials (v) from numpy array.
    /// Copies values into the internal state buffer, respecting padding.
    pub fn set_v(&mut self, v_arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = v_arr.readonly();
        let sl = ro.as_slice()?;
        let n = sl.len().min(self.state.v.len());
        self.state.v[..n].copy_from_slice(&sl[..n]);
        Ok(())
    }

    /// Set recovery variables (u) from numpy array.
    /// Necessary for Izhikevich GPU state sync (v is set via set_v, u via this).
    pub fn set_u(&mut self, u_arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = u_arr.readonly();
        let sl = ro.as_slice()?;
        let n = sl.len().min(self.state.u.len());
        self.state.u[..n].copy_from_slice(&sl[..n]);
        Ok(())
    }

    /// Current membrane potentials (v) — read-only snapshot.
    pub fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.state.v[..self.n_neurons].to_vec().into_pyarray(py)
    }

    /// Current recovery variables (u) — read-only snapshot.
    pub fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.state.u[..self.n_neurons].to_vec().into_pyarray(py)
    }

    /// Run ψ-wave with time-varying external input, record (v, u) trajectory.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT types.
    ///     external_I: ``numpy.ndarray`` shape ``(n_changes, n_neurons)`` — external input per change.
    ///     steps_per_input: PSI steps per external input.
    ///     dt_ms: timestep in milliseconds.
    ///     hbar: Planck constant for ψ-wave.
    ///     record_every: record state every N steps (default 0 = record at end of each input block).
    ///
    /// Returns:
    ///     ``(v_traj, u_traj)`` — each ``(n_records * n_neurons,)`` flat float32 array.
    ///     Reshape with ``.reshape(-1, n_neurons)`` in Python.
    #[pyo3(signature = (connectome, external_I, steps_per_input, dt_ms=0.5, hbar=1.0, record_every=0))]
    pub fn psi_record<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        external_I: Bound<'_, PyArray2<f32>>,
        steps_per_input: usize,
        dt_ms: f32,
        hbar: f32,
        record_every: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let n_neurons = self.n_neurons;
        let n_changes = external_I.len()?;
        if n_changes == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "external_I must have at least one row".to_string()
            ));
        }
        let ro = external_I.readonly();
        let ext_I_sl = ro.as_slice()?;
        if ext_I_sl.len() != n_changes * n_neurons {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("external_I shape ({n_changes},?) expected n_neurons={n_neurons}, got {} elements", ext_I_sl.len())
            ));
        }
        let params = conn::PsiParams { hbar, dt_ms };

        let (v_traj, u_traj) = conn::sim_psi_record(
            &connectome.inner,
            &mut self.state,
            &params,
            ext_I_sl,
            n_changes,
            steps_per_input,
            record_every,
        );

        let v_arr = PyArray1::from_vec(py, v_traj);
        let u_arr = PyArray1::from_vec(py, u_traj);
        Ok((v_arr, u_arr))
    }

    /// Run a batch of LIF steps on GPU (CUDA).
    ///
    /// Requires `--features cuda` build. Falls back to error if GPU unavailable.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT types.
    ///     n_steps: number of simulation steps.
    ///     dt_ms: timestep in milliseconds.
    ///     external_I: external input current ``[n_neurons]``.
    ///     tau_m: membrane time constant ``[n_neurons]``.
    ///     v_rest: resting potential ``[n_neurons]``.
    ///     v_thresh: threshold potential ``[n_neurons]``.
    ///     v_reset: reset potential ``[n_neurons]``.
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (connectome, n_steps, dt_ms, external_I, tau_m, v_rest, v_thresh, v_reset))]
    pub fn step_batch_gpu<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Bound<'_, PyArray1<f32>>,
        tau_m: Bound<'_, PyArray1<f32>>,
        v_rest: Bound<'_, PyArray1<f32>>,
        v_thresh: Bound<'_, PyArray1<f32>>,
        v_reset: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let conn_data = &connectome.inner;
        let ext_I = external_I.readonly();
        let ext_I_sl = ext_I.as_slice()?;
        let tau_m_ro = tau_m.readonly();
        let tau_m_sl = tau_m_ro.as_slice()?;
        let v_rest_ro = v_rest.readonly();
        let v_rest_sl = v_rest_ro.as_slice()?;
        let v_thresh_ro = v_thresh.readonly();
        let v_thresh_sl = v_thresh_ro.as_slice()?;
        let v_reset_ro = v_reset.readonly();
        let v_reset_sl = v_reset_ro.as_slice()?;

        let spike_buf = conn::sim_lif_gpu(
            conn_data, &mut self.state, dt_ms, ext_I_sl,
            n_steps, tau_m_sl, v_rest_sl, v_thresh_sl, v_reset_sl,
        );
        Ok(spike_buf.into_pyarray(py))
    }

    /// Run a batch of Izhikevich steps on GPU (CUDA).
    ///
    /// Requires `--features cuda` build.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT types.
    ///     n_steps: number of simulation steps.
    ///     dt_ms: timestep in milliseconds.
    ///     external_I: external input current ``[n_neurons]``.
    ///     a: Izhikevich recovery rate ``[n_neurons]``.
    ///     b: Izhikevich sensitivity ``[n_neurons]``.
    ///     c: Izhikevich after-spike reset ``[n_neurons]``.
    ///     d: Izhikevich after-spike recovery ``[n_neurons]``.
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (connectome, n_steps, dt_ms, external_I, a, b, c, d))]
    pub fn step_batch_gpu_izhikevich<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Bound<'_, PyArray1<f32>>,
        a: Bound<'_, PyArray1<f32>>,
        b: Bound<'_, PyArray1<f32>>,
        c: Bound<'_, PyArray1<f32>>,
        d: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let conn_data = &connectome.inner;
        let ext_I = external_I.readonly();
        let ext_I_sl = ext_I.as_slice()?;
        let a_ro = a.readonly();
        let a_sl = a_ro.as_slice()?;
        let b_ro = b.readonly();
        let b_sl = b_ro.as_slice()?;
        let c_ro = c.readonly();
        let c_sl = c_ro.as_slice()?;
        let d_ro = d.readonly();
        let d_sl = d_ro.as_slice()?;

        let spike_buf = conn::sim_izhikevich_gpu(
            conn_data, &mut self.state, dt_ms, ext_I_sl,
            n_steps, a_sl, b_sl, c_sl, d_sl,
        );
        Ok(spike_buf.into_pyarray(py))
    }
}

// ─── SpikeRecorder ──────────────────────────────────

/// Counts spikes for named neuron groups from a spike buffer.
///
/// Usage:
/// ```python
/// recorder = tcm.SpikeRecorder(conn.n_neurons)
/// rates = recorder.record_group(spike_buf, n_steps=50, dt_ms=0.5,
///                                indices=dn_list, max_rate=100.0)
/// ```
#[pyclass(name = "SpikeRecorder", module = "taas_checkmate.connectome_brain")]
pub struct PySpikeRecorder {
    inner: conn::SpikeRecorder,
}

#[pymethods]
impl PySpikeRecorder {
    #[new]
    pub fn new(n_neurons: usize) -> Self {
        Self {
            inner: conn::SpikeRecorder::new(n_neurons),
        }
    }

    /// Extract firing rates for a single neuron group.
    ///
    /// Args:
    ///     spike_buffer: ``(n_steps, n_neurons)`` int32 array — 0/1 per neuron per step.
    ///     n_steps: number of timesteps.
    ///     dt_ms: timestep in milliseconds.
    ///     indices: list of neuron indices to count.
    ///     max_rate: max firing rate for normalization (Hz). Rates clamped to [0, 1].
    ///
    /// Returns:
    ///     ``list[float]`` — firing rates in [0, 1] for each neuron in indices.
    pub fn record_group(
        &mut self,
        spike_buffer: Bound<'_, PyArray1<i32>>,
        n_steps: usize,
        dt_ms: f32,
        indices: Vec<usize>,
        max_rate: f32,
    ) -> PyResult<Vec<f32>> {
        let sb = spike_buffer.readonly();
        let sb_sl = sb.as_slice()?;
        Ok(self.inner.record_group(
            sb_sl, n_steps, dt_ms, &indices, max_rate,
        ))
    }

    /// Record rates for multiple named groups at once.
    ///
    /// Returns dict of ``{group_name: list[float]}`` — firing rates in [0, 1].
    pub fn record(
        &mut self,
        spike_buffer: Bound<'_, PyArray1<i32>>,
        n_steps: usize,
        dt_ms: f32,
        group_names: Vec<String>,
        group_indices: Vec<Vec<usize>>,
        max_rate: f32,
    ) -> PyResult<Vec<Vec<f32>>> {
        if group_names.len() != group_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "group_names and group_indices must have same length"
            ));
        }
        let sb = spike_buffer.readonly();
        let sb_sl = sb.as_slice()?;

        let group_strs: Vec<&str> = group_names.iter().map(|s| s.as_str()).collect();
        let groups: Vec<(&str, &[usize])> = group_strs.iter().copied().zip(
            group_indices.iter().map(|v| v.as_slice())
        ).collect();

        let recordings = self.inner.record(sb_sl, n_steps, dt_ms, &groups);

        // Return per-group rates
        let mut result = Vec::with_capacity(group_names.len());
        for name in &group_names {
            if let Some(rates) = recordings.rates.get(name) {
                result.push(rates.clone());
            } else {
                result.push(vec![0.0; 1]); // shouldn't happen
            }
        }
        Ok(result)
    }

    /// Reset internal buffers.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
