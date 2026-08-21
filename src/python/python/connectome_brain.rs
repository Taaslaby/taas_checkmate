//! PyO3 bindings for `taas-connectome` — universal connectome brain simulator.
//!
//! Exposes:
//! - `taas_checkmate.connectome_brain.ConnectomeData` — build from CSR arrays
//! - `taas_checkmate.connectome_brain.BrainRunner` — universal runner (LIF, Izh,
//!   IzhDual, HH, HAAP, Rate, GCO, GPU paths)
//! - `taas_checkmate.connectome_brain.SpikeRecorder` — spike counting for named groups
//!
//! Usage:
//! ```python
//! import taas_checkmate as tcm
//!
//! conn = tcm.ConnectomeData(rp, ci, vals, nt_types, [("dn", dn_ids), ("mn", mn_ids)])
//! runner = tcm.BrainRunner(conn, "lif", dt_ms=0.5)
//! spike_buf = runner.run(n_steps=50, external_I=...)
//! rates = tcm.SpikeRecorder(conn.n_neurons).record_group(
//!     spike_buf, 50, 0.5, conn.group("mn"), max_rate=100.0)
//! ```

#![allow(non_snake_case)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::connectome::{
    self as conn, AlphaLifOutput, AlphaLifWorker, ConnectomeData, ShiuParams,
};

// ─── ConnectomeData ──────────────────────────────────

/// Universal connectome: CSR matrix + NT types + named neuron groups.
///
/// Can be built from any source (FlyWire, MANC, synthetic, etc.)
/// and then passed to ``BrainRunner.run()``.
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
        let groups_refs: Vec<(&str, Vec<usize>)> = groups
            .iter()
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

    /// Set per-synapse receptor_types directly (e.g. loaded from a baked .taas).
    ///
    /// ``receptor_types`` must be ``[n_synapses]`` uint8 with data.rs codes:
    ///   0 = AMPA (exc fallback), 2 = GABA_A (fast inh), 4 = nAChR (fast exc),
    ///   5 = GluCl (fast inh — insects!), 6 = mAChR (slow mod).
    ///
    /// Must be called before constructing a ``BrainRunner(model="hh_synaptic")``.
    pub fn set_receptor_types(
        &mut self,
        receptor_types: Bound<'_, PyArray1<u8>>,
    ) -> PyResult<()> {
        let ro = receptor_types.readonly();
        let sl = ro.as_slice()?;
        if sl.len() != self.inner.n_synapses {
            return Err(PyValueError::new_err(format!(
                "set_receptor_types: expected {} entries (n_synapses), got {}",
                self.inner.n_synapses,
                sl.len()
            )));
        }
        self.inner.receptor_types = sl.to_vec();
        Ok(())
    }

    /// Build per-synapse receptor_types with Drosophila-appropriate FlyWire NT mapping.
    ///
    /// ``nt_map`` must be ``[n_neurons]`` float32 array with FlyWire NT codes:
    ///   +1.0 = ACh → nAChR (fast exc, E_rev=0)
    ///   -1.0 = GABA → GABA_A (shunting inh, E_rev=-65, Cl⁻ shunt at rest)
    ///   +2.0 = GLUT → GluCl (inhibitory Cl⁻ channel, E_rev=-70, Drosophila Cl⁻ influx)
    ///   -2.0 = DA/SER/OCT → mAChR (slow mod, E_rev=-90)
    ///
    /// For Princeton baked data with +1/-1 nt_types, pass ``nt_types`` as nt_map.
    pub fn build_receptor_types_flywire(
        &mut self,
        nt_map: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<()> {
        let ro = nt_map.readonly();
        let sl = ro.as_slice()?;
        self.inner.build_receptor_types_flywire(sl);
        Ok(())
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

// ─── BrainRunner ──────────────────────────────────

/// Universal brain runner — ONE Rust call replaces the slow Python loop.
///
/// Supports all neuron models: ``"lif"``, ``"lif_synaptic"``, ``"izh"``,
/// ``"izh_synaptic"``, ``"izh_dual"``, ``"izh_eprop"``, ``"hh"``,
/// ``"hh_synaptic"``, ``"haap"``, ``"rate"``, ``"gco"``.
///
/// Usage:
/// ```python
/// runner = tcm.BrainRunner(connectome, "lif", dt_ms=0.5)
/// runner.add_poisson_group(asc_indices, rate=150.0, weight=10.0)
/// spike_buf = runner.run(n_steps=50, external_I=ext_I)
/// trains = runner.spike_trains(dt_ms=0.5)
/// rates = runner.firing_rates(dt_ms=0.5)
/// ```
#[pyclass(name = "BrainRunner", module = "taas_checkmate.connectome_brain")]
pub struct PyBrainRunner {
    inner: conn::BrainRunner,
}

#[pymethods]
impl PyBrainRunner {
    #[new]
    #[pyo3(signature = (connectome, model="lif".to_string(), dt_ms=0.5, tau_ref=2.0, tau_m=20.0, v_rest=-70.0, v_thresh=-58.0, v_reset=-70.0, pacemaking=false, tau_syn=5.0, t_delay=1.8))]
    #[allow(unused_variables)] // tau_ref/v_rest/v_thresh/v_reset reserved for future configs
    pub fn new(
        connectome: &PyConnectomeData,
        model: String,
        dt_ms: f32,
        tau_ref: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        pacemaking: bool,
        tau_syn: f32,
        t_delay: f32,
    ) -> PyResult<Self> {
        let config = match model.as_str() {
            "lif" => conn::BrainConfig::lif(dt_ms),
            "lif_synaptic" => conn::BrainConfig::lif_synaptic(dt_ms),
            "lif_alpha" => conn::NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                t_delay,
            },
            "izh" | "izhikevich" => conn::BrainConfig::izh(dt_ms),
            "izh_synaptic" => conn::BrainConfig::izh(dt_ms), // + receptor_types
            "izh_dual" => conn::BrainConfig::izh_dual(dt_ms),
            "izh_eprop" => conn::BrainConfig::izh_eprop(dt_ms),
            "hh" => {
                if pacemaking {
                    conn::BrainConfig::hh_pacemaking(dt_ms)
                } else {
                    conn::BrainConfig::hh(dt_ms)
                }
            }
            "hh_synaptic" => conn::BrainConfig::hh_synaptic(dt_ms),
            "haap" => conn::BrainConfig::haap(dt_ms),
            "haap_synaptic" => conn::BrainConfig::haap_synaptic(dt_ms, tau_syn),
            "rate" => conn::BrainConfig::rate(dt_ms, tau_m),
            "gco" => conn::BrainConfig::gco(dt_ms),
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown model '{other}'. Use: lif, lif_synaptic, lif_alpha, izh, izh_synaptic, izh_dual, izh_eprop, hh, hh_synaptic, haap, haap_synaptic, rate, gco"
                )))
            }
        };

        let inner = conn::BrainRunner::new(connectome.inner.clone(), config)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;

        Ok(Self { inner })
    }

    /// Number of neurons.
    #[getter]
    pub fn n_neurons(&self) -> usize {
        self.inner.n_neurons()
    }

    /// Add a Poisson stimulation group.
    pub fn add_poisson_group(&mut self, indices: Vec<usize>, rate: f32, weight: f32) {
        self.inner.add_poisson_group(&indices, rate, weight);
    }

    /// Clear all Poisson groups.
    pub fn clear_poisson_groups(&mut self) {
        self.inner.clear_poisson_groups();
    }

    /// Override the refractory duration (ms) for specific neurons (alpha model).
    ///
    /// In the Shiu fly brain model, Poisson-stimulated ("exc") neurons have
    /// zero refractory so they can fire on every Poisson event.
    ///
    /// Args:
    ///     indices: neuron indices to override.
    ///     tau_ref_ms: new refractory duration in ms (0 = no refractory).
    pub fn set_alpha_refrac_ms(&mut self, indices: Vec<usize>, tau_ref_ms: f32) {
        self.inner.set_alpha_refrac_ms(&indices, tau_ref_ms);
    }

    /// Run `n_steps` simulation steps.
    ///
    /// Args:
    ///     n_steps: number of simulation steps.
    ///     external_I: external input current ``[n_neurons]`` (optional, zeros if None).
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[pyo3(signature = (n_steps, external_I=None))]
    pub fn run<'py>(
        &mut self,
        py: Python<'py>,
        n_steps: usize,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let n = self.inner.n_neurons();
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

        let spike_buf = self.inner.run(n_steps, &ext_I).to_vec();
        Ok(spike_buf.into_pyarray(py))
    }

    /// Run Izhikevich Dual Memory simulation with slow state pathway.
    ///
    /// The slow state accumulates recent spiking activity per neuron:
    /// ```text
    /// m[t] = τ_m · m[t-1] + (1-τ_m) · spike[t-1]
    /// I += w_m · m[t-1]
    /// ```
    ///
    /// Also applies Hebbian plasticity on CSR weights (in-place):
    /// ``dW = η · m_pre · m_post − α · W · dt``
    ///
    /// Args:
    ///     n_steps: number of simulation steps.
    ///     dt_ms: timestep in milliseconds (default 0.5).
    ///     external_I: external input current ``[n_neurons]`` (optional, zeros if None).
    ///     tau_m: slow memory time constant (0..1, higher = slower decay, default 0.999).
    ///     w_m: memory-to-current weight (how much slow state modulates I, default 5.0).
    ///     hebb_lr: Hebbian learning rate (default 0.001).
    ///     hebb_decay: Hebbian weight decay (default 0.0001).
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    ///
    /// Note: modifies the connectome weights in-place (Hebbian plasticity).
    /// Topology (row_ptr / col_indices) is unchanged, so no CSR refresh needed.
    #[pyo3(signature = (n_steps, dt_ms=0.5, external_I=None, tau_m=0.999, w_m=5.0, hebb_lr=0.001, hebb_decay=0.0001))]
    pub fn step_batch_dual<'py>(
        &mut self,
        py: Python<'py>,
        n_steps: usize,
        dt_ms: f32,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
        tau_m: f32,
        w_m: f32,
        hebb_lr: f32,
        hebb_decay: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let n = self.inner.n_neurons();
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

        let params = conn::IzhikevichDualParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms,
            tau_ref: 2.0,
            tau_m,
            w_m,
        };

        let inner = &mut self.inner;
        let spike_buf = conn::sim_izhikevich_dual(
            &mut inner.connectome,
            &mut inner.state,
            &params,
            &ext_I,
            n_steps,
            hebb_lr,
            hebb_decay,
        );
        Ok(spike_buf.into_pyarray(py))
    }

    /// Run GCO (Graph-Coupled Oscillator) with time-varying external input, record (v, u) trajectory.
    ///
    /// Args:
    ///     external_I: ``numpy.ndarray`` shape ``(n_changes, n_neurons)`` — external input per change.
    ///     steps_per_input: GCO steps per external input.
    ///     dt_ms: timestep in milliseconds.
    ///     tau: phase time scale for GCO.
    ///     record_every: record state every N steps (default 0 = record at end of each input block).
    ///
    /// Returns:
    ///     ``(v_traj, u_traj)`` — each ``(n_records * n_neurons,)`` flat float32 array.
    ///     Reshape with ``.reshape(-1, n_neurons)`` in Python.
    #[pyo3(signature = (external_I, steps_per_input, dt_ms=0.5, tau=1.0, record_every=0))]
    pub fn gco_record<'py>(
        &mut self,
        py: Python<'py>,
        external_I: Bound<'_, PyArray2<f32>>,
        steps_per_input: usize,
        dt_ms: f32,
        #[allow(unused_variables)] tau: f32,
        record_every: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let n_neurons = self.inner.n_neurons();
        let n_changes = external_I.len()?;
        if n_changes == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "external_I must have at least one row".to_string(),
            ));
        }
        let ro = external_I.readonly();
        let ext_I_sl = ro.as_slice()?;
        if ext_I_sl.len() != n_changes * n_neurons {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "external_I shape ({n_changes},?) expected n_neurons={n_neurons}, got {} elements",
                ext_I_sl.len()
            )));
        }
        let params = conn::GcoParams { dt_ms };
        let inner = &mut self.inner;
        let (v_traj, u_traj) = conn::sim_gco_record(
            &inner.connectome,
            &mut inner.state,
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
    #[pyo3(signature = (n_steps, dt_ms, external_I, tau_m, v_rest, v_thresh, v_reset))]
    pub fn step_batch_gpu<'py>(
        &mut self,
        py: Python<'py>,
        n_steps: usize,
        dt_ms: f32,
        external_I: Bound<'_, PyArray1<f32>>,
        tau_m: Bound<'_, PyArray1<f32>>,
        v_rest: Bound<'_, PyArray1<f32>>,
        v_thresh: Bound<'_, PyArray1<f32>>,
        v_reset: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
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

        let inner = &mut self.inner;
        let spike_buf = conn::sim_lif_gpu(
            &inner.connectome,
            &mut inner.state,
            dt_ms,
            ext_I_sl,
            n_steps,
            tau_m_sl,
            v_rest_sl,
            v_thresh_sl,
            v_reset_sl,
        );
        Ok(spike_buf.into_pyarray(py))
    }

    /// Run a batch of Izhikevich steps on GPU (CUDA).
    ///
    /// Requires `--features cuda` build.
    ///
    /// Args:
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
    #[pyo3(signature = (n_steps, dt_ms, external_I, a, b, c, d))]
    pub fn step_batch_gpu_izhikevich<'py>(
        &mut self,
        py: Python<'py>,
        n_steps: usize,
        dt_ms: f32,
        external_I: Bound<'_, PyArray1<f32>>,
        a: Bound<'_, PyArray1<f32>>,
        b: Bound<'_, PyArray1<f32>>,
        c: Bound<'_, PyArray1<f32>>,
        d: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
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

        let inner = &mut self.inner;
        let spike_buf = conn::sim_izhikevich_gpu(
            &inner.connectome,
            &mut inner.state,
            dt_ms,
            ext_I_sl,
            n_steps,
            a_sl,
            b_sl,
            c_sl,
            d_sl,
        );
        Ok(spike_buf.into_pyarray(py))
    }

    /// Set a per-neuron HH parameter column (only for HH / HH_synaptic models).
    ///
    /// Args:
    ///     col: parameter column index (HHModelParamIdx: 1=g_Na, 2=E_Na, 3=G_K,
    ///          4=E_K, 5=G_Leak, 6=E_Leak, 9=Inv_Cm, ...).
    ///     values: per-neuron array ``[n_neurons]``.
    ///
    /// Returns the number of neurons written. This enables literature-based
    /// per-class physiology (e.g. distinct g_Na/g_K/g_l/E_Leak per neuron).
    #[pyo3(signature = (col, values))]
    pub fn set_hh_param_column(&mut self, col: usize, values: Bound<'_, PyArray1<f32>>) -> PyResult<usize> {
        let ro = values.readonly();
        let sl = ro.as_slice()?;
        Ok(self.inner.set_hh_param_column(col, sl))
    }

    /// Set a per-neuron HAAP parameter column (only for the "haap" model).
    ///
    /// Args:
    ///     col: parameter column index (HAAPModelParamIdx: 0=dt, 1=Inv_Cm, 2=G_L,
    ///          3=E_L, 4=G_M, 5=E_K, 6=G_H, 7=E_H, 8=Inv_Tau_U, 9=Inv_Tau_H,
    ///          10=V_Half_U, 11=K_U, 12=V_Half_H, 13=K_H, 14=V_Thresh,
    ///          15=V_Reset, 16=Delta_U).
    ///     values: per-neuron array ``[n_neurons]``.
    ///
    /// Returns the number of neurons written. This enables literature-based
    /// per-class physiology (M-current + h-current per neuron type).
    #[pyo3(signature = (col, values))]
    pub fn set_haap_param_column(&mut self, col: usize, values: Bound<'_, PyArray1<f32>>) -> PyResult<usize> {
        let ro = values.readonly();
        let sl = ro.as_slice()?;
        Ok(self.inner.set_haap_param_column(col, sl))
    }

    /// Build spike trains from the spike buffer.
    ///
    /// Returns dict ``{neuron_idx: list_of_spike_times_ms}``.
    pub fn spike_trains(&self, dt_ms: f32) -> std::collections::HashMap<usize, Vec<f32>> {
        self.inner.spike_trains(dt_ms)
    }

    /// Compute firing rates (Hz) for all neurons.
    pub fn firing_rates(&self, dt_ms: f32) -> Vec<f32> {
        self.inner.firing_rates(dt_ms)
    }

    /// Compute firing rates (Hz) for a neuron group.
    pub fn firing_rates_group(&self, indices: Vec<usize>, n_steps: usize, dt_ms: f32) -> Vec<f32> {
        self.inner.firing_rates_group(&indices, n_steps, dt_ms)
    }

    /// Get current membrane potentials (v) — read-only snapshot.
    pub fn get_v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.get_v().to_vec().into_pyarray(py)
    }

    /// Set membrane potentials from numpy array.
    pub fn set_v(&mut self, v_arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = v_arr.readonly();
        let sl = ro.as_slice()?;
        self.inner.set_v(sl);
        Ok(())
    }

    /// Get current recovery variables (u).
    pub fn get_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.get_u().to_vec().into_pyarray(py)
    }

    /// Set recovery variables from numpy array.
    pub fn set_u(&mut self, u_arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = u_arr.readonly();
        let sl = ro.as_slice()?;
        self.inner.set_u(sl);
        Ok(())
    }

    /// Get the accumulated spike buffer.
    pub fn spike_buffer<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.inner.spike_buffer().to_vec().into_pyarray(py)
    }

    /// Total steps simulated so far.
    pub fn step_counter(&self) -> usize {
        self.inner.step_counter()
    }

    /// Reset simulator state (v, u, I_buf, spikes, kinetics).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get eligibility traces from E-Prop.
    pub fn eligibility_traces(&self) -> Vec<f32> {
        self.inner.eligibility_traces().to_vec()
    }

    /// Apply E-Prop weight update.
    pub fn apply_eprop_update(&mut self, learning_signals: Vec<f32>, lr: f32) {
        self.inner.apply_eprop_update(&learning_signals, lr);
    }
}

// ─── AlphaLifWorker (Shiu fly brain model) ─────────

/// Результат прогона [`PyAlphaLifWorker`] — метрики одного вызова.
#[pyclass(name = "AlphaLifOutput", module = "taas_checkmate.connectome_brain")]
pub struct PyAlphaLifOutput {
    pub inner: AlphaLifOutput,
}

#[pymethods]
impl PyAlphaLifOutput {
    /// Общее число спайков за прогон.
    #[getter]
    pub fn total_spikes(&self) -> usize {
        self.inner.total_spikes
    }

    /// Сколько нейронов спайкнуло хотя бы раз.
    #[getter]
    pub fn active_neurons(&self) -> usize {
        self.inner.active_neurons
    }

    /// Время симуляции мозга (мс).
    #[getter]
    pub fn brain_ms(&self) -> f32 {
        self.inner.brain_ms
    }

    /// Реальное время (с), потраченное на прогон.
    #[getter]
    pub fn wall_seconds(&self) -> f64 {
        self.inner.wall_seconds
    }

    /// Во сколько раз быстрее реального времени (1.0 = реальное время).
    #[getter]
    pub fn realtime_ratio(&self) -> f64 {
        self.inner.realtime_ratio()
    }

    fn __repr__(&self) -> String {
        format!(
            "AlphaLifOutput(spikes={}, active={}, brain_ms={:.1}, wall={:.3}s, ratio={:.3}x)",
            self.inner.total_spikes,
            self.inner.active_neurons,
            self.inner.brain_ms,
            self.inner.wall_seconds,
            self.inner.realtime_ratio()
        )
    }
}

/// Воркер: alpha-LIF модель Шу (Shiu et al. 2024, fly brain) на коннектоме.
///
/// Высокоуровневая обёртка над `BrainRunner(model="lif_alpha")`: сахарные GRN
/// стимулируются Poisson-входом, метрики считаются на лету чанками (без
/// гигантского spike-буфера).
///
/// ```python
/// from taas_checkmate import connectome_brain as cb
///
/// conn = cb.ConnectomeData(rp, ci, vals, nt_types)  # CSR, как обычно
/// worker = cb.AlphaLifWorker(conn)                  # ShiuParams::flywire()
/// worker.set_sugar_grn(sugar_idx)                  # 21 сахарный GRN, 200 Hz
/// out = worker.run_seconds(1.0)
/// print(out)  # spikes=17285 active=385 ...
/// ```
#[pyclass(name = "AlphaLifWorker", module = "taas_checkmate.connectome_brain")]
pub struct PyAlphaLifWorker {
    inner: AlphaLifWorker,
}

#[pymethods]
impl PyAlphaLifWorker {
    /// Создать воркер из коннектома с параметрами модели Шу (по умолчанию
    /// `ShiuParams::flywire()`: dt=0.1ms, v_rest=−52, v_th=−45, τ_mem=20ms,
    /// τ_ref=2.2ms, τ_syn=5ms, t_delay=1.8ms, Poisson 200Hz×68.75мВ).
    #[new]
    pub fn new(connectome: &PyConnectomeData) -> PyResult<Self> {
        let inner = AlphaLifWorker::new(connectome.inner.clone(), ShiuParams::flywire())
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
        Ok(Self { inner })
    }

    /// Задать стимулируемые GRN: рефрактерность 0 + Poisson-вход 200 Гц.
    pub fn set_sugar_grn(&mut self, indices: Vec<usize>) {
        self.inner.set_sugar_grn(&indices);
    }

    /// Прогнать `seconds` секунд мозга, вернуть метрики.
    ///
    /// Состояние сохраняется между вызовами — можно продолжать симуляцию
    /// (режим «always-on»).
    pub fn run_seconds(&mut self, seconds: f32) -> PyAlphaLifOutput {
        PyAlphaLifOutput {
            inner: self.inner.run_seconds(seconds),
        }
    }

    /// Прогнать `steps` шагов с опциональным внешним током, вернуть метрики.
    #[pyo3(signature = (steps, external_i=None))]
    pub fn run_steps(
        &mut self,
        steps: usize,
        external_i: Option<Bound<'_, PyArray1<f32>>>,
    ) -> PyResult<PyAlphaLifOutput> {
        let n = self.inner.n_neurons();
        let ext: Vec<f32> = match external_i {
            Some(arr) => {
                let ro = arr.readonly();
                let sl = ro.as_slice()?;
                let mut v = vec![0.0f32; n];
                let copy_n = sl.len().min(n);
                v[..copy_n].copy_from_slice(&sl[..copy_n]);
                v
            }
            None => Vec::new(),
        };
        Ok(PyAlphaLifOutput {
            inner: self.inner.run_steps(steps, &ext),
        })
    }

    /// Мгновенные мембранные потенциалы (для отладки/визуализации).
    pub fn get_v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.v().to_vec().into_pyarray(py)
    }

    /// Число нейронов в коннектоме.
    #[getter]
    pub fn n_neurons(&self) -> usize {
        self.inner.n_neurons()
    }

    /// Число стимулируемых GRN (0, если set_sugar_grn не вызывался).
    #[getter]
    pub fn n_sugar_grn(&self) -> usize {
        self.inner.n_sugar_grn()
    }

    /// Сбросить состояние (v→v_rest, g=0) — новая симуляция.
    pub fn reset(&mut self) {
        self.inner.reset();
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
        Ok(self
            .inner
            .record_group(sb_sl, n_steps, dt_ms, &indices, max_rate))
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
        _max_rate: f32,
    ) -> PyResult<Vec<Vec<f32>>> {
        if group_names.len() != group_indices.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "group_names and group_indices must have same length",
            ));
        }
        let sb = spike_buffer.readonly();
        let sb_sl = sb.as_slice()?;

        let group_strs: Vec<&str> = group_names.iter().map(|s| s.as_str()).collect();
        let groups: Vec<(&str, &[usize])> = group_strs
            .iter()
            .copied()
            .zip(group_indices.iter().map(|v| v.as_slice()))
            .collect();

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
