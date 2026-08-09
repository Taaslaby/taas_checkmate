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

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::connectome::{
    self as conn, hh_population_pacemaking, hh_population_standard, sim_hh_pillars,
    sim_hh_synaptic_pillars, ConnectomeData, GcoParams, IzhikevichParams, LifParams, NeuronModel,
    RateParams, SynapticKinetics,
};
use crate::pillars::zoo::HHModel;
use crate::pillars::zoo::HHParamIdx;
use crate::pillars::NeuralPopulation;

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

// ─── BrainSimulator ──────────────────────────────────

/// Universal brain simulator for any CSR connectome.
///
/// Supports models: ``"lif"``, ``"izhikevich"``, ``"rate"``, ``"psi"``, ``"hh"``.
///
/// > **Deprecated** — the pillars-native path replaces this for the models it
/// > covers (LIF, Izhikevich, Rate, and conductance-based synapses). Prefer:
/// > ```python
/// > from taas_checkmate import pillars
/// > brain = pillars.ConnectomeBrain(row_ptr, col_idx, weights, delays)   # instant
/// > brain = pillars.SynapticConnectomeBrain(row_ptr, col_idx, weights,
/// >                                        delays, receptor_types, dt)   # conductance
/// > ```
/// > `BrainSimulator` remains for the models without a pillars equivalent yet
/// > (``"gco"``, dual-Izhikevich, GPU paths).
///
/// Usage (legacy):
/// ```python
/// sim = tcm.BrainSimulator(conn.n_neurons, "lif")
/// spike_buf = sim.step_batch(conn, n_steps=50, dt_ms=0.5, external_I=...)
/// ```
#[pyclass(name = "BrainSimulator", module = "taas_checkmate.connectome_brain")]
pub struct PyBrainSimulator {
    state: conn::BrainState,
    model: conn::NeuronModel,
    n_neurons: usize,
    kinetics: Option<SynapticKinetics>,
    hh_pop: Option<(f32, NeuralPopulation<HHModel>)>, // (dt, pop) — lazily created for "hh" model
    hh_use_pacemaking: bool,                          // if true, use 3-channel HH (NaV+KV+CaT)
    hh_i_in: Vec<f32>,                                // last external input (Ii getter snapshot)
}

/// Ensure HH population exists for the given dt. Recreates if dt changed.
///
/// Free function over disjoint fields so callers can borrow `self.kinetics`
/// (or other fields) simultaneously with the returned population.
fn ensure_hh_pop(
    hh_pop: &mut Option<(f32, NeuralPopulation<HHModel>)>,
    hh_use_pacemaking: bool,
    n_neurons: usize,
    dt: f32,
) -> &mut NeuralPopulation<HHModel> {
    match hh_pop {
        Some((stored_dt, _)) if *stored_dt == dt => {}
        _ => {
            let pop = if hh_use_pacemaking {
                hh_population_pacemaking(n_neurons, dt)
            } else {
                hh_population_standard(n_neurons, dt)
            };
            *hh_pop = Some((dt, pop));
        }
    }
    &mut hh_pop.as_mut().unwrap().1
}

#[pymethods]
impl PyBrainSimulator {
    #[new]
    #[pyo3(signature = (n_neurons, model="lif".to_string(), pacemaking=false))]
    pub fn new(n_neurons: usize, model: String, pacemaking: bool) -> PyResult<Self> {
        let (nm, hh_use_pacemaking, hh_config) = match model.as_str() {
            "lif" => (NeuronModel::LIF, false, None),
            "izhikevich" => (NeuronModel::Izhikevich, false, None),
            "rate" => (NeuronModel::Rate, false, None),
            "psi" => (NeuronModel::LIF, false, None), // Psi removed
            "hh" => {
                if pacemaking {
                    let pop = hh_population_pacemaking(n_neurons, 0.5);
                    (NeuronModel::LIF, true, Some((0.0, pop)))
                } else {
                    let pop = hh_population_standard(n_neurons, 0.5);
                    (NeuronModel::LIF, false, Some((0.0, pop)))
                }
            }
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown model '{other}'. Use: lif, izhikevich, rate, psi, hh"
                )))
            }
        };
        let state = conn::BrainState::new(n_neurons, nm);
        Ok(Self {
            state,
            model: nm,
            n_neurons,
            kinetics: None,
            hh_pop: hh_config,
            hh_use_pacemaking,
            hh_i_in: vec![0.0; n_neurons],
        })
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

        let spike_buf = if self.hh_pop.is_some() {
            self.hh_i_in[..n].copy_from_slice(&ext_I[..n]);
            let pop = ensure_hh_pop(&mut self.hh_pop, self.hh_use_pacemaking, n, dt_ms);
            sim_hh_pillars(conn_data, pop, &ext_I, n_steps, scale)
        } else {
            match self.model {
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
                NeuronModel::Gco => {
                    let params = GcoParams { dt_ms };
                    conn::sim_gco(conn_data, &mut self.state, &params, &ext_I, n_steps)
                }
                NeuronModel::Hh => {
                    let pop = ensure_hh_pop(&mut self.hh_pop, self.hh_use_pacemaking, n, dt_ms);
                    sim_hh_pillars(conn_data, pop, &ext_I, n_steps, scale)
                }
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
    ///     scale: synaptic weight multiplier (default 1e10 for FAFB baked weights).
    ///
    /// Returns:
    ///     ``numpy.ndarray`` of shape ``(n_steps, n_neurons)`` — spike buffer (0/1).
    #[pyo3(signature = (connectome, n_steps, dt_ms=0.5, external_I=None, scale=1e10))]
    pub fn step_batch_synaptic<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
        scale: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if connectome.inner.receptor_types.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "receptor_types empty — call connectome.build_receptor_types() first",
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

        let spike_buf = if self.hh_pop.is_some() {
            self.hh_i_in[..n].copy_from_slice(&ext_I[..n]);
            let pop = ensure_hh_pop(&mut self.hh_pop, self.hh_use_pacemaking, n, dt_ms);
            let kinetics = self.kinetics.as_mut().unwrap();
            sim_hh_synaptic_pillars(conn_data, pop, &ext_I, n_steps, kinetics, scale)
        } else {
            let kinetics = self.kinetics.as_mut().unwrap();
            match self.model {
                NeuronModel::LIF => {
                    let tau_m = vec![20.0f32; n];
                    let v_rest = vec![-70.0f32; n];
                    let v_thresh = vec![-58.0f32; n];
                    let v_reset = vec![-70.0f32; n];
                    let params = LifParams::new(&tau_m, &v_rest, &v_thresh, &v_reset, dt_ms, 2.0);
                    conn::sim_lif_synaptic(
                        conn_data,
                        &mut self.state,
                        &params,
                        &ext_I,
                        n_steps,
                        kinetics,
                        scale,
                    )
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
                    conn::sim_izhikevich_synaptic(
                        conn_data,
                        &mut self.state,
                        &params,
                        &ext_I,
                        n_steps,
                        kinetics,
                        scale,
                    )
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Synaptic mode not supported for model {:?}",
                        self.model
                    )))
                }
            }
        };

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
    ///     connectome: ``ConnectomeData`` — CSR graph (weights are modified by Hebbian plasticity).
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
    #[pyo3(signature = (connectome, n_steps, dt_ms=0.5, external_I=None, tau_m=0.999, w_m=5.0, hebb_lr=0.001, hebb_decay=0.0001))]
    pub fn step_batch_dual<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &mut PyConnectomeData,
        n_steps: usize,
        dt_ms: f32,
        external_I: Option<Bound<'_, PyArray1<f32>>>,
        tau_m: f32,
        w_m: f32,
        hebb_lr: f32,
        hebb_decay: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let conn_data = &mut connectome.inner;
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

        let spike_buf = conn::sim_izhikevich_dual(
            conn_data,
            &mut self.state,
            &params,
            &ext_I,
            n_steps,
            hebb_lr,
            hebb_decay,
        );

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
        if let Some((dt, _)) = &self.hh_pop {
            let pop = if self.hh_use_pacemaking {
                hh_population_pacemaking(self.n_neurons, *dt)
            } else {
                hh_population_standard(self.n_neurons, *dt)
            };
            self.hh_pop = Some((*dt, pop));
        }
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
        if let Some((_, pop)) = &mut self.hh_pop {
            let n = sl.len().min(pop.len());
            let p = pop.state_slice_mut(0);
            for i in 0..n {
                p[i] = sl[i];
            }
        } else {
            let n = sl.len().min(self.state.v.len());
            self.state.v[..n].copy_from_slice(&sl[..n]);
        }
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
        if let Some((_, pop)) = &self.hh_pop {
            pop.state_slice(0)[..self.n_neurons]
                .to_vec()
                .into_pyarray(py)
        } else {
            self.state.v[..self.n_neurons].to_vec().into_pyarray(py)
        }
    }

    /// Current recovery variables (u) — read-only snapshot.
    pub fn u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.state.u[..self.n_neurons].to_vec().into_pyarray(py)
    }

    /// Set HH ion channel parameters (for "hh" model only).
    ///
    /// Args:
    ///     g_max_arr: per-channel max conductances ``[n_channels]`` e.g. [120.0, 36.0] for NaV, KV.
    ///     E_rev_arr: per-channel reversal potentials ``[n_channels]`` e.g. [50.0, -77.0] for Na, K.
    ///     g_leak: per-neuron leak conductance ``[n_neurons]`` (scalar accepted, broadcast).
    ///     E_leak: per-neuron leak reversal ``[n_neurons]`` (scalar accepted, broadcast).
    ///
    /// Must be called after constructing BrainSimulator(model="hh").
    /// Takes effect on the next ``step_batch()`` call.
    /// Pass ``dt_ms=`` to lock the HH timestep — prevents ``ensure_hh_config``
    /// from recreating the config with default parameters.
    #[pyo3(signature = (g_max_arr, E_rev_arr, g_leak, E_leak, dt_ms=None))]
    pub fn set_hh_params(
        &mut self,
        g_max_arr: Bound<'_, PyArray1<f32>>,
        E_rev_arr: Bound<'_, PyArray1<f32>>,
        g_leak: Bound<'_, PyArray1<f32>>,
        E_leak: Bound<'_, PyArray1<f32>>,
        dt_ms: Option<f32>,
    ) -> PyResult<()> {
        let hh_pop = match &mut self.hh_pop {
            Some((dt, pop)) => {
                // Lock the dt if provided — prevents ensure_hh_pop from recreating
                if let Some(dt_val) = dt_ms {
                    *dt = dt_val;
                }
                pop
            }
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "BrainSimulator not in HH mode — construct with model='hh'".to_string(),
                ))
            }
        };
        let n = self.n_neurons;
        let n_channels = if self.hh_use_pacemaking { 3 } else { 2 };
        let chans: &[HHParamIdx] = if self.hh_use_pacemaking {
            &[HHParamIdx::G_Na, HHParamIdx::G_K, HHParamIdx::G_CaT]
        } else {
            &[HHParamIdx::G_Na, HHParamIdx::G_K]
        };

        // g_max_arr: [n_channels] (broadcast) or [n_channels * n_neurons] (per-neuron)
        {
            let ro = g_max_arr.readonly();
            let sl = ro.as_slice()?;
            if sl.len() == n_channels {
                // Broadcast scalar per-channel g_max to all neurons
                for (ch_i, &val) in sl.iter().enumerate() {
                    hh_pop.fill_param(chans[ch_i].into(), val);
                }
            } else if sl.len() == n_channels * n {
                // Per-neuron g_max (column-major: [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...])
                for (ch_i, ch) in chans.iter().enumerate() {
                    let p = hh_pop.param_slice_mut((*ch).into());
                    for ni in 0..n {
                        p[ni] = sl[ch_i * n + ni];
                    }
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "g_max_arr length {}: expected n_channels={} (broadcast) or n_channels*n={}",
                    sl.len(),
                    n_channels,
                    n_channels * n
                )));
            }
        }

        // E_rev_arr: must match n_channels
        {
            let ro = E_rev_arr.readonly();
            let sl = ro.as_slice()?;
            if sl.len() != n_channels {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "E_rev_arr length {} != n_channels {}",
                    sl.len(),
                    n_channels
                )));
            }
            let rev_chans: &[HHParamIdx] = if self.hh_use_pacemaking {
                &[HHParamIdx::E_Na, HHParamIdx::E_K, HHParamIdx::E_CaT]
            } else {
                &[HHParamIdx::E_Na, HHParamIdx::E_K]
            };
            for (ch_i, &val) in sl.iter().enumerate() {
                hh_pop.fill_param(rev_chans[ch_i].into(), val);
            }
        }

        // g_leak: per-neuron, length must match n_neurons
        {
            let ro = g_leak.readonly();
            let sl = ro.as_slice()?;
            let m = sl.len().min(n);
            let p = hh_pop.param_slice_mut(HHParamIdx::G_Leak.into());
            for i in 0..m {
                p[i] = sl[i];
            }
        }

        // E_leak: per-neuron, length must match n_neurons
        {
            let ro = E_leak.readonly();
            let sl = ro.as_slice()?;
            let m = sl.len().min(n);
            let p = hh_pop.param_slice_mut(HHParamIdx::E_Leak.into());
            for i in 0..m {
                p[i] = sl[i];
            }
        }

        // Update edge-detection flag to match new v → avoid spike glitch after param change
        {
            let v: Vec<f32> = hh_pop.state_slice(0)[..n].to_vec();
            let thresh: Vec<f32> = hh_pop.param_slice(HHParamIdx::V_Thresh.into())[..n].to_vec();
            let p = hh_pop.state_slice_mut(5); // was_below
            for i in 0..n {
                p[i] = if v[i] < thresh[i] { 1.0 } else { 0.0 };
            }
        }

        Ok(())
    }

    /// Get membrane capacitances — read-only snapshot.
    /// For HH: returns C_m = 1/inv_cm (per-neuron, derived from pillars param).
    /// For LIF/Izhikevich: returns state's internal C_m (default 1.0).
    pub fn C_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        if let Some((_, pop)) = &self.hh_pop {
            let inv = pop.param_slice(HHParamIdx::Inv_Cm.into());
            inv[..self.n_neurons]
                .iter()
                .map(|&v| 1.0 / v)
                .collect::<Vec<f32>>()
                .into_pyarray(py)
        } else {
            // Non-HH: C_m is not stored per-neuron, return defaults
            vec![1.0; self.n_neurons].into_pyarray(py)
        }
    }

    /// Set membrane capacitances (per-neuron).
    /// For HH: stores inv_cm = 1/C_m per-neuron. Affects dv/dt = (I - I_ion) / C_m.
    /// For non-HH: no effect (LIF/Izhikevich use params, not per-neuron C_m).
    pub fn set_C_m(&mut self, values: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = values.readonly();
        let sl = ro.as_slice()?;
        if let Some((_, pop)) = &mut self.hh_pop {
            let n = sl.len().min(self.n_neurons);
            let p = pop.param_slice_mut(HHParamIdx::Inv_Cm.into());
            for i in 0..n {
                p[i] = 1.0 / sl[i];
            }
        }
        // Non-HH: silently ignored (LIF/Izhikevich have fixed params, not per-neuron C_m)
        Ok(())
    }

    /// Current input currents (Ii) — read-only snapshot.
    /// For HH: returns last external input buffer (pillars keeps no persistent Ii).
    /// For others: returns state.I_buf (accumulated synaptic + external).
    pub fn Ii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        if self.hh_pop.is_some() {
            self.hh_i_in[..self.n_neurons].to_vec().into_pyarray(py)
        } else {
            self.state.I_buf[..self.n_neurons].to_vec().into_pyarray(py)
        }
    }

    /// Run GCO (Graph-Coupled Oscillator) with time-varying external input, record (v, u) trajectory.
    ///
    /// Args:
    ///     connectome: ``ConnectomeData`` — CSR graph with NT types.
    ///     external_I: ``numpy.ndarray`` shape ``(n_changes, n_neurons)`` — external input per change.
    ///     steps_per_input: GCO steps per external input.
    ///     dt_ms: timestep in milliseconds.
    ///     tau: phase time scale for GCO.
    ///     record_every: record state every N steps (default 0 = record at end of each input block).
    ///
    /// Returns:
    ///     ``(v_traj, u_traj)`` — each ``(n_records * n_neurons,)`` flat float32 array.
    ///     Reshape with ``.reshape(-1, n_neurons)`` in Python.
    #[pyo3(signature = (connectome, external_I, steps_per_input, dt_ms=0.5, tau=1.0, record_every=0))]
    pub fn gco_record<'py>(
        &mut self,
        py: Python<'py>,
        connectome: &PyConnectomeData,
        external_I: Bound<'_, PyArray2<f32>>,
        steps_per_input: usize,
        dt_ms: f32,
        #[allow(unused_variables)] tau: f32,
        record_every: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let n_neurons = self.n_neurons;
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
        let (v_traj, u_traj) = conn::sim_gco_record(
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
            conn_data,
            &mut self.state,
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
            conn_data,
            &mut self.state,
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
}

// ─── BrainRunner ──────────────────────────────────

/// Universal brain runner — ONE Rust call replaces the slow Python loop.
///
/// Supports all neuron models: ``"lif"``, ``"lif_synaptic"``, ``"izh"``,
/// ``"izh_synaptic"``, ``"izh_dual"``, ``"izh_eprop"``, ``"hh"``,
/// ``"hh_synaptic"``, ``"rate"``, ``"gco"``.
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
    #[pyo3(signature = (connectome, model="lif".to_string(), dt_ms=0.5, tau_ref=2.0, tau_m=20.0, v_rest=-70.0, v_thresh=-58.0, v_reset=-70.0, pacemaking=false))]
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
    ) -> PyResult<Self> {
        let config = match model.as_str() {
            "lif" => conn::BrainConfig::lif(dt_ms),
            "lif_synaptic" => conn::BrainConfig::lif_synaptic(dt_ms),
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
            "rate" => conn::BrainConfig::rate(dt_ms, tau_m),
            "gco" => conn::BrainConfig::gco(dt_ms),
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown model '{other}'. Use: lif, lif_synaptic, izh, izh_synaptic, izh_dual, izh_eprop, hh, hh_synaptic, rate, gco"
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
