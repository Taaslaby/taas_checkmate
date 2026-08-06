//! PyO3 bindings for Karmac — Carmack-style BSP LOD neural engine & GNW workspace.

#![allow(non_snake_case)]
#![allow(deprecated)]

use crate::karmac::CarmackBsp;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Carmack BSP Level-of-Detail (LOD) Manager for neural populations.
///
/// Dynamically assigns simulation resolution based on wavefront distance:
/// - LOD 0 (HhLut): active wavefront (0-10 µm) — 64x cost
/// - LOD 1 (Izhikevich): near wavefront (10-50 µm) — 8x cost
/// - LOD 2 (Lif): quiet zone (50-200 µm) — 2x cost
/// - LOD 3 (Tanh): background (>200 µm) — 1x cost
#[pyclass(name = "CarmackBsp", module = "taas_checkmate.karmac")]
pub struct PyCarmackBsp {
    inner: CarmackBsp,
}

#[pymethods]
impl PyCarmackBsp {
    /// Create Carmack BSP LOD manager from neuron 3D coordinates.
    ///
    /// Args:
    ///     positions: Nx3 numpy array of float32/float64 (x, y, z) in µm
    #[new]
    pub fn new(positions: Bound<'_, PyArray1<f32>>) -> PyResult<Self> {
        let ro = positions.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("positions: {e}")))?;
        if slice.len() % 3 != 0 {
            return Err(PyValueError::new_err(
                "positions length must be a multiple of 3 (x, y, z per neuron)",
            ));
        }
        let n = slice.len() / 3;
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            coords.push((slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]));
        }

        let inner = CarmackBsp::new(&coords);
        Ok(PyCarmackBsp { inner })
    }

    /// Set distance thresholds for LOD levels (in µm).
    ///
    /// Args:
    ///     lod0: radius for LOD 0 HH (default 10.0 µm)
    ///     lod1: radius for LOD 1 Izhikevich (default 50.0 µm)
    ///     lod2: radius for LOD 2 LIF (default 200.0 µm)
    pub fn set_distances(&mut self, lod0: f32, lod1: f32, lod2: f32) {
        self.inner.set_distances(lod0, lod1, lod2);
    }

    /// Set hold counters before demotion (in simulation steps).
    pub fn set_hold_steps(&mut self, hh: u8, izh: u8, lif: u8, tanh: u8) {
        self.inner.set_hold_steps(hh, izh, lif, tanh);
    }

    /// Set stride for LOD updates (e.g. 10 = update Octree LOD once every 10 steps).
    pub fn set_update_stride(&mut self, stride: u64) {
        self.inner.set_update_stride(stride);
    }

    /// Set activity thresholds for quiet/sleep demotion (in steps since last spike).
    pub fn set_activity_thresholds(&mut self, quiet: u16, sleep: u16) {
        self.inner.set_activity_thresholds(quiet, sleep);
    }

    /// Update LOD levels based on active spike indices.
    ///
    /// Args:
    ///     spike_indices: numpy array of neuron IDs that fired in the last step
    pub fn update_lod(&mut self, spike_indices: Bound<'_, PyArray1<usize>>) -> PyResult<()> {
        let ro = spike_indices.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("spike_indices: {e}")))?;
        self.inner.update_lod(slice);
        Ok(())
    }

    /// Update LOD using pure Activity-Based Sliding Window algorithm.
    ///
    /// Args:
    ///     spike_indices: numpy array of neuron IDs that fired in the last step
    ///     hh_window: max inactive steps for HH LOD 0 (default 20)
    ///     izh_window: max inactive steps for Izhikevich LOD 1 (default 100)
    ///     lif_window: max inactive steps for LIF LOD 2 (default 200)
    pub fn update_activity_lod(
        &mut self,
        spike_indices: Bound<'_, PyArray1<usize>>,
        hh_window: Option<u16>,
        izh_window: Option<u16>,
        lif_window: Option<u16>,
    ) -> PyResult<()> {
        let ro = spike_indices.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("spike_indices: {e}")))?;
        self.inner.update_activity_lod(
            slice,
            hh_window.unwrap_or(20),
            izh_window.unwrap_or(100),
            lif_window.unwrap_or(200),
        );
        Ok(())
    }

    /// Update LOD using Dynamic Firing-Rate Spike-Triggered algorithm.
    ///
    /// Args:
    ///     spike_indices: numpy array of neuron IDs that fired in the last step
    ///     min_spikes_hh: min spikes required for promotion to HH (default 2)
    ///     window_steps: sliding window size in steps (default 50)
    pub fn update_firing_rate_lod(
        &mut self,
        spike_indices: Bound<'_, PyArray1<usize>>,
        min_spikes_hh: Option<u16>,
        window_steps: Option<u16>,
    ) -> PyResult<()> {
        let ro = spike_indices.readonly();
        let slice = ro
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("spike_indices: {e}")))?;
        self.inner.update_firing_rate_lod(
            slice,
            min_spikes_hh.unwrap_or(2),
            window_steps.unwrap_or(50),
        );
        Ok(())
    }

    /// Get current LOD levels for all neurons as uint8 array [0=HH, 1=Izh, 2=LIF, 3=Tanh].
    pub fn get_lod_levels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let vec: Vec<u8> = self.inner.lod.iter().map(|&l| l as u8).collect();
        PyArray1::from_vec(py, vec)
    }

    /// Get count of neurons at each LOD level: [n_hh, n_izh, n_lif, n_tanh].
    pub fn get_lod_counts(&self) -> (usize, usize, usize, usize) {
        let counts = self.inner.lod_counts();
        (counts[0], counts[1], counts[2], counts[3])
    }

    /// Estimate speedup factor compared to running 100% HH.
    pub fn speedup_factor(&self) -> f32 {
        self.inner.speedup_factor()
    }
}

/// PyMinBrain: Minimal Brain Model (PFC, GNW, Basal Ganglia)
#[pyclass(name = "MinBrain", module = "taas_checkmate.karmac")]
pub struct PyMinBrain {
    inner: crate::karmac::min_brain::MinBrain,
}

#[pymethods]
impl PyMinBrain {
    #[new]
    #[pyo3(signature = (sensory_dim=128, workspace_size=64, pfc_slots=4, n_actions=8, clear_threshold=0.8))]
    pub fn new(
        sensory_dim: usize,
        workspace_size: usize,
        pfc_slots: usize,
        n_actions: usize,
        clear_threshold: f32,
    ) -> Self {
        let config = crate::karmac::min_brain::MinBrainConfig {
            sensory_dim,
            workspace_size,
            pfc_slots,
            n_actions,
            clear_threshold,
        };
        Self {
            inner: crate::karmac::min_brain::MinBrain::new(config),
        }
    }

    pub fn step(&mut self, sensory_input: Bound<'_, PyArray1<f32>>) -> PyResult<usize> {
        let slice = sensory_input.readonly();
        let s = slice.as_slice().map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(self.inner.step(s))
    }

    pub fn get_pfc_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.pfc_state())
    }
}

// ═══════════════════════════════════════════════════════════════
//  BasalGanglia — Action Selection + Dopamine R-STDP Plasticity
// ═══════════════════════════════════════════════════════════════

/// Basal Ganglia winner-take-all action selection with online Dopamine R-STDP.
///
/// Creates a cortical → striatum → GPi/SNr → thalamus → action pathway.
///
/// Usage:
/// ```python
/// from taas_checkmate.karmac import BasalGanglia
///
/// bg = BasalGanglia(n_cortical=8, n_actions=4)
/// cortex = np.array([0.1, 0.9, 0.3, 0.2, 0.05, 0.1, 0.0, 0.0], dtype=np.float32)
/// action = bg.step(cortex)      # winner-take-all action index
///
/// # Online Dopamine R-STDP weight update
/// bg.update_weights_stdp(dopamine=1.0, lr=0.1, target_action=1)
/// ```
#[pyclass(name = "BasalGanglia", module = "taas_checkmate.karmac")]
pub struct PyBasalGanglia {
    inner: crate::karmac::basal_ganglia::BasalGanglia,
}

#[pymethods]
impl PyBasalGanglia {
    /// Create Basal Ganglia with `n_cortical` input channels and `n_actions` competing actions.
    #[new]
    pub fn new(n_cortical: usize, n_actions: usize) -> Self {
        Self {
            inner: crate::karmac::basal_ganglia::BasalGanglia::new(n_cortical, n_actions),
        }
    }

    /// Run one step of winner-take-all action selection.
    ///
    /// Args:
    ///     cortex_input: float32 numpy array of cortical drive values (length n_cortical)
    ///
    /// Returns:
    ///     int: selected action index (0..n_actions-1)
    pub fn step(&mut self, cortex_input: Bound<'_, PyArray1<f32>>) -> PyResult<usize> {
        let ro = cortex_input.readonly();
        let slice = ro.as_slice().map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(self.inner.step(slice))
    }

    /// 3-Factor Neuromodulated Dopamine R-STDP weight update.
    ///
    /// Reinforces the target action row with +dopamine and suppresses competitors.
    /// ΔW_ij = η · dopamine · state_j (target row)
    /// ΔW_ij = -0.8 · η · state_j    (competing rows)
    ///
    /// Args:
    ///     dopamine: dopamine signal (+1.0 reward, -0.5 punishment)
    ///     lr: learning rate (e.g. 0.1)
    ///     target_action: action index to reinforce
    pub fn update_weights_stdp(&mut self, dopamine: f32, lr: f32, target_action: usize) {
        self.inner.update_weights_stdp(dopamine, lr, target_action);
    }

    /// Get cortico-striatal weights as flat numpy array (n_actions x n_cortical, row-major).
    pub fn get_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.cortico_stratial)
    }

    /// Number of cortical input channels.
    #[getter]
    pub fn n_cortical(&self) -> usize { self.inner.config.n_cortical }

    /// Number of competing actions.
    #[getter]
    pub fn n_actions(&self) -> usize { self.inner.config.n_actions }

    fn __repr__(&self) -> String {
        format!(
            "BasalGanglia(n_cortical={}, n_actions={})",
            self.inner.config.n_cortical, self.inner.config.n_actions
        )
    }
}

// ═══════════════════════════════════════════════════════════════
//  GlobalWorkspace — GNW Attention Ignition
// ═══════════════════════════════════════════════════════════════

/// Global Neuronal Workspace (GNW) attention ignition module.
///
/// Implements Baars/Dehaene GNW with gate → softmax → broadcast residual:
///   output = r + broadcast(softmax(gate(r)))
///
/// Usage:
/// ```python
/// from taas_checkmate.karmac import GlobalWorkspace
///
/// gnw = GlobalWorkspace(n_neurons=512, workspace_size=64)
/// cortex = np.zeros(512, dtype=np.float32)
/// gnw.forward(cortex)  # in-place ignition update
/// ```
#[pyclass(name = "GlobalWorkspace", module = "taas_checkmate.karmac")]
pub struct PyGlobalWorkspace {
    inner: crate::karmac::gnw::GlobalWorkspace,
}

#[pymethods]
impl PyGlobalWorkspace {
    /// Create GNW with `n_neurons` and `workspace_size`.
    ///
    /// Args:
    ///     n_neurons: number of cortical neurons to gate
    ///     workspace_size: bottleneck workspace dimension
    #[new]
    pub fn new(n_neurons: usize, workspace_size: usize) -> Self {
        Self {
            inner: crate::karmac::gnw::GlobalWorkspace::new(n_neurons, workspace_size),
        }
    }

    /// Run GNW forward pass (in-place): gate → softmax → broadcast residual.
    ///
    /// Args:
    ///     cortex: float32 numpy array of neuron activations (length n_neurons, modified in-place)
    pub fn forward(&mut self, cortex: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let mut rw = cortex.readwrite();
        let slice = rw.as_slice_mut().map_err(|e| PyValueError::new_err(format!("{e}")))?;
        self.inner.forward(slice);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("GlobalWorkspace(n={}, ws={})", self.inner.n, self.inner.ws)
    }
}

