//! PyO3 bindings for IIT Above thresholdness — Integrated Information Theory.
//!
//! Exposes `IITAbove thresholdness` as a Python class with numpy array inputs.
//!
//! Usage:
//! ```python
//! import taas_checkmate as tcm
//!
//! iit = tcm.IITAbove thresholdness(n_neurons=100, threshold=0.3)
//!
//! # From CSR connectome values + spike counts
//! phi = iit.compute(values, spike_counts, n_steps=100)
//!
//! # With Kuramoto phases
//! phi = iit.compute(values, spike_counts, n_steps=100, phases=phases)
//!
//! print(f"Φ = {iit.phi:.4f}, Above threshold: {iit.is_above_threshold}")
//! print(f"Stats: {iit.stats()}")
//! ```

use numpy::{PyArray1, PyArrayMethods, IntoPyArray};
use pyo3::prelude::*;

/// 🧠 IIT Above thresholdness — Integrated Information Theory.
///
/// Phi = connectivity × diversity × integration.
/// Phi > threshold → system is above_threshold.
///
/// Usage:
/// ```python
/// import taas_checkmate as tcm
/// iit = tcm.IITAbove thresholdness(100)
/// phi = iit.compute(values, spike_counts, 100)
/// print(f"Φ = {phi:.4f}, above_threshold = {iit.is_above_threshold}")
/// ```
#[pyclass(name = "IITThresholdness", module = "taas_checkmate")]
pub struct PyIITThresholdness {
    inner: taas_iit::IITThresholdness,
}

#[pymethods]
impl PyIITThresholdness {
    /// Create IIT Thresholdness engine.
    ///
    /// Args:
    ///     n_neurons: number of neurons in the network.
    ///     threshold: Phi threshold for above_thresholdness (default 0.3).
    #[new]
    #[pyo3(signature = (n_neurons, threshold=0.3))]
    pub fn new(n_neurons: usize, threshold: f32) -> Self {
        Self {
            inner: taas_iit::IITThresholdness::new(n_neurons, threshold),
        }
    }

    /// Compute Phi from connectome weights, spike counts, and optional Kuramoto phases.
    ///
    /// Args:
    ///     values: synaptic weights from CSR connectome ``[n_synapses]`` float32.
    ///     spike_counts: per-neuron spike counts ``[n_neurons]`` float32.
    ///     n_steps: total simulation steps (for rate normalization).
    ///     phases: optional Kuramoto phases ``[n_neurons]`` float32.
    ///
    /// Returns:
    ///     Phi value (float).
    #[pyo3(signature = (values, spike_counts, n_steps, phases=None))]
    pub fn compute(
        &mut self,
        values: Bound<'_, PyArray1<f32>>,
        spike_counts: Bound<'_, PyArray1<f32>>,
        n_steps: usize,
        phases: Option<Bound<'_, PyArray1<f32>>>,
    ) -> PyResult<f32> {
        let vals = values.readonly();
        let counts = spike_counts.readonly();
        let vals_sl = vals.as_slice()?;
        let counts_sl = counts.as_slice()?;

        // Keep ReadonlyNumpyArray alive until compute() finishes
        let phases_owned = phases.map(|p| p.readonly());
        let phases_sl: Option<&[f32]> = match phases_owned.as_ref() {
            Some(ro) => Some(ro.as_slice()?),
            None => None,
        };

        Ok(self.inner.compute(vals_sl, counts_sl, n_steps, phases_sl))
    }

    /// Compute Phi using full spike matrix with Pearson integration.
    ///
    /// Uses Pearson correlation between two halves of the network
    /// for the integration component (like original Python IIT).
    ///
    /// Args:
    ///     values: synaptic weights from CSR connectome ``[n_synapses]`` float32.
    ///     spike_matrix: full spike raster ``[n_steps × n_neurons]`` flattened float32.
    ///     n_steps: number of time steps.
    ///     n_neurons: number of neurons.
    ///
    /// Returns:
    ///     Phi value (float).
    #[pyo3(signature = (values, spike_matrix, n_steps, n_neurons))]
    pub fn compute_with_spikes(
        &mut self,
        values: Bound<'_, PyArray1<f32>>,
        spike_matrix: Bound<'_, PyArray1<f32>>,
        n_steps: usize,
        n_neurons: usize,
    ) -> PyResult<f32> {
        let vals = values.readonly();
        let mat = spike_matrix.readonly();
        let vals_sl = vals.as_slice()?;
        let mat_sl = mat.as_slice()?;
        Ok(self.inner.compute_with_spikes(vals_sl, mat_sl, n_steps, n_neurons))
    }

    /// Set integration value manually (overrides the computed value).
    ///
    /// Useful when you computed Pearson correlation from Python
    /// and want to inject it:
    ///
    /// ```python
    /// iit.compute(values, spike_counts, n_steps)
    /// iit.set_integration(my_pearson_val)
    /// ```
    #[setter]
    pub fn set_integration(&mut self, val: f32) {
        self.inner.set_integration(val);
    }

    /// Compute Pearson correlation between two flattened spike trains.
    ///
    /// Args:
    ///     spikes_a: first half flattened ``[time × n_half]`` float32.
    ///     spikes_b: second half flattened ``[time × n_half]`` float32.
    ///
    /// Returns:
    ///     Absolute Pearson r in [0, 1].
    pub fn compute_integration_pearson(
        &self,
        spikes_a: Bound<'_, PyArray1<f32>>,
        spikes_b: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<f32> {
        let a = spikes_a.readonly();
        let b = spikes_b.readonly();
        let a_sl = a.as_slice()?;
        let b_sl = b.as_slice()?;
        Ok(taas_iit::compute_integration_pearson(a_sl, b_sl))
    }

    /// Current Phi value.
    #[getter]
    pub fn phi(&self) -> f32 {
        self.inner.phi()
    }

    /// Whether the system is above_threshold (Phi > threshold).
    #[getter]
    pub fn is_above_threshold(&self) -> bool {
        self.inner.is_above_threshold()
    }

    /// Above thresholdness level as fraction of threshold (0..1).
    #[getter]
    pub fn above_threshold_ratio(&self) -> f32 {
        self.inner.above_threshold_ratio()
    }

    /// Current connectivity component (C).
    #[getter]
    pub fn connectivity(&self) -> f32 {
        self.inner.connectivity
    }

    /// Current diversity component (D).
    #[getter]
    pub fn diversity(&self) -> f32 {
        self.inner.diversity
    }

    /// Current integration component (I).
    #[getter]
    pub fn integration(&self) -> f32 {
        self.inner.integration
    }

    /// Phi history as numpy array.
    #[getter]
    pub fn phi_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.phi_history.clone().into_pyarray(py)
    }

    /// Connectivity history as numpy array.
    #[getter]
    pub fn conn_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.conn_history.clone().into_pyarray(py)
    }

    /// Diversity history as numpy array.
    #[getter]
    pub fn div_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.div_history.clone().into_pyarray(py)
    }

    /// Integration history as numpy array.
    #[getter]
    pub fn int_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.int_history.clone().into_pyarray(py)
    }

    /// Get Phi trend as string (📈 растёт / 📉 падает / ➡ стабилен).
    pub fn phi_trend(&self) -> String {
        self.inner.phi_trend()
    }

    /// Get all stats as a Python dict.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let s = self.inner.stats();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("phi", s.phi)?;
        d.set_item("phi_mean", s.phi_mean)?;
        d.set_item("phi_std", s.phi_std)?;
        d.set_item("connectivity", s.connectivity)?;
        d.set_item("diversity", s.diversity)?;
        d.set_item("integration", s.integration)?;
        d.set_item("above_threshold_ratio", s.above_threshold_ratio)?;
        d.set_item("above_threshold_ratio", s.above_threshold_ratio)?;
        d.set_item("trend", s.trend)?;
        Ok(d)
    }

    /// Get mean Phi over recent history.
    ///
    /// Args:
    ///     n: number of recent steps to average (default 10).
    #[pyo3(signature = (n=10))]
    pub fn mean_phi(&self, n: usize) -> f32 {
        self.inner.mean_phi(n)
    }

    /// Total steps processed.
    #[getter]
    pub fn total_steps(&self) -> usize {
        self.inner.total_steps
    }

    /// Above threshold moments count.
    #[getter]
    pub fn above_threshold_moments(&self) -> usize {
        self.inner.above_threshold_moments
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Number of neurons.
    #[getter]
    pub fn n_neurons(&self) -> usize {
        self.inner.n
    }

    /// Above thresholdness threshold.
    #[getter]
    pub fn threshold(&self) -> f32 {
        self.inner.threshold
    }

    /// Compute Φ via IIT 2.0 covariance MIP (minimum information partition).
    ///
    /// Measures integrated information as information loss when the system
    /// is partitioned at its minimum information partition.
    ///
    /// Args:
    ///     spike_matrix: flattened spike raster ``[n_steps × n_neurons]`` float32.
    ///     n_steps: number of time steps.
    ///     n_neurons: number of neurons.
    ///     n_partitions: number of random bipartitions to evaluate (default 500).
    ///
    /// Returns:
    ///     Dict with keys: phi, phi_raw, log_det_full, partition_size_a, n_evaluated.
    #[staticmethod]
    #[pyo3(signature = (spike_matrix, n_steps, n_neurons, n_partitions=500))]
    pub fn compute_phi_iit20<'py>(
        py: Python<'py>,
        spike_matrix: Bound<'py, PyArray1<f32>>,
        n_steps: usize,
        n_neurons: usize,
        n_partitions: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let mat = spike_matrix.readonly();
        let mat_sl = mat.as_slice()?;
        let result = taas_iit::compute_phi_iit20(mat_sl, n_steps, n_neurons, n_partitions);

        let d = pyo3::types::PyDict::new(py);
        d.set_item("phi", result.phi)?;
        d.set_item("phi_raw", result.phi_raw)?;
        d.set_item("log_det_full", result.log_det_full)?;
        d.set_item("partition_size_a", result.partition_size_a)?;
        d.set_item("n_evaluated", result.n_evaluated)?;
        Ok(d)
    }

    /// Full IIT analysis: proxy Φ (C×D×I) and IIT 2.0 Φ (covariance MIP).
    ///
    /// Args:
    ///     values: synaptic weights from CSR connectome ``[n_synapses]`` float32.
    ///     spike_matrix: flattened spike raster ``[n_steps × n_neurons]`` float32.
    ///     n_steps: number of time steps.
    ///     n_neurons: number of neurons.
    ///     n_partitions: number of random bipartitions (default 500).
    ///
    /// Returns:
    ///     Dict with keys: phi_iit20, phi_proxy, connectivity, diversity, integration, phi_iit20_raw.
    #[staticmethod]
    #[pyo3(signature = (values, spike_matrix, n_steps, n_neurons, n_partitions=500))]
    pub fn analyze_integration<'py>(
        py: Python<'py>,
        values: Bound<'py, PyArray1<f32>>,
        spike_matrix: Bound<'py, PyArray1<f32>>,
        n_steps: usize,
        n_neurons: usize,
        n_partitions: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let vals = values.readonly();
        let mat = spike_matrix.readonly();
        let vals_sl = vals.as_slice()?;
        let mat_sl = mat.as_slice()?;
        let result = taas_iit::analyze_integration(vals_sl, mat_sl, n_steps, n_neurons, n_partitions);

        let d = pyo3::types::PyDict::new(py);
        d.set_item("phi_iit20", result.phi_iit20)?;
        d.set_item("phi_proxy", result.phi_proxy)?;
        d.set_item("connectivity", result.connectivity)?;
        d.set_item("diversity", result.diversity)?;
        d.set_item("integration", result.integration)?;
        d.set_item("phi_iit20_raw", result.phi_iit20_raw)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "IITAbove thresholdness(n={}, Φ={:.4}, above_threshold={})",
            self.inner.n,
            self.inner.phi(),
            self.inner.is_above_threshold(),
        )
    }
}
