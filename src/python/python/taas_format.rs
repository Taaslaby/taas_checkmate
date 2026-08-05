//! PyO3 bindings for .taas format — save/load simulation state.

#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3::IntoPyObjectExt;

use crate::taas_format;

use super::connectome_brain::PyConnectomeData;
use super::rl_agent::PyActorCriticAgent;
use super::t3::{PyKalmanDecoder, PyNlmsDecoder, PyTriOscillator};

/// Save simulation state to .taas file.
#[pyfunction]
#[pyo3(name = "save_taas", signature = (path, tri=None, nlms=None, kalman=None, rl_agent=None, metadata=None))]
pub fn py_save_taas(
    path: &str,
    tri: Option<&PyTriOscillator>,
    nlms: Option<&PyNlmsDecoder>,
    kalman: Option<&PyKalmanDecoder>,
    rl_agent: Option<&PyActorCriticAgent>,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();

    if let Some(t) = tri {
        taas_format::save_trioscilator(&t.inner, &mut writer);
    }
    if let Some(d) = nlms {
        taas_format::save_nlms_decoder(&d.inner, &mut writer);
    }
    if let Some(d) = kalman {
        taas_format::save_kalman_decoder(&d.inner, &mut writer);
    }
    if let Some(agent) = rl_agent {
        taas_format::save_actor_critic_agent(&agent.inner, &mut writer);
    }

    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    writer
        .save_to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Load simulation state from .taas file.
#[pyfunction]
#[pyo3(name = "load_taas")]
pub fn py_load_taas(path: &str, py: Python<'_>) -> PyResult<PyObject> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    let dict = PyDict::new(py);

    // TriOscillator detection: try new grouped format first, fall back to old per-field
    if reader.find_component("tri_state").is_ok() || reader.find_component("th1").is_ok() {
        // Extract n from the metadata or from the grouped tensor shape
        let n = if let Ok(entry) = reader.find_component("tri_state") {
            entry.shape.get(1).copied().unwrap_or(0)
        } else if let Ok(data) = reader.get_f32("th1") {
            data.len()
        } else {
            0
        };
        if n > 0 {
            if let Ok(tri) = taas_format::load_trioscilator(&reader, n) {
                dict.set_item("tri", PyTriOscillator::build(tri))?;
            }
        }
    }

    if let Ok(data) = reader.get_f32("nlms_weights") {
        let n = data.len();
        if n > 0 {
            if let Ok(dec) = taas_format::load_nlms_decoder(&reader, n) {
                dict.set_item("nlms", PyNlmsDecoder { inner: dec })?;
            }
        }
    }

    if let Ok(data) = reader.get_f32("kalman_weights") {
        let n = data.len();
        if n > 0 {
            if let Ok(dec) = taas_format::load_kalman_decoder(&reader, n) {
                dict.set_item("kalman", PyKalmanDecoder { inner: dec })?;
            }
        }
    }

    if reader.find_component("rl_w_ih").is_ok() {
        if let Ok(agent) = taas_format::load_actor_critic_agent(&reader) {
            dict.set_item("rl_agent", PyActorCriticAgent { inner: agent })?;
        }
    }

    // Metadata
    let meta = PyDict::new(py);
    meta.set_item("version", reader.manifest.version.as_str())?;
    meta.set_item("created", reader.manifest.created.as_str())?;
    meta.set_item("n_components", reader.manifest.components.len())?;
    for (k, v) in &reader.manifest.metadata {
        meta.set_item(k.as_str(), v.as_str())?;
    }
    dict.set_item("_meta", meta)?;

    // LIF-AT state
    if reader.find_component("lif_v").is_ok() {
        if let Ok((v, theta)) = taas_format::load_lif_at_state(&reader) {
            dict.set_item("lif_v", v)?;
            dict.set_item("lif_theta", theta)?;
        }
    }

    // STDP weights
    if reader.find_component("stdp_weights").is_ok() {
        if let Ok(w) = taas_format::load_stdp_weights(&reader) {
            dict.set_item("stdp_weights", w)?;
        }
    }

    // CSR matrix
    if reader.find_component("csr_row_ptr").is_ok() {
        if let Ok((row_ptr, col_indices, values)) = taas_format::load_csr(&reader) {
            dict.set_item("csr_row_ptr", row_ptr)?;
            dict.set_item("csr_col_indices", col_indices)?;
            dict.set_item("csr_values", values)?;
        }
    }

    // Bridge weights + decoder (generic tensors, stored as flat f32 lists)
    if let Ok(w) = reader.get_f32("bridge_w") {
        dict.set_item("bridge_w", w)?;
    }
    if let Ok(b) = reader.get_f32("bridge_b") {
        dict.set_item("bridge_b", b)?;
    }
    if let Ok(w) = reader.get_f32("decoder_w") {
        dict.set_item("decoder_w", w)?;
    }
    if let Ok(b) = reader.get_f32("decoder_b") {
        dict.set_item("decoder_b", b)?;
    }

    Ok(dict.into())
}

/// Save connectome (ConnectomeData) to .taas file.
#[pyfunction]
#[pyo3(name = "save_connectome", signature = (path, connectome, metadata=None))]
pub fn py_save_connectome(
    path: &str,
    connectome: &PyConnectomeData,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    taas_format::save_connectome(&connectome.inner, &mut writer);
    writer
        .save_to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Load connectome (ConnectomeData) from .taas file.
#[pyfunction]
#[pyo3(name = "load_connectome")]
pub fn py_load_connectome(path: &str) -> PyResult<PyConnectomeData> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    let conn = taas_format::load_connectome(&reader)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
    // Reconstruct ConnectomeData with groups from metadata
    // load_connectome already restores groups from JSON metadata
    Ok(PyConnectomeData { inner: conn })
}

/// Save full model (connectome + bridge + decoder) to a single .taas file.
#[pyfunction]
#[pyo3(name = "save_model", signature = (path, connectome, bridge_weights, bridge_bias, decoder_weights, decoder_bias, metadata=None))]
pub fn py_save_model(
    path: &str,
    connectome: &super::connectome_brain::PyConnectomeData,
    bridge_weights: Vec<f32>,
    bridge_bias: Vec<f32>,
    decoder_weights: Vec<f32>,
    decoder_bias: Vec<f32>,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    if let Some(md) = metadata {
        for (k, v) in md {
            writer.add_metadata(&k, &v);
        }
    }
    taas_format::save_connectome(&connectome.inner, &mut writer);
    writer.add_metadata("type", "Model");
    writer.add_f32("bridge_w", "Model", &bridge_weights);
    writer.add_f32("bridge_b", "Model", &bridge_bias);
    writer.add_f32("decoder_w", "Model", &decoder_weights);
    writer.add_f32("decoder_b", "Model", &decoder_bias);
    writer
        .save_to_file(path)
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

/// Inspect .taas file. Returns human-readable string.
#[pyfunction]
#[pyo3(name = "inspect_taas")]
pub fn py_inspect_taas(path: &str) -> PyResult<String> {
    taas_format::inspect_taas(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
}

// ═══════════════════════════════════════════════════════════
//  Lazy TaasReader — loads components on demand
// ═══════════════════════════════════════════════════════════

/// Lazy .taas file reader — loads components on demand.
///
/// Usage:
/// ```python
/// reader = TaasReader("model.taas")
/// print(reader.components)  # list of component names
/// data = reader.get("tri_state")  # loads only this component
/// ```
#[pyclass(name = "TaasReader")]
pub(crate) struct PyTaasReader {
    reader: crate::taas_format::TaasReader,
}

#[pymethods]
impl PyTaasReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = crate::taas_format::TaasReader::from_file(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>)?;
        Ok(Self { reader })
    }

    /// List all component names in the archive.
    #[getter]
    fn components(&self) -> Vec<String> {
        self.reader
            .list_components()
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }

    /// Number of components.
    #[getter]
    fn n_components(&self) -> usize {
        self.reader.list_components().len()
    }

    /// Total file size in bytes.
    #[getter]
    fn total_size(&self) -> usize {
        self.reader.total_size()
    }

    /// Metadata as a Python dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (k, v) in self.reader.metadata() {
            dict.set_item(k.as_str(), v.as_str())?;
        }
        dict.set_item("version", self.reader.manifest.version.as_str())?;
        dict.set_item("created", self.reader.manifest.created.as_str())?;
        Ok(dict.into())
    }

    /// Get a component's raw bytes as `bytes` (Python).
    /// Raises KeyError if component not found.
    fn get_raw(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        self.reader
            .find_component(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        let data = self
            .reader
            .get_u8(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;
        Ok(PyBytes::new(py, &data).into())
    }

    /// Get a component as a list of floats.
    /// Raises KeyError if not found, TypeError if wrong dtype.
    fn get_f32(&self, name: &str) -> PyResult<Vec<f32>> {
        self.reader.get_f32(name).map_err(|e| {
            if e.contains("not found") {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(e)
            } else if e.contains("is not f32") {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(e)
            } else {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
            }
        })
    }

    /// Get a component as a list of ints.
    fn get_i32(&self, name: &str) -> PyResult<Vec<i32>> {
        self.reader.get_i32(name).map_err(|e| {
            if e.contains("not found") {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(e)
            } else if e.contains("is not i32") {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(e)
            } else {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
            }
        })
    }

    /// Component info for a given name.
    /// Returns dict with keys: name, type, dtype, shape, size, compression.
    fn info(&self, name: &str, py: Python<'_>) -> PyResult<PyObject> {
        let entry = self
            .reader
            .find_component(name)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        let dict = PyDict::new(py);
        dict.set_item("name", entry.name.as_str())?;
        dict.set_item("type", entry.component_type.as_str())?;
        dict.set_item("dtype", entry.dtype.as_str())?;
        dict.set_item("shape", entry.shape.clone())?;
        dict.set_item("size", entry.size)?;
        dict.set_item("compression", entry.compression.as_deref())?;
        if let Some(us) = entry.uncompressed_size {
            dict.set_item("uncompressed_size", us)?;
        }
        Ok(dict.into())
    }

    /// Component info for all components.
    /// Returns list of dicts.
    fn list_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for comp in self.reader.list_components() {
            let dict = PyDict::new(py);
            dict.set_item("name", comp.name.as_str())?;
            dict.set_item("type", comp.component_type.as_str())?;
            dict.set_item("dtype", comp.dtype.as_str())?;
            dict.set_item("shape", comp.shape.clone())?;
            dict.set_item("size", comp.size)?;
            dict.set_item("compression", comp.compression.as_deref())?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Validate archive integrity (CRC32 checks).
    /// Raises IOError if validation fails.
    fn validate(&self) -> PyResult<()> {
        self.reader
            .validate()
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    /// Load the TriOscillator from this archive (if present).
    /// Returns None if not found.
    fn get_tri(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let has_tri = self.reader.find_component("tri_state").is_ok()
            || self.reader.find_component("th1").is_ok();
        if !has_tri {
            return Ok(None);
        }

        let n = if let Ok(entry) = self.reader.find_component("tri_state") {
            entry.shape.get(1).copied().unwrap_or(0)
        } else if let Ok(data) = self.reader.get_f32("th1") {
            data.len()
        } else {
            0
        };

        if n == 0 {
            return Ok(None);
        }

        match taas_format::load_trioscilator(&self.reader, n) {
            Ok(tri) => Ok(Some(PyTriOscillator::build(tri).into_py_any(py)?)),
            Err(_) => Ok(None),
        }
    }

    /// Load the connectome from this archive (if present).
    fn get_connectome(&self) -> PyResult<Option<PyConnectomeData>> {
        if self.reader.find_component("row_ptr").is_err() {
            return Ok(None);
        }
        match taas_format::load_connectome(&self.reader) {
            Ok(conn) => Ok(Some(PyConnectomeData { inner: conn })),
            Err(_) => Ok(None),
        }
    }

    /// Component names that match a type prefix.
    fn components_by_type(&self, type_prefix: &str) -> Vec<String> {
        self.reader
            .list_components()
            .iter()
            .filter(|c| c.component_type.starts_with(type_prefix))
            .map(|c| c.name.clone())
            .collect()
    }
}
