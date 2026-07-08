//! PyO3 bindings for .taas format — save/load simulation state.

#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::taas_format;

use super::t3::{PyTriOscillator, PyNlmsDecoder, PyKalmanDecoder};
use super::connectome_brain::PyConnectomeData;
use super::snn::PySornNetwork;
use super::eprop::PyEPropLIF;
use super::poisson_stp::PyShortTermPlasticity;
use super::rl_agent::PyActorCriticAgent;

/// Save simulation state to .taas file.
#[pyfunction]
#[pyo3(name = "save_taas", signature = (path, tri=None, nlms=None, kalman=None, sorn=None, eprop=None, stp=None, rl_agent=None, metadata=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_save_taas(
    path: &str,
    tri: Option<&PyTriOscillator>,
    nlms: Option<&PyNlmsDecoder>,
    kalman: Option<&PyKalmanDecoder>,
    sorn: Option<&PySornNetwork>,
    eprop: Option<&PyEPropLIF>,
    stp: Option<&PyShortTermPlasticity>,
    rl_agent: Option<&PyActorCriticAgent>,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();

    if let Some(t) = tri { taas_format::save_trioscilator(&t.inner, &mut writer); }
    if let Some(d) = nlms { taas_format::save_nlms_decoder(&d.inner, &mut writer); }
    if let Some(d) = kalman { taas_format::save_kalman_decoder(&d.inner, &mut writer); }
    if let Some(s) = sorn { taas_format::save_sorn_network(&s.inner, &mut writer); }
    if let Some(e) = eprop { taas_format::save_eprop_lif(&e.inner, &mut writer); }
    if let Some(s) = stp { taas_format::save_stp(&s.inner, &mut writer); }
    if let Some(agent) = rl_agent { taas_format::save_actor_critic_agent(&agent.inner, &mut writer); }

    if let Some(md) = metadata { for (k, v) in md { writer.add_metadata(&k, &v); } }
    writer.save_to_file(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

/// Load simulation state from .taas file.
#[pyfunction]
#[pyo3(name = "load_taas")]
pub fn py_load_taas(path: &str, py: Python<'_>) -> PyResult<PyObject> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    let dict = PyDict::new(py);

    if let Ok(data) = reader.get_f32("th1") {
        let n = data.len();
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

    if reader.find_component("sorn_weights").is_ok() {
        if let Ok(theta) = reader.get_f32("sorn_theta") {
            let n = theta.len();
            if n > 0 {
                if let Ok(sorn) = taas_format::load_sorn_network(&reader, n) {
                    dict.set_item("sorn", PySornNetwork { inner: sorn })?;
                }
            }
        }
    }

    if let Ok(data) = reader.get_f32("eprop_v") {
        let n = data.len();
        if n > 0 {
            if let Ok(eprop) = taas_format::load_eprop_lif(&reader, n) {
                dict.set_item("eprop", PyEPropLIF { inner: eprop })?;
            }
        }
    }

    if let Ok(data) = reader.get_f32("stp_x") {
        let n = data.len();
        if n > 0 {
            if let Ok(stp) = taas_format::load_stp(&reader, n) {
                dict.set_item("stp", PyShortTermPlasticity { inner: stp })?;
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
    for (k, v) in &reader.manifest.metadata { meta.set_item(k.as_str(), v.as_str())?; }
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
#[pyo3(name = "save_connectome")]
pub fn py_save_connectome(
    path: &str,
    connectome: &PyConnectomeData,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let mut writer = taas_format::TaasWriter::new();
    if let Some(md) = metadata { for (k, v) in md { writer.add_metadata(&k, &v); } }
    taas_format::save_connectome(&connectome.inner, &mut writer);
    writer.save_to_file(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

/// Load connectome (ConnectomeData) from .taas file.
#[pyfunction]
#[pyo3(name = "load_connectome")]
pub fn py_load_connectome(path: &str) -> PyResult<PyConnectomeData> {
    let reader = taas_format::TaasReader::from_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    let conn = taas_format::load_connectome(&reader)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    // Reconstruct ConnectomeData with groups from metadata
    // load_connectome already restores groups from JSON metadata
    Ok(PyConnectomeData { inner: conn })
}

/// Save full model (connectome + bridge + decoder) to a single .taas file.
#[pyfunction]
#[pyo3(name = "save_model")]
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
    if let Some(md) = metadata { for (k, v) in md { writer.add_metadata(&k, &v); } }
    taas_format::save_connectome(&connectome.inner, &mut writer);
    writer.add_metadata("type", "Model");
    writer.add_f32("bridge_w", "Model", &bridge_weights);
    writer.add_f32("bridge_b", "Model", &bridge_bias);
    writer.add_f32("decoder_w", "Model", &decoder_weights);
    writer.add_f32("decoder_b", "Model", &decoder_bias);
    writer.save_to_file(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}

/// Inspect .taas file. Returns human-readable string.
#[pyfunction]
#[pyo3(name = "inspect_taas")]
pub fn py_inspect_taas(path: &str) -> PyResult<String> {
    taas_format::inspect_taas(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
}
