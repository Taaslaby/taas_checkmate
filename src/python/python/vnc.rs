//! PyO3 bindings for VNC MN→actuator mapping.
//!
//! Sub-module: `taas_checkmate.vnc`
//! Root-level: `taas_checkmate.vnc_step`

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pillars::kernels::vnc;
use pyo3::prelude::*;

/// VNC MN→actuator step: вычисляет actuator drive из MN rates (Rust).
///
/// # Parameters
///
/// * `mn_rates` — MN firing rates [n_mn], float32
/// * `balance_gain` — per-leg balance gain [6], float64
/// * `contact_boost` — per-leg contact boost [6], float64
/// * `actuator_gain` — overall actuator gain, float64
/// * `map_mn_idx` — MN index for each mapping entry [n_mapped], int32
/// * `map_act_idx` — actuator index for each mapping entry [n_mapped], int32
/// * `map_scale` — scale for each mapping entry [n_mapped], float32
/// * `map_leg_idx` — leg index for each mapping entry [n_mapped], int32
/// * `mirror_left` — left actuator indices for symmetrization [n_mirror], int32
/// * `mirror_right` — right actuator indices for symmetrization [n_mirror], int32
/// * `ctrl_out` — output actuator control values [n_actuators], float64 (mutable)
#[pyfunction]
#[pyo3(name = "vnc_step")]
pub fn py_vnc_step(
    mn_rates: Bound<'_, PyArray1<f32>>,
    balance_gain: Bound<'_, PyArray1<f64>>,
    contact_boost: Bound<'_, PyArray1<f64>>,
    actuator_gain: f64,
    map_mn_idx: Bound<'_, PyArray1<i32>>,
    map_act_idx: Bound<'_, PyArray1<i32>>,
    map_scale: Bound<'_, PyArray1<f32>>,
    map_leg_idx: Bound<'_, PyArray1<i32>>,
    mirror_left: Bound<'_, PyArray1<i32>>,
    mirror_right: Bound<'_, PyArray1<i32>>,
    ctrl_out: Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    let mn_rates = mn_rates.readonly();
    let balance_gain = balance_gain.readonly();
    let contact_boost = contact_boost.readonly();
    let map_mn_idx = map_mn_idx.readonly();
    let map_act_idx = map_act_idx.readonly();
    let map_scale = map_scale.readonly();
    let map_leg_idx = map_leg_idx.readonly();
    let mirror_left = mirror_left.readonly();
    let mirror_right = mirror_right.readonly();
    let mut ctrl_out = ctrl_out.readwrite();

    vnc::vnc_step_mapping(
        mn_rates.as_slice()?,
        balance_gain.as_slice()?,
        contact_boost.as_slice()?,
        actuator_gain,
        ctrl_out.as_slice()?.len(),
        map_mn_idx.as_slice()?,
        map_act_idx.as_slice()?,
        map_scale.as_slice()?,
        map_leg_idx.as_slice()?,
        mirror_left.as_slice()?,
        mirror_right.as_slice()?,
        ctrl_out.as_slice_mut()?,
    );

    Ok(())
}
