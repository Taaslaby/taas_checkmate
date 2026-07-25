//! PyO3 bindings for E-Prop auto-tuning — FR probe + config suggestion.
//!
//! Exposes two functions:
//! - `probe_eprop_fr(...)` — returns numpy arrays (biases, frs, snrs)
//! - `suggest_eprop_config(...)` — returns dict with recommended hyperparameters

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::neuron::autotune;

/// Probe firing rates for a range of bias values.
///
/// Uses frozen random weights and input for all bias values so results
/// are directly comparable.
///
/// # Args
/// * `n_pre` — number of input neurons
/// * `n_post` — number of hidden neurons
/// * `biases` — 1D numpy array of bias values to probe
/// * `n_steps` — simulation steps per probe (default 50)
/// * `poisson_activity` — fraction of active inputs per step (default 0.2)
/// * `weight_scale` — weight init scale U[-ws, ws] (default 10/sqrt(n_pre))
/// * `seed` — random seed (default 42)
///
/// # Returns
/// Tuple of 3 numpy arrays: (biases_out, firing_rates, snrs)
#[pyfunction]
#[pyo3(signature = (n_pre, n_post, biases, n_steps=50, poisson_activity=0.2, weight_scale=None, seed=42))]
pub fn probe_eprop_fr<'py>(
    py: Python<'py>,
    n_pre: usize,
    n_post: usize,
    biases: Bound<'_, PyArray1<f32>>,
    n_steps: usize,
    poisson_activity: f32,
    weight_scale: Option<f32>,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let biases_slice = biases.readonly();
    let biases_data = biases_slice.as_slice()?;

    let ws = weight_scale.unwrap_or_else(|| 10.0 / (n_pre as f32).sqrt());

    let points = autotune::probe_fr(
        n_pre,
        n_post,
        biases_data,
        n_steps,
        poisson_activity,
        ws,
        seed,
    );

    let n = points.len();
    // PyArray1::new is unsafe in numpy 0.23
    let biases_out = unsafe { PyArray1::new(py, [n], false) };
    let frs_out = unsafe { PyArray1::new(py, [n], false) };
    let snrs_out = unsafe { PyArray1::new(py, [n], false) };

    {
        let mut bm = biases_out.readwrite();
        let mut fm = frs_out.readwrite();
        let mut sm = snrs_out.readwrite();

        let b_slice = bm.as_slice_mut()?;
        let f_slice = fm.as_slice_mut()?;
        let s_slice = sm.as_slice_mut()?;

        for (i, p) in points.iter().enumerate() {
            b_slice[i] = p.bias;
            f_slice[i] = p.avg_fr;
            s_slice[i] = p.snr;
        }
    }

    Ok((biases_out, frs_out, snrs_out))
}

/// Suggest complete E-Prop Izhikevich Multi configuration.
///
/// Runs an automatic FR probe and selects the best bias for the
/// target firing rate, then computes matching hyperparameters.
///
/// # Args
/// * `n_pre` — number of input neurons
/// * `n_post` — number of hidden neurons
/// * `poisson_activity` — fraction of active inputs per step (default 0.2)
/// * `target_fr` — target firing rate (default 0.1)
/// * `seed` — random seed (default 42)
///
/// # Returns
/// Dict with keys: bias, weight_scale, lr_in, lr_out, d, tau_surrogate,
/// tau_e, n_steps, expected_fr, snr
#[pyfunction]
#[pyo3(signature = (n_pre, n_post, poisson_activity=0.2, target_fr=0.1, seed=42))]
pub fn suggest_eprop_config<'py>(
    py: Python<'py>,
    n_pre: usize,
    n_post: usize,
    poisson_activity: f32,
    target_fr: f32,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = autotune::suggest_config(n_pre, n_post, poisson_activity, target_fr, seed);

    let dict = PyDict::new(py);
    dict.set_item("bias", cfg.bias)?;
    dict.set_item("weight_scale", cfg.weight_scale)?;
    dict.set_item("lr_in", cfg.lr_in)?;
    dict.set_item("lr_out", cfg.lr_out)?;
    dict.set_item("d", cfg.d)?;
    dict.set_item("tau_surrogate", cfg.tau_surrogate)?;
    dict.set_item("tau_e", cfg.tau_e)?;
    dict.set_item("n_steps", cfg.n_steps as i32)?;
    dict.set_item("expected_fr", cfg.expected_fr)?;
    dict.set_item("snr", cfg.snr)?;
    Ok(dict)
}
