//! PyO3 bindings for E-Prop — PyEPropLIF, surrogate_gradient, eprop_update.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pillars::api::neurons as neuron;

/// E-Prop LIF neuron — LIF с eligibility traces для градиентного обучения.
///
/// Использование:
///     eprop = EPropLIF(256, dt=1.0, tau_m=10.0, tau_e=20.0)
///     n_spikes = eprop.step(I, spikes_out, pre_spikes)
///     eprop.update_weights(weights, learning_signals, lr=0.01)
#[pyclass(name = "EPropLIF", module = "taas_checkmate.eprop")]
pub struct PyEPropLIF {
    pub inner: neuron::EPropLIF,
}

#[pymethods]
impl PyEPropLIF {
    #[new]
    #[pyo3(signature = (n, dt=1.0, tau_m=10.0, tau_e=20.0))]
    fn new(n: usize, dt: f32, tau_m: f32, tau_e: f32) -> Self {
        Self {
            inner: neuron::EPropLIF::new(n, dt, tau_m, tau_e),
        }
    }

    fn step(
        &mut self,
        I: Bound<'_, PyArray1<f32>>,
        spikes: Bound<'_, PyArray1<i32>>,
        pre_spikes: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<usize> {
        let I = I.readonly();
        let mut spikes = spikes.readwrite();
        let pre_spikes = pre_spikes.readonly();
        Ok(self.inner.step(
            I.as_slice()?,
            spikes.as_slice_mut()?,
            pre_spikes.as_slice()?,
        ))
    }

    fn update_weights(
        &self,
        weights: Bound<'_, PyArray1<f32>>,
        learning_signals: Bound<'_, PyArray1<f32>>,
        lr: f32,
    ) -> PyResult<()> {
        let mut weights = weights.readwrite();
        let ls = learning_signals.readonly();
        self.inner
            .update_weights(weights.as_slice_mut()?, ls.as_slice()?, lr);
        Ok(())
    }

    /// Eligibility traces как np.ndarray.
    #[getter]
    fn eligibility(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.eligibility.clone()).into()
    }

    /// Мембранный потенциал как np.ndarray.
    #[getter]
    fn v(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.v.clone()).into()
    }

    #[setter]
    fn set_v(&mut self, src: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let src = src.readonly();
        let n = self.inner.n.min(src.len()?);
        self.inner.v[..n].copy_from_slice(&src.as_slice()?[..n]);
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "EPropLIF(n={}, dt={}, τ_m={}, τ_e={})",
            self.inner.n, self.inner.dt, self.inner.tau_m, self.inner.tau_e
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich Multi — per-synapse eligibility (Python)
// ═══════════════════════════════════════════════════════════

/// E-Prop Izhikevich with per-synapse eligibility traces.
///
/// Each post-neuron maintains `n_pre` eligibility traces.
/// Column-major layout: `e[i * n_post + j]` for synapse (i→j).
///
/// `n_post` is padded to the SIMD width. The `step` and `update_weights`
/// methods therefore require `I_ext`, `spikes`, weights, and learning signals
/// to use the padded length exposed by `n_post()`.
///
/// Step computes: I_total_j = I_ext_j + Σ_i weights[i·n+j] · z_pre[i]
/// Then Izhikevich dynamics + per-synapse eligibility:
///   de_ij/dt = -e_ij/τ_e + ψ_j · z_pre_i
#[pyclass(name = "EPropIzhikevichMulti", module = "taas_checkmate.eprop")]
pub struct PyEPropIzhikevichMulti {
    inner: neuron::EPropIzhikevichMulti,
}

#[pymethods]
impl PyEPropIzhikevichMulti {
    #[new]
    #[pyo3(signature = (n_pre, n_post, dt=1.0, tau_e=20.0))]
    fn new(n_pre: usize, n_post: usize, dt: f32, tau_e: f32) -> PyResult<Self> {
        if n_pre == 0 {
            return Err(PyValueError::new_err("n_pre must be > 0"));
        }
        if n_post == 0 {
            return Err(PyValueError::new_err("n_post must be > 0"));
        }
        let n_padded = n_post
            .checked_add(7)
            .and_then(|value| value.checked_div(8))
            .and_then(|lanes| lanes.checked_mul(8))
            .ok_or_else(|| PyValueError::new_err("n_post overflows SIMD padding"))?;
        n_pre
            .checked_mul(n_padded)
            .ok_or_else(|| PyValueError::new_err("n_pre * n_post overflows"))?;
        Ok(Self {
            inner: neuron::EPropIzhikevichMulti::new(n_pre, n_post, dt, tau_e),
        })
    }

    /// One step: fused weight accumulation + Izhikevich + per-synapse eligibility.
    /// Returns number of output spikes.
    #[pyo3(signature = (weights, z_pre, I_ext, spikes))]
    fn step(
        &mut self,
        weights: Bound<'_, PyArray1<f32>>,
        z_pre: Bound<'_, PyArray1<i32>>,
        I_ext: Bound<'_, PyArray1<f32>>,
        spikes: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<usize> {
        let weights = weights.readonly();
        let z_pre = z_pre.readonly();
        let I_ext = I_ext.readonly();
        let mut spikes = spikes.readwrite();
        let expected_weights =
            self.inner
                .n_pre
                .checked_mul(self.inner.n_post)
                .ok_or_else(|| {
                    PyValueError::new_err("EPropIzhikevichMulti: synapse shape overflows")
                })?;
        if weights.len()? != expected_weights {
            return Err(PyValueError::new_err(format!(
                "weights length mismatch: expected {}, got {}",
                expected_weights,
                weights.len()?
            )));
        }
        if z_pre.len()? != self.inner.n_pre {
            return Err(PyValueError::new_err(format!(
                "z_pre length mismatch: expected {}, got {}",
                self.inner.n_pre,
                z_pre.len()?
            )));
        }
        if I_ext.len()? != self.inner.n_post {
            return Err(PyValueError::new_err(format!(
                "I_ext length mismatch: expected padded n_post {}, got {}",
                self.inner.n_post,
                I_ext.len()?
            )));
        }
        if spikes.len()? != self.inner.n_post {
            return Err(PyValueError::new_err(format!(
                "spikes length mismatch: expected padded n_post {}, got {}",
                self.inner.n_post,
                spikes.len()?
            )));
        }
        Ok(self.inner.step(
            weights.as_slice()?,
            z_pre.as_slice()?,
            I_ext.as_slice()?,
            spikes.as_slice_mut()?,
        ))
    }

    /// Per-synapse E-Prop weight update: Δw_ij = η · L_j · e_ij
    /// weights: [n_pre × n_post] column-major, modified in-place
    /// learning_signals: [n_post] learning signal per hidden neuron
    #[pyo3(signature = (weights, learning_signals, lr))]
    fn update_weights(
        &self,
        weights: Bound<'_, PyArray1<f32>>,
        learning_signals: Bound<'_, PyArray1<f32>>,
        lr: f32,
    ) -> PyResult<()> {
        let mut weights = weights.readwrite();
        let ls = learning_signals.readonly();
        let expected_weights =
            self.inner
                .n_pre
                .checked_mul(self.inner.n_post)
                .ok_or_else(|| {
                    PyValueError::new_err("EPropIzhikevichMulti: synapse shape overflows")
                })?;
        if weights.len()? != expected_weights {
            return Err(PyValueError::new_err(format!(
                "weights length mismatch: expected {}, got {}",
                expected_weights,
                weights.len()?
            )));
        }
        if ls.len()? != self.inner.n_post {
            return Err(PyValueError::new_err(format!(
                "learning_signals length mismatch: expected {}, got {}",
                self.inner.n_post,
                ls.len()?
            )));
        }
        self.inner
            .update_weights_multi(weights.as_slice_mut()?, ls.as_slice()?, lr);
        Ok(())
    }

    /// Eligibility matrix [n_pre × n_post] как np.ndarray.
    #[getter]
    fn eligibility(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.e.clone()).into()
    }

    /// Мембранный потенциал [n_post] как np.ndarray.
    #[getter]
    fn v(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.v.clone()).into()
    }

    /// Get number of post-synaptic neurons
    fn n_post(&self) -> usize {
        self.inner.n_post
    }

    /// Get number of pre-synaptic neurons
    fn n_pre(&self) -> usize {
        self.inner.n_pre
    }

    /// Reset all internal state (v, u, psi, eligibility)
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "EPropIzhikevichMulti(n_pre={}, n_post={}, dt={}, τ_e={})",
            self.inner.n_pre, self.inner.n_post, self.inner.dt, self.inner.tau_e
        )
    }

    /// Set Izhikevich neuron parameters (a, b, c, d, tau_surrogate).
    /// Defaults: a=0.02, b=0.2, c=-65, d=8, tau_surrogate=2.0
    #[pyo3(signature = (a, b, c, d, tau_surrogate))]
    fn set_params(&mut self, a: f32, b: f32, c: f32, d: f32, tau_surrogate: f32) {
        self.inner.a = a;
        self.inner.b = b;
        self.inner.c = c;
        self.inner.d = d;
        self.inner.tau_surrogate = tau_surrogate;
    }
}

/// Surrogate gradient: σ'(v) = 1 / (1 + |v - v_th|/τ)²
///
/// Используется для E-Prop — аппроксимация градиента спайка.
#[pyfunction]
#[pyo3(name = "surrogate_gradient")]
pub fn py_surrogate_gradient(v: f32, v_thresh: f32, tau_surrogate: f32) -> f32 {
    neuron::surrogate_gradient(v, v_thresh, tau_surrogate)
}

/// E-Prop weight update: Δw = η · L_j · e_ji
///
/// Обновление весов через eligibility traces и learning signal.
#[pyfunction]
#[pyo3(name = "eprop_update")]
pub fn py_eprop_update(
    weights: Bound<'_, PyArray1<f32>>,
    eligibility: Bound<'_, PyArray1<f32>>,
    learning_signals: Bound<'_, PyArray1<f32>>,
    lr: f32,
) -> PyResult<()> {
    let mut weights = weights.readwrite();
    let elig = eligibility.readonly();
    let ls = learning_signals.readonly();
    let n = weights.len()?.min(elig.len()?).min(ls.len()?);
    let w = weights.as_slice_mut()?;
    let e = elig.as_slice()?;
    let l = ls.as_slice()?;
    for i in 0..n {
        w[i] += lr * l[i] * e[i];
    }
    Ok(())
}
