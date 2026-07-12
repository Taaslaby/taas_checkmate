//! PyO3 bindings for Poisson generator, Short-Term Plasticity, and LFP.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::core::neuron;

// ═══════════════════════════════════════════════════════════
//  Poisson Generator
// ═══════════════════════════════════════════════════════════

/// Poisson генератор спайков.
///
/// Генерирует спайки с заданной частотой по распределению Пуассона.
///
/// Использование:
///     gen = PoissonGenerator(1000, dt=1.0, seed=42)
///     gen.set_rate(0, 50.0)  # нейрон 0: 50 Гц
///     spikes = np.zeros(1000, dtype=np.int32)
///     gen.step(spikes)
#[pyclass(name = "PoissonGenerator", module = "taas_checkmate.stp")]
pub struct PyPoissonGenerator {
    inner: neuron::PoissonGenerator,
}

#[pymethods]
impl PyPoissonGenerator {
    #[new]
    #[pyo3(signature = (n, dt=1.0, seed=42))]
    fn new(n: usize, dt: f32, seed: u32) -> Self {
        Self { inner: neuron::PoissonGenerator::new(n, dt, seed) }
    }

    fn step(&mut self, spikes_out: Bound<'_, PyArray1<i32>>) -> PyResult<()> {
        let mut spikes_out = spikes_out.readwrite();
        self.inner.step(spikes_out.as_slice_mut()?);
        Ok(())
    }

    fn set_rate(&mut self, idx: usize, rate_hz: f32) { self.inner.set_rate(idx, rate_hz); }

    fn set_rates(&mut self, rates: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        self.inner.set_rates(rates.readonly().as_slice()?);
        Ok(())
    }

    fn n_neurons(&self) -> usize { self.inner.n_neurons() }

    fn __repr__(&self) -> String { format!("PoissonGenerator(n={})", self.inner.n_neurons()) }
}

// ═══════════════════════════════════════════════════════════
//  Short-Term Plasticity
// ═══════════════════════════════════════════════════════════

/// Short-Term Plasticity (STP) — Tsodyks-Markram модель.
///
/// Краткосрочная депрессия + фасилитация синапсов.
///
/// Использование:
///     stp = ShortTermPlasticity(1000, dt=1.0)
///     I_out = np.zeros(1000, dtype=np.float32)
///     stp.step(pre_spikes, I_out)
#[pyclass(name = "ShortTermPlasticity", module = "taas_checkmate.stp")]
pub struct PyShortTermPlasticity {
    pub inner: neuron::ShortTermPlasticity,
}

#[pymethods]
impl PyShortTermPlasticity {
    #[new]
    #[pyo3(signature = (n, dt=1.0, heterogeneous=false, seed=42))]
    fn new(n: usize, dt: f32, heterogeneous: bool, seed: u32) -> Self {
        if heterogeneous {
            Self { inner: neuron::ShortTermPlasticity::new_heterogeneous(n, dt, seed) }
        } else {
            Self { inner: neuron::ShortTermPlasticity::new(n, dt) }
        }
    }

    fn step(&mut self, pre_spikes: Bound<'_, PyArray1<i32>>, I_out: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let pre_spikes = pre_spikes.readonly();
        let mut I_out = I_out.readwrite();
        self.inner.step(pre_spikes.as_slice()?, I_out.as_slice_mut()?);
        Ok(())
    }

    /// Депрессия (x) как np.ndarray.
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.x[..self.inner.n].to_vec()).into()
    }

    /// Фасилитация (u) как np.ndarray.
    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.u[..self.inner.n].to_vec()).into()
    }

    fn set_tau_d(&mut self, tau_d: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let tau_d = tau_d.readonly();
        let n = self.inner.n.min(tau_d.len()?);
        self.inner.tau_d[..n].copy_from_slice(&tau_d.as_slice()?[..n]);
        Ok(())
    }

    fn set_tau_f(&mut self, tau_f: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let tau_f = tau_f.readonly();
        let n = self.inner.n.min(tau_f.len()?);
        self.inner.tau_f[..n].copy_from_slice(&tau_f.as_slice()?[..n]);
        Ok(())
    }

    fn reset(&mut self) { self.inner.reset(); }

    fn __repr__(&self) -> String { format!("ShortTermPlasticity(n={})", self.inner.n) }

    fn n_neurons(&self) -> usize { self.inner.n }
}

// ═══════════════════════════════════════════════════════════
//  LFP proxy
// ═══════════════════════════════════════════════════════════

/// LFP proxy: взвешенная сумма спайков по электродам.
///
/// lfp_out[e] = Σ spikes[i] где electrode_map[i] == e
#[pyfunction]
#[pyo3(name = "compute_lfp")]
pub fn py_compute_lfp(
    spikes: Bound<'_, PyArray1<i32>>,
    electrode_map: Bound<'_, PyArray1<i32>>,
    lfp_out: Bound<'_, PyArray1<f32>>,
    n_electrodes: usize,
) -> PyResult<()> {
    let spikes = spikes.readonly();
    let electrode_map = electrode_map.readonly();
    let mut lfp_out = lfp_out.readwrite();
    neuron::compute_lfp(spikes.as_slice()?, electrode_map.as_slice()?, lfp_out.as_slice_mut()?, n_electrodes);
    Ok(())
}
