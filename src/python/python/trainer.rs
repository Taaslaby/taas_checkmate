//! PyO3 bindings for `EPropTrainer` — multi-threaded batch training.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::core::neuron::trainer::EPropTrainer;

/// Multi-threaded batch trainer for E-Prop Izhikevich Multi.
///
/// Uses rayon parallelism internally. Each worker thread has its own
/// `EPropIzhikevichMulti` instance and RNG.
///
/// Call `train_batch(images, targets)` to train on multiple images
/// in parallel, then `predict(image)` for inference.
#[pyclass(name = "EPropTrainer", module = "taas_checkmate.trainer")]
pub struct PyEPropTrainer {
    inner: EPropTrainer,
}

#[pymethods]
impl PyEPropTrainer {
    #[new]
    #[pyo3(signature = (
        ni, nh, no,
        n_steps=50, poisson_max_hz=200.0,
        dt=1.0, tau_e=30.0,
        lr_in=0.05, lr_out=0.02,
        bias=3.0, weight_scale=0.357,
        seed=42, num_workers=None
    ))]
    fn new(
        ni: usize,
        nh: usize,
        no: usize,
        n_steps: usize,
        poisson_max_hz: f32,
        dt: f32,
        tau_e: f32,
        lr_in: f32,
        lr_out: f32,
        bias: f32,
        weight_scale: f32,
        seed: u64,
        num_workers: Option<usize>,
    ) -> Self {
        let nw = num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        Self {
            inner: EPropTrainer::new(
                ni, nh, no,
                n_steps, poisson_max_hz,
                dt, tau_e,
                lr_in, lr_out,
                bias, weight_scale, seed, nw,
            ),
        }
    }

    /// Train on a batch of images in parallel.
    ///
    /// `images`: 2D numpy array shape [batch_size, ni], float32
    /// `targets`: 1D numpy array shape [batch_size], int32
    ///
    /// Returns (avg_loss, correct_count, accuracy).
    #[pyo3(signature = (images, targets))]
    fn train_batch(
        &mut self,
        images: Bound<'_, PyArray1<f32>>,
        targets: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<(f32, u32, f32)> {
        let img = images.readonly();
        let img_slice = img.as_slice()?;
        let tgt = targets.readonly();
        let tgt_slice = tgt.as_slice()?;

        let batch_size = tgt_slice.len();
        let expected_len = batch_size * self.inner.ni;
        if img_slice.len() < expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("images len {} < batch_size {} * ni {}", img_slice.len(), batch_size, self.inner.ni)
            ));
        }

        let (avg_loss, correct) = self.inner.train_batch(img_slice, tgt_slice);
        let acc = correct as f32 / batch_size.max(1) as f32;
        Ok((avg_loss, correct, acc))
    }

    /// Get/set hidden bias (I_ext baseline for all hidden neurons).
    #[pyo3(signature = (bias))]
    fn set_bias_hidden(&mut self, bias: f32) {
        self.inner.bias_hidden.fill(bias);
    }

    fn bias_hidden(&self) -> f32 { self.inner.bias_hidden[0] }

    /// Predict class for a single image.
    #[pyo3(signature = (image))]
    fn predict(&mut self, image: Bound<'_, PyArray1<f32>>) -> PyResult<i32> {
        let img = image.readonly();
        let img_slice = img.as_slice()?;
        Ok(self.inner.predict(img_slice))
    }

    /// Input weights [ni × nh] как np.ndarray.
    #[getter]
    fn weights(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.weights.clone()).into()
    }

    /// Output weights [nh × no] как np.ndarray.
    #[getter]
    fn w_out(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.w_out.clone()).into()
    }

    fn ni(&self) -> usize { self.inner.ni }
    fn nh(&self) -> usize { self.inner.nh }
    fn no(&self) -> usize { self.inner.no }

    fn __repr__(&self) -> String {
        format!(
            "EPropTrainer(ni={}, nh={}, no={}, n_steps={}, lr_in={}, lr_out={}, workers={})",
            self.inner.ni, self.inner.nh, self.inner.no,
            self.inner.n_steps, self.inner.lr_in, self.inner.lr_out,
            self.inner.num_workers(),
        )
    }
}
