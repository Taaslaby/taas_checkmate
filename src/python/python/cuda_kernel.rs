//! Python facade for the trusted generic NVRTC kernel launcher.
//!
//! This module is compiled only with `python,cuda`. It accepts contiguous
//! `float32` NumPy buffers and scalar Python arguments, then delegates all CUDA
//! work to `pillars_cuda::CudaKernel`.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use pillars_cuda::{BufferArg, CudaKernel, LaunchConfig, ScalarArg};

#[pyclass(name = "CudaKernel", unsendable)]
pub struct PyCudaKernel {
    inner: CudaKernel,
}

#[pymethods]
impl PyCudaKernel {
    #[new]
    #[pyo3(signature = (source, name))]
    fn new(source: &str, name: &str) -> PyResult<Self> {
        CudaKernel::compile(source, name)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    /// Upload, launch, synchronize, and download contiguous float32 buffers.
    ///
    /// `buffers` are passed first to the CUDA kernel in order. `scalars` are
    /// passed after them: Python integers become `int32`, floats become
    /// `float32`. `grid` and `block` are one-dimensional launch dimensions.
    #[pyo3(signature = (buffers, scalars=None, grid=1, block=256))]
    fn run(
        &self,
        py: Python<'_>,
        buffers: Vec<Py<PyArray1<f32>>>,
        scalars: Option<Vec<PyObject>>,
        grid: u32,
        block: u32,
    ) -> PyResult<()> {
        if buffers.is_empty() {
            return Err(PyValueError::new_err("buffers must not be empty"));
        }
        let config = LaunchConfig { grid, block };
        config.validate().map_err(PyValueError::new_err)?;

        // Keep NumPy objects alive and validate contiguity while the GIL is
        // held. We later reconstruct disjoint mutable slices from the raw
        // ranges; duplicate/overlapping arrays are rejected first.
        let mut ranges = Vec::with_capacity(buffers.len());
        for array in &buffers {
            let bound = array.bind(py);
            let mut rw = bound.readwrite();
            let slice = rw
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("buffers must be contiguous float32 arrays"))?;
            ranges.push((slice.as_mut_ptr() as usize, slice.len()));
        }
        for i in 0..ranges.len() {
            let (start, len) = ranges[i];
            let end =
                start
                    .checked_add(len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                        PyValueError::new_err("buffer byte range overflows usize")
                    })?)
                    .ok_or_else(|| PyValueError::new_err("buffer byte range overflows usize"))?;
            for &(other_start, other_len) in &ranges[..i] {
                let other_end = other_start
                    .checked_add(
                        other_len
                            .checked_mul(std::mem::size_of::<f32>())
                            .ok_or_else(|| {
                                PyValueError::new_err("buffer byte range overflows usize")
                            })?,
                    )
                    .ok_or_else(|| PyValueError::new_err("buffer byte range overflows usize"))?;
                if start < other_end && other_start < end {
                    return Err(PyValueError::new_err("buffers must not alias or overlap"));
                }
            }
        }

        let mut scalar_args = Vec::with_capacity(scalars.as_ref().map_or(0, Vec::len));
        for scalar in scalars.unwrap_or_default() {
            let bound = scalar.bind(py);
            if let Ok(value) = bound.extract::<i32>() {
                scalar_args.push(ScalarArg::I32(value));
            } else if let Ok(value) = bound.extract::<f32>() {
                scalar_args.push(ScalarArg::F32(value));
            } else if let Ok(value) = bound.extract::<u32>() {
                scalar_args.push(ScalarArg::U32(value));
            } else {
                return Err(PyTypeError::new_err(
                    "scalars must contain only int32, uint32, or float32 values",
                ));
            }
        }

        // The raw ranges were obtained from the live NumPy arrays above and
        // have been checked not to overlap. The arrays remain owned by
        // `buffers` for the duration of this call, so these slices are valid.
        let mut slices = ranges
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) })
            .collect::<Vec<_>>();
        let mut buffer_args = slices
            .iter_mut()
            .map(|slice| BufferArg::F32(slice))
            .collect::<Vec<_>>();

        self.inner
            .launch(&mut buffer_args, &scalar_args, config)
            .map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }
}
