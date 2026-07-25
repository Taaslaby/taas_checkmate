//! PyO3 bindings for Koit — XOR attention and hardware RNG.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::core::koit;

// ─── Koit bindings ───────────────────────────────────

/// XOR attention: бинарный query vs keys, 0 умножений float.
///
/// out[k] = Σ_i values[i] · popcount(query XOR keys[i]) для топ-k ключей.
/// Полностью целочисленная attention — только XOR + popcount.
#[pyfunction]
#[pyo3(name = "xor_attention")]
pub fn py_xor_attention(
    query: Bound<'_, PyArray1<u8>>,
    keys: Bound<'_, PyArray1<u8>>,
    values: Bound<'_, PyArray1<f32>>,
    out: Bound<'_, PyArray1<f32>>,
    n_keys: usize,
    n_bytes: usize,
) -> PyResult<()> {
    koit::xor_attention(
        query.readonly().as_slice()?,
        keys.readonly().as_slice()?,
        values.readonly().as_slice()?,
        out.readwrite().as_slice_mut()?,
        n_keys,
        n_bytes,
    );
    Ok(())
}

/// Аппаратный RNG: заполнить буфер случайными float через CPU rdrand.
///
/// Использует инструкцию RDRAND (Intel/AMD).
#[pyfunction]
#[pyo3(name = "rdrand_fill")]
pub fn py_rdrand_fill(buf: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
    koit::rdrand_fill(buf.readwrite().as_slice_mut()?);
    Ok(())
}
