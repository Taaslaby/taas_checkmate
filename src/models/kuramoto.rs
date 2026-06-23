//! Safe Rust wrappers for Kuramoto ASM kernels.
//! Ported from `kuramoto_engine/c/engine_kuramoto.c`
//! Sparse propagation parallelized with rayon.

#![allow(non_snake_case)]

use crate::kernels;
use rayon::prelude::*;

// ─── Safe CSR access helpers ─────────────────────────────

#[inline(always)]
fn safe_row(row_ptr: &[i32], idx: usize) -> usize {
    if idx >= row_ptr.len() { return 0; }
    let r = row_ptr[idx];
    if r < 0 { 0 } else { r as usize }
}

#[inline(always)]
fn safe_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() { return 0; }
    let c = col_indices[idx];
    if c < 0 || c as usize >= n { 0 } else { c as usize }
}

// ─── Kuramoto (standard) ─────────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Full sparse Kuramoto step. Sparse propagation parallelized with rayon.
pub fn kuramoto_step_sparse(
    phases: &mut [f32], omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], weights: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(n == omegas.len() && n + 1 == row_ptr.len()
        && n <= sin_buf.len() && n <= cos_buf.len() && n <= coupling_buf.len());

    unsafe {
        kernels::kuramoto_prepare(phases.as_mut_ptr(), cos_buf.as_mut_ptr(), sin_buf.as_mut_ptr(), n);
    }

    // Parallel sparse propagation (like C OpenMP)
    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });

    unsafe {
        kernels::kuramoto_update(phases.as_mut_ptr(), omegas.as_ptr(), coupling_buf.as_ptr(), dt, n);
    }
}

/// Direct Kuramoto step (dense). Parallelized.
pub fn kuramoto_step_dense(
    phases: &mut [f32], omegas: &[f32], coupling_matrix: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(n == omegas.len() && n * n == coupling_matrix.len()
        && n <= sin_buf.len() && n <= cos_buf.len() && n <= coupling_buf.len());

    unsafe {
        kernels::kuramoto_prepare(phases.as_mut_ptr(), cos_buf.as_mut_ptr(), sin_buf.as_mut_ptr(), n);
    }

    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let row_start = i * n;
        let mut coupling_sum = 0.0f32;
        for j in 0..n {
            let w = coupling_matrix[row_start + j];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });

    unsafe {
        kernels::kuramoto_update(phases.as_mut_ptr(), omegas.as_ptr(), coupling_buf.as_ptr(), dt, n);
    }
}

// ─── T3 (Temporal Triplet) ───────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Full sparse T3 step. Parallelized.
pub fn t3_step_sparse(
    phases: &mut [f32], amplitudes: &mut [f32], omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], weights: &[f32],
    drives: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32, decay: f32, alpha: f32, gain: f32,
) {
    let n = phases.len();
    assert!(n == amplitudes.len() && n == omegas.len() && n + 1 == row_ptr.len()
        && n == drives.len()
        && n <= sin_buf.len() && n <= cos_buf.len() && n <= coupling_buf.len());

    unsafe {
        kernels::kuramoto_prepare(phases.as_mut_ptr(), cos_buf.as_mut_ptr(), sin_buf.as_mut_ptr(), n);
    }

    coupling_buf.par_iter_mut().enumerate().for_each(|(i, c)| {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let start = safe_row(row_ptr, i);
        let end = safe_row(row_ptr, i + 1);
        let mut coupling_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for idx in start..end {
            let j = safe_col(col_indices, idx, n);
            let w = weights[idx];
            let a_j = amplitudes[j];
            coupling_sum += w * a_j * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        *c = coupling_sum;
    });

    unsafe {
        kernels::t3_update(phases.as_mut_ptr(), amplitudes.as_mut_ptr(), omegas.as_ptr(),
            coupling_buf.as_ptr(), drives.as_ptr(), dt, decay, alpha, gain, n);
    }
}

// ─── Decoders ────────────────────────────────────────────

/// LMS (NLMS) train step.
pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32, eps: f32) {
    let n = weights.len();
    assert!(n == states.len());
    unsafe { kernels::nlms_train(weights.as_mut_ptr(), bias.as_mut_ptr(), states.as_ptr(), error, lr, eps, n); }
}

/// T3 LMS train step.
pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32) {
    let n = weights.len();
    assert!(n == states.len());
    unsafe { kernels::t3_lms_train(weights.as_mut_ptr(), bias.as_mut_ptr(), states.as_ptr(), error, lr, n); }
}

/// PST decoder: decode from ring buffer.
pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32], feat_dim: usize, L: usize, pos: usize) {
    unsafe { kernels::pst_decode(output.as_mut_ptr(), buffer.as_ptr(), weights.as_ptr(), feat_dim, L, pos); }
}

/// Ring synthesis.
pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32], weights: &[f32]) {
    let n = output.len();
    assert!(n == amplitudes.len() && n == trig_buf.len() && n == weights.len());
    unsafe { kernels::ring_synthesis(output.as_mut_ptr(), amplitudes.as_ptr(), trig_buf.as_ptr(), weights.as_ptr(), n); }
}

/// Kalman filter decoder update.
///
/// Updates weights and covariance using a scalar Kalman filter:
///
/// ```text
/// K[i]  = cov[i] · states[i] / s
/// w[i] += K[i] · error
/// cov[i] -= K[i] · states[i] · cov[i]
/// ```
///
/// # Parameters
///
/// * `weights` — decoder weights `[n]`, modified in-place
/// * `cov` — state covariance `[n]`, modified in-place
/// * `states` — input features `[n]`
/// * `error` — prediction error `(target - prediction)`
/// * `s` — **innovation covariance**: `s = R + Σᵢ(cov[i] · states[i]²)`
///
///   Must be computed by the caller as the sum of measurement noise `R` and
///   the projected state covariance. If `s` is just a constant, the Kalman
///   gain will be incorrect. See `KalmanDecoder` for the correct computation.
pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32], error: f32, s: f32) {
    let n = weights.len();
    assert!(n == states.len() && n == cov.len());
    debug_assert!(s > 0.0, "kalman_update: s must be R + Σ(cov·states²) > 0, got {}", s);
    unsafe { kernels::kalman_update(weights.as_mut_ptr(), cov.as_mut_ptr(), states.as_ptr(), error, s, n); }
}

#[allow(clippy::too_many_arguments)]
/// Adam optimizer update.
pub fn adam_train(weights: &mut [f32], m: &mut [f32], v: &mut [f32], states: &[f32],
    error: f32, lr: f32, beta1: f32, beta2: f32, bc1: f32, bc2: f32, eps: f32) {
    let n = weights.len();
    assert!(n == m.len() && n == v.len() && n == states.len());
    unsafe { kernels::adam_train(weights.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(), states.as_ptr(),
        error, lr, beta1, beta2, bc1, bc2, eps, n); }
}

// ─── Interference / Coherence ────────────────────────────

/// Interference between two phase arrays.
pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len(); let nb = phases_b.len();
    assert!(output.len() >= na * nb);
    unsafe { kernels::interference(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), na, nb); }
}

/// Mean interference.
pub fn interference_mean(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let n = phases_a.len();
    assert!(n == phases_b.len() && !output.is_empty(), "interference_mean needs output[0]");
    unsafe { kernels::interference_mean(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), n); }
}

/// Phase coherence.
pub fn phase_coherence(output: &mut [f32], sin_buf: &[f32], cos_buf: &[f32], amplitudes: &[f32]) {
    let n = sin_buf.len();
    assert!(n == cos_buf.len() && n == amplitudes.len() && !output.is_empty());
    unsafe { kernels::phase_coherence(output.as_mut_ptr(), sin_buf.as_ptr(), cos_buf.as_ptr(), amplitudes.as_ptr(), n); }
}

/// Winding number matrix.
pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len(); let nb = phases_b.len();
    assert!(output.len() >= na * nb);
    unsafe { kernels::winding_matrix(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), na, nb); }
}

/// Band energy.
pub fn band_energy(band_energies: &mut [f32], amplitudes: &[f32], omegas: &[f32], band_edges: &[f32]) {
    let n_bands = band_energies.len();
    let n = amplitudes.len();
    assert!(n == omegas.len());
    unsafe { kernels::band_energy(band_energies.as_mut_ptr(), amplitudes.as_ptr(), omegas.as_ptr(), band_edges.as_ptr(), n_bands, n); }
}

// ─── Event-driven / Lazy ─────────────────────────────────

/// Lazy sin/cos.
pub fn lazy_sincos_update(sin_out: &mut [f32], cos_out: &mut [f32], phases: &[f32], phases_prev: &mut [f32], eps: f32) {
    let n = phases.len();
    assert!(n == sin_out.len() && n == cos_out.len() && n == phases_prev.len());
    unsafe { kernels::lazy_sincos_update(sin_out.as_mut_ptr(), cos_out.as_mut_ptr(), phases.as_ptr(), phases_prev.as_mut_ptr(), eps, n); }
}

/// Batch coherence sweep.
pub fn batch_coherence_sweep(output: &mut [f32], areas: &[f32], areas_sin: &[f32], areas_cos: &[f32], n_areas: usize, n_elem: usize) {
    assert!(areas.len() >= n_areas * n_elem && areas_sin.len() >= n_areas * n_elem
        && areas_cos.len() >= n_areas * n_elem && output.len() >= n_areas * n_areas);
    unsafe { kernels::batch_coherence_sweep(output.as_mut_ptr(), areas.as_ptr(), areas_sin.as_ptr(), areas_cos.as_ptr(), n_areas, n_elem); }
}
