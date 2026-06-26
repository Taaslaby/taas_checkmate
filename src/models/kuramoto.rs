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

    kernels::kuramoto_prepare(phases, cos_buf, sin_buf);

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

    kernels::kuramoto_update(phases, omegas, coupling_buf, dt);
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

    kernels::kuramoto_prepare(phases, cos_buf, sin_buf);

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

    kernels::kuramoto_update(phases, omegas, coupling_buf, dt);
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

    kernels::kuramoto_prepare(phases, cos_buf, sin_buf);

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

    kernels::t3_update(phases, amplitudes, omegas, coupling_buf, drives, dt, decay, alpha, gain);
}

// ─── Decoders ────────────────────────────────────────────

/// LMS (NLMS) train step.
pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32, eps: f32) {
    assert_eq!(weights.len(), states.len());
    kernels::nlms_train(weights, bias, states, error, lr, eps);
}

/// T3 LMS train step.
pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32) {
    kernels::t3_lms_train(weights, bias, states, error, lr);
}

/// PST decoder: decode from ring buffer.
pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32], _feat_dim: usize, L: usize, pos: usize) {
    kernels::pst_decode(output, buffer, weights, L, pos);
}

/// Ring synthesis.
pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32], weights: &[f32]) {
    kernels::ring_synthesis(output, amplitudes, trig_buf, weights);
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
    debug_assert!(s > 0.0, "kalman_update: s must be R + Σ(cov·states²) > 0, got {}", s);
    kernels::kalman_update(weights, cov, states, error, s);
}

#[allow(clippy::too_many_arguments)]
/// Adam optimizer update.
pub fn adam_train(weights: &mut [f32], m: &mut [f32], v: &mut [f32], states: &[f32],
    error: f32, lr: f32, beta1: f32, beta2: f32, bc1: f32, bc2: f32, eps: f32) {
    kernels::adam_train(weights, m, v, states, error, lr, beta1, beta2, bc1, bc2, eps);
}

// ─── Interference / Coherence ────────────────────────────

/// Interference between two phase arrays.
pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    kernels::interference(output, phases_a, phases_b);
}

/// Mean interference.
pub fn interference_mean(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    kernels::interference_mean(&mut output[0], phases_a, phases_b);
}

/// Phase coherence.
pub fn phase_coherence(output: &mut [f32], sin_buf: &[f32], cos_buf: &[f32], amplitudes: &[f32]) {
    kernels::phase_coherence(&mut output[0], sin_buf, cos_buf, amplitudes);
}

/// Winding number matrix.
pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    kernels::winding_matrix(output, phases_a, phases_b);
}

/// Band energy.
pub fn band_energy(band_energies: &mut [f32], amplitudes: &[f32], omegas: &[f32], band_edges: &[f32]) {
    kernels::band_energy(band_energies, amplitudes, omegas, band_edges);
}

// ─── Event-driven / Lazy ─────────────────────────────────

/// Lazy sin/cos.
pub fn lazy_sincos_update(sin_out: &mut [f32], cos_out: &mut [f32], phases: &[f32], phases_prev: &mut [f32], eps: f32) {
    kernels::lazy_sincos_update(sin_out, cos_out, phases, phases_prev, eps);
}

/// Batch coherence sweep.
pub fn batch_coherence_sweep(output: &mut [f32], areas: &[f32], areas_sin: &[f32], areas_cos: &[f32], _n_areas: usize, n_elem: usize) {
    kernels::batch_coherence_sweep(output, areas, areas_sin, areas_cos, n_elem);
}
