use super::kuramoto::{kuramoto_prepare, kuramoto_update};
use super::t3::t3_update;

// Перенесено из `core::kuramoto`— rayon-параллельные шаги
// поверх `kuramoto_prepare` / `kuramoto_update` / `t3_update`.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[inline(always)]
fn safe_row(row_ptr: &[i32], idx: usize) -> usize {
    if idx >= row_ptr.len() {
        return 0;
    }
    let r = row_ptr[idx];
    if r < 0 {
        0
    } else {
        r as usize
    }
}

#[inline(always)]
fn safe_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() {
        return 0;
    }
    let c = col_indices[idx];
    if c < 0 || c as usize >= n {
        0
    } else {
        c as usize
    }
}

/// Full sparse Kuramoto step with rayon parallelization.
///
/// Computes coupling via CSR sparse matrix: `coupling[i] = Σⱼ w[i,j]·sin(θⱼ−θᵢ)`,
/// then updates phases via `θ += dt·(ω + coupling)`.
pub fn kuramoto_step_sparse(
    phases: &mut [f32],
    omegas: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    weights: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(
        n == omegas.len()
            && n + 1 == row_ptr.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
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
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
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
        coupling_buf[i] = coupling_sum;
    }

    kuramoto_update(phases, omegas, coupling_buf, dt);
}

/// Dense Kuramoto step with rayon parallelization.
///
/// Same as `kuramoto_step_sparse` but with a full N×N coupling matrix.
pub fn kuramoto_step_dense(
    phases: &mut [f32],
    omegas: &[f32],
    coupling_matrix: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
) {
    let n = phases.len();
    assert!(
        n == omegas.len()
            && n * n == coupling_matrix.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
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
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let cos_i = cos_buf[i];
        let sin_i = sin_buf[i];
        let row_start = i * n;
        let mut coupling_sum = 0.0f32;
        for j in 0..n {
            let w = coupling_matrix[row_start + j];
            coupling_sum += w * (sin_buf[j] * cos_i - cos_buf[j] * sin_i);
        }
        coupling_buf[i] = coupling_sum;
    }

    kuramoto_update(phases, omegas, coupling_buf, dt);
}

/// Full sparse T3 (Temporal Triplet) step with rayon parallelization.
///
/// Combines Kuramoto phase dynamics with Stuart-Landau amplitude evolution:
/// `r += dt·((α−r²)·r + gain·drive − decay·r)`
/// `θ += dt·(ω + coupling + gain·drive/(r+ε))`
pub fn t3_step_sparse(
    phases: &mut [f32],
    amplitudes: &mut [f32],
    omegas: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    weights: &[f32],
    drives: &[f32],
    sin_buf: &mut [f32],
    cos_buf: &mut [f32],
    coupling_buf: &mut [f32],
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
) {
    let n = phases.len();
    assert!(
        n == amplitudes.len()
            && n == omegas.len()
            && n + 1 == row_ptr.len()
            && n == drives.len()
            && n <= sin_buf.len()
            && n <= cos_buf.len()
            && n <= coupling_buf.len()
    );

    kuramoto_prepare(phases, cos_buf, sin_buf);

    #[cfg(feature = "rayon")]
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
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
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
        coupling_buf[i] = coupling_sum;
    }

    t3_update(
        phases,
        amplitudes,
        omegas,
        coupling_buf,
        drives,
        dt,
        decay,
        alpha,
        gain,
    );
}
