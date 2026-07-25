//! # Sparse Connectome Gradients
//!
//! Вычисление градиентов по coupling-весам **только для ненулевых entries** CSR-матрицы.
//!
//! ## Когда это нужно
//!
//! По умолчанию T3 backward pass использует uniform coupling (скаляры `w12`, `w13`, `w23`).
//! Но connectome может иметь **разные веса на каждую синаптическую связь** (CSR-матрица).
//! Для N=1000 осцилляторов, full matrix — 1M пар, из которых >99% — нули.
//! Этот модуль вычисляет `dL/dw[i,j]` только для ненулевых entries CSR.
//!
//! ## Формула
//!
//! Для coupling term `c[post] = Σ_pre w[pre→post] · g · r[pre] · sin(θ_pre - θ_post) / N`:
//!
//! ```text
//! dL/dw[pre→post] = v_th[post] · dt · g · r[pre] · sin(θ_pre - θ_post) / N
//! ```
//!
//! где `g` — phase gate source-популяции, `v_th[post]` — накопленный градиент из BPTT.
//!
//! ## CSR Формат
//!
//! Каждая cross-pair (12, 13, 23) имеет три CSR матрицы: `row_ptr[pre]`, `col[k]`, `val[k]`.
//! `row_ptr[pre]..row_ptr[pre+1]` задаёт диапазон в `col`/`val` для pre-нейрона.
//! `post = col[k]`, вес = `val[k]`.
//!
//! Результат: `grad_w[k]` — градиент для каждого k (совпадает с форматом `val`).

use crate::spike::trajectory::{
    off_r1, off_r2, off_r3, off_th2, off_th3, T3SpikeTrajectory, OFF_TH1,
};

/// Результат sparse coupling backward: градиенты по весам для трёх cross-population пар.
/// Формат совпадает с `val` CSR матрицы — `grad_w[k]` соответствует `val[k]`.
pub struct SparseCouplingGradients {
    pub grad_w12: Vec<f32>,
    pub grad_w13: Vec<f32>,
    pub grad_w23: Vec<f32>,
}

/// Набор CSR матриц для coupling между одной парой популяций.
/// Используется как конфиг для `accumulate_sparse_gradients_into`.
pub struct SparseCouplingCsr<'a> {
    pub row_ptr: &'a [i32],
    pub col: &'a [i32],
    pub val: &'a [f32],
    /// Phase gate применяемый к source population (например g2 для pop1←pop2)
    pub g_source: f32,
}

/// Accumulate coupling gradients into pre-allocated output buffers for one pair.
///
/// ## Iterator pattern
///
/// ```text
/// for pre in 0..n:
///     for k in row_ptr[pre]..row_ptr[pre+1]:
///         post = col[k]
///         grad[k] += Σ_step v_th[post] · dt · g · r[pre] · sin(θ_pre - θ_post) / N
/// ```
fn accumulate_pair<const POP_TARGET: usize, const POP_SOURCE: usize>(
    grad_out: &mut [f32],
    csr: &SparseCouplingCsr,
    v_th_target: &[f32],
    traj: &T3SpikeTrajectory,
    n: usize,
    n_steps: usize,
    dt: f32,
) {
    let dt_n = dt / n as f32;
    let _nf = n as f32;

    // Offsets per population for sin/cos and r
    let off_th_source = match POP_SOURCE {
        1 => OFF_TH1,
        2 => off_th2(n_steps, n),
        3 => off_th3(n_steps, n),
        _ => unreachable!(),
    };
    let off_th_target = match POP_TARGET {
        1 => OFF_TH1,
        2 => off_th2(n_steps, n),
        3 => off_th3(n_steps, n),
        _ => unreachable!(),
    };
    let off_r_source = match POP_SOURCE {
        1 => off_r1(n_steps, n),
        2 => off_r2(n_steps, n),
        3 => off_r3(n_steps, n),
        _ => unreachable!(),
    };

    for step in 0..n_steps {
        let base = step * n;
        let th_source = &traj.flat[off_th_source + base..off_th_source + base + n];
        let th_target = &traj.flat[off_th_target + base..off_th_target + base + n];
        let r_source = &traj.flat[off_r_source + base..off_r_source + base + n];

        // Pre-compute sin/cos for this step (per-population, once per step)
        let mut sin_src = vec![0.0f32; n];
        let mut cos_src = vec![0.0f32; n];
        let mut sin_tgt = vec![0.0f32; n];
        let mut cos_tgt = vec![0.0f32; n];
        for i in 0..n {
            sin_src[i] = th_source[i].sin();
            cos_src[i] = th_source[i].cos();
            sin_tgt[i] = th_target[i].sin();
            cos_tgt[i] = th_target[i].cos();
        }

        let g = csr.g_source;

        // Sparse iteration: только ненулевые entries CSR
        for pre in 0..n {
            let start = csr.row_ptr[pre] as usize;
            let end = csr.row_ptr[pre + 1] as usize;
            if start >= end {
                continue;
            }

            let r_pre = r_source[pre] as f64;
            let sin_pre = sin_src[pre] as f64;
            let cos_pre = cos_src[pre] as f64;
            let dt_n_g = (dt_n * g) as f64;

            for k in start..end {
                let post = csr.col[k] as usize;
                if post >= n {
                    continue;
                }

                // sin(θ_pre - θ_post) = sin_pre·cos_post - cos_pre·sin_post
                let sin_diff = sin_pre * cos_tgt[post] as f64 - cos_pre * sin_tgt[post] as f64;

                // dL/dw[pre→post] = Σ_step v_th[post] · dt/n · g · r[pre] · sin(θ_pre - θ_post)
                grad_out[k] += (v_th_target[post] as f64 * dt_n_g * r_pre * sin_diff) as f32;
            }
        }
    }
}

/// Вычислить градиенты по coupling-весам для ненулевых entries CSR-матриц.
///
/// Использует CSR итерацию: только ненулевые entries, без материализации плотной матрицы.
/// sin/cos вычисляются per-step из траектории (не из BpttWorkspace — он держит только last step).
pub fn compute_sparse_coupling_gradients(
    v_th1: &[f32],
    v_th2: &[f32],
    _v_th3: &[f32],
    traj: &T3SpikeTrajectory,
    n_steps: usize,
    csr_12: &SparseCouplingCsr,
    csr_13: &SparseCouplingCsr,
    csr_23: &SparseCouplingCsr,
    dt: f32,
) -> SparseCouplingGradients {
    let n = traj.n_osc;

    let mut grad_12 = vec![0.0f32; csr_12.val.len()];
    let mut grad_13 = vec![0.0f32; csr_13.val.len()];
    let mut grad_23 = vec![0.0f32; csr_23.val.len()];

    // Pop 1 ← Pop 2: target=1, source=2
    accumulate_pair::<1, 2>(&mut grad_12, csr_12, v_th1, traj, n, n_steps, dt);
    // Pop 1 ← Pop 3
    accumulate_pair::<1, 3>(&mut grad_13, csr_13, v_th1, traj, n, n_steps, dt);
    // Pop 2 ← Pop 3
    accumulate_pair::<2, 3>(&mut grad_23, csr_23, v_th2, traj, n, n_steps, dt);

    SparseCouplingGradients {
        grad_w12: grad_12,
        grad_w13: grad_13,
        grad_w23: grad_23,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike::trajectory::T3SpikeTrajectory;

    fn make_csr_identity(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut row_ptr = vec![0i32; n + 1];
        let mut col = Vec::new();
        let mut val = Vec::new();
        for i in 0..n {
            col.push(i as i32);
            val.push(0.1);
            row_ptr[i + 1] = col.len() as i32;
        }
        (row_ptr, col, val)
    }

    fn make_csr_full(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut row_ptr = vec![0i32; n + 1];
        let mut col = Vec::new();
        let mut val = Vec::new();
        for i in 0..n {
            for j in 0..n {
                col.push(j as i32);
                val.push(0.1);
            }
            row_ptr[i + 1] = col.len() as i32;
        }
        (row_ptr, col, val)
    }

    #[test]
    fn test_sparse_coupling_identity() {
        let n = 8;
        let n_steps = 3;
        let traj =
            T3SpikeTrajectory::new(n, n_steps, 0.05, 10.0, 1.0, 0.1, 0.5, 3.0, 0.3, 0.15, 0.08);
        let (r, c, v) = make_csr_identity(n);
        let csr12 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 1.0,
        };
        let csr13 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 0.8,
        };
        let csr23 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 0.5,
        };

        let v_th = vec![1.0f32; n];
        let result = compute_sparse_coupling_gradients(
            &v_th, &v_th, &v_th, &traj, n_steps, &csr12, &csr13, &csr23, 0.05,
        );

        assert_eq!(result.grad_w12.len(), n);
        assert_eq!(result.grad_w13.len(), n);
        assert_eq!(result.grad_w23.len(), n);

        // All gradients should be finite (trajectory is all-zero, so sin_diff=0 → grad=0)
        for g in result
            .grad_w12
            .iter()
            .chain(result.grad_w13.iter())
            .chain(result.grad_w23.iter())
        {
            assert!(g.is_finite(), "gradient should be finite: {}", g);
        }
    }

    #[test]
    fn test_sparse_coupling_zero_gradients() {
        // v_th = 0 → все градиенты должны быть 0
        let n = 8;
        let n_steps = 3;
        let traj =
            T3SpikeTrajectory::new(n, n_steps, 0.05, 10.0, 1.0, 0.1, 0.5, 3.0, 0.3, 0.15, 0.08);
        let (r, c, v) = make_csr_full(n);
        let csr12 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 1.0,
        };
        let csr13 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 0.8,
        };
        let csr23 = SparseCouplingCsr {
            row_ptr: &r,
            col: &c,
            val: &v,
            g_source: 0.5,
        };

        let zero = vec![0.0f32; n];
        let result = compute_sparse_coupling_gradients(
            &zero, &zero, &zero, &traj, n_steps, &csr12, &csr13, &csr23, 0.05,
        );

        for g in &result.grad_w12 {
            assert!(g.abs() < 1e-10, "zero v_th → grad=0: {}", g);
        }
        for g in &result.grad_w13 {
            assert!(g.abs() < 1e-10, "zero v_th → grad=0: {}", g);
        }
        for g in &result.grad_w23 {
            assert!(g.abs() < 1e-10, "zero v_th → grad=0: {}", g);
        }
    }
}
