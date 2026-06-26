//! # Ridge regression utilities.
//!
//! - [`RidgeSolver`] — Ridge via Conjugate Gradient
//! - [`solve_cg`] — CG solver core
//! - [`compute_pearson`] — Pearson correlation
//! - [`normalize_features`] — feature normalization

// ─── Conjugate Gradient ──────────────────────────────────

/// Solve Ax = b via Conjugate Gradient.
///
/// A — симметричная положительно определённая матрица n×n (flattened row-major).
/// b — правая часть длины n.
pub fn solve_cg(a: &[f32], b: &[f32], n: usize, max_iter: usize, tol: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rsold: f32 = r.iter().map(|&x| x * x).sum();

    for _ in 0..max_iter {
        // Ap = A · p
        let mut ap = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i * n + j] * p[j];
            }
            ap[i] = sum;
        }
        let p_ap: f32 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
        if p_ap.abs() < 1e-16 { break; }
        let alpha = rsold / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rsnew: f32 = r.iter().map(|&x| x * x).sum();
        if rsnew.sqrt() < tol { break; }
        let beta = rsnew / rsold;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rsold = rsnew;
    }
    x
}

// ─── RidgeSolver ─────────────────────────────────────────

/// Ridge regression solver via Conjugate Gradient.
///
/// Решает: min_w ||Xw - y||² + α·||w||²
///
/// Эквивалентно: w = (XᵀX + αI)⁻¹Xᵀy
pub struct RidgeSolver {
    pub alpha: f32,
}

impl RidgeSolver {
    pub fn new(alpha: f32) -> Self { Self { alpha: alpha.max(1e-8) } }

    /// Solve ridge regression.
    ///
    /// - `x`: flattened X matrix [n_samples × d]
    /// - `y`: target vector [n_samples]
    /// - `n`: number of samples
    /// - `d`: feature dimension
    pub fn solve(&self, x: &[f32], y: &[f32], n: usize, d: usize) -> Vec<f32> {
        assert!(x.len() >= n * d && y.len() >= n);
        // Build XᵀX (d×d) and Xᵀy (d)
        let mut a = vec![0.0f32; d * d];
        let mut b = vec![0.0f32; d];
        for i in 0..n {
            for j1 in 0..d {
                let x_i_j1 = x[i * d + j1];
                b[j1] += x_i_j1 * y[i];
                for j2 in 0..d {
                    a[j1 * d + j2] += x_i_j1 * x[i * d + j2];
                }
            }
        }
        // Add Ridge penalty
        for j in 0..d { a[j * d + j] += self.alpha; }
        solve_cg(&a, &b, d, 100, 1e-6)
    }
}

// ─── Helpers ─────────────────────────────────────────────

/// Pearson correlation coefficient between two vectors.
pub fn compute_pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let mean_a: f32 = a.iter().sum::<f32>() / n as f32;
    let mean_b: f32 = b.iter().sum::<f32>() / n as f32;
    let (cov, var_a, var_b) = (0..n).fold((0.0, 0.0, 0.0), |(c, va, vb), i| {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        (c + da * db, va + da * da, vb + db * db)
    });
    if var_a < 1e-10 || var_b < 1e-10 { 0.0 } else { cov / (var_a * var_b).sqrt() }
}

/// Feature normalization: (x - mean) / std (in-place).
pub fn normalize_features(features: &mut [f32], n_samples: usize, dim: usize) {
    if n_samples == 0 { return; }
    for d in 0..dim {
        let mut sum = 0.0f32;
        for i in 0..n_samples { sum += features[i * dim + d]; }
        let mean = sum / n_samples as f32;
        let mut var = 0.0f32;
        for i in 0..n_samples {
            let diff = features[i * dim + d] - mean;
            var += diff * diff;
        }
        let std = (var / n_samples as f32).sqrt().max(1e-8);
        for i in 0..n_samples {
            features[i * dim + d] = (features[i * dim + d] - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_solver() {
        let solver = RidgeSolver::new(0.01);
        // y = 2*x1 + 3*x2
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let y = vec![2.0, 3.0, 5.0, 4.0, 6.0];
        let w = solver.solve(&x, &y, 5, 2);
        assert!((w[0] - 2.0).abs() < 0.1, "w[0]≈2: {}", w[0]);
        assert!((w[1] - 3.0).abs() < 0.1, "w[1]≈3: {}", w[1]);
    }

    #[test]
    fn test_compute_pearson() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = compute_pearson(&a, &b);
        assert!((r - 1.0).abs() < 0.01, "r≈1: {}", r);
    }

    #[test]
    fn test_compute_pearson_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = compute_pearson(&a, &b);
        assert!((r - (-1.0)).abs() < 0.01, "perfect negative corr r≈-1: {}", r);
    }

    #[test]
    fn test_normalize_features() {
        let mut features = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        normalize_features(&mut features, 3, 2);
        // After normalization, each column should have mean≈0, std≈1
        let mean0 = (features[0] + features[2] + features[4]) / 3.0;
        assert!(mean0.abs() < 1e-6, "mean col0≈0: {}", mean0);
    }

    #[test]
    fn test_solve_cg() {
        // 2×2 identity: A = [[2,0],[0,2]], b = [4,6] → x = [2,3]
        let a = vec![2.0, 0.0, 0.0, 2.0];
        let b = vec![4.0, 6.0];
        let x = solve_cg(&a, &b, 2, 100, 1e-6);
        assert!((x[0] - 2.0).abs() < 0.01, "x[0]≈2: {}", x[0]);
        assert!((x[1] - 3.0).abs() < 0.01, "x[1]≈3: {}", x[1]);
    }
}
