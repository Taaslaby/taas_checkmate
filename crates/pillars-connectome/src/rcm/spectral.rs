/// Построить Laplacian L = D - A в CSR формате из списка смежности.
fn build_laplacian_csr(adj: &[Vec<usize>]) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n = adj.len();
    let mut row_ptr = vec![0i32; n + 1];

    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        row_ptr[i] = col_indices.len() as i32;
        col_indices.push(i as i32);
        values.push(adj[i].len() as f32);
        for &j in &adj[i] {
            if j > i {
                col_indices.push(j as i32);
                values.push(-1.0);
            }
        }
    }
    row_ptr[n] = col_indices.len() as i32;

    (row_ptr, col_indices, values)
}

/// Симметричная CSR × vector: y = L · x (только верхний треугольник).
fn csr_symv(row_ptr: &[i32], col_indices: &[i32], values: &[f32], x: &[f32], y: &mut [f32]) {
    let n = x.len();
    y.fill(0.0);
    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut diag_val = 0.0f32;
        for j in start..end {
            let col = col_indices[j] as usize;
            let v = values[j];
            if col == i {
                diag_val = v;
            } else {
                y[i] += v * x[col];
                y[col] += v * x[i];
            }
        }
        y[i] += diag_val * x[i];
    }
}

/// Решить (L - μI)⁻¹ · x через conjugate gradient (CG).
fn solve_shifted_laplacian(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    mu: f32,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f32,
) {
    let n = b.len();
    x.copy_from_slice(b);

    let mut r = vec![0.0f32; n];
    csr_symv(row_ptr, col_indices, values, x, &mut r);
    for i in 0..n {
        r[i] = b[i] - (r[i] - mu * x[i]);
    }

    let mut p = r.clone();
    let r_dot = |a: &[f32]| -> f32 { a.iter().map(|&v| v * v).sum() };

    let mut rsq = r_dot(&r);
    let b_norm = r_dot(b).sqrt().max(1e-10);

    for _iter in 0..max_iter {
        if rsq.sqrt() < tol * b_norm {
            break;
        }

        let mut ap = vec![0.0f32; n];
        csr_symv(row_ptr, col_indices, values, &p, &mut ap);
        for i in 0..n {
            ap[i] -= mu * p[i];
        }

        let p_ap = p
            .iter()
            .zip(ap.iter())
            .map(|(pi, api)| pi * api)
            .sum::<f32>();
        if p_ap.abs() < 1e-12 {
            break;
        }

        let alpha = rsq / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rsq_new = r_dot(&r);
        let beta = rsq_new / rsq;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsq = rsq_new;
    }
}

/// Вычислить Fiedler vector (собственный вектор для λ₂) через Rayleigh quotient iteration.
pub fn fiedler_vector(adj: &[Vec<usize>], max_iter: usize, tol: f32) -> Vec<f32> {
    let n = adj.len();
    if n <= 2 {
        return vec![1.0f32; n];
    }

    let (rp, ci, vals) = build_laplacian_csr(adj);

    let mut x: Vec<f32> = (0..n)
        .map(|i| {
            let h = i.wrapping_mul(6364136223846793005);
            let v = (h.wrapping_mul(1442695040888963407) as f32) / 1.8446744e19;
            v * 2.0 - 1.0
        })
        .collect();

    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    for xi in x.iter_mut() {
        *xi -= mean;
    }

    let max_deg = adj.iter().map(|nei| nei.len()).max().unwrap_or(1) as f32;
    let lambda_max = 2.0 * max_deg;
    let mut sigma = lambda_max * 0.25;

    let mut lambda_old = sigma;
    let mut x_next = vec![0.0f32; n];
    for _iter in 0..max_iter {
        solve_shifted_laplacian(&rp, &ci, &vals, -0.1f32, &x, &mut x_next, 100, tol);
        x.copy_from_slice(&x_next);

        let mean_x: f32 = x.iter().sum::<f32>() / n as f32;
        for xi in x.iter_mut() {
            *xi -= mean_x;
        }

        let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for xi in x.iter_mut() {
                *xi /= norm;
            }
        }

        let mut lx = vec![0.0f32; n];
        csr_symv(&rp, &ci, &vals, &x, &mut lx);
        let rayleigh: f32 = x.iter().zip(lx.iter()).map(|(&xi, &lxi)| xi * lxi).sum();

        sigma = rayleigh;

        if (sigma - lambda_old).abs() < tol * sigma.abs().max(1.0) {
            break;
        }
        lambda_old = sigma;
    }

    let mean_x: f32 = x.iter().sum::<f32>() / n as f32;
    for xi in x.iter_mut() {
        *xi -= mean_x;
    }

    x
}

/// Spectral ordering (Fiedler) — сортировка нейронов по Fiedler vector.
pub fn spectral_ordering(adj: &[Vec<usize>], max_iter: usize, tol: f32) -> Vec<usize> {
    let n = adj.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let fv = fiedler_vector(adj, max_iter, tol);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        fv[i]
            .partial_cmp(&fv[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut perm = vec![0usize; n];
    for (pos, &node) in indices.iter().enumerate() {
        perm[node] = n - 1 - pos;
    }

    perm
}
