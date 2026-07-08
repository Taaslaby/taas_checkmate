use crate::dataset::ConnectomeSample;

/// Graph-theoretic metrics for connectome analysis and validation.
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    pub n_neurons: usize,
    pub n_synapses: usize,
    pub density: f32,
    pub mean_degree: f32,
    pub degree_variance: f32,
    pub ei_ratio: f32,
    pub mean_weight: f32,
    pub weight_std: f32,
    pub n_groups: usize,
    pub reciprocity: f32,
    pub small_world_index: f32,
    pub clustering_coefficient: f32,
    pub avg_path_length: f32,
}

impl GraphMetrics {
    pub fn compute(sample: &ConnectomeSample) -> Self {
        let n = sample.n_neurons;
        let m = sample.n_synapses;

        let density = if n > 1 {
            m as f32 / (n as f32 * (n as f32 - 1.0))
        } else {
            0.0
        };

        let out_degrees: Vec<usize> = (0..n)
            .map(|i| (sample.row_ptr[i + 1] - sample.row_ptr[i]) as usize)
            .collect();

        let mean_deg = out_degrees.iter().sum::<usize>() as f32 / n as f32;
        let deg_var = out_degrees
            .iter()
            .map(|&d| (d as f32 - mean_deg).powi(2))
            .sum::<f32>()
            / n as f32;

        let n_exc = sample.nt_types.iter().filter(|&&t| t > 0.0).count();
        let n_inh = sample.nt_types.iter().filter(|&&t| t < 0.0).count();
        let ei = if n_inh > 0 {
            n_exc as f32 / n_inh as f32
        } else {
            n_exc as f32
        };

        let mean_w = sample.values.iter().map(|v| v.abs()).sum::<f32>() / m as f32;
        let w_var = sample
            .values
            .iter()
            .map(|&v| (v.abs() - mean_w).powi(2))
            .sum::<f32>()
            / m as f32;
        let w_std = w_var.sqrt();

        let recip = self::compute_reciprocity(sample);
        let clust = self::compute_clustering(sample);
        let path = self::compute_avg_path(sample);
        let sw = if path > 0.0 && clust > 0.0 {
            let random_clust = density;
            let random_path = (n as f32).ln() / (mean_deg + 1.0).ln();
            (clust / random_clust) / (path / random_path.max(0.01))
        } else {
            0.0
        };

        Self {
            n_neurons: n,
            n_synapses: m,
            density,
            mean_degree: mean_deg,
            degree_variance: deg_var,
            ei_ratio: ei.min(100.0),
            mean_weight: mean_w,
            weight_std: w_std,
            n_groups: sample.groups.len(),
            reciprocity: recip,
            small_world_index: sw,
            clustering_coefficient: clust,
            avg_path_length: path,
        }
    }
}

fn compute_reciprocity(sample: &ConnectomeSample) -> f32 {
    let n = sample.n_neurons;
    let mut rev = vec![vec![]; n];
    for i in 0..n {
        let s = sample.row_ptr[i] as usize;
        let e = sample.row_ptr[i + 1] as usize;
        for &j in &sample.col_indices[s..e] {
            let j = j as usize;
            if j < n {
                rev[j].push(i);
            }
        }
    }
    let mut reciprocal = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        let s = sample.row_ptr[i] as usize;
        let e = sample.row_ptr[i + 1] as usize;
        for &j in &sample.col_indices[s..e] {
            let j = j as usize;
            if j < n {
                total += 1;
                if rev[j].contains(&i) {
                    reciprocal += 1;
                }
            }
        }
    }
    if total > 0 {
        reciprocal as f32 / total as f32
    } else {
        0.0
    }
}

fn compute_clustering(sample: &ConnectomeSample) -> f32 {
    let n = sample.n_neurons;
    let n_out: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let s = sample.row_ptr[i] as usize;
            let e = sample.row_ptr[i + 1] as usize;
            sample.col_indices[s..e]
                .iter()
                .map(|&j| j as usize)
                .filter(|&j| j < n)
                .collect()
        })
        .collect();

    let mut total_c = 0.0f32;
    let mut valid = 0usize;
    for i in 0..n {
        let neighbors = &n_out[i];
        let k = neighbors.len();
        if k < 2 {
            continue;
        }
        let mut edges = 0usize;
        for (a, &na) in neighbors.iter().enumerate() {
            for &nb in &neighbors[a + 1..] {
                if na < n && nb < n {
                    let row = &n_out[na];
                    if row.contains(&nb) || n_out[nb].contains(&na) {
                        edges += 1;
                    }
                }
            }
        }
        let possible = k * (k - 1) / 2;
        if possible > 0 {
            total_c += edges as f32 / possible as f32;
            valid += 1;
        }
    }
    if valid > 0 {
        total_c / valid as f32
    } else {
        0.0
    }
}

fn compute_avg_path(sample: &ConnectomeSample) -> f32 {
    let n = sample.n_neurons;
    if n > 500 {
        return 0.0;
    }
    let mut dist = vec![f32::INFINITY; n * n];
    for i in 0..n {
        dist[i * n + i] = 0.0;
    }
    for i in 0..n {
        let s = sample.row_ptr[i] as usize;
        let e = sample.row_ptr[i + 1] as usize;
        for &j in &sample.col_indices[s..e] {
            let j = j as usize;
            if j < n {
                dist[i * n + j] = 1.0;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let ik = dist[i * n + k];
            if ik.is_infinite() {
                continue;
            }
            for j in 0..n {
                let kj = dist[k * n + j];
                if kj.is_infinite() {
                    continue;
                }
                let nd = ik + kj;
                if nd < dist[i * n + j] {
                    dist[i * n + j] = nd;
                }
            }
        }
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in 0..n {
            if i != j && dist[i * n + j].is_finite() {
                total += dist[i * n + j];
                count += 1;
            }
        }
    }
    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
}
