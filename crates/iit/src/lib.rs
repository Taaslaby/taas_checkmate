//! # taas-iit — Integrated Information Measure
//!
//! этот модуль не является точным математическим или научным инструментом
//! this module is not an accurate mathematical or scientific tool
//! this module is an aproximation of the integrated information measure
//! this module is not a medical or clinical tool
//! This is not a conscious entity
//! Two Φ calculators:
//! 1. **Proxy Φ** = C × D × I (connectivity × diversity × integration) — fast heuristic.
//! 2. **IIT 2.0 Φ** via covariance MIP — grounded in
//!    Barrett & Seth (2011), measures integrated information as information loss
//!    when the system is partitioned at its MIP.
//!
//! **ВАЖНО**: Φ — мера интегрированной информации (Barrett & Seth 2011),
//! а не сознания. IIT 3.0 — активная область исследований, и proxy Φ особенно
//! далёк от клинической значимости.
//!
//! Мы не несем ответствености за результаты вычислений, а также за возможные
//! неверные выводы, к которым эти результаты могут привести.
//! ИИ не является сознательным.
//! не является медицинским или клиническим инструментом
//!
//! ## Quick start
//!
//! ```rust
//!    use taas_iit::{IITConsciousness, compute_phi_iit20, analyze_phi_iit20};
//!
//! // Proxy Φ
//! let mut iit = IITConsciousness::new(100, 0.3);
//! iit.compute(&[5.0f32; 1000], &[10.0f32; 100], 100, None);
//! println!("Φ = {:.4}, above_threshold: {}", iit.phi(), iit.is_above_threshold());
//!
//! // IIT 2.0 Φ via covariance MIP (more accurate)
//! let spikes = vec![0.0f32; 200 * 100]; // 200 steps × 100 neurons
//! let result = compute_phi_iit20(&spikes, 200, 100, 500);
//! println!("Φ_IIT2.0 = {:.4} (raw: {:.4})", result.phi, result.phi_raw);
//! ```
//!
//! Ported from Luma-brain IIT by @ant0n — @sardana

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]

/// Default threshold for proxy Φ.
pub const INTEGRATION_THRESHOLD: f32 = 0.3;

/// History window size.
pub const PHI_HISTORY_SIZE: usize = 100;

/// Default number of random bipartitions for MIP search.
pub const DEFAULT_N_PARTITIONS: usize = 500;

// ═══════════════════════════════════════════════════════════
//  Proxy Φ components: C × D × I
// ═══════════════════════════════════════════════════════════

/// Connectivity: mean absolute synaptic weight, normalized to [0, 1].
pub fn compute_connectivity(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.01;
    }
    let mean_w: f32 = values.iter().map(|&v| v.abs()).sum::<f32>() / values.len() as f32;
    (mean_w / 10.0).clamp(0.01, 1.0)
}

/// Diversity: normalized spike entropy from firing rates.
pub fn compute_diversity(spike_counts: &[f32], total_steps: f32, h_target: f32) -> f32 {
    let n = spike_counts.len();
    if n == 0 {
        return 0.05;
    }

    let (entropy, _penalty) = taas_kernels::spike_entropy(spike_counts, total_steps, h_target);

    let max_entropy = (n as f32).log2().max(1.0);
    let diversity = (entropy / max_entropy).min(1.0);

    if diversity > 0.01 {
        diversity
    } else {
        // Fallback: coefficient of variation of firing rates
        let mean_rate = spike_counts.iter().sum::<f32>() / n as f32;
        if mean_rate > 1e-8 {
            let variance: f32 = spike_counts
                .iter()
                .map(|&r| (r - mean_rate).powi(2))
                .sum::<f32>()
                / n as f32;
            let cv = variance.sqrt() / mean_rate;
            cv.clamp(0.01, 1.0)
        } else {
            0.05
        }
    }
}

/// Integration: Kuramoto order parameter R.
pub fn compute_integration_kuramoto(phases: &[f32]) -> f32 {
    if phases.is_empty() {
        return 0.05;
    }
    let n = phases.len() as f32;
    let sin_sum: f32 = phases.iter().map(|&p| p.sin()).sum();
    let cos_sum: f32 = phases.iter().map(|&p| p.cos()).sum();
    let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
    r.max(0.01)
}

/// Integration: Pearson correlation between two halves of the network.
pub fn compute_integration_pearson(spikes_a: &[f32], spikes_b: &[f32]) -> f32 {
    let n = spikes_a.len().min(spikes_b.len());
    if n < 2 {
        return 0.05;
    }
    let mean_a = spikes_a.iter().sum::<f32>() / n as f32;
    let mean_b = spikes_b.iter().sum::<f32>() / n as f32;
    let (cov, var_a, var_b) = (0..n).fold((0.0f32, 0.0f32, 0.0f32), |(c, va, vb), i| {
        let da = spikes_a[i] - mean_a;
        let db = spikes_b[i] - mean_b;
        (c + da * db, va + da * da, vb + db * db)
    });
    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.05;
    }
    let r = cov / (var_a * var_b).sqrt();
    r.abs().clamp(0.01, 1.0)
}

/// Integration: from spike count variability (CV of firing rates).
pub fn compute_integration_from_counts(spike_counts: &[f32]) -> f32 {
    let n = spike_counts.len();
    if n < 2 {
        return 0.5;
    }
    let mean = spike_counts.iter().sum::<f32>() / n as f32;
    if mean < 1e-8 {
        return 0.05;
    }
    let variance: f32 = spike_counts
        .iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f32>()
        / n as f32;
    let cv = variance.sqrt() / mean;
    cv.clamp(0.01, 1.0)
}

/// Proxy Φ = connectivity × diversity × integration.
pub fn compute_phi(connectivity: f32, diversity: f32, integration: f32) -> f32 {
    connectivity * diversity * integration
}

// ═══════════════════════════════════════════════════════════
//  IITConsciousness — stateful proxy Φ engine
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct IITConsciousness {
    pub n: usize,
    pub threshold: f32,
    pub phi: f32,
    pub connectivity: f32,
    pub diversity: f32,
    pub integration: f32,
    pub phi_history: Vec<f32>,
    pub conn_history: Vec<f32>,
    pub div_history: Vec<f32>,
    pub int_history: Vec<f32>,
    pub above_threshold_moments: usize,
    pub total_steps: usize,
}

impl IITConsciousness {
    pub fn new(n: usize, threshold: f32) -> Self {
        Self {
            n,
            threshold,
            phi: 0.0,
            connectivity: 0.0,
            diversity: 0.0,
            integration: 0.0,
            phi_history: Vec::with_capacity(PHI_HISTORY_SIZE),
            conn_history: Vec::with_capacity(PHI_HISTORY_SIZE),
            div_history: Vec::with_capacity(PHI_HISTORY_SIZE),
            int_history: Vec::with_capacity(PHI_HISTORY_SIZE),
            above_threshold_moments: 0,
            total_steps: 0,
        }
    }

    pub fn compute(
        &mut self,
        values: &[f32],
        spike_counts: &[f32],
        n_steps: usize,
        phases: Option<&[f32]>,
    ) -> f32 {
        if n_steps == 0 {
            return 0.0;
        }
        self.connectivity = compute_connectivity(values);
        self.diversity = compute_diversity(spike_counts, n_steps as f32, 0.5);
        self.integration = match phases {
            Some(p) => compute_integration_kuramoto(p),
            None => compute_integration_from_counts(spike_counts),
        };
        self.phi = compute_phi(self.connectivity, self.diversity, self.integration);
        self.push_history();
        self.phi
    }

    pub fn compute_with_spikes(
        &mut self,
        values: &[f32],
        spike_matrix: &[f32],
        n_steps: usize,
        n_neurons: usize,
    ) -> f32 {
        if n_steps == 0 || n_neurons == 0 {
            return 0.0;
        }
        self.connectivity = compute_connectivity(values);

        let mut spike_counts = vec![0.0f32; n_neurons];
        for t in 0..n_steps {
            for ni in 0..n_neurons {
                spike_counts[ni] += spike_matrix[t * n_neurons + ni];
            }
        }
        self.diversity = compute_diversity(&spike_counts, n_steps as f32, 0.5);

        let half = n_neurons / 2;
        self.integration = if half >= 1 && n_steps >= 1 {
            let n_pairs = n_steps * half;
            let mut a = Vec::with_capacity(n_pairs);
            let mut b = Vec::with_capacity(n_pairs);
            for t in 0..n_steps {
                let row_offset = t * n_neurons;
                for hi in 0..half {
                    a.push(spike_matrix[row_offset + hi]);
                    b.push(spike_matrix[row_offset + half + hi]);
                }
            }
            compute_integration_pearson(&a, &b)
        } else {
            0.05
        };

        self.phi = compute_phi(self.connectivity, self.diversity, self.integration);
        self.push_history();
        self.phi
    }

    fn push_history(&mut self) {
        self.phi_history.push(self.phi);
        if self.phi_history.len() > PHI_HISTORY_SIZE {
            self.phi_history.remove(0);
        }
        self.conn_history.push(self.connectivity);
        if self.conn_history.len() > PHI_HISTORY_SIZE {
            self.conn_history.remove(0);
        }
        self.div_history.push(self.diversity);
        if self.div_history.len() > PHI_HISTORY_SIZE {
            self.div_history.remove(0);
        }
        self.int_history.push(self.integration);
        if self.int_history.len() > PHI_HISTORY_SIZE {
            self.int_history.remove(0);
        }
        self.total_steps += 1;
        if self.is_above_threshold() {
            self.above_threshold_moments += 1;
        }
    }

    pub fn set_integration(&mut self, val: f32) {
        self.integration = val.clamp(0.01, 1.0);
        self.phi = compute_phi(self.connectivity, self.diversity, self.integration);
    }

    pub fn phi(&self) -> f32 {
        self.phi
    }
    pub fn is_above_threshold(&self) -> bool {
        self.phi > self.threshold
    }
    pub fn normalized_level(&self) -> f32 {
        (self.phi / self.threshold).min(1.0)
    }

    pub fn mean_phi(&self, n: usize) -> f32 {
        let len = self.phi_history.len();
        if len == 0 {
            return 0.0;
        }
        let start = len.saturating_sub(n);
        self.phi_history[start..].iter().sum::<f32>() / (len - start) as f32
    }

    pub fn phi_trend(&self) -> String {
        let len = self.phi_history.len();
        if len < 20 {
            return "накопление".to_string();
        }
        let recent: f32 = self.phi_history[len - 10..].iter().sum::<f32>() / 10.0;
        let prev: f32 = self.phi_history[len - 20..len - 10].iter().sum::<f32>() / 10.0;
        let diff = recent - prev;
        if diff > 0.02 {
            format!("📈 растёт ({:+.3})", diff)
        } else if diff < -0.02 {
            format!("📉 падает ({:+.3})", diff)
        } else {
            format!("➡ стабилен ({:+.3})", diff)
        }
    }

    pub fn stats(&self) -> IITStats {
        let mean = if !self.phi_history.is_empty() {
            self.phi_history.iter().sum::<f32>() / self.phi_history.len() as f32
        } else {
            0.0
        };
        let std = if self.phi_history.len() > 1 {
            let variance: f32 = self
                .phi_history
                .iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f32>()
                / self.phi_history.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };
        IITStats {
            phi: self.phi,
            phi_mean: mean,
            phi_std: std,
            connectivity: self.connectivity,
            diversity: self.diversity,
            integration: self.integration,
            normalized_level: self.normalized_level(),
            threshold_ratio: if self.total_steps > 0 {
                self.above_threshold_moments as f32 / self.total_steps as f32
            } else {
                0.0
            },
            trend: self.phi_trend(),
        }
    }

    pub fn reset(&mut self) {
        self.phi = 0.0;
        self.connectivity = 0.0;
        self.diversity = 0.0;
        self.integration = 0.0;
        self.phi_history.clear();
        self.conn_history.clear();
        self.div_history.clear();
        self.int_history.clear();
        self.above_threshold_moments = 0;
        self.total_steps = 0;
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct IITStats {
    pub phi: f32,
    pub phi_mean: f32,
    pub phi_std: f32,
    pub connectivity: f32,
    pub diversity: f32,
    pub integration: f32,
    pub normalized_level: f32,
    pub threshold_ratio: f32,
    pub trend: String,
}

// ═══════════════════════════════════════════════════════════
//  Φ_IIT 2.0 — covariance MIP (minimum information partition)
//  Barrett & Seth (2011): Practical measures of integrated information
// ═══════════════════════════════════════════════════════════

/// Compute covariance matrix from spike data.
///
/// Returns flattened N×N sample covariance matrix (row-major).
fn covariance_matrix(data: &[f32], n_steps: usize, n: usize) -> Vec<f32> {
    let mut cov = vec![0.0f32; n * n];

    // 1. Means
    let mut means = vec![0.0f32; n];
    for t in 0..n_steps {
        let offset = t * n;
        for i in 0..n {
            means[i] += data[offset + i];
        }
    }
    let inv_n = 1.0 / n_steps as f32;
    for i in 0..n {
        means[i] *= inv_n;
    }

    // 2. Covariance (sample: divide by n-1)
    let inv_nm1 = 1.0 / (n_steps as f32 - 1.0);
    for t in 0..n_steps {
        let offset = t * n;
        for i in 0..n {
            let di = data[offset + i] - means[i];
            for j in 0..n {
                cov[i * n + j] += di * (data[offset + j] - means[j]);
            }
        }
    }
    for val in cov.iter_mut() {
        *val *= inv_nm1;
    }

    cov
}

/// Compute log-determinant via LU decomposition with partial pivoting.
///
/// Returns (log|det|, sign). Uses f64 internally to avoid underflow.
fn log_determinant(matrix: &[f32], n: usize) -> (f64, f32) {
    if n == 0 {
        return (0.0, 1.0);
    }
    if n == 1 {
        let val = matrix[0] as f64;
        return (val.abs().ln(), matrix[0].signum());
    }

    let mut lu: Vec<f64> = matrix.iter().map(|&x| x as f64).collect();
    let mut sign = 1.0f64;

    for k in 0..n - 1 {
        // Partial pivoting
        let mut max_abs = lu[k * n + k].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let abs_val = lu[i * n + k].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_row = i;
            }
        }

        if max_abs < 1e-100 {
            return (f64::NEG_INFINITY, 0.0);
        }

        if max_row != k {
            for j in k..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            sign = -sign;
        }

        // Eliminate
        let pivot = lu[k * n + k];
        for i in k + 1..n {
            let factor = lu[i * n + k] / pivot;
            for j in k + 1..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    let mut log_det = 0.0f64;
    for i in 0..n {
        log_det += lu[i * n + i].abs().ln();
    }

    (log_det, sign as f32)
}

/// Compute Φ for a specific bipartition using log-determinants.
///
/// Φ_part = 0.5 · max(ln(|Σ_AA| · |Σ_BB| / |Σ_full|), 0)
///        = 0.5 · max(log_det_AA + log_det_BB - log_det_full, 0)
fn phi_for_subset(cov_full: &[f32], subset: &[usize], n: usize, log_det_full: f64) -> f64 {
    let k = subset.len();
    let m = n - k;
    if k == 0 || m == 0 {
        return -1.0;
    }

    let mut in_a = vec![false; n];
    for &idx in subset {
        in_a[idx] = true;
    }
    let b_indices: Vec<usize> = (0..n).filter(|&i| !in_a[i]).collect();

    // Extract Σ_AA
    let mut cov_aa = vec![0.0f32; k * k];
    for i in 0..k {
        let ai = subset[i];
        for j in 0..k {
            cov_aa[i * k + j] = cov_full[ai * n + subset[j]];
        }
    }

    // Extract Σ_BB
    let mut cov_bb = vec![0.0f32; m * m];
    for i in 0..m {
        let bi = b_indices[i];
        for j in 0..m {
            cov_bb[i * m + j] = cov_full[bi * n + b_indices[j]];
        }
    }

    let (log_det_aa, _) = log_determinant(&cov_aa, k);
    let (log_det_bb, _) = log_determinant(&cov_bb, m);

    if log_det_aa.is_infinite() || log_det_bb.is_infinite() {
        return -1.0;
    }

    // Φ = 0.5 * max(log|Σ_AA| + log|Σ_BB| - log|Σ_full|, 0)
    0.5 * (log_det_aa + log_det_bb - log_det_full).max(0.0)
}

/// Deterministic LCG random number generator.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u32() as usize) % max
    }
}

/// Result of Φ IIT 2.0 computation via covariance MIP.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhiIIT20Result {
    /// Normalized Φ in [0, 1): Φ_norm = 1 - exp(-Φ_raw)
    pub phi: f32,
    /// Raw Φ = 0.5 · max(log|Σ_AA| + log|Σ_BB| - log|Σ_full|, 0)
    /// Higher → more integrated information.
    pub phi_raw: f32,
    /// log|Σ_full| — log-determinant of full covariance
    pub log_det_full: f64,
    /// log|Σ_AA| — log-determinant of partition A covariance
    pub log_det_aa: f64,
    /// log|Σ_BB| — log-determinant of partition B covariance
    pub log_det_bb: f64,
    /// Size of partition A (number of neurons in first subset)
    pub partition_size_a: usize,
    /// Total number of partitions evaluated
    pub n_evaluated: usize,
}

/// Perturb a subset by swapping `n_swap` elements between A and its complement.
///
/// Keeps the same |A| approximately. Returns new subset sorted.
fn perturb_subset(subset: &[usize], n: usize, n_swap: usize, rng: &mut Lcg) -> Vec<usize> {
    let mut in_set = vec![false; n];
    for &idx in subset {
        in_set[idx] = true;
    }

    let in_set_idx: Vec<usize> = (0..n).filter(|&i| in_set[i]).collect();
    let out_set_idx: Vec<usize> = (0..n).filter(|&i| !in_set[i]).collect();

    let actual_swap = n_swap.min(in_set_idx.len()).min(out_set_idx.len());
    if actual_swap == 0 {
        return subset.to_vec();
    }

    // Randomly pick `actual_swap` from each side
    let mut remove = in_set_idx;
    for i in 0..actual_swap {
        let j = i + rng.next_usize(remove.len() - i);
        remove.swap(i, j);
    }

    let mut add = out_set_idx;
    for i in 0..actual_swap {
        let j = i + rng.next_usize(add.len() - i);
        add.swap(i, j);
    }

    // Swap
    for &idx in &remove[..actual_swap] {
        in_set[idx] = false;
    }
    for &idx in &add[..actual_swap] {
        in_set[idx] = true;
    }

    (0..n).filter(|&i| in_set[i]).collect()
}

/// Compute Φ via IIT 2.0 covariance MIP using hierarchical bisection.
///
/// ## Algorithm
///
/// 1. Compute spike covariance matrix Σ (N×N).
/// 2. Start with midline split A = [0..n/2], B = [n/2..n].
/// 3. **Hierarchical refinement:** iteratively swap small groups of neurons
///    between A and B. Accept swaps that **reduce** Φ (better MIP = smaller
///    integrated information when cut this way). Increase swap size over
///    iterations (start narrow, widen to escape local minima).
/// 4. **Random restarts** for diversity (when budget allows).
/// 5. Φ_raw = min over all evaluated partitions (MIP).
/// 6. Φ_norm = 1 - exp(-Φ_raw) maps [0, ∞) → [0, 1).
///
/// Hierarchical bisection is **more sample-efficient** than pure random
/// search: it exploits the fact that the MIP landscape is locally smooth,
/// so nearby partitions have similar Φ values.
///
/// Reference: Barrett & Seth (2011), PLoS Comp Bio.
pub fn compute_phi_iit20(
    spike_matrix: &[f32],
    n_steps: usize,
    n_neurons: usize,
    n_partitions: usize,
) -> PhiIIT20Result {
    let n = n_neurons;

    if n_steps < 2 || n < 2 {
        return PhiIIT20Result {
            phi: 0.0,
            phi_raw: 0.0,
            log_det_full: 0.0,
            log_det_aa: 0.0,
            log_det_bb: 0.0,
            partition_size_a: n / 2,
            n_evaluated: 0,
        };
    }

    // 1. Covariance matrix
    let cov = covariance_matrix(spike_matrix, n_steps, n);

    // 2. Log-determinant of full covariance
    let (log_det_full, _) = log_determinant(&cov, n);

    let mut rng = Lcg::new(42);
    let half = n / 2;
    let mut min_phi_raw = f64::INFINITY;
    let mut best_size = half;
    let mut evaluated = 0usize;

    // Inline helper to avoid closure borrow issues
    macro_rules! try_subset {
        ($subset:expr) => {{
            let s: &[usize] = $subset;
            let phi = phi_for_subset(&cov, s, n, log_det_full);
            evaluated += 1;
            if phi >= 0.0 && phi < min_phi_raw {
                min_phi_raw = phi;
                best_size = s.len();
                true
            } else {
                false
            }
        }};
    }

    // ═══════════════════════════════════════════════════════════
    //  Phase 1: Midline split
    // ═══════════════════════════════════════════════════════════
    if half >= 1 {
        let midline: Vec<usize> = (0..half).collect();
        try_subset!(&midline);
    }

    // ═══════════════════════════════════════════════════════════
    //  Phase 2: Hierarchical bisection refinement (60% of budget)
    // ═══════════════════════════════════════════════════════════
    let n_refine = (n_partitions as f64 * 0.6).ceil() as usize;
    let mut current_best: Vec<usize> = (0..half).collect();
    let mut stall_count = 0usize;

    for iter in 0..n_refine {
        let swap_frac = 0.02 + 0.13 * (iter as f64 / n_refine.max(1) as f64);
        let n_swap = ((n as f64 * swap_frac).ceil() as usize).max(1);

        let candidate = perturb_subset(&current_best, n, n_swap, &mut rng);
        let improved = try_subset!(&candidate);

        if improved {
            current_best = candidate;
            stall_count = 0;
        } else {
            stall_count += 1;
            if stall_count > 10 {
                let big_swap = ((n as f64 * 0.25).ceil() as usize).max(2);
                let candidate2 = perturb_subset(&current_best, n, big_swap, &mut rng);
                try_subset!(&candidate2);
                stall_count = 0; // full reset after big jump
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Phase 3: Random restarts (budget = 0.4× n_partitions)
    // ═══════════════════════════════════════════════════════════
    let n_random = n_partitions.saturating_sub(evaluated);

    for _ in 0..n_random {
        let min_k = (n as f64 * 0.25) as usize;
        let max_k = (n as f64 * 0.75).min((n - 1) as f64) as usize;
        let k = if max_k > min_k {
            min_k + rng.next_usize(max_k - min_k + 1)
        } else {
            n / 2
        };
        if k == 0 || k >= n {
            continue;
        }

        let mut perm: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + rng.next_usize(n - i);
            perm.swap(i, j);
        }

        try_subset!(&perm[..k]);
    }

    if min_phi_raw.is_infinite() {
        // System is fully deterministic (log|Σ| = -inf)
        // → maximal integration: Φ_norm ≈ 1.0
        min_phi_raw = 10.0;
    }

    let phi_norm = 1.0 - (-min_phi_raw as f32).exp();

    PhiIIT20Result {
        phi: phi_norm,
        phi_raw: min_phi_raw as f32,
        log_det_full,
        log_det_aa: 0.0,
        log_det_bb: 0.0,
        partition_size_a: best_size,
        n_evaluated: evaluated,
    }
}

/// Full IIT analysis: proxy Φ and IIT 2.0 Φ on the same data.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhiComparison {
    pub phi_iit20: f32,
    pub phi_proxy: f32,
    pub connectivity: f32,
    pub diversity: f32,
    pub integration: f32,
    pub phi_iit20_raw: f32,
    pub iit20_result: PhiIIT20Result,
}

pub fn analyze_phi_iit20(
    values: &[f32],
    spike_matrix: &[f32],
    n_steps: usize,
    n_neurons: usize,
    n_partitions: usize,
) -> PhiComparison {
    let c = compute_connectivity(values);

    let mut spike_counts = vec![0.0f32; n_neurons];
    for t in 0..n_steps {
        for ni in 0..n_neurons {
            spike_counts[ni] += spike_matrix[t * n_neurons + ni];
        }
    }
    let d = compute_diversity(&spike_counts, n_steps as f32, 0.5);

    let half = n_neurons / 2;
    let i = if half >= 1 && n_steps >= 1 {
        let n_pairs = n_steps * half;
        let mut a = Vec::with_capacity(n_pairs);
        let mut b = Vec::with_capacity(n_pairs);
        for t in 0..n_steps {
            let row_offset = t * n_neurons;
            for hi in 0..half {
                a.push(spike_matrix[row_offset + hi]);
                b.push(spike_matrix[row_offset + half + hi]);
            }
        }
        compute_integration_pearson(&a, &b)
    } else {
        0.05
    };

    let phi_proxy = compute_phi(c, d, i);
    let iit20 = compute_phi_iit20(spike_matrix, n_steps, n_neurons, n_partitions);

    PhiComparison {
        phi_iit20: iit20.phi,
        phi_proxy,
        connectivity: c,
        diversity: d,
        integration: i,
        phi_iit20_raw: iit20.phi_raw,
        iit20_result: iit20,
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connectivity_empty() {
        let c = compute_connectivity(&[]);
        assert!((c - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_connectivity_uniform() {
        let c = compute_connectivity(&[5.0; 100]);
        assert!((c - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_connectivity_clamp() {
        let c = compute_connectivity(&[100.0; 50]);
        assert!((c - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_diversity_uniform() {
        let d = compute_diversity(&[10.0; 100], 100.0, 0.5);
        assert!(d > 0.5, "uniform -> high, got {d}");
    }

    #[test]
    fn test_diversity_varied() {
        let mut counts = vec![0.0; 100];
        for i in 0..50 {
            counts[i] = 50.0;
        }
        for i in 50..100 {
            counts[i] = 1.0;
        }
        let d = compute_diversity(&counts, 100.0, 0.5);
        assert!(d > 0.3, "varied -> >0.3, got {d}");
    }

    #[test]
    fn test_integration_kuramoto_sync() {
        let r = compute_integration_kuramoto(&[0.0; 100]);
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_phi_product() {
        let phi = compute_phi(0.5, 0.5, 0.5);
        assert!((phi - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_iit_basic() {
        let mut iit = IITConsciousness::new(100, 0.3);
        iit.compute(&[5.0; 1000], &[0.0; 100], 100, None);
        assert!(iit.phi() > 0.0);
        assert!(!iit.is_above_threshold());
    }

    #[test]
    fn test_integration_pearson_perfect() {
        let a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = compute_integration_pearson(&a, &b);
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_iit_reset() {
        let mut iit = IITConsciousness::new(100, 0.3);
        iit.compute(&[3.0; 1000], &[20.0; 100], 100, None);
        assert_eq!(iit.total_steps, 1);
        iit.reset();
        assert_eq!(iit.total_steps, 0);
        assert!(iit.phi_history.is_empty());
    }

    // ═══ IIT 2.0 tests ═══════════════════════════════════

    #[test]
    fn test_log_determinant_1x1() {
        let (log_det, sign) = log_determinant(&[5.0], 1);
        assert!((log_det - 5.0f64.ln()).abs() < 1e-6);
        assert!((sign - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_determinant_2x2() {
        // [[1, 2], [3, 4]] → det = -2
        let (log_det, sign) = log_determinant(&[1.0, 2.0, 3.0, 4.0], 2);
        let actual_det = sign * log_det.exp() as f32;
        assert!((actual_det - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_log_determinant_3x3() {
        // [[3, 0, 1], [2, 1, 0], [1, 0, 2]] → det = 5
        let mat = vec![3.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0];
        let (log_det, sign) = log_determinant(&mat, 3);
        let actual_det = sign * log_det.exp() as f32;
        assert!((actual_det - 5.0).abs() < 1e-4, "det=5, got {actual_det}");
    }

    #[test]
    fn test_log_determinant_identity() {
        let mut mat = vec![0.0f32; 10 * 10];
        for i in 0..10 {
            mat[i * 10 + i] = 1.0;
        }
        let (log_det, sign) = log_determinant(&mat, 10);
        let actual_det = sign * log_det.exp() as f32;
        assert!((actual_det - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_covariance_simple() {
        let data = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let cov = covariance_matrix(&data, 3, 2);
        assert!((cov[0] - cov[3]).abs() < 0.01, "var equal");
        assert!(cov[1] > 0.0 && cov[2] > 0.0, "cov pos");
    }

    #[test]
    fn test_phi_iit20_independent() {
        // 2 independent neurons (coprime periods 3 and 7 → cov ≈ 0) → Φ ≈ 0
        let n_steps = 2000; // many steps for stable estimates
        let n = 2;
        let mut data = vec![0.0f32; n_steps * n];
        for t in 0..n_steps {
            if t % 3 == 0 {
                data[t * n + 0] = 1.0;
            } // period 3
            if t % 7 == 0 {
                data[t * n + 1] = 1.0;
            } // period 7 (coprime → independent)
        }
        let result = compute_phi_iit20(&data, n_steps, n, 100);
        println!(
            "Independent (coprime): Φ={:.4}, raw={:.4}",
            result.phi, result.phi_raw
        );
        assert!(result.phi < 0.3, "independent -> low Φ, got {}", result.phi);
    }

    #[test]
    fn test_phi_iit20_anti_correlated() {
        // 2 perfectly anti-correlated neurons → deterministic relation → high Φ
        let n_steps = 200;
        let n = 2;
        let mut data = vec![0.0f32; n_steps * n];
        for t in 0..n_steps {
            if t % 2 == 0 {
                data[t * n + 0] = 1.0;
            } else {
                data[t * n + 1] = 1.0;
            }
        }
        let result = compute_phi_iit20(&data, n_steps, n, 100);
        println!(
            "Anti-correlated: Φ={:.4}, raw={:.4}",
            result.phi, result.phi_raw
        );
        assert!(
            result.phi > 0.5,
            "anti-correlated -> high Φ, got {:.4}",
            result.phi
        );
    }

    #[test]
    fn test_phi_iit20_sync() {
        // 2 strongly correlated (but not identical) → high Φ
        // Perfectly identical spikes create singular covariance (det=0),
        // which makes log|Σ_full| = -inf. Small jitter prevents singularity.
        let n_steps = 200;
        let n = 2;
        let mut data = vec![0.0f32; n_steps * n];
        for t in 0..n_steps {
            let val = if t % 2 == 0 { 1.0 } else { 0.0 };
            data[t * n + 0] = val;
            // Neuron 1: 99% correlated with neuron 0
            data[t * n + 1] = if t % 7 == 0 { 1.0 - val } else { val };
        }
        let result = compute_phi_iit20(&data, n_steps, n, 100);
        println!(
            "Synchronous: Φ={:.4}, raw={:.4}, n_eval={}",
            result.phi, result.phi_raw, result.n_evaluated
        );
        assert!(result.phi > 0.1, "sync -> high Φ, got {:.4}", result.phi);
    }

    #[test]
    fn test_phi_iit20_larger() {
        // 10 neurons: two internally-synchronized groups → non-trivial Φ
        // Add small jitter to prevent singular covariance within each group
        let n_steps = 300;
        let n = 10;
        let mut data = vec![0.0f32; n_steps * n];
        for t in 0..n_steps {
            let g1 = if (t / 5) % 2 == 0 { 1.0 } else { 0.0 };
            let g2 = if (t / 3) % 2 == 0 { 1.0 } else { 0.0 };
            for i in 0..5 {
                let jitter = if (t + i) % 11 == 0 { 0.0 } else { 1.0 };
                data[t * n + i] = g1 * jitter;
            }
            for i in 5..10 {
                let jitter = if (t + i) % 7 == 0 { 0.0 } else { 1.0 };
                data[t * n + i] = g2 * jitter;
            }
        }
        let result = compute_phi_iit20(&data, n_steps, n, 200);
        println!(
            "10n 2-group: Φ={:.4}, raw={:.4}, n_eval={}",
            result.phi, result.phi_raw, result.n_evaluated
        );
        assert!(
            result.n_evaluated >= 50,
            "need >=50 evaluations, got {}",
            result.n_evaluated
        );
    }

    #[test]
    fn test_analyze_phi_iit20() {
        let n = 10;
        let n_steps = 200;
        let values = vec![5.0f32; 100];

        let mut spike_matrix = vec![0.0f32; n_steps * n];
        for t in 0..n_steps {
            let val = if t % 2 == 0 { 1.0 } else { 0.0 };
            for i in 0..n {
                spike_matrix[t * n + i] = val;
            }
        }

        let comp = analyze_phi_iit20(&values, &spike_matrix, n_steps, n, 100);
        println!(
            "Analysis: proxy Φ={:.4}, IIT2.0 Φ={:.4}, C={:.4}, D={:.4}, I={:.4}",
            comp.phi_proxy, comp.phi_iit20, comp.connectivity, comp.diversity, comp.integration
        );
        assert!(comp.phi_proxy > 0.0);
        assert!(comp.phi_iit20 >= 0.0);
        assert!(comp.connectivity > 0.0);
        assert!(comp.diversity > 0.0);
    }
}
