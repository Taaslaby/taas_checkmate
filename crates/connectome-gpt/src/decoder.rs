use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::dataset::ConnectomeSample;
use crate::graph_vae::{GraphVAE, VaeConfig};
use crate::refiner::{ConnectomeRefiner, RefinerConfig};
use crate::validate::Validator;

/// Configuration for the connectome decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub n_neurons: usize,
    pub density: f32,
    pub ei_ratio: f32,
    pub mean_weight: f32,
    pub weight_noise: f32,
    /// Number of communities (stochastic block model). 0 → no structure.
    pub n_communities: usize,
    /// Within-to-between community edge ratio.
    pub community_strength: f32,
    /// Fraction of neurons assigned as descending neurons.
    pub dn_fraction: f32,
    /// Fraction assigned as motor neurons.
    pub mn_fraction: f32,
    /// Fraction assigned as sensory-afferent.
    pub sensory_fraction: f32,
    /// Use STDP-inspired heavy-tailed weight distribution.
    pub stdp_weights: bool,
    /// Compute topological depth (sensory→central→motor axis).
    pub compute_depth: bool,
    /// Connectome refiner epochs (0 = skip).
    pub refine_epochs: usize,
    pub seed: u64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            n_neurons: 1000,
            density: 0.01,
            ei_ratio: 4.0,
            mean_weight: 0.5,
            weight_noise: 0.1,
            n_communities: 8,
            community_strength: 10.0,
            dn_fraction: 0.02,
            mn_fraction: 0.005,
            sensory_fraction: 0.05,
            stdp_weights: true,
            compute_depth: true,
            refine_epochs: 0,
            seed: 42,
        }
    }
}

/// A high-level wrapper that runs the full brain generation pipeline:
/// VAE → decode → refine → validate.
pub struct BrainGenerator {
    pub vae: GraphVAE,
    pub decoder: CsrDecoder,
    pub refiner_cfg: RefinerConfig,
}

impl BrainGenerator {
    pub fn new(
        vae: GraphVAE,
        decoder_cfg: DecoderConfig,
        refiner_cfg: RefinerConfig,
    ) -> Self {
        Self {
            vae,
            decoder: CsrDecoder::new(decoder_cfg),
            refiner_cfg,
        }
    }

    /// Create with VAE weights baked into the binary.
    /// Uses `GraphVAE::from_embedded()` — zero file I/O at runtime.
    pub fn from_embedded(
        vae_config: VaeConfig,
        decoder_cfg: DecoderConfig,
        refiner_cfg: RefinerConfig,
    ) -> Self {
        Self {
            vae: GraphVAE::from_embedded(vae_config),
            decoder: CsrDecoder::new(decoder_cfg),
            refiner_cfg,
        }
    }

    /// Generate a brain: VAE sample → decode → optionally refine.
    pub fn generate(&mut self) -> ConnectomeSample {
        let features = self.vae.generate();
        let mut sample = self.decoder.generate(&features);
        if self.decoder.config.refine_epochs > 0 {
            let mut refiner = ConnectomeRefiner::new(self.refiner_cfg.clone());
            let (refined, _) = refiner.refine(&sample, self.decoder.config.refine_epochs);
            sample = refined;
        }
        sample
    }

    /// Generate and validate.
    pub fn generate_with_validation(&mut self) -> (ConnectomeSample, ValidationResult) {
        let sample = self.generate();
        let val = Validator::default();
        let result = val.validate(&sample);
        (sample, result)
    }
}

use crate::validate::ValidationResult;

/// Generates CSR connectomes from VAE feature vectors using a hierarchical
/// stochastic block model with biological group structure and optional
/// Kuramoto BPTT refinement.
pub struct CsrDecoder {
    pub config: DecoderConfig,
    rng: StdRng,
}

impl CsrDecoder {
    pub fn new(config: DecoderConfig) -> Self {
        Self { rng: StdRng::seed_from_u64(config.seed), config }
    }

    pub fn generate(&mut self, features: &[f32]) -> ConnectomeSample {
        let cfg = &self.config;
        let n = cfg.n_neurons;
        if n == 0 {
            return empty_sample(features);
        }

        // ── 1. Neuron types (E/I) ──
        let ei_ratio = cfg.ei_ratio.max(0.1);
        let n_exc = (n as f32 * ei_ratio / (ei_ratio + 1.0)).round() as usize;
        let n_exc = n_exc.max(1).min(n - 1);
        let mut nt_types = vec![1.0f32; n];
        for i in n_exc..n { nt_types[i] = -1.0; }
        for i in (1..n).rev() { let j = self.rng.gen_range(0..=i); nt_types.swap(i, j); }

        // ── 2. Communities (SBM) ──
        let n_com = cfg.n_communities.max(1);
        let mut community = vec![0usize; n];
        for i in 0..n { community[i] = i % n_com; }

        // ── 3. Biological groups ──
        let (is_sensory, is_dn, is_mn, is_vnc) = assign_groups(
            n, cfg.sensory_fraction, cfg.dn_fraction, cfg.mn_fraction, &mut self.rng,
        );

        let mut groups: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        let all_idx: Vec<usize> = (0..n).collect();
        groups.insert("sensory".into(), all_idx.iter().copied().filter(|i| is_sensory[*i]).collect());
        groups.insert("dn".into(), all_idx.iter().copied().filter(|i| is_dn[*i]).collect());
        groups.insert("mn".into(), all_idx.iter().copied().filter(|i| is_mn[*i]).collect());
        groups.insert("vnc".into(), all_idx.iter().copied().filter(|i| is_vnc[*i]).collect());

        // ── 4. Edge probabilities ──
        let s = cfg.community_strength.max(1.0);
        let p_out = (cfg.density * n_com as f32 / (s + n_com as f32 - 1.0)).min(1.0);
        let p_in = (p_out * s).min(1.0);
        let baseline_deg = (cfg.density * n as f32).max(0.5);
        let accept_scale = (n as f32 - 1.0) / baseline_deg;

        fn gm(pre_sens: bool, pre_dn: bool, post_dn: bool, post_mn: bool, post_vnc: bool) -> f32 {
            if pre_sens && post_dn { 3.0 } else if pre_dn && post_mn { 4.0 }
            else if pre_dn && post_vnc { 2.5 } else if pre_sens && post_vnc { 2.0 }
            else if pre_dn && post_dn { 0.3 } else { 1.0 }
        }

        let target_deg = baseline_deg.round() as usize;

        // ── 5. Build CSR (O(N × deg) via Floyd sampling) ──
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        // track weight sum for reciprocal correlation (STDP-like)
        let mut total_w_abs = 0.0f32;
        let mut n_syn = 0usize;

        // If using STDP-like weights, pre-generate weight pool
        // (log-normal for exc, broader for inh)
        let weight_pool = if cfg.stdp_weights {
            generate_stdp_weight_pool(n, &mut self.rng)
        } else {
            vec![]
        };

        for i in 0..n {
            let deg_scale = if self.rng.gen::<f32>() < 0.3 { 0.5 + self.rng.gen::<f32>() } else { 1.0 };
            let deg = ((target_deg as f32) * deg_scale).round() as usize;
            let deg = deg.max(1).min(n - 1);
            let ci = community[i];

            // Floyd sampling: O(deg) distinct random targets
            let mut targets = Vec::with_capacity(deg);
            let mut seen = std::collections::HashSet::new();
            seen.insert(i);
            while targets.len() < deg {
                let t = self.rng.gen_range(0..n);
                if seen.insert(t) { targets.push(t); }
            }

            for &t in &targets {
                let com_p = if community[t] == ci { p_in } else { p_out };
                let mod_g = gm(is_sensory[i], is_dn[i], is_dn[t], is_mn[t], is_vnc[t]);
                let accept = (com_p * mod_g * accept_scale).min(1.0);
                if self.rng.gen::<f32>() >= accept { continue; }

                let w = if cfg.stdp_weights {
                    // STDP-inspired: pick from log-normal pool, sign by nt_type
                    let idx = self.rng.gen_range(0..weight_pool.len());
                    let base = weight_pool[idx];
                    let sign = if nt_types[i] > 0.0 { 1.0 } else { -1.0 };
                    sign * (base + self.rng.gen::<f32>() * cfg.weight_noise).abs().max(0.01)
                } else {
                    let raw = self.rng.gen::<f32>() * cfg.mean_weight * 2.0;
                    let sign = if nt_types[i] > 0.0 { 1.0 } else { -1.0 };
                    sign * (raw + self.rng.gen::<f32>() * cfg.weight_noise).abs().max(0.01)
                };
                col_indices.push(t as i32);
                values.push(w);
                total_w_abs += w.abs();
                n_syn += 1;
            }

            row_ptr[i + 1] = col_indices.len() as i32;
        }

        // Normalize mean weight to target if STDP pool was used
        if cfg.stdp_weights && n_syn > 0 {
            let current_mean = total_w_abs / n_syn as f32;
            let scale = cfg.mean_weight / current_mean.max(1e-8);
            if (scale - 1.0).abs() > 0.01 {
                for v in &mut values { *v *= scale; }
            }
        }

        let features_vec = features.to_vec();
        // ── 6. Topological depth (BFS from sensory → motor axis) ──
        let depth = if cfg.compute_depth {
            Some(compute_topological_depth(
                n, &row_ptr, &col_indices,
                groups.get("sensory").map(|v| v.as_slice()).unwrap_or(&[]),
                groups.get("dn").map(|v| v.as_slice()).unwrap_or(&[]),
                groups.get("mn").map(|v| v.as_slice()).unwrap_or(&[]),
            ))
        } else {
            None
        };

        let mut sample = ConnectomeSample {
            source: "super_decoder".into(),
            n_neurons: n,
            n_synapses: values.len(),
            row_ptr, col_indices, values, nt_types,
            groups,
            features: features_vec,
            depth,
        };

        // ── 7. Optional Kuramoto BPTT refinement ──
        if cfg.refine_epochs > 0 {
            let refiner_cfg = RefinerConfig {
                n_steps: 50, dt: 0.05, lr: 0.001,
                omegas: vec![],
                coherence_weight: 1.0, sparsity_weight: 0.01,
                temporal_weight: 0.1, homeostasis_weight: 0.05,
                target_mean_weight: cfg.mean_weight,
                seed: self.rng.gen(),
                ..Default::default()
            };
            let mut refiner = ConnectomeRefiner::new(refiner_cfg);
            let (refined, _) = refiner.refine(&sample, cfg.refine_epochs);
            sample = refined;
        }

        sample
    }

    pub fn generate_from_vae(&mut self, vae: &mut GraphVAE) -> ConnectomeSample {
        self.generate(&vae.generate())
    }

    pub fn generate_conditioned(
        &mut self, vae: &mut GraphVAE,
        target: &[f32], mean: &[f32], std: &[f32],
        steps: usize,
    ) -> ConnectomeSample {
        let mut best = None;
        let mut best_score = f32::MAX;
        for _ in 0..steps {
            let f = vae.generate();
            let s = self.generate(&f);
            let m = crate::metrics::GraphMetrics::compute(&s);
            let actual = vec![
                m.n_neurons as f32, m.n_synapses as f32,
                m.density * 100.0, m.mean_degree,
                m.degree_variance.sqrt(), m.ei_ratio.min(10.0),
                m.mean_weight * 100.0,
                (m.n_synapses as f32 / m.n_neurons.max(1) as f32) / 10.0,
                m.n_groups as f32, m.small_world_index.min(5.0),
            ];
            let score: f32 = actual.iter().zip(target.iter())
                .zip(mean.iter().chain(std::iter::repeat(&0.0)))
                .zip(std.iter().chain(std::iter::repeat(&1.0)))
                .map(|(((a, t), m), s)| {
                    let an = if *s > 0.0 { (a - m) / s } else { *a };
                    let tn = if *s > 0.0 { (t - m) / s } else { *t };
                    (an - tn).powi(2)
                }).sum();
            if score < best_score { best_score = score; best = Some(s); }
        }
        best.unwrap_or_else(|| self.generate(&vae.generate()))
    }

    pub fn refine(&mut self, sample: &ConnectomeSample, epochs: usize) -> ConnectomeSample {
        if epochs == 0 { return sample.clone(); }
        let refiner_cfg = RefinerConfig {
            n_steps: 50, dt: 0.05, lr: 0.001,
            omegas: vec![],
            coherence_weight: 1.0, sparsity_weight: 0.01,
            temporal_weight: 0.1, homeostasis_weight: 0.05,
            target_mean_weight: self.config.mean_weight,
            seed: self.rng.gen(),
            ..Default::default()
        };
        let (refined, _) = ConnectomeRefiner::new(refiner_cfg).refine(sample, epochs);
        refined
    }

    pub fn generate_with_validation(&mut self, features: &[f32]) -> (ConnectomeSample, ValidationResult) {
        let s = self.generate(features);
        let r = Validator::default().validate(&s);
        (s, r)
    }
}

// ── helper: assign biological groups ──
fn assign_groups(
    n: usize, sensory_frac: f32, dn_frac: f32, mn_frac: f32,
    rng: &mut StdRng,
) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut is_s = vec![false; n];
    let mut is_d = vec![false; n];
    let mut is_m = vec![false; n];
    let mut is_v = vec![false; n];
    let n_s = (n as f32 * sensory_frac).round() as usize;
    let n_d = (n as f32 * dn_frac).round() as usize;
    let n_m = (n as f32 * mn_frac).round() as usize;
    let n_v = (n as f32 * 0.10).round() as usize;
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() { let j = rng.gen_range(0..=i); idx.swap(i, j); }
    let mut p = 0;
    for &i in &idx[p..p + n_s.min(n)] { is_s[i] = true; } p += n_s.min(n);
    for &i in &idx[p..p + n_d.min(n - p)] { is_d[i] = true; } p += n_d.min(n - p);
    for &i in &idx[p..p + n_m.min(n - p)] { is_m[i] = true; } p += n_m.min(n - p);
    for &i in &idx[p..p + n_v.min(n - p)] { is_v[i] = true; }
    (is_s, is_d, is_m, is_v)
}

// ── helper: STDP-inspired weight pool (log-normal, heavy-tailed) ──
fn generate_stdp_weight_pool(n: usize, rng: &mut StdRng) -> Vec<f32> {
    let pool_size = (n / 10).max(100);
    let mut pool = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        // Log-normal: exp(μ + σ * N(0,1)) with μ=-0.5, σ=0.8
        let u1 = rng.gen::<f32>();
        let u2 = rng.gen::<f32>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        pool.push((z * 0.8 - 0.5).exp());
    }
    pool
}

// ── helper: topological depth via BFS ──
fn compute_topological_depth(
    n: usize, row_ptr: &[i32], col_indices: &[i32],
    sensory: &[usize], dn: &[usize], mn: &[usize],
) -> Vec<f32> {
    // BFS from motor → sensory axis on reverse graph
    let mut dist = vec![f32::MAX; n];

    // Reverse adjacency for backward BFS from motor/sensory
    let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        for &j in &col_indices[start..end] {
            let j = j as usize;
            if j < n { rev_adj[j].push(i); }
        }
    }

    // Multi-source BFS from motor + DN neurons (depth = 0 ← motor pole)
    let mut queue = std::collections::VecDeque::new();
    for &i in mn { dist[i] = 0.0; queue.push_back(i); }
    for &i in dn {
        if dist[i] == f32::MAX { dist[i] = 0.5; queue.push_back(i); }
    }
    while let Some(u) = queue.pop_front() {
        let nd = dist[u] + 1.0;
        for &v in &rev_adj[u] {
            if nd < dist[v] { dist[v] = nd; queue.push_back(v); }
        }
    }

    // Forward BFS from sensory (depth = max ← sensory)
    let mut fwd_dist = vec![f32::MAX; n];
    let mut fwd_queue = std::collections::VecDeque::new();
    for &i in sensory {
        fwd_dist[i] = 0.0;
        fwd_queue.push_back(i);
    }
    while let Some(u) = fwd_queue.pop_front() {
        let start = row_ptr[u] as usize;
        let end = row_ptr[u + 1] as usize;
        let nd = fwd_dist[u] + 1.0;
        for &j in &col_indices[start..end] {
            let j = j as usize;
            if j < n && nd < fwd_dist[j] {
                fwd_dist[j] = nd;
                fwd_queue.push_back(j);
            }
        }
    }

    // Normalize and blend: depth = 1 - motor_dist_norm + sensory_dist_norm
    let max_motor = dist.iter().cloned().fold(0.0f32, f32::max);
    let max_sensory = fwd_dist.iter().cloned().fold(0.0f32, f32::max);
    let mut depth = vec![0.5f32; n];
    for i in 0..n {
        let m = if max_motor > 0.0 { (dist[i] / max_motor).min(1.0) } else { 0.5 };
        let s = if max_sensory > 0.0 { (fwd_dist[i] / max_sensory).min(1.0) } else { 0.5 };
        depth[i] = 0.6 * (1.0 - m) + 0.4 * s; // 0 = motor pole, 1 = sensory pole
    }
    depth
}

fn empty_sample(features: &[f32]) -> ConnectomeSample {
    ConnectomeSample {
        source: "empty".into(), n_neurons: 0, n_synapses: 0,
        row_ptr: vec![0], col_indices: vec![], values: vec![],
        nt_types: vec![], groups: std::collections::HashMap::new(),
        features: features.to_vec(), depth: None,
    }
}
