use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single connectome sample — either loaded from real data or synthesized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectomeSample {
    /// Name/source of this connectome (e.g. "flywire", "manc", "synthetic")
    pub source: String,
    /// Number of neurons
    pub n_neurons: usize,
    /// Number of synapses
    pub n_synapses: usize,
    /// CSR row pointer \[n_neurons + 1\]
    pub row_ptr: Vec<i32>,
    /// CSR column indices \[n_synapses\]
    pub col_indices: Vec<i32>,
    /// Synaptic weights \[n_synapses\]
    pub values: Vec<f32>,
    /// Neurotransmitter type per pre-neuron: +1 exc, -1 inh \[n_neurons\]
    pub nt_types: Vec<f32>,
    /// Named neuron groups (dn, mn, visual, etc.)
    pub groups: HashMap<String, Vec<usize>>,
    /// Graph-level feature vector for ML
    pub features: Vec<f32>,
    /// Optional topological depth per neuron
    pub depth: Option<Vec<f32>>,
}

/// A dataset of connectome graphs for training the generative model.
#[derive(Debug, Clone)]
pub struct ConnectomeDataset {
    pub samples: Vec<ConnectomeSample>,
    /// Feature dimension after normalization
    pub feature_dim: usize,
    /// Mean of features across dataset (for normalization)
    pub feature_mean: Vec<f32>,
    /// Std of features across dataset
    pub feature_std: Vec<f32>,
}

impl ConnectomeDataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            feature_dim: 0,
            feature_mean: Vec::new(),
            feature_std: Vec::new(),
        }
    }

    pub fn push(&mut self, sample: ConnectomeSample) {
        self.samples.push(sample);
    }

    /// Compute graph-level features for all samples.
    pub fn compute_features(&mut self) {
        for sample in &mut self.samples {
            sample.features = Self::extract_features(sample);
        }
        if self.samples.is_empty() {
            return;
        }
        self.feature_dim = self.samples[0].features.len();
        self.normalize();
    }

    fn extract_features(sample: &ConnectomeSample) -> Vec<f32> {
        let n = sample.n_neurons;
        let m = sample.n_synapses;
        if n == 0 {
            return vec![0.0; 10];
        }

        let density = m as f32 / (n as f32 * n as f32).max(1.0);

        let out_degrees: Vec<f32> = (0..n)
            .map(|i| (sample.row_ptr[i + 1] - sample.row_ptr[i]) as f32)
            .collect();
        let mean_deg = out_degrees.iter().sum::<f32>() / n as f32;
        let var_deg = out_degrees
            .iter()
            .map(|d| (d - mean_deg).powi(2))
            .sum::<f32>()
            / n as f32;

        let n_exc = sample.nt_types.iter().filter(|&&t| t > 0.0).count();
        let n_inh = sample.nt_types.iter().filter(|&&t| t < 0.0).count();
        let ei_ratio = if n_inh > 0 {
            n_exc as f32 / n_inh as f32
        } else {
            n_exc as f32
        };

        let mean_weight = sample.values.iter().map(|v| v.abs()).sum::<f32>() / m as f32;
        let mut pos_w = 0usize;
        let mut neg_w = 0usize;
        for &v in &sample.values {
            if v > 0.0 { pos_w += 1; } else { neg_w += 1; }
        }
        let weight_pos_ratio = if neg_w > 0 {
            pos_w as f32 / neg_w as f32
        } else {
            pos_w as f32
        };

        vec![
            n as f32,
            m as f32,
            density * 100.0,
            mean_deg,
            (var_deg / (mean_deg + 1e-8)).sqrt(),
            ei_ratio.min(10.0),
            mean_weight * 100.0,
            weight_pos_ratio.min(10.0),
            sample.groups.len() as f32,
            sample.depth.as_ref().map(|d| {
                let mean = d.iter().sum::<f32>() / d.len() as f32;
                let var = d.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d.len() as f32;
                var.sqrt()
            }).unwrap_or(0.0),
        ]
    }

    fn normalize(&mut self) {
        let n = self.samples.len();
        if n == 0 {
            return;
        }
        let d = self.feature_dim;
        self.feature_mean = vec![0.0; d];
        self.feature_std = vec![0.0; d];
        for sample in &self.samples {
            for (i, &v) in sample.features.iter().enumerate() {
                self.feature_mean[i] += v;
            }
        }
        for m in &mut self.feature_mean {
            *m /= n as f32;
        }
        for sample in &self.samples {
            for (i, &v) in sample.features.iter().enumerate() {
                self.feature_std[i] += (v - self.feature_mean[i]).powi(2);
            }
        }
        for s in &mut self.feature_std {
            *s = (*s / n as f32).sqrt().max(1e-8);
        }
        for sample in &mut self.samples {
            for (i, v) in sample.features.iter_mut().enumerate() {
                *v = (*v - self.feature_mean[i]) / self.feature_std[i];
            }
        }
    }

    /// Split into training (80%) and validation (20%) sets.
    pub fn split(&self) -> (ConnectomeDataset, ConnectomeDataset) {
        let n = self.samples.len();
        let split = (n * 8 / 10).max(1);
        let mut train = ConnectomeDataset::new();
        let mut val = ConnectomeDataset::new();
        for (i, sample) in self.samples.iter().enumerate() {
            if i < split {
                train.push(sample.clone());
            } else {
                val.push(sample.clone());
            }
        }
        train.feature_dim = self.feature_dim;
        train.feature_mean = self.feature_mean.clone();
        train.feature_std = self.feature_std.clone();
        val.feature_dim = self.feature_dim;
        val.feature_mean = self.feature_mean.clone();
        val.feature_std = self.feature_std.clone();
        (train, val)
    }

    /// Build from a real connectome (FlyWire/MANC format).
    pub fn from_connectome(
        source: &str,
        n_neurons: usize,
        row_ptr: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<f32>,
        nt_types: Vec<f32>,
        groups: HashMap<String, Vec<usize>>,
        depth: Option<Vec<f32>>,
    ) -> Self {
        let n_syn = values.len();
        let mut ds = Self::new();
        let features = Vec::new();
        ds.push(ConnectomeSample {
            source: source.to_string(),
            n_neurons,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values,
            nt_types,
            groups,
            features,
            depth,
        });
        ds.compute_features();
        ds
    }

    /// Generate a random synthetic connectome for testing.
    pub fn random(n_neurons: usize, density: f32, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut row_ptr = vec![0i32; n_neurons + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut nt_types = vec![1.0f32; n_neurons];

        for i in 0..n_neurons {
            nt_types[i] = if rng.gen::<f32>() < 0.2 { -1.0 } else { 1.0 };
            let mut targets: Vec<usize> = (0..n_neurons)
                .filter(|_| rng.gen::<f32>() < density)
                .collect();
            targets.sort();
            targets.dedup();
            col_indices.extend(targets.iter().map(|&t| t as i32));
            let w: Vec<f32> = targets
                .iter()
                .map(|_| {
                    let base: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                    if nt_types[i] > 0.0 { base.abs() } else { -base.abs() }
                })
                .collect();
            values.extend(w);
            row_ptr[i + 1] = col_indices.len() as i32;
        }

        let mut groups = HashMap::new();
        let mut dn: Vec<usize> = (n_neurons / 20..n_neurons / 10).collect();
        dn.sort();
        groups.insert("dn".to_string(), dn);

        Self::from_connectome("synthetic", n_neurons, row_ptr, col_indices, values, nt_types, groups, None)
    }
}
