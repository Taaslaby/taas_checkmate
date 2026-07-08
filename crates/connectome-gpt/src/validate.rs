use crate::dataset::ConnectomeSample;
use crate::metrics::GraphMetrics;
use taas_connectome::sim;
use taas_connectome::ConnectomeData;

/// Results of validating a generated connectome via simulation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub graph_metrics: GraphMetrics,
    pub mean_firing_rate: f32,
    pub firing_rate_std: f32,
    pub dn_rate_mean: f32,
    pub mn_rate_mean: f32,
    pub n_spikes_total: usize,
    pub stable: bool,
    pub score: f32,
}

/// Validates generated connectomes by running neural simulation.
pub struct Validator {
    pub dt_ms: f32,
    pub n_steps: usize,
}

impl Default for Validator {
    fn default() -> Self {
        Self {
            dt_ms: 0.5,
            n_steps: 100,
        }
    }
}

impl Validator {
    pub fn new(dt_ms: f32, n_steps: usize) -> Self {
        Self { dt_ms, n_steps }
    }

    /// Run simulation and extract metrics from a connectome sample.
    pub fn validate(&self, sample: &ConnectomeSample) -> ValidationResult {
        let graph_metrics = GraphMetrics::compute(sample);

        let conn = ConnectomeData::new(
            sample.row_ptr.clone(),
            sample.col_indices.clone(),
            sample.values.clone(),
            sample.nt_types.clone(),
            sample
                .groups
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect(),
        );

        let n = sample.n_neurons;

        let mut state = sim::BrainState::new(n, sim::NeuronModel::Izhikevich);

        let params = sim::IzhikevichParams {
            dt_ms: self.dt_ms,
            tau_ref: 3.0,
            a: vec![0.02; n],
            b: vec![0.2; n],
            c: vec![-65.0; n],
            d: vec![8.0; n],
        };

        let external_i = vec![10.0f32; n];

        let spike_buf = sim::sim_izhikevich(&conn, &mut state, &params, &external_i, self.n_steps);

        let mut dn_spikes = 0usize;
        let mut mn_spikes = 0usize;
        let mut total_spikes = 0usize;
        let mut step_rates = Vec::new();

        for step in 0..self.n_steps {
            let offset = step * n;
            let mut step_count = 0usize;
            for i in 0..n {
                if spike_buf[offset + i] != 0 {
                    step_count += 1;
                    if let Some(dn) = conn.group("dn") {
                        if dn.contains(&i) { dn_spikes += 1; }
                    }
                    if let Some(mn) = conn.group("mn") {
                        if mn.contains(&i) { mn_spikes += 1; }
                    }
                }
            }
            total_spikes += step_count;
            step_rates.push(step_count as f32 / n as f32 * 1000.0 / self.dt_ms);
        }

        let mean_fr = step_rates.iter().sum::<f32>() / step_rates.len() as f32;
        let fr_var = step_rates.iter().map(|&r| (r - mean_fr).powi(2)).sum::<f32>() / step_rates.len() as f32;

        let dt_s = self.dt_ms * self.n_steps as f32 / 1000.0;
        let dn_rate = dn_spikes as f32 / dt_s / 1_000.0;
        let mn_rate = mn_spikes as f32 / dt_s / 1_000.0;

        let stable = total_spikes > 0 && total_spikes < n * self.n_steps / 2;
        let score = self.compute_score(&graph_metrics, mean_fr, stable);

        ValidationResult {
            graph_metrics,
            mean_firing_rate: mean_fr,
            firing_rate_std: fr_var.sqrt(),
            dn_rate_mean: dn_rate,
            mn_rate_mean: mn_rate,
            n_spikes_total: total_spikes,
            stable,
            score,
        }
    }

    fn compute_score(&self, metrics: &GraphMetrics, fr: f32, stable: bool) -> f32 {
        let mut score = 0.0f32;

        if !stable {
            score -= 10.0;
        }

        if metrics.n_neurons > 10 {
            score += 1.0;
        }
        if metrics.n_synapses > metrics.n_neurons {
            score += 1.0;
        }
        if (1.0..=100.0).contains(&metrics.ei_ratio) {
            score += 2.0;
        }
        if metrics.reciprocity > 0.01 && metrics.reciprocity < 0.5 {
            score += 1.0;
        }
        if metrics.clustering_coefficient > 0.0 {
            score += 1.0;
        }
        if metrics.density > 0.001 && metrics.density < 0.5 {
            score += 1.0;
        }
        if (1.0..=200.0).contains(&fr) {
            score += 2.0;
        }
        if metrics.n_groups > 0 {
            score += 1.0;
        }

        score
    }

    /// Validate multiple generated connectomes and return the best.
    pub fn find_best(&self, samples: &[ConnectomeSample]) -> Option<(usize, ValidationResult)> {
        samples
            .iter()
            .enumerate()
            .map(|(i, s)| (i, self.validate(s)))
            .max_by(|(_, a), (_, b)| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}
