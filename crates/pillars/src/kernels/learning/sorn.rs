use super::stdp::stdp_update;

// ═══════════════════════════════════════════════════════════
//  SORN — Self-Organizing Recurrent Network
// ═══════════════════════════════════════════════════════════

/// Intrinsic plasticity: adapts neuron thresholds to maintain a target
/// firing rate. `θ += lr_ip·(rate − target)`, decayed and clamped to [0, 50].
pub fn intrinsic_plasticity(
    theta: &mut [f32],
    spikes: &[i32],
    lr_ip: f32,
    target_rate: f32,
    decay: f32,
) {
    for (t, &s) in theta.iter_mut().zip(spikes.iter()) {
        let rate = if s != 0 { 1.0 } else { 0.0 };
        *t += lr_ip * (rate - target_rate);
        *t *= decay;
        *t = t.clamp(0.0, 50.0);
    }
}

/// Synaptic normalization: rescales each post-neuron's incoming weights to
/// `target_sum`. `weights` is row-major `[n_pre × n_post]`.
pub fn synaptic_normalization(
    weights: &mut [f32],
    n_pre: usize,
    n_post: usize,
    target_sum: f32,
    eps: f32,
) {
    assert!(weights.len() >= n_pre * n_post);
    // Column sums (per post-neuron)
    let mut col_scales = vec![1.0f32; n_post];
    for (j, scale) in col_scales.iter_mut().enumerate() {
        let mut sum: f32 = 0.0;
        for i in 0..n_pre {
            sum += weights[i * n_post + j];
        }
        if sum > eps {
            *scale = target_sum / sum;
        }
    }
    for row in weights.chunks_exact_mut(n_post) {
        for j in 0..n_post {
            row[j] *= col_scales[j];
        }
    }
}

/// SORN network — Self-Organizing Recurrent Neural Network.
///
/// Each learning step applies:
/// 1. **STDP** — spike-timing dependent plasticity
/// 2. **Intrinsic plasticity** — adaptive thresholds
/// 3. **Synaptic normalization** — weight renormalization
///
/// Reference: Lazar, Pipa & Triesch (2009) "SORN: a self-organizing
/// recurrent neural network".
pub struct SornNetwork {
    pub weights: Vec<f32>,
    pub theta: Vec<f32>,
    pub rates: Vec<f32>,
    pub n: usize,
    pub lr_stdp_ltd: f32,
    pub lr_stdp_ltp: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub w_target_sum: f32,
    pub lr_ip: f32,
    pub target_rate: f32,
    pub ip_decay: f32,
    pub sn_eps: f32,
}

impl SornNetwork {
    pub fn new(n: usize, w_init: f32, w_target_sum: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            weights: vec![w_init; n_pad * n_pad],
            theta: vec![0.0; n_pad],
            rates: vec![0.0; n_pad],
            n: n_pad,
            lr_stdp_ltd: 0.01,
            lr_stdp_ltp: 0.005,
            w_min: 0.0,
            w_max: 2.0,
            w_target_sum,
            lr_ip: 0.001,
            target_rate: 0.01,
            ip_decay: 0.999,
            sn_eps: 1e-8,
        }
    }

    /// One SORN learning step: STDP → intrinsic plasticity → normalization.
    pub fn step(&mut self, spikes: &[i32], traces_pre: &[f32], traces_post: &[f32]) {
        stdp_update(
            &mut self.weights,
            traces_pre,
            traces_post,
            self.lr_stdp_ltd,
            self.lr_stdp_ltp,
            self.w_min,
            self.w_max,
        );
        intrinsic_plasticity(
            &mut self.theta,
            spikes,
            self.lr_ip,
            self.target_rate,
            self.ip_decay,
        );
        synaptic_normalization(
            &mut self.weights,
            self.n,
            self.n,
            self.w_target_sum,
            self.sn_eps,
        );
    }

    pub fn reset(&mut self) {
        self.weights.fill(0.1);
        self.theta.fill(0.0);
        self.rates.fill(0.0);
    }
}
