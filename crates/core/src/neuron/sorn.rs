//! SORN — Self-Organizing Recurrent Network.
//!
//! Implements three learning rules:
//! 1. STDP — spike-timing dependent plasticity
//! 2. Intrinsic Plasticity — adaptive thresholds for target firing rate
//! 3. Synaptic Normalization — weight normalization to target sum

use rayon::prelude::*;

use super::stdp::stdp_update;

// ─── Intrinsic Plasticity ─────────────────────────────────

/// Intrinsic Plasticity: адаптирует порог нейрона для поддержания target firing rate.
///
/// θ += lr_ip · (firing_rate - target_rate)
pub fn intrinsic_plasticity(
    theta: &mut [f32],
    spikes: &[i32],
    lr_ip: f32,
    target_rate: f32,
    decay: f32,
) {
    theta
        .par_iter_mut()
        .zip(spikes.par_iter())
        .for_each(|(t, &s)| {
            let rate = if s != 0 { 1.0 } else { 0.0 };
            *t += lr_ip * (rate - target_rate);
            *t *= decay;
            *t = t.clamp(0.0, 50.0);
        });
}

// ─── Synaptic Normalization ───────────────────────────────

/// Synaptic Normalization: нормирует входящие веса каждого нейрона к target_sum.
///
/// w_ij := w_ij · target_sum / max(Σ_k w_kj, eps)
pub fn synaptic_normalization(
    weights: &mut [f32],
    n_pre: usize,
    n_post: usize,
    target_sum: f32,
    eps: f32,
) {
    assert!(weights.len() >= n_pre * n_post);
    let mut col_scales = vec![1.0f32; n_post];
    col_scales
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, scale)| {
            let mut sum: f32 = 0.0;
            for i in 0..n_pre {
                sum += weights[i * n_post + j];
            }
            if sum > eps {
                *scale = target_sum / sum;
            }
        });
    weights.par_chunks_exact_mut(n_post).for_each(|row| {
        for j in 0..n_post {
            row[j] *= col_scales[j];
        }
    });
}

// ═══════════════════════════════════════════════════════════
//  SORN network
// ═══════════════════════════════════════════════════════════

/// SORN network — Self-Organizing Recurrent Neural Network.
///
/// Цикл обучения (каждый шаг):
/// 1. STDP — spike-timing dependent plasticity
/// 2. Intrinsic Plasticity — адаптация порогов
/// 3. Synaptic Normalization — нормировка весов
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

    /// Один шаг обучения SORN:
    /// 1. STDP, 2. Intrinsic Plasticity, 3. Synaptic Normalization
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
