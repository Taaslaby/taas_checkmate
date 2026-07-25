//! E-Prop common utilities — surrogate gradient and standalone weight update.

use rayon::prelude::*;

/// Surrogate gradient: derivative of spike threshold function.
///
/// Uses fast sigmoid surrogate: σ'(v) = 1 / (1 + |v - v_th|/τ)²
pub fn surrogate_gradient(v: f32, v_thresh: f32, tau_surrogate: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_surrogate;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

/// E-Prop standalone weight update: Δw = η · L_j · e_ji
///
/// Parallelized via rayon.
pub fn eprop_update(weights: &mut [f32], eligibility: &[f32], learning_signals: &[f32], lr: f32) {
    let n = weights
        .len()
        .min(eligibility.len())
        .min(learning_signals.len());
    let chunks = n / 8;
    if chunks == 0 {
        return;
    }
    weights[..chunks * 8]
        .par_chunks_exact_mut(8)
        .zip(eligibility[..chunks * 8].par_chunks_exact(8))
        .zip(learning_signals[..chunks * 8].par_chunks_exact(8))
        .for_each(|((w8, e8), l8)| {
            for i in 0..8 {
                w8[i] += lr * l8[i] * e8[i];
            }
        });
    for i in chunks * 8..n {
        weights[i] += lr * learning_signals[i] * eligibility[i];
    }
}
