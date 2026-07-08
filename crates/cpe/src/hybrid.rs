//! Hybrid update — STDP eligibility + T2-SPIGRAD gradients in one pass.
//!
//! Δw = η₁ · δ · e + η₂ · δ · ∇L
//! where:
//! - η₁: STDP learning rate
//! - η₂: T2-SPIGRAD learning rate
//! - δ: dopamine signal (chaos/pattern gate)
//! - e: eligibility trace
//! - ∇L: exact gradient from T2-SPIGRAD backward

use crate::config::CpeConfig;

/// Apply hybrid STDP + T2-SPIGRAD update to input weights w_in.
///
/// # Arguments
/// * `w_in` - input weight matrix [features × n_drives], flattened
/// * `delta` - dopamine signal δ
/// * `grad_w_in` - gradient from T2 backward [features × n_drives], flattened
/// * `dopamine_gate` - timing gate [0, 1]
/// * `config` - CPE configuration
/// * `mode` - which learning path to use
///
/// # Formula
/// Δw = η₁ · δ · 0 + η₂ · δ · ∇L (T2-SPIGRAD only currently;
/// STDP on connectome weights is handled separately)
pub fn hybrid_update_w_in(
    w_in: &mut [f32],
    delta: f32,
    grad_w_in: &[f32],
    dopamine_gate: f32,
    config: &CpeConfig,
) {
    let t2_lr = config.t2_lr;
    let n = w_in.len().min(grad_w_in.len());
    for i in 0..n {
        w_in[i] += t2_lr * delta * grad_w_in[i] * dopamine_gate;
    }
    // Clip
    for w in w_in.iter_mut() {
        *w = w.clamp(-5.0, 5.0);
    }
}

/// Apply hybrid update to output weights w_out.
pub fn hybrid_update_w_out(
    w_out: &mut [f32],
    delta: f32,
    grad_w_out: &[f32],
    dopamine_gate: f32,
    config: &CpeConfig,
) {
    let t2_lr = config.t2_lr;
    let n = w_out.len().min(grad_w_out.len());
    for i in 0..n {
        w_out[i] += t2_lr * delta * grad_w_out[i] * dopamine_gate;
    }
    for w in w_out.iter_mut() {
        *w = w.clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_update_w_in_positive_delta() {
        let mut w = vec![0.1f32; 10];
        let grad = vec![1.0f32; 10];
        let config = CpeConfig::default();
        hybrid_update_w_in(&mut w, 1.0, &grad, 1.0, &config);
        // With T2_LR=0.005, delta=1.0, grad=1.0, gate=1.0: update = 0.005
        for &wi in &w {
            assert!((wi - 0.105).abs() < 1e-6, "expected 0.105, got {wi}");
        }
    }

    #[test]
    fn test_hybrid_update_w_in_zero_gate() {
        let mut w = vec![0.1f32; 10];
        let grad = vec![1.0f32; 10];
        let config = CpeConfig::default();
        hybrid_update_w_in(&mut w, 1.0, &grad, 0.0, &config);
        // Gate=0, no update
        for &wi in &w {
            assert!((wi - 0.1).abs() < 1e-6, "expected 0.1, got {wi}");
        }
    }
}
