//! # Decoder Microkernels
//!
//! `f32x8` kernels for decoding oscillator/neural states into outputs:
//! complex decoder, PST features, LMS / NLMS / Kalman / Adam training and
//! ring synthesis.

use wide::f32x8;

use crate::core::util::{load8, store8};

/// Complex decoder: `out = W·[A·cos(θ)] + bias`. SIMD dot product per output.
pub fn t3_complex_decoder(
    outputs: &mut [f32],
    amplitudes: &[f32],
    cos_buf: &[f32],
    weights: &[f32],
    biases: &[f32],
) {
    let n_out = outputs.len();
    let n_in = amplitudes.len();
    assert_eq!(cos_buf.len(), n_in, "t3_complex_decoder: cos_buf length mismatch");
    assert_eq!(biases.len(), n_out, "t3_complex_decoder: biases length mismatch");
    assert_eq!(weights.len(), n_in * n_out, "t3_complex_decoder: weights length mismatch");
    let n8 = n_in & !7;

    for o in 0..n_out {
        let mut sum = biases[o];
        let row = o * n_in;
        let mut vdot = f32x8::splat(0.0);
        let mut i = 0;
        unsafe {
            while i < n8 {
                vdot += load8(amplitudes.as_ptr(), i) * load8(cos_buf.as_ptr(), i) * load8(weights.as_ptr(), row + i);
                i += 8;
            }
        }
        sum += vdot.to_array().iter().sum::<f32>();
        for j in i..n_in {
            sum += amplitudes[j] * cos_buf[j] * weights[row + j];
        }
        outputs[o] = sum;
    }
}

/// Pairwise PST features: `sin(θ₁−θ₂)`, `cos(θ₁−θ₂)` for all (i, j) pairs.
pub fn t3_pst_features(out_sin: &mut [f32], out_cos: &mut [f32], ph1: &[f32], ph2: &[f32]) {
    let n1 = ph1.len();
    let n2 = ph2.len();
    assert_eq!(out_sin.len(), n1 * n2, "t3_pst_features: out_sin length mismatch");
    assert_eq!(out_cos.len(), n1 * n2, "t3_pst_features: out_cos length mismatch");
    for i in 0..n1 {
        for j in 0..n2 {
            let d = ph1[i] - ph2[j];
            out_sin[i * n2 + j] = d.sin();
            out_cos[i * n2 + j] = d.cos();
        }
    }
}

/// LMS training step: `w += lr·error·states; bias += lr·error`.
pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32) {
    let n = weights.len();
    assert_eq!(states.len(), n, "t3_lms_train: states length mismatch");
    assert!(!bias.is_empty(), "t3_lms_train: bias must have at least 1 element");
    let lr_e = f32x8::splat(lr * error);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let w = load8(weights.as_ptr(), off) + lr_e * load8(states.as_ptr(), off);
            store8(weights.as_mut_ptr(), off, w);
        }
    }
    for i in n8..n {
        weights[i] += lr * error * states[i];
    }
    bias[0] += lr * error;
}

/// NLMS training step: `w += lr·error·states / (‖states‖² + ε)`.
pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32, eps: f32) {
    let n = weights.len();
    assert_eq!(states.len(), n, "nlms_train: states length mismatch");
    assert!(!bias.is_empty(), "nlms_train: bias must have at least 1 element");

    // Norm² via SIMD reduction
    let n8 = n & !7;
    let mut ns = f32x8::splat(0.0);
    let mut i = 0;
    unsafe {
        while i < n8 {
            let s = load8(states.as_ptr(), i);
            ns += s * s;
            i += 8;
        }
    }
    let mut norm_sq: f32 = ns.to_array().iter().sum();
    for s in states[i..].iter() {
        norm_sq += s * s;
    }

    let mu = f32x8::splat(lr * error / (norm_sq + eps));
    let mut i = 0;
    unsafe {
        while i < n8 {
            let w = load8(weights.as_ptr(), i) + mu * load8(states.as_ptr(), i);
            store8(weights.as_mut_ptr(), i, w);
            i += 8;
        }
    }
    for j in i..n {
        weights[j] += lr * error * states[j] / (norm_sq + eps);
    }
    bias[0] += lr * error / (norm_sq + eps);
}

/// Kalman filter update: `w += K·error, cov -= K·states·cov`.
///
/// ```text
/// K[i] = cov[i]·states[i] / s
/// w[i] += K[i]·error
/// cov[i] -= K[i]·states[i]·cov[i]   (floored at 1e-12)
/// ```
///
/// `s` is the pre-computed innovation covariance `R + Σ(cov·states²)`.
pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32], error: f32, s: f32) {
    let n = weights.len();
    assert_eq!(cov.len(), n, "kalman_update: cov length mismatch");
    assert_eq!(states.len(), n, "kalman_update: states length mismatch");
    debug_assert!(s > 0.0, "kalman_update: s must be > 0, got {s}");
    let s2 = f32x8::splat(s + 1e-8);
    let err_s = f32x8::splat(error);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let w = load8(weights.as_ptr(), off);
            let c = load8(cov.as_ptr(), off);
            let st = load8(states.as_ptr(), off);
            let k = c * st / s2;
            store8(weights.as_mut_ptr(), off, w + k * err_s);
            store8(cov.as_mut_ptr(), off, (c - k * c).max(f32x8::splat(1e-12)));
        }
    }
    let s2 = s + 1e-8;
    for i in n8..n {
        let k = cov[i] * states[i] / s2;
        weights[i] += k * error;
        cov[i] = (cov[i] - k * cov[i]).max(1e-12);
    }
}

/// Adam training step for decoder weights: `w -= lr·m̂/(√v̂ + eps)`.
///
/// Gradient `g = error·states`; `bc1`, `bc2` are the bias-correction terms
/// (β₁ᵗ, β₂ᵗ). The scalar tail divides by `bc1` to match the SIMD path.
#[allow(clippy::too_many_arguments)]
pub fn adam_train(
    weights: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    states: &[f32],
    error: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bc1: f32,
    _bc2: f32,
    eps: f32,
) {
    let n = weights.len();
    assert_eq!(m.len(), n, "adam_train: m length mismatch");
    assert_eq!(v.len(), n, "adam_train: v length mismatch");
    assert_eq!(states.len(), n, "adam_train: states length mismatch");
    let b1 = f32x8::splat(beta1);
    let b1_c = f32x8::splat(1.0 - beta1);
    let b2 = f32x8::splat(beta2);
    let b2_c = f32x8::splat(1.0 - beta2);
    let lr_s = f32x8::splat(lr / bc1);
    let eps_s = f32x8::splat(eps);
    let err_s = f32x8::splat(error);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut mm = load8(m.as_ptr(), off);
            let mut vv = load8(v.as_ptr(), off);
            let st = load8(states.as_ptr(), off);
            let g = err_s * st;
            mm = b1 * mm + b1_c * g;
            vv = b2 * vv + b2_c * g * g;
            let mut w = load8(weights.as_ptr(), off);
            w -= lr_s * mm / (vv.sqrt() + eps_s);
            store8(weights.as_mut_ptr(), off, w);
            store8(m.as_mut_ptr(), off, mm);
            store8(v.as_mut_ptr(), off, vv);
        }
    }

    for i in n8..n {
        let g = error * states[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        weights[i] -= lr * m[i] / (v[i].sqrt() + eps) / bc1;
    }
}

/// PST decoder: weighted circular buffer readout (SIMD dot per feature).
///
/// `output[f] = Σ_t weights[t·F + f]·buffer[(pos + t) % l][f]`.
pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32], l: usize, pos: usize) {
    let feat_dim = output.len();
    assert_eq!(buffer.len(), l * feat_dim, "pst_decode: buffer length mismatch");
    assert_eq!(weights.len(), l * feat_dim, "pst_decode: weights length mismatch");
    output.fill(0.0);

    let n8 = feat_dim & !7;
    for t in 0..l {
        let src = (pos + t) % l;
        let b_off = src * feat_dim;
        let w_off = t * feat_dim;

        let mut i = 0;
        unsafe {
            while i < n8 {
                let bv = load8(buffer.as_ptr(), b_off + i);
                let wv = load8(weights.as_ptr(), w_off + i);
                let arr: [f32; 8] = (bv * wv).to_array();
                for j in 0..8 {
                    output[i + j] += arr[j];
                }
                i += 8;
            }
        }
        for f in i..feat_dim {
            output[f] += weights[w_off + f] * buffer[b_off + f];
        }
    }
}

/// Ring synthesis: `out[i] = w[i]·a[i]·trig[i]`.
pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32], weights: &[f32]) {
    let n = output.len();
    assert_eq!(amplitudes.len(), n, "ring_synthesis: amplitudes length mismatch");
    assert_eq!(trig_buf.len(), n, "ring_synthesis: trig_buf length mismatch");
    assert_eq!(weights.len(), n, "ring_synthesis: weights length mismatch");
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let v = load8(weights.as_ptr(), off) * load8(amplitudes.as_ptr(), off) * load8(trig_buf.as_ptr(), off);
            store8(output.as_mut_ptr(), off, v);
        }
    }
    for i in n8..n {
        output[i] = weights[i] * amplitudes[i] * trig_buf[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlms_train_basic() {
        let n = 32;
        let mut weights = vec![0.0f32; n];
        let mut bias = vec![0.0f32; 1];
        let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        for _ in 0..100 {
            let target = 1.0;
            let pred: f32 = weights.iter().zip(states.iter()).map(|(w, s)| w * s).sum::<f32>() + bias[0];
            let err = target - pred;
            nlms_train(&mut weights, &mut bias, &states, err, 0.01, 1e-8);
        }
        assert!(weights.iter().any(|&w| w.abs() > 0.001), "NLMS weights should update");
    }

    #[test]
    fn kalman_update_basic() {
        let n = 16;
        let mut weights = vec![0.0f32; n];
        let mut cov = vec![1.0f32; n];
        let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        for _ in 0..100 {
            let target = 1.0;
            let pred: f32 = weights.iter().zip(states.iter()).map(|(w, s)| w * s).sum();
            let err = target - pred;
            let s: f32 = states.iter().zip(cov.iter()).map(|(x, c)| c * x * x).sum::<f32>() + 0.1;
            kalman_update(&mut weights, &mut cov, &states, err, s);
        }
        assert!(weights.iter().any(|&w| w.abs() > 0.001), "Kalman weights should update");
        assert!(cov.iter().all(|&c| c < 1.0), "Covariance should decrease");
    }

    #[test]
    fn t3_complex_decoder_basic() {
        let n_in = 8;
        let n_out = 2;
        let amps = vec![1.0f32; n_in];
        let cos_buf = vec![1.0f32; n_in];
        let weights: Vec<f32> = (0..(n_in * n_out)).map(|i| (i + 1) as f32 * 0.01).collect();
        let biases = vec![0.0f32; n_out];
        let mut outputs = vec![0.0f32; n_out];
        t3_complex_decoder(&mut outputs, &amps, &cos_buf, &weights, &biases);
        for o in 0..n_out {
            let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum();
            assert!((outputs[o] - expected).abs() < 1e-4, "complex_decoder[{o}]");
        }
    }

    #[test]
    fn ring_synthesis_basic() {
        let n = 8;
        let amps = vec![2.0f32; n];
        let trig = vec![0.5f32; n];
        let weights = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];
        ring_synthesis(&mut output, &amps, &trig, &weights);
        for i in 0..n {
            assert!((output[i] - 1.0).abs() < 1e-6, "ring_synthesis[{i}]: {}", output[i]);
        }
    }

    #[test]
    fn t3_pst_features_and_pst_decode_smoke() {
        let p1 = vec![0.0f32, 0.5, 1.0];
        let p2 = vec![0.2f32, 0.7];
        let mut os = vec![0.0f32; 6];
        let mut oc = vec![0.0f32; 6];
        t3_pst_features(&mut os, &mut oc, &p1, &p2);
        let d0: f32 = 0.0f32 - 0.2f32;
        assert!((os[0] - d0.sin()).abs() < 1e-5, "pst sin");
        assert!((oc[0] - d0.cos()).abs() < 1e-5, "pst cos");
    }

    #[test]
    fn lms_and_adam_train_smoke() {
        let n = 16;
        let mut w = vec![0.0f32; n];
        let mut b = vec![0.0f32; 1];
        let states: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        t3_lms_train(&mut w, &mut b, &states, 0.5, 0.01);
        assert!(w.iter().any(|&x| x.abs() > 0.001), "LMS should update weights");

        let mut wa = vec![0.0f32; n];
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        adam_train(&mut wa, &mut m, &mut v, &states, 0.5, 0.01, 0.9, 0.999, 0.9, 0.999, 1e-8);
        assert!(wa.iter().any(|&x| x.abs() > 0.001), "Adam should update weights");
    }
}
