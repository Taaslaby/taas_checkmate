//! # Signal Processing Microkernels
//!
//! Batch `f32x8` kernels for signal-level operations: safety clamping,
//! stochastic encoding, sigmoid/thalamic/phase gating, ReLU, elementwise add,
//! SGD/Adam optimizers and Hebbian weight updates.
//!
//! Every function processes 8-wide SIMD lanes and a scalar tail. SIMD paths
//! use `fast_exp_poly_simd`, tails use `fast_exp` — bit-identical.
//!
//! See `crate::math::fast_exp` for the underlying exponential pair.

use wide::f32x8;
use wide::CmpGt;

use crate::core::util::{load8, store8};
use crate::math::sigmoid::{sigmoid_poly_s, sigmoid_poly_simd};

/// Clamp all values in `data` to `[min, max]`.
pub fn safety_clamp(data: &mut [f32], min: f32, max: f32) {
    let n = data.len();
    let n8 = n & !7;
    let min_s = f32x8::splat(min);
    let max_s = f32x8::splat(max);
    for off in (0..n8).step_by(8) {
        unsafe {
            let v = load8(data.as_ptr(), off).max(min_s).min(max_s);
            store8(data.as_mut_ptr(), off, v);
        }
    }
    for v in data[n8..].iter_mut() {
        *v = v.clamp(min, max);
    }
}

/// Stochastic spike encoder: `target = spike_val` when `input + noise > threshold`, else 0.
pub fn stochastic_encoder(
    target_i: &mut [f32],
    inputs: &[f32],
    noise: &[f32],
    threshold: f32,
    spike_val: f32,
) {
    let n = target_i.len();
    assert_eq!(
        inputs.len(),
        n,
        "stochastic_encoder: inputs length mismatch"
    );
    assert_eq!(noise.len(), n, "stochastic_encoder: noise length mismatch");
    let th_s = f32x8::splat(threshold);
    let sv_s = f32x8::splat(spike_val);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let inp = load8(inputs.as_ptr(), off);
            let no = load8(noise.as_ptr(), off);
            let mask = (inp + no).cmp_gt(th_s);
            store8(target_i.as_mut_ptr(), off, mask.blend(sv_s, zero));
        }
    }
    for i in n8..n {
        target_i[i] = if inputs[i] + noise[i] > threshold {
            spike_val
        } else {
            0.0
        };
    }
}

/// Linear readout: `output = W·states + bias`. SIMD dot product per output.
pub fn linear_decoder(outputs: &mut [f32], states: &[f32], weights: &[f32], biases: &[f32]) {
    let n_out = outputs.len();
    let n_in = states.len();
    assert_eq!(
        biases.len(),
        n_out,
        "linear_decoder: biases length mismatch"
    );
    assert_eq!(
        weights.len(),
        n_in * n_out,
        "linear_decoder: weights length mismatch (expected {}*{}={})",
        n_in,
        n_out,
        n_in * n_out
    );
    let n8 = n_in & !7;
    for o in 0..n_out {
        let mut sum = biases[o];
        let row = o * n_in;
        let mut vdot = f32x8::splat(0.0);
        let mut i = 0;
        while i < n8 {
            unsafe {
                vdot += load8(weights.as_ptr(), row + i) * load8(states.as_ptr(), i);
            }
            i += 8;
        }
        sum += vdot.to_array().iter().sum::<f32>();
        for j in i..n_in {
            sum += weights[row + j] * states[j];
        }
        outputs[o] = sum;
    }
}

/// Thalamic gate: `target = source × σ(gain·(gate − threshold))`.
pub fn thalamic_gate(
    target_i: &mut [f32],
    source_v: &[f32],
    gate_v: &[f32],
    threshold: f32,
    gain: f32,
) {
    let n = target_i.len();
    assert_eq!(source_v.len(), n, "thalamic_gate: source_v length mismatch");
    assert_eq!(gate_v.len(), n, "thalamic_gate: gate_v length mismatch");
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let sv = load8(source_v.as_ptr(), off);
            let gv = load8(gate_v.as_ptr(), off);
            let gate = sigmoid_poly_simd(gain_s * (gv - th_s));
            store8(target_i.as_mut_ptr(), off, sv * gate);
        }
    }
    for i in n8..n {
        let sig = sigmoid_poly_s(gain * (gate_v[i] - threshold));
        target_i[i] = source_v[i] * sig;
    }
}

/// Elementwise sigmoid: `output = σ(gain·(input − threshold))`.
pub fn sigmoid_gate(output: &mut [f32], input: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "sigmoid_gate: input length mismatch");
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let inp = load8(input.as_ptr(), off);
            store8(
                output.as_mut_ptr(),
                off,
                sigmoid_poly_simd(gain_s * (inp - th_s)),
            );
        }
    }
    for i in n8..n {
        output[i] = sigmoid_poly_s(gain * (input[i] - threshold));
    }
}

/// Phase gate: `output = input × σ(gain·(r − threshold))`.
pub fn phase_gate_apply(output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "phase_gate_apply: input length mismatch");
    assert_eq!(r.len(), n, "phase_gate_apply: r length mismatch");
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let inp = load8(input.as_ptr(), off);
            let rr = load8(r.as_ptr(), off);
            store8(
                output.as_mut_ptr(),
                off,
                inp * sigmoid_poly_simd(gain_s * (rr - th_s)),
            );
        }
    }
    for i in n8..n {
        let gate = sigmoid_poly_s(gain * (r[i] - threshold));
        output[i] = input[i] * gate;
    }
}

/// T3 drive encoder: `drives = gain × inputs`.
pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    let n = drives.len();
    assert_eq!(inputs.len(), n, "t3_drive_encoder: inputs length mismatch");
    let g = f32x8::splat(gain);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            store8(drives.as_mut_ptr(), off, g * load8(inputs.as_ptr(), off));
        }
    }
    for i in n8..n {
        drives[i] = gain * inputs[i];
    }
}

/// T3 TACT generator: output 1.0 when all 3 phases exceed threshold, else 0.0.
pub fn t3_tact_generator(
    pulses: &mut [f32],
    ph1: &[f32],
    ph2: &[f32],
    ph3: &[f32],
    threshold: f32,
) {
    let n = pulses.len();
    assert_eq!(ph1.len(), n, "t3_tact_generator: ph1 length mismatch");
    assert_eq!(ph2.len(), n, "t3_tact_generator: ph2 length mismatch");
    assert_eq!(ph3.len(), n, "t3_tact_generator: ph3 length mismatch");
    let th = f32x8::splat(threshold);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let a = load8(ph1.as_ptr(), off);
            let b = load8(ph2.as_ptr(), off);
            let c = load8(ph3.as_ptr(), off);
            let mask = a.cmp_gt(th) & b.cmp_gt(th) & c.cmp_gt(th);
            store8(pulses.as_mut_ptr(), off, mask.blend(one, zero));
        }
    }
    for i in n8..n {
        pulses[i] = if ph1[i] > threshold && ph2[i] > threshold && ph3[i] > threshold {
            1.0
        } else {
            0.0
        };
    }
}

/// Echo state tap: read one delay tap from an oscillator ring buffer.
///
/// `buffer` has layout `[n_osc × delay_len]`; `delay_idx` wraps mod `delay_len`.
pub fn echo_tap(output: &mut [f32], buffer: &[f32], delay_len: usize, delay_idx: usize) {
    let n_osc = output.len();
    assert_eq!(
        buffer.len(),
        n_osc * delay_len,
        "echo_tap: buffer length mismatch"
    );
    let idx = delay_idx % delay_len;
    for i in 0..n_osc {
        output[i] = buffer[i * delay_len + idx];
    }
}

/// T3 inverse Kuramoto: estimate drive from amplitude changes.
///
/// Linearized inverse of Stuart-Landau amplitude dynamics:
/// `drive ≈ gain·(r' − r)/dt − drive·decay`.
#[allow(clippy::too_many_arguments)]
pub fn t3_inverse_kuramoto(
    drive_out: &mut [f32],
    r: &[f32],
    r_prev: &[f32],
    _alpha: f32,
    gain: f32,
    eps: f32,
    decay: f32,
    dt: f32,
) {
    let n = drive_out.len();
    assert_eq!(r.len(), n, "t3_inverse_kuramoto: r length mismatch");
    assert_eq!(
        r_prev.len(),
        n,
        "t3_inverse_kuramoto: r_prev length mismatch"
    );
    let dt_inv = f32x8::splat(1.0 / dt.max(eps));
    let gain_s = f32x8::splat(gain);
    let decay_s = f32x8::splat(decay);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let rr = load8(r.as_ptr(), off);
            let rp = load8(r_prev.as_ptr(), off);
            let mut dd = load8(drive_out.as_ptr(), off);
            let rdot = (rr - rp) * dt_inv;
            dd = gain_s * rdot - dd * decay_s;
            store8(drive_out.as_mut_ptr(), off, dd);
        }
    }
    let dt_inv_s = 1.0 / dt.max(eps);
    for i in n8..n {
        let rdot = (r[i] - r_prev[i]) * dt_inv_s;
        drive_out[i] = gain * rdot - drive_out[i] * decay;
    }
}

/// Elementwise ReLU: `x = max(0, x)`.
pub fn relu_simd(data: &mut [f32]) {
    let n = data.len();
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let v = load8(data.as_ptr(), off).max(zero);
            store8(data.as_mut_ptr(), off, v);
        }
    }
    for v in data[n8..].iter_mut() {
        *v = v.max(0.0);
    }
}

/// Elementwise add: `a[i] += b[i]`.
pub fn add_simd(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    assert_eq!(b.len(), n, "add_simd: b length mismatch");
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let va = load8(a.as_ptr(), off);
            let vb = load8(b.as_ptr(), off);
            store8(a.as_mut_ptr(), off, va + vb);
        }
    }
    for i in n8..n {
        a[i] += b[i];
    }
}

/// SGD optimizer step with momentum: `v = mom·v + (1−mom)·g; w -= lr·v`.
pub fn sgd_step_simd(
    velocity: &mut [f32],
    weight: &mut [f32],
    grad: &[f32],
    lr: f32,
    momentum: f32,
) {
    let n = velocity.len();
    assert_eq!(weight.len(), n, "sgd_step_simd: weight length mismatch");
    assert_eq!(grad.len(), n, "sgd_step_simd: grad length mismatch");
    let lr_s = f32x8::splat(lr);
    let mom_s = f32x8::splat(momentum);
    let one_minus_mom = f32x8::splat(1.0 - momentum);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut v = load8(velocity.as_ptr(), off);
            let mut w = load8(weight.as_ptr(), off);
            let g = load8(grad.as_ptr(), off);
            v = mom_s * v + one_minus_mom * g;
            w -= lr_s * v;
            store8(velocity.as_mut_ptr(), off, v);
            store8(weight.as_mut_ptr(), off, w);
        }
    }
    for i in n8..n {
        velocity[i] = momentum * velocity[i] + (1.0 - momentum) * grad[i];
        weight[i] -= lr * velocity[i];
    }
}

/// Adam optimizer step: `m = β₁·m + (1−β₁)·g; v = β₂·v + (1−β₂)·g²; w -= lr·m̂/(√v̂ + ε)`.
#[allow(clippy::too_many_arguments)]
pub fn adam_step_simd(
    weight: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_t_corr: f32,
    beta2_t_corr: f32,
) {
    let n = weight.len();
    assert_eq!(m.len(), n, "adam_step_simd: m length mismatch");
    assert_eq!(v.len(), n, "adam_step_simd: v length mismatch");
    assert_eq!(grad.len(), n, "adam_step_simd: grad length mismatch");
    let lr_s = f32x8::splat(lr);
    let b1_s = f32x8::splat(beta1);
    let b2_s = f32x8::splat(beta2);
    let one_mb1 = f32x8::splat(1.0 - beta1);
    let one_mb2 = f32x8::splat(1.0 - beta2);
    let eps_s = f32x8::splat(eps);
    let b1c_s = f32x8::splat(beta1_t_corr);
    let b2c_s = f32x8::splat(beta2_t_corr);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut w = load8(weight.as_ptr(), off);
            let mut mm = load8(m.as_ptr(), off);
            let mut vv = load8(v.as_ptr(), off);
            let g = load8(grad.as_ptr(), off);

            mm = b1_s * mm + one_mb1 * g;
            vv = b2_s * vv + one_mb2 * g * g;

            let m_hat = mm / b1c_s;
            let v_hat = vv / b2c_s;
            w -= lr_s * m_hat / (v_hat.sqrt() + eps_s);

            store8(weight.as_mut_ptr(), off, w);
            store8(m.as_mut_ptr(), off, mm);
            store8(v.as_mut_ptr(), off, vv);
        }
    }
    for i in n8..n {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let m_hat = m[i] / beta1_t_corr;
        let v_hat = v[i] / beta2_t_corr;
        weight[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Hebbian weight update: `w += lr·cos_diff − decay·w·dt`.
pub fn hebbian_update(weights: &mut [f32], cos_diff: &[f32], lr: f32, decay: f32, dt: f32) {
    let n = weights.len();
    assert_eq!(
        cos_diff.len(),
        n,
        "hebbian_update: cos_diff length mismatch"
    );
    let lr_s = f32x8::splat(lr);
    let decay_dt = f32x8::splat(decay * dt);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let w = load8(weights.as_ptr(), off);
            let cd = load8(cos_diff.as_ptr(), off);
            let w = w + lr_s * cd - decay_dt * w;
            store8(weights.as_mut_ptr(), off, w);
        }
    }
    for i in n8..n {
        weights[i] += lr * cos_diff[i] - decay * weights[i] * dt;
    }
}

// The dopamine weight-update kernel lives in [`super::dopamine`].
// The STDP batch update kernel lives in [`super::learning`].

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_clamp_basic() {
        let mut data = vec![-10.0, -5.0, 0.0, 5.0, 10.0, -3.0, 3.0, 100.0, 0.5];
        safety_clamp(&mut data, -1.0, 1.0);
        for &v in &data {
            assert!((-1.0..=1.0).contains(&v), "safety_clamp: {v} out of range");
        }
    }

    #[test]
    fn sigmoid_gate_basic() {
        let n = 16;
        let input: Vec<f32> = (0..n).map(|i| i as f32 - 8.0).collect();
        let mut output = vec![0.0f32; n];
        sigmoid_gate(&mut output, &input, 1.0, 0.0);
        let mid = output[8];
        assert!(
            (mid - 0.5).abs() < 0.05,
            "sigmoid at 0 should be ~0.5, got {mid}"
        );
        for &v in &output {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {v} out of (0,1)");
        }
    }

    #[test]
    fn t3_drive_encoder_basic() {
        let n = 16;
        let inputs: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut drives = vec![0.0f32; n];
        t3_drive_encoder(&mut drives, &inputs, 10.0);
        for i in 0..n {
            assert!(
                (drives[i] - inputs[i] * 10.0).abs() < 1e-6,
                "drive_encoder: {i}"
            );
        }
    }

    #[test]
    fn relu_simd_basic() {
        let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -0.5, 2.0];
        relu_simd(&mut data);
        let expected = [0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 0.0, 2.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn add_simd_basic() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.5; 8];
        add_simd(&mut a, &b);
        for i in 0..8 {
            assert!((a[i] - (i as f32 + 1.5)).abs() < 1e-6, "add[{i}]");
        }
    }

    #[test]
    fn echo_tap_basic() {
        let n_osc = 4;
        let delay_len = 8;
        let mut buffer = vec![0.0f32; n_osc * delay_len];
        for i in 0..n_osc {
            buffer[i * delay_len + 3] = (i + 1) as f32;
        }
        let mut output = vec![0.0f32; n_osc];
        echo_tap(&mut output, &buffer, delay_len, 3);
        for i in 0..n_osc {
            assert!((output[i] - (i + 1) as f32).abs() < 1e-6, "echo_tap[{i}]");
        }
    }

    #[test]
    fn phase_gate_apply_basic() {
        let n = 16;
        let input = vec![1.0f32; n];
        let r_low = vec![0.0f32; n];
        let r_high = vec![10.0f32; n];
        let mut out_low = vec![0.0f32; n];
        let mut out_high = vec![0.0f32; n];
        phase_gate_apply(&mut out_low, &input, &r_low, 4.0, 1.0);
        phase_gate_apply(&mut out_high, &input, &r_high, 4.0, 1.0);
        assert!(out_low[0] < 0.1, "gate closed: {}", out_low[0]);
        assert!(out_high[0] > 0.9, "gate open: {}", out_high[0]);
    }

    #[test]
    fn thalamic_gate_basic() {
        let n = 8;
        let source_v = vec![1.0f32; n];
        let gate_v_high = vec![10.0f32; n];
        let gate_v_low = vec![-10.0f32; n];
        let mut out_high = vec![0.0f32; n];
        let mut out_low = vec![0.0f32; n];
        thalamic_gate(&mut out_high, &source_v, &gate_v_high, 0.0, 1.0);
        thalamic_gate(&mut out_low, &source_v, &gate_v_low, 0.0, 1.0);
        assert!(out_high[0] > 0.9, "thalamic gate high: {}", out_high[0]);
        assert!(out_low[0] < 0.1, "thalamic gate low: {}", out_low[0]);
    }

    #[test]
    fn linear_decoder_basic() {
        let n_in = 8;
        let n_out = 2;
        let states = vec![1.0f32; n_in];
        let weights: Vec<f32> = (0..(n_in * n_out)).map(|i| i as f32 * 0.1).collect();
        let biases = vec![0.5f32; n_out];
        let mut outputs = vec![0.0f32; n_out];
        linear_decoder(&mut outputs, &states, &weights, &biases);
        for o in 0..n_out {
            let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum::<f32>() + biases[o];
            assert!((outputs[o] - expected).abs() < 1e-4, "linear_decoder[{o}]");
        }
    }

    #[test]
    fn sgd_adam_hebbian_smoke() {
        let mut vel = vec![0.0f32; 16];
        let mut w = vec![0.5f32; 16];
        let g = vec![0.1f32; 16];
        sgd_step_simd(&mut vel, &mut w, &g, 0.1, 0.9);
        assert!(w[0] < 0.5, "sgd should decrease w, got {}", w[0]);

        let mut m = vec![0.0f32; 16];
        let mut v = vec![0.0f32; 16];
        let mut wa = vec![0.5f32; 16];
        adam_step_simd(
            &mut wa, &mut m, &mut v, &g, 0.1, 0.9, 0.999, 1e-8, 0.9, 0.999,
        );
        assert!(wa[0] < 0.5, "adam should decrease w, got {}", wa[0]);

        let mut wh = vec![0.5f32; 16];
        let cd = vec![1.0f32; 16];
        hebbian_update(&mut wh, &cd, 0.1, 0.01, 1.0);
        assert!(wh[0] > 0.5, "hebbian should increase w, got {}", wh[0]);
    }

    #[test]
    fn load8_i32_converts_spikes() {
        let spikes = [0i32, 1, 0, 1, 1, 0, 0, 1, 1];
        unsafe {
            let v = crate::core::util::load8_i32(spikes.as_ptr(), 0);
            let a = v.to_array();
            assert_eq!(a, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        }
    }
}
