//! # 🔌 IO / gate compat adapters
//!
//! Min-truncating adapters moved verbatim from the deleted `taas-kernels`
//! compat crate. The canonical implementations live in
//! [`crate::kernels::signal`] / [`crate::kernels::decode`] /
//! [`crate::kernels::input`] (assert-семантика там жёстче); these wrappers
//! keep the **min-семантику** (усечение по минимальной длине — контракт
//! python ABI) and delegate straight to the canon. No math here — only slice
//! adaptation.

#![allow(non_snake_case)]

// ─── Safety ──────────────────────────────────────────────

/// Clamp all values in `data` to `[min, max]` — canon in
/// [`crate::kernels::signal`].
pub use crate::kernels::signal::safety_clamp;

// ─── PWM Encoder (канон: crate::kernels::input) ──────────

pub use crate::kernels::input::PwmEncoder;

// ─── Linear Decoder (канон: crate::kernels::input) ───────

pub use crate::kernels::input::LinearDecoder;

// ─── Stochastic Encoder ──────────────────────────────────

/// Stochastic encoder: Poisson-like spike encoding with noise comparison.
pub fn stochastic_encoder(
    target_I: &mut [f32],
    inputs: &[f32],
    noise: &[f32],
    threshold: f32,
    spike_val: f32,
) {
    let n = target_I.len().min(inputs.len()).min(noise.len());
    crate::kernels::signal::stochastic_encoder(
        &mut target_I[..n],
        &inputs[..n],
        &noise[..n],
        threshold,
        spike_val,
    );
}

// ─── T3 Drive Encoder ────────────────────────────────────

/// T3 drive encoder: gain * inputs.
pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    let n = drives.len().min(inputs.len());
    crate::kernels::signal::t3_drive_encoder(&mut drives[..n], &inputs[..n], gain);
}

// ─── T3 Complex Decoder ──────────────────────────────────

/// T3 complex decoder: outputs = W * [A*cos(θ); A*sin(θ)] + bias.
pub fn t3_complex_decoder(
    outputs: &mut [f32],
    amplitudes: &[f32],
    cos_buf: &[f32],
    weights: &[f32],
    biases: &[f32],
    n_out: usize,
) {
    let n_in = amplitudes.len().min(cos_buf.len());
    let n_out = n_out.min(outputs.len()).min(biases.len());
    let n_w = n_in * n_out;
    crate::kernels::decode::t3_complex_decoder(
        &mut outputs[..n_out],
        &amplitudes[..n_in],
        &cos_buf[..n_in],
        &weights[..n_w],
        &biases[..n_out],
    );
}

// ─── T3 PST Features ─────────────────────────────────────

/// Compute phase difference features between two populations.
///
/// Контракт канона: `out_sin/out_cos` должны быть ровно `ph1.len()·ph2.len()`
/// (assert-семантика, как в легаси `safe::t3_pst_features`) — min-усечение
/// здесь невозможно, т.к. канон проверяет точную длину.
pub fn t3_pst_features(out_sin: &mut [f32], out_cos: &mut [f32], ph1: &[f32], ph2: &[f32]) {
    let (n1, n2) = (ph1.len(), ph2.len());
    let n = n1 * n2;
    crate::kernels::decode::t3_pst_features(&mut out_sin[..n], &mut out_cos[..n], ph1, ph2);
}

// ─── T3 Tact Generator ───────────────────────────────────

/// Generate TACT pulses from 3 phase populations.
pub fn t3_tact_generator(
    pulses: &mut [f32],
    ph1: &[f32],
    ph2: &[f32],
    ph3: &[f32],
    threshold: f32,
) {
    let n = pulses.len().min(ph1.len()).min(ph2.len()).min(ph3.len());
    crate::kernels::signal::t3_tact_generator(
        &mut pulses[..n],
        &ph1[..n],
        &ph2[..n],
        &ph3[..n],
        threshold,
    );
}

// ─── T3 Inverse Kuramoto ─────────────────────────────────

/// Inverse Kuramoto: drive = gain * (r - r_prev) / dt - drive * decay.
/// Inverts Stuart-Landau amplitude dynamics to estimate input drive.
pub fn t3_inverse_kuramoto(
    drive_out: &mut [f32],
    r: &[f32],
    r_prev: &[f32],
    alpha: f32,
    gain: f32,
    eps: f32,
    decay: f32,
    dt: f32,
) {
    let n = drive_out.len().min(r.len()).min(r_prev.len());
    crate::kernels::signal::t3_inverse_kuramoto(
        &mut drive_out[..n],
        &r[..n],
        &r_prev[..n],
        alpha,
        gain,
        eps,
        decay,
        dt,
    );
}

// ─── Echo State Tap ──────────────────────────────────────

/// Echo state network tap: read delay line from oscillator buffer.
pub fn echo_tap(
    output: &mut [f32],
    buffer: &[f32],
    n_osc: usize,
    delay_len: usize,
    delay_idx: usize,
) {
    let n_osc = n_osc.min(output.len());
    let n_buf = n_osc * delay_len;
    let n_buf = n_buf.min(buffer.len());
    crate::kernels::signal::echo_tap(&mut output[..n_osc], &buffer[..n_buf], delay_len, delay_idx);
}

// ─── Thalamic Gate ───────────────────────────────────────

/// Thalamic gate: gates source currents by membrane potential.
pub fn thalamic_gate(
    target_I: &mut [f32],
    source_v: &[f32],
    gate_v: &[f32],
    threshold: f32,
    gain: f32,
) {
    let n = target_I.len().min(source_v.len()).min(gate_v.len());
    crate::kernels::signal::thalamic_gate(
        &mut target_I[..n],
        &source_v[..n],
        &gate_v[..n],
        threshold,
        gain,
    );
}

// ─── Sigmoid Gate ────────────────────────────────────────

/// Sigmoid gate: output = 1 / (1 + exp(-gain * (input - threshold))).
pub fn sigmoid_gate(output: &mut [f32], input: &[f32], gain: f32, threshold: f32) {
    let n = output.len().min(input.len());
    crate::kernels::signal::sigmoid_gate(&mut output[..n], &input[..n], gain, threshold);
}

// ─── Phase Gate ──────────────────────────────────────────

/// Phase gate: output = input * sigmoid(gain * (r - threshold)).
pub fn phase_gate_apply(output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32) {
    let n = output.len().min(input.len()).min(r.len());
    crate::kernels::signal::phase_gate_apply(
        &mut output[..n],
        &input[..n],
        &r[..n],
        gain,
        threshold,
    );
}

// ─── Hebbian Update ──────────────────────────────────────

/// Hebbian update: w += lr * cos(Δθ) - decay * w * dt.
pub fn hebbian_update(weights: &mut [f32], cos_diff: &[f32], lr: f32, decay: f32, dt: f32) {
    let n = weights.len().min(cos_diff.len());
    crate::kernels::signal::hebbian_update(&mut weights[..n], &cos_diff[..n], lr, decay, dt);
}
