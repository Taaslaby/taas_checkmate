//! Safe Rust wrappers for IO and gate SIMD kernels.
//! Ported from `kuramoto_engine/c/io.c` and ASM gate files.

#![allow(non_snake_case)]

use super::safe;

// ─── Safety ──────────────────────────────────────────────

/// Clamp all values in `data` to `[min, max]` using SIMD.
pub fn safety_clamp(data: &mut [f32], min: f32, max: f32) {
    safe::safety_clamp(data, min, max);
}

// ─── PWM Encoder ─────────────────────────────────────────

const GOLDEN_RATIO: f32 = 0.618_034;

/// Pulse-Width Modulation encoder for sensor→spike conversion.
pub struct PwmEncoder {
    num_sensors: usize,
    period: usize,
    tick: usize,
    spike_current: f32,
    phase_shifts: Vec<f32>,
}

impl PwmEncoder {
    pub fn new(n: usize, period: usize, spike_current: f32) -> Self {
        let phase_shifts: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * GOLDEN_RATIO).fract())
            .collect();
        Self {
            num_sensors: n,
            period,
            tick: 0,
            spike_current,
            phase_shifts,
        }
    }

    /// Encode sensor values into spike currents.
    pub fn encode(&mut self, sensors: &[f32], out_currents: &mut [f32]) {
        let t = self.tick as f32 / self.period as f32;
        self.tick = (self.tick + 1) % self.period;
        for i in 0..self.num_sensors.min(sensors.len()).min(out_currents.len()) {
            let threshold = (t + self.phase_shifts[i]).fract();
            out_currents[i] = if sensors[i] > threshold {
                self.spike_current
            } else {
                0.0
            };
        }
    }
}

// ─── Linear Decoder ──────────────────────────────────────

/// Linear decoder: output = W * states + bias.
pub struct LinearDecoder {
    n_inputs: usize,
    n_outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearDecoder {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        Self {
            n_inputs: n_in,
            n_outputs: n_out,
            weights: vec![0.0f32; n_in * n_out],
            bias: vec![0.0f32; n_out],
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Decode: out = W * states + bias.
    pub fn decode(&mut self, states: &[f32], out_buffer: &mut [f32]) {
        assert!(states.len() >= self.n_inputs);
        assert!(out_buffer.len() >= self.n_outputs);

        safe::linear_decoder(
            &mut out_buffer[..self.n_outputs],
            &states[..self.n_inputs],
            &self.weights,
            &self.bias,
        );
    }
}

// ─── Stochastic Encoder ──────────────────────────────────

/// Stochastic encoder: Poisson-like spike encoding with noise comparison.
pub fn stochastic_encoder(
    target_I: &mut [f32], inputs: &[f32], noise: &[f32],
    threshold: f32, spike_val: f32,
) {
    safe::stochastic_encoder(target_I, inputs, noise, threshold, spike_val);
}

// ─── T3 Drive Encoder ────────────────────────────────────

/// T3 drive encoder: gain * inputs.
pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    safe::t3_drive_encoder(drives, inputs, gain);
}

// ─── T3 Complex Decoder ──────────────────────────────────

/// T3 complex decoder: outputs = W * [A*cos(θ); A*sin(θ)] + bias.
pub fn t3_complex_decoder(
    outputs: &mut [f32], amplitudes: &[f32], cos_buf: &[f32], weights: &[f32],
    biases: &[f32], n_out: usize,
) {
    let n_in = amplitudes.len();
    safe::t3_complex_decoder(
        &mut outputs[..n_out],
        amplitudes, cos_buf,
        &weights[..n_in * n_out],
        &biases[..n_out],
    );
}

// ─── T3 PST Features ─────────────────────────────────────

/// Compute phase difference features between two populations.
pub fn t3_pst_features(
    out_sin: &mut [f32], out_cos: &mut [f32],
    ph1: &[f32], ph2: &[f32],
) {
    let (n1, n2) = (ph1.len(), ph2.len());
    safe::t3_pst_features(
        &mut out_sin[..n1 * n2], &mut out_cos[..n1 * n2],
        ph1, ph2,
    );
}

// ─── T3 Tact Generator ───────────────────────────────────

/// Generate TACT pulses from 3 phase populations.
pub fn t3_tact_generator(
    pulses: &mut [f32], ph1: &[f32], ph2: &[f32], ph3: &[f32],
    threshold: f32,
) {
    let n = ph1.len().min(ph2.len()).min(ph3.len()).min(pulses.len());
    safe::t3_tact_generator(&mut pulses[..n], &ph1[..n], &ph2[..n], &ph3[..n], threshold);
}

// ─── T3 Inverse Kuramoto ─────────────────────────────────

/// Inverse Kuramoto: drive = gain * (r - r_prev) - decay * drive.
pub fn t3_inverse_kuramoto(
    drive_out: &mut [f32], r: &[f32], r_prev: &[f32],
    alpha: f32, gain: f32, eps: f32, decay: f32,
) {
    let n = drive_out.len().min(r.len()).min(r_prev.len());
    safe::t3_inverse_kuramoto(&mut drive_out[..n], &r[..n], &r_prev[..n], alpha, gain, eps, decay);
}

// ─── Echo State Tap ──────────────────────────────────────

/// Echo state network tap: read delay line from oscillator buffer.
pub fn echo_tap(
    output: &mut [f32], buffer: &[f32],
    n_osc: usize, delay_len: usize, delay_idx: usize,
) {
    safe::echo_tap(&mut output[..n_osc], &buffer[..n_osc * delay_len], delay_len, delay_idx);
}

// ─── Thalamic Gate ───────────────────────────────────────

/// Thalamic gate: gates source currents by membrane potential.
pub fn thalamic_gate(
    target_I: &mut [f32], source_v: &[f32], gate_v: &[f32],
    threshold: f32, gain: f32,
) {
    let n = target_I.len().min(source_v.len()).min(gate_v.len());
    safe::thalamic_gate(&mut target_I[..n], &source_v[..n], &gate_v[..n], threshold, gain);
}

// ─── Sigmoid Gate ────────────────────────────────────────

/// Sigmoid gate: output = 1 / (1 + exp(-gain * (input - threshold))).
pub fn sigmoid_gate(
    output: &mut [f32], input: &[f32], gain: f32, threshold: f32,
) {
    let n = output.len().min(input.len());
    safe::sigmoid_gate(&mut output[..n], &input[..n], gain, threshold);
}

// ─── Phase Gate ──────────────────────────────────────────

/// Phase gate: output = input * sigmoid(gain * (r - threshold)).
pub fn phase_gate_apply(
    output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32,
) {
    let n = output.len().min(input.len()).min(r.len());
    safe::phase_gate_apply(&mut output[..n], &input[..n], &r[..n], gain, threshold);
}

// ─── Hebbian Update ──────────────────────────────────────

/// Hebbian update: w += lr * cos(Δθ) - decay * w * dt.
pub fn hebbian_update(
    weights: &mut [f32], cos_diff: &[f32], lr: f32, decay: f32, dt: f32,
) {
    let n = weights.len().min(cos_diff.len());
    safe::hebbian_update(&mut weights[..n], &cos_diff[..n], lr, decay, dt);
}
