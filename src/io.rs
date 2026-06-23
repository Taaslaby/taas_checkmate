//! Safe Rust wrappers for IO and gate ASM kernels.
//! Ported from `kuramoto_engine/c/io.c` and ASM gate files.

#![allow(non_snake_case)]

use crate::kernels;

// ─── Safety ──────────────────────────────────────────────

/// Clamp all values in `data` to `[min, max]` using AVX.
pub fn safety_clamp(data: &mut [f32], min: f32, max: f32) {
    unsafe {
        kernels::safety_clamp(
            data.as_mut_ptr(),
            data.len(),
            min,
            max,
        );
    }
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
    /// Uses AVX `linear_decoder_avx` if available, else scalar fallback.
    pub fn decode(&mut self, states: &[f32], out_buffer: &mut [f32]) {
        assert!(states.len() >= self.n_inputs);
        assert!(out_buffer.len() >= self.n_outputs);

        // ASM path (optimized)
        unsafe {
            kernels::linear_decoder(
                out_buffer.as_mut_ptr(), states.as_ptr(),
                self.weights.as_ptr(), self.bias.as_ptr(),
                self.n_inputs, self.n_outputs,
            );
        }
    }
}

// ─── Stochastic Encoder ──────────────────────────────────

/// Stochastic encoder: Poisson-like spike encoding with noise comparison.
pub fn stochastic_encoder(
    target_I: &mut [f32], inputs: &[f32], noise: &[f32],
    threshold: f32, spike_val: f32,
) {
    let n = target_I.len().min(inputs.len()).min(noise.len());
    unsafe {
        kernels::stochastic_encoder(
            target_I.as_mut_ptr(), inputs.as_ptr(), noise.as_ptr(),
            threshold, spike_val, n,
        );
    }
}

// ─── T3 Drive Encoder ────────────────────────────────────

/// T3 drive encoder: gain * inputs.
pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    let n = drives.len().min(inputs.len());
    unsafe {
        kernels::t3_drive_encoder(
            drives.as_mut_ptr(), inputs.as_ptr(), gain, n,
        );
    }
}

// ─── T3 Complex Decoder ──────────────────────────────────

/// T3 complex decoder: outputs = W * [A*cos(θ); A*sin(θ)] + bias.
pub fn t3_complex_decoder(
    outputs: &mut [f32], amplitudes: &[f32], cos_buf: &[f32], weights: &[f32],
    biases: &[f32], n_out: usize,
) {
    let n_in = amplitudes.len();
    assert!(n_in == cos_buf.len());
    assert!(weights.len() >= n_in * n_out);
    assert!(biases.len() >= n_out);
    assert!(outputs.len() >= n_out);
    unsafe {
        kernels::t3_complex_decoder(
            outputs.as_mut_ptr(), amplitudes.as_ptr(), cos_buf.as_ptr(),
            weights.as_ptr(), biases.as_ptr(),
            n_in, n_out,
        );
    }
}

// ─── T3 PST Features ─────────────────────────────────────

/// Compute phase difference features between two populations.
pub fn t3_pst_features(
    out_sin: &mut [f32], out_cos: &mut [f32],
    ph1: &[f32], ph2: &[f32],
) {
    let n1 = ph1.len();
    let n2 = ph2.len();
    assert!(out_sin.len() >= n1 * n2 && out_cos.len() >= n1 * n2);
    unsafe {
        kernels::t3_pst_features(
            out_sin.as_mut_ptr(), out_cos.as_mut_ptr(),
            ph1.as_ptr(), ph2.as_ptr(),
            n1, n2,
        );
    }
}

// ─── T3 Tact Generator ───────────────────────────────────

/// Generate TACT pulses from 3 phase populations.
pub fn t3_tact_generator(
    pulses: &mut [f32], ph1: &[f32], ph2: &[f32], ph3: &[f32],
    threshold: f32,
) {
    let n = ph1.len().min(ph2.len()).min(ph3.len());
    assert!(pulses.len() >= n);
    unsafe {
        kernels::t3_tact_generator(
            pulses.as_mut_ptr(), ph1.as_ptr(), ph2.as_ptr(), ph3.as_ptr(),
            threshold, n,
        );
    }
}

// ─── T3 Inverse Kuramoto ─────────────────────────────────

/// Inverse Kuramoto: drive = gain * (r - r_prev) - decay * drive.
pub fn t3_inverse_kuramoto(
    drive_out: &mut [f32], r: &[f32], r_prev: &[f32],
    alpha: f32, gain: f32, eps: f32, decay: f32,
) {
    let n = drive_out.len().min(r.len()).min(r_prev.len());
    unsafe {
        kernels::t3_inverse_kuramoto(
            drive_out.as_mut_ptr(), r.as_ptr(), r_prev.as_ptr(),
            alpha, gain, eps, decay, n,
        );
    }
}

// ─── Echo State Tap ──────────────────────────────────────

/// Echo state network tap: read delay line from oscillator buffer.
pub fn echo_tap(
    output: &mut [f32], buffer: &[f32],
    n_osc: usize, delay_len: usize, delay_idx: usize,
) {
    unsafe {
        kernels::echo_tap(
            output.as_mut_ptr(), buffer.as_ptr(),
            n_osc, delay_len, delay_idx,
        );
    }
}

// ─── Thalamic Gate ───────────────────────────────────────

/// Thalamic gate: gates source currents by membrane potential.
pub fn thalamic_gate(
    target_I: &mut [f32], source_v: &[f32], gate_v: &[f32],
    threshold: f32, gain: f32,
) {
    let n = target_I.len().min(source_v.len()).min(gate_v.len());
    unsafe {
        kernels::thalamic_gate(
            target_I.as_mut_ptr(), source_v.as_ptr(), gate_v.as_ptr(),
            threshold, gain, n,
        );
    }
}

// ─── Sigmoid Gate ────────────────────────────────────────

/// Sigmoid gate: output = 1 / (1 + exp(-gain * (input - threshold))).
pub fn sigmoid_gate(
    output: &mut [f32], input: &[f32], gain: f32, threshold: f32,
) {
    let n = output.len().min(input.len());
    unsafe {
        kernels::sigmoid_gate(
            output.as_mut_ptr(), input.as_ptr(), gain, threshold,
            n,
        );
    }
}

// ─── Phase Gate ──────────────────────────────────────────

/// Phase gate: output = input * sigmoid(gain * (r - threshold)).
pub fn phase_gate_apply(
    output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32,
) {
    let n = output.len().min(input.len()).min(r.len());
    unsafe {
        kernels::phase_gate_apply(
            output.as_mut_ptr(), input.as_ptr(), r.as_ptr(),
            gain, threshold, n,
        );
    }
}

// ─── Hebbian Update ──────────────────────────────────────

/// Hebbian update: w += lr * cos(Δθ) - decay * w * dt.
pub fn hebbian_update(
    weights: &mut [f32], cos_diff: &[f32], lr: f32, decay: f32, dt: f32,
) {
    let n = weights.len().min(cos_diff.len());
    unsafe {
        kernels::hebbian_update(
            weights.as_mut_ptr(), cos_diff.as_ptr(), lr, decay, dt,
            n,
        );
    }
}
