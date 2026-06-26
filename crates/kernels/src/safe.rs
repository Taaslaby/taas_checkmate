//! Safe wrappers over all SIMD kernel functions.
//!
//! Each function validates slice lengths and alignment requirements before
//! delegating to the corresponding `pub(crate) unsafe` kernel implementation.
//! All SIMD alignment invariants (`n % 8 == 0` not required — scalar tail handled)
//! are satisfied by the calling convention.

use super::{attention, decode, measure, phase, signal};
use super::neuron::{adex, delay, dopamine, fused, izhikevich, lif, saw, stp};

// ═══════════════════════════════════════════════════════════
//  signal
// ═══════════════════════════════════════════════════════════

pub fn safety_clamp(data: &mut [f32], min: f32, max: f32) {
    let n = data.len();
    unsafe { signal::safety_clamp(data.as_mut_ptr(), n, min, max) }
}

pub fn stochastic_encoder(target_i: &mut [f32], inputs: &[f32], noise: &[f32], threshold: f32, spike_val: f32) {
    let n = target_i.len();
    assert_eq!(inputs.len(), n, "stochastic_encoder: inputs length mismatch");
    assert_eq!(noise.len(), n, "stochastic_encoder: noise length mismatch");
    unsafe { signal::stochastic_encoder(target_i.as_mut_ptr(), inputs.as_ptr(), noise.as_ptr(), threshold, spike_val, n) }
}

pub fn linear_decoder(outputs: &mut [f32], states: &[f32], weights: &[f32], biases: &[f32]) {
    let n_out = outputs.len();
    let n_in = states.len();
    assert_eq!(biases.len(), n_out, "linear_decoder: biases length mismatch");
    assert_eq!(weights.len(), n_in * n_out, "linear_decoder: weights length mismatch (expected {}*{}={})", n_in, n_out, n_in * n_out);
    unsafe { signal::linear_decoder(outputs.as_mut_ptr(), states.as_ptr(), weights.as_ptr(), biases.as_ptr(), n_in, n_out) }
}

pub fn thalamic_gate(target_i: &mut [f32], source_v: &[f32], gate_v: &[f32], threshold: f32, gain: f32) {
    let n = target_i.len();
    assert_eq!(source_v.len(), n, "thalamic_gate: source_v length mismatch");
    assert_eq!(gate_v.len(), n, "thalamic_gate: gate_v length mismatch");
    unsafe { signal::thalamic_gate(target_i.as_mut_ptr(), source_v.as_ptr(), gate_v.as_ptr(), threshold, gain, n) }
}

pub fn sigmoid_gate(output: &mut [f32], input: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "sigmoid_gate: input length mismatch");
    unsafe { signal::sigmoid_gate(output.as_mut_ptr(), input.as_ptr(), gain, threshold, n) }
}

pub fn phase_gate_apply(output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "phase_gate_apply: input length mismatch");
    assert_eq!(r.len(), n, "phase_gate_apply: r length mismatch");
    unsafe { signal::phase_gate_apply(output.as_mut_ptr(), input.as_ptr(), r.as_ptr(), gain, threshold, n) }
}

pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    let n = drives.len();
    assert_eq!(inputs.len(), n, "t3_drive_encoder: inputs length mismatch");
    unsafe { signal::t3_drive_encoder(drives.as_mut_ptr(), inputs.as_ptr(), gain, n) }
}

pub fn t3_tact_generator(pulses: &mut [f32], ph1: &[f32], ph2: &[f32], ph3: &[f32], threshold: f32) {
    let n = pulses.len();
    assert_eq!(ph1.len(), n, "t3_tact_generator: ph1 length mismatch");
    assert_eq!(ph2.len(), n, "t3_tact_generator: ph2 length mismatch");
    assert_eq!(ph3.len(), n, "t3_tact_generator: ph3 length mismatch");
    unsafe { signal::t3_tact_generator(pulses.as_mut_ptr(), ph1.as_ptr(), ph2.as_ptr(), ph3.as_ptr(), threshold, n) }
}

pub fn echo_tap(output: &mut [f32], buffer: &[f32], delay_len: usize, delay_idx: usize) {
    let n_osc = output.len();
    assert_eq!(buffer.len(), n_osc * delay_len, "echo_tap: buffer length mismatch");
    unsafe { signal::echo_tap(output.as_mut_ptr(), buffer.as_ptr(), n_osc, delay_len, delay_idx) }
}

pub fn t3_inverse_kuramoto(drive_out: &mut [f32], r: &[f32], r_prev: &[f32], alpha: f32, gain: f32, eps: f32, decay: f32) {
    let n = drive_out.len();
    assert_eq!(r.len(), n, "t3_inverse_kuramoto: r length mismatch");
    assert_eq!(r_prev.len(), n, "t3_inverse_kuramoto: r_prev length mismatch");
    unsafe { signal::t3_inverse_kuramoto(drive_out.as_mut_ptr(), r.as_ptr(), r_prev.as_ptr(), alpha, gain, eps, decay, n) }
}

pub fn relu_simd(data: &mut [f32]) {
    let n = data.len();
    unsafe { signal::relu_simd(data.as_mut_ptr(), n) }
}

pub fn add_simd(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    assert_eq!(b.len(), n, "add_simd: b length mismatch");
    unsafe { signal::add_simd(a.as_mut_ptr(), b.as_ptr(), n) }
}

pub fn sgd_step_simd(velocity: &mut [f32], weight: &mut [f32], grad: &[f32], lr: f32, momentum: f32) {
    let n = velocity.len();
    assert_eq!(weight.len(), n, "sgd_step_simd: weight length mismatch");
    assert_eq!(grad.len(), n, "sgd_step_simd: grad length mismatch");
    unsafe { signal::sgd_step_simd(velocity.as_mut_ptr(), weight.as_mut_ptr(), grad.as_ptr(), lr, momentum, n) }
}

pub fn adam_step_simd(weight: &mut [f32], m: &mut [f32], v: &mut [f32], grad: &[f32], lr: f32, beta1: f32, beta2: f32, eps: f32, beta1_t_corr: f32, beta2_t_corr: f32) {
    let n = weight.len();
    assert_eq!(m.len(), n, "adam_step_simd: m length mismatch");
    assert_eq!(v.len(), n, "adam_step_simd: v length mismatch");
    assert_eq!(grad.len(), n, "adam_step_simd: grad length mismatch");
    unsafe { signal::adam_step_simd(weight.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(), grad.as_ptr(), lr, beta1, beta2, eps, beta1_t_corr, beta2_t_corr, n) }
}

pub fn hebbian_update(weights: &mut [f32], cos_diff: &[f32], lr: f32, decay: f32, dt: f32) {
    let n = weights.len();
    assert_eq!(cos_diff.len(), n, "hebbian_update: cos_diff length mismatch");
    unsafe { signal::hebbian_update(weights.as_mut_ptr(), cos_diff.as_ptr(), lr, decay, dt, n) }
}

// ═══════════════════════════════════════════════════════════
//  decode
// ═══════════════════════════════════════════════════════════

pub fn t3_complex_decoder(outputs: &mut [f32], amplitudes: &[f32], cos_buf: &[f32], weights: &[f32], biases: &[f32]) {
    let n_out = outputs.len();
    let n_in = amplitudes.len();
    assert_eq!(cos_buf.len(), n_in, "t3_complex_decoder: cos_buf length mismatch");
    assert_eq!(biases.len(), n_out, "t3_complex_decoder: biases length mismatch");
    assert_eq!(weights.len(), n_in * n_out, "t3_complex_decoder: weights length mismatch");
    unsafe { decode::t3_complex_decoder(outputs.as_mut_ptr(), amplitudes.as_ptr(), cos_buf.as_ptr(), weights.as_ptr(), biases.as_ptr(), n_in, n_out) }
}

pub fn t3_pst_features(out_sin: &mut [f32], out_cos: &mut [f32], ph1: &[f32], ph2: &[f32]) {
    let n1 = ph1.len();
    let n2 = ph2.len();
    assert_eq!(out_sin.len(), n1 * n2, "t3_pst_features: out_sin length mismatch");
    assert_eq!(out_cos.len(), n1 * n2, "t3_pst_features: out_cos length mismatch");
    unsafe { decode::t3_pst_features(out_sin.as_mut_ptr(), out_cos.as_mut_ptr(), ph1.as_ptr(), ph2.as_ptr(), n1, n2) }
}

pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32) {
    let n = weights.len();
    assert_eq!(states.len(), n, "t3_lms_train: states length mismatch");
    assert!(!bias.is_empty(), "t3_lms_train: bias must have at least 1 element");
    unsafe { decode::t3_lms_train(weights.as_mut_ptr(), bias.as_mut_ptr(), states.as_ptr(), error, lr, n) }
}

pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32, eps: f32) {
    let n = weights.len();
    assert_eq!(states.len(), n, "nlms_train: states length mismatch");
    assert!(!bias.is_empty(), "nlms_train: bias must have at least 1 element");
    unsafe { decode::nlms_train(weights.as_mut_ptr(), bias.as_mut_ptr(), states.as_ptr(), error, lr, eps, n) }
}

pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32], error: f32, s: f32) {
    let n = weights.len();
    assert_eq!(cov.len(), n, "kalman_update: cov length mismatch");
    assert_eq!(states.len(), n, "kalman_update: states length mismatch");
    unsafe { decode::kalman_update(weights.as_mut_ptr(), cov.as_mut_ptr(), states.as_ptr(), error, s, n) }
}

pub fn adam_train(weights: &mut [f32], m: &mut [f32], v: &mut [f32], states: &[f32], error: f32, lr: f32, beta1: f32, beta2: f32, bc1: f32, _bc2: f32, eps: f32) {
    let n = weights.len();
    assert_eq!(m.len(), n, "adam_train: m length mismatch");
    assert_eq!(v.len(), n, "adam_train: v length mismatch");
    assert_eq!(states.len(), n, "adam_train: states length mismatch");
    unsafe { decode::adam_train(weights.as_mut_ptr(), m.as_mut_ptr(), v.as_mut_ptr(), states.as_ptr(), error, lr, beta1, beta2, bc1, _bc2, eps, n) }
}

pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32], l: usize, pos: usize) {
    let feat_dim = output.len();
    assert_eq!(buffer.len(), l * feat_dim, "pst_decode: buffer length mismatch");
    assert_eq!(weights.len(), l * feat_dim, "pst_decode: weights length mismatch");
    unsafe { decode::pst_decode(output.as_mut_ptr(), buffer.as_ptr(), weights.as_ptr(), feat_dim, l, pos) }
}

pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32], weights: &[f32]) {
    let n = output.len();
    assert_eq!(amplitudes.len(), n, "ring_synthesis: amplitudes length mismatch");
    assert_eq!(trig_buf.len(), n, "ring_synthesis: trig_buf length mismatch");
    assert_eq!(weights.len(), n, "ring_synthesis: weights length mismatch");
    unsafe { decode::ring_synthesis(output.as_mut_ptr(), amplitudes.as_ptr(), trig_buf.as_ptr(), weights.as_ptr(), n) }
}

// ═══════════════════════════════════════════════════════════
//  measure
// ═══════════════════════════════════════════════════════════

pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(output.len(), na * nb, "interference: output length mismatch");
    unsafe { measure::interference(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), na, nb) }
}

pub fn interference_mean(output: &mut f32, phases_a: &[f32], phases_b: &[f32]) {
    let n = phases_a.len();
    assert_eq!(phases_b.len(), n, "interference_mean: phases_b length mismatch");
    unsafe { measure::interference_mean(output as *mut f32, phases_a.as_ptr(), phases_b.as_ptr(), n) }
}

pub fn phase_coherence(output: &mut f32, sin_buf: &[f32], cos_buf: &[f32], amplitudes: &[f32]) {
    let n = sin_buf.len();
    assert_eq!(cos_buf.len(), n, "phase_coherence: cos_buf length mismatch");
    assert_eq!(amplitudes.len(), n, "phase_coherence: amplitudes length mismatch");
    unsafe { measure::phase_coherence(output as *mut f32, sin_buf.as_ptr(), cos_buf.as_ptr(), amplitudes.as_ptr(), n) }
}

pub fn band_energy(band_energies: &mut [f32], amplitudes: &[f32], omegas: &[f32], band_edges: &[f32]) {
    let n_bands = band_energies.len();
    let n = amplitudes.len();
    assert_eq!(omegas.len(), n, "band_energy: omegas length mismatch");
    assert_eq!(band_edges.len(), n_bands + 1, "band_energy: band_edges length mismatch (expected {})", n_bands + 1);
    unsafe { measure::band_energy(band_energies.as_mut_ptr(), amplitudes.as_ptr(), omegas.as_ptr(), band_edges.as_ptr(), n_bands, n) }
}

pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(output.len(), na * nb, "winding_matrix: output length mismatch");
    unsafe { measure::winding_matrix(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), na, nb) }
}

pub fn lazy_sincos_update(sin_out: &mut [f32], cos_out: &mut [f32], phases: &[f32], phases_prev: &mut [f32], eps: f32) {
    let n = sin_out.len();
    assert_eq!(cos_out.len(), n, "lazy_sincos_update: cos_out length mismatch");
    assert_eq!(phases.len(), n, "lazy_sincos_update: phases length mismatch");
    assert_eq!(phases_prev.len(), n, "lazy_sincos_update: phases_prev length mismatch");
    unsafe { measure::lazy_sincos_update(sin_out.as_mut_ptr(), cos_out.as_mut_ptr(), phases.as_ptr(), phases_prev.as_mut_ptr(), eps, n) }
}

pub fn batch_coherence_sweep(output: &mut [f32], areas: &[f32], areas_sin: &[f32], areas_cos: &[f32], n_elem: usize) {
    let total = areas.len();
    let n_areas = total / n_elem;
    assert_eq!(total, n_areas * n_elem, "batch_coherence_sweep: areas length not divisible by n_elem");
    assert_eq!(output.len(), n_areas * n_areas, "batch_coherence_sweep: output must be n_areas², got {} expected {}", output.len(), n_areas * n_areas);
    assert_eq!(areas_sin.len(), total, "batch_coherence_sweep: areas_sin length mismatch");
    assert_eq!(areas_cos.len(), total, "batch_coherence_sweep: areas_cos length mismatch");
    unsafe { measure::batch_coherence_sweep(output.as_mut_ptr(), areas.as_ptr(), areas_sin.as_ptr(), areas_cos.as_ptr(), n_areas, n_elem) }
}

// ═══════════════════════════════════════════════════════════
//  phase
// ═══════════════════════════════════════════════════════════

pub fn kuramoto_prepare(phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
    let n = phases.len();
    assert_eq!(cos_out.len(), n, "kuramoto_prepare: cos_out length mismatch");
    assert_eq!(sin_out.len(), n, "kuramoto_prepare: sin_out length mismatch");
    unsafe { phase::kuramoto_prepare(phases.as_mut_ptr(), cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n) }
}

pub fn kuramoto_update(phases: &mut [f32], omegas: &[f32], coupling: &[f32], dt: f32) {
    let n = phases.len();
    assert_eq!(omegas.len(), n, "kuramoto_update: omegas length mismatch");
    assert_eq!(coupling.len(), n, "kuramoto_update: coupling length mismatch");
    unsafe { phase::kuramoto_update(phases.as_mut_ptr(), omegas.as_ptr(), coupling.as_ptr(), dt, n) }
}

pub fn t3_update(phases: &mut [f32], amplitudes: &mut [f32], omegas: &[f32], coupling: &[f32], drives: &[f32], dt: f32, decay: f32, alpha: f32, gain: f32) {
    let n = phases.len();
    assert_eq!(amplitudes.len(), n, "t3_update: amplitudes length mismatch");
    assert_eq!(omegas.len(), n, "t3_update: omegas length mismatch");
    assert_eq!(coupling.len(), n, "t3_update: coupling length mismatch");
    assert_eq!(drives.len(), n, "t3_update: drives length mismatch");
    unsafe { phase::t3_update(phases.as_mut_ptr(), amplitudes.as_mut_ptr(), omegas.as_ptr(), coupling.as_ptr(), drives.as_ptr(), dt, decay, alpha, gain, n) }
}

// ═══════════════════════════════════════════════════════════
//  attention
// ═══════════════════════════════════════════════════════════

pub fn xor_attention(query: &[u8], keys: &[u8], values: &[f32], out: &mut f32) {
    let n_bytes = query.len();
    let n_keys = values.len();
    assert_eq!(keys.len(), n_keys * n_bytes, "xor_attention: keys length mismatch");
    unsafe { attention::xor_attention(query.as_ptr(), keys.as_ptr(), values.as_ptr(), out as *mut f32, n_keys, n_bytes) }
}

pub fn rdrand_fill(buf: &mut [f32]) {
    let n = buf.len();
    unsafe { attention::rdrand_fill(buf.as_mut_ptr(), n) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / izhikevich
// ═══════════════════════════════════════════════════════════

pub fn izhikevich_step(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], Ii: &[f32],
    dt: f32, trace_decay: f32,
    spikes: &mut [i32], c: &[f32], d: &[f32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "izhikevich_step: u length mismatch");
    assert_eq!(trace.len(), n, "izhikevich_step: trace length mismatch");
    assert_eq!(a.len(), n, "izhikevich_step: a length mismatch");
    assert_eq!(b.len(), n, "izhikevich_step: b length mismatch");
    assert_eq!(Ii.len(), n, "izhikevich_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "izhikevich_step: spikes length mismatch");
    assert_eq!(c.len(), n, "izhikevich_step: c length mismatch");
    assert_eq!(d.len(), n, "izhikevich_step: d length mismatch");
    assert_eq!(refractory.len(), n, "izhikevich_step: refractory length mismatch");
    unsafe { izhikevich::izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(), a.as_ptr(), b.as_ptr(), Ii.as_ptr(), dt, trace_decay, spikes.as_mut_ptr(), c.as_ptr(), d.as_ptr(), refractory.as_mut_ptr(), tau_ref, n) }
}

pub fn eprop_izhikevich_step(
    v: &mut [f32], u: &mut [f32], eligibility: &mut [f32], psi: &mut [f32],
    Ii: &[f32], spikes: &mut [i32], pre_spikes: &[i32],
    dt: f32, tau_e: f32, a: f32, b: f32, c: f32, d: f32, tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_izhikevich_step: u length mismatch");
    assert_eq!(eligibility.len(), n, "eprop_izhikevich_step: eligibility length mismatch");
    assert_eq!(psi.len(), n, "eprop_izhikevich_step: psi length mismatch");
    assert_eq!(Ii.len(), n, "eprop_izhikevich_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_izhikevich_step: spikes length mismatch");
    assert_eq!(pre_spikes.len(), n, "eprop_izhikevich_step: pre_spikes length mismatch");
    unsafe { izhikevich::eprop_izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), eligibility.as_mut_ptr(), psi.as_mut_ptr(), Ii.as_ptr(), spikes.as_mut_ptr(), pre_spikes.as_ptr(), dt, tau_e, a, b, c, d, tau_sg, n) }
}

pub fn eprop_izhikevich_step_multi(
    v: &mut [f32], u: &mut [f32], psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    I_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32, tau_e: f32, a: f32, b: f32, c: f32, d: f32, tau_sg: f32,
    spikes: &mut [i32],
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_izhikevich_step_multi: u length mismatch");
    assert_eq!(psi.len(), n, "eprop_izhikevich_step_multi: psi length mismatch");
    assert_eq!(e.len(), n_pre * n, "eprop_izhikevich_step_multi: e length mismatch");
    assert_eq!(I_ext.len(), n, "eprop_izhikevich_step_multi: I_ext length mismatch");
    assert_eq!(weights.len(), n_pre * n, "eprop_izhikevich_step_multi: weights length mismatch");
    assert_eq!(spikes.len(), n, "eprop_izhikevich_step_multi: spikes length mismatch");
    unsafe { izhikevich::eprop_izhikevich_step_multi(v.as_mut_ptr(), u.as_mut_ptr(), psi.as_mut_ptr(), e.as_mut_ptr(), z_pre.as_ptr(), I_ext.as_ptr(), weights.as_ptr(), n, n_pre, dt, tau_e, a, b, c, d, tau_sg, spikes.as_mut_ptr()) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / lif
// ═══════════════════════════════════════════════════════════

pub fn lif_step(v: &mut [f32], Ii: &mut [f32], spikes: &mut [i32], params: &[f32], refractory: &mut [f32], tau_ref: f32) {
    let n = v.len();
    assert_eq!(Ii.len(), n, "lif_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "lif_step: spikes length mismatch");
    assert_eq!(params.len(), n * 5, "lif_step: params length mismatch (expected {}*5={})", n, n * 5);
    assert_eq!(refractory.len(), n, "lif_step: refractory length mismatch");
    unsafe { lif::lif_step(v.as_mut_ptr(), Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(), refractory.as_mut_ptr(), tau_ref, n) }
}

pub fn lif_at_step(v: &mut [f32], theta: &mut [f32], Ii: &mut [f32], spikes: &mut [i32], params: &[f32], refractory: &mut [f32], tau_ref: f32) {
    let n = v.len();
    assert_eq!(theta.len(), n, "lif_at_step: theta length mismatch");
    assert_eq!(Ii.len(), n, "lif_at_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "lif_at_step: spikes length mismatch");
    assert_eq!(params.len(), n * 7, "lif_at_step: params length mismatch (expected {}*7={})", n, n * 7);
    assert_eq!(refractory.len(), n, "lif_at_step: refractory length mismatch");
    unsafe { lif::lif_at_step(v.as_mut_ptr(), theta.as_mut_ptr(), Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(), refractory.as_mut_ptr(), tau_ref, n) }
}

pub fn eprop_lif_step(
    v: &mut [f32], theta: &mut [f32], eligibility: &mut [f32], psi: &mut [f32],
    Ii: &[f32], spikes: &mut [i32], pre_spikes: &[i32],
    dt: f32, tau_m: f32, tau_e: f32, v_rest: f32, v_thresh: f32, v_reset: f32, tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(theta.len(), n, "eprop_lif_step: theta length mismatch");
    assert_eq!(eligibility.len(), n, "eprop_lif_step: eligibility length mismatch");
    assert_eq!(psi.len(), n, "eprop_lif_step: psi length mismatch");
    assert_eq!(Ii.len(), n, "eprop_lif_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_lif_step: spikes length mismatch");
    assert_eq!(pre_spikes.len(), n, "eprop_lif_step: pre_spikes length mismatch");
    unsafe { lif::eprop_lif_step(v.as_mut_ptr(), theta.as_mut_ptr(), eligibility.as_mut_ptr(), psi.as_mut_ptr(), Ii.as_ptr(), spikes.as_mut_ptr(), pre_spikes.as_ptr(), dt, tau_m, tau_e, v_rest, v_thresh, v_reset, tau_sg, n) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / saw
// ═══════════════════════════════════════════════════════════

pub fn saw_step(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], Ii: &[f32],
    dt: f32, trace_decay: f32, noise_amp: f32,
    spikes: &mut [i32],
    refractory: &mut [f32], tau_ref: f32,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "saw_step: u length mismatch");
    assert_eq!(trace.len(), n, "saw_step: trace length mismatch");
    assert_eq!(a.len(), n, "saw_step: a length mismatch");
    assert_eq!(b.len(), n, "saw_step: b length mismatch");
    assert_eq!(Ii.len(), n, "saw_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "saw_step: spikes length mismatch");
    assert_eq!(refractory.len(), n, "saw_step: refractory length mismatch");
    unsafe { saw::saw_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(), a.as_ptr(), b.as_ptr(), Ii.as_ptr(), dt, trace_decay, noise_amp, spikes.as_mut_ptr(), refractory.as_mut_ptr(), tau_ref, n) }
}

pub fn eprop_saw_step(
    v: &mut [f32], u: &mut [f32], eligibility: &mut [f32], psi: &mut [f32],
    Ii: &[f32], spikes: &mut [i32], pre_spikes: &[i32],
    dt: f32, tau_e: f32, a: f32, b: f32, tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_saw_step: u length mismatch");
    assert_eq!(eligibility.len(), n, "eprop_saw_step: eligibility length mismatch");
    assert_eq!(psi.len(), n, "eprop_saw_step: psi length mismatch");
    assert_eq!(Ii.len(), n, "eprop_saw_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_saw_step: spikes length mismatch");
    assert_eq!(pre_spikes.len(), n, "eprop_saw_step: pre_spikes length mismatch");
    unsafe { saw::eprop_saw_step(v.as_mut_ptr(), u.as_mut_ptr(), eligibility.as_mut_ptr(), psi.as_mut_ptr(), Ii.as_ptr(), spikes.as_mut_ptr(), pre_spikes.as_ptr(), dt, tau_e, a, b, tau_sg, n) }
}

pub fn psi_wave_step(
    v: &mut [f32], u: &mut [f32],
    h_real: &mut [f32], h_imag: &mut [f32],
    i_ext: &mut [f32], speed: &mut [f32],
    dt: f32, hbar: f32,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "psi_wave_step: u length mismatch");
    assert_eq!(h_real.len(), n, "psi_wave_step: h_real length mismatch");
    assert_eq!(h_imag.len(), n, "psi_wave_step: h_imag length mismatch");
    assert_eq!(i_ext.len(), n, "psi_wave_step: i_ext length mismatch");
    assert_eq!(speed.len(), n, "psi_wave_step: speed length mismatch");
    unsafe { saw::psi_wave_step(v.as_mut_ptr(), u.as_mut_ptr(), h_real.as_mut_ptr(), h_imag.as_mut_ptr(), i_ext.as_mut_ptr(), speed.as_mut_ptr(), dt, hbar, n) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / adex
// ═══════════════════════════════════════════════════════════

pub fn adex_step(v: &mut [f32], u: &mut [f32], Ii: &mut [f32], spikes: &mut [i32], params: &[f32], refractory: &mut [f32], tau_ref: f32) {
    let n = v.len();
    assert_eq!(u.len(), n, "adex_step: u length mismatch");
    assert_eq!(Ii.len(), n, "adex_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "adex_step: spikes length mismatch");
    assert_eq!(params.len(), n * 9, "adex_step: params length mismatch (expected {}*9={})", n, n * 9);
    assert_eq!(refractory.len(), n, "adex_step: refractory length mismatch");
    unsafe { adex::adex_step(v.as_mut_ptr(), u.as_mut_ptr(), Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(), refractory.as_mut_ptr(), tau_ref, n) }
}

pub fn eprop_adex_step_multi(
    v: &mut [f32], w: &mut [f32], psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    I_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32, tau_m: f32, tau_w: f32,
    v_rest: f32, v_thresh: f32, v_reset: f32, delta_t: f32,
    a: f32, b: f32,
    tau_e: f32, tau_sg: f32,
    spikes: &mut [i32],
) -> usize {
    let n = v.len();
    assert_eq!(w.len(), n, "eprop_adex_step_multi: w length mismatch");
    assert_eq!(psi.len(), n, "eprop_adex_step_multi: psi length mismatch");
    assert_eq!(e.len(), n_pre * n, "eprop_adex_step_multi: e length mismatch");
    assert_eq!(I_ext.len(), n, "eprop_adex_step_multi: I_ext length mismatch");
    assert_eq!(weights.len(), n_pre * n, "eprop_adex_step_multi: weights length mismatch");
    assert_eq!(spikes.len(), n, "eprop_adex_step_multi: spikes length mismatch");
    unsafe { adex::eprop_adex_step_multi(v.as_mut_ptr(), w.as_mut_ptr(), psi.as_mut_ptr(), e.as_mut_ptr(), z_pre.as_ptr(), I_ext.as_ptr(), weights.as_ptr(), n, n_pre, dt, tau_m, tau_w, v_rest, v_thresh, v_reset, delta_t, a, b, tau_e, tau_sg, spikes.as_mut_ptr()) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / stp
// ═══════════════════════════════════════════════════════════

pub fn stp_step(
    x: &mut [f32], u: &mut [f32],
    pre_spikes: &[i32],
    tau_d: &[f32], tau_f: &[f32], u0: &[f32], a: &[f32],
    I_out: &mut [f32], dt: f32,
) {
    let n = x.len();
    assert_eq!(u.len(), n, "stp_step: u length mismatch");
    assert_eq!(pre_spikes.len(), n, "stp_step: pre_spikes length mismatch");
    assert_eq!(tau_d.len(), n, "stp_step: tau_d length mismatch");
    assert_eq!(tau_f.len(), n, "stp_step: tau_f length mismatch");
    assert_eq!(u0.len(), n, "stp_step: u0 length mismatch");
    assert_eq!(a.len(), n, "stp_step: a length mismatch");
    assert_eq!(I_out.len(), n, "stp_step: I_out length mismatch");
    unsafe { stp::stp_step(x.as_mut_ptr(), u.as_mut_ptr(), pre_spikes.as_ptr(), tau_d.as_ptr(), tau_f.as_ptr(), u0.as_ptr(), a.as_ptr(), I_out.as_mut_ptr(), dt, n) }
}

pub fn stdp_update(weights: &mut [f32], traces_pre: &[f32], traces_post: &[f32], lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32) {
    let n = weights.len();
    assert_eq!(traces_pre.len(), n, "stdp_update: traces_pre length mismatch");
    assert_eq!(traces_post.len(), n, "stdp_update: traces_post length mismatch");
    unsafe { stp::stdp_update(weights.as_mut_ptr(), traces_pre.as_ptr(), traces_post.as_ptr(), lr_ltd, lr_ltp, w_min, w_max, n) }
}

pub fn stdp_homeostatic(weights: &mut [f32], traces_pre: &[f32], traces_post: &[f32], lr_ltd: f32, lr_ltp: f32, target_sum: f32, w_max: f32) {
    let n = weights.len();
    assert_eq!(traces_pre.len(), n, "stdp_homeostatic: traces_pre length mismatch");
    assert_eq!(traces_post.len(), n, "stdp_homeostatic: traces_post length mismatch");
    unsafe { stp::stdp_homeostatic(weights.as_mut_ptr(), traces_pre.as_ptr(), traces_post.as_ptr(), lr_ltd, lr_ltp, target_sum, w_max, n) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / delay
// ═══════════════════════════════════════════════════════════

pub fn delay_buffer_circular(output: &mut [f32], buffer: &[f32], pos: usize) {
    let n = output.len();
    assert!(buffer.len() >= n, "delay_buffer_circular: buffer length {} < n {}", buffer.len(), n);
    unsafe { delay::delay_buffer_circular(output.as_mut_ptr(), buffer.as_ptr(), n, pos) }
}

pub fn delay_buffer_multi_tap(output: &mut [f32], buffer: &[f32], delays: &[i32]) {
    let n = buffer.len();
    let n_taps = output.len();
    assert_eq!(delays.len(), n_taps, "delay_buffer_multi_tap: delays length mismatch");
    unsafe { delay::delay_buffer_multi_tap(output.as_mut_ptr(), buffer.as_ptr(), delays.as_ptr(), n, n_taps) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / dopamine
// ═══════════════════════════════════════════════════════════

pub fn dopamine_traces_step(
    trace_pre: &mut [f32], trace_post: &mut [f32], eligibility: &mut [f32],
    pre_spikes: &[i32], post_spikes: &[i32],
    dt: f32, tau_pre: f32, tau_post: f32, tau_e: f32,
    a_ltp: f32, a_ltd: f32,
) {
    let n = trace_pre.len();
    assert_eq!(trace_post.len(), n, "dopamine_traces_step: trace_post length mismatch");
    assert_eq!(eligibility.len(), n, "dopamine_traces_step: eligibility length mismatch");
    assert_eq!(pre_spikes.len(), n, "dopamine_traces_step: pre_spikes length mismatch");
    assert_eq!(post_spikes.len(), n, "dopamine_traces_step: post_spikes length mismatch");
    unsafe { dopamine::dopamine_traces_step(trace_pre.as_mut_ptr(), trace_post.as_mut_ptr(), eligibility.as_mut_ptr(), pre_spikes.as_ptr(), post_spikes.as_ptr(), dt, tau_pre, tau_post, tau_e, a_ltp, a_ltd, n) }
}

pub fn dopamine_weight_update(weights: &mut [f32], eligibility: &[f32], dopamine: f32, lr: f32, w_min: f32, w_max: f32) {
    let n = weights.len();
    assert_eq!(eligibility.len(), n, "dopamine_weight_update: eligibility length mismatch");
    unsafe { dopamine::dopamine_weight_update(weights.as_mut_ptr(), eligibility.as_ptr(), dopamine, lr, w_min, w_max, n) }
}

// ═══════════════════════════════════════════════════════════
//  neuron / fused
// ═══════════════════════════════════════════════════════════

pub fn fused_hidden_izhikevich_step(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32], spikes: &mut [i32], refractory: &mut [f32],
    a: &[f32], b: &[f32], c: &[f32], d: &[f32],
    weights: &[f32],
    input_spikes: &[i32],
    dt: f32, trace_decay: f32, tau_ref: f32,
    ni: usize,
) {
    let nh = v.len();
    assert_eq!(u.len(), nh, "fused_hidden_izhikevich_step: u length mismatch");
    assert_eq!(trace.len(), nh, "fused_hidden_izhikevich_step: trace length mismatch");
    assert_eq!(spikes.len(), nh, "fused_hidden_izhikevich_step: spikes length mismatch");
    assert_eq!(refractory.len(), nh, "fused_hidden_izhikevich_step: refractory length mismatch");
    assert_eq!(a.len(), nh, "fused_hidden_izhikevich_step: a length mismatch");
    assert_eq!(b.len(), nh, "fused_hidden_izhikevich_step: b length mismatch");
    assert_eq!(c.len(), nh, "fused_hidden_izhikevich_step: c length mismatch");
    assert_eq!(d.len(), nh, "fused_hidden_izhikevich_step: d length mismatch");
    assert_eq!(weights.len(), ni * nh, "fused_hidden_izhikevich_step: weights length mismatch");
    assert!(input_spikes.len() >= ni, "fused_hidden_izhikevich_step: input_spikes length {} < ni {}", input_spikes.len(), ni);
    assert!(nh % 8 == 0, "fused_hidden_izhikevich_step: nh must be multiple of 8, got {}", nh);
    unsafe { fused::fused_hidden_izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(), spikes.as_mut_ptr(), refractory.as_mut_ptr(), a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr(), weights.as_ptr(), input_spikes.as_ptr(), dt, trace_decay, tau_ref, ni, nh) }
}

pub fn fused_output_lif_step(
    v: &mut [f32], spikes: &mut [i32], refractory: &mut [f32],
    tau_m: &[f32], v_rest: &[f32], v_th: &[f32], v_reset: &[f32],
    weights: &[f32],
    hidden_spikes: &[i32],
    dt: f32, tau_ref: f32,
    nh: usize,
) {
    let na = v.len();
    assert_eq!(spikes.len(), na, "fused_output_lif_step: spikes length mismatch");
    assert_eq!(refractory.len(), na, "fused_output_lif_step: refractory length mismatch");
    assert_eq!(tau_m.len(), na, "fused_output_lif_step: tau_m length mismatch");
    assert_eq!(v_rest.len(), na, "fused_output_lif_step: v_rest length mismatch");
    assert_eq!(v_th.len(), na, "fused_output_lif_step: v_th length mismatch");
    assert_eq!(v_reset.len(), na, "fused_output_lif_step: v_reset length mismatch");
    assert_eq!(weights.len(), nh * na, "fused_output_lif_step: weights length mismatch");
    assert!(hidden_spikes.len() >= nh, "fused_output_lif_step: hidden_spikes length {} < nh {}", hidden_spikes.len(), nh);
    assert!(na % 8 == 0, "fused_output_lif_step: na must be multiple of 8, got {}", na);
    unsafe { fused::fused_output_lif_step(v.as_mut_ptr(), spikes.as_mut_ptr(), refractory.as_mut_ptr(), tau_m.as_ptr(), v_rest.as_ptr(), v_th.as_ptr(), v_reset.as_ptr(), weights.as_ptr(), hidden_spikes.as_ptr(), dt, tau_ref, nh, na) }
}

pub fn poisson_fused_hidden_step(
    v: &mut [f32], u: &mut [f32], trace: &mut [f32], spikes: &mut [i32], refractory: &mut [f32],
    a: &[f32], b: &[f32], c: &[f32], d: &[f32],
    weights: &[f32],
    rng_states: &mut [u32],
    rates: &[f32],
    input_spikes_out: &mut [i32],
    dt: f32, trace_decay: f32, tau_ref: f32,
    ni: usize,
) {
    let nh = v.len();
    assert_eq!(u.len(), nh, "poisson_fused_hidden_step: u length mismatch");
    assert_eq!(trace.len(), nh, "poisson_fused_hidden_step: trace length mismatch");
    assert_eq!(spikes.len(), nh, "poisson_fused_hidden_step: spikes length mismatch");
    assert_eq!(refractory.len(), nh, "poisson_fused_hidden_step: refractory length mismatch");
    assert_eq!(a.len(), nh, "poisson_fused_hidden_step: a length mismatch");
    assert_eq!(b.len(), nh, "poisson_fused_hidden_step: b length mismatch");
    assert_eq!(c.len(), nh, "poisson_fused_hidden_step: c length mismatch");
    assert_eq!(d.len(), nh, "poisson_fused_hidden_step: d length mismatch");
    assert_eq!(weights.len(), ni * nh, "poisson_fused_hidden_step: weights length mismatch");
    assert!(rng_states.len() >= ni, "poisson_fused_hidden_step: rng_states length {} < ni {}", rng_states.len(), ni);
    assert!(rates.len() >= ni, "poisson_fused_hidden_step: rates length {} < ni {}", rates.len(), ni);
    assert!(input_spikes_out.len() >= ni, "poisson_fused_hidden_step: input_spikes_out length {} < ni {}", input_spikes_out.len(), ni);
    assert!(nh % 8 == 0, "poisson_fused_hidden_step: nh must be multiple of 8, got {}", nh);
    unsafe { fused::poisson_fused_hidden_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(), spikes.as_mut_ptr(), refractory.as_mut_ptr(), a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr(), weights.as_ptr(), rng_states.as_mut_ptr(), rates.as_ptr(), input_spikes_out.as_mut_ptr(), dt, trace_decay, tau_ref, ni, nh) }
}
