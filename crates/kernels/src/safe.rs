//! Safe wrappers over all SIMD kernel functions.
//!
//! Each function validates slice lengths and alignment requirements before
//! delegating to the corresponding `pub(crate) unsafe` kernel implementation.
//! All SIMD alignment invariants (`n % 8 == 0` not required — scalar tail handled)
//! are satisfied by the calling convention.

use super::neuron::{
    adex, bcm, delay, dendrite, dopamine, fhn, fused, glif, haap, hh, izhikevich, lif, morris_lecar, nmda, saw, stp, tsodyks_markram, wc,
};
use super::{decode, measure, phase, signal};

// ═══════════════════════════════════════════════════════════
//  Pillars bridge (Фаза 2 консолидации)
// ═══════════════════════════════════════════════════════════
//
// `safe::*_step` — теперь ТОНКИЕ ОБЁРТКИ над `pillars::run_neuron`.
// Pillars — единственный источник SIMD-нейронов (см. docs/DUPLICATION_MAP.md).
// Legacy SoA-лейауты переупаковываются на границе вызова; thread-local
// scratch исключает аллокации в hot path.

use pillars::core::runner::run_neuron;
use pillars::zoo::Model;
use pillars::zoo::neurons::{AdExModel, HHModel, IzhikevichModel, LIFModel, SAWModel};
use std::cell::RefCell;

// Thread-local scratch для переупаковки параметров и вспомогательных буферов.
// Растёт по требованию — steady-state вызовы не аллоцируют.
thread_local! {
    static PILLARS_PARAMS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static PILLARS_AUX: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static PILLARS_SPIKES: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

/// Выполнить `f` с параметр-буфером `n_cols × n` (растёт по требованию).
/// Closure-паттерн: `RefMut` не покидает `.with()` — нет lifetime-проблем.
fn with_params_scratch<T>(n_cols: usize, n: usize, f: impl FnOnce(&mut [f32]) -> T) -> T {
    PILLARS_PARAMS.with(|c| {
        let mut v = c.borrow_mut();
        let need = n_cols * n;
        if v.len() < need {
            v.resize(need, 0.0);
        }
        f(&mut v[..need])
    })
}

/// Выполнить `f` со вспомогательным f32-буфером (растёт по требованию).
fn with_aux_scratch<T>(need: usize, f: impl FnOnce(&mut [f32]) -> T) -> T {
    PILLARS_AUX.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < need {
            v.resize(need, 0.0);
        }
        f(&mut v[..need])
    })
}

/// Выполнить `f` со вспомогательным i32-буфером (растёт по требованию).
fn with_spikes_scratch<T>(need: usize, f: impl FnOnce(&mut [i32]) -> T) -> T {
    PILLARS_SPIKES.with(|c| {
        let mut v = c.borrow_mut();
        if v.len() < need {
            v.resize(need, 0);
        }
        f(&mut v[..need])
    })
}

// ═══════════════════════════════════════════════════════════
//  signal
// ═══════════════════════════════════════════════════════════

/// Clamp all values in `data` to `[min, max]` — single-sourced from pillars.
///
/// Канон: `pillars::kernels::signal::safety_clamp` (Фаза 1 консолидации).
pub use pillars::kernels::signal::safety_clamp;

/// Stochastic spike encoder: compare `inputs + noise` against `threshold`.
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
    unsafe {
        signal::stochastic_encoder(
            target_i.as_mut_ptr(),
            inputs.as_ptr(),
            noise.as_ptr(),
            threshold,
            spike_val,
            n,
        )
    }
}

/// Linear readout: `output = W·states + bias`. Uses SIMD `f32x8` for inner dot product.
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
    unsafe {
        signal::linear_decoder(
            outputs.as_mut_ptr(),
            states.as_ptr(),
            weights.as_ptr(),
            biases.as_ptr(),
            n_in,
            n_out,
        )
    }
}

/// Thalamic gate: `target = source × sigmoid(gain·(gate − threshold))`.
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
    unsafe {
        signal::thalamic_gate(
            target_i.as_mut_ptr(),
            source_v.as_ptr(),
            gate_v.as_ptr(),
            threshold,
            gain,
            n,
        )
    }
}

/// Elementwise sigmoid: `output = sigmoid(gain·(input − threshold))`.
pub fn sigmoid_gate(output: &mut [f32], input: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "sigmoid_gate: input length mismatch");
    unsafe { signal::sigmoid_gate(output.as_mut_ptr(), input.as_ptr(), gain, threshold, n) }
}

/// Phase gate: `output = input × sigmoid(gain·(r − threshold))`.
pub fn phase_gate_apply(output: &mut [f32], input: &[f32], r: &[f32], gain: f32, threshold: f32) {
    let n = output.len();
    assert_eq!(input.len(), n, "phase_gate_apply: input length mismatch");
    assert_eq!(r.len(), n, "phase_gate_apply: r length mismatch");
    unsafe {
        signal::phase_gate_apply(
            output.as_mut_ptr(),
            input.as_ptr(),
            r.as_ptr(),
            gain,
            threshold,
            n,
        )
    }
}

/// T3 drive encoder: `drives = gain × inputs`.
pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) {
    let n = drives.len();
    assert_eq!(inputs.len(), n, "t3_drive_encoder: inputs length mismatch");
    unsafe { signal::t3_drive_encoder(drives.as_mut_ptr(), inputs.as_ptr(), gain, n) }
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
    unsafe {
        signal::t3_tact_generator(
            pulses.as_mut_ptr(),
            ph1.as_ptr(),
            ph2.as_ptr(),
            ph3.as_ptr(),
            threshold,
            n,
        )
    }
}

/// Echo state tap: read one delay tap from oscillator ring buffer.
pub fn echo_tap(output: &mut [f32], buffer: &[f32], delay_len: usize, delay_idx: usize) {
    let n_osc = output.len();
    assert_eq!(
        buffer.len(),
        n_osc * delay_len,
        "echo_tap: buffer length mismatch"
    );
    unsafe {
        signal::echo_tap(
            output.as_mut_ptr(),
            buffer.as_ptr(),
            n_osc,
            delay_len,
            delay_idx,
        )
    }
}

/// T3 inverse Kuramoto: estimate drive from amplitude changes over time.
///
/// Inverts Stuart-Landau dynamics:
///   r' = r + dt · ((α − r²)·r + gain·drive − decay·r)
/// → drive ≈ gain · (r − r_prev)/dt − drive · decay
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
    let n = drive_out.len();
    assert_eq!(r.len(), n, "t3_inverse_kuramoto: r length mismatch");
    assert_eq!(
        r_prev.len(),
        n,
        "t3_inverse_kuramoto: r_prev length mismatch"
    );
    unsafe {
        signal::t3_inverse_kuramoto(
            drive_out.as_mut_ptr(),
            r.as_ptr(),
            r_prev.as_ptr(),
            alpha,
            gain,
            eps,
            decay,
            dt,
            n,
        )
    }
}

/// Hebbian plasticity on CSR weights using slow states.
///
/// `dW = η · m_pre · m_post − α · W · dt`, clamped to [-1, 1].
///
/// `values` — CSR synaptic weights (mutated in-place).
/// `m` — slow state per neuron (from [`izhikevich_dual_step`]).
pub fn hebbian_csr_step(
    values: &mut [f32],
    col_indices: &[i32],
    row_ptr: &[i32],
    m: &[f32],
    n_neurons: usize,
    lr: f32,
    decay: f32,
    dt: f32,
) {
    let n_synapses = values.len();
    assert_eq!(
        col_indices.len(),
        n_synapses,
        "hebbian_csr_step: col_indices length mismatch"
    );
    assert!(
        row_ptr.len() > n_neurons,
        "hebbian_csr_step: row_ptr too short"
    );
    assert!(m.len() >= n_neurons, "hebbian_csr_step: m too short");
    unsafe {
        izhikevich::hebbian_csr_step(
            values.as_mut_ptr(),
            col_indices.as_ptr(),
            row_ptr.as_ptr(),
            m.as_ptr(),
            n_neurons,
            n_synapses,
            lr,
            decay,
            dt,
        )
    }
}

/// Elementwise ReLU: `x = max(0, x)` using SIMD `f32x8`.
pub fn relu_simd(data: &mut [f32]) {
    let n = data.len();
    unsafe { signal::relu_simd(data.as_mut_ptr(), n) }
}

/// Elementwise add: `a\[i\] += b\[i\]` using SIMD `f32x8`.
pub fn add_simd(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    assert_eq!(b.len(), n, "add_simd: b length mismatch");
    unsafe { signal::add_simd(a.as_mut_ptr(), b.as_ptr(), n) }
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
    unsafe {
        signal::sgd_step_simd(
            velocity.as_mut_ptr(),
            weight.as_mut_ptr(),
            grad.as_ptr(),
            lr,
            momentum,
            n,
        )
    }
}

/// Adam optimizer step: `m = β₁·m + (1−β₁)·g; v = β₂·v + (1−β₂)·g²; w -= lr·m̂/(√v̂ + ε)`.
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
    unsafe {
        signal::adam_step_simd(
            weight.as_mut_ptr(),
            m.as_mut_ptr(),
            v.as_mut_ptr(),
            grad.as_ptr(),
            lr,
            beta1,
            beta2,
            eps,
            beta1_t_corr,
            beta2_t_corr,
            n,
        )
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
    unsafe { signal::hebbian_update(weights.as_mut_ptr(), cos_diff.as_ptr(), lr, decay, dt, n) }
}

// ═══════════════════════════════════════════════════════════
//  decode
// ═══════════════════════════════════════════════════════════

/// Complex decoder: `output = W·[amplitudes ⊙ cos_buf] + bias`.
pub fn t3_complex_decoder(
    outputs: &mut [f32],
    amplitudes: &[f32],
    cos_buf: &[f32],
    weights: &[f32],
    biases: &[f32],
) {
    let n_out = outputs.len();
    let n_in = amplitudes.len();
    assert_eq!(
        cos_buf.len(),
        n_in,
        "t3_complex_decoder: cos_buf length mismatch"
    );
    assert_eq!(
        biases.len(),
        n_out,
        "t3_complex_decoder: biases length mismatch"
    );
    assert_eq!(
        weights.len(),
        n_in * n_out,
        "t3_complex_decoder: weights length mismatch"
    );
    unsafe {
        decode::t3_complex_decoder(
            outputs.as_mut_ptr(),
            amplitudes.as_ptr(),
            cos_buf.as_ptr(),
            weights.as_ptr(),
            biases.as_ptr(),
            n_in,
            n_out,
        )
    }
}

/// Compute pairwise PST features: `sin(θ₁−θ₂)` and `cos(θ₁−θ₂)`.
pub fn t3_pst_features(out_sin: &mut [f32], out_cos: &mut [f32], ph1: &[f32], ph2: &[f32]) {
    let n1 = ph1.len();
    let n2 = ph2.len();
    assert_eq!(
        out_sin.len(),
        n1 * n2,
        "t3_pst_features: out_sin length mismatch"
    );
    assert_eq!(
        out_cos.len(),
        n1 * n2,
        "t3_pst_features: out_cos length mismatch"
    );
    unsafe {
        decode::t3_pst_features(
            out_sin.as_mut_ptr(),
            out_cos.as_mut_ptr(),
            ph1.as_ptr(),
            ph2.as_ptr(),
            n1,
            n2,
        )
    }
}

/// LMS training step: `w += lr·error·states; bias += lr·error`.
pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32], error: f32, lr: f32) {
    let n = weights.len();
    assert_eq!(states.len(), n, "t3_lms_train: states length mismatch");
    assert!(
        !bias.is_empty(),
        "t3_lms_train: bias must have at least 1 element"
    );
    unsafe {
        decode::t3_lms_train(
            weights.as_mut_ptr(),
            bias.as_mut_ptr(),
            states.as_ptr(),
            error,
            lr,
            n,
        )
    }
}

/// NLMS training step: `w += lr·error·states / (‖states‖² + ε)`.
pub fn nlms_train(
    weights: &mut [f32],
    bias: &mut [f32],
    states: &[f32],
    error: f32,
    lr: f32,
    eps: f32,
) {
    let n = weights.len();
    assert_eq!(states.len(), n, "nlms_train: states length mismatch");
    assert!(
        !bias.is_empty(),
        "nlms_train: bias must have at least 1 element"
    );
    unsafe {
        decode::nlms_train(
            weights.as_mut_ptr(),
            bias.as_mut_ptr(),
            states.as_ptr(),
            error,
            lr,
            eps,
            n,
        )
    }
}

/// Kalman filter update: `w += K·error; cov -= K·states·cov` where `K = cov·states / s`.
pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32], error: f32, s: f32) {
    let n = weights.len();
    assert_eq!(cov.len(), n, "kalman_update: cov length mismatch");
    assert_eq!(states.len(), n, "kalman_update: states length mismatch");
    unsafe {
        decode::kalman_update(
            weights.as_mut_ptr(),
            cov.as_mut_ptr(),
            states.as_ptr(),
            error,
            s,
            n,
        )
    }
}

/// Adam training step: update decoder weights using Adam optimizer.
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
    unsafe {
        decode::adam_train(
            weights.as_mut_ptr(),
            m.as_mut_ptr(),
            v.as_mut_ptr(),
            states.as_ptr(),
            error,
            lr,
            beta1,
            beta2,
            bc1,
            _bc2,
            eps,
            n,
        )
    }
}

/// PST decoder: weighted circular buffer readout.
pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32], l: usize, pos: usize) {
    let feat_dim = output.len();
    assert_eq!(
        buffer.len(),
        l * feat_dim,
        "pst_decode: buffer length mismatch"
    );
    assert_eq!(
        weights.len(),
        l * feat_dim,
        "pst_decode: weights length mismatch"
    );
    unsafe {
        decode::pst_decode(
            output.as_mut_ptr(),
            buffer.as_ptr(),
            weights.as_ptr(),
            feat_dim,
            l,
            pos,
        )
    }
}

/// Ring synthesis: `output\[i\] = weights\[i\] × amplitudes\[i\] × trig_buf\[i\]`.
pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32], weights: &[f32]) {
    let n = output.len();
    assert_eq!(
        amplitudes.len(),
        n,
        "ring_synthesis: amplitudes length mismatch"
    );
    assert_eq!(
        trig_buf.len(),
        n,
        "ring_synthesis: trig_buf length mismatch"
    );
    assert_eq!(weights.len(), n, "ring_synthesis: weights length mismatch");
    unsafe {
        decode::ring_synthesis(
            output.as_mut_ptr(),
            amplitudes.as_ptr(),
            trig_buf.as_ptr(),
            weights.as_ptr(),
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  measure
// ═══════════════════════════════════════════════════════════

/// Compute pairwise interference matrix: `out[i,j] = cos(θᵢ − θⱼ)`.
pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(
        output.len(),
        na * nb,
        "interference: output length mismatch"
    );
    unsafe {
        measure::interference(
            output.as_mut_ptr(),
            phases_a.as_ptr(),
            phases_b.as_ptr(),
            na,
            nb,
        )
    }
}

/// Mean interference: average of all pairwise `cos(θᵢ − θⱼ)`.
pub fn interference_mean(output: &mut f32, phases_a: &[f32], phases_b: &[f32]) {
    let n = phases_a.len();
    assert_eq!(
        phases_b.len(),
        n,
        "interference_mean: phases_b length mismatch"
    );
    unsafe {
        measure::interference_mean(output as *mut f32, phases_a.as_ptr(), phases_b.as_ptr(), n)
    }
}

/// Phase coherence (Kuramoto order parameter): amplitude-weighted mean vector length.
pub fn phase_coherence(output: &mut f32, sin_buf: &[f32], cos_buf: &[f32], amplitudes: &[f32]) {
    let n = sin_buf.len();
    assert_eq!(cos_buf.len(), n, "phase_coherence: cos_buf length mismatch");
    assert_eq!(
        amplitudes.len(),
        n,
        "phase_coherence: amplitudes length mismatch"
    );
    unsafe {
        measure::phase_coherence(
            output as *mut f32,
            sin_buf.as_ptr(),
            cos_buf.as_ptr(),
            amplitudes.as_ptr(),
            n,
        )
    }
}

/// Band energy: sum of `amplitudes²` within each frequency band.
pub fn band_energy(
    band_energies: &mut [f32],
    amplitudes: &[f32],
    omegas: &[f32],
    band_edges: &[f32],
) {
    let n_bands = band_energies.len();
    let n = amplitudes.len();
    assert_eq!(omegas.len(), n, "band_energy: omegas length mismatch");
    assert_eq!(
        band_edges.len(),
        n_bands + 1,
        "band_energy: band_edges length mismatch (expected {})",
        n_bands + 1
    );
    unsafe {
        measure::band_energy(
            band_energies.as_mut_ptr(),
            amplitudes.as_ptr(),
            omegas.as_ptr(),
            band_edges.as_ptr(),
            n_bands,
            n,
        )
    }
}

/// Winding number matrix: `out[i,j] = sin(θᵢ − θⱼ)`.
pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(
        output.len(),
        na * nb,
        "winding_matrix: output length mismatch"
    );
    unsafe {
        measure::winding_matrix(
            output.as_mut_ptr(),
            phases_a.as_ptr(),
            phases_b.as_ptr(),
            na,
            nb,
        )
    }
}

/// Lazy sin/cos: only recompute when phase changed by more than `eps`.
pub fn lazy_sincos_update(
    sin_out: &mut [f32],
    cos_out: &mut [f32],
    phases: &[f32],
    phases_prev: &mut [f32],
    eps: f32,
) {
    let n = sin_out.len();
    assert_eq!(
        cos_out.len(),
        n,
        "lazy_sincos_update: cos_out length mismatch"
    );
    assert_eq!(
        phases.len(),
        n,
        "lazy_sincos_update: phases length mismatch"
    );
    assert_eq!(
        phases_prev.len(),
        n,
        "lazy_sincos_update: phases_prev length mismatch"
    );
    unsafe {
        measure::lazy_sincos_update(
            sin_out.as_mut_ptr(),
            cos_out.as_mut_ptr(),
            phases.as_ptr(),
            phases_prev.as_mut_ptr(),
            eps,
            n,
        )
    }
}

/// Spike entropy: H = -Σ p_i · log₂(p_i) where p_i = spike_counts[i] / total_steps.
///
/// Returns (H, H_penalty) where H_penalty = |H - H_target|.
pub fn spike_entropy(spike_counts: &[f32], total_steps: f32, h_target: f32) -> (f32, f32) {
    let n = spike_counts.len();
    let (mut h, mut hp) = (0.0f32, 0.0f32);
    unsafe {
        measure::spike_entropy(
            &mut h,
            &mut hp,
            spike_counts.as_ptr(),
            n,
            total_steps,
            h_target,
        );
    }
    (h, hp)
}

/// Batch coherence sweep: pairwise coherence across area groups.
pub fn batch_coherence_sweep(
    output: &mut [f32],
    areas: &[f32],
    areas_sin: &[f32],
    areas_cos: &[f32],
    n_elem: usize,
) {
    let total = areas.len();
    let n_areas = total / n_elem;
    assert_eq!(
        total,
        n_areas * n_elem,
        "batch_coherence_sweep: areas length not divisible by n_elem"
    );
    assert_eq!(
        output.len(),
        n_areas * n_areas,
        "batch_coherence_sweep: output must be n_areas², got {} expected {}",
        output.len(),
        n_areas * n_areas
    );
    assert_eq!(
        areas_sin.len(),
        total,
        "batch_coherence_sweep: areas_sin length mismatch"
    );
    assert_eq!(
        areas_cos.len(),
        total,
        "batch_coherence_sweep: areas_cos length mismatch"
    );
    unsafe {
        measure::batch_coherence_sweep(
            output.as_mut_ptr(),
            areas.as_ptr(),
            areas_sin.as_ptr(),
            areas_cos.as_ptr(),
            n_areas,
            n_elem,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  phase
// ═══════════════════════════════════════════════════════════

/// Wrap phases to [-π, π) and compute sin/cos in one SIMD pass.
pub fn kuramoto_prepare(phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
    let n = phases.len();
    assert_eq!(
        cos_out.len(),
        n,
        "kuramoto_prepare: cos_out length mismatch"
    );
    assert_eq!(
        sin_out.len(),
        n,
        "kuramoto_prepare: sin_out length mismatch"
    );
    unsafe {
        phase::kuramoto_prepare(
            phases.as_mut_ptr(),
            cos_out.as_mut_ptr(),
            sin_out.as_mut_ptr(),
            n,
        )
    }
}

/// Euler-step Kuramoto phases: `φ += dt·(ω + coupling)` with phase wrapping.
pub fn kuramoto_update(phases: &mut [f32], omegas: &[f32], coupling: &[f32], dt: f32) {
    let n = phases.len();
    assert_eq!(omegas.len(), n, "kuramoto_update: omegas length mismatch");
    assert_eq!(
        coupling.len(),
        n,
        "kuramoto_update: coupling length mismatch"
    );
    unsafe {
        phase::kuramoto_update(
            phases.as_mut_ptr(),
            omegas.as_ptr(),
            coupling.as_ptr(),
            dt,
            n,
        )
    }
}

/// Viscous Kuramoto update: `φ += dt·(ω + coupling + η·(mean(φ) − φ))`
///
/// Добавляет вязкое демпфирование к стандартному Kuramoto:
///   dφ_i/dt = ω_i + coupling_i + η·(⟨φ⟩ − φ_i)
///
/// Вязкий член η·(⟨φ⟩ − φ_i) стягивает фазы к среднему, подавляя
/// хаотические фазовые переключения. При η → ∞ все фазы синхронизуются.
/// При η = 0 эквивалентно `kuramoto_update`.
///
/// Reference: Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence"
pub fn kuramoto_update_viscous(
    phases: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    eta: f32,
    dt: f32,
) {
    let n = phases.len();
    assert_eq!(
        omegas.len(),
        n,
        "kuramoto_update_viscous: omegas length mismatch"
    );
    assert_eq!(
        coupling.len(),
        n,
        "kuramoto_update_viscous: coupling length mismatch"
    );
    unsafe {
        phase::kuramoto_update_viscous(
            phases.as_mut_ptr(),
            omegas.as_ptr(),
            coupling.as_ptr(),
            eta,
            dt,
            n,
        )
    }
}

/// Combined Kuramoto phase + Stuart-Landau amplitude step with external drive.
pub fn t3_update(
    phases: &mut [f32],
    amplitudes: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    drives: &[f32],
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
) {
    let n = phases.len();
    assert_eq!(amplitudes.len(), n, "t3_update: amplitudes length mismatch");
    assert_eq!(omegas.len(), n, "t3_update: omegas length mismatch");
    assert_eq!(coupling.len(), n, "t3_update: coupling length mismatch");
    assert_eq!(drives.len(), n, "t3_update: drives length mismatch");
    unsafe {
        phase::t3_update(
            phases.as_mut_ptr(),
            amplitudes.as_mut_ptr(),
            omegas.as_ptr(),
            coupling.as_ptr(),
            drives.as_ptr(),
            dt,
            decay,
            alpha,
            gain,
            n,
        )
    }
}


/// Add dendritic current `I_dend` to input current `Ii` in-place.
/// No-op if `I_dend` is `None`.
fn add_dendritic_current(Ii: &mut [f32], I_dend: Option<&[f32]>) {
    if let Some(dend) = I_dend {
        assert_eq!(
            Ii.len(),
            dend.len(),
            "add_dendritic_current: I_dend length mismatch"
        );
        for (ii_val, &d) in Ii.iter_mut().zip(dend.iter()) {
            *ii_val += d;
        }
    }
}

/// Merge dendritic current into an immutable Ii slice, returning either the
/// borrowed original (when `I_dend` is `None`) or an owned copy with I_dend added.
///
/// This avoids heap allocation in the common case (no dendrites).
fn with_dendritic_current<'a>(
    ii: &'a [f32],
    I_dend: Option<&'a [f32]>,
    label: &str,
) -> std::borrow::Cow<'a, [f32]> {
    if let Some(dend) = I_dend {
        assert_eq!(ii.len(), dend.len(), "{label}: I_dend length mismatch");
        let mut tmp = ii.to_vec();
        for (v, &d) in tmp.iter_mut().zip(dend.iter()) {
            *v += d;
        }
        std::borrow::Cow::Owned(tmp)
    } else {
        std::borrow::Cow::Borrowed(ii)
    }
}

/// Izhikevich neuron model — simple, efficient, biologically plausible spiking dynamics.
///
/// Reproduces 20+ cortical neuron firing patterns (RS, FS, IB, CH, etc.) with just
/// 4 parameters. Combines quadratic membrane dynamics with linear recovery variable.
///
/// # Equations
/// ```text
/// dv/dt = 0.04·v² + 5·v + 140 − u + I
/// du/dt = a·(b·v − u)
/// 
/// if v ≥ 30 mV:
///     spike
///     v ← c
///     u ← u + d
///     refractory ← τ_ref
/// ```
///
/// # Parameters
/// - `v`: membrane potential (mV) — fast variable with quadratic nullcline
/// - `u`: recovery variable (dimensionless) — slow variable, linear nullcline
/// - `a`: recovery time scale (0.02 for RS, 0.1 for FS)
/// - `b`: recovery sensitivity (0.2 typical)
/// - `c`: reset voltage (mV) — typically -65 for RS, -55 for IB
/// - `d`: recovery jump on spike (8 for RS, 2 for FS)
/// - `τ_ref`: refractory period (ms)
/// - `trace_decay`: STDP trace decay factor (0.95 typical)
///
/// # Pre-configured Patterns (Izhikevich 2003)
/// | Pattern | a    | b   | c    | d   | Description |
/// |---------|------|-----|------|-----|-------------|
/// | RS      | 0.02 | 0.2 | -65  | 8   | Regular Spiking (excitatory) |
/// | FS      | 0.1  | 0.2 | -65  | 2   | Fast Spiking (inhibitory) |
/// | IB      | 0.02 | 0.2 | -55  | 4   | Intrinsically Bursting |
/// | CH      | 0.02 | 0.2 | -50  | 2   | Chattering |
/// | LTS     | 0.02 | 0.25| -65  | 2   | Low-Threshold Spiking |
///
/// # Input Current
/// - `ii`: synaptic + external input (mA/cm²)
/// - **Automatic clamping**: `ii ∈ [-100, 100]` (prevents saturation cascade)
/// - `I_dend`: optional dendritic current (same length as `ii`)
///
/// # Refractory Period
/// During refractory (`refractory > 0`):
/// - `v` and `u` are **frozen** (no integration)
/// - `trace` is **frozen** (STDP eligibility)
/// - Refractory counter decrements by `dt`
/// - No spike detection
///
/// # STDP Trace
/// Legacy side-output for spike-timing dependent plasticity:
/// ```text
/// if spike:
///     trace ← trace · decay + (1 - trace)
/// else:
///     trace ← trace · (1 - decay)
/// ```
/// Frozen during refractory period.
///
/// # Integration Method
/// Forward Euler with 2 half-steps for numerical stability:
/// ```rust
/// // Half-step 1
/// v_mid = v + 0.5·dt·(0.04·v² + 5·v + 140 − u + I)
/// // Half-step 2
/// v_new = v + dt·(0.04·v_mid² + 5·v_mid + 140 − u + I)
/// ```
/// Recovery variable `u` uses single full step (linear, numerically stable).
///
/// # Performance
/// - **SIMD**: 8 neurons/instruction (AVX2)
/// - **Throughput**: ~3.8 ms/step for 1M neurons
/// - **Memory**: Column-major SoA (cache-friendly)
///
/// # Example
/// ```rust,ignore
/// use taas_checkmate::kernels::izhikevich_step;
///
/// let n = 1000;
/// let mut v = vec![-65.0; n];
/// let mut u = vec![-13.0; n];
/// let mut trace = vec![0.0; n];
/// let a = vec![0.02; n];  // RS pattern
/// let b = vec![0.2; n];
/// let c = vec![-65.0; n];
/// let d = vec![8.0; n];
/// let mut ii = vec![10.0; n];
/// let mut spikes = vec![0i32; n];
/// let mut refractory = vec![0.0; n];
///
/// for _ in 0..1000 {
///     izhikevich_step(
///         &mut v, &mut u, &mut trace,
///         &a, &b, &mut ii,
///         0.5,  // dt
///         0.95, // trace_decay
///         &mut spikes,
///         &c, &d,
///         &mut refractory,
///         2.0,  // tau_ref
///         None, // I_dend
///     );
/// }
/// ```
///
/// # Reference
/// Izhikevich, E. M. (2003). "Simple model of spiking neurons."
/// IEEE Transactions on Neural Networks 14(6):1569-1572.
///
/// # Implementation Note
/// This is a **thin wrapper** over `pillars::run_neuron<IzhikevichModel>`.
/// Parameters are repacked from row-major to column-major SoA on each call
/// using thread-local scratch buffers (no heap allocation in steady state).
pub fn izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &mut [f32],
    dt: f32,
    trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "izhikevich_step: u length mismatch");
    assert_eq!(trace.len(), n, "izhikevich_step: trace length mismatch");
    assert_eq!(a.len(), n, "izhikevich_step: a length mismatch");
    assert_eq!(b.len(), n, "izhikevich_step: b length mismatch");
    assert_eq!(ii.len(), n, "izhikevich_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "izhikevich_step: spikes length mismatch");
    assert_eq!(c.len(), n, "izhikevich_step: c length mismatch");
    assert_eq!(d.len(), n, "izhikevich_step: d length mismatch");
    assert_eq!(
        refractory.len(),
        n,
        "izhikevich_step: refractory length mismatch"
    );
    add_dendritic_current(ii, I_dend);

    // ── Pillars bridge (Фаза 2): IzhikevichModel 7-col
    // ── [dt, a, b, c, d, tau_ref, v_thresh=30]; state [v, u, refractory].
    // ── v_thresh=30 — legacy hardcoded (calibration v2 threshold).
    with_params_scratch(7, n, |p| {
        for i in 0..n {
            p[i] = dt;
            p[n + i] = a[i];
            p[2 * n + i] = b[i];
            p[3 * n + i] = c[i];
            p[4 * n + i] = d[i];
            p[5 * n + i] = tau_ref;
            p[6 * n + i] = 30.0; // legacy v_th (hardcoded)
        }
        unsafe {
            run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refractory.as_mut_ptr(), // state 2
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    });
    // ── Legacy STDP side-output: trace (НЕ состояние pillars-модели).
    // ── SIMD semantics: frozen while refractory; on spike tr = tr·td + (1−tr);
    // ── else tr·(1−td). (connectome STDP считает собственный pre_trace из спайков.)
    for i in 0..n {
        if refractory[i] > 0.0 {
            continue; // frozen during refractory
        }
        if spikes[i] != 0 {
            trace[i] = trace[i] * trace_decay + (1.0 - trace[i]);
        } else {
            trace[i] *= 1.0 - trace_decay;
        }
    }
}

/// Eligibility-propagation Izhikevich step: computes eligibility trace and synaptic gradient.
pub fn eprop_izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_sg: f32,
    I_dend: Option<&[f32]>,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_izhikevich_step: u length mismatch");
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_izhikevich_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_izhikevich_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_izhikevich_step: Ii length mismatch");
    assert_eq!(
        spikes.len(),
        n,
        "eprop_izhikevich_step: spikes length mismatch"
    );
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_izhikevich_step: pre_spikes length mismatch"
    );
    let ii = with_dendritic_current(ii, I_dend, "eprop_izhikevich_step");
    unsafe {
        izhikevich::eprop_izhikevich_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            eligibility.as_mut_ptr(),
            psi.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            pre_spikes.as_ptr(),
            dt,
            tau_e,
            a,
            b,
            c,
            d,
            tau_sg,
            n,
        )
    }
}

/// Izhikevich dual memory step: fast Izhikevich dynamics + slow memory state.
///
/// Extends `izhikevich_step` with a per-neuron slow state `m` that accumulates
/// recent spiking activity and modulates input current:
///
/// ```text
/// m[t] = τ_m · m[t-1] + (1-τ_m) · spike[t-1]
/// I += w_m · m[t-1]
/// ```
///
/// The slow state gives neurons a "memory" of recent activity, enabling
/// bistable dynamics and context-dependent behavior (Sun et al. 2026).
///
/// Ii clamp [-100, 100] встроен в SIMD kernel (fused inline load).
pub fn izhikevich_dual_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &mut [f32],
    dt: f32,
    trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    m: &mut [f32], // slow state per neuron
    tau_m: f32,    // slow state time constant (0..1, higher = slower)
    w_m: f32,      // memory-to-current weight
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "izhikevich_dual_step: u length mismatch");
    assert_eq!(
        trace.len(),
        n,
        "izhikevich_dual_step: trace length mismatch"
    );
    assert_eq!(a.len(), n, "izhikevich_dual_step: a length mismatch");
    assert_eq!(b.len(), n, "izhikevich_dual_step: b length mismatch");
    assert_eq!(ii.len(), n, "izhikevich_dual_step: Ii length mismatch");
    assert_eq!(
        spikes.len(),
        n,
        "izhikevich_dual_step: spikes length mismatch"
    );
    assert_eq!(c.len(), n, "izhikevich_dual_step: c length mismatch");
    assert_eq!(d.len(), n, "izhikevich_dual_step: d length mismatch");
    assert_eq!(
        refractory.len(),
        n,
        "izhikevich_dual_step: refractory length mismatch"
    );
    assert_eq!(
        m.len(),
        n,
        "izhikevich_dual_step: m (slow state) length mismatch"
    );
    add_dendritic_current(ii, I_dend);
    unsafe {
        izhikevich::izhikevich_dual_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            trace.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            ii.as_ptr(),
            dt,
            trace_decay,
            spikes.as_mut_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            refractory.as_mut_ptr(),
            tau_ref,
            m.as_mut_ptr(),
            tau_m,
            w_m,
            n,
        )
    }
}

/// Multi-input eprop Izhikevich: like `eprop_izhikevich_step` but with weighted sum of pre-synaptic spikes.
pub fn eprop_izhikevich_step_multi(
    v: &mut [f32],
    u: &mut [f32],
    psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    i_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_sg: f32,
    spikes: &mut [i32],
    I_dend: Option<&[f32]>,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_izhikevich_step_multi: u length mismatch");
    assert_eq!(
        psi.len(),
        n,
        "eprop_izhikevich_step_multi: psi length mismatch"
    );
    assert_eq!(
        e.len(),
        n_pre * n,
        "eprop_izhikevich_step_multi: e length mismatch"
    );
    assert_eq!(
        i_ext.len(),
        n,
        "eprop_izhikevich_step_multi: I_ext length mismatch"
    );
    assert_eq!(
        weights.len(),
        n_pre * n,
        "eprop_izhikevich_step_multi: weights length mismatch"
    );
    assert_eq!(
        spikes.len(),
        n,
        "eprop_izhikevich_step_multi: spikes length mismatch"
    );
    let i_ext = with_dendritic_current(i_ext, I_dend, "eprop_izhikevich_step_multi");
    unsafe {
        izhikevich::eprop_izhikevich_step_multi(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            psi.as_mut_ptr(),
            e.as_mut_ptr(),
            z_pre.as_ptr(),
            i_ext.as_ptr(),
            weights.as_ptr(),
            n,
            n_pre,
            dt,
            tau_e,
            a,
            b,
            c,
            d,
            tau_sg,
            spikes.as_mut_ptr(),
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / lif
// ═══════════════════════════════════════════════════════════

/// Leaky Integrate-and-Fire neuron step.
///
/// Ii clamp [-100, 100] встроен в SIMD kernel (fused inline load).
pub fn lif_step(
    v: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(ii.len(), n, "lif_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "lif_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 5,
        "lif_step: params length mismatch (expected {}*5={})",
        n,
        n * 5
    );
    assert_eq!(refractory.len(), n, "lif_step: refractory length mismatch");
    add_dendritic_current(ii, I_dend);

    // ── Pillars bridge (Фаза 2): legacy 5-col [dt, tau_m, v_rest, v_thresh, v_reset]
    // ── → LIFModel 6-col [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref].
    // ── State: v → slot 0, refractory → slot 1.
    with_params_scratch(6, n, |p| {
        for i in 0..n {
            p[i] = params[i];                          // dt
            p[n + i] = 1.0 / params[n + i].max(1e-6);  // inv_tau_m = 1/τ_m
            p[2 * n + i] = params[2 * n + i];          // v_rest
            p[3 * n + i] = params[3 * n + i];          // v_thresh
            p[4 * n + i] = params[4 * n + i];          // v_reset
            p[5 * n + i] = tau_ref;                    // tau_ref (uniform)
        }
        unsafe {
            run_neuron::<LIFModel>(
                v.as_mut_ptr(),
                refractory.as_mut_ptr(), // state 1 (refractory)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    });
}

/// LIF with adaptive threshold: `τ_m·dv/dt = −(v−v_rest) + I; τ_θ·dθ/dt = −(θ−θ_rest) + b·spike`.
pub fn lif_at_step(
    v: &mut [f32],
    theta: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(theta.len(), n, "lif_at_step: theta length mismatch");
    assert_eq!(ii.len(), n, "lif_at_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "lif_at_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 7,
        "lif_at_step: params length mismatch (expected {}*7={})",
        n,
        n * 7
    );
    assert_eq!(
        refractory.len(),
        n,
        "lif_at_step: refractory length mismatch"
    );
    add_dendritic_current(ii, I_dend);
    unsafe {
        lif::lif_at_step(
            v.as_mut_ptr(),
            theta.as_mut_ptr(),
            ii.as_mut_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            refractory.as_mut_ptr(),
            tau_ref,
            n,
        )
    }
}

/// Eligibility-propagation LIF step: computes eligibility trace for online learning.
/// Now includes refractory period (matches `lif_step` semantics).
pub fn eprop_lif_step(
    v: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    refractory: &mut [f32],
    dt: f32,
    tau_m: f32,
    tau_e: f32,
    tau_ref: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_sg: f32,
    I_dend: Option<&[f32]>,
) -> usize {
    let n = v.len();
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_lif_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_lif_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_lif_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_lif_step: spikes length mismatch");
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_lif_step: pre_spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        n,
        "eprop_lif_step: refractory length mismatch"
    );
    let ii = with_dendritic_current(ii, I_dend, "eprop_lif_step");
    unsafe {
        lif::eprop_lif_step(
            v.as_mut_ptr(),
            eligibility.as_mut_ptr(),
            psi.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            pre_spikes.as_ptr(),
            refractory.as_mut_ptr(),
            dt,
            tau_m,
            tau_e,
            tau_ref,
            v_rest,
            v_thresh,
            v_reset,
            tau_sg,
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / saw
// ═══════════════════════════════════════════════════════════

/// Sawtooth neuron step: Izhikevich-like with sawtooth reset and noise injection.
pub fn saw_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    ii: &[f32],
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "saw_step: u length mismatch");
    assert_eq!(trace.len(), n, "saw_step: trace length mismatch");
    assert_eq!(a.len(), n, "saw_step: a length mismatch");
    assert_eq!(b.len(), n, "saw_step: b length mismatch");
    assert_eq!(ii.len(), n, "saw_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "saw_step: spikes length mismatch");
    assert_eq!(refractory.len(), n, "saw_step: refractory length mismatch");
    let ii = with_dendritic_current(ii, I_dend, "saw_step");

    // ── Pillars bridge (Фаза 2): SAWModel 8-col
    // ── [dt, a, b, trace_decay, noise_amp, tau_ref, v_thresh=30, v_reset=-65].
    // ── Legacy hardcodes v_th=30, v_reset=-65 (kernel saw.rs).
    // ── NOTE: pillars SAWModel is DETERMINISTIC (no RNG noise). Legacy saw
    // ── injects noise_amp·(rand−0.5) on non-spike steps → fall back to the
    // ── legacy kernel for noisy configs so behavior is preserved exactly.
    if noise_amp != 0.0 {
        unsafe {
            saw::saw_step(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                trace.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                ii.as_ptr(),
                dt,
                trace_decay,
                noise_amp,
                spikes.as_mut_ptr(),
                refractory.as_mut_ptr(),
                tau_ref,
                n,
            )
        }
        return;
    }

    with_params_scratch(8, n, |p| {
        for i in 0..n {
            p[i] = dt;
            p[n + i] = a[i];
            p[2 * n + i] = b[i];
            p[3 * n + i] = trace_decay;
            p[4 * n + i] = 0.0; // noise_amp (deterministic path)
            p[5 * n + i] = tau_ref;
            p[6 * n + i] = 30.0;   // legacy v_th
            p[7 * n + i] = -65.0;  // legacy v_reset
        }
        unsafe {
            run_neuron::<SAWModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                trace.as_mut_ptr(),  // state 2 = ψ-wave trace
                refractory.as_mut_ptr(), // state 3
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    });
}

/// Eligibility-propagation sawtooth step: online learning gradient computation.
pub fn eprop_saw_step(
    v: &mut [f32],
    u: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    tau_sg: f32,
    I_dend: Option<&[f32]>,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_saw_step: u length mismatch");
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_saw_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_saw_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_saw_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_saw_step: spikes length mismatch");
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_saw_step: pre_spikes length mismatch"
    );
    let ii = with_dendritic_current(ii, I_dend, "eprop_saw_step");
    unsafe {
        saw::eprop_saw_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            eligibility.as_mut_ptr(),
            psi.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            pre_spikes.as_ptr(),
            dt,
            tau_e,
            a,
            b,
            tau_sg,
            n,
        )
    }
}

/// Graph-coupled oscillator step: v+iu evolves by H_psi on unit circle.
pub fn gco_step(
    v: &mut [f32],
    u: &mut [f32],
    h_real: &mut [f32],
    h_imag: &mut [f32],
    i_ext: &mut [f32],
    speed: &mut [f32],
    dt: f32,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "gco_step: u length mismatch");
    assert_eq!(h_real.len(), n, "gco_step: h_real length mismatch");
    assert_eq!(h_imag.len(), n, "gco_step: h_imag length mismatch");
    assert_eq!(i_ext.len(), n, "gco_step: i_ext length mismatch");
    assert_eq!(speed.len(), n, "gco_step: speed length mismatch");
    unsafe {
        saw::gco_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            h_real.as_mut_ptr(),
            h_imag.as_mut_ptr(),
            i_ext.as_mut_ptr(),
            speed.as_mut_ptr(),
            dt,
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / adex
// ═══════════════════════════════════════════════════════════

/// Adaptive Exponential Integrate-and-Fire (AdEx) — biophysically detailed spiking model.
///
/// Combines exponential spike initiation with spike-frequency adaptation. Captures
/// diverse cortical neuron behaviors: regular spiking, bursting, adaptation, rebound.
///
/// # Equations
/// ```text
/// τ_m · dv/dt = -(v - v_rest) + Δ_T · exp((v - v_thresh) / Δ_T) - w + I
/// τ_w · dw/dt = a · (v - v_rest) - w
/// 
/// if v ≥ v_peak:
///     spike
///     v ← v_reset
///     w ← w + b
///     refractory ← τ_ref
/// ```
///
/// # Parameters (column-major, 9 × n)
/// - `dt`: time step (ms)
/// - `τ_m`: membrane time constant (ms) — typically 5-20 ms
/// - `τ_w`: adaptation time constant (ms) — typically 100-500 ms
/// - `v_rest`: resting potential (mV) — typically -65 mV
/// - `v_thresh`: spike threshold (mV) — typically -50 mV
/// - `v_reset`: reset potential (mV) — typically -65 mV
/// - `Δ_T`: exponential slope factor (mV) — typically 2 mV
/// - `a`: subthreshold adaptation (nS) — typically 0-4 nS
/// - `b`: spike-triggered adaptation (nA) — typically 0-100 nA
///
/// # State Variables
/// - `v`: membrane potential (mV) — fast variable with exponential spike
/// - `w`: adaptation current (nA) — slow variable, linear dynamics
/// - `refractory`: refractory counter (ms)
///
/// # Adaptation Mechanisms
/// 1. **Subthreshold** (`a > 0`): `w` tracks `v - v_rest`, creating resonance
/// 2. **Spike-triggered** (`b > 0`): `w` jumps by `b` on each spike, slowing firing
///
/// # Typical Parameter Sets
/// | Behavior | τ_m | τ_w | Δ_T | a   | b   | Description |
/// |----------|-----|-----|-----|-----|-----|-------------|
/// | RS       | 10  | 100 | 2   | 0.5 | 7   | Regular Spiking |
/// | Burst    | 5   | 100 | 2   | 0.5 | 7   | Bursting |
/// | Fast     | 5   | 100 | 0.5 | 0   | 0   | Fast Spiking (no adaptation) |
/// | Adapt    | 10  | 500 | 2   | 2   | 50  | Strong Adaptation |
///
/// # Input Current
/// - `ii`: synaptic + external input (nA)
/// - **Automatic clamping**: `ii ∈ [-100, 100]` (prevents saturation)
/// - `I_dend`: optional dendritic current (same length as `ii`)
///
/// # Refractory Period
/// During refractory (`refractory > 0`):
/// - `v` and `w` are **frozen** (no integration)
/// - Refractory counter decrements by `dt`
/// - No spike detection
///
/// # Exponential Spike Mechanism
/// The term `Δ_T · exp((v - v_thresh) / Δ_T)` creates a sharp upstroke:
/// - Below threshold: negligible (~0.01 mV/ms)
/// - Near threshold: rapid acceleration (exponential blow-up)
/// - Spike detected at `v ≥ v_peak` (typically `v_peak = v_thresh`)
///
/// # Integration Method
/// - **Membrane**: 2 half-steps (Forward Euler) for numerical stability
/// - **Adaptation**: Single full step (linear, stable)
/// - **Exponential term**: Clamped to `[-20, 20]` before `exp()` to prevent overflow
///
/// # Performance
/// - **SIMD**: 8 neurons/instruction (AVX2)
/// - **Throughput**: ~4.2 ms/step for 1M neurons
/// - **Memory**: Column-major SoA (cache-friendly)
/// - **Overhead vs LIF**: +68% (exponential + adaptation)
///
/// # Example
/// ```rust,ignore
/// use taas_checkmate::kernels::adex_step;
///
/// let n = 1000;
/// let mut v = vec![-65.0; n];
/// let mut w = vec![0.0; n];
/// let mut ii = vec![200.0; n];
/// let mut spikes = vec![0i32; n];
/// let mut refractory = vec![0.0; n];
///
/// // Parameters: [dt, τ_m, τ_w, v_rest, v_thresh, v_reset, Δ_T, a, b]
/// let mut params = vec![0.0; n * 9];
/// for i in 0..n {
///     params[i] = 0.05;           // dt
///     params[n + i] = 5.0;        // τ_m
///     params[2*n + i] = 100.0;    // τ_w
///     params[3*n + i] = -65.0;    // v_rest
///     params[4*n + i] = -50.0;    // v_thresh
///     params[5*n + i] = -65.0;    // v_reset
///     params[6*n + i] = 2.0;      // Δ_T
///     params[7*n + i] = 0.5;      // a
///     params[8*n + i] = 7.0;      // b
/// }
///
/// for _ in 0..1000 {
///     adex_step(
///         &mut v, &mut w, &mut ii,
///         &mut spikes, &params,
///         &mut refractory,
///         2.0,  // τ_ref
///         None, // I_dend
///     );
/// }
/// ```
///
/// # Reference
/// Brette, R., & Gerstner, W. (2005). "Adaptive exponential integrate-and-fire
/// model as an effective description of neuronal activity."
/// Journal of Neurophysiology 94(5):3637-3642.
///
/// # Implementation Note
/// This is a **thin wrapper** over `pillars::run_neuron<AdExModel>`.
/// Parameters are repacked from legacy 9-column to pillars 12-column format
/// (adding pre-inverted time constants and `v_peak`) using thread-local scratch.
pub fn adex_step(
    v: &mut [f32],
    u: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(u.len(), n, "adex_step: u length mismatch");
    assert_eq!(ii.len(), n, "adex_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "adex_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 9,
        "adex_step: params length mismatch (expected {}*9={})",
        n,
        n * 9
    );
    assert_eq!(refractory.len(), n, "adex_step: refractory length mismatch");
    add_dendritic_current(ii, I_dend);

    // ── Pillars bridge (Фаза 2): legacy 9-col
    // ── [dt, tau_m, tau_w, v_rest, v_thresh, v_reset, delta_t, a, b]
    // ── → AdExModel 12-col [dt, inv_tau_m, inv_tau_w, v_rest, v_thresh, v_reset,
    // ──    delta_t, inv_delta_t, a, b, tau_ref, v_peak].
    // ── Legacy spikes at v ≥ v_thresh → v_peak := v_thresh (parity).
    with_params_scratch(12, n, |p| {
        for i in 0..n {
            p[i] = params[i];                          // dt
            p[n + i] = 1.0 / params[n + i].max(1e-6);  // inv_tau_m
            p[2 * n + i] = 1.0 / params[2 * n + i].max(1e-6); // inv_tau_w
            p[3 * n + i] = params[3 * n + i];          // v_rest
            p[4 * n + i] = params[4 * n + i];          // v_thresh
            p[5 * n + i] = params[5 * n + i];          // v_reset
            p[6 * n + i] = params[6 * n + i];          // delta_t
            p[7 * n + i] = 1.0 / params[6 * n + i].max(1e-6); // inv_delta_t
            p[8 * n + i] = params[7 * n + i];          // a
            p[9 * n + i] = params[8 * n + i];          // b
            p[10 * n + i] = tau_ref;
            p[11 * n + i] = params[4 * n + i];         // v_peak := v_thresh
        }
        unsafe {
            run_neuron::<AdExModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refractory.as_mut_ptr(), // state 2
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                ii.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    });
}

/// Multi-input eprop AdEx: eligibility propagation with weighted pre-synaptic input.
pub fn eprop_adex_step_multi(
    v: &mut [f32],
    w: &mut [f32],
    psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    i_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32,
    tau_m: f32,
    tau_w: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    delta_t: f32,
    a: f32,
    b: f32,
    tau_e: f32,
    tau_sg: f32,
    spikes: &mut [i32],
    I_dend: Option<&[f32]>,
) -> usize {
    let n = v.len();
    assert_eq!(w.len(), n, "eprop_adex_step_multi: w length mismatch");
    assert_eq!(psi.len(), n, "eprop_adex_step_multi: psi length mismatch");
    assert_eq!(
        e.len(),
        n_pre * n,
        "eprop_adex_step_multi: e length mismatch"
    );
    assert_eq!(
        i_ext.len(),
        n,
        "eprop_adex_step_multi: I_ext length mismatch"
    );
    assert_eq!(
        weights.len(),
        n_pre * n,
        "eprop_adex_step_multi: weights length mismatch"
    );
    assert_eq!(
        spikes.len(),
        n,
        "eprop_adex_step_multi: spikes length mismatch"
    );
    let i_ext = with_dendritic_current(i_ext, I_dend, "eprop_adex_step_multi");
    unsafe {
        adex::eprop_adex_step_multi(
            v.as_mut_ptr(),
            w.as_mut_ptr(),
            psi.as_mut_ptr(),
            e.as_mut_ptr(),
            z_pre.as_ptr(),
            i_ext.as_ptr(),
            weights.as_ptr(),
            n,
            n_pre,
            dt,
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_e,
            tau_sg,
            spikes.as_mut_ptr(),
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / hh (Hodgkin-Huxley - polynomial)
// ═══════════════════════════════════════════════════════════

/// Build HH channel metadata from standard channel specs.
///
/// Создаёт `Vec<ChannelMeta>` из `[ChannelSpec]`, нужный для `hh_poly_step`.
pub fn build_hh_channel_meta(channels: &[hh::ChannelSpec]) -> Vec<hh::ChannelMeta> {
    hh::build_channel_meta(channels)
}

/// Hodgkin-Huxley step using polynomial rate functions (no LUT).
///
/// `celsius`: температура в °C (6.3 = squid axon @ H&H 1952, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron g_max, layout [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
/// Size: [n_channels * n]
pub fn hh_poly_step(
    v: &mut [f32],
    gates: &mut [f32],
    ii: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    channel_meta: &[hh::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32,
    celsius: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let n_channels = channel_meta.len();
    let n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();
    assert_eq!(
        gates.len(),
        n_gates * n,
        "hh_poly_step: gates length mismatch (expected {}*{}={})",
        n_gates,
        n,
        n_gates * n
    );
    assert_eq!(ii.len(), n, "hh_poly_step: Ii length mismatch");
    assert_eq!(g_leak.len(), n, "hh_poly_step: g_leak length mismatch");
    assert_eq!(e_leak.len(), n, "hh_poly_step: E_leak length mismatch");
    assert_eq!(c_m.len(), n, "hh_poly_step: C_m length mismatch");
    assert_eq!(
        g_max_per_neuron.len(),
        n_channels * n,
        "hh_poly_step: g_max_per_neuron length mismatch"
    );
    let ii = with_dendritic_current(ii, I_dend, "hh_poly_step");

    // ── Pillars bridge (Фаза 2): standard/pacemaking лейаут → HHModel.
    // ── Legacy hh_poly_step НЕ детектит спайки → спайк-буфер — scratch.
    if is_standard_hh_layout(channel_meta) {
        with_params_scratch(HHModel::N_PARAMS, n, |p| {
            with_aux_scratch(2 * n, |aux| {
                with_spikes_scratch(n, |spikes_scratch| {
                    let (c_state, was_below) = aux.split_at_mut(n);
                    fill_hh_params(p, n, g_leak, e_leak, c_m, channel_meta,
                        g_max_per_neuron, dt, celsius);
                    unsafe {
                        run_neuron::<HHModel>(
                            v.as_mut_ptr(),
                            gates.as_mut_ptr(),
                            gates.as_mut_ptr().add(n),
                            gates.as_mut_ptr().add(2 * n),
                            c_state.as_mut_ptr(),
                            was_below.as_mut_ptr(),
                            ii.as_ptr(),
                            spikes_scratch.as_mut_ptr(),
                            p.as_ptr(),
                            n,
                        );
                    }
                })
            })
        });
    } else {
        unsafe {
            hh::hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                channel_meta,
                g_max_per_neuron.as_ptr(),
                dt,
                celsius,
            )
        }
    }
}

/// True для standard (Na m³h + K n⁴) и pacemaking (+CaV-T c) лейаутов,
/// которые pillars HHModel воспроизводит 1:1 (gate order m,h,n[,c]).
/// Произвольный лейаут каналов → fallback на legacy-ядро.
fn is_standard_hh_layout(meta: &[hh::ChannelMeta]) -> bool {
    use hh::GateType;
    match meta {
        [na, k] =>
            na.n_gates == 2 && k.n_gates == 1
                && na.gate_type == [GateType::NaM, GateType::NaH]
                && na.powers == [3, 1]
                && k.gate_type == [GateType::Kn]
                && k.powers == [4],
        [na, k, cat] =>
            na.n_gates == 2 && k.n_gates == 1 && cat.n_gates == 1
                && na.gate_type == [GateType::NaM, GateType::NaH]
                && na.powers == [3, 1]
                && k.gate_type == [GateType::Kn]
                && k.powers == [4]
                && cat.gate_type == [GateType::CatC]
                && cat.powers == [1],
        _ => false,
    }
}

/// Заполнить HHModel-параметры из legacy-лейаута (standard/pacemaking).
///
/// `g_max_per_neuron` даёт per-neuron g_max по каналам; q10 — Q10-фактор
/// температуры (3^(ΔT/10), референс 6.3°C); v_thresh=0 — legacy crossing.
#[inline]
fn fill_hh_params(
    p: &mut [f32],
    n: usize,
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    channel_meta: &[hh::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32,
    celsius: f32,
) {
    let q10 = 3.0f32.powf((celsius - 6.3) / 10.0);
    let has_cat = channel_meta.len() == 3;
    for i in 0..n {
        p[i] = dt;
        p[n + i] = g_max_per_neuron[i];              // g_Na
        p[2 * n + i] = channel_meta[0].e_rev;        // E_Na
        p[3 * n + i] = g_max_per_neuron[n + i];      // g_K
        p[4 * n + i] = channel_meta[1].e_rev;        // E_K
        p[5 * n + i] = g_leak[i];                    // g_Leak
        p[6 * n + i] = e_leak[i];                    // E_Leak
        p[7 * n + i] = q10;
        p[8 * n + i] = 0.0;                          // v_thresh (legacy @ 0)
        p[9 * n + i] = 1.0 / c_m[i].max(1e-6);       // inv_cm
        p[10 * n + i] = if has_cat { g_max_per_neuron[2 * n + i] } else { 0.0 };
        p[11 * n + i] = if has_cat { channel_meta[2].e_rev } else { 120.0 };
    }
}

/// Fused HH step: Hodgkin-Huxley dynamics + spike detection + Q10 в один проход.
///
/// `celsius`: температура в °C (6.3 = squid axon @ H&H 1952, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron g_max, layout [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
/// Size: [n_channels * n]
pub fn fused_hh_poly_step(
    v: &mut [f32],
    gates: &mut [f32],
    spikes: &mut [i32],
    ii: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    channel_meta: &[hh::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32,
    celsius: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    let n_channels = channel_meta.len();
    let n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();
    assert_eq!(
        gates.len(),
        n_gates * n,
        "fused_hh_poly_step: gates length mismatch (expected {}*{}={})",
        n_gates,
        n,
        n_gates * n
    );
    assert_eq!(
        spikes.len(),
        n,
        "fused_hh_poly_step: spikes length mismatch"
    );
    assert_eq!(ii.len(), n, "fused_hh_poly_step: Ii length mismatch");
    assert_eq!(
        g_leak.len(),
        n,
        "fused_hh_poly_step: g_leak length mismatch"
    );
    assert_eq!(
        e_leak.len(),
        n,
        "fused_hh_poly_step: E_leak length mismatch"
    );
    assert_eq!(c_m.len(), n, "fused_hh_poly_step: C_m length mismatch");
    assert_eq!(
        g_max_per_neuron.len(),
        n_channels * n,
        "fused_hh_poly_step: g_max_per_neuron length mismatch"
    );
    let ii = with_dendritic_current(ii, I_dend, "fused_hh_poly_step");

    // ── Pillars bridge (Фаза 2): HHModel для standard/pacemaking лейаута.
    // ── Legacy spike detection — REVERSE zero-crossing (v_old ≥ 0 && v_new < 0);
    // ── HHModel детектит rising-edge → перезаписываем спайки вручную.
    if is_standard_hh_layout(channel_meta) {
        with_params_scratch(HHModel::N_PARAMS, n, |p| {
            with_aux_scratch(3 * n, |aux| {
                let (v_old, rest) = aux.split_at_mut(n);
                let (c_state, was_below) = rest.split_at_mut(n);
                fill_hh_params(p, n, g_leak, e_leak, c_m, channel_meta,
                    g_max_per_neuron, dt, celsius);
                v_old.copy_from_slice(v);
                unsafe {
                    run_neuron::<HHModel>(
                        v.as_mut_ptr(),
                        gates.as_mut_ptr(),
                        gates.as_mut_ptr().add(n),
                        gates.as_mut_ptr().add(2 * n),
                        c_state.as_mut_ptr(),
                        was_below.as_mut_ptr(),
                        ii.as_ptr(),
                        spikes.as_mut_ptr(),
                        p.as_ptr(),
                        n,
                    );
                }
                // Legacy convention: falling edge through 0
                for i in 0..n {
                    spikes[i] = if v_old[i] >= 0.0 && v[i] < 0.0 { 1 } else { 0 };
                }
            })
        });
    } else {
        unsafe {
            hh::fused_hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                channel_meta,
                g_max_per_neuron.as_ptr(),
                dt,
                celsius,
                spikes.as_mut_ptr(),
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / haap (Hyperpolarization-Activated Adaptive Pulse-Bursting)
// ═══════════════════════════════════════════════════════════

/// HAAP neuron step: VNC model with M-current adaptation + HCN h-current.
///
/// Equations:
///   C_m·dV/dt = -g_L(V-E_L) - g_M·u·(V-E_K) - g_h·h·(V-E_h) + I_syn
///   τ_u·du/dt = (u_∞(V) - u)
///   τ_h·dh/dt = (h_∞(V) - h)
///
/// Params layout: dt, C_m, g_L, E_L, g_M, E_K, g_h, E_h, tau_u, tau_h,
///                V_half_u, k_u, V_half_h, k_h, V_thresh, V_reset, delta_u (17 × n)
pub fn haap_step(
    v: &mut [f32],
    u_state: &mut [f32],
    h_state: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    params: &[f32],
) {
    let n = v.len();
    assert_eq!(u_state.len(), n, "haap_step: u_state length mismatch");
    assert_eq!(h_state.len(), n, "haap_step: h_state length mismatch");
    assert_eq!(ii.len(), n, "haap_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "haap_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 17,
        "haap_step: params length mismatch (expected {}*17={})",
        n,
        n * 17
    );
    unsafe {
        haap::haap_step(
            v.as_mut_ptr(),
            u_state.as_mut_ptr(),
            h_state.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / fhn (FitzHugh-Nagumo)
// ═══════════════════════════════════════════════════════════

/// FitzHugh-Nagumo neuron step: simplified Hodgkin-Huxley with cubic nullcline.
///
/// Equations:
///   dv/dt = v - v³/3 - w + I_ext
///   dw/dt = (v + a - b·w) / τ
///
/// Params layout: dt, tau_w, a, b (4 × n, column-major).
pub fn fhn_step(
    v: &mut [f32],
    w: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    params: &[f32],
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(w.len(), n, "fhn_step: w length mismatch");
    assert_eq!(ii.len(), n, "fhn_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "fhn_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 4,
        "fhn_step: params length mismatch (expected {}*4={})",
        n,
        n * 4
    );
    let ii = with_dendritic_current(ii, I_dend, "fhn_step");
    unsafe {
        fhn::fhn_step(
            v.as_mut_ptr(),
            w.as_mut_ptr(),
            ii.as_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / wc (Wilson-Cowan)
// ═══════════════════════════════════════════════════════════

/// Dendritic computation step: compartmentalized dendrites with local PSP integration.
///
/// Обновляет дендритные потенциалы компартментов и аккумулирует ток в сому.
/// После вызова `I_dend` содержит полный дендритный вклад для каждого нейрона
/// (может быть прибавлен к `Ii` в Izhikevich/LIF/AdEx шаге).
///
/// См. [`DendriteState`] для safe API через struct.
pub fn dendrite_step(state: &mut dendrite::DendriteState, V_soma: &[f32], dt: f32) {
    assert_eq!(
        V_soma.len(),
        state.n,
        "dendrite_step: V_soma length mismatch"
    );
    state.step(V_soma, dt);
}


// ═══════════════════════════════════════════════════════════
//  neuron / glif (Generalized LIF — Allen Institute)
// ═══════════════════════════════════════════════════════════

/// GLIF3 step: LIF + after-spike current (I_ahp).
///
/// Params layout (8 × n): [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]
/// State: v[n], g_ahp[n], refractory[n], spikes[n]
///
/// Equations:
///   dv/dt = -g_leak·(v - E_leak) + I_syn + g_ahp·(v - E_ahp)
///   dg_ahp/dt = -g_ahp / τ_ahp    (exponential Euler)
///   On spike: v = v_reset, g_ahp += Δg_ahp
///
/// Allen Institute — Teeter et al. 2018, Nature Communications
pub fn glif3_step(
    v: &mut [f32],
    g_ahp: &mut [f32],
    Ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(g_ahp.len(), n, "glif3_step: g_ahp length mismatch");
    assert_eq!(Ii.len(), n, "glif3_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "glif3_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 8,
        "glif3_step: params length mismatch (expected {}*8={})",
        n,
        n * 8
    );
    assert_eq!(
        refractory.len(),
        n,
        "glif3_step: refractory length mismatch"
    );
    add_dendritic_current(Ii, I_dend);
    unsafe {
        glif::glif3_step(
            v.as_mut_ptr(),
            g_ahp.as_mut_ptr(),
            Ii.as_mut_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            refractory.as_mut_ptr(),
            tau_ref,
            n,
        )
    }
}

/// GLIF4 step: GLIF3 + threshold adaptation (θ).
///
/// Params layout (10 × n): [dt, g_leak, E_leak, v_thresh_0, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]
/// State: v[n], theta[n], g_ahp[n], refractory[n], spikes[n]
///
/// Equations:
///   Same as GLIF3 + adaptive threshold:
///   dθ/dt = -θ / τ_θ  (exponential Euler)
///   Effective threshold = v_thresh_0 + θ
///   On spike: θ += Δθ
///
/// Allen Institute — Teeter et al. 2018, Nature Communications
pub fn glif4_step(
    v: &mut [f32],
    theta: &mut [f32],
    g_ahp: &mut [f32],
    Ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    I_dend: Option<&[f32]>,
) {
    let n = v.len();
    assert_eq!(theta.len(), n, "glif4_step: theta length mismatch");
    assert_eq!(g_ahp.len(), n, "glif4_step: g_ahp length mismatch");
    assert_eq!(Ii.len(), n, "glif4_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "glif4_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 10,
        "glif4_step: params length mismatch (expected {}*10={})",
        n,
        n * 10
    );
    assert_eq!(
        refractory.len(),
        n,
        "glif4_step: refractory length mismatch"
    );
    add_dendritic_current(Ii, I_dend);
    unsafe {
        glif::glif4_step(
            v.as_mut_ptr(),
            theta.as_mut_ptr(),
            g_ahp.as_mut_ptr(),
            Ii.as_mut_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            refractory.as_mut_ptr(),
            tau_ref,
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / bcm — Bienenstock-Cooper-Munro Plasticity
// ═══════════════════════════════════════════════════════════

/// BCM plasticity step: rate-based Hebbian learning with sliding threshold.
///
/// Updates pre/post traces, sliding threshold θ_M, and synaptic weights.
/// LTP when y > θ_M, LTD when y < θ_M.
pub fn bcm_step(
    x: &mut [f32],
    y: &mut [f32],
    theta_m: &mut [f32],
    weights: &mut [f32],
    pre_rates: &[f32],
    post_rates: &[f32],
    dt: f32,
    tau_x: f32,
    tau_y: f32,
    tau_theta: f32,
    eta: f32,
    decay: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = x.len();
    assert_eq!(y.len(), n, "bcm_step: y length mismatch");
    assert_eq!(theta_m.len(), n, "bcm_step: theta_m length mismatch");
    assert_eq!(weights.len(), n, "bcm_step: weights length mismatch");
    assert_eq!(pre_rates.len(), n, "bcm_step: pre_rates length mismatch");
    assert_eq!(post_rates.len(), n, "bcm_step: post_rates length mismatch");
    unsafe {
        bcm::bcm_step(
            x.as_mut_ptr(), y.as_mut_ptr(), theta_m.as_mut_ptr(),
            weights.as_mut_ptr(), pre_rates.as_ptr(), post_rates.as_ptr(),
            dt, tau_x, tau_y, tau_theta, eta, decay, w_min, w_max, n,
        );
    }
}

/// Homeostatic synaptic scaling: Δw = η · (r_target − r_mean) · w · dt
pub fn homeostatic_scale(
    weights: &mut [f32],
    post_rates: &[f32],
    dt: f32,
    eta_h: f32,
    target_rate: f32,
) {
    let n = weights.len();
    assert_eq!(post_rates.len(), n, "homeostatic_scale: post_rates length mismatch");
    unsafe {
        bcm::homeostatic_scale(
            weights.as_mut_ptr(),
            post_rates.as_ptr(),
            n,
            dt,
            eta_h,
            target_rate,
        )
    }
}

pub fn wc_step(
    e: &mut [f32],
    i: &mut [f32],
    i_e: &[f32],
    i_i: &[f32],
    spikes_e: &mut [f32],
    spikes_i: &mut [f32],
    params: &[f32],
) {
    let n = e.len();
    assert_eq!(i.len(), n, "wc_step: i length mismatch");
    assert_eq!(i_e.len(), n, "wc_step: i_e length mismatch");
    assert_eq!(i_i.len(), n, "wc_step: i_i length mismatch");
    assert_eq!(spikes_e.len(), n, "wc_step: spikes_e length mismatch");
    assert_eq!(spikes_i.len(), n, "wc_step: spikes_i length mismatch");
    assert_eq!(
        params.len(),
        n * 8,
        "wc_step: params length mismatch (expected {}*8={})",
        n,
        n * 8
    );
    unsafe {
        wc::wc_step(
            e.as_mut_ptr(),
            i.as_mut_ptr(),
            i_e.as_ptr(),
            i_i.as_ptr(),
            spikes_e.as_mut_ptr(),
            spikes_i.as_mut_ptr(),
            params.as_ptr(),
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / stp
// ═══════════════════════════════════════════════════════════

/// Short-Term Plasticity step: dual-exponential facilitation/depression dynamics.
pub fn stp_step(
    x: &mut [f32],
    u: &mut [f32],
    pre_spikes: &[i32],
    tau_d: &[f32],
    tau_f: &[f32],
    u0: &[f32],
    a: &[f32],
    i_out: &mut [f32],
    dt: f32,
) {
    let n = x.len();
    assert_eq!(u.len(), n, "stp_step: u length mismatch");
    assert_eq!(pre_spikes.len(), n, "stp_step: pre_spikes length mismatch");
    assert_eq!(tau_d.len(), n, "stp_step: tau_d length mismatch");
    assert_eq!(tau_f.len(), n, "stp_step: tau_f length mismatch");
    assert_eq!(u0.len(), n, "stp_step: u0 length mismatch");
    assert_eq!(a.len(), n, "stp_step: a length mismatch");
    assert_eq!(i_out.len(), n, "stp_step: I_out length mismatch");
    unsafe {
        stp::stp_step(
            x.as_mut_ptr(),
            u.as_mut_ptr(),
            pre_spikes.as_ptr(),
            tau_d.as_ptr(),
            tau_f.as_ptr(),
            u0.as_ptr(),
            a.as_ptr(),
            i_out.as_mut_ptr(),
            dt,
            n,
        )
    }
}

/// STDP weight update: `w += lr_ltp·trace_post − lr_ltd·trace_pre`, clamped to [w_min, w_max].
pub fn stdp_update(
    weights: &mut [f32],
    traces_pre: &[f32],
    traces_post: &[f32],
    lr_ltd: f32,
    lr_ltp: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = weights.len();
    assert_eq!(
        traces_pre.len(),
        n,
        "stdp_update: traces_pre length mismatch"
    );
    assert_eq!(
        traces_post.len(),
        n,
        "stdp_update: traces_post length mismatch"
    );
    unsafe {
        stp::stdp_update(
            weights.as_mut_ptr(),
            traces_pre.as_ptr(),
            traces_post.as_ptr(),
            lr_ltd,
            lr_ltp,
            w_min,
            w_max,
            n,
        )
    }
}

/// Homeostatic STDP: same as `stdp_update` but with target firing rate constraint.
pub fn stdp_homeostatic(
    weights: &mut [f32],
    traces_pre: &[f32],
    traces_post: &[f32],
    lr_ltd: f32,
    lr_ltp: f32,
    target_sum: f32,
    w_max: f32,
) {
    let n = weights.len();
    assert_eq!(
        traces_pre.len(),
        n,
        "stdp_homeostatic: traces_pre length mismatch"
    );
    assert_eq!(
        traces_post.len(),
        n,
        "stdp_homeostatic: traces_post length mismatch"
    );
    unsafe {
        stp::stdp_homeostatic(
            weights.as_mut_ptr(),
            traces_pre.as_ptr(),
            traces_post.as_ptr(),
            lr_ltd,
            lr_ltp,
            target_sum,
            w_max,
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / delay
// ═══════════════════════════════════════════════════════════

/// Circular delay buffer read: rotate buffer by `pos` positions.
pub fn delay_buffer_circular(output: &mut [f32], buffer: &[f32], pos: usize) {
    let n = output.len();
    assert!(
        buffer.len() >= n,
        "delay_buffer_circular: buffer length {} < n {}",
        buffer.len(),
        n
    );
    unsafe { delay::delay_buffer_circular(output.as_mut_ptr(), buffer.as_ptr(), n, pos) }
}

/// Multi-tap delay read: read multiple delay values at specified tap positions.
pub fn delay_buffer_multi_tap(output: &mut [f32], buffer: &[f32], delays: &[i32]) {
    let n = buffer.len();
    let n_taps = output.len();
    assert_eq!(
        delays.len(),
        n_taps,
        "delay_buffer_multi_tap: delays length mismatch"
    );
    unsafe {
        delay::delay_buffer_multi_tap(
            output.as_mut_ptr(),
            buffer.as_ptr(),
            delays.as_ptr(),
            n,
            n_taps,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / dopamine
// ═══════════════════════════════════════════════════════════

/// Dopamine-modulated trace step: pre/post traces + eligibility trace with exponential decay.
pub fn dopamine_traces_step(
    trace_pre: &mut [f32],
    trace_post: &mut [f32],
    eligibility: &mut [f32],
    pre_spikes: &[i32],
    post_spikes: &[i32],
    dt: f32,
    tau_pre: f32,
    tau_post: f32,
    tau_e: f32,
    a_ltp: f32,
    a_ltd: f32,
) {
    let n = trace_pre.len();
    assert_eq!(
        trace_post.len(),
        n,
        "dopamine_traces_step: trace_post length mismatch"
    );
    assert_eq!(
        eligibility.len(),
        n,
        "dopamine_traces_step: eligibility length mismatch"
    );
    assert_eq!(
        pre_spikes.len(),
        n,
        "dopamine_traces_step: pre_spikes length mismatch"
    );
    assert_eq!(
        post_spikes.len(),
        n,
        "dopamine_traces_step: post_spikes length mismatch"
    );
    unsafe {
        dopamine::dopamine_traces_step(
            trace_pre.as_mut_ptr(),
            trace_post.as_mut_ptr(),
            eligibility.as_mut_ptr(),
            pre_spikes.as_ptr(),
            post_spikes.as_ptr(),
            dt,
            tau_pre,
            tau_post,
            tau_e,
            a_ltp,
            a_ltd,
            n,
        )
    }
}

/// Dopamine weight update: `w += lr · dopamine · eligibility`, clamped to [w_min, w_max].
pub fn dopamine_weight_update(
    weights: &mut [f32],
    eligibility: &[f32],
    dopamine: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = weights.len();
    assert_eq!(
        eligibility.len(),
        n,
        "dopamine_weight_update: eligibility length mismatch"
    );
    unsafe {
        dopamine::dopamine_weight_update(
            weights.as_mut_ptr(),
            eligibility.as_ptr(),
            dopamine,
            lr,
            w_min,
            w_max,
            n,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  neuron / fused
// ═══════════════════════════════════════════════════════════

/// Fused hidden layer step: CSR spike propagation + Izhikevich dynamics in one kernel.
///
/// `active_buf`: pre-allocated buffer for active pre-synapse indices (avoids per-step Vec allocation).
pub fn fused_hidden_izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    weights: &[f32],
    input_spikes: &[i32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(
        u.len(),
        nh,
        "fused_hidden_izhikevich_step: u length mismatch"
    );
    assert_eq!(
        trace.len(),
        nh,
        "fused_hidden_izhikevich_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "fused_hidden_izhikevich_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "fused_hidden_izhikevich_step: refractory length mismatch"
    );
    assert_eq!(
        a.len(),
        nh,
        "fused_hidden_izhikevich_step: a length mismatch"
    );
    assert_eq!(
        b.len(),
        nh,
        "fused_hidden_izhikevich_step: b length mismatch"
    );
    assert_eq!(
        c.len(),
        nh,
        "fused_hidden_izhikevich_step: c length mismatch"
    );
    assert_eq!(
        d.len(),
        nh,
        "fused_hidden_izhikevich_step: d length mismatch"
    );
    assert_eq!(
        weights.len(),
        ni * nh,
        "fused_hidden_izhikevich_step: weights length mismatch"
    );
    assert!(
        input_spikes.len() >= ni,
        "fused_hidden_izhikevich_step: input_spikes length {} < ni {}",
        input_spikes.len(),
        ni
    );
    assert!(
        nh.is_multiple_of(8),
        "fused_hidden_izhikevich_step: nh must be multiple of 8, got {}",
        nh
    );
    unsafe {
        fused::fused_hidden_izhikevich_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            trace.as_mut_ptr(),
            spikes.as_mut_ptr(),
            refractory.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            weights.as_ptr(),
            input_spikes.as_ptr(),
            dt,
            trace_decay,
            tau_ref,
            ni,
            nh,
            active_buf,
        )
    }
}

/// Fused output layer step: CSR spike propagation + LIF dynamics in one kernel.
pub fn fused_output_lif_step(
    v: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_th: &[f32],
    v_reset: &[f32],
    weights: &[f32],
    hidden_spikes: &[i32],
    dt: f32,
    tau_ref: f32,
    nh: usize,
) {
    let na = v.len();
    assert_eq!(
        spikes.len(),
        na,
        "fused_output_lif_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        na,
        "fused_output_lif_step: refractory length mismatch"
    );
    assert_eq!(
        tau_m.len(),
        na,
        "fused_output_lif_step: tau_m length mismatch"
    );
    assert_eq!(
        v_rest.len(),
        na,
        "fused_output_lif_step: v_rest length mismatch"
    );
    assert_eq!(
        v_th.len(),
        na,
        "fused_output_lif_step: v_th length mismatch"
    );
    assert_eq!(
        v_reset.len(),
        na,
        "fused_output_lif_step: v_reset length mismatch"
    );
    assert_eq!(
        weights.len(),
        nh * na,
        "fused_output_lif_step: weights length mismatch"
    );
    assert!(
        hidden_spikes.len() >= nh,
        "fused_output_lif_step: hidden_spikes length {} < nh {}",
        hidden_spikes.len(),
        nh
    );
    assert!(
        na.is_multiple_of(8),
        "fused_output_lif_step: na must be multiple of 8, got {}",
        na
    );
    unsafe {
        fused::fused_output_lif_step(
            v.as_mut_ptr(),
            spikes.as_mut_ptr(),
            refractory.as_mut_ptr(),
            tau_m.as_ptr(),
            v_rest.as_ptr(),
            v_th.as_ptr(),
            v_reset.as_ptr(),
            weights.as_ptr(),
            hidden_spikes.as_ptr(),
            dt,
            tau_ref,
            nh,
            na,
        )
    }
}

/// Poisson fused hidden step: Poisson input generation + CSR propagation + Izhikevich.
///
/// `active_buf`: pre-allocated buffer for active pre-synapse indices (avoids per-step Vec allocation).
pub fn poisson_fused_hidden_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    weights: &[f32],
    rng_states: &mut [u32],
    rates: &[f32],
    input_spikes_out: &mut [i32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(u.len(), nh, "poisson_fused_hidden_step: u length mismatch");
    assert_eq!(
        trace.len(),
        nh,
        "poisson_fused_hidden_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "poisson_fused_hidden_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "poisson_fused_hidden_step: refractory length mismatch"
    );
    assert_eq!(a.len(), nh, "poisson_fused_hidden_step: a length mismatch");
    assert_eq!(b.len(), nh, "poisson_fused_hidden_step: b length mismatch");
    assert_eq!(c.len(), nh, "poisson_fused_hidden_step: c length mismatch");
    assert_eq!(d.len(), nh, "poisson_fused_hidden_step: d length mismatch");
    assert_eq!(
        weights.len(),
        ni * nh,
        "poisson_fused_hidden_step: weights length mismatch"
    );
    assert!(
        rng_states.len() >= ni,
        "poisson_fused_hidden_step: rng_states length {} < ni {}",
        rng_states.len(),
        ni
    );
    assert!(
        rates.len() >= ni,
        "poisson_fused_hidden_step: rates length {} < ni {}",
        rates.len(),
        ni
    );
    assert!(
        input_spikes_out.len() >= ni,
        "poisson_fused_hidden_step: input_spikes_out length {} < ni {}",
        input_spikes_out.len(),
        ni
    );
    assert!(
        nh.is_multiple_of(8),
        "poisson_fused_hidden_step: nh must be multiple of 8, got {}",
        nh
    );
    unsafe {
        fused::poisson_fused_hidden_step(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            trace.as_mut_ptr(),
            spikes.as_mut_ptr(),
            refractory.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            weights.as_ptr(),
            rng_states.as_mut_ptr(),
            rates.as_ptr(),
            input_spikes_out.as_mut_ptr(),
            dt,
            trace_decay,
            tau_ref,
            ni,
            nh,
            active_buf,
        )
    }
}

/// Morris-Lecar SIMD step for n neurons.
pub fn morris_lecar_step(
    v: &mut [f32],
    w: &mut [f32],
    i_in: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    dt: f32,
) {
    let n = v.len();
    assert_eq!(w.len(), n, "morris_lecar_step: w length mismatch");
    assert_eq!(i_in.len(), n, "morris_lecar_step: i_in length mismatch");
    assert_eq!(spikes.len(), n, "morris_lecar_step: spikes length mismatch");
    assert_eq!(params.len(), n * 16, "morris_lecar_step: params length mismatch");
    unsafe {
        morris_lecar::morris_lecar_step(
            v.as_mut_ptr(),
            w.as_mut_ptr(),
            i_in.as_mut_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            n,
            dt,
        )
    }
}

/// NMDA Synaptic Channel with Mg²⁺ block.
pub fn nmda_mg_block_step(
    v: &[f32],
    g_act: &[f32],
    i_out: &mut [f32],
    params: &[f32],
) {
    let n = v.len();
    assert_eq!(g_act.len(), n, "nmda_mg_block_step: g_act length mismatch");
    assert_eq!(i_out.len(), n, "nmda_mg_block_step: i_out length mismatch");
    assert_eq!(params.len(), n * 4, "nmda_mg_block_step: params length mismatch");
    unsafe {
        nmda::nmda_mg_block_step(
            v.as_ptr(),
            g_act.as_ptr(),
            i_out.as_mut_ptr(),
            params.as_ptr(),
            n,
        )
    }
}

/// Tsodyks-Markram Short-Term Plasticity (STP).
pub fn tsodyks_markram_step(
    x: &mut [f32],
    u: &mut [f32],
    spikes: &[i32],
    w_eff: &mut [f32],
    params: &[f32],
    dt: f32,
) {
    let n = x.len();
    assert_eq!(u.len(), n, "tsodyks_markram_step: u length mismatch");
    assert_eq!(spikes.len(), n, "tsodyks_markram_step: spikes length mismatch");
    assert_eq!(w_eff.len(), n, "tsodyks_markram_step: w_eff length mismatch");
    assert_eq!(params.len(), n * 4, "tsodyks_markram_step: params length mismatch");
    unsafe {
        tsodyks_markram::tsodyks_markram_step(
            x.as_mut_ptr(),
            u.as_mut_ptr(),
            spikes.as_ptr(),
            w_eff.as_mut_ptr(),
            params.as_ptr(),
            n,
            dt,
        )
    }
}
