//! Pure Rust SIMD kernels — replaces all ASM and FFI.
//! Use the [`safe`] module for safe slice-based wrappers.
//! Raw `unsafe` functions are available via internal submodules for crate-internal use.

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::manual_clamp)]

pub mod attention;
pub mod decode;
pub mod measure;
pub mod neuron;
pub mod phase;
pub mod signal;
pub mod safe;
pub mod io;
pub mod vnc;

// Re-export safe wrappers as the primary public API
pub use safe::*;
// Re-export safe utility functions from neuron module
pub use neuron::{fast_exp, fast_exp_simd, fast_sincos_simd};

/// Wrap phase to [-π, π)
#[inline(always)]
pub fn wrap_phase(p: f32) -> f32 {
    let tau = core::f32::consts::TAU;
    p - tau * (p / tau).round()
}

/// Clamp amplitude with floor (совпадает с TriOscillator::step clamp)
#[inline]
pub fn clamp_amp(a: f32) -> f32 {
    a.max(0.001).min(10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── signal tests ──

    #[test]
    fn safety_clamp_basic() {
        let mut data = vec![-10.0, -5.0, 0.0, 5.0, 10.0, -3.0, 3.0, 100.0, 0.5];
        safety_clamp(&mut data, -1.0, 1.0);
        for &v in &data { assert!(v >= -1.0 && v <= 1.0, "safety_clamp: {v} out of range"); }
    }

    #[test]
    fn sigmoid_gate_basic() {
        let n = 16;
        let input: Vec<f32> = (0..n).map(|i| i as f32 - 8.0).collect();
        let mut output = vec![0.0f32; n];
        sigmoid_gate(&mut output, &input, 1.0, 0.0);
        // At x=0, sigmoid(0) = 0.5
        let mid = output[8];
        assert!((mid - 0.5).abs() < 0.05, "sigmoid at 0 should be ~0.5, got {mid}");
        // All outputs in (0, 1)
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
            assert!((drives[i] - inputs[i] * 10.0).abs() < 1e-6, "drive_encoder: {i}");
        }
    }

    #[test]
    fn relu_simd_basic() {
        let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -0.5, 2.0];
        relu_simd(&mut data);
        let expected = [0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 0.0, 2.0];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "relu[{i}]: got {got}, expected {exp}");
        }
    }

    #[test]
    fn add_simd_basic() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        add_simd(&mut a, &b);
        for i in 0..8 { assert!((a[i] - (i as f32 + 1.5)).abs() < 1e-6, "add[{i}]"); }
    }

    #[test]
    fn echo_tap_basic() {
        let n_osc = 4;
        let delay_len = 8;
        let mut buffer = vec![0.0f32; n_osc * delay_len];
        // Write known values at delay index 3
        for i in 0..n_osc { buffer[i * delay_len + 3] = (i + 1) as f32; }
        let mut output = vec![0.0f32; n_osc];
        echo_tap(&mut output, &buffer, delay_len, 3);
        for i in 0..n_osc { assert!((output[i] - (i + 1) as f32).abs() < 1e-6, "echo_tap[{i}]"); }
    }

    #[test]
    fn phase_gate_apply_basic() {
        let n = 16;
        let input = vec![1.0f32; n];
        let r_low = vec![0.0f32; n];   // below threshold → gate closed
        let r_high = vec![10.0f32; n]; // above threshold → gate open
        let mut out_low = vec![0.0f32; n];
        let mut out_high = vec![0.0f32; n];
        phase_gate_apply(&mut out_low, &input, &r_low, 4.0, 1.0);
        phase_gate_apply(&mut out_high, &input, &r_high, 4.0, 1.0);
        // Low r → output near 0, high r → output near 1
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

    // ── decode tests ──

    #[test]
    fn nlms_train_basic() {
        let n = 32;
        let mut weights = vec![0.0f32; n];
        let mut bias = vec![0.0f32; 1];
        let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        // Train for several steps — weights should diverge from 0
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
        // Cov should decrease
        assert!(cov.iter().all(|&c| c < 1.0), "Covariance should decrease");
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
        // Each output = sum(w) + bias
        for o in 0..n_out {
            let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum::<f32>() + biases[o];
            assert!((outputs[o] - expected).abs() < 1e-4, "linear_decoder[{o}]: got {}, expected {}", outputs[o], expected);
        }
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
        // output = sum(amp * cos * w) + bias = sum(w) since amp=cos=1
        for o in 0..n_out {
            let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum();
            assert!((outputs[o] - expected).abs() < 1e-4, "complex_decoder[{o}]: got {}, expected {}", outputs[o], expected);
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

    // ── phase tests ──

    #[test]
    fn kuramoto_prepare_basic() {
        let n = 16;
        let mut phases: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut cos_out = vec![0.0f32; n];
        let mut sin_out = vec![0.0f32; n];
        
        kuramoto_prepare(&mut phases, &mut cos_out, &mut sin_out);
        for i in 0..n {
            let expected_cos = phases[i].cos();
            let expected_sin = phases[i].sin();
            assert!((cos_out[i] - expected_cos).abs() < 1e-4, "cos[{i}]: got {}, expected {}", cos_out[i], expected_cos);
            assert!((sin_out[i] - expected_sin).abs() < 1e-4, "sin[{i}]: got {}, expected {}", sin_out[i], expected_sin);
        }
    }

    #[test]
    fn kuramoto_update_basic() {
        let n = 8;
        let mut phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        kuramoto_update(&mut phases, &omegas, &coupling, 0.01);
        // With omega=1, dt=0.01, no coupling: phases should be ~0.01
        for i in 0..n {
            assert!((phases[i] - 0.01).abs() < 1e-4, "kuramoto_update[{i}]: {}", phases[i]);
        }
    }

    #[test]
    fn t3_update_basic() {
        let n = 16;
        let mut phases = vec![0.0f32; n];
        let mut amps = vec![1.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        let drives = vec![0.0f32; n];
        t3_update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.01, 0.0, 1.0, 10.0);
        // Phases should advance
        assert!(phases[0].abs() > 0.001, "t3_update should advance phases");
        // Amplitudes should stay in range
        for &r in &amps {
            assert!(r >= 0.001 && r <= 10.0, "t3_update amp out of range: {r}");
        }
    }

    // ── attention tests ──

    #[test]
    fn xor_attention_identical_keys() {
        let n_bytes = 8;
        let n_keys = 4;
        let query = vec![0u8; n_bytes];
        let mut keys = vec![0xFFu8; n_keys * n_bytes];
        // First key identical to query
        for i in 0..n_bytes { keys[i] = 0x00; }
        let values = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut out = 0.0f32;
        xor_attention(&query, &keys, &values, &mut out);
        assert!((out - 1.0).abs() < 0.01, "identical key should give value 1.0, got {out}");
    }

    #[test]
    fn rdrand_fill_range() {
        let mut buf = vec![0.0f32; 1024];
        rdrand_fill(&mut buf);
        for &v in &buf {
            assert!(v >= 0.0 && v < 1.0, "rdrand_fill: {v} out of [0,1)");
        }
    }

    // ── neuron tests ──

    #[test]
    fn izhikevich_spikes_with_strong_current() {
        let n = 16;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..200 {
            izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5, &mut spikes, &c, &d, &mut refractory, 3.0, None);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "Expected spikes with I=30");
    }

    #[test]
    fn izhikevich_no_spike_inhibitory() {
        let n = 8;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let mut I = vec![-20.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..100 {
            izhikevich_step(&mut v, &mut u, &mut trace, &a, &b, &mut I, 1.0, 0.5, &mut spikes, &c, &d, &mut refractory, 3.0, None);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert_eq!(total_spikes, 0, "No spikes with I=-20");
    }

    #[test]
    fn lif_spike_with_strong_current() {
        let n = 16;
        let mut v = vec![-70.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 5];
        for i in 0..n {
            params[i] = 1.0;           // dt
            params[n + i] = 10.0;      // tau_m
            params[2*n + i] = -70.0;   // v_rest
            params[3*n + i] = -55.0;   // v_thresh
            params[4*n + i] = -75.0;   // v_reset
        }
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..100 {
            lif_step(&mut v, &mut I, &mut spikes, &params, &mut refractory, 2.0, None);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "LIF should spike with I=30");
    }

    #[test]
    fn stdp_update_basic() {
        let n = 16;
        let mut weights = vec![0.5f32; n];
        let traces_pre = vec![0.0f32; n];
        let traces_post = vec![1.0f32; n]; // strong post-synaptic trace
        stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, 0.0, 1.0);
        // w += lr_ltp * post - lr_ltd * pre = 0.5 + 0.1*1.0 - 0.1*0.0 = 0.6
        for i in 0..n {
            assert!((weights[i] - 0.6).abs() < 0.01, "stdp[{i}]: {}", weights[i]);
        }
    }

    #[test]
    fn stp_step_facilitation() {
        let n = 8;
        let mut x = vec![1.0f32; n];
        let mut u = vec![0.2f32; n];
        let mut i_out = vec![0.0f32; n];
        let pre_spikes = vec![1i32; n]; // all spike
        let tau_d = vec![200.0f32; n];
        let tau_f = vec![200.0f32; n];
        let u0 = vec![0.2f32; n];
        let a = vec![1.0f32; n];
        stp_step(&mut x, &mut u, &pre_spikes, &tau_d, &tau_f, &u0, &a, &mut i_out, 1.0);
        // After one spike: u should increase (facilitation), i_out should be positive
        assert!(u[0] > 0.2, "u should increase: {}", u[0]);
        assert!(i_out[0] > 0.0, "i_out should be positive: {}", i_out[0]);
    }

    #[test]
    fn dopamine_traces_step_basic() {
        let n = 8;
        let mut trace_pre = vec![0.0f32; n];
        let mut trace_post = vec![0.0f32; n];
        let mut eligibility = vec![0.0f32; n];
        let pre_spikes = vec![1i32; n];
        let post_spikes = vec![1i32; n];
        dopamine_traces_step(&mut trace_pre, &mut trace_post, &mut eligibility, &pre_spikes, &post_spikes, 1.0, 20.0, 20.0, 200.0, 0.01, 0.01);
        // Both traces should increase
        assert!(trace_pre[0] > 0.0, "trace_pre should increase");
        assert!(trace_post[0] > 0.0, "trace_post should increase");
    }

    #[test]
    fn dopamine_weight_update_basic() {
        let n = 8;
        let mut weights = vec![0.5f32; n];
        let eligibility = vec![1.0f32; n];
        dopamine_weight_update(&mut weights, &eligibility, 1.0, 0.1, 0.0, 1.0);
        // w += lr * dopamine * e = 0.5 + 0.1 * 1.0 * 1.0 = 0.6
        for i in 0..n {
            assert!((weights[i] - 0.6).abs() < 0.01, "dopamine_weight[{i}]: {}", weights[i]);
        }
    }

    #[test]
    fn delay_buffer_circular_basic() {
        let n = 8;
        let buffer: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; n];
        delay_buffer_circular(&mut output, &buffer, 3);
        // pos=3: output = [3,4,5,6,7,0,1,2]
        assert_eq!(output, vec![3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn delay_buffer_multi_tap_basic() {
        let buffer: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let delays = vec![0i32, 3, 7, 9];
        let mut output = vec![0.0f32; 4];
        delay_buffer_multi_tap(&mut output, &buffer, &delays);
        assert_eq!(output, vec![0.0, 3.0, 7.0, 9.0]);
    }

    // ── utility tests ──

    #[test]
    fn wrap_phase_basic() {
        let pi = std::f32::consts::PI;
        // wrap_phase: p - τ·round(p/τ), τ=2π. Result in [-π, π)
        assert!((wrap_phase(pi * 3.0) - (-pi)).abs() < 1e-4, "wrap 3π → -π");
        assert!((wrap_phase(-pi * 3.0) - pi).abs() < 1e-4, "wrap -3π → π");
        assert!(wrap_phase(0.0).abs() < 1e-4, "wrap 0");
        assert!((wrap_phase(pi * 0.5) - pi * 0.5).abs() < 1e-4, "wrap π/2");
        assert!((wrap_phase(pi * 5.0) - (-pi)).abs() < 1e-4, "wrap 5π → -π");
    }

    #[test]
    fn clamp_amp_basic() {
        assert!((clamp_amp(5.0) - 5.0).abs() < 1e-6);
        assert!((clamp_amp(0.0001) - 0.001).abs() < 1e-6);
        assert!((clamp_amp(100.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn fast_exp_accuracy() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5] {
            let approx = fast_exp(x);
            let exact = x.exp();
            let rel_err = ((approx - exact) / exact).abs();
            assert!(rel_err < 0.05, "fast_exp({x})={approx} vs {exact}, err={rel_err}");
        }
    }
}


