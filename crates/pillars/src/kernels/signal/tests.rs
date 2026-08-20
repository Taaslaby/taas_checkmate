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
