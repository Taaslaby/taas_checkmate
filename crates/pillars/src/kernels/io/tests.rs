use super::*;

// ─── Min-truncation contract ───────────────────────────────────────────────
//
// These wrappers exist to keep the **min-семантику** of the python ABI:
// shorter inputs truncate, never panic. The canon (signal/decode) asserts
// exact lengths — so the wrappers' only job is slicing first.

#[test]
fn stochastic_encoder_truncates_min() {
    let mut target = vec![9.0f32; 4];
    let inputs = [0.8f32; 5];
    let noise = [0.0f32; 5];
    // min(len) = 4: writes first 4, keeps 5th untouched, no panic.
    stochastic_encoder(&mut target, &inputs, &noise, 1.0, 1.0);
    assert_eq!(target, vec![0.0f32; 4], "threshold not exceeded → zero");
}

#[test]
fn stochastic_encoder_math() {
    let mut target = [9.0f32; 4];
    let inputs = [0.2f32, 0.2, 0.8, 0.8];
    let noise = [0.0f32, 0.5, 0.0, 0.5];
    stochastic_encoder(&mut target, &inputs, &noise, 0.6, 7.0);
    // input+noise > threshold → spike_val, else 0.
    assert_eq!(target, [0.0, 7.0, 7.0, 7.0]);
}

#[test]
fn t3_drive_encoder_math_and_truncation() {
    let mut drives = [1.0f32; 4];
    t3_drive_encoder(&mut drives, &[2.0, 3.0, 4.0], 0.5);
    assert_eq!(drives, [1.0, 1.5, 2.0, 1.0], "gain·input over min len");
}

#[test]
fn t3_complex_decoder_math() {
    // outputs = W·[A·cosθ] + bias, row-major W: rows = outputs.
    let mut out = [0.0f32; 2];
    let amps = [1.0f32, 2.0];
    let cos_buf = [0.5f32, 0.25];
    let weights = [2.0f32, 3.0, 1.0, 4.0]; // 2×2
    let biases = [0.5f32, -1.0];
    t3_complex_decoder(&mut out, &amps, &cos_buf, &weights, &biases, 2);
    assert!((out[0] - (0.5 + 1.0 * 0.5 * 2.0 + 2.0 * 0.25 * 3.0)).abs() < 1e-6);
    assert!((out[1] - (-1.0 + 1.0 * 0.5 * 1.0 + 2.0 * 0.25 * 4.0)).abs() < 1e-6);
}

#[test]
fn t3_pst_features_math() {
    // Exact-length contract of the canon — sizes must be ph1·ph2.
    let mut sin_out = vec![9.0f32; 6];
    let mut cos_out = vec![9.0f32; 6];
    let (p1, p2) = (vec![0.1f32, 0.2], vec![0.05f32, 0.15, 0.25]);
    t3_pst_features(&mut sin_out, &mut cos_out, &p1, &p2);
    for i in 0..2 {
        for j in 0..3 {
            let d = p1[i] - p2[j];
            let k = i * 3 + j;
            assert!((sin_out[k] - d.sin()).abs() < 1e-6);
            assert!((cos_out[k] - d.cos()).abs() < 1e-6);
        }
    }
}

#[test]
fn t3_tact_generator_math() {
    let mut pulses = [9.0f32; 4];
    t3_tact_generator(
        &mut pulses,
        &[0.2, 0.7, 0.7, 0.7],
        &[0.2, 0.9, 0.1, 0.9],
        &[0.2, 0.6, 0.6, 0.5],
        0.5,
    );
    // AND of all three phases > threshold.
    assert_eq!(pulses, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn t3_inverse_kuramoto_math() {
    // drive = gain·(r − r_prev)/dt − drive·decay
    let mut drive = [0.0f32; 1];
    t3_inverse_kuramoto(&mut drive, &[1.0], &[0.9], 0.0, 2.0, 1e-6, 0.1, 0.01);
    let expected = 2.0 * (1.0 - 0.9) / 0.01;
    assert!((drive[0] - expected).abs() < 1e-3);
}

#[test]
fn echo_tap_math_wraps_delay_idx() {
    // buffer layout [n_osc × delay_len]; delay_idx wraps mod delay_len.
    let mut out = [9.0f32; 2];
    let buf = [1.0, 2.0, 3.0, 4.0]; // osc0: [1,2], osc1: [3,4]
    echo_tap(&mut out, &buf, 2, 2, 3); // idx 3 % 2 = 1
    assert_eq!(out, [2.0, 4.0]);
    echo_tap(&mut out, &buf, 2, 2, 0);
    assert_eq!(out, [1.0, 3.0]);
}

#[test]
fn thalamic_gate_gates_by_membrane() {
    // gate = sigmoid(gain·(gate_v − threshold)); target = source_v · gate.
    let mut target = [0.0f32; 3];
    let source = [1.0f32; 3];
    let gate = [-10.0f32, 0.0, 10.0];
    thalamic_gate(&mut target, &source, &gate, 0.0, 1.0);
    assert!(target[0] < 0.01, "closed gate → ~0, got {}", target[0]);
    assert!(
        (target[1] - 0.5).abs() < 0.05,
        "at threshold gate ≈ 0.5, got {}",
        target[1]
    );
    assert!(target[2] > 0.95, "open gate → ~source, got {}", target[2]);
}

#[test]
fn sigmoid_gate_midpoint_at_threshold() {
    let mut out = [0.0f32; 3];
    sigmoid_gate(&mut out, &[-1.0, 0.0, 1.0], 2.0, 0.0);
    assert!(out[0] < 0.2, "below threshold → small, got {}", out[0]);
    assert!(
        (out[1] - 0.5).abs() < 0.05,
        "at threshold → 0.5, got {}",
        out[1]
    );
    assert!(out[2] > 0.8, "above threshold → large, got {}", out[2]);
}

#[test]
fn phase_gate_apply_math() {
    // out = input·sigmoid(gain·(r − threshold))
    let mut out = [0.0f32; 3];
    let input = [2.0f32; 3];
    let r = [-10.0f32, 0.0, 10.0];
    phase_gate_apply(&mut out, &input, &r, 1.0, 0.0);
    assert!(out[0] < 0.02);
    assert!((out[1] - 1.0).abs() < 0.1);
    assert!(out[2] > 1.98);
}

#[test]
fn hebbian_update_math() {
    let mut w = [1.0f32; 2];
    hebbian_update(&mut w, &[0.5, -0.5], 0.1, 0.01, 10.0);
    // w += lr·cos − decay·w·dt
    let e1 = 1.0 + 0.1 * 0.5 - 0.01 * 1.0 * 10.0;
    let e2 = 1.0 + 0.1 * (-0.5) - 0.01 * 1.0 * 10.0;
    assert!((w[0] - e1).abs() < 1e-4);
    assert!((w[1] - e2).abs() < 1e-4);
}

#[test]
fn safety_clamp_reexport_works() {
    let mut data = vec![9.0f32; 3];
    safety_clamp(&mut data, -1.0, 1.0);
    assert_eq!(data, vec![1.0; 3]);
}

#[test]
fn shorter_target_truncates_not_panics() {
    // Every min-adapter must accept shorter writes safely (python ABI).
    let mut out = [0.0f32; 1];
    stochastic_encoder(&mut out, &[0.5, 0.5], &[0.0, 0.0], 1.0, 1.0);
    t3_drive_encoder(&mut out, &[1.0, 2.0], 1.0);
    let r = [1.0f32];
    t3_inverse_kuramoto(&mut out, &r, &r, 1.0, 1.0, 1e-6, 0.0, 0.1);
    sigmoid_gate(&mut out, &[0.0, 1.0], 1.0, 0.0);
    phase_gate_apply(&mut out, &[1.0, 0.0], &r, 1.0, 0.0);
    hebbian_update(&mut out, &[0.0, 1.0], 0.1, 0.0, 1.0);
    echo_tap(&mut out, &[1.0, 2.0], 1, 2, 1);
    thalamic_gate(&mut out, &r, &r, 0.0, 1.0);
    assert_eq!(out.len(), 1, "truncation must shrink to the min length");
}