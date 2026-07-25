//! Unit tests for io module — safety_clamp, gates, encoders, decoders, taps.

use taas_checkmate::io;

fn make_padded(n: usize) -> usize {
    (n + 7) / 8 * 8
}

// ─── Safety Clamp ────────────────────────────────────────

#[test]
fn test_safety_clamp_basic() {
    let n = make_padded(16);
    let mut data = vec![0.0f32; n];
    data[0] = -200.0;
    data[1] = 200.0;
    data[2] = 50.0;

    io::safety_clamp(&mut data, -100.0, 100.0);

    assert!(
        (data[0] - (-100.0)).abs() < 1e-5,
        "Clamp min failed: {}",
        data[0]
    );
    assert!(
        (data[1] - 100.0).abs() < 1e-5,
        "Clamp max failed: {}",
        data[1]
    );
    assert!(
        (data[2] - 50.0).abs() < 1e-5,
        "Should not clamp middle: {}",
        data[2]
    );
}

#[test]
fn test_safety_clamp_noop() {
    let n = make_padded(8);
    let mut data = vec![5.0f32; n];
    io::safety_clamp(&mut data, -10.0, 10.0);
    for &v in &data {
        assert!((v - 5.0).abs() < 1e-5, "Clamp should be no-op: {}", v);
    }
}

// ─── Stochastic Encoder ──────────────────────────────────

#[test]
fn test_stochastic_encoder_strong_input() {
    let n = make_padded(8);
    let mut target = vec![0.0f32; n];
    let inputs = vec![100.0f32; n]; // very strong → always fire
    let noise = vec![0.0f32; n]; // no noise
    let threshold = 0.5;
    let spike_val = 40.0;

    io::stochastic_encoder(&mut target, &inputs, &noise, threshold, spike_val);

    for &t in &target {
        assert!(
            (t - spike_val).abs() < 1e-5,
            "Strong input should spike, got {}",
            t
        );
    }
}

#[test]
fn test_stochastic_encoder_weak_input() {
    let n = make_padded(8);
    let mut target = vec![0.0f32; n];
    let inputs = vec![0.0f32; n]; // zero input → never fire
    let noise = vec![0.0f32; n];
    let threshold = 0.5;
    let spike_val = 40.0;

    io::stochastic_encoder(&mut target, &inputs, &noise, threshold, spike_val);

    for &t in &target {
        assert!(
            (t - 0.0).abs() < 1e-5,
            "No input should not spike, got {}",
            t
        );
    }
}

// ─── T3 Drive Encoder ────────────────────────────────────

#[test]
fn test_t3_drive_encoder() {
    let n = make_padded(8);
    let mut drives = vec![0.0f32; n];
    let inputs = vec![2.0f32; n];
    let gain = 3.0;

    io::t3_drive_encoder(&mut drives, &inputs, gain);

    for &d in &drives {
        assert!(
            (d - 6.0).abs() < 1e-5,
            "drive = gain * input = 6.0, got {}",
            d
        );
    }
}

// ─── Thalamic Gate ───────────────────────────────────────

#[test]
fn test_thalamic_gate() {
    let n = make_padded(8);
    let mut target = vec![0.0f32; n];
    let source = vec![10.0f32; n]; // source current
    let gate = vec![1.0f32; n]; // above threshold → pass
    let threshold = 0.0;
    let gain = 1.0;

    io::thalamic_gate(&mut target, &source, &gate, threshold, gain);

    for &t in &target {
        assert!(t > 0.0, "Gate open should pass current, got {}", t);
    }
}

#[test]
fn test_thalamic_gate_closed() {
    let n = make_padded(8);
    let mut target = vec![0.0f32; n];
    let source = vec![10.0f32; n];
    let gate = vec![-1.0f32; n]; // below threshold → sigmoid attenuation
    let threshold = 0.0;
    let gain = 1.0;

    io::thalamic_gate(&mut target, &source, &gate, threshold, gain);

    // sigmoid(1 * (-1 - 0)) = 1/(1+e) ≈ 0.2689
    // target = 10 * 0.2689 ≈ 2.689 (not exactly 0 — sigmoid doesn't fully block)
    for &t in &target {
        assert!(
            (t - 2.689).abs() < 1e-2,
            "Gate should attenuate to ~2.689, got {}",
            t
        );
    }
}

// ─── Sigmoid Gate ────────────────────────────────────────

#[test]
fn test_sigmoid_gate() {
    let n = make_padded(8);
    let mut output = vec![0.0f32; n];
    let input = vec![1.0f32; n];
    let gain = 5.0;
    let threshold = 0.0;

    io::sigmoid_gate(&mut output, &input, gain, threshold);

    // sigmoid(5 * 1) ≈ 0.993
    for &o in &output {
        assert!(o > 0.9 && o < 1.0, "sigmoid(5) ≈ 0.993, got {}", o);
    }
}

// ─── Phase Gate ──────────────────────────────────────────

#[test]
fn test_phase_gate_apply() {
    let n = make_padded(8);
    let mut output = vec![0.0f32; n];
    let input = vec![5.0f32; n];
    let r = vec![0.8f32; n];
    let gain = 10.0;
    let threshold = 0.3;

    io::phase_gate_apply(&mut output, &input, &r, gain, threshold);

    // r (0.8) >> threshold (0.3) → sigmoid close to 1 → output ≈ input
    for &o in &output {
        assert!(
            (o - 5.0).abs() < 0.5,
            "Phase gate should pass ~5.0, got {}",
            o
        );
    }
}

// ─── Hebbian Update ──────────────────────────────────────

#[test]
fn test_hebbian_update() {
    let n = 16;
    let mut weights = vec![0.5f32; n];
    let cos_diff = vec![0.5f32; n];
    let lr = 0.01;
    let decay = 0.1;
    let dt = 0.01;

    io::hebbian_update(&mut weights, &cos_diff, lr, decay, dt);

    // delta = lr * cos_diff - decay * w * dt
    // = 0.01 * 0.5 - 0.1 * 0.5 * 0.01 = 0.005 - 0.0005 = 0.0045
    for &w in &weights {
        // w += (cos_diff - decay*w) * lr*dt
        // = 0.5 + (0.5 - 0.1*0.5) * 0.01*0.01
        // = 0.5 + 0.0045 = 0.5045
        assert!(
            (w - 0.5045).abs() < 1e-5,
            "Hebbian w should be ~0.5045, got {}",
            w
        );
    }
}

// ─── Echo Tap ────────────────────────────────────────────

#[test]
fn test_echo_tap() {
    let n_osc = 8;
    let delay_len = 10;
    // Buffer layout: n_osc × delay_len (row-major: each oscillator row has delay_len entries)
    // Kernel reads: out[i] = buf[i * delay_len + delay_idx]
    let mut buf = vec![0.0f32; n_osc * delay_len];
    for osc in 0..n_osc {
        for t in 0..delay_len {
            buf[osc * delay_len + t] = (osc * 100 + t) as f32;
        }
    }

    let mut output = vec![0.0f32; n_osc];
    io::echo_tap(&mut output, &buf, n_osc, delay_len, 0);

    // delay_idx=0 → buf[i * delay_len + 0] for each osc i
    for osc in 0..n_osc {
        let expected = (osc * 100 + 0) as f32;
        assert!(
            (output[osc] - expected).abs() < 1e-5,
            "Echo tap osc {}: expected {}, got {}",
            osc,
            expected,
            output[osc]
        );
    }
}

#[test]
fn test_echo_tap_delay() {
    let n_osc = 8;
    let delay_len = 20;
    // Buffer layout: n_osc × delay_len (row-major)
    let mut buf = vec![0.0f32; n_osc * delay_len];
    for osc in 0..n_osc {
        for t in 0..delay_len {
            buf[osc * delay_len + t] = (osc * 100 + t) as f32;
        }
    }

    let mut output = vec![0.0f32; n_osc];
    io::echo_tap(&mut output, &buf, n_osc, delay_len, 5);

    // delay_idx=5 → buf[i * delay_len + 5] for each osc i
    for osc in 0..n_osc {
        let expected = (osc * 100 + 5) as f32;
        assert!(
            (output[osc] - expected).abs() < 1e-5,
            "Echo tap delay=5 osc {}: expected {}, got {}",
            osc,
            expected,
            output[osc]
        );
    }
}
