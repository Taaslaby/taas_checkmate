use super::*;

#[test]
fn nlms_train_basic() {
    let n = 32;
    let mut weights = vec![0.0f32; n];
    let mut bias = vec![0.0f32; 1];
    let states: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
    for _ in 0..100 {
        let target = 1.0;
        let pred: f32 = weights
            .iter()
            .zip(states.iter())
            .map(|(w, s)| w * s)
            .sum::<f32>()
            + bias[0];
        let err = target - pred;
        nlms_train(&mut weights, &mut bias, &states, err, 0.01, 1e-8);
    }
    assert!(
        weights.iter().any(|&w| w.abs() > 0.001),
        "NLMS weights should update"
    );
}

#[test]
#[should_panic(expected = "finite and > 0")]
fn kalman_update_rejects_non_positive_innovation() {
    kalman_update(&mut [], &mut [], &[], 0.0, 0.0);
}

#[test]
#[should_panic(expected = "finite and > 0")]
fn kalman_update_rejects_non_finite_innovation() {
    kalman_update(&mut [], &mut [], &[], 0.0, f32::NAN);
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
        let s: f32 = states
            .iter()
            .zip(cov.iter())
            .map(|(x, c)| c * x * x)
            .sum::<f32>()
            + 0.1;
        kalman_update(&mut weights, &mut cov, &states, err, s);
    }
    assert!(
        weights.iter().any(|&w| w.abs() > 0.001),
        "Kalman weights should update"
    );
    assert!(cov.iter().all(|&c| c < 1.0), "Covariance should decrease");
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
    for o in 0..n_out {
        let expected: f32 = weights[o * n_in..(o + 1) * n_in].iter().sum();
        assert!((outputs[o] - expected).abs() < 1e-4, "complex_decoder[{o}]");
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
        assert!(
            (output[i] - 1.0).abs() < 1e-6,
            "ring_synthesis[{i}]: {}",
            output[i]
        );
    }
}

#[test]
fn t3_pst_features_and_pst_decode_smoke() {
    let p1 = vec![0.0f32, 0.5, 1.0];
    let p2 = vec![0.2f32, 0.7];
    let mut os = vec![0.0f32; 6];
    let mut oc = vec![0.0f32; 6];
    t3_pst_features(&mut os, &mut oc, &p1, &p2);
    let d0: f32 = 0.0f32 - 0.2f32;
    assert!((os[0] - d0.sin()).abs() < 1e-5, "pst sin");
    assert!((oc[0] - d0.cos()).abs() < 1e-5, "pst cos");
}

#[test]
fn lms_and_adam_train_smoke() {
    let n = 16;
    let mut w = vec![0.0f32; n];
    let mut b = vec![0.0f32; 1];
    let states: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    t3_lms_train(&mut w, &mut b, &states, 0.5, 0.01);
    assert!(
        w.iter().any(|&x| x.abs() > 0.001),
        "LMS should update weights"
    );

    let mut wa = vec![0.0f32; n];
    let mut m = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    adam_train(
        &mut wa, &mut m, &mut v, &states, 0.5, 0.01, 0.9, 0.999, 0.9, 0.999, 1e-8,
    );
    assert!(
        wa.iter().any(|&x| x.abs() > 0.001),
        "Adam should update weights"
    );
}
