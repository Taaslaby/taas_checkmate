use super::*;

#[test]
fn poisson_zero_rate_no_spikes() {
    let mut gen = PoissonGenerator::new(16, 1.0, 42);
    gen.set_rates(&[0.0f32; 16]);
    let mut spikes = vec![0i32; 16];
    for _ in 0..1000 {
        gen.step(&mut spikes);
    }
    assert!(spikes.iter().all(|&s| s == 0), "0 Hz rate must not spike");
}

#[test]
fn poisson_high_rate_spikes_sometimes() {
    let mut gen = PoissonGenerator::new(16, 1.0, 42);
    gen.set_rates(&[100.0f32; 16]); // p = 100·1.0 clamped to 1.0
    let mut spikes = vec![0i32; 16];
    let mut total = 0;
    for _ in 0..100 {
        gen.step(&mut spikes);
        total += spikes.iter().sum::<i32>();
    }
    assert!(total > 0, "high rate should spike");
}

#[test]
fn poisson_rates_kept_in_padded_bounds() {
    let gen = PoissonGenerator::new(10, 0.5, 1);
    assert_eq!(gen.n_neurons(), 16, "n padded to multiple of 8");
}

#[test]
fn bsa_encodes_step_signal() {
    let signal = vec![1.0f32; 8];
    let kernel = bsa_kernel_hamming(4);
    let mut spikes = vec![0i32; 8];
    let mut recon = vec![0.0f32; 8];
    let count = bsa_encode(&mut spikes, &signal, &kernel, &mut recon, 0.5);
    assert!(count > 0, "step signal should produce spikes");
    // Reconstruction should track the signal (last sample near 1)
    assert!(
        (recon[7] - signal[7]).abs() < 0.6,
        "recon should approach signal"
    );
}

#[test]
fn bsa_kernel_normalized() {
    let k = bsa_kernel_hamming(8);
    let sum: f32 = k.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "kernel should be normalized, sum={sum}"
    );
}

#[test]
fn ttfs_higher_value_spikes_earlier() {
    let mut enc = TtfsEncoder::new(2, 10);
    let values = vec![1.0f32, 0.0];
    let mut spikes = vec![0i32; 2];
    let mut fired_early = false;
    for _ in 0..12 {
        let any = enc.encode_step(&values, &mut spikes);
        if any && spikes[0] == 1 && spikes[1] == 0 {
            fired_early = true;
        }
    }
    assert!(fired_early, "value=1 should spike before value=0");
}

#[test]
fn phase_encode_basic() {
    let mut spikes = vec![0i32; 4];
    let mut phases = vec![0.0f32; 4];
    let cos_phases = vec![0.5f32; 4];
    let signal = vec![0.1f32; 4];
    let count = phase_encode(&mut spikes, &mut phases, &cos_phases, &signal, 0.3, 1.0);
    assert_eq!(count, 4, "cos=0.5 > 0.3 → all spike");
    assert!(
        (phases[0] - 0.1).abs() < 1e-6,
        "phase should advance by signal·gain"
    );
}

#[test]
fn pwm_encoder_threshold_fires() {
    let mut enc = PwmEncoder::new(4, 10, 30.0);
    let mut out = vec![0.0f32; 4];
    // sensors above any threshold in [0,1) always fire
    enc.encode(&[2.0f32; 4], &mut out);
    for &o in &out {
        assert_eq!(o, 30.0, "sensor=2.0 > threshold → spike_current");
    }
}

#[test]
fn pwm_encoder_zero_never_fires() {
    let mut enc = PwmEncoder::new(4, 10, 30.0);
    let mut out = vec![0.0f32; 4];
    // sensors of 0.0 are never above a threshold in [0,1)
    enc.encode(&[0.0f32; 4], &mut out);
    for &o in &out {
        assert_eq!(o, 0.0, "sensor=0.0 < threshold → 0");
    }
}

#[test]
fn pwm_encoder_tick_wraps() {
    let mut enc = PwmEncoder::new(1, 3, 30.0);
    let mut out = vec![0.0f32; 1];
    // n=1 → shift=0.0, period=3 → пороги 0, 1/3, 2/3; sensor=0.4
    // бьёт 0 и 1/3, но не 2/3 → частичная скважность
    let mut fires = 0;
    for _ in 0..6 {
        enc.encode(&[0.4f32], &mut out);
        fires += (out[0] > 0.0) as usize;
    }
    assert!(
        fires > 0 && fires < 6,
        "PWM duty should be partial, got {fires}"
    );
}

#[test]
fn linear_decoder_dot_product() {
    let mut dec = LinearDecoder::new(3, 2);
    dec.weights_mut()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    dec.bias_mut().copy_from_slice(&[0.5, -0.5]);
    let states = [1.0f32, 1.0, 1.0];
    let mut out = vec![0.0f32; 2];
    dec.decode(&states, &mut out);
    assert!(
        (out[0] - 6.5).abs() < 1e-5,
        "out[0] = 1+2+3+0.5 = 6.5, got {}",
        out[0]
    );
    assert!(
        (out[1] - 14.5).abs() < 1e-5,
        "out[1] = 4+5+6−0.5 = 14.5, got {}",
        out[1]
    );
}
