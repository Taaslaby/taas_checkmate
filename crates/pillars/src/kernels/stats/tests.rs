use super::*;

#[test]
fn firing_rates_basic() {
    // 2 neurons, 10 steps, neuron 0 spikes every step, neuron 1 never.
    let mut buf = vec![0i32; 20];
    for t in 0..10 {
        buf[t * 2] = 1;
    }
    let rates = firing_rates(&buf, 2, 10, 1.0);
    assert!(
        (rates[0] - 1.0).abs() < 1e-5,
        "neuron 0: 1 spike/ms → 1 Hz/ms"
    );
    assert_eq!(rates[1], 0.0, "neuron 1 silent");
}

#[test]
fn compute_isi_and_cvs_basic() {
    // Regular firing every 10 steps, dt=0.001s (1ms) → ISI = 10·0.001·1000 = 10ms, CV ≈ 0.
    let mut buf = vec![0i32; 100];
    for t in (0..100).step_by(10) {
        buf[t] = 1;
    }
    let isi = compute_isi(&buf[..100], 0.001);
    assert!(
        (isi[0] - 10.0).abs() < 1e-3,
        "regular ISI should be 10ms, got {}",
        isi[0]
    );
    let cvs = compute_cvs(&buf, 1, 100, 0.001);
    assert!(cvs[0] < 0.05, "regular firing → CV ≈ 0, got {}", cvs[0]);
}

#[test]
fn statistics_use_available_frames_when_request_is_longer() {
    // Only two frames are available although callers request ten. The
    // denominator must be two, and CV must not index past the buffer.
    let buf = [1i32, 0, 1, 0];
    let rates = firing_rates(&buf, 2, 10, 1.0);
    assert_eq!(rates, vec![1.0, 0.0]);
    let cvs = compute_cvs(&buf, 2, 10, 1.0);
    assert_eq!(cvs.len(), 2);
}

#[test]
fn population_rate_decode_empty_population_is_safe() {
    let mut rates = [1.0f32];
    population_rate_decode(&mut rates, &[], 0, 10);
    assert_eq!(rates, [1.0]);
}

#[test]
fn population_rate_decode_basic() {
    let mut buf = vec![0i32; 10];
    for t in 0..5 {
        buf[t] = 1;
    }
    let mut rates = vec![0.0f32; 1];
    population_rate_decode(&mut rates, &buf, 1, 10);
    assert!((rates[0] - 0.5).abs() < 1e-5, "5/10 window → 0.5");
}

#[test]
fn spike_raster_roundtrip() {
    let mut raster = SpikeRaster::new();
    let spikes = [1i32, 0, 0, 1];
    raster.record(0, &spikes, 4);
    raster.record(1, &spikes, 4);
    let (dense, n_steps) = raster.to_dense(4);
    assert_eq!(n_steps, 2);
    assert_eq!(dense[0], 1);
    assert_eq!(dense[1], 0);
    assert_eq!(dense[4], 1);
    let csv = raster.to_csv(0.5);
    assert!(csv.starts_with("time_ms,neuron_id\n"), "csv header");
    assert!(csv.contains("0.00,0"), "csv first event");
}
