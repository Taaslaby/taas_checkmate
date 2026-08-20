use super::*;

#[test]
fn interference_mean_empty_is_zero() {
    let mut output = f32::NAN;
    interference_mean(&mut output, &[], &[]);
    assert_eq!(output, 0.0);
}

#[test]
fn interference_and_coherence_basic() {
    let n = 8;
    let a = vec![0.0f32; n]; // all in-phase
    let b = vec![0.0f32; n];
    let mut out = vec![0.0f32; n * n];
    interference(&mut out, &a, &b);
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "in-phase interference should be 1"
    );

    let sin_buf = vec![0.0f32; n];
    let cos_buf = vec![1.0f32; n];
    let amps = vec![1.0f32; n];
    let mut coherence = 0.0;
    phase_coherence(&mut coherence, &sin_buf, &cos_buf, &amps);
    assert!(
        (coherence - 1.0).abs() < 1e-4,
        "in-phase coherence should be 1"
    );
}

#[test]
fn band_energy_basic() {
    let amplitudes = vec![1.0f32; 6];
    let omegas = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let edges = vec![0.0, 2.0, 4.0, 6.0];
    let mut bands = vec![0.0f32; 3];
    band_energy(&mut bands, &amplitudes, &omegas, &edges);
    assert_eq!(bands, vec![2.0, 2.0, 2.0], "each band gets 2 oscillators");
}

#[test]
fn winding_and_lazy_sincos_smoke() {
    let n = 4;
    let a = vec![0.0f32; n];
    let b = vec![0.0f32; n];
    let mut w = vec![0.0f32; n * n];
    winding_matrix(&mut w, &a, &b);
    assert!((w[0] - 0.0).abs() < 1e-5, "zero phase diff → winding 0");

    let mut sin_out = vec![0.0f32; n];
    let mut cos_out = vec![0.0f32; n];
    let phases = vec![0.5f32; n];
    let mut prev = vec![0.0f32; n];
    lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut prev, 0.01);
    assert!(
        (sin_out[0] - 0.5f32.sin()).abs() < 1e-5,
        "lazy sin should recompute on change"
    );
    // No change → values stay
    let phases2 = vec![0.5f32; n];
    lazy_sincos_update(&mut sin_out, &mut cos_out, &phases2, &mut prev, 0.01);
    assert!(
        (sin_out[0] - 0.5f32.sin()).abs() < 1e-5,
        "no change → sin preserved"
    );
}

#[test]
fn spike_entropy_basic() {
    // Uniform distribution over 4 neurons → H = log2(4) = 2.0
    let counts = vec![1.0f32; 4];
    let (h, penalty) = spike_entropy(&counts, 4.0, 2.0);
    assert!((h - 2.0).abs() < 1e-4, "uniform H should be 2, got {h}");
    assert!(
        penalty < 1e-3,
        "penalty should be ~0 at target, got {penalty}"
    );
}

#[test]
fn batch_coherence_sweep_basic() {
    let n_elem = 4;
    let n_areas = 2;
    // Area 0: in-phase (cos=1). Area 1: also in-phase → coherence 1.
    let areas = vec![1.0f32; n_areas * n_elem];
    let sin_buf = vec![0.0f32; n_areas * n_elem];
    let cos_buf = vec![1.0f32; n_areas * n_elem];
    let mut out = vec![0.0f32; n_areas * n_areas];
    batch_coherence_sweep(&mut out, &areas, &sin_buf, &cos_buf, n_areas, n_elem);
    assert!((out[0] - 1.0).abs() < 1e-4, "self-coherence should be 1");
    assert!((out[1] - 1.0).abs() < 1e-4, "identical areas → coherence 1");
}
