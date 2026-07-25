//! Unit tests for kuramoto module — Kuramoto, T3, decoders, interference, coherence.

use taas_checkmate::kuramoto;

fn make_padded(n: usize) -> usize {
    (n + 7) / 8 * 8
}

// ─── Kuramoto step ───────────────────────────────────────

#[test]
fn test_kuramoto_dense_basic() {
    let n = make_padded(8);
    let mut phases = vec![0.0f32; n];
    let omegas = vec![1.0f32; n];
    let mut coupling = vec![0.0f32; n * n];
    // Simple chain coupling
    for i in 0..n {
        if i > 0 {
            coupling[i * n + (i - 1)] = 0.5;
        }
        if i + 1 < n {
            coupling[i * n + (i + 1)] = 0.5;
        }
    }
    let dt = 0.01;

    kuramoto::kuramoto_step_dense(
        &mut phases,
        &omegas,
        &coupling,
        &mut vec![0.0f32; n],
        &mut vec![0.0f32; n],
        &mut vec![0.0f32; n],
        dt,
    );

    // Phases should advance
    for &p in &phases {
        assert!(p > 0.0, "Phase should advance from 0, got {}", p);
    }
}

#[test]
fn test_kuramoto_dense_sync() {
    let n = make_padded(8);
    let mut phases = vec![0.0f32; n];
    let omegas = vec![1.0f32; n];
    let mut coupling = vec![0.0f32; n * n];
    // All-to-all coupling → synchronize
    for i in 0..n {
        for j in 0..n {
            if i != j {
                coupling[i * n + j] = 1.0 / n as f32;
            }
        }
    }
    let dt = 0.5;

    for _ in 0..100 {
        kuramoto::kuramoto_step_dense(
            &mut phases,
            &omegas,
            &coupling,
            &mut vec![0.0f32; n],
            &mut vec![0.0f32; n],
            &mut vec![0.0f32; n],
            dt,
        );
    }

    // All phases should be close (synchronized)
    let p0 = phases[0];
    for &p in &phases {
        let diff = (p - p0).abs();
        assert!(
            diff < 0.5,
            "Kuramoto sync: phases should be close, diff={}",
            diff
        );
    }
}

// ─── Interference ────────────────────────────────────────

#[test]
fn test_interference_basic() {
    let na = 8;
    let nb = 8;
    let phases_a = vec![0.0f32; na];
    let phases_b = vec![std::f32::consts::FRAC_PI_2; nb]; // 90° apart
    let mut output = vec![0.0f32; na * nb];

    kuramoto::interference(&mut output, &phases_a, &phases_b);

    // interference = (1 + cos(θ_a - θ_b)) / 2, with Taylor approximation error ~0.0004
    for &o in &output {
        assert!(
            (o - 0.5).abs() < 1e-3,
            "interference(-π/2) ≈ 0.5, got {}",
            o
        );
    }
}

#[test]
fn test_interference_mean_zero() {
    let n = make_padded(8);
    let phases_a = vec![0.0f32; n];
    let phases_b = vec![0.0f32; n]; // same phase
    let mut output = vec![0.0f32; 1]; // ASM writes to output[0] only (scalar mean)

    kuramoto::interference_mean(&mut output, &phases_a, &phases_b);

    // interference_mean = mean((1 + cos(0)) / 2) = 1.0
    assert!(
        (output[0] - 1.0).abs() < 1e-5,
        "Same phase interference mean=1.0, got {}",
        output[0]
    );
}

// ─── Phase Coherence ─────────────────────────────────────

#[test]
fn test_phase_coherence() {
    let n = make_padded(8);
    let mut output = vec![0.0f32; 1];
    let sin_buf = vec![0.0f32; n]; // sin(0) = 0
    let cos_buf = vec![1.0f32; n]; // cos(0) = 1
    let amplitudes = vec![1.0f32; n]; // all unit amplitude

    kuramoto::phase_coherence(&mut output, &sin_buf, &cos_buf, &amplitudes);

    // All phases = 0 → R = 1.0
    assert!(
        (output[0] - 1.0).abs() < 1e-5,
        "All-in-phase coherence=1, got {}",
        output[0]
    );
}

#[test]
fn test_phase_coherence_zero() {
    let n = make_padded(8);
    let mut output = vec![0.0f32; 1];
    // Half at 0°, half at 180° → cancel out
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![1.0f32; n];
    for i in n / 2..n {
        sin_buf[i] = 0.0;
        cos_buf[i] = -1.0; // cos(π) = -1
    }
    let amplitudes = vec![1.0f32; n];

    kuramoto::phase_coherence(&mut output, &sin_buf, &cos_buf, &amplitudes);

    assert!(
        (output[0] - 0.0).abs() < 1e-5,
        "Anti-phase coherence=0, got {}",
        output[0]
    );
}

// ─── Band Energy ─────────────────────────────────────────

#[test]
fn test_band_energy() {
    let n = make_padded(8);
    let n_bands = 4;
    let mut energies = vec![0.0f32; n_bands];
    let amplitudes = vec![1.0f32; n];
    let omegas = vec![1.0f32, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0];
    let band_edges = vec![0.0, 4.0, 8.0, 12.0, 50.0];

    kuramoto::band_energy(&mut energies, &amplitudes, &omegas, &band_edges);

    // omega 1,2,3 → band 0 (0-4); omega 5 → band 1 (4-8); etc.
    assert!(
        (energies[0] - 3.0).abs() < 1e-5,
        "Band 0 expected 3, got {}",
        energies[0]
    );
    assert!(
        (energies[1] - 1.0).abs() < 1e-5,
        "Band 1 expected 1, got {}",
        energies[1]
    );
}

// ─── Lazy Sin/Cos ────────────────────────────────────────

#[test]
fn test_lazy_sincos_update() {
    let n = make_padded(8);
    let mut sin_out = vec![0.0f32; n];
    let mut cos_out = vec![0.0f32; n];
    let phases = vec![std::f32::consts::FRAC_PI_2; n];
    let mut phases_prev = vec![0.0f32; n];
    let eps = 0.01;

    kuramoto::lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut phases_prev, eps);

    // Taylor series precision: sin(π/2) ≈ 0.99984, cos(π/2) ≈ ±0.00089
    for i in 0..n {
        assert!(
            (sin_out[i] - 1.0).abs() < 1e-3,
            "sin(π/2) ≈ 1.0, got {}",
            sin_out[i]
        );
        assert!(cos_out[i].abs() < 1e-3, "cos(π/2) ≈ 0, got {}", cos_out[i]);
        assert!(
            (phases_prev[i] - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "phases_prev should update, got {}",
            phases_prev[i]
        );
    }
}

#[test]
fn test_lazy_sincos_skip_small_change() {
    let n = make_padded(8);
    let mut sin_out = vec![42.0f32; n]; // non-zero, shouldn't change
    let mut cos_out = vec![42.0f32; n];
    let phases = vec![0.0001f32; n]; // very small change from prev
    let mut phases_prev = vec![0.0f32; n];
    let eps = 0.01;

    kuramoto::lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut phases_prev, eps);

    // Change is < eps, so sin_out/cos_out should NOT be updated
    for i in 0..n {
        assert!(
            (sin_out[i] - 42.0).abs() < 1e-5,
            "Lazy should skip, sin={}",
            sin_out[i]
        );
        assert!(
            (cos_out[i] - 42.0).abs() < 1e-5,
            "Lazy should skip, cos={}",
            cos_out[i]
        );
    }
}

// ─── Batch Coherence Sweep ──────────────────────────────

#[test]
fn test_batch_coherence_sweep() {
    let n_areas = 2;
    let n_elem = 8; // must be multiple of 8 for AVX
    let mut output = vec![0.0f32; n_areas * n_areas];
    let areas = vec![1.0f32; n_areas * n_elem];
    let areas_sin = vec![0.0f32; n_areas * n_elem];
    let areas_cos = vec![1.0f32; n_areas * n_elem]; // all in phase

    kuramoto::batch_coherence_sweep(&mut output, &areas, &areas_sin, &areas_cos, n_areas, n_elem);

    // All in phase → R = 1.0
    for &o in &output {
        assert!((o - 1.0).abs() < 1e-5, "Coherence sweep R=1, got {}", o);
    }
}
