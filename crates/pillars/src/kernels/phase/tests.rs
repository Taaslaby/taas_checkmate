use super::*;

#[test]
fn gco_step_normalizes_on_unit_circle() {
    let n = 16;
    let mut v: Vec<f32> = (0..n).map(|i| 0.3 + i as f32 * 0.1).collect();
    let mut u: Vec<f32> = (0..n).map(|i| -0.2 + i as f32 * 0.05).collect();
    let h_real = vec![1.0f32; n];
    let h_imag = vec![0.5f32; n];
    let i_ext = vec![0.2f32; n];
    let speed = vec![0.8f32; n];

    for _ in 0..1000 {
        gco_step(&mut v, &mut u, &h_real, &h_imag, &i_ext, &speed, 0.01);
    }
    for i in 0..n {
        let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
        assert!(
            (nrm - 1.0).abs() < 1e-4,
            "GCO[{i}] left the unit circle: |v+iu|={nrm}"
        );
        assert!(v[i].is_finite() && u[i].is_finite(), "GCO[{i}] non-finite");
    }
}

#[test]
fn gco_step_clamps_input() {
    let n = 8;
    let mut v = vec![0.5f32; n];
    let mut u = vec![0.0f32; n];
    let h_real = vec![0.0f32; n];
    let h_imag = vec![0.0f32; n];
    // Хьюг — без клампа осциллятор улетел бы; проверяем конечность
    let i_ext = vec![1e6f32; n];
    let speed = vec![0.0f32; n];
    for _ in 0..10 {
        gco_step(&mut v, &mut u, &h_real, &h_imag, &i_ext, &speed, 0.01);
    }
    for i in 0..n {
        assert!(v[i].is_finite() && u[i].is_finite(), "GCO[{i}] non-finite");
        let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
        assert!((nrm - 1.0).abs() < 1e-3, "GCO[{i}] off-circle: {nrm}");
    }
}

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
        assert!((cos_out[i] - expected_cos).abs() < 5e-4, "cos[{i}]");
        assert!((sin_out[i] - expected_sin).abs() < 5e-4, "sin[{i}]");
    }
}

#[test]
fn kuramoto_update_basic() {
    let n = 8;
    let mut phases = vec![0.0f32; n];
    let omegas = vec![1.0f32; n];
    let coupling = vec![0.0f32; n];
    kuramoto_update(&mut phases, &omegas, &coupling, 0.01);
    for i in 0..n {
        assert!(
            (phases[i] - 0.01).abs() < 1e-4,
            "kuramoto_update[{i}]: {}",
            phases[i]
        );
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
    t3_update(
        &mut phases,
        &mut amps,
        &omegas,
        &coupling,
        &drives,
        0.01,
        0.0,
        1.0,
        10.0,
    );
    assert!(phases[0].abs() > 0.001, "t3_update should advance phases");
    for &r in &amps {
        assert!(
            (0.001..=10.0).contains(&r),
            "t3_update amp out of range: {r}"
        );
    }
}

#[test]
fn kuramoto_viscous_equals_plain_when_eta_zero() {
    let n = 16;
    let mut p1: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut p2 = p1.clone();
    let omegas = vec![0.5f32; n];
    let coupling = vec![0.1f32; n];
    kuramoto_update(&mut p1, &omegas, &coupling, 0.01);
    kuramoto_update_viscous(&mut p2, &omegas, &coupling, 0.0, 0.01);
    for i in 0..n {
        assert!(
            (p1[i] - p2[i]).abs() < 1e-6,
            "eta=0 should match plain update: [{i}]"
        );
    }
}
