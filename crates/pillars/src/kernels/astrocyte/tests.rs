use super::*;

#[test]
fn test_astrocyte_new_and_reset() {
    let astro = AstrocyteState::new(8, 1000.0, 0.1);
    assert_eq!(astro.n, 8);
    assert_eq!(astro.n_padded, 8);
    assert!(astro.a.iter().all(|&v| v == 0.0));
    assert!(astro.ca.iter().all(|&v| (v - 0.1).abs() < 1e-6));
    assert!(astro.k_ext.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    assert!(astro.eaat_occ.iter().all(|&v| v == 0.0));
    assert!(astro.v_gap.iter().all(|&v| (v + 70.0).abs() < 1e-6));

    let mut astro = astro;
    astro.a[0] = 5.0;
    astro.k_ext[0] = 10.0;
    astro.reset();
    assert_eq!(astro.a[0], 0.0);
    assert!((astro.k_ext[0] - 4.0).abs() < 1e-6);
}

#[test]
fn test_astrocyte_a_slow_integration() {
    let mut astro = AstrocyteState::new(8, 1000.0, 0.1);
    let spikes = vec![1i32; 8];
    let gap_pairs = vec![];
    for _ in 0..10 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    assert!(astro.a[0] > 5.0, "A should accumulate: {}", astro.a[0]);

    let spikes = vec![0i32; 8];
    for _ in 0..10 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    assert!(astro.a[0] < 10.0, "A should decay: {}", astro.a[0]);
    assert!(astro.a[0] > 1.0, "A should still be >0: {}", astro.a[0]);
}

#[test]
fn test_astrocyte_ca_integration() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
    let spikes = vec![1i32; 4];
    let gap_pairs = vec![];
    for _ in 0..5 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    assert!(astro.ca[0] > 0.5, "Ca should rise: {}", astro.ca[0]);
}

#[test]
fn test_astrocyte_k_ext_integration() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
    astro.k_per_spike = 0.5;
    astro.tau_k = 100.0;
    let spikes = vec![1i32; 4];
    let gap_pairs = vec![];
    for _ in 0..5 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    assert!(astro.k_ext[0] > 4.5, "K should rise: {}", astro.k_ext[0]);
}

#[test]
fn test_astrocyte_gap_junction_and_ca_wave() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
    astro.g_gap = 0.2;
    let gap_pairs = vec![(0, 1), (1, 2), (2, 3)];
    astro.v_gap[0] = -30.0;
    let spikes = vec![1, 0, 0, 0];
    let empty_gap = vec![];
    for _ in 0..50 {
        astro.step(&spikes, &empty_gap, 1.0);
    }
    for _ in 0..50 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    assert!(astro.v_gap[1] > -70.0, "Gap wave should propagate");
    assert!(
        astro.ca[0] > 0.5,
        "Ca should rise from spike: {}",
        astro.ca[0]
    );
}

#[test]
fn test_astrocyte_compute_gain() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.5);
    astro.a[0] = 0.0;
    astro.a[1] = 1.0;
    astro.a[2] = 2.0;
    astro.ca[0] = 0.1;
    astro.ca[1] = 0.5;
    astro.ca[2] = 1.0;
    let gain = astro.compute_gain();
    assert!((gain[0] - 1.05).abs() < 1e-6, "gain[0]={}", gain[0]);
    assert!((gain[1] - 1.875).abs() < 1e-6, "gain[1]={}", gain[1]);
    assert!((gain[2] - 3.0).abs() < 1e-6, "gain[2]={}", gain[2]);
}

#[test]
fn test_astrocyte_modulate_v_thresh() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
    astro.k_ext[0] = 10.0;
    astro.k_sensitivity = 2.0;
    let mut v_thresh = vec![-50.0f32; 4];
    astro.modulate_v_thresh(&mut v_thresh);
    assert!(
        (v_thresh[0] - (-38.0)).abs() < 1e-6,
        "v_thresh[0]={}",
        v_thresh[0]
    );
    assert!(
        (v_thresh[1] - (-50.0)).abs() < 1e-6,
        "v_thresh[1]={}",
        v_thresh[1]
    );
}

#[test]
fn test_astrocyte_eaat_clearance() {
    let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
    let mut g_glucl = vec![1.0f32; 4];
    astro.eaat_occ[0] = 0.8;
    astro.eaat_strength = 0.5;
    astro.apply_eaat_clearance(&mut g_glucl, 1.0);
    assert!((g_glucl[0] - 0.6).abs() < 0.01, "g_glucl[0]={}", g_glucl[0]);
    assert!((g_glucl[1] - 1.0).abs() < 0.01, "g_glucl[1]={}", g_glucl[1]);
}

#[test]
fn test_astrocyte_apply_to_kinetics_integration() {
    let mut astro = AstrocyteState::new(8, 1000.0, 0.1);
    let mut kin = SynapticKinetics::new(8);
    kin.g_ampa[0] = 1.0;
    kin.g_glucl[1] = 0.5;

    let spikes = vec![1i32; 8];
    let gap_pairs = vec![];
    for _ in 0..50 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    astro.apply_to_kinetics(&mut kin, 1.0);

    assert!(
        kin.g_ampa[0] > 1.0,
        "AMPA should be amplified: {}",
        kin.g_ampa[0]
    );
    assert!(
        kin.g_glucl[1] >= 0.0,
        "GluCl should stay >= 0: {}",
        kin.g_glucl[1]
    );
}

#[test]
fn test_astrocyte_simd_vs_scalar_tail() {
    let mut astro = AstrocyteState::new(9, 1000.0, 0.1);
    let spikes = vec![1i32; 9];
    let gap_pairs = vec![];
    for _ in 0..100 {
        astro.step(&spikes, &gap_pairs, 1.0);
    }
    for (name, arr) in [
        ("a", &astro.a),
        ("ca", &astro.ca),
        ("k_ext", &astro.k_ext),
        ("eaat_occ", &astro.eaat_occ),
    ] {
        let diff = (arr[0] - arr[8]).abs();
        assert!(
            diff < 1e-5,
            "SIMD vs scalar mismatch {name}: {} vs {}",
            arr[0],
            arr[8]
        );
    }
}
