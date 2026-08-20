use super::*;

#[test]
fn test_decay_factors_precision() {
    let factors = DecayFactors::new(0.5); // dt = 0.5 ms
    assert!(
        (factors.ampa - 0.7788).abs() < 0.001,
        "AMPA decay: {}",
        factors.ampa
    );
}

#[test]
fn test_decay_all_exponential() {
    let mut kin = SynapticKinetics::new(4);
    kin.g_ampa[0] = 1.0;
    kin.g_gaba_a[1] = 2.0;
    let factors = DecayFactors::new(2.0);
    kin.decay_all(&factors);
    assert!(
        (kin.g_ampa[0] - 0.3679).abs() < 0.001,
        "AMPA: {}",
        kin.g_ampa[0]
    );
    assert!(
        (kin.g_gaba_a[1] - 1.6375).abs() < 0.001,
        "GABA_A: {}",
        kin.g_gaba_a[1]
    );
}

#[test]
fn test_jump_basic() {
    let mut kin = SynapticKinetics::new(4);
    kin.jump(0, ReceptorType::AMPA, 0.5);
    kin.jump(1, ReceptorType::GABA_A, 1.0);
    kin.jump(0, ReceptorType::NMDA, 0.3);
    assert_eq!(kin.g_ampa[0], 0.5);
    assert_eq!(kin.g_gaba_a[1], 1.0);
    assert_eq!(kin.g_nmda[0], 0.3);
    assert_eq!(kin.total_conductance(0), 0.8);
}

#[test]
fn test_compute_currents_ohmic_direction() {
    // AMPA (E=0) at V=-65 → +6.5 (depol); GABA_A at rest → 0 (shunt)
    let mut kin = SynapticKinetics::new(2);
    let v = vec![-65.0, -65.0];
    let mut i_out = vec![0.0, 0.0];
    kin.g_ampa[0] = 0.1;
    kin.g_gaba_a[1] = 0.2;
    kin.compute_currents(&v, &mut i_out);
    assert!((i_out[0] - 6.5).abs() < 0.01, "AMPA I_syn: {}", i_out[0]);
    assert!(i_out[1].abs() < 0.01, "GABA_A shunt: {}", i_out[1]);
}

#[test]
fn test_nmda_mg_block_voltage_profile() {
    let v_test = [-80.0, -65.0, -40.0, -20.0, 0.0, 20.0, 40.0];
    let mut kin = SynapticKinetics::new(7);
    for (i, _) in v_test.iter().enumerate() {
        kin.g_nmda[i] = 1.0;
    }
    let mut i_out = vec![0.0; 7];
    kin.compute_currents(&v_test, &mut i_out);
    assert!(i_out[0] < 5.0, "NMDA at -80 blocked: {}", i_out[0]);
    assert!(i_out[4].abs() < 1.0, "NMDA at 0: {}", i_out[4]);
    assert!(i_out[5] < -5.0, "NMDA at +20: {}", i_out[5]);
}

#[test]
fn test_gabab_and_machr_inhibitory() {
    let mut kin = SynapticKinetics::new(2);
    let v = vec![-65.0, -65.0];
    let mut i_out = vec![0.0, 0.0];
    kin.g_gaba_b[0] = 1.0;
    kin.g_machr[1] = 1.0;
    kin.compute_currents(&v, &mut i_out);
    assert!((i_out[0] - (-25.0)).abs() < 0.1, "GABA_B: {}", i_out[0]);
    assert!((i_out[1] - (-25.0)).abs() < 0.1, "mAChR: {}", i_out[1]);
}

#[test]
fn test_simd_tail_consistent() {
    // n=13 → 8 SIMD lanes + 5 scalar tail; identical state ⇒ identical current.
    // g_nmda is nonzero for ALL lanes, so the SIMD Mg²⁺-block path is
    // exercised against the scalar reference at both lane 4 and tail 12.
    let n = 13;
    let mut kin = SynapticKinetics::new(n);
    for i in 0..n {
        kin.g_ampa[i] = 0.1 * (i as f32 + 1.0);
        kin.g_nmda[i] = 0.05 * (i as f32 + 1.0);
        kin.g_gaba_a[i] = 0.02;
        kin.g_nachr[i] = 0.01;
    }
    let v: Vec<f32> = (0..n).map(|i| -70.0 + i as f32).collect();
    let mut i_out = vec![0.0f32; n];
    kin.compute_currents(&v, &mut i_out);
    for (idx, tol) in [(0usize, 1e-4), (4, 1e-4), (12, 1e-4)] {
        let vi = v[idx];
        let b = nmda_mg_block(vi);
        let expected = 0.1 * (idx as f32 + 1.0) * (0.0 - vi)
            + 0.05 * (idx as f32 + 1.0) * b * (0.0 - vi)
            + 0.02 * (-65.0 - vi)
            + 0.01 * (0.0 - vi);
        assert!(
            (i_out[idx] - expected).abs() < tol,
            "idx {idx}: got {} vs {expected}",
            i_out[idx]
        );
    }
}

#[test]
fn test_build_receptor_types_csr() {
    let row_ptr = vec![0i32, 2, 4];
    let nt = vec![1.0f32, -1.0];
    let rec = build_receptor_types(&row_ptr, 4, &nt);
    assert_eq!(rec, vec![0u8, 0, 2, 2]);
}
