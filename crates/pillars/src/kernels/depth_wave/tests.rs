use super::*;

#[test]
#[should_panic(expected = "row_ptr must have n + 1 entries")]
fn test_depth_wave_rejects_malformed_csr() {
    let mut dw = GcoState::new(2, vec![0.5; 2], 0.0, 0.1, 1);
    dw.compute_h_psi(&[], &[], &[], &[1.0, 1.0]);
}

#[test]
fn test_depth_wave_evolves_and_normalizes() {
    // Маленький граф: 4 нейрона, 4 синапса (кольцо)
    let n = 4;
    let depth = vec![0.9, 0.5, 0.5, 0.1]; // sensory → intrinsic → motor
    let mut dw = GcoState::new(n, depth, 2.0, 0.5, 42);

    // CSR кольцо: 0→1, 1→2, 2→3, 3→0
    let row_ptr = vec![0u32, 1, 2, 3, 4];
    let col_idx = vec![1u32, 2, 3, 0];
    let vals = vec![1.0f32, 1.0, 1.0, 1.0];
    let nt_signs = vec![1.0f32, 1.0, -1.0, 1.0]; // нейрон 2 тормозной

    // 100 шагов
    for _ in 0..100 {
        dw.step(&row_ptr, &col_idx, &vals, &nt_signs, None);
    }

    // Проверяем нормализацию |v+iu|² ≈ 1
    let norm_sq: f32 = dw.v.iter().zip(&dw.u).map(|(v, u)| v * v + u * u).sum();
    assert!(
        (norm_sq - 1.0).abs() < 0.01,
        "norm should be ~1.0, got {}",
        norm_sq
    );

    // Проверяем speed[0] > speed[3] (sensory быстрее motor)
    assert!(dw.speed[0] > dw.speed[3]);
}

#[test]
fn test_empty_depth_wave_coherence_is_zero() {
    let state = GcoState::new(0, vec![], 0.0, 0.1, 1);
    assert_eq!(state.coherence(), 0.0);
}

#[test]
fn test_depth_wave_coherence() {
    let n = 8;
    let depth = vec![0.5; n];
    let dw = GcoState::new(n, depth, 0.0, 0.5, 42);
    let coh = dw.coherence();
    assert!((0.0..=1.0).contains(&coh));
}

#[test]
fn test_motor_amplitudes() {
    let depth = vec![0.9, 0.5, 0.2, 0.05]; // 2 motor (depth<0.3)
    let dw = GcoState::new(4, depth, 2.0, 0.5, 42);
    let amps = dw.motor_amplitudes();
    assert_eq!(amps.len(), 2);
}
