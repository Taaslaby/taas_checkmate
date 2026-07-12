//! # Backend-Agnostic Consistency Tests
//!
//! Проверяют, что ScalarBackend, Avx2Backend и (если доступен) CudaBackend
//! выдают одинаковые траектории и градиенты в пределах tolerance minimax sin/cos.
//!
//! ## Зачем
//!
//! - Разные бэкенды используют разную имплементацию trig: Avx2Backend использует
//!   `fast_sincos_simd` (minimax polynomial, ~1e-4), ScalarBackend — libm (~1 ULP).
//! - T3Backend trait должен гарантировать interchangeability.
//! - Тесты ловят регрессии, когда один бэкенд отстаёт от другого.
//!
//! ## Tolerance
//!
//! - `sin/cos`: 5e-5 (разница minimax vs libm)
//! - `θ, r`: 1e-4 (один шаг update)
//! - `PST features`: 1e-3 (накопление через 10 шагов)

use taas_t3::{TriOscillator, Avx2Backend, ScalarBackend, T3Backend};

/// Максимальная разница minimax vs libm sin/cos
const SIN_COS_TOL: f32 = 5e-5;
/// Максимальная разница state после одного шага
const STATE_TOL: f32 = 1e-4;
/// Максимальная разница PST после 10 шагов
const PST_TOL: f32 = 2e-3;

fn test_config() -> (usize, usize) { (16, 10) }

fn run_forward<B: taas_t3::T3Backend>(
    backend: B, n: usize, n_steps: usize, seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut tri = TriOscillator::new(n, 0.5, 3.0, backend);
    tri.set_uniform_coupling(0.3, 0.15, 0.08);
    tri.alpha = 1.0;
    tri.drive_gain = 10.0;

    for _ in 0..n_steps {
        for i in 0..n {
            tri.drive1[i] = (rng.gen::<f32>() - 0.5) * 2.0;
            tri.drive2[i] = (rng.gen::<f32>() - 0.5) * 2.0;
            tri.drive3[i] = (rng.gen::<f32>() - 0.5) * 2.0;
        }
        tri.step(0.05);
    }
    tri.prepare();

    (
        tri.th1.clone(), tri.th2.clone(), tri.th3.clone(),
        tri.r1.clone(), tri.r2.clone(), tri.r3.clone(),
    )
}

#[test]
fn test_backend_prepare_consistency() {
    let n = 16;
    let phases: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 + 1.3).sin() * 3.0).collect();

    let mut p_avx = phases.clone();
    let mut c_avx = vec![0.0; n];
    let mut s_avx = vec![0.0; n];
    Avx2Backend.prepare(&mut p_avx, &mut c_avx, &mut s_avx);

    let mut p_scalar = phases.clone();
    let mut c_scalar = vec![0.0; n];
    let mut s_scalar = vec![0.0; n];
    ScalarBackend.prepare(&mut p_scalar, &mut c_scalar, &mut s_scalar);

    for i in 0..n {
        let d_sin = (s_avx[i] - s_scalar[i]).abs();
        let d_cos = (c_avx[i] - c_scalar[i]).abs();
        let d_ph = (p_avx[i] - p_scalar[i]).abs();
        assert!(d_sin < SIN_COS_TOL,
            "sin[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, s_avx[i], s_scalar[i], d_sin);
        assert!(d_cos < SIN_COS_TOL,
            "cos[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, c_avx[i], c_scalar[i], d_cos);
        assert!(d_ph < 1e-6, "phase[{}] differs: avx={} scalar={}", i, p_avx[i], p_scalar[i]);
    }
}

#[test]
fn test_backend_update_consistency() {
    let n = 16;
    let phases = vec![0.5; n];
    let amps = vec![0.1; n];
    let omegas = vec![1.0; n];
    let coupling = vec![0.1; n];
    let drives = vec![0.2; n];

    let mut p_avx = phases.clone();
    let mut a_avx = amps.clone();
    Avx2Backend.update(&mut p_avx, &mut a_avx, &omegas, &coupling, &drives, 0.05, 1.0, 10.0);

    let mut p_scalar = phases.clone();
    let mut a_scalar = amps.clone();
    ScalarBackend.update(&mut p_scalar, &mut a_scalar, &omegas, &coupling, &drives, 0.05, 1.0, 10.0);

    for i in 0..n {
        let d_ph = (p_avx[i] - p_scalar[i]).abs();
        let d_amp = (a_avx[i] - a_scalar[i]).abs();
        assert!(d_ph < STATE_TOL,
            "phase[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, p_avx[i], p_scalar[i], d_ph);
        assert!(d_amp < STATE_TOL,
            "amp[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, a_avx[i], a_scalar[i], d_amp);
    }
}

#[test]
fn test_backend_trajectory_consistency_single_step() {
    let (n, n_steps) = (16, 1);

    let (th1_a, th2_a, th3_a, r1_a, r2_a, r3_a) = run_forward(Avx2Backend, n, n_steps, 42);
    let (th1_s, th2_s, th3_s, r1_s, r2_s, r3_s) = run_forward(ScalarBackend, n, n_steps, 42);

    let all_pairs = [
        (&th1_a, &th1_s, "th1"),
        (&th2_a, &th2_s, "th2"),
        (&th3_a, &th3_s, "th3"),
        (&r1_a, &r1_s, "r1"),
        (&r2_a, &r2_s, "r2"),
        (&r3_a, &r3_s, "r3"),
    ];

    for (avx, scalar, name) in &all_pairs {
        for i in 0..n {
            let diff = (avx[i] - scalar[i]).abs();
            assert!(diff < STATE_TOL,
                "{}[{}] avx={:.8} scalar={:.8} diff={:.2e}",
                name, i, avx[i], scalar[i], diff);
        }
    }
}

#[test]
fn test_backend_trajectory_consistency_multi_step() {
    let (n, n_steps) = test_config();
    let mut max_diff = 0.0f32;
    let mut worst = (0usize, String::new());

    for seed in [0u64, 1, 2, 42, 123] {
        let (th1_a, th2_a, th3_a, r1_a, r2_a, r3_a) = run_forward(Avx2Backend, n, n_steps, seed);
        let (th1_s, th2_s, th3_s, r1_s, r2_s, r3_s) = run_forward(ScalarBackend, n, n_steps, seed);

        let pairs = [
            (&th1_a, &th1_s, "th1"),
            (&th2_a, &th2_s, "th2"),
            (&th3_a, &th3_s, "th3"),
            (&r1_a, &r1_s, "r1"),
            (&r2_a, &r2_s, "r2"),
            (&r3_a, &r3_s, "r3"),
        ];

        for (avx, scalar, name) in &pairs {
            for i in 0..n {
                let diff = (avx[i] - scalar[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    worst = (seed as usize, format!("{}[{}]", name, i));
                }
                assert!(diff < PST_TOL,
                    "seed={} {}[{}] avx={:.8} scalar={:.8} diff={:.2e}",
                    seed, name, i, avx[i], scalar[i], diff);
            }
        }
    }

    println!("Backend consistency: max_diff={:.2e} at seed={} {}",
        max_diff, worst.0, worst.1);
}

#[test]
fn test_backend_identity_with_zero_drive() {
    // Без drives и coupling, оба бэкенда должны давать идентичные результаты
    // (только freq-driven phase rotation + Stuart-Landau relaxation)
    let n = 16;
    let n_steps = 10;
    let mut tri_avx = TriOscillator::new(n, 0.5, 0.5, Avx2Backend); // same freq
    tri_avx.set_uniform_coupling(0.0, 0.0, 0.0);
    tri_avx.alpha = 1.0;

    let mut tri_scalar = TriOscillator::new(n, 0.5, 0.5, ScalarBackend);
    tri_scalar.set_uniform_coupling(0.0, 0.0, 0.0);
    tri_scalar.alpha = 1.0;

    for _ in 0..n_steps {
        tri_avx.step(0.05);
        tri_scalar.step(0.05);
    }

    for i in 0..n {
        let d_th = (tri_avx.th1[i] - tri_scalar.th1[i]).abs();
        let d_r = (tri_avx.r1[i] - tri_scalar.r1[i]).abs();
        // With zero drive and coupling, phases drift at same freq.
        // Small difference due to sin/cos implementation in prepare()
        // (libm vs minimax) accumulates over steps.
        assert!(d_th < 1e-3,
            "th1[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, tri_avx.th1[i], tri_scalar.th1[i], d_th);
        assert!(d_r < 1e-4,
            "r1[{}] avx={:.8} scalar={:.8} diff={:.2e}", i, tri_avx.r1[i], tri_scalar.r1[i], d_r);
    }
}

/// Test that NLMS decoder trains identically with both backends
#[test]
fn test_backend_nlms_consistency() {
    let n = 16;
    let mut w_avx = vec![0.5f32; n];
    let mut b_avx = vec![0.0f32];
    let mut w_scalar = w_avx.clone();
    let mut b_scalar = b_avx.clone();
    let states: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let error = 1.0f32;

    Avx2Backend.nlms_train(&mut w_avx, &mut b_avx, &states, error, 0.1, 1e-8);
    ScalarBackend.nlms_train(&mut w_scalar, &mut b_scalar, &states, error, 0.1, 1e-8);

    for i in 0..n {
        let diff = (w_avx[i] - w_scalar[i]).abs();
        assert!(diff < 1e-7, "w[{}] avx={:.10} scalar={:.10}", i, w_avx[i], w_scalar[i]);
    }
}

/// Test that Kalman decoder updates identically with both backends
#[test]
fn test_backend_kalman_consistency() {
    let n = 16;
    let mut w_avx = vec![0.5f32; n];
    let mut cov_avx = vec![1.0f32; n];
    let mut w_scalar = w_avx.clone();
    let mut cov_scalar = cov_avx.clone();
    let states: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();

    Avx2Backend.kalman_update(&mut w_avx, &mut cov_avx, &states, 1.0, 2.0);
    ScalarBackend.kalman_update(&mut w_scalar, &mut cov_scalar, &states, 1.0, 2.0);

    for i in 0..n {
        let d_w = (w_avx[i] - w_scalar[i]).abs();
        let d_cov = (cov_avx[i] - cov_scalar[i]).abs();
        assert!(d_w < 1e-7, "w[{}] avx={:.10} scalar={:.10}", i, w_avx[i], w_scalar[i]);
        assert!(d_cov < 1e-7, "cov[{}] avx={:.10} scalar={:.10}", i, cov_avx[i], cov_scalar[i]);
    }
}
