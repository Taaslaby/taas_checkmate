use super::*;

#[test]
fn surrogate_gradient_peaks_at_threshold() {
    let g0 = surrogate_gradient(-54.0, -54.0, 1.0);
    assert!((g0 - 1.0).abs() < 1e-6, "at threshold → 1.0, got {g0}");
    let g_far = surrogate_gradient(-70.0, -54.0, 1.0);
    assert!(g_far < g0, "far from threshold → smaller gradient");
}

#[test]
fn eprop_update_basic() {
    let n = 16;
    let mut weights = vec![0.5f32; n];
    let eligibility = vec![1.0f32; n];
    let learning_signals = vec![0.1f32; n];
    eprop_update(&mut weights, &eligibility, &learning_signals, 0.1);
    for i in 0..n {
        assert!(
            (weights[i] - 0.51).abs() < 1e-5,
            "w should be 0.51, got {}",
            weights[i]
        );
    }
}

#[test]
fn intrinsic_plasticity_pushes_toward_target() {
    let n = 8;
    let mut theta = vec![10.0f32; n];
    let spikes = vec![0i32; n]; // below target_rate=0.01 → theta decreases
    intrinsic_plasticity(&mut theta, &spikes, 0.1, 0.01, 0.99);
    assert!(theta[0] < 10.0, "below-target firing should lower theta");
}

#[test]
fn synaptic_normalization_preserves_ratios() {
    let n_pre = 2;
    let n_post = 2;
    let mut w = vec![1.0f32, 2.0, 3.0, 4.0];
    // Column sums: col0=4, col1=6 → scales: 0.25, 1/6
    synaptic_normalization(&mut w, n_pre, n_post, 1.0, 1e-8);
    let col0: f32 = w[0] + w[2];
    let col1: f32 = w[1] + w[3];
    assert!(
        (col0 - 1.0).abs() < 1e-5,
        "col0 sum should be 1, got {col0}"
    );
    assert!(
        (col1 - 1.0).abs() < 1e-5,
        "col1 sum should be 1, got {col1}"
    );
}

#[test]
fn stdp_update_basic() {
    let n = 16;
    let mut weights = vec![0.5f32; n];
    let traces_pre = vec![0.0f32; n];
    let traces_post = vec![1.0f32; n];
    stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, 0.0, 1.0);
    for i in 0..n {
        assert!((weights[i] - 0.6).abs() < 0.01, "stdp[{i}]: {}", weights[i]);
    }
}

#[test]
fn stdp_update_simd_tail_consistent() {
    // n = 10 → 8 SIMD lanes + 2 scalar tail, both must produce identical bits.
    let n = 10;
    let mut w_simd = vec![0.3f32; n];
    let traces_pre: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let traces_post: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.02).collect();
    stdp_update(&mut w_simd, &traces_pre, &traces_post, 0.05, 0.1, 0.0, 1.0);
    // Reference computed scalar (identical arithmetic order for f32).
    for i in 0..n {
        let expected = (0.3 + 0.1 * traces_post[i] - 0.05 * traces_pre[i]).clamp(0.0, 1.0);
        assert!(
            (w_simd[i] - expected).abs() < 1e-6,
            "stdp lane/tail mismatch at {i}: got {}, expected {}",
            w_simd[i],
            expected
        );
    }
}

#[test]
fn sorn_step_runs_without_panic() {
    let mut sorn = SornNetwork::new(8, 0.1, 1.0);
    let spikes = vec![0i32; 8];
    let traces_pre = vec![0.0f32; 64];
    let traces_post = vec![0.0f32; 64];
    sorn.step(&spikes, &traces_pre, &traces_post);
    assert!(
        sorn.weights.iter().all(|&w| w.is_finite()),
        "weights must stay finite"
    );
}

// ─── BCM ────────────────────────────────────────────────────

fn run_bcm(n: usize, post_val: f32, theta_init: f32, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let dt = 1.0;
    let tau_x = 20.0;
    let tau_y = 20.0;
    let tau_theta = 100.0;
    let eta = 0.01;
    let decay = 0.001;
    let w_min = 0.0;
    let w_max = 10.0;

    let mut x = vec![0.5f32; n];
    let mut y = vec![post_val; n];
    let mut theta = vec![theta_init; n];
    let mut weights = vec![1.0f32; n];
    let pre = vec![0.5f32; n];
    let post = vec![post_val; n];

    for _ in 0..steps {
        bcm_step(
            &mut x,
            &mut y,
            &mut theta,
            &mut weights,
            &pre,
            &post,
            dt,
            tau_x,
            tau_y,
            tau_theta,
            eta,
            decay,
            w_min,
            w_max,
        );
    }
    (weights, theta)
}

#[test]
fn bcm_bidirectional_plasticity() {
    // LTP: post (10) > θ (5) → weight increases
    let (w_ltp, _) = run_bcm(8, 10.0, 5.0, 1);
    assert!(
        w_ltp[0] > 1.0,
        "BCM LTP: weight should increase, got {:.6}",
        w_ltp[0]
    );

    // LTD: post (1) < θ (5) → weight decreases
    let (w_ltd, _) = run_bcm(8, 1.0, 5.0, 1);
    assert!(
        w_ltd[0] < 1.0,
        "BCM LTD: weight should decrease, got {:.6}",
        w_ltd[0]
    );

    assert!(
        w_ltp[0] > w_ltd[0],
        "LTP ({:.6}) > LTD ({:.6})",
        w_ltp[0],
        w_ltd[0]
    );
}

#[test]
fn bcm_threshold_follows_activity() {
    // θ_M tracks y² ≈ 21 (post=5, τ_θ=100, 50 steps): θ should rise but < 30
    let (_, theta) = run_bcm(8, 5.0, 0.0, 50);
    assert!(theta[0] > 2.0, "BCM: θ_M should rise, got {:.4}", theta[0]);
    assert!(
        theta[0] < 30.0,
        "BCM: θ_M should be <30, got {:.4}",
        theta[0]
    );
}

#[test]
fn bcm_non_aligned_n_simd_tail_consistent() {
    // n=13 → 8 SIMD lanes + 5 scalar tail; identical params ⇒ same result
    let (weights, theta) = run_bcm(13, 8.0, 5.0, 100);
    assert!(
        (weights[0] - weights[12]).abs() < 1e-5,
        "n=13 lane/tail mismatch: nrn0={:.6}, nrn12={:.6}",
        weights[0],
        weights[12]
    );
    assert!(
        (theta[0] - theta[12]).abs() < 1e-5,
        "n=13 θ mismatch: nrn0={:.6}, nrn12={:.6}",
        theta[0],
        theta[12]
    );
    assert!(
        weights.iter().all(|&w| w.is_finite()),
        "weights must be finite"
    );
}

#[test]
fn homeostatic_scaling_direction() {
    let dt = 1.0;
    let eta_h = 0.01;
    let target = 5.0;

    // High activity (10 > 5) → weights shrink
    let mut w_hi = vec![1.0f32; 8];
    let y_hi = vec![10.0f32; 8];
    homeostatic_scale(&mut w_hi, &y_hi, dt, eta_h, target);
    assert!(
        w_hi[0] < 1.0,
        "high activity should shrink weights, got {}",
        w_hi[0]
    );

    // Low activity (2 < 5) → weights grow
    let mut w_lo = vec![1.0f32; 8];
    let y_lo = vec![2.0f32; 8];
    homeostatic_scale(&mut w_lo, &y_lo, dt, eta_h, target);
    assert!(
        w_lo[0] > 1.0,
        "low activity should grow weights, got {}",
        w_lo[0]
    );

    // At target → no change
    let mut w_at = vec![1.0f32; 8];
    let y_at = vec![5.0f32; 8];
    homeostatic_scale(&mut w_at, &y_at, dt, eta_h, target);
    assert!(
        (w_at[0] - 1.0).abs() < 1e-6,
        "at target: {}, want 1.0",
        w_at[0]
    );
}
