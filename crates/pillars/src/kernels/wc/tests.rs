use super::*;

/// Convenience: params for `n` identical populations (all weights 0).
/// Column-major layout `8 × n`: `[dt, τ_E, τ_I, w_EE, w_EI, w_IE, w_II, β]`
/// (the kernel reads `dt` from `params[0]`, the rest from rows at `r·n + i`).
fn params_for(n: usize, dt: f32, tau_e: f32, tau_i: f32, beta: f32) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 8];
    p[0] = dt;
    for i in 0..n {
        p[n + i] = tau_e;
        p[2 * n + i] = tau_i;
        p[7 * n + i] = beta;
    }
    p
}

fn run(
    steps: usize,
    e: &mut [f32],
    i: &mut [f32],
    ie: &[f32],
    ii_off: &[f32],
    params: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = e.len();
    let mut se = vec![0.0f32; n];
    let mut si = vec![0.0f32; n];
    for _ in 0..steps {
        wc_step(e, i, ie, ii_off, &mut se, &mut si, params);
    }
    (se, si)
}

#[test]
fn zero_weights_converge_to_half_sigmoid() {
    // With all weights zero and no input, fixed point is E = I = σ(0) = 0.5.
    let n = 4;
    let mut e = vec![0.1f32; n];
    let mut i = vec![0.7f32; n];
    let p = params_for(n, 0.05, 10.0, 10.0, 1.0);
    let ie = vec![0.0f32; n];
    let ii = vec![0.0f32; n];
    run(2000, &mut e, &mut i, &ie, &ii, &p);
    for (v, name) in [(e[0], "E"), (i[0], "I")] {
        assert!(
            (v - 0.5).abs() < 1e-3,
            "{name} should converge to σ(0)=0.5, got {v}"
        );
    }
}

#[test]
fn strong_excitation_saturates_activity() {
    // I_E = 3 → E* = σ(3) ≈ 0.9526 (weights all 0).
    let n = 4;
    let mut e = vec![0.0f32; n];
    let mut i = vec![0.0f32; n];
    let p = params_for(n, 0.05, 10.0, 10.0, 1.0);
    let ie = vec![3.0f32; n];
    let ii = vec![0.0f32; n];
    run(2000, &mut e, &mut i, &ie, &ii, &p);
    let expected = 1.0 / (1.0 + (-3.0f32).exp());
    assert!(
        (e[0] - expected).abs() < 1e-3,
        "E should saturate to σ(3)={expected}, got {}",
        e[0]
    );
}

#[test]
fn inhibition_depresses_e() {
    // With w_EI = 10 and I resting at σ(0)=0.5, the E input becomes
    // 3 − 10·0.5 = −2 → E* ≈ σ(−2) ≈ 0.119 — clearly below the 0.95 without it.
    let n = 4;
    let mut e = vec![0.0f32; n];
    let mut i = vec![0.0f32; n];
    let mut p = params_for(n, 0.05, 10.0, 10.0, 1.0);
    for idx in 0..n {
        p[4 * n + idx] = 10.0; // w_EI row
    }
    let ie = vec![3.0f32; n];
    let ii = vec![0.0f32; n];
    run(3000, &mut e, &mut i, &ie, &ii, &p);
    assert!(
        e[0] < 0.3,
        "inhibitory coupling should depress E below 0.3, got {}",
        e[0]
    );
    assert!(e[0] > 0.0, "E should stay positive, got {}", e[0]);
}

#[test]
fn zero_tau_freezes_state() {
    // τ = 0 (or negative): the population must not move at all.
    let n = 4;
    let mut e = vec![0.3f32; n];
    let mut i = vec![0.8f32; n];
    let mut p = params_for(n, 0.05, 10.0, 10.0, 1.0);
    for idx in 0..n {
        p[n + idx] = 0.0; // τ_E
        p[2 * n + idx] = -1.0; // τ_I (negative: also frozen)
    }
    let ie = vec![0.0f32; n];
    let ii = vec![0.0f32; n];
    let (se, si) = run(10, &mut e, &mut i, &ie, &ii, &p);
    assert_eq!(e, vec![0.3f32; n], "τ ≤ 0 must freeze E");
    assert_eq!(i, vec![0.8f32; n], "τ ≤ 0 must freeze I");
    // Even frozen populations produce the continuous sigmoid of their input.
    for &s in se.iter().chain(si.iter()) {
        assert!(
            (0.0..=1.0).contains(&s),
            "sigmoid rate {s} must stay in (0,1)"
        );
    }
}

#[test]
fn beta_scales_input_as_gain() {
    // First step from E=I=0, w=0: f = σ(β·I_E). β is a gain, not a width.
    let n = 4;
    let mut e = vec![0.0f32; n];
    let mut i = vec![0.0f32; n];
    let ie = vec![1.0f32; n];
    let ii = vec![0.0f32; n];
    let (se1, _) = run(
        1,
        &mut e,
        &mut i,
        &ie,
        &ii,
        &params_for(n, 0.01, 10.0, 10.0, 1.0),
    );

    let mut e2 = vec![0.0f32; n];
    let mut i2 = vec![0.0f32; n];
    let (se2, _) = run(
        1,
        &mut e2,
        &mut i2,
        &ie,
        &ii,
        &params_for(n, 0.01, 10.0, 10.0, 2.0),
    );

    let expected1 = 1.0 / (1.0 + (-1.0f32).exp());
    let expected2 = 1.0 / (1.0 + (-2.0f32).exp());
    assert!(
        (se1[0] - expected1).abs() < 1e-5,
        "β=1 rate should be σ(1)={expected1}, got {}",
        se1[0]
    );
    assert!(
        (se2[0] - expected2).abs() < 1e-5,
        "β=2 rate should be σ(2)={expected2}, got {}",
        se2[0]
    );
}

#[test]
fn simd_and_tail_agree_on_trajectory() {
    // Discipline of the project: a population pushed through the pure-SIMD
    // path (n=8) must match the mixed path (n=16 SIMD + 1 scalar tail) — the
    // documented SIMD/tail divergence (polynomial vs f32::exp, ±10 clamp)
    // stays inside float tolerance for well-behaved arguments.
    let n_simd = 8;
    let n_mixed = 17; // 16 SIMD + 1 tail
    let steps = 50;

    let mut e_s = vec![0.2f32; n_simd];
    let mut i_s = vec![0.1f32; n_simd];
    let p_s = params_for(n_simd, 0.01, 10.0, 10.0, 1.0);

    let mut e_m = vec![0.2f32; n_mixed];
    let mut i_m = vec![0.1f32; n_mixed];
    let p_m = params_for(n_mixed, 0.01, 10.0, 10.0, 1.0);

    let ie_s = vec![0.0f32; n_simd];
    let ii_s = vec![0.0f32; n_simd];
    let ie_m = vec![0.0f32; n_mixed];
    let ii_m = vec![0.0f32; n_mixed];
    let (se_s, _) = run(steps, &mut e_s, &mut i_s, &ie_s, &ii_s, &p_s);
    let (se_m, _) = run(steps, &mut e_m, &mut i_m, &ie_m, &ii_m, &p_m);

    // SIMD chunk of the mixed run must be bit-identical to the pure-SIMD run.
    assert_eq!(e_m[..8], e_s[..], "SIMD lanes diverged between runs");
    // The tail neuron converges to the same fixed point within tolerance.
    assert!(
        (e_m[16] - e_s[0]).abs() < 1e-4,
        "tail {} vs SIMD {}",
        e_m[16],
        e_s[0]
    );
    assert!(
        (se_m[16] - se_s[0]).abs() < 1e-4,
        "tail spike-rate {} vs SIMD {}",
        se_m[16],
        se_s[0]
    );
}

#[test]
fn spike_buffers_hold_continuous_rates() {
    // The WC kernel writes *continuous* sigmoid rates (0..1) into "spikes".
    let n = 4;
    let mut e = vec![0.0f32; n];
    let mut i = vec![0.0f32; n];
    let ie = vec![1.0f32; n];
    let ii = vec![0.5f32; n];
    let (se, si) = run(
        1,
        &mut e,
        &mut i,
        &ie,
        &ii,
        &params_for(n, 0.01, 10.0, 10.0, 1.0),
    );
    for &v in se.iter().chain(si.iter()) {
        assert!(
            (0.0..=1.0).contains(&v),
            "rate {v} must be continuous in (0,1)"
        );
    }
}

#[test]
fn extreme_input_clamped_not_nan() {
    // The ±100 input clamp must keep the kernel stable under absurd inputs.
    let n = 4;
    let mut e = vec![0.5f32; n];
    let mut i = vec![0.5f32; n];
    let ie = vec![1e9f32; n];
    let ii = vec![-1e9f32; n];
    let (se, _) = run(
        1,
        &mut e,
        &mut i,
        &ie,
        &ii,
        &params_for(n, 0.01, 10.0, 10.0, 1.0),
    );
    assert!(e.iter().all(|v| v.is_finite()), "E must stay finite");
    assert!(i.iter().all(|v| v.is_finite()), "I must stay finite");
    for &v in se.iter() {
        assert!(
            (0.0..=1.0).contains(&v),
            "rate {v} must stay in (0,1) even under extreme input"
        );
    }
}