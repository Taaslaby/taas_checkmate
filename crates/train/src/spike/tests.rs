#![allow(deprecated)]
use crate::spike::{t3spike_backward, t3spike_forward, T3SpikeTrajectory};
use crate::t3::pst_linear_dim;

fn make_drives(seed: usize, n_steps: usize, n: usize) -> Vec<f32> {
    let mut rng = |i: usize| -> f32 { ((i as f64 * 1.337).sin() * 1.5) as f32 };
    let n_total = n_steps * 3 * n;
    (0..n_total)
        .map(|i| {
            if i < 10 {
                ((seed as f64 * 0.7 + i as f64 * 1.337).sin() * 1.5) as f32
            } else {
                rng(seed * 1000 + i)
            }
        })
        .collect()
}

#[test]
fn test_pst_jacobian_analytic_vs_fd() {
    for (seed, n_osc) in [(0, 1), (1, 1), (2, 2), (3, 4)] {
        let n_steps = 1;
        let drives = make_drives(seed, n_steps, n_osc);

        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        let loss_feat = 2 * n_osc.min(1);
        grad_pst[loss_feat] = 1.0;

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for step in 0..n_steps {
            for pop in 0..3 {
                for osc in 0..n_osc {
                    let idx = step * 3 * n_osc + pop * n_osc + osc;
                    let mut dp = drives.clone();
                    dp[idx] += eps_fd;
                    let mut dm = drives.clone();
                    dm[idx] -= eps_fd;
                    let (pp, _) = t3spike_forward(
                        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
                    );
                    let (pm, _) = t3spike_forward(
                        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
                    );
                    let fd = (pp[loss_feat] - pm[loss_feat]) / (2.0 * eps_fd);
                    if fd.abs() > 1e-3 {
                        let ratio = ana[idx] / fd;
                        assert!((ratio - 1.0).abs() < 0.15,
                            "seed={seed} n={n_osc} step={step} pop={pop} osc={osc}: ana={:.6} fd={:.6} ratio={:.3}",
                            ana[idx], fd, ratio);
                    }
                }
            }
        }
    }
}

#[test]
fn test_full_backward_single_seed_all() {
    let n_osc = 2;
    let n_steps = 3;
    let drives = make_drives(0, n_steps, n_osc);

    let (_, traj) = t3spike_forward(
        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
    );

    let pst_dim = pst_linear_dim(n_osc);
    let mut grad_pst = vec![0.0; pst_dim];
    for i in 0..n_osc {
        grad_pst[2 * i] = 1.0;
    }

    let ana = t3spike_backward(&grad_pst, &traj);

    let eps_fd = 1e-4f32;
    for step in 0..n_steps {
        for pop in 0..3 {
            for osc in 0..n_osc {
                let idx = step * 3 * n_osc + pop * n_osc + osc;
                let mut dp = drives.clone();
                dp[idx] += eps_fd;
                let mut dm = drives.clone();
                dm[idx] -= eps_fd;
                let (pp, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
                );
                let (pm, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
                );
                let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
                let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
                let fd = (loss_p - loss_m) / (2.0 * eps_fd);
                if ana[idx].abs() > 1e-2 || fd.abs() > 1e-2 {
                    let ratio = ana[idx] / fd;
                    assert!(
                        (ratio - 1.0).abs() < 0.25,
                        "step={step} pop={pop} osc={osc}: ana={:.6} fd={:.6} ratio={:.3}",
                        ana[idx],
                        fd,
                        ratio
                    );
                }
            }
        }
    }
}

#[test]
fn test_full_backward_fd_multiple_seeds() {
    for seed in 0..5 {
        let n_osc = 8;
        let n_steps = 6;
        let drives = make_drives(seed, n_steps, n_osc);

        let (_, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        for i in 0..n_osc {
            grad_pst[2 * i] = 1.0;
        }

        let ana = t3spike_backward(&grad_pst, &traj);

        // For seed 2, verify FD sign by printing actual PST values
        if seed == 2 {
            let eps = 1e-4;
            let idx = 4 * 3 * n_osc; // step=4, pop=0, osc=0
            let mut dp = drives.clone();
            dp[idx] += eps;
            let mut dm = drives.clone();
            dm[idx] -= eps;
            let (pp, tp) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
            );
            let (pm, tm) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
            );
            let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
            let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
            println!(
                "seed=2 step=4: baseline loss sum sin={:.6}",
                (0..n_osc).map(|i| traj.th1_final[i].sin()).sum::<f32>()
            );
            println!(
                "seed=2 step=4: loss_p={:.10} loss_m={:.10} diff={:.10}",
                loss_p,
                loss_m,
                loss_p - loss_m
            );

            // Check θ_final and per-osc sin changes
            println!("seed=2: θ_final changes from drive[4,0] perturbation");
            for osc in 0..n_osc {
                let th_p = tp.th1_final[osc];
                let th_m = tm.th1_final[osc];
                let dth = (th_p - th_m) / (2.0 * eps);
                let sin_p = pp[2 * osc];
                let sin_m = pm[2 * osc];
                let dsin = (sin_p - sin_m) / (2.0 * eps);
                println!("  osc={}: dθ={:.6} dsin={:.6}", osc, dth, dsin);
            }
        }

        let eps_fd = 1e-4f32;
        for step in 0..n_steps {
            let idx = step * 3 * n_osc;
            let mut dp = drives.clone();
            dp[idx] += eps_fd;
            let mut dm = drives.clone();
            dm[idx] -= eps_fd;
            let (pp, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
            );
            let (pm, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
            );

            let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
            let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
            let fd = (loss_p - loss_m) / (2.0 * eps_fd);

            if fd.abs() > 1e-3 && ana[idx].abs() > 1e-3 {
                let ratio = ana[idx] / fd;
                assert!(
                    (ratio - 1.0).abs() < 0.25,
                    "seed={seed} step={step} idx={idx}: ana={:.6} fd={:.6} ratio={:.3}",
                    ana[idx],
                    fd,
                    ratio
                );
            }
        }
    }
}

#[test]
fn test_dynamics_jacobian_fd() {
    for (seed, n_osc) in [(0, 1), (1, 2), (2, 4)] {
        let n_steps = 1;
        let drives = make_drives(seed, n_steps, n_osc);

        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        let feat = 0;
        grad_pst[feat] = 1.0;

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for pop in 0..3 {
            for osc in 0..n_osc {
                let idx = pop * n_osc + osc;
                let mut dp = drives.clone();
                dp[idx] += eps_fd;
                let mut dm = drives.clone();
                dm[idx] -= eps_fd;
                let (pp, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
                );
                let (pm, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
                );
                let fd = (pp[feat] - pm[feat]) / (2.0 * eps_fd);
                if fd.abs() > 1e-4 {
                    let ratio = ana[idx] / fd;
                    assert!(
                        (ratio - 1.0).abs() < 0.15,
                        "seed={seed} n={n_osc} pop={pop} osc={osc}: ana={:.6} fd={:.6} ratio={:.3}",
                        ana[idx],
                        fd,
                        ratio
                    );
                }
            }
        }
    }
}

#[test]
fn test_fast_pst_jacobian_analytic_vs_fd() {
    for (seed, n_osc) in [(0, 1), (1, 2), (2, 4)] {
        let n_steps = 1;
        let drives = make_drives(seed, n_steps, n_osc);

        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        let loss_feat = 0;
        grad_pst[loss_feat] = 1.0;

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for pop in 0..3 {
            for osc in 0..n_osc {
                let idx = pop * n_osc + osc;
                let mut dp = drives.clone();
                dp[idx] += eps_fd;
                let mut dm = drives.clone();
                dm[idx] -= eps_fd;
                let (pp, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
                );
                let (pm, _) = t3spike_forward(
                    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
                );
                let fd = (pp[loss_feat] - pm[loss_feat]) / (2.0 * eps_fd);
                if fd.abs() > 1e-3 {
                    let ratio = ana[idx] / fd;
                    assert!(
                        (ratio - 1.0).abs() < 0.15,
                        "seed={seed} n={n_osc} pop={pop} osc={osc}: ana={:.6} fd={:.6} ratio={:.3}",
                        ana[idx],
                        fd,
                        ratio
                    );
                }
            }
        }
    }
}

#[test]
fn test_naked_bptt_no_coupling() {
    for (seed, n_osc, n_steps) in [(0, 1, 2), (1, 1, 3), (2, 1, 4)] {
        let drives = make_drives(seed, n_steps, n_osc);

        let (_, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.0, 0.0, 0.0, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        grad_pst[0] = 1.0;

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for step in 0..n_steps {
            let idx = step * 3 * n_osc;
            let mut dp = drives.clone();
            dp[idx] += eps_fd;
            let mut dm = drives.clone();
            dm[idx] -= eps_fd;
            let (pp, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.0, 0.0, 0.0, &dp,
            );
            let (pm, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.0, 0.0, 0.0, &dm,
            );
            let fd = (pp[0] - pm[0]) / (2.0 * eps_fd);
            if fd.abs() > 1e-5 {
                let ratio = ana[idx] / fd;
                assert!((ratio - 1.0).abs() < 0.15,
                    "seed={seed} n={n_osc} steps={n_steps} step={step}: ana={:.6} fd={:.6} ratio={:.3}",
                    ana[idx], fd, ratio);
            }
        }
    }
}

#[test]
fn test_bptt_with_coupling() {
    for (seed, n_osc, n_steps) in [(0, 1, 3), (1, 1, 3), (2, 1, 3)] {
        let drives = make_drives(seed, n_steps, n_osc);

        let (_, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        grad_pst[0] = 1.0;

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for step in 0..n_steps {
            let idx = step * 3 * n_osc;
            let mut dp = drives.clone();
            dp[idx] += eps_fd;
            let mut dm = drives.clone();
            dm[idx] -= eps_fd;
            let (pp, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
            );
            let (pm, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
            );
            let fd = (pp[0] - pm[0]) / (2.0 * eps_fd);
            if fd.abs() > 1e-4 {
                let ratio = ana[idx] / fd;
                assert!(
                    (ratio - 1.0).abs() < 0.20,
                    "seed={seed} step={step}: ana={:.6} fd={:.6} ratio={:.3}",
                    ana[idx],
                    fd,
                    ratio
                );
            }
        }
    }
}

#[test]
fn test_bptt_multi_osc_with_coupling() {
    for (n_osc, n_steps, seed) in [(2, 3, 0), (4, 3, 0), (8, 3, 0), (8, 6, 0)] {
        let drives = make_drives(seed, n_steps, n_osc);

        let (_, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0; pst_dim];
        for i in 0..n_osc {
            grad_pst[2 * i] = 1.0;
        }

        let ana = t3spike_backward(&grad_pst, &traj);

        let eps_fd = 1e-4f32;
        for step in 0..n_steps {
            let idx = step * 3 * n_osc;
            let mut dp = drives.clone();
            dp[idx] += eps_fd;
            let mut dm = drives.clone();
            dm[idx] -= eps_fd;
            let (pp, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
            );
            let (pm, _) = t3spike_forward(
                n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dm,
            );

            let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
            let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
            let fd = (loss_p - loss_m) / (2.0 * eps_fd);
            if fd.abs() > 1e-3 && ana[idx].abs() > 1e-3 {
                let ratio = ana[idx] / fd;
                assert!(
                    (ratio - 1.0).abs() < 0.20,
                    "n={n_osc} steps={n_steps} step={step}: ana={:.6} fd={:.6} ratio={:.3}",
                    ana[idx],
                    fd,
                    ratio
                );
            }
        }
    }
}

/// Direct check: dθ_post/ddrive for step 3, osc 0 must equal dt*gain/(r+ε)
#[test]
fn test_theta_post_derivative() {
    let seed = 1;
    let n_osc = 8;
    let n_steps = 6;
    let drives = make_drives(seed, n_steps, n_osc);

    // Unperturbed reference
    let (_, traj) = t3spike_forward(
        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives,
    );

    // Perturb drive[3,0,0] by ±eps
    let eps = 1e-4f32;
    let idx = 3 * 3 * n_osc + 0;
    let mut dp = drives.clone();
    let orig = dp[idx];
    dp[idx] = orig + eps;
    let (_, tp) = t3spike_forward(
        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
    );
    dp[idx] = orig - eps;
    let (_, tm) = t3spike_forward(
        n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &dp,
    );

    // θ_post at step 3 = θ_pre at step 4
    let th_p = tp.th1_pre()[4 * n_osc + 0];
    let th_m = tm.th1_pre()[4 * n_osc + 0];
    let fd_dth = (th_p - th_m) / (2.0 * eps);

    let r_pre = traj.r1_pre()[3 * n_osc + 0];
    let r_pre4 = traj.r1_pre()[4 * n_osc + 0];
    let th_pre3 = traj.th1_pre()[3 * n_osc + 0];
    let th_pre4 = traj.th1_pre()[4 * n_osc + 0];
    println!(
        "r_pre[3,0]={:.6} r_pre[4,0]={:.6} th_pre[3,0]={:.6} th_pre[4,0]={:.6}",
        r_pre, r_pre4, th_pre3, th_pre4
    );

    let expected = 0.05f32 * 10.0 / (r_pre + 0.1);
    println!(
        "fd_dth={:.6} expected={:.6} drive_orig={:.6}",
        fd_dth,
        expected,
        drives[3 * 3 * n_osc + 0]
    );
    let tol = 0.05;
    assert!(
        (fd_dth / expected - 1.0).abs() < tol,
        "dθ_post/ddrive FD={:.6} expected={:.6} ratio={:.4}",
        fd_dth,
        expected,
        fd_dth / expected
    );
}

// ─── Gradient survival on large-scale T3 ─────────────────────────────

/// Generate random drives for large-scale tests.
fn random_drives(n_steps: usize, n: usize, seed: u64) -> Vec<f32> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_total = n_steps * 3 * n;
    let mut drives = Vec::with_capacity(n_total);
    for _ in 0..n_total {
        drives.push(rng.gen::<f32>() * 0.2 - 0.1);
    }
    drives
}

#[test]
fn test_gradient_survival_large_scale() {
    let n_osc = 24;
    let n_steps = 500;
    let dt = 0.05;
    let drive_gain = 10.0;
    let alpha = 1.0;
    let coupling_w12 = 0.3;
    let coupling_w13 = 0.15;
    let coupling_w23 = 0.08;

    println!("\n═══════════════════════════════════════════════════");
    println!("Gradient Survival Test: n_osc={n_osc}, n_steps={n_steps}");
    println!("═══════════════════════════════════════════════════");

    for seed in 0..3 {
        println!("\n─── Seed {seed} ───");
        let drives = random_drives(n_steps, n_osc, seed);

        let (_pst, traj) = t3spike_forward(
            n_osc,
            n_steps,
            dt,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &drives,
        );

        // Check trajectory sanity
        let r1_pre = traj.r1_pre();
        for i in 0..r1_pre.len() {
            assert!(r1_pre[i].is_finite(), "r1_pre[{}] = {:?}", i, r1_pre[i]);
            assert!(r1_pre[i] >= 0.0, "r1_pre[{}] = {} < 0", i, r1_pre[i]);
        }

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0f32; pst_dim];
        for i in 0..n_osc {
            grad_pst[2 * i] = 1.0; // sin(θ)
            grad_pst[pst_dim - 3 * n_osc + i] = 1.0; // r
        }

        let grad = crate::spike::t3spike_backward(&grad_pst, &traj);

        // Check no NaN/Inf
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "grad[{}] = {:?} (seed={})", i, g, seed);
        }

        // Per-step statistics
        let per_step = |step: usize| -> (f32, f32, f32) {
            let off = step * 3 * n_osc;
            let slice = &grad[off..off + 3 * n_osc];
            let n = slice.len() as f32;
            let mean = slice.iter().copied().sum::<f32>() / n;
            let max = slice.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            (
                mean,
                max,
                slice.iter().copied().map(|g| g * g).sum::<f32>().sqrt(),
            )
        };

        let (mean0, max0, norm0) = per_step(0);
        let (mean_last, max_last, norm_last) = per_step(n_steps - 1);

        // Find step with max gradient norm
        let (max_norm_step, max_norm) = (0..n_steps)
            .map(|s| (s, per_step(s).2))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("  Step   0: mean={mean0:.6e}, max={max0:.6e}, norm={norm0:.6e}");
        println!("  Step 499: mean={mean_last:.6e}, max={max_last:.6e}, norm={norm_last:.6e}");
        println!("  Max norm: step={max_norm_step}, norm={max_norm:.6e}");
        println!("  Norm ratio (last/first): {:.6e}", norm_last / norm0);

        // Assertions
        let max_all = grad.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!(
            max_all < 1e6,
            "seed={seed}: gradient explosion |grad|={:.6e}",
            max_all
        );

        let ratio = norm_last / norm0;
        assert!(
            ratio < 1e6,
            "seed={seed}: gradient vanishes (norm ratio={:.6e})",
            ratio
        );

        println!("  ✅ Seed {seed}: max={max_all:.6e}, norm_ratio={ratio:.6e}");
    }
}

#[test]
fn test_gradient_survival_analytic_vs_fd() {
    let n_osc = 4;
    let n_steps = 200;
    let dt = 0.05;
    let drive_gain = 10.0;
    let alpha = 1.0;
    let coupling_w12 = 0.3;
    let coupling_w13 = 0.15;
    let coupling_w23 = 0.08;

    println!("\n═══════════════════════════════════════════════════");
    println!("Gradient FD Check: n_osc={n_osc}, n_steps={n_steps}");
    println!("(checking first 3 steps, 1 oscillator per pop)");
    println!("═══════════════════════════════════════════════════");

    for seed in 0..3 {
        println!("\n─── Seed {seed} ───");
        let drives = random_drives(n_steps, n_osc, seed);

        let (_pst, traj) = t3spike_forward(
            n_osc,
            n_steps,
            dt,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0f32; pst_dim];
        for i in 0..n_osc {
            grad_pst[2 * i] = 1.0;
        }

        let ana = crate::spike::t3spike_backward(&grad_pst, &traj);

        let eps = 1e-3f32;
        let mut n_checked = 0;
        let mut n_bad = 0;

        for step in 0..3.min(n_steps) {
            for pop in 0..3 {
                let idx = step * 3 * n_osc + pop * n_osc + 0;
                let mut dp = drives.clone();
                dp[idx] += eps;
                let mut dm = drives.clone();
                dm[idx] -= eps;
                let (pp, _) = t3spike_forward(
                    n_osc,
                    n_steps,
                    dt,
                    drive_gain,
                    alpha,
                    0.5,
                    3.0,
                    coupling_w12,
                    coupling_w13,
                    coupling_w23,
                    &dp,
                );
                let (pm, _) = t3spike_forward(
                    n_osc,
                    n_steps,
                    dt,
                    drive_gain,
                    alpha,
                    0.5,
                    3.0,
                    coupling_w12,
                    coupling_w13,
                    coupling_w23,
                    &dm,
                );

                let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
                let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
                let fd = (loss_p - loss_m) / (2.0 * eps);

                println!(
                    "  step={step} pop={pop}: ana={:.6e} fd={:.6e} ratio={:+.3}",
                    ana[idx],
                    fd,
                    ana[idx] / fd
                );
                if fd.abs() > 5e-2 && ana[idx].abs() > 5e-2 {
                    n_checked += 1;
                    let ratio = ana[idx] / fd;
                    if (ratio - 1.0).abs() > 0.35 {
                        n_bad += 1;
                        println!(
                            "  ⚠ step={step} pop={pop}: ana={:.6e} fd={:.6e} ratio={:.3}",
                            ana[idx], fd, ratio
                        );
                    }
                }
            }
        }

        if n_bad > 0 {
            panic!("seed={seed}: {n_bad}/{n_checked} FD checks failed");
        }
        println!("  ✅ Seed {seed}: {n_checked} checks passed");
    }
}

#[test]
fn test_gradient_sign_debug_seed2() {
    let n_osc = 4;
    let n_steps = 200;
    let dt = 0.05;
    let drive_gain = 10.0;
    let alpha = 1.0;
    let coupling_w12 = 0.3;
    let coupling_w13 = 0.15;
    let coupling_w23 = 0.08;

    let drives = random_drives(n_steps, n_osc, 2);

    let (pst, traj) = t3spike_forward(
        n_osc,
        n_steps,
        dt,
        drive_gain,
        alpha,
        0.5,
        3.0,
        coupling_w12,
        coupling_w13,
        coupling_w23,
        &drives,
    );

    let pst_dim = pst_linear_dim(n_osc);
    let mut grad_pst = vec![0.0f32; pst_dim];
    for i in 0..n_osc {
        grad_pst[2 * i] = 1.0;
    }

    let ana = crate::spike::t3spike_backward(&grad_pst, &traj);

    for eps_val in [1e-2f32, 5e-3, 2e-3, 1e-3, 5e-4] {
        println!("\n  eps={:.0e}:", eps_val);
        for step in 0..3 {
            let idx = step * 3 * n_osc + 2 * n_osc + 0;
            let mut dp = drives.clone();
            dp[idx] += eps_val;
            let mut dm = drives.clone();
            dm[idx] -= eps_val;
            let (pp, _) = t3spike_forward(
                n_osc,
                n_steps,
                dt,
                drive_gain,
                alpha,
                0.5,
                3.0,
                coupling_w12,
                coupling_w13,
                coupling_w23,
                &dp,
            );
            let (pm, _) = t3spike_forward(
                n_osc,
                n_steps,
                dt,
                drive_gain,
                alpha,
                0.5,
                3.0,
                coupling_w12,
                coupling_w13,
                coupling_w23,
                &dm,
            );
            let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
            let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
            let fd = (loss_p - loss_m) / (2.0 * eps_val);
            println!(
                "  step={step} pop=2: ana={:.6e} fd={:.6e} ratio={:+.4}",
                ana[idx],
                fd,
                ana[idx] / fd
            );
        }
    }
}

#[test]
fn test_gradient_sign_vs_nsteps() {
    let n_osc = 2;
    let drive_gain = 10.0;
    let alpha = 1.0;
    let coupling_w12 = 0.3;
    let coupling_w13 = 0.15;
    let coupling_w23 = 0.08;

    let mut drives_base = vec![0.05f32; 200 * 3 * n_osc];
    for i in 0..200 {
        drives_base[i * 3 * n_osc + 2 * n_osc] = 0.03;
        drives_base[i * 3 * n_osc + 2 * n_osc + 1] = 0.04;
    }

    // Also test with random seed 0
    let drives_random = random_drives(200, n_osc, 0);

    let eps = 1e-3f32;
    println!("\n  Scanning n_steps (uniform drives):");
    for n_steps in [1, 2, 3, 5, 10, 20, 50, 100, 200] {
        let drives = &drives_base[..n_steps * 3 * n_osc];
        let (pst, traj) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            drives,
        );
        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0f32; pst_dim];
        grad_pst[0] = 1.0;
        grad_pst[2] = 1.0;
        let ana = crate::spike::t3spike_backward(&grad_pst, &traj);

        let idx = 0 * 3 * n_osc + 1 * n_osc + 0;
        let mut dp = drives.to_vec();
        dp[idx] += eps;
        let mut dm = drives.to_vec();
        dm[idx] -= eps;
        let (pp, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dp,
        );
        let (pm, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dm,
        );
        let fd = ((pp[0] + pp[2]) - (pm[0] + pm[2])) / (2.0 * eps);
        println!(
            "  n_steps={n_steps:3}: ana={:+.6e} fd={:+.6e} ratio={:+.3}",
            ana[idx],
            fd,
            ana[idx] / fd
        );
    }

    println!("\n  n_steps=200 with random seed 0 drives (n_osc=2):");
    for eps_val in [1e-2f32, 1e-3] {
        let n_steps = 200;
        let drives = &drives_random[..n_steps * 3 * n_osc];
        let (pst, traj) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            drives,
        );
        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0f32; pst_dim];
        grad_pst[0] = 1.0;
        grad_pst[2] = 1.0;
        let ana = crate::spike::t3spike_backward(&grad_pst, &traj);

        let idx = 0 * 3 * n_osc + 1 * n_osc + 0;
        let mut dp = drives.to_vec();
        dp[idx] += eps_val;
        let mut dm = drives.to_vec();
        dm[idx] -= eps_val;
        let (pp, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dp,
        );
        let (pm, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dm,
        );
        let fd = ((pp[0] + pp[2]) - (pm[0] + pm[2])) / (2.0 * eps_val);
        println!(
            "  eps={:.0e}: ana={:+.6e} fd={:+.6e} ratio={:+.3}",
            eps_val,
            ana[idx],
            fd,
            ana[idx] / fd
        );
    }

    // Check pop=2 with seed 2 for various n_steps
    println!("\n  Random seed 2, pop=2, scanning n_steps:");
    let drives_s2 = random_drives(200, n_osc, 2);
    for n_steps in [5, 10, 20, 50, 100, 200] {
        let drives = &drives_s2[..n_steps * 3 * n_osc];
        let (pst, traj) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            drives,
        );
        let pst_dim = pst_linear_dim(n_osc);
        let mut grad_pst = vec![0.0f32; pst_dim];
        for i in 0..n_osc {
            grad_pst[2 * i] = 1.0;
        }
        let ana = crate::spike::t3spike_backward(&grad_pst, &traj);

        let idx = 0 * 3 * n_osc + 2 * n_osc + 0;
        let eps_val = 1e-3f32;
        let mut dp = drives.to_vec();
        dp[idx] += eps_val;
        let mut dm = drives.to_vec();
        dm[idx] -= eps_val;
        let (pp, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dp,
        );
        let (pm, _) = t3spike_forward(
            n_osc,
            n_steps,
            0.05,
            drive_gain,
            alpha,
            0.5,
            3.0,
            coupling_w12,
            coupling_w13,
            coupling_w23,
            &dm,
        );
        let loss_p: f32 = (0..n_osc).map(|i| pp[2 * i]).sum();
        let loss_m: f32 = (0..n_osc).map(|i| pm[2 * i]).sum();
        let fd = (loss_p - loss_m) / (2.0 * eps_val);
        println!(
            "  n_steps={n_steps:3}: ana={:+.6e} fd={:+.6e} ratio={:+.3}",
            ana[idx],
            fd,
            ana[idx] / fd
        );
    }
}
