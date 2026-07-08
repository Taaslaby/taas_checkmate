//! Backward pass: BPTT through T3 Stuart-Landau + Kuramoto dynamics.
//!
//! Forward dynamics (per oscillator i, per population):
//! ```text
//! r' = r + dt · ((α - r²)·r + drive · gain)
//! θ' = θ + dt · (ω + coupling + drive · gain / (r + ε))
//! ```
//!
//! coupling1[i] = w12·g2·Σ_j r2[j]·sin(θ2[j] - θ1[i]) + w13·g3·Σ_j r3[j]·sin(θ3[j] - θ1[i])
//! coupling2[i] = w12·g1·Σ_j r1[j]·sin(θ1[j] - θ2[i]) + w23·g3·Σ_j r3[j]·sin(θ3[j] - θ2[i])
//! coupling3[i] = w13·g1·Σ_j r1[j]·sin(θ1[j] - θ3[i]) + w23·g2·Σ_j r2[j]·sin(θ2[j] - θ3[i])
//! where g₁, g₂, g₃ = phase_gate(order_parameter R)
//!
//! Backward: dθ'/dθ = 1 + dt·dcpl/dθ
//! Cross-pop: vθ_other[j] += dt·vθ_self[i]·d(coupling_self[i])/dθ_other[j]

use crate::t3::pst_linear_dim;
use crate::spike::pst_jacobian::{pst_jacobian, compute_pst_jacobian_v};
use crate::spike::trajectory::T3SpikeTrajectory;

const EPS: f32 = 0.1;
const PG_GAIN: f32 = 4.0;

/// Backward pass: BPTT through T3 dynamics (final step only).
pub fn t3spike_backward(grad_pst: &[f32], traj: &T3SpikeTrajectory) -> Vec<f32> {
    t3spike_backward_per_step(grad_pst, traj, None)
}

/// Backward pass with optional per-step PST gradients for full-trajectory SPIGRAD.
pub fn t3spike_backward_per_step(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    per_step_grad_pst: Option<&[f32]>,
) -> Vec<f32> {
    let n = traj.n_osc;
    let pst_dim = pst_linear_dim(n);

    let (d_th1_j, d_th2_j, d_th3_j, d_r1_j, d_r2_j, d_r3_j) = pst_jacobian(
        &traj.th1_final, &traj.r1_final,
        &traj.th2_final, &traj.r2_final,
        &traj.th3_final, &traj.r3_final, n,
    );

    let mut v = compute_pst_jacobian_v(grad_pst, &d_th1_j, &d_th2_j, &d_th3_j,
                                        &d_r1_j, &d_r2_j, &d_r3_j, n, pst_dim);

    bptt_loop(grad_pst, traj, &mut v.th1, &mut v.th2, &mut v.th3,
              &mut v.r1, &mut v.r2, &mut v.r3, per_step_grad_pst)
}

/// Phase gate with per-population gamma threshold.
/// G(R) = 1/(1 + exp(-PG_GAIN * (R - gamma/sqrt(n))))
fn phase_gate(sin: &[f32], cos: &[f32], n: usize, gamma: f32) -> f32 {
    let (ss, sc): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
        (s + sin[i] as f64, c + cos[i] as f64)
    });
    let r = ((ss * ss + sc * sc).sqrt() / n as f64) as f32;
    let threshold = gamma / (n as f32).sqrt();
    1.0 / (1.0 + (-PG_GAIN * (r - threshold)).exp())
}

fn r_clamp_effect(r: f32, drive: f32, dt: f32, gain: f32, alpha: f32) -> f32 {
    // Compute post-update r to detect clamp activity. Use f64 for precision.
    let rp = r as f64 + (dt as f64) * ((alpha as f64 - r as f64 * r as f64) * r as f64
                                        + drive as f64 * gain as f64);
    if rp < 0.001 || rp > 10.0 { 0.0 } else { 1.0 }
}

// Gate only the r-factor (dr'/dr) term in vr_out.
// The cross-term dθ'/dr should NEVER be gated — r_pre affects θ through the dynamics
// even when the r-output is clamped.
fn r_factor_gate(r: f32, drive: f32, dt: f32, gain: f32, alpha: f32) -> f32 {
    r_clamp_effect(r, drive, dt, gain, alpha)
}

/// Inner BPTT loop.
pub fn bptt_loop(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    v_th1: &mut Vec<f32>,
    v_th2: &mut Vec<f32>,
    v_th3: &mut Vec<f32>,
    v_r1: &mut Vec<f32>,
    v_r2: &mut Vec<f32>,
    v_r3: &mut Vec<f32>,
    per_step_grad_pst: Option<&[f32]>,
) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let dt = traj.dt;
    let gain = traj.drive_gain;
    let alpha = traj.alpha;
    let pst_dim = pst_linear_dim(n);
    let nf = n as f64;

    let mut sin1 = vec![0.0; n]; let mut cos1 = vec![0.0; n];
    let mut sin2 = vec![0.0; n]; let mut cos2 = vec![0.0; n];
    let mut sin3 = vec![0.0; n]; let mut cos3 = vec![0.0; n];
    let mut grad_drives = vec![0.0; n_steps * 3 * n];

    for step in (0..n_steps).rev() {
        let off = step * n;
        let d_off = step * 3 * n;

        let pre_th1 = &traj.th1_pre[off..off + n];
        let pre_th2 = &traj.th2_pre[off..off + n];
        let pre_th3 = &traj.th3_pre[off..off + n];
        let pre_r1 = &traj.r1_pre[off..off + n];
        let pre_r2 = &traj.r2_pre[off..off + n];
        let pre_r3 = &traj.r3_pre[off..off + n];
        let d1 = &traj.drive1[off..off + n];
        let d2 = &traj.drive2[off..off + n];
        let d3 = &traj.drive3[off..off + n];

        for i in 0..n {
            sin1[i] = pre_th1[i].sin(); cos1[i] = pre_th1[i].cos();
            sin2[i] = pre_th2[i].sin(); cos2[i] = pre_th2[i].cos();
            sin3[i] = pre_th3[i].sin(); cos3[i] = pre_th3[i].cos();
        }

        let g1 = phase_gate(&sin1, &cos1, n, 0.5) as f64;
        let g2 = phase_gate(&sin2, &cos2, n, 1.0) as f64;
        let g3 = phase_gate(&sin3, &cos3, n, 1.12) as f64;

        // Amplitude-weighted sums: S_rc = Σ r[j]·cos[j], S_rs = Σ r[j]·sin[j]
        let sum = |r: &[f32], sn: &[f32], cs: &[f32]| -> (f64, f64) {
            (0..n).fold((0.0, 0.0), |(s, c), i| {
                (s + r[i] as f64 * sn[i] as f64, c + r[i] as f64 * cs[i] as f64)
            })
        };
        let (r1s1, r1c1) = sum(pre_r1, &sin1, &cos1);
        let (r2s2, r2c2) = sum(pre_r2, &sin2, &cos2);
        let (r3s3, r3c3) = sum(pre_r3, &sin3, &cos3);

        let w12 = traj.w12 as f64 / nf;
        let w13 = traj.w13 as f64 / nf;
        let w23 = traj.w23 as f64 / nf;

        // ── Add per-step PST gradient (SPIGRAD) ──
        // PST at step t uses POST-state (θ_post[t], r_post[t]) for sin/cos, r, etc.
        // For step N-1: θ_post = θ_final. For t < N-1: θ_post[t] = θ_pre[t+1].
        if let Some(ps) = per_step_grad_pst {
            let base = step * pst_dim;
            if (base..base + pst_dim).any(|k| ps[k] != 0.0) {
                let (post_th1, post_r1, post_th2, post_r2, post_th3, post_r3) = if step == n_steps - 1 {
                    (&traj.th1_final[..], &traj.r1_final[..],
                     &traj.th2_final[..], &traj.r2_final[..],
                     &traj.th3_final[..], &traj.r3_final[..])
                } else {
                    let next_off = (step + 1) * n;
                    (&traj.th1_pre[next_off..next_off + n], &traj.r1_pre[next_off..next_off + n],
                     &traj.th2_pre[next_off..next_off + n], &traj.r2_pre[next_off..next_off + n],
                     &traj.th3_pre[next_off..next_off + n], &traj.r3_pre[next_off..next_off + n])
                };
                let (d_th1_s, d_th2_s, d_th3_s, d_r1_s, d_r2_s, d_r3_s) = pst_jacobian(
                    post_th1, post_r1, post_th2, post_r2, post_th3, post_r3, n,
                );
                let vp = compute_pst_jacobian_v(
                    &ps[base..base + pst_dim], &d_th1_s, &d_th2_s, &d_th3_s,
                    &d_r1_s, &d_r2_s, &d_r3_s, n, pst_dim,
                );
                for i in 0..n {
                    v_th1[i] += vp.th1[i]; v_th2[i] += vp.th2[i]; v_th3[i] += vp.th3[i];
                    v_r1[i] += vp.r1[i]; v_r2[i] += vp.r2[i]; v_r3[i] += vp.r3[i];
                }
            }
        }

        // ── Pre-compute cross sums: Σ vθ[k]·cos[k] and Σ vθ[k]·sin[k] ──
        let sum_v = |vth: &[f32], sn: &[f32], cs: &[f32]| -> (f64, f64) {
            (0..n).fold((0.0, 0.0), |(s, c), i| {
                (s + vth[i] as f64 * sn[i] as f64, c + vth[i] as f64 * cs[i] as f64)
            })
        };
        let (svs1, svc1) = sum_v(v_th1, &sin1, &cos1);
        let (svs2, svc2) = sum_v(v_th2, &sin2, &cos2);
        let (svs3, svc3) = sum_v(v_th3, &sin3, &cos3);

        // ── Phase gate derivative pre-factors ──
        // Gate gradient dg₁/dθ₁[i] = PG_GAIN · g₁·(1-g₁) · (S₁·cosθ₁[i] - C₁·sinθ₁[i]) / (n²·R₁)
        // where S₁ = Σ sinθ₁[j], C₁ = Σ cosθ₁[j], R₁ = |order parameter|
        // The cross-pop gate contribution to vθ₁[i] from coupling₂, coupling₃ via g₁:
        //   dt · dg₁/dθ₁[i] · (w₁₂/n · (r₁s₁·svc₂ - r₁c₁·svs₂) + w₁₃/n · (r₁s₁·svc₃ - r₁c₁·svs₃))
        let sum_sin_cos = |sn: &[f32], cs: &[f32]| -> (f64, f64) {
            (0..n).fold((0.0, 0.0), |(s, c), i| (s + sn[i] as f64, c + cs[i] as f64))
        };
        let (sin1_sum, cos1_sum) = sum_sin_cos(&sin1, &cos1);
        let (sin2_sum, cos2_sum) = sum_sin_cos(&sin2, &cos2);
        let (sin3_sum, cos3_sum) = sum_sin_cos(&sin3, &cos3);
        let r1_op = (sin1_sum * sin1_sum + cos1_sum * cos1_sum).sqrt() / nf;
        let r2_op = (sin2_sum * sin2_sum + cos2_sum * cos2_sum).sqrt() / nf;
        let r3_op = (sin3_sum * sin3_sum + cos3_sum * cos3_sum).sqrt() / nf;
        let pg_pre1 = if r1_op > 1e-10 { PG_GAIN as f64 * g1 * (1.0 - g1) / (nf * nf * r1_op) } else { 0.0 };
        let pg_pre2 = if r2_op > 1e-10 { PG_GAIN as f64 * g2 * (1.0 - g2) / (nf * nf * r2_op) } else { 0.0 };
        let pg_pre3 = if r3_op > 1e-10 { PG_GAIN as f64 * g3 * (1.0 - g3) / (nf * nf * r3_op) } else { 0.0 };
        let cross_gate1 = w12 * (r1s1 * svc2 - r1c1 * svs2) + w13 * (r1s1 * svc3 - r1c1 * svs3);
        let cross_gate2 = w12 * (r2s2 * svc1 - r2c2 * svs1) + w23 * (r2s2 * svc3 - r2c2 * svs3);
        let cross_gate3 = w13 * (r3s3 * svc1 - r3c3 * svs1) + w23 * (r3s3 * svc2 - r3c3 * svs2);

        // ── Drive gradient ──
        for i in 0..n {
            let dth_dd = |r: f32| dt * gain / (r + EPS);
            grad_drives[d_off + i] = v_th1[i] * dth_dd(pre_r1[i]) + v_r1[i] * dt * gain * r_clamp_effect(pre_r1[i], d1[i], dt, gain, alpha);
            grad_drives[d_off + n + i] = v_th2[i] * dth_dd(pre_r2[i]) + v_r2[i] * dt * gain * r_clamp_effect(pre_r2[i], d2[i], dt, gain, alpha);
            grad_drives[d_off + 2*n + i] = v_th3[i] * dth_dd(pre_r3[i]) + v_r3[i] * dt * gain * r_clamp_effect(pre_r3[i], d3[i], dt, gain, alpha);
        }

        // ── BPTT step: compute v_prev ──
        let dtf = dt as f64;
        for i in 0..n {
            let s1 = sin1[i] as f64; let c1 = cos1[i] as f64;
            let s2 = sin2[i] as f64; let c2 = cos2[i] as f64;
            let s3 = sin3[i] as f64; let c3 = cos3[i] as f64;
            let r1 = pre_r1[i] as f64; let r2 = pre_r2[i] as f64; let r3 = pre_r3[i] as f64;
            let d1v = d1[i] as f64; let d2v = d2[i] as f64; let d3v = d3[i] as f64;

            // Coupling self-derivative: d(coupling)/dθ_self
            // coupling1[i] depends on θ1[i] (self) and θ2, θ3 (other)
            let dcpl1 = -(w12 * g2 * (c1 * r2c2 + s1 * r2s2) + w13 * g3 * (c1 * r3c3 + s1 * r3s3));
            let dcpl2 = -(w12 * g1 * (c2 * r1c1 + s2 * r1s1) + w23 * g3 * (c2 * r3c3 + s2 * r3s3));
            let dcpl3 = -(w13 * g1 * (c3 * r1c1 + s3 * r1s1) + w23 * g2 * (c3 * r2c2 + s3 * r2s2));

            // θ v_prev: self + cross-pop coupling contributions
            // Self: dθ'/dθ = 1 + dt·dcpl
            let vt1_self = v_th1[i] as f64 * (1.0 + dtf * dcpl1);
            let vt2_self = v_th2[i] as f64 * (1.0 + dtf * dcpl2);
            let vt3_self = v_th3[i] as f64 * (1.0 + dtf * dcpl3);

            // Cross-pop: θ1[i] affects coupling2[j] and coupling3[j] for ALL j
            // vθ1[i] += dt · Σ_j vθ2[j] · d(coupling2[j])/dθ1[i] + dt · Σ_j vθ3[j] · d(coupling3[j])/dθ1[i]
            // coupling2[j] has θ1-θ2 term: d(coupling2[j])/dθ1[i] = w12·g1·r1[i]·cos(θ1[i]-θ2[j])
            // coupling3[j] has θ1-θ3 term: d(coupling3[j])/dθ1[i] = w13·g1·r1[i]·cos(θ1[i]-θ3[j])
            // PLUS gate derivative: dg₁/dθ₁[i] · (w12/n)·(cosθ₂[j]·r1s₁ - sinθ₂[j]·r1c₁)
            let dg1 = pg_pre1 * (sin1_sum * c1 - cos1_sum * s1);
            let vt1_cross = dtf * (
                r1 * (w12 * g1 * (c1 * svc2 + s1 * svs2) + w13 * g1 * (c1 * svc3 + s1 * svs3))
                + dg1 * cross_gate1
            );
            // θ2[i] affects coupling1[j] and coupling3[j]
            let dg2 = pg_pre2 * (sin2_sum * c2 - cos2_sum * s2);
            let vt2_cross = dtf * (
                r2 * (w12 * g2 * (c2 * svc1 + s2 * svs1) + w23 * g2 * (c2 * svc3 + s2 * svs3))
                + dg2 * cross_gate2
            );
            // θ3[i] affects coupling1[j] and coupling2[j]
            let dg3 = pg_pre3 * (sin3_sum * c3 - cos3_sum * s3);
            let vt3_cross = dtf * (
                r3 * (w13 * g3 * (c3 * svc1 + s3 * svs1) + w23 * g3 * (c3 * svc2 + s3 * svs2))
                + dg3 * cross_gate3
            );

            // r v_prev:
            // r' = r + dt·((α - r²)·r + drive·gain)
            // dr'/dr = 1 + dt·(α - 3r²)   — gated when r' would be clamped
            // dθ'/dr = -dt·drive·gain/(r+ε)²   — NEVER gated (r_pre affects θ thru dynamics)
            let r_factor1 = 1.0 + dtf * (alpha as f64 - 3.0 * r1 * r1);
            let rg1 = r_factor_gate(pre_r1[i], d1[i], dt, gain, alpha) as f64;

            let vr1_out = v_r1[i] as f64 * r_factor1 * rg1
                          + v_th1[i] as f64 * (-dtf * d1v * (gain as f64) / ((r1 + EPS as f64).powi(2)));
            let rg2 = r_factor_gate(pre_r2[i], d2[i], dt, gain, alpha) as f64;
            let vr2_out = v_r2[i] as f64 * (1.0 + dtf * (alpha as f64 - 3.0 * r2 * r2)) * rg2
                          + v_th2[i] as f64 * (-dtf * d2v * (gain as f64) / ((r2 + EPS as f64).powi(2)));
            let rg3 = r_factor_gate(pre_r3[i], d3[i], dt, gain, alpha) as f64;
            let vr3_out = v_r3[i] as f64 * (1.0 + dtf * (alpha as f64 - 3.0 * r3 * r3)) * rg3
                          + v_th3[i] as f64 * (-dtf * d3v * (gain as f64) / ((r3 + EPS as f64).powi(2)));

            v_th1[i] = (vt1_self + vt1_cross) as f32;
            v_th2[i] = (vt2_self + vt2_cross) as f32;
            v_th3[i] = (vt3_self + vt3_cross) as f32;
            v_r1[i] = vr1_out as f32;
            v_r2[i] = vr2_out as f32;
            v_r3[i] = vr3_out as f32;
        }

        // ── Velocity correction (direct d(vel)/dθ_pre = -1) ──
        // vel_i = θ_post_i - θ_pre_i → d(vel_i)/d(θ_pre_i) = -1
        // Applied AFTER BPTT because it bypasses dynamics (direct θ_pre dependence).
        if step == n_steps - 1 {
            for i in 0..n {
                v_th1[i] -= grad_pst[6 * n + i];
                v_th2[i] -= grad_pst[7 * n + i];
                v_th3[i] -= grad_pst[8 * n + i];
            }
        }
    }

    grad_drives
}
