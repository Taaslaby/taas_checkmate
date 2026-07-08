//! T3 trajectory recording + PST matrix computation.

/// Записанная траектория T3 для backward pass.
pub struct T3SpikeTrajectory {
    pub n_osc: usize,
    pub n_steps: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,
    pub omega1: Vec<f32>,
    pub omega2: Vec<f32>,
    pub omega3: Vec<f32>,
    pub th1_pre: Vec<f32>,
    pub th2_pre: Vec<f32>,
    pub th3_pre: Vec<f32>,
    pub r1_pre: Vec<f32>,
    pub r2_pre: Vec<f32>,
    pub r3_pre: Vec<f32>,
    pub th1_final: Vec<f32>,
    pub th2_final: Vec<f32>,
    pub th3_final: Vec<f32>,
    pub r1_final: Vec<f32>,
    pub r2_final: Vec<f32>,
    pub r3_final: Vec<f32>,
    pub coupling1: Vec<f32>,
    pub coupling2: Vec<f32>,
    pub coupling3: Vec<f32>,
    pub drive1: Vec<f32>,
    pub drive2: Vec<f32>,
    pub drive3: Vec<f32>,
}

pub fn copy_slice(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "copy_slice: length mismatch");
    dst.copy_from_slice(src);
}

/// Compute full PST feature matrix from trajectory in one pass (Rust-native, no Python overhead).
pub fn t3spike_compute_pst_matrix(traj: &T3SpikeTrajectory) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let pst_dim = crate::t3::pst_linear_dim(n);
    let nf = n as f32;
    let mut out = Vec::with_capacity(n_steps * pst_dim);

    for step in 0..n_steps {
        let (th1, r1, th2, r2, th3, r3): (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]);
        if step == n_steps - 1 {
            th1 = &traj.th1_final; r1 = &traj.r1_final;
            th2 = &traj.th2_final; r2 = &traj.r2_final;
            th3 = &traj.th3_final; r3 = &traj.r3_final;
        } else {
            let next = (step + 1) * n;
            th1 = &traj.th1_pre[next..next + n]; r1 = &traj.r1_pre[next..next + n];
            th2 = &traj.th2_pre[next..next + n]; r2 = &traj.r2_pre[next..next + n];
            th3 = &traj.th3_pre[next..next + n]; r3 = &traj.r3_pre[next..next + n];
        }
        let pre = step * n;
        let prev_th1 = &traj.th1_pre[pre..pre + n];
        let prev_th2 = &traj.th2_pre[pre..pre + n];
        let prev_th3 = &traj.th3_pre[pre..pre + n];

        let mut s1 = Vec::with_capacity(n); let mut c1 = Vec::with_capacity(n);
        let mut s2 = Vec::with_capacity(n); let mut c2 = Vec::with_capacity(n);
        let mut s3 = Vec::with_capacity(n); let mut c3 = Vec::with_capacity(n);
        for i in 0..n {
            s1.push(th1[i].sin()); c1.push(th1[i].cos());
            s2.push(th2[i].sin()); c2.push(th2[i].cos());
            s3.push(th3[i].sin()); c3.push(th3[i].cos());
        }

        for i in 0..n { out.push(s1[i]); out.push(c1[i]); }
        for i in 0..n { out.push(s2[i]); out.push(c2[i]); }
        for i in 0..n { out.push(s3[i]); out.push(c3[i]); }

        for (ph, prev) in [(&th1[..n], prev_th1), (&th2[..n], prev_th2), (&th3[..n], prev_th3)] {
            for i in 0..n {
                let mut d = ph[i] - prev[i];
                if d > std::f32::consts::PI { d -= 2.0 * std::f32::consts::PI; }
                else if d < -std::f32::consts::PI { d += 2.0 * std::f32::consts::PI; }
                out.push(d);
            }
        }

        out.extend_from_slice(r1);
        out.extend_from_slice(r2);
        out.extend_from_slice(r3);

        for i in 0..n { out.push(s1[i] * s1[i]); out.push(c1[i] * c1[i]); }
        for i in 0..n { out.push(s2[i] * s2[i]); out.push(c2[i] * c2[i]); }
        for i in 0..n { out.push(s3[i] * s3[i]); out.push(c3[i] * c3[i]); }

        let cross = |sa: &[f32], ca: &[f32], sb: &[f32], cb: &[f32]| {
            let (ss, cc): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
                (s + (sa[i] as f64 * cb[i] as f64 - ca[i] as f64 * sb[i] as f64),
                 c + (ca[i] as f64 * cb[i] as f64 + sa[i] as f64 * sb[i] as f64))
            });
            ((ss * ss + cc * cc).sqrt() / nf as f64) as f32
        };
        out.push(cross(&s1, &c1, &s2, &c2));
        out.push(cross(&s2, &c2, &s3, &c3));
        out.push(cross(&s1, &c1, &s3, &c3));

        let mu = |arr: &[f32]| arr.iter().sum::<f32>() / nf;
        out.push(mu(&s1)); out.push(mu(&c1)); out.push(mu(r1));
        out.push(mu(&s2)); out.push(mu(&c2)); out.push(mu(r2));
        out.push(mu(&s3)); out.push(mu(&c3)); out.push(mu(r3));
    }
    out
}

/// Fast PST feature matrix — sin(θ) + r per step (6N per step, no trig calls).
/// Ridge decoder: R² > 0.9999, 2× меньше полного PST.
pub fn t3spike_compute_pst_fast_matrix(traj: &T3SpikeTrajectory) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let dim = crate::t3::pst_fast_dim(n);
    let mut out = Vec::with_capacity(n_steps * dim);

    for step in 0..n_steps {
        let (th1, r1, th2, r2, th3, r3): (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]);
        if step == n_steps - 1 {
            th1 = &traj.th1_final; r1 = &traj.r1_final;
            th2 = &traj.th2_final; r2 = &traj.r2_final;
            th3 = &traj.th3_final; r3 = &traj.r3_final;
        } else {
            let next = (step + 1) * n;
            th1 = &traj.th1_pre[next..next + n]; r1 = &traj.r1_pre[next..next + n];
            th2 = &traj.th2_pre[next..next + n]; r2 = &traj.r2_pre[next..next + n];
            th3 = &traj.th3_pre[next..next + n]; r3 = &traj.r3_pre[next..next + n];
        }

        for i in 0..n { out.push(th1[i].sin()); }
        for i in 0..n { out.push(th2[i].sin()); }
        for i in 0..n { out.push(th3[i].sin()); }
        out.extend_from_slice(r1);
        out.extend_from_slice(r2);
        out.extend_from_slice(r3);
    }
    out
}
