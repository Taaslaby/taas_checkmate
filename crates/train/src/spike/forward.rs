//! Forward pass: drives → T3 (N steps) → PST features + trajectory.

use crate::t3::{T3Backend, Avx2Backend, TriOscillator, pst_linear_features};
use crate::spike::trajectory::{T3SpikeTrajectory, copy_slice};

const DEFAULT_EPSILON: f32 = 0.1;

/// Forward pass: drives → T3 (N steps) → PST features + trajectory.
pub fn t3spike_forward_with_backend<B: T3Backend>(
    backend: B,
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: &[f32],
) -> (Vec<f32>, T3SpikeTrajectory) {
    let mut tri = TriOscillator::new(n_osc, freq_min, freq_max, backend);
    tri.drive_gain = drive_gain;
    tri.alpha = alpha;
    tri.epsilon = DEFAULT_EPSILON;
    tri.set_uniform_coupling(coupling_w12, coupling_w13, coupling_w23);

    let step_size = 3 * n_osc;

    let mut traj = T3SpikeTrajectory {
        n_osc,
        n_steps,
        dt,
        drive_gain,
        alpha,
        epsilon: DEFAULT_EPSILON,
        freq_min,
        freq_max,
        w12: coupling_w12,
        w13: coupling_w13,
        w23: coupling_w23,
        omega1: tri.omega1.clone(),
        omega2: tri.omega2.clone(),
        omega3: tri.omega3.clone(),
        th1_pre: vec![0.0; n_steps * n_osc],
        th2_pre: vec![0.0; n_steps * n_osc],
        th3_pre: vec![0.0; n_steps * n_osc],
        r1_pre: vec![0.0; n_steps * n_osc],
        r2_pre: vec![0.0; n_steps * n_osc],
        r3_pre: vec![0.0; n_steps * n_osc],
        th1_final: vec![0.0; n_osc],
        th2_final: vec![0.0; n_osc],
        th3_final: vec![0.0; n_osc],
        r1_final: vec![0.0; n_osc],
        r2_final: vec![0.0; n_osc],
        r3_final: vec![0.0; n_osc],
        coupling1: vec![0.0; n_steps * n_osc],
        coupling2: vec![0.0; n_steps * n_osc],
        coupling3: vec![0.0; n_steps * n_osc],
        drive1: vec![0.0; n_steps * n_osc],
        drive2: vec![0.0; n_steps * n_osc],
        drive3: vec![0.0; n_steps * n_osc],
    };

    for step in 0..n_steps {
        let off = step * n_osc;
        let d_off = step * step_size;

        copy_slice(&mut traj.th1_pre[off..off + n_osc], &tri.th1[..n_osc]);
        copy_slice(&mut traj.th2_pre[off..off + n_osc], &tri.th2[..n_osc]);
        copy_slice(&mut traj.th3_pre[off..off + n_osc], &tri.th3[..n_osc]);
        copy_slice(&mut traj.r1_pre[off..off + n_osc], &tri.r1[..n_osc]);
        copy_slice(&mut traj.r2_pre[off..off + n_osc], &tri.r2[..n_osc]);
        copy_slice(&mut traj.r3_pre[off..off + n_osc], &tri.r3[..n_osc]);

        let d1 = &drives[d_off..d_off + n_osc];
        let d2 = &drives[d_off + n_osc..d_off + 2 * n_osc];
        let d3 = &drives[d_off + 2 * n_osc..d_off + 3 * n_osc];
        tri.drive1[..n_osc].copy_from_slice(d1);
        tri.drive2[..n_osc].copy_from_slice(d2);
        tri.drive3[..n_osc].copy_from_slice(d3);

        copy_slice(&mut traj.drive1[off..off + n_osc], d1);
        copy_slice(&mut traj.drive2[off..off + n_osc], d2);
        copy_slice(&mut traj.drive3[off..off + n_osc], d3);

        tri.step(dt);

        copy_slice(&mut traj.coupling1[off..off + n_osc], &tri.coupling1[..n_osc]);
        copy_slice(&mut traj.coupling2[off..off + n_osc], &tri.coupling2[..n_osc]);
        copy_slice(&mut traj.coupling3[off..off + n_osc], &tri.coupling3[..n_osc]);
    }

    copy_slice(&mut traj.th1_final, &tri.th1[..n_osc]);
    copy_slice(&mut traj.th2_final, &tri.th2[..n_osc]);
    copy_slice(&mut traj.th3_final, &tri.th3[..n_osc]);
    copy_slice(&mut traj.r1_final, &tri.r1[..n_osc]);
    copy_slice(&mut traj.r2_final, &tri.r2[..n_osc]);
    copy_slice(&mut traj.r3_final, &tri.r3[..n_osc]);

    let pst = pst_linear_features(&tri);
    (pst, traj)
}

/// Forward pass with default Avx2Backend (CPU SIMD).
pub fn t3spike_forward(
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: &[f32],
) -> (Vec<f32>, T3SpikeTrajectory) {
    t3spike_forward_with_backend(
        Avx2Backend,
        n_osc, n_steps, dt, drive_gain, alpha,
        freq_min, freq_max,
        coupling_w12, coupling_w13, coupling_w23,
        drives,
    )
}
