//! Forward pass: drives → T3 (N steps) → PST features + trajectory.

use crate::t3::{T3Backend, Avx2Backend, TriOscillator, pst_linear_features};
use crate::spike::trajectory::{T3SpikeTrajectory, copy_slice, OFF_TH1, off_th2, off_th3, off_r1, off_r2, off_r3, off_drive1, off_drive2, off_drive3};

/// Helper macro: get a step slice from flat trajectory data
#[macro_export]
macro_rules! step_slice {
    ($data:expr, $off:expr, $n:expr, $step:expr, $n_steps:expr) => {
        &$data[$off + $step * $n..$off + ($step + 1) * $n]
    };
}
#[macro_export]
macro_rules! step_slice_mut {
    ($data:expr, $off:expr, $n:expr, $step:expr, $n_steps:expr) => {
        &mut $data[$off + $step * $n..$off + ($step + 1) * $n]
    };
}

const DEFAULT_EPSILON: f32 = 0.1;

/// Forward pass: drives → T3 (N steps) → PST features + trajectory.
///
/// Zero-alloc trajectory construction: writes directly into pre-allocated flat Vec.
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

    let mut traj = T3SpikeTrajectory::new(
        n_osc, n_steps, dt, drive_gain, alpha, DEFAULT_EPSILON,
        freq_min, freq_max,
        coupling_w12, coupling_w13, coupling_w23,
    );

    for step in 0..n_steps {
        let _off = step * n_osc;
        let d_off = step * step_size;

        // Write pre-state into flat trajectory
        copy_slice(step_slice_mut!(traj.flat, OFF_TH1, n_osc, step, n_steps), &tri.th1[..n_osc]);
        copy_slice(step_slice_mut!(traj.flat, off_th2(n_steps, n_osc), n_osc, step, n_steps), &tri.th2[..n_osc]);
        copy_slice(step_slice_mut!(traj.flat, off_th3(n_steps, n_osc), n_osc, step, n_steps), &tri.th3[..n_osc]);
        copy_slice(step_slice_mut!(traj.flat, off_r1(n_steps, n_osc), n_osc, step, n_steps), &tri.r1[..n_osc]);
        copy_slice(step_slice_mut!(traj.flat, off_r2(n_steps, n_osc), n_osc, step, n_steps), &tri.r2[..n_osc]);
        copy_slice(step_slice_mut!(traj.flat, off_r3(n_steps, n_osc), n_osc, step, n_steps), &tri.r3[..n_osc]);

        let d1 = &drives[d_off..d_off + n_osc];
        let d2 = &drives[d_off + n_osc..d_off + 2 * n_osc];
        let d3 = &drives[d_off + 2 * n_osc..d_off + 3 * n_osc];
        tri.drive1[..n_osc].copy_from_slice(d1);
        tri.drive2[..n_osc].copy_from_slice(d2);
        tri.drive3[..n_osc].copy_from_slice(d3);

        copy_slice(step_slice_mut!(traj.flat, off_drive1(n_steps, n_osc), n_osc, step, n_steps), d1);
        copy_slice(step_slice_mut!(traj.flat, off_drive2(n_steps, n_osc), n_osc, step, n_steps), d2);
        copy_slice(step_slice_mut!(traj.flat, off_drive3(n_steps, n_osc), n_osc, step, n_steps), d3);

        tri.step(dt);
    }

    copy_slice(&mut traj.th1_final, &tri.th1[..n_osc]);
    copy_slice(&mut traj.th2_final, &tri.th2[..n_osc]);
    copy_slice(&mut traj.th3_final, &tri.th3[..n_osc]);
    copy_slice(&mut traj.r1_final, &tri.r1[..n_osc]);
    copy_slice(&mut traj.r2_final, &tri.r2[..n_osc]);
    copy_slice(&mut traj.r3_final, &tri.r3[..n_osc]);

    tri.prepare();
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
