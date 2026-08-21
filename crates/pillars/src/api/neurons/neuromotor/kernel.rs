#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::zoo::neurons::NeuromotorModel;

/// Обработать чанк из 8 мышечных каналов через pillars-модель.
#[allow(clippy::too_many_arguments)]
#[inline]
fn process_chunk(
    ca: &mut [f32],
    tens: &mut [f32],
    spk: &[i32],
    rst: &[i32],
    d_ca: &[f32],
    tau_pump: &[f32],
    eta_reset: &[f32],
    t_max: &[f32],
    kd: &[f32],
    l0: &[f32],
    v_max: &[f32],
    dt: f32,
    lengths: Option<&[f32]>,
    off: usize,
    velocities: Option<&[f32]>,
) {
    // NeuromotorModel 9-col: [dt, d_ca, ca_decay_const, eta_reset,
    //   t_max, kd, l0, inv_vmax, inv_width]; state [ca, tension, reset_flag,
    //   length, velocity].
    // Легаси-семантика сохранена на границе: ca_decay_const = exp(-dt/τ_pump),
    // inv_vmax = 1/v_max, inv_width = 1/(l0·0.5); нет длины/скорости → слоты
    // = l0 / 0 (f_l = f_v = 1).
    let mut params = [0.0f32; 72];
    let mut i_in = [0.0f32; 8];
    let mut st_rst = [0.0f32; 8];
    let mut st_len = [0.0f32; 8];
    let mut st_vel = [0.0f32; 8];
    let mut spk_out = [0i32; 8];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = d_ca[j];
        params[16 + j] = (-dt / tau_pump[j].max(1e-6)).exp();
        params[24 + j] = eta_reset[j];
        params[32 + j] = t_max[j];
        params[40 + j] = kd[j];
        params[48 + j] = l0[j];
        params[56 + j] = 1.0 / v_max[j].max(1e-6);
        params[64 + j] = 1.0 / (l0[j] * 0.5).max(1e-6);
        i_in[j] = spk[j] as f32;
        st_rst[j] = rst[j] as f32;
        st_len[j] = lengths.map_or(l0[j], |l| l[off + j]);
        st_vel[j] = velocities.map_or(0.0, |v| v[off + j]);
    }
    crate::core::runner::run_neuron_slices::<NeuromotorModel>(
        crate::core::runner::NeuronState {
            v: Some(&mut ca[..]),
            u: Some(&mut tens[..]),
            h: Some(&mut st_rst[..]),
            r: Some(&mut st_len[..]),
            s: Some(&mut st_vel[..]),
            ..Default::default()
        },
        &i_in[..],
        &mut spk_out[..],
        &params[..],
    );
}

/// Обработать сустав `j` (agonista-antagonist tug-of-war).
/// Используется только в serial-ветке (rayon-ветки инлайн-zip).
#[cfg(not(feature = "rayon"))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn process_joint(
    j: usize,
    df: &mut [f32],
    stf: &mut [f32],
    dmp: Option<&mut [f32]>,
    k0: &[f32],
    beta: &[f32],
    b0: Option<&[f32]>,
    alpha: Option<&[f32]>,
    ca: &[f32],
    tension: &[f32],
) {
    let flex_idx = 2 * j;
    let ext_idx = 2 * j + 1;
    let ca_sum = ca[flex_idx] + ca[ext_idx];

    df[j] = tension[ext_idx] - tension[flex_idx];
    stf[j] = k0[j] + beta[j] * ca_sum;
    if let (Some(dmp), Some(b0), Some(alpha)) = (dmp, b0, alpha) {
        dmp[j] = b0[j] + alpha[j] * ca_sum;
    }
}

/// Step `n` Neuromotor Muscle Channels.
/// Parallelized via Rayon (или serial без feature), `n` must be a multiple of 8.
///
/// Joint Output (Optional):
/// - `joint_delta_f`: output net torque/force for `n / 2` joint pairs
/// - `joint_stiffness`: output dynamic joint stiffness for `n / 2` joint pairs
/// - `k0`: baseline joint stiffness per pair (`n / 2`)
/// - `beta`: calcium co-contraction stiffness gain per pair (`n / 2`)
pub fn step_neuromotor(
    ca: &mut [f32],
    tension: &mut [f32],
    spikes: &[i32],
    resets: &[i32],
    dt: f32,
    d_ca: &[f32],
    tau_pump: &[f32],
    eta_reset: &[f32],
    t_max: &[f32],
    kd: &[f32],
    l0: &[f32],
    v_max: &[f32],
    lengths: Option<&[f32]>,
    velocities: Option<&[f32]>,
    joint_delta_f: Option<&mut [f32]>,
    joint_stiffness: Option<&mut [f32]>,
    joint_damping: Option<&mut [f32]>,
    k0: Option<&[f32]>,
    beta: Option<&[f32]>,
    b0: Option<&[f32]>,
    alpha: Option<&[f32]>,
) {
    let n = ca.len();
    assert!(
        n.is_multiple_of(8)
            && n == tension.len()
            && n == spikes.len()
            && n == resets.len()
            && n == d_ca.len()
            && n == tau_pump.len()
            && n == eta_reset.len()
            && n == t_max.len()
            && n == kd.len()
            && n == l0.len()
            && n == v_max.len()
    );

    if let Some(l) = lengths {
        assert_eq!(l.len(), n);
    }
    if let Some(v) = velocities {
        assert_eq!(v.len(), n);
    }

    // 1. Process Muscle Channels (chunks of 8)
    #[cfg(feature = "rayon")]
    ca.par_chunks_exact_mut(8)
        .zip(tension.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact(8))
        .zip(resets.par_chunks_exact(8))
        .zip(d_ca.par_chunks_exact(8))
        .zip(tau_pump.par_chunks_exact(8))
        .zip(eta_reset.par_chunks_exact(8))
        .zip(t_max.par_chunks_exact(8))
        .zip(kd.par_chunks_exact(8))
        .zip(l0.par_chunks_exact(8))
        .zip(v_max.par_chunks_exact(8))
        .enumerate()
        .for_each(
            |(
                chunk_idx,
                (
                    (
                        (
                            (
                                ((((((ca8, tens8), spk8), rst8), d_ca8), tau_pump8), eta_reset8),
                                t_max8,
                            ),
                            kd8,
                        ),
                        l08,
                    ),
                    vmax8,
                ),
            )| {
                process_chunk(
                    ca8,
                    tens8,
                    spk8,
                    rst8,
                    d_ca8,
                    tau_pump8,
                    eta_reset8,
                    t_max8,
                    kd8,
                    l08,
                    vmax8,
                    dt,
                    lengths,
                    chunk_idx * 8,
                    velocities,
                );
            },
        );
    #[cfg(not(feature = "rayon"))]
    for chunk_idx in 0..n / 8 {
        let off = chunk_idx * 8;
        process_chunk(
            &mut ca[off..off + 8],
            &mut tension[off..off + 8],
            &spikes[off..off + 8],
            &resets[off..off + 8],
            &d_ca[off..off + 8],
            &tau_pump[off..off + 8],
            &eta_reset[off..off + 8],
            &t_max[off..off + 8],
            &kd[off..off + 8],
            &l0[off..off + 8],
            &v_max[off..off + 8],
            dt,
            lengths,
            off,
            velocities,
        );
    }

    // 2. Compute Joint Agonist-Antagonist Dynamics if joint buffers provided.
    // rayon-ветки — inline zip-замыкания, serial — process_joint: две копии
    // логики (rayon требует Fn-совместимых замыканий). При правке синхронизировать.
    if let (Some(df), Some(stf), Some(k0_arr), Some(beta_arr)) =
        (joint_delta_f, joint_stiffness, k0, beta)
    {
        let n_joints = n / 2;
        assert!(
            df.len() >= n_joints
                && stf.len() >= n_joints
                && k0_arr.len() >= n_joints
                && beta_arr.len() >= n_joints
        );

        if let (Some(dmp), Some(b0_arr), Some(alpha_arr)) = (joint_damping, b0, alpha) {
            #[cfg(feature = "rayon")]
            df.par_iter_mut()
                .zip(stf.par_iter_mut())
                .zip(dmp.par_iter_mut())
                .zip(k0_arr.par_iter())
                .zip(beta_arr.par_iter())
                .zip(b0_arr.par_iter())
                .zip(alpha_arr.par_iter())
                .enumerate()
                .for_each(
                    |(j, ((((((df_j, stf_j), dmp_j), k0_j), beta_j), b0_j), alpha_j))| {
                        let flex_idx = 2 * j;
                        let ext_idx = 2 * j + 1;
                        let ca_sum = ca[flex_idx] + ca[ext_idx];

                        *df_j = tension[ext_idx] - tension[flex_idx];
                        *stf_j = k0_j + beta_j * ca_sum;
                        *dmp_j = b0_j + alpha_j * ca_sum;
                    },
                );
            #[cfg(not(feature = "rayon"))]
            for j in 0..n_joints {
                process_joint(
                    j,
                    df,
                    stf,
                    Some(dmp),
                    k0_arr,
                    beta_arr,
                    Some(b0_arr),
                    Some(alpha_arr),
                    ca,
                    tension,
                );
            }
        } else {
            #[cfg(feature = "rayon")]
            df.par_iter_mut()
                .zip(stf.par_iter_mut())
                .zip(k0_arr.par_iter())
                .zip(beta_arr.par_iter())
                .enumerate()
                .for_each(|(j, (((df_j, stf_j), k0_j), beta_j))| {
                    let flex_idx = 2 * j;
                    let ext_idx = 2 * j + 1;
                    let ca_sum = ca[flex_idx] + ca[ext_idx];

                    *df_j = tension[ext_idx] - tension[flex_idx];
                    *stf_j = k0_j + beta_j * ca_sum;
                });
            #[cfg(not(feature = "rayon"))]
            for j in 0..n_joints {
                process_joint(j, df, stf, None, k0_arr, beta_arr, None, None, ca, tension);
            }
        }
    }
}
