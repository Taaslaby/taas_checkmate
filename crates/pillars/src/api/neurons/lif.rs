//! LIF (Leaky Integrate-and-Fire) + LIF-AT repacking-адаптеры.
//!
//! Канон: `step_lif` → `run_neuron::<LIFModel>`, `step_lif_at` →
//! `run_neuron::<LIFAtModel>` (волна 6; легаси `kernels::neuron::lif` удалён).
//! E-Prop LIF — `super::EPropLIF`. Перенесено из `core::neuron::lif`.

#![allow(non_snake_case)]

use crate::core::runner::run_neuron;
use crate::zoo::neurons::{LIFAtModel, LIFModel};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ─── LIF-AT ──────────────────────────────────────────────

/// Step `n` LIF-AT (LIF with Adaptive Threshold) neurons. n must be multiple of 8.
///
/// LIF-AT добавляет adaptive threshold offset θ:
/// dv/dt = (v_rest - v + R*I) / tau_m
/// v >= V_th + θ → spike, v = v_reset, θ += beta
/// dθ/dt = -θ / tau_theta
///
/// Канон — `run_neuron::<LIFAtModel>` (GLIF2). Лейаут на чанк 8 (column-major):
///   [dt, inv_tau_m, v_rest, v_thresh, v_reset, inv_tau_theta, beta, tau_ref]
/// `inv_tau_theta = 1e9` для τ_θ ≤ 0 эмулирует легаси-семантику (θ → 0).
pub fn step_lif_at(
    v: &mut [f32],
    theta: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_theta: &[f32],
    beta: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == theta.len()
            && n == I.len()
            && n == spikes.len()
            && n == tau_m.len()
            && n == v_rest.len()
            && n == v_thresh.len()
            && n == v_reset.len()
            && n == tau_theta.len()
            && n == beta.len()
            && n == refractory.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(theta.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((v8, th8), I8), sp8), refr8))| {
            process_chunk_lifat(
                chunk, v8, th8, I8, sp8, refr8, dt, tau_m, v_rest, v_thresh, v_reset, tau_theta,
                beta, tau_ref,
            );
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let i = chunk * 8;
        process_chunk_lifat(
            chunk,
            &mut v[i..i + 8],
            &mut theta[i..i + 8],
            &mut I[i..i + 8],
            &mut spikes[i..i + 8],
            &mut refractory[i..i + 8],
            dt,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_theta,
            beta,
            tau_ref,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk_lifat(
    chunk: usize,
    v8: &mut [f32],
    th8: &mut [f32],
    I8: &mut [f32],
    sp8: &mut [i32],
    refr8: &mut [f32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_theta: &[f32],
    beta: &[f32],
    tau_ref: f32,
) {
    let i = chunk * 8;
    let mut params = [0.0f32; 64];
    for j in 0..8 {
        params[j] = dt;
        let tm = tau_m[i + j];
        params[8 + j] = if tm > 0.0 { 1.0 / tm } else { 0.0 };
        params[16 + j] = v_rest[i + j];
        params[24 + j] = v_thresh[i + j];
        params[32 + j] = v_reset[i + j];
        let tt = tau_theta[i + j];
        params[40 + j] = if tt > 0.0 { 1.0 / tt } else { 1e9 };
        params[48 + j] = beta[i + j];
        params[56 + j] = tau_ref;
    }
    unsafe {
        run_neuron::<LIFAtModel>(
            v8.as_mut_ptr(),
            th8.as_mut_ptr(),
            refr8.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I8.as_ptr(),
            sp8.as_mut_ptr(),
            params.as_ptr(),
            8,
        );
    }
}

// ─── LIF ─────────────────────────────────────────────────

/// Step `n` LIF neurons. n must be multiple of 8.
///
/// Legacy 5-col [dt, tau_m, v_rest, v_thresh, v_reset] → LIFModel 6-col
/// [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref] (транш 3b).
pub fn step_lif(
    v: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == I.len()
            && n == spikes.len()
            && n == tau_m.len()
            && n == v_rest.len()
            && n == v_thresh.len()
            && n == v_reset.len()
            && n == refractory.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, (((v8, I8), sp8), refr8))| {
            process_chunk_lif(
                chunk, v8, I8, sp8, refr8, dt, tau_m, v_rest, v_thresh, v_reset, tau_ref,
            );
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let i = chunk * 8;
        process_chunk_lif(
            chunk,
            &mut v[i..i + 8],
            &mut I[i..i + 8],
            &mut spikes[i..i + 8],
            &mut refractory[i..i + 8],
            dt,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_ref,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk_lif(
    chunk: usize,
    v8: &mut [f32],
    I8: &mut [f32],
    sp8: &mut [i32],
    refr8: &mut [f32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_ref: f32,
) {
    let i = chunk * 8;
    let mut params = [0.0f32; 48];
    for j in 0..8 {
        params[j] = dt;
        let tm = tau_m[i + j];
        params[8 + j] = if tm > 0.0 { 1.0 / tm } else { 0.0 };
        params[16 + j] = v_rest[i + j];
        params[24 + j] = v_thresh[i + j];
        params[32 + j] = v_reset[i + j];
        params[40 + j] = tau_ref;
    }
    unsafe {
        run_neuron::<LIFModel>(
            v8.as_mut_ptr(),
            refr8.as_mut_ptr(), // state 1 (refractory)
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I8.as_ptr(),
            sp8.as_mut_ptr(),
            params.as_ptr(),
            8,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lif_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let tau_m = vec![10.0; n];
        let v_rest = vec![-70.0; n];
        let v_thresh = vec![-54.0; n];
        let v_reset = vec![-70.0; n];
        let mut v = vec![-70.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];

        let mut total = 0i32;
        for _ in 0..100 {
            step_lif(
                &mut v,
                &mut I,
                &mut spikes,
                dt,
                &tau_m,
                &v_rest,
                &v_thresh,
                &v_reset,
                &mut refractory,
                2.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "LIF should fire with I=30");
    }

    #[test]
    fn lif_at_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let tau_m = vec![10.0; n];
        let v_rest = vec![-70.0; n];
        let v_thresh = vec![-54.0; n];
        let v_reset = vec![-70.0; n];
        let tau_theta = vec![100.0; n];
        let beta = vec![0.1; n];
        let mut v = vec![-70.0f32; n];
        let mut theta = vec![0.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];

        let mut total = 0i32;
        for _ in 0..200 {
            step_lif_at(
                &mut v,
                &mut theta,
                &mut I,
                &mut spikes,
                dt,
                &tau_m,
                &v_rest,
                &v_thresh,
                &v_reset,
                &tau_theta,
                &beta,
                &mut refractory,
                2.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "LIF-AT should fire with I=30");
    }
}
