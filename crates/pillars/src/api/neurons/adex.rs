//! AdEx (Adaptive Exponential) neuron — step-адаптер.
//!
//! Канон: `step_adex` → `run_neuron::<AdExModel>` (12-col repacking).
//! E-Prop AdEx Multi — `super::eprop`. Перенесено из `core::neuron::adex`.

use crate::zoo::neurons::AdExModel;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Step `n` AdEx neurons. n must be multiple of 8.
///
/// Adaptive Exponential (AdEx) model dynamics:
///   dv/dt = (v_rest - v + Δ_T·exp((v - v_T)/Δ_T) - w + I) / τ_m
///   dw/dt = (a·(v - v_rest) - w) / τ_w
///   v ≥ v_thresh → spike, v = v_reset, w += b
///
/// AdExModel 12-col на чанк 8:
///   [dt, inv_tau_m, inv_tau_w, v_rest, v_thresh, v_reset, delta_t,
///    inv_delta_t, a, b, tau_ref, v_peak]
/// `v_peak := v_thresh` сохраняет легаси-семантику спайка при v ≥ v_thresh.
pub fn step_adex(
    v: &mut [f32],
    u: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    tau_w: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    delta_t: &[f32],
    a: &[f32],
    b: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == u.len()
            && n == I.len()
            && n == spikes.len()
            && n == tau_m.len()
            && n == tau_w.len()
            && n == v_rest.len()
            && n == v_thresh.len()
            && n == v_reset.len()
            && n == delta_t.len()
            && n == a.len()
            && n == b.len()
            && n == refractory.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((v8, u8), I8), sp8), refr8))| {
            process_chunk(
                chunk, v8, u8, I8, sp8, refr8, dt, tau_m, tau_w, v_rest, v_thresh, v_reset,
                delta_t, a, b, tau_ref,
            );
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let i = chunk * 8;
        process_chunk(
            chunk,
            &mut v[i..i + 8],
            &mut u[i..i + 8],
            &mut I[i..i + 8],
            &mut spikes[i..i + 8],
            &mut refractory[i..i + 8],
            dt,
            tau_m,
            tau_w,
            v_rest,
            v_thresh,
            v_reset,
            delta_t,
            a,
            b,
            tau_ref,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: usize,
    v8: &mut [f32],
    u8: &mut [f32],
    I8: &mut [f32],
    sp8: &mut [i32],
    refr8: &mut [f32],
    dt: f32,
    tau_m: &[f32],
    tau_w: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    delta_t: &[f32],
    a: &[f32],
    b: &[f32],
    tau_ref: f32,
) {
    let i = chunk * 8;
    let mut params = [0.0f32; 96];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = 1.0 / tau_m[i + j].max(1e-6);
        params[16 + j] = 1.0 / tau_w[i + j].max(1e-6);
        params[24 + j] = v_rest[i + j];
        params[32 + j] = v_thresh[i + j];
        params[40 + j] = v_reset[i + j];
        params[48 + j] = delta_t[i + j];
        params[56 + j] = 1.0 / delta_t[i + j].max(1e-6);
        params[64 + j] = a[i + j];
        params[72 + j] = b[i + j];
        params[80 + j] = tau_ref;
        params[88 + j] = v_thresh[i + j]; // v_peak := v_thresh
    }
    crate::core::runner::run_neuron_slices::<AdExModel>(
        crate::core::runner::NeuronState {
            v: Some(v8),
            u: Some(u8),
            h: Some(refr8), // state 2 = refractory
            ..Default::default()
        },
        &I8[..],
        sp8,
        &params,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adex_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let tau_m = vec![10.0; n];
        let tau_w = vec![30.0; n];
        let v_rest = vec![-70.0; n];
        let v_thresh = vec![-54.0; n];
        let v_reset = vec![-70.0; n];
        let delta_t = vec![2.0; n];
        let a = vec![0.02; n];
        let b = vec![0.2; n];
        let mut v = vec![-70.0f32; n];
        let mut u = vec![0.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];

        let mut total = 0i32;
        for _ in 0..200 {
            step_adex(
                &mut v,
                &mut u,
                &mut I,
                &mut spikes,
                dt,
                &tau_m,
                &tau_w,
                &v_rest,
                &v_thresh,
                &v_reset,
                &delta_t,
                &a,
                &b,
                &mut refractory,
                2.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx should fire with I=30");
    }
}
