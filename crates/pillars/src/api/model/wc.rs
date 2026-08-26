//! Wilson-Cowan (WC) population model — excitatory-inhibitory rate dynamics.
//!
//! Канон: `step_wc` → `kernels::wc::wc_step` (волна 7, bit-identical порт).
//! Перенесено из `core::neuron::wc`.
//!
//!  вилсон кованый ты родился слишком рано

#![allow(non_snake_case)]

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Step `n` Wilson-Cowan populations. n must be multiple of 8.
///
/// Each "neuron" is actually a population with a single E and I rate.
/// Output is the sigmoid-activated rates (0 to 1), not spikes.
///
/// Parameters per population:
/// - `dt`: time step (ms), uniform across all populations
/// - `tau_E`: excitatory time constant (ms)
/// - `tau_I`: inhibitory time constant (ms)
/// - `w_EE`: self-excitation weight
/// - `w_EI`: inhibition-to-excitation weight
/// - `w_IE`: excitation-to-inhibition weight
/// - `w_II`: self-inhibition weight
/// - `beta`: sigmoid steepness (default 4.0)
pub fn step_wc(
    e: &mut [f32],
    i: &mut [f32],
    i_e: &[f32],
    i_i: &[f32],
    spikes_e: &mut [f32],
    spikes_i: &mut [f32],
    dt: f32,
    tau_E: &[f32],
    tau_I: &[f32],
    w_EE: &[f32],
    w_EI: &[f32],
    w_IE: &[f32],
    w_II: &[f32],
    beta: &[f32],
) {
    let n = e.len();
    assert!(
        n.is_multiple_of(8)
            && n == i.len()
            && n == i_e.len()
            && n == i_i.len()
            && n == spikes_e.len()
            && n == spikes_i.len()
            && n == tau_E.len()
            && n == tau_I.len()
            && n == w_EE.len()
            && n == w_EI.len()
            && n == w_IE.len()
            && n == w_II.len()
            && n == beta.len()
    );

    #[cfg(feature = "rayon")]
    e.par_chunks_exact_mut(8)
        .zip(i.par_chunks_exact_mut(8))
        .zip(i_e.par_chunks_exact(8))
        .zip(i_i.par_chunks_exact(8))
        .zip(spikes_e.par_chunks_exact_mut(8))
        .zip(spikes_i.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, (((((e8, i8), ie8), ii8), se8), si8))| {
            process_chunk(
                chunk, e8, i8, ie8, ii8, se8, si8, dt, tau_E, tau_I, w_EE, w_EI, w_IE, w_II, beta,
            );
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let off = chunk * 8;
        process_chunk(
            chunk,
            &mut e[off..off + 8],
            &mut i[off..off + 8],
            &i_e[off..off + 8],
            &i_i[off..off + 8],
            &mut spikes_e[off..off + 8],
            &mut spikes_i[off..off + 8],
            dt,
            tau_E,
            tau_I,
            w_EE,
            w_EI,
            w_IE,
            w_II,
            beta,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: usize,
    e8: &mut [f32],
    i8: &mut [f32],
    ie8: &[f32],
    ii8: &[f32],
    se8: &mut [f32],
    si8: &mut [f32],
    dt: f32,
    tau_E: &[f32],
    tau_I: &[f32],
    w_EE: &[f32],
    w_EI: &[f32],
    w_IE: &[f32],
    w_II: &[f32],
    beta: &[f32],
) {
    let idx = chunk * 8;
    let mut params = [0.0f32; 64];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = tau_E[idx + j];
        params[16 + j] = tau_I[idx + j];
        params[24 + j] = w_EE[idx + j];
        params[32 + j] = w_EI[idx + j];
        params[40 + j] = w_IE[idx + j];
        params[48 + j] = w_II[idx + j];
        params[56 + j] = beta[idx + j];
    }
    crate::kernels::wc::wc_step(e8, i8, ie8, ii8, se8, si8, &params);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wc_wrapper_converges() {
        let n = 16;
        let dt = 0.1;
        let tau_E = vec![10.0; n];
        let tau_I = vec![10.0; n];
        let w_EE = vec![1.2; n];
        let w_EI = vec![0.4; n];
        let w_IE = vec![0.6; n];
        let w_II = vec![0.8; n];
        let beta = vec![4.0; n];
        let i_e = vec![0.5; n];
        let i_i = vec![0.3; n];

        let mut e = vec![0.0f32; n];
        let mut ii = vec![0.0f32; n];
        let mut spikes_e = vec![0.0f32; n];
        let mut spikes_i = vec![0.0f32; n];

        for _ in 0..500 {
            step_wc(
                &mut e,
                &mut ii,
                &i_e,
                &i_i,
                &mut spikes_e,
                &mut spikes_i,
                dt,
                &tau_E,
                &tau_I,
                &w_EE,
                &w_EI,
                &w_IE,
                &w_II,
                &beta,
            );
        }

        for idx in 0..n {
            assert!(e[idx] > 0.0 && e[idx] < 1.0, "E[{idx}]={}", e[idx]);
            assert!(ii[idx] > 0.0 && ii[idx] < 1.0, "I[{idx}]={}", ii[idx]);
        }
    }
}
