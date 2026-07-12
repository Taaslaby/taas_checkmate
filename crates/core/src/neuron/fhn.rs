//! FitzHugh-Nagumo (FHN) neuron model — simplified Hodgkin-Huxley.
//!
//! Safe Rust wrapper over SIMD kernel (`kernels::fhn_step`).
//!
//! Equations:
//!   dv/dt = v - v³/3 - w + I_ext
//!   dw/dt = (v + a - b·w) / τ
//!
//! This model is ~2x faster than Izhikevich and captures the essential
//! excitability dynamics: resting → oscillatory → chaotic regimes.
//!
//! Reference: FitzHugh (1961) "Impulses and physiological states in
//! theoretical models of nerve membrane" — Biophysical Journal

use crate::kernels;
use rayon::prelude::*;

/// Step `n` FitzHugh-Nagumo neurons. Parallel, n must be multiple of 8.
///
/// Parameters per neuron:
/// - `dt`: time step (ms), uniform across all neurons
/// - `tau_w`: recovery time constant (ms)
/// - `a`: nullcline parameter (default 0.7)
/// - `b`: nullcline parameter (default 0.8)
///
/// Spike detection: v ≥ threshold (default 0.5, hardcoded in kernel).
#[allow(clippy::too_many_arguments)]
pub fn step_fhn(
    v: &mut [f32], w: &mut [f32], I: &[f32], spikes: &mut [i32],
    dt: f32, tau_w: &[f32], a: &[f32], b: &[f32],
) {
    let n = v.len();
    assert!(n.is_multiple_of(8) && n == w.len() && n == I.len() && n == spikes.len()
        && n == tau_w.len() && n == a.len() && n == b.len());

    kernels::safety_clamp(&mut v.to_vec(), -super::I_CLAMP, super::I_CLAMP);

    // Build params on stack per chunk — avoids heap allocation on every step.
    v.par_chunks_exact_mut(8)
        .zip(w.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, (((v8, w8), I8), sp8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 32];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = tau_w[i + j];
                params[16 + j] = a[i + j];
                params[24 + j] = b[i + j];
            }
            kernels::fhn_step(v8, w8, I8, sp8, &params);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fhn_wrapper_oscillates() {
        let n = 16;
        let dt = 0.01;
        let tau_w = vec![1.25; n];
        let a = vec![0.7; n];
        let b = vec![0.8; n];
        let I = vec![0.5; n];
        let mut v = vec![0.0f32; n];
        let mut w = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        let mut total_spikes = 0i32;
        for _ in 0..10000 {
            step_fhn(&mut v, &mut w, &I, &mut spikes, dt, &tau_w, &a, &b);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "FHN wrapper should oscillate");
    }
}
