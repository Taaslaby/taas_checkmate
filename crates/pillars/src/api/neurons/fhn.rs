//! FitzHugh-Nagumo (FHN) neuron model — simplified Hodgkin-Huxley.
//!
//! Канон: `step_fhn` → `run_neuron::<FHNModel>` (волна 7; легаси
//! `kernels::fhn_step` удалён). Перенесено из `core::neuron::fhn` (Фаза 9).
//!
//! Equations:
//!   dv/dt = v - v³/3 - w + I_ext
//!   dw/dt = (v + a - b·w) / τ
//!
//! **Spike semantics (edge-triggered):** one spike per rising crossing of
//! `v_thresh`, no back-to-back bursts inside the plateau (AGENTS.md rule 6).

use crate::core::runner::run_neuron;
use crate::zoo::neurons::FHNModel;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Default spike threshold — mirrors the legacy kernel's hardcoded 0.5.
/// ⚠️ Связано с дефолтным `v_thresh` модели `zoo/neurons/fhn.rs` (тоже 0.5).
const V_THRESH: f32 = 0.5;

/// Step `n` FitzHugh-Nagumo neurons. n must be multiple of 8.
///
/// Parameters per neuron:
/// - `dt`: time step (ms), uniform across all neurons
/// - `tau_w`: recovery time constant (ms)
/// - `a`: nullcline parameter (default 0.7)
/// - `b`: nullcline parameter (default 0.8)
///
/// Spike detection: rising edge of `v ≥ 0.5` (canonical, edge-triggered).
pub fn step_fhn(
    v: &mut [f32],
    w: &mut [f32],
    I: &[f32],
    spikes: &mut [i32],
    dt: f32,
    tau_w: &[f32],
    a: &[f32],
    b: &[f32],
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == w.len()
            && n == I.len()
            && n == spikes.len()
            && n == tau_w.len()
            && n == a.len()
            && n == b.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(w.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, (((v8, w8), I8), sp8))| {
            process_chunk(chunk, v8, w8, I8, sp8, dt, tau_w, a, b);
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let i = chunk * 8;
        process_chunk(
            chunk,
            &mut v[i..i + 8],
            &mut w[i..i + 8],
            &I[i..i + 8],
            &mut spikes[i..i + 8],
            dt,
            tau_w,
            a,
            b,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: usize,
    v8: &mut [f32],
    w8: &mut [f32],
    I8: &[f32],
    sp8: &mut [i32],
    dt: f32,
    tau_w: &[f32],
    a: &[f32],
    b: &[f32],
) {
    let i = chunk * 8;
    // FHNModel SoA layout per chunk of 8 (column-major):
    //   [dt, inv_tau_w, a, b, v_thresh] × 8
    // `inv_tau_w` guard mirrors the legacy kernel: tau_w ≤ 0 → dw = 0.
    let mut params = [0.0f32; 40];
    let mut was_below = [1.0f32; 8];
    for j in 0..8 {
        params[j] = dt;
        let tw = tau_w[i + j];
        params[8 + j] = if tw > 0.0 { 1.0 / tw.max(1e-10) } else { 0.0 };
        params[16 + j] = a[i + j];
        params[24 + j] = b[i + j];
        params[32 + j] = V_THRESH;
        was_below[j] = if v8[j] < V_THRESH { 1.0 } else { 0.0 };
    }
    unsafe {
        run_neuron::<FHNModel>(
            v8.as_mut_ptr(),
            w8.as_mut_ptr(),
            was_below.as_mut_ptr(),
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

    /// Canonical edge semantics: no back-to-back spikes (no bursts).
    #[test]
    fn fhn_wrapper_no_back_to_back_spikes() {
        let n = 16;
        let dt = 0.01;
        let tau_w = vec![1.25; n];
        let a = vec![0.7; n];
        let b = vec![0.8; n];
        let I = vec![0.5; n];
        let mut v = vec![0.0f32; n];
        let mut w = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        let mut fired = false;
        let mut prev = vec![0i32; n];
        for _ in 0..10000 {
            step_fhn(&mut v, &mut w, &I, &mut spikes, dt, &tau_w, &a, &b);
            for i in 0..n {
                if spikes[i] != 0 {
                    assert_eq!(
                        prev[i], 0,
                        "back-to-back spike at neuron {i} — edge detection violated"
                    );
                    fired = true;
                }
            }
            prev.copy_from_slice(&spikes);
        }
        assert!(fired, "FHN should fire at least once with I=0.5");
    }
}
