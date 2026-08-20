//! SAW (Spike-Adaptive-Width) neuron + graph-coupled oscillator.
//!
//! Канон: `step_saw` → `run_neuron::<SAWModel>` (8-col repacking +
//! noise-инъекция), `step_gco` → `kernels::phase::gco_step`.
//! E-Prop SAW — `super::eprop`. Перенесено из `core::neuron::saw`.
//!
//! пожалуйста не грузите это в бензопилу это может сделать бум градиенты не текут на 2 такктовом двигателе

#![allow(non_snake_case)]

use crate::zoo::neurons::SAWModel;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::cell::Cell;

// LCG-шум для noise-инъекции SAW.
thread_local! {
    static RNG_STATE: Cell<u32> = const { Cell::new(1) };
}

fn rand_f32() -> f32 {
    RNG_STATE.with(|state| {
        let mut s = state.get();
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        state.set(s);
        (s >> 16) as f32 * (1.0 / 65536.0)
    })
}

// ─── SAW ─────────────────────────────────────────────────

/// Step `n` SAW neurons. n must be multiple of 8.
pub fn step_saw(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &[f32],
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == u.len()
            && n == trace.len()
            && n == a.len()
            && n == b.len()
            && n == I.len()
            && n == spikes.len()
            && n == refractory.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(trace.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .zip(
            a.par_chunks_exact(8)
                .zip(b.par_chunks_exact(8))
                .zip(I.par_chunks_exact(8)),
        )
        .for_each(|(((((v8, u8), tr8), sp8), refr8), ((a8, b8), I8))| {
            process_chunk(
                v8,
                u8,
                tr8,
                a8,
                b8,
                I8,
                dt,
                trace_decay,
                noise_amp,
                sp8,
                refr8,
                tau_ref,
            );
        });

    #[cfg(not(feature = "rayon"))]
    for i in (0..n).step_by(8) {
        process_chunk(
            &mut v[i..i + 8],
            &mut u[i..i + 8],
            &mut trace[i..i + 8],
            &a[i..i + 8],
            &b[i..i + 8],
            &I[i..i + 8],
            dt,
            trace_decay,
            noise_amp,
            &mut spikes[i..i + 8],
            &mut refractory[i..i + 8],
            tau_ref,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    v8: &mut [f32],
    u8: &mut [f32],
    tr8: &mut [f32],
    a8: &[f32],
    b8: &[f32],
    I8: &[f32],
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    sp8: &mut [i32],
    refr8: &mut [f32],
    tau_ref: f32,
) {
    // SAWModel 8-col: [dt, a, b, trace_decay, noise_amp=0, tau_ref,
    //   v_thresh=30, v_reset=-65] (легаси-хардкоды сохранены).
    let mut params = [0.0f32; 64];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = a8[j];
        params[16 + j] = b8[j];
        params[24 + j] = trace_decay;
        params[32 + j] = 0.0; // noise_amp (модель детерминирована; шум — ниже)
        params[40 + j] = tau_ref;
        params[48 + j] = 30.0; // legacy v_th
        params[56 + j] = -65.0; // legacy v_reset
    }
    crate::core::runner::run_neuron_slices::<SAWModel>(
        crate::core::runner::NeuronState {
            v: Some(v8),
            u: Some(u8),
            h: Some(tr8),   // state 2 = ψ-wave trace
            r: Some(refr8), // state 3
            ..Default::default()
        },
        I8,
        sp8,
        &params,
    );
    // Шум впрыскивается после обновления и детекции спайка (как в легаси).
    if noise_amp != 0.0 {
        for j in 0..8 {
            if sp8[j] == 0 {
                v8[j] += noise_amp * (rand_f32() - 0.5);
            }
        }
    }
}

// ─── Graph-Coupled Oscillator ──────────────────────────

/// Step `n` graph-coupled oscillators. n must be multiple of 8.
///
/// Канон — `kernels::phase::gco_step` (единственный источник правды;
/// легаси `kernels::gco_step` удалён — parity в `crates/kernels/tests/gco_parity.rs`).
pub fn step_gco(
    v: &mut [f32],
    u: &mut [f32],
    h_real: &mut [f32],
    h_imag: &mut [f32],
    I_ext: &mut [f32],
    speed: &mut [f32],
    dt: f32,
) {
    let n = v.len();
    assert!(
        n.is_multiple_of(8)
            && n == u.len()
            && n == h_real.len()
            && n == h_imag.len()
            && n == I_ext.len()
            && n == speed.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(h_real.par_chunks_exact_mut(8))
        .zip(h_imag.par_chunks_exact_mut(8))
        .zip(I_ext.par_chunks_exact_mut(8))
        .zip(speed.par_chunks_exact_mut(8))
        .for_each(|(((((v8, u8), hr8), hi8), ie8), sp8)| {
            crate::kernels::phase::gco_step(v8, u8, hr8, hi8, ie8, sp8, dt);
        });

    #[cfg(not(feature = "rayon"))]
    for i in (0..n).step_by(8) {
        crate::kernels::phase::gco_step(
            &mut v[i..i + 8],
            &mut u[i..i + 8],
            &h_real[i..i + 8],
            &h_imag[i..i + 8],
            &I_ext[i..i + 8],
            &speed[i..i + 8],
            dt,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saw_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let a = vec![0.02; n];
        let b = vec![0.2; n];
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];

        let mut total = 0i32;
        for _ in 0..200 {
            step_saw(
                &mut v,
                &mut u,
                &mut trace,
                &a,
                &b,
                &I,
                dt,
                0.9,
                0.0,
                &mut spikes,
                &mut refractory,
                2.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "SAW should fire with I=30");
    }

    #[test]
    fn gco_wrapper_conserves_norm() {
        let n = 16;
        let dt = 0.1;
        let mut v = vec![1.0f32; n];
        let mut u = vec![0.0f32; n];
        let mut h_real = vec![0.0f32; n];
        let mut h_imag = vec![0.0f32; n];
        let mut i_ext = vec![0.0f32; n];
        let mut speed = vec![0.0f32; n];

        step_gco(
            &mut v,
            &mut u,
            &mut h_real,
            &mut h_imag,
            &mut i_ext,
            &mut speed,
            dt,
        );
        // Нормировка после шага сохраняет |z| ≈ 1.
        let norm = (v[0] * v[0] + u[0] * u[0]).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "GCO norm={norm}");
    }
}
