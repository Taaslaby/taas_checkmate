//! Izhikevich neuron model — step-адаптер + E-Prop variants.
//!
//! Канон: `step_izhikevich` → `run_neuron::<IzhikevichModel>` (транш 3b),
//! `step_izhikevich_dual` → `run_neuron::<IzhikevichDualModel>` (волна 12).
//! E-Prop классы — `super::eprop`. Перенесено из `core::neuron::izhikevich` (Фаза 9).

#![allow(non_snake_case)]

use crate::core::runner::run_neuron;
use crate::zoo::neurons::{IzhikevichDualModel, IzhikevichModel};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Step `n` Izhikevich neurons by `dt`. n must be multiple of 8.
///
/// Поддерживает refractory через `refractory` buffer (0 = не рефрактерна).
/// 7-col SoA [dt, a, b, c, d, tau_ref, v_thresh=30]; state [v, u, refractory].
/// Legacy STDP side-output: `trace` обновляется вне pillars-модели.
pub fn step_izhikevich(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &mut [f32],
    dt: f32,
    trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
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
            && n == c.len()
            && n == d.len()
            && n == refractory.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(trace.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .zip(
            a.par_chunks_exact(8)
                .zip(b.par_chunks_exact(8))
                .zip(c.par_chunks_exact(8))
                .zip(d.par_chunks_exact(8)),
        )
        .for_each(
            |((((((v8, u8), tr8), I8), sp8), refr8), (((a8, b8), c8), d8))| {
                process_chunk(
                    v8,
                    u8,
                    tr8,
                    a8,
                    b8,
                    I8,
                    dt,
                    trace_decay,
                    sp8,
                    c8,
                    d8,
                    refr8,
                    tau_ref,
                );
            },
        );

    #[cfg(not(feature = "rayon"))]
    for i in (0..n).step_by(8) {
        process_chunk(
            &mut v[i..i + 8],
            &mut u[i..i + 8],
            &mut trace[i..i + 8],
            &a[i..i + 8],
            &b[i..i + 8],
            &mut I[i..i + 8],
            dt,
            trace_decay,
            &mut spikes[i..i + 8],
            &c[i..i + 8],
            &d[i..i + 8],
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
    I8: &mut [f32],
    dt: f32,
    trace_decay: f32,
    sp8: &mut [i32],
    c8: &[f32],
    d8: &[f32],
    refr8: &mut [f32],
    tau_ref: f32,
) {
    let mut params = [0.0f32; 56];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = a8[j];
        params[16 + j] = b8[j];
        params[24 + j] = c8[j];
        params[32 + j] = d8[j];
        params[40 + j] = tau_ref;
        params[48 + j] = 30.0; // legacy v_th (hardcoded)
    }
    unsafe {
        run_neuron::<IzhikevichModel>(
            v8.as_mut_ptr(),
            u8.as_mut_ptr(),
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
    // ── Legacy STDP side-output: trace (НЕ состояние pillars-модели).
    for j in 0..8 {
        if refr8[j] > 0.0 {
            continue; // frozen during refractory
        }
        if sp8[j] != 0 {
            tr8[j] = tr8[j] * trace_decay + (1.0 - tr8[j]);
        } else {
            tr8[j] *= 1.0 - trace_decay;
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Izhikevich Dual (Sun et al. 2026) — fast Izh + slow memory state
// ═══════════════════════════════════════════════════════════

/// Step `n` Izhikevich Dual Memory neurons (Sun et al. 2026, Nature MI).
/// n must be multiple of 8.
///
/// Канон — `run_neuron::<IzhikevichDualModel>` (волна 12; бывший
/// `connectome::sim::step_izhikevich_dual_pillars` перенесён сюда).
/// 10-col SoA [dt, a, b, c, d, tau_ref, v_thresh=30, trace_decay, tau_m, w_m];
/// state [v, u, refractory, trace, m]. Slow state `m` — память недавней
/// активности: `m = τ_m·m + (1−τ_m)·spike_val`, ток модулируется `w_m·m`.
/// Legacy STDP side-output: `trace` обновляется вне pillars-модели.
pub fn step_izhikevich_dual(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &mut [f32],
    dt: f32,
    trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
    m: &mut [f32], // slow state per neuron
    tau_m: f32,    // slow state time constant (0..1, higher = slower)
    w_m: f32,      // memory-to-current weight
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
            && n == c.len()
            && n == d.len()
            && n == refractory.len()
            && n == m.len()
    );

    #[cfg(feature = "rayon")]
    v.par_chunks_exact_mut(8)
        .zip(u.par_chunks_exact_mut(8))
        .zip(trace.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .zip(refractory.par_chunks_exact_mut(8))
        .zip(m.par_chunks_exact_mut(8))
        .zip(
            a.par_chunks_exact(8)
                .zip(b.par_chunks_exact(8))
                .zip(c.par_chunks_exact(8))
                .zip(d.par_chunks_exact(8)),
        )
        .for_each(
            |(((((((v8, u8), tr8), I8), sp8), refr8), m8), (((a8, b8), c8), d8))| {
                process_chunk_dual(
                    v8,
                    u8,
                    tr8,
                    a8,
                    b8,
                    I8,
                    dt,
                    trace_decay,
                    sp8,
                    c8,
                    d8,
                    refr8,
                    tau_ref,
                    m8,
                    tau_m,
                    w_m,
                );
            },
        );

    #[cfg(not(feature = "rayon"))]
    for i in (0..n).step_by(8) {
        process_chunk_dual(
            &mut v[i..i + 8],
            &mut u[i..i + 8],
            &mut trace[i..i + 8],
            &a[i..i + 8],
            &b[i..i + 8],
            &mut I[i..i + 8],
            dt,
            trace_decay,
            &mut spikes[i..i + 8],
            &c[i..i + 8],
            &d[i..i + 8],
            &mut refractory[i..i + 8],
            tau_ref,
            &mut m[i..i + 8],
            tau_m,
            w_m,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_chunk_dual(
    v8: &mut [f32],
    u8: &mut [f32],
    tr8: &mut [f32],
    a8: &[f32],
    b8: &[f32],
    I8: &mut [f32],
    dt: f32,
    trace_decay: f32,
    sp8: &mut [i32],
    c8: &[f32],
    d8: &[f32],
    refr8: &mut [f32],
    tau_ref: f32,
    m8: &mut [f32],
    tau_m: f32,
    w_m: f32,
) {
    let mut params = [0.0f32; 80];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = a8[j];
        params[16 + j] = b8[j];
        params[24 + j] = c8[j];
        params[32 + j] = d8[j];
        params[40 + j] = tau_ref;
        params[48 + j] = 30.0; // legacy v_th (hardcoded)
        params[56 + j] = trace_decay;
        params[64 + j] = tau_m;
        params[72 + j] = w_m;
    }
    unsafe {
        run_neuron::<IzhikevichDualModel>(
            v8.as_mut_ptr(),
            u8.as_mut_ptr(),
            refr8.as_mut_ptr(),
            tr8.as_mut_ptr(),
            m8.as_mut_ptr(),
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
    fn izhikevich_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let a = vec![0.02; n];
        let b = vec![0.2; n];
        let c = vec![-65.0; n];
        let d = vec![2.0; n];
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];

        let mut total = 0i32;
        for _ in 0..100 {
            step_izhikevich(
                &mut v,
                &mut u,
                &mut trace,
                &a,
                &b,
                &mut I,
                dt,
                0.9,
                &mut spikes,
                &c,
                &d,
                &mut refractory,
                2.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "Izhikevich should fire with I=30");
    }

    #[test]
    fn izhikevich_dual_wrapper_fires() {
        let n = 16;
        let dt = 1.0;
        let a = vec![0.02; n];
        let b = vec![0.2; n];
        let c = vec![-65.0; n];
        let d = vec![8.0; n];
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refractory = vec![0.0f32; n];
        let mut m = vec![0.5f32; n];

        let mut total = 0i32;
        for _ in 0..100 {
            step_izhikevich_dual(
                &mut v,
                &mut u,
                &mut trace,
                &a,
                &b,
                &mut I,
                dt,
                0.99,
                &mut spikes,
                &c,
                &d,
                &mut refractory,
                2.0,
                &mut m,
                0.999,
                5.0,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "Izhikevich dual should fire with I=30, w_m=5");
        assert!(
            m.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "slow state m must stay in [0,1]"
        );
    }
}
