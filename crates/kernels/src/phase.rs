//! Pure Rust SIMD phase dynamics — `wide::f32x8` for Kuramoto, T3.
//! Replaces all ASM phase kernels with portable SIMD.
//!
//! sin/cos are computed per-element (no SIMD for trig in `wide`),
//! but all arithmetic (addition, multiplication, wrapping) uses f32x8.
//!
//! ## Kuramoto models
//! - `kuramoto_update` — стандартная: φ += dt·(ω + coupling)
//! - `kuramoto_update_viscous` — с вязким демпфированием: φ += dt·(ω + coupling + η·∇²φ)
//!   Вязкий член η·∇²φ = η·(mean_neighbor_phase - φ) сглаживает фазовые разности
//!   и подавляет хаотический шум при переключении фаз.

use super::wrap_phase;
use crate::neuron::{load8, store8};
use core::f32::consts::TAU;
use wide::f32x8;

/// Wrap phases to [0, 2π) and compute sin/cos in one pass.
///
/// sin/cos are computed per-element (no SIMD trig in `wide`).
/// Phase wrapping uses f32x8 for the arithmetic part.
///
/// # Safety
/// `phases[n]`, `cos_out[n]`, `sin_out[n]` must be valid, non-overlapping.
pub(crate) unsafe fn kuramoto_prepare(
    phases: *mut f32,
    cos_out: *mut f32,
    sin_out: *mut f32,
    n: usize,
) {
    let tau_s = f32x8::splat(TAU);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let pv = load8(phases, off);
        // Wrap: ϕ − τ·round(ϕ/τ)
        let wrapped = pv - tau_s * (pv / tau_s).round();
        // SIMD sin/cos via minimax polynomial (~1e-4 accuracy, ~4× faster than libm)
        let (si, co) = super::fast_sincos_simd(wrapped);
        store8(phases, off, wrapped);
        store8(cos_out, off, co);
        store8(sin_out, off, si);
    }

    if n8 < n {
        let p = std::slice::from_raw_parts_mut(phases, n);
        let co = std::slice::from_raw_parts_mut(cos_out, n);
        let si = std::slice::from_raw_parts_mut(sin_out, n);
        for i in n8..n {
            let phi = wrap_phase(p[i]);
            p[i] = phi;
            co[i] = phi.cos();
            si[i] = phi.sin();
        }
    }
}

/// Euler-step Kuramoto phases: φ += dt·(ω + coupling).
/// Uses f32x8 SIMD for all arithmetic.
///
/// # Safety
/// `phases[n]`, `omegas[n]`, `coupling[n]` must be valid, non-overlapping.
pub(crate) unsafe fn kuramoto_update(
    phases: *mut f32,
    omegas: *const f32,
    coupling: *const f32,
    dt: f32,
    n: usize,
) {
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut pv = load8(phases, off);
        let ov = load8(omegas, off);
        let cv = load8(coupling, off);
        pv += dt_s * (ov + cv);
        // Wrap phase
        pv = pv - tau_s * (pv / tau_s).round();
        store8(phases, off, pv);
    }

    if n8 < n {
        let p = std::slice::from_raw_parts_mut(phases, n);
        let o = std::slice::from_raw_parts(omegas, n);
        let c = std::slice::from_raw_parts(coupling, n);
        for i in n8..n {
            p[i] = wrap_phase(p[i] + dt * (o[i] + c[i]));
        }
    }
}

/// Viscous Kuramoto update: φ += dt·(ω + coupling + η·∇²φ)
///
/// Вязкий член η·∇²φ аппроксимируется как разность с соседями по CSR:
///   η·∇²φ_i = η · Σ_j W_ij · sin(φ_j − φ_i)  — для разреженной связности
///
/// Для fully-connected попарной связности (когда каждый с каждым):
///   η·∇²φ_i = η · (mean(φ) − φ_i)
///   где mean(φ) — средняя фаза всех осцилляторов
///
/// Вязкое демпфирование подавляет высокочастотные фазовые разности,
/// сглаживая переходы между синхронными состояниями. Без этого члена
/// Kuramoto-ансамбль может хаотически переключаться между модами.
///
/// Reference: Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence"
///
/// # Args
/// - `phases`: фазы осцилляторов [n]
/// - `omegas`: собственные частоты [n]
/// - `coupling`: внешняя связность (через sin) [n]
/// - `eta`: коэффициент вязкости (η ≥ 0). η=0 → обычный Kuramoto.
/// - `dt`: шаг интегрирования
/// - `n`: количество осцилляторов
///
/// # Safety
/// All pointers valid for `n` elements, non-overlapping.
pub(crate) unsafe fn kuramoto_update_viscous(
    phases: *mut f32,
    omegas: *const f32,
    coupling: *const f32,
    eta: f32,
    dt: f32,
    n: usize,
) {
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);

    if eta <= 0.0 {
        return kuramoto_update(phases, omegas, coupling, dt, n);
    }

    // 1. Compute mean phase (scalar, one pass)
    let mean_phase = {
        let p = std::slice::from_raw_parts(phases, n);
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += p[i];
        }
        sum / n as f32
    };

    // 2. SIMD viscous update: φ += dt·(ω + c + η·(mean − φ))
    let eta_s = f32x8::splat(eta);
    let mean_s = f32x8::splat(mean_phase);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut pv = load8(phases, off);
        let ov = load8(omegas, off);
        let cv = load8(coupling, off);

        let viscous = eta_s * (mean_s - pv);
        pv += dt_s * (ov + cv + viscous);
        pv = pv - tau_s * (pv / tau_s).round();
        store8(phases, off, pv);
    }

    // 3. Scalar tail
    if n8 < n {
        let p = std::slice::from_raw_parts_mut(phases, n);
        let o = std::slice::from_raw_parts(omegas, n);
        let c = std::slice::from_raw_parts(coupling, n);
        for i in n8..n {
            let viscous = eta * (mean_phase - p[i]);
            p[i] = wrap_phase(p[i] + dt * (o[i] + c[i] + viscous));
        }
    }
}

/// Euler-step phase (Kuramoto) + amplitude (Stuart-Landau).
/// Matches [`ScalarBackend::update`](crate::t3::backend::ScalarBackend::update) exactly:
///
/// ```text
/// r += dt · ((α − r²)·r + gain·drive − decay·r)     ← Stuart-Landau
/// θ += dt · (ω + coupling + gain·drive / (r + ε))    ← Kuramoto + drive
/// ```
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn t3_update(
    phases: *mut f32,
    amplitudes: *mut f32,
    omegas: *const f32,
    coupling: *const f32,
    drives: *const f32,
    dt: f32,
    decay: f32,
    alpha: f32,
    gain: f32,
    n: usize,
) {
    let tau_s = f32x8::splat(TAU);
    let dt_s = f32x8::splat(dt);
    let decay_s = f32x8::splat(decay);
    let alpha_s = f32x8::splat(alpha);
    let gain_s = f32x8::splat(gain);
    let eps_s = f32x8::splat(0.1);
    let min_amp = f32x8::splat(0.001);
    let max_amp = f32x8::splat(10.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut pv = load8(phases, off);
        let mut av = load8(amplitudes, off);
        let ov = load8(omegas, off);
        let cv = load8(coupling, off);
        let dv = load8(drives, off);

        // Kuramoto phase FIRST (uses OLD r — matches ScalarBackend):
        // θ += dt · (ω + c + gain·drive / (r + ε))
        let drive_phase = gain_s * dv / (av + eps_s);
        pv += dt_s * (ov + cv + drive_phase);
        pv = pv - tau_s * (pv / tau_s).round();

        // Stuart-Landau amplitude SECOND: r += dt · ((α − r²)·r + gain·drive − decay·r)
        let r2 = av * av;
        let r_dot = (alpha_s - r2) * av + gain_s * dv - decay_s * av;
        av += dt_s * r_dot;
        av = av.max(min_amp).min(max_amp);

        store8(phases, off, pv);
        store8(amplitudes, off, av);
    }

    if n8 < n {
        let p = std::slice::from_raw_parts_mut(phases, n);
        let a = std::slice::from_raw_parts_mut(amplitudes, n);
        let o = std::slice::from_raw_parts(omegas, n);
        let c = std::slice::from_raw_parts(coupling, n);
        let d = std::slice::from_raw_parts(drives, n);
        for i in n8..n {
            let r = a[i];
            // Phase first (uses OLD r)
            let drive_phase = gain * d[i] / (r + 0.1);
            p[i] = wrap_phase(p[i] + dt * (o[i] + c[i] + drive_phase));
            // Amplitude second
            let r_dot = (alpha - r * r) * r + gain * d[i] - decay * r;
            a[i] = (r + dt * r_dot).max(0.001).min(10.0);
        }
    }
}
