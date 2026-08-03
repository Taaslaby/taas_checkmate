//! Pure Rust SIMD neuron dynamics — `wide::f32x8` for AVX2/NEON/WASM.
//!
//! Each function processes `n` neurons. When `n ≥ 8`, uses `f32x8` for
//! 8-wide SIMD (AVX2 on x86_64, NEON on ARM, portable emulation elsewhere),
//! with a scalar tail for remaining elements.
//!
//! ## Implemented models
//!
//! | Function | Model | Integration |
//! |---|---|---|
//! | `lif_at_step` | LIF-AT (adaptive threshold) | Exponential Euler + θ·exp(-dt/τ_θ) |
//! | `eprop_lif_step` | eProp LIF (eligibility trace) | Exponential Euler + exact eligibility |
//! | `saw_step` | SAW (Spike-Adaptive-Width) | Forward Euler + noise |
//! | `gco_step` | Graph-coupled oscillator | coupled phases on unit circle |
//! | `hh_lut_step` | LUT-HH (Hodgkin-Huxley) | LUT-interpolated exponential Euler |
//!
//! > **Примечание (Фаза 3 консолидации):** batch-ядра `lif_step`, `izhikevich_step`,
//! > `adex_step` удалены из `neuron/*` — они живут в `crate::safe` как тонкие
//! > обёртки над `pillars::run_neuron` (см. `safe.rs` и `docs/DUPLICATION_MAP.md`).

#![allow(non_snake_case)]

use wide::f32x8;

const RNG_SCALE_16: f32 = 1.0 / 65536.0;
/// 24-bit scale: for Poisson spike generation (fused kernel uses state>>8)
pub(super) const RNG_SCALE_24: f32 = 1.0 / 16777216.0;

/// Per-thread RNG state for deterministic noise sequences.
use core::cell::Cell;

thread_local! {
    static RNG_STATE: Cell<u32> = const { Cell::new(1) };
}

// ═══════════════════════════════════════════════════════
//  Helper: load f32x8 from raw pointer (index offset)
// ═══════════════════════════════════════════════════════

#[inline(always)]
pub(crate) unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([
        *p,
        *p.add(1),
        *p.add(2),
        *p.add(3),
        *p.add(4),
        *p.add(5),
        *p.add(6),
        *p.add(7),
    ])
}
#[inline(always)]
pub(crate) unsafe fn load8_mut(ptr: *mut f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([
        *p,
        *p.add(1),
        *p.add(2),
        *p.add(3),
        *p.add(4),
        *p.add(5),
        *p.add(6),
        *p.add(7),
    ])
}
#[inline(always)]
unsafe fn load8_i32(ptr: *const i32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([
        *p as f32,
        *p.add(1) as f32,
        *p.add(2) as f32,
        *p.add(3) as f32,
        *p.add(4) as f32,
        *p.add(5) as f32,
        *p.add(6) as f32,
        *p.add(7) as f32,
    ])
}
#[inline(always)]
pub(crate) unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
    let a = v.to_array();
    let p = ptr.add(off);
    *p = a[0];
    *p.add(1) = a[1];
    *p.add(2) = a[2];
    *p.add(3) = a[3];
    *p.add(4) = a[4];
    *p.add(5) = a[5];
    *p.add(6) = a[6];
    *p.add(7) = a[7];
}

#[inline(always)]
fn spike_mask_to_i32(mask: f32x8) -> [i32; 8] {
    let arr = mask.to_array();
    [
        (arr[0].to_bits() != 0) as i32,
        (arr[1].to_bits() != 0) as i32,
        (arr[2].to_bits() != 0) as i32,
        (arr[3].to_bits() != 0) as i32,
        (arr[4].to_bits() != 0) as i32,
        (arr[5].to_bits() != 0) as i32,
        (arr[6].to_bits() != 0) as i32,
        (arr[7].to_bits() != 0) as i32,
    ]
}

/// Per-thread deterministic RNG for SAW noise.
fn rand_f32() -> f32 {
    RNG_STATE.with(|state| {
        let mut s = state.get();
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        state.set(s);
        (s >> 16) as f32 * RNG_SCALE_16
    })
}

// ═══════════════════════════════════════════════════════
//  Fast exp — single-sourced from pillars::math (Phase 1)
// ═══════════════════════════════════════════════════════
//
//  Канонические реализации живут в `pillars::math::fast_exp`:
//  - `fast_exp`     (скаляр)     ↔ `pillars::math::fast_exp::fast_exp`
//  - `fast_exp_simd` (SIMD-полином) ↔ `pillars::math::fast_exp::fast_exp_poly_simd`//  ⚠️ НЕ `pillars::fast_exp_simd` — тот аппаратный vexp (≈1 ULP)!
//  Бит-идентичность подтверждена `tests/parity_pillars.rs` на сетке [-87, 87]
//  (max rel diff скаляра = 0.0, SIMD — побитово).
//
//  Контракт точности — ТОЛЬКО на [-87, 87]. За пределами: старый kernels-путь
//  клампил вход к ±87 (exp(±87) ≈ 2.6e37 / ~0), pillars-полином НЕ клампит,
//  а блендит шкалу к 0.0/inf при n<-126/n>127. Все kernels-вызовы (hh vtrap,
//  adex arg-clamp-to-20, nmda mg-block, dt/τ-decays, pre-clamped synapse args)
//  лежат внутри ±87 — граница недостижима на практике.

pub use pillars::math::fast_exp::fast_exp;
pub use pillars::math::fast_exp::fast_exp_poly_simd as fast_exp_simd;

#[inline(always)]
pub fn fast_sincos_simd(x: f32x8) -> (f32x8, f32x8) {
    use core::f32::consts::{FRAC_2_PI, FRAC_PI_2};
    let pi2 = f32x8::splat(FRAC_PI_2);
    let rcp_pi2 = f32x8::splat(FRAC_2_PI);

    let t = x * rcp_pi2;
    let n = t.round();
    let r = x - n * pi2;
    let r2 = r * r;

    let sin_poly = r2.mul_add(f32x8::splat(-0.000_198_409), f32x8::splat(0.008_333_332));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(-0.166_666_67));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(1.0));
    let sin_val = r * sin_poly;

    let cos_poly = r2.mul_add(f32x8::splat(-0.0013888889), f32x8::splat(0.041_666_668));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(-0.5));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(1.0));
    let cos_val = cos_poly;

    let four = f32x8::splat(4.0);
    let n_mod_4 = n - four * (n / four).floor();

    let n_arr: [f32; 8] = n_mod_4.to_array();
    let s_arr: [f32; 8] = sin_val.to_array();
    let c_arr: [f32; 8] = cos_val.to_array();
    let mut out_s = [0.0f32; 8];
    let mut out_c = [0.0f32; 8];
    for i in 0..8 {
        let q = n_arr[i] as i32 & 3;
        match q {
            0 => {
                out_s[i] = s_arr[i];
                out_c[i] = c_arr[i];
            }
            1 => {
                out_s[i] = c_arr[i];
                out_c[i] = -s_arr[i];
            }
            2 => {
                out_s[i] = -s_arr[i];
                out_c[i] = -c_arr[i];
            }
            _ => {
                out_s[i] = -c_arr[i];
                out_c[i] = s_arr[i];
            }
        }
    }

    (f32x8::from(out_s), f32x8::from(out_c))
}

#[inline(always)]
fn surrogate_gradient(v: f32, v_thresh: f32, tau_sg: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_sg;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

// ═══════════════════════════════════════════════════════
//  🧠 Spiking Neuron Models (биофизические модели нейронов)
// ═══════════════════════════════════════════════════════
pub mod adex; // Adaptive Exponential (AdEx)
pub mod fhn; // FitzHugh-Nagumo
pub mod glif; // GLIF3/4 (Generalized LIF — Allen Institute)
pub mod haap; // HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting) — VNC neuron
pub mod hh; // LUT Hodgkin-Huxley (NaV, KV, CaV-T)
pub mod izhikevich; // Izhikevich (2003)
pub mod lif; // LIF + LIF-AT + eProp-LIF
pub mod hr; // Hindmarsh-Rose (bursting model)
pub mod morris_lecar; // Morris-Lecar (2-variable, barnacle muscle)
pub mod nmda; // NMDA synapse with Mg²⁺ block
pub mod tsodyks_markram; // Tsodyks-Markram short-term plasticity

pub use morris_lecar::*;
pub use nmda::*;
pub use tsodyks_markram::*;

// ═══════════════════════════════════════════════════════
//  🌊 Wave / Oscillator Models (волновые и осцилляторные)
// ═══════════════════════════════════════════════════════
pub mod depth_wave; // DepthGCO (depth-modulated graph-coupled oscillator on CSR)
pub mod saw; // SAW + gco_step (graph-coupled oscillator)
pub mod wc; // Wilson-Cowan (population rate model)

// ═══════════════════════════════════════════════════════
//  🔗 Synaptic / Plasticity Kernels (синапсы и пластичность)
// ═══════════════════════════════════════════════════════
pub mod dopamine; // DA-modulated STDP
pub mod eprop; // Eligibility Propagation traces (Bellec 2020)
pub mod bcm; // BCM rule (Bienenstock-Cooper-Munro 1982) + Homeostatic Scaling
pub mod stp; // Short-Term Plasticity (Tsodyks-Markram)
pub mod synapse; // Multi-receptor dynamics (nAChR, GluCl, GABA_A/B)

// ═══════════════════════════════════════════════════════
//  🏋️ Motor / Embodied Kernels (мышцы и телесная физика)
// ═══════════════════════════════════════════════════════
pub mod atlasov_neuromotor; // Atlasov Ca²⁺ muscle + Tug-of-War joints
pub mod chordotonal; // Chordotonal sensory organ (load → firing rate)
pub mod dendrite; // Cable dendrite + active Na⁺/K⁺/Ca²⁺ channels

// ═══════════════════════════════════════════════════════
//  🧫 Unconventional (нестандартные биосистемы)
// ═══════════════════════════════════════════════════════
pub mod astrocyte; // Astrocyte Ca²⁺ waves + Gap Junctions

// ═══════════════════════════════════════════════════════
//  ⚡ Infrastructure (инфраструктура и оптимизации)
// ═══════════════════════════════════════════════════════
pub mod delay; // Circular delay buffer
pub mod fused; // Fused LIF+synapse (single-pass SIMD)
