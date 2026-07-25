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
//! | `lif_step` | LIF (Leaky Integrate-and-Fire) | Exponential Euler — exact for constant I |
//! | `lif_at_step` | LIF-AT (adaptive threshold) | Exponential Euler + θ·exp(-dt/τ_θ) |
//! | `eprop_lif_step` | eProp LIF (eligibility trace) | Exponential Euler + exact eligibility |
//! | `izhikevich_step` | Izhikevich (2003) | Forward Euler (quadratic v² term) |
//! | `adex_step` | AdEx (adaptive exponential) | Forward Euler + exponential spike |
//! | `saw_step` | SAW (Spike-Adaptive-Width) | Forward Euler + noise |
//! | `gco_step` | Graph-coupled oscillator | coupled phases on unit circle |
//! | `hh_lut_step` | LUT-HH (Hodgkin-Huxley) | LUT-interpolated exponential Euler |

#![allow(non_snake_case)]

use wide::f32x8;
use wide::i32x8;

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
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
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
unsafe fn load8_mut(ptr: *mut f32, off: usize) -> f32x8 {
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
unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
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
//  Fast exp — SIMD range reduction + polynomial
// ═══════════════════════════════════════════════════════

#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = std::f32::consts::LOG2_E;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    let c1 = std::f32::consts::LN_2;
    let c2 = 0.240_226_5_f32;
    let c3 = 0.055_504_11_f32;
    let c4 = 0.009_618_129_f32;
    let c5 = 0.001_333_356_f32;
    let c6 = 0.000_154_035_3_f32;
    let poly = (((c6 * f + c5) * f + c4) * f + c3) * f + c2;
    let poly = (poly * f + c1) * f + 1.0;
    let exp2f = poly;

    let exp2n = if n < -126 {
        0.0
    } else if n > 127 {
        f32::INFINITY
    } else {
        let biased = (n + 127) as u32;
        f32::from_bits(biased << 23)
    };

    exp2n * exp2f
}

#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    let x = x.max(f32x8::splat(-87.0)).min(f32x8::splat(87.0));
    let log2e = f32x8::splat(std::f32::consts::LOG2_E);
    let t = x * log2e;

    let n = t.floor();
    let ff = t - n;

    let c1 = f32x8::splat(std::f32::consts::LN_2);
    let c2 = f32x8::splat(0.240_226_5_f32);
    let c3 = f32x8::splat(0.055_504_11_f32);
    let c4 = f32x8::splat(0.009_618_129_f32);
    let c5 = f32x8::splat(0.001_333_356_f32);
    let c6 = f32x8::splat(0.000_154_035_3_f32);

    let poly = ff.mul_add(c6, c5);
    let poly = ff.mul_add(poly, c4);
    let poly = ff.mul_add(poly, c3);
    let poly = ff.mul_add(poly, c2);
    let poly = ff.mul_add(poly, c1);
    let poly = ff.mul_add(poly, f32x8::splat(1.0));

    let n_i32: i32x8 = n.fast_round_int();
    let biased = n_i32 + i32x8::splat(127);
    let exp2n: f32x8 = bytemuck::cast(biased << 23_u32);

    exp2n * poly
}

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
pub mod hh; // LUT Hodgkin-Huxley (NaV, KV, CaV-T)
pub mod hr;
pub mod izhikevich; // Izhikevich (2003)
pub mod lif; // LIF + LIF-AT + eProp-LIF // Hindmarsh-Rose (bursting)

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
pub mod eprop;
pub mod stp; // Short-Term Plasticity (Tsodyks-Markram)
pub mod synapse; // Multi-receptor dynamics (nAChR, GluCl, GABA_A/B) // Eligibility Propagation traces (Bellec 2020)

// ═══════════════════════════════════════════════════════
//  🏋️ Motor / Embodied Kernels (мышцы и телесная физика)
// ═══════════════════════════════════════════════════════
pub mod atlasov_neuromotor; // Atlasov Ca²⁺ muscle + Tug-of-War joints
pub mod dendrite; // Cable dendrite + active Na⁺/K⁺/Ca²⁺ channels

// ═══════════════════════════════════════════════════════
//  🧫 Unconventional (нестандартные биосистемы)
// ═══════════════════════════════════════════════════════
pub mod astrocyte; // Astrocyte Ca²⁺ waves + Gap Junctions

// ═══════════════════════════════════════════════════════
//  ⚡ Infrastructure (инфраструктура и оптимизации)
// ═══════════════════════════════════════════════════════
pub mod delay;
pub mod fused; // Fused LIF+synapse (single-pass SIMD) // Circular delay buffer
