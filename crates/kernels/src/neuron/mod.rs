//! Pure Rust SIMD neuron dynamics — `wide::f32x8` for AVX2/NEON/WASM.
//!
//! Each function processes `n` neurons. When `n ≥ 8`, uses `f32x8` for
//! 8-wide SIMD (AVX2 on x86_64, NEON on ARM, portable emulation elsewhere),
//! with a scalar tail for remaining elements.
//!
//! ## Implemented models
//!
//! | Function | Model | Key equations |
//! |---|---|---|
//! | `izhikevich_step` | Izhikevich (2003) | dv/dt = 0.04v² + 5v + 140 − u + I |
//! | `saw_step` | SAW (Spike-Adaptive-Width) | SAW + noise |
//! | `psi_wave_step` | ψ-wave (quantum) | Schrödinger-like + normalization |
//! | `lif_step` | LIF | dv/dt = (v_rest − v + I) / τ_m |
//! | `lif_at_step` | LIF-AT | LIF + adaptive threshold θ |
//! | `adex_step` | AdEx | exponential + adaptation current w |
//! | `hh_lut_step` | LUT-HH | HH ion channels via LUT, zero exp() at runtime |

#![allow(non_snake_case)]
use wide::f32x8;

const RNG_SCALE_24: f32 = 1.0 / 16777216.0;
const RNG_SCALE_16: f32 = 1.0 / 65536.0;

// ═══════════════════════════════════════════════════════
//  Helper: load f32x8 from raw pointer (index offset)
// ═══════════════════════════════════════════════════════

#[inline(always)]
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_mut(ptr: *mut f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_i32(ptr: *const i32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p as f32, *p.add(1) as f32, *p.add(2) as f32, *p.add(3) as f32,
                 *p.add(4) as f32, *p.add(5) as f32, *p.add(6) as f32, *p.add(7) as f32])
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
    // f32x8 comparison returns all-1s (true → 0xFFFFFFFF) or all-0s (false)
    // Reinterpret bits to u32, then convert nonzero → 1
    let bits: [u32; 8] = unsafe { core::mem::transmute(mask.to_array()) };
    [
        (bits[0] != 0) as i32,
        (bits[1] != 0) as i32,
        (bits[2] != 0) as i32,
        (bits[3] != 0) as i32,
        (bits[4] != 0) as i32,
        (bits[5] != 0) as i32,
        (bits[6] != 0) as i32,
        (bits[7] != 0) as i32,
    ]
}

/// Deterministic RNG for SAW noise (thread-safe atomic counter)
fn rand_f32() -> f32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static SEED: AtomicU32 = AtomicU32::new(1);
    let mut s = SEED.fetch_add(1103515245, Ordering::Relaxed);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    SEED.store(s, Ordering::Relaxed);
    (s >> 16) as f32 * RNG_SCALE_16
}

// ═══════════════════════════════════════════════════════
//  Fast exp — SIMD range reduction + polynomial
// ═══════════════════════════════════════════════════════

/// Fast `exp(x)` approximation for a single `f32`.
///
/// Uses the same 4th-order minimax polynomial as `fast_exp_simd`:
///   1. t = x · log2(e) — range reduction
///   2. n = floor(t), f = t − n — split integer/fractional
///   3. 2^f ≈ P(f) — 4th-order polynomial on [0, 1)
///   4. 2^n = reinterpret((n + 127) << 23) — IEEE 754 bit trick
///   5. e^x = 2^n · 2^f
///
/// Accuracy: ~1e-4 rel. error on [-10, 10].
/// Speed: ~2–3× faster than `f32::exp()`.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = std::f32::consts::LOG2_E;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    // 2^f: 4th-order minimax polynomial on [0, 1)
    let c1 = 0.6930321207;
    let c2 = 0.2413797629;
    let c3 = 0.0520972777;
    let c4 = 0.0093553389;
    let poly = ((c4 * f + c3) * f + c2) * f + c1;
    let exp2f = poly * f + 1.0;

    // 2^n via IEEE 754 bit trick
    let biased = n.wrapping_add(127);
    let exp2n = if biased <= 0 {
        0.0
    } else if biased >= 255 {
        f32::INFINITY
    } else {
        f32::from_bits((biased as u32) << 23)
    };

    exp2n * exp2f
}

/// Fast `exp(x)` approximation using `f32x8` SIMD.
///
/// Algorithm:
/// 1. `t = x * log2(e)` — range reduction
/// 2. `n = floor(t)`, `f = t - n` — split integer/fractional
/// 3. `2^f ≈ P(f)` — 4th-order minimax polynomial on [0, 1)
/// 4. `2^n` = `reinterpret((n + 127) << 23)` — IEEE 754 bit trick
/// 5. `e^x = 2^n * 2^f`
///
/// Accuracy: ~1e-4 relative error on [-10, 10].
/// Speed: ~8× faster than 8 separate `f32::exp()` calls.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    // log2(e) ≈ 1.4426950408889634
    let log2e = f32x8::splat(std::f32::consts::LOG2_E);
    let t = x * log2e;

    // Split into integer and fractional parts (per-element for floor + bit ops)
    let arr: [f32; 8] = t.to_array();
    let mut int_part = [0i32; 8];
    let mut frac = [0.0f32; 8];
    let mut exp2_int = [0.0f32; 8]; // 2^int_part

    for j in 0..8 {
        int_part[j] = arr[j].floor() as i32;
        frac[j] = arr[j] - int_part[j] as f32;

        // IEEE 754: 2^n = f32::from_bits((n + 127) << 23)
        let biased = int_part[j].wrapping_add(127);
        if biased <= 0 {
            exp2_int[j] = 0.0; // underflow → 0
        } else if biased >= 255 {
            exp2_int[j] = f32::INFINITY; // overflow
        } else {
            exp2_int[j] = f32::from_bits((biased as u32) << 23);
        }
    }

    // 4th-order minimax polynomial for 2^f on [0, 1):
    // P(f) = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    // Coefficients minimise max relative error on [0, 1).
    let ff = f32x8::from(frac);
    let c1 = f32x8::splat(0.6930321207);       // ln(2) — tuned
    let c2 = f32x8::splat(0.2413797629);       // ln(2)²/2! — tuned
    let c3 = f32x8::splat(0.0520972777);       // ln(2)³/3! — tuned
    let c4 = f32x8::splat(0.0093553389);       // ln(2)⁴/4! — tuned

    // Horner's method: P(f) = (((c4*f + c3)*f + c2)*f + c1)*f + 1
    let poly = ff.mul_add(c4, c3);             // c4·f + c3
    let poly = ff.mul_add(poly, c2);           // (c4·f + c3)·f + c2
    let poly = ff.mul_add(poly, c1);           // ((c4·f + c3)·f + c2)·f + c1
    let poly = ff.mul_add(poly, f32x8::splat(1.0)); // (((c4·f + c3)·f + c2)·f + c1)·f + 1

    // e^x = 2^int_part * poly(frac)
    let exp2 = f32x8::from(exp2_int);
    exp2 * poly
}

/// Surrogate gradient: σ'(v) = 1 / (1 + |v - v_th|/τ)²
#[inline(always)]
fn surrogate_gradient(v: f32, v_thresh: f32, tau_sg: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_sg;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

pub mod izhikevich;
pub mod saw;
pub mod lif;
pub mod adex;
pub mod hh;
pub mod stp;
pub mod dopamine;
pub mod fused;
pub mod delay;


