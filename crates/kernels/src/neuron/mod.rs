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
//! | `psi_wave_step` | ψ-wave (quantum) | Schrödinger-like + normalization |
//! | `hh_lut_step` | LUT-HH (Hodgkin-Huxley) | LUT-interpolated exponential Euler |

#![allow(non_snake_case)]
use wide::f32x8;
use wide::i32x8;

const RNG_SCALE_16: f32 = 1.0 / 65536.0;
/// 24-bit scale: for Poisson spike generation (fused kernel uses state>>8)
pub(super) const RNG_SCALE_24: f32 = 1.0 / 16777216.0;

/// Per-thread RNG state for deterministic noise sequences.
/// Each thread gets its own independent LCG with the same initial seed,
/// making noise fully deterministic within a single-threaded context
/// and independent across threads.
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
    // Use safe f32::to_bits() instead of transmute — same codegen, no unsafe.
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
/// Each thread has its own seed, so concurrent calls produce independent sequences.
/// Always seeded to 1 at thread start — deterministic within a single step().
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

/// Fast `exp(x)` approximation for a single `f32`.
///
/// Uses a 6th-order Taylor-series polynomial for 2^f on [0, 1):
///   1. t = x · log2(e) — range reduction
///   2. n = floor(t), f = t − n — split integer/fractional
///   3. 2^f ≈ P(f) — 6th-order polynomial on [0, 1)
///   4. 2^n = reinterpret((n + 127) << 23) — IEEE 754 bit trick
///   5. e^x = 2^n · 2^f
///
/// Accuracy: ~1.7e-5 rel. error on [-10, 10] (6× better than previous 4th-order).
/// Speed: ~2–3× faster than `f32::exp()`.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = std::f32::consts::LOG2_E;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    // 2^f: 6th-order Taylor-series polynomial on [0, 1)
    // Coefficients = (ln 2)^k / k! for k=1..6
    // c1 = ln(2) = std::f32::consts::LN_2
    let c1 = std::f32::consts::LN_2;
    let c2 = 0.240_226_5_f32;
    let c3 = 0.055_504_11_f32;
    let c4 = 0.009_618_129_f32;
    let c5 = 0.001_333_356_f32;
    let c6 = 0.000_154_035_3_f32;
    // Horner: (((((c6·f + c5)·f + c4)·f + c3)·f + c2)·f + c1)·f + 1
    let poly = (((c6 * f + c5) * f + c4) * f + c3) * f + c2;
    let poly = (poly * f + c1) * f + 1.0;
    let exp2f = poly;

    // 2^n via IEEE 754 bit trick
    // Guard against n < -126 (wrapping would produce huge positive value)
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

/// Fast `exp(x)` approximation using `f32x8` SIMD.
///
/// **Fully SIMD** — zero scalar loops or lane extraction.
/// Uses `i32x8` integer arithmetic for the IEEE-754 2^n bit trick,
/// eliminating the previous scalar-loop bottleneck that stalled the
/// SIMD pipeline on every call.
///
/// Algorithm:
/// 1. Clamp x to [-87, 87] (full f32 exp range without overflow/underflow)
/// 2. Range reduction: t = x · log2(e)
/// 3. n = floor(t), f = t - n (SIMD via f32x8::floor)
/// 4. 2^f via 6th-order Taylor polynomial on [0, 1) (FMA chain)
/// 5. 2^n via SIMD integer arithmetic: i32x8 `(n + 127) << 23` → f32x8 reinterpret
///
/// Accuracy: ~1.7e-5 relative error on [-87, 87].
/// Speed: ~8× faster than 8 separate `f32::exp()` calls.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    // Clamp to full f32 exp range (exp(-87) ≈ 1.6e-38, exp(87) ≈ 6.1e37)
    let x = x.max(f32x8::splat(-87.0)).min(f32x8::splat(87.0));

    // t = x * log2(e)  (SIMD)
    let log2e = f32x8::splat(std::f32::consts::LOG2_E);
    let t = x * log2e;

    // SIMD floor: split into integer and fractional parts
    let n = t.floor();                       // SIMD floor
    let ff = t - n;                          // SIMD fractional  [0, 1)

    // 6th-order Taylor-series polynomial for 2^f on [0, 1):
    // Coefficients = (ln 2)^k / k! for k=1..6
    let c1 = f32x8::splat(std::f32::consts::LN_2);
    let c2 = f32x8::splat(0.240_226_5_f32);
    let c3 = f32x8::splat(0.055_504_11_f32);
    let c4 = f32x8::splat(0.009_618_129_f32);
    let c5 = f32x8::splat(0.001_333_356_f32);
    let c6 = f32x8::splat(0.000_154_035_3_f32);

    // Horner: P(f) = (((((c6·f + c5)·f + c4)·f + c3)·f + c2)·f + c1)·f + 1
    let poly = ff.mul_add(c6, c5);
    let poly = ff.mul_add(poly, c4);
    let poly = ff.mul_add(poly, c3);
    let poly = ff.mul_add(poly, c2);
    let poly = ff.mul_add(poly, c1);
    let poly = ff.mul_add(poly, f32x8::splat(1.0));

    // 2^n: fully SIMD IEEE-754 bit trick via i32x8
    // Convert n (float) → i32, add bias 127, shift left 23 → reinterpret as f32
    // Input is clamped to [-87, 87], so n ∈ [-126, 125] — biased ∈ [1, 252], always valid.
    let n_i32: i32x8 = n.fast_round_int();   // floor already integer-valued → round is exact
    let biased = n_i32 + i32x8::splat(127);
    let exp2n: f32x8 = bytemuck::cast(biased << 23_u32);

    // e^x = 2^n · 2^f
    exp2n * poly
}

/// Fast sin/cos approximation using `f32x8` SIMD.
///
/// All SIMD arithmetic (range reduction, polynomial) is SIMD via f32x8,
/// but quadrant adjustment uses to_array() + scalar loop since the
/// blend semantics in `wide` are confusing and bug-prone.
///
/// Algorithm:
/// 1. Range reduction: `x → r ∈ [-π/4, π/4]` via `r = x - n·π/2`
/// 2. Odd polynomial for sin(r) on [-π/4, π/4]:
///    `sin(r) ≈ r - r³/6 + r⁵/120 - r⁷/5040`
/// 3. Even polynomial for cos(r) on [-π/4, π/4]:
///    `cos(r) ≈ 1 - r²/2 + r⁴/24 - r⁶/720`
/// 4. Quadrant-based sign/swap from `n mod 4`
///
/// Accuracy: ~1e-4 absolute error on [-10π, 10π].
/// Speed: ~4× faster than 8× sin() + 8× cos() via libm.
#[inline(always)]
pub fn fast_sincos_simd(x: f32x8) -> (f32x8, f32x8) {
    use core::f32::consts::{FRAC_PI_2, FRAC_2_PI};
    let pi2 = f32x8::splat(FRAC_PI_2);
    let rcp_pi2 = f32x8::splat(FRAC_2_PI);

    // 1. Range reduction: x = n·π/2 + r, r ∈ [-π/4, π/4]
    let t = x * rcp_pi2;
    let n = t.round();
    let r = x - n * pi2;
    let r2 = r * r;

    // 2. sin(r) via Taylor: r - r³/6 + r⁵/120 - r⁷/5040
    //    Horner on r²: r * (((c6·r² + c4)·r² + c2)·r² + c0)
    let sin_poly = r2.mul_add(f32x8::splat(-0.000_198_409), f32x8::splat(0.008_333_332));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(-0.166_666_67));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(1.0));
    let sin_val = r * sin_poly;

    // 3. cos(r) via Taylor: 1 - r²/2 + r⁴/24 - r⁶/720
    let cos_poly = r2.mul_add(f32x8::splat(-0.0013888889), f32x8::splat(0.041_666_668));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(-0.5));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(1.0));
    let cos_val = cos_poly;

    // 4. Quadrant adjustment via scalar loop (avoiding blend confusion)
    //    n mod 4: 0→(sin,cos), 1→(cos,-sin), 2→(-sin,-cos), 3→(-cos,sin)
    let four = f32x8::splat(4.0);
    let n_mod_4 = n - four * (n / four).floor();

    let n_arr: [f32; 8] = n_mod_4.to_array();
    let s_arr: [f32; 8] = sin_val.to_array();
    let c_arr: [f32; 8] = cos_val.to_array();
    let mut out_s = [0.0f32; 8];
    let mut out_c = [0.0f32; 8];
    for i in 0..8 {
        let q = n_arr[i] as i32 & 3;  // n mod 4, safe (n is integer from round)
        match q {
            0 => { out_s[i] = s_arr[i]; out_c[i] = c_arr[i]; }
            1 => { out_s[i] = c_arr[i]; out_c[i] = -s_arr[i]; }
            2 => { out_s[i] = -s_arr[i]; out_c[i] = -c_arr[i]; }
            _ => { out_s[i] = -c_arr[i]; out_c[i] = s_arr[i]; }  // 3 or -1
        }
    }

    (f32x8::from(out_s), f32x8::from(out_c))
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
pub mod fhn;
pub mod hr;
pub mod atlasov_neuromotor;
pub mod wc;
pub mod stp;
pub mod synapse;
pub mod dopamine;
pub mod fused;
pub mod delay;
pub mod dendrite;


