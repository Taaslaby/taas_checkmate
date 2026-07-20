//! LUT-based Hodgkin-Huxley ion channels — zero exp() calls at runtime.
//!
//! # Idea
//! Standard HH: α(V), β(V) via exp() → m∞, τ → m(t+dt) = m∞ + (m-m∞)·exp(-dt/τ)
//! LUT HH:   precompute m∞(V), decay(V) = exp(-dt/τ(V)) for 1024 V-levels
//!            → runtime: quantize V, LUT → m∞, decay, FMA update. Zero exp().
//!
//! # LUT size
//! 1024 V-levels × (m∞, decay) × N_gates × 4 bytes = 8KB per gate.
//! Typical HH (3 gates: Na-m, Na-h, K-n) = 24KB. L1 cache.
//!
//! # Channel types
//! - `NaV`: m³h — Na⁺ spike
//! - `KV`:  n⁴  — K⁺ rectifier
//! - `Leak`: passive (no gates, just G_leak · (E_leak − V))
//! - Extensible: CaV, KA, KM, HCN, etc.

#![allow(non_snake_case)]
use wide::f32x8;
use wide::CmpLt;
use super::{load8, store8, fast_exp, fast_exp_simd};

// ═══════════════════════════════════════════════════════
//  LUT Generation (call at init, offline)
// ═══════════════════════════════════════════════════════

/// Rate-function closure: α(V) or β(V) in [1/ms].
pub type RateFn = fn(f32) -> f32;

/// A single gate (e.g. Na-m, Na-h, K-n) with its LUT.
#[derive(Debug, Clone)]
pub struct GateLut {
    pub m_inf: Vec<f32>,   // [N_BINS] steady-state
    pub decay: Vec<f32>,   // [N_BINS] exp(-dt / τ(V))
}

/// Channel specification: one or more gates + reversal + conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    pub E_rev: f32,       // mV
    pub g_max: f32,        // mS/cm²
}

#[derive(Debug, Clone)]
pub struct GateSpec {
    pub label: &'static str,
    pub alpha: RateFn,
    pub beta: RateFn,
    pub power: u32,         // exponent in conductance: m^power
}

/// Precompute LUTs for all channels at a given dt and V-range.
///
/// # Safety
/// The channels must have valid alpha/beta rate functions that are defined
/// for the entire voltage range [v_min, v_max].
pub fn build_luts(channels: &[ChannelSpec], dt: f32,                   v_min: f32, v_max: f32, n_bins: usize) -> Vec<Vec<GateLut>> {
    channels.iter().map(|ch| {
        ch.gates.iter().map(|g| {
            let mut m_inf = Vec::with_capacity(n_bins + 1);
            let mut decay = Vec::with_capacity(n_bins + 1);
            for bi in 0..=n_bins {
                let v = v_min + (v_max - v_min) * (bi as f32) / (n_bins as f32);
                let a = (g.alpha)(v);
                let b = (g.beta)(v);
                let sum = a + b;
                let minf = if sum > 1e-12 { a / sum } else { 0.5 };
                let tau = if sum > 1e-12 { 1.0 / sum } else { 1.0 };
                m_inf.push(minf);
                decay.push((-dt / tau).exp());
            }
            GateLut { m_inf, decay }
        }).collect()
    }).collect()
}

/// Quantize voltage to LUT index, clamped.
#[inline]
pub fn v_to_idx(v: f32, v_min: f32, inv_dv: f32, n_bins: usize) -> usize {
    let idx = ((v - v_min) * inv_dv) as isize;
    idx.clamp(0, n_bins as isize) as usize
}

// ═══════════════════════════════════════════════════════
//  Standard HH rate functions (reference)
// ═══════════════════════════════════════════════════════

/// Na activation (m-gate) — Hodgkin & Huxley 1952, squid axon @ 6.3°C
pub fn na_m_alpha(v: f32) -> f32 {
    let v_ = v + 40.0;
    if v_.abs() < 1e-6 { 1.0 } else { 0.1 * v_ / (1.0 - (-v_ / 10.0).exp()) }
}
pub fn na_m_beta(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

/// Na inactivation (h-gate)
pub fn na_h_alpha(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}
pub fn na_h_beta(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

/// K activation (n-gate)
pub fn k_n_alpha(v: f32) -> f32 {
    let v_ = v + 55.0;
    if v_.abs() < 1e-6 { 0.1 } else { 0.01 * v_ / (1.0 - (-v_ / 10.0).exp()) }
}
pub fn k_n_beta(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// CaV-T (T-type calcium) — low-threshold, pacemaking.
/// Hill & Tononi 2005, Wang & Rinzel 1993.
/// Single activation gate (c), constant tau_c = 0.6ms.
pub fn cat_c_alpha(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + (-(v + 56.0) / 6.2).exp());
    m_inf / 0.6
}
pub fn cat_c_beta(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + (-(v + 56.0) / 6.2).exp());
    (1.0 - m_inf) / 0.6
}

/// Standard HH channels: NaV + KV.
/// These produce fast spiking (like Izhikevich RS).
pub fn standard_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec { label: "m", alpha: na_m_alpha, beta: na_m_beta, power: 3 },
                GateSpec { label: "h", alpha: na_h_alpha, beta: na_h_beta, power: 1 },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![
                GateSpec { label: "n", alpha: k_n_alpha, beta: k_n_beta, power: 4 },
            ],
            E_rev: -77.0,
            g_max: 36.0,
        },
    ]
}

/// Compute steady-state gate values at a given membrane potential.
/// Uses alpha/beta rate functions: m_inf = α(V) / (α(V) + β(V))
pub fn steady_state_gates(v: f32, channels: &[ChannelSpec]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channels {
        for gate in &ch.gates {
            let a = (gate.alpha)(v);
            let b = (gate.beta)(v);
            let sum = a + b;
            let minf = if sum > 1e-12 { a / sum } else { 0.5 };
            gates.push(minf);
        }
    }
    gates
}

// ═══════════════════════════════════════════════════════
//  Polynomial HH: fast_exp_simd for rate functions (no LUT)
// ═══════════════════════════════════════════════════════
//
// Replaces LUT gather with direct α(V)/β(V) computation via fast_exp_simd.
// No gather, no precomputation, pure FMA arithmetic.
// ~8× wider than scalar f32::exp(), no cache misses from LUT.

/// SIMD α/β computation helpers (inlined)

/// Compute NaV m-gate α(V) and β(V) using fast_exp_simd.
/// α = 0.1·(v+40) / (1 − exp(−(v+40)/10)), β = 4.0·exp(−(v+65)/18)
#[inline(always)]
unsafe fn na_m_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp40 = v + f32x8::splat(40.0);
    let one = f32x8::splat(1.0);
    // α: handle vp40 ≈ 0 via blend (avoid 0/0)
    let _exp_arg = -vp40 / f32x8::splat(10.0);
    let clamp_min = f32x8::splat(-10.0);
    let clamp_max = f32x8::splat(10.0);
    let exp_val = fast_exp_simd((-vp40 / f32x8::splat(10.0)).max(clamp_min).min(clamp_max));
    let denom = one - exp_val;
    let alpha_num = f32x8::splat(0.1) * vp40;
    let small_mask = vp40.abs().cmp_lt(f32x8::splat(1e-6));
    // denom < 0 для vp40 > 0, denom > 0 для vp40 < 0. Без clamping — small_mask
    // обрабатывает vp40 ≈ 0 через blend. Для |vp40| ≥ 1e-6 denom ≠ 0 всегда.
    let alpha = small_mask.blend(f32x8::splat(1.0), alpha_num / denom);
    let beta = f32x8::splat(4.0) * fast_exp_simd((-(v + f32x8::splat(65.0)) / f32x8::splat(18.0)).max(clamp_min).min(clamp_max));
    (alpha, beta)
}

/// Compute NaV h-gate α(V) and β(V) using fast_exp_simd.
/// α = 0.07·exp(−(v+65)/20), β = 1/(1 + exp(−(v+35)/10))
#[inline(always)]
unsafe fn na_h_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let clamp_min = f32x8::splat(-10.0);
    let clamp_max = f32x8::splat(10.0);
    let alpha = f32x8::splat(0.07) * fast_exp_simd((-(v + f32x8::splat(65.0)) / f32x8::splat(20.0)).max(clamp_min).min(clamp_max));
    let exp_arg = (-(v + f32x8::splat(35.0)) / f32x8::splat(10.0)).max(clamp_min).min(clamp_max);
    let beta = one / (one + fast_exp_simd(exp_arg));
    (alpha, beta)
}

/// Compute KV n-gate α(V) and β(V) using fast_exp_simd.
/// α = 0.01·(v+55) / (1 − exp(−(v+55)/10)), β = 0.125·exp(−(v+65)/80)
#[inline(always)]
unsafe fn k_n_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp55 = v + f32x8::splat(55.0);
    let one = f32x8::splat(1.0);
    let clamp_min = f32x8::splat(-10.0);
    let clamp_max = f32x8::splat(10.0);
    let exp_val = fast_exp_simd((-vp55 / f32x8::splat(10.0)).max(clamp_min).min(clamp_max));
    let denom = one - exp_val;
    let alpha_num = f32x8::splat(0.01) * vp55;
    let small_mask = vp55.abs().cmp_lt(f32x8::splat(1e-6));
    // Без sign-preserving clamp — denom ≠ 0 для |vp55| ≥ 1e-6, small_mask
    // обрабатывает vp55 ≈ 0.
    let alpha = small_mask.blend(f32x8::splat(0.1), alpha_num / denom);
    let beta = f32x8::splat(0.125) * fast_exp_simd((-(v + f32x8::splat(65.0)) / f32x8::splat(80.0)).max(clamp_min).min(clamp_max));
    (alpha, beta)
}

/// Compute CaT c-gate α(V) and β(V) using fast_exp_simd.
/// α = m_inf / 0.6, β = (1 − m_inf) / 0.6
/// where m_inf = 1/(1 + exp(−(v+56)/6.2))
#[inline(always)]
unsafe fn cat_c_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let clamp_min = f32x8::splat(-10.0);
    let clamp_max = f32x8::splat(10.0);
    let m_inf = one / (one + fast_exp_simd((-(v + f32x8::splat(56.0)) / f32x8::splat(6.2)).max(clamp_min).min(clamp_max)));
    let tau_inv = f32x8::splat(1.0 / 0.6);
    (m_inf * tau_inv, (one - m_inf) * tau_inv)
}

/// Channel gate index metadata (precomputed at init from ChannelSpec).
pub struct ChannelMeta {
    pub n_gates: usize,        // gates per channel
    pub e_rev: f32,
    pub g_max: f32,
    pub powers: Vec<u32>,      // exponent per gate
    pub gate_type: Vec<GateType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    NaM, NaH, Kn, CatC,
}

/// Build ChannelMeta from standard ChannelSpec (for poly HH).
pub fn build_channel_meta(channels: &[ChannelSpec]) -> Vec<ChannelMeta> {
    channels.iter().map(|ch| {
        let n_gates = ch.gates.len();
        let mut powers = Vec::with_capacity(n_gates);
        let mut gate_type = Vec::with_capacity(n_gates);
        for g in &ch.gates {
            powers.push(g.power);
            gate_type.push(match g.label {
                "m" => GateType::NaM,
                "h" => GateType::NaH,
                "n" => GateType::Kn,
                "c" => GateType::CatC,
                _ => GateType::NaM, // fallback
            });
        }
        ChannelMeta { n_gates, e_rev: ch.E_rev, g_max: ch.g_max, powers, gate_type }
    }).collect()
}

/// SIMD polynomial HH step — direct rate computation (no LUT).
///
/// For each 8-neuron SIMD block:
///   1. Load V
///   2. For each channel: compute m_inf(V), τ(V) via fast_exp_simd → gate update
///   3. I_ion += g_ch · (V − E_rev)
///   4. V += (I_ext − I_ion − I_leak) / C_m · dt
///
/// # Safety
/// All pointers valid for n elements, non-overlapping.
pub(crate) unsafe fn hh_poly_step_simd(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    dt: f32,
) {
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let eps = f32x8::splat(1e-12);
    let dt_v = f32x8::splat(dt);

    for i in (0..n).step_by(8) {
        if i + 8 > n { break; }
        let vi = unsafe { load8(v, i) };
        let mut I_ion = zero;
        let mut gi = 0;  // global gate index

        for ch in channel_meta {
            let e_rev_v = f32x8::splat(ch.e_rev);
            let g_max_v = f32x8::splat(ch.g_max);
            let mut g_ch = g_max_v;

            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                // Compute α(V), β(V) via fast_exp_simd
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => na_m_alpha_beta_simd(vi),
                    GateType::NaH => na_h_alpha_beta_simd(vi),
                    GateType::Kn => k_n_alpha_beta_simd(vi),
                    GateType::CatC => cat_c_alpha_beta_simd(vi),
                };

                let sum = (alpha + beta).max(eps);
                let m_inf = alpha / sum;
                let tau = one / sum;
                let decay = fast_exp_simd((-dt_v / tau).max(f32x8::splat(-10.0)).min(f32x8::splat(10.0)));

                // Update gate: m(t+dt) = m_inf + (m − m_inf)·decay
                let gi_ptr = gates.add(gi * n + i);
                let gv = unsafe { load8(gi_ptr, 0) };
                let new_gv = m_inf + (gv - m_inf) * decay;
                unsafe { store8(gi_ptr, 0, new_gv); }

                // g_ch *= m^power
                if power == 1 {
                    g_ch *= new_gv;
                } else if power == 3 {
                    let sq = new_gv * new_gv;
                    g_ch = g_ch * sq * new_gv;
                } else if power == 4 {
                    let sq = new_gv * new_gv;
                    g_ch = g_ch * sq * sq;
                } else {
                    g_ch *= new_gv.powf(power as f32);
                }
                gi += 1;
            }
            I_ion += g_ch * (vi - e_rev_v);
        }

        let I_leak = unsafe { load8(g_leak, i) } * (vi - unsafe { load8(E_leak, i) });
        let cm = unsafe { load8(C_m, i) };
        let dV = (unsafe { load8(Ii, i) } - I_ion - I_leak) / cm;
        unsafe { store8(v, i, vi + dV * dt_v); }
    }
}

/// Scalar polynomial HH step — for remainder (n % 8 != 0).
/// Uses scalar fast_exp for rate functions.
///
/// `n`: number of neurons in this chunk (remainder).
/// `total_n`: total neurons in the original array (column-major stride for gates).
///
/// # Safety
/// All pointer args (`v`, `gates`, `Ii`, `g_leak`, `E_leak`, `C_m`) must be valid
/// for `n` elements. `channel_meta` must be valid for its length.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hh_poly_step_scalar(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize, total_n: usize,
    channel_meta: &[ChannelMeta],
    dt: f32,
) {
    for i in 0..n {
        let vi = *v.add(i);
        let mut I_ion = 0.0;
        let mut gi = 0;

        for ch in channel_meta {
            let mut g_ch = ch.g_max;
            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                // Scalar α/β via fast_exp
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => {
                        let vp40 = vi + 40.0;
                        let a = if vp40.abs() < 1e-6 { 1.0 } else { 0.1 * vp40 / (1.0 - fast_exp(-vp40 / 10.0)) };
                        let b = 4.0 * super::fast_exp(-(vi + 65.0) / 18.0);
                        (a, b)
                    },
                    GateType::NaH => {
                        let a = 0.07 * super::fast_exp(-(vi + 65.0) / 20.0);
                        let b = 1.0 / (1.0 + super::fast_exp(-(vi + 35.0) / 10.0));
                        (a, b)
                    },
                    GateType::Kn => {
                        let vp55 = vi + 55.0;
                        let a = if vp55.abs() < 1e-6 { 0.1 } else { 0.01 * vp55 / (1.0 - fast_exp(-vp55 / 10.0)) };
                        let b = 0.125 * super::fast_exp(-(vi + 65.0) / 80.0);
                        (a, b)
                    },
                    GateType::CatC => {
                        let m_inf = 1.0 / (1.0 + super::fast_exp(-(vi + 56.0) / 6.2));
                        (m_inf / 0.6, (1.0 - m_inf) / 0.6)
                    },
                };
                let sum = alpha + beta;
                let m_inf = if sum > 1e-12 { alpha / sum } else { 0.5 };
                let tau = if sum > 1e-12 { 1.0 / sum } else { 1.0 };
                let decay = fast_exp(-dt / tau);

                // Gates are column-major: strides of total_n, not n
                let p = gates.add(gi * total_n + i);
                *p = m_inf + (*p - m_inf) * decay;
                g_ch *= (*p).powi(power as i32);
                gi += 1;
            }
            I_ion += g_ch * (vi - ch.e_rev);
        }
        let I_leak = *g_leak.add(i) * (vi - *E_leak.add(i));
        let dV = (*Ii.add(i) - I_ion - I_leak) / *C_m.add(i);
        *v.add(i) = vi + dV * dt;
    }
}

/// Public polynomial HH step — auto-selects SIMD or scalar.
///
/// Replaces LUT-based `hh_lut_step` for architectures where SIMD gather is slow.
/// Pure arithmetic: α(V), β(V) computed via `fast_exp` (minimax polynomial).
pub unsafe fn hh_poly_step(
    v: *mut f32, gates: *mut f32, Ii: *const f32,
    g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    dt: f32,
) {
    if n >= 8 {
        hh_poly_step_simd(v, gates, Ii, g_leak, E_leak, C_m,
            n, channel_meta, dt);
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        // Pass total_n = n for column-major stride correction
        hh_poly_step_scalar(
            v.add(simd_end), gates.add(simd_end), Ii.add(simd_end),
            g_leak.add(simd_end), E_leak.add(simd_end), C_m.add(simd_end),
            remainder, n, channel_meta, dt);
    }
}

/// HH with CaV-T (T-type calcium) for pacemaking.
/// Adds low-threshold Ca²⁺ current — neurons fire spontaneously without input.
/// 4 gates: NaV-m, NaV-h, KV-n, CaT-c
pub fn pacemaking_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec { label: "m", alpha: na_m_alpha, beta: na_m_beta, power: 3 },
                GateSpec { label: "h", alpha: na_h_alpha, beta: na_h_beta, power: 1 },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![
                GateSpec { label: "n", alpha: k_n_alpha, beta: k_n_beta, power: 4 },
            ],
            E_rev: -77.0,
            g_max: 36.0,
        },
        ChannelSpec {
            name: "CaT",
            gates: vec![
                GateSpec { label: "c", alpha: cat_c_alpha, beta: cat_c_beta, power: 1 },
            ],
            E_rev: 120.0,
            g_max: 0.5,
        },
    ]
}

// ═══════════════════════════════════════════════════════
//  SIMD LUT-HH step kernel
// ═══════════════════════════════════════════════════════

/// Scalar HH step using LUTs.
/// Channels are interleaved in params: for each channel (E_rev, g_max, n_gates_in_ch).
/// gates array: column-major, [gate_0(nrn 0..n), gate_1(nrn 0..n), ...]
///
/// # Safety
/// All pointer args (`v`, `gates`, `Ii`, `g_leak`, `E_leak`, `C_m`, `m_inf_luts`,
/// `decay_luts`, `E_rev_arr`, `g_max_arr`, `powers`, `n_gates_per_ch`) must be
/// valid for their respective lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn hh_lut_step_scalar(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize,
    _n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: usize,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize,
    _g_leak_idx: usize, dt: f32,
) {
    let inv_dv = (n_bins as f32) / (v_max - v_min);
    for i in 0..n {
        let vi = *v.add(i);
        let idx = v_to_idx(vi, v_min, inv_dv, n_bins);
        let mut I_ion = 0.0_f32;
        let mut gi = 0;  // global gate index
        
        for ch in 0..n_channels {
            let ng = *n_gates_per_ch.add(ch);
            let e_rev = *E_rev_arr.add(ch);
            let g_max = *g_max_arr.add(ch);
            let mut g_ch = g_max;
            
            for _ in 0..ng {
                let m_inf_arr = *m_inf_luts.add(gi);
                let decay_arr = *decay_luts.add(gi);
                let power = *powers.add(gi);
                let p = gates.add(gi * n + i);
                let m_inf = *m_inf_arr.add(idx);
                let decay = *decay_arr.add(idx);
                *p = m_inf + (*p - m_inf) * decay;
                g_ch *= (*p).powi(power as i32);
                gi += 1;
            }
            I_ion += g_ch * (vi - e_rev);
        }
        let I_leak = *g_leak.add(i) * (vi - *E_leak.add(i));
        let cm = *C_m.add(i);
        let dV = (*Ii.add(i) - I_ion - I_leak) / cm;
        *v.add(i) = vi + dV * dt;
    }
}

// ═══════════════════════════════════════════════════════
//  SIMD (f32x8) HH step — 8 neurons at once
// ═══════════════════════════════════════════════════════

pub(crate) unsafe fn hh_lut_step_simd(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize, _n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: f32,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize,
    dt: f32,
) {
    let inv_dv = n_bins / (v_max - v_min);
    let v_min_v = f32x8::splat(v_min);
    let inv_dv_v = f32x8::splat(inv_dv);
    let zero = f32x8::splat(0.0);
    let n_bins_f = f32x8::splat(n_bins);
    
    for i in (0..n).step_by(8) {
        if i + 8 > n { break; }
        let vi = load8(v, i);
        let idx_f = ((vi - v_min_v) * inv_dv_v).max(zero).min(n_bins_f);
        let mut I_ion = f32x8::splat(0.0);
        let mut gi = 0;  // global gate index
        
        for ch in 0..n_channels {
            let ng = *n_gates_per_ch.add(ch);
            let e_rev_v = f32x8::splat(*E_rev_arr.add(ch));
            let g_max_v = f32x8::splat(*g_max_arr.add(ch));
            let mut g_ch = g_max_v;
            
            for _ in 0..ng {
                let m_inf_arr = *m_inf_luts.add(gi);
                let decay_arr = *decay_luts.add(gi);
                let power = *powers.add(gi);
                
                let mut m_inf_a = [0.0f32; 8];
                let mut decay_a = [0.0f32; 8];
                let idx_a = idx_f.to_array();
                for k in 0..8 {
                    let fi = (idx_a[k] as usize).min(n_bins as usize);
                    m_inf_a[k] = *m_inf_arr.add(fi);
                    decay_a[k] = *decay_arr.add(fi);
                }
                let m_inf_v = f32x8::from(m_inf_a);
                let decay_v = f32x8::from(decay_a);
                
                let gi_ptr = gates.add(gi * n + i);
                let gv = load8(gi_ptr, 0);
                let new_gv = m_inf_v + (gv - m_inf_v) * decay_v;
                store8(gi_ptr, 0, new_gv);
                g_ch *= new_gv.powf(power as f32);
                gi += 1;
            }
            I_ion += g_ch * (vi - e_rev_v);
        }
        
        let I_leak = load8(g_leak, i) * (vi - load8(E_leak, i));
        let cm = load8(C_m, i);
        let dV = (load8(Ii, i) - I_ion - I_leak) / cm;
        store8(v, i, vi + dV * f32x8::splat(dt));
    }
}

/// Public entry: auto-chooses SIMD or scalar.
pub unsafe fn hh_lut_step(
    v: *mut f32, gates: *mut f32, Ii: *const f32,
    g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize, n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: f32,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize, dt: f32,
) {
    if n >= 8 {
        hh_lut_step_simd(v, gates, Ii, g_leak, E_leak, C_m,
            n, n_gates, m_inf_luts, decay_luts, v_min, v_max, n_bins,
            E_rev_arr, g_max_arr, powers, n_gates_per_ch, n_channels, dt);
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        hh_lut_step_scalar(
            v.add(simd_end), gates.add(simd_end), Ii.add(simd_end),
            g_leak.add(simd_end), E_leak.add(simd_end), C_m.add(simd_end),
            remainder, n_gates,
            m_inf_luts, decay_luts, v_min, v_max, n_bins as usize,
            E_rev_arr, g_max_arr, powers,
            n_gates_per_ch, n_channels, !0, dt);
    }
}
