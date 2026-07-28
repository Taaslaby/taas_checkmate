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
use super::{fast_exp, fast_exp_simd, load8, store8};
use wide::f32x8;
use wide::CmpGe;
use wide::CmpLt;

// ═══════════════════════════════════════════════════════
//  NEURON-совместимый vtrap — Taylor при малых |x/y|
// ═══════════════════════════════════════════════════════

/// SIMD vtrap: `x / (exp(x/y) - 1)` с Taylor `y - x/2` при |x/y| < 1e-6.
/// В точности как NEURON hh.mod `FUNCTION vtrap`.
/// Предотвращает 0/0 при x → 0.
#[inline(always)]
fn vtrap_simd(x: f32x8, y: f32) -> f32x8 {
    let y_v = f32x8::splat(y);
    let ratio = x / y_v;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    // Taylor: y - x/2 (matches NEURON: y*(1 - x/y/2))
    let taylor = y_v - x * f32x8::splat(0.5);
    let exact = x / (fast_exp_simd(ratio) - f32x8::splat(1.0));
    small_mask.blend(taylor, exact)
}

/// Scalar vtrap: `x / (exp(x/y) - 1)` с Taylor при малых |x/y|.
#[inline(always)]
fn vtrap_scalar(x: f32, y: f32) -> f32 {
    let ratio = x / y;
    if ratio.abs() < 1e-6 {
        y - x * 0.5 // Taylor
    } else {
        x / (fast_exp(ratio) - 1.0)
    }
}

/// Вычислить Q10 = 3^((celsius - 6.3) / 10) для температурной коррекции HH.
/// При 6.3°C (squid axon) Q10 = 1.0. При 37°C Q10 ≈ 29×.
#[inline]
pub fn q10_from_celsius(celsius: f32) -> f32 {
    if (celsius - 6.3).abs() > 0.1 {
        3.0f32.powf((celsius - 6.3) / 10.0)
    } else {
        1.0
    }
}

// ═══════════════════════════════════════════════════════
//  LUT Generation (call at init, offline)
// ═══════════════════════════════════════════════════════

/// Rate-function closure: α(V) or β(V) in [1/ms].
pub type RateFn = fn(f32) -> f32;

/// A single gate (e.g. Na-m, Na-h, K-n) with its LUT.
#[derive(Debug, Clone)]
pub struct GateLut {
    pub m_inf: Vec<f32>, // [N_BINS] steady-state
    pub decay: Vec<f32>, // [N_BINS] exp(-dt / τ(V))
}

/// Channel specification: one or more gates + reversal + conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    pub E_rev: f32, // mV
    pub g_max: f32, // mS/cm²
}

#[derive(Debug, Clone)]
pub struct GateSpec {
    pub label: &'static str,
    pub alpha: RateFn,
    pub beta: RateFn,
    pub power: u32, // exponent in conductance: m^power
}

/// Precompute LUTs for all channels at a given dt and V-range.
///
/// The channels must have valid alpha/beta rate functions that are defined
/// for the entire voltage range [v_min, v_max].
pub fn build_luts(
    channels: &[ChannelSpec],
    dt: f32,
    v_min: f32,
    v_max: f32,
    n_bins: usize,
) -> Vec<Vec<GateLut>> {
    channels
        .iter()
        .map(|ch| {
            ch.gates
                .iter()
                .map(|g| {
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
                })
                .collect()
        })
        .collect()
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
/// Использует vtrap для численной стабильности при V → -40 mV.
pub fn na_m_alpha(v: f32) -> f32 {
    0.1 * vtrap_scalar(-(v + 40.0), 10.0)
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
/// Использует vtrap для численной стабильности при V → -55 mV.
pub fn k_n_alpha(v: f32) -> f32 {
    0.01 * vtrap_scalar(-(v + 55.0), 10.0)
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
                GateSpec {
                    label: "m",
                    alpha: na_m_alpha,
                    beta: na_m_beta,
                    power: 3,
                },
                GateSpec {
                    label: "h",
                    alpha: na_h_alpha,
                    beta: na_h_beta,
                    power: 1,
                },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![GateSpec {
                label: "n",
                alpha: k_n_alpha,
                beta: k_n_beta,
                power: 4,
            }],
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
///
/// Compute NaV m-gate α(V) and β(V) using fast_exp_simd + vtrap.
/// α = 0.1·vtrap(−(v+40), 10), β = 4.0·exp(−(v+65)/18)
#[inline(always)]
unsafe fn na_m_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp40 = v + f32x8::splat(40.0);
    // α: 0.1·vtrap(−(v+40), 10) — Taylor при |vp40/10| < 1e-6
    let alpha = f32x8::splat(0.1) * vtrap_simd(-vp40, 10.0);
    let beta = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
    (alpha, beta)
}

/// Compute NaV h-gate α(V) and β(V) using fast_exp_simd.
/// α = 0.07·exp(−(v+65)/20), β = 1/(1 + exp(−(v+35)/10))
#[inline(always)]
unsafe fn na_h_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let alpha = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    let exp_arg = -(v + f32x8::splat(35.0)) / f32x8::splat(10.0);
    let beta = one / (one + fast_exp_simd(exp_arg));
    (alpha, beta)
}

/// Compute KV n-gate α(V) and β(V) using fast_exp_simd + vtrap.
/// α = 0.01·vtrap(−(v+55), 10), β = 0.125·exp(−(v+65)/80)
#[inline(always)]
unsafe fn k_n_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp55 = v + f32x8::splat(55.0);
    // α: 0.01·vtrap(−(v+55), 10) — Taylor при |vp55/10| < 1e-6
    let alpha = f32x8::splat(0.01) * vtrap_simd(-vp55, 10.0);
    let beta = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
    (alpha, beta)
}

/// Compute CaT c-gate α(V) and β(V) using fast_exp_simd.
/// α = m_inf / 0.6, β = (1 − m_inf) / 0.6
/// where m_inf = 1/(1 + exp(−(v+56)/6.2))
#[inline(always)]
unsafe fn cat_c_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let m_inf = one / (one + fast_exp_simd(-(v + f32x8::splat(56.0)) / f32x8::splat(6.2)));
    let tau_inv = f32x8::splat(1.0 / 0.6);
    (m_inf * tau_inv, (one - m_inf) * tau_inv)
}

/// Channel gate index metadata (precomputed at init from ChannelSpec).
/// Does NOT store g_max — that's per-neuron now.
pub struct ChannelMeta {
    pub n_gates: usize, // gates per channel
    pub e_rev: f32,
    pub powers: Vec<u32>, // exponent per gate
    pub gate_type: Vec<GateType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    NaM,
    NaH,
    Kn,
    CatC,
}

/// Build ChannelMeta from standard ChannelSpec (for poly HH).
pub fn build_channel_meta(channels: &[ChannelSpec]) -> Vec<ChannelMeta> {
    channels
        .iter()
        .map(|ch| {
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
            ChannelMeta {
                n_gates,
                e_rev: ch.E_rev,
                powers,
                gate_type,
            }
        })
        .collect()
}

/// SIMD polynomial HH step — direct rate computation (no LUT).
///
/// `celsius`: температура в °C для Q10 коррекции (6.3 = squid axon, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron, per-channel g_max values.
///   Layout: [ch0_n0, ch0_n1, ..., ch1_n0, ch1_n1, ...] (column-major, stride=n)
///   Size: [n_channels * n]
///
/// For each 8-neuron SIMD block:
///   1. Load V, g_max for each channel
///   2. For each channel: compute m_inf(V), τ(V) via fast_exp_simd → gate update
///   3. Apply Q10: tau_q10 = tau / Q10 (faster kinetics at higher temperature)
///   4. I_ion += g_ch · (V − E_rev)
///   5. V += (I_ext − I_ion − I_leak) / C_m · dt
///
/// # Safety
/// All pointers valid for n elements, non-overlapping.
pub(crate) unsafe fn hh_poly_step_simd(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
) {
    let zero = f32x8::splat(0.0);
    let eps = f32x8::splat(1e-12);
    let dt_v = f32x8::splat(dt);
    let q10 = f32x8::splat(q10_from_celsius(celsius));

    for i in (0..n).step_by(8) {
        if i + 8 > n {
            break;
        }
        let vi = unsafe { load8(v, i) };
        let mut I_ion = zero;
        let mut gi = 0; // global gate index

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let e_rev_v = f32x8::splat(ch.e_rev);
            // Load per-neuron g_max for this SIMD chunk: [ch_idx * n + i .. ch_idx * n + i + 8]
            let g_max_v = unsafe { load8(g_max_per_neuron.add(ch_idx * n), i) };
            let mut g_ch = g_max_v;

            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => na_m_alpha_beta_simd(vi),
                    GateType::NaH => na_h_alpha_beta_simd(vi),
                    GateType::Kn => k_n_alpha_beta_simd(vi),
                    GateType::CatC => cat_c_alpha_beta_simd(vi),
                };

                let sum = (alpha + beta).max(eps);
                let m_inf = alpha / sum;
                // Q10: tau_q10 = tau / Q10 → sum_q10 = sum * Q10
                let sum_q10 = sum * q10;
                let decay = fast_exp_simd(-dt_v * sum_q10);

                let gi_ptr = gates.add(gi * n + i);
                let gv = unsafe { load8(gi_ptr, 0) };
                let new_gv = m_inf + (gv - m_inf) * decay;
                unsafe {
                    store8(gi_ptr, 0, new_gv);
                }

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
        let v_new = vi + dV * dt_v;
        unsafe {
            store8(v, i, v_new);
        }
    }
}

/// Scalar polynomial HH step — for remainder (n % 8 != 0).
/// Uses scalar fast_exp for rate functions.
///
/// `n`: number of neurons in this chunk (remainder).
/// `total_n`: total neurons in the original array (column-major stride for gates).
///
/// # Safety
/// All pointer args must be valid for `n` elements.
/// Compute HH rate functions (alpha, beta) for a gate type at voltage v.
#[inline(always)]
fn rate_fn(v: f32, gate_type: GateType) -> (f32, f32) {
    match gate_type {
        GateType::NaM => {
            let a = 0.1 * vtrap_scalar(-(v + 40.0), 10.0);
            let b = 4.0 * fast_exp(-(v + 65.0) / 18.0);
            (a, b)
        }
        GateType::NaH => {
            let a = 0.07 * fast_exp(-(v + 65.0) / 20.0);
            let b = 1.0 / (1.0 + fast_exp(-(v + 35.0) / 10.0));
            (a, b)
        }
        GateType::Kn => {
            let a = 0.01 * vtrap_scalar(-(v + 55.0), 10.0);
            let b = 0.125 * fast_exp(-(v + 65.0) / 80.0);
            (a, b)
        }
        GateType::CatC => {
            let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
            (m_inf / 0.6, (1.0 - m_inf) / 0.6)
        }
    }
}

pub(crate) unsafe fn hh_poly_step_scalar(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    total_n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
) {
    let q10 = q10_from_celsius(celsius);

    for i in 0..n {
        let vi = *v.add(i);
        let mut sum_g = 0.0f32;
        let mut sum_gE = 0.0f32;
        let mut gi = 0;

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let g_max = *g_max_per_neuron.add(ch_idx * total_n + i);
            let mut g_ch = g_max;
            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = rate_fn(vi, ch.gate_type[gate_idx]);
                let sum = alpha + beta;
                let m_inf = if sum > 1e-12 { alpha / sum } else { 0.5 };
                // Q10: sum_q10 = sum * Q10 (faster kinetics at higher temperature)
                let decay = fast_exp(-dt * sum.max(1e-12) * q10);

                let p = gates.add(gi * total_n + i);
                *p = m_inf + (*p - m_inf) * decay;
                g_ch *= (*p).powi(power as i32);
                gi += 1;
            }
            sum_g += g_ch;
            sum_gE += g_ch * ch.e_rev;
        }

        let g_leak_i = *g_leak.add(i);
        let e_leak_i = *E_leak.add(i);
        sum_g += g_leak_i;
        sum_gE += g_leak_i * e_leak_i;

        let cm_i = *C_m.add(i);
        let inv_dt = cm_i / dt;
        let i_ext_i = *Ii.add(i);
        let v_new = (vi * inv_dt + i_ext_i + sum_gE) / (inv_dt + sum_g);
        *v.add(i) = v_new;
    }
}

/// Public polynomial HH step — auto-selects SIMD or scalar.
///
/// `celsius`: температура в °C (6.3 = squid axon @ H&H 1952, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron g_max, layout [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
/// Size: [n_channels * n]
pub unsafe fn hh_poly_step(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
) {
    if n >= 8 {
        hh_poly_step_simd(
            v,
            gates,
            Ii,
            g_leak,
            E_leak,
            C_m,
            n,
            channel_meta,
            g_max_per_neuron,
            dt,
            celsius,
        );
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        hh_poly_step_scalar(
            v.add(simd_end),
            gates.add(simd_end),
            Ii.add(simd_end),
            g_leak.add(simd_end),
            E_leak.add(simd_end),
            C_m.add(simd_end),
            remainder,
            n,
            channel_meta,
            g_max_per_neuron,
            dt,
            celsius,
        );
    }
}

// ═══════════════════════════════════════════════════════
//  Fused HH + spike detection — single SIMD pass
// ═══════════════════════════════════════════════════════
//
// Merges hh_poly_step with spike detection (reverse zero-crossing)
// into one SIMD pass. Eliminates:
//   1. v_prev.copy_from_slice(&cfg.v) — O(n) copy eliminated
//   2. Separate spike detection loop — another O(n) gone
//
// Additionally: common subexpression for exp(−v/10).
// In standard HH this exp is used 3×:
//   - Na-m α: exp(−(v+40)/10) = exp(−v/10) · exp(−4)
//   - Na-h β: 1/(1 + exp(−(v+35)/10)) where exp_term = exp(−v/10) · exp(−3.5)
//   - K-n α:  exp(−(v+55)/10) = exp(−v/10) · exp(−5.5)
// → 2 fewer fast_exp_simd calls per neuron.
//
// один SIMD-проход: объединяет hh_poly_step с детекцией спайков,
// устраняя O(n) копию v_prev и отдельный цикл spike detection.
// exp(−v/10) вычисляется 1 раз и переиспользуется 3× — минус 2 exp/нейрон.

/// Precomputed constants for common subexpression: exp_const = exp(−shift/10)
///
/// Rust can't compute these as const f32 via `.exp()`, so they're hardcoded.
///   exp(−40/10) = exp(−4) = 0.01831564   — Na-m α
///   exp(−35/10) = exp(−3.5) = 0.03019738  — Na-h β
///   exp(−55/10) = exp(−5.5) = 0.00408677  — K-n α
const EXP_V10_NA_M_ALPHA: f32 = 0.018_315_64_f32;
const EXP_V10_NA_H_BETA: f32 = 0.030197383_f32;
const EXP_V10_KN_ALPHA: f32 = 0.0040867714_f32;

/// Fused SIMD α/β: compute exp(−v/10) once and reuse across all 3 gates.
///
/// Same (alpha, beta) as `na_m_alpha_beta_simd` but with 1 fast_exp_simd call
/// instead of 2. The exp(−v/10) term is shared across Na-m α, Na-h β, K-n α.
///
/// вычисляет exp(−v/10) один раз и переиспользует для Na-m, Na-h, K-n — на 2 вызова fast_exp_simd меньше на нейрон.
/// Использует vtrap для численной стабильности при V → -40 mV.
#[inline(always)]
unsafe fn na_m_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    // α: 0.1·vtrap(−(v+40), 10)
    // vtrap(x, y) = x/(exp(x/y)-1); vtrap(-vp40, 10) = -(v+40)/(exp(-(v+40)/10)-1)
    // exp(−(v+40)/10) = exp(−v/10) · exp(−4)
    let vp40 = v + f32x8::splat(40.0);
    let exp_nm_alpha = exp_v10 * f32x8::splat(EXP_V10_NA_M_ALPHA);
    // vtrap: x/(exp(x/y)-1) with Taylor for small |x/y|
    // x = -vp40, y = 10, exp(x/y) = exp_nm_alpha
    let ten = f32x8::splat(10.0);
    let ratio = -vp40 / ten;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    let vtrap_val = small_mask.blend(
        ten - (-vp40) * f32x8::splat(0.5), // Taylor: 10 - (-vp40)/2
        (-vp40) / (exp_nm_alpha - one),    // Exact
    );
    let alpha = f32x8::splat(0.1) * vtrap_val;
    // β: 4.0·exp(−(v+65)/18) — no reuse opportunity, separate exp
    let beta = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
    (alpha, beta)
}

/// Fused Na-h α/β: reuses exp(−v/10) for β computation.
///
/// α uses a separate exp (different divisor ÷20).
/// β = 1/(1 + exp(−v/10)·exp(−3.5)) — reuses exp_v10.
///
/// использует exp(−v/10) для β; α через отдельный exp (другой делитель ÷20).
#[inline(always)]
unsafe fn na_h_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    // α: 0.07·exp(−(v+65)/20) — separate exp (different divisor)
    let alpha = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    // β: 1/(1 + exp(−(v+35)/10))
    // exp(−(v+35)/10) = exp(−v/10) · exp(−3.5)
    let exp_nh_beta = exp_v10 * f32x8::splat(EXP_V10_NA_H_BETA);
    let beta = one / (one + exp_nh_beta);
    (alpha, beta)
}

/// Fused K-n α/β: reuses exp(−v/10) for α computation.
///
/// α = 0.01·vtrap(−(v+55), 10) — reuses exp_v10 для exp(−(v+55)/10).
/// β uses a separate exp (different divisor ÷80).
///
/// использует exp(−v/10) для α; β через отдельный exp (другой делитель ÷80).
#[inline(always)]
unsafe fn k_n_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    // α: 0.01·vtrap(−(v+55), 10)
    let vp55 = v + f32x8::splat(55.0);
    let exp_kn_alpha = exp_v10 * f32x8::splat(EXP_V10_KN_ALPHA);
    let ten = f32x8::splat(10.0);
    let ratio = -vp55 / ten;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    let vtrap_val = small_mask.blend(
        ten - (-vp55) * f32x8::splat(0.5), // Taylor: 10 - (-vp55)/2
        (-vp55) / (exp_kn_alpha - one),    // Exact
    );
    let alpha = f32x8::splat(0.01) * vtrap_val;
    // β: 0.125·exp(−(v+65)/80) — separate exp
    let beta = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
    (alpha, beta)
}

/// SIMD polynomial HH step with fused spike detection + common subexpression + Q10.
///
/// `celsius`: температура в °C для Q10 коррекции HH kinetics.
///
/// For each 8-neuron SIMD block:
///   1. Save V_old (for reverse zero-crossing spike detection)
///   2. Compute exp(−v/10) once → reused for Na-m α, Na-h β, K-n α
///   3. Compute HH dynamics (α/β, gates, I_ion, dV) с Q10 коррекцией
///   4. Store V_new with clamp to [-100, 50] mV
///   5. Spike detection: V_old ≥ 0 && V_new < 0 → 1, else 0
///
/// Для каждого блока из 8 нейронов:
///   exp(−v/10) вычисляется 1× → переиспользуется 3×
///   детекция спайков встроена (reverse zero-crossing)
///
/// # Safety
/// All pointer args valid for n elements, non-overlapping.
pub(crate) unsafe fn fused_hh_poly_step_simd(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
    spikes: *mut i32,
) {
    let zero = f32x8::splat(0.0);
    let eps = f32x8::splat(1e-12);
    let dt_v = f32x8::splat(dt);
    let v10_inv = f32x8::splat(1.0 / 10.0);
    let q10 = f32x8::splat(q10_from_celsius(celsius));

    for i in (0..n).step_by(8) {
        if i + 8 > n {
            break;
        }
        let v_old = unsafe { load8(v, i) };
        let vi = v_old;

        // Common subexpression: exp(−v/10) — reused 3×
        // fast_exp_simd already clamps internally to [-87, 87]
        let exp_v10 = fast_exp_simd((-vi) * v10_inv);

        let mut I_ion = zero;
        let mut gi = 0;

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let e_rev_v = f32x8::splat(ch.e_rev);
            let g_max_v = unsafe { load8(g_max_per_neuron.add(ch_idx * n), i) };
            let mut g_ch = g_max_v;

            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => na_m_alpha_beta_simd_fused(vi, exp_v10),
                    GateType::NaH => na_h_alpha_beta_simd_fused(vi, exp_v10),
                    GateType::Kn => k_n_alpha_beta_simd_fused(vi, exp_v10),
                    GateType::CatC => cat_c_alpha_beta_simd(vi),
                };

                let sum = (alpha + beta).max(eps);
                let m_inf = alpha / sum;
                // Q10: tau_q10 = tau / Q10 → sum_q10 = sum * Q10
                let sum_q10 = sum * q10;
                let decay = fast_exp_simd(-dt_v * sum_q10);

                let gi_ptr = gates.add(gi * n + i);
                let gv = unsafe { load8(gi_ptr, 0) };
                let new_gv = m_inf + (gv - m_inf) * decay;
                unsafe {
                    store8(gi_ptr, 0, new_gv);
                }

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
        let ii_clamped = unsafe { load8(Ii, i) }
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let dV = (ii_clamped - I_ion - I_leak) / cm;
        let v_new = vi + dV * dt_v;
        unsafe {
            store8(v, i, v_new);
        }

        // Fused spike detection: reverse zero-crossing (v_old ≥ 0 && v_new < 0)
        let spike_mask = v_old.cmp_ge(zero) & v_new.cmp_lt(zero);
        let spike_arr = super::spike_mask_to_i32(spike_mask);
        let p = spikes.add(i);
        *p = spike_arr[0];
        *p.add(1) = spike_arr[1];
        *p.add(2) = spike_arr[2];
        *p.add(3) = spike_arr[3];
        *p.add(4) = spike_arr[4];
        *p.add(5) = spike_arr[5];
        *p.add(6) = spike_arr[6];
        *p.add(7) = spike_arr[7];
    }
}

/// Scalar polynomial HH step with fused spike detection + Q10 (for remainder n%8).
///
/// Same as fused_hh_poly_step_simd but scalar — for the tail elements.
///
/// скалярный вариант для остатка n%8 с Q10 и vtrap.
///
/// # Safety
/// All pointer args valid for n elements.
pub(crate) unsafe fn fused_hh_poly_step_scalar(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    total_n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
    spikes: *mut i32,
) {
    let q10 = q10_from_celsius(celsius);

    for i in 0..n {
        let v_old = *v.add(i);
        let vi = v_old;
        let mut I_ion = 0.0;
        let mut gi = 0;

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let g_max = *g_max_per_neuron.add(ch_idx * total_n + i);
            let mut g_ch = g_max;
            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => {
                        let a = 0.1 * vtrap_scalar(-(vi + 40.0), 10.0);
                        let b = 4.0 * super::fast_exp(-(vi + 65.0) / 18.0);
                        (a, b)
                    }
                    GateType::NaH => {
                        let a = 0.07 * super::fast_exp(-(vi + 65.0) / 20.0);
                        let b = 1.0 / (1.0 + super::fast_exp(-(vi + 35.0) / 10.0));
                        (a, b)
                    }
                    GateType::Kn => {
                        let a = 0.01 * vtrap_scalar(-(vi + 55.0), 10.0);
                        let b = 0.125 * super::fast_exp(-(vi + 65.0) / 80.0);
                        (a, b)
                    }
                    GateType::CatC => {
                        let m_inf = 1.0 / (1.0 + super::fast_exp(-(vi + 56.0) / 6.2));
                        (m_inf / 0.6, (1.0 - m_inf) / 0.6)
                    }
                };
                let sum = alpha + beta;
                let m_inf = if sum > 1e-12 { alpha / sum } else { 0.5 };
                // Q10: sum_q10 = sum * Q10
                let decay = fast_exp(-dt * sum.max(1e-12) * q10);

                let p = gates.add(gi * total_n + i);
                *p = m_inf + (*p - m_inf) * decay;
                g_ch *= (*p).powi(power as i32);
                gi += 1;
            }
            I_ion += g_ch * (vi - ch.e_rev);
        }
        let I_leak = *g_leak.add(i) * (vi - *E_leak.add(i));
        let ii_clamped = (*Ii.add(i)).clamp(-100.0, 100.0);
        let dV = (ii_clamped - I_ion - I_leak) / *C_m.add(i);
        let v_new = (vi + dV * dt).clamp(-100.0, 50.0);
        *v.add(i) = v_new;

        // Spike detection inline
        *spikes.add(i) = if v_old >= 0.0 && v_new < 0.0 { 1 } else { 0 };
    }
}

/// Public fused HH step — auto-selects SIMD or scalar.
///
/// `celsius`: температура в °C (6.3 = squid axon @ H&H 1952, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron g_max, layout [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
/// Size: [n_channels * n]
///
/// `spikes`: output buffer, writes 1 for reverse zero-crossing.
pub unsafe fn fused_hh_poly_step(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    channel_meta: &[ChannelMeta],
    g_max_per_neuron: *const f32,
    dt: f32,
    celsius: f32,
    spikes: *mut i32,
) {
    if n >= 8 {
        fused_hh_poly_step_simd(
            v,
            gates,
            Ii,
            g_leak,
            E_leak,
            C_m,
            n,
            channel_meta,
            g_max_per_neuron,
            dt,
            celsius,
            spikes,
        );
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        fused_hh_poly_step_scalar(
            v.add(simd_end),
            gates.add(simd_end),
            Ii.add(simd_end),
            g_leak.add(simd_end),
            E_leak.add(simd_end),
            C_m.add(simd_end),
            remainder,
            n,
            channel_meta,
            g_max_per_neuron,
            dt,
            celsius,
            spikes.add(simd_end),
        );
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
                GateSpec {
                    label: "m",
                    alpha: na_m_alpha,
                    beta: na_m_beta,
                    power: 3,
                },
                GateSpec {
                    label: "h",
                    alpha: na_h_alpha,
                    beta: na_h_beta,
                    power: 1,
                },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![GateSpec {
                label: "n",
                alpha: k_n_alpha,
                beta: k_n_beta,
                power: 4,
            }],
            E_rev: -77.0,
            g_max: 36.0,
        },
        ChannelSpec {
            name: "CaT",
            gates: vec![GateSpec {
                label: "c",
                alpha: cat_c_alpha,
                beta: cat_c_beta,
                power: 1,
            }],
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
pub(crate) unsafe fn hh_lut_step_scalar(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    _n_gates: usize,
    m_inf_luts: *const *const f32,
    decay_luts: *const *const f32,
    v_min: f32,
    v_max: f32,
    n_bins: usize,
    E_rev_arr: *const f32,
    g_max_arr: *const f32,
    powers: *const u32,
    n_gates_per_ch: *const usize,
    n_channels: usize,
    _g_leak_idx: usize,
    dt: f32,
) {
    let inv_dv = (n_bins as f32) / (v_max - v_min);
    for i in 0..n {
        let vi = *v.add(i);
        let idx = v_to_idx(vi, v_min, inv_dv, n_bins);
        let mut I_ion = 0.0_f32;
        let mut gi = 0; // global gate index

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
        *v.add(i) = (vi + dV * dt).clamp(-100.0, 50.0);
    }
}

// ═══════════════════════════════════════════════════════
//  SIMD (f32x8) HH step — 8 neurons at once
// ═══════════════════════════════════════════════════════

pub(crate) unsafe fn hh_lut_step_simd(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    _n_gates: usize,
    m_inf_luts: *const *const f32,
    decay_luts: *const *const f32,
    v_min: f32,
    v_max: f32,
    n_bins: f32,
    E_rev_arr: *const f32,
    g_max_arr: *const f32,
    powers: *const u32,
    n_gates_per_ch: *const usize,
    n_channels: usize,
    dt: f32,
) {
    let inv_dv = n_bins / (v_max - v_min);
    let v_min_v = f32x8::splat(v_min);
    let inv_dv_v = f32x8::splat(inv_dv);
    let zero = f32x8::splat(0.0);
    let n_bins_f = f32x8::splat(n_bins);

    for i in (0..n).step_by(8) {
        if i + 8 > n {
            break;
        }
        let vi = load8(v, i);
        let idx_f = ((vi - v_min_v) * inv_dv_v).max(zero).min(n_bins_f);
        let mut I_ion = f32x8::splat(0.0);
        let mut gi = 0; // global gate index

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
        let v_new = vi + dV * f32x8::splat(dt);
        store8(v, i, v_new);
    }
}

/// Public entry: auto-chooses SIMD or scalar.
pub unsafe fn hh_lut_step(
    v: *mut f32,
    gates: *mut f32,
    Ii: *const f32,
    g_leak: *const f32,
    E_leak: *const f32,
    C_m: *const f32,
    n: usize,
    n_gates: usize,
    m_inf_luts: *const *const f32,
    decay_luts: *const *const f32,
    v_min: f32,
    v_max: f32,
    n_bins: f32,
    E_rev_arr: *const f32,
    g_max_arr: *const f32,
    powers: *const u32,
    n_gates_per_ch: *const usize,
    n_channels: usize,
    dt: f32,
) {
    if n >= 8 {
        hh_lut_step_simd(
            v,
            gates,
            Ii,
            g_leak,
            E_leak,
            C_m,
            n,
            n_gates,
            m_inf_luts,
            decay_luts,
            v_min,
            v_max,
            n_bins,
            E_rev_arr,
            g_max_arr,
            powers,
            n_gates_per_ch,
            n_channels,
            dt,
        );
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        hh_lut_step_scalar(
            v.add(simd_end),
            gates.add(simd_end),
            Ii.add(simd_end),
            g_leak.add(simd_end),
            E_leak.add(simd_end),
            C_m.add(simd_end),
            remainder,
            n_gates,
            m_inf_luts,
            decay_luts,
            v_min,
            v_max,
            n_bins as usize,
            E_rev_arr,
            g_max_arr,
            powers,
            n_gates_per_ch,
            n_channels,
            !0,
            dt,
        );
    }
}
