//! # HH Channel Definitions + Polynomial Engine — standard squid-axon & pacemaking (CaV-T)
//!
//! Port of the **entire** legacy `crates/kernels/src/neuron/hh.rs` (волна 8,
//! транш 1e). Two layers:
//!
//! 1. **Channel specs** — what a conductance-based model is made of: the channel
//!    list, per-gate rate functions α(V)/β(V) and steady-state helpers.
//! 2. **Polynomial engine** — `hh_poly_step` / `fused_hh_poly_step`: batch
//!    `f32x8` SIMD + scalar-tail integrators with Q10 temperature correction,
//!    bit-identical to the deleted legacy kernels (SIMD explicit-Euler,
//!    scalar implicit-Euler divergence preserved; fused-scalar clamps v to
//!    [-100, 50], fused-SIMD clamps only Ii to ±100 — as in legacy).
//!
//! The zoo `HHModel` (`run_neuron`) remains the canonical integrator for the
//! standard/pacemaking layouts; this engine is the generic-arity fallback and
//! the reference implementation for parity tests.
//!
//! # Channel sets
//!
//! - [`standard_channels`] — NaV (m³h) + KV (n⁴): the classic Hodgkin-Huxley
//!   squid axon. Requires external drive to spike.
//! - [`pacemaking_channels`] — standard + **CaV-T (T-type calcium)**: a single
//!   low-threshold activation gate `c`. The T-type current depolarizes the
//!   membrane past NaV threshold on its own → the neuron fires **spontaneously**
//!   without any input current (the basis of rhythmic VNC/CPG activity).

#![allow(non_snake_case)]

use crate::math::fast_exp::fast_exp;

// ═══════════════════════════════════════════════════════════
//  Channel specification types
// ═══════════════════════════════════════════════════════════

/// Rate-function closure: α(V) or β(V) in [1/ms].
pub type RateFn = fn(f32) -> f32;

/// One gating variable (e.g. Na-m, Na-h, K-n, CaT-c).
#[derive(Debug, Clone, Copy)]
pub struct GateSpec {
    pub label: &'static str,
    pub alpha: RateFn,
    pub beta: RateFn,
    /// Exponent in the conductance term: `gating = gate^power`.
    pub power: u32,
}

/// One ion channel: one or more gates + reversal potential + max conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    /// Reversal potential in mV.
    pub E_rev: f32,
    /// Max conductance in mS/cm².
    pub g_max: f32,
}

// ═══════════════════════════════════════════════════════════
//  Squid-axon rate functions (Hodgkin & Huxley 1952, 6.3 °C)
// ═══════════════════════════════════════════════════════════

/// vtrap: `x / (exp(x/y) − 1)` with a Taylor `y − x/2` when `|x/y| < 1e-6`.
/// Prevents 0/0 at x → 0, exactly like NEURON's `FUNCTION vtrap`.
///
/// This is the HH rate-function helper (x = voltage shift, y = slope divisor).
/// Do not confuse it with [`crate::math::vtrap::vtrap`] — the sigmoid-shaped
/// per-lane primitive used by gates — the two share a name but not semantics.
#[inline(always)]
pub fn vtrap(x: f32, y: f32) -> f32 {
    let ratio = x / y;
    if ratio.abs() < 1e-6 {
        y - x * 0.5
    } else {
        x / (fast_exp(ratio) - 1.0)
    }
}

/// Na activation (m-gate) α: `0.1·vtrap(−(v+40), 10)`.
pub fn na_m_alpha(v: f32) -> f32 {
    0.1 * vtrap(-(v + 40.0), 10.0)
}
/// Na activation (m-gate) β: `4·exp(−(v+65)/18)`.
pub fn na_m_beta(v: f32) -> f32 {
    4.0 * fast_exp(-(v + 65.0) / 18.0)
}

/// Na inactivation (h-gate) α: `0.07·exp(−(v+65)/20)`.
pub fn na_h_alpha(v: f32) -> f32 {
    0.07 * fast_exp(-(v + 65.0) / 20.0)
}
/// Na inactivation (h-gate) β: `1/(1 + exp(−(v+35)/10))`.
pub fn na_h_beta(v: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-(v + 35.0) / 10.0))
}

/// K activation (n-gate) α: `0.01·vtrap(−(v+55), 10)`.
pub fn k_n_alpha(v: f32) -> f32 {
    0.01 * vtrap(-(v + 55.0), 10.0)
}
/// K activation (n-gate) β: `0.125·exp(−(v+65)/80)`.
pub fn k_n_beta(v: f32) -> f32 {
    0.125 * fast_exp(-(v + 65.0) / 80.0)
}

/// CaV-T (T-type calcium) activation — low-threshold, pacemaking.
/// Hill & Tononi 2005 / Wang & Rinzel 1993. Single gate `c`, τ_c = 0.6 ms.
/// α = m_inf/τ where m_inf = 1/(1 + exp(−(v+56)/6.2)).
pub fn cat_c_alpha(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    m_inf / 0.6
}
/// CaV-T β: `(1 − m_inf)/τ`.
pub fn cat_c_beta(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    (1.0 - m_inf) / 0.6
}

// ═══════════════════════════════════════════════════════════
//  Channel sets
// ═══════════════════════════════════════════════════════════

/// Classic HH squid axon: NaV (m³h) + KV (n⁴).
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

/// HH + CaV-T (T-type calcium) for **pacemaking**: 4 gates (NaV-m, NaV-h, KV-n,
/// CaT-c). The T-type current makes neurons fire spontaneously without input.
pub fn pacemaking_channels() -> Vec<ChannelSpec> {
    let mut ch = standard_channels();
    ch.push(ChannelSpec {
        name: "CaT",
        gates: vec![GateSpec {
            label: "c",
            alpha: cat_c_alpha,
            beta: cat_c_beta,
            power: 1,
        }],
        E_rev: 120.0,
        g_max: 0.5,
    });
    ch
}

// ═══════════════════════════════════════════════════════════
//  Steady-state helpers
// ═══════════════════════════════════════════════════════════

/// Steady-state gate values `m_inf = α(V)/(α(V)+β(V))` at a given potential,
/// in gate order (channel-major).
pub fn steady_state_gates(v: f32, channels: &[ChannelSpec]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channels {
        for gate in &ch.gates {
            let a = (gate.alpha)(v);
            let b = (gate.beta)(v);
            let sum = a + b;
            gates.push(if sum > 1e-12 { a / sum } else { 0.5 });
        }
    }
    gates
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
//  Polynomial HH engine — batch SIMD + scalar tail (no LUT)
// ═══════════════════════════════════════════════════════
//  Бит-в-бит порт движка легаси `kernels::neuron::hh` (волна 8, транш 1e).
//  Сохранены документированные расхождения легаси SIMD/скаляр:
//    1. SIMD-путь — explicit Euler; скалярный хвост — implicit Euler;
//    2. fused-scalar клампит v_new к [-100, 50], fused-SIMD — только Ii к ±100;
//    3. q10 с guard-зоной |c−6.3| ≤ 0.1 → 1.0 (легаси-семантика).
//  Синаптические/токовые пути используют fast_exp / fast_exp_poly_simd — те же
//  примитивы, что реэкспортировал легаси-модуль (был `fast_exp_simd` = poly).

use crate::core::util::{load8, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd as fast_exp_simd;
use wide::f32x8;
use wide::{CmpGe, CmpLt};

/// Вычислить Q10 = 3^((celsius - 6.3) / 10) для температурной коррекции HH.
/// При 6.3°C (squid axon) Q10 = 1.0. При 37°C Q10 ≈ 29×.
///
/// Guard-зона `|c−6.3| ≤ 0.1` → 1.0 — легаси-семантика (в отличие от
/// [`crate::q10_from_celsius_s`](crate::q10_from_celsius_s), у которого
/// guard-зоны нет; для 6.3 °C оба дают 1.0 бит-в-бит).
#[inline]
pub fn q10_from_celsius(celsius: f32) -> f32 {
    if (celsius - 6.3).abs() > 0.1 {
        3.0f32.powf((celsius - 6.3) / 10.0)
    } else {
        1.0
    }
}

/// SIMD vtrap: `x / (exp(x/y) - 1)` с Taylor `y - x/2` при |x/y| < 1e-6.
/// В точности как NEURON hh.mod `FUNCTION vtrap`. Предотвращает 0/0 при x → 0.
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

/// SIMD α/β computation helpers (inlined) — порт из легаси (используют
/// `fast_exp_poly_simd`, ровно как легаси `super::fast_exp_simd`).
#[inline(always)]
unsafe fn na_m_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp40 = v + f32x8::splat(40.0);
    let alpha = f32x8::splat(0.1) * vtrap_simd(-vp40, 10.0);
    let beta = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
    (alpha, beta)
}

#[inline(always)]
unsafe fn na_h_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let alpha = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    let exp_arg = -(v + f32x8::splat(35.0)) / f32x8::splat(10.0);
    let beta = one / (one + fast_exp_simd(exp_arg));
    (alpha, beta)
}

#[inline(always)]
unsafe fn k_n_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let vp55 = v + f32x8::splat(55.0);
    let alpha = f32x8::splat(0.01) * vtrap_simd(-vp55, 10.0);
    let beta = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
    (alpha, beta)
}

#[inline(always)]
unsafe fn cat_c_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let m_inf = one / (one + fast_exp_simd(-(v + f32x8::splat(56.0)) / f32x8::splat(6.2)));
    let tau_inv = f32x8::splat(1.0 / 0.6);
    (m_inf * tau_inv, (one - m_inf) * tau_inv)
}

/// SIMD polynomial HH step — direct rate computation (no LUT).
/// Explicit Euler. Only full 8-lane blocks; scalar tail handles the rest.
///
/// `g_max_per_neuron` layout: [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...] (stride = n).
/// # Safety
/// Все указатели валидны на n элементов, не пересекаются.
#[inline]
unsafe fn hh_poly_step_simd(
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
        let vi = load8(v, i);
        let mut I_ion = zero;
        let mut gi = 0; // global gate index

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let e_rev_v = f32x8::splat(ch.e_rev);
            let g_max_v = load8(g_max_per_neuron.add(ch_idx * n), i);
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
                let gv = load8(gi_ptr, 0);
                let new_gv = m_inf + (gv - m_inf) * decay;
                store8(gi_ptr, 0, new_gv);

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

        let I_leak = load8(g_leak, i) * (vi - load8(E_leak, i));
        let cm = load8(C_m, i);
        let dV = (load8(Ii, i) - I_ion - I_leak) / cm;
        let v_new = vi + dV * dt_v;
        store8(v, i, v_new);
    }
}

/// Compute HH rate functions (alpha, beta) for a gate type at voltage v.
/// Скалярный путь — `fast_exp` (полином), как в легаси `rate_fn`.
#[inline(always)]
fn rate_fn(v: f32, gate_type: GateType) -> (f32, f32) {
    match gate_type {
        GateType::NaM => {
            let a = 0.1 * vtrap(-(v + 40.0), 10.0);
            let b = 4.0 * fast_exp(-(v + 65.0) / 18.0);
            (a, b)
        }
        GateType::NaH => {
            let a = 0.07 * fast_exp(-(v + 65.0) / 20.0);
            let b = 1.0 / (1.0 + fast_exp(-(v + 35.0) / 10.0));
            (a, b)
        }
        GateType::Kn => {
            let a = 0.01 * vtrap(-(v + 55.0), 10.0);
            let b = 0.125 * fast_exp(-(v + 65.0) / 80.0);
            (a, b)
        }
        GateType::CatC => {
            let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
            (m_inf / 0.6, (1.0 - m_inf) / 0.6)
        }
    }
}

/// Scalar polynomial HH step — для остатка (n % 8 != 0).
/// Implicit Euler (как в легаси-хвосте): `v_new = (v·C/dt + I_ext + ΣgE) / (C/dt + Σg)`.
///
/// `n` — размер чанка (остаток), `total_n` — полный размер (stride для gates).
/// # Safety
/// Все указатели валидны на n элементов.
#[inline]
unsafe fn hh_poly_step_scalar(
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

/// Public polynomial HH step — auto-selects SIMD (full 8-lane blocks) or scalar tail.
///
/// `celsius`: температура в °C (6.3 = squid axon @ H&H 1952, 24 = Drosophila).
/// `g_max_per_neuron`: per-neuron g_max, layout [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
/// Size: [n_channels * n]
///
/// # Safety
/// Все указатели валидны на n элементов, не пересекаются.
#[allow(clippy::too_many_arguments)]
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
//  Общий subexpression: exp(−v/10) переиспользуется 3× (Na-m α, Na-h β, K-n α)
//  → 2 вызова fast_exp_simd меньше на нейрон. Reverse zero-crossing спайки.

/// Предвычисленные константы общего subexpression: exp_const = exp(−shift/10)
const EXP_V10_NA_M_ALPHA: f32 = 0.018_315_64_f32;
const EXP_V10_NA_H_BETA: f32 = 0.030197383_f32;
const EXP_V10_KN_ALPHA: f32 = 0.0040867714_f32;

#[inline(always)]
unsafe fn na_m_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let vp40 = v + f32x8::splat(40.0);
    let exp_nm_alpha = exp_v10 * f32x8::splat(EXP_V10_NA_M_ALPHA);
    let ten = f32x8::splat(10.0);
    let ratio = -vp40 / ten;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    let vtrap_val = small_mask.blend(
        ten - (-vp40) * f32x8::splat(0.5),
        (-vp40) / (exp_nm_alpha - one),
    );
    let alpha = f32x8::splat(0.1) * vtrap_val;
    let beta = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
    (alpha, beta)
}

#[inline(always)]
unsafe fn na_h_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let alpha = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    let exp_nh_beta = exp_v10 * f32x8::splat(EXP_V10_NA_H_BETA);
    let beta = one / (one + exp_nh_beta);
    (alpha, beta)
}

#[inline(always)]
unsafe fn k_n_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let vp55 = v + f32x8::splat(55.0);
    let exp_kn_alpha = exp_v10 * f32x8::splat(EXP_V10_KN_ALPHA);
    let ten = f32x8::splat(10.0);
    let ratio = -vp55 / ten;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    let vtrap_val = small_mask.blend(
        ten - (-vp55) * f32x8::splat(0.5),
        (-vp55) / (exp_kn_alpha - one),
    );
    let alpha = f32x8::splat(0.01) * vtrap_val;
    let beta = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
    (alpha, beta)
}

/// SIMD polynomial HH step with fused spike detection + common subexpression + Q10.
/// Клампит Ii к ±100; v_new НЕ клампит (легаси-SIMD поведение).
/// Reverse zero-crossing: v_old ≥ 0 && v_new < 0 → 1.
#[inline]
unsafe fn fused_hh_poly_step_simd(
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
        let v_old = load8(v, i);
        let vi = v_old;

        // Common subexpression: exp(−v/10) — reused 3×
        let exp_v10 = fast_exp_simd((-vi) * v10_inv);

        let mut I_ion = zero;
        let mut gi = 0;

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let e_rev_v = f32x8::splat(ch.e_rev);
            let g_max_v = load8(g_max_per_neuron.add(ch_idx * n), i);
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
                let sum_q10 = sum * q10;
                let decay = fast_exp_simd(-dt_v * sum_q10);

                let gi_ptr = gates.add(gi * n + i);
                let gv = load8(gi_ptr, 0);
                let new_gv = m_inf + (gv - m_inf) * decay;
                store8(gi_ptr, 0, new_gv);

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

        let I_leak = load8(g_leak, i) * (vi - load8(E_leak, i));
        let cm = load8(C_m, i);
        let ii_clamped = load8(Ii, i)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let dV = (ii_clamped - I_ion - I_leak) / cm;
        let v_new = vi + dV * dt_v;
        store8(v, i, v_new);

        // Fused spike detection: reverse zero-crossing (v_old ≥ 0 && v_new < 0)
        let spike_mask = v_old.cmp_ge(zero) & v_new.cmp_lt(zero);
        let spike_arr = mask_to_spikes(spike_mask);
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

/// Scalar polynomial HH step with fused spike detection + Q10 (для остатка n%8).
/// Клампит Ii к ±100 И v_new к [-100, 50] (легаси-скаляр поведение).
#[inline]
unsafe fn fused_hh_poly_step_scalar(
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
                        let a = 0.1 * vtrap(-(vi + 40.0), 10.0);
                        let b = 4.0 * fast_exp(-(vi + 65.0) / 18.0);
                        (a, b)
                    }
                    GateType::NaH => {
                        let a = 0.07 * fast_exp(-(vi + 65.0) / 20.0);
                        let b = 1.0 / (1.0 + fast_exp(-(vi + 35.0) / 10.0));
                        (a, b)
                    }
                    GateType::Kn => {
                        let a = 0.01 * vtrap(-(vi + 55.0), 10.0);
                        let b = 0.125 * fast_exp(-(vi + 65.0) / 80.0);
                        (a, b)
                    }
                    GateType::CatC => {
                        let m_inf = 1.0 / (1.0 + fast_exp(-(vi + 56.0) / 6.2));
                        (m_inf / 0.6, (1.0 - m_inf) / 0.6)
                    }
                };
                let sum = alpha + beta;
                let m_inf = if sum > 1e-12 { alpha / sum } else { 0.5 };
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
/// `spikes`: output buffer, writes 1 for reverse zero-crossing.
/// # Safety
/// Все указатели валидны на n элементов, не пересекаются.
#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar exponential-Euler HH step over arbitrary channels.
    /// Returns `true` if a spike (reverse zero-crossing through 0 mV) occurred.
    #[allow(clippy::too_many_arguments)]
    fn hh_step_scalar(
        v: &mut f32,
        gates: &mut [f32],
        i_ext: f32,
        channels: &[ChannelSpec],
        g_leak: f32,
        e_leak: f32,
        c_m: f32,
        dt: f32,
    ) -> bool {
        let v_old = *v;
        let mut i_ion = 0.0f32;
        let mut gi = 0usize;
        for ch in channels {
            let mut g_ch = ch.g_max;
            for gate in &ch.gates {
                let a = (gate.alpha)(*v);
                let b = (gate.beta)(*v);
                let sum = a + b;
                let m_inf = if sum > 1e-12 { a / sum } else { 0.5 };
                let decay = fast_exp(-dt * sum.max(1e-12));
                let gv = &mut gates[gi];
                *gv = m_inf + (*gv - m_inf) * decay;
                g_ch *= gv.powi(gate.power as i32);
                gi += 1;
            }
            i_ion += g_ch * (*v - ch.E_rev);
        }
        let i_leak = g_leak * (*v - e_leak);
        let dv = (i_ext - i_ion - i_leak) / c_m;
        let v_new = *v + dv * dt;
        *v = v_new;
        v_old >= 0.0 && v_new < 0.0
    }

    /// Standard squid axon: must NOT fire spontaneously with zero input.
    #[test]
    fn standard_channels_quiescent_without_input() {
        let channels = standard_channels();
        let dt = 0.05;
        let mut v = -65.0f32;
        let mut gates = steady_state_gates(v, &channels);
        let mut spikes = 0u32;
        for _ in 0..4000 {
            // 200 ms
            if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
                spikes += 1;
            }
        }
        assert_eq!(spikes, 0, "standard HH should not self-spike, got {spikes}");
        // Resting potential of the squid axon is ≈ −65 mV (NOT the leak
        // reversal −54.387); the membrane must sit near rest, not drift.
        assert!(
            v > -75.0 && v < -45.0,
            "v should stay near resting potential, got {v}"
        );
    }

    /// Pacemaking (CaV-T): must fire spontaneously with zero input.
    #[test]
    fn pacemaking_fires_spontaneously() {
        let channels = pacemaking_channels();
        let dt = 0.05;
        let mut v = -65.0f32;
        let mut gates = steady_state_gates(v, &channels);
        let mut spikes = 0u32;
        for _ in 0..4000 {
            // 200 ms
            if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
                spikes += 1;
            }
            assert!(v.is_finite() && v > -120.0 && v < 80.0, "v runaway: {v}");
        }
        assert!(
            spikes >= 3,
            "pacemaking HH should fire ≥3 spikes in 200 ms, got {spikes}"
        );
    }

    /// Gate ordering: 4 gates for pacemaking (m, h, n, c), steady states in [0,1].
    #[test]
    fn gate_order_and_steady_state() {
        let ch = pacemaking_channels();
        assert_eq!(ch.len(), 3);
        let n_gates: usize = ch.iter().map(|c| c.gates.len()).sum();
        assert_eq!(n_gates, 4);
        let g = steady_state_gates(-65.0, &ch);
        assert_eq!(g.len(), 4);
        for (i, &x) in g.iter().enumerate() {
            assert!((0.0..=1.0).contains(&x), "gate {i} = {x} out of [0,1]");
        }
        // CaV-T is the last gate (index 3)
        assert_eq!(ch[2].name, "CaT");
        assert_eq!(ch[2].gates[0].label, "c");
        assert_eq!(ch[2].E_rev, 120.0);
    }

    /// Rate-function sanity: steady-state m_inf saturates with depolarization.
    #[test]
    fn na_m_inf_saturates() {
        let channels = standard_channels();
        let g_hyper = steady_state_gates(-80.0, &channels);
        let g_depol = steady_state_gates(0.0, &channels);
        assert!(
            g_depol[0] > g_hyper[0],
            "m_inf should rise with depolarization"
        );
    }
}
