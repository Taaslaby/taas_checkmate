#![allow(non_snake_case)] // Ii/E_rev/m_inf — физическая нотация

// ═══════════════════════════════════════════════════════
//  Polynomial HH engine — batch SIMD + scalar tail (no LUT)
// ═══════════════════════════════════════════════════════
//  Порт движка легаси `kernels::neuron::hh` (волна 8, транш 1e) с исправленной
//  физикой: мембранный интегратор УНИФИЦИРОВАН — implicit Euler (derivimplicit)
//  в SIMD-блоке И скалярном хвосте (тот же метод, что `HHModel`/NEURON), а
//  кламп fused v_new одинаков в обоих путях. Прежние расхождения (explicit-Euler
//  SIMD против implicit-Euler хвоста; разный кламп fused) устранены — один и тот
//  же нейрон больше не меняет динамику от попадания в SIMD-блок или хвост.
//  Сохранена только q10-семантика с guard-зоной |c−6.3| ≤ 0.1 → 1.0 (легаси).
//  Синаптические/токовые пути используют fast_exp / fast_exp_poly_simd — те же
//  примитивы, что реэкспортировал легаси-модуль (был `fast_exp_simd` = poly).

use super::spec::{ChannelSpec, GateKinetics};
use crate::core::util::{load8, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp;
use crate::math::fast_exp::fast_exp_poly_simd as fast_exp_simd;
use wide::f32x8;
use wide::{CmpGe, CmpLt};

/// Вычислить Q10 = 3^((celsius - 6.3) / 10) для температурной коррекции HH.
/// При 6.3°C (squid axon) Q10 = 1.0. При 37°C Q10 ≈ 29×.
///
/// Guard-зона `|c−6.3| ≤ 0.1` → 1.0 — легаси-семантика: формула та же, что в
/// [`crate::q10_from_celsius_s`], у которого guard-зоны нет. Для 6.3 °C обе
/// дают 1.0 бит-в-бит.
#[inline]
pub fn q10_from_celsius(celsius: f32) -> f32 {
    if (celsius - 6.3).abs() > 0.1 {
        crate::q10_from_celsius_s(celsius)
    } else {
        1.0
    }
}

/// Channel gate index metadata (precomputed at init from ChannelSpec).
/// Does NOT store g_max — that's per-neuron now.
pub struct ChannelMeta {
    pub n_gates: usize, // gates per channel
    pub e_rev: f32,
    pub powers: Vec<u32>,            // exponent per gate
    pub gate_type: Vec<GateType>,    // fused fast-path key (4 canonical families)
    pub kinetics: Vec<GateKinetics>, // general data-driven engine
}

/// Семейство канонических гейтов — ключ **fused** fast-path (`fused_hh_poly_step`),
/// который факторизует общий `exp(−v/10)` для Na-m/Na-h/K-n. Обобщённый
/// (non-fused) движок `hh_poly_step` использует data-driven [`GateKinetics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    NaM,
    NaH,
    Kn,
    CatC,
}

/// Build ChannelMeta from ChannelSpec (for poly HH).
///
/// `gate_type` — для fused fast-path (выводится из `label`); `kinetics` — для
/// обобщённого движка (копируется из `GateSpec`). Для standard/pacemaking
/// наборов оба представления бит-в-бит согласованы.
pub fn build_channel_meta(channels: &[ChannelSpec]) -> Vec<ChannelMeta> {
    channels
        .iter()
        .map(|ch| {
            let n_gates = ch.gates.len();
            let mut powers = Vec::with_capacity(n_gates);
            let mut gate_type = Vec::with_capacity(n_gates);
            let mut kinetics = Vec::with_capacity(n_gates);
            for g in &ch.gates {
                powers.push(g.power);
                gate_type.push(match g.label {
                    "m" => GateType::NaM,
                    "h" => GateType::NaH,
                    "n" => GateType::Kn,
                    "c" => GateType::CatC,
                    _ => GateType::NaM, // fallback
                });
                kinetics.push(g.kinetics.clone());
            }
            ChannelMeta {
                n_gates,
                e_rev: ch.E_rev,
                powers,
                gate_type,
                kinetics,
            }
        })
        .collect()
}

/// CaV-T SIMD α/β — единственный non-fused хелпер, оставшийся после перехода
/// на data-driven `GateKinetics`: его всё ещё использует fused fast-path
/// (`fused_hh_poly_step_simd`, вариант `GateType::CatC`).
#[inline(always)]
unsafe fn cat_c_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
    let one = f32x8::splat(1.0);
    let m_inf = one / (one + fast_exp_simd(-(v + f32x8::splat(56.0)) / f32x8::splat(6.2)));
    let tau_inv = f32x8::splat(1.0 / 0.6);
    (m_inf * tau_inv, (one - m_inf) * tau_inv)
}

/// SIMD polynomial HH step — direct rate computation (no LUT).
/// Implicit Euler (derivimplicit) — тот же мембранный интегратор, что скалярный
/// хвост и `HHModel`: `v_new = (v·C/dt + I + ΣgE)/(C/dt + Σg)`. Гейты — на
/// старом V (semi-implicit). Только полные 8-lane блоки; скалярный хвост — остаток.
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
        let mut sum_g = zero;
        let mut sum_gE = zero;
        let mut gi = 0; // global gate index

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let e_rev_v = f32x8::splat(ch.e_rev);
            let g_max_v = load8(g_max_per_neuron.add(ch_idx * n), i);
            let mut g_ch = g_max_v;

            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = ch.kinetics[gate_idx].alpha_beta_simd(vi);

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
            sum_g += g_ch;
            sum_gE += g_ch * e_rev_v;
        }

        let g_leak_v = load8(g_leak, i);
        let e_leak_v = load8(E_leak, i);
        sum_g += g_leak_v;
        sum_gE += g_leak_v * e_leak_v;

        let cm = load8(C_m, i);
        let inv_dt = cm / dt_v;
        let v_new = (vi * inv_dt + load8(Ii, i) + sum_gE) / (inv_dt + sum_g);
        store8(v, i, v_new);
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
                let (alpha, beta) = ch.kinetics[gate_idx].alpha_beta(vi);
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

/// Safe slice-based front-end to [`hh_poly_step`] — same kernel, no pointers.
///
/// `n = v.len()`; `gates`/`g_max` are channel-major with `n` elements per
/// channel (`gates.len() ≥ n_gates_total × n`, `g_max.len() ≥ n_channels × n`,
/// where the counts come from `channel_meta`). All other buffers must be at
/// least `n` long.
///
/// # Panics
/// If any buffer is shorter than its contract above.
pub fn hh_poly_step_slices(
    v: &mut [f32],
    gates: &mut [f32],
    i_in: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    channel_meta: &[ChannelMeta],
    g_max: &[f32],
    dt: f32,
    celsius: f32,
) {
    let n = v.len();
    let n_gates_total: usize = channel_meta.iter().map(|c| c.n_gates).sum();
    let n_channels = channel_meta.len();
    assert!(
        gates.len() >= n_gates_total * n && g_max.len() >= n_channels * n,
        "hh_poly_step_slices: gates/g_max too short for n={n}"
    );
    assert!(
        i_in.len() >= n && g_leak.len() >= n && e_leak.len() >= n && c_m.len() >= n,
        "hh_poly_step_slices: per-neuron buffers shorter than n={n}"
    );
    // SAFETY: lengths checked above — every pointer is valid for `n` elements
    // and the buffers don't overlap (owned `AlignedVec`s in the population).
    unsafe {
        hh_poly_step(
            v.as_mut_ptr(),
            gates.as_mut_ptr(),
            i_in.as_ptr(),
            g_leak.as_ptr(),
            e_leak.as_ptr(),
            c_m.as_ptr(),
            n,
            channel_meta,
            g_max.as_ptr(),
            dt,
            celsius,
        )
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

/// Скалярные зеркала factored fused-гейтов — та же формула, что в SIMD-версиях
/// выше (общий `exp_v10 = exp(−v/10)` и те же константы `EXP_V10_*`). Раньше
/// скалярный хвост `fused_hh_poly_step` считал гейты напрямую
/// (`exp(−(v+40)/10)`), а SIMD — факторизованно (`exp(−v/10)·exp(−4)`), что при
/// экстремальном токе давало расхождение SIMD/хвост ~2e-3 мВ. Теперь оба пути
/// используют одну формулу — хвост бит-совместим с SIMD-блоком.
#[inline(always)]
fn na_m_alpha_beta_scalar_fused(v: f32, exp_v10: f32) -> (f32, f32) {
    let vp40 = v + 40.0;
    let exp_nm_alpha = exp_v10 * EXP_V10_NA_M_ALPHA;
    let ratio = -vp40 / 10.0;
    let vtrap_val = if ratio.abs() < 1e-6 {
        10.0 - (-vp40) * 0.5
    } else {
        (-vp40) / (exp_nm_alpha - 1.0)
    };
    let alpha = 0.1 * vtrap_val;
    let beta = 4.0 * fast_exp(-(v + 65.0) / 18.0);
    (alpha, beta)
}

#[inline(always)]
fn na_h_alpha_beta_scalar_fused(v: f32, exp_v10: f32) -> (f32, f32) {
    let alpha = 0.07 * fast_exp(-(v + 65.0) / 20.0);
    let exp_nh_beta = exp_v10 * EXP_V10_NA_H_BETA;
    let beta = 1.0 / (1.0 + exp_nh_beta);
    (alpha, beta)
}

#[inline(always)]
fn k_n_alpha_beta_scalar_fused(v: f32, exp_v10: f32) -> (f32, f32) {
    let vp55 = v + 55.0;
    let exp_kn_alpha = exp_v10 * EXP_V10_KN_ALPHA;
    let ratio = -vp55 / 10.0;
    let vtrap_val = if ratio.abs() < 1e-6 {
        10.0 - (-vp55) * 0.5
    } else {
        (-vp55) / (exp_kn_alpha - 1.0)
    };
    let alpha = 0.01 * vtrap_val;
    let beta = 0.125 * fast_exp(-(v + 65.0) / 80.0);
    (alpha, beta)
}

/// SIMD polynomial HH step with fused spike detection + common subexpression + Q10.
/// Implicit Euler (как скалярный хвост): клампит Ii к ±100 И v_new к [-100, 50].
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

        let mut sum_g = zero;
        let mut sum_gE = zero;
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
            sum_g += g_ch;
            sum_gE += g_ch * e_rev_v;
        }

        let g_leak_v = load8(g_leak, i);
        let e_leak_v = load8(E_leak, i);
        sum_g += g_leak_v;
        sum_gE += g_leak_v * e_leak_v;

        let cm = load8(C_m, i);
        let inv_dt = cm / dt_v;
        let ii_clamped = load8(Ii, i)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let v_new = ((vi * inv_dt + ii_clamped + sum_gE) / (inv_dt + sum_g))
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(50.0));
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
/// Implicit Euler: клампит Ii к ±100 И v_new к [-100, 50] (как SIMD-путь).
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
        // Общий subexpression — тот же, что в SIMD-блоке (bit-совместимость).
        let exp_v10 = fast_exp(-vi * (1.0 / 10.0));
        let mut sum_g = 0.0f32;
        let mut sum_gE = 0.0f32;
        let mut gi = 0;

        for (ch_idx, ch) in channel_meta.iter().enumerate() {
            let g_max = *g_max_per_neuron.add(ch_idx * total_n + i);
            let mut g_ch = g_max;
            for gate_idx in 0..ch.n_gates {
                let power = ch.powers[gate_idx];
                let (alpha, beta) = match ch.gate_type[gate_idx] {
                    GateType::NaM => na_m_alpha_beta_scalar_fused(vi, exp_v10),
                    GateType::NaH => na_h_alpha_beta_scalar_fused(vi, exp_v10),
                    GateType::Kn => k_n_alpha_beta_scalar_fused(vi, exp_v10),
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
            sum_g += g_ch;
            sum_gE += g_ch * ch.e_rev;
        }
        let g_leak_i = *g_leak.add(i);
        let e_leak_i = *E_leak.add(i);
        sum_g += g_leak_i;
        sum_gE += g_leak_i * e_leak_i;

        let cm_i = *C_m.add(i);
        let inv_dt = cm_i / dt;
        let ii_clamped = (*Ii.add(i)).clamp(-100.0, 100.0);
        let v_new = ((vi * inv_dt + ii_clamped + sum_gE) / (inv_dt + sum_g)).clamp(-100.0, 50.0);
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
