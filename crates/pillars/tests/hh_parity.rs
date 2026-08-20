//! # HH engine parity — pillars `hh_channels` vs inline legacy reference
//!
//!: движок легаси `kernels::neuron::hh` (`ChannelMeta`,
//! `build_channel_meta`, `hh_poly_step`, `fused_hh_poly_step`) портирован в
//! `pillars::kernels::hh_channels` бит-в-бит. Легаси-файл удалён; путь
//! `kernels::neuron::hh` — compat-реэкспорт pillars.
//!
//! Здесь движок pillars сравнивается с **настоящим inline-эталоном**, извлечённым
//! из git HEAD удалённого файла (wc_parity-паттерн: не тавтология, а бит-в-бит
//! проверка порта). Оба движка используют одни и те же примитивы
//! (`pillars::math::fast_exp::{fast_exp, fast_exp_poly_simd}` = pillars poly) — как и
//! легаси через `use super::{fast_exp, fast_exp_simd}`.
//!
//! ## Мембранный интегратор унифицирован (implicit Euler везде)
//!
//! Прежние расхождения «SIMD explicit Euler / скалярный implicit Euler» и
//! разный кламп `fused` v_new устранены: оба пути теперь derivimplicit (тот же
//! метод, что `HHModel`/NEURON). Один и тот же нейрон больше не меняет динамику
//! от попадания в SIMD-блок или скалярный хвост (см. n=8/9/15 parity-тесты ниже).
//!
//! - `q10_from_celsius` — guard-зона `|c−6.3| ≤ 0.1` → 1.0 (сохранена).
//! - Spec-слой (rate-функции) НЕ тестируется здесь: pillars-канон использует
//!   `fast_exp`, легаси — `.exp()` — это принятое ранее расхождение (волна 1),
//!   движок от spec-слоя не зависит (свои inline α/β через fast_exp).

#![allow(non_snake_case)]

use wide::f32x8;
use wide::{CmpGe, CmpLt};

use pillars::math::fast_exp::{fast_exp, fast_exp_poly_simd as fast_exp_simd};

// ═══════════════════════════════════════════════════════════
//  Inline legacy reference — verbatim из git HEAD
//  crates/kernels/src/neuron/hh.rs (удалён в транше 1e)
// ═══════════════════════════════════════════════════════════
mod legacy {
    use super::*;

    #[inline]
    pub fn q10_from_celsius(celsius: f32) -> f32 {
        if (celsius - 6.3).abs() > 0.1 {
            3.0f32.powf((celsius - 6.3) / 10.0)
        } else {
            1.0
        }
    }

    #[inline(always)]
    pub fn vtrap_simd(x: f32x8, y: f32) -> f32x8 {
        let y_v = f32x8::splat(y);
        let ratio = x / y_v;
        let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
        let taylor = y_v - x * f32x8::splat(0.5);
        let exact = x / (fast_exp_simd(ratio) - f32x8::splat(1.0));
        small_mask.blend(taylor, exact)
    }

    #[inline(always)]
    pub fn vtrap_scalar(x: f32, y: f32) -> f32 {
        let ratio = x / y;
        if ratio.abs() < 1e-6 {
            y - x * 0.5
        } else {
            x / (fast_exp(ratio) - 1.0)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GateType {
        NaM,
        NaH,
        Kn,
        CatC,
    }

    pub struct ChannelMeta {
        pub n_gates: usize,
        pub e_rev: f32,
        pub powers: Vec<u32>,
        pub gate_type: Vec<GateType>,
    }

    pub fn build_channel_meta(
        channels: &[pillars::kernels::hh_channels::ChannelSpec],
    ) -> Vec<ChannelMeta> {
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
                        _ => GateType::NaM,
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

    #[inline(always)]
    unsafe fn na_m_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
        let vp40 = v + f32x8::splat(40.0);
        let alpha = f32x8::splat(0.1) * vtrap_simd(-vp40, 10.0);
        let beta =
            f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
        (alpha, beta)
    }

    #[inline(always)]
    unsafe fn na_h_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
        let one = f32x8::splat(1.0);
        let alpha =
            f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
        let exp_arg = -(v + f32x8::splat(35.0)) / f32x8::splat(10.0);
        let beta = one / (one + fast_exp_simd(exp_arg));
        (alpha, beta)
    }

    #[inline(always)]
    unsafe fn k_n_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
        let vp55 = v + f32x8::splat(55.0);
        let alpha = f32x8::splat(0.01) * vtrap_simd(-vp55, 10.0);
        let beta =
            f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
        (alpha, beta)
    }

    #[inline(always)]
    unsafe fn cat_c_alpha_beta_simd(v: f32x8) -> (f32x8, f32x8) {
        let one = f32x8::splat(1.0);
        let m_inf = one / (one + fast_exp_simd(-(v + f32x8::splat(56.0)) / f32x8::splat(6.2)));
        let tau_inv = f32x8::splat(1.0 / 0.6);
        (m_inf * tau_inv, (one - m_inf) * tau_inv)
    }

    pub unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
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

    pub unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
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

    pub fn spike_mask_to_i32(mask: f32x8) -> [i32; 8] {
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

    /// SIMD polynomial HH step — explicit Euler, только полные 8-lane блоки.
    #[inline]
    pub unsafe fn hh_poly_step_simd(
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
            let mut gi = 0;

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

    #[inline(always)]
    pub fn rate_fn(v: f32, gate_type: GateType) -> (f32, f32) {
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

    /// Scalar polynomial HH step — implicit Euler (легаси-хвост).
    #[inline]
    pub unsafe fn hh_poly_step_scalar(
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
        let beta =
            f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));
        (alpha, beta)
    }

    #[inline(always)]
    unsafe fn na_h_alpha_beta_simd_fused(v: f32x8, exp_v10: f32x8) -> (f32x8, f32x8) {
        let one = f32x8::splat(1.0);
        let alpha =
            f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
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
        let beta =
            f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));
        (alpha, beta)
    }

    // Скалярные зеркала factored fused-гейтов — та же формула, что и SIMD-версии
    // выше (общий `exp_v10` + константы `EXP_V10_*`), чтобы хвост бит-совместимо
    // совпадал с SIMD-блоком (раньше хвост считал гейты напрямую).
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

    /// Fused SIMD HH step + reverse zero-crossing spike detection.
    #[inline]
    pub unsafe fn fused_hh_poly_step_simd(
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

            let spike_mask = v_old.cmp_ge(zero) & v_new.cmp_lt(zero);
            let spike_arr = spike_mask_to_i32(spike_mask);
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

    /// Fused scalar HH step + Q10 (для остатка n%8). Клампит v_new к [-100, 50].
    #[inline]
    pub unsafe fn fused_hh_poly_step_scalar(
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
            let v_new =
                ((vi * inv_dt + ii_clamped + sum_gE) / (inv_dt + sum_g)).clamp(-100.0, 50.0);
            *v.add(i) = v_new;

            *spikes.add(i) = if v_old >= 0.0 && v_new < 0.0 { 1 } else { 0 };
        }
    }

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
}

// ═══════════════════════════════════════════════════════════
//  Тесты — pillars канон (через compat-путь kernels::neuron::hh)
//  против inline legacy-эталона
// ═══════════════════════════════════════════════════════════

use pillars::kernels::hh_channels as hh;

fn assert_bits_eq(a: f32, b: f32, ctx: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{ctx}: pillars {a} (0x{:08x}) != legacy {b} (0x{:08x})",
        a.to_bits(),
        b.to_bits()
    );
}

fn run_both_poly(n: usize, channels: &[hh::ChannelSpec], dt: f32, celsius: f32, steps: usize) {
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|c| c.gates.len()).sum();

    // Идентичная детерминированная инициализация (НЕ через steady_state —
    // spec-слой fast_exp/exp расходится, движок от него не зависит).
    let mut v_p = Vec::new();
    let mut v_l = Vec::new();
    for i in 0..n {
        let base = -70.0 + (i % 9) as f32 * 2.0;
        v_p.push(base);
        v_l.push(base);
    }
    let mut gates_p = vec![0.0f32; n * n_gates];
    let mut gates_l = vec![0.0f32; n * n_gates];
    for (gi, g) in gates_p.iter_mut().enumerate() {
        *g = 0.05 + (gi % 17) as f32 * 0.03;
    }
    gates_l.copy_from_slice(&gates_p);

    let ii: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 5.0).collect();
    let g_leak: Vec<f32> = vec![0.3; n];
    let e_leak: Vec<f32> = vec![-54.3; n];
    let c_m: Vec<f32> = vec![1.0; n];
    let mut g_max = Vec::with_capacity(n * n_channels);
    for _ in 0..n {
        for ch in channels {
            g_max.push(ch.g_max);
        }
    }

    let meta_p = hh::build_channel_meta(channels);
    let meta_l = legacy::build_channel_meta(channels);

    for _ in 0..steps {
        unsafe {
            hh::hh_poly_step(
                v_p.as_mut_ptr(),
                gates_p.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta_p,
                g_max.as_ptr(),
                dt,
                celsius,
            );
            legacy::hh_poly_step(
                v_l.as_mut_ptr(),
                gates_l.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta_l,
                g_max.as_ptr(),
                dt,
                celsius,
            );
        }
    }

    for i in 0..n {
        assert_bits_eq(v_p[i], v_l[i], &format!("v[{i}] n={n}"));
    }
    for i in 0..n * n_gates {
        assert_bits_eq(gates_p[i], gates_l[i], &format!("gates[{i}] n={n}"));
    }
}

#[test]
fn hh_poly_step_parity_standard_n16() {
    run_both_poly(16, &hh::standard_channels(), 0.05, 6.3, 100);
}

#[test]
fn hh_poly_step_parity_standard_n13_tail() {
    // n=13: SIMD блок 8 + скалярный хвост 5 (implicit Euler)
    run_both_poly(13, &hh::standard_channels(), 0.05, 24.0, 100);
}

#[test]
fn hh_poly_step_parity_pacemaking_n13() {
    run_both_poly(13, &hh::pacemaking_channels(), 0.05, 24.0, 100);
}

#[test]
fn hh_poly_step_parity_pure_scalar_n7() {
    run_both_poly(7, &hh::standard_channels(), 0.1, 6.3, 50);
}

#[test]
fn hh_poly_step_parity_pure_scalar_n3() {
    run_both_poly(3, &hh::pacemaking_channels(), 0.1, 24.0, 50);
}

#[test]
fn hh_poly_step_parity_q10_guard_band() {
    // celsius в guard-зоне |c−6.3| ≤ 0.1 → q10 = 1.0 в обоих
    run_both_poly(16, &hh::standard_channels(), 0.05, 6.35, 50);
    run_both_poly(16, &hh::standard_channels(), 0.05, 6.25, 50);
}

/// Кастомный лейаут: один канал с power=2 гейтом → `build_channel_meta`
/// fallback `_ => GateType::NaM` + powf-ветка (power ∉ {1,3,4}) движка.
/// Именно этот generic-путь использует safe.rs для нестандартных лейаутов.
fn custom_layout_channels() -> Vec<hh::ChannelSpec> {
    vec![hh::ChannelSpec {
        name: "CustomX",
        gates: vec![hh::GateSpec {
            label: "q", // не m/h/n/c → fallback GateType::NaM в legacy build_channel_meta
            kinetics: hh::GateKinetics::Rates {
                forward: hh::RateForm::Closed(hh::ClosedForm::ExpLinear {
                    rate: 0.1,
                    midpoint: 40.0,
                    scale: 10.0,
                }),
                reverse: hh::RateForm::Closed(hh::ClosedForm::Exp {
                    rate: 4.0,
                    midpoint: 65.0,
                    scale: 18.0,
                }),
            },
            power: 2, // не 1/3/4 → powf-ветка
        }],
        E_rev: 40.0,
        g_max: 15.0,
    }]
}

#[test]
fn hh_poly_step_parity_custom_layout() {
    let ch = custom_layout_channels();
    // Бит-в-бит build_channel_meta (fallback GateType::NaM на label "q")
    let p = hh::build_channel_meta(&ch);
    let l = legacy::build_channel_meta(&ch);
    assert_eq!(p.len(), l.len());
    assert_eq!(p[0].powers, l[0].powers);
    let p_types: Vec<legacy::GateType> = p[0]
        .gate_type
        .iter()
        .map(|g| match g {
            hh::GateType::NaM => legacy::GateType::NaM,
            hh::GateType::NaH => legacy::GateType::NaH,
            hh::GateType::Kn => legacy::GateType::Kn,
            hh::GateType::CatC => legacy::GateType::CatC,
        })
        .collect();
    assert_eq!(p_types, l[0].gate_type);
    // Trajectory: SIMD + tail + powf power=2
    run_both_poly(13, &ch, 0.05, 24.0, 100);
    run_both_poly(7, &ch, 0.1, 6.3, 50);
}

fn run_both_fused(n: usize, channels: &[hh::ChannelSpec], dt: f32, celsius: f32, steps: usize) {
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|c| c.gates.len()).sum();

    let mut v_p = Vec::new();
    let mut v_l = Vec::new();
    for i in 0..n {
        let base = -70.0 + (i % 9) as f32 * 2.0;
        v_p.push(base);
        v_l.push(base);
    }
    let mut gates_p = vec![0.0f32; n * n_gates];
    let mut gates_l = vec![0.0f32; n * n_gates];
    for (gi, g) in gates_p.iter_mut().enumerate() {
        *g = 0.05 + (gi % 17) as f32 * 0.03;
    }
    gates_l.copy_from_slice(&gates_p);

    // Часть нейронов уже на нулевом переходе → fused должен детектить спайки
    for i in 0..n {
        if i % 3 == 0 {
            v_p[i] = -2.0;
            v_l[i] = -2.0;
        }
    }

    let ii: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 5.0).collect();
    let g_leak: Vec<f32> = vec![0.3; n];
    let e_leak: Vec<f32> = vec![-54.3; n];
    let c_m: Vec<f32> = vec![1.0; n];
    let mut g_max = Vec::with_capacity(n * n_channels);
    for _ in 0..n {
        for ch in channels {
            g_max.push(ch.g_max);
        }
    }

    let meta_p = hh::build_channel_meta(channels);
    let meta_l = legacy::build_channel_meta(channels);
    let mut spikes_p = vec![0i32; n];
    let mut spikes_l = vec![0i32; n];

    for _ in 0..steps {
        unsafe {
            hh::fused_hh_poly_step(
                v_p.as_mut_ptr(),
                gates_p.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta_p,
                g_max.as_ptr(),
                dt,
                celsius,
                spikes_p.as_mut_ptr(),
            );
            legacy::fused_hh_poly_step(
                v_l.as_mut_ptr(),
                gates_l.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta_l,
                g_max.as_ptr(),
                dt,
                celsius,
                spikes_l.as_mut_ptr(),
            );
        }
        for i in 0..n {
            assert_bits_eq(v_p[i], v_l[i], &format!("fused v[{i}] n={n}"));
        }
        for i in 0..n {
            assert_eq!(
                spikes_p[i], spikes_l[i],
                "fused spikes[{i}] n={n}: pillars {} != legacy {}",
                spikes_p[i], spikes_l[i]
            );
        }
    }
}

#[test]
fn fused_hh_poly_step_parity_n16() {
    run_both_fused(16, &hh::pacemaking_channels(), 0.05, 24.0, 100);
}

#[test]
fn fused_hh_poly_step_parity_n13_tail() {
    run_both_fused(13, &hh::standard_channels(), 0.05, 24.0, 100);
}

#[test]
fn fused_hh_poly_step_parity_pure_scalar_n7() {
    run_both_fused(7, &hh::pacemaking_channels(), 0.1, 6.3, 50);
}

#[test]
fn build_channel_meta_parity() {
    for channels in [hh::standard_channels(), hh::pacemaking_channels()] {
        let p = hh::build_channel_meta(&channels);
        let l = legacy::build_channel_meta(&channels);
        assert_eq!(p.len(), l.len(), "channel count");
        for (pi, li) in p.iter().zip(l.iter()) {
            assert_eq!(pi.n_gates, li.n_gates, "n_gates");
            assert_eq!(pi.e_rev.to_bits(), li.e_rev.to_bits(), "e_rev");
            assert_eq!(pi.powers, li.powers, "powers");
            let p_types: Vec<legacy::GateType> = pi
                .gate_type
                .iter()
                .map(|g| match g {
                    hh::GateType::NaM => legacy::GateType::NaM,
                    hh::GateType::NaH => legacy::GateType::NaH,
                    hh::GateType::Kn => legacy::GateType::Kn,
                    hh::GateType::CatC => legacy::GateType::CatC,
                })
                .collect();
            assert_eq!(p_types, li.gate_type, "gate_type");
        }
    }
}

#[test]
fn q10_from_celsius_parity() {
    for &c in &[0.0f32, 6.0, 6.25, 6.3, 6.35, 6.4, 10.0, 24.0, 37.0, 100.0] {
        assert_bits_eq(
            hh::q10_from_celsius(c),
            legacy::q10_from_celsius(c),
            &format!("q10({c})"),
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  SIMD-блок vs скалярный хвост: ИДЕНТИЧНЫЕ лейны
// ═══════════════════════════════════════════════════════════
//  Унифицированный мембранный интегратор гарантирует: один и тот же нейрон
//  (идентичное состояние/параметры/вход) эволюционирует одинаково, попал ли он
//  в SIMD-блок (индекс < 8) или в скалярный хвост (индекс ≥ 8 при n%8 ≠ 0).
//  Раньше SIMD-блок шёл explicit Euler, а хвост — implicit Euler, и нейрон
//  менял динамику от размера популяции.

/// Все n нейронов с идентичным состоянием/параметрами/входом; прогон через
/// `hh_poly_step`. Возвращает max |V[i] − V[0]| по i (SIMD vs хвост).
fn run_identical_lanes_poly(n: usize, dt: f32, celsius: f32, steps: usize, ii_val: f32) -> f32 {
    let channels = hh::standard_channels();
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|c| c.gates.len()).sum();

    let mut v = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * n_gates];
    for g in 0..n_gates {
        let q = 0.2 + g as f32 * 0.1;
        for i in 0..n {
            gates[g * n + i] = q;
        }
    }
    let ii = vec![ii_val; n];
    let g_leak = vec![0.3f32; n];
    let e_leak = vec![-54.3f32; n];
    let c_m = vec![1.0f32; n];
    // Канальный лейаут: [ch0×n][ch1×n] (как `ConductancePopulation`).
    let mut g_max = Vec::with_capacity(n * n_channels);
    for ch in &channels {
        for _ in 0..n {
            g_max.push(ch.g_max);
        }
    }
    let meta = hh::build_channel_meta(&channels);

    for _ in 0..steps {
        unsafe {
            hh::hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta,
                g_max.as_ptr(),
                dt,
                celsius,
            );
        }
    }

    let mut max_dv = 0.0f32;
    for i in 1..n {
        max_dv = max_dv.max((v[i] - v[0]).abs());
    }
    max_dv
}

/// То же для `fused_hh_poly_step` (со спайк-детекцией).
fn run_identical_lanes_fused(n: usize, dt: f32, celsius: f32, steps: usize, ii_val: f32) -> f32 {
    let channels = hh::pacemaking_channels();
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|c| c.gates.len()).sum();

    let mut v = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * n_gates];
    for g in 0..n_gates {
        let q = 0.2 + g as f32 * 0.1;
        for i in 0..n {
            gates[g * n + i] = q;
        }
    }
    let ii = vec![ii_val; n];
    let g_leak = vec![0.3f32; n];
    let e_leak = vec![-54.3f32; n];
    let c_m = vec![1.0f32; n];
    let mut g_max = Vec::with_capacity(n * n_channels);
    for ch in &channels {
        for _ in 0..n {
            g_max.push(ch.g_max);
        }
    }
    let meta = hh::build_channel_meta(&channels);
    let mut spikes = vec![0i32; n];

    for _ in 0..steps {
        unsafe {
            hh::fused_hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                ii.as_ptr(),
                g_leak.as_ptr(),
                e_leak.as_ptr(),
                c_m.as_ptr(),
                n,
                &meta,
                g_max.as_ptr(),
                dt,
                celsius,
                spikes.as_mut_ptr(),
            );
        }
    }

    let mut max_dv = 0.0f32;
    for i in 1..n {
        max_dv = max_dv.max((v[i] - v[0]).abs());
    }
    max_dv
}

#[test]
fn hh_poly_step_simd_tail_identical_lanes() {
    for n in [9usize, 15] {
        let dv = run_identical_lanes_poly(n, 0.05, 24.0, 100, 1.0);
        println!("hh_poly_step n={n}: max |ΔV| SIMD/tail = {dv:.3e} mV");
        assert!(
            dv < 1e-3,
            "n={n}: SIMD-блок и хвост должны совпадать, max |ΔV| = {dv:.3e} mV"
        );
    }
}

#[test]
fn fused_hh_poly_step_simd_tail_identical_lanes() {
    for n in [9usize, 15] {
        let dv = run_identical_lanes_fused(n, 0.05, 24.0, 100, 1.0);
        println!("fused_hh_poly_step n={n}: max |ΔV| SIMD/tail = {dv:.3e} mV");
        assert!(
            dv < 1e-3,
            "n={n}: SIMD-блок и хвост должны совпадать, max |ΔV| = {dv:.3e} mV"
        );
    }
}

/// Экстремальный ток (1000 нА → кламп в ±100 в fused; без клампа в poly)
/// не должен расщеплять SIMD-блок и хвост: один и тот же нейрон даёт одну
/// траекторию при n=8 (полный блок), 9/15 (блок+хвост), 17 (2 блока+хвост).
#[test]
fn hh_poly_step_extreme_input_simd_tail_identical_lanes() {
    for n in [8usize, 9, 15, 17] {
        let dv = run_identical_lanes_poly(n, 0.05, 24.0, 100, 1000.0);
        assert!(
            dv < 1e-3,
            "poly n={n}: SIMD/хвост расходятся при экстремальном токе, max |ΔV| = {dv:.3e} mV"
        );
    }
}

#[test]
fn fused_hh_poly_step_extreme_input_simd_tail_identical_lanes() {
    for n in [8usize, 9, 15, 17] {
        let dv = run_identical_lanes_fused(n, 0.05, 24.0, 100, 1000.0);
        assert!(
            dv < 1e-3,
            "fused n={n}: SIMD/хвост расходятся при экстремальном токе, max |ΔV| = {dv:.3e} mV"
        );
    }
}
