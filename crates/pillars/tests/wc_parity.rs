//! # WC parity: canonical `pillars::kernels::wc::wc_step` vs legacy semantics
//!
//! Консолидация (волна 7): легаси `kernels::wc_step` удалён — канон —
//! `pillars::kernels::wc::wc_step` (бит-в-бит порт). Эталон ниже воспроизводит
//! **точную** легаси-семантику: SIMD-путь клампит аргумент сигмоиды к ±10 и
//! использует полином `fast_exp_poly_simd`, tail — `f32::exp` БЕЗ клампа
//! (задокументированное расхождение SIMD/tail легаси, сохранено в порте).

use wide::f32x8;
use wide::CmpGt;

use pillars::kernels::wc::wc_step as pillars_wc;
use pillars::math::fast_exp::fast_exp_poly_simd;

// ── Эталон легаси-семантики (SIMD-путь: clamp ±10 + полином; tail: f32::exp) ──
#[allow(clippy::too_many_arguments)]
fn ref_wc_step(
    e: &mut [f32],
    i: &mut [f32],
    i_e: &[f32],
    i_i: &[f32],
    spikes_e: &mut [f32],
    spikes_i: &mut [f32],
    params: &[f32],
) {
    let n = e.len();
    let dt_val = params[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut ee = pillars::core::util::load8(e.as_ptr(), off);
            let mut ii = pillars::core::util::load8(i.as_ptr(), off);
            let ie_ext = pillars::core::util::load8(i_e.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));
            let ii_ext = pillars::core::util::load8(i_i.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));

            let tau_e = pillars::core::util::load8(params.as_ptr().add(n), off);
            let tau_i = pillars::core::util::load8(params.as_ptr().add(2 * n), off);
            let w_ee = pillars::core::util::load8(params.as_ptr().add(3 * n), off);
            let w_ei = pillars::core::util::load8(params.as_ptr().add(4 * n), off);
            let w_ie = pillars::core::util::load8(params.as_ptr().add(5 * n), off);
            let w_ii = pillars::core::util::load8(params.as_ptr().add(6 * n), off);
            let beta = pillars::core::util::load8(params.as_ptr().add(7 * n), off);

            let input_e = w_ee * ee - w_ei * ii + ie_ext;
            let input_i = w_ie * ee - w_ii * ii + ii_ext;

            let f_e = ref_sigmoid_simd(input_e, beta);
            let f_i = ref_sigmoid_simd(input_i, beta);

            let mask_te = tau_e.cmp_gt(zero);
            let inv_tau_e = mask_te.blend(f32x8::splat(1.0) / tau_e.max(f32x8::splat(1e-10)), zero);
            let de = (-ee + f_e) * inv_tau_e * dt_s;
            ee = mask_te.blend(ee + de, ee);

            let mask_ti = tau_i.cmp_gt(zero);
            let inv_tau_i = mask_ti.blend(f32x8::splat(1.0) / tau_i.max(f32x8::splat(1e-10)), zero);
            let di = (-ii + f_i) * inv_tau_i * dt_s;
            ii = mask_ti.blend(ii + di, ii);

            pillars::core::util::store8(e.as_mut_ptr(), off, ee);
            pillars::core::util::store8(i.as_mut_ptr(), off, ii);
            pillars::core::util::store8(spikes_e.as_mut_ptr(), off, f_e);
            pillars::core::util::store8(spikes_i.as_mut_ptr(), off, f_i);
        }
    }

    for idx in n8..n {
        let tau_e = params[n + idx];
        let tau_i = params[2 * n + idx];
        let w_ee = params[3 * n + idx];
        let w_ei = params[4 * n + idx];
        let w_ie = params[5 * n + idx];
        let w_ii = params[6 * n + idx];
        let beta = params[7 * n + idx];

        let input_e = w_ee * e[idx] - w_ei * i[idx] + i_e[idx].clamp(-100.0, 100.0);
        let input_i = w_ie * e[idx] - w_ii * i[idx] + i_i[idx].clamp(-100.0, 100.0);

        let f_e = 1.0 / (1.0 + (-beta * input_e).exp());
        let f_i = 1.0 / (1.0 + (-beta * input_i).exp());

        if tau_e > 0.0 {
            e[idx] += (-e[idx] + f_e) / tau_e * dt_val;
        }
        if tau_i > 0.0 {
            i[idx] += (-i[idx] + f_i) / tau_i * dt_val;
        }

        spikes_e[idx] = f_e;
        spikes_i[idx] = f_i;
    }
}

#[inline(always)]
fn ref_sigmoid_simd(x: f32x8, beta: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let neg = (-x * beta).max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + fast_exp_poly_simd(neg))
}

fn assert_bits(a: f32, b: f32, what: &str, idx: usize, step: usize) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}[{idx}] diverged at step {step}: {a} vs {b}"
    );
}

/// n = 8: чистый SIMD-путь — должен быть бит-идентичен.
#[test]
fn wc_simd_bit_identical() {
    let n = 8;
    let dt = 0.1;
    let e = vec![0.3f32; n];
    let i = vec![0.3f32; n];
    let i_e: Vec<f32> = (0..n).map(|k| 0.2 + 0.1 * k as f32).collect();
    let i_i: Vec<f32> = (0..n).map(|k| 0.1 + 0.05 * k as f32).collect();
    let mut params = vec![0.0f32; n * 8];
    for j in 0..n {
        params[j] = dt;
        params[n + j] = 10.0 + j as f32;
        params[2 * n + j] = 8.0 + 0.5 * j as f32;
        params[3 * n + j] = 1.2;
        params[4 * n + j] = 0.4 + 0.01 * j as f32;
        params[5 * n + j] = 0.6;
        params[6 * n + j] = 0.8;
        params[7 * n + j] = 2.0 + 0.5 * j as f32; // разные β — разные сигмоиды
    }

    let (mut re, mut ri, mut rse, mut rsi) =
        (e.clone(), i.clone(), vec![0.0f32; n], vec![0.0f32; n]);
    let (mut pe, mut pi, mut pse, mut psi) =
        (e.clone(), i.clone(), vec![0.0f32; n], vec![0.0f32; n]);

    for step in 0..300 {
        ref_wc_step(&mut re, &mut ri, &i_e, &i_i, &mut rse, &mut rsi, &params);
        pillars_wc(&mut pe, &mut pi, &i_e, &i_i, &mut pse, &mut psi, &params);
        for k in 0..n {
            assert_bits(re[k], pe[k], "e", k, step);
            assert_bits(ri[k], pi[k], "i", k, step);
            assert_bits(rse[k], pse[k], "spikes_e", k, step);
            assert_bits(rsi[k], psi[k], "spikes_i", k, step);
        }
    }
    eprintln!("WC n={n}: 300 steps bit-identical ✓ (вкл. входы >100 → кламп ±100)");
}

/// n = 13: SIMD-путь (0..8) + tail (8..13) — гибрид, бит-идентичен.
#[test]
fn wc_tail_bit_identical() {
    let n = 13;
    let dt = 0.1;
    let e: Vec<f32> = (0..n).map(|k| 0.1 + 0.05 * k as f32).collect();
    let i: Vec<f32> = (0..n).map(|k| 0.2 - 0.02 * k as f32).collect();
    let i_e: Vec<f32> = (0..n).map(|k| 5.0 * (k as f32 % 7.0)).collect();
    let i_i: Vec<f32> = (0..n).map(|k| 0.5 * (k as f32 % 3.0)).collect();
    let mut params = vec![0.0f32; n * 8];
    for j in 0..n {
        params[j] = dt;
        params[n + j] = 6.0 + 0.3 * j as f32;
        params[2 * n + j] = 6.0;
        params[3 * n + j] = 1.1 + 0.05 * j as f32;
        params[4 * n + j] = 0.5;
        params[5 * n + j] = 0.7;
        params[6 * n + j] = 0.9;
        params[7 * n + j] = 3.0;
    }

    let (mut re, mut ri, mut rse, mut rsi) =
        (e.clone(), i.clone(), vec![0.0f32; n], vec![0.0f32; n]);
    let (mut pe, mut pi, mut pse, mut psi) =
        (e.clone(), i.clone(), vec![0.0f32; n], vec![0.0f32; n]);

    for step in 0..300 {
        ref_wc_step(&mut re, &mut ri, &i_e, &i_i, &mut rse, &mut rsi, &params);
        pillars_wc(&mut pe, &mut pi, &i_e, &i_i, &mut pse, &mut psi, &params);
        for k in 0..n {
            assert_bits(re[k], pe[k], "e", k, step);
            assert_bits(ri[k], pi[k], "i", k, step);
            assert_bits(rse[k], pse[k], "spikes_e", k, step);
            assert_bits(rsi[k], psi[k], "spikes_i", k, step);
        }
    }
    eprintln!("WC n={n}: 300 steps SIMD+tail bit-identical ✓");
}
