//! # Fused layer parity: `pillars::kernels::fused` vs scalar formula reference
//!
//! Консолидация/3): legacy `kernels::neuron::fused` больше НЕ
//! вызывается из `safe.rs` — обёртки `safe::{fused_hidden_izhikevich_step,
//! fused_output_lif_step, poisson_fused_hidden_step}` делегируют канону
//! `pillars::kernels::fused`. Живой потребитель — hot path `train/rl.rs`
//! (Actor-Critic RL, dopamine-STDP).
//!
//! ## Что тестируется
//! 1. **Скрытый слой** (`fused_hidden_izhikevich_step`) — бит-в-бит против
//!    скалярного эталона с той же FMA-группировкой (`mul_add`), тем же
//!    активным списком, той же последовательностью накопления I.
//! 2. **Выходной LIF** (`fused_output_lif_step`) — бит-в-бит против эталона
//!    с полиномом [`fast_exp`](pillars::math::fast_exp::fast_exp) (тот же
//!    полином, что `fast_exp_poly_simd` в SIMD-пути).
//! 3. **Poisson-fused** (`poisson_fused_hidden_step`) — детерминированный LCG
//!    + та же динамика против эталона.
//!
//! ⚠️ **Coupling удаления:** эталон повторяет формулу легаси-кернелов
//! (`kernels/neuron/fused.rs`). Когда легаси-кернел будет удалён,
//! этот тест остаётся — он закрепляет поведение канона независимо от легаси.

use pillars::kernels::fused::{
    fused_hidden_izhikevich_step, fused_output_lif_step, poisson_fused_hidden_step,
};
use pillars::math::fast_exp::fast_exp;

const NH: usize = 8;
const NI: usize = 8;
const DT: f32 = 0.5;
const TRACE_DECAY: f32 = 0.5;
const TAU_REF: f32 = 3.0;

/// Column-major weights `[ni·nh]`, small positive values.
fn make_weights(n_pre: usize, n_post: usize) -> Vec<f32> {
    (0..n_pre * n_post)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect()
}

fn make_sparse_input(ni: usize) -> Vec<i32> {
    let mut z = vec![0i32; ni];
    z[1] = 1;
    z[4] = 1;
    z[6] = 1;
    z
}

/// Scalar reference for `fused_hidden_izhikevich_step` — SAME op grouping.
fn ref_hidden_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    weights: &[f32],
    input_spikes: &[i32],
    ni: usize,
) {
    let nh = v.len();
    for j in 0..nh {
        // ⚠️ НЕ fused: `wide::f32x8::mul_add` в SIMD-кернеле — это mul+add
        // (два округления), в отличие от fused `f32::mul_add`. Эталон обязан
        // повторять ту же не-fused группировку для бит-в-бит равенства.
        let mut i_val = 0.0f32;
        for i in 0..ni {
            if input_spikes[i] != 0 {
                i_val += weights[i * nh + j] * 10.0;
            }
        }
        if refractory[j] > 0.0 {
            refractory[j] = (refractory[j] - DT).max(0.0);
            spikes[j] = 0;
            continue;
        }
        let orig_v = v[j];
        let v5 = 0.04 * orig_v + 5.0;
        let dv = v5 * orig_v + 140.0 - u[j] + i_val;
        let du = a[j] * (b[j] * orig_v - u[j]);
        v[j] += DT * dv;
        u[j] += DT * du;
        if v[j] >= 30.0 {
            spikes[j] = 1;
            v[j] = c[j];
            u[j] += d[j];
            refractory[j] = TAU_REF;
            trace[j] = trace[j] * TRACE_DECAY + (1.0 - trace[j]);
        } else {
            spikes[j] = 0;
            trace[j] *= 1.0 - TRACE_DECAY;
        }
    }
}

/// Scalar reference for `fused_output_lif_step` — polynomial decay (same bits
/// as `fast_exp_poly_simd` in the SIMD path).
fn ref_output_lif_step(
    v: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_th: &[f32],
    v_reset: &[f32],
    weights: &[f32],
    hidden_spikes: &[i32],
    nh: usize,
) {
    let na = v.len();
    for k in 0..na {
        // Не-fused накопление (см. ref_hidden_step): wide mul_add ≠ f32 mul_add.
        let mut i_val = 0.0f32;
        for j in 0..nh {
            if hidden_spikes[j] != 0 {
                i_val += weights[j * na + k] * 10.0;
            }
        }
        if refractory[k] > 0.0 {
            refractory[k] = (refractory[k] - DT).max(0.0);
            spikes[k] = 0;
            continue;
        }
        if tau_m[k] > 0.0 {
            let decay = fast_exp(-DT / tau_m[k]);
            v[k] = v_rest[k] + (v[k] - v_rest[k]) * decay + i_val * (1.0 - decay);
        }
        if v[k] >= v_th[k] {
            spikes[k] = 1;
            v[k] = v_reset[k];
            refractory[k] = TAU_REF;
        } else {
            spikes[k] = 0;
        }
    }
}

#[test]
fn hidden_izhikevich_bit_identical_to_reference() {
    let mut v = vec![-65.0f32; NH];
    let mut u = vec![-13.0f32; NH];
    let mut trace = vec![0.0f32; NH];
    let mut spikes = vec![0i32; NH];
    let mut refractory = vec![0.0f32; NH];
    let a = vec![0.02f32; NH];
    let b = vec![0.2f32; NH];
    let c = vec![-65.0f32; NH];
    let d = vec![8.0f32; NH];
    let weights = make_weights(NI, NH);
    let input_spikes = make_sparse_input(NI);
    let mut active_buf = Vec::with_capacity(NI);

    let mut rv = vec![-65.0f32; NH];
    let mut ru = vec![-13.0f32; NH];
    let mut rt = vec![0.0f32; NH];
    let mut rs = vec![0i32; NH];
    let mut rr = vec![0.0f32; NH];

    for _ in 0..200 {
        fused_hidden_izhikevich_step(
            &mut v,
            &mut u,
            &mut trace,
            &mut spikes,
            &mut refractory,
            &a,
            &b,
            &c,
            &d,
            &weights,
            &input_spikes,
            DT,
            TRACE_DECAY,
            TAU_REF,
            NI,
            &mut active_buf,
        );
        ref_hidden_step(
            &mut rv,
            &mut ru,
            &mut rt,
            &mut rs,
            &mut rr,
            &a,
            &b,
            &c,
            &d,
            &weights,
            &input_spikes,
            NI,
        );
        assert_eq!(v, rv, "v diverged");
        assert_eq!(u, ru, "u diverged");
        assert_eq!(trace, rt, "trace diverged");
        assert_eq!(refractory, rr, "refractory diverged");
        assert_eq!(spikes, rs, "spikes diverged");
    }
    eprintln!(
        "[parity] fused hidden: 200 steps bit-identical, spikes={} ✓",
        spikes.iter().sum::<i32>()
    );
}

#[test]
fn output_lif_bit_identical_to_reference() {
    let na = NH;
    let mut v = vec![-70.0f32; na];
    let mut spikes = vec![0i32; na];
    let mut refractory = vec![0.0f32; na];
    let tau_m = vec![10.0f32; na];
    let v_rest = vec![-70.0f32; na];
    let v_th = vec![-55.0f32; na];
    let v_reset = vec![-75.0f32; na];
    let weights = make_weights(NH, na);
    let mut hidden_spikes = vec![0i32; NH];
    hidden_spikes[0] = 1;
    hidden_spikes[3] = 1;
    hidden_spikes[5] = 1;

    let mut rv = vec![-70.0f32; na];
    let mut rs = vec![0i32; na];
    let mut rr = vec![0.0f32; na];

    for _ in 0..200 {
        fused_output_lif_step(
            &mut v,
            &mut spikes,
            &mut refractory,
            &tau_m,
            &v_rest,
            &v_th,
            &v_reset,
            &weights,
            &hidden_spikes,
            DT,
            TAU_REF,
            NH,
        );
        ref_output_lif_step(
            &mut rv,
            &mut rs,
            &mut rr,
            &tau_m,
            &v_rest,
            &v_th,
            &v_reset,
            &weights,
            &hidden_spikes,
            NH,
        );
        assert_eq!(v, rv, "v diverged");
        assert_eq!(refractory, rr, "refractory diverged");
        assert_eq!(spikes, rs, "spikes diverged");
    }
    eprintln!(
        "[parity] fused output LIF: 200 steps bit-identical, spikes={} ✓",
        spikes.iter().sum::<i32>()
    );
}

#[test]
fn poisson_fused_matches_reference() {
    // One run with the pillars kernel, one with the scalar reference (same LCG).
    let run = |kernel: bool| -> (i32, Vec<f32>, Vec<f32>, Vec<i32>) {
        let mut v = vec![-65.0f32; NH];
        let mut u = vec![-13.0f32; NH];
        let mut trace = vec![0.0f32; NH];
        let mut spikes = vec![0i32; NH];
        let mut refractory = vec![0.0f32; NH];
        let a = vec![0.02f32; NH];
        let b = vec![0.2f32; NH];
        let c = vec![-65.0f32; NH];
        let d = vec![8.0f32; NH];
        let weights = make_weights(NI, NH);
        let mut rng = vec![42u32; NI];
        let rates = vec![120.0f32; NI];
        let mut in_sp = vec![0i32; NI];
        let mut active_buf = Vec::with_capacity(NI);
        let mut total = 0i32;
        for _ in 0..100 {
            let mut input_flags = vec![0i32; NI];
            if kernel {
                poisson_fused_hidden_step(
                    &mut v,
                    &mut u,
                    &mut trace,
                    &mut spikes,
                    &mut refractory,
                    &a,
                    &b,
                    &c,
                    &d,
                    &weights,
                    &mut rng,
                    &rates,
                    &mut in_sp,
                    DT,
                    TRACE_DECAY,
                    TAU_REF,
                    NI,
                    &mut active_buf,
                );
            } else {
                // Reference: identical LCG + identical dynamics
                for i in 0..NI {
                    let st = rng[i].wrapping_mul(1664525).wrapping_add(1013904223);
                    rng[i] = st;
                    let r = (st >> 8) as f32 * (1.0 / 16777216.0);
                    let p = (rates[i] * DT / 1000.0).clamp(0.0, 1.0);
                    input_flags[i] = if r < p { 1 } else { 0 };
                }
                ref_hidden_step(
                    &mut v,
                    &mut u,
                    &mut trace,
                    &mut spikes,
                    &mut refractory,
                    &a,
                    &b,
                    &c,
                    &d,
                    &weights,
                    &input_flags,
                    NI,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        (total, v, u, in_sp)
    };

    let (k_total, kv, ku, k_in) = run(true);
    let (r_total, rv, ru, r_in) = run(false);
    assert_eq!(k_total, r_total, "spike totals diverged");
    assert_eq!(kv, rv, "v diverged");
    assert_eq!(ku, ru, "u diverged");
    assert_eq!(k_in, r_in, "input poisson flags diverged");
    eprintln!(
        "[parity] poisson-fused hidden: 100 steps bit-identical, spikes={} ✓",
        k_total
    );
}
