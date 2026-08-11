//! # eprop_parity — pillars::kernels::eprop vs scalar formula references
//!
//!консолидации: `pillars::kernels::eprop` — канонический порт
//! легаси `kernels::neuron::{lif,izhikevich,adex,saw}` e-prop batch-кернелов.
//!
//! Каждый тест гоняет pillars-версию и независимый скалярный эталон на
//! одинаковых входных данных и требует **бит-идентичности** всех состояний
//! (v/u/e/psi/refr), спайков и total_spikes.
//!
//! ## Точность эталона
//!
//! - Экспонента в SIMD-пути — полином `fast_exp_poly_simd` (эталон вызывает
//!   ту же функцию через `pillars::math`), в tail — `f32::exp` (как легаси).
//! - FMA-группировка — НЕ-fused (`wide::mul_add` = mul+add): эталон повторяет
//!   обычную арифметику в том же порядке (IEEE-сложение коммутативно, поэтому
//!   `w.mul_add(scale, I)` == `I + w*scale` бит-в-бит).
//! - surrogate gradient tail — `pillars::kernels::learning::surrogate_gradient`
//!   (та же формула, что в pillars-порту).

use pillars::math::fast_exp::fast_exp_poly_simd;
use wide::f32x8;

/// Полиномная экспонента SIMD-пути (первая лейна).
fn exp_poly(x: f32) -> f32 {
    fast_exp_poly_simd(f32x8::splat(x)).to_array()[0]
}

/// Бит-идентичность всех буферов + возврата.
fn assert_bits(
    name: &str,
    a: &[f32],
    b: &[f32],
    sa: &[i32],
    sb: &[i32],
    total_a: usize,
    total_b: usize,
) {
    assert_eq!(
        total_a, total_b,
        "{name}: total_spikes {total_a} != {total_b}"
    );
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "{name}: f32[{i}] {:e} != {:e}",
            a[i],
            b[i]
        );
    }
    assert_eq!(sa, sb, "{name}: spikes mismatch");
}

// ═══════════════════════════════════════════════════════════
//  e-prop LIF
// ═══════════════════════════════════════════════════════════

fn ref_lif(
    v: &mut [f32],
    e: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre: &[i32],
    refr: &mut [f32],
    dt: f32,
    tau_m: f32,
    tau_e: f32,
    tau_ref: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_sg: f32,
    simd_until: usize,
) -> usize {
    let n = v.len();
    let mut total = 0usize;
    for i in 0..n {
        let is_simd = i < simd_until;
        if refr[i] > 0.0 {
            refr[i] = (refr[i] - dt).max(0.0);
            spikes[i] = 0;
            // SIMD-путь легаси ЯВНО пишет ψ = 0 в рефрактерности; tail не трогает.
            if is_simd {
                psi[i] = 0.0;
            }
            continue;
        }
        v[i] += (v_rest - v[i] + ii[i]) / tau_m * dt;
        let diff = ((v[i] - v_thresh).abs() / tau_sg).max(if is_simd { 1e-6 } else { 0.0 });
        psi[i] = 1.0 / ((1.0 + diff) * (1.0 + diff));
        let z = if pre[i] != 0 { 1.0 } else { 0.0 };
        let e_decay = if is_simd {
            exp_poly(-dt / tau_e)
        } else {
            (-dt / tau_e).exp()
        };
        e[i] = e[i] * e_decay + psi[i] * z * tau_e * (1.0 - e_decay);
        if v[i] >= v_thresh {
            spikes[i] = 1;
            total += 1;
            v[i] = v_reset;
            refr[i] = tau_ref;
        } else {
            spikes[i] = 0;
        }
    }
    total
}

#[test]
fn eprop_lif_bit_identical() {
    const N: usize = 8;
    let (dt, tau_m, tau_e, tau_ref) = (0.5f32, 10.0, 20.0, 3.0);
    let (v_rest, v_thresh, v_reset, tau_sg) = (-65.0f32, -50.0, -70.0, 1.0);
    let ii = vec![30.0f32; N];
    let pre = vec![1i32; N];

    let mut vp = vec![-65.0f32; N];
    let mut ep = vec![0.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut sp = vec![0i32; N];
    let mut rp = vec![0.0f32; N];
    let mut vr = vp.clone();
    let mut er = ep.clone();
    let mut pr = pp.clone();
    let mut sr = sp.clone();
    let mut rr = rp.clone();

    for _ in 0..300 {
        let ta = pillars::kernels::eprop::eprop_lif_step(
            &mut vp, &mut ep, &mut pp, &ii, &mut sp, &pre, &mut rp, dt, tau_m, tau_e, tau_ref,
            v_rest, v_thresh, v_reset, tau_sg,
        );
        let tb = ref_lif(
            &mut vr, &mut er, &mut pr, &ii, &mut sr, &pre, &mut rr, dt, tau_m, tau_e, tau_ref,
            v_rest, v_thresh, v_reset, tau_sg, N,
        );
        assert_bits("eprop_lif", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_e", &ep, &er, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_psi", &pp, &pr, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_refr", &rp, &rr, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_lif SIMD: bit-identical (300 steps)");
}

#[test]
fn eprop_lif_tail_bit_identical() {
    const N: usize = 13; // non-aligned → scalar tail
    let (dt, tau_m, tau_e, tau_ref) = (0.5f32, 10.0, 20.0, 3.0);
    let (v_rest, v_thresh, v_reset, tau_sg) = (-65.0f32, -50.0, -70.0, 1.0);
    let ii = vec![30.0f32; N];
    let pre = vec![1i32; N];

    let mut vp = vec![-65.0f32; N];
    let mut ep = vec![0.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut sp = vec![0i32; N];
    let mut rp = vec![0.0f32; N];
    let mut vr = vp.clone();
    let mut er = ep.clone();
    let mut pr = pp.clone();
    let mut sr = sp.clone();
    let mut rr = rp.clone();

    for _ in 0..300 {
        let ta = pillars::kernels::eprop::eprop_lif_step(
            &mut vp, &mut ep, &mut pp, &ii, &mut sp, &pre, &mut rp, dt, tau_m, tau_e, tau_ref,
            v_rest, v_thresh, v_reset, tau_sg,
        );
        // pillars обрабатывает 0..7 SIMD (exp_poly), 8..12 tail (f32::exp)
        let tb = ref_lif(
            &mut vr, &mut er, &mut pr, &ii, &mut sr, &pre, &mut rr, dt, tau_m, tau_e, tau_ref,
            v_rest, v_thresh, v_reset, tau_sg, 8,
        );
        assert_bits("eprop_lif_tail", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_tail_e", &ep, &er, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_tail_psi", &pp, &pr, &sp, &sr, ta, tb);
        assert_bits("eprop_lif_tail_refr", &rp, &rr, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_lif tail (n=13): bit-identical (300 steps)");
}

// ═══════════════════════════════════════════════════════════
//  e-prop Izhikevich (single-input)
// ═══════════════════════════════════════════════════════════

fn ref_izh(
    v: &mut [f32],
    u: &mut [f32],
    e: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_sg: f32,
    simd_exp: bool,
) -> usize {
    let n = v.len();
    let mut total = 0usize;
    let e_decay = if simd_exp {
        exp_poly(-dt / tau_e)
    } else {
        (-dt / tau_e).exp()
    };
    for i in 0..n {
        let vv = v[i];
        let v5 = vv * 0.04 + 5.0;
        let dv = v5 * vv + 140.0 - u[i] + ii[i];
        let du = a * (b * vv - u[i]);
        v[i] += dt * dv;
        u[i] += dt * du;
        let diff = (v[i] - 30.0).abs() / tau_sg;
        psi[i] = 1.0 / ((1.0 + diff) * (1.0 + diff));
        let z = if pre[i] != 0 { 1.0 } else { 0.0 };
        e[i] = e[i] * e_decay + psi[i] * z * tau_e * (1.0 - e_decay);
        if v[i] >= 30.0 {
            spikes[i] = 1;
            total += 1;
            v[i] = c;
            u[i] += d;
        } else {
            spikes[i] = 0;
        }
    }
    total
}

#[test]
fn eprop_izhikevich_bit_identical() {
    const N: usize = 8;
    let (dt, tau_e, a, b, c, d, tau_sg) = (0.5f32, 20.0, 0.02, 0.2, -65.0, 8.0, 1.0);
    let ii = vec![15.0f32; N];
    let pre = vec![1i32; N];

    let mut vp = vec![-65.0f32; N];
    let mut up = vec![-13.0f32; N];
    let mut ep = vec![0.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut sp = vec![0i32; N];
    let mut vr = vp.clone();
    let mut ur = up.clone();
    let mut er = ep.clone();
    let mut pr = pp.clone();
    let mut sr = sp.clone();

    for _ in 0..300 {
        let ta = pillars::kernels::eprop::eprop_izhikevich_step(
            &mut vp, &mut up, &mut ep, &mut pp, &ii, &mut sp, &pre, dt, tau_e, a, b, c, d, tau_sg,
        );
        let tb = ref_izh(
            &mut vr, &mut ur, &mut er, &mut pr, &ii, &mut sr, &pre, dt, tau_e, a, b, c, d, tau_sg,
            true,
        );
        assert_bits("eprop_izh", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_u", &up, &ur, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_e", &ep, &er, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_psi", &pp, &pr, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_izhikevich SIMD: bit-identical (300 steps)");
}

// ═══════════════════════════════════════════════════════════
//  e-prop Izhikevich Multi (dense weights)
// ═══════════════════════════════════════════════════════════

fn ref_izh_multi(
    v: &mut [f32],
    u: &mut [f32],
    psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    i_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_sg: f32,
    spikes: &mut [i32],
    simd_exp: bool,
) -> usize {
    let n = v.len();
    let mut total = 0usize;
    let e_decay = if simd_exp {
        exp_poly(-dt / tau_e)
    } else {
        (-dt / tau_e).exp()
    };
    for j in 0..n {
        let mut i_total = i_ext[j];
        for i in 0..n_pre {
            if z_pre[i] != 0 {
                i_total += weights[i * n + j] * 10.0;
            }
        }
        let vv = v[j];
        let v5 = vv * 0.04 + 5.0;
        let dv = v5 * vv + 140.0 - u[j] + i_total;
        let du = a * (b * vv - u[j]);
        v[j] += dt * dv;
        u[j] += dt * du;
        let diff = (v[j] - 30.0).abs() / tau_sg;
        psi[j] = 1.0 / ((1.0 + diff) * (1.0 + diff));
        for i in 0..n_pre {
            let z = z_pre[i] as f32;
            let ep = &mut e[i * n + j];
            *ep = *ep * e_decay + psi[j] * z * tau_e * (1.0 - e_decay);
        }
        if v[j] >= 30.0 {
            spikes[j] = 1;
            total += 1;
            v[j] = c;
            u[j] += d;
        } else {
            spikes[j] = 0;
        }
    }
    total
}

#[test]
fn eprop_izhikevich_multi_bit_identical() {
    const N: usize = 8;
    const N_PRE: usize = 4;
    let (dt, tau_e, a, b, c, d, tau_sg) = (0.5f32, 20.0, 0.02, 0.2, -65.0, 8.0, 1.0);
    let z_pre = vec![1i32; N_PRE];
    let i_ext = vec![0.0f32; N];
    let weights: Vec<f32> = (0..N_PRE * N).map(|i| 0.05 + 0.001 * i as f32).collect();

    let mut vp = vec![-65.0f32; N];
    let mut up = vec![-13.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut ep = vec![0.0f32; N_PRE * N];
    let mut sp = vec![0i32; N];
    let mut vr = vp.clone();
    let mut ur = up.clone();
    let mut pr = pp.clone();
    let mut er = ep.clone();
    let mut sr = sp.clone();

    for _ in 0..200 {
        let ta = pillars::kernels::eprop::eprop_izhikevich_step_multi(
            &mut vp, &mut up, &mut pp, &mut ep, &z_pre, &i_ext, &weights, N_PRE, dt, tau_e, a, b,
            c, d, tau_sg, &mut sp,
        );
        let tb = ref_izh_multi(
            &mut vr, &mut ur, &mut pr, &mut er, &z_pre, &i_ext, &weights, N_PRE, dt, tau_e, a, b,
            c, d, tau_sg, &mut sr, true,
        );
        assert_bits("eprop_izh_multi", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_multi_u", &up, &ur, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_multi_psi", &pp, &pr, &sp, &sr, ta, tb);
        assert_bits("eprop_izh_multi_e", &ep, &er, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_izhikevich_multi SIMD: bit-identical (200 steps)");
}

// ═══════════════════════════════════════════════════════════
//  e-prop AdEx Multi (dense weights)
// ═══════════════════════════════════════════════════════════

fn ref_adex_multi(
    v: &mut [f32],
    w: &mut [f32],
    psi: &mut [f32],
    e: &mut [f32],
    z_pre: &[i32],
    i_ext: &[f32],
    weights: &[f32],
    n_pre: usize,
    dt: f32,
    tau_m: f32,
    tau_w: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    delta_t: f32,
    a: f32,
    b: f32,
    tau_e: f32,
    tau_sg: f32,
    spikes: &mut [i32],
    simd_exp: bool,
) -> usize {
    let n = v.len();
    let mut total = 0usize;
    let e_decay = if simd_exp {
        exp_poly(-dt / tau_e)
    } else {
        (-dt / tau_e).exp()
    };
    for j in 0..n {
        let mut i_total = i_ext[j].clamp(-100.0, 100.0);
        for i in 0..n_pre {
            let z = z_pre[i] as f32;
            if z != 0.0 {
                i_total += weights[i * n + j] * z;
            }
        }
        let vv = v[j];
        let exp_term = if vv > v_thresh - 20.0 {
            let arg = ((vv - v_thresh) / delta_t.max(1.0)).clamp(-10.0, 10.0);
            let ev = if simd_exp { exp_poly(arg) } else { arg.exp() };
            delta_t * ev
        } else {
            0.0
        };
        let dv = (v_rest - vv + exp_term - w[j] + i_total) / tau_m * dt;
        let dw = (a * (vv - v_rest) - w[j]) / tau_w * dt;
        v[j] += dv;
        w[j] += dw;
        let diff = (v[j] - v_thresh).abs() / tau_sg;
        psi[j] = 1.0 / ((1.0 + diff) * (1.0 + diff));
        for i in 0..n_pre {
            let z = z_pre[i] as f32;
            let ep = &mut e[i * n + j];
            *ep = *ep * e_decay + psi[j] * z * tau_e * (1.0 - e_decay);
        }
        if v[j] >= v_thresh || v[j].is_nan() {
            spikes[j] = 1;
            total += 1;
            v[j] = v_reset;
            w[j] += b;
        } else {
            spikes[j] = 0;
        }
    }
    total
}

#[test]
fn eprop_adex_multi_bit_identical() {
    const N: usize = 8;
    const N_PRE: usize = 3;
    let (dt, tau_m, tau_w) = (0.1f32, 10.0, 30.0);
    let (v_rest, v_thresh, v_reset, delta_t) = (-65.0f32, -50.0, -70.0, 2.0);
    let (a, b, tau_e, tau_sg) = (0.03f32, 0.25, 20.0, 1.0);
    let z_pre = vec![1i32; N_PRE];
    let i_ext = vec![30.0f32; N];
    let weights: Vec<f32> = (0..N_PRE * N).map(|i| 0.02 + 0.001 * i as f32).collect();

    let mut vp = vec![-65.0f32; N];
    let mut wp = vec![0.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut ep = vec![0.0f32; N_PRE * N];
    let mut sp = vec![0i32; N];
    let mut vr = vp.clone();
    let mut wr = wp.clone();
    let mut pr = pp.clone();
    let mut er = ep.clone();
    let mut sr = sp.clone();

    for _ in 0..300 {
        let ta = pillars::kernels::eprop::eprop_adex_step_multi(
            &mut vp, &mut wp, &mut pp, &mut ep, &z_pre, &i_ext, &weights, N_PRE, dt, tau_m, tau_w,
            v_rest, v_thresh, v_reset, delta_t, a, b, tau_e, tau_sg, &mut sp,
        );
        let tb = ref_adex_multi(
            &mut vr, &mut wr, &mut pr, &mut er, &z_pre, &i_ext, &weights, N_PRE, dt, tau_m, tau_w,
            v_rest, v_thresh, v_reset, delta_t, a, b, tau_e, tau_sg, &mut sr, true,
        );
        assert_bits("eprop_adex", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_adex_w", &wp, &wr, &sp, &sr, ta, tb);
        assert_bits("eprop_adex_psi", &pp, &pr, &sp, &sr, ta, tb);
        assert_bits("eprop_adex_e", &ep, &er, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_adex_multi SIMD: bit-identical (300 steps)");
}

// ═══════════════════════════════════════════════════════════
//  e-prop SAW
// ═══════════════════════════════════════════════════════════

fn ref_saw(
    v: &mut [f32],
    u: &mut [f32],
    e: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    tau_sg: f32,
    simd_exp: bool,
) -> usize {
    let n = v.len();
    let mut total = 0usize;
    for i in 0..n {
        let vv = v[i];
        let v5 = vv * 0.04 + 5.0;
        let dv = v5 * vv + 140.0 - u[i] + ii[i].clamp(-100.0, 100.0);
        let du = a * (b * vv - u[i]);
        v[i] += dt * dv;
        u[i] += dt * du;
        let diff = (v[i] - 30.0).abs() / tau_sg;
        psi[i] = 1.0 / ((1.0 + diff) * (1.0 + diff));
        let z = if pre[i] != 0 { 1.0 } else { 0.0 };
        let e_decay = if simd_exp {
            exp_poly(-dt / tau_e)
        } else {
            (-dt / tau_e).exp()
        };
        e[i] = e[i] * e_decay + psi[i] * z * tau_e * (1.0 - e_decay);
        if v[i] >= 30.0 {
            spikes[i] = 1;
            total += 1;
            v[i] = -65.0;
            u[i] += 8.0;
        } else {
            spikes[i] = 0;
        }
    }
    total
}

#[test]
fn eprop_saw_bit_identical() {
    const N: usize = 8;
    let (dt, tau_e, a, b, tau_sg) = (0.5f32, 20.0, 0.02, 0.2, 1.0);
    let ii = vec![15.0f32; N];
    let pre = vec![1i32; N];

    let mut vp = vec![-65.0f32; N];
    let mut up = vec![-13.0f32; N];
    let mut ep = vec![0.0f32; N];
    let mut pp = vec![0.0f32; N];
    let mut sp = vec![0i32; N];
    let mut vr = vp.clone();
    let mut ur = up.clone();
    let mut er = ep.clone();
    let mut pr = pp.clone();
    let mut sr = sp.clone();

    for _ in 0..300 {
        let ta = pillars::kernels::eprop::eprop_saw_step(
            &mut vp, &mut up, &mut ep, &mut pp, &ii, &mut sp, &pre, dt, tau_e, a, b, tau_sg,
        );
        let tb = ref_saw(
            &mut vr, &mut ur, &mut er, &mut pr, &ii, &mut sr, &pre, dt, tau_e, a, b, tau_sg, true,
        );
        assert_bits("eprop_saw", &vp, &vr, &sp, &sr, ta, tb);
        assert_bits("eprop_saw_u", &up, &ur, &sp, &sr, ta, tb);
        assert_bits("eprop_saw_e", &ep, &er, &sp, &sr, ta, tb);
        assert_bits("eprop_saw_psi", &pp, &pr, &sp, &sr, ta, tb);
    }
    eprintln!("[parity] eprop_saw SIMD: bit-identical (300 steps)");
}

// ═══════════════════════════════════════════════════════════
//  Hebbian CSR
// ═══════════════════════════════════════════════════════════

#[test]
fn hebbian_csr_bit_identical() {
    // 3-neuron ring: 0→1, 1→2, 2→0 (как легаси-тест hebbian_csr_basic)
    let n_neurons = 3;
    let col_indices = [1i32, 2, 0];
    let row_ptr = [0i32, 1, 2, 3];
    let m = [0.8f32, 0.2, 0.6];
    let (lr, decay, dt) = (0.01f32, 0.001, 1.0);

    let mut vp = vec![0.5f32, 0.3, 0.7];
    let mut vr = vp.clone();
    for _ in 0..50 {
        pillars::kernels::learning::hebbian_csr_step(
            &mut vp,
            &col_indices,
            &row_ptr,
            &m,
            n_neurons,
            lr,
            decay,
            dt,
        );
        // скалярный эталон
        for pre in 0..n_neurons {
            let start = row_ptr[pre].max(0) as usize;
            let end = (row_ptr[pre + 1] as usize).min(vr.len());
            for j in start..end {
                let post = col_indices[j] as usize;
                let post_m = if post < n_neurons { m[post] } else { 0.0 };
                let dw = lr * m[pre] * post_m - decay * vr[j] * dt;
                vr[j] = (vr[j] + dw).clamp(-1.0, 1.0);
            }
        }
    }
    assert_eq!(vp, vr, "hebbian_csr: values mismatch");
    eprintln!("[parity] hebbian_csr: bit-identical (50 steps)");
}
