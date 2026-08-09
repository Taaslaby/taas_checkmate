//! # 🧮 Fused layer kernels — spike propagation + neuron dynamics, one SIMD pass
//!
//! Canonical port of the legacy `kernels::neuron::fused` batch kernels
//! (Фаза 2/3 консолидации — single source of truth в pillars):
//!
//! | Function | Dynamics | Used by |
//! |----------|----------|---------|
//! | [`fused_hidden_izhikevich_step`] | Izhikevich (single-step Euler) + weight accumulation | RL hidden layer, `neuron_tests` |
//! | [`fused_output_lif_step`] | LIF (Exponential Euler) + weight accumulation | RL output readout |
//! | [`poisson_fused_hidden_step`] | Poisson RNG → weights → Izhikevich | RL hidden layer (fused encoder) |
//!
//! ## Philosophy
//!
//! - **Safe API, SIMD internals** — flat `&[f32]` slices in, `f32x8` lanes
//!   (AVX2/NEON) with a scalar tail out.
//! - **Bit-identical to legacy** — same FMA grouping (`mul_add`), same
//!   polynomial [`fast_exp_poly_simd`](crate::math::fast_exp::fast_exp_poly_simd)
//!   for LIF decay, same LCG Poisson RNG (`RNG_SCALE_24`). Guarded by
//!   `crates/kernels/tests/fused_parity.rs`.
//! - **Zero allocation hot path** — the caller-supplied `active_buf` is the
//!   only scratch; no per-step `Vec` creation.
//!
//! # ⚠️ RNG semantics — НЕ упрощать до `input::PoissonGenerator`
//!
//! `poisson_fused_hidden_step` продвигает LCG **всегда** (даже при rate=0),
//! повторяя легаси-кернел и `train/rl.rs` траектории. `input::PoissonGenerator::
//! step` продвигает RNG **только при p > 0** — траектории разойдутся, если
//! заменить fused-путь на `PoissonGenerator` + `fused_hidden_izhikevich_step`.
//!
//! # Weight layout
//!
//! Weights are **column-major**: `weights[i * n_post + j]` — row `i`
//! (pre-neuron), column `j` (post-neuron). Post-size must be padded to ×8.

use wide::{f32x8, CmpGe, CmpGt};

use crate::core::util::{load8, mask_to_spikes, store8};
use crate::kernels::RNG_SCALE_24;
use crate::math::fast_exp::fast_exp_poly_simd;

// ═══════════════════════════════════════════════════════════
//  Fused Izhikevich with weight accumulation (hidden layer)
// ═══════════════════════════════════════════════════════════

/// Fused hidden layer: accumulate input spikes → Izhikevich step in one pass.
///
/// ```text
/// I[j..j+8] += Σ_{i active} weights[i·nh + j..j+8] · z_pre[i] · 10.0
/// Izhikevich single-step Forward Euler (v_th = 30, reset c/d, trace STDP)
/// ```
///
/// Uses the caller-supplied `active_buf` (pre-allocated) to iterate only the
/// active pre-synapses instead of all `ni` inputs (sparse optimization).
///
/// # Panics
/// On slice-length mismatch; `v.len()` (nh) must be a multiple of 8.
pub fn fused_hidden_izhikevich_step(
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
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(
        u.len(),
        nh,
        "fused_hidden_izhikevich_step: u length mismatch"
    );
    assert_eq!(
        trace.len(),
        nh,
        "fused_hidden_izhikevich_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "fused_hidden_izhikevich_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "fused_hidden_izhikevich_step: refractory length mismatch"
    );
    assert_eq!(
        a.len(),
        nh,
        "fused_hidden_izhikevich_step: a length mismatch"
    );
    assert_eq!(
        b.len(),
        nh,
        "fused_hidden_izhikevich_step: b length mismatch"
    );
    assert_eq!(
        c.len(),
        nh,
        "fused_hidden_izhikevich_step: c length mismatch"
    );
    assert_eq!(
        d.len(),
        nh,
        "fused_hidden_izhikevich_step: d length mismatch"
    );
    assert_eq!(
        weights.len(),
        ni * nh,
        "fused_hidden_izhikevich_step: weights length mismatch (expected {}*{}={})",
        ni,
        nh,
        ni * nh
    );
    assert!(
        input_spikes.len() >= ni,
        "fused_hidden_izhikevich_step: input_spikes length {} < ni {ni}",
        input_spikes.len()
    );
    assert!(
        nh.is_multiple_of(8),
        "fused_hidden_izhikevich_step: nh must be multiple of 8, got {nh}"
    );

    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let scale10 = f32x8::splat(10.0);
    let tr_s = f32x8::splat(tau_ref);

    // Active pre-synapse list (sparse optimization) — reuse caller buffer.
    active_buf.clear();
    for i in 0..ni {
        if input_spikes[i] != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    let n8 = nh & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let mut tr = load8(trace.as_ptr(), off);
            let mut refr = load8(refractory.as_ptr(), off);
            let aa = load8(a.as_ptr(), off);
            let bb = load8(b.as_ptr(), off);
            let cc = load8(c.as_ptr(), off);
            let dd = load8(d.as_ptr(), off);

            // Accumulate currents from active pre-synapses only
            let mut i_s = zero;
            for &i in active {
                let w = load8(weights.as_ptr(), i * nh + off);
                i_s = w.mul_add(scale10, i_s); // I += w · 10
            }

            let refr_mask = refr.cmp_gt(zero);

            let v5 = vv.mul_add(c04, c5);
            let dv = v5.mul_add(vv, c140) - uu + i_s;
            let du = aa * (bb * vv - uu);
            vv += refr_mask.blend(zero, dt_s * dv);
            uu += refr_mask.blend(zero, dt_s * du);

            let not_refr = !refr_mask;
            let sm = vv.cmp_ge(v_th) & not_refr;

            vv = refr_mask.blend(vv, sm.blend(cc, vv));
            uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

            refr -= dt_s;
            refr = sm.blend(tr_s, refr);
            refr = refr.max(zero);

            // STDP trace: on spike tr = tr·td + (1−tr), else tr·(1−td)
            let tr_s_val = tr.mul_add(td_s, one - tr);
            let tr_n_val = tr * (one - td_s);
            tr = refr_mask.blend(tr, sm.blend(tr_s_val, tr_n_val));

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(trace.as_mut_ptr(), off, tr);
            store8(refractory.as_mut_ptr(), off, refr);

            let sp = mask_to_spikes(sm);
            spikes[off..off + 8].copy_from_slice(&sp);
        }
    }

    // Scalar tail (empty since nh is padded to ×8 — kept for parity).
    if n8 < nh {
        for j in n8..nh {
            let mut i_val = 0.0f32;
            for &i in active {
                i_val += weights[i * nh + j] * 10.0;
            }
            if refractory[j] > 0.0 {
                refractory[j] = (refractory[j] - dt).max(0.0);
                spikes[j] = 0;
                continue;
            }
            let orig_v = v[j];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[j] + i_val;
            let du = a[j] * (b[j] * orig_v - u[j]);
            v[j] += dt * dv;
            u[j] += dt * du;
            if v[j] >= 30.0 {
                spikes[j] = 1;
                v[j] = c[j];
                u[j] += d[j];
                refractory[j] = tau_ref;
                trace[j] += trace_decay * (1.0 - trace[j]);
            } else {
                spikes[j] = 0;
                trace[j] *= 1.0 - trace_decay;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Fused LIF with weight accumulation (output layer)
// ═══════════════════════════════════════════════════════════

/// Fused output layer: accumulate hidden spikes → LIF step in one pass.
///
/// ```text
/// I[k..k+8] += Σ_{j: z_j≠0} weights[j·na + k..k+8] · z_hidden[j] · 10.0
/// LIF Exponential Euler: v = v_rest + (v−v_rest)·exp(−dt/τ) + I·(1−exp(−dt/τ))
/// ```
///
/// # Panics
/// On slice-length mismatch; `v.len()` (na) must be a multiple of 8.
pub fn fused_output_lif_step(
    v: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_th: &[f32],
    v_reset: &[f32],
    weights: &[f32],
    hidden_spikes: &[i32],
    dt: f32,
    tau_ref: f32,
    nh: usize,
) {
    let na = v.len();
    assert_eq!(
        spikes.len(),
        na,
        "fused_output_lif_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        na,
        "fused_output_lif_step: refractory length mismatch"
    );
    assert_eq!(
        tau_m.len(),
        na,
        "fused_output_lif_step: tau_m length mismatch"
    );
    assert_eq!(
        v_rest.len(),
        na,
        "fused_output_lif_step: v_rest length mismatch"
    );
    assert_eq!(
        v_th.len(),
        na,
        "fused_output_lif_step: v_th length mismatch"
    );
    assert_eq!(
        v_reset.len(),
        na,
        "fused_output_lif_step: v_reset length mismatch"
    );
    assert_eq!(
        weights.len(),
        nh * na,
        "fused_output_lif_step: weights length mismatch (expected {}*{}={})",
        nh,
        na,
        nh * na
    );
    assert!(
        hidden_spikes.len() >= nh,
        "fused_output_lif_step: hidden_spikes length {} < nh {nh}",
        hidden_spikes.len()
    );
    assert!(
        na.is_multiple_of(8),
        "fused_output_lif_step: na must be multiple of 8, got {na}"
    );

    let dt_s = f32x8::splat(dt);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let scale10 = f32x8::splat(10.0);

    let n8 = na & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut refr = load8(refractory.as_ptr(), off);
            let v_rest_p = load8(v_rest.as_ptr(), off);
            let v_th_p = load8(v_th.as_ptr(), off);
            let v_reset_p = load8(v_reset.as_ptr(), off);
            let tau_m_p = load8(tau_m.as_ptr(), off);

            // Accumulate current from all hidden spikes directly in register
            let mut i_s = zero;
            for j in 0..nh {
                if hidden_spikes[j] != 0 {
                    let w = load8(weights.as_ptr(), j * na + off);
                    i_s = w.mul_add(scale10, i_s);
                }
            }

            // LIF dynamics with refractory — Exponential Euler
            let refr_mask = refr.cmp_gt(zero);
            let mask_tau = tau_m_p.cmp_gt(zero);
            let decay = mask_tau.blend(fast_exp_poly_simd(-dt_s / tau_m_p), zero);
            let one_minus_decay = one - decay;
            let new_v = v_rest_p + (vv - v_rest_p) * decay + i_s * one_minus_decay;
            vv = refr_mask.blend(vv, mask_tau.blend(new_v, vv));

            let not_refr = !refr_mask;
            let sm = vv.cmp_ge(v_th_p) & not_refr;
            vv = refr_mask.blend(vv, sm.blend(v_reset_p, vv));

            refr -= dt_s;
            refr = sm.blend(tr_s, refr);
            refr = refr.max(zero);

            store8(v.as_mut_ptr(), off, vv);
            store8(refractory.as_mut_ptr(), off, refr);

            let sp = mask_to_spikes(sm);
            spikes[off..off + 8].copy_from_slice(&sp);
        }
    }

    // Scalar tail (empty since na is padded to ×8 — kept for parity).
    if n8 < na {
        for k in n8..na {
            let mut i_val = 0.0f32;
            for j in 0..nh {
                if hidden_spikes[j] != 0 {
                    i_val += weights[j * na + k] * 10.0;
                }
            }
            if refractory[k] > 0.0 {
                refractory[k] = (refractory[k] - dt).max(0.0);
                spikes[k] = 0;
                continue;
            }
            if tau_m[k] > 0.0 {
                let decay = (-dt / tau_m[k]).exp();
                v[k] = v_rest[k] + (v[k] - v_rest[k]) * decay + i_val * (1.0 - decay);
            }
            if v[k] >= v_th[k] {
                spikes[k] = 1;
                v[k] = v_reset[k];
                refractory[k] = tau_ref;
            } else {
                spikes[k] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Poisson-fused Izhikevich — RNG → weights → dynamics in one pass
// ═══════════════════════════════════════════════════════════

/// Fused kernel: Poisson spike generation + weight accumulation + Izhikevich
/// in a single SIMD pass (eliminates the encoder step entirely).
///
/// ```text
/// for each input i:  Poisson spike via LCG (per-neuron RNG state)
/// for each 8-block:  I += Σ_{i active} weights[i·nh + j..] · 10
///                    Izhikevich single-step Forward Euler (as above)
/// ```
///
/// `input_spikes_out` receives the generated Poisson flags (for downstream
/// eligibility traces). ~20% faster than separate encoder + fused hidden step.
///
/// # Panics
/// On slice-length mismatch; `v.len()` (nh) must be a multiple of 8.
pub fn poisson_fused_hidden_step(
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
    rng_states: &mut [u32],
    rates: &[f32],
    input_spikes_out: &mut [i32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(u.len(), nh, "poisson_fused_hidden_step: u length mismatch");
    assert_eq!(
        trace.len(),
        nh,
        "poisson_fused_hidden_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "poisson_fused_hidden_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "poisson_fused_hidden_step: refractory length mismatch"
    );
    assert_eq!(a.len(), nh, "poisson_fused_hidden_step: a length mismatch");
    assert_eq!(b.len(), nh, "poisson_fused_hidden_step: b length mismatch");
    assert_eq!(c.len(), nh, "poisson_fused_hidden_step: c length mismatch");
    assert_eq!(d.len(), nh, "poisson_fused_hidden_step: d length mismatch");
    assert_eq!(
        weights.len(),
        ni * nh,
        "poisson_fused_hidden_step: weights length mismatch (expected {}*{}={})",
        ni,
        nh,
        ni * nh
    );
    assert!(
        rng_states.len() >= ni,
        "poisson_fused_hidden_step: rng_states length {} < ni {ni}",
        rng_states.len()
    );
    assert!(
        rates.len() >= ni,
        "poisson_fused_hidden_step: rates length {} < ni {ni}",
        rates.len()
    );
    assert!(
        input_spikes_out.len() >= ni,
        "poisson_fused_hidden_step: input_spikes_out length {} < ni {ni}",
        input_spikes_out.len()
    );
    assert!(
        nh.is_multiple_of(8),
        "poisson_fused_hidden_step: nh must be multiple of 8, got {nh}"
    );

    let dt_sec = dt / 1000.0; // ms → sec for Poisson probability
    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let scale10 = f32x8::splat(10.0);
    let tr_s = f32x8::splat(tau_ref);

    // Step 1: Poisson spikes for all inputs (1 LCG each) — bit-identical LCG.
    for i in 0..ni {
        let state = rng_states[i].wrapping_mul(1664525).wrapping_add(1013904223);
        rng_states[i] = state;
        let r = (state >> 8) as f32 * RNG_SCALE_24;
        let p = (rates[i] * dt_sec).clamp(0.0, 1.0);
        input_spikes_out[i] = if r < p { 1 } else { 0 };
    }

    // Step 2: active pre-synapse list (sparse) — reuse caller buffer.
    active_buf.clear();
    for i in 0..ni {
        if input_spikes_out[i] != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    // Step 3: Fused Izhikevich — ⚠️ ДУБЛИКАТ блока из fused_hidden_izhikevich_step
    // (дословная верность легаси). Правка динамики — в ОБА места синхронно.
    let n8 = nh & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let mut tr = load8(trace.as_ptr(), off);
            let mut refr = load8(refractory.as_ptr(), off);
            let aa = load8(a.as_ptr(), off);
            let bb = load8(b.as_ptr(), off);
            let cc = load8(c.as_ptr(), off);
            let dd = load8(d.as_ptr(), off);

            let mut i_s = zero;
            for &i in active {
                let w = load8(weights.as_ptr(), i * nh + off);
                i_s = w.mul_add(scale10, i_s);
            }

            let refr_mask = refr.cmp_gt(zero);
            let v5 = vv.mul_add(c04, c5);
            let dv = v5.mul_add(vv, c140) - uu + i_s;
            let du = aa * (bb * vv - uu);
            vv += refr_mask.blend(zero, dt_s * dv);
            uu += refr_mask.blend(zero, dt_s * du);

            let not_refr = !refr_mask;
            let sm = vv.cmp_ge(v_th) & not_refr;
            vv = refr_mask.blend(vv, sm.blend(cc, vv));
            uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

            refr -= dt_s;
            refr = sm.blend(tr_s, refr);
            refr = refr.max(zero);

            let tr_s_val = tr.mul_add(td_s, one - tr);
            let tr_n_val = tr * (one - td_s);
            tr = refr_mask.blend(tr, sm.blend(tr_s_val, tr_n_val));

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(trace.as_mut_ptr(), off, tr);
            store8(refractory.as_mut_ptr(), off, refr);

            let sp = mask_to_spikes(sm);
            spikes[off..off + 8].copy_from_slice(&sp);
        }
    }

    // Scalar tail (empty since nh is padded to ×8 — kept for parity).
    if n8 < nh {
        for j in n8..nh {
            let mut i_val = 0.0f32;
            for &i in active {
                i_val += weights[i * nh + j] * 10.0;
            }
            if refractory[j] > 0.0 {
                refractory[j] = (refractory[j] - dt).max(0.0);
                spikes[j] = 0;
                continue;
            }
            let orig_v = v[j];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[j] + i_val;
            let du = a[j] * (b[j] * orig_v - u[j]);
            v[j] += dt * dv;
            u[j] += dt * du;
            if v[j] >= 30.0 {
                spikes[j] = 1;
                v[j] = c[j];
                u[j] += d[j];
                refractory[j] = tau_ref;
                trace[j] += trace_decay * (1.0 - trace[j]);
            } else {
                spikes[j] = 0;
                trace[j] *= 1.0 - trace_decay;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_izhikevich_accumulates_active_inputs() {
        let nh = 8;
        let ni = 8;
        let mut v = vec![-65.0f32; nh];
        let mut u = vec![-13.0f32; nh];
        let mut trace = vec![0.0f32; nh];
        let mut spikes = vec![0i32; nh];
        let mut refractory = vec![0.0f32; nh];
        let a = vec![0.02f32; nh];
        let b = vec![0.2f32; nh];
        let c = vec![-65.0f32; nh];
        let d = vec![8.0f32; nh];
        let weights: Vec<f32> = (0..ni * nh).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut input_spikes = vec![0i32; ni];
        input_spikes[0] = 1;
        input_spikes[3] = 1;
        let mut active_buf = Vec::with_capacity(ni);
        let mut total = 0i32;
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
                0.5,
                0.5,
                3.0,
                ni,
                &mut active_buf,
            );
            total += spikes.iter().sum::<i32>();
        }
        // I[j] = (weights[j] + weights[24+j])·10 ≈ 2.6–4.0 — лейны 5–7 (I ≥ 3.6)
        // раскачивают v до 30 и спайкают → total > 0.
        assert!(total > 0, "hidden layer should spike, got {total}");
    }

    #[test]
    fn output_lif_integrates_hidden_spikes() {
        let na = 8;
        let nh = 8;
        let mut v = vec![-70.0f32; na];
        let mut spikes = vec![0i32; na];
        let mut refractory = vec![0.0f32; na];
        let tau_m = vec![10.0f32; na];
        let v_rest = vec![-70.0f32; na];
        let v_th = vec![-55.0f32; na];
        let v_reset = vec![-75.0f32; na];
        let weights: Vec<f32> = (0..nh * na).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let hidden_spikes = vec![1i32; nh];
        let mut total = 0i32;
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
                1.0,
                3.0,
                nh,
            );
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "output LIF should fire with strong drive");
    }

    #[test]
    fn poisson_fused_is_deterministic() {
        let nh = 8;
        let ni = 8;
        let run_once = |v0: f32| {
            let mut v = vec![v0; nh];
            let mut u = vec![-13.0f32; nh];
            let mut trace = vec![0.0f32; nh];
            let mut spikes = vec![0i32; nh];
            let mut refractory = vec![0.0f32; nh];
            let a = vec![0.02f32; nh];
            let b = vec![0.2f32; nh];
            let c = vec![-65.0f32; nh];
            let d = vec![8.0f32; nh];
            let weights: Vec<f32> = (0..ni * nh).map(|i| (i as f32 + 1.0) * 0.01).collect();
            let mut rng = vec![42u32; ni];
            let rates = vec![100.0f32; ni];
            let mut in_sp = vec![0i32; ni];
            let mut active_buf = Vec::with_capacity(ni);
            let mut total = 0i32;
            for _ in 0..50 {
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
                    0.5,
                    0.5,
                    3.0,
                    ni,
                    &mut active_buf,
                );
                total += spikes.iter().sum::<i32>();
            }
            (total, v, u)
        };
        let (t1, v1, u1) = run_once(-65.0);
        let (t2, v2, u2) = run_once(-65.0);
        assert_eq!(t1, t2, "LCG must be deterministic");
        assert_eq!(v1, v2, "v must be deterministic");
        assert_eq!(u1, u2, "u must be deterministic");
    }
}
