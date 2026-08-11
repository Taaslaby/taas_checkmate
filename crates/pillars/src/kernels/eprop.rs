//! # 🎓 E-Prop microkernels — eligibility-trace SNN training, one SIMD pass
//!
//! Canonical port of the legacy `kernels::neuron::{lif,izhikevich,adex,saw}`
//! e-prop batch kernels (волна 5 консолидации — single source of truth в
//! pillars):
//!
//! | Function | Dynamics | Used by |
//! |----------|----------|---------|
//! | [`eprop_lif_step`] | LIF (Forward Euler) + eligibility, refractory-freeze | `core::neuron::lif` (Python), `neuron_tests` |
//! | [`eprop_izhikevich_step`] | Izhikevich (single-step Euler) + eligibility | `core::neuron::izhikevich` (Python) |
//! | [`eprop_izhikevich_step_multi`] | dense weights + per-synapse eligibility | `core::neuron::izhikevich` (Python) |
//! | [`eprop_adex_step_multi`] | AdEx (Euler, exp-clamped) + per-synapse eligibility | `core::neuron::adex` (Python) |
//! | [`eprop_saw_step`] | SAW dynamics + eligibility | `pillars::zoo::neurons::SAWModel` (канон) |
//!
//! ## Philosophy
//!
//! - **Safe API, SIMD internals** — flat `&[f32]`/`&[i32]` slices in, `f32x8`
//!   lanes (AVX2/NEON) with a scalar tail out.
//! - **Bit-identical to legacy** — same FMA grouping (`wide::mul_add` is
//!   NOT fused), same polynomial [`fast_exp_poly_simd`](crate::math::fast_exp::fast_exp_poly_simd)
//!   for τ_e decay and AdEx exp-term, same NaN-spike semantics. Guarded by
//!   `crates/kernels/tests/eprop_parity.rs`.
//! - **Zero allocation** — dense `e[n_pre × n]` is written in place.
//!
//! # Semantics kept from legacy (НЕ упрощать)
//!
//! - `eprop_lif_step`/`eprop_saw_step` clamp I to ±100; `eprop_adex_step_multi`
//!   clamps I to ±100 **and** the exp argument to ±10 with a `v > v_th − 20`
//!   mask (exp term zeroed below).
//! - `eprop_adex_step_multi` treats NaN v as a spike (`v.cmp_ne(v)` OR-ed into
//!   the spike mask).
//! - `eprop_lif_step` freezes eligibility/psi during refractory and zeroes psi
//!   there; the surrogate `diff` is floored at `1e-6` (lif only).
//! - `eprop_izhikevich_step_multi` uses the 10× weight convention (matches
//!   `fused_hidden_izhikevich_step`); eligibility decays only when `z_pre = 0`.

use wide::{f32x8, CmpGe, CmpGt, CmpNe};

use crate::core::util::{load8, load8_i32, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;

#[inline(always)]
fn count_spikes(sm: f32x8) -> usize {
    mask_to_spikes(sm).iter().sum::<i32>() as usize
}

// ═══════════════════════════════════════════════════════════
//  E-Prop LIF
// ═══════════════════════════════════════════════════════════

/// Step `n` e-prop LIF neurons with eligibility traces (refractory-aware).
///
/// ```text
/// LIF (Forward Euler): v += (v_rest − v + I) / τ_m · dt   (no exp)
/// surrogate:          ψ = 1 / (1 + |v − v_th|/τ_sg)²   (diff floored at 1e-6)
/// eligibility:        e = e·e_decay + ψ·z_pre·τ_e·(1 − e_decay)
/// ```
///
/// During refractory, `v`, `e` and `ψ` are frozen (spike detection disabled).
///
/// # Returns
/// Total number of output spikes across all neurons.
pub fn eprop_lif_step(
    v: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    refractory: &mut [f32],
    dt: f32,
    tau_m: f32,
    tau_e: f32,
    tau_ref: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_lif_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_lif_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_lif_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_lif_step: spikes length mismatch");
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_lif_step: pre_spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        n,
        "eprop_lif_step: refractory length mismatch"
    );

    let dt_s = f32x8::splat(dt);
    let tau_m_s = f32x8::splat(tau_m);
    let tau_e_s = f32x8::splat(tau_e);
    let v_rest_s = f32x8::splat(v_rest);
    let v_thresh_s = f32x8::splat(v_thresh);
    let v_reset_s = f32x8::splat(v_reset);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    let e_decay = fast_exp_poly_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut ee = load8(eligibility.as_ptr(), off);
            let mut refr = load8(refractory.as_ptr(), off);
            let ii_v = load8(ii.as_ptr(), off);
            let pre = load8_i32(pre_spikes.as_ptr(), off);
            let refr_mask = refr.cmp_gt(zero);

            // LIF dynamics (skip if refractory — eligibility/psi also frozen)
            let dv = (v_rest_s - vv + ii_v) / tau_m_s * dt_s;
            vv = refr_mask.blend(vv, vv + dv);

            // Surrogate gradient (only non-refractory)
            let diff = ((vv - v_thresh_s).abs() / tau_sg_s).max(f32x8::splat(1e-6));
            let one_plus_diff = one + diff;
            let mut ps = one / (one_plus_diff * one_plus_diff);
            ps = refr_mask.blend(zero, ps);

            // Eligibility trace (frozen during refractory)
            let z_pre = pre.cmp_gt(zero).blend(one, zero);
            let mut e_new = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;
            e_new = refr_mask.blend(ee, e_new);
            ee = e_new;

            // Spike detection (only non-refractory)
            let not_refr = !refr_mask;
            let sm = vv.cmp_ge(v_thresh_s) & not_refr;
            vv = refr_mask.blend(vv, sm.blend(v_reset_s, vv));

            // Decrement refractory for all, set to tau_ref on spike
            refr -= dt_s;
            refr = sm.blend(tr_s, refr);
            refr = refr.max(zero);

            total_spikes += count_spikes(sm);

            store8(v.as_mut_ptr(), off, vv);
            store8(eligibility.as_mut_ptr(), off, ee);
            store8(psi.as_mut_ptr(), off, ps);
            store8(refractory.as_mut_ptr(), off, refr);
            spikes[off..off + 8].copy_from_slice(&mask_to_spikes(sm));
        }
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for i in n8..n {
            if refractory[i] > 0.0 {
                refractory[i] = (refractory[i] - dt).max(0.0);
                spikes[i] = 0;
                continue; // v, eligibility, psi frozen during refractory
            }
            v[i] += (v_rest - v[i] + ii[i]) / tau_m * dt;
            psi[i] = crate::kernels::learning::surrogate_gradient(v[i], v_thresh, tau_sg);
            let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
            eligibility[i] =
                eligibility[i] * e_decay_s + psi[i] * z_pre * tau_e * one_minus_e_decay_s;
            if v[i] >= v_thresh {
                spikes[i] = 1;
                total_spikes += 1;
                v[i] = v_reset;
                refractory[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich (single-input)
// ═══════════════════════════════════════════════════════════

/// Step `n` e-prop Izhikevich neurons with eligibility traces.
///
/// ```text
/// Izhikevich (Forward Euler): dv = (0.04·v + 5)·v + 140 − u + I
/// if v ≥ 30: v = c, u += d
/// eligibility (exponential Euler): e = e·e_decay + ψ·z_pre·τ_e·(1 − e_decay)
/// ```
///
/// # Returns
/// Total number of output spikes across all neurons.
pub fn eprop_izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_izhikevich_step: u length mismatch");
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_izhikevich_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_izhikevich_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_izhikevich_step: Ii length mismatch");
    assert_eq!(
        spikes.len(),
        n,
        "eprop_izhikevich_step: spikes length mismatch"
    );
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_izhikevich_step: pre_spikes length mismatch"
    );

    let dt_s = f32x8::splat(dt);
    let tau_e_s = f32x8::splat(tau_e);
    let a_s = f32x8::splat(a);
    let b_s = f32x8::splat(b);
    let c_s = f32x8::splat(c);
    let d_s = f32x8::splat(d);
    let v_th = f32x8::splat(30.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    let e_decay = fast_exp_poly_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let mut ee = load8(eligibility.as_ptr(), off);
            let ii_v = load8(ii.as_ptr(), off);
            let pre = load8_i32(pre_spikes.as_ptr(), off);

            let v5 = vv.mul_add(c04, c5);
            let dv = v5.mul_add(vv, c140) - uu + ii_v;
            let du = a_s * (b_s * vv - uu);
            vv += dt_s * dv;
            uu += dt_s * du;

            let diff = (vv - v_th).abs() / tau_sg_s;
            let one_plus_diff = one + diff;
            let ps = one / (one_plus_diff * one_plus_diff);

            let z_pre = pre.cmp_gt(zero).blend(one, zero);
            ee = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;

            let sm = vv.cmp_ge(v_th);
            vv = sm.blend(c_s, vv);
            uu += sm.blend(d_s, zero);

            total_spikes += count_spikes(sm);

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(eligibility.as_mut_ptr(), off, ee);
            store8(psi.as_mut_ptr(), off, ps);
            spikes[off..off + 8].copy_from_slice(&mask_to_spikes(sm));
        }
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for i in n8..n {
            let vv = v[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u[i] + ii[i];
            let du = a * (b * vv - u[i]);
            v[i] += dt * dv;
            u[i] += dt * du;
            psi[i] = crate::kernels::learning::surrogate_gradient(v[i], 30.0, tau_sg);
            let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
            eligibility[i] =
                eligibility[i] * e_decay_s + psi[i] * z_pre * tau_e * one_minus_e_decay_s;
            if v[i] >= 30.0 {
                spikes[i] = 1;
                total_spikes += 1;
                v[i] = c;
                u[i] += d;
            } else {
                spikes[i] = 0;
            }
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════════
//  E-Prop Izhikevich Multi (dense weights, per-synapse eligibility)
// ═══════════════════════════════════════════════════════════

/// Step `n` e-prop Izhikevich neurons with **per-synapse** eligibility traces
/// over `n_pre` presynaptic inputs (dense column-major weights).
///
/// ```text
/// I_total = I_ext + 10 · Σ_i weights[i·n + j] · z_pre[i]   (10× convention)
/// Izhikevich (Forward Euler); eligibility e_ij per synapse (exponential Euler)
/// ```
///
/// `e` and `weights` are column-major: `[n_pre × n]` (`e[i·n + j]`).
///
/// # Returns
/// Total number of output spikes across all neurons.
pub fn eprop_izhikevich_step_multi(
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
) -> usize {
    assert!(n_pre > 0, "eprop_izhikevich_step_multi: n_pre must be > 0");
    let n = v.len(); // n_post (column stride)
    let synapse_len = n_pre
        .checked_mul(n)
        .expect("eprop_izhikevich_step_multi: n_pre * n_post overflows");
    assert_eq!(u.len(), n, "eprop_izhikevich_step_multi: u length mismatch");
    assert_eq!(
        psi.len(),
        n,
        "eprop_izhikevich_step_multi: psi length mismatch"
    );
    assert_eq!(
        e.len(),
        synapse_len,
        "eprop_izhikevich_step_multi: e length mismatch"
    );
    assert_eq!(
        z_pre.len(),
        n_pre,
        "eprop_izhikevich_step_multi: z_pre length mismatch"
    );
    assert_eq!(
        i_ext.len(),
        n,
        "eprop_izhikevich_step_multi: I_ext length mismatch"
    );
    assert_eq!(
        weights.len(),
        synapse_len,
        "eprop_izhikevich_step_multi: weights length mismatch"
    );
    assert_eq!(
        spikes.len(),
        n,
        "eprop_izhikevich_step_multi: spikes length mismatch"
    );

    let dt_s = f32x8::splat(dt);
    let tau_e_s = f32x8::splat(tau_e);
    let a_s = f32x8::splat(a);
    let b_s = f32x8::splat(b);
    let c_s = f32x8::splat(c);
    let d_s = f32x8::splat(d);
    let v_th = f32x8::splat(30.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let scale = f32x8::splat(10.0);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    let e_decay = fast_exp_poly_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let ii_v = load8(i_ext.as_ptr(), off);

            // I_total = I_ext + SCALE · Σ_i weights[i·n + j] · z_pre[i]
            let mut i_s = ii_v;
            for i in 0..n_pre {
                if *z_pre.get_unchecked(i) != 0 {
                    let w = load8(weights.as_ptr(), i * n + off);
                    i_s = w.mul_add(scale, i_s);
                }
            }

            let v5 = vv.mul_add(c04, c5);
            let dv = v5.mul_add(vv, c140) - uu + i_s;
            let du = a_s * (b_s * vv - uu);
            vv += dt_s * dv;
            uu += dt_s * du;

            let diff = (vv - v_th).abs() / tau_sg_s;
            let one_plus_diff = one + diff;
            let ps = one / (one_plus_diff * one_plus_diff);

            let sm = vv.cmp_ge(v_th);
            vv = sm.blend(c_s, vv);
            uu += sm.blend(d_s, zero);

            total_spikes += count_spikes(sm);

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(psi.as_mut_ptr(), off, ps);

            // Per-synapse eligibility: exponential Euler
            for i in 0..n_pre {
                let z = *z_pre.get_unchecked(i) as f32;
                let e_off = i * n + off;
                let mut e_simd = load8(e.as_ptr(), e_off);
                if z != 0.0 {
                    e_simd = e_simd * e_decay + ps * f32x8::splat(z) * tau_e_s * one_minus_e_decay;
                } else {
                    e_simd *= e_decay;
                }
                store8(e.as_mut_ptr(), e_off, e_simd);
            }

            spikes[off..off + 8].copy_from_slice(&mask_to_spikes(sm));
        }
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for j in n8..n {
            let mut vv = v[j];
            let mut uu = u[j];

            let mut i_total = i_ext[j];
            for i in 0..n_pre {
                if z_pre[i] != 0 {
                    i_total += weights[i * n + j] * 10.0;
                }
            }

            let dv = (0.04 * vv + 5.0) * vv + 140.0 - uu + i_total;
            let du = a * (b * vv - uu);
            vv += dt * dv;
            uu += dt * du;

            psi[j] = crate::kernels::learning::surrogate_gradient(vv, 30.0, tau_sg);

            for i in 0..n_pre {
                let z = z_pre[i] as f32;
                let e_ptr = &mut e[i * n + j];
                *e_ptr = *e_ptr * e_decay_s + psi[j] * z * tau_e * one_minus_e_decay_s;
            }

            if vv >= 30.0 {
                spikes[j] = 1;
                total_spikes += 1;
                vv = c;
                uu += d;
            } else {
                spikes[j] = 0;
            }

            v[j] = vv;
            u[j] = uu;
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════════
//  E-Prop AdEx Multi (dense weights, per-synapse eligibility)
// ═══════════════════════════════════════════════════════════

/// Step `n` e-prop AdEx neurons with per-synapse eligibility traces.
///
/// ```text
/// dv = (v_rest − v + Δ·exp((v − v_T)/Δ) − w + I) / τ_m · dt   (exp clamped ±10)
/// dw = (a·(v − v_rest) − w) / τ_w · dt
/// v ≥ v_thresh (или v NaN) → v = v_reset, w += b
/// ```
///
/// I_ext clamped to ±100; the exp-term is zeroed for `v ≤ v_thresh − 20`.
/// `e` and `weights` are column-major `[n_pre × n]`.
///
/// # Returns
/// Total number of output spikes across all neurons.
pub fn eprop_adex_step_multi(
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
) -> usize {
    assert!(n_pre > 0, "eprop_adex_step_multi: n_pre must be > 0");
    assert!(
        tau_m > 0.0 && tau_w > 0.0,
        "eprop_adex_step_multi: tau_m/tau_w must be > 0"
    );
    let n = v.len(); // n_post (column stride)
    let synapse_len = n_pre
        .checked_mul(n)
        .expect("eprop_adex_step_multi: n_pre * n_post overflows");
    assert_eq!(w.len(), n, "eprop_adex_step_multi: w length mismatch");
    assert_eq!(psi.len(), n, "eprop_adex_step_multi: psi length mismatch");
    assert_eq!(
        e.len(),
        synapse_len,
        "eprop_adex_step_multi: e length mismatch"
    );
    assert_eq!(
        z_pre.len(),
        n_pre,
        "eprop_adex_step_multi: z_pre length mismatch"
    );
    assert_eq!(
        i_ext.len(),
        n,
        "eprop_adex_step_multi: I_ext length mismatch"
    );
    assert_eq!(
        weights.len(),
        synapse_len,
        "eprop_adex_step_multi: weights length mismatch"
    );
    assert_eq!(
        spikes.len(),
        n,
        "eprop_adex_step_multi: spikes length mismatch"
    );

    let dt_s = f32x8::splat(dt);
    let tau_m_s = f32x8::splat(tau_m);
    let tau_w_s = f32x8::splat(tau_w);
    let v_rest_s = f32x8::splat(v_rest);
    let v_th_s = f32x8::splat(v_thresh);
    let v_reset_s = f32x8::splat(v_reset);
    let delta_t_s = f32x8::splat(delta_t);
    let a_s = f32x8::splat(a);
    let b_s = f32x8::splat(b);
    let tau_e_s = f32x8::splat(tau_e);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    let e_decay = fast_exp_poly_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut ww = load8(w.as_ptr(), off);
            // Fused Ii clamp — предотвращает saturation cascade от oversized weights
            let ii_v = load8(i_ext.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));

            // I_total = I_ext + Σ_i weights[i·n + j] · z_pre[i]
            let mut i_s = ii_v;
            for i in 0..n_pre {
                let z = *z_pre.get_unchecked(i) as f32;
                if z != 0.0 {
                    let wgt = load8(weights.as_ptr(), i * n + off);
                    i_s = wgt.mul_add(f32x8::splat(z), i_s);
                }
            }

            // AdEx dynamics (exp arg clamped ±10)
            let exp_arg = ((vv - v_th_s) / delta_t_s.max(f32x8::splat(1.0)))
                .max(f32x8::splat(-10.0))
                .min(f32x8::splat(10.0));
            let exp_val = fast_exp_poly_simd(exp_arg);
            let exp_term = delta_t_s * exp_val;
            // Zero out where v ≤ v_T − 20
            let v_gt_mask = vv.cmp_gt(v_th_s - f32x8::splat(20.0));
            let exp_term = v_gt_mask.blend(exp_term, zero);

            let dv = (v_rest_s - vv + exp_term - ww + i_s) / tau_m_s * dt_s;
            let dw = (a_s * (vv - v_rest_s) - ww) / tau_w_s * dt_s;
            vv += dv;
            ww += dw;

            let diff = (vv - v_th_s).abs() / tau_sg_s;
            let one_plus_diff = one + diff;
            let ps = one / (one_plus_diff * one_plus_diff);

            // Spike detection: v ≥ v_thresh OR NaN
            let nan_mask = vv.cmp_ne(vv);
            let sm = vv.cmp_ge(v_th_s) | nan_mask;
            vv = sm.blend(v_reset_s, vv);
            ww += sm.blend(b_s, zero);

            total_spikes += count_spikes(sm);

            store8(v.as_mut_ptr(), off, vv);
            store8(w.as_mut_ptr(), off, ww);
            store8(psi.as_mut_ptr(), off, ps);

            // Per-synapse eligibility: exponential Euler
            for i in 0..n_pre {
                let z = *z_pre.get_unchecked(i) as f32;
                let e_off = i * n + off;
                let mut e_simd = load8(e.as_ptr(), e_off);
                if z != 0.0 {
                    e_simd = e_simd * e_decay + ps * f32x8::splat(z) * tau_e_s * one_minus_e_decay;
                } else {
                    e_simd *= e_decay;
                }
                store8(e.as_mut_ptr(), e_off, e_simd);
            }

            spikes[off..off + 8].copy_from_slice(&mask_to_spikes(sm));
        }
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for j in n8..n {
            let mut vv = v[j];
            let mut ww = w[j];

            let mut i_total = i_ext[j].clamp(-100.0, 100.0);
            for i in 0..n_pre {
                let z = z_pre[i] as f32;
                if z != 0.0 {
                    i_total += weights[i * n + j] * z;
                }
            }

            let exp_term = if vv > v_thresh - 20.0 {
                delta_t * ((vv - v_thresh) / delta_t.max(1.0)).exp()
            } else {
                0.0
            };
            let dv = (v_rest - vv + exp_term - ww + i_total) / tau_m * dt;
            let dw = (a * (vv - v_rest) - ww) / tau_w * dt;
            vv += dv;
            ww += dw;

            psi[j] = crate::kernels::learning::surrogate_gradient(vv, v_thresh, tau_sg);

            for i in 0..n_pre {
                let z = z_pre[i] as f32;
                let e_ptr = &mut e[i * n + j];
                *e_ptr = *e_ptr * e_decay_s + psi[j] * z * tau_e * one_minus_e_decay_s;
            }

            if vv >= v_thresh || vv.is_nan() {
                spikes[j] = 1;
                total_spikes += 1;
                vv = v_reset;
                ww += b;
            } else {
                spikes[j] = 0;
            }

            v[j] = vv;
            w[j] = ww;
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════════
//  E-Prop SAW (Spike-Adaptive-Width)
// ═══════════════════════════════════════════════════════════

/// Step `n` e-prop SAW neurons with eligibility traces.
///
/// SAW dynamics match Izhikevich (hardcoded reset v = −65, u += 8),
/// I input clamped to ±100.
///
/// # Returns
/// Total number of output spikes across all neurons.
pub fn eprop_saw_step(
    v: &mut [f32],
    u: &mut [f32],
    eligibility: &mut [f32],
    psi: &mut [f32],
    ii: &[f32],
    spikes: &mut [i32],
    pre_spikes: &[i32],
    dt: f32,
    tau_e: f32,
    a: f32,
    b: f32,
    tau_sg: f32,
) -> usize {
    let n = v.len();
    assert_eq!(u.len(), n, "eprop_saw_step: u length mismatch");
    assert_eq!(
        eligibility.len(),
        n,
        "eprop_saw_step: eligibility length mismatch"
    );
    assert_eq!(psi.len(), n, "eprop_saw_step: psi length mismatch");
    assert_eq!(ii.len(), n, "eprop_saw_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "eprop_saw_step: spikes length mismatch");
    assert_eq!(
        pre_spikes.len(),
        n,
        "eprop_saw_step: pre_spikes length mismatch"
    );

    let dt_s = f32x8::splat(dt);
    let tau_e_s = f32x8::splat(tau_e);
    let a_s = f32x8::splat(a);
    let b_s = f32x8::splat(b);
    let v_reset = f32x8::splat(-65.0);
    let u_add = f32x8::splat(8.0);
    let v_th = f32x8::splat(30.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let mut ee = load8(eligibility.as_ptr(), off);
            let ii_clamped = load8(ii.as_ptr(), off)
                .max(f32x8::splat(-100.0))
                .min(f32x8::splat(100.0));
            let pre = load8_i32(pre_spikes.as_ptr(), off);

            // SAW dynamics (same as Izhikevich)
            let v5 = vv.mul_add(c04, c5);
            let dv = v5.mul_add(vv, c140) - uu + ii_clamped;
            let du = a_s * (b_s * vv - uu);
            vv += dt_s * dv;
            uu += dt_s * du;

            let diff = (vv - v_th).abs() / tau_sg_s;
            let one_plus_diff = one + diff;
            let ps = one / (one_plus_diff * one_plus_diff);

            // Eligibility trace: exponential Euler
            let z_pre = pre.cmp_gt(zero).blend(one, zero);
            let e_decay = fast_exp_poly_simd(-dt_s / tau_e_s);
            let one_minus_e_decay = one - e_decay;
            ee = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;

            // Spike detection: v ≥ 30 → hardcoded reset
            let sm = vv.cmp_ge(v_th);
            vv = sm.blend(v_reset, vv);
            uu += sm.blend(u_add, zero);

            total_spikes += count_spikes(sm);

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(eligibility.as_mut_ptr(), off, ee);
            store8(psi.as_mut_ptr(), off, ps);
            spikes[off..off + 8].copy_from_slice(&mask_to_spikes(sm));
        }
    }

    // Scalar tail (n & 7)
    if n8 < n {
        for i in n8..n {
            let vv = v[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u[i] + ii[i].clamp(-100.0, 100.0);
            let du = a * (b * vv - u[i]);
            v[i] += dt * dv;
            u[i] += dt * du;
            psi[i] = crate::kernels::learning::surrogate_gradient(v[i], 30.0, tau_sg);
            let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
            let e_decay = (-dt / tau_e).exp();
            eligibility[i] = eligibility[i] * e_decay + psi[i] * z_pre * tau_e * (1.0 - e_decay);
            if v[i] >= 30.0 {
                spikes[i] = 1;
                total_spikes += 1;
                v[i] = -65.0;
                u[i] += 8.0;
            } else {
                spikes[i] = 0;
            }
        }
    }

    total_spikes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eprop_lif_spikes_and_eligibility() {
        let n = 8;
        let mut v = vec![-65.0f32; n];
        let mut e = vec![0.0f32; n];
        let mut psi = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut refr = vec![0.0f32; n];
        let ii = vec![30.0f32; n];
        let pre = vec![1i32; n];
        let mut total = 0usize;
        for _ in 0..200 {
            total += eprop_lif_step(
                &mut v,
                &mut e,
                &mut psi,
                &ii,
                &mut spikes,
                &pre,
                &mut refr,
                0.1,
                10.0,
                20.0,
                3.0,
                -65.0,
                -50.0,
                -70.0,
                1.0,
            );
        }
        assert!(total > 0, "LIF should fire with I=30");
        assert!(e.iter().any(|&x| x > 0.0), "eligibility should accumulate");
    }

    #[test]
    fn eprop_izh_multi_accumulates_weights() {
        let n = 8;
        let n_pre = 4;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut psi = vec![0.0f32; n];
        let mut e = vec![0.0f32; n_pre * n];
        let z_pre = vec![1i32; n_pre];
        let i_ext = vec![0.0f32; n];
        let weights = vec![0.5f32; n_pre * n];
        let mut spikes = vec![0i32; n];
        let mut total = 0usize;
        for _ in 0..100 {
            total += eprop_izhikevich_step_multi(
                &mut v,
                &mut u,
                &mut psi,
                &mut e,
                &z_pre,
                &i_ext,
                &weights,
                n_pre,
                0.5,
                20.0,
                0.02,
                0.2,
                -65.0,
                8.0,
                1.0,
                &mut spikes,
            );
        }
        assert!(total > 0, "strong weights should drive spikes");
    }
}
