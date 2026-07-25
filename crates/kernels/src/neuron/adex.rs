#![allow(non_snake_case)]
use super::{fast_exp_simd, load8, load8_mut, spike_mask_to_i32, store8, surrogate_gradient};
use wide::{f32x8, CmpGe, CmpGt, CmpNe};

// ═══════════════════════════════════════════════════════
//  AdEx (Adaptive Exponential Integrate-and-Fire)
// ═══════════════════════════════════════════════════════

/// Step `n` AdEx neurons with refractory period. Uses `f32x8` SIMD with fast exponential.
///
/// Params: dt, tau_m, tau_w, v_rest, v_thresh, v_reset, delta_t, a, b (9 × n)
///
/// The exponential term uses a SIMD fast-exp approximation (~1e-4 rel. error)
/// instead of per-element `f32::exp()` — ~8× faster.
///
/// # Safety
/// `v[n]`, `u[n]`, `Ii[n]`, `spikes[n]`, `params[9·n]`, `refractory[n]` must be valid.
pub(crate) unsafe fn adex_step(
    v: *mut f32,
    u: *mut f32,
    Ii: *mut f32,
    spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32,
    tau_ref: f32,
    n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 9);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut refr = load8_mut(refractory, off);
        // Fused Ii clamp — предотвращает saturation cascade от oversized weights
        let ii = load8_mut(Ii, off)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let v_rest = load8(params.wrapping_add(3 * n), off);
        let v_th = load8(params.wrapping_add(4 * n), off);
        let v_reset = load8(params.wrapping_add(5 * n), off);
        let delta_t = load8(params.wrapping_add(6 * n), off);
        let aa = load8(params.wrapping_add(7 * n), off);
        let bb = load8(params.wrapping_add(8 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);
        let tau_w = load8(params.wrapping_add(2 * n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // Exponential term: Δ·exp((v − v_T) / Δ) — only for non-refractory
        let exp_arg = ((vv - v_th) / delta_t.max(f32x8::splat(1.0)))
            .max(f32x8::splat(-10.0))
            .min(f32x8::splat(10.0));
        let exp_val = fast_exp_simd(exp_arg);
        let exp_term = delta_t * exp_val;
        let v_gt_mask = vv.cmp_gt(v_th - f32x8::splat(20.0));
        let exp_term = v_gt_mask.blend(exp_term, zero);
        // Zero out exp_term during refractory
        let exp_term = refr_mask.blend(zero, exp_term);

        // Membrane potential (skip if refractory)
        // Save v(t) for use in dw — parallel Forward Euler (Brette & Gerstner 2005)
        let vv_old = vv;
        let mask_tm = tau_m.cmp_gt(zero);
        let inv_tm = mask_tm.blend(one / tau_m, zero);
        let dv = (v_rest - vv_old + exp_term - uu + ii) * inv_tm * dt_s;
        vv += refr_mask.blend(zero, mask_tm.blend(dv, zero));

        // Adaptation current (uses vv_old = v(t), still decays during refractory)
        let mask_tw = tau_w.cmp_gt(zero);
        let inv_tw = mask_tw.blend(one / tau_w, zero);
        let dw = (aa * (vv_old - v_rest) - uu) * inv_tw * dt_s;
        uu += mask_tw.blend(dw, zero);

        // Spike detection (only non-refractory)
        let nan_mask = vv.cmp_ne(vv);
        let not_refr = !refr_mask;
        let sm = (vv.cmp_ge(v_th) | nan_mask) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));
        uu += sm.blend(bb, zero);

        // Decrement refractory for all, set to tau_ref on spike
        refr -= dt_s; // Always decrement
        refr = sm.blend(tr_s, refr); // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(u, off, uu);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0];
        *sp8.add(1) = sp_i[1];
        *sp8.add(2) = sp_i[2];
        *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4];
        *sp8.add(5) = sp_i[5];
        *sp8.add(6) = sp_i[6];
        *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let ii = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt_val).max(0.0);
                // w still decays during refractory
                let tau_w = p[2 * n + i];
                if tau_w > 0.0 {
                    let aa_p = p[7 * n + i];
                    let v_rest_p = p[3 * n + i];
                    u_sl[i] += (aa_p * (v[i] - v_rest_p) - u_sl[i]) / tau_w * dt_val;
                }
                spikes[i] = 0;
                continue;
            }
            let tau_m = p[n + i];
            let tau_w = p[2 * n + i];
            let v_rest = p[3 * n + i];
            let v_thresh = p[4 * n + i];
            let v_reset = p[5 * n + i];
            let delta_t = p[6 * n + i];
            let aa = p[7 * n + i];
            let bb = p[8 * n + i];
            let exp_term = if v[i] > v_thresh - 20.0 {
                delta_t * ((v[i] - v_thresh) / delta_t.max(1.0)).exp()
            } else {
                0.0
            };
            let i_clamped = ii[i].clamp(-100.0, 100.0);
            // Save v(t) before update — parallel Forward Euler
            let v_old = v[i];
            if tau_m > 0.0 {
                v[i] += (v_rest - v_old + exp_term - u_sl[i] + i_clamped) / tau_m * dt_val;
            }
            if tau_w > 0.0 {
                u_sl[i] += (aa * (v_old - v_rest) - u_sl[i]) / tau_w * dt_val; // v_old = v(t)
            }
            if v[i] >= v_thresh || v[i].is_nan() {
                spikes[i] = 1;
                v[i] = v_reset;
                u_sl[i] += bb;
                refr_sl[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop AdEx Multi — Adaptive Exponential with per-synapse eligibility
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop AdEx neurons with per-synapse eligibility traces.
/// Column-major layout: `e[i * n + j]` = eligibility for pre-synapse `i`, post-neuron `j`.
///
/// AdEx equations:
///   dv/dt = (v_rest - v + Δ_T·exp((v - v_T)/Δ_T) - w + I) / τ_m
///   dw/dt = (a·(v - v_rest) - w) / τ_w
///   v ≥ v_T → v = v_reset, w += b
///
/// E-Prop eligibility (Exponential Euler):
///   e(t+dt) = e·exp(-dt/τ_e) + ψ·z_pre·τ_e·(1 - exp(-dt/τ_e))
///
/// Uses `f32x8` SIMD with fast exponential approximation.
///
/// # Safety
/// All pointer args valid for given dimensions; `e[n_pre·n]`, `z_pre[n_pre]`,
/// `I_ext[n]`, `weights[n_pre·n]`, `v[n]`, `w[n]`, `psi[n]`, `spikes[n]`.
pub(crate) unsafe fn eprop_adex_step_multi(
    v: *mut f32,
    w: *mut f32,
    psi: *mut f32,
    e: *mut f32,         // [n_pre × n] column-major
    z_pre: *const i32,   // [n_pre]
    I_ext: *const f32,   // [n]
    weights: *const f32, // [n_pre × n] column-major
    n: usize,            // n_post (column stride)
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
    spikes: *mut i32,
) -> usize {
    assert!(n_pre > 0);
    assert!(tau_m > 0.0 && tau_w > 0.0);
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

    // Precompute e_decay — constant across all SIMD blocks
    let e_decay = fast_exp_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut ww = load8_mut(w, off);
        // Fused Ii clamp — предотвращает saturation cascade от oversized weights
        let ii = load8(I_ext, off)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));

        // Compute I_total = I_ext + Σ_i weights[i·n + j] · z_pre[i]
        let mut I_s = ii;
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            if z != 0.0 {
                let wgt = load8(weights, i * n + off);
                I_s = wgt.mul_add(f32x8::splat(z), I_s);
            }
        }

        // AdEx dynamics
        // dv = (v_rest - v + Δ·exp((v - v_T)/Δ) - w + I) / τ_m · dt
        // Clamp exp arg to avoid overflow
        let exp_arg = ((vv - v_th_s) / delta_t_s.max(f32x8::splat(1.0)))
            .max(f32x8::splat(-10.0))
            .min(f32x8::splat(10.0));
        let exp_val = fast_exp_simd(exp_arg);
        let exp_term = delta_t_s * exp_val;
        // Zero out where v ≤ v_T - 20
        let v_gt_mask = vv.cmp_gt(v_th_s - f32x8::splat(20.0));
        let exp_term = v_gt_mask.blend(exp_term, zero);

        let dv = (v_rest_s - vv + exp_term - ww + I_s) / tau_m_s * dt_s;
        let dw = (a_s * (vv - v_rest_s) - ww) / tau_w_s * dt_s;
        vv += dv;
        ww += dw;

        // Surrogate gradient
        let diff = (vv - v_th_s).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Spike detection: v ≥ v_thresh OR NaN
        let nan_mask = vv.cmp_ne(vv);
        let sm = vv.cmp_ge(v_th_s) | nan_mask;
        vv = sm.blend(v_reset_s, vv);
        ww += sm.blend(b_s, zero);

        // Count spikes
        for &val in sm.to_array().iter() {
            if val.to_bits() != 0 {
                total_spikes += 1;
            }
        }

        store8(v, off, vv);
        store8(w, off, ww);
        store8(psi, off, ps);

        // Per-synapse eligibility: exponential Euler
        // e_ij(t+dt) = e_ij·e_decay + ψ_j·z_pre·τ_e·(1 - e_decay)
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            let mut e_simd = load8_mut(e, i * n + off);
            if z != 0.0 {
                e_simd = e_simd * e_decay + ps * f32x8::splat(z) * tau_e_s * one_minus_e_decay;
            } else {
                e_simd *= e_decay;
            }
            store8(e, i * n + off, e_simd);
        }

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0];
        *sp8.add(1) = sp_i[1];
        *sp8.add(2) = sp_i[2];
        *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4];
        *sp8.add(5) = sp_i[5];
        *sp8.add(6) = sp_i[6];
        *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v_sl = std::slice::from_raw_parts_mut(v, n);
        let w_sl = std::slice::from_raw_parts_mut(w, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(I_ext, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for j in n8..n {
            let mut vv = v_sl[j];
            let mut ww = w_sl[j];

            // Compute I_total
            let mut I_total = i_sl[j].clamp(-100.0, 100.0);
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                if z != 0.0 {
                    I_total += *weights.add(i * n + j) * z;
                }
            }

            // AdEx dynamics
            let exp_term = if vv > v_thresh - 20.0 {
                delta_t * ((vv - v_thresh) / delta_t.max(1.0)).exp()
            } else {
                0.0
            };
            let dv = (v_rest - vv + exp_term - ww + I_total) / tau_m * dt;
            let dw = (a * (vv - v_rest) - ww) / tau_w * dt;
            vv += dv;
            ww += dw;

            psi_sl[j] = surrogate_gradient(vv, v_thresh, tau_sg);

            // Per-synapse eligibility (exponential Euler)
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                let e_ptr = e.add(i * n + j);
                *e_ptr = *e_ptr * e_decay_s + psi_sl[j] * z * tau_e * one_minus_e_decay_s;
            }

            if vv >= v_thresh || vv.is_nan() {
                sp_sl[j] = 1;
                total_spikes += 1;
                vv = v_reset;
                ww += b;
            } else {
                sp_sl[j] = 0;
            }

            v_sl[j] = vv;
            w_sl[j] = ww;
        }
    }

    total_spikes
}
