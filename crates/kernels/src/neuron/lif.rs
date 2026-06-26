#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{load8, load8_mut, load8_i32, store8, spike_mask_to_i32, surrogate_gradient};

// ═══════════════════════════════════════════════════════
//  LIF (Leaky Integrate-and-Fire)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset (5 × n)
///
/// # Safety
/// `v[n]`, `Ii[n]`, `spikes[n]`, `params[5·n]`, `refractory[n]` must be valid.
pub(crate) unsafe fn lif_step(
    v: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 5);
    let dt_val = p[0];  // dt is uniform across all neurons
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(one / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tau.blend(dv, zero));

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                continue;
            }
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if v[i_idx] >= v_thresh {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  LIF-AT (LIF with Adaptive Threshold)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF-AT neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset, tau_theta, beta (7 × n)
///
/// # Safety
/// `v[n]`, `theta[n]`, `Ii[n]`, `spikes[n]`, `params[7·n]`, `refractory[n]` valid.
pub(crate) unsafe fn lif_at_step(
    v: *mut f32, theta: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 7);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut th = load8_mut(theta, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_th = load8(params.wrapping_add(5 * n), off);
        let beta = load8(params.wrapping_add(6 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory — theta still decays)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(one / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tau.blend(dv, zero));

        // Adaptive threshold decay (always decays, even during refractory)
        let mask_tt = tau_th.cmp_gt(zero);
        let dth = mask_tt.blend(dt_s * (-th / tau_th), zero);
        th = th + dth;

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th + th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));
        th = th + sm.blend(beta, zero);

        // Decrement refractory for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(theta, off, th);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let theta = std::slice::from_raw_parts_mut(theta, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                // Theta still decays during refractory
                let tau_theta = p[5 * n + i_idx];
                if tau_theta > 0.0 {
                    theta[i_idx] += dt_val * (-theta[i_idx] / tau_theta);
                }
                continue;
            }
            let dt_val = p[i_idx];
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            let tau_theta = p[5 * n + i_idx];
            let beta = p[6 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if tau_theta > 0.0 {
                theta[i_idx] += dt_val * (-theta[i_idx] / tau_theta);
            }
            if v[i_idx] >= v_thresh + theta[i_idx] {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                theta[i_idx] += beta;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop LIF (Eligibility Propagation) — SIMD kernel
// ═══════════════════════════════════════════════════════



/// Step `n` E-Prop LIF neurons with eligibility traces.
/// Uses `f32x8` SIMD for LIF dynamics + eligibility + surrogate gradient.
///
/// Params: none (all scalar params passed directly)
/// Updates: v, theta, eligibility, psi, spikes
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn eprop_lif_step(
    v: *mut f32, theta: *mut f32, eligibility: *mut f32, psi: *mut f32,
    Ii: *const f32, spikes: *mut i32, pre_spikes: *const i32,
    dt: f32, tau_m: f32, tau_e: f32,
    v_rest: f32, v_thresh: f32, v_reset: f32, tau_sg: f32,
    n: usize,
) -> usize {
    let dt_s = f32x8::splat(dt);
    let tau_m_s = f32x8::splat(tau_m);
    let tau_e_s = f32x8::splat(tau_e);
    let v_rest_s = f32x8::splat(v_rest);
    let v_thresh_s = f32x8::splat(v_thresh);
    let v_reset_s = f32x8::splat(v_reset);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let th = load8(theta, off);
        let mut ee = load8_mut(eligibility, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);

        // LIF dynamics: dv/dt = (v_rest - v + I) / tau_m
        let inv_tau = one / tau_m_s;
        let dv = (v_rest_s - vv + ii) * inv_tau * dt_s;
        vv = vv + dv;

        // Surrogate gradient per lane (SIMD)
        let diff = (vv - v_thresh_s).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: de/dt = -e/τ_e + ψ · z_pre
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        let de = -ee / tau_e_s + ps * z_pre;
        ee = ee + de * dt_s;

        // Spike detection: v ≥ v_thresh + theta
        let sm = vv.cmp_ge(v_thresh_s + th);
        vv = sm.blend(v_reset_s, vv);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(eligibility, off, ee);
        store8(psi, off, ps);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v_sl = std::slice::from_raw_parts_mut(v, n);
        let theta_sl = std::slice::from_raw_parts_mut(theta, n);
        let e_sl = std::slice::from_raw_parts_mut(eligibility, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(Ii, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        let pre_sl = std::slice::from_raw_parts(pre_spikes, n);
        for i in n8..n {
            let dv = (v_rest - v_sl[i] + i_sl[i]) / tau_m * dt;
            v_sl[i] += dv;
            psi_sl[i] = surrogate_gradient(v_sl[i], v_thresh, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] += dt * (-e_sl[i] / tau_e + psi_sl[i] * z_pre);
            if v_sl[i] >= v_thresh + theta_sl[i] {
                sp_sl[i] = 1;
                total_spikes += 1;
                v_sl[i] = v_reset;
            } else {
                sp_sl[i] = 0;
            }
        }
    }

    total_spikes
}
