#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{load8, load8_mut, load8_i32, store8, spike_mask_to_i32, surrogate_gradient};

// ═══════════════════════════════════════════════════════
//  Izhikevich
// ═══════════════════════════════════════════════════════

/// Step `n` Izhikevich neurons with refractory period.
///
/// Equations:
///   dv/dt = 0.04·v² + 5·v + 140 − u + I
///   du/dt = a·(b·v − u)
///   if v ≥ 30: v = c, u += d, trace += trace_decay·(1 − trace)
///   after spike: refractory = tau_ref (no dynamics until counter reaches 0)
///
/// Uses `f32x8` SIMD for the hot arithmetic. Falls back to scalar for `n < 8`.
///
/// # Safety
/// All pointer args must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn izhikevich_step(
    v: *mut f32, u: *mut f32, trace: *mut f32,
    a: *const f32, b: *const f32, Ii: *const f32,
    dt: f32, trace_decay: f32,
    spikes: *mut i32, c: *const f32, d: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let tr_s = f32x8::splat(tau_ref);

    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut tr = load8_mut(trace, off);
        let mut refr = load8_mut(refractory, off);
        let aa = load8(a, off);
        let bb = load8(b, off);
        let ii = load8(Ii, off);
        let cc = load8(c, off);
        let dd = load8(d, off);

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii;
        let du = aa * (bb * vv - uu);
        vv = vv + refr_mask.blend(zero, dt_s * dv);  // skip if refractory
        uu = uu + refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr; // spike if v≥30 AND NOT refractory

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);  // clamp to 0

        // trace update (only non-refractory)
        let tr_s_val = tr.mul_add(td_s, one - tr);
        let tr_n_val = tr * (one - td_s);
        tr = refr_mask.blend(tr, sm.blend(tr_s_val, tr_n_val));

        store8(v, off, vv);
        store8(u, off, uu);
        store8(trace, off, tr);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u = std::slice::from_raw_parts_mut(u, n);
        let trace = std::slice::from_raw_parts_mut(trace, n);
        let a_sl = std::slice::from_raw_parts(a, n);
        let b_sl = std::slice::from_raw_parts(b, n);
        let ii = std::slice::from_raw_parts(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let c_sl = std::slice::from_raw_parts(c, n);
        let d_sl = std::slice::from_raw_parts(d, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt).max(0.0);
                spikes[i] = 0;
                continue;
            }
            let orig_v = v[i];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[i] + ii[i];
            let du = a_sl[i] * (b_sl[i] * orig_v - u[i]);
            v[i] += dt * dv;
            u[i] += dt * du;
            if v[i] >= 30.0 {
                spikes[i] = 1;
                v[i] = c_sl[i];
                u[i] += d_sl[i];
                refr_sl[i] = tau_ref;
                trace[i] += trace_decay * (1.0 - trace[i]);
            } else {
                spikes[i] = 0;
                trace[i] *= 1.0 - trace_decay;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop Izhikevich — Eligibility Propagation for Izhikevich
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop Izhikevich neurons with per-synapse eligibility traces.
/// Column-major layout: `e[i * n + j]` = eligibility for pre-synapse `i`, post-neuron `j`.
/// Same for `weights`.
///
/// Kernel computes `I_total[j] = I_ext[j] + Σ_i weights[i·n+j] · z_pre[i]`,
/// steps Izhikevich dynamics, and updates eligibility for each pre-synapse.
///
/// Uses `f32x8` SIMD for the hot path.
/// `n` = total number of post-neurons (also the column stride in column-major).
///
/// # Returns
/// Total number of output spikes across all neurons.
///
/// # Safety
/// All pointer args valid for given dimensions; `e[n_pre·n]`, `z_pre[n_pre]`,
/// `I_ext[n]`, `weights[n_pre·n]` column-major; `v[n]`, `u[n]`, `psi[n]`, `spikes[n]`.
pub(crate) unsafe fn eprop_izhikevich_step_multi(
    v: *mut f32, u: *mut f32, psi: *mut f32,
    e: *mut f32,          // [n_pre × n] column-major
    z_pre: *const i32,    // [n_pre]
    I_ext: *const f32,    // [n]
    weights: *const f32,  // [n_pre × n] column-major
    n: usize,             // n_post (also column stride)
    n_pre: usize,
    dt: f32, tau_e: f32, a: f32, b: f32, c: f32, d: f32, tau_sg: f32,
    spikes: *mut i32,
) -> usize {
    assert!(n_pre > 0);
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

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let ii = load8(I_ext, off);

        // I_total = I_ext + SCALE · Σ_i weights[i·n + j] · z_pre[i]
        // SCALE = 10.0 matches fused_hidden_izhikevich_step convention.
        // Without this scale, weight contribution is negligible vs bias.
        let scale = f32x8::splat(10.0);
        let mut I_s = ii;
        for i in 0..n_pre {
            if *z_pre.add(i) != 0 {
                let w = load8(weights, i * n + off);
                I_s = w.mul_add(scale, I_s);  // I_s += w * 10.0
            }
        }

        // Izhikevich dynamics
        // Euler: v += dt * dv, u += dt * du
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
        let du = a_s * (b_s * vv - uu);
        vv = vv + dt_s * dv;
        uu = uu + dt_s * du;

        // Surrogate gradient
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);
        uu = uu + sm.blend(d_s, zero);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(u, off, uu);
        store8(psi, off, ps);

        // Per-synapse eligibility: e_ij = e_ij * (1 - dt/τ_e) + dt * ψ_j * z_pre[i]
        let decay = one - dt_s / tau_e_s;
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            let mut e_simd = load8_mut(e, i * n + off);
            if z != 0.0 {
                e_simd = e_simd * decay + ps * f32x8::splat(z * dt);
            } else {
                e_simd = e_simd * decay;
            }
            store8(e, i * n + off, e_simd);
        }

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v_sl = std::slice::from_raw_parts_mut(v, n);
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(I_ext, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        for j in n8..n {
            let mut vv = v_sl[j];
            let mut uu = u_sl[j];

            // Compute I_total for this post-neuron (10x weight scale)
            let mut I_total = i_sl[j];
            for i in 0..n_pre {
                if *z_pre.add(i) != 0 {
                    I_total += *weights.add(i * n + j) * 10.0;
                }
            }

            // Izhikevich dynamics
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - uu + I_total;
            let du = a * (b * vv - uu);
            vv += dt * dv;
            uu += dt * du;

            psi_sl[j] = surrogate_gradient(vv, 30.0, tau_sg);

            // Per-synapse eligibility
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                let e_ptr = e.add(i * n + j);
                *e_ptr += dt * (-*e_ptr / tau_e + psi_sl[j] * z);
            }

            if vv >= 30.0 {
                sp_sl[j] = 1;
                total_spikes += 1;
                vv = c;
                uu += d;
            } else {
                sp_sl[j] = 0;
            }

            v_sl[j] = vv;
            u_sl[j] = uu;
        }
    }

    total_spikes
}

/// Step `n` E-Prop Izhikevich neurons with eligibility traces.
/// Uses `f32x8` SIMD for Izhikevich dynamics + eligibility + surrogate gradient.
///
/// Izhikevich equations:
///   dv/dt = 0.04·v² + 5·v + 140 − u + I
///   du/dt = a·(b·v − u)
///   if v ≥ 30: v = c, u += d
///
/// E-Prop: de/dt = −e/τ_e + ψ·z_pre
///   ψ = surrogate gradient = 1 / (1 + |v − 30|/τ)²
///
/// # Returns
/// Total number of output spikes across all neurons.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn eprop_izhikevich_step(
    v: *mut f32, u: *mut f32, eligibility: *mut f32, psi: *mut f32,
    Ii: *const f32, spikes: *mut i32, pre_spikes: *const i32,
    dt: f32, tau_e: f32, a: f32, b: f32, c: f32, d: f32, tau_sg: f32,
    n: usize,
) -> usize {
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

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut ee = load8_mut(eligibility, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);

        // Izhikevich dynamics
        // dv = (0.04·v + 5)·v + 140 − u + I
        // Euler: v += dt * dv
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii;
        let du = a_s * (b_s * vv - uu);
        vv = vv + dt_s * dv;
        uu = uu + dt_s * du;

        // Surrogate gradient per lane (SIMD)
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: de/dt = -e/τ_e + ψ · z_pre
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        let de = -ee / tau_e_s + ps * z_pre;
        ee = ee + de * dt_s;

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);    // v = c where spike
        uu = uu + sm.blend(d_s, zero); // u += d where spike

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(u, off, uu);
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
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let e_sl = std::slice::from_raw_parts_mut(eligibility, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(Ii, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        let pre_sl = std::slice::from_raw_parts(pre_spikes, n);
        for i in n8..n {
            let vv = v_sl[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u_sl[i] + i_sl[i];
            let du = a * (b * vv - u_sl[i]);
            v_sl[i] += dt * dv;
            u_sl[i] += dt * du;
            psi_sl[i] = surrogate_gradient(v_sl[i], 30.0, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] += dt * (-e_sl[i] / tau_e + psi_sl[i] * z_pre);
            if v_sl[i] >= 30.0 {
                sp_sl[i] = 1;
                total_spikes += 1;
                v_sl[i] = c;
                u_sl[i] += d;
            } else {
                sp_sl[i] = 0;
            }
        }
    }

    total_spikes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_spikes_with_strong_current() {
        let n = 16;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..50 {
            unsafe { izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5, spikes.as_mut_ptr(),
                c.as_ptr(), d.as_ptr(), refractory.as_mut_ptr(), 3.0, n); }
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "Expected spikes with I=30");
    }

    #[test]
    fn izhikevich_no_spike_inhibitory() {
        let n = 8;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let mut I = vec![-20.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..50 {
            unsafe { izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5, spikes.as_mut_ptr(),
                c.as_ptr(), d.as_ptr(), refractory.as_mut_ptr(), 3.0, n); }
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert_eq!(total_spikes, 0, "No spikes with I=-20");
    }

    #[test]
    fn izhikevich_non_aligned_n() {
        let n = 13;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let mut I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;
        for _ in 0..50 {
            unsafe { izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5, spikes.as_mut_ptr(),
                c.as_ptr(), d.as_ptr(), refractory.as_mut_ptr(), 3.0, n); }
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "Spikes with I=30, n=13");
    }

}
