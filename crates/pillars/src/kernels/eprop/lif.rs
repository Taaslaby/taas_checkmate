use super::count_spikes;

use crate::core::util::{load8, load8_i32, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;
use wide::{f32x8, CmpGe, CmpGt};

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
