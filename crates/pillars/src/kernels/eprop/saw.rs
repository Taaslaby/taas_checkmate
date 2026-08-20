use super::count_spikes;

use crate::core::util::{load8, load8_i32, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;
use wide::{f32x8, CmpGe, CmpGt};

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
            let v5 = vv * c04 + c5;
            let dv = v5 * vv + c140 - uu + ii_clamped;
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
