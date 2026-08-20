use super::count_spikes;

use crate::core::util::{load8, load8_i32, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;
use wide::{f32x8, CmpGe, CmpGt};

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

            let v5 = vv * c04 + c5;
            let dv = v5 * vv + c140 - uu + ii_v;
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
