use super::count_spikes;

use crate::core::util::{load8, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;
use wide::{f32x8, CmpGe};

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
                    i_s = w * scale + i_s;
                }
            }

            let v5 = vv * c04 + c5;
            let dv = v5 * vv + c140 - uu + i_s;
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
