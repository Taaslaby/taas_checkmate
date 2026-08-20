use super::count_spikes;

use crate::core::util::{load8, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;
use wide::{f32x8, CmpGe, CmpGt, CmpNe};

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
                    i_s = wgt * f32x8::splat(z) + i_s;
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
