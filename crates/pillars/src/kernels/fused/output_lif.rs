use crate::core::util::{load8, mask_to_spikes, store8};
use crate::math::fast_exp::fast_exp_poly_simd;

use wide::{f32x8, CmpGe, CmpGt};

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
                    i_s = w * scale10 + i_s;
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
