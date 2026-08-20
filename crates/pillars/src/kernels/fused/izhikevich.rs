use crate::core::util::{load8, mask_to_spikes, store8};

use wide::{f32x8, CmpGe, CmpGt};

// ═══════════════════════════════════════════════════════════
//  Fused Izhikevich with weight accumulation (hidden layer)
// ═══════════════════════════════════════════════════════════

/// Fused hidden layer: accumulate input spikes → Izhikevich step in one pass.
///
/// ```text
/// I[j..j+8] += Σ_{i active} weights[i·nh + j..j+8] · z_pre[i] · 10.0
/// Izhikevich single-step Forward Euler (v_th = 30, reset c/d, trace STDP)
/// ```
///
/// Uses the caller-supplied `active_buf` (pre-allocated) to iterate only the
/// active pre-synapses instead of all `ni` inputs (sparse optimization).
///
/// # Panics
/// On slice-length mismatch; `v.len()` (nh) must be a multiple of 8.
pub fn fused_hidden_izhikevich_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    weights: &[f32],
    input_spikes: &[i32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(
        u.len(),
        nh,
        "fused_hidden_izhikevich_step: u length mismatch"
    );
    assert_eq!(
        trace.len(),
        nh,
        "fused_hidden_izhikevich_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "fused_hidden_izhikevich_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "fused_hidden_izhikevich_step: refractory length mismatch"
    );
    assert_eq!(
        a.len(),
        nh,
        "fused_hidden_izhikevich_step: a length mismatch"
    );
    assert_eq!(
        b.len(),
        nh,
        "fused_hidden_izhikevich_step: b length mismatch"
    );
    assert_eq!(
        c.len(),
        nh,
        "fused_hidden_izhikevich_step: c length mismatch"
    );
    assert_eq!(
        d.len(),
        nh,
        "fused_hidden_izhikevich_step: d length mismatch"
    );
    assert_eq!(
        weights.len(),
        ni * nh,
        "fused_hidden_izhikevich_step: weights length mismatch (expected {}*{}={})",
        ni,
        nh,
        ni * nh
    );
    assert!(
        input_spikes.len() >= ni,
        "fused_hidden_izhikevich_step: input_spikes length {} < ni {ni}",
        input_spikes.len()
    );
    assert!(
        nh.is_multiple_of(8),
        "fused_hidden_izhikevich_step: nh must be multiple of 8, got {nh}"
    );

    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let scale10 = f32x8::splat(10.0);
    let tr_s = f32x8::splat(tau_ref);

    // Active pre-synapse list (sparse optimization) — reuse caller buffer.
    active_buf.clear();
    for i in 0..ni {
        if input_spikes[i] != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    let n8 = nh & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut vv = load8(v.as_ptr(), off);
            let mut uu = load8(u.as_ptr(), off);
            let mut tr = load8(trace.as_ptr(), off);
            let mut refr = load8(refractory.as_ptr(), off);
            let aa = load8(a.as_ptr(), off);
            let bb = load8(b.as_ptr(), off);
            let cc = load8(c.as_ptr(), off);
            let dd = load8(d.as_ptr(), off);

            // Accumulate currents from active pre-synapses only
            let mut i_s = zero;
            for &i in active {
                let w = load8(weights.as_ptr(), i * nh + off);
                i_s = w * scale10 + i_s; // I += w · 10
            }

            let refr_mask = refr.cmp_gt(zero);

            let v5 = vv * c04 + c5;
            let dv = v5 * vv + c140 - uu + i_s;
            let du = aa * (bb * vv - uu);
            vv += refr_mask.blend(zero, dt_s * dv);
            uu += refr_mask.blend(zero, dt_s * du);

            let not_refr = !refr_mask;
            let sm = vv.cmp_ge(v_th) & not_refr;

            vv = refr_mask.blend(vv, sm.blend(cc, vv));
            uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

            refr -= dt_s;
            refr = sm.blend(tr_s, refr);
            refr = refr.max(zero);

            // STDP trace: on spike tr = tr·td + (1−tr), else tr·(1−td)
            let tr_s_val = tr * td_s + (one - tr);
            let tr_n_val = tr * (one - td_s);
            tr = refr_mask.blend(tr, sm.blend(tr_s_val, tr_n_val));

            store8(v.as_mut_ptr(), off, vv);
            store8(u.as_mut_ptr(), off, uu);
            store8(trace.as_mut_ptr(), off, tr);
            store8(refractory.as_mut_ptr(), off, refr);

            let sp = mask_to_spikes(sm);
            spikes[off..off + 8].copy_from_slice(&sp);
        }
    }

    // Scalar tail (empty since nh is padded to ×8 — kept for parity).
    if n8 < nh {
        for j in n8..nh {
            let mut i_val = 0.0f32;
            for &i in active {
                i_val += weights[i * nh + j] * 10.0;
            }
            if refractory[j] > 0.0 {
                refractory[j] = (refractory[j] - dt).max(0.0);
                spikes[j] = 0;
                continue;
            }
            let orig_v = v[j];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[j] + i_val;
            let du = a[j] * (b[j] * orig_v - u[j]);
            v[j] += dt * dv;
            u[j] += dt * du;
            if v[j] >= 30.0 {
                spikes[j] = 1;
                v[j] = c[j];
                u[j] += d[j];
                refractory[j] = tau_ref;
                trace[j] += trace_decay * (1.0 - trace[j]);
            } else {
                spikes[j] = 0;
                trace[j] *= 1.0 - trace_decay;
            }
        }
    }
}
