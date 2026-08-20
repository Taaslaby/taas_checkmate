use crate::core::util::{load8, mask_to_spikes, store8};
use crate::kernels::RNG_SCALE_24;

use wide::{f32x8, CmpGe, CmpGt};

// ═══════════════════════════════════════════════════════════
//  Poisson-fused Izhikevich — RNG → weights → dynamics in one pass
// ═══════════════════════════════════════════════════════════

/// Fused kernel: Poisson spike generation + weight accumulation + Izhikevich
/// in a single SIMD pass (eliminates the encoder step entirely).
///
/// ```text
/// for each input i:  Poisson spike via LCG (per-neuron RNG state)
/// for each 8-block:  I += Σ_{i active} weights[i·nh + j..] · 10
///                    Izhikevich single-step Forward Euler (as above)
/// ```
///
/// `input_spikes_out` receives the generated Poisson flags (for downstream
/// eligibility traces). ~20% faster than separate encoder + fused hidden step.
///
/// # Panics
/// On slice-length mismatch; `v.len()` (nh) must be a multiple of 8.
pub fn poisson_fused_hidden_step(
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
    rng_states: &mut [u32],
    rates: &[f32],
    input_spikes_out: &mut [i32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    ni: usize,
    active_buf: &mut Vec<usize>,
) {
    let nh = v.len();
    assert_eq!(u.len(), nh, "poisson_fused_hidden_step: u length mismatch");
    assert_eq!(
        trace.len(),
        nh,
        "poisson_fused_hidden_step: trace length mismatch"
    );
    assert_eq!(
        spikes.len(),
        nh,
        "poisson_fused_hidden_step: spikes length mismatch"
    );
    assert_eq!(
        refractory.len(),
        nh,
        "poisson_fused_hidden_step: refractory length mismatch"
    );
    assert_eq!(a.len(), nh, "poisson_fused_hidden_step: a length mismatch");
    assert_eq!(b.len(), nh, "poisson_fused_hidden_step: b length mismatch");
    assert_eq!(c.len(), nh, "poisson_fused_hidden_step: c length mismatch");
    assert_eq!(d.len(), nh, "poisson_fused_hidden_step: d length mismatch");
    assert_eq!(
        weights.len(),
        ni * nh,
        "poisson_fused_hidden_step: weights length mismatch (expected {}*{}={})",
        ni,
        nh,
        ni * nh
    );
    assert!(
        rng_states.len() >= ni,
        "poisson_fused_hidden_step: rng_states length {} < ni {ni}",
        rng_states.len()
    );
    assert!(
        rates.len() >= ni,
        "poisson_fused_hidden_step: rates length {} < ni {ni}",
        rates.len()
    );
    assert!(
        input_spikes_out.len() >= ni,
        "poisson_fused_hidden_step: input_spikes_out length {} < ni {ni}",
        input_spikes_out.len()
    );
    assert!(
        nh.is_multiple_of(8),
        "poisson_fused_hidden_step: nh must be multiple of 8, got {nh}"
    );

    let dt_sec = dt / 1000.0; // ms → sec for Poisson probability
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

    // Step 1: Poisson spikes for all inputs (1 LCG each) — bit-identical LCG.
    for i in 0..ni {
        let state = rng_states[i].wrapping_mul(1664525).wrapping_add(1013904223);
        rng_states[i] = state;
        let r = (state >> 8) as f32 * RNG_SCALE_24;
        let p = (rates[i] * dt_sec).clamp(0.0, 1.0);
        input_spikes_out[i] = if r < p { 1 } else { 0 };
    }

    // Step 2: active pre-synapse list (sparse) — reuse caller buffer.
    active_buf.clear();
    for i in 0..ni {
        if input_spikes_out[i] != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    // Step 3: Fused Izhikevich — ⚠️ ДУБЛИКАТ блока из fused_hidden_izhikevich_step
    // (дословная верность легаси). Правка динамики — в ОБА места синхронно.
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

            let mut i_s = zero;
            for &i in active {
                let w = load8(weights.as_ptr(), i * nh + off);
                i_s = w * scale10 + i_s;
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
