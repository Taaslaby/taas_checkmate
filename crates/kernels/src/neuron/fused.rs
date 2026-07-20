#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{fast_exp_simd, load8, load8_mut, store8, spike_mask_to_i32, RNG_SCALE_24};

// ═══════════════════════════════════════════════════════
//  Fused Izhikevich with weight accumulation (SIMD)
// ═══════════════════════════════════════════════════════

/// Fused kernel: accumulate input weights → Izhikevich step in one SIMD pass.
/// Uses active pre-synapse list for sparse inputs (~5% firing rate).
/// Eliminates the intermediate I_h buffer and memory round-trip.
///
/// For each block of 8 hidden neurons:
///   I[j..j+8] += Σ_i weights[i·nh + j..j+8] · z_pre[i] · 10.0
///   Izhikevich_step(j)
///
/// All state updates happen in-register — no scalar accumulation loop,
/// no intermediate buffer. ~3x faster than separate weight loop + izhikevich.
///
/// # Sparsity optimization
/// Pre-builds a list of active (spiking) pre-synaptic indices. Instead of
/// iterating ALL `ni` inputs and checking `if zi != 0`, we iterate only
/// the active set. For 5% firing rate with ni=1000, this reduces loop
/// iterations from 1000 to 50 per SIMD block.
///
/// # Safety
/// All pointers must be valid for the given dimensions. `nh` must be padded
/// to a multiple of 8. `v[nh]`, `u[nh]`, `trace[nh]`, `spikes[nh]`, `refractory[nh]`,
/// `a[nh]`, `b[nh]`, `c[nh]`, `d[nh]` valid; `weights[ni·nh]` column-major; `input_spikes[ni]`.
pub(crate) unsafe fn fused_hidden_izhikevich_step(
    v: *mut f32, u: *mut f32, trace: *mut f32, spikes: *mut i32, refractory: *mut f32,
    a: *const f32, b: *const f32, c: *const f32, d: *const f32,
    weights: *const f32,
    input_spikes: *const i32,
    dt: f32, trace_decay: f32, tau_ref: f32,
    ni: usize, nh: usize,
    active_buf: &mut Vec<usize>,    // pre-allocated, cleared on entry
) {
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

    // Build active pre-synapse list (sparse optimization) — reuse pre-allocated buffer.
    // Caller provides &mut active_buf to avoid per-step Vec::with_capacity allocation.
    active_buf.clear();
    for i in 0..ni {
        if *input_spikes.add(i) != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    let n8 = nh & !7;
    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut tr = load8_mut(trace, off);
        let mut refr = load8_mut(refractory, off);
        let aa = load8(a, off);
        let bb = load8(b, off);
        let cc = load8(c, off);
        let dd = load8(d, off);

        // Accumulate currents from active pre-synapses only
        let mut I_s = zero;
        for &i in active {
            let w = load8(weights, i * nh + off);
            I_s = w.mul_add(scale10, I_s);  // I_s += w * 10.0
        }

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
        let du = aa * (bb * vv - uu);
        vv += refr_mask.blend(zero, dt_s * dv);
        uu += refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

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
    }        // Scalar tail (should be empty since nh is padded to ×8)
        if n8 < nh {
            let v = std::slice::from_raw_parts_mut(v, nh);
            let u = std::slice::from_raw_parts_mut(u, nh);
            let trace = std::slice::from_raw_parts_mut(trace, nh);
            let spikes = std::slice::from_raw_parts_mut(spikes, nh);
            let refr_sl = std::slice::from_raw_parts_mut(refractory, nh);
            let a_sl = std::slice::from_raw_parts(a, nh);
            let b_sl = std::slice::from_raw_parts(b, nh);
            let c_sl = std::slice::from_raw_parts(c, nh);
            let d_sl = std::slice::from_raw_parts(d, nh);
            for j in n8..nh {
                // Accumulate current from active pre-synapses only
                let mut I_val = 0.0f32;
                for &i in active {
                    I_val += *weights.add(i * nh + j) * 10.0;
                }

                if refr_sl[j] > 0.0 {
                    refr_sl[j] = (refr_sl[j] - dt).max(0.0);
                    spikes[j] = 0;
                    continue;
                }
                let orig_v = v[j];
                let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[j] + I_val;
                let du = a_sl[j] * (b_sl[j] * orig_v - u[j]);
                v[j] += dt * dv;
                u[j] += dt * du;
                if v[j] >= 30.0 {
                    spikes[j] = 1;
                    v[j] = c_sl[j];
                    u[j] += d_sl[j];
                    refr_sl[j] = tau_ref;
                    trace[j] += trace_decay * (1.0 - trace[j]);
                } else {
                    spikes[j] = 0;
                    trace[j] *= 1.0 - trace_decay;
                }
            }
        }
    }

// ═══════════════════════════════════════════════════════
//  Fused LIF with weight accumulation (SIMD)
// ═══════════════════════════════════════════════════════

/// Fused kernel: accumulate hidden spikes → LIF step in one SIMD pass.
/// Eliminates the intermediate I_o buffer.
///
/// For each block of 8 output neurons:
///   I[k..k+8] += Σ_j weights[j·na + k..k+8] · z_hidden[j] · 10.0
///   LIF_step(k)
///
/// # Safety
/// All pointers must be valid. `na` must be padded to ×8.
/// `v[na]`, `spikes[na]`, `refractory[na]`, `tau_m[na]`, `v_rest[na]`,
/// `v_th[na]`, `v_reset[na]` valid; `weights[nh·na]` column-major; `hidden_spikes[nh]`.
pub(crate) unsafe fn fused_output_lif_step(
    v: *mut f32, spikes: *mut i32, refractory: *mut f32,
    tau_m: *const f32, v_rest: *const f32, v_th: *const f32, v_reset: *const f32,
    weights: *const f32,
    hidden_spikes: *const i32,
    dt: f32, tau_ref: f32,
    nh: usize, na: usize,
) {
    let dt_s = f32x8::splat(dt);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let scale10 = f32x8::splat(10.0);

    let n8 = na & !7;
    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut refr = load8_mut(refractory, off);
        let v_rest_p = load8(v_rest, off);
        let v_th_p = load8(v_th, off);
        let v_reset_p = load8(v_reset, off);
        let tau_m_p = load8(tau_m, off);

        // Accumulate current from all hidden spikes directly in SIMD register
        let mut I_s = zero;
        for j in 0..nh {
            let zj = *hidden_spikes.add(j);
            if zj != 0 {
                let w = load8(weights, j * na + off);
                I_s = w.mul_add(scale10, I_s);  // I_s += w * 10.0
            }
        }

        // LIF dynamics with refractory — Exponential Euler
        // v = v_rest + (v - v_rest)·exp(-dt/τ) + I·(1 - exp(-dt/τ))
        let refr_mask = refr.cmp_gt(zero);
        let mask_tau = tau_m_p.cmp_gt(zero);
        let decay = mask_tau.blend(fast_exp_simd(-dt_s / tau_m_p), zero);
        let one_minus_decay = one - decay;
        let new_v = v_rest_p + (vv - v_rest_p) * decay + I_s * one_minus_decay;
        vv = refr_mask.blend(vv, mask_tau.blend(new_v, vv));

        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th_p) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset_p, vv));

        // Decrement refractory for all, set to tau_ref on spike
        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (should be empty since na is padded to ×8)
    if n8 < na {
        let v = std::slice::from_raw_parts_mut(v, na);
        let spikes = std::slice::from_raw_parts_mut(spikes, na);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, na);
        let tau_m_sl = std::slice::from_raw_parts(tau_m, na);
        let v_rest_sl = std::slice::from_raw_parts(v_rest, na);
        let v_th_sl = std::slice::from_raw_parts(v_th, na);
        let v_reset_sl = std::slice::from_raw_parts(v_reset, na);
        for k in n8..na {
            // Accumulate current (scalar)
            let mut I_val = 0.0f32;
            for j in 0..nh {
                let zj = *hidden_spikes.add(j);
                if zj != 0 {
                    I_val += *weights.add(j * na + k) * 10.0;
                }
            }

            if refr_sl[k] > 0.0 {
                refr_sl[k] = (refr_sl[k] - dt).max(0.0);
                spikes[k] = 0;
                continue;
            }
            let tau_m = tau_m_sl[k];
            let v_rest = v_rest_sl[k];
            let v_thresh = v_th_sl[k];
            let v_reset = v_reset_sl[k];
            if tau_m > 0.0 {
                let decay = (-dt / tau_m).exp();
                v[k] = v_rest + (v[k] - v_rest) * decay + I_val * (1.0 - decay);
            }
            if v[k] >= v_thresh {
                spikes[k] = 1;
                v[k] = v_reset;
                refr_sl[k] = tau_ref;
            } else {
                spikes[k] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Poisson-Fused Izhikevich — RNG → weights → dynamics in one pass
// ═══════════════════════════════════════════════════════

/// Fused kernel: Poisson spike generation + weight accumulation + Izhikevich
/// in a single SIMD pass. Eliminates the encoder step entirely.
///
/// For each input neuron i:
///   1. Generate Poisson spike via LCG (per-neuron RNG state)
///   2. Write spike flag to input_spikes_out[i]
///
/// For each SIMD block of 8 hidden neurons:
///   3. Accumulate weights from active inputs into SIMD register
///   4. Izhikevich step with refractory (same as fused_hidden_izhikevich_step)
///
/// All state updates happen in-register. No intermediate buffers.
/// ~20% faster than separate encoder.step() + fused_hidden_izhikevich_step().
///
/// # Safety
/// All pointers valid for given dims. `v[nh]`, `u[nh]`, `trace[nh]`, `spikes[nh]`,
/// `refractory[nh]`, `a[nh]`, `b[nh]`, `c[nh]`, `d[nh]` valid;
/// `weights[ni·nh]` column-major; `rng_states[ni]`, `rates[ni]`, `input_spikes_out[ni]`.
pub(crate) unsafe fn poisson_fused_hidden_step(
    v: *mut f32, u: *mut f32, trace: *mut f32, spikes: *mut i32, refractory: *mut f32,
    a: *const f32, b: *const f32, c: *const f32, d: *const f32,
    weights: *const f32,
    rng_states: *mut u32,      // [ni] per-input-neuron LCG states
    rates: *const f32,         // [ni] Poisson rates in Hz
    input_spikes_out: *mut i32, // [ni] — spike flags for eligibility
    dt: f32,                    // ms (Poisson uses dt/1000 for probability)
    trace_decay: f32, tau_ref: f32,
    ni: usize, nh: usize,
    active_buf: &mut Vec<usize>,    // pre-allocated, cleared on entry
) {
    let dt_sec = dt / 1000.0;  // ms → sec for Poisson probability
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

    // Step 1: Generate Poisson spikes for all inputs (1 LCG each)
    for i in 0..ni {
        let state = (*rng_states.add(i))
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        *rng_states.add(i) = state;
        let r = (state >> 8) as f32 * RNG_SCALE_24;
        let p = (*rates.add(i) * dt_sec).clamp(0.0, 1.0);
        *input_spikes_out.add(i) = if r < p { 1 } else { 0 };
    }

    // Step 2: Build active pre-synapse list (sparse) — reuse pre-allocated buffer.
    active_buf.clear();
    for i in 0..ni {
        if *input_spikes_out.add(i) != 0 {
            active_buf.push(i);
        }
    }
    let active: &[usize] = &*active_buf;

    // Step 3: Fused Izhikevich (same as fused_hidden_izhikevich_step)
    let n8 = nh & !7;
    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut tr = load8_mut(trace, off);
        let mut refr = load8_mut(refractory, off);
        let aa = load8(a, off);
        let bb = load8(b, off);
        let cc = load8(c, off);
        let dd = load8(d, off);

        // Accumulate currents from active pre-synapses only
        let mut I_s = zero;
        for &i in active {
            let w = load8(weights, i * nh + off);
            I_s = w.mul_add(scale10, I_s);
        }

        // Izhikevich dynamics (same as fused_hidden_izhikevich_step)
        let refr_mask = refr.cmp_gt(zero);
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
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

    // Scalar tail (nh should be ×8)
    if n8 < nh {
        let v = std::slice::from_raw_parts_mut(v, nh);
        let u = std::slice::from_raw_parts_mut(u, nh);
        let trace = std::slice::from_raw_parts_mut(trace, nh);
        let spikes = std::slice::from_raw_parts_mut(spikes, nh);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, nh);
        let a_sl = std::slice::from_raw_parts(a, nh);
        let b_sl = std::slice::from_raw_parts(b, nh);
        let c_sl = std::slice::from_raw_parts(c, nh);
        let d_sl = std::slice::from_raw_parts(d, nh);
        for j in n8..nh {
            // Accumulate current from active pre-synapses only
            let mut I_val = 0.0f32;
            for &i in active {
                I_val += *weights.add(i * nh + j) * 10.0;
            }
            if refr_sl[j] > 0.0 {
                refr_sl[j] = (refr_sl[j] - dt).max(0.0);
                spikes[j] = 0;
                continue;
            }
            let orig_v = v[j];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[j] + I_val;
            let du = a_sl[j] * (b_sl[j] * orig_v - u[j]);
            v[j] += dt * dv;
            u[j] += dt * du;
            if v[j] >= 30.0 {
                spikes[j] = 1;
                v[j] = c_sl[j];
                u[j] += d_sl[j];
                refr_sl[j] = tau_ref;
                trace[j] += trace_decay * (1.0 - trace[j]);
            } else {
                spikes[j] = 0;
                trace[j] *= 1.0 - trace_decay;
            }
        }
    }
}
