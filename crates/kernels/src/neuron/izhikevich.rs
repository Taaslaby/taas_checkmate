#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{fast_exp_simd, load8, load8_mut, load8_i32, store8, spike_mask_to_i32, surrogate_gradient};

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
        vv += refr_mask.blend(zero, dt_s * dv);  // skip if refractory
        uu += refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr; // spike if v≥30 AND NOT refractory

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr -= dt_s;  // Always decrement
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
//  Izhikevich Dual Memory — Fast + Slow Pathways
//  (Sun et al. 2026, Nature Machine Intelligence)
// ═══════════════════════════════════════════════════════

/// Izhikevich with dual memory pathway (Sun et al. 2026, Nature MI).
///
/// Extends standard Izhikevich with a **slow memory state** `m` per neuron:
///
/// ```text
/// Slow pathway (memory):
///   m[t] = τ_m · m[t-1] + (1 - τ_m) · spike[t-1]
///
/// Fast pathway (Izhikevich, modulated):
///   dv/dt = 0.04·v² + 5·v + 140 − u + (I + w_m · m[t-1])
///   du/dt = a·(b·v − u)
/// ```
///
/// The slow state `m` accumulates recent spiking activity with time constant
/// `τ_m` (e.g., τ_m = 0.999 → ~1000ms memory at 0.1ms dt). The memory
/// readout `w_m · m` modulates the input current, implementing a
/// **fast-slow** neural dynamics where past activity shapes present excitability.
///
/// Reference: Sun et al. (2026) "Algorithm–hardware co-design of
/// neuromorphic networks with dual memory pathways", Nature Machine Intelligence.
///
/// # Parameters
/// - `m`: slow state array [n] — accumulates spike history
/// - `tau_m`: slow state time constant (0..1, higher = slower decay)
/// - `w_m`: memory-to-current weight (scales slow state contribution to I)
///
/// All other params match `izhikevich_step`.
///
/// # Safety
/// All pointer args must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn izhikevich_dual_step(
    v: *mut f32, u: *mut f32, trace: *mut f32,
    a: *const f32, b: *const f32, Ii: *const f32,
    dt: f32, trace_decay: f32,
    spikes: *mut i32, c: *const f32, d: *const f32,
    refractory: *mut f32, tau_ref: f32,
    m: *mut f32,           // slow state per neuron
    tau_m: f32,            // slow state time constant
    w_m: f32,              // memory-to-current weight
    n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let half = f32x8::splat(0.5);
    let zero = f32x8::splat(0.0);
    let c04 = f32x8::splat(0.04);
    let c5 = f32x8::splat(5.0);
    let c140 = f32x8::splat(140.0);
    let tr_s = f32x8::splat(tau_ref);
    let tau_m_s = f32x8::splat(tau_m);
    let o_m_tau = f32x8::splat(1.0 - tau_m);
    let w_m_s = f32x8::splat(w_m);

    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut tr = load8_mut(trace, off);
        let mut refr = load8_mut(refractory, off);
        let mut mm = load8_mut(m, off);  // ← slow state loaded
        let aa = load8(a, off);
        let bb = load8(b, off);
        let mut ii = load8(Ii, off);     // ← Ii is mutable now (we'll add w_m * m)
        let cc = load8(c, off);
        let dd = load8(d, off);

        // ── Slow pathway: modulate input current ──
        // I += w_m * m[t-1]   (slow state modulates fast dynamics)
        ii = w_m_s.mul_add(mm, ii);

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii;
        let du = aa * (bb * vv - uu);
        vv += refr_mask.blend(zero, dt_s * dv);
        uu += refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter
        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

        // trace update
        let tr_s_val = tr.mul_add(td_s, one - tr);
        let tr_n_val = tr * (one - td_s);
        tr = refr_mask.blend(tr, sm.blend(tr_s_val, tr_n_val));

        // ── Slow state update: m[t] = τ_m · m[t-1] + (1-τ_m) · spike[t] ──
        // spike = 1.0 where sm is true (and not refractory), 0.5 elsewhere as neutral baseline
        // Using 0.5 as neutral gives a nice sigmoidal accumulation
        let spike_val = sm.blend(one, half);
        mm = tau_m_s * mm + o_m_tau * spike_val;
        // Clamp m to [0, 1]
        mm = mm.max(zero).min(one);

        store8(v, off, vv);
        store8(u, off, uu);
        store8(trace, off, tr);
        store8(refractory, off, refr);
        store8(m, off, mm);  // ← slow state stored

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
        let m = std::slice::from_raw_parts_mut(m, n);
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
                // Slow state decay even during refractory
                m[i] = tau_m * m[i] + (1.0 - tau_m) * 0.5;
                continue;
            }
            let orig_v = v[i];
            // Slow pathway: modulate current
            let I_total = ii[i] + w_m * m[i];
            let dv = (0.04 * orig_v + 5.0) * orig_v + 140.0 - u[i] + I_total;
            let du = a_sl[i] * (b_sl[i] * orig_v - u[i]);
            v[i] += dt * dv;
            u[i] += dt * du;
            if v[i] >= 30.0 {
                spikes[i] = 1;
                v[i] = c_sl[i];
                u[i] += d_sl[i];
                refr_sl[i] = tau_ref;
                trace[i] += trace_decay * (1.0 - trace[i]);
                // Slow state: spike pushes m toward 1
                m[i] = tau_m * m[i] + (1.0 - tau_m) * 1.0;
            } else {
                spikes[i] = 0;
                trace[i] *= 1.0 - trace_decay;
                // Slow state: no spike pushes m toward 0.5
                m[i] = tau_m * m[i] + (1.0 - tau_m) * 0.5;
            }
            m[i] = m[i].clamp(0.0, 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop Izhikevich — Eligibility Propagation for Izhikevich
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop Izhikevich neurons with per-synapse eligibility traces.
///
/// **Dense (column-major) version** — рекомендуется для малых полносвязных
/// слоёв (n < 10K). Хранит веса и eligibility плотно: `e[i·n + j]`.
///
/// # Dense vs Sparse E-Prop
///
/// | Версия | Расположение | Когда использовать |
/// |--------|-------------|-------------------|
/// | `eprop_izhikevich_step_multi` (эта) | Column-major dense | n < 10K, полносвязные |
/// | `sim_izhikevich_eprop` (connectome) | CSR sparse | n > 10K, биологические |
///
/// **Dense cache warning:** при n > 10K column-major stride = n × 4 bytes.
/// Для n=100K stride = 400KB — гарантированный L2/L3 cache miss на каждый
/// пресинаптический нейрон. Для больших сетей используйте CSR-версию.
///
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

    // Precompute e_decay — constant across all SIMD blocks and neurons
    let e_decay = fast_exp_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

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
        vv += dt_s * dv;
        uu += dt_s * du;

        // Surrogate gradient
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);
        uu += sm.blend(d_s, zero);

        // Count spikes
        for &val in sm.to_array().iter() {
            if val.to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(u, off, uu);
        store8(psi, off, ps);

        // Per-synapse eligibility: exponential Euler
        // e_ij(t+dt) = e_ij·e_decay + ψ_j·z_pre·τ_e·(1 - e_decay)
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            let mut e_simd = load8_mut(e, i * n + off);
            if z != 0.0 {
                e_simd = e_simd * e_decay + ps * f32x8::splat(z) * tau_e_s * one_minus_e_decay;
            } else {
                e_simd *= e_decay;
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
        // Precompute scalar e_decay for the scalar tail
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
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

            // Per-synapse eligibility (exponential Euler)
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                let e_ptr = e.add(i * n + j);
                *e_ptr = *e_ptr * e_decay_s + psi_sl[j] * z * tau_e * one_minus_e_decay_s;
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
/// Step `n` E-Prop Izhikevich neurons with eligibility traces.
/// Uses `f32x8` SIMD for Izhikevich dynamics + eligibility + surrogate gradient.
///
/// Izhikevich equations (Forward Euler):
///   dv/dt = 0.04·v² + 5·v + 140 − u + I
///   du/dt = a·(b·v − u)
///   if v ≥ 30: v = c, u += d
///
/// E-Prop eligibility (Exponential Euler):
///   e(t+dt) = e·exp(-dt/τ_e) + ψ·z_pre·τ_e·(1 - exp(-dt/τ_e))
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

    // Precompute e_decay — constant across all SIMD blocks and neurons
    let e_decay = fast_exp_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

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
        vv += dt_s * dv;
        uu += dt_s * du;

        // Surrogate gradient per lane (SIMD)
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: exponential Euler
        // de/dt = -e/τ_e + ψ · z_pre → e(t+dt) = e·decay + ψ·z_pre·τ_e·(1 - decay)
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        ee = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);    // v = c where spike
        uu += sm.blend(d_s, zero); // u += d where spike

        // Count spikes
        for &val in sm.to_array().iter() {
            if val.to_bits() != 0 { total_spikes += 1; }
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
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for i in n8..n {
            let vv = v_sl[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u_sl[i] + i_sl[i];
            let du = a * (b * vv - u_sl[i]);
            v_sl[i] += dt * dv;
            u_sl[i] += dt * du;
            psi_sl[i] = surrogate_gradient(v_sl[i], 30.0, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] = e_sl[i] * e_decay_s + psi_sl[i] * z_pre * tau_e * one_minus_e_decay_s;
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

// ═══════════════════════════════════════════════════════
//  Hebbian Plasticity — dW = η · (rᵢ · rⱼ) − α · W
//  Applied on CSR synaptic weights.
// ═══════════════════════════════════════════════════════

/// Hebbian plasticity update for CSR synaptic weights.
///
/// Implements: `dW = η · r_pre · r_post − α · W`
///
/// Where:
/// - `η` = learning rate (lr)
/// - `r_pre` = slow state `m` of presynaptic neuron (recent firing rate proxy)
/// - `r_post` = slow state `m` of postsynaptic neuron
/// - `α` = weight decay (prevents unbounded growth)
///
/// This is applied directly to CSR values in-place:
/// `W[idx] += lr · m[pre] · m[post] − decay · W[idx] · dt`
///
/// Uses `r_pre · r_post` as the Hebbian correlation signal (from slow states),
/// which is more stable than instantaneous spike coincidence.
///
/// # Reference
/// Sun et al. (2026) — Hebbian plasticity on 15M synapses leads to emergent
/// individuality: two identical connectomes develop 76K+ different synapses
/// after 24h of simulation.
///
/// # Safety
/// All pointer args valid for their respective lengths.
pub(crate) unsafe fn hebbian_csr_step(
    values: *mut f32,  // CSR synaptic weights [n_synapses] — mutated in-place
    col_indices: *const i32,  // CSR column indices [n_synapses]
    row_ptr: *const i32,     // CSR row pointer [n_neurons + 1]
    m: *const f32,           // slow state per neuron [n_neurons]
    n_neurons: usize,
    n_synapses: usize,
    lr: f32,                 // Hebbian learning rate
    decay: f32,              // weight decay
    dt: f32,                 // timestep
) {
    let _zero = f32x8::splat(0.0);
    let lr_s = f32x8::splat(lr);
    let decay_dt_s = f32x8::splat(decay * dt);
    let min_w = f32x8::splat(-1.0);
    let max_w = f32x8::splat(1.0);

    for pre in 0..n_neurons {
        let start = (*row_ptr.add(pre)).max(0) as usize;
        let end = (*row_ptr.add(pre + 1) as usize).min(n_synapses);
        if start >= end { continue; }

        let m_pre = f32x8::splat(*m.add(pre));

        for off in (start..end).step_by(8) {
            let batch_end = (off + 8).min(end);
            let batch_n = batch_end - off;

            // Build post_m vector via plain array (8× faster than match-arms per element)
            let mut post_arr = [0.0f32; 8];
            let mut w_arr = [0.0f32; 8];
            for j in 0..batch_n {
                let post_idx = *col_indices.add(off + j) as usize;
                post_arr[j] = if post_idx < n_neurons { *m.add(post_idx) } else { 0.0 };
                w_arr[j] = *values.add(off + j);
            }
            // Fill remaining with last valid value (avoids zero-bias on partial blocks)
            if batch_n < 8 && batch_n > 0 {
                let last_m = post_arr[batch_n - 1];
                let last_w = w_arr[batch_n - 1];
                for j in batch_n..8 {
                    post_arr[j] = last_m;
                    w_arr[j] = last_w;
                }
            }
            let post_m = f32x8::from(post_arr);
            let ww = f32x8::from(w_arr);

            // Hebbian update: dw = lr * m_pre * m_post - decay * w * dt
            let hebb = m_pre * post_m;
            let dw = lr_s * hebb - decay_dt_s * ww;
            let new_w = (ww + dw).max(min_w).min(max_w);

            // Store back
            let new_arr = new_w.to_array();
            for j in 0..batch_n {
                *values.add(off + j) = new_arr[j];
            }
        }
    }
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
        let I = vec![30.0f32; n];
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
        let I = vec![-20.0f32; n];
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
        let I = vec![30.0f32; n];
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

    // ─── Dual Memory tests ────────────────────────────

    #[test]
    fn izhikevich_dual_basic() {
        let n = 16;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let I = vec![10.0f32; n];  // moderate current
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut m = vec![0.5f32; n];  // slow state starts at 0.5
        let tau_m = 0.99;              // slow memory (~100ms at dt=1)
        let w_m = 5.0;                // memory-to-current weight

        let mut total_spikes = 0i32;
        for _ in 0..100 {
            unsafe { izhikevich_dual_step(
                v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5,
                spikes.as_mut_ptr(), c.as_ptr(), d.as_ptr(),
                refractory.as_mut_ptr(), 3.0,
                m.as_mut_ptr(), tau_m, w_m, n,
            ); }
            total_spikes += spikes.iter().sum::<i32>();
        }

        // With w_m=5.0 and I=10, the slow state modulation should help spiking
        assert!(total_spikes > 0, "Dual memory Izhikevich should spike with I=10, w_m=5");

        // Slow state should have evolved from 0.5
        let m_mean: f32 = m.iter().sum::<f32>() / n as f32;
        assert!(m_mean > 0.45 && m_mean < 1.0, "Slow state should be in (0.45, 1.0), got {m_mean}");
    }

    #[test]
    fn izhikevich_dual_slow_state_accumulates() {
        let n = 8;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let I = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut m = vec![0.0f32; n];  // start at 0 (no history)
        let tau_m = 0.9f32;            // fast decay for testing
        let w_m = 1.0f32;

        // Run 20 steps — slow state should increase from spiking
        for _ in 0..20 {
            unsafe { izhikevich_dual_step(
                v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5,
                spikes.as_mut_ptr(), c.as_ptr(), d.as_ptr(),
                refractory.as_mut_ptr(), 3.0,
                m.as_mut_ptr(), tau_m, w_m, n,
            ); }
        }

        // Slow state should be > 0 (accumulated spiking)
        let m_sum: f32 = m.iter().sum();
        assert!(m_sum > 0.0, "Slow state should accumulate, got sum={m_sum}");
    }

    #[test]
    fn izhikevich_dual_non_aligned_n() {
        let n = 13;
        let mut v = vec![-65.0f32; n];
        let mut u = vec![-13.0f32; n];
        let mut trace = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let I = vec![10.0f32; n];
        let mut spikes = vec![0i32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        let mut refractory = vec![0.0f32; n];
        let mut m = vec![0.5f32; n];
        let tau_m = 0.99f32;
        let w_m = 5.0f32;

        let mut total_spikes = 0i32;
        for _ in 0..100 {
            unsafe { izhikevich_dual_step(
                v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
                a.as_ptr(), b.as_ptr(), I.as_ptr(), 1.0, 0.5,
                spikes.as_mut_ptr(), c.as_ptr(), d.as_ptr(),
                refractory.as_mut_ptr(), 3.0,
                m.as_mut_ptr(), tau_m, w_m, n,
            ); }
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "Dual memory n=13 should spike");
    }

    // ─── Hebbian CSR test ─────────────────────────────

    #[test]
    fn hebbian_csr_basic() {
        // Simple 3-neuron ring: 0→1, 1→2, 2→0
        let n_neurons = 3;
        let n_synapses = 3;
        let mut values = vec![0.5f32, 0.3, 0.7];
        let col_indices = vec![1i32, 2, 0];
        let row_ptr = vec![0i32, 1, 2, 3];
        let m = vec![0.8f32, 0.2, 0.6];  // slow states
        let lr = 0.01f32;
        let decay = 0.001f32;
        let dt = 1.0f32;

        let old_values = values.clone();

        unsafe {
            hebbian_csr_step(
                values.as_mut_ptr(),
                col_indices.as_ptr(),
                row_ptr.as_ptr(),
                m.as_ptr(),
                n_neurons,
                n_synapses,
                lr,
                decay,
                dt,
            );
        }

        // Check that weights changed
        for i in 0..n_synapses {
            let pre = i; // pre = i for this ring
            let post = col_indices[i] as usize;
            let expected_dw = lr * m[pre] * m[post] - decay * old_values[i] * dt;
            let expected = (old_values[i] + expected_dw).clamp(-1.0, 1.0);
            assert!((values[i] - expected).abs() < 1e-5,
                "Hebbian CSR [{i}]: got {}, expected {}", values[i], expected);
        }
    }
}
