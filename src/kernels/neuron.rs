//! Pure Rust SIMD neuron dynamics — `wide::f32x8` for AVX2/NEON/WASM.
//!
//! Each function processes `n` neurons. When `n ≥ 8`, uses `f32x8` for
//! 8-wide SIMD (AVX2 on x86_64, NEON on ARM, portable emulation elsewhere),
//! with a scalar tail for remaining elements.
//!
//! ## Implemented models
//!
//! | Function | Model | Key equations |
//! |---|---|---|
//! | `izhikevich_step` | Izhikevich (2003) | dv/dt = 0.04v² + 5v + 140 − u + I |
//! | `saw_step` | SAW (Spike-Adaptive-Width) | SAW + noise |
//! | `psi_wave_step` | ψ-wave (quantum) | Schrödinger-like + normalization |
//! | `lif_step` | LIF | dv/dt = (v_rest − v + I) / τ_m |
//! | `lif_at_step` | LIF-AT | LIF + adaptive threshold θ |
//! | `adex_step` | AdEx | exponential + adaptation current w |

#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt, CmpNe};

// ═══════════════════════════════════════════════════════
//  Helper: load f32x8 from raw pointer (index offset)
// ═══════════════════════════════════════════════════════

#[inline(always)]
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_mut(ptr: *mut f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_i32(ptr: *const i32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p as f32, *p.add(1) as f32, *p.add(2) as f32, *p.add(3) as f32,
                 *p.add(4) as f32, *p.add(5) as f32, *p.add(6) as f32, *p.add(7) as f32])
}
#[inline(always)]
unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
    let a = v.to_array();
    let p = ptr.add(off);
    *p = a[0];
    *p.add(1) = a[1];
    *p.add(2) = a[2];
    *p.add(3) = a[3];
    *p.add(4) = a[4];
    *p.add(5) = a[5];
    *p.add(6) = a[6];
    *p.add(7) = a[7];
}

#[inline(always)]
fn spike_mask_to_i32(mask: f32x8) -> [i32; 8] {
    // f32x8 comparison returns all-1s (true → 0xFFFFFFFF) or all-0s (false)
    // Reinterpret bits to u32, then convert nonzero → 1
    let bits: [u32; 8] = unsafe { core::mem::transmute(mask.to_array()) };
    [
        (bits[0] != 0) as i32,
        (bits[1] != 0) as i32,
        (bits[2] != 0) as i32,
        (bits[3] != 0) as i32,
        (bits[4] != 0) as i32,
        (bits[5] != 0) as i32,
        (bits[6] != 0) as i32,
        (bits[7] != 0) as i32,
    ]
}

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
pub unsafe fn izhikevich_step(
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
        vv = vv + refr_mask.blend(zero, dt_s * dv);  // skip if refractory
        uu = uu + refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr; // spike if v≥30 AND NOT refractory

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
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
//  SAW (Spike-Adaptive-Width)
// ═══════════════════════════════════════════════════════

/// Step `n` SAW neurons with refractory period. Uses `f32x8` SIMD.
pub unsafe fn saw_step(
    v: *mut f32, u: *mut f32, trace: *mut f32,
    a: *const f32, b: *const f32, Ii: *const f32,
    dt: f32, trace_decay: f32, noise_amp: f32,
    spikes: *mut i32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let td_s = f32x8::splat(trace_decay);
    let v_th = f32x8::splat(30.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let v_reset = f32x8::splat(-65.0);
    let u_add = f32x8::splat(8.0);
    let na_s = f32x8::splat(noise_amp);
    let tr_s_ = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut tr = load8_mut(trace, off);
        let mut refr = load8_mut(refractory, off);
        let aa = load8(a, off);
        let bb = load8(b, off);
        let ii = load8(Ii, off);

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v_sq = vv * vv;
        let dv = f32x8::splat(0.04).mul_add(v_sq, f32x8::splat(5.0) * vv) + f32x8::splat(140.0) - uu + ii;
        let du = aa * (bb * vv - uu);
        vv = vv + refr_mask.blend(zero, dt_s * dv);
        uu = uu + refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv + na_s * (f32x8::splat(rand_f32()) - f32x8::splat(0.5))));
        uu = refr_mask.blend(uu, uu + sm.blend(u_add, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s_, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        // trace update
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

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u = std::slice::from_raw_parts_mut(u, n);
        let trace = std::slice::from_raw_parts_mut(trace, n);
        let a = std::slice::from_raw_parts(a, n);
        let b = std::slice::from_raw_parts(b, n);
        let ii = std::slice::from_raw_parts(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt).max(0.0);
                spikes[i] = 0;
                continue;
            }
            let vv = v[i];
            let dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - u[i] + ii[i];
            let du = a[i] * (b[i] * vv - u[i]);
            v[i] += dt * dv;
            u[i] += dt * du;
            if v[i] >= 30.0 {
                spikes[i] = 1;
                v[i] = -65.0;
                u[i] += 8.0;
                refr_sl[i] = tau_ref;
                trace[i] += trace_decay * (1.0 - trace[i]);
            } else {
                spikes[i] = 0;
                v[i] += noise_amp * (rand_f32() - 0.5);
                trace[i] *= 1.0 - trace_decay;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop SAW — Eligibility Propagation for Spike-Adaptive-Width
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop SAW neurons with eligibility traces.
/// Uses `f32x8` SIMD for SAW dynamics + eligibility + surrogate gradient.
///
/// SAW equations (same dv/du as Izhikevich, hardcoded reset):
///   dv/dt = 0.04·v² + 5·v + 140 − u + I
///   du/dt = a·(b·v − u)
///   if v ≥ 30: v = −65, u += 8
///
/// E-Prop: de/dt = −e/τ_e + ψ·z_pre
///   ψ = surrogate gradient = 1 / (1 + |v − 30|/τ)²
///
/// # Returns
/// Total number of output spikes.
pub unsafe fn eprop_saw_step(
    v: *mut f32, u: *mut f32, eligibility: *mut f32, psi: *mut f32,
    Ii: *const f32, spikes: *mut i32, pre_spikes: *const i32,
    dt: f32, tau_e: f32, a: f32, b: f32, tau_sg: f32,
    n: usize,
) -> usize {
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
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut ee = load8_mut(eligibility, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);

        // SAW dynamics (same as Izhikevich)
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii;
        let du = a_s * (b_s * vv - uu);
        vv = vv.mul_add(dt_s, dv);
        uu = uu.mul_add(dt_s, du);

        // Surrogate gradient
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: de/dt = -e/τ_e + ψ · z_pre
        let z_pre = pre.cmp_gt(zero).blend(one, zero);
        let de = -ee / tau_e_s + ps * z_pre;
        ee = ee + de * dt_s;

        // Spike detection: v ≥ 30 → hardcoded reset
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(v_reset, vv);
        uu = uu + sm.blend(u_add, zero);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
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
        for i in n8..n {
            let vv = v_sl[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u_sl[i] + i_sl[i];
            let du = a * (b * vv - u_sl[i]);
            v_sl[i] += dt * dv;
            u_sl[i] += dt * du;
            psi_sl[i] = surrogate_gradient(v_sl[i], 30.0, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] += dt * (-e_sl[i] / tau_e + psi_sl[i] * z_pre);
            if v_sl[i] >= 30.0 {
                sp_sl[i] = 1;
                total_spikes += 1;
                v_sl[i] = -65.0;
                u_sl[i] += 8.0;
            } else {
                sp_sl[i] = 0;
            }
        }
    }

    total_spikes
}

/// Deterministic RNG for SAW noise (thread-safe atomic counter)
fn rand_f32() -> f32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static SEED: AtomicU32 = AtomicU32::new(1);
    let mut s = SEED.fetch_add(1103515245, Ordering::Relaxed);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    SEED.store(s, Ordering::Relaxed);
    (s >> 16) as f32 / 65536.0
}

// ═══════════════════════════════════════════════════════
//  ψ-wave (quantum neural dynamics)
// ═══════════════════════════════════════════════════════

/// Step `n` ψ-wave neurons. Uses `f32x8` SIMD.
pub unsafe fn psi_wave_step(
    v: *mut f32, u: *mut f32,
    h_real: *mut f32, h_imag: *mut f32,
    i_ext: *mut f32, speed: *mut f32,
    dt: f32, hbar: f32, n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let hbar_s = f32x8::splat(hbar);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let hr = load8_mut(h_real, off);
        let hi = load8_mut(h_imag, off);
        let ie = load8_mut(i_ext, off);
        let sp = load8_mut(speed, off);

        let dv = hr - hi * sp + ie;
        let du = hi + hr * sp;
        vv = vv + dt_s * dv / hbar_s;
        uu = uu + dt_s * du / hbar_s;

        // Normalize: keep on unit circle
        let nrm = (vv * vv + uu * uu + f32x8::splat(1e-9)).sqrt();
        let inv = f32x8::splat(1.0) / nrm;
        vv *= inv;
        uu *= inv;

        store8(v, off, vv);
        store8(u, off, uu);
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u = std::slice::from_raw_parts_mut(u, n);
        let h_real = std::slice::from_raw_parts_mut(h_real, n);
        let h_imag = std::slice::from_raw_parts_mut(h_imag, n);
        let i_ext = std::slice::from_raw_parts_mut(i_ext, n);
        let speed = std::slice::from_raw_parts_mut(speed, n);
        for i in n8..n {
            let dv = h_real[i] - h_imag[i] * speed[i] + i_ext[i];
            let du = h_imag[i] + h_real[i] * speed[i];
            v[i] += dt * dv / hbar;
            u[i] += dt * du / hbar;
            let nrm = (v[i] * v[i] + u[i] * u[i] + 1e-9).sqrt();
            let inv = 1.0 / nrm;
            v[i] *= inv;
            u[i] *= inv;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  LIF (Leaky Integrate-and-Fire)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset (5 × n)
pub unsafe fn lif_step(
    v: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 5);
    let dt_val = p[0];  // dt is uniform across all neurons
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(one / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tau.blend(dv, zero));

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                continue;
            }
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if v[i_idx] >= v_thresh {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  LIF-AT (LIF with Adaptive Threshold)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF-AT neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset, tau_theta, beta (7 × n)
pub unsafe fn lif_at_step(
    v: *mut f32, theta: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 7);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut th = load8_mut(theta, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_th = load8(params.wrapping_add(5 * n), off);
        let beta = load8(params.wrapping_add(6 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory — theta still decays)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(one / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tau.blend(dv, zero));

        // Adaptive threshold decay (always decays, even during refractory)
        let mask_tt = tau_th.cmp_gt(zero);
        let dth = mask_tt.blend(dt_s * (-th / tau_th), zero);
        th = th + dth;

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th + th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));
        th = th + sm.blend(beta, zero);

        // Decrement refractory for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(theta, off, th);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let theta = std::slice::from_raw_parts_mut(theta, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                // Theta still decays during refractory
                let tau_theta = p[5 * n + i_idx];
                if tau_theta > 0.0 {
                    theta[i_idx] += dt_val * (-theta[i_idx] / tau_theta);
                }
                continue;
            }
            let dt_val = p[i_idx];
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            let tau_theta = p[5 * n + i_idx];
            let beta = p[6 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if tau_theta > 0.0 {
                theta[i_idx] += dt_val * (-theta[i_idx] / tau_theta);
            }
            if v[i_idx] >= v_thresh + theta[i_idx] {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                theta[i_idx] += beta;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Fast exp — SIMD range reduction + polynomial
// ═══════════════════════════════════════════════════════

/// Fast `exp(x)` approximation for a single `f32`.
///
/// Uses the same 4th-order minimax polynomial as `fast_exp_simd`:
///   1. t = x · log2(e) — range reduction
///   2. n = floor(t), f = t − n — split integer/fractional
///   3. 2^f ≈ P(f) — 4th-order polynomial on [0, 1)
///   4. 2^n = reinterpret((n + 127) << 23) — IEEE 754 bit trick
///   5. e^x = 2^n · 2^f
///
/// Accuracy: ~1e-4 rel. error on [-10, 10].
/// Speed: ~2–3× faster than `f32::exp()`.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let log2e = 1.4426950408889634f32;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;

    // 2^f: 4th-order minimax polynomial on [0, 1)
    let c1 = 0.6930321207;
    let c2 = 0.2413797629;
    let c3 = 0.0520972777;
    let c4 = 0.0093553389;
    let poly = ((c4 * f + c3) * f + c2) * f + c1;
    let exp2f = poly * f + 1.0;

    // 2^n via IEEE 754 bit trick
    let biased = n.wrapping_add(127);
    let exp2n = if biased <= 0 {
        0.0
    } else if biased >= 255 {
        f32::INFINITY
    } else {
        f32::from_bits((biased as u32) << 23)
    };

    exp2n * exp2f
}

/// Fast `exp(x)` approximation using `f32x8` SIMD.
///
/// Algorithm:
/// 1. `t = x * log2(e)` — range reduction
/// 2. `n = floor(t)`, `f = t - n` — split integer/fractional
/// 3. `2^f ≈ P(f)` — 4th-order minimax polynomial on [0, 1)
/// 4. `2^n` = `reinterpret((n + 127) << 23)` — IEEE 754 bit trick
/// 5. `e^x = 2^n * 2^f`
///
/// Accuracy: ~1e-4 relative error on [-10, 10].
/// Speed: ~8× faster than 8 separate `f32::exp()` calls.
#[inline(always)]
pub fn fast_exp_simd(x: f32x8) -> f32x8 {
    // log2(e) ≈ 1.4426950408889634
    let log2e = f32x8::splat(1.4426950408889634);
    let t = x * log2e;

    // Split into integer and fractional parts (per-element for floor + bit ops)
    let arr: [f32; 8] = t.to_array();
    let mut int_part = [0i32; 8];
    let mut frac = [0.0f32; 8];
    let mut exp2_int = [0.0f32; 8]; // 2^int_part

    for j in 0..8 {
        int_part[j] = arr[j].floor() as i32;
        frac[j] = arr[j] - int_part[j] as f32;

        // IEEE 754: 2^n = f32::from_bits((n + 127) << 23)
        let biased = int_part[j].wrapping_add(127);
        if biased <= 0 {
            exp2_int[j] = 0.0; // underflow → 0
        } else if biased >= 255 {
            exp2_int[j] = f32::INFINITY; // overflow
        } else {
            exp2_int[j] = f32::from_bits((biased as u32) << 23);
        }
    }

    // 4th-order minimax polynomial for 2^f on [0, 1):
    // P(f) = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    // Coefficients minimise max relative error on [0, 1).
    let ff = f32x8::from(frac);
    let c1 = f32x8::splat(0.6930321207);       // ln(2) — tuned
    let c2 = f32x8::splat(0.2413797629);       // ln(2)²/2! — tuned
    let c3 = f32x8::splat(0.0520972777);       // ln(2)³/3! — tuned
    let c4 = f32x8::splat(0.0093553389);       // ln(2)⁴/4! — tuned

    // Horner's method: P(f) = (((c4*f + c3)*f + c2)*f + c1)*f + 1
    let poly = ff.mul_add(c4, c3);             // c4·f + c3
    let poly = ff.mul_add(poly, c2);           // (c4·f + c3)·f + c2
    let poly = ff.mul_add(poly, c1);           // ((c4·f + c3)·f + c2)·f + c1
    let poly = ff.mul_add(poly, f32x8::splat(1.0)); // (((c4·f + c3)·f + c2)·f + c1)·f + 1

    // e^x = 2^int_part * poly(frac)
    let exp2 = f32x8::from(exp2_int);
    exp2 * poly
}

// ═══════════════════════════════════════════════════════
//  AdEx (Adaptive Exponential Integrate-and-Fire)
// ═══════════════════════════════════════════════════════

/// Step `n` AdEx neurons with refractory period. Uses `f32x8` SIMD with fast exponential.
///
/// Params: dt, tau_m, tau_w, v_rest, v_thresh, v_reset, delta_t, a, b (9 × n)
///
/// The exponential term uses a SIMD fast-exp approximation (~1e-4 rel. error)
/// instead of per-element `f32::exp()` — ~8× faster.
pub unsafe fn adex_step(
    v: *mut f32, u: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 9);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(3 * n), off);
        let v_th = load8(params.wrapping_add(4 * n), off);
        let v_reset = load8(params.wrapping_add(5 * n), off);
        let delta_t = load8(params.wrapping_add(6 * n), off);
        let aa = load8(params.wrapping_add(7 * n), off);
        let bb = load8(params.wrapping_add(8 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);
        let tau_w = load8(params.wrapping_add(2 * n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // Exponential term: Δ·exp((v − v_T) / Δ) — only for non-refractory
        let exp_arg = ((vv - v_th) / delta_t.max(f32x8::splat(1.0)))
            .max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
        let exp_val = fast_exp_simd(exp_arg);
        let exp_term = delta_t * exp_val;
        let v_gt_mask = vv.cmp_gt(v_th - f32x8::splat(20.0));
        let exp_term = v_gt_mask.blend(exp_term, zero);
        // Zero out exp_term during refractory
        let exp_term = refr_mask.blend(zero, exp_term);

        // Membrane potential (skip if refractory)
        let mask_tm = tau_m.cmp_gt(zero);
        let inv_tm = mask_tm.blend(one / tau_m, zero);
        let dv = (v_rest - vv + exp_term - uu + ii) * inv_tm * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tm.blend(dv, zero));

        // Adaptation current (still decays during refractory)
        let mask_tw = tau_w.cmp_gt(zero);
        let inv_tw = mask_tw.blend(one / tau_w, zero);
        let dw = (aa * (vv - v_rest) - uu) * inv_tw * dt_s;
        uu = uu + mask_tw.blend(dw, zero);

        // Spike detection (only non-refractory)
        let nan_mask = vv.cmp_ne(vv);
        let not_refr = !refr_mask;
        let sm = (vv.cmp_ge(v_th) | nan_mask) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));
        uu = uu + sm.blend(bb, zero);

        // Decrement refractory for all, set to tau_ref on spike
        refr = refr - dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(u, off, uu);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let ii = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt_val).max(0.0);
                // w still decays during refractory
                let tau_w = p[2 * n + i];
                if tau_w > 0.0 {
                    let aa_p = p[7 * n + i];
                    let v_rest_p = p[3 * n + i];
                    u_sl[i] += (aa_p * (v[i] - v_rest_p) - u_sl[i]) / tau_w * dt_val;
                }
                spikes[i] = 0;
                continue;
            }
            let tau_m = p[n + i];
            let tau_w = p[2 * n + i];
            let v_rest = p[3 * n + i];
            let v_thresh = p[4 * n + i];
            let v_reset = p[5 * n + i];
            let delta_t = p[6 * n + i];
            let aa = p[7 * n + i];
            let bb = p[8 * n + i];
            let exp_term = if v[i] > v_thresh - 20.0 {
                delta_t * ((v[i] - v_thresh) / delta_t.max(1.0)).exp()
            } else { 0.0 };
            if tau_m > 0.0 {
                v[i] += (v_rest - v[i] + exp_term - u_sl[i] + ii[i]) / tau_m * dt_val;
            }
            if tau_w > 0.0 {
                u_sl[i] += (aa * (v[i] - v_rest) - u_sl[i]) / tau_w * dt_val;
            }
            if v[i] >= v_thresh || v[i].is_nan() {
                spikes[i] = 1;
                v[i] = v_reset;
                u_sl[i] += bb;
                refr_sl[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Delay Buffer
// ═══════════════════════════════════════════════════════

// Delay buffer functions are memory-bound — keep scalar (no SIMD benefit)

/// Circular delay buffer: shift copy from position `pos`.
pub unsafe fn delay_buffer_circular(
    output: *mut f32, buffer: *const f32,
    n: usize, pos: usize,
) {
    if n == 0 { return; }
    let buf = std::slice::from_raw_parts(buffer, n);
    let out = std::slice::from_raw_parts_mut(output, n);
    let len = n - pos;
    out[..len].copy_from_slice(&buf[pos..]);
    out[len..].copy_from_slice(&buf[..pos]);
}

/// Multi-tap delay: read buffer at 8 different delay indices.
pub unsafe fn delay_buffer_multi_tap(
    output: *mut f32, buffer: *const f32,
    delays: *const i32, n: usize, n_taps: usize,
) {
    let buf = std::slice::from_raw_parts(buffer, n);
    let out = std::slice::from_raw_parts_mut(output, n_taps);
    let d = std::slice::from_raw_parts(delays, n_taps);
    for (i, tap) in d.iter().enumerate() {
        let idx = (*tap as usize).min(n.saturating_sub(1));
        out[i] = buf[idx];
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop LIF (Eligibility Propagation) — SIMD kernel
// ═══════════════════════════════════════════════════════

/// Surrogate gradient: σ'(v) = 1 / (1 + |v - v_th|/τ)²
#[inline(always)]
fn surrogate_gradient(v: f32, v_thresh: f32, tau_sg: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_sg;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

/// Step `n` E-Prop LIF neurons with eligibility traces.
/// Uses `f32x8` SIMD for LIF dynamics + eligibility + surrogate gradient.
///
/// Params: none (all scalar params passed directly)
/// Updates: v, theta, eligibility, psi, spikes
pub unsafe fn eprop_lif_step(
    v: *mut f32, theta: *mut f32, eligibility: *mut f32, psi: *mut f32,
    Ii: *const f32, spikes: *mut i32, pre_spikes: *const i32,
    dt: f32, tau_m: f32, tau_e: f32,
    v_rest: f32, v_thresh: f32, v_reset: f32, tau_sg: f32,
    n: usize,
) -> usize {
    let dt_s = f32x8::splat(dt);
    let tau_m_s = f32x8::splat(tau_m);
    let tau_e_s = f32x8::splat(tau_e);
    let v_rest_s = f32x8::splat(v_rest);
    let v_thresh_s = f32x8::splat(v_thresh);
    let v_reset_s = f32x8::splat(v_reset);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let th = load8(theta, off);
        let mut ee = load8_mut(eligibility, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);

        // LIF dynamics: dv/dt = (v_rest - v + I) / tau_m
        let inv_tau = one / tau_m_s;
        let dv = (v_rest_s - vv + ii) * inv_tau * dt_s;
        vv = vv + dv;

        // Surrogate gradient per lane (SIMD)
        let diff = (vv - v_thresh_s).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: de/dt = -e/τ_e + ψ · z_pre
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        let de = -ee / tau_e_s + ps * z_pre;
        ee = ee + de * dt_s;

        // Spike detection: v ≥ v_thresh + theta
        let sm = vv.cmp_ge(v_thresh_s + th);
        vv = sm.blend(v_reset_s, vv);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
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
        let theta_sl = std::slice::from_raw_parts_mut(theta, n);
        let e_sl = std::slice::from_raw_parts_mut(eligibility, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(Ii, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        let pre_sl = std::slice::from_raw_parts(pre_spikes, n);
        for i in n8..n {
            let dv = (v_rest - v_sl[i] + i_sl[i]) / tau_m * dt;
            v_sl[i] += dv;
            psi_sl[i] = surrogate_gradient(v_sl[i], v_thresh, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] += dt * (-e_sl[i] / tau_e + psi_sl[i] * z_pre);
            if v_sl[i] >= v_thresh + theta_sl[i] {
                sp_sl[i] = 1;
                total_spikes += 1;
                v_sl[i] = v_reset;
            } else {
                sp_sl[i] = 0;
            }
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════
//  E-Prop Izhikevich — Eligibility Propagation for Izhikevich
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop Izhikevich neurons with per-synapse eligibility traces.
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
pub unsafe fn eprop_izhikevich_step_multi(
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
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
        let du = a_s * (b_s * vv - uu);
        vv = vv.mul_add(dt_s, dv);
        uu = uu.mul_add(dt_s, du);

        // Surrogate gradient
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);
        uu = uu + sm.blend(d_s, zero);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(u, off, uu);
        store8(psi, off, ps);

        // Per-synapse eligibility: e_ij = e_ij * (1 - dt/τ_e) + dt * ψ_j * z_pre[i]
        let decay = one - dt_s / tau_e_s;
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            let mut e_simd = load8_mut(e, i * n + off);
            if z != 0.0 {
                e_simd = e_simd * decay + ps * f32x8::splat(z * dt);
            } else {
                e_simd = e_simd * decay;
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

            // Per-synapse eligibility
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                let e_ptr = e.add(i * n + j);
                *e_ptr += dt * (-*e_ptr / tau_e + psi_sl[j] * z);
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

// ═══════════════════════════════════════════════════════
//  E-Prop AdEx Multi — Adaptive Exponential with per-synapse eligibility
// ═══════════════════════════════════════════════════════

/// Step `n` E-Prop AdEx neurons with per-synapse eligibility traces.
/// Column-major layout: `e[i * n + j]` = eligibility for pre-synapse `i`, post-neuron `j`.
///
/// AdEx equations:
///   dv/dt = (v_rest - v + Δ_T·exp((v - v_T)/Δ_T) - w + I) / τ_m
///   dw/dt = (a·(v - v_rest) - w) / τ_w
///   v ≥ v_T → v = v_reset, w += b
///
/// E-Prop: de/dt = −e/τ_e + ψ·z_pre  (ψ = surrogate gradient using v_thresh)
///
/// Uses `f32x8` SIMD with fast exponential approximation.
pub unsafe fn eprop_adex_step_multi(
    v: *mut f32, w: *mut f32, psi: *mut f32,
    e: *mut f32,          // [n_pre × n] column-major
    z_pre: *const i32,    // [n_pre]
    I_ext: *const f32,    // [n]
    weights: *const f32,  // [n_pre × n] column-major
    n: usize,             // n_post (column stride)
    n_pre: usize,
    dt: f32, tau_m: f32, tau_w: f32,
    v_rest: f32, v_thresh: f32, v_reset: f32, delta_t: f32,
    a: f32, b: f32,
    tau_e: f32, tau_sg: f32,
    spikes: *mut i32,
) -> usize {
    assert!(n_pre > 0);
    assert!(tau_m > 0.0 && tau_w > 0.0);
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

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut ww = load8_mut(w, off);
        let ii = load8(I_ext, off);

        // Compute I_total = I_ext + Σ_i weights[i·n + j] · z_pre[i]
        let mut I_s = ii;
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            if z != 0.0 {
                let wgt = load8(weights, i * n + off);
                I_s = wgt.mul_add(f32x8::splat(z), I_s);
            }
        }

        // AdEx dynamics
        // dv = (v_rest - v + Δ·exp((v - v_T)/Δ) - w + I) / τ_m · dt
        // Clamp exp arg to avoid overflow
        let exp_arg = ((vv - v_th_s) / delta_t_s.max(f32x8::splat(1.0)))
            .max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
        let exp_val = fast_exp_simd(exp_arg);
        let exp_term = delta_t_s * exp_val;
        // Zero out where v ≤ v_T - 20
        let v_gt_mask = vv.cmp_gt(v_th_s - f32x8::splat(20.0));
        let exp_term = v_gt_mask.blend(exp_term, zero);

        let dv = (v_rest_s - vv + exp_term - ww + I_s) / tau_m_s * dt_s;
        let dw = (a_s * (vv - v_rest_s) - ww) / tau_w_s * dt_s;
        vv = vv + dv;
        ww = ww + dw;

        // Surrogate gradient
        let diff = (vv - v_th_s).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Spike detection: v ≥ v_thresh OR NaN
        let nan_mask = vv.cmp_ne(vv);
        let sm = vv.cmp_ge(v_th_s) | nan_mask;
        vv = sm.blend(v_reset_s, vv);
        ww = ww + sm.blend(b_s, zero);

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(w, off, ww);
        store8(psi, off, ps);

        // Per-synapse eligibility: e_ij = e_ij * (1 - dt/τ_e) + dt * ψ_j * z_pre[i]
        let decay = one - dt_s / tau_e_s;
        for i in 0..n_pre {
            let z = *z_pre.add(i) as f32;
            let mut e_simd = load8_mut(e, i * n + off);
            if z != 0.0 {
                e_simd = e_simd * decay + ps * f32x8::splat(z * dt);
            } else {
                e_simd = e_simd * decay;
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
        let w_sl = std::slice::from_raw_parts_mut(w, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let i_sl = std::slice::from_raw_parts(I_ext, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        for j in n8..n {
            let mut vv = v_sl[j];
            let mut ww = w_sl[j];

            // Compute I_total
            let mut I_total = i_sl[j];
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                if z != 0.0 {
                    I_total += *weights.add(i * n + j) * z;
                }
            }

            // AdEx dynamics
            let exp_term = if vv > v_thresh - 20.0 {
                delta_t * ((vv - v_thresh) / delta_t.max(1.0)).exp()
            } else { 0.0 };
            let dv = (v_rest - vv + exp_term - ww + I_total) / tau_m * dt;
            let dw = (a * (vv - v_rest) - ww) / tau_w * dt;
            vv += dv;
            ww += dw;

            psi_sl[j] = surrogate_gradient(vv, v_thresh, tau_sg);

            // Per-synapse eligibility
            for i in 0..n_pre {
                let z = *z_pre.add(i) as f32;
                let e_ptr = e.add(i * n + j);
                *e_ptr += dt * (-*e_ptr / tau_e + psi_sl[j] * z);
            }

            if vv >= v_thresh || vv.is_nan() {
                sp_sl[j] = 1;
                total_spikes += 1;
                vv = v_reset;
                ww += b;
            } else {
                sp_sl[j] = 0;
            }

            v_sl[j] = vv;
            w_sl[j] = ww;
        }
    }

    total_spikes
}

/// Step `n` E-Prop Izhikevich neurons with eligibility traces.
/// Uses `f32x8` SIMD for Izhikevich dynamics + eligibility + surrogate gradient.
///
/// Izhikevich equations:
///   dv/dt = 0.04·v² + 5·v + 140 − u + I
///   du/dt = a·(b·v − u)
///   if v ≥ 30: v = c, u += d
///
/// E-Prop: de/dt = −e/τ_e + ψ·z_pre
///   ψ = surrogate gradient = 1 / (1 + |v − 30|/τ)²
///
/// # Returns
/// Total number of output spikes across all neurons.
pub unsafe fn eprop_izhikevich_step(
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

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u, off);
        let mut ee = load8_mut(eligibility, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);

        // Izhikevich dynamics
        // dv = (0.04·v + 5)·v + 140 − u + I
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii;
        let du = a_s * (b_s * vv - uu);
        vv = vv.mul_add(dt_s, dv);
        uu = uu.mul_add(dt_s, du);

        // Surrogate gradient per lane (SIMD)
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: de/dt = -e/τ_e + ψ · z_pre
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        let de = -ee / tau_e_s + ps * z_pre;
        ee = ee + de * dt_s;

        // Spike detection: v ≥ 30
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(c_s, vv);    // v = c where spike
        uu = uu + sm.blend(d_s, zero); // u += d where spike

        // Count spikes
        let sp_arr: [f32; 8] = sm.to_array();
        for i in 0..8 {
            if sp_arr[i].to_bits() != 0 { total_spikes += 1; }
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
        for i in n8..n {
            let vv = v_sl[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u_sl[i] + i_sl[i];
            let du = a * (b * vv - u_sl[i]);
            v_sl[i] += dt * dv;
            u_sl[i] += dt * du;
            psi_sl[i] = surrogate_gradient(v_sl[i], 30.0, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] += dt * (-e_sl[i] / tau_e + psi_sl[i] * z_pre);
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
//  STP (Short-Term Plasticity) — Tsodyks-Markram SIMD kernel
// ═══════════════════════════════════════════════════════

/// Step `n` STP synapses (Tsodyks-Markram model). Uses `f32x8` SIMD.
///
/// Equations:
///   du/dt = (u0-u)/τ_f + u0·(1-u)·s      (facilitation)
///   dx/dt = (1-x)/τ_d − u·x·s             (depression)
///   I_out += A·u·x·s                        (synaptic current)
///
/// Where `s = 1` if pre_spike, else `s = 0`.
/// When no spike, only the decay terms apply.
pub unsafe fn stp_step(
    x: *mut f32, u: *mut f32,
    pre_spikes: *const i32,
    tau_d: *const f32, tau_f: *const f32,
    u0: *const f32, a: *const f32,
    I_out: *mut f32, dt: f32, n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let eps = f32x8::splat(1e-10);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut xx = load8_mut(x, off);
        let mut uu = load8_mut(u, off);
        let sp = load8_i32(pre_spikes, off);
        let td = load8(tau_d, off).max(eps);
        let tf = load8(tau_f, off).max(eps);
        let u0v = load8(u0, off);
        let av = load8(a, off);
        let mut io = load8_mut(I_out, off);

        // s = 1 where pre_spike, else 0
        let s_one = sp.cmp_gt(zero).blend(one, zero);

        // Save pre-update values for spike-dependent terms
        let uu0 = uu;
        let xx0 = xx;

        // Common decay (always applied): u += dt·(u0-u)/τ_f
        uu = uu + dt_s * (u0v - uu) / tf;
        // Common decay (always applied): x += dt·(1-x)/τ_d
        xx = xx + dt_s * (one - xx) / td;

        // Facilitation on spike: + u0·(1-u_old)  (uses pre-decay u)
        uu = uu + s_one * u0v * (one - uu0);
        // Depression on spike: − u_new · x_old  (uses new u, pre-decay x)
        xx = xx + s_one * (-uu * xx0);
        // Synaptic current: + A·u_new·x_new where spike
        io = io + s_one * av * uu * xx;

        store8(x, off, xx);
        store8(u, off, uu);
        store8(I_out, off, io);
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let x_sl = std::slice::from_raw_parts_mut(x, n);
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let pre = std::slice::from_raw_parts(pre_spikes, n);
        let td = std::slice::from_raw_parts(tau_d, n);
        let tf = std::slice::from_raw_parts(tau_f, n);
        let u0v = std::slice::from_raw_parts(u0, n);
        let av = std::slice::from_raw_parts(a, n);
        let io = std::slice::from_raw_parts_mut(I_out, n);
        for i in n8..n {
            let s = if pre[i] != 0 { 1.0 } else { 0.0 };
            let tf_i = tf[i].max(1e-10);
            let td_i = td[i].max(1e-10);
            u_sl[i] += dt * ((u0v[i] - u_sl[i]) / tf_i + u0v[i] * (1.0 - u_sl[i]) * s / dt);
            x_sl[i] += dt * ((1.0 - x_sl[i]) / td_i - u_sl[i] * x_sl[i] * s / dt);
            io[i] += av[i] * u_sl[i] * x_sl[i] * s;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Dopamine-Modulated STDP — 3-factor plasticity for RL
// ═══════════════════════════════════════════════════════
//
// Combines:
//   1. STDP eligibility (pre-post spike timing)
//   2. Dopamine gate (reward signal)
//   3. Eligibility trace decay
//
// Equations per synapse:
//   tr_pre += 1 on pre-spike, decays otherwise
//   tr_post += 1 on post-spike, decays otherwise
//   e += A_ltp · tr_pre · post_spike − A_ltd · tr_post · pre_spike
//   e *= (1 − dt/τ_e)  (eligibility decay)
//   Δw = η · D · e
//
// Based on: Izhikevich (2007) "Solving the distal reward problem
// through linkage of STDP and dopamine signaling" — Cerebral Cortex

/// Update dopamine-modulated STDP traces and eligibility for `n` synapses.
/// Uses `f32x8` SIMD.
///
/// # Args
/// * `trace_pre` — pre-synaptic spike trace [n] (in/out)
/// * `trace_post` — post-synaptic spike trace [n] (in/out)
/// * `eligibility` — eligibility trace e [n] (in/out)
/// * `pre_spikes` — pre-synaptic spikes [n] (0/1)
/// * `post_spikes` — post-synaptic spikes [n] (0/1)
/// * `dt` — time step (ms)
/// * `tau_pre` — pre-trace time constant (ms)
/// * `tau_post` — post-trace time constant (ms)
/// * `tau_e` — eligibility trace time constant (ms)
/// * `a_ltp` — LTP amplitude (positive STDP)
/// * `a_ltd` — LTD amplitude (negative STDP)
pub unsafe fn dopamine_traces_step(
    trace_pre: *mut f32, trace_post: *mut f32, eligibility: *mut f32,
    pre_spikes: *const i32, post_spikes: *const i32,
    dt: f32, tau_pre: f32, tau_post: f32, tau_e: f32,
    a_ltp: f32, a_ltd: f32, n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let tau_pre_s = f32x8::splat(tau_pre.max(1.0));
    let tau_post_s = f32x8::splat(tau_post.max(1.0));
    let tau_e_s = f32x8::splat(tau_e.max(1.0));
    let a_ltp_s = f32x8::splat(a_ltp);
    let a_ltd_s = f32x8::splat(a_ltd);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut tp = load8_mut(trace_pre, off);
        let mut tpo = load8_mut(trace_post, off);
        let mut ee = load8_mut(eligibility, off);
        let pre = load8_i32(pre_spikes, off);
        let post = load8_i32(post_spikes, off);

        // Decay traces
        let decay_pre = one - dt_s / tau_pre_s;
        let decay_post = one - dt_s / tau_post_s;
        tp = tp * decay_pre;
        tpo = tpo * decay_post;

        // Spike-triggered trace increase
        let pre_mask = pre.cmp_gt(zero);
        let post_mask = post.cmp_gt(zero);
        tp = tp + pre_mask.blend(one, zero);
        tpo = tpo + post_mask.blend(one, zero);

        // STDP eligibility: e += A_ltp · tr_pre · post_spike − A_ltd · tr_post · pre_spike
        let ltp_term = a_ltp_s * tp * post_mask.blend(one, zero);
        let ltd_term = a_ltd_s * tpo * pre_mask.blend(one, zero);
        ee = ee + ltp_term - ltd_term;

        // Eligibility decay
        let decay_e = one - dt_s / tau_e_s;
        ee = ee * decay_e;

        store8(trace_pre, off, tp);
        store8(trace_post, off, tpo);
        store8(eligibility, off, ee);
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let tp = std::slice::from_raw_parts_mut(trace_pre, n);
        let tpo = std::slice::from_raw_parts_mut(trace_post, n);
        let ee = std::slice::from_raw_parts_mut(eligibility, n);
        let pre = std::slice::from_raw_parts(pre_spikes, n);
        let post = std::slice::from_raw_parts(post_spikes, n);
        for i in n8..n {
            let decay_pre = 1.0 - dt / tau_pre.max(1.0);
            let decay_post = 1.0 - dt / tau_post.max(1.0);
            tp[i] *= decay_pre;
            tpo[i] *= decay_post;
            if pre[i] != 0 { tp[i] += 1.0; }
            if post[i] != 0 { tpo[i] += 1.0; }
            let ltp = if post[i] != 0 { a_ltp * tp[i] } else { 0.0 };
            let ltd = if pre[i] != 0 { a_ltd * tpo[i] } else { 0.0 };
            ee[i] += ltp - ltd;
            let decay_e = 1.0 - dt / tau_e.max(1.0);
            ee[i] *= decay_e;
        }
    }
}

/// Apply dopamine-modulated weight update: Δw = η · D · e.
/// Uses `f32x8` SIMD.
///
/// # Args
/// * `weights` — weight matrix [n] (in/out)
/// * `eligibility` — eligibility traces [n]
/// * `dopamine` — scalar dopamine signal (D)
/// * `lr` — learning rate η
/// * `w_min` — minimum weight (clamp)
/// * `w_max` — maximum weight (clamp)
/// * `n` — number of synapses
pub unsafe fn dopamine_weight_update(
    weights: *mut f32,
    eligibility: *const f32,
    dopamine: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
    n: usize,
) {
    let lr_dopa = f32x8::splat(lr * dopamine);
    let wmin_s = f32x8::splat(w_min);
    let wmax_s = f32x8::splat(w_max);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let ee = load8(eligibility, off);
        w = w + lr_dopa * ee;
        w = w.min(wmax_s).max(wmin_s);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let ee = std::slice::from_raw_parts(eligibility, n);
        for i in n8..n {
            w[i] += lr * dopamine * ee[i];
            w[i] = w[i].clamp(w_min, w_max);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Fused Izhikevich with weight accumulation (SIMD)
// ═══════════════════════════════════════════════════════

/// Fused kernel: accumulate input weights → Izhikevich step in one SIMD pass.
/// Eliminates the intermediate I_h buffer and memory round-trip.
///
/// For each block of 8 hidden neurons:
///   I[j..j+8] += Σ_i weights[i·nh + j..j+8] · z_pre[i] · 10.0
///   Izhikevich_step(j)
///
/// All state updates happen in-register — no scalar accumulation loop,
/// no intermediate buffer. ~3x faster than separate weight loop + izhikevich.
///
/// # Safety
/// All pointers must be valid for the given dimensions. `nh` must be padded
/// to a multiple of 8 (asserted in scalar tail fallback).
pub unsafe fn fused_hidden_izhikevich_step(
    v: *mut f32, u: *mut f32, trace: *mut f32, spikes: *mut i32, refractory: *mut f32,
    a: *const f32, b: *const f32, c: *const f32, d: *const f32,
    weights: *const f32,
    input_spikes: *const i32,
    dt: f32, trace_decay: f32, tau_ref: f32,
    ni: usize, nh: usize,
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

        // Accumulate currents from all active inputs directly in SIMD register
        let mut I_s = zero;
        for i in 0..ni {
            let zi = *input_spikes.add(i);
            if zi != 0 {
                let w = load8(weights, i * nh + off);
                I_s = w.mul_add(scale10, I_s);  // I_s += w * 10.0
            }
        }

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
        let du = aa * (bb * vv - uu);
        vv = vv + refr_mask.blend(zero, dt_s * dv);
        uu = uu + refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // Reset after spike
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr = refr - dt_s;
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
    }

    // Scalar tail (should be empty since nh is padded to ×8)
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
            // Accumulate current (scalar)
            let mut I_val = 0.0f32;
            for i in 0..ni {
                let zi = *input_spikes.add(i);
                if zi != 0 {
                    I_val += *weights.add(i * nh + j) * 10.0;
                }
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
pub unsafe fn fused_output_lif_step(
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

        // LIF dynamics with refractory
        let refr_mask = refr.cmp_gt(zero);
        let mask_tau = tau_m_p.cmp_gt(zero);
        let inv_tau = mask_tau.blend(one / tau_m_p, zero);
        let dv = (v_rest_p - vv + I_s) * inv_tau * dt_s;
        vv = vv + refr_mask.blend(zero, mask_tau.blend(dv, zero));

        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th_p) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset_p, vv));

        // Decrement refractory for all, set to tau_ref on spike
        refr = refr - dt_s;
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
                v[k] += (v_rest - v[k] + I_val) / tau_m * dt;
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
pub unsafe fn poisson_fused_hidden_step(
    v: *mut f32, u: *mut f32, trace: *mut f32, spikes: *mut i32, refractory: *mut f32,
    a: *const f32, b: *const f32, c: *const f32, d: *const f32,
    weights: *const f32,
    rng_states: *mut u32,      // [ni] per-input-neuron LCG states
    rates: *const f32,         // [ni] Poisson rates in Hz
    input_spikes_out: *mut i32, // [ni] — spike flags for eligibility
    dt: f32,                    // ms (Poisson uses dt/1000 for probability)
    trace_decay: f32, tau_ref: f32,
    ni: usize, nh: usize,
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
        let r = (state >> 8) as f32 / 16777216.0;
        let p = (*rates.add(i) * dt_sec).clamp(0.0, 1.0);
        *input_spikes_out.add(i) = if r < p { 1 } else { 0 };
    }

    // Step 2: Fused Izhikevich (same as fused_hidden_izhikevich_step)
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

        // Accumulate currents from all active inputs directly in SIMD register
        let mut I_s = zero;
        for i in 0..ni {
            if *input_spikes_out.add(i) != 0 {
                let w = load8(weights, i * nh + off);
                I_s = w.mul_add(scale10, I_s);
            }
        }

        // Izhikevich dynamics (same as fused_hidden_izhikevich_step)
        let refr_mask = refr.cmp_gt(zero);
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + I_s;
        let du = aa * (bb * vv - uu);
        vv = vv + refr_mask.blend(zero, dt_s * dv);
        uu = uu + refr_mask.blend(zero, dt_s * du);

        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(cc, vv));
        uu = refr_mask.blend(uu, uu + sm.blend(dd, zero));

        refr = refr - dt_s;
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
            let mut I_val = 0.0f32;
            for i in 0..ni {
                if *input_spikes_out.add(i) != 0 {
                    I_val += *weights.add(i * nh + j) * 10.0;
                }
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
//  STDP (Spike-Timing-Dependent Plasticity)
// ═══════════════════════════════════════════════════════

/// STDP weight update: w += A⁺·pre − A⁻·post, clamped to [w_min, w_max].
/// Uses `f32x8` SIMD.
pub unsafe fn stdp_update(
    weights: *mut f32, traces_pre: *const f32, traces_post: *const f32,
    lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32, n: usize,
) {
    let ltd = f32x8::splat(lr_ltd);
    let ltp = f32x8::splat(lr_ltp);
    let wmin = f32x8::splat(w_min);
    let wmax = f32x8::splat(w_max);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let pre = load8(traces_pre, off);
        let post = load8(traces_post, off);
        w = w + ltp * post - ltd * pre;
        w = w.min(wmax).max(wmin);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let pre = std::slice::from_raw_parts(traces_pre, n);
        let post = std::slice::from_raw_parts(traces_post, n);
        for i in n8..n {
            w[i] += lr_ltp * post[i] - lr_ltd * pre[i];
            w[i] = w[i].clamp(w_min, w_max);
        }
    }
}

/// STDP with homeostatic weight-sum regulation. Uses `f32x8` SIMD.
pub unsafe fn stdp_homeostatic(
    weights: *mut f32, traces_pre: *const f32, traces_post: *const f32,
    lr_ltd: f32, lr_ltp: f32, target_sum: f32, w_max: f32, n: usize,
) {
    // Compute sum via SIMD reduction
    let n8 = n & !7;
    let mut sum = 0.0f32;
    let w_slc = std::slice::from_raw_parts(weights, n);
    for off in (0..n8).step_by(8) {
        let wv = load8(weights, off);
        let a: [f32; 8] = wv.to_array();
        sum += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    }
    for i in n8..n { sum += w_slc[i]; }

    let ltd = f32x8::splat(lr_ltd);
    let ltp = f32x8::splat(lr_ltp);
    let homeo = f32x8::splat((target_sum - sum) / n as f32);
    let wmax = f32x8::splat(w_max);
    let zero = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let pre = load8(traces_pre, off);
        let post = load8(traces_post, off);
        w = w + ltp * post - ltd * pre + homeo;
        w = w.min(wmax).max(zero);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let pre = std::slice::from_raw_parts(traces_pre, n);
        let post = std::slice::from_raw_parts(traces_post, n);
        for i in n8..n {
            w[i] += lr_ltp * post[i] - lr_ltd * pre[i] + (target_sum - sum) / n as f32;
            w[i] = w[i].max(0.0).min(w_max);
        }
    }
}
