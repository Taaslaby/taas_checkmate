#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{fast_exp_simd, load8, load8_mut, load8_i32, store8, spike_mask_to_i32, rand_f32, surrogate_gradient};

// ═══════════════════════════════════════════════════════
//  SAW (Spike-Adaptive-Width)
// ═══════════════════════════════════════════════════════

/// Step `n` SAW neurons with refractory period. Uses `f32x8` SIMD.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn saw_step(
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
        let ii = load8(Ii, off).max(f32x8::splat(-100.0)).min(f32x8::splat(100.0));

        // Refractory mask: refr > 0 → skip dynamics
        let refr_mask = refr.cmp_gt(zero);

        // Compute dv/du only for non-refractory neurons
        let v_sq = vv * vv;
        let dv = f32x8::splat(0.04).mul_add(v_sq, f32x8::splat(5.0) * vv) + f32x8::splat(140.0) - uu + ii;
        let du = aa * (bb * vv - uu);
        vv += refr_mask.blend(zero, dt_s * dv);
        uu += refr_mask.blend(zero, dt_s * du);

        // Spike detection only for non-refractory
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // Reset after spike
        // Generate 8 independent random values for noise (each neuron gets its own)
        let rand_vals = [
            rand_f32(), rand_f32(), rand_f32(), rand_f32(),
            rand_f32(), rand_f32(), rand_f32(), rand_f32(),
        ];
        let rand_v = f32x8::from(rand_vals);
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv + na_s * (rand_v - f32x8::splat(0.5))));
        uu = refr_mask.blend(uu, uu + sm.blend(u_add, zero));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr -= dt_s;  // Always decrement
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
            let dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - u[i] + ii[i].clamp(-100.0, 100.0);
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
/// E-Prop eligibility (Exponential Euler):
///   e(t+dt) = e·exp(-dt/τ_e) + ψ·z_pre·τ_e·(1 - exp(-dt/τ_e))
///   ψ = surrogate gradient = 1 / (1 + |v − 30|/τ)²
///
/// # Returns
/// Total number of output spikes.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn eprop_saw_step(
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
        let ii_clamped = load8(Ii, off).max(f32x8::splat(-100.0)).min(f32x8::splat(100.0));
        let pre = load8_i32(pre_spikes, off);

        // SAW dynamics (same as Izhikevich)
        // Euler: v += dt * dv
        let v5 = vv.mul_add(c04, c5);
        let dv = v5.mul_add(vv, c140) - uu + ii_clamped;
        let du = a_s * (b_s * vv - uu);
        vv += dt_s * dv;
        uu += dt_s * du;

        // Surrogate gradient
        let diff = (vv - v_th).abs() / tau_sg_s;
        let one_plus_diff = one + diff;
        let ps = one / (one_plus_diff * one_plus_diff);

        // Eligibility trace: exponential Euler
        // de/dt = -e/τ_e + ψ · z_pre → e(t+dt) = e·exp(-dt/τ_e) + ψ·z_pre·τ_e·(1 - exp(-dt/τ_e))
        let z_pre = pre.cmp_gt(zero).blend(one, zero);
        let e_decay = fast_exp_simd(-dt_s / tau_e_s);
        let one_minus_e_decay = one - e_decay;
        ee = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;

        // Spike detection: v ≥ 30 → hardcoded reset
        let sm = vv.cmp_ge(v_th);
        vv = sm.blend(v_reset, vv);
        uu += sm.blend(u_add, zero);

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
        for i in n8..n {
            let vv = v_sl[i];
            let dv = (0.04 * vv + 5.0) * vv + 140.0 - u_sl[i] + i_sl[i].clamp(-100.0, 100.0);
            let du = a * (b * vv - u_sl[i]);
            v_sl[i] += dt * dv;
            u_sl[i] += dt * du;
            psi_sl[i] = surrogate_gradient(v_sl[i], 30.0, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            let e_decay = (-dt / tau_e).exp();
            e_sl[i] = e_sl[i] * e_decay + psi_sl[i] * z_pre * tau_e * (1.0 - e_decay);
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

// ═══════════════════════════════════════════════════════
//  ψ-wave (quantum neural dynamics)
// ═══════════════════════════════════════════════════════

/// Step `n` ψ-wave neurons. Uses `f32x8` SIMD.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn psi_wave_step(
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
        let ie = load8_mut(i_ext, off).max(f32x8::splat(-100.0)).min(f32x8::splat(100.0));
        let sp = load8_mut(speed, off);

        let dv = hr - hi * sp + ie;
        let du = hi + hr * sp;
        vv += dt_s * dv / hbar_s;
        uu += dt_s * du / hbar_s;

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
            let dv = h_real[i] - h_imag[i] * speed[i] + i_ext[i].clamp(-100.0, 100.0);
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
