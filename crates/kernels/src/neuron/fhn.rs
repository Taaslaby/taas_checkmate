#![allow(non_snake_case)]
use wide::{f32x8, CmpGt, CmpGe};
use super::{load8, load8_mut, store8, spike_mask_to_i32};

// ═══════════════════════════════════════════════════════
//  FitzHugh-Nagumo (FHN) — simplified Hodgkin-Huxley
// ═══════════════════════════════════════════════════════
//
// Equations (coupled ODEs):
//   dv/dt = v - v³/3 - w + I_ext
//   dw/dt = (v + a - b·w) / τ
//
// v: membrane potential (fast variable, cubic nullcline)
// w: recovery variable (slow variable, linear nullcline)
// a, b: shape parameters
// τ: time constant for recovery variable
//
// Spike detection: v crosses threshold from below (v ≥ v_thresh AND prev_v < v_thresh)
// or simply v ≥ v_thresh (simpler, used here for SIMD efficiency).
//
// This model is ~2x faster than Izhikevich and captures the essential
// excitability dynamics: resting → oscillatory → chaotic regimes.

/// Step `n` FitzHugh-Nagumo neurons. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_w, a, b (4 × n, column-major)
///   - dt: time step (ms), uniform across all neurons (params[0])
///   - tau_w: recovery time constant (ms)
///   - a: nullcline parameter (default 0.7)
///   - b: nullcline parameter (default 0.8)
///
/// # Safety
/// `v[n]`, `w[n]`, `Ii[n]`, `spikes[n]`, `params[4·n]` must be valid, non-overlapping.
pub(crate) unsafe fn fhn_step(
    v: *mut f32, w: *mut f32, Ii: *const f32, spikes: *mut i32,
    params: *const f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 4);
    let dt_val = p[0];  // dt uniform across neurons
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one_third = f32x8::splat(1.0 / 3.0);
    let v_thresh = f32x8::splat(0.5);  // default spike threshold
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut ww = load8_mut(w, off);
        let ii = load8(Ii, off);
        let tau_w = load8(params.wrapping_add(n), off);
        let aa = load8(params.wrapping_add(2 * n), off);
        let bb = load8(params.wrapping_add(3 * n), off);

        // FHN dynamics:
        // dv/dt = v - v³/3 - w + I
        let v3 = vv * vv * vv;
        let dv = vv - v3 * one_third - ww + ii;

        // dw/dt = (v + a - b·w) / τ
        // Use mask to avoid division by zero
        let mask_tau = tau_w.cmp_gt(zero);
        let inv_tau = mask_tau.blend(f32x8::splat(1.0) / tau_w.max(f32x8::splat(1e-10)), zero);
        let dw = (vv + aa - bb * ww) * inv_tau;

        // Forward Euler
        vv = vv + dt_s * dv;
        ww = ww + dt_s * dw;

        // Spike detection: v ≥ threshold
        let sm = vv.cmp_ge(v_thresh);

        store8(v, off, vv);
        store8(w, off, ww);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let w = std::slice::from_raw_parts_mut(w, n);
        let ii = std::slice::from_raw_parts(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        for i in n8..n {
            let tau_w = p[n + i];
            let a = p[2 * n + i];
            let b = p[3 * n + i];
            let v3 = v[i] * v[i] * v[i];
            let dv = v[i] - v3 / 3.0 - w[i] + ii[i];
            let dw = if tau_w > 0.0 { (v[i] + a - b * w[i]) / tau_w } else { 0.0 };
            v[i] += dt_val * dv;
            w[i] += dt_val * dw;
            spikes[i] = if v[i] >= 0.5 { 1 } else { 0 };
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// FHN: verify excitability — with sufficient I, v should oscillate.
    #[test]
    fn fhn_oscillates_with_current() {
        let n = 8;
        let dt = 0.01;
        let tau_w = 1.25;
        let a = 0.7;
        let b = 0.8;
        let I_val = 0.5;

        let mut v = vec![0.0f32; n];
        let mut w = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 4];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_w;
            params[2 * n + j] = a;
            params[3 * n + j] = b;
        }

        let mut spike_count = 0i32;
        for _ in 0..10000 {
            unsafe { fhn_step(v.as_mut_ptr(), w.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
            spike_count += spikes.iter().sum::<i32>();
        }
        assert!(spike_count > 0, "FHN should oscillate with I={I_val}");
    }

    /// FHN: verify no spikes with very low input (resting state).
    #[test]
    fn fhn_resting_no_spikes() {
        let n = 8;
        let dt = 0.01;
        let tau_w = 1.25;
        let a = 0.7;
        let b = 0.8;
        let I_val = 0.0;  // below excitability threshold

        let mut v = vec![0.0f32; n];
        let mut w = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 4];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_w;
            params[2 * n + j] = a;
            params[3 * n + j] = b;
        }

        for _ in 0..1000 {
            unsafe { fhn_step(v.as_mut_ptr(), w.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "FHN should not spike with I=0");
    }

    /// FHN: verify non-aligned n works (scalar tail).
    #[test]
    fn fhn_non_aligned_n() {
        let n = 13;
        let dt = 0.01;
        let tau_w = 1.25;
        let a = 0.7;
        let b = 0.8;
        let I_val = 0.5;

        let mut v = vec![0.0f32; n];
        let mut w = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 4];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_w;
            params[2 * n + j] = a;
            params[3 * n + j] = b;
        }

        let mut spike_count = 0i32;
        for _ in 0..10000 {
            unsafe { fhn_step(v.as_mut_ptr(), w.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
            spike_count += spikes.iter().sum::<i32>();
        }
        assert!(spike_count > 0, "FHN non-aligned should oscillate");
    }
}
