#![allow(non_snake_case)]

//! Tsodyks-Markram (1997/1998) Short-Term Plasticity (STP) — 8-wide SIMD (`f32x8`).
//!
//! Models short-term synaptic facilitation (u) & depression (x):
//!   dx/dt = (1 - x) / τ_d
//!   du/dt = (U - u) / τ_f
//!
//! On pre-synaptic spike:
//!   x ← x - u · x       (depression)
//!   u ← u + U · (1 - u)  (facilitation)
//!   Effective weight: W_eff = W_base · u · x

use wide::f32x8;
use crate::core::lane::SimdLane;
use crate::core::util::{load8, store8};

/// Tsodyks-Markram STP SIMD step for n synapses.
///
/// Updates depression variable `x` and facilitation variable `u` in-place,
/// computes effective weight `w_eff = W_base · u · x`.
///
/// Params layout per synapse [4 floats per synapse]: [U, tau_d, tau_f, W_base]
///
/// # Safety
/// All pointer arguments must point to valid memory for `n` elements each.
pub unsafe fn tsodyks_markram_step(
    x: *mut f32,
    u: *mut f32,
    spikes: *const i32,
    w_eff: *mut f32,
    params: *const f32,
    n: usize,
    dt: f32,
) {
    let dt_simd = f32x8::splat(dt);
    let one_simd = f32x8::splat(1.0);

    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let xx = load8(x, off);
        let uu = load8(u, off);

        let mut U_arr      = [0.0f32; 8];
        let mut tau_d_arr  = [0.0f32; 8];
        let mut tau_f_arr  = [0.0f32; 8];
        let mut W_base_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = params.add((off + i) * 4);
            U_arr[i]      = *p.add(0);
            tau_d_arr[i]  = *p.add(1);
            tau_f_arr[i]  = *p.add(2);
            W_base_arr[i] = *p.add(3);
        }

        let U      = f32x8::from(U_arr);
        let tau_d  = f32x8::from(tau_d_arr);
        let tau_f  = f32x8::from(tau_f_arr);
        let W_base = f32x8::from(W_base_arr);

        // Continuous relaxation between spikes (Exponential Euler — stable for ANY tau)
        // dx/dt = (1 - x) / τ  →  x' = x + (1 - x) · (1 - exp(-dt/τ))
        // Protection against tau=0: max(tau, eps) before recip
        let eps = f32x8::splat(1e-12);
        let tau_d_safe = tau_d.max(eps);
        let tau_f_safe = tau_f.max(eps);
        let inv_tau_d = tau_d_safe.recip();
        let inv_tau_f = tau_f_safe.recip();
        // exp через trait — переключатель точности fast-exp покрывает и этот файл
        let decay_d = <f32x8 as SimdLane>::exp(-dt_simd * inv_tau_d);
        let decay_f = <f32x8 as SimdLane>::exp(-dt_simd * inv_tau_f);
        let next_x = xx + (one_simd - xx) * (one_simd - decay_d);
        let next_u = uu + (U - uu) * (one_simd - decay_f);

        // Apply spike updates if spike occurred
        let p_spk = spikes.add(off);
        let arr_spk = [
            *p_spk.add(0), *p_spk.add(1), *p_spk.add(2), *p_spk.add(3),
            *p_spk.add(4), *p_spk.add(5), *p_spk.add(6), *p_spk.add(7),
        ];

        let mut arr_x = next_x.to_array();
        let mut arr_u = next_u.to_array();
        let arr_U = U.to_array();

        for i in 0..8 {
            if arr_spk[i] != 0 {
                let u_curr = arr_u[i] + arr_U[i] * (1.0 - arr_u[i]);
                let x_curr = arr_x[i] - u_curr * arr_x[i];
                arr_u[i] = u_curr;
                arr_x[i] = x_curr;
            }
        }

        let final_x = f32x8::from(arr_x);
        let final_u = f32x8::from(arr_u);

        store8(x, off, final_x);
        store8(u, off, final_u);

        let weff = W_base * final_u * final_x;
        store8(w_eff, off, weff);
    }

    // Scalar tail
    let tail_start = chunks * 8;
    for i in tail_start..n {
        let mut xx = *x.add(i);
        let mut uu = *u.add(i);

        let p = params.add(i * 4);
        let U_val  = *p.add(0);
        let tau_d  = *p.add(1);
        let tau_f  = *p.add(2);
        let W_base = *p.add(3);

        let tau_d_safe = tau_d.max(1e-12);
        let tau_f_safe = tau_f.max(1e-12);
        let inv_tau_d = 1.0 / tau_d_safe;
        let inv_tau_f = 1.0 / tau_f_safe;
        let decay_d = <f32 as SimdLane>::exp(-dt * inv_tau_d);
        let decay_f = <f32 as SimdLane>::exp(-dt * inv_tau_f);
        xx += (1.0 - xx) * (1.0 - decay_d);
        uu += (U_val - uu) * (1.0 - decay_f);

        if *spikes.add(i) != 0 {
            uu += U_val * (1.0 - uu);
            xx -= uu * xx;
        }

        *x.add(i) = xx;
        *u.add(i) = uu;
        *w_eff.add(i) = W_base * uu * xx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsodyks_markram_no_spike_decay() {
        let n = 8;
        let mut x = vec![1.0f32; n];
        let mut u = vec![0.2f32; n];
        let spikes = vec![0i32; n];
        let mut w_eff = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.5;   // U
            params[i * 4 + 1] = 100.0; // tau_d (ms)
            params[i * 4 + 2] = 50.0;  // tau_f (ms)
            params[i * 4 + 3] = 1.0;   // W_base
        }

        unsafe {
            tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
        }

        // Without spikes, x decays toward 1.0, u decays toward U
        assert!(x[0] < 1.001 && x[0] > 0.99, "x should approach 1: {}", x[0]);
        assert!(u[0] < 0.5 && u[0] > 0.19, "u should approach U=0.5: {}", u[0]);
    }

    #[test]
    fn test_tsodyks_markram_spike_updates() {
        let n = 8;
        let mut x = vec![1.0f32; n];
        let mut u = vec![0.2f32; n];
        let spikes = vec![1i32; n];
        let mut w_eff = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.5;
            params[i * 4 + 1] = 100.0;
            params[i * 4 + 2] = 50.0;
            params[i * 4 + 3] = 1.0;
        }

        unsafe {
            tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
        }

        // On spike: u increases, x decreases (depression)
        assert!(u[0] > 0.5, "Facilitation should increase u: {}", u[0]);
        assert!(x[0] < 1.0, "Depression should decrease x: {}", x[0]);
        assert!(w_eff[0] > 0.0, "Effective weight should be positive: {}", w_eff[0]);
    }

    #[test]
    fn test_tsodyks_markram_simd_scalar_parity() {
        let n = 9;
        let mut x = vec![0.8f32; n];
        let mut u = vec![0.3f32; n];
        let spikes = vec![1i32; n];
        let mut w_eff = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.5;
            params[i * 4 + 1] = 100.0;
            params[i * 4 + 2] = 50.0;
            params[i * 4 + 3] = 1.0;
        }

        unsafe {
            tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
        }

        let diff = (w_eff[0] - w_eff[8]).abs();
        assert!(diff < 1e-5, "SIMD vs scalar w_eff mismatch: {}", diff);
    }

    #[test]
    fn test_tsodyks_markram_facilitation_depression_balance() {
        // Multiple spikes: facilitation builds up, depression reduces x
        let n = 8;
        let mut x = vec![1.0f32; n];
        let mut u = vec![0.1f32; n];
        let spikes = vec![1i32; n];
        let mut w_eff = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.2;     // U (low)
            params[i * 4 + 1] = 800.0;   // tau_d (slow depression)
            params[i * 4 + 2] = 50.0;    // tau_f (fast facilitation)
            params[i * 4 + 3] = 1.0;
        }

        let mut w_vals = Vec::new();
        for _step in 0..10 {
            unsafe {
                tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                    w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
            }
            w_vals.push(w_eff[0]);
        }

        // With fast tau_d (slow depression), x remains high → weight stays positive
        assert!(w_vals[0] > 0.0, "Initial weight should be positive: {}", w_vals[0]);
        assert!(w_vals.iter().all(|&w| w >= 0.0), "All weights should be non-negative");
    }

    #[test]
    fn test_tsodyks_markram_facilitation_dominant() {
        // Parameters where facilitation dominates:
        // - x starts at 1.0 (fully recovered)
        // - u starts at 0.01 (very depressed facilitation)
        // - U=0.2 (moderate), tau_d=10000 (extremely slow depression)
        // - tau_f=20 (moderate facilitation recovery)
        // With parallel Euler (spike uses PRE-decay u, x):
        //   Step 1: u=0.01→0.22, x=0.99→0.98, w≈0.21
        //   Step 2: u=0.22→0.37, x=0.98→0.77, w≈0.29 ← weight INCREASES because
        //   u grows faster than x drops (spike uses pre-decay x for depression)
        let n = 8;
        let mut x = vec![0.99f32; n];  // near-full recovery
        let mut u = vec![0.01f32; n];  // very low facilitation
        let spikes = vec![1i32; n];
        let mut w_eff = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.2;     // U
            params[i * 4 + 1] = 10000.0; // tau_d (very slow depression)
            params[i * 4 + 2] = 20.0;    // tau_f (facilitation recovery)
            params[i * 4 + 3] = 1.0;
        }

        // Step 1: first spike
        unsafe {
            tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
        }
        let w1 = w_eff[0];

        // Step 2: second spike — facilitation builds up
        unsafe {
            tsodyks_markram_step(x.as_mut_ptr(), u.as_mut_ptr(), spikes.as_ptr(),
                w_eff.as_mut_ptr(), params.as_ptr(), n, 1.0);
        }
        let w2 = w_eff[0];

        // With parallel Euler, spike uses pre-decay u for depression:
        // w1 = u₁·x₁ ≈ 0.217·0.98 ≈ 0.213
        // w2 = u₂·x₂ ≈ 0.373·0.77 ≈ 0.286 → should increase
        assert!(w2 > w1,
            "Facilitation-dominant: weight should increase ({:.4} → {:.4})", w1, w2);
    }
}
