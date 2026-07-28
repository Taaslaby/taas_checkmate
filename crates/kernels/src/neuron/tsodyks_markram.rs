//! Tsodyks-Markram (1997/1998) Short-Term Plasticity (STP) — 8-wide SIMD (`f32x8`).
//!
//! Models short-term synaptic facilitation (u) & depression (x):
//!   dx/dt = (1 - x) / τ_d
//!   du/dt = (U - u) / τ_f
//!
//! On spike:
//!   x ← x - u · x
//!   u ← u + U · (1 - u)
//!   Effective weight: W_eff = W_base · u · x

use wide::f32x8;
use super::{load8, store8};

/// Tsodyks-Markram STP SIMD step for n synapses.
///
/// Params layout per synapse [4 floats per synapse]: [U, tau_d, tau_f, W_base]
/// Outputs effective weights `w_eff` array.
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

        // Continuous relaxation between spikes
        let dx = (one_simd - xx) / tau_d;
        let du = (U - uu) / tau_f;

        let next_x = xx + dx * dt_simd;
        let next_u = uu + du * dt_simd;

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

    let tail_start = chunks * 8;
    for i in tail_start..n {
        let mut xx = *x.add(i);
        let mut uu = *u.add(i);

        let p = params.add(i * 4);
        let U_val  = *p.add(0);
        let tau_d  = *p.add(1);
        let tau_f  = *p.add(2);
        let W_base = *p.add(3);

        xx += ((1.0 - xx) / tau_d) * dt;
        uu += ((U_val - uu) / tau_f) * dt;

        if *spikes.add(i) != 0 {
            uu += U_val * (1.0 - uu);
            xx -= uu * xx;
        }

        *x.add(i) = xx;
        *u.add(i) = uu;
        *w_eff.add(i) = W_base * uu * xx;
    }
}
