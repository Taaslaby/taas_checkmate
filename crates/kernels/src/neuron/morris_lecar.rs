//! Morris-Lecar (1981) neuron model — 8-wide SIMD (`f32x8`).
//!
//! Two-variable conductance-based model for barnacle muscle fibers & cortical oscillations:
//!   dV/dt = (I_ext - g_L·(V - E_L) - g_Ca·m_inf(V)·(V - E_Ca) - g_K·w·(V - E_K)) / C
//!   dw/dt = (w_inf(V) - w) / τ_w(V)
//!
//!   m_inf(V) = 0.5 · (1 + tanh((V - V1) / V2))
//!   w_inf(V) = 0.5 · (1 + tanh((V - V3) / V4))
//!   τ_w(V)   = 1 / (ϕ · cosh((V - V3) / (2·V4)))

use wide::{f32x8, CmpGt, CmpLt};
use super::{load8, store8};

#[inline(always)]
fn fast_tanh_simd(x: f32x8) -> f32x8 {
    let x_clamped = x.min(f32x8::splat(3.0)).max(f32x8::splat(-3.0));
    let x2 = x_clamped * x_clamped;
    let num = x_clamped * (f32x8::splat(27.0) + x2);
    let den = f32x8::splat(27.0) + f32x8::splat(9.0) * x2;
    let res = num / den;
    let mask_gt = x.cmp_gt(f32x8::splat(3.0));
    let mask_lt = x.cmp_lt(f32x8::splat(-3.0));
    mask_gt.blend(f32x8::splat(1.0), mask_lt.blend(f32x8::splat(-1.0), res))
}

#[inline(always)]
fn fast_cosh_simd(x: f32x8) -> f32x8 {
    let x_clamped = x.min(f32x8::splat(3.0)).max(f32x8::splat(-3.0));
    let x2 = x_clamped * x_clamped;
    let x4 = x2 * x2;
    f32x8::splat(1.0) + f32x8::splat(0.5) * x2 + f32x8::splat(0.04166667) * x4
}

/// Morris-Lecar SIMD step for n neurons.
///
/// Params layout (16 floats per neuron):
/// [C, g_L, E_L, g_Ca, E_Ca, g_K, E_K, V1, V2, V3, V4, phi, v_thresh, v_reset, _, _]
pub unsafe fn morris_lecar_step(
    v: *mut f32,
    w: *mut f32,
    Ii: *mut f32,
    spikes: *mut i32,
    params: *const f32,
    n: usize,
    dt: f32,
) {
    let dt_simd = f32x8::splat(dt);
    let half_simd = f32x8::splat(0.5);
    let one_simd = f32x8::splat(1.0);

    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let vv = load8(v, off);
        let ww = load8(w, off);
        let ii = load8(Ii, off);

        let mut C_arr    = [0.0f32; 8];
        let mut g_L_arr  = [0.0f32; 8];
        let mut E_L_arr  = [0.0f32; 8];
        let mut g_Ca_arr = [0.0f32; 8];
        let mut E_Ca_arr = [0.0f32; 8];
        let mut g_K_arr  = [0.0f32; 8];
        let mut E_K_arr  = [0.0f32; 8];
        let mut V1_arr   = [0.0f32; 8];
        let mut V2_arr   = [0.0f32; 8];
        let mut V3_arr   = [0.0f32; 8];
        let mut V4_arr   = [0.0f32; 8];
        let mut phi_arr  = [0.0f32; 8];
        let mut v_th_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = params.add((off + i) * 16);
            C_arr[i]    = *p.add(0);
            g_L_arr[i]  = *p.add(1);
            E_L_arr[i]  = *p.add(2);
            g_Ca_arr[i] = *p.add(3);
            E_Ca_arr[i] = *p.add(4);
            g_K_arr[i]  = *p.add(5);
            E_K_arr[i]  = *p.add(6);
            V1_arr[i]   = *p.add(7);
            V2_arr[i]   = *p.add(8);
            V3_arr[i]   = *p.add(9);
            V4_arr[i]   = *p.add(10);
            phi_arr[i]  = *p.add(11);
            v_th_arr[i] = *p.add(12);
        }

        let C_val    = f32x8::from(C_arr);
        let g_L      = f32x8::from(g_L_arr);
        let E_L      = f32x8::from(E_L_arr);
        let g_Ca     = f32x8::from(g_Ca_arr);
        let E_Ca     = f32x8::from(E_Ca_arr);
        let g_K      = f32x8::from(g_K_arr);
        let E_K      = f32x8::from(E_K_arr);
        let V1       = f32x8::from(V1_arr);
        let V2       = f32x8::from(V2_arr);
        let V3       = f32x8::from(V3_arr);
        let V4       = f32x8::from(V4_arr);
        let phi      = f32x8::from(phi_arr);
        let v_thresh = f32x8::from(v_th_arr);

        let m_inf = half_simd * (one_simd + fast_tanh_simd((vv - V1) / V2));
        let w_inf = half_simd * (one_simd + fast_tanh_simd((vv - V3) / V4));
        let tau_w = one_simd / (phi * fast_cosh_simd((vv - V3) / (f32x8::splat(2.0) * V4)));

        let I_leak = g_L * (vv - E_L);
        let I_Ca   = g_Ca * m_inf * (vv - E_Ca);
        let I_K    = g_K * ww * (vv - E_K);

        let dv = (ii - I_leak - I_Ca - I_K) / C_val;
        let dw = (w_inf - ww) / tau_w;

        let new_v = vv + dv * dt_simd;
        let new_w = ww + dw * dt_simd;

        let is_spike = new_v.cmp_gt(v_thresh);

        store8(v, off, new_v);
        store8(w, off, new_w);

        let p_spk = spikes.add(off);
        let arr_spk = is_spike.to_array();
        for i in 0..8 {
            *p_spk.add(i) = if arr_spk[i].to_bits() != 0 { 1 } else { 0 };
        }
    }

    let tail_start = chunks * 8;
    for i in tail_start..n {
        let vv = *v.add(i);
        let ww = *w.add(i);
        let ii = *Ii.add(i);

        let p = params.add(i * 16);
        let C_val    = *p.add(0);
        let g_L      = *p.add(1);
        let E_L      = *p.add(2);
        let g_Ca     = *p.add(3);
        let E_Ca     = *p.add(4);
        let g_K      = *p.add(5);
        let E_K      = *p.add(6);
        let V1       = *p.add(7);
        let V2       = *p.add(8);
        let V3       = *p.add(9);
        let V4       = *p.add(10);
        let phi      = *p.add(11);
        let v_thresh = *p.add(12);

        let m_inf = 0.5 * (1.0 + ((vv - V1) / V2).tanh());
        let w_inf = 0.5 * (1.0 + ((vv - V3) / V4).tanh());
        let tau_w = 1.0 / (phi * ((vv - V3) / (2.0 * V4)).cosh());

        let I_leak = g_L * (vv - E_L);
        let I_Ca   = g_Ca * m_inf * (vv - E_Ca);
        let I_K    = g_K * ww * (vv - E_K);

        let dv = (ii - I_leak - I_Ca - I_K) / C_val;
        let dw = (w_inf - ww) / tau_w;

        let new_v = vv + dv * dt;
        let new_w = ww + dw * dt;

        *v.add(i) = new_v;
        *w.add(i) = new_w;
        *spikes.add(i) = if new_v > v_thresh { 1 } else { 0 };
    }
}
