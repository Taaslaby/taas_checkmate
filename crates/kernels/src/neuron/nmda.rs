//! NMDA Synaptic Channel with Voltage-Dependent Mg²⁺ Block — 8-wide SIMD (`f32x8`).
//!
//! Destexhe et al. (1998) model for non-linear NMDA dendritic integration:
//!   B(V) = 1 / (1 + ([Mg²⁺] / 3.57) · exp(-0.062 · V))
//!   I_NMDA = g_NMDA · B(V) · (V - E_syn)

use wide::f32x8;
use super::{load8, store8, fast_exp_simd};

/// Compute NMDA currents for n neurons with Mg²⁺ block in-place or into I_out.
///
/// Params layout per neuron [4 × n]: [g_nmda, E_syn, mg_conc, _]
pub unsafe fn nmda_mg_block_step(
    v: *const f32,
    g_act: *const f32,
    i_out: *mut f32,
    params: *const f32,
    n: usize,
) {
    let one_simd = f32x8::splat(1.0);
    let scale_mg = f32x8::splat(1.0 / 3.57);
    let coeff_v = f32x8::splat(-0.062);

    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let vv = load8(v, off);
        let g_nmda = load8(g_act, off);

        let mut E_syn_arr   = [0.0f32; 8];
        let mut mg_conc_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = params.add((off + i) * 4);
            E_syn_arr[i]   = *p.add(1);
            mg_conc_arr[i] = *p.add(2);
        }

        let E_syn   = f32x8::from(E_syn_arr);
        let mg_conc = f32x8::from(mg_conc_arr);

        let block = one_simd / (one_simd + (mg_conc * scale_mg) * fast_exp_simd(coeff_v * vv));
        let i_nmda = g_nmda * block * (vv - E_syn);

        let current_i = load8(i_out, off);
        store8(i_out, off, current_i + i_nmda);
    }

    let tail_start = chunks * 8;
    for i in tail_start..n {
        let vv = *v.add(i);
        let g_nmda = *g_act.add(i);
        let E_syn = *params.add(i * 4 + 1);
        let mg_conc = *params.add(i * 4 + 2);

        let block = 1.0 / (1.0 + (mg_conc / 3.57) * (-0.062 * vv).exp());
        let i_nmda = g_nmda * block * (vv - E_syn);

        *i_out.add(i) += i_nmda;
    }
}
