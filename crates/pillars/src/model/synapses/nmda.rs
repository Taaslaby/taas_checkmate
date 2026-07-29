#![allow(non_snake_case)]

//! NMDA Synaptic Channel with Voltage-Dependent Mg²⁺ Block — 8-wide SIMD (`f32x8`).
//!
//! Destexhe et al. (1998) model for non-linear NMDA dendritic integration:
//!   B(V) = 1 / (1 + ([Mg²⁺] / 3.57) · exp(-0.062 · V))
//!   I_NMDA = g_NMDA · B(V) · (V - E_syn)

use wide::f32x8;
use crate::model::util::{load8, store8};

/// Compute NMDA currents for n neurons with Mg²⁺ block in-place into I_out.
///
/// Params layout per neuron [4 × n]: [g_nmda, E_syn, mg_conc, _]
///
/// # Safety
/// All pointer arguments must point to valid memory for `n` elements each.
/// `n` may be any positive integer (SIMD handles chunks + scalar tail).
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

        let block = one_simd / (one_simd + (mg_conc * scale_mg) * (coeff_v * vv).exp());
        let i_nmda = g_nmda * block * (vv - E_syn);

        let current_i = load8(i_out, off);
        store8(i_out, off, current_i + i_nmda);
    }

    // Scalar tail
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmda_mg_block_voltage_dependence() {
        let n = 8;
        let v = vec![-65.0f32, -50.0, -35.0, -20.0, 0.0, 20.0, 40.0, 60.0];
        let g_act = vec![0.5f32; n];
        let mut i_out = vec![0.0f32; n];

        // 4 params per synapse: [g_nmda, E_syn, mg_conc, _]
        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.5;      // g_nmda
            params[i * 4 + 1] = 0.0;      // E_syn
            params[i * 4 + 2] = 1.0;      // mg_conc (mM)
            params[i * 4 + 3] = 0.0;      // unused
        }

        unsafe {
            nmda_mg_block_step(v.as_ptr(), g_act.as_ptr(), i_out.as_mut_ptr(), params.as_ptr(), n);
        }

        // At -65mV, Mg block is partial → small inward current (~-1.9)
        // At +60mV, Mg block is relieved → larger outward current (~+20)
        assert!(i_out[0] < 0.0 && i_out[7] > 0.0,
            "NMDA current reverses sign: V=-65mV gives {} (inward), V=+60mV gives {} (outward)",
            i_out[0], i_out[7]);
        assert!(i_out[0] > -5.0 && i_out[0] < 0.0,
            "NMDA current at -65mV: small inward (~-1.9), got {}", i_out[0]);
    }

    #[test]
    fn test_nmda_current_direction() {
        let n = 8;
        let v = vec![20.0f32; n]; // depolarized
        let g_act = vec![1.0f32; n];
        let mut i_out = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 1.0;
            params[i * 4 + 1] = 0.0;  // E_syn = 0mV
            params[i * 4 + 2] = 1.0;
        }

        unsafe {
            nmda_mg_block_step(v.as_ptr(), g_act.as_ptr(), i_out.as_mut_ptr(), params.as_ptr(), n);
        }

        // V > E_syn → outward (positive) current
        assert!(i_out[0] > 0.0, "NMDA current should be outward when V > E_syn: {}", i_out[0]);
    }

    #[test]
    fn test_nmda_simd_scalar_parity() {
        let n = 9; // 8 SIMD + 1 scalar tail
        let v = vec![-30.0f32; n];
        let g_act = vec![0.5f32; n];
        let mut i_out = vec![0.0f32; n];

        let mut params = vec![0.0f32; n * 4];
        for i in 0..n {
            params[i * 4]     = 0.5;
            params[i * 4 + 1] = 0.0;
            params[i * 4 + 2] = 1.0;
        }

        unsafe {
            nmda_mg_block_step(v.as_ptr(), g_act.as_ptr(), i_out.as_mut_ptr(), params.as_ptr(), n);
        }

        let diff = (i_out[0] - i_out[8]).abs();
        assert!(diff < 1e-5, "SIMD vs scalar NMDA current mismatch: {}", diff);
    }
}
