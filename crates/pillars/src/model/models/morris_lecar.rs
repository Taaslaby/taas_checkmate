//! Morris-Lecar (1981) — two-variable conductance-based model
//!
//! Equations:
//!   dV/dt = (I - g_L·(V-E_L) - g_Ca·m_∞·(V-E_Ca) - g_K·w·(V-E_K)) / C
//!   dw/dt = φ·(w_∞ - w)                    (τ_w = 1/φ approximation)
//!   m_∞ = sigmoid(V, V1, V2/2)             = 1/(1+exp(-2(V-V1)/V2))
//!   w_∞ = sigmoid(V, V3, V4/2)             = 1/(1+exp(-2(V-V3)/V4))
//!
//! # Ohm's Law Convention
//! All ionic currents use I = g·(V - E_rev), subtracted from I_inj.
//! This is the standard NEURON convention: dv = (I_inj - I_ion) / C
//!
//! Params (15 per neuron, column-major SoA):
//!   dt, inv_C, g_L, E_L, g_Ca, E_Ca, g_K, E_K, V1, V2, V3, V4, phi, v_thresh, v_reset

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Morris-Lecar Neuron Model
pub struct MorrisLecarModel;

crate::validate_model!(MorrisLecarModel {
    params: 15, state: 2, dt: 0,
    v_thresh: 13,
    v_reset: 14,
    erev: [(3,"E_L",-60.0), (5,"E_Ca",120.0), (7,"E_K",-84.0)],
    forward_euler: true,
    ohms_law: true
});


impl Model for MorrisLecarModel {
    /// Params SoA: [0]dt, [1]inv_C, [2]g_L, [3]E_L, [4]g_Ca, [5]E_Ca,
    ///   [6]g_K, [7]E_K, [8]V1, [9]V2, [10]V3, [11]V4, [12]phi, [13]v_thresh, [14]v_reset
    const N_PARAMS: usize = 15;
    const N_STATE: usize = 2;  // v, w (K activation)
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let w = ctx.s(1);

        // m_∞ = 1/(1+exp(-2(V-V1)/V2)) = sigmoid(V, V1, V2/2)
        let m_inf = S::sigmoid(v, ctx.pf(8), ctx.pf(9) * 0.5);

        // w_∞ = 1/(1+exp(-2(V-V3)/V4)) = sigmoid(V, V3, V4/2)
        let w_inf = S::sigmoid(v, ctx.pf(10), ctx.pf(11) * 0.5);

        // τ_w = 1/φ (ignoring cosh approximation for SIMD efficiency)
        let tau_w_inv = ctx.p(12);

        // 🧬 CORRECTED: Ohmic currents use I = g·(V - E_rev), subtracted from I_inj
        // (was g·(E_rev - V) which inverted sign of Ca²⁺ current)
        let i_leak = ctx.p(2) * (v - ctx.p(3));
        let i_ca   = ctx.p(4) * m_inf * (v - ctx.p(5));
        let i_k    = ctx.p(6) * w * (v - ctx.p(7));

        // dv/dt = (I_ext - I_leak - I_Ca - I_K) * inv_C — NEURON convention
        let dv = (ctx.i() - i_leak - i_ca - i_k) * ctx.p(1);
        let dw = (w_inf - w) * tau_w_inv;

        let v_new = v + dv * ctx.dt();
        let w_new = w + dw * ctx.dt();

        let spike = v_new.cmp_ge(ctx.p(13));
        ctx.spike(spike);
        ctx.set_s(0, S::blend(spike, ctx.p(14), v_new));
        ctx.set_s(1, w_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    #[test]
    fn test_ml_spikes_with_current() {
        let n = 8;
        let mut p = vec![0.0f32; n * 15];
        let dt = 1.0;
        for i in 0..n {
            p[i] = dt;               // dt
            p[n + i] = 1.0 / 20.0;   // inv_C (C=20)
            p[2*n + i] = 2.0;        // g_L
            p[3*n+i] = -60.0; // E_L
            p[4*n+i] = 4.4;   // g_Ca
            p[5*n+i] = 120.0; // E_Ca
            p[6*n+i] = 8.0;   // g_K
            p[7*n+i] = -84.0; // E_K
            p[8*n+i] = -1.2;  // V1
            p[9*n+i] = 18.0;  // V2
            p[10*n+i] = 2.0;  // V3
            p[11*n+i] = 30.0; // V4
            p[12*n+i] = 0.04; // phi
            p[13*n+i] = 0.0;  // v_thresh
            p[14*n+i] = -20.0; // v_reset
        }

        let mut v = vec![-60.0; n]; let mut w = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![100.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe { run_neuron::<MorrisLecarModel>(
                v.as_mut_ptr(), w.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), p.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "ML should spike with I=100");
    }
    // ─── Stress tests ────────────────────────────────────────────
}
