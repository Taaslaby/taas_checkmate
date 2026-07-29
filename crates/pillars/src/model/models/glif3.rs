//! GLIF3 — Generalized LIF with after-spike current (AHP)
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: g += dt·(-g·inv_tau) — no gate_integrate
//! - **Unbounded AHP**: g_ahp accumulates without .min(1.0)
//! - **Ohm's Law**: i_total = I + g_leak·E_leak + g_ahp·E_ahp - (g_leak+g_ahp)·v
//!
//! Params (9): dt, g_leak, E_leak, v_thresh, v_reset, inv_tau_ahp, delta_g, E_ahp, tau_ref
//! State (3): v, g_ahp, refractory

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// GLIF3 — Generalized LIF with After-Spike Current
pub struct GLIF3Model;

crate::validate_model!(GLIF3Model {
    params: 9, state: 3, dt: 0,
    inv_tau: [5],
    v_thresh: 3,
    v_reset: 4,
    refractory_at_input: true, forward_euler: true, ohms_law: true
});


impl Model for GLIF3Model {
    const N_PARAMS: usize = 9;
    const N_STATE: usize = 3;  // v, g_ahp, refractory
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let g = ctx.s(1);  // g_ahp
        let refr = ctx.s(2);

        let zero = S::zero();
        let is_refr = refr.cmp_gt(zero).cond();

        // PRINCIPLE: Refractory clamp at INPUT — v = v_reset during refractory.
        let v = is_refr.then(ctx.p(4)).else_(v_raw);

        // GLIF3 dynamics: i_total = I + g_leak·E_leak + g_ahp·E_ahp - (g_leak+g_ahp)·v
        let i_total = (ctx.i() + ctx.p(1) * ctx.p(2) + g * ctx.p(7)) - (ctx.p(1) + g) * v;
        let v_new = v + i_total * ctx.dt();

        // Forward Euler for g_ahp: g += dt·(-g·inv_tau)
        // Unbounded — allows deep AHP adaptation in bursting
        let g_new = g + ctx.dt() * (-g * ctx.p(5));

        // Spike detection: v_reset < v_thresh guarantees false during refractory
        let spike = v_new.cmp_ge(ctx.p(3));
        ctx.spike(spike);

        // Reset on spike
        ctx.set_s(0, spike.cond().then(ctx.p(4)).else_(v_new));
        let g_bump = g_new + ctx.p(6);  // unbounded
        ctx.set_s(1, spike.cond().then(g_bump).else_(g_new));

        // Refractory counter
        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(8)).else_(refr_dec);
        ctx.set_s(2, refr_set.cmp_gt(zero).cond().then(refr_set).else_(zero));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let inv_tau_ahp = 1.0 / 50.0;
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[i] = 1.0;              // dt
            p[n+i] = 0.1;            // g_leak
            p[2*n+i] = -65.0;        // E_leak
            p[3*n+i] = -50.0;        // v_thresh
            p[4*n+i] = -65.0;        // v_reset
            p[5*n+i] = inv_tau_ahp;  // inv_tau_ahp
            p[6*n+i] = 0.01;         // delta_g
            p[7*n+i] = -80.0;        // E_ahp
            p[8*n+i] = tau_ref;      // tau_ref
        }
        p
    }

    #[test]
    fn test_glif3_basic_spiking() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "GLIF3 should spike with I=30");
        assert!(g[0] > 0.0, "g_ahp should accumulate");
    }

    /// GLIF3: During refractory, v clamped to v_reset.
    #[test]
    fn test_glif3_refractory_clamp() {
        let n = 8; let params = make_params(n, 5.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        unsafe { run_neuron::<GLIF3Model>(
            v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!((v[0] - (-65.0)).abs() < 0.1, "After spike v should be v_reset, got {:.2}", v[0]);
        assert!(spikes[0] == 1, "Should have spiked");

        // Still refractory, v must stay at v_reset
        unsafe { run_neuron::<GLIF3Model>(
            v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!((v[0] - (-65.0)).abs() < 0.1,
            "During refractory v clamped to v_reset, got {:.2}", v[0]);
    }

    // ─── Stress tests ────────────────────────────────────────────

    #[test]
    fn test_glif3_stress_silence() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!g[i].is_nan(), "g[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "GLIF3 silence: no spikes");
    }

    #[test]
    fn test_glif3_stress_explosion() {
        let n = 8; let mut params = make_params(n, 2.0);
        // Increase delta_g for faster AHP accumulation
        for i in 0..n { params[6*n+i] = 0.1; }
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite step {step}");
            }
        }
        assert!(spikes.iter().sum::<i32>() > 0, "GLIF3 explosion: should spike");
    }
}
