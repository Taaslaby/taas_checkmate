//! GLIF4 — Generalized LIF with threshold adaptation + after-spike current
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: g += dt·(-g·inv_tau); θ += dt·(-θ·inv_tau)
//! - **Unbounded**: g_ahp and θ accumulate without .min() clamps
//!
//! Params (11): dt, g_leak, E_leak, v_thresh_0, v_reset,
//!   inv_tau_ahp, delta_g, E_ahp, inv_tau_theta, delta_theta, tau_ref
//! State (4): v, theta, g_ahp, refractory

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// GLIF4 — GLIF with threshold adaptation + after-spike current
pub struct GLIF4Model;

crate::validate_model!(GLIF4Model {
    params: 11, state: 4, dt: 0,
    inv_tau: [5, 7],
    v_thresh: 3,
    v_reset: 4,
    refractory_at_input: true, forward_euler: true
});


impl Model for GLIF4Model {
    const N_PARAMS: usize = 11;
    const N_STATE: usize = 4;    // v, theta, g_ahp, refractory
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let th = ctx.s(1);  // adaptive threshold offset θ
        let g = ctx.s(2);   // g_ahp
        let refr = ctx.s(3);

        let is_refr = refr.cmp_gt(S::zero()).cond();

        // PRINCIPLE: Refractory clamp at INPUT — v = v_reset during refractory.
        let v = is_refr.then(ctx.p(4)).else_(v_raw);

        // Effective threshold = v_thresh_0 + θ
        let v_th_eff = ctx.p(3) + th;

        // GLIF4 dynamics
        let i_total = (ctx.i() + ctx.p(1) * ctx.p(2) + g * ctx.p(7)) - (ctx.p(1) + g) * v;
        let v_new = v + i_total * ctx.dt();

        // Forward Euler for θ and g
        let theta_decay = th + ctx.dt() * (-th * ctx.p(8));  // inv_tau_theta
        let g_decay = g + ctx.dt() * (-g * ctx.p(5));       // inv_tau_ahp

        // Spike detection
        let spike = v_new.cmp_ge(v_th_eff);
        ctx.spike(spike);

        // Reset on spike
        ctx.set_s(0, spike.cond().then(ctx.p(4)).else_(v_new));             // v = v_reset
        ctx.set_s(1, spike.cond().then(theta_decay + ctx.p(9)).else_(theta_decay)); // θ += Δθ
        ctx.set_s(2, spike.cond().then(g_decay + ctx.p(6)).else_(g_decay));         // g += Δg

        // Refractory counter
        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(10)).else_(refr_dec);
        ctx.set_s(3, refr_set.max(S::zero()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let inv_tau_ahp = 1.0 / 30.0;
        let inv_tau_theta = 1.0 / 30.0;
        let mut p = vec![0.0f32; n * 11];
        for i in 0..n {
            p[i] = 1.0;                // dt
            p[n+i] = 0.1;              // g_leak
            p[2*n+i] = -65.0;          // E_leak
            p[3*n+i] = -50.0;          // v_thresh_0
            p[4*n+i] = -65.0;          // v_reset
            p[5*n+i] = inv_tau_ahp;    // inv_tau_ahp
            p[6*n+i] = 0.01;           // delta_g
            p[7*n+i] = -80.0;          // E_ahp
            p[8*n+i] = inv_tau_theta;  // inv_tau_theta
            p[9*n+i] = 3.0;            // delta_theta
            p[10*n+i] = tau_ref;       // tau_ref
        }
        p
    }

    #[test]
    fn test_glif4_basic_spiking() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<GLIF4Model>(
                v.as_mut_ptr(), theta.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "GLIF4 should spike with I=30");
        assert!(theta[0] > 0.0, "theta should accumulate");
    }

    #[test]
    fn test_glif4_theta_decay() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut theta = vec![10.0; n];
        let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        // FE: theta += dt * (-theta * inv_tau) = 10 + 1*(-10/30) = 9.667
        unsafe { run_neuron::<GLIF4Model>(
            v.as_mut_ptr(), theta.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        let expected = 10.0 + 1.0 * (-10.0 / 30.0);
        assert!((theta[0] - expected).abs() < 0.02,
            "theta FE decay: got {:.4}, expected {:.4}", theta[0], expected);
    }

    #[test]
    fn test_glif4_non_aligned_n() {
        let n = 13; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<GLIF4Model>(
                v.as_mut_ptr(), theta.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "GLIF4 n=13 should spike, got {}", total);
        for i in 0..n {
            assert!(!v[i].is_nan(), "v[{i}] is NaN");
            assert!(!theta[i].is_nan(), "theta[{i}] is NaN");
            assert!(!g[i].is_nan(), "g[{i}] is NaN");
        }
    }

    // ─── Stress tests ────────────────────────────────────────────

    #[test]
    fn test_glif4_stress_silence() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF4Model>(
                v.as_mut_ptr(), theta.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!theta[i].is_nan(), "θ[{i}] NaN step {step}");
                assert!(!g[i].is_nan(), "g[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "GLIF4 silence: no spikes");
    }

    #[test]
    fn test_glif4_stress_explosion() {
        let n = 8; let mut params = make_params(n, 2.0);
        for i in 0..n { params[6*n+i] = 0.1; params[9*n+i] = 1.0; }
        let mut v = vec![-65.0; n]; let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF4Model>(
                v.as_mut_ptr(), theta.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite step {step}");
            }
        }
        assert!(spikes.iter().sum::<i32>() > 0, "GLIF4 explosion: should spike");
    }
}
