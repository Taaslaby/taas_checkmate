//! ⚡ HAAP — Hyperpolarization-Activated Adaptive Pulse-Bursting
//!
//! # Principles
//! - **Ohm's Law**: I = g·(V - E_rev), subtracted from I_inj
//! - **Forward Euler**: no gate_integrate, no hidden abstractions
//! - **Zero divisions**: all tau stored as inv_tau at init
//!
//! Equations:
//!   I_leak = g_L·(V - E_L)
//!   I_M    = g_M·u·(V - E_K)
//!   I_h    = g_h·h·(V - E_h)
//!   C_m·dV/dt = I_syn - I_leak - I_M - I_h
//!   du/dt = (u_∞(V) - u) / τ_u   (Forward Euler)
//!   dh/dt = (h_∞(V) - h) / τ_h   (Forward Euler)
//!   Spike: V ≥ V_thresh → V = V_reset, u += delta_u
//!
//! Params (17): [0]dt, [1]inv_C_m, [2]g_L, [3]E_L, [4]g_M, [5]E_K,
//!   [6]g_h, [7]E_h, [8]inv_tau_u, [9]inv_tau_h, [10]V_half_u, [11]k_u,
//!   [12]V_half_h, [13]k_h, [14]V_thresh, [15]V_reset, [16]delta_u

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

pub struct HAAPModel;

crate::validate_model!(HAAPModel {
    params: 17, state: 3, dt: 0,
    inv_tau: [8, 9],
    v_thresh: 14,
    v_reset: 15,
    erev: [(3,"E_L",-65.0), (5,"E_K",-80.0), (7,"E_h",-30.0)],
    refractory_at_input: false, forward_euler: true, ohms_law: true
});


impl Model for HAAPModel {
    const N_PARAMS: usize = 17;
    const N_STATE: usize = 3;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let u = ctx.s(1);
        let h = ctx.s(2);

        // Ohmic currents: I = g·(V - E), subtracted from I_inj
        let i_leak = ctx.p(2) * (v - ctx.p(3));
        let i_m    = ctx.p(4) * u * (v - ctx.p(5));
        let i_h    = ctx.p(6) * h * (v - ctx.p(7));

        let i_total = i_leak + i_m + i_h;
        let dv = (ctx.i() - i_total) * ctx.p(1);
        let v_new = v + dv * ctx.dt();

        // PRINCIPLE: Forward Euler for gating variables — no gate_integrate
        // u_∞ = sigmoid((V - V_half_u) / k_u)
        let u_inf = S::sigmoid(v_new, ctx.pf(10), ctx.pf(11));
        let u_new = u + ctx.dt() * (u_inf - u) * ctx.p(8);

        // h_∞ = sigmoid((V_half_h - V) / k_h)
        let h_inf = S::sigmoid(ctx.p(12) - v_new, 0.0, ctx.pf(13));
        let h_new = h + ctx.dt() * (h_inf - h) * ctx.p(9);

        // Clamp gates to [0, 1]
        let u_new = u_new.clamp01();
        let h_new = h_new.clamp01();

        // Spike detection
        let spike = v_new.cmp_ge(ctx.p(14));
        ctx.spike(spike);

        ctx.set_s(0, spike.cond().then(ctx.p(15)).else_(v_new));
        let u_bump = (u_new + ctx.p(16)).clamp01();
        ctx.set_s(1, spike.cond().then(u_bump).else_(u_new));
        ctx.set_s(2, h_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_vnc_params(n: usize, dt: f32) -> Vec<f32> {
        let inv_tau_u = 1.0 / 100.0;
        let inv_tau_h = 1.0 / 200.0;
        let mut p = vec![0.0f32; n * 17];
        for i in 0..n {
            p[i] = dt; p[n+i] = 1.0; p[2*n+i] = 0.1; p[3*n+i] = -65.0;
            p[4*n+i] = 0.5; p[5*n+i] = -80.0; p[6*n+i] = 0.3; p[7*n+i] = -30.0;
            p[8*n+i] = inv_tau_u; p[9*n+i] = inv_tau_h;
            p[10*n+i] = -35.0; p[11*n+i] = 5.0; p[12*n+i] = -75.0; p[13*n+i] = 6.0;
            p[14*n+i] = -40.0; p[15*n+i] = -55.0; p[16*n+i] = 0.02;
        }
        p
    }

    #[test] fn test_haap_spikes_with_current() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![15.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP should spike with I=15");
    }

    #[test] fn test_haap_silent_with_inhibition() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![-10.0; n]; let mut spikes = vec![0i32; n];
        for _ in 0..1000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HAAP should not spike with I=-10");
    }

    #[test] fn test_haap_non_aligned() {
        let n = 13; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![15.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP non-aligned should spike");
    }

    #[test] fn test_haap_stress_silence() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n]; let Ii = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!u[i].is_nan(), "u[{i}] NaN step {step}");
                assert!(!h[i].is_nan(), "h[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HAAP silence: no spikes");
    }

    #[test] fn test_haap_stress_explosion() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n]; let Ii = vec![1000.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite step {step}");
                assert!(u[i].is_finite(), "u[{i}] non-finite step {step}");
                assert!(h[i].is_finite(), "h[{i}] non-finite step {step}");
            }
        }
    }
}
