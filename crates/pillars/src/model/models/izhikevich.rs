//! Izhikevich Simple Model of Spiking Neurons
//!
//! # Principles
//! - **Refractory at input**: v clamped to reset BEFORE substepping
//! - **Substepping**: two 0.5·dt steps for Forward Euler stability
//! - **Horner**: v·(0.04·v + 5.0) + 140 — polynomial optimization
//! - **Per-type threshold**: ctx.p(6) from calibration v2
//!
//! Params: [0]dt, [1]a, [2]b, [3]c, [4]d, [5]tau_ref, [6]v_thresh
//! State: [0]v, [1]u, [2]refractory_counter

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Izhikevich Simple Model of Spiking Neurons
pub struct IzhikevichModel;

crate::validate_model!(IzhikevichModel {
    params: 7, state: 3, dt: 0,
    v_thresh: 6,
    v_reset: 3,
    refractory_at_input: true, forward_euler: true
});


impl Model for IzhikevichModel {
    const N_PARAMS: usize = 7;
    const N_STATE: usize = 3;  // v, u, refractory_counter
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let u = ctx.s(1);
        let refr = ctx.s(2);

        let zero = S::zero();
        let is_refr = refr.cmp_gt(zero).cond();

        // PRINCIPLE: Refractory clamp at INPUT.
        // During refractory, v is held at v_reset (c, p[3]).
        let v = is_refr.then(ctx.p(3)).else_(v_raw);

        // Substepping: two 0.5*dt steps for Forward Euler stability
        let half_dt = ctx.dt() * S::splat(0.5);

        // Sub-step 1: Horner v·(0.04·v + 5.0)
        let c04 = S::splat(0.04);
        let c5 = S::splat(5.0);
        let c140 = S::splat(140.0);
        let v_poly1 = v * (c04 * v + c5);
        let dv1 = (v_poly1 + c140 - u + ctx.i()) * half_dt;
        let v_mid = v + dv1;

        // Sub-step 2
        let v_poly2 = v_mid * (c04 * v_mid + c5);
        let dv2 = (v_poly2 + c140 - u + ctx.i()) * half_dt;
        let v_new = v_mid + dv2;

        // u update only when NOT in refractory
        let du = ctx.p(1) * (ctx.p(2) * v_new - u) * ctx.dt();
        let u_new = is_refr.then(u).else_(u + du);

        // Spike detection: v_reset < v_thresh prevents false spikes in refractory
        let spike = v_new.cmp_ge(ctx.p(6));
        ctx.spike(spike);

        // Reset on spike
        ctx.set_s(0, spike.cond().then(ctx.p(3)).else_(v_new));           // v = c
        ctx.set_s(1, spike.cond().then(u_new + ctx.p(4)).else_(u_new));   // u += d

        // Refractory timer
        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(5)).else_(refr_dec);
        ctx.set_s(2, refr_set.cmp_gt(zero).cond().then(refr_set).else_(zero));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 7];
        for i in 0..n {
            p[i] = 0.5;            // dt
            p[n+i] = 0.02;         // a
            p[2*n+i] = 0.2;        // b
            p[3*n+i] = -65.0;      // c (v_reset)
            p[4*n+i] = 8.0;        // d
            p[5*n+i] = 2.0;        // tau_ref
            p[6*n+i] = 30.0;       // v_thresh
        }
        p
    }

    #[test]
    fn test_izhikevich_spikes() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![10.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "Izhikevich should spike with I=10, got {}", total);
    }

    #[test]
    fn test_izhikevich_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for _ in 0..100 {
            unsafe { run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes with I=0");
    }

    // ─── Stress tests ────────────────────────────────────────────

    #[test]
    fn test_izhikevich_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!(!u[i].is_nan(), "u[{i}] NaN at step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes at I=0");
    }

    #[test]
    fn test_izhikevich_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite at step {step}");
            }
        }
        // Explosion test: stability only (runner clamps I to [-100,100],
        // Izhikevich v_thresh=30 may not be reached with clamped I)
    }
}
