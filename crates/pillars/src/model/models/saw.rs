//! SAW (Spike-Adaptive-Width / ψ-wave) — Izhikevich-like neuron.
//!
//! Params (8): dt, a, b, trace_decay, noise_amp, tau_ref, v_thresh, v_reset
//! State (4): v, u, trace, refractory

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

pub struct SAWModel;

crate::validate_model!(SAWModel {
    params: 8, state: 4, dt: 0,
    v_thresh: 6,
    v_reset: 7,
    refractory_at_input: true, forward_euler: true
});


impl Model for SAWModel {
    const N_PARAMS: usize = 8;
    const N_STATE: usize = 4;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let u = ctx.s(1);
        let tr = ctx.s(2);
        let refr = ctx.s(3);

        let zero = S::zero();
        let one = S::one();
        let is_refr = refr.cmp_gt(zero).cond();
        let v_reset = ctx.p(7);

        let v = is_refr.then(v_reset).else_(v_raw);

        let c04 = S::splat(0.04);
        let c5 = S::splat(5.0);
        let c140 = S::splat(140.0);
        let half_dt = ctx.dt() * S::splat(0.5);

        let v_poly1 = v * (c04 * v + c5);
        let dv1 = (v_poly1 + c140 - u + ctx.i()) * half_dt;
        let v_mid = v + dv1;

        let v_poly2 = v_mid * (c04 * v_mid + c5);
        let dv2 = (v_poly2 + c140 - u + ctx.i()) * half_dt;
        let v_new = v_mid + dv2;

        let du = ctx.p(1) * (ctx.p(2) * v - u);
        let u_new = is_refr.then(u).else_(u + du * ctx.dt());

        let spike_raw = v_new.cmp_ge(ctx.p(6));
        let spike = is_refr.then(S::zero()).else_(spike_raw);
        ctx.spike(spike);

        ctx.set_s(0, spike.cond().then(v_reset).else_(v_new));
        ctx.set_s(1, u_new + spike.cond().then(S::splat(8.0)).else_(S::zero()));

        let dec = ctx.p(3);
        let tr_inc = tr * (one - dec) + dec;
        let tr_dec = tr * (one - dec);
        let tr_new = spike.cond().then(tr_inc).else_(tr_dec);
        ctx.set_s(2, tr_new.max(zero).min(one));

        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(5)).else_(refr_dec);
        ctx.set_s(3, refr_set.max(zero));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let mut p = vec![0.0f32; n * SAWModel::N_PARAMS];
        for i in 0..n {
            p[i] = 1.0; p[n+i] = 0.02; p[2*n+i] = 0.2;
            p[3*n+i] = 0.01; p[4*n+i] = 0.0; p[5*n+i] = tau_ref;
            p[6*n+i] = 30.0; p[7*n+i] = -65.0; // v_thresh, v_reset
        }
        p
    }

    #[test] fn test_saw_regular_spiking() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![15.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<SAWModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), tr.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0);
    }

    #[test] fn test_saw_silent_without_current() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for _ in 0..500 {
            unsafe { run_neuron::<SAWModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), tr.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test] fn test_saw_non_aligned_n() {
        let n = 11; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![30.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe { run_neuron::<SAWModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), tr.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0);
        for i in 0..n { assert!(!v[i].is_nan()); assert!(!u[i].is_nan()); }
    }

    // ─── Stress tests ────────────────────────────────────────────

    #[test] fn test_saw_stress_silence() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<SAWModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), tr.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n { assert!(!v[i].is_nan(), "v[{i}] NaN {step}"); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test] fn test_saw_stress_explosion() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![1000.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<SAWModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), tr.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite {step}");
            }
        }
    }
}
