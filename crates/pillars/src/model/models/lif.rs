//! Leaky Integrate-and-Fire (LIF) Neuron Model
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: no divisions, pure FMA arithmetic
//! - **Convention**: I_inj drives voltage directly, v_reset < v_thresh
//!
//! Params: [0]dt, [1]inv_tau_m, [2]v_rest, [3]v_thresh, [4]v_reset, [5]tau_ref
//! State: [0]v, [1]refractory_counter

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Leaky Integrate-and-Fire (LIF) Neuron Model
pub struct LIFModel;

crate::validate_model!(LIFModel {
    params: 6, state: 2, dt: 0,
    inv_tau: [1],
    v_thresh: 3, v_reset: 4,
    refractory_at_input: true,
    forward_euler: true,
});

impl Model for LIFModel {
    const N_PARAMS: usize = 6;
    const N_STATE: usize = 2;  // v, refractory_counter
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let refr = ctx.s(1);

        let zero = S::zero();
        let is_refr = refr.cmp_gt(zero).cond();

        // PRINCIPLE: Refractory clamp at INPUT — before any computation.
        // During refractory, v is held at v_reset (p[4]).
        // This prevents integration from a spike-peak baseline.
        let v = is_refr.then(ctx.p(4)).else_(v_raw);

        // Forward Euler: dv = (v_rest - v + I_syn) * inv_tau_m * dt
        let dv = (ctx.p(2) - v + ctx.i()) * ctx.p(1) * ctx.dt();
        let v_new = v + dv;

        // Spike detection: v_reset < v_thresh guarantees false during refractory
        let spike = v_new.cmp_ge(ctx.p(3));
        ctx.spike(spike);

        // Reset on spike: v = v_reset
        ctx.set_s(0, spike.cond().then(ctx.p(4)).else_(v_new));

        // Refractory timer: decrement, or set to tau_ref on spike
        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(5)).else_(refr_dec);
        ctx.set_s(1, refr_set.cmp_gt(zero).cond().then(refr_set).else_(zero));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 6];
        for i in 0..n {
            p[i] = 0.5;            // dt
            p[n+i] = 1.0 / 10.0;   // inv_tau_m
            p[2*n+i] = -70.0;      // v_rest
            p[3*n+i] = -54.0;      // v_thresh
            p[4*n+i] = -70.0;      // v_reset
            p[5*n+i] = 2.0;        // tau_ref
        }
        p
    }

    #[test]
    fn test_lif_basic_spiking() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..100 {
            unsafe { run_neuron::<LIFModel>(
                v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "LIF should spike with I=30, got {}", total);
    }

    #[test]
    fn test_lif_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for _ in 0..100 {
            unsafe { run_neuron::<LIFModel>(
                v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "LIF should not spike with I=0");
    }

    /// Stress: I=0 for 10k steps — v must stay at rest, no NaN.
    #[test]
    fn test_lif_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<LIFModel>(
                v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!((v[i] - (-70.0)).abs() < 1.0,
                    "v[{i}]={:.2} drifted from rest at step {step}", v[i]);
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes at I=0");
    }

    /// Stress: I=1000 for 10k steps — bounded voltage, no NaN.
    #[test]
    fn test_lif_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<LIFModel>(
                v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite at step {step}");
                assert!(v[i] < 100.0, "v[{i}]={:.1} above bound at step {step}", v[i]);
            }
        }
        assert!(spikes.iter().sum::<i32>() > 0, "Should spike with I=1000");
    }
}
