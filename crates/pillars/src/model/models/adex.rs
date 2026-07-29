use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Adaptive Exponential Integrate-and-Fire (AdEx) Neuron Model
///
/// Features pre-inverted inv_delta_t, early refractory clamping to prevent exponential explosion,
/// and peak detection threshold.
pub struct AdExModel;

crate::validate_model!(AdExModel {
    params: 12, state: 3, dt: 0,
    inv_tau: [1, 2],
    v_thresh: 4,
    v_reset: 5,
    refractory_at_input: true, forward_euler: true
});


impl Model for AdExModel {
    /// Params: [0]dt, [1]inv_tau_m, [2]inv_tau_w, [3]v_rest, [4]v_thresh, [5]v_reset, [6]delta_t, [7]inv_delta_t (1/delta_t), [8]a, [9]b, [10]tau_ref, [11]v_peak
    const N_PARAMS: usize = 12;
    /// State: [0]v, [1]w (adaptation variable), [2]refractory_counter
    const N_STATE: usize = 3;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let w = ctx.s(1);
        let refr = ctx.s(2);

        let inv_tau_m = ctx.p(1);
        let inv_tau_w = ctx.p(2);
        let v_rest = ctx.p(3);
        let v_thresh = ctx.p(4);
        let v_reset = ctx.p(5);
        let delta_t = ctx.p(6);
        let inv_delta_t = ctx.p(7); // Pre-inverted 1.0 / delta_t
        let a = ctx.p(8);
        let b = ctx.p(9);
        let tau_ref = ctx.p(10);
        let v_peak = ctx.p(11); // Spike peak threshold (e.g. 0.0 mV)

        // CRITICAL FIX: Clamp v to v_reset IMMEDIATELY if in refractory.
        // Prevents exponent explosion inside the refractory period!
        let is_refr = refr.cmp_gt(S::zero()).cond();
        let v = is_refr.then(v_reset).else_(v_raw);

        // Pure SIMD multiplication instead of vector division
        let exp_arg = (v - v_thresh) * inv_delta_t;
        let exp_val = exp_arg.max(S::splat(-10.0)).min(S::splat(10.0)).exp();
        let exp_term = delta_t * exp_val;

        // dv = (v_rest - v + exp_term - w + I_syn) * inv_tau_m * dt
        let dv = (v_rest - v + exp_term - w + ctx.i()) * inv_tau_m * ctx.dt();
        let v_new = v + dv;

        // Adaptation dw calculated on stable v
        let dw = (a * (v - v_rest) - w) * inv_tau_w * ctx.dt();
        let w_new = w + dw;

        // Spike detection at peak threshold v_peak (v_peak > v_thresh)
        let spike = v_new.cmp_ge(v_peak);
        ctx.spike(spike);

        // Reset state on spike
        ctx.set_s(0, spike.cond().then(v_reset).else_(v_new));
        ctx.set_s(1, spike.cond().then(w_new + b).else_(w_new));

        // Refractory counter update
        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(tau_ref).else_(refr_dec);
        ctx.set_s(2, refr_set.cmp_gt(S::zero()).cond().then(refr_set).else_(S::zero()));
    }
    // ─── Stress tests ────────────────────────────────────────────
}
