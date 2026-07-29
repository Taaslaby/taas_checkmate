use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// FitzHugh-Nagumo (FHN) Neuron Model
///
/// Features Horner's polynomial scheme, bitwise mask AND for edge detection, and mask NOT.
pub struct FHNModel;

crate::validate_model!(FHNModel {
    params: 5, state: 3, dt: 0,
    v_thresh: 4,
    forward_euler: true, edge_triggered: true
});


impl Model for FHNModel {
    /// Params: [0]dt, [1]inv_tau_w, [2]a, [3]b, [4]v_thresh
    const N_PARAMS: usize = 5;
    /// State: [0]v, [1]w, [2]was_below_thresh_flag (1.0 = true, 0.0 = false)
    const N_STATE: usize = 3;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let w = ctx.s(1);
        let was_below_flag = ctx.s(2);

        let inv_tau_w = ctx.p(1);
        let a = ctx.p(2);
        let b = ctx.p(3);
        let v_thresh = ctx.p(4);
        let one_third = S::splat(1.0 / 3.0);

        // Horner's method: v - v³/3 => v * (1 - v²/3)
        let v_poly = v * (S::one() - v * v * one_third);

        // Forward Euler
        let dv = (v_poly - w + ctx.i()) * ctx.dt();
        let dw = (v + a - b * w) * inv_tau_w * ctx.dt();

        let v_new = v + dv;
        let w_new = w + dw;

        let is_above = v_new.cmp_ge(v_thresh);
        let was_below = was_below_flag.cmp_gt(S::zero());

        // Fast bitwise AND instead of heavy blend instruction
        let spike = is_above.and(was_below);
        ctx.spike(spike);

        ctx.set_s(0, v_new);
        ctx.set_s(1, w_new);

        // Fast mask NOT inversion instead of second vector comparison instruction
        ctx.set_s(2, is_above.not().cond().then(S::one()).else_(S::zero()));
    }
    // ─── Stress tests ────────────────────────────────────────────
}
