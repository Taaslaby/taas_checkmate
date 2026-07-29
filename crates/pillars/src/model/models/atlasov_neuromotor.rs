use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Atlasov Neuromotor (Biomechanical Muscle & Calcium Dynamics) Model
///
/// Features precalculated decay constants, Gaussian length-tension stability,
/// and clamped eccentric force scaling.
pub struct AtlasovNeuromotorModel;

crate::validate_model!(AtlasovNeuromotorModel {
    params: 9, state: 5, dt: 0,
    forward_euler: true
});


impl Model for AtlasovNeuromotorModel {
    /// Params: [0]dt, [1]d_ca, [2]ca_decay_const (exp(-dt/tau)), [3]eta_reset, [4]T_max, [5]Kd, [6]L_0, [7]inv_vmax, [8]inv_width
    const N_PARAMS: usize = 9;
    /// State: [0]ca (Calcium conc), [1]tension (Muscle Force), [2]reset_flag, [3]length, [4]velocity
    const N_STATE: usize = 5;
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let ca_raw = ctx.s(0);
        let reset_flag = ctx.s(2);
        let length = ctx.s(3);
        let velocity = ctx.s(4);

        let d_ca = ctx.p(1);
        let ca_decay_const = ctx.p(2); // Precalculated exp(-dt * inv_tau_pump)
        let eta_reset = ctx.p(3);
        let t_max = ctx.p(4);
        let kd = ctx.p(5);
        let l0 = ctx.p(6);
        let inv_vmax = ctx.p(7);
        let inv_width = ctx.p(8);

        // 1. Spike Input & Calcium Influx (Direct blend optimization)
        let spike_mask = ctx.i().cmp_gt(S::zero());
        let ca_influx = ca_raw + S::blend(spike_mask, d_ca, S::zero());

        // 2. Exponential Pump Clearance (Pure multiplication, no exp() in hot loop)
        let ca_decayed = ca_influx * ca_decay_const;

        // 3. Active Mechanical Reset
        let reset_mask = reset_flag.cmp_gt(S::zero());
        let reset_factor = S::one() - S::blend(reset_mask, eta_reset, S::zero());
        let ca_next = (ca_decayed * reset_factor).max(S::zero());

        // 4. Cooperative Saturation Tension (Hill-type Calcium Activation)
        let ca2 = ca_next * ca_next;
        let kd2 = kd * kd;
        let denom = (ca2 + kd2).max(S::splat(1e-8));
        let ca_factor = ca2 / denom;

        // 5. Gaussian Length-Tension Force Curve f(L) = exp(-diff²)
        let diff = (length - l0) * inv_width;
        let f_l = (-diff * diff).exp();

        // 6. Velocity-Tension Force Modifier f(V) with extreme stretch protection
        let is_conc = velocity.cmp_ge(S::zero()); // Shortening / concentric
        let c = S::splat(3.0); // Hill constant (~0.33)
        let v_rel = velocity * inv_vmax;

        // Concentric shortening
        let conc_denom = S::one() + c * v_rel;
        let conc = ((S::one() - v_rel) / conc_denom).max(S::zero());

        // Safe eccentric lengthening: clamp v_rel to prevent runaway stretch singularities
        let v_rel_clamped = v_rel.max(S::splat(-1.0));
        let ecc = S::one() + (-v_rel_clamped).min(S::splat(0.5));

        let f_v = S::blend(is_conc, conc, ecc);

        // Final Isometric/Isokinetic Tension
        let tension_next = t_max * ca_factor * f_l * f_v;

        // Save states
        ctx.set_s(0, ca_next);
        ctx.set_s(1, tension_next);
    }
}
