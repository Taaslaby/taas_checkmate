//! Atlasov Neuromotor (Biomechanical Muscle & Calcium Dynamics) Model
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **No spikes**: a muscle, not a neuron → `#[no_spike]`
//! - **Hill force-velocity**: branchless conc/ecc blend via masks
//! - **Ca²⁺-tension**: T = T_max · Ca²/(Ca²+Kd²) · f_l · f_v

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum AtlasParamIdx {
    DT = 0,           // ms
    D_Ca = 1,         // Ca²⁺ influx per spike
    Ca_Decay_Const = 2, // exp(-dt/tau_pump)
    Eta_Reset = 3,    // active reset factor
    T_Max = 4,        // max isometric tension
    Kd = 5,           // Ca²⁺ sensitivity (half-activation)
    L0 = 6,           // optimal muscle length
    Inv_Vmax = 7,     // 1/V_max (inverse max velocity)
    Inv_Width = 8,    // 1/width of length-tension curve
}

impl From<AtlasParamIdx> for usize {
    #[inline(always)] fn from(idx: AtlasParamIdx) -> Self { idx as usize }
}

/// Biomechanical muscle with Ca²⁺ dynamics + Hill force-velocity.
///
/// # States (5)
/// - `ca`: Ca²⁺ concentration
/// - `tension`: isometric tension
/// - `reset_flag`: active reset flag (1.0 = reset pump active)
/// - `length`: current muscle length
/// - `velocity`: shortening velocity (positive = concentric)
///
/// # Params (9)
/// - `dt`, `d_ca`, `ca_decay_const`, `eta_reset`, `t_max`,
///   `kd`, `l0`, `inv_vmax`, `inv_width`
#[derive(NeuronModel)]
#[no_spike]
pub struct AtlasovNeuromotorModel {
    #[state] pub ca: f32,
    #[state] pub tension: f32,
    #[state] pub reset_flag: f32,
    #[state] pub length: f32,
    #[state] pub velocity: f32,

    #[param] pub dt: f32,
    #[param] pub d_ca: f32,
    #[param] pub ca_decay_const: f32,
    #[param] pub eta_reset: f32,
    #[param] pub t_max: f32,
    #[param] pub kd: f32,
    #[param] pub l0: f32,
    #[param] pub inv_vmax: f32,
    #[param] pub inv_width: f32,
}

impl ComputeNeuron for AtlasovNeuromotorModel {
    type State<S: SimdLane> = AtlasovNeuromotorState<S>;
    type Params<S: SimdLane> = AtlasovNeuromotorParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut AtlasovNeuromotorState<S>,
        params: &AtlasovNeuromotorParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        let one = S::one();
        let zero = S::zero();

        // ─── Ca²⁺ influx on spike (input bus = spike mask) ─────────
        let spike_mask = ctx.i_syn.cmp_gt(zero);
        let ca_influx = state.ca + spike_mask.cond().then(params.d_ca).else_(zero);
        let ca_decayed = ca_influx * params.ca_decay_const;

        // ─── Active reset (Ca²⁺ pump) ──────────────────────────────
        let reset_mask = state.reset_flag.cmp_gt(zero);
        let reset_factor = one - reset_mask.cond().then(params.eta_reset).else_(zero);
        let ca_next = (ca_decayed * reset_factor).max(zero);

        // ─── Ca²⁺-tension relation: T ∝ Ca²/(Ca² + Kd²) ────────────
        let ca2 = ca_next * ca_next;
        let kd2 = params.kd * params.kd;
        let denom = (ca2 + kd2).max(S::splat(1e-8));
        let ca_factor = ca2 / denom;

        // ─── Length-tension: f_l = exp(−((L−L0)·inv_width)²) ───────
        let diff = (state.length - params.l0) * params.inv_width;
        let f_l = (-diff * diff).exp();

        // ─── Hill force-velocity (branchless conc/ecc blend) ───────
        let is_conc = state.velocity.cmp_ge(zero);
        let v_rel = state.velocity * params.inv_vmax;
        let c = S::splat(3.0);

        // Concentric: f = (1 − v)/(1 + 3·v)
        let conc_denom = one + c * v_rel;
        let conc = ((one - v_rel) / conc_denom).max(zero);

        // Eccentric: f = 1 − min(−v, 0.5)
        let v_rel_clamped = v_rel.max(S::splat(-1.0));
        let ecc = one + (-v_rel_clamped).min(S::splat(0.5));

        let f_v = is_conc.cond().then(conc).else_(ecc);

        // ─── Total tension ─────────────────────────────────────────
        let tension_next = params.t_max * ca_factor * f_l * f_v;

        state.ca = ca_next;
        state.tension = tension_next;
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[AtlasParamIdx::DT as usize * n + i] = 1.0;
            p[AtlasParamIdx::D_Ca as usize * n + i] = 0.1;
            p[AtlasParamIdx::Ca_Decay_Const as usize * n + i] = 0.9;
            p[AtlasParamIdx::Eta_Reset as usize * n + i] = 0.5;
            p[AtlasParamIdx::T_Max as usize * n + i] = 1.0;
            p[AtlasParamIdx::Kd as usize * n + i] = 0.2;
            p[AtlasParamIdx::L0 as usize * n + i] = 1.0;
            p[AtlasParamIdx::Inv_Vmax as usize * n + i] = 1.0;
            p[AtlasParamIdx::Inv_Width as usize * n + i] = 1.0;
        }
        p
    }

    #[test]
    fn test_atlasov_spike_triggers_ca() {
        let n = 8; let params = make_params(n);
        let mut ca = vec![0.0; n]; let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n]; let mut length = vec![1.0; n]; let mut vel = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1.0; n]; let mut spikes = vec![0i32; n];

        unsafe { run_neuron::<AtlasovNeuromotorModel>(
            ca.as_mut_ptr(), tension.as_mut_ptr(), reset.as_mut_ptr(),
            length.as_mut_ptr(), vel.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!(ca[0] > 0.0, "Ca should rise on spike: {:.4}", ca[0]);
    }

    #[test]
    fn test_atlasov_no_spike_no_ca() {
        let n = 8; let params = make_params(n);
        let mut ca = vec![0.0; n]; let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n]; let mut length = vec![1.0; n]; let mut vel = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        unsafe { run_neuron::<AtlasovNeuromotorModel>(
            ca.as_mut_ptr(), tension.as_mut_ptr(), reset.as_mut_ptr(),
            length.as_mut_ptr(), vel.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!(ca[0] < 0.01, "Ca should stay near 0 without spike: {:.4}", ca[0]);
    }
}
