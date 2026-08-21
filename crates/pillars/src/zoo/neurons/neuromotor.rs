//! Neuromotor (Biomechanical Muscle & Calcium Dynamics) Model
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **No spikes**: a muscle, not a neuron → `#[no_spike]`
//! - **Hill force-velocity**: branchless conc/ecc blend via masks
//! - **Ca²⁺-tension**: T = T_max · Ca²/(Ca²+Kd²) · f_l · f_v

#![allow(clippy::module_inception)] // deliberate: `pub mod neuromotor` mirrors the file module

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

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
#[dual_neuron]
pub mod neuromotor {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    #[no_spike]
    pub struct NeuromotorModel {
        #[state]
        pub ca: f32,
        #[state]
        pub tension: f32,
        #[state]
        pub reset_flag: f32,
        #[state]
        pub length: f32,
        #[state]
        pub velocity: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub d_ca: f32,
        #[param]
        pub ca_decay_const: f32,
        #[param]
        pub eta_reset: f32,
        #[param]
        pub t_max: f32,
        #[param]
        pub kd: f32,
        #[param]
        pub l0: f32,
        #[param]
        pub inv_vmax: f32,
        #[param]
        pub inv_width: f32,
    }

    impl ComputeNeuron for NeuromotorModel {
        type State<S: SimdLane> = NeuromotorState<S>;
        type Params<S: SimdLane> = NeuromotorParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut NeuromotorState<S>,
            params: &NeuromotorParams<S>,
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
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::neuromotor::NeuromotorModel` etc.
pub use neuromotor::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[NeuromotorModelParamIdx::dt as usize * n + i] = 1.0;
            p[NeuromotorModelParamIdx::d_ca as usize * n + i] = 0.1;
            p[NeuromotorModelParamIdx::ca_decay_const as usize * n + i] = 0.9;
            p[NeuromotorModelParamIdx::eta_reset as usize * n + i] = 0.5;
            p[NeuromotorModelParamIdx::t_max as usize * n + i] = 1.0;
            p[NeuromotorModelParamIdx::kd as usize * n + i] = 0.2;
            p[NeuromotorModelParamIdx::l0 as usize * n + i] = 1.0;
            p[NeuromotorModelParamIdx::inv_vmax as usize * n + i] = 1.0;
            p[NeuromotorModelParamIdx::inv_width as usize * n + i] = 1.0;
        }
        p
    }

    #[test]
    fn test_neuromotor_spike_triggers_ca() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.0; n];
        let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n];
        let mut length = vec![1.0; n];
        let mut vel = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1.0; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron::<NeuromotorModel>(
                ca.as_mut_ptr(),
                tension.as_mut_ptr(),
                reset.as_mut_ptr(),
                length.as_mut_ptr(),
                vel.as_mut_ptr(),
                dummy.as_mut_ptr(),
                I.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(ca[0] > 0.0, "Ca should rise on spike: {:.4}", ca[0]);
    }

    #[test]
    fn test_neuromotor_no_spike_no_ca() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.0; n];
        let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n];
        let mut length = vec![1.0; n];
        let mut vel = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron::<NeuromotorModel>(
                ca.as_mut_ptr(),
                tension.as_mut_ptr(),
                reset.as_mut_ptr(),
                length.as_mut_ptr(),
                vel.as_mut_ptr(),
                dummy.as_mut_ptr(),
                I.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(
            ca[0] < 0.01,
            "Ca should stay near 0 without spike: {:.4}",
            ca[0]
        );
    }

    /// The calcium state is an exact first-order discrete pump, not an
    /// unconstrained accumulator. This test locks the documented recurrence:
    /// `ca_next = (ca + spike*d_ca) * exp(-dt/tau_pump)`.
    #[test]
    fn physics_calcium_recurrence_matches_exponential_pump() {
        let n = 8;
        let mut params = make_params(n);
        let decay = 0.9_f32;
        for i in 0..n {
            params[NeuromotorModelParamIdx::ca_decay_const as usize * n + i] = decay;
        }
        let mut ca = vec![0.0; n];
        let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n];
        let mut length = vec![1.0; n];
        let mut velocity = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let input = vec![1.0; n];
        let mut spikes = vec![0_i32; n];

        let mut expected = 0.0_f32;
        for _ in 0..32 {
            expected = (expected + 0.1) * decay;
            unsafe {
                run_neuron::<NeuromotorModel>(
                    ca.as_mut_ptr(),
                    tension.as_mut_ptr(),
                    reset.as_mut_ptr(),
                    length.as_mut_ptr(),
                    velocity.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    input.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            assert!(
                (ca[0] - expected).abs() < 2e-6,
                "ca={} expected={expected}",
                ca[0]
            );
            assert!(ca[0].is_finite() && ca[0] >= 0.0);
        }
    }

    #[test]
    fn physics_reset_reduces_calcium_without_changing_input() {
        let n = 8;
        let params = make_params(n);
        let mut ca_passive = vec![1.0; n];
        let mut ca_reset = vec![1.0; n];
        let mut tension_passive = vec![0.0; n];
        let mut tension_reset = vec![0.0; n];
        let mut reset_passive = vec![0.0; n];
        let mut reset_active = vec![1.0; n];
        let mut length_passive = vec![1.0; n];
        let mut length_reset = vec![1.0; n];
        let mut velocity_passive = vec![0.0; n];
        let mut velocity_reset = vec![0.0; n];
        let mut dummy_passive = vec![0.0; n];
        let mut dummy_reset = vec![0.0; n];
        let input = vec![0.0; n];
        let mut spikes = vec![0_i32; n];

        unsafe {
            run_neuron::<NeuromotorModel>(
                ca_passive.as_mut_ptr(),
                tension_passive.as_mut_ptr(),
                reset_passive.as_mut_ptr(),
                length_passive.as_mut_ptr(),
                velocity_passive.as_mut_ptr(),
                dummy_passive.as_mut_ptr(),
                input.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
            run_neuron::<NeuromotorModel>(
                ca_reset.as_mut_ptr(),
                tension_reset.as_mut_ptr(),
                reset_active.as_mut_ptr(),
                length_reset.as_mut_ptr(),
                velocity_reset.as_mut_ptr(),
                dummy_reset.as_mut_ptr(),
                input.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(ca_reset[0] < ca_passive[0]);
        assert!(ca_reset[0] >= 0.0);
    }

    /// Length and velocity are external kinematic inputs. The active force
    /// must peak at optimal length, be bounded by the eccentric plateau, and
    /// remain finite for extreme but valid kinematics.
    #[test]
    fn physics_force_length_velocity_is_bounded_and_shaped() {
        let n = 8;
        let params = make_params(n);
        let lengths = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 1.0];
        let velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0];
        let mut ca = vec![1.0; n];
        let mut tension = vec![0.0; n];
        let mut reset = vec![0.0; n];
        let mut length = lengths.to_vec();
        let mut velocity = velocities.to_vec();
        let mut dummy = vec![0.0; n];
        let input = vec![0.0; n];
        let mut spikes = vec![0_i32; n];

        unsafe {
            run_neuron::<NeuromotorModel>(
                ca.as_mut_ptr(),
                tension.as_mut_ptr(),
                reset.as_mut_ptr(),
                length.as_mut_ptr(),
                velocity.as_mut_ptr(),
                dummy.as_mut_ptr(),
                input.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(tension[3] > tension[2] && tension[3] > tension[4]);
        assert!(
            tension[7] <= 1.5 + 2e-6,
            "eccentric plateau exceeded: {}",
            tension[7]
        );
        for (i, &value) in tension.iter().enumerate() {
            assert!(value.is_finite(), "tension[{i}] is not finite");
            assert!((0.0..=1.5 + 2e-6).contains(&value), "tension[{i}]={value}");
        }

        // At optimal length and fixed calcium, positive shortening velocity
        // lowers force; negative lengthening velocity raises it up to 1.5x.
        let mut tension_velocity = vec![0.0; n];
        let mut velocity_sweep = vec![-2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0];
        let mut ca_velocity = vec![1.0; n];
        let mut reset_velocity = vec![0.0; n];
        let mut length_velocity = vec![1.0; n];
        let mut dummy_velocity = vec![0.0; n];
        unsafe {
            run_neuron::<NeuromotorModel>(
                ca_velocity.as_mut_ptr(),
                tension_velocity.as_mut_ptr(),
                reset_velocity.as_mut_ptr(),
                length_velocity.as_mut_ptr(),
                velocity_sweep.as_mut_ptr(),
                dummy_velocity.as_mut_ptr(),
                input.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(tension_velocity[0] > tension_velocity[2]);
        assert!(tension_velocity[2] > tension_velocity[3]);
        assert!(tension_velocity[3] > tension_velocity[4]);
        assert_eq!(tension_velocity[0].to_bits(), tension_velocity[1].to_bits());
        assert_eq!(tension_velocity[6].to_bits(), tension_velocity[7].to_bits());
    }
}
