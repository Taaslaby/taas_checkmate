//! Morris-Lecar (1981) — two-variable conductance-based model
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` + `ops::*` for typed API.
//!
//! # Equations
//!
//! Membrane potential:
//! ```text
//! C·dV/dt = I_inj - g_L·(V-E_L) - g_Ca·m_∞·(V-E_Ca) - g_K·w·(V-E_K)
//! ```
//!
//! Ca²⁺ activation (instantaneous):
//! ```text
//! m_∞ = 0.5·(1 + tanh((V - V1) / V2)) = 1 / (1 + exp(-2·(V - V1) / V2))
//! ```
//!
//! K⁺ activation (slow, with voltage-dependent time constant):
//! ```text
//! w_∞     = 0.5·(1 + tanh((V - V3) / V4)) = 1 / (1 + exp(-2·(V - V3) / V4))
//! τ_w(V)  = 1 / cosh((V - V3) / (2·V4))
//! dw/dt   = φ · (w_∞ - w) / τ_w(V) = φ · (w_∞ - w) · cosh((V - V3) / (2·V4))
//! ```
//!
//! # Biological Notes
//! - Morris-Lecar is a **conductance-based** model. Spikes emerge naturally from
//!   channel dynamics (Ca²⁺ activation → depolarization → K⁺ activation → repolarization).
//! - There is **NO artificial reset** — `v` and `w` evolve continuously through the
//!   action potential, exactly as in the original 1981 paper.
//! - `v_thresh` is used ONLY for spike detection output (routing to downstream
//!   synapses), NOT for state modification.
//! - Edge-triggered spike detection prevents multiple firing per AP.
//!
//! # Principles
//! - Conductance-based: all ionic currents via `ops::ohmic_current` / `ops::ohmic_gated`
//! - Membrane integration via `ops::membrane_depolarize` (NEURON convention)
//! - Gate kinetics via pure SIMD sigmoids (no scalar `pf()` hack)
//! - τ_w(V) computed from cosh — no division in hot path

#![allow(non_snake_case)]
#![allow(clippy::module_inception)] // deliberate: `pub mod morris_lecar` mirrors the file module

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum MLParamIdx {
    DT = 0,
    Inv_C = 1,
    G_L = 2,
    E_L = 3,
    G_Ca = 4,
    E_Ca = 5,
    G_K = 6,
    E_K = 7,
    V1 = 8,
    V2 = 9,
    V3 = 10,
    V4 = 11,
    Phi = 12,
    V_Thresh = 13,
}

impl From<MLParamIdx> for usize {
    #[inline(always)]
    fn from(idx: MLParamIdx) -> Self {
        idx as usize
    }
}

/// Morris-Lecar (1981) two-variable conductance-based model.
///
/// # States (3)
/// - `v`: membrane potential (mV)
/// - `w`: K⁺ activation gate
/// - `was_below`: edge-detection flag (1.0 = was below v_thresh last step)
///
/// # Params (14)
/// - `dt`, `inv_C`, `g_L`, `E_L`, `g_Ca`, `E_Ca`, `g_K`, `E_K`,
///   `V1`, `V2`, `V3`, `V4`, `phi`, `v_thresh`
///
/// (v_reset не нужен: у conductance-based модели нет искусственного сброса —
/// спайк и реполяризация возникают естественно через динамику каналов.)
#[dual_neuron]
pub mod morris_lecar {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use crate::ops::{membrane_depolarize, ohmic_current, ohmic_gated};
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct MorrisLecarModel {
        #[state]
        pub v: f32,
        #[state]
        pub w: f32,
        #[state]
        pub was_below: f32, // edge-triggered spike detection

        #[param]
        pub dt: f32,
        #[param]
        pub inv_C: f32,
        #[param]
        pub g_L: f32,
        #[param]
        pub E_L: f32,
        #[param]
        pub g_Ca: f32,
        #[param]
        pub E_Ca: f32,
        #[param]
        pub g_K: f32,
        #[param]
        pub E_K: f32,
        #[param]
        pub V1: f32,
        #[param]
        pub V2: f32,
        #[param]
        pub V3: f32,
        #[param]
        pub V4: f32,
        #[param]
        pub phi: f32,
        #[param]
        pub v_thresh: f32,
    }

    impl ComputeNeuron for MorrisLecarModel {
        type State<S: SimdLane> = MorrisLecarState<S>;
        type Params<S: SimdLane> = MorrisLecarParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let one = S::one();
            let two = S::splat(2.0);
            let half = S::splat(0.5);

            // ─── Gate Kinetics ───────────────────────────────────────
            // m_∞ = 0.5·(1 + tanh((V - V1) / V2)) = 1/(1 + exp(-2(V-V1)/V2))
            let m_arg = -two * (state.v - params.V1) / params.V2;
            let m_inf = one / (one + m_arg.exp());

            // w_∞ = 0.5·(1 + tanh((V - V3) / V4)) = 1/(1 + exp(-2(V-V3)/V4))
            let w_arg = -two * (state.v - params.V3) / params.V4;
            let w_inf = one / (one + w_arg.exp());

            // ─── Ionic Currents (ops::ohmic, NEURON convention) ──────
            // I = g·(V - E_rev) — subtracted from I_inj in membrane_depolarize
            let i_leak = ohmic_current(params.g_L, state.v, params.E_L);
            let i_ca = ohmic_gated(params.g_Ca, state.v, params.E_Ca, m_inf);
            let i_k = ohmic_gated(params.g_K, state.v, params.E_K, state.w);

            // ─── Membrane Integration (ops::membrane) ─────────────────
            // V' = V + (I_inj - ΣI_ion) · inv_C · dt   (Forward Euler)
            let v_new = membrane_depolarize(
                state.v,
                ctx.i_syn,
                i_leak + i_ca + i_k,
                params.inv_C,
                ctx.dt_simd(),
            );

            // ─── K⁺ Gate Integration ─────────────────────────────────
            // τ_w(V) = 1 / cosh((V - V3) / (2·V4))
            // dw/dt = φ · (w_∞ - w) · cosh((V - V3) / (2·V4))
            //        = φ · (w_∞ - w) / τ_w(V)
            // cosh(x) = (e^x + e^-x)/2 — flat expression (no block: the
            // #[dual_neuron] transpiler is expression-only).
            let cosh_arg = (state.v - params.V3) / (two * params.V4);
            let cosh_x = (cosh_arg.exp() + (-cosh_arg).exp()) * half;
            let dw = params.phi * (w_inf - state.w) * cosh_x;
            let w_new = state.w + dw * ctx.dt_simd();

            // ─── Spike Detection (Edge-Triggered, No State Reset) ───
            // ❌ НЕ сбрасываем v — conductance-based model создаёт AP
            //    естественно через динамику каналов.
            // ✅ Только edge-triggered выход для downstream синапсов.
            let is_above = v_new.cmp_ge(params.v_thresh);
            let was_below = state.was_below.cmp_gt(zero);
            let spike = is_above.and(was_below); // rising edge only
            ctx.emit_spike(spike);

            // ─── State Update (No Reset!) ────────────────────────────
            state.v = v_new; // natural dynamics
            state.w = w_new; // continuous K⁺ gate
            state.was_below = S::blend(is_above, zero, one);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::morris_lecar::MorrisLecarModel` etc.
pub use morris_lecar::*;

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * MorrisLecarModel::N_PARAMS];
        for i in 0..n {
            p[MLParamIdx::DT as usize * n + i] = 1.0;
            p[MLParamIdx::Inv_C as usize * n + i] = 1.0 / 20.0;
            p[MLParamIdx::G_L as usize * n + i] = 2.0;
            p[MLParamIdx::E_L as usize * n + i] = -60.0;
            p[MLParamIdx::G_Ca as usize * n + i] = 4.4;
            p[MLParamIdx::E_Ca as usize * n + i] = 120.0;
            p[MLParamIdx::G_K as usize * n + i] = 8.0;
            p[MLParamIdx::E_K as usize * n + i] = -84.0;
            p[MLParamIdx::V1 as usize * n + i] = -1.2;
            p[MLParamIdx::V2 as usize * n + i] = 18.0;
            p[MLParamIdx::V3 as usize * n + i] = 2.0;
            p[MLParamIdx::V4 as usize * n + i] = 30.0;
            p[MLParamIdx::Phi as usize * n + i] = 0.04;
            p[MLParamIdx::V_Thresh as usize * n + i] = 0.0;
        }
        p
    }

    /// Morris-Lecar должна генерировать спайки при I=100.
    /// С edge-triggered детекцией каждый AP даёт ровно 1 спайк.
    #[test]
    fn test_ml_spikes_with_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut w = vec![0.0; n];
        let mut was_below = vec![1.0; n]; // 1.0 = was below thresh
        let mut dummy = vec![0.0; n];
        let I = vec![100.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<MorrisLecarModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    was_below.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "ML should spike with I=100, got {total}");
    }

    #[test]
    fn test_ml_population_spiking() {
        let mut pop = NeuralPopulation::<MorrisLecarModel>::new(8);
        pop.fill_state(0, -60.0); // v
        pop.fill_state(1, 0.0); // w
        pop.fill_state(2, 1.0); // was_below
        for idx in 0..MorrisLecarModel::N_PARAMS {
            pop.fill_param(idx, make_params(1)[idx]);
        }
        let i_in = vec![100.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..1000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "ML pop should spike with I=100, got {total}");
    }

    #[test]
    fn test_ml_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut w = vec![0.0; n];
        let mut was_below = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for _ in 0..1000 {
            unsafe {
                run_neuron::<MorrisLecarModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    was_below.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "ML should be silent at I=0");
    }

    #[test]
    fn test_ml_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut w = vec![0.0; n];
        let mut was_below = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe {
                run_neuron::<MorrisLecarModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    was_below.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, (&v_i, &w_i)) in v.iter().zip(&w).enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
                assert!(!w_i.is_nan(), "w[{i}] NaN step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite ste {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "ML should be silent at I=0");
        // Проверка: потенциал релаксирует к resting (-60)
        for (i, &v_i) in v.iter().enumerate() {
            assert!(
                (v_i - (-60.0)).abs() < 1.0,
                "v[{i}] should relax to -60, got {}",
                v_i
            );
        }
    }

    #[test]
    fn test_ml_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut w = vec![0.0; n];
        let mut was_below = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        let mut total_spikes = 0i64;

        // Edge-triggered: each AP produces exactly 1 spike on rising edge.
        // With natural dynamics at extreme I, ML may enter depolarization block
        // (v stays above threshold) after initial transient. We count total
        // across ALL steps, not just the last one.
        for step in 0..10000 {
            unsafe {
                run_neuron::<MorrisLecarModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    was_below.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite step {step}");
                assert!(
                    v_i < 200.0,
                    "v[{i}] exceeded 200 mV at step {step}: {}",
                    v_i
                );
                assert!(v_i > -200.0, "v[{i}] below -200 mV at step {step}: {}", v_i);
            }
        }
        // Should have at least one spike per neuron on initial rising edge
        assert!(
            total_spikes >= n as i64,
            "ML should spike at I=1000, got {total_spikes} total (expected ≥{n})"
        );
        // Sanity: voltage should be finite and within biological bounds
        for (i, &v_i) in v.iter().enumerate() {
            assert!(v_i.is_finite(), "v[{i}] non-finite at end");
        }
    }
}
