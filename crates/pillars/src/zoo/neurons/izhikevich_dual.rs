//! Izhikevich Dual Memory — Fast + Slow Pathways (Sun et al. 2026, Nature MI).
//!
//! Extends standard Izhikevich with a **slow memory state** `m` per neuron:
//!
//! ```text
//! Slow pathway (memory):
//!   m[t] = τ_m · m[t-1] + (1 − τ_m) · spike_val[t-1]     (spike_val: 1.0 on spike, 0.5 otherwise)
//!   m clamped to [0, 1]
//!
//! Fast pathway (Izhikevich, modulated):
//!   dv/dt = 0.04·v² + 5·v + 140 − u + (I_clamped + w_m · m[t-1])
//!   du/dt = a·(b·v − u)
//! ```
//!
//! The slow state accumulates recent spiking activity with time constant
//! `τ_m` (e.g. τ_m = 0.999 → ~1000 ms memory at 0.1 ms dt). The memory
//! readout `w_m · m` modulates the input current — fast-slow dynamics where
//! past activity shapes present excitability.
//!
//! # Legacy parity (canonical port of `kernels::izhikevich_dual_step`)
//!
//! - **Single-step Forward Euler** (НЕ substepping, в отличие от pillars
//!   `IzhikevichModel`) — точная легаси-семантика dual-кернела.
//! - `I` клампится до ±100 в раннере (legacy inline clamp), модуляция
//!   `w_m·m` применяется ПОСЛЕ клампа.
//! - `m` обновляется **всегда** (включая рефрактерность): 1.0 на спайке,
//!   0.5 иначе; clamp [0, 1].
//! - `trace` (STDP) заморожен в рефрактерности; на спайке `tr·td + (1−tr)`,
//!   иначе `tr·(1−td)`.
//! - Порог спайка хардкод 30.0 (передаётся как параметр `v_thresh`).
//! - Бит-идентичность: guarded by `crates/kernels/tests/izhikevich_dual_parity.rs`
//!   (обычная не-fused арифметика — та же группировка, что у легаси).
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.

#![allow(clippy::module_inception)] // deliberate: `pub mod izhikevich_dual` mirrors the file module

use pillars_macros::dual_neuron;

/// Izhikevich Dual Memory (Sun et al. 2026) — fast Izh + slow memory state `m`.
#[dual_neuron]
pub mod izhikevich_dual {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct IzhikevichDualModel {
        #[state]
        pub v: f32,
        #[state]
        pub u: f32,
        #[state]
        pub refr: f32,
        #[state]
        pub trace: f32,
        #[state]
        pub m: f32, // slow memory state

        #[param]
        pub dt: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub c: f32,
        #[param]
        pub d: f32,
        #[param]
        pub tau_ref: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub trace_decay: f32,
        #[param]
        pub tau_m: f32, // slow state time constant (0..1, higher = slower decay)
        #[param]
        pub w_m: f32, // memory-to-current weight
    }

    impl ComputeNeuron for IzhikevichDualModel {
        type State<S: SimdLane> = IzhikevichDualState<S>;
        type Params<S: SimdLane> = IzhikevichDualParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let one = S::one();
            let half = S::splat(0.5);
            let c04 = S::splat(0.04);
            let c5 = S::splat(5.0);
            let c140 = S::splat(140.0);

            // Slow pathway: modulate input current. `ctx.i_syn` уже клампнут
            // раннером до ±100 (legacy inline clamp), модуляция — после.
            let i_mod = ctx.i_syn + params.w_m * state.m;

            let is_refr = state.refr.cmp_gt(zero);

            // Single-step Forward Euler (legacy dual semantics, NOT substepping).
            // Не-fused группировка: та же последовательность mul/add, что у
            // легаси `v5.mul_add(c04, c5)` (wide::mul_add = mul+add, 2 округления).
            let v5 = state.v * (c04 * state.v + c5);
            let dv = v5 + c140 - state.u + i_mod;
            let du = params.a * (params.b * state.v - state.u);

            let v_new = is_refr.cond().then(state.v).else_(state.v + params.dt * dv);
            let u_new = is_refr.cond().then(state.u).else_(state.u + params.dt * du);

            // Spike detection: only non-refractory, v ≥ v_thresh (= 30.0 legacy).
            let spike = is_refr
                .cond()
                .then(zero)
                .else_(v_new.cmp_ge(params.v_thresh));
            ctx.emit_spike(spike);

            state.v = spike.cond().then(params.c).else_(v_new);
            state.u = spike.cond().then(u_new + params.d).else_(u_new);

            // Refractory counter: decrement always, set to τ_ref on spike.
            let refr_dec = state.refr - params.dt;
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);

            // STDP trace: frozen during refractory; on spike tr = tr·td + (1−tr),
            // else tr·(1−td) (legacy `tr.mul_add(td, one − tr)` = non-fused).
            let tr_spike = state.trace * params.trace_decay + (one - state.trace);
            let tr_no = state.trace * (one - params.trace_decay);
            state.trace = is_refr
                .cond()
                .then(state.trace)
                .else_(spike.cond().then(tr_spike).else_(tr_no));

            // Slow state: m обновляется ВСЕГДА (включая рефрактерность).
            // spike_val = 1.0 на спайке, 0.5 иначе; m = τ_m·m + (1−τ_m)·spike_val.
            let spike_val = spike.cond().then(one).else_(half);
            state.m = (params.tau_m * state.m + (one - params.tau_m) * spike_val)
                .max(zero)
                .min(one);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::izhikevich_dual::IzhikevichDualModel` etc.
pub use izhikevich_dual::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    const DT: f32 = 1.0;
    const TRACE_DECAY: f32 = 0.5;
    const TAU_REF: f32 = 3.0;

    fn make_params(n: usize, tau_m: f32, w_m: f32) -> Vec<f32> {
        let mut p = vec![0.0f32; n * IzhikevichDualModel::N_PARAMS];
        for i in 0..n {
            p[i] = DT;
            p[n + i] = 0.02; // a
            p[2 * n + i] = 0.2; // b
            p[3 * n + i] = -65.0; // c
            p[4 * n + i] = 8.0; // d
            p[5 * n + i] = TAU_REF;
            p[6 * n + i] = 30.0; // v_thresh (legacy hardcoded)
            p[7 * n + i] = TRACE_DECAY;
            p[8 * n + i] = tau_m;
            p[9 * n + i] = w_m;
        }
        p
    }

    fn run(
        n: usize,
        I: f32,
        tau_m: f32,
        w_m: f32,
        m_init: f32,
        steps: usize,
    ) -> (Vec<f32>, Vec<f32>, i32) {
        let params = make_params(n, tau_m, w_m);
        let mut v = vec![-65.0; n];
        let mut u = vec![-13.0; n];
        let mut refr = vec![0.0; n];
        let mut trace = vec![0.0; n];
        let mut m = vec![m_init; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![I; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..steps {
            unsafe {
                run_neuron::<IzhikevichDualModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    trace.as_mut_ptr(),
                    m.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        (m, trace, total)
    }

    /// С памятью w_m=5 и I=10 нейрон спайкает; m накапливается (легаси-тест).
    #[test]
    fn test_dual_spikes_and_memory_accumulates() {
        let (m, _, total) = run(8, 10.0, 0.99, 5.0, 0.5, 100);
        assert!(total > 0, "dual Izh should spike with I=10, w_m=5");
        let m_mean: f32 = m.iter().sum::<f32>() / 8.0;
        assert!(
            m_mean > 0.45 && m_mean < 1.0,
            "slow state in (0.45, 1.0), got {m_mean}"
        );
    }

    /// m накапливается от спайков даже стартуя с 0.
    #[test]
    fn test_dual_slow_state_accumulates_from_zero() {
        let (m, _, _) = run(8, 30.0, 0.9, 1.0, 0.0, 20);
        assert!(
            m.iter().sum::<f32>() > 0.0,
            "slow state should accumulate, got {m:?}"
        );
    }

    /// Абсолютная рефрактерность: спайк возможен только вне рефрактерного окна.
    #[test]
    fn test_dual_no_back_to_back_spikes() {
        let n = 8;
        let params = make_params(n, 0.99, 5.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![-13.0; n];
        let mut refr = vec![0.0; n];
        let mut trace = vec![0.0; n];
        let mut m = vec![0.5; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1000.0; n]; // гигантский ток (клампится раннером до ±100)
        let mut spikes = vec![0i32; n];
        let mut fired = false;
        for _ in 0..2000 {
            let refr_before = refr.clone();
            unsafe {
                run_neuron::<IzhikevichDualModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    trace.as_mut_ptr(),
                    m.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                if spikes[i] != 0 {
                    assert!(
                        refr_before[i] <= 0.0,
                        "back-to-back spike: refr_before={}",
                        refr_before[i]
                    );
                    fired = true;
                }
            }
        }
        assert!(fired, "dual Izh should spike at least once at I=1000");
    }

    /// Не-выровненный n (13): tail-путь run_neuron должен спайкать и держать m в [0,1].
    /// NOTE: scalar tail использует SIMD-семантику trace (`tr·td + (1−tr)`), в отличие
    /// от легаси-tail (`tr += td·(1−tr)`) — задокументированное расхождение, см. parity.
    #[test]
    fn test_dual_non_aligned_n() {
        let n = 13;
        let params = make_params(n, 0.99, 5.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![-13.0; n];
        let mut refr = vec![0.0; n];
        let mut trace = vec![0.0; n];
        let mut m = vec![0.5; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![10.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..100 {
            unsafe {
                run_neuron::<IzhikevichDualModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    trace.as_mut_ptr(),
                    m.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "dual Izh n=13 should spike with I=10, w_m=5");
        for i in 0..n {
            assert!(
                (0.0..=1.0).contains(&m[i]),
                "m[{i}] = {} out of [0,1]",
                m[i]
            );
            assert!(v[i].is_finite(), "v[{i}] non-finite");
        }
    }

    /// Стресс: никаких NaN при огромном токе (10 000 шагов).
    #[test]
    fn test_dual_stress_explosion() {
        let n = 8;
        let params = make_params(n, 0.99, 5.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![-13.0; n];
        let mut refr = vec![0.0; n];
        let mut trace = vec![0.0; n];
        let mut m = vec![0.5; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<IzhikevichDualModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    trace.as_mut_ptr(),
                    m.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert!(
                    v[i].is_finite() && u[i].is_finite() && m[i].is_finite(),
                    "non-finite at step {step}: v={} u={} m={}",
                    v[i],
                    u[i],
                    m[i]
                );
            }
        }
    }
}
