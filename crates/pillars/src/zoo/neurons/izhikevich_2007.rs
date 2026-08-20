//! Izhikevich 2007 "Simple Model of Spiking Neurons"
//! (Dynamical Systems in Neuroscience, ch. 5, eq. 5.2).
//!
//! This is the **NeuroML2 `izhikevich2007Cell`** — a *different* parametrization
//! than the 2003 `IzhikevichModel` (which uses `0.04·v² + 5·v + 140`). The 2007
//! form is conductance-based and generalizes it:
//!
//! ```text
//! C·dv/dt = k·(v − vr)·(v − vt) − u + I
//! du/dt   = a·(b·(v − vr) − u)
//! if v ≥ vpeak:  v ← c,  u ← u + d        (instantaneous, no refractory)
//! ```
//!
//! Canonical parameters (tonic spiking, from the paper): `C=100 pF`, `k=0.7`,
//! `vr=−60 mV`, `vt=−40 mV`, `vpeak=35 mV`, `a=0.03`, `b=−2`, `c=−50 mV`,
//! `d=100`. With `C=1`, `k=0.04`, `vr=−82.6556`, `vt=−42.3444` it reduces to
//! the 2003 model — the two families are equivalent, not competing.
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Dual-source (CPU SIMD + CUDA)
//! The `#[dual_neuron]` wrapper emits `IZHIKEVICH_2007_CUDA_SOURCE: &str` — the
//! same step body transpiled to CUDA C++ (NVRTC-ready). `pillars-cuda` imports
//! that string; no CUDA runtime is needed to build `pillars`.

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

#[dual_neuron]
pub mod izhikevich_2007 {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// Izhikevich 2007 "Simple Model of Spiking Neurons".
    ///
    /// # Fields
    /// - `#[state]`: membrane potential `v`, recovery variable `u`
    /// - `#[param]`: `dt`, `cap` (C), `k`, `vr`, `vt`, `vpeak`, `a`, `b`,
    ///   `c` (reset), `d` (spike bump)
    #[derive(NeuronModel)]
    pub struct Izhikevich2007Model {
        #[state]
        pub v: f32,
        #[state]
        pub u: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub cap: f32,
        #[param]
        pub k: f32,
        #[param]
        pub vr: f32,
        #[param]
        pub vt: f32,
        #[param]
        pub vpeak: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub c: f32,
        #[param]
        pub d: f32,
    }

    impl ComputeNeuron for Izhikevich2007Model {
        type State<S: SimdLane> = Izhikevich2007State<S>;
        type Params<S: SimdLane> = Izhikevich2007Params<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Izhikevich2007State<S>,
            params: &Izhikevich2007Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            // C·dv/dt = k·(v − vr)·(v − vt) − u + I
            // Forward Euler: dv = (…)/C · dt
            let dv = (params.k * (state.v - params.vr) * (state.v - params.vt) - state.u
                + ctx.i_syn)
                / params.cap;
            let v_new = state.v + dv * ctx.dt_simd();

            // du/dt = a·(b·(v − vr) − u)
            let du = params.a * (params.b * (state.v - params.vr) - state.u) * ctx.dt_simd();
            let u_new = state.u + du;

            // Spike: v ≥ vpeak → v ← c, u ← u + d (instantaneous reset, no
            // refractory — faithful to the 2007 model, unlike the 2003 model).
            let spike = v_new.cmp_ge(params.vpeak);
            ctx.emit_spike(spike);

            state.v = spike.cond().then(params.c).else_(v_new);
            state.u = spike.cond().then(u_new + params.d).else_(u_new);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::izhikevich_2007::Izhikevich2007Model` etc.
pub use izhikevich_2007::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток, C — ёмкость
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    /// Canonical tonic-spiking parameters (Izhikevich 2007).
    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * Izhikevich2007Model::N_PARAMS];
        for i in 0..n {
            p[Izhikevich2007ModelParamIdx::dt as usize * n + i] = 1.0;
            p[Izhikevich2007ModelParamIdx::cap as usize * n + i] = 100.0;
            p[Izhikevich2007ModelParamIdx::k as usize * n + i] = 0.7;
            p[Izhikevich2007ModelParamIdx::vr as usize * n + i] = -60.0;
            p[Izhikevich2007ModelParamIdx::vt as usize * n + i] = -40.0;
            p[Izhikevich2007ModelParamIdx::vpeak as usize * n + i] = 35.0;
            p[Izhikevich2007ModelParamIdx::a as usize * n + i] = 0.03;
            p[Izhikevich2007ModelParamIdx::b as usize * n + i] = -2.0;
            p[Izhikevich2007ModelParamIdx::c as usize * n + i] = -50.0;
            p[Izhikevich2007ModelParamIdx::d as usize * n + i] = 100.0;
        }
        p
    }

    #[test]
    fn test_izh2007_spikes_with_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n]; // v0 = vr
        let mut u = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![100.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<Izhikevich2007Model>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    dummy.as_mut_ptr(),
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
        assert!(
            total > 0,
            "Izhikevich 2007 should spike with I=100, got {total}"
        );
    }

    #[test]
    fn test_izh2007_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut u = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for _ in 0..1000 {
            unsafe {
                run_neuron::<Izhikevich2007Model>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    dummy.as_mut_ptr(),
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
        assert_eq!(
            spikes.iter().sum::<i32>(),
            0,
            "Izhikevich 2007 should be silent with I=0"
        );
    }

    #[test]
    fn test_izh2007_population() {
        let mut pop = NeuralPopulation::<Izhikevich2007Model>::new(8);
        pop.fill_param(Izhikevich2007ModelParamIdx::dt as usize, 1.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::cap as usize, 100.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::k as usize, 0.7);
        pop.fill_param(Izhikevich2007ModelParamIdx::vr as usize, -60.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::vt as usize, -40.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::vpeak as usize, 35.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::a as usize, 0.03);
        pop.fill_param(Izhikevich2007ModelParamIdx::b as usize, -2.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::c as usize, -50.0);
        pop.fill_param(Izhikevich2007ModelParamIdx::d as usize, 100.0);
        pop.fill_state(0, -60.0);
        pop.fill_state(1, 0.0);

        let i_in = vec![100.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..1000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(
            total > 0,
            "Izh 2007 population should spike with I=100, got {total}"
        );
    }

    #[test]
    fn test_izh2007_stress_no_nan() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-60.0; n];
        let mut u = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<Izhikevich2007Model>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &v_i) in v.iter().enumerate() {
                assert!(v_i.is_finite(), "v[{i}] non-finite at step {step}");
                assert!(u[i].is_finite(), "u[{i}] non-finite at step {step}");
            }
        }
    }
}
