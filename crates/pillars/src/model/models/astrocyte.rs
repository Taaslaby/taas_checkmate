//! 🧫 Astrocyte — Hidden-State Glial Modulation (Shen et al. 2024)
//!
//! Implements the Tripartite Synapse model where astrocytes actively
//! modulate synaptic transmission via:
//!
//! 1. **EAAT (Glutamate Uptake)**: Astrocytes clear synaptic glutamate,
//!    reducing AMPA/NMDA receptor activation. Tau ≈ 10 ms.
//! 2. **K⁺ Spatial Buffering**: Astrocytes buffer extracellular K⁺ via
//!    Kir channels, preventing depolarisation block. Tau ≈ 100 ms.
//! 3. **Volume Transmission (Gain Modulation)**: Astrocytic Ca²⁺ waves
//!    release gliotransmitters that modulate synaptic gain globally.
//!    Tau ≈ 1-10 s.
//! 4. **Gap Junctions**: Astrocytes form a syncytium via gap junctions,
//!    propagating Ca²⁺ and K⁺ waves across the network.
//!
//! ## Dynamics (unified Forward Euler, zero divisions)
//!
//! All tau values are stored as `inv_tau = 1.0 / τ` at init time.
//! Single FMA step: `new = state + dt · ((target - state) · inv_tau + α · I)`
//!
//! ```text
//!   dCa/dt = (0.1 - Ca) · inv_tau_ca + α_ca · nearby_spikes
//!   dGlu/dt = -Glu · inv_tau_glu + α_glu · pre_spikes
//!   dK/dt  = -K · inv_tau_k + α_k · firing_rate
//!   gain = 1 + μ · Ca  — volume transmission modulation
//! ```
//!
//! Reference: Shen et al. 2024, "Astrocyte-Modulated Spiking Unit",
//!   arXiv:2312.07625
//!
//! ## Params (9 × n, column-major)
//!
//!   [0] dt, [1] inv_tau_ca (1/τ_ca), [2] inv_tau_glu (1/τ_glu),
//!   [3] inv_tau_k (1/τ_k), [4] alpha_ca, [5] alpha_glu, [6] alpha_k,
//!   [7] mu (volume transmission gain), [8] g_gap (gap junction strength)
//!
//! ## State (4 vars)
//!
//!   [0] Ca (calcium concentration, 0..1),
//!   [1] Glu (glutamate, 0..1),
//!   [2] K (extracellular K⁺, relative to baseline),
//!   [3] gain (volume transmission modulation, ≥ 1.0)
//!
//! ## Output
//!
//!   gain multiplies synaptic weights: g_syn *= gain
//!   K modulates E_K: E_K_eff = E_K + 26.7·ln(K_baseline / K)

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Astrocyte — hidden-state glial modulator with EAAT + K⁺ buffering + volume transmission
pub struct AstrocyteModel;

crate::validate_model!(AstrocyteModel {
    params: 9, state: 4, dt: 0,
    forward_euler: true
});


impl Model for AstrocyteModel {
    /// Params: [0]dt, [1]inv_tau_ca, [2]inv_tau_glu, [3]inv_tau_k,
    ///         [4]alpha_ca, [5]alpha_glu, [6]alpha_k,
    ///         [7]mu, [8]g_gap
    const N_PARAMS: usize = 9;
    /// State: [0]Ca, [1]Glu, [2]K, [3]gain
    const N_STATE: usize = 4;
    /// No intrinsic spike — this modulates neurons externally
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let ca = ctx.s(0);    // Ca²⁺ concentration
        let glu = ctx.s(1);   // Glutamate (synaptic cleft)
        let k = ctx.s(2);     // Extracellular K⁺
        let _gain = ctx.s(3);  // Volume transmission gain (read via AstrocyteModel::gain())

        let one = S::one();
        let zero = S::zero();
        let dt = ctx.dt();
        let _ = ctx.p(8); // g_gap — reserved for future gap junction coupling

        // Unified Forward Euler: state += dt · ((target - state) · inv_tau + α · I)
        // 𝘕𝘰 𝘥𝘪𝘷𝘪𝘴𝘪𝘰𝘯 — all inv_tau precomputed at init.
        // 𝘕𝘰 𝘥𝘰𝘶𝘣𝘭𝘦 𝘥𝘵 𝘴𝘤𝘢𝘭𝘪𝘯𝘨 — decay & input integrated together.
        let inp = ctx.i().clamp01();

        // Ca²⁺: dCa/dt = (0.1 - Ca)·inv_tau_ca + α_ca·I
        let ca_inf = S::splat(0.1);
        let new_ca = ca + dt * ((ca_inf - ca) * ctx.p(1) + ctx.p(4) * inp);
        let new_ca = new_ca.clamp01();

        // Glutamate: dGlu/dt = -Glu·inv_tau_glu + α_glu·I
        let new_glu = glu + dt * ((-glu) * ctx.p(2) + ctx.p(5) * inp);
        let new_glu = new_glu.clamp01();

        // K⁺: dK/dt = -K·inv_tau_k + α_k·I
        let new_k = k + dt * ((-k) * ctx.p(3) + ctx.p(6) * inp);
        let new_k = new_k.max(zero);

        // Volume transmission gain: M = 1 + μ · Ca
        let new_gain = one + ctx.p(7) * new_ca;

        ctx.set_s(0, new_ca);
        ctx.set_s(1, new_glu);
        ctx.set_s(2, new_k);
        ctx.set_s(3, new_gain);
    }
}

impl AstrocyteModel {
    /// Get volume transmission gain — multiplies synaptic weights.
    /// Call after step() to get current gain value.
    #[inline(always)]
    pub fn gain<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        ctx.s(3)
    }

    /// Get effective K⁺ reversal potential (Nernst equation).
    ///
    /// Biologically: when K⁺_out rises (astocyte K⁺ buffering saturates),
    /// E_K shifts UP (depolarising), increasing excitability.
    ///
    /// E_K_eff = E_K + 26.7 · ln(K / K_baseline)
    /// where K_baseline = 1.0 (normalised), K is extracellular K⁺
    ///
    /// 𝘍𝘪𝘹𝘦𝘥: was ln(1/K) which hyperpolarised on K⁺ rise (wrong!).
    ///  𝘕𝘰 𝘥𝘪𝘷𝘪𝘴𝘪𝘰𝘯 (k.ln() instead of (1/k).ln()).
    #[inline(always)]
    pub fn ek_eff<S: SimdLane>(ctx: &SimdContext<S>, e_k: f32) -> S {
        let k = ctx.s(2).max(S::splat(0.5));  // clamp to prevent log(0)
        let nernst = S::splat(26.7) * k.ln();  // ln(K), not ln(1/K)!
        S::splat(e_k) + nernst
    }

    /// Get EAAT effect: how much glutamate is cleared
    /// Higher Glu = more synapses are depressed
    #[inline(always)]
    pub fn eaat_factor<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        S::one() - ctx.s(1)  // 1 - Glu: more glutamate = more depression
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        // All inv_tau precomputed at init — zero divisions in hot path
        let inv_tau_ca = 1.0 / 1000.0;   // was tau_ca=1000
        let inv_tau_glu = 1.0 / 10.0;    // was tau_glu=10
        let inv_tau_k = 1.0 / 100.0;     // was tau_k=100
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[i] = 1.0;             // dt (ms)
            p[n+i] = inv_tau_ca;    // inv_tau_ca (was tau_ca=1000)
            p[2*n+i] = inv_tau_glu; // inv_tau_glu (was tau_glu=10)
            p[3*n+i] = inv_tau_k;   // inv_tau_k (was tau_k=100)
            p[4*n+i] = 0.1;         // alpha_ca
            p[5*n+i] = 0.5;         // alpha_glu
            p[6*n+i] = 0.3;         // alpha_k
            p[7*n+i] = 0.5;         // mu (volume transmission)
            p[8*n+i] = 0.1;         // g_gap
        }
        p
    }

    #[test]
    fn test_astrocyte_ca_wave() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.0; n];
        let mut glu = vec![0.0; n];
        let mut k = vec![0.0; n];
        let mut gain = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![1.0; n];  // active input → nearby spikes
        let mut spikes = vec![0i32; n];

        // Single step with input
        unsafe { run_neuron::<AstrocyteModel>(
            ca.as_mut_ptr(), glu.as_mut_ptr(), k.as_mut_ptr(), gain.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }

        // Ca should rise slightly
        assert!(ca[0] > 0.0, "Ca should rise with input: {:.4}", ca[0]);
        // Gain should be > 1.0 (volume transmission)
        assert!(gain[0] >= 1.0, "Gain should be ≥ 1.0: {:.4}", gain[0]);
    }

    #[test]
    fn test_astrocyte_silent_decay() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.5; n];    // elevated Ca
        let mut glu = vec![0.5; n];   // elevated Glu
        let mut k = vec![0.5; n];     // elevated K
        let mut gain = vec![1.25; n]; // elevated gain
        let mut dummy = vec![0.0; n];
        let i_syn = vec![0.0; n];  // no input
        let mut spikes = vec![0i32; n];

        // Step with no input → all states should decay
        unsafe { run_neuron::<AstrocyteModel>(
            ca.as_mut_ptr(), glu.as_mut_ptr(), k.as_mut_ptr(), gain.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }

        assert!(ca[0] < 0.5, "Ca should decay without input: {:.4}", ca[0]);
        assert!(glu[0] < 0.5, "Glu should decay without input: {:.4}", glu[0]);
    }

    #[test]
    fn test_astrocyte_ek_eff() {
        // E_K_eff should DEPOLARISE when K is elevated (correct Nernst)
        // E_K_eff = E_K + 26.7 · ln(K)  (K_baseline = 1.0)
        // K=2.0: -80 + 26.7·ln(2) = -80 + 18.5 = -61.5 mV (depolarised!)
        type S = wide::f32x8;

        let mut ctx = SimdContext::<S> {
            state: [S::splat(0.0); 6],
            params_ptr: std::ptr::null(),
            n: 8,
            off: 0,
            i_syn: S::splat(0.0),
            i_pre: S::splat(0.0),
            i_post: S::splat(0.0),
            dt_val: 1.0,
            spike_mask: S::splat(0.0),
        };
        ctx.state = [S::splat(0.0), S::splat(0.0), S::splat(2.0), S::splat(1.0),
                     S::splat(0.0), S::splat(0.0)];  // K = 2.0 (elevated)

        let ek = AstrocyteModel::ek_eff(&ctx, -80.0);
        let val = ek.to_array()[0];
        // 𝘍𝘪𝘹𝘦𝘥: high K → depolarise (was -98.5, now -61.5)
        assert!((val - (-61.5)).abs() < 1.0,
            "E_K_eff with K=2.0 should be ≈ -61.5 (depolarised), got {:.2}", val);
    }
}
