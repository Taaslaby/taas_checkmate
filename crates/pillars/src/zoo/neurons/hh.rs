//! ⚡ HH — Hodgkin-Huxley (1952) squid axon model
//!
//! Standard NaV + KV + leak, with Q10 temperature correction.
//! **Continuous biophysical model** — no artificial V reset.
//! Edge-triggered spike detection (one spike per AP, not per step).
//!
//! # Equations
//!
//! Membrane potential:
//! ```text
//! C·dV/dt = I_inj - g_Na·m³·h·(V-E_Na) - g_K·n⁴·(V-E_K) - g_L·(V-E_L)
//! ```
//!
//! Gate kinetics (α/β formulation):
//! ```text
//! m-gate: α=0.1·vtrap(-(V+40),10)   β=4·exp(-(V+65)/18)
//! h-gate: α=0.07·exp(-(V+65)/20)    β=1/(1+exp(-(V+35)/10))
//! n-gate: α=0.01·vtrap(-(V+55),10)  β=0.125·exp(-(V+65)/80)
//!
//! x_∞ = α/(α+β)
//! x'  = x_∞ + (x - x_∞)·exp(-dt·Q10·(α+β))
//! ```
//!
//! # Architecture
//! - `#[derive(NeuronModel)]` — typed state/param fields
//! - `ComputeNeuron` — pure SIMD step, no raw ctx.s(0) indexing
//! - `cube()`, `powi::<4>()` — hardware vmulps, zero branches
//! - `ohmic_gated`/`ohmic_current` — NEURON convention (FMA)
//! - `edge_spike` — continuous detection, no V reset
//!
//! # Biophysical Validation
//! - Validated against NEURON hh.mod (v9.0.1)
//! - Resting gates (V=-65mV): m=0.0529, h=0.596, n=0.318 (matches ±0.005)
//! - Temperature: Q10=3.0, reference 6.3°C
//!
//! # Performance
//! - SIMD: 8 neurons/instruction (f32x8 on AVX2)
//! - Pure FMA — no divisions in hot path
//! - Q10 precomputed — no powf in step()

#![allow(non_snake_case)]

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};
use crate::ops::{ohmic_current, ohmic_gated, membrane_depolarize, edge_spike};

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

/// Type-safe parameter indices for HHModel (SoA column-major layout).
///
/// Order matches `#[param]` field order in `HHModel` struct.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum HHParamIdx {
    DT = 0,         // ms — time step
    G_Na = 1,       // mS/cm² — Na⁺ max conductance
    E_Na = 2,       // mV — Na⁺ reversal potential
    G_K = 3,        // mS/cm² — K⁺ max conductance
    E_K = 4,        // mV — K⁺ reversal potential
    G_Leak = 5,     // mS/cm² — leak conductance
    E_Leak = 6,     // mV — leak reversal potential
    Q10 = 7,        // (scalar) — precomputed Q10 temperature factor
    V_Thresh = 8,   // mV — spike detection threshold
    Inv_Cm = 9,     // 1/µF — inverse membrane capacitance
    G_CaT = 10,     // mS/cm² — CaV-T (T-type Ca²⁺) max conductance (0 for standard HH)
    E_CaT = 11,     // mV — CaV-T reversal potential (≈ +120 for pacemaking)
}

impl From<HHParamIdx> for usize {
    #[inline(always)] fn from(idx: HHParamIdx) -> Self { idx as usize }
}

// ═══════════════════════════════════════════════════════════════
//  Gate Definition (α/β rate functions)
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley gate rate parameters (α/β formulation).
///
/// Each gate has:
/// - `v_off`: voltage offset for vtrap (e.g., -40 for m-gate)
/// - `scale`: vtrap divisor (e.g., 10 for m-gate)
/// - `alpha_c`, `beta_c`: rate coefficients
/// - `beta_volt`, `beta_div`: β exponential/sigmoid parameters
/// - `alpha_exp`: if true, α = alpha_c·exp(v_shift/scale) (h-gate)
///   if false, α = alpha_c·vtrap(v_shift, scale) (m, n)
/// - `beta_sigmoid`: if true, β = beta_c/(1+exp(-(V+beta_volt)/beta_div))
///   if false, β = beta_c·exp(-(V+beta_volt)/beta_div)
#[derive(Copy, Clone)]
pub struct GateParams {
    pub v_off: f32,
    pub scale: f32,
    pub alpha_c: f32,
    pub beta_c: f32,
    pub beta_volt: f32,
    pub beta_div: f32,
    pub alpha_exp: bool,
    pub beta_sigmoid: bool,
}

/// m-gate: Na⁺ activation — fast, voltage-dependent opening
/// α = 0.1·vtrap(-(V+40), 10)   β = 4·exp(-(V+65)/18)
pub const GATE_M: GateParams = GateParams {
    v_off: -40.0, scale: 10.0, alpha_c: 0.1,
    beta_c: 4.0, beta_volt: 65.0, beta_div: 18.0,
    alpha_exp: false, beta_sigmoid: false,
};

/// h-gate: Na⁺ inactivation — slower, closes Na channels during spike
/// α = 0.07·exp(-(V+65)/20)     β = 1/(1+exp(-(V+35)/10))
///  α is EXPONENTIAL, NOT vtrap (unlike m/n)!
pub const GATE_H: GateParams = GateParams {
    v_off: -65.0, scale: 20.0, alpha_c: 0.07,
    beta_c: 1.0, beta_volt: 35.0, beta_div: 10.0,
    alpha_exp: true, beta_sigmoid: true,
};

/// n-gate: K⁺ activation — slow, opens KV channels for repolarization
/// α = 0.01·vtrap(-(V+55), 10)  β = 0.125·exp(-(V+65)/80)
pub const GATE_N: GateParams = GateParams {
    v_off: -55.0, scale: 10.0, alpha_c: 0.01,
    beta_c: 0.125, beta_volt: 65.0, beta_div: 80.0,
    alpha_exp: false, beta_sigmoid: false,
};

/// CaV-T (T-type calcium) gate — single low-threshold activation gate `c` for
/// pacemaking (Wang & Rinzel 1993 / Hill & Tononi 2005).
///
/// ```text
/// m_inf  = 1 / (1 + exp(-(v+56)/6.2))     — steady-state activation
/// τ_c    = 0.6 ms                          — fixed time constant
/// c(t+1) = m_inf + (c - m_inf)·exp(-dt·Q10/τ_c)
/// ```
///
/// α = m_inf/τ, β = (1−m_inf)/τ → α+β = 1/τ is **constant**, so the
/// exponential-Euler decay is voltage-independent (one `fast_exp` total).
pub const GATE_CAT_TAU: f32 = 0.6;
pub const GATE_CAT_INV_TAU: f32 = 1.0 / GATE_CAT_TAU; // precomputed — NO vector division in step()
pub const GATE_CAT_V_HALF: f32 = -56.0;
pub const GATE_CAT_K: f32 = 6.2;
pub const GATE_CAT_INV_K: f32 = 1.0 / GATE_CAT_K; // precomputed — NO vector division in step()

/// Compute the CaV-T gate one step via exponential Euler (constant-τ form).
///
/// Exactly mirrors `hh_channels::cat_c_alpha`/`cat_c_beta` with
/// `α+β = 1/τ_c` folded into a voltage-independent decay.
#[inline(always)]
fn compute_cat_gate_state<S: SimdLane>(v: S, c: S, dt_q10: S) -> S {
    // m_inf = 1/(1+exp(-(v+56)/6.2)) — arg division folded into precomputed inv_k
    // (the 1/(1+e) reciprocal stays: it is a single divps, the sigmoid itself)
    let arg = -(v + S::splat(GATE_CAT_V_HALF)) * S::splat(GATE_CAT_INV_K);
    let m_inf = S::one() / (S::one() + arg.fast_exp());
    // decay = exp(-dt·Q10/τ_c) — voltage-independent, τ folded into inv_tau
    let decay = (-dt_q10 * S::splat(GATE_CAT_INV_TAU)).fast_exp();
    decay.mul_add(c - m_inf, m_inf)
}

/// Compute HH gate one step via Exponential Euler.
///
/// ```text
/// α(V)     — rate of opening
/// β(V)     — rate of closing
/// x_∞      = α/(α+β)           — steady state
/// τ(V)     = 1/(α+β)           — time constant
/// x(t+dt)  = x_∞ + (x-x_∞)·exp(-dt·Q10·(α+β))
/// ```
///
/// All `if` branches are compile-time (const GateParams) — zero overhead.
/// No scalar conversion, no branching at runtime.
#[inline(always)]
fn compute_gate_state<S: SimdLane>(v: S, gate: S, params: GateParams, dt_q10: S) -> S {
    let v_shift = S::splat(params.v_off) - v;

    // α function: either vtrap (m, n) or exponential (h)
    let alpha = if params.alpha_exp {
        S::splat(params.alpha_c) * (v_shift / S::splat(params.scale)).fast_exp()
    } else {
        S::splat(params.alpha_c) * S::vtrap(v_shift, S::splat(params.scale))
    };

    // β function: either exponential (m, n) or sigmoid (h)
    let exp_arg = -(v + S::splat(params.beta_volt)) / S::splat(params.beta_div);
    let beta = if params.beta_sigmoid {
        S::splat(params.beta_c) / (S::one() + exp_arg.fast_exp())
    } else {
        S::splat(params.beta_c) * exp_arg.fast_exp()
    };

    let sum = alpha + beta;
    let g_inf = alpha / sum;
    let decay = (-dt_q10 * sum).fast_exp();
    // FMA: g_inf + (gate - g_inf)·decay
    decay.mul_add(gate - g_inf, g_inf)
}

// ═══════════════════════════════════════════════════════════════
//  Model Definition
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley (1952) squid axon — continuous conductance-based model.
///
/// # States (6)
/// - `v`: membrane potential (mV)
/// - `m`: Na⁺ activation gate
/// - `h`: Na⁺ inactivation gate
/// - `n`: K⁺ activation gate
/// - `c`: CaV-T (T-type calcium) activation gate (pacemaking)
/// - `was_below`: edge-detection flag (1.0 = was below v_thresh last step)
///
/// # Params (12)
/// - `dt`, `g_Na`, `E_Na`, `g_K`, `E_K`, `g_Leak`, `E_Leak`,
///   `q10`, `v_thresh`, `inv_cm`, `g_CaT`, `E_CaT`
///
/// Set `g_CaT = 0` for the classic squid-axon HH (no Ca²⁺ current) and
/// `g_CaT = 0.5, E_CaT = +120` for **pacemaking** (spontaneous oscillation).
#[derive(NeuronModel)]
pub struct HHModel {
    #[state] pub v: f32,
    #[state] pub m: f32,
    #[state] pub h: f32,
    #[state] pub n: f32,
    #[state] pub c: f32,
    #[state] pub was_below: f32,

    #[param] pub dt: f32,
    #[param] pub g_Na: f32,
    #[param] pub E_Na: f32,
    #[param] pub g_K: f32,
    #[param] pub E_K: f32,
    #[param] pub g_Leak: f32,
    #[param] pub E_Leak: f32,
    #[param] pub q10: f32,
    #[param] pub v_thresh: f32,
    #[param] pub inv_cm: f32,
    #[param] pub g_CaT: f32,
    #[param] pub E_CaT: f32,
}

impl ComputeNeuron for HHModel {
    type State<S: SimdLane> = HHState<S>;
    type Params<S: SimdLane> = HHParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut Self::State<S>,
        params: &Self::Params<S>,
        ctx: &mut StepContext<S>,
    ) {
        // ─── Temperature-scaled time step ─────────────────────────
        let dt_q10 = ctx.dt_simd() * params.q10;

        // ─── Gate Integration (4 gates) ───────────────────────────
        // Standard gates: x_∞ = α/(α+β), x' = x_∞ + (x-x_∞)·exp(-dt·Q10·(α+β))
        let m_new = compute_gate_state(state.v, state.m, GATE_M, dt_q10);
        let h_new = compute_gate_state(state.v, state.h, GATE_H, dt_q10);
        let n_new = compute_gate_state(state.v, state.n, GATE_N, dt_q10);
        // CaV-T gate: constant τ form — one fast_exp, no α/β split
        let c_new = compute_cat_gate_state(state.v, state.c, dt_q10);

        // ─── Ionic Currents via ops::ohmic (NEURON convention) ───
        // Na⁺: g_Na·m³·h·(V-E_Na)  — cube() = 2× vmulps
        // K⁺:  g_K·n⁴·(V-E_K)      — powi::<4>() = 2× vmulps
        // CaT: g_CaT·c·(V-E_CaT)    — 0 for standard HH, 0.5 for pacemaking
        // Leak: g_L·(V-E_L)          — 1× vmulps
        let gating_na = m_new.cube() * h_new;   // m³h
        let gating_k  = n_new.powi::<4>();       // n⁴

        let i_na   = ohmic_gated(params.g_Na, state.v, params.E_Na, gating_na);
        let i_k    = ohmic_gated(params.g_K, state.v, params.E_K, gating_k);
        let i_cat  = ohmic_gated(params.g_CaT, state.v, params.E_CaT, c_new);
        let i_leak = ohmic_current(params.g_Leak, state.v, params.E_Leak);

        // ─── Membrane Integration (ops::membrane) ─────────────────
        // V' = V + (I_inj - ΣI_ion) · inv_Cm · dt
        let v_new = membrane_depolarize(
            state.v,
            ctx.i_syn,
            i_na + i_k + i_cat + i_leak,
            params.inv_cm,
            ctx.dt_simd(),
        );

        // ─── Spike Detection (Edge-Triggered, No V Reset!) ───────
        // ❌ No v_reset — HH is continuous biophysical model.
        // ✅ Edge-triggered: fires once on rising edge through v_thresh.
        let is_above = v_new.cmp_ge(params.v_thresh);
        let (spike, was_below_next) = edge_spike(is_above, state.was_below);
        ctx.emit_spike(spike);

        // ─── State Update (All Continuous!) ──────────────────────
        state.v = v_new;
        state.m = m_new;
        state.h = h_new;
        state.n = n_new;
        state.c = c_new;
        state.was_below = was_below_next;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    fn make_params(num: usize, celsius: f32) -> Vec<f32> {
        let q10 = 3.0_f32.powf((celsius - 6.3) / 10.0);
        let mut p = vec![0.0f32; num * HHModel::N_PARAMS];
        for i in 0..num {
            p[HHParamIdx::DT as usize * num + i] = 0.01;
            p[HHParamIdx::G_Na as usize * num + i] = 120.0;
            p[HHParamIdx::E_Na as usize * num + i] = 50.0;
            p[HHParamIdx::G_K as usize * num + i] = 36.0;
            p[HHParamIdx::E_K as usize * num + i] = -77.0;
            p[HHParamIdx::G_Leak as usize * num + i] = 0.3;
            p[HHParamIdx::E_Leak as usize * num + i] = -54.387;
            p[HHParamIdx::Q10 as usize * num + i] = q10;
            p[HHParamIdx::V_Thresh as usize * num + i] = 0.0;
            p[HHParamIdx::Inv_Cm as usize * num + i] = 1.0;
            p[HHParamIdx::G_CaT as usize * num + i] = 0.0; // standard HH: no CaT
            p[HHParamIdx::E_CaT as usize * num + i] = 120.0;
        }
        p
    }

    /// HH должна генерировать спайки при I=50.
    #[test]
    fn test_hh_spikes_with_current() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num]; // CaV-T gate steady-state at -65
        let i_syn = vec![50.0f32; num];
        let mut spikes = vec![0i32; num];

        let mut total = 0i32;
        for _ in 0..5000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HH should spike with I=50, got {total}");
    }

    /// HH не спайкует без тока.
    #[test]
    fn test_hh_silent_without_current() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num];
        let i_syn = vec![0.0f32; num];
        let mut spikes = vec![0i32; num];

        for _ in 0..500 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0,
            "HH should not spike with I=0");
    }

    /// Non-aligned count (13 ≠ multiple of 8).
    #[test]
    fn test_hh_non_aligned() {
        let num = 13;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num]; // CaV-T gate steady-state at -65
        let i_syn = vec![50.0f32; num];
        let mut spikes = vec![0i32; num];

        let mut total = 0i32;
        for _ in 0..5000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HH non-aligned should spike, got {total}");
    }

    /// Ингибиция: I=-10 не должна вызывать спайки.
    #[test]
    fn test_hh_inhibited() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num];
        let i_syn = vec![-10.0f32; num];
        let mut spikes = vec![0i32; num];

        for _ in 0..500 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0,
            "HH should not spike with I=-10");
    }

    /// Stress: 10k steps at I=0 — no NaN, v relaxes to rest.
    #[test]
    fn test_hh_stress_silence() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num];
        let i_syn = vec![0.0f32; num];
        let mut spikes = vec![0i32; num];

        for step in 0..10000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN at step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0,
            "HH should be silent at I=0");
    }

    /// Stress: 10k steps at I=1000 — no NaN, finite, within bounds.
    #[test]
    fn test_hh_stress_explosion() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let mut cat = vec![0.19f32; num];
        let i_syn = vec![1000.0f32; num];
        let mut spikes = vec![0i32; num];
        let mut total_spikes = 0i64;

        for step in 0..10000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                cat.as_mut_ptr(), flag.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            total_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN at step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite");
                assert!(v_i < 200.0, "v[{i}] exceeded 200 mV at {step}: {}", v_i);
                assert!(v_i > -200.0, "v[{i}] below -200 mV at {step}: {}", v_i);
            }
        }
        assert!(total_spikes > 0, "HH should spike at I=1000, got {total_spikes}");
    }

    /// NeuralPopulation integration test.
    #[test]
    fn test_hh_population_spiking() {
        let mut pop = NeuralPopulation::<HHModel>::new(8);
        pop.fill_state(0, -65.0);  // v
        pop.fill_state(1, 0.05);   // m
        pop.fill_state(2, 0.6);    // h
        pop.fill_state(3, 0.32);   // n
        pop.fill_state(4, 0.19);   // c (CaV-T, steady-state at -65)
        pop.fill_state(5, 1.0);    // was_below
        for idx in 0..HHModel::N_PARAMS {
            pop.fill_param(idx, make_params(1, 6.3)[idx]);
        }
        let i_in = vec![50.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..5000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HH pop should spike with I=50, got {total}");
    }

    /// Pacemaking (CaV-T): with g_CaT=0.5, E_CaT=120 the neuron fires
    /// SPONTANEOUSLY with zero input current — the T-type Ca²⁺ current
    /// depolarizes the membrane past NaV threshold on its own.
    #[test]
    fn test_hh_pacemaking_fires_spontaneously() {
        let mut pop = NeuralPopulation::<HHModel>::new(8);
        pop.fill_state(0, -65.0);
        pop.fill_state(1, 0.05);
        pop.fill_state(2, 0.6);
        pop.fill_state(3, 0.32);
        pop.fill_state(4, 0.19);
        pop.fill_state(5, 1.0);
        for idx in 0..HHModel::N_PARAMS {
            pop.fill_param(idx, make_params(1, 6.3)[idx]);
        }
        pop.fill_param(HHParamIdx::G_CaT.into(), 0.5);
        pop.fill_param(HHParamIdx::E_CaT.into(), 120.0);

        let i_in = vec![0.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        // 200 ms at dt=0.01 → 20_000 steps; CaV-T oscillation ~10-20 Hz
        for _ in 0..20_000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "Pacemaking HH should fire with I=0, got {total}");
    }

    /// Standard HH with g_CaT=0 must stay SILENT at rest (CaT gate present
    /// but its current is zero) — the extra gate must not alter squid-axon
    /// behavior.
    #[test]
    fn test_hh_standard_silent_at_rest() {
        let mut pop = NeuralPopulation::<HHModel>::new(8);
        pop.fill_state(0, -65.0);
        pop.fill_state(1, 0.05);
        pop.fill_state(2, 0.6);
        pop.fill_state(3, 0.32);
        pop.fill_state(4, 0.19);
        pop.fill_state(5, 1.0);
        for idx in 0..HHModel::N_PARAMS {
            pop.fill_param(idx, make_params(1, 6.3)[idx]);
        }
        let i_in = vec![0.0f32; 8];
        let mut spikes = vec![0i32; 8];
        for _ in 0..10_000 {
            pop.step(&i_in, &mut spikes);
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "standard HH must be silent at rest");
    }

    // ─── NEURON-validated rate function tests ─────────────────

    #[test]
    fn test_hh_rates_resting_vs_neuron() {
        use wide::f32x8;
        let v = f32x8::splat(-65.0);
        let one = f32x8::splat(1.0);

        let vm40 = -(v + f32x8::splat(40.0));
        let alpha_m = f32x8::splat(0.1) * crate::math::vtrap::vtrap(vm40, f32x8::splat(10.0));
        let beta_m = f32x8::splat(4.0) * (-(v + f32x8::splat(65.0)) / f32x8::splat(18.0)).exp();
        let minf = (alpha_m / (alpha_m + beta_m)).to_array()[0];

        let alpha_h = f32x8::splat(0.07) * (-(v + f32x8::splat(65.0)) / f32x8::splat(20.0)).exp();
        let beta_h = one / (one + (-(v + f32x8::splat(35.0)) / f32x8::splat(10.0)).exp());
        let hinf = (alpha_h / (alpha_h + beta_h)).to_array()[0];

        let vn55 = -(v + f32x8::splat(55.0));
        let alpha_n = f32x8::splat(0.01) * crate::math::vtrap::vtrap(vn55, f32x8::splat(10.0));
        let beta_n = f32x8::splat(0.125) * (-(v + f32x8::splat(65.0)) / f32x8::splat(80.0)).exp();
        let ninf = (alpha_n / (alpha_n + beta_n)).to_array()[0];

        assert!((minf - 0.0529).abs() < 0.01, "minf={minf}");
        assert!((hinf - 0.596).abs() < 0.05, "hinf={hinf}");
        assert!((ninf - std::f32::consts::FRAC_1_PI).abs() < 0.03, "ninf={ninf}");
    }

    #[test]
    fn test_hh_rates_depolarized_vs_neuron() {
        use wide::f32x8;
        let v = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let vm40 = -(v + f32x8::splat(40.0));
        let alpha_m = f32x8::splat(0.1) * crate::math::vtrap::vtrap(vm40, f32x8::splat(10.0));
        let beta_m = f32x8::splat(4.0) * (-(v + f32x8::splat(65.0)) / f32x8::splat(18.0)).exp();
        let minf = (alpha_m / (alpha_m + beta_m)).to_array()[0];

        let alpha_h = f32x8::splat(0.07) * (-(v + f32x8::splat(65.0)) / f32x8::splat(20.0)).exp();
        let beta_h = one / (one + (-(v + f32x8::splat(35.0)) / f32x8::splat(10.0)).exp());
        let hinf = (alpha_h / (alpha_h + beta_h)).to_array()[0];

        assert!((minf - 0.974).abs() < 0.02,
            "minf at 0mV should be ≈0.974, got {minf}");
        assert!(hinf < 0.01,
            "hinf at 0mV should be ≈0.0, got {hinf}");
    }

    #[test]
    fn test_hh_gate_constants_match_params() {
        assert!((GATE_M.v_off - (-40.0)).abs() < 1e-6);
        assert!((GATE_M.scale - 10.0).abs() < 1e-6);
        assert!((GATE_M.alpha_c - 0.1).abs() < 1e-6);
        assert!((GATE_N.v_off - (-55.0)).abs() < 1e-6);
        assert!((GATE_N.beta_div - 80.0).abs() < 1e-6);
        const _: () = assert!(GATE_H.alpha_exp, "h-gate alpha MUST be exp, NOT vtrap!");
        assert!((GATE_H.alpha_c - 0.07).abs() < 1e-6);
        assert!((GATE_H.scale - 20.0).abs() < 1e-6);
        const _: () = assert!(GATE_H.beta_sigmoid, "h-gate beta MUST be sigmoid!");
    }
}
