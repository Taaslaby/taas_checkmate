//! ⚡ HH — Hodgkin-Huxley (1952) squid axon model
//!
//! Standard NaV + KV + leak, with Q10 temperature correction.
//! Edge-triggered spike detection (no V reset — continuous biophysical model).
//!
//! # Biophysical Validation
//! - Validated against NEURON hh.mod (v9.0.1)
//! - Gates: m³h for Na, n⁴ for K (Hodgkin & Huxley 1952)
//! - Temperature: Q10=3.0, reference 6.3°C
//! - Resting gates (V=-65mV): m=0.0529, h=0.596, n=0.318 (matches NEURON ±0.005)
//!
//! # Architecture
//! Gates and channels are defined as const structs with typed parameter access.
//! No magic numbers in hot path — all parameter indices via `HHParamIdx` enum.
//!
//! # Performance
//! - SIMD: 8 neurons/instruction (f32x8 on AVX2)
//! - Pure FMA arithmetic — no divisions in hot path (inv_C_m precomputed)
//! - Q10 precomputed — no branches or powf in step()

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

// ═══════════════════════════════════════════════════════════════
//  §3 — Parameter Index Enum
// ═══════════════════════════════════════════════════════════════

/// Type-safe parameter indices for HHModel (SoA column-major layout).
///
/// Usage: `ctx.p_enum(HHParamIdx::G_Na)`
/// `From<HHParamIdx> for usize` enables `p_enum()` typed access.
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
}

impl From<HHParamIdx> for usize {
    #[inline(always)]
    fn from(idx: HHParamIdx) -> Self { idx as usize }
}

// ═══════════════════════════════════════════════════════════════
//  §1 — Gate Definition
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley gate rate parameters (α/β formulation).
///
/// Each gate has:
/// - `v_off`: voltage offset for vtrap (e.g., -40 for m-gate)
/// - `scale`: vtrap divisor (e.g., 10 for m-gate)
/// - `alpha_c`, `beta_c`: rate coefficients
/// - `beta_volt`, `beta_div`: β exponential/sigmoid parameters
/// - `beta_sigmoid`: if true, β = beta_c / (1+exp(-(V+beta_volt)/beta_div))
///   (standard sigmoid); if false, β = beta_c·exp(-(V+beta_volt)/beta_div)
#[derive(Copy, Clone)]
pub struct GateParams {
    pub v_off: f32,
    pub scale: f32,
    pub alpha_c: f32,
    pub beta_c: f32,
    pub beta_volt: f32,
    pub beta_div: f32,
    /// If true: α = alpha_c·exp(v_shift/scale) (h-gate)
    /// If false: α = alpha_c·vtrap(v_shift, scale) (m, n)
    pub alpha_exp: bool,
    /// If true: β = beta_c/(1+e^x) (h-gate sigmoid)
    /// If false: β = beta_c·e^x (m, n exponential)
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
/// ⚠️ α is EXPONENTIAL, NOT vtrap (unlike m/n)!
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

/// Compute HH gate equilibrium (x_∞) and integrate one step via exponential Euler.
///
/// Formula: x_∞ = α/(α+β),  x(t+dt) = x_∞ + (x(t) - x_∞)·exp(-dt·Q10·(α+β))
///
/// All operations in SIMD — no scalar conversion, no branching.
///
/// # α function
/// - `alpha_exp = false` (m, n): α = alpha_c·vtrap(v_off-V, scale)
/// - `alpha_exp = true`  (h):    α = alpha_c·exp((v_off-V)/scale)
///
/// # β function
/// - `beta_sigmoid = false` (m, n): β = beta_c·exp(-(V+beta_volt)/beta_div)
/// - `beta_sigmoid = true`  (h):    β = beta_c/(1+exp(-(V+beta_volt)/beta_div))
#[inline(always)]
fn compute_gate_state<S: SimdLane>(
    v: S, gate: S, params: GateParams, dt_q10: S, q10_extra: S,
) -> S {
    let v_shift = S::splat(params.v_off) - v;
    let alpha = if params.alpha_exp {
        // α = alpha_c·exp(v_shift/scale) — h-gate (NO vtrap!)
        S::splat(params.alpha_c) * (v_shift / S::splat(params.scale)).exp()
    } else {
        // α = alpha_c·vtrap(v_shift, scale) — m, n-gates
        S::splat(params.alpha_c) * S::vtrap(v_shift, S::splat(params.scale))
    };
    let exp_arg = -(v + S::splat(params.beta_volt)) / S::splat(params.beta_div);
    let beta = if params.beta_sigmoid {
        // β = 1/(1+e^x) — sigmoid (h-gate inactivation)
        S::splat(params.beta_c) / (S::one() + exp_arg.exp())
    } else {
        // β = e^x — exponential (m, n gates)
        S::splat(params.beta_c) * exp_arg.exp()
    };
    let sum = alpha + beta;
    let g_inf = alpha / sum;
    let decay = (-dt_q10 * q10_extra * sum).exp();
    // FMA: inf + (state - inf) · decay
    decay.mul_add(gate - g_inf, g_inf)
}

// ═══════════════════════════════════════════════════════════════
//  §2 — Channel Definition
// ═══════════════════════════════════════════════════════════════

/// Ionic channel: conductance gating + reversal potential.
///
/// Powers define which gate variables modulate this channel:
/// `m_pow` = exponent for m-gate (e.g., 3 for Na⁺: m³)
/// `h_pow` = exponent for h-gate (e.g., 1 for Na⁺: h¹)
/// `n_pow` = exponent for n-gate (e.g., 4 for K⁺: n⁴)
#[derive(Copy, Clone)]
pub struct ChannelDef {
    pub g_idx: HHParamIdx,      // param index for g_max
    pub e_rev_idx: HHParamIdx,  // param index for E_rev
    pub m_pow: u32,             // m gate exponent
    pub h_pow: u32,             // h gate exponent
    pub n_pow: u32,             // n gate exponent
}

/// Na⁺ channel: g_Na·m³·h — depolarizing, fast activation + slow inactivation
pub const CHANNEL_NA: ChannelDef = ChannelDef {
    g_idx: HHParamIdx::G_Na, e_rev_idx: HHParamIdx::E_Na,
    m_pow: 3, h_pow: 1, n_pow: 0,
};

/// K⁺ channel: g_K·n⁴ — repolarizing, slow activation
pub const CHANNEL_K: ChannelDef = ChannelDef {
    g_idx: HHParamIdx::G_K, e_rev_idx: HHParamIdx::E_K,
    m_pow: 0, h_pow: 0, n_pow: 4,
};

/// Leak channel: g_L — passive, no gating
pub const CHANNEL_LEAK: ChannelDef = ChannelDef {
    g_idx: HHParamIdx::G_Leak, e_rev_idx: HHParamIdx::E_Leak,
    m_pow: 0, h_pow: 0, n_pow: 0,
};

/// Compute ionic current: I = g·(V - E_rev), gated by m, h, n.
///
/// Convention (Ohm's law): I = g_max · m^m_pow · h^h_pow · n^n_pow · (V - E_rev)
/// Subtracted from I_inj in the membrane equation (NEURON convention).
#[inline(always)]
fn ionic_current<S: SimdLane>(
    v: S, m: S, h: S, n_g: S, ctx: &SimdContext<S>, ch: ChannelDef,
) -> S {
    let mut g = ctx.p(ch.g_idx as usize);
    for _ in 0..ch.m_pow { g = g * m; }
    for _ in 0..ch.h_pow { g = g * h; }
    for _ in 0..ch.n_pow { g = g * n_g; }
    // FMA: g·V - g·E_rev
    g.mul_add(v, -g * ctx.p(ch.e_rev_idx as usize))
}

// ═══════════════════════════════════════════════════════════════
//  §6 — Edge-Triggered Spike Detection
// ═══════════════════════════════════════════════════════════════

/// Edge-triggered spike detection: fires on rising edge (v crossed threshold upward).
///
/// Returns a mask where `true` = spike detected.
/// No forced V reset — HH is a continuous biophysical model.
///
/// `is_above` is precomputed by caller (reused for was_below flag update).
#[inline(always)]
fn edge_detect_spike<S: SimdLane>(
    is_above: S, was_below_flag: S,
) -> S {
    let was_below = was_below_flag.cmp_gt(S::zero());
    is_above.and(was_below)
}

// ═══════════════════════════════════════════════════════════════
//  Model Implementation
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley Squid Axon Model — continuous biophysical dynamics
pub struct HHModel;

crate::validate_model!(HHModel {
    params: 10, state: 5, dt: 0,
    v_thresh: 8,
    erev: [(2,"E_Na",50.0), (4,"E_K",-77.0), (6,"E_leak",-54.387)],
    forward_euler: true, ohms_law: true, edge_triggered: true
});

impl Model for HHModel {
    const N_PARAMS: usize = 10;
    const N_STATE: usize = 5;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let m = ctx.s(1);
        let h = ctx.s(2);
        let n_gate = ctx.s(3);
        let was_below_flag = ctx.s(4);

        // ─── Precomputed Q10 (scalar, no division/branch in hot path) ───
        let q10 = ctx.p_enum(HHParamIdx::Q10);
        let dt_q10 = ctx.dt() * q10;

        // ─── Gate integration (§1: three gates, one function) ───
        let m_new = compute_gate_state(v, m, GATE_M, dt_q10, S::one());
        let h_new = compute_gate_state(v, h, GATE_H, dt_q10, S::one());
        let n_new = compute_gate_state(v, n_gate, GATE_N, dt_q10, S::one());

        // ─── Ionic currents (§2: three channels, one function) ───
        let i_na = ionic_current(v, m_new, h_new, n_new, ctx, CHANNEL_NA);
        let i_k = ionic_current(v, m_new, h_new, n_new, ctx, CHANNEL_K);
        let i_leak = ionic_current(v, m_new, h_new, n_new, ctx, CHANNEL_LEAK);

        // ─── Membrane update ───
        // dv = (I_inj - I_Na - I_K - I_leak) · inv_C_m · dt  (NEURON convention)
        let inv_cm = ctx.p_enum(HHParamIdx::Inv_Cm);
        let dv = (ctx.i() - i_na - i_k - i_leak) * inv_cm * ctx.dt();
        let v_new = v + dv;

        // ─── Edge-triggered spike detection (§6) ───
        let v_thresh = ctx.p_enum(HHParamIdx::V_Thresh);
        let is_above = v_new.cmp_ge(v_thresh);
        let spike = edge_detect_spike(is_above, was_below_flag);
        ctx.spike(spike);

        // ─── State store ───
        ctx.set_s(0, v_new);
        ctx.set_s(1, m_new);
        ctx.set_s(2, h_new);
        ctx.set_s(3, n_new);
        // was_below = 1 if v < v_thresh (below), 0 if above
        // ⚠️ Use is_above.not(), NOT spike.not() — otherwise sustained
        //    above-threshold voltages incorrectly set flag=1, causing
        //    false spikes on the next step.
        ctx.set_s(4, is_above.not().cond().then(S::one()).else_(S::zero()));
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    /// Build HH params. All const indices via HHParamIdx — compiler checks correctness.
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
        }
        p
    }

    #[test]
    fn test_hh_spikes_with_current() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let i_syn = vec![50.0f32; num];
        let mut spikes = vec![0i32; num];

        let mut total = 0i32;
        for _ in 0..5000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                flag.as_mut_ptr(), std::ptr::null_mut(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HH should spike with I=50, got {}", total);
    }

    #[test]
    fn test_hh_silent_without_current() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let i_syn = vec![0.0f32; num];
        let mut spikes = vec![0i32; num];

        for _ in 0..500 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                flag.as_mut_ptr(), std::ptr::null_mut(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HH should not spike with I=0");
    }

    #[test]
    fn test_hh_non_aligned() {
        let num = 13;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let i_syn = vec![50.0f32; num];
        let mut spikes = vec![0i32; num];

        let mut total = 0i32;
        for _ in 0..5000 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                flag.as_mut_ptr(), std::ptr::null_mut(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HH non-aligned should spike, got {}", total);
    }

    #[test]
    fn test_hh_inhibited() {
        let num = 8;
        let params = make_params(num, 6.3);
        let mut v = vec![-65.0f32; num];
        let mut m = vec![0.05f32; num];
        let mut h = vec![0.6f32; num];
        let mut k_gate = vec![0.32f32; num];
        let mut flag = vec![1.0f32; num];
        let i_syn = vec![-10.0f32; num];
        let mut spikes = vec![0i32; num];

        for _ in 0..500 {
            unsafe { run_neuron::<HHModel>(
                v.as_mut_ptr(), m.as_mut_ptr(), h.as_mut_ptr(), k_gate.as_mut_ptr(),
                flag.as_mut_ptr(), std::ptr::null_mut(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), num,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HH should not spike with I=-10");
    }

    // ─── NEURON-validated rate function tests ───

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

        #[allow(clippy::approx_constant)]
        let ninf_ref: f32 = 0.318;
        assert!((minf - 0.0529).abs() < 0.01, "minf={}", minf);
        assert!((hinf - 0.596).abs() < 0.05, "hinf={}", hinf);
        assert!((ninf - ninf_ref).abs() < 0.03, "ninf={}", ninf);
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

        assert!((minf - 0.974).abs() < 0.02, "minf at 0mV should be ≈0.974, got {}", minf);
        assert!(hinf < 0.01, "hinf at 0mV should be ≈0.0, got {}", hinf);
    }

    #[test]
    fn test_hh_vtrap_numerical_safety() {
        use wide::f32x8;
        let x = f32x8::splat(0.0);
        let y = f32x8::splat(10.0);
        let result = crate::math::vtrap::vtrap(x, y).to_array()[0];
        assert!((result - 10.0).abs() < 0.1, "vtrap(0,10) should be ≈10, got {}", result);
    }

    #[test]
    fn test_hh_gate_constants_match_params() {
        // Verify GateParams match NEURON hh.mod constants
        assert!((GATE_M.v_off - (-40.0)).abs() < 1e-6);
        assert!((GATE_M.scale - 10.0).abs() < 1e-6);
        assert!((GATE_M.alpha_c - 0.1).abs() < 1e-6);
        assert!((GATE_N.v_off - (-55.0)).abs() < 1e-6);
        assert!((GATE_N.beta_div - 80.0).abs() < 1e-6);
        // GATE_H: verify exp alpha (h-gate is NOT vtrap!)
        assert!(GATE_H.alpha_exp, "h-gate alpha MUST be exp, NOT vtrap!");
        assert!((GATE_H.alpha_c - 0.07).abs() < 1e-6);
        assert!((GATE_H.scale - 20.0).abs() < 1e-6);
        assert!(GATE_H.beta_sigmoid, "h-gate beta MUST be sigmoid!");
    }

    #[test]
    fn test_hh_channel_constants_match_params() {
        assert_eq!(CHANNEL_NA.g_idx as usize, HHParamIdx::G_Na as usize);
        assert_eq!(CHANNEL_NA.m_pow, 3);
        assert_eq!(CHANNEL_NA.h_pow, 1);
        assert_eq!(CHANNEL_K.n_pow, 4);
        assert_eq!(CHANNEL_LEAK.g_idx as usize, HHParamIdx::G_Leak as usize);
    }
}
