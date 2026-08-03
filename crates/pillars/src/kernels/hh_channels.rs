//! # HH Channel Definitions — standard squid-axon & pacemaking (CaV-T)
//!
//! Port of the channel-spec layer from the legacy `crates/kernels/src/neuron/hh.rs`.
//! This module owns **what** a conductance-based model is made of — the channel
//! list, per-gate rate functions α(V)/β(V) and steady-state helpers — while the
//! integrator itself lives in the zoo (`HHModel`) and in `tests` here.
//!
//! # Channel sets
//!
//! - [`standard_channels`] — NaV (m³h) + KV (n⁴): the classic Hodgkin-Huxley
//!   squid axon. Requires external drive to spike.
//! - [`pacemaking_channels`] — standard + **CaV-T (T-type calcium)**: a single
//!   low-threshold activation gate `c`. The T-type current depolarizes the
//!   membrane past NaV threshold on its own → the neuron fires **spontaneously**
//!   without any input current (the basis of rhythmic VNC/CPG activity).

use crate::math::fast_exp::fast_exp;

// ═══════════════════════════════════════════════════════════
//  Channel specification types
// ═══════════════════════════════════════════════════════════

/// Rate-function closure: α(V) or β(V) in [1/ms].
pub type RateFn = fn(f32) -> f32;

/// One gating variable (e.g. Na-m, Na-h, K-n, CaT-c).
#[derive(Debug, Clone, Copy)]
pub struct GateSpec {
    pub label: &'static str,
    pub alpha: RateFn,
    pub beta: RateFn,
    /// Exponent in the conductance term: `gating = gate^power`.
    pub power: u32,
}

/// One ion channel: one or more gates + reversal potential + max conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    /// Reversal potential in mV.
    pub E_rev: f32,
    /// Max conductance in mS/cm².
    pub g_max: f32,
}

// ═══════════════════════════════════════════════════════════
//  Squid-axon rate functions (Hodgkin & Huxley 1952, 6.3 °C)
// ═══════════════════════════════════════════════════════════

/// vtrap: `x / (exp(x/y) − 1)` with a Taylor `y − x/2` when `|x/y| < 1e-6`.
/// Prevents 0/0 at x → 0, exactly like NEURON's `FUNCTION vtrap`.
///
/// This is the HH rate-function helper (x = voltage shift, y = slope divisor).
/// Do not confuse it with [`crate::math::vtrap::vtrap`] — the sigmoid-shaped
/// per-lane primitive used by gates — the two share a name but not semantics.
#[inline(always)]
pub fn vtrap(x: f32, y: f32) -> f32 {
    let ratio = x / y;
    if ratio.abs() < 1e-6 {
        y - x * 0.5
    } else {
        x / (fast_exp(ratio) - 1.0)
    }
}

/// Na activation (m-gate) α: `0.1·vtrap(−(v+40), 10)`.
pub fn na_m_alpha(v: f32) -> f32 {
    0.1 * vtrap(-(v + 40.0), 10.0)
}
/// Na activation (m-gate) β: `4·exp(−(v+65)/18)`.
pub fn na_m_beta(v: f32) -> f32 {
    4.0 * fast_exp(-(v + 65.0) / 18.0)
}

/// Na inactivation (h-gate) α: `0.07·exp(−(v+65)/20)`.
pub fn na_h_alpha(v: f32) -> f32 {
    0.07 * fast_exp(-(v + 65.0) / 20.0)
}
/// Na inactivation (h-gate) β: `1/(1 + exp(−(v+35)/10))`.
pub fn na_h_beta(v: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-(v + 35.0) / 10.0))
}

/// K activation (n-gate) α: `0.01·vtrap(−(v+55), 10)`.
pub fn k_n_alpha(v: f32) -> f32 {
    0.01 * vtrap(-(v + 55.0), 10.0)
}
/// K activation (n-gate) β: `0.125·exp(−(v+65)/80)`.
pub fn k_n_beta(v: f32) -> f32 {
    0.125 * fast_exp(-(v + 65.0) / 80.0)
}

/// CaV-T (T-type calcium) activation — low-threshold, pacemaking.
/// Hill & Tononi 2005 / Wang & Rinzel 1993. Single gate `c`, τ_c = 0.6 ms.
/// α = m_inf/τ where m_inf = 1/(1 + exp(−(v+56)/6.2)).
pub fn cat_c_alpha(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    m_inf / 0.6
}
/// CaV-T β: `(1 − m_inf)/τ`.
pub fn cat_c_beta(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    (1.0 - m_inf) / 0.6
}

// ═══════════════════════════════════════════════════════════
//  Channel sets
// ═══════════════════════════════════════════════════════════

/// Classic HH squid axon: NaV (m³h) + KV (n⁴).
pub fn standard_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec { label: "m", alpha: na_m_alpha, beta: na_m_beta, power: 3 },
                GateSpec { label: "h", alpha: na_h_alpha, beta: na_h_beta, power: 1 },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![GateSpec { label: "n", alpha: k_n_alpha, beta: k_n_beta, power: 4 }],
            E_rev: -77.0,
            g_max: 36.0,
        },
    ]
}

/// HH + CaV-T (T-type calcium) for **pacemaking**: 4 gates (NaV-m, NaV-h, KV-n,
/// CaT-c). The T-type current makes neurons fire spontaneously without input.
pub fn pacemaking_channels() -> Vec<ChannelSpec> {
    let mut ch = standard_channels();
    ch.push(ChannelSpec {
        name: "CaT",
        gates: vec![GateSpec { label: "c", alpha: cat_c_alpha, beta: cat_c_beta, power: 1 }],
        E_rev: 120.0,
        g_max: 0.5,
    });
    ch
}

// ═══════════════════════════════════════════════════════════
//  Steady-state helpers
// ═══════════════════════════════════════════════════════════

/// Steady-state gate values `m_inf = α(V)/(α(V)+β(V))` at a given potential,
/// in gate order (channel-major).
pub fn steady_state_gates(v: f32, channels: &[ChannelSpec]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channels {
        for gate in &ch.gates {
            let a = (gate.alpha)(v);
            let b = (gate.beta)(v);
            let sum = a + b;
            gates.push(if sum > 1e-12 { a / sum } else { 0.5 });
        }
    }
    gates
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar exponential-Euler HH step over arbitrary channels.
    /// Returns `true` if a spike (reverse zero-crossing through 0 mV) occurred.
    #[allow(clippy::too_many_arguments)]
    fn hh_step_scalar(
        v: &mut f32,
        gates: &mut [f32],
        i_ext: f32,
        channels: &[ChannelSpec],
        g_leak: f32,
        e_leak: f32,
        c_m: f32,
        dt: f32,
    ) -> bool {
        let v_old = *v;
        let mut i_ion = 0.0f32;
        let mut gi = 0usize;
        for ch in channels {
            let mut g_ch = ch.g_max;
            for gate in &ch.gates {
                let a = (gate.alpha)(*v);
                let b = (gate.beta)(*v);
                let sum = a + b;
                let m_inf = if sum > 1e-12 { a / sum } else { 0.5 };
                let decay = fast_exp(-dt * sum.max(1e-12));
                let gv = &mut gates[gi];
                *gv = m_inf + (*gv - m_inf) * decay;
                g_ch *= gv.powi(gate.power as i32);
                gi += 1;
            }
            i_ion += g_ch * (*v - ch.E_rev);
        }
        let i_leak = g_leak * (*v - e_leak);
        let dv = (i_ext - i_ion - i_leak) / c_m;
        let v_new = *v + dv * dt;
        *v = v_new;
        v_old >= 0.0 && v_new < 0.0
    }

    /// Standard squid axon: must NOT fire spontaneously with zero input.
    #[test]
    fn standard_channels_quiescent_without_input() {
        let channels = standard_channels();
        let dt = 0.05;
        let mut v = -65.0f32;
        let mut gates = steady_state_gates(v, &channels);
        let mut spikes = 0u32;
        for _ in 0..4000 {
            // 200 ms
            if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
                spikes += 1;
            }
        }
        assert_eq!(spikes, 0, "standard HH should not self-spike, got {spikes}");
        // Resting potential of the squid axon is ≈ −65 mV (NOT the leak
        // reversal −54.387); the membrane must sit near rest, not drift.
        assert!(
            v > -75.0 && v < -45.0,
            "v should stay near resting potential, got {v}"
        );
    }

    /// Pacemaking (CaV-T): must fire spontaneously with zero input.
    #[test]
    fn pacemaking_fires_spontaneously() {
        let channels = pacemaking_channels();
        let dt = 0.05;
        let mut v = -65.0f32;
        let mut gates = steady_state_gates(v, &channels);
        let mut spikes = 0u32;
        for _ in 0..4000 {
            // 200 ms
            if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
                spikes += 1;
            }
            assert!(v.is_finite() && v > -120.0 && v < 80.0, "v runaway: {v}");
        }
        assert!(spikes >= 3, "pacemaking HH should fire ≥3 spikes in 200 ms, got {spikes}");
    }

    /// Gate ordering: 4 gates for pacemaking (m, h, n, c), steady states in [0,1].
    #[test]
    fn gate_order_and_steady_state() {
        let ch = pacemaking_channels();
        assert_eq!(ch.len(), 3);
        let n_gates: usize = ch.iter().map(|c| c.gates.len()).sum();
        assert_eq!(n_gates, 4);
        let g = steady_state_gates(-65.0, &ch);
        assert_eq!(g.len(), 4);
        for (i, &x) in g.iter().enumerate() {
            assert!((0.0..=1.0).contains(&x), "gate {i} = {x} out of [0,1]");
        }
        // CaV-T is the last gate (index 3)
        assert_eq!(ch[2].name, "CaT");
        assert_eq!(ch[2].gates[0].label, "c");
        assert_eq!(ch[2].E_rev, 120.0);
    }

    /// Rate-function sanity: steady-state m_inf saturates with depolarization.
    #[test]
    fn na_m_inf_saturates() {
        let channels = standard_channels();
        let g_hyper = steady_state_gates(-80.0, &channels);
        let g_depol = steady_state_gates(0.0, &channels);
        assert!(g_depol[0] > g_hyper[0], "m_inf should rise with depolarization");
    }
}
