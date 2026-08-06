//! # HCN — Hyperpolarization-Activated Cyclic Nucleotide-Gated Current
//!
//! ```text
//! I_h = g_h · m · (V - E_h)
//! m_∞(V) = 1 / (1 + exp((V - v_half) / k))
//! dm/dt = (m_∞ - m) / τ_m  (exponential Euler)
//! ```
//!
//! The "h-current" or "pacemaker current". UNIQUE: ACTIVATED by
//! HYPERPOLARIZATION. At rest (-65mV), I_h is partially activated.
//!
//! Parameters (CA1 neuron, ~37°C):
//!   V_half ≈ -75 mV, k ≈ -5.5, τ_m ≈ 30 ms, E_h ≈ -30 mV
//!
//! In Drosophila: HCN-like in LPTC (optomotor response).

use crate::core::lane::SimdLane;

/// One step of HCN current: integrates m-gate, returns I_h.
///
/// `v_half` and `k` are scalar constants (same for all neurons in group).
/// HCN uses INVERTED sigmoid: `k` positive → activates at hyperpolarization.
/// Implemented via `S::sigmoid(v, v_half, -k)` where negative k inverts.
#[inline(always)]
pub fn hcn_current<S: SimdLane>(
    v: S,
    m: S,
    g_max: S,
    e_rev: S,
    v_half: f32,
    k: f32,
    inv_tau: S,
    dt: S,
) -> (S, S) {
    // m_inf = 1 / (1 + exp(-(V - v_half) / k)) — standard sigmoid
    // For HCN: k is positive → pass sign-inverted: -k inverts the sigmoid
    let m_inf = S::sigmoid(v, v_half, -k); // same as 1/(1+exp((V-v_half)/k))
                                           // Exponential Euler: m' = m_inf + (m - m_inf) · exp(-dt·inv_tau)
    let decay = (-dt * inv_tau).exp();
    let m_new = m_inf + (m - m_inf) * decay;
    let i = g_max * m_new * (v - e_rev);
    (i, m_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_hcn_hyperpolarized_activates() {
        let v = f32x8::splat(-100.0); // hyperpolarized
        let m = f32x8::splat(0.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-30.0);
        let vh: f32 = -75.0;
        let k: f32 = 5.5;
        let inv_tau = f32x8::splat(1.0 / 30.0);
        let dt = f32x8::splat(1.0);
        let (i, _m_new) = hcn_current(v, m, g, e, vh, k, inv_tau, dt);
        // At V=-100, far below v_half=-75: m_inf ≈ 1/(1+exp((-100+75)/5.5)) = 1/(1+exp(-4.54)) = 0.99
        // I ≈ 1*0.99*(-100+30) = -69.3 (inward, depolarizing)
        assert!(
            i.to_array()[0] < 0.0,
            "HCN at -100mV should be inward: {:.4}",
            i.to_array()[0]
        );
    }
}
