//! # KM — M-Type K⁺ Current (Spike Frequency Adaptation)
//!
//! ```text
//! I_M = g_M · m · (V - E_K)
//! m_∞(V) = 1 / (1 + exp(-(V - v_half) / k))
//! dm/dt = (m_∞ - m) / τ_m
//! ```
//!
//! KM channels generate the M-current (muscarinic-sensitive).
//! Slow, non-inactivating K⁺ current. Main driver of spike
//! frequency adaptation (slows firing during sustained input).
//!
//! Parameters: V_half ≈ -35 mV, k ≈ 5, τ_m ≈ 50-100 ms
//! Blocked by: muscarine, XE-991, linopirdine

use crate::core::lane::SimdLane;

/// One step of KM current: integrates m-gate, returns I_M.
/// Parameters are `S` (SIMD vectors) because they're per-neuron in SoA.
/// `v_half` and `k` are scalar constants (same for all neurons in group).
#[inline(always)]
pub fn km_current<S: SimdLane>(
    v: S,
    m: S,
    g_max: S,
    e_rev: S,
    v_half: f32,
    k: f32,
    inv_tau: S,
    dt: S,
) -> (S, S) {
    let m_inf = S::sigmoid(v, v_half, k);
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
    fn test_km_resting() {
        let v = f32x8::splat(-65.0);
        let m = f32x8::splat(0.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0);
        let vh: f32 = -35.0;
        let k: f32 = 5.0;
        let inv_tau = f32x8::splat(1.0 / 50.0);
        let dt = f32x8::splat(1.0);
        let (i, _m_new) = km_current(v, m, g, e, vh, k, inv_tau, dt);
        // At -65mV, far below -35, m_inf ≈ 0, current near 0
        assert!(
            i.to_array()[0].abs() < 0.1,
            "KM at -65 should be small: {:.6}",
            i.to_array()[0]
        );
    }
}
