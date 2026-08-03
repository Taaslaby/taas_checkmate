//! # Field — Mean-Field Gap Junction and Ionic Drift-Diffusion
//!
//! ## `mean_field_gap`
//! ```text
//! I = g · (V - α · V_avg)
//! ```
//! Mean-field gap junction: neuron coupled to average population voltage.
//! This is NOT true ephaptic coupling (which modifies gate activation energies).
//! Ephaptic coupling requires shifting V_m = V_i - V_e, which changes gate
//! activation (m_∞, h_∞), not just adding a current.
//!
//! ## `ion_drift_diffusion`
//! ```text
//! J = -D · ∇C - mobility · C · ∇V
//! dC/dt = D · ∇²C - mobility · E · ∇C
//! ```
//! Simplified Nernst-Planck for ion movement through extracellular space.

use crate::core::lane::SimdLane;

/// Mean-field gap junction: `I = g · (V - α · V_avg)`
///
/// This is a gap junction between a neuron and the population average.
/// `α` is the coupling strength to the mean field.
/// Biological implementation: many gap junctions + large neighbor pool.
///
/// ⚠️ NOT ephaptic coupling. True ephaptic coupling shifts the
/// extracellular potential V_e, affecting gate voltage dependence:
/// V_m_effective = V_i - V_e. This requires modifying α/β rate functions.
#[inline(always)]
pub fn mean_field_gap<S: SimdLane>(v: S, v_avg: S, coupling: S, g_gap: S) -> S {
    g_gap * (v - coupling * v_avg)
}

/// 1D drift-diffusion: `C' = C + (D·∇²C - μ·E·∇C)·dt`
#[inline(always)]
pub fn drift_diffusion_1d<S: SimdLane>(c: S, c_left: S, c_right: S, e_field: S,
    diff_const: S, mobility: S, inv_dx2: S, inv_2dx: S, dt: S) -> S
{
    let laplacian = (c_left + c_right - S::splat(2.0) * c) * inv_dx2;
    let gradient = (c_right - c_left) * inv_2dx;
    let transport = diff_const * laplacian - mobility * e_field * gradient;
    c + transport * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_mean_field_gap_zero_conductance() {
        let v = f32x8::splat(-65.0);
        let v_avg = f32x8::splat(-65.0);
        let alpha = f32x8::splat(1.0);
        let g = f32x8::splat(0.0);  // g = 0 → no current
        let i = mean_field_gap(v, v_avg, alpha, g);
        assert!(i.to_array()[0].abs() < 1e-6, "no conductance → no current, got {:.8}", i.to_array()[0]);
    }

    #[test]
    fn test_mean_field_gap_depol() {
        let v = f32x8::splat(-60.0);
        let v_avg = f32x8::splat(-65.0);
        let alpha = f32x8::splat(1.0);
        let g = f32x8::splat(0.1);
        let i = mean_field_gap(v, v_avg, alpha, g);
        // I = 0.1 * (-60 - 1*(-65)) = 0.1 * 5 = 0.5 (positive → hyperpolarizing for v=-60)
        assert!((i.to_array()[0] - 0.5).abs() < 1e-6);
    }
}
