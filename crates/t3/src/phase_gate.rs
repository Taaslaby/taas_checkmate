//! Phase Gate — sigmoid attenuator for cross-coupling.
//!
//! Из оригинального taas_chaser toroidal.rs.
//! Защищает медленные слои от шума быстрых через order parameter R.
//!
//! ```text
//! G(r) = 1 / (1 + exp(-gain · (r - γ/√N)))
//! ```

/// Sigmoid phase gate: G(r) = 1/(1 + exp(-gain * (r - threshold))).
///
/// - `r`: order parameter этой популяции
/// - `gamma`: layer-specific threshold factor
/// - `n`: количество осцилляторов в популяции
/// - `gain`: крутизна sigmoid (обычно 4.0–8.0)
///
/// Если R < γ/√N, выход ≈ 0 (gate closed).
/// Если R >> γ/√N, выход ≈ 1 (gate open).
pub fn phase_gate(r: f32, gamma: f32, n: usize, gain: f32) -> f32 {
    let threshold = gamma / (n as f32).sqrt();
    1.0 / (1.0 + (-gain * (r - threshold)).exp())
}

/// Apply phase gate to cross-coupling weight.
/// `w_gated = w * gate(source_layer_r)`
pub fn gate_coupling(w: f32, r_source: f32, gamma: f32, n_source: usize, gain: f32) -> f32 {
    w * phase_gate(r_source, gamma, n_source, gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_gate_high_r() {
        // threshold = 0.5/sqrt(16) = 0.125, R=1.0 >> threshold → gate ≈ 1
        let g = phase_gate(1.0, 0.5, 16, 8.0);
        assert!(g > 0.999, "high R should open gate: {}", g);
    }

    #[test]
    fn test_phase_gate_low_r() {
        // threshold = 2.0/sqrt(8) ≈ 0.707, R=0.0 << threshold → gate ≈ 0
        let g = phase_gate(0.0, 2.0, 8, 8.0);
        assert!(g < 0.01, "low R should close gate: {}", g);
    }

    #[test]
    fn test_phase_gate_threshold() {
        let gamma = 0.5;
        let n = 32;
        let threshold = gamma / (n as f32).sqrt();
        let g = phase_gate(threshold, gamma, n, 4.0);
        assert!((g - 0.5).abs() < 0.1, "at threshold gate≈0.5: {}", g);
    }

    #[test]
    fn test_gate_coupling() {
        let w = 0.5;
        let gated = gate_coupling(w, 1.0, 0.5, 16, 8.0);
        assert!((gated - w).abs() < 0.01, "high R → w unchanged: {}", gated);
    }
}
