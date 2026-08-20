// ═══════════════════════════════════════════════════════════
//  e-prop
// ═══════════════════════════════════════════════════════════

/// Surrogate gradient: derivative of the spike threshold function.
///
/// Fast sigmoid surrogate: `σ'(v) = 1 / (1 + |v − v_th|/τ)²`.
#[inline(always)]
pub fn surrogate_gradient(v: f32, v_thresh: f32, tau_surrogate: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_surrogate;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

/// E-prop standalone weight update: `Δw = η · L_j · e_ji`.
///
/// All three arrays have length `n`; the update is elementwise.
pub fn eprop_update(weights: &mut [f32], eligibility: &[f32], learning_signals: &[f32], lr: f32) {
    let n = weights
        .len()
        .min(eligibility.len())
        .min(learning_signals.len());
    for i in 0..n {
        weights[i] += lr * learning_signals[i] * eligibility[i];
    }
}
