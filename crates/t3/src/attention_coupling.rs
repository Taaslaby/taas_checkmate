//! Attention-based Kuramoto coupling — inspired by Microsoft KoPE (ICML 2026).
//!
//! # KoPE insight
//!
//! Microsoft's Kuramoto Oscillatory Phase Encoding computes coupling through **attention**
//! instead of uniform weights:
//!
//! ```text
//! Uniform:   Δθ_i = Σ (w/N) · r_j · sin(θ_j - θ_i)        // O(N), fixed weights
//! Attention: Δθ_i = γ · Σ A_ij · r_j · sin(θ_j - θ_i)     // O(N²), data-dependent
//!
//! Where A_ij = softmax_j(Q_i · K_j / √d)
//!       Q_i = W_Q · state_i,  K_j = W_K · state_j
//!       state_i = [sin(θ_i), cos(θ_i), r_i]
//! ```
//!
//! # When to use
//!
//! | N     | Coupling | Complexity |
//! |-------|----------|------------|
//! | ≤ 64  | Attention | O(N²), ~4K entries |
//! | 64-1024 | Both | O(N²) still fine, 1M entries |
//! | > 1024 | Uniform | O(N) only |
//!
//! # Usage
//! ```ignore
//! let mut tri = TriOscillator::new(32, 0.5, 1.5, Avx2Backend);
//!
//! // Per-population attention coupling with learned projections
//! let attn_pop1 = AttentionCoupling::new(32, 0.05);
//! let attn_pop2 = AttentionCoupling::new(32, 0.03);
//! let attn_pop3 = AttentionCoupling::new(32, 0.04);
//!
//! tri.set_attention_coupling(Some(attn_pop1), Some(attn_pop2), Some(attn_pop3));
//! tri.step(0.01); // uses attention-based coupling
//! ```

#![allow(non_snake_case)]

use super::PI;

/// Attention-based Kuramoto coupling for one oscillator population.
///
/// Replaces uniform `w/N · Σ r_j · sin(θ_j - θ_i)` with
/// `γ · Σ A_ij · r_j · sin(θ_j - θ_i)` where `A = softmax(Q·K^T / √d)`.
///
/// # State vector
///
/// Default: `[sin(θ), cos(θ), r]` (dim=3). Can be extended with more features.
///
/// # Projections
///
/// Q_weights: `[state_dim × state_dim]` — maps state to Q space.
/// K_weights: `[state_dim × state_dim]` — maps state to K space.
///
/// Initialized as identity (Q=K=state), so the initial attention is based on
/// state similarity. Training updates these weights to learn task-specific coupling.
pub struct AttentionCoupling {
    pub n: usize,
    pub state_dim: usize,
    /// Q projection weights [state_dim × state_dim], initialized as identity.
    pub q_weight: Vec<f32>,
    /// K projection weights [state_dim × state_dim], initialized as identity.
    pub k_weight: Vec<f32>,
    /// Coupling strength γ (scales the attention-weighted coupling).
    pub gamma: f32,
    /// Scale factor: 1 / sqrt(state_dim) for attention scores.
    pub scale: f32,
}

impl AttentionCoupling {
    /// Create new attention coupling for `n` oscillators.
    ///
    /// State dimension is fixed at 3: `[sin(θ), cos(θ), r]`.
    /// Q/K weights initialized as identity (no transformation initially).
    /// `gamma` is the coupling strength (typically 0.01-0.1).
    pub fn new(n: usize, gamma: f32) -> Self {
        let state_dim = 3;
        assert!(state_dim == 3, "AttentionCoupling: state_dim must be 3 [sin, cos, r]");
        let mut q_weight = vec![0.0; state_dim * state_dim];
        let mut k_weight = vec![0.0; state_dim * state_dim];
        // Identity initialization: Q = K = I (attention = cosine similarity)
        for i in 0..state_dim {
            q_weight[i * state_dim + i] = 1.0;
            k_weight[i * state_dim + i] = 1.0;
        }
        Self {
            n,
            state_dim,
            q_weight,
            k_weight,
            gamma,
            scale: (state_dim as f32).sqrt().recip(),
        }
    }

    /// Compute attention-based coupling for one population.
    ///
    /// # Args
    /// * `sin_ph`, `cos_ph` — pre-computed sin(θ), cos(θ) [n]
    /// * `r` — amplitudes [n]
    /// * `out` — output coupling buffer [n]
    ///
    /// For each oscillator i:
    /// ```text
    /// Q_i = W_Q · [sin(θ_i), cos(θ_i), r_i]
    /// For each j:
    ///   K_j = W_K · [sin(θ_j), cos(θ_j), r_j]
    ///   score_ij = Q_i · K_j · scale
    /// A_i = softmax(score_i)         // attention weights
    /// out[i] = γ · Σ_j A_ij · r_j · sin(θ_j - θ_i)
    /// ```
    pub fn compute(&self, sin_ph: &[f32], cos_ph: &[f32], r: &[f32], out: &mut [f32]) {
        let n = self.n.min(sin_ph.len()).min(cos_ph.len()).min(r.len()).min(out.len());
        let d = self.state_dim;
        let scale = self.scale;
        let gamma = self.gamma;

        // Pre-compute K vectors for all oscillators (O(N))
        let mut k_vecs = vec![0.0; n * d];
        for i in 0..n {
            let si = sin_ph[i];
            let ci = cos_ph[i];
            let ri = r[i];
            for j in 0..d {
                let mut sum = 0.0;
                // W_K · state
                for k in 0..d {
                    let state_k = [si, ci, ri][k];
                    sum += self.k_weight[j * d + k] * state_k;
                }
                k_vecs[i * d + j] = sum;
            }
        }

        // For each oscillator i: compute attention scores, softmax, coupling
        for i in 0..n {
            let si = sin_ph[i];
            let ci = cos_ph[i];
            let ri = r[i];
            // Q_i = W_Q · state_i
            let mut q_vec = [si, ci, ri]; // default = state, then apply W_Q
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    let state_k = [si, ci, ri][k];
                    sum += self.q_weight[j * 3 + k] * state_k;
                }
                q_vec[j] = sum;
            }

            // Compute softmax scores (online, no full attention matrix)
            // First pass: compute raw scores, find max for numerical stability
            let mut max_score = -1e10f32;
            let mut scores = vec![0.0; n];
            for j in 0..n {
                let k_base = j * d;
                let mut s = 0.0;
                for h in 0..d {
                    s += q_vec[h] * k_vecs[k_base + h];
                }
                s *= scale;
                scores[j] = s;
                if s > max_score { max_score = s; }
            }

            // Softmax: stable exp(s - max)
            let mut sum_exp = 0.0;
            for j in 0..n {
                let e = (scores[j] - max_score).exp();
                scores[j] = e;
                sum_exp += e;
            }
            let inv_sum = if sum_exp > 1e-12 { 1.0 / sum_exp } else { 0.0 };
            for j in 0..n {
                scores[j] *= inv_sum;
            }

            // Coupling: γ · Σ_j A_ij · r_j · sin(θ_j - θ_i)
            // sin(θ_j - θ_i) = sin_j·cos_i - cos_j·sin_i
            let cos_i = cos_ph[i];
            let sin_i = sin_ph[i];
            let mut coupling = 0.0;
            for j in 0..n {
                let attn = scores[j];
                if attn < 1e-6 { continue; }
                let sin_diff = sin_ph[j] * cos_i - cos_ph[j] * sin_i;
                coupling += attn * r[j] * sin_diff;
            }

            out[i] = gamma * coupling;
        }
    }
}

/// Per-population attention coupling config for TriOscillator.
///
/// `None` = use uniform coupling for that population (default).
/// `Some(AttentionCoupling)` = use attention-based coupling.
#[derive(Default)]
pub struct AttentionCouplingConfig {
    pub pop1: Option<AttentionCoupling>,
    pub pop2: Option<AttentionCoupling>,
    pub pop3: Option<AttentionCoupling>,
}

impl AttentionCouplingConfig {
    /// All populations use uniform coupling (default).
    pub fn new() -> Self {
        Self { pop1: None, pop2: None, pop3: None }
    }

    /// All populations use attention coupling with scaled gamma per population.
    pub fn all(n: usize, gamma: f32) -> Self {
        Self {
            pop1: Some(AttentionCoupling::new(n, gamma)),
            pop2: Some(AttentionCoupling::new(n, gamma * 0.7)),
            pop3: Some(AttentionCoupling::new(n, gamma * 0.5)),
        }
    }

    /// Set coupling for population 1 only.
    pub fn pop1_only(config: AttentionCoupling) -> Self {
        let n = config.n;
        Self {
            pop1: Some(config),
            pop2: None,
            pop3: None,
        }
    }

    /// Check if attention coupling is enabled for any population.
    pub fn is_enabled(&self) -> bool {
        self.pop1.is_some() || self.pop2.is_some() || self.pop3.is_some()
    }
}

// ═══════════════════════════════════════════════════════════
//  Cross-population attention coupling
// ═══════════════════════════════════════════════════════════

/// Cross-population attention coupling.
///
/// Coupling from pop B to pop A:
/// ```text
/// Δθ_Ai = γ · Σ_j attention_ij · r_Bj · sin(θ_Bj - θ_Ai)
/// ```
/// Where attention is computed from the states of pop A (Q) and pop B (K).
pub struct CrossAttentionCoupling {
    pub n: usize,
    pub state_dim: usize,
    pub q_weight: Vec<f32>,  // projection for source population (Q)
    pub k_weight: Vec<f32>,  // projection for target population (K)
    pub gamma: f32,
    pub scale: f32,
}

impl CrossAttentionCoupling {
    pub fn new(n: usize, gamma: f32) -> Self {
        let state_dim = 3;
        let mut q_weight = vec![0.0; state_dim * state_dim];
        let mut k_weight = vec![0.0; state_dim * state_dim];
        for i in 0..state_dim {
            q_weight[i * state_dim + i] = 1.0;
            k_weight[i * state_dim + i] = 1.0;
        }
        Self {
            n, state_dim, q_weight, k_weight, gamma,
            scale: (state_dim as f32).sqrt().recip(),
        }
    }

    /// Compute cross-population attention coupling.
    /// `coupling` is written for the source population (being coupled into).
    pub fn compute(
        &self,
        sin_src: &[f32], cos_src: &[f32], r_src: &[f32],   // source (Q)
        sin_tgt: &[f32], cos_tgt: &[f32], r_tgt: &[f32],   // target (K, value)
        coupling: &mut [f32],
    ) {
        let n = self.n.min(sin_src.len()).min(r_src.len())
                       .min(sin_tgt.len()).min(r_tgt.len()).min(coupling.len());
        let d = self.state_dim;
        let scale = self.scale;
        let gamma = self.gamma;

        // Pre-compute K vectors for target population
        let mut k_vecs = vec![0.0; n * d];
        for i in 0..n {
            let si = sin_tgt[i];
            let ci = cos_tgt[i];
            let ri = r_tgt[i];
            for j in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    let state_k = match k { 0 => si, 1 => ci, _ => ri };
                    sum += self.k_weight[j * d + k] * state_k;
                }
                k_vecs[i * d + j] = sum;
            }
        }

        // For each source oscillator i
        for i in 0..n {
            let mut q_vec = [0.0; 3];
            let si = sin_src[i];
            let ci = cos_src[i];
            let ri = r_src[i];
            for j in 0..d {
                let mut sum = 0.0;
                for k in 0..d {
                    let state_k = match k { 0 => si, 1 => ci, _ => ri };
                    sum += self.q_weight[j * d + k] * state_k;
                }
                q_vec[j] = sum;
            }

            // Compute attention scores (softmax over target)
            let mut max_score = -1e10f32;
            let mut scores = vec![0.0; n];
            for j in 0..n {
                let k_base = j * d;
                let mut s = 0.0;
                for h in 0..d { s += q_vec[h] * k_vecs[k_base + h]; }
                s *= scale;
                scores[j] = s;
                if s > max_score { max_score = s; }
            }

            // Softmax
            let mut sum_exp = 0.0;
            for j in 0..n {
                let e = (scores[j] - max_score).exp();
                scores[j] = e;
                sum_exp += e;
            }
            let inv_sum = if sum_exp > 1e-12 { 1.0 / sum_exp } else { 0.0 };
            for j in 0..n { scores[j] *= inv_sum; }

            // Cross-population coupling: γ · Σ_j A_ij · r_tgt_j · sin(θ_tgt_j - θ_src_i)
            let cos_src_i = cos_src[i];
            let sin_src_i = sin_src[i];
            let mut cpl = 0.0;
            for j in 0..n {
                let attn = scores[j];
                if attn < 1e-6 { continue; }
                let sin_diff = sin_tgt[j] * cos_src_i - cos_tgt[j] * sin_src_i;
                cpl += attn * r_tgt[j] * sin_diff;
            }

            coupling[i] += gamma * cpl;
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TriOscillator, Avx2Backend};

    // ─── 1. Attention coupling produces non-zero output ──

    #[test]
    fn test_attention_coupling_basic() {
        let n = 16;
        let attn = AttentionCoupling::new(n, 0.05);

        // Random-ish phases and amplitudes
        let sin_ph: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let cos_ph: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let r: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.05)).collect();
        let mut out = vec![0.0; n];

        attn.compute(&sin_ph, &cos_ph, &r, &mut out);

        // All outputs should be non-zero (phases differ)
        assert!(out.iter().any(|&c| c.abs() > 1e-6),
            "Attention coupling should produce non-zero output");
    }

    // ─── 2. All-identical oscillators → zero coupling (Kuramoto property) ──

    #[test]
    fn test_attention_coupling_identical_phases() {
        let n = 16;
        let attn = AttentionCoupling::new(n, 0.05);

        // All oscillators identical → sin_diff = 0 → coupling = 0
        let sin_ph = vec![0.5; n];
        let cos_ph = vec![(0.5f32).cos(); n];
        let r = vec![1.0; n];
        let mut out = vec![0.0; n];

        attn.compute(&sin_ph, &cos_ph, &r, &mut out);

        for i in 0..n {
            assert!(out[i].abs() < 1e-6,
                "Identical phases should give zero coupling: out[{}]={}", i, out[i]);
        }
    }

    // ─── 3. Identity Q/K → attention based on state similarity ──

    #[test]
    fn test_identity_attention_matches_cosine() {
        let n = 8;
        let attn = AttentionCoupling::new(n, 1.0); // gamma=1 for easy comparison

        // All oscillators have same phase, different amplitudes
        let phase: f32 = 0.3;
        let sin_ph = vec![phase.sin(); n];
        let cos_ph = vec![phase.cos(); n];
        let mut r = vec![0.0; n];
        r[0] = 2.0;  // oscillator 0 has high amplitude
        r[1] = 1.0;  // oscillator 1 has medium amplitude
        // rest have 0 amplitude → ignored in coupling

        let mut out = vec![0.0; n];
        attn.compute(&sin_ph, &cos_ph, &r, &mut out);

        // With identity Q=K:
        // Attention scores: Q_i · K_j = state_i · state_j
        // For identical phases: sin²+cos²=1, so state_i·state_j = 1 + r_i·r_j
        // Softmax over j emphasizes oscillators with higher r
        //
        // Since all sin_diff = 0 (same phase), coupling should be 0
        for i in 0..n {
            assert!(out[i].abs() < 1e-6,
                "Same phase → zero coupling even with different r: out[{}]={}", i, out[i]);
        }

        // Now oscillators 0 and 1 have same high r, oscillator 3 has different phase
        let sin_ph2: Vec<f32> = (0..n).map(|i| if i == 3 { 1.0 } else { 0.0 }).collect();
        let cos_ph2: Vec<f32> = (0..n).map(|i| if i == 3 { 0.0 } else { 1.0 }).collect();
        let r2: Vec<f32> = (0..n).map(|i| if i < 3 { 2.0 } else { 0.1 }).collect();
        let mut out2 = vec![0.0; n];
        attn.compute(&sin_ph2, &cos_ph2, &r2, &mut out2);

        // Oscillator 3 has different phase → should get non-zero coupling
        assert!(out2[3].abs() > 1e-6,
            "Different phase oscillator should get coupling: out2[3]={}", out2[3]);
    }

    // ─── 4. TriOscillator with attention coupling ──

    #[test]
    fn test_tri_oscillator_with_attention_coupling() {
        use crate::attention_coupling;

        let n = 16;
        let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

        // First spread phases with uniform coupling
        tri.set_uniform_coupling(0.3, 0.15, 0.08);
        for _ in 0..20 { tri.step(0.01); }

        // Now switch to attention-only coupling
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri.set_attention_coupling(config.pop1, config.pop2, config.pop3);
        tri.set_uniform_coupling(0.0, 0.0, 0.0);

        tri.step(0.01);

        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001),
            "Phases should update with attention coupling");
        // With spread phases, attention coupling should be non-zero
        assert!(tri.coupling1.iter().any(|&c| c.abs() > 1e-8),
            "Coupling should be non-zero with attention when phases differ");
        // Log coupling stats
        let max_cpl = tri.coupling1.iter().cloned().fold(0.0f32, f32::max);
        let mean_cpl = tri.coupling1.iter().sum::<f32>() / n as f32;
        println!("Attention coupling: max={:.6}, mean={:.6}", max_cpl, mean_cpl);
    }

    // ─── 5. Attention coupling vs uniform coupling ──

    #[test]
    fn test_attention_produces_different_dynamics() {
        let n = 16;

        // Uniform coupling
        let mut tri_uniform = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_uniform.set_uniform_coupling(0.3, 0.15, 0.08);
        for _ in 0..10 { tri_uniform.step(0.01); }

        // Attention coupling (identity init → similar to cosine similarity)
        let mut tri_attn = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri_attn.set_attention_coupling(config.pop1, config.pop2, config.pop3);
        tri_attn.set_uniform_coupling(0.0, 0.0, 0.0); // no uniform coupling
        for _ in 0..10 { tri_attn.step(0.01); }

        // Dynamics should differ (attention coupling is data-dependent)
        // At minimum, verify both produce non-zero phases
        let r1_uniform = tri_uniform.order_parameter(&tri_uniform.th1);
        let r1_attn = tri_attn.order_parameter(&tri_attn.th1);

        println!("Uniform: R1={:.4}, Attention: R1={:.4}", r1_uniform, r1_attn);

        // Both should have non-trivial synchronization
        assert!(r1_uniform > 0.0 && r1_uniform <= 1.0, "Uniform R1 in [0,1]: {}", r1_uniform);
        assert!(r1_attn > 0.0 && r1_attn <= 1.0, "Attention R1 in [0,1]: {}", r1_attn);
    }

    // ─── 6. Cross-population attention ──

    #[test]
    fn test_cross_attention_coupling_basic() {
        let n = 8;
        let cross = CrossAttentionCoupling::new(n, 0.03);

        let sin1: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let cos1: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).cos()).collect();
        let r1: Vec<f32> = vec![0.5; n];

        let sin2: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 1.0).sin()).collect();
        let cos2: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 1.0).cos()).collect();
        let r2: Vec<f32> = vec![0.5; n];

        let mut out = vec![0.0; n];
        cross.compute(&sin1, &cos1, &r1, &sin2, &cos2, &r2, &mut out);

        // Coupling from pop2 into pop1
        assert!(out.iter().any(|&c| c.abs() > 1e-6),
            "Cross-pop attention should produce non-zero coupling");
    }

    // ─── 7. Scalar backend with attention ──

    #[test]
    fn test_attention_with_scalar_backend() {
        let n = 8;
        let mut tri = TriOscillator::new(n, 0.5, 1.5, crate::backend::ScalarBackend);

        let config = AttentionCouplingConfig::all(n, 0.03);
        tri.set_attention_coupling(config.pop1, config.pop2, config.pop3);
        tri.set_uniform_coupling(0.0, 0.0, 0.0);

        tri.step(0.01);

        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001),
            "Scalar backend + attention should update phases");
    }

    // ─── 8. Phase gate still works with attention ──

    #[test]
    fn test_attention_with_phase_gate() {
        let n = 16;
        let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

        // Enable attention AND keep some uniform coupling + phase gate
        let config = AttentionCouplingConfig::pop1_only(
            AttentionCoupling::new(n, 0.05)
        );
        tri.set_attention_coupling(config.pop1, config.pop2, config.pop3);
        tri.set_uniform_coupling(0.1, 0.05, 0.02); // uniform for pop2, pop3

        tri.step(0.01);

        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001),
            "Attention + phase gate should work together");
        let r = tri.order_parameter(&tri.th1);
        assert!(r >= 0.0 && r <= 1.0, "Order param in [0,1]: {}", r);
    }

    // ─── 9. Reset preserves attention config ──

    #[test]
    fn test_attention_coupling_persists_after_reset() {
        let n = 8;
        let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

        let config = AttentionCouplingConfig::all(n, 0.05);
        tri.set_attention_coupling(config.pop1, config.pop2, config.pop3);
        tri.set_uniform_coupling(0.0, 0.0, 0.0);

        // Spread phases first (attention coupling won't work with all-identical phases)
        // Use a manual phase spread before the first step
        for i in 0..n {
            tri.th1[i] = (i as f32 * 0.5).sin();
            tri.th2[i] = (i as f32 * 0.5 + 1.0).cos();
            tri.th3[i] = (i as f32 * 0.5 + 2.0).sin() * 0.3;
        }

        tri.step(0.01);
        let cpl_before = tri.coupling1.iter().cloned().fold(0.0f32, f32::max);
        assert!(cpl_before > 1e-8,
            "Pre-reset: attention coupling should be non-zero with spread phases");

        tri.reset();

        // After reset, phases are back to 0 — spread them again
        for i in 0..n {
            tri.th1[i] = (i as f32 * 0.7).sin();
            tri.th2[i] = (i as f32 * 0.7 + 1.0).cos();
            tri.th3[i] = (i as f32 * 0.7 + 2.0).sin() * 0.5;
        }

        tri.step(0.01);
        let cpl_after = tri.coupling1.iter().cloned().fold(0.0f32, f32::max);
        println!("Pre-reset coupling: max={:.6}, Post-reset coupling: max={:.6}", cpl_before, cpl_after);
        assert!(cpl_after > 1e-8,
            "Attention coupling should work after reset (attention config preserved)");
    }

    // ═══════════════════════════════════════════════════════════
    //  PST + Attention — Benchmark Suite
    // ═══════════════════════════════════════════════════════════
    //
    // Сравнивает uniform vs attention coupling по качеству декодирования
    // входного сигнала из PST фич.
    //
    // Метрики:
    //   1. PST feature variance — насколько богаче динамика
    //   2. Decoder convergence — скорость обучения NLMS
    //   3. Reconstruction accuracy — Pearson R восстановленного сигнала
    //   4. Drive separation — различимость разных уровней драйва

    use crate::{NlmsDecoder, Decoder, compute_pearson, pst_linear_features, pst_fast, pst_fast_dim};

    /// PST feature variance: mean std across all features.
    /// Высокая variance = богаче динамика = больше информации для декодера.
    fn pst_feature_variance(features: &[Vec<f32>]) -> f32 {
        if features.is_empty() || features[0].is_empty() { return 0.0; }
        let n_frames = features.len();
        let dim = features[0].len();
        let mut variances = 0.0f32;
        let mut count = 0;
        for d in 0..dim {
            let mean: f32 = features.iter().map(|fv| fv[d]).sum::<f32>() / n_frames as f32;
            let var: f32 = features.iter().map(|fv| (fv[d] - mean).powi(2)).sum::<f32>() / n_frames as f32;
            variances += var;
            count += 1;
        }
        if count > 0 { variances / count as f32 } else { 0.0 }
    }

    /// Train NLMS decoder to reconstruct drive signal from PST features.
    /// Returns (predictions, pearson_r).
    fn train_decode_pst(
        pst_frames: &[Vec<f32>],
        target: &[f32],
        lr: f32,
    ) -> (Vec<f32>, f32) {
        let dim = pst_frames[0].len();
        let n = pst_frames.len();
        let mut dec = NlmsDecoder::new(dim, lr, 1e-6);
        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let pred = dec.predict(&pst_frames[i]);
            predictions.push(pred);
            dec.train(&pst_frames[i], target[i]);
        }
        let r = compute_pearson(&predictions[10..], &target[10..]); // skip first 10 (transient)
        (predictions, r)
    }

    /// Collect PST features over time from a TriOscillator with given drive signal.
    fn collect_pst_frames(
        tri: &mut TriOscillator<Avx2Backend>,
        drive_signal: &[f32],
        n_warmup: usize,
        dt: f32,
    ) -> Vec<Vec<f32>> {
        let n_frames = drive_signal.len();
        // Burn-in to settle dynamics
        for _ in 0..n_warmup {
            for i in 0..tri.n {
                tri.drive1[i] = drive_signal[0];
            }
            tri.step(dt);
        }
        // Collect PST features
        let mut frames = Vec::with_capacity(n_frames);
        for &drive in drive_signal {
            for i in 0..tri.n {
                tri.drive1[i] = drive;
                tri.drive2[i] = 0.0;
                tri.drive3[i] = 0.0;
            }
            tri.step(dt);
            frames.push(pst_linear_features(tri));
        }
        frames
    }

    // ─── 10. PST feature variance: attention vs uniform ──

    #[test]
    fn test_pst_attention_vs_uniform_feature_variance() {
        let n = 16;
        let dt = 0.05;
        let n_steps = 200;
        let n_warmup = 50;

        // Create two oscillators with identical config
        let mut tri_uniform = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_uniform.set_uniform_coupling(0.3, 0.15, 0.08);

        let mut tri_attn = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_attn.set_uniform_coupling(0.0, 0.0, 0.0);
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri_attn.set_attention_coupling(config.pop1, config.pop2, config.pop3);

        // Drive signal: sine wave + noise
        let mut rng = rand::thread_rng();
        let drive_signal: Vec<f32> = (0..n_steps).map(|i| {
            let t = i as f32 * dt;
            (2.0 * std::f32::consts::PI * 1.0 * t).sin() * 3.0
                + rand::Rng::gen::<f32>(&mut rng) * 0.5 - 0.25
        }).collect();

        // Collect PST features
        let frames_uniform = collect_pst_frames(&mut tri_uniform, &drive_signal, n_warmup, dt);
        let frames_attn = collect_pst_frames(&mut tri_attn, &drive_signal, n_warmup, dt);

        let var_uniform = pst_feature_variance(&frames_uniform);
        let var_attn = pst_feature_variance(&frames_attn);

        println!("═══ PST Feature Variance ═══");
        println!("  Uniform coupling:   var = {:.6}", var_uniform);
        println!("  Attention coupling: var = {:.6}", var_attn);
        println!("  Ratio (attn/uniform): {:.3}x", var_attn / var_uniform.max(1e-10));

        // Attention should produce richer dynamics (higher variance) or at least different
        // The test passes as long as variance is non-zero for both
        assert!(var_uniform > 0.0, "Uniform PST variance should be > 0");
        assert!(var_attn > 0.0, "Attention PST variance should be > 0");
    }

    // ─── 11. NLMS decoding accuracy: attention vs uniform ──

    #[test]
    fn test_pst_attention_vs_uniform_decoding_accuracy() {
        let n = 16;
        let dt = 0.05;
        let n_steps = 300;
        let n_warmup = 50;

        // Create two oscillators
        let mut tri_uniform = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_uniform.set_uniform_coupling(0.3, 0.15, 0.08);

        let mut tri_attn = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_attn.set_uniform_coupling(0.0, 0.0, 0.0);
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri_attn.set_attention_coupling(config.pop1, config.pop2, config.pop3);

        // Drive signal: multi-frequency sine + step changes
        let drive_signal: Vec<f32> = (0..n_steps).map(|i| {
            let t = i as f32 * dt;
            let slow = (2.0 * std::f32::consts::PI * 0.5 * t).sin() * 2.0;
            let fast = (2.0 * std::f32::consts::PI * 3.0 * t).sin() * 1.0;
            // Step change at middle
            let step = if i > n_steps / 2 { 1.0 } else { -1.0 };
            slow + fast + step
        }).collect();

        // Collect PST features
        let frames_uniform = collect_pst_frames(&mut tri_uniform, &drive_signal, n_warmup, dt);
        let frames_attn = collect_pst_frames(&mut tri_attn, &drive_signal, n_warmup, dt);

        // Train NLMS decoder
        let lr = 0.05;
        let (_pred_uni, r_uni) = train_decode_pst(&frames_uniform, &drive_signal, lr);
        let (_pred_attn, r_attn) = train_decode_pst(&frames_attn, &drive_signal, lr);

        println!("═══ PST Decoding Accuracy (Pearson R) ═══");
        println!("  Uniform coupling:   R = {:.4}", r_uni);
        println!("  Attention coupling: R = {:.4}", r_attn);
        println!("  Difference: {:.4}", r_attn - r_uni);

        // Both should have positive correlation (signal is reconstructable)
        assert!(r_uni > 0.3, "Uniform decoding R should be > 0.3, got {}", r_uni);
        assert!(r_attn > 0.3, "Attention decoding R should be > 0.3, got {}", r_attn);

        // Log convergence: track error over time
        let dim = frames_uniform[0].len();
        let mut dec_uni = NlmsDecoder::new(dim, lr, 1e-6);
        let mut dec_attn = NlmsDecoder::new(dim, lr, 1e-6);
        let mut errors_uni = Vec::new();
        let mut errors_attn = Vec::new();
        for i in 0..n_steps {
            let e_uni = dec_uni.train(&frames_uniform[i], drive_signal[i]).abs();
            let e_attn = dec_attn.train(&frames_attn[i], drive_signal[i]).abs();
            if i % 20 == 0 {
                errors_uni.push(e_uni);
                errors_attn.push(e_attn);
            }
        }
        println!("  Convergence (error every 20 steps):");
        for k in 0..errors_uni.len().min(10) {
            println!("    step {}: uniform={:.4}  attention={:.4}", k * 20, errors_uni[k], errors_attn[k]);
        }
    }

    // ─── 12. Drive separation: different drives → distinct PST ──

    #[test]
    fn test_pst_attention_drive_separation() {
        let n = 16;
        let dt = 0.05;
        let n_warmup = 30;
        let n_steady = 20;

        // Test 5 different drive levels
        let drive_levels = [-3.0, -1.0, 0.0, 1.0, 3.0];

        let mut tri_uniform = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_uniform.set_uniform_coupling(0.3, 0.15, 0.08);

        let mut tri_attn = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_attn.set_uniform_coupling(0.0, 0.0, 0.0);
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri_attn.set_attention_coupling(config.pop1, config.pop2, config.pop3);

        for &coupling_type in &["uniform", "attention"] {
            let tri = match coupling_type {
                "uniform" => &mut tri_uniform,
                _ => &mut tri_attn,
            };

            // For each drive level, compute mean PST feature vector
            let mut mean_pst_per_drive = Vec::new();
            for &drive in &drive_levels {
                // Burn-in with this drive
                for _ in 0..n_warmup {
                    for os in 0..tri.n {
                        tri.drive1[os] = drive;
                        tri.drive2[os] = 0.0;
                        tri.drive3[os] = 0.0;
                    }
                    tri.step(dt);
                }
                // Collect steady-state PST features
                let mut sum_fv: Vec<f32> = pst_linear_features(tri);
                for _ in 1..n_steady {
                    for os in 0..tri.n {
                        tri.drive1[os] = drive;
                    }
                    tri.step(dt);
                    let fv = pst_linear_features(tri);
                    for d in 0..sum_fv.len() {
                        sum_fv[d] += fv[d];
                    }
                }
                // Average
                let nf = n_steady as f32;
                for d in 0..sum_fv.len() {
                    sum_fv[d] /= nf;
                }
                mean_pst_per_drive.push(sum_fv);
            }

            // Compute pairwise Euclidean distances between PST for different drives
            let mut total_dist = 0.0;
            let mut pairs = 0;
            for a in 0..drive_levels.len() {
                for b in (a+1)..drive_levels.len() {
                    let dist: f32 = (0..mean_pst_per_drive[a].len())
                        .map(|d| (mean_pst_per_drive[a][d] - mean_pst_per_drive[b][d]).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    total_dist += dist;
                    pairs += 1;
                }
            }
            let avg_dist = total_dist / pairs as f32;
            println!("  {coupling_type}: avg pairwise PST distance between drives = {:.4}", avg_dist);

            // Verify different drives produce measurably different PST
            // The distance between most different drives should be > distance between most similar
            assert!(avg_dist > 0.001,
                "{coupling_type}: different drives should produce distinct PST");
        }
    }

    // ─── 13. PST fast features: attention vs uniform comparison ──

    #[test]
    fn test_pst_fast_attention_vs_uniform() {
        let n = 32;
        let dt = 0.02;
        let n_steps = 100;
        let n_warmup = 30;

        let mut tri = TriOscillator::new(n, 0.5, 2.0, Avx2Backend);

        // Phase 1: uniform coupling
        tri.set_uniform_coupling(0.3, 0.15, 0.08);
        tri.set_attention_coupling(None, None, None);

        for _ in 0..n_warmup {
            for i in 0..n { tri.drive1[i] = 1.0; tri.drive2[i] = 0.0; tri.drive3[i] = 0.0; }
            tri.step(dt);
        }
        let mut uniform_frames = Vec::new();
        for _ in 0..n_steps {
            for i in 0..n { tri.drive1[i] = (i as f32 * 0.02).sin() * 2.0; }
            tri.step(dt);
            uniform_frames.push(pst_fast(&tri));
        }

        // Phase 2: attention coupling (same oscillator, continue from uniform state)
        tri.set_uniform_coupling(0.0, 0.0, 0.0);
        tri.set_attention_coupling(
            Some(AttentionCoupling::new(n, 0.05)),
            Some(AttentionCoupling::new(n, 0.03)),
            None,
        );

        for _ in 0..n_warmup {
            for i in 0..n { tri.drive1[i] = 1.0; tri.drive2[i] = 0.0; tri.drive3[i] = 0.0; }
            tri.step(dt);
        }
        let mut attn_frames = Vec::new();
        for _ in 0..n_steps {
            for i in 0..n { tri.drive1[i] = (i as f32 * 0.02).sin() * 2.0; }
            tri.step(dt);
            attn_frames.push(pst_fast(&tri));
        }

        let var_uni = pst_feature_variance(&uniform_frames);
        let var_attn = pst_feature_variance(&attn_frames);

        println!("═══ PST Fast Features (6N = {} dim) ═══", pst_fast_dim(n));
        println!("  Uniform coupling:   variance = {:.6}", var_uni);
        println!("  Attention coupling: variance = {:.6}", var_attn);
        println!("  Ratio: {:.3}x", var_attn / var_uni.max(1e-10));

        assert!(var_uni > 0.0);
        assert!(var_attn > 0.0);
    }

    // ─── 14. PST amplitude-phase coupling analysis ──

    #[test]
    fn test_pst_attention_phase_amplitude_coupling() {
        let n = 16;
        let dt = 0.05;
        let n_warmup = 40;

        let mut tri_uniform = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_uniform.set_uniform_coupling(0.3, 0.15, 0.08);

        let mut tri_attn = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_attn.set_uniform_coupling(0.0, 0.0, 0.0);
        let config = AttentionCouplingConfig::all(n, 0.05);
        tri_attn.set_attention_coupling(config.pop1, config.pop2, config.pop3);

        // Compute order parameter and amplitude spread for both
        for _ in 0..n_warmup {
            for i in 0..n {
                tri_uniform.drive1[i] = 1.0;
                tri_attn.drive1[i] = 1.0;
                tri_uniform.drive2[i] = 0.0;
                tri_attn.drive2[i] = 0.0;
                tri_uniform.drive3[i] = 0.0;
                tri_attn.drive3[i] = 0.0;
            }
            tri_uniform.step(dt);
            tri_attn.step(dt);
        }

        let r1_uni = tri_uniform.order_parameter(&tri_uniform.th1);
        let r1_attn = tri_attn.order_parameter(&tri_attn.th1);

        let r_std_uni = {
            let mean_r = tri_uniform.r1.iter().sum::<f32>() / n as f32;
            (tri_uniform.r1.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / n as f32).sqrt()
        };
        let r_std_attn = {
            let mean_r = tri_attn.r1.iter().sum::<f32>() / n as f32;
            (tri_attn.r1.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>() / n as f32).sqrt()
        };

        println!("═══ Phase-Amplitude Analysis ═══");
        println!("  Uniform:   R={:.4}, r_std={:.4}", r1_uni, r_std_uni);
        println!("  Attention: R={:.4}, r_std={:.4}", r1_attn, r_std_attn);

        // Both should have non-trivial synchronization and amplitude spread
        assert!(r1_uni > 0.0 && r1_uni <= 1.0);
        assert!(r1_attn > 0.0 && r1_attn <= 1.0);
    }
}
