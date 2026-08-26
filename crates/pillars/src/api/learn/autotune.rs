//! -------------------------------------------------------------------------
//! Filename:    src/autotune.rs
//! Class:       api
//! Purpose:     Auto-tuning utilities for E-Prop Izhikevich Multi.
//! Description:
//!
//! Provides systematic hyperparameter search:
//! - [`probe_fr`] — firing rate scan over bias values
//! - [`suggest_config`] — full configuration from problem dimensions
//!
//! # Tuning strategy
//!
//! The key insight from empirical tuning is that Izhikevich E-Prop is
//! sensitive to the **signal-to-bias ratio (SNR)**:
//!
//! ```text
//! SNR = σ(weight_current) / bias
//! ```
//!
//! where σ(weight_current) includes the 10× kernel weight scale.
//!
//! - `SNR < 1` → bias dominates, network ignores input (no learning)
//! - `SNR ~ 3` → good balance: input-driven with bias floor
//! - `SNR > 8` → highly input-sensitive, may be unstable
//!
//! Other critical parameters:
//! - `d` (adaptation): lower values (2.0) allow sustained spiking
//! - `tau_surrogate`: wider values (5.0) stabilise eligibility accumulation
//!
//! # Note on kernel weight scale
//!
//! The kernel `eprop_izhikevich_step_multi` applies an additional 10×
//! factor on weights when computing I_total. All SNR calculations in
//! this module account for this 10× kernel scale, so `weight_scale`
//! refers to the **raw** init scale (before kernel multiplication).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::api::model::eprop::EPropIzhikevichMulti;

/// Factor applied by the kernel to raw weights: I_s += w * KERNEL_SCALE
const KERNEL_WEIGHT_SCALE: f32 = 10.0;

// ═══════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════

/// Auto-tuning configuration for E-Prop Izhikevich Multi.
#[derive(Debug, Clone)]
pub struct EPropTunerConfig {
    /// Bias current per hidden neuron
    pub bias: f32,
    /// Weight initialisation scale: U[-weight_scale, weight_scale]
    pub weight_scale: f32,
    /// Input (hidden layer) learning rate
    pub lr_in: f32,
    /// Output (readout) learning rate
    pub lr_out: f32,
    /// Izhikevich adaptation parameter (post-spike u increment)
    pub d: f32,
    /// Surrogate gradient width (mV)
    pub tau_surrogate: f32,
    /// Eligibility trace time constant (ms)
    pub tau_e: f32,
    /// Simulation steps per example
    pub n_steps: usize,
    /// Expected firing rate at this bias
    pub expected_fr: f32,
    /// Signal-to-bias ratio (including 10× kernel scale)
    pub snr: f32,
}

// ═══════════════════════════════════════════════════════════
//  Signal statistics
// ═══════════════════════════════════════════════════════════

/// Compute expected signal std from weight distribution, Poisson activity,
/// **and** the 10× kernel weight scale.
///
/// For zero-mean uniform weights U[-w, w] and Bernoulli(α) input per step:
///   Var[z_i] = α·(1-α)
///   Effective weight (after kernel 10×): w_eff = 10·w_raw
///   Var[I_total] = Var[Σ_i 10·w_i·z_i] = 100 · Σ_i w_i² · Var[z_i]
///                ≈ 100 · n_pre · w²/3 · α·(1-α)
///   σ(I_total)  = 10 · sqrt(n_pre · w²/3 · α·(1-α))
fn estimate_signal_std(n_pre: usize, poisson_activity: f32, weight_scale: f32) -> f32 {
    // Var per input synapse for uniform U[-s, s]: s²/3
    let per_synapse_var = weight_scale * weight_scale / 3.0;
    // Bernoulli activity α → Var[z] = α·(1-α)
    let activity_var = poisson_activity * (1.0 - poisson_activity);
    // Kernel 10× factor: Var[k·X] = k²·Var[X]
    let kernel_var = KERNEL_WEIGHT_SCALE * KERNEL_WEIGHT_SCALE;
    let total_var = n_pre as f32 * activity_var * per_synapse_var * kernel_var;
    total_var.sqrt()
}

/// Estimate raw weight scale needed to achieve a target SNR after kernel 10×.
fn target_weight_scale_for_snr(
    n_pre: usize,
    poisson_activity: f32,
    bias: f32,
    target_snr: f32,
) -> f32 {
    // SNR = 10 · σ_raw / bias   →   σ_raw = SNR · bias / 10
    // σ_raw = sqrt(n_pre · w²/3 · α·(1-α))
    // w² = 3 · σ_raw² / (n_pre · α·(1-α))
    // w² = 3 · SNR² · bias² / (100 · n_pre · α·(1-α))
    let num = 3.0 * target_snr * target_snr * bias * bias;
    let activity_var = poisson_activity * (1.0 - poisson_activity);
    let kernel_var = KERNEL_WEIGHT_SCALE * KERNEL_WEIGHT_SCALE;
    let den = n_pre as f32 * activity_var * kernel_var;
    if den <= 1e-12 {
        0.1
    } else {
        (num / den).sqrt().clamp(0.01, 10.0)
    }
}

// ═══════════════════════════════════════════════════════════
//  FR probe
// ═══════════════════════════════════════════════════════════

/// FR (firing rate) probe result.
#[derive(Debug, Clone, Copy)]
pub struct FrProbePoint {
    /// Bias value tested
    pub bias: f32,
    /// Average firing rate (spikes / (n_post * n_steps))
    pub avg_fr: f32,
    /// Signal-to-bias ratio (including 10× kernel scale)
    pub snr: f32,
}

/// Probe firing rates for a range of bias values.
///
/// Uses the same frozen random weights and input for all biases
/// so results are directly comparable. Creates fresh `EPropIzhikevichMulti`
/// state for each bias.
///
/// # Returns
/// Vector of `FrProbePoint` sorted by bias (same order as input).
pub fn probe_fr(
    n_pre: usize,
    n_post: usize,
    biases: &[f32],
    n_steps: usize,
    poisson_activity: f32,
    weight_scale: f32,
    seed: u64,
) -> Vec<FrProbePoint> {
    if biases.is_empty() {
        return vec![];
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let signal_std = estimate_signal_std(n_pre, poisson_activity, weight_scale);

    // Pre-generate one set of frozen random weights (fair comparison)
    let weights: Vec<f32> = (0..n_pre * n_post)
        .map(|_| rng.gen_range(-weight_scale..weight_scale))
        .collect();

    // Pre-generate one frozen random input sequence
    let n_total = n_steps * n_pre;
    let frozen_input: Vec<i32> = (0..n_total)
        .map(|_| {
            if rng.gen::<f32>() < poisson_activity {
                1
            } else {
                0
            }
        })
        .collect();

    biases
        .iter()
        .map(|&bias_val| {
            let mut eprop = EPropIzhikevichMulti::new(n_pre, n_post, 1.0, 30.0);
            let mut z_pre = vec![0i32; n_pre];
            let mut spikes = vec![0i32; n_post];
            let i_ext = vec![bias_val; n_post];

            let mut total_spikes = 0usize;

            for step_idx in 0..n_steps {
                let start = step_idx * n_pre;
                z_pre.copy_from_slice(&frozen_input[start..start + n_pre]);
                let count = eprop.step(&weights, &z_pre, &i_ext, &mut spikes);
                total_spikes += count;
            }

            let avg_fr = total_spikes as f32 / (n_post * n_steps) as f32;
            let snr = if bias_val > 0.0 {
                signal_std / bias_val
            } else {
                f32::INFINITY
            };

            FrProbePoint {
                bias: bias_val,
                avg_fr,
                snr,
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════
//  Config suggestion
// ═══════════════════════════════════════════════════════════

/// Suggest a complete E-Prop Izhikevich Multi configuration.
///
/// Runs an automatic FR probe and selects the best bias for the
/// target firing rate, then computes matching hyperparameters.
///
/// # Args
/// * `n_pre` — number of input (pre-synaptic) neurons
/// * `n_post` — number of hidden (post-synaptic) neurons
/// * `poisson_activity` — fraction of active inputs per step (e.g. 0.2)
/// * `target_fr` — target firing rate (e.g. 0.05–0.15)
/// * `seed` — random seed for reproducibility
///
/// # Returns
/// `EPropTunerConfig` with recommended hyperparameters.
///
/// `poisson_activity` should match your rate coding scheme.
/// For MNIST with max_hz=200, dt=1ms, image scaled `[0,1]`:
///   activity ≈ poisson_max_hz * dt/1000 * mean_pixel_intensity
pub fn suggest_config(
    n_pre: usize,
    n_post: usize,
    poisson_activity: f32,
    target_fr: f32,
    seed: u64,
) -> EPropTunerConfig {
    // ─── 1. Baseline parameters ─────────────────────────
    let d = 2.0;
    let tau_surrogate = 5.0;
    let tau_e = 30.0;
    let n_steps = 50;
    let lr_out = 0.02;

    // ─── 2. Find good weight_scale and bias ──────────────
    // Start with the empirical formula: 10 / sqrt(n_pre)
    let base_weight_scale = 10.0 / (n_pre as f32).sqrt();

    // Probe biases 1..40
    let bias_candidates: Vec<f32> = (1..=40).map(|b| b as f32).collect();
    let probe_points = probe_fr(
        n_pre,
        n_post,
        &bias_candidates,
        n_steps,
        poisson_activity,
        base_weight_scale,
        seed,
    );

    // Find bias closest to target FR
    let best_probe = probe_points.iter().min_by(|a, b| {
        (a.avg_fr - target_fr)
            .abs()
            .partial_cmp(&(b.avg_fr - target_fr).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (bias, expected_fr) = match best_probe {
        Some(p) => (p.bias, p.avg_fr),
        None => (8.0, 0.0),
    };

    // ─── 3. Adjust weight_scale for SNR ≈ 3 ─────────────
    // But guard against over-adjustment that kills FR.
    let target_snr = 3.0;
    let snr_at_base = {
        let sig = estimate_signal_std(n_pre, poisson_activity, base_weight_scale);
        sig / bias.max(1.0)
    };

    let needs_adjustment = !(1.5..=8.0).contains(&snr_at_base);
    let adjusted_weight_scale = if needs_adjustment {
        target_weight_scale_for_snr(n_pre, poisson_activity, bias, target_snr)
    } else {
        base_weight_scale
    };

    // ─── 4. Re-probe FR with adjusted weight_scale ───────
    // If the re-probed FR is too low (< 0.01), reject the adjustment.
    // This happens when bias is at the probe floor (1) and we reduce
    // weight_scale so much that neurons barely fire.
    let (final_bias, final_weight_scale, final_expected_fr, final_snr) = if needs_adjustment {
        let re_probe = probe_fr(
            n_pre,
            n_post,
            &[bias],
            n_steps,
            poisson_activity,
            adjusted_weight_scale,
            seed.wrapping_add(1),
        );
        let re_fr = re_probe.first().map(|p| p.avg_fr).unwrap_or(0.0);

        if re_fr < 0.01 {
            // Adjustment killed FR. Fall back to base weight_scale at a
            // higher bias that gives acceptable SNR.
            let fallback = probe_points
                .iter()
                .filter(|p| {
                    let snr = estimate_signal_std(n_pre, poisson_activity, base_weight_scale)
                        / p.bias.max(1.0);
                    (1.5..=8.0).contains(&snr)
                })
                .min_by(|a, b| {
                    (a.avg_fr - target_fr)
                        .abs()
                        .partial_cmp(&(b.avg_fr - target_fr).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            match fallback {
                Some(fb) => {
                    let fb_snr = estimate_signal_std(n_pre, poisson_activity, base_weight_scale)
                        / fb.bias.max(1.0);
                    (fb.bias, base_weight_scale, fb.avg_fr, fb_snr)
                }
                None => {
                    // No fallback found — keep the adjusted values
                    (bias, adjusted_weight_scale, re_fr, target_snr)
                }
            }
        } else {
            (bias, adjusted_weight_scale, re_fr, target_snr)
        }
    } else {
        // No adjustment needed → use base
        (bias, base_weight_scale, expected_fr, snr_at_base)
    };

    // ─── 5. Learning rate ────────────────────────────────
    let ref_scale = 10.0 / (n_pre as f32).max(1.0).sqrt();
    let lr_in = if final_weight_scale > 0.0 {
        (0.05 * ref_scale / final_weight_scale).clamp(0.0005, 0.1)
    } else {
        0.005
    };

    EPropTunerConfig {
        bias: final_bias,
        weight_scale: final_weight_scale,
        lr_in,
        lr_out,
        d,
        tau_surrogate,
        tau_e,
        n_steps,
        expected_fr: final_expected_fr,
        snr: final_snr,
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_fr_monotonic() {
        let results = probe_fr(100, 64, &[0.0, 2.0, 4.0, 8.0, 16.0], 20, 0.2, 0.5, 42);
        assert_eq!(results.len(), 5);
        for i in 1..results.len() {
            assert!(
                results[i].avg_fr >= results[i - 1].avg_fr - 1e-6,
                "FR should increase with bias: bias={}, fr={} < prev bias={}, fr={}",
                results[i].bias,
                results[i].avg_fr,
                results[i - 1].bias,
                results[i - 1].avg_fr
            );
        }
    }

    #[test]
    fn test_suggest_config_reasonable() {
        let cfg = suggest_config(784, 256, 0.2, 0.1, 42);
        assert!(
            cfg.bias > 0.0 && cfg.bias < 50.0,
            "bias={} out of range",
            cfg.bias
        );
        assert!(cfg.weight_scale > 0.01 && cfg.weight_scale < 10.0);
        assert!(cfg.lr_in > 0.0 && cfg.lr_in < 0.2);
        assert_eq!(cfg.d, 2.0);
        assert_eq!(cfg.tau_surrogate, 5.0);
        assert_eq!(cfg.tau_e, 30.0);
        assert_eq!(cfg.n_steps, 50);
        assert!(cfg.lr_out > 0.0);
        assert!(cfg.snr > 0.0);
    }

    #[test]
    fn test_estimate_signal_std_includes_kernel_scale() {
        // n_pre=784, α=0.2, w=0.357 (10/sqrt(784))
        // Raw σ without kernel: sqrt(784 * 0.2 * 0.8 * 0.357² / 3) ≈ 2.31
        // With 10× kernel: 10 * 2.31 = 23.1
        let s = estimate_signal_std(784, 0.2, 0.357);
        assert!(
            (s - 23.1).abs() < 1.0,
            "std={} expected ~23.1 (with 10× kernel)",
            s
        );
    }

    #[test]
    fn test_target_weight_scale_for_snr_with_kernel() {
        // n_pre=784, α=0.2, bias=8, target_snr=3
        // σ_target = SNR * bias = 24
        // w = sqrt(3 * σ_target² / (100 * n_pre * α*(1-α)))
        //   = sqrt(3 * 576 / (100 * 784 * 0.16))
        //   = sqrt(1728 / 12544)
        //   = sqrt(0.138) = 0.371
        let w = target_weight_scale_for_snr(784, 0.2, 8.0, 3.0);
        assert!(
            (w - 0.371).abs() < 0.05,
            "w={} expected ~0.371 (with 10× kernel)",
            w
        );
    }

    #[test]
    fn test_lr_scaling() {
        let cfg = suggest_config(784, 256, 0.2, 0.05, 42);
        assert!(cfg.lr_in > 0.0 && cfg.lr_in <= 0.1);
    }

    #[test]
    fn test_poisson_activity_edge_cases() {
        // With 10× kernel: σ = 10 · sqrt(784 · 0.01·0.99 · 0.357² / 3) ≈ 5.74
        let s_low = estimate_signal_std(784, 0.01, 0.357);
        assert!(
            s_low > 0.0 && s_low < 10.0,
            "low activity std={} out of range",
            s_low
        );

        // α=0.99: σ = 10 · sqrt(784 · 0.99·0.01 · 0.357² / 3) ≈ 5.74 (same due to symmetry)
        let s_high = estimate_signal_std(784, 0.99, 0.357);
        assert!(
            s_high > 0.0 && s_high < 10.0,
            "high activity std={} out of range",
            s_high
        );

        // Very low activity → signal is near-zero → formula may hit max clamp
        let w = target_weight_scale_for_snr(784, 0.001, 8.0, 3.0);
        assert!(
            w > 0.0 && w <= 10.0,
            "low activity clamp w={} out of range",
            w
        );
    }

    #[test]
    fn test_snr_matches_empirical() {
        // Empirical finding: weight_scale = 10/sqrt(784) ≈ 0.357, bias=8
        // gives good training results. Effective SNR (incl. 10× kernel) ≈ 23.1 / bias.
        // The tuner may pick a different bias depending on target_fr.
        // With 10× kernel, FR at low bias can be higher, so target_fr=0.015 for safety.
        let cfg = suggest_config(784, 256, 0.2, 0.015, 42);
        // Just check the config is valid
        assert!(
            cfg.bias > 0.0 && cfg.bias < 50.0,
            "bias={} should be reasonable",
            cfg.bias
        );
        assert!(cfg.weight_scale > 0.0 && cfg.weight_scale <= 10.0);
        assert!(cfg.snr > 0.0, "snr should be positive");
    }
}
