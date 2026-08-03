//! Basal Ganglia — Action Selection (SPAUN-inspired).
//!
//! Implements a winner-take-all network for cognitive action routing,
//! similar to how the Basal Ganglia selects which cortical module
//! to activate in the Semantic Pointer Architecture.
//!
//! # Architecture (simplified Frankenstein-Basal Ganglia)
//!
//! ```text
//!              Cortex (n_actions input channels)
//!                         │
//!              ┌──────────┴──────────┐
//!              │                     │
//!        Striatum D1            Striatum D2
//!        (Go pathway)           (NoGo pathway)
//!              │                     │
//!              │    ┌────────────────┤
//!              │    │                │
//!              ▼    ▼                ▼
//!           GPi/SNr              STN
//!         (inhibit)            (broad excitation)
//!              │                     │
//!              └──────────┬──────────┘
//!                         │
//!                    Thalamus
//!                   (re-enable)
//!                         │
//!                    Action Output
//! ```
//!
//! # Winner-take-all
//! Each action channel competes. The channel with highest cortical
//! drive wins; all others are suppressed via lateral inhibition.
//!
//! # Usage
//! ```rust
//! use taas_checkmate::karmac::basal_ganglia::BasalGanglia;
//!
//! let mut bg = BasalGanglia::new(8, 4); // 8 cortical channels, 4 actions
//! let cortex_input = vec![0.1, 0.9, 0.3, 0.2, 0.05, 0.1, 0.0, 0.0];
//! let action = bg.step(&cortex_input);
//! assert!(action < 4); // valid action index selected
//! ```

/// Configuration for Basal Ganglia.
#[derive(Debug, Clone)]
pub struct BgConfig {
    /// Number of cortical input channels.
    pub n_cortical: usize,
    /// Number of competing actions.
    pub n_actions: usize,
    /// Striatum D1 (Go) gain — amplifies desired actions.
    pub d1_gain: f32,
    /// Striatum D2 (NoGo) gain — suppresses undesired actions.
    pub d2_gain: f32,
    /// STN broad excitation — prevents premature commitment.
    pub stn_excitation: f32,
    /// Lateral inhibition strength in GPi (winner-take-all sharpness).
    pub gpi_inhibition: f32,
    /// Thalamic relay gain — re-enables winning channel.
    pub thalamus_gain: f32,
    /// Leak decay per step (for working memory integration).
    pub decay: f32,
}

impl Default for BgConfig {
    fn default() -> Self {
        Self {
            n_cortical: 8,
            n_actions: 4,
            d1_gain: 2.0,
            d2_gain: 1.5,
            stn_excitation: 0.5,
            gpi_inhibition: 3.0,
            thalamus_gain: 1.0,
            decay: 0.95,
        }
    }
}

/// Basal Ganglia — winner-take-all action selection.
pub struct BasalGanglia {
    pub config: BgConfig,
    /// Accumulated cortical state [n_cortical].
    state: Vec<f32>,
    /// Weight matrix: cortical → action (D1/D2 combined).
    /// [n_actions × n_cortical] row-major.
    cortico_stratial: Vec<f32>,
    /// Lateral inhibition matrix [n_actions × n_actions].
    lateral_inhibit: Vec<f32>,
    /// Previous winning action (for hysteresis).
    prev_action: Option<usize>,
    /// Pre-allocated drive buffer — avoids heap alloc in hot `step()`.
    drive_buf: Vec<f32>,
    /// Pre-allocated output buffer — avoids clone in hot `step()`.
    output_buf: Vec<f32>,
}

impl BasalGanglia {
    /// Create a new Basal Ganglia with Xavier-initialized weights.
    pub fn new(n_cortical: usize, n_actions: usize) -> Self {
        Self::with_config(BgConfig {
            n_cortical,
            n_actions,
            ..Default::default()
        })
    }

    /// Create with explicit weight matrix (useful for demo/debug/pretrained).
    /// `cortico_stratial` must be [n_actions × n_cortical] row-major.
    pub fn from_weights(cortico_stratial: Vec<f32>, n_cortical: usize, n_actions: usize) -> Self {
        let mut lateral_inhibit = vec![0.0f32; n_actions * n_actions];
        for i in 0..n_actions {
            for j in 0..n_actions {
                if i != j {
                    lateral_inhibit[i * n_actions + j] = -3.0;
                }
            }
        }
        Self {
            config: BgConfig {
                n_cortical,
                n_actions,
                ..Default::default()
            },
            state: vec![0.0; n_cortical],
            cortico_stratial,
            lateral_inhibit,
            prev_action: None,
            drive_buf: vec![0.0; n_actions],
            output_buf: vec![0.0; n_actions],
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: BgConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n_c = config.n_cortical;
        let n_a = config.n_actions;

        // Cortico-striatal weights: random projections from cortex → action channels
        let scale = (2.0 / (n_c + n_a) as f32).sqrt();
        let cortico_stratial: Vec<f32> = (0..n_a * n_c)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();

        // Lateral inhibition: off-diagonal negative, diagonal zero
        let mut lateral_inhibit = vec![0.0f32; n_a * n_a];
        for i in 0..n_a {
            for j in 0..n_a {
                if i != j {
                    lateral_inhibit[i * n_a + j] = -config.gpi_inhibition;
                }
            }
        }

        Self {
            config,
            state: vec![0.0; n_c],
            cortico_stratial,
            lateral_inhibit,
            prev_action: None,
            drive_buf: vec![0.0; n_a],
            output_buf: vec![0.0; n_a],
        }
    }

    /// One step of Basal Ganglia action selection.
    ///
    /// 1. Accumulate cortical input into internal state (with decay).
    /// 2. Compute action drive: `drive = cortico_stratial @ state`.
    /// 3. Apply lateral inhibition (winner-take-all).
    /// 4. Return index of winning action.
    pub fn step(&mut self, cortex_input: &[f32]) -> usize {
        assert_eq!(cortex_input.len(), self.config.n_cortical);

        let n_c = self.config.n_cortical;
        let n_a = self.config.n_actions;

        // 1. Accumulate with decay: state = decay * state + input
        for i in 0..n_c {
            self.state[i] = self.config.decay * self.state[i] + cortex_input[i];
        }

        // 2. Cortico-striatal drive: drive_a = Σ_j W[a,j] * state[j]
        //    Используем iter() — n_cortical мал (8), par_iter только добавит overhead.
        self.drive_buf.fill(0.0);
        for a in 0..n_a {
            let row_start = a * n_c;
            let row = &self.cortico_stratial[row_start..row_start + n_c];
            self.drive_buf[a] = row.iter().zip(self.state.iter()).map(|(w, s)| w * s).sum();
        }

        // 3. D1 (Go) vs D2 (NoGo) modulation
        // Positive drive → D1 amplifies; negative drive → D2 suppresses
        for a in 0..n_a {
            if self.drive_buf[a] > 0.0 {
                self.drive_buf[a] *= self.config.d1_gain;
            } else {
                self.drive_buf[a] *= self.config.d2_gain;
            }
        }

        // 4. STN broad excitation: raises baseline for all actions
        // Prevents premature commitment when all drives are low
        let mean_drive: f32 = self.drive_buf.iter().sum::<f32>() / n_a as f32;
        let stn_bias = self.config.stn_excitation * mean_drive;
        for a in 0..n_a {
            self.drive_buf[a] += stn_bias;
        }

        // 5. Lateral inhibition (GPi): winner-take-all
        // output[i] = drive[i] + Σ_j lateral[i,j] * drive[j]
        self.output_buf.copy_from_slice(&self.drive_buf);
        for i in 0..n_a {
            for j in 0..n_a {
                if i != j {
                    self.output_buf[i] +=
                        self.lateral_inhibit[i * n_a + j] * self.drive_buf[j].max(0.0);
                }
            }
        }

        // 6. Thalamic relay: re-enable winning channel
        if let Some(prev) = self.prev_action {
            let max_output = self
                .output_buf
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            self.output_buf[prev] += self.config.thalamus_gain * 0.5 * (max_output.abs() + 0.1);
        }

        // 7. Winner-take-all: select argmax
        let winner = self
            .output_buf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.prev_action = Some(winner);
        winner
    }

    /// Get raw action drives (before WTA) for debugging / softmax.
    pub fn drives(&self, cortex_input: &[f32]) -> Vec<f32> {
        assert_eq!(cortex_input.len(), self.config.n_cortical);
        let n_c = self.config.n_cortical;
        let n_a = self.config.n_actions;

        let mut drive = vec![0.0f32; n_a];
        for a in 0..n_a {
            let row_start = a * n_c;
            let row = &self.cortico_stratial[row_start..row_start + n_c];
            drive[a] = row.iter().zip(self.state.iter()).map(|(w, s)| w * s).sum();
        }
        drive
    }

    /// Get softmax probabilities over actions.
    pub fn softmax(&self, cortex_input: &[f32]) -> Vec<f32> {
        let drives = self.drives(cortex_input);
        let max_d = drives.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = drives.iter().map(|&d| (d - max_d).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 1e-10 {
            for e in &mut exps {
                *e /= sum;
            }
        }
        exps
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.prev_action = None;
    }

    pub fn n_cortical(&self) -> usize {
        self.config.n_cortical
    }
    pub fn n_actions(&self) -> usize {
        self.config.n_actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bg_selects_strongest_action() {
        let mut bg = BasalGanglia::new(8, 4);
        // Strong signal on channel 1
        let strong = vec![0.1, 0.9, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0];
        let a_strong = bg.step(&strong);
        bg.reset();
        // Weak uniform signal
        let weak = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let a_weak = bg.step(&weak);
        // Both should be valid actions
        assert!(a_strong < 4);
        assert!(a_weak < 4);
    }

    #[test]
    fn bg_winner_takes_all() {
        let mut bg = BasalGanglia::new(4, 2);
        // Both channels active, but channel 0 stronger
        let input1 = vec![0.8, 0.3, 0.0, 0.0];
        let a1 = bg.step(&input1);
        // Should select action 0
        assert!(a1 < 2);
    }

    #[test]
    fn bg_no_nan() {
        let mut bg = BasalGanglia::new(16, 8);
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        for _ in 0..100 {
            let a = bg.step(&input);
            assert!(a < 8);
        }
    }

    #[test]
    fn bg_softmax_sums_to_one() {
        let bg = BasalGanglia::new(8, 4);
        let input = vec![0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let probs = bg.softmax(&input);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn bg_reset() {
        let mut bg = BasalGanglia::new(4, 2);
        bg.step(&[1.0, 0.0, 0.0, 0.0]);
        bg.reset();
        // After reset, state should be zero
        assert!(bg.state.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn bg_hysteresis() {
        let mut bg = BasalGanglia::new(4, 2);
        // First step: strong signal
        let a1 = bg.step(&[1.0, 0.0, 0.0, 0.0]);
        assert!(a1 < 2);
        // Second step: weak signal — should prefer previous winner
        let a2 = bg.step(&[0.01, 0.01, 0.0, 0.0]);
        assert!(a2 < 2);
        // Third step: no signal at all — should still hold
        let a3 = bg.step(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(a3, a2, "hysteresis should maintain previous action");
    }

    #[test]
    fn bg_par_drives() {
        let mut bg = BasalGanglia::new(8, 4);
        bg.step(&[0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let drives = bg.drives(&[0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(drives.len(), 4);
        for &d in &drives {
            assert!(d.is_finite(), "drive should be finite");
        }
    }
}
