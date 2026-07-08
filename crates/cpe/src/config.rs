//! CPE configuration — all hyperparameters in one place.

/// ChaosPatternEngine configuration.
///
/// Default values are tuned for Pong on a 800-neuron connectome:
/// - τ_trace ≈ 85ms (0.90^10 per 100ms tick)
/// - LR = 0.02 (conservative STDP)
/// - β = 0.3 (moderate chaos/pattern gap)
/// - α = 0.3 (moderate entropy penalty)
#[derive(Clone, Debug)]
pub struct CpeConfig {
    // ── Kuramoto dynamics ──
    pub dt: f32,
    pub spike_threshold: f32,
    pub omega_base: f32,
    pub omega_spread: f32,

    // ── STDP ──
    pub lr_stdp: f32,
    pub elig_decay: f32,   // per-step eligibility decay (0.90 → τ≈85ms @ DT=0.01)
    pub elig_add: f32,     // per-spike eligibility increment
    pub w_min: f32,
    pub w_max: f32,

    // ── Dopamine gate ──
    pub alpha: f32,        // entropy penalty weight
    pub beta: f32,         // chaos/pattern gap weight
    pub h_target: f32,     // target entropy

    // ── T2-SPIGRAD ──
    pub t2_lr: f32,
    pub t2_n_osc: usize,   // oscillators per population (×3 populations)
    pub t2_drive_gain: f32,
    pub t2_alpha: f32,
    pub t2_freq_min: f32,
    pub t2_freq_max: f32,
    pub t2_c12: f32,       // coupling pop1→pop2
    pub t2_c13: f32,       // coupling pop1→pop3
    pub t2_c23: f32,       // coupling pop2→pop3
}

impl Default for CpeConfig {
    fn default() -> Self {
        Self {
            // Kuramoto
            dt: 0.01,
            spike_threshold: 2.5,
            omega_base: 3.0,
            omega_spread: 1.5,

            // STDP
            lr_stdp: 0.02,
            elig_decay: 0.90,
            elig_add: 0.15,
            w_min: 0.01,
            w_max: 3.0,

            // Dopamine
            alpha: 0.3,
            beta: 0.3,
            h_target: 3.0,

            // T2-SPIGRAD
            t2_lr: 0.005,
            t2_n_osc: 16,
            t2_drive_gain: 1.0,
            t2_alpha: 1.0,
            t2_freq_min: 1.0,
            t2_freq_max: 8.0,
            t2_c12: 0.3,
            t2_c13: 0.2,
            t2_c23: 0.1,
        }
    }
}
