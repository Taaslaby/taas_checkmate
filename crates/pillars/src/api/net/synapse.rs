//! -------------------------------------------------------------------------
//! Filename:    src/synapse.rs
//! Class:       Synaptic kinetics
//! Purpose:     (`api::synapse`)
//! Description:
//!
//! Arbitrary-tau exponential synaptic **conductances** — NeuroML
//! `expOneSynapse` / `expTwoSynapse` / `alphaSynapse` / `expThreeSynapse` —
//! with the correct temporal shape and reversal potential.
//!
//! The injected current is `g(t)·(erev − V)` (NEURON/`ops::ohmic` convention:
//! a positive term depolarises). Per-spike magnitudes follow the NeuroML/LEMS
//! component definitions exactly:
//!
//! text
//! expOneSynapse:  g(t) = A · e^(−t/τ_decay)                    peak A (instant)
//! expTwoSynapse:  g(t) = A · norm · (e^(−t/τ_decay) − e^(−t/τ_rise))
//!                                                               peak A at t_peak
//! alphaSynapse:   g(t) = A · (t/τ) · e^(1 − t/τ)               peak A at t = τ
//! expThreeSynapse: g(t) = gbase1·(B−A) + gbase2·(C−A)          two decay components
//!                                                               sharing one rise time
//!
//! where `A = weight · gbase` is the per-spike peak conductance (see
//! [`Network::register_kinetics`](crate::api::Network::register_kinetics)).
//!
//! **Short-term plasticity** (Tsodyks–Markram 1998) is exposed as
//! [`PlasticityProfile`] and applied per spike: the delivered weight is scaled
//! by `R·U` (recovery × release probability), with the canonical spike-time
//! updates `R ← R·(1−U)` (depression) and `U ← U + U0·(1−U)` (facilitation).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use std::fmt;

/// A synaptic conductance waveform.
///
/// * [`SynapseKinetics::exp_decay`] — `expOneSynapse`: `g(t) = A·e^(−t/τ)`.
/// * [`SynapseKinetics::exp_rise_decay`] — `expTwoSynapse`:
///   `g(t) = A·norm·(e^(−t/τ_decay) − e^(−t/τ_rise))`, peak = `A`.
/// * [`SynapseKinetics::alpha`] — `alphaSynapse`: `g(t) = A·(t/τ)·e^(1−t/τ)`,
///   peak = `A` at `t = τ`.
/// * [`SynapseKinetics::exp_three`] — `expThreeSynapse`: two exponential
///   decay components (`gbase1`, `gbase2`) sharing a single rise time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynapseKinetics {
    /// `expOneSynapse`: instantaneous rise, exponential decay with `tau_decay`.
    ExpOne { tau_decay: f32, erev: f32 },
    /// `expTwoSynapse`: exponential rise (`tau_rise`) + decay (`tau_decay`),
    /// peak-normalised to the per-spike jump.
    ExpTwo {
        tau_rise: f32,
        tau_decay: f32,
        erev: f32,
    },
    /// `alphaSynapse`: alpha function, rise time = decay time = `tau`.
    Alpha { tau: f32, erev: f32 },
    /// `expThreeSynapse`: two decay components (`tau_decay1`, `tau_decay2` with
    /// baseline conductances `gbase1`, `gbase2`) sharing one rise time
    /// (`tau_rise`).
    ExpThree {
        tau_rise: f32,
        tau_decay1: f32,
        tau_decay2: f32,
        gbase1: f32,
        gbase2: f32,
        erev: f32,
    },
}

impl SynapseKinetics {
    /// Single-exponential decay (`expOneSynapse`), time constant `tau_decay`.
    pub fn exp_decay(tau_decay: f32) -> Self {
        Self::ExpOne {
            tau_decay,
            erev: 0.0,
        }
    }

    /// Double-exponential rise + decay (`expTwoSynapse`).
    pub fn exp_rise_decay(tau_rise: f32, tau_decay: f32) -> Self {
        Self::ExpTwo {
            tau_rise,
            tau_decay,
            erev: 0.0,
        }
    }

    /// Alpha function (`alphaSynapse`), rise = decay = `tau`.
    pub fn alpha(tau: f32) -> Self {
        Self::Alpha { tau, erev: 0.0 }
    }

    /// Two-decay-component synapse (`expThreeSynapse`) sharing one rise time.
    pub fn exp_three(
        tau_rise: f32,
        tau_decay1: f32,
        tau_decay2: f32,
        gbase1: f32,
        gbase2: f32,
    ) -> Self {
        Self::ExpThree {
            tau_rise,
            tau_decay1,
            tau_decay2,
            gbase1,
            gbase2,
            erev: 0.0,
        }
    }

    /// Set the reversal potential (builder-style).
    pub fn with_erev(mut self, erev: f32) -> Self {
        match &mut self {
            Self::ExpOne { erev: e, .. }
            | Self::ExpTwo { erev: e, .. }
            | Self::Alpha { erev: e, .. }
            | Self::ExpThree { erev: e, .. } => *e = erev,
        }
        self
    }

    /// Reversal potential (mV).
    pub fn erev(&self) -> f32 {
        match self {
            Self::ExpOne { erev, .. }
            | Self::ExpTwo { erev, .. }
            | Self::Alpha { erev, .. }
            | Self::ExpThree { erev, .. } => *erev,
        }
    }

    /// Validate the profile: finite, positive time constants, `tau_rise <`
    /// the decay constants, non-negative `gbase1`/`gbase2` (with a positive
    /// sum), and finite `erev`.
    pub fn validate(&self) -> Result<(), String> {
        let erev_ok = self.erev().is_finite();
        if !erev_ok {
            return Err("erev must be finite".into());
        }
        match *self {
            Self::ExpOne { tau_decay, .. } => {
                if !tau_decay.is_finite() || tau_decay <= 0.0 {
                    return Err(format!("tau_decay must be finite and > 0, got {tau_decay}"));
                }
            }
            Self::ExpTwo {
                tau_rise,
                tau_decay,
                ..
            } => {
                if !tau_decay.is_finite() || tau_decay <= 0.0 {
                    return Err(format!("tau_decay must be finite and > 0, got {tau_decay}"));
                }
                if !tau_rise.is_finite() || tau_rise <= 0.0 {
                    return Err(format!("tau_rise must be finite and > 0, got {tau_rise}"));
                }
                if tau_rise >= tau_decay {
                    return Err(format!(
                        "tau_rise ({tau_rise}) must be < tau_decay ({tau_decay})"
                    ));
                }
            }
            Self::Alpha { tau, .. } => {
                if !tau.is_finite() || tau <= 0.0 {
                    return Err(format!("tau must be finite and > 0, got {tau}"));
                }
            }
            Self::ExpThree {
                tau_rise,
                tau_decay1,
                tau_decay2,
                gbase1,
                gbase2,
                ..
            } => {
                if !tau_rise.is_finite() || tau_rise <= 0.0 {
                    return Err(format!("tau_rise must be finite and > 0, got {tau_rise}"));
                }
                if !tau_decay1.is_finite() || tau_decay1 <= tau_rise {
                    return Err(format!(
                        "tau_decay1 ({tau_decay1}) must be finite and > tau_rise ({tau_rise})"
                    ));
                }
                if !tau_decay2.is_finite() || tau_decay2 <= tau_rise {
                    return Err(format!(
                        "tau_decay2 ({tau_decay2}) must be finite and > tau_rise ({tau_rise})"
                    ));
                }
                for (name, g) in [("gbase1", gbase1), ("gbase2", gbase2)] {
                    if !g.is_finite() || g < 0.0 {
                        return Err(format!("{name} must be finite and >= 0, got {g}"));
                    }
                }
                if gbase1 + gbase2 <= 0.0 {
                    return Err("gbase1 + gbase2 must be > 0".into());
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for SynapseKinetics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpOne { tau_decay, .. } => write!(f, "expOne(tauDecay={tau_decay})"),
            Self::ExpTwo {
                tau_rise,
                tau_decay,
                ..
            } => write!(f, "expTwo(tauRise={tau_rise}, tauDecay={tau_decay})"),
            Self::Alpha { tau, .. } => write!(f, "alpha(tau={tau})"),
            Self::ExpThree {
                tau_rise,
                tau_decay1,
                tau_decay2,
                ..
            } => write!(
                f,
                "expThree(tauRise={tau_rise}, tauDecay1={tau_decay1}, tauDecay2={tau_decay2})"
            ),
        }
    }
}

/// Short-term plasticity profile (Tsodyks–Markram 1998).
///
/// Each incoming spike is delivered with its weight scaled by the synapse's
/// current `plasticityFactor = R·U` (post-spike, exactly as the NeuroML
/// `blockingPlasticSynapse`/`tsodyksMarkram*Mechanism` OnEvent semantics):
///
/// ```text
/// depression:            dR/dt = (1 − R)/τ_rec      on spike: R ← R·(1 − U)
/// depression + facilitation:  + dU/dt = (U0 − U)/τ_fac
///                                                   on spike: U ← U + U0·(1 − U)
/// delivered weight = weight · R · U
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasticityProfile {
    /// Baseline release probability `U0` (0..1].
    pub init_release_prob: f32,
    /// Recovery time constant (ms) — `R` relaxes back to 1.
    pub tau_rec: f32,
    /// Facilitation time constant (ms); `None` = depression-only.
    pub tau_fac: Option<f32>,
}

impl PlasticityProfile {
    /// Depression-only Tsodyks–Markram (1997).
    pub fn depression(init_release_prob: f32, tau_rec: f32) -> Self {
        Self {
            init_release_prob,
            tau_rec,
            tau_fac: None,
        }
    }

    /// Depression + facilitation (Tsodyks, Pawelzik & Markram 1998).
    pub fn depression_facilitation(init_release_prob: f32, tau_rec: f32, tau_fac: f32) -> Self {
        Self {
            init_release_prob,
            tau_rec,
            tau_fac: Some(tau_fac),
        }
    }

    /// Validate: finite `0 < U0 <= 1`, finite positive `tau_rec`, and, when
    /// set, finite positive `tau_fac`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.init_release_prob.is_finite()
            || self.init_release_prob <= 0.0
            || self.init_release_prob > 1.0
        {
            return Err(format!(
                "init_release_prob must be in (0, 1], got {}",
                self.init_release_prob
            ));
        }
        if !self.tau_rec.is_finite() || self.tau_rec <= 0.0 {
            return Err(format!(
                "tau_rec must be finite and > 0, got {}",
                self.tau_rec
            ));
        }
        if let Some(tf) = self.tau_fac {
            if !tf.is_finite() || tf <= 0.0 {
                return Err(format!("tau_fac must be finite and > 0, got {tf}"));
            }
        }
        Ok(())
    }
}

/// Peak-normalisation constant for a rise+decay double-exponential.
///
/// `wf·(e^(−t/τ_decay) − e^(−t/τ_rise))` peaks at exactly `1.0` at
/// `t_peak = τ_rise·τ_decay/(τ_decay−τ_rise)·ln(τ_decay/τ_rise)` — the
/// NeuroML/LEMS `waveformFactor`. Requires `0 < τ_rise < τ_decay`.
pub(crate) fn exp_waveform_factor(tau_rise: f32, tau_decay: f32) -> f32 {
    let t_peak = (tau_rise * tau_decay) / (tau_decay - tau_rise) * (tau_decay / tau_rise).ln();
    let h_peak = (-t_peak / tau_decay).exp() - (-t_peak / tau_rise).exp();
    1.0 / h_peak
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_decay_validation() {
        assert!(SynapseKinetics::exp_decay(3.0).validate().is_ok());
        assert!(SynapseKinetics::exp_decay(0.0).validate().is_err());
        assert!(SynapseKinetics::exp_decay(-1.0).validate().is_err());
        assert!(SynapseKinetics::exp_decay(f32::NAN).validate().is_err());
    }

    #[test]
    fn exp_two_rise_must_be_less_than_decay() {
        assert!(SynapseKinetics::exp_rise_decay(1.0, 3.0).validate().is_ok());
        assert!(SynapseKinetics::exp_rise_decay(3.0, 1.0)
            .validate()
            .is_err());
        assert!(SynapseKinetics::exp_rise_decay(2.0, 2.0)
            .validate()
            .is_err());
        assert!(SynapseKinetics::exp_rise_decay(0.0, 3.0)
            .validate()
            .is_err());
    }

    #[test]
    fn alpha_validation() {
        assert!(SynapseKinetics::alpha(2.0).validate().is_ok());
        assert!(SynapseKinetics::alpha(0.0).validate().is_err());
        assert!(SynapseKinetics::alpha(-1.0).validate().is_err());
        assert!(SynapseKinetics::alpha(f32::NAN).validate().is_err());
    }

    #[test]
    fn exp_three_validation() {
        assert!(SynapseKinetics::exp_three(1.0, 3.0, 10.0, 1.0, 0.5)
            .validate()
            .is_ok());
        // Decay must be slower than rise.
        assert!(SynapseKinetics::exp_three(3.0, 1.0, 10.0, 1.0, 0.5)
            .validate()
            .is_err());
        // Non-negative conductances, positive sum.
        assert!(SynapseKinetics::exp_three(1.0, 3.0, 10.0, -1.0, 0.5)
            .validate()
            .is_err());
        assert!(SynapseKinetics::exp_three(1.0, 3.0, 10.0, 0.0, 0.0)
            .validate()
            .is_err());
        assert!(SynapseKinetics::exp_three(1.0, 3.0, f32::NAN, 1.0, 0.5)
            .validate()
            .is_err());
    }

    #[test]
    fn plasticity_profile_validation() {
        assert!(PlasticityProfile::depression(0.5, 800.0).validate().is_ok());
        assert!(PlasticityProfile::depression_facilitation(0.2, 500.0, 50.0)
            .validate()
            .is_ok());
        assert!(PlasticityProfile::depression(0.0, 800.0)
            .validate()
            .is_err());
        assert!(PlasticityProfile::depression(1.5, 800.0)
            .validate()
            .is_err());
        assert!(PlasticityProfile::depression(f32::NAN, 800.0)
            .validate()
            .is_err());
        assert!(PlasticityProfile::depression(0.5, 0.0).validate().is_err());
        assert!(PlasticityProfile::depression_facilitation(0.2, 500.0, -1.0)
            .validate()
            .is_err());
    }

    #[test]
    fn exp_two_factor_peaks_at_one() {
        // τ_rise=1, τ_decay=3: the normalised double-exponential must peak at
        // exactly 1.0 at t_peak = (1·3)/(3−1)·ln(3/1).
        let wf = exp_waveform_factor(1.0, 3.0);
        let t_peak = (1.0f32 * 3.0) / (3.0 - 1.0) * (3.0f32 / 1.0).ln();
        let peak = wf * ((-t_peak / 3.0).exp() - (-t_peak / 1.0).exp());
        assert!((peak - 1.0).abs() < 1e-6, "peak = {peak}");
    }
}
