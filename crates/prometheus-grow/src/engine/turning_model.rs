//! NETMORPH terminal-segment turning models (`turning_models.cc/.hh`,
//! Randal A. Koene, 20070509).
//!
//! A TSTM decides *whether* a growth cone turns this step; the direction of
//! the turn is a separate concern (direction models). Each terminal is
//! assigned one TSTM; on branching the model is cloned into both daughters
//! (`handle_branching`).
//!
//! Ported models:
//!
//! - [`NoneTSTM`] — never turns (`predict_turn` returns −1: no uniform draw
//!   in [0,1] can be ≤ it). Disables turning for selected arbors.
//! - [`EachDtTSTM`] — exactly one turn per time step.
//! - [`LinearRateTSTM`] — turns as a Poisson process with rate `turn_rate`
//!   per unit length (fixed-step mode) or per second. Default 0.1.
//! - [`BranchCoupledTSTM`] — turn probability is `(1 + tf) · p_branch`, i.e.
//!   coupled to the arbor's branching probability with factor `tf`
//!   (NETMORPH default 9.25).
//!
//! The NETMORPH chaining protocol (contributing models weighted and summed)
//! is folded into [`TurnEvent::combined`] — the common case in practice.

use rand::Rng;

/// Which TSTM a terminal carries. `Clone` implements the NETMORPH
/// `handle_branching` protocol (both daughters inherit the parent's model).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TurnModel {
    None,
    /// One turn per step.
    EachDt {
        t_prev: f64,
    },
    /// Poisson turns at `turn_rate` per world unit of elongation.
    LinearRate {
        turn_rate: f64,
    },
    /// Turn probability `(1 + tf) · p_branch` of the branching probability.
    BranchCoupled {
        tf: f64,
    },
}

impl TurnModel {
    pub fn linear_rate(turn_rate: f64) -> Self {
        TurnModel::LinearRate { turn_rate }
    }

    pub fn branch_coupled(tf: f64) -> Self {
        TurnModel::BranchCoupled { tf }
    }

    /// NETMORPH default: `linear_rate_TSTM` with `turn_rate = 0.1`.
    pub fn default_linear() -> Self {
        TurnModel::LinearRate { turn_rate: 0.1 }
    }

    /// Probability that a turn event happens this step.
    ///
    /// `elongated_len` — length grown by this terminal since the last event
    /// (used by the fixed-step linear-rate model). `p_branch` — the arbor's
    /// branching probability this step (used by the branch-coupled model).
    pub fn predict_turn(&mut self, elongated_len: f64, p_branch: f64) -> f64 {
        match self {
            // Any r in [0,1] fails r <= -1 → never turns.
            TurnModel::None => -1.0,
            TurnModel::EachDt { .. } => 1.0,
            TurnModel::LinearRate { turn_rate } => (*turn_rate * elongated_len).min(1.0),
            TurnModel::BranchCoupled { tf } => ((1.0 + *tf) * p_branch).min(1.0),
        }
    }

    /// Draw the turn decision: `r ≤ P_turn`.
    pub fn turn_event(&mut self, elongated_len: f64, p_branch: f64, rng: &mut impl Rng) -> bool {
        let p = self.predict_turn(elongated_len, p_branch);
        rng.gen::<f64>() <= p
    }

    /// NETMORPH chaining: combine two models' probabilities with weights.
    /// Returns the combined probability normalized by the summed weights.
    pub fn combined(a: f64, wa: f64, b: f64, wb: f64) -> f64 {
        if wa + wb <= 0.0 {
            return a;
        }
        (a * wa + b * wb) / (wa + wb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn none_never_turns() {
        let mut m = TurnModel::None;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            assert!(!m.turn_event(10.0, 1.0, &mut rng));
        }
    }

    #[test]
    fn each_dt_always_turns() {
        let mut m = TurnModel::EachDt { t_prev: 0.0 };
        assert_eq!(m.predict_turn(0.0, 0.0), 1.0);
    }

    #[test]
    fn linear_rate_probability_scales_with_length() {
        let mut m = TurnModel::linear_rate(0.1);
        assert!((m.predict_turn(5.0, 0.0) - 0.5).abs() < 1e-12);
        assert_eq!(m.predict_turn(50.0, 0.0), 1.0); // clamped
    }

    #[test]
    fn branch_coupled_follows_branch_probability() {
        let mut m = TurnModel::branch_coupled(9.25);
        assert!((m.predict_turn(0.0, 0.01) - 0.1025).abs() < 1e-12);
        // Clamped to certainty.
        assert_eq!(m.predict_turn(0.0, 0.5), 1.0);
    }

    #[test]
    fn turn_event_is_deterministic_per_seed() {
        let run = |seed: u64| {
            let mut m = TurnModel::linear_rate(0.3);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            (0..100)
                .filter(|_| m.turn_event(1.0, 0.0, &mut rng))
                .count()
        };
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn combined_weights_normalize() {
        // Equal weights → plain average.
        assert!((TurnModel::combined(0.2, 1.0, 0.6, 1.0) - 0.4).abs() < 1e-12);
        // Heavier weight dominates.
        assert!((TurnModel::combined(0.2, 3.0, 0.6, 1.0) - 0.3).abs() < 1e-12);
    }
}
