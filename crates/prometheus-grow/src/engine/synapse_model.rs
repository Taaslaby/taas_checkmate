//! NETMORPH synapse-formation model (`synapse_formation_model.cc` / `.hh`).
//!
//! The original model splits connectivity into two stages:
//!
//! 1. **Candidate sites** — an axon segment and a dendrite segment are close
//!    enough when their distance is below a *per-pair* maximum fibre distance
//!    (`maxfibredistance[pre][post]`, CLI `D_synmax.*.*`).
//! 2. **Actual synapses** — each candidate draws the postsynaptic receptors it
//!    will carry from a pre×post compatibility matrix
//!    (`possible_syntypes[pre][post][receptor]`, CLI `P_receptor.*.*`).
//!    A value `< 0` means the receptor can never form there.
//!
//! The crucial idea ported here: **the sign of a connection is a property of
//! the PRE × POST pair, not of the presynaptic cell alone.** The same
//! transmitter can be excitatory onto one postsynaptic type and inhibitory
//! onto another, depending on which receptor the postsynaptic cell expresses.
//! In *Drosophila* this is literal biology: glutamate (VGlut) excites through
//! AMPA/NMDA receptors and inhibits through the GluCl chloride channel.

use rand::Rng;

use crate::world::{SynapseKind};

/// Postsynaptic receptor type (NETMORPH `synapse_type`, three candidates:
/// AMPAR, NMDAR, GABAR).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Receptor {
    /// Fast excitatory glutamate receptor (NETMORPH `syntype_AMPAR`).
    Ampa,
    /// Slow excitatory glutamate receptor (NETMORPH `syntype_NMDAR`).
    Nmda,
    /// Inhibitory GABA-A chloride receptor (NETMORPH `syntype_GABAR`).
    Gaba,
}

impl Receptor {
    pub fn as_usize(self) -> usize {
        match self {
            Receptor::Ampa => 0,
            Receptor::Nmda => 1,
            Receptor::Gaba => 2,
        }
    }

    pub fn is_excitatory(self) -> bool {
        matches!(self, Receptor::Ampa | Receptor::Nmda)
    }

    pub fn is_inhibitory(self) -> bool {
        self == Receptor::Gaba
    }

    pub const ALL: [Receptor; 3] = [Receptor::Ampa, Receptor::Nmda, Receptor::Gaba];
}

/// NETMORPH-style synapse-formation model. Matrices are indexed by
/// [`CellType::as_usize`] (0 = Excitatory, 1 = Inhibitory, 2 = Glia).
#[derive(Clone, Debug)]
pub struct SynapseFormationModel {
    /// Maximum fibre distance per (pre, post) type pair — NETMORPH
    /// `maxfibredistance` / `D_synmax.*.*`. Units are the caller's world units
    /// (NETMORPH uses µm; this engine is unit-agnostic, so scale to suit).
    /// `<= 0.0` disallows the pair.
    pub max_fibre_distance: [[f64; 3]; 3],
    /// Probability that receptor `r` forms at a pre→post connection —
    /// NETMORPH `possible_syntypes` / `P_receptor.*.*`. `< 0.0` means the
    /// receptor can never form at that connection.
    pub receptor_prob: [[[f64; 3]; 3]; 3],
    /// Never form autapses (pre == post), NETMORPH `no_autapses`.
    pub no_autapses: bool,
}

impl SynapseFormationModel {
    /// NETMORPH defaults adapted to the fly 2-type + glia model:
    ///
    /// * excitatory pre (ACh / glutamate) → AMPA/NMDA (excitatory);
    /// * inhibitory pre (GABA) → GABA-A (inhibitory);
    /// * glia never synapses.
    pub fn netmorph_defaults() -> Self {
        let exc = [1.0, 0.4, -9.9]; // AMPA always, NMDA 40%, GABA never
        let inh = [-9.9, -9.9, 1.0]; // only GABA
        let none = [-9.9, -9.9, -9.9];
        SynapseFormationModel {
            // NETMORPH literal values (µm): spiny excitatory pairs bridge 1.0,
            // interneuron pairs 0.1, untyped/glia 0.0.
            max_fibre_distance: [
                [1.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            receptor_prob: [
                // pre = Excitatory
                [exc, exc, none],
                // pre = Inhibitory
                [inh, inh, none],
                // pre = Glia
                [none, none, none],
            ],
            no_autapses: true,
        }
    }

    /// The "receptor switch" — one presynaptic transmitter, opposite signs on
    /// different postsynaptic types. This is the fly GluCl story:
    ///
    /// ```text
    /// glutamate → AMPA/NMDA on some posts  → Excitatory
    /// glutamate → GluCl (Cl⁻) on others    → Inhibitory
    /// ```
    ///
    /// Here the excitatory pre type is excitatory onto excitatory posts and
    /// inhibitory onto inhibitory posts. The sign is decided by the pre×post
    /// receptor matrix, not by the presynaptic cell alone.
    pub fn receptor_switch() -> Self {
        let exc_post = [1.0, 0.4, -9.9]; // AMPA/NMDA → excitatory
        let inh_post = [-9.9, -9.9, 1.0]; // GluCl/GABA → inhibitory
        let inh = [-9.9, -9.9, 1.0];
        let none = [-9.9, -9.9, -9.9];
        SynapseFormationModel {
            max_fibre_distance: [
                [1.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            receptor_prob: [
                // pre = Excitatory: exc onto exc, inh onto inh (the switch)
                [exc_post, inh_post, none],
                // pre = Inhibitory: always inhibitory
                [inh, inh, none],
                [none, none, none],
            ],
            no_autapses: true,
        }
    }

    /// Draw the receptors that form at a pre→post connection and return the
    /// resulting synapse sign, or `None` if no receptor forms (no synapse).
    ///
    /// NETMORPH draws each receptor independently
    /// (`rand <= possible_syntypes[pre][post][r]`); a connection can therefore
    /// carry several receptors. Since [`SynapseKind`] is binary, an inhibitory
    /// (GABA) receptor takes precedence, mirroring the fact that a GABAergic
    /// component dominates the sign.
    pub fn synapse_kind(&self, pre: usize, post: usize, rng: &mut impl Rng) -> Option<SynapseKind> {
        let mut excitatory = false;
        let mut inhibitory = false;
        for r in Receptor::ALL {
            let p = self.receptor_prob[pre][post][r.as_usize()];
            if p >= 0.0 && rng.gen::<f64>() <= p {
                if r.is_excitatory() {
                    excitatory = true;
                }
                if r.is_inhibitory() {
                    inhibitory = true;
                }
            }
        }
        if inhibitory {
            Some(SynapseKind::Inhibitory)
        } else if excitatory {
            Some(SynapseKind::Excitatory)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn netmorph_defaults_gives_pre_determined_sign() {
        let m = SynapseFormationModel::netmorph_defaults();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..64 {
            // excitatory pre → excitatory posts (AMPA always present).
            assert_eq!(
                m.synapse_kind(0, 0, &mut rng),
                Some(SynapseKind::Excitatory)
            );
            // inhibitory pre → always inhibitory (GABA always present).
            assert_eq!(
                m.synapse_kind(1, 0, &mut rng),
                Some(SynapseKind::Inhibitory)
            );
        }
        // glia never synapses.
        assert_eq!(m.synapse_kind(2, 0, &mut rng), None);
    }

    #[test]
    fn receptor_switch_flips_sign_by_post_type() {
        let m = SynapseFormationModel::receptor_switch();
        let mut rng = StdRng::seed_from_u64(2);
        // Same excitatory pre: excitatory onto excitatory posts…
        assert_eq!(
            m.synapse_kind(0, 0, &mut rng),
            Some(SynapseKind::Excitatory)
        );
        // …but inhibitory onto inhibitory posts (GluCl switch).
        assert_eq!(
            m.synapse_kind(0, 1, &mut rng),
            Some(SynapseKind::Inhibitory)
        );
    }
}
