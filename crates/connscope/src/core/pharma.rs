//! # Drug / pharmacology domain (`core::pharma`)
//!
//! A drug is a mapping from a real molecule to a modeled target:
//!
//! ```text
//! concentration (µM)  ──Hill──▶  occupancy (0..1)  ──▶  scale (multiplier)
//! ```
//!
//! * **channel targets** (`NaV`, `KV`, `CaT`, `Leak`, `Hcn`, `M`) scale a
//!   conductance in HH / HAAP;
//! * **receptor targets** (`GabaA`, `Nach`, `GluCl`) scale the sign/strength of
//!   synapses by neurotransmitter.
//!
//! The library itself is data: `resources/pharma/drugs.toml`, embedded at
//! compile time via [`DRUGS_TOML`]. Values are literature-representative
//! (order-of-magnitude), each entry carries a `reference` to check — see the
//! TOML header for the honesty contract.
//!
//! `scale(0.0) == 1.0` always: zero concentration means "no drug".

use serde::Deserialize;

use super::nt::Nt;

/// The `resources/pharma/drugs.toml` payload, embedded at compile time so the
/// binary stays self-contained (same pattern as the branding PNGs).
const DRUGS_TOML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/pharma/drugs.toml"
));

/// A drug target: a modeled conductance or a neurotransmitter receptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Target {
    /// Voltage-gated Na⁺ (HH `G_Na`).
    NaV,
    /// Voltage-gated K⁺ (HH `G_K`, Morris-Lecar `G_K`).
    KV,
    /// T-type Ca²⁺ (HH `G_CaT`).
    CaT,
    /// Leak conductance (HH `G_Leak`).
    Leak,
    /// Hyperpolarization-activated HCN / Ih (HAAP `g_h`).
    Hcn,
    /// KCNQ/Kv7 M-current (HAAP `g_m`).
    M,
    /// GABA_A receptor → GABA synapses (inhibitory).
    GabaA,
    /// Nicotinic acetylcholine receptor → ACh synapses (excitatory).
    Nach,
    /// Glutamate-gated Cl⁻ channel → Glu synapses (inhibitory in the fly).
    GluCl,
}

impl Target {
    /// Short display name.
    pub fn label(self) -> &'static str {
        match self {
            Target::NaV => "Na⁺ (NaV)",
            Target::KV => "K⁺ (KV)",
            Target::CaT => "Ca²⁺ (CaV-T)",
            Target::Leak => "Leak",
            Target::Hcn => "Ih (HCN)",
            Target::M => "M-ток (Kv7)",
            Target::GabaA => "GABA_A",
            Target::Nach => "nAChR",
            Target::GluCl => "GluCl",
        }
    }

    /// `true` for conductance targets, `false` for synaptic (receptor) targets.
    pub fn is_channel(self) -> bool {
        matches!(
            self,
            Target::NaV | Target::KV | Target::CaT | Target::Leak | Target::Hcn | Target::M
        )
    }

    /// The neurotransmitter a receptor target acts on (`None` for channels).
    /// Used by `sim::apply_pharma` to scale synapses by presynaptic NT.
    pub fn receptor_nt(self) -> Option<Nt> {
        match self {
            Target::GabaA => Some(Nt::Gaba),
            Target::Nach => Some(Nt::Ach),
            Target::GluCl => Some(Nt::Glut),
            _ => None,
        }
    }
}

/// What the drug does to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Effect {
    /// Reduces conductance (channel blocker). `potency_um` is IC50.
    Block,
    /// Increases conductance (channel opener). `potency_um` is EC50.
    Open,
    /// Reduces postsynaptic effect (receptor antagonist). `potency_um` is IC50.
    Antagonize,
    /// Activates the receptor (agonist). `potency_um` is EC50.
    Agonize,
}

impl Effect {
    /// Short display name.
    pub fn label(self) -> &'static str {
        match self {
            Effect::Block => "блокатор",
            Effect::Open => "активатор",
            Effect::Antagonize => "антагонист",
            Effect::Agonize => "агонист",
        }
    }

    /// Whether this effect *reduces* the target's contribution.
    pub fn is_depressing(self) -> bool {
        matches!(self, Effect::Block | Effect::Antagonize)
    }
}

fn default_hill() -> f32 {
    1.0
}

/// One drug entry in the library.
#[derive(Debug, Clone, Deserialize)]
pub struct Drug {
    /// Display / lookup name.
    pub name: String,
    pub target: Target,
    pub effect: Effect,
    /// IC50 for blockers/antagonists, EC50 for openers/agonists — µM.
    pub potency_um: f32,
    /// Hill coefficient (steepness). Default `1.0`.
    #[serde(default = "default_hill")]
    pub hill: f32,
    /// Saturating enhancement factor for openers/agonists (ignored otherwise).
    #[serde(default)]
    pub emax: f32,
    /// One-line mechanism.
    pub mechanism: String,
    /// Source (IUPHAR/BPS ligand or paper).
    pub reference: String,
    /// Honest caveat (e.g. subtype sensitivity). Optional.
    #[serde(default)]
    pub note: Option<String>,
}

impl Drug {
    /// Fraction of target bound, `0..1`, by the Hill equation.
    pub fn occupancy(&self, c_um: f32) -> f32 {
        if c_um <= 0.0 {
            return 0.0;
        }
        let k = self.potency_um.max(1e-6);
        let n = self.hill.max(0.1);
        let cn = c_um.powf(n);
        let kn = k.powf(n);
        cn / (cn + kn)
    }

    /// Multiplier on the target conductance / synapse strength.
    ///
    /// * `1.0` = no drug (always at `c = 0`);
    /// * blockers/antagonists → `1 − occupancy` (→ `0` at saturation);
    /// * openers/agonists → `1 + emax·occupancy` (→ `1 + emax` at saturation).
    pub fn scale(&self, c_um: f32) -> f32 {
        let occ = self.occupancy(c_um);
        if self.effect.is_depressing() {
            1.0 - occ
        } else {
            1.0 + self.emax * occ
        }
    }

    /// `true` if the target is a conductance (channel), not a receptor.
    #[allow(dead_code)] // API surface for the next step (drug → conductance wiring).
    pub fn is_channel(&self) -> bool {
        self.target.is_channel()
    }
}

/// The parsed drug library.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DrugLibrary {
    pub drugs: Vec<Drug>,
}

impl DrugLibrary {
    /// Parse the embedded `drugs.toml`. Panics on malformed data (it is a
    /// compile-time-checked asset, so a bad library is a bug, not a runtime error).
    pub fn load() -> Self {
        let lib: DrugLibrary =
            toml::from_str(DRUGS_TOML).expect("malformed resources/pharma/drugs.toml");
        assert!(!lib.drugs.is_empty(), "drug library must not be empty");
        lib
    }

    /// Look up a drug by name (case-insensitive).
    /// Used by CLI/console flags `--drug <имя>` (`cli::pharma_dose_from_flags`).
    pub fn get(&self, name: &str) -> Option<&Drug> {
        self.drugs
            .iter()
            .find(|d| d.name.eq_ignore_ascii_case(name))
    }

    /// All drugs acting on `target`.
    #[allow(dead_code)] // API surface for the next step (dose-response sweep).
    pub fn by_target(&self, target: Target) -> impl Iterator<Item = &Drug> {
        self.drugs.iter().filter(move |d| d.target == target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> DrugLibrary {
        DrugLibrary::load()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn library_loads_and_is_non_empty() {
        let lib = lib();
        assert!(!lib.drugs.is_empty());
        // Every drug must carry a sane potency and a reference.
        for d in &lib.drugs {
            assert!(d.potency_um > 0.0, "{} potency must be > 0", d.name);
            assert!(!d.reference.is_empty(), "{} missing reference", d.name);
        }
    }

    #[test]
    fn zero_concentration_is_neutral_for_every_drug() {
        for d in &lib().drugs {
            assert!(
                approx(d.scale(0.0), 1.0),
                "{} must be neutral at c=0",
                d.name
            );
        }
    }

    #[test]
    fn ttx_half_max_at_ic50() {
        let l = lib();
        let ttx = l.get("ttx").unwrap();
        assert_eq!(ttx.target, Target::NaV);
        assert_eq!(ttx.effect, Effect::Block);
        // Hill n=1 → at C=IC50 occupancy = 0.5 → scale = 0.5.
        assert!(approx(ttx.occupancy(0.01), 0.5));
        assert!(approx(ttx.scale(0.01), 0.5));
    }

    #[test]
    fn blocker_saturates_to_zero() {
        let l = lib();
        let ttx = l.get("ttx").unwrap();
        // 1000 µM >> IC50 (0.01 µM) → essentially fully blocked.
        assert!(ttx.scale(1000.0) < 0.001);
    }

    #[test]
    fn opener_saturates_to_one_plus_emax() {
        let l = lib();
        let retigabine = l.get("retigabine").unwrap();
        assert_eq!(retigabine.effect, Effect::Open);
        assert!(approx(retigabine.scale(0.0), 1.0));
        // 1e6 µM >> EC50 (10 µM) → occupancy ≈ 1 → scale ≈ 1 + emax.
        assert!(approx(retigabine.scale(1e6), 1.0 + retigabine.emax));
    }

    #[test]
    fn hill_steepness_increases_occupancy() {
        // A hypothetical steep blocker: at C=10·IC50, n=2 gives occupancy
        // 100/(100+1) ≈ 0.990, while n=1 gives 10/11 ≈ 0.909.
        let n1 = Drug {
            name: "steep1".into(),
            target: Target::KV,
            effect: Effect::Block,
            potency_um: 1.0,
            hill: 1.0,
            emax: 0.0,
            mechanism: String::new(),
            reference: String::new(),
            note: None,
        };
        let n2 = Drug {
            hill: 2.0,
            ..n1.clone()
        };
        assert!(n2.occupancy(10.0) > n1.occupancy(10.0));
    }

    #[test]
    fn receptor_targets_map_to_neurotransmitters() {
        let l = lib();
        let picrotoxin = l.get("picrotoxin").unwrap();
        assert_eq!(picrotoxin.target.receptor_nt(), Some(Nt::Gaba));
        let ivermectin = l.get("ivermectin").unwrap();
        assert_eq!(ivermectin.target, Target::GluCl);
        assert_eq!(ivermectin.target.receptor_nt(), Some(Nt::Glut));
        // Channel targets have no neurotransmitter.
        let ttx = l.get("ttx").unwrap();
        assert_eq!(ttx.target.receptor_nt(), None);
        assert!(ttx.is_channel());
    }
}
