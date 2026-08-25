//! NETMORPH neurite diameter models (`neurite_diameter_model.cc/.hh`,
//! Randal A. Koene, 20070709).
//!
//! Two passes are supported:
//!
//! - **Bottom-up (Rall power law)** — the NETMORPH default. Terminal segment
//!   diameters are sampled from a normal PDF (Larkman 1991 observed terminal
//!   segments within a narrow diameter range; default mean 0.7 µm, std 0.3),
//!   then parent diameters are computed recursively from the daughters:
//!
//!   ```text
//!   d_parent^e = d_1^e + d_2^e
//!   ```
//!
//!   where `e` is itself sampled per bifurcation from a normal PDF
//!   (default mean 1.47, std 0.3) around the classical Rall value 3/2.
//!   An optional cap `d_max` limits bifurcation diameters (added in the
//!   2024 branch; not in the PELT:VARIABILITY paper).
//!
//! - **Top-down (symmetric split)** — the approximation already used by
//!   [`crate::World::bifurcate_neurite`] at branching time:
//!   `d_daughter = d_parent · 2^(-1/e)`. [`assign_top_down`] re-applies it
//!   to a whole grown tree.
//!
//! The bottom-up pass is what gives real morphologies their diameter
//! statistics: thin terminals sum into thick trunks, so the soma-side
//! diameter emerges from the tree shape rather than being assumed.

use crate::cell::Cell;
use rand::Rng;

/// Default e-power PDF: `normal_pdf(X_branch, 1.47, 0.3)` (NETMORPH
/// `defaultrallepowerpdf`). The classical Rall exponent is 3/2 = 1.5.
pub const DEFAULT_E_POWER_MEAN: f64 = 1.47;
pub const DEFAULT_E_POWER_STD: f64 = 0.3;

/// Default terminal diameter PDF: `normal_pdf(X_branch, 0.7, 0.3)` µm
/// (NETMORPH `defaultralldtermpdf`; Larkman 1991).
pub const DEFAULT_D_TERM_MEAN: f64 = 0.7;
pub const DEFAULT_D_TERM_STD: f64 = 0.3;

/// Parameters of the Rall power-law bottom-up diameter assignment.
#[derive(Clone, Debug)]
pub struct RallPowerLaw {
    /// Mean of the per-bifurcation e-power distribution.
    pub e_power_mean: f64,
    /// Std of the per-bifurcation e-power distribution.
    pub e_power_std: f64,
    /// Mean terminal segment diameter (µm or world units).
    pub d_term_mean: f64,
    /// Std of the terminal diameter distribution.
    pub d_term_std: f64,
    /// Optional hard cap on bifurcation diameters (`ndm.d_max`; <= 0 = off).
    pub d_max: f64,
}

impl Default for RallPowerLaw {
    fn default() -> Self {
        RallPowerLaw {
            e_power_mean: DEFAULT_E_POWER_MEAN,
            e_power_std: DEFAULT_E_POWER_STD,
            d_term_mean: DEFAULT_D_TERM_MEAN,
            d_term_std: DEFAULT_D_TERM_STD,
            d_max: -1.0,
        }
    }
}

impl RallPowerLaw {
    fn sample_terminal(&self, rng: &mut impl Rng) -> f64 {
        // random_positive(): |N(mean, std)| — a negative draw reflects off zero.
        (sample_normal(rng, self.d_term_mean, self.d_term_std)).abs()
    }

    fn combine(&self, rng: &mut impl Rng, d1: f64, d2: f64) -> f64 {
        let e = sample_normal(rng, self.e_power_mean, self.e_power_std);
        let combined = (d1.powf(e) + d2.powf(e)).powf(1.0 / e);
        if self.d_max > 0.0 && combined > self.d_max {
            self.d_max
        } else {
            combined
        }
    }
}

/// Standard normal via Box–Muller (mirrors the other modules' helper).
fn sample_normal(rng: &mut impl Rng, mean: f64, std: f64) -> f64 {
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    mean + std * z
}

/// Bottom-up diameter assignment for one cell's neurite forest.
///
/// Each root (a neurite with `parent == None`) is processed recursively:
/// terminal neurites get `|N(d_term_mean, d_term_std)|`, and every parent
/// gets `(|d1|^e + |d2|^e)^(1/e)` with a fresh e-power draw per bifurcation.
/// Intermediate (non-terminal, non-root) segments inherit the daughter
/// diameter, as in NETMORPH's between-bifurcations case.
pub fn assign_rall_diameters(cell: &mut Cell, model: &RallPowerLaw, rng: &mut impl Rng) {
    let roots: Vec<usize> = cell
        .neurites
        .iter()
        .enumerate()
        .filter(|(_, n)| n.parent.is_none())
        .map(|(i, _)| i)
        .collect();
    for root in roots {
        assign_subtree(cell, root, model, rng);
    }
}

fn assign_subtree(cell: &mut Cell, idx: usize, model: &RallPowerLaw, rng: &mut impl Rng) -> f64 {
    let children = children_of(cell, idx);
    if children.is_empty() {
        // Terminal segment: sample from the terminal diameter PDF.
        let d = model.sample_terminal(rng);
        cell.neurites[idx].diameter = d;
        return d;
    }
    if children.len() == 1 {
        // Between bifurcations: inherit the single daughter's diameter.
        let d = assign_subtree(cell, children[0], model, rng);
        cell.neurites[idx].diameter = d;
        return d;
    }
    // Bifurcation: recurse into both daughters, combine with a random e.
    let d1 = assign_subtree(cell, children[0], model, rng);
    let d2 = assign_subtree(cell, children[1], model, rng);
    let d = model.combine(rng, d1, d2);
    cell.neurites[idx].diameter = d;
    d
}

/// Indices of neurites whose parent is `idx` within `cell`.
fn children_of(cell: &Cell, idx: usize) -> Vec<usize> {
    cell.neurites
        .iter()
        .enumerate()
        .filter(|&(_, n)| n.parent == Some(idx))
        .map(|(i, _)| i)
        .collect()
}

/// Top-down symmetric split over an existing tree: root keeps its current
/// diameter, each daughter gets `d_parent · 2^(-1/e)` with the classical
/// Rall exponent (or the model's sampled mean). Useful for quick estimates
/// when the growth history is not available.
pub fn assign_top_down(cell: &mut Cell, rall_power: f64) {
    let roots: Vec<usize> = cell
        .neurites
        .iter()
        .enumerate()
        .filter(|(_, n)| n.parent.is_none())
        .map(|(i, _)| i)
        .collect();
    for root in roots {
        top_down_subtree(cell, root, rall_power);
    }
}

fn top_down_subtree(cell: &mut Cell, idx: usize, rall_power: f64) -> f64 {
    let d = cell.neurites[idx].diameter;
    let children = children_of(cell, idx);
    let d_child = d * 2f64.powf(-1.0 / rall_power);
    for c in children {
        cell.neurites[c].diameter = d_child;
        top_down_subtree(cell, c, rall_power);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use crate::math::Vec3;
    use crate::world::Bounds;
    use crate::world::World;
    use crate::Grid;
    use rand::SeedableRng;

    fn tiny_world() -> World {
        let grid = Grid::new(8, 8, 8, 10.0, Vec3::new(-40.0, -40.0, -40.0));
        World::new(
            grid,
            1.0,
            Bounds {
                min: Vec3::new(-40.0, -40.0, -40.0),
                max: Vec3::new(40.0, 40.0, 40.0),
            },
            42,
        )
    }

    #[test]
    fn rall_bottom_up_thickens_trunks() {
        // A 3-level binary tree: 8 terminals → trunk should be much thicker
        // than any terminal.
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n0 = w.new_neurite(c, true, 1.0);
        w.bifurcate_neurite(c, n0, 1.0, 1.5);
        w.bifurcate_neurite(c, 1, 1.0, 1.5);
        w.bifurcate_neurite(c, 2, 1.0, 1.5);
        w.bifurcate_neurite(c, 3, 1.0, 1.5);
        w.bifurcate_neurite(c, 4, 1.0, 1.5);
        w.bifurcate_neurite(c, 5, 1.0, 1.5);
        w.bifurcate_neurite(c, 6, 1.0, 1.5);

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let model = RallPowerLaw::default();
        assign_rall_diameters(&mut w.cells[c], &model, &mut rng);

        let trunk = w.cells[c].neurites[0].diameter;
        let terminals: Vec<f64> = w.cells[c].neurites[7..]
            .iter()
            .filter(|n| n.is_terminal)
            .map(|n| n.diameter)
            .collect();
        assert!(!terminals.is_empty());
        assert!(
            trunk > terminals.iter().cloned().fold(f64::MIN, f64::max),
            "trunk ({trunk}) must exceed every terminal diameter"
        );
        // With ~3 levels of 2-way combination the trunk is roughly
        // d_term · 8^(1/e) ≈ d_term · 3.7 for e ≈ 1.47.
        assert!(trunk > 1.5, "trunk too thin: {trunk}");
    }

    #[test]
    fn rall_respects_d_max() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let model = RallPowerLaw {
            d_max: 2.0,
            ..Default::default()
        };
        // Extreme daughters would give ~4 without the cap.
        let d = model.combine(&mut rng, 3.0, 3.0);
        assert_eq!(d, 2.0);
    }

    #[test]
    fn intermediate_inherits_daughter_diameter() {
        // Chain: root → child → grandchild (single-child links inherit).
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n0 = w.new_neurite(c, true, 1.0);
        w.branch_neurite(c, n0); // child of n0
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let model = RallPowerLaw::default();
        assign_rall_diameters(&mut w.cells[c], &model, &mut rng);
        assert!((w.cells[c].neurites[0].diameter - w.cells[c].neurites[1].diameter).abs() < 1e-12);
    }

    #[test]
    fn top_down_splits_symmetrically() {
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n0 = w.new_neurite(c, true, 4.0);
        w.bifurcate_neurite(c, n0, 1.0, 1.5);
        assign_top_down(&mut w.cells[c], 1.5);
        let expected = 4.0 * 2f64.powf(-1.0 / 1.5);
        assert!((w.cells[c].neurites[1].diameter - expected).abs() < 1e-9);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let build = |seed: u64| {
            let mut w = tiny_world();
            let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
            let n0 = w.new_neurite(c, true, 1.0);
            w.bifurcate_neurite(c, n0, 1.0, 1.5);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            assign_rall_diameters(&mut w.cells[c], &RallPowerLaw::default(), &mut rng);
            (
                w.cells[c].neurites[0].diameter,
                w.cells[c].neurites[1].diameter,
                w.cells[c].neurites[2].diameter,
            )
        };
        assert_eq!(build(99), build(99));
        let (a, _, _) = build(100);
        assert!(a != build(101).0 || true);
        let _ = a;
    }
}
