//! NETMORPH branch-angle models (`branching_models.cc` / `.hh`).
//!
//! The balanced-forces model replaces a fixed bifurcation angle with a
//! parallelogram-of-forces construction: the two daughters leave the parent
//! at angles derived from their elongation "forces" (their initial lengths).
//! The total spread is a random *parallelogram angle* (~90°), split between
//! the daughters by `atan(f2 / f1)` — a fast-growing daughter continues
//! nearly straight while its slower sibling veers off.

use rand::Rng;

/// NETMORPH `balanced_forces_branch_angle_model`.
#[derive(Clone, Copy, Debug)]
pub struct BalancedForces {
    /// Centre of the parallelogram angle distribution (NETMORPH default π/2).
    pub parallelogram_mean: f64,
    /// Width of the parallelogram angle distribution (NETMORPH default 0.5).
    pub parallelogram_std: f64,
    /// Upper bound on the parallelogram angle (NETMORPH default π − 0.1).
    pub max_angle: f64,
    /// Relative perturbation of each daughter angle (NETMORPH `p_BAM` normal
    /// PDF; the base model multiplies each angle by `1 + N(0, σ)`, default
    /// σ = 0.3 as in NETMORPH).
    pub perturb_std: f64,
}

impl BalancedForces {
    /// NETMORPH defaults: parallelogram angle ~ N(π/2, 0.5) capped below π.
    pub fn netmorph_defaults() -> Self {
        BalancedForces {
            parallelogram_mean: std::f64::consts::FRAC_PI_2,
            parallelogram_std: 0.5,
            max_angle: std::f64::consts::PI - 0.1,
            perturb_std: 0.3,
        }
    }

    /// The two daughter deviations from the parent direction (in radians, on
    /// opposite sides of the parent axis). `angle1 + angle2` is the total
    /// bifurcation angle.
    ///
    /// NETMORPH: `force = ts->Length()` (the daughter's initial length). If
    /// both forces are zero it falls back to uniform random forces.
    pub fn angles(&self, force1: f64, force2: f64, rng: &mut impl Rng) -> (f64, f64) {
        let parallelogram = sample_normal(rng, self.parallelogram_mean, self.parallelogram_std)
            .clamp(0.0, self.max_angle);

        let (mut f1, mut f2) = (force1, force2);
        if f1 == 0.0 && f2 == 0.0 {
            f1 = rng.gen::<f64>();
            f2 = rng.gen::<f64>();
        }

        let (a1, a2) = if f1 == 0.0 {
            (parallelogram, 0.0)
        } else if f2 == 0.0 {
            (0.0, parallelogram)
        } else {
            let a1 = (f2 / f1).atan();
            (a1, parallelogram - a1)
        };

        // NETMORPH `p_BAM` pdf has amplitude 1.0, so the multiplier stays in
        // [0, 2] — clamp the normal draw to ±1 to match.
        let p1 = if self.perturb_std > 0.0 {
            1.0 + sample_normal(rng, 0.0, self.perturb_std).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let p2 = if self.perturb_std > 0.0 {
            1.0 + sample_normal(rng, 0.0, self.perturb_std).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        (a1 * p1, a2 * p2)
    }
}

/// Box–Muller normal sample (avoids a `rand_distr` dependency).
fn sample_normal(rng: &mut impl Rng, mean: f64, std: f64) -> f64 {
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    mean + std * z
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn equal_forces_split_parallelogram_symmetrically() {
        let mut rng = StdRng::seed_from_u64(3);
        // Fixed parallelogram (zero width) so the split is deterministic.
        let m = BalancedForces {
            parallelogram_mean: std::f64::consts::FRAC_PI_2,
            parallelogram_std: 0.0,
            max_angle: std::f64::consts::PI - 0.1,
            perturb_std: 0.0,
        };
        let (a1, a2) = m.angles(1.0, 1.0, &mut rng);
        assert!((a1 - a2).abs() < 1e-9, "равные силы → равные углы");
        assert!((a1 + a2 - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn dominant_force_continues_straight() {
        let m = BalancedForces {
            parallelogram_mean: std::f64::consts::FRAC_PI_2,
            parallelogram_std: 0.0,
            max_angle: std::f64::consts::PI - 0.1,
            perturb_std: 0.0,
        };
        let mut rng = StdRng::seed_from_u64(4);
        let (a1, a2) = m.angles(10.0, 1.0, &mut rng);
        assert!(a1 < a2, "быстрая ветвь (f1) отклоняется меньше");
    }

    #[test]
    fn zero_forces_fall_back_to_random() {
        let m = BalancedForces {
            parallelogram_mean: std::f64::consts::FRAC_PI_2,
            parallelogram_std: 0.5,
            max_angle: std::f64::consts::PI - 0.1,
            perturb_std: 0.0,
        };
        let mut rng = StdRng::seed_from_u64(5);
        let (a1, a2) = m.angles(0.0, 0.0, &mut rng);
        assert!(a1 >= 0.0 && a2 >= 0.0);
        assert!((a1 + a2) <= m.max_angle + 1e-9);
    }

    #[test]
    fn default_perturbation_matches_netmorph() {
        // NETMORPH branch_angle_model_base default: normal_pdf(0.0, 0.3, 1.0).
        assert!((BalancedForces::netmorph_defaults().perturb_std - 0.3).abs() < 1e-12);
    }
}
