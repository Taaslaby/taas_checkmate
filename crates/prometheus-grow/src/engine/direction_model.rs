//! NETMORPH axon direction models (`axon_direction_model.cc/.hh`).
//!
//! The **tension direction model** is the default growth-cone steering of
//! NETMORPH and the single biggest determinant of realistic arbor geometry.
//!
//! Hypothesis (Koene et al.): each segment in the terminal's history pulls
//! the growth cone toward itself with an amplitude analogous to gravitation:
//!
//! ```text
//! pull_i = mass_i / dist_i^history_power
//! mass_i = len_i · π · r_i²        (segment "mass")
//! dist_i = fibre distance from the segment midpoint to the cone tip
//! predicted = Σᵢ pull_i · unit_dir_i   (then normalized)
//! ```
//!
//! The new step direction is the normalized pull sum, optionally perturbed
//! by a veer angle drawn uniformly in `[veer_min, veer_max]` around it.
//! Segments far along the fibre path contribute little (distance falloff),
//! which yields the characteristic straight trunks with smoothly curving
//! distal arborization.
//!
//! The **radial direction model** steers back toward a radial direction from
//! the soma (optionally the Samsonovich hypothesis: return to the initial
//! radial bearing). Ported as [`RadialSteering`] for use alongside or
//! instead of tension.

use crate::cell::Neurite;
use crate::math::Vec3;
use rand::Rng;

/// Geometry snapshot of a neurite for direction computation — decouples the
/// steering math from the world borrow (needed because the RNG lives inside
/// [`crate::World`]).
pub struct GeometryView {
    pub points: Vec<Vec3>,
    pub diameter: f64,
    /// Direction to use when there is no pull history (single-point polyline).
    pub axis_fallback: Vec3,
}

/// Parameters of the tension direction model
/// (NETMORPH `tension_dm_parameters`, default `history_power = 2.0`).
#[derive(Clone, Copy, Debug)]
pub struct TensionSteering {
    /// Exponent on the along-fibre distance in the gravitational analogy.
    pub history_power: f64,
    /// Uniform veer-angle bounds in radians applied around the predicted
    /// direction (`veeranglemin/max`; 0 = deterministic).
    pub veer_min: f64,
    pub veer_max: f64,
}

impl Default for TensionSteering {
    fn default() -> Self {
        TensionSteering {
            history_power: 2.0,
            veer_min: 0.0,
            veer_max: 0.0,
        }
    }
}

impl TensionSteering {
    /// Weighted pull-sum over one neurite's polyline history, treating the
    /// whole polyline as the segment history (NETMORPH tracks per-segment
    /// histories; the polyline carries the same information).
    ///
    /// Returns the un-normalized predicted direction.
    pub fn predict(&self, neurite: &Neurite) -> Vec3 {
        let pts = &neurite.points;
        let n = pts.len();
        if n < 2 {
            return Vec3::ZERO;
        }
        let tip = pts[n - 1];
        let mut acc = Vec3::ZERO;
        let mut distance_by_fibre = 0.0;
        // Walk the history from the oldest segment to the tip.
        for w in pts.windows(2) {
            let seg = w[1] - w[0];
            let l = seg.len();
            if l <= 0.0 {
                continue;
            }
            let radius = (neurite.diameter * 0.5).max(1e-9);
            let mass = l * std::f64::consts::PI * radius * radius;
            let distance = distance_by_fibre + l * 0.5;
            let pull = mass / distance.powf(self.history_power);
            acc = acc + seg.normalized() * pull;
            distance_by_fibre += l;
        }
        let _ = tip;
        acc
    }

    /// Compute the next step direction for a growing terminal: the normalized
    /// pull plus a uniform random veer rotation about a random perpendicular
    /// axis. Deterministic given `rng`.
    pub fn direction(&self, neurite: &Neurite, rng: &mut impl Rng) -> Vec3 {
        let predicted = self.predict(neurite);
        if predicted.len() <= 1e-12 {
            // No history pull (single-point or degenerate) — keep going straight.
            return neurite.axis();
        }
        let dir = predicted.normalized();
        let veer = if self.veer_max > self.veer_min {
            rng.gen_range(self.veer_min..self.veer_max)
        } else {
            0.0
        };
        if veer.abs() <= 1e-12 {
            return dir;
        }
        // Rotate by the veer angle about a random axis perpendicular to dir.
        let any = if dir.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let perp = dir.cross(any).normalized();
        let angle = rng.gen::<f64>() * std::f64::consts::TAU;
        let tilt = dir * angle.cos() + perp * angle.sin();
        crate::math::rotate_vec(dir, tilt, veer).normalized()
    }

    /// Direction from a geometry snapshot — for callers that cannot hold a
    /// `&Neurite` across an RNG draw (e.g. inside [`crate::World`]). The veer
    /// perturbation (NETMORPH `veeranglemin/max`) is drawn from the world RNG.
    pub fn direction_view(&self, g: &GeometryView, world: &mut crate::world::World) -> Vec3 {
        let predicted = self.predict_view(g);
        if predicted.len() <= 1e-12 {
            return g.axis_fallback;
        }
        let dir = predicted.normalized();
        let veer = if self.veer_max > self.veer_min {
            world.rand_range(self.veer_min, self.veer_max)
        } else {
            0.0
        };
        if veer.abs() <= 1e-12 {
            return dir;
        }
        // Rotate by the veer angle about a random axis perpendicular to dir.
        let any = if dir.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let perp = dir.cross(any).normalized();
        let angle = world.rand_angle();
        let tilt = dir * angle.cos() + perp * angle.sin();
        crate::math::rotate_vec(dir, tilt, veer).normalized()
    }

    /// Pull-sum over a geometry snapshot.
    pub fn predict_view(&self, g: &GeometryView) -> Vec3 {
        let pts = &g.points;
        let n = pts.len();
        if n < 2 {
            return Vec3::ZERO;
        }
        let mut acc = Vec3::ZERO;
        let mut distance_by_fibre = 0.0;
        for w in pts.windows(2) {
            let seg = w[1] - w[0];
            let l = seg.len();
            if l <= 0.0 {
                continue;
            }
            let radius = (g.diameter * 0.5).max(1e-9);
            let mass = l * std::f64::consts::PI * radius * radius;
            let distance = distance_by_fibre + l * 0.5;
            let pull = mass / distance.powf(self.history_power);
            acc = acc + seg.normalized() * pull;
            distance_by_fibre += l;
        }
        acc
    }
}

/// Radial direction model (NETMORPH `radial_direction_model`). Steers the
/// growth cone to maintain a radial bearing from the soma; with the
/// Samsonovich hypothesis the target bearing is fixed at branch time rather
/// than continuously recomputed — approximated here by keeping the neurite's
/// initial (root) direction.
#[derive(Clone, Copy, Debug, Default)]
pub struct RadialSteering {
    /// Samsonovich hypothesis: prefer the original radial bearing.
    pub samsonovich: bool,
}

impl RadialSteering {
    /// Predicted direction: from the soma toward the current tip (or the
    /// root direction under Samsonovich), i.e. grow outward radially.
    pub fn direction(&self, cell_pos: Vec3, neurite: &Neurite) -> Vec3 {
        if self.samsonovich && neurite.points.len() >= 2 {
            let root_dir = neurite.points[1] - neurite.points[0];
            if root_dir.len() > 1e-12 {
                return root_dir.normalized();
            }
        }
        let radial = neurite.tip() - cell_pos;
        if radial.len() <= 1e-12 {
            neurite.axis()
        } else {
            radial.normalized()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn straight_neurite(len: usize) -> Neurite {
        Neurite {
            is_axon: true,
            diameter: 1.0,
            points: (0..len).map(|i| Vec3::new(i as f64, 0.0, 0.0)).collect(),
            modules: vec![],
            excrescences: vec![],
            branch_order: 0,
            parent: None,
            elongation_speed: 1.0,
            is_terminal: true,
            intracellular: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn straight_history_keeps_straight() {
        // A perfectly straight history has zero transverse pull → the
        // predicted direction equals the last segment direction.
        let n = straight_neurite(10);
        let t = TensionSteering::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let d = t.direction(&n, &mut rng);
        assert!((d - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn curved_history_predicts_curvature() {
        // History bends +Y near the tip → prediction tilts toward +Y.
        let mut n = straight_neurite(10);
        let last = n.points.len() - 1;
        n.points[last] = Vec3::new(9.0, 5.0, 0.0);
        let t = TensionSteering::default();
        let p = t.predict(&n);
        assert!(p.y > 0.0, "expected +Y pull, got {}", p.y);
    }

    #[test]
    fn heavier_recent_segments_pull_more() {
        // The gravitational analogy weights by mass (∝ segment length): a
        // longer sideways segment at the tip tilts the prediction more than
        // a short one.
        let short = {
            let mut n = straight_neurite(11);
            let l = n.points.len() - 1;
            n.points[l] = Vec3::new(10.0, 1.0, 0.0);
            n
        };
        let long = {
            let mut n = straight_neurite(11);
            let l = n.points.len() - 1;
            n.points[l] = Vec3::new(14.0, 4.0, 0.0);
            n
        };
        let t = TensionSteering::default();
        let p_short = t.predict(&short);
        let p_long = t.predict(&long);
        assert!(
            p_long.y / p_long.x > p_short.y / p_short.x,
            "longer tip segment must pull harder: {} vs {}",
            p_long.y / p_long.x,
            p_short.y / p_short.x
        );
    }

    #[test]
    fn distant_history_contributes_less() {
        // Truncating distant history barely changes the prediction at high
        // history_power (distance falloff), demonstrating the decay.
        let mut n = straight_neurite(12);
        let l = n.points.len() - 1;
        n.points[l] = Vec3::new(11.0, 1.0, 0.0);
        let t = TensionSteering {
            history_power: 4.0,
            ..Default::default()
        };
        let full = t.predict(&n);
        n.points.drain(..6); // drop the oldest half of the history
        let tail = t.predict(&n);
        // Both nearly aligned with +X; the tail-only prediction is close to full.
        assert!((full.y / full.x - tail.y / tail.x).abs() < 1e-2);
    }

    #[test]
    fn degenerate_neurite_falls_back_to_axis() {
        let mut n = straight_neurite(1);
        n.points = vec![Vec3::new(0.0, 0.0, 0.0)];
        let t = TensionSteering::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let d = t.direction(&n, &mut rng);
        assert_eq!(d, n.axis());
    }

    #[test]
    fn veer_stays_unit_length() {
        let n = straight_neurite(6);
        let t = TensionSteering {
            history_power: 2.0,
            veer_min: 0.0,
            veer_max: 0.8,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..50 {
            let d = t.direction(&n, &mut rng);
            assert!((d.len() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn radial_points_outward() {
        let mut n = straight_neurite(5);
        n.points = vec![
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(12.0, 0.0, 0.0),
            Vec3::new(14.0, 0.0, 0.0),
        ];
        let r = RadialSteering::default();
        let d = r.direction(Vec3::ZERO, &n);
        assert!((d - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn samsonovich_keeps_root_bearing() {
        let mut n = straight_neurite(5);
        n.points = vec![
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 1.0, 0.0),
            Vec3::new(10.0, 2.0, 0.0),
        ];
        let r = RadialSteering { samsonovich: true };
        // Root bearing is +Y even though the soma sits at origin.
        let d = r.direction(Vec3::ZERO, &n);
        assert!((d - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-9);
    }
}
