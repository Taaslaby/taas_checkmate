//! BioDynaMo cell mechanics (adhesion + repulsion + elastic).
//!
//!   F = F_adhesion + F_repulsion + F_elastic
//!
//! - **Adhesion**: cells of compatible types attract (e.g. cadherin bonds).
//!   Force proportional to overlap of adhesive zones.
//!
//! - **Repulsion**: cells repel when their somas overlap (volume exclusion).
//!   Soft Hertz-like force: F ~ (radii_sum − distance) / radii_sum.
//!
//! - **Elastic**: cells connected by junctions (synapses, gap junctions)
//!   experience spring forces pulling them toward equilibrium distance.
//!
//! All forces are scaled by dt and clamped to prevent explosion at tiny distances.

use crate::Vec3;

/// ─── Parameters ──────────────────────────────────────────────────────────

/// Mechanical parameters for the simulation.
#[derive(Clone, Debug)]
pub struct MechanicsParams {
    /// Strength of cell-cell adhesion (cadherin-like).
    pub adhesion_strength: f64,
    /// Strength of volume-exclusion repulsion.
    pub repulsion_strength: f64,
    /// Strength of elastic spring forces between connected cells.
    pub elastic_strength: f64,
    /// Equilibrium distance for elastic springs (fraction of soma radii sum).
    pub elastic_equilibrium: f64,
    /// Maximum force magnitude per axis (prevents explosion).
    pub max_force: f64,
    /// Damping factor (velocity *= 1 − damping each step).
    pub damping: f64,
    /// Cutoff distance multiplier — ignore interactions beyond adhesion_cutoff × (r1+r2).
    pub adhesion_cutoff: f64,
    /// Adhesion matrix: adhesion_matrix[type_a][type_b] → strength multiplier (0..1).
    pub adhesion_matrix: [[f64; 3]; 3],
    /// Whether to apply mechanics (skip if false).
    pub enabled: bool,
}

impl Default for MechanicsParams {
    fn default() -> Self {
        Self {
            adhesion_strength: 0.5,
            repulsion_strength: 2.0,
            elastic_strength: 0.3,
            elastic_equilibrium: 0.8,
            max_force: 5.0,
            damping: 0.1,
            adhesion_cutoff: 2.5,
            // Excitatory↔Excitatory adhere strongest (like-to-like via cadherins),
            // inhibitory↔inhibitory moderate, cross-type = none.
            adhesion_matrix: [
                [1.0, 0.0, 0.0], // Exc → Exc only
                [0.0, 0.6, 0.0], // Inh → Inh only
                [0.0, 0.0, 0.0], // Glia → nothing
            ],
            enabled: true,
        }
    }
}

/// Per-cell velocity (persistent across steps).
#[derive(Clone, Debug)]
pub struct CellVelocity {
    pub v: Vec3,
}

impl Default for CellVelocity {
    fn default() -> Self {
        Self { v: Vec3::ZERO }
    }
}

/// ─── Force computation ───────────────────────────────────────────────────

/// Compute the total mechanical force on a cell at `pos` with `radius`.
///
/// Returns the displacement vector for this step (force × dt, clamped).
pub fn compute_mechanics(
    pos: Vec3,
    radius: f64,
    cell_type: usize,
    all_positions: &[Vec3],
    all_radii: &[f64],
    all_types: &[usize],
    skip_idx: usize,
    edges: &[(usize, usize)],
    params: &MechanicsParams,
    dt: f64,
) -> Vec3 {
    if !params.enabled {
        return Vec3::ZERO;
    }

    let mut total_force = Vec3::ZERO;
    let n = all_positions.len();

    for j in 0..n {
        if j == skip_idx {
            continue;
        }

        let r_sum = radius + all_radii[j];
        let dir = pos - all_positions[j];
        let dist = dir.len();

        if dist < 1e-9 {
            // Cells exactly overlap — push apart in random direction.
            let tiny = Vec3::new(0.01, 0.0, 0.0);
            total_force = total_force + tiny * params.repulsion_strength * params.max_force;
            continue;
        }

        let dir_norm = dir * (1.0 / dist);
        let overlap = r_sum - dist;

        // ── Repulsion: soft Hertz-like when somas overlap ──
        if overlap > 0.0 {
            let f_mag = params.repulsion_strength * overlap / r_sum;
            total_force = total_force + dir_norm * f_mag;
        }

        // ── Adhesion: only within cutoff range, type-gated ──
        let adh = params.adhesion_matrix[cell_type][all_types[j]];
        if adh > 0.0 && dist < params.adhesion_cutoff * r_sum && overlap <= 0.0 {
            // Adhesion pulls cells together (negative force along dir).
            let f_mag =
                params.adhesion_strength * adh * (1.0 - dist / (params.adhesion_cutoff * r_sum));
            total_force = total_force - dir_norm * f_mag; // −dir = toward the partner
        }
    }

    // ── Elastic: spring forces from connected cells ──
    for &(pre, post) in edges {
        let partner = if pre == skip_idx {
            post
        } else if post == skip_idx {
            pre
        } else {
            continue;
        };
        if partner >= n {
            continue;
        }

        let r_sum = radius + all_radii[partner];
        let eq_dist = params.elastic_equilibrium * r_sum;
        let dir = all_positions[partner] - pos;
        let dist = dir.len();

        if dist < 1e-9 {
            continue;
        }

        let dir_norm = dir * (1.0 / dist);
        let strain = dist - eq_dist;
        let f_mag = params.elastic_strength * strain;
        total_force = total_force + dir_norm * f_mag;
    }

    // Clamp and scale by dt.
    let mag = total_force.len();
    if mag > params.max_force {
        total_force = total_force * (params.max_force / mag);
    }

    total_force * dt
}

/// Apply mechanics to all cells in-place (moves somas). No-op if disabled.
pub fn step_mechanics(
    positions: &mut [Vec3],
    radii: &[f64],
    types: &[usize],
    edges: &[(usize, usize)],
    velocities: &mut [CellVelocity],
    params: &MechanicsParams,
    dt: f64,
) {
    if !params.enabled || positions.is_empty() {
        return;
    }

    let n = positions.len();
    // Compute forces for each cell.
    let mut forces = vec![Vec3::ZERO; n];
    for i in 0..n {
        forces[i] = compute_mechanics(
            positions[i],
            radii[i],
            types[i],
            positions,
            radii,
            types,
            i,
            edges,
            params,
            dt,
        );
    }

    // Apply with velocity and damping.
    for i in 0..n {
        // Semi-implicit Euler: v += F·dt/m (mass ∝ r³), then pos += v·dt.
        let mass = radii[i].powi(3).max(0.001);
        velocities[i].v = velocities[i].v + forces[i] * (1.0 / mass);
        // Damping.
        velocities[i].v = velocities[i].v * (1.0 - params.damping);
        // Clamp velocity.
        let vmag = velocities[i].v.len();
        if vmag > params.max_force {
            velocities[i].v = velocities[i].v * (params.max_force / vmag);
        }
        positions[i] = positions[i] + velocities[i].v;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repulsion_pushes_overlapping_cells_apart() {
        let params = MechanicsParams::default();
        let pos = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let rad = vec![3.0, 3.0];
        let typ = vec![0usize, 0usize];
        let edges = vec![];
        let f = compute_mechanics(
            pos[0], rad[0], typ[0], &pos, &rad, &typ, 0, &edges, &params, 1.0,
        );
        // Overlap = 6 − 1 = 5, repulsion pushes cell[0] right (away from cell[1]).
        // dir = (0−1, 0−0) = (−1, 0), normalized = (−1, 0).
        // repulsion_strength=2.0 * overlap/r_sum = 2.0*5/6 ≈ 1.67
        // force ≈ (−1,0) * 1.67 = (−1.67, 0)
        // But since dir = pos − partner, dir_norm = (−1, 0)
        // Wait: overlap > 0 → total_force += dir_norm * f_mag
        // dir = pos[0] − pos[1] = (−1, 0)
        // dir_norm = (−1, 0)
        // So force pushes cell[0] LEFT — away from cell[1] (which is to its right).
        // That's correct! cell[0] is at x=0, cell[1] at x=1, cell[0] should be pushed LEFT.
        assert!(
            f.x < -0.1,
            "repulsion должен толкать клетки врозь: fx={}",
            f.x
        );
        assert!(f.y.abs() < 0.01, "нет поперечной силы: fy={}", f.y);
    }

    #[test]
    fn adhesion_pulls_nearby_same_type_cells() {
        let mut params = MechanicsParams::default();
        params.repulsion_strength = 0.0; // disable repulsion
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 0.0, 0.0)];
        let rad = vec![3.0, 3.0];
        let typ = vec![0usize, 0usize]; // both excitatory → adhesion_matrix=1.0
        let edges = vec![];
        let f = compute_mechanics(
            pos[0], rad[0], typ[0], &pos, &rad, &typ, 0, &edges, &params, 1.0,
        );
        // dist=8, r_sum=6, adhesion_cutoff*r_sum=2.5*6=15 → inside adhesion range
        // dir = pos[0] − pos[1] = (−8, 0), dir_norm = (−1, 0)
        // adhesion pulls toward partner: total_force −= dir_norm * f_mag
        // = −(−1,0) * f_mag = (+1, 0) * f_mag — toward cell[1]
        assert!(f.x > 0.01, "адгезия должна притягивать: fx={}", f.x);
        assert!(f.y.abs() < 0.01);
    }

    #[test]
    fn no_adhesion_cross_type_by_default() {
        let mut params = MechanicsParams::default();
        params.repulsion_strength = 0.0;
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(7.0, 0.0, 0.0)];
        let rad = vec![3.0, 3.0];
        let typ = vec![0usize, 1usize]; // exc + inh → matrix[0][1] = 0.0
        let edges = vec![];
        let f = compute_mechanics(
            pos[0], rad[0], typ[0], &pos, &rad, &typ, 0, &edges, &params, 1.0,
        );
        // overlap < 0 (dist=7, r_sum=6) → no repulsion.
        // adhesion_matrix[0][1]=0 → no adhesion.
        // Should be ~zero.
        assert!(f.len() < 0.01, "разные типы не склеиваются: f={:?}", f);
    }

    #[test]
    fn elastic_pulls_toward_equilibrium() {
        let params = MechanicsParams::default();
        // equilibrium = 0.8 * 6 = 4.8
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let rad = vec![3.0, 3.0];
        let typ = vec![0usize, 0usize];
        let edges = vec![(0usize, 1usize)];
        let f = compute_mechanics(
            pos[0], rad[0], typ[0], &pos, &rad, &typ, 0, &edges, &params, 1.0,
        );
        // dist=10, eq=4.8, strain=5.2
        // dir = partner − pos = (10,0), dir_norm = (1,0)
        // elastic: f = dir_norm * elastic_strength(0.3) * strain(5.2) = (1.56, 0)
        assert!(f.x > 0.1, "эластик тянет к равновесию: fx={}", f.x);
    }

    #[test]
    fn step_mechanics_moves_cells() {
        let params = MechanicsParams::default();
        let mut pos = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let rad = vec![3.0, 3.0];
        let typ = vec![0usize, 0usize];
        let edges = vec![];
        let mut vel = vec![CellVelocity::default(), CellVelocity::default()];
        step_mechanics(&mut pos, &rad, &typ, &edges, &mut vel, &params, 1.0);
        // Cells should have moved apart.
        let new_dist = pos[0].dist(pos[1]);
        assert!(new_dist > 1.0, "клетки разошлись: dist={}", new_dist);
    }
}
