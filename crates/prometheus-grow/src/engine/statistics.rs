//! NETMORPH network-generated statistics (`Network_Generated_Statistics.cc/.hh`
//! + `fibre_structure.cc: Collect_Data`).
//!
//! Collects the standard morphometrics used to validate grown arbors against
//! experimental neuroanatomy (van Pelt & Uylings; Koene et al.):
//!
//! - per-arbor total length and terminal-segment count
//! - terminal / intermediate segment lengths
//! - lengths between successive bifurcations, terminal length since the
//!   last bifurcation, and since the soma
//! - branch angles at bifurcations (angle between daughter directions)
//! - cartesian-to-fiber ratio: how straight a path is — the euclidean
//!   displacement divided by the path length (1.0 = perfectly straight)
//!
//! Statistics are gathered as raw samples into [`SampleStats`] (n, mean,
//! std, min, max), split by axon vs dendrite, mirroring the NETMORPH
//! `stat_labels` enum.
//!
//! The traversal mirrors `fibre_segment::Collect_Data`: walk each neurite's
//! polyline tree depth-first, accumulating length since the soma and since
//! the last bifurcation, resetting the latter at every bifurcation.

use crate::cell::Cell;
use crate::math::Vec3;

/// Running statistics over collected samples (NETMORPH
/// `network_statistics_data`: n, mean, std, min, max).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SampleStats {
    pub n: usize,
    pub sum: f64,
    pub sum_sq: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    pub fn new() -> Self {
        SampleStats {
            n: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    pub fn collect(&mut self, d: f64) {
        self.n += 1;
        self.sum += d;
        self.sum_sq += d * d;
        if d < self.min {
            self.min = d;
        }
        if d > self.max {
            self.max = d;
        }
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    /// Population standard deviation (as in NETMORPH's SumofSquares/mean).
    pub fn std(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            let m = self.mean();
            ((self.sum_sq / self.n as f64) - m * m).max(0.0).sqrt()
        }
    }

    pub fn min(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.min
        }
    }

    pub fn max(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.max
        }
    }
}

/// Full morphometric summary of one cell's neurite forest, split by
/// axon/dendrite (NETMORPH collects axons and dendrites separately).
#[derive(Clone, Debug, Default)]
pub struct ArborStats {
    pub axon: FiberStats,
    pub dendrite: FiberStats,
}

#[derive(Clone, Debug, Default)]
pub struct FiberStats {
    /// Total fiber length across all arbors of this kind.
    pub arbor_length: SampleStats,
    /// Number of terminal segments per arbor.
    pub terminal_segments: SampleStats,
    /// Individual segment lengths (all segments).
    pub segment_len: SampleStats,
    /// Terminal segment lengths.
    pub term_seg_len: SampleStats,
    /// Intermediate (between-bifurcation) segment lengths.
    pub interseg_len: SampleStats,
    /// Path length between successive bifurcations.
    pub len_between_bifurcations: SampleStats,
    /// Terminal path length since the last bifurcation.
    pub term_len_since_bifurcation: SampleStats,
    /// Terminal path length since the soma.
    pub term_len_since_soma: SampleStats,
    /// Angle between daughter directions at bifurcations (radians).
    pub branch_angles: SampleStats,
    /// Cartesian-to-fiber ratio between bifurcations.
    pub cartratio_between_bifurcations: SampleStats,
    /// Cartesian-to-fiber ratio from last bifurcation to terminal.
    pub cartratio_bifurcation_to_term: SampleStats,
    /// Cartesian-to-fiber ratio from soma to terminal.
    pub cartratio_soma_to_term: SampleStats,
}

impl FiberStats {
    fn collect_arbor(&mut self, total_len: f64, terminals: usize) {
        self.arbor_length.collect(total_len);
        self.terminal_segments.collect(terminals as f64);
    }
}

/// Angle between the directions of two daughter branches at their common
/// origin (the bifurcation point).
fn branch_angle(d1: Vec3, d2: Vec3) -> f64 {
    let dot = d1.normalized().dot(d2.normalized()).clamp(-1.0, 1.0);
    dot.acos()
}

/// Cartesian-to-fiber ratio for a path: euclidean displacement from the
/// reference point to the current point divided by the along-fibre length.
/// Returns 1.0 for degenerate zero-length paths (as in NETMORPH).
fn cart_fib_ratio(from: Vec3, to: Vec3, fibre_len: f64, subtract: f64) -> f64 {
    if fibre_len <= 0.0 {
        return 1.0;
    }
    let mut d = from.dist(to);
    d -= subtract;
    d / fibre_len
}

/// Per-path accumulator mirroring NETMORPH's `fibre_collected_data`.
struct PathState {
    len_since_soma: f64,
    len_since_bifurcation: f64,
    ref_point: Vec3,
}

/// Collect morphometrics for one cell by traversing its neurite forest.
///
/// Each neurite with `parent == None` is an arbor root; children are found
/// via the `parent` index chain. Bifurcations are neurites with exactly two
/// children (NETMORPH binary trees); single-child links are intermediate
/// continuations.
pub fn collect_cell_stats(cell: &Cell) -> ArborStats {
    let mut stats = ArborStats::default();
    let roots: Vec<usize> = cell
        .neurites
        .iter()
        .enumerate()
        .filter(|(_, n)| n.parent.is_none())
        .map(|(i, _)| i)
        .collect();

    for root in roots {
        let is_axon = cell.neurites[root].is_axon;
        let fs = if is_axon {
            &mut stats.axon
        } else {
            &mut stats.dendrite
        };
        let mut arbor_len = 0.0;
        let mut terminals = 0usize;
        let soma_pos = cell.soma.pos;
        let start = cell.neurites[root].points[0];
        let state = PathState {
            len_since_soma: 0.0,
            len_since_bifurcation: 0.0,
            ref_point: start,
        };
        walk(
            cell,
            root,
            state,
            soma_pos,
            fs,
            &mut arbor_len,
            &mut terminals,
        );
        fs.collect_arbor(arbor_len, terminals);
    }
    stats
}

/// Recursively walk a neurite subtree, collecting segment statistics.
///
/// A neurite's polyline contributes all its internal segments; the handoff
/// to a child continues the path from this neurite's tip.
#[allow(clippy::too_many_arguments)]
fn walk(
    cell: &Cell,
    idx: usize,
    mut state: PathState,
    soma_pos: Vec3,
    fs: &mut FiberStats,
    arbor_len: &mut f64,
    terminals: &mut usize,
) {
    let neurite = &cell.neurites[idx];
    let pts = &neurite.points;
    let children: Vec<usize> = cell
        .neurites
        .iter()
        .enumerate()
        .filter(|&(_, n)| n.parent == Some(idx))
        .map(|(i, _)| i)
        .collect();

    // Accumulate this neurite's polyline segments.
    for w in pts.windows(2) {
        let seg_len = w[0].dist(w[1]);
        state.len_since_soma += seg_len;
        state.len_since_bifurcation += seg_len;
        *arbor_len += seg_len;
        fs.segment_len.collect(seg_len);

        let is_last = std::ptr::eq(w.last().unwrap(), pts.last().unwrap());
        if is_last && !children.is_empty() {
            // This segment ends where children branch off.
            continue;
        }
        if children.is_empty() && is_last {
            // Terminal segment of a terminal neurite.
            fs.term_seg_len.collect(seg_len);
            fs.term_len_since_bifurcation
                .collect(state.len_since_bifurcation);
            fs.term_len_since_soma.collect(state.len_since_soma);
            fs.cartratio_bifurcation_to_term.collect(cart_fib_ratio(
                state.ref_point,
                pts[pts.len() - 1],
                state.len_since_bifurcation,
                0.0,
            ));
            fs.cartratio_soma_to_term.collect(cart_fib_ratio(
                soma_pos,
                pts[pts.len() - 1],
                state.len_since_soma,
                0.0,
            ));
            *terminals += 1;
        } else {
            fs.interseg_len.collect(seg_len);
        }
    }

    match children.len() {
        0 => {}
        1 => {
            // Continuation: pass through without resetting bifurcation state.
            let child = children[0];
            walk(cell, child, state, soma_pos, fs, arbor_len, terminals);
        }
        _ => {
            // Bifurcation (or multi-split): record inter-bifurcation stats,
            // then reset the per-path accumulators for each daughter.
            fs.len_between_bifurcations
                .collect(state.len_since_bifurcation);
            fs.cartratio_between_bifurcations.collect(cart_fib_ratio(
                state.ref_point,
                neurite.tip(),
                state.len_since_bifurcation,
                0.0,
            ));
            let tip = neurite.tip();
            for &child in &children {
                let cn = &cell.neurites[child];
                if cn.points.len() >= 2 {
                    let d_parent = (tip - second_to_last(pts)).normalized();
                    let d_child = (cn.points[1] - cn.points[0]).normalized();
                    fs.branch_angles.collect(branch_angle(d_parent, d_child));
                }
                let child_state = PathState {
                    len_since_soma: state.len_since_soma,
                    len_since_bifurcation: 0.0,
                    ref_point: tip,
                };
                walk(cell, child, child_state, soma_pos, fs, arbor_len, terminals);
            }
        }
    }
}

fn second_to_last(pts: &[Vec3]) -> Vec3 {
    if pts.len() >= 2 {
        pts[pts.len() - 2]
    } else {
        pts[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use crate::Bounds;
    use crate::Grid;
    use crate::World;

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
    fn sample_stats_math() {
        let mut s = SampleStats::new();
        for v in [1.0, 2.0, 3.0] {
            s.collect(v);
        }
        assert_eq!(s.n, 3);
        assert!((s.mean() - 2.0).abs() < 1e-12);
        assert!((s.std() - (2.0f64 / 3.0).sqrt()).abs() < 1e-12); // population std
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.max(), 3.0);
    }

    #[test]
    fn empty_stats_are_zeroed() {
        let s = SampleStats::new();
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.std(), 0.0);
        assert_eq!(s.min(), 0.0);
        assert_eq!(s.max(), 0.0);
    }

    #[test]
    fn straight_single_neurite_is_perfectly_straight() {
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n = w.new_neurite(c, true, 1.0);
        for _ in 0..10 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        let stats = collect_cell_stats(&w.cells[c]);
        assert_eq!(stats.axon.terminal_segments.n, 1);
        assert!((stats.axon.arbor_length.mean() - 10.0).abs() < 1e-9);
        assert_eq!(stats.axon.branch_angles.n, 0, "no bifurcations");
        // The single terminal path runs from the soma to the tip.
        let cr = stats.axon.cartratio_soma_to_term.mean();
        assert!(
            (cr - 1.0).abs() < 0.15,
            "straight line should have ratio ≈1: {cr}"
        );
    }

    #[test]
    fn bifurcation_records_angle_and_lengths() {
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n = w.new_neurite(c, true, 1.0);
        for _ in 0..4 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        // 90° bifurcation: daughters go +Y and +Z.
        w.bifurcate_neurite_asym(
            c,
            n,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            1.5,
        );
        let stats = collect_cell_stats(&w.cells[c]);
        // One angle sample per daughter (parent→child convention).
        assert_eq!(stats.axon.branch_angles.n, 2);
        assert!(
            (stats.axon.branch_angles.mean() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "expected ~90° angle, got {}",
            stats.axon.branch_angles.mean().to_degrees()
        );
        // Two daughter terminals after one bifurcation.
        assert_eq!(stats.axon.terminal_segments.max(), 2.0);
        // Length before the bifurcation was 4.
        assert!((stats.axon.len_between_bifurcations.mean() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn axon_and_dendrite_collected_separately() {
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let a = w.new_neurite(c, true, 1.0);
        let d = w.new_neurite(c, false, 1.0);
        w.extend_neurite(c, a, Vec3::new(1.0, 0.0, 0.0));
        w.extend_neurite(c, d, Vec3::new(0.0, 1.0, 0.0));
        let stats = collect_cell_stats(&w.cells[c]);
        assert_eq!(stats.axon.arbor_length.n, 1);
        assert_eq!(stats.dendrite.arbor_length.n, 1);
        assert!((stats.axon.arbor_length.mean() - 1.0).abs() < 1e-9);
        assert!((stats.dendrite.arbor_length.mean() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn curved_path_has_ratio_below_one() {
        let mut w = tiny_world();
        let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
        let n = w.new_neurite(c, true, 1.0);
        // L-shaped path: 5 units +X then 5 units +Y → displacement √50 ≈ 7.07,
        // fiber length 10 → ratio ≈ 0.707.
        for _ in 0..5 {
            w.extend_neurite(c, n, Vec3::new(1.0, 0.0, 0.0));
        }
        for _ in 0..5 {
            w.extend_neurite(c, n, Vec3::new(0.0, 1.0, 0.0));
        }
        let stats = collect_cell_stats(&w.cells[c]);
        let cr = stats.axon.cartratio_soma_to_term.mean();
        let expected = (50.0f64).sqrt() / 10.0;
        assert!(
            (cr - expected).abs() < 0.05,
            "L-path ratio should be ≈{expected}: {cr}"
        );
    }

    #[test]
    fn deterministic_collection() {
        let build = || {
            let mut w = tiny_world();
            let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
            let n = w.new_neurite(c, true, 1.0);
            w.bifurcate_neurite(c, n, 1.0, 1.5);
            collect_cell_stats(&w.cells[c])
        };
        let a = build();
        let b = build();
        assert_eq!(a.axon.branch_angles, b.axon.branch_angles);
    }
}
