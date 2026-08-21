//! Extracellular morphogens: a concentration grid with diffusion/degradation
//! plus optional analytic gradients (the dpp/wg/hh analog).
//!
//! CX3D supports both *real* diffusion (over a Delaunay triangulation) and
//! *artificial* gradients (analytic gaussian/linear functions). Here real
//! diffusion runs on a regular grid (explicit Euler Laplacian) and gradients
//! are analytic — enough to reproduce the tutorial behaviour.

use crate::Vec3;

/// Regular 3D grid holding concentration fields.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// World-units per grid cell.
    pub cell: f64,
    pub origin: Vec3,
}

impl Grid {
    pub fn new(nx: usize, ny: usize, nz: usize, cell: f64, origin: Vec3) -> Self {
        Grid {
            nx,
            ny,
            nz,
            cell,
            origin,
        }
    }

    pub fn volume(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub fn flat(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// Trilinear interpolation of `field` at world position `p`.
    /// Clamped to the grid interior (positions outside read the boundary).
    pub fn trilinear(&self, field: &[f64], p: Vec3) -> f64 {
        let gx = (p.x - self.origin.x) / self.cell;
        let gy = (p.y - self.origin.y) / self.cell;
        let gz = (p.z - self.origin.z) / self.cell;

        let i0 = gx.floor().clamp(0.0, (self.nx.saturating_sub(2)) as f64) as usize;
        let j0 = gy.floor().clamp(0.0, (self.ny.saturating_sub(2)) as f64) as usize;
        let k0 = gz.floor().clamp(0.0, (self.nz.saturating_sub(2)) as f64) as usize;

        let fx = (gx - i0 as f64).clamp(0.0, 1.0);
        let fy = (gy - j0 as f64).clamp(0.0, 1.0);
        let fz = (gz - k0 as f64).clamp(0.0, 1.0);

        let c000 = field[self.flat(i0, j0, k0)];
        let c100 = field[self.flat(i0 + 1, j0, k0)];
        let c010 = field[self.flat(i0, j0 + 1, k0)];
        let c110 = field[self.flat(i0 + 1, j0 + 1, k0)];
        let c001 = field[self.flat(i0, j0, k0 + 1)];
        let c101 = field[self.flat(i0 + 1, j0, k0 + 1)];
        let c011 = field[self.flat(i0, j0 + 1, k0 + 1)];
        let c111 = field[self.flat(i0 + 1, j0 + 1, k0 + 1)];

        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// 6-neighbour Laplacian at an interior cell (1..n-1).
    pub fn laplacian(&self, field: &[f64], i: usize, j: usize, k: usize) -> f64 {
        let c = field[self.flat(i, j, k)];
        let s = field[self.flat(i - 1, j, k)]
            + field[self.flat(i + 1, j, k)]
            + field[self.flat(i, j - 1, k)]
            + field[self.flat(i, j + 1, k)]
            + field[self.flat(i, j, k - 1)]
            + field[self.flat(i, j, k + 1)];
        s - 6.0 * c
    }

    /// Nearest grid cell for a world position (for point secretion).
    pub fn nearest(&self, p: Vec3) -> (usize, usize, usize) {
        let gx = ((p.x - self.origin.x) / self.cell).round();
        let gy = ((p.y - self.origin.y) / self.cell).round();
        let gz = ((p.z - self.origin.z) / self.cell).round();
        (
            gx.clamp(0.0, (self.nx - 1) as f64) as usize,
            gy.clamp(0.0, (self.ny - 1) as f64) as usize,
            gz.clamp(0.0, (self.nz - 1) as f64) as usize,
        )
    }
}

/// An analytic concentration profile (CX3D "artificial gradient").
#[derive(Clone, Copy, Debug)]
pub enum ArtificialGradient {
    /// Bell-shaped profile along one axis.
    Gaussian {
        axis: usize,
        max: f64,
        mean: f64,
        sigma: f64,
    },
    /// Linear profile between two points along one axis.
    Linear {
        axis: usize,
        max: f64,
        peak: f64,
        zero: f64,
    },
}

impl ArtificialGradient {
    pub fn value(&self, p: Vec3) -> f64 {
        match *self {
            ArtificialGradient::Gaussian {
                axis,
                max,
                mean,
                sigma,
            } => {
                let d = p.axis(axis);
                let e = (d - mean) / sigma;
                max * (-0.5 * e * e).exp()
            }
            ArtificialGradient::Linear {
                axis,
                max,
                peak,
                zero,
            } => {
                let d = p.axis(axis);
                let (lo, hi) = if peak > zero {
                    (zero, peak)
                } else {
                    (peak, zero)
                };
                if d <= lo || d >= hi {
                    0.0
                } else {
                    max * (d - zero) / (peak - zero)
                }
            }
        }
    }
}

/// An extracellular morphogen with its own concentration field and gradients.
pub struct Substance {
    pub id: String,
    pub diffusion: f64,
    pub degradation: f64,
    pub field: Vec<f64>,
    pub gradients: Vec<ArtificialGradient>,
}

impl Substance {
    pub fn new(
        id: &str,
        diffusion: f64,
        degradation: f64,
        grid: &Grid,
        gradients: Vec<ArtificialGradient>,
    ) -> Self {
        Substance {
            id: id.to_string(),
            diffusion,
            degradation,
            field: vec![0.0; grid.volume()],
            gradients,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::new(8, 9, 10, 1.0, Vec3::ZERO)
    }

    #[test]
    fn volume_and_flat_indexing() {
        let g = grid();
        assert_eq!(g.volume(), 8 * 9 * 10);
        assert_eq!(g.flat(0, 0, 0), 0);
        assert_eq!(g.flat(0, 0, 1), 1);
        assert_eq!(g.flat(0, 1, 0), 10);
        assert_eq!(g.flat(1, 0, 0), 9 * 10);
        // Max index fits inside the field.
        assert!(g.flat(7, 8, 9) < g.volume());
    }

    #[test]
    fn trilinear_reads_grid_points() {
        let g = grid();
        let mut field = vec![0.0; g.volume()];
        field[g.flat(2, 3, 4)] = 7.0;
        // At a grid node the interpolation returns that node's value.
        let p = Vec3::new(2.0, 3.0, 4.0);
        assert_eq!(g.trilinear(&field, p), 7.0);
        // Midway between the node and zero neighbours halves it.
        let mid = Vec3::new(2.5, 3.0, 4.0);
        assert!((g.trilinear(&field, mid) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn trilinear_clamps_outside() {
        let g = grid();
        let mut field = vec![0.0; g.volume()];
        field[g.flat(7, 8, 9)] = 5.0;
        // Past the last interior node the sample clamps to the boundary value.
        let outside = Vec3::new(100.0, 100.0, 100.0);
        assert_eq!(g.trilinear(&field, outside), 5.0);
        let negative = Vec3::new(-100.0, 0.0, 0.0);
        assert_eq!(g.trilinear(&field, negative), 0.0);
    }

    #[test]
    fn laplacian_6_neighbour() {
        let g = Grid::new(4, 4, 4, 1.0, Vec3::ZERO);
        let mut field = vec![0.0; g.volume()];
        // A uniform field has zero Laplacian.
        field.iter_mut().for_each(|v| *v = 2.0);
        assert_eq!(g.laplacian(&field, 2, 2, 2), 0.0);
        // A peak at the centre gives a negative Laplacian (curvature down).
        let mut field = vec![0.0; g.volume()];
        field[g.flat(2, 2, 2)] = 6.0;
        let lap = g.laplacian(&field, 2, 2, 2);
        // 3D: sum(neighbours) - 6*centre = 0 - 36.
        assert_eq!(lap, -36.0);
    }

    #[test]
    fn nearest_clamps_to_grid() {
        let g = grid();
        assert_eq!(g.nearest(Vec3::new(2.4, 3.4, 4.4)), (2, 3, 4));
        assert_eq!(g.nearest(Vec3::new(-5.0, 0.0, 0.0)), (0, 0, 0));
        assert_eq!(g.nearest(Vec3::new(1e6, 1e6, 1e6)), (7, 8, 9));
    }

    #[test]
    fn gaussian_gradient_peaks_at_mean() {
        let g = ArtificialGradient::Gaussian {
            axis: 2,
            max: 2.0,
            mean: 0.0,
            sigma: 1.0,
        };
        assert_eq!(g.value(Vec3::new(0.0, 0.0, 0.0)), 2.0);
        assert!((g.value(Vec3::new(0.0, 0.0, 1.0)) - 2.0 * (-0.5f64).exp()).abs() < 1e-12);
        // Far away it decays to zero.
        assert!(g.value(Vec3::new(0.0, 0.0, 50.0)).abs() < 1e-12);
    }

    #[test]
    fn linear_gradient_between_peaks() {
        let g = ArtificialGradient::Linear {
            axis: 1,
            max: 10.0,
            peak: 2.0,
            zero: 0.0,
        };
        assert_eq!(g.value(Vec3::new(0.0, 0.5, 0.0)), 2.5);
        assert_eq!(g.value(Vec3::new(0.0, 1.0, 0.0)), 5.0);
        assert_eq!(g.value(Vec3::new(0.0, 0.0, 0.0)), 0.0);
        // The ramp is (lo, hi) exclusive — zero at and past the peak.
        assert_eq!(g.value(Vec3::new(0.0, 2.0, 0.0)), 0.0);
        assert_eq!(g.value(Vec3::new(0.0, 3.0, 0.0)), 0.0);
        assert_eq!(g.value(Vec3::new(0.0, -1.0, 0.0)), 0.0);
    }
}
