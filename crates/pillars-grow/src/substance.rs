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
