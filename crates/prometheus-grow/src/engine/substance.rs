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

    /// Analytic gradient of the trilinear interpolant at world position `p`.
    ///
    /// The old `gradient` computed a numerical derivative via 6 `trilinear`
    /// calls (48 grid loads per point). The interpolant is piecewise
    /// trilinear, so its gradient is available in closed form from the *same*
    /// 8 corner values (8 loads):
    ///
    /// ```text
    /// dC/dx = 1/h · Σ_{j,k} (c_{1jk} − c_{0jk}) · w_j(y) · w_k(z)
    /// ```
    ///
    /// with `w_0(t) = 1−t`, `w_1(t) = t`. Same 8 corners, 6× fewer loads.
    pub fn trilinear_gradient(&self, field: &[f64], p: Vec3) -> Vec3 {
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

        let inv_h = 1.0 / self.cell;
        let w0x = 1.0 - fx;
        let w1x = fx;
        let w0y = 1.0 - fy;
        let w1y = fy;
        let w0z = 1.0 - fz;
        let w1z = fz;

        // dC/dx = 1/h · Σ_{j,k} (c_{1jk} − c_{0jk}) · w_j(fy) · w_k(fz)
        let dx = ((c100 - c000) * w0y + (c110 - c010) * w1y) * w0z
            + ((c101 - c001) * w0y + (c111 - c011) * w1y) * w1z;
        // dC/dy = 1/h · Σ_{i,k} (c_{i1k} − c_{i0k}) · w_i(fx) · w_k(fz)
        let dy = ((c010 - c000) * w0z + (c011 - c001) * w1z) * w0x
            + ((c110 - c100) * w0z + (c111 - c101) * w1z) * w1x;
        // dC/dz = 1/h · Σ_{i,j} (c_{ij1} − c_{ij0}) · w_i(fx) · w_j(fy)
        let dz = ((c001 - c000) * w0y + (c011 - c010) * w1y) * w0x
            + ((c101 - c100) * w0y + (c111 - c110) * w1y) * w1x;

        Vec3::new(dx * inv_h, dy * inv_h, dz * inv_h)
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

    /// Analytic gradient of the profile at `p` (zero outside its support).
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        match *self {
            ArtificialGradient::Gaussian {
                axis,
                max,
                mean,
                sigma,
            } => {
                let d = p.axis(axis);
                let e = (d - mean) / sigma;
                let g = max * (-0.5 * e * e).exp() * (-e / sigma);
                axis_vec(axis, g)
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
                    Vec3::ZERO
                } else {
                    axis_vec(axis, max / (peak - zero))
                }
            }
        }
    }
}

fn axis_vec(axis: usize, v: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(v, 0.0, 0.0),
        1 => Vec3::new(0.0, v, 0.0),
        _ => Vec3::new(0.0, 0.0, v),
    }
}

/// BioDynaMo-style boundary condition (Dirichlet/Neumann/Open/Closed)
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition { Dirichlet(f64), Neumann(f64), Open, Closed }

/// An extracellular morphogen with its own concentration field and gradients.
/// BioDynaMo `DiffusionGrid` port: adds thresholds + boundary condition + double buffer.
pub struct Substance {
    pub id: String,
    pub diffusion: f64,
    pub degradation: f64,
    pub field: Vec<f64>,
    pub gradients: Vec<ArtificialGradient>,
    pub upper_threshold: f64,
    pub lower_threshold: f64,
    pub boundary: BoundaryCondition,
    pub c2: Vec<f64>, // double buffer à la BioDynaMo c1_/c2_
}

impl Substance {
    pub fn new(
        id: &str,
        diffusion: f64,
        degradation: f64,
        grid: &Grid,
        gradients: Vec<ArtificialGradient>,
    ) -> Self {
        let n = grid.volume();
        Substance {
            id: id.to_string(),
            diffusion,
            degradation,
            field: vec![0.0; n],
            gradients,
            upper_threshold: 1e15,
            lower_threshold: 0.0,
            boundary: BoundaryCondition::Neumann(0.0),
            c2: vec![0.0; n],
        }
    }
    pub fn with_thresholds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_threshold = lower;
        self.upper_threshold = upper;
        self
    }
    pub fn with_boundary(mut self, bc: BoundaryCondition) -> Self {
        self.boundary = bc;
        self
    }
}
