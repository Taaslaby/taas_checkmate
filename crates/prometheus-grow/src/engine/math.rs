//! Minimal 3D vector math (positions, directions, gradients).

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn len(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    /// Unit vector, or [`Vec3::ZERO`] for (near-)zero length.
    pub fn normalized(self) -> Vec3 {
        let l = self.len();
        if l <= 1e-12 {
            Vec3::ZERO
        } else {
            self.scale(1.0 / l)
        }
    }

    pub fn dist(self, o: Vec3) -> f64 {
        (self - o).len()
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Coordinate along an axis: 0=x, 1=y, 2=z.
    pub fn axis(&self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

/// Rotate `v` around a normalized `axis` by `angle` radians (Rodrigues).
pub fn rotate_vec(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    let (vx, vy, vz) = (v.x, v.y, v.z);
    Vec3::new(
        (t * x * x + c) * vx + (t * x * y - s * z) * vy + (t * x * z + s * y) * vz,
        (t * x * y + s * z) * vx + (t * y * y + c) * vy + (t * y * z - s * x) * vz,
        (t * x * z - s * y) * vx + (t * y * z + s * x) * vy + (t * z * z + c) * vz,
    )
}

/// Distance from point `p` to the segment `a`–`b` (used by fasciculation).
pub fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 <= 1e-12 {
        return p.dist(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.dist(a + ab * t)
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        self.scale(s)
    }
}
