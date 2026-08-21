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

    /// Coordinate along an axis: 0=x, 1=y, 2=z.
    pub fn axis(&self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_len_dist() {
        let a = Vec3::new(1.0, 2.0, 2.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        assert_eq!(a.dot(b), 4.0);
        assert!((a.len() - 3.0).abs() < 1e-12);
        assert!((a.dist(Vec3::ZERO) - 3.0).abs() < 1e-12);
        assert!((a.dist(b) - 17.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn normalized_unit_and_zero() {
        let u = Vec3::new(3.0, 0.0, 4.0).normalized();
        assert!((u.len() - 1.0).abs() < 1e-12);
        assert!((u.x - 0.6).abs() < 1e-12);
        assert!((u.z - 0.8).abs() < 1e-12);
        // Near-zero vectors normalize to ZERO instead of NaNs.
        assert_eq!(Vec3::ZERO.normalized(), Vec3::ZERO);
        assert_eq!(Vec3::new(1e-14, 0.0, 0.0).normalized(), Vec3::ZERO);
    }

    #[test]
    fn scale_and_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a + a, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a - a, Vec3::ZERO);
    }

    #[test]
    fn axis_accessors() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a.axis(0), 1.0);
        assert_eq!(a.axis(1), 2.0);
        assert_eq!(a.axis(2), 3.0);
        assert_eq!(a.axis(9), 3.0); // fallback = z
    }
}
