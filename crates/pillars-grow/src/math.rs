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

    /// Cross product (right-hand rule).
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
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

/// Rotate `v` by `angle` radians around a unit `axis` (Rodrigues' formula).
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

/// Closest distance from point `p` to the segment `a`–`b`.
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

    #[test]
    fn rotate_vec_quarter_turn() {
        // (1,0,0) rotated +90° around z → (0,1,0).
        let r = rotate_vec(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_2,
        );
        assert!(r.x.abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
        assert!(r.z.abs() < 1e-12);
    }

    #[test]
    fn cross_product_orthogonal() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-12);
        assert!((z.y).abs() < 1e-12);
        assert!((z.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dist_point_segment_cases() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 0.0, 0.0);
        // On the line between: 0. At the ends: distance to endpoint.
        assert!(dist_point_segment(Vec3::new(5.0, 0.0, 0.0), a, b).abs() < 1e-12);
        assert!((dist_point_segment(Vec3::new(3.0, 4.0, 0.0), a, b) - 4.0).abs() < 1e-12);
        // Beyond an endpoint → distance to that endpoint.
        assert!((dist_point_segment(Vec3::new(15.0, 0.0, 0.0), a, b) - 5.0).abs() < 1e-12);
        // Degenerate segment → distance to the point.
        assert!((dist_point_segment(Vec3::new(2.0, 0.0, 0.0), a, a) - 2.0).abs() < 1e-12);
    }
}
