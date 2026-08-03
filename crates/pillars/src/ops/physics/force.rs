//! # Force — Mechanical Force Primitives
//!
//! ## `hookes_law`
//! ```text
//! F = -k · x
//! ```
//! Spring force (Hooke's Law). For muscles, tendons, exoskeleton.
//!
//! ## `viscous_damping`
//! ```text
//! F = -b · v
//! ```
//! Damping force proportional to velocity.

use crate::core::lane::SimdLane;

/// Hooke's Law: `F = -k · x`
#[inline(always)]
pub fn hookes_law<S: SimdLane>(displacement: S, stiffness: S) -> S {
    -stiffness * displacement
}

/// Viscous damping: `F = -b · v`
#[inline(always)]
pub fn viscous_damping<S: SimdLane>(velocity: S, damping: S) -> S {
    -damping * velocity
}

/// Spring + damper combined: `F = -k·x - b·v`
#[inline(always)]
pub fn spring_damper<S: SimdLane>(displacement: S, velocity: S, stiffness: S, damping: S) -> S {
    -(stiffness * displacement + damping * velocity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_hookes_restoring() {
        let x = f32x8::splat(1.0);  // stretched
        let k = f32x8::splat(2.0);
        let f = hookes_law(x, k);
        assert!((f.to_array()[0] - (-2.0)).abs() < 1e-6, "should restore: {:.4}", f.to_array()[0]);
    }
}
