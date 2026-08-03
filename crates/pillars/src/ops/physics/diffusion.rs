//! # Diffusion — Ionic Diffusion and Reaction-Diffusion
//!
//! ## `diffusion_decay`
//! ```text
//! C' = C₀ + (C - C₀) · exp(-dt / τ)
//! ```
//! Exponential decay toward baseline (e.g., Ca²⁺ extrusion).
//!
//! ## `laplacian_1d`
//! ```text
//! ∇²C = (C_left + C_right - 2·C) / dx²
//! ```
//! 1D spatial diffusion (Fick's second law).

use crate::core::lane::SimdLane;

/// Exponential decay toward baseline: `C' = C₀ + (C - C₀) · exp(-dt · inv_tau)`
#[inline(always)]
pub fn diffusion_decay<S: SimdLane>(c: S, c0: S, inv_tau: S, dt: S) -> S {
    let decay = (-dt * inv_tau).exp();
    c0 + (c - c0) * decay
}

/// 1D Laplacian: `(C_left + C_right - 2·C) / dx²`
#[inline(always)]
pub fn laplacian_1d<S: SimdLane>(c: S, c_left: S, c_right: S, inv_dx2: S) -> S {
    (c_left + c_right - S::splat(2.0) * c) * inv_dx2
}

/// 1D Diffusion step: `C' = C + D · ∇²C · dt`
#[inline(always)]
pub fn diffusive_step<S: SimdLane>(c: S, c_left: S, c_right: S, d: S, inv_dx2: S, dt: S) -> S {
    c + d * laplacian_1d(c, c_left, c_right, inv_dx2) * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_diffusion_decay_to_baseline() {
        let c = f32x8::splat(0.5);
        let c0 = f32x8::splat(0.1);
        let inv_tau = f32x8::splat(1.0);
        let dt = f32x8::splat(1.0);
        let new = diffusion_decay(c, c0, inv_tau, dt);
        // decay = exp(-1) ≈ 0.368, new = 0.1 + 0.4*0.368 = 0.247
        assert!((new.to_array()[0] - 0.2472).abs() < 0.01);
    }

    #[test]
    fn test_laplacian_flat() {
        let c = f32x8::splat(1.0);
        let cl = f32x8::splat(1.0);
        let cr = f32x8::splat(1.0);
        let inv_dx2 = f32x8::splat(1.0);
        let lap = laplacian_1d(c, cl, cr, inv_dx2);
        // 1 + 1 - 2*1 = 0
        assert!((lap.to_array()[0]).abs() < 1e-6);
    }
}
