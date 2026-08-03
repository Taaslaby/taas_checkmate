//! # Optimizer — Gradient Descent Optimizers
//!
//! ## SGD
//! ```text
//! w' = w - η · ∇L
//! ```
//!
//! ## SGD with momentum
//! ```text
//! v' = β·v - η·∇L
//! w' = w + v'
//! ```

use crate::core::lane::SimdLane;

/// SGD step: `w' = w - lr · grad`
#[inline(always)]
pub fn sgd_step<S: SimdLane>(w: S, grad: S, lr: S) -> S {
    w - lr * grad
}

/// SGD with momentum: returns (w', velocity').
#[inline(always)]
pub fn sgd_momentum<S: SimdLane>(w: S, velocity: S, grad: S, lr: S, momentum: S) -> (S, S) {
    let v_new = momentum * velocity - lr * grad;
    let w_new = w + v_new;
    (w_new, v_new)
}

/// Weight decay (L2 regularization): `w' = w - λ·w`
#[inline(always)]
pub fn weight_decay<S: SimdLane>(w: S, lambda: S) -> S {
    w * (S::one() - lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_sgd_decrease() {
        let w = f32x8::splat(1.0);
        let grad = f32x8::splat(0.5);
        let lr = f32x8::splat(0.1);
        let new = sgd_step(w, grad, lr);
        assert!((new.to_array()[0] - 0.95).abs() < 1e-6);
    }
}
