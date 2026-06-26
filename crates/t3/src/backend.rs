//! # T3Backend trait — interchangeable compute backends.
//!
//! Позволяет TriOscillator работать через любой бэкенд:
//!
//! - [`Avx2Backend`] — AVX2 ASM через neuron-engine FFI (по умолчанию, fast)
//! - [`ScalarBackend`] — чистый Rust, без ASM (portable, для отладки)
//! - [`CudaBackend`] — GPU через `cuda-kernels` крейт (planned)
//!
//! ## Архитектура
//!
//! ```text
//! TriOscillator<B: T3Backend>
//!   ├── B::prepare(phases, cos, sin)     →  sin/cos из фаз
//!   ├── B::update(phases, amps, omega,   →  Stuart-Landau + Kuramoto шаг
//!   │           coupling, drives, ...)
//!   ├── B::nlms_train(...)                →  NLMS weight update
//!   └── B::kalman_update(...)             →  Kalman weight update
//! ```

use crate::{PI, TWO_PI};

// ─── T3Backend trait ─────────────────────────────────────

/// Бэкенд для T3 вычислений.
///
/// Каждый метод принимает `&self` — для CPU бэкендов это ZST,
/// для GPU бэкендов это контекст с persistent device memory.
pub trait T3Backend: Send + Sync {
    /// Compute sin/cos from phases (in-place buffers).
    /// NOTE: phases is mutable because the AVX kernel also wraps phases to [-π, π].
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]);

    /// Update phases and amplitudes with Stuart-Landau dynamics:
    ///
    /// ```text
    /// r += dt · ((α - r²)·r + drive · gain)
    /// θ += dt · (ω + coupling + drive·gain/(r+ε))
    /// ```
    fn update(
        &self,
        phases: &mut [f32],
        amplitudes: &mut [f32],
        omegas: &[f32],
        coupling: &[f32],
        drives: &[f32],
        dt: f32,
        alpha: f32,
        gain: f32,
    );

    /// NLMS train step: weights += lr · error · states / (||states||² + eps)
    fn nlms_train(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
        states: &[f32],
        error: f32,
        lr: f32,
        eps: f32,
    );

    /// Kalman filter decoder update.
    ///
    /// ```text
    /// K[i]  = cov[i] · states[i] / s
    /// w[i] += K[i] · error
    /// cov[i] -= K[i] · states[i] · cov[i]
    /// ```
    ///
    /// # Important: `s` must be the innovation covariance
    ///
    /// `s = R + Σᵢ(cov[i] · states[i]²)` where `R` is measurement noise.
    /// Compute it in the caller *before* calling this method, then pass as `s`.
    /// See `KalmanDecoder::train()` for the canonical pattern.
    fn kalman_update(
        &self,
        weights: &mut [f32],
        cov: &mut [f32],
        states: &[f32],
        error: f32,
        s: f32,
    );
}

// ─── Avx2Backend (default) ───────────────────────────────

/// AVX2 backend via pure Rust SIMD kernels.
///
/// Zero-sized struct — все методы вызывают pure Rust kernels напрямую.
pub struct Avx2Backend;

impl T3Backend for Avx2Backend {
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
        crate::kernels::kuramoto_prepare(phases, cos_out, sin_out);
    }

    fn update(
        &self,
        phases: &mut [f32],
        amplitudes: &mut [f32],
        omegas: &[f32],
        coupling: &[f32],
        drives: &[f32],
        dt: f32,
        alpha: f32,
        gain: f32,
    ) {
        crate::kernels::t3_update(phases, amplitudes, omegas, coupling, drives, dt, 0.0, alpha, gain);
    }

    fn nlms_train(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
        states: &[f32],
        error: f32,
        lr: f32,
        eps: f32,
    ) {
        crate::kernels::nlms_train(weights, bias, states, error, lr, eps);
    }

    fn kalman_update(
        &self,
        weights: &mut [f32],
        cov: &mut [f32],
        states: &[f32],
        error: f32,
        s: f32,
    ) {
        crate::kernels::kalman_update(weights, cov, states, error, s);
    }
}

// ─── ScalarBackend (pure Rust, no ASM) ───────────────────

/// Pure Rust scalar backend — no AVX2 ASM required.
///
/// Полезно для:
/// - Отладки (можно поставить breakpoint в Rust коде)
/// - Сравнения точности ASM vs scalar
/// - Платформ без AVX2 (ARM, RISC-V, старые x86)
///
/// ## Выбор backend
///
/// ```rust
/// use taas_t3::{TriOscillator, Avx2Backend, ScalarBackend};
///
/// // ASM (fast, requires AVX2)
/// let tri_avx = TriOscillator::new(64, 0.5, 1.5, Avx2Backend);
///
/// // Pure Rust (portable)
/// let tri_scalar = TriOscillator::new(64, 0.5, 1.5, ScalarBackend);
/// ```
pub struct ScalarBackend;

impl T3Backend for ScalarBackend {
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
        for i in 0..phases.len() {
            let p = phases[i];
            // Wrap phase to [-π, π]
            let wrapped = (p + PI).rem_euclid(TWO_PI) - PI;
            phases[i] = wrapped;
            cos_out[i] = wrapped.cos();
            sin_out[i] = wrapped.sin();
        }
    }

    fn update(
        &self,
        phases: &mut [f32],
        amplitudes: &mut [f32],
        omegas: &[f32],
        coupling: &[f32],
        drives: &[f32],
        dt: f32,
        alpha: f32,
        gain: f32,
    ) {
        let eps = 0.1; // amplitude floor
        for i in 0..phases.len() {
            let r = amplitudes[i];
            // Stuart-Landau: r += dt · ((α - r²)·r + drive · gain)
            let drive = drives.get(i).copied().unwrap_or(0.0);
            let r_dot = (alpha - r * r) * r + drive * gain;
            amplitudes[i] = r + dt * r_dot;

            // Kuramoto: θ += dt · (ω + coupling + drive · gain / (r + ε))
            let omega = omegas.get(i).copied().unwrap_or(0.0);
            let cpl = coupling.get(i).copied().unwrap_or(0.0);
            let drive_term = drive * gain / (r + eps);
            phases[i] += dt * (omega + cpl + drive_term);
        }
    }

    fn nlms_train(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
        states: &[f32],
        error: f32,
        lr: f32,
        eps: f32,
    ) {
        // μ = lr / (||states||² + eps)
        let norm_sq: f32 = states.iter().map(|&x| x * x).sum();
        let mu = lr / (norm_sq + eps);
        for (w, &s) in weights.iter_mut().zip(states.iter()) {
            *w += mu * error * s;
        }
        bias[0] += mu * error;
    }

    fn kalman_update(
        &self,
        weights: &mut [f32],
        cov: &mut [f32],
        states: &[f32],
        error: f32,
        s: f32,
    ) {
        if s <= 1e-12 { return; }
        for i in 0..weights.len() {
            let k = cov[i] * states[i] / s;
            weights[i] += k * error;
            cov[i] -= k * cov[i];
        }
    }
}

// ─── CudaBackend (placeholder — requires cuda-kernels crate) ───
//
// To implement: add `cuda-kernels` as dependency, create CudaBackend
// struct with GpuContext, implement T3Backend trait.
//
// ```ignore
// pub struct CudaBackend {
//     ctx: cuda_kernels::GpuContext,
//     n: usize,
// }
// impl T3Backend for CudaBackend { ... }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_backend_prepare() {
        let backend = Avx2Backend;
        // Use n=8 (multiple of AVX2 vector width)
        let mut phases = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut cos_out = vec![0.0; 8];
        let mut sin_out = vec![0.0; 8];
        backend.prepare(&mut phases, &mut cos_out, &mut sin_out);
        assert!((cos_out[0] - 1.0).abs() < 0.01, "cos(0)≈1: {}", cos_out[0]);
        assert!(sin_out[0].abs() < 0.01, "sin(0)≈0: {}", sin_out[0]);
    }

    #[test]
    fn test_avx2_backend_update() {
        let backend = Avx2Backend;
        let mut phases = vec![0.0; 8];
        let mut amps = vec![0.1; 8];
        let omegas = vec![1.0; 8];
        let coupling = vec![0.0; 8];
        let drives = vec![0.0; 8];
        backend.update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.5, 1.0, 10.0);
        assert!(phases.iter().any(|&p| p != 0.0), "phases should update");
        assert!((amps[0] - 0.1).abs() > 0.001, "amps should update: {:.6}", amps[0]);
    }

    #[test]
    fn test_avx2_backend_nlms() {
        let backend = Avx2Backend;
        // Use n=8 (multiple of AVX2 vector width = 8 × f32)
        let mut weights = vec![0.0; 8];
        let mut bias = vec![0.0];
        let states = vec![1.0, 0.5, -0.3, 0.0, -0.5, 0.3, 0.2, -0.1];
        backend.nlms_train(&mut weights, &mut bias, &states, 1.0, 0.01, 1e-8);
        assert!(weights.iter().any(|&w| w != 0.0));
    }

    // ─── ScalarBackend tests ───────────────────────────

    #[test]
    fn test_scalar_backend_prepare() {
        let backend = ScalarBackend;
        let mut phases = vec![0.0, 1.0, 2.0, 3.0];
        let mut cos_out = vec![0.0; 4];
        let mut sin_out = vec![0.0; 4];
        backend.prepare(&mut phases, &mut cos_out, &mut sin_out);
        assert!((cos_out[0] - 1.0).abs() < 0.01, "cos(0)≈1: {}", cos_out[0]);
        assert!(sin_out[0].abs() < 0.01, "sin(0)≈0: {}", sin_out[0]);
    }

    #[test]
    fn test_scalar_backend_update() {
        let backend = ScalarBackend;
        let mut phases = vec![0.0; 4];
        let mut amps = vec![0.1; 4];
        let omegas = vec![1.0; 4];
        let coupling = vec![0.0; 4];
        let drives = vec![0.0; 4];
        backend.update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.5, 1.0, 10.0);
        assert!(phases.iter().any(|&p| p != 0.0), "phases should update");
        assert!((amps[0] - 0.1).abs() > 0.001, "amps should update: {:.6}", amps[0]);
    }

    #[test]
    fn test_scalar_backend_nlms() {
        let backend = ScalarBackend;
        let mut weights = vec![0.0; 4];
        let mut bias = vec![0.0];
        let states = vec![1.0, 0.5, -0.3, 0.0];
        backend.nlms_train(&mut weights, &mut bias, &states, 1.0, 0.01, 1e-8);
        assert!(weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_scalar_backend_kalman() {
        let backend = ScalarBackend;
        let mut weights = vec![0.0; 4];
        let mut cov = vec![1.0; 4];
        let states = vec![0.5, -0.3, 0.8, 0.1];
        backend.kalman_update(&mut weights, &mut cov, &states, 1.0, 2.0);
        assert!(weights.iter().any(|&w| w != 0.0));
        assert!(cov[0] < 1.0, "cov should decrease: {}", cov[0]);
    }

    #[test]
    fn test_scalar_backend_prepare_properties() {
        // Фундаментальные свойства prepare:
        // 1. Phases wrapped to [-π, π]
        // 2. sin² + cos² ≈ 1
        let backend = ScalarBackend;
        let mut phases = vec![-10.0, 0.0, 3.5, 100.0];
        let mut cos_out = vec![0.0; 4];
        let mut sin_out = vec![0.0; 4];
        backend.prepare(&mut phases, &mut cos_out, &mut sin_out);

        for i in 0..4 {
            assert!(phases[i] >= -std::f32::consts::PI && phases[i] <= std::f32::consts::PI,
                "phase {i} not wrapped: {}", phases[i]);
            let pyth = (sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i] - 1.0).abs();
            assert!(pyth < 0.001, "sin²+cos²≠1 at {i}: {}", pyth);
        }
    }
}
