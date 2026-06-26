//! # TriOscillator — 3-population Kuramoto oscillator with Stuart-Landau amplitudes.
//!
//! Generic over [`T3Backend`](super::backend::T3Backend) — supports AVX2 and CUDA backends.
//!
//! ## Dynamics
//!
//! ```text
//! ṙ = (α - r²)·r + drive · gain          ← amplitude (Stuart-Landau)
//! θ̇ = ω + K/N·Σ wⱼ·Aⱼ·sin(θⱼ-θᵢ) · G  + drive·gain/(r+ε)  ← phase (Kuramoto + Phase Gate)
//! ```
//!
//! Cross-coupling умножается на [`phase_gate`](super::phase_gate::phase_gate) function
//! для защиты медленных слоёв от шума быстрых.

use crate::{
    PI, TWO_PI,
    backend::T3Backend,
    phase_gate::phase_gate,
};

/// 3-population oscillator state with generic backend.
///
/// Type parameter `B` — compute backend:
/// - [`Avx2Backend`](super::backend::Avx2Backend) (default) — AVX2 ASM
/// - `CudaBackend` — GPU (requires `cuda-kernels` crate, not yet implemented)
pub struct TriOscillator<B: T3Backend = super::backend::Avx2Backend> {
    pub n: usize,
    pub backend: B,

    // Phases
    pub th1: Vec<f32>, pub th2: Vec<f32>, pub th3: Vec<f32>,
    // Previous phases (for dθ/dt)
    pub prev_th1: Vec<f32>, pub prev_th2: Vec<f32>, pub prev_th3: Vec<f32>,
    // Amplitudes (Stuart-Landau)
    pub r1: Vec<f32>, pub r2: Vec<f32>, pub r3: Vec<f32>,
    // Previous amplitudes
    pub prev_r1: Vec<f32>, pub prev_r2: Vec<f32>, pub prev_r3: Vec<f32>,
    // Natural frequencies
    pub omega1: Vec<f32>, pub omega2: Vec<f32>, pub omega3: Vec<f32>,
    // Drive inputs
    pub drive1: Vec<f32>, pub drive2: Vec<f32>, pub drive3: Vec<f32>,
    // Sin/cos buffers
    pub sin1: Vec<f32>, pub cos1: Vec<f32>,
    pub sin2: Vec<f32>, pub cos2: Vec<f32>,
    pub sin3: Vec<f32>, pub cos3: Vec<f32>,
    // Coupling
    pub coupling1: Vec<f32>, pub coupling2: Vec<f32>, pub coupling3: Vec<f32>,
    pub cross_w12: Vec<f32>, pub cross_w13: Vec<f32>, pub cross_w23: Vec<f32>,
    // Stuart-Landau params
    pub alpha: f32,
    pub drive_gain: f32,
    pub epsilon: f32,
    // Phase Gate params
    pub phase_gate_gain: f32,
    pub gamma1: f32, pub gamma2: f32, pub gamma3: f32,
    // State
    pub initialised: bool,
}

impl<B: T3Backend> TriOscillator<B> {
    /// Create new TriOscillator with N oscillators per population.
    /// Frequencies are linearly spaced from `freq_min` to `freq_max`.
    pub fn new(n: usize, freq_min: f32, freq_max: f32, backend: B) -> Self {
        let freq_spread = freq_max - freq_min;
        Self {
            n,
            backend,
            th1: vec![0.0; n], th2: vec![0.0; n], th3: vec![0.0; n],
            prev_th1: vec![0.0; n], prev_th2: vec![0.0; n], prev_th3: vec![0.0; n],
            r1: vec![0.1; n], r2: vec![0.1; n], r3: vec![0.1; n],
            prev_r1: vec![0.1; n], prev_r2: vec![0.1; n], prev_r3: vec![0.1; n],
            omega1: (0..n).map(|i| freq_min + freq_spread * (i as f32 / n as f32)).collect(),
            omega2: (0..n).map(|i| freq_min + freq_spread * ((i + n/3) as f32 / n as f32)).collect(),
            omega3: (0..n).map(|i| freq_min + freq_spread * ((i + 2*n/3) as f32 / n as f32)).collect(),
            drive1: vec![0.0; n], drive2: vec![0.0; n], drive3: vec![0.0; n],
            sin1: vec![0.0; n], cos1: vec![0.0; n],
            sin2: vec![0.0; n], cos2: vec![0.0; n],
            sin3: vec![0.0; n], cos3: vec![0.0; n],
            coupling1: vec![0.0; n], coupling2: vec![0.0; n], coupling3: vec![0.0; n],
            cross_w12: vec![0.0; n * n], cross_w13: vec![0.0; n * n], cross_w23: vec![0.0; n * n],
            alpha: 1.0,
            drive_gain: 10.0,
            epsilon: 0.1,
            phase_gate_gain: 4.0,
            gamma1: 0.5, gamma2: 1.0, gamma3: 1.12,
            initialised: false,
        }
    }

    /// Set uniform cross-coupling strengths.
    pub fn set_uniform_coupling(&mut self, w12: f32, w13: f32, w23: f32) {
        let n = self.n;
        let norm = n as f32;
        for i in 0..n { for j in 0..n {
            self.cross_w12[i * n + j] = w12 / norm;
            self.cross_w13[i * n + j] = w13 / norm;
            self.cross_w23[i * n + j] = w23 / norm;
        }}
    }

    /// Step all 3 populations with Stuart-Landau + Kuramoto dynamics.
    ///
    /// 1. Save previous state
    /// 2. Compute sin/cos via backend
    /// 3. Compute order parameters + Phase Gate
    /// 4. Cross-population coupling with amplitudes × Phase Gate
    /// 5. Backend update (amplitudes + phases)
    /// 6. Wrap phases, clamp amplitudes
    pub fn step(&mut self, dt: f32) {
        let n = self.n;

        // 1. Save previous states
        if self.initialised {
            self.prev_th1.copy_from_slice(&self.th1);
            self.prev_th2.copy_from_slice(&self.th2);
            self.prev_th3.copy_from_slice(&self.th3);
            self.prev_r1.copy_from_slice(&self.r1);
            self.prev_r2.copy_from_slice(&self.r2);
            self.prev_r3.copy_from_slice(&self.r3);
        }
        self.initialised = true;

        // 2. Compute sin/cos via backend (also wraps phases)
        self.backend.prepare(&mut self.th1, &mut self.cos1, &mut self.sin1);
        self.backend.prepare(&mut self.th2, &mut self.cos2, &mut self.sin2);
        self.backend.prepare(&mut self.th3, &mut self.cos3, &mut self.sin3);

        // 3. Compute order parameters for Phase Gate
        let r1 = self.order_parameter(&self.th1);
        let r2 = self.order_parameter(&self.th2);
        let r3 = self.order_parameter(&self.th3);
        let g1 = phase_gate(r1, self.gamma1, n, self.phase_gate_gain);
        let g2 = phase_gate(r2, self.gamma2, n, self.phase_gate_gain);
        let g3 = phase_gate(r3, self.gamma3, n, self.phase_gate_gain);

        // 4. Cross-population coupling with amplitudes + Phase Gate
        for i in 0..n {
            // θ₁ from θ₂ (gated by g2) and θ₃ (gated by g3)
            let cos_i = self.cos1[i]; let sin_i = self.sin1[i];
            let mut c12 = 0.0; let mut c13 = 0.0;
            for j in 0..n {
                c12 += self.cross_w12[i * n + j] * self.r2[j]
                    * (self.sin2[j] * cos_i - self.cos2[j] * sin_i) * g2;
                c13 += self.cross_w13[i * n + j] * self.r3[j]
                    * (self.sin3[j] * cos_i - self.cos3[j] * sin_i) * g3;
            }
            self.coupling1[i] = c12 + c13;

            // θ₂ from θ₁ (gated by g1) and θ₃ (gated by g3)
            let cos_i = self.cos2[i]; let sin_i = self.sin2[i];
            let mut c21 = 0.0; let mut c23 = 0.0;
            for j in 0..n {
                c21 += self.cross_w12[j * n + i] * self.r1[j]
                    * (self.sin1[j] * cos_i - self.cos1[j] * sin_i) * g1;
                c23 += self.cross_w23[i * n + j] * self.r3[j]
                    * (self.sin3[j] * cos_i - self.cos3[j] * sin_i) * g3;
            }
            self.coupling2[i] = c21 + c23;

            // θ₃ from θ₁ (gated by g1) and θ₂ (gated by g2)
            let cos_i = self.cos3[i]; let sin_i = self.sin3[i];
            let mut c31 = 0.0; let mut c32 = 0.0;
            for j in 0..n {
                c31 += self.cross_w13[j * n + i] * self.r1[j]
                    * (self.sin1[j] * cos_i - self.cos1[j] * sin_i) * g1;
                c32 += self.cross_w23[j * n + i] * self.r2[j]
                    * (self.sin2[j] * cos_i - self.cos2[j] * sin_i) * g2;
            }
            self.coupling3[i] = c31 + c32;
        }

        // 5. Backend update (amplitudes + phases)
        self.backend.update(
            &mut self.th1, &mut self.r1, &self.omega1,
            &self.coupling1, &self.drive1, dt, self.alpha, self.drive_gain,
        );
        self.backend.update(
            &mut self.th2, &mut self.r2, &self.omega2,
            &self.coupling2, &self.drive2, dt, self.alpha, self.drive_gain,
        );
        self.backend.update(
            &mut self.th3, &mut self.r3, &self.omega3,
            &self.coupling3, &self.drive3, dt, self.alpha, self.drive_gain,
        );

        // 6. Wrap phases to [-π, π] and clamp amplitudes
        for p in self.th1.iter_mut() { *p = (*p + PI).rem_euclid(TWO_PI) - PI; }
        for p in self.th2.iter_mut() { *p = (*p + PI).rem_euclid(TWO_PI) - PI; }
        for p in self.th3.iter_mut() { *p = (*p + PI).rem_euclid(TWO_PI) - PI; }
        for r in self.r1.iter_mut() { *r = r.clamp(0.001, 10.0); }
        for r in self.r2.iter_mut() { *r = r.clamp(0.001, 10.0); }
        for r in self.r3.iter_mut() { *r = r.clamp(0.001, 10.0); }
    }

    /// Compute order parameter R for a population: |mean(exp(iθ))|
    pub fn order_parameter(&self, phases: &[f32]) -> f32 {
        let n = phases.len();
        if n == 0 { return 0.0; }
        let (sum_sin, sum_cos): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &p| {
            (s + p.sin() as f64, c + p.cos() as f64)
        });
        ((sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / n as f64) as f32
    }

    /// Cross order parameter between two populations.
    pub fn order_parameter_cross(&self, ph1: &[f32], ph2: &[f32]) -> f32 {
        let n = ph1.len().min(ph2.len());
        if n == 0 { return 0.0; }
        let (sum_sin, sum_cos): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
            let d = (ph1[i] - ph2[i]) as f64;
            (s + d.sin(), c + d.cos())
        });
        ((sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / n as f64) as f32
    }

    /// Get phases of all 3 populations as slices.
    pub fn phases(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.th1, &self.th2, &self.th3)
    }

    /// Get amplitudes of all 3 populations as slices.
    pub fn amplitudes(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.r1, &self.r2, &self.r3)
    }

    /// Get natural frequencies of all 3 populations as slices.
    pub fn omegas(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.omega1, &self.omega2, &self.omega3)
    }

    /// Reset oscillator to initial state (all phases=0, amps=0.1, drives=0).
    /// Preserves n, backend, frequencies, and coupling matrix.
    pub fn reset(&mut self) {
        self.th1.fill(0.0);
        self.th2.fill(0.0);
        self.th3.fill(0.0);
        self.r1.fill(0.1);
        self.r2.fill(0.1);
        self.r3.fill(0.1);
        self.prev_th1.fill(0.0);
        self.prev_th2.fill(0.0);
        self.prev_th3.fill(0.0);
        self.prev_r1.fill(0.1);
        self.prev_r2.fill(0.1);
        self.prev_r3.fill(0.1);
        self.drive1.fill(0.0);
        self.drive2.fill(0.0);
        self.drive3.fill(0.0);
        self.sin1.fill(0.0);
        self.cos1.fill(0.0);
        self.sin2.fill(0.0);
        self.cos2.fill(0.0);
        self.sin3.fill(0.0);
        self.cos3.fill(0.0);
        self.coupling1.fill(0.0);
        self.coupling2.fill(0.0);
        self.coupling3.fill(0.0);
        self.initialised = false;
    }

    /// Set uniform scalar drive on all oscillators (all 3 populations).
    pub fn set_scalar_drive(&mut self, drive: f32) {
        self.drive1.fill(drive);
        self.drive2.fill(drive);
        self.drive3.fill(drive);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Avx2Backend;

    #[test]
    fn test_tri_oscillator_step() {
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);
        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001), "phases should update");
        assert!(tri.r1.iter().any(|&r| r > 0.0), "amplitudes should exist");
    }

    #[test]
    fn test_tri_oscillator_with_phase_gate() {
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(1.0, 0.5, 0.2);
        tri.step(0.01);
        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001));
        let r = tri.order_parameter(&tri.th1);
        assert!(r > 0.0 && r <= 1.0, "order parameter in [0,1]: {}", r);
    }

    #[test]
    fn test_tri_phase_gate_integration() {
        // Phase Gate should not break basic oscillator dynamics
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        // Phase Gate enabled with defaults
        tri.step(0.01);
        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001));
        let r = tri.order_parameter(&tri.th1);
        assert!(r > 0.0 && r <= 1.0, "order parameter in [0,1]: {}", r);
    }

    #[test]
    fn test_tri_synchronization() {
        // Identical frequencies + strong coupling → should stay synchronized
        // Without intra-pop coupling, cross-pop coupling alone keeps sync
        let mut tri = TriOscillator::new(8, 1.0, 1.0, Avx2Backend); // all same omega
        tri.set_uniform_coupling(2.0, 1.0, 1.0);
        tri.gamma1 = 0.0; tri.gamma2 = 0.0; tri.gamma3 = 0.0; // gate always open
        tri.phase_gate_gain = 100.0;
        tri.step(0.1);
        // After one step, all phases should stay together (same omega, coupling 0 on first step)
        let r = tri.order_parameter(&tri.th1);
        assert!(r > 0.99, "same omega + coupling ≈ same phase: {}", r);
    }

    #[test]
    fn test_order_parameter() {
        let tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        // All phases at 0 → R = 1.0
        let r = tri.order_parameter(&vec![0.0; 8]);
        assert!((r - 1.0).abs() < 0.01, "all 0 → R=1: {}", r);
        // Uniform distribution → R ≈ 0
        let r2 = tri.order_parameter(&vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(r2 < 0.5, "uniform → R<0.5: {}", r2);
    }
}
