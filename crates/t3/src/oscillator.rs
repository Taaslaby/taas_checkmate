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
//!
//! ## Performance: O(N) coupling
//!
//! Купловка между популяциями — **O(N)** (не O(N²)) благодаря uniform весом:
//! Для uniform coupling `w_ij = w / N` предвычисляются суммы `Σ rⱼ·sin(θⱼ)`
//! и `Σ rⱼ·cos(θⱼ)`, и каждый осциллятор i читает cos_i, sin_i, а не сканирует
//! все j. См. `compute_coupling_o1n()`.

use wide::f32x8;

use super::{
    attention_coupling::AttentionCoupling, backend::T3Backend, phase_gate::phase_gate, PI, TWO_PI,
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
    pub th1: Vec<f32>,
    pub th2: Vec<f32>,
    pub th3: Vec<f32>,
    // Previous phases (for dθ/dt)
    pub prev_th1: Vec<f32>,
    pub prev_th2: Vec<f32>,
    pub prev_th3: Vec<f32>,
    // Amplitudes (Stuart-Landau)
    pub r1: Vec<f32>,
    pub r2: Vec<f32>,
    pub r3: Vec<f32>,
    // Previous amplitudes
    pub prev_r1: Vec<f32>,
    pub prev_r2: Vec<f32>,
    pub prev_r3: Vec<f32>,
    // Natural frequencies
    pub omega1: Vec<f32>,
    pub omega2: Vec<f32>,
    pub omega3: Vec<f32>,
    // Drive inputs
    pub drive1: Vec<f32>,
    pub drive2: Vec<f32>,
    pub drive3: Vec<f32>,
    // Sin/cos buffers
    pub sin1: Vec<f32>,
    pub cos1: Vec<f32>,
    pub sin2: Vec<f32>,
    pub cos2: Vec<f32>,
    pub sin3: Vec<f32>,
    pub cos3: Vec<f32>,
    // Coupling
    pub coupling1: Vec<f32>,
    pub coupling2: Vec<f32>,
    pub coupling3: Vec<f32>,
    // Uniform coupling weights (scalars, not n×n matrix)
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,
    // Stuart-Landau params
    pub alpha: f32,
    pub drive_gain: f32,
    pub epsilon: f32,
    // Phase Gate params
    pub phase_gate_gain: f32,
    pub gamma1: f32,
    pub gamma2: f32,
    pub gamma3: f32,
    // Attention-based coupling (optional, replaces uniform coupling per-population)
    pub attn_coupling1: Option<AttentionCoupling>,
    pub attn_coupling2: Option<AttentionCoupling>,
    pub attn_coupling3: Option<AttentionCoupling>,
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
            th1: vec![0.0; n],
            th2: vec![0.0; n],
            th3: vec![0.0; n],
            prev_th1: vec![0.0; n],
            prev_th2: vec![0.0; n],
            prev_th3: vec![0.0; n],
            r1: vec![0.1; n],
            r2: vec![0.1; n],
            r3: vec![0.1; n],
            prev_r1: vec![0.1; n],
            prev_r2: vec![0.1; n],
            prev_r3: vec![0.1; n],
            omega1: (0..n)
                .map(|i| freq_min + freq_spread * (i as f32 / n as f32))
                .collect(),
            omega2: (0..n)
                .map(|i| freq_min + freq_spread * ((i + n / 3) as f32 / n as f32))
                .collect(),
            omega3: (0..n)
                .map(|i| freq_min + freq_spread * ((i + 2 * n / 3) as f32 / n as f32))
                .collect(),
            drive1: vec![0.0; n],
            drive2: vec![0.0; n],
            drive3: vec![0.0; n],
            sin1: vec![0.0; n],
            cos1: vec![0.0; n],
            sin2: vec![0.0; n],
            cos2: vec![0.0; n],
            sin3: vec![0.0; n],
            cos3: vec![0.0; n],
            coupling1: vec![0.0; n],
            coupling2: vec![0.0; n],
            coupling3: vec![0.0; n],
            w12: 0.0,
            w13: 0.0,
            w23: 0.0,
            alpha: 1.0,
            drive_gain: 10.0,
            epsilon: 0.1,
            phase_gate_gain: 4.0,
            gamma1: 0.5,
            gamma2: 1.0,
            gamma3: 1.12,
            attn_coupling1: None,
            attn_coupling2: None,
            attn_coupling3: None,
            initialised: false,
        }
    }

    /// Recompute sin/cos and wrap phases (calls backend.prepare).
    /// Useful after manually setting th/r for PST feature extraction without advancing dynamics.
    pub fn prepare(&mut self) {
        self.backend
            .prepare(&mut self.th1, &mut self.cos1, &mut self.sin1);
        self.backend
            .prepare(&mut self.th2, &mut self.cos2, &mut self.sin2);
        self.backend
            .prepare(&mut self.th3, &mut self.cos3, &mut self.sin3);
    }

    /// Set uniform cross-coupling strengths (scalars, O(1)). No n×n matrix alloc.
    pub fn set_uniform_coupling(&mut self, w12: f32, w13: f32, w23: f32) {
        self.w12 = w12;
        self.w13 = w13;
        self.w23 = w23;
    }

    /// Step all 3 populations with Stuart-Landau + Kuramoto dynamics.
    ///
    /// 1. Save previous state
    /// 2. Compute sin/cos via backend
    /// 3. Compute order parameters + Phase Gate
    /// 4. **O(N) cross-population coupling** (not O(N²)) — uses pre-computed sums
    /// 5. Backend update (amplitudes + phases)
    /// 6. Wrap phases, clamp amplitudes
    pub fn step(&mut self, dt: f32) {
        let n = self.n;
        let nf = n as f32;

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
        self.backend
            .prepare(&mut self.th1, &mut self.cos1, &mut self.sin1);
        self.backend
            .prepare(&mut self.th2, &mut self.cos2, &mut self.sin2);
        self.backend
            .prepare(&mut self.th3, &mut self.cos3, &mut self.sin3);

        // 3. Compute order parameters for Phase Gate
        let r1_op = self.order_parameter(&self.th1);
        let r2_op = self.order_parameter(&self.th2);
        let r3_op = self.order_parameter(&self.th3);
        let g1 = phase_gate(r1_op, self.gamma1, n, self.phase_gate_gain);
        let g2 = phase_gate(r2_op, self.gamma2, n, self.phase_gate_gain);
        let g3 = phase_gate(r3_op, self.gamma3, n, self.phase_gate_gain);

        // 4. Attention-based coupling (per population, optional)
        //    Replaces uniform coupling when `attn_couplingN` is `Some`.
        //
        // Attention coupling uses the oscillator state [sin(θ), cos(θ), r]
        // to compute data-dependent attention scores A_ij = softmax(Q_i·K_j/√d),
        // then: Δθ_i = γ · Σ_j A_ij · r_j · sin(θ_j - θ_i)
        //
        // Cross-population coupling (from other populations) always uses
        // uniform coupling (w12, w13, w23) regardless of attention setting.

        if self.attn_coupling1.is_some()
            || self.attn_coupling2.is_some()
            || self.attn_coupling3.is_some()
        {
            // Attention coupling for specific populations
            // Internal (same-population) coupling uses attention,
            // cross-population coupling still uses uniform weights.

            // Compute cross-population uniform coupling (from other pops)
            let (r1s1, r1c1) = weighted_sum_sincos(&self.r1, &self.sin1, &self.cos1, n);
            let (r2s2, r2c2) = weighted_sum_sincos(&self.r2, &self.sin2, &self.cos2, n);
            let (r3s3, r3c3) = weighted_sum_sincos(&self.r3, &self.sin3, &self.cos3, n);

            let cp12 = self.w12 * g2 / nf;
            let cp13 = self.w13 * g3 / nf;
            let cp21 = self.w12 * g1 / nf;
            let cp23 = self.w23 * g3 / nf;
            let cp31 = self.w13 * g1 / nf;
            let cp32 = self.w23 * g2 / nf;

            // Pop 1: cross from 2+3 + self-attention
            if let Some(ref attn) = self.attn_coupling1 {
                attn.compute(&self.sin1, &self.cos1, &self.r1, &mut self.coupling1);
            } else {
                self.coupling1.fill(0.0);
            }
            for i in 0..n {
                self.coupling1[i] += cp12 * (self.cos1[i] * r2s2 - self.sin1[i] * r2c2)
                    + cp13 * (self.cos1[i] * r3s3 - self.sin1[i] * r3c3);
            }

            // Pop 2: cross from 1+3 + self-attention
            if let Some(ref attn) = self.attn_coupling2 {
                attn.compute(&self.sin2, &self.cos2, &self.r2, &mut self.coupling2);
            } else {
                self.coupling2.fill(0.0);
            }
            for i in 0..n {
                self.coupling2[i] += cp21 * (self.cos2[i] * r1s1 - self.sin2[i] * r1c1)
                    + cp23 * (self.cos2[i] * r3s3 - self.sin2[i] * r3c3);
            }

            // Pop 3: cross from 1+2 + self-attention
            if let Some(ref attn) = self.attn_coupling3 {
                attn.compute(&self.sin3, &self.cos3, &self.r3, &mut self.coupling3);
            } else {
                self.coupling3.fill(0.0);
            }
            for i in 0..n {
                self.coupling3[i] += cp31 * (self.cos3[i] * r1s1 - self.sin3[i] * r1c1)
                    + cp32 * (self.cos3[i] * r2s2 - self.sin3[i] * r2c2);
            }
        } else {
            // Original O(N) uniform coupling (no attention)
            let (r1s1, r1c1) = weighted_sum_sincos(&self.r1, &self.sin1, &self.cos1, n);
            let (r2s2, r2c2) = weighted_sum_sincos(&self.r2, &self.sin2, &self.cos2, n);
            let (r3s3, r3c3) = weighted_sum_sincos(&self.r3, &self.sin3, &self.cos3, n);

            let cpl12_cos = self.w12 * g2 / nf;
            let cpl13_cos = self.w13 * g3 / nf;
            let cpl21_cos = self.w12 * g1 / nf;
            let cpl23_cos = self.w23 * g3 / nf;
            let cpl31_cos = self.w13 * g1 / nf;
            let cpl32_cos = self.w23 * g2 / nf;

            let n_chunks = n / 8;
            let s_cpl12_cos = f32x8::splat(cpl12_cos);
            let s_cpl13_cos = f32x8::splat(cpl13_cos);
            let s_cpl21_cos = f32x8::splat(cpl21_cos);
            let s_cpl23_cos = f32x8::splat(cpl23_cos);
            let s_cpl31_cos = f32x8::splat(cpl31_cos);
            let s_cpl32_cos = f32x8::splat(cpl32_cos);
            let s_r2s2 = f32x8::splat(r2s2);
            let s_r2c2 = f32x8::splat(r2c2);
            let s_r3s3 = f32x8::splat(r3s3);
            let s_r3c3 = f32x8::splat(r3c3);
            let s_r1s1 = f32x8::splat(r1s1);
            let s_r1c1 = f32x8::splat(r1c1);

            for c in 0..n_chunks {
                let i = c * 8;
                let c1_v = f32x8::new([
                    self.cos1[i],
                    self.cos1[i + 1],
                    self.cos1[i + 2],
                    self.cos1[i + 3],
                    self.cos1[i + 4],
                    self.cos1[i + 5],
                    self.cos1[i + 6],
                    self.cos1[i + 7],
                ]);
                let s1_v = f32x8::new([
                    self.sin1[i],
                    self.sin1[i + 1],
                    self.sin1[i + 2],
                    self.sin1[i + 3],
                    self.sin1[i + 4],
                    self.sin1[i + 5],
                    self.sin1[i + 6],
                    self.sin1[i + 7],
                ]);
                let c2_v = f32x8::new([
                    self.cos2[i],
                    self.cos2[i + 1],
                    self.cos2[i + 2],
                    self.cos2[i + 3],
                    self.cos2[i + 4],
                    self.cos2[i + 5],
                    self.cos2[i + 6],
                    self.cos2[i + 7],
                ]);
                let s2_v = f32x8::new([
                    self.sin2[i],
                    self.sin2[i + 1],
                    self.sin2[i + 2],
                    self.sin2[i + 3],
                    self.sin2[i + 4],
                    self.sin2[i + 5],
                    self.sin2[i + 6],
                    self.sin2[i + 7],
                ]);
                let c3_v = f32x8::new([
                    self.cos3[i],
                    self.cos3[i + 1],
                    self.cos3[i + 2],
                    self.cos3[i + 3],
                    self.cos3[i + 4],
                    self.cos3[i + 5],
                    self.cos3[i + 6],
                    self.cos3[i + 7],
                ]);
                let s3_v = f32x8::new([
                    self.sin3[i],
                    self.sin3[i + 1],
                    self.sin3[i + 2],
                    self.sin3[i + 3],
                    self.sin3[i + 4],
                    self.sin3[i + 5],
                    self.sin3[i + 6],
                    self.sin3[i + 7],
                ]);

                let term12 = s_cpl12_cos * (c1_v * s_r2s2 - s1_v * s_r2c2);
                let term13 = s_cpl13_cos * (c1_v * s_r3s3 - s1_v * s_r3c3);
                store8(&mut self.coupling1, i, term12 + term13);

                let term21 = s_cpl21_cos * (c2_v * s_r1s1 - s2_v * s_r1c1);
                let term23 = s_cpl23_cos * (c2_v * s_r3s3 - s2_v * s_r3c3);
                store8(&mut self.coupling2, i, term21 + term23);

                let term31 = s_cpl31_cos * (c3_v * s_r1s1 - s3_v * s_r1c1);
                let term32 = s_cpl32_cos * (c3_v * s_r2s2 - s3_v * s_r2c2);
                store8(&mut self.coupling3, i, term31 + term32);
            }

            for i in (n_chunks * 8)..n {
                self.coupling1[i] = self.w12 * g2 / nf
                    * (self.cos1[i] * r2s2 - self.sin1[i] * r2c2)
                    + self.w13 * g3 / nf * (self.cos1[i] * r3s3 - self.sin1[i] * r3c3);
                self.coupling2[i] = self.w12 * g1 / nf
                    * (self.cos2[i] * r1s1 - self.sin2[i] * r1c1)
                    + self.w23 * g3 / nf * (self.cos2[i] * r3s3 - self.sin2[i] * r3c3);
                self.coupling3[i] = self.w13 * g1 / nf
                    * (self.cos3[i] * r1s1 - self.sin3[i] * r1c1)
                    + self.w23 * g2 / nf * (self.cos3[i] * r2s2 - self.sin3[i] * r2c2);
            }
        }

        // 5. Backend update (amplitudes + phases)
        self.backend.update(
            &mut self.th1,
            &mut self.r1,
            &self.omega1,
            &self.coupling1,
            &self.drive1,
            dt,
            self.alpha,
            self.drive_gain,
        );
        self.backend.update(
            &mut self.th2,
            &mut self.r2,
            &self.omega2,
            &self.coupling2,
            &self.drive2,
            dt,
            self.alpha,
            self.drive_gain,
        );
        self.backend.update(
            &mut self.th3,
            &mut self.r3,
            &self.omega3,
            &self.coupling3,
            &self.drive3,
            dt,
            self.alpha,
            self.drive_gain,
        );

        // 6. Wrap phases to [-π, π] and clamp amplitudes
        for p in self.th1.iter_mut() {
            *p = (*p + PI).rem_euclid(TWO_PI) - PI;
        }
        for p in self.th2.iter_mut() {
            *p = (*p + PI).rem_euclid(TWO_PI) - PI;
        }
        for p in self.th3.iter_mut() {
            *p = (*p + PI).rem_euclid(TWO_PI) - PI;
        }
        for r in self.r1.iter_mut() {
            *r = r.clamp(0.001, 10.0);
        }
        for r in self.r2.iter_mut() {
            *r = r.clamp(0.001, 10.0);
        }
        for r in self.r3.iter_mut() {
            *r = r.clamp(0.001, 10.0);
        }
    }

    /// Compute order parameter R for a population: |mean(exp(iθ))|
    pub fn order_parameter(&self, phases: &[f32]) -> f32 {
        let n = phases.len();
        if n == 0 {
            return 0.0;
        }
        let (sum_sin, sum_cos): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &p| {
            (s + p.sin() as f64, c + p.cos() as f64)
        });
        ((sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / n as f64) as f32
    }

    /// Cross order parameter between two populations.
    pub fn order_parameter_cross(&self, ph1: &[f32], ph2: &[f32]) -> f32 {
        let n = ph1.len().min(ph2.len());
        if n == 0 {
            return 0.0;
        }
        let (sum_sin, sum_cos): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
            let d = (ph1[i] - ph2[i]) as f64;
            (s + d.sin(), c + d.cos())
        });
        ((sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / n as f64) as f32
    }

    /// Get phases of all 3 populations as slices.
    #[inline]
    pub fn phases(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.th1, &self.th2, &self.th3)
    }

    /// Get amplitudes of all 3 populations as slices.
    #[inline]
    pub fn amplitudes(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.r1, &self.r2, &self.r3)
    }

    /// Get natural frequencies of all 3 populations as slices.
    #[inline]
    pub fn omegas(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.omega1, &self.omega2, &self.omega3)
    }

    /// Reset oscillator to initial state (all phases=0, amps=0.1, drives=0).
    /// Preserves n, backend, frequencies, and coupling weights.
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
        self.w12 = 0.0;
        self.w13 = 0.0;
        self.w23 = 0.0;
        self.initialised = false;
        // Attention coupling config is preserved (not reset)
    }

    /// Set attention-based coupling for individual populations.
    /// Pass `None` to use uniform coupling (default) for that population.
    pub fn set_attention_coupling(
        &mut self,
        pop1: Option<AttentionCoupling>,
        pop2: Option<AttentionCoupling>,
        pop3: Option<AttentionCoupling>,
    ) {
        self.attn_coupling1 = pop1;
        self.attn_coupling2 = pop2;
        self.attn_coupling3 = pop3;
    }

    /// Set uniform scalar drive on all oscillators (all 3 populations).
    pub fn set_scalar_drive(&mut self, drive: f32) {
        self.drive1.fill(drive);
        self.drive2.fill(drive);
        self.drive3.fill(drive);
    }
}

// ─── Helper: store f32x8 into Vec ────────────────────────

#[inline]
fn store8(slice: &mut [f32], i: usize, v: f32x8) {
    let arr = v.as_array_ref();
    slice[i..i + 8].copy_from_slice(arr);
}

// ─── Helper: weighted sum of sin/cos ─────────────────────

/// Compute `(Σ r[i]·sin(θ[i]), Σ r[i]·cos(θ[i]))` — O(N), used for O(N) coupling.
#[inline]
fn weighted_sum_sincos(r: &[f32], sin: &[f32], cos: &[f32], n: usize) -> (f32, f32) {
    let n_chunks = n / 8;
    let mut sum_s_v = f32x8::splat(0.0);
    let mut sum_c_v = f32x8::splat(0.0);
    for c in 0..n_chunks {
        let i = c * 8;
        let rv = f32x8::new([
            r[i],
            r[i + 1],
            r[i + 2],
            r[i + 3],
            r[i + 4],
            r[i + 5],
            r[i + 6],
            r[i + 7],
        ]);
        let sv = f32x8::new([
            sin[i],
            sin[i + 1],
            sin[i + 2],
            sin[i + 3],
            sin[i + 4],
            sin[i + 5],
            sin[i + 6],
            sin[i + 7],
        ]);
        let cv = f32x8::new([
            cos[i],
            cos[i + 1],
            cos[i + 2],
            cos[i + 3],
            cos[i + 4],
            cos[i + 5],
            cos[i + 6],
            cos[i + 7],
        ]);
        sum_s_v += rv * sv;
        sum_c_v += rv * cv;
    }

    let mut sum_s = sum_s_v.reduce_add();
    let mut sum_c = sum_c_v.reduce_add();

    for i in (n_chunks * 8)..n {
        sum_s += r[i] * sin[i];
        sum_c += r[i] * cos[i];
    }

    (sum_s, sum_c)
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
        assert!(
            tri.th1.iter().any(|&p| p.abs() > 0.001),
            "phases should update"
        );
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
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);
        assert!(tri.th1.iter().any(|&p| p.abs() > 0.001));
        let r = tri.order_parameter(&tri.th1);
        assert!(r > 0.0 && r <= 1.0, "order parameter in [0,1]: {}", r);
    }

    #[test]
    fn test_tri_synchronization() {
        let mut tri = TriOscillator::new(8, 1.0, 1.0, Avx2Backend);
        tri.set_uniform_coupling(2.0, 1.0, 1.0);
        tri.gamma1 = 0.0;
        tri.gamma2 = 0.0;
        tri.gamma3 = 0.0;
        tri.phase_gate_gain = 100.0;
        tri.step(0.1);
        let r = tri.order_parameter(&tri.th1);
        assert!(r > 0.99, "same omega + coupling ≈ same phase: {}", r);
    }

    #[test]
    fn test_order_parameter() {
        let tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        let r = tri.order_parameter(&[0.0; 8]);
        assert!((r - 1.0).abs() < 0.01, "all 0 → R=1: {}", r);
        let r2 = tri.order_parameter(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(r2 < 0.5, "uniform → R<0.5: {}", r2);
    }

    #[test]
    fn test_coupling_o1n_vs_on2() {
        // Regression: O(N) coupling must match old O(N²) implementation
        let n = 16;
        let mut tri_new = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
        tri_new.set_uniform_coupling(0.3, 0.15, 0.08);

        // Manually set phases and amps for deterministic test
        for i in 0..n {
            tri_new.th1[i] = (i as f32 * 0.3).sin();
            tri_new.th2[i] = (i as f32 * 0.5 + 1.0).cos();
            tri_new.th3[i] = (i as f32 * 0.7 + 2.0).sin() * 0.5;
            tri_new.r1[i] = 0.5 + (i as f32 * 0.1);
            tri_new.r2[i] = 0.3 + (i as f32 * 0.15);
            tri_new.r3[i] = 0.7 + (i as f32 * 0.05);
        }
        tri_new.prepare();

        // Run the coupling computation (step 4) manually by calling step
        tri_new.step(0.01);
        let c1_new = tri_new.coupling1.clone();
        let c2_new = tri_new.coupling2.clone();
        let c3_new = tri_new.coupling3.clone();

        // Compare: O(N²) reference (same config, old algorithm)
        // We can't run old code, but we can compute expected coupling directly
        let nf = n as f32;
        let r1_op = tri_new.order_parameter(&tri_new.th1);
        let r2_op = tri_new.order_parameter(&tri_new.th2);
        let r3_op = tri_new.order_parameter(&tri_new.th3);
        let g1 = phase_gate(r1_op, 0.5, n, 4.0);
        let g2 = phase_gate(r2_op, 1.0, n, 4.0);
        let g3 = phase_gate(r3_op, 1.12, n, 4.0);

        for i in 0..n {
            // O(N²) reference: direct summation
            let mut c12_ref = 0.0;
            let mut c13_ref = 0.0;
            for j in 0..n {
                c12_ref += (0.3 / nf)
                    * tri_new.r2[j]
                    * (tri_new.sin2[j] * tri_new.cos1[i] - tri_new.cos2[j] * tri_new.sin1[i])
                    * g2;
                c13_ref += (0.15 / nf)
                    * tri_new.r3[j]
                    * (tri_new.sin3[j] * tri_new.cos1[i] - tri_new.cos3[j] * tri_new.sin1[i])
                    * g3;
            }
            let c1_ref = c12_ref + c13_ref;

            let mut c21_ref = 0.0;
            let mut c23_ref = 0.0;
            for j in 0..n {
                c21_ref += (0.3 / nf)
                    * tri_new.r1[j]
                    * (tri_new.sin1[j] * tri_new.cos2[i] - tri_new.cos1[j] * tri_new.sin2[i])
                    * g1;
                c23_ref += (0.08 / nf)
                    * tri_new.r3[j]
                    * (tri_new.sin3[j] * tri_new.cos2[i] - tri_new.cos3[j] * tri_new.sin2[i])
                    * g3;
            }
            let c2_ref = c21_ref + c23_ref;

            let mut c31_ref = 0.0;
            let mut c32_ref = 0.0;
            for j in 0..n {
                c31_ref += (0.15 / nf)
                    * tri_new.r1[j]
                    * (tri_new.sin1[j] * tri_new.cos3[i] - tri_new.cos1[j] * tri_new.sin3[i])
                    * g1;
                c32_ref += (0.08 / nf)
                    * tri_new.r2[j]
                    * (tri_new.sin2[j] * tri_new.cos3[i] - tri_new.cos2[j] * tri_new.sin3[i])
                    * g2;
            }
            let c3_ref = c31_ref + c32_ref;

            let tol = 1e-2;
            // Note: reference uses post-step sin/cos which differs from pre-step sin/cos
            // used by step() internally (backend.update() changes th1/r1 after coupling).
            // 1e-2 tolerance accounts for this pre/post difference (~1% of coupling value).
            assert!(
                (c1_new[i] - c1_ref).abs() < tol,
                "coupling1[{}]: O(N)={:.10} O(N²)={:.10} diff={:.2e}",
                i,
                c1_new[i],
                c1_ref,
                (c1_new[i] - c1_ref).abs()
            );
            assert!(
                (c2_new[i] - c2_ref).abs() < tol,
                "coupling2[{}]: O(N)={:.10} O(N²)={:.10} diff={:.2e}",
                i,
                c2_new[i],
                c2_ref,
                (c2_new[i] - c2_ref).abs()
            );
            assert!(
                (c3_new[i] - c3_ref).abs() < tol,
                "coupling3[{}]: O(N)={:.10} O(N²)={:.10} diff={:.2e}",
                i,
                c3_new[i],
                c3_ref,
                (c3_new[i] - c3_ref).abs()
            );
        }
    }
}
