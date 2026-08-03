//! 🌊 DepthGCO — depth-modulated graph-coupled oscillator kernel.
//!
//! Ключевая идея: топологическая глубина нейрона определяет скорость
//! его фазовой эволюции:
//!
//!   speed[i] = 1.0 + depth[i] * mod_range
//!     → sensory (depth≈1) эволюционирует БЫСТРО
//!     → motor   (depth≈0) — МЕДЛЕННО (интегрирует дольше)
//!
//! Один шаг:
//!   1. CSR SpMV: H_real = H @ u,  H_imag = -(H @ v)
//!   2. SIMD gco_step: v += dt * speed * H_real, u += dt * speed * H_imag
//!   3. Global L2 normalize: Σ|v+iu|² = 1
//!
//! Вычислительная стоимость: O(2N + E) vs HH O(4N + E)
//!   → в ~2× дешевле HH на том же графе.

/// Состояние DepthGCO — depth-modulated coupled oscillators на CSR графе.
#[derive(Debug, Clone)]
pub struct GcoState {
    /// Re(v+iu) — действительная часть [n]
    pub v: Vec<f32>,
    /// Im(v+iu) — мнимая часть [n]
    pub u: Vec<f32>,
    /// Топологическая глубина каждого нейрона [0, 1]. sensory=1, motor=0
    pub depth: Vec<f32>,
    /// Скорость эволюции: speed[i] = 1 + depth[i] * mod_range [n]
    pub speed: Vec<f32>,
    /// Буфер H @ u (h_real) [n]
    pub h_real: Vec<f32>,
    /// Буфер -(H @ v) (h_imag) [n]
    pub h_imag: Vec<f32>,
    /// Количество нейронов
    pub n: usize,
    /// Временной шаг
    pub dt: f32,
    /// Диапазон модуляции скорости
    pub mod_range: f32,
}

impl GcoState {
    /// Создать новое состояние DepthGCO.
    ///
    /// # Аргументы
    /// * `n` — количество нейронов
    /// * `depth` — топологическая глубина [0..1], длина n
    /// * `mod_range` — диапазон модуляции скорости (0 = plain GCO)
    /// * `dt` — шаг эволюции
    /// * `seed` — seed для начальных фаз
    pub fn new(n: usize, depth: Vec<f32>, mod_range: f32, dt: f32, seed: u64) -> Self {
        assert_eq!(depth.len(), n);

        // Начальные фазы из простого хэша
        let mut v = vec![0.0f32; n];
        let mut u = vec![0.0f32; n];
        let mut rng = seed;
        for i in 0..n {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let phase = (rng as f32) / (u64::MAX as f32) * std::f32::consts::TAU;
            v[i] = phase.cos();
            u[i] = phase.sin();
        }

        // speed[i] = 1.0 + depth[i] * mod_range
        let speed: Vec<f32> = depth.iter().map(|&d| 1.0 + d * mod_range).collect();

        Self {
            v,
            u,
            depth,
            speed,
            h_real: vec![0.0; n],
            h_imag: vec![0.0; n],
            n,
            dt,
            mod_range,
        }
    }

    /// CSR SpMV: вычислить H @ ψ.
    ///
    /// h_real[i] = Σ_j  H[i,j] * u[j]
    /// h_imag[i] = Σ_j -H[i,j] * v[j]
    ///
    /// Где H[i,j] = vals[idx] * nt_signs[i] (пост-синаптический знак).
    pub fn compute_h_psi(
        &mut self,
        row_ptr: &[u32],
        col_idx: &[u32],
        vals: &[f32],
        nt_signs: &[f32],
    ) {
        use rayon::prelude::*;
        let u = &self.u;
        let v = &self.v;

        self.h_real
            .par_iter_mut()
            .zip(self.h_imag.par_iter_mut())
            .enumerate()
            .for_each(|(i, (hr, hi))| {
                let s = row_ptr[i] as usize;
                let e = row_ptr[i + 1] as usize;
                let sign_i = nt_signs[i];
                let mut sum_u = 0.0f32;
                let mut sum_v = 0.0f32;
                for idx in s..e {
                    let j = col_idx[idx] as usize;
                    let w = vals[idx] * sign_i;
                    sum_u += w * u[j];
                    sum_v += w * v[j];
                }
                *hr = sum_u;
                *hi = -sum_v;
            });
    }

    /// Один шаг depth-модулированной фазовой эволюции.
    ///
    /// v[i] += dt * speed[i] * h_real[i] + dt * sensory_weight[i] * i_ext[i]
    /// u[i] += dt * speed[i] * h_imag[i]
    /// Нормализация: Σ|v+iu|² → 1
    pub fn evolve(&mut self, i_ext: Option<&[f32]>) {
        use rayon::prelude::*;
        let dt = self.dt;
        let speed = &self.speed;
        let depth = &self.depth;
        let h_real = &self.h_real;
        let h_imag = &self.h_imag;

        self.v
            .par_iter_mut()
            .zip(self.u.par_iter_mut())
            .enumerate()
            .for_each(|(i, (vi, ui))| {
                let sp = speed[i];
                *vi += dt * sp * h_real[i];
                *ui += dt * sp * h_imag[i];

                if let Some(ext) = i_ext {
                    if i < ext.len() {
                        let sensory_w = 1.0 - depth[i];
                        *vi += dt * sensory_w * ext[i];
                    }
                }
            });

        // Глобальная L2 нормализация: Σ|v+iu|² = 1
        let norm_sq: f32 = self
            .v
            .par_iter()
            .zip(self.u.par_iter())
            .map(|(v, u)| v * v + u * u)
            .sum();
        let inv_norm = 1.0 / (norm_sq + 1e-9).sqrt();
        self.v
            .par_iter_mut()
            .zip(self.u.par_iter_mut())
            .for_each(|(vi, ui)| {
                *vi *= inv_norm;
                *ui *= inv_norm;
            });
    }

    /// Полный шаг: CSR SpMV + evolve.
    pub fn step(
        &mut self,
        row_ptr: &[u32],
        col_idx: &[u32],
        vals: &[f32],
        nt_signs: &[f32],
        i_ext: Option<&[f32]>,
    ) {
        self.compute_h_psi(row_ptr, col_idx, vals, nt_signs);
        self.evolve(i_ext);
    }

    /// Когерентность = |mean(v+iu)|
    pub fn coherence(&self) -> f32 {
        let n = self.n as f32;
        let mean_re: f32 = self.v.iter().sum::<f32>() / n;
        let mean_im: f32 = self.u.iter().sum::<f32>() / n;
        (mean_re * mean_re + mean_im * mean_im).sqrt()
    }

    /// |v+iu| для моторных нейронов (depth < 0.3)
    pub fn motor_amplitudes(&self) -> Vec<f32> {
        (0..self.n)
            .filter(|&i| self.depth[i] < 0.3)
            .map(|i| (self.v[i] * self.v[i] + self.u[i] * self.u[i]).sqrt())
            .collect()
    }

    /// Сброс состояния.
    pub fn reset(&mut self, seed: u64) {
        let mut rng = seed;
        for i in 0..self.n {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let phase = (rng as f32) / (u64::MAX as f32) * std::f32::consts::TAU;
            self.v[i] = phase.cos();
            self.u[i] = phase.sin();
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_wave_evolves_and_normalizes() {
        // Маленький граф: 4 нейрона, 4 синапса (кольцо)
        let n = 4;
        let depth = vec![0.9, 0.5, 0.5, 0.1]; // sensory → intrinsic → motor
        let mut dw = GcoState::new(n, depth, 2.0, 0.5, 42);

        // CSR кольцо: 0→1, 1→2, 2→3, 3→0
        let row_ptr = vec![0u32, 1, 2, 3, 4];
        let col_idx = vec![1u32, 2, 3, 0];
        let vals = vec![1.0f32, 1.0, 1.0, 1.0];
        let nt_signs = vec![1.0f32, 1.0, -1.0, 1.0]; // нейрон 2 тормозной

        // 100 шагов
        for _ in 0..100 {
            dw.step(&row_ptr, &col_idx, &vals, &nt_signs, None);
        }

        // Проверяем нормализацию |v+iu|² ≈ 1
        let norm_sq: f32 = dw.v.iter().zip(&dw.u).map(|(v, u)| v * v + u * u).sum();
        assert!(
            (norm_sq - 1.0).abs() < 0.01,
            "norm should be ~1.0, got {}",
            norm_sq
        );

        // Проверяем speed[0] > speed[3] (sensory быстрее motor)
        assert!(dw.speed[0] > dw.speed[3]);
    }

    #[test]
    fn test_depth_wave_coherence() {
        let n = 8;
        let depth = vec![0.5; n];
        let dw = GcoState::new(n, depth, 0.0, 0.5, 42);
        let coh = dw.coherence();
        assert!((0.0..=1.0).contains(&coh));
    }

    #[test]
    fn test_motor_amplitudes() {
        let depth = vec![0.9, 0.5, 0.2, 0.05]; // 2 motor (depth<0.3)
        let dw = GcoState::new(4, depth, 2.0, 0.5, 42);
        let amps = dw.motor_amplitudes();
        assert_eq!(amps.len(), 2);
    }
}
