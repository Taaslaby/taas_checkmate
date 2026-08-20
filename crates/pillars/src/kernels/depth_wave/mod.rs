//! # 🌊 DepthGCO — depth-modulated graph-coupled oscillator kernel
//!
//! Канон `GcoState` — бит-в-бит порт легаси `kernels::neuron::depth_wave`
//! (волна 8, транш 1f). Единственное документированное расхождение: легаси
//! использовал `rayon::par_iter().sum()` для глобальной L2-нормы — порядок
//! редукции зависел от пула потоков, результат был **недетерминирован**.
//! Канон считает норму детерминированным последовательным циклом (тот же
//! порядок индексов → тот же бит-результат при одном потоке rayon).
//!
//! Ключевая идея: топологическая глубина нейрона определяет скорость
//! его фазовой эволюции:
//!
//! ```text
//! speed[i] = 1.0 + depth[i] * mod_range
//!   → sensory (depth≈1) эволюционирует БЫСТРО
//!   → motor   (depth≈0) — МЕДЛЕННО (интегрирует дольше)
//! ```
//!
//! Один шаг:
//!   1. CSR SpMV: `H_real = H @ u`,  `H_imag = -(H @ v)`
//!   2. `v += dt·speed·H_real`, `u += dt·speed·H_imag` (+ внешний ток с весом `1−depth`)
//!   3. Глобальная L2-нормализация: `Σ|v+iu|² = 1`

/// Состояние DepthGCO — depth-modulated coupled oscillators на CSR графе.
#[derive(Debug, Clone)]
pub struct GcoState {
    /// Re(v+iu) — действительная часть `[n]`
    pub v: Vec<f32>,
    /// Im(v+iu) — мнимая часть `[n]`
    pub u: Vec<f32>,
    /// Топологическая глубина каждого нейрона `[0, 1]`. sensory=1, motor=0
    pub depth: Vec<f32>,
    /// Скорость эволюции: `speed[i] = 1 + depth[i] * mod_range`, длина `[n]`.
    pub speed: Vec<f32>,
    /// Буфер H @ u (h_real) `[n]`
    pub h_real: Vec<f32>,
    /// Буфер -(H @ v) (h_imag) `[n]`
    pub h_imag: Vec<f32>,
    /// Количество нейронов
    pub n: usize,
    /// Временной шаг
    pub dt: f32,
    /// Диапазон модуляции скорости
    pub mod_range: f32,
}

impl GcoState {
    #[inline]
    fn assert_internal_state(&self) {
        assert_eq!(self.v.len(), self.n, "GcoState: v length must equal n");
        assert_eq!(self.u.len(), self.n, "GcoState: u length must equal n");
        assert_eq!(
            self.depth.len(),
            self.n,
            "GcoState: depth length must equal n"
        );
        assert_eq!(
            self.speed.len(),
            self.n,
            "GcoState: speed length must equal n"
        );
        assert_eq!(
            self.h_real.len(),
            self.n,
            "GcoState: h_real length must equal n"
        );
        assert_eq!(
            self.h_imag.len(),
            self.n,
            "GcoState: h_imag length must equal n"
        );
    }

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
    /// `h_real[i] = Σ_j H[i,j] * u[j]`
    /// `h_imag[i] = Σ_j -H[i,j] * v[j]`
    ///
    /// Где `H[i,j] = vals[idx] * nt_signs[i]` (пост-синаптический знак).
    pub fn compute_h_psi(
        &mut self,
        row_ptr: &[u32],
        col_idx: &[u32],
        vals: &[f32],
        nt_signs: &[f32],
    ) {
        self.assert_internal_state();
        assert_eq!(
            row_ptr.len(),
            self.n + 1,
            "GcoState::compute_h_psi: row_ptr must have n + 1 entries"
        );
        assert_eq!(
            col_idx.len(),
            vals.len(),
            "GcoState::compute_h_psi: col_idx/vals length mismatch"
        );
        assert_eq!(
            nt_signs.len(),
            self.n,
            "GcoState::compute_h_psi: nt_signs length mismatch"
        );
        assert_eq!(
            row_ptr[0], 0,
            "GcoState::compute_h_psi: row_ptr must start at 0"
        );
        for (i, window) in row_ptr.windows(2).enumerate() {
            assert!(
                window[0] <= window[1],
                "GcoState::compute_h_psi: row_ptr must be monotonic at index {i}"
            );
            assert!(
                window[1] as usize <= col_idx.len(),
                "GcoState::compute_h_psi: row_ptr exceeds synapse count"
            );
        }
        for (i, &col) in col_idx.iter().enumerate() {
            assert!(
                (col as usize) < self.n,
                "GcoState::compute_h_psi: col_idx[{i}] out of range"
            );
        }

        let u = &self.u;
        let v = &self.v;

        for i in 0..self.n {
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
            self.h_real[i] = sum_u;
            self.h_imag[i] = -sum_v;
        }
    }

    /// Один шаг depth-модулированной фазовой эволюции.
    ///
    /// `v[i] += dt·speed[i]·h_real[i] + dt·(1−depth[i])·i_ext[i]`
    /// `u[i] += dt·speed[i]·h_imag[i]`
    /// Нормализация: Σ|v+iu|² → 1
    pub fn evolve(&mut self, i_ext: Option<&[f32]>) {
        self.assert_internal_state();
        let dt = self.dt;

        for i in 0..self.n {
            let sp = self.speed[i];
            self.v[i] += dt * sp * self.h_real[i];
            self.u[i] += dt * sp * self.h_imag[i];

            if let Some(ext) = i_ext {
                if i < ext.len() {
                    let sensory_w = 1.0 - self.depth[i];
                    self.v[i] += dt * sensory_w * ext[i];
                }
            }
        }

        // Глобальная L2 нормализация: Σ|v+iu|² = 1.
        // Детерминированный последовательный порядок (легаси-rayon был
        // недетерминирован — см. doc-заголовок модуля).
        let norm_sq: f32 = self
            .v
            .iter()
            .zip(self.u.iter())
            .map(|(v, u)| v * v + u * u)
            .sum();
        let inv_norm = 1.0 / (norm_sq + 1e-9).sqrt();
        for i in 0..self.n {
            self.v[i] *= inv_norm;
            self.u[i] *= inv_norm;
        }
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
        self.assert_internal_state();
        if self.n == 0 {
            return 0.0;
        }
        let n = self.n as f32;
        let mean_re: f32 = self.v.iter().sum::<f32>() / n;
        let mean_im: f32 = self.u.iter().sum::<f32>() / n;
        (mean_re * mean_re + mean_im * mean_im).sqrt()
    }

    /// |v+iu| для моторных нейронов (depth < 0.3)
    pub fn motor_amplitudes(&self) -> Vec<f32> {
        self.assert_internal_state();
        (0..self.n)
            .filter(|&i| self.depth[i] < 0.3)
            .map(|i| (self.v[i] * self.v[i] + self.u[i] * self.u[i]).sqrt())
            .collect()
    }

    /// Сброс состояния.
    pub fn reset(&mut self, seed: u64) {
        self.assert_internal_state();
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

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests;
