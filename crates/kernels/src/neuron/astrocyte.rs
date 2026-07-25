//! 🧫 ASTROCYTE & GAP-JUNCTION GLIAL WAVE KERNEL
//!
//! Невидимые осцилляторы мозга и ВНС: астроциты, глия и электрические
//! щелевые контакты (Gap Junctions / Volume Transmission).
//!
//! Особенности физиологии астроцитарной сети:
//!   1. Медленные кальциевые волны (Ca²⁺ waves): τ = 1.0 – 10.0 сек (0.1 – 2 Гц)
//!   2. Электрическая связь через Gap Junctions: I_gap = g_gap * Σ (V_j - V_i)
//!   3. Объемная нейромодуляция (Volume Transmission): M_i = 1.0 + μ * Ca_i

/// Состояние сети астроцитов и щелевых контактов.
#[derive(Debug, Clone)]
pub struct AstrocyteState {
    /// Количество узлов глии/астроцитов
    pub n: usize,
    /// Концентрация кальция саркоплазмы [Ca²⁺]
    pub ca: Vec<f32>,
    /// Электрический потенциал щелевых контактов [mV]
    pub v_gap: Vec<f32>,
    /// Модуляция объемной передачи (Volume Transmission Gain) [1.0 .. 2.0]
    pub volume_gain: Vec<f32>,
    /// Постоянная времени выкачивания Ca²⁺ (мс)
    pub tau_astro_ms: f32,
    /// Проводимость щелевых контактов g_gap
    pub g_gap: f32,
    /// Коэффициент объемной модуляции μ
    pub mu: f32,
}

impl AstrocyteState {
    /// Создать новую сеть астроцитов.
    pub fn new(n: usize, tau_astro_ms: f32, g_gap: f32, mu: f32) -> Self {
        Self {
            n,
            ca: vec![0.1; n],
            v_gap: vec![-70.0; n],
            volume_gain: vec![1.0; n],
            tau_astro_ms,
            g_gap,
            mu,
        }
    }

    /// Сброс состояний.
    pub fn reset(&mut self) {
        self.ca.fill(0.1);
        self.v_gap.fill(-70.0);
        self.volume_gain.fill(1.0);
    }

    /// Один шаг медленной биоэлектрической кинетики астроцитов.
    pub fn step(&mut self, dt_ms: f32, nearby_spikes: &[i32], gap_neighbors: &[(usize, usize)]) {
        let n = self.n;
        let alpha = (-dt_ms / self.tau_astro_ms.max(1.0)).exp();

        // 1. Распространение электрического тока через Gap Junctions
        let mut i_gap = vec![0.0f32; n];
        for &(i, j) in gap_neighbors {
            if i < n && j < n {
                let diff = self.v_gap[j] - self.v_gap[i];
                let current = self.g_gap * diff;
                i_gap[i] += current;
                i_gap[j] -= current;
            }
        }

        // 2. Обновление V_gap и медленной волны Ca²⁺
        for i in 0..n {
            self.v_gap[i] += i_gap[i] * dt_ms * 0.1;

            let spike_input = if i < nearby_spikes.len() {
                nearby_spikes[i] as f32 * 0.2
            } else {
                0.0
            };

            // Медленное затухание + выброс Ca²⁺ от нейронных спайков
            self.ca[i] = self.ca[i] * alpha + (1.0 - alpha) * 0.1 + spike_input;

            // Volume transmission gain: M_i = 1.0 + μ * Ca_i
            self.volume_gain[i] = 1.0 + self.mu * self.ca[i];
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
    fn test_astrocyte_gap_junction_and_ca_wave() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.2, 0.5);
        let gap_pairs = vec![(0, 1), (1, 2), (2, 3)];

        // Подача потенциала на первый узел
        astro.v_gap[0] = -30.0;
        let spikes = vec![1, 0, 0, 0];

        // Симуляция 100 мс
        for _ in 0..100 {
            astro.step(1.0, &spikes, &gap_pairs);
        }

        // 1. Проверяем передачу электрической волны через Gap Junction (V_gap[1] деполяризовался)
        assert!(astro.v_gap[1] > -70.0);
        // 2. Проверяем рост Ca²⁺ волны и Volume Transmission gain
        assert!(astro.ca[0] > 0.1);
        assert!(astro.volume_gain[0] > 1.0);
    }
}
