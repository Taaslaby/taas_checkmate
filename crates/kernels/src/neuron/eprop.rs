//! 🧠 ELIGIBILITY PROPAGATION (e-prop) SNN LEARNING KERNEL
//!
//! Реализация алгоритма обучения импульсных нейронных сетей (SNN) без
//! обратного прогона во времени (e-prop: Bellec et al., Nature 2020).
//!
//! Математика e-prop:
//!   - Суррогатный градиент спайкинга: ψ(V) = γ / (1 + |V - V_thresh|)
//!   - След следа пригодности (Eligibility Trace):
//!       e_ij(t+1) = α_e * e_ij(t) + ψ_i(t) * S_j(t)
//!   - Обновление синаптического веса w_ij по локальной ошибке err_i(t):
//!       Δw_ij = -η * err_i(t) * e_ij(t)

/// Состояние следов пригодности e-prop для синаптической матрицы.
#[derive(Debug, Clone)]
pub struct EPropState {
    /// Количество целевых (постсинаптических) нейронов.
    pub n_post: usize,
    /// Количество исходных (пресинаптических) нейронов.
    pub n_pre: usize,
    /// Следы пригодности e_ij [n_post * n_pre]
    pub eligibility: Vec<f32>,
    /// Затухание следа пригодности (alpha_e = exp(-dt / tau_e))
    pub alpha_e: f32,
    /// Коэффициент суррогатного градиента (gamma)
    pub gamma: f32,
    /// Порог спайкинга для суррогатного градиента
    pub v_thresh: f32,
    /// Скорость обучения (learning rate eta)
    pub eta: f32,
}

impl EPropState {
    /// Создать новый вектор следов пригодности e-prop.
    pub fn new(n_post: usize, n_pre: usize, tau_e_ms: f32, dt_ms: f32, eta: f32) -> Self {
        let alpha_e = (-dt_ms / tau_e_ms.max(1e-3)).exp();
        Self {
            n_post,
            n_pre,
            eligibility: vec![0.0; n_post * n_pre],
            alpha_e,
            gamma: 0.3,
            v_thresh: -55.0,
            eta,
        }
    }

    /// Сброс следов пригодности.
    pub fn reset(&mut self) {
        self.eligibility.fill(0.0);
    }

    /// Обновление следов пригодности и синаптических весов по ошибке err_post.
    pub fn update(
        &mut self,
        weights: &mut [f32],
        v_post: &[f32],
        spikes_pre: &[i32],
        errors_post: &[f32],
    ) {
        let n_post = self.n_post;
        let n_pre = self.n_pre;

        for i in 0..n_post {
            // Суррогатный градиент ψ_i(V)
            let v_diff = (v_post[i] - self.v_thresh).abs();
            let psi_i = self.gamma / (1.0 + v_diff);
            let err_i = errors_post[i];

            let row_off = i * n_pre;
            for j in 0..n_pre {
                let idx = row_off + j;
                let s_j = spikes_pre[j] as f32;

                // e_ij(t+1) = α_e * e_ij(t) + ψ_i(V) * S_j(t)
                let e_val = self.alpha_e * self.eligibility[idx] + psi_i * s_j;
                self.eligibility[idx] = e_val;

                // Δw_ij = -η * err_i * e_ij
                let delta_w = -self.eta * err_i * e_val;
                weights[idx] += delta_w;
            }
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
    fn test_eprop_eligibility_accumulation_and_decay() {
        let mut eprop = EPropState::new(2, 2, 20.0, 1.0, 0.01);
        let mut weights = vec![1.0; 4];
        let v_post = vec![-54.0, -70.0];
        let spikes_pre = vec![1, 0];
        let errors_post = vec![0.5, -0.2];

        // Шаг 1: Подача спайков
        eprop.update(&mut weights, &v_post, &spikes_pre, &errors_post);
        assert!(eprop.eligibility[0] > 0.0);
        assert!(weights[0] < 1.0); // Вес уменьшился из-за положительной ошибки

        // Шаг 2: Без спайков след затухает по alpha_e
        let e_prev = eprop.eligibility[0];
        let spikes_zero = vec![0, 0];
        eprop.update(&mut weights, &v_post, &spikes_zero, &vec![0.0, 0.0]);
        assert!(eprop.eligibility[0] < e_prev);
    }
}
