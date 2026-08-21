//! 🧠 TrainableSpikeSource — дифференцируемый спайковый источник для SPIGRAD.
//!
//! Математика идентична `TransformerModel` из pillars zoo (softmax-внимание
//! над 8 слотами памяти + leaky-интегратор `acc`), НО без встроенного STDP:
//! обучение идёт полным BPTT извне — градиенты `dL/dw`, `dL/dv` приходят из
//! функции потерь через мост [`crate::spike_grad::SpikeGradBridge`].
//!
//! ```text
//! features[t,n] → TrainableSpikeSource(w,v) → spikes[t,n] (rate-code acc)
//!      ▲                                                    │
//!      └──── dL/dw, dL/dv ◀── backward(grad_spikes) ◀───────┘
//!                        ▲
//!                        └── grad_spikes = SpikeGradBridge::backward_full().2
//! ```
//!
//! Выход источника — непрерывное значение `acc` (rate-code), поэтому вся
//! цепочка гладкая: никаких surrogate-градиентов, как и в самом SPIGRAD.
//! Пороговый спайк в движке остаётся форматом передачи; для обучения
//! используется дифференцируемая амплитуда.
//!
//! Forward (шаг `t`, нейрон `d`):
//! ```text
//! q        = features[t,d]
//! score_i  = clamp(q · w[d,i] · k_scale, ±50)
//! a_i      = softmax(score)
//! out      = Σ_i a_i · v[d,i] · mem_i
//! pre      = acc + dt·(out − acc·inv_tau)
//! acc      = max(pre, 0)                     ← «спайк» = acc (rate-code)
//! mem      = сдвиг с затуханием
//! ```
//!
//! Backward (шаг `t`, нейрон `d`) — `grad_spikes[t,d]` из моста:
//! ```text
//! dacc     = grad_spikes[t,d] + dacc_next
//! dpre     = (pre > 0) ? dacc : 0
//! dacc_next= dpre · (1 − dt·inv_tau)
//! dout     = dpre · dt
//! dv[d,i] += dout · a_i · mem_i
//! ds[i]    = a_i · (da_i − Σ a_j·da_j)       ← softmax-Jakobian
//! dw[d,i] += ds[i] · q · k_scale             (если не клампнут)
//! ```

use rand::Rng;
use rand::SeedableRng;

/// Число слотов памяти (совпадает с движковым TransformerModel).
pub const SLOTS: usize = 8;
/// Кламп score до exp: защита от переполнения f32.
const SCORE_CLAMP: f32 = 50.0;

/// Запись одного шага одного нейрона — всё, что нужно для backward.
#[derive(Clone, Copy, Default)]
pub struct SpikeStepRec {
    /// Вход признака q = features[t,d]
    pub q: f32,
    /// Внимание a_i = softmax(score)
    pub a: [f32; SLOTS],
    /// Память ДО сдвига (на момент чтения)
    pub mem: [f32; SLOTS],
    /// pre = acc + dt·(out − acc·inv_tau) — для ReLU-гейта
    pub pre: f32,
    /// Кламп score (градиент w обнуляется)
    pub clamp: [bool; SLOTS],
}

/// Дифференцируемый спайковый источник: [n_neurons × SLOTS] параметров w/v.
///
/// `features` — `[n_steps × n_neurons]` row-major; выход `spikes` — тоже
/// `[n_steps × n_neurons]` (rate-code `acc`).
pub struct TrainableSpikeSource {
    /// Ключи внимания: [n_neurons × SLOTS]
    pub w: Vec<f32>,
    /// Значения: [n_neurons × SLOTS]
    pub v: Vec<f32>,
    pub n_neurons: usize,
    pub dt: f32,
    pub decay: f32,
    pub gain: f32,
    pub inv_tau: f32,
    pub k_scale: f32,
}

impl TrainableSpikeSource {
    /// Проверка конфигурации.
    fn assert_layout(&self) {
        assert!(self.n_neurons > 0, "TrainableSpikeSource: n_neurons must be > 0");
        assert_eq!(
            self.w.len(),
            self.n_neurons * SLOTS,
            "TrainableSpikeSource: w storage length mismatch"
        );
        assert_eq!(
            self.v.len(),
            self.n_neurons * SLOTS,
            "TrainableSpikeSource: v storage length mismatch"
        );
        assert!(
            self.dt.is_finite()
                && self.decay.is_finite()
                && self.gain.is_finite()
                && self.inv_tau.is_finite()
                && self.k_scale.is_finite(),
            "TrainableSpikeSource: parameters must be finite"
        );
    }

    /// Новый источник: w=v=1 (как движковый TransformerModel на старте).
    pub fn new(n_neurons: usize, dt: f32, decay: f32, gain: f32, inv_tau: f32, k_scale: f32) -> Self {
        assert!(n_neurons > 0, "TrainableSpikeSource: n_neurons must be > 0");
        Self {
            w: vec![1.0; n_neurons * SLOTS],
            v: vec![1.0; n_neurons * SLOTS],
            n_neurons,
            dt,
            decay,
            gain,
            inv_tau,
            k_scale,
        }
    }

    /// Случайный источник (для учителя в тестах/примерах).
    pub fn new_random(
        n_neurons: usize,
        dt: f32,
        decay: f32,
        gain: f32,
        inv_tau: f32,
        k_scale: f32,
        seed: u64,
    ) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut src = Self::new(n_neurons, dt, decay, gain, inv_tau, k_scale);
        for x in src.w.iter_mut() {
            *x = 0.5 + rng.gen::<f32>(); // [0.5, 1.5]
        }
        for x in src.v.iter_mut() {
            *x = 0.5 + rng.gen::<f32>();
        }
        src
    }

    /// Forward: `features` [n_steps × n_neurons] → `spikes` [n_steps × n_neurons].
    ///
    /// Возвращает (spikes, записи шагов для backward). Память и acc между
    /// последовательностями НЕ переносятся — каждый вызов начинается с нуля.
    pub fn forward(&self, features: &[f32], n_steps: usize) -> (Vec<f32>, Vec<SpikeStepRec>) {
        self.assert_layout();
        let n = self.n_neurons;
        assert_eq!(
            features.len(),
            n_steps * n,
            "TrainableSpikeSource::forward: features must be n_steps × n_neurons"
        );
        assert!(features.iter().all(|x| x.is_finite()), "features must be finite");

        let mut spikes = vec![0.0f32; n_steps * n];
        let mut recs = vec![SpikeStepRec::default(); n_steps * n];

        for d in 0..n {
            let mut mem = [0.0f32; SLOTS];
            let mut acc = 0.0f32;
            for t in 0..n_steps {
                let q = features[t * n + d];
                let mut e = [0.0f32; SLOTS];
                let mut clamp = [false; SLOTS];
                let mut esum = 0.0f32;
                for i in 0..SLOTS {
                    let raw = q * self.w[d * SLOTS + i] * self.k_scale;
                    if raw > SCORE_CLAMP {
                        e[i] = SCORE_CLAMP.exp();
                        clamp[i] = true;
                    } else if raw < -SCORE_CLAMP {
                        e[i] = (-SCORE_CLAMP).exp();
                        clamp[i] = true;
                    } else {
                        e[i] = raw.exp();
                    }
                    esum += e[i];
                }
                let mut a = [0.0f32; SLOTS];
                for i in 0..SLOTS {
                    a[i] = e[i] / esum;
                }

                let mut out = 0.0f32;
                for (i, &ai) in a.iter().enumerate() {
                    out += ai * self.v[d * SLOTS + i] * mem[i];
                }
                let pre = acc + self.dt * (out - acc * self.inv_tau);
                acc = if pre > 0.0 { pre } else { 0.0 };
                spikes[t * n + d] = acc;

                recs[t * n + d] = SpikeStepRec { q, a, mem, pre, clamp };

                for i in (1..SLOTS).rev() {
                    mem[i] = mem[i - 1] * self.decay;
                }
                mem[0] = q * self.gain;
            }
        }
        (spikes, recs)
    }

    /// Backward: `grad_spikes` [n_steps × n_neurons] → (dw, dv).
    ///
    /// Формулы соответствуют `Net::backward` из examples/mnist_grad.rs
    /// (там же проверены конечными разностями). Отличие: grad приходит
    /// по-шагово из моста (dL/dacc_t), а не общий на сумму признаков.
    pub fn backward(
        &self,
        grad_spikes: &[f32],
        n_steps: usize,
        recs: &[SpikeStepRec],
    ) -> (Vec<f32>, Vec<f32>) {
        self.assert_layout();
        let n = self.n_neurons;
        assert_eq!(
            grad_spikes.len(),
            n_steps * n,
            "TrainableSpikeSource::backward: grad_spikes must be n_steps × n_neurons"
        );
        assert_eq!(
            recs.len(),
            n_steps * n,
            "TrainableSpikeSource::backward: records length mismatch"
        );
        assert!(
            grad_spikes.iter().all(|x| x.is_finite()),
            "grad_spikes must be finite"
        );

        let mut dw = vec![0.0f32; n * SLOTS];
        let mut dv = vec![0.0f32; n * SLOTS];

        for d in 0..n {
            let mut dacc_next = 0.0f32;
            for t in (0..n_steps).rev() {
                let r = &recs[t * n + d];
                let dacc = grad_spikes[t * n + d] + dacc_next;
                let dpre = if r.pre > 0.0 { dacc } else { 0.0 };
                dacc_next = dpre * (1.0 - self.dt * self.inv_tau);
                let dout = dpre * self.dt;

                let base = d * SLOTS;
                let mut da = [0.0f32; SLOTS];
                for (i, da_i) in da.iter_mut().enumerate() {
                    dv[base + i] += dout * r.a[i] * r.mem[i];
                    *da_i = dout * self.v[base + i] * r.mem[i];
                }
                let mut dot = 0.0f32;
                for (i, &ai) in r.a.iter().enumerate() {
                    dot += ai * da[i];
                }
                for (i, &ai) in r.a.iter().enumerate() {
                    let ds = ai * (da[i] - dot);
                    if !r.clamp[i] {
                        dw[base + i] += ds * r.q * self.k_scale;
                    }
                }
            }
        }
        (dw, dv)
    }

    /// Adam-обновление w/v (все градиенты в одном плоском векторе).
    pub fn update_adam(
        &mut self,
        dw: &[f32],
        dv: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        t: u32,
        m_w: &mut [f32],
        v_w: &mut [f32],
        m_v: &mut [f32],
        v_v: &mut [f32],
    ) {
        self.assert_layout();
        let np = self.n_neurons * SLOTS;
        assert_eq!(dw.len(), np, "TrainableSpikeSource: dw length mismatch");
        assert_eq!(dv.len(), np, "TrainableSpikeSource: dv length mismatch");
        assert_eq!(m_w.len(), np);
        assert_eq!(v_w.len(), np);
        assert_eq!(m_v.len(), np);
        assert_eq!(v_v.len(), np);
        assert!(lr.is_finite() && lr >= 0.0 && t > 0);

        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        for i in 0..np {
            m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * dw[i];
            v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * dw[i] * dw[i];
            let mh = m_w[i] / bc1;
            let vh = v_w[i] / bc2;
            self.w[i] -= lr * mh / (vh.sqrt() + eps);

            m_v[i] = beta1 * m_v[i] + (1.0 - beta1) * dv[i];
            v_v[i] = beta2 * v_v[i] + (1.0 - beta2) * dv[i] * dv[i];
            let mh = m_v[i] / bc1;
            let vh = v_v[i] / bc2;
            self.v[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests: gradient check конечными разностями
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn structured_features(n_neurons: usize, n_steps: usize) -> Vec<f32> {
        let mut f = vec![0.0f32; n_steps * n_neurons];
        for t in 0..n_steps {
            for d in 0..n_neurons {
                // Периодический паттерн со сдвигом по нейронам — невырожденный
                let phase = t as f32 * 0.35 + d as f32 * 0.77;
                f[t * n_neurons + d] = 0.3 + 0.8 * (phase.sin() * 0.5 + 0.5);
            }
        }
        f
    }

    /// Суммарная «потеря»: Σ_t acc_t (как признак в mnist_grad) — ловит
    /// все пути градиента через интегратор.
    fn loss_of(src: &TrainableSpikeSource, features: &[f32], n_steps: usize) -> f32 {
        let (spikes, _) = src.forward(features, n_steps);
        spikes.iter().sum()
    }

    #[test]
    fn test_source_forward_shapes() {
        let src = TrainableSpikeSource::new(4, 0.5, 0.9, 3.0, 0.2, 2.0);
        let f = structured_features(4, 16);
        let (spikes, recs) = src.forward(&f, 16);
        assert_eq!(spikes.len(), 16 * 4);
        assert_eq!(recs.len(), 16 * 4);
        assert!(spikes.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(spikes.iter().any(|x| *x > 0.0), "must produce nonzero acc");
    }

    #[test]
    fn test_source_backward_gradient_check() {
        let src = TrainableSpikeSource::new(4, 0.5, 0.9, 3.0, 0.2, 2.0);
        let n_steps = 12;
        let f = structured_features(4, n_steps);
        // dL/dacc_t = 1 (потеря = сумма acc)
        let grad_spikes = vec![1.0f32; n_steps * 4];
        let (_, recs) = src.forward(&f, n_steps);
        let (dw, dv) = src.backward(&grad_spikes, n_steps, &recs);

        let eps = 1e-2f32;
        let mut max_rel = 0.0f32;
        let mut nonzero = 0usize;

        // w и v по одному атому
        let mut src_p = src;
        for &(i, is_w) in &[
            (0usize, true),
            (3, true),
            (7, true),
            (12, true),
            (20, false),
            (29, false),
        ] {
            let analytic = if is_w { dw[i] } else { dv[i] };
            if is_w {
                let orig = src_p.w[i];
                src_p.w[i] = orig + eps;
                let lp = loss_of(&src_p, &f, n_steps);
                src_p.w[i] = orig - eps;
                let lm = loss_of(&src_p, &f, n_steps);
                src_p.w[i] = orig;
                let numeric = (lp - lm) / (2.0 * eps);
                let rel = (analytic - numeric).abs() / (analytic.abs() + 1e-4);
                if analytic.abs() > 1e-4 {
                    max_rel = max_rel.max(rel);
                }
                if analytic.abs() > 1e-6 {
                    nonzero += 1;
                }
            } else {
                let orig = src_p.v[i];
                src_p.v[i] = orig + eps;
                let lp = loss_of(&src_p, &f, n_steps);
                src_p.v[i] = orig - eps;
                let lm = loss_of(&src_p, &f, n_steps);
                src_p.v[i] = orig;
                let numeric = (lp - lm) / (2.0 * eps);
                let rel = (analytic - numeric).abs() / (analytic.abs() + 1e-4);
                if analytic.abs() > 1e-4 {
                    max_rel = max_rel.max(rel);
                }
                if analytic.abs() > 1e-6 {
                    nonzero += 1;
                }
            }
        }
        assert!(nonzero >= 4, "degenerate gradient check: only {nonzero} nonzero");
        assert!(
            max_rel < 0.05,
            "source gradient check FAILED: max rel err {max_rel:.4}"
        );
    }

    #[test]
    fn test_source_adam_updates() {
        let mut src = TrainableSpikeSource::new(2, 0.5, 0.9, 3.0, 0.2, 2.0);
        let np = 2 * SLOTS;
        let w_before = src.w.clone();
        let v_before = src.v.clone();
        let mut m_w = vec![0.0; np];
        let mut v_w = vec![0.0; np];
        let mut m_v = vec![0.0; np];
        let mut v_v = vec![0.0; np];
        src.update_adam(
            &vec![0.1; np],
            &vec![0.05; np],
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
            &mut m_w,
            &mut v_w,
            &mut m_v,
            &mut v_v,
        );
        assert!(src.w.iter().zip(&w_before).any(|(a, b)| (a - b).abs() > 1e-8));
        assert!(src.v.iter().zip(&v_before).any(|(a, b)| (a - b).abs() > 1e-8));
    }
}
