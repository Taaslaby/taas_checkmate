//! Equilibrium Propagation / Contrastive Hebbian Learning — «STDP, но градиентный».
//!
//! Идея (Антон): обратный проход — это вторая фаза той же «твинки» (пары
//! осцилляторов). Вес меняется на РАЗНИЦЕ активностей двух фаз:
//!
//! ```text
//! Δw ∝ pre · (post₂ − post₁)
//! ```
//!
//! Форма — как у STDP (`pre·post`), но в «post»-слоте стоит КОНТРАСТ двух фаз =
//! ошибка = градиент. Это Equilibrium Propagation (Scellier–Bengio 2017).
//!
//! Сеть — симметричная энергетическая (h ↔ y через одни и те же веса V):
//!
//! ```text
//! h = tanh(Wx + Vᵀ·y)
//! y = tanh(V·[1,h] + β·(t − y))    ← nudge: выход тянется к цели ОШИБКОЙ
//! ```
//!
//! Две фазы (релаксация до равновесия):
//! - free   (β=0):       → (h_free,  y_free)
//! - nudged (β>0, к цели) → (h_nudge, y_nudge)
//!
//! ```text
//! ΔW = (1/β)·(h_nudge·x − h_free·x)
//! ΔV = (1/β)·(y_nudge·[1,h_nudge] − y_free·[1,h_free])
//! ```
//!
//! Доказательство на XOR (линейно неразделим): контрастное правило сходится к
//! 100%, чистый Хебб (без контраста = без credit assignment) — нет.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn tanh(z: f64) -> f64 {
    z.tanh()
}

/// 2-слойная симметричная сеть: вход (2+bias) → скрытые (H, tanh) → выход (1, tanh).
///
/// `w` — (H × 3) row-major, `v` — (H+1,) [bias + веса h→y].
pub struct EqPropNet {
    pub n_hidden: usize,
    pub w: Vec<f64>, // (n_hidden, 3)
    pub v: Vec<f64>, // (n_hidden + 1)
}

impl EqPropNet {
    pub fn new(n_hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = (0..n_hidden * 3)
            .map(|_| rng.gen::<f64>() * 0.6 - 0.3)
            .collect();
        let v = (0..n_hidden + 1)
            .map(|_| rng.gen::<f64>() * 0.6 - 0.3)
            .collect();
        Self { n_hidden, w, v }
    }

    #[inline]
    fn w_at(&self, i: usize, j: usize) -> f64 {
        self.w[i * 3 + j]
    }

    /// h = tanh(Wx).
    fn hidden(&self, x: &[f64; 3]) -> Vec<f64> {
        (0..self.n_hidden)
            .map(|i| tanh(self.w_at(i, 0) * x[0] + self.w_at(i, 1) * x[1] + self.w_at(i, 2) * x[2]))
            .collect()
    }

    /// y = tanh(V·[1,h] + nudge).
    fn output_from(&self, h: &[f64], nudge: f64) -> f64 {
        let z = self.v[0]
            + h.iter()
                .zip(self.v.iter().skip(1))
                .map(|(hi, vi)| hi * vi)
                .sum::<f64>()
            + nudge;
        tanh(z)
    }

    /// Релаксация до равновесия. `target=None` → free; иначе nudge к цели.
    pub fn settle(&self, x: &[f64; 3], target: Option<f64>, beta: f64, t_steps: usize) -> (Vec<f64>, f64) {
        // Gauss–Seidel: сначала y (по предыдущему h), затем h (по новому y).
        let mut h = vec![0.0f64; self.n_hidden];
        let mut y = 0.0f64;
        for _ in 0..t_steps {
            let nudge = match target {
                Some(tg) => beta * (tg - y),
                None => 0.0,
            };
            y = self.output_from(&h, nudge);
            for i in 0..self.n_hidden {
                let z = self.w_at(i, 0) * x[0]
                    + self.w_at(i, 1) * x[1]
                    + self.w_at(i, 2) * x[2]
                    + self.v[i + 1] * y;
                h[i] = tanh(z);
            }
        }
        (h, y)
    }

    /// Контрастное правило: Δw ∝ pre·(post₂ − post₁).
    pub fn train_contrastive(&mut self, x_data: &[[f64; 2]], t: &[f64], lr: f64, beta: f64) {
        let mut dw = vec![0.0f64; self.w.len()];
        let mut dv = vec![0.0f64; self.v.len()];
        for (x2, &tg) in x_data.iter().zip(t) {
            let x = [1.0, x2[0], x2[1]];
            let (h_free, y_free) = self.settle(&x, None, 0.0, 200);
            let (h_nudge, y_nudge) = self.settle(&x, Some(tg), beta, 200);
            for i in 0..self.n_hidden {
                for j in 0..3 {
                    dw[i * 3 + j] += (h_nudge[i] - h_free[i]) * x[j];
                }
            }
            dv[0] += y_nudge - y_free;
            for i in 0..self.n_hidden {
                dv[i + 1] += y_nudge * h_nudge[i] - y_free * h_free[i];
            }
        }
        let scale = lr / beta / x_data.len() as f64;
        for i in 0..self.w.len() {
            self.w[i] += scale * dw[i];
        }
        for i in 0..self.v.len() {
            self.v[i] += scale * dv[i];
        }
    }

    /// Чистый Хебб Δw ∝ pre·post (без контраста = без ошибки).
    pub fn train_hebbian(&mut self, x_data: &[[f64; 2]], lr: f64, clip: f64) {
        for x2 in x_data {
            let x = [1.0, x2[0], x2[1]];
            let (h, y) = self.settle(&x, None, 0.0, 200);
            for i in 0..self.n_hidden {
                for j in 0..3 {
                    self.w[i * 3 + j] += lr * h[i] * x[j];
                }
            }
            self.v[0] += lr * y;
            for i in 0..self.n_hidden {
                self.v[i + 1] += lr * y * h[i];
            }
        }
        for wi in &mut self.w {
            *wi = wi.clamp(-clip, clip);
        }
        for vi in &mut self.v {
            *vi = vi.clamp(-clip, clip);
        }
    }

    /// Точный backprop-градиент dL/dW, dL/dV (loss = ½·Σ(t−y)²) — для сравнения направлений.
    pub fn backprop_grad(&self, x_data: &[[f64; 2]], t: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut dw = vec![0.0f64; self.w.len()];
        let mut dv = vec![0.0f64; self.v.len()];
        for (x2, &tg) in x_data.iter().zip(t) {
            let x = [1.0, x2[0], x2[1]];
            let h = self.hidden(&x);
            let y = self.output_from(&h, 0.0);
            let dy = (y - tg) * (1.0 - y * y);
            dv[0] += dy;
            for i in 0..self.n_hidden {
                dv[i + 1] += dy * h[i];
            }
            let mut dh = vec![0.0f64; self.n_hidden];
            for i in 0..self.n_hidden {
                dh[i] = self.v[i + 1] * dy * (1.0 - h[i] * h[i]);
            }
            for i in 0..self.n_hidden {
                for j in 0..3 {
                    dw[i * 3 + j] += dh[i] * x[j];
                }
            }
        }
        (dw, dv)
    }

    /// Предсказание: 0/1 (выход tanh → порог 0).
    pub fn predict(&self, x2: [f64; 2]) -> usize {
        let (_, y) = self.settle(&[1.0, x2[0], x2[1]], None, 0.0, 200);
        if y > 0.0 {
            1
        } else {
            0
        }
    }

    /// MSE loss = ½·⟨(t−y)²⟩.
    pub fn loss(&self, x_data: &[[f64; 2]], t: &[f64]) -> f64 {
        let mut l = 0.0;
        for (x2, &tg) in x_data.iter().zip(t) {
            let (_, y) = self.settle(&[1.0, x2[0], x2[1]], None, 0.0, 200);
            l += 0.5 * (tg - y) * (tg - y);
        }
        l / x_data.len() as f64
    }

    /// Доля верных предсказаний.
    pub fn accuracy(&self, x_data: &[[f64; 2]], t: &[usize]) -> f64 {
        let mut c = 0usize;
        for (x2, &tg) in x_data.iter().zip(t) {
            if self.predict(*x2) == tg {
                c += 1;
            }
        }
        c as f64 / x_data.len() as f64
    }
}

/// Косинус угла между двумя (развёрнутыми) градиентами.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eqprop_solves_xor() {
        let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let t_usize = [0usize, 1, 1, 0];
        let t: Vec<f64> = t_usize.iter().map(|&v| 2.0 * v as f64 - 1.0).collect();
        let mut net = EqPropNet::new(8, 0);
        for _ in 0..3000 {
            net.train_contrastive(&x, &t, 0.02, 0.5);
        }
        assert!(
            net.accuracy(&x, &t_usize) > 0.99,
            "EqProp должен решить XOR, acc={}",
            net.accuracy(&x, &t_usize)
        );
    }

    #[test]
    fn test_contrast_aligns_with_gradient() {
        let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let t: Vec<f64> = [0usize, 1, 1, 0].iter().map(|&v| 2.0 * v as f64 - 1.0).collect();
        let net = EqPropNet::new(8, 1);
        let (dw_bp, _dv_bp) = net.backprop_grad(&x, &t);

        let mut dw_ch = vec![0.0f64; net.w.len()];
        for (x2, &tg) in x.iter().zip(&t) {
            let xv = [1.0, x2[0], x2[1]];
            let (h_free, _) = net.settle(&xv, None, 0.0, 200);
            let (h_nudge, _) = net.settle(&xv, Some(tg), 0.5, 200);
            for i in 0..net.n_hidden {
                for j in 0..3 {
                    dw_ch[i * 3 + j] += (h_nudge[i] - h_free[i]) * xv[j];
                }
            }
        }
        let c = cosine(&dw_ch, &dw_bp);
        assert!(
            c.abs() > 0.9,
            "контраст должен смотреть вдоль градиента, cos={c}"
        );
    }
}
