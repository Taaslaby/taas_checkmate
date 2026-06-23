//! Функции потерь и метрики.

use crate::core::tensor::Tensor;

// ─── CrossEntropyLoss ──────────────────────────────────────────────────────

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self { Self }

    pub fn forward(&self, pred: &Tensor, target: &[usize]) -> f32 {
        let n = pred.shape()[0];
        let n_classes = pred.shape()[1];
        let mut loss = 0.0;
        for i in 0..n {
            let t = target[i];
            let p = pred.data()[i * n_classes + t].max(1e-15);
            loss -= p.ln();
        }
        loss / n as f32
    }

    pub fn backward(&self, pred: &Tensor, target: &[usize]) -> Tensor {
        let n = pred.shape()[0];
        let n_classes = pred.shape()[1];
        let mut grad = vec![0.0; n * n_classes];
        for i in 0..n {
            let t = target[i];
            for c in 0..n_classes {
                let one_hot = if c == t { 1.0 } else { 0.0 };
                grad[i * n_classes + c] = (pred.data()[i * n_classes + c] - one_hot) / n as f32;
            }
        }
        Tensor::new(grad, pred.shape().to_vec())
    }
}

// ─── MSELoss ───────────────────────────────────────────────────────────────

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self { Self }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> f32 {
        let mut loss = 0.0;
        for i in 0..pred.len() {
            let d = pred.data()[i] - target.data()[i];
            loss += d * d;
        }
        loss / pred.len() as f32
    }

    pub fn backward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let n = pred.len() as f32;
        let grad: Vec<f32> = pred.data().iter()
            .zip(target.data().iter())
            .map(|(p, t)| 2.0 * (p - t) / n)
            .collect();
        Tensor::new(grad, pred.shape().to_vec())
    }
}

// ─── Accuracy ──────────────────────────────────────────────────────────────

pub fn accuracy(pred: &Tensor, target: &[usize]) -> f32 {
    let pred_classes = pred.argmax();
    let correct = pred_classes.iter().zip(target.iter()).filter(|(p, t)| p == t).count();
    correct as f32 / target.len() as f32
}
