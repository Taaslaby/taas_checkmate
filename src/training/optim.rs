//! Оптимизаторы — SGD и Adam для обновления весов.
//! Используют SIMD f32x8 ядра из kernels::signal для ~8× ускорения.

use crate::core::tensor::Tensor;
use crate::kernels;
use std::collections::HashMap;

// ─── SGD ───────────────────────────────────────────────────────────────────

pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    velocities: HashMap<String, Tensor>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr, momentum: 0.0, velocities: HashMap::new() }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn step(&mut self, weight: &mut Tensor, grad: &Tensor, key: &str) {
        let n = weight.len();
        if self.momentum > 0.0 {
            let vel = self.velocities
                .entry(key.to_string())
                .or_insert_with(|| Tensor::zeros(&[n]));
            unsafe {
                kernels::sgd_step_simd(
                    vel.data_mut().as_mut_ptr(),
                    weight.data_mut().as_mut_ptr(),
                    grad.data().as_ptr(),
                    self.lr, self.momentum, n,
                );
            }
        } else {
            // plain SGD через SIMD (momentum=0)
            unsafe {
                let zeros = weight.data().to_vec(); // dummy — не используется при momentum=0
                kernels::sgd_step_simd(
                    zeros.as_ptr() as *mut f32, // dummy velocity, не используется
                    weight.data_mut().as_mut_ptr(),
                    grad.data().as_ptr(),
                    self.lr, 0.0, n,
                );
            }
        }
    }
}

// ─── Adam ──────────────────────────────────────────────────────────────────

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: usize,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: HashMap::new(), v: HashMap::new() }
    }

    /// Вызывается раз за батч (не раз за параметр!)
    pub fn new_batch(&mut self) {
        self.t += 1;
    }

    pub fn step(&mut self, weight: &mut Tensor, grad: &Tensor, key: &str) {
        let n = weight.len();
        let m = self.m.entry(key.to_string()).or_insert_with(|| Tensor::zeros(&[n]));
        let v = self.v.entry(key.to_string()).or_insert_with(|| Tensor::zeros(&[n]));
        let beta1_t = self.beta1.powi(self.t as i32);
        let beta2_t = self.beta2.powi(self.t as i32);
        let beta1_t_corr = 1.0 - beta1_t;
        let beta2_t_corr = 1.0 - beta2_t;

        unsafe {
            kernels::adam_step_simd(
                weight.data_mut().as_mut_ptr(),
                m.data_mut().as_mut_ptr(),
                v.data_mut().as_mut_ptr(),
                grad.data().as_ptr(),
                self.lr, self.beta1, self.beta2, self.eps,
                beta1_t_corr, beta2_t_corr, n,
            );
        }
    }

    pub fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}
