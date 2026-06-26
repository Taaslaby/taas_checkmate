//! Нейросетевые слои — Dense, Softmax, Dropout, BatchNorm.
//! Izhikevich/LIF через скопированные SIMD ядра.

use crate::core::tensor::Tensor;
use crate::kernels;
use rand::Rng;

// ─── Активации ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU, Sigmoid, Tanh, Izhikevich, LIF, Linear,
}

// ─── Dense слой ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Dense {
    pub weight: Tensor,
    pub bias: Tensor,
    pub activation: Activation,
    pub n_in: usize,
    pub n_out: usize,
}

impl Dense {
    pub fn new(n_in: usize, n_out: usize, activation: Activation) -> Self {
        Self {
            weight: Tensor::randn(&[n_out, n_in]),
            bias: Tensor::zeros(&[n_out]),
            activation, n_in, n_out,
        }
    }

    /// Forward: y = activation(W·x + b)
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let h = input.matmul(&self.weight.transpose()).add(&self.bias);
        match self.activation {
            Activation::ReLU => h.relu(),
            Activation::Sigmoid => h.sigmoid(),
            Activation::Tanh => h.elementwise(|x| x.tanh()),
            Activation::Izhikevich => izhikevich_forward(&h),
            Activation::LIF => lif_forward(&h),
            Activation::Linear => h,
        }
    }
}

// ─── Izhikevich forward через SIMD ядро ────────────────────────────────────

fn izhikevich_forward(input: &Tensor) -> Tensor {
    let n = input.len();
    let mut v = vec![-70.0f32; n];
    let mut u = vec![-14.0f32; n];
    let mut trace = vec![0.0f32; n];
    let mut refractory = vec![0.0f32; n];
    let mut spikes_i32 = vec![0i32; n];
    let mut data = input.data().to_vec();

    // Per-neuron params (scalar, одинаковые для всех нейронов)
    let a = vec![0.02f32; n];   // a
    let b = vec![0.2f32; n];    // b
    let c = vec![-65.0f32; n];  // c (reset potential)
    let d = vec![8.0f32; n];    // d

    kernels::izhikevich_step(
        &mut v, &mut u, &mut trace,
        &a, &b, &data,
        0.5, 0.0,
        &mut spikes_i32, &c, &d,
        &mut refractory, 0.0,
    );

    // i32 → f32 для совместимости с autograd
    let spikes: Vec<f32> = spikes_i32.iter().map(|&s| s as f32).collect();
    Tensor::new(spikes, input.shape().to_vec())
}

// ─── LIF forward через SIMD ядро ───────────────────────────────────────────

fn lif_forward(input: &Tensor) -> Tensor {
    let n = input.len();
    let mut v = vec![-70.0f32; n];
    let mut refractory = vec![0.0f32; n];
    let mut spikes_i32 = vec![0i32; n];
    let mut Ii = input.data().to_vec();

    // Params flat array: dt, tau_m, v_rest, v_thresh, v_reset (5 × n)
    let dt = 0.5;
    let mut params = Vec::with_capacity(5 * n);
    for _ in 0..n {
        params.push(dt);            // dt
        params.push(10.0);           // tau_m
        params.push(-70.0);          // v_rest
        params.push(-55.0);          // v_thresh
        params.push(-75.0);          // v_reset
    }

    kernels::lif_step(
        &mut v, &mut Ii, &mut spikes_i32, &params,
        &mut refractory, 0.0,
    );

    let spikes: Vec<f32> = spikes_i32.iter().map(|&s| s as f32).collect();
    Tensor::new(spikes, input.shape().to_vec())
}

// ─── Softmax ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self { Self }
    pub fn forward(&self, input: &Tensor) -> Tensor { input.softmax() }
}

// ─── Dropout ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Dropout {
    pub rate: f32,
}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        assert!(rate >= 0.0 && rate < 1.0, "Dropout rate must be in [0, 1)");
        Self { rate }
    }

    pub fn forward(&self, input: &Tensor, training: bool) -> Tensor {
        if !training || self.rate == 0.0 { return input.clone(); }
        let scale = 1.0 / (1.0 - self.rate);
        let mut rng = rand::thread_rng();
        let result: Vec<f32> = input.data().iter()
            .map(|&x| if rng.gen::<f32>() < self.rate { 0.0 } else { x * scale })
            .collect();
        Tensor::new(result, input.shape().to_vec())
    }
}

// ─── BatchNorm ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    n_features: usize,
}

impl BatchNorm {
    pub fn new(n_features: usize) -> Self {
        Self {
            gamma: Tensor::from_vec(vec![1.0; n_features]),
            beta: Tensor::zeros(&[n_features]),
            n_features,
        }
    }

    pub fn forward(&self, input: &Tensor, training: bool) -> Tensor {
        let n = input.len();
        let c = self.n_features;
        let batch_size = n / c;

        let mut mean = vec![0.0f32; c];
        for j in 0..c {
            let mut sum = 0.0;
            for i in 0..batch_size { sum += input.data()[i * c + j]; }
            mean[j] = sum / batch_size as f32;
        }

        let mut var = vec![0.0f32; c];
        for j in 0..c {
            let mut sum = 0.0;
            for i in 0..batch_size {
                let d = input.data()[i * c + j] - mean[j];
                sum += d * d;
            }
            var[j] = sum / batch_size as f32;
        }

        let mut result = vec![0.0f32; n];
        for i in 0..n {
            let j = i % c;
            let normalized = (input.data()[i] - mean[j]) / (var[j] + 1e-5).sqrt();
            result[i] = self.gamma.data()[j] * normalized + self.beta.data()[j];
        }

        Tensor::new(result, input.shape().to_vec())
    }
}

// ─── Layer trait ───────────────────────────────────────────────────────────

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor { Dense::forward(self, input) }
}

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor { Softmax::forward(self, input) }
}

impl Layer for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor { Dropout::forward(self, input, true) }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &Tensor) -> Tensor { BatchNorm::forward(self, input, true) }
}
