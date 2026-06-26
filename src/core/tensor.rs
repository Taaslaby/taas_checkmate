//! Tensor — основной тип данных.
//! Все матричные операции через `kernels::linear_decoder` (SIMD f32x8).
//! Elementwise ops через SIMD ядра. Matmul через rayon.

use crate::kernels;
use rand::Rng;
use rayon::prelude::*;

// ─── Tensor ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Tensor: data.len != shape.product");
        Self { data, shape }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    pub fn from_slice(data: &[f32]) -> Self {
        Self::from_vec(data.to_vec())
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    /// Xavier init
    pub fn randn(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let n_in = shape[0];
        let n_out = shape.get(1).copied().unwrap_or(1);
        let scale = (6.0 / (n_in + n_out) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect();
        Self { data, shape: shape.to_vec() }
    }

    pub fn data(&self) -> &[f32] { &self.data }
    pub fn data_mut(&mut self) -> &mut [f32] { &mut self.data }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    // ─── Matmul через SIMD linear_decoder ──────────────────────────

    /// Умножение матриц: self @ other
    /// Использует `kernels::linear_decoder` (f32x8 SIMD)
    /// Для 2D: транспонирует other → [n×k], затем linear_decoder за 1 вызов на строку
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul: self must be 2D");
        let (m, k) = (self.shape[0], self.shape[1]);

        let (n, result_shape) = if other.shape.len() == 1 {
            assert_eq!(other.shape[0], k, "matmul: inner dim mismatch {} != {}", k, other.shape[0]);
            (1, vec![m])
        } else {
            assert_eq!(other.shape[0], k, "matmul: inner dim mismatch {} != {}", k, other.shape[0]);
            (other.shape[1], vec![m, other.shape[1]])
        };

        let mut result = vec![0.0f32; m * n];
        let zero_bias = vec![0.0f32; n]; // bias = 0 для matmul без смещения

        if n == 1 {
            // 1D: other = [k], self = [m × k]
            result.par_iter_mut().enumerate().for_each(|(i, res)| {
                kernels::linear_decoder(
                    std::slice::from_mut(res),
                    &self.data[i * k..(i + 1) * k],
                    &other.data,
                    &zero_bias,
                );
            });
        } else {
            // 2D: other = [k × n], транспонируем → [n × k]
            let other_t = other.transpose(); // [n × k]

            result.par_chunks_exact_mut(n).enumerate().for_each(|(i, row)| {
                kernels::linear_decoder(
                    row,
                    &self.data[i * k..(i + 1) * k],
                    &other_t.data,
                    &zero_bias,
                );
            });
        }

        Tensor::new(result, result_shape)
    }

    // ─── Add bias (SIMD) ───────────────────────────────────────────

    /// Add bias: result = self + bias. Использует SIMD f32x8.
    pub fn add(&self, bias: &Tensor) -> Tensor {
        assert_eq!(bias.shape.len(), 1, "add: bias must be 1D");
        let mut result = self.data.clone();
        let bias_len = bias.len();
        let n = result.len();

        if bias_len == n {
            kernels::add_simd(&mut result, &bias.data);
        } else {
            // Broadcasting: параллелим по полным чанкам bias_len
            result.par_chunks_exact_mut(bias_len).for_each(|chunk| {
                kernels::add_simd(chunk, &bias.data);
            });
            // Хвост (если n не кратен bias_len)
            let tail_start = (n / bias_len) * bias_len;
            for i in tail_start..n {
                result[i] += bias.data[i % bias_len];
            }
        }

        Tensor::new(result, self.shape.clone())
    }

    // ─── Activations (SIMD) ────────────────────────────────────────

    /// ReLU: max(0, x). Использует SIMD f32x8.
    pub fn relu(&self) -> Tensor {
        let mut result = self.data.clone();
        kernels::relu_simd(&mut result);
        Tensor::new(result, self.shape.clone())
    }

    /// Sigmoid через fast_exp
    pub fn sigmoid(&self) -> Tensor {
        self.elementwise(|x| 1.0 / (1.0 + kernels::fast_exp(-x)))
    }

    /// Softmax (по последней оси). Параллельно через rayon.
    pub fn softmax(&self) -> Tensor {
        let last_dim = self.shape[self.shape.len() - 1];
        let n = self.data.len();
        let mut result = vec![0.0f32; n];

        result.par_chunks_exact_mut(last_dim)
            .zip(self.data.par_chunks_exact(last_dim))
            .for_each(|(out_row, in_row)| {
                let max = in_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (i, &x) in in_row.iter().enumerate() {
                    let e = kernels::fast_exp(x - max);
                    out_row[i] = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;
                for v in out_row.iter_mut() { *v *= inv_sum; }
            });

        Tensor::new(result, self.shape.clone())
    }

    // ─── Fused matmul + bias ─────────────────────────────────────────

    /// Fused matmul + bias: self @ other + bias. Один проход, без промежуточного Tensor.
    /// Использует `linear_decoder` с переданным bias вместо нулевого.
    pub fn matmul_add(&self, other: &Tensor, bias: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul_add: self must be 2D");
        assert_eq!(bias.shape.len(), 1, "matmul_add: bias must be 1D");
        let (m, k) = (self.shape[0], self.shape[1]);

        let (n, result_shape) = if other.shape.len() == 1 {
            assert_eq!(other.shape[0], k);
            (1, vec![m])
        } else {
            assert_eq!(other.shape[0], k);
            (other.shape[1], vec![m, other.shape[1]])
        };

        assert_eq!(bias.len(), n, "matmul_add: bias.len {} != n {}", bias.len(), n);

        let mut result = vec![0.0f32; m * n];

        if n == 1 {
            result.par_iter_mut().enumerate().for_each(|(i, res)| {
                kernels::linear_decoder(
                    std::slice::from_mut(res),
                    &self.data[i * k..(i + 1) * k],
                    &other.data,
                    &bias.data,
                );
            });
        } else {
            let other_t = other.transpose();
            result.par_chunks_exact_mut(n).enumerate().for_each(|(i, row)| {
                kernels::linear_decoder(
                    row,
                    &self.data[i * k..(i + 1) * k],
                    &other_t.data,
                    &bias.data,
                );
            });
        }

        Tensor::new(result, result_shape)
    }

    // ─── Utility ───────────────────────────────────────────────────

    pub fn elementwise<F>(&self, f: F) -> Tensor
    where F: Fn(f32) -> f32 {
        let result: Vec<f32> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn scale(&self, s: f32) -> Tensor {
        self.elementwise(|x| x * s)
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "transpose: only 2D");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor::new(result, vec![cols, rows])
    }

    pub fn argmax(&self) -> Vec<usize> {
        let last_dim = self.shape[self.shape.len() - 1];
        let n = self.data.len();
        let mut result = Vec::with_capacity(n / last_dim);
        for chunk_start in (0..n).step_by(last_dim) {
            let chunk = &self.data[chunk_start..chunk_start + last_dim];
            let max_idx = chunk.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            result.push(max_idx);
        }
        result
    }
}

use std::fmt;
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?})", self.shape)
    }
}
