//! # BabyGrad Autograd 🔄
//!
//! **Automatic differentiation** как в PyTorch.
//! Каждая операция строит граф вычислений, `backward()` считает градиенты через chain rule.
//!
//! ## Пример
//! ```rust
//! use taas_checkmate::core::autograd::Var;
//!
//! let x = Var::new(vec![2.0, 3.0], vec![1, 2]);
//! let w = Var::randn(&[2, 2]).requires_grad(true);
//! let b = Var::zeros(&[2]).requires_grad(true);
//!
//! let h = x.matmul(&w).add(&b).relu();
//! let loss = h.softmax().cross_entropy(&[0]);
//! loss.backward();
//!
//! let dw = w.grad().unwrap();  // [∂loss/∂w]
//! ```

use std::rc::Rc;
use std::cell::RefCell;
use crate::kernels;
use rand::Rng;

// ─── Типы ──────────────────────────────────────────────────────────────────

type VarRef = Rc<RefCell<VarData>>;

/// Переменная с графом вычислений для autograd.
#[derive(Clone)]
pub struct Var {
    inner: VarRef,
}

/// Внутренние данные переменной.
pub struct VarData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,
    pub op: Option<Op>,
    pub parents: Vec<VarRef>,
}

// ─── Операции графа ────────────────────────────────────────────────────────

/// Тип операции, которая создала эту переменную.
pub enum Op {
    Matmul {
        a: VarRef,
        b: VarRef,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    Add {
        a: VarRef,
        b: VarRef,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    Relu {
        input: VarRef,
    },
    Sigmoid {
        input: VarRef,
    },
    Tanh {
        input: VarRef,
    },
    Softmax {
        input: VarRef,
    },
    Scale {
        input: VarRef,
        factor: f32,
    },
    CrossEntropy {
        pred: VarRef,
        target: Vec<usize>,
    },
    Transpose {
        input: VarRef,
        orig_shape: Vec<usize>,
    },
}

// ─── Конструкторы Var ──────────────────────────────────────────────────────

impl Var {
    fn new_inner(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Var: data.len != shape.product");
        Var {
            inner: Rc::new(RefCell::new(VarData {
                data, shape, grad: None, requires_grad,
                op: None, parents: vec![],
            })),
        }
    }

    /// Создать Var из данных и формы.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new_inner(data, shape, false)
    }

    /// Создать 1D Var из вектора.
    pub fn from_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        Self::new_inner(data, vec![len], false)
    }

    /// Создать Var из среза.
    pub fn from_slice(data: &[f32]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// Var заполненный нулями.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::new_inner(vec![0.0; n], shape.to_vec(), false)
    }

    /// Var с Xavier инициализацией (для sigmoid/tanh).
    pub fn randn(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let n_in = shape[0];
        let n_out = shape.get(1).copied().unwrap_or(1);
        let scale = (6.0 / (n_in + n_out) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect();
        Self::new_inner(data, shape.to_vec(), false)
    }

    /// Var с Kaiming (He) инициализацией (для ReLU).
    /// scale = √(2 / n_in), uniform в [-scale, scale]
    pub fn kaiming(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let n_in = shape[0];
        let scale = (2.0 / n_in as f32).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect();
        Self::new_inner(data, shape.to_vec(), false)
    }



    /// Включить/выключить requires_grad.
    pub fn requires_grad(self, val: bool) -> Self {
        self.inner.borrow_mut().requires_grad = val;
        self
    }
}

// ─── Доступ к данным ──────────────────────────────────────────────────────

impl Var {
    pub fn data(&self) -> Vec<f32> {
        self.inner.borrow().data.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.borrow().grad.clone()
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.borrow().data.is_empty()
    }

    /// Поменять данные in-place (для optim.step)
    pub fn set_data(&self, new_data: Vec<f32>) {
        self.inner.borrow_mut().data = new_data;
    }

    pub fn requires_grad_val(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Сбросить градиент (нужно между шагами обучения)
    pub fn clear_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }
}

// ─── Операции (forward + строят граф) ──────────────────────────────────────

impl Var {
    /// Умножение матриц: self @ other (SIMD f32x8)
    /// Поддерживает 1D @ 2D (вектор на матрицу) и 2D @ 1D (матрица на вектор)
    pub fn matmul(&self, other: &Var) -> Var {
        let a_shape = self.inner.borrow().shape.clone();
        let b_shape = other.inner.borrow().shape.clone();

        // Определяем размерности (1D → трактуем как batch=1)
        let a_is_1d = a_shape.len() == 1;
        let b_is_1d = b_shape.len() == 1;

        let result_data = self.matmul_raw(&other.inner.borrow().data, &a_shape, &b_shape);

        let result_shape = if b_is_1d {
            if a_is_1d { vec![1] } else { vec![a_shape[0]] }
        } else {
            if a_is_1d { vec![b_shape[1]] } else { vec![a_shape[0], b_shape[1]] }
        };

        let requires = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, result_shape, false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Matmul {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_shape: a_shape.clone(),
                b_shape: b_shape.clone(),
            });
            r.parents = vec![self.inner.clone(), other.inner.clone()];
        }

        result
    }

    /// Сложение: self + other (с broadcast)
    pub fn add(&self, other: &Var) -> Var {
        let a_shape = self.inner.borrow().shape.clone();
        let b_shape = other.inner.borrow().shape.clone();
        let a_data = self.inner.borrow().data.clone();
        let b_data = other.inner.borrow().data.clone();

        let result_data: Vec<f32> = if b_shape.len() == 1 && a_shape.len() == 2 {
            // bias broadcast: [m, n] + [n]
            let (m, n) = (a_shape[0], a_shape[1]);
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = a_data[i * n + j] + b_data[j];
                }
            }
            out
        } else {
            a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect()
        };

        let requires = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, a_shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Add {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_shape: a_shape.clone(),
                b_shape: b_shape.clone(),
            });
            r.parents = vec![self.inner.clone(), other.inner.clone()];
        }

        result
    }

    /// ReLU активация
    pub fn relu(&self) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        let result_data: Vec<f32> = data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Relu { input: self.inner.clone() });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// Sigmoid активация (через fast_exp)
    pub fn sigmoid(&self) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        let result_data: Vec<f32> = data.iter()
            .map(|&x| 1.0 / (1.0 + kernels::fast_exp(-x)))
            .collect();

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Sigmoid { input: self.inner.clone() });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// Tanh активация
    pub fn tanh(&self) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        let result_data: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Tanh { input: self.inner.clone() });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// Softmax (по последней оси)
    pub fn softmax(&self) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        let last_dim = shape[shape.len() - 1];
        let n = data.len();
        let mut result_data = vec![0.0f32; n];

        for chunk_start in (0..n).step_by(last_dim) {
            let chunk = &data[chunk_start..chunk_start + last_dim];
            let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (i, &x) in chunk.iter().enumerate() {
                let e = kernels::fast_exp(x - max);
                result_data[chunk_start + i] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for i in chunk_start..chunk_start + last_dim {
                result_data[i] *= inv_sum;
            }
        }

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Softmax { input: self.inner.clone() });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// CrossEntropyLoss: -log(pred[target]) с fused softmax+CE backward
    /// Поддерживает как 1D (один пример), так и 2D (батч) входы
    pub fn cross_entropy(&self, target: &[usize]) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        // 1D → трактуем как [1, n_classes]
        let (n, n_classes) = if shape.len() == 1 {
            (1, shape[0])
        } else {
            (shape[0], shape[1])
        };

        // Loss value (скаляр)
        let mut loss = 0.0f32;
        for i in 0..n {
            let t = target[i];
            let p = data[i * n_classes + t].max(1e-15);
            loss -= p.ln();
        }
        loss /= n as f32;

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(vec![loss], vec![1], false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::CrossEntropy {
                pred: self.inner.clone(),
                target: target.to_vec(),
            });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// Умножение на скаляр
    pub fn scale(&self, factor: f32) -> Var {
        let shape = self.inner.borrow().shape.clone();
        let data = self.inner.borrow().data.clone();
        let result_data: Vec<f32> = data.iter().map(|&x| x * factor).collect();

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, shape.clone(), false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires; // propagate through the graph
            r.op = Some(Op::Scale {
                input: self.inner.clone(),
                factor,
            });
            r.parents = vec![self.inner.clone()];
        }

        result
    }

    /// Транспонирование (для 2D) с autograd
    pub fn t(&self) -> Var {
        let shape = self.inner.borrow().shape.clone();
        assert_eq!(shape.len(), 2, "t(): only 2D");
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.inner.borrow().data.clone();
        let mut result_data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = data[i * cols + j];
            }
        }

        let requires = self.inner.borrow().requires_grad;
        let result = Var::new_inner(result_data, vec![cols, rows], false);

        if requires {
            let mut r = result.inner.borrow_mut();
            r.requires_grad = requires;
            r.op = Some(Op::Transpose {
                input: self.inner.clone(),
                orig_shape: shape.clone(),
            });
            r.parents = vec![self.inner.clone()];
        }

        result
    }
}

// ─── Raw matmul (для backward) ─────────────────────────────────────────────

impl Var {
    /// Сырое матричное умножение без autograd (для backward прохода)
    /// Поддерживает 1D и 2D входы
    fn matmul_raw(&self, b_data: &[f32], a_shape: &[usize], b_shape: &[usize]) -> Vec<f32> {
        let a_is_1d = a_shape.len() == 1;
        let b_is_1d = b_shape.len() == 1;
        let m = if a_is_1d { 1 } else { a_shape[0] };
        let k = if a_is_1d { a_shape[0] } else { a_shape[1] };
        let n = if b_is_1d { 1 } else { b_shape[1] };

        assert!(b_is_1d || b_shape[0] == k, "matmul_raw: inner dim mismatch");

        let a_data = self.inner.borrow().data.clone();
        let mut result = vec![0.0f32; m * n];
        let zero_bias = vec![0.0f32; n];

        if n == 1 {
            for i in 0..m {
                unsafe {
                    kernels::linear_decoder(
                        result[i..].as_mut_ptr(),
                        a_data[i * k..].as_ptr(),
                        b_data.as_ptr(),
                        zero_bias.as_ptr(),
                        k, 1,
                    );
                }
            }
        } else {
            // Транспонируем b для contiguous доступа
            let mut b_t = vec![0.0f32; k * n];
            for j in 0..n {
                for t in 0..k {
                    b_t[j * k + t] = b_data[t * n + j];
                }
            }

            for i in 0..m {
                unsafe {
                    kernels::linear_decoder(
                        result[i * n..].as_mut_ptr(),
                        a_data[i * k..].as_ptr(),
                        b_t.as_ptr(),
                        zero_bias.as_ptr(),
                        k, n,
                    );
                }
            }
        }

        result
    }
}

// ─── Backward ──────────────────────────────────────────────────────────────

impl Var {
    /// Запустить обратное распространение.
    /// Считает градиенты для всех Var с `requires_grad = true` в графе.
    pub fn backward(&self) {
        // 1. Топологическая сортировка
        let order = self.topo_sort();

        // 2. Устанавливаем начальный градиент dL/dL = 1
        self.inner.borrow_mut().grad = Some(vec![1.0; self.len()]);

        // 3. Проход в обратном порядке
        // Пропускаем листовые узлы (без op) — их градиенты не нужно забирать
        for node_ref in order.iter().rev() {
            // Проверяем есть ли op (пропускаем листовые узлы вроде a, b)
            let has_op = node_ref.borrow().op.is_some();
            if !has_op { continue; }

            // Извлекаем op и grad из узла (чтобы не держать borrow)
            let (grad_output, data, op) = {
                let mut node = node_ref.borrow_mut();
                let grad = node.grad.take();
                let data = node.data.clone();
                let op = node.op.take();
                (grad, data, op)
            };

            let grad_output = match grad_output {
                Some(g) => g,
                None => continue,
            };

            let op = match op {
                Some(o) => o,
                None => continue,
            };

            // Применяем backward для этой операции
            op.backward(&grad_output, &data);
        }
    }

    /// Топологическая сортировка графа (DFS post-order)
    fn topo_sort(&self) -> Vec<VarRef> {
        let mut order = vec![];
        let mut visited = std::collections::HashSet::new();
        self.topo_dfs(&self.inner, &mut visited, &mut order);
        order
    }

    fn topo_dfs(
        &self,
        node: &VarRef,
        visited: &mut std::collections::HashSet<*const ()>,
        order: &mut Vec<VarRef>,
    ) {
        let ptr = Rc::as_ptr(node) as *const ();
        if !visited.insert(ptr) {
            return;
        }
        let data = node.borrow();
        if let Some(ref _op) = data.op {
            for child in &data.parents {
                self.topo_dfs(child, visited, order);
            }
        }
        drop(data);
        order.push(node.clone());
    }
}

// ─── Backward для каждой операции ──────────────────────────────────────────

impl Op {
    fn backward(&self, grad_output: &[f32], output_data: &[f32]) {
        match self {
            Op::Matmul { a, b, a_shape, b_shape } => {
                // Поддержка 1D: трактуем как m=1 или n=1
                let a_is_1d = a_shape.len() == 1;
                let b_is_1d = b_shape.len() == 1;
                let m = if a_is_1d { 1 } else { a_shape[0] };
                let k = if a_is_1d { a_shape[0] } else { a_shape[1] };
                let n = if b_is_1d { 1 } else { b_shape[1] };

                let a_req = a.borrow().requires_grad;
                let b_req = b.borrow().requires_grad;

                if a_req {
                    // dL/dA = grad_output @ B^T
                    // B: [k, n], B^T: [n, k]
                    // weights[o*n + t] = B[o][t] = B^T[t][o]  ← не нужно транспонировать!
                    // linear_decoder(grad_a[i], grad_output[i], B, _, n, k)
                    //   → grad_a[i][o] = Σ_t grad_output[i][t] * B[o][t]  ✓
                    let b_data = b.borrow().data.clone(); // B: [k, n]
                    let mut grad_a = vec![0.0f32; m * k];
                    let zero_bias = vec![0.0f32; k];
                    for i in 0..m {
                        unsafe {
                            kernels::linear_decoder(
                                grad_a[i * k..].as_mut_ptr(),
                                grad_output[i * n..].as_ptr(),
                                b_data.as_ptr(),
                                zero_bias.as_ptr(),
                                n, k,
                            );
                        }
                    }
                    add_to_grad(a, &grad_a);
                }

                if b_req {
                    // dL/dB = A^T @ grad_output
                    // A: [m, k], A^T: [k, m]
                    // grad_output: [m, n]
                    // result: [k, m] @ [m, n] = [k, n]
                    let a_data = a.borrow().data.clone();

                    // A^T: [k, m]
                    let mut a_t = vec![0.0f32; k * m];
                    for t in 0..k {
                        for i in 0..m {
                            a_t[t * m + i] = a_data[i * k + t];
                        }
                    }

                    let mut grad_b = vec![0.0f32; k * n];
                    let zero_bias = vec![0.0f32; n];
                    for t in 0..k {
                        unsafe {
                            kernels::linear_decoder(
                                grad_b[t * n..].as_mut_ptr(),
                                a_t[t * m..].as_ptr(),
                                grad_output.as_ptr(),
                                zero_bias.as_ptr(),
                                m, n,
                            );
                        }
                    }
                    add_to_grad(b, &grad_b);
                }
            }

            Op::Add { a, b, a_shape, b_shape } => {
                let a_req = a.borrow().requires_grad;
                let b_req = b.borrow().requires_grad;

                if a_req {
                    add_to_grad(a, grad_output);
                }

                if b_req {
                    if a_shape.len() == 2 && b_shape.len() == 1 {
                        // bias broadcast: sum по батчу
                        let (m, n) = (a_shape[0], a_shape[1]);
                        let mut grad_b = vec![0.0f32; n];
                        for i in 0..m {
                            for j in 0..n {
                                grad_b[j] += grad_output[i * n + j];
                            }
                        }
                        add_to_grad(b, &grad_b);
                    } else {
                        add_to_grad(b, grad_output);
                    }
                }
            }

            Op::Relu { input } => {
                // dL/dx = dL/dy * (x > 0)
                if input.borrow().requires_grad {
                    let input_data = input.borrow().data.clone();
                    let mut grad_input = vec![0.0f32; input_data.len()];
                    for i in 0..grad_output.len() {
                        grad_input[i] = if input_data[i] > 0.0 { grad_output[i] } else { 0.0 };
                    }
                    add_to_grad(input, &grad_input);
                }
            }

            Op::Sigmoid { input } => {
                // dL/dx = dL/dy * y * (1 - y)
                if input.borrow().requires_grad {
                    let mut grad_input = vec![0.0f32; grad_output.len()];
                    for i in 0..grad_output.len() {
                        let y = output_data[i];
                        grad_input[i] = grad_output[i] * y * (1.0 - y);
                    }
                    add_to_grad(input, &grad_input);
                }
            }

            Op::Tanh { input } => {
                // dL/dx = dL/dy * (1 - y²)
                if input.borrow().requires_grad {
                    let mut grad_input = vec![0.0f32; grad_output.len()];
                    for i in 0..grad_output.len() {
                        let y = output_data[i];
                        grad_input[i] = grad_output[i] * (1.0 - y * y);
                    }
                    add_to_grad(input, &grad_input);
                }
            }

            Op::Softmax { input } => {
                // Jacobian: J = diag(y) - y·y^T
                // grad_input = y * (grad_output - Σ(grad_output * y))
                if input.borrow().requires_grad {
                    let shape = input.borrow().shape.clone();
                    let last_dim = shape[shape.len() - 1];
                    let n = grad_output.len();
                    let mut grad_input = vec![0.0f32; n];

                    for chunk_start in (0..n).step_by(last_dim) {
                        let y = &output_data[chunk_start..chunk_start + last_dim];
                        let go = &grad_output[chunk_start..chunk_start + last_dim];
                        let mut gi = vec![0.0f32; last_dim];

                        // sum(grad_output * y)
                        let mut s = 0.0;
                        for j in 0..last_dim {
                            s += go[j] * y[j];
                        }

                        for j in 0..last_dim {
                            gi[j] = y[j] * (go[j] - s);
                        }

                        for j in 0..last_dim {
                            grad_input[chunk_start + j] = gi[j];
                        }
                    }

                    add_to_grad(input, &grad_input);
                }
            }

            Op::Scale { input, factor } => {
                if input.borrow().requires_grad {
                    let grad_input: Vec<f32> = grad_output.iter().map(|&g| g * factor).collect();
                    add_to_grad(input, &grad_input);
                }
            }

            Op::Transpose { input, orig_shape } => {
                // dL/d(input) = transpose of grad_output
                if input.borrow().requires_grad {
                    let (rows, cols) = (orig_shape[0], orig_shape[1]);
                    let mut grad_input = vec![0.0f32; rows * cols];
                    for i in 0..rows {
                        for j in 0..cols {
                            grad_input[i * cols + j] = grad_output[j * rows + i];
                        }
                    }
                    add_to_grad(input, &grad_input);
                }
            }

            Op::CrossEntropy { pred, target } => {
                // NLL backward: dL/dp_i = -1/p_target для target, 0 для остальных
                // Softmax backward затем превратит это в (softmax - one_hot)
                if pred.borrow().requires_grad {
                    let pred_data = pred.borrow().data.clone();
                    let shape = pred.borrow().shape.clone();
                    let (n, n_classes) = if shape.len() == 1 {
                        (1, shape[0])
                    } else {
                        (shape[0], shape[1])
                    };
                    let mut grad_pred = vec![0.0f32; n * n_classes];

                    for i in 0..n {
                        let t = target[i];
                        let p_t = pred_data[i * n_classes + t].max(1e-15);
                        grad_pred[i * n_classes + t] = -1.0 / p_t / n as f32;
                    }

                    // grad_output (от loss) — это скаляр [1.0]
                    if !grad_output.is_empty() && grad_output[0] != 1.0 {
                        for g in grad_pred.iter_mut() {
                            *g *= grad_output[0];
                        }
                    }

                    add_to_grad(pred, &grad_pred);
                }
            }
        }
    }
}

// ─── Утилиты ───────────────────────────────────────────────────────────────

/// Добавить градиент к переменной (суммирование).
fn add_to_grad(node: &VarRef, grad: &[f32]) {
    let mut node = node.borrow_mut();
    if node.grad.is_none() {
        node.grad = Some(vec![0.0; grad.len()]);
    }
    if let Some(ref mut g) = node.grad {
        for i in 0..grad.len() {
            g[i] += grad[i];
        }
    }
}

// ─── Display ───────────────────────────────────────────────────────────────

use std::fmt;
impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.inner.borrow();
        write!(f, "Var(shape={:?}, requires_grad={})", data.shape, data.requires_grad)
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.inner.borrow();
        write!(f, "Var([")?;
        for (i, v) in data.data.iter().take(6).enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.3}", v)?;
        }
        if data.data.len() > 6 {
            write!(f, ", ...")?;
        }
        write!(f, "], shape={:?}", data.shape)?;
        if data.requires_grad {
            write!(f, ", grad={:?}", data.grad.as_ref().map(|g| g.len()))?;
        }
        write!(f, ")")
    }
}

// ─── Тесты ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add() {
        let a = Var::from_vec(vec![2.0, 3.0]).requires_grad(true);
        let b = Var::from_vec(vec![1.0, 4.0]).requires_grad(true);
        let c = a.add(&b);
        let loss = c.softmax().cross_entropy(&[0]);
        loss.backward();

        let ga = a.grad().unwrap();
        assert!(ga[0] != 0.0, "grad should not be zero");
    }

    #[test]
    fn test_relu_backward() {
        let a = Var::from_vec(vec![2.0, -1.0, 0.5, -3.0]).requires_grad(true);
        let r = a.relu();
        let sum = r.scale(1.0);
        // fuse: sum of all elements → backward gives all ones
        // Actually scale(1.0) is identity, so grad_output for r is [1,1,1,1]
        let mut grad = vec![1.0f32; 4];
        // Manually: for relu, grad_input = grad_output * (x > 0)
        // So: [2*1, 0*1, 0.5*1, 0*1] = [1, 0, 1, 0] wait no: [1, 0, 1, 0]
        // grad_output is [1,1,1,1], input is [2, -1, 0.5, -3]
        // grad_input = [1*(2>0), 1*(-1>0), 1*(0.5>0), 1*(-3>0)] = [1, 0, 1, 0]

        sum.backward();
        let ga = a.grad().unwrap();
        assert!((ga[0] - 1.0).abs() < 1e-6, "ga[0] should be 1.0, got {}", ga[0]);
        assert!((ga[1] - 0.0).abs() < 1e-6, "ga[1] should be 0.0, got {}", ga[1]);
        assert!((ga[2] - 1.0).abs() < 1e-6, "ga[2] should be 1.0, got {}", ga[2]);
        assert!((ga[3] - 0.0).abs() < 1e-6, "ga[3] should be 0.0, got {}", ga[3]);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Var::randn(&[2, 3]).requires_grad(true);
        let b = Var::randn(&[3, 4]).requires_grad(true);
        let c = a.matmul(&b);
        let loss = c.softmax().cross_entropy(&[0, 1]);
        loss.backward();

        let ga = a.grad().unwrap();
        let gb = b.grad().unwrap();
        assert_eq!(ga.len(), 6);
        assert_eq!(gb.len(), 12);
        // Just check they're non-zero
        assert!(ga.iter().any(|&x| x != 0.0), "grad_a should not be all zero");
        assert!(gb.iter().any(|&x| x != 0.0), "grad_b should not be all zero");
    }

    #[test]
    fn test_no_grad() {
        let a = Var::from_vec(vec![2.0, 3.0]); // no requires_grad
        let b = Var::from_vec(vec![1.0, 4.0]).requires_grad(true);
        let c = a.add(&b);
        let loss = c.softmax().cross_entropy(&[0]);
        loss.backward();

        let ga = a.grad();
        assert!(ga.is_none(), "a should not have grad");
        let gb = b.grad().unwrap();
        assert!(gb.iter().any(|&x| x != 0.0), "b should have grad");
    }

    #[test]
    fn test_sigmoid_tanh_backward() {
        let a = Var::from_vec(vec![1.0, -1.0]).requires_grad(true);
        let s = a.sigmoid();
        let t = a.tanh();
        let s_loss = s.softmax().cross_entropy(&[0]);
        let t_loss = t.softmax().cross_entropy(&[1]);

        // Just check they don't crash
        s_loss.backward();
        let ga_s = a.grad().unwrap();
        assert!(ga_s.iter().any(|&x| x != 0.0));

        // Reset grad
        a.inner.borrow_mut().grad = None;
        t_loss.backward();
        let ga_t = a.grad().unwrap();
        assert!(ga_t.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_multiple_backward_paths() {
        let x = Var::from_vec(vec![1.0, 2.0]).requires_grad(true);
        let w1 = Var::randn(&[2, 3]).requires_grad(true);
        let w2 = Var::randn(&[2, 3]).requires_grad(true);

        let h1 = x.matmul(&w1).relu();
        let h2 = x.matmul(&w2).sigmoid();
        let h = h1.add(&h2);

        let loss = h.softmax().cross_entropy(&[0]);
        loss.backward();

        let gx = x.grad().unwrap();
        assert!(gx.iter().any(|&x| x != 0.0));
    }
}
