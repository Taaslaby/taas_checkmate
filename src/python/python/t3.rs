//! PyO3 classes for T3 — clean high-level API.
//!
//! Instead of:
//!   tri.set_drive(d, 1); tri.step(0.05); pst_linear_features(tri, out);
//!
//! Now:
//!   tri.step(0.05, drive=d)          # → pst features
//!   tri.run(10, 0.05, drive=d)        # → (steps, pst_dim) array
//!   tri.drive = d                      # setter
//!   tri.coupling = (0.3, 0.15, 0.08)  # setter
//!   tri.pst                            # getter → ndarray
//!   tri.pst_dim                        # getter → int
//!   tri.order_params                   # getter → (r1, r2, r3)

#![allow(non_snake_case, deprecated)]

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::t3::Decoder;

// ═══════════════════════════════════════════════════════════
//  T3 TriOscillator
// ═══════════════════════════════════════════════════════════

/// T3 Triple Oscillator — три связанные популяции осцилляторов.
///
/// Каждая популяция: Kuramoto фазовая динамика + Stuart-Landau амплитуды.
/// Три популяции связаны cross-coupling (w12, w13, w23).
///
/// Использование:
///     tri = TriOscillator(64, 0.5, 3.0)
///     tri.set_coupling(0.3, 0.15, 0.08)
///     pst = tri.step(0.05, drive=drives)       # один шаг → PST фичи
///     tri.set_drive(drives)                      # установить драйв
///     pst = tri.pst()                            # текущие PST фичи
///     r1, r2, r3 = tri.order_params()           # параметры порядка
///     ph1 = tri.phases(1)                        # фазы популяции 1
///
/// PST фичи: 18n+12 размерность (sin/cos, скорости, r, порядок, cross-order)
#[pyclass(name = "TriOscillator")]
pub struct PyTriOscillator {
    pub inner: crate::t3::TriOscillator,
    _fv: Vec<f32>,
}

impl PyTriOscillator {
    /// Construct from an existing TriOscillator (used by taas_format loader).
    pub fn build(tri: crate::t3::TriOscillator) -> Self {
        let n = tri.n;
        Self {
            inner: tri,
            _fv: vec![0.0; crate::t3::pst_linear_dim(n)],
        }
    }
}

#[pymethods]
impl PyTriOscillator {
    #[new]
    fn new(n: usize, freq_min: f32, freq_max: f32) -> Self {
        Self::build(crate::t3::TriOscillator::new(n, freq_min, freq_max, crate::t3::Avx2Backend))
    }

    // ── coupling setter (как обычный метод, не property — 3 аргумента) ──

    fn set_coupling(&mut self, w12: f32, w13: f32, w23: f32) {
        self.inner.set_uniform_coupling(w12, w13, w23);
    }

    // ── drive setter (как обычный метод) ──

    /// Установить драйв для осцилляторов.
    ///
    /// Принимает:
    ///     drive (n_osc,) — устанавливает только популяцию 1
    ///     drive (n_osc*3,) — устанавливает все 3 популяции
    fn set_drive(&mut self, drive: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let sl = {
            let ro = drive.readonly();
            ro.as_slice()?.to_vec()
        };
        let n = self.inner.n;
        if sl.len() == n {
            self.inner.drive1[..n].copy_from_slice(&sl[..n]);
        } else if sl.len() == n * 3 {
            self.inner.drive1[..n].copy_from_slice(&sl[..n]);
            self.inner.drive2[..n].copy_from_slice(&sl[n..n * 2]);
            self.inner.drive3[..n].copy_from_slice(&sl[n * 2..]);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("drive: ожидается (n_osc,)={} или (n_osc*3,)={}, получено {}",
                        n, n * 3, sl.len())));
        }
        Ok(())
    }

    // ── step ──

    /// Один шаг T3 осциллятора.
    ///
    /// Если передан drive — сначала установить его.
    /// Возвращает PST-фичи как np.ndarray (18n+12,).
    #[pyo3(signature = (dt, drive=None))]
    fn step(&mut self, dt: f32, drive: Option<Bound<'_, PyArray1<f32>>>) -> PyResult<Py<PyArray1<f32>>> {
        if let Some(d) = drive {
            self.set_drive(d)?;
        }
        self.inner.step(dt);
        let fv = crate::t3::pst_linear_features(&self.inner);
        self._fv[..fv.len()].copy_from_slice(&fv);
        Python::with_gil(|py| {
            Ok(PyArray1::from_vec(py, self._fv.clone()).into())
        })
    }

    // ── run ──

    /// N шагов T3 осциллятора.
    ///
    /// Если передан drive — установить его перед запуском.
    /// Возвращает np.ndarray формы (steps, pst_dim).
    #[pyo3(signature = (steps, dt, drive=None))]
    fn run(&mut self, steps: usize, dt: f32, drive: Option<Bound<'_, PyArray1<f32>>>) -> PyResult<Py<PyArray2<f32>>> {
        if let Some(d) = drive {
            self.set_drive(d)?;
        }
        let dim = crate::t3::pst_linear_dim(self.inner.n);
        let mut buf = Vec::with_capacity(steps * dim);
        for _ in 0..steps {
            self.inner.step(dt);
            let fv = crate::t3::pst_linear_features(&self.inner);
            buf.extend_from_slice(&fv);
        }
        Python::with_gil(|py| {
            let arr = PyArray2::from_vec2(py, &buf.chunks(dim).map(|c| c.to_vec()).collect::<Vec<_>>())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
            Ok(arr.into())
        })
    }

    // ── pst property ──

    /// PST-фичи текущего состояния T3 как np.ndarray (18n+12,).
    fn pst(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        let fv = crate::t3::pst_linear_features(&self.inner);
        PyArray1::from_vec(py, fv).into()
    }

    /// Размерность PST-фич (18n+12) для данного n осцилляторов.
    fn pst_dim(&self) -> usize {
        crate::t3::pst_linear_dim(self.inner.n)
    }

    fn n_osc(&self) -> usize {
        self.inner.n
    }

    // ── order params ──

    /// Параметры порядка (синхронизация) трёх популяций.
    ///
    /// Returns: (r1, r2, r3) — средняя синхронизация каждой популяции.
    /// 0 = полностью несинхронизированы, 1 = полная синхронизация.
    fn order_params(&self) -> (f32, f32, f32) {
        (
            self.inner.order_parameter(&self.inner.th1),
            self.inner.order_parameter(&self.inner.th2),
            self.inner.order_parameter(&self.inner.th3),
        )
    }

    // ── фазы / амплитуды (для продвинутого доступа) ──

    /// Фазы популяции pop (1, 2, 3) как список float.
    fn phases(&self, pop: usize) -> Vec<f32> {
        let src = match pop { 1 => &self.inner.th1, 2 => &self.inner.th2, _ => &self.inner.th3 };
        src.clone()
    }

    /// Амплитуды (Stuart-Landau r) популяции pop (1, 2, 3) как список float.
    fn amplitudes(&self, pop: usize) -> Vec<f32> {
        let src = match pop { 1 => &self.inner.r1, 2 => &self.inner.r2, _ => &self.inner.r3 };
        src.clone()
    }

    fn __repr__(&self) -> String {
        format!("TriOscillator(n={}, pst_dim={})", self.inner.n, crate::t3::pst_linear_dim(self.inner.n))
    }
}

// ═══════════════════════════════════════════════════════════
//  NlmsDecoder
// ═══════════════════════════════════════════════════════════

/// NLMS (Normalized Least Mean Squares) адаптивный декодер.
///
/// Онлайн-обучение: w += lr * error * x / (||x||² + eps).
///
/// Использование:
///     dec = NlmsDecoder(pst_dim, lr=0.1, eps=1e-8)
///     prediction = dec.predict(features)
///     dec.train(features, target)
///     weights = dec.weights()
#[pyclass(name = "NlmsDecoder")]
pub struct PyNlmsDecoder {
    pub inner: crate::t3::NlmsDecoder,
}

#[pymethods]
impl PyNlmsDecoder {
    #[new]
    fn new(n: usize, lr: f32, eps: f32) -> Self {
        Self { inner: crate::t3::NlmsDecoder::new(n, lr, eps) }
    }

    fn predict(&self, states: Bound<'_, PyArray1<f32>>) -> PyResult<f32> {
        Ok(self.inner.predict(states.readonly().as_slice()?))
    }

    fn train(&mut self, states: Bound<'_, PyArray1<f32>>, target: f32) -> PyResult<f32> {
        Ok(self.inner.train(states.readonly().as_slice()?, target))
    }

    fn weights(&self) -> Vec<f32> {
        self.inner.weights.clone()
    }

    fn __repr__(&self) -> String {
        format!("NlmsDecoder(n={}, lr={})", self.inner.weights.len(), self.inner.lr)
    }
}

// ═══════════════════════════════════════════════════════════
//  KalmanDecoder
// ═══════════════════════════════════════════════════════════

/// Kalman filter адаптивный декодер.
///
/// Использует фильтр Калмана для оценки весов.
///
/// Использование:
///     dec = KalmanDecoder(pst_dim, q=1.0, r=0.1)
///     prediction = dec.predict(features)
///     dec.train(features, target)
#[pyclass(name = "KalmanDecoder")]
pub struct PyKalmanDecoder {
    pub inner: crate::t3::KalmanDecoder,
}

#[pymethods]
impl PyKalmanDecoder {
    #[new]
    fn new(n: usize, q: f32, r: f32) -> Self {
        Self { inner: crate::t3::KalmanDecoder::new(n, q, r) }
    }

    fn predict(&self, states: Bound<'_, PyArray1<f32>>) -> PyResult<f32> {
        Ok(self.inner.predict(states.readonly().as_slice()?))
    }

    fn train(&mut self, states: Bound<'_, PyArray1<f32>>, target: f32) -> PyResult<f32> {
        Ok(self.inner.train(states.readonly().as_slice()?, target))
    }

    fn __repr__(&self) -> String {
        format!("KalmanDecoder(n={}, q={}, r={})", self.inner.weights.len(), self.inner.q, self.inner.r)
    }
}

// ═══════════════════════════════════════════════════════════
//  Standalone PST functions (legacy compatibility)
// ═══════════════════════════════════════════════════════════

/// Вычислить PST sin/cos фичи между двумя массивами фаз.
///
/// Синоним: t3_pst_features из модуля io.
#[pyfunction]
#[pyo3(name = "pst_features")]
pub fn py_pst_features(
    out_sin: Bound<'_, PyArray1<f32>>,
    out_cos: Bound<'_, PyArray1<f32>>,
    ph1: Bound<'_, PyArray1<f32>>,
    ph2: Bound<'_, PyArray1<f32>>,
) -> PyResult<()> {
    let (s, c) = crate::t3::pst_features(ph1.readonly().as_slice()?, ph2.readonly().as_slice()?);
    let n = s.len().min(out_sin.len()?);
    out_sin.readwrite().as_slice_mut()?[..n].copy_from_slice(&s[..n]);
    out_cos.readwrite().as_slice_mut()?[..n].copy_from_slice(&c[..n]);
    Ok(())
}

/// Вычислить полный PST вектор из TriOscillator.
///
/// Возвращает np.ndarray (18n+12,).
/// Для нового кода рекомендуется tri.pst() — быстрее и короче.
#[pyfunction]
#[pyo3(name = "pst_linear_features")]
pub fn py_pst_linear_features(tri: &PyTriOscillator, py: Python<'_>) -> Py<PyArray1<f32>> {
    let fv = crate::t3::pst_linear_features(&tri.inner);
    PyArray1::from_vec(py, fv).into()
}

/// Размерность PST вектора для n осцилляторов.
///
/// Формула: 18n + 12 (sin/cos, скорости, амплитуды, порядок, cross-order).
#[pyfunction]
#[pyo3(name = "pst_linear_dim")]
pub fn py_pst_linear_dim(n: usize) -> usize {
    crate::t3::pst_linear_dim(n)
}

/// Размерность fast PST вектора: 6n (sin + r для 3 популяций).
#[pyfunction]
#[pyo3(name = "pst_fast_dim")]
pub fn py_pst_fast_dim(n: usize) -> usize {
    crate::t3::pst_fast_dim(n)
}

// ═══════════════════════════════════════════════════════════
//  T2SpigradTrajectory — Python class
// ═══════════════════════════════════════════════════════════

/// T2-SPIGRAD trajectory — хранит траекторию T3 для backward pass.
///
/// Создаётся функцией t2spigrad_forward(), передаётся в t2spigrad_backward().
/// Содержит все фазы, амплитуды, coupling и драйвы на каждом шаге.
#[pyclass(name = "T2SpigradTrajectory")]
// (Rust struct PyT2SpigradTrajectory — Python видит как T2SpigradTrajectory)
pub struct PyT2SpigradTrajectory {
    inner: crate::train::spike::T3SpikeTrajectory,
}

#[pymethods]
impl PyT2SpigradTrajectory {
    fn __repr__(&self) -> String {
        format!("T2SpigradTrajectory(n_osc={}, n_steps={})", self.inner.n_osc, self.inner.n_steps)
    }

    fn n_osc(&self) -> usize { self.inner.n_osc }
    fn n_steps(&self) -> usize { self.inner.n_steps }
    fn dt(&self) -> f32 { self.inner.dt }
    fn drive_gain(&self) -> f32 { self.inner.drive_gain }
    fn alpha(&self) -> f32 { self.inner.alpha }
    fn epsilon(&self) -> f32 { self.inner.epsilon }
    fn freq_min(&self) -> f32 { self.inner.freq_min }
    fn freq_max(&self) -> f32 { self.inner.freq_max }
    fn w12(&self) -> f32 { self.inner.w12 }
    fn w13(&self) -> f32 { self.inner.w13 }
    fn w23(&self) -> f32 { self.inner.w23 }

    fn omega1(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.omega1.clone()).into() }
    fn omega2(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.omega2.clone()).into() }
    fn omega3(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.omega3.clone()).into() }
    fn th1_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th1_pre.clone()).into() }
    fn th2_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th2_pre.clone()).into() }
    fn th3_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th3_pre.clone()).into() }
    fn r1_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r1_pre.clone()).into() }
    fn r2_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r2_pre.clone()).into() }
    fn r3_pre(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r3_pre.clone()).into() }
    fn th1_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th1_final.clone()).into() }
    fn th2_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th2_final.clone()).into() }
    fn th3_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.th3_final.clone()).into() }
    fn r1_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r1_final.clone()).into() }
    fn r2_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r2_final.clone()).into() }
    fn r3_final(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.r3_final.clone()).into() }
    fn coupling1(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.coupling1.clone()).into() }
    fn coupling2(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.coupling2.clone()).into() }
    fn coupling3(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.coupling3.clone()).into() }
    fn drive1(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.drive1.clone()).into() }
    fn drive2(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.drive2.clone()).into() }
    fn drive3(&self, py: Python<'_>) -> Py<PyArray1<f32>>  { PyArray1::from_vec(py, self.inner.drive3.clone()).into() }
}

// ═══════════════════════════════════════════════════════════
//  T2-SPIGRAD — T3-SPIKE GRADient
// ═══════════════════════════════════════════════════════════

/// T2-SPIGRAD forward: драйвы → T3 (N шагов) → PST фичи + траектория.
///
/// Градиентный движок для T3. Вычисляет forward pass и сохраняет
/// траекторию для последующего backward pass (BPTT).
///
/// Args:
///     n_osc: количество осцилляторов
///     n_steps: количество шагов
///     dt: шаг по времени
///     drive_gain: усиление драйва
///     alpha: коэффициент Stuart-Landau
///     freq_min / freq_max: диапазон частот
///     coupling_w12 / w13 / w23: cross-coupling между популяциями
///     drives: np.ndarray (n_steps, 3*n_osc) — драйвы для всех шагов
///
/// Returns:
///     (pst, trajectory) — PST фичи и объект T2SpigradTrajectory для backward
#[pyfunction]
#[pyo3(name = "t2spigrad_forward", signature = (n_osc, n_steps, dt, drive_gain, alpha, freq_min, freq_max, coupling_w12, coupling_w13, coupling_w23, drives))]
pub fn py_t2spigrad_forward(
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: Bound<'_, PyArray2<f32>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let drives_data = drives.readonly().as_slice()?.to_vec();

    let (pst, traj) = crate::train::spike::t3spike_forward(
        n_osc, n_steps, dt, drive_gain, alpha,
        freq_min, freq_max,
        coupling_w12, coupling_w13, coupling_w23,
        &drives_data,
    );

    let py_traj = PyT2SpigradTrajectory { inner: traj };
    let pst_arr = PyArray1::from_vec(py, pst);
    let tuple = PyTuple::new(py, &[pst_arr.into_py(py), py_traj.into_py(py)])?.unbind();
    Ok(tuple.into())
}

/// T2-SPIGRAD backward: градиент PST → BPTT → градиент драйвов.
///
/// Вычисляет градиент драйвов по градиенту PST фичей
/// через Backpropagation Through Time (BPTT).
///
/// Args:
///     grad_pst: np.ndarray (pst_dim,) — градиент потерь по PST
///     traj: T2SpigradTrajectory — траектория из forward
///     per_step_grad_pst: optional np.ndarray (n_steps, pst_dim) — градиент потерь
///         по PST для КАЖДОГО шага. Если передан, выполняет single-pass BPTT
///         для полной траектории (SPIGRAD). Если None, использует только grad_pst
///         для финального шага.
///
/// Returns:
///     np.ndarray (n_steps, 3*n_osc) — градиент потерь по драйвам
#[pyfunction]
#[pyo3(name = "t2spigrad_backward", signature = (grad_pst, traj, per_step_grad_pst=None))]
pub fn py_t2spigrad_backward(
    grad_pst: Bound<'_, PyArray1<f32>>,
    traj: &PyT2SpigradTrajectory,
    per_step_grad_pst: Option<Bound<'_, PyArray2<f32>>>,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<f32>>> {
    let n = traj.inner.n_osc;
    let pst_dim = crate::t3::pst_linear_dim(n);

    let mut grad_pst_vec = grad_pst.readonly().to_vec()?;
    if grad_pst_vec.len() > pst_dim {
        grad_pst_vec.truncate(pst_dim);
    } else {
        grad_pst_vec.resize(pst_dim, 0.0);
    }

    let per_step = per_step_grad_pst.map(|arr| {
        arr.readonly().as_slice().map(|s| s.to_vec())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("per_step_grad_pst: {:?}", e)))
    }).transpose()?;

    let grad_drives = crate::train::spike::t3spike_backward_per_step(
        &grad_pst_vec, &traj.inner, per_step.as_deref(),
    );

    let arr = PyArray2::from_vec2(py, &grad_drives.chunks(3 * n).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok(arr.into())
}

/// Lazily initialize CUDA device (called once, cached globally).
#[cfg(feature = "cuda")]
fn ensure_cuda_device() -> Result<(), String> {
    use std::sync::OnceLock;
    static CUDA_INIT: OnceLock<Result<(), String>> = OnceLock::new();
    CUDA_INIT.get_or_init(|| crate::cuda::set_device(0)).clone()
}

/// T2-SPIGRAD backward на GPU (CUDA).
///
/// Требует сборки с --features cuda.
/// GPU контекст кэшируется — нет переаллокации при повторных вызовах.
///
/// Args:
///     grad_pst: np.ndarray (pst_dim,) — градиент потерь по PST
///     traj: T2SpigradTrajectory — траектория из forward
///     per_step_grad_pst: optional np.ndarray (n_steps, pst_dim) — градиент потерь
///         по PST для КАЖДОГО шага. Single-pass BPTT для полной траектории (SPIGRAD).
///
/// Returns:
///     np.ndarray (n_steps, 3*n_osc) — градиент потерь по драйвам
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(name = "t2spigrad_backward_cuda", signature = (grad_pst, traj, per_step_grad_pst=None))]
pub fn py_t2spigrad_backward_cuda(
    grad_pst: Bound<'_, PyArray1<f32>>,
    traj: &PyT2SpigradTrajectory,
    per_step_grad_pst: Option<Bound<'_, PyArray2<f32>>>,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<f32>>> {
    let n = traj.inner.n_osc;
    let pst_dim = crate::t3::pst_linear_dim(n);

    let mut grad_pst_vec = grad_pst.readonly().to_vec()?;
    if grad_pst_vec.len() > pst_dim {
        grad_pst_vec.truncate(pst_dim);
    } else {
        grad_pst_vec.resize(pst_dim, 0.0);
    }

    let per_step = per_step_grad_pst.map(|arr| {
        arr.readonly().as_slice().map(|s| s.to_vec())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("per_step_grad_pst: {:?}", e)))
    }).transpose()?;

    // Initialize CUDA device once
    ensure_cuda_device().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("cuda init: {e}"))
    })?;

    // Get cached GPU context (thread-local, no alloc on repeated calls)
    let n_steps = traj.inner.n_steps;
    use std::cell::RefCell;
    thread_local! {
        static CTX: RefCell<Option<(usize, usize, crate::train::spike_cuda::CudaBackwardContext)>> = const { RefCell::new(None) };
    }

    let grad_drives = CTX.with(|cell| -> Result<Vec<f32>, String> {
        let mut borrow = cell.borrow_mut();
        let ctx = match borrow.as_mut() {
            Some((cached_n, cached_ns, ctx)) if *cached_n == n && *cached_ns == n_steps => ctx,
            _ => {
                let new_ctx = crate::train::spike_cuda::CudaBackwardContext::new(n, n_steps)?;
                *borrow = Some((n, n_steps, new_ctx));
                &mut borrow.as_mut().expect("just set ctx").2
            }
        };
        crate::train::spike_cuda::t3spike_backward_cuda_per_step(
            &grad_pst_vec, &traj.inner, ctx, per_step.as_deref(),
        )
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("backward_cuda: {e}")))?;

    let arr = PyArray2::from_vec2(py, &grad_drives.chunks(3 * n).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok(arr.into())
}

/// T2-SPIGRAD forward на GPU (CUDA).
///
/// Требует сборки с --features cuda.
///
/// Returns: (pst, trajectory) — PST фичи и T2SpigradTrajectory для backward.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(name = "t2spigrad_forward_cuda", signature = (n_osc, n_steps, dt, drive_gain, alpha, freq_min, freq_max, coupling_w12, coupling_w13, coupling_w23, drives))]
pub fn py_t2spigrad_forward_cuda(
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: Bound<'_, PyArray2<f32>>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let drives_data = drives.readonly().as_slice()?.to_vec();

    // Initialize CUDA device once
    ensure_cuda_device().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("cuda init: {e}"))
    })?;
    let backend = crate::t3::CudaBackend::new(n_osc)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CudaBackend::new: {e}")))?;

    let (pst, traj) = crate::train::spike::t3spike_forward_with_backend(
        backend,
        n_osc, n_steps, dt, drive_gain, alpha,
        freq_min, freq_max,
        coupling_w12, coupling_w13, coupling_w23,
        &drives_data,
    );

    let py_traj = PyT2SpigradTrajectory { inner: traj };
    let pst_arr = PyArray1::from_vec(py, pst);
    let tuple = PyTuple::new(py, &[pst_arr.into_py(py), py_traj.into_py(py)])?.unbind();
    Ok(tuple.into())
}

/// Вычислить полную PST матрицу из траектории (один проход на Rust, без Python-накладок).
///
/// Возвращает np.ndarray формы (n_steps, pst_dim) — PST фичи для каждого шага.
#[pyfunction]
#[pyo3(name = "t2spigrad_pst_matrix")]
pub fn py_t2spigrad_pst_matrix(
    traj: &PyT2SpigradTrajectory,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<f32>>> {
    let result = crate::train::spike::t3spike_compute_pst_matrix(&traj.inner);
    let n = traj.inner.n_osc;
    let n_steps = traj.inner.n_steps;
    let pst_dim = crate::t3::pst_linear_dim(n);
    let arr = PyArray2::from_vec2(py, &result.chunks(pst_dim).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok(arr.into())
}

/// Быстрая PST матрица — sin(θ) + r для всех шагов (6N на шаг, 0 trig calls).
///
/// Возвращает np.ndarray формы (n_steps, 6n).
#[pyfunction]
#[pyo3(name = "t2spigrad_pst_fast_matrix")]
pub fn py_t2spigrad_pst_fast_matrix(
    traj: &PyT2SpigradTrajectory,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<f32>>> {
    let result = crate::train::spike::t3spike_compute_pst_fast_matrix(&traj.inner);
    let n = traj.inner.n_osc;
    let n_steps = traj.inner.n_steps;
    let dim = crate::t3::pst_fast_dim(n);
    let arr = PyArray2::from_vec2(py, &result.chunks(dim).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
    Ok(arr.into())
}

/// Вычислить корреляцию Пирсона между двумя массивами.
#[pyfunction]
#[pyo3(name = "compute_pearson")]
pub fn py_compute_pearson(
    a: Bound<'_, PyArray1<f32>>,
    b: Bound<'_, PyArray1<f32>>,
) -> PyResult<f32> {
    Ok(crate::t3::compute_pearson(a.readonly().as_slice()?, b.readonly().as_slice()?))
}

/// Нормализовать матрицу фич in-place.
#[pyfunction]
#[pyo3(name = "normalize_features")]
pub fn py_normalize_features(
    features: Bound<'_, PyArray1<f32>>,
    n_samples: usize,
    dim: usize,
) -> PyResult<()> {
    crate::t3::normalize_features(features.readwrite().as_slice_mut()?, n_samples, dim);
    Ok(())
}

#[pyclass(name = "PstControl")]
pub struct PyPstControl {
    inner: crate::t3::PstControl,
}

#[pymethods]
impl PyPstControl {
    #[new]
    #[pyo3(signature = (n=16, L=2, n_classes=3, ridge_alpha=10.0, inverse_blend=0.3,
                        drive_min=0.0, drive_max=1.5, n_warmup=70, dt=0.01))]
    fn new(
        n: usize, L: usize, n_classes: usize, ridge_alpha: f32,
        inverse_blend: f32, drive_min: f32, drive_max: f32,
        n_warmup: usize, dt: f32,
    ) -> Self {
        let config = crate::t3::PstControlConfig {
            n, L, n_classes, ridge_alpha, inverse_blend,
            drive_min, drive_max, n_warmup, dt,
        };
        Self { inner: crate::t3::PstControl::new(config) }
    }

    fn feat_dim(&self) -> usize { self.inner.feat_dim() }
    fn base_feat_dim(&self) -> usize { self.inner.base_feat_dim() }
    fn n_classes(&self) -> usize { self.inner.n_classes() }
    fn is_fitted(&self) -> bool { self.inner.is_fitted() }

    fn predict_from_pst(&self, pst: Bound<'_, PyArray1<f32>>) -> PyResult<(usize, Vec<f32>)> {
        let pst_binding = pst.readonly();
        let pst_slice = pst_binding.as_slice()?;
        Ok(self.inner.predict_from_pst(pst_slice))
    }

    fn target_pst(&self, target_class: usize) -> Vec<f32> {
        self.inner.target_pst(target_class)
    }

    fn check_target_pst(&self, target_class: usize) -> (usize, f32) {
        self.inner.check_target_pst(target_class)
    }

    fn fit_ridge(
        &mut self,
        x: Bound<'_, PyArray2<f32>>,
        y: Bound<'_, PyArray2<f32>>,
        alpha: f32,
    ) -> PyResult<f32> {
        let x_ro = x.readonly();
        let y_ro = y.readonly();
        let x_slice = x_ro.as_slice()?;
        let y_slice = y_ro.as_slice()?;
        // Infer n from slice length / feat_dim
        let n = x_slice.len() / self.inner.feat_dim();
        Ok(self.inner.fit_ridge(x_slice, y_slice, n, alpha))
    }

    fn find_drive(
        &self,
        tri: &mut PyTriOscillator,
        target_class: usize,
        n_trials: usize,
    ) -> (f32, f32) {
        self.inner.find_drive(&mut tri.inner, target_class, n_trials)
    }

    fn optimize_drive(
        &self,
        tri: &mut PyTriOscillator,
        target_class: usize,
        steps: usize,
        lr: f32,
    ) -> (f32, usize) {
        self.inner.optimize_drive(&mut tri.inner, target_class, steps, lr)
    }

    fn drive_for(
        &self,
        tri: &mut PyTriOscillator,
        target_class: usize,
        method: &str,
    ) -> (f32, usize) {
        self.inner.drive_for(&mut tri.inner, target_class, method)
    }

    fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    #[staticmethod]
    fn load_from_file(path: &str) -> PyResult<Self> {
        crate::t3::PstControl::load_from_file(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    fn weights(&self) -> Vec<f32> { self.inner.weights().to_vec() }
    fn bias(&self) -> Vec<f32> { self.inner.bias().to_vec() }

    fn __repr__(&self) -> String {
        format!("PstControl(n={}, L={}, classes={}, fitted={})",
                self.inner.config.n, self.inner.config.L,
                self.inner.n_classes(), self.inner.is_fitted())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Sparker — спайки → trace → hologram
// ═══════════════════════════════════════════════════════════════

/// Спарковка: спайки → trace → hologram(3) для FM-драйва T³.
///
/// Использование:
///     positions = np.array([[x1,y1], [x2,y2], ...])  # (N, 2)
///     sparker = Sparker(positions, tau_ms=50.0)
///     h = sparker.feed(spikes)      # np.ndarray (3,) — голограмма
///     h = sparker.hologram()        # текущая голограмма без feed
#[pyclass(name = "Sparker")]
pub struct PySparker {
    inner: crate::t3::Sparker,
}

#[pymethods]
impl PySparker {
    #[new]
    fn new(positions: Bound<'_, PyArray2<f32>>, tau_ms: f32, dt_ms: f32) -> PyResult<Self> {
        let pos: Vec<[f32; 2]> = positions.readonly().as_slice()?
            .chunks(2).map(|c| [c[0], c[1]]).collect();
        Ok(Self { inner: crate::t3::Sparker::new(&pos, tau_ms, dt_ms) })
    }

    /// Подать спайки и получить голограмму.
    fn feed(&mut self, spikes: Bound<'_, PyArray1<f32>>) -> PyResult<Vec<f32>> {
        self.inner.feed(spikes.readonly().as_slice()?);
        Ok(self.inner.hologram())
    }

    fn hologram(&self) -> Vec<f32> {
        self.inner.hologram()
    }

    fn trace(&self) -> Vec<f32> {
        self.inner.trace().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("Sparker(n={})", self.inner.hologram().len())
    }
}

/// HologramEncoder — cos-проекция N→K с PCA-базисом.
///
/// Использование:
///     enc = HologramEncoder(positions, k=3)
///     h = enc.encode(trace)      # np.ndarray (k,)
#[pyclass(name = "HologramEncoder")]
pub struct PyHologramEncoder {
    inner: crate::t3::HologramEncoder,
}

#[pymethods]
impl PyHologramEncoder {
    #[new]
    #[pyo3(signature = (positions, k=3))]
    fn new(positions: Bound<'_, PyArray2<f32>>, k: usize) -> PyResult<Self> {
        let pos: Vec<[f32; 2]> = positions.readonly().as_slice()?
            .chunks(2).map(|c| [c[0], c[1]]).collect();
        Ok(Self { inner: crate::t3::HologramEncoder::with_k(&pos, k) })
    }

    fn encode(&self, trace: Bound<'_, PyArray1<f32>>) -> PyResult<Vec<f32>> {
        Ok(self.inner.encode(trace.readonly().as_slice()?))
    }

    fn k(&self) -> usize { self.inner.k() }
    fn n(&self) -> usize { self.inner.n() }

    fn __repr__(&self) -> String {
        format!("HologramEncoder(n={}, k={})", self.inner.n(), self.inner.k())
    }
}

/// NormalizedSparker — Спарковка с online нормализацией.
///
/// Убирает DC-дрейф и масштабирует к target_std.
#[pyclass(name = "NormalizedSparker")]
pub struct PyNormalizedSparker {
    inner: crate::t3::NormalizedSparker,
}

#[pymethods]
impl PyNormalizedSparker {
    #[new]
    fn new(positions: Bound<'_, PyArray2<f32>>, tau_ms: f32, dt_ms: f32, target_std: f32) -> PyResult<Self> {
        let pos: Vec<[f32; 2]> = positions.readonly().as_slice()?
            .chunks(2).map(|c| [c[0], c[1]]).collect();
        Ok(Self { inner: crate::t3::NormalizedSparker::new(&pos, tau_ms, dt_ms, target_std) })
    }

    fn feed(&mut self, spikes: Bound<'_, PyArray1<f32>>) -> PyResult<Vec<f32>> {
        self.inner.feed(spikes.readonly().as_slice()?);
        Ok(self.inner.hologram())
    }

    fn hologram(&mut self) -> Vec<f32> {
        self.inner.hologram()
    }

    fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }

    fn __repr__(&self) -> String {
        "NormalizedSparker(k=3)".into()
    }
}

// ═══════════════════════════════════════════════════════════
//  SpigradBridge — Python class
// ═══════════════════════════════════════════════════════════

/// SPIGRAD bridge: learnable spike-to-drive projection with BPTT through T3.
///
/// The bridge makes spiking networks differentiable:
/// spikes → [W, b] → T3 drives → T3 → PST → decoder → behavior
/// Gradient flows back through T3 physics (BPTT) to the bridge weights.
#[pyclass(name = "SpigradBridge", module = "taas_checkmate")]
pub struct PySpigradBridge {
    pub inner: crate::train::spigrad::SpigradBridge,
}

#[pymethods]
impl PySpigradBridge {
    #[new]
    #[pyo3(signature = (spike_dim, n_osc, dt=0.05, drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0, coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08, zero_init=false))]
    fn new(
        spike_dim: usize,
        n_osc: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
        zero_init: bool,
    ) -> Self {
        if zero_init {
            Self {
                inner: crate::train::spigrad::SpigradBridge::new_zero(
                    spike_dim, n_osc, dt, drive_gain, alpha,
                    freq_min, freq_max,
                    coupling_w12, coupling_w13, coupling_w23,
                ),
            }
        } else {
            Self {
                inner: crate::train::spigrad::SpigradBridge::new(
                    spike_dim, n_osc, dt, drive_gain, alpha,
                    freq_min, freq_max,
                    coupling_w12, coupling_w13, coupling_w23,
                ),
            }
        }
    }

    /// Create bridge from saved weights and bias.
    #[staticmethod]
    #[pyo3(name = "from_weights", signature = (spike_dim, n_osc, weights, bias, dt=0.05, drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0, coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08))]
    fn from_weights(
        spike_dim: usize,
        n_osc: usize,
        weights: Bound<'_, PyArray1<f32>>,
        bias: Bound<'_, PyArray1<f32>>,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        freq_min: f32,
        freq_max: f32,
        coupling_w12: f32,
        coupling_w13: f32,
        coupling_w23: f32,
    ) -> Self {
        let w = weights.readonly().to_vec().expect("from_weights: weights");
        let b = bias.readonly().to_vec().expect("from_weights: bias");
        Self {
            inner: crate::train::spigrad::SpigradBridge::from_weights(
                spike_dim, n_osc, dt, drive_gain, alpha,
                freq_min, freq_max,
                coupling_w12, coupling_w13, coupling_w23,
                &w, &b,
            ),
        }
    }

    /// Forward pass: spikes → drives → T3 → PST.
    ///
    /// Args:
    ///     spikes: np.ndarray (T, spike_dim) — spike features per timestep
    ///
    /// Returns:
    ///     (pst_mat: np.ndarray (T, pst_dim), trajectory: T2SpigradTrajectory)
    #[pyo3(name = "forward")]
    fn py_forward<'py>(
        &self,
        spikes: Bound<'_, PyArray2<f32>>,
        n_steps: usize,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let spikes_ro = spikes.readonly();
        let spikes_slice = spikes_ro.as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("spikes: {:?}", e)))?;

        let (_, traj) = self.inner.forward(spikes_slice, n_steps);
        let pst_mat = crate::train::spike::t3spike_compute_pst_matrix(&traj);
        let pst_dim = crate::t3::pst_linear_dim(self.inner.n_osc);

        let pst_arr = PyArray2::from_vec2(
            py,
            &pst_mat.chunks(pst_dim).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

        let py_traj = PyT2SpigradTrajectory { inner: traj };
        let tuple = PyTuple::new(py, &[pst_arr.into_py(py), py_traj.into_py(py)])?.unbind();
        Ok(tuple.into())
    }

    /// Backward pass: BPTT through T3 → gradient on bridge weights.
    ///
    /// Args:
    ///     grad_pst: np.ndarray (pst_dim,) — dLoss/dPST at final step
    ///     traj: T2SpigradTrajectory — from forward()
    ///     spikes: np.ndarray (T, spike_dim) — same as forward input
    ///     per_step_grad_pst: optional np.ndarray (T, pst_dim) — per-step PST gradients
    ///
    /// Returns:
    ///     (grad_w: np.ndarray (spike_dim, drive_dim), grad_b: np.ndarray (drive_dim,))
    #[pyo3(name = "backward")]
    fn py_backward(
        &self,
        grad_pst: Bound<'_, PyArray1<f32>>,
        traj: &PyT2SpigradTrajectory,
        spikes: Bound<'_, PyArray2<f32>>,
        per_step_grad_pst: Option<Bound<'_, PyArray2<f32>>>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let gp = grad_pst.readonly().to_vec()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("grad_pst: {:?}", e)))?;
        let spikes_ro = spikes.readonly();
        let sp = spikes_ro.as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("spikes: {:?}", e)))?;

        let per_step = per_step_grad_pst.map(|arr| {
            let ro = arr.readonly();
            ro.as_slice()
                .map(|s| s.to_vec())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("per_step_grad_pst: {:?}", e)))
        }).transpose()?;

        let (gw, gb) = self.inner.backward(&gp, &traj.inner, sp, per_step.as_deref());

        let sd = self.inner.spike_dim;
        let dd = self.inner.drive_dim;

        let gw_arr = PyArray2::from_vec2(
            py,
            &gw.chunks(dd).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

        let gb_arr = PyArray1::from_vec(py, gb);
        let tuple = PyTuple::new(py, &[gw_arr.into_py(py), gb_arr.into_py(py)])?.unbind();
        Ok(tuple.into())
    }

    /// Update bridge weights with gradient descent.
    #[pyo3(name = "update")]
    fn py_update(&mut self, grad_w: Bound<'_, PyArray2<f32>>, grad_b: Bound<'_, PyArray1<f32>>, lr: f32) -> PyResult<()> {
        let gw_ro = grad_w.readonly();
        let gw = gw_ro.as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("grad_w: {:?}", e)))?;
        let gb_ro = grad_b.readonly();
        let gb = gb_ro.to_vec()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("grad_b: {:?}", e)))?;
        self.inner.update(gw, &gb, lr);
        Ok(())
    }

    /// Compute drives from spikes without running T3.
    #[pyo3(name = "compute_drives")]
    fn py_compute_drives(&self, spikes: Bound<'_, PyArray2<f32>>, n_steps: usize) -> PyResult<Vec<f32>> {
        let spikes_ro = spikes.readonly();
        let sp = spikes_ro.as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("spikes: {:?}", e)))?;
        Ok(self.inner.compute_drives(sp, n_steps))
    }

    /// Get bridge weights and bias.
    fn weights(&self) -> Vec<f32> { self.inner.w.clone() }
    fn bias(&self) -> Vec<f32> { self.inner.b.clone() }
    fn spike_dim(&self) -> usize { self.inner.spike_dim }
    fn drive_dim(&self) -> usize { self.inner.drive_dim }
    fn n_osc(&self) -> usize { self.inner.n_osc }

    fn __repr__(&self) -> String {
        format!("SpigradBridge(spike_dim={}, n_osc={}, drive_dim={})",
            self.inner.spike_dim, self.inner.n_osc, self.inner.drive_dim)
    }
}

/// Train a SpigradBridge on connectome spikes to chase a moving target.
#[cfg(feature = "connectome")]
#[pyfunction]
#[pyo3(name = "train_connectome_chase")]
pub fn py_train_connectome_chase(
    conn: &super::connectome_brain::PyConnectomeData,
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    n_epochs: usize,
    spike_dim: usize,
    readout_idx: Vec<usize>,
    lr_bridge: f32,
    lr_decoder: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    dt_ms: f32,
    tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    seed: u64,
) -> PyResult<(PySpigradBridge, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let config = crate::train::connectome_chase::ChaseConfig {
        n_osc,
        n_steps,
        dt,
        drive_gain,
        alpha,
        freq_min,
        freq_max,
        coupling_w12,
        coupling_w13,
        coupling_w23,
        spike_dim,
        lr_bridge,
        lr_decoder,
        n_epochs,
        grad_clip: 5.0,
        dt_ms,
        readout_idx,
        tau_m,
        v_rest,
        v_thresh,
        v_reset,
    };

    let n = conn.inner.n_neurons;
    let mut ep = 0usize;

    let result = crate::train::connectome_chase::train_connectome_chase(
        &conn.inner,
        &config,
        || {
            use rand::Rng;
            let mut ext = vec![0.0f32; n];
            let mut rng = rand::thread_rng();
            for i in 0..n.min(100) {
                ext[i] = rng.gen_range(10.0..40.0);
            }
            ep += 1;
            ext
        },
        seed,
    );

    let py_bridge = PySpigradBridge { inner: result.bridge };
    Ok((py_bridge, result.decoder_w, result.decoder_b, result.loss_history))
}



