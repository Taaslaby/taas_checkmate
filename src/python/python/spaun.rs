//! Python bindings for SPAUN cognitive pipeline — чистый CPU, супер-гибкие вызовы.
//!
//! Вся динамика на CPU (SIMD+rayon). Для 1024-5120 нейронов
//! GPU-оверхед не окупается — прямой NLMS на CPU быстрее любых H2D/D2H копирований.
//!
//! Usage:
//! ```python
//! from taas_checkmate import SpaunPipeline
//!
//! # Load from .taas file (ConnectomeGPT)
//! pipe = SpaunPipeline("brain.taas")
//!
//! # Конфигурируем архитектуру
//! pipe.set_n_outputs(10)
//! pipe.set_n_pixels(100)
//! pipe.set_sp_dim(128)
//! pipe.set_cortical_dim(512)
//! pipe.set_workspace_size(64)
//! pipe.set_wm_slots(4)
//!
//! # Обучаем readout
//! loss = pipe.train(n_epochs=50, steps=10, lr=0.1)
//!
//! # Когнитивные задачи
//! digit = pipe.copy(3, steps=10)
//! seq = pipe.serial_recall([3, 7, 1], steps=10)
//! out = pipe.counting(3, steps=10)
//!
//! # Прямой доступ к numpy
//! import numpy as np
//! pipe.encode(np.zeros(100, dtype=np.float32))
//! ```
//!
//! Build:  cargo build --release --features python

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods, IntoPyArray};
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use rand::Rng;
use rand::SeedableRng;
use std::path::PathBuf;
use std::collections::HashMap;

use taas_connectome::ConnectomeData;
use taas_format::{TaasReader, load_connectome};
use karmac::gnw::GlobalWorkspace;
use karmac::working_memory::WorkingMemory;
use karmac::semantic_pointer::SemanticPointer;

/// Python-доступная SPAUN pipeline.
///
/// Чистый CPU (SIMD+rayon). Все внутренние состояния доступны как numpy массивы.
#[pyclass(name = "SpaunPipeline", module = "taas_checkmate.spaun")]
pub struct PySpaunPipeline {
    // ── Core components ──
    gnw: GlobalWorkspace,
    wm: WorkingMemory,
    sp: SemanticPointer,

    // ── Connectome ──
    conn: ConnectomeData,
    state: Vec<f32>,

    // ── Encoding matrices (rand-init) ──
    w_encode: Vec<f32>,
    w_sp_to_cortex: Vec<f32>,
    w_motor: Vec<f32>,

    // ── Dimensions ──
    n_pixels: usize,
    sp_dim: usize,
    cortical_dim: usize,
    n_neurons: usize,
    n_outputs: usize,
    workspace_size: usize,
    wm_slots: usize,

    // ── Cached group indices ──
    sensory_indices: Vec<usize>,

    // ── Seed ──
    seed: u64,
}

#[pymethods]
impl PySpaunPipeline {
    // ═══════════════════════════════════════════════════════════
    //  Constructor
    // ═══════════════════════════════════════════════════════════

    /// Create SPAUN pipeline from a .taas connectome file.
    ///
    /// Args:
    ///     brain_path: path to .taas file (from ConnectomeGPT).
    ///
    /// Example:
    ///     pipe = SpaunPipeline("brain.taas")
    #[new]
    #[pyo3(signature = (brain_path=None, seed=42))]
    pub fn new(brain_path: Option<String>, seed: u64) -> PyResult<Self> {
        let conn = match brain_path {
            Some(path) => {
                let p = PathBuf::from(&path);
                let reader = TaasReader::from_file(p.as_os_str().to_str().unwrap())
                    .map_err(|e| PyRuntimeError::new_err(format!(
                        "Failed to load .taas: {e}"
                    )))?;
                load_connectome(&reader)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse connectome: {e}")))?
            }
            None => {
                // Generate a minimal default connectome (for testing)
                let n = 256;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let row_ptr: Vec<i32> = (0..=n as i32).collect();
                let col_indices: Vec<i32> = (0..n as i32).map(|i| (i + 1) % n as i32).collect();
                let values: Vec<f32> = (0..n).map(|_| rng.gen::<f32>() * 0.5 - 0.25).collect();
                let nt_types: Vec<f32> = (0..n).map(|i| if i < n * 3 / 4 { 1.0 } else { -1.0 }).collect();
                let groups = [("sensory".to_string(), (0..22.min(n)).collect::<Vec<_>>()),
                    ("mn".to_string(), (n-5..n).collect::<Vec<_>>())];
                let groups_refs: Vec<(&str, Vec<usize>)> = groups.iter()
                    .map(|(n, i)| (n.as_str(), i.clone())).collect();
                ConnectomeData::new(row_ptr, col_indices, values, nt_types, groups_refs)
            }
        };

        let n_neurons = conn.n_neurons;
        let n_pixels = 100;
        let sp_dim = 128.min(n_neurons.next_power_of_two());
        let cortical_dim = n_neurons.min(512);
        let ws = (n_neurons / 8).max(8).min(64);
        let n_outputs = 10;
        let n_wm_slots = 4;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Encoding: pixels → SP
        let enc_scale = (2.0 / (n_pixels + sp_dim) as f32).sqrt();
        let w_encode: Vec<f32> = (0..sp_dim * n_pixels)
            .map(|_| rng.gen::<f32>() * 2.0 * enc_scale - enc_scale).collect();

        // SP → cortex
        let sp2c_scale = (2.0 / (sp_dim + cortical_dim) as f32).sqrt();
        let w_sp_to_cortex: Vec<f32> = (0..cortical_dim * sp_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * sp2c_scale - sp2c_scale).collect();

        // Motor readout
        let m_scale = (2.0 / (cortical_dim + n_outputs) as f32).sqrt();
        let w_motor: Vec<f32> = (0..n_outputs * cortical_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * m_scale - m_scale).collect();

        let sensory = conn.group("sensory").unwrap_or(&[]).to_vec();

        Ok(Self {
            gnw: GlobalWorkspace::new(cortical_dim, ws),
            wm: WorkingMemory::new(n_wm_slots, sp_dim),
            sp: SemanticPointer::new(sp_dim),
            conn,
            state: vec![0.0; n_neurons],
            w_encode, w_sp_to_cortex, w_motor,
            n_pixels, sp_dim, cortical_dim, n_neurons, n_outputs,
            workspace_size: ws, wm_slots: n_wm_slots,
            sensory_indices: sensory,
            seed,
        })
    }

    /// Build pipeline from raw CSR connectome arrays (numpy).
    ///
    /// Args:
    ///     row_ptr: CSR row pointer [n_neurons + 1] int32.
    ///     col_indices: CSR column indices [n_synapses] int32.
    ///     values: CSR weights [n_synapses] float32.
    ///     nt_types: NT types (+1/-1) [n_neurons] float32.
    ///     groups: dict of {name: list[int]} — neuron group indices.
    ///     seed: random seed for encoding matrices.
    #[staticmethod]
    #[pyo3(signature = (row_ptr, col_indices, values, nt_types, groups=None, seed=42))]
    pub fn from_connectome(
        py: Python<'_>,
        row_ptr: Bound<'_, PyArray1<i32>>,
        col_indices: Bound<'_, PyArray1<i32>>,
        values: Bound<'_, PyArray1<f32>>,
        nt_types: Bound<'_, PyArray1<f32>>,
        groups: Option<HashMap<String, Vec<usize>>>,
        seed: u64,
    ) -> PyResult<Self> {
        let rp = row_ptr.readonly();
        let ci = col_indices.readonly();
        let vl = values.readonly();
        let nt = nt_types.readonly();

        let groups_vec: Vec<(String, Vec<usize>)> = match groups {
            Some(g) => g.into_iter().collect(),
            None => vec![],
        };
        let groups_refs: Vec<(&str, Vec<usize>)> = groups_vec.iter()
            .map(|(n, i)| (n.as_str(), i.clone())).collect();

        let conn = ConnectomeData::new(
            rp.as_slice()?.to_vec(),
            ci.as_slice()?.to_vec(),
            vl.as_slice()?.to_vec(),
            nt.as_slice()?.to_vec(),
            groups_refs,
        );

        // Drop Python GIL before heavier Rust init
        drop((rp, ci, vl, nt));
        py.allow_threads(move || {
            Self::from_connectome_inner(conn, seed)
        })
    }

    // ═══════════════════════════════════════════════════════════
    //  Configuration (методы .set_* — вызывать до train/copy)
    // ═══════════════════════════════════════════════════════════

    /// Set number of output classes (default 10 for digits 0-9).
    pub fn set_n_outputs(&mut self, n: usize) {
        self.n_outputs = n;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let m_scale = (2.0 / (self.cortical_dim + n) as f32).sqrt();
        self.w_motor = (0..n * self.cortical_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * m_scale - m_scale).collect();
    }

    /// Set pixel grid size (default 100 = 10×10).
    pub fn set_n_pixels(&mut self, n: usize) {
        self.n_pixels = n;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let enc_scale = (2.0 / (n + self.sp_dim) as f32).sqrt();
        self.w_encode = (0..self.sp_dim * n)
            .map(|_| rng.gen::<f32>() * 2.0 * enc_scale - enc_scale).collect();
    }

    /// Set semantic pointer dimensionality (default 128, must be power of 2).
    pub fn set_sp_dim(&mut self, dim: usize) -> PyResult<()> {
        if !dim.is_power_of_two() {
            return Err(PyValueError::new_err("sp_dim must be power of 2"));
        }
        self.sp_dim = dim;
        self.sp = SemanticPointer::new(dim);
        self.wm = WorkingMemory::new(self.wm_slots, dim);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let enc_scale = (2.0 / (self.n_pixels + dim) as f32).sqrt();
        self.w_encode = (0..dim * self.n_pixels)
            .map(|_| rng.gen::<f32>() * 2.0 * enc_scale - enc_scale).collect();
        let sp2c_scale = (2.0 / (dim + self.cortical_dim) as f32).sqrt();
        self.w_sp_to_cortex = (0..self.cortical_dim * dim)
            .map(|_| rng.gen::<f32>() * 2.0 * sp2c_scale - sp2c_scale).collect();
        Ok(())
    }

    /// Set cortical (GNW) dimension (default min(n_neurons, 512)).
    pub fn set_cortical_dim(&mut self, dim: usize) {
        self.cortical_dim = dim;
        self.gnw = GlobalWorkspace::new(dim, self.workspace_size);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let sp2c_scale = (2.0 / (self.sp_dim + dim) as f32).sqrt();
        self.w_sp_to_cortex = (0..dim * self.sp_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * sp2c_scale - sp2c_scale).collect();
        let m_scale = (2.0 / (dim + self.n_outputs) as f32).sqrt();
        self.w_motor = (0..self.n_outputs * dim)
            .map(|_| rng.gen::<f32>() * 2.0 * m_scale - m_scale).collect();
    }

    /// Set GNW workspace size (default min(n_neurons/8, 64)).
    pub fn set_workspace_size(&mut self, ws: usize) {
        self.workspace_size = ws;
        self.gnw = GlobalWorkspace::new(self.cortical_dim, ws);
    }

    /// Set working memory slots (default 4).
    pub fn set_wm_slots(&mut self, n: usize) {
        self.wm_slots = n;
        self.wm = WorkingMemory::new(n, self.sp_dim);
    }

    // ═══════════════════════════════════════════════════════════
    //  Core pipeline methods
    // ═══════════════════════════════════════════════════════════

    /// Encode pixel vector → semantic pointer → cortex.
    /// Returns numpy array of shape (cortical_dim,).
    pub fn encode<'py>(&self, py: Python<'py>, pixels: Bound<'_, PyArray1<f32>>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = pixels.readonly();
        let sl = ro.as_slice()?;
        if sl.len() != self.n_pixels {
            return Err(PyValueError::new_err(format!(
                "expected {} pixels, got {}", self.n_pixels, sl.len()
            )));
        }

        let sp = self.encode_pixels(sl);
        let cortex = self.sp_to_cortex(&sp);
        Ok(cortex.into_pyarray(py))
    }

    /// Full forward pass: pixels → SP → GNW → connectome → motor logits.
    /// Returns numpy array of shape (n_outputs,).
    pub fn forward<'py>(&mut self, py: Python<'py>, pixels: Bound<'_, PyArray1<f32>>,
        steps: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = pixels.readonly();
        let sl = ro.as_slice()?;
        if sl.len() != self.n_pixels {
            return Err(PyValueError::new_err(format!(
                "expected {} pixels, got {}", self.n_pixels, sl.len()
            )));
        }

        let sp = self.encode_pixels(sl);
        let mut cortex = self.sp_to_cortex(&sp);

        for _ in 0..steps {
            self.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            self.connectome_step(&cortex)?;
        }

        let logits = self.motor_readout(&cortex);
        Ok(logits.into_pyarray(py))
    }

    /// Train motor readout weights using NLMS.
    ///
    /// Args:
    ///     n_epochs: number of training epochs (default 50).
    ///     steps: GNW+connectome steps per forward pass (default 10).
    ///     lr: learning rate (default 0.1).
    ///
    /// Returns: final average cross-entropy loss.
    #[pyo3(signature = (n_epochs=50, steps=10, lr=0.1))]
    pub fn train(&mut self, py: Python<'_>, n_epochs: usize, steps: usize, lr: f32) -> PyResult<f32> {
        let loss = py.allow_threads(move || {
            self.train_inner(n_epochs, steps, lr)
        })?;
        Ok(loss)
    }

    /// Task: Copy — see digit, reproduce it.
    /// Args:
    ///     digit: digit to copy (0-9).
    ///     steps: processing steps per forward pass (default 10).
    /// Returns: predicted digit.
    #[pyo3(signature = (digit, steps=10))]
    pub fn copy(&mut self, digit: usize, steps: usize) -> PyResult<usize> {
        let pixels = self.digit_pixels(digit, self.seed);
        let sp = self.encode_pixels(&pixels);
        let mut cortex = self.sp_to_cortex(&sp);
        for _ in 0..steps {
            self.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            self.connectome_step(&cortex)?;
        }
        let logits = self.motor_readout(&cortex);
        Ok(PySpaunPipeline::argmax(&logits))
    }

    /// Task: Serial Recall — store sequence, recall in order.
    ///
    /// Args:
    ///     sequence: list of digits to remember.
    ///     steps: steps per recall.
    /// Returns: list of recalled digits.
    #[pyo3(signature = (sequence, steps=10))]
    pub fn serial_recall(&mut self, sequence: Vec<usize>, steps: usize) -> PyResult<Vec<usize>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut output = Vec::new();

        // Store
        for &digit in &sequence {
            let pixels = self.digit_pixels(digit, rng.gen());
            let sp = self.encode_pixels(&pixels);
            self.wm.store(&sp);
        }

        // Recall
        for slot in 0..sequence.len() {
            let sp = self.wm.recall(slot);
            let mut cortex = self.sp_to_cortex(&sp);
            for _ in 0..steps {
                self.gnw.forward(&mut cortex);
                for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
                self.connectome_step(&cortex)?;
            }
            let logits = self.motor_readout(&cortex);
            output.push(PySpaunPipeline::argmax(&logits));
        }

        Ok(output)
    }

    /// Task: Counting — see N, output 1,2,...,N.
    ///
    /// Args:
    ///     n: count up to N.
    ///     steps: steps per count.
    /// Returns: list of digits 1..N.
    #[pyo3(signature = (n, steps=10))]
    pub fn counting(&mut self, n: usize, steps: usize) -> PyResult<Vec<usize>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut output = Vec::new();

        // Encode N as input
        let mut n_pixels = vec![0.0f32; self.n_pixels];
        let n_fill = n.min(self.n_pixels);
        for i in 0..n_fill {
            n_pixels[i] = 1.0;
        }
        let n_sp = self.encode_pixels(&n_pixels);
        let n_cortex = self.sp_to_cortex(&n_sp);

        for count in 1..=n {
            let count_pixels = self.digit_pixels(count, rng.gen());
            let count_sp = self.encode_pixels(&count_pixels);
            let combined = self.sp.superpose(&[&n_sp, &count_sp]);
            let mut cortex = self.sp_to_cortex(&combined);
            for i in 0..self.cortical_dim {
                cortex[i] += n_cortex[i] * 0.3;
            }
            for _ in 0..steps {
                self.gnw.forward(&mut cortex);
                for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
                self.connectome_step(&cortex)?;
            }
            let logits = self.motor_readout(&cortex);
            output.push(PySpaunPipeline::argmax(&logits));
        }

        Ok(output)
    }

    // ═══════════════════════════════════════════════════════════
    //  Properties (getters)
    // ═══════════════════════════════════════════════════════════

    /// Current motor readout weights as numpy array [n_outputs × cortical_dim].
    pub fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.w_motor.clone().into_pyarray(py)
    }

    /// Current connectome state (after tanh CSR dynamics) as numpy [n_neurons].
    pub fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.state.clone().into_pyarray(py)
    }

    /// Number of neurons in the connectome.
    pub fn n_neurons(&self) -> usize { self.n_neurons }

    /// Number of synapses in the connectome.
    pub fn n_synapses(&self) -> usize { self.conn.n_synapses }

    /// Number of output classes.
    pub fn n_outputs(&self) -> usize { self.n_outputs }

    /// Semantic pointer dimension.
    pub fn sp_dim(&self) -> usize { self.sp_dim }

    /// Cortical (GNW) dimension.
    pub fn cortical_dim(&self) -> usize { self.cortical_dim }

    /// List connectome group names with their sizes.
    pub fn groups(&self) -> Vec<(String, usize)> {
        self.conn.groups.iter()
            .map(|g| (g.name.clone(), g.indices.len()))
            .collect()
    }

    /// Get connectome group indices by name.
    pub fn group(&self, name: &str) -> Option<Vec<usize>> {
        self.conn.group(name).map(|s| s.to_vec())
    }

    /// Get current working memory buffer as numpy [n_slots × sp_dim].
    pub fn working_memory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.wm.recall_all().to_vec().into_pyarray(py)
    }

    /// Raw CSR row pointer (read-only copy).
    pub fn row_ptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.conn.row_ptr.clone().into_pyarray(py)
    }

    /// Raw CSR column indices (read-only copy).
    pub fn col_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.conn.col_indices.clone().into_pyarray(py)
    }

    /// Raw CSR values (read-only copy).
    pub fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.conn.values.clone().into_pyarray(py)
    }
}

// ═══════════════════════════════════════════════════════════
//  Internal methods (pure Rust, no Python)
// ═══════════════════════════════════════════════════════════

impl PySpaunPipeline {
    fn from_connectome_inner(conn: ConnectomeData, seed: u64) -> PyResult<Self> {
        let n_neurons = conn.n_neurons;
        let n_pixels = 100;
        let sp_dim = 128.min(n_neurons.next_power_of_two());
        let cortical_dim = n_neurons.min(512);
        let ws = (n_neurons / 8).max(8).min(64);
        let n_outputs = 10;
        let n_wm_slots = 4;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let enc_scale = (2.0 / (n_pixels + sp_dim) as f32).sqrt();
        let w_encode: Vec<f32> = (0..sp_dim * n_pixels)
            .map(|_| rng.gen::<f32>() * 2.0 * enc_scale - enc_scale).collect();

        let sp2c_scale = (2.0 / (sp_dim + cortical_dim) as f32).sqrt();
        let w_sp_to_cortex: Vec<f32> = (0..cortical_dim * sp_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * sp2c_scale - sp2c_scale).collect();

        let m_scale = (2.0 / (cortical_dim + n_outputs) as f32).sqrt();
        let w_motor: Vec<f32> = (0..n_outputs * cortical_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * m_scale - m_scale).collect();

        let sensory = conn.group("sensory").unwrap_or(&[]).to_vec();

        Ok(Self {
            gnw: GlobalWorkspace::new(cortical_dim, ws),
            wm: WorkingMemory::new(n_wm_slots, sp_dim),
            sp: SemanticPointer::new(sp_dim),
            conn,
            state: vec![0.0; n_neurons],
            w_encode, w_sp_to_cortex, w_motor,
            n_pixels, sp_dim, cortical_dim, n_neurons, n_outputs,
            workspace_size: ws, wm_slots: n_wm_slots,
            sensory_indices: sensory,
            seed,
        })
    }

    /// Encode pixel vector into semantic pointer.
    fn encode_pixels(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(pixels.len(), self.n_pixels);
        let mut sp = vec![0.0f32; self.sp_dim];
        for j in 0..self.sp_dim {
            let row = &self.w_encode[j * self.n_pixels..(j + 1) * self.n_pixels];
            sp[j] = row.iter().zip(pixels.iter()).map(|(w, p)| w * p).sum();
        }
        self.sp.normalize(&sp)
    }

    /// Project semantic pointer into cortex space.
    fn sp_to_cortex(&self, sp: &[f32]) -> Vec<f32> {
        let mut cortex = vec![0.0f32; self.cortical_dim];
        for j in 0..self.cortical_dim {
            let row = &self.w_sp_to_cortex[j * self.sp_dim..(j + 1) * self.sp_dim];
            cortex[j] = row.iter().zip(sp.iter()).map(|(w, s)| w * s).sum();
        }
        cortex
    }

    /// Run one step of connectome dynamics на CPU.
    fn connectome_step(&mut self, input: &[f32]) -> PyResult<()> {
        self.connectome_step_cpu(input);
        Ok(())
    }

    /// CPU connectome step: CSR propagate + input injection into sensory group.
    fn connectome_step_cpu(&mut self, input: &[f32]) {
        let n = self.n_neurons;
        let mut new_state = vec![0.0f32; n];
        let sensory = &self.sensory_indices;

        for i in 0..n {
            let start = self.conn.row_ptr[i].max(0) as usize;
            let end = self.conn.row_ptr[i + 1].max(0) as usize;
            let mut sum = 0.0f32;
            for idx in start..end {
                if idx >= self.conn.col_indices.len() || idx >= self.conn.values.len() { break; }
                let j = self.conn.col_indices[idx];
                if j < 0 || j as usize >= n { continue; }
                sum += self.conn.values[idx] * self.state[j as usize];
            }
            // Input injection into sensory group
            if let Some(pos) = sensory.iter().position(|&s| s == i) {
                if pos < input.len() {
                    sum += input[pos] * 0.5;
                }
            }
            new_state[i] = sum.tanh();
        }
        self.state.copy_from_slice(&new_state);
    }


    /// Motor readout from cortex.
    fn motor_readout(&self, cortex: &[f32]) -> Vec<f32> {
        let n_use = cortex.len().min(self.cortical_dim);
        let mut logits = vec![0.0f32; self.n_outputs];
        for k in 0..self.n_outputs {
            let row_start = k * self.cortical_dim;
            let mut sum = 0.0f32;
            for j in 0..n_use {
                sum += self.w_motor[row_start + j] * cortex[j];
            }
            logits[k] = sum;
        }
        logits
    }

    /// Train inner (called from Python's train() with GIL released).
    fn train_inner(&mut self, n_epochs: usize, steps: usize, lr: f32) -> PyResult<f32> {
        let mut final_loss = 0.0f32;

        for epoch in 0..n_epochs {
            let mut epoch_loss = 0.0f32;

            for digit in 0..10 {
                let pixels = self.digit_pixels(digit, self.seed);

                let sp = self.encode_pixels(&pixels);
                let mut cortex = self.sp_to_cortex(&sp);
                for _ in 0..steps {
                    self.gnw.forward(&mut cortex);
                    for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
                    self.connectome_step(&cortex)?;
                }

                // Prediction
                let logits = self.motor_readout(&cortex);
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
                let sum: f32 = exps.iter().sum();
                if sum > 1e-10 { for e in &mut exps { *e /= sum; } }

                let prob = exps[digit].max(1e-10);
                epoch_loss -= prob.ln();

                // NLMS update (CPU — для 5120 весов GPU-оверхед не окупается)
                self.train_nlms_cpu(&cortex, &exps, digit, lr);
            }

            final_loss = epoch_loss / 10.0;
            if (epoch + 1) % 5 == 0 || epoch == 0 || epoch == n_epochs - 1 {
                println!("  Epoch {:3}/{}: avg CE loss = {:.4}", epoch + 1, n_epochs, final_loss);
            }
        }

        Ok(final_loss)
    }

    /// CPU NLMS training.
    fn train_nlms_cpu(&mut self, cortex: &[f32], exps: &[f32], digit: usize, lr: f32) {
        let n_out = self.n_outputs;
        let n_cortex = self.cortical_dim;
        for k in 0..n_out {
            let target = if k == digit { 1.0f32 } else { 0.0f32 };
            let error = target - exps[k];
            let w_start = k * n_cortex;
            let w_slice = &mut self.w_motor[w_start..w_start + n_cortex];
            nlms_train_weights(w_slice, cortex, error, lr, 1e-6);
        }
    }

    /// Argmax.
    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Generate digit pixel pattern (grid, deterministic from seed).
    fn digit_pixels(&self, digit: usize, base_seed: u64) -> Vec<f32> {
        let side = (self.n_pixels as f32).sqrt() as usize;
        let mut pixels = vec![0.0f32; self.n_pixels];
        let mut local_rng = rand::rngs::StdRng::seed_from_u64(digit as u64 * 1337 + base_seed);
        for p in pixels.iter_mut() {
            if local_rng.gen::<f32>() < 0.3 {
                *p = 1.0;
            }
        }
        let center_boost = (digit as f32) * 0.05;
        let c_start = side / 2 - 2;
        let c_end = (c_start + 4).min(side);
        for i in c_start..c_end {
            for j in c_start..c_end {
                let idx = i * side + j;
                if idx < self.n_pixels {
                    pixels[idx] += center_boost;
                }
            }
        }
        pixels
    }
}

/// NLMS training helper: w += lr·error·x / (‖x‖² + ε).
fn nlms_train_weights(weights: &mut [f32], state: &[f32], error: f32, lr: f32, eps: f32) {
    let n = weights.len().min(state.len());
    let norm2: f32 = state.iter().map(|x| x * x).sum::<f32>().max(eps);
    let norm_lr = lr / norm2;
    for i in 0..n {
        weights[i] += norm_lr * error * state[i];
    }
}
