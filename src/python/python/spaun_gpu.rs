#![allow(non_snake_case)]

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use std::ffi::{c_float, c_int};

use karmac::gnw::GlobalWorkspace;
use karmac::semantic_pointer::SemanticPointer;
use karmac::working_memory::WorkingMemory;
use taas_connectome::ConnectomeData;
use taas_format::{load_connectome, TaasReader};

use cuda_kernels::{csr_spmv_tanh as cuda_csr_spmv_tanh, csr_stdp_dopamine_devptr, DevicePtr};

/// GPU SPAUN pipeline: 1M+ нейронов на CUDA.
#[pyclass(name = "SpaunPipelineGPU", module = "taas_checkmate.spaun_gpu")]
pub struct PySpaunPipelineGPU {
    gnw: GlobalWorkspace,
    wm: WorkingMemory,
    sp: SemanticPointer,
    conn: ConnectomeData,

    n_pixels: usize,
    sp_dim: usize,
    cortical_dim: usize,
    n_neurons: usize,
    n_synapses: usize,
    n_outputs: usize,
    workspace_size: usize,
    wm_slots: usize,

    w_encode: Vec<f32>,
    w_sp_to_cortex: Vec<f32>,
    w_motor: Vec<f32>,
    sensory_indices: Vec<usize>,

    // GPU state — отдельные поля, чтобы borrow checker видел disjointness
    d_state_in: Option<DevicePtr<f32>>,
    d_state_out: Option<DevicePtr<f32>>,
    d_input_vec: Option<DevicePtr<f32>>,
    d_row_ptr: Option<DevicePtr<i32>>,
    d_col_indices: Option<DevicePtr<i32>>,
    d_values: Option<DevicePtr<f32>>,
    d_trace: Option<DevicePtr<f32>>,
    gpu_ready: bool,
    state_cpu: Vec<f32>,
    trace_cpu: Vec<f32>,
    seed: u64,
}

#[pymethods]
impl PySpaunPipelineGPU {
    #[new]
    #[pyo3(signature = (brain_path=None, seed=42))]
    pub fn new(brain_path: Option<String>, seed: u64) -> PyResult<Self> {
        let conn = match brain_path {
            Some(path) => {
                let reader = TaasReader::from_file(&path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Load .taas: {e}")))?;
                load_connectome(&reader)
                    .map_err(|e| PyRuntimeError::new_err(format!("Parse connectome: {e}")))?
            }
            None => {
                let n = 256;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                ConnectomeData::new(
                    (0..=n as i32).collect(),
                    (0..n as i32).map(|i| (i + 1) % n as i32).collect(),
                    (0..n).map(|_| rng.gen::<f32>() * 0.5 - 0.25).collect(),
                    (0..n)
                        .map(|i| if i < n * 3 / 4 { 1.0 } else { -1.0 })
                        .collect(),
                    [
                        ("sensory", (0..22.min(n)).collect()),
                        ("mn", (n - 5..n).collect()),
                    ]
                    .to_vec(),
                )
            }
        };
        let n_neurons = conn.n_neurons;
        let n_synapses = conn.n_synapses;
        let sensory_indices = conn.group("sensory").unwrap_or(&[]).to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let sp_dim = 128.min(n_neurons.next_power_of_two());
        let cortical_dim = n_neurons.min(512);
        let ws = (n_neurons / 8).max(8).min(64);

        Ok(Self {
            gnw: GlobalWorkspace::new(cortical_dim, ws),
            wm: WorkingMemory::new(4, sp_dim),
            sp: SemanticPointer::new(sp_dim),
            conn,
            n_pixels: 100,
            sp_dim,
            cortical_dim,
            n_neurons,
            n_synapses,
            n_outputs: 10,
            workspace_size: ws,
            wm_slots: 4,
            w_encode: init_weights(sp_dim, 100, &mut rng),
            w_sp_to_cortex: init_weights(cortical_dim, sp_dim, &mut rng),
            w_motor: init_weights(10, cortical_dim, &mut rng),
            sensory_indices,
            d_state_in: None,
            d_state_out: None,
            d_input_vec: None,
            d_row_ptr: None,
            d_col_indices: None,
            d_values: None,
            d_trace: None,
            gpu_ready: false,
            state_cpu: vec![0.0; n_neurons],
            trace_cpu: vec![0.0; n_neurons],
            seed,
        })
    }

    pub fn set_n_outputs(&mut self, n: usize) {
        self.n_outputs = n;
        self.w_motor = init_weights(
            n,
            self.cortical_dim,
            &mut rand::rngs::StdRng::seed_from_u64(self.seed),
        );
    }
    pub fn set_n_pixels(&mut self, n: usize) {
        self.n_pixels = n;
        self.w_encode = init_weights(
            self.sp_dim,
            n,
            &mut rand::rngs::StdRng::seed_from_u64(self.seed),
        );
    }
    pub fn set_sp_dim(&mut self, dim: usize) -> PyResult<()> {
        if !dim.is_power_of_two() {
            return Err(PyValueError::new_err("sp_dim must be power of 2"));
        }
        self.sp_dim = dim;
        self.sp = SemanticPointer::new(dim);
        self.wm = WorkingMemory::new(self.wm_slots, dim);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        self.w_encode = init_weights(dim, self.n_pixels, &mut rng);
        self.w_sp_to_cortex = init_weights(self.cortical_dim, dim, &mut rng);
        Ok(())
    }
    pub fn set_cortical_dim(&mut self, dim: usize) {
        self.cortical_dim = dim;
        self.gnw = GlobalWorkspace::new(dim, self.workspace_size);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        self.w_sp_to_cortex = init_weights(dim, self.sp_dim, &mut rng);
        self.w_motor = init_weights(self.n_outputs, dim, &mut rng);
    }
    pub fn set_workspace_size(&mut self, ws: usize) {
        self.workspace_size = ws;
        self.gnw = GlobalWorkspace::new(self.cortical_dim, ws);
    }
    pub fn set_wm_slots(&mut self, n: usize) {
        self.wm_slots = n;
        self.wm = WorkingMemory::new(n, self.sp_dim);
    }

    pub fn forward<'py>(
        &mut self,
        py: Python<'py>,
        pixels: Bound<'_, PyArray1<f32>>,
        steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = pixels.readonly();
        let sl = ro.as_slice()?;
        if sl.len() != self.n_pixels {
            return Err(PyValueError::new_err(format!(
                "expected {} pixels, got {}",
                self.n_pixels,
                sl.len()
            )));
        }
        let mut cortex = self.sp_to_cortex(&self.encode_pixels(sl));
        for _ in 0..steps {
            self.gnw.forward(&mut cortex);
            for c in &mut cortex {
                *c = c.clamp(-5.0, 5.0);
            }
            self.gpu_step(&cortex)?;
        }
        Ok(self.motor_readout(&cortex).into_pyarray(py))
    }

    #[pyo3(signature = (n_epochs=50, steps=10, lr=0.2))]
    pub fn train(
        &mut self,
        py: Python<'_>,
        n_epochs: usize,
        steps: usize,
        lr: f32,
    ) -> PyResult<f32> {
        py.allow_threads(move || self.train_inner(n_epochs, steps, lr))
    }

    #[pyo3(signature = (digit, steps=10))]
    pub fn copy(&mut self, digit: usize, steps: usize) -> PyResult<usize> {
        let mut cortex =
            self.sp_to_cortex(&self.encode_pixels(&self.digit_pixels(digit, self.seed)));
        for _ in 0..steps {
            self.gnw.forward(&mut cortex);
            for c in &mut cortex {
                *c = c.clamp(-5.0, 5.0);
            }
            self.gpu_step(&cortex)?;
        }
        Ok(PySpaunPipelineGPU::argmax(&self.motor_readout(&cortex)))
    }

    #[pyo3(signature = (sequence, steps=10))]
    pub fn serial_recall(&mut self, sequence: Vec<usize>, steps: usize) -> PyResult<Vec<usize>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        for &d in &sequence {
            self.wm
                .store(&self.encode_pixels(&self.digit_pixels(d, rng.gen())));
        }
        let mut out = Vec::new();
        for slot in 0..sequence.len() {
            let mut cortex = self.sp_to_cortex(&self.wm.recall(slot));
            for _ in 0..steps {
                self.gnw.forward(&mut cortex);
                for c in &mut cortex {
                    *c = c.clamp(-5.0, 5.0);
                }
                self.gpu_step(&cortex)?;
            }
            out.push(PySpaunPipelineGPU::argmax(&self.motor_readout(&cortex)));
        }
        Ok(out)
    }

    #[pyo3(signature = (n, steps=10))]
    pub fn counting(&mut self, n: usize, steps: usize) -> PyResult<Vec<usize>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut out = Vec::new();
        let mut np = vec![0.0; self.n_pixels];
        for i in 0..n.min(self.n_pixels) {
            np[i] = 1.0;
        }
        let n_sp = self.encode_pixels(&np);
        let n_cortex = self.sp_to_cortex(&n_sp);
        for count in 1..=n {
            let combined = self.sp.superpose(&[
                &n_sp,
                &self.encode_pixels(&self.digit_pixels(count, rng.gen())),
            ]);
            let mut cortex = self.sp_to_cortex(&combined);
            for i in 0..self.cortical_dim {
                cortex[i] += n_cortex[i] * 0.3;
            }
            for _ in 0..steps {
                self.gnw.forward(&mut cortex);
                for c in &mut cortex {
                    *c = c.clamp(-5.0, 5.0);
                }
                self.gpu_step(&cortex)?;
            }
            out.push(PySpaunPipelineGPU::argmax(&self.motor_readout(&cortex)));
        }
        Ok(out)
    }

    pub fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.w_motor.clone().into_pyarray(py)
    }

    pub fn set_weights(&mut self, _py: Python<'_>, arr: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let ro = arr.readonly();
        let sl = ro.as_slice()?;
        if sl.len() != self.n_outputs * self.cortical_dim {
            return Err(PyValueError::new_err(format!(
                "expected {} weights, got {}",
                self.n_outputs * self.cortical_dim,
                sl.len()
            )));
        }
        self.w_motor.copy_from_slice(sl);
        Ok(())
    }
    pub fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.state_cpu.clone().into_pyarray(py)
    }
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }
    pub fn n_synapses(&self) -> usize {
        self.n_synapses
    }
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }
    pub fn sp_dim(&self) -> usize {
        self.sp_dim
    }
    pub fn cortical_dim(&self) -> usize {
        self.cortical_dim
    }
    pub fn groups(&self) -> Vec<(String, usize)> {
        self.conn
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.indices.len()))
            .collect()
    }
    pub fn group(&self, name: &str) -> Option<Vec<usize>> {
        self.conn.group(name).map(|s| s.to_vec())
    }
    pub fn working_memory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.wm.recall_all().to_vec().into_pyarray(py)
    }

    /// Apply dopamine-gated STDP to connectome weights (on GPU).
    /// delta > 0 → strengthen correlated synapses, delta < 0 → weaken.
    /// Uses device pointers directly — no H2D/D2H copy for weights.
    #[pyo3(signature = (delta=0.1, lr=0.02, w_min=0.01, w_max=3.0))]
    pub fn apply_stdp(&mut self, delta: f32, lr: f32, w_min: f32, w_max: f32) -> PyResult<()> {
        self.ensure_gpu()?;

        // Upload trace_cpu → d_trace (device trace buffer)
        self.d_trace
            .as_ref()
            .unwrap()
            .copy_from_host(&self.trace_cpu)
            .map_err(|e| PyRuntimeError::new_err(format!("upload trace: {e}")))?;

        // Extract raw device pointers — same pattern as gpu_step() for csr_spmv_tanh
        let d_w = self.d_values.as_mut().unwrap().as_mut_ptr() as *mut std::ffi::c_float;
        let d_rp = self.d_row_ptr.as_ref().unwrap().as_ptr() as *const std::ffi::c_int;
        let d_ci = self.d_col_indices.as_ref().unwrap().as_ptr() as *const std::ffi::c_int;
        let d_tr = self.d_trace.as_ref().unwrap().as_ptr() as *const std::ffi::c_float;

        // Run CSR STDP with dopamine on GPU (kernel modifies d_values in-place)
        csr_stdp_dopamine_devptr(
            d_w,
            d_rp,
            d_ci,
            d_tr,
            lr,
            delta,
            w_min,
            w_max,
            self.n_neurons,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("STDP: {e}")))?;

        // Sync CPU-side values with GPU after STDP update
        // (GPU modified d_values in-place, download to stay consistent)
        self.d_values
            .as_ref()
            .unwrap()
            .copy_to_host(&mut self.conn.values)
            .map_err(|e| PyRuntimeError::new_err(format!("sync weights: {e}")))?;

        Ok(())
    }

    /// Reset eligibility traces to zero.
    pub fn reset_traces(&mut self) {
        self.trace_cpu.fill(0.0);
    }

    /// Get current traces as numpy array.
    pub fn get_traces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.trace_cpu.clone().into_pyarray(py)
    }
    pub fn row_ptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.conn.row_ptr.clone().into_pyarray(py)
    }
    pub fn col_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        self.conn.col_indices.clone().into_pyarray(py)
    }
    pub fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.conn.values.clone().into_pyarray(py)
    }
}

// ═══════════════════════════════════════════════════════════
//  Internal
// ═══════════════════════════════════════════════════════════

impl PySpaunPipelineGPU {
    fn ensure_gpu(&mut self) -> PyResult<()> {
        if self.gpu_ready {
            return Ok(());
        }
        let n = self.n_neurons;
        let syn = self.n_synapses;

        self.d_state_in =
            Some(DevicePtr::new(n).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);
        self.d_state_out =
            Some(DevicePtr::new(n).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);
        self.d_input_vec =
            Some(DevicePtr::new(n).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);
        self.d_row_ptr =
            Some(DevicePtr::new(n + 1).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);
        self.d_col_indices =
            Some(DevicePtr::new(syn).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);
        self.d_values =
            Some(DevicePtr::new(syn).map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?);

        self.d_row_ptr
            .as_ref()
            .unwrap()
            .copy_from_host(&self.conn.row_ptr)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;
        self.d_col_indices
            .as_ref()
            .unwrap()
            .copy_from_host(&self.conn.col_indices)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;
        self.d_values
            .as_ref()
            .unwrap()
            .copy_from_host(&self.conn.values)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;

        self.d_trace = Some(
            DevicePtr::new(n).map_err(|e| PyRuntimeError::new_err(format!("GPU trace: {e}")))?,
        );

        let zero = vec![0.0f32; n];
        self.d_state_in
            .as_ref()
            .unwrap()
            .copy_from_host(&zero)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;
        self.d_state_out
            .as_ref()
            .unwrap()
            .copy_from_host(&zero)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;
        self.d_input_vec
            .as_ref()
            .unwrap()
            .copy_from_host(&zero)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;
        self.d_trace
            .as_ref()
            .unwrap()
            .copy_from_host(&zero)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;

        self.gpu_ready = true;
        Ok(())
    }

    fn gpu_step(&mut self, cortex: &[f32]) -> PyResult<()> {
        self.ensure_gpu()?;
        let n = self.n_neurons;

        // Build input_vec: sensory input injection
        let mut iv = vec![0.0f32; n];
        let ns = self.sensory_indices.len().min(cortex.len());
        for (pos, &idx) in self.sensory_indices.iter().enumerate().take(ns) {
            if idx < n {
                iv[idx] = cortex[pos] * 0.5;
            }
        }

        // Upload input (disjoint fields — borrow checker approves)
        self.d_input_vec
            .as_ref()
            .unwrap()
            .copy_from_host(&iv)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU: {e}")))?;

        // CSR SpMV + tanh
        // Extract raw pointers — these don't borrow, just copy the pointer value
        let out_ptr = self.d_state_out.as_mut().unwrap().as_mut_ptr() as *mut c_float;
        let in_ptr = self.d_state_in.as_ref().unwrap().as_ptr() as *const c_float;
        let rp_ptr = self.d_row_ptr.as_ref().unwrap().as_ptr() as *const c_int;
        let ci_ptr = self.d_col_indices.as_ref().unwrap().as_ptr() as *const c_int;
        let vl_ptr = self.d_values.as_ref().unwrap().as_ptr() as *const c_float;
        let iv_ptr = self.d_input_vec.as_ref().unwrap().as_ptr() as *const c_float;

        cuda_csr_spmv_tanh(
            out_ptr, in_ptr, rp_ptr, ci_ptr, vl_ptr, iv_ptr, 1.0, n, None,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("GPU SpMV: {e}")))?;

        // Download: d_state_out → state_cpu
        self.d_state_out
            .as_ref()
            .unwrap()
            .copy_to_host(&mut self.state_cpu)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU download: {e}")))?;

        // Update traces: trace[i] = decay * trace[i] + abs(state[i])
        // We do this on CPU since state is already downloaded
        let decay = 0.95f32;
        for i in 0..n {
            self.trace_cpu[i] = self.trace_cpu[i] * decay + self.state_cpu[i].abs();
        }

        // Swap buffers: state_out becomes next step's state_in
        std::mem::swap(&mut self.d_state_in, &mut self.d_state_out);
        Ok(())
    }

    fn encode_pixels(&self, pixels: &[f32]) -> Vec<f32> {
        let mut sp = vec![0.0; self.sp_dim];
        for j in 0..self.sp_dim {
            sp[j] = self.w_encode[j * self.n_pixels..(j + 1) * self.n_pixels]
                .iter()
                .zip(pixels)
                .map(|(w, p)| w * p)
                .sum();
        }
        self.sp.normalize(&sp)
    }

    fn sp_to_cortex(&self, sp: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0; self.cortical_dim];
        for j in 0..self.cortical_dim {
            c[j] = self.w_sp_to_cortex[j * self.sp_dim..(j + 1) * self.sp_dim]
                .iter()
                .zip(sp)
                .map(|(w, s)| w * s)
                .sum();
        }
        c
    }

    fn motor_readout(&self, cortex: &[f32]) -> Vec<f32> {
        let nu = cortex.len().min(self.cortical_dim);
        let mut logits = vec![0.0; self.n_outputs];
        for k in 0..self.n_outputs {
            logits[k] = (0..nu)
                .map(|j| self.w_motor[k * self.cortical_dim + j] * cortex[j])
                .sum();
        }
        logits
    }

    fn train_inner(&mut self, n_epochs: usize, steps: usize, lr: f32) -> PyResult<f32> {
        self.ensure_gpu()?;
        let zero = vec![0.0f32; self.n_neurons];
        let mut final_loss = 0.0;

        for epoch in 0..n_epochs {
            let mut el = 0.0;
            for digit in 0..10 {
                let mut cortex =
                    self.sp_to_cortex(&self.encode_pixels(&self.digit_pixels(digit, self.seed)));

                // Reset GPU state between digits
                self.d_state_in.as_ref().unwrap().copy_from_host(&zero).ok();
                self.d_state_out
                    .as_ref()
                    .unwrap()
                    .copy_from_host(&zero)
                    .ok();
                self.state_cpu.fill(0.0);

                for _ in 0..steps {
                    self.gnw.forward(&mut cortex);
                    for c in &mut cortex {
                        *c = c.clamp(-5.0, 5.0);
                    }
                    self.gpu_step(&cortex)?;
                }

                let logits = self.motor_readout(&cortex);
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
                let sm: f32 = exps.iter().sum();
                if sm > 1e-10 {
                    for e in &mut exps {
                        *e /= sm;
                    }
                }
                el -= exps[digit].max(1e-10).ln();

                for k in 0..self.n_outputs {
                    let err = (if k == digit { 1.0 } else { 0.0 }) - exps[k];
                    nlms_train_weights(
                        &mut self.w_motor[k * self.cortical_dim..(k + 1) * self.cortical_dim],
                        &cortex,
                        err,
                        lr,
                        1e-6,
                    );
                }
            }
            final_loss = el / 10.0;
            if (epoch + 1) % 5 == 0 || epoch == 0 || epoch == n_epochs - 1 {
                println!(
                    "  Epoch {:3}/{}: avg CE loss = {:.4}",
                    epoch + 1,
                    n_epochs,
                    final_loss
                );
            }
        }
        Ok(final_loss)
    }

    fn digit_pixels(&self, digit: usize, base_seed: u64) -> Vec<f32> {
        let side = (self.n_pixels as f32).sqrt() as usize;
        let mut p = vec![0.0; self.n_pixels];
        let mut rng = rand::rngs::StdRng::seed_from_u64(digit as u64 * 1337 + base_seed);
        for x in &mut p {
            if rng.gen::<f32>() < 0.3 {
                *x = 1.0;
            }
        }
        let cs = side / 2 - 2;
        let ce = (cs + 4).min(side);
        for i in cs..ce {
            for j in cs..ce {
                let idx = i * side + j;
                if idx < self.n_pixels {
                    p[idx] += digit as f32 * 0.05;
                }
            }
        }
        p
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

fn init_weights(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<f32> {
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
        .collect()
}

fn nlms_train_weights(w: &mut [f32], s: &[f32], err: f32, lr: f32, eps: f32) {
    let n = w.len().min(s.len());
    let n2: f32 = s[..n].iter().map(|x| x * x).sum::<f32>().max(eps);
    let nl = lr / n2;
    for i in 0..n {
        w[i] += nl * err * s[i];
    }
}
