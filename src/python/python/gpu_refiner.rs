#![allow(non_snake_case)]

//! PyO3 bindings for GPU-accelerated Kuramoto BPTT ConnectomeRefiner.
//!
//! Full GPU pipeline: forward (D2D trajectory) → loss grad → BPTT → weight update.
//!
//! Usage:
//! ```python
//! from taas_checkmate.gpu_refiner import GpuConnectomeRefiner
//!
//! refiner = GpuConnectomeRefiner("fafb_dros.taas")
//! history = refiner.refine(n_epochs=500, lr=0.08)
//!
//! print(f"Final coherence: {history['coherence'][-1]:.4f}")
//! refiner.save("refined.taas")
//! ```

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::cuda::{GpuContext, DevicePtr};

/// Refinement history record.
#[pyclass(name = "RefinerHistory", module = "taas_checkmate.gpu_refiner")]
#[derive(Debug, Clone)]
pub struct PyRefinerHistory {
    #[pyo3(get)]
    pub epoch: Vec<usize>,
    #[pyo3(get)]
    pub coherence: Vec<f32>,
    #[pyo3(get)]
    pub mean_abs_w: Vec<f32>,
    #[pyo3(get)]
    pub loss: Vec<f32>,
    #[pyo3(get)]
    pub weight_change_norm: Vec<f32>,
}

/// GPU-accelerated ConnectomeRefiner.
///
/// Runs Kuramoto BPTT refinement entirely on GPU.
/// 36× faster than CPU for dense connectomes (576 syn/neu).
///
/// Example:
/// ```python
/// refiner = GpuConnectomeRefiner("fafb_dros.taas")
/// history = refiner.refine(n_epochs=500)
/// refiner.save("refined.taas")
/// ```
#[pyclass(name = "GpuConnectomeRefiner", module = "taas_checkmate.gpu_refiner")]
pub struct PyGpuConnectomeRefiner {
    // Connectome data (always keep host copy for save)
    n: usize,
    s: usize,
    row_ptr: Vec<i32>,
    col_indices: Vec<i32>,
    values: Vec<f32>,
    nt_types: Vec<f32>,

    // GPU context
    #[cfg(feature = "cuda")]
    ctx: Option<GpuContext>,

    // Init phases & omegas (fixed per refiner)
    init_phases: Vec<f32>,
    omegas: Vec<f32>,

    // Most recent run stats
    last_coherence: f32,
    last_mean_w: f32,
}

#[pymethods]
impl PyGpuConnectomeRefiner {
    /// Load a .taas file and prepare for GPU refinement.
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let mut f = std::fs::File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot open {path}: {e}")))?;

        let conn = taas_connectome::ConnectomeData::load_baked(&mut f)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot load .taas: {e}")))?;

        let n = conn.n_neurons;
        let s = conn.n_synapses;

        if n == 0 || s == 0 {
            return Err(PyValueError::new_err("Empty connectome"));
        }

        // Init phases & omegas (reproducible, same as Rust example)
        let mut rng = StdRng::seed_from_u64(42);
        let init_phases: Vec<f32> = (0..n)
            .map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU)
            .collect();
        let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();

        // Initialize GPU
        #[cfg(feature = "cuda")]
        {
            let n_gpus = crate::cuda::get_device_count().unwrap_or(0);
            if n_gpus == 0 {
                return Err(PyRuntimeError::new_err("No CUDA devices found"));
            }
            let _ = crate::cuda::set_device(0);
        }

        Ok(Self {
            n,
            s,
            row_ptr: conn.row_ptr,
            col_indices: conn.col_indices,
            values: conn.values,
            nt_types: conn.nt_types,
            init_phases,
            omegas,
            #[cfg(feature = "cuda")]
            ctx: None,
            last_coherence: 0.0,
            last_mean_w: 0.0,
        })
    }

    /// Create from raw CSR arrays (numpy).
    #[staticmethod]
    pub fn from_csr(
        row_ptr: Bound<'_, PyArray1<i32>>,
        col_indices: Bound<'_, PyArray1<i32>>,
        values: Bound<'_, PyArray1<f32>>,
        nt_types: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Self> {
        let rp = row_ptr.readonly();
        let ci = col_indices.readonly();
        let vl = values.readonly();
        let nt = nt_types.readonly();

        let rp_s = rp.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("row_ptr: {e}")))?;
        let ci_s = ci.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("col_indices: {e}")))?;
        let vl_s = vl.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("values: {e}")))?;
        let nt_s = nt.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("nt_types: {e}")))?;

        let n = rp_s.len() - 1;
        let s = vl_s.len();

        if n == 0 || s == 0 {
            return Err(PyValueError::new_err("Empty connectome"));
        }

        let mut rng = StdRng::seed_from_u64(42);
        let init_phases: Vec<f32> = (0..n)
            .map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU)
            .collect();
        let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();

        #[cfg(feature = "cuda")]
        {
            let n_gpus = crate::cuda::get_device_count().unwrap_or(0);
            if n_gpus == 0 {
                return Err(PyRuntimeError::new_err("No CUDA devices found"));
            }
            let _ = crate::cuda::set_device(0);
        }

        Ok(Self {
            n,
            s,
            row_ptr: rp_s.to_vec(),
            col_indices: ci_s.to_vec(),
            values: vl_s.to_vec(),
            nt_types: nt_s.to_vec(),
            init_phases,
            omegas,
            #[cfg(feature = "cuda")]
            ctx: None,
            last_coherence: 0.0,
            last_mean_w: 0.0,
        })
    }

    // ═══════════════════════════════════════════════════
    //  Properties
    // ═══════════════════════════════════════════════════

    #[getter]
    pub fn n_neurons(&self) -> usize { self.n }

    #[getter]
    pub fn n_synapses(&self) -> usize { self.s }

    #[getter]
    pub fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.values.clone()))
    }

    #[getter]
    pub fn last_coherence(&self) -> f32 { self.last_coherence }

    #[getter]
    pub fn last_mean_w(&self) -> f32 { self.last_mean_w }

    // ═══════════════════════════════════════════════════
    //  Run refinement
    // ═══════════════════════════════════════════════════

    /// Run GPU-accelerated refinement.
    ///
    /// Args:
    ///     n_epochs: Number of refinement epochs (default 500)
    ///     n_steps: Kuramoto integration steps per epoch (default 80)
    ///     dt: Time step (default 0.05)
    ///     lr: Learning rate for weight update (default 0.08)
    ///     coherence_weight: Weight of coherence loss (default 1.0)
    ///     sparsity_weight: Weight of L1 sparsity penalty (default 0.002)
    ///     temporal_weight: Weight of temporal smoothness (default 0.01)
    ///     homeostasis_weight: Weight of homeostatic regulation (default 0.01)
    ///     target_mean_weight: Target mean absolute weight (default 0.5)
    ///     weight_clamp: Max absolute weight value (default 5.0, 0 = no clamp)
    ///     report_every: Print progress every N epochs (default 100, 0 = silent)
    ///
    /// Returns: dict with keys: epoch, coherence, mean_abs_w
    #[pyo3(signature = (
        n_epochs=500,
        n_steps=80,
        dt=0.05,
        lr=0.08,
        coherence_weight=1.0,
        sparsity_weight=0.002,
        temporal_weight=0.01,
        homeostasis_weight=0.01,
        target_mean_weight=0.5,
        weight_clamp=5.0,
        report_every=100,
    ))]
    pub fn refine(
        &mut self,
        n_epochs: usize,
        n_steps: usize,
        dt: f32,
        lr: f32,
        coherence_weight: f32,
        sparsity_weight: f32,
        temporal_weight: f32,
        homeostasis_weight: f32,
        target_mean_weight: f32,
        weight_clamp: f32,
        report_every: usize,
    ) -> PyResult<HashMap<String, Vec<f64>>> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (n_epochs, n_steps, dt, lr, coherence_weight, sparsity_weight,
                     temporal_weight, homeostasis_weight, target_mean_weight, weight_clamp, report_every);
            return Err(PyRuntimeError::new_err(
                "CUDA not available — rebuild with --features cuda,python"));
        }

        #[cfg(feature = "cuda")]
        {
            let n = self.n;
            let s = self.s;
            let coherence_start = n_steps / 2;
            let grad_clip_norm = 1.0_f32;  // gradient clipping threshold (0 = disable)

            // Initialize GPU context + upload connectome
            if self.ctx.is_none() {
                let ctx = GpuContext::new(n, s)
                    .map_err(|e| PyRuntimeError::new_err(format!("GpuContext: {e}")))?;
                let ext_I = vec![0.0f32; n];
                ctx.upload_connectome(&self.row_ptr, &self.col_indices, &self.values,
                                      &self.nt_types, &ext_I)
                    .map_err(|e| PyRuntimeError::new_err(format!("upload_connectome: {e}")))?;
                self.ctx = Some(ctx);
            }

            let ctx = self.ctx.as_ref().unwrap();

            // Allocate trajectory buffers on GPU
            let mut d_phases_traj = DevicePtr::<f32>::new(n_steps * n)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc phases_traj: {e}")))?;
            let mut d_coupling_traj = DevicePtr::<f32>::new(n_steps * n)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc coupling_traj: {e}")))?;
            let mut d_dtheta_traj = DevicePtr::<f32>::new(n_steps * n)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc dtheta_traj: {e}")))?;
            let mut d_grad_weights = DevicePtr::<f32>::new(s)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc grad_weights: {e}")))?;

            // Gradient norm buffers (fully on GPU — no D2H)
            let n_partial_blocks = s.div_ceil(256);
            let mut d_grad_norm = DevicePtr::<f32>::new(1)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc grad_norm: {e}")))?;
            let mut d_grad_norm_partial = DevicePtr::<f32>::new(n_partial_blocks)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc grad_norm_partial: {e}")))?;

            let mut h_phases_full = vec![0.0f32; n_steps * n];
            let mut h_weights = vec![0.0f32; s];
            let mut mean_abs_w = self.values.iter().map(|w| w.abs()).sum::<f32>() / s as f32;

            // History
            let mut epochs_out = Vec::new();
            let mut coherence_out = Vec::new();
            let mut mean_w_out = Vec::new();

            for epoch in 0..n_epochs {
                // Phase 1: Upload init phases + GPU forward with D2D trajectory
                ctx.upload_kuramoto(&self.init_phases, &self.omegas, &vec![0.0f32; n])
                    .map_err(|e| PyRuntimeError::new_err(format!("upload: {e}")))?;

                // Save initial phases
                ctx.copy_phases_to_traj_async(d_phases_traj.as_mut_ptr(), 0)
                    .map_err(|e| PyRuntimeError::new_err(format!("copy_init: {e}")))?;

                for step in 0..n_steps - 1 {
                    ctx.kuramoto_prepare()
                        .map_err(|e| PyRuntimeError::new_err(format!("prepare: {e}")))?;
                    ctx.kuramoto_csr_coupling()
                        .map_err(|e| PyRuntimeError::new_err(format!("coupling: {e}")))?;
                    ctx.copy_coupling_to_traj_async(d_coupling_traj.as_mut_ptr(), step)
                        .map_err(|e| PyRuntimeError::new_err(format!("copy_cpl: {e}")))?;
                    ctx.kuramoto_update(dt)
                        .map_err(|e| PyRuntimeError::new_err(format!("update: {e}")))?;
                    ctx.copy_phases_to_traj_async(d_phases_traj.as_mut_ptr(), step + 1)
                        .map_err(|e| PyRuntimeError::new_err(format!("copy_ph: {e}")))?;
                }

                // Last coupling
                ctx.kuramoto_prepare()
                    .map_err(|e| PyRuntimeError::new_err(format!("prep_last: {e}")))?;
                ctx.kuramoto_csr_coupling()
                    .map_err(|e| PyRuntimeError::new_err(format!("cpl_last: {e}")))?;
                ctx.copy_coupling_to_traj_async(d_coupling_traj.as_mut_ptr(), n_steps - 1)
                    .map_err(|e| PyRuntimeError::new_err(format!("copy_cpl_last: {e}")))?;

                // Phase 2: GPU loss gradient
                ctx.bptt_loss_grad(
                    d_phases_traj.as_mut_ptr(),
                    d_coupling_traj.as_mut_ptr(),
                    d_dtheta_traj.as_mut_ptr(),
                    coherence_start, coherence_weight, temporal_weight, dt, n_steps,
                ).map_err(|e| PyRuntimeError::new_err(format!("loss_grad: {e}")))?;

                // Phase 3: GPU BPTT backward
                ctx.zero_grad_weights_async(d_grad_weights.as_mut_ptr())
                    .map_err(|e| PyRuntimeError::new_err(format!("zero_grad: {e}")))?;

                for bw_step in (0..n_steps - 1).rev() {
                    let dtheta_step = unsafe { d_dtheta_traj.as_mut_ptr().add(bw_step * n) } as *mut f32;
                    let dtheta_next = unsafe { d_dtheta_traj.as_ptr().add((bw_step + 1) * n) } as *const f32;
                    let phases_step = unsafe { d_phases_traj.as_ptr().add(bw_step * n) } as *const f32;

                    ctx.bptt_step(
                        d_grad_weights.as_mut_ptr() as *mut f32,
                        dtheta_step,
                        dtheta_next,
                        phases_step,
                        dt,
                    ).map_err(|e| PyRuntimeError::new_err(format!("bptt_step: {e}")))?;
                }

                // Phase 4: GPU Gradient Clipping
                if grad_clip_norm > 0.0 {
                    ctx.grad_norm_async(
                        d_grad_weights.as_ptr() as *const f32,
                        d_grad_norm.as_mut_ptr(),
                        d_grad_norm_partial.as_mut_ptr(),
                    ).map_err(|e| PyRuntimeError::new_err(format!("grad_norm: {e}")))?;
                    ctx.grad_clip_async(
                        d_grad_weights.as_mut_ptr() as *mut f32,
                        d_grad_norm.as_ptr() as *const f32,
                        grad_clip_norm,
                    ).map_err(|e| PyRuntimeError::new_err(format!("grad_clip: {e}")))?;
                }

                // Phase 5: GPU weight update
                ctx.download_csr_values(&mut h_weights)
                    .map_err(|e| PyRuntimeError::new_err(format!("dl_weights: {e}")))?;
                mean_abs_w = h_weights.iter().map(|w| w.abs()).sum::<f32>() / s as f32;

                let sparsity_scale = sparsity_weight / s as f32;
                let homeo_diff = mean_abs_w - target_mean_weight;
                let homeostasis_scale = 2.0 * homeostasis_weight * homeo_diff / s as f32;

                let sparsity_scale = if sparsity_scale.is_finite() { sparsity_scale } else { 0.0 };
                let homeostasis_scale = if homeostasis_scale.is_finite() { homeostasis_scale } else { 0.0 };

                ctx.weight_update_async(
                    d_grad_weights.as_ptr() as *const f32,
                    lr, sparsity_scale, homeostasis_scale, weight_clamp,
                ).map_err(|e| PyRuntimeError::new_err(format!("weight_update: {e}")))?;

                ctx.sync()
                    .map_err(|e| PyRuntimeError::new_err(format!("sync: {e}")))?;

                // Stats
                if report_every > 0 && (epoch % report_every == 0 || epoch == n_epochs - 1) {
                    d_phases_traj.copy_to_host(&mut h_phases_full)
                        .map_err(|e| PyRuntimeError::new_err(format!("dl_phases: {e}")))?;
                    let final_off = (n_steps - 1) * n;
                    let (sum_s, sum_c): (f64, f64) = h_phases_full[final_off..final_off + n]
                        .iter().fold((0.0, 0.0), |(s, c), &th| (s + th.sin() as f64, c + th.cos() as f64));
                    let coherence = ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32;

                    epochs_out.push(epoch as f64);
                    coherence_out.push(coherence as f64);
                    mean_w_out.push(mean_abs_w as f64);

                    let msg = format!("  epoch {:>4}/{}: R={:.4} mean|w|={:.4}", epoch, n_epochs, coherence, mean_abs_w);
                    println!("{}", msg);
                }
            }

            // Save refined weights back to host
            ctx.download_csr_values(&mut self.values)
                .map_err(|e| PyRuntimeError::new_err(format!("dl_final: {e}")))?;

            self.last_coherence = *coherence_out.last().unwrap_or(&0.0) as f32;
            self.last_mean_w = mean_abs_w;

            let mut result = HashMap::new();
            result.insert("epoch".to_string(), epochs_out);
            result.insert("coherence".to_string(), coherence_out);
            result.insert("mean_abs_w".to_string(), mean_w_out);
            Ok(result)
        }
    }

    /// Save refined connectome to .taas file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        let conn = taas_connectome::ConnectomeData {
            n_neurons: self.n,
            n_synapses: self.s,
            row_ptr: self.row_ptr.clone(),
            col_indices: self.col_indices.clone(),
            values: self.values.clone(),
            nt_types: self.nt_types.clone(),
            receptor_types: Vec::new(),
            groups: Vec::new(),
        };

        let mut f = std::fs::File::create(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot create {path}: {e}")))?;
        conn.save_baked(&mut f)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot save .taas: {e}")))?;
        Ok(())
    }
}


