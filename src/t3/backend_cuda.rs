//! # CudaBackend — GPU T3 backend via cuda-kernels.
//!
//! ## Async pipeline (single sync per step)
//!
//! ```text
//! prepare(): [upload phases H2D]* → async kernel → [D2H cos/sin]* → sync
//!                                               ↓
//! update():  H2D coupling + drives → async kernel → D2H phases+amps → sync
//! ```
//!
//! * = first step only
//!
//! Before: 3–5 syncs per step (one inside each upload/download call).
//! After:  1 sync per step (explicit at the end).

use std::sync::atomic::{AtomicBool, Ordering};
use crate::t3::T3Backend;
use cuda_kernels::GpuContext;

/// CUDA GPU backend for T3 computations with async transfers.
pub struct CudaBackend {
    ctx: GpuContext,
    n: usize,
    state_on_gpu: AtomicBool,
}

impl CudaBackend {
    pub fn new(n: usize) -> Result<Self, String> {
        if n == 0 { return Err("CudaBackend: n must be > 0".into()); }
        let ctx = GpuContext::new(n, n)?;
        Ok(Self { ctx, n, state_on_gpu: AtomicBool::new(false) })
    }

    pub fn is_available() -> bool {
        cuda_kernels::get_device_count().unwrap_or(0) > 0
    }

    pub fn sync(&self) -> Result<(), String> { self.ctx.sync() }

    pub fn size(&self) -> usize { self.n }
}

impl T3Backend for CudaBackend {
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
        let n = self.n.min(phases.len()).min(cos_out.len()).min(sin_out.len());

        if self.state_on_gpu.load(Ordering::Relaxed) {
            // Async: kernel → D2H cos/sin → single sync
            if let Err(e) = self.ctx.t3_prepare() {
                panic!("CudaBackend::prepare kernel failed: {e}");
            }
            if let Err(e) = self.ctx.t3_download_cos_sin_async(
                &mut cos_out[..n], &mut sin_out[..n],
            ) {
                panic!("CudaBackend::prepare D2H failed: {e}");
            }
            if let Err(e) = self.ctx.sync() {
                panic!("CudaBackend::prepare sync failed: {e}");
            }
        } else {
            // First call: sync host wrapper (H2D + kernel + D2H + sync inside)
            let _ = cuda_kernels::t3_prepare(
                &mut phases[..n], &mut cos_out[..n], &mut sin_out[..n],
            );
        }
    }

    fn update(
        &self,
        phases: &mut [f32],
        amplitudes: &mut [f32],
        omegas: &[f32],
        coupling: &[f32],
        drives: &[f32],
        dt: f32,
        alpha: f32,
        gain: f32,
    ) {
        let n = self.n.min(phases.len()).min(amplitudes.len());
        let initialized = self.state_on_gpu.load(Ordering::Relaxed);

        if !initialized {
            // First call: async upload (H2D) → kernel → D2H → sync
            if let Err(e) = self.ctx.t3_upload_async(
                &phases[..n], &amplitudes[..n], &omegas[..n],
            ) {
                panic!("CudaBackend::update upload failed: {e}");
            }
            self.state_on_gpu.store(true, Ordering::Relaxed);
        }

        // Every step: async upload coupling + drives
        if let Err(e) = self.ctx.t3_upload_drives_async(&drives[..n]) {
            panic!("CudaBackend::update upload drives failed: {e}");
        }
        if let Err(e) = self.ctx.t3_upload_coupling_async(&coupling[..n]) {
            panic!("CudaBackend::update upload coupling failed: {e}");
        }

        // Async kernel (Stuart-Landau + Kuramoto)
        if let Err(e) = self.ctx.t3_update(dt, alpha, gain) {
            panic!("CudaBackend::update kernel failed: {e}");
        }

        // Async download + ONE sync
        if let Err(e) = self.ctx.t3_download_async(&mut phases[..n], &mut amplitudes[..n]) {
            panic!("CudaBackend::update download failed: {e}");
        }
        if let Err(e) = self.ctx.sync() {
            panic!("CudaBackend::update sync failed: {e}");
        }
    }

    fn nlms_train(&self, weights: &mut [f32], bias: &mut [f32],
        states: &[f32], error: f32, lr: f32, eps: f32)
    {
        if weights.len() < 1024 {
            let norm_sq: f32 = states.iter().map(|&x| x * x).sum();
            let mu = lr / (norm_sq + eps);
            for (w, &s) in weights.iter_mut().zip(states.iter()) { *w += mu * error * s; }
            bias[0] += mu * error;
        } else {
            let _ = cuda_kernels::nlms_train(weights, bias, states, error, lr, eps);
        }
    }

    fn kalman_update(&self, weights: &mut [f32], cov: &mut [f32],
        states: &[f32], error: f32, s: f32)
    {
        if s <= 1e-12 { return; }
        if weights.len() < 1024 {
            for i in 0..weights.len() {
                let k = cov[i] * states[i] / s;
                weights[i] += k * error;
                cov[i] -= k * cov[i];
            }
        } else {
            let _ = cuda_kernels::kalman_update(weights, cov, states, error, s);
        }
    }
}
