//! # CudaBackend — GPU T3 backend via pillars-cuda.
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

use crate::cuda::GpuContext;
use crate::T3Backend;
use std::sync::atomic::{AtomicBool, Ordering};

/// CUDA GPU backend for T3 computations with async transfers.
pub struct CudaBackend {
    ctx: GpuContext,
    n: usize,
    state_on_gpu: AtomicBool,
}

impl CudaBackend {
    pub fn new(n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("CudaBackend: n must be > 0".into());
        }
        let ctx = GpuContext::new(n, n)?;
        Ok(Self {
            ctx,
            n,
            state_on_gpu: AtomicBool::new(false),
        })
    }

    pub fn is_available() -> bool {
        crate::cuda::get_device_count().unwrap_or(0) > 0
    }

    pub fn sync(&self) -> Result<(), String> {
        self.ctx.sync()
    }

    pub fn size(&self) -> usize {
        self.n
    }
}

impl T3Backend for CudaBackend {
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
        let n = self
            .n
            .min(phases.len())
            .min(cos_out.len())
            .min(sin_out.len());

        if self.state_on_gpu.load(Ordering::Relaxed) {
            if let Err(e) = self
                .ctx
                .t3_prepare()
                .and_then(|()| {
                    self.ctx
                        .t3_download_cos_sin_async(&mut cos_out[..n], &mut sin_out[..n])
                })
                .and_then(|()| self.ctx.sync())
            {
                eprintln!("CudaBackend::prepare failed: {e}");
            }
        } else {
            let _ = crate::cuda::t3_prepare(&mut phases[..n], &mut cos_out[..n], &mut sin_out[..n]);
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
        // Always upload — each call may be a different population (pop1/pop2/pop3)
        // with its own phases/amplitudes/omegas. GPU buffers are shared.
        if let Err(e) = self
            .ctx
            .t3_upload_async(&phases[..n], &amplitudes[..n], &omegas[..n])
            .and_then(|()| self.ctx.t3_upload_drives_async(&drives[..n]))
            .and_then(|()| self.ctx.t3_upload_coupling_async(&coupling[..n]))
            .and_then(|()| self.ctx.t3_update(dt, alpha, gain))
            .and_then(|()| {
                self.ctx
                    .t3_download_async(&mut phases[..n], &mut amplitudes[..n])
            })
            .and_then(|()| self.ctx.sync())
        {
            eprintln!("CudaBackend::update failed: {e}");
        }
    }

    fn nlms_train(
        &self,
        weights: &mut [f32],
        bias: &mut [f32],
        states: &[f32],
        error: f32,
        lr: f32,
        eps: f32,
    ) {
        // Canonical size contract (mirrors pillars::kernels::decode::nlms_train):
        // a short `states` would silently truncate the update via `zip`, and
        // an empty `bias` would panic on `bias[0]` — both caught here up front.
        assert_eq!(
            states.len(),
            weights.len(),
            "CudaBackend::nlms_train: states length mismatch"
        );
        assert!(
            !bias.is_empty(),
            "CudaBackend::nlms_train: bias must have at least 1 element"
        );
        if weights.len() < 1024 {
            let norm_sq: f32 = states.iter().map(|&x| x * x).sum();
            let mu = lr / (norm_sq + eps);
            for (w, &s) in weights.iter_mut().zip(states.iter()) {
                *w += mu * error * s;
            }
            bias[0] += mu * error;
        } else {
            // NOTE: the GPU kernel normalizes per-lane (`si² + ε`) instead of
            // the canonical global norm² — documented divergence in
            // cuda/kernels/decoders.cu; revisit when the GPU path is revived.
            let _ = crate::cuda::nlms_train(weights, bias, states, error, lr, eps);
        }
    }

    fn kalman_update(
        &self,
        weights: &mut [f32],
        cov: &mut [f32],
        states: &[f32],
        error: f32,
        s: f32,
    ) {
        // Canonical size contract (mirrors pillars::kernels::decode::kalman_update).
        assert_eq!(
            cov.len(),
            weights.len(),
            "CudaBackend::kalman_update: cov length mismatch"
        );
        assert_eq!(
            states.len(),
            weights.len(),
            "CudaBackend::kalman_update: states length mismatch"
        );
        if s <= 1e-12 {
            return;
        }
        if weights.len() < 1024 {
            // Scalar fallback — MUST use the canonical pillars formula
            // `cov -= K·states·cov` (floored at 1e-12). The legacy
            // `cov -= K·cov` here was a real bug: with a negative state
            // component it *grew* the covariance instead of shrinking it.
            let s_denom = s + 1e-8;
            for i in 0..weights.len() {
                let k = cov[i] * states[i] / s_denom;
                weights[i] += k * error;
                cov[i] = (cov[i] - k * states[i] * cov[i]).max(1e-12);
            }
        } else {
            let _ = crate::cuda::kalman_update(weights, cov, states, error, s);
        }
    }
}
