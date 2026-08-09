use crate::gpu::device::ok_or;
use crate::gpu::ffi::*;
use crate::gpu::ptr::*;
use std::ffi::{c_float, c_int, c_void};
// ═══════════════════════════════════════════════════════════

pub(crate) struct GpuCtx;

extern "C" {
    fn gpu_context_create(n_neurons: c_int, n_synapses: c_int, n_decoders: c_int) -> *mut GpuCtx;
    fn gpu_context_destroy(ctx: *mut GpuCtx);
    fn gpu_context_sync(ctx: *mut GpuCtx) -> c_int;

    // Upload initial state
    fn gpu_context_upload_neurons(
        ctx: *mut GpuCtx,
        v: *const c_float,
        u: *const c_float,
        a: *const c_float,
        b: *const c_float,
        c: *const c_float,
        d: *const c_float,
        n: c_int,
    ) -> c_int;
    fn gpu_context_upload_kuramoto(
        ctx: *mut GpuCtx,
        phases: *const c_float,
        omegas: *const c_float,
        coupling: *const c_float,
        n: c_int,
    ) -> c_int;

    // Async kernels (no sync inside)
    fn gpu_izhikevich_step_async(ctx: *mut GpuCtx, dt: c_float, trace_decay: c_float) -> c_int;
    fn gpu_lif_step_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_context_upload_lif_params(
        ctx: *mut GpuCtx,
        dt: c_float,
        tau_m: *const c_float,
        v_rest: *const c_float,
        v_thresh: *const c_float,
        v_reset: *const c_float,
        n: c_int,
    ) -> c_int;
    fn gpu_stdp_update_async(
        ctx: *mut GpuCtx,
        lr_ltd: c_float,
        lr_ltp: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;
    fn gpu_kuramoto_prepare_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_kuramoto_update_async(ctx: *mut GpuCtx, dt: c_float) -> c_int;
    fn gpu_kuramoto_csr_coupling_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_bptt_loss_grad_async(
        ctx: *mut GpuCtx,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta: *mut c_float,
        coherence_start: c_int,
        coherence_weight: c_float,
        temporal_scale: c_float,
        dt: c_float,
        n_steps: c_int,
    ) -> c_int;
    fn gpu_bptt_step_async(
        ctx: *mut GpuCtx,
        d_grad_weights: *mut c_float,
        d_dtheta: *mut c_float,
        d_dtheta_next: *const c_float,
        d_phases: *const c_float,
        dt: c_float,
        n_synapses: c_int,
    ) -> c_int;

    // Trajectory D2D copy (async — enqueues on stream)
    fn gpu_context_copy_phases_to_traj_async(
        ctx: *mut GpuCtx,
        d_traj: *mut c_float,
        step: c_int,
    ) -> c_int;
    fn gpu_context_copy_coupling_to_traj_async(
        ctx: *mut GpuCtx,
        d_traj: *mut c_float,
        step: c_int,
    ) -> c_int;

    // GPU weight optimizer (async)
    fn gpu_zero_grad_weights_async(ctx: *mut GpuCtx, d_grad: *mut c_float, n: c_int) -> c_int;
    fn gpu_weight_update_async(
        ctx: *mut GpuCtx,
        d_grad_weights: *const c_float,
        lr: c_float,
        sparsity_scale: c_float,
        homeostasis_scale: c_float,
        weight_clamp: c_float,
    ) -> c_int;
    // Gradient norm + clipping (async, fully on GPU)
    fn gpu_grad_norm_async(
        ctx: *mut GpuCtx,
        d_grad: *const c_float,
        d_norm: *mut c_float,
        d_partial: *mut c_float,
        n: c_int,
    ) -> c_int;
    fn gpu_grad_clip_async(
        ctx: *mut GpuCtx,
        d_grad: *mut c_float,
        d_norm: *const c_float,
        max_norm: c_float,
        n: c_int,
    ) -> c_int;
    // Fused BPTT epoch (all kernels — forward + loss + BPTT + clip)
    fn gpu_bptt_epoch_async(
        ctx: *mut GpuCtx,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta_traj: *mut c_float,
        d_grad_weights: *mut c_float,
        d_grad_norm: *mut c_float,
        d_grad_norm_partial: *mut c_float,
        n_steps: c_int,
        dt: c_float,
        coherence_start: c_int,
        coherence_weight: c_float,
        temporal_weight: c_float,
        grad_clip_norm: c_float,
    ) -> c_int;

    fn gpu_safety_clamp_async(ctx: *mut GpuCtx, min_val: c_float, max_val: c_float) -> c_int;

    // Fused kernels
    fn gpu_fused_izhikevich_stdp_async(
        ctx: *mut GpuCtx,
        dt: c_float,
        trace_decay: c_float,
        lr_ltd: c_float,
        lr_ltp: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;

    // Connectome CSR upload + fused CSR step + Izhikevich step
    fn gpu_context_upload_connectome(
        ctx: *mut GpuCtx,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        values: *const c_float,
        nt_types: *const c_float,
        external_I: *const c_float,
        n_neurons: c_int,
        n_synapses: c_int,
    ) -> c_int;
    fn gpu_connectome_csr_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_connectome_csr_plastic_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_connectome_stdp_async(
        ctx: *mut GpuCtx,
        lr_ltp: c_float,
        lr_ltd: c_float,
        w_min: c_float,
        w_max: c_float,
    ) -> c_int;
    fn gpu_context_init_plastic_weights(ctx: *mut GpuCtx) -> c_int;
    fn gpu_count_spikes_async(ctx: *mut GpuCtx, d_total_spikes: *mut c_int) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemset(ptr: *mut c_void, val: c_int, count: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;

    // Upload/download
    fn gpu_context_upload_I(ctx: *mut GpuCtx, I: *const c_float, n: c_int) -> c_int;
    fn gpu_context_download_spikes(ctx: *mut GpuCtx, spikes: *mut c_int, n: c_int) -> c_int;
    fn gpu_context_batch_lif_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        dev_spike_buf: *mut c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
    ) -> c_int;
    fn gpu_context_batch_izhikevich_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        dt: c_float,
        trace_decay: c_float,
        dev_spike_buf: *mut c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        host_u: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
    ) -> c_int;
    fn gpu_context_batch_hh_steps(
        ctx: *mut GpuCtx,
        external_I: *const c_float,
        n_external: c_int,
        host_spike_buf: *mut c_int,
        host_v: *mut c_float,
        host_I_syn: *mut c_float,
        n_steps: c_int,
        n_neurons: c_int,
        dt: c_float,
        tau_ref: c_float,
        tau_syn: c_float,
        weight_scale: c_float,
    ) -> c_int;
    fn gpu_context_upload_v_only(ctx: *mut GpuCtx, v: *const c_float, n: c_int) -> c_int;
    fn gpu_context_download_v(ctx: *mut GpuCtx, v: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_download_u(ctx: *mut GpuCtx, u: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_download_phases(ctx: *mut GpuCtx, phases: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_download_adex_w(ctx: *mut GpuCtx, w: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_upload_adex_w(ctx: *mut GpuCtx, w: *const c_float, n: c_int) -> c_int;
    fn gpu_context_download_weights(ctx: *mut GpuCtx, w: *mut c_float, n: c_int) -> c_int;

    // CUDA Graph
    fn gpu_context_ensure_all(ctx: *mut GpuCtx) -> c_int;
    fn gpu_context_get_d_noise(ctx: *mut GpuCtx) -> *mut c_void;
    fn gpu_graph_begin_capture(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_end_capture(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_replay(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_is_captured(ctx: *mut GpuCtx) -> c_int;
    // GCO CUDA Graph
}

/// Persistent GPU state — keeps everything on device between steps.
pub struct GpuContext {
    ctx: *mut GpuCtx,
    n_neurons: usize,
    n_synapses: usize,
}

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }
    pub fn n_synapses(&self) -> usize {
        self.n_synapses
    }

    pub fn new(n_neurons: usize, n_synapses: usize) -> Result<Self, String> {
        let ctx = unsafe {
            gpu_context_create(n_neurons as c_int, n_synapses as c_int, n_neurons as c_int)
        };
        if ctx.is_null() {
            Err("GpuContext::new failed".into())
        } else {
            Ok(GpuContext {
                ctx,
                n_neurons,
                n_synapses,
            })
        }
    }

    /// Upload neuron parameters (v, u, a, b, c, d).
    pub fn upload_neurons(
        &self,
        v: &[f32],
        u: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
    ) -> Result<(), String> {
        let n = v.len().min(u.len()).min(a.len());
        let err = unsafe {
            gpu_context_upload_neurons(
                self.ctx,
                v.as_ptr(),
                u.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                d.as_ptr(),
                n as c_int,
            )
        };
        ok_or(err, "upload_neurons")
    }

    /// Upload input current I to GPU.
    pub fn upload_I(&self, I: &[f32]) -> Result<(), String> {
        let n = I.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_upload_I(self.ctx, I.as_ptr(), n as c_int) },
            "upload_I",
        )
    }

    /// Upload Kuramoto state.
    pub fn upload_kuramoto(
        &self,
        phases: &[f32],
        omegas: &[f32],
        coupling: &[f32],
    ) -> Result<(), String> {
        let n = phases.len().min(omegas.len()).min(coupling.len());
        let err = unsafe {
            gpu_context_upload_kuramoto(
                self.ctx,
                phases.as_ptr(),
                omegas.as_ptr(),
                coupling.as_ptr(),
                n as c_int,
            )
        };
        ok_or(err, "upload_kuramoto")
    }

    /// Run single Izhikevich step (async, no sync).
    pub fn izhikevich_step(&self, dt: f32, trace_decay: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_izhikevich_step_async(self.ctx, dt, trace_decay) },
            "izhikevich",
        )
    }

    /// Run STDP update (async).
    pub fn stdp_update(
        &self,
        lr_ltd: f32,
        lr_ltp: f32,
        w_min: f32,
        w_max: f32,
    ) -> Result<(), String> {
        ok_or(
            unsafe { gpu_stdp_update_async(self.ctx, lr_ltd, lr_ltp, w_min, w_max) },
            "stdp",
        )
    }

    /// Run fused Izhikevich + STDP (async).
    pub fn fused_step(
        &self,
        dt: f32,
        trace_decay: f32,
        lr_ltd: f32,
        lr_ltp: f32,
        w_min: f32,
        w_max: f32,
    ) -> Result<(), String> {
        ok_or(
            unsafe {
                gpu_fused_izhikevich_stdp_async(
                    self.ctx,
                    dt,
                    trace_decay,
                    lr_ltd,
                    lr_ltp,
                    w_min,
                    w_max,
                )
            },
            "fused_step",
        )
    }

    /// Run Kuramoto prepare (phases → cos/sin, async).
    pub fn kuramoto_prepare(&self) -> Result<(), String> {
        ok_or(
            unsafe { gpu_kuramoto_prepare_async(self.ctx) },
            "kuramoto_prepare",
        )
    }

    /// Run Kuramoto update (async).
    pub fn kuramoto_update(&self, dt: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_kuramoto_update_async(self.ctx, dt) },
            "kuramoto_update",
        )
    }

    /// Compute CSR Kuramoto coupling on GPU: coupling[i] = Σ_j W_ij · sin(θⱼ - θᵢ).
    /// Uses d_cos_buf/d_sin_buf (from kuramoto_prepare), d_row_ptr/d_col_indices/d_csr_values (from upload_connectome).
    /// Writes result to d_coupling.
    /// Requires upload_connectome() and kuramoto_prepare() to have been called first.
    pub fn kuramoto_csr_coupling(&self) -> Result<(), String> {
        ok_or(
            unsafe { gpu_kuramoto_csr_coupling_async(self.ctx) },
            "kuramoto_csr_coupling",
        )
    }

    /// BPTT: compute loss gradient (coherence + temporal) entirely on GPU.
    /// Requires trajectory buffers to be pre-allocated on device.
    pub fn bptt_loss_grad(
        &self,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta: *mut c_float,
        coherence_start: usize,
        coherence_weight: f32,
        temporal_weight: f32,
        dt: f32,
        n_steps: usize,
    ) -> Result<(), String> {
        let temporal_scale = temporal_weight / ((n_steps - 1).max(1) * self.n_neurons) as f32;
        ok_or(
            unsafe {
                gpu_bptt_loss_grad_async(
                    self.ctx,
                    d_phases_traj,
                    d_coupling_traj,
                    d_dtheta,
                    coherence_start as c_int,
                    coherence_weight,
                    temporal_scale,
                    dt,
                    n_steps as c_int,
                )
            },
            "bptt_loss_grad",
        )
    }

    /// BPTT: one backward step (weight gradients + diagonal + cross-term backprop) on GPU.
    pub fn bptt_step(
        &self,
        d_grad_weights: *mut c_float,
        d_dtheta: *mut c_float,
        d_dtheta_next: *const c_float,
        d_phases: *const c_float,
        dt: f32,
    ) -> Result<(), String> {
        let n_syn = self.n_synapses as c_int;
        ok_or(
            unsafe {
                gpu_bptt_step_async(
                    self.ctx,
                    d_grad_weights,
                    d_dtheta,
                    d_dtheta_next,
                    d_phases,
                    dt,
                    n_syn,
                )
            },
            "bptt_step",
        )
    }

    /// Upload CSR values (synaptic weights) to GPU (without re-uploading entire connectome).
    /// Copies host `values` into `d_csr_values` on device.
    pub fn upload_csr_values(&self, values: &[f32]) -> Result<(), String> {
        let n = values.len().min(self.n_synapses) as c_int;
        ok_or(
            unsafe { cuda_upload_csr_values(self.ctx, values.as_ptr(), n) },
            "upload_csr_values",
        )
    }

    /// Download CSR values (synaptic weights) from GPU to host.
    pub fn download_csr_values(&self, values: &mut [f32]) -> Result<(), String> {
        let n = values.len().min(self.n_synapses) as c_int;
        ok_or(
            unsafe { cuda_download_csr_values(self.ctx, values.as_mut_ptr(), n) },
            "download_csr_values",
        )
    }

    // ─── T3 async operations ──────────────────────────

    /// Upload T3 oscillator state to GPU.
    pub fn t3_upload(
        &self,
        phases: &[f32],
        amplitudes: &[f32],
        omegas: &[f32],
    ) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()).min(omegas.len()) as c_int;
        ok_or(
            unsafe {
                gpu_t3_upload(
                    self.ctx,
                    phases.as_ptr(),
                    amplitudes.as_ptr(),
                    omegas.as_ptr(),
                    n,
                )
            },
            "t3_upload",
        )
    }

    /// Upload T3 drives to GPU.
    pub fn t3_upload_drives(&self, drives: &[f32]) -> Result<(), String> {
        let n = drives.len() as c_int;
        ok_or(
            unsafe { gpu_t3_upload_drives(self.ctx, drives.as_ptr(), n) },
            "t3_upload_drives",
        )
    }

    /// Upload T3 coupling to GPU.
    pub fn t3_upload_coupling(&self, coupling: &[f32]) -> Result<(), String> {
        let n = coupling.len() as c_int;
        ok_or(
            unsafe { gpu_t3_upload_coupling(self.ctx, coupling.as_ptr(), n) },
            "t3_upload_coupling",
        )
    }

    /// Download T3 oscillator state from GPU.
    pub fn t3_download(&self, phases: &mut [f32], amplitudes: &mut [f32]) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()) as c_int;
        ok_or(
            unsafe { gpu_t3_download(self.ctx, phases.as_mut_ptr(), amplitudes.as_mut_ptr(), n) },
            "t3_download",
        )
    }

    /// Download T3 cos/sin from GPU (for CPU coupling computation).
    pub fn t3_download_cos_sin(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
    ) -> Result<(), String> {
        let n = cos_out.len().min(sin_out.len()) as c_int;
        ok_or(
            unsafe {
                gpu_t3_download_cos_sin(self.ctx, cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n)
            },
            "t3_download_cos_sin",
        )
    }

    /// Upload T3 omegas to GPU (done once at init).
    pub fn t3_upload_omegas(&self, omegas: &[f32]) -> Result<(), String> {
        let n = omegas.len() as c_int;
        ok_or(
            unsafe { gpu_t3_upload_omegas(self.ctx, omegas.as_ptr(), n) },
            "t3_upload_omegas",
        )
    }

    // ─── T3 async transfers (no sync inside) ───────────

    /// Async upload (no sync). Call sync() after all transfers + kernel.
    pub fn t3_upload_async(
        &self,
        phases: &[f32],
        amplitudes: &[f32],
        omegas: &[f32],
    ) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()).min(omegas.len()) as c_int;
        ok_or(
            unsafe {
                gpu_t3_upload_async(
                    self.ctx,
                    phases.as_ptr(),
                    amplitudes.as_ptr(),
                    omegas.as_ptr(),
                    n,
                )
            },
            "t3_upload_async",
        )
    }

    /// Async upload drives (no sync).
    pub fn t3_upload_drives_async(&self, drives: &[f32]) -> Result<(), String> {
        let n = drives.len() as c_int;
        ok_or(
            unsafe { gpu_t3_upload_drives_async(self.ctx, drives.as_ptr(), n) },
            "t3_upload_drives_async",
        )
    }

    /// Async upload coupling (no sync).
    pub fn t3_upload_coupling_async(&self, coupling: &[f32]) -> Result<(), String> {
        let n = coupling.len() as c_int;
        ok_or(
            unsafe { gpu_t3_upload_coupling_async(self.ctx, coupling.as_ptr(), n) },
            "t3_upload_coupling_async",
        )
    }

    /// Async download (no sync).
    pub fn t3_download_async(
        &self,
        phases: &mut [f32],
        amplitudes: &mut [f32],
    ) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()) as c_int;
        ok_or(
            unsafe {
                gpu_t3_download_async(self.ctx, phases.as_mut_ptr(), amplitudes.as_mut_ptr(), n)
            },
            "t3_download_async",
        )
    }

    /// Async download cos/sin (no sync).
    pub fn t3_download_cos_sin_async(
        &self,
        cos_out: &mut [f32],
        sin_out: &mut [f32],
    ) -> Result<(), String> {
        let n = cos_out.len().min(sin_out.len()) as c_int;
        ok_or(
            unsafe {
                gpu_t3_download_cos_sin_async(
                    self.ctx,
                    cos_out.as_mut_ptr(),
                    sin_out.as_mut_ptr(),
                    n,
                )
            },
            "t3_download_cos_sin_async",
        )
    }

    /// Run T3 prepare (phases → cos/sin, async).
    pub fn t3_prepare(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_t3_prepare_async(self.ctx) }, "t3_prepare")
    }

    /// Run T3 update (Stuart-Landau + Kuramoto, async).
    pub fn t3_update(&self, dt: f32, alpha: f32, gain: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_t3_update_async(self.ctx, dt, alpha, gain) },
            "t3_update",
        )
    }

    /// Run T3 Stuart-Landau only (amplitude update, async).
    pub fn t3_stuart_landau(&self, dt: f32, alpha: f32, gain: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_t3_stuart_landau_async(self.ctx, dt, alpha, gain) },
            "t3_stuart_landau",
        )
    }

    // Safety clamp membrane potentials (async).
    // ═══════════════════════════════════════════════════════
    //  GPU trajectory + weight optimizer for Kuramoto BPTT
    // ═══════════════════════════════════════════════════════

    /// D2D copy: ctx.d_phases → d_traj[step * n] (async).
    /// Call after kuramoto_update() to save phase trajectory on GPU.
    pub fn copy_phases_to_traj_async(
        &self,
        d_traj: *mut c_float,
        step: usize,
    ) -> Result<(), String> {
        ok_or(
            unsafe { gpu_context_copy_phases_to_traj_async(self.ctx, d_traj, step as c_int) },
            "copy_phases_to_traj",
        )
    }

    /// D2D copy: ctx.d_coupling → d_traj[step * n] (async).
    /// Call after kuramoto_csr_coupling() (BEFORE kuramoto_update overwrites coupling).
    pub fn copy_coupling_to_traj_async(
        &self,
        d_traj: *mut c_float,
        step: usize,
    ) -> Result<(), String> {
        ok_or(
            unsafe { gpu_context_copy_coupling_to_traj_async(self.ctx, d_traj, step as c_int) },
            "copy_coupling_to_traj",
        )
    }

    /// Zero out grad_weights buffer on GPU (async).
    pub fn zero_grad_weights_async(&self, d_grad: *mut c_float) -> Result<(), String> {
        let n = self.n_synapses as c_int;
        ok_or(
            unsafe { gpu_zero_grad_weights_async(self.ctx, d_grad, n) },
            "zero_grad_weights",
        )
    }

    /// Apply weight update on GPU: SGD + sparsity + homeostasis + clamp (async).
    /// Updates ctx.d_csr_values in-place.
    pub fn weight_update_async(
        &self,
        d_grad_weights: *const c_float,
        lr: f32,
        sparsity_scale: f32,
        homeostasis_scale: f32,
        weight_clamp: f32,
    ) -> Result<(), String> {
        ok_or(
            unsafe {
                gpu_weight_update_async(
                    self.ctx,
                    d_grad_weights,
                    lr,
                    sparsity_scale,
                    homeostasis_scale,
                    weight_clamp,
                )
            },
            "weight_update",
        )
    }

    /// Compute gradient L2 norm via two-pass reduction (async, fully on GPU).
    ///
    /// Pass 1 (many blocks): each block computes sum(g²) for its slice.
    /// Pass 2 (single block): reduces partials → sqrt → writes to d_norm[0].
    ///
    /// d_partial must be pre-allocated with at least ((n_synapses + 255) / 256) floats.
    /// After this, call sync() then use d_norm value for clipping.
    pub fn grad_norm_async(
        &self,
        d_grad: *const c_float,
        d_norm: *mut c_float,
        d_partial: *mut c_float,
    ) -> Result<(), String> {
        let n = self.n_synapses as c_int;
        ok_or(
            unsafe { gpu_grad_norm_async(self.ctx, d_grad, d_norm, d_partial, n) },
            "grad_norm",
        )
    }

    /// Clip gradients by norm: if d_norm[0] > max_norm, scale g *= max_norm / norm (async).
    /// Reads d_norm from GPU memory — no D2H transfer needed.
    /// Must call sync() before weight_update_async to ensure clipping is applied.
    pub fn grad_clip_async(
        &self,
        d_grad: *mut c_float,
        d_norm: *const c_float,
        max_norm: f32,
    ) -> Result<(), String> {
        let n = self.n_synapses as c_int;
        ok_or(
            unsafe { gpu_grad_clip_async(self.ctx, d_grad, d_norm, max_norm, n) },
            "grad_clip",
        )
    }

    /// Run one BPTT epoch as a CUDA Graph (first call captures, subsequent replays).
    ///
    /// Does: forward → loss grad → BPTT backward → grad clip (if enabled).
    /// Weight update is NOT included — must be called separately after sync().
    ///
    /// After this returns, call sync() then weight_update_async() + sync().
    pub fn bptt_epoch_graph_replay(
        &self,
        d_phases_traj: *mut c_float,
        d_coupling_traj: *mut c_float,
        d_dtheta_traj: *mut c_float,
        d_grad_weights: *mut c_float,
        d_grad_norm: *mut c_float,
        d_grad_norm_partial: *mut c_float,
        n_steps: usize,
        dt: f32,
        coherence_start: usize,
        coherence_weight: f32,
        temporal_weight: f32,
        grad_clip_norm: f32,
    ) -> Result<(), String> {
        let has_graph = unsafe { gpu_graph_is_captured(self.ctx) } != 0;
        if !has_graph {
            // Capture the epoch as a CUDA Graph
            self.graph_begin_capture()?;
            unsafe {
                let err = gpu_bptt_epoch_async(
                    self.ctx,
                    d_phases_traj,
                    d_coupling_traj,
                    d_dtheta_traj,
                    d_grad_weights,
                    d_grad_norm,
                    d_grad_norm_partial,
                    n_steps as c_int,
                    dt,
                    coherence_start as c_int,
                    coherence_weight,
                    temporal_weight,
                    grad_clip_norm,
                );
                if err != 0 {
                    return Err(format!("CUDA bptt_epoch_async: error {err}"));
                }
            }
            self.graph_end_capture()?;
        }
        // Replay the graph (kernels execute with current d_phases, d_csr_values, etc.)
        ok_or(unsafe { gpu_graph_replay(self.ctx) }, "bptt_epoch_replay")
    }

    pub fn safety_clamp(&self, min_val: f32, max_val: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_safety_clamp_async(self.ctx, min_val, max_val) },
            "safety_clamp",
        )
    }

    /// Download spikes from GPU.
    pub fn download_spikes(&self, spikes: &mut [i32]) -> Result<(), String> {
        let n = spikes.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_download_spikes(self.ctx, spikes.as_mut_ptr(), n as c_int) },
            "download_spikes",
        )
    }

    /// Upload only v (no u/a/b/c/d — saves bandwidth for LIF path).
    pub fn upload_v_only(&self, v: &[f32]) -> Result<(), String> {
        let n = v.len().min(self.n_neurons) as c_int;
        ok_or(
            unsafe { gpu_context_upload_v_only(self.ctx, v.as_ptr(), n) },
            "upload_v_only",
        )
    }

    /// Run `n_steps` of Izhikevich + CSR propagate on GPU with batched spike accumulation.
    ///
    /// Same pattern as batch_lif_steps. Also downloads u at the end for state persistence.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn batch_izhikevich_steps(
        &self,
        external_I: &[f32],
        dt: f32,
        trace_decay: f32,
        dev_spike_buf: *mut c_int,
        host_spike_buf: &mut [i32],
        host_v: &mut [f32],
        host_u: &mut [f32],
        n_steps: usize,
    ) -> Result<(), String> {
        let n_external = external_I.len().min(self.n_neurons) as c_int;
        let n_neurons = self.n_neurons as c_int;
        ok_or(
            unsafe {
                gpu_context_batch_izhikevich_steps(
                    self.ctx,
                    external_I.as_ptr(),
                    n_external,
                    dt,
                    trace_decay,
                    dev_spike_buf,
                    host_spike_buf.as_mut_ptr(),
                    host_v.as_mut_ptr(),
                    host_u.as_mut_ptr(),
                    n_steps as c_int,
                    n_neurons,
                )
            },
            "batch_izhikevich_steps",
        )
    }

    /// Download Kuramoto phases from GPU.
    pub fn download_phases(&self, phases: &mut [f32]) -> Result<(), String> {
        let n = phases.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_download_phases(self.ctx, phases.as_mut_ptr(), n as c_int) },
            "download_phases",
        )
    }

    /// Download membrane potentials from GPU.
    pub fn download_v(&self, v: &mut [f32]) -> Result<(), String> {
        let n = v.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_download_v(self.ctx, v.as_mut_ptr(), n as c_int) },
            "download_v",
        )
    }

    /// Run `n_steps` of LIF + CSR propagate on GPU with batched spike accumulation.
    ///
    /// Accumulates spikes in `dev_spike_buf` via per-step D2D copies (no per-step D2H).
    /// Downloads entire spike buffer + final v in ONE sync at the end.
    /// Also updates `d_external_I` (per-step reset baseline) — fixes stale baseline bug.
    ///
    /// Args:
    ///   external_I: current external input current (updates both baseline and working I)
    ///   dev_spike_buf: pre-allocated device buffer `[n_steps × n_neurons]` ints
    ///   host_spike_buf: output buffer `[n_steps × n_neurons]` ints
    ///   host_v: output buffer `[n_neurons]` floats (final membrane potentials)
    ///   n_steps: number of LIF steps to run
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn batch_lif_steps(
        &self,
        external_I: &[f32],
        dev_spike_buf: *mut c_int,
        host_spike_buf: &mut [i32],
        host_v: &mut [f32],
        n_steps: usize,
    ) -> Result<(), String> {
        let n_external = external_I.len().min(self.n_neurons) as c_int;
        let n_neurons = self.n_neurons as c_int;
        ok_or(
            unsafe {
                gpu_context_batch_lif_steps(
                    self.ctx,
                    external_I.as_ptr(),
                    n_external,
                    dev_spike_buf,
                    host_spike_buf.as_mut_ptr(),
                    host_v.as_mut_ptr(),
                    n_steps as c_int,
                    n_neurons,
                )
            },
            "batch_lif_steps",
        )
    }

    /// Download recovery variable u from GPU.
    pub fn download_u(&self, u: &mut [f32]) -> Result<(), String> {
        let n = u.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_download_u(self.ctx, u.as_mut_ptr(), n as c_int) },
            "download_u",
        )
    }

    /// Download plastic weights from GPU.
    pub fn download_weights(&self, w: &mut [f32]) -> Result<(), String> {
        let n = w.len().min(self.n_synapses);
        ok_or(
            unsafe { gpu_context_download_weights(self.ctx, w.as_mut_ptr(), n as c_int) },
            "download_weights",
        )
    }

    /// Download adaptive threshold (theta / adex_w) from GPU.
    pub fn download_theta(&self, theta: &mut [f32]) -> Result<(), String> {
        let n = theta.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_download_adex_w(self.ctx, theta.as_mut_ptr(), n as c_int) },
            "download_theta",
        )
    }

    /// Upload adaptive threshold (theta / adex_w) to GPU.
    pub fn upload_theta(&self, theta: &[f32]) -> Result<(), String> {
        let n = theta.len().min(self.n_neurons);
        ok_or(
            unsafe { gpu_context_upload_adex_w(self.ctx, theta.as_ptr(), n as c_int) },
            "upload_theta",
        )
    }

    /// Get persistent noise state device pointer (allocates if NULL on first call).
    /// Used for SAW noise in genetic algorithm training.
    pub fn d_noise(&self) -> *mut std::ffi::c_void {
        unsafe { gpu_context_get_d_noise(self.ctx) }
    }

    // ─── Connectome CSR operations ─────────────────────

    /// Upload CSR connectome arrays to GPU (row_ptr, col_indices, values,
    /// nt_types, external_I). Store values in dedicated d_csr_values buffer.
    pub fn upload_connectome(
        &self,
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
        nt_types: &[f32],
        external_I: &[f32],
    ) -> Result<(), String> {
        let n_neurons = row_ptr.len() - 1;
        let n_synapses = values.len();
        let n_neurons_i = n_neurons.min(nt_types.len()).min(external_I.len()) as c_int;
        let n_synapses_i = n_synapses.min(col_indices.len()) as c_int;
        let err = unsafe {
            gpu_context_upload_connectome(
                self.ctx,
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                values.as_ptr(),
                nt_types.as_ptr(),
                external_I.as_ptr(),
                n_neurons_i,
                n_synapses_i,
            )
        };
        ok_or(err, "upload_connectome")
    }

    /// Initialize plastic weights from static connectome weights.
    pub fn init_plastic_weights(&self) -> Result<(), String> {
        ok_or(
            unsafe { gpu_context_init_plastic_weights(self.ctx) },
            "init_plastic_weights",
        )
    }

    /// Run one connectome STDP step: Izhikevich → count → STDP → CSR propagate (plastic).
    ///
    /// Order:
    ///   1. Izhikevich — produces spikes + updates traces
    ///   2. Count spikes
    ///   3. CSR STDP — updates d_weights from pre/post traces
    ///   4. CSR propagate using d_weights (plastic) — reset I + propagate
    ///
    /// Requires `init_plastic_weights()` to have been called after `upload_connectome()`.
    pub fn connectome_stdp_step(
        &self,
        dt: f32,
        trace_decay: f32,
        lr_ltp: f32,
        lr_ltd: f32,
        w_min: f32,
        w_max: f32,
        d_total_spikes: Option<*mut c_int>,
    ) -> Result<(), String> {
        let d_spike_ptr = d_total_spikes.unwrap_or(core::ptr::null_mut());
        // Step 1: Izhikevich — produces spikes + updates traces
        self.izhikevich_step(dt, trace_decay)?;
        // Step 2: Count spikes
        ok_or(
            unsafe { gpu_count_spikes_async(self.ctx, d_spike_ptr) },
            "count_spikes",
        )?;
        // Step 3: CSR STDP — update weights from traces
        ok_or(
            unsafe { gpu_connectome_stdp_async(self.ctx, lr_ltp, lr_ltd, w_min, w_max) },
            "connectome_stdp",
        )?;
        // Step 4: CSR propagate using plastic weights
        ok_or(
            unsafe { gpu_connectome_csr_plastic_async(self.ctx) },
            "connectome_csr_plastic",
        )
    }

    /// Upload LIF parameters to d_lif_params on GPU (sync).
    ///
    /// Packs as: [dt, tau_m[0..n], v_rest[0..n], v_thresh[0..n], v_reset[0..n]]
    /// This should be called once before running connectome_lif_step.
    pub fn upload_lif_params(
        &self,
        dt: f32,
        tau_m: &[f32],
        v_rest: &[f32],
        v_thresh: &[f32],
        v_reset: &[f32],
    ) -> Result<(), String> {
        let n = self
            .n_neurons
            .min(tau_m.len())
            .min(v_rest.len())
            .min(v_thresh.len())
            .min(v_reset.len());
        ok_or(
            unsafe {
                gpu_context_upload_lif_params(
                    self.ctx,
                    dt,
                    tau_m.as_ptr(),
                    v_rest.as_ptr(),
                    v_thresh.as_ptr(),
                    v_reset.as_ptr(),
                    n as c_int,
                )
            },
            "upload_lif_params",
        )
    }

    /// Run LIF step async (no sync). Requires params uploaded via upload_lif_params().
    pub fn lif_step_async(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_lif_step_async(self.ctx) }, "lif_step")
    }

    /// Run one LIF connectome step on GPU: LIF → count spikes → CSR propagate.
    /// All data stays on GPU — no H2D/D2H transfers.
    ///
    /// Order:
    ///   1. LIF step (produces NEW spikes from current I)
    ///   2. Count spikes (device-side counter)
    ///   3. Reset I (external baseline) + CSR propagate (for next step)
    ///
    /// Requires upload_lif_params() and upload_I(external_I) called once before the loop.
    pub fn connectome_lif_step(&self, d_total_spikes: Option<*mut c_int>) -> Result<(), String> {
        let d_spike_ptr = d_total_spikes.unwrap_or(core::ptr::null_mut());
        // Step 1: LIF — produces spikes from current I
        self.lif_step_async()?;
        // Step 2: Count spikes just produced
        ok_or(
            unsafe { gpu_count_spikes_async(self.ctx, d_spike_ptr) },
            "count_spikes",
        )?;
        // Step 3: Reset I to external baseline + CSR propagate for next step
        ok_or(
            unsafe { gpu_connectome_csr_async(self.ctx) },
            "connectome_csr",
        )
    }

    /// Run one connectome step on GPU: Izhikevich → count spikes → CSR propagate.
    /// All data stays on GPU — no H2D/D2H transfers.
    ///
    /// Order:
    ///   1. Izhikevich step (produces NEW spikes from current I)
    ///   2. Count spikes from izhikevich output (device-side counter)
    ///   3. Reset I (external baseline) + CSR propagate (for next step)
    ///
    /// This ensures ALL steps' spikes are counted (no off-by-one).
    ///
    /// Note: d_I must be initialized to external_I before calling this
    /// for the first time. Use `upload_I(external_I)` once before the loop.
    ///
    /// Uses `d_total_spikes` as device-side running spike counter.
    /// Pass `None` to skip spike counting.
    pub fn connectome_step(
        &self,
        dt: f32,
        trace_decay: f32,
        d_total_spikes: Option<*mut c_int>,
    ) -> Result<(), String> {
        let d_spike_ptr = d_total_spikes.unwrap_or(core::ptr::null_mut());
        // Step 1: Izhikevich — produces spikes from current I
        self.izhikevich_step(dt, trace_decay)?;
        // Step 2: Count spikes just produced by izhikevich
        ok_or(
            unsafe { gpu_count_spikes_async(self.ctx, d_spike_ptr) },
            "count_spikes",
        )?;
        // Step 3: Reset I to external baseline + CSR propagate for next step
        ok_or(
            unsafe { gpu_connectome_csr_async(self.ctx) },
            "connectome_csr",
        )
    }

    /// Allocate a device-side integer for spike counting.
    pub fn alloc_spike_counter(&self) -> Result<*mut c_int, String> {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, 4) };
        if err != 0 {
            return Err(format!("cudaMalloc spike counter: {err}"));
        }
        // Zero it out (sync — one-time init, no stream needed)
        let err = unsafe { cudaMemset(ptr, 0, 4) };
        if err != 0 {
            return Err(format!("cudaMemset spike counter: {err}"));
        }
        Ok(ptr as *mut c_int)
    }

    /// Read the device-side spike counter back to host (sync).
    pub fn read_spike_counter(&self, d_ptr: *mut c_int) -> Result<i32, String> {
        let mut host_val: c_int = 0;
        let err = unsafe {
            cudaMemcpy(
                &mut host_val as *mut c_int as *mut c_void,
                d_ptr as *const c_void,
                4,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 {
            return Err(format!("read_spike_counter: {err}"));
        }
        Ok(host_val as i32)
    }

    /// Free a device-side spike counter.
    pub fn free_spike_counter(&self, d_ptr: *mut c_int) {
        if !d_ptr.is_null() {
            unsafe {
                cudaFree(d_ptr as *mut c_void);
            }
        }
    }

    // ─── SNN async operations (LIF-AT, SORN) ───────────

    /// Run LIF-AT step via GPU context (uploads params, runs kernel, syncs).
    ///
    /// NOTE: v/theta/I state must already be on GPU via `upload_neurons()`.
    /// This function only uploads the per-step params (dt, tau_m, etc.) to `d_lif_params`.
    /// The theta offset is stored in `d_adex_w` (shared with AdEx — use one model per context).
    pub fn lif_at_step(
        &self,
        dt: f32,
        tau_m: &[f32],
        v_rest: &[f32],
        v_thresh: &[f32],
        v_reset: &[f32],
        tau_theta: &[f32],
        beta: &[f32],
    ) -> Result<(), String> {
        let n = self
            .n_neurons
            .min(tau_m.len())
            .min(v_rest.len())
            .min(v_thresh.len())
            .min(v_reset.len())
            .min(tau_theta.len())
            .min(beta.len());
        let mut flat = vec![0.0f32; 1 + 6 * n];
        flat[0] = dt;
        for i in 0..n {
            flat[1 + i] = tau_m[i];
            flat[1 + n + i] = v_rest[i];
            flat[1 + 2 * n + i] = v_thresh[i];
            flat[1 + 3 * n + i] = v_reset[i];
            flat[1 + 4 * n + i] = tau_theta[i];
            flat[1 + 5 * n + i] = beta[i];
        }
        ok_or(
            unsafe { gpu_lif_at_step(self.ctx, flat.as_ptr(), n as c_int) },
            "lif_at_step",
        )
    }

    /// Run Intrinsic Plasticity (threshold adaptation) via GPU context.
    pub fn intrinsic_plasticity(
        &self,
        lr_ip: f32,
        target_rate: f32,
        decay: f32,
    ) -> Result<(), String> {
        ok_or(
            unsafe { gpu_intrinsic_plasticity(self.ctx, lr_ip, target_rate, decay) },
            "intrinsic_plasticity",
        )
    }

    // ─── GCO (Graph-Coupled Oscillator) async operations ──

    /// Upload GCO state (v, u) to GPU (sync).
    pub fn gco_upload(&self, v: &[f32], u: &[f32]) -> Result<(), String> {
        let n = v.len().min(u.len()).min(self.n_neurons) as c_int;
        ok_or(
            unsafe { gpu_gco_upload_async(self.ctx, v.as_ptr(), u.as_ptr(), n) },
            "gco_upload",
        )
    }

    /// Upload GCO external drive (i_ext) to GPU (sync).
    pub fn gco_upload_drive(&self, i_ext: &[f32]) -> Result<(), String> {
        let n = i_ext.len().min(self.n_neurons) as c_int;
        ok_or(
            unsafe { gpu_gco_upload_drive_async(self.ctx, i_ext.as_ptr(), n) },
            "gco_upload_drive",
        )
    }

    /// Download GCO state (v, u) from GPU (sync).
    pub fn gco_download(&self, v: &mut [f32], u: &mut [f32]) -> Result<(), String> {
        let n = v.len().min(u.len()).min(self.n_neurons) as c_int;
        ok_or(
            unsafe { gpu_gco_download_async(self.ctx, v.as_mut_ptr(), u.as_mut_ptr(), n) },
            "gco_download",
        )
    }

    /// Compute GCO hamiltonian (CSR matmul) on GPU (async).
    /// After this, call `gco_step_async` to complete the step.
    pub fn gco_hamiltonian_async(&self) -> Result<(), String> {
        ok_or(
            unsafe { gpu_gco_hamiltonian_async(self.ctx) },
            "gco_hamiltonian",
        )
    }

    /// Apply Euler step + normalize to GCO (async).
    /// Must be called after `gco_hamiltonian_async` or `gco_combined_step_async`.
    pub fn gco_step_async(&self, dt: f32, tau: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_gco_step_async(self.ctx, dt, tau) }, "gco_step")
    }

    /// Combined GCO step: hamiltonian + Euler/normalize (async).
    /// Equivalent to `gco_hamiltonian_async` + `gco_step_async`.
    pub fn gco_combined_step_async(&self, dt: f32, tau: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_gco_combined_step_async(self.ctx, dt, tau) },
            "gco_combined_step",
        )
    }

    /// Full GCO step: hamiltonian → Euler/normalize, then sync.
    pub fn gco_step(&self, dt: f32, tau: f32) -> Result<(), String> {
        self.gco_combined_step_async(dt, tau)?;
        self.sync()
    }

    // ─── Aberton async operations ────────────────────────

    /// Upload Aberton state (real, imag) to GPU (sync).
    pub fn aber_upload(&self, real: &[f32], imag: &[f32]) -> Result<(), String> {
        let n = real.len().min(imag.len()).min(self.n_neurons) as c_int;
        ok_or(
            unsafe { gpu_aber_upload_async(self.ctx, real.as_ptr(), imag.as_ptr(), n) },
            "aber_upload",
        )
    }

    /// Download Aberton state + collapsed flags from GPU (sync).
    pub fn aber_download(
        &self,
        real: &mut [f32],
        imag: &mut [f32],
        collapsed: &mut [i32],
    ) -> Result<(), String> {
        let n = real
            .len()
            .min(imag.len())
            .min(collapsed.len())
            .min(self.n_neurons) as c_int;
        ok_or(
            unsafe {
                gpu_aber_download_async(
                    self.ctx,
                    real.as_mut_ptr(),
                    imag.as_mut_ptr(),
                    collapsed.as_mut_ptr(),
                    n,
                )
            },
            "aber_download",
        )
    }

    /// Run Aberton step + collapse on GPU (async).
    /// If use_synaptic_field=1, driver = h_real/h_imag (synaptic field),
    /// else driver = ψ-wave state (d_psi_v/d_psi_u).
    pub fn aber_step_async(
        &self,
        gamma: f32,
        omega: f32,
        gain: f32,
        threshold: f32,
        collapse_scale: f32,
        dt: f32,
        use_synaptic_field: bool,
    ) -> Result<(), String> {
        ok_or(
            unsafe {
                gpu_aber_step_async(
                    self.ctx,
                    gamma,
                    omega,
                    gain,
                    threshold,
                    collapse_scale,
                    dt,
                    use_synaptic_field as c_int,
                )
            },
            "aber_step",
        )
    }

    /// Full Aberton step + sync.
    pub fn aber_step(
        &self,
        gamma: f32,
        omega: f32,
        gain: f32,
        threshold: f32,
        collapse_scale: f32,
        dt: f32,
        use_synaptic_field: bool,
    ) -> Result<(), String> {
        self.aber_step_async(
            gamma,
            omega,
            gain,
            threshold,
            collapse_scale,
            dt,
            use_synaptic_field,
        )?;
        self.sync()
    }

    /// Combined GCO + Aberton step (async).
    /// Flow: hamiltonian → GCO normalize → aberton step + collapse.
    pub fn gco_aberton_combined_async(
        &self,
        dt: f32,
        tau: f32,
        aber_gamma: f32,
        aber_omega: f32,
        aber_gain: f32,
        aber_threshold: f32,
        aber_collapse_scale: f32,
        use_synaptic_field: bool,
    ) -> Result<(), String> {
        ok_or(
            unsafe {
                gpu_gco_aberton_combined_async(
                    self.ctx,
                    dt,
                    tau,
                    aber_gamma,
                    aber_omega,
                    aber_gain,
                    aber_threshold,
                    aber_collapse_scale,
                    use_synaptic_field as c_int,
                )
            },
            "gco_aberton_combined",
        )
    }

    /// Full combined GCO + Aberton step, then sync.
    pub fn gco_aberton_step(
        &self,
        dt: f32,
        tau: f32,
        aber_gamma: f32,
        aber_omega: f32,
        aber_gain: f32,
        aber_threshold: f32,
        aber_collapse_scale: f32,
        use_synaptic_field: bool,
    ) -> Result<(), String> {
        self.gco_aberton_combined_async(
            dt,
            tau,
            aber_gamma,
            aber_omega,
            aber_gain,
            aber_threshold,
            aber_collapse_scale,
            use_synaptic_field,
        )?;
        self.sync()
    }

    /// Synchronize stream.
    pub fn sync(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_context_sync(self.ctx) }, "sync")
    }

    // ─── CUDA Graph API ─────────────────────────────────

    /// Begin CUDA Graph capture. All subsequent kernel launches are recorded.
    pub fn graph_begin_capture(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_graph_begin_capture(self.ctx) }, "graph_begin")
    }

    /// End capture and instantiate the graph.
    pub fn graph_end_capture(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_graph_end_capture(self.ctx) }, "graph_end")
    }

    /// Replay the captured graph (single launch, zero overhead).
    pub fn graph_replay(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_graph_replay(self.ctx) }, "graph_replay")
    }

    /// Capture the standard simulation step as a CUDA Graph:
    ///   safety_clamp → fused_step (izhikevich+stdp) → kuramoto_prepare → kuramoto_update
    ///
    /// After calling this, use `graph_replay()` for zero-overhead replays.
    /// Note: during capture, error checking on individual kernel launches is skipped
    /// (captured kernels are not executed). Use `sync()` after `graph_replay()` to catch errors.
    pub fn capture_simulation_step(
        &self,
        dt: f32,
        trace_decay: f32,
        lr_ltd: f32,
        lr_ltp: f32,
        w_min: f32,
        w_max: f32,
        clamp_min: f32,
        clamp_max: f32,
    ) -> Result<(), String> {
        // Pre-allocate all arrays before capture (cudaMalloc forbidden during capture)
        ok_or(unsafe { gpu_context_ensure_all(self.ctx) }, "ensure_all")?;

        self.graph_begin_capture()?;

        // During capture, kernels are recorded, not executed.
        // We ignore individual launch errors and check at the end.
        unsafe {
            gpu_safety_clamp_async(self.ctx, clamp_min, clamp_max);
        }
        unsafe {
            gpu_fused_izhikevich_stdp_async(
                self.ctx,
                dt,
                trace_decay,
                lr_ltd,
                lr_ltp,
                w_min,
                w_max,
            );
        }
        unsafe {
            gpu_kuramoto_prepare_async(self.ctx);
        }
        unsafe {
            gpu_kuramoto_update_async(self.ctx, dt);
        }

        self.graph_end_capture()?;
        Ok(())
    }

    /// Run one simulation step.
    ///
    /// First call captures the CUDA Graph (expensive), subsequent calls
    /// only replay it (microsecond overhead).
    pub fn simulation_step(
        &self,
        dt: f32,
        trace_decay: f32,
        lr_ltd: f32,
        lr_ltp: f32,
        w_min: f32,
        w_max: f32,
        clamp_min: f32,
        clamp_max: f32,
    ) -> Result<(), String> {
        // Check if graph already captured (no-op peek)
        let has_graph = unsafe { gpu_graph_is_captured(self.ctx) } != 0;
        if !has_graph {
            self.capture_simulation_step(
                dt,
                trace_decay,
                lr_ltd,
                lr_ltp,
                w_min,
                w_max,
                clamp_min,
                clamp_max,
            )?;
        }
        self.graph_replay()?;
        self.sync()
    }

    // ═══════════════════════════════════════════════════════
    //  HH operations (direct computation with __expf())
    // ═══════════════════════════════════════════════════════

    /// Upload HH state: v, gates, Ii, g_leak, E_leak, C_m.
    /// Allocates device arrays on first call.
    ///
    /// `gates` is row-major `[n_neurons * n_gates]` — gate variables per neuron.
    /// `n_gates` is the total number of gates across all channels (e.g. 3 for NaV+KV).
    pub fn hh_upload_state(
        &self,
        v: &[f32],
        gates: &[f32],
        Ii: &[f32],
        g_leak: &[f32],
        E_leak: &[f32],
        C_m: &[f32],
        n_gates: usize,
    ) -> Result<(), String> {
        let n = self
            .n_neurons
            .min(v.len())
            .min(gates.len() / n_gates.max(1))
            .min(Ii.len())
            .min(g_leak.len())
            .min(E_leak.len())
            .min(C_m.len());
        ok_or(
            unsafe {
                gpu_hh_upload_state(
                    self.ctx,
                    v.as_ptr(),
                    gates.as_ptr(),
                    Ii.as_ptr(),
                    g_leak.as_ptr(),
                    E_leak.as_ptr(),
                    C_m.as_ptr(),
                    n as c_int,
                    n_gates as c_int,
                )
            },
            "hh_upload_state",
        )
    }

    /// Upload HH channel info (E_rev, g_max, powers, n_gates_per_ch).
    /// Must be called once after `hh_upload_state()`.
    ///
    /// On GPU, HH uses direct computation with `__expf()` — no LUTs.
    /// Channel arrays provide per-channel params to the kernel.
    pub fn hh_upload_channels(
        &self,
        E_rev_arr: &[f32],
        g_max_arr: &[f32],
        powers: &[u32],
        n_gates_per_ch: &[i32],
        n_gates: usize,
        n_channels: usize,
    ) -> Result<(), String> {
        ok_or(
            unsafe {
                gpu_hh_upload_channels(
                    self.ctx,
                    E_rev_arr.as_ptr(),
                    g_max_arr.as_ptr(),
                    powers.as_ptr(),
                    n_gates_per_ch.as_ptr(),
                    n_gates as c_int,
                    n_channels as c_int,
                )
            },
            "hh_upload_channels",
        )
    }

    /// Run one HH step (async, no sync).
    /// Requires state + channels to have been uploaded.
    /// tau_ref = 0 means no refractory period (CPU HHLUT compatible).
    pub fn hh_step_async(&self, dt: f32, tau_ref: f32) -> Result<(), String> {
        ok_or(
            unsafe { gpu_hh_step_async(self.ctx, dt, tau_ref) },
            "hh_step",
        )
    }

    /// Run one HH step, then sync.
    pub fn hh_step(&self, dt: f32, tau_ref: f32) -> Result<(), String> {
        self.hh_step_async(dt, tau_ref)?;
        self.sync()
    }

    /// Run `n_steps` of HH + CSR propagate + I_syn decay on GPU.
    ///
    /// All per-step work stays on GPU — no H2D/D2H transfers per step.
    /// Single D2H at end for spike buffer, final v, and final I_syn.
    ///
    /// Requires connectome to have been uploaded via `upload_connectome()`.
    /// Device spike buffer is managed internally (cached between calls).
    ///
    /// Args:
    ///   external_I — host buffer [n_neurons], uploaded to d_external_I once
    ///   host_spike_buf — output buffer [n_steps × n_neurons] ints
    ///   host_v — output buffer [n_neurons] floats (final v)
    ///   host_I_syn — output buffer [n_neurons] floats (final I_syn, modified in-place)
    ///   n_steps — number of HH steps to run
    ///   dt — HH timestep (ms)
    ///   tau_ref — refractory period (ms)
    ///   tau_syn — synaptic decay time constant (ms)
    ///   weight_scale — additional multiplier for CSR weights (e.g. 0.005 for HH)
    pub fn batch_hh_steps(
        &self,
        external_I: &[f32],
        host_spike_buf: &mut [i32],
        host_v: &mut [f32],
        host_I_syn: &mut [f32],
        n_steps: usize,
        dt: f32,
        tau_ref: f32,
        tau_syn: f32,
        weight_scale: f32,
    ) -> Result<(), String> {
        let n_external = external_I.len().min(self.n_neurons) as c_int;
        let n_neurons = self.n_neurons as c_int;
        ok_or(
            unsafe {
                gpu_context_batch_hh_steps(
                    self.ctx,
                    external_I.as_ptr(),
                    n_external,
                    host_spike_buf.as_mut_ptr(),
                    host_v.as_mut_ptr(),
                    host_I_syn.as_mut_ptr(),
                    n_steps as c_int,
                    n_neurons,
                    dt,
                    tau_ref,
                    tau_syn,
                    weight_scale,
                )
            },
            "batch_hh_steps",
        )
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                gpu_context_destroy(self.ctx);
            }
        }
    }
}
