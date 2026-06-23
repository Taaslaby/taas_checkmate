#![allow(unused_doc_comments, dead_code, improper_ctypes, non_snake_case)]

//! CUDA kernels for neuron-engine — максималочка GPU.
//!
//! ## Optimizations applied
//!
//! - **Persistent device memory** — all state lives on GPU between steps
//! - **CUDA stream** — async H2D/kernel/D2H pipelining
//! - **CUDA Graph** — capture simulation step as DAG, zero CPU launch overhead
//! - **Fused kernels** — Izhikevich+STDP, Kuramoto step, gates combined
//! - **sm_89 + fast math** — Ada Lovelace native codegen

use std::ffi::{c_float, c_int, c_void};

// ═══════════════════════════════════════════════════════════
//  Device management
// ═══════════════════════════════════════════════════════════

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    fn cuda_set_device(device: c_int) -> c_int;
    fn cuda_get_device_count(count: *mut c_int) -> c_int;
}

pub fn get_device_count() -> Result<i32, String> {
    let mut count: c_int = 0;
    let err = unsafe { cuda_get_device_count(&mut count as *mut c_int) } as i32;
    if err != 0 { Err(format!("cudaGetDeviceCount: error {err}")) }
    else { Ok(count as i32) }
}

pub fn set_device(device: i32) -> Result<(), String> {
    let err = unsafe { cuda_set_device(device as c_int) } as i32;
    if err != 0 { Err(format!("cudaSetDevice: error {err}")) }
    else { Ok(()) }
}

// ═══════════════════════════════════════════════════════════
//  GPU Context — persistent memory + streams + graphs
// ═══════════════════════════════════════════════════════════

struct GpuCtx;

extern "C" {
    fn gpu_context_create(n_neurons: c_int, n_synapses: c_int, n_decoders: c_int) -> *mut GpuCtx;
    fn gpu_context_destroy(ctx: *mut GpuCtx);
    fn gpu_context_sync(ctx: *mut GpuCtx) -> c_int;

    // Upload initial state
    fn gpu_context_upload_neurons(ctx: *mut GpuCtx,
        v: *const c_float, u: *const c_float,
        a: *const c_float, b: *const c_float,
        c: *const c_float, d: *const c_float,
        n: c_int) -> c_int;
    fn gpu_context_upload_kuramoto(ctx: *mut GpuCtx,
        phases: *const c_float, omegas: *const c_float, coupling: *const c_float,
        n: c_int) -> c_int;

    // Async kernels (no sync inside)
    fn gpu_izhikevich_step_async(ctx: *mut GpuCtx, dt: c_float, trace_decay: c_float) -> c_int;
    fn gpu_stdp_update_async(ctx: *mut GpuCtx,
        lr_ltd: c_float, lr_ltp: c_float, w_min: c_float, w_max: c_float) -> c_int;
    fn gpu_kuramoto_prepare_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_kuramoto_update_async(ctx: *mut GpuCtx, dt: c_float) -> c_int;
    fn gpu_safety_clamp_async(ctx: *mut GpuCtx, min_val: c_float, max_val: c_float) -> c_int;

    // Fused kernels
    fn gpu_fused_izhikevich_stdp_async(ctx: *mut GpuCtx,
        dt: c_float, trace_decay: c_float,
        lr_ltd: c_float, lr_ltp: c_float, w_min: c_float, w_max: c_float) -> c_int;

    // Connectome CSR upload + fused CSR step + Izhikevich step
    fn gpu_context_upload_connectome(ctx: *mut GpuCtx,
        row_ptr: *const c_int, col_indices: *const c_int, values: *const c_float,
        nt_types: *const c_float, external_I: *const c_float,
        n_neurons: c_int, n_synapses: c_int) -> c_int;
    fn gpu_connectome_csr_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_count_spikes_async(ctx: *mut GpuCtx, d_total_spikes: *mut c_int) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemset(ptr: *mut c_void, val: c_int, count: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;

    // Upload/download
    fn gpu_context_upload_I(ctx: *mut GpuCtx, I: *const c_float, n: c_int) -> c_int;
    fn gpu_context_download_spikes(ctx: *mut GpuCtx, spikes: *mut c_int, n: c_int) -> c_int;
    fn gpu_context_download_v(ctx: *mut GpuCtx, v: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_download_adex_w(ctx: *mut GpuCtx, w: *mut c_float, n: c_int) -> c_int;
    fn gpu_context_upload_adex_w(ctx: *mut GpuCtx, w: *const c_float, n: c_int) -> c_int;

    // CUDA Graph
    fn gpu_graph_begin_capture(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_end_capture(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_replay(ctx: *mut GpuCtx) -> c_int;
    fn gpu_graph_is_captured(ctx: *mut GpuCtx) -> c_int;
    // ψ-wave CUDA Graph
    fn gpu_psi_graph_replay(ctx: *mut GpuCtx, dt: c_float, hbar: c_float) -> c_int;
    fn gpu_psi_graph_invalidate(ctx: *mut GpuCtx) -> c_int;
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
    pub fn new(n_neurons: usize, n_synapses: usize) -> Result<Self, String> {
        let ctx = unsafe {
            gpu_context_create(n_neurons as c_int, n_synapses as c_int, n_neurons as c_int)
        };
        if ctx.is_null() {
            Err("GpuContext::new failed".into())
        } else {
            Ok(GpuContext { ctx, n_neurons, n_synapses })
        }
    }

    /// Upload neuron parameters (v, u, a, b, c, d).
    pub fn upload_neurons(&self, v: &[f32], u: &[f32],
        a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Result<(), String>
    {
        let n = v.len().min(u.len()).min(a.len());
        let err = unsafe {
            gpu_context_upload_neurons(self.ctx,
                v.as_ptr(), u.as_ptr(), a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr(),
                n as c_int)
        };
        ok_or(err, "upload_neurons")
    }

    /// Upload input current I to GPU.
    pub fn upload_I(&self, I: &[f32]) -> Result<(), String> {
        let n = I.len().min(self.n_neurons);
        ok_or(unsafe { gpu_context_upload_I(self.ctx, I.as_ptr(), n as c_int) }, "upload_I")
    }

    /// Upload Kuramoto state.
    pub fn upload_kuramoto(&self, phases: &[f32], omegas: &[f32], coupling: &[f32]) -> Result<(), String> {
        let n = phases.len().min(omegas.len()).min(coupling.len());
        let err = unsafe {
            gpu_context_upload_kuramoto(self.ctx, phases.as_ptr(), omegas.as_ptr(), coupling.as_ptr(), n as c_int)
        };
        ok_or(err, "upload_kuramoto")
    }

    /// Run single Izhikevich step (async, no sync).
    pub fn izhikevich_step(&self, dt: f32, trace_decay: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_izhikevich_step_async(self.ctx, dt, trace_decay) }, "izhikevich")
    }

    /// Run STDP update (async).
    pub fn stdp_update(&self, lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_stdp_update_async(self.ctx, lr_ltd, lr_ltp, w_min, w_max) }, "stdp")
    }

    /// Run fused Izhikevich + STDP (async).
    pub fn fused_step(&self, dt: f32, trace_decay: f32,
        lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32) -> Result<(), String>
    {
        ok_or(unsafe {
            gpu_fused_izhikevich_stdp_async(self.ctx, dt, trace_decay, lr_ltd, lr_ltp, w_min, w_max)
        }, "fused_step")
    }

    /// Run Kuramoto prepare (phases → cos/sin, async).
    pub fn kuramoto_prepare(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_kuramoto_prepare_async(self.ctx) }, "kuramoto_prepare")
    }

    /// Run Kuramoto update (async).
    pub fn kuramoto_update(&self, dt: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_kuramoto_update_async(self.ctx, dt) }, "kuramoto_update")
    }

    // ─── T3 async operations ──────────────────────────

    /// Upload T3 oscillator state to GPU.
    pub fn t3_upload(&self, phases: &[f32], amplitudes: &[f32], omegas: &[f32]) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()).min(omegas.len()) as c_int;
        ok_or(unsafe { gpu_t3_upload(self.ctx, phases.as_ptr(), amplitudes.as_ptr(), omegas.as_ptr(), n) }, "t3_upload")
    }

    /// Upload T3 drives to GPU.
    pub fn t3_upload_drives(&self, drives: &[f32]) -> Result<(), String> {
        let n = drives.len() as c_int;
        ok_or(unsafe { gpu_t3_upload_drives(self.ctx, drives.as_ptr(), n) }, "t3_upload_drives")
    }

    /// Upload T3 coupling to GPU.
    pub fn t3_upload_coupling(&self, coupling: &[f32]) -> Result<(), String> {
        let n = coupling.len() as c_int;
        ok_or(unsafe { gpu_t3_upload_coupling(self.ctx, coupling.as_ptr(), n) }, "t3_upload_coupling")
    }

    /// Download T3 oscillator state from GPU.
    pub fn t3_download(&self, phases: &mut [f32], amplitudes: &mut [f32]) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()) as c_int;
        ok_or(unsafe { gpu_t3_download(self.ctx, phases.as_mut_ptr(), amplitudes.as_mut_ptr(), n) }, "t3_download")
    }

    /// Download T3 cos/sin from GPU (for CPU coupling computation).
    pub fn t3_download_cos_sin(&self, cos_out: &mut [f32], sin_out: &mut [f32]) -> Result<(), String> {
        let n = cos_out.len().min(sin_out.len()) as c_int;
        ok_or(unsafe { gpu_t3_download_cos_sin(self.ctx, cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n) }, "t3_download_cos_sin")
    }

    /// Upload T3 omegas to GPU (done once at init).
    pub fn t3_upload_omegas(&self, omegas: &[f32]) -> Result<(), String> {
        let n = omegas.len() as c_int;
        ok_or(unsafe { gpu_t3_upload_omegas(self.ctx, omegas.as_ptr(), n) }, "t3_upload_omegas")
    }

    // ─── T3 async transfers (no sync inside) ───────────

    /// Async upload (no sync). Call sync() after all transfers + kernel.
    pub fn t3_upload_async(&self, phases: &[f32], amplitudes: &[f32], omegas: &[f32]) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()).min(omegas.len()) as c_int;
        ok_or(unsafe { gpu_t3_upload_async(self.ctx, phases.as_ptr(), amplitudes.as_ptr(), omegas.as_ptr(), n) }, "t3_upload_async")
    }

    /// Async upload drives (no sync).
    pub fn t3_upload_drives_async(&self, drives: &[f32]) -> Result<(), String> {
        let n = drives.len() as c_int;
        ok_or(unsafe { gpu_t3_upload_drives_async(self.ctx, drives.as_ptr(), n) }, "t3_upload_drives_async")
    }

    /// Async upload coupling (no sync).
    pub fn t3_upload_coupling_async(&self, coupling: &[f32]) -> Result<(), String> {
        let n = coupling.len() as c_int;
        ok_or(unsafe { gpu_t3_upload_coupling_async(self.ctx, coupling.as_ptr(), n) }, "t3_upload_coupling_async")
    }

    /// Async download (no sync).
    pub fn t3_download_async(&self, phases: &mut [f32], amplitudes: &mut [f32]) -> Result<(), String> {
        let n = phases.len().min(amplitudes.len()) as c_int;
        ok_or(unsafe { gpu_t3_download_async(self.ctx, phases.as_mut_ptr(), amplitudes.as_mut_ptr(), n) }, "t3_download_async")
    }

    /// Async download cos/sin (no sync).
    pub fn t3_download_cos_sin_async(&self, cos_out: &mut [f32], sin_out: &mut [f32]) -> Result<(), String> {
        let n = cos_out.len().min(sin_out.len()) as c_int;
        ok_or(unsafe { gpu_t3_download_cos_sin_async(self.ctx, cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n) }, "t3_download_cos_sin_async")
    }

    /// Run T3 prepare (phases → cos/sin, async).
    pub fn t3_prepare(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_t3_prepare_async(self.ctx) }, "t3_prepare")
    }

    /// Run T3 update (Stuart-Landau + Kuramoto, async).
    pub fn t3_update(&self, dt: f32, alpha: f32, gain: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_t3_update_async(self.ctx, dt, alpha, gain) }, "t3_update")
    }

    /// Run T3 Stuart-Landau only (amplitude update, async).
    pub fn t3_stuart_landau(&self, dt: f32, alpha: f32, gain: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_t3_stuart_landau_async(self.ctx, dt, alpha, gain) }, "t3_stuart_landau")
    }

    /// Safety clamp membrane potentials (async).
    pub fn safety_clamp(&self, min_val: f32, max_val: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_safety_clamp_async(self.ctx, min_val, max_val) }, "safety_clamp")
    }

    /// Download spikes from GPU.
    pub fn download_spikes(&self, spikes: &mut [i32]) -> Result<(), String> {
        let n = spikes.len().min(self.n_neurons);
        ok_or(unsafe { gpu_context_download_spikes(self.ctx, spikes.as_mut_ptr(), n as c_int) }, "download_spikes")
    }

    /// Download membrane potentials from GPU.
    pub fn download_v(&self, v: &mut [f32]) -> Result<(), String> {
        let n = v.len().min(self.n_neurons);
        ok_or(unsafe { gpu_context_download_v(self.ctx, v.as_mut_ptr(), n as c_int) }, "download_v")
    }

    /// Download adaptive threshold (theta / adex_w) from GPU.
    pub fn download_theta(&self, theta: &mut [f32]) -> Result<(), String> {
        let n = theta.len().min(self.n_neurons);
        ok_or(unsafe { gpu_context_download_adex_w(self.ctx, theta.as_mut_ptr(), n as c_int) }, "download_theta")
    }

    /// Upload adaptive threshold (theta / adex_w) to GPU.
    pub fn upload_theta(&self, theta: &[f32]) -> Result<(), String> {
        let n = theta.len().min(self.n_neurons);
        ok_or(unsafe { gpu_context_upload_adex_w(self.ctx, theta.as_ptr(), n as c_int) }, "upload_theta")
    }

    // ─── Connectome CSR operations ─────────────────────

    /// Upload CSR connectome arrays to GPU (row_ptr, col_indices, values,
    /// nt_types, external_I). Store values in dedicated d_csr_values buffer.
    pub fn upload_connectome(&self,
        row_ptr: &[i32], col_indices: &[i32], values: &[f32],
        nt_types: &[f32], external_I: &[f32],
    ) -> Result<(), String> {
        let n_neurons = row_ptr.len() - 1;
        let n_synapses = values.len();
        let n_neurons_i = n_neurons.min(nt_types.len()).min(external_I.len()) as c_int;
        let n_synapses_i = n_synapses.min(col_indices.len()) as c_int;
        let err = unsafe {
            gpu_context_upload_connectome(self.ctx,
                row_ptr.as_ptr(), col_indices.as_ptr(), values.as_ptr(),
                nt_types.as_ptr(), external_I.as_ptr(),
                n_neurons_i, n_synapses_i)
        };
        ok_or(err, "upload_connectome")
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
    pub fn connectome_step(&self, dt: f32, trace_decay: f32,
        d_total_spikes: Option<*mut c_int>) -> Result<(), String>
    {
        let d_spike_ptr = d_total_spikes.unwrap_or(core::ptr::null_mut());
        // Step 1: Izhikevich — produces spikes from current I
        self.izhikevich_step(dt, trace_decay)?;
        // Step 2: Count spikes just produced by izhikevich
        ok_or(unsafe { gpu_count_spikes_async(self.ctx, d_spike_ptr) }, "count_spikes")?;
        // Step 3: Reset I to external baseline + CSR propagate for next step
        ok_or(unsafe { gpu_connectome_csr_async(self.ctx) }, "connectome_csr")
    }

    /// Allocate a device-side integer for spike counting.
    pub fn alloc_spike_counter(&self) -> Result<*mut c_int, String> {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, 4) };
        if err != 0 { return Err(format!("cudaMalloc spike counter: {err}")); }
        // Zero it out (sync — one-time init, no stream needed)
        let err = unsafe { cudaMemset(ptr, 0, 4) };
        if err != 0 { return Err(format!("cudaMemset spike counter: {err}")); }
        Ok(ptr as *mut c_int)
    }

    /// Read the device-side spike counter back to host (sync).
    pub fn read_spike_counter(&self, d_ptr: *mut c_int) -> Result<i32, String> {
        let mut host_val: c_int = 0;
        let err = unsafe {
            cudaMemcpy(&mut host_val as *mut c_int as *mut c_void,
                d_ptr as *const c_void, 4, CUDA_MEMCPY_DEVICE_TO_HOST)
        };
        if err != 0 { return Err(format!("read_spike_counter: {err}")); }
        Ok(host_val as i32)
    }

    /// Free a device-side spike counter.
    pub fn free_spike_counter(&self, d_ptr: *mut c_int) {
        if !d_ptr.is_null() {
            unsafe { cudaFree(d_ptr as *mut c_void); }
        }
    }

    // ─── SNN async operations (LIF-AT, SORN) ───────────

    /// Run LIF-AT step via GPU context (uploads params, runs kernel, syncs).
    ///
    /// NOTE: v/theta/I state must already be on GPU via `upload_neurons()`.
    /// This function only uploads the per-step params (dt, tau_m, etc.) to `d_lif_params`.
    /// The theta offset is stored in `d_adex_w` (shared with AdEx — use one model per context).
    pub fn lif_at_step(&self,
        dt: f32, tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32],
        v_reset: &[f32], tau_theta: &[f32], beta: &[f32]) -> Result<(), String>
    {
        let n = self.n_neurons.min(tau_m.len()).min(v_rest.len()).min(v_thresh.len())
            .min(v_reset.len()).min(tau_theta.len()).min(beta.len());
        let mut flat = vec![0.0f32; 1 + 6*n];
        flat[0] = dt;
        for i in 0..n {
            flat[1+i] = tau_m[i];
            flat[1+n+i] = v_rest[i];
            flat[1+2*n+i] = v_thresh[i];
            flat[1+3*n+i] = v_reset[i];
            flat[1+4*n+i] = tau_theta[i];
            flat[1+5*n+i] = beta[i];
        }
        ok_or(unsafe {
            gpu_lif_at_step(self.ctx, flat.as_ptr(), n as c_int)
        }, "lif_at_step")
    }

    /// Run Intrinsic Plasticity (threshold adaptation) via GPU context.
    pub fn intrinsic_plasticity(&self,
        lr_ip: f32, target_rate: f32, decay: f32) -> Result<(), String>
    {
        ok_or(unsafe {
            gpu_intrinsic_plasticity(self.ctx, lr_ip, target_rate, decay)
        }, "intrinsic_plasticity")
    }

    // ─── ψ-wave async operations ──────────────────────────

    /// Upload ψ-wave state (v, u) to GPU (sync).
    pub fn psi_upload(&self, v: &[f32], u: &[f32]) -> Result<(), String> {
        let n = v.len().min(u.len()).min(self.n_neurons) as c_int;
        ok_or(unsafe { gpu_psi_upload_async(self.ctx, v.as_ptr(), u.as_ptr(), n) }, "psi_upload")
    }

    /// Upload ψ-wave external drive (i_ext) to GPU (sync).
    pub fn psi_upload_drive(&self, i_ext: &[f32]) -> Result<(), String> {
        let n = i_ext.len().min(self.n_neurons) as c_int;
        ok_or(unsafe { gpu_psi_upload_drive_async(self.ctx, i_ext.as_ptr(), n) }, "psi_upload_drive")
    }

    /// Download ψ-wave state (v, u) from GPU (sync).
    pub fn psi_download(&self, v: &mut [f32], u: &mut [f32]) -> Result<(), String> {
        let n = v.len().min(u.len()).min(self.n_neurons) as c_int;
        ok_or(unsafe { gpu_psi_download_async(self.ctx, v.as_mut_ptr(), u.as_mut_ptr(), n) }, "psi_download")
    }

    /// Compute ψ-wave hamiltonian (CSR matmul) on GPU (async).
    /// After this, call `psi_step_async` to complete the step.
    pub fn psi_hamiltonian_async(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_psi_hamiltonian_async(self.ctx) }, "psi_hamiltonian")
    }

    /// Apply Euler step + normalize to ψ-wave (async).
    /// Must be called after `psi_hamiltonian_async` or `psi_combined_step_async`.
    pub fn psi_step_async(&self, dt: f32, hbar: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_psi_step_async(self.ctx, dt, hbar) }, "psi_step")
    }

    /// Combined ψ-wave step: hamiltonian + Euler/normalize (async).
    /// Equivalent to `psi_hamiltonian_async` + `psi_step_async`.
    pub fn psi_combined_step_async(&self, dt: f32, hbar: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_psi_combined_step_async(self.ctx, dt, hbar) }, "psi_combined_step")
    }

    /// Full ψ-wave step: hamiltonian → Euler/normalize, then sync.
    pub fn psi_step(&self, dt: f32, hbar: f32) -> Result<(), String> {
        self.psi_combined_step_async(dt, hbar)?;
        self.sync()
    }

    // ─── Aberton async operations ────────────────────────

    /// Upload Aberton state (real, imag) to GPU (sync).
    pub fn aber_upload(&self, real: &[f32], imag: &[f32]) -> Result<(), String> {
        let n = real.len().min(imag.len()).min(self.n_neurons) as c_int;
        ok_or(unsafe { gpu_aber_upload_async(self.ctx, real.as_ptr(), imag.as_ptr(), n) }, "aber_upload")
    }

    /// Download Aberton state + collapsed flags from GPU (sync).
    pub fn aber_download(&self, real: &mut [f32], imag: &mut [f32],
        collapsed: &mut [i32]) -> Result<(), String>
    {
        let n = real.len().min(imag.len()).min(collapsed.len()).min(self.n_neurons) as c_int;
        ok_or(unsafe { gpu_aber_download_async(self.ctx, real.as_mut_ptr(), imag.as_mut_ptr(), collapsed.as_mut_ptr(), n) }, "aber_download")
    }

    /// Run Aberton step + collapse on GPU (async).
    /// If use_synaptic_field=1, driver = h_real/h_imag (synaptic field),
    /// else driver = ψ-wave state (d_psi_v/d_psi_u).
    pub fn aber_step_async(&self, gamma: f32, omega: f32, gain: f32,
        threshold: f32, collapse_scale: f32, dt: f32,
        use_synaptic_field: bool) -> Result<(), String>
    {
        ok_or(unsafe {
            gpu_aber_step_async(self.ctx, gamma, omega, gain,
                threshold, collapse_scale, dt, use_synaptic_field as c_int)
        }, "aber_step")
    }

    /// Full Aberton step + sync.
    pub fn aber_step(&self, gamma: f32, omega: f32, gain: f32,
        threshold: f32, collapse_scale: f32, dt: f32,
        use_synaptic_field: bool) -> Result<(), String>
    {
        self.aber_step_async(gamma, omega, gain, threshold, collapse_scale, dt, use_synaptic_field)?;
        self.sync()
    }

    /// Combined ψ-wave + Aberton step (async).
    /// Flow: hamiltonian → ψ normalize → aberton step + collapse.
    pub fn psi_aberton_combined_async(&self, dt: f32, hbar: f32,
        aber_gamma: f32, aber_omega: f32, aber_gain: f32,
        aber_threshold: f32, aber_collapse_scale: f32,
        use_synaptic_field: bool) -> Result<(), String>
    {
        ok_or(unsafe {
            gpu_psi_aberton_combined_async(self.ctx, dt, hbar,
                aber_gamma, aber_omega, aber_gain,
                aber_threshold, aber_collapse_scale,
                use_synaptic_field as c_int)
        }, "psi_aberton_combined")
    }

    /// Full combined ψ-wave + Aberton step, then sync.
    pub fn psi_aberton_step(&self, dt: f32, hbar: f32,
        aber_gamma: f32, aber_omega: f32, aber_gain: f32,
        aber_threshold: f32, aber_collapse_scale: f32,
        use_synaptic_field: bool) -> Result<(), String>
    {
        self.psi_aberton_combined_async(dt, hbar,
            aber_gamma, aber_omega, aber_gain,
            aber_threshold, aber_collapse_scale,
            use_synaptic_field)?;
        self.sync()
    }

    /// Synchronize stream.
    pub fn sync(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_context_sync(self.ctx) }, "sync")
    }

    // ─── ψ-wave CUDA Graph ────────────────────────────────

    /// Replay ψ-wave CUDA Graph. Captures automatically on first call
    /// or when dt/hbar change.  Subsequent replays skip host-side launch.
    pub fn psi_graph_replay(&self, dt: f32, hbar: f32) -> Result<(), String> {
        ok_or(unsafe { gpu_psi_graph_replay(self.ctx, dt, hbar) }, "psi_graph_replay")
    }

    /// Invalidate ψ-wave CUDA Graph so next replay re-captures.
    pub fn psi_graph_invalidate(&self) -> Result<(), String> {
        ok_or(unsafe { gpu_psi_graph_invalidate(self.ctx) }, "psi_graph_invalidate")
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
    pub fn capture_simulation_step(&self, dt: f32, trace_decay: f32,
        lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32,
        clamp_min: f32, clamp_max: f32) -> Result<(), String>
    {
        self.graph_begin_capture()?;

        // During capture, kernels are recorded, not executed.
        // We ignore individual launch errors and check at the end.
        unsafe { gpu_safety_clamp_async(self.ctx, clamp_min, clamp_max); }
        unsafe { gpu_fused_izhikevich_stdp_async(self.ctx, dt, trace_decay,
            lr_ltd, lr_ltp, w_min, w_max); }
        unsafe { gpu_kuramoto_prepare_async(self.ctx); }
        unsafe { gpu_kuramoto_update_async(self.ctx, dt); }

        self.graph_end_capture()?;
        Ok(())
    }

    /// Run one simulation step.
    ///
    /// First call captures the CUDA Graph (expensive), subsequent calls
    /// only replay it (microsecond overhead).
    pub fn simulation_step(&self, dt: f32, trace_decay: f32,
        lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32,
        clamp_min: f32, clamp_max: f32) -> Result<(), String>
    {
        // Check if graph already captured (no-op peek)
        let has_graph = unsafe { gpu_graph_is_captured(self.ctx) } != 0;
        if !has_graph {
            self.capture_simulation_step(dt, trace_decay, lr_ltd, lr_ltp,
                w_min, w_max, clamp_min, clamp_max)?;
        }
        self.graph_replay()?;
        self.sync()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { gpu_context_destroy(self.ctx); }
        }
    }
}

fn ok_or(err: c_int, name: &str) -> Result<(), String> {
    if err == 0 { Ok(()) } else { Err(format!("CUDA {name} error: {err}")) }
}

// ═══════════════════════════════════════════════════════════
//  Legacy individual kernel calls (kept for compat)
// ═══════════════════════════════════════════════════════════

// XOR Attention
struct XorCtx;
extern "C" {
    fn xor_attention_create(n_keys: c_int, n_bytes: c_int) -> *mut XorCtx;
    fn xor_attention_compute(ctx: *mut XorCtx,
        query: *const u8, keys: *const u8, values: *const c_float, output: *mut c_float) -> c_int;
    fn xor_attention_destroy(ctx: *mut XorCtx);
    fn cuda_xor_attention(query: *const u8, keys: *const u8, values: *const c_float,
        output: *mut c_float, n_keys: c_int, n_bytes: c_int) -> c_int;
}

pub struct XorAttention {
    ctx: *mut XorCtx, n_keys: usize, n_bytes: usize,
}

impl XorAttention {
    pub fn new(n_keys: usize, n_bytes: usize) -> Result<Self, String> {
        assert!(n_bytes % 8 == 0);
        let ctx = unsafe { xor_attention_create(n_keys as c_int, n_bytes as c_int) };
        if ctx.is_null() { Err("xor_attention_create failed".into()) }
        else { Ok(XorAttention { ctx, n_keys, n_bytes }) }
    }

    pub fn compute(&self, query: &[u8], keys: &[u8], values: &[f32], output: &mut [f32]) -> Result<(), String> {
        let err = unsafe {
            xor_attention_compute(self.ctx,
                query.as_ptr(), keys.as_ptr(), values.as_ptr(), output.as_mut_ptr())
        };
        ok_or(err, "xor_attention")
    }

    pub fn compute_scalar(&self, query: &[u8], keys: &[u8], values: &[f32]) -> Result<f32, String> {
        let mut out = vec![0.0f32];
        self.compute(query, keys, values, &mut out)?;
        Ok(out[0])
    }
}

impl Drop for XorAttention {
    fn drop(&mut self) {
        if !self.ctx.is_null() { unsafe { xor_attention_destroy(self.ctx); } }
    }
}

// Legacy XOR attention single-call
pub fn xor_attention(query: &[u8], keys: &[u8], values: &[f32],
    output: &mut [f32], n_keys: usize, n_bytes: usize) -> Result<(), String>
{
    assert!(n_bytes % 8 == 0);
    ok_or(unsafe {
        cuda_xor_attention(query.as_ptr(), keys.as_ptr(), values.as_ptr(),
            output.as_mut_ptr(), n_keys as c_int, n_bytes as c_int)
    }, "xor_attention")
}

// ─── Individual wrappers (single-kernel API) ─────────────

extern "C" {
    fn cuda_izhikevich_step(v: *mut c_float, u: *mut c_float, trace: *mut c_float,
        a: *const c_float, b: *const c_float, I: *const c_float,
        dt: c_float, trace_decay: c_float, spikes: *mut c_int,
        c: *const c_float, d: *const c_float,
        refractory: *mut c_float, tau_ref: c_float, n: c_int) -> c_int;
    fn cuda_lif_step(v: *mut c_float, I: *mut c_float, spikes: *mut c_int,
        params: *const c_float, n: c_int) -> c_int;
    fn cuda_adex_step(v: *mut c_float, u: *mut c_float, I: *mut c_float,
        spikes: *mut c_int, params: *const c_float, n: c_int) -> c_int;
    fn cuda_stdp_update(weights: *mut c_float, traces_pre: *const c_float,
        traces_post: *const c_float, lr_ltd: c_float, lr_ltp: c_float,
        w_min: c_float, w_max: c_float, n: c_int) -> c_int;
    fn cuda_hebbian_update(weights: *mut c_float, cos_diff: *const c_float,
        lr: c_float, decay: c_float, dt: c_float, n: c_int) -> c_int;
    fn cuda_kuramoto_prepare(phases: *const c_float, cos_out: *mut c_float,
        sin_out: *mut c_float, n: c_int) -> c_int;
    fn cuda_kuramoto_update(phases: *mut c_float, omegas: *const c_float,
        coupling: *const c_float, dt: c_float, n: c_int) -> c_int;
    fn cuda_safety_clamp(data: *mut c_float, min_val: c_float, max_val: c_float, n: c_int) -> c_int;
    fn cuda_stochastic_encoder(target_I: *mut c_float, inputs: *const c_float,
        noise: *const c_float, threshold: c_float, spike_val: c_float, n: c_int) -> c_int;
    fn cuda_thalamic_gate(target_I: *mut c_float, source_v: *const c_float,
        gate_v: *const c_float, threshold: c_float, gain: c_float, n: c_int) -> c_int;
    fn cuda_sigmoid_gate(output: *mut c_float, input: *const c_float,
        gain: c_float, threshold: c_float, n: c_int) -> c_int;
    fn cuda_phase_gate_apply(output: *mut c_float, input: *const c_float,
        r: *const c_float, gain: c_float, threshold: c_float, n: c_int) -> c_int;
    fn cuda_t3_drive_encoder(drives: *mut c_float, inputs: *const c_float,
        gain: c_float, n: c_int) -> c_int;
    fn cuda_t3_inverse_kuramoto(drive_out: *mut c_float, r: *const c_float,
        r_prev: *const c_float, alpha: c_float, gain: c_float,
        eps: c_float, decay: c_float, n: c_int) -> c_int;
    fn cuda_nlms_train(weights: *mut c_float, bias: *mut c_float,
        states: *const c_float, error: c_float, lr: c_float, eps: c_float, n: c_int) -> c_int;
    fn cuda_kalman_update(weights: *mut c_float, cov: *mut c_float,
        states: *const c_float, error: c_float, s: c_float, n: c_int) -> c_int;
    fn cuda_echo_tap(output: *mut c_float, buffer: *const c_float,
        n_osc: c_int, delay_len: c_int, delay_idx: c_int) -> c_int;

    // T3 CUDA kernels
    fn cuda_t3_update(phases: *mut c_float, amplitudes: *mut c_float,
        omegas: *const c_float, coupling: *const c_float, drives: *const c_float,
        dt: c_float, alpha: c_float, gain: c_float, n: c_int) -> c_int;
    fn cuda_t3_prepare(phases: *mut c_float, cos_out: *mut c_float,
        sin_out: *mut c_float, n: c_int) -> c_int;

    // ψ-wave synchronous kernel
    fn cuda_psi_step(
        v: *mut c_float, u: *mut c_float,
        row_ptr: *const c_int, col_indices: *const c_int, values: *const c_float,
        i_ext: *const c_float,
        dt: c_float, hbar: c_float,
        n_neurons: c_int) -> c_int;

    fn gpu_t3_prepare_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_t3_update_async(ctx: *mut GpuCtx,
        dt: c_float, alpha: c_float, gain: c_float) -> c_int;
    fn gpu_t3_stuart_landau_async(ctx: *mut GpuCtx,
        dt: c_float, alpha: c_float, gain: c_float) -> c_int;
    fn gpu_t3_upload(ctx: *mut GpuCtx,
        phases: *const c_float, amplitudes: *const c_float,
        omegas: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_drives(ctx: *mut GpuCtx,
        drives: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_coupling(ctx: *mut GpuCtx,
        coupling: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_download(ctx: *mut GpuCtx,
        phases: *mut c_float, amplitudes: *mut c_float, n: c_int) -> c_int;
    fn gpu_t3_download_cos_sin(ctx: *mut GpuCtx,
        cos_out: *mut c_float, sin_out: *mut c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_omegas(ctx: *mut GpuCtx,
        omegas: *const c_float, n: c_int) -> c_int;
    // Async transfers (no sync inside — caller syncs once)
    fn gpu_t3_upload_async(ctx: *mut GpuCtx,
        phases: *const c_float, amplitudes: *const c_float,
        omegas: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_drives_async(ctx: *mut GpuCtx,
        drives: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_coupling_async(ctx: *mut GpuCtx,
        coupling: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_upload_omegas_async(ctx: *mut GpuCtx,
        omegas: *const c_float, n: c_int) -> c_int;
    fn gpu_t3_download_async(ctx: *mut GpuCtx,
        phases: *mut c_float, amplitudes: *mut c_float, n: c_int) -> c_int;
    fn gpu_t3_download_cos_sin_async(ctx: *mut GpuCtx,
        cos_out: *mut c_float, sin_out: *mut c_float, n: c_int) -> c_int;

    // ψ-wave async operations
    fn gpu_psi_hamiltonian_async(ctx: *mut GpuCtx) -> c_int;
    fn gpu_psi_step_async(ctx: *mut GpuCtx, dt: c_float, hbar: c_float) -> c_int;
    fn gpu_psi_combined_step_async(ctx: *mut GpuCtx, dt: c_float, hbar: c_float) -> c_int;
    fn gpu_psi_upload_async(ctx: *mut GpuCtx,
        host_v: *const c_float, host_u: *const c_float, n: c_int) -> c_int;
    fn gpu_psi_download_async(ctx: *mut GpuCtx,
        host_v: *mut c_float, host_u: *mut c_float, n: c_int) -> c_int;
    fn gpu_psi_upload_drive_async(ctx: *mut GpuCtx,
        host_i_ext: *const c_float, n: c_int) -> c_int;

    // Aberton async operations
    fn gpu_aber_upload_async(ctx: *mut GpuCtx,
        host_real: *const c_float, host_imag: *const c_float, n: c_int) -> c_int;
    fn gpu_aber_download_async(ctx: *mut GpuCtx,
        host_real: *mut c_float, host_imag: *mut c_float,
        host_collapsed: *mut c_int, n: c_int) -> c_int;
    fn gpu_aber_step_async(ctx: *mut GpuCtx,
        gamma: c_float, omega: c_float, gain: c_float,
        threshold: c_float, collapse_scale: c_float,
        dt: c_float, use_synaptic_field: c_int) -> c_int;
    fn gpu_psi_aberton_combined_async(ctx: *mut GpuCtx,
        dt: c_float, hbar: c_float,
        aber_gamma: c_float, aber_omega: c_float, aber_gain: c_float,
        aber_threshold: c_float, aber_collapse_scale: c_float,
        use_synaptic_field: c_int) -> c_int;
}

pub fn step_izhikevich(v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], I: &mut [f32], dt: f32, trace_decay: f32,
    spikes: &mut [i32], c: &[f32], d: &[f32],
    refractory: &mut [f32], tau_ref: f32) -> Result<(), String>
{
    let n = v.len();
    ok_or(unsafe {
        cuda_izhikevich_step(v.as_mut_ptr(), u.as_mut_ptr(), trace.as_mut_ptr(),
            a.as_ptr(), b.as_ptr(), I.as_mut_ptr(), dt, trace_decay,
            spikes.as_mut_ptr(), c.as_ptr(), d.as_ptr(),
            refractory.as_mut_ptr(), tau_ref, n as c_int)
    }, "izhikevich")
}

pub fn step_lif(v: &mut [f32], I: &mut [f32], spikes: &mut [i32],
    dt: f32, tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32], v_reset: &[f32]) -> Result<(), String>
{
    let n = v.len();
    let mut flat = vec![0.0f32; 1 + 4*n];
    flat[0] = dt;
    for i in 0..n {
        flat[1+i] = tau_m[i];
        flat[1+n+i] = v_rest[i];
        flat[1+2*n+i] = v_thresh[i];
        flat[1+3*n+i] = v_reset[i];
    }
    ok_or(unsafe { cuda_lif_step(v.as_mut_ptr(), I.as_mut_ptr(), spikes.as_mut_ptr(), flat.as_ptr(), n as c_int) }, "lif")
}

pub fn step_psi(v: &mut [f32], u: &mut [f32],
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    i_ext: Option<&[f32]>,
    dt: f32, hbar: f32) -> Result<(), String>
{
    let n = v.len() as c_int;
    let i_ext_ptr = i_ext.map(|x| x.as_ptr()).unwrap_or(core::ptr::null());
    ok_or(unsafe {
        cuda_psi_step(v.as_mut_ptr(), u.as_mut_ptr(),
            row_ptr.as_ptr(), col_indices.as_ptr(), values.as_ptr(),
            i_ext_ptr, dt, hbar, n)
    }, "psi_step")
}

pub fn stdp_update(weights: &mut [f32], traces_pre: &[f32], traces_post: &[f32],
    lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32) -> Result<(), String>
{
    let n = weights.len();
    ok_or(unsafe { cuda_stdp_update(weights.as_mut_ptr(), traces_pre.as_ptr(),
        traces_post.as_ptr(), lr_ltd, lr_ltp, w_min, w_max, n as c_int) }, "stdp")
}

pub fn hebbian_update(weights: &mut [f32], cos_diff: &[f32],
    lr: f32, decay: f32, dt: f32) -> Result<(), String>
{
    let n = weights.len();
    ok_or(unsafe { cuda_hebbian_update(weights.as_mut_ptr(), cos_diff.as_ptr(),
        lr, decay, dt, n as c_int) }, "hebbian")
}

pub fn kuramoto_prepare(phases: &[f32], cos_out: &mut [f32], sin_out: &mut [f32]) -> Result<(), String> {
    let n = phases.len();
    ok_or(unsafe { cuda_kuramoto_prepare(phases.as_ptr(), cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n as c_int) }, "kuramoto_prepare")
}

pub fn kuramoto_update(phases: &mut [f32], omegas: &[f32], coupling: &[f32], dt: f32) -> Result<(), String> {
    let n = phases.len();
    ok_or(unsafe { cuda_kuramoto_update(phases.as_mut_ptr(), omegas.as_ptr(), coupling.as_ptr(), dt, n as c_int) }, "kuramoto_update")
}

pub fn safety_clamp(data: &mut [f32], min_val: f32, max_val: f32) -> Result<(), String> {
    let n = data.len() as c_int;
    ok_or(unsafe { cuda_safety_clamp(data.as_mut_ptr(), min_val, max_val, n) }, "safety_clamp")
}

pub fn stochastic_encoder(target_I: &mut [f32], inputs: &[f32], noise: &[f32],
    threshold: f32, spike_val: f32) -> Result<(), String>
{
    let n = target_I.len().min(inputs.len()).min(noise.len()) as c_int;
    ok_or(unsafe { cuda_stochastic_encoder(target_I.as_mut_ptr(), inputs.as_ptr(),
        noise.as_ptr(), threshold, spike_val, n) }, "stochastic_encoder")
}

pub fn thalamic_gate(target_I: &mut [f32], source_v: &[f32], gate_v: &[f32],
    threshold: f32, gain: f32) -> Result<(), String>
{
    let n = target_I.len().min(source_v.len()).min(gate_v.len()) as c_int;
    ok_or(unsafe { cuda_thalamic_gate(target_I.as_mut_ptr(), source_v.as_ptr(),
        gate_v.as_ptr(), threshold, gain, n) }, "thalamic_gate")
}

pub fn sigmoid_gate(output: &mut [f32], input: &[f32], gain: f32, threshold: f32) -> Result<(), String> {
    let n = output.len().min(input.len()) as c_int;
    ok_or(unsafe { cuda_sigmoid_gate(output.as_mut_ptr(), input.as_ptr(), gain, threshold, n) }, "sigmoid_gate")
}

pub fn phase_gate_apply(output: &mut [f32], input: &[f32], r: &[f32],
    gain: f32, threshold: f32) -> Result<(), String>
{
    let n = output.len().min(input.len()).min(r.len()) as c_int;
    ok_or(unsafe { cuda_phase_gate_apply(output.as_mut_ptr(), input.as_ptr(), r.as_ptr(), gain, threshold, n) }, "phase_gate")
}

pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) -> Result<(), String> {
    let n = drives.len().min(inputs.len()) as c_int;
    ok_or(unsafe { cuda_t3_drive_encoder(drives.as_mut_ptr(), inputs.as_ptr(), gain, n) }, "t3_drive")
}

pub fn t3_inverse_kuramoto(drive_out: &mut [f32], r: &[f32], r_prev: &[f32],
    alpha: f32, gain: f32, eps: f32, decay: f32) -> Result<(), String>
{
    let n = drive_out.len().min(r.len()).min(r_prev.len()) as c_int;
    ok_or(unsafe { cuda_t3_inverse_kuramoto(drive_out.as_mut_ptr(), r.as_ptr(),
        r_prev.as_ptr(), alpha, gain, eps, decay, n) }, "t3_inverse")
}

pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32],
    error: f32, lr: f32, eps: f32) -> Result<(), String>
{
    let n = weights.len() as c_int;
    ok_or(unsafe { cuda_nlms_train(weights.as_mut_ptr(), bias.as_mut_ptr(),
        states.as_ptr(), error, lr, eps, n) }, "nlms")
}

pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32],
    error: f32, s: f32) -> Result<(), String>
{
    let n = weights.len() as c_int;
    ok_or(unsafe { cuda_kalman_update(weights.as_mut_ptr(), cov.as_mut_ptr(),
        states.as_ptr(), error, s, n) }, "kalman")
}

pub fn echo_tap(output: &mut [f32], buffer: &[f32],
    n_osc: usize, delay_len: usize, delay_idx: usize) -> Result<(), String>
{
    ok_or(unsafe {
        cuda_echo_tap(output.as_mut_ptr(), buffer.as_ptr(),
            n_osc as c_int, delay_len as c_int, delay_idx as c_int)
    }, "echo_tap")
}

// ═══════════════════════════════════════════════════════════
//  T3 kernels (host wrappers)
// ═══════════════════════════════════════════════════════════

pub fn t3_update(phases: &mut [f32], amplitudes: &mut [f32],
    omegas: &[f32], coupling: &[f32], drives: &[f32],
    dt: f32, alpha: f32, gain: f32) -> Result<(), String>
{
    let n = phases.len() as c_int;
    ok_or(unsafe {
        cuda_t3_update(phases.as_mut_ptr(), amplitudes.as_mut_ptr(),
            omegas.as_ptr(), coupling.as_ptr(), drives.as_ptr(),
            dt, alpha, gain, n)
    }, "t3_update")
}

pub fn t3_prepare(phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]) -> Result<(), String> {
    let n = phases.len() as c_int;
    ok_or(unsafe { cuda_t3_prepare(phases.as_mut_ptr(), cos_out.as_mut_ptr(), sin_out.as_mut_ptr(), n) }, "t3_prepare")
}

// ═══════════════════════════════════════════════════════════
//  T3 BPTT Jacobian kernel (K1)
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_pst_jacobian(
        d_final_state: *const c_float,
        d_grad_pst: *const c_float,
        d_sums: *mut c_float,
        d_v: *mut c_float,
        n: c_int,
    ) -> c_int;
}

/// Compute PST Jacobian-vector product on GPU.
///
/// `final_state`: packed [th1(n), th2(n), th3(n), r1(n), r2(n), r3(n)] — on device
/// `grad_pst`: [pst_dim] — on device
/// `d_sums`: scratch [12] — on device (can be reused)
/// `v_out`: [6*n] packed as [v_th1, v_th2, v_th3, v_r1, v_r2, v_r3] — on device
pub fn pst_jacobian_gpu(
    d_final_state: &DevicePtr<f32>,
    d_grad_pst: &DevicePtr<f32>,
    d_sums: &DevicePtr<f32>,
    d_v: &DevicePtr<f32>,
    n: usize,
) -> Result<(), String> {
    ok_or(unsafe {
        cuda_pst_jacobian(
            d_final_state.as_ptr() as *const c_float,
            d_grad_pst.as_ptr() as *const c_float,
            d_sums.as_ptr() as *mut c_float,
            d_v.as_ptr() as *mut c_float,
            n as c_int,
        )
    }, "pst_jacobian_gpu")
}

// ─── Device memory helper ─────────────────────────────────

/// A simple RAII wrapper for CUDA device memory.
pub struct DevicePtr<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DevicePtr<T> {
    pub fn new(len: usize) -> Result<Self, String> {
        let mut ptr: *mut T = std::ptr::null_mut();
        let err = unsafe {
            cudaMalloc(
                &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
                len * std::mem::size_of::<T>(),
            )
        };
        if err != 0 || ptr.is_null() {
            return Err(format!("cudaMalloc({}×{}B): error {err}", len, std::mem::size_of::<T>()));
        }
        Ok(Self { ptr, len })
    }

    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
    pub fn len(&self) -> usize { self.len }

    pub fn copy_from_host(&self, src: &[T]) -> Result<(), String> {
        let n = src.len().min(self.len);
        let err = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                n * std::mem::size_of::<T>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 { Err(format!("cudaMemcpy H2D: error {err}")) } else { Ok(()) }
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), String> {
        let n = dst.len().min(self.len);
        let err = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                n * std::mem::size_of::<T>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 { Err(format!("cudaMemcpy D2H: error {err}")) } else { Ok(()) }
    }
}

impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr as *mut std::ffi::c_void); }
        }
    }
}

// Required for extern block — these are already declared elsewhere but we need them here
// The cudaMalloc/cudaMemcpy/cudaFree are already declared in the connectome section above

// ═══════════════════════════════════════════════════════════
//  SNN kernels (LIF-AT, SORN, Spike Encoding)
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_lif_at_step(v: *mut c_float, theta: *mut c_float, I: *mut c_float,
        spikes: *mut c_int, params: *const c_float, n: c_int) -> c_int;
    fn cuda_intrinsic_plasticity(theta: *mut c_float, spikes: *const c_int,
        lr_ip: c_float, target_rate: c_float, decay: c_float, n: c_int) -> c_int;
    fn cuda_synaptic_normalization(weights: *mut c_float,
        n_pre: c_int, n_post: c_int, target_sum: c_float, eps: c_float) -> c_int;
    fn cuda_bsa_encode(spikes_out: *mut c_int, signal: *const c_float,
        reconstructed: *const c_float, threshold: c_float, n: c_int) -> c_int;
    fn cuda_phase_encode(spikes_out: *mut c_int, phases: *mut c_float,
        cos_phases: *const c_float, signal: *const c_float,
        threshold: c_float, gain: c_float, n: c_int) -> c_int;
    fn cuda_ttfs_init(fire_times: *mut c_int, values: *const c_float,
        max_delay: c_int, n: c_int) -> c_int;
    fn cuda_ttfs_step(spikes_out: *mut c_int, fire_times: *const c_int,
        fired: *mut c_int, current_step: c_int, n: c_int) -> c_int;
    fn cuda_population_rate_decode(rates_out: *mut c_float, spikes: *const c_int,
        n_neurons: c_int, window: c_int) -> c_int;
    // Sync variants via GpuCtx (upload→kernel→sync in one call)
    fn gpu_lif_at_step(ctx: *mut GpuCtx, host_params: *const c_float, n: c_int) -> c_int;
    fn gpu_intrinsic_plasticity(ctx: *mut GpuCtx,
        lr_ip: c_float, target_rate: c_float, decay: c_float) -> c_int;
}

pub fn step_lif_at(v: &mut [f32], theta: &mut [f32], I: &mut [f32], spikes: &mut [i32],
    dt: f32, tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32],
    v_reset: &[f32], tau_theta: &[f32], beta: &[f32]) -> Result<(), String>
{
    let n = v.len();
    let mut flat = vec![0.0f32; 1 + 6*n];
    flat[0] = dt;
    for i in 0..n {
        flat[1+i] = tau_m[i];
        flat[1+n+i] = v_rest[i];
        flat[1+2*n+i] = v_thresh[i];
        flat[1+3*n+i] = v_reset[i];
        flat[1+4*n+i] = tau_theta[i];
        flat[1+5*n+i] = beta[i];
    }
    ok_or(unsafe {
        cuda_lif_at_step(v.as_mut_ptr(), theta.as_mut_ptr(), I.as_mut_ptr(),
            spikes.as_mut_ptr(), flat.as_ptr(), n as c_int)
    }, "lif_at")
}

pub fn snn_intrinsic_plasticity(theta: &mut [f32], spikes: &[i32],
    lr_ip: f32, target_rate: f32, decay: f32) -> Result<(), String>
{
    let n = theta.len() as c_int;
    ok_or(unsafe {
        cuda_intrinsic_plasticity(theta.as_mut_ptr(), spikes.as_ptr(),
            lr_ip, target_rate, decay, n)
    }, "intrinsic_plasticity")
}

pub fn snn_synaptic_normalization(weights: &mut [f32],
    n_pre: usize, n_post: usize, target_sum: f32, eps: f32) -> Result<(), String>
{
    ok_or(unsafe {
        cuda_synaptic_normalization(weights.as_mut_ptr(),
            n_pre as c_int, n_post as c_int, target_sum, eps)
    }, "synaptic_normalization")
}

pub fn snn_bsa_encode(spikes_out: &mut [i32], signal: &[f32],
    reconstructed: &[f32], threshold: f32) -> Result<(), String>
{
    let n = spikes_out.len().min(signal.len()).min(reconstructed.len()) as c_int;
    ok_or(unsafe {
        cuda_bsa_encode(spikes_out.as_mut_ptr(), signal.as_ptr(),
            reconstructed.as_ptr(), threshold, n)
    }, "bsa_encode")
}

pub fn snn_phase_encode(spikes_out: &mut [i32], phases: &mut [f32],
    cos_phases: &[f32], signal: &[f32], threshold: f32, gain: f32) -> Result<(), String>
{
    let n = spikes_out.len().min(phases.len()).min(cos_phases.len()).min(signal.len()) as c_int;
    ok_or(unsafe {
        cuda_phase_encode(spikes_out.as_mut_ptr(), phases.as_mut_ptr(),
            cos_phases.as_ptr(), signal.as_ptr(), threshold, gain, n)
    }, "phase_encode")
}

pub fn snn_ttfs_init(fire_times: &mut [i32], values: &[f32], max_delay: usize) -> Result<(), String> {
    let n = fire_times.len().min(values.len()) as c_int;
    ok_or(unsafe {
        cuda_ttfs_init(fire_times.as_mut_ptr(), values.as_ptr(), max_delay as c_int, n)
    }, "ttfs_init")
}

pub fn snn_ttfs_step(spikes_out: &mut [i32], fire_times: &[i32],
    fired: &mut [i32], current_step: usize) -> Result<(), String>
{
    let n = spikes_out.len().min(fire_times.len()).min(fired.len()) as c_int;
    ok_or(unsafe {
        cuda_ttfs_step(spikes_out.as_mut_ptr(), fire_times.as_ptr(),
            fired.as_mut_ptr(), current_step as c_int, n)
    }, "ttfs_step")
}

pub fn snn_population_rate_decode(rates_out: &mut [f32], spikes: &[i32],
    n_neurons: usize, window: usize) -> Result<(), String>
{
    ok_or(unsafe {
        cuda_population_rate_decode(rates_out.as_mut_ptr(), spikes.as_ptr(),
            n_neurons as c_int, window as c_int)
    }, "population_rate_decode")
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn require_gpu() { check_gpu!(); }

    #[test]
    fn test_device_count() {
        let count = get_device_count().unwrap_or(0);
        println!("CUDA devices: {count}");
        if count > 0 { assert!(set_device(0).is_ok()); }
    }

    #[test]
    fn test_gpu_context_create_destroy() {
        check_gpu!();
        let ctx = GpuContext::new(1024, 4096).unwrap();
        println!("GpuContext: {} neurons, {} synapses", ctx.n_neurons, ctx.n_synapses);
        // Drop frees everything
    }

    #[test]
    fn test_gpu_context_upload_and_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload neurons
        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Upload Kuramoto
        let phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

        // Run fused step
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();

        // Download results
        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s > 0).count();
        println!("Fused step: {n_spikes} spikes out of {n}");
    }

    #[test]
    fn test_gpu_context_fused_vs_separate() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Separate: izhikevich + stdp
        ctx.izhikevich_step(0.001, 0.99).unwrap();
        ctx.stdp_update(0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        let mut spikes_sep = vec![0i32; n];
        ctx.download_spikes(&mut spikes_sep).unwrap();

        // Re-upload state
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Fused: izhikevich + stdp in one launch
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        let mut spikes_fused = vec![0i32; n];
        ctx.download_spikes(&mut spikes_fused).unwrap();

        assert_eq!(spikes_sep, spikes_fused, "Fused and separate must match");
        println!("Fused vs separate: ✅ identical ({} spikes each)", spikes_sep.iter().filter(|&&s| s>0).count());
    }

    #[test]
    fn test_gpu_graph_simulation_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        let phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

        // Run through simulation_step which uses CUDA Graphs
        ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0).unwrap();

        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s > 0).count();
        println!("CUDA Graph simulation step: {n_spikes} spikes");
    }

    #[test]
    fn test_gpu_graph_replay_speed() {
        check_gpu!();
        let n = 10000;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // First call captures the graph
        ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0).unwrap();

        // Subsequent calls replay
        let n_iter = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            ctx.graph_replay().unwrap();
        }
        ctx.sync().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
        println!("CUDA Graph replay {n_iter}×: {ms:.4}ms per step (n={n})");
    }

    // ─── Legacy tests (unchanged) ───────────────────────

    #[test]
    fn test_xor_attention() {
        check_gpu!();
        let q = vec![0u8; 8];
        let mut k = vec![0xFFu8; 64];
        for i in 0..8 { k[i] = 0x00; }
        let v = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut o = vec![0.0f32];
        xor_attention(&q, &k, &v, &mut o, 8, 8).unwrap();
        println!("XOR attention: {:.6}", o[0]);
        assert!((o[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_safety_clamp() {
        check_gpu!();
        let mut data = vec![-10.0f32, -5.0, 0.0, 5.0, 10.0, -3.0, 3.0, 100.0];
        safety_clamp(&mut data, -1.0, 1.0).unwrap();
        for &v in &data { assert!(v >= -1.0 && v <= 1.0); }
        println!("safety_clamp: {:?}", data);
    }

    #[test]
    fn test_stdp_update() {
        check_gpu!();
        let n = 64;
        let mut w = vec![0.5f32; n];
        let tp = vec![1.0f32; n];
        let tn = vec![0.0f32; n];
        stdp_update(&mut w, &tp, &tn, 0.1, 0.1, 0.0, 1.0).unwrap();
        println!("STDP[0]: {:.4}", w[0]);
        assert!((w[0] - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_nlms_train() {
        check_gpu!();
        let n = 32;
        let mut w = vec![0.0f32; n];
        let mut bias = vec![0.0f32; 1];
        let s: Vec<f32> = (0..n).map(|i| (i+1) as f32 * 0.1).collect();
        nlms_train(&mut w, &mut bias, &s, 1.0, 0.01, 1e-8).unwrap();
        println!("NLMS[0]: {:.6}", w[0]);
        assert!(w.iter().all(|&x| x != 0.0), "All weights non-zero");
    }

    #[test]
    fn test_echo_tap() {
        check_gpu!();
        let n_osc = 8;
        let delay_len = 16;
        let mut buf = vec![0.0f32; n_osc * delay_len];
        for i in 0..n_osc { buf[i * delay_len + 3] = i as f32; }
        let mut out = vec![0.0f32; n_osc];
        echo_tap(&mut out, &buf, n_osc, delay_len, 3).unwrap();
        println!("echo_tap[0]: {:.4}", out[0]);
        assert!((out[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_kuramoto_prepare() {
        check_gpu!();
        let n = 64;
        let phases: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut cos_out = vec![0.0f32; n];
        let mut sin_out = vec![0.0f32; n];
        kuramoto_prepare(&phases, &mut cos_out, &mut sin_out).unwrap();
        println!("kuramoto_prepare[0]: cos={:.4} sin={:.4}", cos_out[0], sin_out[0]);
        assert!((cos_out[0] - 1.0).abs() < 0.01);
        assert!(sin_out[0].abs() < 0.01);
    }

    // ─── LIF-AT tests ────────────────────────────────────

    #[test]
    fn test_gpu_lif_at_basic_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload initial v at resting potential
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Inject strong current to drive spiking
        let I = vec![20.0f32; n];
        ctx.upload_I(&I).unwrap();

        // Run LIF-AT step
        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];
        ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta).unwrap();

        // Download spikes
        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s != 0).count();

        // Download v
        let mut v_out = vec![0.0f32; n];
        ctx.download_v(&mut v_out).unwrap();

        // With strong current, v should increase from -70
        let avg_v: f32 = v_out.iter().sum::<f32>() / n as f32;
        println!("LIF-AT basic step: avg v = {:.4}, spikes = {n_spikes}/{n}", avg_v);

        // At least some v increased (even if not all spiked yet)
        assert!(avg_v > -70.0, "v should increase from -70, got avg {avg_v}");
    }

    #[test]
    fn test_gpu_lif_at_spike_with_strong_current() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload state
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Very strong current to guarantee spikes within a few steps
        let I = vec![100.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];

        // Multiple steps to ensure spike
        let mut total_spikes = 0;
        for _step in 0..30 {
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta).unwrap();
            let mut spikes = vec![0i32; n];
            ctx.download_spikes(&mut spikes).unwrap();
            total_spikes += spikes.iter().filter(|&&s| s != 0).count();
        }

        println!("LIF-AT spike test: {total_spikes} total spikes over 30 steps");
        assert!(total_spikes > 0, "Expected at least 1 spike with I=100, got 0");
    }

    #[test]
    fn test_gpu_lif_at_no_spike_with_negative_current() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Inhibitory current — no spikes expected
        let I = vec![-10.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];

        let mut total_spikes = 0;
        for _step in 0..30 {
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta).unwrap();
            let mut spikes = vec![0i32; n];
            ctx.download_spikes(&mut spikes).unwrap();
            total_spikes += spikes.iter().filter(|&&s| s != 0).count();
        }

        println!("LIF-AT inhibitory test: {total_spikes} spikes (expected 0)");
        assert_eq!(total_spikes, 0, "Expected no spikes with I=-10, got {total_spikes}");
    }

    // ─── CPU vs CUDA cross-validation ───────────────────

    #[test]
    fn test_lif_at_cpu_gpu_identity() {
        check_gpu!();
        let raw_n = 64;
        let n = (raw_n + 7) / 8 * 8;  // pad for CPU ASM
        let ctx = GpuContext::new(n, n).unwrap();

        // Identical initial state for both backends
        let mut v_cpu: Vec<f32> = (0..n).map(|i| -70.0 + (i % 8) as f32 * 0.5).collect();
        let mut theta_cpu: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.3).collect();
        let mut I_cpu: Vec<f32> = (0..n).map(|i| 15.0 + (i % 7) as f32 * 2.0).collect();
        let mut spikes_cpu = vec![0i32; n];

        // Per-neuron params
        let tau_m: Vec<f32> = (0..n).map(|i| 10.0 + (i % 3) as f32).collect();
        let v_rest: Vec<f32> = (0..n).map(|_| -70.0).collect();
        let v_thresh: Vec<f32> = (0..n).map(|_| -55.0).collect();
        let v_reset: Vec<f32> = (0..n).map(|_| -75.0).collect();
        let tau_theta: Vec<f32> = (0..n).map(|i| 50.0 + (i % 4) as f32 * 10.0).collect();
        let beta: Vec<f32> = (0..n).map(|i| 5.0 + (i % 3) as f32).collect();
        let dt = 1.0;

        // Upload state to GPU (identical to CPU init)
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_cpu, &u, &a, &b, &c, &d).unwrap();
        ctx.upload_I(&I_cpu).unwrap();
        ctx.upload_theta(&theta_cpu).unwrap();

        // Run 10 steps on both backends
        for _step in 0..10 {
            // CPU step
            neuron_engine::neuron::step_lif_at(
                &mut v_cpu, &mut theta_cpu, &mut I_cpu, &mut spikes_cpu,
                dt, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta,
            );

            // GPU step
            ctx.lif_at_step(dt, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta).unwrap();

            // Download GPU results
            let mut v_gpu = vec![0.0f32; n];
            let mut theta_gpu = vec![0.0f32; n];
            let mut spikes_gpu = vec![0i32; n];
            ctx.download_v(&mut v_gpu).unwrap();
            ctx.download_theta(&mut theta_gpu).unwrap();
            ctx.download_spikes(&mut spikes_gpu).unwrap();

            // Compare (must be bit-identical within f32 precision)
            for i in 0..n {
                let v_diff = (v_cpu[i] - v_gpu[i]).abs();
                let th_diff = (theta_cpu[i] - theta_gpu[i]).abs();
                // CPU ASM stores -1 for spike (all-ones from vcmpps),
                // CUDA stores +1. Both use 0 for no spike.
                let sp_cpu = spikes_cpu[i] != 0;
                let sp_gpu = spikes_gpu[i] != 0;

                assert!(v_diff < 1e-5,
                    "Step {_step}, neuron {i}: v CPU={} GPU={}", v_cpu[i], v_gpu[i]);
                assert!(th_diff < 1e-5,
                    "Step {_step}, neuron {i}: theta CPU={} GPU={}", theta_cpu[i], theta_gpu[i]);
                assert_eq!(sp_cpu, sp_gpu,
                    "Step {_step}, neuron {i}: spike CPU={} GPU={}", spikes_cpu[i], spikes_gpu[i]);
            }
        }

        println!("CPU vs GPU LIF-AT: ✅ bit-identical across 10 steps (n={n})");
    }

    // ─── Intrinsic Plasticity CUDA test ──────────────────

    #[test]
    fn test_gpu_intrinsic_plasticity() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        // Set up: some spiked, some didn't
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Upload initial theta (all 1.0) and spikes
        let theta_init = vec![1.0f32; n];
        ctx.upload_theta(&theta_init).unwrap();

        // Inject current to generate spikes
        let I = vec![100.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![50.0f32; n];
        let beta = vec![5.0f32; n];

        // Run a few LIF-AT steps to generate spikes
        for _ in 0..5 {
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta).unwrap();
        }

        // Download theta before IP
        let mut theta_before = vec![0.0f32; n];
        ctx.download_theta(&mut theta_before).unwrap();

        // Apply Intrinsic Plasticity
        let lr_ip = 0.1;
        let target_rate = 0.5;
        let decay = 0.9;
        ctx.intrinsic_plasticity(lr_ip, target_rate, decay).unwrap();

        // Download theta after IP
        let mut theta_after = vec![0.0f32; n];
        ctx.download_theta(&mut theta_after).unwrap();

        // Verify theta changed
        let mut changed = false;
        for i in 0..n {
            // θ = θ + lr·(rate - target), θ *= decay
            // rate=1 if spiked, 0 if not
            let theta_manual = theta_before[i] + lr_ip * (1.0 - target_rate);
            let theta_manual = theta_manual * decay;
            let theta_manual = theta_manual.clamp(0.0, 50.0);
            let diff = (theta_after[i] - theta_manual).abs();
            assert!(diff < 1e-5,
                "IP mismatch neuron {i}: before={:.6} expected={:.6} got={:.6}",
                theta_before[i], theta_manual, theta_after[i]);
            if (theta_after[i] - theta_before[i]).abs() > 1e-6 {
                changed = true;
            }
        }

        assert!(changed, "IP should have changed at least one theta value");
        println!("GPU Intrinsic Plasticity: ✅ all thetas match expected values");
    }
}
