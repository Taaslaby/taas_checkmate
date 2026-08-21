//! Alpha-LIF (Shiu et al. 2024, fly brain) — full GPU worker.
//!
//! Mirrors the CPU path in `pillars-connectome` (`AlphaLifWorker` /
//! `BrainRunner::run_lif_alpha`), running the whole per-step loop on-device and
//! **capturing it into a CUDA graph** so each step is a single replay (no
//! per-kernel launch overhead):
//!
//! ```text
//! step t (captured graph, two variants for the spike double-buffer):
//!   1. Poisson inject      — v[grn] += weight  (schedule precomputed on host RNG)
//!   2. external inject     — v[i] += ext[i]     (guarded by a device flag)
//!   3. neuron step         — alphalif_step (same #[dual_neuron] source as CPU)
//!                             + zero the ring slot it just read + count spikes
//!   4. scatter             — delayed[post] = Σ w[pre→post]·spike[pre] (atomic push)
//!   5. advance             — ring_pos/step scalars bumped for the next replay
//! ```
//!
//! # Ordering (critical)
//!
//! The per-step scalars (`ring_pos`, `step`) live in a device `args` buffer and
//! are advanced **on-device** by a single-thread kernel at the end of each
//! graph — so the host replay loop is pure async `cuGraphLaunch` calls with no
//! per-step host↔device copies or syncs. Both graph variants are replayed on
//! the legacy NULL stream (FIFO-ordered), giving a strict per-step dependency
//! chain. The graphs are *captured* on a temporary non-NULL stream, destroyed
//! once both are instantiated.
//!
//! # Parity model
//!
//! - **Neuron** — bit-identical: reuses `ALPHA_LIF_CUDA_SOURCE` (proven
//!   `max bits=0` vs CPU SIMD in `zoo_parity::alpha_lif_parity`). NVRTC is
//!   compiled with `--fmad=false` so `a*b+c` stays a separate mul+add, matching
//!   the Rust `f32x8` path (no auto-FMA drift).
//! - **Poisson** — bit-identical: `rand::rngs::StdRng` (same seed `[0u8;32]`,
//!   same `rand 0.8`) drives the schedule on the host; flags are uploaded per
//!   chunk and applied on-device in the same consumption order as the CPU.
//! - **Scatter** — *atomic* (spike-driven push, `O(fired_edges)`), NOT
//!   bit-exact. A bit-exact gather is `O(E)` per step (read every edge every
//!   step) and was measured ~7× *slower* than the CPU. The reference GPU
//!   simulators (PyTorch / NEST / GeNN) use the same atomic accumulation, so
//!   results land in the published tolerance band rather than bit-matching our
//!   CPU serialization order.
//!
//! Buffers live in VRAM for the whole run — the only host↔device traffic is
//! the per-chunk Poisson flag upload + a final metrics readback.

use std::ffi::c_void;

use rand::{Rng, SeedableRng};

use crate::engine::{
    check_cu, compile_source_function, device_context, module_function, CUDA_CALL_LOCK,
};
use crate::ffi::{
    driver, CUdeviceptr, CUfunction, CUgraph, CUgraphExec, CUmodule, CUstream,
    CU_GRAPH_INSTANTIATE_DEFAULT, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_DEFAULT,
};
use pillars::zoo::neurons::ALPHA_LIF_CUDA_SOURCE;

const BLOCK: u32 = 256;

/// Number of state channels (`v`, `g`, `refr_cnt`) and param columns of
/// [`pillars::zoo::AlphaLifModel`].
const N_STATE: usize = 3;
const N_PARAM: usize = 7;

/// Device args layout. Indices 3 (`ring_pos`) and 4 (`step`) are advanced
/// on-device by the `alpha_lif_advance_kernel` at the end of each graph replay;
/// index 7 (`has_ext`) is set once per `run_steps` call; the rest are constant.
const ARG_DT: usize = 0;
const ARG_N: usize = 1;
const ARG_STRIDE: usize = 2;
const ARG_RING_POS: usize = 3;
const ARG_STEP: usize = 4;
const ARG_N_GRN: usize = 5;
const ARG_WEIGHT: usize = 6;
const ARG_HAS_EXT: usize = 7;
const ARG_SIZE: usize = 8;
const N_ARGS: usize = 9;

/// Auxiliary `extern "C"` kernels compiled alongside `ALPHA_LIF_CUDA_SOURCE`.
///
/// All kernels take a shared `const float* args` device buffer for the per-step
/// scalars, so a captured CUDA graph keeps FIXED pointers and re-reads the
/// scalars each replay (with `ring_pos`/`step` advanced on-device).
const AUX_SOURCE: &str = r#"
// ── neuron + count (wrapper around the generated alphalif_step) ──
// Reads the delayed input from the ring slot (ring_pos from args), zeroes that
// slot (so the scatter's atomicAdd starts clean), steps the neuron, and counts.
extern "C" __global__ void alpha_lif_runner_kernel(
    float* states, const float* params, float* ring, int* spikes, const float* args,
    unsigned char* active, int* total
) {
    const float dt = args[0];
    const int n = (int)args[1];
    const int stride = (int)args[2];
    const int ring_pos = (int)args[3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float S[3];
    #pragma unroll
    for (int k = 0; k < 3; k++) S[k] = states[k * stride + i];
    float P[7];
    #pragma unroll
    for (int k = 0; k < 7; k++) P[k] = params[k * stride + i];
    float* slot = ring + (size_t)ring_pos * (size_t)n;
    float delayed = slot[i];
    slot[i] = 0.0f;   // reset the slot for this step's scatter accumulation
    float spike = 0.0f;
    alphalif_step(S, P, delayed, dt, &spike);
    #pragma unroll
    for (int k = 0; k < 3; k++) states[k * stride + i] = S[k];
    int spk = (spike != 0.0f) ? 1 : 0;
    spikes[i] = spk;
    if (spk) { active[i] = 1; atomicAdd(total, 1); }
}

// ── spike-driven atomic scatter: delayed[post] += w[pre→post] ──
extern "C" __global__ void alpha_lif_scatter_kernel(
    const int* row_ptr, const int* col_indices, const float* values,
    const int* spikes, float* ring, const float* args
) {
    const int n = (int)args[1];
    const int ring_pos = (int)args[3];
    int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n) return;
    if (spikes[pre] == 0) return;
    float* slot = ring + (size_t)ring_pos * (size_t)n;
    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    for (int k = start; k < end; ++k) {
        atomicAdd(&slot[col_indices[k]], values[k]);
    }
}

// ── Poisson inject: v[grn] += weight for the flagged GRN this step ──
extern "C" __global__ void alpha_lif_poisson_kernel(
    float* states, const int* grn_idx, const unsigned char* fire_flags, const float* args
) {
    const int step = (int)args[4];
    const int n_grn = (int)args[5];
    const float weight = args[6];
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_grn) return;
    if (fire_flags[step * n_grn + g]) {
        states[grn_idx[g]] += weight;
    }
}

// ── External inject: v[i] += ext[i] (no-op when args[7] == 0) ──
extern "C" __global__ void alpha_lif_ext_kernel(float* states, const float* ext, const float* args) {
    if (args[7] == 0.0f) return;   // ext disabled for this run → skip entirely
    const int n = (int)args[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    states[i] += ext[i];
}

// ── advance: bump ring_pos / step for the next replay (single thread) ──
// Runs last in the graph (after every reader), so the next replay sees the
// updated scalars. `size` (ring slots) lives at args[8].
extern "C" __global__ void alpha_lif_advance_kernel(float* args) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int size = (int)args[8];
    int ring_pos = (int)args[3];
    args[3] = (float)((ring_pos + 1) % size);
    args[4] += 1.0f;
}
"#;

/// Model parameters of the Shiu fly-brain alpha-LIF (mirrors
/// `pillars_connectome::ShiuParams`, minus the Poisson rate/weight which are
/// passed to [`AlphaLifGpu::new`] separately).
#[derive(Debug, Clone, Copy)]
pub struct AlphaLifParams {
    pub dt_ms: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub tau_m: f32,
    pub tau_ref: f32,
    pub tau_syn: f32,
    pub t_delay: f32,
}

impl AlphaLifParams {
    /// The exact `Drosophila_brain_model` defaults (Sugar GRNs 200 Hz).
    pub fn flywire() -> Self {
        Self {
            dt_ms: 0.1,
            v_rest: -52.0,
            v_thresh: -45.0,
            v_reset: -52.0,
            tau_m: 20.0,
            tau_ref: 2.2,
            tau_syn: 5.0,
            t_delay: 1.8,
        }
    }
}

/// Result of a GPU run — same shape as `AlphaLifWorker`'s output.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlphaLifOutput {
    pub total_spikes: u64,
    pub active_neurons: u64,
    pub brain_ms: f32,
    pub wall_seconds: f64,
}

/// Persistent GPU worker for the alpha-LIF fly brain. Device buffers stay in
/// VRAM across `run_steps` calls (always-on worker, like the CPU version).
pub struct AlphaLifGpu {
    // kernels (one module, kept alive for the kernels' lifetime)
    _module: CUmodule,

    // device buffers
    d_states: CUdeviceptr,      // 3 * stride * 4
    d_params: CUdeviceptr,      // 7 * stride * 4
    d_spikes: [CUdeviceptr; 2], // double-buffered spikes (n * 4 each)
    d_ring: CUdeviceptr,        // size * n * 4
    d_args: CUdeviceptr,        // N_ARGS * 4
    d_row_ptr: CUdeviceptr,     // (n + 1) * 4
    d_col_indices: CUdeviceptr, // E * 4
    d_values: CUdeviceptr,      // E * 4
    d_grn_idx: CUdeviceptr,     // n_grn * 4
    d_fire_flags: CUdeviceptr,  // chunk * n_grn bytes
    d_ext: CUdeviceptr,         // n * 4
    d_active: CUdeviceptr,      // n bytes
    d_total: CUdeviceptr,       // 4 bytes

    // captured graphs (one per spike-buffer parity), replayed on the NULL stream
    graph_execs: [CUgraphExec; 2],

    // host-side geometry / state
    n: usize,
    stride: usize,
    size: usize, // ring slots = steps_delay + 1
    n_grn: usize,
    chunk_steps: usize,
    spike_cur: usize, // which spike buffer the runner writes next
    dt: f32,
    poisson_rate: f32,
    v_rest: f32,
    default_refr_steps: f32,

    // Poisson schedule driver (persistent across run_steps — same as CPU)
    rng: rand::rngs::StdRng,
    has_grn: bool,

    brain_ms: f32,
}

impl AlphaLifGpu {
    /// Build the worker from raw CSR connectome arrays + model params.
    ///
    /// `sugar_grn` are the Poisson-stimulated GRN (refractory disabled, jump
    /// `poisson_weight` into `v` at `poisson_rate` Hz). `row_ptr`/`col_indices`
    /// are the CSR (pre → post) graph; `values` are the signed edge weights
    /// (mV, already scaled — used verbatim in the scatter).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
        params: AlphaLifParams,
        sugar_grn: &[usize],
        poisson_rate: f32,
        poisson_weight: f32,
    ) -> Result<Self, String> {
        let n = row_ptr.len().checked_sub(1).ok_or("empty row_ptr")?;
        if n == 0 {
            return Err("AlphaLifGpu: connectome has 0 neurons".into());
        }
        if col_indices.len() != values.len() {
            return Err("AlphaLifGpu: col_indices/values length mismatch".into());
        }

        // stride = CPU NeuralPopulation capacity (round up to a multiple of 8).
        let stride = n.saturating_add(7) / 8 * 8;
        let steps_delay = (params.t_delay / params.dt_ms).round().max(0.0) as usize;
        let size = steps_delay + 1;
        let chunk_steps = ((100.0 / params.dt_ms).round() as usize).max(1);

        // Packed param columns: [dt, tf, v_rest, v_thresh, v_reset, refr_steps,
        // syn_decay] — broadcast-uniform except refr_steps (0 for sugar GRN).
        let tf = params.dt_ms / params.tau_m.max(1e-6);
        let syn_decay = 1.0 - params.dt_ms / params.tau_syn.max(1e-6);
        let refr_steps = (params.tau_ref / params.dt_ms).round().max(1.0);
        let mut flat_params = vec![0.0f32; N_PARAM * stride];
        let (dt_c, tf_c, vr_c, vt_c, vreset_c, refr_c, dec_c) = (0, 1, 2, 3, 4, 5, 6);
        for i in 0..n {
            flat_params[dt_c * stride + i] = params.dt_ms;
            flat_params[tf_c * stride + i] = tf;
            flat_params[vr_c * stride + i] = params.v_rest;
            flat_params[vt_c * stride + i] = params.v_thresh;
            flat_params[vreset_c * stride + i] = params.v_reset;
            flat_params[refr_c * stride + i] = refr_steps;
            flat_params[dec_c * stride + i] = syn_decay;
        }
        for &g in sugar_grn {
            if g < n {
                flat_params[refr_c * stride + g] = 0.0;
            }
        }

        // Initial state: v = v_rest, g = 0, refr_cnt = refr_steps (out of
        // refractory at start — matches alpha_lif_population). Column-major SoA:
        // col 0 = v (offset 0), col 1 = g (+stride), col 2 = refr_cnt (+2·stride).
        let mut flat_state = vec![0.0f32; N_STATE * stride];
        for i in 0..n {
            flat_state[i] = params.v_rest;
            flat_state[stride + i] = 0.0;
            flat_state[2 * stride + i] = refr_steps;
        }

        let grn_idx: Vec<i32> = sugar_grn.iter().map(|&g| g as i32).collect();
        let n_grn = grn_idx.len();

        // ── Compile one NVRTC module: neuron source + aux kernels ──
        let source = format!("{ALPHA_LIF_CUDA_SOURCE}\n\n{AUX_SOURCE}");
        let (module, runner_kernel) = compile_source_function(&source, "alpha_lif_runner_kernel")?;
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let scatter_kernel = module_function(module, "alpha_lif_scatter_kernel")?;
        let poisson_kernel = module_function(module, "alpha_lif_poisson_kernel")?;
        let ext_kernel = module_function(module, "alpha_lif_ext_kernel")?;
        let advance_kernel = module_function(module, "alpha_lif_advance_kernel")?;

        // ── Allocate + upload device buffers ──
        let mut args_host = [0.0f32; N_ARGS];
        args_host[ARG_DT] = params.dt_ms;
        args_host[ARG_N] = n as f32;
        args_host[ARG_STRIDE] = stride as f32;
        args_host[ARG_RING_POS] = (size - 1) as f32; // initial ring read slot
        args_host[ARG_STEP] = 0.0;
        args_host[ARG_N_GRN] = n_grn as f32;
        args_host[ARG_WEIGHT] = poisson_weight;
        args_host[ARG_HAS_EXT] = 0.0;
        args_host[ARG_SIZE] = size as f32;

        let grid = (n as u32).div_ceil(BLOCK);

        unsafe {
            let d_states = alloc(d, N_STATE * stride * 4)?;
            let d_params = alloc(d, N_PARAM * stride * 4)?;
            let d_spikes = [alloc(d, n * 4)?, alloc(d, n * 4)?];
            let d_ring = alloc(d, size * n * 4)?;
            let d_args = alloc(d, N_ARGS * 4)?;
            let d_row_ptr = alloc(d, (n + 1) * 4)?;
            let d_col_indices = alloc(d, col_indices.len() * 4)?;
            let d_values = alloc(d, values.len() * 4)?;
            let d_grn_idx = alloc(d, n_grn * 4)?;
            let d_fire_flags = alloc(d, chunk_steps * n_grn.max(1))?;
            let d_ext = alloc(d, n * 4)?;
            let d_active = alloc(d, n)?;
            let d_total = alloc(d, 4)?;

            upload(d, d_states, &flat_state)?;
            upload(d, d_params, &flat_params)?;
            // Zero the spike buffers: step 0's scatter reads the *previous*
            // step's buffer (d_spikes[1]), which must start empty.
            let zeros_spikes = vec![0i32; n];
            upload(d, d_spikes[0], &zeros_spikes)?;
            upload(d, d_spikes[1], &zeros_spikes)?;
            let zeros_ring = vec![0.0f32; size * n];
            upload(d, d_ring, &zeros_ring)?;
            upload(d, d_args, &args_host)?;
            upload(d, d_row_ptr, row_ptr)?;
            upload(d, d_col_indices, col_indices)?;
            upload(d, d_values, values)?;
            upload(d, d_grn_idx, &grn_idx)?;
            let zeros_ext = vec![0.0f32; n];
            upload(d, d_ext, &zeros_ext)?;

            // Capture the per-step sequence into two graphs (spike double-buffer
            // parity: graph[0] writes d_spikes[0] / reads d_spikes[1]; graph[1]
            // vice-versa). Both are captured on one temporary stream, then
            // replayed on the NULL stream (FIFO-ordered).
            let mut cap_stream: CUstream = std::ptr::null_mut();
            check_cu(d, (d.cu_stream_create)(&mut cap_stream, CU_STREAM_DEFAULT))
                .map_err(|e| format!("capture stream create: {e}"))?;

            let graph_execs = [
                capture_graph(
                    d,
                    cap_stream,
                    runner_kernel,
                    scatter_kernel,
                    poisson_kernel,
                    ext_kernel,
                    advance_kernel,
                    d_states,
                    d_params,
                    d_ring,
                    d_args,
                    d_active,
                    d_total,
                    d_row_ptr,
                    d_col_indices,
                    d_values,
                    d_grn_idx,
                    d_fire_flags,
                    d_ext,
                    d_spikes[0],
                    d_spikes[1],
                    grid,
                    n_grn > 0,
                )?,
                capture_graph(
                    d,
                    cap_stream,
                    runner_kernel,
                    scatter_kernel,
                    poisson_kernel,
                    ext_kernel,
                    advance_kernel,
                    d_states,
                    d_params,
                    d_ring,
                    d_args,
                    d_active,
                    d_total,
                    d_row_ptr,
                    d_col_indices,
                    d_values,
                    d_grn_idx,
                    d_fire_flags,
                    d_ext,
                    d_spikes[1],
                    d_spikes[0],
                    grid,
                    n_grn > 0,
                )?,
            ];
            let _ = (d.cu_stream_destroy)(cap_stream);

            Ok(Self {
                _module: module,
                d_states,
                d_params,
                d_spikes,
                d_ring,
                d_args,
                d_row_ptr,
                d_col_indices,
                d_values,
                d_grn_idx,
                d_fire_flags,
                d_ext,
                d_active,
                d_total,
                graph_execs,
                n,
                stride,
                size,
                n_grn,
                chunk_steps,
                spike_cur: 0,
                dt: params.dt_ms,
                poisson_rate,
                v_rest: params.v_rest,
                default_refr_steps: refr_steps,
                rng: rand::rngs::StdRng::from_seed([0u8; 32]),
                has_grn: n_grn > 0,
                brain_ms: 0.0,
            })
        }
    }

    /// Number of neurons.
    pub fn n_neurons(&self) -> usize {
        self.n
    }

    /// Run `steps` brain steps, optionally adding `external_i` to `v` every
    /// step (length `n` or empty). State persists between calls.
    pub fn run_steps(
        &mut self,
        steps: usize,
        external_i: &[f32],
    ) -> Result<AlphaLifOutput, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = std::time::Instant::now();

        unsafe {
            // Normalize + upload the external boost (or zeros), and set the
            // device flag the ext kernel checks. Done once per call.
            let has_ext = !external_i.is_empty();
            let ext_flag = [if has_ext { 1.0f32 } else { 0.0f32 }];
            upload(d, self.d_args + (ARG_HAS_EXT * 4) as CUdeviceptr, &ext_flag)?;
            let mut ext = vec![0.0f32; self.n];
            if has_ext {
                for (i, slot) in ext.iter_mut().enumerate() {
                    *slot = external_i.get(i).copied().unwrap_or(0.0);
                }
            }
            upload(d, self.d_ext, &ext)?;

            // Zero the run's metrics accumulators.
            upload(d, self.d_total, &[0i32; 1])?;
            let zeros = vec![0u8; self.n];
            upload(d, self.d_active, &zeros)?;

            let mut remaining = steps;
            while remaining > 0 {
                let chunk = remaining.min(self.chunk_steps);
                self.run_chunk(d, chunk)?;
                remaining -= chunk;
            }

            // Download metrics.
            let mut total = [0i32; 1];
            download(d, &mut total, self.d_total)?;
            let mut active = vec![0u8; self.n];
            download(d, &mut active, self.d_active)?;
            let active_count = active.iter().filter(|&&a| a != 0).count();

            let wall = t0.elapsed().as_secs_f64();
            let brain_ms = steps as f32 * self.dt;
            self.brain_ms += brain_ms;
            Ok(AlphaLifOutput {
                total_spikes: total[0].max(0) as u64,
                active_neurons: active_count as u64,
                brain_ms,
                wall_seconds: wall,
            })
        }
    }

    /// Run `seconds` seconds of brain time.
    pub fn run_seconds(&mut self, seconds: f32) -> Result<AlphaLifOutput, String> {
        let steps = (seconds * 1000.0 / self.dt).round() as usize;
        self.run_steps(steps, &[])
    }

    /// Download the final membrane potentials (for parity checks).
    pub fn download_v(&self) -> Result<Vec<f32>, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut v = vec![0.0f32; self.stride];
        unsafe {
            download(d, &mut v, self.d_states)?; // column 0 = v (offset 0)
        }
        v.truncate(self.n);
        Ok(v)
    }

    /// Reset state (v→v_rest, g=0, ring zeroed, ring pointer rewound).
    /// Mirrors `BrainRunner::reset`: RNG and GRN/poisson config persist, but
    /// the per-neuron refr_steps override is NOT re-applied (same contract).
    pub fn reset(&mut self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            let v_col = vec![self.v_rest; self.n];
            let g_col = vec![0.0f32; self.n];
            let refr_col = vec![self.default_refr_steps; self.n];
            upload(d, self.d_states, &v_col)?; // column 0 (v) at offset 0
            upload(d, self.d_states + (self.stride * 4) as CUdeviceptr, &g_col)?;
            upload(
                d,
                self.d_states + (2 * self.stride * 4) as CUdeviceptr,
                &refr_col,
            )?;
            let zeros_ring = vec![0.0f32; self.size * self.n];
            upload(d, self.d_ring, &zeros_ring)?;
            let zeros_spikes = vec![0i32; self.n];
            upload(d, self.d_spikes[0], &zeros_spikes)?;
            upload(d, self.d_spikes[1], &zeros_spikes)?;
            // Rewind the device-side per-step scalars.
            let ring_reset = [(self.size - 1) as f32];
            upload(
                d,
                self.d_args + (ARG_RING_POS * 4) as CUdeviceptr,
                &ring_reset,
            )?;
            let step_reset = [0.0f32];
            upload(d, self.d_args + (ARG_STEP * 4) as CUdeviceptr, &step_reset)?;
        }
        self.spike_cur = 0;
        self.brain_ms = 0.0;
        Ok(())
    }

    // ── internal ─────────────────────────────────────────────

    unsafe fn run_chunk(&mut self, d: &crate::ffi::Driver, chunk: usize) -> Result<(), String> {
        // Precompute Poisson flags for this chunk (host RNG, CPU-identical).
        let mut flags = vec![0u8; chunk * self.n_grn.max(1)];
        if self.has_grn {
            let p = self.poisson_rate * self.dt / 1000.0;
            let mut idx = 0usize;
            for _step in 0..chunk {
                for _g in 0..self.n_grn {
                    if self.rng.gen::<f32>() < p {
                        flags[idx] = 1;
                    }
                    idx += 1;
                }
            }
        }
        upload(d, self.d_fire_flags, &flags)?;

        // Reset the device step counter (flags are indexed within the chunk);
        // ring_pos is NOT reset — it is continuous across chunks.
        let step_reset = [0.0f32];
        upload(d, self.d_args + (ARG_STEP * 4) as CUdeviceptr, &step_reset)?;

        // Pure async replay loop: the advance kernel bumps ring_pos/step
        // on-device, so the host never stalls per step. Both graphs alternate
        // on the NULL stream (FIFO) → strict per-step ordering.
        for _ in 0..chunk {
            check_cu(
                d,
                (d.cu_graph_launch)(
                    self.graph_execs[self.spike_cur],
                    std::ptr::null_mut(), // legacy NULL stream (FIFO)
                ),
            )?;
            self.spike_cur = 1 - self.spike_cur;
        }

        // Hard boundary before the next chunk's flags/step upload.
        check_cu(d, (d.cu_ctx_synchronize)())
    }
}

impl Drop for AlphaLifGpu {
    fn drop(&mut self) {
        if let Ok(d) = driver().as_ref() {
            if device_context(d).is_ok() {
                let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    for &p in &[
                        self.d_states,
                        self.d_params,
                        self.d_spikes[0],
                        self.d_spikes[1],
                        self.d_ring,
                        self.d_args,
                        self.d_row_ptr,
                        self.d_col_indices,
                        self.d_values,
                        self.d_grn_idx,
                        self.d_fire_flags,
                        self.d_ext,
                        self.d_active,
                        self.d_total,
                    ] {
                        let _ = (d.cu_mem_free)(p);
                    }
                    for &exec in &self.graph_execs {
                        let _ = (d.cu_graph_exec_destroy)(exec);
                    }
                    let _ = (d.cu_module_unload)(self._module);
                }
            }
        }
    }
}

// ── CUDA graph capture ─────────────────────────────────────────────

/// Capture the per-step kernel sequence `[poisson?, ext?, runner, scatter,
/// advance]` into a CUDA graph on `stream`, using the fixed device pointers.
/// The per-step scalars (`ring_pos`, `step`) live in `d_args` and are advanced
/// on-device by the trailing `advance` kernel.
///
/// `spikes_write` is where the runner writes this step's spikes; `spikes_read`
/// is the *previous* step's spikes that the scatter consumes (the double-buffer
/// parity flips per step — hence two captured graphs).
///
/// The `ext` kernel is always captured; whether it does anything is controlled
/// by the `args[7]` flag at replay time. The returned `CUgraphExec` is replayed
/// on the NULL stream by the caller.
#[allow(clippy::too_many_arguments)]
unsafe fn capture_graph(
    d: &crate::ffi::Driver,
    stream: CUstream,
    runner_kernel: CUfunction,
    scatter_kernel: CUfunction,
    poisson_kernel: CUfunction,
    ext_kernel: CUfunction,
    advance_kernel: CUfunction,
    d_states: CUdeviceptr,
    d_params: CUdeviceptr,
    d_ring: CUdeviceptr,
    d_args: CUdeviceptr,
    d_active: CUdeviceptr,
    d_total: CUdeviceptr,
    d_row_ptr: CUdeviceptr,
    d_col_indices: CUdeviceptr,
    d_values: CUdeviceptr,
    d_grn_idx: CUdeviceptr,
    d_fire_flags: CUdeviceptr,
    d_ext: CUdeviceptr,
    spikes_write: CUdeviceptr,
    spikes_read: CUdeviceptr,
    grid: u32,
    has_grn: bool,
) -> Result<CUgraphExec, String> {
    if let Err(e) = check_cu(
        d,
        (d.cu_stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
    ) {
        return Err(format!("graph begin_capture: {e}"));
    }

    // 1. Poisson (1 block of 32 — n_grn <= 32 for sugar GRN).
    if has_grn {
        let mut st = d_states;
        let mut gi = d_grn_idx;
        let mut ff = d_fire_flags;
        let mut ar = d_args;
        let mut a: [*mut c_void; 4] = [
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut gi as *mut CUdeviceptr as *mut c_void,
            &mut ff as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            poisson_kernel,
            1,
            1,
            1,
            32,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture poisson: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 2. External inject (always captured; no-op when args[7] == 0).
    {
        let mut st = d_states;
        let mut ex = d_ext;
        let mut ar = d_args;
        let mut a: [*mut c_void; 3] = [
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut ex as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            ext_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture ext: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 3. Runner (neuron + zero slot + count).
    {
        let mut st = d_states;
        let mut pr = d_params;
        let mut rg = d_ring;
        let mut sp = spikes_write;
        let mut ar = d_args;
        let mut ac = d_active;
        let mut tt = d_total;
        let mut a: [*mut c_void; 7] = [
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut pr as *mut CUdeviceptr as *mut c_void,
            &mut rg as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
            &mut ac as *mut CUdeviceptr as *mut c_void,
            &mut tt as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            runner_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture runner: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 4. Scatter (atomic push) — reads previous spikes.
    {
        let mut rp = d_row_ptr;
        let mut ci = d_col_indices;
        let mut vl = d_values;
        let mut sp = spikes_read;
        let mut rg = d_ring;
        let mut ar = d_args;
        let mut a: [*mut c_void; 6] = [
            &mut rp as *mut CUdeviceptr as *mut c_void,
            &mut ci as *mut CUdeviceptr as *mut c_void,
            &mut vl as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut rg as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            scatter_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture scatter: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 5. Advance ring_pos/step for the next replay (single thread, runs last).
    {
        let mut ar = d_args;
        let mut a: [*mut c_void; 1] = [&mut ar as *mut CUdeviceptr as *mut c_void];
        let rc = (d.cu_launch_kernel)(
            advance_kernel,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture advance: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // End capture + instantiate.
    let mut graph: CUgraph = std::ptr::null_mut();
    if let Err(e) = check_cu(d, (d.cu_stream_end_capture)(stream, &mut graph)) {
        return Err(format!("graph end_capture: {e}"));
    }
    let mut exec: CUgraphExec = std::ptr::null_mut();
    let rc = (d.cu_graph_instantiate)(&mut exec, graph, CU_GRAPH_INSTANTIATE_DEFAULT);
    let _ = (d.cu_graph_destroy)(graph);
    if rc != 0 {
        return Err(format!(
            "graph instantiate: {}",
            crate::engine::driver_error(d, rc)
        ));
    }

    Ok(exec)
}

// ── helpers ────────────────────────────────────────────────────────

unsafe fn alloc(d: &crate::ffi::Driver, bytes: usize) -> Result<CUdeviceptr, String> {
    let mut p: CUdeviceptr = 0;
    check_cu(d, (d.cu_mem_alloc)(&mut p, bytes))?;
    Ok(p)
}

unsafe fn upload<T: Sized>(
    d: &crate::ffi::Driver,
    dst: CUdeviceptr,
    src: &[T],
) -> Result<(), String> {
    check_cu(
        d,
        (d.cu_memcpy_htod)(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
        ),
    )
}

unsafe fn download<T: Sized>(
    d: &crate::ffi::Driver,
    dst: &mut [T],
    src: CUdeviceptr,
) -> Result<(), String> {
    check_cu(
        d,
        (d.cu_memcpy_dtoh)(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
        ),
    )
}
