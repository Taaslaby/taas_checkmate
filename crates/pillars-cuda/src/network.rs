//! # CompiledNetwork — a full spiking network on the GPU
//!
//! The network counterpart of [`CompiledModel`](crate::engine::CompiledModel):
//! multiple **heterogeneous** populations + a CSR projection graph + axonal
//! delays, run entirely on-device. Mirrors `pillars::api::Network::step`
//! (read ring slot → step populations → zero slot → scatter spikes into future
//! slots → advance), so the same NeuroML document compiled to `Network` can run
//! on CPU SIMD **and** here.
//!
//! ```text
//! step s:
//!   1. copy      i_in[:] = ring[s%32][:]                              (device scalar cur_slot)
//!   2. per pop p:  launch {model}_kernel(i_in = i_in[offset..]) → spikes
//!   3. zero      ring[s%32][:] = 0                                    → freed for future writes
//!   4. scatter   ring[(s + max(delay,1)) % 32][post] += w             (atomic push)
//!   5. s += 1
//! ```
//!
//! Per-edge axonal delays are respected (unlike a uniform 1-step pipeline):
//! the scatter kernel reads each edge's delay and writes into the matching
//! future ring slot — bit-for-bit the same rule as
//! `pillars::api::push::push_spikes`.
//!
//! # Scope (documented honestly)
//!
//! * **heterogeneous** populations — each pop compiles its own
//!   `#[dual_neuron]` kernel (any model in `pillars::zoo::neurons`);
//! * **CUDA-graph capture** — a single graph is captured on first use and
//!   replayed thereafter. The only per-step variable (`step % 32`) lives in a
//!   device scalar updated before each replay (the same pattern
//!   `CompiledModel` uses for `dt`), so one graph is exact and there is no
//!   per-kernel launch overhead;
//! * **atomic scatter** (`O(fired_edges)`), not bit-exact vs the CPU's
//!   serialization order — same tolerance band as NEST/GeNN/PyTorch.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Mutex;

use crate::engine::{
    check_cu, compile_dual_kernel, compile_source_function, device_context, driver_error,
    module_function, CUDA_CALL_LOCK,
};
use crate::ffi::{
    driver, CUdeviceptr, CUfunction, CUgraph, CUgraphExec, CUmodule, CUstream,
    CU_GRAPH_INSTANTIATE_DEFAULT, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_DEFAULT,
};
use pillars::api::neuroml::LoadedNetwork;

const BLOCK: u32 = 256;
const MAX_DELAY_SLOTS: usize = 32;

/// STDP source: trace decay + CSR weight update.
/// Compiled once via NVRTC, used in the CUDA graph.
const STDP_SOURCE: &str = r#"
// Decay trace and increment on spike: trace = trace * decay + spike
extern "C" __global__ void trace_update_kernel(
    float* trace, const int* spikes, float decay, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    trace[i] = trace[i] * decay + (float)(spikes[i] != 0);
}

// CSR STDP: for each synapse (pre→post), update weight from traces.
// trace[pre] = pre-synaptic trace, trace[post] = post-synaptic trace.
extern "C" __global__ void csr_stdp_kernel(
    float* weights, const int* row_ptr, const int* col_indices,
    const float* trace, float lr_ltp, float lr_ltd,
    float w_min, float w_max, int n_neurons
) {
    int pre = blockIdx.x;
    if (pre >= n_neurons) return;
    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    float tp = trace[pre];
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        int post = col_indices[k];
        if (post < 0 || post >= n_neurons) continue;
        float tn = trace[post];
        float w = weights[k] + lr_ltp * tp - lr_ltd * tn;
        weights[k] = fminf(w_max, fmaxf(w_min, w));
    }
}
"#;

/// Auxiliary kernels compiled alongside the per-population `#[dual_neuron]`
/// sources. All of them are model-independent (they operate on the global
/// ring / input / CSR buffers) and are `extern "C"` (unmangled).
///
/// The only value that changes between steps is `*cur_slot` (`step % 32`),
/// which each kernel reads from the shared device scalar [`d_slot`]; that lets
/// a **single** captured graph replay the whole step for any slot.
const AUX_SOURCE: &str = r#"
// Copy the current ring slot into the stable per-step input buffer
// (the per-population kernels read a fixed `i_in` offset from it).
extern "C" __global__ void network_copy_slot_kernel(
    const float* ring, float* i_in, const int* cur_slot, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    i_in[i] = ring[(size_t)(*cur_slot) * n + i];
}

// Zero the current ring slot (freed for future writes).
extern "C" __global__ void network_zero_kernel(float* ring, const int* cur_slot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ring[(size_t)(*cur_slot) * n + i] = 0.0f;
}

// Spike-driven atomic scatter: delayed[post] += w[pre->post] into the slot
// matching each edge's axonal delay (delay 0 = next step, like the CPU push).
// `*cur_slot` is `step % slots`; the ring is circular, so `(slot + d) % slots`
// is exact and independent of the absolute step.
extern "C" __global__ void network_scatter_kernel(
    const int* row_ptr, const int* col_idx, const float* weights, const int* delays,
    const int* spikes, float* ring, const int* cur_slot, int n, int slots
) {
    int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n) return;
    if (spikes[pre] == 0) return;
    int slot = *cur_slot;
    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    for (int k = start; k < end; ++k) {
        int post = col_idx[k];
        int d = delays[k];
        if (d < 1) d = 1;
        int tgt = (slot + d) % slots;
        atomicAdd(&ring[(size_t)tgt * n + post], weights[k]);
    }
}

// External injection: slot[global_idx[i]] += values[i] (same-step arrival,
// mirroring Network::inject_current which adds into the current ring slot).
extern "C" __global__ void network_inject_kernel(float* slot, const int* indices, const float* values, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    slot[indices[i]] += values[i];
}
"#;

/// A single population to compile and run on-device.
///
/// `states` / `params` are **column-major SoA** with column stride
/// `n.div_ceil(8) * 8` (the same padding as `NeuralPopulation`), i.e.
/// `states[col * stride + i]`. `source` / `kernel_name` come from the model's
/// `#[dual_neuron]` constants (`LIF_CUDA_SOURCE`, `"lif_kernel"`, …).
pub struct NetworkPop {
    pub source: &'static str,
    pub kernel_name: &'static str,
    pub ns: usize,
    pub np: usize,
    pub n: usize,
    pub states: Vec<f32>,
    pub params: Vec<f32>,
}

/// Per-population device state (module/kernel + buffers).
struct PopDevice {
    module: CUmodule,
    func: CUfunction,
    d_states: CUdeviceptr,
    d_params: CUdeviceptr,
    /// 3-float scalar args `[dt, n, stride]` read by the generated kernel.
    d_args: CUdeviceptr,
    /// Global neuron offset of this population (for input/spike slicing).
    offset: usize,
    n: usize,
    state_bytes: usize,
}

/// Captured CUDA-graph state: one instantiated graph + the dedicated stream it
/// replays on. The graph reads the per-step `cur_slot` from a device scalar
/// (updated before each replay), so a single graph covers all 32 ring slots.
struct NetworkGraphState {
    exec: CUgraphExec,
    stream: CUstream,
}

/// A compiled spiking network: heterogeneous populations + CSR projections,
/// run on the GPU. State and buffers persist in VRAM across `step()` calls.
pub struct CompiledNetwork {
    pops: Vec<PopDevice>,
    d_ring: CUdeviceptr,    // MAX_DELAY_SLOTS * n_total * 4
    d_i_in: CUdeviceptr,    // n_total * 4 (stable per-step input buffer)
    d_slot: CUdeviceptr,    // 4 bytes: int `step % 32`, read by the graph kernels
    d_spikes: CUdeviceptr,  // n_total * 4
    d_row_ptr: CUdeviceptr, // (n_total + 1) * 4
    d_col_idx: CUdeviceptr, // n_syn * 4
    d_weights: CUdeviceptr, // n_syn * 4
    d_delays: CUdeviceptr,  // n_syn * 4
    copy_func: CUfunction,
    scatter_func: CUfunction,
    zero_func: CUfunction,
    inject_func: CUfunction,
    d_inj_idx: CUdeviceptr, // n_total * 4 (reused per inject_current call)
    d_inj_val: CUdeviceptr, // n_total * 4
    /// STDP: per-neuron trace buffer [n_total]. Decays each step, increments on spike.
    d_trace: CUdeviceptr, // n_total * 4
    /// STDP: pre-compiled trace update + STDP kernels.
    stdp_func: CUfunction,
    trace_decay_func: CUfunction,
    _aux_module: CUmodule,
    _stdp_module: CUmodule,
    n_total: usize,
    n_syn: usize,
    step: usize,
    /// Lazily captured on first [`CompiledNetwork::step_graph`] (one graph).
    /// `Mutex<Option<>>` guards the capture-once race.
    graph: Mutex<Option<NetworkGraphState>>,
}

impl CompiledNetwork {
    /// Build the network from population specs + a global CSR projection graph.
    ///
    /// * `row_ptr` — CSR row pointers, length `n_total + 1` (n_total = Σ pop n);
    /// * `col_idx` / `weights` / `delays` — per-synapse arrays (length n_syn);
    /// * `delays` — axonal delays in steps, `0..MAX_DELAY_SLOTS`.
    ///
    /// All populations' global indices are concatenated in `pops` order.
    pub fn new(
        pops: &[NetworkPop],
        row_ptr: &[u32],
        col_idx: &[u32],
        weights: &[f32],
        delays: &[u8],
        dt: f32,
    ) -> Result<Self, String> {
        let n_total: usize = pops.iter().map(|p| p.n).sum();
        if n_total == 0 {
            return Err("CompiledNetwork: no neurons".into());
        }
        if row_ptr.len() != n_total + 1 {
            return Err(format!(
                "CompiledNetwork: row_ptr len {} != n_total+1 {}",
                row_ptr.len(),
                n_total + 1
            ));
        }
        let n_syn = col_idx.len();
        if weights.len() != n_syn || delays.len() != n_syn {
            return Err("CompiledNetwork: col_idx/weights/delays length mismatch".into());
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err("CompiledNetwork: dt must be finite and > 0".into());
        }

        // ── 1. Compile every kernel FIRST (each locks CUDA_CALL_LOCK
        //       internally — must not be called while WE hold that lock). ──
        let mut compiled: Vec<(CUmodule, CUfunction)> = Vec::with_capacity(pops.len());
        for p in pops {
            if p.n == 0 {
                continue;
            }
            // The generated `#[dual_neuron]` kernels are C++-mangled, so use
            // the mangling-aware path (not compile_source_function, which
            // expects `extern "C"` kernels).
            compiled.push(compile_dual_kernel(p.source, p.kernel_name)?);
        }
        let (aux_module, scatter_func) =
            compile_source_function(AUX_SOURCE, "network_scatter_kernel")?;
        // STDP kernels (compiled before lock — same pattern as aux_module).
        let (stdp_module, trace_decay_func, stdp_func) = {
            let (mod_, trace_fn) = compile_source_function(STDP_SOURCE, "trace_update_kernel")?;
            let stdp_fn = module_function(mod_, "csr_stdp_kernel")?;
            (mod_, trace_fn, stdp_fn)
        };

        // ── 2. Context + lock for the alloc/upload/load section. ──
        //     Lock BEFORE `cuCtxSetCurrent`: the context is shared process-
        //     wide, and `cuCtxSetCurrent` / `cuGraphInstantiate` must not race
        //     across threads (parallel tests run several networks at once).
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        let copy_func = module_function(aux_module, "network_copy_slot_kernel")?;
        let zero_func = module_function(aux_module, "network_zero_kernel")?;
        let inject_func = module_function(aux_module, "network_inject_kernel")?;

        // ── 3. Allocate + upload each population. ──
        let mut pop_devs = Vec::with_capacity(pops.len());
        let mut offset = 0usize;
        let mut comp_iter = compiled.into_iter();
        for p in pops {
            if p.n == 0 {
                continue;
            }
            let (module, func) = comp_iter
                .next()
                .expect("compiled pop count must match non-empty pops");
            let stride = p.n.div_ceil(8) * 8;
            let state_len = p.ns * stride;
            let param_len = p.np * stride;
            if p.states.len() != state_len || p.params.len() != param_len {
                return Err(format!(
                    "CompiledNetwork: pop states/params len mismatch (expected {state_len}/{param_len}, got {}/{})",
                    p.states.len(),
                    p.params.len()
                ));
            }

            let d_states = alloc(d, state_len * 4)?;
            let d_params = alloc(d, param_len * 4)?;
            let d_args = alloc(d, 12)?;
            upload(d, d_states, &p.states)?;
            upload(d, d_params, &p.params)?;
            let args = [dt, p.n as f32, stride as f32];
            upload(d, d_args, &args)?;

            pop_devs.push(PopDevice {
                module,
                func,
                d_states,
                d_params,
                d_args,
                offset,
                n: p.n,
                state_bytes: state_len * 4,
            });
            offset += p.n;
        }

        // ── 4. Global buffers. ──
        let d_ring = alloc(d, MAX_DELAY_SLOTS * n_total * 4)?;
        let d_i_in = alloc(d, n_total * 4)?;
        let d_slot = alloc(d, 4)?;
        let d_spikes = alloc(d, n_total * 4)?;
        let d_row_ptr = alloc(d, (n_total + 1) * 4)?;
        let d_col_idx = alloc(d, n_syn * 4)?;
        let d_weights = alloc(d, n_syn * 4)?;
        let d_delays = alloc(d, n_syn * 4)?;
        let d_inj_idx = alloc(d, n_total * 4)?;
        let d_inj_val = alloc(d, n_total * 4)?;
        // STDP trace buffer: per-neuron trace, zeroed at start.
        let d_trace = alloc(d, n_total * 4)?;
        let zero_trace = vec![0.0f32; n_total];
        upload(d, d_trace, &zero_trace)?;

        // Zero ring + input + spikes + slot at start.
        let zero_ring = vec![0.0f32; MAX_DELAY_SLOTS * n_total];
        upload(d, d_ring, &zero_ring)?;
        let zero_in = vec![0.0f32; n_total];
        upload(d, d_i_in, &zero_in)?;
        let zero_slot = [0i32];
        upload(d, d_slot, &zero_slot)?;
        let zero_spikes = vec![0i32; n_total];
        upload(d, d_spikes, &zero_spikes)?;
        // Upload CSR (delays widened to i32 for the kernel).
        upload(d, d_row_ptr, row_ptr)?;
        upload(d, d_col_idx, col_idx)?;
        upload(d, d_weights, weights)?;
        let delays_i32: Vec<i32> = delays.iter().map(|&x| x as i32).collect();
        upload(d, d_delays, &delays_i32)?;

        Ok(Self {
            pops: pop_devs,
            d_ring,
            d_i_in,
            d_slot,
            d_spikes,
            d_row_ptr,
            d_col_idx,
            d_weights,
            d_delays,
            copy_func,
            scatter_func,
            zero_func,
            inject_func,
            d_inj_idx,
            d_inj_val,
            d_trace,
            stdp_func,
            trace_decay_func,
            _aux_module: aux_module,
            _stdp_module: stdp_module,
            n_total,
            n_syn,
            step: 0,
            graph: Mutex::new(None),
        })
    }

    /// Build the GPU network directly from a compiled NeuroML
    /// [`LoadedNetwork`]: the populations' states/params are read from the
    /// pillars [`Network`](pillars::api::Network) (column-major SoA), the CSR
    /// comes from the network's built graph, and each population's CUDA
    /// source/kernel is taken from the [`GpuPopMeta`] the NeuroML compiler
    /// recorded. This closes the NeuroML → CPU SIMD / GPU loop in one call.
    pub fn from_loaded(ln: &LoadedNetwork) -> Result<Self, String> {
        // `Network::state/param_slice` return the LOGICAL `n`-length columns;
        // the GPU SoA layout needs them padded to the SIMD stride (capacity =
        // n.div_ceil(8) * 8), exactly like `NeuralPopulation`'s internal layout.
        let mut pops = Vec::with_capacity(ln.gpu_pops.len());
        for (i, meta) in ln.gpu_pops.iter().enumerate() {
            let stride = meta.n.div_ceil(8) * 8;
            let mut states = Vec::with_capacity(meta.ns * stride);
            for c in 0..meta.ns {
                let col = ln.network.state(i, c);
                states.extend_from_slice(col);
                states.resize(states.len() + (stride - col.len()), 0.0);
            }
            let mut params = Vec::with_capacity(meta.np * stride);
            for c in 0..meta.np {
                let col = ln.network.param_slice(i, c);
                params.extend_from_slice(col);
                params.resize(params.len() + (stride - col.len()), 0.0);
            }
            pops.push(NetworkPop {
                source: meta.source,
                kernel_name: meta.kernel_name,
                ns: meta.ns,
                np: meta.np,
                n: meta.n,
                states,
                params,
            });
        }
        let g = ln
            .network
            .graph()
            .ok_or("CompiledNetwork::from_loaded: network not built")?;
        let n_syn = g.num_synapses();
        Self::new(
            &pops,
            g.row_ptr(),
            g.col_idx(),
            &g.weights()[..n_syn],
            g.delays(),
            ln.dt_ms,
        )
    }

    /// Total neuron count across all populations.
    pub fn n_total(&self) -> usize {
        self.n_total
    }

    /// Number of populations.
    pub fn n_pops(&self) -> usize {
        self.pops.len()
    }

    /// Number of neurons in population `pop`.
    pub fn pop_n(&self, pop: usize) -> usize {
        self.pops.get(pop).map_or(0, |p| p.n)
    }

    /// Advance one simulation step via a captured CUDA graph and sync.
    ///
    /// Semantically identical to the previous naive launch sequence, but the
    /// graph is captured once and replayed thereafter — no per-launch driver
    /// overhead. The whole capture/replay/sync cycle is serialized under
    /// [`CUDA_CALL_LOCK`] (same contract as `CompiledModel::step`). Blocks
    /// until the step completes (see [`CompiledNetwork::step_graph`] for the
    /// async variant).
    pub fn step(&mut self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        self.step_graph()?;
        unsafe { check_cu(d, (d.cu_ctx_synchronize)()) }
    }

    /// Graph-accelerated step (async): captures the graph on first use, then
    /// replays it with `cur_slot` set to the current step. Returns before the
    /// device finishes; call [`CompiledNetwork::sync`] (or
    /// [`CompiledNetwork::download_spikes`], which syncs via the next
    /// host↔device copy) before reading results.
    ///
    /// **Async caveat:** the per-step `cur_slot` lives in a single device
    /// scalar, so back-to-back async replays without an intervening sync would
    /// overwrite the slot before earlier replays read it. Use
    /// [`CompiledNetwork::step`] (sync) for the common case, or sync between
    /// async steps. Concurrent capture from multiple networks also requires
    /// caller-side serialization (one capture session per context).
    pub fn step_graph(&mut self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let slot = self.step % MAX_DELAY_SLOTS;
        unsafe {
            let (exec, stream) = {
                let mut guard = self
                    .graph
                    .lock()
                    .map_err(|e| format!("network graph lock: {e}"))?;
                if guard.is_none() {
                    *guard = Some(capture_network_graph(d, self)?);
                }
                let g = guard.as_ref().expect("graph set above");
                (g.exec, g.stream)
            };
            // Update the device `cur_slot` scalar before replay (the graph
            // reads it), then launch.
            let slot_i = slot as c_int;
            check_cu(
                d,
                (d.cu_memcpy_htod)(self.d_slot, &slot_i as *const c_int as *const c_void, 4),
            )
            .map_err(|e| format!("network step_graph slot upload: {e}"))?;
            check_cu(d, (d.cu_graph_launch)(exec, stream))
                .map_err(|e| format!("network step_graph launch: {e}"))?;
            self.step += 1;
            Ok(())
        }
    }

    /// Block until all queued device work (graph steps, copies) completes.
    pub fn sync(&self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        unsafe { check_cu(d, (d.cu_ctx_synchronize)()) }
    }

    /// Download the spike flags (length `n_total`).
    pub fn download_spikes(&self, out: &mut [i32]) -> Result<(), String> {
        assert_eq!(out.len(), self.n_total, "spikes len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        download(d, out, self.d_spikes)
    }

    /// Download a full state snapshot for one population (column-major SoA,
    /// length `ns * stride`).
    pub fn download_states(&self, pop: usize, out: &mut [f32]) -> Result<(), String> {
        let p = self
            .pops
            .get(pop)
            .ok_or_else(|| format!("CompiledNetwork: pop {pop} out of range"))?;
        assert_eq!(out.len(), p.state_bytes / 4, "states len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        download(d, out, p.d_states)
    }

    /// Full STDP step on GPU: trace → neurons → STDP → propagate.
    /// Everything stays on device — zero CPU downloads.
    pub fn stdp_step(
        &mut self,
        trace_decay: f32,
        lr_ltp: f32,
        lr_ltd: f32,
        w_min: f32,
        w_max: f32,
    ) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        let _slot = self.step % MAX_DELAY_SLOTS;

        // 1. Trace decay + spike increment
        {
            let grid = (self.n_total as u32).div_ceil(BLOCK).max(1);
            let decay = trace_decay;
            let n = self.n_total as c_int;
            let mut args: [*mut c_void; 4] = [
                &self.d_trace as *const _ as *mut c_void,
                &self.d_spikes as *const _ as *mut c_void,
                &decay as *const f32 as *mut c_void,
                &n as *const c_int as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        self.trace_decay_func,
                        grid,
                        1,
                        1,
                        BLOCK,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }

        // 2. Copy ring slot → input buffer
        {
            let grid = (self.n_total as u32).div_ceil(BLOCK);
            let mut ring = self.d_ring;
            let mut i_in = self.d_i_in;
            let mut cur_slot = self.d_slot;
            let mut n = self.n_total as c_int;
            let mut cargs: [*mut c_void; 4] = [
                &mut ring as *mut _ as *mut c_void,
                &mut i_in as *mut _ as *mut c_void,
                &mut cur_slot as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        self.copy_func,
                        grid,
                        1,
                        1,
                        BLOCK,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        cargs.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }

        // 3. Step each population
        for p in &self.pops {
            let grid = (p.n as u32).div_ceil(BLOCK);
            let mut states = p.d_states;
            let mut params = p.d_params;
            let mut i_in = self.d_i_in + (p.offset * 4) as u64;
            let mut spikes = self.d_spikes + (p.offset * 4) as u64;
            let mut args = p.d_args;
            let mut kargs: [*mut c_void; 5] = [
                &mut states as *mut _ as *mut c_void,
                &mut params as *mut _ as *mut c_void,
                &mut i_in as *mut _ as *mut c_void,
                &mut spikes as *mut _ as *mut c_void,
                &mut args as *mut _ as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        p.func,
                        grid,
                        1,
                        1,
                        BLOCK,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        kargs.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }

        // 4. STDP weight update from traces
        {
            let lr_l = lr_ltp;
            let lr_d = lr_ltd;
            let wmn = w_min;
            let wmx = w_max;
            let nn = self.n_total as c_int;
            let mut args: [*mut c_void; 9] = [
                &self.d_weights as *const _ as *mut c_void,
                &self.d_row_ptr as *const _ as *mut c_void,
                &self.d_col_idx as *const _ as *mut c_void,
                &self.d_trace as *const _ as *mut c_void,
                &lr_l as *const f32 as *mut c_void,
                &lr_d as *const f32 as *mut c_void,
                &wmn as *const f32 as *mut c_void,
                &wmx as *const f32 as *mut c_void,
                &nn as *const c_int as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        self.stdp_func,
                        self.n_total as u32,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }

        // 5. Zero slot + scatter spikes with plastic weights
        {
            let grid = (self.n_total as u32).div_ceil(BLOCK);
            let mut ring = self.d_ring;
            let mut cur_slot = self.d_slot;
            let mut n = self.n_total as c_int;
            let mut zargs: [*mut c_void; 3] = [
                &mut ring as *mut _ as *mut c_void,
                &mut cur_slot as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        self.zero_func,
                        grid,
                        1,
                        1,
                        BLOCK,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        zargs.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }
        if self.n_syn > 0 {
            let grid = (self.n_total as u32).div_ceil(BLOCK);
            let mut rp = self.d_row_ptr;
            let mut ci = self.d_col_idx;
            let mut w = self.d_weights;
            let mut dl = self.d_delays;
            let mut sp = self.d_spikes;
            let mut ring = self.d_ring;
            let mut cur_slot = self.d_slot;
            let mut n = self.n_total as c_int;
            let mut slots = MAX_DELAY_SLOTS as c_int;
            let mut sargs: [*mut c_void; 9] = [
                &mut rp as *mut _ as *mut c_void,
                &mut ci as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut dl as *mut _ as *mut c_void,
                &mut sp as *mut _ as *mut c_void,
                &mut ring as *mut _ as *mut c_void,
                &mut cur_slot as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut slots as *mut _ as *mut c_void,
            ];
            unsafe {
                check_cu(
                    d,
                    (d.cu_launch_kernel)(
                        self.scatter_func,
                        grid,
                        1,
                        1,
                        BLOCK,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        sargs.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )
            }?;
        }
        self.step += 1;
        Ok(())
    }

    /// Upload new synapse weights from CPU to GPU (for STDP/dopamine learning).
    pub fn upload_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        if weights.len() != self.n_syn {
            return Err(format!(
                "upload_weights: length {} != n_syn {}",
                weights.len(),
                self.n_syn
            ));
        }
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        upload(d, self.d_weights, weights)
    }

    /// Download current synapse weights from GPU to CPU.
    pub fn download_weights(&self, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.n_syn {
            return Err(format!(
                "download_weights: length {} != n_syn {}",
                out.len(),
                self.n_syn
            ));
        }
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        download(d, out, self.d_weights)
    }

    /// Inject external current into specific neurons of one population.
    ///
    /// Adds into the current ring slot (`step % 32`) — the same slot the next
    /// [`CompiledNetwork::step`] reads — so the current drives the population
    /// on the following step (same semantics as `Network::inject_current`).
    pub fn inject_current(
        &mut self,
        pop: usize,
        indices: &[u32],
        values: &[f32],
    ) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("inject_current: indices/values length mismatch".into());
        }
        if indices.len() > self.n_total {
            return Err("inject_current: too many indices".into());
        }
        let p = self
            .pops
            .get(pop)
            .ok_or_else(|| format!("inject_current: pop {pop} out of range"))?;

        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;

        let global: Vec<i32> = indices
            .iter()
            .map(|&i| {
                assert!((i as usize) < p.n, "inject_current: index {i} out of range");
                (p.offset + i as usize) as i32
            })
            .collect();

        unsafe {
            upload(d, self.d_inj_idx, &global)?;
            upload(d, self.d_inj_val, values)?;
            let slot = self.step % MAX_DELAY_SLOTS;
            let mut ring = self.d_ring + ((slot * self.n_total) * 4) as u64;
            let mut idx = self.d_inj_idx;
            let mut val = self.d_inj_val;
            let mut count = indices.len() as c_int;
            let grid = (indices.len() as u32).div_ceil(BLOCK).max(1);
            let mut iargs: [*mut c_void; 4] = [
                &mut ring as *mut CUdeviceptr as *mut c_void,
                &mut idx as *mut CUdeviceptr as *mut c_void,
                &mut val as *mut CUdeviceptr as *mut c_void,
                &mut count as *mut c_int as *mut c_void,
            ];
            check_cu(
                d,
                (d.cu_launch_kernel)(
                    self.inject_func,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    iargs.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )?;
            check_cu(d, (d.cu_ctx_synchronize)())
        }
    }

    /// Current simulation step.
    pub fn step_counter(&self) -> usize {
        self.step
    }

    /// Reset all populations to their initial states + clear the ring/spikes.
    ///
    /// `pops` must be the same specs passed to [`CompiledNetwork::new`].
    pub fn reset(&mut self, pops: &[NetworkPop]) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        device_context(d)?;
        for (dev, spec) in self.pops.iter().zip(pops.iter()) {
            upload(d, dev.d_states, &spec.states)?;
        }
        let zero_ring = vec![0.0f32; MAX_DELAY_SLOTS * self.n_total];
        upload(d, self.d_ring, &zero_ring)?;
        let zero_spikes = vec![0i32; self.n_total];
        upload(d, self.d_spikes, &zero_spikes)?;
        let zero_slot = [0i32];
        upload(d, self.d_slot, &zero_slot)?;
        self.step = 0;
        Ok(())
    }
}

impl Drop for CompiledNetwork {
    fn drop(&mut self) {
        if let Ok(d) = driver().as_ref() {
            if device_context(d).is_ok() {
                let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    if let Ok(mut guard) = self.graph.lock() {
                        if let Some(g) = guard.take() {
                            let _ = (d.cu_graph_exec_destroy)(g.exec);
                            let _ = (d.cu_stream_destroy)(g.stream);
                        }
                    }
                    for p in &self.pops {
                        let _ = (d.cu_mem_free)(p.d_states);
                        let _ = (d.cu_mem_free)(p.d_params);
                        let _ = (d.cu_mem_free)(p.d_args);
                        let _ = (d.cu_module_unload)(p.module);
                    }
                    for &ptr in &[
                        self.d_ring,
                        self.d_i_in,
                        self.d_slot,
                        self.d_spikes,
                        self.d_row_ptr,
                        self.d_col_idx,
                        self.d_weights,
                        self.d_delays,
                        self.d_inj_idx,
                        self.d_inj_val,
                    ] {
                        let _ = (d.cu_mem_free)(ptr);
                    }
                    let _ = (d.cu_module_unload)(self._aux_module);
                }
            }
        }
    }
}

// ── CUDA graph capture ─────────────────────────────────────────────

/// Launch the full per-step kernel sequence onto `stream` (the capture
/// primitive): copy the current slot into `i_in`, step each population, zero
/// the slot, then scatter spikes. Every device pointer is **stable** across
/// steps — only `*cur_slot` (in `net.d_slot`) varies, so the resulting single
/// graph replays the whole step for any slot.
///
/// # Safety
/// `stream` must be a valid stream in capture mode; all device pointers belong
/// to `net` and stay alive for the graph's lifetime.
unsafe fn launch_network_step(
    d: &crate::ffi::Driver,
    net: &CompiledNetwork,
    stream: CUstream,
) -> Result<(), String> {
    // 1. Copy the current ring slot into the stable input buffer.
    {
        let grid = (net.n_total as u32).div_ceil(BLOCK);
        let mut ring = net.d_ring;
        let mut i_in = net.d_i_in;
        let mut cur_slot = net.d_slot;
        let mut n = net.n_total as c_int;
        let mut cargs: [*mut c_void; 4] = [
            &mut ring as *mut CUdeviceptr as *mut c_void,
            &mut i_in as *mut CUdeviceptr as *mut c_void,
            &mut cur_slot as *mut CUdeviceptr as *mut c_void,
            &mut n as *mut c_int as *mut c_void,
        ];
        check_cu(
            d,
            (d.cu_launch_kernel)(
                net.copy_func,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                cargs.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )?;
    }

    // 2. Step each population, reading its slice of the stable input buffer.
    for pop in &net.pops {
        let grid = (pop.n as u32).div_ceil(BLOCK);
        let mut states = pop.d_states;
        let mut params = pop.d_params;
        let mut i_in = net.d_i_in + (pop.offset * 4) as u64;
        let mut spikes = net.d_spikes + (pop.offset * 4) as u64;
        let mut args = pop.d_args;
        let mut kargs: [*mut c_void; 5] = [
            &mut states as *mut CUdeviceptr as *mut c_void,
            &mut params as *mut CUdeviceptr as *mut c_void,
            &mut i_in as *mut CUdeviceptr as *mut c_void,
            &mut spikes as *mut CUdeviceptr as *mut c_void,
            &mut args as *mut CUdeviceptr as *mut c_void,
        ];
        check_cu(
            d,
            (d.cu_launch_kernel)(
                pop.func,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                kargs.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )?;
    }

    // 3. Zero the slot that was just read.
    {
        let grid = (net.n_total as u32).div_ceil(BLOCK);
        let mut ring = net.d_ring;
        let mut cur_slot = net.d_slot;
        let mut n = net.n_total as c_int;
        let mut zargs: [*mut c_void; 3] = [
            &mut ring as *mut CUdeviceptr as *mut c_void,
            &mut cur_slot as *mut CUdeviceptr as *mut c_void,
            &mut n as *mut c_int as *mut c_void,
        ];
        check_cu(
            d,
            (d.cu_launch_kernel)(
                net.zero_func,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                zargs.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )?;
    }

    // 4. Scatter spikes through the CSR into future slots.
    if net.n_syn > 0 {
        let grid = (net.n_total as u32).div_ceil(BLOCK);
        let mut rp = net.d_row_ptr;
        let mut ci = net.d_col_idx;
        let mut w = net.d_weights;
        let mut dl = net.d_delays;
        let mut sp = net.d_spikes;
        let mut ring = net.d_ring;
        let mut cur_slot = net.d_slot;
        let mut n = net.n_total as c_int;
        let mut slots = MAX_DELAY_SLOTS as c_int;
        let mut sargs: [*mut c_void; 9] = [
            &mut rp as *mut CUdeviceptr as *mut c_void,
            &mut ci as *mut CUdeviceptr as *mut c_void,
            &mut w as *mut CUdeviceptr as *mut c_void,
            &mut dl as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut ring as *mut CUdeviceptr as *mut c_void,
            &mut cur_slot as *mut CUdeviceptr as *mut c_void,
            &mut n as *mut c_int as *mut c_void,
            &mut slots as *mut c_int as *mut c_void,
        ];
        check_cu(
            d,
            (d.cu_launch_kernel)(
                net.scatter_func,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                sargs.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )?;
    }

    Ok(())
}

/// Capture the single per-step graph and instantiate it.
///
/// # Safety
/// `net` must be fully uploaded (states/params/CSR/ring) before capture; the
/// graph holds pointers into `net`'s buffers, which stay alive for the graph's
/// lifetime. No other thread may be capturing on this context concurrently.
unsafe fn capture_network_graph(
    d: &crate::ffi::Driver,
    net: &CompiledNetwork,
) -> Result<NetworkGraphState, String> {
    let mut stream: CUstream = std::ptr::null_mut();
    check_cu(d, (d.cu_stream_create)(&mut stream, CU_STREAM_DEFAULT))
        .map_err(|e| format!("network graph stream create: {e}"))?;

    if let Err(e) = check_cu(
        d,
        (d.cu_stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
    ) {
        let _ = (d.cu_stream_destroy)(stream);
        return Err(format!("network graph begin_capture: {e}"));
    }
    if let Err(e) = launch_network_step(d, net, stream) {
        let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
        let _ = (d.cu_stream_destroy)(stream);
        return Err(e);
    }
    let mut graph: CUgraph = std::ptr::null_mut();
    if let Err(e) = check_cu(d, (d.cu_stream_end_capture)(stream, &mut graph)) {
        let _ = (d.cu_stream_destroy)(stream);
        return Err(format!("network graph end_capture: {e}"));
    }
    let mut exec: CUgraphExec = std::ptr::null_mut();
    let rc = (d.cu_graph_instantiate)(&mut exec, graph, CU_GRAPH_INSTANTIATE_DEFAULT);
    let _ = (d.cu_graph_destroy)(graph);
    if rc != 0 {
        let _ = (d.cu_stream_destroy)(stream);
        return Err(format!(
            "network graph instantiate: {}",
            driver_error(d, rc)
        ));
    }

    Ok(NetworkGraphState { exec, stream })
}

// ── helpers ────────────────────────────────────────────────────────

fn alloc(d: &crate::ffi::Driver, bytes: usize) -> Result<CUdeviceptr, String> {
    let mut p: CUdeviceptr = 0;
    // SAFETY: `p` is a valid out-pointer to a fresh CUdeviceptr; the FFI call
    // writes the allocated device pointer into it.
    check_cu(d, unsafe { (d.cu_mem_alloc)(&mut p, bytes) })?;
    Ok(p)
}

fn upload<T: Sized>(d: &crate::ffi::Driver, dst: CUdeviceptr, src: &[T]) -> Result<(), String> {
    // SAFETY: `dst` is a device pointer with at least `size_of_val(src)` bytes
    // allocated; `src` is a valid host slice of the same length.
    check_cu(d, unsafe {
        (d.cu_memcpy_htod)(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
        )
    })
}

fn download<T: Sized>(
    d: &crate::ffi::Driver,
    dst: &mut [T],
    src: CUdeviceptr,
) -> Result<(), String> {
    // SAFETY: `src` is a device pointer with at least `size_of_val(dst)` bytes;
    // `dst` is a valid writable host slice of the same length.
    check_cu(d, unsafe {
        (d.cu_memcpy_dtoh)(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
        )
    })
}
