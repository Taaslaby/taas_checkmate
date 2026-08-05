//! NVRTC engine — compiles the `#[dual_neuron]` C++ string at runtime and
//! drives it against column-major SoA buffers (same layout as
//! `pillars::NeuralPopulation`).
//!
//! ```text
//! #[dual_neuron] source  ──►  NVRTC (nvrtcCompileProgram)
//!         │                        │  PTX
//!    CPU SIMD (pillars)            ▼
//!    unchanged                 cuModuleLoadDataEx → cuLaunchKernel
//! ```

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Mutex;

use crate::ffi::*;

const BLOCK: u32 = 256;

/// CUDA source for the fixed (model-independent) spike-compaction kernel:
/// each active spike (`spikes[i] != 0`) atomically claims a slot in
/// `compacted` and stores its index. `counter` (device int, zeroed per step)
/// doubles as the output length — we then download only `counter` ints
/// instead of the whole `n`-element spike array.
const COMPACT_CUDA_SOURCE: &str = r#"
__device__ __forceinline__ int atomic_add_int(int* ptr, int val) {
    return atomicAdd(ptr, val);
}
__global__ void compact_spikes_kernel(const int* spikes, int* compacted, int* counter, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (spikes[i] != 0) {
        int slot = atomic_add_int(counter, 1);
        compacted[slot] = i;
    }
}
"#;

/// Compile a CUDA C++ source string (from `#[dual_neuron]`) and load the
/// named kernel. Holds the kernel handle + device buffers. All models share
/// one process-global CUDA context (see [`global_ctx`]).
/// Optional CUDA Graph state: the kernel launch is captured into a graph on
/// first use and replayed on every subsequent step — eliminating per-launch
/// driver overhead (`cuLaunchKernel` setup) and allowing async pipelining.
struct GraphState {
    exec: CUgraphExec,
    stream: CUstream,
    /// Device buffer holding the per-step scalar kernel args `[dt, n, stride]`
    /// (f32, i32, i32) — the graph is instantiated with a pointer to this
    /// buffer, and we update `dt` in it via `cuMemcpyHtoD` before each replay.
    d_args: CUdeviceptr,
}

pub struct CompiledModel {
    func: CUfunction,
    compact_func: CUfunction,
    d_states: CUdeviceptr,
    d_params: CUdeviceptr,
    d_i_in: CUdeviceptr,
    d_spikes: CUdeviceptr,
    /// Compaction buffers: d_compacted (n ints), d_counter (1 int).
    d_compacted: CUdeviceptr,
    d_counter: CUdeviceptr,
    n: usize,
    stride: usize,
    bytes_state: usize,
    bytes_params: usize,
    /// Lazily built on first `step_graph` call (capture-once, replay-after).
    /// `OnceLock` gives interior mutability so graph capture works through
    /// `&self` (the kernel reads device memory; no host state is mutated by
    /// replays).
    graph: OnceLock<GraphState>,
}

impl CompiledModel {
    /// Compile `source` (as emitted by `#[dual_neuron]`), load kernel
    /// `kernel_name` and allocate device buffers for `n` neurons with
    /// `ns` state / `np` param columns (column stride = `stride`).
    pub fn new(
        source: &str,
        kernel_name: &str,
        ns: usize,
        np: usize,
        n: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            let (ctx, major, minor) = *global_ctx().as_ref().map_err(|e| e.clone())?;
            let ctx: CUcontext = ctx.0;
            // Try the device's native PTX ISA first; fall back to compute_90
            // if this NVRTC doesn't know the newest architectures.
            let func = match compile_and_load(d, ctx, source, kernel_name, major, minor) {
                Ok(f) => f,
                Err(native_err) => {
                    compile_and_load(d, ctx, source, kernel_name, 90, 0).map_err(|fb_err| {
                        format!("{native_err}\nfallback compute_90 failed: {fb_err}")
                    })?
                }
            };

            let bytes_state = ns * stride * 4;
            let bytes_params = np * stride * 4;
            let ptrs = alloc_all(d, &[bytes_state, bytes_params, n * 4, n * 4, n * 4, 4])?;
            let (d_states, d_params, d_i_in, d_spikes, d_compacted, d_counter) =
                (ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5]);

            // Compile the fixed compaction kernel (same context).
            let compact_func = match compile_and_load(
                d,
                ctx,
                COMPACT_CUDA_SOURCE,
                "compact_spikes_kernel",
                major,
                minor,
            ) {
                Ok(f) => f,
                Err(native_err) => {
                    compile_and_load(d, ctx, COMPACT_CUDA_SOURCE, "compact_spikes_kernel", 90, 0)
                        .map_err(|fb_err| {
                            format!("{native_err}\nfallback compute_90 failed: {fb_err}")
                        })?
                }
            };

            Ok(Self {
                func,
                compact_func,
                d_states,
                d_params,
                d_i_in,
                d_spikes,
                d_compacted,
                d_counter,
                n,
                stride,
                bytes_state,
                bytes_params,
                graph: OnceLock::new(),
            })
        }
    }

    // ── Convenience: one-shot step (upload → kernel → download) ──────

    /// Run one simulation step: uploads states/params/current, launches the
    /// kernel, syncs, downloads states + spikes.
    ///
    /// Buffers are column-major SoA: `states[col * stride + i]`,
    /// `params[col * stride + i]` — exactly `NeuralPopulation`'s layout.
    ///
    /// This is the *naive* path (host↔device round-trip every step). For
    /// performance use the persistent API: [`CompiledModel::upload`] once,
    /// then [`CompiledModel::step_device`] + [`CompiledModel::download_spikes`].
    pub fn step(
        &self,
        states: &mut [f32],
        params: &[f32],
        i_in: &[f32],
        spikes: &mut [i32],
        dt: f32,
    ) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        // One lock for the WHOLE step: CUDA allows only one stream-capture
        // session per context, and any legacy-stream op (memcpy, sync) issued
        // during another thread's capture fails. Serializing the full
        // upload→capture→launch→download cycle makes concurrent `step()`
        // calls safe. (The persistent API below is single-threaded per model
        // — benchmarks run one model at a time.)
        let _lock = STEP_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        self.upload(states, params, i_in)?;
        self.step_device(dt)?;
        self.download_states(states)?;
        self.download_spikes(spikes)
    }

    // ── Persistent path: state/params live in VRAM, only spikes cross ──

    /// Upload states, params and input current to device memory.
    ///
    /// Call once before a persistent loop. Column-major SoA layout.
    pub fn upload(&self, states: &[f32], params: &[f32], i_in: &[f32]) -> Result<(), String> {
        assert_eq!(states.len(), self.bytes_state / 4, "states len");
        assert_eq!(params.len(), self.bytes_params / 4, "params len");
        assert_eq!(i_in.len(), self.n, "i_in len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            check_cu(
                d,
                (d.cu_memcpy_htod)(
                    self.d_states,
                    states.as_ptr() as *const c_void,
                    self.bytes_state,
                ),
            )?;
            check_cu(
                d,
                (d.cu_memcpy_htod)(
                    self.d_params,
                    params.as_ptr() as *const c_void,
                    self.bytes_params,
                ),
            )?;
            check_cu(
                d,
                (d.cu_memcpy_htod)(self.d_i_in, i_in.as_ptr() as *const c_void, self.n * 4),
            )
        }
    }

    /// Upload only the input current (for per-step changing inputs).
    pub fn upload_i_in(&self, i_in: &[f32]) -> Result<(), String> {
        assert_eq!(i_in.len(), self.n, "i_in len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            check_cu(
                d,
                (d.cu_memcpy_htod)(self.d_i_in, i_in.as_ptr() as *const c_void, self.n * 4),
            )
        }
    }

    /// Launch the kernel and sync — **state stays in VRAM** (no copies).
    ///
    /// Pairs with [`CompiledModel::upload`] (once) and
    /// [`CompiledModel::download_spikes`] / [`CompiledModel::download_states`].
    pub fn step_device(&self, dt: f32) -> Result<(), String> {
        self.step_graph(dt)?;
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe { check_cu(d, (d.cu_ctx_synchronize)()) }
    }

    /// CUDA-Graph-accelerated step: launch via a captured graph (first call
    /// captures the kernel, subsequent calls replay it — no per-launch driver
    /// overhead). **Async**: returns before the kernel finishes; call
    /// [`CompiledModel::sync`] (or `download_spikes`) before reading results.
    ///
    /// `dt` is copied into a device args buffer each call (the graph itself
    /// is instantiated with a pointer to that buffer), so changing `dt`
    /// between steps is supported.
    pub fn step_graph(&self, dt: f32) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        // NOTE: the convenience `step()` serializes the full
        // upload→capture→launch→download cycle with `STEP_LOCK`. The
        // persistent path (upload/step_device/download_*) does NOT take the
        // lock — CUDA allows only ONE capture session per context, so
        // concurrent use of the persistent API from multiple threads requires
        // caller-side serialization (captures may otherwise race).
        unsafe {
            // Capture once, replay after. `OnceLock::get_or_try_init` is
            // unstable, so do the set-if-empty dance by hand. If capture
            // fails, the lock stays empty and the next call retries.
            if self.graph.get().is_none() {
                let g = capture_graph(d, self, dt)?;
                let _ = self.graph.set(g);
            }
            let graph = self.graph.get().expect("graph set above");
            // Update dt in the device args buffer, then replay. The graph was
            // instantiated with a pointer to `d_args`; the kernel reads the
            // scalars from device memory, so a fresh dt takes effect without
            // re-instantiating.
            let dt_arr = [dt, self.n as f32, self.stride as f32];
            check_cu(
                d,
                (d.cu_memcpy_htod)(graph.d_args, dt_arr.as_ptr() as *const c_void, 12),
            )
            .map_err(|e| format!("step_graph args upload: {e}"))?;
            check_cu(d, (d.cu_graph_launch)(graph.exec, graph.stream))
                .map_err(|e| format!("step_graph launch: {e}"))
        }
    }

    /// Block until all queued device work (graph steps, copies) completes.
    pub fn sync(&self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe { check_cu(d, (d.cu_ctx_synchronize)()) }
    }

    /// Compact the spike buffer on-device into a dense index list.
    ///
    /// Call after the model step (and any sync). Returns the number of active
    /// spikes; download them with [`CompiledModel::download_compact`].
    pub fn compact_spikes(&self) -> Result<u32, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            // Reset the counter to 0.
            let zero = 0i32;
            check_cu(
                d,
                (d.cu_memcpy_htod)(self.d_counter, &zero as *const i32 as *const c_void, 4),
            )
            .map_err(|e| format!("compact counter reset: {e}"))?;

            // Launch the compaction kernel.
            let mut spikes_ptr = self.d_spikes;
            let mut compacted_ptr = self.d_compacted;
            let mut counter_ptr = self.d_counter;
            let mut n_v = self.n as c_int;
            let mut args: [*mut c_void; 4] = [
                &mut spikes_ptr as *mut CUdeviceptr as *mut c_void,
                &mut compacted_ptr as *mut CUdeviceptr as *mut c_void,
                &mut counter_ptr as *mut CUdeviceptr as *mut c_void,
                &mut n_v as *mut c_int as *mut c_void,
            ];
            let grid = (self.n as u32).div_ceil(BLOCK);
            check_cu(
                d,
                (d.cu_launch_kernel)(
                    self.compact_func,
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
            .map_err(|e| format!("compact launch: {e}"))?;
            check_cu(d, (d.cu_ctx_synchronize)()).map_err(|e| format!("compact sync: {e}"))?;

            // Read back the count.
            let mut count = 0i32;
            check_cu(
                d,
                (d.cu_memcpy_dtoh)(&mut count as *mut i32 as *mut c_void, self.d_counter, 4),
            )
            .map_err(|e| format!("compact count readback: {e}"))?;
            Ok(count.max(0) as u32)
        }
    }

    /// Download the compacted spike indices (exactly `count` ints, from
    /// [`CompiledModel::compact_spikes`]).
    pub fn download_compact(&self, indices: &mut [i32], count: u32) -> Result<(), String> {
        assert!(indices.len() as u32 >= count, "indices buffer too small");
        if count == 0 {
            return Ok(());
        }
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            check_cu(
                d,
                (d.cu_memcpy_dtoh)(
                    indices.as_mut_ptr() as *mut c_void,
                    self.d_compacted,
                    count as usize * 4,
                ),
            )
            .map_err(|e| format!("compact download: {e}"))
        }
    }

    /// Download spikes (device → host).
    pub fn download_spikes(&self, spikes: &mut [i32]) -> Result<(), String> {
        assert_eq!(spikes.len(), self.n, "spikes len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            check_cu(
                d,
                (d.cu_memcpy_dtoh)(
                    spikes.as_mut_ptr() as *mut c_void,
                    self.d_spikes,
                    self.n * 4,
                ),
            )
        }
    }

    /// Download the full state snapshot (device → host).
    pub fn download_states(&self, states: &mut [f32]) -> Result<(), String> {
        assert_eq!(states.len(), self.bytes_state / 4, "states len");
        let d = driver().as_ref().map_err(|e| e.clone())?;
        set_current_ctx(d)?;
        unsafe {
            check_cu(
                d,
                (d.cu_memcpy_dtoh)(
                    states.as_ptr() as *mut c_void,
                    self.d_states,
                    self.bytes_state,
                ),
            )
        }
    }

    /// Run `n_steps` identical time steps (constant input current).
    pub fn run(
        &self,
        states: &mut [f32],
        params: &[f32],
        i_in: &[f32],
        spikes: &mut [i32],
        dt: f32,
        n_steps: usize,
    ) -> Result<(), String> {
        for _ in 0..n_steps {
            self.step(states, params, i_in, spikes, dt)?;
        }
        Ok(())
    }
}

impl Drop for CompiledModel {
    fn drop(&mut self) {
        if let Ok(d) = driver().as_ref() {
            // Best effort: if the context can't be made current on this
            // thread, the frees would fail anyway — skip them silently.
            if set_current_ctx(d).is_ok() {
                let _lock = STEP_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    let _ = (d.cu_mem_free)(self.d_states);
                    let _ = (d.cu_mem_free)(self.d_params);
                    let _ = (d.cu_mem_free)(self.d_i_in);
                    let _ = (d.cu_mem_free)(self.d_spikes);
                    let _ = (d.cu_mem_free)(self.d_compacted);
                    let _ = (d.cu_mem_free)(self.d_counter);
                    if let Some(g) = self.graph.get() {
                        let _ = (d.cu_graph_exec_destroy)(g.exec);
                        let _ = (d.cu_stream_destroy)(g.stream);
                        let _ = (d.cu_mem_free)(g.d_args);
                    }
                    // The CUDA context is process-global — never destroyed here.
                }
            }
        }
    }
}

/// Compile a CUDA C++ source string through NVRTC (host-side compiler — no
/// GPU required) and check it produces PTX. Used to validate the
/// `#[dual_neuron]`-generated sources in tests on machines without a device.
pub fn nvrtc_compile_check(source: &str) -> Result<(), String> {
    // NVRTC is a host-side compiler: no GPU required. Try descending compute
    // archs — the newest one this NVRTC knows compiles; older archs are always
    // supported too, so the first success is the natural pick.
    // `nvrtc_to_ptx` formats `compute_{major}{minor}`, so pass the digits
    // separately (8, 9 → compute_89), not the arch number.
    let archs: &[(i32, i32)] = &[
        (9, 0),
        (8, 9),
        (8, 6),
        (8, 0),
        (7, 5),
        (7, 0),
        (6, 1),
        (6, 0),
    ];
    let mut last_err = "no arch attempted".to_string();
    for &(major, minor) in archs {
        match nvrtc_to_ptx(source, major, minor) {
            Ok(ptx) if !ptx.is_empty() => return Ok(()),
            Ok(_) => return Err("NVRTC produced empty PTX".to_string()),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

// ──────────────────────────────────────────────────────────────────
//  Process-global CUDA context
// ──────────────────────────────────────────────────────────────────

use std::sync::OnceLock;

/// Raw CUDA context pointer, safely sharable: the context is process-global
/// and lives until process exit, so sharing it between `CompiledModel`s (and
/// threads that use it sequentially) is sound.
#[derive(Copy, Clone)]
struct CtxHandle(CUcontext);
unsafe impl Send for CtxHandle {}
unsafe impl Sync for CtxHandle {}

/// One CUDA context per process, created lazily.
///
/// Multiple `CompiledModel`s share it — the driver only ever has ONE
/// current context, so per-model contexts would silently invalidate each
/// other (the last `cuCtxCreate` becomes current and the earlier models'
/// buffers become "foreign" → CUDA_ERROR_INVALID_ARGUMENT).
static GLOBAL_CTX: OnceLock<Result<(CtxHandle, i32, i32), String>> = OnceLock::new();

fn global_ctx() -> &'static Result<(CtxHandle, i32, i32), String> {
    GLOBAL_CTX.get_or_init(|| {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        unsafe {
            check_cu(d, (d.cu_init)(0))?;
            let mut dev: CUdevice = 0;
            check_cu(d, (d.cu_device_get)(&mut dev, 0))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check_cu(d, (d.cu_ctx_create)(&mut ctx, 0, dev))?;
            let (major, minor) = compute_capability(d, dev)?;
            Ok((CtxHandle(ctx), major, minor))
        }
    })
}

/// Make the process-global CUDA context **current on the calling thread**.
///
/// The CUDA driver API (`cuModuleLoadDataEx`, `cuLaunchKernel`, …) uses the
/// *calling thread's* current context — not the context that was created.
/// `global_ctx()` is created lazily on whichever thread happens to call it
/// first; every other thread would otherwise see a null current context and
/// fail with `CUDA_ERROR_INVALID_CONTEXT`. Calling this at the top of every
/// public entry point makes the engine thread-agnostic (e.g. libtest runs
/// each test on its own worker thread).
///
/// NOTE: `cuCtxPushCurrent` returns CUDA_ERROR_INVALID_CONTEXT (201) on
/// some drivers — the legacy context-stack API is unreliable. `cuCtxSetCurrent`
/// unconditionally sets the thread's current context, which is exactly what
/// we need (the process uses one context, no restore required).
/// Serializes graph capture + replay across `CompiledModel`s (see
/// [`CompiledModel::step_graph`]).
static STEP_LOCK: Mutex<()> = Mutex::new(());

fn set_current_ctx(d: &'static Driver) -> Result<(), String> {
    let (ctx, _, _) = *global_ctx().as_ref().map_err(|e| e.clone())?;
    unsafe { check_cu(d, (d.cu_ctx_set_current)(ctx.0)) }
}

// ──────────────────────────────────────────────────────────────────
//  Internals
// ──────────────────────────────────────────────────────────────────

/// Capture the kernel launch into a CUDA Graph and instantiate it.
///
/// The kernel is launched once inside a stream-capture region; the resulting
/// graph is instantiated for replay. Scalar args (dt/n/stride) live in a
/// device buffer pointed to by the graph — updating that buffer between
/// replays changes dt without re-capturing.
///
/// # Safety
/// `model` must be fully uploaded before the first `step_graph` (the capture
/// launch reads whatever is in device memory — typically zeros, harmless).
/// The graph captures pointers to `model`'s device buffers; those must stay
/// alive for the graph's lifetime (they do — owned by `CompiledModel`).
unsafe fn capture_graph(d: &Driver, model: &CompiledModel, _dt: f32) -> Result<GraphState, String> {
    // 1. Device buffer for scalar args [dt, n, stride] (12 bytes).
    let mut d_args: CUdeviceptr = 0;
    check_cu(d, (d.cu_mem_alloc)(&mut d_args, 12))
        .map_err(|e| format!("graph d_args alloc: {e}"))?;

    // 2. Stream + capture region.
    let mut stream: CUstream = std::ptr::null_mut();
    if let Err(e) = check_cu(d, (d.cu_stream_create)(&mut stream, CU_STREAM_DEFAULT)) {
        let _ = (d.cu_mem_free)(d_args);
        return Err(format!("graph stream create: {e}"));
    }
    if let Err(e) = check_cu(
        d,
        (d.cu_stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
    ) {
        let _ = (d.cu_stream_destroy)(stream);
        let _ = (d.cu_mem_free)(d_args);
        return Err(format!("graph begin_capture: {e}"));
    }

    // 3. Launch the kernel inside the capture region with the SAME args as
    //    `step_graph` will use — device pointers + d_args for the scalars.
    let grid = (model.n as u32).div_ceil(BLOCK);
    let mut d_states = model.d_states;
    let mut d_params = model.d_params;
    let mut d_i_in = model.d_i_in;
    let mut d_spikes = model.d_spikes;
    let mut args: [*mut c_void; 5] = [
        &mut d_states as *mut CUdeviceptr as *mut c_void,
        &mut d_params as *mut CUdeviceptr as *mut c_void,
        &mut d_i_in as *mut CUdeviceptr as *mut c_void,
        &mut d_spikes as *mut CUdeviceptr as *mut c_void,
        &d_args as *const CUdeviceptr as *mut c_void,
    ];
    let launch_rc = (d.cu_launch_kernel)(
        model.func,
        grid,
        1,
        1,
        BLOCK,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if launch_rc != 0 {
        let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
        let _ = (d.cu_stream_destroy)(stream);
        let _ = (d.cu_mem_free)(d_args);
        return Err(format!("graph capture launch: {}", cu_error(d, launch_rc)));
    }

    // 4. End capture, instantiate.
    let mut graph: CUgraph = std::ptr::null_mut();
    if let Err(e) = check_cu(d, (d.cu_stream_end_capture)(stream, &mut graph)) {
        let _ = (d.cu_stream_destroy)(stream);
        let _ = (d.cu_mem_free)(d_args);
        return Err(format!("graph end_capture: {e}"));
    }
    let mut exec: CUgraphExec = std::ptr::null_mut();
    let rc = (d.cu_graph_instantiate)(
        &mut exec,
        graph,
        CU_GRAPH_INSTANTIATE_DEFAULT,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
    );
    let _ = (d.cu_graph_destroy)(graph); // exec is independent of graph
    if rc != 0 {
        let _ = (d.cu_stream_destroy)(stream);
        let _ = (d.cu_mem_free)(d_args);
        return Err(format!("graph instantiate: {}", cu_error(d, rc)));
    }

    Ok(GraphState {
        exec,
        stream,
        d_args,
    })
}

/// Allocate several device buffers; rolls back (frees everything already
/// allocated) if any allocation fails.
unsafe fn alloc_all(d: &Driver, sizes: &[usize]) -> Result<Vec<CUdeviceptr>, String> {
    let mut out: Vec<CUdeviceptr> = Vec::with_capacity(sizes.len());
    for &sz in sizes {
        let mut p: CUdeviceptr = 0;
        if let Err(e) = check_cu(d, (d.cu_mem_alloc)(&mut p, sz)) {
            for &q in &out {
                let _ = (d.cu_mem_free)(q);
            }
            return Err(e);
        }
        out.push(p);
    }
    Ok(out)
}

fn compute_capability(d: &Driver, dev: CUdevice) -> Result<(i32, i32), String> {
    unsafe {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        check_cu(
            d,
            (d.cu_device_get_attribute)(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            ),
        )?;
        check_cu(
            d,
            (d.cu_device_get_attribute)(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            ),
        )?;
        Ok((major, minor))
    }
}

fn compile_and_load(
    d: &Driver,
    _ctx: CUcontext,
    source: &str,
    kernel_name: &str,
    major: i32,
    minor: i32,
) -> Result<CUfunction, String> {
    unsafe {
        let ptx = nvrtc_to_ptx(source, major, minor)?;
        let mut module: CUmodule = std::ptr::null_mut();
        check_cu(
            d,
            (d.cu_module_load_data_ex)(
                &mut module,
                ptx.as_ptr() as *const c_void,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
        )?;
        // NVRTC compiles as C++ → kernel names are mangled (e.g.
        // `_Z9hh_kernelPfPKfS1_Pifii`). Read the real name out of the PTX.
        let mangled = extract_entry_name(&ptx, kernel_name);
        let name = CString::new(mangled).map_err(|_| "kernel name contains NUL")?;
        let mut func: CUfunction = std::ptr::null_mut();
        let rc = (d.cu_module_get_function)(&mut func, module, name.as_ptr());
        if rc != 0 {
            let ptx_head = String::from_utf8_lossy(&ptx[..ptx.len().min(800)]).into_owned();
            return Err(format!(
                "kernel `{kernel_name}` not found in module ({}):\n{ptx_head}\n---\n{}",
                cu_error(d, rc),
                first_lines(source, 20)
            ));
        }
        Ok(func)
    }
}

fn nvrtc_to_ptx(source: &str, major: i32, minor: i32) -> Result<Vec<u8>, String> {
    unsafe {
        let mut prog: nvrtcProgram = std::ptr::null_mut();
        let src = CString::new(source).map_err(|_| "source contains NUL")?;
        let fname = CString::new("dual.cu").unwrap();
        let rc = nvrtcCreateProgram(
            &mut prog,
            src.as_ptr(),
            fname.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if rc != 0 {
            return Err(format!("nvrtcCreateProgram failed: {}", nvrtc_error(rc)));
        }

        let arch = format!("--gpu-architecture=compute_{major}{minor}");
        let opt = CString::new(arch).unwrap();
        let opts = [opt.as_ptr()];
        let rc = nvrtcCompileProgram(prog, 1, opts.as_ptr());
        if rc != 0 {
            let log = nvrtc_log(prog);
            let _ = nvrtcDestroyProgram(&mut prog);
            return Err(format!("NVRTC compile error: {}\n{}", nvrtc_error(rc), log));
        }

        let mut ptx_size: usize = 0;
        let _ = nvrtcGetPTXSize(prog, &mut ptx_size);
        let mut ptx = vec![0u8; ptx_size];
        let _ = nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char);
        let _ = nvrtcDestroyProgram(&mut prog);
        Ok(ptx)
    }
}

fn nvrtc_log(prog: nvrtcProgram) -> String {
    unsafe {
        let mut size: usize = 0;
        if nvrtcGetProgramLogSize(prog, &mut size) != 0 || size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        if nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) != 0 {
            return String::new();
        }
        String::from_utf8_lossy(&buf).into_owned()
    }
}

// Note: NVRTC error strings are served by `crate::ffi::nvrtc_error`; engine
// code calls `nvrtc_error(rc)` re-exported below.
pub use crate::ffi::nvrtc_error;

fn cu_error(d: &Driver, rc: CUresult) -> String {
    unsafe {
        let mut p: *const c_char = std::ptr::null();
        if (d.cu_get_error_string)(rc, &mut p) != 0 || p.is_null() {
            return format!("CUDA error {rc}");
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn check_cu(d: &Driver, rc: CUresult) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(cu_error(d, rc))
    }
}

fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join("\n")
}

/// Extract the (mangled) kernel name from PTX — the token after
/// `.visible .entry` up to the opening `(`.
fn extract_entry_name(ptx: &[u8], fallback: &str) -> String {
    let text = String::from_utf8_lossy(ptx);
    for line in text.lines() {
        if let Some(idx) = line.find(".entry ") {
            let rest = &line[idx + ".entry ".len()..];
            if let Some(open) = rest.find('(') {
                let name = rest[..open].trim();
                if !name.is_empty() {
                    return name.to_string();
                }
            }
        }
    }
    fallback.to_string()
}
