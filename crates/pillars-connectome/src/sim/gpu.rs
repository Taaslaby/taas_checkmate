//! CUDA GPU LIF simulation.

use super::BrainState;
use crate::data::ConnectomeData;
use pillars_cuda::gpu::{DevicePtr, GpuContext};
use std::sync::Mutex;

// ─── GPU LIF simulation (CUDA) ──────────────────────────

static GPU_CTX: Mutex<Option<(GpuContext, usize, usize, bool)>> = Mutex::new(None);
/// Cached GPU spike buffer — avoids alloc/free churn between batch calls.
static GPU_SPIKE_BUF: Mutex<Option<(DevicePtr<i32>, usize)>> = Mutex::new(None);

/// Get or create a GPU spike buffer of the required size.
fn get_spike_buf(n_ints: usize) -> DevicePtr<i32> {
    let mut cache = GPU_SPIKE_BUF.lock().unwrap();
    if let Some((ptr, size)) = cache.take() {
        if size == n_ints {
            return ptr; // reuse
        }
        // size changed — old buf dropped here
    }
    let ptr = DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer");
    *cache = Some((ptr, n_ints));
    // Return a fresh allocation (cache holds the same-sized one for next call)
    DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer")
}

/// Return a spike buffer to cache for reuse.
fn put_spike_buf(buf: DevicePtr<i32>, n_ints: usize) {
    let mut cache = GPU_SPIKE_BUF.lock().unwrap();
    *cache = Some((buf, n_ints));
}

/// Simulate `n_steps` of LIF dynamics on a connectome using GPU (CUDA).
///
/// Использует `batch_lif_steps` — аккумулирует спайки на GPU через per-step D2D,
/// скачивает весь spike buffer + v одним D2H в конце.
/// Также чинит stale external_I baseline (обновляет d_external_I каждый batch).
///
/// Первый вызов создаёт GpuContext, последующие используют persistent cache.
/// Uploads only v (not dummy u/a/b/c/d) — conditional upload saves ~5× bandwidth.
pub fn sim_lif_gpu(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    dt_ms: f32,
    external_I: &[f32],
    n_steps: usize,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_synapses = connectome.n_synapses;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut v_back = vec![0.0f32; n_neurons];

    // Get or create persistent GPU context
    let mut cache = GPU_CTX.lock().unwrap();
    let (gpu_ctx, needs_init) = match cache.take() {
        Some((ctx, cached_n, cached_s, uploaded))
            if cached_n == n_neurons && cached_s == n_synapses =>
        {
            (ctx, !uploaded)
        }
        _ => {
            let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
            (ctx, true)
        }
    };

    // Upload v only (no dummy u/a/b/c/d — LIF kernel doesn't use them)
    gpu_ctx.upload_v_only(&state.v[..n_neurons]).unwrap();

    if needs_init {
        gpu_ctx
            .upload_lif_params(dt_ms, tau_m, v_rest, v_thresh, v_reset)
            .unwrap();
        gpu_ctx
            .upload_connectome(
                &connectome.row_ptr,
                &connectome.col_indices,
                &connectome.values,
                &connectome.nt_types,
                external_I,
            )
            .unwrap();
    }

    // Get cached spike buffer or allocate new one
    let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

    // Batch run
    gpu_ctx
        .batch_lif_steps(
            external_I,
            d_spike_buf.as_mut_ptr(),
            &mut spike_buffer,
            &mut v_back,
            n_steps,
        )
        .expect("GPU: batch_lif_steps");

    // Return spike buffer to cache
    put_spike_buf(d_spike_buf, n_steps * n_neurons);

    // Copy v back to state
    state.v[..n_neurons].copy_from_slice(&v_back);

    *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

    spike_buffer
}

/// Simulate `n_steps` of Izhikevich dynamics on a connectome using GPU (CUDA).
///
/// Same batched approach as `sim_lif_gpu`:
/// - Per-step D2D spike accumulation, single D2H at end
/// - Updates d_external_I + d_I every call (fixes stale baseline)
/// - Uploads full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
/// - Downloads v + u at end for state persistence
///
/// Default Izhikevich params: a=0.02, b=0.2, c=-65, d=8, trace_decay=0.99
pub fn sim_izhikevich_gpu(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    dt_ms: f32,
    external_I: &[f32],
    n_steps: usize,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_synapses = connectome.n_synapses;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut v_back = vec![0.0f32; n_neurons];
    let mut u_back = vec![0.0f32; n_neurons];
    let trace_decay = 0.99f32;

    // Get or create persistent GPU context
    let mut cache = GPU_CTX.lock().unwrap();
    let (gpu_ctx, needs_init) = match cache.take() {
        Some((ctx, cached_n, cached_s, uploaded))
            if cached_n == n_neurons && cached_s == n_synapses =>
        {
            (ctx, !uploaded)
        }
        _ => {
            let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
            (ctx, true)
        }
    };

    // Upload full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
    gpu_ctx
        .upload_neurons(&state.v[..n_neurons], &state.u[..n_neurons], a, b, c, d)
        .unwrap();

    if needs_init {
        gpu_ctx
            .upload_connectome(
                &connectome.row_ptr,
                &connectome.col_indices,
                &connectome.values,
                &connectome.nt_types,
                external_I,
            )
            .unwrap();
    }

    // Get cached spike buffer
    let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

    // Batch run
    gpu_ctx
        .batch_izhikevich_steps(
            external_I,
            dt_ms,
            trace_decay,
            d_spike_buf.as_mut_ptr(),
            &mut spike_buffer,
            &mut v_back,
            &mut u_back,
            n_steps,
        )
        .expect("GPU: batch_izhikevich_steps");

    // Return spike buffer to cache
    put_spike_buf(d_spike_buf, n_steps * n_neurons);

    // Copy v + u back to state
    state.v[..n_neurons].copy_from_slice(&v_back);
    state.u[..n_neurons].copy_from_slice(&u_back);

    *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

    spike_buffer
}
