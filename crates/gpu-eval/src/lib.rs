//! GPU LIF connectome evaluator — C ABI for Python ctypes.
//!
//! Usage from Python:
//! ```python
//! import ctypes
//! lib = ctypes.CDLL("libgpu_lif.so")
//!
//! ctx = ctypes.c_void_p(lib.gpu_lif_create(
//!     row_ptr.ctypes.data, col_indices.ctypes.data,
//!     values.ctypes.data, nt_types.ctypes.data,
//!     ctypes.c_int32(n_neurons), ctypes.c_int32(n_synapses),
//!     tau_m.ctypes.data, v_rest.ctypes.data,
//!     v_thresh.ctypes.data, v_reset.ctypes.data,
//!     ctypes.c_float(dt_ms),
//! ))
//!
//! lib.gpu_lif_step(ctx, external_I.ctypes.data, n_steps)
//!
//! # Read motor neuron states
//! motor_out = (ctypes.c_float * len(motor_idx))()
//! lib.gpu_lif_read_motor(ctx, motor_out, motor_idx.ctypes.data, ctypes.c_int32(len(motor_idx)))
//!
//! lib.gpu_lif_destroy(ctx)
//! ```

use std::sync::Mutex;
use cuda_kernels::GpuContext;

type CachedCtx = (GpuContext, usize, usize, bool);

static GPU_CTX: Mutex<Option<CachedCtx>> = Mutex::new(None);

/// Create/reuse GPU context and upload connectome + LIF params.
/// Returns opaque handle (1 = success).
#[no_mangle]
pub extern "C" fn gpu_lif_create(
    row_ptr: *const i32, col_indices: *const i32,
    values: *const f32, nt_types: *const f32,
    n_neurons: i32, n_synapses: i32,
    tau_m: *const f32, v_rest: *const f32,
    v_thresh: *const f32, v_reset: *const f32,
    dt_ms: f32,
) -> i32 {
    let n = n_neurons as usize;
    let s = n_synapses as usize;

    if n == 0 || row_ptr.is_null() || col_indices.is_null() || values.is_null() {
        return -1;
    }

    let rp = unsafe { std::slice::from_raw_parts(row_ptr, n + 1) };
    let ci = unsafe { std::slice::from_raw_parts(col_indices, s) };
    let vl = unsafe { std::slice::from_raw_parts(values, s) };
    let nt = if nt_types.is_null() {
        vec![1.0f32; n]
    } else {
        unsafe { std::slice::from_raw_parts(nt_types, n).to_vec() }
    };

    let tau_m_sl = if tau_m.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(tau_m, n) } };
    let v_rest_sl = if v_rest.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(v_rest, n) } };
    let v_thresh_sl = if v_thresh.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(v_thresh, n) } };
    let v_reset_sl = if v_reset.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(v_reset, n) } };

    let mut cache = match GPU_CTX.lock() {
        Ok(c) => c,
        Err(_) => return -2,
    };

    let (gpu_ctx, needs_init) = match cache.take() {
        Some((ctx, cn, cs, uploaded)) if cn == n && cs == s => (ctx, !uploaded),
        _ => {
            let ctx = match GpuContext::new(n, s) {
                Ok(c) => c,
                Err(e) => { eprintln!("gpu_lif: GpuContext::new failed: {e}"); return -3; }
            };
            (ctx, true)
        }
    };

    // Upload initial v (resting) + dummy Izhikevich params (unused by LIF)
    let v_init = vec![v_rest_sl.first().copied().unwrap_or(-70.0); n];
    let dummy_u = vec![0.0f32; n];
    let dummy_a = vec![0.02f32; n];
    let dummy_b = vec![0.2f32; n];
    let dummy_c = vec![-70.0f32; n];
    let dummy_d = vec![2.0f32; n];
    if let Err(e) = gpu_ctx.upload_neurons(&v_init, &dummy_u, &dummy_a, &dummy_b, &dummy_c, &dummy_d) {
        eprintln!("gpu_lif: upload_neurons failed: {e}"); return -4;
    }        if needs_init {
            // Upload LIF params (one-time)
            if !tau_m_sl.is_empty() {
                if let Err(e) = gpu_ctx.upload_lif_params(dt_ms, tau_m_sl, v_rest_sl, v_thresh_sl, v_reset_sl) {
                    eprintln!("gpu_lif: upload_lif_params failed: {e}"); return -5;
                }
            }
            // Upload connectome STRUCTURE (row_ptr, col_indices — one-time, never changes)
            if let Err(e) = gpu_ctx.upload_connectome(rp, ci, vl, &nt, &vec![0.0f32; n]) {
                eprintln!("gpu_lif: upload_connectome failed: {e}"); return -6;
            }
        } else {
            // Always upload WEIGHTS (values change every generation!)
            // Re-upload the full connectome with new weights
            if let Err(e) = gpu_ctx.upload_connectome(rp, ci, vl, &nt, &vec![0.0f32; n]) {
                eprintln!("gpu_lif: upload_connectome (weights update) failed: {e}"); return -6;
            }
        }

    // Upload initial external_I = 0
    if let Err(e) = gpu_ctx.upload_I(&vec![0.0f32; n]) {
        eprintln!("gpu_lif: upload_I failed: {e}"); return -7;
    }

    *cache = Some((gpu_ctx, n, s, true));
    1 // success
}

/// Upload external current and run N LIF steps on GPU.
/// external_I must be at least n_neurons floats.
#[no_mangle]
pub extern "C" fn gpu_lif_step(
    _handle: i32,
    external_I: *const f32,
    n_neurons: i32,
    n_steps: i32,
) -> i32 {
    let n = n_neurons as usize;
    let n_st = n_steps as usize;

    let ext = if external_I.is_null() {
        return -1;
    } else {
        unsafe { std::slice::from_raw_parts(external_I, n).to_vec() }
    };

    let mut cache = match GPU_CTX.lock() {
        Ok(c) => c,
        Err(_) => return -2,
    };

    let (gpu_ctx, _, _, _) = match cache.as_ref() {
        Some(ctx) => ctx,
        None => return -3,
    };

    // Upload fresh external_I
    if let Err(e) = gpu_ctx.upload_I(&ext) {
        eprintln!("gpu_lif_step: upload_I failed: {e}"); return -4;
    }

    // Allocate spike counter
    let d_counter = match gpu_ctx.alloc_spike_counter() {
        Ok(p) => p,
        Err(e) => { eprintln!("gpu_lif_step: alloc_spike_counter failed: {e}"); return -5; }
    };

    // Run N LIF steps (all on GPU, no H2D/D2H between steps)
    for _ in 0..n_st {
        if let Err(e) = gpu_ctx.connectome_lif_step(Some(d_counter)) {
            gpu_ctx.free_spike_counter(d_counter);
            eprintln!("gpu_lif_step: connectome_lif_step failed: {e}");
            return -6;
        }
    }

    gpu_ctx.free_spike_counter(d_counter);

    // Sync and return
    let _ = gpu_ctx.sync();
    0 // success
}

/// Download spike vector (0/1 for each neuron).
/// spike_out must have n_neurons elements.
#[no_mangle]
pub extern "C" fn gpu_lif_read_spikes(
    _handle: i32,
    spike_out: *mut i32,
    n_neurons: i32,
) -> i32 {
    let n = n_neurons as usize;

    let mut cache = match GPU_CTX.lock() {
        Ok(c) => c,
        Err(_) => return -2,
    };

    let (gpu_ctx, _, _, _) = match cache.as_ref() {
        Some(ctx) => ctx,
        None => return -3,
    };

    let mut spikes = vec![0i32; n];
    if let Err(e) = gpu_ctx.download_spikes(&mut spikes) {
        eprintln!("gpu_lif_read_spikes: download_spikes failed: {e}"); return -4;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(spikes.as_ptr(), spike_out, n);
    }
    0 // success
}

/// Download neuron membrane potentials (v).
/// v_out must have n_neurons elements.
#[no_mangle]
pub extern "C" fn gpu_lif_read_v(
    _handle: i32,
    v_out: *mut f32,
    n_neurons: i32,
) -> i32 {
    let n = n_neurons as usize;

    let mut cache = match GPU_CTX.lock() {
        Ok(c) => c,
        Err(_) => return -2,
    };

    let (gpu_ctx, _, _, _) = match cache.as_ref() {
        Some(ctx) => ctx,
        None => return -3,
    };

    let mut v = vec![0.0f32; n];
    if let Err(e) = gpu_ctx.download_v(&mut v) {
        eprintln!("gpu_lif_read_v: download_v failed: {e}"); return -4;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), v_out, n);
    }
    0 // success
}

/// Destroy GPU context and free resources.
#[no_mangle]
pub extern "C" fn gpu_lif_destroy() -> i32 {
    let mut cache = match GPU_CTX.lock() {
        Ok(c) => c,
        Err(_) => return -1,
    };
    *cache = None;
    0 // success
}
