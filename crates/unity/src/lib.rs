//! # taas-unity — C-ABI Unity adapter for TaaS neural engine.
//!
//! ## C API (P/Invoke compatible)
//!
//! ```c
//! // Lifecycle
//! void* brain_new(int n_neurons, const char* model);     // "lif" | "izhikevich"
//! void  brain_free(void* brain);
//! void* connectome_load(const int* row_ptr, int n_neurons,
//!                       const int* col_indices, const float* values,
//!                       const float* nt_types, int n_synapses);
//! void  connectome_free(void* conn);
//!
//! // Step
//! int   brain_step(void* brain, void* conn,
//!                  const float* external_I, float dt_ms);
//!
//! // State
//! int   brain_get_v(void* brain, float* out, int max_n);
//! int   brain_get_u(void* brain, float* out, int max_n);
//! int   brain_get_spikes(void* brain, int* out, int max_n);
//! int   brain_get_I_buf(void* brain, float* out, int max_n);
//!
//! // Param overrides
//! int   brain_set_lif_params(void* brain,
//!                            const float* tau_m, const float* v_rest,
//!                            const float* v_thresh, const float* v_reset,
//!                            int n, float tau_ref, float dt_ms);
//! int   brain_set_izhikevich_params(void* brain,
//!                                   const float* a, const float* b,
//!                                   const float* c, const float* d,
//!                                   int n, float tau_ref, float dt_ms);
//!
//! // Reset
//! void  brain_reset(void* brain);
//!
//! // Info
//! int   brain_n_neurons(void* brain);
//! int   connectome_n_neurons(void* conn);
//! int   connectome_n_synapses(void* conn);
//! ```
//!
//! ## C# usage (Unity)
//!
//! ```csharp
//! [DllImport("taas_unity")] static extern IntPtr brain_new(int n, string model);
//! [DllImport("taas_unity")] static extern void   brain_free(IntPtr brain);
//! [DllImport("taas_unity")] static extern int    brain_step(IntPtr brain, IntPtr conn, float[] I, float dt);
//!
//! IntPtr brain = brain_new(100000, "lif");
//! IntPtr conn = connectome_load(rp, n, ci, vals, nt, n_syn);
//! int spikes = brain_step(brain, conn, I, 0.5f);
//! ```

#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;

use taas_connectome::{ConnectomeData, BrainState, NeuronModel};
use taas_kernels::safe;
use taas_connectome::{push_spikes_to_queue, csr_propagate_from_queue};

// ═══════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════

const MAGIC_BRAIN: u32 = 0xABADBABE;
const MAGIC_CONN: u32 = 0xDEADBEEF;

// Error codes returned by C API functions
const OK: i32 = 0;
const ERR_MAGIC: i32 = -2;
const ERR_MODEL: i32 = -3;
const ERR_PARAM: i32 = -5;

// Default Izhikevich params for ~regular-spiking (RS) neurons
const DEFAULT_A: f32 = 0.02;
const DEFAULT_B: f32 = 0.2;
const DEFAULT_C: f32 = -65.0;
const DEFAULT_D: f32 = 8.0;
const DEFAULT_TAU_REF: f32 = 2.0;
const DEFAULT_DT_MS: f32 = 0.5;
const TRACE_DECAY: f32 = 0.99;

// Step flags for brain_step_ex — compose your own pipeline
pub const STEP_SET_INPUT: i32 = 1;
pub const STEP_NEURON: i32 = 2;
pub const STEP_PUSH: i32 = 4;
pub const STEP_PROPAGATE: i32 = 8;
pub const STEP_FULL: i32 = 15;  // all stages

// Default LIF params
const DEFAULT_TAU_M: f32 = 10.0;
const DEFAULT_V_REST: f32 = -70.0;
const DEFAULT_V_THRESH: f32 = -52.0;
const DEFAULT_V_RESET: f32 = -70.0;

// ═══════════════════════════════════════════════════════════
//  Opaque handle types
// ═══════════════════════════════════════════════════════════

/// Opaque handle for a brain simulation (LIF or Izhikevich).
///
/// Stores:
/// - BrainState: v, u, trace, I_buf, spikes, refractory, queue
/// - Model type + params
/// - Padded arrays for SIMD kernel calls
#[repr(C)]
pub struct BrainHandle {
    magic: u32,
    state: BrainState,
    model: NeuronModel,
    n_neurons: usize,
    n_padded: usize,
    dt_ms: f32,
    params: BrainParams,
}

enum BrainParams {
    Lif(LifStorage),
    Izhikevich(IzhikevichStorage),
}

/// LIF model storage — packed params matching SIMD kernel layout.
///
/// Kernel expects `params[n * 5]` with layout:
///   [0..n]     = dt      (slot 0, same for all neurons)
///   [n..2n]    = tau_m   (slot 1)
///   [2n..3n]   = v_rest  (slot 2)
///   [3n..4n]   = v_thresh(slot 3)
///   [4n..5n]   = v_reset (slot 4)
struct LifStorage {
    packed: Vec<f32>,   // [n_padded * 5]
    tau_ref: f32,
}

/// Izhikevich model storage — per-neuron params padded to n_padded.
struct IzhikevichStorage {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    tau_ref: f32,
}

/// Opaque handle for a connectome (CSR matrix).
#[repr(C)]
pub struct ConnectomeHandle {
    magic: u32,
    data: ConnectomeData,
}

// ═══════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════

/// Pad a slice to length `n_padded` by replicating the last element.
fn pad_to(v: &[f32], n_padded: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    if out.len() < n_padded {
        let last = *out.last().unwrap_or(&0.0);
        out.resize(n_padded, last);
    }
    out
}

/// Pack LIF params into the SIMD kernel format.
///
/// Layout per neuron (5 slots): [dt, tau_m, v_rest, v_thresh, v_reset]
fn pack_lif(
    tau_m: &[f32], v_rest: &[f32], v_thresh: &[f32], v_reset: &[f32],
    dt_ms: f32, n_padded: usize,
) -> Vec<f32> {
    let n = tau_m.len().min(v_rest.len()).min(v_thresh.len()).min(v_reset.len());
    let mut packed = Vec::with_capacity(n_padded * 5);
    for i in 0..n_padded {
        let idx = if i < n { i } else { n - 1 };
        packed.push(dt_ms);                            // slot 0: dt
        packed.push(tau_m[idx]);                       // slot 1: tau_m
        packed.push(v_rest[idx]);                      // slot 2: v_rest
        packed.push(v_thresh[idx]);                    // slot 3: v_thresh
        packed.push(v_reset[idx]);                     // slot 4: v_reset
    }
    packed
}

/// Update dt entries in a packed LIF array (avoids full repack).
fn update_dt_in_packed(packed: &mut [f32], dt_ms: f32, n_padded: usize) {
    for i in 0..n_padded {
        packed[i] = dt_ms;
    }
}

/// Create a brain state with default params for the given model.
fn create_brain(n_neurons: usize, model: NeuronModel) -> Box<BrainHandle> {
    let state = BrainState::new(n_neurons, model);
    let n_padded = state.n_padded;

    let params = match model {
        NeuronModel::LIF => {
            let tau_m = vec![DEFAULT_TAU_M; n_neurons];
            let v_rest = vec![DEFAULT_V_REST; n_neurons];
            let v_thresh = vec![DEFAULT_V_THRESH; n_neurons];
            let v_reset = vec![DEFAULT_V_RESET; n_neurons];
            let packed = pack_lif(&tau_m, &v_rest, &v_thresh, &v_reset, DEFAULT_DT_MS, n_padded);
            BrainParams::Lif(LifStorage { packed, tau_ref: DEFAULT_TAU_REF })
        }
        NeuronModel::Izhikevich => {
            BrainParams::Izhikevich(IzhikevichStorage {
                a: pad_to(&[DEFAULT_A], n_padded),
                b: pad_to(&[DEFAULT_B], n_padded),
                c: pad_to(&[DEFAULT_C], n_padded),
                d: pad_to(&[DEFAULT_D], n_padded),
                tau_ref: DEFAULT_TAU_REF,
            })
        }
        _ => {
            // Fallback: LIF
            let tau_m = vec![DEFAULT_TAU_M; n_neurons];
            let v_rest = vec![DEFAULT_V_REST; n_neurons];
            let v_thresh = vec![DEFAULT_V_THRESH; n_neurons];
            let v_reset = vec![DEFAULT_V_RESET; n_neurons];
            let packed = pack_lif(&tau_m, &v_rest, &v_thresh, &v_reset, DEFAULT_DT_MS, n_padded);
            BrainParams::Lif(LifStorage { packed, tau_ref: DEFAULT_TAU_REF })
        }
    };

    Box::new(BrainHandle {
        magic: MAGIC_BRAIN,
        state,
        model,
        n_neurons,
        n_padded,
        dt_ms: DEFAULT_DT_MS,
        params,
    })
}

/// Validate a brain handle pointer.
fn as_brain<'a>(ptr: *mut c_void) -> Option<&'a mut BrainHandle> {
    if ptr.is_null() { return None; }
    let brain = unsafe { &mut *(ptr as *mut BrainHandle) };
    if brain.magic != MAGIC_BRAIN { return None; }
    Some(brain)
}

/// Validate a connectome handle pointer.
fn as_connectome<'a>(ptr: *mut c_void) -> Option<&'a ConnectomeHandle> {
    if ptr.is_null() { return None; }
    let conn = unsafe { &*(ptr as *const ConnectomeHandle) };
    if conn.magic != MAGIC_CONN { return None; }
    Some(conn)
}

/// Validate a connectome handle pointer (mutable).
fn as_connectome_mut<'a>(ptr: *mut c_void) -> Option<&'a mut ConnectomeHandle> {
    if ptr.is_null() { return None; }
    let conn = unsafe { &mut *(ptr as *mut ConnectomeHandle) };
    if conn.magic != MAGIC_CONN { return None; }
    Some(conn)
}

// ═══════════════════════════════════════════════════════════
//  Lifecycle: brain
// ═══════════════════════════════════════════════════════════

/// Create a new brain simulator.
///
/// `model` must be "lif" or "izhikevich".
/// Returns opaque handle, or null on error.
#[no_mangle]
pub extern "C" fn brain_new(n_neurons: i32, model: *const c_char) -> *mut c_void {
    if n_neurons <= 0 || model.is_null() {
        return ptr::null_mut();
    }

    let model_str = match unsafe { CStr::from_ptr(model) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let nm = match model_str {
        "lif" => NeuronModel::LIF,
        "izhikevich" => NeuronModel::Izhikevich,
        _ => return ptr::null_mut(),
    };

    let brain = create_brain(n_neurons as usize, nm);
    Box::into_raw(brain) as *mut c_void
}

/// Destroy a brain simulator created with `brain_new`.
#[no_mangle]
pub extern "C" fn brain_free(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    // Validate magic before freeing
    if let Some(brain) = as_brain(ptr) {
        brain.magic = 0; // invalidate
    }
    // Free regardless of magic (safe because we zeroed it)
    unsafe { let _ = Box::from_raw(ptr as *mut BrainHandle); }
}

// ═══════════════════════════════════════════════════════════
//  Lifecycle: connectome
// ═══════════════════════════════════════════════════════════

/// Load a connectome from CSR arrays.
///
/// Takes ownership of the provided data (copies into Rust heap).
/// Returns opaque handle, or null on error.
#[no_mangle]
pub extern "C" fn connectome_load(
    row_ptr: *const i32,
    n_neurons: i32,
    col_indices: *const i32,
    values: *const f32,
    nt_types: *const f32,
    n_synapses: i32,
) -> *mut c_void {
    if row_ptr.is_null() || col_indices.is_null() || values.is_null()
        || n_neurons <= 0 || n_synapses <= 0
    {
        return ptr::null_mut();
    }

    let nn = n_neurons as usize;
    let ns = n_synapses as usize;

    let rp = unsafe { std::slice::from_raw_parts(row_ptr, nn + 1) }.to_vec();
    let ci = unsafe { std::slice::from_raw_parts(col_indices, ns) }.to_vec();
    let vl = unsafe { std::slice::from_raw_parts(values, ns) }.to_vec();
    let nt = if nt_types.is_null() {
        vec![1.0f32; nn]
    } else {
        unsafe { std::slice::from_raw_parts(nt_types, nn) }.to_vec()
    };

    // Validate row_ptr invariants
    if rp[0] != 0 || rp[nn] as usize != ns {
        // Invalid CSR — row_ptr[0] must be 0, row_ptr[n] must be n_synapses
        return ptr::null_mut();
    }

    let data = ConnectomeData {
        n_neurons: nn,
        n_synapses: ns,
        row_ptr: rp,
        col_indices: ci,
        values: vl,
        nt_types: nt,
        receptor_types: vec![],
        groups: vec![],
    };

    let handle = Box::new(ConnectomeHandle {
        magic: MAGIC_CONN,
        data,
    });

    Box::into_raw(handle) as *mut c_void
}

/// Destroy a connectome created with `connectome_load`.
#[no_mangle]
pub extern "C" fn connectome_free(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    if let Some(conn) = as_connectome_mut(ptr) {
        conn.magic = 0;
    }
    unsafe { let _ = Box::from_raw(ptr as *mut ConnectomeHandle); }
}

// ═══════════════════════════════════════════════════════════
//  Step
// ═══════════════════════════════════════════════════════════

/// Run one simulation step.
///
/// 1. Sets I_buf = external_I (zeros if null)
/// 2. Neuron step (LIF or Izhikevich) — updates v, u, spikes
/// 3. Pushes spikes to internal queue
/// 4. Resets I_buf to external_I
/// 5. CSR propagates queued spikes → I_buf
///
/// Returns number of spikes fired this step, or negative error code.
#[no_mangle]
pub extern "C" fn brain_step(
    brain: *mut c_void,
    conn: *mut c_void,
    external_I: *const f32,
    dt_ms: f32,
) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    let conn = match as_connectome(conn) {
        Some(c) => c,
        None => return ERR_MAGIC,
    };

    let n = conn.data.n_neurons;
    let n_neurons = conn.data.row_ptr.len() - 1;

    // Update dt if changed
    brain.dt_ms = if dt_ms > 0.0 { dt_ms } else { brain.dt_ms };

    // 1. Set I_buf = external_I (or zeros)
    let n_use = n.min(brain.state.I_buf.len());
    if !external_I.is_null() {
        let ext = unsafe { std::slice::from_raw_parts(external_I, n_use) };
        brain.state.I_buf[..n_use].copy_from_slice(&ext[..n_use]);
        for i in n_use..brain.state.I_buf.len() {
            brain.state.I_buf[i] = 0.0;
        }
    } else {
        brain.state.I_buf.fill(0.0);
    }

    // 2. Neuron step
    match &mut brain.params {
        BrainParams::Lif(storage) => {
            // Update dt in packed array if needed
            if (storage.packed[0] - brain.dt_ms).abs() > 1e-8 {
                update_dt_in_packed(&mut storage.packed, brain.dt_ms, brain.n_padded);
            }
            safe::lif_step(
                &mut brain.state.v,
                &mut brain.state.I_buf,
                &mut brain.state.spikes,
                &storage.packed,
                &mut brain.state.refractory,
                storage.tau_ref,
                None,
            );
        }
        BrainParams::Izhikevich(storage) => {
            safe::izhikevich_step(
                &mut brain.state.v,
                &mut brain.state.u,
                &mut brain.state.trace,
                &storage.a,
                &storage.b,
                &mut brain.state.I_buf,
                brain.dt_ms,
                TRACE_DECAY,
                &mut brain.state.spikes,
                &storage.c,
                &storage.d,
                &mut brain.state.refractory,
                storage.tau_ref,
                None,
            );
        }
    }

    // 4. Push spikes to queue
    let spike_count = push_spikes_to_queue(&brain.state.spikes, n_neurons, &mut brain.state.queue);

    // 5. CSR propagate from queue → I_buf
    csr_propagate_from_queue(
        &conn.data.row_ptr,
        &conn.data.col_indices,
        &conn.data.values,
        &conn.data.nt_types,
        &mut brain.state.queue,
        &mut brain.state.I_buf,
        1.0, // scale
    );

    spike_count.min(i32::MAX as u32) as i32
}

// ═══════════════════════════════════════════════════════════
//  Flexible step API
// ═══════════════════════════════════════════════════════════

/// Run a simulation step with composable flags.
///
/// `flags` is a bitmask of `STEP_*` constants:
/// - `STEP_SET_INPUT` (1): copy `external_I` → I_buf (zeros if null)
/// - `STEP_NEURON`    (2): run neuron model (LIF/Izhikevich) using current I_buf
/// - `STEP_PUSH`      (4): push spikes to internal queue
/// - `STEP_PROPAGATE` (8): CSR propagate queued spikes → I_buf
///
/// Examples:
/// ```c
/// // Just update neurons (input already set, don't propagate)
/// brain_step_ex(brain, null, null, dt, STEP_NEURON);
///
/// // Set input + neuron step (read spikes manually)
/// brain_step_ex(brain, null, I, dt, STEP_SET_INPUT | STEP_NEURON);
///
/// // Propagate existing queue (after manual spike injection)
/// brain_step_ex(brain, conn, null, dt, STEP_PROPAGATE);
///
/// // Full step (same as brain_step)
/// brain_step_ex(brain, conn, I, dt, STEP_FULL);
/// ```
///
/// Returns number of spikes fired this step, or negative error code.
#[no_mangle]
pub extern "C" fn brain_step_ex(
    brain: *mut c_void,
    conn: *mut c_void,
    external_I: *const f32,
    dt_ms: f32,
    flags: i32,
) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };

    // Update dt if changed
    if dt_ms > 0.0 {
        brain.dt_ms = dt_ms;
    }

    let n_neurons = brain.n_neurons;

    // 1. Set input (if flag set)
    if (flags & STEP_SET_INPUT) != 0 {
        let n_use = n_neurons.min(brain.state.I_buf.len());
        if !external_I.is_null() {
            let ext = unsafe { std::slice::from_raw_parts(external_I, n_use) };
            brain.state.I_buf[..n_use].copy_from_slice(&ext[..n_use]);
            for i in n_use..brain.state.I_buf.len() {
                brain.state.I_buf[i] = 0.0;
            }
        } else {
            brain.state.I_buf.fill(0.0);
        }
    }

    // 2. Neuron step (if flag set)
    if (flags & STEP_NEURON) != 0 {
        match &mut brain.params {
            BrainParams::Lif(storage) => {
                if (storage.packed[0] - brain.dt_ms).abs() > 1e-8 {
                    update_dt_in_packed(&mut storage.packed, brain.dt_ms, brain.n_padded);
                }
                safe::lif_step(
                    &mut brain.state.v,
                    &mut brain.state.I_buf,
                    &mut brain.state.spikes,
                    &storage.packed,
                    &mut brain.state.refractory,
                    storage.tau_ref,
                    None,
                );
            }
            BrainParams::Izhikevich(storage) => {
                safe::izhikevich_step(
                    &mut brain.state.v,
                    &mut brain.state.u,
                    &mut brain.state.trace,
                    &storage.a,
                    &storage.b,
                    &mut brain.state.I_buf,
                    brain.dt_ms,
                    TRACE_DECAY,
                    &mut brain.state.spikes,
                    &storage.c,
                    &storage.d,
                    &mut brain.state.refractory,
                    storage.tau_ref,
                    None,
                );
            }
        }
    }

    // 3. Push spikes to queue (if flag set)
    let spike_count = if (flags & STEP_PUSH) != 0 {
        push_spikes_to_queue(&brain.state.spikes, n_neurons, &mut brain.state.queue)
    } else {
        // Scan for spike count even without pushing
        brain.state.spikes.iter().take(n_neurons).filter(|&&s| s > 0).count() as u32
    };

    // 4. CSR propagate (if flag set and conn provided)
    if (flags & STEP_PROPAGATE) != 0 {
        if let Some(conn) = as_connectome(conn) {
            csr_propagate_from_queue(
                &conn.data.row_ptr,
                &conn.data.col_indices,
                &conn.data.values,
                &conn.data.nt_types,
                &mut brain.state.queue,
                &mut brain.state.I_buf,
                1.0,
            );
        }
    }

    spike_count.min(i32::MAX as u32) as i32
}

/// Run multiple simulation steps in Rust (no P/Invoke overhead).
///
/// Runs `n_steps` full steps (equivalent to `brain_step_ex(..., STEP_FULL)` n times).
/// Returns total spikes across all steps.
#[no_mangle]
pub extern "C" fn brain_step_n(
    brain: *mut c_void,
    conn: *mut c_void,
    external_I: *const f32,
    dt_ms: f32,
    n_steps: i32,
) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    let conn = match as_connectome(conn) {
        Some(c) => c,
        None => return ERR_MAGIC,
    };

    let steps = n_steps.max(1) as usize;
    let n = conn.data.n_neurons;
    let n_neurons = conn.data.row_ptr.len() - 1;

    if dt_ms > 0.0 {
        brain.dt_ms = dt_ms;
    }

    let mut total_spikes: u32 = 0;

    for _ in 0..steps {
        // 1. Set I_buf = external_I
        let n_use = n.min(brain.state.I_buf.len());
        if !external_I.is_null() {
            let ext = unsafe { std::slice::from_raw_parts(external_I, n_use) };
            brain.state.I_buf[..n_use].copy_from_slice(&ext[..n_use]);
            for i in n_use..brain.state.I_buf.len() {
                brain.state.I_buf[i] = 0.0;
            }
        } else {
            brain.state.I_buf.fill(0.0);
        }

        // 2. Neuron step
        match &mut brain.params {
            BrainParams::Lif(storage) => {
                if (storage.packed[0] - brain.dt_ms).abs() > 1e-8 {
                    update_dt_in_packed(&mut storage.packed, brain.dt_ms, brain.n_padded);
                }
                safe::lif_step(
                    &mut brain.state.v,
                    &mut brain.state.I_buf,
                    &mut brain.state.spikes,
                    &storage.packed,
                    &mut brain.state.refractory,
                    storage.tau_ref,
                    None,
                );
            }
            BrainParams::Izhikevich(storage) => {
                safe::izhikevich_step(
                    &mut brain.state.v,
                    &mut brain.state.u,
                    &mut brain.state.trace,
                    &storage.a,
                    &storage.b,
                    &mut brain.state.I_buf,
                    brain.dt_ms,
                    TRACE_DECAY,
                    &mut brain.state.spikes,
                    &storage.c,
                    &storage.d,
                    &mut brain.state.refractory,
                    storage.tau_ref,
                    None,
                );
            }
        }

        // 3. Push spikes to queue
        let spike_count = push_spikes_to_queue(
            &brain.state.spikes, n_neurons, &mut brain.state.queue,
        );
        total_spikes += spike_count;

        // 4. CSR propagate
        csr_propagate_from_queue(
            &conn.data.row_ptr,
            &conn.data.col_indices,
            &conn.data.values,
            &conn.data.nt_types,
            &mut brain.state.queue,
            &mut brain.state.I_buf,
            1.0,
        );
    }

    total_spikes.min(i32::MAX as u32) as i32
}

// ═══════════════════════════════════════════════════════════
//  State readers
// ═══════════════════════════════════════════════════════════

/// Copy membrane potentials into `out` buffer.
/// Returns number of elements copied, or negative error.
#[no_mangle]
pub extern "C" fn brain_get_v(brain: *mut c_void, out: *mut f32, max_n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if out.is_null() || max_n <= 0 {
        return ERR_PARAM;
    }
    let n = brain.n_neurons.min(max_n as usize);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    out_slice.copy_from_slice(&brain.state.v[..n]);
    n as i32
}

/// Copy recovery variables into `out` buffer.
/// Returns number of elements copied, or negative error.
#[no_mangle]
pub extern "C" fn brain_get_u(brain: *mut c_void, out: *mut f32, max_n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if out.is_null() || max_n <= 0 {
        return ERR_PARAM;
    }
    let n = brain.n_neurons.min(max_n as usize);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    out_slice.copy_from_slice(&brain.state.u[..n]);
    n as i32
}

/// Copy spike flags (0/1) into `out` buffer.
/// Returns number of elements copied, or negative error.
#[no_mangle]
pub extern "C" fn brain_get_spikes(brain: *mut c_void, out: *mut i32, max_n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if out.is_null() || max_n <= 0 {
        return ERR_PARAM;
    }
    let n = brain.n_neurons.min(max_n as usize);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    out_slice.copy_from_slice(&brain.state.spikes[..n]);
    n as i32
}

/// Copy input current buffer into `out`.
/// Returns number of elements copied, or negative error.
#[no_mangle]
pub extern "C" fn brain_get_I_buf(brain: *mut c_void, out: *mut f32, max_n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if out.is_null() || max_n <= 0 {
        return ERR_PARAM;
    }
    let n = brain.n_neurons.min(max_n as usize);
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    out_slice.copy_from_slice(&brain.state.I_buf[..n]);
    n as i32
}

// ═══════════════════════════════════════════════════════════
//  Param setters
// ═══════════════════════════════════════════════════════════

/// Override LIF parameters for all/some neurons.
///
/// Pass NULL for any param to keep current value.
/// `n` specifies the length of provided arrays (can be < n_neurons, remaining keep defaults).
/// Returns 0 on success, negative error otherwise.
#[no_mangle]
pub extern "C" fn brain_set_lif_params(
    brain: *mut c_void,
    tau_m: *const f32,
    v_rest: *const f32,
    v_thresh: *const f32,
    v_reset: *const f32,
    n: i32,
    tau_ref: f32,
    dt_ms: f32,
) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };

    let storage = match &mut brain.params {
        BrainParams::Lif(s) => s,
        _ => return ERR_MODEL,
    };

    let nn = if n > 0 { n as usize } else { brain.n_neurons };
    let nn = nn.min(brain.n_neurons);

    let tau_m_arr = if !tau_m.is_null() {
        Some(unsafe { std::slice::from_raw_parts(tau_m, nn) })
    } else {
        None
    };
    let v_rest_arr = if !v_rest.is_null() {
        Some(unsafe { std::slice::from_raw_parts(v_rest, nn) })
    } else {
        None
    };
    let v_thresh_arr = if !v_thresh.is_null() {
        Some(unsafe { std::slice::from_raw_parts(v_thresh, nn) })
    } else {
        None
    };
    let v_reset_arr = if !v_reset.is_null() {
        Some(unsafe { std::slice::from_raw_parts(v_reset, nn) })
    } else {
        None
    };

    // Read current params from packed array
    let mut cur_tau_m = vec![DEFAULT_TAU_M; brain.n_neurons];
    let mut cur_v_rest = vec![DEFAULT_V_REST; brain.n_neurons];
    let mut cur_v_thresh = vec![DEFAULT_V_THRESH; brain.n_neurons];
    let mut cur_v_reset = vec![DEFAULT_V_RESET; brain.n_neurons];

    for i in 0..brain.n_neurons {
        let idx = (i % brain.n_padded) as usize;
        cur_tau_m[i] = storage.packed[brain.n_padded + idx];
        cur_v_rest[i] = storage.packed[2 * brain.n_padded + idx];
        cur_v_thresh[i] = storage.packed[3 * brain.n_padded + idx];
        cur_v_reset[i] = storage.packed[4 * brain.n_padded + idx];
    }

    // Override with provided arrays
    if let Some(arr) = tau_m_arr {
        for i in 0..nn.min(arr.len()) { cur_tau_m[i] = arr[i]; }
    }
    if let Some(arr) = v_rest_arr {
        for i in 0..nn.min(arr.len()) { cur_v_rest[i] = arr[i]; }
    }
    if let Some(arr) = v_thresh_arr {
        for i in 0..nn.min(arr.len()) { cur_v_thresh[i] = arr[i]; }
    }
    if let Some(arr) = v_reset_arr {
        for i in 0..nn.min(arr.len()) { cur_v_reset[i] = arr[i]; }
    }

    if tau_ref > 0.0 { storage.tau_ref = tau_ref; }
    let dt = if dt_ms > 0.0 { dt_ms } else { brain.dt_ms };
    brain.dt_ms = dt;

    // Repack
    storage.packed = pack_lif(&cur_tau_m, &cur_v_rest, &cur_v_thresh, &cur_v_reset, dt, brain.n_padded);

    OK
}

/// Override Izhikevich parameters for all/some neurons.
///
/// Pass NULL for any param to keep current value.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn brain_set_izhikevich_params(
    brain: *mut c_void,
    a: *const f32,
    b: *const f32,
    c: *const f32,
    d: *const f32,
    n: i32,
    tau_ref: f32,
    dt_ms: f32,
) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };

    let storage = match &mut brain.params {
        BrainParams::Izhikevich(s) => s,
        _ => return ERR_MODEL,
    };

    let nn = if n > 0 { n as usize } else { brain.n_neurons };
    let nn = nn.min(brain.n_neurons);

    if !a.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(a, nn) };
        for i in 0..nn { storage.a[i] = arr[i]; }
    }
    if !b.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(b, nn) };
        for i in 0..nn { storage.b[i] = arr[i]; }
    }
    if !c.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(c, nn) };
        for i in 0..nn { storage.c[i] = arr[i]; }
    }
    if !d.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(d, nn) };
        for i in 0..nn { storage.d[i] = arr[i]; }
    }

    if tau_ref > 0.0 { storage.tau_ref = tau_ref; }
    if dt_ms > 0.0 { brain.dt_ms = dt_ms; }

    OK
}

// ═══════════════════════════════════════════════════════════
//  State writers
// ═══════════════════════════════════════════════════════════

/// Write membrane potentials `v` directly into the brain state.
/// Useful for setting initial conditions, injecting perturbations, or restoring checkpoints.
/// Returns number of elements written, or negative error.
#[no_mangle]
pub extern "C" fn brain_write_v(brain: *mut c_void, src: *const f32, n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if src.is_null() || n <= 0 {
        return ERR_PARAM;
    }
    let nn = (n as usize).min(brain.n_neurons);
    let slice = unsafe { std::slice::from_raw_parts(src, nn) };
    brain.state.v[..nn].copy_from_slice(slice);
    nn as i32
}

/// Write recovery variables `u` directly into the brain state.
/// Only meaningful for Izhikevich models (LIF ignores u after writes).
/// Returns number of elements written, or negative error.
#[no_mangle]
pub extern "C" fn brain_write_u(brain: *mut c_void, src: *const f32, n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if src.is_null() || n <= 0 {
        return ERR_PARAM;
    }
    let nn = (n as usize).min(brain.n_neurons);
    let slice = unsafe { std::slice::from_raw_parts(src, nn) };
    brain.state.u[..nn].copy_from_slice(slice);
    nn as i32
}

/// Write input current buffer `I_buf` directly.
/// Use this for complex input patterns (not just uniform arrays).
/// Returns number of elements written, or negative error.
#[no_mangle]
pub extern "C" fn brain_write_I_buf(brain: *mut c_void, src: *const f32, n: i32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if src.is_null() || n <= 0 {
        return ERR_PARAM;
    }
    let nn = (n as usize).min(brain.n_neurons);
    let slice = unsafe { std::slice::from_raw_parts(src, nn) };
    brain.state.I_buf[..nn].copy_from_slice(slice);
    nn as i32
}

/// Inject current into a single neuron.
/// Adds `current` to `I_buf[neuron_idx]`.
/// Returns 0 on success, negative error otherwise.
#[no_mangle]
pub extern "C" fn brain_inject_current(brain: *mut c_void, neuron_idx: i32, current: f32) -> i32 {
    let brain = match as_brain(brain) {
        Some(b) => b,
        None => return ERR_MAGIC,
    };
    if neuron_idx < 0 || neuron_idx as usize >= brain.n_neurons {
        return ERR_PARAM;
    }
    brain.state.I_buf[neuron_idx as usize] += current;
    OK
}

// ═══════════════════════════════════════════════════════════
//  Reset
// ═══════════════════════════════════════════════════════════

/// Reset brain state to initial values with small noise.
/// Clears v, u, trace, spikes, refractory counters AND the spike queue.
#[no_mangle]
pub extern "C" fn brain_reset(ptr: *mut c_void) {
    if let Some(brain) = as_brain(ptr) {
        let new_state = BrainState::new(brain.n_neurons, brain.model);
        brain.state = new_state;
        brain.state.reset_with_noise(brain.n_neurons);
        // Queue is already cleared by BrainState::new
    }
}

// ═══════════════════════════════════════════════════════════
//  Info
// ═══════════════════════════════════════════════════════════

/// Get number of neurons in the brain.
#[no_mangle]
pub extern "C" fn brain_n_neurons(ptr: *mut c_void) -> i32 {
    match as_brain(ptr) {
        Some(b) => b.n_neurons as i32,
        None => ERR_MAGIC,
    }
}

/// Get number of neurons in the connectome.
#[no_mangle]
pub extern "C" fn connectome_n_neurons(ptr: *mut c_void) -> i32 {
    match as_connectome(ptr) {
        Some(c) => c.data.n_neurons as i32,
        None => ERR_MAGIC,
    }
}

/// Get number of synapses in the connectome.
#[no_mangle]
pub extern "C" fn connectome_n_synapses(ptr: *mut c_void) -> i32 {
    match as_connectome(ptr) {
        Some(c) => c.data.n_synapses as i32,
        None => ERR_MAGIC,
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Create a simple ring connectome: i → (i+1)%n
    fn make_ring_connectome(n: usize) -> Box<ConnectomeHandle> {
        let nn = n;
        let mut rp = vec![0i32; nn + 1];
        let mut ci = Vec::with_capacity(n);
        let vals: Vec<f32> = vec![1.0; n];
        let nt: Vec<f32> = vec![1.0; n];

        for i in 0..nn {
            ci.push(((i + 1) % nn) as i32);
            rp[i + 1] = (i + 1) as i32;
        }

        Box::new(ConnectomeHandle {
            magic: MAGIC_CONN,
            data: ConnectomeData {
                n_neurons: nn,
                n_synapses: nn,
                row_ptr: rp,
                col_indices: ci,
                values: vals,
                nt_types: nt,
                receptor_types: vec![],
                groups: vec![],
            },
        })
    }

    // ─── Tests for new flexible API ───────────────────────

    #[test]
    fn test_brain_step_ex_neuron_only() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let I: Vec<f32> = vec![30.0; n];
        // Set input + neuron step only (no push, no propagate)
        let spikes = brain_step_ex(brain, ptr::null_mut(), I.as_ptr(), 0.5,
            STEP_SET_INPUT | STEP_NEURON);
        assert!(spikes >= 0);

        let mut v_buf = vec![0.0f32; n];
        brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        // v should have changed from resting
        assert!((v_buf[0] - (-52.0)).abs() > 0.01,
            "v should have changed after neuron step");

        brain_free(brain);
    }

    #[test]
    fn test_brain_step_ex_propagate() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        // Full step via brain_step_ex
        let spikes = brain_step_ex(brain, conn, I.as_ptr(), 0.5, STEP_FULL);
        assert!(spikes >= 0);

        let mut v_buf = vec![0.0f32; n];
        brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        assert!((v_buf[0] - (-52.0)).abs() > 0.01);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_step_n() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        // Run 50 steps in one call
        let total = brain_step_n(brain, conn, I.as_ptr(), 0.5, 50);
        assert!(total > 0, "expected spikes after 50 batch steps");

        let mut spike_buf = vec![0i32; n];
        brain_get_spikes(brain, spike_buf.as_mut_ptr(), n as i32);
        let last_spikes: i32 = spike_buf.iter().sum();
        assert!(last_spikes >= 0);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_write_v() {
        let n = 100usize;
        let model = CString::new("izhikevich").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let src: Vec<f32> = vec![-60.0; n];
        let written = brain_write_v(brain, src.as_ptr(), n as i32);
        assert_eq!(written, n as i32);

        let mut out = vec![0.0f32; n];
        brain_get_v(brain, out.as_mut_ptr(), n as i32);
        assert!((out[0] - (-60.0)).abs() < 1.0,
            "v after write should be -60, got {}", out[0]);

        brain_free(brain);
    }

    #[test]
    fn test_brain_write_u() {
        let n = 100usize;
        let model = CString::new("izhikevich").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let src: Vec<f32> = vec![5.0; n];
        let written = brain_write_u(brain, src.as_ptr(), n as i32);
        assert_eq!(written, n as i32);

        let mut out = vec![0.0f32; n];
        brain_get_u(brain, out.as_mut_ptr(), n as i32);
        assert!((out[0] - 5.0).abs() < 0.01,
            "u after write should be 5, got {}", out[0]);

        brain_free(brain);
    }

    #[test]
    fn test_brain_write_I_buf() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let src: Vec<f32> = vec![50.0; n];
        let written = brain_write_I_buf(brain, src.as_ptr(), n as i32);
        assert_eq!(written, n as i32);

        let mut out = vec![0.0f32; n];
        brain_get_I_buf(brain, out.as_mut_ptr(), n as i32);
        assert!((out[0] - 50.0).abs() < 0.01,
            "I_buf after write should be 50, got {}", out[0]);

        brain_free(brain);
    }

    #[test]
    fn test_brain_inject_current() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        // Inject current into neuron 42
        let ret = brain_inject_current(brain, 42, 100.0);
        assert_eq!(ret, OK);

        // Read I_buf and check
        let mut out = vec![0.0f32; n];
        brain_get_I_buf(brain, out.as_mut_ptr(), n as i32);
        assert!((out[42] - 100.0).abs() < 0.01,
            "I_buf[42] should be 100, got {}", out[42]);
        assert!((out[0]).abs() < 0.01,
            "I_buf[0] should be 0, got {}", out[0]);

        // Invalid neuron index
        let ret = brain_inject_current(brain, -1, 10.0);
        assert_eq!(ret, ERR_PARAM);
        let ret = brain_inject_current(brain, n as i32, 10.0);
        assert_eq!(ret, ERR_PARAM);

        brain_free(brain);
    }

    #[test]
    fn test_brain_step_ex_compose_pipeline() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;
        let I: Vec<f32> = vec![30.0; n];

        // Step 1: set input + neuron step
        let s1 = brain_step_ex(brain, conn, I.as_ptr(), 0.5, STEP_SET_INPUT | STEP_NEURON);
        assert!(s1 >= 0);

        // Read spikes between neuron step and propagate
        let mut spike_buf = vec![0i32; n];
        let n_spikes = brain_get_spikes(brain, spike_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_spikes, n as i32);

        // Step 2: push + propagate (no new input, no neuron step)
        let s2 = brain_step_ex(brain, conn, ptr::null(), 0.5, STEP_PUSH | STEP_PROPAGATE);
        assert!(s2 >= 0);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_reset_clears_queue() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        brain_step(brain, conn, I.as_ptr(), 0.5);

        brain_reset(brain);

        // After reset, queue should be empty and v back to near-resting
        let mut v_buf = vec![0.0f32; n];
        brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        assert!((v_buf[0] - (-52.0)).abs() < 5.0,
            "v after reset should be near -52, got {}", v_buf[0]);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_step_n_performance() {
        let n = 100usize;  // keep small to avoid queue overflow (cap = max(256, n/4))
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        let total = brain_step_n(brain, conn, I.as_ptr(), 0.5, 100);
        assert!(total > 0);

        connectome_free(conn);
        brain_free(brain);
    }

    // ─── Existing tests ───────────────────────────────────

    #[test]
    fn test_brain_new_free() {
        let model = CString::new("lif").unwrap();
        let ptr = brain_new(100, model.as_ptr());
        assert!(!ptr.is_null(), "brain_new returned null");
        assert_eq!(brain_n_neurons(ptr), 100);
        brain_free(ptr);
    }

    #[test]
    fn test_brain_new_izhikevich() {
        let model = CString::new("izhikevich").unwrap();
        let ptr = brain_new(1000, model.as_ptr());
        assert!(!ptr.is_null());
        assert_eq!(brain_n_neurons(ptr), 1000);
        brain_free(ptr);
    }

    #[test]
    fn test_brain_new_invalid_model() {
        let model = CString::new("hh").unwrap();
        let ptr = brain_new(100, model.as_ptr());
        assert!(ptr.is_null(), "expected null for invalid model");
    }

    #[test]
    fn test_connectome_load_free() {
        let n = 10usize;
        let nn = n;
        let rp: Vec<i32> = (0..=nn).map(|x| x as i32).collect();
        let ci: Vec<i32> = (1..=nn).map(|i| (i % nn) as i32).collect();
        let vals: Vec<f32> = vec![0.5; nn];
        let nt: Vec<f32> = vec![1.0; nn];

        let ptr = connectome_load(rp.as_ptr(), n as i32, ci.as_ptr(), vals.as_ptr(), nt.as_ptr(), n as i32);
        assert!(!ptr.is_null());
        assert_eq!(connectome_n_neurons(ptr), n as i32);
        assert_eq!(connectome_n_synapses(ptr), n as i32);
        connectome_free(ptr);
    }

    #[test]
    fn test_brain_step_lif() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        // Strong input to all neurons
        let I: Vec<f32> = vec![30.0; n];
        let spikes = brain_step(brain, conn, I.as_ptr(), 0.5);
        assert!(spikes >= 0, "brain_step returned error: {spikes}");

        // Read spikes
        let mut spike_buf = vec![0i32; n];
        let n_read = brain_get_spikes(brain, spike_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_read, n as i32);
        let total: i32 = spike_buf.iter().sum();
        assert!(total >= 0);

        // Read v
        let mut v_buf = vec![0.0f32; n];
        let n_v = brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_v, n as i32);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_step_izhikevich() {
        let n = 100usize;
        let model = CString::new("izhikevich").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        let spikes = brain_step(brain, conn, I.as_ptr(), 0.5);
        assert!(spikes >= 0);

        let mut v_buf = vec![0.0f32; n];
        let n_v = brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_v, n as i32);

        // After one Izhikevich step with strong I, v should have increased from -65
        assert!(v_buf[0] > -70.0, "v[0] should have increased, got {}", v_buf[0]);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_multiple_steps() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        let mut total_spikes = 0i32;

        for _ in 0..50 {
            total_spikes += brain_step(brain, conn, I.as_ptr(), 0.5);
        }

        assert!(total_spikes > 0, "expected spikes after 50 steps, got {total_spikes}");

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_reset() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![0.0; n];
        let _ = brain_step(brain, conn, I.as_ptr(), 0.5);

        // Read v before reset
        let mut v_before = vec![0.0f32; n];
        brain_get_v(brain, v_before.as_mut_ptr(), n as i32);

        brain_reset(brain);

        // Read v after reset — should be back to -52 (LIF resting)
        let mut v_after = vec![0.0f32; n];
        brain_get_v(brain, v_after.as_mut_ptr(), n as i32);
        assert!((v_after[0] - (-52.0)).abs() < 5.0,
            "v after reset should be near -52, got {}", v_after[0]);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_null_external_I() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        // Null external_I should work (zeros)
        let spikes = brain_step(brain, conn, ptr::null_mut(), 0.5);
        assert!(spikes >= 0);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_invalid_handles() {
        assert!(brain_new(-1, ptr::null_mut()).is_null());

        let model = CString::new("lif").unwrap();
        let brain = brain_new(10i32, model.as_ptr());
        assert!(!brain.is_null());

        // Step with null connectome
        assert_eq!(brain_step(brain, ptr::null_mut(), ptr::null(), 0.5), ERR_MAGIC);

        brain_free(brain);
    }

    #[test]
    fn test_set_lif_params() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let tau_m: Vec<f32> = vec![20.0; n];
        let v_rest: Vec<f32> = vec![-65.0; n];
        let v_thresh: Vec<f32> = vec![-50.0; n];
        let v_reset: Vec<f32> = vec![-65.0; n];

        let ret = brain_set_lif_params(
            brain,
            tau_m.as_ptr(), v_rest.as_ptr(), v_thresh.as_ptr(), v_reset.as_ptr(),
            n as i32, 3.0, 1.0,
        );
        assert_eq!(ret, OK);

        brain_free(brain);
    }

    #[test]
    fn test_set_izhikevich_params() {
        let n = 100usize;
        let model = CString::new("izhikevich").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let a: Vec<f32> = vec![0.01; n];
        let b: Vec<f32> = vec![0.3; n];
        let c: Vec<f32> = vec![-70.0; n];
        let d: Vec<f32> = vec![10.0; n];

        let ret = brain_set_izhikevich_params(
            brain, a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr(),
            n as i32, 1.0, 0.5,
        );
        assert_eq!(ret, OK);

        brain_free(brain);
    }

    #[test]
    fn test_set_wrong_model_params() {
        let model = CString::new("lif").unwrap();
        let brain = brain_new(10i32, model.as_ptr());
        assert!(!brain.is_null());

        // Set Izhikevich params on LIF brain → should fail
        let a: Vec<f32> = vec![0.02; 10usize];
        let ret = brain_set_izhikevich_params(
            brain, a.as_ptr(), ptr::null(), ptr::null(), ptr::null(),
            10, 0.0, 0.0,
        );
        assert_eq!(ret, ERR_MODEL);

        brain_free(brain);
    }

    #[test]
    fn test_null_connectome() {
        let model = CString::new("lif").unwrap();
        let brain = brain_new(10i32, model.as_ptr());
        assert!(!brain.is_null());

        assert!(connectome_load(ptr::null_mut(), 10, ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), 0).is_null());
        assert_eq!(connectome_n_neurons(ptr::null_mut()), ERR_MAGIC);

        brain_free(brain);
    }
}
