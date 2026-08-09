//! # Unity C-ABI adapter for TaaS neural engine.
//!
//! ## C API (P/Invoke compatible)
//!
//! ```c
//! // Lifecycle
//! void* brain_new(int n_neurons, const char* model);     // "lif" | "izhikevich" | "rate" | "gco" | "hh"
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
//! ## Step pipeline (`brain_step` / `brain_step_ex`)
//!
//! `brain_step` always runs the full pipeline:
//! 1. `STEP_SET_INPUT` — copy `external_I` into `I_buf` (zeros if NULL)
//! 2. `STEP_NEURON` — one SIMD neuron step (LIF/Izhikevich/Rate/GCO/HH)
//! 3. `STEP_PUSH` — push emitted spikes into the CSR queue
//! 4. `STEP_PROPAGATE` — distribute currents to post-synaptic neurons
//!
//! `brain_step_ex` lets you compose these stages with the `STEP_*` flags.
//! **Invariant**: `STEP_PROPAGATE` implies `STEP_PUSH` — if you request
//! propagation without push, the queue is empty and nothing is delivered.
//! The flag is normalized automatically (PUSH is OR-ed in).
//!
//! ## C# usage (Unity)
//!
//! ```csharp
//! [DllImport("taas_checkmate")] static extern IntPtr brain_new(int n, string model);
//! [DllImport("taas_checkmate")] static extern void   brain_free(IntPtr brain);
//! [DllImport("taas_checkmate")] static extern int    brain_step(IntPtr brain, IntPtr conn, float[] I, float dt);
//!
//! IntPtr brain = brain_new(100000, "lif");
//! IntPtr conn = connectome_load(rp, n, ci, vals, nt, n_syn);
//! int spikes = brain_step(brain, conn, I, 0.5f);
//! ```

#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;

use crate::connectome::{csr_propagate_from_queue, push_spikes_to_queue};
use crate::connectome::{hh_population_standard, BrainState, ConnectomeData, NeuronModel};
use pillars::core::NeuralPopulation;
use pillars::kernels::phase::gco_step;
use pillars::zoo::{HHModel, HHParamIdx};

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
pub const STEP_FULL: i32 = 15; // all stages

// Default LIF params
const DEFAULT_TAU_M: f32 = 10.0;
const DEFAULT_V_REST: f32 = -70.0;
const DEFAULT_V_THRESH: f32 = -52.0;
const DEFAULT_V_RESET: f32 = -70.0;

// ═══════════════════════════════════════════════════════════
//  Opaque handle types
// ═══════════════════════════════════════════════════════════

/// Opaque handle for a brain simulation (LIF or Izhikevich).
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
    Rate(RateStorage),
    /// GCO — phase oscillator driven by the CSR Hamiltonian; the speed
    /// vector is stored once here (length n_padded).
    Gco(Vec<f32>),
    /// HH — a pillars `NeuralPopulation<HHModel>` with the standard
    /// squid-axon channel set (see `connectome::sim::hh_population_standard`).
    Hh(NeuralPopulation<HHModel>),
}

/// LIF model storage — packed params matching SIMD kernel layout.
struct LifStorage {
    packed: Vec<f32>, // [n_padded * 5]
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

/// Rate model storage — first-order low-pass on clamped input.
struct RateStorage {
    /// Membrane time constant (ms).
    tau: f32,
}

/// Default rate-model time constant (ms).
const DEFAULT_RATE_TAU: f32 = 10.0;

/// Default GCO integration speed (rad/s).
const DEFAULT_GCO_SPEED: f32 = 1.0;

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
fn pack_lif(
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    dt_ms: f32,
    n_padded: usize,
) -> Vec<f32> {
    let n = tau_m
        .len()
        .min(v_rest.len())
        .min(v_thresh.len())
        .min(v_reset.len());
    let mut packed = Vec::with_capacity(n_padded * 5);
    for i in 0..n_padded {
        let idx = if i < n { i } else { n - 1 };
        packed.push(dt_ms); // slot 0: dt
        packed.push(tau_m[idx]); // slot 1: tau_m
        packed.push(v_rest[idx]); // slot 2: v_rest
        packed.push(v_thresh[idx]); // slot 3: v_thresh
        packed.push(v_reset[idx]); // slot 4: v_reset
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
            let packed = pack_lif(
                &tau_m,
                &v_rest,
                &v_thresh,
                &v_reset,
                DEFAULT_DT_MS,
                n_padded,
            );
            BrainParams::Lif(LifStorage {
                packed,
                tau_ref: DEFAULT_TAU_REF,
            })
        }
        NeuronModel::Izhikevich => BrainParams::Izhikevich(IzhikevichStorage {
            a: pad_to(&[DEFAULT_A], n_padded),
            b: pad_to(&[DEFAULT_B], n_padded),
            c: pad_to(&[DEFAULT_C], n_padded),
            d: pad_to(&[DEFAULT_D], n_padded),
            tau_ref: DEFAULT_TAU_REF,
        }),
        NeuronModel::Rate => BrainParams::Rate(RateStorage {
            tau: DEFAULT_RATE_TAU,
        }),
        NeuronModel::Gco => BrainParams::Gco(vec![DEFAULT_GCO_SPEED; n_padded]),
        NeuronModel::Hh => BrainParams::Hh(hh_population_standard(n_neurons, DEFAULT_DT_MS)),
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
    if ptr.is_null() {
        return None;
    }
    let brain = unsafe { &mut *(ptr as *mut BrainHandle) };
    if brain.magic != MAGIC_BRAIN {
        return None;
    }
    Some(brain)
}

/// Validate a connectome handle pointer.
fn as_connectome<'a>(ptr: *mut c_void) -> Option<&'a ConnectomeHandle> {
    if ptr.is_null() {
        return None;
    }
    let conn = unsafe { &*(ptr as *const ConnectomeHandle) };
    if conn.magic != MAGIC_CONN {
        return None;
    }
    Some(conn)
}

/// Run the neuron step for the brain's active model.
///
/// `conn` is only needed for `Gco` (the CSR Hamiltonian H·ψ is built from the
/// connectome); other models ignore it. After the step, `state.spikes` holds
/// the emitted spikes for the current tick.
fn step_neuron(brain: &mut BrainHandle, conn: Option<&ConnectomeData>) {
    let n_neurons = brain.n_neurons;
    match &mut brain.params {
        BrainParams::Lif(storage) => {
            if (storage.packed[0] - brain.dt_ms).abs() > 1e-8 {
                update_dt_in_packed(&mut storage.packed, brain.dt_ms, brain.n_padded);
            }
            crate::connectome::step_lif_pillars(
                &mut brain.state.v,
                &mut brain.state.I_buf,
                &mut brain.state.spikes,
                &storage.packed,
                &mut brain.state.refractory,
                storage.tau_ref,
            );
        }
        BrainParams::Izhikevich(storage) => {
            crate::connectome::step_izhikevich_pillars(
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
            );
        }
        BrainParams::Rate(storage) => {
            // First-order low-pass on clamped input (pillars RateModel):
            //   r[t] = exp(-dt/τ)·r[t-1] + clamp01(I)·(1 - exp(-dt/τ))
            // spike emitted when r > 0.5 (dt in seconds).
            let dt_s = brain.dt_ms / 1000.0;
            let decay = (-dt_s / (storage.tau / 1000.0).max(1e-6)).exp();
            for i in 0..n_neurons {
                let input = brain.state.I_buf[i].clamp(0.0, 1.0);
                brain.state.v[i] = decay * brain.state.v[i] + input * (1.0 - decay);
                brain.state.spikes[i] = if brain.state.v[i] > 0.5 { 1 } else { 0 };
            }
        }
        BrainParams::Gco(speed) => {
            // H·ψ from CSR (same construction as connectome::sim_gco):
            //   h_real[i] = Σ W[j,i]·u[j], h_imag[i] = -Σ W[j,i]·v[j]
            let mut h_real = vec![0.0f32; brain.n_padded];
            let mut h_imag = vec![0.0f32; brain.n_padded];
            if let Some(c) = conn {
                let max_syn = c.col_indices.len().min(c.values.len());
                for pre in 0..n_neurons {
                    let start = c.row_ptr[pre].max(0) as usize;
                    let end = (c.row_ptr[pre + 1] as usize).min(max_syn);
                    let mult = if pre < c.nt_types.len() {
                        c.nt_types[pre]
                    } else {
                        1.0
                    };
                    for j in start..end {
                        let post = c.col_indices[j] as usize;
                        if post < brain.n_padded {
                            h_real[post] += c.values[j] * mult * brain.state.u[pre];
                            h_imag[post] -= c.values[j] * mult * brain.state.v[pre];
                        }
                    }
                }
            }
            gco_step(
                &mut brain.state.v,
                &mut brain.state.u,
                &h_real,
                &h_imag,
                &brain.state.I_buf,
                speed,
                brain.dt_ms / 1000.0,
            );
            // "Spike" when |v + iu|² exceeds the threshold (same as sim_gco).
            for i in 0..n_neurons {
                let prob =
                    brain.state.v[i] * brain.state.v[i] + brain.state.u[i] * brain.state.u[i];
                brain.state.spikes[i] = if prob > 0.5 { 1 } else { 0 };
            }
        }
        BrainParams::Hh(pop) => {
            // The population packs `dt` into its params at construction; keep
            // it in sync with the current step size (brain.dt_ms) so changing
            // dt at runtime doesn't silently integrate with a stale dt.
            let dt_slice = pop.param_slice_mut(HHParamIdx::DT.into());
            if dt_slice[0] != brain.dt_ms {
                dt_slice.fill(brain.dt_ms);
            }
            // `NeuralPopulation::step` asserts `i_in.len() == pop.n == n_neurons`,
            // while BrainState buffers are padded to n_padded — slice them down
            // so non-multiple-of-8 counts don't panic at the C boundary.
            pop.step(
                &brain.state.I_buf[..n_neurons],
                &mut brain.state.spikes[..n_neurons],
            );
        }
    }
}

/// Validate a connectome handle pointer (mutable).
fn as_connectome_mut<'a>(ptr: *mut c_void) -> Option<&'a mut ConnectomeHandle> {
    if ptr.is_null() {
        return None;
    }
    let conn = unsafe { &mut *(ptr as *mut ConnectomeHandle) };
    if conn.magic != MAGIC_CONN {
        return None;
    }
    Some(conn)
}

// ═══════════════════════════════════════════════════════════
//  Lifecycle: brain
// ═══════════════════════════════════════════════════════════

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
        "rate" => NeuronModel::Rate,
        "gco" => NeuronModel::Gco,
        "hh" => NeuronModel::Hh,
        _ => return ptr::null_mut(),
    };

    let brain = create_brain(n_neurons as usize, nm);
    Box::into_raw(brain) as *mut c_void
}

/// Free a brain handle.
///
/// Double-free semantics: the handle's magic is zeroed on the first free, so a
/// second `brain_free` on the same pointer observes `as_brain == None` and
/// **leaks** (skips `Box::from_raw`) instead of crashing or invoking UB. This
/// is the standard trade-off for C-ABI ownership: memory-safe in the face of
/// caller bugs, at the cost of a leak on misuse. Callers should still treat
/// handles as owned-once.
#[no_mangle]
pub extern "C" fn brain_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    if as_brain(ptr).is_none() {
        // Already freed (magic zeroed) or invalid pointer — leave it alone.
        return;
    }
    unsafe {
        let mut handle = Box::from_raw(ptr as *mut BrainHandle);
        handle.magic = 0;
    }
}

// ═══════════════════════════════════════════════════════════
//  Lifecycle: connectome
// ═══════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn connectome_load(
    row_ptr: *const i32,
    n_neurons: i32,
    col_indices: *const i32,
    values: *const f32,
    nt_types: *const f32,
    n_synapses: i32,
) -> *mut c_void {
    // An empty graph (n_synapses == 0, isolated population) is valid: only the
    // row_ptr is required then, col_indices/values may be NULL.
    if row_ptr.is_null() || n_neurons <= 0 || n_synapses < 0 {
        return ptr::null_mut();
    }

    let nn = n_neurons as usize;
    let ns = n_synapses as usize;

    let rp = unsafe { std::slice::from_raw_parts(row_ptr, nn + 1) }.to_vec();

    if rp[0] != 0 || rp[nn] as usize != ns {
        return ptr::null_mut();
    }

    // For ns > 0 the CSR arrays must be present; for ns == 0 they may be NULL.
    if ns > 0 && (col_indices.is_null() || values.is_null()) {
        return ptr::null_mut();
    }

    let ci = if ns > 0 {
        unsafe { std::slice::from_raw_parts(col_indices, ns) }.to_vec()
    } else {
        vec![]
    };
    let vl = if ns > 0 {
        unsafe { std::slice::from_raw_parts(values, ns) }.to_vec()
    } else {
        vec![]
    };
    let nt = if nt_types.is_null() {
        vec![1.0f32; nn]
    } else {
        unsafe { std::slice::from_raw_parts(nt_types, nn) }.to_vec()
    };

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

#[no_mangle]
pub extern "C" fn connectome_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    if let Some(conn) = as_connectome_mut(ptr) {
        conn.magic = 0;
    }
    unsafe {
        let _ = Box::from_raw(ptr as *mut ConnectomeHandle);
    }
}

// ═══════════════════════════════════════════════════════════
//  Step
// ═══════════════════════════════════════════════════════════

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
    if n != brain.n_neurons {
        return ERR_PARAM;
    }

    brain.dt_ms = if dt_ms > 0.0 { dt_ms } else { brain.dt_ms };

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

    step_neuron(brain, Some(&conn.data));

    let spike_count = push_spikes_to_queue(&brain.state.spikes, n_neurons, &mut brain.state.queue);

    csr_propagate_from_queue(
        &conn.data.row_ptr,
        &conn.data.col_indices,
        &conn.data.values,
        &conn.data.nt_types,
        &mut brain.state.queue,
        &mut brain.state.I_buf,
        1.0,
    );

    spike_count.min(i32::MAX as u32) as i32
}

// ═══════════════════════════════════════════════════════════
//  Flexible step API
// ═══════════════════════════════════════════════════════════

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

    if dt_ms > 0.0 {
        brain.dt_ms = dt_ms;
    }

    let n_neurons = brain.n_neurons;

    // Invariant: propagation requires queued spikes — normalize the flags so
    // `STEP_PROPAGATE` implies `STEP_PUSH` (silent no-op prevention).
    let flags = if (flags & STEP_PROPAGATE) != 0 {
        flags | STEP_PUSH
    } else {
        flags
    };

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

    if (flags & STEP_NEURON) != 0 {
        let conn_data = as_connectome(conn).map(|c| &c.data);
        // The GCO model reads the CSR graph (row_ptr[pre+1]) inside the neuron
        // step — validate sizes here too, not only in the PROPAGATE branch.
        if let Some(c) = conn_data {
            if c.n_neurons != n_neurons {
                return ERR_PARAM;
            }
        }
        step_neuron(brain, conn_data);
    }

    let spike_count = if (flags & STEP_PUSH) != 0 {
        push_spikes_to_queue(&brain.state.spikes, n_neurons, &mut brain.state.queue)
    } else {
        brain
            .state
            .spikes
            .iter()
            .take(n_neurons)
            .filter(|&&s| s > 0)
            .count() as u32
    };

    if (flags & STEP_PROPAGATE) != 0 {
        if let Some(conn) = as_connectome(conn) {
            if conn.data.n_neurons != n_neurons {
                return ERR_PARAM;
            }
            csr_propagate_from_queue(
                &conn.data.row_ptr,
                &conn.data.col_indices,
                &conn.data.values,
                &conn.data.nt_types,
                &mut brain.state.queue,
                &mut brain.state.I_buf,
                1.0,
            );
        } else {
            // Propagation requested but no connectome: drain the queue so it
            // never accumulates across calls (spikes are counted as emitted
            // but have no targets to reach).
            brain.state.queue.clear();
        }
    }

    spike_count.min(i32::MAX as u32) as i32
}

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
    if n != brain.n_neurons {
        return ERR_PARAM;
    }

    if dt_ms > 0.0 {
        brain.dt_ms = dt_ms;
    }

    let mut total_spikes: u32 = 0;

    for _ in 0..steps {
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

        step_neuron(brain, Some(&conn.data));

        let spike_count =
            push_spikes_to_queue(&brain.state.spikes, n_neurons, &mut brain.state.queue);
        total_spikes += spike_count;

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
    // HH keeps its membrane state inside the population, not in BrainState.
    match &brain.params {
        BrainParams::Hh(pop) => {
            out_slice.copy_from_slice(&pop.state_slice(0)[..n]);
        }
        _ => {
            out_slice.copy_from_slice(&brain.state.v[..n]);
        }
    }
    n as i32
}

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
    // HH has no u-state; the recovery variable lives in BrainState only for
    // Izhikevich. Return 0s for HH (documented ABI behavior).
    match &brain.params {
        BrainParams::Hh(_) => out_slice.fill(0.0),
        _ => out_slice.copy_from_slice(&brain.state.u[..n]),
    }
    n as i32
}

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

    // Fast path: when only tau_ref/dt change (all four param arrays NULL),
    // avoid the full repack — just update the dt slot and tau_ref.
    if tau_m.is_null() && v_rest.is_null() && v_thresh.is_null() && v_reset.is_null() {
        if tau_ref > 0.0 {
            storage.tau_ref = tau_ref;
        }
        let dt = if dt_ms > 0.0 { dt_ms } else { brain.dt_ms };
        if dt != brain.dt_ms {
            brain.dt_ms = dt;
            update_dt_in_packed(&mut storage.packed, dt, brain.n_padded);
        }
        return OK;
    }

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

    if let Some(arr) = tau_m_arr {
        for i in 0..nn.min(arr.len()) {
            cur_tau_m[i] = arr[i];
        }
    }
    if let Some(arr) = v_rest_arr {
        for i in 0..nn.min(arr.len()) {
            cur_v_rest[i] = arr[i];
        }
    }
    if let Some(arr) = v_thresh_arr {
        for i in 0..nn.min(arr.len()) {
            cur_v_thresh[i] = arr[i];
        }
    }
    if let Some(arr) = v_reset_arr {
        for i in 0..nn.min(arr.len()) {
            cur_v_reset[i] = arr[i];
        }
    }

    if tau_ref > 0.0 {
        storage.tau_ref = tau_ref;
    }
    let dt = if dt_ms > 0.0 { dt_ms } else { brain.dt_ms };
    brain.dt_ms = dt;

    storage.packed = pack_lif(
        &cur_tau_m,
        &cur_v_rest,
        &cur_v_thresh,
        &cur_v_reset,
        dt,
        brain.n_padded,
    );

    OK
}

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
        for i in 0..nn {
            storage.a[i] = arr[i];
        }
    }
    if !b.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(b, nn) };
        for i in 0..nn {
            storage.b[i] = arr[i];
        }
    }
    if !c.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(c, nn) };
        for i in 0..nn {
            storage.c[i] = arr[i];
        }
    }
    if !d.is_null() {
        let arr = unsafe { std::slice::from_raw_parts(d, nn) };
        for i in 0..nn {
            storage.d[i] = arr[i];
        }
    }

    if tau_ref > 0.0 {
        storage.tau_ref = tau_ref;
    }
    if dt_ms > 0.0 {
        brain.dt_ms = dt_ms;
    }

    OK
}

// ═══════════════════════════════════════════════════════════
//  State writers
// ═══════════════════════════════════════════════════════════

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
    match &mut brain.params {
        BrainParams::Hh(pop) => pop.state_slice_mut(0)[..nn].copy_from_slice(slice),
        _ => brain.state.v[..nn].copy_from_slice(slice),
    }
    nn as i32
}

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
    // HH has no u-state — reject writes to it.
    if matches!(brain.params, BrainParams::Hh(_)) {
        return ERR_MODEL;
    }
    brain.state.u[..nn].copy_from_slice(slice);
    nn as i32
}

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

#[no_mangle]
pub extern "C" fn brain_reset(ptr: *mut c_void) {
    if let Some(brain) = as_brain(ptr) {
        // HH state lives in the population — rebuild it on reset so all gates
        // return to steady state (not just the inert BrainState.v).
        if let BrainParams::Hh(pop) = &mut brain.params {
            *pop = hh_population_standard(brain.n_neurons, brain.dt_ms);
        }
        let new_state = BrainState::new(brain.n_neurons, brain.model);
        brain.state = new_state;
        brain.state.reset_with_noise(brain.n_neurons);
    }
}

// ═══════════════════════════════════════════════════════════
//  Info
// ═══════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn brain_n_neurons(ptr: *mut c_void) -> i32 {
    match as_brain(ptr) {
        Some(b) => b.n_neurons as i32,
        None => ERR_MAGIC,
    }
}

#[no_mangle]
pub extern "C" fn connectome_n_neurons(ptr: *mut c_void) -> i32 {
    match as_connectome(ptr) {
        Some(c) => c.data.n_neurons as i32,
        None => ERR_MAGIC,
    }
}

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

    #[test]
    fn test_brain_step_ex_neuron_only() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let I: Vec<f32> = vec![30.0; n];
        let spikes = brain_step_ex(
            brain,
            ptr::null_mut(),
            I.as_ptr(),
            0.5,
            STEP_SET_INPUT | STEP_NEURON,
        );
        assert!(spikes >= 0);

        let mut v_buf = vec![0.0f32; n];
        brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        assert!(
            (v_buf[0] - (-52.0)).abs() > 0.01,
            "v should have changed after neuron step"
        );

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
        assert!(
            (out[0] - (-60.0)).abs() < 1.0,
            "v after write should be -60, got {}",
            out[0]
        );

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
        assert!(
            (out[0] - 5.0).abs() < 0.01,
            "u after write should be 5, got {}",
            out[0]
        );

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
        assert!(
            (out[0] - 50.0).abs() < 0.01,
            "I_buf after write should be 50, got {}",
            out[0]
        );

        brain_free(brain);
    }

    #[test]
    fn test_brain_inject_current() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let ret = brain_inject_current(brain, 42, 100.0);
        assert_eq!(ret, OK);

        let mut out = vec![0.0f32; n];
        brain_get_I_buf(brain, out.as_mut_ptr(), n as i32);
        assert!(
            (out[42] - 100.0).abs() < 0.01,
            "I_buf[42] should be 100, got {}",
            out[42]
        );
        assert!(
            (out[0]).abs() < 0.01,
            "I_buf[0] should be 0, got {}",
            out[0]
        );

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

        let s1 = brain_step_ex(brain, conn, I.as_ptr(), 0.5, STEP_SET_INPUT | STEP_NEURON);
        assert!(s1 >= 0);

        let mut spike_buf = vec![0i32; n];
        let n_spikes = brain_get_spikes(brain, spike_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_spikes, n as i32);

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

        let mut v_buf = vec![0.0f32; n];
        brain_get_v(brain, v_buf.as_mut_ptr(), n as i32);
        // BrainState::new initializes LIF to -58.0 + small noise from reset_with_noise
        assert!(
            (v_buf[0] - (-58.0)).abs() < 1.0,
            "v after reset should be near -58, got {}",
            v_buf[0]
        );

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_step_n_performance() {
        let n = 100usize;
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
        let model = CString::new("bogus").unwrap();
        let ptr = brain_new(100, model.as_ptr());
        assert!(ptr.is_null(), "expected null for invalid model");
    }

    #[test]
    fn test_brain_new_rate_model() {
        let model = CString::new("rate").unwrap();
        let ptr = brain_new(64, model.as_ptr());
        assert!(!ptr.is_null());
        assert_eq!(brain_n_neurons(ptr), 64);

        // Strong clamped input → r ramps above 0.5 → spikes fire.
        let I: Vec<f32> = vec![1.0; 64];
        let mut total = 0i32;
        for _ in 0..100 {
            total += brain_step_ex(ptr, ptr::null_mut(), I.as_ptr(), 5.0, STEP_FULL);
        }
        assert!(total > 0, "rate model should fire with drive 1.0");
        brain_free(ptr);
    }

    #[test]
    fn test_brain_new_gco_model() {
        let model = CString::new("gco").unwrap();
        let ptr = brain_new(64, model.as_ptr());
        assert!(!ptr.is_null());

        let conn = make_ring_connectome(64);
        let conn = Box::into_raw(conn) as *mut c_void;

        let I: Vec<f32> = vec![1.0; 64];
        let ret = brain_step(ptr, conn, I.as_ptr(), 1.0);
        assert!(ret >= 0);

        let mut v_buf = vec![0.0f32; 64];
        brain_get_v(ptr, v_buf.as_mut_ptr(), 64);
        assert!(v_buf.iter().all(|x| x.is_finite()));

        connectome_free(conn);
        brain_free(ptr);
    }

    #[test]
    fn test_brain_new_hh_non_multiple_of_8() {
        // Regression: NeuralPopulation::step asserts i_in.len() == pop.n, but
        // BrainState buffers are padded to n_padded — slicing must handle
        // non-multiple-of-8 neuron counts without panicking.
        let n = 100i32; // 100 % 8 = 4
        let model = CString::new("hh").unwrap();
        let ptr = brain_new(n, model.as_ptr());
        assert!(!ptr.is_null());

        let I: Vec<f32> = vec![15.0; n as usize];
        let ret = brain_step_ex(ptr, ptr::null_mut(), I.as_ptr(), 0.01, STEP_FULL);
        assert!(ret >= 0, "HH step should not panic, got {ret}");
        brain_free(ptr);
    }

    #[test]
    fn test_brain_new_hh_model() {
        let model = CString::new("hh").unwrap();
        let ptr = brain_new(64, model.as_ptr());
        assert!(!ptr.is_null());
        assert_eq!(brain_n_neurons(ptr), 64);

        let conn = make_ring_connectome(64);
        let conn = Box::into_raw(conn) as *mut c_void;

        // Canonical HH drive (pillars-connectome sim_hh test): dt=0.01 ms,
        // I=15 µA/cm² suprathreshold — the explicit Euler integrator is
        // numerically unstable at dt=0.05 with recurrent ring input.
        let I: Vec<f32> = vec![15.0; 64];
        let mut total = 0i32;
        for _ in 0..5000 {
            total += brain_step(ptr, conn, I.as_ptr(), 0.01);
        }
        assert!(total > 0, "HH should fire under drive 15 at dt=0.01");

        // get_v reads the population's membrane state.
        let mut v_buf = vec![0.0f32; 64];
        brain_get_v(ptr, v_buf.as_mut_ptr(), 64);
        assert!(v_buf[0].is_finite());
        // get_u returns 0s for HH (no u-state) and write_u is rejected.
        let mut u_buf = vec![99.0f32; 64];
        brain_get_u(ptr, u_buf.as_mut_ptr(), 64);
        assert_eq!(u_buf[0], 0.0);
        let src = vec![5.0f32; 4];
        assert_eq!(brain_write_u(ptr, src.as_ptr(), 4), ERR_MODEL);

        // reset rebuilds the population (v returns to -65 steady state).
        brain_reset(ptr);
        brain_get_v(ptr, v_buf.as_mut_ptr(), 64);
        assert!(
            (v_buf[0] - (-65.0)).abs() < 0.2,
            "v after reset should be near -65, got {}",
            v_buf[0]
        );

        connectome_free(conn);
        brain_free(ptr);
    }

    #[test]
    fn test_connectome_load_empty_graph() {
        // Isolated population: no synapses, col_indices/values NULL.
        let n = 32i32;
        let rp: Vec<i32> = vec![0; (n + 1) as usize]; // all rows empty
        let ptr = connectome_load(rp.as_ptr(), n, ptr::null(), ptr::null(), ptr::null(), 0);
        assert!(!ptr.is_null(), "empty graph should load");
        assert_eq!(connectome_n_neurons(ptr), n);
        assert_eq!(connectome_n_synapses(ptr), 0);
        connectome_free(ptr);
    }

    #[test]
    fn test_brain_step_size_mismatch() {
        let model = CString::new("lif").unwrap();
        let brain = brain_new(100i32, model.as_ptr());
        assert!(!brain.is_null());

        // connectome with a different neuron count → ERR_PARAM.
        let conn = make_ring_connectome(50);
        let conn = Box::into_raw(conn) as *mut c_void;

        let I: Vec<f32> = vec![0.0; 100];
        assert_eq!(brain_step(brain, conn, I.as_ptr(), 0.5), ERR_PARAM);
        assert_eq!(brain_step_n(brain, conn, I.as_ptr(), 0.5, 3), ERR_PARAM);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_brain_step_ex_propagate_implies_push() {
        let n = 100usize;
        let model = CString::new("lif").unwrap();
        let brain = brain_new(n as i32, model.as_ptr());
        assert!(!brain.is_null());

        let conn_handle = make_ring_connectome(n);
        let conn = Box::into_raw(conn_handle) as *mut c_void;

        let I: Vec<f32> = vec![30.0; n];
        // Request PROPAGATE without PUSH — the ABI normalizes it (PUSH implied).
        let spikes = brain_step_ex(brain, conn, I.as_ptr(), 0.5, STEP_PROPAGATE);
        assert!(spikes >= 0);

        connectome_free(conn);
        brain_free(brain);
    }

    #[test]
    fn test_connectome_load_free() {
        let n = 10usize;
        let nn = n;
        let rp: Vec<i32> = (0..=nn).map(|x| x as i32).collect();
        let ci: Vec<i32> = (1..=nn).map(|i| (i % nn) as i32).collect();
        let vals: Vec<f32> = vec![0.5; nn];
        let nt: Vec<f32> = vec![1.0; nn];

        let ptr = connectome_load(
            rp.as_ptr(),
            n as i32,
            ci.as_ptr(),
            vals.as_ptr(),
            nt.as_ptr(),
            n as i32,
        );
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

        let I: Vec<f32> = vec![30.0; n];
        let spikes = brain_step(brain, conn, I.as_ptr(), 0.5);
        assert!(spikes >= 0, "brain_step returned error: {spikes}");

        let mut spike_buf = vec![0i32; n];
        let n_read = brain_get_spikes(brain, spike_buf.as_mut_ptr(), n as i32);
        assert_eq!(n_read, n as i32);
        let total: i32 = spike_buf.iter().sum();
        assert!(total >= 0);

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

        assert!(
            v_buf[0] > -70.0,
            "v[0] should have increased, got {}",
            v_buf[0]
        );

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

        assert!(
            total_spikes > 0,
            "expected spikes after 50 steps, got {total_spikes}"
        );

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

        let mut v_before = vec![0.0f32; n];
        brain_get_v(brain, v_before.as_mut_ptr(), n as i32);

        brain_reset(brain);

        let mut v_after = vec![0.0f32; n];
        brain_get_v(brain, v_after.as_mut_ptr(), n as i32);
        // BrainState::new initializes LIF to -58.0 + small noise from reset_with_noise
        assert!(
            (v_after[0] - (-58.0)).abs() < 1.0,
            "v after reset should be near -58, got {}",
            v_after[0]
        );

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

        assert_eq!(
            brain_step(brain, ptr::null_mut(), ptr::null(), 0.5),
            ERR_MAGIC
        );

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
            tau_m.as_ptr(),
            v_rest.as_ptr(),
            v_thresh.as_ptr(),
            v_reset.as_ptr(),
            n as i32,
            3.0,
            1.0,
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
            brain,
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            n as i32,
            1.0,
            0.5,
        );
        assert_eq!(ret, OK);

        brain_free(brain);
    }

    #[test]
    fn test_set_wrong_model_params() {
        let model = CString::new("lif").unwrap();
        let brain = brain_new(10i32, model.as_ptr());
        assert!(!brain.is_null());

        let a: Vec<f32> = vec![0.02; 10usize];
        let ret = brain_set_izhikevich_params(
            brain,
            a.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            10,
            0.0,
            0.0,
        );
        assert_eq!(ret, ERR_MODEL);

        brain_free(brain);
    }

    #[test]
    fn test_null_connectome() {
        let model = CString::new("lif").unwrap();
        let brain = brain_new(10i32, model.as_ptr());
        assert!(!brain.is_null());

        assert!(connectome_load(
            ptr::null_mut(),
            10,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0
        )
        .is_null());
        assert_eq!(connectome_n_neurons(ptr::null_mut()), ERR_MAGIC);

        brain_free(brain);
    }
}
