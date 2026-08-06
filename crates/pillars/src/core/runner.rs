//! # Neuron Runners — Raw-Pointer SIMD Execution
//!
//! === BOUNDARY: Safe API → unsafe SoA ===
//! All raw-pointer arithmetic and `read_unaligned`/`write_unaligned`
//! for the model runner is consolidated here.
//!
//! ## Architecture
//!
//! ```text
//!   NeuralPopulation::step()   — safe API (in population.rs)
//!     ↓ delegates to
//!   run_neuron() / run_neuron_ext()   — unsafe runners (this file)
//!     ↓ SIMD inner loop
//!   load_state_simd! → M::step() → save_state_simd!
//! ```
//!
//! ## Safety
//! All functions require valid pointers for `n` elements.
//! Column stride = `n` (SoA layout: [col0×n] [col1×n] …).

use crate::core::context::{SimdContext, MAX_STATES};
use crate::core::lane::SimdLane;
use crate::zoo::Model;
use std::marker::PhantomData;
use std::ptr::NonNull;
use wide::f32x8;

// Import I_SYN_CLAMP_RANGE from population module
use super::population::I_SYN_CLAMP_RANGE;

// ═══════════════════════════════════════════════════════════════
//  SendablePtrs — Type-Safe Pointer Wrapper for Rayon
// ═══════════════════════════════════════════════════════════════

/// Wrapper for raw pointers encoded as usize to enable Send across rayon threads.
///
/// # Safety Contract
/// The caller (run_neuron_par) guarantees that:
/// 1. All pointers are valid for the entire parallel execution
/// 2. No aliasing occurs (each thread accesses disjoint memory ranges)
/// 3. The pointed-to memory outlives all parallel tasks
///
/// # Why usize?
/// Raw pointers (*mut T, *const T) are !Send by default. Encoding as usize
/// allows the closure to be Send, with the safety contract enforced by the
/// caller (run_neuron_par ensures disjoint memory access per thread).
#[derive(Copy, Clone)]
struct SendablePtrs {
    state: [usize; 6],
    i_in: usize,
    spikes: usize,
    params: usize,
}

unsafe impl Send for SendablePtrs {}
unsafe impl Sync for SendablePtrs {}

// ═══════════════════════════════════════════════════════════════
//  Memory Overlap Detection (Debug Mode)
// ═══════════════════════════════════════════════════════════════

#[cfg(debug_assertions)]
fn check_no_overlap(ranges: &[(usize, usize)]) {
    for i in 0..ranges.len() {
        for j in (i + 1)..ranges.len() {
            let (start_i, end_i) = ranges[i];
            let (start_j, end_j) = ranges[j];
            assert!(
                end_i <= start_j || end_j <= start_i,
                "Memory overlap detected: range {}=[{}, {}) overlaps with range {}=[{}, {})",
                i,
                start_i,
                end_i,
                j,
                start_j,
                end_j
            );
        }
    }
}

impl SendablePtrs {
    #[inline(always)]
    unsafe fn new(
        v: *mut f32,
        u: *mut f32,
        h: *mut f32,
        r: *mut f32,
        s: *mut f32,
        t: *mut f32,
        i_in: *const f32,
        spikes: *mut i32,
        params: *const f32,
    ) -> Self {
        Self {
            state: [
                v as usize, u as usize, h as usize, r as usize, s as usize, t as usize,
            ],
            i_in: i_in as usize,
            spikes: spikes as usize,
            params: params as usize,
        }
    }

    #[inline(always)]
    unsafe fn state_ptrs(&self) -> [*mut f32; 6] {
        [
            self.state[0] as *mut f32,
            self.state[1] as *mut f32,
            self.state[2] as *mut f32,
            self.state[3] as *mut f32,
            self.state[4] as *mut f32,
            self.state[5] as *mut f32,
        ]
    }

    #[inline(always)]
    unsafe fn i_in_ptr(&self) -> *const f32 {
        self.i_in as *const f32
    }

    #[inline(always)]
    unsafe fn spikes_ptr(&self) -> *mut i32 {
        self.spikes as *mut i32
    }

    #[inline(always)]
    unsafe fn params_ptr(&self) -> *const f32 {
        self.params as *const f32
    }
}

// ═══════════════════════════════════════════════════════════════
//  State load/save macros — loop up to M::N_STATE (no hardcoded 6)
// ═══════════════════════════════════════════════════════════════
//
//  Each macro produces/consumes a `[T; MAX_STATES]` array. The loop bound
//  `M::N_STATE` is a const, so LLVM fully unrolls it — zero runtime cost.
//  Adding a 7th (or 16th) state channel to a model requires NO changes here.

/// Load `M::N_STATE` scalar channels from SoA memory into `[f32; MAX_STATES]`.
/// `$ptr` — base pointer to v at index 0, `$cap` — capacity (column stride), `$i` — neuron index.
/// Compile-time guard: `M::N_STATE` must fit in [`MAX_STATES`](crate::core::context::MAX_STATES)
/// (inline `const {}` block — evaluated at monomorphization, may reference `M`).
#[macro_export]
macro_rules! load_state_scalar {
    ($ptr:expr, $cap:expr, $i:expr) => {{
        // Compile-time guard via inline const block (stable since 1.79):
        // `assert!` inside const context errors at compile time if M::N_STATE
        // exceeds MAX_STATES — zero runtime cost, fully optimized away.
        const {
            assert!(M::N_STATE <= $crate::core::context::MAX_STATES);
        }
        unsafe {
            let mut arr = [0.0f32; $crate::core::context::MAX_STATES];
            let mut k = 0usize;
            while k < M::N_STATE {
                arr[k] = *$ptr.add($cap * k + $i);
                k += 1;
            }
            arr
        }
    }};
}

/// Save `M::N_STATE` scalar channels from a `SimdContext<f32>` back to SoA memory.
/// Compile-time guard: `M::N_STATE` must fit in [`MAX_STATES`](crate::core::context::MAX_STATES)
/// (inline `const {}` block).
#[macro_export]
macro_rules! save_state_scalar {
    ($ptr:expr, $cap:expr, $i:expr, $ctx:expr) => {{
        const {
            assert!(M::N_STATE <= $crate::core::context::MAX_STATES);
        }
        unsafe {
            let mut k = 0usize;
            while k < M::N_STATE {
                *$ptr.add($cap * k + $i) = $ctx.s(k);
                k += 1;
            }
        }
    }};
}

/// Load `M::N_STATE` SIMD channels from SoA memory into `[S; MAX_STATES]`.
/// `$ty` — SIMD type (e.g. `f32x8`). Compile-time guard: `M::N_STATE` ≤ MAX_STATES
/// (inline `const {}` block).
#[macro_export]
macro_rules! load_state_simd {
    ($ptr:expr, $cap:expr, $off:expr, $ty:ty) => {{
        const {
            assert!(M::N_STATE <= $crate::core::context::MAX_STATES);
        }
        unsafe {
            let mut arr = [<$ty>::splat(0.0); $crate::core::context::MAX_STATES];
            let mut k = 0usize;
            while k < M::N_STATE {
                arr[k] = ($ptr.add($cap * k + $off) as *const $ty).read_unaligned();
                k += 1;
            }
            arr
        }
    }};
}

/// Save `M::N_STATE` SIMD channels from a `SimdContext<$ty>` back to SoA memory.
/// Compile-time guard: `M::N_STATE` must fit in [`MAX_STATES`](crate::core::context::MAX_STATES)
/// (inline `const {}` block).
#[macro_export]
macro_rules! save_state_simd {
    ($ptr:expr, $cap:expr, $off:expr, $ctx:expr, $ty:ty) => {{
        const {
            assert!(M::N_STATE <= $crate::core::context::MAX_STATES);
        }
        unsafe {
            let mut k = 0usize;
            while k < M::N_STATE {
                ($ptr.add($cap * k + $off) as *mut $ty).write_unaligned($ctx.s(k));
                k += 1;
            }
        }
    }};
}

// ═══════════════════════════════════════════════════════════════
//  Low-Level Runner Functions
// ═══════════════════════════════════════════════════════════════

/// Legacy raw runner — accepts exactly **6 state pointers** (v,u,h,r,s,t).
///
/// Models with more than 6 states must use [`NeuralPopulation`](crate::core::NeuralPopulation)
/// instead — its state channels are unbounded (up to [`MAX_STATES`]).
///
/// # Safety
/// All six state pointers, `i_in`, `spikes` and `params` must be valid for `n`
/// elements. Column stride is `n` (SoA layout). `v`/`u`/`h`/`r`/`s`/`t` may be
/// null only if the model uses fewer than 6 state channels.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_neuron<M: Model>(
    v: *mut f32,
    u: *mut f32,
    h: *mut f32,
    r: *mut f32,
    s: *mut f32,
    t: *mut f32,
    i_in: *const f32,
    spikes: *mut i32,
    params: *const f32,
    n: usize,
) {
    run_neuron_ext::<M>(
        v,
        u,
        h,
        r,
        s,
        t,
        i_in,
        spikes,
        params,
        n,
        std::ptr::null(),
        std::ptr::null(),
    )
}

/// Legacy extended runner — accepts exactly **6 state pointers** (v,u,h,r,s,t)
/// plus optional pre/post-synaptic buses for STDP/plasticity.
///
/// Models with more than 6 states must use [`NeuralPopulation`](crate::core::NeuralPopulation).
///
/// # Safety
/// All pointers must be valid for `n` elements. `i_pre_in` / `i_post_in` may be null.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_neuron_ext<M: Model>(
    v: *mut f32,
    u: *mut f32,
    h: *mut f32,
    r: *mut f32,
    s: *mut f32,
    t: *mut f32,
    i_in: *const f32,
    spikes: *mut i32,
    params: *const f32,
    n: usize,
    i_pre_in: *const f32,
    i_post_in: *const f32,
) {
    debug_assert!(
        M::N_STATE <= 6,
        "run_neuron* accepts at most 6 state channels; use NeuralPopulation for more"
    );
    if n == 0 {
        return;
    }
    let n8 = n & !7;
    let dt = *params;
    let ptrs = [v, u, h, r, s, t];

    for off in (0..n8).step_by(8) {
        if off + 64 < n8 {
            crate::core::util::prefetch_read(params, off + 64);
            if !v.is_null() {
                crate::core::util::prefetch_read(v, off + 64);
            }
        }

        // Load first M::N_STATE channels from the six pointers (rest stay zero).

        let mut state = [f32x8::zero(); MAX_STATES];
        for (k, p) in ptrs.iter().copied().take(M::N_STATE).enumerate() {
            state[k] = (p.add(off) as *const f32x8).read_unaligned();
        }

        let i_syn = (i_in.add(off) as *const f32x8)
            .read_unaligned()
            .max(f32x8::splat(I_SYN_CLAMP_RANGE.0))
            .min(f32x8::splat(I_SYN_CLAMP_RANGE.1));

        let i_pre_val = if i_pre_in.is_null() {
            f32x8::zero()
        } else {
            (i_pre_in.add(off) as *const f32x8).read_unaligned()
        };
        let i_post_val = if i_post_in.is_null() {
            f32x8::zero()
        } else {
            (i_post_in.add(off) as *const f32x8).read_unaligned()
        };

        let mut ctx = SimdContext::<f32x8> {
            state,
            params_ptr: NonNull::new(params as *mut f32).expect("params_ptr must be non-null"),
            n,
            off,
            i_syn,
            i_pre: i_pre_val,
            i_post: i_post_val,
            dt_val: dt,
            spike_mask: f32x8::zero(),
            _marker: PhantomData,
        };

        M::step(&mut ctx);

        for (k, p) in ptrs.iter().copied().take(M::N_STATE).enumerate() {
            (p.add(off) as *mut f32x8).write_unaligned(ctx.s(k));
        }

        if M::HAS_SPIKE {
            let m = ctx.spike_mask().move_mask();
            if m == 0 {
                for j in 0..8 {
                    *spikes.add(off + j) = 0;
                }
            } else {
                for j in 0..8 {
                    *spikes.add(off + j) = (m >> j) & 1;
                }
            }
        }
    }

    if n8 < n {
        for i in n8..n {
            let ii = (*i_in.add(i)).clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
            let i_pre = if i_pre_in.is_null() {
                0.0
            } else {
                *i_pre_in.add(i)
            };
            let i_post = if i_post_in.is_null() {
                0.0
            } else {
                *i_post_in.add(i)
            };

            let mut state = [0.0f32; MAX_STATES];
            for k in 0..M::N_STATE {
                state[k] = *ptrs[k].add(i);
            }

            let mut ctx = SimdContext::<f32> {
                state,
                params_ptr: NonNull::new(params as *mut f32).expect("params_ptr must be non-null"),
                n,
                off: i,
                i_syn: ii,
                i_pre,
                i_post,
                dt_val: dt,
                spike_mask: 0.0,
                _marker: PhantomData,
            };

            M::step(&mut ctx);

            for (k, p) in ptrs.iter().copied().take(M::N_STATE).enumerate() {
                *p.add(i) = ctx.s(k);
            }
            if M::HAS_SPIKE {
                *spikes.add(i) = ctx.spike_mask().move_mask() as i32;
            }
        }
    }
}

/// Legacy parallel runner — accepts exactly **6 state pointers** (v,u,h,r,s,t).
///
/// Models with more than 6 states must use [`NeuralPopulation`](crate::core::NeuralPopulation).
///
/// # Safety
/// All pointers must be valid for `n` elements. Column stride = `n` for all SoA columns.
/// `chunk_size` should be ≥ 64 and a multiple of 8 for best SIMD alignment.
#[cfg(feature = "rayon")]
#[inline(always)]
pub unsafe fn run_neuron_par<M: Model>(
    v: *mut f32,
    u: *mut f32,
    h: *mut f32,
    r: *mut f32,
    s: *mut f32,
    t: *mut f32,
    i_in: *const f32,
    spikes: *mut i32,
    params: *const f32,
    n: usize,
    chunk_size: usize,
) {
    use rayon::prelude::*;

    if n == 0 {
        return;
    }
    let cs = chunk_size.max(64) & !7;
    let nc = n.div_ceil(cs);

    #[cfg(debug_assertions)]
    {
        let ranges: Vec<(usize, usize)> = (0..nc)
            .map(|ci| {
                let start = ci * cs;
                let end = (start + cs).min(n);
                (start, end)
            })
            .collect();
        check_no_overlap(&ranges);
    }

    let ptrs = SendablePtrs::new(v, u, h, r, s, t, i_in, spikes, params);

    (0..nc).into_par_iter().for_each(move |ci| {
        let off = ci * cs;
        let len = (n - off).min(cs);
        let state_ptrs = unsafe { ptrs.state_ptrs() };
        let i_in_ptr = unsafe { ptrs.i_in_ptr() };
        let spikes_ptr = unsafe { ptrs.spikes_ptr() };
        let params_ptr = unsafe { ptrs.params_ptr() };

        let n8 = len & !7;
        let dt = *params_ptr;

        for chunk_off in (0..n8).step_by(8) {
            let total_off = off + chunk_off;
            let mut state = [f32x8::zero(); MAX_STATES];
            for (k, p) in state_ptrs.iter().copied().take(M::N_STATE).enumerate() {
                state[k] = (p.add(total_off) as *const f32x8).read_unaligned();
            }

            let i_syn = (i_in_ptr.add(total_off) as *const f32x8)
                .read_unaligned()
                .max(f32x8::splat(I_SYN_CLAMP_RANGE.0))
                .min(f32x8::splat(I_SYN_CLAMP_RANGE.1));

            let mut ctx = SimdContext::<f32x8> {
                state,
                params_ptr: NonNull::new(params_ptr as *mut f32)
                    .expect("params_ptr must be non-null"),
                n,
                off: total_off,
                i_syn,
                i_pre: f32x8::zero(),
                i_post: f32x8::zero(),
                dt_val: dt,
                spike_mask: f32x8::zero(),
                _marker: PhantomData,
            };

            M::step(&mut ctx);

            for (k, p) in state_ptrs.iter().copied().take(M::N_STATE).enumerate() {
                (p.add(total_off) as *mut f32x8).write_unaligned(ctx.s(k));
            }

            if M::HAS_SPIKE {
                let m = ctx.spike_mask().move_mask();
                if m == 0 {
                    for j in 0..8 {
                        *spikes_ptr.add(total_off + j) = 0;
                    }
                } else {
                    for j in 0..8 {
                        *spikes_ptr.add(total_off + j) = (m >> j) & 1;
                    }
                }
            }
        }

        if n8 < len {
            for j in n8..len {
                let idx = off + j;
                let ii = (*i_in_ptr.add(idx)).clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
                let mut state = [0.0f32; MAX_STATES];
                for k in 0..M::N_STATE {
                    state[k] = *state_ptrs[k].add(idx);
                }
                let mut ctx = SimdContext::<f32> {
                    state,
                    params_ptr: NonNull::new(params_ptr as *mut f32)
                        .expect("params_ptr must be non-null"),
                    n,
                    off: idx,
                    i_syn: ii,
                    i_pre: 0.0,
                    i_post: 0.0,
                    dt_val: dt,
                    spike_mask: 0.0,
                    _marker: PhantomData,
                };
                M::step(&mut ctx);
                for (k, p) in state_ptrs.iter().copied().take(M::N_STATE).enumerate() {
                    *p.add(idx) = ctx.s(k);
                }
                if M::HAS_SPIKE {
                    *spikes_ptr.add(idx) = ctx.spike_mask().move_mask() as i32;
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::neurons::LIFModel;

    #[allow(clippy::type_complexity)] // тестовый хелпер: 5 буферов состояния
    fn make_lif_basic(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let dt = 0.5;
        let inv_tau = 0.1;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_ref = 2.0;
        let mut params = vec![0.0f32; n * 6];
        for i in 0..n {
            params[i] = dt;
            params[n + i] = inv_tau;
            params[2 * n + i] = v_rest;
            params[3 * n + i] = v_thresh;
            params[4 * n + i] = v_reset;
            params[5 * n + i] = tau_ref;
        }
        let v = vec![-70.0; n];
        let u = vec![0.0; n];
        let h = vec![0.0; n];
        let r = vec![0.0; n]; // refractory counter
        (params, v, u, h, r)
    }

    #[test]
    fn test_run_neuron_basic() {
        let n = 8;
        let (params, v, u, h, r) = make_lif_basic(n);
        let mut v = v;
        let mut u = u;
        let mut h = h;
        let mut r = r;
        let i_in = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let s_ptr = std::ptr::null_mut(); // 5th state not used
        let t_ptr = std::ptr::null_mut(); // 6th state not used

        unsafe {
            run_neuron::<LIFModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                h.as_mut_ptr(),
                r.as_mut_ptr(),
                s_ptr,
                t_ptr,
                i_in.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(v[0] > -70.0, "v should depolarize");
    }

    #[test]
    fn test_run_neuron_zero_n() {
        unsafe {
            run_neuron::<LIFModel>(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
            );
        }
        // Should not crash with n=0
    }

    #[test]
    fn test_run_neuron_ext_buses() {
        let n = 8;
        let (params, v, u, h, r) = make_lif_basic(n);
        let mut v = v;
        let mut u = u;
        let mut h = h;
        let mut r = r;
        let i_in = vec![30.0f32; n];
        let i_pre = vec![0.0f32; n];
        let i_post = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron_ext::<LIFModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                h.as_mut_ptr(),
                r.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                i_in.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
                i_pre.as_ptr(),
                i_post.as_ptr(),
            );
        }
        assert!(v[0] > -70.0, "v should depolarize with buses");
    }
}
