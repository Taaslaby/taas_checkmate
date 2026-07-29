//! Neuron model runner — dispatches SIMD chunks and scalar tail.
//!
//! # Architecture
//!
//! ```text
//!   run_neuron::<M>(n neurons)
//!     ├── SIMD loop:  chunk of 8 → f32x8 context → M::step() → write back
//!     └── Scalar tail: n%8 remaining → f32 context → M::step() → write back
//! ```
//!
//! # Clean Bus
//! - `i_in` → ctx.i_syn (post-synaptic current)
//! - `i_pre_in` → ctx.i_pre (pre-synaptic spikes, for STDP)
//! - `i_post_in` → ctx.i_post (post-synaptic spikes, for STDP)

use wide::f32x8;
use super::lane::SimdLane;
use super::context::SimdContext;
use super::Model;

/// Run a neuron model across `n` neurons using auto-SIMD vectorisation.
///
/// # Safety
/// All pointers must be valid for `n` elements.
#[inline(always)]
pub unsafe fn run_neuron<M: Model>(
    v: *mut f32, u: *mut f32, h: *mut f32,
    r: *mut f32, s: *mut f32, t: *mut f32,
    i_in: *const f32, spikes: *mut i32,
    params: *const f32, n: usize,
) {
    // Use null pointers for i_pre and i_post when not provided
    run_neuron_ext::<M>(v, u, h, r, s, t, i_in, spikes, params, n,
                        std::ptr::null(), std::ptr::null())
}

/// Extended runner with pre/post-synaptic buses for STDP/plasticity.
///
/// # Safety
/// All pointers must be valid for `n` elements.
#[inline(always)]
pub unsafe fn run_neuron_ext<M: Model>(
    v: *mut f32, u: *mut f32, h: *mut f32,
    r: *mut f32, s: *mut f32, t: *mut f32,
    i_in: *const f32, spikes: *mut i32,
    params: *const f32, n: usize,
    i_pre_in: *const f32,
    i_post_in: *const f32,
) {
    debug_assert!(M::N_STATE <= 6);
    let n8 = n & !7;
    let dt = *params;

    // ── SIMD chunk (8 neurons at once via f32x8, stride=8) ──
    for off in (0..n8).step_by(8) {
        let st0 = (v.add(off) as *const f32x8).read_unaligned();
        let st1 = if M::N_STATE > 1 { (u.add(off) as *const f32x8).read_unaligned() } else { f32x8::zero() };
        let st2 = if M::N_STATE > 2 { (h.add(off) as *const f32x8).read_unaligned() } else { f32x8::zero() };
        let st3 = if M::N_STATE > 3 { (r.add(off) as *const f32x8).read_unaligned() } else { f32x8::zero() };
        let st4 = if M::N_STATE > 4 { (s.add(off) as *const f32x8).read_unaligned() } else { f32x8::zero() };
        let st5 = if M::N_STATE > 5 { (t.add(off) as *const f32x8).read_unaligned() } else { f32x8::zero() };

        let i_syn = (i_in.add(off) as *const f32x8)
            .read_unaligned()
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));

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
            state: [st0, st1, st2, st3, st4, st5],
            params_ptr: params,
            n,
            off,
            i_syn,
            i_pre: i_pre_val,
            i_post: i_post_val,
            dt_val: dt,
            spike_mask: f32x8::zero(),
        };

        M::step(&mut ctx);

        (v.add(off) as *mut f32x8).write_unaligned(ctx.s(0));
        if M::N_STATE > 1 { (u.add(off) as *mut f32x8).write_unaligned(ctx.s(1)); }
        if M::N_STATE > 2 { (h.add(off) as *mut f32x8).write_unaligned(ctx.s(2)); }
        if M::N_STATE > 3 { (r.add(off) as *mut f32x8).write_unaligned(ctx.s(3)); }
        if M::N_STATE > 4 { (s.add(off) as *mut f32x8).write_unaligned(ctx.s(4)); }
        if M::N_STATE > 5 { (t.add(off) as *mut f32x8).write_unaligned(ctx.s(5)); }

        if M::HAS_SPIKE {
            let m = ctx.spike_mask().move_mask();
            if m == 0 {
                (spikes.add(off) as *mut f32x8).write_unaligned(f32x8::zero());
            } else {
                for j in 0..8 {
                    *spikes.add(off + j) = (m >> j) & 1;
                }
            }
        }
    }

    // ── Scalar tail (1 neuron at a time) ──
    if n8 < n {
        for i in n8..n {
            let ii = (*i_in.add(i)).clamp(-100.0, 100.0);
            let i_pre = if i_pre_in.is_null() { 0.0 } else { *i_pre_in.add(i) };
            let i_post = if i_post_in.is_null() { 0.0 } else { *i_post_in.add(i) };

            let mut ctx = SimdContext::<f32> {
                state: [
                    *v.add(i),
                    if M::N_STATE > 1 { *u.add(i) } else { 0.0 },
                    if M::N_STATE > 2 { *h.add(i) } else { 0.0 },
                    if M::N_STATE > 3 { *r.add(i) } else { 0.0 },
                    if M::N_STATE > 4 { *s.add(i) } else { 0.0 },
                    if M::N_STATE > 5 { *t.add(i) } else { 0.0 },
                ],
                params_ptr: params,
                n,
                off: i,
                i_syn: ii,
                i_pre,
                i_post,
                dt_val: dt,
                spike_mask: 0.0,
            };

            M::step(&mut ctx);

            *v.add(i) = ctx.s(0);
            if M::N_STATE > 1 { *u.add(i) = ctx.s(1); }
            if M::N_STATE > 2 { *h.add(i) = ctx.s(2); }
            if M::N_STATE > 3 { *r.add(i) = ctx.s(3); }
            if M::N_STATE > 4 { *s.add(i) = ctx.s(4); }
            if M::N_STATE > 5 { *t.add(i) = ctx.s(5); }
            if M::HAS_SPIKE {
                *spikes.add(i) = ctx.spike_mask().move_mask() as i32;
            }
        }
    }
}

/// Parallel runner via Rayon. Splits `n` neurons into chunks processed
/// by separate threads, each running the SIMD+scalar path internally.
///
/// # Safety
/// All pointer arguments must point to valid memory for `n` elements each.
/// `chunk_size` must be > 0.
#[cfg(feature = "rayon")]
#[inline(always)]
pub unsafe fn run_neuron_par<M: Model>(
    v: *mut f32, u: *mut f32, h: *mut f32,
    r: *mut f32, s: *mut f32, t: *mut f32,
    i_in: *const f32, spikes: *mut i32,
    params: *const f32, n: usize,
    chunk_size: usize,
) {
    use rayon::prelude::*;

    let cs = chunk_size.max(64) & !7;
    let nc = n.div_ceil(cs);

    let ptrs = [v as usize, u as usize, h as usize, r as usize,
                s as usize, t as usize, i_in as usize, spikes as usize, params as usize];

    (0..nc).into_par_iter().for_each(move |ci| {
        let off = ci * cs;
        let len = (n - off).min(cs);
        run_neuron::<M>(
            ptrs[0] as *mut f32, ptrs[1] as *mut f32,
            ptrs[2] as *mut f32, ptrs[3] as *mut f32,
            ptrs[4] as *mut f32, ptrs[5] as *mut f32,
            ptrs[6] as *const f32, ptrs[7] as *mut i32,
            ptrs[8] as *const f32, len,
        );
    });
}
