//! Shared SIMD load/store utilities for neuron models.
//!
//! Provides `load8`/`store8` helpers used by standalone SIMD functions
//! (sensors, synapses) that need raw `f32x8` pointer access without
//! going through the `Model` trait.

use wide::f32x8;

/// Load 8 f32 values from `ptr + off` into a `f32x8` vector.
///
/// # Safety
/// `ptr.add(off)` must be valid for 8 readable elements.
#[inline(always)]
pub unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3),
                 *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}

/// Store a `f32x8` vector to 8 locations at `ptr + off`.
///
/// 🛡️ BUGFIX: Use `write_unaligned` instead of `to_array()` + scalar stores.
/// The old code spilled the AVX2 register to the stack via to_array(), then
/// wrote each float one-by-one — completely defeating SIMD throughput.
///
/// # Safety
/// `ptr.add(off)` must be valid for 8 writable elements.
#[inline(always)]
pub unsafe fn store8(ptr: *mut f32, off: usize, val: f32x8) {
    (ptr.add(off) as *mut f32x8).write_unaligned(val);
}

/// Load 8 `i32` spike flags from `ptr + off` as a `f32x8` (1.0 / 0.0).
///
/// Converts integer spikes to float in-lane — used by plasticity kernels
/// (STDP, dopamine) that consume hard 0/1 spike buffers directly.
///
/// # Safety
/// `ptr.add(off)` must be valid for 8 readable `i32` elements.
#[inline(always)]
pub unsafe fn load8_i32(ptr: *const i32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([
        *p as f32, *p.add(1) as f32, *p.add(2) as f32, *p.add(3) as f32,
        *p.add(4) as f32, *p.add(5) as f32, *p.add(6) as f32, *p.add(7) as f32,
    ])
}
