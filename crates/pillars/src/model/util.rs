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
/// # Safety
/// `ptr.add(off)` must be valid for 8 writable elements.
#[inline(always)]
pub unsafe fn store8(ptr: *mut f32, off: usize, val: f32x8) {
    let a = val.to_array();
    let p = ptr.add(off);
    *p      = a[0]; *p.add(1) = a[1];
    *p.add(2) = a[2]; *p.add(3) = a[3];
    *p.add(4) = a[4]; *p.add(5) = a[5];
    *p.add(6) = a[6]; *p.add(7) = a[7];
}
