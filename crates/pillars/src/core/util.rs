//! # 🔧 SIMD Plumbing — load8 / store8 Raw Access
//!
//! Low-level `f32x8` load/store helpers. Used by standalone SIMD kernels
//! (sensors, synapses) that need raw pointer access without going through
//! the `Model` trait or `NeuralPopulation`.
//!
//! ## Functions
//!
//! | Helper | Signature | Notes |
//! |--------|----------|-------|
//! | [`load8`] | `(ptr, off) → f32x8` | ЗD развёрнутый load — безопасен для unaligned (scalar штучно) |
//! | [`store8`] | `(ptr, off, val) → ()` | `write_unaligned` — одна инструкция store, без spilling в стек |
//! | [`load8_i32`] | `(ptr, off) → f32x8` | i32→f32 конверсия на лету (STDP, dopamine) |
//!
//! ## Why write_unaligned?
//!
//! Старая версия `store8` вызывала `to_array()` → спиливала AVX2-регистр в
//! стек → затем писала 8 float по одному — полностью убивая SIMD-пропускную
//! способность. Исправлено в версии с `write_unaligned`: одна инструкция
//! `vmovups` на AVX2, даже для невыровненных адресов (пенальти ~10%).

use wide::f32x8;

/// Load 8 f32 values from `ptr + off` into a `f32x8` vector.
///
/// # Safety
/// `ptr.add(off)` must be valid for 8 readable elements.
#[inline(always)]
pub unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    (ptr.add(off) as *const f32x8).read_unaligned()
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

/// Prefetch memory at `ptr + off` into L1 cache for subsequent SIMD iterations.
///
/// Reduces L3 cache miss latency during steps over large populations.
#[inline(always)]
pub unsafe fn prefetch_read(ptr: *const f32, off: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_prefetch(ptr.add(off) as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
        let _ = off;
    }
}

