//! # AlignedVec — SIMD-aligned heap buffer
//!
//! === BOUNDARY: Safe Rust → heap allocation ===
//! The **only** place where `std::alloc::alloc` is called.
//! All raw-pointer SIMD access in core/population.rs reads/writes through
//! `AlignedVec` — guaranteeing 32-byte alignment and zero padding.
//!
//! ## Memory model
//!
//! ```text
//!   AlignedVec::new(10, 0.0)
//!     → capacity = 16 (10.div_ceil(8) * 8)
//!     → alloc(64 bytes, align=32)
//!     → data[0..10] = 0.0, data[10..16] = 0.0 (padding) — never uninit
//! ```
//!
//! ## Why not `Vec` of f32
//!
//! | Property | `AlignedVec` | `Vec<f32>` |
//! |:---------|:------------|:-----------|
//! | Alignment | 32 bytes | 4 bytes (align_of f32) |
//! | Padding | Auto to multiple of 8 | Manual |
//! | Chunk iter | `f32x8` + scalar tail | None |
//! | Allocation | `std::alloc::alloc` | `std::alloc::alloc` (same) |
//!
//! ## API layers
//!
//! 1. **Buffer management**: `new()`, `from_slice()`, `get()`, `set()`, `fill()`
//! 2. **SIMD chunk iteration**: `chunks()` / `chunks_mut()` — split at `len/8` boundary
//! 3. **Free functions**: `split_simd()` / `split_simd_ro()` — raw slice → `(&[f32x8], &[f32])`

use super::arch;
use std::alloc::{self, Layout};
use std::ptr::NonNull;
use wide::f32x8;

/// Aligned vector with auto-padding to 8-wide SIMD.
///
/// # Guarantees
/// - `len()` is always a multiple of 8
/// - Data aligned to `arch::ALIGN` bytes
#[derive(Debug)]
pub struct AlignedVec {
    ptr: NonNull<f32>,
    logical_len: usize,
    capacity: usize,
}

impl AlignedVec {
    /// Create a new aligned buffer. `len` is rounded up to the next multiple of 8.
    #[must_use]
    pub fn new(len: usize, init: f32) -> Self {
        let capacity = len.div_ceil(8) * 8;
        if capacity == 0 {
            return Self { ptr: NonNull::dangling(), logical_len: 0, capacity: 0 };
        }

        let size_bytes = capacity * 4;
        let layout = Layout::from_size_align(size_bytes, arch::ALIGN)
            .expect("AlignedVec layout");
        let ptr = unsafe {
            let p = alloc::alloc(layout);
            NonNull::new(p as *mut f32).expect("AlignedVec alloc failed")
        };

        // Initialise data + zero padding (prevents UB in full_slice)
        for i in 0..len {
            unsafe { *ptr.as_ptr().add(i) = init; }
        }
        for i in len..capacity {
            unsafe { *ptr.as_ptr().add(i) = 0.0; }
        }

        Self { ptr, logical_len: len, capacity }
    }

    /// Create from an existing slice (copies into aligned buffer).
    pub fn from_slice(data: &[f32]) -> Self {
        let mut buf = Self::new(data.len(), 0.0);
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);
        buf
    }

    #[inline(always)] pub fn len(&self) -> usize { self.logical_len }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.logical_len == 0 }
    #[inline(always)] pub fn capacity(&self) -> usize { self.capacity }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const f32 { self.ptr.as_ptr() }
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr.as_ptr() }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        if self.logical_len == 0 { &[] }
        else { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.logical_len) } }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.logical_len == 0 { &mut [] }
        else { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.logical_len) } }
    }

    #[inline(always)]
    pub fn full_slice(&self) -> &[f32] {
        if self.capacity == 0 { &[] }
        else { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) } }
    }

    /// Get element at index.
    ///
    /// # Panics (debug only)
    /// Panics if `index >= len()`.
    #[inline(always)]
    pub fn get(&self, index: usize) -> f32 {
        debug_assert!(index < self.logical_len);
        unsafe { *self.ptr.as_ptr().add(index) }
    }

    /// Set element at index.
    ///
    /// # Panics (debug only)
    /// Panics if `index >= len()`.
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: f32) {
        debug_assert!(index < self.logical_len);
        unsafe { *self.ptr.as_ptr().add(index) = value; }
    }

    /// Fill buffer with a value.
    #[inline]
    pub fn fill(&mut self, value: f32) {
        for i in 0..self.logical_len {
            unsafe { *self.ptr.as_ptr().add(i) = value; }
        }
    }

    /// Iterate over SIMD chunks + scalar tail (mutable).
    ///
    /// Uses `logical_len` as the boundary between SIMD chunks and scalar tail.
    /// #1 SIMD chunks: first `logical_len / 8` f32x8 blocks (guaranteed aligned)
    /// #2 Scalar tail: remaining `logical_len % 8` elements
    #[inline]
    pub fn chunks_mut<F, G>(&mut self, mut simd_fn: F, mut tail_fn: G)
    where
        F: FnMut(&mut [f32x8]),
        G: FnMut(&mut [f32]),
    {
        // FIX #4: Use logical_len (not capacity) for SIMD/tail boundary
        let logical = self.logical_len;
        if logical == 0 { return; }
        let n8 = logical / 8;
        let remainder = logical % 8;
        if n8 > 0 {
            let simd_slice = unsafe {
                std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f32x8, n8)
            };
            simd_fn(simd_slice);
        }
        if remainder > 0 {
            let tail_slice = unsafe {
                std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(n8 * 8), remainder)
            };
            tail_fn(tail_slice);
        }
    }

    /// Iterate over SIMD chunks + scalar tail (read-only).
    /// Uses `logical_len` as the boundary.
    #[inline]
    pub fn chunks<F, G>(&self, mut simd_fn: F, mut tail_fn: G)
    where
        F: FnMut(&[f32x8]),
        G: FnMut(&[f32]),
    {
        let logical = self.logical_len;
        if logical == 0 { return; }
        let n8 = logical / 8;
        let remainder = logical % 8;
        if n8 > 0 {
            let simd_slice = unsafe {
                std::slice::from_raw_parts(self.ptr.as_ptr() as *const f32x8, n8)
            };
            simd_fn(simd_slice);
        }
        if remainder > 0 {
            let tail_slice = unsafe {
                std::slice::from_raw_parts(self.ptr.as_ptr().add(n8 * 8), remainder)
            };
            tail_fn(tail_slice);
        }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::from_size_align(self.capacity * 4, arch::ALIGN)
                .expect("AlignedVec drop layout");
            unsafe { alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout); }
        }
    }
}

// SAFETY: AlignedVec owns its memory (alloc + dealloc via Layout).
// The NonNull<f32> points to an exclusive allocation managed by Drop.
// Rust's borrow checker prevents aliasing at compile time.
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        let mut buf = Self::new(self.logical_len, 0.0);
        buf.as_mut_slice().copy_from_slice(self.as_slice());
        buf
    }
}

/// Split an aligned slice into `f32x8` SIMD part + scalar tail.
///
/// # Safety
/// Slice must be aligned to `arch::ALIGN` (32 bytes).
/// Use `AlignedVec` for guaranteed alignment.
#[inline]
pub unsafe fn split_simd(slice: &mut [f32]) -> (&mut [f32x8], &mut [f32]) {
    debug_assert!(arch::is_slice_aligned(slice), "split_simd: slice not aligned to {}", arch::ALIGN);
    let n8 = slice.len() / 8 * 8;
    let (simd_bytes, tail) = slice.split_at_mut(n8);
    let simd = std::slice::from_raw_parts_mut(simd_bytes.as_mut_ptr() as *mut f32x8, n8 / 8);
    (simd, tail)
}

/// Read-only version of `split_simd`.
///
/// # Safety
/// Slice must be aligned to `arch::ALIGN` (32 bytes).
#[inline]
pub unsafe fn split_simd_ro(slice: &[f32]) -> (&[f32x8], &[f32]) {
    debug_assert!(arch::is_slice_aligned(slice), "split_simd_ro: slice not aligned to {}", arch::ALIGN);
    let n8 = slice.len() / 8 * 8;
    let (simd_bytes, tail) = slice.split_at(n8);
    let simd = std::slice::from_raw_parts(simd_bytes.as_ptr() as *const f32x8, n8 / 8);
    (simd, tail)
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_aligned_vec_basic() {
        let v = AlignedVec::new(10, -65.0);
        assert_eq!(v.len(), 10);
        assert!(v.capacity() >= 16);
    }

    #[test]
    fn test_aligned_vec_zero() {
        let v = AlignedVec::new(0, 0.0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_aligned_vec_assign() {
        let mut v = AlignedVec::new(8, 0.0);
        for i in 0..8 { v.set(i, i as f32); }
        for i in 0..8 { assert!((v.get(i) - i as f32).abs() < 0.001); }
    }

    #[test]
    fn test_chunks_mut() {
        let mut v = AlignedVec::new(10, 1.0);
        v.chunks_mut(|simd| {
            for chunk in simd.iter_mut() { *chunk *= f32x8::splat(2.0); }
        }, |tail| {
            for t in tail.iter_mut() { *t *= 2.0; }
        });
        for i in 0..10 { assert!((v.get(i) - 2.0).abs() < 0.001); }
    }

    #[test]
    fn test_split_simd() {
        // Use AlignedVec for proper alignment
        let mut v = AlignedVec::new(9, 0.0);
        for i in 0..9 { v.set(i, (i + 1) as f32); }
        let (simd, tail) = unsafe { split_simd(v.as_mut_slice()) };
        assert_eq!(simd.len(), 1); // 8 elements → 1 f32x8
        assert_eq!(tail.len(), 1); // 1 remaining
        assert!((tail[0] - 9.0).abs() < 0.001);
    }
}
