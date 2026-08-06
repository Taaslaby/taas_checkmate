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
        // `Layout::array` does the multiplication overflow-checked — a raw
        // `capacity * 4` could wrap for external `len` near usize::MAX/4 and
        // allocate too little, letting the init loop write out of bounds.
        let capacity = len.div_ceil(8) * 8;
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                logical_len: 0,
                capacity: 0,
            };
        }
        // Layout must request arch::ALIGN (32 B), NOT f32's natural 4 B:
        // a plain Layout::array::<f32> would silently drop the SIMD
        // alignment guarantee and misalign every f32x8 load/store.
        // align_to() keeps the overflow-checked size computation.
        // SAFETY: `align_to` only raises the alignment to a power of two
        // (32), the total size is unchanged; layout stays valid.
        let layout = Layout::array::<f32>(capacity)
            .expect("AlignedVec layout")
            .align_to(arch::ALIGN)
            .expect("AlignedVec align");
        let ptr = unsafe {
            let p = alloc::alloc(layout);
            if p.is_null() {
                alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(p as *mut f32)
        };

        // SAFETY: ptr is valid for `capacity` f32s (just allocated).
        // Initialise data + zero padding (prevents UB in full_slice).
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, capacity);
        }
        if init != 0.0 {
            // SAFETY: same 0..len region as above, already zero-initialised.
            unsafe {
                for i in 0..len {
                    *ptr.as_ptr().add(i) = init;
                }
            }
        }

        Self {
            ptr,
            logical_len: len,
            capacity,
        }
    }

    /// Create from an existing slice (copies into aligned buffer).
    pub fn from_slice(data: &[f32]) -> Self {
        let mut buf = Self::new(data.len(), 0.0);
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);
        buf
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.logical_len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.logical_len == 0
    }
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        if self.logical_len == 0 {
            &[]
        } else {
            // SAFETY: ptr is valid for `capacity >= logical_len` f32s (own
            // allocation, alive for `&self`), and the first `logical_len`
            // elements are always initialized (new()/from_slice write them).
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.logical_len) }
        }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.logical_len == 0 {
            &mut []
        } else {
            // SAFETY: as above, plus &mut self gives exclusive ownership of
            // the region, so no aliasing &mut is outstanding.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.logical_len) }
        }
    }

    /// Full buffer including zero padding — `len()..capacity()` are 0.0
    /// (never uninit). Safe for SIMD loads that read the whole last chunk.
    #[inline(always)]
    pub fn full_slice(&self) -> &[f32] {
        if self.capacity == 0 {
            &[]
        } else {
            // SAFETY: ptr is valid for `capacity` f32s; ALL of them are
            // initialized — data by new()/from_slice, padding explicitly
            // zeroed in new(). This is the invariant that makes reading the
            // padded last SIMD chunk sound.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
        }
    }

    /// Get element at index.
    ///
    /// # Panics (debug only)
    /// Panics if `index >= len()` (release: caller MUST guarantee it — an
    /// out-of-range index reads uninitialized heap memory).
    #[inline(always)]
    pub fn get(&self, index: usize) -> f32 {
        debug_assert!(index < self.logical_len);
        // SAFETY: debug_asserted `index < logical_len`; ptr is valid for
        // `logical_len` initialized elements.
        unsafe { *self.ptr.as_ptr().add(index) }
    }

    /// Set element at index.
    ///
    /// # Panics (debug only)
    /// Panics if `index >= len()` (release: caller MUST guarantee it — an
    /// out-of-range index writes to heap memory outside the logical region).
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: f32) {
        debug_assert!(index < self.logical_len);
        // SAFETY: as `get`, plus &mut self gives exclusive write access.
        unsafe {
            *self.ptr.as_ptr().add(index) = value;
        }
    }

    /// Fill buffer with a value.
    #[inline]
    pub fn fill(&mut self, value: f32) {
        // SAFETY: `as_mut_slice` exposes exactly `logical_len` initialized
        // elements of the exclusive allocation; fill is bounds-safe by
        // construction.
        self.as_mut_slice().fill(value);
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
        if logical == 0 {
            return;
        }
        let n8 = logical / 8;
        let remainder = logical % 8;
        // Build both sub-slices through one `split_at_mut` (like `split_simd`
        // below): a single reborrow chain is sound under Stacked/Tree Borrows,
        // whereas two independent `from_raw_parts_mut` from the same base
        // pointer are more fragile (no aliasing today, but weaker for future
        // provenance checks).
        // SAFETY: `as_mut_slice` is valid for `logical` elements; split_at_mut
        // guarantees disjoint sub-slices, and the f32x8 view of the first
        // `n8*8` bytes is 32-byte aligned because AlignedVec is (align >= 8*4).
        let (simd_part, tail_part) = self.as_mut_slice().split_at_mut(n8 * 8);
        if n8 > 0 {
            let simd_slice =
                unsafe { std::slice::from_raw_parts_mut(simd_part.as_mut_ptr() as *mut f32x8, n8) };
            simd_fn(simd_slice);
        }
        if remainder > 0 {
            tail_fn(tail_part);
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
        if logical == 0 {
            return;
        }
        let n8 = logical / 8;
        let remainder = logical % 8;
        // Same single-split pattern as chunks_mut (see SAFETY there).
        // SAFETY: `as_slice` is valid for `logical` elements; split_at gives
        // disjoint sub-slices; the f32x8 view is 32-byte aligned by AlignedVec.
        let (simd_part, tail_part) = self.as_slice().split_at(n8 * 8);
        if n8 > 0 {
            let simd_slice =
                unsafe { std::slice::from_raw_parts(simd_part.as_ptr() as *const f32x8, n8) };
            simd_fn(simd_slice);
        }
        if remainder > 0 {
            tail_fn(tail_part);
        }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // SAFETY: identical layout to `new` (array size + align_to 32);
            // dealloc must be called with the exact (size, align) it was
            // allocated with, and this is the unique owner (exclusive ptr,
            // no aliases).
            let layout = Layout::array::<f32>(self.capacity)
                .expect("AlignedVec drop layout")
                .align_to(arch::ALIGN)
                .expect("AlignedVec drop align");
            unsafe {
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
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
    debug_assert!(
        arch::is_slice_aligned(slice),
        "split_simd: slice not aligned to {}",
        arch::ALIGN
    );
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
    debug_assert!(
        arch::is_slice_aligned(slice),
        "split_simd_ro: slice not aligned to {}",
        arch::ALIGN
    );
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
        for i in 0..8 {
            v.set(i, i as f32);
        }
        for i in 0..8 {
            assert!((v.get(i) - i as f32).abs() < 0.001);
        }
    }

    #[test]
    fn test_chunks_mut() {
        let mut v = AlignedVec::new(10, 1.0);
        v.chunks_mut(
            |simd| {
                for chunk in simd.iter_mut() {
                    *chunk *= f32x8::splat(2.0);
                }
            },
            |tail| {
                for t in tail.iter_mut() {
                    *t *= 2.0;
                }
            },
        );
        for i in 0..10 {
            assert!((v.get(i) - 2.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_split_simd() {
        // Use AlignedVec for proper alignment
        let mut v = AlignedVec::new(9, 0.0);
        for i in 0..9 {
            v.set(i, (i + 1) as f32);
        }
        let (simd, tail) = unsafe { split_simd(v.as_mut_slice()) };
        assert_eq!(simd.len(), 1); // 8 elements → 1 f32x8
        assert_eq!(tail.len(), 1); // 1 remaining
        assert!((tail[0] - 9.0).abs() < 0.001);
    }

    /// The key invariant of AlignedVec: padding beyond `len()` is always
    /// zero-initialized (never uninit) — that is what makes reading the
    /// padded last SIMD chunk sound.
    #[test]
    fn test_padding_is_zero() {
        let v = AlignedVec::new(10, 5.0);
        let full = v.full_slice();
        assert_eq!(full.len(), 16, "capacity = len.div_ceil(8)*8");
        // data[0..10] = 5.0, padding[10..16] = 0.0
        assert!(
            full[..10].iter().all(|&x| x == 5.0),
            "data region must hold init value"
        );
        assert_eq!(&full[10..16], &[0.0f32; 6]);
    }

    /// read-only chunks(): SIMD part must cover the full chunks, tail the rest.
    #[test]
    fn test_chunks_readonly() {
        let mut v = AlignedVec::new(10, 0.0);
        for i in 0..10 {
            v.set(i, i as f32);
        }
        let mut simd_sum = 0.0f32;
        let mut tail_sum = 0.0f32;
        v.chunks(
            |simd| {
                for c in simd.iter() {
                    simd_sum += c.to_array().iter().sum::<f32>();
                }
            },
            |tail| {
                tail_sum = tail.iter().sum();
            },
        );
        assert!(
            (simd_sum - 28.0).abs() < 1e-3,
            "sum of 0..8 = 28, got {simd_sum}"
        );
        assert!(
            (tail_sum - 17.0).abs() < 1e-3,
            "sum of 8,9 = 17, got {tail_sum}"
        );
    }

    #[test]
    fn test_split_simd_ro() {
        let v = AlignedVec::new(9, 0.0);
        let (simd, tail) = unsafe { split_simd_ro(v.as_slice()) };
        assert_eq!(simd.len(), 1);
        assert_eq!(tail.len(), 1);
    }

    /// Clone must be a deep copy: independent allocation, same contents,
    /// zero padding (mutating one must not affect the other).
    #[test]
    fn test_clone_is_independent() {
        let mut a = AlignedVec::new(10, 1.0);
        let mut b = a.clone();
        a.set(0, 99.0);
        assert_eq!(b.get(0), 1.0, "clone must be an independent allocation");
        assert_eq!(
            b.full_slice()[10..16],
            [0.0f32; 6],
            "clone padding must be zeroed"
        );
        b.set(3, -5.0);
        assert_eq!(a.get(3), 1.0, "mutating clone must not touch original");
    }

    /// Explicit alignment check straight from `new` (not via arch helpers).
    #[test]
    fn test_alignment_is_32_bytes() {
        for n in [1, 8, 10, 100, 1000] {
            let v = AlignedVec::new(n, 0.0);
            assert!(
                (v.as_ptr() as usize).is_multiple_of(arch::ALIGN),
                "AlignedVec::new({n}) misaligned: {:p} vs align {}",
                v.as_ptr(),
                arch::ALIGN
            );
        }
    }
}
