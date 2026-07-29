//! 🧫 Storage Pillar: выровненный Structure-of-Arrays буфер для SIMD.
//!
//! Выделяет непрерывные блоки памяти, выровненные по 64 байт (кэш-линия CPU).
//! Layout: [Params 0..N][State 0..M] в SoA формате (column-major).

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use super::lane::SimdLane;
use super::Model;

/// 🧫 Storage Pillar: выровненный SoA буфер для SIMD-вычислений.
///
/// # Layout
/// ```text
/// param[0]×capacity | param[1]×capacity | ... | state[0]×capacity | state[1]×capacity | ...
/// ```
pub struct StorageBuffer<M: Model> {
    ptr: *mut f32,
    capacity: usize,
    _marker: PhantomData<M>,
}

impl<M: Model> StorageBuffer<M> {
    /// Создать новый буфер на `count` сущностей, выровненный по 64 байт.
    pub fn new<S: SimdLane>(count: usize) -> Self {
        let lane_size = S::SIZE;
        let capacity = (count + lane_size - 1) & !(lane_size - 1);
        let total_floats = (M::N_PARAMS + M::N_STATE) * capacity;
        let layout = Layout::from_size_align(
            total_floats * std::mem::size_of::<f32>(),
            64,
        ).expect("Invalid memory alignment for StorageBuffer");

        let ptr = unsafe {
            let p = alloc_zeroed(layout) as *mut f32;
            if p.is_null() {
                panic!("StorageBuffer: failed to allocate aligned SoA memory");
            }
            p
        };
        Self { ptr, capacity, _marker: PhantomData }
    }

    #[inline(always)] pub fn capacity(&self) -> usize { self.capacity }
    #[inline(always)] pub fn as_ptr(&self) -> *const f32 { self.ptr }

    /// Указатель на SoA-пласт параметра по индексу.
    ///
    /// # Safety
    /// `param_idx` must be < `M::N_PARAMS`. Returned pointer is valid for `self.capacity()` elements.
    #[inline(always)]
    pub unsafe fn param_ptr(&self, param_idx: usize) -> *const f32 {
        debug_assert!(param_idx < M::N_PARAMS);
        self.ptr.add(param_idx * self.capacity)
    }

    /// Мутабельный указатель на SoA-пласт состояния по индексу.
    ///
    /// # Safety
    /// `state_idx` must be < `M::N_STATE`. Returned pointer is valid for `self.capacity()` elements.
    #[inline(always)]
    pub unsafe fn state_mut_ptr(&mut self, state_idx: usize) -> *mut f32 {
        debug_assert!(state_idx < M::N_STATE);
        self.ptr.add((M::N_PARAMS + state_idx) * self.capacity)
    }
}

impl<M: Model> Drop for StorageBuffer<M> {
    fn drop(&mut self) {
        let total_floats = (M::N_PARAMS + M::N_STATE) * self.capacity;
        if let Ok(layout) = Layout::from_size_align(total_floats * std::mem::size_of::<f32>(), 64) {
            unsafe { dealloc(self.ptr as *mut u8, layout); }
        }
    }
}

unsafe impl<M: Model> Send for StorageBuffer<M> {}
unsafe impl<M: Model> Sync for StorageBuffer<M> {}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;
    use crate::model::models::LIFModel;

    #[test]
    fn test_storage_buffer_alloc_and_write() {
        let mut buf = StorageBuffer::<LIFModel>::new::<f32x8>(10);
        let cap = buf.capacity();
        // capacity should be rounded up to multiple of 8
        assert!(cap >= 10);
        assert_eq!(cap % 8, 0);

        unsafe {
            // Write a test value to param[2] column (v_rest)
            let param_ptr = buf.param_ptr(2) as *mut f32;
            for i in 0..10 {
                *param_ptr.add(i) = -70.0 + i as f32;
            }
            // Read back
            for i in 0..10 {
                assert!((*param_ptr.add(i) - (-70.0 + i as f32)).abs() < 1e-6,
                    "param[2][{i}] = {}", *param_ptr.add(i));
            }

            // Write to state[0] column (v)
            let state_ptr = buf.state_mut_ptr(0);
            for i in 0..10 {
                *state_ptr.add(i) = -65.0;
            }
            for i in 0..10 {
                assert!((*state_ptr.add(i) - (-65.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_storage_buffer_zeroed() {
        let buf = StorageBuffer::<LIFModel>::new::<f32x8>(16);
        unsafe {
            // All memory should be zero-initialized
            for i in 0..buf.capacity() {
                let v = *buf.param_ptr(0).add(i);
                assert_eq!(v, 0.0, "param[0][{i}] should be zero, got {v}");
            }
        }
    }
}
