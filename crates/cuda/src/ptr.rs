use std::ffi::c_int;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    pub(crate) fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    pub(crate) fn cudaMemset(ptr: *mut std::ffi::c_void, val: c_int, count: usize) -> c_int;
    pub(crate) fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    pub(crate) fn cudaFree(ptr: *mut std::ffi::c_void) -> c_int;
}

/// A simple RAII wrapper for CUDA device memory.
pub struct DevicePtr<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DevicePtr<T> {
    pub fn new(len: usize) -> Result<Self, String> {
        let mut ptr: *mut T = std::ptr::null_mut();
        let err = unsafe {
            cudaMalloc(
                &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
                len * std::mem::size_of::<T>(),
            )
        };
        if err != 0 || ptr.is_null() {
            return Err(format!(
                "cudaMalloc({}×{}B): error {err}",
                len,
                std::mem::size_of::<T>()
            ));
        }
        Ok(Self { ptr, len })
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copy_from_host(&self, src: &[T]) -> Result<(), String> {
        let n = src.len().min(self.len);
        let err = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                n * std::mem::size_of::<T>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            Err(format!("cudaMemcpy H2D: error {err}"))
        } else {
            Ok(())
        }
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), String> {
        let n = dst.len().min(self.len);
        let err = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                n * std::mem::size_of::<T>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 {
            Err(format!("cudaMemcpy D2H: error {err}"))
        } else {
            Ok(())
        }
    }
}

impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl<T> Send for DevicePtr<T> {}
unsafe impl<T> Sync for DevicePtr<T> {}
