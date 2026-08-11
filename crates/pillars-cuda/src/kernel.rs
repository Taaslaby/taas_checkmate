//! Generic trusted NVRTC kernel launcher.
//!
//! This layer is intentionally independent from [`crate::CompiledModel`].
//! `CompiledModel` remains the optimized neuron/population API; [`CudaKernel`]
//! is the lower-level escape hatch for physics, chemistry, particles, and
//! other data-parallel computations.
//!
//! # Safety model
//!
//! The source is trusted CUDA C++ supplied by the application. NVRTC does not
//! sandbox code. Public callers must validate source provenance and enforce
//! deployment limits before exposing this API to untrusted users.

use std::ffi::c_void;

use crate::engine::{compile_source_function, device_context, driver_error, CUDA_CALL_LOCK};
use crate::ffi::{driver, CUdeviceptr, CUfunction, CUmodule};

/// A scalar kernel argument passed by value.
#[derive(Clone, Copy, Debug)]
pub enum ScalarArg {
    F32(f32),
    I32(i32),
    U32(u32),
}

/// A host-owned typed buffer copied to device memory for a launch and copied
/// back afterwards.
#[derive(Debug)]
pub enum BufferArg<'a> {
    F32(&'a mut [f32]),
    I32(&'a mut [i32]),
}

impl BufferArg<'_> {
    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }

    fn bytes(&self) -> Result<usize, String> {
        self.len()
            .checked_mul(4)
            .ok_or_else(|| "CUDA buffer byte size overflows usize".to_string())
    }

    fn host_ptr(&self) -> *const c_void {
        match self {
            Self::F32(values) => values.as_ptr() as *const c_void,
            Self::I32(values) => values.as_ptr() as *const c_void,
        }
    }

    fn host_mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::F32(values) => values.as_mut_ptr() as *mut c_void,
            Self::I32(values) => values.as_mut_ptr() as *mut c_void,
        }
    }
}

/// Explicit one-dimensional CUDA launch configuration.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: u32,
    pub block: u32,
}

impl LaunchConfig {
    /// Choose a 256-thread block and enough blocks to cover `n` elements.
    pub fn for_len(n: usize) -> Result<Self, String> {
        let n: u32 = n
            .try_into()
            .map_err(|_| "launch length exceeds CUDA grid range".to_string())?;
        Ok(Self {
            grid: n.div_ceil(256).max(1),
            block: 256,
        })
    }

    pub fn validate(self) -> Result<(), String> {
        if self.grid == 0 || self.block == 0 {
            return Err("CUDA launch grid and block must be > 0".to_string());
        }
        if self.block > 1024 {
            return Err("CUDA one-dimensional block must be <= 1024 threads".to_string());
        }
        Ok(())
    }
}

/// A compiled arbitrary CUDA kernel with temporary host-buffer arguments.
pub struct CudaKernel {
    module: CUmodule,
    function: CUfunction,
    name: String,
}

impl CudaKernel {
    /// Compile `source` with NVRTC and load the named `__global__` function.
    pub fn compile(source: &str, kernel_name: &str) -> Result<Self, String> {
        if source.is_empty() {
            return Err("CUDA source must not be empty".to_string());
        }
        if kernel_name.is_empty() || kernel_name.as_bytes().contains(&0) {
            return Err("CUDA kernel name must be non-empty and contain no NUL".to_string());
        }
        let (module, function) = compile_source_function(source, kernel_name)?;
        Ok(Self {
            module,
            function,
            name: kernel_name.to_string(),
        })
    }

    /// Upload buffers, launch the kernel, synchronize, then download buffers.
    ///
    /// Kernel pointer arguments appear first in `buffers`, followed by scalar
    /// arguments in `scalars`. This mirrors ordinary CUDA C++ signatures:
    /// `kernel(float* x, const float* force, float dt, int n)`.
    pub fn launch(
        &self,
        buffers: &mut [BufferArg<'_>],
        scalars: &[ScalarArg],
        config: LaunchConfig,
    ) -> Result<(), String> {
        config.validate()?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let driver = driver().as_ref().map_err(|e| e.clone())?;
        device_context(driver)?;

        let buffer_sizes: Vec<usize> = buffers
            .iter()
            .map(BufferArg::bytes)
            .collect::<Result<_, _>>()?;
        if buffer_sizes.contains(&0) {
            return Err(format!("CUDA buffer for `{}` must not be empty", self.name));
        }

        let mut device_buffers = Vec::with_capacity(buffers.len());
        unsafe {
            for (buffer, &bytes) in buffers.iter().zip(&buffer_sizes) {
                let mut ptr: CUdeviceptr = 0;
                let rc = (driver.cu_mem_alloc)(&mut ptr, bytes);
                if rc != 0 {
                    free_buffers(driver, &device_buffers);
                    return Err(format!(
                        "{}: cuMemAlloc failed: {}",
                        self.name,
                        driver_error(driver, rc)
                    ));
                }
                let rc = (driver.cu_memcpy_htod)(ptr, buffer.host_ptr(), bytes);
                if rc != 0 {
                    free_buffers(driver, &device_buffers);
                    let _ = (driver.cu_mem_free)(ptr);
                    return Err(format!(
                        "{}: upload failed: {}",
                        self.name,
                        driver_error(driver, rc)
                    ));
                }
                device_buffers.push(ptr);
            }

            let mut args: Vec<*mut c_void> =
                Vec::with_capacity(device_buffers.len() + scalars.len());
            let mut ptr_values = device_buffers.clone();
            for ptr in &mut ptr_values {
                args.push(ptr as *mut CUdeviceptr as *mut c_void);
            }

            enum ScalarStorage {
                F32(f32),
                I32(i32),
                U32(u32),
            }
            let mut scalar_storage = scalars
                .iter()
                .map(|scalar| match *scalar {
                    ScalarArg::F32(value) => ScalarStorage::F32(value),
                    ScalarArg::I32(value) => ScalarStorage::I32(value),
                    ScalarArg::U32(value) => ScalarStorage::U32(value),
                })
                .collect::<Vec<_>>();
            for scalar in &mut scalar_storage {
                let ptr = match scalar {
                    ScalarStorage::F32(value) => value as *mut f32 as *mut c_void,
                    ScalarStorage::I32(value) => value as *mut i32 as *mut c_void,
                    ScalarStorage::U32(value) => value as *mut u32 as *mut c_void,
                };
                args.push(ptr);
            }

            let rc = (driver.cu_launch_kernel)(
                self.function,
                config.grid,
                1,
                1,
                config.block,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if rc != 0 {
                free_buffers(driver, &device_buffers);
                return Err(format!(
                    "{}: launch failed: {}",
                    self.name,
                    driver_error(driver, rc)
                ));
            }

            let sync_rc = (driver.cu_ctx_synchronize)();
            if sync_rc != 0 {
                free_buffers(driver, &device_buffers);
                return Err(format!(
                    "{}: launch sync failed: {}",
                    self.name,
                    driver_error(driver, sync_rc)
                ));
            }

            for ((buffer, &ptr), &bytes) in
                buffers.iter_mut().zip(&device_buffers).zip(&buffer_sizes)
            {
                let rc = (driver.cu_memcpy_dtoh)(buffer.host_mut_ptr(), ptr, bytes);
                if rc != 0 {
                    free_buffers(driver, &device_buffers);
                    return Err(format!(
                        "{}: download failed: {}",
                        self.name,
                        driver_error(driver, rc)
                    ));
                }
            }
            free_buffers(driver, &device_buffers);
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if let Ok(driver) = driver().as_ref() {
            if device_context(driver).is_ok() {
                unsafe {
                    let _ = (driver.cu_module_unload)(self.module);
                }
            }
        }
    }
}

unsafe fn free_buffers(driver: &crate::ffi::Driver, buffers: &[CUdeviceptr]) {
    for &ptr in buffers {
        let _ = (driver.cu_mem_free)(ptr);
    }
}
