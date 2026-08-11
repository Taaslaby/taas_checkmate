//! FFI layer: NVRTC (dynamic load) + CUDA **driver API**.
//!
//! On Windows the toolkit ships an *empty* `cuda.lib` stub — the real driver
//! lives in `nvcuda.dll` (in System32) and has no import library in the
//! toolkit. The robust, version-agnostic approach is to load it dynamically
//! (`LoadLibraryW` on Windows, `dlopen` on Unix) and resolve the handful of
//! entry points we need via `GetProcAddress`/`dlsym`.
//!
//! NVRTC is loaded the same way: it ships in the CUDA toolkit `bin` dir, so a
//! hard link would make any consumer binary (e.g. the Python `.pyd`) fail to
//! load on machines where the toolkit `bin` is not on the OS search path.
//!
//! All unsafe FFI is confined to this module; `engine.rs` uses the safe
//! [`Driver`] and [`Nvrtc`] handles.

#![allow(non_camel_case_types, dead_code, clippy::upper_case_acronyms)]

use std::ffi::{c_char, CStr, CString};
use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

// ── Opaque CUDA driver types ────────────────────────────────────────
pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

// cuStreamCreate flags
pub const CU_STREAM_DEFAULT: u32 = 0x00;
// cuStreamBeginCapture flags
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 0x00;
// cuGraphInstantiate flags
pub const CU_GRAPH_INSTANTIATE_DEFAULT: u64 = 0;

// ── NVRTC (dynamically loaded — same reason as the driver) ─────────
pub type nvrtcResult = c_int;
pub type nvrtcProgram = *mut c_void;

pub type NvrtcCreateProgramFn = unsafe extern "C" fn(
    *mut nvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> nvrtcResult;
pub type NvrtcCompileProgramFn =
    unsafe extern "C" fn(nvrtcProgram, c_int, *const *const c_char) -> nvrtcResult;
pub type NvrtcGetPTXSizeFn = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
pub type NvrtcGetPTXFn = unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult;
pub type NvrtcGetProgramLogSizeFn = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
pub type NvrtcGetProgramLogFn = unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult;
pub type NvrtcGetErrorStringFn = unsafe extern "C" fn(nvrtcResult) -> *const c_char;
pub type NvrtcDestroyProgramFn = unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult;

/// Resolved NVRTC entry points (host-side compiler — no GPU required).
pub struct Nvrtc {
    pub nvrtc_create_program: NvrtcCreateProgramFn,
    pub nvrtc_compile_program: NvrtcCompileProgramFn,
    pub nvrtc_get_ptx_size: NvrtcGetPTXSizeFn,
    pub nvrtc_get_ptx: NvrtcGetPTXFn,
    pub nvrtc_get_program_log_size: NvrtcGetProgramLogSizeFn,
    pub nvrtc_get_program_log: NvrtcGetProgramLogFn,
    pub nvrtc_get_error_string: NvrtcGetErrorStringFn,
    pub nvrtc_destroy_program: NvrtcDestroyProgramFn,
}

static NVRTC: OnceLock<Result<Nvrtc, String>> = OnceLock::new();

/// Load (once) and return the NVRTC compiler API handle.
pub fn nvrtc() -> &'static Result<Nvrtc, String> {
    NVRTC.get_or_init(Nvrtc::load)
}

impl Nvrtc {
    fn load() -> Result<Self, String> {
        #[cfg(windows)]
        let lib = find_nvrtc_windows()?;
        #[cfg(unix)]
        let lib = find_nvrtc_unix()?;
        #[cfg(not(any(windows, unix)))]
        return Err("unsupported platform".to_string());

        macro_rules! sym {
            ($t:ty, $name:literal) => {
                unsafe { load_symbol::<$t>($name, lib)? }
            };
        }

        Ok(Self {
            nvrtc_create_program: sym!(NvrtcCreateProgramFn, "nvrtcCreateProgram"),
            nvrtc_compile_program: sym!(NvrtcCompileProgramFn, "nvrtcCompileProgram"),
            nvrtc_get_ptx_size: sym!(NvrtcGetPTXSizeFn, "nvrtcGetPTXSize"),
            nvrtc_get_ptx: sym!(NvrtcGetPTXFn, "nvrtcGetPTX"),
            nvrtc_get_program_log_size: sym!(NvrtcGetProgramLogSizeFn, "nvrtcGetProgramLogSize"),
            nvrtc_get_program_log: sym!(NvrtcGetProgramLogFn, "nvrtcGetProgramLog"),
            nvrtc_get_error_string: sym!(NvrtcGetErrorStringFn, "nvrtcGetErrorString"),
            nvrtc_destroy_program: sym!(NvrtcDestroyProgramFn, "nvrtcDestroyProgram"),
        })
    }
}

/// Known `nvrtc64_*.dll` names, newest first (CUDA 11 → 13).
#[cfg(windows)]
const NVRTC_WINDOWS_CANDIDATES: &[&str] = &[
    "nvrtc64_130_0.dll",
    "nvrtc64_131.dll",
    "nvrtc64_130.dll",
    "nvrtc64_122.dll",
    "nvrtc64_121.dll",
    "nvrtc64_120.dll",
    "nvrtc64_112_0.dll",
    "nvrtc64_111_0.dll",
    "nvrtc64_110_0.dll",
];

#[cfg(windows)]
fn find_nvrtc_windows() -> Result<*mut c_void, String> {
    let mut errors: Vec<String> = Vec::new();
    // 1) Bare names — LoadLibraryW searches the app dir, System32 and PATH.
    for name in NVRTC_WINDOWS_CANDIDATES {
        match os::load(name) {
            Ok(h) => return Ok(h),
            Err(e) => errors.push(e),
        }
    }
    // 2) CUDA_PATH\bin[x64] — full path, no PATH dependency.
    if let Ok(path) = std::env::var("CUDA_PATH") {
        for sub in ["bin\\x64", "bin"] {
            if let Some(full) = nvrtc_dll_in_dir(&format!("{path}\\{sub}")) {
                if let Ok(h) = os::load(&full) {
                    return Ok(h);
                }
            }
        }
    }
    // 3) Default toolkit root — any installed version, newest last.
    if let Ok(progfiles) = std::env::var("ProgramFiles") {
        let root = format!("{progfiles}\\NVIDIA GPU Computing Toolkit\\CUDA");
        if let Ok(entries) = std::fs::read_dir(&root) {
            let mut versions: Vec<String> = entries
                .flatten()
                .map(|e| e.path().to_string_lossy().into_owned())
                .collect();
            versions.sort();
            for dir in versions.into_iter().rev() {
                for sub in ["bin\\x64", "bin"] {
                    if let Some(full) = nvrtc_dll_in_dir(&format!("{dir}\\{sub}")) {
                        if let Ok(h) = os::load(&full) {
                            return Ok(h);
                        }
                    }
                }
            }
        }
    }
    Err(format!(
        "NVRTC (nvrtc64_*.dll) not found: {}",
        errors.first().cloned().unwrap_or_default()
    ))
}

/// First `nvrtc64*.dll` in `dir` (newest by name), if any.
#[cfg(windows)]
fn nvrtc_dll_in_dir(dir: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("nvrtc64") && n.ends_with(".dll"))
        .collect();
    names.sort();
    names.into_iter().next_back().map(|n| format!("{dir}\\{n}"))
}

#[cfg(unix)]
fn find_nvrtc_unix() -> Result<*mut c_void, String> {
    const CANDIDATES: &[&str] = &[
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.so",
    ];
    let mut errors: Vec<String> = Vec::new();
    for name in CANDIDATES {
        match os_unix::load(name) {
            Ok(h) => return Ok(h),
            Err(e) => errors.push(e),
        }
    }
    for dir in [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/cuda/lib64",
    ] {
        for name in CANDIDATES {
            if let Ok(h) = os_unix::load(&format!("{dir}/{name}")) {
                return Ok(h);
            }
        }
    }
    Err(format!(
        "NVRTC (libnvrtc.so*) not found: {}",
        errors.first().cloned().unwrap_or_default()
    ))
}

// ── Driver API types (dynamically loaded) ───────────────────────────
pub type CuInitFn = unsafe extern "C" fn(u32) -> CUresult;
pub type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
pub type CuDeviceGetAttributeFn = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
pub type CuCtxCreateFn = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
pub type CuCtxDestroyFn = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type CuCtxSetCurrentFn = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type CuModuleLoadDataExFn = unsafe extern "C" fn(
    *mut CUmodule,
    *const c_void,
    u32,
    *mut c_int,
    *mut *mut c_void,
) -> CUresult;
pub type CuModuleGetFunctionFn =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
pub type CuModuleUnloadFn = unsafe extern "C" fn(CUmodule) -> CUresult;
pub type CuMemAllocFn = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
pub type CuMemFreeFn = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
pub type CuMemcpyHtoDFn = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
pub type CuMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
pub type CuLaunchKernelFn = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
pub type CuCtxSynchronizeFn = unsafe extern "C" fn() -> CUresult;
pub type CuGetErrorStringFn = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

// ── Stream + CUDA Graph API (for graph-captured kernel replay) ─────
pub type CuStreamCreateFn = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
pub type CuStreamDestroyFn = unsafe extern "C" fn(CUstream) -> CUresult;
pub type CuStreamBeginCaptureFn = unsafe extern "C" fn(CUstream, u32) -> CUresult;
pub type CuStreamEndCaptureFn = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
pub type CuGraphInstantiateFn =
    unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64, *mut c_void, *mut c_void, u64) -> CUresult;
pub type CuGraphLaunchFn = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
pub type CuGraphDestroyFn = unsafe extern "C" fn(CUgraph) -> CUresult;
pub type CuGraphExecDestroyFn = unsafe extern "C" fn(CUgraphExec) -> CUresult;

/// Resolved driver API entry points.
pub struct Driver {
    pub cu_init: CuInitFn,
    pub cu_device_get: CuDeviceGetFn,
    pub cu_device_get_attribute: CuDeviceGetAttributeFn,
    pub cu_ctx_create: CuCtxCreateFn,
    pub cu_ctx_destroy: CuCtxDestroyFn,
    pub cu_ctx_set_current: CuCtxSetCurrentFn,
    pub cu_module_load_data_ex: CuModuleLoadDataExFn,
    pub cu_module_get_function: CuModuleGetFunctionFn,
    pub cu_module_unload: CuModuleUnloadFn,
    pub cu_mem_alloc: CuMemAllocFn,
    pub cu_mem_free: CuMemFreeFn,
    pub cu_memcpy_htod: CuMemcpyHtoDFn,
    pub cu_memcpy_dtoh: CuMemcpyDtoHFn,
    pub cu_launch_kernel: CuLaunchKernelFn,
    pub cu_ctx_synchronize: CuCtxSynchronizeFn,
    pub cu_get_error_string: CuGetErrorStringFn,
    pub cu_stream_create: CuStreamCreateFn,
    pub cu_stream_destroy: CuStreamDestroyFn,
    pub cu_stream_begin_capture: CuStreamBeginCaptureFn,
    pub cu_stream_end_capture: CuStreamEndCaptureFn,
    pub cu_graph_instantiate: CuGraphInstantiateFn,
    pub cu_graph_launch: CuGraphLaunchFn,
    pub cu_graph_destroy: CuGraphDestroyFn,
    pub cu_graph_exec_destroy: CuGraphExecDestroyFn,
}

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

/// Load (once) and return the CUDA driver API handle.
pub fn driver() -> &'static Result<Driver, String> {
    DRIVER.get_or_init(Driver::load)
}

impl Driver {
    fn load() -> Result<Self, String> {
        #[cfg(windows)]
        let lib = load_windows_dll()?;
        #[cfg(unix)]
        let lib = load_unix_so()?;
        #[cfg(not(any(windows, unix)))]
        return Err("unsupported platform".to_string());

        // A function pointer and a void* are both pointer-sized on every
        // supported target, so copying the bits is sound for these types.
        macro_rules! sym {
            ($t:ty, $name:literal) => {
                unsafe { load_symbol::<$t>($name, lib)? }
            };
        }

        Ok(Self {
            cu_init: sym!(CuInitFn, "cuInit"),
            cu_device_get: sym!(CuDeviceGetFn, "cuDeviceGet"),
            cu_device_get_attribute: sym!(CuDeviceGetAttributeFn, "cuDeviceGetAttribute"),
            cu_ctx_create: sym!(CuCtxCreateFn, "cuCtxCreate"),
            cu_ctx_destroy: sym!(CuCtxDestroyFn, "cuCtxDestroy"),
            cu_ctx_set_current: sym!(CuCtxSetCurrentFn, "cuCtxSetCurrent"),
            cu_module_load_data_ex: sym!(CuModuleLoadDataExFn, "cuModuleLoadDataEx"),
            cu_module_get_function: sym!(CuModuleGetFunctionFn, "cuModuleGetFunction"),
            cu_module_unload: sym!(CuModuleUnloadFn, "cuModuleUnload"),
            cu_mem_alloc: sym!(CuMemAllocFn, "cuMemAlloc"),
            cu_mem_free: sym!(CuMemFreeFn, "cuMemFree"),
            cu_memcpy_htod: sym!(CuMemcpyHtoDFn, "cuMemcpyHtoD"),
            cu_memcpy_dtoh: sym!(CuMemcpyDtoHFn, "cuMemcpyDtoH"),
            cu_launch_kernel: sym!(CuLaunchKernelFn, "cuLaunchKernel"),
            cu_ctx_synchronize: sym!(CuCtxSynchronizeFn, "cuCtxSynchronize"),
            cu_get_error_string: sym!(CuGetErrorStringFn, "cuGetErrorString"),
            cu_stream_create: sym!(CuStreamCreateFn, "cuStreamCreate"),
            cu_stream_destroy: sym!(CuStreamDestroyFn, "cuStreamDestroy"),
            cu_stream_begin_capture: sym!(CuStreamBeginCaptureFn, "cuStreamBeginCapture"),
            cu_stream_end_capture: sym!(CuStreamEndCaptureFn, "cuStreamEndCapture"),
            cu_graph_instantiate: sym!(CuGraphInstantiateFn, "cuGraphInstantiate"),
            cu_graph_launch: sym!(CuGraphLaunchFn, "cuGraphLaunch"),
            cu_graph_destroy: sym!(CuGraphDestroyFn, "cuGraphDestroy"),
            cu_graph_exec_destroy: sym!(CuGraphExecDestroyFn, "cuGraphExecDestroy"),
        })
    }
}

/// Resolve one driver entry point by name.
///
/// # Safety
/// `T` must be a function-pointer type, which is pointer-sized on every
/// supported target; `debug_assert` guards that invariant so a future
/// refactor cannot silently widen `T` (it would trip in debug builds).
#[cfg(any(windows, unix))]
unsafe fn load_symbol<T: Copy>(name: &str, lib: *mut c_void) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    #[cfg(windows)]
    let p = os::GetProcAddress(lib, c.as_ptr());
    #[cfg(unix)]
    let p = os_unix::dlsym(lib, c.as_ptr());
    if p.is_null() {
        Err(format!("CUDA driver missing entry point `{name}`"))
    } else {
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*const c_void>()
        );
        Ok(std::mem::transmute_copy(&p))
    }
}

// ── OS plumbing ─────────────────────────────────────────────────────

#[cfg(windows)]
mod os {
    use std::ffi::OsStr;
    use std::os::raw::c_void;
    use std::os::windows::ffi::OsStrExt;

    pub type HMODULE = *mut c_void;

    #[link(name = "kernel32")]
    extern "C" {
        pub fn LoadLibraryW(name: *const u16) -> HMODULE;
        pub fn GetProcAddress(module: HMODULE, name: *const i8) -> *const c_void;
    }

    pub fn load(name: &str) -> Result<HMODULE, String> {
        let wide: Vec<u16> = OsStr::new(name).encode_wide().chain(Some(0)).collect();
        unsafe {
            let h = LoadLibraryW(wide.as_ptr());
            if h.is_null() {
                Err(format!(
                    "LoadLibraryW({name}) failed (CUDA driver not installed?)"
                ))
            } else {
                Ok(h)
            }
        }
    }
}

#[cfg(windows)]
fn load_windows_dll() -> Result<*mut std::os::raw::c_void, String> {
    // nvcuda.dll sits in System32 (driver package); also try the toolkit bin dir.
    match os::load("nvcuda.dll") {
        Ok(h) => Ok(h),
        Err(first) => {
            if let Ok(path) = std::env::var("CUDA_PATH") {
                let alt = format!("{path}\\bin\\nvcuda.dll");
                if let Ok(h) = os::load(&alt) {
                    return Ok(h);
                }
            }
            Err(first)
        }
    }
}

#[cfg(unix)]
mod os_unix {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_void};

    #[link(name = "dl")]
    extern "C" {
        pub fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub fn load(name: &str) -> Result<*mut c_void, String> {
        let c = CString::new(name).unwrap();
        unsafe {
            let h = dlopen(c.as_ptr(), 0x002 /* RTLD_NOW */);
            if h.is_null() {
                Err(format!("dlopen({name}) failed"))
            } else {
                Ok(h)
            }
        }
    }
}

#[cfg(unix)]
fn load_unix_so() -> Result<*mut std::os::raw::c_void, String> {
    for name in ["libcuda.so.1", "libcuda.so"] {
        if let Ok(h) = os_unix::load(name) {
            return Ok(h);
        }
    }
    Err("libcuda.so not found — no NVIDIA driver?".to_string())
}

// ── Error strings ───────────────────────────────────────────────────

/// Human-readable NVRTC error message (static string, no free needed).
pub fn nvrtc_error(rc: nvrtcResult) -> String {
    match nvrtc().as_ref() {
        Ok(n) => unsafe {
            let p = (n.nvrtc_get_error_string)(rc);
            if p.is_null() {
                format!("unknown NVRTC error {rc}")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        },
        Err(e) => format!("NVRTC unavailable: {e}"),
    }
}
