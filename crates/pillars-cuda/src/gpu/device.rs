use std::ffi::c_int;

extern "C" {
    fn cuda_set_device(device: c_int) -> c_int;
    fn cuda_get_device_count(count: *mut c_int) -> c_int;
}

pub fn get_device_count() -> Result<i32, String> {
    let mut count: c_int = 0;
    let err = unsafe { cuda_get_device_count(&mut count as *mut c_int) } as i32;
    if err != 0 {
        Err(format!("cudaGetDeviceCount: error {err}"))
    } else {
        Ok(count as i32)
    }
}

pub fn set_device(device: i32) -> Result<(), String> {
    let err = unsafe { cuda_set_device(device as c_int) } as i32;
    if err != 0 {
        Err(format!("cudaSetDevice: error {err}"))
    } else {
        Ok(())
    }
}

pub(crate) fn ok_or(err: c_int, name: &str) -> Result<(), String> {
    if err == 0 {
        Ok(())
    } else {
        Err(format!("CUDA {name} error: {err}"))
    }
}
