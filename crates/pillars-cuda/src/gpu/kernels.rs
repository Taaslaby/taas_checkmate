use crate::gpu::device::ok_or;
use crate::gpu::ffi::*;
use std::ffi::c_int;
pub fn step_izhikevich(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &mut [f32],
    dt: f32,
    trace_decay: f32,
    spikes: &mut [i32],
    c: &[f32],
    d: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) -> Result<(), String> {
    let n = v.len();
    ok_or(
        unsafe {
            cuda_izhikevich_step(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                trace.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                I.as_mut_ptr(),
                dt,
                trace_decay,
                30.0, // v_thresh=30.0 (default)
                spikes.as_mut_ptr(),
                c.as_ptr(),
                d.as_ptr(),
                refractory.as_mut_ptr(),
                tau_ref,
                n as c_int,
            )
        },
        "izhikevich",
    )
}

pub fn step_lif(
    v: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) -> Result<(), String> {
    let n = v.len();
    let mut flat = vec![0.0f32; 1 + 4 * n];
    flat[0] = dt;
    for i in 0..n {
        flat[1 + i] = tau_m[i];
        flat[1 + n + i] = v_rest[i];
        flat[1 + 2 * n + i] = v_thresh[i];
        flat[1 + 3 * n + i] = v_reset[i];
    }
    ok_or(
        unsafe {
            cuda_lif_step(
                v.as_mut_ptr(),
                I.as_mut_ptr(),
                spikes.as_mut_ptr(),
                flat.as_ptr(),
                refractory.as_mut_ptr(),
                tau_ref,
                n as c_int,
            )
        },
        "lif",
    )
}

pub fn step_gco(
    v: &mut [f32],
    u: &mut [f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    i_ext: Option<&[f32]>,
    dt: f32,
    tau: f32,
) -> Result<(), String> {
    let n = v.len() as c_int;
    let i_ext_ptr = i_ext.map(|x| x.as_ptr()).unwrap_or(core::ptr::null());
    ok_or(
        unsafe {
            cuda_gco_step(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                values.as_ptr(),
                i_ext_ptr,
                dt,
                tau,
                n,
            )
        },
        "gco_step",
    )
}

pub fn stdp_update(
    weights: &mut [f32],
    traces_pre: &[f32],
    traces_post: &[f32],
    lr_ltd: f32,
    lr_ltp: f32,
    w_min: f32,
    w_max: f32,
) -> Result<(), String> {
    let n = weights.len();
    ok_or(
        unsafe {
            cuda_stdp_update(
                weights.as_mut_ptr(),
                traces_pre.as_ptr(),
                traces_post.as_ptr(),
                lr_ltd,
                lr_ltp,
                w_min,
                w_max,
                n as c_int,
            )
        },
        "stdp",
    )
}

pub fn saw_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &mut [f32],
    dt: f32,
    trace_decay: f32,
    noise_amp: f32,
    spikes: &mut [i32],
    refractory: &mut [f32],
    tau_ref: f32,
    // Persistent noise state must be provided (non-NULL) for GA reproducibility.
    // Use GpuContext::d_noise to get a device pointer that persists across steps.
    d_noise: *mut u32,
) -> Result<(), String> {
    let n = v.len() as c_int;
    ok_or(
        unsafe {
            cuda_saw_step(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                trace.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                I.as_mut_ptr(),
                dt,
                trace_decay,
                noise_amp,
                spikes.as_mut_ptr(),
                refractory.as_mut_ptr(),
                tau_ref,
                d_noise,
                30.0, // v_thresh=30.0 (default)
                n,
            )
        },
        "saw_step",
    )
}

pub fn step_morris_lecar(
    v: &mut [f32],
    w: &mut [f32],
    was_below: &mut [f32],
    i_in: &[f32],
    spikes: &mut [i32],
    params: &[f32],
    dt: f32,
) -> Result<(), String> {
    let n = v.len();
    assert_eq!(w.len(), n, "morris_lecar: w length must match v");
    assert_eq!(
        was_below.len(),
        n,
        "morris_lecar: was_below length must match v"
    );
    assert_eq!(i_in.len(), n, "morris_lecar: input length must match v");
    assert_eq!(spikes.len(), n, "morris_lecar: spikes length must match v");
    let expected_params = n
        .checked_mul(14)
        .expect("morris_lecar: parameter length overflows usize");
    assert_eq!(
        params.len(),
        expected_params,
        "morris_lecar: params must be 14*n SoA values"
    );
    let n: c_int = n
        .try_into()
        .expect("morris_lecar: population length exceeds CUDA int range");
    ok_or(
        unsafe {
            cuda_morris_lecar_step(
                v.as_mut_ptr(),
                w.as_mut_ptr(),
                was_below.as_mut_ptr(),
                i_in.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                dt,
                n,
            )
        },
        "morris_lecar_step",
    )
}

pub fn step_nmda_mg_block(
    v: &[f32],
    g_act: &[f32],
    i_out: &mut [f32],
    params: &[f32],
) -> Result<(), String> {
    let n = v.len();
    assert_eq!(g_act.len(), n, "nmda_mg_block: g_act length must match v");
    assert_eq!(i_out.len(), n, "nmda_mg_block: output length must match v");
    let expected_params = n
        .checked_mul(4)
        .expect("nmda_mg_block: parameter length overflows usize");
    assert_eq!(
        params.len(),
        expected_params,
        "nmda_mg_block: params must be 4*n AoS values"
    );
    let n: c_int = n
        .try_into()
        .expect("nmda_mg_block: population length exceeds CUDA int range");
    ok_or(
        unsafe {
            cuda_nmda_mg_block_step(
                v.as_ptr(),
                g_act.as_ptr(),
                i_out.as_mut_ptr(),
                params.as_ptr(),
                n,
            )
        },
        "nmda_mg_block_step",
    )
}

pub fn step_tsodyks_markram(
    x: &mut [f32],
    u: &mut [f32],
    spikes: &[i32],
    w_eff: &mut [f32],
    params: &[f32],
    dt: f32,
) -> Result<(), String> {
    let n = x.len();
    assert_eq!(u.len(), n, "tsodyks_markram: u length must match x");
    assert_eq!(
        spikes.len(),
        n,
        "tsodyks_markram: spikes length must match x"
    );
    assert_eq!(
        w_eff.len(),
        n,
        "tsodyks_markram: output length must match x"
    );
    let expected_params = n
        .checked_mul(4)
        .expect("tsodyks_markram: parameter length overflows usize");
    assert_eq!(
        params.len(),
        expected_params,
        "tsodyks_markram: params must be 4*n AoS values"
    );
    let n: c_int = n
        .try_into()
        .expect("tsodyks_markram: population length exceeds CUDA int range");
    ok_or(
        unsafe {
            cuda_tsodyks_markram_step(
                x.as_mut_ptr(),
                u.as_mut_ptr(),
                spikes.as_ptr(),
                w_eff.as_mut_ptr(),
                params.as_ptr(),
                dt,
                n,
            )
        },
        "tsodyks_markram_step",
    )
}
