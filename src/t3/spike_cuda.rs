//! # T2-SPIGRAD GPU Backward (K1 + K2 + K3)
//!
//! Полностью GPU-ускоренный backward pass:
//!
//! 1. Upload trajectory (pre-states, drives) to GPU (step-major layout)
//! 2. Upload grad_pst to GPU
//! 3. K1 (pst_jacobian): d_v = J^T · grad_pst
//! 4. BPTT loop (CPU-side, GPU kernels):
//!    for step in rev(0..n_steps):
//!      K3+K2 (bptt_step_gpu): grad_drives[step] + v = v_prev
//! 5. Download grad_drives from GPU
//!
//! Memory layout (step-major):
//!   d_state [n_steps × 6n]:    [step0: th1..r3][step1: th1..r3]...
//!   d_drive [n_steps × 3n]:    [step0: d1..d3][step1: d1..d3]...
//!   d_grads [n_steps × 3n]:    same layout as d_drive

use crate::t3::{
    pst_linear_dim,
    spike::T3SpikeTrajectory,
};
use cuda_kernels::{
    DevicePtr, pst_jacobian_gpu,
};

/// Fully GPU-accelerated backward pass.
pub fn t3spike_backward_cuda(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    ctx: &mut CudaBackwardContext,
) -> Result<Vec<f32>, String> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let _pst_dim = pst_linear_dim(n);

    // ── 1. Upload trajectory to GPU (step-major) ──
    ctx.upload_trajectory(traj);

    // ── 2. Upload grad_pst ──
    ctx.d_grad_pst.copy_from_host(grad_pst)?;

    // ── 3. K1: upload final state + run PST Jacobian ──
    ctx.pack_and_upload_final_state(traj, n);
    pst_jacobian_gpu(
        &ctx.d_final_state,
        &ctx.d_grad_pst,
        &ctx.d_scratch,
        &ctx.d_v,
        n,
    )?;

    let _n_blocks = (n as u32 + 255) / 256;

    // ── 4. BPTT loop: step from N-1 to 0 ──
    for step in (0..n_steps).rev() {
        let is_last = step == n_steps - 1;

        // Compute pointer offsets for this step
        let state_off = step * 6 * n;
        let drive_off = step * 3 * n;
        let grad_off  = step * 3 * n;

        // Use DevicePtr as raw pointer + offset for kernel
        unsafe {
            use std::ffi::c_float;
            let d_state_step = ctx.d_state.as_ptr().add(state_off) as *const c_float;
            let d_drive_step = ctx.d_drive.as_ptr().add(drive_off) as *const c_float;
            let d_grad_step  = ctx.d_grads.as_ptr().add(grad_off) as *mut c_float;

            bptt_step_gpu_raw(
                d_state_step,
                d_drive_step,
                ctx.d_v.as_ptr() as *mut c_float,
                ctx.d_scratch.as_ptr() as *mut c_float,
                d_grad_step,
                ctx.d_grad_pst.as_ptr() as *const c_float,
                traj.w12, traj.w13, traj.w23,
                traj.dt, traj.drive_gain, traj.alpha, traj.epsilon,
                n, is_last,
            )?;
        }
    }

    // ── 5. Download grad_drives from GPU ──
    let mut grad_drives = vec![0.0f32; n_steps * 3 * n];
    ctx.d_grads.copy_to_host(&mut grad_drives)?;

    Ok(grad_drives)
}

/// Raw (unsafe) wrapper for `cuda_bptt_step` that takes raw pointers.
/// Used inside the BPTT loop for pointer-offset access.
unsafe fn bptt_step_gpu_raw(
    d_state_step: *const f32,
    d_drive_step: *const f32,
    d_v: *mut f32,
    d_scratch: *mut f32,
    d_grad_step: *mut f32,
    d_grad_pst: *const f32,
    w12: f32, w13: f32, w23: f32,
    dt: f32, gain: f32, alpha: f32, eps: f32,
    n: usize, is_last_step: bool,
) -> Result<(), String> {
    use std::ffi::c_int;
    extern "C" {
        fn cuda_bptt_step(
            d_state_step: *const f32,
            d_drive_step: *const f32,
            d_v: *mut f32,
            d_scratch: *mut f32,
            d_grad_step: *mut f32,
            d_grad_pst: *const f32,
            w12: f32, w13: f32, w23: f32,
            dt: f32, gain: f32, alpha: f32, eps: f32,
            n: c_int, is_last_step: c_int,
        ) -> c_int;
    }

    let err = cuda_bptt_step(
        d_state_step, d_drive_step, d_v, d_scratch,
        d_grad_step, d_grad_pst,
        w12, w13, w23, dt, gain, alpha, eps,
        n as c_int, is_last_step as c_int,
    );
    if err == 0 { Ok(()) }
    else { Err(format!("bptt_step_gpu error: {err}")) }
}

// ═══════════════════════════════════════════════════════════
//  GPU Context for Backward
// ═══════════════════════════════════════════════════════════

/// Persistent GPU memory buffers for complete GPU backward pass.
///
/// RAII — все DevicePtr освобождаются при drop.
pub struct CudaBackwardContext {
    pub n: usize,
    pub n_steps: usize,

    // K1 buffers
    pub d_final_state: DevicePtr<f32>,  // [6n]
    pub d_grad_pst:    DevicePtr<f32>,  // [pst_dim]
    pub d_v:           DevicePtr<f32>,  // [6n] current v (updated in-place)

    // Trajectory buffers (step-major)
    pub d_state:  DevicePtr<f32>,  // [n_steps × 6n]
    pub d_drive:  DevicePtr<f32>,  // [n_steps × 3n]
    pub d_grads:  DevicePtr<f32>,  // [n_steps × 3n]

    // Scratch
    pub d_scratch: DevicePtr<f32>, // [max(12, 12 × n_blocks)] for reductions
}

impl CudaBackwardContext {
    /// Allocate ALL persistent GPU buffers.
    pub fn new(n: usize, n_steps: usize) -> Result<Self, String> {
    let pst_dim = pst_linear_dim(n);
    let n_blocks = (n as u32 + 255) / 256;
        let scratch_size = (n_blocks * 12) as usize;

        Ok(Self {
            n,
            n_steps,
            d_final_state: DevicePtr::new(6 * n)?,
            d_grad_pst:    DevicePtr::new(pst_dim)?,
            d_v:           DevicePtr::new(6 * n)?,
            d_state:       DevicePtr::new(n_steps * 6 * n)?,
            d_drive:       DevicePtr::new(n_steps * 3 * n)?,
            d_grads:       DevicePtr::new(n_steps * 3 * n)?,
            d_scratch:     DevicePtr::new(scratch_size.max(12))?,
        })
    }

    /// Upload trajectory to GPU in step-major layout.
    ///
    /// d_state [n_steps × 6n]: for step t, offset t*6n:
    ///   [th1_t(n), th2_t(n), th3_t(n), r1_t(n), r2_t(n), r3_t(n)]
    ///
    /// d_drive [n_steps × 3n]: for step t, offset t*3n:
    ///   [drive1_t(n), drive2_t(n), drive3_t(n)]
    fn upload_trajectory(&mut self, traj: &T3SpikeTrajectory) {
        let n = self.n;
        let ns = self.n_steps;

        // Pack d_state: step-major [th1_t][th2_t][th3_t][r1_t][r2_t][r3_t]
        let mut state = vec![0.0f32; ns * 6 * n];
        for t in 0..ns {
            let off = t * 6 * n;
            let s_off = t * n;
            state[off + 0*n..off + 1*n].copy_from_slice(&traj.th1_pre[s_off..s_off + n]);
            state[off + 1*n..off + 2*n].copy_from_slice(&traj.th2_pre[s_off..s_off + n]);
            state[off + 2*n..off + 3*n].copy_from_slice(&traj.th3_pre[s_off..s_off + n]);
            state[off + 3*n..off + 4*n].copy_from_slice(&traj.r1_pre[s_off..s_off + n]);
            state[off + 4*n..off + 5*n].copy_from_slice(&traj.r2_pre[s_off..s_off + n]);
            state[off + 5*n..off + 6*n].copy_from_slice(&traj.r3_pre[s_off..s_off + n]);
        }
        self.d_state.copy_from_host(&state).unwrap();

        // Pack d_drive: step-major [drive1_t][drive2_t][drive3_t]
        let mut drive = vec![0.0f32; ns * 3 * n];
        for t in 0..ns {
            let off = t * 3 * n;
            let s_off = t * n;
            drive[off + 0*n..off + 1*n].copy_from_slice(&traj.drive1[s_off..s_off + n]);
            drive[off + 1*n..off + 2*n].copy_from_slice(&traj.drive2[s_off..s_off + n]);
            drive[off + 2*n..off + 3*n].copy_from_slice(&traj.drive3[s_off..s_off + n]);
        }
        self.d_drive.copy_from_host(&drive).unwrap();
    }

    /// Pack final state (for K1) and upload.
    fn pack_and_upload_final_state(&mut self, traj: &T3SpikeTrajectory, n: usize) {
        let mut final_state = vec![0.0f32; 6 * n];
        final_state[0*n..1*n].copy_from_slice(&traj.th1_final);
        final_state[1*n..2*n].copy_from_slice(&traj.th2_final);
        final_state[2*n..3*n].copy_from_slice(&traj.th3_final);
        final_state[3*n..4*n].copy_from_slice(&traj.r1_final);
        final_state[4*n..5*n].copy_from_slice(&traj.r2_final);
        final_state[5*n..6*n].copy_from_slice(&traj.r3_final);
        self.d_final_state.copy_from_host(&final_state).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::t3::spike::{t3spike_forward, t3spike_backward};

    /// Debug: compare K1 (PST Jacobian) — v = J^T · grad_pst vs CPU compute_pst_jacobian_v
    #[test]
    fn test_gpu_k1_vs_cpu() {
        if cuda_kernels::get_device_count().unwrap_or(0) == 0 { return; }
        cuda_kernels::set_device(0).unwrap();

        let n_osc = 8;
        let n_steps = 3;
        let n_drives = n_steps * 3 * n_osc;
        let mut drives = vec![0.0f32; n_drives];
        for i in 0..n_drives { drives[i] = ((i as f32) * 0.3).cos() * 1.5; }

        let (pst, traj) = t3spike_forward(n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives);
        let pst_dim = pst.len();

        // ── CPU K1: compute full Jacobian J, then v = J^T · grad_pst ──
        use crate::t3::spike::pst_jacobian;
        let (d_th1_j, d_th2_j, d_th3_j, d_r1_j, d_r2_j, d_r3_j) = pst_jacobian(
            &traj.th1_final, &traj.r1_final,
            &traj.th2_final, &traj.r2_final,
            &traj.th3_final, &traj.r3_final,
            n_osc,
        );
        let cpu_v = crate::t3::spike::compute_pst_jacobian_v(
            &pst,  // use PST as grad to test all features
            &d_th1_j, &d_th2_j, &d_th3_j,
            &d_r1_j, &d_r2_j, &d_r3_j,
            n_osc, pst_dim,
        );

        // ── GPU K1: run pst_jacobian_gpu ──
        let mut ctx = CudaBackwardContext::new(n_osc, n_steps).unwrap();
        ctx.d_grad_pst.copy_from_host(&pst).unwrap();
        ctx.pack_and_upload_final_state(&traj, n_osc);
        pst_jacobian_gpu(&ctx.d_final_state, &ctx.d_grad_pst, &ctx.d_scratch, &ctx.d_v, n_osc).unwrap();

        let mut gpu_v = vec![0.0f32; 6 * n_osc];
        ctx.d_v.copy_to_host(&mut gpu_v).unwrap();

        // ── Compare K1 output ──
        let cpu_v_vec: Vec<f32> = [
            cpu_v.th1.as_slice(),
            cpu_v.th2.as_slice(),
            cpu_v.th3.as_slice(),
            cpu_v.r1.as_slice(),
            cpu_v.r2.as_slice(),
            cpu_v.r3.as_slice(),
        ].concat();

        let mut k1_max_diff = 0.0f32;
        let mut k1_worst = 0;
        for i in 0..6*n_osc {
            let diff = (cpu_v_vec[i] - gpu_v[i]).abs();
            if diff > k1_max_diff { k1_max_diff = diff; k1_worst = i; }
        }
        println!("K1 (PST Jacobian) vs CPU:");
        println!("  max_diff={:.10} at idx {} (group {})", k1_max_diff, k1_worst, k1_worst / n_osc);
        println!("  CPU v[:6]: {:?}", &cpu_v_vec[..6.min(6*n_osc)]);
        println!("  GPU v[:6]: {:?}", &gpu_v[..6.min(6*n_osc)]);

        assert!(k1_max_diff < 2e-4, "K1 mismatch: max_diff={:.10}", k1_max_diff);
        println!("✅ K1 matches");
    }

    /// Debug: compare 1 BPTT step (K3+K2) vs CPU
    #[test]
    fn test_gpu_bptt_one_step() {
        if cuda_kernels::get_device_count().unwrap_or(0) == 0 { return; }
        cuda_kernels::set_device(0).unwrap();

        let n_osc = 8;
        let n_steps = 3;
        let n_drives = n_steps * 3 * n_osc;
        let mut drives = vec![0.0f32; n_drives];
        for i in 0..n_drives { drives[i] = ((i as f32) * 0.3).cos() * 1.5; }

        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives);

        let pst_dim = pst.len();
        let mut grad_pst = vec![0.0f32; pst_dim];
        grad_pst[0] = 1.0;

        // CPU full backward
        let cpu_grad = t3spike_backward(&grad_pst, &traj);

        // GPU full backward
        let mut ctx = CudaBackwardContext::new(n_osc, n_steps).unwrap();
        let gpu_grad = t3spike_backward_cuda(&grad_pst, &traj, &mut ctx).unwrap();

        let mut max_diff = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..cpu_grad.len() {
            let diff = (cpu_grad[i] - gpu_grad[i]).abs();
            if diff > max_diff { max_diff = diff; worst_idx = i; }
        }

        println!("BPTT 3 steps, n_osc=8");
        println!("  Max diff: {:.10} at index {}", max_diff, worst_idx);
        println!("  CPU[{}] = {:.6}, GPU[{}] = {:.6}", worst_idx, cpu_grad[worst_idx], worst_idx, gpu_grad[worst_idx]);
        println!("  First 6 CPU: {:?}", &cpu_grad[..6]);
        println!("  First 6 GPU: {:?}", &gpu_grad[..6]);

        // Debug: compare per-step
        let n = n_osc;
        for step in 0..n_steps {
            let off = step * 3 * n;
            let mut step_max = 0.0f32;
            for i in 0..3*n {
                step_max = step_max.max((cpu_grad[off + i] - gpu_grad[off + i]).abs());
            }
            println!("  Step {} max diff: {:.10}", step, step_max);
        }

        assert!(max_diff < 2e-4, "1-step BPTT mismatch: max_diff={:.10}", max_diff);
        println!("✅ BPTT 1 step: CPU vs GPU match");
    }

    #[test]
    fn test_cpu_gpu_backward_cross_validation_n100() {
        if cuda_kernels::get_device_count().unwrap_or(0) == 0 { return; }
        cuda_kernels::set_device(0).unwrap();

        let n_osc = 16;
        let n_steps = 100;
        let n_drives = n_steps * 3 * n_osc;
        let mut drives = vec![0.0f32; n_drives];
        for i in 0..n_drives { drives[i] = ((i as f32) * 0.1).sin() * 2.0; }

        let (pst, traj) = t3spike_forward(n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0, 0.3, 0.15, 0.08, &drives);
        let pst_dim = pst.len();

        let mut grad_pst = vec![0.0f32; pst_dim];
        for i in 0..pst_dim.min(10) { grad_pst[i] = 1.0; }
        let vel_start = 6 * n_osc;
        if vel_start + 3 * n_osc <= pst_dim {
            for i in 0..n_osc.min(4) {
                grad_pst[vel_start + i] = 0.5;
                grad_pst[vel_start + n_osc + i] = -0.3;
                grad_pst[vel_start + 2 * n_osc + i] = 0.2;
            }
        }
        if pst_dim >= 12 {
            grad_pst[pst_dim - 1] = 1.0;
            grad_pst[pst_dim - 2] = -1.0;
            grad_pst[pst_dim - 3] = 0.5;
        }

        let cpu_grad = t3spike_backward(&grad_pst, &traj);
        let mut ctx = CudaBackwardContext::new(n_osc, n_steps).unwrap();
        let gpu_grad = t3spike_backward_cuda(&grad_pst, &traj, &mut ctx).unwrap();
        let gpu_grad2 = t3spike_backward_cuda(&grad_pst, &traj, &mut ctx).unwrap();

        assert_eq!(cpu_grad.len(), gpu_grad.len());

        let mut max_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        let mut sum_abs_cpu = 0.0f32;
        let mut gpu_self_max_diff = 0.0f32;

        for i in 0..cpu_grad.len() {
            let diff = (cpu_grad[i] - gpu_grad[i]).abs();
            max_diff = max_diff.max(diff);
            sum_abs_diff += diff;
            sum_abs_cpu += cpu_grad[i].abs();
            gpu_self_max_diff = gpu_self_max_diff.max((gpu_grad[i] - gpu_grad2[i]).abs());
        }

        println!("\n=== CPU vs GPU Backward (n={}, steps={}) ===", n_osc, n_steps);
        println!("Max diff:     {:.10}", max_diff);
        println!("Mean diff:    {:.10}", sum_abs_diff / cpu_grad.len() as f32);
        println!("Rel err:      {:.6e}", sum_abs_diff / (sum_abs_cpu + 1e-10));
        println!("GPU self max: {:.10}", gpu_self_max_diff);
        println!("First 6 CPU:  {:?}", &cpu_grad[..6.min(cpu_grad.len())]);
        println!("First 6 GPU:  {:?}", &gpu_grad[..6.min(gpu_grad.len())]);

        assert!(max_diff < 2e-4, "max_diff={:.10}", max_diff);
        assert!(gpu_self_max_diff < 1e-7, "GPU not deterministic");
        println!("✅ All match");
    }
}
