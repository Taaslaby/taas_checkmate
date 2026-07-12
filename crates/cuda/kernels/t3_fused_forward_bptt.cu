/**
 * t3_fused_forward_bptt.cu — Fused T3 forward + BPTT backward on GPU.
 *
 * ## What this fuses
 *
 * Traditionally, the T3 forward pass runs on CPU (Avx2Backend), saving the
 * full trajectory to RAM.  The backward pass then uploads that trajectory
 * to GPU (H2D) before running the BPTT kernels.
 *
 * This file provides a **fully GPU-native** pipeline:
 *
 *   Host:  upload drives (H2D once)
 *          ↓
 *   GPU:   loop over steps:
 *            prepare (sin/cos) → coupling → update → save state
 *          ↓
 *   GPU:   PST Jacobian (K1)
 *          ↓
 *   GPU:   BPTT loop (K3+K2) — reads trajectory from VRAM
 *          ↓
 *   Host:  download grad_drives (D2H once)
 *
 * **Zero H2D between forward and backward.**  The trajectory never
 * touches host memory.
 *
 * ## Trajectory format (matches CPU layout)
 *
 * For step t, the trajectory saves the state BEFORE the update.
 * This is what `bptt_step_kernel` (in t3_bptt_step.cu) expects.
 *
 *   d_traj_state [t * 6n .. (t+1) * 6n] = state at step t (pre-update)
 *   d_traj_drive [t * 3n .. (t+1) * 3n] = drives at step t
 *
 * After the last step, d_state holds the final state (post-update),
 * which is copied to d_final_state for PST Jacobian K1.
 *
 * ## Buffer layout (packed, step-major)
 *
 *   d_state [6n]   = [th1, th2, th3, r1, r2, r3]     — current state
 *   d_drive [3n]   = [drive1, drive2, drive3]         — current drives
 *   d_sincos [6n]  = [sin1, cos1, sin2, cos2, sin3, cos3]
 *   d_coupling [3n]= [coupling1, coupling2, coupling3]
 *   d_omega [3n]   = [omega1, omega2, omega3]         — natural frequencies
 *
 *   d_traj_state [n_steps × 6n]  — forward trajectory (step-major)
 *   d_grad_drives [n_steps × 3n] — output gradients
 *
 *   d_scratch [12 × max_blocks]  — reduction scratch
 *
 * ## PST Jacobian buffers (shared with K1)
 *
 *   d_final_state [6n] — copy of final state for PST Jacobian
 *   d_grad_pst [pst_dim] — input loss gradient (final step only)
 *   d_per_step_grad [n_steps × pst_dim] — per-step PST gradients (SPIGRAD, optional, NULL = skip)
 *   d_v [6n] — output v = J^T · grad_pst  (also used as BPTT state)
 *   d_k1_sums [12] — K1 reduction scratch
 */
#include <cuda_runtime.h>
#include <math.h>

#define PHASE_GATE_GAIN 4.0f
#define GAMMA1 0.5f
#define GAMMA2 1.0f
#define GAMMA3 1.12f
#define EPS 0.1f

// Forward declarations for functions defined in other .cu files.
extern "C" int cuda_pst_jacobian(
    const float* d_final_state,
    const float* d_grad_pst,
    float* d_sums,
    float* d_v,
    int n);

extern "C" int cuda_pst_jacobian_add(
    const float* d_state,
    const float* d_grad_pst,
    float* d_sums,
    float* d_v,
    int n);

extern "C" int cuda_bptt_step(
    const float* d_state_step,
    const float* d_drive_step,
    float* d_v,
    float* d_scratch,
    float* d_grad_step,
    const float* d_grad_pst,
    float w12, float w13, float w23,
    float dt, float gain, float alpha, float eps,
    int n, int is_last_step);

// ═══════════════════════════════════════════════════════════
//  KERNEL A1: Coupling reduction — compute 12 sums
// ═══════════════════════════════════════════════════════════
//
// Single-block kernel: each thread iterates with stride=blockDim.x
// over ALL n oscillators, accumulating into shared memory.
// Then writes the 12 sums to d_sums.
//
// No multi-block race — only 1 block, all threads participate.

extern "C" __global__ void t3_coupling_reduce_kernel(
    const float* d_sincos,      // [6n] sin1,cos1,sin2,cos2,sin3,cos3
    const float* d_state,       // [6n] th1,th2,th3,r1,r2,r3 (uses r1,r2,r3)
    float* d_sums,              // [12] output sums
    int n) {

    __shared__ float s_sums[12];
    int tid = threadIdx.x;

    for (int s = tid; s < 12; s += blockDim.x) {
        s_sums[s] = 0.0f;
    }
    __syncthreads();

    // Grid-stride loop: each thread processes multiple oscillators
    for (int i = tid; i < n; i += blockDim.x) {
        float s1 = d_sincos[0*n + i], c1 = d_sincos[1*n + i];
        float s2 = d_sincos[2*n + i], c2 = d_sincos[3*n + i];
        float s3 = d_sincos[4*n + i], c3 = d_sincos[5*n + i];
        float r1_val = d_state[3*n + i], r2_val = d_state[4*n + i], r3_val = d_state[5*n + i];

        atomicAdd(&s_sums[0],  r1_val * s1);
        atomicAdd(&s_sums[1],  r1_val * c1);
        atomicAdd(&s_sums[2],  r2_val * s2);
        atomicAdd(&s_sums[3],  r2_val * c2);
        atomicAdd(&s_sums[4],  r3_val * s3);
        atomicAdd(&s_sums[5],  r3_val * c3);
        atomicAdd(&s_sums[6],  s1);
        atomicAdd(&s_sums[7],  c1);
        atomicAdd(&s_sums[8],  s2);
        atomicAdd(&s_sums[9],  c2);
        atomicAdd(&s_sums[10], s3);
        atomicAdd(&s_sums[11], c3);
    }
    __syncthreads();

    if (tid < 12) d_sums[tid] = s_sums[tid];
}

// ═══════════════════════════════════════════════════════════
//  KERNEL A2: Coupling apply — compute per-oscillator coupling
// ═══════════════════════════════════════════════════════════
//
// Reads pre-computed d_sums from A1. Computes phase gates + coupling.
// One thread per oscillator — fast.

extern "C" __global__ void t3_coupling_apply_kernel(
    const float* d_sincos,      // [6n] sin1,cos1,sin2,cos2,sin3,cos3
    float* d_coupling,          // [3n] output coupling1,coupling2,coupling3
    const float* d_sums,        // [12] sums from A1
    float w12, float w13, float w23,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Read pre-computed sums
    float S_r1s1 = d_sums[0], S_r1c1 = d_sums[1];
    float S_r2s2 = d_sums[2], S_r2c2 = d_sums[3];
    float S_r3s3 = d_sums[4], S_r3c3 = d_sums[5];
    float sum_s1 = d_sums[6], sum_c1 = d_sums[7];
    float sum_s2 = d_sums[8], sum_c2 = d_sums[9];
    float sum_s3 = d_sums[10], sum_c3 = d_sums[11];
    float nf = (float)n;

    // Phase gates
    float R1 = sqrtf(sum_s1 * sum_s1 + sum_c1 * sum_c1) / nf;
    float R2 = sqrtf(sum_s2 * sum_s2 + sum_c2 * sum_c2) / nf;
    float R3 = sqrtf(sum_s3 * sum_s3 + sum_c3 * sum_c3) / nf;

    float g1 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R1 - GAMMA1)));
    float g2 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R2 - GAMMA2)));
    float g3 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R3 - GAMMA3)));

    // Per-oscillator coupling (divide by n — sums are raw, not averages)
    float w12_n = w12 / nf, w13_n = w13 / nf, w23_n = w23 / nf;
    float s1 = d_sincos[0*n + i], c1 = d_sincos[1*n + i];
    float s2 = d_sincos[2*n + i], c2 = d_sincos[3*n + i];
    float s3 = d_sincos[4*n + i], c3 = d_sincos[5*n + i];

    d_coupling[0*n + i] = w12_n * g2 * (c1 * S_r2s2 - s1 * S_r2c2)
                        + w13_n * g3 * (c1 * S_r3s3 - s1 * S_r3c3);
    d_coupling[1*n + i] = w12_n * g1 * (c2 * S_r1s1 - s2 * S_r1c1)
                        + w23_n * g3 * (c2 * S_r3s3 - s2 * S_r3c3);
    d_coupling[2*n + i] = w13_n * g1 * (c3 * S_r1s1 - s3 * S_r1c1)
                        + w23_n * g2 * (c3 * S_r2s2 - s3 * S_r2c2);
}

// ═══════════════════════════════════════════════════════════
//  KERNEL B: sin/cos from packed state
// ═══════════════════════════════════════════════════════════

extern "C" __global__ void t3_packed_prepare_kernel(
    const float* d_state,   // [6n] = th1,th2,th3,r1,r2,r3 (uses th1,th2,th3)
    float* d_sincos,        // [6n] = sin1,cos1,sin2,cos2,sin3,cos3
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float th1 = d_state[0*n + i];
    d_sincos[0*n + i] = sinf(th1);
    d_sincos[1*n + i] = cosf(th1);

    float th2 = d_state[1*n + i];
    d_sincos[2*n + i] = sinf(th2);
    d_sincos[3*n + i] = cosf(th2);

    float th3 = d_state[2*n + i];
    d_sincos[4*n + i] = sinf(th3);
    d_sincos[5*n + i] = cosf(th3);
}

// ═══════════════════════════════════════════════════════════
//  KERNEL C: Stuart-Landau + Kuramoto update from packed state
// ═══════════════════════════════════════════════════════════

extern "C" __global__ void t3_packed_update_kernel(
    float* d_state,              // [6n] = th1,th2,th3,r1,r2,r3 (in/out)
    const float* d_omega,        // [3n] = omega1,omega2,omega3
    const float* d_coupling,     // [3n] = coupling1,coupling2,coupling3
    const float* d_drive,        // [3n] = drive1,drive2,drive3
    float dt, float alpha, float gain,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float eps = 0.1f;

    // ── Population 1 ──
    // IMPORTANT: phase first (uses OLD r), amplitude second — matches SIMD t3_update
    float r1 = d_state[3*n + i];
    float d1 = d_drive[0*n + i];
    float cpl1 = d_coupling[0*n + i];
    float omega1 = d_omega[0*n + i];
    float th1 = d_state[0*n + i];

    float drive_phase1 = gain * d1 / (r1 + eps);
    th1 = th1 + dt * (omega1 + cpl1 + drive_phase1);
    // Wrap to [0, 2π) then shift to [-π, π)
    float wrapped1 = fmodf(th1 + (float)M_PI, 2.0f * (float)M_PI);
    if (wrapped1 < 0) wrapped1 += 2.0f * (float)M_PI;
    d_state[0*n + i] = wrapped1 - (float)M_PI;

    float r1_dot = (alpha - r1 * r1) * r1 + d1 * gain;
    r1 += dt * r1_dot;
    d_state[3*n + i] = fminf(fmaxf(r1, 0.001f), 10.0f);

    // ── Population 2 ──
    float r2 = d_state[4*n + i];
    float d2 = d_drive[1*n + i];
    float cpl2 = d_coupling[1*n + i];
    float omega2 = d_omega[1*n + i];
    float th2 = d_state[1*n + i];

    float drive_phase2 = gain * d2 / (r2 + eps);
    th2 = th2 + dt * (omega2 + cpl2 + drive_phase2);
    float wrapped2 = fmodf(th2 + (float)M_PI, 2.0f * (float)M_PI);
    if (wrapped2 < 0) wrapped2 += 2.0f * (float)M_PI;
    d_state[1*n + i] = wrapped2 - (float)M_PI;

    float r2_dot = (alpha - r2 * r2) * r2 + d2 * gain;
    r2 += dt * r2_dot;
    d_state[4*n + i] = fminf(fmaxf(r2, 0.001f), 10.0f);

    // ── Population 3 ──
    float r3 = d_state[5*n + i];
    float d3 = d_drive[2*n + i];
    float cpl3 = d_coupling[2*n + i];
    float omega3 = d_omega[2*n + i];
    float th3 = d_state[2*n + i];

    float drive_phase3 = gain * d3 / (r3 + eps);
    th3 = th3 + dt * (omega3 + cpl3 + drive_phase3);
    float wrapped3 = fmodf(th3 + (float)M_PI, 2.0f * (float)M_PI);
    if (wrapped3 < 0) wrapped3 += 2.0f * (float)M_PI;
    d_state[2*n + i] = wrapped3 - (float)M_PI;

    float r3_dot = (alpha - r3 * r3) * r3 + d3 * gain;
    r3 += dt * r3_dot;
    d_state[5*n + i] = fminf(fmaxf(r3, 0.001f), 10.0f);
}

// ═══════════════════════════════════════════════════════════
//  KERNEL D: Save pre-state to trajectory buffer (D2D copy)
// ═══════════════════════════════════════════════════════════
//
// NOTE: drives are not saved — BPTT reads them directly from d_all_drives.

extern "C" __global__ void t3_save_state_kernel(
    const float* d_state,        // [6n] current state
    float* d_traj_state,         // [n_steps × 6n] trajectory (output)
    int step,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int off = step * 6 * n;
    d_traj_state[off + 0*n + i] = d_state[0*n + i];
    d_traj_state[off + 1*n + i] = d_state[1*n + i];
    d_traj_state[off + 2*n + i] = d_state[2*n + i];
    d_traj_state[off + 3*n + i] = d_state[3*n + i];
    d_traj_state[off + 4*n + i] = d_state[4*n + i];
    d_traj_state[off + 5*n + i] = d_state[5*n + i];
}

// ═══════════════════════════════════════════════════════════
//  KERNEL E: Copy final state for PST Jacobian
// ═══════════════════════════════════════════════════════════

extern "C" __global__ void t3_copy_final_state_kernel(
    const float* d_state,   // [6n]
    float* d_final_state,   // [6n]
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_final_state[0*n + i] = d_state[0*n + i];
    d_final_state[1*n + i] = d_state[1*n + i];
    d_final_state[2*n + i] = d_state[2*n + i];
    d_final_state[3*n + i] = d_state[3*n + i];
    d_final_state[4*n + i] = d_state[4*n + i];
    d_final_state[5*n + i] = d_state[5*n + i];
}

// ═══════════════════════════════════════════════════════════
//  HOST ENTRY POINT: Full fused forward + BPTT backward
// ═══════════════════════════════════════════════════════════
//
// Orchestrates 3 phases entirely on GPU:
//
//   Phase 1 (Forward):  for step in 0..n_steps:
//                          save PRE-state to trajectory
//                          prepare (sin/cos)
//                          coupling (reduce → apply)
//                          update (Stuart-Landau + Kuramoto)
//                        sync
//
//   Phase 2 (K1):       PST Jacobian from final state
//
//   Phase 3 (BPTT):     for step in rev(0..n_steps):
//                          K3 + K2: grad_drives[step] + v_prev

extern "C" int cuda_t3_fused_forward_bptt(
    const float* d_all_drives,      // [n_steps × 3n]
    float* d_grad_drives,           // [n_steps × 3n] output
    float* d_state,                 // [6n] initial → final state
    float* d_sincos,                // [6n] workspace
    float* d_coupling,              // [3n] workspace
    const float* d_omega,           // [3n]
    float* d_traj_state,            // [n_steps × 6n]
    float* d_final_state,           // [6n]
    const float* d_grad_pst,        // [pst_dim] final step gradient
    const float* d_per_step_grad,   // [n_steps × pst_dim] per-step gradients or NULL
    float* d_v,                     // [6n]
    float* d_scratch,               // [12] for coupling reduction sums
    float* d_k1_sums,               // [12] for PST Jacobian sums
    float w12, float w13, float w23,
    float dt, float gain, float alpha, float eps,
    int n, int n_steps) {

    int bs = 256;
    int gs = (n + bs - 1) / bs;
    int one_block_gs = 1;
    cudaError_t e;

    // ═══════════════════════════════════════════════════════
    // Phase 1: Forward pass — all n_steps on GPU
    // ═══════════════════════════════════════════════════════

    for (int step = 0; step < n_steps; step++) {
        const float* d_drive_step = d_all_drives + step * 3 * n;

        // 1a. Save PRE-state to trajectory (BEFORE update — matches CPU layout!)
        t3_save_state_kernel<<<gs, bs>>>(d_state, d_traj_state, step, n);
        e = cudaGetLastError();
        if (e) return (int)e;

        // 1b. Compute sin/cos from current state
        t3_packed_prepare_kernel<<<gs, bs>>>(d_state, d_sincos, n);
        e = cudaGetLastError();
        if (e) return (int)e;

        // 1c. Coupling reduction
        t3_coupling_reduce_kernel<<<one_block_gs, bs>>>(d_sincos, d_state, d_scratch, n);
        e = cudaGetLastError();
        if (e) return (int)e;

        // 1d. Coupling apply
        t3_coupling_apply_kernel<<<gs, bs>>>(d_sincos, d_coupling, d_scratch,
                                              w12, w13, w23, n);
        e = cudaGetLastError();
        if (e) return (int)e;

        // 1e. Update
        t3_packed_update_kernel<<<gs, bs>>>(d_state, d_omega, d_coupling,
                                             d_drive_step, dt, alpha, gain, n);
        e = cudaGetLastError();
        if (e) return (int)e;
    }

    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e) return (int)e;

    // ═══════════════════════════════════════════════════════
    // Phase 2: PST Jacobian (K1) — final gradient
    // ═══════════════════════════════════════════════════════

    t3_copy_final_state_kernel<<<gs, bs>>>(d_state, d_final_state, n);
    e = cudaGetLastError();
    if (e) return (int)e;

    int r = cuda_pst_jacobian(d_final_state, d_grad_pst, d_k1_sums, d_v, n);
    if (r) return r;

    // ═══════════════════════════════════════════════════════
    // Phase 3: BPTT backward (K3 + K2) — reverse loop
    // ═══════════════════════════════════════════════════════

    // PST dimension: sin1/cos1(2n) + sin2/cos2(2n) + sin3/cos3(2n) + vel(3n) + r(3n)
    // + sin²1/cos²1(2n) + sin²2/cos²2(2n) + sin²3/cos²3(2n) + R12/R23/R13(3) + means(9)
    // = 18n + 12
    int pst_dim = 18 * n + 12;

    for (int step = n_steps - 1; step >= 0; step--) {
        int is_last = (step == n_steps - 1) ? 1 : 0;

        const float* d_state_step = d_traj_state + step * 6 * n;
        const float* d_drive_step = d_all_drives + step * 3 * n;
        float* d_grad_step = d_grad_drives + step * 3 * n;

        // Per-step PST Jacobian (K1') — adds J_t^T · per_step[t] to d_v
        if (d_per_step_grad != NULL) {
            const float* d_ps = d_per_step_grad + step * pst_dim;
            r = cuda_pst_jacobian_add(d_state_step, d_ps, d_k1_sums, d_v, n);
            if (r) return r;
        }

        r = cuda_bptt_step(
            d_state_step, d_drive_step,
            d_v,
            d_scratch,
            d_grad_step,
            d_grad_pst,
            w12, w13, w23,
            dt, gain, alpha, eps,
            n, is_last);
        if (r) return r;
    }

    return 0;
}
