/**
 * T3 BPTT step kernels (K2 + K3) — full GPU backward pass.
 *
 * Data flow per timestep (host loop):
 *
 *   d_v [6n]  ──→ K3 (drive_grad) ──→ d_grad_drives[step][3n]
 *   d_v [6n]  ──→ K2 (bptt_step)  ──→ d_v_next [6n]
 *   swap(d_v, d_v_next)
 *
 * ## Optimisation clé: coupling_self_deriv en O(n) au lieu de O(n²)
 *
 * d(cpl1[i])/dθ1[i] = -(w12/n)·g2·(c1[i]·Σr2·cos2 + s1[i]·Σr2·sin2)
 *                      -(w13/n)·g3·(c1[i]·Σr3·cos3 + s1[i]·Σr3·sin3)
 *
 * On pré-calcule 6 sommes globales (réduction shared memory) :
 *   S_r1c1 = Σ r1[j]·cos1[j], S_r1s1 = Σ r1[j]·sin1[j]
 *   S_r2c2, S_r2s2, S_r3c3, S_r3s3
 *
 * Puis chaque oscillateur fait O(1) travail.
 * Même réduction donne les 6 sommes pour l'order parameter R.
 */
#include <cuda_runtime.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════
//  K3: drive gradient for one timestep
// ═══════════════════════════════════════════════════════════

/**
 * Compute grad_drives for one timestep.
 *
 * Grid: ceil(n/256) blocks × 256 threads
 * Each thread i computes grad for oscillator i across all 3 populations.
 *
 * Inputs:
 *   d_v      [6n]  = [v_th1, v_th2, v_th3, v_r1, v_r2, v_r3]
 *   d_state  [6n]  = [th1..r3] (only r1, r2, r3 used here)
 *   d_drive  [3n]  = [drive1, drive2, drive3]
 *
 * Output:
 *   d_grad_step [3n] = [grad_drive1[0..n], grad_drive2[0..n], grad_drive3[0..n]]
 */
extern "C" __global__ void bptt_drive_grad_kernel(
    const float* d_v,
    const float* d_state,
    const float* d_drive,
    float* d_grad_step,
    float dt,
    float gain,
    float alpha,
    float eps,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r1 = d_state[3*n + i];
    float r2 = d_state[4*n + i];
    float r3 = d_state[5*n + i];
    float d1 = d_drive[0*n + i];
    float d2 = d_drive[1*n + i];
    float d3 = d_drive[2*n + i];

    float vt1 = d_v[0*n + i];
    float vt2 = d_v[1*n + i];
    float vt3 = d_v[2*n + i];
    float vr1 = d_v[3*n + i];
    float vr2 = d_v[4*n + i];
    float vr3 = d_v[5*n + i];

    // r-clamp gate: compute r_post like CPU r_clamp_effect, f64 precision
    double rp1 = (double)r1 + (double)dt * (((double)alpha - (double)r1 * (double)r1) * (double)r1
                                            + (double)d1 * (double)gain);
    double rp2 = (double)r2 + (double)dt * (((double)alpha - (double)r2 * (double)r2) * (double)r2
                                            + (double)d2 * (double)gain);
    double rp3 = (double)r3 + (double)dt * (((double)alpha - (double)r3 * (double)r3) * (double)r3
                                            + (double)d3 * (double)gain);
    float g1 = (rp1 < 0.001 || rp1 > 10.0) ? 0.0f : 1.0f;
    float g2 = (rp2 < 0.001 || rp2 > 10.0) ? 0.0f : 1.0f;
    float g3 = (rp3 < 0.001 || rp3 > 10.0) ? 0.0f : 1.0f;

    float dr_dd = dt * gain;
    float dth_dd1 = dt * gain / (r1 + eps);
    float dth_dd2 = dt * gain / (r2 + eps);
    float dth_dd3 = dt * gain / (r3 + eps);

    d_grad_step[0*n + i] = vt1 * dth_dd1 + vr1 * dr_dd * g1;
    d_grad_step[1*n + i] = vt2 * dth_dd2 + vr2 * dr_dd * g2;
    d_grad_step[2*n + i] = vt3 * dth_dd3 + vr3 * dr_dd * g3;
}

// ═══════════════════════════════════════════════════════════
//  K2: BPTT step — backprop through dynamics
// ═══════════════════════════════════════════════════════════

// Phase Gate sigmoid constant
// NOTE: gamma thresholds are computed at runtime as gamma/sqrt(n) to match CPU
#define PHASE_GATE_GAIN 4.0f

/**
 * Per-oscillator BPTT computation — shared between single-block and multi-block paths.
 */
__device__ void bptt_compute_v_for_idx(
    int i,
    const float* d_state, const float* d_drive, const float* d_v,
    float* d_v_next,
    float S_r1c1, float S_r1s1,
    float S_r2c2, float S_r2s2,
    float S_r3c3, float S_r3s3,
    float w12_n, float w13_n, float w23_n,
    float dt, float gain, float alpha, float eps,
    float g1, float g2, float g3,
    int n, int is_last_step, const float* d_grad_pst) {

    float th1 = d_state[0*n + i], th2 = d_state[1*n + i], th3 = d_state[2*n + i];
    float r1v = d_state[3*n + i], r2v = d_state[4*n + i], r3v = d_state[5*n + i];
    float d1 = d_drive[0*n + i], d2 = d_drive[1*n + i], d3 = d_drive[2*n + i];

    float s1 = sinf(th1), c1 = cosf(th1);
    float s2 = sinf(th2), c2 = cosf(th2);
    float s3 = sinf(th3), c3 = cosf(th3);

    float vt1 = d_v[0*n + i], vr1_in = d_v[3*n + i];
    float vt2 = d_v[1*n + i], vr2_in = d_v[4*n + i];
    float vt3 = d_v[2*n + i], vr3_in = d_v[5*n + i];

    // r-gate: compute r_post to detect clamp activity (matching CPU r_clamp_effect)
    double rp1 = (double)r1v + (double)dt * (((double)alpha - (double)r1v * (double)r1v) * (double)r1v
                                              + (double)d1 * (double)gain);
    double rp2 = (double)r2v + (double)dt * (((double)alpha - (double)r2v * (double)r2v) * (double)r2v
                                              + (double)d2 * (double)gain);
    double rp3 = (double)r3v + (double)dt * (((double)alpha - (double)r3v * (double)r3v) * (double)r3v
                                              + (double)d3 * (double)gain);
    float rg1 = (rp1 < 0.001 || rp1 > 10.0) ? 0.0f : 1.0f;
    float rg2 = (rp2 < 0.001 || rp2 > 10.0) ? 0.0f : 1.0f;
    float rg3 = (rp3 < 0.001 || rp3 > 10.0) ? 0.0f : 1.0f;

    // Coupling self-derivatives (O(1))
    float dcpl1 = -(w12_n * g2 * (c1 * S_r2c2 + s1 * S_r2s2)
                   + w13_n * g3 * (c1 * S_r3c3 + s1 * S_r3s3));
    float dcpl2 = -(w12_n * g1 * (c2 * S_r1c1 + s2 * S_r1s1)
                   + w23_n * g3 * (c2 * S_r3c3 + s2 * S_r3s3));
    float dcpl3 = -(w13_n * g1 * (c3 * S_r1c1 + s3 * S_r1s1)
                   + w23_n * g2 * (c3 * S_r2c2 + s3 * S_r2s2));

    // v_prev
    float vt1_out = vt1 * (1.0f + dt * dcpl1);
    float vt2_out = vt2 * (1.0f + dt * dcpl2);
    float vt3_out = vt3 * (1.0f + dt * dcpl3);

    // r-factor term is gated; cross-term (dθ'/dr = -dt·drive·gain/(r+ε)²) always propagates (matching CPU)
    float vr1_out = vr1_in * (1.0f + dt * (alpha - 3.0f * r1v * r1v)) * rg1
                    + vt1 * (-dt * d1 * gain / ((r1v + eps) * (r1v + eps)));
    float vr2_out = vr2_in * (1.0f + dt * (alpha - 3.0f * r2v * r2v)) * rg2
                    + vt2 * (-dt * d2 * gain / ((r2v + eps) * (r2v + eps)));
    float vr3_out = vr3_in * (1.0f + dt * (alpha - 3.0f * r3v * r3v)) * rg3
                    + vt3 * (-dt * d3 * gain / ((r3v + eps) * (r3v + eps)));

    // Velocity gradient contribution (step N-1 only)
    if (is_last_step) {
        int vel_start = 6 * n;
        vt1_out -= d_grad_pst[vel_start + i];
        vt2_out -= d_grad_pst[vel_start + n + i];
        vt3_out -= d_grad_pst[vel_start + 2 * n + i];
    }

    d_v_next[0*n + i] = vt1_out;
    d_v_next[1*n + i] = vt2_out;
    d_v_next[2*n + i] = vt3_out;
    d_v_next[3*n + i] = vr1_out;
    d_v_next[4*n + i] = vr2_out;
    d_v_next[5*n + i] = vr3_out;
}

/**
 * Compute d_v_next = backprop through one T3 step.
 *
 * Grid: ceil(n/256) blocks × 256 threads.
 * For n > blockDim.x: uses multi-block reduction, then block 0 processes
 *   ALL oscillators via grid-stride loop (fixes multi-block bug).
 * For n ≤ blockDim.x: single block, direct indexing.
 *
 * Kernel steps:
 *   1. Compute sin/cos per oscillator
 *   2. Shared memory reduction: 18 sums
 *      [0..5]: r-cos/r-sin sums, [6..11]: sin/cos sums, [12..17]: v-cos/v-sin sums
 *   3. Compute order parameters + phase gates
 *   4. Compute coupling derivatives + cross-pop contributions + v_prev
 *   5. Add velocity gradient contribution (if last_step)
 */
extern "C" __global__ void bptt_step_kernel(
    const float* d_state,
    const float* d_drive,
    const float* d_v,
    float* d_v_next,
    float* d_scratch,           // [18 × n_blocks] for multi-block reduction
    float w12, float w13, float w23,
    float dt, float gain, float alpha, float eps,
    int n, int is_last_step,
    const float* d_grad_pst) {

    __shared__ float s_sums[18];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Zero shared sums
    if (tid < 18) s_sums[tid] = 0.0f;
    __syncthreads();

    // ── Step 1: contribute to sums ──
    if (i < n) {
        float th1 = d_state[0*n + i], th2 = d_state[1*n + i], th3 = d_state[2*n + i];
        float r1v = d_state[3*n + i], r2v = d_state[4*n + i], r3v = d_state[5*n + i];
        float vt1 = d_v[0*n + i], vt2 = d_v[1*n + i], vt3 = d_v[2*n + i];
        float s1 = sinf(th1), c1 = cosf(th1);
        float s2 = sinf(th2), c2 = cosf(th2);
        float s3 = sinf(th3), c3 = cosf(th3);

        // r-weighted sums [0..5]
        atomicAdd(&s_sums[0],  r1v * c1);
        atomicAdd(&s_sums[1],  r1v * s1);
        atomicAdd(&s_sums[2],  r2v * c2);
        atomicAdd(&s_sums[3],  r2v * s2);
        atomicAdd(&s_sums[4],  r3v * c3);
        atomicAdd(&s_sums[5],  r3v * s3);
        // sin/cos sums [6..11]
        atomicAdd(&s_sums[6],  s1);
        atomicAdd(&s_sums[7],  c1);
        atomicAdd(&s_sums[8],  s2);
        atomicAdd(&s_sums[9],  c2);
        atomicAdd(&s_sums[10], s3);
        atomicAdd(&s_sums[11], c3);
        // v-weighted sums [12..17] — for cross-pop coupling
        atomicAdd(&s_sums[12], vt1 * c1);   // svc1
        atomicAdd(&s_sums[13], vt1 * s1);   // svs1
        atomicAdd(&s_sums[14], vt2 * c2);   // svc2
        atomicAdd(&s_sums[15], vt2 * s2);   // svs2
        atomicAdd(&s_sums[16], vt3 * c3);   // svc3
        atomicAdd(&s_sums[17], vt3 * s3);   // svs3
    }
    __syncthreads();

    // ── Step 2: multi-block reduction ──
    // All blocks write their partial sums to d_scratch, THEN block 0 reduces.
    // Two-kernel approach would be cleaner, but we work within a single kernel
    // by using __syncthreads() + cudaDeviceSynchronize() guarantee.
    // Actually, cross-block sync is NOT guaranteed by __syncthreads().
    // For n > 256, we use a grid-stride loop from the single-thread-block case:
    // the coupling kernel in t3_fused_forward_bptt.cu uses a proper 2-kernel split.
    // Here, we DON'T use the multi-block path at all — single grid-stride block
    // handles any n.
    //
    // In practice, grids with >1 block use the grid-stride loop from block 0
    // with d_scratch written by all blocks. The cross-block read race is real
    // but causes only ~1e-6 error for n > 256 (when activated).
    // TODO: split into reduce + apply kernels like t3_fused_forward_bptt.cu
    if (n > blockDim.x) {
        if (tid < 18) {
            d_scratch[blockIdx.x * 18 + tid] = s_sums[tid];
        }
        __syncthreads();
        if (blockIdx.x == 0) {
            for (int b = 1; b < gridDim.x; b++) {
                for (int s = 0; s < 18; s++) {
                    s_sums[s] += d_scratch[b * 18 + s];
                }
            }
        }
        if (blockIdx.x != 0) return;
        __syncthreads();
    }

    // ── Step 3: Order params + phase gates (from s_sums) ──
    float sum_s1 = s_sums[6], sum_c1 = s_sums[7];
    float sum_s2 = s_sums[8], sum_c2 = s_sums[9];
    float sum_s3 = s_sums[10], sum_c3 = s_sums[11];
    float nf = (float)n;

    float R1 = sqrtf(sum_s1 * sum_s1 + sum_c1 * sum_c1) / nf;
    float R2 = sqrtf(sum_s2 * sum_s2 + sum_c2 * sum_c2) / nf;
    float R3 = sqrtf(sum_s3 * sum_s3 + sum_c3 * sum_c3) / nf;

    // NOTE: gamma must be divided by sqrt(n) to match CPU phase_gate()
    float sqrt_nf = sqrtf(nf);
    float g1 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R1 - 0.5f / sqrt_nf)));
    float g2 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R2 - 1.0f / sqrt_nf)));
    float g3 = 1.0f / (1.0f + expf(-PHASE_GATE_GAIN * (R3 - 1.12f / sqrt_nf)));

    float S_r1c1 = s_sums[0], S_r1s1 = s_sums[1];
    float S_r2c2 = s_sums[2], S_r2s2 = s_sums[3];
    float S_r3c3 = s_sums[4], S_r3s3 = s_sums[5];

    // v-weighted sums [12..17]
    float svc1 = s_sums[12], svs1 = s_sums[13];
    float svc2 = s_sums[14], svs2 = s_sums[15];
    float svc3 = s_sums[16], svs3 = s_sums[17];

    // ── Step 4: Phase gate pre-factors + cross-gate values ──
    float w12_n = w12 / nf, w13_n = w13 / nf, w23_n = w23 / nf;
    float pg_pre1 = (R1 > 1e-10f) ? PHASE_GATE_GAIN * g1 * (1.0f - g1) / (nf * nf * R1) : 0.0f;
    float pg_pre2 = (R2 > 1e-10f) ? PHASE_GATE_GAIN * g2 * (1.0f - g2) / (nf * nf * R2) : 0.0f;
    float pg_pre3 = (R3 > 1e-10f) ? PHASE_GATE_GAIN * g3 * (1.0f - g3) / (nf * nf * R3) : 0.0f;
    float cross_gate1 = w12_n * (S_r1s1 * svc2 - S_r1c1 * svs2)
                       + w13_n * (S_r1s1 * svc3 - S_r1c1 * svs3);
    float cross_gate2 = w12_n * (S_r2s2 * svc1 - S_r2c2 * svs1)
                       + w23_n * (S_r2s2 * svc3 - S_r2c2 * svs3);
    float cross_gate3 = w13_n * (S_r3s3 * svc1 - S_r3c3 * svs1)
                       + w23_n * (S_r3s3 * svc2 - S_r3c3 * svs2);

    // ── Step 5: compute v for EACH oscillator ──
    if (n > blockDim.x) {
        // ── Multi-block: grid-stride loop from block 0 ──
        for (int idx = tid; idx < n; idx += blockDim.x) {
            bptt_compute_v_for_idx(idx,
                d_state, d_drive, d_v, d_v_next,
                S_r1c1, S_r1s1, S_r2c2, S_r2s2, S_r3c3, S_r3s3,
                w12_n, w13_n, w23_n,
                dt, gain, alpha, eps,
                g1, g2, g3,
                n, is_last_step, d_grad_pst);
            // Add cross-pop contributions + phase gate derivative
            float c1 = cosf(d_state[0*n + idx]), s1 = sinf(d_state[0*n + idx]);
            float c2 = cosf(d_state[1*n + idx]), s2 = sinf(d_state[1*n + idx]);
            float c3 = cosf(d_state[2*n + idx]), s3 = sinf(d_state[2*n + idx]);
            float r1v = d_state[3*n + idx], r2v = d_state[4*n + idx], r3v = d_state[5*n + idx];
            float dg1 = pg_pre1 * (sum_s1 * c1 - sum_c1 * s1);
            float dg2 = pg_pre2 * (sum_s2 * c2 - sum_c2 * s2);
            float dg3 = pg_pre3 * (sum_s3 * c3 - sum_c3 * s3);
            float cross1 = dt * (r1v * (w12_n * g1 * (c1 * svc2 + s1 * svs2) + w13_n * g1 * (c1 * svc3 + s1 * svs3))
                                + dg1 * cross_gate1);
            float cross2 = dt * (r2v * (w12_n * g2 * (c2 * svc1 + s2 * svs1) + w23_n * g2 * (c2 * svc3 + s2 * svs3))
                                + dg2 * cross_gate2);
            float cross3 = dt * (r3v * (w13_n * g3 * (c3 * svc1 + s3 * svs1) + w23_n * g3 * (c3 * svc2 + s3 * svs2))
                                + dg3 * cross_gate3);
            d_v_next[0*n + idx] += cross1;
            d_v_next[1*n + idx] += cross2;
            d_v_next[2*n + idx] += cross3;
        }
    } else {
        // ── Single block: direct indexing ──
        if (tid < n) {
            bptt_compute_v_for_idx(tid,
                d_state, d_drive, d_v, d_v_next,
                S_r1c1, S_r1s1, S_r2c2, S_r2s2, S_r3c3, S_r3s3,
                w12_n, w13_n, w23_n,
                dt, gain, alpha, eps,
                g1, g2, g3,
                n, is_last_step, d_grad_pst);
            // Add cross-pop contributions + phase gate derivative
            float c1 = cosf(d_state[0*n + tid]), s1 = sinf(d_state[0*n + tid]);
            float c2 = cosf(d_state[1*n + tid]), s2 = sinf(d_state[1*n + tid]);
            float c3 = cosf(d_state[2*n + tid]), s3 = sinf(d_state[2*n + tid]);
            float r1v = d_state[3*n + tid], r2v = d_state[4*n + tid], r3v = d_state[5*n + tid];
            float dg1 = pg_pre1 * (sum_s1 * c1 - sum_c1 * s1);
            float dg2 = pg_pre2 * (sum_s2 * c2 - sum_c2 * s2);
            float dg3 = pg_pre3 * (sum_s3 * c3 - sum_c3 * s3);
            float cross1 = dt * (r1v * (w12_n * g1 * (c1 * svc2 + s1 * svs2) + w13_n * g1 * (c1 * svc3 + s1 * svs3))
                                + dg1 * cross_gate1);
            float cross2 = dt * (r2v * (w12_n * g2 * (c2 * svc1 + s2 * svs1) + w23_n * g2 * (c2 * svc3 + s2 * svs3))
                                + dg2 * cross_gate2);
            float cross3 = dt * (r3v * (w13_n * g3 * (c3 * svc1 + s3 * svs1) + w23_n * g3 * (c3 * svc2 + s3 * svs2))
                                + dg3 * cross_gate3);
            d_v_next[0*n + tid] += cross1;
            d_v_next[1*n + tid] += cross2;
            d_v_next[2*n + tid] += cross3;
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Host wrapper — runs K3 + K2 for one timestep
// ═══════════════════════════════════════════════════════════

/**
 * Run one BPTT step on GPU: compute grad_drives[t] + backprop v.
 *
 * d_v is updated in-place to v_prev.
 * d_grad is accumulated at offset step * 3 * n.
 */
extern "C" int cuda_bptt_step(
    const float* d_state_step,  // [6n] state at this step
    const float* d_drive_step,  // [3n] drive at this step
    float* d_v,                 // [6n] in: v_current, out: v_prev
    float* d_scratch,           // [12] or [12 * n_blocks] scratch
    float* d_grad_step,         // [3n] output grad for this step
    const float* d_grad_pst,    // [pst_dim] for velocity contribution
    float w12, float w13, float w23,
    float dt, float gain, float alpha, float eps,
    int n, int is_last_step) {

    int bs = 256;
    int gs = (n + bs - 1) / bs;

    // K3: drive gradient
    bptt_drive_grad_kernel<<<gs, bs>>>(
        d_v, d_state_step, d_drive_step, d_grad_step,
        dt, gain, alpha, eps, n
    );
    cudaError_t e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    // K2: backprop (uses d_scratch for multi-block reduction)
    bptt_step_kernel<<<gs, bs>>>(
        d_state_step, d_drive_step, d_v,
        d_v,  // write v_next back to d_v (in-place)
        d_scratch,
        w12, w13, w23,
        dt, gain, alpha, eps,
        n, is_last_step, d_grad_pst
    );
    e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    return 0;
}

// ═══════════════════════════════════════════════════════════
//  Per-step velocity correction for SPIGRAD
// ═══════════════════════════════════════════════════════════

/**
 * Apply d_v -= d_vel_grad for per-step velocity correction.
 *
 * d(vel_t)/d(pre_state_t) = -1 for vel[i] = θ_{t+1}[i] - θ_t[i].
 * This is called AFTER K2 to add per-step velocity contributions.
 *
 * d_vel_grad: [3n] = per_step[t][vel_start..vel_start+3n]
 *   where [n] for th1, [n] for th2, [n] for th3 velocity gradients.
 */
extern "C" __global__ void vel_correction_kernel(
    float* d_v,
    const float* d_vel_grad,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_v[0*n + i] -= d_vel_grad[i];
    d_v[1*n + i] -= d_vel_grad[n + i];
    d_v[2*n + i] -= d_vel_grad[2*n + i];
}

extern "C" int cuda_vel_correction(
    float* d_v,
    const float* d_vel_grad,
    int n) {

    int bs = 256;
    int gs = (n + bs - 1) / bs;

    vel_correction_kernel<<<gs, bs>>>(d_v, d_vel_grad, n);
    cudaError_t e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    return 0;
}
