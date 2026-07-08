/**
 * T3 BPTT Jacobian kernels (K1).
 *
 * Two-kernel approach:
 *   Kernel A — reduction: compute cross-order sums + mean sums
 *   Kernel B — compute v = J^T · grad_pst for each state variable
 *
 * ## PST feature layout (pst_dim = 18n + 12)
 *
 *   0..2n:       sin1[0], cos1[0], ... (interleaved)
 *   2n..4n:      sin2, cos2 (interleaved)
 *   4n..6n:      sin3, cos3 (interleaved)
 *   6n..7n:      vel1[0..n]
 *   7n..8n:      vel2[0..n]
 *   8n..9n:      vel3[0..n]
 *   9n..10n:     r1[0..n]
 *   10n..11n:    r2[0..n]
 *   11n..12n:    r3[0..n]
 *   12n..14n:    sin²1, cos²1 (interleaved)
 *   14n..16n:    sin²2, cos²2 (interleaved)
 *   16n..18n:    sin²3, cos²3 (interleaved)
 *   18n+0:       R12
 *   18n+1:       R23
 *   18n+2:       R13
 *   18n+3..+11:  mean_sin1, mean_cos1, mean_r1, mean_sin2, ...
 */
#include <cuda_runtime.h>
#include <math.h>

// ─── Kernel A: compute cross-order + mean sums ───────────

/**
 * Compute sums needed for cross-order params and mean features.
 *
 * Output packed: [S12, C12, S23, C23, S13, C13, sum_s1, sum_c1, sum_s2, sum_c2, sum_s3, sum_c3]
 *   where S12 = Σ sin(θ1[j]-θ2[j]), C12 = Σ cos(θ1[j]-θ2[j])
 *         sum_s1 = Σ sin(θ1[j]), sum_c1 = Σ cos(θ1[j])
 */
extern "C" __global__ void pst_jacobian_sums_kernel(
    const float* d_final_state,
    float* d_sums,
    int n) {
    // d_final_state: [th1, th2, th3, r1, r2, r3] each n
    // d_sums: [12] — output
    
    __shared__ float s_sums[16]; // padded to avoid bank conflicts
    if (threadIdx.x < 12) s_sums[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ⚠ MUST NOT return before __syncthreads — all threads in block must participate
    if (i < n) {
    float th1 = d_final_state[0*n + i];
    float th2 = d_final_state[1*n + i];
    float th3 = d_final_state[2*n + i];
    float s1 = sinf(th1), c1 = cosf(th1);
    float s2 = sinf(th2), c2 = cosf(th2);
    float s3 = sinf(th3), c3 = cosf(th3);

    // Cross order: sin(θ1-θ2) = s1*c2 - c1*s2, cos(θ1-θ2) = c1*c2 + s1*s2
    atomicAdd(&s_sums[0],  s1*c2 - c1*s2);  // S12
    atomicAdd(&s_sums[1],  c1*c2 + s1*s2);  // C12
    atomicAdd(&s_sums[2],  s2*c3 - c2*s3);  // S23
    atomicAdd(&s_sums[3],  c2*c3 + s2*s3);  // C23
    atomicAdd(&s_sums[4],  s1*c3 - c1*s3);  // S13
    atomicAdd(&s_sums[5],  c1*c3 + s1*s3);  // C13
    // Mean sums
    atomicAdd(&s_sums[6],  s1);             // sum_s1
    atomicAdd(&s_sums[7],  c1);             // sum_c1
    atomicAdd(&s_sums[8],  s2);             // sum_s2
    atomicAdd(&s_sums[9],  c2);             // sum_c2
    atomicAdd(&s_sums[10], s3);             // sum_s3
    atomicAdd(&s_sums[11], c3);             // sum_c3
    } // if (i < n)

    __syncthreads();
    if (threadIdx.x < 12) d_sums[threadIdx.x] = s_sums[threadIdx.x];
} // pst_jacobian_sums_kernel

// ─── Kernel B: compute v = J^T · grad_pst ────────────────

/**
 * Compute v for ALL 6 state variable groups in one kernel.
 *
 * Grid: ceil(n/256) blocks × 256 threads
 * Each thread i computes v_th1[i], v_th2[i], v_th3[i], v_r1[i], v_r2[i], v_r3[i]
 * using d_sums (from Kernel A) for cross-order + mean contributions.
 *
 * Output packed: d_v [v_th1(n), v_th2(n), v_th3(n), v_r1(n), v_r2(n), v_r3(n)]
 */
extern "C" __global__ void pst_jacobian_compute_kernel(
    const float* d_final_state,
    const float* d_grad_pst,
    const float* d_sums,
    float* d_v,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // ── Load state for this oscillator ────────────────
    float th1 = d_final_state[0*n + i];
    float th2 = d_final_state[1*n + i];
    float th3 = d_final_state[2*n + i];
    float r1_val = d_final_state[3*n + i];
    float r2_val = d_final_state[4*n + i];
    float r3_val = d_final_state[5*n + i];

    float s1 = sinf(th1), c1 = cosf(th1);
    float s2 = sinf(th2), c2 = cosf(th2);
    float s3 = sinf(th3), c3 = cosf(th3);

    const float* g = d_grad_pst;
    const float* sums = d_sums;
    float S12 = sums[0], C12 = sums[1];
    float S23 = sums[2], C23 = sums[3];
    float S13 = sums[4], C13 = sums[5];
    float sum_s1 = sums[6], sum_c1 = sums[7];
    float sum_s2 = sums[8], sum_c2 = sums[9];
    float sum_s3 = sums[10], sum_c3 = sums[11];
    float nf = (float)n;
    float eps = 1e-12f;

    // ── v_th1[i] ──────────────────────────────────────
    float v1 = 0.0f;

    // Group 1: sin1[i], cos1[i] at 2*i, 2*i+1
    v1 += g[2*i]   * c1;        // d(sin1[i])/dθ1[i] = cos1[i]
    v1 += g[2*i+1] * (-s1);     // d(cos1[i])/dθ1[i] = -sin1[i]
    // Group 2: vel1[i] at 6n + i
    v1 += g[6*n + i];
    // Group 4: sin²1[i], cos²1[i] at 12n + 2*i, 12n + 2*i+1
    float dsc = 2.0f * s1 * c1;
    v1 += g[12*n + 2*i]   * dsc;
    v1 += g[12*n + 2*i+1] * (-dsc);

    // Group 5: R12 at 18n — dR12/dθ1[i]
    float norm12 = sqrtf(S12*S12 + C12*C12);
    if (norm12 > eps) {
        float inv_norm_n = (1.0f / norm12) / nf;
        // d(sum_s)/dθ1 = cos1·cos2 + sin1·sin2 = cos(θ1-θ2)
        float dss = c1 * c2 + s1 * s2;
        // d(sum_c)/dθ1 = -sin1·cos2 + cos1·sin2 = -sin(θ1-θ2)
        float dsc_deriv = -s1 * c2 + c1 * s2;
        v1 += g[18*n] * (S12 * dss + C12 * dsc_deriv) * inv_norm_n;
    }
    // Group 5: R13 at 18n+2 — dR13/dθ1[i]
    float norm13 = sqrtf(S13*S13 + C13*C13);
    if (norm13 > eps) {
        float inv_norm_n = (1.0f / norm13) / nf;
        float dss = c1 * c3 + s1 * s3;
        float dsc_deriv = -s1 * c3 + c1 * s3;
        v1 += g[18*n + 2] * (S13 * dss + C13 * dsc_deriv) * inv_norm_n;
    }

    // Group 6: mean_sin1 at 18n+3, mean_cos1 at 18n+4
    v1 += g[18*n + 3] * (c1 / nf);   // d(mean_sin1)/dθ1[i] = cos1/n
    v1 += g[18*n + 4] * (-s1 / nf);  // d(mean_cos1)/dθ1[i] = -sin1/n

    d_v[0*n + i] = v1;

    // ── v_th2[i] ──────────────────────────────────────
    float v2 = 0.0f;

    v2 += g[2*n + 2*i]   * c2;
    v2 += g[2*n + 2*i+1] * (-s2);
    v2 += g[7*n + i];

    dsc = 2.0f * s2 * c2;
    v2 += g[14*n + 2*i]   * dsc;
    v2 += g[14*n + 2*i+1] * (-dsc);

    // R12 at 18n — dR12/dθ2[i]
    if (norm12 > eps) {
        float inv_norm_n = (1.0f / norm12) / nf;
        // d(sum_s)/dθ2 = -(cos1·cos2 + sin1·sin2) = -cos(θ1-θ2)
        float dss = -(c1 * c2 + s1 * s2);
        // d(sum_c)/dθ2 = -cos1·sin2 + sin1·cos2 = sin(θ1-θ2)
        float dsc_deriv = -c1 * s2 + s1 * c2;
        v2 += g[18*n] * (S12 * dss + C12 * dsc_deriv) * inv_norm_n;
    }
    // R23 at 18n+1 — dR23/dθ2[i]
    float norm23 = sqrtf(S23*S23 + C23*C23);
    if (norm23 > eps) {
        float inv_norm_n = (1.0f / norm23) / nf;
        float dss = c2 * c3 + s2 * s3;
        float dsc_deriv = -s2 * c3 + c2 * s3;
        v2 += g[18*n + 1] * (S23 * dss + C23 * dsc_deriv) * inv_norm_n;
    }

    v2 += g[18*n + 6] * (c2 / nf);
    v2 += g[18*n + 7] * (-s2 / nf);

    d_v[1*n + i] = v2;

    // ── v_th3[i] ──────────────────────────────────────
    float v3 = 0.0f;

    v3 += g[4*n + 2*i]   * c3;
    v3 += g[4*n + 2*i+1] * (-s3);
    v3 += g[8*n + i];

    dsc = 2.0f * s3 * c3;
    v3 += g[16*n + 2*i]   * dsc;
    v3 += g[16*n + 2*i+1] * (-dsc);

    // R23 at 18n+1 — dR23/dθ3[i]
    if (norm23 > eps) {
        float inv_norm_n = (1.0f / norm23) / nf;
        float dss = -(c2 * c3 + s2 * s3);
        float dsc_deriv = -c2 * s3 + s2 * c3;
        v3 += g[18*n + 1] * (S23 * dss + C23 * dsc_deriv) * inv_norm_n;
    }
    // R13 at 18n+2 — dR13/dθ3[i]
    if (norm13 > eps) {
        float inv_norm_n = (1.0f / norm13) / nf;
        float dss = -(c1 * c3 + s1 * s3);
        float dsc_deriv = -c1 * s3 + s1 * c3;
        v3 += g[18*n + 2] * (S13 * dss + C13 * dsc_deriv) * inv_norm_n;
    }

    v3 += g[18*n + 9]  * (c3 / nf);
    v3 += g[18*n + 10] * (-s3 / nf);

    d_v[2*n + i] = v3;

    // ── v_r1[i] ───────────────────────────────────────
    float vr1 = 0.0f;
    vr1 += g[9*n + i];           // r1[i] feature
    vr1 += g[18*n + 5] / nf;     // mean_r1
    d_v[3*n + i] = vr1;

    // ── v_r2[i] ───────────────────────────────────────
    float vr2 = 0.0f;
    vr2 += g[10*n + i];
    vr2 += g[18*n + 8] / nf;
    d_v[4*n + i] = vr2;

    // ── v_r3[i] ───────────────────────────────────────
    float vr3 = 0.0f;
    vr3 += g[11*n + i];
    vr3 += g[18*n + 11] / nf;
    d_v[5*n + i] = vr3;
}

// ─── Host wrapper ────────────────────────────────────────

extern "C" int cuda_pst_jacobian(
    const float* d_final_state,
    const float* d_grad_pst,
    float* d_sums,
    float* d_v,
    int n) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;

    // Kernel A: sums
    pst_jacobian_sums_kernel<<<gs, bs>>>(d_final_state, d_sums, n);
    cudaError_t e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    // Kernel B: compute v (uses d_sums from kernel A)
    pst_jacobian_compute_kernel<<<gs, bs>>>(d_final_state, d_grad_pst, d_sums, d_v, n);
    e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    return 0;
}

// ─── ADD variant: d_v += J^T · d_grad_pst ────────────────

/**
 * Kernel B' — same as pst_jacobian_compute_kernel but ADDS to d_v.
 */
extern "C" __global__ void pst_jacobian_compute_add_kernel(
    const float* d_final_state,
    const float* d_grad_pst,
    const float* d_sums,
    float* d_v,
    int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float th1 = d_final_state[0*n + i];
    float th2 = d_final_state[1*n + i];
    float th3 = d_final_state[2*n + i];
    float r1_val = d_final_state[3*n + i];
    float r2_val = d_final_state[4*n + i];
    float r3_val = d_final_state[5*n + i];

    float s1 = sinf(th1), c1 = cosf(th1);
    float s2 = sinf(th2), c2 = cosf(th2);
    float s3 = sinf(th3), c3 = cosf(th3);

    const float* g = d_grad_pst;
    const float* sums = d_sums;
    float S12 = sums[0], C12 = sums[1];
    float S23 = sums[2], C23 = sums[3];
    float S13 = sums[4], C13 = sums[5];
    float sum_s1 = sums[6], sum_c1 = sums[7];
    float sum_s2 = sums[8], sum_c2 = sums[9];
    float sum_s3 = sums[10], sum_c3 = sums[11];
    float nf = (float)n;
    float eps = 1e-12f;

    float v1 = 0.0f;
    v1 += g[2*i]   * c1;
    v1 += g[2*i+1] * (-s1);
    v1 += g[6*n + i];
    float dsc = 2.0f * s1 * c1;
    v1 += g[12*n + 2*i]   * dsc;
    v1 += g[12*n + 2*i+1] * (-dsc);

    float norm12 = sqrtf(S12*S12 + C12*C12);
    if (norm12 > eps) {
        float inv_norm_n = (1.0f / norm12) / nf;
        float dss = c1 * c2 + s1 * s2;
        float dsc_deriv = -s1 * c2 + c1 * s2;
        v1 += g[18*n] * (S12 * dss + C12 * dsc_deriv) * inv_norm_n;
    }
    float norm13 = sqrtf(S13*S13 + C13*C13);
    if (norm13 > eps) {
        float inv_norm_n = (1.0f / norm13) / nf;
        float dss = c1 * c3 + s1 * s3;
        float dsc_deriv = -s1 * c3 + c1 * s3;
        v1 += g[18*n + 2] * (S13 * dss + C13 * dsc_deriv) * inv_norm_n;
    }
    v1 += g[18*n + 3] * (c1 / nf);
    v1 += g[18*n + 4] * (-s1 / nf);

    d_v[0*n + i] += v1;

    float v2 = 0.0f;
    v2 += g[2*n + 2*i]   * c2;
    v2 += g[2*n + 2*i+1] * (-s2);
    v2 += g[7*n + i];
    dsc = 2.0f * s2 * c2;
    v2 += g[14*n + 2*i]   * dsc;
    v2 += g[14*n + 2*i+1] * (-dsc);

    if (norm12 > eps) {
        float inv_norm_n = (1.0f / norm12) / nf;
        float dss = -(c1 * c2 + s1 * s2);
        float dsc_deriv = -c1 * s2 + s1 * c2;
        v2 += g[18*n] * (S12 * dss + C12 * dsc_deriv) * inv_norm_n;
    }
    float norm23 = sqrtf(S23*S23 + C23*C23);
    if (norm23 > eps) {
        float inv_norm_n = (1.0f / norm23) / nf;
        float dss = c2 * c3 + s2 * s3;
        float dsc_deriv = -s2 * c3 + c2 * s3;
        v2 += g[18*n + 1] * (S23 * dss + C23 * dsc_deriv) * inv_norm_n;
    }
    v2 += g[18*n + 6] * (c2 / nf);
    v2 += g[18*n + 7] * (-s2 / nf);

    d_v[1*n + i] += v2;

    float v3 = 0.0f;
    v3 += g[4*n + 2*i]   * c3;
    v3 += g[4*n + 2*i+1] * (-s3);
    v3 += g[8*n + i];
    dsc = 2.0f * s3 * c3;
    v3 += g[16*n + 2*i]   * dsc;
    v3 += g[16*n + 2*i+1] * (-dsc);

    if (norm23 > eps) {
        float inv_norm_n = (1.0f / norm23) / nf;
        float dss = -(c2 * c3 + s2 * s3);
        float dsc_deriv = -c2 * s3 + s2 * c3;
        v3 += g[18*n + 1] * (S23 * dss + C23 * dsc_deriv) * inv_norm_n;
    }
    if (norm13 > eps) {
        float inv_norm_n = (1.0f / norm13) / nf;
        float dss = -(c1 * c3 + s1 * s3);
        float dsc_deriv = -c1 * s3 + s1 * c3;
        v3 += g[18*n + 2] * (S13 * dss + C13 * dsc_deriv) * inv_norm_n;
    }
    v3 += g[18*n + 9]  * (c3 / nf);
    v3 += g[18*n + 10] * (-s3 / nf);

    d_v[2*n + i] += v3;

    float vr1 = g[9*n + i] + g[18*n + 5] / nf;
    d_v[3*n + i] += vr1;

    float vr2 = g[10*n + i] + g[18*n + 8] / nf;
    d_v[4*n + i] += vr2;

    float vr3 = g[11*n + i] + g[18*n + 11] / nf;
    d_v[5*n + i] += vr3;
}

/**
 * Host wrapper for K1' — runs Kernel A + B' (ADDS to d_v).
 */
extern "C" int cuda_pst_jacobian_add(
    const float* d_final_state,
    const float* d_grad_pst,
    float* d_sums,
    float* d_v,
    int n) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;

    pst_jacobian_sums_kernel<<<gs, bs>>>(d_final_state, d_sums, n);
    cudaError_t e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    pst_jacobian_compute_add_kernel<<<gs, bs>>>(d_final_state, d_grad_pst, d_sums, d_v, n);
    e = cudaGetLastError();
    if (e) return (int)e;
    cudaDeviceSynchronize();

    return 0;
}
