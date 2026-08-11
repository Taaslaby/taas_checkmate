/**
 * CUDA kernels for biophysical models:
 * - Morris-Lecar (1981) neuron model
 * - NMDA Mg²⁺-Block Synaptic Channel (Destexhe 1998)
 * - Tsodyks-Markram Short-Term Plasticity (1997/1998)
 */
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void morris_lecar_step_kernel(
    float* v, float* w, float* was_below, const float* I_in, int* spikes,
    const float* params, float dt, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float vv = v[idx];
    float ww = w[idx];
    float ii = I_in[idx];

    // Canonical pillars layout is SoA: each parameter occupies one n-wide
    // plane, matching NeuralPopulation::param_slice().
    float inv_C    = params[1 * n + idx];
    float g_L      = params[2 * n + idx];
    float E_L      = params[3 * n + idx];
    float g_Ca     = params[4 * n + idx];
    float E_Ca     = params[5 * n + idx];
    float g_K      = params[6 * n + idx];
    float E_K      = params[7 * n + idx];
    float V1       = params[8 * n + idx];
    float V2       = params[9 * n + idx];
    float V3       = params[10 * n + idx];
    float V4       = params[11 * n + idx];
    float phi      = params[12 * n + idx];
    float v_thresh = params[13 * n + idx];

    float m_inf = 1.0f / (1.0f + expf(-2.0f * (vv - V1) / V2));
    float w_inf = 1.0f / (1.0f + expf(-2.0f * (vv - V3) / V4));
    float cosh_arg = (vv - V3) / (2.0f * V4);
    float cosh_x = 0.5f * (expf(cosh_arg) + expf(-cosh_arg));

    float I_leak = g_L * (vv - E_L);
    float I_Ca   = g_Ca * m_inf * (vv - E_Ca);
    float I_K    = g_K * ww * (vv - E_K);

    float new_v = vv + (ii - I_leak - I_Ca - I_K) * inv_C * dt;
    float new_w = ww + phi * (w_inf - ww) * cosh_x * dt;

    // Match the zoo model: emit only a rising edge and retain the
    // `was_below` state; no artificial voltage reset.
    int rising = (new_v >= v_thresh && was_below[idx] > 0.0f) ? 1 : 0;
    v[idx] = new_v;
    w[idx] = new_w;
    spikes[idx] = rising;
    was_below[idx] = (new_v >= v_thresh) ? 0.0f : 1.0f;
}

extern "C" __global__ void nmda_mg_block_step_kernel(
    const float* v, const float* g_act, float* i_out,
    const float* params, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float vv = v[idx];
    float g_nmda = g_act[idx];

    const float* p = params + idx * 4;
    float E_syn   = p[1];
    float mg_conc = p[2];

    float block = 1.0f / (1.0f + (mg_conc / 3.57f) * expf(-0.062f * vv));
    float i_nmda = g_nmda * block * (vv - E_syn);

    i_out[idx] += i_nmda;
}

extern "C" __global__ void tsodyks_markram_step_kernel(
    float* x, float* u, const int* spikes, float* w_eff,
    const float* params, float dt, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float xx = x[idx];
    float uu = u[idx];
    int spk = spikes[idx];

    const float* p = params + idx * 4;
    float U_val  = p[0];
    float tau_d  = p[1];
    float tau_f  = p[2];
    float W_base = p[3];

    float tau_d_safe = fmaxf(tau_d, 1e-12f);
    float tau_f_safe = fmaxf(tau_f, 1e-12f);
    xx += (1.0f - xx) * (1.0f - expf(-dt / tau_d_safe));
    uu += (U_val - uu) * (1.0f - expf(-dt / tau_f_safe));

    if (spk != 0) {
        uu += U_val * (1.0f - uu);
        xx -= uu * xx;
    }

    x[idx] = xx;
    u[idx] = uu;
    w_eff[idx] = W_base * uu * xx;
}

// ─── Host wrappers ───────────────────────────────────────
// The Rust API accepts host slices. Keep the device kernels pointer-pure and
// make the ownership boundary explicit here: upload inputs, launch, synchronize,
// download outputs, and release temporary buffers on every call.

extern "C" int cuda_morris_lecar_step(
    float* v, float* w, float* was_below, const float* I_in, int* spikes,
    const float* params, float dt, int n
) {
    if (n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;

    const size_t nf = (size_t)n * sizeof(float);
    const size_t ni = (size_t)n * sizeof(int);
    const size_t np = (size_t)n * 14u * sizeof(float);
    float *d_v = nullptr, *d_w = nullptr, *d_below = nullptr, *d_I = nullptr, *d_params = nullptr;
    int *d_spikes = nullptr;
    cudaError_t e = cudaMalloc(&d_v, nf);
    if (e) return (int)e;
    if ((e = cudaMalloc(&d_w, nf)) || (e = cudaMalloc(&d_below, nf)) ||
        (e = cudaMalloc(&d_I, nf)) || (e = cudaMalloc(&d_params, np)) ||
        (e = cudaMalloc(&d_spikes, ni))) {
        cudaFree(d_v); cudaFree(d_w); cudaFree(d_below); cudaFree(d_I); cudaFree(d_params); cudaFree(d_spikes);
        return (int)e;
    }
    if ((e = cudaMemcpy(d_v, v, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_w, w, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_below, was_below, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_I, I_in, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_params, params, np, cudaMemcpyHostToDevice))) {
        cudaFree(d_v); cudaFree(d_w); cudaFree(d_below); cudaFree(d_I); cudaFree(d_params); cudaFree(d_spikes);
        return (int)e;
    }

    const int bs = 256;
    morris_lecar_step_kernel<<<(n + bs - 1) / bs, bs>>>(
        d_v, d_w, d_below, d_I, d_spikes, d_params, dt, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(v, d_v, nf, cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(w, d_w, nf, cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(was_below, d_below, nf, cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(spikes, d_spikes, ni, cudaMemcpyDeviceToHost);
    cudaFree(d_v); cudaFree(d_w); cudaFree(d_below); cudaFree(d_I); cudaFree(d_params); cudaFree(d_spikes);
    return (int)e;
}

extern "C" int cuda_nmda_mg_block_step(
    const float* v, const float* g_act, float* i_out,
    const float* params, int n
) {
    if (n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;

    const size_t nf = (size_t)n * sizeof(float);
    const size_t np = (size_t)n * 4u * sizeof(float);
    float *d_v = nullptr, *d_g = nullptr, *d_i = nullptr, *d_params = nullptr;
    cudaError_t e = cudaMalloc(&d_v, nf);
    if (e) return (int)e;
    if ((e = cudaMalloc(&d_g, nf)) || (e = cudaMalloc(&d_i, nf)) ||
        (e = cudaMalloc(&d_params, np))) {
        cudaFree(d_v); cudaFree(d_g); cudaFree(d_i); cudaFree(d_params);
        return (int)e;
    }
    if ((e = cudaMemcpy(d_v, v, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_g, g_act, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_i, i_out, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_params, params, np, cudaMemcpyHostToDevice))) {
        cudaFree(d_v); cudaFree(d_g); cudaFree(d_i); cudaFree(d_params);
        return (int)e;
    }

    const int bs = 256;
    nmda_mg_block_step_kernel<<<(n + bs - 1) / bs, bs>>>(
        d_v, d_g, d_i, d_params, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(i_out, d_i, nf, cudaMemcpyDeviceToHost);
    cudaFree(d_v); cudaFree(d_g); cudaFree(d_i); cudaFree(d_params);
    return (int)e;
}

extern "C" int cuda_tsodyks_markram_step(
    float* x, float* u, const int* spikes, float* w_eff,
    const float* params, float dt, int n
) {
    if (n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;

    const size_t nf = (size_t)n * sizeof(float);
    const size_t ni = (size_t)n * sizeof(int);
    const size_t np = (size_t)n * 4u * sizeof(float);
    float *d_x = nullptr, *d_u = nullptr, *d_w = nullptr, *d_params = nullptr;
    int *d_spikes = nullptr;
    cudaError_t e = cudaMalloc(&d_x, nf);
    if (e) return (int)e;
    if ((e = cudaMalloc(&d_u, nf)) || (e = cudaMalloc(&d_spikes, ni)) ||
        (e = cudaMalloc(&d_w, nf)) || (e = cudaMalloc(&d_params, np))) {
        cudaFree(d_x); cudaFree(d_u); cudaFree(d_spikes); cudaFree(d_w); cudaFree(d_params);
        return (int)e;
    }
    if ((e = cudaMemcpy(d_x, x, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_u, u, nf, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_spikes, spikes, ni, cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_params, params, np, cudaMemcpyHostToDevice))) {
        cudaFree(d_x); cudaFree(d_u); cudaFree(d_spikes); cudaFree(d_w); cudaFree(d_params);
        return (int)e;
    }

    const int bs = 256;
    tsodyks_markram_step_kernel<<<(n + bs - 1) / bs, bs>>>(
        d_x, d_u, d_spikes, d_w, d_params, dt, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(x, d_x, nf, cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(u, d_u, nf, cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(w_eff, d_w, nf, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_u); cudaFree(d_spikes); cudaFree(d_w); cudaFree(d_params);
    return (int)e;
}
