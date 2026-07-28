/**
 * CUDA kernels for biophysical models:
 * - Morris-Lecar (1981) neuron model
 * - NMDA Mg²⁺-Block Synaptic Channel (Destexhe 1998)
 * - Tsodyks-Markram Short-Term Plasticity (1997/1998)
 */
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void morris_lecar_step_kernel(
    float* v, float* w, float* I_in, int* spikes,
    const float* params, float dt, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float vv = v[idx];
    float ww = w[idx];
    float ii = I_in[idx];

    const float* p = params + idx * 16;
    float C_val    = p[0];
    float g_L      = p[1];
    float E_L      = p[2];
    float g_Ca     = p[3];
    float E_Ca     = p[4];
    float g_K      = p[5];
    float E_K      = p[6];
    float V1       = p[7];
    float V2       = p[8];
    float V3       = p[9];
    float V4       = p[10];
    float phi      = p[11];
    float v_thresh = p[12];

    float m_inf = 0.5f * (1.0f + tanhf((vv - V1) / V2));
    float w_inf = 0.5f * (1.0f + tanhf((vv - V3) / V4));
    float tau_w = 1.0f / (phi * coshf((vv - V3) / (2.0f * V4)));

    float I_leak = g_L * (vv - E_L);
    float I_Ca   = g_Ca * m_inf * (vv - E_Ca);
    float I_K    = g_K * ww * (vv - E_K);

    float dv = (ii - I_leak - I_Ca - I_K) / C_val;
    float dw = (w_inf - ww) / tau_w;

    float new_v = vv + dv * dt;
    float new_w = ww + dw * dt;

    int spiked = (new_v >= v_thresh) ? 1 : 0;

    v[idx] = new_v;
    w[idx] = new_w;
    spikes[idx] = spiked;
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

    xx += ((1.0f - xx) / tau_d) * dt;
    uu += ((U_val - uu) / tau_f) * dt;

    if (spk != 0) {
        uu += U_val * (1.0f - uu);
        xx -= uu * xx;
    }

    x[idx] = xx;
    u[idx] = uu;
    w_eff[idx] = W_base * uu * xx;
}
