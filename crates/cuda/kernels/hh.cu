/**
 * Hodgkin-Huxley ion channels — direct computation on GPU.
 *
 * GPU has __expf() as a single-cycle SFU instruction. Direct computation
 * is faster than LUT (no global memory traffic, no quantization error).
 *
 * Channels (dispatch by channel index, not gate index):
 *   ch=0 NaV: m³h — E_rev=50mV, g_max=120 mS/cm² — spike generation
 *   ch=1 KV:  n⁴  — E_rev=-77mV, g_max=36 mS/cm² — repolarization
 *   ch=2 CaT: c¹  — E_rev=120mV, g_max=0.5 mS/cm² — T-type calcium, pacemaking
 *
 * Gate layout (SOA — fully coalesced):
 *   gates[gate_idx * n + neuron_idx]
 *   For standard HH: gates[0*n + neuron] = m, gates[1*n + neuron] = h,
 *                    gates[2*n + neuron] = n
 *   With CaV-T:      gates[3*n + neuron] = c
 */
#include <cuda_runtime.h>
#include <math.h>

// ═══════════════════════════════════════════════════════
//  vtrap — численно стабильная экспонента (NEURON hh.mod)
// ═══════════════════════════════════════════════════════

/// vtrap(x, y) = x / (exp(x/y) - 1) с Taylor expansion при |x/y| < 1e-6.
/// В точности как NEURON hh.mod `FUNCTION vtrap`.
/// Предотвращает 0/0 при x → 0.
__device__ inline float vtrap(float x, float y) {
    float ratio = x / y;
    if (fabsf(ratio) < 1e-6f) {
        return y - x * 0.5f; // Taylor: y - x/2
    }
    return x / (__expf(ratio) - 1.0f);
}

// ═══════════════════════════════════════════════════════
//  Rate functions — NaV (Hodgkin & Huxley 1952)
// ═══════════════════════════════════════════════════════

__device__ inline float na_m_alpha(float v) {
    return 0.1f * vtrap(-(v + 40.0f), 10.0f);
}
__device__ inline float na_m_beta(float v) {
    return 4.0f * __expf(-(v + 65.0f) / 18.0f);
}

__device__ inline float na_h_alpha(float v) {
    return 0.07f * __expf(-(v + 65.0f) / 20.0f);
}
__device__ inline float na_h_beta(float v) {
    return 1.0f / (1.0f + __expf(-(v + 35.0f) / 10.0f));
}

// ═══════════════════════════════════════════════════════
//  Rate functions — KV (Hodgkin & Huxley 1952)
// ═══════════════════════════════════════════════════════

__device__ inline float k_n_alpha(float v) {
    return 0.01f * vtrap(-(v + 55.0f), 10.0f);
}
__device__ inline float k_n_beta(float v) {
    return 0.125f * __expf(-(v + 65.0f) / 80.0f);
}

// ═══════════════════════════════════════════════════════
//  Rate functions — CaV-T (Wang & Rinzel 1993)
// ═══════════════════════════════════════════════════════
//
// T-type calcium: single activation gate (c), tau=0.6ms constant.
// Activates at low voltage (~-60mV) — provides depolarizing drive
// for pacemaking. E_rev=+120mV (Ca²⁺ reversal is very positive).

__device__ inline float cat_alpha(float v) {
    float m_inf = 1.0f / (1.0f + __expf(-(v + 56.0f) / 6.2f));
    return m_inf / 0.6f;
}
__device__ inline float cat_beta(float v) {
    float m_inf = 1.0f / (1.0f + __expf(-(v + 56.0f) / 6.2f));
    return (1.0f - m_inf) / 0.6f;
}

// ═══════════════════════════════════════════════════════
//  Per-channel rate function dispatchers
// ═══════════════════════════════════════════════════════
//
// Each channel type knows its own alpha/beta functions.
// Dispatch by channel index (ch), not global gate index (gi).
// This makes adding new channels trivial — add a new _rate() function.

__device__ inline void na_rate(float v, int gate_in_ch, float* alpha, float* beta) {
    if (gate_in_ch == 0) { *alpha = na_m_alpha(v); *beta = na_m_beta(v); }
    else                 { *alpha = na_h_alpha(v); *beta = na_h_beta(v); }
}

__device__ inline void k_rate(float v, int gate_in_ch, float* alpha, float* beta) {
    *alpha = k_n_alpha(v); *beta = k_n_beta(v);
}

__device__ inline void cat_rate(float v, int gate_in_ch, float* alpha, float* beta) {
    *alpha = cat_alpha(v); *beta = cat_beta(v);
}

// ═══════════════════════════════════════════════════════
//  Device kernel — step ALL neurons by dt
// ═══════════════════════════════════════════════════════

extern "C" __global__ void hh_step_kernel(
    float* v,                // [n] membrane potential (in/out)
    float* gates,            // [n * n_gates] row-major gate vars (in/out)
    const float* Ii,         // [n] input current
    int* spikes,             // [n] output spikes (1=spiked, 0=not)
    float* refractory,       // [n] refractory timer (in/out)
    const float* g_leak,     // [n] leak conductance
    const float* E_leak,     // [n] leak reversal potential
    const float* C_m,        // [n] membrane capacitance
    int n,                   // number of neurons
    int n_gates,             // total gates (3 for NaV+KV, 4 with CaV-T)
    float dt,                // timestep (ms)
    float tau_ref,           // refractory period (ms)
    const float* E_rev_arr,  // [n_channels] reversal per channel
    const float* g_max_arr,  // [n_channels] conductance per channel
    const unsigned int* powers,      // [n_gates] gate power (exponent)
    const int* n_gates_per_ch,       // [n_channels] gates per channel
    int n_channels,          // number of channels (2=NaV+KV, 3=+CaV-T)
    float celsius            // temperature (°C), Q10 = 3^((celsius-6.3)/10)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // ─── Temperature correction (Q10) ─
    float q10 = 1.0f;
    if (fabsf(celsius - 6.3f) > 0.1f) {
        q10 = powf(3.0f, (celsius - 6.3f) / 10.0f);
    }

    // ─── Refractory Timer Update (Do NOT skip ODE dynamics) ─
    float ref = refractory[idx];
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
    }

    float vi = v[idx];

    int gi = 0;  // global gate index
    float I_ion_sum = 0.0f; // Sum of g_k * E_k
    float g_ion_sum = 0.0f; // Sum of g_k

    for (int ch = 0; ch < n_channels; ch++) {
        int ng = n_gates_per_ch[ch];
        float e_rev = E_rev_arr[ch];
        float g_max = g_max_arr[ch];
        float g_ch = g_max;

        for (int gate_in_ch = 0; gate_in_ch < ng; gate_in_ch++) {
            // Dispatch rate functions by channel type
            float alpha, beta;
            if (ch == 0) {
                na_rate(vi, gate_in_ch, &alpha, &beta);
            } else if (ch == 1) {
                k_rate(vi, gate_in_ch, &alpha, &beta);
            } else {
                // ch >= 2: CaV-T (and future channels)
                cat_rate(vi, gate_in_ch, &alpha, &beta);
            }

            float sum_q10 = (alpha + beta + 1e-12f) * q10;
            float m_inf = alpha / (alpha + beta + 1e-12f);  // Q10 не влияет на steady-state
            float decay = __expf(-dt * sum_q10);

            // SOA gate access: gates[gate_idx * n + neuron_idx]
            float* g_ptr = &gates[gi * n];

            // Rush-Larsen gate update: g(t+dt) = m_inf + (g(t) - m_inf) * exp(-dt * sum_q10)
            float g_val = m_inf + (g_ptr[idx] - m_inf) * decay;
            g_ptr[idx] = g_val;

            // Apply gate power: g_ch *= gate^power
            unsigned int p = powers[gi];
            #pragma unroll
            for (unsigned int k = 0; k < p; k++) {
                g_ch *= g_val;
            }

            gi++;
        }

        g_ion_sum += g_ch;
        I_ion_sum += g_ch * e_rev;
    }

    // ─── Exponential Euler Membrane Voltage Integration (Unconditionally Stable) ─
    float g_l = g_leak[idx];
    float e_l = E_leak[idx];
    float cm = C_m[idx];
    float Ii_val = Ii[idx];

    float g_tot = g_ion_sum + g_l + 1e-12f;
    float v_inf = (Ii_val + I_ion_sum + g_l * e_l) / g_tot;
    float tau_v = cm / g_tot;
    float decay_v = __expf(-dt / tau_v);

    float v_prev = vi;
    vi = v_inf + (v_prev - v_inf) * decay_v;

    // ─── Standard Rising-Edge Spike Threshold Detection ───
    int spiked = 0;
    const float V_THRESH = 0.0f; // 0 mV threshold

    if (ref <= 0.0f && v_prev < V_THRESH && vi >= V_THRESH) {
        spiked = 1;
        refractory[idx] = tau_ref;
    }

    v[idx] = vi;
    spikes[idx] = spiked;
}

// ═══════════════════════════════════════════════════════
//  Host wrapper (sync — one-shot call, no context)
// ═══════════════════════════════════════════════════════

extern "C" int cuda_hh_step(
    float* v, float* gates,
    const float* Ii, int* spikes, float* refractory,
    const float* g_leak, const float* E_leak, const float* C_m,
    int n, int n_gates, float dt, float tau_ref,
    const float* E_rev_arr, const float* g_max_arr,
    const unsigned int* powers, const int* n_gates_per_ch,
    int n_channels
) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    hh_step_kernel<<<gs, bs>>>(
        v, gates, Ii, spikes, refractory,
        g_leak, E_leak, C_m,
        n, n_gates, dt, tau_ref,
        E_rev_arr, g_max_arr, powers, n_gates_per_ch, n_channels);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════
//  Async stream-based launch (for GpuContext)
// ═══════════════════════════════════════════════════════

#include "gpu_context.h"

extern "C" int gpu_hh_step_async(void* vctx, float dt, float tau_ref) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_spikes, ctx->n_neurons*sizeof(int));
    if (!ctx->d_refractory) {
        cudaError_t _e = cudaMalloc(&ctx->d_refractory, ctx->n_neurons*sizeof(float));
        if (_e) return (int)_e;
        cudaMemset(ctx->d_refractory, 0, ctx->n_neurons*sizeof(float));
    }
    int bs = 256;
    int n = ctx->n_neurons;
    int n_gates = ctx->n_hh_gates;
    int n_channels = ctx->n_hh_channels;

    hh_step_kernel<<<gs(n, bs), bs, 0, ctx->stream>>>(
        ctx->d_v,
        ctx->d_hh_gates,
        ctx->d_I,
        ctx->d_spikes,
        ctx->d_refractory,
        ctx->d_hh_g_leak,
        ctx->d_hh_E_leak,
        ctx->d_hh_C_m,
        n, n_gates, dt, tau_ref,
        ctx->d_hh_E_rev_arr,
        ctx->d_hh_g_max_arr,
        ctx->d_hh_powers,
        ctx->d_hh_n_gates_per_ch,
        n_channels);
    return (int)cudaGetLastError();
}
