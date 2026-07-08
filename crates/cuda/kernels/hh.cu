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
 * Gate layout (row-major, coalesced):
 *   gates[neuron * n_gates + gate_idx]
 *   For standard HH: gates[neuron * 3 + 0] = m, [1] = h, [2] = n
 *   With CaV-T:      gates[neuron * 4 + 0] = m, [1] = h, [2] = n, [3] = c
 */
#include <cuda_runtime.h>
#include <math.h>

// ═══════════════════════════════════════════════════════
//  Rate functions — NaV (Hodgkin & Huxley 1952)
// ═══════════════════════════════════════════════════════

__device__ inline float na_m_alpha(float v) {
    float v_ = v + 40.0f;
    if (fabsf(v_) < 1e-6f) return 1.0f;
    return 0.1f * v_ / (1.0f - __expf(-v_ / 10.0f));
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
    float v_ = v + 55.0f;
    if (fabsf(v_) < 1e-6f) return 0.1f;
    return 0.01f * v_ / (1.0f - __expf(-v_ / 10.0f));
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
    int n_channels           // number of channels (2=NaV+KV, 3=+CaV-T)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // ─── Refractory: skip all dynamics ─────────────────
    float ref = refractory[idx];
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        spikes[idx] = 0;
        return;
    }

    float vi = v[idx];
    float* g = &gates[idx * n_gates];

    int gi = 0;  // global gate index
    float I_ion = 0.0f;

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

            float sum = alpha + beta + 1e-12f;
            float m_inf = alpha / sum;
            float tau = 1.0f / sum;
            float decay = __expf(-dt / tau);

            // Gate update: g(t+dt) = m_inf + (g(t) - m_inf) * exp(-dt/tau)
            g[gi] = m_inf + (g[gi] - m_inf) * decay;

            // Apply gate power: g_ch *= gate^power
            unsigned int p = powers[gi];
            float g_factor = g[gi];
            #pragma unroll
            for (unsigned int k = 0; k < p; k++) {
                g_ch *= g_factor;
            }

            gi++;
        }

        I_ion += g_ch * (vi - e_rev);
    }

    // Leak current
    float I_leak = g_leak[idx] * (vi - E_leak[idx]);
    float dV = (Ii[idx] - I_ion - I_leak) / C_m[idx];
    vi += dV * dt;
    vi = fminf(50.0f, fmaxf(-100.0f, vi));

    // ─── Spike detection (backward zero-crossing) ───
    // Match CPU HHLUT: detect when v crosses 0 mV going DOWN.
    // This captures the spike after its peak, allowing the natural
    // AHP to develop (v falls to ~-75 mV, KV deactivation takes ~15ms).
    // NO artificial v reset — let HH dynamics play out.
    int spiked = 0;
    float v_prev = v[idx];  // v from previous step (before update)
    if (v_prev >= 0.0f && vi < 0.0f) {
        spiked = 1;
    }
    if (spiked) {
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
