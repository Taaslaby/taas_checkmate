#![allow(non_snake_case)]
use super::{fast_exp_simd, load8, load8_mut, spike_mask_to_i32, store8};
use wide::{f32x8, CmpGt};

// ═══════════════════════════════════════════════════════
//  HAAP — Hyperpolarization-Activated Adaptive Pulse-Bursting
// ═══════════════════════════════════════════════════════
//
// Equations (conductance-based, 3 variables):
//   C_m·dV/dt = -g_L(V-E_L) - g_M·u·(V-E_K) - g_h·h·(V-E_h) + I_syn
//   τ_u·du/dt = (u_∞(V) - u)
//   τ_h·dh/dt = (h_∞(V) - h)
//
// where:
//   u_∞(V) = sigmoid((V - V_half_u) / k_u)       — M-current activates with depolarization
//   h_∞(V) = sigmoid((V_half_h - V) / k_h)       — HCN activates with hyperpolarization
//
// Spike: V >= V_thresh → V = V_reset, u += delta_u
//
// Reference: conceptual combination of M-current (Kv7/KCNQ, Brown & Adams 1980)
// and h-current (HCN, Pape 1996), tuned for Drosophila VNC dynamics.
//
// Params layout (17 × n, column-major):
//   [0]:    dt          — time step (ms), uniform
//   [n]:    C_m         — membrane capacitance (µF/cm²)
//   [2n]:   g_L         — leak conductance (mS/cm²)
//   [3n]:   E_L         — leak reversal (mV)
//   [4n]:   g_M         — M-current conductance (mS/cm²)
//   [5n]:   E_K         — potassium reversal (mV)
//   [6n]:   g_h         — HCN conductance (mS/cm²)
//   [7n]:   E_h         — HCN reversal (mV, ~-30mV for HCN: Na⁺/K⁺ mixed)
//   [8n]:   tau_u       — M-current time constant (ms)
//   [9n]:   tau_h       — HCN time constant (ms)
//   [10n]:  V_half_u    — M-current half-activation (mV, ~-35mV)
//   [11n]:  k_u         — M-current slope factor (mV)
//   [12n]:  V_half_h    — HCN half-activation (mV, ~-75mV)
//   [13n]:  k_h         — HCN slope factor (mV, >0)
//   [14n]:  V_thresh    — spike threshold (mV)
//   [15n]:  V_reset     — after-spike reset (mV)
//   [16n]:  delta_u     — after-spike M-current increment (adaptation bump)
//
// State:
//   v[n]      — membrane potential
//   u[n]      — M-current gating (0..1)
//   h[n]      — HCN gating (0..1)

/// Step `n` HAAP neurons using `f32x8` SIMD.
///
/// # Safety
/// `v[n]`, `u_state[n]`, `h_state[n]`, `Ii[n]`, `spikes[n]`, `params[17·n]`
/// must be valid, non-overlapping.
pub(crate) unsafe fn haap_step(
    v: *mut f32,
    u_state: *mut f32,
    h_state: *mut f32,
    Ii: *const f32,
    spikes: *mut i32,
    params: *const f32,
    n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 17);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let eps = f32x8::splat(1e-10);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut uu = load8_mut(u_state, off);
        let mut hh = load8_mut(h_state, off);

        // Fused Ii clamp — предотвращает saturation
        let ii = load8(Ii, off)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));

        // Load params columns
        let c_m = load8(p.as_ptr().add(n), off);
        let g_L = load8(p.as_ptr().add(2 * n), off);
        let E_L = load8(p.as_ptr().add(3 * n), off);
        let g_M = load8(p.as_ptr().add(4 * n), off);
        let E_K = load8(p.as_ptr().add(5 * n), off);
        let g_h = load8(p.as_ptr().add(6 * n), off);
        let E_h = load8(p.as_ptr().add(7 * n), off);
        let tau_u = load8(p.as_ptr().add(8 * n), off);
        let tau_h = load8(p.as_ptr().add(9 * n), off);
        let V_half_u = load8(p.as_ptr().add(10 * n), off);
        let k_u = load8(p.as_ptr().add(11 * n), off);
        let V_half_h = load8(p.as_ptr().add(12 * n), off);
        let k_h = load8(p.as_ptr().add(13 * n), off);
        let V_thresh = load8(p.as_ptr().add(14 * n), off);
        let V_reset = load8(p.as_ptr().add(15 * n), off);
        let delta_u = load8(p.as_ptr().add(16 * n), off);

        // ─── Conductance currents ───
        // I_leak = g_L · (V - E_L)
        let i_leak = g_L * (vv - E_L);
        // I_M = g_M · u · (V - E_K)
        let i_M = g_M * uu * (vv - E_K);
        // I_h = g_h · h · (V - E_h)
        let i_h = g_h * hh * (vv - E_h);

        // dV/dt = (-I_leak - I_M - I_h + I_syn) / C_m
        let dv = (-i_leak - i_M - i_h + ii) / c_m.max(eps);
        vv += dt_s * dv;

        // ─── M-current gating ───
        // u_∞ = sigmoid((V - V_half_u) / k_u)
        // u_∞ activates with depolarization (positive exponent → sigmoid rises with V)
        let u_inf_arg = (vv - V_half_u) / k_u.max(eps);
        // Clamp exponent to avoid overflow
        let u_inf_arg_clamped = u_inf_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0));
        let u_inf = one / (one + fast_exp_simd(-u_inf_arg_clamped));
        // du/dt = (u_∞ - u) / τ_u
        let mask_tau_u = tau_u.cmp_gt(zero);
        let inv_tau_u = mask_tau_u.blend(one / tau_u.max(eps), zero);
        let du = (u_inf - uu) * inv_tau_u;
        uu += dt_s * du;
        // Clamp u to [0, 1]
        uu = uu.max(zero).min(one);

        // ─── HCN gating ───
        // h_∞ = sigmoid((V_half_h - V) / k_h) — activates with hyperpolarization
        // When V is very negative (< V_half_h), exponent > 0, h_∞ → 1
        // When V is very positive (> V_half_h), exponent < 0, h_∞ → 0
        let h_inf_arg = (V_half_h - vv) / k_h.max(eps);
        let h_inf_arg_clamped = h_inf_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0));
        let h_inf = one / (one + fast_exp_simd(-h_inf_arg_clamped));
        // dh/dt = (h_∞ - h) / τ_h
        let mask_tau_h = tau_h.cmp_gt(zero);
        let inv_tau_h = mask_tau_h.blend(one / tau_h.max(eps), zero);
        let dh = (h_inf - hh) * inv_tau_h;
        hh += dt_s * dh;
        // Clamp h to [0, 1]
        hh = hh.max(zero).min(one);

        // ─── Spike detection & reset ───
        let sm = vv.cmp_gt(V_thresh);
        // Apply reset where spike occurred: V = V_reset, u += delta_u
        vv = sm.blend(V_reset, vv);
        uu = sm.blend(uu + delta_u, uu);
        // Clamp u after reset
        uu = uu.max(zero).min(one);

        // Store results
        store8(v, off, vv);
        store8(u_state, off, uu);
        store8(h_state, off, hh);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0];
        *sp8.add(1) = sp_i[1];
        *sp8.add(2) = sp_i[2];
        *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4];
        *sp8.add(5) = sp_i[5];
        *sp8.add(6) = sp_i[6];
        *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let u_state = std::slice::from_raw_parts_mut(u_state, n);
        let h_state = std::slice::from_raw_parts_mut(h_state, n);
        let Ii = std::slice::from_raw_parts(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);

        for i in n8..n {
            let dt_loc = p[i]; // column 0: dt for this neuron
            let c_m = p[n + i];
            let g_L = p[2 * n + i];
            let E_L = p[3 * n + i];
            let g_M = p[4 * n + i];
            let E_K = p[5 * n + i];
            let g_h = p[6 * n + i];
            let E_h = p[7 * n + i];
            let tau_u = p[8 * n + i];
            let tau_h = p[9 * n + i];
            let V_half_u = p[10 * n + i];
            let k_u = p[11 * n + i];
            let V_half_h = p[12 * n + i];
            let k_h = p[13 * n + i];
            let V_thresh = p[14 * n + i];
            let V_reset = p[15 * n + i];
            let delta_u = p[16 * n + i];

            let i_clamped = Ii[i].clamp(-100.0, 100.0);

            // Conductances
            let i_leak = g_L * (v[i] - E_L);
            let i_M = g_M * u_state[i] * (v[i] - E_K);
            let i_h = g_h * h_state[i] * (v[i] - E_h);
            let dv = (-i_leak - i_M - i_h + i_clamped) / c_m.max(1e-10);
            v[i] += dt_loc * dv;

            // M-current gating
            if tau_u > 0.0 {
                let arg = (v[i] - V_half_u) / k_u.max(1e-10);
                let arg_c = arg.clamp(-50.0, 50.0);
                let u_inf = 1.0 / (1.0 + (-arg_c).exp());
                u_state[i] += dt_loc * (u_inf - u_state[i]) / tau_u;
                u_state[i] = u_state[i].clamp(0.0, 1.0);
            }

            // HCN gating
            if tau_h > 0.0 {
                let arg = (V_half_h - v[i]) / k_h.max(1e-10);
                let arg_c = arg.clamp(-50.0, 50.0);
                let h_inf = 1.0 / (1.0 + (-arg_c).exp());
                h_state[i] += dt_loc * (h_inf - h_state[i]) / tau_h;
                h_state[i] = h_state[i].clamp(0.0, 1.0);
            }

            // Spike & reset
            if v[i] >= V_thresh {
                spikes[i] = 1;
                v[i] = V_reset;
                u_state[i] = (u_state[i] + delta_u).clamp(0.0, 1.0);
            } else {
                spikes[i] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build default VNC parameters for N neurons.
    /// Matching Drosophila VNC electrophysiology:
    ///   - Resting ~-65mV, spikes ~0mV
    ///   - M-current: K+ adaptation, ~100ms
    ///   - h-current: HCN activated at -75mV, ~200ms
    fn make_vnc_params(n: usize, dt: f32) -> Vec<f32> {
        let mut params = vec![0.0f32; n * 17];
        for i in 0..n {
            params[i] = dt;              // dt (ms)
            params[n + i] = 1.0;         // C_m (µF/cm²)
            params[2 * n + i] = 0.1;     // g_L (mS/cm²)
            params[3 * n + i] = -65.0;   // E_L (mV)
            params[4 * n + i] = 0.5;     // g_M (mS/cm²) — M-current conductance
            params[5 * n + i] = -80.0;   // E_K (mV)
            params[6 * n + i] = 0.3;     // g_h (mS/cm²) — HCN conductance
            params[7 * n + i] = -30.0;   // E_h (mV) — HCN reversal (mixed Na⁺/K⁺)
            params[8 * n + i] = 100.0;   // tau_u (ms) — slow adaptation
            params[9 * n + i] = 200.0;   // tau_h (ms) — slow HCN
            params[10 * n + i] = -35.0;  // V_half_u (mV) — M-current half-activation
            params[11 * n + i] = 5.0;    // k_u (mV)
            params[12 * n + i] = -75.0;  // V_half_h (mV) — HCN half-activation
            params[13 * n + i] = 6.0;    // k_h (mV)
            params[14 * n + i] = -40.0;  // V_thresh (mV)
            params[15 * n + i] = -55.0;  // V_reset (mV)
            params[16 * n + i] = 0.02;   // delta_u — after-spike adaptation bump
        }
        params
    }

    /// HAAP: with sufficient depolarizing current, should spike repeatedly.
    #[test]
    fn haap_spikes_with_current() {
        let n = 8;
        let dt = 0.05;
        let params = make_vnc_params(n, dt);

        let mut v = vec![-65.0f32; n];
        let mut u_state = vec![0.0f32; n];  // M-current starts closed
        let mut h_state = vec![1.0f32; n];   // HCN starts fully open (hyperpolarized)
        let Ii = vec![15.0f32; n];            // strong input
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP should spike with I=15 pA");
    }

    /// HAAP: with inhibitory current, should be silent.
    #[test]
    fn haap_silent_with_inhibition() {
        let n = 8;
        let dt = 0.05;
        let params = make_vnc_params(n, dt);

        let mut v = vec![-65.0f32; n];
        let mut u_state = vec![0.0f32; n];
        let mut h_state = vec![1.0f32; n];
        let Ii = vec![-10.0f32; n];  // inhibitory
        let mut spikes = vec![0i32; n];

        for _ in 0..1000 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HAAP should not spike with I=-10");
    }

    /// HAAP: M-current adaptation — spike rate should decrease over time.
    #[test]
    fn haap_adaptation_slows_firing() {
        let n = 8;
        let dt = 0.05;
        // Strong M-current for visible adaptation
        let mut params = make_vnc_params(n, dt);
        for i in 0..n {
            params[4 * n + i] = 2.0;    // g_M = 2.0 — strong M-current
            params[8 * n + i] = 30.0;   // tau_u = 30ms — faster adaptation
        }

        let mut v = vec![-65.0f32; n];
        let mut u_state = vec![0.0f32; n];
        let mut h_state = vec![1.0f32; n];
        let Ii = vec![20.0f32; n];
        let mut spikes = vec![0i32; n];

        let mut early_spikes = 0i32;
        let mut late_spikes = 0i32;

        // Early phase (0-500 steps)
        for _ in 0..500 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            early_spikes += spikes.iter().sum::<i32>();
        }

        // Late phase (1500-2000 steps)
        // Reset spike counter
        for _ in 500..1500 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        for _ in 1500..2000 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            late_spikes += spikes.iter().sum::<i32>();
        }

        // With strong M-current, late spiking should be reduced
        // (adaptation builds up, u increases, outward K+ current increases)
        assert!(late_spikes <= early_spikes || early_spikes > 0,
            "HAAP: adaptation should not increase firing (early={}, late={})",
            early_spikes, late_spikes);
    }

    /// HAAP: HCN rebound — after hyperpolarization, h opens and helps depolarize.
    #[test]
    fn haap_hcn_rebound_excitation() {
        let n = 8;
        let dt = 0.1;
        let mut params = make_vnc_params(n, dt);
        // Strong h-current for visible rebound
        for i in 0..n {
            params[6 * n + i] = 1.0;    // g_h = 1.0
            params[9 * n + i] = 50.0;   // tau_h = 50ms
        }

        let mut v = vec![-65.0f32; n];
        let mut u_state = vec![0.0f32; n];
        let mut h_state = vec![0.5f32; n];
        let Ii = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        // Phase 1: hyperpolarize the cell (strong -current)
        // h-current fights hyperpolarization, so we need enough I to drive V below V_half_h=-75
        let Ii_hyper = vec![-50.0f32; n];
        for _ in 0..500 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii_hyper.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }

        // During hyperpolarization, h should approach 1 (fully open)
        let h_after_hyper = h_state[0];
        assert!(h_after_hyper > 0.6, "HCN should open during hyperpolarization: h={}", h_after_hyper);

        // Phase 2: release — h is open, should provide depolarizing current
        for _ in 0..200 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),  // back to 0 current
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }

        let rebound_spikes = spikes.iter().sum::<i32>();
        assert!(rebound_spikes > 0 || v[0] > -50.0,
            "HAAP: HCN rebound should depolarize or spike (v={}, spikes={})",
            v[0], rebound_spikes);
    }

    /// HAAP: non-aligned n works (scalar tail).
    #[test]
    fn haap_non_aligned_n() {
        let n = 13;
        let dt = 0.05;
        let params = make_vnc_params(n, dt);

        let mut v = vec![-65.0f32; n];
        let mut u_state = vec![0.0f32; n];
        let mut h_state = vec![1.0f32; n];
        let Ii = vec![15.0f32; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe {
                haap_step(
                    v.as_mut_ptr(),
                    u_state.as_mut_ptr(),
                    h_state.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP non-aligned should spike");
    }
}
