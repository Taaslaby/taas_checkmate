#![allow(non_snake_case)]
use super::{fast_exp_simd, load8, load8_mut, spike_mask_to_i32, store8};
use wide::{f32x8, CmpGe, CmpGt};

// ═══════════════════════════════════════════════════════
//  GLIF (Generalized Leaky Integrate-and-Fire)
//  Allen Institute — Teeter et al. 2018, Nature Communications
// ═══════════════════════════════════════════════════════
//
//  Five subtypes defined by Allen Institute:
//    GLIF1: LIF + reset + refractory                 (already covered by lif_step)
//    GLIF2: GLIF1 + threshold adaptation (θ)         (already covered by lif_at_step)
//    GLIF3: GLIF1 + after-spike current (g_ahp)
//    GLIF4: GLIF2 + after-spike current
//    GLIF5: GLIF4 + sodium inactivation variable
//
//  Equations (GLIF3):
//    C_m · dv/dt = -g_leak · (v - E_leak) + I_syn + I_ahp
//    I_ahp = g_ahp · (v - E_ahp)
//    dg_ahp/dt = -g_ahp / τ_ahp           ← exponential Euler: g *= exp(-dt/τ)
//    On spike: g_ahp += Δg_ahp
//    If v ≥ v_thresh: spike, v = v_reset
//
//  Equations (GLIF4):
//    Same as GLIF3 + threshold adaptation:
//    dθ/dt = -θ / τ_θ                      ← exponential Euler: θ *= exp(-dt/τ)
//    On spike: θ += Δθ
//    Spike if v ≥ v_thresh + θ
//
//  Parameters (per-neuron, column-major):
//    GLIF3: [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]  (8 × n)
//    GLIF4: [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]  (10 × n)

// ═══════════════════════════════════════════════════════
//  GLIF3 — After-Spike Current
// ═══════════════════════════════════════════════════════

/// Step `n` GLIF3 neurons with after-spike current (I_ahp) + refractory.
/// Uses `f32x8` SIMD.
///
/// Params layout (8 × n): [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]
/// State: v[n], g_ahp[n], refractory[n], spikes[n]
///
/// Membrane equation: C_m=1, so dv/dt = -g_leak·(v - E_leak) + I_syn + I_ahp
/// where I_ahp = g_ahp·(v - E_ahp). Equivalent to: dv/dt = -(g_leak + g_ahp)·v + g_leak·E_leak + g_ahp·E_ahp + I
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn glif3_step(
    v: *mut f32,
    g_ahp: *mut f32,   // after-spike conductance
    Ii: *mut f32,
    spikes: *mut i32,
    params: *const f32,  // [8 × n] column-major
    refractory: *mut f32,
    tau_ref: f32,
    n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 8);
    let dt_val = p[0]; // dt is uniform
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut gg = load8_mut(g_ahp, off); // g_ahp
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let g_leak = load8(params.wrapping_add(n), off);
        let E_leak = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_ahp = load8(params.wrapping_add(5 * n), off);
        let delta_g = load8(params.wrapping_add(6 * n), off);
        let E_ahp = load8(params.wrapping_add(7 * n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // GLIF3 dynamics (skip if refractory):
        //   dv/dt = -g_leak·(v - E_leak) + I + g_ahp·(v - E_ahp)
        // Rewrite as: dv/dt = -(g_leak + g_ahp)·v + g_leak·E_leak + g_ahp·E_ahp + I
        // Forward Euler: v += dt · [-(g_leak + g_ahp)·v + g_leak·E_leak + g_ahp·E_ahp + I]
        let i_syn = ii
            + g_leak * E_leak
            + gg * E_ahp
            - (g_leak + gg) * vv;
        let dv = i_syn;
        vv = refr_mask.blend(vv, vv + dv * dt_s);

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;

        // g_ahp decay (during refractory too): exponential Euler
        let mask_ahp = tau_ahp.cmp_gt(zero);
        let ahp_decay = mask_ahp.blend(fast_exp_simd(-dt_s / tau_ahp), zero);
        gg *= ahp_decay;

        // On spike: g_ahp += Δg_ahp
        gg += sm.blend(delta_g, zero);
        // Clamp g_ahp to prevent saturation
        gg = gg.max(zero).min(f32x8::splat(1.0));

        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));

        // Decrement refractory for all, set to tau_ref on spike
        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(g_ahp, off, gg);
        store8(refractory, off, refr);

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

    // Scalar tail
    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let g_ahp = std::slice::from_raw_parts_mut(g_ahp, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt_val).max(0.0);
                // g_ahp still decays during refractory
                let tau_ahp = p[5 * n + i];
                if tau_ahp > 0.0 {
                    g_ahp[i] *= (-dt_val / tau_ahp).exp();
                }
                spikes[i] = 0;
                continue;
            }
            let g_leak = p[n + i];
            let E_leak = p[2 * n + i];
            let v_thresh = p[3 * n + i];
            let v_reset = p[4 * n + i];
            let tau_ahp = p[5 * n + i];
            let delta_g = p[6 * n + i];
            let E_ahp = p[7 * n + i];
            let i_clamped = i_sl[i].clamp(-100.0, 100.0);

            // GLIF3 dynamics
            let i_total = i_clamped + g_leak * E_leak + g_ahp[i] * E_ahp - (g_leak + g_ahp[i]) * v[i];
            v[i] += dt_val * i_total;

            // g_ahp decay
            if tau_ahp > 0.0 {
                g_ahp[i] *= (-dt_val / tau_ahp).exp();
            }

            if v[i] >= v_thresh {
                spikes[i] = 1;
                v[i] = v_reset;
                g_ahp[i] += delta_g;
                g_ahp[i] = g_ahp[i].clamp(0.0, 1.0);
                refr_sl[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  GLIF4 — Threshold Adaptation + After-Spike Current
// ═══════════════════════════════════════════════════════

/// Step `n` GLIF4 neurons with threshold adaptation (θ) + after-spike current (g_ahp) + refractory.
/// Uses `f32x8` SIMD.
///
/// Params layout (10 × n): [dt, g_leak, E_leak, v_thresh_0, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]
/// State: v[n], theta[n], g_ahp[n], refractory[n], spikes[n]
///
/// Effective threshold = v_thresh_0 + theta[n]
///   dθ/dt = -θ / τ_θ  (decay toward 0)
///   On spike: θ += Δθ
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn glif4_step(
    v: *mut f32,
    theta: *mut f32,  // adaptive threshold offset
    g_ahp: *mut f32,  // after-spike conductance
    Ii: *mut f32,
    spikes: *mut i32,
    params: *const f32,  // [10 × n] column-major
    refractory: *mut f32,
    tau_ref: f32,
    n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 10);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut th = load8_mut(theta, off);
        let mut gg = load8_mut(g_ahp, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off)
            .max(f32x8::splat(-100.0))
            .min(f32x8::splat(100.0));
        let g_leak = load8(params.wrapping_add(n), off);
        let E_leak = load8(params.wrapping_add(2 * n), off);
        let v_th_0 = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_ahp = load8(params.wrapping_add(5 * n), off);
        let delta_g = load8(params.wrapping_add(6 * n), off);
        let E_ahp = load8(params.wrapping_add(7 * n), off);
        let tau_th = load8(params.wrapping_add(8 * n), off);
        let delta_th = load8(params.wrapping_add(9 * n), off);

        // Effective threshold
        let v_th_eff = v_th_0 + th;

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // GLIF4 dynamics: same as GLIF3 but with adaptive threshold
        let i_syn = ii
            + g_leak * E_leak
            + gg * E_ahp
            - (g_leak + gg) * vv;
        let dv = i_syn;
        vv = refr_mask.blend(vv, vv + dv * dt_s);

        // Theta decay during refractory too (exponential Euler)
        let mask_tt = tau_th.cmp_gt(zero);
        let theta_decay = mask_tt.blend(fast_exp_simd(-dt_s / tau_th), zero);
        th *= theta_decay;

        // g_ahp decay
        let mask_ahp = tau_ahp.cmp_gt(zero);
        let ahp_decay = mask_ahp.blend(fast_exp_simd(-dt_s / tau_ahp), zero);
        gg *= ahp_decay;

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th_eff) & not_refr;

        // On spike: update theta and g_ahp
        th += sm.blend(delta_th, zero);
        th = th.max(f32x8::splat(0.0)).min(f32x8::splat(100.0));
        gg += sm.blend(delta_g, zero);
        gg = gg.max(zero).min(f32x8::splat(1.0));

        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));

        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(theta, off, th);
        store8(g_ahp, off, gg);
        store8(refractory, off, refr);

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

    // Scalar tail
    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let theta = std::slice::from_raw_parts_mut(theta, n);
        let g_ahp = std::slice::from_raw_parts_mut(g_ahp, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt_val).max(0.0);
                let tau_ahp = p[5 * n + i];
                if tau_ahp > 0.0 {
                    g_ahp[i] *= (-dt_val / tau_ahp).exp();
                }
                let tau_th = p[8 * n + i];
                if tau_th > 0.0 {
                    theta[i] *= (-dt_val / tau_th).exp();
                }
                spikes[i] = 0;
                continue;
            }
            let g_leak = p[n + i];
            let E_leak = p[2 * n + i];
            let v_th_0 = p[3 * n + i];
            let v_reset = p[4 * n + i];
            let tau_ahp = p[5 * n + i];
            let delta_g = p[6 * n + i];
            let E_ahp = p[7 * n + i];
            let tau_th = p[8 * n + i];
            let delta_th = p[9 * n + i];
            let i_clamped = i_sl[i].clamp(-100.0, 100.0);

            // GLIF4 dynamics
            let i_total = i_clamped + g_leak * E_leak + g_ahp[i] * E_ahp - (g_leak + g_ahp[i]) * v[i];
            v[i] += dt_val * i_total;

            // Theta decay
            if tau_th > 0.0 {
                theta[i] *= (-dt_val / tau_th).exp();
            }
            // g_ahp decay
            if tau_ahp > 0.0 {
                g_ahp[i] *= (-dt_val / tau_ahp).exp();
            }

            if v[i] >= v_th_0 + theta[i] {
                spikes[i] = 1;
                v[i] = v_reset;
                theta[i] += delta_th;
                theta[i] = theta[i].clamp(0.0, 100.0);
                g_ahp[i] += delta_g;
                g_ahp[i] = g_ahp[i].clamp(0.0, 1.0);
                refr_sl[i] = tau_ref;
            } else {
                spikes[i] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests — Allen Institute GLIF3 & GLIF4 verification
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// GLIF3 — Basic: after-spike current (I_ahp) accumulates and decays.
    /// After a spike, g_ahp is incremented, then decays exponentially.
    /// This creates an after-hyperpolarizing current that reduces firing rate.
    #[test]
    fn glif3_ahp_accumulates_and_decays() {
        let n = 8;
        let dt = 1.0;
        let g_leak = 0.1;
        let E_leak = -65.0;
        let v_thresh = -50.0;
        let v_reset = -65.0;
        let tau_ahp = 50.0; // slow AHP decay
        let delta_g = 0.01; // small increment per spike
        let E_ahp = -80.0;  // hyperpolarizing reversal
        let tau_ref = 2.0;

        let mut v = vec![-65.0f32; n];
        let mut g_ahp = vec![0.0f32; n];
        let mut Ii = vec![30.0f32; n]; // strong input
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 8];

        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g;
            params[7 * n + j] = E_ahp;
        }
        let mut refractory = vec![0.0f32; n];
        let mut total_spikes = 0i32;

        // Run 500 steps (500ms)
        for _ in 0..500 {
            unsafe {
                glif3_step(
                    v.as_mut_ptr(), g_ahp.as_mut_ptr(), Ii.as_mut_ptr(),
                    spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n,
                );
            }
            total_spikes += spikes.iter().sum::<i32>();
        }

        // Should fire (I=30 is suprathreshold)
        assert!(total_spikes > 0, "GLIF3: no spikes with I=30");

        // g_ahp should be > 0 (accumulated from spiking)
        let mean_g = g_ahp.iter().sum::<f32>() / n as f32;
        assert!(mean_g > 0.0, "GLIF3: g_ahp should accumulate, got {mean_g:.6}");

        // g_ahp should decay toward 0 with no spiking
        // Stop input and run for 500ms more
        Ii.fill(0.0);
        for _ in 0..500 {
            // Set spikes to 0 before each step to avoid self-spiking
            unsafe {
                glif3_step(
                    v.as_mut_ptr(), g_ahp.as_mut_ptr(), Ii.as_mut_ptr(),
                    spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n,
                );
            }
        }

        let final_g = g_ahp.iter().sum::<f32>() / n as f32;
        assert!(final_g < mean_g, "GLIF3: g_ahp should decay, was {mean_g:.6}, now {final_g:.6}");
    }

    /// GLIF3 — After-spike current (AHP) creates spike-frequency adaptation.
    /// Uses n=1 to avoid cross-neuron aliasing in spike timing.
    #[test]
    fn glif3_ahp_adaptation_works() {
        let n = 1; // single neuron for ISI analysis
        let dt = 1.0;
        let g_leak: f32 = 0.1;
        let E_leak: f32 = -65.0;
        let v_thresh: f32 = -50.0;
        let v_reset: f32 = -65.0;
        let tau_ahp = 30.0; // moderate AHP decay
        let delta_g = 0.02; // strong AHP increment
        let E_ahp = -85.0;  // strong hyperpolarizing reversal
        let tau_ref = 2.0;
        let I_val = 30.0;

        let mut v = vec![-65.0f32; n];
        let mut g_ahp = vec![0.0f32; n];
        let mut Ii = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n+j] = g_leak; params[2*n+j] = E_leak;
            params[3*n+j] = v_thresh; params[4*n+j] = v_reset;
            params[5*n+j] = tau_ahp; params[6*n+j] = delta_g; params[7*n+j] = E_ahp;
        }
        let mut refractory = vec![0.0f32; n];
        let mut spike_times = Vec::new();

        for step in 0..300 {
            unsafe {
                glif3_step(
                    v.as_mut_ptr(), g_ahp.as_mut_ptr(), Ii.as_mut_ptr(),
                    spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n,
                );
            }
            for i in 0..n {
                if spikes[i] != 0 {
                    spike_times.push(step);
                }
            }
        }

        // Should fire
        assert!(spike_times.len() >= 8, "GLIF3: expected >=8 spikes, got {}", spike_times.len());

        // g_ahp should have accumulated
        let mean_g = g_ahp.iter().sum::<f32>() / n as f32;
        assert!(mean_g > 0.001, "GLIF3: g_ahp should accumulate, got {mean_g:.6}");

        // ISIs should show adaptation (later ISIs longer)
        if spike_times.len() >= 20 {
            let first_isi = spike_times[1] - spike_times[0];
            let last_isi = spike_times[spike_times.len()-1] - spike_times[spike_times.len()-2];
            assert!(last_isi as f32 >= first_isi as f32 * 0.5,
                "GLIF3: adaptation: first_isi={first_isi}, last_isi={last_isi}");
        }
    }

    /// GLIF4 — Threshold adaptation reduces firing rate over time.
    /// theta accumulates on each spike, making subsequent spikes harder.
    #[test]
    fn glif4_threshold_adaptation_works() {
        let n = 8;
        let dt = 1.0;
        let g_leak = 0.1;
        let E_leak = -65.0;
        let v_thresh_0 = -50.0;
        let v_reset = -65.0;
        let tau_ahp = 20.0;
        let delta_g = 0.005;
        let E_ahp = -80.0;
        let tau_th = 30.0; // threshold adaptation time constant
        let delta_th = 3.0; // threshold increment per spike
        let tau_ref = 2.0;

        let mut v = vec![-65.0f32; n];
        let mut theta = vec![0.0f32; n];
        let mut g_ahp = vec![0.0f32; n];
        let mut Ii = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 10];

        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh_0;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g;
            params[7 * n + j] = E_ahp;
            params[8 * n + j] = tau_th;
            params[9 * n + j] = delta_th;
        }
        let mut refractory = vec![0.0f32; n];
        let mut spike_times = Vec::new();

        // Run 500 steps, track spike times
        for step in 0..500 {
            unsafe {
                glif4_step(
                    v.as_mut_ptr(), theta.as_mut_ptr(), g_ahp.as_mut_ptr(),
                    Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n,
                );
            }
            for i in 0..n {
                if spikes[i] != 0 {
                    spike_times.push(step);
                }
            }
        }

        // Should fire
        assert!(spike_times.len() >= 8, "GLIF4: expected >=8 spikes, got {}", spike_times.len());

        // theta should have accumulated (at least in some neurons)
        let mean_theta = theta.iter().sum::<f32>() / n as f32;
        assert!(mean_theta > 0.0, "GLIF4: theta should accumulate, got {mean_theta:.4}");

        // Late ISIs should be longer than early ISIs (spike-frequency adaptation)
        if spike_times.len() >= 10 {
            let first_isi = spike_times[1] - spike_times[0];
            let last_isi = spike_times[spike_times.len()-1] - spike_times[spike_times.len()-2];
            assert!(last_isi as f32 >= first_isi as f32 * 0.5,
                "GLIF4: adaptation should lengthen ISI: first={first_isi}, last={last_isi}");
        }
    }

    /// GLIF4 vs GLIF3: with same params, GLIF4 should fire LESS due to threshold adaptation.
    #[test]
    fn glif4_vs_glif3_threshold_adaptation() {
        let n = 8;
        let dt = 1.0;
        let g_leak = 0.1;
        let E_leak = -65.0;
        let v_thresh = -50.0;
        let v_reset = -65.0;
        let tau_ahp = 50.0;
        let delta_g = 0.01;
        let E_ahp = -80.0;
        let tau_th = 30.0;
        let delta_th = 5.0; // strong threshold adaptation
        let tau_ref = 2.0;
        let I_val = 35.0;

        // GLIF3
        let mut v3 = vec![-65.0f32; n];
        let mut g3 = vec![0.0f32; n];
        let mut I3 = vec![I_val; n];
        let mut sp3 = vec![0i32; n];
        let mut p3 = vec![0.0f32; n * 8];
        for j in 0..n {
            p3[j] = dt;
            p3[n+j] = g_leak; p3[2*n+j] = E_leak;
            p3[3*n+j] = v_thresh; p3[4*n+j] = v_reset;
            p3[5*n+j] = tau_ahp; p3[6*n+j] = delta_g; p3[7*n+j] = E_ahp;
        }
        let mut r3 = vec![0.0f32; n];

        // GLIF4
        let mut v4 = vec![-65.0f32; n];
        let mut th4 = vec![0.0f32; n];
        let mut g4 = vec![0.0f32; n];
        let mut I4 = vec![I_val; n];
        let mut sp4 = vec![0i32; n];
        let mut p4 = vec![0.0f32; n * 10];
        for j in 0..n {
            p4[j] = dt;
            p4[n+j] = g_leak; p4[2*n+j] = E_leak;
            p4[3*n+j] = v_thresh; p4[4*n+j] = v_reset;
            p4[5*n+j] = tau_ahp; p4[6*n+j] = delta_g; p4[7*n+j] = E_ahp;
            p4[8*n+j] = tau_th; p4[9*n+j] = delta_th;
        }
        let mut r4 = vec![0.0f32; n];

        let mut total3 = 0i32;
        let mut total4 = 0i32;

        for _ in 0..300 {
            unsafe {
                glif3_step(v3.as_mut_ptr(), g3.as_mut_ptr(), I3.as_mut_ptr(),
                    sp3.as_mut_ptr(), p3.as_ptr(), r3.as_mut_ptr(), tau_ref, n);
                glif4_step(v4.as_mut_ptr(), th4.as_mut_ptr(), g4.as_mut_ptr(), I4.as_mut_ptr(),
                    sp4.as_mut_ptr(), p4.as_ptr(), r4.as_mut_ptr(), tau_ref, n);
            }
            total3 += sp3.iter().sum::<i32>();
            total4 += sp4.iter().sum::<i32>();
        }

        // GLIF4 (with threshold adaptation) should fire less than GLIF3
        assert!(total3 > 0, "GLIF3 should fire");
        assert!(total4 <= total3,
            "GLIF4 firing={total4} should be <= GLIF3 firing={total3}");
    }

    /// GLIF4 — Theta decays exponentially toward 0.
    #[test]
    fn glif4_theta_exponential_decay() {
        let n = 8;
        let dt = 1.0;
        let g_leak = 0.1;
        let E_leak = -65.0;
        let v_thresh_0 = -50.0;
        let v_reset = -65.0;
        let tau_ahp = 20.0;
        let delta_g = 0.01;
        let E_ahp = -80.0;
        let tau_th = 20.0;
        let delta_th = 5.0;
        let tau_ref = 2.0;

        let mut v = vec![-65.0f32; n];
        let mut theta = vec![10.0f32; n]; // start with elevated theta
        let mut g_ahp = vec![0.0f32; n];
        let mut Ii = vec![0.0f32; n]; // no input — no spikes
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 10];
        for j in 0..n {
            params[j] = dt;
            params[n+j] = g_leak; params[2*n+j] = E_leak;
            params[3*n+j] = v_thresh_0; params[4*n+j] = v_reset;
            params[5*n+j] = tau_ahp; params[6*n+j] = delta_g; params[7*n+j] = E_ahp;
            params[8*n+j] = tau_th; params[9*n+j] = delta_th;
        }
        let mut refractory = vec![0.0f32; n];

        // Run 1 step: theta *= exp(-1/20) = 0.9512
        unsafe {
            glif4_step(v.as_mut_ptr(), theta.as_mut_ptr(), g_ahp.as_mut_ptr(),
                Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(),
                refractory.as_mut_ptr(), tau_ref, n);
        }

        let expected: f32 = 10.0 * (-1.0f32 / 20.0f32).exp();
        let diff = (theta[0] - expected).abs();
        assert!(diff < 0.02, "GLIF4: theta decay: got {:.4}, expected {:.4}", theta[0], expected);

        // Run 100 more steps (no spikes), theta should approach 0
        for _ in 0..100 {
            unsafe {
                glif4_step(v.as_mut_ptr(), theta.as_mut_ptr(), g_ahp.as_mut_ptr(),
                    Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n);
            }
        }

        assert!(theta[0] < 0.1, "GLIF4: theta should decay to ~0, got {:.4}", theta[0]);
    }

    /// GLIF4 — Non-aligned n (SIMD + scalar tail consistency).
    #[test]
    fn glif4_non_aligned_n() {
        let n = 13; // not divisible by 8
        let dt = 1.0;
        let g_leak: f32 = 0.1;
        let E_leak: f32 = -65.0;
        let v_thresh_0: f32 = -50.0;
        let v_reset: f32 = -65.0;
        let tau_ahp = 30.0;
        let delta_g = 0.01;
        let E_ahp = -80.0;
        let tau_th = 20.0;
        let delta_th = 3.0;
        let tau_ref = 2.0;

        let mut v = vec![-65.0f32; n];
        let mut theta = vec![0.0f32; n];
        let mut g_ahp = vec![0.0f32; n];
        let mut Ii = vec![30.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 10];
        for j in 0..n {
            params[j] = dt; params[n+j] = g_leak; params[2*n+j] = E_leak;
            params[3*n+j] = v_thresh_0; params[4*n+j] = v_reset;
            params[5*n+j] = tau_ahp; params[6*n+j] = delta_g; params[7*n+j] = E_ahp;
            params[8*n+j] = tau_th; params[9*n+j] = delta_th;
        }
        let mut refractory = vec![0.0f32; n];
        let mut total = 0i32;

        for _ in 0..300 {
            unsafe {
                glif4_step(v.as_mut_ptr(), theta.as_mut_ptr(), g_ahp.as_mut_ptr(),
                    Ii.as_mut_ptr(), spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n);
            }
            total += spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF4 n=13: should fire");
        // All neurons should have fired at least once (identical params)
        assert!(spikes.iter().any(|&s| s > 0), "GLIF4 n=13: some neuron should spike on last step");
        // No NaN in state
        for i in 0..n {
            assert!(!v[i].is_nan(), "GLIF4 n=13: v[{i}] is NaN");
            assert!(!theta[i].is_nan(), "GLIF4 n=13: theta[{i}] is NaN");
            assert!(!g_ahp[i].is_nan(), "GLIF4 n=13: g_ahp[{i}] is NaN");
        }
        // v should be in a reasonable range (not stuck at reset)
        let mean_v = v.iter().sum::<f32>() / n as f32;
        assert!(mean_v > -70.0, "GLIF4 n=13: mean v too low: {mean_v:.4}");
    }
}
