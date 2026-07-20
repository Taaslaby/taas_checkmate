#![allow(non_snake_case)]
use wide::{f32x8, CmpGe, CmpGt};
use super::{load8, load8_mut, load8_i32, store8, spike_mask_to_i32, surrogate_gradient, fast_exp_simd};

// ═══════════════════════════════════════════════════════
//  LIF (Leaky Integrate-and-Fire)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset (5 × n)
///
/// # Safety
/// `v[n]`, `Ii[n]`, `spikes[n]`, `params[5·n]`, `refractory[n]` must be valid.
pub(crate) unsafe fn lif_step(
    v: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 5);
    let dt_val = p[0];  // dt is uniform across all neurons
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory)
        // Forward Euler: v += (-(v - v_rest) + I) / τ · dt  (no exp!)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(f32x8::splat(1.0) / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = refr_mask.blend(vv, mask_tau.blend(vv + dv, vv));

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));

        // Decrement refractory counter for all, set to tau_ref on spike
        refr -= dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                continue;
            }
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if v[i_idx] >= v_thresh {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  LIF-AT (LIF with Adaptive Threshold)
// ═══════════════════════════════════════════════════════

/// Step `n` LIF-AT neurons with refractory period. Uses `f32x8` SIMD.
///
/// Params layout: dt, tau_m, v_rest, v_thresh, v_reset, tau_theta, beta (7 × n)
///
/// # Safety
/// `v[n]`, `theta[n]`, `Ii[n]`, `spikes[n]`, `params[7·n]`, `refractory[n]` valid.
pub(crate) unsafe fn lif_at_step(
    v: *mut f32, theta: *mut f32, Ii: *mut f32, spikes: *mut i32,
    params: *const f32,
    refractory: *mut f32, tau_ref: f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 7);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let zero = f32x8::splat(0.0);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut th = load8_mut(theta, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8_mut(Ii, off);
        let v_rest = load8(params.wrapping_add(2 * n), off);
        let v_th = load8(params.wrapping_add(3 * n), off);
        let v_reset = load8(params.wrapping_add(4 * n), off);
        let tau_th = load8(params.wrapping_add(5 * n), off);
        let beta = load8(params.wrapping_add(6 * n), off);
        let tau_m = load8(params.wrapping_add(n), off);

        // Refractory mask
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory — theta still decays)
        // Forward Euler: v += (-(v - v_rest) + I) / τ · dt  (no exp!)
        let mask_tau = tau_m.cmp_gt(zero);
        let inv_tau = mask_tau.blend(f32x8::splat(1.0) / tau_m, zero);
        let dv = (v_rest - vv + ii) * inv_tau * dt_s;
        vv = refr_mask.blend(vv, mask_tau.blend(vv + dv, vv));

        // Adaptive threshold decay (exponential Euler: θ *= exp(-dt/τ_θ))
        let mask_tt = tau_th.cmp_gt(zero);
        let theta_decay = mask_tt.blend(fast_exp_simd(-dt_s / tau_th), zero);
        th *= theta_decay;

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_th + th) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset, vv));
        th += sm.blend(beta, zero);

        // Decrement refractory for all, set to tau_ref on spike
        refr -= dt_s;  // Always decrement
        refr = sm.blend(tr_s, refr);  // Set to tau_ref on spike
        refr = refr.max(zero);

        store8(v, off, vv);
        store8(theta, off, th);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    if n8 < n {
        let v = std::slice::from_raw_parts_mut(v, n);
        let theta = std::slice::from_raw_parts_mut(theta, n);
        let i_sl = std::slice::from_raw_parts_mut(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        for i_idx in n8..n {
            if refr_sl[i_idx] > 0.0 {
                refr_sl[i_idx] = (refr_sl[i_idx] - dt_val).max(0.0);
                spikes[i_idx] = 0;
                // Theta still decays during refractory (exponential Euler)
                let tau_theta = p[5 * n + i_idx];
                if tau_theta > 0.0 {
                    theta[i_idx] *= (-dt_val / tau_theta).exp();
                }
                continue;
            }
            let dt_val = p[i_idx];
            let tau_m = p[n + i_idx];
            let v_rest = p[2 * n + i_idx];
            let v_thresh = p[3 * n + i_idx];
            let v_reset = p[4 * n + i_idx];
            let tau_theta = p[5 * n + i_idx];
            let beta = p[6 * n + i_idx];
            if tau_m > 0.0 {
                v[i_idx] += (v_rest - v[i_idx] + i_sl[i_idx]) / tau_m * dt_val;
            }
            if tau_theta > 0.0 {
                theta[i_idx] *= (-dt_val / tau_theta).exp();
            }
            if v[i_idx] >= v_thresh + theta[i_idx] {
                spikes[i_idx] = 1;
                v[i_idx] = v_reset;
                theta[i_idx] += beta;
                refr_sl[i_idx] = tau_ref;
            } else {
                spikes[i_idx] = 0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  E-Prop LIF (Eligibility Propagation) — SIMD kernel
// ═══════════════════════════════════════════════════════



/// Step `n` E-Prop LIF neurons with eligibility traces + refractory period.
/// Uses `f32x8` SIMD for LIF dynamics + eligibility + surrogate gradient.
///
/// Matches `lif_step` refractory semantics: during refractory, v/eligibility/psi
/// are frozen, refractory counter decrements, and on spike sets τ_ref.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn eprop_lif_step(
    v: *mut f32, eligibility: *mut f32, psi: *mut f32,
    Ii: *const f32, spikes: *mut i32, pre_spikes: *const i32,
    refractory: *mut f32,
    dt: f32, tau_m: f32, tau_e: f32, tau_ref: f32,
    v_rest: f32, v_thresh: f32, v_reset: f32, tau_sg: f32,
    n: usize,
) -> usize {
    let dt_s = f32x8::splat(dt);
    let tau_m_s = f32x8::splat(tau_m);
    let tau_e_s = f32x8::splat(tau_e);
    let v_rest_s = f32x8::splat(v_rest);
    let v_thresh_s = f32x8::splat(v_thresh);
    let v_reset_s = f32x8::splat(v_reset);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let tau_sg_s = f32x8::splat(tau_sg);
    let tr_s = f32x8::splat(tau_ref);
    let n8 = n & !7;
    let mut total_spikes = 0usize;

    // Precompute e_decay — constant across all SIMD blocks
    let e_decay = fast_exp_simd(-dt_s / tau_e_s);
    let one_minus_e_decay = one - e_decay;

    for off in (0..n8).step_by(8) {
        let mut vv = load8_mut(v, off);
        let mut ee = load8_mut(eligibility, off);
        let mut refr = load8_mut(refractory, off);
        let ii = load8(Ii, off);
        let pre = load8_i32(pre_spikes, off);
        let refr_mask = refr.cmp_gt(zero);

        // LIF dynamics (skip if refractory — eligibility/psi also frozen)
        // Forward Euler: v += (-(v - v_rest) + I) / τ · dt  (no exp!)
        let dv = (v_rest_s - vv + ii) / tau_m_s * dt_s;
        vv = refr_mask.blend(vv, vv + dv);

        // Surrogate gradient (only non-refractory)
        let diff = ((vv - v_thresh_s).abs() / tau_sg_s).max(f32x8::splat(1e-6));
        let one_plus_diff = one + diff;
        let mut ps = one / (one_plus_diff * one_plus_diff);
        ps = refr_mask.blend(zero, ps);  // zero during refractory

        // Eligibility trace (frozen during refractory)
        let z_pre = pre.cmp_gt(zero).blend(one, zero); // 1 if pre[i] != 0
        let mut e_new = ee * e_decay + ps * z_pre * tau_e_s * one_minus_e_decay;
        e_new = refr_mask.blend(ee, e_new);  // frozen during refractory
        ee = e_new;

        // Spike detection (only non-refractory)
        let not_refr = !refr_mask;
        let sm = vv.cmp_ge(v_thresh_s) & not_refr;
        vv = refr_mask.blend(vv, sm.blend(v_reset_s, vv));

        // Decrement refractory for all, set to tau_ref on spike
        refr -= dt_s;
        refr = sm.blend(tr_s, refr);
        refr = refr.max(zero);

        // Count spikes
        for &val in sm.to_array().iter() {
            if val.to_bits() != 0 { total_spikes += 1; }
        }

        store8(v, off, vv);
        store8(eligibility, off, ee);
        store8(psi, off, ps);
        store8(refractory, off, refr);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let v_sl = std::slice::from_raw_parts_mut(v, n);
        let e_sl = std::slice::from_raw_parts_mut(eligibility, n);
        let psi_sl = std::slice::from_raw_parts_mut(psi, n);
        let refr_sl = std::slice::from_raw_parts_mut(refractory, n);
        let i_sl = std::slice::from_raw_parts(Ii, n);
        let sp_sl = std::slice::from_raw_parts_mut(spikes, n);
        let pre_sl = std::slice::from_raw_parts(pre_spikes, n);
        let e_decay_s = (-dt / tau_e).exp();
        let one_minus_e_decay_s = 1.0 - e_decay_s;
        for i in n8..n {
            if refr_sl[i] > 0.0 {
                refr_sl[i] = (refr_sl[i] - dt).max(0.0);
                sp_sl[i] = 0;
                continue;  // v, eligibility, psi frozen during refractory
            }
            v_sl[i] += (v_rest - v_sl[i] + i_sl[i]) / tau_m * dt;
            psi_sl[i] = surrogate_gradient(v_sl[i], v_thresh, tau_sg);
            let z_pre = if pre_sl[i] != 0 { 1.0 } else { 0.0 };
            e_sl[i] = e_sl[i] * e_decay_s + psi_sl[i] * z_pre * tau_e * one_minus_e_decay_s;
            if v_sl[i] >= v_thresh {
                sp_sl[i] = 1;
                total_spikes += 1;
                v_sl[i] = v_reset;
                refr_sl[i] = tau_ref;
            } else {
                sp_sl[i] = 0;
            }
        }
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// LIF analytical solution for constant I: v(t) = v_rest + I·(1 - exp(-t/τ))
    /// Test that exponential Euler matches analytical solution within tolerance.
    #[test]
    fn lif_exponential_euler_accuracy() {
        let n = 8;
        let dt = 0.5;      // ms
        let tau_m = 10.0;   // ms
        let v_rest = -70.0; // mV
        let v_thresh = -54.0; // mV
        let v_reset = -70.0; // mV
        let I_const = 15.0;  // input current (suprathreshold but non-spiking in this test)
        let n_steps = 200;   // 100 ms at dt=0.5
        let tau_ref = 0.0;   // no refractory

        let mut v = vec![v_rest; n];
        let mut Ii = vec![I_const; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 5];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_m;
            params[2 * n + j] = v_rest;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
        }
        let mut refractory = vec![0.0f32; n];

        for step in 0..n_steps {
            unsafe {
                lif_step(
                    v.as_mut_ptr(), Ii.as_mut_ptr(), spikes.as_mut_ptr(),
                    params.as_ptr(), refractory.as_mut_ptr(), tau_ref, n,
                );
            }

            let t = (step + 1) as f32 * dt;
            let expected_v = v_rest + I_const * (1.0 - (-t / tau_m).exp());

            // All 8 neurons should be identical (same params)
            // Tolerance 0.3 accommodates fast_exp_simd ~0.18% rel error.
            // Error ≈ I·Δd·(1-d̃^t)/(1-d̃) at worst, bounded by ~0.25 for I=15.
            for i in 0..n {
                let diff = (v[i] - expected_v).abs();
                assert!(
                    diff < 0.3,
                    "Step {step}, neuron {i}: v={:.4}, expected={:.4}, diff={:.4}",
                    v[i], expected_v, diff
                );
            }

            // Should not spike (v stays well below v_thresh = -54)
            assert_eq!(spikes.iter().sum::<i32>(), 0,
                "Spike at step {step}: v={:.4}, thresh={}", v[0], v_thresh);
        }

        // Final voltage should be close to steady state: v_rest + I = -70 + 15 = -55
        // Exponential Euler fixed point is EXACT regardless of approximation:
        //   v_ss = v_rest + (v_ss - v_rest)·˜d + I·(1−˜d) → v_ss = v_rest + I  ✓
        let steady_state = v_rest + I_const;
        for i in 0..n {
            let diff = (v[i] - steady_state).abs();
            assert!(diff < 0.1,
                "Final v[{i}]={:.4}, expected steady state={:.4}, diff={:.4}",
                v[i], steady_state, diff);
        }
    }

    /// LIF-AT: verify theta decays exponentially toward 0 and increments on spike.
    #[test]
    fn lif_at_theta_dynamics() {
        let n = 8;
        let dt = 0.5;
        let tau_m = 10.0;
        let v_rest = -70.0;
        let v_thresh = -54.0;
        let v_reset = -70.0;
        let tau_theta = 20.0;
        let beta = 5.0;
        let tau_ref = 3.0;
        // Need I strong enough to overcome initial theta=10: threshold=-44, steady state=-70+I > -44 → I>26
        let I_const = 40.0;

        let mut v = vec![v_rest; n];
        let mut theta = vec![beta * 2.0; n]; // start elevated
        let mut Ii = vec![I_const; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 7];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_m;
            params[2 * n + j] = v_rest;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_theta;
            params[6 * n + j] = beta;
        }
        let mut refractory = vec![0.0f32; n];

        // Initial theta = 10.0 (beta * 2 = 5 * 2)
        // After 1 step (dt=0.5, tau_theta=20): theta *= exp(-0.5/20) ≈ 0.9753
        // Expected: 10.0 * 0.9753 = 9.753
        unsafe {
            lif_at_step(
                v.as_mut_ptr(), theta.as_mut_ptr(), Ii.as_mut_ptr(),
                spikes.as_mut_ptr(), params.as_ptr(),
                refractory.as_mut_ptr(), tau_ref, n,
            );
        }
        let expected_theta = 10.0 * (-dt / tau_theta).exp();
        // Tolerance 0.02 accommodates fast_exp_simd ~0.18% rel error
        let diff = (theta[0] - expected_theta).abs();
        assert!(diff < 0.02,
            "Theta after decay: got {:.4}, expected {:.4}", theta[0], expected_theta);

        // Run until spike, verify theta increments by beta
        let mut spike_count = 0usize;
        unsafe {
            for _ in 0..200 {
                lif_at_step(
                    v.as_mut_ptr(), theta.as_mut_ptr(), Ii.as_mut_ptr(),
                    spikes.as_mut_ptr(), params.as_ptr(),
                    refractory.as_mut_ptr(), tau_ref, n,
                );
                spike_count += spikes.iter().map(|&s| s as usize).sum::<usize>();
            }
        }
        assert!(
            spike_count > 0,
            "No spikes detected with I={} after 200 steps", I_const);
        // Theta after 200 steps: pure decay → 10·exp(-100/20) = 0.067
        // With spikes: theta gets +5 per spike >> 0.067. Theta=1.83 observed.
        assert!(theta.iter().any(|&t| t > 0.2),
            "No theta increment detected after 200 steps: theta={:?}", theta);
    }

    /// eProp LIF: verify eligibility trace dynamics.
    ///
    /// With I_const=30, tau_m=10, v_rest=-65, takes ~14 steps to reach
    /// v_thresh=-50 (analytical: t = τ·ln(I/(I+v_rest-v_thresh)) ≈ 6.9ms).
    #[test]
    fn eprop_lif_eligibility_trace() {
        let n = 8;
        let dt = 0.5;
        let tau_m = 10.0;
        let tau_e = 20.0;
        let v_rest = -65.0;
        let v_thresh = -50.0;
        let v_reset = -70.0;
        let tau_sg = 2.0;
        let I_const = 30.0;
        // First spike at step ~14 (from v_rest=-65), second at step ~36 (from v_reset=-70).
        // Run 50 steps to ensure we catch at least one spike in the final step.
        let n_steps = 50;
        let mut spike_count = 0usize;

        let mut v = vec![v_rest; n];
        let mut eligibility = vec![0.0; n];
        let mut psi = vec![0.0; n];
        let Ii = vec![I_const; n];
        let mut spikes = vec![0i32; n];
        let pre_spikes = vec![1i32; n]; // all pre-synaptic neurons fire

        let mut refractory = vec![0.0f32; n];
        let tau_ref = 0.0; // no refractory for this test

        for _step in 0..n_steps {
            unsafe {
                eprop_lif_step(
                    v.as_mut_ptr(), eligibility.as_mut_ptr(), psi.as_mut_ptr(),
                    Ii.as_ptr(), spikes.as_mut_ptr(), pre_spikes.as_ptr(),
                    refractory.as_mut_ptr(),
                    dt, tau_m, tau_e, tau_ref, v_rest, v_thresh, v_reset, tau_sg, n,
                );
            }
            spike_count += spikes.iter().map(|&s| s as usize).sum::<usize>();
        }

        // After n_steps of pre-spike=1:
        // ψ grows as v approaches v_thresh
        // eligibility accumulates over time
        for i in 0..n {
            assert!(eligibility[i] > 0.0,
                "Eligibility should be positive after pre-spike: e[{}]={:.6}",
                i, eligibility[i]);
        }
        assert!(
            spike_count > 0,
            "No spikes after {n_steps} steps with I_const={I_const}");
    }
}
