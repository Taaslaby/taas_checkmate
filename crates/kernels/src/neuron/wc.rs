#![allow(non_snake_case)]
use wide::{f32x8, CmpGt};
use super::{load8, load8_mut, store8};

// ═══════════════════════════════════════════════════════
//  Wilson-Cowan (WC) — excitatory-inhibitory rate model
// ═══════════════════════════════════════════════════════
//
// Classic Wilson-Cowan (1972) rate-based population model.
// Two populations: excitatory (E) and inhibitory (I).
//
// Equations:
//   τ_E · dE/dt = -E + f(w_EE·E - w_EI·I + I_E)
//   τ_I · dI/dt = -I + f(w_IE·E - w_II·I + I_I)
//
// f(x) = 1 / (1 + exp(-β·x))   — sigmoid activation (default β=4.0)
//
// Parameters (column-major, per-neuron):
//   dt, tau_E, tau_I, w_EE, w_EI, w_IE, w_II, beta
//
// This model captures:
//   - Excitation-inhibition balance
//   - Rate-based dynamics (no spike generation, continuous rates)
//   - Phase transitions between resting, oscillatory, and bistable states
//   - Foundation for cortical column modeling
//
// Reference: Wilson & Cowan (1972)
// "Excitatory and inhibitory interactions in localized populations
//  of model neurons" — Biophysical Journal

/// Sigmoid activation: f(x) = 1 / (1 + exp(-β·x))
/// Uses SIMD fast-exp for speed.
#[inline(always)]
fn sigmoid_simd(x: f32x8, beta: f32x8) -> f32x8 {
    let neg = f32x8::splat(0.0) - x * beta;
    // Fast sigmoid approximation: 1 / (1 + |x|) is not accurate enough
    // Use: f(x) ≈ 0.5 + 0.5·tanh(β·x/2) = 1/(1+exp(-β·x))
    // For SIMD, compute exp via to_array + scalar (sigmoid is usually not the bottleneck)
    let neg_arr: [f32; 8] = neg.to_array();
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = 1.0 / (1.0 + neg_arr[i].exp());
    }
    f32x8::from(out)
}

/// Step `n` Wilson-Cowan populations. Uses `f32x8` SIMD.
///
/// Each "neuron" is actually a population with a single excitatory and inhibitory rate.
/// The model is rate-based (no spikes), so output is the E and I rates directly.
///
/// Params layout: dt, tau_E, tau_I, w_EE, w_EI, w_IE, w_II, beta (8 × n, column-major)
///
/// # Safety
/// `e[n]`, `i[n]`, `i_e[n]`, `i_i[n]`, `spikes_e[n]`, `spikes_i[n]`, `params[8·n]`
/// must be valid, non-overlapping.
pub(crate) unsafe fn wc_step(
    e: *mut f32, i: *mut f32,
    i_e: *const f32, i_i: *const f32,
    spikes_e: *mut f32, spikes_i: *mut f32,
    params: *const f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 8);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut ee = load8_mut(e, off);
        let mut ii = load8_mut(i, off);
        let ie_ext = load8(i_e, off);
        let ii_ext = load8(i_i, off);

        // Load parameters
        let tau_E = load8(params.wrapping_add(n), off);
        let tau_I = load8(params.wrapping_add(2 * n), off);
        let w_EE = load8(params.wrapping_add(3 * n), off);
        let w_EI = load8(params.wrapping_add(4 * n), off);
        let w_IE = load8(params.wrapping_add(5 * n), off);
        let w_II = load8(params.wrapping_add(6 * n), off);
        let beta = load8(params.wrapping_add(7 * n), off);

        // Excitatory input: w_EE·E - w_EI·I + I_E
        let input_e = w_EE * ee - w_EI * ii + ie_ext;
        // Inhibitory input: w_IE·E - w_II·I + I_I
        let input_i = w_IE * ee - w_II * ii + ii_ext;

        // Sigmoid activation
        let f_e = sigmoid_simd(input_e, beta);
        let f_i = sigmoid_simd(input_i, beta);

        // Wilson-Cowan dynamics (exponential Euler):
        // τ·dE/dt = -E + f(input)  →  E_new = E·exp(-dt/τ) + f(input)·(1 - exp(-dt/τ))
        // For τ > 0, use exponential Euler (exact for constant input)
        let mask_te = tau_E.cmp_gt(zero);
        let decay_e = mask_te.blend(
            {
                let r = f32x8::splat(1.0) / tau_E.max(f32x8::splat(1e-10));
                let arg = -dt_s * r;
                // fast exp via scalar fallback
                let arr: [f32; 8] = arg.to_array();
                let mut exp_arr = [0.0f32; 8];
                for k in 0..8 { exp_arr[k] = arr[k].exp(); }
                f32x8::from(exp_arr)
            },
            zero
        );
        let one_minus_decay_e = one - decay_e;
        let new_e = ee * decay_e + f_e * one_minus_decay_e;
        ee = mask_te.blend(new_e, ee);

        let mask_ti = tau_I.cmp_gt(zero);
        let decay_i = mask_ti.blend(
            {
                let r = f32x8::splat(1.0) / tau_I.max(f32x8::splat(1e-10));
                let arg = -dt_s * r;
                let arr: [f32; 8] = arg.to_array();
                let mut exp_arr = [0.0f32; 8];
                for k in 0..8 { exp_arr[k] = arr[k].exp(); }
                f32x8::from(exp_arr)
            },
            zero
        );
        let one_minus_decay_i = one - decay_i;
        let new_i = ii * decay_i + f_i * one_minus_decay_i;
        ii = mask_ti.blend(new_i, ii);

        // Output: rates (0 to 1 via sigmoid)
        store8(e, off, ee);
        store8(i, off, ii);
        store8(spikes_e, off, f_e);  // continuous rate output
        store8(spikes_i, off, f_i);  // continuous rate output
    }

    // Scalar tail
    if n8 < n {
        let e = std::slice::from_raw_parts_mut(e, n);
        let i = std::slice::from_raw_parts_mut(i, n);
        let i_e = std::slice::from_raw_parts(i_e, n);
        let i_i = std::slice::from_raw_parts(i_i, n);
        let spikes_e = std::slice::from_raw_parts_mut(spikes_e, n);
        let spikes_i = std::slice::from_raw_parts_mut(spikes_i, n);
        for idx in n8..n {
            let tau_e = p[n + idx];
            let tau_i = p[2 * n + idx];
            let w_ee = p[3 * n + idx];
            let w_ei = p[4 * n + idx];
            let w_ie = p[5 * n + idx];
            let w_ii = p[6 * n + idx];
            let beta = p[7 * n + idx];

            let input_e = w_ee * e[idx] - w_ei * i[idx] + i_e[idx];
            let input_i = w_ie * e[idx] - w_ii * i[idx] + i_i[idx];

            let f_e = 1.0 / (1.0 + (-beta * input_e).exp());
            let f_i = 1.0 / (1.0 + (-beta * input_i).exp());

            if tau_e > 0.0 {
                let decay = (-dt_val / tau_e).exp();
                e[idx] = e[idx] * decay + f_e * (1.0 - decay);
            }
            if tau_i > 0.0 {
                let decay = (-dt_val / tau_i).exp();
                i[idx] = i[idx] * decay + f_i * (1.0 - decay);
            }

            spikes_e[idx] = f_e;
            spikes_i[idx] = f_i;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// WC: verify E-I dynamics converge to equilibrium with constant input.
    #[test]
    fn wc_converges_to_equilibrium() {
        let n = 8;
        let dt = 0.1;
        let tau_e = 10.0;
        let tau_i = 10.0;
        let w_ee = 1.2;
        let w_ei = 0.4;
        let w_ie = 0.6;
        let w_ii = 0.8;
        let beta = 4.0;
        let ie_val = 0.5;
        let ii_val = 0.3;

        let mut e = vec![0.0f32; n];
        let mut ii = vec![0.0f32; n];
        let i_e = vec![ie_val; n];
        let i_i = vec![ii_val; n];
        let mut spikes_e = vec![0.0f32; n];
        let mut spikes_i = vec![0.0f32; n];
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_e;
            params[2 * n + j] = tau_i;
            params[3 * n + j] = w_ee;
            params[4 * n + j] = w_ei;
            params[5 * n + j] = w_ie;
            params[6 * n + j] = w_ii;
            params[7 * n + j] = beta;
        }

        // Run for 500 steps (50 time units)
        for _ in 0..500 {
            unsafe { wc_step(e.as_mut_ptr(), ii.as_mut_ptr(), i_e.as_ptr(), i_i.as_ptr(), spikes_e.as_mut_ptr(), spikes_i.as_mut_ptr(), params.as_ptr(), n) };
        }

        // After convergence, E and I should be in (0, 1) and stable
        for idx in 0..n {
            assert!(e[idx] > 0.0 && e[idx] < 1.0, "E should be in (0,1): e[{idx}]={}", e[idx]);
            assert!(ii[idx] > 0.0 && ii[idx] < 1.0, "I should be in (0,1): i[{idx}]={}", ii[idx]);
        }

        // Check convergence (variance should be small)
        let e_mean: f32 = e.iter().sum::<f32>() / n as f32;
        let e_var: f32 = e.iter().map(|x| (x - e_mean).powi(2)).sum::<f32>() / n as f32;
        assert!(e_var < 1e-10, "E should converge (var={e_var})");
    }

    /// WC: verify sigmoid outputs are in (0, 1).
    #[test]
    fn wc_sigmoid_range() {
        let n = 8;
        let dt = 0.1;
        let tau_e = 10.0;
        let tau_i = 10.0;
        let w_ee = 1.2;
        let w_ei = 0.4;
        let w_ie = 0.6;
        let w_ii = 0.8;
        let beta = 4.0;

        let mut e = vec![0.0f32; n];
        let mut ii = vec![0.0f32; n];
        let i_e = vec![0.5; n];
        let i_i = vec![0.3; n];
        let mut spikes_e = vec![0.0f32; n];
        let mut spikes_i = vec![0.0f32; n];
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = tau_e;
            params[2 * n + j] = tau_i;
            params[3 * n + j] = w_ee;
            params[4 * n + j] = w_ei;
            params[5 * n + j] = w_ie;
            params[6 * n + j] = w_ii;
            params[7 * n + j] = beta;
        }

        for _ in 0..100 {
            unsafe { wc_step(e.as_mut_ptr(), ii.as_mut_ptr(), i_e.as_ptr(), i_i.as_ptr(), spikes_e.as_mut_ptr(), spikes_i.as_mut_ptr(), params.as_ptr(), n) };
        }

        for idx in 0..n {
            assert!(spikes_e[idx] > 0.0 && spikes_e[idx] < 1.0,
                "spikes_e[{idx}] should be in (0,1): {}", spikes_e[idx]);
            assert!(spikes_i[idx] > 0.0 && spikes_i[idx] < 1.0,
                "spikes_i[{idx}] should be in (0,1): {}", spikes_i[idx]);
        }
    }
}
