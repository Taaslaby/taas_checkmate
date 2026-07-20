//! Pure Rust SIMD signal processing — `wide::f32x8`.
//! Replaces all ASM IO/gate kernels with portable SIMD.

use wide::{f32x8, CmpGt};

#[inline(always)]
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3),
                 *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_mut(ptr: *mut f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3),
                 *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
    let a = v.to_array();
    let p = ptr.add(off);
    *p = a[0]; *p.add(1) = a[1]; *p.add(2) = a[2]; *p.add(3) = a[3];
    *p.add(4) = a[4]; *p.add(5) = a[5]; *p.add(6) = a[6]; *p.add(7) = a[7];
}

// Sigmoid via fast_exp_simd from neuron module (~3× faster than scalar f32::exp).
// f32x8 doesn't have exp() — use the crate's SIMD polynomial approximation.
#[inline(always)]
fn simd_sigmoid(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    // Clamp x to [-10, 10] for fast_exp_simd range (overflow protection)
    let x_clamped = x.max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + super::fast_exp_simd(-x_clamped))
}

/// Safety-clamp values in [min, max]. Uses f32x8 SIMD.
///
/// # Safety
/// `data` must be non-null, properly aligned, and valid for `n` elements.
pub(crate) unsafe fn safety_clamp(data: *mut f32, n: usize, min: f32, max: f32) {
    let min_s = f32x8::splat(min);
    let max_s = f32x8::splat(max);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let v = load8(data, off).max(min_s).min(max_s);
        store8(data, off, v);
    }
    if n8 < n {
        let s = std::slice::from_raw_parts_mut(data, n);
        for v in s[n8..].iter_mut() { *v = v.clamp(min, max); }
    }
}

/// Stochastic spike encoder: compare input+noise against threshold.
///
/// # Safety
/// `target_i`, `inputs`, `noise` must each be non-null, aligned, and valid for `n` elements.
pub(crate) unsafe fn stochastic_encoder(
    target_i: *mut f32, inputs: *const f32,
    noise: *const f32, threshold: f32, spike_val: f32, n: usize,
) {
    let th_s = f32x8::splat(threshold);
    let sv_s = f32x8::splat(spike_val);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let inp = load8(inputs, off);
        let no = load8(noise, off);
        let mask = (inp + no).cmp_gt(th_s);
        store8(target_i, off, mask.blend(sv_s, zero));
    }
    if n8 < n {
        let t = std::slice::from_raw_parts_mut(target_i, n);
        let inp = std::slice::from_raw_parts(inputs, n);
        let no = std::slice::from_raw_parts(noise, n);
        for i in n8..n {
            t[i] = if inp[i] + no[i] > threshold { spike_val } else { 0.0 };
        }
    }
}

/// Linear readout: output = W·states + bias.
/// Inner loop is scalar (matrix-vector is hard to SIMDize with wide).
///
/// # Safety
/// Pointers must be valid: `outputs[n_out]`, `states[n_in]`, `weights[n_in·n_out]`, `biases[n_out]`.
pub(crate) unsafe fn linear_decoder(
    outputs: *mut f32, states: *const f32,
    weights: *const f32, biases: *const f32,
    n_in: usize, n_out: usize,
) {
    let out = std::slice::from_raw_parts_mut(outputs, n_out);
    let s = std::slice::from_raw_parts(states, n_in);
    let w = std::slice::from_raw_parts(weights, n_in * n_out);
    let b = std::slice::from_raw_parts(biases, n_out);
    for o in 0..n_out {
        let mut sum = b[o];
        // Use f32x8 for inner dot product
        let mut vi = f32x8::splat(0.0);
        let n8 = n_in & !7;
        let row_off = o * n_in;
        let mut i = 0;
        while i < n8 {
            let ws = load8(weights, row_off + i);
            let ss = load8(states, i);
            vi += ws * ss;
            i += 8;
        }
        let arr: [f32; 8] = vi.to_array();
        sum += arr.iter().sum::<f32>();
        for j in i..n_in { sum += w[row_off + j] * s[j]; }
        out[o] = sum;
    }
}

/// Thalamic gate: target_i = source_v × σ(gain·(gate_v − threshold)).
///
/// # Safety
/// `target_i`, `source_v`, `gate_v` must be valid for `n` elements.
pub(crate) unsafe fn thalamic_gate(
    target_i: *mut f32, source_v: *const f32,
    gate_v: *const f32, threshold: f32, gain: f32, n: usize,
) {
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let sv = load8(source_v, off);
        let gv = load8(gate_v, off);
        let gate = simd_sigmoid(gain_s * (gv - th_s));
        store8(target_i, off, sv * gate);
    }
    if n8 < n {
        let t = std::slice::from_raw_parts_mut(target_i, n);
        let s = std::slice::from_raw_parts(source_v, n);
        let g = std::slice::from_raw_parts(gate_v, n);
        for i in n8..n {
            let sig = 1.0 / (1.0 + (-gain * (g[i] - threshold)).exp());
            t[i] = s[i] * sig;
        }
    }
}

/// Elementwise sigmoid: σ(gain·(x − threshold)).
///
/// # Safety
/// `output`, `input` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn sigmoid_gate(
    output: *mut f32, input: *const f32,
    gain: f32, threshold: f32, n: usize,
) {
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let inp = load8(input, off);
        store8(output, off, simd_sigmoid(gain_s * (inp - th_s)));
    }
    if n8 < n {
        let out = std::slice::from_raw_parts_mut(output, n);
        let inp = std::slice::from_raw_parts(input, n);
        for i in n8..n {
            out[i] = 1.0 / (1.0 + (-gain * (inp[i] - threshold)).exp());
        }
    }
}

/// Phase gate: output = input × σ(gain·(r − threshold)).
///
/// # Safety
/// `output`, `input`, `r` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn phase_gate_apply(
    output: *mut f32, input: *const f32,
    r: *const f32, gain: f32, threshold: f32, n: usize,
) {
    let gain_s = f32x8::splat(gain);
    let th_s = f32x8::splat(threshold);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let inp = load8(input, off);
        let rr = load8(r, off);
        store8(output, off, inp * simd_sigmoid(gain_s * (rr - th_s)));
    }
    if n8 < n {
        let out = std::slice::from_raw_parts_mut(output, n);
        let inp = std::slice::from_raw_parts(input, n);
        let rr = std::slice::from_raw_parts(r, n);
        for i in n8..n {
            let gate = 1.0 / (1.0 + (-gain * (rr[i] - threshold)).exp());
            out[i] = inp[i] * gate;
        }
    }
}

/// T3 drive encoder: output = gain × input.
///
/// # Safety
/// `drives`, `inputs` must be valid for `n` elements.
pub(crate) unsafe fn t3_drive_encoder(drives: *mut f32, inputs: *const f32, gain: f32, n: usize) {
    let g = f32x8::splat(gain);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        store8(drives, off, g * load8(inputs, off));
    }
    if n8 < n {
        let d = std::slice::from_raw_parts_mut(drives, n);
        let inp = std::slice::from_raw_parts(inputs, n);
        for i in n8..n { d[i] = gain * inp[i]; }
    }
}

/// T3 TACT generator: pulse when all 3 phases exceed threshold.
/// Comparison per-element (simple conditional).
///
/// # Safety
/// `pulses`, `ph1`, `ph2`, `ph3` must be valid for `n` elements.
pub(crate) unsafe fn t3_tact_generator(
    pulses: *mut f32, ph1: *const f32, ph2: *const f32,
    ph3: *const f32, threshold: f32, n: usize,
) {
    let th = f32x8::splat(threshold);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let a = load8(ph1, off);
        let b = load8(ph2, off);
        let c = load8(ph3, off);
        let mask = a.cmp_gt(th) & b.cmp_gt(th) & c.cmp_gt(th);
        store8(pulses, off, mask.blend(one, zero));
    }
    if n8 < n {
        let p = std::slice::from_raw_parts_mut(pulses, n);
        let a = std::slice::from_raw_parts(ph1, n);
        let b = std::slice::from_raw_parts(ph2, n);
        let c = std::slice::from_raw_parts(ph3, n);
        for i in n8..n {
            p[i] = if a[i] > threshold && b[i] > threshold && c[i] > threshold { 1.0 } else { 0.0 };
        }
    }
}

/// Echo state tap: read one delay line from oscillator buffer.
/// Memory-bound — scalar is fine.
///
/// # Safety
/// `output` must be valid for `n_osc` elements; `buffer` for `n_osc·delay_len`.
pub(crate) unsafe fn echo_tap(
    output: *mut f32, buffer: *const f32,
    n_osc: usize, delay_len: usize, delay_idx: usize,
) {
    let out = std::slice::from_raw_parts_mut(output, n_osc);
    let buf = std::slice::from_raw_parts(buffer, n_osc * delay_len);
    let idx = delay_idx % delay_len;
    for i in 0..n_osc {
        out[i] = buf[i * delay_len + idx];
    }
}

/// T3 inverse Kuramoto: estimate drive from amplitude changes.
///
/// Linearized inverse of Stuart-Landau amplitude dynamics:
///   r' = r + dt · ((α − r²)·r + gain·drive − decay·r)
///
/// Simplified for linear regime (small dt), dropping the (α − r²)·r term:
///   drive ≈ gain · (r' − r)/dt − drive_old · decay
///
/// The Stuart-Landau `alpha` parameter is accepted for API compatibility
/// but not used in the linearized formula — the `(α − r²)·r` term is
/// assumed negligible near steady-state amplitudes.
///
/// # Safety
/// `drive_out`, `r`, `r_prev` must be valid for `n` elements.
pub(crate) unsafe fn t3_inverse_kuramoto(
    drive_out: *mut f32, r: *const f32, r_prev: *const f32,
    _alpha: f32, gain: f32, eps: f32, decay: f32,
    dt: f32, n: usize,
) {
    let dt_inv = f32x8::splat(1.0 / dt.max(eps));
    let gain_s = f32x8::splat(gain);
    let decay_s = f32x8::splat(decay);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let rr = load8(r, off);
        let rp = load8(r_prev, off);
        let mut dd = load8(drive_out, off);
        // r_dot = (r - r_prev) / dt  (finite difference, corrected from alpha)
        let rdot = (rr - rp) * dt_inv;
        // drive = gain · r_dot − drive · decay  (linearized inverse)
        dd = gain_s * rdot - dd * decay_s;
        store8(drive_out, off, dd);
    }
    if n8 < n {
        let d = std::slice::from_raw_parts_mut(drive_out, n);
        let rr = std::slice::from_raw_parts(r, n);
        let rp = std::slice::from_raw_parts(r_prev, n);
        let dt_inv_s = 1.0 / dt.max(eps);
        for i in n8..n {
            let rdot = (rr[i] - rp[i]) * dt_inv_s;
            d[i] = gain * rdot - d[i] * decay;
        }
    }
}

/// SIMD ReLU: x = max(0, x) для всех элементов. Использует f32x8.
///
/// # Safety
/// `data` must be non-null, aligned, valid for `n` elements.
pub(crate) unsafe fn relu_simd(data: *mut f32, n: usize) {
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let v = load8_mut(data, off).max(zero);
        store8(data, off, v);
    }
    if n8 < n {
        let s = std::slice::from_raw_parts_mut(data, n);
        for v in s[n8..].iter_mut() { *v = v.max(0.0); }
    }
}

/// SIMD Add: a[i] += b[i] для всех элементов. Использует f32x8.
///
/// # Safety
/// `a` and `b` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn add_simd(a: *mut f32, b: *const f32, n: usize) {
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let va = load8_mut(a, off);
        let vb = load8(b, off);
        store8(a, off, va + vb);
    }
    if n8 < n {
        let sa = std::slice::from_raw_parts_mut(a, n);
        let sb = std::slice::from_raw_parts(b, n);
        for i in n8..n { sa[i] += sb[i]; }
    }
}

/// SIMD SGD step: v = mom·v + (1-mom)·g; w -= lr·v. Использует f32x8.
///
/// Params: velocity, weight, grad — все длины n.
///
/// # Safety
/// `velocity`, `weight`, `grad` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn sgd_step_simd(
    velocity: *mut f32, weight: *mut f32, grad: *const f32,
    lr: f32, momentum: f32, n: usize,
) {
    let lr_s = f32x8::splat(lr);
    let mom_s = f32x8::splat(momentum);
    let one_minus_mom = f32x8::splat(1.0 - momentum);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut v = load8_mut(velocity, off);
        let mut w = load8_mut(weight, off);
        let g = load8(grad, off);
        v = mom_s * v + one_minus_mom * g;
        w -= lr_s * v;
        store8(velocity, off, v);
        store8(weight, off, w);
    }
    if n8 < n {
        let vel = std::slice::from_raw_parts_mut(velocity, n);
        let wgt = std::slice::from_raw_parts_mut(weight, n);
        let grd = std::slice::from_raw_parts(grad, n);
        for i in n8..n {
            vel[i] = momentum * vel[i] + (1.0 - momentum) * grd[i];
            wgt[i] -= lr * vel[i];
        }
    }
}

/// SIMD Adam step: m = β1·m + (1-β1)·g; v = β2·v + (1-β2)·g²; w -= lr·m/(√v + ε).
///
/// # Safety
/// `weight`, `m`, `v`, `grad` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn adam_step_simd(
    weight: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    beta1_t_corr: f32, beta2_t_corr: f32, n: usize,
) {
    let lr_s = f32x8::splat(lr);
    let b1_s = f32x8::splat(beta1);
    let b2_s = f32x8::splat(beta2);
    let one_mb1 = f32x8::splat(1.0 - beta1);
    let one_mb2 = f32x8::splat(1.0 - beta2);
    let eps_s = f32x8::splat(eps);
    let b1c_s = f32x8::splat(beta1_t_corr);
    let b2c_s = f32x8::splat(beta2_t_corr);
    let _zero = f32x8::splat(0.0);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weight, off);
        let mut mm = load8_mut(m, off);
        let mut vv = load8_mut(v, off);
        let g = load8(grad, off);

        mm = b1_s * mm + one_mb1 * g;
        vv = b2_s * vv + one_mb2 * g * g;

        let m_hat = mm / b1c_s;
        let v_hat = vv / b2c_s;
        w -= lr_s * m_hat / (v_hat.sqrt() + eps_s);

        store8(weight, off, w);
        store8(m, off, mm);
        store8(v, off, vv);
    }
    if n8 < n {
        let wgt = std::slice::from_raw_parts_mut(weight, n);
        let mm = std::slice::from_raw_parts_mut(m, n);
        let vv = std::slice::from_raw_parts_mut(v, n);
        let grd = std::slice::from_raw_parts(grad, n);
        for i in n8..n {
            mm[i] = beta1 * mm[i] + (1.0 - beta1) * grd[i];
            vv[i] = beta2 * vv[i] + (1.0 - beta2) * grd[i] * grd[i];
            let m_hat = mm[i] / beta1_t_corr;
            let v_hat = vv[i] / beta2_t_corr;
            wgt[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Hebbian update: w += lr·cos_diff − decay·w·dt.
///
/// # Safety
/// `weights`, `cos_diff` must be valid for `n` elements, non-overlapping.
pub(crate) unsafe fn hebbian_update(
    weights: *mut f32, cos_diff: *const f32,
    lr: f32, decay: f32, dt: f32, n: usize,
) {
    let lr_s = f32x8::splat(lr);
    let decay_dt = f32x8::splat(decay * dt);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut w = load8(weights, off);
        let cd = load8(cos_diff, off);
        w = w + lr_s * cd - decay_dt * w;
        store8(weights, off, w);
    }
    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let cd = std::slice::from_raw_parts(cos_diff, n);
        for i in n8..n { w[i] += lr * cd[i] - decay * w[i] * dt; }
    }
}
