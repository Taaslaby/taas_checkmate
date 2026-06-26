//! Pure Rust SIMD decoders — `wide::f32x8`.
//! Replaces all ASM decoder kernels with portable SIMD.

use wide::f32x8;

#[inline(always)]
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
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

/// Complex decoder: out = W·[A·cos(θ)] + bias.
/// Inner dot product uses f32x8 SIMD.
///
/// # Safety
/// Pointers must be valid: `outputs[n_out]`, `amplitudes[n_in]`, `cos_buf[n_in]`,
/// `weights[n_in·n_out]`, `biases[n_out]`.
pub(crate) unsafe fn t3_complex_decoder(
    outputs: *mut f32, amplitudes: *const f32, cos_buf: *const f32,
    weights: *const f32, biases: *const f32, n_in: usize, n_out: usize,
) {
    let out = std::slice::from_raw_parts_mut(outputs, n_out);
    let a = std::slice::from_raw_parts(amplitudes, n_in);
    let c = std::slice::from_raw_parts(cos_buf, n_in);
    let w = std::slice::from_raw_parts(weights, n_in * n_out);
    let b = std::slice::from_raw_parts(biases, n_out);
    let n8 = n_in & !7;

    for o in 0..n_out {
        let mut sum = b[o];
        let row = o * n_in;
        let mut vdot = f32x8::splat(0.0);
        let mut i = 0;
        while i < n8 {
            vdot = vdot + load8(amplitudes, i) * load8(cos_buf, i) * load8(weights, row + i);
            i += 8;
        }
        let arr: [f32; 8] = vdot.to_array();
        sum += arr.iter().sum::<f32>();
        for j in i..n_in { sum += a[j] * c[j] * w[row + j]; }
        out[o] = sum;
    }
}

/// Pairwise PST features: sin(θ₁−θ₂), cos(θ₁−θ₂).
///
/// # Safety
/// `out_sin`, `out_cos` must be valid for `n1·n2` elements; `ph1[n1]`, `ph2[n2]`.
pub(crate) unsafe fn t3_pst_features(
    out_sin: *mut f32, out_cos: *mut f32,
    ph1: *const f32, ph2: *const f32, n1: usize, n2: usize,
) {
    let os = std::slice::from_raw_parts_mut(out_sin, n1 * n2);
    let oc = std::slice::from_raw_parts_mut(out_cos, n1 * n2);
    let p1 = std::slice::from_raw_parts(ph1, n1);
    let p2 = std::slice::from_raw_parts(ph2, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            let d = p1[i] - p2[j];
            os[i * n2 + j] = d.sin();
            oc[i * n2 + j] = d.cos();
        }
    }
}

/// LMS: w += lr·error·states; bias += lr·error.
///
/// # Safety
/// `weights[n]`, `states[n]` must be valid; `bias` must be non-null.
pub(crate) unsafe fn t3_lms_train(
    weights: *mut f32, bias: *mut f32,
    states: *const f32, error: f32, lr: f32, n: usize,
) {
    let lr_e = f32x8::splat(lr * error);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        store8(weights, off, load8(weights, off) + lr_e * load8(states, off));
    }
    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let s = std::slice::from_raw_parts(states, n);
        for i in n8..n { w[i] += lr * error * s[i]; }
    }
    *bias += lr * error;
}

/// NLMS: w += lr·error·states / (||states||² + eps).
///
/// # Safety
/// `weights[n]`, `states[n]` must be valid; `bias` must be non-null.
pub(crate) unsafe fn nlms_train(
    weights: *mut f32, bias: *mut f32,
    states: *const f32, error: f32, lr: f32, eps: f32, n: usize,
) {
    // Compute norm² via SIMD reduction
    let n8 = n & !7;
    let mut ns = f32x8::splat(0.0);
    let mut i = 0;
    while i < n8 { ns = ns + load8(states, i) * load8(states, i); i += 8; }
    let arr: [f32; 8] = ns.to_array();
    let mut norm_sq: f32 = arr.iter().sum();
    for j in i..n { let s = *states.add(j); norm_sq += s * s; }

    let mu = f32x8::splat(lr * error / (norm_sq + eps));
    let mut i = 0;
    while i < n8 { store8(weights, i, load8(weights, i) + mu * load8(states, i)); i += 8; }
    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let s = std::slice::from_raw_parts(states, n);
        for j in n8..n { w[j] += lr * error * s[j] / (norm_sq + eps); }
    }
    *bias += lr * error / (norm_sq + eps);
}

/// Kalman filter update: w += K·error, cov -= K·states·cov.
///
/// # Important: `s` must be the innovation covariance
///
/// ```text
/// s = R + Σᵢ(cov[i] · states[i]²)
/// ```
///
/// where `R` is the measurement noise covariance and `Σ(cov·states²)` is the
/// projection of the state covariance onto the measurement space.
///
/// If `s` is incorrect (e.g. just a constant), the Kalman gain will be wrong:
///   K[i] = cov[i] · states[i] / s
///   w[i] += K[i] · error
///   cov[i] -= K[i] · states[i] · cov[i]
///
/// `s` is NOT computed inside this kernel because it requires a global reduction
/// over ALL elements — the kernel operates on independent SIMD chunks.
/// Compute `s` in the caller, then pass it here.
///
/// # Safety
/// `weights[n]`, `cov[n]`, `states[n]` must be valid, non-overlapping.
pub(crate) unsafe fn kalman_update(
    weights: *mut f32, cov: *mut f32,
    states: *const f32, error: f32, s: f32, n: usize,
) {
    debug_assert!(s > 0.0, "kalman_update: s = R + Σ(cov·states²) must be > 0, got {}", s);
    let s2 = f32x8::splat(s + 1e-8);
    let err_s = f32x8::splat(error);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut w = load8(weights, off);
        let mut c = load8(cov, off);
        let st = load8(states, off);
        let k = (c * st) / (st * st * c + s2);
        w = w + k * err_s;
        c = (c - k * st * c).max(f32x8::splat(1e-12));
        store8(weights, off, w);
        store8(cov, off, c);
    }
    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let c = std::slice::from_raw_parts_mut(cov, n);
        let st = std::slice::from_raw_parts(states, n);
        let s2 = s + 1e-8;
        for i in n8..n {
            let k = c[i] * st[i] / (st[i] * st[i] * c[i] + s2);
            w[i] += k * error;
            c[i] -= k * st[i] * c[i];
            c[i] = c[i].max(1e-12);
        }
    }
}

/// Adam optimizer: w -= lr·m̂/(√v̂ + eps).
///
/// # Safety
/// `weights[n]`, `m[n]`, `v[n]`, `states[n]` must be valid, non-overlapping.
pub(crate) unsafe fn adam_train(
    weights: *mut f32, m: *mut f32, v: *mut f32,
    states: *const f32, error: f32, lr: f32,
    beta1: f32, beta2: f32, bc1: f32, _bc2: f32, eps: f32, n: usize,
) {
    let b1 = f32x8::splat(beta1);
    let b1_c = f32x8::splat(1.0 - beta1);
    let b2 = f32x8::splat(beta2);
    let b2_c = f32x8::splat(1.0 - beta2);
    let lr_s = f32x8::splat(lr / bc1);
    let eps_s = f32x8::splat(eps);
    let err_s = f32x8::splat(error);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut mm = load8(m, off);
        let mut vv = load8(v, off);
        let st = load8(states, off);
        let g = err_s * st;                     // g = error·states
        mm = b1 * mm + b1_c * g;                // m = β₁·m + (1-β₁)·g
        vv = b2 * vv + b2_c * g * g;            // v = β₂·v + (1-β₂)·g²
        let mut w = load8(weights, off);
        w = w - lr_s * mm / (vv.sqrt() + eps_s); // w -= lr·m̂/(√v̂ + ε)
        store8(weights, off, w);
        store8(m, off, mm);
        store8(v, off, vv);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let mm = std::slice::from_raw_parts_mut(m, n);
        let vv = std::slice::from_raw_parts_mut(v, n);
        let s = std::slice::from_raw_parts(states, n);
        for i in n8..n {
            let g = error * s[i];
            mm[i] = beta1 * mm[i] + (1.0 - beta1) * g;
            vv[i] = beta2 * vv[i] + (1.0 - beta2) * g * g;
            w[i] -= lr * mm[i] / (vv[i].sqrt() + eps) / bc1;
        }
    }
}

/// PST decoder: weighted circular buffer readout (SIMD dot per feature).
///
/// # Safety
/// `output[feat_dim]`, `buffer[l·feat_dim]`, `weights[l·feat_dim]` must be valid.
pub(crate) unsafe fn pst_decode(
    output: *mut f32, buffer: *const f32,
    weights: *const f32, feat_dim: usize, l: usize, pos: usize,
) {
    let out = std::slice::from_raw_parts_mut(output, feat_dim);
    let buf = std::slice::from_raw_parts(buffer, l * feat_dim);
    let w = std::slice::from_raw_parts(weights, l * feat_dim);
    out.fill(0.0);

    let n8 = feat_dim & !7;
    for t in 0..l {
        let src = (pos + t) % l;
        let b_off = src * feat_dim;
        let w_off = t * feat_dim;

        // SIMD accumulation
        let mut i = 0;
        while i < n8 {
            let bv = load8(buffer, b_off + i);
            let wv = load8(weights, w_off + i);
            let arr: [f32; 8] = (bv * wv).to_array();
            for j in 0..8 { out[i + j] += arr[j]; }
            i += 8;
        }
        for f in i..feat_dim { out[f] += w[w_off + f] * buf[b_off + f]; }
    }
}

/// Ring synthesis: out_i = w_i·a_i·trig_i.
///
/// # Safety
/// `output[n]`, `amplitudes[n]`, `trig_buf[n]`, `weights[n]` must be valid, non-overlapping.
pub(crate) unsafe fn ring_synthesis(
    output: *mut f32, amplitudes: *const f32,
    trig_buf: *const f32, weights: *const f32, n: usize,
) {
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        store8(output, off, load8(weights, off) * load8(amplitudes, off) * load8(trig_buf, off));
    }
    if n8 < n {
        let out = std::slice::from_raw_parts_mut(output, n);
        let a = std::slice::from_raw_parts(amplitudes, n);
        let t = std::slice::from_raw_parts(trig_buf, n);
        let w = std::slice::from_raw_parts(weights, n);
        for i in n8..n { out[i] = w[i] * a[i] * t[i]; }
    }
}
