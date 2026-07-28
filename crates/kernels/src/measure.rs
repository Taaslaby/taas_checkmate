//! Pure Rust SIMD measurement functions — `wide::f32x8`.
//! Replaces all ASM measure kernels with portable SIMD.

use crate::neuron::{load8, store8};
use core::f32::consts::PI;
use wide::{f32x8, CmpGt};

/// Pairwise interference: (1 + cos(φ_a − φ_b)) / 2.
/// Trig per-element.
///
/// # Safety
/// `output[na·nb]`, `phases_a[na]`, `phases_b[nb]` must be valid.
pub(crate) unsafe fn interference(
    output: *mut f32,
    phases_a: *const f32,
    phases_b: *const f32,
    na: usize,
    nb: usize,
) {
    let out = std::slice::from_raw_parts_mut(output, na * nb);
    let a = std::slice::from_raw_parts(phases_a, na);
    let b = std::slice::from_raw_parts(phases_b, nb);
    for i in 0..na {
        for j in 0..nb {
            out[i * nb + j] = (1.0 + (a[i] - b[j]).cos()) * 0.5;
        }
    }
}

/// Mean interference: average of (1 + cos(φ_a − φ_b)) / 2.
/// Uses f32x8 SIMD reduction.
///
/// # Safety
/// `phases_a[n]`, `phases_b[n]` must be valid; `output` non-null.
pub(crate) unsafe fn interference_mean(
    output: *mut f32,
    phases_a: *const f32,
    phases_b: *const f32,
    n: usize,
) {
    let n8 = n & !7;
    let mut sum = f32x8::splat(0.0);
    for off in (0..n8).step_by(8) {
        let a = load8(phases_a, off);
        let b = load8(phases_b, off);
        // compute cos per element (no SIMD trig)
        let d = a - b;
        let da: [f32; 8] = d.to_array();
        let mut cv = [0.0f32; 8];
        for j in 0..8 {
            cv[j] = da[j].cos();
        }
        sum += f32x8::from(cv);
    }
    let arr: [f32; 8] = sum.to_array();
    let mut total: f32 = arr.iter().sum();
    if n8 < n {
        let a = std::slice::from_raw_parts(phases_a, n);
        let b = std::slice::from_raw_parts(phases_b, n);
        for i in n8..n {
            total += (a[i] - b[i]).cos();
        }
    }
    *output = (1.0 + total / n as f32) * 0.5;
}

/// Order parameter: |∑ a·e^{iφ}| / ∑ a.
/// Uses f32x8 SIMD reduction.
///
/// # Safety
/// `sin_buf[n]`, `cos_buf[n]`, `amplitudes[n]` valid; `output` non-null.
pub(crate) unsafe fn phase_coherence(
    output: *mut f32,
    sin_buf: *const f32,
    cos_buf: *const f32,
    amplitudes: *const f32,
    n: usize,
) {
    let n8 = n & !7;
    let (mut ss, mut cc, mut sa) = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
    for off in (0..n8).step_by(8) {
        let si = load8(sin_buf, off);
        let co = load8(cos_buf, off);
        let a = load8(amplitudes, off);
        ss += a * si;
        cc += a * co;
        sa += a;
    }
    let ss_a: [f32; 8] = ss.to_array();
    let cc_a: [f32; 8] = cc.to_array();
    let sa_a: [f32; 8] = sa.to_array();
    let mut sum_s: f32 = ss_a.iter().sum();
    let mut sum_c: f32 = cc_a.iter().sum();
    let mut sum_a: f32 = sa_a.iter().sum();

    if n8 < n {
        let si = std::slice::from_raw_parts(sin_buf, n);
        let co = std::slice::from_raw_parts(cos_buf, n);
        let a = std::slice::from_raw_parts(amplitudes, n);
        for i in n8..n {
            sum_s += a[i] * si[i];
            sum_c += a[i] * co[i];
            sum_a += a[i];
        }
    }
    *output = (sum_s * sum_s + sum_c * sum_c).sqrt() / sum_a.max(1e-12);
}

/// Band energy: accumulate amplitude energy into frequency bands.
/// Branchy per-element — scalar is fine.
///
/// # Safety
/// `band_energies[n_bands]`, `amplitudes[n]`, `omegas[n]`, `band_edges[n_bands+1]` valid.
pub(crate) unsafe fn band_energy(
    band_energies: *mut f32,
    amplitudes: *const f32,
    omegas: *const f32,
    band_edges: *const f32,
    n_bands: usize,
    n: usize,
) {
    let be = std::slice::from_raw_parts_mut(band_energies, n_bands);
    let a = std::slice::from_raw_parts(amplitudes, n);
    let o = std::slice::from_raw_parts(omegas, n);
    let edges = std::slice::from_raw_parts(band_edges, n_bands + 1);
    be.fill(0.0);
    for i in 0..n {
        let freq = o[i].abs();
        for b in 0..n_bands {
            if freq >= edges[b] && freq < edges[b + 1] {
                be[b] += a[i];
                break;
            }
        }
    }
}

/// Normalized winding number: atan2(sin Δφ, cos Δφ) / π.
///
/// # Safety
/// `output[na·nb]`, `phases_a[na]`, `phases_b[nb]` must be valid.
pub(crate) unsafe fn winding_matrix(
    output: *mut f32,
    phases_a: *const f32,
    phases_b: *const f32,
    na: usize,
    nb: usize,
) {
    let out = std::slice::from_raw_parts_mut(output, na * nb);
    let a = std::slice::from_raw_parts(phases_a, na);
    let b = std::slice::from_raw_parts(phases_b, nb);
    for i in 0..na {
        for j in 0..nb {
            let d = a[i] - b[j];
            out[i * nb + j] = d.sin().atan2(d.cos()) / PI;
        }
    }
}

/// Lazy sin/cos: recompute only when phase changes > ε.
/// Comparison per-element — scalar is fine (conditional).
///
/// # Safety
/// `sin_out[n]`, `cos_out[n]`, `phases[n]`, `phases_prev[n]` valid, non-overlapping.
pub(crate) unsafe fn lazy_sincos_update(
    sin_out: *mut f32,
    cos_out: *mut f32,
    phases: *const f32,
    phases_prev: *mut f32,
    eps: f32,
    n: usize,
) {
    let eps_s = f32x8::splat(eps);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let p = load8(phases, off);
        let pp = load8(phases_prev, off);
        let diff = (p - pp).abs();
        let mask = diff.cmp_gt(eps_s);
        // Compute sin/cos per element where mask is true
        let p_arr: [f32; 8] = p.to_array();
        let mask_bits = mask.to_array().map(|x| x.to_bits());
        let mut si = [0.0f32; 8];
        let mut co = [0.0f32; 8];
        for j in 0..8 {
            if mask_bits[j] != 0 {
                si[j] = p_arr[j].sin();
                co[j] = p_arr[j].cos();
            }
        }
        // For unchanged lanes, keep old values
        let old_si = load8(sin_out, off);
        let old_co = load8(cos_out, off);
        let new_si = f32x8::from(si);
        let new_co = f32x8::from(co);
        store8(sin_out, off, mask.blend(new_si, old_si));
        store8(cos_out, off, mask.blend(new_co, old_co));
        // Update prev where changed
        store8(phases_prev, off, mask.blend(p, pp));
    }
    if n8 < n {
        let si = std::slice::from_raw_parts_mut(sin_out, n);
        let co = std::slice::from_raw_parts_mut(cos_out, n);
        let p = std::slice::from_raw_parts(phases, n);
        let pp = std::slice::from_raw_parts_mut(phases_prev, n);
        for i in n8..n {
            if (p[i] - pp[i]).abs() > eps {
                si[i] = p[i].sin();
                co[i] = p[i].cos();
                pp[i] = p[i];
            }
        }
    }
}

/// Spike entropy: H = -Σ p_i · log₂(p_i) where p_i = spike_counts[i] / total_steps.
///
/// Also computes H_penalty = |H - H_target| — отклонение от целевой энтропии.
/// Уточнение пользователя: штраф как |H - H_target|, а не абсолютная H,
/// чтобы избежать dead neuron problem (все p→0, H→0, штрафа нет).
///
/// # Safety
/// `spike_counts[n]` valid; `h_output`, `h_penalty` non-null.
pub(crate) unsafe fn spike_entropy(
    h_output: *mut f32,
    h_penalty: *mut f32,
    spike_counts: *const f32,
    n: usize,
    total_steps: f32,
    h_target: f32,
) {
    let eps = 1e-10f32;
    let inv_steps = 1.0 / total_steps.max(1.0);
    let log2e = core::f32::consts::LOG2_E;

    let n8 = n & !7;
    let mut h_sum_simd = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        let counts = load8(spike_counts, off);
        let p = counts * f32x8::splat(inv_steps);
        let p_arr: [f32; 8] = p.to_array();
        let mut hc = [0.0f32; 8];
        for (j, &pj) in p_arr.iter().enumerate() {
            if pj > 0.0 {
                hc[j] = pj * pj.max(eps).ln() * log2e;
            }
        }
        h_sum_simd -= f32x8::from(hc);
    }

    let arr: [f32; 8] = h_sum_simd.to_array();
    let mut total_h: f32 = arr.iter().sum();

    if n8 < n {
        let counts = std::slice::from_raw_parts(spike_counts, n);
        for i in n8..n {
            let p = counts[i] * inv_steps;
            if p > 0.0 {
                total_h -= p * p.max(eps).ln() * log2e;
            }
        }
    }

    *h_output = total_h.max(0.0);
    *h_penalty = (*h_output - h_target).abs();
}

/// All-pairs coherence between groups of oscillators.
/// Uses f32x8 SIMD for reduction within each area.
///
/// # Safety
/// `output[n_areas²]`, `areas[n_areas·n_elem]`, `areas_sin[n_areas·n_elem]`,
/// `areas_cos[n_areas·n_elem]` must be valid.
pub(crate) unsafe fn batch_coherence_sweep(
    output: *mut f32,
    areas: *const f32,
    areas_sin: *const f32,
    areas_cos: *const f32,
    n_areas: usize,
    n_elem: usize,
) {
    let out = std::slice::from_raw_parts_mut(output, n_areas * n_areas);
    let areas_data = std::slice::from_raw_parts(areas, n_areas * n_elem);
    let sin_data = std::slice::from_raw_parts(areas_sin, n_areas * n_elem);
    let cos_data = std::slice::from_raw_parts(areas_cos, n_areas * n_elem);
    let n8 = n_elem & !7;

    for i in 0..n_areas {
        // Sum over area i using SIMD
        let (mut ss_i, mut cc_i) = (f32x8::splat(0.0), f32x8::splat(0.0));
        let base_i = i * n_elem;
        let mut k = 0;
        while k < n8 {
            let a = load8(areas, base_i + k);
            let s = load8(areas_sin, base_i + k);
            let c = load8(areas_cos, base_i + k);
            ss_i += a * s;
            cc_i += a * c;
            k += 8;
        }
        let mut sum_si: f32 = ss_i.to_array().iter().sum();
        let mut sum_ci: f32 = cc_i.to_array().iter().sum();
        for k2 in k..n_elem {
            let idx = base_i + k2;
            sum_si += areas_data[idx] * sin_data[idx];
            sum_ci += areas_data[idx] * cos_data[idx];
        }
        let norm_i = (sum_si * sum_si + sum_ci * sum_ci).sqrt().max(1e-12);

        for j in 0..n_areas {
            let (mut ss_j, mut cc_j) = (f32x8::splat(0.0), f32x8::splat(0.0));
            let base_j = j * n_elem;
            let mut k = 0;
            while k < n8 {
                let a = load8(areas, base_j + k);
                let s = load8(areas_sin, base_j + k);
                let c = load8(areas_cos, base_j + k);
                ss_j += a * s;
                cc_j += a * c;
                k += 8;
            }
            let mut sum_sj: f32 = ss_j.to_array().iter().sum();
            let mut sum_cj: f32 = cc_j.to_array().iter().sum();
            for k2 in k..n_elem {
                let idx = base_j + k2;
                sum_sj += areas_data[idx] * sin_data[idx];
                sum_cj += areas_data[idx] * cos_data[idx];
            }
            let norm_j = (sum_sj * sum_sj + sum_cj * sum_cj).sqrt().max(1e-12);
            out[i * n_areas + j] = (sum_si * sum_sj + sum_ci * sum_cj) / (norm_i * norm_j);
        }
    }
}
