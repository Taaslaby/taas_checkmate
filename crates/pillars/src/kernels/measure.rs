//! # 📏 Measurement Microkernels — Coherence, Interference, Entropy
//!
//! `f32x8` kernels for network-level measurement. All kernels operate on
//! flat `&[f32]` slices with SIMD inner loops + scalar tails.
//!
//! ## Functions
//!
//! | Kernel | Formula | Use |
//! |--------|---------|-----|
//! | [`interference`] | `(1+cos(φₐ−φᵦ))/2` | Pairwise phase similarity |
//! | [`interference_mean`] | `(1+mean(cos(φₐ−φᵦ)))/2` | Global phase coherence |
//! | [`phase_coherence`] | `|Σ a·e^{iφ}| / Σ a` | Kuramoto order parameter R |
//! | [`band_energy`] | `Σ a·[|ω| ∈ band]` | Spectral decomposition |
//! | [`winding_matrix`] | `atan2(sinΔφ, cosΔφ)/π` | Normalized phase winding |
//! | [`lazy_sincos_update`] | sin/cos recompute on change > ε | Cheap envelope tracking |
//! | [`spike_entropy`] | `−Σ p·log₂(p)` | Firing pattern diversity |
//! | [`batch_coherence_sweep`] | Cosine similarity of complex mean-field | Pairwise area coherence |

use wide::f32x8;
use wide::CmpGt;

use crate::core::util::{load8, store8};
use core::f32::consts::PI;

/// Pairwise interference matrix: `out[i,j] = (1 + cos(φ_a[i] − φ_b[j])) / 2`.
pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(
        output.len(),
        na * nb,
        "interference: output length mismatch"
    );
    for i in 0..na {
        for j in 0..nb {
            output[i * nb + j] = (1.0 + (phases_a[i] - phases_b[j]).cos()) * 0.5;
        }
    }
}

/// Mean interference: average of `(1 + cos(φ_a − φ_b)) / 2`. SIMD reduction.
pub fn interference_mean(output: &mut f32, phases_a: &[f32], phases_b: &[f32]) {
    let n = phases_a.len();
    assert_eq!(
        phases_b.len(),
        n,
        "interference_mean: phases_b length mismatch"
    );
    if n == 0 {
        *output = 0.0;
        return;
    }
    let n8 = n & !7;
    let mut sum = f32x8::splat(0.0);
    for off in (0..n8).step_by(8) {
        unsafe {
            let a = load8(phases_a.as_ptr(), off);
            let b = load8(phases_b.as_ptr(), off);
            let d = a - b;
            let da: [f32; 8] = d.to_array();
            let mut cv = [0.0f32; 8];
            for j in 0..8 {
                cv[j] = da[j].cos();
            }
            sum += f32x8::from(cv);
        }
    }
    let mut total: f32 = sum.to_array().iter().sum();
    for i in n8..n {
        total += (phases_a[i] - phases_b[i]).cos();
    }
    *output = (1.0 + total / n as f32) * 0.5;
}

/// Kuramoto order parameter: `|Σ a·e^{iφ}| / Σ a`. SIMD reduction.
pub fn phase_coherence(output: &mut f32, sin_buf: &[f32], cos_buf: &[f32], amplitudes: &[f32]) {
    let n = sin_buf.len();
    assert_eq!(cos_buf.len(), n, "phase_coherence: cos_buf length mismatch");
    assert_eq!(
        amplitudes.len(),
        n,
        "phase_coherence: amplitudes length mismatch"
    );
    let n8 = n & !7;
    let (mut ss, mut cc, mut sa) = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
    for off in (0..n8).step_by(8) {
        unsafe {
            let si = load8(sin_buf.as_ptr(), off);
            let co = load8(cos_buf.as_ptr(), off);
            let a = load8(amplitudes.as_ptr(), off);
            ss += a * si;
            cc += a * co;
            sa += a;
        }
    }
    let mut sum_s: f32 = ss.to_array().iter().sum();
    let mut sum_c: f32 = cc.to_array().iter().sum();
    let mut sum_a: f32 = sa.to_array().iter().sum();

    for i in n8..n {
        sum_s += amplitudes[i] * sin_buf[i];
        sum_c += amplitudes[i] * cos_buf[i];
        sum_a += amplitudes[i];
    }
    *output = (sum_s * sum_s + sum_c * sum_c).sqrt() / sum_a.max(1e-12);
}

/// Band energy: accumulate amplitude energy into frequency bands.
///
/// `band_edges` has `n_bands + 1` entries; each oscillator falls into the
/// first band where `edges[b] ≤ |ω| < edges[b+1]`.
pub fn band_energy(
    band_energies: &mut [f32],
    amplitudes: &[f32],
    omegas: &[f32],
    band_edges: &[f32],
) {
    let n_bands = band_energies.len();
    let n = amplitudes.len();
    assert_eq!(omegas.len(), n, "band_energy: omegas length mismatch");
    assert_eq!(
        band_edges.len(),
        n_bands + 1,
        "band_energy: band_edges length mismatch (expected {})",
        n_bands + 1
    );
    band_energies.fill(0.0);
    for i in 0..n {
        let freq = omegas[i].abs();
        for b in 0..n_bands {
            if freq >= band_edges[b] && freq < band_edges[b + 1] {
                band_energies[b] += amplitudes[i];
                break;
            }
        }
    }
}

/// Normalized winding number: `atan2(sin Δφ, cos Δφ) / π`.
pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
    let na = phases_a.len();
    let nb = phases_b.len();
    assert_eq!(
        output.len(),
        na * nb,
        "winding_matrix: output length mismatch"
    );
    for i in 0..na {
        for j in 0..nb {
            let d = phases_a[i] - phases_b[j];
            output[i * nb + j] = d.sin().atan2(d.cos()) / PI;
        }
    }
}

/// Lazy sin/cos: recompute only when phase changed by more than `eps`.
///
/// Unchanged lanes keep their old sin/cos; `phases_prev` is updated in-place.
pub fn lazy_sincos_update(
    sin_out: &mut [f32],
    cos_out: &mut [f32],
    phases: &[f32],
    phases_prev: &mut [f32],
    eps: f32,
) {
    let n = sin_out.len();
    assert_eq!(cos_out.len(), n, "lazy_sincos: cos_out length mismatch");
    assert_eq!(phases.len(), n, "lazy_sincos: phases length mismatch");
    assert_eq!(
        phases_prev.len(),
        n,
        "lazy_sincos: phases_prev length mismatch"
    );
    let eps_s = f32x8::splat(eps);
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        unsafe {
            let p = load8(phases.as_ptr(), off);
            let pp = load8(phases_prev.as_ptr(), off);
            let diff = (p - pp).abs();
            let mask = diff.cmp_gt(eps_s);
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
            let old_si = load8(sin_out.as_ptr(), off);
            let old_co = load8(cos_out.as_ptr(), off);
            let new_si = f32x8::from(si);
            let new_co = f32x8::from(co);
            store8(sin_out.as_mut_ptr(), off, mask.blend(new_si, old_si));
            store8(cos_out.as_mut_ptr(), off, mask.blend(new_co, old_co));
            store8(phases_prev.as_mut_ptr(), off, mask.blend(p, pp));
        }
    }
    for i in n8..n {
        if (phases[i] - phases_prev[i]).abs() > eps {
            sin_out[i] = phases[i].sin();
            cos_out[i] = phases[i].cos();
            phases_prev[i] = phases[i];
        }
    }
}

/// Spike entropy: `H = −Σ p_i·log₂(p_i)` where `p_i = counts[i] / total_steps`.
///
/// Also computes `H_penalty = |H − H_target|` — deviation from the target
/// entropy (avoids the dead-neuron problem: all p→0 gives H→0 but a large
/// penalty against a non-zero target).
pub fn spike_entropy(spike_counts: &[f32], total_steps: f32, h_target: f32) -> (f32, f32) {
    let n = spike_counts.len();
    let eps = 1e-10f32;
    let inv_steps = 1.0 / total_steps.max(1.0);
    let log2e = core::f32::consts::LOG2_E;

    let n8 = n & !7;
    let mut h_sum_simd = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        unsafe {
            let counts = load8(spike_counts.as_ptr(), off);
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
    }

    let mut total_h: f32 = h_sum_simd.to_array().iter().sum();

    for i in n8..n {
        let p = spike_counts[i] * inv_steps;
        if p > 0.0 {
            total_h -= p * p.max(eps).ln() * log2e;
        }
    }

    let h = total_h.max(0.0);
    (h, (h - h_target).abs())
}

/// All-pairs coherence between groups of oscillators.
///
/// `areas`/`areas_sin`/`areas_cos` have layout `[n_areas × n_elem]`.
/// `out[i,j] = (Σ_s·Σ_s + Σ_c·Σ_c) / (‖i‖·‖j‖)` — cosine similarity of
/// complex mean-field vectors.
pub fn batch_coherence_sweep(
    output: &mut [f32],
    areas: &[f32],
    areas_sin: &[f32],
    areas_cos: &[f32],
    n_areas: usize,
    n_elem: usize,
) {
    assert_eq!(
        output.len(),
        n_areas * n_areas,
        "batch_coherence: output length mismatch"
    );
    assert_eq!(
        areas.len(),
        n_areas * n_elem,
        "batch_coherence: areas length mismatch"
    );
    assert_eq!(
        areas_sin.len(),
        n_areas * n_elem,
        "batch_coherence: areas_sin length mismatch"
    );
    assert_eq!(
        areas_cos.len(),
        n_areas * n_elem,
        "batch_coherence: areas_cos length mismatch"
    );
    let n8 = n_elem & !7;

    // Precompute per-area mean-field vectors
    let mut sums_s = vec![0.0f32; n_areas];
    let mut sums_c = vec![0.0f32; n_areas];
    let mut norms = vec![0.0f32; n_areas];
    for i in 0..n_areas {
        let base = i * n_elem;
        let (mut ss, mut cc) = (f32x8::splat(0.0), f32x8::splat(0.0));
        let mut k = 0;
        unsafe {
            while k < n8 {
                let a = load8(areas.as_ptr(), base + k);
                let s = load8(areas_sin.as_ptr(), base + k);
                let c = load8(areas_cos.as_ptr(), base + k);
                ss += a * s;
                cc += a * c;
                k += 8;
            }
        }
        let mut sum_s: f32 = ss.to_array().iter().sum();
        let mut sum_c: f32 = cc.to_array().iter().sum();
        for k2 in k..n_elem {
            let idx = base + k2;
            sum_s += areas[idx] * areas_sin[idx];
            sum_c += areas[idx] * areas_cos[idx];
        }
        sums_s[i] = sum_s;
        sums_c[i] = sum_c;
        norms[i] = (sum_s * sum_s + sum_c * sum_c).sqrt().max(1e-12);
    }

    for i in 0..n_areas {
        for j in 0..n_areas {
            output[i * n_areas + j] =
                (sums_s[i] * sums_s[j] + sums_c[i] * sums_c[j]) / (norms[i] * norms[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interference_mean_empty_is_zero() {
        let mut output = f32::NAN;
        interference_mean(&mut output, &[], &[]);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn interference_and_coherence_basic() {
        let n = 8;
        let a = vec![0.0f32; n]; // all in-phase
        let b = vec![0.0f32; n];
        let mut out = vec![0.0f32; n * n];
        interference(&mut out, &a, &b);
        assert!(
            (out[0] - 1.0).abs() < 1e-5,
            "in-phase interference should be 1"
        );

        let sin_buf = vec![0.0f32; n];
        let cos_buf = vec![1.0f32; n];
        let amps = vec![1.0f32; n];
        let mut coherence = 0.0;
        phase_coherence(&mut coherence, &sin_buf, &cos_buf, &amps);
        assert!(
            (coherence - 1.0).abs() < 1e-4,
            "in-phase coherence should be 1"
        );
    }

    #[test]
    fn band_energy_basic() {
        let amplitudes = vec![1.0f32; 6];
        let omegas = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
        let edges = vec![0.0, 2.0, 4.0, 6.0];
        let mut bands = vec![0.0f32; 3];
        band_energy(&mut bands, &amplitudes, &omegas, &edges);
        assert_eq!(bands, vec![2.0, 2.0, 2.0], "each band gets 2 oscillators");
    }

    #[test]
    fn winding_and_lazy_sincos_smoke() {
        let n = 4;
        let a = vec![0.0f32; n];
        let b = vec![0.0f32; n];
        let mut w = vec![0.0f32; n * n];
        winding_matrix(&mut w, &a, &b);
        assert!((w[0] - 0.0).abs() < 1e-5, "zero phase diff → winding 0");

        let mut sin_out = vec![0.0f32; n];
        let mut cos_out = vec![0.0f32; n];
        let phases = vec![0.5f32; n];
        let mut prev = vec![0.0f32; n];
        lazy_sincos_update(&mut sin_out, &mut cos_out, &phases, &mut prev, 0.01);
        assert!(
            (sin_out[0] - 0.5f32.sin()).abs() < 1e-5,
            "lazy sin should recompute on change"
        );
        // No change → values stay
        let phases2 = vec![0.5f32; n];
        lazy_sincos_update(&mut sin_out, &mut cos_out, &phases2, &mut prev, 0.01);
        assert!(
            (sin_out[0] - 0.5f32.sin()).abs() < 1e-5,
            "no change → sin preserved"
        );
    }

    #[test]
    fn spike_entropy_basic() {
        // Uniform distribution over 4 neurons → H = log2(4) = 2.0
        let counts = vec![1.0f32; 4];
        let (h, penalty) = spike_entropy(&counts, 4.0, 2.0);
        assert!((h - 2.0).abs() < 1e-4, "uniform H should be 2, got {h}");
        assert!(
            penalty < 1e-3,
            "penalty should be ~0 at target, got {penalty}"
        );
    }

    #[test]
    fn batch_coherence_sweep_basic() {
        let n_elem = 4;
        let n_areas = 2;
        // Area 0: in-phase (cos=1). Area 1: also in-phase → coherence 1.
        let areas = vec![1.0f32; n_areas * n_elem];
        let sin_buf = vec![0.0f32; n_areas * n_elem];
        let cos_buf = vec![1.0f32; n_areas * n_elem];
        let mut out = vec![0.0f32; n_areas * n_areas];
        batch_coherence_sweep(&mut out, &areas, &sin_buf, &cos_buf, n_areas, n_elem);
        assert!((out[0] - 1.0).abs() < 1e-4, "self-coherence should be 1");
        assert!((out[1] - 1.0).abs() < 1e-4, "identical areas → coherence 1");
    }
}
