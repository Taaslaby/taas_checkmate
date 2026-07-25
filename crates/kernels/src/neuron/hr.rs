#![allow(non_snake_case)]
use wide::{f32x8, CmpGe};
use super::{load8, load8_mut, store8, spike_mask_to_i32};

// ═══════════════════════════════════════════════════════
//  Hindmarsh-Rose (HR) — 3-variable spiking & bursting neuron
// ═══════════════════════════════════════════════════════
//
// Equations (coupled ODEs):
//   dx/dt = y - a·x³ + b·x² - z + I_ext
//   dy/dt = c - d·x² - y
//   dz/dt = r·(s·(x - x_R) - z)
//
// x: membrane potential (fast variable)
// y: recovery / gating variable
// z: adaptation / slow bursting variable
//
// Standard parameters:
//   a = 1.0, b = 3.0, c = 1.0, d = 5.0
//   r = 0.006 (slow adaptation speed)
//   s = 4.0, x_R = -1.6
//
// Spike detection: x ≥ threshold (default 1.0)
//
// This model is 100% polynomial (zero exp() calls) and captures tonic
// spiking, bursting, and chaotic dynamics with high SIMD efficiency.

/// Step `n` Hindmarsh-Rose neurons. Uses `f32x8` SIMD.
///
/// Params layout: dt, a, b, c, d, r, s, x_R (8 × n, column-major)
///   - dt: time step (ms), uniform across all neurons (params[0])
///   - a, b, c, d: nullcline shape parameters
///   - r: slow bursting rate constant
///   - s: adaptation sensitivity
///   - x_R: resting potential offset
///
/// # Safety
/// `x[n]`, `y[n]`, `z[n]`, `Ii[n]`, `spikes[n]`, `params[8·n]` must be valid, non-overlapping.
pub unsafe fn hr_step(
    x: *mut f32, y: *mut f32, z: *mut f32, Ii: *const f32, spikes: *mut i32,
    params: *const f32, n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 8);
    let dt_val = p[0];
    let dt_s = f32x8::splat(dt_val);
    let x_thresh = f32x8::splat(1.0); // default spike threshold
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut xx = load8_mut(x, off);
        let mut yy = load8_mut(y, off);
        let mut zz = load8_mut(z, off);

        let ii = load8(Ii, off).max(f32x8::splat(-100.0)).min(f32x8::splat(100.0));

        let aa = load8(params.wrapping_add(n), off);
        let bb = load8(params.wrapping_add(2 * n), off);
        let cc = load8(params.wrapping_add(3 * n), off);
        let dd = load8(params.wrapping_add(4 * n), off);
        let rr = load8(params.wrapping_add(5 * n), off);
        let ss = load8(params.wrapping_add(6 * n), off);
        let xr = load8(params.wrapping_add(7 * n), off);

        // HR dynamics:
        // dx/dt = y - a·x³ + b·x² - z + I
        let x2 = xx * xx;
        let x3 = x2 * xx;
        let dx = yy - aa * x3 + bb * x2 - zz + ii;

        // dy/dt = c - d·x² - y
        let dy = cc - dd * x2 - yy;

        // dz/dt = r·(s·(x - x_R) - z)
        let dz = rr * (ss * (xx - xr) - zz);

        // Forward Euler
        xx += dt_s * dx;
        yy += dt_s * dy;
        zz += dt_s * dz;

        // Spike detection: x ≥ 1.0
        let sm = xx.cmp_ge(x_thresh);

        store8(x, off, xx);
        store8(y, off, yy);
        store8(z, off, zz);

        let sp_i = spike_mask_to_i32(sm);
        let sp8 = spikes.add(off);
        *sp8 = sp_i[0]; *sp8.add(1) = sp_i[1]; *sp8.add(2) = sp_i[2]; *sp8.add(3) = sp_i[3];
        *sp8.add(4) = sp_i[4]; *sp8.add(5) = sp_i[5]; *sp8.add(6) = sp_i[6]; *sp8.add(7) = sp_i[7];
    }

    // Scalar tail (n & 7)
    if n8 < n {
        let x = std::slice::from_raw_parts_mut(x, n);
        let y = std::slice::from_raw_parts_mut(y, n);
        let z = std::slice::from_raw_parts_mut(z, n);
        let ii = std::slice::from_raw_parts(Ii, n);
        let spikes = std::slice::from_raw_parts_mut(spikes, n);

        for i in n8..n {
            let a = p[n + i];
            let b = p[2 * n + i];
            let c = p[3 * n + i];
            let d = p[4 * n + i];
            let r = p[5 * n + i];
            let s = p[6 * n + i];
            let xr = p[7 * n + i];

            let i_clamped = ii[i].clamp(-100.0, 100.0);
            let x2 = x[i] * x[i];
            let x3 = x2 * x[i];

            let dx = y[i] - a * x3 + b * x2 - z[i] + i_clamped;
            let dy = c - d * x2 - y[i];
            let dz = r * (s * (x[i] - xr) - z[i]);

            x[i] += dt_val * dx;
            y[i] += dt_val * dy;
            z[i] += dt_val * dz;

            spikes[i] = if x[i] >= 1.0 { 1 } else { 0 };
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// HR: verify bursting/spiking with sufficient current.
    #[test]
    fn hr_oscillates_with_current() {
        let n = 8;
        let dt = 0.01;
        let a = 1.0;
        let b = 3.0;
        let c = 1.0;
        let d = 5.0;
        let r = 0.006;
        let s = 4.0;
        let xr = -1.6;
        let I_val = 3.0; // Current to drive bursting

        let mut x = vec![-1.6f32; n];
        let mut y = vec![-12.0f32; n];
        let mut z = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 8];

        for j in 0..n {
            params[j] = dt;
            params[n + j] = a;
            params[2 * n + j] = b;
            params[3 * n + j] = c;
            params[4 * n + j] = d;
            params[5 * n + j] = r;
            params[6 * n + j] = s;
            params[7 * n + j] = xr;
        }

        let mut spike_count = 0i32;
        for _ in 0..10000 {
            unsafe { hr_step(x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
            spike_count += spikes.iter().sum::<i32>();
        }
        assert!(spike_count > 0, "Hindmarsh-Rose should oscillate/burst with I={I_val}");
    }

    /// HR: verify resting state without current.
    #[test]
    fn hr_resting_no_spikes() {
        let n = 8;
        let dt = 0.01;
        let a = 1.0;
        let b = 3.0;
        let c = 1.0;
        let d = 5.0;
        let r = 0.006;
        let s = 4.0;
        let xr = -1.6;
        let I_val = 0.0;

        let mut x = vec![-1.6f32; n];
        let mut y = vec![-12.0f32; n];
        let mut z = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 8];

        for j in 0..n {
            params[j] = dt;
            params[n + j] = a;
            params[2 * n + j] = b;
            params[3 * n + j] = c;
            params[4 * n + j] = d;
            params[5 * n + j] = r;
            params[6 * n + j] = s;
            params[7 * n + j] = xr;
        }

        for _ in 0..1000 {
            unsafe { hr_step(x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HR should not spike with I=0");
    }

    /// HR: verify scalar tail execution for non-aligned N.
    #[test]
    fn hr_non_aligned_n() {
        let n = 11;
        let dt = 0.01;
        let a = 1.0;
        let b = 3.0;
        let c = 1.0;
        let d = 5.0;
        let r = 0.006;
        let s = 4.0;
        let xr = -1.6;
        let I_val = 3.0;

        let mut x = vec![-1.6f32; n];
        let mut y = vec![-12.0f32; n];
        let mut z = vec![0.0f32; n];
        let I = vec![I_val; n];
        let mut spikes = vec![0i32; n];
        let mut params = vec![0.0f32; n * 8];

        for j in 0..n {
            params[j] = dt;
            params[n + j] = a;
            params[2 * n + j] = b;
            params[3 * n + j] = c;
            params[4 * n + j] = d;
            params[5 * n + j] = r;
            params[6 * n + j] = s;
            params[7 * n + j] = xr;
        }

        let mut spike_count = 0i32;
        for _ in 0..10000 {
            unsafe { hr_step(x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(), I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n) };
            spike_count += spikes.iter().sum::<i32>();
        }
        assert!(spike_count > 0, "HR non-aligned should burst");
    }
}
