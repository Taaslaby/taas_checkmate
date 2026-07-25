//! Hindmarsh-Rose (HR) 3-variable spiking & bursting neuron model.
//! ой ой это может вызывать у людей панические атаки
//! Safe Rust wrapper over SIMD kernel (`kernels::hr_step`).
//!
//! Equations:
//!   dx/dt = y - a·x³ + b·x² - z + I_ext
//!   dy/dt = c - d·x² - y
//!   dz/dt = r·(s·(x - x_R) - z)
//!
//! Reference: Hindmarsh & Rose (1984) "A model of neuronal bursting using three
//! coupled first order differential equations" — Proc. R. Soc. Lond. B.

use crate::kernels;
use rayon::prelude::*;

/// Step `n` Hindmarsh-Rose neurons. Parallel via Rayon, `n` must be a multiple of 8.
///
/// Parameters per neuron:
/// - `dt`: time step (ms), uniform across all neurons
/// - `a, b, c, d`: nullcline shape parameters (defaults: 1.0, 3.0, 1.0, 5.0)
/// - `r`: slow adaptation rate (default: 0.006)
/// - `s`: adaptation sensitivity (default: 4.0)
/// - `xr`: resting potential offset (default: -1.6)
///
/// Spike detection: x ≥ 1.0.
pub fn step_hr(
    x: &mut [f32],
    y: &mut [f32],
    z: &mut [f32],
    I: &[f32],
    spikes: &mut [i32],
    dt: f32,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    r: &[f32],
    s: &[f32],
    xr: &[f32],
) {
    let n = x.len();
    assert!(
        n.is_multiple_of(8)
            && n == y.len()
            && n == z.len()
            && n == I.len()
            && n == spikes.len()
            && n == a.len()
            && n == b.len()
            && n == c.len()
            && n == d.len()
            && n == r.len()
            && n == s.len()
            && n == xr.len()
    );

    x.par_chunks_exact_mut(8)
        .zip(y.par_chunks_exact_mut(8))
        .zip(z.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((x8, y8), z8), I8), sp8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 64];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = a[i + j];
                params[16 + j] = b[i + j];
                params[24 + j] = c[i + j];
                params[32 + j] = d[i + j];
                params[40 + j] = r[i + j];
                params[48 + j] = s[i + j];
                params[56 + j] = xr[i + j];
            }
            unsafe {
                kernels::neuron::hr::hr_step(
                    x8.as_mut_ptr(),
                    y8.as_mut_ptr(),
                    z8.as_mut_ptr(),
                    I8.as_ptr(),
                    sp8.as_mut_ptr(),
                    params.as_ptr(),
                    8,
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hr_wrapper_oscillates() {
        let n = 16;
        let dt = 0.01;
        let a = vec![1.0; n];
        let b = vec![3.0; n];
        let c = vec![1.0; n];
        let d = vec![5.0; n];
        let r = vec![0.006; n];
        let s = vec![4.0; n];
        let xr = vec![-1.6; n];
        let I = vec![3.0; n];

        let mut x = vec![-1.6f32; n];
        let mut y = vec![-12.0f32; n];
        let mut z = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        let mut total_spikes = 0i32;
        for _ in 0..10000 {
            step_hr(
                &mut x,
                &mut y,
                &mut z,
                &I,
                &mut spikes,
                dt,
                &a,
                &b,
                &c,
                &d,
                &r,
                &s,
                &xr,
            );
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "HR wrapper should burst/oscillate");
    }
}
