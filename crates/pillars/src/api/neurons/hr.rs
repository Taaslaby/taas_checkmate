//! Hindmarsh-Rose (HR) 3-variable spiking & bursting neuron model.
//!
//! Equations:
//!   dx/dt = y - a·x³ + b·x² - z + I_ext
//!   dy/dt = c - d·x² - y
//!   dz/dt = r·(s·(x - x_R) - z)
//!
//! Reference: Hindmarsh & Rose (1984) "A model of neuronal bursting using three
//! coupled first order differential equations" — Proc. R. Soc. Lond. B.
//!
//! Канон — `zoo::neurons::HRModel` через `run_neuron` (волна 7);
//! перенесено из `core::neuron::hr` (Фаза 8).

use crate::core::runner::run_neuron;
use crate::zoo::neurons::HRModel;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Pack a single 8-lane chunk and run `HRModel` on it.
///
/// HRModel 9-col: [dt, a, b, c, d, r, s, x_r, v_thresh=1.0]
/// (порог 1.0 — легаси-хардкод `x >= 1.0`).
#[inline]
fn process_chunk(
    chunk_idx: usize,
    x8: &mut [f32],
    y8: &mut [f32],
    z8: &mut [f32],
    I8: &[f32],
    sp8: &mut [i32],
    dt: f32,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    r: &[f32],
    s: &[f32],
    xr: &[f32],
) {
    let i = chunk_idx * 8;
    let mut params = [0.0f32; 72];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = a[i + j];
        params[16 + j] = b[i + j];
        params[24 + j] = c[i + j];
        params[32 + j] = d[i + j];
        params[40 + j] = r[i + j];
        params[48 + j] = s[i + j];
        params[56 + j] = xr[i + j];
        params[64 + j] = 1.0; // v_thresh
    }
    unsafe {
        run_neuron::<HRModel>(
            x8.as_mut_ptr(),
            y8.as_mut_ptr(),
            z8.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I8.as_ptr(),
            sp8.as_mut_ptr(),
            params.as_ptr(),
            8,
        );
    }
}

/// Step `n` Hindmarsh-Rose neurons. `n` must be a multiple of 8.
///
/// Parameters per neuron:
/// - `dt`: time step (ms), uniform across all neurons
/// - `a, b, c, d`: nullcline shape parameters (defaults: 1.0, 3.0, 1.0, 5.0)
/// - `r`: slow adaptation rate (default: 0.006)
/// - `s`: adaptation sensitivity (default: 4.0)
/// - `xr`: resting potential offset (default: -1.6)
///
/// Spike detection: x ≥ 1.0.
///
/// Распараллеливание через rayon под `feature = "rayon"` (default),
/// serial-fallback иначе — как `api::neurons::atlasov`.
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

    #[cfg(feature = "rayon")]
    x.par_chunks_exact_mut(8)
        .zip(y.par_chunks_exact_mut(8))
        .zip(z.par_chunks_exact_mut(8))
        .zip(I.par_chunks_exact(8))
        .zip(spikes.par_chunks_exact_mut(8))
        .enumerate()
        .for_each(|(chunk, ((((x8, y8), z8), I8), sp8))| {
            process_chunk(chunk, x8, y8, z8, I8, sp8, dt, a, b, c, d, r, s, xr);
        });

    #[cfg(not(feature = "rayon"))]
    for chunk in 0..n / 8 {
        let i = chunk * 8;
        process_chunk(
            chunk,
            &mut x[i..i + 8],
            &mut y[i..i + 8],
            &mut z[i..i + 8],
            &I[i..i + 8],
            &mut spikes[i..i + 8],
            dt,
            a,
            b,
            c,
            d,
            r,
            s,
            xr,
        );
    }
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
