//! ⚡ Hindmarsh-Rose (HR) — 3-variable bursting neuron
//!
//! Equations:
//!   dx/dt = y - a·x³ + b·x² - z + I
//!   dy/dt = c - d·x² - y
//!   dz/dt = r·(s·(x - x_R) - z)
//!
//! 100% polynomial — zero exp() calls. Pure SIMD multiplications.
//!
//! Params (9 × n): dt, a, b, c, d, r, s, x_R, v_thresh

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Hindmarsh-Rose Bursting Neuron Model
pub struct HRModel;

crate::validate_model!(HRModel {
    params: 9, state: 3, dt: 0,
    v_thresh: 8,
    forward_euler: true
});


impl Model for HRModel {
    /// Params (9 × n): [0]dt, [1]a, [2]b, [3]c, [4]d, [5]r, [6]s, [7]x_R, [8]v_thresh
    const N_PARAMS: usize = 9;
    const N_STATE: usize = 3;  // x(membrane), y(recovery), z(adaptation)
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let x = ctx.s(0);
        let y = ctx.s(1);
        let z = ctx.s(2);

        let x2 = x * x;

        // Horner's method: dx/dt = y + x²·(b - a·x) - z + I
        let poly_x = x2 * (ctx.p(2) - ctx.p(1) * x);
        let dx = y + poly_x - z + ctx.i();
        // dy/dt = c - d·x² - y
        let dy = ctx.p(3) - ctx.p(4) * x2 - y;
        // dz/dt = r·(s·(x - x_R) - z)
        let dz = ctx.p(5) * (ctx.p(6) * (x - ctx.p(7)) - z);

        let x_new = x + dx * ctx.dt();
        let y_new = y + dy * ctx.dt();
        let z_new = z + dz * ctx.dt();

        // Per-type spike threshold (ctx.p(8) from calibration v2)
        let spike = x_new.cmp_ge(ctx.p(8));
        ctx.spike(spike);

        ctx.set_s(0, x_new);
        ctx.set_s(1, y_new);
        ctx.set_s(2, z_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        // N_PARAMS=9: [dt, a, b, c, d, r, s, x_R, v_thresh]
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[i] = 0.01; p[n+i] = 1.0; p[2*n+i] = 3.0; p[3*n+i] = 1.0;
            p[4*n+i] = 5.0; p[5*n+i] = 0.006; p[6*n+i] = 4.0; p[7*n+i] = -1.6;
            p[8*n+i] = 1.0; // default v_thresh
        }
        p
    }

    #[test]
    fn test_hr_bursts_with_current() {
        let n = 8; let dt = 0.01;
        let params = make_params(n);
        let mut x = vec![-1.6; n]; let mut y = vec![-12.0; n]; let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![3.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..10000 {
            unsafe { run_neuron::<HRModel>(
                x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HR should burst with I=3");
    }

    #[test]
    fn test_hr_resting_no_spikes() {
        let n = 8; let dt = 0.01;
        let params = make_params(n);
        let mut x = vec![-1.6; n]; let mut y = vec![-12.0; n]; let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for _ in 0..1000 {
            unsafe { run_neuron::<HRModel>(
                x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HR should not spike with I=0");
    }
}
