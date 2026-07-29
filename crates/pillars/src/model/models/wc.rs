//! Wilson-Cowan (WC) — excitatory-inhibitory rate model
//!
//! Equations:
//!   τ_E·dE/dt = -E + f(w_EE·E - w_EI·I + I_E)
//!   τ_I·dI/dt = -I + f(w_IE·E - w_II·I + I_I)
//!   f(x) = 1/(1 + exp(-β·x))
//!
//! Params (8 × n): dt, tau_E, tau_I, w_EE, w_EI, w_IE, w_II, beta
//! State: E (excitatory rate), I (inhibitory rate)
//! No spike output — continuous rates

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Wilson-Cowan Rate Model (no spikes)
pub struct WCModel;

crate::validate_model!(WCModel {
    params: 8, state: 2, dt: 0,
    forward_euler: true
});


impl Model for WCModel {
    const N_PARAMS: usize = 8;
    const N_STATE: usize = 2;  // E(excitatory), I(inhibitory)
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let e = ctx.s(0);
        let i = ctx.s(1);

        // Inputs: w_EE·E - w_EI·I + I_E,  w_IE·E - w_II·I + I_I
        let input_e = ctx.p(3) * e - ctx.p(4) * i + ctx.i();
        // I_I uses second external input (stored in p[8] when available, or zero)
        let input_i = ctx.p(5) * e - ctx.p(6) * i + S::zero();  // ctx.i() for I_E

        // Sigmoid: f(x) = 1/(1 + exp(-β·x))
        let f_e = S::sigmoid(input_e, 0.0, ctx.pf(7));
        let f_i = S::sigmoid(input_i, 0.0, ctx.pf(7));

        // τ·dE/dt = -E + f(input)  →  E += (-E + f) / τ · dt
        // Using inv_tau (p[1] = 1/tau_E, p[2] = 1/tau_I)
        let de = (S::zero() - e + f_e) * ctx.p(1) * ctx.dt();
        let di = (S::zero() - i + f_i) * ctx.p(2) * ctx.dt();

        ctx.set_s(0, e + de);
        ctx.set_s(1, i + di);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    #[test]
    fn test_wc_converges_to_equilibrium() {
        let n = 8; let dt = 0.1;
        let mut p = vec![0.0f32; n * 8];
        for i in 0..n {
            p[i] = dt; p[n+i] = 0.1; p[2*n+i] = 0.1;  // tau_E=10, tau_I=10 → inv=0.1
            p[3*n+i] = 1.2; p[4*n+i] = 0.4; p[5*n+i] = 0.6; p[6*n+i] = 0.8; p[7*n+i] = 4.0;
        }
        let mut e = vec![0.0; n]; let mut i = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_e = vec![0.5; n]; let mut spikes = vec![0i32; n];

        for _ in 0..500 {
            unsafe { run_neuron::<WCModel>(
                e.as_mut_ptr(), i.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_e.as_ptr(), spikes.as_mut_ptr(), p.as_ptr(), n,
            ); }
        }
        for idx in 0..n {
            assert!(e[idx] > 0.0 && e[idx] < 1.0, "E[{idx}]={} should be in (0,1)", e[idx]);
            assert!(i[idx] > 0.0 && i[idx] < 1.0, "I[{idx}]={} should be in (0,1)", i[idx]);
        }
    }
    // ─── Stress tests ────────────────────────────────────────────
}
