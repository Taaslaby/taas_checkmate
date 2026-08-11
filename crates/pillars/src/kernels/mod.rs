//! # 🧰 Asm Microkernels — batch `f32x8` kernels
//!
//! The lowest SIMD layer of Pillars. Where `core/` gives you the typed
//! `SimdLane` engine and `ops/` gives you per-lane primitives, this module
//! gives you **batch kernels**: flat `&[f32]` slices in, SIMD loop + scalar
//! tail out, nothing else.
//!
//! ## Philosophy
//!
//! - **Safe API, SIMD internals** — every function validates slice lengths,
//!   then runs 8-wide `f32x8` lanes (AVX2/NEON) with a scalar tail.
//! - **Zero overhead** — no allocation in the hot loop; pointers loaded via
//!   [`crate::core::util::load8`] / [`store8`](crate::core::util::store8).
//! - **Bit-identical SIMD/scalar on exp paths** — exponential-based kernels
//!   use [`fast_exp_poly_simd`](crate::math::fast_exp::fast_exp_poly_simd) on
//!   lanes and [`fast_exp`](crate::math::fast_exp::fast_exp) in tails: the
//!   same polynomial, the same bits. Trig paths (`kuramoto_prepare`,
//!   `interference_mean`) use the minimax [`fast_sincos_simd`] on lanes and
//!   libm in tails — accurate to ~1e-4, not bit-identical.
//!
//! ## Module map
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`signal`] | clamp, stochastic encoder, sigmoid/thalamic/phase gates, ReLU, add, SGD/Adam, Hebbian |
//! | [`delay`] | circular + multi-tap delay buffers |
//! | [`dopamine`] | dopamine-modulated STDP traces + weight update |
//! | [`phase`] | Kuramoto + T3 phase/amplitude dynamics |
//! | [`decode`] | LMS / NLMS / Kalman / Adam decoders, PST decode, ring synthesis |
//! | [`measure`] | interference, coherence, band energy, winding, spike entropy |
//! | [`dendrite`] | compartmentalized cable dendrites with 7 receptor types |
//! | [`input`] | Poisson generator, BSA / TTFS / phase encoders |
//! | [`io`] | min-truncating IO/gate compat adapters— бывший `kernels::io`) |
//! | [`fused`] | fused layer kernels — spike propagation + Izhikevich/LIF in one pass (RL) |
//! | [`eprop`] | e-prop batch kernels — LIF/Izhikevich/AdEx/SAW eligibility traces (волна 5) |
//! | [`wc`] | Wilson-Cowan rate dynamics — bit-identical port of `kernels::wc_step` (волна 7) |
//! | [`learning`] | e-prop surrogate + weight update, STDP/BCM batch update, Hebbian CSR, homeostatic scaling, SORN |
//! | [`synapse`] | conductance-based synaptic kinetics — 7 receptor types (AMPA…mAChR) |
//! | [`hh_channels`] | HH channel specs + polynomial engine — standard & pacemaking (CaV-T) |
//! | [`depth_wave`] | DepthGCO — depth-modulated coupled oscillators on CSR (транш 1f) |
//! | [`stats`] | firing rates, ISI, CV, SpikeRaster |
//! | [`vnc`] | VNC motor-neuron → actuator mapping |
//!
//! > `depth_wave` — канон `kernels/depth_wave.rs` (транш 1f): GcoState на CSR
//! > графе, детерминированная L2-норма (легаси-rayon был недетерминирован).

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]

pub mod astrocyte;
pub mod csr;
pub mod decode;
pub mod delay;
pub mod dendrite;
pub mod depth_wave;
pub mod dopamine;
pub mod eprop;
pub mod fused;
pub mod hh_channels;
pub mod input;
pub mod io;
pub mod learning;
pub mod measure;
pub mod phase;
pub mod signal;
pub mod stats;
pub mod synapse;
pub mod vnc;
pub mod wc;

/// 24-bit RNG scale — for Poisson spike generation (LCG state >> 8).
pub(crate) const RNG_SCALE_24: f32 = 1.0 / 16_777_216.0;

/// Wrap phase to [-π, π).
#[inline(always)]
pub fn wrap_phase(p: f32) -> f32 {
    let tau = core::f32::consts::TAU;
    p - tau * (p / tau).round()
}

// ═══════════════════════════════════════════════════════
//  Flat compat surface— бывший публичный API крейта
//  `taas-kernels`. Все batch-кернелы доступны как
//  `pillars::kernels::<name>` без префикса подмодуля.
// ═══════════════════════════════════════════════════════
pub use self::csr::{
    apply_stdp_ltd_with_topology, compute_hamiltonian_scatter_with_topology,
    compute_hamiltonian_with_topology, propagate_spikes_with_topology, rate_step_with_topology,
    step_gco_full_with_topology, ValidatedCsrGraph, ValidatedCsrTopology,
};
pub use self::decode::{
    adam_train, kalman_update, nlms_train, pst_decode, ring_synthesis, t3_complex_decoder,
    t3_lms_train, t3_pst_features,
};
pub use self::delay::{delay_buffer_circular, delay_buffer_multi_tap};
pub use self::dopamine::{dopamine_traces_step, dopamine_weight_update};
pub use self::fused::{
    fused_hidden_izhikevich_step, fused_output_lif_step, poisson_fused_hidden_step,
};
pub use self::learning::{bcm_step, homeostatic_scale, stdp_homeostatic, stdp_update};
pub use self::measure::{
    band_energy, interference, interference_mean, lazy_sincos_update, phase_coherence,
    spike_entropy, winding_matrix,
};
pub use self::phase::{kuramoto_update_viscous, t3_update};
pub use self::signal::{
    adam_step_simd, add_simd, echo_tap, hebbian_update, linear_decoder, phase_gate_apply,
    relu_simd, safety_clamp, sgd_step_simd, sigmoid_gate, stochastic_encoder, t3_drive_encoder,
    t3_inverse_kuramoto, t3_tact_generator, thalamic_gate,
};
pub use self::wc::wc_step;
// Экспоненциальный полином — общий для exp-путей всех кернелов.
pub use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd as fast_exp_simd};

/// Clamp amplitude to [0.001, 10.0] (matches Stuart-Landau amplitude bounds).
#[inline]
pub fn clamp_amp(a: f32) -> f32 {
    a.clamp(0.001, 10.0)
}

/// Fast SIMD sin/cos via minimax polynomial (~1e-4 accuracy, ~4× faster than libm).
///
/// Range-reduces to [-π/2, π/2], evaluates sin/cos Taylor polynomials, then
/// applies the correct quadrant per lane (0..3) to reconstruct full-range
/// sin/cos. Returns `(sin(x), cos(x))`.
#[inline(always)]
pub fn fast_sincos_simd(x: wide::f32x8) -> (wide::f32x8, wide::f32x8) {
    use core::f32::consts::{FRAC_2_PI, FRAC_PI_2};
    use wide::f32x8;
    let pi2 = f32x8::splat(FRAC_PI_2);
    let rcp_pi2 = f32x8::splat(FRAC_2_PI);

    let t = x * rcp_pi2;
    let n = t.round();
    let r = x - n * pi2;
    let r2 = r * r;

    let sin_poly = r2.mul_add(f32x8::splat(-0.000_198_409), f32x8::splat(0.008_333_332));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(-0.166_666_67));
    let sin_poly = r2.mul_add(sin_poly, f32x8::splat(1.0));
    let sin_val = r * sin_poly;

    let cos_poly = r2.mul_add(f32x8::splat(-0.001_388_888_9), f32x8::splat(0.041_666_668));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(-0.5));
    let cos_poly = r2.mul_add(cos_poly, f32x8::splat(1.0));
    let cos_val = cos_poly;

    let four = f32x8::splat(4.0);
    let n_mod_4 = n - four * (n / four).floor();

    let n_arr: [f32; 8] = n_mod_4.to_array();
    let s_arr: [f32; 8] = sin_val.to_array();
    let c_arr: [f32; 8] = cos_val.to_array();
    let mut out_s = [0.0f32; 8];
    let mut out_c = [0.0f32; 8];
    for i in 0..8 {
        let q = n_arr[i] as i32 & 3;
        match q {
            0 => {
                out_s[i] = s_arr[i];
                out_c[i] = c_arr[i];
            }
            1 => {
                out_s[i] = c_arr[i];
                out_c[i] = -s_arr[i];
            }
            2 => {
                out_s[i] = -s_arr[i];
                out_c[i] = -c_arr[i];
            }
            _ => {
                out_s[i] = -c_arr[i];
                out_c[i] = s_arr[i];
            }
        }
    }

    (f32x8::from(out_s), f32x8::from(out_c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_phase_basic() {
        let pi = std::f32::consts::PI;
        assert!((wrap_phase(pi * 3.0) - (-pi)).abs() < 1e-4, "wrap 3π → -π");
        assert!((wrap_phase(-pi * 3.0) - pi).abs() < 1e-4, "wrap -3π → π");
        assert!(wrap_phase(0.0).abs() < 1e-4, "wrap 0");
        assert!((wrap_phase(pi * 0.5) - pi * 0.5).abs() < 1e-4, "wrap π/2");
        assert!((wrap_phase(pi * 5.0) - (-pi)).abs() < 1e-4, "wrap 5π → -π");
    }

    #[test]
    fn clamp_amp_basic() {
        assert!((clamp_amp(5.0) - 5.0).abs() < 1e-6);
        assert!((clamp_amp(0.0001) - 0.001).abs() < 1e-6);
        assert!((clamp_amp(100.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn fast_sincos_simd_accuracy() {
        use wide::f32x8;
        for &x in &[0.0f32, 0.5, 1.0, 1.5, 2.0, -1.0, -3.0, 6.0, 100.0] {
            let (s, c) = fast_sincos_simd(f32x8::splat(x));
            assert!((s.to_array()[0] - x.sin()).abs() < 2e-3, "sin({x})");
            assert!((c.to_array()[0] - x.cos()).abs() < 2e-3, "cos({x})");
        }
    }
}
