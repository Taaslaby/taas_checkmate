use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;



/// NEURON-canonical HH rate functions (PROCEDURE rates(v) from hh.mod).
///
/// Computes ALL 6 rate variables in one SIMD pass:
///   - minf, mtau (Na⁺ activation)
///   - hinf, htau (Na⁺ inactivation)
///   - ninf, ntau (K⁺ activation)
///
/// With:
///   - vtrap for numerical stability
///   - Q10 temperature correction
///
/// Output registers (set in regs[]):
///   minf_reg = alpha_m / (alpha_m + beta_m)
///   mtau_reg = 1 / (q10 * (alpha_m + beta_m))
///   hinf_reg = alpha_h / (alpha_h + beta_h)
///   htau_reg = 1 / (q10 * (alpha_h + beta_h))
///   ninf_reg = alpha_n / (alpha_n + beta_n)
///   ntau_reg = 1 / (q10 * (alpha_n + beta_n))
#[inline(always)]
pub fn op_bio_nrn_hh_rates(
    regs: &mut [f32x8; 16],
    v_reg: u8,
    minf_reg: u8,
    mtau_reg: u8,
    hinf_reg: u8,
    htau_reg: u8,
    ninf_reg: u8,
    ntau_reg: u8,
    celsius: f32,
) {
    let v = regs[v_reg as usize];

    // ─── Q10 temperature correction ───
    // From NEURON hh.mod: q10 = 3^((celsius - 6.3) / 10)
    // At 6.3°C: q10 = 1.0 (squid standard)
    // At 37°C:   q10 ≈ 4.3 (mammalian)
    let q10 = if (celsius - 6.3).abs() > 0.1 {
        f32x8::splat(3.0f32.powf((celsius - 6.3) / 10.0))
    } else {
        f32x8::splat(1.0) // near 6.3°C — skip for perf
    };

    let one = f32x8::splat(1.0);

    // ═══════════════════════════════════════════
    //  m-gate: Na⁺ activation
    // ═══════════════════════════════════════════
    // alpha_m = 0.1 * vtrap(-(v+40), 10)
    // beta_m  = 4.0 * exp(-(v+65)/18)
    let vm40 = -(v + f32x8::splat(40.0));
    let alpha_m = f32x8::splat(0.1) * crate::math::vtrap::vtrap(vm40, f32x8::splat(10.0));

    let beta_m = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));

    let sum_m = alpha_m + beta_m;
    regs[minf_reg as usize] = alpha_m / sum_m;
    regs[mtau_reg as usize] = one / (q10 * sum_m);

    // ═══════════════════════════════════════════
    //  h-gate: Na⁺ inactivation
    // ═══════════════════════════════════════════
    // alpha_h = 0.07 * exp(-(v+65)/20)
    // beta_h  = 1 / (exp(-(v+35)/10) + 1)
    let alpha_h = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    let beta_h = one / (one + fast_exp_simd(-(v + f32x8::splat(35.0)) / f32x8::splat(10.0)));

    let sum_h = alpha_h + beta_h;
    regs[hinf_reg as usize] = alpha_h / sum_h;
    regs[htau_reg as usize] = one / (q10 * sum_h);

    // ═══════════════════════════════════════════
    //  n-gate: K⁺ activation
    // ═══════════════════════════════════════════
    // alpha_n = 0.01 * vtrap(-(v+55), 10)
    // beta_n  = 0.125 * exp(-(v+65)/80)
    let vn55 = -(v + f32x8::splat(55.0));
    let alpha_n = f32x8::splat(0.01) * crate::math::vtrap::vtrap(vn55, f32x8::splat(10.0));

    let beta_n = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));

    let sum_n = alpha_n + beta_n;
    regs[ninf_reg as usize] = alpha_n / sum_n;
    regs[ntau_reg as usize] = one / (q10 * sum_n);
}

// Note: HH rate function tests (resting, depolarised, vtrap safety, temperature Q10)
// have been moved to model/models/hh.rs to run in default feature set.
// These tests are available here when building with --features vm.
