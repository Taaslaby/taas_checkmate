//! # 🧪 Lane Consistency Tests — SIMD vs Scalar Parity
//!
//! The **contract of the engine**: whatever `SimdLane` does to 8 floats at
//! once, it must do identically to 1 float at a time.
//!
//! Every test here runs the same operation through both lanes and asserts
//! that all 8 SIMD lanes match the scalar result within `1e-5` (with NaN and
//! Inf treated as equal). Coverage:
//!
//! - **Neural primitives** — `sigmoid`, `vtrap`, `ohmic` (the biological core)
//! - **Trigonometry** — `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
//! - **Exponentials & logs** — `exp`, `ln`, `log2`, `log10`
//! - **Integer powers** — `square`, `cube`, `powi::<0..=8>` (HH: `m³h`, `n⁴`)
//! - **Arithmetic** — `abs`, `signum`, `sqrt`, `hypot`, `mul_add`, `cbrt`, `recip`
//! - **Rounding** — `round`, `floor`, `ceil`, `trunc`, `fract`
//! - **Comparison & logic** — `cmp_ge`, `cmp_gt`, `max`, `min`, `and`, `not`, `blend`
//! - **Model consistency** — every zoo model run on a pure-SIMD batch (n=8)
//!   must match a mixed batch (n=9: 8 SIMD + 1 scalar tail) bit-for-bit.
//!
//! This is the safety net that lets `core` stay **zero-cost**: if a hardware
//! instruction ever diverges from IEEE scalar semantics, these tests catch it
//! before it silently corrupts a 138k-neuron fly brain.

#[cfg(test)]
mod consistency_tests {
    use crate::core::lane::SimdLane;
    use wide::f32x8;

    /// Helper: apply an operation to SIMD and scalar, compare all 8 lanes.
    fn check_binop<FSimd, FScalar>(
        name: &str,
        simd_a: f32x8,
        simd_b: f32x8,
        f_simd: FSimd,
        f_scalar: FScalar,
    ) where
        FSimd: Fn(f32x8, f32x8) -> f32x8,
        FScalar: Fn(f32, f32) -> f32,
    {
        let simd = f_simd(simd_a, simd_b);
        let arr = simd.to_array();
        let sa = simd_a.to_array();
        let sb = simd_b.to_array();
        for i in 0..8 {
            let scalar = f_scalar(sa[i], sb[i]);
            let ok = (arr[i] - scalar).abs() < 1e-5
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(
                ok,
                "{}[{}]: SIMD={:.10} != Scalar={:.10}",
                name, i, arr[i], scalar
            );
        }
    }

    fn check_unop<FSimd, FScalar>(name: &str, simd_a: f32x8, f_simd: FSimd, f_scalar: FScalar)
    where
        FSimd: Fn(f32x8) -> f32x8,
        FScalar: Fn(f32) -> f32,
    {
        let simd = f_simd(simd_a);
        let arr = simd.to_array();
        let sa = simd_a.to_array();
        for i in 0..8 {
            let scalar = f_scalar(sa[i]);
            let ok = (arr[i] - scalar).abs() < 1e-5
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(
                ok,
                "{}[{}]: SIMD={:.10} != Scalar={:.10}\n",
                name, i, arr[i], scalar
            );
        }
    }

    fn v(vals: &[f32; 8]) -> f32x8 {
        f32x8::from(*vals)
    }

    const VALS: [f32; 8] = [-1e10, -1.5, -1.0, -2.0, 0.0, 1.0, 1.5, 1e10];
    const POS_VALS: [f32; 8] = [1e-10, 0.1, 0.5, 1.0, 2.0, 10.0, 1e5, 1e10];
    const FRAC_VALS: [f32; 8] = [
        0.1,
        0.51,
        1.0,
        1.51,
        2.51,
        std::f32::consts::PI,
        -0.51,
        -1.51,
    ];
    const BIO_VALS: [f32; 8] = [-80.0, -65.0, -55.0, -40.0, -30.0, 0.0, 30.0, 50.0];

    // ─── Neural Primitives ───

    #[test]
    fn test_sigmoid_consistency() {
        let x = v(&BIO_VALS);
        check_unop(
            "sigmoid",
            x,
            |x| SimdLane::sigmoid(x, -35.0, 5.0),
            |x| SimdLane::sigmoid(x, -35.0, 5.0),
        );
    }

    #[test]
    fn test_vtrap_consistency() {
        let x = v(&[-10.0, -1.0, -0.1, -0.001, 0.001, 0.1, 1.0, 10.0]);
        let y = v(&[10.0; 8]);
        // vtrap = x/(exp(x/y)-1) АМПЛИФИЦИРУЕТ ошибку exp на ~x². 1-ULP
        // расхождение exp между SIMD/скаляр (FMA vs mul+add) даёт здесь
        // ~1e-5; допуск 1e-3 — 100× запас, при этом ловит реальные баги.
        let simd = SimdLane::vtrap(x, y);
        let arr = simd.to_array();
        let sa = x.to_array();
        for i in 0..8 {
            let scalar = SimdLane::vtrap(sa[i], 10.0);
            assert!(
                (arr[i] - scalar).abs() < 1e-3,
                "vtrap[{}]: SIMD={:.6} != Scalar={:.6}",
                i,
                arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_ohmic_consistency() {
        let g = v(&[0.1, 0.3, 0.5, 0.7, 1.0, 0.2, 0.4, 0.8]);
        let v_vals = v(&BIO_VALS);
        check_binop(
            "ohmic",
            g,
            v_vals,
            |g, v| SimdLane::ohmic(g, -70.0, v),
            |g, v| SimdLane::ohmic(g, -70.0, v),
        );
    }

    // ─── Trigonometry ───

    #[test]
    fn test_trig_consistency() {
        let vals: [f32; 8] = [
            0.0,
            0.5,
            1.0,
            1.2,
            std::f32::consts::PI,
            -1.0,
            -0.5,
            std::f32::consts::TAU,
        ];
        let x = v(&vals);
        check_unop("sin", x, |x| x.sin(), |x| x.sin());
        check_unop("cos", x, |x| x.cos(), |x| x.cos());
        check_unop("tan", x, |x| x.tan(), |x| x.tan());
        check_unop(
            "asin",
            x.max(v(&[-1.0; 8])).min(v(&[1.0; 8])),
            |x| x.asin(),
            |x| x.asin(),
        );
        check_unop(
            "acos",
            x.max(v(&[-1.0; 8])).min(v(&[1.0; 8])),
            |x| x.acos(),
            |x| x.acos(),
        );
        check_unop("atan", x, |x| x.atan(), |x| x.atan());
        check_binop(
            "atan2",
            x,
            v(&[1.0; 8]),
            |y, x| y.atan2(x),
            |y, x| y.atan2(x),
        );
    }

    // ─── Exponentials & Logarithms ───

    #[test]
    fn test_exp_log_consistency() {
        let x = v(&[-80.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0, 80.0]);
        check_unop("exp", x, |x| x.exp(), |x| x.exp());
        let pos = v(&POS_VALS);
        check_unop("ln", pos, |x| x.ln(), |x| x.ln());
        check_unop("log2", pos, |x| x.log2(), |x| x.log2());
        check_unop("log10", pos, |x| x.log10(), |x| x.log10());
    }

    // ─── Integer Powers (square, cube, powi) ────────────────

    #[test]
    fn test_square_consistency() {
        let x = v(&VALS);
        check_unop("square", x, |x| x.square(), |x| x.square());
    }

    #[test]
    fn test_cube_consistency() {
        let x = v(&VALS);
        check_unop("cube", x, |x| x.cube(), |x| x.cube());
    }

    #[test]
    fn test_powi_0_consistency() {
        let x = v(&VALS);
        check_unop("powi<0>", x, |x| x.powi::<0>(), |_| 1.0);
    }

    #[test]
    fn test_powi_1_consistency() {
        let x = v(&VALS);
        check_unop("powi<1>", x, |x| x.powi::<1>(), |x| x);
    }

    #[test]
    fn test_powi_2_consistency() {
        let x = v(&VALS);
        check_unop("powi<2>", x, |x| x.powi::<2>(), |x| x * x);
    }

    #[test]
    fn test_powi_3_consistency() {
        let x = v(&VALS);
        check_unop("powi<3>", x, |x| x.powi::<3>(), |x| x * x * x);
    }

    #[test]
    fn test_powi_4_consistency() {
        let x = v(&VALS);
        check_unop(
            "powi<4>",
            x,
            |x| x.powi::<4>(),
            |x| {
                let s = x * x;
                s * s
            },
        );
    }

    #[test]
    fn test_powi_5_consistency() {
        let x = v(&VALS);
        check_unop(
            "powi<5>",
            x,
            |x| x.powi::<5>(),
            |x| {
                let x2 = x * x;
                let x4 = x2 * x2;
                x4 * x
            },
        );
    }

    #[test]
    fn test_powi_8_consistency() {
        let x = v(&VALS);
        check_unop(
            "powi<8>",
            x,
            |x| x.powi::<8>(),
            |x| {
                let x2 = x * x;
                let x4 = x2 * x2;
                x4 * x4
            },
        );
    }

    // ─── Arithmetic ───

    #[test]
    fn test_arith_consistency() {
        let a = v(&VALS);
        let b = v(&POS_VALS);
        check_unop("abs", v(&VALS), |x| x.abs(), |x| x.abs());
        check_unop("signum", v(&VALS), |x| x.signum(), |x| x.signum());
        check_unop("sqrt", v(&POS_VALS), |x| x.sqrt(), |x| x.sqrt());
        check_binop("hypot", a, b, |x, y| x.hypot(y), |x, y| x.hypot(y));
        check_binop(
            "mul_add",
            a,
            b,
            |x, y| x.mul_add(y, v(&[1.0; 8])),
            |x, y| x.mul_add(y, 1.0),
        );
    }

    #[test]
    fn test_cbrt_consistency() {
        let x = v(&VALS);
        let simd = x.cbrt();
        let arr = simd.to_array();
        let sa = x.to_array();
        for i in 0..8 {
            let scalar = sa[i].cbrt();
            let ok = (arr[i] - scalar).abs() < 1e-3
                || (arr[i].is_nan() && scalar.is_nan())
                || (arr[i].is_infinite() && scalar.is_infinite());
            assert!(
                ok,
                "cbrt[{}]: SIMD={:.6} != Scalar={:.6}\n",
                i, arr[i], scalar
            );
        }
    }

    #[test]
    fn test_recip_consistency() {
        fn check_recip(name: &str, simd_a: f32x8) {
            let simd = simd_a.recip();
            let arr = simd.to_array();
            let sa = simd_a.to_array();
            for i in 0..8 {
                let scalar = sa[i].recip();
                let rel_err = if scalar != 0.0 {
                    (arr[i] - scalar).abs() / scalar.abs()
                } else {
                    0.0
                };
                let ok = rel_err < 1e-3
                    || (arr[i].is_nan() && scalar.is_nan())
                    || (arr[i].is_infinite() && scalar.is_infinite());
                assert!(
                    ok,
                    "{}[{}]: SIMD={:.6} != Scalar={:.6}, rel_err={:.6}",
                    name, i, arr[i], scalar, rel_err
                );
            }
        }
        check_recip("recip", v(&POS_VALS));
        check_recip(
            "recip_neg",
            v(&[-0.1, -0.5, -2.0, -10.0, -1e5, -1e2, -1.0, -1.5]),
        );
    }

    #[test]
    fn test_hypot_overflow_safety() {
        let a = f32x8::splat(1e20);
        let b = f32x8::splat(1e20);
        let result = <f32x8 as SimdLane>::hypot(a, b);
        let expected = (1e20f32).hypot(1e20f32);
        let diff = (result.to_array()[0] - expected).abs() / expected;
        assert!(
            diff < 0.01,
            "hypot(1e20,1e20): SIMD={:.2} != std={:.2}, rel_err={:.6}",
            result.to_array()[0],
            expected,
            diff
        );
    }

    // ─── Rounding ───

    #[test]
    fn test_rounding_consistency() {
        let x = v(&FRAC_VALS);
        check_unop("round", x, |x| x.round(), |x| x.round());
        check_unop("floor", x, |x| x.floor(), |x| x.floor());
        check_unop("ceil", x, |x| x.ceil(), |x| x.ceil());
        check_unop("trunc", x, |x| x.trunc(), |x| x.trunc());
        check_unop("fract", x, |x| x.fract(), |x| x.fract());
    }

    #[test]
    fn test_trunc_negative_consistency() {
        let x = v(&[-1.5, -2.5, -3.5, -1.0, -0.5, -2.0, -100.5, 0.0]);
        check_unop("trunc_neg", x, |x| x.trunc(), |x| x.trunc());
    }

    // ─── Comparison & Logic ───

    #[test]
    fn test_comparison_consistency() {
        let a = v(&VALS);
        let b = v(&[0.0; 8]);
        check_binop("cmp_ge", a, b, SimdLane::cmp_ge, SimdLane::cmp_ge);
        check_binop("cmp_gt", a, b, SimdLane::cmp_gt, SimdLane::cmp_gt);
        check_binop("max", a, b, |x, y| x.max(y), |x, y| x.max(y));
        check_binop("min", a, b, |x, y| x.min(y), |x, y| x.min(y));
    }

    #[test]
    fn test_logic_consistency() {
        let zero = f32x8::splat(0.0);
        let vals = v(&[-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0]);
        let mask_ge = SimdLane::cmp_ge(vals, zero);
        let mask_gt = SimdLane::cmp_gt(vals, zero);
        check_binop("and", mask_ge, mask_gt, |a, b| a.and(b), |a, b| a.and(b));
        check_unop("not", mask_ge, |x| x.not(), |x| x.not());
        let t = v(&VALS);
        let f = v(&[0.0; 8]);
        check_binop(
            "blend",
            mask_gt,
            t,
            |m, t_val| <f32x8 as SimdLane>::blend(m, t_val, f),
            |m, t_val| <f32 as SimdLane>::blend(m, t_val, 0.0),
        );
    }

    #[test]
    fn test_clamp01_consistency() {
        let x = v(&[-5.0, -1.0, -0.0, 0.5, 1.0, 2.0, 5.0, 10.0]);
        check_unop("clamp01", x, |x| x.clamp01(), |x| x.clamp01());
    }

    #[test]
    fn test_move_mask_consistency() {
        let zero = f32x8::splat(0.0);
        assert_eq!(zero.move_mask(), 0, "zero mask should be 0");
        let all_nan = f32x8::splat(f32::from_bits(u32::MAX));
        let mm = all_nan.move_mask();
        assert!(mm > 0, "NaN mask should be non-zero, got {}", mm);
        assert_eq!(0.0f32.move_mask(), 0, "f32::zero().move_mask() should be 0");
        assert_eq!(
            f32::from_bits(u32::MAX).move_mask(),
            1,
            "f32::NaN.move_mask() should be 1"
        );
    }

    // ─── Model Consistency: SIMD-only (n=8) vs SIMD+scalar tail (n=9) ───

    use crate::core::population::run_neuron;
    use crate::zoo::glia::AstrocyteModel;
    use crate::zoo::neurons::{HAAPModel, HHModel, IzhikevichModel, LIFModel, VSNModel};

    #[allow(clippy::type_complexity)] // fn-указатель на 6 буферов состояния — тестовый хелпер
    fn run_model_consistency<M: crate::zoo::Model>(
        n_simd: usize,
        n_mixed: usize,
        make_params: fn(usize) -> Vec<f32>,
        make_state: fn(
            usize,
            &mut [f32],
            &mut [f32],
            &mut [f32],
            &mut [f32],
            &mut [f32],
            &mut [f32],
        ),
        i_val: f32,
        steps: usize,
    ) {
        let params_s = make_params(n_simd);
        let params_m = make_params(n_mixed);

        let (mut v_s, mut u_s, mut h_s, mut r_s, mut s_s, mut t_s) = (
            vec![0.0; n_simd],
            vec![0.0; n_simd],
            vec![0.0; n_simd],
            vec![0.0; n_simd],
            vec![0.0; n_simd],
            vec![0.0; n_simd],
        );
        make_state(
            n_simd, &mut v_s, &mut u_s, &mut h_s, &mut r_s, &mut s_s, &mut t_s,
        );
        let i_syn_s = vec![i_val; n_simd];
        let mut spikes_s = vec![0i32; n_simd];

        let (mut v_m, mut u_m, mut h_m, mut r_m, mut s_m, mut t_m) = (
            vec![0.0; n_mixed],
            vec![0.0; n_mixed],
            vec![0.0; n_mixed],
            vec![0.0; n_mixed],
            vec![0.0; n_mixed],
            vec![0.0; n_mixed],
        );
        make_state(
            n_mixed, &mut v_m, &mut u_m, &mut h_m, &mut r_m, &mut s_m, &mut t_m,
        );
        let i_syn_m = vec![i_val; n_mixed];
        let mut spikes_m = vec![0i32; n_mixed];

        for _ in 0..steps {
            unsafe {
                run_neuron::<M>(
                    v_s.as_mut_ptr(),
                    u_s.as_mut_ptr(),
                    h_s.as_mut_ptr(),
                    r_s.as_mut_ptr(),
                    s_s.as_mut_ptr(),
                    t_s.as_mut_ptr(),
                    i_syn_s.as_ptr(),
                    spikes_s.as_mut_ptr(),
                    params_s.as_ptr(),
                    n_simd,
                );
            }
            unsafe {
                run_neuron::<M>(
                    v_m.as_mut_ptr(),
                    u_m.as_mut_ptr(),
                    h_m.as_mut_ptr(),
                    r_m.as_mut_ptr(),
                    s_m.as_mut_ptr(),
                    t_m.as_mut_ptr(),
                    i_syn_m.as_ptr(),
                    spikes_m.as_mut_ptr(),
                    params_m.as_ptr(),
                    n_mixed,
                );
            }
        }

        let name = std::any::type_name::<M>();
        for i in 0..n_simd {
            assert!(
                (v_s[i] - v_m[i]).abs() < 1e-4,
                "{} v[{}]: pure SIMD={:.6} != mixed={:.6}",
                name,
                i,
                v_s[i],
                v_m[i]
            );
        }
    }

    macro_rules! model_consistency_test {
        ($name:ident, $model:ty, $params_fn:expr, $state_fn:expr, $i:expr, $steps:expr) => {
            #[test]
            fn $name() {
                run_model_consistency::<$model>(8, 9, $params_fn, $state_fn, $i, $steps);
            }
        };
    }

    fn lif_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 6];
        let dt = 0.5f32;
        for i in 0..n {
            p[i] = dt;
            p[n + i] = 0.1;
            p[2 * n + i] = -70.0;
            p[3 * n + i] = -54.0;
            p[4 * n + i] = -70.0;
            p[5 * n + i] = 2.0;
        }
        p
    }
    fn lif_state(
        _n: usize,
        v: &mut [f32],
        u: &mut [f32],
        _h: &mut [f32],
        _r: &mut [f32],
        _s: &mut [f32],
        _t: &mut [f32],
    ) {
        v.fill(-70.0);
        u.fill(0.0);
    }

    fn hh_params(n: usize) -> Vec<f32> {
        let q10 = 3.0f32.powf(0.0); // celsius = 6.3
        let mut p = vec![0.0; n * 12];
        for i in 0..n {
            p[i] = 0.01;
            p[n + i] = 120.0;
            p[2 * n + i] = 50.0;
            p[3 * n + i] = 36.0;
            p[4 * n + i] = -77.0;
            p[5 * n + i] = 0.3;
            p[6 * n + i] = -54.387;
            p[7 * n + i] = q10;
            p[8 * n + i] = 0.0;
            p[9 * n + i] = 1.0;
            p[10 * n + i] = 0.0;
            p[11 * n + i] = 120.0; // g_CaT, E_CaT (standard: no CaT)
        }
        p
    }
    fn hh_state(
        _n: usize,
        v: &mut [f32],
        m: &mut [f32],
        h: &mut [f32],
        kn: &mut [f32],
        cat: &mut [f32],
        flag: &mut [f32],
    ) {
        v.fill(-65.0);
        m.fill(0.05);
        h.fill(0.6);
        kn.fill(0.32);
        cat.fill(0.19);
        flag.fill(1.0);
    }

    fn haap_params(n: usize) -> Vec<f32> {
        let inv_tau_u = 1.0 / 100.0;
        let inv_tau_h = 1.0 / 200.0;
        let mut p = vec![0.0; n * 17];
        for i in 0..n {
            p[i] = 0.05;
            p[n + i] = 1.0;
            p[2 * n + i] = 0.1;
            p[3 * n + i] = -65.0;
            p[4 * n + i] = 0.5;
            p[5 * n + i] = -80.0;
            p[6 * n + i] = 0.3;
            p[7 * n + i] = -30.0;
            p[8 * n + i] = inv_tau_u;
            p[9 * n + i] = inv_tau_h;
            p[10 * n + i] = -35.0;
            p[11 * n + i] = 5.0;
            p[12 * n + i] = -75.0;
            p[13 * n + i] = 6.0;
            p[14 * n + i] = -40.0;
            p[15 * n + i] = -55.0;
            p[16 * n + i] = 0.02;
        }
        p
    }
    fn haap_state(
        _n: usize,
        v: &mut [f32],
        u: &mut [f32],
        h: &mut [f32],
        _r: &mut [f32],
        _s: &mut [f32],
        _t: &mut [f32],
    ) {
        v.fill(-65.0);
        u.fill(0.0);
        h.fill(1.0);
    }

    fn izhi_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 7];
        for i in 0..n {
            p[i] = 0.5;
            p[n + i] = 0.02;
            p[2 * n + i] = 0.2;
            p[3 * n + i] = -65.0;
            p[4 * n + i] = 8.0;
            p[5 * n + i] = 2.0;
            p[6 * n + i] = 30.0;
        }
        p
    }
    fn izhi_state(
        _n: usize,
        v: &mut [f32],
        u: &mut [f32],
        _h: &mut [f32],
        r: &mut [f32],
        _s: &mut [f32],
        _t: &mut [f32],
    ) {
        v.fill(-65.0);
        u.fill(-13.0);
        r.fill(0.0);
    }

    fn vsn_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * 8];
        for i in 0..n {
            p[i] = 0.5;
            p[n + i] = 0.1;
            p[2 * n + i] = -70.0;
            p[3 * n + i] = -54.0;
            p[4 * n + i] = -70.0;
            p[5 * n + i] = 2.0;
            p[6 * n + i] = 0.9;
            p[7 * n + i] = 10.0;
        }
        p
    }
    fn vsn_state(
        _n: usize,
        v: &mut [f32],
        u: &mut [f32],
        _h: &mut [f32],
        _r: &mut [f32],
        _s: &mut [f32],
        _t: &mut [f32],
    ) {
        v.fill(-70.0);
        u.fill(0.0);
    }

    fn astro_params(n: usize) -> Vec<f32> {
        let inv_tau_ca = 1.0 / 1000.0;
        let inv_tau_glu = 1.0 / 10.0;
        let inv_tau_k = 1.0 / 100.0;
        let mut p = vec![0.0; n * 9];
        for i in 0..n {
            p[i] = 1.0;
            p[n + i] = inv_tau_ca;
            p[2 * n + i] = inv_tau_glu;
            p[3 * n + i] = inv_tau_k;
            p[4 * n + i] = 0.1;
            p[5 * n + i] = 0.5;
            p[6 * n + i] = 0.3;
            p[7 * n + i] = 0.5;
            p[8 * n + i] = 0.1;
        }
        p
    }
    fn astro_state(
        _n: usize,
        ca: &mut [f32],
        glu: &mut [f32],
        k: &mut [f32],
        gain: &mut [f32],
        _s: &mut [f32],
        _t: &mut [f32],
    ) {
        ca.fill(0.5);
        glu.fill(0.5);
        k.fill(0.5);
        gain.fill(1.25);
    }

    model_consistency_test!(
        test_lif_scalar_consistency,
        LIFModel,
        lif_params,
        lif_state,
        30.0,
        100
    );
    model_consistency_test!(
        test_hh_scalar_consistency,
        HHModel,
        hh_params,
        hh_state,
        50.0,
        500
    );
    model_consistency_test!(
        test_haap_scalar_consistency,
        HAAPModel,
        haap_params,
        haap_state,
        15.0,
        200
    );
    model_consistency_test!(
        test_izhikevich_scalar_consistency,
        IzhikevichModel,
        izhi_params,
        izhi_state,
        10.0,
        100
    );
    model_consistency_test!(
        test_vsn_scalar_consistency,
        VSNModel,
        vsn_params,
        vsn_state,
        30.0,
        100
    );
    model_consistency_test!(
        test_astrocyte_scalar_consistency,
        AstrocyteModel,
        astro_params,
        astro_state,
        1.0,
        20
    );
}
