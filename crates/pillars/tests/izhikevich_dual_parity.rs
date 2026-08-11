//! # Izhikevich Dual parity: `pillars::IzhikevichDualModel` vs scalar reference
//!
//! Консолидация/3): legacy `kernels::neuron::izhikevich::izhikevich_dual_step`
//! больше НЕ вызывается из `safe.rs` — обёртка `safe::izhikevich_dual_step`
//! делегирует канону `pillars::zoo::neurons::IzhikevichDualModel` (Sun et al. 2026).
//! Живые потребители — `connectome/sim.rs` и `connectome/runner.rs` (через safe).
//!
//! ## Что тестируется
//! 1. **Бит-в-бит** v/u/trace/refr/m/spikes против скалярного эталона с той же
//!    не-fused группировкой (0.04·v + 5.0 → ·v + 140), тем же клампом входа ±100
//!    (`I_SYN_CLAMP_RANGE` в pillars = legacy inline clamp), той же семантикой
//!    slow state (m обновляется всегда: 1.0 на спайке / 0.5 иначе, clamp [0,1]).
//! 2. **Single-step Forward Euler** — канон НЕ substepping (в отличие от
//!    pillars `IzhikevichModel`), поэтому эталон — простой Euler, как легаси.
//!
//! ⚠️ **Coupling удаления:** эталон повторяет формулу легаси-кернела. Когда
//! легаси `izhikevich_dual_step` будет удалён, этот тест остаётся —
//! он закрепляет поведение канона независимо от легаси.
//!
//! ⚠️ **Scalar tail (n % 8 ≠ 0):** канон через `run_neuron` использует SIMD-
//! семантику trace (`tr·td + (1−tr)`) и на скалярном tail. Легаси-tail писал
//! `tr += td·(1−tr)` — внутренне расходился с собственным SIMD-путём. Канон
//! унифицирует поведение (см. `test_dual_non_aligned_n` в модели).

use pillars::core::runner::run_neuron;
use pillars::zoo::neurons::IzhikevichDualModel;

const N: usize = 8;
const DT: f32 = 1.0;
const TRACE_DECAY: f32 = 0.5;
const TAU_REF: f32 = 3.0;
const TAU_M: f32 = 0.99;
const W_M: f32 = 5.0;
const STEPS: usize = 200;

fn pillars_params() -> Vec<f32> {
    let mut p = vec![0.0f32; 10 * N];
    for i in 0..N {
        p[i] = DT;
        p[N + i] = 0.02; // a
        p[2 * N + i] = 0.2; // b
        p[3 * N + i] = -65.0; // c
        p[4 * N + i] = 8.0; // d
        p[5 * N + i] = TAU_REF;
        p[6 * N + i] = 30.0; // v_thresh (legacy hardcoded)
        p[7 * N + i] = TRACE_DECAY;
        p[8 * N + i] = TAU_M;
        p[9 * N + i] = W_M;
    }
    p
}

/// Scalar reference — the exact legacy `izhikevich_dual_step` formula
/// (non-fused grouping, clamp ±100, m updated always).
fn ref_step(
    v: &mut [f32],
    u: &mut [f32],
    trace: &mut [f32],
    spikes: &mut [i32],
    refractory: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ii: &[f32],
    m: &mut [f32],
    dt: f32,
    trace_decay: f32,
    tau_ref: f32,
    tau_m: f32,
    w_m: f32,
) {
    let n = v.len();
    for i in 0..n {
        if refractory[i] > 0.0 {
            refractory[i] = (refractory[i] - dt).max(0.0);
            spikes[i] = 0;
            // Slow state decay even during refractory (toward 0.5)
            m[i] = tau_m * m[i] + (1.0 - tau_m) * 0.5;
            continue;
        }
        let orig_v = v[i];
        // Slow pathway: modulate current AFTER clamp (legacy order)
        let i_total = ii[i].clamp(-100.0, 100.0) + w_m * m[i];
        let v5 = 0.04 * orig_v + 5.0;
        let dv = v5 * orig_v + 140.0 - u[i] + i_total;
        let du = a[i] * (b[i] * orig_v - u[i]);
        v[i] += dt * dv;
        u[i] += dt * du;
        if v[i] >= 30.0 {
            spikes[i] = 1;
            v[i] = c[i];
            u[i] += d[i];
            refractory[i] = tau_ref;
            // SIMD-семантика легаси: tr·td + (1−tr) (wide mul_add = mul+add).
            // НЕ tail-формула `tr += td·(1−tr)` — легаси tail расходится с SIMD!
            trace[i] = trace[i] * trace_decay + (1.0 - trace[i]);
            m[i] = tau_m * m[i] + (1.0 - tau_m) * 1.0;
        } else {
            spikes[i] = 0;
            trace[i] *= 1.0 - trace_decay;
            m[i] = tau_m * m[i] + (1.0 - tau_m) * 0.5;
        }
        m[i] = m[i].clamp(0.0, 1.0);
    }
}

#[test]
fn dual_memory_bit_identical_to_reference() {
    // ── pillars canon path ──
    let mut v = vec![-65.0f32; N];
    let mut u = vec![-13.0f32; N];
    let mut refr = vec![0.0f32; N];
    let mut trace = vec![0.0f32; N];
    let mut m = vec![0.5f32; N];
    let mut spikes = vec![0i32; N];
    let params = pillars_params();
    let i_in = vec![10.0f32; N];
    let mut dummy = vec![0.0f32; N];

    // ── scalar reference path ──
    let mut rv = vec![-65.0f32; N];
    let mut ru = vec![-13.0f32; N];
    let mut rrefr = vec![0.0f32; N];
    let mut rtrace = vec![0.0f32; N];
    let mut rm = vec![0.5f32; N];
    let mut rspikes = vec![0i32; N];
    let a = vec![0.02f32; N];
    let b = vec![0.2f32; N];
    let c = vec![-65.0f32; N];
    let d = vec![8.0f32; N];

    let mut total = 0i32;
    for _ in 0..STEPS {
        unsafe {
            run_neuron::<IzhikevichDualModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refr.as_mut_ptr(),
                trace.as_mut_ptr(),
                m.as_mut_ptr(),
                dummy.as_mut_ptr(),
                i_in.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                N,
            );
        }
        ref_step(
            &mut rv,
            &mut ru,
            &mut rtrace,
            &mut rspikes,
            &mut rrefr,
            &a,
            &b,
            &c,
            &d,
            &i_in,
            &mut rm,
            DT,
            TRACE_DECAY,
            TAU_REF,
            TAU_M,
            W_M,
        );
        assert_eq!(v, rv, "v diverged");
        assert_eq!(u, ru, "u diverged");
        assert_eq!(trace, rtrace, "trace diverged");
        assert_eq!(refr, rrefr, "refractory diverged");
        assert_eq!(m, rm, "slow state m diverged");
        assert_eq!(spikes, rspikes, "spikes diverged");
        total += spikes.iter().sum::<i32>();
    }
    assert!(total > 0, "dual Izh must spike at I=10, w_m=5");
    eprintln!(
        "[parity] izhikevich_dual: {STEPS} steps bit-identical, spikes={total}, m_mean={:.4} ✓",
        m.iter().sum::<f32>() / N as f32
    );
}
