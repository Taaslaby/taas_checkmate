//! 📡 BEAM — Recurrent Beam Neuron (кольцевой «луч»)
//!
//! Нейрон, который не накапливает заряд, а **гоняет волну по кольцу**:
//! фаза `φ` растёт как пилообразный сигнал и, добежав до конца круга,
//! **возвращается к себе же** — это и есть рекуррентность («бим»
//! догоняет собственный хвост). Каждый полный круг с достаточной
//! энергией луча даёт импульс.
//!
//! ```text
//! wave   = 1 − |2·φ − 1|                      (треугольная волна, «волнообразный» луч)
//! drive  = f0 + k_i·I·wave + k_echo·echo      (вход толкает волну только у гребня)
//! adv    = clamp(drive·dt, 0, 1−ε)           (≤ 1 круг за шаг — Найквист кольца)
//! φ     += adv;  если φ ≥ 1 → φ − 1          (луч делает круг и заходит на новый)
//! amp   += dt·((amp0 + k_amp·I − amp)·inv_tau_amp)   (энергия луча)
//! pulse  = (φ обернулась) ∧ (amp ≥ thresh)    (круг завершён с достаточной энергией)
//! amp    = pulse ? amp·drain : amp            (импульс «выстреливает» энергию)
//! echo  += −echo·inv_tau_echo·dt;  pulse → echo += w_echo   (рекуррентный эхо-контур)
//! ```
//!
//! Эхо-состояние — собственные недавние импульсы, заведённые обратно
//! в фазу (автапс/самосинапс). Чем больше бим «стреляет», тем быстрее
//! он бежит — персистентная активность и ускорение собственного ритма,
//! как в кольцевых осцилляторах VNC и фазовых осцилляторах Курамото.
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//! Только транслируемые в CUDA операции (abs/cond/max) — sin/cos не нужны:
//! пила из фазы, волна из `abs`.

#![allow(clippy::module_inception)] // deliberate: `pub mod beam` mirrors the file module

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

/// Recurrent beam neuron — a wave that runs around a ring and feeds its
/// own pulses back into its phase (self-chasing «beam»).
///
/// # States (3)
/// - `phase`: beam phase in turns (0..1), wraps each lap — the recurrence
/// - `amp`: beam amplitude / energy, gates pulse emission
/// - `echo`: recurrent echo — recent own pulses fed back into the phase
///
/// # Params (11)
/// - `dt`, `f0`, `k_i`, `k_echo`, `inv_tau_echo`, `w_echo`,
///   `amp0`, `inv_tau_amp`, `k_amp`, `thresh`, `drain`
#[dual_neuron]
pub mod beam {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// Max phase advance per step (in turns): one lap per step is the ring's
    /// Nyquist limit — more would need multiple wraps per step.
    const MAX_LAP: f32 = 0.99;

    #[derive(NeuronModel)]
    pub struct BeamModel {
        #[state]
        pub phase: f32,
        #[state]
        pub amp: f32,
        #[state]
        pub echo: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub f0: f32,
        #[param]
        pub k_i: f32,
        #[param]
        pub k_echo: f32,
        #[param]
        pub inv_tau_echo: f32,
        #[param]
        pub w_echo: f32,
        #[param]
        pub amp0: f32,
        #[param]
        pub inv_tau_amp: f32,
        #[param]
        pub k_amp: f32,
        #[param]
        pub thresh: f32,
        #[param]
        pub drain: f32,
    }

    impl ComputeNeuron for BeamModel {
        type State<S: SimdLane> = BeamState<S>;
        type Params<S: SimdLane> = BeamParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut BeamState<S>,
            params: &BeamParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let one = S::one();
            let zero = S::zero();
            let dt = ctx.dt();

            // ─── Triangle wave («волнообразный» луч) from the CURRENT phase
            //     (the beam's position at the start of this step) ───
            let wave = one - (state.phase * S::splat(2.0) - one).abs();

            // ─── Recurrent drive: intrinsic speed + input, caught only at
            //     the wave crest (wave ∈ [0,1] — the beam «grabs» the current
            //     mid-lap, lets it go at the wrap) + the beam's OWN echo
            //     (its pulses coming back around the ring) ───
            let drive = params.f0 + params.k_i * ctx.i_syn * wave + params.k_echo * state.echo;

            // ─── Advance clamp: at most one lap per step (the ring's Nyquist
            //     limit), and never backwards — this keeps φ ∈ [0,1) and the
            //     wave positive for ANY input (a giant current can't blow up
            //     the phase) ───
            let advance = (drive.max(zero) * dt).min(S::splat(MAX_LAP));

            // ─── Phase wrap: the beam runs one full lap and re-enters the
            //     ring — «рекуррентный бим», догоняющий свой хвост ───
            let new_phase = state.phase + advance;
            let wrapped = new_phase.cmp_ge(one).cond();
            state.phase = wrapped.then(new_phase - one).else_(new_phase);

            // ─── Beam energy: relaxes toward amp0 + input charge ───
            let amp_new = (state.amp
                + dt * ((params.amp0 + params.k_amp * ctx.i_syn - state.amp) * params.inv_tau_amp))
                .max(zero);

            // ─── Pulse: a completed lap WITH enough energy ───
            let amp_ok = amp_new.cmp_ge(params.thresh);
            let spike = wrapped.then(amp_ok).else_(zero);
            ctx.emit_spike(spike);

            // ─── The pulse «fires» part of the beam's energy ───
            state.amp = spike.cond().then(amp_new * params.drain).else_(amp_new);

            // ─── Recurrent echo: decay always, inject on pulse ───
            let echo_decay = state.echo + dt * (-state.echo * params.inv_tau_echo);
            state.echo = spike
                .cond()
                .then(echo_decay + params.w_echo)
                .else_(echo_decay);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::beam::BeamModel` etc.
pub use beam::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    const DT: f32 = 0.5;
    const F0: f32 = 0.1; // 100 Гц в покое
    const THRESH: f32 = 0.5;
    const DRAIN: f32 = 0.6;

    #[allow(clippy::too_many_arguments)]
    fn make_params(
        n: usize,
        amp0: f32,
        k_amp: f32,
        w_echo: f32,
        k_echo: f32,
        drain: f32,
        inv_tau_amp: f32,
    ) -> Vec<f32> {
        let mut p = vec![0.0f32; n * BeamModel::N_PARAMS];
        for i in 0..n {
            p[BeamModelParamIdx::dt as usize * n + i] = DT;
            p[BeamModelParamIdx::f0 as usize * n + i] = F0;
            p[BeamModelParamIdx::k_i as usize * n + i] = 0.05;
            p[BeamModelParamIdx::k_echo as usize * n + i] = k_echo;
            p[BeamModelParamIdx::inv_tau_echo as usize * n + i] = 1.0 / 20.0;
            p[BeamModelParamIdx::w_echo as usize * n + i] = w_echo;
            p[BeamModelParamIdx::amp0 as usize * n + i] = amp0;
            p[BeamModelParamIdx::inv_tau_amp as usize * n + i] = inv_tau_amp;
            p[BeamModelParamIdx::k_amp as usize * n + i] = k_amp;
            p[BeamModelParamIdx::thresh as usize * n + i] = THRESH;
            p[BeamModelParamIdx::drain as usize * n + i] = drain;
        }
        p
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        n: usize,
        I: f32,
        amp0: f32,
        k_amp: f32,
        w_echo: f32,
        k_echo: f32,
        drain: f32,
        inv_tau_amp: f32,
        steps: usize,
    ) -> (Vec<f32>, i32) {
        let params = make_params(n, amp0, k_amp, w_echo, k_echo, drain, inv_tau_amp);
        let mut phase = vec![0.0; n];
        let mut amp = vec![amp0; n];
        let mut echo = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![I; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..steps {
            unsafe {
                run_neuron::<BeamModel>(
                    phase.as_mut_ptr(),
                    amp.as_mut_ptr(),
                    echo.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        (phase, total)
    }

    /// Заряженный бим (amp0 ≥ thresh) пульсирует в покое с частотой f0.
    /// drain=0.9 → после выстрела остаётся 0.81 ≥ thresh: каждый круг даёт
    /// импульс — 500 мс × 100 Гц = 50/нейрон × 8 ≈ 400.
    #[test]
    fn test_beam_pulses_at_rest() {
        let n = 8;
        let params = make_params(n, 0.9, 0.0, 0.0, 0.0, 0.9, 0.02);
        let mut phase = vec![0.0; n];
        let mut amp = vec![0.9; n];
        let mut echo = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<BeamModel>(
                    phase.as_mut_ptr(),
                    amp.as_mut_ptr(),
                    echo.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        // 500 мс × 100 Гц = 50 импульсов/нейрон × 8 нейронов ≈ 400
        assert!(
            total > 350,
            "beam should pulse ~100 Hz at rest, got {total}"
        );
    }

    /// Слабый бим (amp0 < thresh) молча крутит фазу — импульсов нет,
    /// но волна продолжает бежать по кольцу (фаза аккуратно оборачивается).
    #[test]
    fn test_beam_silent_when_weak() {
        let (phase, total) = run(8, 0.0, 0.3, 0.0, 0.0, 0.0, DRAIN, 0.02, 2000);
        assert_eq!(total, 0, "weak beam must not pulse, got {total}");
        for p in phase.iter().take(4) {
            assert!(
                (0.0..1.0).contains(p),
                "phase should stay wrapped in [0,1), got {p}"
            );
        }
    }

    /// Луч заряжается током (лазерная накачка!): сначала тишина, потом —
    /// после пересечения порога — начинаются импульсы.
    #[test]
    fn test_beam_charges_then_pulses() {
        let n = 8;
        // amp0=0.1 < 0.5; I=1.0 → равновесие amp = 0.1 + 0.5·1.0 = 0.6 > 0.5
        let params = make_params(n, 0.1, 0.5, 0.0, 0.0, 0.6, 0.02);
        let mut phase = vec![0.0; n];
        let mut amp = vec![0.1; n];
        let mut echo = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1.0; n];
        let mut spikes = vec![0i32; n];
        let mut early = 0i32;
        let mut total = 0i32;
        for step in 0..1000 {
            unsafe {
                run_neuron::<BeamModel>(
                    phase.as_mut_ptr(),
                    amp.as_mut_ptr(),
                    echo.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
            if step < 100 {
                early += spikes.iter().sum::<i32>();
            }
        }
        assert_eq!(early, 0, "no pulses before the beam is charged");
        assert!(total > 0, "charged beam must pulse, got {total}");
    }

    /// Рекуррентность: с echo-контуром (w_echo, k_echo > 0) бим разгоняет
    /// сам себя — импульсов заметно больше, чем без обратной связи.
    /// Быстрая перезарядка amp (inv_tau_amp=0.5) + мягкий drain: энергия
    /// НЕ лимитирует частоту — лимитирует только скорость фазы.
    #[test]
    fn test_beam_echo_accelerates() {
        // без echo: частота ≈ f0 (100 Гц)
        let (_, plain) = run(8, 1.0, 0.1, 0.5, 0.0, 0.0, 0.85, 0.5, 2000);
        // с echo: каждый импульс подгоняет фазу → луч бежит заметно быстрее
        let (_, with_echo) = run(8, 1.0, 0.1, 0.5, 1.5, 0.2, 0.85, 0.5, 2000);
        assert!(plain > 0, "charged beam should pulse without echo");
        assert!(
            with_echo > plain,
            "recurrent echo must accelerate the beam: echo={with_echo} plain={plain}"
        );
    }

    /// Не-выровненный n (13): tail-путь run_neuron должен вести себя как SIMD.
    #[test]
    fn test_beam_non_aligned() {
        let (_, total) = run(13, 0.0, 0.9, 0.0, 0.0, 0.0, DRAIN, 0.02, 1000);
        assert!(total > 0, "beam n=13 should pulse, got {total}");
    }

    /// Стресс: никаких NaN при огромном токе (фаза оборачивается по много
    /// раз за шаг, эхо насыщается).
    #[test]
    fn test_beam_stress_explosion() {
        let n = 8;
        let params = make_params(n, 0.9, 0.5, 2.0, 0.2, 0.6, 0.02);
        let mut phase = vec![0.0; n];
        let mut amp = vec![0.9; n];
        let mut echo = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<BeamModel>(
                    phase.as_mut_ptr(),
                    amp.as_mut_ptr(),
                    echo.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert!(
                    phase[i].is_finite() && amp[i].is_finite() && echo[i].is_finite(),
                    "non-finite at step {step}: phase={} amp={} echo={}",
                    phase[i],
                    amp[i],
                    echo[i]
                );
            }
        }
    }
}
