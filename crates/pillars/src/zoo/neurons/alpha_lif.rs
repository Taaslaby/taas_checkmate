//! AlphaLif — LIF с alpha-синапсом (модель Шу / Shiu et al. 2024, fly brain).
//!
//! Точная SIMD-реализация динамики, которую воспроизводит скалярный
//! `run_lif_alpha` в pillars-connectome (`runner.rs`). Семантика **не**
//! стандартного `LIFModel`:
//!
//! - **conductance `g` живёт между шагами**: `g ← g·(1−dt/τ_syn) + delayed`,
//!   где `delayed` — рекуррентный вход, доставленный с аксональной задержкой
//!   (ring buffer в раннере) и **срезанный рефрактерностью**;
//! - **рефрактерность маскирует только вход**, а НЕ замораживает `v`:
//!   `v` интегрируется всегда (в отличие от `LIFModel`, где refr фризит v);
//! - **`v` интегрируется со СТАРЫМ `g`** (до decay/скачка этого шага);
//! - **спайк сбрасывает и `v`, и `g`** (`g = 0` на спайке);
//! - **порог строгий** (`v > v_thresh`, не `≥`).
//!
//! # Уравнения (порядок = порядку в `run_lif_alpha`)
//!
//! ```text
//! g_old = g
//! mask  = refr_cnt ≥ refr_steps            (рефрактерность режет вход)
//! v     += time_factor · (g_old − (v − v_rest))     [v уже содержит стимул]
//! spike = v > v_thresh
//! v     = spike ? v_reset : v
//! g     = spike ? 0 : g_old·syn_decay + delayed·mask
//! refr_cnt = spike ? 0 : refr_cnt + 1
//! ```
//!
//! # Распределение ответственности
//!
//! - **Здесь**: пер-нейронная динамика (v, g, refr) — SIMD f32x8.
//! - **В раннере** (`run_lif_alpha`): CSR scatter по спайкам, ring buffer
//!   задержки, Poisson-скачок напряжения прямо в `v` (до `pop.step()`).
//!
//! `delayed` приходит как `ctx.i_syn` (вход проводимости этого шага).

use pillars_macros::dual_neuron;

#[dual_neuron]
pub mod alpha_lif {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// LIF + alpha-синапс. 3 состояния, 7 параметров.
    ///
    /// # States
    /// - `v` — мембранный потенциал (мВ)
    /// - `g` — проводимость (мВ-эквивалент, затухает с τ_syn)
    /// - `refr_cnt` — счётчик шагов с последнего спайка
    ///
    /// # Params
    /// - `dt` — шаг (мс)
    /// - `time_factor` — `dt/τ_mem`
    /// - `v_rest`, `v_thresh`, `v_reset` — потенциалы (мВ)
    /// - `refr_steps` — порог рефрактерности в шагах (0 = без рефрактерности)
    /// - `syn_decay` — `1 − dt/τ_syn` (коэффициент затухания проводимости)
    /// `#[no_clamp]`: `i_syn` здесь — проводимость в мВ-эквиваленте
    /// (рекуррентный `delayed` суммирует веса и легально превышает ±100 мВ),
    /// поэтому снимаем дефолтный кламп `i_syn` в ±100 из `NeuralPopulation`.
    #[derive(NeuronModel)]
    #[no_clamp]
    pub struct AlphaLifModel {
        #[state]
        pub v: f32,
        #[state]
        pub g: f32,
        #[state]
        pub refr_cnt: f32,

        // `#[broadcast]`: константа по популяции — читается как скаляр (splat),
        // а не как per-neuron колонка → экономия ~3.3 MB/шаг на 138K нейронах.
        #[param]
        #[broadcast]
        pub dt: f32,
        #[param]
        #[broadcast]
        pub time_factor: f32,
        #[param]
        #[broadcast]
        pub v_rest: f32,
        #[param]
        #[broadcast]
        pub v_thresh: f32,
        #[param]
        #[broadcast]
        pub v_reset: f32,
        #[param]
        pub refr_steps: f32,
        #[param]
        #[broadcast]
        pub syn_decay: f32,
    }

    impl ComputeNeuron for AlphaLifModel {
        type State<S: SimdLane> = AlphaLifState<S>;
        type Params<S: SimdLane> = AlphaLifParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            // ─── Старый g (v интегрируется с ним, до decay/скачка) ───
            let g_old = state.g;

            // ─── Рефрактерность маскирует ТОЛЬКО вход (не замораживает v) ───
            let not_refr = state.refr_cnt.cmp_ge(params.refr_steps);
            let delayed = not_refr.cond().then(ctx.i_syn).else_(S::zero());

            // ─── v: стимул уже добавлен в v раннером (Poisson прямо в v) ───
            // Точный порядок как в run_lif_alpha: v += tf·(g_old − (v − v_rest)).
            let integrand = g_old - (state.v - params.v_rest);
            let v_new = state.v + params.time_factor * integrand;

            // ─── Строгий порог (>, не ≥) ───
            let spike = v_new.cmp_gt(params.v_thresh);
            ctx.emit_spike(spike);

            // ─── Сброс v и g на спайке ───
            state.v = spike.cond().then(params.v_reset).else_(v_new);
            let g_new = g_old * params.syn_decay + delayed;
            state.g = spike.cond().then(S::zero()).else_(g_new);

            // ─── Таймер рефрактерности (для следующего шага) ───
            state.refr_cnt = spike
                .cond()
                .then(S::zero())
                .else_(state.refr_cnt + S::one());
        }
    }
}

pub use alpha_lif::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeuralPopulation;

    fn make_pop(n: usize) -> NeuralPopulation<AlphaLifModel> {
        let mut pop = NeuralPopulation::<AlphaLifModel>::new(n);
        // Params: dt, time_factor, v_rest, v_thresh, v_reset, refr_steps, syn_decay
        pop.fill_param(0, 0.1); // dt
        pop.fill_param(1, 0.1 / 20.0); // time_factor = dt/tau_mem
        pop.fill_param(2, -52.0); // v_rest
        pop.fill_param(3, -45.0); // v_thresh
        pop.fill_param(4, -52.0); // v_reset
        pop.fill_param(5, 22.0); // refr_steps (tau_ref=2.2ms / dt=0.1ms)
        pop.fill_param(6, 1.0 - 0.1 / 5.0); // syn_decay = 1 - dt/tau_syn
        pop.fill_state(0, -52.0); // v
        pop.fill_state(1, 0.0); // g
        pop.fill_state(2, 22.0); // refr_cnt (не в рефрактерности на старте)
        pop
    }

    /// Постоянный вход проводимости g=50 мВ должен довести v до порога → спайк.
    #[test]
    fn alpha_lif_spikes_with_conductance() {
        let mut pop = make_pop(8);
        let i_in = vec![50.0f32; 8]; // delayed = 50 мВ каждый шаг
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..200 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AlphaLif should spike with g=50, got {total}");
    }

    /// Без входа — тишина (v релаксирует к v_rest, спайков нет).
    #[test]
    fn alpha_lif_silent_without_input() {
        let mut pop = make_pop(8);
        let i_in = vec![0.0f32; 8];
        let mut spikes = vec![0i32; 8];
        for _ in 0..500 {
            pop.step(&i_in, &mut spikes);
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "no spikes at g=0");
    }

    /// Строгий порог: v ровно на v_thresh НЕ спайкует.
    #[test]
    fn alpha_lif_strict_threshold() {
        let mut pop = make_pop(8);
        // v_rest=-52, v_thresh=-45, time_factor=0.005, g ровно столько,
        // чтобы v_new == v_thresh == -45. v_new = v + tf*(g - (v - v_rest))
        // при v=-52: v_new = -52 + 0.005*(g - 0) = -45 → g = 7/0.005 = 1400.
        let i_in = vec![1400.0f32; 8];
        let mut spikes = vec![0i32; 8];
        pop.step(&i_in, &mut spikes);
        assert_eq!(
            spikes.iter().sum::<i32>(),
            0,
            "v == v_thresh must NOT spike (strict >)"
        );
    }

    /// Рефрактерность режет вход: после спайка ~refr_steps шагов delayed не
    /// добавляется к g, поэтому v не может снова достичь порога.
    ///
    /// Семантика «v со старым g»: delayed входит в g, а g влияет на v лишь на
    /// следующем шаге — поэтому первый спайк приходит на шаге 2, не 0.
    #[test]
    fn alpha_lif_refractory_masks_input() {
        let mut pop = make_pop(8);
        let i_in = vec![1000.0f32; 8];
        let mut spikes = vec![0i32; 8];

        // Гоним до первого спайка (шаг 2 при g_old=0 → 1000 → ~1980).
        let mut first_spike_at = None;
        for step in 0..50 {
            pop.step(&i_in, &mut spikes);
            if spikes[0] == 1 && first_spike_at.is_none() {
                first_spike_at = Some(step);
                break;
            }
        }
        let t_spike = first_spike_at.expect("must spike with strong input");
        assert!(
            t_spike >= 1,
            "first spike must be >= step 1 (old-g semantics)"
        );

        // После спайка refr_cnt = 0, refr_steps = 22 → вход маскируется.
        // g затухает к 0, v дрейфует к v_rest — повторного спайка нет.
        let mut total_after = 0i32;
        for _ in 0..20 {
            pop.step(&i_in, &mut spikes);
            total_after += spikes.iter().sum::<i32>();
        }
        assert_eq!(
            total_after, 0,
            "refractory must mask input, got {total_after}"
        );
    }

    /// `#[no_clamp]` (`CLAMP_I_SYN = false`): вход НЕ должен клампиться к ±100.
    /// `i_syn = 500` (проводимость в мВ-эквиваленте) проходит целиком в `g`.
    /// Покрывает scalar (n=1), SIMD-блок и хвост (n=9) пути `NeuralPopulation`.
    #[test]
    fn alpha_lif_no_clamp_receives_unclamped_input() {
        for n in [1usize, 9usize] {
            let mut pop = make_pop(n);
            let i_in = vec![500.0f32; n];
            let mut spikes = vec![0i32; n];
            pop.step(&i_in, &mut spikes);
            for i in 0..n {
                let g = pop.state_slice(1)[i];
                assert!(
                    (g - 500.0).abs() < 1e-3,
                    "n={n} neuron {i}: g={g}, ожидался неклампированный вход 500"
                );
            }
        }
    }

    /// SIMD/tail-паритет: n=13 — хвостовые нейроны бит-в-бит совпадают с SIMD.
    #[test]
    fn alpha_lif_simd_tail_parity() {
        let mut pop = make_pop(13);
        let i_in = vec![30.0f32; 13];
        let mut spikes = vec![0i32; 13];
        for _ in 0..100 {
            pop.step(&i_in, &mut spikes);
            for i in 1..13 {
                assert_eq!(
                    pop.state_slice(0)[i].to_bits(),
                    pop.state_slice(0)[0].to_bits(),
                    "neuron {i} diverged from neuron 0"
                );
                assert_eq!(
                    pop.state_slice(1)[i].to_bits(),
                    pop.state_slice(1)[0].to_bits(),
                    "g[{i}] diverged"
                );
            }
        }
    }
}
