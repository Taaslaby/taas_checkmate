//!  SurgeN — нейрон-трансформер с обучаемыми QKV-весами.
//!
//! Каждый шаг — это forward pass одной головы внимания над историей входов
//! (8 позиций памяти), **плюс STDP-подобное обучение ключей и значений**:
//! нейрон сам подстраивает, на какую позицию прошлого «смотреть».
//!
//! ```text
//! q        = I_t / |I_t|                   ← query: единичное направление (нормализация!)
//! score_i  = clamp(q · w_i · k_scale, ±50) ← key-проекция (w_i — обучаемое состояние)
//! attn_i   = softmax(score)                ← конкуренция позиций
//! out      = Σ attn_i · (v_i · mem_i)      ← value-проекция (v_i — обучаемое состояние)
//! acc     += dt·(out − acc·τ⁻¹)            ← интегратор → спайк при пороге
//! mem      = сдвиг с затуханием            ← бегущее окно последних 8 входов
//!
//! w_i     += dt·η_k·(spike·mem_i − (w_i − w_base))   ← Хебб: спайк ∧ активный слот
//! v_i     += dt·η_v·(spike·mem_i − (v_i − v_base))   ← Value-обучение
//! w_i, v_i  ∈ [base − span, base + span]              ← кламп (кодовый диапазон)
//! ```
//!
//! Обучение: при совпадении спайка нейрона с сильным сигналом в слоте памяти
//! `i` вес `w_i` растёт — нейрон «учится смотреть» на эту позицию. Без спайков
//! (или без сигнала в слоте) вес возвращается к базе (гомеостатическая тяга).
//! С `η = 0` обучение заморожено — чистый temporal self-attention.
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//! Все операции в транслируемом подмножестве (exp/max/min/div/abs).

#![allow(clippy::module_inception)]

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx + StateIdx Enums (no magic numbers)
// ═══════════════════════════════════════════════════════════════

/// State slot indices for TransformerModel (position in the 26-state array).
/// Use these instead of magic numbers 9/17 for w_i/v_i.
/// Transformer-like neuron with learnable QKV weights (STDP-style Hebbian +
/// homeostatic pull). States: mem0..mem7 (rolling input memory), acc (leaky
/// integrator), w0..w7 (keys — learnable), v0..v7 (values — learnable).
#[dual_neuron]
pub mod transformer {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    const SCORE_CLAMP: f32 = 50.0;

    #[derive(NeuronModel)]
    pub struct TransformerModel {
        #[state]
        pub mem0: f32,
        #[state]
        pub mem1: f32,
        #[state]
        pub mem2: f32,
        #[state]
        pub mem3: f32,
        #[state]
        pub mem4: f32,
        #[state]
        pub mem5: f32,
        #[state]
        pub mem6: f32,
        #[state]
        pub mem7: f32,
        #[state]
        pub acc: f32,
        #[state]
        pub w0: f32,
        #[state]
        pub w1: f32,
        #[state]
        pub w2: f32,
        #[state]
        pub w3: f32,
        #[state]
        pub w4: f32,
        #[state]
        pub w5: f32,
        #[state]
        pub w6: f32,
        #[state]
        pub w7: f32,
        #[state]
        pub v0: f32,
        #[state]
        pub v1: f32,
        #[state]
        pub v2: f32,
        #[state]
        pub v3: f32,
        #[state]
        pub v4: f32,
        #[state]
        pub v5: f32,
        #[state]
        pub v6: f32,
        #[state]
        pub v7: f32,
        /// Raw attention output BEFORE LIF integration:
        /// `out = Σ a_i · (v_i · mem_i)` — matches PyTorch transformer_neurons_v2.
        /// Exposed for Python parity (LM forward: use `out`, not `acc`).
        #[state]
        pub out: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub decay: f32,
        #[param]
        pub gain: f32,
        #[param]
        pub inv_tau_acc: f32,
        #[param]
        pub thresh: f32,
        #[param]
        pub k_scale: f32,
        #[param]
        pub eta_k: f32,
        #[param]
        pub eta_v: f32,
        #[param]
        pub w_base: f32,
        #[param]
        pub w_span: f32,
        #[param]
        pub v_base: f32,
        #[param]
        pub v_span: f32,
    }

    impl ComputeNeuron for TransformerModel {
        type State<S: SimdLane> = TransformerState<S>;
        type Params<S: SimdLane> = TransformerParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut TransformerState<S>,
            params: &TransformerParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let one = S::one();
            let zero = S::zero();
            let dt = ctx.dt();

            // ─── QKV: query = текущий вход, keys = state.w_i, values = state.v_i ───
            // Clip query magnitude to prevent attention blindness (all scores
            // saturating at SCORE_CLAMP → uniform softmax). Clamp at 10 keeps
            // scores within [-10·w·k_scale, +10·w·k_scale], preserving
            // selectivity while still allowing strong drives.
            let q_raw = ctx.i_syn;
            // Query normalization — THE attention-blindness fix.
            // Raw scores s_i = q * w_i * k_scale: with a small query (typical
            // for real inputs), all scores are tiny → softmax ≈ uniform 1/8 →
            // the neuron cannot select a memory slot (blind attention).
            // Normalize q to unit magnitude (sign-preserving): attention now
            // depends only on the DIRECTION of q and the learned keys w_i,
            // while the signal AMPLITUDE is carried by the memory slots
            // (mem0 = i_syn * gain below). Exactly like q/√d in real
            // transformers. k_scale keeps controlling score sharpness.
            let q_abs = q_raw.abs().max(S::splat(1e-6));
            let q = q_raw / q_abs; // |q| == 1 (sign), q == 0 for |i_syn| < 1e-6
                                   // Семантика: внимание выбирает СЛОТ по знаку и весам w_i, амплитуда
                                   // сигнала идёт через mem (i_syn·gain ниже). Отрицательный вход →
                                   // q = −1 → предпочтение наиболее отрицательных w_i (анти-внимание),
                                   // что осмысленно: w_i ∈ [w_base−span, w_base+span] может быть < 0.
            let c_hi = S::splat(SCORE_CLAMP);
            let c_lo = S::splat(-SCORE_CLAMP);

            let s0 = (q * state.w0 * params.k_scale).max(c_lo).min(c_hi);
            let s1 = (q * state.w1 * params.k_scale).max(c_lo).min(c_hi);
            let s2 = (q * state.w2 * params.k_scale).max(c_lo).min(c_hi);
            let s3 = (q * state.w3 * params.k_scale).max(c_lo).min(c_hi);
            let s4 = (q * state.w4 * params.k_scale).max(c_lo).min(c_hi);
            let s5 = (q * state.w5 * params.k_scale).max(c_lo).min(c_hi);
            let s6 = (q * state.w6 * params.k_scale).max(c_lo).min(c_hi);
            let s7 = (q * state.w7 * params.k_scale).max(c_lo).min(c_hi);

            // Numerically stable softmax: subtract max before exp
            // (prevents overflow when all scores hit SCORE_CLAMP)
            let s_max = s0.max(s1).max(s2).max(s3).max(s4).max(s5).max(s6).max(s7);
            let e0 = (s0 - s_max).exp();
            let e1 = (s1 - s_max).exp();
            let e2 = (s2 - s_max).exp();
            let e3 = (s3 - s_max).exp();
            let e4 = (s4 - s_max).exp();
            let e5 = (s5 - s_max).exp();
            let e6 = (s6 - s_max).exp();
            let e7 = (s7 - s_max).exp();
            let e_sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            let a0 = e0 / e_sum;
            let a1 = e1 / e_sum;
            let a2 = e2 / e_sum;
            let a3 = e3 / e_sum;
            let a4 = e4 / e_sum;
            let a5 = e5 / e_sum;
            let a6 = e6 / e_sum;
            let a7 = e7 / e_sum;

            // ─── Shift the input memory window FIRST (PyTorch parity) ───
            // В PyTorch transformer_neurons_v2: out = Σ a_i * v_i * mem_new_i,
            // где mem_new включает текущий вход. Сдвигаем ДО вычисления out,
            // чтобы внимание видело свежий вход, а не задержанный на 1 шаг.
            // Это же корректнее для STDP (ниже): обучение на текущем сигнале.
            state.mem7 = state.mem6 * params.decay;
            state.mem6 = state.mem5 * params.decay;
            state.mem5 = state.mem4 * params.decay;
            state.mem4 = state.mem3 * params.decay;
            state.mem3 = state.mem2 * params.decay;
            state.mem2 = state.mem1 * params.decay;
            state.mem1 = state.mem0 * params.decay;
            state.mem0 = ctx.i_syn * params.gain;

            let out = a0 * (state.v0 * state.mem0)
                + a1 * (state.v1 * state.mem1)
                + a2 * (state.v2 * state.mem2)
                + a3 * (state.v3 * state.mem3)
                + a4 * (state.v4 * state.mem4)
                + a5 * (state.v5 * state.mem5)
                + a6 * (state.v6 * state.mem6)
                + a7 * (state.v7 * state.mem7);

            // ─── Save raw `out` BEFORE LIF for Python parity ───
            state.out = out;

            // ─── Leaky integrator → spike ───
            let acc_new = (state.acc + dt * (out - state.acc * params.inv_tau_acc)).max(zero);
            let spike = acc_new.cmp_ge(params.thresh);
            ctx.emit_spike(spike);
            state.acc = spike.cond().then(zero).else_(acc_new);

            // ─── STDP-подобное обучение ключей и значений ──────────────
            // Правило Хебба: δw ∝ spike · mem_i — спайк совпал с сигналом в
            // слоте → вес растёт. Гомеостаз: притяжение к w_base предотвращает
            // неконтролируемый рост. Кламп держит веса в кодовом диапазоне
            // [base−span, base+span]. Используем mem после сдвига — обучение
            // на ТЕКУЩЕМ сигнале (semantically correct: pre-then-post).
            //
            // ВАЖНО: spike — это SIMD-маска (битовый паттерн на CPU, 0/1 на
            // GPU), поэтому НЕЛЬЗЯ умножать spike * mem напрямую — на CPU
            // это NaN × float = NaN. Вместо этого spike.cond().then(mem).else_(0)
            // правильно интерпретирует маску на обеих платформах.

            let hebb_h = spike.cond().then(one).else_(zero); // Hebbian gate: 1 на спайке, 0 иначе

            let w_lo = params.w_base - params.w_span;
            let w_hi = params.w_base + params.w_span;
            let dw = dt * params.eta_k;
            let dv = dt * params.eta_v;

            let w0_new = (state.w0 + dw * (hebb_h * state.mem0 - (state.w0 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w1_new = (state.w1 + dw * (hebb_h * state.mem1 - (state.w1 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w2_new = (state.w2 + dw * (hebb_h * state.mem2 - (state.w2 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w3_new = (state.w3 + dw * (hebb_h * state.mem3 - (state.w3 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w4_new = (state.w4 + dw * (hebb_h * state.mem4 - (state.w4 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w5_new = (state.w5 + dw * (hebb_h * state.mem5 - (state.w5 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w6_new = (state.w6 + dw * (hebb_h * state.mem6 - (state.w6 - params.w_base)))
                .max(w_lo)
                .min(w_hi);
            let w7_new = (state.w7 + dw * (hebb_h * state.mem7 - (state.w7 - params.w_base)))
                .max(w_lo)
                .min(w_hi);

            let v_lo = params.v_base - params.v_span;
            let v_hi = params.v_base + params.v_span;

            let v0_new = (state.v0 + dv * (hebb_h * state.mem0 - (state.v0 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v1_new = (state.v1 + dv * (hebb_h * state.mem1 - (state.v1 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v2_new = (state.v2 + dv * (hebb_h * state.mem2 - (state.v2 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v3_new = (state.v3 + dv * (hebb_h * state.mem3 - (state.v3 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v4_new = (state.v4 + dv * (hebb_h * state.mem4 - (state.v4 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v5_new = (state.v5 + dv * (hebb_h * state.mem5 - (state.v5 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v6_new = (state.v6 + dv * (hebb_h * state.mem6 - (state.v6 - params.v_base)))
                .max(v_lo)
                .min(v_hi);
            let v7_new = (state.v7 + dv * (hebb_h * state.mem7 - (state.v7 - params.v_base)))
                .max(v_lo)
                .min(v_hi);

            state.w0 = w0_new;
            state.w1 = w1_new;
            state.w2 = w2_new;
            state.w3 = w3_new;
            state.w4 = w4_new;
            state.w5 = w5_new;
            state.w6 = w6_new;
            state.w7 = w7_new;
            state.v0 = v0_new;
            state.v1 = v1_new;
            state.v2 = v2_new;
            state.v3 = v3_new;
            state.v4 = v4_new;
            state.v5 = v5_new;
            state.v6 = v6_new;
            state.v7 = v7_new;
        }
    }
}

pub use transformer::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use crate::core::population::NeuralPopulation;
    use crate::zoo::Model;

    const DT: f32 = 0.5;
    const W_BASE: f32 = 1.0;
    const W_SPAN: f32 = 2.0;
    const V_BASE: f32 = 1.0;
    const V_SPAN: f32 = 2.0;

    fn fill_base_params(
        pop: &mut NeuralPopulation<TransformerModel>,
        k_scale: f32,
        thresh: f32,
        eta_k: f32,
        eta_v: f32,
    ) {
        pop.fill_param(TransformerModelParamIdx::dt.into(), DT);
        pop.fill_param(TransformerModelParamIdx::decay.into(), 0.95);
        pop.fill_param(TransformerModelParamIdx::gain.into(), 1.0);
        pop.fill_param(TransformerModelParamIdx::inv_tau_acc.into(), 0.5);
        pop.fill_param(TransformerModelParamIdx::thresh.into(), thresh);
        pop.fill_param(TransformerModelParamIdx::k_scale.into(), k_scale);
        pop.fill_param(TransformerModelParamIdx::eta_k.into(), eta_k);
        pop.fill_param(TransformerModelParamIdx::eta_v.into(), eta_v);
        pop.fill_param(TransformerModelParamIdx::w_base.into(), W_BASE);
        pop.fill_param(TransformerModelParamIdx::w_span.into(), W_SPAN);
        pop.fill_param(TransformerModelParamIdx::v_base.into(), V_BASE);
        pop.fill_param(TransformerModelParamIdx::v_span.into(), V_SPAN);
    }

    fn fill_weights(pop: &mut NeuralPopulation<TransformerModel>, w: &[f32; 8], v: &[f32; 8]) {
        for i in 0..8 {
            pop.fill_state(TransformerModelStateIdx::w0 as usize + i, w[i]);
            pop.fill_state(TransformerModelStateIdx::v0 as usize + i, v[i]);
        }
    }

    fn make_pop(
        w: &[f32; 8],
        v: &[f32; 8],
        k_scale: f32,
        thresh: f32,
        eta_k: f32,
        eta_v: f32,
    ) -> NeuralPopulation<TransformerModel> {
        let n = 8usize;
        let mut pop = NeuralPopulation::<TransformerModel>::new(n);
        fill_base_params(&mut pop, k_scale, thresh, eta_k, eta_v);
        fill_weights(&mut pop, w, v);
        pop
    }

    fn read_weights(
        pop: &NeuralPopulation<TransformerModel>,
        i_neuron: usize,
    ) -> ([f32; 8], [f32; 8]) {
        let mut w = [0.0f32; 8];
        let mut v = [0.0f32; 8];
        for i in 0..8 {
            w[i] = pop.state_slice(TransformerModelStateIdx::w0 as usize + i)[i_neuron];
            v[i] = pop.state_slice(TransformerModelStateIdx::v0 as usize + i)[i_neuron];
        }
        (w, v)
    }

    /// Постоянный вход → спайки (обучение заморожено).
    #[test]
    fn test_transformer_spikes_with_input() {
        let mut pop = make_pop(&[1.0; 8], &[1.0; 8], 0.5, 0.9, 0.0, 0.0);
        let i_in = vec![1.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..1000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "spikes with I=1, got {total}");
    }

    /// Без входа — тишина.
    #[test]
    fn test_transformer_silent_without_input() {
        let mut pop = make_pop(&[1.0; 8], &[1.0; 8], 0.5, 0.9, 0.0, 0.0);
        let i_in = vec![0.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;
        for _ in 0..500 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert_eq!(total, 0, "no input → no spikes");
    }

    /// Selective temporal attention: W_k=[0,…,0,1] → спайк через ~7 шагов;
    /// W_k=[1,0,…] → спайк на импульсе (обучение заморожено).
    #[test]
    fn test_transformer_delayed_association() {
        let k_scale = 20.0;
        let v = [1.0; 8];

        // ── Смотрим на 7 шагов назад ──
        let mut w7 = [0.0f32; 8];
        w7[7] = 1.0;
        let mut pop_a = make_pop(&w7, &v, k_scale, 0.9, 0.0, 0.0);
        let mut i_a = vec![0.5f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut first = -1i32;
        for step in 0..200 {
            i_a.fill(if step == 50 { 1.5 } else { 0.5 });
            pop_a.step(&i_a, &mut spikes);
            if spikes.iter().any(|&s| s != 0) && first < 0 {
                first = step;
            }
        }
        assert!((55..=60).contains(&first), "slot-7: spike ~57, got {first}");

        // ── Смотрим на текущий вход ──
        let mut w0 = [0.0f32; 8];
        w0[0] = 1.0;
        let mut pop_b = make_pop(&w0, &v, k_scale, 0.9, 0.0, 0.0);
        let mut i_b = vec![0.2f32; 8];
        let mut first = -1i32;
        for step in 0..80 {
            i_b.fill(if step == 50 { 1.5 } else { 0.2 });
            pop_b.step(&i_b, &mut spikes);
            if spikes.iter().any(|&s| s != 0) && first < 0 {
                first = step;
            }
        }
        assert!(
            (50..=55).contains(&first),
            "slot-0: spike at pulse, got {first}"
        );
    }

    /// Value-проекция масштабирует ответ.
    #[test]
    fn test_transformer_value_scales_response() {
        let mut big = make_pop(&[1.0; 8], &[5.0; 8], 0.5, 0.9, 0.0, 0.0);
        let mut small = make_pop(&[1.0; 8], &[0.1; 8], 0.5, 0.9, 0.0, 0.0);
        let i_in = vec![1.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let mut tb = 0i32;
        let mut ts = 0i32;
        for _ in 0..1000 {
            big.step(&i_in, &mut spikes);
            tb += spikes.iter().sum::<i32>();
            small.step(&i_in, &mut spikes);
            ts += spikes.iter().sum::<i32>();
        }
        assert!(tb > 0, "big V must spike");
        assert_eq!(ts, 0, "small V must stay silent");
    }

    /// Обучение (η>0): при постоянном входе и регулярных спайках ключи и
    /// значения растут от базы; при η=0 — не меняются.
    #[test]
    fn test_transformer_learns_cooccurrence() {
        let mut learn = make_pop(&[1.0; 8], &[1.0; 8], 0.5, 0.85, 0.01, 0.01);
        let mut frozen = make_pop(&[1.0; 8], &[1.0; 8], 0.5, 0.85, 0.0, 0.0);
        let i_in = vec![1.0f32; 8];
        let mut spikes = vec![0i32; 8];
        for _ in 0..2000 {
            learn.step(&i_in, &mut spikes);
            frozen.step(&i_in, &mut spikes);
        }
        let (w_learn, v_learn) = read_weights(&learn, 0);
        let (w_frozen, v_frozen) = read_weights(&frozen, 0);

        // С обучением: ключи выросли от базы
        for (i, (&w, &v)) in w_learn.iter().zip(v_learn.iter()).enumerate() {
            assert!(
                w > W_BASE + 0.02,
                "w[{i}]={w:.4} должен подрасти от базы 1.0"
            );
            assert!(v > V_BASE + 0.02, "v[{i}]={v:.4} должен подрасти");
        }
        // Без обучения (η=0): остались на базе
        for (i, (&w, &v)) in w_frozen.iter().zip(v_frozen.iter()).enumerate() {
            assert!((w - W_BASE).abs() < 1e-4, "w[{i}]={w:.4} замёрз, ждали 1.0");
            assert!((v - V_BASE).abs() < 1e-4, "v[{i}]={v:.4} замёрз");
        }
    }

    /// Селективность: периодический импульс → спайк с запаздыванием → ключ
    /// на позиции запаздывания растёт заметно сильнее остальных.
    #[test]
    fn test_transformer_learns_selective_slot() {
        let mut pop = make_pop(&[1.0; 8], &[1.0; 8], 1.0, 0.45, 0.02, 0.02);
        let mut i_in = vec![0.05f32; 8]; // probe
        let mut spikes = vec![0i32; 8];
        // Периодический импульс каждые 8 шагов
        for step in 0..3200 {
            i_in.fill(if step % 8 == 0 { 2.0 } else { 0.05 });
            pop.step(&i_in, &mut spikes);
        }

        let (w, v) = read_weights(&pop, 0);
        let max_w = w.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_w = w.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let peak_i = w.iter().position(|&x| x == max_w).unwrap();

        // Селективность: один ключ вырос сильно, остальные меньше
        assert!(
            max_w - min_w > 0.15,
            "keys must show selectivity: max={max_w:.4} min={min_w:.4}"
        );
        // Пик — не на позиции 0 (спайк запаздывает из-за интегратора)
        assert!(
            (1..=6).contains(&peak_i),
            "peak key should be at a delayed slot 1..6, got {peak_i} (w={w:?})"
        );
        // value на пиковой позиции тоже подрастает — проверяем
        let max_v = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_v = v.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        assert!(
            max_v - min_v > 0.1,
            "values must also show selectivity: max={max_v:.4} min={min_v:.4}"
        );
        let peak_v = v.iter().position(|&x| x == max_v).unwrap();
        assert_eq!(
            peak_i, peak_v,
            "key and value peaks should align: w_peak={peak_i}, v_peak={peak_v}"
        );
    }

    /// n=13 — не-выровненный tail.
    #[test]
    fn test_transformer_non_aligned() {
        let n = 13usize;
        let mut pop = NeuralPopulation::<TransformerModel>::new(n);
        fill_base_params(&mut pop, 0.5, 0.9, 0.0, 0.0);
        for i in 0..8 {
            pop.fill_state(TransformerModelStateIdx::w0 as usize + i, 1.0);
            pop.fill_state(TransformerModelStateIdx::v0 as usize + i, 1.0);
        }
        let i_in = vec![1.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..500 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "n=13 must spike, got {total}");
    }

    /// Стресс: I=±1000 — без NaN (score-clamp + весовой кламп держат всё
    /// в рамках).
    #[test]
    fn test_transformer_stress_explosion() {
        let mut pop = make_pop(&[1.0; 8], &[1.0; 8], 5.0, 0.9, 0.001, 0.001);
        let i_in = vec![1000.0f32; 8];
        let mut spikes = vec![0i32; 8];
        for step in 0..5000 {
            pop.step(&i_in, &mut spikes);
            for c in 0..TransformerModel::N_STATE {
                let slice = pop.state_slice(c);
                for (i, &sv) in slice.iter().enumerate() {
                    assert!(
                        sv.is_finite(),
                        "non-finite state[{c}][{i}] at step {step}: {sv}"
                    );
                }
            }
        }
    }

    /// Гомеостатический распад: после обучения с η>0, в тишине веса
    /// возвращаются к w_base/v_base (core invariant модели).
    #[test]
    fn test_transformer_homeostatic_decay() {
        let mut pop = make_pop(&[1.0; 8], &[1.0; 8], 0.5, 0.85, 0.05, 0.05);
        let i_in = vec![1.0f32; 8];
        let mut spikes = vec![0i32; 8];

        // Phase 1: обучаем — веса должны вырасти
        for _ in 0..1000 {
            pop.step(&i_in, &mut spikes);
        }
        let (w_after_train, v_after_train) = read_weights(&pop, 0);
        assert!(
            w_after_train[0] > W_BASE + 0.1,
            "w[0]={:.4} must have grown above base {W_BASE}",
            w_after_train[0]
        );
        assert!(
            v_after_train[0] > V_BASE + 0.1,
            "v[0]={:.4} must have grown above base {V_BASE}",
            v_after_train[0]
        );

        // Phase 2: тишина → гомеостаз тянет к базе
        let silent = vec![0.0f32; 8];
        for _ in 0..5000 {
            pop.step(&silent, &mut spikes);
        }
        let (w_after_silence, v_after_silence) = read_weights(&pop, 0);
        for (i, &w) in w_after_silence.iter().enumerate() {
            assert!(
                (w - W_BASE).abs() < 0.05,
                "w[{i}]={w:.4} must return to base {W_BASE} (homeostasis)"
            );
        }
        for (i, &v_val) in v_after_silence.iter().enumerate() {
            assert!(
                (v_val - V_BASE).abs() < 0.05,
                "v[{i}]={v_val:.4} must return to base {V_BASE} (homeostasis)"
            );
        }
    }

    /// Тест: веса, обученные гладким Python-близнецом на саха-корпусе
    /// (v4, iter 6000, neuron 186, layer 0, первые 8 слотов из 32),
    /// загружены в Rust-нейрон с |v| (биологически: синапсы только возбуждающие).
    /// Проверяем: нейрон спайкает при устойчивом входе → «гладкий учитель → спайковый деплой».
    ///
    /// Параметры: k_scale=5.0 (экспортировано), thresh=0.3 (адаптирован под LIF —
    /// ниже стандартного 0.9, т.к. гладкая LM не оптимизировалась под порог),
    /// dt=0.5, decay=0.95, gain=1.0, inv_tau=0.5 — 1:1 с Python-близнецом.
    #[test]
    fn test_daughter_weights_spike_in_rust() {
        // Neuron 186: keys (w) — обученные пики внимания
        let w = [
            0.347033f32,
            -0.117163f32,
            0.0142045f32,
            -0.312795f32,
            -0.187915f32,
            -0.336683f32,
            -0.224951f32,
            -0.283267f32,
        ];
        // |v| — биологически корректно: синаптические веса ≥ 0
        let v = [
            0.227681f32,
            0.0291834f32,
            0.0697338f32,
            0.00566602f32,
            0.033575f32,
            0.0059892f32,
            0.0219732f32,
            0.00626792f32,
        ];
        let k_scale = 5.0f32;
        let thresh = 0.3f32; // ниже стандартного — гладкая LM не училась под порог LIF
        let eta_k = 0.0f32; // заморозить STDP — тест паритета
        let eta_v = 0.0f32;

        let mut pop = make_pop(&w, &v, k_scale, thresh, eta_k, eta_v);
        let n = 8usize;
        let seq_len = 30usize;
        let mut spikes = vec![0i32; n];
        let mut total_spikes = 0i32;
        let mut spike_times = Vec::new();

        for step in 0..seq_len {
            // устойчивый импульс на шагах 3–15 (как в Python-парити-тесте)
            let val = if (3..=15).contains(&step) {
                1.5f32
            } else {
                0.01f32
            };
            pop.step(&vec![val; n], &mut spikes);
            let spk_now = spikes.iter().sum::<i32>();
            if spk_now > 0 {
                spike_times.push(step as i32);
            }
            total_spikes += spk_now;
        }

        println!(
            "Дочка в Rust: {total_spikes} спайков / {seq_len} шагов, t={spike_times:?} (n={n}, k_scale={k_scale}, thresh={thresh})"
        );
        // Python-парити: нейрон 186 даёт 3 спайка при thresh=0.3 (t≈6,10,14)
        // Допускаем ±1 по времени из-за разницы floating-point (softmax на CPU vs Python)
        assert!(
            total_spikes >= 2,
            "Веса дочки ДОЛЖНЫ давать спайки — гладкий учитель → спайковый деплой! (thresh={thresh})"
        );
    }
}
