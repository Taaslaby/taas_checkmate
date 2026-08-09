//! Parity: `pillars::kernels::input::{PwmEncoder, LinearDecoder}` vs inline
//! legacy references extracted from git HEAD (`crates/kernels/src/io.rs`).
//!
//! Транш 2c: структуры переехали в pillars-канон. Тест замораживает
//! бит-в-бит поведение против **независимых** inline-эталонов (написаны
//! вручную из git HEAD, не layer-to-layer — иначе сравнение стало бы
//! тавтологией). Плюс проверка, что min-адаптеры `kernels::io::*`
//! эквивалентны прямому pillars-вызову при равных длинах.

use pillars::kernels::input::{LinearDecoder, PwmEncoder};

// ─── inline-эталоны из git HEAD: crates/kernels/src/io.rs ───

const GOLDEN_RATIO: f32 = 0.618_034;

fn legacy_pwm_new(n: usize, period: usize, spike_current: f32) -> LegacyPwm {
    let phase_shifts: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * GOLDEN_RATIO).fract())
        .collect();
    LegacyPwm {
        num_sensors: n,
        period,
        tick: 0,
        spike_current,
        phase_shifts,
    }
}

struct LegacyPwm {
    num_sensors: usize,
    period: usize,
    tick: usize,
    spike_current: f32,
    phase_shifts: Vec<f32>,
}

impl LegacyPwm {
    fn encode(&mut self, sensors: &[f32], out_currents: &mut [f32]) {
        let t = self.tick as f32 / self.period as f32;
        self.tick = (self.tick + 1) % self.period;
        for i in 0..self.num_sensors.min(sensors.len()).min(out_currents.len()) {
            let threshold = (t + self.phase_shifts[i]).fract();
            out_currents[i] = if sensors[i] > threshold {
                self.spike_current
            } else {
                0.0
            };
        }
    }
}

fn legacy_linear_decoder_new(n_in: usize, n_out: usize) -> LegacyLinearDecoder {
    LegacyLinearDecoder {
        n_inputs: n_in,
        n_outputs: n_out,
        weights: vec![0.0f32; n_in * n_out],
        bias: vec![0.0f32; n_out],
    }
}

struct LegacyLinearDecoder {
    n_inputs: usize,
    n_outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LegacyLinearDecoder {
    fn decode(&mut self, states: &[f32], out_buffer: &mut [f32]) {
        assert!(states.len() >= self.n_inputs);
        assert!(out_buffer.len() >= self.n_outputs);
        // legacy: out = W·states + bias (scalar, без SIMD — итог бит-в-бит)
        for (o, out) in out_buffer.iter_mut().take(self.n_outputs).enumerate() {
            let mut acc = self.bias[o];
            for (j, &state) in states.iter().take(self.n_inputs).enumerate() {
                acc += self.weights[o * self.n_inputs + j] * state;
            }
            *out = acc;
        }
    }
}

/// Детерминированный псевдослучайный массив.
fn data(n: usize, seed: i64) -> Vec<f32> {
    let mut s = seed.max(1);
    let a = 6364136223846793005i64;
    let c = 1442695040888963407i64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(a).wrapping_add(c);
            ((s >> 33) as u32 as f32 / (u32::MAX as f32 / 4.0)) - 2.0
        })
        .collect()
}

// ─── PwmEncoder ─────────────────────────────────────────

#[test]
fn pwm_encoder_bit_exact() {
    for n in [3usize, 8, 13, 17] {
        for period in [4usize, 10, 31] {
            let mut canon = PwmEncoder::new(n, period, 7.5);
            let mut legacy = legacy_pwm_new(n, period, 7.5);
            let sensors = data(n, 42 + n as i64);
            let mut out_c = vec![0.0f32; n];
            let mut out_l = vec![0.0f32; n];
            for _ in 0..(period * 2 + 3) {
                canon.encode(&sensors, &mut out_c);
                legacy.encode(&sensors, &mut out_l);
                assert_eq!(
                    out_c, out_l,
                    "PwmEncoder n={n} period={period}: bit mismatch"
                );
            }
        }
    }
}

#[test]
fn pwm_encoder_partial_slices() {
    // min-усечение: сенсоров больше, чем выходов — канон пишет в min(len)
    let mut canon = PwmEncoder::new(8, 10, 3.0);
    let mut legacy = legacy_pwm_new(8, 10, 3.0);
    let sensors = vec![0.9f32; 8];
    let mut out_c = vec![0.0f32; 5];
    let mut out_l = vec![0.0f32; 5];
    for _ in 0..12 {
        canon.encode(&sensors, &mut out_c);
        legacy.encode(&sensors, &mut out_l);
        assert_eq!(out_c, out_l, "PwmEncoder partial slice mismatch");
    }
}

// ─── LinearDecoder ──────────────────────────────────────

#[test]
fn linear_decoder_bit_exact() {
    let n_in = 5;
    let n_out = 3;
    let mut canon = LinearDecoder::new(n_in, n_out);
    let mut legacy = legacy_linear_decoder_new(n_in, n_out);
    let w = data(n_in * n_out, 7);
    let b = data(n_out, 8);
    canon.weights_mut().copy_from_slice(&w);
    canon.bias_mut().copy_from_slice(&b);
    legacy.weights.copy_from_slice(&w);
    legacy.bias.copy_from_slice(&b);

    let states = data(n_in, 9);
    let mut out_c = vec![0.0f32; n_out];
    let mut out_l = vec![0.0f32; n_out];
    canon.decode(&states, &mut out_c);
    legacy.decode(&states, &mut out_l);
    assert_eq!(out_c, out_l, "LinearDecoder bit mismatch");
}

#[test]
fn linear_decoder_zero_init() {
    let mut canon = LinearDecoder::new(4, 2);
    let mut legacy = legacy_linear_decoder_new(4, 2);
    let states = data(4, 11);
    let mut out_c = vec![9.9f32; 2];
    let mut out_l = vec![9.9f32; 2];
    canon.decode(&states, &mut out_c);
    legacy.decode(&states, &mut out_l);
    assert_eq!(out_c, out_l, "LinearDecoder zero-init bit mismatch");
}

#[test]
fn linear_decoder_large_simd_matches_direct_pillars() {
    // n_in=13 → pillars SIMD-путь (n8 = 13 & !7 = 8). Канон `LinearDecoder::decode`
    // делегирует в `signal::linear_decoder` — прямой вызов pillars обязан дать
    // бит-в-бит тот же результат (это проверка самой делегации, не тавтология:
    // структура усекает до n_inputs/n_outputs).
    let n_in = 13;
    let n_out = 5;
    let mut canon = LinearDecoder::new(n_in, n_out);
    let w = data(n_in * n_out, 31);
    let b = data(n_out, 32);
    canon.weights_mut().copy_from_slice(&w);
    canon.bias_mut().copy_from_slice(&b);
    let states = data(n_in, 33);

    let mut out_c = vec![0.0f32; n_out];
    canon.decode(&states, &mut out_c);

    let mut out_p = vec![0.0f32; n_out];
    pillars::kernels::signal::linear_decoder(&mut out_p, &states, &w, &b);
    assert_eq!(
        out_c, out_p,
        "LinearDecoder SIMD path: canon != direct pillars"
    );
}

#[test]
fn linear_decoder_large_close_to_scalar_reference() {
    // SIMD-сумма (lane-wise) может отличаться от последовательной скалярной на
    // ~1 ULP — поэтому здесь допуск, а не бит-в-бит. Проверяет, что SIMD-путь
    // pillars не сломал математику (не тавтология: эталон — скалярный цикл).
    let n_in = 13;
    let n_out = 5;
    let mut canon = LinearDecoder::new(n_in, n_out);
    let w = data(n_in * n_out, 41);
    let b = data(n_out, 42);
    canon.weights_mut().copy_from_slice(&w);
    canon.bias_mut().copy_from_slice(&b);
    let states = data(n_in, 43);

    let mut out_c = vec![0.0f32; n_out];
    canon.decode(&states, &mut out_c);

    let mut legacy = legacy_linear_decoder_new(n_in, n_out);
    legacy.weights.copy_from_slice(&w);
    legacy.bias.copy_from_slice(&b);
    let mut out_l = vec![0.0f32; n_out];
    legacy.decode(&states, &mut out_l);

    for i in 0..n_out {
        let d = (out_c[i] - out_l[i]).abs();
        assert!(
            d <= 1e-4 * out_l[i].abs().max(1.0),
            "linear_decoder[{i}]: canon {} vs scalar ref {} (diff {d:.2e})",
            out_c[i],
            out_l[i]
        );
    }
}

// ─── io.rs min-адаптеры ≡ pillars (равные длины) ─────────

#[test]
fn io_adapters_match_pillars() {
    // sigmoid_gate
    let n = 13; // SIMD-хвост
    let inp = data(n, 21);
    let mut io_out = vec![0.0f32; n];
    let mut p_out = vec![0.0f32; n];
    pillars::kernels::io::sigmoid_gate(&mut io_out, &inp, 1.5, 0.2);
    pillars::kernels::signal::sigmoid_gate(&mut p_out, &inp, 1.5, 0.2);
    assert_eq!(io_out, p_out, "sigmoid_gate io vs pillars");

    // phase_gate_apply
    let r = data(n, 22);
    let mut io_out = vec![0.0f32; n];
    let mut p_out = vec![0.0f32; n];
    pillars::kernels::io::phase_gate_apply(&mut io_out, &inp, &r, 2.0, 0.1);
    pillars::kernels::signal::phase_gate_apply(&mut p_out, &inp, &r, 2.0, 0.1);
    assert_eq!(io_out, p_out, "phase_gate_apply io vs pillars");

    // hebbian_update
    let mut io_w = data(n, 23);
    let mut p_w = io_w.clone();
    let cos_diff = data(n, 24);
    pillars::kernels::io::hebbian_update(&mut io_w, &cos_diff, 0.01, 0.1, 0.001);
    pillars::kernels::signal::hebbian_update(&mut p_w, &cos_diff, 0.01, 0.1, 0.001);
    assert_eq!(io_w, p_w, "hebbian_update io vs pillars");

    // t3_drive_encoder
    let mut io_d = vec![0.0f32; n];
    let mut p_d = vec![0.0f32; n];
    pillars::kernels::io::t3_drive_encoder(&mut io_d, &inp, 3.0);
    pillars::kernels::signal::t3_drive_encoder(&mut p_d, &inp, 3.0);
    assert_eq!(io_d, p_d, "t3_drive_encoder io vs pillars");

    // echo_tap
    let delay_len = 6;
    let mut buf = vec![0.0f32; n * delay_len];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = i as f32 * 0.5;
    }
    let mut io_e = vec![0.0f32; n];
    let mut p_e = vec![0.0f32; n];
    pillars::kernels::io::echo_tap(&mut io_e, &buf, n, delay_len, 3);
    pillars::kernels::signal::echo_tap(&mut p_e, &buf, delay_len, 3);
    assert_eq!(io_e, p_e, "echo_tap io vs pillars");
}
