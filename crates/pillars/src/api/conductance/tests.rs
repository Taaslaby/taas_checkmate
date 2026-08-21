use super::*;
use crate::api::Population;
use crate::kernels::hh_channels::{
    build_channel_meta, pacemaking_channels, standard_channels, ClosedForm, MarkovGateSpec,
    MarkovRate, MarkovTransition,
};

/// Population size for the behavioural tests (SIMD block width, no tail).
const N_NEURONS: usize = 8;
/// Integration step (ms).
const DT_MS: f32 = 0.05;
/// Reference temperature (°C) — 6.3 is the squid-axon Q10 = 1 anchor.
const CELSIUS: f32 = 6.3;
/// Simulation length: 4_000 steps × 0.05 ms = 200 ms.
const RUN_STEPS: usize = 4_000;
/// Membrane capacitance (µF/cm²), leak conductance (mS/cm²) and reversal
/// (mV) — the canonical squid-axon passive parameters shared with HHModel.
const CM_UF: f32 = 1.0;
const G_LEAK_MS: f32 = 0.3;
const E_LEAK_MV: f32 = -54.387;
/// Resting membrane potential (mV) for the gate steady-state init.
const REST_MV: f32 = -65.0;
/// Minimum spontaneous spike count expected from the pacemaking set.
const MIN_PACEMAKING_SPIKES: i64 = 3;

/// The standard squid-axon channel set, run through `ConductancePopulation`,
/// must stay silent without input (quiescent at rest) — the same invariant
/// `hh_channels` tests assert for the raw engine.
#[test]
fn standard_channels_quiescent_without_input() {
    let mut pop = build_population(standard_channels());

    let i_in = vec![0.0f32; N_NEURONS];
    let mut spikes = vec![0i32; N_NEURONS];
    let mut total = 0i64;
    for _ in 0..RUN_STEPS {
        pop.step(&i_in, &mut spikes);
        total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
    }
    assert_eq!(total, 0, "standard HH should not self-spike, got {total}");
    let v = pop.state_slice(0)[0];
    assert!(
        (-75.0..-45.0).contains(&v),
        "v should stay near rest, got {v}"
    );
}

/// Pacemaking channels (standard + CaV-T) must fire spontaneously at zero
/// input — the generic engine reproduces the canonical pacemaking behavior.
#[test]
fn pacemaking_channels_fire_spontaneously() {
    let mut pop = build_population(pacemaking_channels());

    let i_in = vec![0.0f32; N_NEURONS];
    let mut spikes = vec![0i32; N_NEURONS];
    let mut total = 0i64;
    for _ in 0..RUN_STEPS {
        pop.step(&i_in, &mut spikes);
        total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
    }
    assert!(
        total >= MIN_PACEMAKING_SPIKES,
        "pacemaking HH should fire ≥{MIN_PACEMAKING_SPIKES} spikes in 200 ms, got {total}"
    );
}

/// Build a `ConductancePopulation` from a channel set, applying the shared
/// canonical passive parameters and resting-state init used by both tests.
fn build_population(
    channels: Vec<crate::kernels::hh_channels::ChannelSpec>,
) -> ConductancePopulation {
    let meta = build_channel_meta(&channels);
    let mut pop = ConductancePopulation::new(meta, N_NEURONS, DT_MS, CELSIUS);
    pop.set_cm(CM_UF);
    pop.set_leak(G_LEAK_MS, E_LEAK_MV);
    for (i, ch) in channels.iter().enumerate() {
        pop.set_g_max(i, ch.g_max);
    }
    pop.set_resting_state(REST_MV);
    pop
}

/// `MarkovPopulation` с тем же стандартным набором NaV/KV (опущенным до
/// 2-состоянийного Маркова) должен вести себя как `ConductancePopulation`:
/// молчать без входа и держать мембрану у покоя. Это parity-якорь между
/// SIMD HH-движком и скалярным master-equation движком.
#[test]
fn markov_population_standard_channels_quiescent() {
    let channels = standard_channels();
    let mc: Vec<MarkovChannel> = channels
        .iter()
        .map(|ch| MarkovChannel {
            gates: ch
                .gates
                .iter()
                .map(|g| {
                    crate::kernels::hh_channels::gate_kinetics_to_markov(
                        g.kinetics.clone(),
                        g.power,
                    )
                })
                .collect(),
            e_rev: ch.E_rev,
            g_max: ch.g_max,
        })
        .collect();
    let mut pop = MarkovPopulation::new(mc, N_NEURONS, DT_MS, CELSIUS);
    pop.set_cm(CM_UF);
    pop.set_leak(G_LEAK_MS, E_LEAK_MV);
    for (i, ch) in channels.iter().enumerate() {
        pop.set_g_max(i, ch.g_max);
    }
    pop.set_resting_state(REST_MV);

    let i_in = vec![0.0f32; N_NEURONS];
    let mut spikes = vec![0i32; N_NEURONS];
    let mut total = 0i64;
    for _ in 0..RUN_STEPS {
        pop.step(&i_in, &mut spikes);
        total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
    }
    assert_eq!(
        total, 0,
        "Markov-lowered HH should not self-spike, got {total}"
    );
    let v = pop.state_slice(0)[0];
    assert!(
        (-75.0..-45.0).contains(&v),
        "v should stay near rest, got {v}"
    );
}

/// `MarkovPopulation::new` должен отвергнуть малформированный public
/// `MarkovGateSpec` (выход за границы перехода) честной паникой, а не
/// пропустить его до паники индексации внутри `markov_step`.
#[test]
#[should_panic(expected = "invalid gate")]
fn markov_population_rejects_malformed_gate() {
    let bad = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![MarkovTransition {
            from: 0,
            to: 2, // out of range
            rate: MarkovRate::Directed {
                rate: ClosedForm::Exp {
                    rate: 1.0,
                    midpoint: 0.0,
                    scale: 10.0,
                },
            },
        }],
        power: 1,
    };
    let ch = MarkovChannel {
        gates: vec![bad],
        e_rev: 50.0,
        g_max: 1.0,
    };
    let _ = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
}

/// Implicit Euler (derivimplicit) мембрана **безусловно устойчива**: при
/// dt = 10 мс (≈3× мембранной τ = C/gL = 3.33 мс) explicit Euler осциллирует
/// и расходится, а implicit монотонно сходится к покою без перескока.
///
/// Проверка на чистой RC-мембране (без каналов): каждый шаг обязан
/// приближать V к E_leak (никогда не пересекая его с другой стороны), что
/// явно исключает осцилляторный/расходящийся explicit-Euler режим.
#[test]
fn markov_membrane_implicit_stable_at_large_dt() {
    // Leak-only cell: τ = C/gL = 1.0 / 0.3 ≈ 3.33 ms, dt = 10 ms > 2τ.
    let mut pop = MarkovPopulation::new(Vec::new(), 1, 10.0, CELSIUS);
    pop.set_cm(CM_UF);
    pop.set_leak(G_LEAK_MS, E_LEAK_MV);
    pop.set_resting_state(-20.0); // далеко от покоя (-54.387)

    let i_in = [0.0f32];
    let mut spikes = [0i32];
    let mut v = -20.0f32;
    for _ in 0..20 {
        pop.step(&i_in, &mut spikes);
        v = pop.state_slice(0)[0];
        // Никогда не пересекаем покой с другой стороны (нет осцилляции).
        assert!(v >= E_LEAK_MV, "overshoot past rest: {v}");
        assert!(v.is_finite() && v > -1000.0, "diverged: {v}");
    }
    // 20 шагов × 10 мс достаточно, чтобы монотонно сесть на покой.
    assert!(
        (v - E_LEAK_MV).abs() < 1.0,
        "should converge to rest, got {v}"
    );
}

#[test]
#[should_panic(expected = "dt must be finite and > 0")]
fn conductance_population_rejects_zero_dt() {
    let meta = build_channel_meta(&standard_channels());
    let _ = ConductancePopulation::new(meta, N_NEURONS, 0.0, CELSIUS);
}

#[test]
#[should_panic(expected = "celsius must be finite")]
fn conductance_population_rejects_nan_celsius() {
    let meta = build_channel_meta(&standard_channels());
    let _ = ConductancePopulation::new(meta, N_NEURONS, DT_MS, f32::NAN);
}

#[test]
#[should_panic(expected = "capacitance must be finite and > 0")]
fn conductance_population_rejects_zero_cm() {
    let mut pop = build_population(standard_channels());
    pop.set_cm(0.0);
}

#[test]
#[should_panic(expected = "conductance must be finite and >= 0")]
fn conductance_population_rejects_nan_leak() {
    let mut pop = build_population(standard_channels());
    pop.set_leak(f32::NAN, E_LEAK_MV);
}

#[test]
#[should_panic(expected = "reversal must be finite")]
fn markov_population_rejects_inf_reversal() {
    let mut pop = MarkovPopulation::new(Vec::new(), N_NEURONS, DT_MS, CELSIUS);
    pop.set_leak(G_LEAK_MS, f32::INFINITY);
}

#[test]
#[should_panic(expected = "dt must be finite and > 0")]
fn markov_population_rejects_zero_dt() {
    let _ = MarkovPopulation::new(Vec::new(), N_NEURONS, 0.0, CELSIUS);
}

#[test]
#[should_panic(expected = "capacitance must be finite and > 0")]
fn markov_population_rejects_zero_cm() {
    let mut pop = MarkovPopulation::new(Vec::new(), N_NEURONS, DT_MS, CELSIUS);
    pop.set_cm(0.0);
}

#[test]
#[should_panic(expected = "conductance must be finite and >= 0")]
fn markov_population_rejects_negative_g_max() {
    let ch = MarkovChannel {
        gates: vec![],
        e_rev: 0.0,
        g_max: 1.0,
    };
    let mut pop = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
    pop.set_g_max(0, -1.0);
}

#[test]
#[should_panic(expected = "e_rev must be finite")]
fn markov_population_rejects_nan_channel_reversal() {
    let ch = MarkovChannel {
        gates: vec![],
        e_rev: f32::NAN,
        g_max: 1.0,
    };
    let _ = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
}
