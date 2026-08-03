//! Интеграционные тесты: классическое обусловливание через публичный API.

use taas_rule::*;

fn output_program() -> Rule {
    Rule::from_ops(vec![
        OpCode::NominalThreshold(Operand::Value(0.05)),
        OpCode::LeakageRate(Operand::Value(0.2)),
        OpCode::RestState(Operand::Value(0.0)),
        OpCode::Persistence(Operand::Value(0.4)),
        OpCode::Fire(Operand::Value(0.6)),
        OpCode::Stop,
    ])
}

fn make_lobe() -> Lobe {
    let tract = Tract::with_dendrites(
        TractConfig::default(),
        vec![
            Dendrite::new(0, 0.1, 0.1),
            Dendrite::new(1, 1.0, 1.0),
        ],
    );
    Lobe::new(vec![Neuron::new(output_program())], vec![tract], vec![], 7)
}

fn run_phase(
    lobe: &mut Lobe,
    n_ticks: usize,
    cue: bool,
    us: bool,
    reward: f32,
    learning: bool,
) -> (u32, (f32, f32)) {
    let activity = [if cue { 1.0 } else { 0.0 }, if us { 1.0 } else { 0.0 }];
    let mut spikes = 0u32;
    for _ in 0..n_ticks {
        if lobe.step(&activity, reward, 0.0, learning)[0] {
            spikes += 1;
        }
    }
    (spikes, lobe.tracts[0].weight(0).unwrap())
}

/// Полный сценарий: baseline → conditioning → test → extinction.
fn full_experiment() -> (u32, u32, u32, (f32, f32), (f32, f32)) {
    let mut lobe = make_lobe();
    let (baseline, _w0) = run_phase(&mut lobe, 200, true, false, 0.0, false);
    let (_train, _wt) = run_phase(&mut lobe, 600, true, true, 1.0, true);
    let (test, w1) = run_phase(&mut lobe, 200, true, false, 0.0, false);
    let (ext, w2) = run_phase(&mut lobe, 200, true, false, 0.0, true);
    (baseline, test, ext, w1, w2)
}

#[test]
fn test_conditioning_baseline_silent_test_fires() {
    let (baseline, test, _, w1, _) = full_experiment();
    assert_eq!(baseline, 0, "baseline: без обучения реакции нет");
    assert!(test > 0, "после conditioning сигнал спайкает (условный рефлекс)");
    assert!(w1.0 > 0.6, "ST должен вырасти: {}", w1.0);
    assert!(w1.1 > 0.2, "LT должен консолидировать: {}", w1.1);
}

#[test]
fn test_extinction_two_timescale_memory() {
    let (_, test, ext, w1, w2) = full_experiment();
    assert!(
        ext > 0 && ext < test,
        "рефлекс гаснет постепенно: {ext} < {test}"
    );
    assert!(
        w2.0 < w1.0 * 0.5,
        "ST обязан резко упасть при угасании: {} → {}",
        w1.0,
        w2.0
    );
    assert!(
        w2.1 > w2.0,
        "LT держит память дольше ST: {} vs {}",
        w2.1,
        w2.0
    );
}

#[test]
fn test_learning_frozen_in_observation_phase() {
    // Наблюдение без обучения не должно менять веса.
    let mut lobe = make_lobe();
    let (_, w0) = run_phase(&mut lobe, 100, true, false, 0.0, false);
    assert!((w0.0 - 0.1).abs() < 1e-6, "ST не должен меняться без обучения: {}", w0.0);
}

#[test]
fn test_chemical_reward_gating() {
    // Тракт обучается только когда reward выше порога; порог читается из Learning.
    let mut lobe = make_lobe();
    // reward = 0.4 < порог 0.5 → обучения нет.
    let (_s, w0) = run_phase(&mut lobe, 200, true, true, 0.4, true);
    assert!(
        w0.0 < 0.3,
        "ниже порога reward не должен учить: ST = {}",
        w0.0
    );
    // reward = 1.0 > порог → учится.
    let (_s, w1) = run_phase(&mut lobe, 400, true, true, 1.0, true);
    assert!(w1.0 > 0.6, "выше порога reward должен учить: ST = {}", w1.0);
}

#[test]
fn test_wta_via_lobe() {
    // Победитель-забирает-всё: максимальный STATE получает OUTPUT = 1.
    let rule = Rule::from_ops(vec![OpCode::WinnerTakesAll, OpCode::Stop]);
    let n1 = Neuron::with_vars(
        rule.clone(),
        NeuronVars {
            state: 0.9,
            ..Default::default()
        },
    );
    let n2 = Neuron::with_vars(
        rule,
        NeuronVars {
            state: 0.3,
            ..Default::default()
        },
    );
    let mut lobe = Lobe::new(
        vec![n1, n2],
        vec![Tract::new(TractConfig::default()), Tract::new(TractConfig::default())],
        vec![],
        1,
    );
    lobe.step(&[0.0, 0.0], 0.0, 0.0, false);
    assert_eq!(lobe.neurons[0].vars.output, 1.0);
    assert_eq!(lobe.neurons[1].vars.output, 0.0);
}
