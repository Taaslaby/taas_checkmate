//! # Классическое обусловливание на программных нейронах
//!
//! Демонстрация возможностей `taas-rule`:
//!
//! 1. **Baseline** — сигнал в одиночку: слабая связь, реакции нет.
//! 2. **Conditioning** — сигнал + безусловный стимул + reward: ST-вес растёт,
//!    LT консолидирует (двухконтурная память).
//! 3. **Test** — сигнал в одиночку: нейрон спайкает! (условный рефлекс)
//! 4. **Extinction** — сигнал без reward: ST стирается (рефлекс гаснет),
//!    LT удерживает память.
//!
//! Запуск: `cargo run -p taas-rule --example conditioning --release`

use taas_rule::*;

/// Программа выходного нейрона: порог входа → утечка → покой → постоянство → спайк.
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

/// Доля: один выходной нейрон, тракт с двумя дендритами.
///
/// - дендрит 0: сигнал (ST/LT = 0.1) — учится
/// - дендрит 1: безусловный стимул (ST/LT = 1.0) — врождённый
fn make_lobe() -> Lobe {
    let tract = Tract::with_dendrites(
        TractConfig::default(),
        vec![
            Dendrite::new(0, 0.1, 0.1), // сигнал — слабый, учится
            Dendrite::new(1, 1.0, 1.0), // стимул — сильный, врождённый
        ],
    );
    Lobe::new(vec![Neuron::new(output_program())], vec![tract], vec![], 42)
}

/// Прогон одной фазы. Возвращает (число спайков, (ST, LT) дендрита 0).
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
        let fired = lobe.step(&activity, reward, 0.0, learning);
        if fired[0] {
            spikes += 1;
        }
    }
    (spikes, lobe.tracts[0].weight(0).unwrap())
}

fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  taas-rule — классическое обусловливание на программных");
    println!("  нейронах (VM правил + двухконтурные ST/LT веса)");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("   [сигнал] ──(ST/LT 0.1)──▶ [выход]   ← учится");
    println!("   [стимул] ──(ST/LT 1.0)──▶ [выход]   ← врождённо");
    println!();

    let mut lobe = make_lobe();

    // 1. Baseline: сигнал без стимула → реакции нет (связь слаба).
    let (baseline, w0) = run_phase(&mut lobe, 200, true, false, 0.0, false);

    // 2. Conditioning: сигнал + стимул + reward → ST растёт, LT консолидирует.
    let (_train, _wt) = run_phase(&mut lobe, 600, true, true, 1.0, true);

    // 3. Test: сигнал в одиночку, обучение выключено → чистый условный рефлекс.
    let (test, w1) = run_phase(&mut lobe, 200, true, false, 0.0, false);

    // 4. Extinction: сигнал без reward, обучение включено → ST тает, LT помнит.
    let (ext, w2) = run_phase(&mut lobe, 200, true, false, 0.0, true);

    println!("───────────────────────── ФАЗЫ ───────────────────────────────");
    println!("  Baseline    : сигнал без стимула  → спайков: {baseline:>4}  (связь слаба)");
    println!("  Conditioning: сигнал + стимул + reward → ST растёт, LT консолидирует");
    println!("  Test        : сигнал в одиночку   → спайков: {test:>4}  (условный рефлекс!)");
    println!("  Extinction  : сигнал без reward    → спайков: {ext:>4}  (рефлекс гаснет)");
    println!();
    println!("───────────────────────── ВЕСА (сигнал→выход) ────────────────");
    println!("  после обучения: ST = {:.3}  LT = {:.3}", w1.0, w1.1);
    println!("  после угасания: ST = {:.3}  LT = {:.3}", w2.0, w2.1);
    println!();
    println!("  Двухконтурная память: ST тает без reward ({:.3} → {:.3}),", w0.0, w2.0);
    println!("  а LT удерживает след ({:.3} → {:.3}).", w1.1, w2.1);
    println!();

    // Итоговая проверка концепта.
    assert_eq!(baseline, 0, "baseline: без обучения реакции нет");
    assert!(test > 0, "test: после conditioning сигнал обязан спайкать");
    assert!(
        ext > 0 && ext < test,
        "extinction: рефлекс гаснет постепенно: {ext} < {test}"
    );
    assert!(w2.0 < w1.0 * 0.5, "ST обязан резко упасть при угасании");
    assert!(w2.1 > w2.0, "LT держит память дольше ST");

    println!("✅ КОНЦЕПТ ДОКАЗАН: программа нейрона + ST/LT веса + reward-химия");
    println!("   порождают условный рефлекс, угасание и долговременный след.");
}
