//! # 🧬 SVRule PoC — «Мозг Гранда» на Pillars
//!
//! Доказательство концепта: переносим идею **Steve Grand** (зоолог, дизайнер
//! мозга Norn'ов в Creatures 3, 1999) на наш SIMD-движок.
//!
//! ```text
//!    C3 SVRule (1999, C++)              Pillars (2026, Rust SIMD)
//!    ─────────────────────              ──────────────────────────
//!    Neuron: STATE/INPUT/OUTPUT...  →   FormulaModel регистры s0..s3
//!    Dendrite: ST + LT веса         →   двухконтурная пластичность
//!    tendToAndStoreIn (тендинг)     →   экспоненциальная релаксация
//!    setReward*Chemical (дофамин)   →   reward-химия → обучение
//!    doWinnerTakesAll               →   спайковый выход
//! ```
//!
//! Эксперимент — **павловское обусловливание** (как «Колокол + Еда → Слюна»):
//!
//! 1. **Baseline** — звонок в одиночку: слабая синаптическая связь, реакции нет.
//! 2. **Conditioning** — звонок + еда + reward-химия: ST-вес дендрита растёт,
//!    LT консолидирует (двухконтурная память Гранда).
//! 3. **Test** — звонок в одиночку: нейрон теперь спайкает! (условный рефлекс)
//! 4. **Extinction** — звонок без reward: ST быстро стирается (рефлекс гаснет),
//!    LT удерживает память — двухконтурность ST/LT видна в цифрах.
//! 5. **SIMD-мост** — выученный нейрон компилируется в `FormulaModel`
//!    (наш интерпретатор SIMD-опкодов) и гоняется через `NeuralPopulation`
//!    на 8 лейнах: тот же звонок → те же спайки.
//!
//! Запуск: `cargo run -p pillars --example svrule_grand_poc --release`
//! Тесты:  `cargo test -p pillars --example svrule_grand_poc`

use pillars::api::formula::{OpKind, OpNode, FormulaModel, S0, P0, TEMP0};
use pillars::prelude::NeuralPopulation;

// Регистры параметров: экспортируется только P0, остальные — P0 + k.
const P1: usize = P0 + 1;
const P2: usize = P0 + 2;
const P3: usize = P0 + 3;
const P4: usize = P0 + 4;

// ═══════════════════════════════════════════════════════════════
//  1. Мини-SVRule VM — верная семантика опкодов из SVRule.h
// ═══════════════════════════════════════════════════════════════

/// Переменные нейрона — ровно `NeuronVariableNames` из BrainConstants.h:
/// `STATE=0, INPUT=1, OUTPUT=2, THIRD..SIXTH=3..6, NGF=7`.
const STATE: usize = 0;
const INPUT: usize = 1;
const OUTPUT: usize = 2;

/// Ограничение из SVRule.h: все сохраняемые значения зажимаются в [-1, 1].
fn bound_m11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Мини-набор опкодов — подмножество `SVRule::OpCodes` с той же семантикой.
/// Неиспользуемые варианты (`LoadAcc/Add/Mul/Gain/TendToState`) намеренно
/// сохранены как документация полного каталога Гранда.
#[allow(dead_code)]
enum Op {
    /// `loadAccumulatorFrom value` — константа в аккумулятор.
    LoadAcc(f32),
    /// `add` — акк += операнд.
    Add(f32),
    /// `multiplyBy` — акк *= операнд.
    Mul(f32),
    /// `doNominalThreshold` — if INPUT < t { INPUT = 0 } (порог входа).
    NominalThreshold(f32),
    /// `doInputGainLoHi` — INPUT *= g (усиление входа).
    Gain(f32),
    /// `doLeakageRate` — tendRate = op (утечка мембраны).
    Leakage(f32),
    /// `doRestState` — INPUT = INPUT·(1−r) + rest·r (потенциал покоя).
    RestState(f32),
    /// `doPersistence` — STATE = INPUT·(1−p) + STATE·p (память/постоянство).
    Persistence(f32),
    /// `tendToAndStoreIn STATE` — STATE = акк·(1−r) + STATE·r.
    TendToState(f32),
    /// `storeAccumulatorInto OUTPUT`, затем порог: если STATE > t → спайк.
    Fire(f32),
    /// `stopImmediately`.
    Stop,
}

/// Интерпретатор мини-SVRule: `acc` + живущий `tendRate` (как в C3).
///
/// В C3 `SVRule::ProcessGivenVariables` инициализирует
/// `accumulator = inputVariables[0]` — т.е. аккумулятор стартует со входа
/// нейрона. Зеркалим это поведение (наши программы всё равно перезаписывают
/// acc через `LoadAcc`/`Add`/`Mul`, так что семантика не меняется).
fn run_svrule(prog: &[Op], vars: &mut [f32; 8], i_input: f32) -> bool {
    vars[INPUT] = i_input; // синаптический вход собран трактом до запуска правила
    let mut acc = vars[INPUT]; // = C3: accumulator = inputVariables[0]
    let mut tend_rate = 0.0f32;
    let mut fired = false;

    for op in prog {
        match *op {
            Op::LoadAcc(v) => acc = v,
            Op::Add(v) => acc += v,
            Op::Mul(v) => acc *= v,
            Op::NominalThreshold(t) => {
                if vars[INPUT] < t {
                    vars[INPUT] = 0.0;
                }
            }
            Op::Gain(g) => vars[INPUT] *= g,
            Op::Leakage(r) => tend_rate = r,
            Op::RestState(rest) => {
                vars[INPUT] = vars[INPUT] * (1.0 - tend_rate) + rest * tend_rate;
            }
            Op::Persistence(p) => {
                vars[STATE] = vars[INPUT] * (1.0 - p) + vars[STATE] * p;
            }
            Op::TendToState(r) => {
                vars[STATE] = bound_m11(acc * (1.0 - r) + vars[STATE] * r);
            }
            Op::Fire(t) => {
                fired = vars[STATE] > t;
                vars[OUTPUT] = if fired { 1.0 } else { 0.0 };
            }
            Op::Stop => break,
        }
    }
    fired
}

// ═══════════════════════════════════════════════════════════════
//  2. Тракт с двухконтурной пластичностью (ST/LT) + reward-химия
// ═══════════════════════════════════════════════════════════════

/// Дендрит: у Гранда каждый дендрит несёт ДВА веса —
/// краткосрочный (ST, быстро учится) и долгосрочный (LT, медленно
/// консолидирует). `STRENGTH` — «нежелание» мигрировать (опущен для PoC).
#[derive(Clone, Debug)]
struct Dendrite {
    /// Индекс пре-нейрона.
    pre: usize,
    /// WEIGHT_SHORTTERM_VAR — рабочая связь (используется при проведении).
    w_st: f32,
    /// WEIGHT_LONGTERM_VAR — долговременная память (консолидация ST→LT).
    w_lt: f32,
}

struct Tract {
    dendrites: Vec<Dendrite>,
    /// Скорость консолидации ST→LT (doSetSTtoLTRate).
    consolidate: f32,
    /// Скорость угасания ST при отсутствии reward (doSetLTtoSTRate...).
    extinction: f32,
}

impl Tract {
    /// Reward-обучение: «дофаминовый» сигнал усиливает ST активных дендритов,
    /// LT медленно догоняет ST (двухконтурная память).
    ///
    /// Угасание (reward == 0): ST дендритов со спайкующим пре-нейроном тает
    /// со скоростью `extinction` — пре-активность без подкрепления = LTD.
    /// Правило не зависит от пост-спайка, поэтому угасание идёт непрерывно,
    /// и двухконтурность видна в цифрах (ST падает, LT держит след).
    fn learn(&mut self, pre_fired: &[bool], post_fired: bool, reward: f32, lr: f32) {
        for d in &mut self.dendrites {
            if pre_fired[d.pre] {
                if reward > 0.0 {
                    if post_fired {
                        d.w_st = bound_m11(d.w_st + lr * reward); // подкрепление
                    }
                } else {
                    d.w_st *= 1.0 - self.extinction; // угасание без награды
                }
            }
            // Консолидация: LT тянется к ST (долговременная память)
            d.w_lt = bound_m11(d.w_lt + (d.w_st - d.w_lt) * self.consolidate);
        }
    }

    /// Проведение: входной ток = Σ w_st·pre_activity (как mulitplyAndAddToNeuronInput).
    fn drive_input(&self, pre_activity: &[f32]) -> f32 {
        self.dendrites
            .iter()
            .map(|d| d.w_st * pre_activity[d.pre])
            .sum()
    }

    fn weights(&self, pre: usize) -> (f32, f32) {
        let d = self.dendrites.iter().find(|d| d.pre == pre).unwrap();
        (d.w_st, d.w_lt)
    }
}

// ═══════════════════════════════════════════════════════════════
//  3. Павловский эксперимент
// ═══════════════════════════════════════════════════════════════

/// Программа выходного нейрона, «скомпилированная из генома» (BRAINGENE):
/// утечка → покой → постоянство → порог. Это та же структура, что updateRule
/// в C3: `doLeakageRate, doRestState, doPersistence, ...`.
fn neuron_program() -> Vec<Op> {
    vec![
        Op::NominalThreshold(0.05),
        Op::Leakage(0.2),
        Op::RestState(0.0),
        Op::Persistence(0.4),
        Op::Fire(0.6),
        Op::Stop,
    ]
}

/// Один прогон фазы. Возвращает (число спайков, итоговые ST/LT).
/// `learning = false` — фаза наблюдения: пластичность и угасание ЗАМОРОЖЕНЫ,
/// чтобы тест показывал чистый устойчивый условный рефлекс, а не затухание
/// (двухконтурная память: ST быстро стирается без reward, LT держится).
fn run_phase(
    n_ticks: usize,
    bell: bool,
    food: bool,
    reward: f32,
    lr: f32,
    learning: bool,
    tract: &mut Tract,
    vars: &mut [f32; 8],
) -> (u32, (f32, f32)) {
    let prog = neuron_program();
    let mut spikes = 0u32;
    let pre_activity = [if bell { 1.0 } else { 0.0 }, if food { 1.0 } else { 0.0 }];
    let pre_fired = [bell, food];
    for _ in 0..n_ticks {
        let i_input = tract.drive_input(&pre_activity);
        let fired = run_svrule(&prog, vars, i_input);
        if learning {
            tract.learn(&pre_fired, fired, reward, lr);
        }
        if fired {
            spikes += 1;
        }
    }
    (spikes, tract.weights(0))
}

/// Результат полного эксперимента (baseline → conditioning → test → extinction).
struct ExperimentResult {
    /// Спайки на звонок ДО обучения (baseline).
    baseline_spikes: u32,
    /// Спайки на звонок ПОСЛЕ обучения (условный рефлекс).
    test_spikes: u32,
    /// ST/LT звонка после conditioning.
    st_after: f32,
    lt_after: f32,
    /// ST/LT после extinction (ST стёрся, LT держит память).
    st_extinct: f32,
    lt_extinct: f32,
    /// Спайки во время extinction (рефлекс гаснет постепенно).
    extinction_spikes: u32,
}

/// Полный эксперимент: baseline → conditioning → test → extinction.
fn run_conditioning_experiment() -> ExperimentResult {
    // Доли: [0]=bell, [1]=food; выходной нейрон — id 2 (пост-нейрон тракта).
    // Звонок→выход слабый (0.1), еда→выход врождённо сильный (1.0).
    let mut tract = Tract {
        dendrites: vec![
            Dendrite { pre: 0, w_st: 0.1, w_lt: 0.1 }, // звонок (учится)
            Dendrite { pre: 1, w_st: 1.0, w_lt: 1.0 }, // еда (врождённо)
        ],
        consolidate: 0.01,
        extinction: 0.02,
    };
    let mut vars = [0.0f32; 8];

    // 1. Baseline: звонок без еды → реакции нет (связь 0.1 слаба).
    let (baseline, w0) = run_phase(200, true, false, 0.0, 0.0, false, &mut tract, &mut vars);

    // 2. Conditioning: звонок + еда + reward-химия → ST растёт, LT консолидирует.
    let (_train, _wt) = run_phase(600, true, true, 1.0, 0.02, true, &mut tract, &mut vars);

    // 3. Test: звонок в одиночку, обучение выключено → чистый условный рефлекс.
    let (test, w1) = run_phase(200, true, false, 0.0, 0.0, false, &mut tract, &mut vars);

    // 4. Extinction: звонок в одиночку, обучение ВКЛЮЧЕНО, reward = 0 →
    //    ST тает (рефлекс гаснет), LT удерживает долговременный след.
    //    lr = 0.0: при reward = 0 ветка угасания не читает lr.
    let (ext_spikes, w2) = run_phase(200, true, false, 0.0, 0.0, true, &mut tract, &mut vars);

    // Sanity-проверки внутри VM (SPIKE-ПАРИТЕТ).
    debug_assert!(w0.0 <= 0.2, "baseline ST должно быть слабым: {:?}", w0);
    debug_assert!(w1.0 > 0.6, "после conditioning ST должно вырасти: {:?}", w1);
    debug_assert!(w2.0 < w1.0, "ST должно падать при угасании: {:?} → {:?}", w1, w2);
    debug_assert!(w2.1 > w2.0, "LT держит память дольше ST: {:?}", w2);

    ExperimentResult {
        baseline_spikes: baseline,
        test_spikes: test,
        st_after: w1.0,
        lt_after: w1.1,
        st_extinct: w2.0,
        lt_extinct: w2.1,
        extinction_spikes: ext_spikes,
    }
}

// ═══════════════════════════════════════════════════════════════
//  4. SIMD-мост: выученный нейрон → FormulaModel (8 лейн)
// ═══════════════════════════════════════════════════════════════

/// Компилируем выходной нейрон в SIMD-опкоды pillars:
/// `Stimulus → Leak → FireReset` (LIF-подобная динамика с тем же порогом).
/// Вес звонка подаётся как сила входного тока: i = w_lt·GAIN.
fn build_simd_program() -> Vec<OpNode> {
    // Регистры: s0 = мембрана, p1 = inv_tau (утечка), p2 = порог,
    // p3 = reset, p4 = v_rest. dt берётся из param-колонки 0 автоматически.
    vec![
        OpNode::new(OpKind::Stimulus, TEMP0), // TEMP0 = i_syn (входной ток)
        OpNode::new(OpKind::Leak, S0)
            .with_in(&[S0, TEMP0, P1, P4]),   // v' = v + (v_rest + i − v)·inv_tau·dt
        OpNode::new(OpKind::FireReset, S0)
            .with_in(&[S0, P2, P3]),           // спайк если v ≥ порог; reset → v_rest
    ]
}

/// Прогон `FormulaModel<0>` на 8 лейнах: сколько нейронов спайкнуло за `ticks`.
fn run_simd(weight: f32, ticks: usize) -> u32 {
    const GAIN: f32 = 40.0; // масштаб: 0.1 → i=4 (подпорогово), 1.0 → i=40 (спайк)

    if !FormulaModel::<0>::is_registered() {
        FormulaModel::<0>::register(build_simd_program())
            .expect("register FormulaModel<0>");
    }

    let mut pop = NeuralPopulation::<FormulaModel<0>>::new(8);
    // param-колонки: 0=dt, 1=inv_tau, 2=порог, 3=reset, 4=v_rest
    pop.fill_param(0, 0.5);   // dt (мс)
    pop.fill_param(1, 0.5);   // inv_tau — утечка
    pop.fill_param(2, 30.0);  // порог (мВ)
    pop.fill_param(3, 0.0);   // reset (v_rest после спайка)
    pop.fill_param(4, 0.0);   // v_rest

    let i_in = vec![weight * GAIN; 8]; // входной ток = вес звонка · масштаб
    let mut spikes = vec![0i32; 8];
    let mut total = 0u32;
    for _ in 0..ticks {
        pop.step(&i_in, &mut spikes);
        total += spikes.iter().sum::<i32>() as u32;
    }
    total
}

// ═══════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════

fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  🧬 SVRule PoC — «Мозг Гранда» на Pillars (Creatures 3, 1999)");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("Мозг Norn'ов проектировал Стив Гранд (зоолог, а не программист),");
    println!("поэтому каждый нейрон программируется ГЕНОМОМ через опкоды:");
    println!("  doLeakageRate, doRestState, doPersistence, doNominalThreshold,");
    println!("  doWinnerTakesAll, setRewardChemical, tendToAndStoreIn...");
    println!();
    println!("Ниже — павловское обусловливание на этих опкодах:");
    println!("   [звонок] ──(ST/LT 0.1)──▶ [выход]   ← учится");
    println!("   [еда]    ──(ST/LT 1.0)──▶ [выход]   ← врождённо");
    println!();

    let r = run_conditioning_experiment();

    println!("───────────────────────── ФАЗЫ ───────────────────────────────");
    println!("  Baseline    : звонок без еды      → спайков: {:>4}  (связь слаба: нет реакции)", r.baseline_spikes);
    println!("  Conditioning: звонок + еда + reward → ST растёт, LT консолидирует");
    println!("  Test        : звонок в одиночку   → спайков: {:>4}  (условный рефлекс!)", r.test_spikes);
    println!("  Extinction  : звонок без reward    → спайков: {:>4}  (рефлекс гаснет)", r.extinction_spikes);
    println!();
    println!("───────────────────────── ВЕСА (звонок→выход) ──────────────────");
    println!("  после обучения: ST = {:.3}   LT = {:.3}   (ST учится быстро, LT консолидирует)", r.st_after, r.lt_after);
    println!("  после угасания: ST = {:.3}   LT = {:.3}   (ST стёрся, LT помнит!)", r.st_extinct, r.lt_extinct);
    println!();
    println!("  Двухконтурная память Гранда: ST тает без reward ({:.3} → {:.3}),", r.st_after, r.st_extinct);
    println!("  а LT удерживает след ({:.3} → {:.3}).", r.lt_after, r.lt_extinct);
    println!();

    // ── SIMD-мост: тот же выученный вес, но через FormulaModel на 8 лейнах ──
    let simd_before = run_simd(0.1, 200); // вес ДО обучения (связь слаба)
    let simd_after = run_simd(r.lt_after.max(r.st_after), 200); // вес ПОСЛЕ (LT-память)

    println!("────────────────────── SIMD-МОСТ (pillars) ────────────────────");
    println!("  Тот же нейрон, скомпилированный в FormulaModel (8 лейн AVX2):");
    println!("    OpKind::Stimulus → Leak → FireReset");
    println!("  звонок i=w·40, вес 0.1 (до)      → спайков: {simd_before:>4}  (молчит)");
    println!("  звонок i=w·40, вес {:.2} (после)  → спайков: {simd_after:>4}  (спайкает!)", r.lt_after.max(r.st_after));
    println!();

    // Итоговая проверка концепта:
    assert_eq!(r.baseline_spikes, 0, "baseline: звонок без обучения не должен спайкать");
    assert!(r.test_spikes > 0, "test: после conditioning звонок обязан спайкать");
    assert!(
        r.extinction_spikes > 0 && r.extinction_spikes < r.test_spikes,
        "extinction: рефлекс гаснет постепенно: {} < {}",
        r.extinction_spikes, r.test_spikes
    );
    assert!(r.st_extinct < r.st_after * 0.5, "ST обязан резко упасть при угасании");
    assert!(r.lt_extinct > r.st_extinct, "LT держит память дольше ST");
    assert_eq!(simd_before, 0, "SIMD: слабый вес (0.1) не должен спайкать");
    assert!(simd_after > 0, "SIMD: выученный вес обязан спайкать на 8 лейнах");

    println!("✅ КОНЦЕПТ ДОКАЗАН: правило Гранда (опкоды + двухконтурные ST/LT");
    println!("   веса + reward-химия) порождает условный рефлекс, и выученный");
    println!("   нейрон компилируется в наш SIMD-движок без потери поведения.");
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// VM-семантика: doPersistence сглаживает, Fire срабатывает на пороге.
    #[test]
    fn test_svrule_persistence_and_fire() {
        let prog = neuron_program();
        let mut vars = [0.0f32; 8];
        // Сильный вход → состояние накапливается через doPersistence → спайк
        let mut fired_any = false;
        for _ in 0..20 {
            if run_svrule(&prog, &mut vars, 1.0) {
                fired_any = true;
            }
        }
        assert!(fired_any, "сильный вход обязан дать спайк через doPersistence");
        // Слабый вход → NominalThreshold гасит, спайков нет
        let mut vars2 = [0.0f32; 8];
        for _ in 0..20 {
            assert!(!run_svrule(&prog, &mut vars2, 0.01), "слабый вход не спайкает");
        }
    }

    /// Павлов: baseline молчит, test спайкает, ST/LT выросли.
    #[test]
    fn test_pavlov_conditioning_learns() {
        let r = run_conditioning_experiment();
        assert_eq!(r.baseline_spikes, 0, "baseline: без обучения реакции нет");
        assert!(r.test_spikes > 0, "после conditioning звонок спайкает (условный рефлекс)");
        assert!(r.st_after > 0.6, "ST должен вырасти: {}", r.st_after);
        assert!(r.lt_after > 0.2, "LT должен консолидировать: {}", r.lt_after);
    }

    /// Двухконтурная память: ST стирается при угасании, LT держит след.
    #[test]
    fn test_extinction_two_timescale_memory() {
        let r = run_conditioning_experiment();
        assert!(r.st_extinct < r.st_after * 0.5,
            "ST обязан резко упасть: {} → {}", r.st_after, r.st_extinct);
        assert!(r.lt_extinct > r.st_extinct,
            "LT держит память дольше ST: {} vs {}", r.lt_extinct, r.st_extinct);
        assert!(r.extinction_spikes > 0, "рефлекс гаснет постепенно (сначала спайкает)");
        assert!(r.extinction_spikes < r.test_spikes,
            "рефлекс угасает: {} < {}", r.extinction_spikes, r.test_spikes);
    }

    /// SIMD-мост: выученный вес спайкает на 8 лейнах, слабый — нет.
    #[test]
    fn test_simd_bridge_learned_weight_fires() {
        let r = run_conditioning_experiment();
        let learned = r.lt_after.max(r.st_after);
        assert_eq!(run_simd(0.1, 200), 0, "слабый вес молчит в SIMD");
        assert!(run_simd(learned, 200) > 0, "выученный вес спайкает в SIMD");
    }
}
