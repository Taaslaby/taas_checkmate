//! Каталог инструкций VM и сборка программ.

use crate::vars::{DendriteVar, NeuronVar};

/// Максимальная длина программы нейрона (геном-кодируемая).
pub const MAX_INSTRUCTIONS: usize = 16;

/// Адрес операнда — откуда VM берёт значение.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    /// Переменная нейрона (слот).
    NeuronVar(NeuronVar),
    /// Переменная дендрита (слот).
    DendriteVar(DendriteVar),
    /// Текущее значение аккумулятора.
    Accumulator,
    /// Случайное число из [0, 1).
    Rnd,
    /// Химическая концентрация, индексированная id источника.
    ChemicalBySource(u32),
    /// Химическая концентрация, индексированная id цели.
    ChemicalByTarget(u32),
    /// Константа 0.
    Zero,
    /// Константа 1.
    One,
    /// Зашитое в инструкцию значение.
    Value(f32),
    /// −value.
    NegValue(f32),
    /// value × 10.
    ValueTimes10(f32),
    /// value ÷ 10.
    ValueDiv10(f32),
    /// value × 248.
    ValueTimes248(f32),
    /// Номер нейрона.
    NeuronId,
    /// Номер дендрита. В текущей сборке дендритные правила не исполняются
    /// (тракты обучаются кодом), поэтому резолвится в id цели (пост-нейрона).
    DendriteId,
}

/// Куда пишется результат операции.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// Слот переменных нейрона.
    Neuron(NeuronVar),
    /// Слот переменных дендрита.
    Dendrite(DendriteVar),
}

/// Каталог инструкций VM.
///
/// Условные переходы — **только вперёд** (`new_loc > current`); программы
/// линейны, без циклов. Условные пропуски пропускают ровно одну следующую
/// инструкцию, если условие ложно.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpCode {
    // ── Управление потоком ──────────────────────────────────────────
    /// Остановить исполнение.
    Stop,
    /// Безусловный переход вперёд на строку `n`.
    Goto(usize),
    /// Если аккумулятор == 0 → переход вперёд на `n`.
    IfZeroGoto(usize),
    /// Если аккумулятор != 0 → переход вперёд на `n`.
    IfNZeroGoto(usize),
    /// Если аккумулятор < 0 → переход вперёд на `n`.
    IfNegativeGoto(usize),
    /// Если аккумулятор > 0 → переход вперёд на `n`.
    IfPositiveGoto(usize),
    /// Если аккумулятор == 0 → остановиться.
    IfZeroStop,
    /// Если аккумулятор != 0 → остановиться.
    IfNZeroStop,

    // ── Условные пропуски (пропустить следующую, если условие ложно) ──
    IfEqualTo(Operand),
    IfNotEqualTo(Operand),
    IfGreaterThan(Operand),
    IfLessThan(Operand),
    IfGreaterThanOrEqualTo(Operand),
    IfLessThanOrEqualTo(Operand),
    IfZero(Operand),
    IfNonZero(Operand),
    IfPositive(Operand),
    IfNegative(Operand),
    IfNonNegative(Operand),
    IfNonPositive(Operand),

    // ── Арифметика аккумулятора ─────────────────────────────────────
    Add(Operand),
    Subtract(Operand),
    SubtractFrom(Operand),
    MultiplyBy(Operand),
    DivideBy(Operand),
    DivideInto(Operand),
    MinIntoAccumulator(Operand),
    MaxIntoAccumulator(Operand),
    NegateOperandIntoAccumulator(Operand),
    AbsValueIntoAccumulator(Operand),
    DistanceTo(Operand),
    FlipAccumulatorAround(Operand),
    BoundInZeroOne,
    BoundInMinusOnePlusOne,

    // ── Тендинг (экспоненциальная релаксация) ───────────────────────
    /// tendRate = |операнд|.
    SetTendRate(Operand),
    /// акк = акк·(1−r) + операнд·r.
    TendAccumulatorToOperandAtTendRate(Operand),
    /// цель = акк·(1−r) + цель·r.
    TendToAndStoreIn(Target),

    // ── Динамика нейрона ────────────────────────────────────────────
    /// if INPUT < операнд { INPUT = 0 }.
    NominalThreshold(Operand),
    /// INPUT *= операнд (усиление входа).
    InputGain(Operand),
    /// tendRate = операнд (утечка мембраны).
    LeakageRate(Operand),
    /// INPUT = INPUT·(1−r) + операнд·r (потенциал покоя).
    RestState(Operand),
    /// STATE = INPUT·(1−p) + STATE·p (постоянство/память).
    Persistence(Operand),
    /// STATE += операнд·Rnd() (шум сигнала).
    SignalNoise(Operand),
    /// Запросить победителя-забирает-всё в доле.
    WinnerTakesAll,

    // ── Обучение (пишут в Learning тракта) ──────────────────────────
    /// Скорость консолидации ST → LT.
    SetShortToLongRate(Operand),
    /// Скорость возврата LT → ST.
    SetLongToShortRate(Operand),
    /// Порог награды (reward threshold).
    SetRewardThreshold(Operand),
    /// Индекс химического вещества-награды.
    SetRewardChemical(u32),
    /// Индекс химического вещества-наказания.
    SetPunishmentChemical(u32),

    // ── Выход ───────────────────────────────────────────────────────
    /// Сохранить аккумулятор в цель (с зажатием в [-1, 1]).
    StoreAccumulatorInto(Target),
    /// OUTPUT = (STATE > порог) ? 1 : 0.
    Fire(Operand),
}

/// Программа нейрона — упорядоченный список инструкций (≤ 16).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rule {
    code: Vec<OpCode>,
}

impl Rule {
    /// Пустая программа.
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Программа из готового списка инструкций.
    ///
    /// # Panics
    /// Паникует, если инструкций больше [`MAX_INSTRUCTIONS`].
    pub fn from_ops(ops: Vec<OpCode>) -> Self {
        assert!(
            ops.len() <= MAX_INSTRUCTIONS,
            "программа не может превышать {MAX_INSTRUCTIONS} инструкций, получено {}",
            ops.len()
        );
        Self { code: ops }
    }

    /// Добавить инструкцию.
    ///
    /// # Panics
    /// Паникует, если программа уже заполнена.
    pub fn push(&mut self, op: OpCode) -> &mut Self {
        assert!(
            self.code.len() < MAX_INSTRUCTIONS,
            "программа уже заполнена ({MAX_INSTRUCTIONS} инструкций)"
        );
        self.code.push(op);
        self
    }

    /// Доступ к инструкциям.
    pub fn code(&self) -> &[OpCode] {
        &self.code
    }

    /// Количество инструкций.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    /// Пустая ли программа.
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_cap() {
        let mut rule = Rule::new();
        for i in 0..MAX_INSTRUCTIONS {
            rule.push(OpCode::Add(Operand::Value(i as f32)));
        }
        assert_eq!(rule.len(), MAX_INSTRUCTIONS);
    }

    #[test]
    #[should_panic]
    fn test_rule_cap_overflow() {
        let mut rule = Rule::new();
        for i in 0..=MAX_INSTRUCTIONS {
            rule.push(OpCode::Add(Operand::Value(i as f32)));
        }
    }
}
