//! Интерпретатор VM правил: аккумулятор + tendRate.

use crate::opcode::{Operand, OpCode, Rule, Target};
use crate::rng::Rng;
use crate::tract::Learning;
use crate::vars::{bound_m11, DendriteVars, NeuronVars};

/// Результат исполнения программы за один такт.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VmOutcome {
    /// Нейрон спайкнул (OUTPUT установлен в 1).
    pub fired: bool,
    /// Программа запросила WTA в доле.
    pub wta: bool,
}

/// Контекст исполнения: всё, что VM может читать и писать, кроме аккумулятора.
pub struct VmContext<'a> {
    /// Переменные нейрона (цель динамики).
    pub vars: &'a mut NeuronVars,
    /// Переменные дендрита (если правило исполняется в контексте дендрита).
    pub dendrite: Option<&'a mut DendriteVars>,
    /// Химические концентрации тела.
    pub chemicals: &'a [f32],
    /// id источника (пре-нейрона).
    pub source_id: u32,
    /// id цели (пост-нейрона).
    pub target_id: u32,
    /// Параметры обучения тракта (могут меняться опкодами обучения).
    pub learning: Option<&'a mut Learning>,
    /// Генератор случайности.
    pub rng: &'a mut Rng,
}

/// Состояние интерпретатора: аккумулятор + скорость тендинга.
///
/// `tendRate` живёт между шагами (задаётся `LeakageRate`/`SetTendRate`),
/// аккумулятор переинициализируется входом на каждом шаге.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RuleVm {
    pub acc: f32,
    pub tend_rate: f32,
}

/// Удобная обёртка: собрать контекст, исполнить, вернуть результат.
///
/// ⚠️ Создаёт **свежее** состояние VM на каждый вызов — `tendRate` не
/// персистится между тактами. Для персистентного тендинга используйте
/// [`crate::neuron::Neuron::step`] или [`crate::lobe::Lobe`].
#[allow(clippy::too_many_arguments)]
pub fn run_program(
    rule: &Rule,
    vars: &mut NeuronVars,
    dendrite: Option<&mut DendriteVars>,
    chemicals: &[f32],
    source_id: u32,
    target_id: u32,
    learning: Option<&mut Learning>,
    rng: &mut Rng,
) -> VmOutcome {
    let mut vm = RuleVm::default();
    let mut ctx = VmContext {
        vars,
        dendrite,
        chemicals,
        source_id,
        target_id,
        learning,
        rng,
    };
    vm.execute(rule, &mut ctx)
}

impl RuleVm {
    /// Исполнить программу один такт.
    pub fn execute(&mut self, rule: &Rule, ctx: &mut VmContext<'_>) -> VmOutcome {
        // Аккумулятор стартует со входа нейрона.
        self.acc = ctx.vars.input;

        let code = rule.code();
        let mut i = 0usize;
        let mut fired = false;
        let mut wta = false;

        while i < code.len() {
            let op = code[i];
            let mut skip_next = false;
            let mut jump: Option<usize> = None;
            let mut stop = false;

            match op {
                OpCode::Stop => stop = true,
                OpCode::Goto(n) => jump = (n > i).then_some(n.saturating_sub(1)),
                OpCode::IfZeroGoto(n) => {
                    if self.acc == 0.0 {
                        jump = (n > i).then_some(n.saturating_sub(1));
                    }
                }
                OpCode::IfNZeroGoto(n) => {
                    if self.acc != 0.0 {
                        jump = (n > i).then_some(n.saturating_sub(1));
                    }
                }
                OpCode::IfNegativeGoto(n) => {
                    if self.acc < 0.0 {
                        jump = (n > i).then_some(n.saturating_sub(1));
                    }
                }
                OpCode::IfPositiveGoto(n) => {
                    if self.acc > 0.0 {
                        jump = (n > i).then_some(n.saturating_sub(1));
                    }
                }
                OpCode::IfZeroStop => {
                    if self.acc == 0.0 {
                        stop = true;
                    }
                }
                OpCode::IfNZeroStop => {
                    if self.acc != 0.0 {
                        stop = true;
                    }
                }

                // Условные пропуски: пропустить следующую инструкцию, если ложно.
                OpCode::IfEqualTo(o) => skip_next = self.acc != self.resolve(ctx, o),
                OpCode::IfNotEqualTo(o) => skip_next = self.acc == self.resolve(ctx, o),
                OpCode::IfGreaterThan(o) => skip_next = self.acc <= self.resolve(ctx, o),
                OpCode::IfLessThan(o) => skip_next = self.acc >= self.resolve(ctx, o),
                OpCode::IfGreaterThanOrEqualTo(o) => skip_next = self.acc < self.resolve(ctx, o),
                OpCode::IfLessThanOrEqualTo(o) => skip_next = self.acc > self.resolve(ctx, o),
                OpCode::IfZero(o) => skip_next = self.resolve(ctx, o) != 0.0,
                OpCode::IfNonZero(o) => skip_next = self.resolve(ctx, o) == 0.0,
                OpCode::IfPositive(o) => skip_next = self.resolve(ctx, o) <= 0.0,
                OpCode::IfNegative(o) => skip_next = self.resolve(ctx, o) >= 0.0,
                OpCode::IfNonNegative(o) => skip_next = self.resolve(ctx, o) < 0.0,
                OpCode::IfNonPositive(o) => skip_next = self.resolve(ctx, o) > 0.0,

                // Арифметика аккумулятора.
                OpCode::Add(o) => self.acc += self.resolve(ctx, o),
                OpCode::Subtract(o) => self.acc -= self.resolve(ctx, o),
                OpCode::SubtractFrom(o) => self.acc = self.resolve(ctx, o) - self.acc,
                OpCode::MultiplyBy(o) => self.acc *= self.resolve(ctx, o),
                OpCode::DivideBy(o) => {
                    let d = self.resolve(ctx, o);
                    if d != 0.0 {
                        self.acc /= d;
                    }
                }
                OpCode::DivideInto(o) => {
                    let a = self.acc;
                    if a != 0.0 {
                        self.acc = self.resolve(ctx, o) / a;
                    }
                }
                OpCode::MinIntoAccumulator(o) => self.acc = self.acc.min(self.resolve(ctx, o)),
                OpCode::MaxIntoAccumulator(o) => self.acc = self.acc.max(self.resolve(ctx, o)),
                OpCode::NegateOperandIntoAccumulator(o) => self.acc = -self.resolve(ctx, o),
                OpCode::AbsValueIntoAccumulator(o) => self.acc = self.resolve(ctx, o).abs(),
                OpCode::DistanceTo(o) => self.acc = (self.acc - self.resolve(ctx, o)).abs(),
                OpCode::FlipAccumulatorAround(o) => self.acc = self.resolve(ctx, o) - self.acc,
                OpCode::BoundInZeroOne => self.acc = self.acc.clamp(0.0, 1.0),
                OpCode::BoundInMinusOnePlusOne => self.acc = self.acc.clamp(-1.0, 1.0),

                // Тендинг.
                OpCode::SetTendRate(o) => self.tend_rate = self.resolve(ctx, o).abs(),
                OpCode::LeakageRate(o) => self.tend_rate = self.resolve(ctx, o),
                OpCode::TendAccumulatorToOperandAtTendRate(o) => {
                    let v = self.resolve(ctx, o);
                    let r = self.tend_rate;
                    self.acc = self.acc * (1.0 - r) + v * r;
                }
                OpCode::TendToAndStoreIn(t) => {
                    let cur = match t {
                        Target::Neuron(v) => ctx.vars.get(v),
                        Target::Dendrite(v) => {
                            ctx.dendrite.as_ref().map(|d| d.get(v)).unwrap_or(0.0)
                        }
                    };
                    let r = self.tend_rate;
                    let value = self.acc * (1.0 - r) + cur * r;
                    self.store(ctx, t, value);
                }

                // Динамика нейрона.
                OpCode::NominalThreshold(o) => {
                    let t = self.resolve(ctx, o);
                    if ctx.vars.input < t {
                        ctx.vars.input = 0.0;
                    }
                }
                OpCode::InputGain(o) => ctx.vars.input *= self.resolve(ctx, o),
                OpCode::RestState(o) => {
                    let v = self.resolve(ctx, o);
                    let r = self.tend_rate;
                    ctx.vars.input = ctx.vars.input * (1.0 - r) + v * r;
                }
                OpCode::Persistence(o) => {
                    let p = self.resolve(ctx, o);
                    ctx.vars.state = ctx.vars.input * (1.0 - p) + ctx.vars.state * p;
                }
                OpCode::SignalNoise(o) => {
                    let a = self.resolve(ctx, o);
                    ctx.vars.state += a * ctx.rng.next_f32();
                }
                OpCode::WinnerTakesAll => wta = true,

                // Обучение.
                OpCode::SetShortToLongRate(o) => {
                    let v = self.resolve(ctx, o);
                    if let Some(l) = ctx.learning.as_deref_mut() {
                        l.st_to_lt_rate = v;
                    }
                }
                OpCode::SetLongToShortRate(o) => {
                    let v = self.resolve(ctx, o);
                    if let Some(l) = ctx.learning.as_deref_mut() {
                        l.lt_to_st_rate = v;
                    }
                }
                OpCode::SetRewardThreshold(o) => {
                    let v = self.resolve(ctx, o);
                    if let Some(l) = ctx.learning.as_deref_mut() {
                        l.reward_threshold = v;
                    }
                }
                OpCode::SetRewardChemical(c) => {
                    if let Some(l) = ctx.learning.as_deref_mut() {
                        l.reward_chemical = c as usize;
                    }
                }
                OpCode::SetPunishmentChemical(c) => {
                    if let Some(l) = ctx.learning.as_deref_mut() {
                        l.punishment_chemical = c as usize;
                    }
                }

                // Выход.
                OpCode::StoreAccumulatorInto(t) => self.store(ctx, t, self.acc),
                OpCode::Fire(t) => {
                    let th = self.resolve(ctx, t);
                    fired = ctx.vars.state > th;
                    ctx.vars.output = if fired { 1.0 } else { 0.0 };
                }
            }

            if stop {
                break;
            }
            if let Some(n) = jump {
                i = n;
            } else if skip_next {
                i += 1;
            }
            i += 1;
        }

        VmOutcome { fired, wta }
    }

    /// Разрешить операнд в значение.
    fn resolve(&self, ctx: &mut VmContext<'_>, op: Operand) -> f32 {
        match op {
            Operand::Accumulator => self.acc,
            Operand::NeuronVar(v) => ctx.vars.get(v),
            Operand::DendriteVar(v) => ctx.dendrite.as_ref().map(|d| d.get(v)).unwrap_or(0.0),
            Operand::Rnd => ctx.rng.next_f32(),
            Operand::Zero => 0.0,
            Operand::One => 1.0,
            Operand::Value(v) => v,
            Operand::NegValue(v) => -v,
            Operand::ValueTimes10(v) => v * 10.0,
            Operand::ValueDiv10(v) => v / 10.0,
            Operand::ValueTimes248(v) => v * 248.0,
            Operand::ChemicalBySource(i) => {
                let n = ctx.chemicals.len();
                if n == 0 {
                    0.0
                } else {
                    ctx.chemicals[((i as usize) + ctx.source_id as usize) % n]
                }
            }
            Operand::ChemicalByTarget(i) => {
                let n = ctx.chemicals.len();
                if n == 0 {
                    0.0
                } else {
                    ctx.chemicals[((i as usize) + ctx.target_id as usize) % n]
                }
            }
            Operand::NeuronId => ctx.source_id as f32,
            Operand::DendriteId => ctx.target_id as f32,
        }
    }

    /// Сохранить значение в цель с зажатием в [-1, 1].
    fn store(&self, ctx: &mut VmContext<'_>, t: Target, value: f32) {
        let value = bound_m11(value);
        match t {
            Target::Neuron(v) => ctx.vars.set(v, value),
            Target::Dendrite(v) => {
                if let Some(d) = ctx.dendrite.as_deref_mut() {
                    d.set(v, value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::Operand;
    use crate::opcode::{OpCode, Rule, Target};
    use crate::vars::NeuronVar;

    fn run(ops: Vec<OpCode>, input: f32) -> (NeuronVars, VmOutcome) {
        let mut vars = NeuronVars {
            input,
            ..Default::default()
        };
        let out = run_program(&Rule::from_ops(ops), &mut vars, None, &[], 0, 0, None, &mut Rng::new(1));
        (vars, out)
    }

    #[test]
    fn test_accumulator_starts_from_input() {
        let (vars, _) = run(
            vec![OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third))],
            0.5,
        );
        assert_eq!(vars.third, 0.5);
    }

    #[test]
    fn test_store_bounds_m11() {
        let (vars, _) = run(
            vec![
                OpCode::Add(Operand::Value(100.0)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            100.0,
        );
        assert_eq!(vars.third, 1.0);
    }

    #[test]
    fn test_tending_relaxation() {
        // tend_rate = 0.5; STATE = акк·0.5 + STATE·0.5
        let (vars, _) = run(
            vec![
                OpCode::SetTendRate(Operand::Value(0.5)),
                OpCode::TendToAndStoreIn(Target::Neuron(NeuronVar::State)),
            ],
            0.8,
        );
        assert!((vars.state - 0.4).abs() < 1e-6); // 0.8·0.5 + 0·0.5
    }

    #[test]
    fn test_conditional_skip() {
        // акк=0.2, IfZero → ложь (0.2 != 0) → следующая инструкция пропускается
        let (vars, _) = run(
            vec![
                OpCode::IfZero(Operand::Accumulator),
                OpCode::Add(Operand::Value(10.0)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert_eq!(vars.third, 0.2); // Add(10) пропущен
    }

    #[test]
    fn test_forward_only_goto() {
        // Goto(2) вперёд → пропустить Add; акк = 0.2
        let (vars, _) = run(
            vec![
                OpCode::Goto(2),
                OpCode::Add(Operand::Value(10.0)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert_eq!(vars.third, 0.2);
    }

    #[test]
    fn test_chemical_operand() {
        let mut vars = NeuronVars {
            input: 0.0,
            ..Default::default()
        };
        let chemicals = [0.1, 0.2, 0.3, 0.4];
        // ChemicalBySource(1): chemicals[(1 + src=2) % 4] = chemicals[3] = 0.4
        let out = run_program(
            &Rule::from_ops(vec![
                OpCode::Add(Operand::ChemicalBySource(1)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ]),
            &mut vars,
            None,
            &chemicals,
            2,
            0,
            None,
            &mut Rng::new(1),
        );
        assert!((vars.third - 0.4).abs() < 1e-6);
        assert!(!out.fired);
    }

    #[test]
    fn test_backward_goto_ignored() {
        // Goto(0) — назад (0 > 0 ложно) → игнорируется, исполняется Add.
        // 0.2 + 0.5 = 0.7 (в [-1,1], store не зажимает).
        let (vars, _) = run(
            vec![
                OpCode::Goto(0),
                OpCode::Add(Operand::Value(0.5)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert!((vars.third - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_comparison_skips() {
        // акк = 0.2: IfGreaterThan(0.5) — ложь → пропустить Add, store 0.2.
        let (vars, _) = run(
            vec![
                OpCode::IfGreaterThan(Operand::Value(0.5)),
                OpCode::Add(Operand::Value(0.5)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert!((vars.third - 0.2).abs() < 1e-6);
        // акк = 0.2: IfLessThanOrEqualTo(0.2) — истина → Add исполняется: 0.7.
        let (vars, _) = run(
            vec![
                OpCode::IfLessThanOrEqualTo(Operand::Value(0.2)),
                OpCode::Add(Operand::Value(0.5)),
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert!((vars.third - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_zero_stop() {
        // акк = 0 → IfZeroStop останавливает до Store.
        let (vars, _) = run(
            vec![
                OpCode::IfZeroStop,
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.0,
        );
        assert_eq!(vars.third, 0.0);
        // акк = 0.2 → IfZeroStop пропускает, Store исполняется.
        let (vars, _) = run(
            vec![
                OpCode::IfZeroStop,
                OpCode::StoreAccumulatorInto(Target::Neuron(NeuronVar::Third)),
            ],
            0.2,
        );
        assert_eq!(vars.third, 0.2);
    }

    #[test]
    fn test_fire_threshold() {
        // Без динамики STATE=0 → порог 0.5 не пробит.
        let (_, out) = run(vec![OpCode::Fire(Operand::Value(0.5))], 1.0);
        assert!(!out.fired);
        // STATE накапливается через Persistence → пробивает.
        let mut vars = NeuronVars {
            input: 1.0,
            ..Default::default()
        };
        let rule = Rule::from_ops(vec![
            OpCode::Persistence(Operand::Value(0.4)),
            OpCode::Fire(Operand::Value(0.5)),
        ]);
        let mut fired_any = false;
        for _ in 0..10 {
            let out = run_program(&rule, &mut vars, None, &[], 0, 0, None, &mut Rng::new(1));
            if out.fired {
                fired_any = true;
            }
        }
        assert!(fired_any);
    }
}
