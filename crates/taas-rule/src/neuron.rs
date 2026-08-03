//! Нейрон: переменные + программа + состояние VM.

use crate::opcode::Rule;
use crate::rng::Rng;
use crate::tract::Learning;
use crate::vars::NeuronVars;
use crate::vm::{RuleVm, VmContext, VmOutcome};

/// Нейрон с геном-программируемой динамикой.
///
/// Динамика полностью определяется программой [`Rule`]: интерпретатор
/// `RuleVm` исполняет её каждый такт, читая и записывая [`NeuronVars`].
///
/// Состояние VM (`tendRate`) хранится в самом нейроне и **живёт между
/// шагами** — используйте [`Neuron::step`], если нужен персистентный
/// тендинг (в отличие от свободной `vm::run_program`, которая создаёт
/// свежее состояние на каждый вызов).
#[derive(Clone, Debug)]
pub struct Neuron {
    /// Переменные нейрона (STATE, INPUT, OUTPUT, ...).
    pub vars: NeuronVars,
    /// Программа, определяющая поведение.
    pub rule: Rule,
    /// Состояние интерпретатора (аккумулятор + tendRate).
    pub vm: RuleVm,
}

impl Neuron {
    /// Нейрон с пустыми переменными и заданной программой.
    pub fn new(rule: Rule) -> Self {
        Self {
            vars: NeuronVars::default(),
            rule,
            vm: RuleVm::default(),
        }
    }

    /// Нейрон с заданными переменными и программой.
    pub fn with_vars(rule: Rule, vars: NeuronVars) -> Self {
        Self {
            vars,
            rule,
            vm: RuleVm::default(),
        }
    }

    /// Исполнить программу один такт, сохраняя состояние VM (tendRate).
    ///
    /// Возвращает спайковый результат и флаг WTA.
    pub fn step(
        &mut self,
        chemicals: &[f32],
        source_id: u32,
        target_id: u32,
        learning: Option<&mut Learning>,
        rng: &mut Rng,
    ) -> VmOutcome {
        let mut ctx = VmContext {
            vars: &mut self.vars,
            dendrite: None,
            chemicals,
            source_id,
            target_id,
            learning,
            rng,
        };
        self.vm.execute(&self.rule, &mut ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::{OpCode, Operand};
    use crate::vm::run_program;

    #[test]
    fn test_neuron_default_program() {
        // Пустая программа — нейрон молчит.
        let mut n = Neuron::new(Rule::new());
        n.vars.input = 5.0;
        let out = run_program(&n.rule, &mut n.vars, None, &[], 0, 0, None, &mut Rng::new(1));
        assert!(!out.fired);
        assert_eq!(n.vars.output, 0.0);
    }

    #[test]
    fn test_neuron_with_vars() {
        let n = Neuron::with_vars(
            Rule::new(),
            NeuronVars {
                state: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(n.vars.state, 0.5);
    }

    #[test]
    fn test_neuron_step_persists_tend_rate() {
        // LeakageRate задаёт tendRate один раз — через Neuron::step он живёт
        // между тактами (в отличие от run_program, создающей свежую VM).
        let rule = Rule::from_ops(vec![OpCode::LeakageRate(Operand::Value(0.5)), OpCode::Stop]);
        let mut n = Neuron::new(rule);
        n.vars.input = 1.0;
        n.step(&[], 0, 0, None, &mut Rng::new(1));
        assert!(
            (n.vm.tend_rate - 0.5).abs() < 1e-6,
            "tendRate должен жить между шагами: {}",
            n.vm.tend_rate
        );
    }
}
