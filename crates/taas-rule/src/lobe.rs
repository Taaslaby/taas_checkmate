//! Доля: решётка нейронов + тракты + WTA + обучение.

use crate::neuron::Neuron;
use crate::rng::Rng;
use crate::tract::Tract;
use crate::vm::{VmContext, VmOutcome};

/// Доля — двумерная (или просто индексированная) решётка нейронов с
/// входящими трактами. Оркестрирует такт: сбор токов → исполнение программ
/// → WTA → обучение.
#[derive(Clone, Debug)]
pub struct Lobe {
    /// Нейроны доли.
    pub neurons: Vec<Neuron>,
    /// Входящий тракт для каждого нейрона (одноимённые индексы).
    pub tracts: Vec<Tract>,
    /// Химические концентрации тела (для операндов ChemicalBy*).
    pub chemicals: Vec<f32>,
    /// Генератор случайности доли.
    pub rng: Rng,
}

impl Lobe {
    /// Доля из нейронов и трактов (длины должны совпадать).
    ///
    /// # Panics
    /// Паникует, если `neurons.len() != tracts.len()`.
    pub fn new(neurons: Vec<Neuron>, tracts: Vec<Tract>, chemicals: Vec<f32>, seed: u64) -> Self {
        assert_eq!(
            neurons.len(),
            tracts.len(),
            "количество нейронов и трактов должно совпадать"
        );
        Self {
            neurons,
            tracts,
            chemicals,
            rng: Rng::new(seed),
        }
    }

    /// Один такт симуляции.
    ///
    /// `activity` — активность пре-нейронов (внешняя или предыдущий такт).
    /// Возвращает вектор спайков (1/0) для каждого нейрона доли.
    ///
    /// Этапы:
    /// 1. Для каждого нейрона: ток из его тракта → INPUT → исполнение программы.
    /// 2. Если хотя бы один нейрон запросил WTA — победитель (макс. STATE)
    ///    получает OUTPUT = 1, остальные 0.
    /// 3. Если `learning` — обучение всех трактов по спайкам и reward.
    pub fn step(
        &mut self,
        activity: &[f32],
        reward: f32,
        punishment: f32,
        learning: bool,
    ) -> Vec<bool> {
        let n = self.neurons.len();
        let mut fired = vec![false; n];
        let mut wta_requested = vec![false; n];

        for i in 0..n {
            let input = self.tracts[i].drive_input(activity);
            let Neuron { vars, rule, vm } = &mut self.neurons[i];
            vars.input = input;
            let mut ctx = VmContext {
                vars,
                dendrite: None,
                chemicals: &self.chemicals,
                source_id: i as u32,
                target_id: i as u32,
                learning: Some(&mut self.tracts[i].learning),
                rng: &mut self.rng,
            };
            let outcome: VmOutcome = vm.execute(rule, &mut ctx);
            fired[i] = outcome.fired;
            wta_requested[i] = outcome.wta;
        }

        // WTA: победитель — нейрон с максимальным STATE среди запросивших.
        // Победитель считается спайкнувшим (fired = true) — иначе его дендриты
        // не получили бы обучения за выигрыш.
        if wta_requested.iter().any(|&w| w) {
            let best = wta_requested
                .iter()
                .enumerate()
                .filter(|(_, &w)| w)
                .max_by(|(a, _), (b, _)| {
                    self.neurons[*a]
                        .vars
                        .state
                        .total_cmp(&self.neurons[*b].vars.state)
                })
                .map(|(i, _)| i);
            if let Some(best) = best {
                for (i, neuron) in self.neurons.iter_mut().enumerate() {
                    neuron.vars.output = if i == best { 1.0 } else { 0.0 };
                }
                fired[best] = true;
            }
        }

        if learning {
            let pre_fired: Vec<bool> = activity.iter().map(|&a| a > 0.5).collect();
            for (i, tract) in self.tracts.iter_mut().enumerate() {
                tract.learn(&pre_fired, fired[i], reward, punishment);
            }
        }

        fired
    }

    /// Один такт с авто-резолюцией reward/punishment из химических
    /// концентраций тела: `reward = chemicals[learning.reward_chemical]`,
    /// `punishment = chemicals[learning.punishment_chemical]` — индексы можно
    /// задать опкодами [`OpCode::SetRewardChemical`] / `SetPunishmentChemical`.
    pub fn step_auto(&mut self, activity: &[f32], learning: bool) -> Vec<bool> {
        // Индексы берём из первого нейрона (лобальный гейтинг химией).
        let l = self.tracts[0].learning;
        let reward = l.reward_value(&self.chemicals);
        let punishment = l.punishment_value(&self.chemicals);
        self.step(activity, reward, punishment, learning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::OpCode;
    use crate::tract::TractConfig;

    #[test]
    fn test_lobe_passthrough() {
        // Нейрон: OUTPUT = INPUT (передаёт активность).
        let rule = crate::Rule::from_ops(vec![
            OpCode::StoreAccumulatorInto(crate::Target::Neuron(crate::NeuronVar::Output)),
        ]);
        let lobe = Lobe::new(vec![Neuron::new(rule)], vec![Tract::new(TractConfig::default())], vec![], 1);
        assert_eq!(lobe.neurons.len(), 1);
    }

    #[test]
    fn test_wta_picks_max_state() {
        // Два нейрона: STATE 0.9 и 0.3, оба запрашивают WTA.
        let rule = crate::Rule::from_ops(vec![OpCode::WinnerTakesAll, OpCode::Stop]);
        let n1 = Neuron::with_vars(
            rule.clone(),
            crate::NeuronVars {
                state: 0.9,
                ..Default::default()
            },
        );
        let n2 = Neuron::with_vars(
            rule,
            crate::NeuronVars {
                state: 0.3,
                ..Default::default()
            },
        );
        let mut lobe = Lobe::new(
            vec![n1, n2],
            vec![
                Tract::new(TractConfig::default()),
                Tract::new(TractConfig::default()),
            ],
            vec![],
            1,
        );
        lobe.step(&[0.0, 0.0], 0.0, 0.0, false);
        assert_eq!(lobe.neurons[0].vars.output, 1.0);
        assert_eq!(lobe.neurons[1].vars.output, 0.0);
    }
}
