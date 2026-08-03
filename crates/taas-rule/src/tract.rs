//! Тракт: пучок дендритов + обучение (reward/punishment химия).

use crate::dendrite::Dendrite;
use crate::vars::bound_m11;

/// Параметры обучения, изменяемые опкодами программы нейрона.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Learning {
    /// Скорость консолидации ST → LT (doSetSTtoLTRate).
    pub st_to_lt_rate: f32,
    /// Скорость возврата LT → ST (doSetLTtoSTRate).
    pub lt_to_st_rate: f32,
    /// Порог награды.
    pub reward_threshold: f32,
    /// Индекс химического вещества-награды.
    pub reward_chemical: usize,
    /// Индекс химического вещества-наказания.
    pub punishment_chemical: usize,
}

impl Default for Learning {
    fn default() -> Self {
        Self {
            st_to_lt_rate: 0.01,
            lt_to_st_rate: 0.0,
            reward_threshold: 0.5,
            reward_chemical: 0,
            punishment_chemical: 1,
        }
    }
}

impl Learning {
    /// Концентрация химического вещества-награды (индекс задан опкодом
    /// [`crate::opcode::OpCode::SetRewardChemical`]).
    pub fn reward_value(&self, chemicals: &[f32]) -> f32 {
        chemicals.get(self.reward_chemical).copied().unwrap_or(0.0)
    }

    /// Концентрация химического вещества-наказания (индекс задан опкодом
    /// [`crate::opcode::OpCode::SetPunishmentChemical`]).
    pub fn punishment_value(&self, chemicals: &[f32]) -> f32 {
        chemicals.get(self.punishment_chemical).copied().unwrap_or(0.0)
    }
}

/// Статические скорости обучения тракта.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TractConfig {
    /// Прирост ST за такт при подкреплении.
    pub potentiation: f32,
    /// Доля ST, тающая за такт без награды (угасание).
    pub extinction: f32,
}

impl Default for TractConfig {
    fn default() -> Self {
        Self {
            potentiation: 0.02,
            extinction: 0.02,
        }
    }
}

/// Пучок дендритов, входящих в один пост-нейрон.
///
/// Проводит ток от пре-нейронов и обучает дендриты по двухконтурной
/// схеме: подкрепление усиливает `ST`, консолидация перекачивает в `LT`,
/// отсутствие награды гасит `ST` (угасание).
#[derive(Clone, Debug)]
pub struct Tract {
    /// Входящие дендриты.
    pub dendrites: Vec<Dendrite>,
    /// Динамические параметры обучения (пишутся опкодами).
    pub learning: Learning,
    /// Статические скорости обучения.
    pub config: TractConfig,
}

impl Tract {
    /// Пустой тракт с параметрами по умолчанию.
    pub fn new(config: TractConfig) -> Self {
        Self {
            dendrites: Vec::new(),
            learning: Learning::default(),
            config,
        }
    }

    /// Тракт с готовыми дендритами.
    pub fn with_dendrites(config: TractConfig, dendrites: Vec<Dendrite>) -> Self {
        Self {
            dendrites,
            learning: Learning::default(),
            config,
        }
    }

    /// Добавить дендрит.
    pub fn add(&mut self, d: Dendrite) -> &mut Self {
        self.dendrites.push(d);
        self
    }

    /// Проведение: входной ток = Σ conduction_weight · активность пре-нейрона.
    pub fn drive_input(&self, pre_activity: &[f32]) -> f32 {
        self.dendrites
            .iter()
            .map(|d| {
                let a = pre_activity.get(d.pre).copied().unwrap_or(0.0);
                d.conduction_weight(self.learning.lt_to_st_rate) * a
            })
            .sum()
    }

    /// Обучение за один такт.
    ///
    /// - Подкрепление (`reward > threshold` и пост спайкнул): `ST += lr·reward`.
    /// - Слабый reward (`0 < reward ≤ threshold`): сигнал есть, но ниже порога
    ///   обучения — веса не меняются (нет ни усиления, ни угасания).
    /// - Наказание (`punishment > 0` при отсутствии reward, пост спайкнул):
    ///   `ST -= lr·punishment`.
    /// - Без наград (`reward ≤ 0`, пре активен): `ST *= 1 − extinction` (угасание).
    /// - Всегда: консолидация `LT += (ST − LT)·st_to_lt_rate`.
    pub fn learn(&mut self, pre_fired: &[bool], post_fired: bool, reward: f32, punishment: f32) {
        let lr = self.config.potentiation;
        let th = self.learning.reward_threshold;
        for d in &mut self.dendrites {
            let pre_fired = pre_fired.get(d.pre).copied().unwrap_or(false);
            if pre_fired {
                if reward > th {
                    if post_fired {
                        d.vars.weight_st = bound_m11(d.vars.weight_st + lr * reward);
                    }
                } else if reward > 0.0 {
                    // Подпороговый reward: обучения нет, угасания тоже.
                } else if punishment > 0.0 {
                    if post_fired {
                        d.vars.weight_st = bound_m11(d.vars.weight_st - lr * punishment);
                    }
                } else {
                    d.vars.weight_st *= 1.0 - self.config.extinction;
                }
            }
            d.vars.weight_lt = bound_m11(
                d.vars.weight_lt + (d.vars.weight_st - d.vars.weight_lt) * self.learning.st_to_lt_rate,
            );
        }
    }

    /// (ST, LT) дендрита от пре-нейрона `pre`.
    pub fn weight(&self, pre: usize) -> Option<(f32, f32)> {
        self.dendrites
            .iter()
            .find(|d| d.pre == pre)
            .map(|d| (d.vars.weight_st, d.vars.weight_lt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drive_input() {
        let mut t = Tract::new(TractConfig::default());
        t.add(Dendrite::new(0, 0.1, 0.1));
        t.add(Dendrite::new(1, 1.0, 1.0));
        let activity = [1.0f32, 1.0];
        assert!((t.drive_input(&activity) - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_learn_potentiation_and_consolidation() {
        let mut t = Tract::new(TractConfig::default());
        t.add(Dendrite::new(0, 0.1, 0.1));
        let pre = [true];
        for _ in 0..100 {
            t.learn(&pre, true, 1.0, 0.0); // reward выше порога
        }
        let (st, lt) = t.weight(0).unwrap();
        assert!(st > 0.9, "ST должен вырасти: {st}");
        assert!(lt > 0.5, "LT должен консолидировать: {lt}");
    }

    #[test]
    fn test_learn_extinction_decays_st_holds_lt() {
        let mut t = Tract::new(TractConfig::default());
        t.add(Dendrite::new(0, 1.0, 1.0));
        let pre = [true];
        // Обучим до насыщения.
        for _ in 0..500 {
            t.learn(&pre, true, 1.0, 0.0);
        }
        let (st0, lt0) = t.weight(0).unwrap();
        assert!(st0 > 0.9 && lt0 > 0.9);
        // Угасание: без награды ST тает, LT удерживает.
        for _ in 0..100 {
            t.learn(&pre, true, 0.0, 0.0);
        }
        let (st1, lt1) = t.weight(0).unwrap();
        assert!(st1 < st0 * 0.5, "ST должен упасть: {st0} → {st1}");
        assert!(lt1 > st1, "LT держит память дольше ST: {lt1} vs {st1}");
    }

    #[test]
    fn test_learn_punishment() {
        let mut t = Tract::new(TractConfig::default());
        t.add(Dendrite::new(0, 0.5, 0.5));
        let pre = [true];
        for _ in 0..50 {
            t.learn(&pre, true, 0.0, 1.0); // наказание
        }
        let (st, _) = t.weight(0).unwrap();
        assert!(st < 0.4, "ST должен уменьшиться при наказании: {st}");
    }
}
