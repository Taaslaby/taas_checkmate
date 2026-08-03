//! Переменные нейрона и дендрита — именованные слоты.

/// Слоты переменных нейрона (8 штук).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeuronVar {
    /// Аккумулятор состояния (медленная переменная).
    State,
    /// Вход (синаптический ток, собранный трактом).
    Input,
    /// Выход (спайковый сигнал).
    Output,
    Third,
    Fourth,
    Fifth,
    Sixth,
    /// Фактор роста — управляет миграцией дендритов.
    Ngf,
}

impl NeuronVar {
    /// Количество переменных нейрона.
    pub const COUNT: usize = 8;

    /// Индекс слота в SoA-массиве.
    pub fn index(self) -> usize {
        match self {
            NeuronVar::State => 0,
            NeuronVar::Input => 1,
            NeuronVar::Output => 2,
            NeuronVar::Third => 3,
            NeuronVar::Fourth => 4,
            NeuronVar::Fifth => 5,
            NeuronVar::Sixth => 6,
            NeuronVar::Ngf => 7,
        }
    }

    /// Слот по индексу.
    pub fn from_index(i: usize) -> Option<Self> {
        Some(match i {
            0 => NeuronVar::State,
            1 => NeuronVar::Input,
            2 => NeuronVar::Output,
            3 => NeuronVar::Third,
            4 => NeuronVar::Fourth,
            5 => NeuronVar::Fifth,
            6 => NeuronVar::Sixth,
            7 => NeuronVar::Ngf,
            _ => return None,
        })
    }
}

/// Набор из 8 переменных нейрона.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NeuronVars {
    pub state: f32,
    pub input: f32,
    pub output: f32,
    pub third: f32,
    pub fourth: f32,
    pub fifth: f32,
    pub sixth: f32,
    pub ngf: f32,
}

impl NeuronVars {
    /// Прочитать слот.
    pub fn get(&self, v: NeuronVar) -> f32 {
        match v {
            NeuronVar::State => self.state,
            NeuronVar::Input => self.input,
            NeuronVar::Output => self.output,
            NeuronVar::Third => self.third,
            NeuronVar::Fourth => self.fourth,
            NeuronVar::Fifth => self.fifth,
            NeuronVar::Sixth => self.sixth,
            NeuronVar::Ngf => self.ngf,
        }
    }

    /// Записать слот.
    pub fn set(&mut self, v: NeuronVar, x: f32) {
        match v {
            NeuronVar::State => self.state = x,
            NeuronVar::Input => self.input = x,
            NeuronVar::Output => self.output = x,
            NeuronVar::Third => self.third = x,
            NeuronVar::Fourth => self.fourth = x,
            NeuronVar::Fifth => self.fifth = x,
            NeuronVar::Sixth => self.sixth = x,
            NeuronVar::Ngf => self.ngf = x,
        }
    }
}

/// Слоты переменных дендрита (8 штук).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DendriteVar {
    /// Краткосрочный вес (рабочая связь).
    WeightSt,
    /// Долгосрочный вес (консолидированная память).
    WeightLt,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    /// Сила — «нежелание» дендрита мигрировать.
    Strength,
}

impl DendriteVar {
    /// Количество переменных дендрита.
    pub const COUNT: usize = 8;

    /// Индекс слота.
    pub fn index(self) -> usize {
        match self {
            DendriteVar::WeightSt => 0,
            DendriteVar::WeightLt => 1,
            DendriteVar::Second => 2,
            DendriteVar::Third => 3,
            DendriteVar::Fourth => 4,
            DendriteVar::Fifth => 5,
            DendriteVar::Sixth => 6,
            DendriteVar::Strength => 7,
        }
    }

    /// Слот по индексу.
    pub fn from_index(i: usize) -> Option<Self> {
        Some(match i {
            0 => DendriteVar::WeightSt,
            1 => DendriteVar::WeightLt,
            2 => DendriteVar::Second,
            3 => DendriteVar::Third,
            4 => DendriteVar::Fourth,
            5 => DendriteVar::Fifth,
            6 => DendriteVar::Sixth,
            7 => DendriteVar::Strength,
            _ => return None,
        })
    }
}

/// Набор из 8 переменных дендрита.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DendriteVars {
    pub weight_st: f32,
    pub weight_lt: f32,
    pub second: f32,
    pub third: f32,
    pub fourth: f32,
    pub fifth: f32,
    pub sixth: f32,
    pub strength: f32,
}

impl DendriteVars {
    /// Прочитать слот.
    pub fn get(&self, v: DendriteVar) -> f32 {
        match v {
            DendriteVar::WeightSt => self.weight_st,
            DendriteVar::WeightLt => self.weight_lt,
            DendriteVar::Second => self.second,
            DendriteVar::Third => self.third,
            DendriteVar::Fourth => self.fourth,
            DendriteVar::Fifth => self.fifth,
            DendriteVar::Sixth => self.sixth,
            DendriteVar::Strength => self.strength,
        }
    }

    /// Записать слот.
    pub fn set(&mut self, v: DendriteVar, x: f32) {
        match v {
            DendriteVar::WeightSt => self.weight_st = x,
            DendriteVar::WeightLt => self.weight_lt = x,
            DendriteVar::Second => self.second = x,
            DendriteVar::Third => self.third = x,
            DendriteVar::Fourth => self.fourth = x,
            DendriteVar::Fifth => self.fifth = x,
            DendriteVar::Sixth => self.sixth = x,
            DendriteVar::Strength => self.strength = x,
        }
    }
}

/// Ограничение всех сохраняемых значений в [-1, 1] — числовая стабильность.
pub fn bound_m11(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_roundtrip() {
        let mut n = NeuronVars::default();
        for v in [NeuronVar::State, NeuronVar::Input, NeuronVar::Output, NeuronVar::Ngf] {
            n.set(v, 0.42);
            assert_eq!(n.get(v), 0.42);
        }
        let mut d = DendriteVars::default();
        for v in [DendriteVar::WeightSt, DendriteVar::WeightLt, DendriteVar::Strength] {
            d.set(v, -0.7);
            assert_eq!(d.get(v), -0.7);
        }
    }

    #[test]
    fn test_from_index() {
        for i in 0..8 {
            assert_eq!(NeuronVar::from_index(i).map(NeuronVar::index), Some(i));
            assert_eq!(DendriteVar::from_index(i).map(DendriteVar::index), Some(i));
        }
        assert_eq!(NeuronVar::from_index(8), None);
    }

    #[test]
    fn test_bound_m11() {
        assert_eq!(bound_m11(5.0), 1.0);
        assert_eq!(bound_m11(-5.0), -1.0);
        assert_eq!(bound_m11(0.3), 0.3);
    }
}
