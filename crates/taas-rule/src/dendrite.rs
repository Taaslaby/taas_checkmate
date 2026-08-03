//! Дендрит: пре-нейрон + двухконтурные веса.

use crate::vars::DendriteVars;

/// Дендрит — связь от пре-нейрона к пост-нейрону.
///
/// Несёт ДВА веса: краткосрочный `ST` (рабочая связь, быстро учится) и
/// долгосрочный `LT` (консолидированная память). `STRENGTH` — «нежелание»
/// дендрита мигрировать.
#[derive(Clone, Debug)]
pub struct Dendrite {
    /// Индекс пре-нейрона.
    pub pre: usize,
    /// Переменные дендрита (ST, LT, STRENGTH, ...).
    pub vars: DendriteVars,
}

impl Dendrite {
    /// Дендрит с заданными ST/LT весами (остальное — по умолчанию).
    pub fn new(pre: usize, weight_st: f32, weight_lt: f32) -> Self {
        Self {
            pre,
            vars: DendriteVars {
                weight_st,
                weight_lt,
                ..Default::default()
            },
        }
    }

    /// Текущий проводимый вес — смесь ST и LT по скорости возврата LT→ST.
    ///
    /// При `lt_to_st_rate = 0` проводит чистый ST (быстрое обучение),
    /// при `lt_to_st_rate = 1` — чистый LT (долговременная память).
    pub fn conduction_weight(&self, lt_to_st_rate: f32) -> f32 {
        self.vars.weight_st * (1.0 - lt_to_st_rate) + self.vars.weight_lt * lt_to_st_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conduction_blend() {
        let d = Dendrite::new(3, 0.2, 0.8);
        assert!((d.conduction_weight(0.0) - 0.2).abs() < 1e-6);
        assert!((d.conduction_weight(1.0) - 0.8).abs() < 1e-6);
        assert!((d.conduction_weight(0.5) - 0.5).abs() < 1e-6);
    }
}
