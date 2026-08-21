//! Реестр OpKind — порядок сериализации, палитра, имена регистров.

use pillars::api::formula::{OpKind, I_SYN, TEMP0};

/// Все ops в стабильном порядке (индекс сериализации).
pub(crate) const ALL_KINDS: &[OpKind] = &[
    OpKind::State,
    OpKind::Stimulus,
    OpKind::Const,
    OpKind::Leak,
    OpKind::Ohmic,
    OpKind::Add,
    OpKind::Mul,
    OpKind::Div,
    OpKind::CmpGe,
    OpKind::Blend,
    OpKind::GateStep,
    OpKind::Sigmoid,
    OpKind::Exp,
    OpKind::FireReset,
    OpKind::RefrClamp,
    OpKind::RefrTimer,
    OpKind::Scope,
    OpKind::CmpGt, // appended last — keeps indices stable
];

pub(crate) fn kind_index(kind: OpKind) -> u8 {
    ALL_KINDS.iter().position(|&k| k == kind).unwrap_or(0) as u8
}

pub(crate) fn kind_at(i: u8) -> Option<OpKind> {
    ALL_KINDS.get(i as usize).copied()
}

/// Ops в палитре (State — фиксированная системная нода, не в палитре).
pub(crate) fn palette_kinds() -> &'static [OpKind] {
    &ALL_KINDS[1..]
}

/// Человекочитаемое имя регистра.
pub(crate) fn reg_name(r: usize) -> String {
    if r < 4 {
        format!("s{r}")
    } else if r == I_SYN {
        "i_syn".to_string()
    } else {
        format!("t{}", r - TEMP0)
    }
}
