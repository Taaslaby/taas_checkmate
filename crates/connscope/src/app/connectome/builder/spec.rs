//! Спецификация ноды — какие пины у каждого `OpKind`.

use pillars::api::formula::OpKind;

/// Шаблон ноды: имя + имена входных/выходных пинов.
pub(crate) struct Spec {
    pub name: &'static str,
    pub ins: &'static [&'static str],
    pub outs: &'static [&'static str],
}

/// Описание пина — для тултипа (понятный UX).
pub(crate) fn pin_desc(kind: OpKind, pin: usize, is_input: bool) -> &'static str {
    match (kind, pin, is_input) {
        (OpKind::Leak, 0, true) => "v — мембранный потенциал, мВ",
        (OpKind::Leak, 1, true) => "i — синаптический ток, мА",
        (OpKind::Leak, 2, true) => "inv_tau — 1/τ, мс⁻¹",
        (OpKind::Leak, 3, true) => "v_rest — потенциал покоя, мВ",
        (OpKind::FireReset, 0, true) => "v — потенциал",
        (OpKind::FireReset, 1, true) => "thresh — порог спайка",
        (OpKind::FireReset, 2, true) => "reset — сброс после спайка",
        (OpKind::Ohmic, 0, true) => "g — проводимость",
        (OpKind::Ohmic, 1, true) => "v — потенциал",
        (OpKind::Ohmic, 2, true) => "e — реверс-потенциал",
        (OpKind::GateStep, 0, true) => "g — текущее значение гейта",
        (OpKind::GateStep, 1, true) => "g_inf — стационар",
        (OpKind::GateStep, 2, true) => "inv_tau — скорость",
        _ => "",
    }
}

/// Шаблон для `kind` — имя и пины.
pub(crate) fn spec_of(kind: OpKind) -> Spec {
    match kind {
        OpKind::State => Spec {
            name: "State",
            ins: &[],
            outs: &["s0", "s1", "s2", "s3"],
        },
        OpKind::Stimulus => Spec {
            name: "Stimulus",
            ins: &[],
            outs: &["I"],
        },
        OpKind::Const => Spec {
            name: "Const",
            ins: &[],
            outs: &["out"],
        },
        OpKind::Leak => Spec {
            name: "Leak",
            ins: &["v", "i", "inv_tau", "v_rest"],
            outs: &["v'"],
        },
        OpKind::Ohmic => Spec {
            name: "Ohmic",
            ins: &["g", "v", "e"],
            outs: &["I"],
        },
        OpKind::Add => Spec {
            name: "Add",
            ins: &["a", "b"],
            outs: &["out"],
        },
        OpKind::Mul => Spec {
            name: "Mul",
            ins: &["a", "b"],
            outs: &["out"],
        },
        OpKind::Div => Spec {
            name: "Div",
            ins: &["a", "b"],
            outs: &["out"],
        },
        OpKind::CmpGe => Spec {
            name: "CmpGe",
            ins: &["a", "b"],
            outs: &["mask"],
        },
        OpKind::CmpGt => Spec {
            name: "CmpGt",
            ins: &["a", "b"],
            outs: &["mask"],
        },
        OpKind::Blend => Spec {
            name: "Blend",
            ins: &["mask", "a", "b"],
            outs: &["out"],
        },
        OpKind::GateStep => Spec {
            name: "GateStep",
            ins: &["g", "g_inf", "inv_tau"],
            outs: &["g'"],
        },
        OpKind::Sigmoid => Spec {
            name: "Sigmoid",
            ins: &["x"],
            outs: &["out"],
        },
        OpKind::Exp => Spec {
            name: "Exp",
            ins: &["x"],
            outs: &["out"],
        },
        OpKind::FireReset => Spec {
            name: "FireReset",
            ins: &["v", "thresh", "reset"],
            outs: &["v"],
        },
        OpKind::RefrClamp => Spec {
            name: "RefrClamp",
            ins: &["v", "refr", "reset"],
            outs: &["v"],
        },
        OpKind::RefrTimer => Spec {
            name: "RefrTimer",
            ins: &["refr", "tau_ref"],
            outs: &["refr'"],
        },
        OpKind::Scope => Spec {
            name: "Scope",
            ins: &["x"],
            outs: &[],
        },
    }
}
