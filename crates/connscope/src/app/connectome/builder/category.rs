//! Визуальные семейства нод — цвет/иконка/тинт.

use eframe::egui::Color32;
use pillars::api::formula::OpKind;

/// Семейство ноды — для раскраски хедера и проводов.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeCat {
    System,   // State / Scope — инфраструктура графа
    Source,   // Stimulus / Const — входы
    Membrane, // Leak / Ohmic / FireReset / Refr*
    Math,     // Add / Mul / Div
    Gate,     // GateStep / Sigmoid / Exp
    Logic,    // CmpGe / Blend
}

pub(crate) fn cat_of(kind: OpKind) -> NodeCat {
    use OpKind::*;
    match kind {
        State | Scope => NodeCat::System,
        Stimulus | Const => NodeCat::Source,
        Leak | Ohmic | FireReset | RefrClamp | RefrTimer => NodeCat::Membrane,
        Add | Mul | Div => NodeCat::Math,
        GateStep | Sigmoid | Exp => NodeCat::Gate,
        CmpGe | CmpGt | Blend => NodeCat::Logic,
    }
}

/// Проф muted акценты — Houdini/Nuke, не неон.
pub(crate) fn cat_color(cat: NodeCat) -> Color32 {
    match cat {
        NodeCat::System => Color32::from_rgb(128, 90, 165),
        NodeCat::Source => Color32::from_rgb(68, 155, 110),
        NodeCat::Membrane => Color32::from_rgb(75, 145, 195),
        NodeCat::Gate => Color32::from_rgb(62, 158, 148),
        NodeCat::Math => Color32::from_rgb(195, 135, 65),
        NodeCat::Logic => Color32::from_rgb(195, 88, 78),
    }
}

pub(crate) fn cat_tint(cat: NodeCat) -> Color32 {
    let c = cat_color(cat);
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 52)
}

pub(crate) fn cat_icon(cat: NodeCat) -> &'static str {
    // Phosphor — чёткие, монохром, один вес, как в остальном connscope
    match cat {
        NodeCat::System => egui_phosphor::regular::PUZZLE_PIECE,
        NodeCat::Source => egui_phosphor::regular::DOWNLOAD_SIMPLE,
        NodeCat::Membrane => egui_phosphor::regular::WAVE_SINE,
        NodeCat::Math => egui_phosphor::regular::CALCULATOR,
        NodeCat::Gate => egui_phosphor::regular::DOOR,
        NodeCat::Logic => egui_phosphor::regular::SHUFFLE,
    }
}
