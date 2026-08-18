//! # Карточка растормаживания (`cards::disin`)
//!
//! Детектор `тормоз →⊣ тормоз →⊣ нейрон` для выбранного тормозного нейрона.
//! Сборка «что показать» — в чистой [`disin_card_data`] (owned-строки цепей),
//! рендер — в [`render_disin_card`], который работает только с `DisinState`
//! и данными и возвращает команду [`DisinAction`]. Оркестратор
//! (`card_disinhibition` в `inspector.rs`) применяет команду к `self`.

use eframe::egui;

use egui_phosphor::regular as ic;

use crate::app::connectome::state::DisinState;
use crate::app::widgets::ActionTone;
use crate::io::data::MancData;

/// Команда карточки растормаживания — рендер не знает про `ConnApp`,
/// только про `DisinState` и данные; действие применяет оркестратор.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisinAction {
    /// 🔍 — запустить поиск цепей `тормоз →⊣ тормоз →⊣ нейрон`.
    Find,
}

/// Собранные строки карточки растормаживания — owned-данные для рендера.
pub(crate) struct DisinCardData {
    /// Строки найденных цепей (пусто, если цепей нет).
    pub chains: Vec<String>,
    /// Статус последнего поиска.
    pub status: String,
}

/// Чистый сборщик «что показать» для растормаживания: по данным и состоянию
/// собирает owned-строки цепей. Не трогает UI — легко тестировать.
pub(crate) fn disin_card_data(d: Option<&MancData>, disin: &DisinState) -> DisinCardData {
    let chains: Vec<String> = match d {
        Some(d) => disin
            .chains
            .iter()
            .map(|&(mid, y, w1, w2)| {
                let ty = d.type_name(y);
                format!(
                    "#{y} {} ({}) через #{mid} w={w1:.0}·{w2:.0}",
                    if ty.is_empty() { "?" } else { ty },
                    d.nt(y).name()
                )
            })
            .collect(),
        None => Vec::new(),
    };
    DisinCardData {
        chains,
        status: disin.status.clone(),
    }
}

/// Чистый рендер карточки растормаживания: работает только с `DisinState`
/// и данными (узкая сигнатура), а действие возвращает как [`DisinAction`] —
/// оркестратор применяет его к остальному состоянию (`run_disin`).
pub(crate) fn render_disin_card(
    ui: &mut egui::Ui,
    disin: &DisinState,
    d: Option<&MancData>,
) -> Option<DisinAction> {
    let card = disin_card_data(d, disin);
    let mut action = None;
    crate::app::widgets::section_card(ui, ic::LOCK_OPEN, "Растормаживание", |ui| {
        // Полной ширины — длинный текст не обрезается в панели.
        if crate::app::widgets::full_width_action(
            ui,
            ic::FUNNEL,
            "Найти тормоз→⊣тормоз→⊣нейрон",
            ActionTone::Primary,
        ) {
            action = Some(DisinAction::Find);
        }
        if !card.status.is_empty() {
            ui.label(egui::RichText::new(&card.status).weak());
        }
        if !card.chains.is_empty() {
            egui::ScrollArea::vertical()
                .max_height(ui.available_height() * 0.35)
                .show(ui, |ui| {
                    for lbl in &card.chains {
                        ui.label(lbl);
                    }
                });
        }
    });
    action
}
