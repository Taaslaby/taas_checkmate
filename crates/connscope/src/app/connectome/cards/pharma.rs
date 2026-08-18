//! # Карточка фармакологии (`cards::pharma`)
//!
//! Препарат + концентрация → Хилл-маппинг в множитель проводимости/синапса.
//! Работает только с `PharmaState` (узкая сигнатура — не требует `ConnApp`).
//! Данные — `resources/pharma/drugs.toml`, математика — `core::pharma`.

use eframe::egui::{self, Color32};

use elegance::{Checkbox, Slider as ElegSlider};

use egui_phosphor::regular as ic;

use crate::app::connectome::state::PharmaState;
use crate::app::tokens;

/// Чистая свободная функция фармакологии: работает только с `PharmaState`.
pub(crate) fn render_pharma_card(ui: &mut egui::Ui, pharma: &mut PharmaState) {
    // Owned-строки собираем заранее: внутри карточки мы мутируем `pharma`,
    // а библиотеку нельзя держать взаймы.
    let names: Vec<String> = pharma
        .library
        .drugs
        .iter()
        .map(|d| {
            format!(
                "{}  ·  {}  ·  {}",
                d.name,
                d.target.label(),
                d.effect.label()
            )
        })
        .collect();
    let mut idx = pharma.drug.min(names.len().saturating_sub(1));
    let mut active = pharma.active;
    let mut logc = pharma.log_conc;

    crate::app::widgets::section_card(ui, ic::PILL, "Фармакология", |ui| {
        egui::ComboBox::from_id_salt("pharma_drug")
            .selected_text(names[idx].clone())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (i, lbl) in names.iter().enumerate() {
                    if ui.selectable_value(&mut idx, i, lbl).changed() {
                        pharma.drug = i;
                    }
                }
            });
        if ui
            .add(Checkbox::new(&mut active, "применить препарат"))
            .changed()
        {
            pharma.active = active;
        }
        // Концентрация — log₁₀ шкала (0.01 µM .. 1000 µM), значение
        // показываем уже в µM (фармакология всегда логарифмична).
        ui.add(
            ElegSlider::new(&mut logc, -2.0..=3.0)
                .label("концентрация")
                .decimals(1)
                .value_fmt(|v| format!("{:.2} µM", 10f64.powf(v))),
        );
        pharma.log_conc = logc;

        ui.add_space(tokens::SP_XS);
        ui.separator();

        let d = &pharma.library.drugs[idx];
        let conc = pharma.conc_um();
        let occ = d.occupancy(conc);
        let scale = d.scale(conc);
        // Депрессант = холодный, активатор = тёплый.
        let (status, col) = if d.effect.is_depressing() {
            (
                format!("блокирует {:.0}%", occ * 100.0),
                Color32::from_rgb(225, 120, 120),
            )
        } else {
            (
                format!("усиливает до {:.2}×", scale),
                Color32::from_rgb(120, 220, 150),
            )
        };
        ui.label(
            egui::RichText::new(format!("{}  →  {}  (×{:.2})", d.name, status, scale))
                .strong()
                .color(col),
        );
        ui.label(
            egui::RichText::new(format!(
                "мишень: {} · {:.0}% связан",
                d.target.label(),
                occ * 100.0
            ))
            .weak()
            .small(),
        );
        ui.label(egui::RichText::new(&d.mechanism).small());
        ui.label(
            egui::RichText::new(format!("ref: {}", d.reference))
                .weak()
                .small(),
        );
        if let Some(note) = &d.note {
            ui.label(
                egui::RichText::new(format!("⚠ {note}"))
                    .weak()
                    .small()
                    .color(Color32::from_rgb(225, 180, 90)),
            );
        }
    });
}
