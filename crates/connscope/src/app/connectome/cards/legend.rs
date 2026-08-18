//! # Карточка легенды (`cards::legend`)
//!
//! Шпаргалка цветов текущего режима окраски + реальная статистика мозга.
//! Сборка строк — в чистой [`legend_data`] (без UI), рендер — в
//! [`render_legend_card`], который не мутирует состояние (не требует
//! `&mut ConnApp`).

use eframe::egui::{self, Color32};

use elegance::CollapsingSection;

use egui_phosphor::regular as ic;

use crate::app::tokens;
use crate::core::color::ColorMode;
use crate::core::nt::Nt;
use crate::io::data::MancData;

/// Строка легенды: имя, цвет, число нейронов (`None` = без счёта, шкала).
pub(crate) type LegendRow = (String, Color32, Option<u64>);

/// Чистая свободная функция легенды: работает только с данными и режимом
/// окраски — не знает про `ConnApp`, легко тестировать и переиспользовать.
pub(crate) fn render_legend_card(ui: &mut egui::Ui, d: Option<&MancData>, mode: ColorMode) {
    let legend = legend_data(d, mode);
    crate::app::widgets::section_card(ui, ic::PALETTE, "Легенда", |ui| {
        CollapsingSection::new("legend_colors", &legend.0)
            .default_open(true)
            .show(ui, |ui| {
                ui.label(egui::RichText::new(legend.1).weak().small());
                ui.add_space(tokens::SP_XS);
                // Строки попарно в ДВЕ РАВНЫЕ колонки: `ui.columns`
                // делит ширину карточки поровну, а не по содержимому —
                // поэтому колонки не «разъезжаются» от длинных имён.
                for pair in legend.2.chunks(tokens::LEGEND_COLS) {
                    ui.columns(tokens::LEGEND_COLS, |cols| {
                        for (ci, (name, col, cnt)) in pair.iter().enumerate() {
                            crate::app::widgets::legend_swatch(&mut cols[ci], name, *col, *cnt);
                        }
                    });
                }
            });
    });
}

/// Собирает строки легенды для выбранного режима окраски. Чистая функция:
/// не трогает UI и состояние приложения — легко тестировать и переиспользовать.
pub(crate) fn legend_data(
    d: Option<&MancData>,
    mode: ColorMode,
) -> (String, &'static str, Vec<LegendRow>) {
    match mode {
        ColorMode::Nt => {
            let mut hist = [0u64; 7];
            if let Some(d) = d {
                for i in 0..d.n {
                    hist[d.nt[i] as usize] += 1;
                }
            }
            let rows = (0..7)
                .filter(|&nt| hist[nt] > 0)
                .map(|nt| {
                    let n = Nt::from_code(nt as u8);
                    (n.name().to_string(), n.color(), Some(hist[nt]))
                })
                .collect();
            (
                "Нейротрансмиттеры".into(),
                "цвет = нейромедиатор нейрона",
                rows,
            )
        }
        ColorMode::Klass => {
            let mut hist = [0u64; 8];
            if let Some(d) = d {
                for i in 0..d.n {
                    hist[d.klass[i] as usize] += 1;
                }
            }
            let rows = (0..8)
                .filter(|&k| hist[k] > 0)
                .map(|k| {
                    let n = crate::core::nt::Klass::from_code(k as u8);
                    (
                        n.name().to_string(),
                        crate::core::color::klass_color(k as u8),
                        Some(hist[k]),
                    )
                })
                .collect();
            ("Классы клеток".into(), "цвет = класс клетки", rows)
        }
        ColorMode::Type => {
            let (ntypes, top8): (usize, Vec<(String, Color32, u64)>) = match d {
                Some(d) => {
                    let mut hist = vec![0u64; d.types.len()];
                    for &ti in &d.type_idx {
                        hist[ti as usize] += 1;
                    }
                    let mut rank: Vec<usize> = (0..hist.len()).collect();
                    rank.sort_by_key(|&i| std::cmp::Reverse(hist[i]));
                    let top8 = rank
                        .iter()
                        .take(8)
                        .map(|&i| {
                            (
                                d.types[i].clone(),
                                crate::core::color::type_color(i as u16),
                                hist[i],
                            )
                        })
                        .collect();
                    (d.types.len(), top8)
                }
                None => (0, Vec::new()),
            };
            let rows = top8
                .into_iter()
                .map(|(n, c, cnt)| (n, c, Some(cnt)))
                .collect();
            (
                format!("Типы клеток — {ntypes} уникальных"),
                "цвет = тип клетки (золотой угол); ниже — топ по числу нейронов",
                rows,
            )
        }
        ColorMode::Degree => {
            let rows = ["1", "10", "100", "1К", "10К", "100К"]
                .iter()
                .enumerate()
                .map(|(p, lbl)| {
                    (
                        lbl.to_string(),
                        crate::core::color::degree_color(10usize.pow(p as u32)),
                        None,
                    )
                })
                .collect();
            (
                "Степень (исходящие связи)".into(),
                "цвет = число синапсов ИЗ нейрона: мало (синий) → много (красный)",
                rows,
            )
        }
    }
}
