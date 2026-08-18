//! # Библиотека плоттеров (`app::plots`)
//!
//! Отрисовка всех графиков симуляции. Модуль разбит по ответственности:
//!
//! ```text
//! plots/
//! ├── mod.rs    стили (карточка, сетка), диспетчер render_plot_card, ConnApp
//! ├── sci.rs    чистая математика: БПФ, спектр, ISI, гистограмма, антифаза
//! ├── local.rs  графики ego-контура: растр, частоты, V(t)
//! └── full.rs   научные графики полного VNC: PSTH, спектр, тепловая, ISI, фазы
//! ```
//!
//! Включение/открепление плоттеров — модалка «Плоттеры»; откреплённые
//! рисуются в **нативных окнах ОС** (`plots_windows` через
//! `show_viewport_immediate`), остальные — рядом в нижней панели
//! (`render_sim_plots`).
//!
//! Дизайн единый: любой плот (и в панели, и в нативном окне) проходит через
//! [`render_plot_card`], поэтому карточка-обёртка и стили идентичны везде.

use eframe::egui::{self, Color32};

use egui_phosphor::regular as ic;
use egui_plot::{CoordinatesFormatter, Corner, Legend, Plot};
use elegance::{Checkbox, Modal};

use crate::app::connectome::ConnectomeMode;
use crate::app::tokens::{self, icon_label};
use crate::core::color::ColorMode;
use crate::io::data::MancData;
use crate::sim::SimOutcome;

use super::plotlib::{PlotDesc, PlotKind, CATALOG};

mod full;
mod local;
mod sci;

use full::{card_full_heatmap, card_full_isi, card_full_phase, card_full_psth, card_full_spectrum};
use local::{card_local_raster, card_local_rates, card_local_v};

// ── общие стили и обёртки ────────────────────────────────────

/// Единая высота плота (egui_plot и тепловая карта) — чтобы графики
/// в нижней панели и в нативных окнах выглядели одинаково.
const PLOT_HEIGHT: f32 = 230.0;

/// Минимальная комфортная ширина одного плота в ряду. Если доступная ширина
/// дока не вмещает столько колонок, плоты перестраиваются в столбик
/// (вертикальная ориентация), а не сжимаются «в кучку».
const MIN_PLOT_COL_WIDTH: f32 = 300.0;

/// Цвет сетки плота из текущей темы (тёмная/светлая) — плоты сливаются
/// с остальным интерфейсом, а не торчат белым прямоугольником.
fn grid_color(ui: &egui::Ui) -> Color32 {
    if ui.style().visuals.dark_mode {
        Color32::from_rgba_unmultiplied(255, 255, 255, 16)
    } else {
        Color32::from_rgba_unmultiplied(0, 0, 0, 26)
    }
}

/// Единая стилизация плота: сетка по теме, легенда справа-снизу,
/// координаты курсора слева-снизу с единицами X-оси.
fn styled_plot<'a>(ui: &egui::Ui, id: impl egui::AsId + 'a, x_unit: &'static str) -> Plot<'a> {
    Plot::new(id)
        .height(PLOT_HEIGHT)
        .grid_color(grid_color(ui))
        .legend(
            Legend::default()
                .position(Corner::RightBottom)
                .background_alpha(0.9)
                .text_style(egui::TextStyle::Small),
        )
        .coordinates_formatter(
            Corner::LeftBottom,
            CoordinatesFormatter::new(move |pos, _| format!("{:.1} {x_unit}\n{:.3}", pos.x, pos.y)),
        )
}

/// Карточка-обёртка плота: иконка + заголовок, подпись, график.
/// Делегирует общей обёртке `app::widgets::compact_card` — единый стиль
/// карточек во всём приложении (инспектор, плоттер, модалки).
fn plot_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    sub: &str,
    body: impl FnOnce(&mut egui::Ui),
) {
    crate::app::widgets::compact_card(ui, icon, title, sub, body);
}

// ── диспетчер библиотеки ─────────────────────────────────────

/// Отрисовка одного плота из библиотеки (карточка-обёртка). Несовпадение
/// рода плота и результата (`Local` vs `Full`) тихо игнорируется.
pub(crate) fn render_plot_card(
    ui: &mut egui::Ui,
    kind: PlotKind,
    data: &MancData,
    mode: ColorMode,
    r: &SimOutcome,
) {
    match (kind, r) {
        (PlotKind::LocalRaster, SimOutcome::Local(rr)) => card_local_raster(ui, rr),
        (PlotKind::LocalRates, SimOutcome::Local(rr)) => card_local_rates(ui, data, rr, mode),
        (PlotKind::LocalV, SimOutcome::Local(rr)) => card_local_v(ui, data, rr, mode),
        (PlotKind::FullPsth, SimOutcome::Full(rr)) => {
            card_full_psth(ui, rr, data.n.max(1) as f64, rr.dt_ms as f64)
        }
        (PlotKind::FullSpectrum, SimOutcome::Full(rr)) => {
            card_full_spectrum(ui, rr, rr.dt_ms as f64)
        }
        (PlotKind::FullHeatmap, SimOutcome::Full(rr)) => {
            card_full_heatmap(ui, rr, data.n.max(1) as f64, rr.dt_ms as f64)
        }
        (PlotKind::FullIsi, SimOutcome::Full(rr)) => card_full_isi(ui, rr),
        (PlotKind::FullPhase, SimOutcome::Full(rr)) => card_full_phase(ui, data, rr),
        _ => {}
    }
}

impl ConnectomeMode {
    /// Модалка «Плоттеры»: библиотека графиков симуляции — включай галочкой,
    /// открепляй в нативное окно ОС кнопкой (как модульные окна в редакторах).
    pub(crate) fn plots_library_modal(&mut self, ctx: &egui::Context) {
        let mut plots_open = self.show_plots;
        Modal::new("plots_modal", &mut plots_open)
            .heading(icon_label(ic::CHART_LINE, "Библиотека плоттеров", tokens::ICON_SM))
            .subtitle("Включай графики симуляции · галочка = показ, ⧉ = открепить в нативное окно ОС")
            .header_icon(ic::CHART_LINE)
            .max_width(680.0)
            .show(ctx, |ui| {
                // Первый блок — плоты ego-контура (локальная симуляция).
                ui.label(egui::RichText::new("Локальный контур (ego)").strong());
                for desc in CATALOG.iter().filter(|d| d.local) {
                    plot_lib_row(ui, self, desc);
                }
                ui.add_space(tokens::SP_SM);
                ui.separator();
                ui.label(egui::RichText::new("Полный прогон (научные)").strong());
                for desc in CATALOG.iter().filter(|d| !d.local) {
                    plot_lib_row(ui, self, desc);
                }
                ui.add_space(tokens::SP_SM);
                ui.label(
                    egui::RichText::new("Откреплённый плот — отдельное окно системы (его можно двигать на второй монитор); закрытие крестиком вернёт его в нижнюю панель.")
                        .weak()
                        .small(),
                );
            });
        self.show_plots = plots_open;
    }

    /// Ряд включённых (не откреплённых) плоттеров — нижняя панель.
    pub(crate) fn render_sim_plots(&self, ui: &mut egui::Ui, r: &SimOutcome) {
        let Some(data) = self.data.as_ref() else {
            return;
        };
        let local = matches!(r, SimOutcome::Local(_));
        let kinds: Vec<PlotKind> = CATALOG
            .iter()
            .filter(|d| {
                if d.local != local {
                    return false;
                }
                let st = self.plots.get(d.kind);
                st.enabled && !st.detached
            })
            .map(|d| d.kind)
            .collect();
        if kinds.is_empty() {
            return;
        }
        // Адаптивная сетка: в узкой (вертикальной) ориентации дока плоты
        // выстраиваются в столбик на всю ширину, в широкой — в ряд по столько
        // колонок, сколько помещается с комфортной шириной карточки.
        let cols = ((ui.available_width() / MIN_PLOT_COL_WIDTH).floor() as usize)
            .clamp(1, kinds.len());
        for row in kinds.chunks(cols) {
            if row.len() == 1 {
                render_plot_card(ui, row[0], data, self.filters.color_mode, r);
            } else {
                ui.columns(row.len(), |cols_ui| {
                    for (i, k) in row.iter().enumerate() {
                        render_plot_card(&mut cols_ui[i], *k, data, self.filters.color_mode, r);
                    }
                });
            }
            ui.add_space(tokens::SP_SM);
        }
    }

    /// Откреплённые плоттеры: **нативные окна ОС** (не внутри приложения).
    ///
    /// Каждый откреплённый плот рендерится в собственное системное окно через
    /// [`egui::Context::show_viewport_immediate`] (мультивьюпорт egui/eframe).
    /// Внутри рисуется ровно тот же [`render_plot_card`], что и в нижней панели —
    /// единый дизайн. Закрытие крестиком ОС (`close_requested`) возвращает плот
    /// в нижнюю панель.
    pub(crate) fn plots_windows(&mut self, ctx: &egui::Context) {
        let Some(r) = self.sim.result.as_ref() else {
            return;
        };
        let local = matches!(r, SimOutcome::Local(_));
        let Some(data) = self.data.as_ref() else {
            return;
        };
        for desc in CATALOG.iter() {
            if desc.local != local {
                continue;
            }
            let st = self.plots.get(desc.kind);
            if !st.enabled || !st.detached {
                continue;
            }
            let kind = desc.kind;
            let mode = self.filters.color_mode;
            // Стабильный id окна на плот — чтобы ОС запоминала его позицию/размер
            // между откреплениями (не плодить окна).
            let viewport_id = egui::ViewportId::from_hash_of(("neoMyelin.plot", kind as usize));
            let builder = egui::ViewportBuilder::default()
                .with_title(format!("{} {}", desc.icon, desc.name))
                .with_inner_size([640.0, 400.0])
                .with_min_inner_size([360.0, 280.0])
                .with_taskbar(true);
            // Колбэк возвращает, запросил ли пользователь закрытие окна —
            // тогда возвращаем плот в нижнюю панель (detached = false).
            let close = ctx.show_viewport_immediate(viewport_id, builder, |ui, _class| {
                render_plot_card(ui, kind, data, mode, r);
                ui.ctx().input(|i| i.viewport().close_requested())
            });
            if close {
                let mut s = self.plots.get(kind);
                s.detached = false;
                self.plots.set(kind, s);
            }
        }
    }
}

/// Одна строка библиотеки плоттеров: чекбокс включения + кнопка открепления
/// + пояснение. Состояние пишется обратно в `PlotLib` сразу.
fn plot_lib_row(ui: &mut egui::Ui, mode: &mut ConnectomeMode, desc: &PlotDesc) {
    let kind = desc.kind;
    let mut st = mode.plots.get(kind);
    let mut enabled = st.enabled;
    let mut detached = st.detached;
    ui.horizontal_wrapped(|ui| {
        if ui
            .add(Checkbox::new(
                &mut enabled,
                format!("{} {}", desc.icon, desc.name),
            ))
            .on_hover_text(desc.desc)
            .changed()
        {
            st.enabled = enabled;
        }
        // ⧉ открепить / свернуть: активное состояние = откреплён (рамка
        // акцента) — тот же визуальный язык, что у рейла и верхнего бара
        // (`widgets::icon_button`).
        let det_icon = if detached {
            ic::ARROWS_IN
        } else {
            ic::ARROWS_OUT
        };
        let hover = if detached {
            "Вернуть в нижнюю панель"
        } else {
            "Открепить в нативное окно ОС"
        };
        if crate::app::widgets::icon_button(ui, det_icon, detached, tokens::ICON_SM)
            .on_hover_text(hover)
            .clicked()
        {
            detached = !detached;
            st.detached = detached;
        }
        if detached {
            ui.label(egui::RichText::new("откреплён").weak().small());
        }
    });
    ui.label(egui::RichText::new(desc.desc).weak().small());
    mode.plots.set(kind, st);
}
