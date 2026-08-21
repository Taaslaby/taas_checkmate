//! # Панели и контент вкладок (`app::shell::panels`)
//!
//! Переиспользуемые компоненты раскладки и контент вкладок рабочего
//! пространства коннектома:
//!
//! * [`section_header`] — сворачиваемый заголовок секции (иконка + заголовок
//!   SemiBold + шеврон) — общий для попапов рейла и карточек;
//! * [`ConnectomeMode::dock_tab_plots`] / [`ConnectomeMode::dock_tab_sim`] —
//!   контент вкладок «Плоттер» и «Симуляция» единого дерева egui_dock.
//!
//! Само дерево (и перетаскивание вкладок) владеет оболочка — см.
//! `ConnApp::dock_area` в `crate::app`; режим отдаёт только контент вкладок.

use eframe::egui;

use elegance::Button as ElegButton;

use egui_phosphor::regular as ic;

use crate::app::connectome::ConnectomeMode;
use crate::app::tokens::{self, icon_label};

/// Сворачиваемый заголовок секции: шеврон + иконка + заголовок SemiBold.
/// Состояние `open` живёт у вызывающего (персистентно, где нужно).
/// Возвращает `true`, если секцию свернули/развернули кликом.
pub(crate) fn section_header(ui: &mut egui::Ui, icon: &str, title: &str, open: &mut bool) -> bool {
    let mut toggled = false;
    ui.horizontal(|ui| {
        let chev = if *open {
            ic::CARET_DOWN
        } else {
            ic::CARET_RIGHT
        };
        if ui.button(tokens::icon(chev, tokens::ICON_XS)).clicked() {
            *open = !*open;
            toggled = true;
        }
        ui.label(tokens::heading(tokens::icon_label(
            icon,
            title,
            tokens::ICON_SM,
        )));
    });
    toggled
}

impl ConnectomeMode {
    /// Вкладка «Плоттер»: графики последнего результата симуляции + трейс
    /// «Конструктора» (он живёт всегда). Без результата — живая библиотека
    /// плоттеров (что можно включить), а не пустая заглушка.
    pub(crate) fn dock_tab_plots(&mut self, ui: &mut egui::Ui) {
        self.render_builder_scope(ui);
        if let Some(r) = &self.sim.result {
            self.render_sim_plots(ui, r);
        } else {
            self.render_plot_library(ui);
        }
    }

    /// Вкладка «Симуляция»: статус прогонов + управление живым потоком.
    pub(crate) fn dock_tab_sim(&mut self, ui: &mut egui::Ui) {
        if !self.sim.status.is_empty() {
            ui.label(&self.sim.status);
        }
        if !self.live.status.is_empty() {
            ui.label(egui::RichText::new(&self.live.status).weak());
        }
        if self.live.stream.is_some() {
            ui.label(
                egui::RichText::new(format!(
                    "живой поток · шаг {}/{}",
                    self.live.steps_done, self.sim.steps
                ))
                .weak(),
            );
            if ui
                .add(ElegButton::new(icon_label(
                    ic::STOP,
                    "■ Остановить поток",
                    tokens::ICON_SM,
                )))
                .clicked()
            {
                self.stop_live();
            }
        } else {
            ui.weak("Параметры и запуск — карточка «Симуляция» в инспекторе справа.");
        }
    }
}
