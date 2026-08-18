//! # Статус-бар (`app::status`)
//!
//! Одна тонкая строка внизу окна (как в Blender/VS Code):
//!
//! * слева — контент режима: hover-инфо о нейроне под курсором, а без
//!   наведения — контекстная подсказка, которая следует за состоянием
//!   приложения (2D-атлас / 3D-облако / идущая симуляция);
//! * справа — FPS оболочки + счётчики режима (данные, zoom, статус симуляции).
//!
//! Рамку и FPS рисует оболочка, левую/правую части — активный режим
//! (`Mode::status_left` / `Mode::status_right`).

use eframe::egui;

use crate::app::connectome::ConnectomeMode;
use crate::app::tokens;
use crate::app::ConnApp;

impl ConnApp {
    // ── статус-бар ──────────────────────────────────────────────
    pub(crate) fn hover_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status")
            .exact_size(tokens::STATUS_BAR_H)
            .show(ui, |ui| {
                super::profiler::record("status", ui.max_rect());
                ui.horizontal_centered(|ui| {
                    let r = ui.label(egui::RichText::new(self.mode.status_left()).weak());
                    super::profiler::record("status_left", r.rect);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.ui.fps > 0.0 {
                            ui.label(egui::RichText::new(format!("{:.0} fps", self.ui.fps)).weak());
                        }
                        self.mode.status_right(ui);
                    });
                });
            });
    }
}

impl ConnectomeMode {
    /// Контекстная подсказка левого края статус-бара: следует за режимом.
    pub(crate) fn context_hint(&self) -> String {
        if self.live.stream.is_some() {
            format!(
                "живой поток · шаг {}/{} · ЛКМ — вращение · колесо — зум · [ ] — вкладки",
                self.live.steps_done, self.sim.steps
            )
        } else if self.camera.view3d {
            "ЛКМ — вращение · колесо — зум · клик — ego-граф · F — вписать".into()
        } else {
            "ЛКМ — выбрать · колесо — зум · drag — панорама · F — вписать · Ctrl+S — поиск".into()
        }
    }
}
