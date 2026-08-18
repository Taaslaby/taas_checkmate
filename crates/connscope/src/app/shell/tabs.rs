//! # Ego-граф (`app::tabs`)
//!
//! Контент вкладки «Ego-граф»: большой граф на всю центральную область
//! (или в откреплённом окне): режимы раскладки (силы / координаты), Fit,
//! наведение атласа, клик по узлу = выбор нейрона.
//!
//! Сами вкладки режимов (Атлас / Ego) живут в тонком верхнем баре —
//! [`ConnApp::mode_tabs`](super::top).

use eframe::egui::{self, vec2};

use egui_graphs::{
    set_layout_state, FruchtermanReingoldState, GraphView, SettingsInteraction, SettingsNavigation,
    SettingsStyle,
};
use elegance::SegmentedControl;

use egui_phosphor::regular as ic;

use crate::app::connectome::ego::EgoView;
use crate::app::connectome::ConnectomeMode;
use crate::app::tokens::{self, icon_label};

impl ConnectomeMode {
    // ── ego-граф (большая вкладка / окно) ─────────────────────
    fn ego_view_id(&self) -> Option<String> {
        self.sel.selected.map(|s| {
            format!(
                "ego_{s}_{}",
                if self.sel.ego_layout_coords { "c" } else { "f" }
            )
        })
    }

    /// Инициализация layout при смене выбора/режима: в режиме координат
    /// позиции заданы анатомией — силы выключены (`is_running=false`),
    /// иначе свежий Fruchterman–Reingold.
    fn init_ego_layout(&mut self, ui: &mut egui::Ui, id: &Option<String>) {
        if self.sel.ego_shown_key != self.sel.selected {
            if let Some(id) = id {
                if self.sel.ego_layout_coords {
                    let state = FruchtermanReingoldState {
                        is_running: false,
                        ..Default::default()
                    };
                    set_layout_state::<FruchtermanReingoldState>(ui, state, Some(id.clone()));
                } else {
                    // Свежий FR со «спокойными» параметрами: мельче шаг,
                    // сильнее демпфирование — без дёрганья и разлёта узлов.
                    let state = FruchtermanReingoldState {
                        dt: 0.018,
                        damping: 0.42,
                        max_step: 3.0,
                        ..Default::default()
                    };
                    set_layout_state::<FruchtermanReingoldState>(ui, state, Some(id.clone()));
                }
            }
            self.sel.ego_shown_key = self.sel.selected;
        }
    }

    /// Большой ego-граф на всю центральную область (или в откреплённом окне):
    /// строка управления + GraphView на всю оставшуюся высоту.
    pub(crate) fn ego_view(&mut self, ui: &mut egui::Ui) {
        // В режиме координат граф обязан следовать проекции атласа.
        if self.sel.ego_layout_coords && self.sel.ego_shown_proj != self.camera.proj {
            self.rebuild_ego_graph();
        }
        let (nn, ne) = (
            self.sel.ego_graph.g().node_count(),
            self.sel.ego_graph.g().edge_count(),
        );
        ui.horizontal_wrapped(|ui| {
            // Режим раскладки — сегментированный переключатель.
            let mut mode = if self.sel.ego_layout_coords { 1 } else { 0 };
            ui.add(SegmentedControl::new(
                &mut mode,
                [
                    icon_label(ic::LIGHTNING, "Силы", tokens::ICON_XS),
                    icon_label(ic::MAP_TRIFOLD, "Координаты", tokens::ICON_XS),
                ],
            ));
            if (mode == 1) != self.sel.ego_layout_coords {
                self.sel.ego_layout_coords = mode == 1;
                self.sel.ego_shown_key = None; // пере-инициализировать layout
                self.rebuild_ego_graph();
            }
            if ui
                .button(icon_label(ic::ARROW_CLOCKWISE, "Fit", tokens::ICON_SM))
                .on_hover_text("Вписать граф в окно")
                .clicked()
            {
                self.ego_fit_once = true;
            }
            if ui
                .button(icon_label(ic::CROSSHAIR, "Навести атлас", tokens::ICON_SM))
                .clicked()
            {
                self.focus_ego();
            }
            ui.separator();
            ui.label(
                egui::RichText::new(format!("{nn} узлов · {ne} агрегированных связей")).strong(),
            );
            ui.label(
                egui::RichText::new("колесо — масштаб · MMB — панорама · клик — выбрать")
                    .weak()
                    .small(),
            );
        });
        ui.separator();
        if nn > 1 {
            let ego_id = self.ego_view_id();
            self.init_ego_layout(ui, &ego_id);
            // Подсветка сильных рёбер: толщина штриха ~ весу синапса.
            // Хук получает `order` ребра — рёбра добавлялись в порядке
            // `sel.ego`, так что order = индекс в таблице весов.
            let weights: std::sync::Arc<Vec<f32>> =
                std::sync::Arc::new(self.sel.ego.iter().map(|&(_, w, _)| w.abs()).collect());
            let w_ref = weights.clone();
            let w_max = weights.iter().cloned().fold(0.0f32, f32::max).max(1e-3);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Сила связи").weak().small());
                ui.label(egui::RichText::new("тонкая").weak().small());
                ui.add(egui::ProgressBar::new(0.5).desired_width(54.0).text(""));
                ui.label(egui::RichText::new("толстая").weak().small());
                ui.label(
                    egui::RichText::new(format!("max {:.1}", w_max))
                        .weak()
                        .small(),
                );
            });
            let style = SettingsStyle::new()
                // Подписи только у hovered/selected узлов — без каши из 240 ярлыков
                // (egui_graphs показывает метку при hover/selected/dragged).
                .with_labels_always(false)
                .with_edge_stroke_hook(move |selected, order, stroke, _style| {
                    let w = w_ref.get(order).copied().unwrap_or(0.0) / w_max;
                    let mut s = stroke;
                    s.width = 0.8 + 2.4 * w + if selected { 0.7 } else { 0.0 };
                    s
                });
            let mut view: EgoView = GraphView::new(&mut self.sel.ego_graph)
                .with_interactions(
                    &SettingsInteraction::new()
                        .with_node_selection_enabled(true)
                        .with_dragging_enabled(true),
                )
                .with_styles(&style)
                .with_navigations(
                    &SettingsNavigation::new().with_fit_to_screen_enabled(self.ego_fit_once),
                )
                .with_id(ego_id);
            self.ego_fit_once = false;
            ui.add_sized(ui.available_size(), &mut view);
            // Клик по узлу = выбор нейрона (кроме драйва).
            let clicked: Vec<usize> = self
                .sel
                .ego_graph
                .g()
                .node_weights()
                .filter(|n| n.selected())
                .map(|n| n.payload().idx)
                .collect();
            if let Some(&idx) = clicked.iter().find(|&&i| self.sel.selected != Some(i)) {
                self.select(idx);
            }
        } else {
            ui.weak("Выбери нейрон на атласе — его окрестность появится здесь.");
            ui.weak("Можно открыть ego-граф и из инспектора (справа).");
        }
    }

    /// Откреплённый ego-граф: обычное перетаскиваемое окно (drag за заголовок),
    /// закрытие крестиком возвращает его в таб.
    pub(crate) fn ego_window(&mut self, ctx: &egui::Context) {
        let mut open = true;
        egui::Window::new(format!(
            "{} Ego-граф — перетащи меня за заголовок",
            ic::NETWORK
        ))
        .open(&mut open)
        .default_size(vec2(860.0, 580.0))
        .resizable(true)
        .show(ctx, |ui| {
            self.ego_view(ui);
        });
        if !open {
            self.ego_undocked = false; // свернули в таб
        }
    }
}
