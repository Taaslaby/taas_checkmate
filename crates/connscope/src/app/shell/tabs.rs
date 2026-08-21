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
    set_layout_state, FruchtermanReingoldState, GraphView, MetadataFrame, SettingsInteraction,
    SettingsNavigation,
};
use elegance::SegmentedControl;

use egui_phosphor::regular as ic;

use crate::app::connectome::ego::{edge_color, EgoView};
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

    /// Инициализация layout при смене выбора/режима. Оба режима заморожены
    /// (`is_running=false`): «координаты» берут позиции из анатомии, а
    /// силовую раскладку доводит до сходимости `ego_view` (fast_forward_*).
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
                    // FR заморожен (is_running=false): раскладка доводится до
                    // сходимости один раз в ego_view через fast_forward_*,
                    // дальше граф статичен — без вечного дрожания узлов.
                    let state = FruchtermanReingoldState {
                        is_running: false,
                        dt: 0.05,
                        damping: 0.6,
                        max_step: 10.0,
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
            let fresh = self.sel.ego_shown_key != self.sel.selected;
            self.init_ego_layout(ui, &ego_id);
            // Силовую раскладку доводим до сходимости один раз (на кадре
            // пересборки) и замораживаем: узлы не «живут своей жизнью» при
            // каждой отрисовке, а граф не дёргается.
            if fresh && !self.sel.ego_layout_coords {
                EgoView::fast_forward_until_stable_budgeted_force_run(
                    ui,
                    &mut self.sel.ego_graph,
                    0.3,  // среднее смещение узла за шаг, ниже — остановка
                    2000, // предел числа шагов
                    200,  // предел времени, мс
                    ego_id.clone(),
                );
            }
            // Легенда: цвет ребра = полярность пресинаптического НТ,
            // толщина = вес связи (рисует EgoEdgeShape — см. ego.rs).
            let w_max = self
                .sel
                .ego
                .iter()
                .map(|&(_, w, _, _)| w.abs())
                .fold(0.0f32, f32::max)
                .max(1e-3);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Связи").weak().small());
                legend_dot(ui, edge_color(1.0));
                ui.label(egui::RichText::new("возб.").weak().small());
                legend_dot(ui, edge_color(-1.0));
                ui.label(egui::RichText::new("торм.").weak().small());
                ui.label(
                    egui::RichText::new(format!("max {:.1}", w_max))
                        .weak()
                        .small(),
                );
            });
            // Клонируем id до `.with_id` (тот забирает `ego_id` по значению),
            // чтобы после отрисовки загрузить те же zoom/pan для hover-теста.
            let meta_id = ego_id.clone();
            let mut view: EgoView = GraphView::new(&mut self.sel.ego_graph)
                .with_interactions(
                    &SettingsInteraction::new()
                        .with_node_selection_enabled(true)
                        .with_dragging_enabled(true),
                )
                .with_navigations(
                    &SettingsNavigation::new().with_fit_to_screen_enabled(self.ego_fit_once),
                )
                .with_id(ego_id);
            self.ego_fit_once = false;
            let resp = ui.add_sized(ui.available_size(), &mut view);
            // Hover по ребру: тултип с числом синапсов и типом пресинаптического
            // нейрона. egui_graphs отслеживает hover только узлов, поэтому
            // используем его же хит-тест `edge_by_screen_pos` в координатах
            // виджета (pan метаданных относителен к левому-верхнему углу).
            self.edge_hover_tooltip(ui, resp.rect.left_top(), meta_id);
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

    /// Тултип при наведении на ребро ego-графа: пресинаптический нейрон
    /// (индекс + тип), число агрегированных синапсов и направление связи.
    /// egui_graphs отслеживает hover только узлов, поэтому используем его же
    /// хит-тест `edge_by_screen_pos` в координатах виджета.
    fn edge_hover_tooltip(&self, ui: &egui::Ui, origin: egui::Pos2, meta_id: Option<String>) {
        let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) else {
            return;
        };
        // pan метаданных относителен к левому-верхнему углу виджета —
        // переводим глобальную позицию курсора в локальную.
        let meta = MetadataFrame::new(meta_id).load(ui);
        let local = pointer - origin.to_vec2();
        let Some(edge_idx) = self.sel.ego_graph.edge_by_screen_pos(&meta, local) else {
            return;
        };
        let Some(edge) = self.sel.ego_graph.g().edge_weight(edge_idx) else {
            return;
        };
        let p = edge.payload();
        let type_label = self
            .data
            .as_ref()
            .map(|d| d.type_name(p.presyn).to_string())
            .unwrap_or_default();
        let type_label = if type_label.is_empty() {
            "?".to_string()
        } else {
            type_label
        };
        let dir = if p.out {
            "→ от драйва к соседу"
        } else {
            "← от соседа к драйву"
        };
        egui::Tooltip::always_open(
            ui.ctx().clone(),
            ui.layer_id(),
            egui::Id::new("ego_edge_tip"),
            egui::PopupAnchor::Position(pointer),
        )
        .show(|ui| {
            ui.label(egui::RichText::new(format!("#{idx} {type_label}", idx = p.presyn)).strong());
            ui.label(format!("{} синапсов · вес {:.1}", p.count, p.weight));
            ui.label(egui::RichText::new(dir).weak());
        });
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

/// Маленький цветной кружок для легенды (цвет ребра по полярности НТ).
fn legend_dot(ui: &mut egui::Ui, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
}
