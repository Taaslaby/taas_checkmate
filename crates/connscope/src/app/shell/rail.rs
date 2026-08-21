//! # Левый рейл инструментов (`app::rail`)
//!
//! Узкая вертикальная полоса иконок (~44 px) слева от вьюпорта — как панель
//! инструментов Blender/Adobe. Все «вьюпортные» контролы, которые раньше
//! забивали верхний тулбар, живут здесь:
//!
//! ```text
//! rail
//! ├── XZ / XY / YZ    проекции 2D (активная — акцентом)
//! ├── 🌐 3D           переключение в 3D-облако
//! ├── (если 3D) ⚙     попап: ракурсы, орто/перс, автовращение, GPU,
//! │                   yaw/pitch/перспектива, размер/непрозрачность точек
//! ├── 🎨              попап: режим окраски (НТ/класс/тип/степень)
//! ├── 🔽              попап «Фильтры»: НТ-свачи, класс, мин. степень
//! └── ⟳               Fit view (F)
//! ```
//!
//! Каждая кнопка — иконка + тултип с горячей клавишей; активное состояние
//! подсвечивается сигнальным акцентом темы (`theme::accent(ui)`).

use std::f32::consts::PI;

use eframe::egui::{self, Color32};

use elegance::{Checkbox, Slider as ElegSlider};

use egui_phosphor::regular as ic;

use crate::cli::class_name;
use crate::core::color::ColorMode;
use crate::core::nt::Nt;
use crate::io::data::Proj;

use crate::app::connectome::state::Render3d;
use crate::app::connectome::ConnectomeMode;
use crate::app::tokens::{self};

/// Внешняя ширина рейла. `egui::Panel` оставляет около 16 px на frame,
/// поэтому для 56 px полезной колонки нужен внешний размер 72 px.
const RAIL_W: f32 = 72.0;
/// Полезная ширина колонки после panel frame.
const RAIL_CONTENT_W: f32 = 56.0;
/// Высота одной кнопки рейла: одинаковая для проекций и иконок.
const RAIL_BUTTON_H: f32 = 28.0;

impl ConnectomeMode {
    /// Левый рейл инструментов: проекции, 3D, окраска, фильтры, Fit.
    pub(crate) fn tool_rail(&mut self, ui: &mut egui::Ui) {
        if !self.rail_visible {
            return;
        }
        egui::Panel::left("rail")
            .exact_size(RAIL_W)
            .resizable(false)
            .show(ui, |ui| {
                tokens::spacing(ui);
                super::profiler::record("rail", ui.max_rect());
                // Все rail-кнопки должны иметь одну геометрию. Стандартный
                // вертикальный button padding делал текстовые XZ/XY/YZ и
                // Phosphor-иконки разной высоты.
                ui.spacing_mut().button_padding = egui::vec2(8.0, 0.0);
                ui.spacing_mut().item_spacing.y = tokens::SP_XS;
                // `top_down_justified` растягивает каждый виджет до всей
                // доступной высоты; для вертикального тулбара это превращало
                // кнопки в 48px-блоки. Обычный top_down сохраняет фиксированный
                // размер и оставляет одинаковый gap между инструментами.
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    // ── проекции 2D: XZ | XY | YZ ──
                    for (label, proj) in [("XZ", Proj::Xz), ("XY", Proj::Xy), ("YZ", Proj::Yz)] {
                        let active = !self.camera.view3d && self.camera.proj == proj;
                        if rail_button(ui, label, active, 13.0)
                            .on_hover_text(format!("Проекция {label}"))
                            .clicked()
                        {
                            self.camera.view3d = false;
                            self.camera.proj = proj;
                            self.camera.atlas = None;
                            self.camera.atlas3d = None;
                        }
                    }
                    rail_sep(ui);
                    // ── 3D-облако ──
                    if rail_button(ui, ic::GLOBE, self.camera.view3d, tokens::ICON_SM)
                        .on_hover_text("3D-облако точек (как в меню «Вид»)")
                        .clicked()
                    {
                        self.toggle_3d(ui.ctx());
                    }
                    if self.camera.view3d {
                        // ── настройки 3D-камеры/облака ──
                        let btn = rail_button(ui, ic::SLIDERS_HORIZONTAL, false, tokens::ICON_SM)
                            .on_hover_text("3D: ракурсы, камера, точки");
                        egui::Popup::menu(&btn).show(|ui| {
                            ui.set_min_width(230.0);
                            self.rail_popup_3d(ui);
                        });
                    }
                    rail_sep(ui);
                    // ── окраска ──
                    let mode_label = ColorMode::all()
                        .iter()
                        .find(|m| **m == self.filters.color_mode)
                        .map(|m| m.label())
                        .unwrap_or("?");
                    // Подсветка акцентом, когда окраска не дефолтная (НТ) —
                    // видно, что «что-то перекрашено», без ховера.
                    let non_default = self.filters.color_mode != ColorMode::Nt;
                    let btn = rail_button(ui, ic::PALETTE, non_default, tokens::ICON_SM)
                        .on_hover_text(format!("Окраска: {mode_label}"));
                    egui::Popup::menu(&btn).show(|ui| {
                        ui.set_min_width(220.0);
                        for m in ColorMode::all() {
                            if ui
                                .selectable_label(self.filters.color_mode == m, m.label())
                                .clicked()
                            {
                                self.filters.color_mode = m;
                                self.camera.atlas = None;
                                self.camera.atlas3d = None;
                            }
                        }
                    });
                    // ── фильтры ──
                    let btn = rail_button(ui, ic::FUNNEL, false, tokens::ICON_SM)
                        .on_hover_text("Фильтры: НТ, класс, мин. степень");
                    egui::Popup::menu(&btn).show(|ui| {
                        ui.set_min_width(260.0);
                        self.rail_popup_filters(ui);
                    });
                    rail_sep(ui);
                    // ── Fit view ──
                    if rail_button(ui, ic::ARROW_CLOCKWISE, false, tokens::ICON_SM)
                        .on_hover_text("Вписать вид [F]")
                        .clicked()
                    {
                        self.reset_camera();
                    }
                });
            });
    }

    /// Попап «3D»: ракурсы, орто/перс, автовращение, GPU и слайдеры камеры.
    fn rail_popup_3d(&mut self, ui: &mut egui::Ui) {
        // Ракурсы — как навигационный куб (секция сворачивается).
        if super::panels::section_header(ui, ic::COMPASS, "Ракурс", &mut self.cam_views_open)
        {
            ui.ctx().request_repaint();
        }
        if self.cam_views_open {
            ui.horizontal(|ui| {
                // Ракурсы — компактные `icon_button` (единый вид тулбарных
                // кнопок). Активное состояние подсвечивает текущий канонический
                // ракурс — но только если камера не крутится (автовращение
                // непрерывно меняет yaw, и подсветка была бы ложной).
                let spin = self.camera.spin3d;
                for (icon, name, yaw, pitch) in [
                    (ic::ARROW_UP, "Дорсаль (сверху)", 0.0f32, -1.5f32),
                    (ic::ARROW_DOWN, "Вентраль (снизу)", 0.0, 1.5),
                    (ic::EYE, "Фронтал (спереди)", 0.0, 0.0),
                    (ic::ARROWS_LEFT_RIGHT, "Сагиттал (сбоку)", 1.57, 0.0),
                ] {
                    let active =
                        !spin && near(self.camera.yaw, yaw) && near(self.camera.pitch, pitch);
                    if crate::app::widgets::icon_button(ui, icon, active, tokens::ICON_SM)
                        .on_hover_text(name)
                        .clicked()
                    {
                        self.set_camera(yaw, pitch);
                    }
                }
            });
        }
        ui.separator();
        let mut ort = if self.camera.ortho { 0 } else { 1 };
        ui.horizontal(|ui| {
            ui.add_sized(
                [ui.available_width(), tokens::TOOL_SLIDER_H + 12.0],
                elegance::SegmentedControl::new(&mut ort, ["Орто", "Перс"]),
            );
        });
        if (ort == 0) != self.camera.ortho {
            self.camera.ortho = ort == 0;
            self.camera.atlas3d = None;
        }
        let mut spin = self.camera.spin3d;
        if ui.add(Checkbox::new(&mut spin, "Автовращение")).changed() {
            self.camera.spin3d = spin;
            self.camera.atlas3d = None;
        }
        let mut gpu = self.camera.use_gpu3d;
        if ui
            .add(Checkbox::new(
                &mut gpu,
                "Шейдеры (GPU) — выкл. = CPU-запечка",
            ))
            .changed()
        {
            self.camera.use_gpu3d = gpu;
            self.camera.atlas3d = None;
        }
        let mut render = match self.camera.render3d {
            Render3d::Spheres => 0,
            Render3d::Points => 1,
            Render3d::Neon => 2,
        };
        ui.label(egui::RichText::new("РЕНДЕР").small().weak());
        if ui
            .add(elegance::SegmentedControl::new(
                &mut render,
                Render3d::ALL.map(|mode| mode.label()),
            ))
            .changed()
        {
            self.camera.render3d = Render3d::ALL[render.min(2)];
        }
        ui.separator();
        if ui
            .add(
                ElegSlider::new(&mut self.camera.yaw, -PI..=PI)
                    .label("yaw")
                    .suffix(" рад")
                    .decimals(2),
            )
            .changed()
        {
            self.camera.atlas3d = None;
        }
        if ui
            .add(
                ElegSlider::new(&mut self.camera.pitch, -1.5..=1.5)
                    .label("pitch")
                    .suffix(" рад")
                    .decimals(2),
            )
            .changed()
        {
            self.camera.atlas3d = None;
        }
        if !self.camera.ortho
            && ui
                .add(
                    ElegSlider::new(&mut self.camera.persp, 0.1..=1.5)
                        .label("перспектива")
                        .suffix("×")
                        .decimals(2),
                )
                .changed()
        {
            self.camera.atlas3d = None;
        }
        ui.separator();
        if ui
            .add(
                ElegSlider::new(&mut self.camera.pt_size, 1.0..=10.0)
                    .label("размер точек")
                    .suffix("px")
                    .decimals(1),
            )
            .changed()
        {
            self.camera.atlas3d = None;
        }
        if ui
            .add(
                ElegSlider::new(&mut self.camera.pt_alpha, 0.1..=1.0)
                    .label("непрозрачность")
                    .decimals(2),
            )
            .changed()
        {
            self.camera.atlas3d = None;
        }
    }

    /// Попап «Фильтры»: НТ-свачи, класс клетки, минимальная степень.
    fn rail_popup_filters(&mut self, ui: &mut egui::Ui) {
        if super::panels::section_header(
            ui,
            ic::DNA,
            "Нейротрансмиттеры",
            &mut self.filters_nt_open,
        ) {
            ui.ctx().request_repaint();
        }
        if self.filters_nt_open {
            ui.horizontal(|ui| {
                for nt in 0..7 {
                    let n = Nt::from_code(nt as u8);
                    let on = self.filters.show_nt[nt];
                    let col = n.color();
                    let dot = if on { col } else { col.gamma_multiply(0.3) };
                    let mut btn = egui::Button::new(egui::RichText::new("●").color(dot).size(16.0))
                        .frame(false);
                    if on {
                        btn = btn.fill(Color32::from_rgba_unmultiplied(
                            col.r(),
                            col.g(),
                            col.b(),
                            45,
                        ));
                    }
                    if ui.add(btn).on_hover_text(n.name()).clicked() {
                        self.filters.show_nt[nt] = !on;
                        self.camera.atlas = None;
                        self.camera.atlas3d = None;
                    }
                }
            });
        }
        ui.separator();
        egui::ComboBox::from_id_salt("rail_class_filter")
            .selected_text(class_name(self.filters.class_filter))
            .width(tokens::TOOL_COMBO_W)
            .show_ui(ui, |ui| {
                for (v, name) in [
                    (-1i8, "все"),
                    (0, "other"),
                    (1, "motor"),
                    (2, "sensory"),
                    (3, "ascending"),
                    (4, "descending"),
                    (5, "intrinsic"),
                    (6, "glia"),
                    (7, "efferent"),
                ] {
                    if ui
                        .selectable_value(&mut self.filters.class_filter, v, name)
                        .changed()
                    {
                        self.camera.atlas = None;
                    }
                }
            });
        if ui
            .add_sized(
                [tokens::TOOL_SLIDER_W, tokens::TOOL_SLIDER_H],
                ElegSlider::new(&mut self.filters.min_deg, 0..=200),
            )
            .on_hover_text("Мин. степень: нейроны с меньшим числом исходящих связей скрываются")
            .changed()
        {
            self.camera.atlas = None;
        }
        ui.label(
            egui::RichText::new("Нейроны с меньшим числом исходящих связей скрываются")
                .weak()
                .small(),
        );
    }
}

/// Кнопка рейла: иконка (или короткий текст-«иконка» XZ/XY/YZ); активное
/// состояние — сигнальный акцент темы. Рендер — в общем [`widgets::tool_button`]
/// (единый вид тулбарных кнопок рейла/топа/модалок); здесь — только
/// rail-геометрия (полная ширина колонки, фиксированная высота) и профайлер.
fn rail_button(ui: &mut egui::Ui, icon: &str, active: bool, size: f32) -> egui::Response {
    let response =
        crate::app::widgets::tool_button(ui, icon, active, size, RAIL_CONTENT_W, RAIL_BUTTON_H);
    super::profiler::record(format!("rail_btn_{icon}"), response.rect);
    response
}

/// Тонкий разделитель внутри рейла.
fn rail_sep(ui: &mut egui::Ui) {
    ui.add_space(tokens::SP_XS);
    ui.separator();
    ui.add_space(tokens::SP_XS);
}

/// Два угла (радианы) совпадают «в пределах допуска»: после клика по ракурсу
/// камера стоит ровно в канонической точке, после драга/автовращения гаснет.
fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.15
}
