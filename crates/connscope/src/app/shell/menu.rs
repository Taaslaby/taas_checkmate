//! # Меню и модальные окна (`app::menu`)
//!
//! * **`menu_bar`** — «Файл / Правка / Вид / Окна / Справка» на дизайн-системе
//!   elegance: режимная секция «Вид» отдаётся активному режиму
//!   (`Mode::view_menu`), темы и акценты (со свотч-кружками), «Окна» — список
//!   поверхностей рабочего пространства (переоткрытие/фокус), выход;
//! * **`modal_windows`** — модалки оболочки: «О программе», «Открыть файл»
//!   и «Горячие клавиши» на встроенном [`egui::containers::Modal`]
//!   (затемнение фона, Esc/клик мимо = закрыть). Модалки режимов
//!   («Плоттеры», «NeuroML») рисует сам режим в `Mode::windows`;
//! * **`theme_dot` / `accent_dot`** — свотч-кружки подменю «Тема/Акцент».

use std::path::{Path, PathBuf};

use eframe::egui::{self, Color32};

use elegance::{BuiltInTheme, Button as ElegButton, MenuBar, Modal, StatCard};

use egui_phosphor::regular as ic;

use crate::app::mode::Surface;
use crate::app::neuroml::editor::EditorEdit;
use crate::app::shell::shortcuts::SHORTCUT_LIST;
use crate::app::tokens::{self, icon_label};
use crate::app::{AccentPreset, ConnApp, DockPreset};

/// Единая геометрия строк dropdown-меню.
/// Фиксированная ширина не даёт меню прыгать при смене пункта,
/// а одинаковая высота делает «Файл / Вид / Справка» одним компонентом.
const MENU_ITEM_W: f32 = 300.0;
const MENU_ITEM_H: f32 = 30.0;

fn menu_section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(tokens::SP_XS);
    ui.label(tokens::eyebrow(title));
}

fn menu_item(ui: &mut egui::Ui, icon: &str, label: &str) -> bool {
    ui.add_sized(
        [MENU_ITEM_W, MENU_ITEM_H],
        egui::Button::new(icon_label(icon, label, tokens::ICON_SM)),
    )
    .clicked()
}

fn menu_toggle(ui: &mut egui::Ui, checked: &mut bool, icon: &str, label: &str) -> bool {
    ui.add_sized(
        [MENU_ITEM_W, MENU_ITEM_H],
        egui::Checkbox::new(checked, icon_label(icon, label, tokens::ICON_SM)),
    )
    .clicked()
}

impl ConnApp {
    // ── меню ──────────────────────────────────────────────────
    pub(crate) fn menu_bar(&mut self, ui: &mut egui::Ui) {
        // exact_size: без него egui берёт высоту из сохранённого PanelState
        // (persist между запусками) — после рефакторинга оставался старый
        // размер и под меню висела пустая полоса ~20px.
        // 32.0 = натуральная высота elegance::MenuBar (STRIP_PAD_Y 4×2 +
        // кнопка-триггер body 14 + 2×5): меньше — триггеры обрезаются снизу.
        egui::Panel::top("menu_bar")
            .exact_size(32.0)
            .show(ui, |ui| {
                super::profiler::record("menu", ui.max_rect());
                MenuBar::new("conn_menu").show(ui, |m| {
                    m.menu(
                        icon_label(ic::FOLDER_OPEN, "Файл", tokens::ICON_SM),
                        |ui| {
                            ui.set_min_width(MENU_ITEM_W);
                            menu_section(ui, "ДАННЫЕ");
                            if menu_item(ui, ic::FOLDER_OPEN, "Открыть данные…") {
                                self.ui.open_path = self.ui.data_path.display().to_string();
                                self.ui.show_open = true;
                            }
                            if menu_item(ui, ic::CIRCUITRY, "Открыть NeuroML (.nml)…") {
                                self.neuroml.show = true;
                            }
                            if menu_item(ui, ic::ARROW_CLOCKWISE, "Перезагрузить текущий файл")
                            {
                                let p = self.ui.data_path.clone();
                                self.mode.load_data(&p);
                            }
                            ui.separator();
                            if menu_item(ui, ic::SIGN_OUT, "Выйти") {
                                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        },
                    );
                    m.menu(
                        icon_label(ic::PENCIL_SIMPLE, "Правка", tokens::ICON_SM),
                        |ui| {
                            ui.set_min_width(MENU_ITEM_W);
                            menu_section(ui, "РЕДАКТОР NEUROML");
                            if menu_item(ui, ic::ARROW_BEND_UP_LEFT, "Отменить (Ctrl+Z)") {
                                self.focus_surface(Surface::Editor);
                                self.neuroml.editor.pending_edit = Some(EditorEdit::Undo);
                            }
                            if menu_item(ui, ic::ARROW_BEND_UP_RIGHT, "Вернуть (Ctrl+Y)") {
                                self.focus_surface(Surface::Editor);
                                self.neuroml.editor.pending_edit = Some(EditorEdit::Redo);
                            }
                            ui.separator();
                            if menu_item(ui, ic::FLOPPY_DISK, "Сохранить NeuroML (Ctrl+S)") {
                                self.focus_surface(Surface::Editor);
                                self.neuroml.editor_save();
                            }
                        },
                    );
                    m.menu_keep_open(icon_label(ic::EYE, "Вид", tokens::ICON_SM), |ui| {
                        ui.set_min_width(MENU_ITEM_W);
                        // Режимная секция (камера/проекции/инструменты) —
                        // её рисует активный режим; он может попросить фокус
                        // вкладки рабочего пространства.
                        if let Some(surface) = self.mode.view_menu(ui) {
                            self.focus_surface(surface);
                        }
                        ui.separator();
                        menu_section(ui, "РАСКЛАДКА");
                        for p in DockPreset::ALL {
                            if menu_item(ui, ic::SQUARES_FOUR, p.label()) {
                                self.apply_preset(p);
                            }
                        }
                        menu_section(ui, "ТЕМА");
                        for t in BuiltInTheme::all() {
                            let sel = self.ui.theme == t;
                            ui.horizontal(|ui| {
                                ui.colored_label(theme_dot(t), "●");
                                if ui
                                    .add_sized(
                                        [MENU_ITEM_W - 24.0, MENU_ITEM_H],
                                        egui::Button::new(
                                            egui::RichText::new(t.label())
                                                .strong()
                                                .size(tokens::ICON_SM),
                                        )
                                        .fill(if sel {
                                            ui.visuals().selection.bg_fill
                                        } else {
                                            Color32::TRANSPARENT
                                        }),
                                    )
                                    .clicked()
                                {
                                    self.ui.theme = t;
                                }
                            });
                        }
                        menu_section(ui, "АКЦЕНТ");
                        for a in AccentPreset::all() {
                            let sel = self.ui.accent == a;
                            ui.horizontal(|ui| {
                                ui.colored_label(accent_dot(a), "●");
                                if ui
                                    .add_sized(
                                        [MENU_ITEM_W - 24.0, MENU_ITEM_H],
                                        egui::Button::new(
                                            egui::RichText::new(a.label())
                                                .strong()
                                                .size(tokens::ICON_SM),
                                        )
                                        .fill(if sel {
                                            ui.visuals().selection.bg_fill
                                        } else {
                                            Color32::TRANSPARENT
                                        }),
                                    )
                                    .clicked()
                                {
                                    self.ui.accent = a;
                                }
                            });
                        }
                    });
                    m.menu(
                        icon_label(ic::SQUARES_FOUR, "Окна", tokens::ICON_SM),
                        |ui| {
                            ui.set_min_width(MENU_ITEM_W);
                            menu_section(ui, "ПОВЕРХНОСТИ");
                            for s in Surface::ALL {
                                let focused = self.focused_surface() == Some(s);
                                let label = format!(
                                    "{}  {}",
                                    if focused { "●" } else { "○" },
                                    s.label()
                                );
                                if ui
                                    .add_sized(
                                        [MENU_ITEM_W, MENU_ITEM_H],
                                        egui::Button::new(icon_label(
                                            s.icon(),
                                            &label,
                                            tokens::ICON_SM,
                                        )),
                                    )
                                    .on_hover_text("Сфокусировать/вернуть вкладку")
                                    .clicked()
                                {
                                    self.focus_surface(s);
                                }
                            }
                            ui.separator();
                            ui.label(
                                egui::RichText::new(
                                    "Закрытую вкладку можно вернуть отсюда; вкладки в плавающих окнах при закрытии тоже возвращаются в дерево.",
                                )
                                .weak()
                                .small(),
                            );
                        },
                    );
                    m.menu(
                        icon_label(ic::QUESTION, "Справка", tokens::ICON_SM),
                        |ui| {
                            ui.set_min_width(MENU_ITEM_W);
                            menu_section(ui, "СПРАВКА");
                            if menu_item(ui, ic::KEYBOARD, "Горячие клавиши (F1)") {
                                self.ui.show_shortcuts = true;
                            }
                            if crate::app::widgets::action_button(
                                ui,
                                ic::INFO,
                                "О программе",
                                crate::app::widgets::ActionTone::Primary,
                                MENU_ITEM_W,
                            )
                            .on_hover_text("Версия, формат и статистика текущего набора")
                            .clicked()
                            {
                                self.ui.show_about = true;
                            }
                        },
                    );
                });
            });
    }

    // ── модальные окна оболочки ───────────────────────────────
    pub(crate) fn modal_windows(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // ── «О программе» ──
        let (n_neurons, n_syn) = self.mode.data_stats();
        let mut about_open = self.ui.show_about;
        // Текстура логотипа уже загружена при старте; клонируем дешёвый
        // handle, чтобы не держать `self` заимствованным внутри замыкания.
        let logo_tex = self.ui.logo_tex.clone();
        let clicked_close = Modal::new("about_modal", &mut about_open)
            .heading(icon_label(ic::MICROSCOPE, "neoMyelin", tokens::ICON_LG))
            .subtitle("Интерактивная коннектомика — нативный стенд для MANC VNC.")
            .header_icon(ic::MICROSCOPE)
            .max_width(500.0)
            .show(ctx, |ui| {
                // Словознак neoMyelin — белый на прозрачном, тонируется под
                // цвет текста темы (виден и в тёмной, и в светлой теме).
                if let Some(logo) = &logo_tex {
                    ui.add(
                        egui::Image::from_texture((logo.id(), logo.size_vec2()))
                            .max_width(460.0)
                            .tint(ui.visuals().strong_text_color()),
                    );
                    ui.add_space(tokens::SP_SM);
                }
                // Метрики — StatCard'ы, как на дашборде.
                ui.horizontal(|ui| {
                    ui.add(
                        StatCard::new("Нейронов")
                            .value(n_neurons.to_string())
                            .width(210.0),
                    );
                    ui.add(
                        StatCard::new("Синапсов")
                            .value(n_syn.to_string())
                            .width(210.0),
                    );
                });
                ui.add_space(tokens::SP_MD);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Формат данных:").weak());
                    ui.label("manc_viz.bin (self-contained)");
                });
                ui.add_space(tokens::SP_LG);
                crate::app::widgets::action_button(
                    ui,
                    ic::X,
                    "Закрыть",
                    crate::app::widgets::ActionTone::Neutral,
                    140.0,
                )
                .clicked()
            })
            .unwrap_or(false);
        if clicked_close {
            about_open = false;
        }
        self.ui.show_about = about_open;

        // ── «Открыть файл» ──
        let mut open_open = self.ui.show_open;
        let mut open_path = self.ui.open_path.clone();
        // Приёмник нативного диалога ОС (rfd, фоновый поток) — заполнит путь.
        let mut picker_rx: Option<std::sync::mpsc::Receiver<Option<std::path::PathBuf>>> = None;
        let action = Modal::new("open_modal", &mut open_open)
            .heading("Открыть файл данных")
            .subtitle("Запечённые .bin/.taas или сырые .graphml/.npz (без конвертации)")
            .header_icon(ic::FOLDER_OPEN)
            .max_width(640.0)
            .show(ctx, |ui| {
                // Быстрые пресеты: запечённые мозги из resources/examples/data.
                // Клик = сразу загрузить (заполняет путь и открывает).
                let mut act = 0u8; // 0 = ничего, 1 = открыть, 2 = отмена
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Запечённые:").weak());
                    for (label, fname) in [
                        ("VNC (муха)", "manc_viz.bin"),
                        ("Мозг мухи (FAFB)", "fafb.bin"),
                        ("C. elegans", "elegans.bin"),
                        ("Кора мыши", "mouse_vc.bin"),
                    ] {
                        // clean_path убирает «..\..» из отображаемого пути
                        let full = crate::io::data::clean_path(
                            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                                .join("../../resources/examples/data")
                                .join(fname),
                        );
                        if ui
                            .button(label)
                            .on_hover_text(full.display().to_string())
                            .clicked()
                        {
                            open_path = full.display().to_string();
                            act = 1;
                        }
                    }
                });
                ui.add_space(tokens::SP_SM);
                // Сырые коннектомы: парсятся нативно в Rust (GraphML / npz),
                // показываем только те, что есть на диске.
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Сырые (без конвертации):").weak());
                    for (label, path) in [
                        (
                            "C. elegans (graphml)",
                            "D:/connectoms/datasets/c.elegans_neural.male_1.graphml",
                        ),
                        (
                            "Кора мыши (graphml)",
                            "D:/connectoms/datasets/mouse_visual.cortex_2.graphml",
                        ),
                        (
                            "Мозг мухи FAFB (npz)",
                            "D:/connectoms/datasets/fafb_brain_connectome.npz",
                        ),
                    ] {
                        if !Path::new(path).exists() {
                            continue;
                        }
                        if ui
                            .button(label)
                            .on_hover_text(format!("{path} (+ координаты и аннотации рядом)"))
                            .clicked()
                        {
                            open_path = path.to_string();
                            act = 1;
                        }
                    }
                });
                ui.add_space(tokens::SP_SM);
                // Путь: поле ввода + кнопка нативного диалога ОС (rfd).
                // `set_parent(frame)` — задаём владельца (наше окно): иначе
                // диалог Windows уходит ЗА wgpu-окно и его не видно.
                // Открывается на фоновом потоке — UI не замирает.
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut open_path).desired_width(520.0));
                    if ui
                        .add(ElegButton::new(icon_label(
                            ic::FOLDER_OPEN,
                            "Обзор…",
                            tokens::ICON_SM,
                        )))
                        .on_hover_text("Нативный диалог Windows (как в Проводнике)")
                        .clicked()
                    {
                        let start_dir = if open_path.trim().is_empty() {
                            None
                        } else {
                            std::path::Path::new(open_path.trim())
                                .parent()
                                .map(|p| p.to_path_buf())
                        };
                        let (tx, rx) = std::sync::mpsc::channel();
                        let mut dlg = rfd::FileDialog::new()
                            .add_filter("Коннектомы", &["bin", "taas", "graphml", "npz"])
                            .add_filter("Все файлы", &["*"])
                            .set_parent(frame);
                        if let Some(dir) = start_dir {
                            dlg = dlg.set_directory(dir);
                        }
                        std::thread::spawn(move || {
                            let _ = tx.send(dlg.pick_file());
                        });
                        picker_rx = Some(rx);
                    }
                });
                ui.add_space(tokens::SP_SM);
                ui.horizontal(|ui| {
                    if crate::app::widgets::action_button(
                        ui,
                        ic::CHECK,
                        "Открыть",
                        crate::app::widgets::ActionTone::Primary,
                        120.0,
                    )
                    .clicked()
                    {
                        act = 1;
                    }
                    if crate::app::widgets::action_button(
                        ui,
                        ic::X,
                        "Отмена",
                        crate::app::widgets::ActionTone::Neutral,
                        120.0,
                    )
                    .clicked()
                    {
                        act = 2;
                    }
                });
                act
            })
            .unwrap_or(0);
        // Подхватываем результат нативного диалога (когда поток вернул путь)
        if let Some(rx) = &picker_rx {
            if let Ok(Some(p)) = rx.try_recv() {
                open_path = p.display().to_string();
            }
        }
        if action == 1 {
            let p = PathBuf::from(open_path.trim());
            self.mode.load_data(&p);
            self.ui.data_path = crate::io::data::clean_path(&p);
            open_open = false;
        } else if action == 2 {
            open_open = false;
        }
        self.ui.open_path = open_path;
        self.ui.show_open = open_open;

        // ── «Горячие клавиши»: единая шпаргалка (F1 / `?`) ──
        let mut sc_open = self.ui.show_shortcuts;
        let clicked_close = Modal::new("shortcuts_modal", &mut sc_open)
            .heading(icon_label(ic::KEYBOARD, "Горячие клавиши", tokens::ICON_SM))
            .subtitle("Все макросы приложения — один экран")
            .header_icon(ic::KEYBOARD)
            .max_width(560.0)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([tokens::SP_LG, tokens::SP_SM])
                    .show(ui, |ui| {
                        for (key, desc) in SHORTCUT_LIST {
                            ui.monospace(egui::RichText::new(*key).strong());
                            ui.label(egui::RichText::new(*desc).weak());
                            ui.end_row();
                        }
                    });
                ui.add_space(tokens::SP_MD);
                crate::app::widgets::action_button(
                    ui,
                    ic::X,
                    "Закрыть",
                    crate::app::widgets::ActionTone::Neutral,
                    140.0,
                )
                .clicked()
            })
            .unwrap_or(false);
        if clicked_close {
            sc_open = false;
        }
        self.ui.show_shortcuts = sc_open;
    }
}

/// Режимная секция меню «Вид» (коннектом): камера, проекции, инструменты.
/// Рисуется активным режимом внутри меню оболочки (`Mode::view_menu`).
impl crate::app::connectome::ConnectomeMode {
    pub(crate) fn view_menu(&mut self, ui: &mut egui::Ui) -> Option<Surface> {
        use crate::io::data::Proj;

        menu_section(ui, "КАМЕРА");
        if menu_toggle(ui, &mut self.camera.view3d, ic::GLOBE, "3D-облако точек") {
            if self.camera.view3d {
                self.enter_3d(ui.available_size());
            }
            self.camera.atlas = None;
            self.camera.atlas3d = None;
        }
        if menu_item(ui, ic::ARROW_CLOCKWISE, "Вписать вид (F)") {
            self.reset_camera();
        }
        if menu_item(ui, ic::EYE, "Показать все нейротрансмиттеры") {
            self.filters.show_nt = [true; 7];
            self.camera.atlas = None;
            self.camera.atlas3d = None;
        }
        menu_section(ui, "ПРОЕКЦИЯ");
        for p in Proj::ALL {
            let selected = !self.camera.view3d && self.camera.proj == p;
            let text = if selected {
                egui::RichText::new(format!("●  {}", p.name())).strong()
            } else {
                egui::RichText::new(format!("○  {}", p.name()))
            };
            if ui
                .add_sized([MENU_ITEM_W, MENU_ITEM_H], egui::Button::new(text))
                .clicked()
            {
                self.camera.proj = p;
                self.camera.view3d = false;
                self.camera.atlas = None;
            }
        }
        ui.separator();
        menu_section(ui, "ИНСТРУМЕНТЫ");
        let mut focus = None;
        if menu_item(ui, ic::TERMINAL, "Консоль (F12)") {
            focus = Some(Surface::Console);
        }
        if menu_item(ui, ic::CHART_LINE, "Библиотека плоттеров") {
            self.show_plots = true;
        }
        focus
    }
}

fn theme_dot(t: BuiltInTheme) -> Color32 {
    match t {
        BuiltInTheme::Slate => Color32::from_rgb(0x38, 0xbd, 0xf8),
        BuiltInTheme::Charcoal => Color32::from_rgb(0x22, 0xd3, 0xee),
        BuiltInTheme::Frost => Color32::from_rgb(0x03, 0x74, 0xb0),
        BuiltInTheme::Paper => Color32::from_rgb(0xf0, 0xb4, 0x29),
        // BuiltInTheme — #[non_exhaustive] в elegance: страхуемся.
        _ => Color32::from_rgb(0x38, 0xbd, 0xf8),
    }
}

fn accent_dot(a: AccentPreset) -> Color32 {
    a.fill()
}
