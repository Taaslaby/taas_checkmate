//! # Горячие клавиши / макросы (`app::shortcuts`)
//!
//! Единый реестр горячих клавиш: таблица (комбинация + действие), одна
//! функция-диспетчер и таблица [`SHORTCUT_LIST`] для модалки «Горячие
//! клавиши» (F1 / `?`). **Новый макрос = одна строка в таблице + одна
//! ветка в `dispatch`** — никаких разбросанных `key_pressed` по панелям.
//!
//! Клавиши разделены по владельцам:
//!
//! ```text
//! Оболочка (ConnApp)               Режим коннектома (ConnectomeMode)
//! ? / F1   справка                 F        fit камеры
//! Ctrl+O   открыть файл            + / = / − зум к центру канваса
//! Ctrl+R   перезагрузить файл      H        левый рейл инструментов
//! Ctrl+Shift+D  дамп раскладки
//! F12      консоль (вкладка дока)
//! [ / ]    цикл вкладок рабочего пространства
//! Ctrl+S   сохранить редактор NeuroML / фокус на поиск
//! ```
//!
//! `F12` и `[`/`]` живут в оболочке: дерево `egui_dock` владеет она.
//! Режим обрабатывает остальные клавиши в `Mode::handle_shortcuts` — новый
//! режим приносит свои клавиши вместе с собой.

use eframe::egui::{self, Id, Key, KeyboardShortcut, Modifiers};

use super::profiler;
use crate::app::connectome::ConnectomeMode;
use crate::app::mode::Surface;
use crate::app::ConnApp;

/// Стабильный id поискового поля — по нему `Ctrl+S` отдаёт фокус.
pub(crate) const SEARCH_FIELD_ID: &str = "search_field";

// ── клавиши оболочки ───────────────────────────────────────────

/// Действие оболочки, привязанное к горячей клавише.
#[derive(Clone, Copy)]
enum ShellAction {
    OpenFile,
    Reload,
    DumpLayout,
    /// Модалка «Горячие клавиши».
    ShowShortcuts,
    /// Фокус на вкладку «Консоль» (F12).
    FocusConsole,
    /// Цикл вкладок рабочего пространства (`false` — назад, `true` — вперёд).
    CycleDock(bool),
    /// Ctrl+S: сохранить редактор NeuroML, если он в фокусе, иначе —
    /// фокус на поиск нейрона.
    SaveOrSearch,
}

/// Таблица макросов оболочки: комбинация → действие. Порядок не важен —
/// в кадре срабатывает ровно одно (первое совпадение), нажатие «съедается».
const SHELL_SHORTCUTS: &[(KeyboardShortcut, ShellAction)] = &[
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::Questionmark),
        ShellAction::ShowShortcuts,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::F1),
        ShellAction::ShowShortcuts,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::F12),
        ShellAction::FocusConsole,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::OpenBracket),
        ShellAction::CycleDock(false),
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::CloseBracket),
        ShellAction::CycleDock(true),
    ),
    (
        // COMMAND = Ctrl на Windows/Linux, Cmd на macOS.
        KeyboardShortcut::new(Modifiers::COMMAND, Key::O),
        ShellAction::OpenFile,
    ),
    (
        KeyboardShortcut::new(Modifiers::COMMAND, Key::R),
        ShellAction::Reload,
    ),
    (
        KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::D),
        ShellAction::DumpLayout,
    ),
    (
        KeyboardShortcut::new(Modifiers::COMMAND, Key::S),
        ShellAction::SaveOrSearch,
    ),
];

// ── клавиши режима коннектома ──────────────────────────────────

/// Действие режима, привязанное к горячей клавише.
#[derive(Clone, Copy)]
enum ModeAction {
    FitReset,
    ZoomIn,
    ZoomOut,
    /// Показать/скрыть левый рейл инструментов.
    ToggleRail,
}

/// Таблица макросов режима: F (fit), +/− (зум), H (рейл), Ctrl+S (поиск).
const MODE_SHORTCUTS: &[(KeyboardShortcut, ModeAction)] = &[
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::F),
        ModeAction::FitReset,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::Plus),
        ModeAction::ZoomIn,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::Equals),
        ModeAction::ZoomIn,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::Minus),
        ModeAction::ZoomOut,
    ),
    (
        KeyboardShortcut::new(Modifiers::NONE, Key::H),
        ModeAction::ToggleRail,
    ),
];

/// Реестр для модалки «Горячие клавиши»: (комбинация, описание).
pub(crate) const SHORTCUT_LIST: &[(&str, &str)] = &[
    ("F12", "консоль-комбайн (фокус вкладки «Консоль»)"),
    (
        "F",
        "вписать вид (Fit): zoom=1, панорама по умолчанию, в 3D — стартовый ракурс",
    ),
    ("+ / =", "зум к центру канваса"),
    ("−", "зум от центра канваса"),
    ("H", "показать/скрыть левый рейл инструментов"),
    ("[ / ]", "предыдущая/следующая вкладка рабочего пространства"),
    ("? / F1", "эта справка"),
    ("Ctrl+S", "сохранить редактор NeuroML / фокус на поиск нейрона"),
    ("Ctrl+O", "открыть файл данных (.bin/.taas/.graphml/.npz)"),
    ("Ctrl+R", "перезагрузить текущий файл"),
    ("Ctrl+Shift+D", "дамп раскладки в connscope_profiler.json"),
    (
        "Мышь",
        "колесо — зум · drag — панорама (в 3D — вращение) · клик — выбрать нейрон",
    ),
];

impl ConnApp {
    /// Обрабатывает глобальные горячие клавиши **оболочки** (справка,
    /// открыть/перезагрузить файл, профайлер, вкладки дока). Клавиши режима
    /// обрабатывает сам режим (`Mode::handle_shortcuts`). Вызывается первым
    /// делом в `ConnApp::ui` — до раскладки панелей, чтобы запрошенный фокус
    /// подхватило поле, создаваемое в этом же кадре.
    pub(crate) fn handle_shortcuts(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx();
        // «Печатные» клавиши без модификаторов (?, [) не перехватываем, пока
        // фокус в текстовом поле (поиск/консоль); F1/F12 — можно.
        let typing = ctx.egui_wants_keyboard_input();
        let mut triggered = None;
        ctx.input_mut(|i| {
            for (shortcut, action) in SHELL_SHORTCUTS {
                let bare = shortcut.modifiers == Modifiers::NONE
                    && !matches!(shortcut.logical_key, Key::F12 | Key::F1);
                if bare && typing {
                    continue;
                }
                if i.consume_shortcut(shortcut) {
                    triggered = Some(*action);
                    break;
                }
            }
        });
        match triggered {
            Some(ShellAction::ShowShortcuts) => self.ui.show_shortcuts = true,
            Some(ShellAction::OpenFile) => {
                self.ui.open_path = self.ui.data_path.display().to_string();
                self.ui.show_open = true;
            }
            Some(ShellAction::Reload) => {
                let p = self.ui.data_path.clone();
                self.mode.load_data(&p);
            }
            Some(ShellAction::DumpLayout) => profiler::arm(),
            Some(ShellAction::FocusConsole) => self.focus_surface(Surface::Console),
            Some(ShellAction::CycleDock(forward)) => self.cycle_surface(forward),
            Some(ShellAction::SaveOrSearch) => {
                // Когда текст редактора NeuroML в фокусе — Ctrl+S сохраняет
                // файл; в остальном интерфейсе — фокус на поиск нейрона.
                if self.neuroml.editor.focused {
                    self.neuroml.editor_save();
                } else {
                    ctx.memory_mut(|m| m.request_focus(Id::new(SEARCH_FIELD_ID)));
                }
            }
            None => {}
        }
    }
}

impl ConnectomeMode {
    /// Горячие клавиши режима коннектома (см. таблицу [`MODE_SHORTCUTS`]).
    /// Возвращает `true`, если клавиша обработана.
    pub(crate) fn handle_shortcuts(&mut self, ctx: &egui::Context) -> bool {
        // «Печатные» клавиши (F, +/−, H, Ctrl+S) не перехватываем, пока фокус
        // в текстовом поле (поиск/консоль).
        let typing = ctx.egui_wants_keyboard_input();
        let mut triggered = None;
        ctx.input_mut(|i| {
            for (shortcut, action) in MODE_SHORTCUTS {
                if shortcut.modifiers == Modifiers::NONE && typing {
                    continue;
                }
                if i.consume_shortcut(shortcut) {
                    triggered = Some(*action);
                    break;
                }
            }
        });
        match triggered {
            Some(ModeAction::FitReset) => self.reset_camera(),
            Some(ModeAction::ZoomIn) | Some(ModeAction::ZoomOut) => {
                let factor = if matches!(triggered, Some(ModeAction::ZoomIn)) {
                    1.25
                } else {
                    0.8
                };
                // Якорь — центр окна: точный размер канваса тут недоступен,
                // а приблизительный центр экрана даёт верное ощущение «зум к середине».
                let center = ctx.input(|i| {
                    i.raw
                        .screen_rect
                        .map_or(egui::vec2(780.0, 480.0), |r| r.size())
                }) * 0.5;
                self.zoom_step(factor, center);
                ctx.request_repaint();
            }
            Some(ModeAction::ToggleRail) => self.rail_visible = !self.rail_visible,
            None => return false,
        }
        true
    }
}
