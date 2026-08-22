//! # Приложение connscope — ядро (`app`)
//!
//! Логика GUI-стенда живёт в папке `app/`, организованной как «воркбенч»:
//! общая рамка приложения (`shell`) + режимы работы (`connectome`, `neuroml`):
//!
//! ```text
//! app/
//! ├── mod.rs            ← ConnApp (тонкая оболочка): UiState, диспетчеризация
//! │                       кадра по режимам, run()
//! ├── mode.rs           ← трейт Mode + ShellServices (контракт режима)
//! ├── tokens.rs         ← дизайн-токены (общий UI-кит для всех режимов)
//! ├── shell/            ← рамка приложения: меню, статус-бар, модалки,
//! │                       профайлер, тема (консоль и поверхности режима —
//! │                       в самом режиме, они работают с его данными)
//! ├── connectome/       ← режим «Коннектом»: ConnectomeMode + все его
//! │                       поверхности (canvas, ego, инспектор, симуляция,
//! │                       плоттеры, док, рейл, верхняя панель, консоль)
//! └── neuroml/          ← режим «NeuroML»: NeuromlMode — модалка загрузки +
//!                         окно «Морфология» (дерево компартментов, клик,
//!                         V(t) по компартментам, слайдер времени) +
//!                         IDE-редактор (подсветка XML, валидация, ошибки)
//! ```
//!
//! **Диспетчеризация кадра** (`ConnApp::ui_frame`): оболочка рисует своё
//! (меню, статус-бар, модалки «О программе/Открыть/Горячие клавиши», тема),
//! а каждую поверхность активного режима вызывает через трейт [`Mode`]:
//!
//! ```text
//! menu_bar(view_menu) → mode.top_bar → mode.rail → hover_bar
//! → modal_windows → mode.windows → neuroml.windows → dock_area(dock_surface)
//! ```
//!
//! Рабочее пространство — единое дерево `egui_dock` в `CentralPanel`
//! (см. [`DockPreset`]); панели `top/bottom` рисуются до него. Новый режим =
//! новая структура с `impl Mode` — оболочка не меняется.

use std::path::PathBuf;

use eframe::egui::{self, vec2};
use egui_dock::tab_viewer::OnCloseResponse;
use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};
use elegance::{BuiltInTheme, Theme};

use crate::app::connectome::ConnectomeMode;
use crate::app::grow::GrowMode;
use crate::app::mode::{Mode, ShellServices, Surface};
use crate::app::neuroml::NeuromlMode;
use crate::app::shell::theme;

pub(crate) mod connectome;
pub(crate) mod grow;
pub(crate) mod mode;
pub(crate) mod neuroml;
pub(crate) mod shell;
pub(crate) mod tokens;
pub(crate) mod widgets;

/// Преустановленные акценты интерфейса — **единственный источник цвета
/// акцента**. Каждый пресет задаёт `focus` (структурный акцент: табы,
/// выделение, ползунки — и наши паинтерные кнопки через `theme::accent(ui)`)
/// и `blue`/`blue_hover` (заливка кнопок elegance). Переключается в меню
/// «Вид → Акцент».
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccentPreset {
    /// Фиолетовый — проф, как в Houdini.
    Violet,
    /// Золотой — тёплый акцент.
    Gold,
    /// Зелёный — сигнатурный люминофор neoMyelin.
    Green,
    /// Розовый — для выделения.
    Rose,
    /// Бирюзовый — холодный.
    Cyan,
}

impl AccentPreset {
    /// Подпись без эмодзи: цвет и так показан цветной точкой (`accent_dot`),
    /// а глифы вроде 🔵/🩵 не входят в шрифты egui и рисуются квадратами.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Violet => "Фиолетовый",
            Self::Gold => "Золотой",
            Self::Green => "Зелёный",
            Self::Rose => "Розовый",
            Self::Cyan => "Бирюзовый",
        }
    }

    pub(crate) const fn all() -> [AccentPreset; 5] {
        [
            Self::Violet,
            Self::Gold,
            Self::Green,
            Self::Rose,
            Self::Cyan,
        ]
    }

    /// `(focus, blue, blue_hover)` — единственный источник цветов акцента.
    ///
    /// `focus` — структурный акцент (табы, выделение, активное), его же рисуют
    /// паинтерные виджеты через `theme::accent(ui)`; `blue`/`blue_hover` —
    /// чуть более тёмные варианты для заливки кнопок elegance и свотча в меню.
    const fn palette(self) -> (egui::Color32, egui::Color32, egui::Color32) {
        const fn rgb(r: u8, g: u8, b: u8) -> egui::Color32 {
            egui::Color32::from_rgb(r, g, b)
        }
        match self {
            Self::Violet => (
                rgb(0x8f, 0x7b, 0xfa),
                rgb(0x7a, 0x5c, 0xf5),
                rgb(0x6a, 0x4a, 0xe0),
            ),
            Self::Gold => (
                rgb(0xf2, 0xb5, 0x44),
                rgb(0xe8, 0xa0, 0x20),
                rgb(0xd1, 0x8f, 0x16),
            ),
            // Проф teal — приглушённый, как в Houdini/DaVinci (был 5a d6 b2 — неон).
            Self::Green => (
                rgb(0x44, 0x9e, 0x8b),
                rgb(0x2e, 0x8a, 0x77),
                rgb(0x24, 0x74, 0x64),
            ),
            Self::Rose => (
                rgb(0xf4, 0x6f, 0x9e),
                rgb(0xe8, 0x54, 0x8a),
                rgb(0xd2, 0x3e, 0x76),
            ),
            Self::Cyan => (
                rgb(0x29, 0xc7, 0xe0),
                rgb(0x19, 0xb3, 0xcf),
                rgb(0x14, 0x9a, 0xb3),
            ),
        }
    }

    /// Более тёмный вариант акцента — заливка кнопок elegance и свотч в меню.
    pub(crate) const fn fill(self) -> egui::Color32 {
        self.palette().1
    }

    /// Перекрашивает тему: `focus` (структурный акцент) + `blue`/`blue_hover`
    /// (семантический акцент заливки). Единственная точка, где палитра elegance
    /// видит наш акцент.
    pub(crate) fn apply(self, theme: &mut Theme) {
        let (focus, blue, blue_hover) = self.palette();
        theme.palette.focus = focus;
        theme.palette.blue = blue;
        theme.palette.blue_hover = blue_hover;
    }
}

/// Пресеты раскладки рабочего пространства (дерева `egui_dock`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DockPreset {
    /// Дефолт: инспектор слева + атлас в центре — чистый старт, остальное
    /// открывается по требованию через «Окна»/«Вид».
    Workspace,
    /// Все поверхности сразу: инспектор | центр (атлас/ego/морфо/редактор/
    /// конструктор) + нижняя лента (плоттер/консоль/сим) — «всё на столе».
    Full,
    /// Только атлас — минимальный «чистый» вид.
    Minimal,
    /// Все поверхности вкладками в одном листе.
    Tabs,
}

impl DockPreset {
    pub(crate) const ALL: [DockPreset; 4] =
        [Self::Workspace, Self::Full, Self::Minimal, Self::Tabs];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Workspace => "Рабочий стол",
            Self::Full => "Все панели",
            Self::Minimal => "Только атлас",
            Self::Tabs => "Вкладки",
        }
    }

    /// Индекс для персистентности (egui `insert_persisted` требует `serde`,
    /// а пресет — обычный Copy-энум; храним порядковый номер).
    fn index(self) -> u8 {
        match self {
            Self::Workspace => 0,
            Self::Full => 1,
            Self::Minimal => 2,
            Self::Tabs => 3,
        }
    }

    fn from_index(i: u8) -> Option<Self> {
        match i {
            0 => Some(Self::Workspace),
            1 => Some(Self::Full),
            2 => Some(Self::Minimal),
            3 => Some(Self::Tabs),
            _ => None,
        }
    }
}

// ── Геометрические константы ──────────────────────────────────
/// Масштаб «мира» → экран: 1.0 мировой единицы = `BASE` пикселей при zoom=1.
const BASE: f32 = 720.0;
/// Разрешение текстуры 3D-облака точек.
const ATLAS_3D: usize = 1024;
/// Размер миникарты в углу канваса.
const MINIMAP: egui::Vec2 = vec2(180.0, 132.0);
/// Шаг сетки координат на 2D-атласе (в мировых единицах).
const GRID: f32 = 0.1;
/// Лимит узлов для локальной симуляции (ego-окрестность не больше N).
const MAX_SIM_NODES: usize = 60;

/// Состояние оболочки: тема, акцент, модалки оболочки, путь к данным.
/// Всё режимное состояние (камера, выбор, симуляция…) живёт в режимах —
/// см. [`ConnectomeMode`] и [`NeuromlMode`].
struct UiState {
    /// Текущий путь к данным (для «Открыть файл» и Ctrl+R).
    pub data_path: PathBuf,
    /// Поле «путь» модалки «Открыть файл».
    pub open_path: String,
    /// Модалка «Открыть файл» открыта.
    pub show_open: bool,
    /// Модалка «О программе» открыта.
    pub show_about: bool,
    /// Тема (палитра elegance).
    pub theme: BuiltInTheme,
    /// Акцент поверх темы.
    pub accent: AccentPreset,
    /// Модалка «Горячие клавиши» открыта.
    pub show_shortcuts: bool,
    /// Последний применённый пресет раскладки (для галочки в меню «Вид»).
    pub active_preset: DockPreset,
    /// Сглаженный FPS (EMA по stable_dt) для статус-бара.
    pub fps: f32,
    /// Логотип neoMyelin (загружается один раз при старте, рисуется в «О программе»).
    pub logo_tex: Option<egui::TextureHandle>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            data_path: PathBuf::new(),
            open_path: String::new(),
            show_open: false,
            show_about: false,
            // Charcoal — нейтральный монохромный тёмный фон (без синевы
            // дефолтного Slate 0x0f172a, который выглядел «дёшево»).
            theme: BuiltInTheme::Charcoal,
            // Сигнатурный «люминофор» — единый акцент по умолчанию.
            accent: AccentPreset::Green,
            show_shortcuts: false,
            active_preset: DockPreset::Workspace,
            fps: 0.0,
            logo_tex: None,
        }
    }
}

/// Оболочка приложения: тема/модалки/меню + активный режим + дерево доков.
///
/// Поля оболочки минимальны — всё остальное ушло в режимы (см. `mode.rs`).
/// Единственное «состояние раскладки» здесь — `DockState<Surface>`: оно
/// владеет кросс-режимными поверхностями (например «Морфология» из NeuromlMode).
struct ConnApp {
    ui: UiState,
    /// Активный режим (контракт — трейт [`Mode`]).
    mode: Box<dyn Mode>,
    /// Режим «NeuroML» (модалка + редактор + поверхность «Морфология»).
    neuroml: NeuromlMode,
    /// Режим «Развитие» (рост микросхемы + анализ, кросс-режимные поверхности).
    grow: GrowMode,
    /// Дерево доков рабочего пространства (владеет оболочка).
    dock: DockState<Surface>,
    /// Раскладка дока уже загружена из персистентного хранилища egui.
    dock_loaded: bool,
    /// Вкладки, закрытые в плавающих окнах — вернуть их в дерево после рендера.
    return_to_dock: Vec<Surface>,
}

impl Default for ConnApp {
    fn default() -> Self {
        let mut ui = UiState::default();
        let mut mode = ConnectomeMode::default();
        // Файл данных по умолчанию — рядом с крейтом; можно переопределить
        // первым аргументом командной строки (`connscope <file>`).
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../resources/examples/data/manc_viz.bin");
        let path = std::env::args().nth(1).map(PathBuf::from).unwrap_or(p);
        ui.data_path = path.clone();
        mode.load_data(&path);
        // Файл NeuroML по умолчанию — пример exc↔inh рядом с ресурсами.
        let mut neuroml = NeuromlMode::default();
        let mut nml = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        nml.push("../../resources/examples/neuroml/exc_inh_loop.nml");
        neuroml.path = nml;
        // Режим «Развитие» — растёт и симулирует по кнопке (в фоновом потоке).
        let grow = GrowMode::default();
        // Стартовый вид камера сама «впишет» при первом кадре канваса
        // (см. `fit_pending` в `load_data`) — жёсткий pan тут не нужен.
        Self {
            ui,
            mode: Box::new(mode),
            neuroml,
            grow,
            dock: default_dock(),
            dock_loaded: false,
            return_to_dock: Vec::new(),
        }
    }
}

// ── Кадр приложения (eframe 0.35: `ui`, а не `update`) ─────────
impl ConnApp {
    /// Один кадр: диспетчеризация по режиму — оболочка рисует своё, режим —
    /// свои поверхности (порядок фиксирован, см. документацию модуля).
    ///
    /// Не метод trait: вызывается из `eframe::App::ui` внутри `catch_unwind`,
    /// чтобы паника в любом коде кадра (пикинг, ego, симуляция, атласы)
    /// не убивала процесс.
    fn ui_frame(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Дизайн-система elegance: применяем тему (дёшево — пропускается,
        // если тема не менялась), поверх неё — выбранный акцент,
        // затем рисуем интерфейс.
        let mut theme = self.ui.theme.theme();
        self.ui.accent.apply(&mut theme);
        theme.install(ui.ctx());

        // Горячие клавиши: сначала оболочка (F1/? — справка, Ctrl+O — открыть,
        // Ctrl+R — перезагрузить, Ctrl+Shift+D — профайлер), затем режим
        // (F — fit, +/− — зум, H — рейл, [ ] — док, F12 — консоль, Ctrl+S — поиск).
        self.handle_shortcuts(ui);
        self.mode.handle_shortcuts(&ctx);

        // Редактор NeuroML — вкладка дока: сбрасываем флаг «текст в фокусе»
        // каждый кадр, а `editor_dock_view` выставит его снова, если вкладка
        // видна и поле имеет фокус (точное значение для Ctrl+S).
        self.neuroml.editor.focused = false;

        // Профайлер раскладки: если хоткеем взведён захват — начинаем
        // собирать фактические Rect виджетов этого кадра.
        shell::profiler::begin_frame();

        // Покадровый тик режима (камера, живые потоки, repaint).
        self.mode.frame_tick(&ctx);
        self.grow.frame_tick(&ctx);

        // Сглаженный FPS для статус-бара (EMA).
        let dt = ctx.input(|i| i.stable_dt).max(1e-4);
        self.ui.fps = self.ui.fps * 0.9 + (1.0 / dt) * 0.1;

        // Панели в порядке (Blender/Adobe-раскладка): меню, тонкий верхний
        // бар, левый рейл инструментов, статус-бар; модалки/окна — поверх;
        // рабочее пространство — единое дерево egui_dock в самом конце.
        self.menu_bar(ui);

        // Сервисы оболочки создаём в скоупе: `show_shortcuts` держит
        // `&mut self.ui.*`, а `self.focus_surface` заимствует весь `self` —
        // поэтому фокус читаем и применяем уже после того, как сервисы ушли.
        let focused = self.focused_surface();
        let focus = {
            let mut services = ShellServices {
                show_shortcuts: &mut self.ui.show_shortcuts,
                focus_surface: None,
                focused,
            };
            self.mode.top_bar(ui, &mut services);
            services.focus_surface
        };
        if let Some(surface) = focus {
            self.focus_surface(surface);
        }

        self.mode.rail(ui);
        self.hover_bar(ui);
        self.modal_windows(&ctx, frame);

        let focused = self.focused_surface();
        let focus = {
            let mut services = ShellServices {
                show_shortcuts: &mut self.ui.show_shortcuts,
                focus_surface: None,
                focused,
            };
            self.mode.windows(&ctx, frame, &mut services);
            self.neuroml.windows(&ctx, frame, &mut services);
            services.focus_surface
        };
        if let Some(surface) = focus {
            self.focus_surface(surface);
        }

        // Рабочее пространство — единое дерево egui_dock (инспектор, атлас,
        // ego, плоттер/консоль/сим, морфология) вместо фиксированных панелей.
        egui::CentralPanel::default_margins().show(ui, |ui| {
            shell::profiler::record("workspace", ui.max_rect());
            self.dock_area(ui);
        });

        // Конец кадра профайлера: если шёл захват — пишем JSON-снапшот.
        if let Some(json) = shell::profiler::end_frame() {
            match std::fs::write("connscope_profiler.json", &json) {
                Ok(()) => eprintln!("[profiler] dumped connscope_profiler.json"),
                Err(e) => eprintln!("[profiler] write failed: {e}"),
            }
        }
    }
}

// ── Рабочее пространство: единое дерево egui_dock ─────────────
impl ConnApp {
    /// Рендер дерева доков + загрузка/сохранение персистентной раскладки.
    fn dock_area(&mut self, ui: &mut egui::Ui) {
        self.load_dock_state(ui);
        // Классический разъём заимствований: DockArea держит `&mut DockState`,
        // а TabViewer — `&mut self`; вынимаем дерево на время рендера.
        let mut state = std::mem::replace(&mut self.dock, DockState::new(Vec::new()));
        DockArea::new(&mut state)
            .id(egui::Id::new("workspace_dock"))
            .style(theme::dock_style(ui))
            .show_close_buttons(false)
            .show_add_buttons(false)
            .show_add_popup(false)
            .show_inside(ui, self);
        self.dock = state;
        // Вернуть вкладки, закрытые в плавающих окнах, обратно в дерево.
        for surface in self.return_to_dock.drain(..) {
            if self.dock.find_tab(&surface).is_none() {
                self.dock.push_to_focused_leaf(surface);
            }
        }
        self.save_dock_state(ui);
    }

    fn load_dock_state(&mut self, ui: &egui::Ui) {
        if self.dock_loaded {
            return;
        }
        let id = egui::Id::new("workspace_dock_state");
        if let Some(saved) = ui
            .ctx()
            .data_mut(|d| d.get_persisted::<DockState<Surface>>(id))
        {
            self.dock = saved;
        }
        // Последний применённый пресет тоже переживает рестарт.
        let preset_id = egui::Id::new("workspace_active_preset");
        if let Some(idx) = ui.ctx().data_mut(|d| d.get_persisted::<u8>(preset_id)) {
            if let Some(p) = DockPreset::from_index(idx) {
                self.ui.active_preset = p;
            }
        }
        self.dock_loaded = true;
    }

    fn save_dock_state(&self, ui: &egui::Ui) {
        let id = egui::Id::new("workspace_dock_state");
        ui.ctx()
            .data_mut(|d| d.insert_persisted(id, self.dock.clone()));
        let preset_id = egui::Id::new("workspace_active_preset");
        ui.ctx()
            .data_mut(|d| d.insert_persisted(preset_id, self.ui.active_preset.index()));
    }

    /// Сфокусировать вкладку рабочего пространства (активировать её в листе).
    /// Если вкладку закрыли (или её убрал пресет) — возвращает её в фокусный
    /// лист, чтобы пользователь никогда не оставался без способа вернуть
    /// поверхность (защита от «закрыли всё»).
    fn focus_surface(&mut self, surface: Surface) {
        if let Some(path) = self.dock.find_tab(&surface) {
            if let Ok(leaf) = self.dock.leaf_mut(path.node_path()) {
                leaf.active = path.tab;
            }
            self.dock.set_focused_node_and_surface(path.node_path());
        } else {
            self.dock.push_to_focused_leaf(surface);
        }
    }

    /// Вкладка рабочего пространства в фокусе (для `[`/`]`/F12).
    fn focused_surface(&self) -> Option<Surface> {
        let path = self.dock.focused_leaf()?;
        let leaf = self.dock[path].get_leaf()?;
        leaf.tabs.get(leaf.active.0).copied()
    }

    /// Открыта ли поверхность в дереве сейчас (для меню «Окна»: отличать
    /// открытую неактивную вкладку от закрытой).
    fn surface_open(&self, surface: Surface) -> bool {
        self.dock.find_tab(&surface).is_some()
    }

    /// Цикл вкладок рабочего пространства (`[` назад, `]` вперёд).
    /// Если вкладки, которой нет в дереве (пресет «Только атлас»), — no-op.
    fn cycle_surface(&mut self, forward: bool) {
        let cur = self
            .focused_surface()
            .and_then(|s| Surface::ALL.iter().position(|&x| x == s));
        let target = match cur {
            Some(i) => {
                let n = Surface::ALL.len();
                let d = if forward { 1 } else { n - 1 };
                Surface::ALL[(i + d) % n]
            }
            None => {
                if forward {
                    Surface::ALL[0]
                } else {
                    Surface::ALL[Surface::ALL.len() - 1]
                }
            }
        };
        self.focus_surface(target);
    }

    /// Применить пресет раскладки (перестраивает дерево с нуля).
    fn apply_preset(&mut self, preset: DockPreset) {
        self.dock = match preset {
            DockPreset::Workspace => default_dock(),
            DockPreset::Full => full_dock(),
            DockPreset::Minimal => DockState::new(vec![Surface::Atlas]),
            DockPreset::Tabs => DockState::new(Surface::ALL.to_vec()),
        };
        // Помним последний пресет: меню «Вид» показывает галочку, а значение
        // переживает рестарт (см. save_dock_state / load_dock_state).
        self.ui.active_preset = preset;
    }
}

impl TabViewer for ConnApp {
    type Tab = Surface;

    fn title(&mut self, surface: &mut Self::Tab) -> egui::WidgetText {
        // Индикатор несохранённых изменений у вкладки «Редактор» (был в
        // заголовке отдельного окна — теперь живёт в заголовке вкладки).
        let mut label = surface.label().to_string();
        if *surface == Surface::Editor && self.neuroml.editor.modified {
            label.push_str(" ●");
        }
        tokens::icon_label(surface.icon(), &label, tokens::ICON_XS).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, surface: &mut Self::Tab) {
        let focused = self.focused_surface();
        let focus_request = {
            let mut services = ShellServices {
                show_shortcuts: &mut self.ui.show_shortcuts,
                focus_surface: None,
                focused,
            };
            match *surface {
                Surface::Morphology => self.neuroml.cable_dock_view(ui),
                Surface::Editor => self.neuroml.editor_dock_view(ui),
                Surface::Grow => self.grow.grow_dock_view(ui),
                Surface::GrowPlots => self.grow.plots_dock_view(ui),
                _ => self.mode.dock_surface(ui, *surface, &mut services),
            }
            services.focus_surface
        };
        if let Some(surface) = focus_request {
            self.focus_surface(surface);
        }
    }

    fn id(&mut self, surface: &mut Self::Tab) -> egui::Id {
        egui::Id::new(("workspace_surface", surface.label()))
    }

    /// Поверхность «закрывается» только из плавающего окна; на основных
    /// вкладках кнопка скрыта (`show_close_buttons(false)`), так что `true`
    /// здесь управляет именно окном.
    fn is_closeable(&self, _surface: &Self::Tab) -> bool {
        true
    }

    /// Разрешаем вытаскивать вкладку в плавающее окно (undock).
    fn allowed_in_windows(&self, _surface: &mut Self::Tab) -> bool {
        true
    }

    /// Закрытие плавающего окна не удаляет поверхность, а возвращает её в
    /// дерево: запоминаем вкладку, `dock_area` кладёт её обратно в focused
    /// leaf после рендера.
    fn on_close(&mut self, surface: &mut Self::Tab) -> OnCloseResponse {
        self.return_to_dock.push(*surface);
        OnCloseResponse::Close
    }
}

/// Дефолтная «рабочая» раскладка — **голая** (Blender-пресет Layout):
/// инспектор слева + атлас в центре. Остальные поверхности (ego, морфология,
/// редактор, конструктор, плоттер/консоль/сим) открываются по требованию
/// через меню «Окна»/«Вид» — дефолт не должен быть кашей из девяти вкладок.
fn default_dock() -> DockState<Surface> {
    let mut dock = DockState::new(vec![Surface::Inspector]);
    dock.main_surface_mut()
        .split_right(NodeIndex::root(), 0.22, vec![Surface::Atlas]);
    dock
}

/// Пресет «Все панели» — старая полная раскладка: инспектор слева, центр
/// (атлас/ego/морфология/редактор/конструктор) + нижняя лента
/// (плоттер/консоль/сим). Доступен из меню «Вид», дефолт — чистый [`default_dock`].
fn full_dock() -> DockState<Surface> {
    let mut dock = DockState::new(vec![Surface::Inspector]);
    let [_, center] = dock.main_surface_mut().split_right(
        NodeIndex::root(),
        0.22,
        vec![
            Surface::Atlas,
            Surface::Ego,
            Surface::Grow,
            Surface::GrowPlots,
            Surface::Morphology,
            Surface::Editor,
            Surface::Builder,
        ],
    );
    dock.main_surface_mut().split_below(
        center,
        0.72,
        vec![Surface::Plots, Surface::Console, Surface::Sim],
    );
    dock
}

impl eframe::App for ConnApp {
    /// Один кадр — обёрнут в `catch_unwind`: паника в любом коде кадра
    /// (пикинг, ego, симуляция, атласы) больше не убивает процесс; причина
    /// попадает в `connscope_panic.log` через panic hook из `main`.
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.ui_frame(ui, frame);
        }));
        if result.is_err() {
            eprintln!("[connscope] frame crashed, continuing");
            ui.ctx().request_repaint();
        }
    }
}

/// Точка входа GUI: окно 1560×960 с загруженным VNC.
///
/// Рендерер — **wgpu**: нужен для шейдерного облака точек
/// (`splat`). Формат поверхности берём из `CreationContext::wgpu_render_state`.
#[allow(clippy::field_reassign_with_default)] // gpu_target_format известен только из cc
pub fn run() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1560.0, 960.0])
            .with_title("neoMyelin — Excel для коннектомики")
            .with_icon(window_icon()),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "neoMyelin",
        options,
        Box::new(|cc| {
            // Иконочный шрифт Phosphor (Regular + Bold): добавляем через
            // `add_font` поверх существующих шрифтов с наивысшим приоритетом —
            // глифы иконок берутся из него, буквы/кириллица — из основных
            // шрифтов. Bold нужен: elegance сам делает заголовки карточек
            // `.strong()`, и жирный глиф иконки ищется в bold-наборе.
            // Ничего не затираем (в отличие от `set_fonts`).
            for (name, variant) in [
                ("phosphor", egui_phosphor::Variant::Regular),
                ("phosphor-bold", egui_phosphor::Variant::Bold),
            ] {
                cc.egui_ctx.add_font(egui::epaint::text::FontInsert::new(
                    name,
                    egui::FontData::from_static(variant.font_bytes()),
                    vec![egui::epaint::text::InsertFontFamily {
                        family: egui::FontFamily::Proportional,
                        priority: egui::epaint::text::FontPriority::Highest,
                    }],
                ));
            }
            // Шрифты (Plex/JetBrains Mono) + палитра «прибора» — после
            // Phosphor: наборы глифов не пересекаются, порядок не важен.
            shell::theme::install(&cc.egui_ctx);
            let mut app = ConnApp::default();
            // Логотип neoMyelin — декодируем PNG один раз в текстуру (белый,
            // тонируется под тему в «О программе»).
            app.ui.logo_tex = load_logo_texture(&cc.egui_ctx);
            // Headless-отладка вёрстки: NEOMYELIN_PROFILE_LAYOUT=1 → захватить
            // первый кадр и написать connscope_profiler.json (без GUI-кликов).
            if std::env::var_os("NEOMYELIN_PROFILE_LAYOUT").is_some() {
                shell::profiler::arm();
            }
            // Формат поверхности нужен пайплайну облака точек (wgpu паникует
            // на неверном формате) — берём его из RenderState при старте
            // и передаём режиму через трейт.
            app.mode
                .set_gpu_target_format(cc.wgpu_render_state.as_ref().map(|rs| rs.target_format));
            Ok(Box::new(app))
        }),
    )
}

/// Иконка окна (таскбар/заголовок): белый знак neoMyelin на тёмно-индиго фоне.
fn window_icon() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/branding/neoMyelin_icon.png"
    )))
    .expect("neoMyelin_icon.png должен декодироваться")
}

/// Декод PNG-словознака neoMyelin в текстуру egui (белый на прозрачном —
/// в «О программе» тонируется под цвет текста темы).
fn load_logo_texture(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/branding/neoMyelin.png"
    ));
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    Some(ctx.load_texture("neoMyelin_logo", color, egui::TextureOptions::LINEAR))
}

fn load_file_icon(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<egui::TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

// ── Тесты раскладки рабочего пространства ─────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> ConnApp {
        ConnApp {
            ui: UiState::default(),
            mode: Box::new(ConnectomeMode::default()),
            neuroml: NeuromlMode::default(),
            grow: GrowMode::default(),
            dock: default_dock(),
            dock_loaded: false,
            return_to_dock: Vec::new(),
        }
    }

    /// Все вкладки дерева, отсортированные по дискриминанту `Surface` —
    /// канонический порядок совпадает с `Surface::ALL`.
    fn sorted_tabs(dock: &DockState<Surface>) -> Vec<Surface> {
        let mut v: Vec<Surface> = dock.iter_all_tabs().map(|(_, t)| *t).collect();
        v.sort_by_key(|s| *s as u8);
        v
    }

    #[test]
    fn default_dock_is_lean() {
        // Дефолт «Рабочий стол» — чистый: инспектор + атлас, без дублей.
        let tabs = sorted_tabs(&default_dock());
        assert_eq!(tabs, vec![Surface::Inspector, Surface::Atlas]);
        // Полная раскладка «Все панели» — все поверхности ровно по разу.
        let tabs = sorted_tabs(&full_dock());
        assert_eq!(
            tabs.len(),
            Surface::ALL.len(),
            "дубль/потеря вкладки: {tabs:?}"
        );
        for (i, surface) in Surface::ALL.iter().enumerate() {
            assert_eq!(
                tabs[i], *surface,
                "вкладка {surface:?} не на месте/отсутствует"
            );
        }
    }

    #[test]
    fn presets_have_expected_surfaces() {
        let mut app = test_app();

        app.apply_preset(DockPreset::Minimal);
        assert_eq!(sorted_tabs(&app.dock), vec![Surface::Atlas]);

        app.apply_preset(DockPreset::Tabs);
        assert_eq!(sorted_tabs(&app.dock).len(), Surface::ALL.len());

        // Дефолт «Рабочий стол» — чистый: инспектор + атлас.
        app.apply_preset(DockPreset::Workspace);
        assert_eq!(
            sorted_tabs(&app.dock),
            vec![Surface::Inspector, Surface::Atlas]
        );

        // «Все панели» — старая полная раскладка со всеми поверхностями.
        app.apply_preset(DockPreset::Full);
        assert_eq!(sorted_tabs(&app.dock).len(), Surface::ALL.len());
    }

    #[test]
    fn focus_surface_activates_requested_tab() {
        let mut app = test_app();
        for surface in Surface::ALL {
            app.focus_surface(surface);
            assert_eq!(app.focused_surface(), Some(surface), "фокус {surface:?}");
        }
    }

    #[test]
    fn cycle_surface_wraps_forward_and_backward() {
        let mut app = test_app();
        app.focus_surface(Surface::Inspector);
        app.cycle_surface(true);
        assert_eq!(app.focused_surface(), Some(Surface::Atlas));
        app.cycle_surface(false);
        assert_eq!(app.focused_surface(), Some(Surface::Inspector));

        // Назад с первой вкладки — оборачиваемся на последнюю.
        app.focus_surface(Surface::ALL[0]);
        app.cycle_surface(false);
        assert_eq!(
            app.focused_surface(),
            Some(Surface::ALL[Surface::ALL.len() - 1])
        );
    }
}
