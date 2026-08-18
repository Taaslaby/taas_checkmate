//! # Режимы приложения (`app::mode`)
//!
//! Трейт [`Mode`] — контракт режима внутри рамки приложения (shell).
//! Режим владеет своим состоянием и рисует свои поверхности; shell вызывает
//! хуки в фиксированном порядке кадра и передаёт [`ShellServices`] (сервисы
//! оболочки) тем хукам, которым они нужны.
//!
//! **Новый режим = новая структура + `impl Mode`** — оболочка не меняется:
//!
//! ```text
//! shell (ConnApp)                     режим (Mode)
//! ├── меню Файл/Вид/Справка           ├── top_bar      (верхняя панель)
//! ├── статус-бар (рамка, FPS)         ├── rail         (левый рейл)
//! ├── модалки «О программе/Открыть/   ├── dock_surface (поверхности в дереве
//! │   Горячие клавиши»                │                  egui_dock: инспектор,
//! ├── дерево egui_dock (владеет shell)│                  атлас, ego, плоттер,
//! ├── тема/акцент/пресеты раскладки   │                  консоль, сим)
//! └── ─                              ├── windows      (окна/модалки режима)
//!                                     └── status_left/status_right
//! ```
//!
//! Хуки срабатывают в фиксированном порядке кадра (`frame_tick` → `top_bar` →
//! `rail` → дерево дока → окна), поэтому раскладка не зависит от активного
//! режима. Само дерево (`DockState<Surface>`) владеет оболочка — это
//! позволяет вкладывать кросс-режимные поверхности (например «Морфология» из
//! `NeuromlMode`).

use std::path::Path;

use eframe::egui;
use egui_phosphor::regular as ic;

/// Докируемая поверхность рабочего пространства — вкладка в едином дереве
/// `egui_dock`, которым владеет оболочка. ConnectomeMode рисует первые шесть,
/// [`Surface::Morphology`] рисует shell из `NeuromlMode` (кросс-режимная
/// поверхность).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) enum Surface {
    /// Правая панель-инспектор нейрона/симуляции/фармы.
    Inspector,
    /// Атлас всего мозга (2D/3D).
    Atlas,
    /// Большой ego-граф выбранного нейрона.
    Ego,
    /// Плоты результата симуляции.
    Plots,
    /// Встроенная консоль-комбайн.
    Console,
    /// Статус/управление симуляцией.
    Sim,
    /// Дерево компартментов морфо-клетки (NeuroML).
    Morphology,
}

impl Surface {
    pub(crate) const ALL: [Surface; 7] = [
        Surface::Inspector,
        Surface::Atlas,
        Surface::Ego,
        Surface::Plots,
        Surface::Console,
        Surface::Sim,
        Surface::Morphology,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Inspector => "Инспектор",
            Self::Atlas => "Атлас",
            Self::Ego => "Ego-граф",
            Self::Plots => "Плоттер",
            Self::Console => "Консоль",
            Self::Sim => "Симуляция",
            Self::Morphology => "Морфология",
        }
    }

    pub(crate) const fn icon(self) -> &'static str {
        match self {
            Self::Inspector => ic::BRAIN,
            Self::Atlas => ic::COMPASS,
            Self::Ego => ic::NETWORK,
            Self::Plots => ic::CHART_LINE,
            Self::Console => ic::TERMINAL,
            Self::Sim => ic::LIGHTNING,
            Self::Morphology => ic::TREE_STRUCTURE,
        }
    }
}

/// Сервисы оболочки, доступные режиму. Сейчас это модалки оболочки и запрос
/// фокуса вкладки рабочего пространства; новые сервисы добавляются сюда по
/// мере появления потребностей — режим не знает про `ConnApp` напрямую.
pub(crate) struct ShellServices<'a> {
    /// Открыть модалку «Горячие клавиши» (рисует оболочка).
    pub show_shortcuts: &'a mut bool,
    /// Режим просит сфокусировать вкладку [`Surface`] в дереве дока; оболочка
    /// применяет запрос после вызова хука.
    pub focus_surface: Option<Surface>,
}

/// Режим приложения: владеет своим состоянием и рисует свои поверхности.
///
/// Порядок вызовов в кадре (задаёт оболочка в `ConnApp::ui_frame`):
/// `frame_tick` → `view_menu` → `top_bar` → `rail` →
/// `status_left`/`status_right` → `windows` → `dock_surface`.
/// Методы с реализацией по умолчанию — опциональные поверхности.
pub(crate) trait Mode {
    /// Покадровый тик до раскладки панелей: камера, живые потоки, repaint.
    fn frame_tick(&mut self, _ctx: &egui::Context) {}

    /// Верхняя панель (одна строка под меню): вкладки режима, поиск, действия.
    fn top_bar(&mut self, _ui: &mut egui::Ui, _services: &mut ShellServices) {}

    /// Левый рейл инструментов.
    fn rail(&mut self, _ui: &mut egui::Ui) {}

    /// Докируемая поверхность рабочего пространства (см. [`Surface`]).
    /// Рисуется внутри единого дерева `egui_dock`, которым владеет оболочка.
    fn dock_surface(&mut self, _ui: &mut egui::Ui, _surface: Surface, _services: &mut ShellServices) {}

    /// Левая часть статус-бара: hover-инфо или контекстная подсказка режима.
    fn status_left(&self) -> String {
        String::new()
    }

    /// Правая часть статус-бара: счётчики данных, зум, статус симуляции.
    fn status_right(&mut self, _ui: &mut egui::Ui) {}

    /// Плавающие окна и модалки режима (рисуются поверх панелей). Может
    /// попросить оболочку сфокусировать [`Surface`] через [`ShellServices`].
    fn windows(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame, _services: &mut ShellServices) {}

    /// Режимная секция меню «Вид». Возвращает [`Surface`], которую режим
    /// просит сфокусировать (оболочка применяет запрос после вызова).
    fn view_menu(&mut self, _ui: &mut egui::Ui) -> Option<Surface> {
        None
    }

    /// Горячие клавиши режима. Возвращает `true`, если клавиша обработана.
    fn handle_shortcuts(&mut self, _ctx: &egui::Context) -> bool {
        false
    }

    /// Загрузка данных режима (Файл → Открыть / Ctrl+R).
    fn load_data(&mut self, _path: &Path) {}

    /// Статистика данных для модалки «О программе» (нейроны, синапсы).
    fn data_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Передача формата поверхности wgpu (для шейдерного рендера режима).
    fn set_gpu_target_format(&mut self, _format: Option<wgpu::TextureFormat>) {}
}
