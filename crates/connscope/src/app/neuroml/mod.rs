//! # Режим «NeuroML» (`app::neuroml`)
//!
//! Загрузка и прогон NeuroML-моделей (CPU SIMD / GPU CUDA): модалка
//! «NeuroML → pillars» (см. `ui`), окно-IDE «Редактор» и поверхность
//! «Морфология» (вкладка дерева egui_dock, рисуется через `cable_dock_view`).
//!
//! ```text
//! NeuromlMode (режим)
//! ├── show     модалка открыта (меню «Файл → Открыть NeuroML»)
//! ├── path     путь к .nml
//! ├── loaded   компилированная CPU-сеть (pillars::api::neuroml)
//! ├── gpu      GPU-зеркало (feature cuda, опционально)
//! └── …        метрики, бэкенд, результат, статус
//! ```

use std::path::PathBuf;

use eframe::egui;

#[cfg(feature = "cuda")]
use crate::sim::neuroml::GpuNetwork;
use pillars::api::neuroml::{CableInfo, LoadedNetwork};

use crate::app::mode::{Mode, ShellServices};
use crate::sim::neuroml::{Backend, CableTraces, RunResult};

pub(crate) mod cable;
pub(crate) mod editor;
pub(crate) mod highlight;
pub(crate) mod ui;

use editor::EditorState;

/// Состояние виджета «Морфология»: геометрия кабельной популяции + трассы.
pub(crate) struct CableState {
    /// Population id of the cable population in the loaded network.
    pub pop: usize,
    pub info: CableInfo,
    /// V(t) per compartment (после «Запустить и записать трассы»).
    pub traces: Option<CableTraces>,
    /// Выбранный (кликом) компартмент.
    pub selected: Option<usize>,
    /// Позиция слайдера времени (индекс сэмпла трассы).
    pub t_sample: usize,
    /// Окрашивать дерево по V(t) на слайдере, а не по конечному V.
    pub color_by_time: bool,
    /// Band-layout: y-центр полосы каждого компартмента (µm).
    pub layout_y: Vec<f32>,
    /// Масштаб (px/µm) и панорама (px) окна.
    pub zoom: f32,
    pub pan: egui::Vec2,
}

/// Режим «NeuroML → pillars»: состояние загруженной модели + модалка.
pub(crate) struct NeuromlMode {
    /// Модалка открыта (меню «Файл → Открыть NeuroML (.nml)…»).
    pub show: bool,
    pub path: PathBuf,
    /// Компилированная CPU-сеть (одна и та же для обоих бэкендов).
    pub loaded: Option<LoadedNetwork>,
    /// GPU-зеркало сети (собирается при загрузке, если есть CUDA).
    /// Существует только с feature `cuda`.
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuNetwork>,
    pub gpu_warning: Option<String>,
    pub pop_sizes: Vec<usize>,
    pub n_neurons: usize,
    pub n_syn: usize,
    pub backend: Backend,
    pub result: Option<RunResult>,
    pub status: String,
    /// Виджет «Морфология» (вкладка с деревом компартментов) — если
    /// загружена морфо-клетка.
    pub cable: Option<CableState>,
    /// IDE-виджет «Редактор» (подсветка + валидация + ошибки).
    pub editor: EditorState,
    /// Окно «Редактор» открыто.
    pub show_editor: bool,
}

impl Default for NeuromlMode {
    fn default() -> Self {
        Self {
            show: false,
            path: PathBuf::new(),
            loaded: None,
            #[cfg(feature = "cuda")]
            gpu: None,
            gpu_warning: None,
            pop_sizes: Vec::new(),
            n_neurons: 0,
            n_syn: 0,
            backend: Backend::Cpu,
            result: None,
            status: String::new(),
            cable: None,
            editor: EditorState::default(),
            show_editor: false,
        }
    }
}

impl Mode for NeuromlMode {
    /// Модалка «NeuroML → pillars» + окно «Редактор». Поверхность
    /// «Морфология» рисует оболочка через `cable_dock_view` (вкладка дока).
    fn windows(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame, services: &mut ShellServices) {
        self.neuroml_modal(ctx, frame, services);
        self.editor_window(ctx);
    }
}
