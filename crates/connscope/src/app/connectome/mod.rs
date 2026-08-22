//! # Режим «Коннектом» (`app::connectome`)
//!
//! Текущий вьюер коннектома: камера/проекции, пикинг и фильтры,
//! отрисовка атласа, ego-граф, инспектор нейрона, UI симуляций
//! и плоттеры (plotlib + cards).
//!
//! Режим реализует [`Mode`]: оболочка вызывает его поверхности через трейт,
//! а всё состояние режима живёт в [`ConnectomeMode`] (см. `state.rs`).

use std::path::Path;

use eframe::egui;

use crate::app::connectome::state::{
    CameraState, DisinState, Filters, LiveState, PharmaState, Selection, SimState,
};
use crate::app::mode::{Mode, ShellServices, Surface};
use crate::app::shell::console::Console;
use crate::io::data::MancData;

pub(crate) mod builder;
pub(crate) mod camera;
pub(crate) mod canvas;
pub(crate) mod cards;
pub(crate) mod ego;
pub(crate) mod inspector;
pub(crate) mod picking;
pub(crate) mod plotlib;
pub(crate) mod plots;
pub(crate) mod sim_ui;
pub(crate) mod state;
// Совместимость: старый путь `connectome::neuron_builder::...` → `builder::...`
pub(crate) use builder as neuron_builder;

/// Режим «Коннектом»: все данные и UI вьюера коннектома.
///
/// Раньше эти поля висели в `ConnApp` (~50 полей god-объекта). Теперь режим
/// самодостаточен: оболочка видит только его поверхности (трейт [`Mode`]),
/// а добавить новый режим — значит написать вторую структуру с `impl Mode`.
pub(crate) struct ConnectomeMode {
    // ── данные ──
    pub data: Option<MancData>,
    /// Статус загрузки/ошибки (показывается в статус-баре и на пустом канвасе).
    pub status: String,

    // ── компоненты состояния (см. `state`) ──
    pub camera: CameraState,
    pub filters: Filters,
    pub sel: Selection,
    pub sim: SimState,
    pub pharma: PharmaState,
    pub live: LiveState,
    pub disin: DisinState,

    // ── конструктор нейрона (нодовый редактор Formula) ──
    pub builder: builder::NeuronBuilder,

    // ── библиотека плоттеров (включение / открепление) ──
    pub plots: plotlib::PlotLib,

    // ── встроенная консоль (комбайн): команды работают с данными режима ──
    pub console: Console,

    // ── UI режима (окна, рейл) ──
    /// ego-граф откреплён в перетаскиваемое окно (иначе — вкладка дока Ego).
    pub ego_undocked: bool,
    /// Разовый «⟳ Fit» для ego-графа (включает fit на один кадр).
    pub ego_fit_once: bool,
    /// Модалка «Плоттеры» открыта.
    pub show_plots: bool,
    /// Левый рейл инструментов (Blender-style) виден.
    pub rail_visible: bool,
    /// Попап рейла «Фильтры»: секция НТ развёрнута.
    pub filters_nt_open: bool,
    /// Попап рейла «3D»: секция ракурсов развёрнута.
    pub cam_views_open: bool,
}

impl Default for ConnectomeMode {
    fn default() -> Self {
        Self {
            data: None,
            status: String::new(),
            camera: CameraState::default(),
            filters: Filters::default(),
            sel: Selection::default(),
            sim: SimState::default(),
            pharma: PharmaState::default(),
            live: LiveState::default(),
            disin: DisinState::default(),
            builder: builder::NeuronBuilder::default(),
            plots: plotlib::PlotLib::new(),
            console: Console::new(),
            ego_undocked: false,
            ego_fit_once: false,
            show_plots: false,
            rail_visible: true,
            filters_nt_open: true,
            cam_views_open: true,
        }
    }
}

impl Mode for ConnectomeMode {
    /// Покадровый тик: обновление ppp для шейдеров, автовращение 3D, живой поток.
    fn frame_tick(&mut self, ctx: &egui::Context) {
        self.camera.ppp = ctx.pixels_per_point(); // для шейдерного рендера (размер точек)
                                                  // Автовращение 3D-облака.
        if self.camera.view3d && self.camera.spin3d {
            self.camera.yaw += 0.004;
            // GPU orbit updates uniforms without rebuilding the CPU atlas.
            // CPU fallback stays static while spinning to avoid allocating a
            // new 1024² texture every frame.
            if self.camera.use_gpu3d {
                self.camera.atlas3d = None;
            }
            ctx.request_repaint();
        }
        // Живой сигнал: потребляем шаги потока → свечение, гасим хвосты.
        self.tick_live(ctx);
    }

    /// Верхняя панель: вкладки Атлас/Ego, поиск, действия камеры.
    fn top_bar(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        self.top_panel(ui, services);
    }

    /// Левый рейл инструментов: проекции, 3D, окраска, фильтры, Fit.
    fn rail(&mut self, ui: &mut egui::Ui) {
        self.tool_rail(ui);
    }

    /// Докируемые поверхности режима: инспектор, атлас, ego, плоттер/консоль/сим.
    fn dock_surface(&mut self, ui: &mut egui::Ui, surface: Surface, services: &mut ShellServices) {
        match surface {
            Surface::Inspector => self.inspector_content(ui, services),
            Surface::Atlas => self.atlas_canvas(ui),
            Surface::Ego => {
                if self.ego_undocked {
                    ui.weak("Ego-граф открыт в отдельном окне — закрой его, чтобы вернуть сюда.");
                } else {
                    self.ego_view(ui);
                }
            }
            Surface::Plots => self.dock_tab_plots(ui),
            Surface::Console => self.console_view(ui),
            Surface::Sim => self.dock_tab_sim(ui),
            Surface::Builder => self.builder.dock_view(ui),
            // Рисуют оболочка из NeuromlMode/GrowMode; сюда не приходят.
            Surface::Morphology | Surface::Editor | Surface::Grow | Surface::GrowPlots => {}
        }
    }

    /// Левая часть статус-бара: hover-инфо о нейроне или контекстная подсказка.
    fn status_left(&self) -> String {
        if let Some(h) = self.sel.hover {
            if let Some(d) = self.data.as_ref() {
                if h < d.n {
                    return format!(
                        "#{h} · {} · {} · {} · deg={}",
                        if d.type_name(h).is_empty() {
                            "?"
                        } else {
                            d.type_name(h)
                        },
                        d.klass(h).name(),
                        d.nt(h).name(),
                        d.out_deg(h)
                    );
                }
            }
        }
        self.context_hint()
    }

    /// Правая часть статус-бара: счётчики данных, зум, статус симуляции.
    fn status_right(&mut self, ui: &mut egui::Ui) {
        if let Some(d) = &self.data {
            ui.label(egui::RichText::new(format!("{}/{}", d.n, d.s)).weak());
        }
        ui.label(egui::RichText::new(format!("zoom {:.0}%", self.camera.zoom * 100.0)).weak());
        if !self.sim.status.is_empty() {
            use egui_phosphor::regular as ic;
            ui.label(
                egui::RichText::new(format!("{} {}", ic::LIGHTNING, self.sim.status))
                    .color(crate::app::shell::theme::accent(ui))
                    .weak(),
            );
        }
        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).weak());
        }
    }

    /// Окна и модалки режима: «Плоттеры», откреплённый ego-граф, плоты в окнах ОС.
    fn windows(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
        _services: &mut ShellServices,
    ) {
        if self.show_plots {
            self.plots_library_modal(ctx);
        }
        if self.ego_undocked {
            self.ego_window(ctx);
        }
        self.plots_windows(ctx);
    }

    /// Режимная секция меню «Вид»: камера, проекции, инструменты.
    fn view_menu(&mut self, ui: &mut egui::Ui) -> Option<Surface> {
        self.view_menu(ui)
    }

    /// Горячие клавиши режима: F (fit), +/− (зум), H (рейл), [ ] (док),
    /// F12 (консоль), Ctrl+S (поиск).
    fn handle_shortcuts(&mut self, ctx: &egui::Context) -> bool {
        self.handle_shortcuts(ctx)
    }

    /// Загрузка коннектома (.bin/.taas/.graphml/.npz) с консольным отчётом.
    fn load_data(&mut self, path: &Path) {
        self.load_data(path);
    }

    /// Статистика данных для «О программе».
    fn data_stats(&self) -> (usize, usize) {
        self.data.as_ref().map(|d| (d.n, d.s)).unwrap_or((0, 0))
    }

    /// Формат поверхности wgpu — нужен шейдерному облаку точек (splat).
    fn set_gpu_target_format(&mut self, format: Option<wgpu::TextureFormat>) {
        self.camera.gpu_target_format = format;
    }
}

impl ConnectomeMode {
    /// Загружает `.bin`/`.taas`/`.graphml`/`.npz` (формат определяется по
    /// magic в `MancData::detect`), сбрасывает выбор/симуляцию/кеши атласов
    /// и обновляет строку статуса. Путь хранит оболочка (`ConnApp::ui`).
    pub(crate) fn load_data(&mut self, path: &Path) {
        self.stop_live(); // живой поток живёт на старых данных — гасим
        match MancData::detect(path) {
            Ok(d) => {
                let npos = d.has_pos.iter().filter(|&&x| x != 0).count();
                // Полная статистика — в консоль (одноразовое подтверждение
                // загрузки). В статус-баре остаётся только компактный счётчик
                // нейронов/синапсов — иначе одно и то же число висит в двух
                // местах и растягивает строку.
                self.console.push_text(&format!(
                    "✓ загружено: {} нейронов, {} синапсов, {} с позицией, {} типов клеток",
                    d.n(),
                    d.s(),
                    npos,
                    d.types.len()
                ));
                self.status.clear(); // ошибок нет — убираем прошлый статус
                self.filters.color_mode = crate::core::color::ColorMode::auto(&d); // мозги без НТ не серые
                self.camera.lod = crate::render::lod::build_lod(&d); // уровни детализации
                let n = d.n();
                self.data = Some(d);
                self.sel.selected = None;
                self.sel.hover = None;
                self.sel.ego.clear();
                self.sim.result = None;
                self.live.glow = vec![0.0; n];
                self.camera.atlas = None;
                self.camera.atlas3d = None;
                // Кеш GPU-облака живёт на старых данных — сбрасываем
                // (иначе новый мозг с теми же фильтрами покажет старые точки).
                self.camera.splat_cache = None;
                // На следующем кадре канваса — мгновенный фит (центр по
                // фактическому размеру канваса).
                self.camera.fit_pending = true;
            }
            Err(e) => {
                self.status = format!("⚠ {e}");
            }
        }
    }
}
