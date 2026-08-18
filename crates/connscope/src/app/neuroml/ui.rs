//! # NeuroML → pillars (модалка + запуск) (`app::neuroml_ui`)
//!
//! Окно «NeuroML → pillars»: загрузи `.nml` (или возьми встроенный пример),
//! выбери бэкенд (CPU SIMD / GPU CUDA) и прогони сеть. Логика живёт в
//! [`crate::sim::neuroml`] — здесь только UI и диспетчеризация действий.
//!
//! * `neuroml_modal` — сама модалка (паттерн [`super::menu`], локальные копии
//!   полей + запись обратно после закрытия, чтобы не держать `self` в closure);
//! * `neuroml_load` / `neuroml_load_sample` / `neuroml_run` — действия.

use std::path::{Path, PathBuf};

use eframe::egui::{self};
#[cfg(feature = "cuda")]
use elegance::SegmentedControl;
use elegance::{Button as ElegButton, Modal, StatCard};

use egui_phosphor::regular as ic;

use crate::sim::neuroml::{self, Backend};

use crate::app::mode::{ShellServices, Surface};
use crate::app::neuroml::NeuromlMode;
use crate::app::tokens::{self, icon_label};
use crate::app::widgets::{action_button, ActionTone};

/// Встроенный пример (exc↔inh цикл) — тот же файл, что в
/// `resources/examples/neuroml/exc_inh_loop.nml`, зашит на этапе компиляции.
pub(crate) const SAMPLE_NML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/examples/neuroml/exc_inh_loop.nml"
));

/// Курсовой пресет: классический Hodgkin–Huxley аксон под импульсом
/// (`resources/examples/neuroml/hh_pulse.nml`).
pub(crate) const HH_SAMPLE_NML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/examples/neuroml/hh_pulse.nml"
));

/// Канонический пример с GitHub (NeuroML/NML2_LEMS_Examples): фигура 4 из
/// Brette & Gerstner 2005 — regular-spiking и bursting AdEx под импульсами.
/// Проверяет нормализацию единиц AdEx (nA→мВ через gL).
pub(crate) const ADEX_SAMPLE_NML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/examples/neuroml/Figure4_AdEx.nml"
));

/// Встроенный пример NeuroML (текст примеров живёт здесь же, рядом с кнопками
/// загрузки). Общий для модалки и редактора.
#[derive(Clone, Copy)]
pub(crate) enum SampleKind {
    /// exc ↔ inh цикл (Izhikevich).
    ExcInh,
    /// Классический HH-аксон.
    HhAxon,
    /// AdEx (Brette & Gerstner 2005, фиг. 4).
    AdEx,
}

/// Действие, выбранное в модалке «NeuroML → pillars». Возвращается из closure
/// (пока `self` занят локальными копиями полей) и выполняется после —
/// типизированно, без магических `u8`-кодов.
#[derive(Clone, Copy)]
enum NeuromlAction {
    /// Загрузить `.nml` из поля пути.
    LoadFile,
    /// Прогнать загруженную сеть.
    Run,
    /// Загрузить встроенный пример.
    LoadSample(SampleKind),
    /// Закрыть модалку.
    Close,
    /// Открыть окно «Морфология».
    OpenCable,
    /// Открыть окно «Редактор».
    OpenEditor,
}

// Ширины кнопок примеров — единые для модалки и редактора.
const SAMPLE_BTN_W: f32 = 90.0;
const SAMPLE_ADEX_BTN_W: f32 = 70.0;

/// Три встроенных примера одной строкой кнопок (единый стиль `action_button`,
/// без палитры elegance). Возвращает выбранный пример — одно нажатие за кадр.
pub(crate) fn sample_buttons(ui: &mut egui::Ui) -> Option<SampleKind> {
    let mut chosen = None;
    if action_button(ui, "", "exc ↔ inh", ActionTone::Neutral, SAMPLE_BTN_W)
        .on_hover_text("Встроенный пример: 2 популяции + проекции + импульс")
        .clicked()
    {
        chosen = Some(SampleKind::ExcInh);
    }
    if action_button(ui, "", "HH-аксон", ActionTone::Neutral, SAMPLE_BTN_W)
        .on_hover_text("Классический HH 1952: Na/K через channelDensity + импульс → спайки")
        .clicked()
    {
        chosen = Some(SampleKind::HhAxon);
    }
    if action_button(ui, "", "AdEx", ActionTone::Neutral, SAMPLE_ADEX_BTN_W)
        .on_hover_text(
            "Каноническая фигура 4 (Brette & Gerstner 2005): regular-spiking + bursting AdEx",
        )
        .clicked()
    {
        chosen = Some(SampleKind::AdEx);
    }
    chosen
}

impl NeuromlMode {
    // ── модалка ────────────────────────────────────────────────
    pub(crate) fn neuroml_modal(
        &mut self,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
        services: &mut ShellServices,
    ) {
        let mut open = self.show;
        let mut path = self.path.display().to_string();
        let mut backend = self.backend;

        // Снапшот состояния для показа внутри closure (не держим `self`).
        let n_pops = self.pop_sizes.len();
        let n_neurons = self.n_neurons;
        let n_syn = self.n_syn;
        // Доступен ли GPU-бэкенд: с feature `cuda` — наличие скомпилированного
        // зеркала, без неё — всегда false (метод учтёт cfg сам).
        #[cfg(feature = "cuda")]
        let gpu_available = self.gpu.is_some() && Backend::Gpu.available();
        #[cfg(not(feature = "cuda"))]
        let gpu_available = Backend::Gpu.available(); // false: сборка без cuda
        let gpu_warning = self.gpu_warning.clone();
        #[cfg(not(feature = "cuda"))]
        let _ = gpu_warning;
        let status = self.status.clone();
        let result = self.result.clone();

        let mut picker_rx: Option<std::sync::mpsc::Receiver<Option<PathBuf>>> = None;
        let mut action: Option<NeuromlAction> = None;

        Modal::new("neuroml_modal", &mut open)
            .heading(icon_label(
                ic::CIRCUITRY,
                "NeuroML → pillars",
                tokens::ICON_SM,
            ))
            .subtitle("Загрузи .nml и прогони сеть на CPU SIMD или GPU (CUDA)")
            .header_icon(ic::CIRCUITRY)
            .max_width(660.0)
            .show(ctx, |ui| {
                // Курсовые примеры одним кликом (зашиты в бинарник) — единый
                // стиль кнопок с редактором (см. sample_buttons).
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Пример:").weak());
                    if let Some(kind) = sample_buttons(ui) {
                        action = Some(NeuromlAction::LoadSample(kind));
                    }
                });
                ui.add_space(tokens::SP_SM);

                // Путь + нативный диалог ОС (rfd) с фильтром .nml.
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut path).desired_width(500.0));
                    if ui
                        .add(ElegButton::new(icon_label(
                            ic::FOLDER_OPEN,
                            "Обзор…",
                            tokens::ICON_SM,
                        )))
                        .on_hover_text("Нативный диалог Windows (фильтр .nml)")
                        .clicked()
                    {
                        let start_dir = if path.trim().is_empty() {
                            None
                        } else {
                            Path::new(path.trim()).parent().map(|p| p.to_path_buf())
                        };
                        let (tx, rx) = std::sync::mpsc::channel();
                        let mut dlg = rfd::FileDialog::new()
                            .add_filter("NeuroML", &["nml", "xml"])
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

                // Бэкенд: CPU SIMD / GPU CUDA.
                #[cfg(feature = "cuda")]
                {
                    let mut bi = if backend == Backend::Gpu { 1 } else { 0 };
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Бэкенд:").weak());
                        ui.add(SegmentedControl::new(
                            &mut bi,
                            [Backend::Cpu.label(), Backend::Gpu.label()],
                        ));
                    });
                    backend = if bi == 1 { Backend::Gpu } else { Backend::Cpu };
                }
                // Без feature `cuda`: выбора нет — CPU, GPU показан как отключенный.
                #[cfg(not(feature = "cuda"))]
                {
                    backend = Backend::Cpu;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Бэкенд:").weak());
                        ui.label(Backend::Cpu.label());
                        ui.label(
                            egui::RichText::new("GPU (CUDA) — недоступен (сборка без cuda)")
                                .weak()
                                .small(),
                        );
                    });
                }
                if backend == Backend::Gpu && !gpu_available {
                    ui.label(
                        egui::RichText::new(format!(
                            "GPU недоступен: {}",
                            gpu_warning
                                .clone()
                                .unwrap_or_else(|| "нет драйвера CUDA".into())
                        ))
                        .weak()
                        .small(),
                    );
                }
                ui.add_space(tokens::SP_SM);

                // Метрики загруженной модели.
                ui.horizontal(|ui| {
                    ui.add(
                        StatCard::new("Популяций")
                            .value(n_pops.to_string())
                            .width(170.0),
                    );
                    ui.add(
                        StatCard::new("Нейронов")
                            .value(n_neurons.to_string())
                            .width(170.0),
                    );
                    ui.add(
                        StatCard::new("Синапсов")
                            .value(n_syn.to_string())
                            .width(170.0),
                    );
                });
                ui.add_space(tokens::SP_SM);

                // Действия — единый язык кнопок (акцент темы для первичных,
                // нейтральный для закрытия), без палитры elegance.
                ui.horizontal(|ui| {
                    if crate::app::widgets::action_button(
                        ui,
                        ic::DOWNLOAD,
                        "Загрузить",
                        crate::app::widgets::ActionTone::Primary,
                        130.0,
                    )
                    .clicked()
                    {
                        action = Some(NeuromlAction::LoadFile);
                    }
                    if crate::app::widgets::action_button(
                        ui,
                        ic::PLAY,
                        "Запустить",
                        crate::app::widgets::ActionTone::Primary,
                        130.0,
                    )
                    .on_hover_text("Прогнать сеть на выбранном бэкенде")
                    .clicked()
                    {
                        action = Some(NeuromlAction::Run);
                    }
                    if crate::app::widgets::action_button(
                        ui,
                        ic::X,
                        "Закрыть",
                        crate::app::widgets::ActionTone::Neutral,
                        130.0,
                    )
                    .clicked()
                    {
                        action = Some(NeuromlAction::Close);
                    }
                });
                // Окна «Редактор» (всегда) и «Морфология» (при наличии морфо-клетки).
                ui.add_space(tokens::SP_SM);
                ui.horizontal(|ui| {
                    if crate::app::widgets::action_button(
                        ui,
                        ic::TEXT_T,
                        "Редактор…",
                        crate::app::widgets::ActionTone::Neutral,
                        130.0,
                    )
                    .on_hover_text(
                        "Фокус на вкладку «Редактор»: IDE + подсветка + ошибки валидации",
                    )
                    .clicked()
                    {
                        action = Some(NeuromlAction::OpenEditor);
                    }
                    if self.cable.is_some()
                        && crate::app::widgets::action_button(
                            ui,
                            ic::TREE_STRUCTURE,
                            "Морфология…",
                            crate::app::widgets::ActionTone::Neutral,
                            130.0,
                        )
                        .on_hover_text(
                            "Открыть дерево компартментов с V(t) по компартментам",
                        )
                        .clicked()
                    {
                        action = Some(NeuromlAction::OpenCable);
                    }
                });
                ui.add_space(tokens::SP_SM);

                // Статус + результат.
                if !status.is_empty() {
                    ui.label(&status);
                }
                if let Some(r) = &result {
                    ui.label(
                        egui::RichText::new(format!(
                            "Шагов: {} · суммарно спайков: {}",
                            r.steps, r.total_spikes
                        ))
                        .strong(),
                    );
                    for (i, &c) in r.spike_counts.iter().enumerate() {
                        ui.label(format!("  популяция #{i}: {c} спайков"));
                    }
                }
                action
            })
            .unwrap_or(None);

        // Результат нативного диалога (когда поток вернул путь).
        if let Some(rx) = &picker_rx {
            if let Ok(Some(p)) = rx.try_recv() {
                path = p.display().to_string();
            }
        }

        // Записываем обратно локальные поля.
        self.show = open;
        self.path = PathBuf::from(path.trim());
        self.backend = backend;

        if let Some(action) = action {
            if let Some(surface) = self.apply_modal_action(action) {
                services.focus_surface = Some(surface);
            }
        }
    }

    /// Выполнить действие, выбранное в модалке (типизированно — без u8-кодов).
    /// Возвращает [`Surface`], которую надо сфокусировать (оболочка применит).
    fn apply_modal_action(&mut self, action: NeuromlAction) -> Option<Surface> {
        match action {
            NeuromlAction::LoadFile => self.neuroml_load(),
            NeuromlAction::Run => self.neuroml_run(),
            NeuromlAction::LoadSample(kind) => match kind {
                SampleKind::ExcInh => self.neuroml_load_sample(),
                SampleKind::HhAxon => self.neuroml_load_hh_sample(),
                SampleKind::AdEx => self.neuroml_load_adex_sample(),
            },
            NeuromlAction::Close => self.show = false,
            NeuromlAction::OpenCable => return Some(Surface::Morphology),
            NeuromlAction::OpenEditor => {
                // Первое открытие редактора: подставить пример, чтобы было
                // с чего начать (а не пустой экран).
                if self.editor.text.trim().is_empty() {
                    self.editor.load_text(SAMPLE_NML.to_string());
                    self.editor.validate_if_dirty();
                }
                return Some(Surface::Editor);
            }
        }
        None
    }

    // ── действия ───────────────────────────────────────────────
    /// Загрузить `.nml` из `self.path`, собрать CPU-сеть + GPU-зеркало.
    /// Текст файла попадает и в редактор (открыть → редактировать).
    fn neuroml_load(&mut self) {
        let path = self.path.clone();
        // Текст для редактора — до применения (даже если модель битая,
        // пользователь видит её в редакторе с ошибками).
        if let Ok(text) = std::fs::read_to_string(&path) {
            self.editor.load_text(text);
        }
        self.apply_bundle(neuroml::load(&path));
    }

    /// Загрузить встроенный пример (путь не трогаем).
    fn neuroml_load_sample(&mut self) {
        self.editor.load_text(SAMPLE_NML.to_string());
        let bundle = neuroml::load_str(SAMPLE_NML).map_err(|e| e.to_string());
        if bundle.is_ok() {
            self.status = "✓ загружен встроенный пример (exc ↔ inh цикл)".into();
        }
        self.apply_bundle(bundle);
    }

    /// Загрузить курсовой HH-пресет (классический аксон под импульсом).
    fn neuroml_load_hh_sample(&mut self) {
        self.editor.load_text(HH_SAMPLE_NML.to_string());
        let bundle = neuroml::load_str(HH_SAMPLE_NML).map_err(|e| e.to_string());
        if bundle.is_ok() {
            self.status = "✓ загружен HH-аксон (Hodgkin–Huxley 1952)".into();
        }
        self.apply_bundle(bundle);
    }

    /// Загрузить канонический AdEx-пресет (Brette & Gerstner 2005, фиг. 4).
    fn neuroml_load_adex_sample(&mut self) {
        self.editor.load_text(ADEX_SAMPLE_NML.to_string());
        let bundle = neuroml::load_str(ADEX_SAMPLE_NML).map_err(|e| e.to_string());
        if bundle.is_ok() {
            self.status = "✓ загружен AdEx (Brette–Gerstner 2005, фиг. 4)".into();
        }
        self.apply_bundle(bundle);
    }

    /// Общий обработчик результата загрузки: раскладывает [`NeuromlBundle`]
    /// по полям состояния и сбрасывает прошлый результат. Публичный для
    /// редактора (cable.rs/editor.rs): «Применить» идёт тем же путём, что и
    /// загрузка из модалки.
    pub(crate) fn apply_bundle(&mut self, bundle: Result<neuroml::NeuromlBundle, String>) {
        match bundle {
            Ok(b) => {
                // Предупреждения о молчаливо пропущенных элементах (gap
                // junctions, graded synapses, …) — показываем в статусе.
                let warn_note = if b.loaded.warnings.is_empty() {
                    String::new()
                } else {
                    format!(" · ⚠ пропущено элементов: {}", b.loaded.warnings.len())
                };
                // GPU-заметка в статусе: есть только с feature `cuda`.
                #[cfg(feature = "cuda")]
                let gpu_note = match &b.gpu {
                    Some(_) => " · GPU готов".to_string(),
                    None => format!(
                        " · GPU недоступен: {}",
                        b.gpu_warning.as_deref().unwrap_or("нет драйвера CUDA")
                    ),
                };
                #[cfg(not(feature = "cuda"))]
                let gpu_note = " · CPU-only (сборка без cuda)".to_string();
                self.status = format!(
                    "✓ загружено: {} популяций, {} нейронов, {} синапсов{gpu_note}{warn_note}",
                    b.n_pops, b.n_neurons, b.n_syn
                );
                // Мост в GUI: если есть кабельная популяция — готовим виджет
                // «Морфология» (геометрия + band-layout) ДО перемещения сети.
                self.cable = b.loaded.cable.iter().enumerate().find_map(|(pid, ci)| {
                    ci.as_ref()
                        .map(|info| crate::app::neuroml::CableState::new(pid, info.clone()))
                });
                self.loaded = Some(b.loaded);
                #[cfg(feature = "cuda")]
                {
                    self.gpu = b.gpu;
                }
                self.gpu_warning = b.gpu_warning;
                self.pop_sizes = b.pop_sizes;
                self.n_neurons = b.n_neurons;
                self.n_syn = b.n_syn;
                self.result = None;
            }
            Err(e) => {
                self.status = format!("⚠ {e}");
                self.loaded = None;
                #[cfg(feature = "cuda")]
                {
                    self.gpu = None;
                }
                self.gpu_warning = None;
                self.result = None;
                self.cable = None;
            }
        }
    }

    /// Прогнать загруженную сеть на выбранном бэкенде.
    pub(crate) fn neuroml_run(&mut self) {
        if self.loaded.is_none() {
            self.status = "⚠ сначала загрузи .nml (или пример)".into();
            return;
        }
        let backend = self.backend;
        #[cfg(feature = "cuda")]
        let pop_sizes = self.pop_sizes.clone();
        let res: Result<neuroml::RunResult, String> = match backend {
            Backend::Cpu => {
                let ln = self.loaded.as_mut().expect("checked above");
                Ok(neuroml::run_cpu(ln))
            }
            // GPU-прогон существует только с feature `cuda`.
            #[cfg(feature = "cuda")]
            Backend::Gpu => match self.gpu.as_mut() {
                Some(gpu) => {
                    let ln = self.loaded.as_ref().expect("checked above");
                    neuroml::run_gpu(gpu, ln, &pop_sizes)
                }
                None => Err("GPU недоступен — нет драйвера/устройства CUDA".into()),
            },
            #[cfg(not(feature = "cuda"))]
            Backend::Gpu => Err("GPU недоступен: бинарник собран без feature cuda".into()),
        };
        match res {
            Ok(r) => {
                self.status = format!(
                    "✓ {}: {} шагов, {} спайков",
                    backend.label(),
                    r.steps,
                    r.total_spikes
                );
                self.result = Some(r);
            }
            Err(e) => {
                self.status = format!("⚠ {e}");
                self.result = None;
            }
        }
    }
}
