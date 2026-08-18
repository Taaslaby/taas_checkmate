//! # Редактор NeuroML (`app::neuroml::editor`)
//!
//! IDE-виджет: редактирование NeuroML-документа с **подсветкой синтаксиса**
//! (XML-токены), **живой валидацией** на каждый кадр после изменения и
//! списком ошибок с позициями `строка:колонка` (структурированно — из
//! [`crate::sim::neuroml::NeuromlError`]). Строки с ошибками подсвечиваются
//! фоном в редакторе.
//!
//! Чистая подсветка (токенизатор + `LayoutJob`) живёт в [`super::highlight`],
//! здесь — состояние редактора, тулбар, окно и диспетчеризация действий.
//!
//! «Применить»/«Запустить» идут тем же путём, что и модалка
//! ([`super::ui::NeuromlMode::apply_bundle`]) — редактор и модалка делят
//! состояние загруженной сети, морфологии и результата.

use eframe::egui::{self, TextEdit};

use egui_phosphor::regular as ic;

use crate::app::neuroml::NeuromlMode;
use crate::app::shell::theme::{self, DANGER, WARNING};
use crate::app::tokens::{self, icon_label};
use crate::app::widgets::{action_button, ActionTone};
use crate::sim::neuroml;

use super::highlight::build_job;
use super::ui::{sample_buttons, SampleKind, ADEX_SAMPLE_NML, HH_SAMPLE_NML, SAMPLE_NML};

/// Состояние редактора: буфер текста + результат последней валидации.
#[derive(Default)]
pub(crate) struct EditorState {
    /// Редактируемый NeuroML-документ.
    pub text: String,
    /// Текст менялся с последней валидации — перевалидировать в этом кадре.
    pub dirty: bool,
    /// Ошибки валидации: (строка, колонка) + сообщение.
    pub errors: Vec<(Option<(u32, u32)>, String)>,
    /// Последняя валидация успешна (документ компилируется в сеть).
    pub valid: bool,
}

impl EditorState {
    /// Заменить текст (загрузка файла/примера) и пометить «грязным».
    pub(crate) fn load_text(&mut self, text: String) {
        self.text = text;
        self.dirty = true;
        self.errors.clear();
        self.valid = false;
    }

    /// Перевалидировать, если текст менялся. Дёшево для типичных .nml
    /// (единицы КБ) — вызывается один раз за кадр из редактора.
    pub(crate) fn validate_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        match neuroml::validate_str(&self.text) {
            Ok(b) => {
                self.valid = true;
                self.errors = b
                    .warnings
                    .iter()
                    .map(|w| (None, format!("⚠ {w}")))
                    .collect();
            }
            Err(e) => {
                // `pos` приходит структурированно (без разбора строки) — так
                // подсветка строки с ошибкой не зависит от формата сообщения.
                self.errors = vec![(e.pos, e.msg)];
                self.valid = false;
            }
        }
    }
}

/// Номера строк (1-based) с ошибками — для подсветки фона в редакторе.
fn error_lines(errors: &[(Option<(u32, u32)>, String)]) -> Vec<u32> {
    errors
        .iter()
        .filter_map(|(pos, _)| pos.map(|(r, _)| r))
        .collect()
}

// ── Действия тулбара ────────────────────────────────────────────

/// Действие тулбара редактора. Возвращается из closure (пока текст взят в
/// `TextEdit`, вызывать `&mut self` нельзя) и выполняется после — типизированно,
/// без магических `u8`-кодов, которые можно перепутать.
enum EditorAction {
    /// Скомпилировать текст в сеть (обновить модалку/морфологию).
    Apply,
    /// Применить + прогнать сеть на выбранном бэкенде.
    Run,
    /// Загрузить встроенный пример.
    LoadSample(SampleKind),
    /// Открыть модалку загрузки `.nml`.
    OpenModal,
}

// ── Геометрия окна редактора ───────────────────────────────────
const EDITOR_DEFAULT_SIZE: egui::Vec2 = egui::vec2(880.0, 620.0);
const EDITOR_ROWS: usize = 22;
const ERRORS_MAX_HEIGHT: f32 = 150.0;
const W_APPLY: f32 = 120.0;
const W_RUN: f32 = 120.0;
const W_OPEN: f32 = 100.0;

impl NeuromlMode {
    /// Окно «Редактор NeuroML» (floating): тулбар + подсвеченный редактор +
    /// список ошибок.
    pub(crate) fn editor_window(&mut self, ctx: &egui::Context) {
        if !self.show_editor {
            return;
        }
        let mut open = true;
        egui::Window::new(format!("{} Редактор NeuroML", ic::TEXT_T))
            .open(&mut open)
            .default_size(EDITOR_DEFAULT_SIZE)
            .resizable(true)
            .show(ctx, |ui| {
                tokens::spacing(ui);
                let action = editor_toolbar(ui);

                // ── редактор с подсветкой + живая валидация ──
                let err_lines = error_lines(&self.editor.errors);
                let mut layouter = move |ui: &egui::Ui,
                                         buf: &dyn egui::widgets::text_edit::TextBuffer,
                                         wrap: f32| {
                    let font = egui::TextStyle::Monospace.resolve(ui.style());
                    let job = build_job(buf.as_str(), &err_lines, wrap, font, theme::accent(ui));
                    ui.fonts_mut(|f| f.layout_job(job))
                };
                let resp = ui.add(
                    TextEdit::multiline(&mut self.editor.text)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(EDITOR_ROWS)
                        .layouter(&mut layouter),
                );
                if resp.changed() {
                    self.editor.dirty = true;
                }
                self.editor.validate_if_dirty();

                ui.separator();
                editor_footer(ui, &self.editor, &self.status);

                // Действие выполняем после closure (отпущен заимствованный
                // `self.editor.text`), чтобы не дёргать `&mut self` внутри.
                if let Some(action) = action {
                    self.apply_editor_action(action);
                }
            });
        if !open {
            self.show_editor = false;
        }
    }

    /// «Применить» (и, опционально, «Запустить») текст редактора: тот же путь,
    /// что и загрузка из модалки (apply_bundle), поэтому морфология/результат
    /// обновляются одинаково.
    fn editor_apply(&mut self, run: bool) {
        self.editor.validate_if_dirty();
        if !self.editor.valid {
            self.status = "⚠ модель не валидна — исправь ошибки в редакторе".into();
            return;
        }
        let bundle = neuroml::load_str(&self.editor.text).map_err(|e| e.to_string());
        self.apply_bundle(bundle);
        if run {
            self.neuroml_run();
        }
    }

    fn apply_editor_action(&mut self, action: EditorAction) {
        match action {
            EditorAction::Apply => self.editor_apply(false),
            EditorAction::Run => self.editor_apply(true),
            EditorAction::LoadSample(kind) => {
                let text = match kind {
                    SampleKind::ExcInh => SAMPLE_NML,
                    SampleKind::HhAxon => HH_SAMPLE_NML,
                    SampleKind::AdEx => ADEX_SAMPLE_NML,
                };
                self.editor.load_text(text.to_string());
                self.editor.validate_if_dirty();
            }
            EditorAction::OpenModal => self.show = true,
        }
    }
}

/// Тулбар окна редактора: действия + встроенные примеры. Чистая функция —
/// возвращает выбранное действие (его выполняет владелец `&mut self`).
fn editor_toolbar(ui: &mut egui::Ui) -> Option<EditorAction> {
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        if action_button(ui, ic::CHECK, "Применить", ActionTone::Neutral, W_APPLY)
            .on_hover_text("Скомпилировать текст в сеть (обновить модалку/морфологию)")
            .clicked()
        {
            action = Some(EditorAction::Apply);
        }
        if action_button(ui, ic::PLAY, "Запустить", ActionTone::Primary, W_RUN)
            .on_hover_text("Применить + прогнать сеть на выбранном бэкенде")
            .clicked()
        {
            action = Some(EditorAction::Run);
        }
        ui.separator();
        ui.label(egui::RichText::new("Пример:").weak());
        if let Some(kind) = sample_buttons(ui) {
            action = Some(EditorAction::LoadSample(kind));
        }
        ui.separator();
        if action_button(ui, ic::FOLDER_OPEN, "Файл…", ActionTone::Neutral, W_OPEN)
            .on_hover_text("Открыть модалку загрузки .nml")
            .clicked()
        {
            action = Some(EditorAction::OpenModal);
        }
    });
    ui.separator();
    action
}

/// Статус валидации + список ошибок (с подсветкой строк в редакторе).
fn editor_footer(ui: &mut egui::Ui, editor: &EditorState, status: &str) {
    ui.horizontal_wrapped(|ui| {
        if editor.valid {
            ui.colored_label(
                theme::accent(ui),
                icon_label(ic::CHECK_CIRCLE, "документ валиден", tokens::ICON_SM),
            );
        } else if !editor.errors.is_empty() {
            ui.colored_label(DANGER, format!("{} ошибок", editor.errors.len()));
        }
        if !status.is_empty() {
            ui.separator();
            ui.label(status);
        }
    });
    if !editor.errors.is_empty() {
        ui.add_space(tokens::SP_XS);
        egui::ScrollArea::vertical()
            .max_height(ERRORS_MAX_HEIGHT)
            .id_salt("neuroml_editor_errors")
            .show(ui, |ui| {
                for (pos, msg) in &editor.errors {
                    match pos {
                        Some((r, c)) => {
                            ui.colored_label(DANGER, format!("строка {r}, колонка {c}: {msg}"));
                        }
                        None => {
                            ui.colored_label(WARNING, msg);
                        }
                    }
                }
            });
    }
}

// ── Тесты ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_lines_are_one_based() {
        let errs = vec![(Some((3, 1)), "x".into()), (None, "y".into())];
        assert_eq!(error_lines(&errs), vec![3]);
    }

    #[test]
    fn validate_marks_broken_document() {
        let mut ed = EditorState::default();
        ed.load_text("<neuroml>\n  <izhikevichCell a=\"0.02/>\n</neuroml>".to_string());
        ed.validate_if_dirty();
        assert!(!ed.valid);
        assert!(!ed.errors.is_empty());
        // Позиция XML-ошибки дошла до редактора **структурированно**.
        assert!(ed.errors.iter().any(|(p, _)| p.is_some()));
    }

    #[test]
    fn validate_keeps_structured_position() {
        let mut ed = EditorState::default();
        ed.load_text("<neuroml>\n  <izhikevichCell a=\"0.02/>\n</neuroml>".to_string());
        ed.validate_if_dirty();
        let (pos, msg) = &ed.errors[0];
        assert_eq!(*pos, Some((3, 1)), "XML-ошибка должна нести позицию 3:1");
        assert!(
            !msg.contains("neuroml parse error:"),
            "сообщение без префикса, got: {msg}"
        );
    }

    #[test]
    fn validate_accepts_good_document() {
        let mut ed = EditorState::default();
        ed.load_text(SAMPLE_NML.to_string());
        ed.validate_if_dirty();
        assert!(ed.valid, "пример должен валидироваться: {:?}", ed.errors);
        assert!(ed.errors.is_empty());
    }
}
