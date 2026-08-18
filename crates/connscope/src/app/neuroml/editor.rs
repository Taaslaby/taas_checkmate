//! # Редактор NeuroML (`app::neuroml::editor`)
//!
//! IDE-виджет: редактирование NeuroML-документа с **подсветкой синтаксиса**
//! (XML-токены), **номерами строк** (gutter), **подсветкой строки курсора**,
//! **живой валидацией** на каждый кадр после изменения и списком ошибок с
//! позициями `строка:колонка` (клик по ошибке — курсор прыгает на место).
//! Внизу — статус-бар (имя файла, индикатор несохранённых изменений, позиция
//! курсора), в тулбаре — зум текста.
//!
//! Чистая подсветка (токенизатор + `LayoutJob`) живёт в [`super::highlight`],
//! здесь — состояние редактора, тулбар, вкладка дока и диспетчеризация действий.
//!
//! «Применить»/«Запустить» идут тем же путём, что и модалка
//! ([`super::ui::NeuromlMode::apply_bundle`]) — редактор и модалка делят
//! состояние загруженной сети, морфологии и результата.

use std::path::Path;

use eframe::egui::text::{CCursor, CCursorRange, LayoutJob, TextFormat};
use eframe::egui::{self, FontId, TextEdit};

use egui_phosphor::regular as ic;

use crate::app::neuroml::NeuromlMode;
use crate::app::shell::theme::{self, DANGER, WARNING};
use crate::app::tokens::{self, icon_label};
use crate::app::widgets::{action_button, icon_button, ActionTone};
use crate::sim::neuroml;

use super::highlight::build_job;
use super::ui::{sample_buttons, SampleKind, ADEX_SAMPLE_NML, HH_SAMPLE_NML, SAMPLE_NML};

/// Состояние редактора: буфер текста + результат последней валидации.
pub(crate) struct EditorState {
    /// Редактируемый NeuroML-документ.
    pub text: String,
    /// Текст менялся с последней валидации — перевалидировать в этом кадре.
    pub dirty: bool,
    /// Текст менялся с последнего сохранения (индикатор «●» в статус-баре).
    pub modified: bool,
    /// Ошибки валидации: (строка, колонка) + сообщение.
    pub errors: Vec<(Option<(u32, u32)>, String)>,
    /// Последняя валидация успешна (документ компилируется в сеть).
    pub valid: bool,
    /// Фокус в текстовом поле (для Ctrl+S = «Сохранить» в оболочке).
    pub focused: bool,
    /// Размер шрифта редактора (зум, 9–24).
    pub font_size: f32,
    /// Позиция курсора (строка, колонка), 1-based — для статус-бара и
    /// подсветки текущей строки (используется со следующего кадра).
    pub cursor_pos: (u32, u32),
    /// Поколение документа: растёт при каждой загрузке, чтобы сбросить
    /// undo-историю egui (новый `id_salt` → новый `TextEditState`).
    generation: u64,
    /// Отложенная команда Undo/Redo из меню «Правка» — выполняется после
    /// рендера TextEdit в этом же кадре (нужен живой `TextEditState`).
    pub pending_edit: Option<EditorEdit>,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            text: String::new(),
            dirty: false,
            modified: false,
            errors: Vec::new(),
            valid: false,
            focused: false,
            font_size: 12.5,
            cursor_pos: (1, 1),
            generation: 0,
            pending_edit: None,
        }
    }
}

/// Команда меню «Правка» (Undo/Redo), применяемая к живому `TextEditState`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditorEdit {
    Undo,
    Redo,
}

impl EditorState {
    /// Заменить текст (загрузка файла/примера) и пометить «грязным».
    /// Сбрасывает undo-историю egui (новое поколение документа).
    pub(crate) fn load_text(&mut self, text: String) {
        self.text = text;
        self.dirty = true;
        self.modified = true;
        self.errors.clear();
        self.valid = false;
        self.cursor_pos = (1, 1);
        self.generation = self.generation.wrapping_add(1);
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

// ── позиции «символ ⇄ строка:колонка» ─────────────────────────

/// Число строк в тексте (1-минимум; пустой текст = 1 строка).
fn count_lines(text: &str) -> usize {
    text.bytes().filter(|&b| b == b'\n').count() + 1
}

/// Строка:колонка (1-based) символа с char-индексом `char_idx`.
fn line_col_of_char(text: &str, char_idx: usize) -> (u32, u32) {
    let mut line = 1u32;
    let mut col = 1u32;
    for (i, ch) in text.chars().enumerate() {
        if i == char_idx {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Char-индекс начала (строка, колонка) — для прыжка курсора по ошибке.
/// Колонка за концом строки клампится к её концу; строка за пределами
/// текста — к концу файла.
fn char_of_line_col(text: &str, line: u32, col: u32) -> usize {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    // Char-индекс начала строки `line` (1-based).
    let mut line_start = 0usize;
    let mut cur_line = 1u32;
    for (ci, (_, ch)) in chars.iter().enumerate() {
        if cur_line == line {
            break;
        }
        if *ch == '\n' {
            cur_line += 1;
            line_start = ci + 1;
        }
    }
    // Строка за пределами текста — курсор в конец файла.
    if cur_line < line {
        return text.len();
    }
    // Шагаем по символам строки до колонки `col` (или её конца).
    let mut cur = line_start;
    let mut cur_col = 1u32;
    while cur_col < col && cur < chars.len() && chars[cur].1 != '\n' {
        cur += 1;
        cur_col += 1;
    }
    chars.get(cur).map_or(text.len(), |(byte, _)| *byte)
}

// ── Действия тулбара ────────────────────────────────────────────

/// Действие тулбара редактора. Возвращается из closure (пока текст взят в
/// `TextEdit`, вызывать `&mut self` нельзя) и выполняется после — типизированно,
/// без магических `u8`-кодов, которые можно перепутать.
enum EditorAction {
    /// Записать текст в файл на диске.
    Save,
    /// Скомпилировать текст в сеть (обновить модалку/морфологию).
    Apply,
    /// Применить + прогнать сеть на выбранном бэкенде.
    Run,
    /// Загрузить встроенный пример.
    LoadSample(SampleKind),
    /// Открыть модалку загрузки `.nml`.
    OpenModal,
    /// Зум текста: ±`delta` pt (9–24).
    Zoom(f32),
}

// ── Геометрия вкладки редактора ─────────────────────────────────
const ERRORS_MAX_HEIGHT: f32 = 150.0;
/// Высота статус-бара (имя файла · валидация · позиция курсора).
const STATUS_H: f32 = 26.0;
const W_SAVE: f32 = 110.0;
const W_APPLY: f32 = 120.0;
const W_RUN: f32 = 120.0;
const W_OPEN: f32 = 100.0;

impl NeuromlMode {
    /// Вкладка «Редактор NeuroML» в дереве доков: тулбар, gutter + редактор,
    /// ошибки (кликабельны) и статус-бар.
    pub(crate) fn editor_dock_view(&mut self, ui: &mut egui::Ui) {
        tokens::spacing(ui);
        let action = editor_toolbar(ui, self.editor.font_size);

        // ── редактор: gutter + подсвеченный текст в общем скролле ──
        // Номера строк и текст лежат в одном ScrollArea и одном шрифте,
        // поэтому скроллятся вместе и выровнены по рядам всегда.
        let font_size = self.editor.font_size;
        let font = FontId::monospace(font_size);
        let row_height = ui.fonts_mut(|f| f.row_height(&font));
        let n_lines = count_lines(&self.editor.text);
        let err_lines = error_lines(&self.editor.errors);
        let cursor_line = self.editor.cursor_pos.0;
        let accent = theme::accent(ui);

        // Высота области редактора: остаток вкладки минус статус-бар и
        // (если есть) список ошибок — вкладка не «распухает» и не режет.
        let errors_h = if err_lines.is_empty() { 0.0 } else { ERRORS_MAX_HEIGHT + 8.0 };
        let avail_h = (ui.available_height() - STATUS_H - errors_h).max(80.0);
        let rows_for_h = (avail_h / row_height).ceil() as usize;
        let desired_rows = n_lines.max(rows_for_h).max(1);

        // Подсветка строки курсора живёт в LayoutJob (под глифами).
        // Замыкание владеет копиями; оригиналы нужны gutter'у ниже.
        let mut layouter = {
            let err_lines = err_lines.clone();
            let font = font.clone();
            move |ui: &egui::Ui,
                  buf: &dyn egui::widgets::text_edit::TextBuffer,
                  _wrap: f32| {
                let job = build_job(
                    buf.as_str(),
                    &err_lines,
                    cursor_line,
                    f32::INFINITY,
                    font.clone(),
                    accent,
                );
                ui.fonts_mut(|f| f.layout_job(job))
            }
        };

        // Gutter-галея: номера строк, правая выключа с пробелом-полем.
        let digits = n_lines.max(1).to_string().len().max(2);
        let gutter_galley = {
            let mut job = LayoutJob {
                break_on_newline: true,
                wrap: egui::epaint::text::TextWrapping {
                    max_width: f32::INFINITY,
                    ..Default::default()
                },
                ..Default::default()
            };
            for line in 1..=n_lines {
                let is_err = err_lines.binary_search(&(line as u32)).is_ok();
                let color = if is_err { DANGER } else { GUTTER_COLOR };
                job.append(
                    &format!("{line:>w$}", w = digits + 1),
                    0.0,
                    TextFormat {
                        font_id: font.clone(),
                        color,
                        ..Default::default()
                    },
                );
                if line < n_lines {
                    job.append(
                        "\n",
                        0.0,
                        TextFormat {
                            font_id: font.clone(),
                            color: egui::Color32::TRANSPARENT,
                            ..Default::default()
                        },
                    );
                }
            }
            ui.fonts_mut(|f| f.layout_job(job))
        };

        let resp = egui::ScrollArea::vertical()
            .id_salt("nml_editor_scroll")
            .max_height(avail_h)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    // Тот же inner-margin (4×2), что у TextEdit — иначе
                    // первая строка gutter сдвинута относительно текста.
                    egui::Frame::NONE
                        .inner_margin(egui::Margin::symmetric(4, 2))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(gutter_galley.clone()));
                        });
                    ui.separator();
                    TextEdit::multiline(&mut self.editor.text)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(desired_rows)
                        // Поколение документа — ключ состояния egui:
                        // новая загрузка = новый undo-стек.
                        .id_salt(("neuroml_editor_text", self.editor.generation))
                        .layouter(&mut layouter)
                        .show(ui)
                })
                .inner
            })
            .inner;

        if resp.response.changed() {
            self.editor.dirty = true;
            self.editor.modified = true;
        }
        // Фокус нужен оболочке: Ctrl+S сохраняет только из редактора.
        self.editor.focused = resp.response.has_focus();
        self.editor.cursor_pos = resp
            .cursor_range
            .map(|r| line_col_of_char(&self.editor.text, r.primary.index.0))
            .unwrap_or(self.editor.cursor_pos);
        self.editor.validate_if_dirty();

        // Меню «Правка → Отменить/Вернуть» — тот же undo-стек egui,
        // что и встроенные Ctrl+Z/Y в самом редакторе.
        if let Some(edit) = self.editor.pending_edit.take() {
            self.apply_editor_edit(ui.ctx(), &resp, edit);
        }

        ui.separator();
        // Клик по ошибке — прыжок курсора на строку/колонку.
        let jump = editor_errors(ui, &self.editor, &self.status);
        editor_status_bar(ui, &self.editor, &self.path);
        if let Some((line, col)) = jump {
            jump_to(ui.ctx(), &resp, &self.editor.text, line, col);
        }

        // Действие выполняем после того, как `TextEdit` отпустил
        // `&mut self.editor.text`.
        if let Some(action) = action {
            self.apply_editor_action(action);
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

    /// Записать текущий текст в `self.path` (путь заполняется при загрузке
    /// файла; для встроенных примеров без файла — подсказка).
    pub(crate) fn editor_save(&mut self) {
        if self.path.as_os_str().is_empty() {
            self.status = "⚠ некуда сохранять — открой .nml через «Файл…» или модалку".into();
            return;
        }
        match std::fs::write(&self.path, &self.editor.text) {
            Ok(()) => {
                self.editor.modified = false;
                self.status = format!("✓ сохранено: {}", self.path.display());
            }
            Err(e) => self.status = format!("⚠ не удалось сохранить: {e}"),
        }
    }

    /// Применить команду Undo/Redo из меню «Правка» к живому `TextEditState`.
    /// Берём тот же курсор, что TextEdit скормил undoer'у в этом кадре, чтобы
    /// меню и клавиатура работали на одном механизме, а не на двух копиях.
    fn apply_editor_edit(
        &mut self,
        ctx: &egui::Context,
        resp: &egui::text_edit::TextEditOutput,
        edit: EditorEdit,
    ) {
        let mut state = resp.state.clone();
        let mut undoer = state.undoer();
        let cursor = state.cursor.range(&resp.galley).unwrap_or_default();
        let current = (cursor, self.editor.text.clone());
        let next = match edit {
            EditorEdit::Undo => undoer.undo(&current).cloned(),
            EditorEdit::Redo => undoer.redo(&current).cloned(),
        };
        if let Some((next_cursor, next_text)) = next {
            self.editor.text = next_text;
            self.editor.dirty = true;
            self.editor.modified = true;
            state.cursor.set_char_range(Some(next_cursor));
            state.set_undoer(undoer);
            state.store(ctx, resp.response.id);
            ctx.memory_mut(|m| m.request_focus(resp.response.id));
        }
    }

    fn apply_editor_action(&mut self, action: EditorAction) {
        match action {
            EditorAction::Save => self.editor_save(),
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
            EditorAction::Zoom(delta) => {
                self.editor.font_size = (self.editor.font_size + delta).clamp(9.0, 24.0);
            }
        }
    }
}

/// Цвет номеров строк (тихий, не конкурирует с кодом).
const GUTTER_COLOR: egui::Color32 = egui::Color32::from_gray(0x66);

/// Тулбар окна редактора: действия + примеры + зум. Чистая функция —
/// возвращает выбранное действие (его выполняет владелец `&mut self`).
fn editor_toolbar(ui: &mut egui::Ui, font_size: f32) -> Option<EditorAction> {
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        if action_button(ui, ic::FLOPPY_DISK, "Сохранить", ActionTone::Neutral, W_SAVE)
            .on_hover_text("Записать текст в файл на диске (Ctrl+S)")
            .clicked()
        {
            action = Some(EditorAction::Save);
        }
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
        ui.separator();
        if icon_button(ui, ic::MINUS, false, tokens::ICON_SM)
            .on_hover_text("Уменьшить текст")
            .clicked()
        {
            action = Some(EditorAction::Zoom(-1.0));
        }
        ui.label(
            egui::RichText::new(format!("{font_size:.0}"))
                .weak()
                .monospace()
                .size(tokens::TEXT_SMALL),
        )
        .on_hover_text("Размер текста");
        if icon_button(ui, ic::PLUS, false, tokens::ICON_SM)
            .on_hover_text("Увеличить текст")
            .clicked()
        {
            action = Some(EditorAction::Zoom(1.0));
        }
    });
    ui.separator();
    action
}

/// Статус валидации + список ошибок. Клик по ошибке с позицией возвращает
/// (строка, колонка) — владелец прыгает курсором в редактор.
fn editor_errors(ui: &mut egui::Ui, editor: &EditorState, status: &str) -> Option<(u32, u32)> {
    let mut jump = None;
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
                            if ui
                                .add(
                                    egui::Label::new(
                                        egui::RichText::new(format!(
                                            "строка {r}, колонка {c}: {msg}"
                                        ))
                                        .color(DANGER),
                                    )
                                    .sense(egui::Sense::click()),
                                )
                                .on_hover_text("Клик — перейти к строке")
                                .clicked()
                            {
                                jump = Some((*r, *c));
                            }
                        }
                        None => {
                            ui.colored_label(WARNING, msg);
                        }
                    }
                }
            });
    }
    jump
}

/// Статус-бар: имя файла (+ «●» не сохранено), валидация, справа — позиция
/// курсора и язык.
fn editor_status_bar(ui: &mut egui::Ui, editor: &EditorState, path: &Path) {
    ui.separator();
    ui.horizontal(|ui| {
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "безымянный.nml".to_string());
        if editor.modified {
            ui.label(egui::RichText::new(format!("● {name}")).color(theme::accent(ui)))
                .on_hover_text("Не сохранено — Ctrl+S");
        } else {
            ui.label(egui::RichText::new(name).weak());
        }
        ui.separator();
        if editor.valid {
            ui.colored_label(theme::accent(ui), "валиден");
        } else if !editor.errors.is_empty() {
            ui.colored_label(DANGER, format!("{} ошибок", editor.errors.len()));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.monospace(egui::RichText::new(format!(
                "Стр {}, Кол {}",
                editor.cursor_pos.0, editor.cursor_pos.1
            )));
            ui.separator();
            ui.label(egui::RichText::new("NeuroML").weak());
        });
    });
}

/// Прыжок курсора в редактор на (строка, колонка) — по клику на ошибку.
fn jump_to(
    ctx: &egui::Context,
    resp: &egui::text_edit::TextEditOutput,
    text: &str,
    line: u32,
    col: u32,
) {
    let mut state = resp.state.clone();
    let target = char_of_line_col(text, line, col);
    state.cursor.set_char_range(Some(CCursorRange::one(CCursor::new(target))));
    state.store(ctx, resp.response.id);
    ctx.memory_mut(|m| m.request_focus(resp.response.id));
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

    /// Загрузка нового документа помечает изменения несохранёнными.
    #[test]
    fn load_text_marks_modified() {
        let mut ed = EditorState::default();
        assert!(!ed.modified);
        ed.load_text("<neuroml/>".to_string());
        assert!(ed.modified);
    }

    /// Маппинг «символ → строка:колонка» и обратно согласован.
    #[test]
    fn line_col_round_trip() {
        let doc = "ab\ncd\nef";
        // a=1:1, b=1:2, \n=1:3, c=2:1 …
        assert_eq!(line_col_of_char(doc, 0), (1, 1));
        assert_eq!(line_col_of_char(doc, 1), (1, 2));
        assert_eq!(line_col_of_char(doc, 3), (2, 1));
        assert_eq!(line_col_of_char(doc, 7), (3, 2));
        for i in 0..doc.len() {
            let (l, c) = line_col_of_char(doc, i);
            assert_eq!(char_of_line_col(doc, l, c), i, "round-trip {i}");
        }
    }

    #[test]
    fn char_of_line_col_clamps_to_end() {
        let doc = "a\nb";
        assert_eq!(char_of_line_col(doc, 99, 1), doc.len());
        assert_eq!(char_of_line_col(doc, 1, 99), 1);
    }

    #[test]
    fn count_lines_basics() {
        assert_eq!(count_lines(""), 1);
        assert_eq!(count_lines("a"), 1);
        assert_eq!(count_lines("a\nb"), 2);
        assert_eq!(count_lines("a\nb\n"), 3);
    }
}
