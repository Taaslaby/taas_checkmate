//! # Переиспользуемые виджеты (`app::widgets`)
//!
//! Примитивы интерфейса, которые не знают про `ConnApp` и работают только с
//! `ui` + переданными данными. Это и есть «виджетизация»: панели собираются
//! из маленьких кирпичиков, каждый можно тестировать и использовать в любом
//! месте (инспектор, плоттер, модалки) без дублирования.
//!
//! ```text
//! widgets/
//! ├── section_card      карточка с заголовком (иконка + текст) и отступами
//! ├── key_value_fields  сетка «ключ: значение» (LayoutJob, неразрывный текст)
//! ├── badge_grid        сетка бейджей 2×N на всю ширину карточки
//! ├── legend_swatch     строка легенды: цветной квадрат + имя + счётчик
//! └── full_width_action кнопка действия на всю доступную ширину
//! ```

use eframe::egui::{self, Color32};

use elegance::{Badge, BadgeTone, Card};

use crate::app::shell::theme::{self, DANGER, STROKE_SOFT, TEXT, TEXT_DIM};
use crate::app::tokens;

// ── карточка с заголовком ─────────────────────────────────────
/// Обёртка над `elegance::Card` с единым заголовком (иконка + текст) и
/// стандартным паддингом. Все карточки приложения используют её, чтобы
/// у карточек был одинаковый вид, а не по-разному настроенный `Card`.
pub(crate) fn section_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    body: impl FnOnce(&mut egui::Ui),
) {
    Card::new()
        .heading(tokens::heading(tokens::icon_label(
            icon,
            title,
            tokens::ICON_SM,
        )))
        .padding(tokens::PAD_CARD)
        .show(ui, body);
}

/// Компактная карточка (плоты): меньше паддинг, заголовок без SemiBold.
pub(crate) fn compact_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    sub: &str,
    body: impl FnOnce(&mut egui::Ui),
) {
    Card::new()
        .heading(tokens::icon_label(icon, title, tokens::ICON_SM).strong())
        .padding(tokens::PAD_CARD_SM)
        .show(ui, |ui| {
            if !sub.is_empty() {
                ui.label(egui::RichText::new(sub).weak().small());
                ui.add_space(tokens::SP_XS);
            }
            body(ui);
        });
}

// ── поля «ключ: значение» ─────────────────────────────────────
/// Список полей «ключ: значение» одним неразрывным текстом на строку.
/// Ключ — `strong_text_color` (самый яркий цвет темы), значение — обычный
/// текст: ключи не могут слиться с фоном ни на одной теме. У каждой строки
/// есть тултип-пояснение. Длинные значения обрезаются с `…` (truncate) —
/// полный текст в тултипе + клик копирует.
pub(crate) fn key_value_fields(ui: &mut egui::Ui, fields: &[(&str, String, &str)]) {
    let key_col = ui.visuals().strong_text_color();
    let val_col = ui.visuals().text_color();
    for (k, v, tip) in fields {
        // Контейнер с min-w-0 — позволяет truncate внутри flex.
        ui.horizontal(|ui| {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                &format!("{k}:"),
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Body.resolve(ui.style()),
                    color: key_col,
                    ..Default::default()
                },
            );
            // Значение — в truncate-контейнере: при переполнении покажет …
            // В egui Label автоматически делает break-words; ограничиваем шириной.
            let full = v.clone();
            // Обрезаем визуально до ~28 символов, полный — в тултипе.
            let display = if full.chars().count() > 28 {
                let truncated: String = full.chars().take(28).collect();
                format!("  {truncated}…")
            } else {
                format!("  {v}")
            };
            job.append(
                &display,
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Body.resolve(ui.style()),
                    color: val_col,
                    ..Default::default()
                },
            );
            let resp = ui.label(job).on_hover_text(format!("{tip}\n{full}"));
            if resp.clicked() {
                ui.ctx().copy_text(full);
            }
        });
    }
}

// ── сетка бейджей ─────────────────────────────────────────────
/// Бейджи ровной сеткой по N колонок: каждый растянут на равную долю
/// доступной ширины — одинаковые ширина и высота, без «разъезжания».
pub(crate) fn badge_grid(ui: &mut egui::Ui, cols: usize, badges: &[(&str, BadgeTone)]) {
    for chunk in badges.chunks(cols) {
        ui.columns(cols, |cols_ui| {
            for (ci, (label, tone)) in chunk.iter().enumerate() {
                cols_ui[ci].add_sized(
                    [cols_ui[ci].available_width(), 0.0],
                    Badge::new(*label, *tone),
                );
            }
        });
    }
}

// ── строка легенды ────────────────────────────────────────────
/// Цветной квадрат-свотч + имя + опциональный счётчик. Свотч рисуется
/// краской, а не символом «██» (его нет в шрифте).
pub(crate) fn legend_swatch(ui: &mut egui::Ui, name: &str, color: Color32, count: Option<u64>) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(tokens::SWATCH, tokens::SWATCH),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(tokens::RADIUS_SM as u8),
            color,
        );
        ui.label(egui::RichText::new(name).size(tokens::LEGEND_TEXT));
        if let Some(c) = count {
            ui.label(egui::RichText::new(fmt_count(c)).weak().small());
        }
    });
}

/// Форматирование счётчика: 1234 → «1.2К».
fn fmt_count(c: u64) -> String {
    if c >= 1000 {
        format!("{:.1}К", c as f64 / 1000.0)
    } else {
        c.to_string()
    }
}

// ── кнопки-действия: единый язык ─────────────────────────────
/// Тон кнопки-действия. Единый дизайн: **один акцент темы** для первичных,
/// нейтральные для остальных, приглушённый красный — только для опасных
/// (стоп). Никакой палитры elegance — все кнопки рисуются нашим паинтером
/// и совпадают с акцентом приложения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActionTone {
    /// Главное действие — заливка акцентом темы (запуск, открыть, найти).
    Primary,
    /// Обычное действие — тихий фон + рамка (закрыть, отмена).
    Neutral,
    /// Опасное действие — приглушённый красный (остановить поток).
    Danger,
}

/// Смешивание цвета с белым (t>0) или чёрным (t<0): hover/pressed-состояния.
fn shade(c: Color32, t: f32) -> Color32 {
    let f = |a: u8, b: u8| {
        (a as f32 + (b as f32 - a as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    let (tr, tg, tb) = if t >= 0.0 { (255, 255, 255) } else { (0, 0, 0) };
    Color32::from_rgba_unmultiplied(f(c.r(), tr), f(c.g(), tg), f(c.b(), tb), c.a())
}

/// Единая кнопка-действие: иконка + текст, тон по [`ActionTone`], фиксированная
/// высота, ширина задаётся. Рисуется нашим паинтером — один стиль везде
/// (инспектор, модалки, NeuroML), не зависит от палитры elegance.
///
/// Доступность: фокус-кольцо при `has_focus()`, активация Space/Enter через
/// `Sense::click()`.
pub(crate) fn action_button(
    ui: &mut egui::Ui,
    icon: &str,
    label: &str,
    tone: ActionTone,
    width: f32,
) -> egui::Response {
    // Обычная строка (painter::text не принимает RichText): размер задаёт
    // FontId ниже, иконка+два пробела+текст — как в tokens::icon_label.
    // Пустая иконка — чистый текст (кнопки-пресеты без глифа, не «  label»).
    let text = if icon.is_empty() {
        label.to_string()
    } else {
        format!("{icon}  {label}")
    };
    let height = tokens::CONTROL_H;
    let width = width.min(ui.available_width()).max(1.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let pressed = response.is_pointer_button_down_on();
    let hovered = response.hovered() && !pressed;
    // focus-visible кольцо — рисуем поверх заливки, под текст
    let focused = response.has_focus();
    let (fill, stroke, fg) = match tone {
        // Единственное заметное действие в карточке. Акцент резервируется
        // только для явного запуска/подтверждения, чтобы не превращать
        // научный интерфейс в набор конкурирующих CTA.
        ActionTone::Primary => {
            let base = theme::accent(ui);
            let fill = if pressed {
                shade(base, -0.20)
            } else if hovered {
                shade(base, 0.08)
            } else {
                base
            };
            (
                fill,
                egui::Stroke::new(1.0, shade(base, -0.16)),
                Color32::from_rgb(0x0C, 0x1B, 0x18),
            )
        }
        // Приглушённый красный — только для опасных (стоп). Единственное
        // цветное исключение, стандарт любого профессионального тула.
        ActionTone::Danger => {
            let base = DANGER;
            let fill = if pressed {
                shade(base, -0.18)
            } else if hovered {
                shade(base, 0.08)
            } else {
                base
            };
            (fill, egui::Stroke::NONE, Color32::WHITE)
        }
        ActionTone::Neutral => {
            let base = Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 10);
            let fill = if pressed {
                Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 22)
            } else if hovered {
                Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 16)
            } else {
                base
            };
            (fill, egui::Stroke::new(1.0, STROKE_SOFT), TEXT)
        }
    };
    painter.rect_filled(
        rect,
        egui::CornerRadius::same(tokens::RADIUS_MD as u8),
        fill,
    );
    if stroke.width > 0.0 {
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(tokens::RADIUS_MD as u8),
            stroke,
            egui::StrokeKind::Inside,
        );
    }
    if focused {
        painter.rect_stroke(
            rect.expand(1.5),
            egui::CornerRadius::same(tokens::RADIUS_MD as u8),
            egui::Stroke::new(2.0, theme::accent(ui)),
            egui::StrokeKind::Outside,
        );
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(tokens::TEXT_BODY),
        fg,
    );
    response
}

/// Кнопка-действие на всю доступную ширину карточки (как в Blender):
/// иконка + текст, тон по [`ActionTone`]. Возвращает `true`, если нажата.
pub(crate) fn full_width_action(
    ui: &mut egui::Ui,
    icon: &str,
    label: &str,
    tone: ActionTone,
) -> bool {
    action_button(ui, icon, label, tone, ui.available_width()).clicked()
}

// ── тулбарные кнопки (рейл / верхний бар / модалки) ──────────
/// Blender-style «тулбарная» кнопка: фиксированный размер, иконка/текст по
/// центру. Активное состояние — сигнальный акцент темы (фон + рамка),
/// hover — мягкая подсветка, нажатие — затемнение. Одна геометрия для всех
/// инструментов: ширина не больше доступной, высота — ровно заданная.
/// Используется в левом рейле (`rail.rs`), верхнем баре и модалках —
/// единый вид всех «инструментных» кнопок приложения.
///
/// Доступность: `Sense::click()` делает кнопку фокусируемой — Space/Enter
/// срабатывают через `response.clicked()`. Фокус рисуется явным кольцом
/// (`has_focus()`), т.к. кастомный паинтер обходит стандартный `Button`.
pub(crate) fn tool_button(
    ui: &mut egui::Ui,
    icon: &str,
    active: bool,
    size: f32,
    width: f32,
    height: f32,
) -> egui::Response {
    // Желаемая ширина не больше доступной (рейл может быть уже запрошенной).
    let width = width.min(ui.available_width()).max(1.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    let painter = ui.painter_at(rect);
    // Активный = сигнальный фон + рамка; hover = лёгкая подсветка;
    // нажатие = чуть темнее, чтобы был тактильный отклик.
    let pressed = response.is_pointer_button_down_on();
    // Активный = утопленный нейтральный фон + тонкая рамка акцента (не яркая
    // заливка — иначе «игрушечный» вид). hover/pressed — мягкая подсветка.
    let fill = if pressed {
        Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 20)
    } else if active || response.hovered() {
        Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 14)
    } else {
        Color32::TRANSPARENT
    };
    painter.rect_filled(
        rect,
        egui::CornerRadius::same(tokens::RADIUS_TB as u8),
        fill,
    );
    if active {
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(tokens::RADIUS_TB as u8),
            egui::Stroke::new(1.0, theme::accent(ui)),
            egui::StrokeKind::Inside,
        );
        // Узкая сигнальная метка делает активный инструмент различимым даже
        // периферийным зрением в длинной вертикальной панели.
        painter.rect_filled(
            egui::Rect::from_min_size(rect.left_top(), egui::vec2(2.0, rect.height())),
            egui::CornerRadius::same(tokens::RADIUS_XS as u8),
            theme::accent(ui),
        );
    }
    // Focus-visible кольцо — 2px акцента снаружи, не перекрывает контент.
    if response.has_focus() {
        painter.rect_stroke(
            rect.expand(1.5),
            egui::CornerRadius::same(tokens::RADIUS_TB as u8),
            egui::Stroke::new(2.0, theme::accent(ui)),
            egui::StrokeKind::Outside,
        );
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(size),
        if active { theme::accent(ui) } else { TEXT_DIM },
    );
    response
}

/// Полярный отступ вокруг глифа в [`icon_button`]: 15px иконка → кнопка 43px
/// (было 27px, поднято до touch-target ~44px по гайдлайну).
const ICON_BTN_PAD: f32 = 14.0;

/// Компактная квадратная иконка-кнопка для строк тулбаров и модалок: та же
/// сигнальная подсветка active/hover, что у [`tool_button`], но фиксированный
/// размер — встаёт в одну строку с текстом и чекбоксами (рейл, верхний бар,
/// библиотека плоттеров).
pub(crate) fn icon_button(
    ui: &mut egui::Ui,
    icon: &str,
    active: bool,
    size: f32,
) -> egui::Response {
    let side = size + ICON_BTN_PAD * 2.0;
    tool_button(ui, icon, active, size, side, side)
}
