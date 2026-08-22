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

use elegance::{Badge, BadgeTone};

use crate::app::shell::theme::{self, DANGER, STROKE_SOFT, TEXT, TEXT_DIM};
use crate::app::tokens;

// ── карточка с заголовком ─────────────────────────────────────
/// Обёртка над `elegance::Card` с единым заголовком (иконка + текст) и
/// стандартным паддингом. Все карточки приложения используют её, чтобы
/// у карточек был одинаковый вид, а не по-разному настроенный `Card`.
/// Красивее: стекло + мягкая тень + градиент хедера (Aqua).
pub(crate) fn section_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    body: impl FnOnce(&mut egui::Ui),
) {
    // Стекло: полупрозрачный фон + размытая тень + тонкий бордер
    let frame = egui::Frame::new()
        .fill(crate::app::shell::theme::card_bg(ui.ctx()).gamma_multiply(0.9))
        .stroke(egui::Stroke::new(1.0, crate::app::shell::theme::stroke_soft(ui.ctx())))
        .corner_radius(10)
        .inner_margin(tokens::PAD_CARD)
        .shadow(egui::Shadow {
            offset: [0, 6],
            blur: 16,
            spread: 0,
            color: Color32::from_black_alpha(50),
        });
    frame.show(ui, |ui| {
        // Хедер — иконка в кружке + заголовок
        ui.horizontal(|ui| {
            // Иконка в кружке с градиентом
            let (r, _) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::hover());
            ui.painter().circle_filled(r.center(), 11.0, theme::accent(ui).gamma_multiply(0.18));
            ui.painter().circle_stroke(r.center(), 11.0, egui::Stroke::new(1.0, theme::accent(ui).gamma_multiply(0.5)));
            ui.painter().text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                icon,
                egui::FontId::proportional(tokens::ICON_SM),
                theme::accent(ui),
            );
            ui.label(tokens::heading(tokens::icon_label("", title, tokens::ICON_SM)));
        });
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);
        body(ui);
    });
}

/// Компактная карточка (плоты): меньше паддинг, заголовок без SemiBold.
pub(crate) fn compact_card(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    sub: &str,
    body: impl FnOnce(&mut egui::Ui),
) {
    // Компактное стекло — как section_card, но меньше паддинг и заголовок без SemiBold
    let frame = egui::Frame::new()
        .fill(crate::app::shell::theme::card_bg(ui.ctx()).gamma_multiply(0.8))
        .stroke(egui::Stroke::new(1.0, crate::app::shell::theme::stroke_soft(ui.ctx())))
        .corner_radius(8)
        .inner_margin(tokens::PAD_CARD_SM)
        .shadow(egui::Shadow {
            offset: [0, 4],
            blur: 12,
            spread: 0,
            color: Color32::from_black_alpha(40),
        });
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
            ui.painter().circle_filled(r.center(), 9.0, theme::accent(ui).gamma_multiply(0.15));
            ui.painter().text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                icon,
                egui::FontId::proportional(tokens::ICON_XS),
                theme::accent(ui),
            );
            ui.label(egui::RichText::new(title).strong().size(tokens::TEXT_SMALL + 0.5));
            if !sub.is_empty() {
                ui.label(egui::RichText::new(format!("· {sub}")).weak().small());
            }
        });
        ui.add_space(tokens::SP_XS);
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
/// Цветной квадрат-свотч + имя + опциональный счётчик. Свотч — скевоморф:
/// градиент + блик + тень, как старый Apple.
pub(crate) fn legend_swatch(ui: &mut egui::Ui, name: &str, color: Color32, count: Option<u64>) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(tokens::SWATCH, tokens::SWATCH),
            egui::Sense::hover(),
        );
        // Тень
        ui.painter().rect_filled(
            rect.translate(egui::vec2(0.0, 1.0)),
            egui::CornerRadius::same(tokens::RADIUS_SM as u8),
            Color32::from_black_alpha(30),
        );
        // Градиент — светлее сверху
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(tokens::RADIUS_SM as u8),
            color,
        );
        ui.painter().rect_filled(
            egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.min.y + rect.height() * 0.45)),
            egui::CornerRadius {
                nw: tokens::RADIUS_SM as u8,
                ne: tokens::RADIUS_SM as u8,
                sw: 0,
                se: 0,
            },
            Color32::from_white_alpha(22),
        );
        ui.painter().rect_stroke(
            rect,
            egui::CornerRadius::same(tokens::RADIUS_SM as u8),
            egui::Stroke::new(1.0, Color32::from_white_alpha(18)),
            egui::StrokeKind::Inside,
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
    // Inner highlight — Aqua gloss (старый Apple), 1px сверху
    painter.rect_filled(
        egui::Rect::from_min_max(
            rect.min,
            egui::pos2(rect.max.x, rect.min.y + 1.2),
        ),
        egui::CornerRadius {
            nw: tokens::RADIUS_MD as u8,
            ne: tokens::RADIUS_MD as u8,
            sw: 0,
            se: 0,
        },
        Color32::from_white_alpha(18),
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
