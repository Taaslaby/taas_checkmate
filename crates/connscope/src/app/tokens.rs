//! # Design tokens (`app::tokens`)
//!
//! A single scale of spacing, padding and icon sizes — instead of scattered
//! magic numbers across the panels. Base grid is **4 px**:
//!
//! ```text
//! XS=4  SM=6  MD=8  LG=12  XL=16  XXL=24
//! ```
//!
//! Usage rules:
//! * spacing **within** a row (between chips, buttons) — `SP_MD` (item_spacing);
//! * spacing **between** sections/cards — `SP_MD`..`SP_LG`;
//! * card padding — `PAD_CARD` / `PAD_CARD_SM`;
//! * Phosphor icons — via [`icon`]/[`icon_label`] with a size from `ICON_*`,
//!   never by inlining a glyph into the text (glyphs carry inner padding).

use eframe::egui::{self, Margin, Vec2};

// ── spacing scale (4 px base unit) ───────────────────────────
/// 4 px — minimal gap (between an icon and text in a row).
pub(crate) const SP_XS: f32 = 4.0;
/// 6 px — tight vertical gap (between a label and a plot).
pub(crate) const SP_SM: f32 = 6.0;
/// 8 px — base gap: panel item_spacing, gap between cards.
pub(crate) const SP_MD: f32 = 8.0;
/// 12 px — card padding, gap between large blocks.
pub(crate) const SP_LG: f32 = 12.0;
// ── shell geometry ─────────────────────────────────────────────
pub(crate) const STATUS_BAR_H: f32 = 24.0;

// ── paddings ─────────────────────────────────────────────────
/// Padding of a standard card (inspector, modals).
pub(crate) const PAD_CARD: Margin = Margin::same(SP_LG as i8);
/// Padding of a compact card (plotter — plots take the full width).
pub(crate) const PAD_CARD_SM: Margin = Margin::same(SP_MD as i8);
/// Default panel `item_spacing`: 8×6.
pub(crate) const ITEM_SPACING: Vec2 = Vec2::new(SP_MD, SP_SM);
/// Gap between inspector cards.
pub(crate) const GAP_CARDS: f32 = SP_MD;

// ── icon sizes (Phosphor) ────────────────────────────────────
/// 13 px — icon inline with small text (tabs, chips).
pub(crate) const ICON_XS: f32 = 13.0;
/// 15 px — icon in buttons and headings.
pub(crate) const ICON_SM: f32 = 15.0;
/// 20 px — icon in a modal heading.
pub(crate) const ICON_LG: f32 = 20.0;

// ── corner radii ─────────────────────────────────────────────
/// 1 px — узкая сигнальная метка активного инструмента (tool_button).
pub(crate) const RADIUS_XS: f32 = 1.0;
/// 3 px — цветовой свотч легенды.
pub(crate) const RADIUS_SM: f32 = 3.0;
/// 4 px — квадратные тулбарные кнопки (строже карточных).
pub(crate) const RADIUS_TB: f32 = 4.0;
/// 6 px — карточки, кнопки-действия, стандартные виджеты.
pub(crate) const RADIUS_MD: f32 = 6.0;

// ── типографическая шкала (px) ───────────────────────────────
/// Заголовок (SemiBold через family `heading`).
pub(crate) const TEXT_HEADING: f32 = 16.0;
/// Тело и кнопки.
pub(crate) const TEXT_BODY: f32 = 13.5;
/// Подписи, единицы измерения.
pub(crate) const TEXT_SMALL: f32 = 12.0;
/// Данные: консоль, индексы, числа.
pub(crate) const TEXT_MONO: f32 = 12.5;

/// Числа в столбцах/сравнениях — tabular-nums (равная ширина цифр).
/// В egui нет CSS font-variant-numeric, поэтому числа рендерим моноширинным
/// начертанием; этот хелпер возвращает RichText с JetBrains Mono.
#[allow(dead_code)]
pub(crate) fn tabular(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text)
        .family(egui::FontFamily::Monospace)
        .size(TEXT_MONO)
}

// ── controls ─────────────────────────────────────────────────
/// Высота паинтерной кнопки-действия (`action_button` не берёт высоту из стиля).
pub(crate) const CONTROL_H: f32 = 30.0;

// ── legend (color cheat-sheet in the inspector) ──────────────
/// Colored swatch square (px).
pub(crate) const SWATCH: f32 = 18.0;
/// Label next to a swatch — small text (px), NOT the icon size:
/// text and icons have different optical heights.
pub(crate) const LEGEND_TEXT: f32 = 13.0;
/// Legend columns — split the card width evenly.
pub(crate) const LEGEND_COLS: usize = 2;

// ── toolbar control sizing (top card row) ───────────────────
/// Standard width of a toolbar combo box (class / color mode) — one grid rule
/// so combos don't each have their own magic width.
pub(crate) const TOOL_COMBO_W: f32 = 110.0;
/// Standard width of a toolbar slider. Sliders sit on ONE line with the other
/// controls — so they get no stacked label (a label would make the row crooked).
pub(crate) const TOOL_SLIDER_W: f32 = 120.0;
/// Height of the toolbar slider — 24px для touch (guideline 44px, но слайдер
/// узкий трек; 24px даёт нормальный хит-таргет без раздувания тулбара).
pub(crate) const TOOL_SLIDER_H: f32 = 24.0;
/// Width of the toolbar search field.
pub(crate) const TOOL_SEARCH_W: f32 = 200.0;
/// Height of the single-line top toolbar — 44px для touch-target 43px.
pub(crate) const TOP_BAR_H: f32 = 44.0;
/// Horizontal gap between top-toolbar controls.
pub(crate) const TOP_GAP_X: f32 = SP_SM;
/// Vertical gap inside a single toolbar row.
pub(crate) const TOP_GAP_Y: f32 = 0.0;

// ── helpers ──────────────────────────────────────────────────
/// Apply the default `item_spacing` (8×6) to a panel/window — first thing
/// inside every `Panel::show` / `Window::show`.
pub(crate) fn spacing(ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing = ITEM_SPACING;
}

/// A Phosphor icon as `RichText` of a given size — one consistent optical
/// weight, so glyphs don't look randomly bigger/smaller.
pub(crate) fn icon(icon: &str, size: f32) -> egui::RichText {
    egui::RichText::new(icon).size(size)
}

/// "Icon + text" on one line: two spaces after the glyph, because Phosphor
/// glyphs carry inner padding — a single space would look glued together.
pub(crate) fn icon_label(icon: &str, label: &str, size: f32) -> egui::RichText {
    egui::RichText::new(format!("{icon}  {label}")).size(size)
}

/// Карточный заголовок: SemiBold-семейство `heading` (см. `shell::theme` —
/// в egui нет весов, SemiBold живёт отдельной FontFamily).
/// Принимает готовый `RichText` (например `icon_label(...)`).
pub(crate) fn heading(text: egui::RichText) -> egui::RichText {
    text.family(egui::FontFamily::Name(
        crate::app::shell::theme::HEADING_FAMILY.into(),
    ))
}

/// Подпись-«eyebrow» над секцией: маленький приглушённый ярлык
/// (категория данных, а не заголовок — без SemiBold).
pub(crate) fn eyebrow(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).small().weak()
}
