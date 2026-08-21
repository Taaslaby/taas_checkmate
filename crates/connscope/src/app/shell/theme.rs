//! # Тема neoMyelin (`shell::theme`)
//!
//! Единый визуальный язык приложения: шрифты, палитра, типографика.
//!
//! **Типографика** — научный инструмент, а не сайт:
//! * IBM Plex Sans — интерфейс (Regular тело / SemiBold заголовки через
//!   семейство `heading` — в egui нет весов, поэтому SemiBold живёт
//!   отдельной FontFamily);
//! * JetBrains Mono — данные: консоль, индексы, числа.
//! * `.strong()` в egui — это лишь более яркий цвет; настоящая иерархия —
//!   `tokens::heading()`.
//!
//! **Палитра** — тёмный «прибор»: **нейтральный тёмно-серый** (R=G=B, без
//! синевы — синий подтон фона выглядел «дёшево»), панели разделены
//! ступенью яркости, а не жёсткими рамками; единственный сигнальный
//! акцент (умеренный осциллографический зелёный) расходуется только на
//! активное/живое: выделение, активную вкладку, идущие симуляции. Всё
//! остальное — тихие нейтральные серые. Иконки Phosphor уже вставлены в
//! `run()` — наборы глифов не пересекаются, fallback работает в обе стороны.

use crate::app::tokens;
use eframe::egui;

// ── палитра ───────────────────────────────────────────────────
/// Базовая поверхность оболочки — проф тёмный нейтральный (R=G=B, Houdini/Nuke),
/// без цветного подтона. Панели делятся яркостью, не рамками.
pub(crate) const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(0x18, 0x1A, 0x1C);
/// Поднятая поверхность карточек, доков и окон — на ступень светлее панели.
pub(crate) const BG_CARD: egui::Color32 = egui::Color32::from_rgb(0x22, 0x24, 0x26);
/// Поля ввода и «утопленные» области — чуть темнее панели.
pub(crate) const BG_INSET: egui::Color32 = egui::Color32::from_rgb(0x12, 0x14, 0x15);
/// Тихая рамка-разделитель — нейтральная, без синевы.
pub(crate) const STROKE_SOFT: egui::Color32 = egui::Color32::from_rgb(0x2E, 0x32, 0x34);
/// Основной текст — тёплый белый, не чисто белый (меньше засветки).
pub(crate) const TEXT: egui::Color32 = egui::Color32::from_rgb(0xE8, 0xE8, 0xEA);
/// Второстепенный текст — средне-серый, WCAG AA на BG_PANEL.
pub(crate) const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(0x9A, 0x9E, 0xA2);
/// Приглушённый красный — единственное цветное исключение: ошибки и «стоп».
/// Один источник danger-красного для всего приложения (`ActionTone::Danger`).
pub(crate) const DANGER: egui::Color32 = egui::Color32::from_rgb(0xA8, 0x44, 0x44);
/// Приглушённый янтарный — предупреждения (валидация, пропущенные элементы).
pub(crate) const WARNING: egui::Color32 = egui::Color32::from_rgb(0xE0, 0xB8, 0x4A);
/// Фон неактивного контрола (кнопки, комбо) — чуть светлее панели.
pub(crate) const BG_CTRL: egui::Color32 = egui::Color32::from_rgb(0x25, 0x2A, 0x2B);
/// Hover контрола.
pub(crate) const BG_CTRL_HOVER: egui::Color32 = egui::Color32::from_rgb(0x31, 0x38, 0x39);
/// Нажатый контрол.
pub(crate) const BG_CTRL_ACTIVE: egui::Color32 = egui::Color32::from_rgb(0x3A, 0x43, 0x43);
/// Открытый/развёрнутый контрол (попап-кнопка).
pub(crate) const BG_CTRL_OPEN: egui::Color32 = egui::Color32::from_rgb(0x2A, 0x30, 0x31);

/// Имя семейства заголовков (SemiBold-начертание Plex).
pub(crate) const HEADING_FAMILY: &str = "heading";

/// Текущий сигнальный акцент интерфейса — читается из `visuals`, куда его
/// кладёт elegance (`Palette::focus` зеркалится в `hyperlink_color` при
/// `Theme::install`). Единая точка для паинтерных кнопок/тулбара/статусов:
/// смена `AccentPreset` перекрашивает их вместе с вкладками и выделением.
pub(crate) fn accent(ui: &egui::Ui) -> egui::Color32 {
    ui.visuals().hyperlink_color
}

/// Стиль `egui_dock` под дизайн-систему neoMyelin.
///
/// Строится от **runtime** `ui.visuals()` (тема уже установлена) + [`accent`],
/// поэтому док автоматически следует активной теме и перекрашивается при смене
/// [`crate::app::AccentPreset`]. Принцип тот же, что у паинтерных виджетов:
/// активная вкладка — единственный сигнальный акцент, остальное — тихие тона
/// темы, поверхности делятся яркостью, а не жёсткими рамками.
pub(crate) fn dock_style(ui: &egui::Ui) -> egui_dock::Style {
    let accent = accent(ui);
    let v = ui.visuals();
    let mut s = egui_dock::Style::from_egui(ui.style().as_ref());

    // ── вкладки: приподнятый «документ», активная сигналит акцентом ──
    s.tab.active.bg_fill = v.window_fill;
    s.tab.active.text_color = accent;
    s.tab.active.outline_color = v.widgets.noninteractive.bg_stroke.color;
    s.tab.focused = s.tab.active.clone();
    s.tab.inactive.bg_fill = egui::Color32::TRANSPARENT;
    s.tab.inactive.text_color = v.weak_text_color();
    s.tab.inactive.outline_color = egui::Color32::TRANSPARENT;
    s.tab.hovered.bg_fill = v.widgets.hovered.bg_fill;
    s.tab.hovered.text_color = v.strong_text_color();
    // Варианты «с фокусом клавиатуры» — как базовые, чтобы вкладки не «прыгали».
    s.tab.active_with_kb_focus = s.tab.active.clone();
    s.tab.inactive_with_kb_focus = s.tab.inactive.clone();
    s.tab.focused_with_kb_focus = s.tab.focused.clone();

    // Тело вкладки — та же «карточка», что и остальной контент, без рамки.
    s.tab.tab_body.bg_fill = v.window_fill;
    s.tab.tab_body.stroke = egui::Stroke::NONE;
    s.tab.tab_body.inner_margin = egui::Margin::same(tokens::SP_MD as i8);

    // ── таб-бар: полоса в цвет панели, тихая линия под ним, скругление сверху ──
    s.tab_bar.bg_fill = v.panel_fill;
    s.tab_bar.hline_color = v.widgets.noninteractive.bg_stroke.color;
    let radius = egui::CornerRadius {
        nw: tokens::RADIUS_TB as u8,
        ne: tokens::RADIUS_TB as u8,
        sw: 0,
        se: 0,
    };
    s.tab_bar.corner_radius = radius;
    for state in [
        &mut s.tab.active,
        &mut s.tab.inactive,
        &mut s.tab.focused,
        &mut s.tab.hovered,
        &mut s.tab.inactive_with_kb_focus,
        &mut s.tab.active_with_kb_focus,
        &mut s.tab.focused_with_kb_focus,
    ] {
        state.corner_radius = radius;
    }

    // ── сплиты и оверлей перетаскивания — сигнальный акцент ──
    s.separator.color_idle = v.widgets.noninteractive.bg_stroke.color;
    s.separator.color_hovered = accent;
    s.separator.color_dragged = accent;
    s.overlay.selection_color = accent;

    s
}

/// Шрифты + стиль. Вызывается один раз из [`crate::app::run`] после
/// вставки Phosphor (наборы глифов не пересекаются — порядок не важен).
pub(crate) fn install(ctx: &egui::Context) {
    fonts(ctx);
    visuals(ctx);
}

fn fonts(ctx: &egui::Context) {
    let insert =
        |name: &str, file: &'static [u8], families: Vec<egui::epaint::text::InsertFontFamily>| {
            ctx.add_font(egui::epaint::text::FontInsert::new(
                name,
                egui::FontData::from_static(file),
                families,
            ));
        };
    let primary = |family: egui::FontFamily| {
        vec![egui::epaint::text::InsertFontFamily {
            family,
            priority: egui::epaint::text::FontPriority::Highest,
        }]
    };

    // Тело интерфейса: кириллица Plex перекрывает дефолтный Ubuntu,
    // недостающие глифы (иконки) падают дальше по списку семейства.
    insert(
        "plex",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../resources/fonts/IBMPlexSans-Regular.ttf"
        )),
        primary(egui::FontFamily::Proportional),
    );
    // Заголовки: SemiBold живёт отдельным семейством — см. tokens::heading().
    insert(
        "plex-semibold",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../resources/fonts/IBMPlexSans-SemiBold.ttf"
        )),
        primary(egui::FontFamily::Name(HEADING_FAMILY.into())),
    );
    // Данные: консоль, body_id, числа — моно с кириллицей.
    insert(
        "jetbrains-mono",
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../resources/fonts/JetBrainsMono-Regular.ttf"
        )),
        primary(egui::FontFamily::Monospace),
    );
    // Medium не вставляем: egui не умеет веса, третья копия только
    // распухает текстурах — двух начертаний достаточно.
}

fn visuals(ctx: &egui::Context) {
    use egui::Color32;
    use egui::Visuals;

    let mut v = Visuals::dark();

    v.panel_fill = BG_PANEL;
    v.window_fill = BG_CARD;
    v.extreme_bg_color = BG_INSET; // поля ввода, «утопленные» области
    v.faint_bg_color = BG_CARD; // чередование строк таблиц
                                // Акцент и выделение не задаём здесь: их перекрашивает elegance
                                // (`Theme::install` → `Palette::focus`), а виджеты читают `theme::accent(ui)`.

    // Панели разделяются яркостью (panel_fill vs window_fill); рамка окна
    // — едва заметная (у панелей рамки в egui 0.35 нет вообще).
    v.window_stroke = egui::Stroke::new(0.5, STROKE_SOFT);

    let w = &mut v.widgets;
    w.inactive.bg_fill = BG_CTRL;
    w.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    w.inactive.bg_stroke = egui::Stroke::new(0.5, STROKE_SOFT);
    w.hovered.bg_fill = BG_CTRL_HOVER;
    w.hovered.fg_stroke = egui::Stroke::new(1.0, Color32::WHITE);
    w.active.bg_fill = BG_CTRL_ACTIVE;
    w.active.fg_stroke = egui::Stroke::new(1.0, Color32::WHITE);
    w.open.bg_fill = BG_CTRL_OPEN;
    // Круглее дефолтных «пилюль», но не пузыри.
    let radius = egui::CornerRadius::same(tokens::RADIUS_MD as u8);
    w.inactive.corner_radius = radius;
    w.hovered.corner_radius = radius;
    w.active.corner_radius = radius;
    w.noninteractive.bg_fill = BG_PANEL;
    w.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    w.noninteractive.bg_stroke = egui::Stroke::NONE;

    // Меню/тосты: чуть заметная тень вместо «коробки».
    let soft_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(120),
    };
    v.popup_shadow = soft_shadow;
    v.window_shadow = soft_shadow;

    let mut style_dark = (*ctx.style_of(egui::Theme::Dark)).clone();
    style_dark.visuals = v;

    // Типографическая шкала — из tokens::TEXT_* (единый источник размеров).
    use egui::{FontId, TextStyle};
    style_dark.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(
                tokens::TEXT_HEADING,
                egui::FontFamily::Name(HEADING_FAMILY.into()),
            ),
        ),
        (
            TextStyle::Body,
            FontId::new(tokens::TEXT_BODY, egui::FontFamily::Proportional),
        ),
        (
            TextStyle::Button,
            FontId::new(tokens::TEXT_BODY, egui::FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(tokens::TEXT_SMALL, egui::FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(tokens::TEXT_MONO, egui::FontFamily::Monospace),
        ),
    ]
    .into();

    // Спacing-шкала ядра egui — в терминах нашего кита (4px-сетки).
    style_dark.spacing.item_spacing = tokens::ITEM_SPACING;
    style_dark.spacing.button_padding = egui::vec2(tokens::SP_MD + 1.0, tokens::SP_SM - 1.0);
    style_dark.spacing.menu_margin = egui::Margin::same(tokens::SP_SM as i8);
    style_dark.spacing.window_margin = egui::Margin::same(tokens::SP_LG as i8);

    ctx.set_style_of(egui::Theme::Dark, style_dark.clone());

    // Светлая тема — без чёрных полос: светлые панели, тёмный текст, те же акценты
    let mut v_light = Visuals::light();
    v_light.panel_fill = Color32::from_rgb(0xF0, 0xF2, 0xF5);
    v_light.window_fill = Color32::WHITE;
    v_light.extreme_bg_color = Color32::from_rgb(0xE8, 0xEA, 0xED);
    v_light.faint_bg_color = Color32::WHITE;
    v_light.window_stroke = egui::Stroke::new(1.0, Color32::from_rgb(0xD0, 0xD4, 0xD8));
    v_light.widgets.inactive.bg_fill = Color32::from_rgb(0xE8, 0xEA, 0xED);
    v_light.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, Color32::from_rgb(0x1A, 0x1D, 0x1F));
    v_light.widgets.hovered.bg_fill = Color32::from_rgb(0xDE, 0xE2, 0xE6);
    v_light.widgets.active.bg_fill = Color32::from_rgb(0xD0, 0xD4, 0xD8);
    v_light.widgets.noninteractive.bg_fill = Color32::from_rgb(0xF0, 0xF2, 0xF5);
    let mut style_light = (*ctx.style_of(egui::Theme::Light)).clone();
    style_light.visuals = v_light;
    style_light.text_styles = style_dark.text_styles.clone();
    style_light.spacing = style_dark.spacing;
    ctx.set_style_of(egui::Theme::Light, style_light);
}
