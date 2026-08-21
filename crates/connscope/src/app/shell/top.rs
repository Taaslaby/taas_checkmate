//! # Тонкий верхний бар (`app::top`)
//!
//! **Одна строка** (Blender/Adobe): вкладки режимов · глобальный поиск ·
//! быстрые действия справа. Название приложения живёт в заголовке окна и
//! намеренно не дублируется внутри тулбара. Всё «вьюпортное» (проекции,
//! окраска, фильтры, настройки 3D) переехало в левый рейл инструментов
//! (`app::rail`), графики и консоль — в нижний док (`app::panels`).
//!
//! ```text
//! top_panel (40 px)
//! ├── tool_shelf      модульная полка: режимы, будущие инструменты
//! ├── search_group    поиск по типу/классу/НТ/ноге/body_id
//! └── action_group    ? справка (F1) — 3D/Fit живут в рейле и меню «Вид»
//! ```

use eframe::egui::{self, Key};

use egui_phosphor::regular as ic;

use super::shortcuts::SEARCH_FIELD_ID;
use crate::app::connectome::ConnectomeMode;
use crate::app::mode::{ShellServices, Surface};
use crate::app::tokens::{self, icon_label};

impl ConnectomeMode {
    // ── верхняя панель: одна строка ───────────────────────────
    pub(crate) fn top_panel(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        // Фиксированная строка под чипы, поиск и action-кнопки.
        // Для top используется отдельное spacing-правило: общий panel spacing
        // 8×6 делал разные оптические центры у чипов и текстового поля.
        egui::Panel::top("controls")
            .exact_size(tokens::TOP_BAR_H)
            .show(ui, |ui| {
                super::profiler::record("top", ui.max_rect());
                ui.spacing_mut().item_spacing = egui::vec2(tokens::TOP_GAP_X, tokens::TOP_GAP_Y);
                ui.spacing_mut().button_padding = egui::vec2(tokens::SP_MD, tokens::SP_XS);
                ui.horizontal_centered(|ui| {
                    // Каждый блок занимает только свою ответственность:
                    // добавление нового control не требует править всю строку.
                    self.top_tool_shelf(ui, services);
                    ui.separator();
                    self.top_search_group(ui);
                    ui.add_space(tokens::SP_MD);
                    // Оставшееся место отдаём action-группе справа.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        self.top_action_group(ui, services);
                    });
                });
            });
    }

    /// Модульная полка инструментов. Сейчас на ней находятся режимы атласа и
    /// ego-графа; новые инструменты добавляются сюда отдельными модулями, не
    /// превращая верхнюю строку в набор разрозненных кнопок.
    fn top_tool_shelf(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        egui::Frame::new()
            .fill(crate::app::shell::theme::BG_INSET)
            .stroke(egui::Stroke::new(
                1.0,
                crate::app::shell::theme::STROKE_SOFT,
            ))
            .corner_radius(egui::CornerRadius::same(
                crate::app::tokens::RADIUS_MD as u8,
            ))
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                self.top_mode_group(ui, services);
            });
    }

    /// Левая группа тулбара: переключение на вкладки рабочего пространства
    /// (Атлас / Ego) и открепление Ego-графа.
    fn top_mode_group(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        let rect = self.mode_tabs(ui, services);
        super::profiler::record("top_modes", rect);
    }

    /// Средняя группа тулбара: глобальный поиск.
    fn top_search_group(&mut self, ui: &mut egui::Ui) {
        let rect = self.quick_search(ui);
        super::profiler::record("top_search_group", rect);
    }

    /// Правая группа тулбара: только справка. Действия камеры (3D, Fit)
    /// живут в левом рейле и меню «Вид» — один инструмент в одном месте,
    /// топ-бар остаётся чистой навигацией (вкладки + поиск), как в Blender.
    fn top_action_group(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        if crate::app::widgets::icon_button(ui, ic::QUESTION, false, tokens::ICON_SM)
            .on_hover_text("Горячие клавиши [F1]")
            .clicked()
        {
            // Модалку «Горячие клавиши» рисует оболочка — просим её через сервисы.
            *services.show_shortcuts = true;
        }
        super::profiler::record("top_actions", ui.min_rect());
    }

    // ── вкладки режимов ───────────────────────────────────────
    /// Атлас / Ego — теперь вкладки рабочего пространства: клик фокусирует
    /// соответствующую вкладку в дереве доков (через `services.focus_surface`).
    /// Рядом — открепление/прикрепление Ego-графа.
    ///
    /// Обычные `selectable_label` (как чипы), а не elegance `TabBar`: TabBar
    /// жёстко 38px и в тонком баре не центрировался; чипы ровные.
    fn mode_tabs(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) -> egui::Rect {
        let mut rect = egui::Rect::NOTHING;
        // Активную вкладку подсвечиваем по фокусу дерева (оболочка передаёт
        // его в `services.focused`) — раньше оба чипа всегда были `false`.
        let atlas = ui
            .selectable_label(
                services.focused == Some(Surface::Atlas),
                icon_label(ic::COMPASS, "Атлас", tokens::ICON_XS),
            )
            .on_hover_text("Сфокусировать вкладку «Атлас»");
        rect = rect.union(atlas.rect);
        if atlas.clicked() {
            services.focus_surface = Some(Surface::Atlas);
        }
        let ego = ui
            .selectable_label(
                services.focused == Some(Surface::Ego),
                icon_label(ic::NETWORK, "Ego-граф", tokens::ICON_XS),
            )
            .on_hover_text("Сфокусировать вкладку «Ego-граф»");
        rect = rect.union(ego.rect);
        if ego.clicked() {
            // Клик по «Ego-граф» прикрепляет вкладку (убирает окно).
            self.ego_undocked = false;
            services.focus_surface = Some(Surface::Ego);
        }
        // Ego-граф: открепить в окно / вернуть в таб. Тот же чип-вид, что у
        // соседних вкладок (не отдельная кнопка) + подсветка, пока откреплён.
        let (txt, hint) = if self.ego_undocked {
            (
                icon_label(ic::ARROWS_OUT, "Прикрепить", tokens::ICON_XS),
                "Вернуть Ego-граф в таб",
            )
        } else {
            (
                icon_label(ic::ARROWS_OUT, "Открепить", tokens::ICON_XS),
                "Вытащить Ego-граф в перетаскиваемое окно (drag за заголовок)",
            )
        };
        let undock = ui
            .selectable_label(self.ego_undocked, txt)
            .on_hover_text(hint);
        rect = rect.union(undock.rect);
        if undock.clicked() {
            self.ego_undocked = !self.ego_undocked;
        }
        rect
    }

    // ── глобальный поиск ──────────────────────────────────────
    /// Поиск нейрона по типу/классу/НТ/ноге/body_id + статус результата.
    /// Доступность: видимый label + hint, статус — aria-live эквивалент
    /// (弱-текст с ролью статуса, читается скринридером при обновлении).
    fn quick_search(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let mut rect = egui::Rect::NOTHING;
        ui.label(
            egui::RichText::new("Поиск нейрона")
                .weak()
                .small()
                .color(crate::app::shell::theme::TEXT_DIM),
        );
        let resp = ui.add(
            egui::TextEdit::singleline(&mut self.sel.search)
                .id_salt(SEARCH_FIELD_ID)
                .desired_width(tokens::TOOL_SEARCH_W)
                .hint_text("тип / класс / НТ / нога / body_id …  (Ctrl+S)"),
        );
        // Явный accessible description — в egui попадает в WidgetInfo via
        // on_hover_text + id.
        let hover_rect = resp.rect;
        resp.clone()
            .on_hover_text("Поиск по типу, классу, нейротрансмиттеру, ноге или body_id");
        rect = rect.union(hover_rect);
        if resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
            self.search_neuron();
        }
        // Кнопка поиска — тот же единый `icon_button`, что у остальных
        // тулбарных кнопок (рейл/ракурсы): без чужой палитры elegance.
        let btn =
            crate::app::widgets::icon_button(ui, ic::MAGNIFYING_GLASS, false, tokens::ICON_SM)
                .on_hover_text("Найти [Enter]");
        rect = rect.union(btn.rect);
        if btn.clicked() {
            self.search_neuron();
        }
        if !self.sel.search_status.is_empty() {
            // aria-live: polite — статус поиска читается при изменении, но не
            // прерывает ввод. В egui эмулируем слабым текстом + selectable.
            let r = ui.label(
                egui::RichText::new(&self.sel.search_status)
                    .weak()
                    .small()
                    .italics(),
            );
            rect = rect.union(r.rect);
        }
        rect
    }
}
