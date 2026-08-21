//! # Инспектор (`app::inspector`)
//!
//! Вкладка рабочего пространства, собранная из **карточек как API**: каждая
//! плашка живёт в своём модуле `app::connectome::cards` (чистый сборщик
//! данных + свободный рендер с командой `*Action`), а здесь — только тонкие
//! оркестраторы `card_*`, которые применяют команды к остальному состоянию
//! `ConnApp`.
//!
//! ```text
//! Инспектор (вкладка дока)
//! ├── card_neuron         карточка выбранного нейрона (поля, плашки, связи, ego)
//! ├── card_simulation     модель · полный VNC · трасса · слайдеры · запуск · поток
//! ├── card_pharma         препарат + концентрация → множитель проводимости
//! ├── card_disinhibition  детектор тормоз →⊣ тормоз →⊣ нейрон
//! └── card_legend         шпаргалка цветов текущего режима окраски
//! ```
//!
//! Адаптивность: панель ресайзится (280–640 pt), контент скроллится, а
//! слайдеры растягиваются на всю доступную ширину карточки (elegance по
//! умолчанию берёт `ui.available_width()` — жёстких ширин здесь нет).

use eframe::egui;

use crate::app::connectome::cards::{
    neuron_card_data, render_disin_card, render_legend_card, render_neuron_card,
    render_pharma_card, render_sim_card, DisinAction, NeuronAction, SimAction,
};
use crate::app::connectome::ConnectomeMode;
use crate::app::mode::{ShellServices, Surface};
use crate::app::tokens;

impl ConnectomeMode {
    // ── инспектор (правая панель) ─────────────────────────────
    /// Правая панель: заголовок + скроллируемый список карточек.
    ///
    /// Порядок карточек задаётся здесь — это и есть «реестр» плашек. Новую
    /// карточку добавляем в конец списка (или между соседними).
    /// Инспектор как вкладка рабочего пространства (без `Panel`-обёртки —
    /// контейнер даёт дерево egui_dock).
    pub(crate) fn inspector_content(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        crate::app::shell::profiler::record("inspector", ui.max_rect());
        tokens::spacing(ui);
        ui.heading("Инспектор");
        // Весь контент скроллится: на маленьком окне карточки не
        // обрезаются, а уезжают под скролл.
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.card_neuron(ui, services);
                ui.add_space(tokens::GAP_CARDS);
                self.card_simulation(ui);
                ui.add_space(tokens::GAP_CARDS);
                self.card_pharma(ui);
                ui.add_space(tokens::GAP_CARDS);
                self.card_disinhibition(ui);
                ui.add_space(tokens::GAP_CARDS);
                self.card_legend(ui);
            });
    }

    // ── карточка нейрона ──────────────────────────────────────
    /// Оркестратор карточки нейрона: сборка — в `cards::neuron_card_data`,
    /// рендер — в `cards::render_neuron_card` (возвращает [`NeuronAction`]);
    /// здесь команда применяется к view-состоянию и атласу.
    fn card_neuron(&mut self, ui: &mut egui::Ui, services: &mut ShellServices) {
        let Some(i) = self.sel.selected else {
            ui.weak("Кликните по нейрону — увидите его карточку и связи.");
            return;
        };
        let Some(d) = self.data.as_mut() else {
            return;
        };
        let card = neuron_card_data(d, i);
        let (nn, ne) = (
            self.sel.ego_graph.g().node_count(),
            self.sel.ego_graph.g().edge_count(),
        );
        match render_neuron_card(ui, &card, nn, ne) {
            Some(NeuronAction::OpenTab) => {
                self.ego_undocked = false;
                services.focus_surface = Some(Surface::Ego);
            }
            Some(NeuronAction::UndockWindow) => self.ego_undocked = true,
            Some(NeuronAction::FocusNeighborhood) => self.focus_ego(),
            None => {}
        }
    }

    // ── карточка симуляции ────────────────────────────────────
    /// Оркестратор карточки симуляции: рендер — в `cards::render_sim_card`
    /// (работает только с `SimState`/`LiveState`, возвращает [`SimAction`]);
    /// здесь команда применяется к остальному состоянию (запуск/остановка
    /// потока, сброс 3D-кеша трассы).
    fn card_simulation(&mut self, ui: &mut egui::Ui) {
        match render_sim_card(ui, &mut self.sim, &mut self.live) {
            Some(SimAction::Run) => self.run_sim(),
            Some(SimAction::Stop) => self.stop_live(),
            Some(SimAction::TraceToggled) => self.camera.atlas3d = None,
            None => {}
        }
    }

    // ── карточка фармакологии ─────────────────────────────────
    /// Прокси на чистую свободную функцию `cards::render_pharma_card`:
    /// карточка трогает только `self.pharma` (модель — для подсказки
    /// совместимости мишени с проводимостями модели).
    fn card_pharma(&mut self, ui: &mut egui::Ui) {
        render_pharma_card(ui, &mut self.pharma, self.sim.model);
    }

    // ── карточка растормаживания ──────────────────────────────
    /// Оркестратор карточки растормаживания: рендер — в
    /// `cards::render_disin_card` (работает только с `DisinState` и данными,
    /// возвращает [`DisinAction`]); здесь команда применяется к `self`.
    fn card_disinhibition(&mut self, ui: &mut egui::Ui) {
        if render_disin_card(ui, &self.disin, self.data.as_ref()) == Some(DisinAction::Find) {
            self.run_disin();
        }
    }

    // ── карточка легенды ──────────────────────────────────────
    /// Прокси на чистую свободную функцию `cards::render_legend_card` —
    /// карточка не мутирует состояние, поэтому не требует `&mut self`.
    fn card_legend(&mut self, ui: &mut egui::Ui) {
        render_legend_card(ui, self.data.as_ref(), self.filters.color_mode);
    }
}
