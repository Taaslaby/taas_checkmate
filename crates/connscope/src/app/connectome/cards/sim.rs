//! # Карточка симуляции (`cards::sim`)
//!
//! Модель, полный VNC, трасса, слайдеры, запуск и живой поток. Сборка
//! «что показать» — в чистой [`sim_card_data`] (owned-строки), рендер — в
//! [`render_sim_card`], который работает только с `SimState`/`LiveState` и
//! возвращает команду [`SimAction`]. Оркестратор (`card_simulation` в
//! `inspector.rs`) применяет команду к остальному состоянию.

use eframe::egui;

use elegance::{Checkbox, MetricSlider, SegmentedControl, Slider as ElegSlider};

use egui_phosphor::regular as ic;

use crate::app::connectome::state::{LiveState, SimState};
use crate::app::tokens::{self, icon_label};
use crate::app::widgets::ActionTone;
use crate::sim::models;

/// Команда, которую карточка симуляции возвращает оркестратору — чтобы
/// рендер не зависел от всего `ConnApp`, а только от `SimState`/`LiveState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimAction {
    /// ▶ — запустить симуляцию (для полного VNC — живой поток в 3D).
    Run,
    /// ■ — остановить живой поток.
    Stop,
    /// Переключена «Трасса сигнала» — оркестратор сбрасывает кеш 3D.
    TraceToggled,
}

/// Собранные строки карточки симуляции — owned-данные для рендера.
pub(crate) struct SimCardData {
    /// Ярлыки моделей в порядке `sim::models::MODELS`.
    pub model_labels: Vec<&'static str>,
    /// Текст кнопки запуска (живой поток vs локальный прогон).
    pub run_label: &'static str,
    /// Иконка кнопки запуска.
    pub run_icon: &'static str,
    /// Живой поток активен (показываем «Остановить» вместо слайдера).
    pub live_running: bool,
    /// Счётчик шагов живого потока (если активен).
    pub live_counter: Option<String>,
    /// Модель поддерживает h-ток (показываем слайдер g_h).
    pub has_h_current: bool,
    /// Подсказка под слайдером скорости потока.
    pub live_hint: &'static str,
}

/// Чистый сборщик «что показать» для карточки симуляции: по `SimState` и
/// `LiveState` собирает owned-строки. Не трогает UI и состояние приложения —
/// легко тестировать и переиспользовать.
pub(crate) fn sim_card_data(sim: &SimState, live: &LiveState) -> SimCardData {
    let model_labels: Vec<&'static str> = models::MODELS.iter().map(|x| x.short_name()).collect();
    // Полный VNC всегда идёт живым потоком — кнопка и иконка это отражают.
    let (run_label, run_icon) = if sim.full {
        ("Запустить (живой поток в 3D)", ic::BROADCAST)
    } else {
        ("Запустить", ic::PLAY)
    };
    let live_running = live.stream.is_some();
    let live_counter = live_running.then(|| {
        format!(
            "шаг {}/{} · вспышки гаснут ~0.5 с",
            live.steps_done, sim.steps
        )
    });
    SimCardData {
        model_labels,
        run_label,
        run_icon,
        live_running,
        live_counter,
        has_h_current: models::MODELS[sim.model].has_h_current(),
        live_hint: "Полный VNC идёт живым потоком — включи 3D и смотри, как сигнал коммутируется.",
    }
}

/// Чистый рендер карточки симуляции: работает только с `SimState`/`LiveState`
/// (узкая сигнатура), а действия возвращает как [`SimAction`] — оркестратор
/// применяет их к камере/данным. Это и есть граница: карточка физически не
/// может трогать то, что ей не передали.
pub(crate) fn render_sim_card(
    ui: &mut egui::Ui,
    sim: &mut SimState,
    live: &mut LiveState,
) -> Option<SimAction> {
    let d = sim_card_data(sim, live);
    let mut action: Option<SimAction> = None;
    crate::app::widgets::section_card(ui, ic::LIGHTNING, "Симуляция", |ui| {
        // Модель — сегментированный переключатель из реестра плагинов:
        // добавил модель в `models::MODELS` — она сама появится здесь.
        let mut m = sim.model;
        ui.add(SegmentedControl::new(&mut m, d.model_labels));
        if m != sim.model {
            sim.model = m;
        }
        let mut full = sim.full;
        if ui
            .add(Checkbox::new(
                &mut full,
                icon_label(ic::LIGHTNING, "Полный VNC (вся сеть)", tokens::ICON_SM),
            ))
            .changed()
        {
            sim.full = full;
            if full && sim.steps > 400 {
                sim.steps = 200;
            }
        }
        let mut trace = sim.show_trace;
        if ui
            .add(Checkbox::new(
                &mut trace,
                icon_label(ic::THERMOMETER, "Трасса сигнала (3D)", tokens::ICON_SM),
            ))
            .changed()
        {
            sim.show_trace = trace;
            action = Some(SimAction::TraceToggled);
        }
        if sim.full {
            ui.add(ElegSlider::new(&mut sim.steps, 50..=1000).label("шаги"));
        } else {
            ui.add(ElegSlider::new(&mut sim.steps, 200..=4000).label("шаги"));
        }
        // MetricSlider — компактный слайдер с подписью и суффиксом.
        ui.add(
            MetricSlider::new(&mut sim.bias, 2.0..=60.0)
                .label("ток драйва")
                .suffix(" мА"),
        );
        ui.add(
            MetricSlider::new(&mut sim.pulse, 0.0..=1.0)
                .label("импульс")
                .decimals(2),
        );
        if d.has_h_current {
            ui.add(
                MetricSlider::new(&mut sim.gh, 0.1..=2.0)
                    .label("h-ток g_h")
                    .decimals(2),
            );
        }
        ui.add_space(tokens::SP_XS);
        // Кнопка действия — на всю ширину карточки, как в Blender.
        if crate::app::widgets::full_width_action(ui, d.run_icon, d.run_label, ActionTone::Primary)
        {
            action = Some(SimAction::Run);
        }
        if !sim.status.is_empty() {
            ui.label(egui::RichText::new(&sim.status).weak());
        }

        ui.add_space(tokens::SP_XS);
        ui.separator();
        // ── живой поток ──
        // Шаги сима идут в фоне, вспышки сработавших нейронов
        // летят на 3D-атлас — видно, как сигнал идёт по мозгу.
        if d.live_running {
            if crate::app::widgets::full_width_action(
                ui,
                ic::STOP,
                "■ Остановить поток",
                ActionTone::Danger,
            ) {
                action = Some(SimAction::Stop);
            }
            if let Some(counter) = &d.live_counter {
                ui.label(egui::RichText::new(counter).weak());
            }
        } else {
            ui.add(
                MetricSlider::new(&mut live.speed, 10.0..=500.0)
                    .label("скорость потока")
                    .suffix(" шагов/с")
                    .decimals(0),
            );
            ui.label(egui::RichText::new(d.live_hint).weak());
        }
        if !live.status.is_empty() {
            ui.label(egui::RichText::new(&live.status).weak());
        }
    });
    action
}
