//! # Карточки инспектора (`app::connectome::cards`)
//!
//! Каждая плашка правой панели — **отдельный модуль** с чистыми границами:
//!
//! ```text
//! cards/
//! ├── neuron.rs   выбранный нейрон: поля, плашки, связи, ego-граф
//! ├── sim.rs      модель · полный VNC · трасса · слайдеры · запуск · поток
//! ├── disin.rs    детектор тормоз →⊣ тормоз →⊣ нейрон
//! ├── pharma.rs   препарат + концентрация → множитель проводимости
//! └── legend.rs   шпаргалка цветов текущего режима окраски
//! ```
//!
//! Паттерн у всех карточек один: **чистый сборщик данных** (`*_card_data`,
//! owned-строки, без UI) + **свободный рендер** (`render_*_card`, узкая
//! сигнатура, возвращает команду `*Action`) + **тонкий оркестратор**
//! (`card_*` в `inspector.rs`, применяет команду к остальному состоянию).
//! Так карточка физически не может трогать то, что ей не передали.

pub(crate) mod disin;
pub(crate) mod legend;
pub(crate) mod neuron;
pub(crate) mod pharma;
pub(crate) mod sim;

// Re-exports — инспектору нужны только рендеры, команды и общий сборщик
// нейрона; остальные сборщики (`legend_data`, `sim_card_data`, …) живут в
// своих модулях и используются тестами напрямую.
pub(crate) use disin::{render_disin_card, DisinAction};
pub(crate) use legend::render_legend_card;
pub(crate) use neuron::{neuron_card_data, render_neuron_card, NeuronAction};
pub(crate) use pharma::render_pharma_card;
pub(crate) use sim::{render_sim_card, SimAction};

#[cfg(test)]
mod tests {
    use super::disin::disin_card_data;
    use super::legend::legend_data;
    use super::neuron::neuron_card_data;
    use super::sim::sim_card_data;
    use crate::app::connectome::state::{DisinState, LiveState, SimState};
    use crate::core::color::ColorMode;
    use crate::io::data::MancData;
    use std::path::Path;

    /// Реальный запечённый VNC (есть в ресурсах — как в других тестах).
    fn tiny_data() -> MancData {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/examples/data");
        let path = dir.join("manc_viz.bin");
        MancData::detect(&path).expect("manc_viz.bin парсится")
    }

    /// Карточка нейрона: заголовок начинается с `#`, поля не пустые.
    #[test]
    fn neuron_card_has_heading_and_fields() {
        let mut d = tiny_data();
        let card = neuron_card_data(&mut d, 0);
        assert!(card.heading.starts_with('#'), "заголовок = #id [+ тип]");
        assert!(!card.fields.is_empty(), "поля карточки не пустые");
        // Первое поле — ID тела (всегда есть).
        assert_eq!(card.fields[0].0, "ID тела");
        assert!(!card.klass.is_empty());
        assert!(!card.nt.is_empty());
    }

    /// Легенда по НТ: строки для каждого присутствующего медиатора.
    #[test]
    fn legend_nt_has_rows() {
        let d = tiny_data();
        let (title, _, rows) = legend_data(Some(&d), ColorMode::Nt);
        assert_eq!(title, "Нейротрансмиттеры");
        assert!(!rows.is_empty(), "легенда НТ не пустая");
        // Каждая строка: имя + цвет + счётчик.
        for (name, _, cnt) in &rows {
            assert!(!name.is_empty());
            assert!(cnt.is_some(), "счётчик нейронов всегда есть");
        }
    }

    /// Легенда по классам: строки для присутствующих классов.
    #[test]
    fn legend_klass_has_rows() {
        let d = tiny_data();
        let (title, _, rows) = legend_data(Some(&d), ColorMode::Klass);
        assert_eq!(title, "Классы клеток");
        assert!(!rows.is_empty());
    }

    /// Легенда по типам: заголовок с числом уникальных типов, не больше 8
    /// строк (топ по числу нейронов), у каждой строки есть счётчик.
    #[test]
    fn legend_type_has_top8_with_counts() {
        let d = tiny_data();
        let (title, _, rows) = legend_data(Some(&d), ColorMode::Type);
        assert!(
            title.starts_with("Типы клеток — "),
            "заголовок = Типы клеток — N"
        );
        assert!(!rows.is_empty());
        assert!(rows.len() <= 8, "топ-8, получили {}", rows.len());
        for (name, _, cnt) in &rows {
            assert!(!name.is_empty());
            assert!(cnt.is_some(), "у типа всегда есть счётчик");
        }
    }

    /// Легенда по степени: ровно 6 строк шкалы без счётчиков (градиент).
    #[test]
    fn legend_degree_has_scale_without_counts() {
        let d = tiny_data();
        let (title, _, rows) = legend_data(Some(&d), ColorMode::Degree);
        assert_eq!(title, "Степень (исходящие связи)");
        assert_eq!(rows.len(), 6, "шкала 1..100К — 6 ступеней");
        for (_, _, cnt) in &rows {
            assert!(cnt.is_none(), "у шкалы нет счётчика");
        }
    }

    /// Сборщик карточки симуляции: дефолтное состояние — локальный прогон,
    /// поток не запущен, счётчика нет, подсказка на месте.
    #[test]
    fn sim_card_default_is_local_and_idle() {
        let sim = SimState::default();
        let live = LiveState::default();
        let d = sim_card_data(&sim, &live);
        assert!(!d.model_labels.is_empty(), "ярлыки моделей есть");
        assert_eq!(
            d.run_label, "Запустить",
            "локальный прогон — обычная кнопка"
        );
        assert!(!d.live_running, "поток не запущен по умолчанию");
        assert!(d.live_counter.is_none(), "без потока счётчика нет");
        assert!(!d.live_hint.is_empty());
    }

    /// Полный VNC всегда идёт живым потоком — кнопка/иконка это отражают.
    #[test]
    fn sim_card_full_switches_to_live_label() {
        let sim = SimState {
            full: true,
            ..Default::default()
        };
        let live = LiveState::default();
        let d = sim_card_data(&sim, &live);
        assert_eq!(d.run_label, "Запустить (живой поток в 3D)");
        assert_eq!(d.run_icon, egui_phosphor::regular::BROADCAST);
    }

    /// Счётчик шагов появляется только при активном потоке и форматируется
    /// как «шаг N/M · вспышки гаснут ~0.5 с».
    #[test]
    fn sim_card_live_counter_formats_steps() {
        let sim = SimState {
            steps: 800,
            ..Default::default()
        };
        let (_tx, rx) = std::sync::mpsc::channel();
        let live = LiveState {
            stream: Some(crate::sim::SimStream {
                rx,
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                handle: None,
            }),
            steps_done: 12,
            ..Default::default()
        };
        let d = sim_card_data(&sim, &live);
        assert!(d.live_running, "поток активен");
        assert_eq!(
            d.live_counter.as_deref(),
            Some("шаг 12/800 · вспышки гаснут ~0.5 с")
        );
    }

    /// Сборщик растормаживания: цепь `(mid, y, w1, w2)` форматируется
    /// как «#id тип (НТ) через #mid w=…·…», статус копируется.
    #[test]
    fn disin_card_data_formats_chains() {
        let d = tiny_data();
        let disin = DisinState {
            chains: vec![(1, 0, 2.0, 3.0)],
            status: "1 цепей тормоз→⊣тормоз→⊣нейрон".into(),
        };
        let card = disin_card_data(Some(&d), &disin);
        assert_eq!(card.status, disin.status, "статус копируется");
        assert_eq!(card.chains.len(), 1);
        assert!(card.chains[0].starts_with("#0 "), "цепь начинается с #id");
        assert!(card.chains[0].contains("через #1"), "промежуточный тормоз");
        assert!(card.chains[0].contains("w=2·3"), "веса в строке");
    }

    /// Без данных и цепей — пустые строки, без паники.
    #[test]
    fn disin_card_data_empty_without_data() {
        let disin = DisinState::default();
        let card = disin_card_data(None, &disin);
        assert!(card.chains.is_empty());
        assert!(card.status.is_empty());
    }
}
