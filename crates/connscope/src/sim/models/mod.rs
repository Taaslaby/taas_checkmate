//! # Плагины моделей нейронов (`sim::models`)
//!
//! Каждая модель — отдельный файл, реализующий трейт [`NeuronModel`].
//! Вся механика (имя, конфиг pillars, настройка runner'а, доп. слайдер)
//! живёт в этом файле, а [`MODELS`] — просто реестр в порядке отображения.
//!
//! **Добавить новый нейрон = создать файл `foo.rs` + одну строку в [`MODELS`].**
//! Никакого enum и `match` по моделям в остальном коде больше нет — вместо
//! этого в состоянии хранится **индекс** в [`MODELS`].

use pillars_connectome::{BrainRunner, NeuronConfig};

use crate::core::pharma::Target;

mod haap;
mod hh;
mod izh;
mod lif;

/// Модель нейрона как плагин: имя + конфиг pillars + опциональная настройка
/// runner'а и признак доп. параметра (h-ток).
pub trait NeuronModel: Sync {
    /// Короткое имя для сегментированного переключателя («LIF», «Izh», …).
    fn short_name(&self) -> &'static str;
    /// Полное имя для строк статуса («LIF», «Izhikevich», …).
    fn name(&self) -> &'static str;
    /// Конфиг pillars-connectome для этой модели при шаге `dt_ms`.
    fn config(&self, dt_ms: f32) -> NeuronConfig;
    /// Донастройка runner'а после создания (например, колонка h-тока HAAP).
    /// По умолчанию — нет.
    fn tune(&self, _runner: &mut BrainRunner, _n: usize, _gh: f32) {}
    /// Показывать ли в UI слайдер h-тока (`g_h`). По умолчанию — нет.
    fn has_h_current(&self) -> bool {
        false
    }
    /// Есть ли у модели проводимость-мишень препарата (канальные мишени).
    /// Рецепторные мишени (синапсы) подходят любой модели — сюда не входят.
    fn has_channel(&self, _target: Target) -> bool {
        false
    }
}

/// Все модели в порядке отображения в UI. **Новый нейрон — одна строка здесь.**
pub const MODELS: &[&dyn NeuronModel] = &[&lif::Lif, &izh::Izh, &hh::Hh, &haap::Haap];

/// Индекс модели по короткому/полному имени (без учёта регистра). CLI/консоль.
pub fn index_of_name(s: &str) -> Option<usize> {
    let s = s.to_ascii_lowercase();
    MODELS.iter().position(|m| {
        m.short_name().to_ascii_lowercase() == s || m.name().to_ascii_lowercase() == s
    })
}
