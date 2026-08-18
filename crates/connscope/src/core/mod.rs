//! # Домен нейробиологии (`core`)
//!
//! Независимые от рендера/IO типы предметной области:
//!
//! * [`nt`] — нейромедиаторы и классы нейронов: имена, цвета, признаки
//!   тормозности ([`nt::Nt`], [`nt::Klass`]);
//! * [`color`] — режимы окраски точки ([`color::ColorMode`]) и функции
//!   `color_for` / `klass_color` / `type_color` / `degree_color`;
//! * [`pharma`] — библиотека препаратов: [`pharma::Drug`], [`pharma::Target`],
//!   [`pharma::Effect`] и Хилл-маппинг концентрация → множитель.

pub(crate) mod color;
pub(crate) mod nt;
pub(crate) mod pharma;
