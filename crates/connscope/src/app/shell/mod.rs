//! # Рамка приложения (`app::shell`)
//!
//! Общие элементы каркаса, не привязанные к конкретному режиму:
//! меню, статус-бар, модалки оболочки, горячие клавиши, профайлер и тема.
//!
//! Поверхности, работающие с данными режима (верхняя панель, рейл, док,
//! консоль, ego-граф), живут в самих режимах — здесь они реализуют
//! [`ConnectomeMode`](crate::app::connectome::ConnectomeMode), потому что
//! их код исторически собран в `shell/` (воркбенч).

pub(crate) mod console;
pub(crate) mod console_ui;
pub(crate) mod menu;
pub(crate) mod panels;
pub(crate) mod profiler;
pub(crate) mod rail;
pub(crate) mod shortcuts;
pub(crate) mod status;
pub(crate) mod tabs;
pub(crate) mod theme;
pub(crate) mod top;
