//! # Профайлер раскладки (`app::profiler`)
//!
//! Отладка вёрстки не ASCII-артом, а фактами: собирает фактические `Rect`
//! виджетов и пишет их в JSON. По хоткею `Ctrl+Shift+D` захватывается **ровно
//! один кадр** — в `connscope_profiler.json` падает список `{label, x, y, w, h,
//! cy, top, bottom}`, по `cy` (y-центру) видно, что и на сколько съехало.
//!
//! Использование: жми `Ctrl+Shift+D` в момент, когда нужный бар на экране →
//! читай `connscope_profiler.json` (или лог — туда тоже пишется путь).

use std::cell::{Cell, RefCell};

use eframe::egui::Rect;

thread_local! {
    /// Буфер замеров текущего захваченного кадра.
    static ENTRIES: RefCell<Vec<(String, Rect)>> = const { RefCell::new(Vec::new()) };
    /// Захват «взведён» хоткеем — начнётся на следующем кадре.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    /// Идёт захват текущего кадра.
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// Взводит захват (вызывается хоткеем).
pub(crate) fn arm() {
    ARMED.set(true);
}

/// Начало кадра: если был взведён захват — включаем сбор и чистим буфер.
pub(crate) fn begin_frame() {
    if ARMED.get() {
        ARMED.set(false);
        ENTRIES.with(|e| e.borrow_mut().clear());
        CAPTURING.set(true);
    }
}

/// Записать фактический прямоугольник виджета. Дешёвый no-op вне захвата.
pub(crate) fn record(label: impl Into<String>, rect: Rect) {
    if CAPTURING.get() {
        ENTRIES.with(|e| e.borrow_mut().push((label.into(), rect)));
    }
}

/// Конец кадра: если шёл захват — возвращает JSON-снапшот и гасит захват.
pub(crate) fn end_frame() -> Option<String> {
    if !CAPTURING.get() {
        return None;
    }
    CAPTURING.set(false);
    Some(serialize())
}

/// Сериализует буфер в компактный JSON.
fn serialize() -> String {
    let entries = ENTRIES.with(|e| e.borrow().clone());
    let mut s = String::from("{\n  \"entries\": [\n");
    for (i, (label, r)) in entries.iter().enumerate() {
        let comma = if i + 1 < entries.len() { "," } else { "" };
        s.push_str(&format!(
            "    {{\"label\": \"{label}\", \"x\": {:.1}, \"y\": {:.1}, \"w\": {:.1}, \"h\": {:.1}, \"cy\": {:.1}, \"top\": {:.1}, \"bottom\": {:.1}}}{comma}\n",
            r.left(),
            r.top(),
            r.width(),
            r.height(),
            r.center().y,
            r.top(),
            r.bottom(),
        ));
    }
    s.push_str("  ]\n}\n");
    s
}
