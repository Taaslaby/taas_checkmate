//! # Универсальная окраска нейронов (`color`)
//!
//! Цвет точки на атласе можно считать четырьмя способами:
//!
//! * **`Nt`** — по нейромедиатору (ACh/GABA/GLUT/…). Нужен для мозга мухи
//!   VNC; у коннектомов без НТ всё будет серым.
//! * **`Klass`** — по классу клетки (motor/sensory/ascending/…);
//! * **`Type`** — по типу клетки (оттенок = золотой угол от индекса типа):
//!   у C. elegans 272 типа → мозг цветной по идентичности клеток;
//! * **`Degree`** — по исходящей степени (лог-тепловая карта синий→красный):
//!   работает на ЛЮБОМ графе, подсвечивает хабы (нейроны с большим числом
//!   связей) — единственный режим, который не может быть «все серые».
//!
//! При загрузке (`ColorMode::auto`) выбирается первый режим, в котором
//! есть данные; пользователь может переключить вручную в тулките.

use eframe::egui::Color32;

use crate::io::data::MancData;

/// Режим окраски точек атласа/ego-графа.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    /// По нейромедиатору (только если в данных есть НТ).
    Nt,
    /// По классу клетки.
    Klass,
    /// По типу клетки (золотой угол от индекса типа).
    Type,
    /// По исходящей степени (лог-тепловая карта) — работает везде.
    Degree,
}

impl ColorMode {
    pub const fn all() -> [ColorMode; 4] {
        [Self::Nt, Self::Klass, Self::Type, Self::Degree]
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Nt => "НТ",
            Self::Klass => "Класс",
            Self::Type => "Тип",
            Self::Degree => "Степень",
        }
    }

    /// Автовыбор при загрузке мозга: первый режим с реальными данными.
    /// Сканирует дешёвые массивы (nt/тип-таблица/klass) — O(n), разово.
    pub fn auto(d: &MancData) -> ColorMode {
        if (0..d.n).any(|i| d.nt[i] != 0) {
            return Self::Nt;
        }
        // Тип клетки (T4b, Dm3q, …) — самая «настоящая» идентичность:
        // мозг мухи с аннотациями открывается цветным по типам.
        if d.types.len() > 1 {
            return Self::Type;
        }
        if (0..d.n).any(|i| d.klass[i] != 0) {
            return Self::Klass;
        }
        Self::Degree
    }
}

/// Цвет нейрона `i` в выбранном режиме.
pub fn color_for(d: &MancData, i: usize, mode: ColorMode) -> Color32 {
    match mode {
        ColorMode::Nt => d.nt(i).color(),
        ColorMode::Klass => klass_color(d.klass[i]),
        ColorMode::Type => type_color(d.type_idx[i]),
        ColorMode::Degree => degree_color(d.out_deg(i)),
    }
}

/// Палитра классов клеток (коды 0..7 из формата MNCV).
/// Публичная: легенда в инспекторе рисует те же цвета.
pub(crate) fn klass_color(k: u8) -> Color32 {
    match k {
        1 => Color32::from_rgb(235, 90, 90),   // motor
        2 => Color32::from_rgb(80, 200, 235),  // sensory
        3 => Color32::from_rgb(245, 165, 70),  // ascending
        4 => Color32::from_rgb(155, 110, 240), // descending
        5 => Color32::from_rgb(110, 210, 120), // intrinsic
        6 => Color32::from_rgb(205, 205, 215), // glia
        7 => Color32::from_rgb(240, 145, 235), // efferent
        _ => Color32::from_rgb(115, 115, 125), // other / неизвестно
    }
}

/// Оттенок по индексу типа: золотой угол (≈137.5°) — соседние типы
/// всегда разных цветов, ничего не сливается.
pub(crate) fn type_color(type_idx: u16) -> Color32 {
    let hue = (type_idx as f32 * 137.508) % 360.0;
    hsv(hue, 0.72, 0.92)
}

/// Лог-тепловая карта по исходящей степени: deg=0 → синий, deg≥10⁴ → красный.
/// Фиксированная шкала (без глобальной нормализации — O(1) на нейрон).
pub(crate) fn degree_color(deg: usize) -> Color32 {
    let t = ((deg as f32 + 1.0).log10() / 4.0).clamp(0.0, 1.0);
    hsv(240.0 - 240.0 * t, 0.85, 0.95)
}

/// HSV → Color32 (h: 0..360).
fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
