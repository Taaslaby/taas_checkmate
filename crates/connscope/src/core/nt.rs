//! Нейротрансмиттеры и классы нейронов (цвета, названия, признаки тормозности).

use eframe::egui::Color32;

// ─── NT (нейромедиатор) — цвет как в инсектной нейрохимии ──────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nt {
    Unknown = 0,
    Ach = 1,
    Gaba = 2,
    Glut = 3,
    Da = 4,
    Ser = 5,
    Oct = 6,
}

impl Nt {
    pub fn from_code(c: u8) -> Nt {
        match c {
            1 => Nt::Ach,
            2 => Nt::Gaba,
            3 => Nt::Glut,
            4 => Nt::Da,
            5 => Nt::Ser,
            6 => Nt::Oct,
            _ => Nt::Unknown,
        }
    }
    pub fn color(self) -> Color32 {
        match self {
            Nt::Ach => Color32::from_rgb(80, 210, 140), // возб. у насекомых
            Nt::Gaba => Color32::from_rgb(225, 85, 85), // торм.
            Nt::Glut => Color32::from_rgb(85, 145, 235), // GluCl — торм. Cl⁻
            Nt::Da => Color32::from_rgb(215, 120, 225),
            Nt::Ser => Color32::from_rgb(245, 165, 85),
            Nt::Oct => Color32::from_rgb(235, 215, 95),
            Nt::Unknown => Color32::from_rgb(125, 125, 135),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Nt::Ach => "ACh (возб.)",
            Nt::Gaba => "GABA (торм.)",
            Nt::Glut => "GLUT (торм.)",
            Nt::Da => "DA (модул.)",
            Nt::Ser => "SER (модул.)",
            Nt::Oct => "OCT (модул.)",
            Nt::Unknown => "unknown",
        }
    }
    pub fn inhibitory(self) -> bool {
        matches!(self, Nt::Gaba | Nt::Glut)
    }
}

// ─── Класс нейрона ─────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Klass {
    Other = 0,
    Motor = 1,
    Sensory = 2,
    Ascending = 3,
    Descending = 4,
    Intrinsic = 5,
    Glia = 6,
    Efferent = 7,
}

impl Klass {
    pub fn from_code(c: u8) -> Klass {
        match c {
            1 => Klass::Motor,
            2 => Klass::Sensory,
            3 => Klass::Ascending,
            4 => Klass::Descending,
            5 => Klass::Intrinsic,
            6 => Klass::Glia,
            7 => Klass::Efferent,
            _ => Klass::Other,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Klass::Other => "other",
            Klass::Motor => "motor",
            Klass::Sensory => "sensory",
            Klass::Ascending => "ascending",
            Klass::Descending => "descending",
            Klass::Intrinsic => "intrinsic",
            Klass::Glia => "glia",
            Klass::Efferent => "efferent",
        }
    }
}
