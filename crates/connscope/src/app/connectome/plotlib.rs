//! # Библиотека плоттеров (`app::plotlib`)
//!
//! Каталог всех графиков симуляции + их состояние: **включён ли** плот и
//! **откреплён ли** в плавающее окно (как ego-граф). Модалка «Плоттеры»
//! управляет состоянием, `app::plots` рисует.
//!
//! Плоты делятся на два рода — к результату **локальной** симуляции
//! (ego-контур) и к **полному** прогону (PSTH/спектр/тепловая карта).

use egui_phosphor::regular as ic;

use super::plotlib::PlotKind::*;

/// Тип плота (индекс в `CATALOG` / массиве состояний).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlotKind {
    LocalRaster,
    LocalRates,
    LocalV,
    FullPsth,
    FullSpectrum,
    FullHeatmap,
    FullIsi,
    FullPhase,
}

/// Описание плота в каталоге: имя, иконка, пояснение, род результата.
pub struct PlotDesc {
    pub kind: PlotKind,
    pub name: &'static str,
    pub icon: &'static str,
    pub desc: &'static str,
    /// `true` — плот локальной симуляции, `false` — полного прогона.
    pub local: bool,
}

/// Каталог всех плоттеров — «библиотека».
pub const CATALOG: [PlotDesc; 8] = [
    PlotDesc {
        kind: LocalRaster,
        name: "Растр спайков",
        icon: ic::PULSE,
        desc: "Спайки ego-контура: время × нейрон",
        local: true,
    },
    PlotDesc {
        kind: LocalRates,
        name: "Частоты (Гц)",
        icon: ic::CHART_BAR,
        desc: "Средняя частота по нейронам контура, цвет — нейромедиатор",
        local: true,
    },
    PlotDesc {
        kind: LocalV,
        name: "Мембранный потенциал",
        icon: ic::CHART_LINE,
        desc: "V(t) топ-5 нейронов + драйв (жирная линия)",
        local: true,
    },
    PlotDesc {
        kind: FullPsth,
        name: "PSTH",
        icon: ic::PULSE,
        desc: "Популяционная частота (Гц на нейрон) + сглаживание",
        local: false,
    },
    PlotDesc {
        kind: FullSpectrum,
        name: "Спектр мощности",
        icon: ic::ACTIVITY,
        desc: "БПФ с окном Ханна + доминирующая частота",
        local: false,
    },
    PlotDesc {
        kind: FullHeatmap,
        name: "Тепловая карта",
        icon: ic::PALETTE,
        desc: "Активность по классам: время (мс) × класс, яркость = Гц",
        local: false,
    },
    PlotDesc {
        kind: FullIsi,
        name: "ISI-гистограмма",
        icon: ic::TIMER,
        desc: "Распределение межспайковых интервалов по сети (медиана + хвост)",
        local: false,
    },
    PlotDesc {
        kind: FullPhase,
        name: "Фазы моторов",
        icon: ic::CLOCK,
        desc: "Фаза спайков моторных нейронов относительно ритма сети + антифаза r2",
        local: false,
    },
];

/// Состояние одного плота.
#[derive(Clone, Copy, Debug)]
pub struct PlotState {
    pub enabled: bool,
    pub detached: bool,
}

/// Состояние всей библиотеки (по одному на каждый `PlotKind`).
#[derive(Clone, Copy)]
pub struct PlotLib {
    pub states: [PlotState; 8],
}

impl PlotLib {
    pub fn new() -> Self {
        Self {
            states: [PlotState {
                enabled: false,
                detached: false,
            }; 8],
        }
    }

    pub fn get(&self, k: PlotKind) -> PlotState {
        self.states[k as usize]
    }

    pub fn set(&mut self, k: PlotKind, s: PlotState) {
        self.states[k as usize] = s;
    }

    /// Включает все плоты указанного рода, **если ни один ещё не включён** —
    /// чтобы после первого прогона панель не была пустой, но выбор
    /// пользователя не перетирался повторными запусками.
    pub fn enable_kind_if_none(&mut self, local: bool) {
        let any = CATALOG
            .iter()
            .any(|d| d.local == local && self.states[d.kind as usize].enabled);
        if !any {
            for d in CATALOG.iter().filter(|d| d.local == local) {
                self.states[d.kind as usize].enabled = true;
            }
        }
    }
}

impl Default for PlotLib {
    fn default() -> Self {
        Self::new()
    }
}
