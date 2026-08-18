//! # Плоты полной симуляции (`plots::full`)
//!
//! Научные графики полного прогона VNC: **PSTH** (популяционная частота,
//! Гц на нейрон), **спектр мощности** (БПФ + окно Ханна + доминирующая
//! частота), **тепловая карта** время × класс, **ISI-гистограмма** и
//! **фазы моторов** (детектор антифазы). Математика — в `plots::sci`,
//! стили и карточка-обёртка — общие из `plots::mod`.

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Rect};

use egui_phosphor::regular as ic;

use egui_plot::{Bar, BarChart, Line, PlotPoints, Points};

use crate::io::data::MancData;
use crate::sim::FullSimResult;

use super::sci::{antiphase_stats, hist_linear, isis_ms, power_spectrum};
use super::{plot_card, styled_plot, PLOT_HEIGHT};

/// Имена классов нейронов (индекс = код класса, см. `data::Klass`).
const CLASS_NAMES: [&str; 8] = [
    "other",
    "motor",
    "sensory",
    "ascending",
    "descending",
    "intrinsic",
    "glia",
    "efferent",
];

/// Цвета серий по классам — различимы и на тёмной, и на светлой теме.
const CLASS_COLORS: [Color32; 8] = [
    Color32::from_rgb(150, 150, 160),
    Color32::from_rgb(235, 120, 60),
    Color32::from_rgb(90, 210, 120),
    Color32::from_rgb(80, 150, 255),
    Color32::from_rgb(255, 170, 60),
    Color32::from_rgb(200, 110, 230),
    Color32::from_rgb(120, 210, 230),
    Color32::from_rgb(180, 220, 90),
];

/// PSTH: популяционная частота (Гц на нейрон) + сглаживание окном 9 шагов.
pub(super) fn card_full_psth(ui: &mut egui::Ui, r: &FullSimResult, n: f64, dt: f64) {
    let psth: Vec<[f64; 2]> = r
        .activity
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64 * dt, v as f64 / n / dt * 1000.0])
        .collect();
    let w = 9usize;
    let smoothed: Vec<[f64; 2]> = r
        .activity
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let lo = i.saturating_sub(w / 2);
            let hi = (i + w / 2 + 1).min(r.steps);
            let mut s = 0.0f64;
            for t in lo..hi {
                s += r.activity[t] as f64;
            }
            let k = (hi - lo).max(1) as f64;
            [i as f64 * dt, s / k / n / dt * 1000.0]
        })
        .collect();
    let activated_pct = r.activated.last().copied().unwrap_or(0.0) * 100.0;
    plot_card(
        ui,
        ic::PULSE,
        "PSTH — популяционная частота",
        &format!(
            "Гц (на нейрон) · драйв #{} · активировано {activated_pct:.0}%",
            r.drive
        ),
        |ui| {
            styled_plot(ui, "full_psth", "мс")
                .include_y(0.0)
                .show(ui, |p| {
                    p.line(
                        Line::new("PSTH", PlotPoints::from(psth))
                            .color(Color32::from_rgb(200, 200, 210))
                            .width(1.0_f32),
                    );
                    p.line(
                        Line::new("сглажено", PlotPoints::from(smoothed))
                            .color(Color32::from_rgb(245, 170, 60))
                            .width(2.2_f32),
                    );
                });
        },
    );
}

/// Спектр мощности (БПФ, окно Ханна) + доминирующая частота.
pub(super) fn card_full_spectrum(ui: &mut egui::Ui, r: &FullSimResult, dt: f64) {
    let (spec, dom_f, dom_p) = power_spectrum(&r.activity, 1000.0 / dt);
    let cap = if dom_p > 0.0 {
        format!("доминирующая частота: {dom_f:.1} Гц · БПФ с окном Ханна")
    } else {
        "спектр пуст — сигнала нет".to_string()
    };
    plot_card(
        ui,
        ic::ACTIVITY,
        "Спектр мощности",
        &cap,
        |ui| {
            styled_plot(ui, "full_spectrum", "Гц")
                .include_y(0.0)
                .show(ui, |p| {
                    if !spec.is_empty() {
                        p.line(
                            Line::new("мощность", PlotPoints::from(spec))
                                .color(Color32::from_rgb(120, 200, 255))
                                .width(1.2_f32),
                        );
                    }
                    if dom_p > 0.0 {
                        p.points(
                            Points::new("пик", vec![[dom_f, dom_p]])
                                .radius(3.5_f32)
                                .color(Color32::from_rgb(255, 90, 90)),
                        );
                    }
                });
        },
    );
}

/// Тепловая карта в единой карточке-обёртке (как остальные плоты).
pub(super) fn card_full_heatmap(ui: &mut egui::Ui, r: &FullSimResult, n: f64, dt: f64) {
    plot_card(
        ui,
        ic::PALETTE,
        "Тепловая карта",
        "активность по классам: время × класс, яркость = Гц",
        |ui| heatmap_classes(ui, r, n, dt),
    );
}

/// Тепловая карта: время (X) × класс (Y), яркость = частота в Гц.
fn heatmap_classes(ui: &mut egui::Ui, r: &FullSimResult, n: f64, dt: f64) {
    let (resp, painter) = ui.allocate_painter(
        vec2(ui.available_width(), PLOT_HEIGHT),
        egui::Sense::hover(),
    );
    let rect = resp.rect;
    if r.steps == 0 || rect.width() < 20.0 {
        return;
    }
    // Нормировка на максимальную частоту по всем классам (Гц на нейрон).
    let mut max_hz = 0.0f64;
    for ca in &r.class_activity {
        for &v in ca.iter().take(8) {
            max_hz = max_hz.max(v as f64 / n / dt * 1000.0);
        }
    }
    if max_hz <= 0.0 {
        max_hz = 1.0;
    }
    // Биним по времени до 512 колонок (шагов может быть до 1000).
    let cols_n = r.steps.min(512);
    let bin = (r.steps as f64 / cols_n as f64).ceil().max(1.0) as usize;
    let left_lbl = 56.0f32; // место под имена классов слева
    let area = Rect::from_min_max(
        pos2(rect.left() + left_lbl, rect.top()),
        pos2(rect.right(), rect.bottom()),
    );
    let cw = area.width() / cols_n as f32;
    let rh = area.height() / 8.0;
    for k in 0..8 {
        let y0 = area.top() + rh * k as f32;
        let y1 = y0 + rh;
        for c in 0..cols_n {
            let mut s = 0.0f32;
            let lo = c * bin;
            let hi = ((c + 1) * bin).min(r.steps);
            for t in lo..hi {
                s += r.class_activity[t][k];
            }
            let hz = s as f64 / bin as f64 / n / dt * 1000.0;
            let v = (hz / max_hz).clamp(0.0, 1.0) as f32;
            let col = heat_color(v);
            let x0 = area.left() + cw * c as f32;
            let x1 = x0 + cw;
            painter.rect_filled(Rect::from_min_max(pos2(x0, y0), pos2(x1, y1)), 0.0, col);
        }
        painter.text(
            pos2(area.left() - 4.0, y0 + rh * 0.5),
            Align2::RIGHT_CENTER,
            CLASS_NAMES[k],
            FontId::monospace(9.0),
            CLASS_COLORS[k],
        );
    }
    painter.rect_stroke(
        area,
        1.0,
        egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 120, 150, 90)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        pos2(area.left(), area.bottom() + 2.0),
        Align2::LEFT_TOP,
        format!("{:.1} мс → {:.1} мс", 0.0, r.steps as f64 * dt),
        FontId::monospace(9.0),
        Color32::from_rgba_unmultiplied(160, 160, 180, 200),
    );
}

/// ISI-гистограмма: распределение межспайковых интервалов по всей сети.
/// Ось X в мс (до 99-го процентиля, чтобы хвост выбросов не сплющил шкалу).
pub(super) fn card_full_isi(ui: &mut egui::Ui, r: &FullSimResult) {
    let isis = isis_ms(&r.spike_times, r.dt_ms);
    let n_with = r.spike_times.iter().filter(|tr| tr.len() >= 2).count();
    if isis.is_empty() {
        plot_card(
            ui,
            ic::TIMER,
            "ISI-гистограмма",
            "нет нейронов с ≥2 спайками",
            |_| {},
        );
        return;
    }
    // 99-й процентиль — граница шкалы X
    let mut sorted = isis.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cap = sorted[(sorted.len() as f64 * 0.99) as usize].max(1.0) as f64;
    let median = sorted[sorted.len() / 2];
    let bins = 28usize;
    let counts = hist_linear(&isis, bins, cap as f32);
    let n_spikes: usize = r.spike_times.iter().map(|t| t.len()).sum();
    let bars: Vec<Bar> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let x = cap * (i as f64 + 0.5) / bins as f64;
            Bar::new(x, c)
                .width(cap / bins as f64 * 0.9)
                .fill(Color32::from_rgb(130, 170, 255))
        })
        .collect();
    plot_card(
        ui,
        ic::TIMER,
        "ISI-гистограмма",
        &format!("медиана {median:.1} мс · {n_with} нейронов · {n_spikes} спайков"),
        |ui| {
            styled_plot(ui, "full_isi", "мс")
                .allow_zoom(false)
                .allow_drag(false)
                .include_y(0.0)
                .show(ui, |p| {
                    p.bar_chart(BarChart::new("ISI", bars));
                });
        },
    );
}

/// Фазовая гистограмма моторных нейронов: фаза каждого спайка мотора
/// относительно ритма сети (период = 1/доминирующая частота из БПФ).
/// Если моторы спайкают в антифазе (half-center CPG) — гистограмма
/// двугорбая, а r2 (второй круговой момент) ≈ 1.
pub(super) fn card_full_phase(ui: &mut egui::Ui, data: &MancData, r: &FullSimResult) {
    let dt = r.dt_ms as f64;
    let (_, dom_f, dom_p) = power_spectrum(&r.activity, 1000.0 / dt);
    if dom_p <= 0.0 {
        plot_card(
            ui,
            ic::CLOCK,
            "Фазы моторов",
            "нет ритма сети — спектр пуст",
            |_| {},
        );
        return;
    }
    let period_ms = 1000.0 / dom_f;
    // фазы спайков всех моторных нейронов (класс Motor = 1)
    let mut phases: Vec<f64> = Vec::new();
    let mut motors = 0usize;
    for (i, tr) in r.spike_times.iter().enumerate() {
        if i >= data.n || data.klass[i] as usize != 1 {
            continue;
        }
        motors += 1;
        for &t in tr {
            let t_ms = t as f64 * dt;
            phases.push((t_ms / period_ms).fract());
        }
    }
    if phases.is_empty() {
        plot_card(
            ui,
            ic::CLOCK,
            "Фазы моторов",
            "моторные нейроны не спайкали",
            |_| {},
        );
        return;
    }
    let (r2, axis) = antiphase_stats(&phases);
    let bins = 24usize;
    let counts = hist_linear(
        &phases.iter().map(|&p| p as f32).collect::<Vec<f32>>(),
        bins,
        1.0,
    );
    let max_c = counts.iter().cloned().fold(0.0f64, f64::max).max(1.0);
    let frac: Vec<Bar> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let x = (i as f64 + 0.5) / bins as f64;
            Bar::new(x, c / max_c)
                .width(1.0 / bins as f64 * 0.9)
                .fill(Color32::from_rgb(255, 150, 90))
        })
        .collect();
    let verdict = if r2 > 0.5 {
        "есть антифаза"
    } else if r2 > 0.25 {
        "слабая модуляция"
    } else {
        "фазы равномерны"
    };
    plot_card(
        ui,
        ic::CLOCK,
        "Фазы моторов",
        &format!("ритм {dom_f:.1} Гц · {motors} моторов · r2={r2:.2} (ось {axis:.2}) · {verdict}"),
        |ui| {
            styled_plot(ui, "full_phase", "цикл")
                .allow_zoom(false)
                .allow_drag(false)
                .include_y(0.0)
                .show(ui, |p| {
                    p.bar_chart(BarChart::new("моторы", frac));
                });
        },
    );
}

/// Тепловая шкала: 0 → тёмно-синий фон, 0.5 → циан, 1 → жёлто-белый.
fn heat_color(v: f32) -> Color32 {
    let v = v.clamp(0.0, 1.0);
    let (r, g, b) = if v < 0.5 {
        let t = v / 0.5; // 0 → 1 (тёмно-синий → циан)
        (10.0 + 40.0 * t, 30.0 + 140.0 * t, 60.0 + 150.0 * t)
    } else {
        let t = (v - 0.5) / 0.5; // 0 → 1 (циан → жёлтый)
        (50.0 + 205.0 * t, 170.0 + 80.0 * t, 210.0 * (1.0 - t))
    };
    Color32::from_rgb(r as u8, g as u8, b as u8)
}
