//! # Плоты локальной симуляции (`plots::local`)
//!
//! Графики ego-контура (маленькая сеть вокруг выбранного нейрона):
//! растр спайков, частоты по нейронам, мембранные потенциалы.
//! Стили и карточка-обёртка — общие из `plots::mod` ([`super::plot_card`],
//! [`super::styled_plot`]).

use eframe::egui;

use egui_phosphor::regular as ic;

use egui_plot::{Bar, BarChart, Line, PlotPoints, Points};

use crate::core::color::ColorMode;
use crate::io::data::MancData;
use crate::sim::SimResult;

use super::{plot_card, styled_plot};

pub(super) fn card_local_raster(ui: &mut egui::Ui, r: &SimResult) {
    let k = r.idx.len();
    let mut spike_pts: Vec<[f64; 2]> = Vec::new();
    for step in 0..r.steps {
        let base = step * k;
        for i in 0..k {
            if r.raster[base + i] != 0 {
                spike_pts.push([step as f64 * r.dt_ms as f64, i as f64]);
            }
        }
    }
    plot_card(
        ui,
        ic::PULSE,
        "Растр спайков",
        &format!("{} нейронов · {} шагов", k, r.steps),
        |ui| {
            styled_plot(ui, "raster", "мс").show(ui, |p| {
                p.points(Points::new("спайки", PlotPoints::from(spike_pts)).radius(2.2_f32));
            });
        },
    );
}

pub(super) fn card_local_rates(ui: &mut egui::Ui, data: &MancData, r: &SimResult, mode: ColorMode) {
    let bars: Vec<Bar> = r
        .rates
        .iter()
        .enumerate()
        .map(|(i, &rate)| {
            let col = crate::core::color::color_for(data, r.idx[i], mode);
            Bar::new(i as f64, rate as f64)
                .horizontal()
                .width(0.6)
                .fill(col)
        })
        .collect();
    plot_card(
        ui,
        ic::CHART_BAR,
        "Частоты",
        "Гц по нейронам",
        |ui| {
            styled_plot(ui, "rates", "#")
                .allow_zoom(false)
                .allow_drag(false)
                .show(ui, |p| {
                    p.bar_chart(BarChart::new("частоты", bars));
                });
        },
    );
}

pub(super) fn card_local_v(ui: &mut egui::Ui, data: &MancData, r: &SimResult, mode: ColorMode) {
    let labels: Vec<String> = r
        .idx
        .iter()
        .map(|&g| {
            format!(
                "#{g} {}",
                if data.type_name(g).is_empty() {
                    "?"
                } else {
                    data.type_name(g)
                }
            )
        })
        .collect();
    let k = r.idx.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        r.rates[b]
            .partial_cmp(&r.rates[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut show: Vec<usize> = Vec::new();
    if let Some(d) = r.drive_local {
        show.push(d);
    }
    for i in order.into_iter().take(5) {
        if !show.contains(&i) {
            show.push(i);
        }
    }
    plot_card(
        ui,
        ic::CHART_LINE,
        "Мембранный потенциал",
        "V(t), мВ · топ-5 + драйв",
        |ui| {
            styled_plot(ui, "vtrace", "мс")
                .include_y(-90.0)
                .include_y(50.0)
                .show(ui, |p| {
                    for i in show {
                        let pts: Vec<[f64; 2]> = r
                            .vtraces
                            .iter()
                            .enumerate()
                            .map(|(c, v)| {
                                [
                                    (c as f64 + 1.0) * r.chunk as f64 * r.dt_ms as f64,
                                    v[i] as f64,
                                ]
                            })
                            .collect();
                        let col = crate::core::color::color_for(data, r.idx[i], mode);
                        let w = if Some(i) == r.drive_local { 2.6 } else { 1.4 };
                        p.line(
                            Line::new(labels[i].clone(), PlotPoints::from(pts))
                                .color(col)
                                .width(w),
                        );
                    }
                });
        },
    );
}
