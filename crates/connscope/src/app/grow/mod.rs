//! # Режим «Развитие» (`app::grow`)
//!
//! Живая демонстрация обоих движков в одном конвейере: кнопка «Вырастить»
//! растит E/I микросхему из правил развития ([`prometheus_grow`] — диффузия
//! морфогена, конусы роста, контактное замыкание синапсов), а «Симулировать»
//! гоняет её на SIMD-движке pillars (LIF-α). Поверхности:
//!
//! * **«Развитие»** — 3D-канвас выросшей схемы (сомы по E/I, нейриты,
//!   синапсы; вращение мышью, зум колесом) + крутилки конфигурации;
//! * **«Анализ гамма»** — популяционная частота, спектр (БПФ) и
//!   пространственная синхронизация (корреляция спайков vs. расстояние)
//!   в сравнении с контролем тех же степеней (shuffled).
//!
//! Весь расчёт — общий модуль [`pillars_connectome::grow_engine`] (тот же
//! код, что у CLI-примера `genome_to_gamma`); здесь только UI. Рост +
//! симуляция (~0.7 с) крутятся в фоновом потоке, чтобы кадр не замирал.
//!
//! Режим подключён как «кросс-режим» (как NeuromlMode): поверхности живут
//! в общем дереве доков, оболочка зовёт [`GrowMode::grow_dock_view`] /
//! [`GrowMode::plots_dock_view`] из `TabViewer`.

use std::sync::mpsc;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};
use pillars_connectome::grow_engine::{run_experiment, spectrum, GrowConfig, GrowReport};

/// Цвета типов клеток (совпадают с палитрой коннектома: ACh/GABA).
const E_COLOR: Color32 = Color32::from_rgb(74, 222, 128);
const I_COLOR: Color32 = Color32::from_rgb(248, 113, 113);
const SHUF_COLOR: Color32 = Color32::from_rgb(148, 163, 184);

pub(crate) struct GrowMode {
    cfg: GrowConfig,
    report: Option<GrowReport>,
    /// Фоновый расчёт: `Some(rx)` = считаем.
    worker: Option<mpsc::Receiver<Result<GrowReport, String>>>,
    error: Option<String>,
    // Камера 3D-канваса.
    yaw: f32,
    pitch: f32,
    dist: f32,
    show_edges: bool,
    show_shuffled: bool,
}

impl Default for GrowMode {
    fn default() -> Self {
        Self {
            cfg: GrowConfig::default(),
            report: None,
            worker: None,
            error: None,
            yaw: 0.6,
            pitch: 0.35,
            dist: 260.0,
            show_edges: true,
            show_shuffled: false,
        }
    }
}

impl GrowMode {
    /// Запустить рост + симуляцию в фоновом потоке.
    fn start_run(&mut self) {
        let cfg = self.cfg.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = std::panic::catch_unwind(|| run_experiment(&cfg))
                .map_err(|_| "паника в движке роста".to_string());
            let _ = tx.send(res);
        });
        self.worker = Some(rx);
        self.error = None;
    }

    /// Покадровый опрос фонового потока (зовёт оболочка из `ConnApp`).
    pub(crate) fn frame_tick(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.worker {
            match rx.try_recv() {
                Ok(Ok(r)) => {
                    self.report = Some(r);
                    self.worker = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(e);
                    self.worker = None;
                }
                Err(_) => ctx.request_repaint(),
            }
        }
    }

    // ── Поверхность «Развитие» ─────────────────────────────────────────────
    pub(crate) fn grow_dock_view(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🌱 Вырастить + Симулировать")
                .on_hover_text("200 нейронов: диффузия → рост → синапсы → SIMD LIF-α (≈0.7 с)")
                .clicked()
            {
                self.start_run();
            }
            ui.checkbox(&mut self.show_edges, "синапсы");
            ui.checkbox(&mut self.show_shuffled, "контроль (shuffled)");
            if self.worker.is_some() {
                ui.spinner();
                ui.weak("расту и считаю…");
            }
        });
        if let Some(e) = &self.error {
            ui.colored_label(Color32::RED, format!("ошибка: {e}"));
        }

        ui.add_space(4.0);
        egui::Grid::new("grow_cfg")
            .num_columns(4)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.label("драйв");
                ui.add(
                    egui::Slider::new(&mut self.cfg.drive, 0.05..=0.35)
                        .logarithmic(true)
                        .show_value(true),
                );
                ui.label("E-веса");
                ui.add(
                    egui::Slider::new(&mut self.cfg.w_e_gain, 0.05..=0.4)
                        .logarithmic(true)
                        .show_value(true),
                );
                ui.end_row();
                ui.label("I-веса");
                ui.add(
                    egui::Slider::new(&mut self.cfg.w_i_gain, 0.1..=1.5)
                        .logarithmic(true)
                        .show_value(true),
                );
                ui.label("τ_syn, мс");
                ui.add(egui::Slider::new(&mut self.cfg.tau_syn, 1.0..=8.0).show_value(true));
                ui.end_row();
            });

        ui.separator();

        let report = self.report.take();
        match &report {
            None => {
                ui.add_space(12.0);
                ui.weak(
                    "Нажми «Вырастить + Симулировать» — вырастет E/I микросхема\n\
                     (200 нейронов) и прогонится на SIMD-движке pillars.",
                );
            }
            Some(r) => {
                let n = self.cfg.n();
                ui.weak(format!(
                    "{} нейронов ({}E/{}I) · {} синапсов · рост {:.2} с · сим {:.0} М клетко-шагов/с · \
                     C = {:.2}",
                    n,
                    self.cfg.n_e,
                    self.cfg.n_i,
                    r.circuit.edges.len(),
                    r.circuit.growth_s,
                    n as f64 * self.cfg.sim_steps as f64 / r.sim_s / 1e6,
                    r.cc_grown,
                ));
                self.canvas(ui, r);
            }
        }
        self.report = report;
    }

    /// 3D-канвас: сомы (цвет по E/I), нейриты-ломаные, синапсы.
    fn canvas(&mut self, ui: &mut egui::Ui, r: &GrowReport) {
        let avail = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(avail, Sense::hover());
        let resp = ui.interact(rect, ui.id().with("grow_canvas"), Sense::drag());
        if resp.dragged() {
            let d = resp.drag_delta();
            self.yaw += d.x * 0.008;
            self.pitch = (self.pitch + d.y * 0.008).clamp(-1.4, 1.4);
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            self.dist = (self.dist * (1.0 - scroll * 0.001)).clamp(80.0, 900.0);
        }

        let p = ui.painter();
        p.rect_filled(rect, 6.0, Color32::from_gray(14));

        // Центр масс — точка вращения камеры.
        let pos = &r.circuit.positions;
        let n = pos.len().max(1);
        let center = [
            pos.iter().map(|q| q[0]).sum::<f64>() / n as f64,
            pos.iter().map(|q| q[1]).sum::<f64>() / n as f64,
            pos.iter().map(|q| q[2]).sum::<f64>() / n as f64,
        ];

        // Синапсы (рёбра) — тонкие тёмные линии.
        if self.show_edges {
            for &(a, b, _) in &r.circuit.edges {
                let pa = self.project(pos[a], center, rect);
                let pb = self.project(pos[b], center, rect);
                p.line_segment([pa, pb], Stroke::new(0.5, Color32::from_gray(55)));
            }
            if self.show_shuffled {
                for &(a, b, _) in &r.shuffled {
                    let pa = self.project(pos[a], center, rect);
                    let pb = self.project(pos[b], center, rect);
                    p.line_segment(
                        [pa, pb],
                        Stroke::new(0.5, Color32::from_rgba_unmultiplied(crate::app::shell::theme::stroke_soft(ui.ctx()).r(), crate::app::shell::theme::stroke_soft(ui.ctx()).g(), crate::app::shell::theme::stroke_soft(ui.ctx()).b(), 50)),
                    );
                }
            }
        }

        // Нейриты.
        for (ci, cell) in r.circuit.neurites.iter().enumerate() {
            let col = if r.circuit.nt[ci] > 0.0 {
                E_COLOR.gamma_multiply(0.55)
            } else {
                I_COLOR.gamma_multiply(0.55)
            };
            for nb in cell {
                let pts: Vec<Pos2> = nb.iter().map(|q| self.project(*q, center, rect)).collect();
                for w in pts.windows(2) {
                    p.line_segment([w[0], w[1]], Stroke::new(1.1, col));
                }
            }
        }

        // Сомы.
        for (i, q) in pos.iter().enumerate() {
            let col = if r.circuit.nt[i] > 0.0 {
                Color32::from_rgb(134, 239, 172)
            } else {
                Color32::from_rgb(252, 165, 165)
            };
            let pt = self.project(*q, center, rect);
            p.circle_filled(pt, 2.6, col);
            p.circle_stroke(pt, 2.6, Stroke::new(0.6, Color32::from_black_alpha(140)));
        }

        p.text(
            rect.left_top() + Vec2::new(8.0, 8.0),
            Align2::LEFT_TOP,
            "drag = вращение · колесо = зум · зелёный E / красный I",
            FontId::proportional(12.0),
            Color32::from_gray(150),
        );
    }

    /// Перспективная проекция точки мира в экранные координаты.
    fn project(&self, p: [f64; 3], center: [f64; 3], rect: Rect) -> Pos2 {
        let (dx, dy, dz) = (p[0] - center[0], p[1] - center[1], p[2] - center[2]);
        let (cy, sy) = (self.yaw.cos() as f64, self.yaw.sin() as f64);
        let x1 = dx * cy - dz * sy;
        let z1 = dx * sy + dz * cy;
        let (cp, sp) = (self.pitch.cos() as f64, self.pitch.sin() as f64);
        let y2 = dy * cp - z1 * sp;
        let z2 = dy * sp + z1 * cp;
        let focal = rect.width().min(rect.height()) as f64 * 1.5;
        let s = focal / (self.dist as f64 + z2).max(60.0);
        Pos2::new(
            rect.center().x + (x1 * s) as f32,
            rect.center().y - (y2 * s) as f32,
        )
    }

    // ── Поверхность «Анализ гамма» ──────────────────────────────────────────
    pub(crate) fn plots_dock_view(&mut self, ui: &mut egui::Ui) {
        let Some(r) = &self.report else {
            ui.add_space(12.0);
            ui.weak("Сначала «Вырастить + Симулировать» на вкладке «Развитие».");
            return;
        };

        // Сводка-вердикт.
        let (d0, c0) = r.scg[0];
        let (d1, c1) = *r.scg.last().unwrap();
        let (_, s0) = r.scs[0];
        let (_, s1) = *r.scs.last().unwrap();
        let decay_g = c0 - c1;
        let decay_s = s0 - s1;
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("C(grown) = {:.2} vs C(shuffled) = {:.2}", r.cc_grown, r.cc_shuf));
            ui.separator();
            ui.label(format!(
                "ко-файринг {d0:.0}µm: r={c0:.3} (×{:.1}) → {d1:.0}µm: r={c1:.3}",
                c0 / s0.max(1e-6)
            ));
            ui.separator();
            ui.label(format!("ритм: {:.1} Гц · пик {:.1}× фон · осц. {:.2}", r.fg, r.cg, r.og));
        });
        ui.separator();
        if decay_g > decay_s + 0.03 && c0 > s0 {
            ui.colored_label(
                E_COLOR,
                format!("✅ геометрия важна: градиент ко-файринга {decay_g:.2} против {decay_s:.2} у контроля"),
            );
        }

        let bin_ms: f32 = 4.0;
        // 1. Популяционная частота.
        let t_g: Vec<[f64; 2]> = r
            .pop_grown
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64 * bin_ms as f64, *v as f64])
            .collect();
        let t_s: Vec<[f64; 2]> = r
            .pop_shuf
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64 * bin_ms as f64, *v as f64])
            .collect();
        ui.label("Популяционная частота, Гц");
        Plot::new("grow_pop")
            .height(130.0)
            .legend(Legend::default())
            .show(ui, |p| {
                p.line(
                    Line::new("grown", PlotPoints::from(t_g))
                        .color(E_COLOR)
                        .width(1.4),
                );
                p.line(
                    Line::new("shuffled", PlotPoints::from(t_s))
                        .color(SHUF_COLOR)
                        .width(1.2),
                );
            });

        // 2. Спектр (нормирован на пик).
        let (freqs, powers) = spectrum(&r.pop_grown, bin_ms);
        let maxp = powers.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
        let spec: Vec<[f64; 2]> = freqs
            .iter()
            .zip(powers.iter())
            .map(|(f, p)| [*f, p / maxp])
            .collect();
        ui.label("Спектр популяционной частоты (норм.)");
        Plot::new("grow_spec").height(130.0).show(ui, |p| {
            p.line(Line::new("спектр", PlotPoints::from(spec)).color(Color32::from_rgb(250, 204, 21)));
        });

        // 3. Пространственная синхронизация.
        let bars_g: Vec<Bar> = r.scg.iter().map(|(d, c)| Bar::new(*d, *c).width(6.0)).collect();
        let bars_s: Vec<Bar> = r.scs.iter().map(|(d, c)| Bar::new(*d, *c).width(6.0)).collect();
        ui.label("Спайк-корреляция vs. расстояние между сомами (50 мс бины)");
        Plot::new("grow_spatial")
            .height(130.0)
            .legend(Legend::default())
            .show(ui, |p| {
                p.bar_chart(BarChart::new("grown", bars_g).color(E_COLOR));
                p.bar_chart(BarChart::new("shuffled", bars_s).color(SHUF_COLOR));
            });
    }
}
