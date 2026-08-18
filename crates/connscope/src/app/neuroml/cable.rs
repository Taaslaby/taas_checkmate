//! # Морфология кабеля (`app::neuroml::cable`)
//!
//! Виджет «Морфология» — мост NeuroML-кабеля (pillars) в GUI: рисует дерево
//! компартментов (ось x = путь от сомы, y = ширина поддерева), позволяет
//! кликнуть по компартменту, прогнать сеть с записью **покомпартментных
//! трасс** ([`run_traced`]) и посмотреть V(t) выбранного компартмента.
//! Слайдер времени окрашивает дерево по V(t) в выбранный момент.
//!
//! Данные приходят из [`CableInfo`] (геометрия) + [`Network::cable_voltage`]
//! (покомпартментный V) — см. мост в `crates/pillars/src/api/neuroml`.

use eframe::egui::{self, vec2, Color32, Pos2, Rect, Sense, Stroke, Vec2};

use egui_phosphor::regular as ic;

use egui_plot::{Line, Plot, PlotPoints, VLine};

use pillars::api::neuroml::CableInfo;

use super::{CableState, NeuromlMode};
use crate::app::shell::theme;
use crate::app::tokens;
use crate::app::widgets::{action_button, ActionTone};
use crate::sim::neuroml::{run_traced, CableTraces};

impl CableState {
    /// Build the widget state: geometry + a fresh band-layout (y-centres).
    pub(crate) fn new(pop: usize, info: CableInfo) -> Self {
        let layout_y = compute_layout(&info);
        Self {
            pop,
            info,
            traces: None,
            selected: Some(0), // по умолчанию — сома
            t_sample: 0,
            color_by_time: false,
            layout_y,
            zoom: 0.0, // fit при первом кадре
            pan: Vec2::ZERO,
        }
    }
}

impl NeuromlMode {
    /// Вкладка «Морфология» в дереве доков (рисует оболочка): дерево + трассы.
    pub(crate) fn cable_dock_view(&mut self, ui: &mut egui::Ui) {
        if self.cable.is_none() {
            ui.weak(
                "Морфология недоступна — загрузи NeuroML с кабельной клеткой (пример «HH-аксон»).",
            );
            return;
        }
        tokens::spacing(ui);
        let mut want_run = false;
        if let Some(state) = self.cable.as_mut() {
            want_run = cable_view(state, ui);
        }
        if want_run {
            self.cable_run();
        }
    }

    /// Прогнать сеть и записать покомпартментные трассы кабельной популяции.
    fn cable_run(&mut self) {
        let Some(pid) = self.cable.as_ref().map(|c| c.pop) else {
            return;
        };
        let result = match self.loaded.as_mut() {
            Some(ln) => run_traced(ln, pid, 2000),
            None => Err("нет загруженной сети".into()),
        };
        match result {
            Ok(tr) => {
                if let Some(state) = self.cable.as_mut() {
                    state.traces = Some(tr);
                    state.t_sample = 0;
                }
                self.status = format!("✓ записаны трассы {pid} компартментов");
            }
            Err(e) => self.status = format!("⚠ {e}"),
        }
    }
}

/// Рисует виджет; возвращает `true`, если нажата кнопка «Запустить и записать
/// трассы» (запуск обрабатывается снаружи, где доступен `loaded`).
fn cable_view(state: &mut CableState, ui: &mut egui::Ui) -> bool {
    let mut want_run = false;
    let n = state.info.compartments.len();
    let max_path = state
        .info
        .compartments
        .iter()
        .map(|c| c.path_um)
        .fold(0.0f32, f32::max);

    // ── верхняя панель: запуск, режим окраски, слайдер времени, статистика ──
    ui.horizontal_wrapped(|ui| {
        if action_button(
            ui,
            ic::PLAY,
            "Запустить и записать трассы",
            ActionTone::Primary,
            250.0,
        )
        .on_hover_text("Прогнать сеть и записать V(t) каждого компартмента")
        .clicked()
        {
            want_run = true;
        }
        let has_traces = state.traces.is_some();
        ui.add_enabled_ui(has_traces, |ui| {
            ui.checkbox(&mut state.color_by_time, "Окраска по времени")
                .on_hover_text("Красить дерево по V(t) на слайдере, а не по конечному V");
        });
        if let Some(tr) = &state.traces {
            let last = tr.t_ms.len().saturating_sub(1);
            let t0 = state.t_sample.min(last);
            let label = format!("t = {:.2} мс", tr.t_ms[t0]);
            ui.add(egui::Slider::new(&mut state.t_sample, 0..=last).text(label));
        }
        ui.separator();
        ui.label(
            egui::RichText::new(format!(
                "{n} компартментов · {} сегментов · {max_path:.0} мкм",
                state.info.segment_names.len()
            ))
            .strong(),
        );
        ui.label(egui::RichText::new("колесо — масштаб · drag — панорама · клик — выбрать").weak());
    });
    ui.separator();

    // ── дерево компартментов ──
    let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
    let rect = resp.rect;

    // Fit-масштаб при первом кадре (zoom == 0).
    if state.zoom <= 0.0 {
        let y_min = state.layout_y.iter().cloned().fold(f32::MAX, f32::min);
        let y_max = state.layout_y.iter().cloned().fold(f32::MIN, f32::max);
        let y_span = (y_max - y_min).max(1.0);
        let sx = (rect.width() - 48.0).max(20.0) / max_path.max(1.0);
        let sy = (rect.height() - 48.0).max(20.0) / y_span;
        state.zoom = sx.min(sy) * 0.92;
        state.pan = Vec2::ZERO;
    }
    // Зум колесом вокруг курсора.
    if let Some(hover) = resp.hover_pos() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let factor = (1.0015f32).powf(scroll);
            let anchor = screen_to_world(state, rect, hover);
            state.zoom = (state.zoom * factor).clamp(1e-4, 1e6);
            let after = screen_to_world(state, rect, hover);
            state.pan += vec2(
                (anchor.0 - after.0) * state.zoom,
                (anchor.1 - after.1) * state.zoom,
            );
        }
    }
    // Панорама drag'ом.
    if resp.dragged() {
        state.pan += resp.drag_delta();
    }

    // Мир → экран.
    let y_min = state.layout_y.iter().cloned().fold(f32::MAX, f32::min);
    let to_screen = |state: &CableState, rect: Rect, (xw, yw): (f32, f32)| -> Pos2 {
        Pos2::new(
            rect.left() + 24.0 + xw * state.zoom + state.pan.x,
            rect.bottom() - 24.0 - (yw - y_min) * state.zoom + state.pan.y,
        )
    };

    // Hover/клик → ближайший компартмент.
    let mut hovered = None;
    if let Some(pos) = resp.hover_pos() {
        hovered = nearest_comp(state, to_screen(state, rect, (0.0, 0.0)), pos);
    }
    if resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            state.selected = nearest_comp(state, to_screen(state, rect, (0.0, 0.0)), pos);
        }
    }

    // Рисуем линии родитель → ребёнок.
    let comps = &state.info.compartments;
    for (i, c) in comps.iter().enumerate() {
        if i == state.info.root {
            continue;
        }
        let p_parent = to_screen(
            state,
            rect,
            (
                comps[c.parent.unwrap()].path_um,
                state.layout_y[c.parent.unwrap()],
            ),
        );
        let p_self = to_screen(state, rect, (c.path_um, state.layout_y[i]));
        let width = (c.diameter_um * state.zoom).clamp(0.5, 18.0);
        let color = comp_color(state, i).unwrap_or(Color32::from_gray(140));
        painter.line_segment([p_parent, p_self], Stroke::new(width, color));
    }
    // Сома — кружок.
    let r = state.info.root;
    painter.circle_filled(
        to_screen(state, rect, (comps[r].path_um, state.layout_y[r])),
        (comps[r].diameter_um * state.zoom).clamp(3.0, 14.0),
        comp_color(state, r).unwrap_or(Color32::from_gray(140)),
    );
    // Выбранный компартмент — белая обводка.
    if let Some(s) = state.selected {
        let p = to_screen(state, rect, (comps[s].path_um, state.layout_y[s]));
        painter.circle_stroke(p, 6.0, Stroke::new(1.6, Color32::WHITE));
    }
    // Hover — подсветка + тултип.
    if let Some(h) = hovered {
        let p = to_screen(state, rect, (comps[h].path_um, state.layout_y[h]));
        painter.circle_stroke(p, 7.0, Stroke::new(1.2, theme::accent(ui)));
        let name = segment_name(&state.info, comps[h].segment_id);
        let v = comp_color(state, h)
            .map(|_| format!(" · V = {:.2} мВ", comp_v(state, h).unwrap_or(0.0)))
            .unwrap_or_default();
        resp.clone().on_hover_text(format!(
            "компартмент #{h}{}\nсегмент: {name}\nL = {:.1} мкм · d = {:.1} мкм · путь = {:.1} мкм{v}",
            if h == state.info.root { " (сома)" } else { "" },
            comps[h].length_um,
            comps[h].diameter_um,
            comps[h].path_um,
        ));
    }

    if let Some(hover) = resp.hover_pos() {
        let (xw, _) = screen_to_world(state, rect, hover);
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        ui.label(format!("путь: {xw:.1} мкм"));
    }

    ui.separator();

    // ── V(t) выбранного компартмента ──
    if let Some(tr) = &state.traces {
        cable_plot(ui, state, tr);
    } else {
        ui.weak(
            "Трассы ещё не записаны — нажми «Запустить и записать трассы», чтобы увидеть V(t) по компартментам.",
        );
    }

    want_run
}

/// V(t)-плот выбранного компартмента (сома по умолчанию) + вертикальная
/// линия на позиции слайдера времени.
/// Цвет трассы V(t) (данные; как синие трассы в плоттерах коннектома).
const TRACE_COLOR: Color32 = Color32::from_rgb(90, 160, 240);

fn cable_plot(ui: &mut egui::Ui, state: &CableState, tr: &CableTraces) {
    let comp = state.selected.unwrap_or(state.info.root);
    let n = tr.t_ms.len().min(tr.v[comp].len());
    let pts: Vec<[f64; 2]> = (0..n)
        .map(|k| [tr.t_ms[k] as f64, tr.v[comp][k] as f64])
        .collect();
    let name = segment_name(&state.info, state.info.compartments[comp].segment_id);
    // Акцент читаем до `Plot::show`: замыкание `|p|` захватывает `ui` неизменяемо,
    // а `show` требует `&mut ui` — вычисление заранее снимает конфликт заимствования.
    let accent = theme::accent(ui);
    Plot::new("cable_v")
        .height(170.0)
        .legend(egui_plot::Legend::default())
        .show(ui, |p| {
            p.line(
                Line::new(format!("V(t) #{comp} ({name})"), PlotPoints::from(pts))
                    .color(TRACE_COLOR)
                    .width(1.8),
            );
            if state.color_by_time {
                let k = state.t_sample.min(n.saturating_sub(1));
                p.vline(VLine::new("t", tr.t_ms[k] as f64).color(accent));
            }
        });
}

/// V компартмента для окраски: из трасс (слайдер или финал) либо `None`.
fn comp_v(state: &CableState, comp: usize) -> Option<f32> {
    let tr = state.traces.as_ref()?;
    let trace = tr.v.get(comp)?;
    if state.color_by_time {
        let k = state.t_sample.min(trace.len().saturating_sub(1));
        trace.get(k).copied()
    } else {
        trace.last().copied()
    }
}

/// Цвет компартмента по V (синий покой → красный спайк); серый без трасс.
fn comp_color(state: &CableState, comp: usize) -> Option<Color32> {
    comp_v(state, comp).map(v_to_color)
}

/// Линейная интерполяция цветовой шкалы V: -80 синий … +40 красный.
/// Это «данные» (тепловая карта на морфологии, как Degree в коннектоме),
/// а не хром оболочки — радуга здесь легитимна.
fn v_to_color(v: f32) -> Color32 {
    // (V, r, g, b)
    const STOPS: [(f32, u8, u8, u8); 6] = [
        (-80.0, 45, 70, 190),
        (-65.0, 70, 130, 210),
        (-50.0, 70, 190, 120),
        (-20.0, 220, 200, 70),
        (0.0, 235, 120, 45),
        (40.0, 205, 45, 45),
    ];
    let v = v.clamp(-80.0, 40.0);
    for w in STOPS.windows(2) {
        let (v0, r0, g0, b0) = w[0];
        let (v1, r1, g1, b1) = w[1];
        if v <= v1 {
            let t = ((v - v0) / (v1 - v0)).clamp(0.0, 1.0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
            return Color32::from_rgb(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1));
        }
    }
    Color32::from_rgb(205, 45, 45)
}

/// Имя сегмента по его id (или `#id`, если не назван).
fn segment_name(info: &CableInfo, segment_id: usize) -> String {
    info.segment_names
        .iter()
        .find(|(id, _)| *id == segment_id)
        .map(|(_, n)| n.clone())
        .unwrap_or_else(|| format!("#{segment_id}"))
}

/// Ближайший компартмент к экранной точке в радиусе 12 px.
fn nearest_comp(state: &CableState, origin: Pos2, pos: Pos2) -> Option<usize> {
    let zoom = state.zoom.max(1e-4);
    let mut best = None;
    let mut best_d = 12.0 * 12.0;
    for (i, c) in state.info.compartments.iter().enumerate() {
        let p = Pos2::new(
            origin.x + c.path_um * zoom,
            origin.y - (state.layout_y[i] - layout_min(state)) * zoom,
        );
        let d = pos.distance_sq(p);
        if d < best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

fn layout_min(state: &CableState) -> f32 {
    state.layout_y.iter().cloned().fold(f32::MAX, f32::min)
}

/// Экранная точка → мировые координаты (path, band-y).
fn screen_to_world(state: &CableState, rect: Rect, pos: Pos2) -> (f32, f32) {
    let xw = (pos.x - rect.left() - 24.0 - state.pan.x) / state.zoom.max(1e-4);
    let yw =
        layout_min(state) + (rect.bottom() - 24.0 - pos.y + state.pan.y) / state.zoom.max(1e-4);
    (xw, yw)
}

/// Band-layout: y-центр полосы каждого компартмента. Ширина поддерева =
/// max(диаметр, Σ ширин детей); узлы делят родительскую полосу пропорционально
/// ширинам поддеревьев. Цепочки вырождаются в горизонтальную линию, ветвления
/// разводятся по вертикали.
fn compute_layout(info: &CableInfo) -> Vec<f32> {
    let n = info.compartments.len();
    let mut children = vec![Vec::<usize>::new(); n];
    for (i, c) in info.compartments.iter().enumerate() {
        if let Some(p) = c.parent {
            children[p].push(i);
        }
    }
    let mut width = vec![0.0f32; n];
    subtree_width(info, &children, info.root, &mut width);
    let mut out = vec![0.0f32; n];
    let total = width[info.root].max(1.0);
    assign_y(&children, &width, info.root, 0.0, total, &mut out);
    out
}

fn subtree_width(info: &CableInfo, children: &[Vec<usize>], i: usize, width: &mut [f32]) -> f32 {
    let kids_sum: f32 = children[i]
        .iter()
        .map(|&c| subtree_width(info, children, c, width))
        .sum();
    let d = info.compartments[i].diameter_um;
    width[i] = if children[i].is_empty() {
        d
    } else {
        d.max(kids_sum)
    };
    width[i]
}

fn assign_y(children: &[Vec<usize>], width: &[f32], i: usize, y0: f32, y1: f32, out: &mut [f32]) {
    out[i] = (y0 + y1) * 0.5;
    let kids = &children[i];
    if kids.is_empty() {
        return;
    }
    let total: f32 = kids.iter().map(|&c| width[c]).sum::<f32>().max(1e-9);
    let mut cur = y0;
    for &c in kids {
        let h = width[c] / total * (y1 - y0);
        assign_y(children, width, c, cur, cur + h, out);
        cur += h;
    }
}
