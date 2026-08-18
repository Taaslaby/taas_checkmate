//! # Интерактивный ego-граф (`app::ego`)
//!
//! Окрестность выбранного нейрона, отрисованная через **egui_graphs 0.31**:
//! узлы окрашены по нейромедиатору, рёбра подписаны весом и направлением
//! (`→ w=…` / `← w=…`).
//!
//! Два режима раскладки (`ConnApp::ego_layout_coords`):
//! * **Силы** — force-directed Fruchterman–Reingold, старт с реальной
//!   анатомии (data.world → 2D, фолбэк — кольцо для нейронов без позиции);
//! * **Координаты** — реальные 2D-позиции (та же проекция, что у атласа),
//!   нормализованные по ego-bbox без искажения пропорций; при смене
//!   проекции атласа граф пересобирается автоматически.
//!
//! * `EgoNode` / `EgoEdge` — полезная нагрузка узлов/рёбер для egui_graphs;
//! * `EgoGraph` / `EgoView` — конкретизированные generic-типы;
//! * `rebuild_ego_graph` — собирает граф заново при каждом выборе нейрона.

use eframe::egui::pos2;
use egui_graphs::{
    DefaultEdgeShape, DefaultNodeShape, FruchtermanReingold, FruchtermanReingoldState, Graph,
    GraphView, LayoutForceDirected,
};
use petgraph::stable_graph::{DefaultIx, NodeIndex, StableGraph};
use petgraph::Directed;

use crate::app::connectome::ConnectomeMode;

/// Полезная нагрузка узла ego-графа: индекс нейрона в `MancData`.
#[derive(Clone)]
pub(crate) struct EgoNode {
    pub(crate) idx: usize,
}

/// Полезная нагрузка ребра: вес синапса и направление (out = от драйва).
#[derive(Clone)]
pub(crate) struct EgoEdge {
    pub(crate) weight: f32,
    pub(crate) out: bool,
}

/// Конкретный граф: узлы = нейроны, рёбра = связи, направленный.
pub(crate) type EgoGraph =
    Graph<EgoNode, EgoEdge, Directed, DefaultIx, DefaultNodeShape, DefaultEdgeShape>;

/// Конкретное вью: та же конфигурация + силовая раскладка FR.
pub(crate) type EgoView<'a> = GraphView<
    'a,
    EgoNode,
    EgoEdge,
    Directed,
    DefaultIx,
    DefaultNodeShape,
    DefaultEdgeShape,
    FruchtermanReingoldState,
    LayoutForceDirected<FruchtermanReingold>,
>;

impl ConnectomeMode {
    /// Пересобирает ego-граф вокруг `self.sel.selected`:
    /// * узлы дедуплицируются по индексу, **стартуют на кольце** (чтобы
    ///   силовая раскладка не взрывалась из точки (0,0)), красятся по НТ,
    ///   подписываются `#idx тип`;
    /// * рёбра идут от драйва (out) или к драйву (in), подпись `→/← w=…`;
    /// * драйв помечается выбранным, чтобы его было видно в раскладке.
    pub(crate) fn rebuild_ego_graph(&mut self) {
        let mut g: EgoGraph = Graph::new(StableGraph::default());
        let Some(d) = self.data.as_ref() else { return };
        let Some(sel) = self.sel.selected else { return };
        if sel >= d.n {
            return;
        }
        // Полный список узлов: драйв + все ego-соседи (без дублей).
        let mut ids: Vec<usize> = vec![sel];
        for &(t, _, _) in &self.sel.ego {
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
        let n = ids.len() as f32;
        let rank: std::collections::HashMap<usize, usize> =
            ids.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        // Ego-bbox в текущей проекции (нужен для режима «реальные координаты»).
        let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &idx in &ids {
            let w = d.world(idx, self.camera.proj);
            minx = minx.min(w.x);
            maxx = maxx.max(w.x);
            miny = miny.min(w.y);
            maxy = maxy.max(w.y);
        }
        let ccx = (minx + maxx) / 2.0;
        let ccy = (miny + maxy) / 2.0;
        let scale = (maxx - minx).max(maxy - miny).max(1e-3);
        let coords_mode = self.sel.ego_layout_coords;
        // Радиус кольца растёт с числом узлов — FR стартует с «веера».
        // Keep the layout in the same 560×400 coordinate space as anatomical
        // positions. The previous radius grew past the graph viewport and
        // made FR appear to explode for large ego neighbourhoods.
        let radius = (92.0 + 7.0 * n.sqrt()).min(150.0);
        let mut ensure = |idx: usize| -> NodeIndex<DefaultIx> {
            // Анатомические координаты (проекция как у атласа, отцентрована по
            // ego-bbox с сохранением пропорций). В режиме «координаты» это
            // финальные позиции; в силовом — стартовые для FR, чтобы граф
            // напоминал реальную анатомию, а не случайный блоб.
            let anatomy = d.has_pos(idx).then(|| {
                let w = d.world(idx, self.camera.proj);
                let w_ = 560.0;
                let h_ = 400.0;
                let pad = 40.0;
                pos2(
                    w_ / 2.0 + (w.x - ccx) / scale * (w_ / 2.0 - pad),
                    h_ / 2.0 + (ccy - w.y) / scale * (h_ / 2.0 - pad),
                )
            });
            let loc = if coords_mode {
                anatomy.unwrap_or_else(|| pos2(radius, radius))
            } else {
                anatomy.unwrap_or_else(|| {
                    // Фолбэк без позиции — кольцо.
                    let r = rank[&idx] as f32;
                    let angle = r * std::f32::consts::TAU / n;
                    pos2(280.0 + radius * angle.cos(), 200.0 + radius * angle.sin())
                })
            };
            let ni = g.add_node_with_location(EgoNode { idx }, loc);
            if let Some(node) = g.g_mut().node_weight_mut(ni) {
                node.set_color(crate::core::color::color_for(
                    d,
                    idx,
                    self.filters.color_mode,
                ));
                let ty = d.type_name(idx);
                node.set_label(format!("#{idx} {}", if ty.is_empty() { "?" } else { ty }));
            }
            ni
        };
        let sel_ni = ensure(sel);
        // Сначала собрать все цели, чтобы отпустить заимствование `ensure`
        // (оно держит `&mut g`), и только потом добавлять рёбра.
        let targets: Vec<(NodeIndex<DefaultIx>, f32, bool)> = self
            .sel
            .ego
            .iter()
            .map(|&(t, w, out)| (ensure(t), w, out))
            .collect();
        for (t_ni, w, out) in targets {
            let e_ni = g.add_edge(sel_ni, t_ni, EgoEdge { weight: w, out });
            if let Some(e) = g.g_mut().edge_weight_mut(e_ni) {
                let (ew, eo) = (e.payload().weight, e.payload().out);
                e.set_label(format!("{}{ew:.1}", if eo { "→" } else { "←" }));
            }
        }
        // Драйв подсвечен как выбранный.
        if let Some(n) = g.g_mut().node_weight_mut(sel_ni) {
            n.set_selected(true);
        }
        self.sel.ego_shown_proj = self.camera.proj;
        self.sel.ego_graph = g;
    }
}
