//! # Интерактивный ego-граф (`app::ego`)
//!
//! Окрестность выбранного нейрона, отрисованная через **egui_graphs 0.31**.
//! Вместо дефолтных «5px-кружков и серых линий» — читаемая анатомия:
//! * **узлы** окрашены по нейромедиатору, радиус растёт с силой связи
//!   (хабностью), драйв выделен акцентным кольцом, подпись видна всегда;
//! * **рёбра** — направленные стрелки, цвет по полярности пресинаптического
//!   НТ (возб. зелёный / торм. красный), толщина по весу, подпись `→/← w=…`.
//!
//! Два режима раскладки (`ConnApp::ego_layout_coords`):
//! * **Силы** — force-directed Fruchterman–Reingold, старт с реальной
//!   анатомии (data.world → 2D, фолбэк — кольцо для нейронов без позиции);
//! * **Координаты** — реальные 2D-позиции (та же проекция, что у атласа),
//!   нормализованные по ego-bbox без искажения пропорций; при смене
//!   проекции атласа граф пересобирается автоматически.
//!
//! * `EgoNode` / `EgoEdge` — полезная нагрузка узлов/рёбер для egui_graphs;
//! * `EgoNodeShape` / `EgoEdgeShape` — кастомная отрисовка (размер, стрелки);
//! * `EgoGraph` / `EgoView` — конкретизированные generic-типы;
//! * `rebuild_ego_graph` — собирает граф заново при каждом выборе нейрона.

use eframe::egui::epaint::{CircleShape, TextShape};
use eframe::egui::{pos2, Color32, FontId, Pos2, Shape, Stroke, Vec2};
use egui_graphs::{
    DisplayEdge, DisplayNode, DrawContext, EdgeProps, FruchtermanReingold, FruchtermanReingoldState,
    Graph, GraphView, LayoutForceDirected, Node, NodeProps,
};
use petgraph::stable_graph::{DefaultIx, IndexType, NodeIndex, StableGraph};
use petgraph::{Directed, EdgeType};

use crate::app::connectome::ConnectomeMode;

/// Порог зума, начиная с которого подписи узлов видны всегда (иначе —
/// только у выбранного/наведённого узла). Десятки всегда видимых подписей —
/// главный источник «каши» на ego-графе.
const NODE_LABEL_ZOOM: f32 = 1.4;
/// Порог зума, начиная с которого подпись веса ребра видна (иначе — только
/// у выделенного ребра). Вес уже закодирован толщиной/цветом, поэтому
/// текст не обязан висеть на каждом ребре.
const EDGE_LABEL_ZOOM: f32 = 1.6;

/// Полезная нагрузка узла ego-графа: индекс нейрона в `MancData` + хабность.
#[derive(Clone)]
pub(crate) struct EgoNode {
    pub(crate) idx: usize,
    /// Хабность (0..1): |вес| связи с выбранным нейроном, нормированный по
    /// максимуму окрестности — задаёт размер узла.
    pub(crate) hub: f32,
}

/// Полезная нагрузка ребра: вес синапса, направление (out = от драйва),
/// полярность пресинаптического нейрона (для окраски возб./торм.), число
/// агрегированных синапсов и индекс пресинаптического нейрона (для тултипа).
#[derive(Clone)]
pub(crate) struct EgoEdge {
    pub(crate) weight: f32,
    pub(crate) out: bool,
    /// Полярность пресинаптического НТ: >0 возб., <0 торм., 0 неизвестно.
    pub(crate) pol: f32,
    /// Нормированный |вес| (0..1) — толщина штриха.
    pub(crate) w_norm: f32,
    /// Число агрегированных синапсов (до суммаризации).
    pub(crate) count: usize,
    /// Индекс пресинаптического нейрона (драйв для out, сосед для in).
    pub(crate) presyn: usize,
}

/// Цвет ребра по полярности: возб. — тихий зелёный, торм. — тихий красный,
/// неизвестно — нейтральный серый. Читается на тёмной панели и не спорит с
/// сигнальным акцентом темы (тот расходуется на выбранный узел).
pub(crate) fn edge_color(pol: f32) -> Color32 {
    if pol > 0.0 {
        Color32::from_rgb(0x6f, 0xd0, 0x8f)
    } else if pol < 0.0 {
        Color32::from_rgb(0xe0, 0x70, 0x70)
    } else {
        Color32::from_rgb(0x8a, 0x92, 0x93)
    }
}

/// Кастомная форма узла ego-графа: радиус растёт с хабностью, выбранный
/// центр — крупный с акцентным кольцом, подпись видна всегда (граф ≤ ~64
/// соседей). `DefaultNodeShape` рисует все узлы одинаковыми 5px-кружками —
/// для окрестности нейрона это нечитаемо.
#[derive(Clone)]
pub(crate) struct EgoNodeShape {
    pos: Pos2,
    selected: bool,
    hovered: bool,
    color: Color32,
    label: String,
    hub: f32,
}

impl From<NodeProps<EgoNode>> for EgoNodeShape {
    fn from(p: NodeProps<EgoNode>) -> Self {
        Self {
            pos: p.location(),
            selected: p.selected,
            hovered: p.hovered,
            color: p.color().unwrap_or(Color32::WHITE),
            label: p.label,
            hub: p.payload.hub,
        }
    }
}

impl EgoNodeShape {
    /// Радиус в canvas-координатах: центр крупный, соседи — по силе связи.
    fn radius(&self) -> f32 {
        if self.selected {
            16.0
        } else {
            7.0 + 7.0 * self.hub.clamp(0.0, 1.0)
        }
    }
}

impl<E: Clone, Ty: EdgeType, Ix: IndexType> DisplayNode<EgoNode, E, Ty, Ix> for EgoNodeShape {
    fn is_inside(&self, pos: Pos2) -> bool {
        (pos - self.pos).length() <= self.radius()
    }

    fn closest_boundary_point(&self, dir: Vec2) -> Pos2 {
        self.pos + dir.normalized() * self.radius()
    }

    fn shapes(&mut self, ctx: &DrawContext) -> Vec<Shape> {
        let center = ctx.meta.canvas_to_screen_pos(self.pos);
        let radius = ctx.meta.canvas_to_screen_size(self.radius());
        // Акцентный цвет темы (hyperlink_color зеркалится из elegance focus).
        let accent = ctx.ctx.global_style().visuals.hyperlink_color;
        let stroke = if self.selected {
            Stroke::new(2.5, accent)
        } else if self.hovered {
            Stroke::new(1.5, Color32::WHITE)
        } else {
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 0, 0, 130))
        };
        let mut res = Vec::with_capacity(2);
        res.push(
            CircleShape {
                center,
                radius,
                fill: self.color,
                stroke,
            }
            .into(),
        );
        // Подпись — только у выбранного/наведённого узла, а у остальных
        // лишь при достаточном приближении; иначе десятки подписей
        // накладываются в нечитаемую кашу. Размер фиксированный (не
        // масштабируется с зумом).
        let show_label = self.selected || self.hovered || ctx.meta.zoom >= NODE_LABEL_ZOOM;
        if show_label && !self.label.is_empty() {
            let galley = ctx.ctx.fonts_mut(|f| {
                f.layout_no_wrap(
                    self.label.clone(),
                    FontId::proportional(11.0),
                    Color32::from_rgba_unmultiplied(0xd8, 0xe4, 0xe1, 0xf0),
                )
            });
            let label_pos = Pos2::new(center.x - galley.size().x / 2.0, center.y + radius + 2.0);
            res.push(TextShape::new(label_pos, galley, Color32::WHITE).into());
        }
        res
    }

    fn update(&mut self, state: &NodeProps<EgoNode>) {
        self.pos = state.location();
        self.selected = state.selected;
        self.hovered = state.hovered;
        self.color = state.color().unwrap_or(self.color);
        self.label = state.label.clone();
        self.hub = state.payload.hub;
    }
}

/// Кастомная форма ребра: направленная стрелка с цветом по полярности и
/// толщиной по весу. `DefaultEdgeShape` рисует все рёбра одним серым и не
/// даёт окрасить ребро по payload — для ego-графа это теряет главную
/// информацию (возбуждение/торможение).
#[derive(Clone)]
pub(crate) struct EgoEdgeShape {
    selected: bool,
    pol: f32,
    w_norm: f32,
    label: String,
}

impl From<EdgeProps<EgoEdge>> for EgoEdgeShape {
    fn from(p: EdgeProps<EgoEdge>) -> Self {
        Self {
            selected: p.selected,
            pol: p.payload.pol,
            w_norm: p.payload.w_norm,
            label: p.label,
        }
    }
}

impl<Ty: EdgeType, Ix: IndexType, D: DisplayNode<EgoNode, EgoEdge, Ty, Ix>>
    DisplayEdge<EgoNode, EgoEdge, Ty, Ix, D> for EgoEdgeShape
{
    fn shapes(
        &mut self,
        start: &Node<EgoNode, EgoEdge, Ty, Ix, D>,
        end: &Node<EgoNode, EgoEdge, Ty, Ix, D>,
        ctx: &DrawContext,
    ) -> Vec<Shape> {
        let from = start.location();
        let to = end.location();
        let dir = (to - from).normalized();
        let a = start.display().closest_boundary_point(dir);
        let b = end.display().closest_boundary_point(-dir);
        let a_s = ctx.meta.canvas_to_screen_pos(a);
        let b_s = ctx.meta.canvas_to_screen_pos(b);

        let color = edge_color(self.pol);
        let width = 0.8 + 2.4 * self.w_norm + if self.selected { 0.7 } else { 0.0 };
        let stroke = Stroke::new(ctx.meta.canvas_to_screen_size(width), color);

        let dir_s = (b_s - a_s).normalized();
        let tip_size = ctx.meta.canvas_to_screen_size(10.0);
        let tip_angle = std::f32::consts::TAU / 24.0;
        // Тело ребра обрывается перед наконечником, чтобы стрелка не сливалась.
        let tip_base = b_s - dir_s * tip_size;
        let rotate = |v: Vec2, ang: f32| {
            let (c, s) = (ang.cos(), ang.sin());
            Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
        };

        let mut res = Vec::with_capacity(3);
        res.push(Shape::line_segment([a_s, tip_base], stroke));
        res.push(Shape::convex_polygon(
            vec![
                b_s,
                b_s - rotate(dir_s, tip_angle) * tip_size,
                b_s - rotate(dir_s, -tip_angle) * tip_size,
            ],
            color,
            Stroke::NONE,
        ));
        // Подпись `→/← w=…` — только при приближении или выделении ребра:
        // вес уже закодирован толщиной и цветом, а вечные подписи на каждом
        // ребре превращают граф в кашу.
        let show_label = self.selected || ctx.meta.zoom >= EDGE_LABEL_ZOOM;
        if show_label && !self.label.is_empty() {
            let galley = ctx
                .ctx
                .fonts_mut(|f| f.layout_no_wrap(self.label.clone(), FontId::monospace(9.0), color));
            let mid = a_s + (b_s - a_s) / 2.0;
            let lp = Pos2::new(mid.x - galley.size().x / 2.0, mid.y - galley.size().y - 2.0);
            res.push(TextShape::new(lp, galley, color).into());
        }
        res
    }

    fn update(&mut self, state: &EdgeProps<EgoEdge>) {
        self.selected = state.selected;
        self.pol = state.payload.pol;
        self.w_norm = state.payload.w_norm;
        self.label = state.label.clone();
    }

    fn is_inside(
        &self,
        start: &Node<EgoNode, EgoEdge, Ty, Ix, D>,
        end: &Node<EgoNode, EgoEdge, Ty, Ix, D>,
        pos: Pos2,
    ) -> bool {
        let a = start.location();
        let b = end.location();
        let ab = b - a;
        let len2 = ab.length_sq();
        if len2 <= f32::EPSILON {
            return (pos - a).length() <= 4.0;
        }
        let t = ((pos - a).dot(ab) / len2).clamp(0.0, 1.0);
        (pos - (a + ab * t)).length() <= 4.0 + 4.0 * self.w_norm
    }
}

/// Конкретный граф: узлы = нейроны, рёбра = связи, направленный.
pub(crate) type EgoGraph =
    Graph<EgoNode, EgoEdge, Directed, DefaultIx, EgoNodeShape, EgoEdgeShape>;

/// Конкретное вью: та же конфигурация + силовая раскладка FR.
pub(crate) type EgoView<'a> = GraphView<
    'a,
    EgoNode,
    EgoEdge,
    Directed,
    DefaultIx,
    EgoNodeShape,
    EgoEdgeShape,
    FruchtermanReingoldState,
    LayoutForceDirected<FruchtermanReingold>,
>;

impl ConnectomeMode {
    /// Пересобирает ego-граф вокруг `self.sel.selected`:
    /// * узлы дедуплицируются по индексу, **стартуют на кольце** (чтобы
    ///   силовая раскладка не взрывалась из точки (0,0)), красятся по НТ,
    ///   подписываются `#idx тип`, радиус — по хабности;
    /// * рёбра направлены по факту (out: драйв→сосед, in: сосед→драйв),
    ///   окрашены по полярности пресинаптического НТ, подпись `→/← w=…`;
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
        for &(t, _, _, _) in &self.sel.ego {
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
        let n = ids.len() as f32;
        let rank: std::collections::HashMap<usize, usize> =
            ids.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        // Хабность и масштаб весов: максимум по окрестности (оба направления).
        let w_max = self
            .sel
            .ego
            .iter()
            .map(|&(_, w, _, _)| w.abs())
            .fold(0.0f32, f32::max)
            .max(1e-3);
        let mut hub: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for &(t, w, _, _) in &self.sel.ego {
            let e = hub.entry(t).or_insert(0.0);
            *e = (*e).max(w.abs());
        }
        for v in hub.values_mut() {
            *v = (*v / w_max).clamp(0.0, 1.0);
        }
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
            let ni = g.add_node_with_location(
                EgoNode {
                    idx,
                    hub: hub.get(&idx).copied().unwrap_or(0.0),
                },
                loc,
            );
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
        let targets: Vec<(usize, NodeIndex<DefaultIx>, f32, bool, usize)> = self
            .sel
            .ego
            .iter()
            .map(|&(t, w, out, count)| (t, ensure(t), w, out, count))
            .collect();
        for (t, t_ni, w, out, count) in targets {
            // Полярность — по пресинаптическому нейрону: out → драйв,
            // in → сосед.
            let presyn = if out { sel } else { t };
            let pol = d.nt_types.get(presyn).copied().unwrap_or(0.0);
            let w_norm = (w.abs() / w_max).clamp(0.0, 1.0);
            // Направление по факту: out — драйв→сосед, in — сосед→драйв,
            // чтобы стрелка смотрела верно, а не всегда «от драйва».
            let (a, b) = if out { (sel_ni, t_ni) } else { (t_ni, sel_ni) };
            let e_ni = g.add_edge(
                a,
                b,
                EgoEdge {
                    weight: w,
                    out,
                    pol,
                    w_norm,
                    count,
                    presyn,
                },
            );
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
