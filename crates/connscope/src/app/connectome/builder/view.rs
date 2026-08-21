//! Вью — пан/зум, хит-тестинг, миникарта.

use eframe::egui::{pos2, vec2, Pos2, Rect, Vec2};

use super::geometry::{HIT_R, NODE_W, node_size_of};
use super::graph::GEdge;
use super::spec::spec_of;

use super::NeuronBuilder;

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = ap.dot(ab) / ab.dot(ab).clamp(1e-6, f32::MAX);
    let t = t.clamp(0.0, 1.0);
    (ap - ab * t).length()
}

impl NeuronBuilder {
    pub(crate) fn to_screen(&self, w: Vec2) -> Vec2 {
        w * self.zoom + self.pan
    }
    pub(crate) fn screen_to_world(&self, s: Vec2) -> Vec2 {
        (s - self.pan) / self.zoom
    }
    pub(crate) fn node_rect(&self, i: usize) -> Rect {
        let node = &self.nodes[i];
        Rect::from_min_size(node.pos, node_size_of(node.kind))
    }
    pub(crate) fn in_pin_pos(&self, i: usize, pin: usize) -> Vec2 {
        let node = &self.nodes[i];
        vec2(
            node.pos.x,
            node.pos.y + super::geometry::HEADER_H + (pin as f32 + 0.5) * super::geometry::PIN_H,
        )
    }
    pub(crate) fn out_pin_pos(&self, i: usize, pin: usize) -> Vec2 {
        let node = &self.nodes[i];
        vec2(
            node.pos.x + NODE_W,
            node.pos.y + super::geometry::HEADER_H + (pin as f32 + 0.5) * super::geometry::PIN_H,
        )
    }
    pub(crate) fn hit_node(&self, p: Vec2) -> Option<usize> {
        (0..self.nodes.len()).find(|&i| self.node_rect(i).contains(pos2(p.x, p.y)))
    }
    pub(crate) fn hit_out_pin(&self, p: Vec2) -> Option<(usize, usize)> {
        for i in 0..self.nodes.len() {
            for pin in 0..spec_of(self.nodes[i].kind).outs.len() {
                if (self.out_pin_pos(i, pin) - p).length() < HIT_R {
                    return Some((i, pin));
                }
            }
        }
        None
    }
    pub(crate) fn hit_in_pin(&self, p: Vec2) -> Option<(usize, usize)> {
        for i in 0..self.nodes.len() {
            for pin in 0..spec_of(self.nodes[i].kind).ins.len() {
                if (self.in_pin_pos(i, pin) - p).length() < HIT_R {
                    return Some((i, pin));
                }
            }
        }
        None
    }
    pub(crate) fn dist_to_edge(&self, e: &GEdge, p: Vec2) -> f32 {
        if e.from >= self.nodes.len() || e.to >= self.nodes.len() {
            return f32::INFINITY;
        }
        let a = self.out_pin_pos(e.from, e.from_pin);
        let b = self.in_pin_pos(e.to, e.to_pin);
        let pts = super::geometry::bezier_polyline(pos2(a.x, a.y), pos2(b.x, b.y), 16);
        pts.windows(2)
            .map(|w| dist_to_segment(p, w[0].to_vec2(), w[1].to_vec2()))
            .fold(f32::INFINITY, f32::min)
    }
    pub(crate) fn hit_edge(&self, p: Vec2) -> Option<usize> {
        let tol = 8.0 / self.zoom;
        (0..self.edges.len())
            .filter(|&i| self.dist_to_edge(&self.edges[i], p) < tol)
            .min_by(|&a, &b| {
                self.dist_to_edge(&self.edges[a], p)
                    .partial_cmp(&self.dist_to_edge(&self.edges[b], p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    pub(crate) fn graph_world_bounds(&self) -> (Vec2, Vec2) {
        let mut mn = vec2(f32::INFINITY, f32::INFINITY);
        let mut mx = vec2(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for n in &self.nodes {
            let sz = node_size_of(n.kind);
            mn.x = mn.x.min(n.pos.x);
            mn.y = mn.y.min(n.pos.y);
            mx.x = mx.x.max(n.pos.x + sz.x);
            mx.y = mx.y.max(n.pos.y + sz.y);
        }
        if !mn.x.is_finite() {
            return (vec2(-220.0, -220.0), vec2(220.0, 220.0));
        }
        (mn - vec2(40.0, 40.0), mx + vec2(40.0, 40.0))
    }
    pub(crate) fn minimap_rect(&self, canvas_size: Vec2) -> Rect {
        let mm = vec2(190.0, 132.0);
        Rect::from_min_size(
            vec2(canvas_size.x - mm.x - 14.0, canvas_size.y - mm.y - 14.0).to_pos2(),
            mm,
        )
    }
    pub(crate) fn minimap_map(&self, canvas_size: Vec2) -> (Rect, f32, Vec2) {
        let (gmn, gmx) = self.graph_world_bounds();
        let mm = self.minimap_rect(canvas_size);
        let gs = gmx - gmn;
        let scale = ((mm.width() - 10.0) / gs.x)
            .min((mm.height() - 10.0) / gs.y)
            .clamp(0.01, 1.0);
        let drawn = vec2(gs.x * scale, gs.y * scale);
        let off = mm.min + vec2((mm.width() - drawn.x) * 0.5, (mm.height() - drawn.y) * 0.5);
        (Rect::from_min_size(off, drawn), scale, gmn)
    }
    pub(crate) fn minimap_pan_to(&mut self, canvas_size: Vec2, pointer_screen: Option<Vec2>) {
        let (grect, scale, gmn) = self.minimap_map(canvas_size);
        if let Some(s) = pointer_screen {
            let wp = gmn + (s - grect.min.to_vec2()) / scale;
            self.pan = canvas_size * 0.5 - wp * self.zoom;
        }
    }
}
