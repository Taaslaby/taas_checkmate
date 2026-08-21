//! Геометрия нод — размеры, сетка, bezier-провода.

use eframe::egui::{pos2, Pos2, Vec2};

use super::spec::spec_of;
use pillars::api::formula::OpKind;

// ─── Layout constants ───────────────────────────────────────────
pub(crate) const NODE_W: f32 = 150.0;
pub(crate) const HEADER_H: f32 = 26.0;
pub(crate) const PIN_H: f32 = 22.0;
pub(crate) const PIN_R: f32 = 5.0;
pub(crate) const HIT_R: f32 = 12.0;
pub(crate) const SCOPE_POINTS: usize = 4000;
pub(crate) const GRID: f32 = 40.0;

/// Размер тела ноды в world-координатах.
pub(crate) fn node_size_of(kind: OpKind) -> Vec2 {
    let spec = spec_of(kind);
    let rows = spec.ins.len().max(spec.outs.len()).max(1) as f32;
    Vec2::new(NODE_W, HEADER_H + rows * PIN_H)
}

// ─── Bezier wires ──────────────────────────────────────────────
pub(crate) fn bezier_ctrl(a: Pos2, b: Pos2) -> (Pos2, Pos2) {
    let dx = (b.x - a.x).abs().max(40.0) * 0.5;
    (pos2(a.x + dx, a.y), pos2(b.x - dx, b.y))
}

pub(crate) fn bezier_eval(a: Pos2, b: Pos2, t: f32) -> Pos2 {
    let (c1, c2) = bezier_ctrl(a, b);
    let u = 1.0 - t;
    pos2(
        u * u * u * a.x + 3.0 * u * u * t * c1.x + 3.0 * u * t * t * c2.x + t * t * t * b.x,
        u * u * u * a.y + 3.0 * u * u * t * c1.y + 3.0 * u * t * t * c2.y + t * t * t * b.y,
    )
}

pub(crate) fn bezier_polyline(a: Pos2, b: Pos2, steps: usize) -> Vec<Pos2> {
    (0..=steps)
        .map(|i| bezier_eval(a, b, i as f32 / steps as f32))
        .collect()
}
