//! Граф нод — Node/Edge и операции над ними.

use eframe::egui::Pos2;
use pillars::api::formula::OpKind;

/// Нода графа — один Op с позицией и выходным регистром.
#[derive(Clone)]
pub(crate) struct Node {
    pub kind: OpKind,
    pub pos: Pos2,
    pub out_reg: usize,
    pub c: f32,
    pub c2: f32,
}

/// Ребро — связь выходного пина `from:from_pin` → входного `to:to_pin`.
#[derive(Clone)]
pub(crate) struct Edge {
    pub from: usize,
    pub from_pin: usize,
    pub to: usize,
    pub to_pin: usize,
}

// Совместимость со старым именем GNode/GEdge внутри mod.rs
pub(crate) type GNode = Node;
pub(crate) type GEdge = Edge;

/// Добавить ноду в список, вернуть её индекс.
pub(crate) fn add_node(
    nodes: &mut Vec<Node>,
    kind: OpKind,
    pos: Pos2,
    out_reg: usize,
    c: f32,
    c2: f32,
) -> usize {
    nodes.push(Node {
        kind,
        pos,
        out_reg,
        c,
        c2,
    });
    nodes.len() - 1
}
