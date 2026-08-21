//! Компиляция графа → программа Formula.

use pillars::api::formula::{OpKind, OpNode, N_REGS, S0};

use super::graph::{GEdge, GNode};
use super::spec::spec_of;

/// Регистр, питающий `node.pin` (State → s0..s3, иначе out_reg).
/// Bounds-guarded — stale edges пропускаются, а не паникуют.
pub(crate) fn input_reg_of(nodes: &[GNode], edges: &[GEdge], node: usize, pin: usize) -> usize {
    for e in edges {
        if e.to == node && e.to_pin == pin && e.from < nodes.len() {
            return if nodes[e.from].kind == OpKind::State {
                S0 + e.from_pin
            } else {
                nodes[e.from].out_reg
            };
        }
    }
    S0 // неподключенный пин → мембрана
}

/// Граф → плоская программа Formula (shader order).
/// State/Scope — только граф, не эмитятся. Регистры клампятся к N_REGS.
pub(crate) fn graph_to_program(nodes: &[GNode], edges: &[GEdge]) -> Vec<OpNode> {
    let mut prog = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        if matches!(node.kind, OpKind::State | OpKind::Scope) {
            continue;
        }
        let n_in = spec_of(node.kind).ins.len();
        let mut in_regs = [S0; 4];
        for (pin, reg) in in_regs[..n_in].iter_mut().enumerate() {
            *reg = input_reg_of(nodes, edges, i, pin);
        }
        let mut op = OpNode::new(node.kind, node.out_reg.min(N_REGS - 1));
        op.in_regs = in_regs;
        op.c = node.c;
        op.c2 = node.c2;
        prog.push(op);
    }
    prog
}
