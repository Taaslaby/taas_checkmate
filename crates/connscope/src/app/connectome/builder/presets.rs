//! Пресеты графов — LIF / Morris-Lecar / HH-lite.

use eframe::egui::pos2;
use pillars::api::formula::{OpKind, I_SYN, S0, S1, S2, S3, TEMP0};

/// Временный билдер для пресетов — авто-аллоцирует temp-регистры.
pub(crate) struct PresetBuilder {
    pub nodes: Vec<super::graph::Node>,
    pub edges: Vec<super::graph::Edge>,
    next_t: usize,
}

impl PresetBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_t: TEMP0,
        }
    }
    fn alloc_t(&mut self) -> usize {
        let t = self.next_t;
        self.next_t += 1;
        t
    }
    pub fn node(&mut self, kind: OpKind, x: f32, y: f32, out_reg: usize, c: f32, c2: f32) -> usize {
        super::graph::add_node(&mut self.nodes, kind, pos2(x, y), out_reg, c, c2)
    }
    pub fn const_t(&mut self, x: f32, y: f32, value: f32) -> usize {
        let t = self.alloc_t();
        self.node(OpKind::Const, x, y, t, value, 0.0)
    }
    pub fn tmp(&mut self, kind: OpKind, x: f32, y: f32, c: f32, c2: f32) -> usize {
        let t = self.alloc_t();
        self.node(kind, x, y, t, c, c2)
    }
    pub fn wire(&mut self, from: usize, from_pin: usize, to: usize, to_pin: usize) {
        self.edges.push(super::Edge {
            from,
            from_pin,
            to,
            to_pin,
        });
    }
}

/// Канонический LIF — absolute-FREEZE refractory.
pub(crate) fn default_lif_graph() -> (Vec<super::Node>, Vec<super::Edge>) {
    let mut b = PresetBuilder::new();
    let state = b.node(OpKind::State, 20.0, 40.0, 0, 0.0, 0.0);
    let stim = b.node(OpKind::Stimulus, 20.0, 210.0, I_SYN, 0.0, 0.0);
    let c_rest = b.const_t(20.0, 310.0, -70.0);
    let c_tau = b.const_t(20.0, 366.0, 0.1);
    let c_thresh = b.const_t(20.0, 422.0, -54.0);
    let c_reset = b.const_t(20.0, 478.0, -70.0);
    let c_tref = b.const_t(20.0, 534.0, 2.0);
    let c_zero = b.const_t(20.0, 590.0, 0.0);
    let is_refr = b.tmp(OpKind::CmpGt, 360.0, 20.0, 0.0, 0.0);
    let leak = b.tmp(OpKind::Leak, 360.0, 110.0, 0.0, 0.0);
    let gated = b.tmp(OpKind::Blend, 360.0, 210.0, 0.0, 0.0);
    let fire = b.node(OpKind::FireReset, 360.0, 310.0, S0, 0.0, 0.0);
    let timer = b.node(OpKind::RefrTimer, 360.0, 420.0, S1, 0.0, 0.0);
    let scope = b.node(OpKind::Scope, 700.0, 160.0, 0, 0.0, 0.0);
    b.wire(state, 1, is_refr, 0);
    b.wire(c_zero, 0, is_refr, 1);
    b.wire(state, 0, leak, 0);
    b.wire(stim, 0, leak, 1);
    b.wire(c_tau, 0, leak, 2);
    b.wire(c_rest, 0, leak, 3);
    b.wire(is_refr, 0, gated, 0);
    b.wire(state, 0, gated, 1);
    b.wire(leak, 0, gated, 2);
    b.wire(gated, 0, fire, 0);
    b.wire(c_thresh, 0, fire, 1);
    b.wire(c_reset, 0, fire, 2);
    b.wire(state, 1, timer, 0);
    b.wire(c_tref, 0, timer, 1);
    b.wire(state, 0, scope, 0);
    (b.nodes, b.edges)
}

/// Morris-Lecar 2D.
pub(crate) fn ml_graph() -> (Vec<super::Node>, Vec<super::Edge>) {
    let mut b = PresetBuilder::new();
    let state = b.node(OpKind::State, 20.0, 40.0, 0, 0.0, 0.0);
    let stim = b.node(OpKind::Stimulus, 20.0, 540.0, I_SYN, 0.0, 0.0);
    let c_neg = b.const_t(20.0, 100.0, -1.0);
    let g_ca = b.const_t(20.0, 160.0, 1.1);
    let e_ca = b.const_t(20.0, 220.0, 100.0);
    let g_k = b.const_t(20.0, 280.0, 2.0);
    let e_k = b.const_t(20.0, 340.0, -70.0);
    let g_l = b.const_t(20.0, 400.0, 0.5);
    let e_l = b.const_t(20.0, 460.0, -50.0);
    let inv_c = b.const_t(20.0, 600.0, 1.0);
    let phi = b.const_t(20.0, 660.0, 0.04);
    let zero = b.const_t(20.0, 720.0, 0.0);
    let t_m = b.alloc_t();
    let t_winf = b.alloc_t();
    let sig_m = b.node(OpKind::Sigmoid, 220.0, 40.0, t_m, -1.2, 9.0);
    let sig_w = b.node(OpKind::Sigmoid, 220.0, 130.0, t_winf, 12.0, 8.7);
    let gate_w = b.node(OpKind::GateStep, 420.0, 130.0, S1, 0.0, 0.0);
    let mul_gca = b.tmp(OpKind::Mul, 420.0, 40.0, 0.0, 0.0);
    let mul_gk = b.tmp(OpKind::Mul, 420.0, 220.0, 0.0, 0.0);
    let i_ca = b.tmp(OpKind::Ohmic, 620.0, 40.0, 0.0, 0.0);
    let i_k = b.tmp(OpKind::Ohmic, 620.0, 140.0, 0.0, 0.0);
    let i_l = b.tmp(OpKind::Ohmic, 620.0, 240.0, 0.0, 0.0);
    let sum1 = b.tmp(OpKind::Add, 820.0, 40.0, 0.0, 0.0);
    let sum2 = b.tmp(OpKind::Add, 820.0, 130.0, 0.0, 0.0);
    let neg = b.tmp(OpKind::Mul, 820.0, 220.0, 0.0, 0.0);
    let net = b.tmp(OpKind::Add, 820.0, 310.0, 0.0, 0.0);
    let drive = b.tmp(OpKind::Add, 1020.0, 40.0, 0.0, 0.0);
    let leak = b.node(OpKind::Leak, 1020.0, 130.0, S0, 0.0, 0.0);
    b.wire(state, 0, sig_m, 0);
    b.wire(state, 0, sig_w, 0);
    b.wire(sig_m, 0, mul_gca, 0);
    b.wire(g_ca, 0, mul_gca, 1);
    b.wire(mul_gca, 0, i_ca, 0);
    b.wire(state, 0, i_ca, 1);
    b.wire(e_ca, 0, i_ca, 2);
    b.wire(gate_w, 0, mul_gk, 0);
    b.wire(g_k, 0, mul_gk, 1);
    b.wire(mul_gk, 0, i_k, 0);
    b.wire(state, 0, i_k, 1);
    b.wire(e_k, 0, i_k, 2);
    b.wire(g_l, 0, i_l, 0);
    b.wire(state, 0, i_l, 1);
    b.wire(e_l, 0, i_l, 2);
    b.wire(i_ca, 0, sum1, 0);
    b.wire(i_k, 0, sum1, 1);
    b.wire(sum1, 0, sum2, 0);
    b.wire(i_l, 0, sum2, 1);
    b.wire(c_neg, 0, neg, 0);
    b.wire(sum2, 0, neg, 1);
    b.wire(neg, 0, net, 0);
    b.wire(stim, 0, net, 1);
    b.wire(state, 0, drive, 0);
    b.wire(net, 0, drive, 1);
    b.wire(state, 0, leak, 0);
    b.wire(drive, 0, leak, 1);
    b.wire(inv_c, 0, leak, 2);
    b.wire(zero, 0, leak, 3);
    b.wire(state, 1, gate_w, 0);
    b.wire(sig_w, 0, gate_w, 1);
    b.wire(phi, 0, gate_w, 2);
    (b.nodes, b.edges)
}

/// HH-lite — 3 гейта, m³h + n⁴.
pub(crate) fn hh_lite_graph() -> (Vec<super::Node>, Vec<super::Edge>) {
    let mut b = PresetBuilder::new();
    let state = b.node(OpKind::State, 20.0, 40.0, 0, 0.0, 0.0);
    let stim = b.node(OpKind::Stimulus, 20.0, 600.0, I_SYN, 0.0, 0.0);
    let c_neg = b.const_t(20.0, 100.0, -1.0);
    let g_na = b.const_t(20.0, 160.0, 120.0);
    let e_na = b.const_t(20.0, 220.0, 50.0);
    let g_k = b.const_t(20.0, 280.0, 36.0);
    let e_k = b.const_t(20.0, 340.0, -77.0);
    let g_l = b.const_t(20.0, 400.0, 0.3);
    let e_l = b.const_t(20.0, 460.0, -54.4);
    let inv_c = b.const_t(20.0, 660.0, 1.0);
    let zero = b.const_t(20.0, 720.0, 0.0);
    let t_m = b.alloc_t();
    let t_h = b.alloc_t();
    let t_n = b.alloc_t();
    let sig_m = b.node(OpKind::Sigmoid, 220.0, 40.0, t_m, -40.0, 6.0);
    let sig_h = b.node(OpKind::Sigmoid, 220.0, 130.0, t_h, -60.0, -6.0);
    let sig_n = b.node(OpKind::Sigmoid, 220.0, 220.0, t_n, -50.0, 10.0);
    let tau_m = b.const_t(420.0, 40.0, 5.0);
    let tau_h = b.const_t(420.0, 130.0, 0.5);
    let tau_n = b.const_t(420.0, 220.0, 1.0);
    let gate_m = b.node(OpKind::GateStep, 420.0, 60.0, S1, 0.0, 0.0);
    let gate_h = b.node(OpKind::GateStep, 420.0, 150.0, S2, 0.0, 0.0);
    let gate_n = b.node(OpKind::GateStep, 420.0, 240.0, S3, 0.0, 0.0);
    let m2 = b.tmp(OpKind::Mul, 620.0, 40.0, 0.0, 0.0);
    let m3 = b.tmp(OpKind::Mul, 620.0, 100.0, 0.0, 0.0);
    let m3h = b.tmp(OpKind::Mul, 620.0, 160.0, 0.0, 0.0);
    let gna_m = b.tmp(OpKind::Mul, 620.0, 220.0, 0.0, 0.0);
    let n2 = b.tmp(OpKind::Mul, 620.0, 280.0, 0.0, 0.0);
    let n4 = b.tmp(OpKind::Mul, 620.0, 340.0, 0.0, 0.0);
    let gk_n = b.tmp(OpKind::Mul, 620.0, 400.0, 0.0, 0.0);
    let i_na = b.tmp(OpKind::Ohmic, 820.0, 40.0, 0.0, 0.0);
    let i_k = b.tmp(OpKind::Ohmic, 820.0, 140.0, 0.0, 0.0);
    let i_l = b.tmp(OpKind::Ohmic, 820.0, 240.0, 0.0, 0.0);
    let sum1 = b.tmp(OpKind::Add, 820.0, 40.0, 0.0, 0.0);
    let sum2 = b.tmp(OpKind::Add, 820.0, 130.0, 0.0, 0.0);
    let neg = b.tmp(OpKind::Mul, 820.0, 220.0, 0.0, 0.0);
    let net = b.tmp(OpKind::Add, 820.0, 310.0, 0.0, 0.0);
    let drive = b.tmp(OpKind::Add, 1220.0, 40.0, 0.0, 0.0);
    let leak = b.node(OpKind::Leak, 1220.0, 130.0, S0, 0.0, 0.0);
    b.wire(state, 1, gate_m, 0);
    b.wire(sig_m, 0, gate_m, 1);
    b.wire(tau_m, 0, gate_m, 2);
    b.wire(state, 2, gate_h, 0);
    b.wire(sig_h, 0, gate_h, 1);
    b.wire(tau_h, 0, gate_h, 2);
    b.wire(state, 3, gate_n, 0);
    b.wire(sig_n, 0, gate_n, 1);
    b.wire(tau_n, 0, gate_n, 2);
    b.wire(state, 1, m2, 0);
    b.wire(state, 1, m2, 1);
    b.wire(m2, 0, m3, 0);
    b.wire(state, 1, m3, 1);
    b.wire(m3, 0, m3h, 0);
    b.wire(state, 2, m3h, 1);
    b.wire(g_na, 0, gna_m, 0);
    b.wire(m3h, 0, gna_m, 1);
    b.wire(gna_m, 0, i_na, 0);
    b.wire(state, 0, i_na, 1);
    b.wire(e_na, 0, i_na, 2);
    b.wire(state, 3, n2, 0);
    b.wire(state, 3, n2, 1);
    b.wire(n2, 0, n4, 0);
    b.wire(n2, 0, n4, 1);
    b.wire(g_k, 0, gk_n, 0);
    b.wire(n4, 0, gk_n, 1);
    b.wire(gk_n, 0, i_k, 0);
    b.wire(state, 0, i_k, 1);
    b.wire(e_k, 0, i_k, 2);
    b.wire(g_l, 0, i_l, 0);
    b.wire(state, 0, i_l, 1);
    b.wire(e_l, 0, i_l, 2);
    b.wire(i_na, 0, sum1, 0);
    b.wire(i_k, 0, sum1, 1);
    b.wire(sum1, 0, sum2, 0);
    b.wire(i_l, 0, sum2, 1);
    b.wire(c_neg, 0, neg, 0);
    b.wire(sum2, 0, neg, 1);
    b.wire(neg, 0, net, 0);
    b.wire(stim, 0, net, 1);
    b.wire(state, 0, drive, 0);
    b.wire(net, 0, drive, 1);
    b.wire(state, 0, leak, 0);
    b.wire(drive, 0, leak, 1);
    b.wire(inv_c, 0, leak, 2);
    b.wire(zero, 0, leak, 3);
    (b.nodes, b.edges)
}
