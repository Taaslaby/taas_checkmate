//! # Конструктор нейрона — нодовый редактор поверх Formula (`pillars::api::formula`)
//!
//! Собери нейрон из биологических опов как **нодовый граф**, запусти
//! одноклеточную симуляцию и смотри мембранный потенциал на живом осциллографе.
//! Портирован из `pillars/examples/viz_neuron_builder.rs` в dock-поверхность
//! [`Surface::Builder`](crate::app::mode::Surface).
//!
//! ```text
//!    ┌──────────┐   ┌────────┐   ┌────────────┐   ┌────────┐   ┌──────────┐
//!    │  State   │──▶│ CmpGt  │──▶│    Blend   │──▶│FireReset│──▶ s0
//!    │ s0 · s1  │   └────────┘   └────────────┘   └────────┘   └──────────┘
//!    └──────────┘      │  ▲                              │
//!     ▶ Scope (scope)  │  └── Leak(s0, I, …) ────────────┘
//!                      ▼
//!                  ┌──────────┐   ┌────────────┐
//!                  │RefrTimer │◀──│ Const 2.0  │
//!                  └──────────┘   └────────────┘
//! ```
//!
//! LIF refractory is absolute-FREEZE: `CmpGt(s1, 0)` masks refractory,
//! `Blend` keeps the frozen v, so `FireReset` can never spike during
//! the refractory period (no breakthrough at high currents).
//!
//! ## Interaction
//! - **Drag** an output pin (orange) to an input pin (cyan) to connect.
//! - **Drag** a node by its body to move it; **left-click** to inspect.
//! - **Click a wire** to select it — then `Del` or the inspector button deletes it.
//! - **Right-click** empty canvas → context menu with ops to add at cursor;
//!   right-click a node → duplicate / delete.
//! - **Scroll** = zoom around cursor; **drag empty canvas** = pan.
//! - **Delete / Backspace** — delete selection · **Ctrl+D** — duplicate node
//!   · **Ctrl+Z / Ctrl+Y** — undo / redo · **Space** — run/pause.
//! - Presets build a full graph in one click: **LIF**, **Morris-Lecar**,
//!   **HH-lite** (3 gates). Graphs can be **saved / loaded as JSON**.
//!
//! ## Visual language (Blender / ComfyUI inspired)
//! - **Nodes are colour-coded by family** — violet = system (State/Scope),
//!   green = sources (Stimulus/Const), blue = membrane (Leak/Ohmic/…),
//!   amber = math, teal = gates, red = logic. A coloured header strip with an
//!   icon badge makes families recognizable at a glance.
//! - **Wires are cubic béziers** that inherit the source node's colour and
//!   glow on hover/selection; while the sim runs, a bright **data pulse**
//!   travels along every wire.
//! - **Ports** are hollow rings when unconnected and filled (with a dark
//!   core) once a wire is attached.
//! - **Minimap** (top-right of the canvas, toggleable) shows the whole graph;
//!   click / drag inside it to navigate.
//!
//! ## Run
//! ```text
//! cargo run -p pillars --example viz_neuron_builder --release
//! ```
//!
//! Engine: [`pillars::api::formula`] — a runtime opcode interpreter over
//! [`crate::ops`] with a persistent register file `s0..s3 / p0..p7 / i_syn / t0..t10`.
//!
//! ## Module map (2272 строки — секционирован, но монофайл)
//! ```text
//! builder/mod.rs  ← ты здесь — фасад + состояние Builder
//!   1. Layout constants       NODE_W/HEADER_H/PIN_R/GRID
//!   2. Spec                  Spec/spec_of — шаблон пинов ноды
//!   3. Category              NodeCat/cat_of/cat_color — цвет/иконка семейства
//!   4. Geometry              bezier_ctrl/eval, node_size_of, HIT_R
//!   5. Kind registry         ALL_KINDS/kind_index/palette_kinds/reg_name
//!   6. Graph model           GNode/GEdge/Drag/CtxMenu/NeuronBuilder
//!   7. Presets               PBuilder + default_lif/ml/hh_lite graphs
//!   8. IO                    FileGraph — JSON save/load
//!   9. Compile               input_reg_of/graph_to_program — граф → OpNode[]
//!  10. View                  pan/zoom, minimap, grid, hit testing
//!  11. Render                draw_grid/draw_canvas/draw_wire — паинтер
//!  12. Sim                   step_sim/reset_sim, history/scope
//!  13. UI                    dock_view, toolbar, inspector, context menus
//! ```
//! План: вынести 2-5 в `spec.rs`/`category.rs`/`geometry.rs`, 7 в `presets.rs`,
//! 8 в `io.rs`, 10-11 в `view.rs`/`render.rs` — сейчас уже лежит в `builder/`
//! директории, `connectome::neuron_builder` оставлен как алиас (`pub use builder as neuron_builder`).

use std::collections::VecDeque;

use eframe::egui::{
    self, pos2, vec2, Align2, Color32, CornerRadius, FontId, Key, Modifiers, PointerButton, Pos2,
    Rect, Sense, Stroke, StrokeKind, Vec2,
};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, VLine};
use pillars::api::formula::{self, OpKind, OpNode, I_SYN, N_REGS, S0, S1, S2, S3, TEMP0};

// ─── Layout constants ───────────────────────────────────────────

mod category;
mod spec;
use category::{cat_color, cat_icon, cat_of, cat_tint};
use spec::spec_of;
mod geometry;
mod kind;
#[allow(unused_imports)]
use geometry::{GRID, HEADER_H, HIT_R, NODE_W, PIN_H, PIN_R, SCOPE_POINTS, bezier_ctrl, bezier_eval, bezier_polyline, node_size_of};
#[allow(unused_imports)]
use kind::{ALL_KINDS, kind_at, kind_index, palette_kinds, reg_name};








// ─── Graph state ────────────────────────────────────────────────
#[derive(Clone)]
struct GNode {
    kind: OpKind,
    pos: Pos2,
    out_reg: usize,
    /// Const value / sigmoid v_half.
    c: f32,
    /// Sigmoid slope k.
    c2: f32,
}

#[derive(Clone)]
struct GEdge {
    from: usize,
    from_pin: usize,
    to: usize,
    to_pin: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    None,
    Wire {
        from: usize,
        from_pin: usize,
    },
    Node {
        idx: usize,
    },
    Pan,
    /// Panning the view by dragging inside the minimap.
    Minimap,
}

/// Context menu target.
enum CtxMenu {
    /// Right-click on empty canvas: add op at this world position.
    Canvas { screen: Pos2, world: Vec2 },
    /// Right-click on a node: duplicate / delete.
    Node { screen: Pos2, idx: usize },
    /// Right-click on a wire: delete it.
    Wire { screen: Pos2, idx: usize },
}

// ─── App ────────────────────────────────────────────────────────
pub(crate) struct NeuronBuilder {
    nodes: Vec<GNode>,
    edges: Vec<GEdge>,
    drag: Drag,
    selected: Option<usize>,
    selected_edge: Option<usize>,
    ctx_menu: Option<CtxMenu>,
    // undo / redo snapshots
    undo: Vec<(Vec<GNode>, Vec<GEdge>)>,
    redo: Vec<(Vec<GNode>, Vec<GEdge>)>,
    // view
    pan: Vec2,
    zoom: f32,
    snap: bool,
    show_minimap: bool,
    /// Whether the scope strip under the canvas is shown (collapsible).
    show_plot: bool,
    // hover (recomputed every frame from the pointer)
    hover_node: Option<usize>,
    hover_edge: Option<usize>,
    hover_tip: Option<String>,
    // sim
    running: bool,
    steps_per_frame: usize,
    dt: f32,
    i_syn: f32,
    v_init: f32,
    /// Resting gate values applied by `reset_sim` (s1..s3).
    /// Conductance models (HH, ML) can't spike from a cold gate start.
    init_gates: [f32; 3],
    regs: [f32; N_REGS],
    t: f64,
    history: VecDeque<(f64, f64)>,
    spike_times: Vec<f64>,
    total_spikes: usize,
    prev_trace: f32,
    spike_thresh: f32,
    window_ms: f64,
    /// 0 = Scope node, 1..=4 = s0..s3
    trace: usize,
    // files
    filename: String,
    status: String,
}

fn add_node(
    nodes: &mut Vec<GNode>,
    kind: OpKind,
    pos: Pos2,
    out_reg: usize,
    c: f32,
    c2: f32,
) -> usize {
    nodes.push(GNode {
        kind,
        pos,
        out_reg,
        c,
        c2,
    });
    nodes.len() - 1
}

// ─── Preset builders ────────────────────────────────────────────
/// Small builder that auto-allocates temp registers.
struct PBuilder {
    nodes: Vec<GNode>,
    edges: Vec<GEdge>,
    next_t: usize,
}

impl PBuilder {
    fn new() -> Self {
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

    fn node(&mut self, kind: OpKind, x: f32, y: f32, out_reg: usize, c: f32, c2: f32) -> usize {
        add_node(&mut self.nodes, kind, pos2(x, y), out_reg, c, c2)
    }

    /// Const node into a fresh temp register; returns the **node index**
    /// (wire endpoints are node indices, not registers).
    fn const_t(&mut self, x: f32, y: f32, value: f32) -> usize {
        let t = self.alloc_t();
        self.node(OpKind::Const, x, y, t, value, 0.0)
    }

    /// Op node with a fresh temp output register.
    fn tmp(&mut self, kind: OpKind, x: f32, y: f32, c: f32, c2: f32) -> usize {
        let t = self.alloc_t();
        self.node(kind, x, y, t, c, c2)
    }

    fn wire(&mut self, from: usize, from_pin: usize, to: usize, to_pin: usize) {
        self.edges.push(GEdge {
            from,
            from_pin,
            to,
            to_pin,
        });
    }
}

/// The canonical LIF graph — mirrors `api::formula::tests::lif_program`.
///
/// Absolute-FREEZE refractory: `CmpGt(s1, 0)` builds the is-refractory mask,
/// `Blend(mask, s0, v_raw)` keeps the frozen v, and `FireReset` on the frozen
/// (sub-threshold) v cannot spike — no breakthrough at high currents.
fn default_lif_graph() -> (Vec<GNode>, Vec<GEdge>) {
    let mut b = PBuilder::new();
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

    b.wire(state, 1, is_refr, 0); // s1 → cmp.a
    b.wire(c_zero, 0, is_refr, 1); // 0 → cmp.b
    b.wire(state, 0, leak, 0); // s0 → leak.v
    b.wire(stim, 0, leak, 1); // I → leak.i
    b.wire(c_tau, 0, leak, 2); // inv_tau → leak.inv_tau
    b.wire(c_rest, 0, leak, 3); // v_rest → leak.v_rest
    b.wire(is_refr, 0, gated, 0); // mask
    b.wire(state, 0, gated, 1); // a = frozen s0
    b.wire(leak, 0, gated, 2); // b = v_raw
    b.wire(gated, 0, fire, 0); // v → fire.v
    b.wire(c_thresh, 0, fire, 1); // thresh → fire.thresh
    b.wire(c_reset, 0, fire, 2); // reset → fire.reset
    b.wire(state, 1, timer, 0); // s1 → timer.refr
    b.wire(c_tref, 0, timer, 1); // tau_ref → timer.tau_ref
    b.wire(state, 0, scope, 0); // s0 → scope.x
    (b.nodes, b.edges)
}

/// Morris-Lecar (1981) 2D conductance model, assembled from ops only.
///
/// ```text
/// m_inf = sigmoid(V, V1, V2/2)          w_inf = sigmoid(V, V3, V4/2)
/// I_Ca = g_Ca·m_inf·(V−E_Ca)            I_K = g_K·w·(V−E_K)
/// I_L  = g_L·(V−E_L)
/// dV   = (I_inj − I_Ca − I_K − I_L) · inv_C · dt
/// dw   = phi·(w_inf − w)  (GateStep)
/// ```
///
/// Note: the `w` gate is updated *before* `I_K` reads it (staggered
/// semi-implicit update — gate writes S1, then the ohmic currents read the
/// fresh value). Deterministic and oscillatory for the default params.
/// A true explicit-Euler read of the *old* `w` would need a snapshot
/// register, which the current op set can't express without an extra node;
/// in program order `gate_w` always runs before `mul_gk`, so both `State.s1`
/// and the gate output resolve to the same post-update S1.
fn ml_graph() -> (Vec<GNode>, Vec<GEdge>) {
    let mut b = PBuilder::new();
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

    // sigmoids read s0 (old v)
    let t_m = b.alloc_t();
    let t_winf = b.alloc_t();
    let sig_m = b.node(OpKind::Sigmoid, 220.0, 40.0, t_m, -1.2, 9.0);
    let sig_w = b.node(OpKind::Sigmoid, 220.0, 130.0, t_winf, 12.0, 8.7);

    // w gate must run before I_K reads s1
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
    // Leak with v_rest=0 and i=(v + I_net) ⇒ v' = v + I_net·inv_C·dt
    let leak = b.node(OpKind::Leak, 1020.0, 130.0, S0, 0.0, 0.0);

    b.wire(state, 0, sig_m, 0);
    b.wire(state, 0, sig_w, 0);

    b.wire(sig_m, 0, mul_gca, 0);
    b.wire(g_ca, 0, mul_gca, 1);
    b.wire(mul_gca, 0, i_ca, 0);
    b.wire(state, 0, i_ca, 1);
    b.wire(e_ca, 0, i_ca, 2);

    b.wire(gate_w, 0, mul_gk, 0); // w (s1, after gate) → mul
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

    // w gate: g=s1(prev w), g_inf=w_inf, inv_tau=phi
    b.wire(state, 1, gate_w, 0);
    b.wire(sig_w, 0, gate_w, 1);
    b.wire(phi, 0, gate_w, 2);

    (b.nodes, b.edges)
}

/// HH-lite: 3 sigmoid gates (m/h/n), m³h and n⁴ gating, ohmic currents.
///
/// Same staggered update as the ML preset: gates write s1..s3 first, then
/// the m³h / n⁴ multipliers read the fresh gate values and the ohmic
/// currents read the old `v` (s0 is written last by `Leak`).
fn hh_lite_graph() -> (Vec<GNode>, Vec<GEdge>) {
    let mut b = PBuilder::new();
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

    // gates: m fast, h slow, n medium
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

    // m³h
    let m2 = b.tmp(OpKind::Mul, 620.0, 40.0, 0.0, 0.0);
    let m3 = b.tmp(OpKind::Mul, 620.0, 100.0, 0.0, 0.0);
    let m3h = b.tmp(OpKind::Mul, 620.0, 160.0, 0.0, 0.0);
    let gna_m = b.tmp(OpKind::Mul, 620.0, 220.0, 0.0, 0.0);
    // n⁴
    let n2 = b.tmp(OpKind::Mul, 620.0, 280.0, 0.0, 0.0);
    let n4 = b.tmp(OpKind::Mul, 620.0, 340.0, 0.0, 0.0);
    let gk_n = b.tmp(OpKind::Mul, 620.0, 400.0, 0.0, 0.0);

    let i_na = b.tmp(OpKind::Ohmic, 820.0, 40.0, 0.0, 0.0);
    let i_k = b.tmp(OpKind::Ohmic, 820.0, 140.0, 0.0, 0.0);
    let i_l = b.tmp(OpKind::Ohmic, 820.0, 240.0, 0.0, 0.0);

    let sum1 = b.tmp(OpKind::Add, 1020.0, 40.0, 0.0, 0.0);
    let sum2 = b.tmp(OpKind::Add, 1020.0, 130.0, 0.0, 0.0);
    let neg = b.tmp(OpKind::Mul, 1020.0, 220.0, 0.0, 0.0);
    let net = b.tmp(OpKind::Add, 1020.0, 310.0, 0.0, 0.0);

    let drive = b.tmp(OpKind::Add, 1220.0, 40.0, 0.0, 0.0);
    let leak = b.node(OpKind::Leak, 1220.0, 130.0, S0, 0.0, 0.0);

    // gate inputs (g, g_inf, inv_tau)
    b.wire(state, 1, gate_m, 0);
    b.wire(sig_m, 0, gate_m, 1);
    b.wire(tau_m, 0, gate_m, 2);
    b.wire(state, 2, gate_h, 0);
    b.wire(sig_h, 0, gate_h, 1);
    b.wire(tau_h, 0, gate_h, 2);
    b.wire(state, 3, gate_n, 0);
    b.wire(sig_n, 0, gate_n, 1);
    b.wire(tau_n, 0, gate_n, 2);

    // m³h from integrated states s1,s2
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

    // n⁴ from s3
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

impl Default for NeuronBuilder {
    fn default() -> Self {
        let (nodes, edges) = default_lif_graph();
        let mut app = Self {
            nodes,
            edges,
            drag: Drag::None,
            selected: None,
            selected_edge: None,
            ctx_menu: None,
            undo: Vec::new(),
            redo: Vec::new(),
            pan: Vec2::ZERO,
            zoom: 1.0,
            snap: true,
            show_minimap: true,
            show_plot: true,
            hover_node: None,
            hover_edge: None,
            hover_tip: None,
            running: true,
            steps_per_frame: 4,
            dt: 0.5,
            i_syn: 30.0,
            v_init: -70.0,
            init_gates: [0.0; 3],
            regs: [0.0; N_REGS],
            t: 0.0,
            history: VecDeque::new(),
            spike_times: Vec::new(),
            total_spikes: 0,
            prev_trace: 0.0,
            spike_thresh: -54.0,
            window_ms: 500.0,
            trace: 0,
            filename: "neuron_graph.json".to_string(),
            status: "Drag from orange pin → cyan pin · scroll = zoom · minimap: click/drag to navigate · right-click canvas for ops".to_string(),
        };
        app.reset_sim();
        app
    }
}

// ─── Serialization ──────────────────────────────────────────────
#[derive(serde::Serialize, serde::Deserialize)]
struct FileGraph {
    version: u32,
    nodes: Vec<FileNode>,
    edges: Vec<FileEdge>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FileNode {
    kind: u8,
    x: f32,
    y: f32,
    out_reg: usize,
    c: f32,
    c2: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FileEdge {
    from: usize,
    from_pin: usize,
    to: usize,
    to_pin: usize,
}

/// Source register feeding `node.pin` (State nodes → s0..s3).
///
/// Bounds-guarded: stale edges (from an old binary, a hand-edited JSON, or
/// an undo snapshot) are skipped instead of panicking.
fn input_reg_of(nodes: &[GNode], edges: &[GEdge], node: usize, pin: usize) -> usize {
    for e in edges {
        if e.to == node && e.to_pin == pin && e.from < nodes.len() {
            return if nodes[e.from].kind == OpKind::State {
                S0 + e.from_pin
            } else {
                nodes[e.from].out_reg
            };
        }
    }
    S0 // disconnected pin falls back to the membrane state
}

/// Resolve a node graph into a flat op program (shader order).
///
/// State/Scope nodes are graph-only and never emitted; every other node
/// becomes one [`OpNode`]. Register indices are clamped to `N_REGS` so a
/// stale graph degrades gracefully instead of panicking in the interpreter.
fn graph_to_program(nodes: &[GNode], edges: &[GEdge]) -> Vec<OpNode> {
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

impl NeuronBuilder {
    // ── Graph → program ────────────────────────────────────────

    /// Source register feeding `node.pin` (State nodes → s0..s3).
    fn input_reg(&self, node: usize, pin: usize) -> usize {
        input_reg_of(&self.nodes, &self.edges, node, pin)
    }

    /// Register sampled by the scope (first Scope node, else s0).
    fn scope_reg(&self) -> usize {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.kind == OpKind::Scope {
                return self.input_reg(i, 0);
            }
        }
        S0
    }

    /// Register the oscilloscope plots (0 = Scope node, 1..=4 = s0..s3).
    fn trace_reg(&self) -> usize {
        match self.trace {
            1 => S0,
            2 => S1,
            3 => S2,
            4 => S3,
            _ => self.scope_reg(),
        }
    }

    fn build_program(&self) -> Vec<OpNode> {
        graph_to_program(&self.nodes, &self.edges)
    }

    // ── Simulation ─────────────────────────────────────────────

    fn step_sim(&mut self) {
        let prog = self.build_program();
        formula::run_program::<f32>(&prog, &mut self.regs, self.dt, self.i_syn);
        self.t += self.dt as f64;

        // Universal spike detection: upward crossing of the threshold.
        let reg = self.trace_reg();
        let cur = self.regs[reg];
        if self.prev_trace < self.spike_thresh && cur >= self.spike_thresh {
            self.total_spikes += 1;
            self.spike_times.push(self.t);
            if self.spike_times.len() > 128 {
                self.spike_times.drain(..self.spike_times.len() - 128);
            }
        }
        self.prev_trace = cur;

        self.history.push_back((self.t, cur as f64));
        while self.history.len() > SCOPE_POINTS {
            self.history.pop_front();
        }
    }

    fn reset_sim(&mut self) {
        self.regs = [0.0; N_REGS];
        self.regs[S0] = self.v_init;
        self.regs[S1] = self.init_gates[0];
        self.regs[S2] = self.init_gates[1];
        self.regs[S3] = self.init_gates[2];
        self.t = 0.0;
        self.history.clear();
        self.spike_times.clear();
        self.total_spikes = 0;
        self.prev_trace = self.v_init;
    }

    // ── Undo / redo ────────────────────────────────────────────

    fn push_undo(&mut self) {
        self.undo.push((self.nodes.clone(), self.edges.clone()));
        if self.undo.len() > 50 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn undo(&mut self) {
        if let Some((n, e)) = self.undo.pop() {
            self.redo.push((self.nodes.clone(), self.edges.clone()));
            self.nodes = n;
            self.edges = e;
            self.selected = None;
            self.selected_edge = None;
            self.drag = Drag::None;
            self.running = false;
            self.reset_sim();
        }
    }

    fn redo(&mut self) {
        if let Some((n, e)) = self.redo.pop() {
            self.undo.push((self.nodes.clone(), self.edges.clone()));
            self.nodes = n;
            self.edges = e;
            self.selected = None;
            self.selected_edge = None;
            self.drag = Drag::None;
            self.running = false;
            self.reset_sim();
        }
    }

    // ── Graph editing ──────────────────────────────────────────

    fn alloc_reg(&self) -> usize {
        (TEMP0..N_REGS)
            .find(|&r| !self.nodes.iter().any(|n| n.out_reg == r))
            .unwrap_or(TEMP0)
    }

    fn add_node_at(&mut self, kind: OpKind, world: Vec2) {
        self.push_undo();
        let out_reg = match kind {
            OpKind::State | OpKind::Scope => 0,
            OpKind::Stimulus => I_SYN,
            _ => self.alloc_reg(),
        };
        let pos = if self.snap {
            pos2(
                (world.x / GRID).round() * GRID,
                (world.y / GRID).round() * GRID,
            )
        } else {
            pos2(world.x, world.y)
        };
        let idx = self.nodes.len();
        self.nodes.push(GNode {
            kind,
            pos,
            out_reg,
            c: 0.0,
            c2: 0.0,
        });
        self.selected = Some(idx);
        self.selected_edge = None;
        self.running = false;
    }

    fn add_node(&mut self, kind: OpKind) {
        let n = self.nodes.len() as f32;
        let world = self.screen_to_world(vec2(80.0 + (n % 5.0) * 40.0, 80.0 + (n % 6.0) * 44.0));
        self.add_node_at(kind, world);
    }

    fn delete_node(&mut self, idx: usize) {
        self.push_undo();
        self.edges.retain(|e| e.from != idx && e.to != idx);
        self.nodes.remove(idx);
        for e in &mut self.edges {
            if e.from > idx {
                e.from -= 1;
            }
            if e.to > idx {
                e.to -= 1;
            }
        }
        match self.selected {
            Some(s) if s == idx => self.selected = None,
            Some(s) if s > idx => self.selected = Some(s - 1),
            _ => {}
        }
        // A drag wire may reference the deleted node — abort it.
        self.drag = Drag::None;
        self.selected_edge = None;
        self.running = false;
    }

    fn duplicate_node(&mut self, idx: usize) {
        self.push_undo();
        let node = self.nodes[idx].clone();
        let out_reg = match node.kind {
            OpKind::State | OpKind::Scope => node.out_reg,
            OpKind::Stimulus => I_SYN,
            _ => self.alloc_reg(),
        };
        let new_idx = self.nodes.len();
        self.nodes.push(GNode {
            pos: node.pos + vec2(50.0, 50.0),
            out_reg,
            ..node
        });
        // copy incoming wires
        let incoming: Vec<GEdge> = self.edges.iter().filter(|e| e.to == idx).cloned().collect();
        for mut e in incoming {
            e.to = new_idx;
            self.edges.push(e);
        }
        self.selected = Some(new_idx);
        self.selected_edge = None;
        self.running = false;
    }

    fn delete_edge(&mut self, idx: usize) {
        self.push_undo();
        self.edges.remove(idx);
        self.drag = Drag::None;
        self.selected_edge = None;
        self.running = false;
    }

    fn add_edge(&mut self, from: usize, from_pin: usize, to: usize, to_pin: usize) {
        if from == to {
            return;
        }
        self.push_undo();
        self.edges.retain(|e| !(e.to == to && e.to_pin == to_pin));
        self.edges.push(GEdge {
            from,
            from_pin,
            to,
            to_pin,
        });
        self.running = false;
    }

    /// Label of the edge feeding `node.pin`, if any.
    fn edge_source(&self, node: usize, pin: usize) -> Option<String> {
        for e in &self.edges {
            if e.to == node && e.to_pin == pin && e.from < self.nodes.len() {
                let from = &self.nodes[e.from];
                return Some(if from.kind == OpKind::State {
                    format!("s{}", e.from_pin)
                } else if from.kind == OpKind::Stimulus {
                    "I".to_string()
                } else {
                    reg_name(from.out_reg)
                });
            }
        }
        None
    }

    /// Tooltip text for whatever is under the pointer (pin or wire).
    fn hover_tip_text(&self, p: Vec2) -> Option<String> {
        if let Some((n, pin)) = self.hit_out_pin(p) {
            let spec = spec_of(self.nodes[n].kind);
            let reg = if self.nodes[n].kind == OpKind::State {
                format!("s{pin}")
            } else if self.nodes[n].kind == OpKind::Stimulus {
                "I".to_string()
            } else {
                reg_name(self.nodes[n].out_reg)
            };
            Some(format!("{} · {} → {}", spec.name, spec.outs[pin], reg))
        } else if let Some((n, pin)) = self.hit_in_pin(p) {
            let spec = spec_of(self.nodes[n].kind);
            let src = self.edge_source(n, pin).unwrap_or_else(|| "—".to_string());
            Some(format!("{} · {} ← {}", spec.name, spec.ins[pin], src))
        } else if let Some(ei) = self.hover_edge {
            if ei < self.edges.len() {
                let e = &self.edges[ei];
                if e.from < self.nodes.len() && e.to < self.nodes.len() {
                    let from = spec_of(self.nodes[e.from].kind).name;
                    let to = spec_of(self.nodes[e.to].kind).name;
                    return Some(format!("wire · {from} → {to}  (Del to remove)"));
                }
            }
            None
        } else {
            None
        }
    }

    // ── View transform ─────────────────────────────────────────

    /// World → screen (relative to canvas origin).
    fn to_screen(&self, w: Vec2) -> Vec2 {
        w * self.zoom + self.pan
    }

    /// Screen (relative to canvas origin) → world.
    fn screen_to_world(&self, s: Vec2) -> Vec2 {
        (s - self.pan) / self.zoom
    }

    // ── Node geometry ──────────────────────────────────────────

    fn node_rect(&self, i: usize) -> Rect {
        let node = &self.nodes[i];
        Rect::from_min_size(node.pos, node_size_of(node.kind))
    }

    fn in_pin_pos(&self, i: usize, pin: usize) -> Vec2 {
        let node = &self.nodes[i];
        vec2(
            node.pos.x,
            node.pos.y + HEADER_H + (pin as f32 + 0.5) * PIN_H,
        )
    }

    fn out_pin_pos(&self, i: usize, pin: usize) -> Vec2 {
        let node = &self.nodes[i];
        vec2(
            node.pos.x + NODE_W,
            node.pos.y + HEADER_H + (pin as f32 + 0.5) * PIN_H,
        )
    }

    fn hit_node(&self, p: Vec2) -> Option<usize> {
        (0..self.nodes.len()).find(|&i| self.node_rect(i).contains(pos2(p.x, p.y)))
    }

    fn hit_out_pin(&self, p: Vec2) -> Option<(usize, usize)> {
        for i in 0..self.nodes.len() {
            for pin in 0..spec_of(self.nodes[i].kind).outs.len() {
                if (self.out_pin_pos(i, pin) - p).length() < HIT_R {
                    return Some((i, pin));
                }
            }
        }
        None
    }

    fn hit_in_pin(&self, p: Vec2) -> Option<(usize, usize)> {
        for i in 0..self.nodes.len() {
            for pin in 0..spec_of(self.nodes[i].kind).ins.len() {
                if (self.in_pin_pos(i, pin) - p).length() < HIT_R {
                    return Some((i, pin));
                }
            }
        }
        None
    }

    /// Distance from `p` to the cubic bezier actually drawn by `draw_wire`.
    fn dist_to_edge(&self, e: &GEdge, p: Vec2) -> f32 {
        if e.from >= self.nodes.len() || e.to >= self.nodes.len() {
            return f32::INFINITY; // stale edge — never hittable
        }
        let a = self.out_pin_pos(e.from, e.from_pin);
        let b = self.in_pin_pos(e.to, e.to_pin);
        let pts = bezier_polyline(pos2(a.x, a.y), pos2(b.x, b.y), 16);
        pts.windows(2)
            .map(|w| dist_to_segment(p, w[0].to_vec2(), w[1].to_vec2()))
            .fold(f32::INFINITY, f32::min)
    }

    fn hit_edge(&self, p: Vec2) -> Option<usize> {
        let tol = 8.0 / self.zoom;
        (0..self.edges.len())
            .filter(|&i| self.dist_to_edge(&self.edges[i], p) < tol)
            .min_by(|&a, &b| {
                self.dist_to_edge(&self.edges[a], p)
                    .partial_cmp(&self.dist_to_edge(&self.edges[b], p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // ── Minimap (bird's-eye view + click-to-navigate) ──────────

    /// World-space bounding box of the whole graph (+ padding).
    fn graph_world_bounds(&self) -> (Vec2, Vec2) {
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

    /// Screen rect of the minimap overlay (bottom-right of the canvas).
    fn minimap_rect(&self, canvas_size: Vec2) -> Rect {
        let mm = vec2(190.0, 132.0);
        Rect::from_min_size(
            vec2(canvas_size.x - mm.x - 14.0, canvas_size.y - mm.y - 14.0).to_pos2(),
            mm,
        )
    }

    /// (rect, scale, world-min) — maps world coords onto the minimap rect.
    fn minimap_map(&self, canvas_size: Vec2) -> (Rect, f32, Vec2) {
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

    /// Center the view on the world point under the cursor inside the minimap.
    fn minimap_pan_to(&mut self, canvas_size: Vec2, pointer_screen: Option<Vec2>) {
        let (grect, scale, gmn) = self.minimap_map(canvas_size);
        if let Some(s) = pointer_screen {
            let wp = gmn + (s - grect.min.to_vec2()) / scale;
            self.pan = canvas_size * 0.5 - wp * self.zoom;
        }
    }

    fn draw_minimap(&self, painter: &egui::Painter, canvas_size: Vec2) {
        let mm = self.minimap_rect(canvas_size);
        // мягкая тень под миникартой + muted бордер под тему
        painter.rect_filled(
            mm.expand(3.0),
            CornerRadius::same(8),
            Color32::from_rgba_unmultiplied(0, 0, 0, 40),
        );
        painter.rect_filled(
            mm,
            CornerRadius::same(6),
            Color32::from_rgba_unmultiplied(22, 22, 32, 225),
        );
        painter.rect_stroke(
            mm,
            CornerRadius::same(6),
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(70, 70, 95, 140)),
            StrokeKind::Inside,
        );
        // caption stays visible even for an empty graph (after Clear)
        painter.text(
            mm.min + vec2(6.0, 5.0),
            Align2::LEFT_TOP,
            "🗺 minimap — click / drag to navigate",
            FontId::proportional(10.0),
            Color32::from_rgba_unmultiplied(170, 170, 200, 170),
        );
        if self.nodes.is_empty() {
            return;
        }
        let (grect, scale, gmn) = self.minimap_map(canvas_size);
        for n in &self.nodes {
            let sz = node_size_of(n.kind);
            let p = grect.min + (n.pos.to_vec2() - gmn) * scale;
            painter.rect_filled(
                Rect::from_min_size(p, vec2(sz.x * scale, sz.y * scale)),
                CornerRadius::same(1),
                cat_color(cat_of(n.kind)),
            );
        }
        // viewport rectangle (what is currently on screen)
        let w0 = self.screen_to_world(Vec2::ZERO);
        let w1 = self.screen_to_world(canvas_size);
        let v0 = grect.min + (w0 - gmn) * scale;
        let v1 = grect.min + (w1 - gmn) * scale;
        let vr = Rect::from_min_max(v0, v1);
        painter.rect_filled(
            vr,
            CornerRadius::same(1),
            Color32::from_rgba_unmultiplied(255, 255, 255, 14),
        );
        painter.rect_stroke(
            vr,
            CornerRadius::same(1),
            Stroke::new(1.0_f32, Color32::from_rgb(255, 255, 255)),
            StrokeKind::Inside,
        );
    }

    // ── Rendering ──────────────────────────────────────────────

    fn draw_grid(&self, painter: &egui::Painter, origin: Pos2, size: Vec2) {
        // Приглушённая сетка — как в Blender (тонкие пунктирные точки, не линии)
        let color = Color32::from_rgba_unmultiplied(65, 65, 80, 28);
        let dot = Color32::from_rgba_unmultiplied(75, 75, 92, 42);
        let w_min = self.screen_to_world(Vec2::ZERO);
        let w_max = self.screen_to_world(size);
        // линии — очень тонкие и прозрачные
        let mut x = (w_min.x / GRID).floor() * GRID;
        while x <= w_max.x {
            let sx = origin.x + self.to_screen(vec2(x, 0.0)).x;
            painter.line_segment(
                [pos2(sx, origin.y), pos2(sx, origin.y + size.y)],
                Stroke::new(0.7_f32, color),
            );
            x += GRID;
        }
        let mut y = (w_min.y / GRID).floor() * GRID;
        while y <= w_max.y {
            let sy = origin.y + self.to_screen(vec2(0.0, y)).y;
            painter.line_segment(
                [pos2(origin.x, sy), pos2(origin.x + size.x, sy)],
                Stroke::new(0.7_f32, color),
            );
            y += GRID;
        }
        // точки на пересечениях — чуть заметнее, как в Houdini
        let mut x = (w_min.x / GRID).floor() * GRID;
        while x <= w_max.x {
            let mut y = (w_min.y / GRID).floor() * GRID;
            while y <= w_max.y {
                let p = origin + self.to_screen(vec2(x, y));
                painter.circle_filled(p, 1.0 * self.zoom, dot);
                y += GRID;
            }
            x += GRID;
        }
    }

    fn draw_canvas(
        &self,
        painter: &egui::Painter,
        origin: Pos2,
        size: Vec2,
        pointer_world: Option<Vec2>,
    ) {
        self.draw_grid(painter, origin, size);

        // ── wires: cubic bezier, muted проф + outer glow (Houdini style) ──
        for (i, e) in self.edges.iter().enumerate() {
            if e.from >= self.nodes.len() || e.to >= self.nodes.len() {
                continue; // stale edge from an old snapshot — skip, don't panic
            }
            let a = origin + self.to_screen(self.out_pin_pos(e.from, e.from_pin));
            let b = origin + self.to_screen(self.in_pin_pos(e.to, e.to_pin));
            let base = cat_color(cat_of(self.nodes[e.from].kind));
            let (color, width) = if self.selected_edge == Some(i) {
                (Color32::from_rgb(212, 175, 90), 3.0) // muted gold, не неон 255,230,120
            } else if self.hover_edge == Some(i) {
                (base.gamma_multiply(1.45), 2.8)
            } else {
                (base.gamma_multiply(1.05), 1.9) // чуть тоньше и мягче
            };
            // outer glow — размытый след под проводом
            let glow = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 32);
            draw_wire(painter, a, b, glow, width + 4.0 * self.zoom);
            draw_wire(painter, a, b, color, width * self.zoom);
            // animated “data pulse” — мягче и меньше
            if self.running {
                let phase = (self.t * 0.010 + i as f64 * 0.618).fract() as f32;
                let pulse = bezier_eval(a, b, phase);
                painter.circle_filled(pulse, 3.2 * self.zoom, glow);
                painter.circle_filled(pulse, 1.8 * self.zoom, color.gamma_multiply(1.4));
            }
        }
        // wire being dragged from an output pin
        if let Drag::Wire { from, from_pin } = self.drag {
            if from < self.nodes.len() {
                if let Some(p) = pointer_world {
                    let a = origin + self.to_screen(self.out_pin_pos(from, from_pin));
                    let b = origin + self.to_screen(p);
                    draw_wire(painter, a, b, Color32::from_rgb(255, 200, 80), 2.5);
                }
            }
        }

        // ── nodes: shadow, header strip, hover/selection glow ──
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let spec = spec_of(node.kind);
            let cat = cat_of(node.kind);
            let accent = cat_color(cat);
            let rect_world = self.node_rect(i);
            let rect = Rect::from_min_size(
                origin + self.to_screen(rect_world.min.to_vec2()),
                rect_world.size() * self.zoom,
            );
            let rounding = CornerRadius::same((8.0 * self.zoom) as u8);

            // мягкая тень в 2 слоя — объём как в Blender (не плоский «штамп»)
            let shadow_r = CornerRadius::same((10.0 * self.zoom) as u8);
            painter.rect_filled(
                rect.translate(vec2(0.0, 6.0 * self.zoom))
                    .expand(4.0 * self.zoom),
                shadow_r,
                Color32::from_rgba_unmultiplied(0, 0, 0, 32),
            );
            painter.rect_filled(
                rect.translate(vec2(0.0, 3.0 * self.zoom)),
                rounding,
                Color32::from_rgba_unmultiplied(0, 0, 0, 70),
            );

            // body — чуть светлее + subtle inner highlight сверху
            let bg = if self.selected == Some(i) {
                Color32::from_rgb(42, 42, 66)
            } else {
                Color32::from_rgb(30, 30, 48)
            };
            painter.rect_filled(rect, rounding, bg);
            // inner highlight — 1px сверху, 14 alpha
            painter.rect_filled(
                Rect::from_min_max(rect.min, pos2(rect.max.x, rect.min.y + 1.5 * self.zoom)),
                CornerRadius {
                    nw: rounding.nw,
                    ne: rounding.ne,
                    sw: 0,
                    se: 0,
                },
                Color32::from_rgba_unmultiplied(255, 255, 255, 14),
            );

            // header strip: category tint + accent line + highlight
            let header_h = HEADER_H * self.zoom;
            let header = Rect::from_min_max(rect.min, pos2(rect.max.x, rect.min.y + header_h));
            painter.rect_filled(
                header,
                CornerRadius {
                    nw: rounding.nw,
                    ne: rounding.ne,
                    sw: 0,
                    se: 0,
                },
                cat_tint(cat),
            );
            painter.rect_filled(
                Rect::from_min_max(rect.min, pos2(rect.max.x, rect.min.y + 2.0 * self.zoom)),
                CornerRadius::ZERO,
                accent,
            );
            // header inner highlight
            painter.rect_filled(
                Rect::from_min_max(
                    pos2(rect.min.x, rect.min.y + 2.0 * self.zoom),
                    pos2(rect.max.x, rect.min.y + 3.2 * self.zoom),
                ),
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(255, 255, 255, 18),
            );
            painter.text(
                pos2(rect.min.x + 8.0 * self.zoom, rect.min.y + header_h * 0.5),
                Align2::LEFT_CENTER,
                format!("{}  {}", cat_icon(cat), spec.name),
                FontId::proportional(12.5 * self.zoom),
                Color32::from_rgb(235, 235, 245),
            );

            // border: muted gold = selected, category = hovered, soft gray = idle
            let border = if self.selected == Some(i) {
                Color32::from_rgb(212, 175, 90) // muted gold, не неон 255,205,60
            } else if self.hover_node == Some(i) {
                accent.gamma_multiply(1.3)
            } else {
                Color32::from_rgba_unmultiplied(80, 80, 105, 180)
            };
            let bw = if self.selected == Some(i) { 1.8 } else { 1.0 };
            painter.rect_stroke(
                rect,
                rounding,
                Stroke::new(bw * self.zoom, border),
                StrokeKind::Inside,
            );

            // пины — приглушённые, с outer glow при ховере/коннекте (Houdini style)
            let pr = PIN_R * self.zoom;
            for (pin, label) in spec.ins.iter().enumerate() {
                let c = origin + self.to_screen(self.in_pin_pos(i, pin));
                let col = Color32::from_rgb(110, 175, 215); // muted cyan, был 90,200,255 — неон
                let connected = self.edges.iter().any(|e| e.to == i && e.to_pin == pin);
                let pin_w = self.in_pin_pos(i, pin);
                let hovered = pointer_world
                    .map(|pw| (pw - pin_w).length() < HIT_R)
                    .unwrap_or(false);
                if connected {
                    // outer glow
                    painter.circle_filled(c, pr + 3.0 * self.zoom, col.gamma_multiply(0.18));
                    painter.circle_filled(c, pr + 1.5 * self.zoom, col);
                    painter.circle_filled(c, pr, Color32::from_rgb(22, 24, 38));
                } else {
                    if hovered {
                        painter.circle_filled(c, pr + 3.0 * self.zoom, col.gamma_multiply(0.22));
                    }
                    painter.circle_stroke(
                        c,
                        pr,
                        Stroke::new(
                            1.4 * self.zoom,
                            col.gamma_multiply(if hovered { 1.0 } else { 0.55 }),
                        ),
                    );
                    painter.circle_filled(
                        c,
                        pr * 0.32,
                        col.gamma_multiply(if hovered { 0.9 } else { 0.45 }),
                    );
                }
                painter.text(
                    c + vec2(-7.0 * self.zoom, 0.0),
                    Align2::RIGHT_CENTER,
                    *label,
                    FontId::proportional(10.5 * self.zoom),
                    if hovered {
                        Color32::from_rgb(195, 205, 230)
                    } else {
                        Color32::from_rgb(155, 165, 185)
                    },
                );
            }
            for (pin, label) in spec.outs.iter().enumerate() {
                let c = origin + self.to_screen(self.out_pin_pos(i, pin));
                let col = Color32::from_rgb(210, 145, 70); // muted amber, был 255,170,60
                let connected = self.edges.iter().any(|e| e.from == i && e.from_pin == pin);
                let pin_w = self.out_pin_pos(i, pin);
                let hovered = pointer_world
                    .map(|pw| (pw - pin_w).length() < HIT_R)
                    .unwrap_or(false);
                if connected {
                    painter.circle_filled(c, pr + 3.0 * self.zoom, col.gamma_multiply(0.18));
                    painter.circle_filled(c, pr + 1.5 * self.zoom, col);
                    painter.circle_filled(c, pr, Color32::from_rgb(22, 24, 38));
                } else {
                    if hovered {
                        painter.circle_filled(c, pr + 3.0 * self.zoom, col.gamma_multiply(0.22));
                    }
                    painter.circle_stroke(
                        c,
                        pr,
                        Stroke::new(
                            1.4 * self.zoom,
                            col.gamma_multiply(if hovered { 1.0 } else { 0.55 }),
                        ),
                    );
                    painter.circle_filled(
                        c,
                        pr * 0.32,
                        col.gamma_multiply(if hovered { 0.9 } else { 0.45 }),
                    );
                }
                let out_label = match node.kind {
                    OpKind::State => (*label).to_string(),
                    OpKind::Stimulus => "I".to_string(),
                    _ => reg_name(node.out_reg),
                };
                painter.text(
                    c + vec2(7.0 * self.zoom, 0.0),
                    Align2::LEFT_CENTER,
                    out_label,
                    FontId::proportional(11.0 * self.zoom),
                    Color32::from_rgb(215, 180, 120),
                );
            }
        }

        // minimap floats above everything
        if self.show_minimap {
            self.draw_minimap(painter, size);
        }
    }

    // ── Panels ─────────────────────────────────────────────────

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("🧠 Neuron Builder").strong().size(16.0));
            ui.separator();
            if ui
                .button(if self.running { "⏸ Pause" } else { "▶ Run" })
                .clicked()
            {
                self.running = !self.running;
            }
            if ui.button("↺ Reset").clicked() {
                self.reset_sim();
            }
            if ui.button("⟳ Fit view").clicked() {
                self.pan = vec2(20.0, 20.0);
                self.zoom = 1.0;
            }
            ui.checkbox(&mut self.show_minimap, "🗺 minimap");
            ui.checkbox(&mut self.show_plot, "📈 scope");
            ui.separator();
            ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=200).text("steps/frame"));
            ui.add(egui::Slider::new(&mut self.dt, 0.01..=2.0).text("dt ms"));
            ui.add(egui::Slider::new(&mut self.i_syn, -50.0..=100.0).text("I_syn"));
            ui.add(egui::Slider::new(&mut self.v_init, -100.0..=50.0).text("v_init"));
            ui.checkbox(&mut self.snap, "snap");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Presets:");
            if ui.button("LIF").clicked() {
                self.dt = 0.5;
                self.i_syn = 30.0;
                self.v_init = -70.0;
                self.init_gates = [0.0; 3];
                self.spike_thresh = -54.0;
                self.load_graph(default_lif_graph());
            }
            if ui.button("Morris-Lecar").clicked() {
                self.dt = 0.05;
                self.i_syn = 50.0;
                self.v_init = -60.0;
                // w∞(-60) ≈ 0 — w gate starts at rest.
                self.init_gates = [0.0003, 0.0, 0.0];
                self.spike_thresh = 0.0;
                self.load_graph(ml_graph());
            }
            if ui.button("HH-lite").clicked() {
                self.dt = 0.05;
                self.i_syn = 10.0;
                self.v_init = -65.0;
                // Resting squid gates at v=-65: m≈0.015, h≈0.70, n≈0.18.
                // A cold start (all 0) leaves h inactivated — the model
                // can never regenerate a spike.
                self.init_gates = [0.0153, 0.697, 0.182];
                self.spike_thresh = 0.0;
                self.load_graph(hh_lite_graph());
            }
            ui.separator();
            ui.label("Trace:");
            let trace_names = ["Scope", "s0", "s1", "s2", "s3"];
            let old_trace = self.trace;
            egui::ComboBox::from_id_salt("trace")
                .selected_text(trace_names[self.trace])
                .show_ui(ui, |ui| {
                    for (i, name) in trace_names.iter().enumerate() {
                        ui.selectable_value(&mut self.trace, i, *name);
                    }
                });
            if self.trace != old_trace {
                // avoid a false spike from a stale crossing baseline
                self.prev_trace = self.regs[self.trace_reg()];
            }
            ui.add(egui::Slider::new(&mut self.spike_thresh, -100.0..=50.0).text("spike mV"));
            ui.add(egui::Slider::new(&mut self.window_ms, 50.0..=4000.0).text("window ms"));
            ui.separator();
            ui.label(format!("t = {:.1} ms", self.t));
            ui.label(format!("⚡ {}", self.total_spikes));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("File:");
            ui.add(egui::TextEdit::singleline(&mut self.filename).desired_width(180.0));
            if ui.button("💾 Save").clicked() {
                self.save_graph();
            }
            if ui.button("📂 Load").clicked() {
                self.load_file();
            }
            ui.separator();
            ui.label("Ops:");
            for &kind in palette_kinds() {
                if ui.button(spec_of(kind).name).clicked() {
                    self.add_node(kind);
                }
            }
            if ui.button("🗑 Clear").clicked() {
                self.push_undo();
                self.nodes.clear();
                self.edges.clear();
                self.selected = None;
                self.selected_edge = None;
                self.running = false;
                self.reset_sim();
            }
        });
        ui.separator();
    }

    fn status_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&self.status).weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("zoom {:.0}%", self.zoom * 100.0));
                if self.selected_edge.is_some() {
                    ui.label("wire selected — Del to remove");
                } else if let Some(s) = self.selected {
                    ui.label(format!("node {} ({})", s, spec_of(self.nodes[s].kind).name));
                }
            });
        });
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        if let Some(ei) = self.selected_edge {
            if let Some(e) = self.edges.get(ei) {
                let from = spec_of(self.nodes[e.from].kind).name;
                let to = spec_of(self.nodes[e.to].kind).name;
                ui.label(format!("Wire: {from}.{} → {to}.{}", e.from_pin, e.to_pin));
                if ui.button("🗑 Delete wire").clicked() {
                    self.delete_edge(ei);
                }
                ui.separator();
            }
        }
        match self.selected {
            Some(i) => self.inspector_node(ui, i),
            None if self.selected_edge.is_none() => {
                ui.weak("Select a node (left-click), a wire, or drag from an orange output pin to a cyan input pin.");
            }
            None => {}
        }
        ui.separator();
        egui::CollapsingHeader::new("Program (opcodes)")
            .default_open(true)
            .show(ui, |ui| {
                let prog = self.build_program();
                for (k, op) in prog.iter().enumerate() {
                    ui.monospace(format!("{k}: {:?} → {}", op.kind, reg_name(op.out_reg)));
                }
            });
    }

    fn inspector_node(&mut self, ui: &mut egui::Ui, i: usize) {
        let spec = spec_of(self.nodes[i].kind);
        ui.label(egui::RichText::new(spec.name).strong().size(16.0));
        ui.separator();
        for (pin, label) in spec.ins.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("◀ {label}"));
                if let Some(src) = self.edge_source(i, pin) {
                    ui.monospace(src);
                    if ui.small_button("✕").clicked() {
                        self.push_undo();
                        self.edges.retain(|e| !(e.to == i && e.to_pin == pin));
                    }
                } else {
                    ui.weak("—");
                }
            });
        }
        if !spec.outs.is_empty() && !matches!(self.nodes[i].kind, OpKind::State | OpKind::Stimulus)
        {
            ui.separator();
            ui.label("Output register:");
            egui::ComboBox::from_id_salt("out_reg")
                .selected_text(reg_name(self.nodes[i].out_reg))
                .show_ui(ui, |ui| {
                    for r in 0..N_REGS {
                        ui.selectable_value(&mut self.nodes[i].out_reg, r, reg_name(r));
                    }
                });
        }
        match self.nodes[i].kind {
            OpKind::Const => {
                ui.add(egui::Slider::new(&mut self.nodes[i].c, -500.0..=500.0).text("value"));
            }
            OpKind::Sigmoid => {
                ui.add(egui::Slider::new(&mut self.nodes[i].c, -200.0..=100.0).text("v_half"));
                ui.add(egui::Slider::new(&mut self.nodes[i].c2, -100.0..=100.0).text("k"));
            }
            _ => {}
        }
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("📄 Duplicate").clicked() {
                self.duplicate_node(i);
            }
            if ui.button("🗑 Delete node").clicked() {
                self.delete_node(i);
            }
        });
    }

    fn plot_ui(&self, ui: &mut egui::Ui) {
        let t_min = self.t - self.window_ms;
        let pts: Vec<[f64; 2]> = self
            .history
            .iter()
            .filter(|&&(x, _)| x >= t_min)
            .map(|&(x, y)| [x, y])
            .collect();
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(_, y) in &self.history {
            ymin = ymin.min(y as f32);
            ymax = ymax.max(y as f32);
        }
        if !ymin.is_finite() {
            ymin = -95.0;
            ymax = 65.0;
        }
        let pad = (ymax - ymin).abs().max(10.0) * 0.15;
        let plot = Plot::new("scope")
            .height(150.0)
            .legend(Legend::default())
            .allow_scroll(false)
            .allow_zoom(false)
            .allow_drag(false)
            .include_x(t_min)
            .include_x(self.t)
            .include_y(ymin - pad)
            .include_y(ymax + pad)
            .label_formatter(|pos| match pos {
                egui_plot::HoverPosition::NearDataPoint {
                    plot_name,
                    position,
                    ..
                } => Some(format!(
                    "{plot_name}: {:.2} mV @ t={:.2} ms",
                    position.y, position.x
                )),
                _ => None,
            });
        plot.show(ui, |pui| {
            if !pts.is_empty() {
                pui.line(
                    Line::new("V(t)", PlotPoints::from(pts)).color(Color32::from_rgb(90, 220, 130)),
                );
            }
            pui.hline(
                HLine::new("thresh", self.spike_thresh as f64)
                    .color(Color32::from_rgb(120, 160, 255)),
            );
            for &s in &self.spike_times {
                if s >= t_min {
                    pui.vline(VLine::new("spike", s).color(Color32::from_rgb(255, 130, 100)));
                }
            }
        });
    }

    // ── Save / load ────────────────────────────────────────────

    fn save_graph(&mut self) {
        let file = FileGraph {
            version: 1,
            nodes: self
                .nodes
                .iter()
                .map(|n| FileNode {
                    kind: kind_index(n.kind),
                    x: n.pos.x,
                    y: n.pos.y,
                    out_reg: n.out_reg,
                    c: n.c,
                    c2: n.c2,
                })
                .collect(),
            edges: self
                .edges
                .iter()
                .map(|e| FileEdge {
                    from: e.from,
                    from_pin: e.from_pin,
                    to: e.to,
                    to_pin: e.to_pin,
                })
                .collect(),
        };
        match serde_json::to_string_pretty(&file) {
            Ok(json) => match std::fs::write(&self.filename, json) {
                Ok(_) => self.set_status(format!("💾 saved → {}", self.filename)),
                Err(e) => self.set_status(format!("save failed: {e}")),
            },
            Err(e) => self.set_status(format!("serialize failed: {e}")),
        }
    }

    fn load_file(&mut self) {
        match std::fs::read_to_string(&self.filename) {
            Ok(json) => match serde_json::from_str::<FileGraph>(&json) {
                Ok(file) => {
                    let mut nodes = Vec::with_capacity(file.nodes.len());
                    for fn_ in file.nodes {
                        let Some(kind) = kind_at(fn_.kind) else {
                            self.set_status(format!("unknown op kind {}", fn_.kind));
                            return;
                        };
                        nodes.push(GNode {
                            kind,
                            pos: pos2(fn_.x, fn_.y),
                            out_reg: fn_.out_reg.min(N_REGS - 1),
                            c: fn_.c,
                            c2: fn_.c2,
                        });
                    }
                    let edges: Vec<GEdge> = file
                        .edges
                        .iter()
                        .filter(|e| e.from < nodes.len() && e.to < nodes.len())
                        .map(|e| GEdge {
                            from: e.from,
                            from_pin: e.from_pin,
                            to: e.to,
                            to_pin: e.to_pin,
                        })
                        .collect();
                    self.push_undo();
                    self.nodes = nodes;
                    self.edges = edges;
                    self.selected = None;
                    self.selected_edge = None;
                    self.drag = Drag::None;
                    self.running = false;
                    self.reset_sim();
                    self.set_status(format!("📂 loaded ← {}", self.filename));
                }
                Err(e) => self.set_status(format!("parse failed: {e}")),
            },
            Err(e) => self.set_status(format!("read failed: {e}")),
        }
    }

    fn load_graph(&mut self, (nodes, edges): (Vec<GNode>, Vec<GEdge>)) {
        self.push_undo();
        self.nodes = nodes;
        self.edges = edges;
        self.selected = None;
        self.selected_edge = None;
        self.drag = Drag::None;
        self.running = false;
        self.reset_sim();
    }

    fn set_status(&mut self, s: String) {
        self.status = s;
    }

    /// Снапшот трейса осциллографа для общего плоттера (`Surface::Plots`).
    /// Инкапсулирует внутренности (history/trace_reg), отдаёт готовые точки.
    pub(crate) fn scope_snapshot(&self) -> BuilderScopeData {
        let t_min = self.t - self.window_ms;
        let pts: Vec<[f64; 2]> = self
            .history
            .iter()
            .filter(|&&(x, _)| x >= t_min)
            .map(|&(x, y)| [x, y])
            .collect();
        let spikes: Vec<f64> = self
            .spike_times
            .iter()
            .copied()
            .filter(|&s| s >= t_min)
            .collect();
        BuilderScopeData {
            pts,
            spikes,
            spike_thresh: self.spike_thresh as f64,
            t_min,
            t_max: self.t,
            total_spikes: self.total_spikes,
        }
    }
}

/// Данные трейса «Конструктора» для общего плоттера.
pub(crate) struct BuilderScopeData {
    pub pts: Vec<[f64; 2]>,
    pub spikes: Vec<f64>,
    pub spike_thresh: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub total_spikes: usize,
}

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= 1e-9 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Cubic-bezier wire with a soft glow underneath (readable over the grid).
fn draw_wire(painter: &egui::Painter, a: Pos2, b: Pos2, color: Color32, width: f32) {
    let pts = bezier_polyline(a, b, 24);
    painter.add(egui::Shape::line(
        pts.clone(),
        Stroke::new(
            width + 3.0,
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 42),
        ),
    ));
    painter.add(egui::Shape::line(pts, Stroke::new(width, color)));
}

impl NeuronBuilder {
    /// Dock-поверхность: тулбар + канвас/осциллограф + инспектор + статус.
    /// Рисуется внутри вкладки [`Surface::Builder`] единого дерева дока.
    pub(crate) fn dock_view(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        // ── Keyboard shortcuts (skip while typing in a text field) ──
        if !ctx.egui_wants_keyboard_input() {
            if ctx.input_mut(|i| {
                i.consume_key(Modifiers::NONE, Key::Delete)
                    || i.consume_key(Modifiers::NONE, Key::Backspace)
            }) {
                if let Some(e) = self.selected_edge {
                    self.delete_edge(e);
                } else if let Some(n) = self.selected {
                    self.delete_node(n);
                }
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Z)) {
                self.undo();
            }
            if ctx.input_mut(|i| {
                i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
                    || i.consume_key(Modifiers::COMMAND, Key::Y)
            }) {
                self.redo();
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::D)) {
                if let Some(n) = self.selected {
                    self.duplicate_node(n);
                }
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Space)) {
                self.running = !self.running;
            }
            if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
                self.ctx_menu = None;
                self.selected = None;
                self.selected_edge = None;
            }
        }

        if self.running {
            for _ in 0..self.steps_per_frame {
                self.step_sim();
            }
        }
        ctx.request_repaint();

        self.toolbar(ui);

        // ── статус-строка снизу ──
        egui::Panel::bottom("builder_status")
            .resizable(false)
            .show(ui, |ui| self.status_row(ui));

        // ── инспектор справа: ресайзбельная панель (не фиксированные 300px) ──
        egui::Panel::right("builder_inspector")
            .resizable(true)
            .default_size(300.0)
            .size_range(220.0..=560.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.inspector(ui));
            });

        // ── канвас + осциллограф: весь оставшийся центр ──
        egui::CentralPanel::no_frame().show(ui, |ui| self.canvas_ui(ui));
    }

    /// Канвас нодового графа + осциллограф под ним (бывший CentralPanel).
    fn canvas_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        {
            let avail = ui.available_size();
            let canvas_w = avail.x.max(160.0); // узкая вкладка → не уходить в минус
                                               // Резерв под осциллограф снизу: 150px, если он показан.
                                               // Клэмп обязателен — маленькая вкладка не должна уходить в минус.
            let plot_reserve = if self.show_plot { 150.0 + 8.0 } else { 0.0 };
            let canvas_h = (avail.y - plot_reserve).max(120.0);
            let (mut resp, painter) =
                ui.allocate_painter(vec2(canvas_w, canvas_h), Sense::click_and_drag());
            let origin = resp.rect.min;
            let size = resp.rect.size();
            let pointer_screen = resp.interact_pointer_pos().map(|p| p - origin);
            let pointer_world = pointer_screen.map(|s| self.screen_to_world(s));

            // ── minimap: click / drag to navigate (overrides canvas drag) ──
            let over_minimap = self.show_minimap
                && pointer_screen.is_some_and(|s| self.minimap_rect(size).contains(pos2(s.x, s.y)));

            // ── hover state: node/wire glow + pin tooltips ──
            // Suppressed under the minimap so a covered node doesn't glow
            // through the overlay (clicks there navigate instead).
            if over_minimap || matches!(self.drag, Drag::Minimap) {
                // also while a minimap pan continues outside the overlay
                self.hover_node = None;
                self.hover_edge = None;
                self.hover_tip = None;
            } else {
                self.hover_node = pointer_world.and_then(|p| self.hit_node(p));
                self.hover_edge = pointer_world.and_then(|p| self.hit_edge(p));
                self.hover_tip = pointer_world.and_then(|p| self.hover_tip_text(p));
            }
            if let Some(tip) = self.hover_tip.clone() {
                resp = resp.on_hover_text(tip);
            }

            // ── zoom around cursor ──
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if (zoom_delta - 1.0).abs() > 1e-4 {
                let anchor = pointer_world.unwrap_or(self.screen_to_world(size * 0.5));
                self.zoom = (self.zoom * zoom_delta).clamp(0.2, 5.0);
                // keep the world point under the cursor fixed
                if let Some(s) = pointer_screen {
                    self.pan = s - anchor * self.zoom;
                }
            }
            // ── middle-drag or empty-drag pans ──
            if resp.dragged_by(PointerButton::Middle) {
                self.pan += resp.drag_delta();
            }

            if over_minimap && resp.drag_started() {
                self.drag = Drag::Minimap;
            }
            if over_minimap && resp.clicked() {
                self.minimap_pan_to(size, pointer_screen);
            }

            // ── drag / click logic ──
            if !over_minimap && resp.drag_started() {
                self.ctx_menu = None;
                let p = pointer_world.unwrap_or(Vec2::ZERO);
                if let Some((n, pin)) = self.hit_out_pin(p) {
                    self.drag = Drag::Wire {
                        from: n,
                        from_pin: pin,
                    };
                } else if let Some(n) = self.hit_node(p) {
                    self.drag = Drag::Node { idx: n };
                    self.selected = Some(n);
                    self.selected_edge = None;
                } else if resp.dragged_by(PointerButton::Primary) {
                    self.drag = Drag::Pan;
                }
            }
            if !over_minimap && resp.clicked() {
                self.ctx_menu = None;
                if let Some(p) = pointer_world {
                    if let Some(n) = self.hit_node(p) {
                        self.selected = Some(n);
                        self.selected_edge = None;
                    } else if let Some(e) = self.hit_edge(p) {
                        self.selected_edge = Some(e);
                        self.selected = None;
                    } else {
                        self.selected = None;
                        self.selected_edge = None;
                    }
                }
            }
            if !over_minimap && resp.secondary_clicked() {
                if let Some(p) = pointer_world {
                    let screen = resp.interact_pointer_pos().unwrap_or(origin);
                    if let Some(n) = self.hit_node(p) {
                        self.selected = Some(n);
                        self.ctx_menu = Some(CtxMenu::Node { screen, idx: n });
                    } else if let Some(e) = self.hit_edge(p) {
                        self.selected_edge = Some(e);
                        self.ctx_menu = Some(CtxMenu::Wire { screen, idx: e });
                    } else {
                        self.ctx_menu = Some(CtxMenu::Canvas { screen, world: p });
                    }
                }
            }
            match self.drag {
                Drag::Node { idx } => {
                    if resp.dragged() {
                        let mut new_pos = self.nodes[idx].pos + resp.drag_delta() / self.zoom;
                        if self.snap {
                            new_pos.x = (new_pos.x / GRID).round() * GRID;
                            new_pos.y = (new_pos.y / GRID).round() * GRID;
                        }
                        self.nodes[idx].pos = new_pos;
                    }
                    if resp.drag_stopped() {
                        self.drag = Drag::None;
                    }
                }
                Drag::Wire { from, from_pin } => {
                    if resp.drag_stopped() {
                        if let Some(p) = pointer_world {
                            if let Some((to, to_pin)) = self.hit_in_pin(p) {
                                self.add_edge(from, from_pin, to, to_pin);
                            }
                        }
                        self.drag = Drag::None;
                    }
                }
                Drag::Pan => {
                    if resp.dragged() {
                        self.pan += resp.drag_delta();
                    }
                    if resp.drag_stopped() {
                        self.drag = Drag::None;
                    }
                }
                Drag::Minimap => {
                    if resp.dragged() {
                        self.minimap_pan_to(size, pointer_screen);
                    }
                    if resp.drag_stopped() {
                        self.drag = Drag::None;
                    }
                }
                Drag::None => {}
            }

            self.draw_canvas(&painter, origin, size, pointer_world);

            // ── context menu ──
            if let Some(menu) = self.ctx_menu.take() {
                let (screen, action) = match menu {
                    CtxMenu::Canvas { screen, world } => (screen, CtxMenuAction::Canvas { world }),
                    CtxMenu::Node { screen, idx } => (screen, CtxMenuAction::Node { idx }),
                    CtxMenu::Wire { screen, idx } => (screen, CtxMenuAction::Wire { idx }),
                };
                egui::Area::new(egui::Id::new("ctx_menu"))
                    .fixed_pos(screen)
                    .order(egui::Order::Foreground)
                    .show(&ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_min_width(140.0);
                            match action {
                                CtxMenuAction::Canvas { world } => {
                                    ui.label("Add op:");
                                    for &kind in palette_kinds() {
                                        if ui.button(spec_of(kind).name).clicked() {
                                            self.add_node_at(kind, world);
                                        }
                                    }
                                }
                                CtxMenuAction::Node { idx } => {
                                    ui.label("Node:");
                                    if ui.button("📄 Duplicate").clicked() {
                                        self.duplicate_node(idx);
                                    }
                                    if ui.button("🗑 Delete").clicked() {
                                        self.delete_node(idx);
                                    }
                                }
                                CtxMenuAction::Wire { idx } => {
                                    ui.label("Wire:");
                                    if ui.button("🗑 Delete").clicked() {
                                        self.delete_edge(idx);
                                    }
                                }
                            }
                        });
                    });
            }

            if self.show_plot {
                ui.separator();
                self.plot_ui(ui);
            }
        }
    }
}

/// What a context-menu button does (decoupled so the menu can own it by value).
enum CtxMenuAction {
    Canvas { world: Vec2 },
    Node { idx: usize },
    Wire { idx: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use pillars::api::formula::run_program;

    /// (имя, граф, dt, i_syn, v_init, старт ворот s1..s3)
    type Preset = (
        &'static str,
        (Vec<GNode>, Vec<GEdge>),
        f32,
        f32,
        f32,
        [f32; 3],
    );

    /// Every preset must fit the register file and simulate without NaN.
    ///
    /// Regression: the ML/HH-lite presets used to allocate temps beyond
    /// `N_REGS = 24` (up to `TEMP0+29 = 42`), panicking the moment a
    /// preset button was clicked. `N_REGS` is now 64 — this locks it in.
    #[test]
    fn presets_fit_register_file_and_simulate() {
        let presets: [Preset; 3] = [
            (
                "LIF",
                default_lif_graph(),
                0.5,
                30.0,
                -70.0,
                [0.0, 0.0, 0.0],
            ),
            ("ML", ml_graph(), 0.05, 50.0, -60.0, [0.0003, 0.0, 0.0]),
            (
                "HH-lite",
                hh_lite_graph(),
                0.05,
                10.0,
                -65.0,
                [0.0153, 0.697, 0.182],
            ),
        ];
        for (name, (nodes, edges), dt, i_syn, v_init, gates) in presets {
            // 1. Register file bounds: every node's out_reg must be valid.
            for (i, n) in nodes.iter().enumerate() {
                assert!(
                    n.out_reg < N_REGS,
                    "{name}: node {i} ({:?}) out_reg {} out of range (N_REGS={N_REGS})",
                    n.kind,
                    n.out_reg
                );
            }
            // 2. Edge bounds: every edge must reference existing nodes.
            for (i, e) in edges.iter().enumerate() {
                assert!(
                    e.from < nodes.len() && e.to < nodes.len(),
                    "{name}: edge {i} references {}-{} but nodes len is {}",
                    e.from,
                    e.to,
                    nodes.len()
                );
            }
            // 3. Simulation: 1000 steps, no NaN, no panic, and it must be alive.
            let prog = graph_to_program(&nodes, &edges);
            assert!(!prog.is_empty(), "{name}: empty program");
            let mut regs = [0.0f32; N_REGS];
            regs[S0] = v_init;
            regs[S1] = gates[0];
            regs[S2] = gates[1];
            regs[S3] = gates[2];
            let mut spike_masks = 0; // LIF: FireReset returns a spike mask
            let mut crossings = 0; // ML/HH-lite: count upward threshold crossings
            let mut prev = regs[S0];
            for _ in 0..1000 {
                let sp = run_program::<f32>(&prog, &mut regs, dt, i_syn);
                assert!(regs[S0].is_finite(), "{name}: v became NaN/inf");
                if sp != 0.0 {
                    spike_masks += 1;
                }
                // ML/HH-lite fire naturally: v swings up through 0 and back.
                if prev < 0.0 && regs[S0] >= 0.0 {
                    crossings += 1;
                }
                prev = regs[S0];
            }
            assert!(
                spike_masks > 0 || crossings > 0,
                "{name}: no spike activity in 1000 steps (masks={spike_masks}, crossings={crossings})"
            );
        }
    }

    /// Stale edges (index out of range) must be skipped, not panic.
    #[test]
    fn stale_edges_are_skipped() {
        let (nodes, mut edges) = default_lif_graph();
        // Simulate corruption from an old snapshot: edge to node 999.
        edges.push(GEdge {
            from: 16,
            from_pin: 0,
            to: 11,
            to_pin: 0,
        });
        edges.push(GEdge {
            from: 0,
            from_pin: 0,
            to: 999,
            to_pin: 0,
        });
        // These must not panic: program builds, sim runs.
        let prog = graph_to_program(&nodes, &edges);
        let mut regs = [0.0f32; N_REGS];
        regs[S0] = -70.0;
        for _ in 0..10 {
            run_program::<f32>(&prog, &mut regs, 0.5, 30.0);
        }
        assert!(regs[S0].is_finite());
        // input_reg_of: валидное ребро (state→scope) резолвится в s0,
        // битые цели (999) откатываются на дефолт S0 без паники.
        assert_eq!(input_reg_of(&nodes, &edges, 13, 0), S0);
        assert_eq!(input_reg_of(&nodes, &edges, 999, 0), S0);
    }

    /// Deleting a node must remap remaining edges consistently.
    #[test]
    fn delete_node_keeps_edges_valid() {
        let (mut nodes, mut edges) = default_lif_graph();
        // Replay the app's delete path on a middle node (index 5 = c_reset).
        let idx = 5;
        edges.retain(|e| e.from != idx && e.to != idx);
        nodes.remove(idx);
        for e in &mut edges {
            if e.from > idx {
                e.from -= 1;
            }
            if e.to > idx {
                e.to -= 1;
            }
        }
        for e in &edges {
            assert!(
                e.from < nodes.len() && e.to < nodes.len(),
                "edge after delete out of range: {}-{} vs len {}",
                e.from,
                e.to,
                nodes.len()
            );
        }
        // And the reduced graph still simulates.
        let prog = graph_to_program(&nodes, &edges);
        let mut regs = [0.0f32; N_REGS];
        regs[S0] = -70.0;
        for _ in 0..10 {
            run_program::<f32>(&prog, &mut regs, 0.5, 30.0);
        }
        assert!(regs[S0].is_finite());
    }
}
