//! -------------------------------------------------------------------------
//! Filename:    src/formula.rs
//! Class:       api
//! Purpose:     Formula — Assemble Neurons from Ops
//! Description:
//!
//! The runtime counterpart of the compile-time `#[derive(NeuronModel)]`
//! bridge. A [`OpNode`] program is a flat list of biological ops
//! (the LEGO bricks from [`crate::ops`]) executed in order over a small
//! **register file**:
//!
//! ```text
//!   regs[0..4]   = state  s0..s3  (persisted across steps)
//!   regs[4..12]  = params p0..p7  (per-neuron constants, optional)
//!   regs[12]     = i_syn          (injected current bus)
//!   regs[13..64] = temps  t0..t50 (transient intermediates)
//!
//! `N_REGS = 64` gives the node editor headroom to assemble arbitrarily
//! large graphs (the ML and HH-lite presets alone need ~22 and ~30 temps).
//! ```
//!
//! [`run_program`] is generic over [`SimdLane`]: the same program compiles
//! to 8-wide AVX2 (`f32x8`) or scalar (`f32`) — identical semantics.
//!
//! This is the engine behind the interactive node editor
//! (`examples/viz_neuron_builder.rs`): a node graph serializes to a
//! `Vec<OpNode>`, and a single neuron can be simulated and plotted live.
//!
//! ## LIF as a program
//!
//! ```text
//! t0 = const(v_rest = -70)        t3 = const(v_reset = -70)
//! t1 = const(inv_tau = 0.1)       t4 = const(tau_ref = 2.0)
//! t2 = const(v_thresh = -54)      t5 = const(0.0)
//!
//! t6 = CmpGt(refr=s1, 0)                   // is_refr mask (1.0/0.0)
//! t7 = Leak(v=s0, i=i_syn, inv_tau=t1, v_rest=t0)  // raw Euler
//! t8 = Blend(mask=t6, a=s0, b=t7)          // freeze v while refractory
//! s0 = FireReset(v=t8, thresh=t2, reset=t3)
//! s1 = RefrTimer(refr=s1, tau_ref=t4)       // uses spike flag
//! ```
//!
//! Absolute refractory FREEZES v (no integration, no spike breakthrough),
//! matching [`crate::zoo::neurons::LIFModel`] (verified by the
//! `formula_lif_matches_zoo_lif` test).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::core::context::SimdContext;
use crate::core::lane::SimdLane;
use crate::ops::gate::integrate_gate;
use crate::ops::membrane::leak_integrate;
use crate::ops::ohmic::ohmic_current;
use crate::ops::refractory::{refractory_clamp, refractory_timer};
use crate::ops::spike::fire_reset;
use crate::zoo::Model;

// ═══════════════════════════════════════════════════════════════
//  Register file
// ═══════════════════════════════════════════════════════════════

/// State register 0 — conventionally the membrane potential `v`.
pub const S0: usize = 0;
/// State register 1 — e.g. refractory counter / adaptation.
pub const S1: usize = 1;
/// State register 2 — user-defined.
pub const S2: usize = 2;
/// State register 3 — user-defined.
pub const S3: usize = 3;
/// First parameter register (p0..p7 = regs 4..12).
pub const P0: usize = 4;
/// Injected synaptic current bus.
pub const I_SYN: usize = 12;
/// First transient register (t0..t50 = regs 13..63).
pub const TEMP0: usize = 13;
/// Total register count — sized for arbitrarily large editor graphs.
pub const N_REGS: usize = 64;

// ═══════════════════════════════════════════════════════════════
//  Op set
// ═══════════════════════════════════════════════════════════════

/// The biological opcode set — one per LEGO brick in [`crate::ops`].
///
/// Each variant reads its input registers (`in_regs`), computes, and writes
/// one output register (`out_reg`). Order matters: the program is executed
/// top-to-bottom, exactly like a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// `v' = v + (v_rest − v + i) · inv_tau · dt` — Forward Euler LIF leak.
    Leak,
    /// `I = g · (V − E_rev)` — Ohm's law (NEURON convention, FMA).
    Ohmic,
    /// `out = a + b`
    Add,
    /// `out = a · b`
    Mul,
    /// `out = a / b`
    Div,
    /// `out = (a ≥ b) ? 1.0 : 0.0` — comparison to float mask.
    CmpGe,
    /// `out = (a > b) ? 1.0 : 0.0` — strict comparison to float mask.
    CmpGt,
    /// `out = |mask| > 0 ? a : b` — branchless select.
    Blend,
    /// `g' = g_inf + (g − g_inf) · exp(−dt · inv_tau)` — exponential gate.
    GateStep,
    /// `out = 1 / (1 + exp(−(x − v_half)/k))` — scalar v_half/k constants.
    Sigmoid,
    /// `out = exp(x)`
    Exp,
    /// `out = (v ≥ thresh) ? reset : v`; sets the internal spike flag.
    FireReset,
    /// `out = (refr > 0) ? reset : v` — refractory clamp at input.
    RefrClamp,
    /// `out = spike ? tau_ref : max(refr − dt, 0)` — uses the spike flag.
    RefrTimer,
    /// `out = value` — literal constant stored in the node.
    Const,
    /// `out = i_syn` — the current bus (node editor: "Stimulus").
    Stimulus,
    /// No-op; samples its input register (node editor: oscilloscope probe).
    Scope,
    /// Graph-only: source of the persistent state registers `s0..s3`
    /// (node editor "State" node). Never emitted into a program.
    State,
}

/// One resolved program instruction.
#[derive(Debug, Clone, Copy)]
pub struct OpNode {
    pub kind: OpKind,
    /// Input registers for pins 0..3 (unused pins are ignored).
    pub in_regs: [usize; 4],
    /// Output register.
    pub out_reg: usize,
    /// Literal value — used by `Const` (and sigmoid `v_half`, `k`).
    pub c: f32,
    /// Secondary literal — sigmoid slope `k`.
    pub c2: f32,
}

impl OpNode {
    /// Create an instruction with default (harmless) inputs.
    #[inline(always)]
    pub fn new(kind: OpKind, out_reg: usize) -> Self {
        Self {
            kind,
            in_regs: [S0; 4],
            out_reg,
            c: 0.0,
            c2: 0.0,
        }
    }

    /// Constant literal.
    #[inline(always)]
    pub fn const_(out_reg: usize, value: f32) -> Self {
        Self {
            kind: OpKind::Const,
            in_regs: [S0; 4],
            out_reg,
            c: value,
            c2: 0.0,
        }
    }

    /// Override input registers; shorter slices are padded with `S0`.
    #[inline(always)]
    pub fn with_in(mut self, in_regs: &[usize]) -> Self {
        let n = in_regs.len().min(4);
        self.in_regs[..n].copy_from_slice(&in_regs[..n]);
        self
    }
}

// ═══════════════════════════════════════════════════════════════
//  Interpreter
// ═══════════════════════════════════════════════════════════════

/// Execute one simulation step of a formula program.
///
/// # Arguments
/// * `program` — ops in execution order.
/// * `regs` — persistent register file; state regs `s0..s3` survive across
///   calls, temps are rewritten every step.
/// * `dt` — simulation timestep (ms).
/// * `i_syn` — injected current for this step.
///
/// # Returns
/// The spike mask of this step (from the last `FireReset` executed).
#[inline]
pub fn run_program<S: SimdLane>(program: &[OpNode], regs: &mut [S; N_REGS], dt: S, i_syn: S) -> S {
    let zero = S::zero();
    let one = S::one();
    regs[I_SYN] = i_syn;

    let mut spike = zero;
    for op in program {
        assert!(
            op.out_reg < N_REGS
                && op
                    .in_regs
                    .iter()
                    .take(op_arity(op.kind))
                    .all(|&r| r < N_REGS),
            "formula register out of range"
        );
        assert!(
            !(P0..P0 + 8).contains(&op.out_reg),
            "formula parameter registers are read-only"
        );
        match op.kind {
            OpKind::Leak => {
                let (v, i, inv_tau, v_rest) = (
                    regs[op.in_regs[0]],
                    regs[op.in_regs[1]],
                    regs[op.in_regs[2]],
                    regs[op.in_regs[3]],
                );
                regs[op.out_reg] = leak_integrate(v, v_rest, i, inv_tau, dt);
            }
            OpKind::Ohmic => {
                let (g, v, e) = (
                    regs[op.in_regs[0]],
                    regs[op.in_regs[1]],
                    regs[op.in_regs[2]],
                );
                regs[op.out_reg] = ohmic_current(g, v, e);
            }
            OpKind::Add => regs[op.out_reg] = regs[op.in_regs[0]] + regs[op.in_regs[1]],
            OpKind::Mul => regs[op.out_reg] = regs[op.in_regs[0]] * regs[op.in_regs[1]],
            OpKind::Div => regs[op.out_reg] = regs[op.in_regs[0]] / regs[op.in_regs[1]],
            OpKind::CmpGe => {
                let m = regs[op.in_regs[0]].cmp_ge(regs[op.in_regs[1]]);
                regs[op.out_reg] = m.cond().then(one).else_(zero);
            }
            OpKind::CmpGt => {
                let m = regs[op.in_regs[0]].cmp_gt(regs[op.in_regs[1]]);
                regs[op.out_reg] = m.cond().then(one).else_(zero);
            }
            OpKind::Blend => {
                let active = regs[op.in_regs[0]].abs().cmp_gt(zero);
                regs[op.out_reg] = active
                    .cond()
                    .then(regs[op.in_regs[1]])
                    .else_(regs[op.in_regs[2]]);
            }
            OpKind::GateStep => {
                let (g, g_inf, inv_tau) = (
                    regs[op.in_regs[0]],
                    regs[op.in_regs[1]],
                    regs[op.in_regs[2]],
                );
                regs[op.out_reg] = integrate_gate(g, g_inf, inv_tau, dt);
            }
            OpKind::Sigmoid => regs[op.out_reg] = S::sigmoid(regs[op.in_regs[0]], op.c, op.c2),
            OpKind::Exp => regs[op.out_reg] = regs[op.in_regs[0]].exp(),
            OpKind::FireReset => {
                let (v, thresh, reset) = (
                    regs[op.in_regs[0]],
                    regs[op.in_regs[1]],
                    regs[op.in_regs[2]],
                );
                let (v_new, sp) = fire_reset(v, thresh, reset);
                regs[op.out_reg] = v_new;
                spike = sp;
            }
            OpKind::RefrClamp => {
                let (v, refr, reset) = (
                    regs[op.in_regs[0]],
                    regs[op.in_regs[1]],
                    regs[op.in_regs[2]],
                );
                regs[op.out_reg] = refractory_clamp(v, refr, reset);
            }
            OpKind::RefrTimer => {
                let (refr, tau_ref) = (regs[op.in_regs[0]], regs[op.in_regs[1]]);
                regs[op.out_reg] = refractory_timer(refr, spike, tau_ref, dt);
            }
            OpKind::Const => regs[op.out_reg] = S::splat(op.c),
            OpKind::Stimulus => regs[op.out_reg] = i_syn,
            OpKind::Scope | OpKind::State => {}
        }
    }
    spike
}

/// Number of input pins each opcode actually reads during execution.
///
/// `in_regs` — фиксированный `[usize; 4]`, так что чтение лишнего пина не
/// паникует — но незаполненный пин молча читает дефолтный регистр S0
/// (мембранный потенциал) вместо ошибки — трудноловимый логический баг.
/// Редактор графов должен вызывать [`validate_program_arity`] с числом
/// реально подключённых пинов.
#[inline]
pub const fn op_arity(kind: OpKind) -> usize {
    match kind {
        OpKind::Leak => 4,
        OpKind::Ohmic | OpKind::Blend | OpKind::GateStep => 3,
        OpKind::FireReset | OpKind::RefrClamp => 3,
        OpKind::Add | OpKind::Mul | OpKind::Div | OpKind::CmpGe | OpKind::CmpGt => 2,
        OpKind::RefrTimer => 2,
        OpKind::Sigmoid | OpKind::Exp | OpKind::Scope => 1,
        OpKind::Const | OpKind::Stimulus | OpKind::State => 0,
    }
}

/// Validate a program before execution — every output and input register
/// must be within `0..N_REGS`.
///
/// The interpreter re-checks register bounds on every operation; call this
/// once at build time (e.g. in the node editor) to fail fast with a clear
/// message before execution.
pub fn validate_program(program: &[OpNode]) -> Result<(), String> {
    for (i, op) in program.iter().enumerate() {
        if op.out_reg >= N_REGS {
            return Err(format!(
                "op {i} ({:?}): out_reg {} >= N_REGS ({N_REGS})",
                op.kind, op.out_reg
            ));
        }
        if (P0..P0 + 8).contains(&op.out_reg) {
            return Err(format!(
                "op {i} ({:?}): parameter register {} is read-only",
                op.kind, op.out_reg
            ));
        }
        // Проверяем только реально читаемые пины: неиспользуемые могут
        // содержать мусор от редактора — это не ошибка.
        for (j, &r) in op.in_regs.iter().take(op_arity(op.kind)).enumerate() {
            if r >= N_REGS {
                return Err(format!(
                    "op {i} ({:?}): in_reg[{j}] {r} >= N_REGS ({N_REGS})",
                    op.kind
                ));
            }
        }
    }
    Ok(())
}

/// Strict validation for graph editors: additionally checks that the caller
/// connected at least [`op_arity`] input pins to each node.
///
/// `connected_pins[i]` — число пинов, реально подключённых к узлу `i`
/// (длина должна совпадать с `program.len()`).
pub fn validate_program_arity(program: &[OpNode], connected_pins: &[usize]) -> Result<(), String> {
    validate_program(program)?;
    if connected_pins.len() != program.len() {
        return Err(format!(
            "connected_pins length {} != program length {}",
            connected_pins.len(),
            program.len()
        ));
    }
    for (i, (op, &pins)) in program.iter().zip(connected_pins).enumerate() {
        let need = op_arity(op.kind);
        if pins < need {
            return Err(format!(
                "op {i} ({:?}): {pins} input pin(s) connected, needs {need}",
                op.kind
            ));
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
//  FormulaModel — a zoo Model executed by the interpreter
// ═══════════════════════════════════════════════════════════════

/// Maximum number of simultaneously registered formula programs.
pub const MAX_FORMULA_MODELS: usize = 16;

/// Registered program plus the bitmask of param registers it actually reads
/// (so the bridge only loads the SoA columns the program needs).
struct RegisteredProgram {
    ops: Box<[OpNode]>,
    used_params: u8,
}

static FORMULA_PROGRAMS: [OnceLock<RegisteredProgram>; MAX_FORMULA_MODELS] =
    [const { OnceLock::new() }; MAX_FORMULA_MODELS];

/// Bitmask of param registers `P0..P0+8` read by the program's input pins.
fn used_param_mask(program: &[OpNode]) -> u8 {
    let mut mask = 0u8;
    for op in program {
        for &r in &op.in_regs {
            if (P0..P0 + 8).contains(&r) {
                mask |= 1 << (r - P0);
            }
        }
    }
    mask
}

/// A [`Model`] that executes a registered [`OpNode`] program.
///
/// This is the **runtime bridge from the DSL to the zoo**: a program built in
/// the node editor (or by hand) becomes a first-class [`Model`] and runs
/// through [`NeuralPopulation`](crate::core::NeuralPopulation) — SIMD chunks,
/// scalar tail, and (with `rayon`) `run_neuron_par` — with zero safety
/// escapes, exactly like a compile-time model.
///
/// # Register file ↔ population mapping
///
/// ```text
///   s0..s3  ← ctx.s(0..3)  state channels   (persisted across steps)
///   p0..p7  ← ctx.p(0..7)  param columns    (SoA, per-neuron)
///   i_syn   ← ctx.i()
///   dt      ← ctx.dt()     (param column 0, per-lane)
/// ```
///
/// Programs reference params by register (`P0 + idx`). For example a LIF
/// written against columns `[0]dt [1]inv_tau [2]v_rest [3]v_thresh
/// [4]v_reset [5]tau_ref` (absolute-freeze refractory — v is held frozen
/// while `s1 > 0`, so `FireReset` on the frozen sub-threshold v cannot
/// spike through the refractory period):
///
/// ```text
/// t0 = CmpGt(refr=s1, 0)                        // is_refr mask
/// t1 = Leak(v=s0, i=i_syn, inv_tau=p1, v_rest=p2)
/// t2 = Blend(mask=t0, a=s0, b=t1)               // frozen v while refr
/// s0 = FireReset(v=t2, thresh=p3, reset=p4)
/// s1 = RefrTimer(refr=s1, tau_ref=p5)
/// ```
///
/// Only the param columns the program actually reads are loaded from SoA
/// (bitmask computed at registration) — the bridge is zero-waste.
///
/// ⚠️ Param registers `p0..p7` are **read-only inputs**: a program must not
/// write to them (e.g. via `Const → P0+2`) — the write is overwritten by the
/// SoA pre-load on the next step. Use temps `t0..t50` for scratch.
///
/// ℹ️ `dt` is passed **per-lane** via [`SimdContext::dt`] (SoA column 0).
/// Derived models (`#[derive(NeuronModel)]`) instead broadcast the first
/// neuron's dt via `ctx.dt_scalar()` — the two execution paths agree for
/// homogeneous dt and diverge only for per-neuron heterogeneous dt.
///
/// # Example
///
/// ```rust,ignore
/// use pillars::api::formula::*;
/// use pillars::prelude::*;
///
/// FormulaModel::<0>::register(lif_program()).unwrap();
/// let mut pop = NeuralPopulation::<FormulaModel<0>>::new(138_000);
/// pop.fill_param(0, 0.5);   // dt
/// pop.fill_param(1, 0.1);   // inv_tau_m
/// pop.fill_state(0, -70.0); // v
/// pop.step(&vec![30.0; 138_000], &mut vec![0i32; 138_000]);
/// ```
pub struct FormulaModel<const ID: usize>;

impl<const ID: usize> FormulaModel<ID> {
    /// Register the program executed by this model. Fails if the program
    /// fails [`validate_program`] or the slot is already taken.
    pub fn register(program: Vec<OpNode>) -> Result<(), String> {
        const {
            assert!(ID < MAX_FORMULA_MODELS);
        }
        validate_program(&program)?;
        let used = used_param_mask(&program);
        let slot = &FORMULA_PROGRAMS[ID];
        slot.set(RegisteredProgram {
            ops: program.into_boxed_slice(),
            used_params: used,
        })
        .map_err(|_| format!("FormulaModel<{ID}>: program already registered"))
    }

    /// Whether a program has been registered for this model.
    pub fn is_registered() -> bool {
        const {
            assert!(ID < MAX_FORMULA_MODELS);
        }
        FORMULA_PROGRAMS[ID].get().is_some()
    }
}

impl<const ID: usize> Model for FormulaModel<ID> {
    const N_PARAMS: usize = 8; // p0..p7 — the bridge loads only used columns
    const N_STATE: usize = 4; // s0..s3
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        const {
            assert!(ID < MAX_FORMULA_MODELS);
        }
        let prog = FORMULA_PROGRAMS[ID].get().expect(
            "FormulaModel: no program registered — call FormulaModel::<ID>::register() first",
        );

        // State s0..s3 → registers (persisted across steps)
        let mut regs = [S::zero(); N_REGS];
        regs[S0] = ctx.s(0);
        regs[S1] = ctx.s(1);
        regs[S2] = ctx.s(2);
        regs[S3] = ctx.s(3);

        // Params p0..p7 → SoA columns (only the ones the program reads)
        let used = prog.used_params;
        for k in 0..8 {
            if used & (1 << k) != 0 {
                regs[P0 + k] = ctx.p(k);
            }
        }

        // Execute the program (dt per-lane from param column 0, i_syn from bus)
        let spike = run_program(&prog.ops, &mut regs, ctx.dt(), ctx.i());

        // State back, spike out
        ctx.set_s(0, regs[S0]);
        ctx.set_s(1, regs[S1]);
        ctx.set_s(2, regs[S2]);
        ctx.set_s(3, regs[S3]);
        ctx.spike(spike);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::neurons::LIFModel;
    use crate::NeuralPopulation;

    /// The canonical LIF program — mirrors `zoo/neurons/lif.rs` op-by-op.
    ///
    /// Absolute refractory FREEZES v: `CmpGt(s1, 0)` builds the is-refractory
    /// mask, `Blend` keeps the frozen v instead of the integrated value, and
    /// `FireReset` on the frozen (sub-threshold) v cannot spike — no
    /// breakthrough during the refractory period.
    fn lif_program() -> Vec<OpNode> {
        let t_rest = TEMP0;
        let t_tau = TEMP0 + 1;
        let t_thresh = TEMP0 + 2;
        let t_reset = TEMP0 + 3;
        let t_tref = TEMP0 + 4;
        let t_zero = TEMP0 + 5;
        let t_is_refr = TEMP0 + 6;
        let t_raw = TEMP0 + 7;
        let t_gated = TEMP0 + 8;
        vec![
            OpNode::const_(t_rest, -70.0),   // v_rest
            OpNode::const_(t_tau, 0.1),      // inv_tau_m = 1/10
            OpNode::const_(t_thresh, -54.0), // v_thresh
            OpNode::const_(t_reset, -70.0),  // v_reset
            OpNode::const_(t_tref, 2.0),     // tau_ref
            OpNode::const_(t_zero, 0.0),     // 0 for CmpGt
            OpNode::new(OpKind::CmpGt, t_is_refr).with_in(&[S1, t_zero]),
            OpNode::new(OpKind::Leak, t_raw).with_in(&[S0, I_SYN, t_tau, t_rest]),
            OpNode::new(OpKind::Blend, t_gated).with_in(&[t_is_refr, S0, t_raw]),
            OpNode::new(OpKind::FireReset, S0).with_in(&[t_gated, t_thresh, t_reset]),
            OpNode::new(OpKind::RefrTimer, S1).with_in(&[S1, t_tref]),
        ]
    }

    #[test]
    fn test_validate_program_ok_and_bad() {
        let t0 = TEMP0;
        let prog = vec![
            OpNode::const_(t0, 1.0),
            OpNode::new(OpKind::Add, t0).with_in(&[t0, t0]),
        ];
        assert!(validate_program(&prog).is_ok());

        // Bad output register.
        let bad_out = vec![OpNode::const_(N_REGS, 1.0)];
        let err = validate_program(&bad_out).unwrap_err();
        assert!(err.contains("out_reg"), "{err}");

        // Bad input register (0 and 1 are fine, 64 is not).
        let bad_in = vec![OpNode::new(OpKind::Add, t0).with_in(&[t0, N_REGS])];
        let err = validate_program(&bad_in).unwrap_err();
        assert!(err.contains("in_reg"), "{err}");

        // Parameter registers are inputs only; writing one would make the
        // program depend on instruction order and be overwritten next step.
        let writes_param = vec![OpNode::const_(P0, 1.0)];
        let err = validate_program(&writes_param).unwrap_err();
        assert!(err.contains("read-only"), "{err}");
    }

    /// Мусор в НЕиспользуемом пине не должен валить валидацию:
    /// Add читает только пины 0..2, пин 3 интерпретатор не трогает.
    #[test]
    fn test_validate_ignores_unused_pins() {
        let t0 = TEMP0;
        let mut op = OpNode::new(OpKind::Add, t0).with_in(&[t0, t0]);
        op.in_regs[3] = usize::MAX; // мусор в неиспользуемом пине
        assert!(
            validate_program(&[op]).is_ok(),
            "unused pin garbage must not fail validation"
        );
    }

    /// Строгая арность: редактор сообщает, сколько пинов подключено.
    #[test]
    fn test_validate_arity() {
        let t0 = TEMP0;
        let prog = vec![OpNode::new(OpKind::Leak, t0).with_in(&[S0, I_SYN, t0, t0])];
        // Leak требует 4 пина: 4 подключено — ok, 2 — ошибка
        assert!(validate_program_arity(&prog, &[4]).is_ok());
        let err = validate_program_arity(&prog, &[2]).unwrap_err();
        assert!(err.contains("needs 4"), "{err}");
        // Длины не совпали — тоже ошибка
        assert!(validate_program_arity(&prog, &[]).is_err());
    }

    #[test]
    #[should_panic(expected = "formula register out of range")]
    fn test_run_program_rejects_malformed_program_in_release() {
        let mut regs = [0.0f32; N_REGS];
        let bad = [OpNode::new(OpKind::Add, TEMP0).with_in(&[TEMP0, N_REGS])];
        let _ = run_program(&bad, &mut regs, 0.5f32, 0.0f32);
    }

    #[test]
    #[should_panic(expected = "formula parameter registers are read-only")]
    fn test_run_program_rejects_parameter_destination_without_prevalidation() {
        let mut regs = [0.0f32; N_REGS];
        let bad = [OpNode::const_(P0, 1.0)];
        let _ = run_program(&bad, &mut regs, 0.5f32, 0.0f32);
    }

    #[test]
    fn test_formula_arithmetic() {
        let mut regs = [0.0f32; N_REGS];
        let t0 = TEMP0;
        let t1 = TEMP0 + 1;
        let t2 = TEMP0 + 2;
        let prog = vec![
            OpNode::const_(t0, 2.0),
            OpNode::const_(t1, 3.0),
            OpNode::new(OpKind::Add, t2).with_in(&[t0, t1]),
            OpNode::new(OpKind::CmpGe, t1).with_in(&[t2, t0]), // 5 ≥ 2 → 1.0
            OpNode::new(OpKind::Blend, t0).with_in(&[t1, t2, S0]), // mask → a = 5
        ];
        run_program(&prog, &mut regs, 0.5, 0.0);
        assert_eq!(regs[t2], 5.0, "add");
        assert_eq!(regs[t1], 1.0, "cmpge");
        assert_eq!(regs[t0], 5.0, "blend");
    }

    #[test]
    fn test_formula_div() {
        let mut regs = [0.0f32; N_REGS];
        let t0 = TEMP0;
        let t1 = TEMP0 + 1;
        let t2 = TEMP0 + 2;
        let prog = vec![
            OpNode::const_(t0, 6.0),
            OpNode::const_(t1, 3.0),
            OpNode::new(OpKind::Div, t2).with_in(&[t0, t1]), // 6 / 3 = 2
        ];
        run_program(&prog, &mut regs, 0.5, 0.0);
        assert_eq!(regs[t2], 2.0, "div");
    }

    #[test]
    fn test_formula_lif_matches_zoo_lif() {
        // ── Formula interpreter (scalar f32) ──
        let program = lif_program();
        let mut regs = [0.0f32; N_REGS];
        regs[S0] = -70.0;
        let dt = 0.5f32;
        let i_syn = 30.0f32;
        let mut v_traj = Vec::with_capacity(200);
        let mut formula_spikes = 0i32;
        for _ in 0..200 {
            let sp = run_program(&program, &mut regs, dt, i_syn);
            if sp != 0.0 {
                // fired: scalar masks are NaN (all-bits) or ±1.0 — never 0.0
                formula_spikes += 1;
            }
            v_traj.push(regs[S0]);
        }

        // ── Reference: zoo LIF via NeuralPopulation (scalar path, n = 1) ──
        let mut pop = NeuralPopulation::<LIFModel>::new(1);
        pop.fill_param(0, 0.5); // dt
        pop.fill_param(1, 0.1); // inv_tau_m
        pop.fill_param(2, -70.0); // v_rest
        pop.fill_param(3, -54.0); // v_thresh
        pop.fill_param(4, -70.0); // v_reset
        pop.fill_param(5, 2.0); // tau_ref
        pop.fill_state(0, -70.0); // v
        let i_in = vec![i_syn];
        let mut spikes_out = vec![0i32];
        let mut zoo_spikes = 0i32;
        for (step, &v_formula) in v_traj.iter().enumerate() {
            pop.step(&i_in, &mut spikes_out);
            zoo_spikes += spikes_out[0];
            let v_zoo = pop.state_slice(0)[0];
            assert!(
                (v_formula - v_zoo).abs() < 1e-4,
                "step {step}: formula v={v_formula:.6} != zoo v={v_zoo:.6}"
            );
        }

        assert_eq!(
            formula_spikes, zoo_spikes,
            "spike counts must match: formula={formula_spikes} zoo={zoo_spikes}"
        );
        assert!(zoo_spikes > 0, "LIF with I=30 must spike");
    }

    // ═══════════════════════════════════════════════════════
    //  FormulaModel — SIMD bridge to the zoo
    // ═══════════════════════════════════════════════════════

    /// LIF program referencing SoA param columns `[0]dt [1]inv_tau [2]v_rest
    /// [3]v_thresh [4]v_reset [5]tau_ref` via registers `P0+1..P0+5` — mirrors
    /// `zoo/neurons/lif.rs` op-by-op (frozen refractory, same as `lif_program`).
    fn lif_param_program() -> Vec<OpNode> {
        let t_zero = TEMP0;
        let t_is_refr = TEMP0 + 1;
        let t_raw = TEMP0 + 2;
        let t_gated = TEMP0 + 3;
        vec![
            OpNode::const_(t_zero, 0.0),
            OpNode::new(OpKind::CmpGt, t_is_refr).with_in(&[S1, t_zero]),
            OpNode::new(OpKind::Leak, t_raw).with_in(&[S0, I_SYN, P0 + 1, P0 + 2]),
            OpNode::new(OpKind::Blend, t_gated).with_in(&[t_is_refr, S0, t_raw]),
            OpNode::new(OpKind::FireReset, S0).with_in(&[t_gated, P0 + 3, P0 + 4]),
            OpNode::new(OpKind::RefrTimer, S1).with_in(&[S1, P0 + 5]),
        ]
    }

    fn fill_lif_params<M: Model>(pop: &mut NeuralPopulation<M>) {
        pop.fill_param(0, 0.5); // dt
        pop.fill_param(1, 0.1); // inv_tau_m
        pop.fill_param(2, -70.0); // v_rest
        pop.fill_param(3, -54.0); // v_thresh
        pop.fill_param(4, -70.0); // v_reset
        pop.fill_param(5, 2.0); // tau_ref
        pop.fill_state(0, -70.0); // v
    }

    #[test]
    fn test_run_program_simd_matches_scalar() {
        use wide::f32x8;
        let program = lif_param_program();
        let dt = 0.5f32;
        let i_syns = [5.0f32, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];

        // Scalar reference: 8 independent register files
        let mut scalar = [[0.0f32; N_REGS]; 8];
        for row in scalar.iter_mut() {
            row[P0 + 1] = 0.1;
            row[P0 + 2] = -70.0;
            row[P0 + 3] = -54.0;
            row[P0 + 4] = -70.0;
            row[P0 + 5] = 2.0;
            row[S0] = -70.0;
        }

        // SIMD: 8 lanes, same params splatted, per-lane i_syn
        let mut simd = [f32x8::zero(); N_REGS];
        for k in [1usize, 2, 3, 4, 5] {
            simd[P0 + k] = f32x8::splat(scalar[0][P0 + k]);
        }
        simd[S0] = f32x8::splat(-70.0);

        let mut simd_total = 0u32;
        let mut scalar_total = 0u32;
        for step in 0..300 {
            let sp_simd = run_program(&program, &mut simd, f32x8::splat(dt), f32x8::new(i_syns));
            // NOTE: `wide::f32x8` has an inherent `move_mask() -> i32` that shadows
            // the trait method — cast to u32 for a uniform bitmask.
            let mask = sp_simd.move_mask() as u32;
            for lane in 0..8 {
                let sp = run_program(&program, &mut scalar[lane], dt, i_syns[lane]);
                let fired = sp.move_mask();
                assert_eq!(
                    (mask >> lane) & 1,
                    fired,
                    "step {step} lane {lane}: SIMD={} scalar={}",
                    (mask >> lane) & 1,
                    fired
                );
                for k in 0..4 {
                    assert!(
                        (simd[k].to_array()[lane] - scalar[lane][k]).abs() < 1e-5,
                        "step {step} lane {lane} state[{k}]: simd={} scalar={}",
                        simd[k].to_array()[lane],
                        scalar[lane][k]
                    );
                }
                scalar_total += fired;
            }
            simd_total += mask.count_ones();
        }
        assert_eq!(simd_total, scalar_total, "total spikes must match");
    }

    #[test]
    fn test_formula_model_matches_zoo_lif() {
        // Unique ID to avoid clashes with other tests in this binary
        FormulaModel::<3>::register(lif_param_program()).unwrap();
        assert!(FormulaModel::<3>::is_registered());
        assert!(
            FormulaModel::<3>::register(lif_param_program()).is_err(),
            "double register must fail"
        );

        for &n in &[1usize, 8, 10, 64] {
            let mut fpop = NeuralPopulation::<FormulaModel<3>>::new(n);
            fill_lif_params(&mut fpop);

            let mut zpop = NeuralPopulation::<LIFModel>::new(n);
            fill_lif_params(&mut zpop);

            let i_in = vec![30.0f32; n];
            let mut f_spikes = vec![0i32; n];
            let mut z_spikes = vec![0i32; n];
            let mut f_total = 0i32;
            let mut z_total = 0i32;

            for step in 0..300 {
                fpop.step(&i_in, &mut f_spikes);
                zpop.step(&i_in, &mut z_spikes);
                f_total += f_spikes.iter().sum::<i32>();
                z_total += z_spikes.iter().sum::<i32>();
                for i in 0..n {
                    assert!(
                        (fpop.state_slice(0)[i] - zpop.state_slice(0)[i]).abs() < 1e-5,
                        "n={n} step {step} neuron {i}: formula={} zoo={}",
                        fpop.state_slice(0)[i],
                        zpop.state_slice(0)[i]
                    );
                    assert!(
                        (fpop.state_slice(1)[i] - zpop.state_slice(1)[i]).abs() < 1e-5,
                        "refr mismatch n={n} step {step} neuron {i}"
                    );
                    assert_eq!(
                        f_spikes[i], z_spikes[i],
                        "spike mismatch n={n} step {step} neuron {i}"
                    );
                }
            }
            assert_eq!(f_total, z_total, "spike totals must match, n={n}");
            assert!(z_total > 0, "LIF must spike with I=30, n={n}");
        }
    }

    #[test]
    fn test_formula_model_population_spikes() {
        FormulaModel::<4>::register(lif_param_program()).unwrap();
        let mut pop = NeuralPopulation::<FormulaModel<4>>::new(10);
        fill_lif_params(&mut pop);
        let i_in = vec![100.0f32; 10];
        let mut spikes = vec![0i32; 10];
        let mut total = 0i32;
        for _ in 0..100 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "FormulaModel LIF must spike with I=100");
    }

    #[test]
    fn test_formula_model_register_validation() {
        let bad = vec![OpNode::const_(N_REGS, 1.0)]; // out-of-range out_reg
        assert!(
            FormulaModel::<5>::register(bad).is_err(),
            "invalid program must be rejected"
        );
    }
}
