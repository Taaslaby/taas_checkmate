//! # HH Channel Definitions + Polynomial Engine — standard squid-axon & pacemaking (CaV-T)
//!
//! Port of the **entire** legacy `crates/kernels/src/neuron/hh.rs` (волна 8,
//! транш 1e). Two layers:
//!
//! 1. **Channel specs** — what a conductance-based model is made of: the channel
//!    list, per-gate rate functions α(V)/β(V) and steady-state helpers.
//! 2. **Polynomial engine** — `hh_poly_step` / `fused_hh_poly_step`: batch
//!    `f32x8` SIMD + scalar-tail integrators with Q10 temperature correction,
//!    bit-identical to the deleted legacy kernels (SIMD explicit-Euler,
//!    scalar implicit-Euler divergence preserved; fused-scalar clamps v to
//!    [-100, 50], fused-SIMD clamps only Ii to ±100 — as in legacy).
//!
//! The zoo `HHModel` (`run_neuron`) remains the canonical integrator for the
//! standard/pacemaking layouts; this engine is the generic-arity fallback and
//! the reference implementation for parity tests.
//!
//! # Channel sets
//!
//! - [`standard_channels`] — NaV (m³h) + KV (n⁴): the classic Hodgkin-Huxley
//!   squid axon. Requires external drive to spike.
//! - [`pacemaking_channels`] — standard + **CaV-T (T-type calcium)**: a single
//!   low-threshold activation gate `c`. The T-type current depolarizes the
//!   membrane past NaV threshold on its own → the neuron fires **spontaneously**
//!   without any input current (the basis of rhythmic VNC/CPG activity).

#![allow(non_snake_case)]

use crate::math::fast_exp::fast_exp;
use crate::math::fast_exp::fast_exp_poly_simd as fast_exp_simd;
use wide::f32x8;
use wide::CmpLt;

/// SIMD vtrap: `x / (exp(x/y) - 1)` с Taylor `y - x/2` при |x/y| < 1e-6.
/// В точности как NEURON hh.mod `FUNCTION vtrap`. Предотвращает 0/0 при x → 0.
#[inline(always)]
fn vtrap_simd(x: f32x8, y: f32) -> f32x8 {
    let y_v = f32x8::splat(y);
    let ratio = x / y_v;
    let small_mask = ratio.abs().cmp_lt(f32x8::splat(1e-6));
    // Taylor: y - x/2 (matches NEURON: y*(1 - x/y/2))
    let taylor = y_v - x * f32x8::splat(0.5);
    let exact = x / (fast_exp_simd(ratio) - f32x8::splat(1.0));
    small_mask.blend(taylor, exact)
}

// ═══════════════════════════════════════════════════════════
//  Channel specification types
// ═══════════════════════════════════════════════════════════

/// Замкнутая форма HH-величины (rate-функция α/β или steady-state) в конвенции
/// движка: аргумент всегда `−(v + midpoint)/scale`. Соответствует NeuroML2
/// `HHExpRate` / `HHSigmoidRate` / `HHExpLinearRate` (их `*Variable`-двойники
/// структурно идентичны, поэтому один enum на обе роли).
///
/// Выбор этой конвенции (а не NeuroML `(v − midpoint)/scale`) намеренный: она
/// бит-в-бит воспроизводит канонические скалярные функции `na_m_alpha` и т.д.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClosedForm {
    /// `rate · exp(−(v + midpoint)/scale)`.
    Exp {
        rate: f32,
        midpoint: f32,
        scale: f32,
    },
    /// `rate / (1 + exp(−(v + midpoint)/scale))`.
    Sigmoid {
        rate: f32,
        midpoint: f32,
        scale: f32,
    },
    /// `rate · vtrap(−(v + midpoint), scale)`.
    ExpLinear {
        rate: f32,
        midpoint: f32,
        scale: f32,
    },
}

impl ClosedForm {
    /// Scalar evaluation.
    #[inline(always)]
    pub fn eval(self, v: f32) -> f32 {
        match self {
            ClosedForm::Exp {
                rate,
                midpoint,
                scale,
            } => rate * fast_exp(-(v + midpoint) / scale),
            ClosedForm::Sigmoid {
                rate,
                midpoint,
                scale,
            } => rate / (1.0 + fast_exp(-(v + midpoint) / scale)),
            ClosedForm::ExpLinear {
                rate,
                midpoint,
                scale,
            } => rate * vtrap(-(v + midpoint), scale),
        }
    }

    /// SIMD evaluation (8 lanes).
    #[inline(always)]
    pub fn eval_simd(self, v: f32x8) -> f32x8 {
        let one = f32x8::splat(1.0);
        match self {
            ClosedForm::Exp {
                rate,
                midpoint,
                scale,
            } => {
                f32x8::splat(rate)
                    * fast_exp_simd(-(v + f32x8::splat(midpoint)) / f32x8::splat(scale))
            }
            ClosedForm::Sigmoid {
                rate,
                midpoint,
                scale,
            } => {
                f32x8::splat(rate)
                    / (one + fast_exp_simd(-(v + f32x8::splat(midpoint)) / f32x8::splat(scale)))
            }
            ClosedForm::ExpLinear {
                rate,
                midpoint,
                scale,
            } => f32x8::splat(rate) * vtrap_simd(-(v + f32x8::splat(midpoint)), scale),
        }
    }
}

/// Comparison operator for a LEMS `ConditionalDerivedVariable` `Case`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
}

/// One `Case` of a piecewise expression: `lhs OP rhs -> value`.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseCase {
    pub op: CmpOp,
    pub lhs: LemsExpr,
    pub rhs: LemsExpr,
    pub value: LemsExpr,
}

/// A scalar expression over membrane voltage `v` (mV), compiled from a custom
/// LEMS `ComponentType` (`baseVoltageDepTime` / `baseVoltageDepVariable`).
///
/// Real OSB models define gate time courses / steady states as arbitrary
/// expressions (e.g. the Poirazi 2003 CA1 Ih `Poirazi_h_tau`, whose tau is a
/// piecewise double-exponential of `v`), so the NeuroML parser lowers them to
/// this small expression tree instead of erroring on the custom `ComponentType`.
#[derive(Debug, Clone, PartialEq)]
pub enum LemsExpr {
    /// Numeric constant (already unit-converted at parse time).
    Const(f32),
    /// Membrane voltage `v` (mV).
    V,
    /// Internal ion concentration `caConc` (mM) — the compartment's [Ca²⁺]
    /// for Ca-dependent gates (e.g. Poirazi `h2 = ki/(ki+caConc)`). Evaluated
    /// with the compartment concentration via [`LemsExpr::eval_at`]; in
    /// point-neuron (no-Ca) contexts it evaluates to `0.0`.
    CaConc,
    /// The `rateScale` `<Requirement>` of a custom tau — the Q10 multiplier
    /// of the gate (jNeuroML semantics). Bound to a constant by
    /// [`LemsExpr::bind_rate_scale`]; defaults to `1.0` (no Q10).
    RateScale,
    /// The simulation temperature `<Requirement>` (K) — e.g. Poirazi `cagk`
    /// rates depend on `temperature`. Bound to a constant by
    /// [`LemsExpr::bind_temperature`] (celsius + 273.15); defaults to `1.0`.
    Temperature,
    Neg(Box<LemsExpr>),
    Add(Box<LemsExpr>, Box<LemsExpr>),
    Sub(Box<LemsExpr>, Box<LemsExpr>),
    Mul(Box<LemsExpr>, Box<LemsExpr>),
    Div(Box<LemsExpr>, Box<LemsExpr>),
    /// `exp(...)`.
    Exp(Box<LemsExpr>),
    /// Placeholder for a `<Parameter>` (index into the gate-bound values).
    /// Only present in the parsed `ComponentType` body; replaced by
    /// [`LemsExpr::bind_params`] before the gate reaches the runtime.
    Param(usize),
    /// A symbol that could not be resolved (e.g. a `<Requirement>`).
    Unresolved(String),
    /// Piecewise: the first `Case` whose condition holds wins; else `default`.
    Piecewise {
        cases: Vec<PiecewiseCase>,
        default: Box<LemsExpr>,
    },
}

impl LemsExpr {
    /// Evaluate at membrane voltage `v` (mV), with `caConc` = 0 (no Ca
    /// context).
    #[inline]
    pub fn eval(&self, v: f32) -> f32 {
        self.eval_at(v, 0.0)
    }

    /// Evaluate at membrane voltage `v` (mV) and internal Ca²⁺ concentration
    /// `ca` (mM).
    #[inline]
    pub fn eval_at(&self, v: f32, ca: f32) -> f32 {
        match self {
            LemsExpr::Const(c) => *c,
            LemsExpr::V => v,
            LemsExpr::CaConc => ca,
            LemsExpr::RateScale => 1.0,
            LemsExpr::Temperature => 1.0,
            LemsExpr::Neg(e) => -e.eval_at(v, ca),
            LemsExpr::Add(a, b) => a.eval_at(v, ca) + b.eval_at(v, ca),
            LemsExpr::Sub(a, b) => a.eval_at(v, ca) - b.eval_at(v, ca),
            LemsExpr::Mul(a, b) => a.eval_at(v, ca) * b.eval_at(v, ca),
            LemsExpr::Div(a, b) => a.eval_at(v, ca) / b.eval_at(v, ca),
            LemsExpr::Exp(e) => fast_exp(e.eval_at(v, ca)),
            // `Param`/`Unresolved` must be bound/resolved before the gate is
            // stored — reaching `eval` means a parser bug.
            LemsExpr::Param(_) | LemsExpr::Unresolved(_) => 0.0,
            LemsExpr::Piecewise { cases, default } => {
                for c in cases {
                    let l = c.lhs.eval_at(v, ca);
                    let r = c.rhs.eval_at(v, ca);
                    let hit = match c.op {
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        CmpOp::Ge => l >= r,
                        CmpOp::Eq => l == r,
                        CmpOp::Neq => l != r,
                    };
                    if hit {
                        return c.value.eval_at(v, ca);
                    }
                }
                default.eval_at(v, ca)
            }
        }
    }

    /// Replace `RateScale` placeholders with `Const(rate_scale)` — the gate's
    /// Q10 multiplier (jNeuroML binds the `rateScale` Requirement from
    /// `q10Settings`).
    pub fn bind_rate_scale(&self, rate_scale: f32) -> LemsExpr {
        self.bind_requirement(LemsExpr::RateScale, rate_scale)
    }

    /// Replace `Temperature` placeholders with `Const(temp_k)` — the
    /// simulation temperature in Kelvin (e.g. Poirazi `cagk` rates).
    pub fn bind_temperature(&self, temp_k: f32) -> LemsExpr {
        self.bind_requirement(LemsExpr::Temperature, temp_k)
    }

    /// Generic placeholder binding: replace every `needle` leaf with
    /// `Const(value)` (recursively).
    fn bind_requirement(&self, needle: LemsExpr, value: f32) -> LemsExpr {
        if *self == needle {
            return LemsExpr::Const(value);
        }
        match self {
            LemsExpr::Const(_) | LemsExpr::V | LemsExpr::CaConc | LemsExpr::Unresolved(_) => {
                self.clone()
            }
            LemsExpr::RateScale | LemsExpr::Temperature => self.clone(),
            LemsExpr::Param(i) => LemsExpr::Param(*i),
            LemsExpr::Neg(a) => LemsExpr::Neg(Box::new(a.bind_requirement(needle.clone(), value))),
            LemsExpr::Exp(a) => LemsExpr::Exp(Box::new(a.bind_requirement(needle.clone(), value))),
            LemsExpr::Add(a, b) => LemsExpr::Add(
                Box::new(a.bind_requirement(needle.clone(), value)),
                Box::new(b.bind_requirement(needle.clone(), value)),
            ),
            LemsExpr::Sub(a, b) => LemsExpr::Sub(
                Box::new(a.bind_requirement(needle.clone(), value)),
                Box::new(b.bind_requirement(needle.clone(), value)),
            ),
            LemsExpr::Mul(a, b) => LemsExpr::Mul(
                Box::new(a.bind_requirement(needle.clone(), value)),
                Box::new(b.bind_requirement(needle.clone(), value)),
            ),
            LemsExpr::Div(a, b) => LemsExpr::Div(
                Box::new(a.bind_requirement(needle.clone(), value)),
                Box::new(b.bind_requirement(needle.clone(), value)),
            ),
            LemsExpr::Piecewise { cases, default } => LemsExpr::Piecewise {
                cases: cases
                    .iter()
                    .map(|c| PiecewiseCase {
                        op: c.op,
                        lhs: c.lhs.bind_requirement(needle.clone(), value),
                        rhs: c.rhs.bind_requirement(needle.clone(), value),
                        value: c.value.bind_requirement(needle.clone(), value),
                    })
                    .collect(),
                default: Box::new(default.bind_requirement(needle.clone(), value)),
            },
        }
    }

    /// Replace `Param(i)` placeholders with `Const(params[i])` — final binding
    /// of a `ComponentType`'s `<Parameter>`s from the gate node's attributes.
    pub fn bind_params(&self, params: &[f32]) -> LemsExpr {
        match self {
            LemsExpr::Const(_)
            | LemsExpr::V
            | LemsExpr::CaConc
            | LemsExpr::RateScale
            | LemsExpr::Temperature
            | LemsExpr::Unresolved(_) => self.clone(),
            LemsExpr::Param(i) => LemsExpr::Const(params[*i]),
            LemsExpr::Neg(a) => LemsExpr::Neg(Box::new(a.bind_params(params))),
            LemsExpr::Exp(a) => LemsExpr::Exp(Box::new(a.bind_params(params))),
            LemsExpr::Add(a, b) => LemsExpr::Add(
                Box::new(a.bind_params(params)),
                Box::new(b.bind_params(params)),
            ),
            LemsExpr::Sub(a, b) => LemsExpr::Sub(
                Box::new(a.bind_params(params)),
                Box::new(b.bind_params(params)),
            ),
            LemsExpr::Mul(a, b) => LemsExpr::Mul(
                Box::new(a.bind_params(params)),
                Box::new(b.bind_params(params)),
            ),
            LemsExpr::Div(a, b) => LemsExpr::Div(
                Box::new(a.bind_params(params)),
                Box::new(b.bind_params(params)),
            ),
            LemsExpr::Piecewise { cases, default } => LemsExpr::Piecewise {
                cases: cases
                    .iter()
                    .map(|c| PiecewiseCase {
                        op: c.op,
                        lhs: c.lhs.bind_params(params),
                        rhs: c.rhs.bind_params(params),
                        value: c.value.bind_params(params),
                    })
                    .collect(),
                default: Box::new(default.bind_params(params)),
            },
        }
    }
}

/// Steady-state of a gate: a standard [`ClosedForm`] (fast SIMD path) or a
/// custom LEMS expression (evaluated per-lane).
#[derive(Debug, Clone, PartialEq)]
pub enum VarForm {
    Closed(ClosedForm),
    Expr(LemsExpr),
}

impl VarForm {
    #[inline(always)]
    pub fn eval(&self, v: f32) -> f32 {
        self.eval_at(v, 0.0)
    }

    /// Evaluate with an internal Ca²⁺ concentration `ca` (mM) — needed by
    /// Ca-dependent steady states (`baseVoltageConcDepVariable`, e.g. Poirazi
    /// `kca_ss`, `cal_ssh2`).
    #[inline(always)]
    pub fn eval_at(&self, v: f32, ca: f32) -> f32 {
        match self {
            VarForm::Closed(cf) => cf.eval(v),
            VarForm::Expr(e) => e.eval_at(v, ca),
        }
    }

    #[inline(always)]
    pub fn eval_simd(&self, v: f32x8) -> f32x8 {
        match self {
            VarForm::Closed(cf) => cf.eval_simd(v),
            VarForm::Expr(e) => {
                let arr = v.to_array();
                let mut out = [0.0f32; 8];
                for (i, &vi) in arr.iter().enumerate() {
                    out[i] = e.eval(vi);
                }
                f32x8::from(out)
            }
        }
    }
}

/// Time course of a gate: fixed (`fixedTimeCourse`) or a custom V-dependent
/// LEMS expression (ms).
#[derive(Debug, Clone, PartialEq)]
pub enum TauForm {
    /// `fixedTimeCourse` — constant tau (ms).
    Fixed(f32),
    /// Custom `ComponentType` — V-dependent tau (ms).
    Expr(LemsExpr),
}

impl TauForm {
    #[inline(always)]
    pub fn eval(&self, v: f32) -> f32 {
        self.eval_at(v, 0.0)
    }

    /// Evaluate with an internal Ca²⁺ concentration `ca` (mM) — for
    /// Ca-dependent time courses (`baseVoltageConcDepTime`, e.g. Poirazi
    /// `kca_tau`).
    #[inline(always)]
    pub fn eval_at(&self, v: f32, ca: f32) -> f32 {
        match self {
            TauForm::Fixed(t) => *t,
            TauForm::Expr(e) => e.eval_at(v, ca),
        }
    }

    #[inline(always)]
    pub fn eval_simd(&self, v: f32x8) -> f32x8 {
        match self {
            TauForm::Fixed(t) => f32x8::splat(*t),
            TauForm::Expr(e) => {
                let arr = v.to_array();
                let mut out = [0.0f32; 8];
                for (i, &vi) in arr.iter().enumerate() {
                    out[i] = e.eval(vi);
                }
                f32x8::from(out)
            }
        }
    }
}

/// A rate (α or β) of a `gateHHrates`: a standard closed form (fast SIMD
/// path) or a custom LEMS expression (Ca/temperature-dependent, e.g. Poirazi
/// `cagk`).
#[derive(Debug, Clone, PartialEq)]
pub enum RateForm {
    Closed(ClosedForm),
    Expr(LemsExpr),
}

impl RateForm {
    /// Convert a parsed [`VarForm`] (closed or LEMS expression) into a rate.
    pub fn from_var_form(vf: VarForm) -> RateForm {
        match vf {
            VarForm::Closed(cf) => RateForm::Closed(cf),
            VarForm::Expr(e) => RateForm::Expr(e),
        }
    }

    #[inline(always)]
    pub fn eval(&self, v: f32) -> f32 {
        self.eval_at(v, 0.0)
    }

    /// Evaluate with an internal Ca²⁺ concentration `ca` (mM) — for
    /// Ca-dependent rates (`baseVoltageConcDepRate`).
    #[inline(always)]
    pub fn eval_at(&self, v: f32, ca: f32) -> f32 {
        match self {
            RateForm::Closed(cf) => cf.eval(v),
            RateForm::Expr(e) => e.eval_at(v, ca),
        }
    }

    #[inline(always)]
    pub fn eval_simd(&self, v: f32x8) -> f32x8 {
        match self {
            RateForm::Closed(cf) => cf.eval_simd(v),
            RateForm::Expr(e) => {
                let arr = v.to_array();
                let mut out = [0.0f32; 8];
                for (i, &vi) in arr.iter().enumerate() {
                    out[i] = e.eval_at(vi, 0.0);
                }
                f32x8::from(out)
            }
        }
    }
}

/// Кинетика гейта — как NeuroML2 `gateHHrates` / `gateHHtauInf`. Обобщённый
/// data-driven слой, который NeuroML-парсер заполняет произвольными каналами.
#[derive(Debug, Clone, PartialEq)]
pub enum GateKinetics {
    /// `gateHHrates`: inf = α/(α+β), tau = 1/((α+β)·rateScale).
    Rates {
        forward: RateForm,
        reverse: RateForm,
    },
    /// `gateHHtauInf`: inf = steady_state, tau — фиксированный или V-зависимый
    /// (кастомный LEMS `ComponentType`, как у реального Ih).
    TauInf { steady_state: VarForm, tau: TauForm },
    /// `gateHHInstantaneous`: algebraic gate — `q = steady_state(v[, ca])`
    /// every step, no ODE. Used by real Ca channels (Poirazi `cal`/`cat`
    /// `h2 = ki/(ki+caConc)` — the fast Ca-dependent inactivation).
    Instantaneous { steady_state: VarForm },
}

impl GateKinetics {
    /// Scalar α/β. Для `TauInf` — деление `inf/tau` (легаси-скаляр).
    /// Ca-dependent forms are evaluated at `ca = 0` (no-Ca contexts).
    #[inline(always)]
    pub fn alpha_beta(&self, v: f32) -> (f32, f32) {
        self.alpha_beta_at(v, 0.0)
    }

    /// Scalar α/β with an explicit internal Ca²⁺ concentration `ca` (mM).
    /// Ca-dependent `Rates`/`TauInf` forms (`baseVoltageConcDepRate`/
    /// `baseVoltageConcDepVariable`/`baseVoltageConcDepTime`, e.g. Poirazi
    /// `cagk`/`kca`) must be evaluated per-compartment with the compartment's
    /// actual [Ca²⁺] — `alpha_beta(v)` would silently pin them to `ca = 0`.
    #[inline(always)]
    pub fn alpha_beta_at(&self, v: f32, ca: f32) -> (f32, f32) {
        match self {
            GateKinetics::Rates { forward, reverse } => {
                (forward.eval_at(v, ca), reverse.eval_at(v, ca))
            }
            GateKinetics::TauInf { steady_state, tau } => {
                let inf = steady_state.eval_at(v, ca);
                let tau = tau.eval_at(v, ca);
                (inf / tau, (1.0 - inf) / tau)
            }
            // No ODE — `gate_step` handles `Instantaneous` directly (q = ss).
            GateKinetics::Instantaneous { steady_state } => (steady_state.eval_at(v, ca), 0.0),
        }
    }

    /// Steady-state gate value `x_∞ = α(v)/(α(v)+β(v))` at potential `v`.
    ///
    /// Numerical floor `1e-12` on `α + β` with neutral fallback `0.5` — the
    /// canonical rule used by [`steady_state_gates`](self::steady_state_gates),
    /// `ConductancePopulation` resting init and Markov resting states. Single
    /// source of truth so engine, api and population layers can never drift.
    #[inline(always)]
    pub fn steady_state_at(&self, v: f32) -> f32 {
        let (a, b) = self.alpha_beta(v);
        let sum = a + b;
        if sum > 1e-12 {
            a / sum
        } else {
            0.5
        }
    }

    /// SIMD α/β. Для `TauInf` — умножение на `1/tau` (легаси-SIMD).
    #[inline(always)]
    pub fn alpha_beta_simd(&self, v: f32x8) -> (f32x8, f32x8) {
        let one = f32x8::splat(1.0);
        match self {
            GateKinetics::Rates { forward, reverse } => {
                (forward.eval_simd(v), reverse.eval_simd(v))
            }
            GateKinetics::Instantaneous { steady_state } => {
                let inf = steady_state.eval_simd(v);
                (inf, f32x8::splat(0.0))
            }
            GateKinetics::TauInf { steady_state, tau } => {
                let inf = steady_state.eval_simd(v);
                let tau_inv = match tau {
                    TauForm::Fixed(t) => f32x8::splat(1.0 / t),
                    TauForm::Expr(e) => {
                        let arr = v.to_array();
                        let mut out = [0.0f32; 8];
                        for (i, &vi) in arr.iter().enumerate() {
                            out[i] = 1.0 / e.eval(vi);
                        }
                        f32x8::from(out)
                    }
                };
                (inf * tau_inv, (one - inf) * tau_inv)
            }
        }
    }
}

/// One gating variable (e.g. Na-m, Na-h, K-n, CaT-c).
#[derive(Debug, Clone)]
pub struct GateSpec {
    pub label: &'static str,
    /// Data-driven kinetics — источник истины для обобщённого движка `hh_poly_step`.
    pub kinetics: GateKinetics,
    /// Exponent in the conductance term: `gating = gate^power`.
    pub power: u32,
}

/// One ion channel: one or more gates + reversal potential + max conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    /// Reversal potential in mV.
    pub E_rev: f32,
    /// Max conductance in mS/cm².
    pub g_max: f32,
}

// ═══════════════════════════════════════════════════════════
//  Squid-axon rate functions (Hodgkin & Huxley 1952, 6.3 °C)
// ═══════════════════════════════════════════════════════════

/// vtrap: `x / (exp(x/y) − 1)` with a Taylor `y − x/2` when `|x/y| < 1e-6`.
/// Prevents 0/0 at x → 0, exactly like NEURON's `FUNCTION vtrap`.
///
/// This is the HH rate-function helper (x = voltage shift, y = slope divisor).
/// Do not confuse it with [`crate::math::vtrap::vtrap`] — the sigmoid-shaped
/// per-lane primitive used by gates — the two share a name but not semantics.
#[inline(always)]
pub fn vtrap(x: f32, y: f32) -> f32 {
    let ratio = x / y;
    if ratio.abs() < 1e-6 {
        y - x * 0.5
    } else {
        x / (fast_exp(ratio) - 1.0)
    }
}

/// Na activation (m-gate) α: `0.1·vtrap(−(v+40), 10)`.
pub fn na_m_alpha(v: f32) -> f32 {
    0.1 * vtrap(-(v + 40.0), 10.0)
}
/// Na activation (m-gate) β: `4·exp(−(v+65)/18)`.
pub fn na_m_beta(v: f32) -> f32 {
    4.0 * fast_exp(-(v + 65.0) / 18.0)
}

/// Na inactivation (h-gate) α: `0.07·exp(−(v+65)/20)`.
pub fn na_h_alpha(v: f32) -> f32 {
    0.07 * fast_exp(-(v + 65.0) / 20.0)
}
/// Na inactivation (h-gate) β: `1/(1 + exp(−(v+35)/10))`.
pub fn na_h_beta(v: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-(v + 35.0) / 10.0))
}

/// K activation (n-gate) α: `0.01·vtrap(−(v+55), 10)`.
pub fn k_n_alpha(v: f32) -> f32 {
    0.01 * vtrap(-(v + 55.0), 10.0)
}
/// K activation (n-gate) β: `0.125·exp(−(v+65)/80)`.
pub fn k_n_beta(v: f32) -> f32 {
    0.125 * fast_exp(-(v + 65.0) / 80.0)
}

/// CaV-T (T-type calcium) activation — low-threshold, pacemaking.
/// Hill & Tononi 2005 / Wang & Rinzel 1993. Single gate `c`, τ_c = 0.6 ms.
/// α = m_inf/τ where m_inf = 1/(1 + exp(−(v+56)/6.2)).
pub fn cat_c_alpha(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    m_inf / 0.6
}
/// CaV-T β: `(1 − m_inf)/τ`.
pub fn cat_c_beta(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + fast_exp(-(v + 56.0) / 6.2));
    (1.0 - m_inf) / 0.6
}

// ═══════════════════════════════════════════════════════════
//  Channel sets
// ═══════════════════════════════════════════════════════════

/// Classic HH squid axon: NaV (m³h) + KV (n⁴).
pub fn standard_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec {
                    label: "m",
                    kinetics: GateKinetics::Rates {
                        forward: RateForm::Closed(ClosedForm::ExpLinear {
                            rate: 0.1,
                            midpoint: 40.0,
                            scale: 10.0,
                        }),
                        reverse: RateForm::Closed(ClosedForm::Exp {
                            rate: 4.0,
                            midpoint: 65.0,
                            scale: 18.0,
                        }),
                    },
                    power: 3,
                },
                GateSpec {
                    label: "h",
                    kinetics: GateKinetics::Rates {
                        forward: RateForm::Closed(ClosedForm::Exp {
                            rate: 0.07,
                            midpoint: 65.0,
                            scale: 20.0,
                        }),
                        reverse: RateForm::Closed(ClosedForm::Sigmoid {
                            rate: 1.0,
                            midpoint: 35.0,
                            scale: 10.0,
                        }),
                    },
                    power: 1,
                },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![GateSpec {
                label: "n",
                kinetics: GateKinetics::Rates {
                    forward: RateForm::Closed(ClosedForm::ExpLinear {
                        rate: 0.01,
                        midpoint: 55.0,
                        scale: 10.0,
                    }),
                    reverse: RateForm::Closed(ClosedForm::Exp {
                        rate: 0.125,
                        midpoint: 65.0,
                        scale: 80.0,
                    }),
                },
                power: 4,
            }],
            E_rev: -77.0,
            g_max: 36.0,
        },
    ]
}

/// HH + CaV-T (T-type calcium) for **pacemaking**: 4 gates (NaV-m, NaV-h, KV-n,
/// CaT-c). The T-type current makes neurons fire spontaneously without input.
pub fn pacemaking_channels() -> Vec<ChannelSpec> {
    let mut ch = standard_channels();
    ch.push(ChannelSpec {
        name: "CaT",
        gates: vec![GateSpec {
            label: "c",
            kinetics: GateKinetics::TauInf {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 56.0,
                    scale: 6.2,
                }),
                tau: TauForm::Fixed(0.6),
            },
            power: 1,
        }],
        E_rev: 120.0,
        g_max: 0.5,
    });
    ch
}

// ═══════════════════════════════════════════════════════════
//  Steady-state helpers
// ═══════════════════════════════════════════════════════════

/// Steady-state gate values `m_inf = α(V)/(α(V)+β(V))` at a given potential,
/// in gate order (channel-major).
pub fn steady_state_gates(v: f32, channels: &[ChannelSpec]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channels {
        for gate in &ch.gates {
            gates.push(gate.kinetics.steady_state_at(v));
        }
    }
    gates
}
