//! # formula — собирай своего нейрона из ops (реалтайм, Python)
//!
//! Красивая Python-обёртка над `pillars::api::formula` — runtime-интерпретатором
//! программ из био-кирпичей. Тот же движок, что у `#[derive(NeuronModel)]`, но
//! нейрон собирается в рантайме из опкодов `Leak / Ohmic / FireReset / …`.
//!
//! ```python
//! from taas_checkmate.formula import Neuron
//!
//! n = Neuron("MyLIF")
//! v    = n.state("v", -70.0)          # состояние + значение по умолчанию
//! refr = n.state("refr", 0.0)
//! n.param("dt", 0.5)                  # dt обязан быть первым параметром
//! n.param("inv_tau", 0.1)
//! n.param("v_rest", -70.0)
//! n.param("v_thresh", -54.0)
//! n.param("v_reset", -70.0)
//! n.param("tau_ref", 2.0)
//!
//! # сборка — dataflow: результат одного op передаётся в другой
//! is_refr = n.cmp_gt(refr, 0.0)                    # → t0
//! raw     = n.leak(v, n.I, n.inv_tau, n.v_rest)    # v + (v_rest − v + i)·inv_tau·dt
//! gated   = n.blend(is_refr, v, raw)               # заморозка в рефрактерности
//! n.fire_reset(gated, n.v_thresh, n.v_reset, into=v)  # спайк-флаг
//! n.refr_timer(refr, n.tau_ref, into=refr)
//!
//! print(n)                # читаемый список опкодов
//! tr = n.simulate(steps=200, i_syn=30.0)   # быстрый прогон: tr.v, tr.spikes
//!
//! model = n.compile()                         # → FormulaModel (слот движка)
//! pop = model.population(138_000)             # SIMD-популяция f32x8
//! pop.set(v_rest=-70.0, v_thresh=-54.0)
//! pop.step(i_in, spikes)                      # zero-copy numpy
//! ```
//!
//! Параметры — атрибуты билдера (`n.inv_tau`) и одновременно имена колонок
//! популяции. Сырой слой для продвинутых: `OpKind`, `OpNode`, `run_program`.

#![allow(non_snake_case)]

use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pillars::api::formula::{self, OpKind, OpNode, FormulaModel, I_SYN, N_REGS, P0, S0, TEMP0};
use pillars::ops::spike::spike_to_float;
use pillars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ═══════════════════════════════════════════════════════════════
//  Регистры и имена
// ═══════════════════════════════════════════════════════════════

/// Хэндл регистра: `v`, `refr`, `n.inv_tau`, `n.I` или авто-темп `t0…t50`.
/// Возвращается каждым op и передаётся в следующий — dataflow-сборка.
#[pyclass(name = "Reg", module = "taas_checkmate.formula")]
#[derive(Clone)]
pub struct PyReg {
    pub idx: usize,
    pub name: String,
}

#[pymethods]
impl PyReg {
    fn __repr__(&self) -> String {
        self.name.clone()
    }
    #[getter]
    fn idx(&self) -> usize {
        self.idx
    }
}

fn op_kind_name(kind: OpKind) -> &'static str {
    match kind {
        OpKind::Leak => "Leak",
        OpKind::Ohmic => "Ohmic",
        OpKind::Add => "Add",
        OpKind::Mul => "Mul",
        OpKind::Div => "Div",
        OpKind::CmpGe => "CmpGe",
        OpKind::CmpGt => "CmpGt",
        OpKind::Blend => "Blend",
        OpKind::GateStep => "GateStep",
        OpKind::Sigmoid => "Sigmoid",
        OpKind::Exp => "Exp",
        OpKind::FireReset => "FireReset",
        OpKind::RefrClamp => "RefrClamp",
        OpKind::RefrTimer => "RefrTimer",
        OpKind::Const => "Const",
        OpKind::Stimulus => "Stimulus",
        OpKind::Scope => "Scope",
        OpKind::State => "State",
    }
}

// ═══════════════════════════════════════════════════════════════
//  OpKind / OpNode — сырой слой
// ═══════════════════════════════════════════════════════════════

/// Опкоды биологического набора — один в один с `pillars::api::formula::OpKind`.
#[pyclass(eq, eq_int, name = "OpKind", module = "taas_checkmate.formula")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyOpKind {
    LEAK,
    OHMIC,
    ADD,
    MUL,
    DIV,
    CMPGE,
    CMPGT,
    BLEND,
    GATESTEP,
    SIGMOID,
    EXP,
    FIRERESET,
    REFRCLAMP,
    REFRTIMER,
    CONST,
    STIMULUS,
    SCOPE,
    STATE,
}

impl PyOpKind {
    fn to_rust(self) -> OpKind {
        match self {
            PyOpKind::LEAK => OpKind::Leak,
            PyOpKind::OHMIC => OpKind::Ohmic,
            PyOpKind::ADD => OpKind::Add,
            PyOpKind::MUL => OpKind::Mul,
            PyOpKind::DIV => OpKind::Div,
            PyOpKind::CMPGE => OpKind::CmpGe,
            PyOpKind::CMPGT => OpKind::CmpGt,
            PyOpKind::BLEND => OpKind::Blend,
            PyOpKind::GATESTEP => OpKind::GateStep,
            PyOpKind::SIGMOID => OpKind::Sigmoid,
            PyOpKind::EXP => OpKind::Exp,
            PyOpKind::FIRERESET => OpKind::FireReset,
            PyOpKind::REFRCLAMP => OpKind::RefrClamp,
            PyOpKind::REFRTIMER => OpKind::RefrTimer,
            PyOpKind::CONST => OpKind::Const,
            PyOpKind::STIMULUS => OpKind::Stimulus,
            PyOpKind::SCOPE => OpKind::Scope,
            PyOpKind::STATE => OpKind::State,
        }
    }
    fn from_rust(kind: OpKind) -> Self {
        match kind {
            OpKind::Leak => PyOpKind::LEAK,
            OpKind::Ohmic => PyOpKind::OHMIC,
            OpKind::Add => PyOpKind::ADD,
            OpKind::Mul => PyOpKind::MUL,
            OpKind::Div => PyOpKind::DIV,
            OpKind::CmpGe => PyOpKind::CMPGE,
            OpKind::CmpGt => PyOpKind::CMPGT,
            OpKind::Blend => PyOpKind::BLEND,
            OpKind::GateStep => PyOpKind::GATESTEP,
            OpKind::Sigmoid => PyOpKind::SIGMOID,
            OpKind::Exp => PyOpKind::EXP,
            OpKind::FireReset => PyOpKind::FIRERESET,
            OpKind::RefrClamp => PyOpKind::REFRCLAMP,
            OpKind::RefrTimer => PyOpKind::REFRTIMER,
            OpKind::Const => PyOpKind::CONST,
            OpKind::Stimulus => PyOpKind::STIMULUS,
            OpKind::Scope => PyOpKind::SCOPE,
            OpKind::State => PyOpKind::STATE,
        }
    }
}

/// Одна инструкция программы — продвинутый путь (билдер `Neuron` использует их внутри).
#[pyclass(name = "OpNode", module = "taas_checkmate.formula")]
#[derive(Clone)]
pub struct PyOpNode {
    pub inner: OpNode,
}

#[pymethods]
impl PyOpNode {
    /// `OpNode(OpKind.LEAK, out_reg)` — инструкция с дефолтными входами.
    #[new]
    fn new(kind: PyOpKind, out_reg: usize) -> Self {
        Self {
            inner: OpNode::new(kind.to_rust(), out_reg),
        }
    }

    /// Литеральная константа: `OpNode.const_(out_reg, value)`.
    #[staticmethod]
    fn const_(out_reg: usize, value: f32) -> Self {
        Self {
            inner: OpNode::const_(out_reg, value),
        }
    }

    /// Подключить входные регистры (короче 4 — дополняются s0).
    fn with_in(&mut self, in_regs: Vec<usize>) -> Self {
        let mut out = self.clone();
        out.inner = out.inner.with_in(&in_regs);
        out
    }

    #[getter]
    fn kind(&self) -> PyOpKind {
        PyOpKind::from_rust(self.inner.kind)
    }
    #[getter]
    fn out_reg(&self) -> usize {
        self.inner.out_reg
    }
    #[getter]
    fn in_regs(&self) -> Vec<usize> {
        self.inner.in_regs.to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "{}(in={:?}, out={})",
            op_kind_name(self.inner.kind),
            self.inner.in_regs,
            self.inner.out_reg
        )
    }
}

/// Продвинутый путь: прогнать программу `OpNode` вручную по регистровому файлу.
///
/// `regs` — 64 числа (`N_REGS`): s0..s3, p0..p7, i_syn, t0..t50.
/// Возвращает `(regs_after, spike)` — спайк = 1.0/0.0.
#[pyfunction(name = "run_program")]
#[pyo3(signature = (program, regs, dt, i_syn))]
fn py_run_program(
    py: Python<'_>,
    program: Vec<Py<PyOpNode>>,
    regs: Vec<f32>,
    dt: f32,
    i_syn: f32,
) -> PyResult<(Vec<f32>, f32)> {
    if regs.len() != N_REGS {
        return Err(PyValueError::new_err(format!(
            "run_program: regs must have exactly {N_REGS} elements, got {}",
            regs.len()
        )));
    }
    let ops: Vec<OpNode> = program.iter().map(|p| p.borrow(py).inner).collect();
    let mut r = [0f32; N_REGS];
    r.copy_from_slice(&regs);
    let sp = formula::run_program(&ops, &mut r, dt, i_syn);
    Ok((r.to_vec(), spike_to_float(sp)))
}

// ═══════════════════════════════════════════════════════════════
//  Neuron — красивый билдер
// ═══════════════════════════════════════════════════════════════

fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Имена, занятые методами/свойствами билдера: параметр/состояние с таким
/// именем не будет доступно как атрибут (`__getattr__` вызывается при промахе).
fn is_reserved_name(name: &str) -> bool {
    matches!(
        name,
        "I" | "name" | "ops" | "n_ops" | "n_states" | "n_params"
            | "state" | "param" | "compile" | "simulate"
            | "leak" | "ohmic" | "add" | "mul" | "div" | "cmp_ge" | "cmp_gt"
            | "blend" | "gate_step" | "sigmoid" | "exp" | "fire_reset"
            | "refr_clamp" | "refr_timer" | "const_" | "stimulus" | "scope"
    )
}

// Макросы слотов движка обязаны быть объявлены ДО `compile()`,
// который их использует (Rust требует макрос до точки вызова в модуле).

macro_rules! formula_slot_registered {
    ($id:expr) => {
        match $id {
            0 => FormulaModel::<0>::is_registered(),
            1 => FormulaModel::<1>::is_registered(),
            2 => FormulaModel::<2>::is_registered(),
            3 => FormulaModel::<3>::is_registered(),
            4 => FormulaModel::<4>::is_registered(),
            5 => FormulaModel::<5>::is_registered(),
            6 => FormulaModel::<6>::is_registered(),
            7 => FormulaModel::<7>::is_registered(),
            8 => FormulaModel::<8>::is_registered(),
            9 => FormulaModel::<9>::is_registered(),
            10 => FormulaModel::<10>::is_registered(),
            11 => FormulaModel::<11>::is_registered(),
            12 => FormulaModel::<12>::is_registered(),
            13 => FormulaModel::<13>::is_registered(),
            14 => FormulaModel::<14>::is_registered(),
            15 => FormulaModel::<15>::is_registered(),
            _ => true,
        }
    };
}

macro_rules! formula_slot_register {
    ($id:expr, $program:expr) => {
        match $id {
            0 => FormulaModel::<0>::register($program),
            1 => FormulaModel::<1>::register($program),
            2 => FormulaModel::<2>::register($program),
            3 => FormulaModel::<3>::register($program),
            4 => FormulaModel::<4>::register($program),
            5 => FormulaModel::<5>::register($program),
            6 => FormulaModel::<6>::register($program),
            7 => FormulaModel::<7>::register($program),
            8 => FormulaModel::<8>::register($program),
            9 => FormulaModel::<9>::register($program),
            10 => FormulaModel::<10>::register($program),
            11 => FormulaModel::<11>::register($program),
            12 => FormulaModel::<12>::register($program),
            13 => FormulaModel::<13>::register($program),
            14 => FormulaModel::<14>::register($program),
            15 => FormulaModel::<15>::register($program),
            _ => Err("formula: slot out of range".into()),
        }
    };
}

/// Собирает нейрона из ops: состояния/параметры — атрибуты, ops — методы,
/// результат одного op можно передать в другой (dataflow).
#[pyclass(name = "Neuron", module = "taas_checkmate.formula")]
pub struct PyNeuronBuilder {
    pub name: String,
    pub states: Vec<(String, f32)>,
    pub params: Vec<(String, f32)>,
    pub program: Vec<OpNode>,
    pub next_temp: usize,
}

impl PyNeuronBuilder {
    fn emit(
        &mut self,
        kind: OpKind,
        ins: &[&PyReg],
        c: f32,
        c2: f32,
        into: Option<&PyReg>,
    ) -> PyResult<PyReg> {
        let out_reg = match into {
            Some(r) => r.idx,
            None => {
                let t = self.next_temp;
                self.next_temp += 1;
                t
            }
        };
        for r in ins {
            if r.idx >= N_REGS {
                return Err(value_error(format!(
                    "formula: input register {} out of range (N_REGS = {N_REGS})",
                    r.idx
                )));
            }
        }
        if out_reg >= N_REGS {
            return Err(value_error(format!(
                "formula: register {out_reg} out of range (N_REGS = {N_REGS})"
            )));
        }
        if (P0..P0 + 8).contains(&out_reg) {
            return Err(value_error(
                "formula: param registers p0..p7 are read-only — write into a state or temp",
            ));
        }
        let mut in_regs = [S0; 4];
        for (j, r) in ins.iter().enumerate().take(4) {
            in_regs[j] = r.idx;
        }
        let name = match into {
            Some(r) => r.name.clone(),
            None => format!("t{}", out_reg - TEMP0),
        };
        self.program.push(OpNode {
            kind,
            in_regs,
            out_reg,
            c,
            c2,
        });
        Ok(PyReg { idx: out_reg, name })
    }

    fn state_by_idx(&self, i: usize) -> PyReg {
        PyReg {
            idx: S0 + i,
            name: self.states[i].0.clone(),
        }
    }
    fn param_by_idx(&self, i: usize) -> PyReg {
        PyReg {
            idx: P0 + i,
            name: self.params[i].0.clone(),
        }
    }
    /// Регистр входного тока `i_syn` (в Python — свойство `n.I`).
    pub fn i_syn_reg(&self) -> PyReg {
        PyReg {
            idx: I_SYN,
            name: "i_syn".into(),
        }
    }

    fn reg_name(&self, r: usize) -> String {
        if r < S0 + self.states.len() {
            return self.states[r - S0].0.clone();
        }
        if r < S0 + 4 {
            return format!("s{}", r - S0);
        }
        if r >= P0 && r < P0 + self.params.len() {
            return self.params[r - P0].0.clone();
        }
        if (P0..P0 + 8).contains(&r) {
            return format!("p{}", r - P0);
        }
        if r == I_SYN {
            return "i_syn".into();
        }
        if (TEMP0..N_REGS).contains(&r) {
            return format!("t{}", r - TEMP0);
        }
        format!("r{r}")
    }

    fn program_lines(&self) -> Vec<String> {
        self.program
            .iter()
            .map(|op| {
                let kind = op_kind_name(op.kind);
                let arity = formula::op_arity(op.kind);
                let out = self.reg_name(op.out_reg);
                // repr обязан быть ASCII (Windows-консоль cp1251 падает на юникоде)
                let line = match op.kind {
                    OpKind::Const => format!("{out} <- Const({})", op.c),
                    OpKind::Stimulus => format!("{out} <- Stimulus"),
                    _ => {
                        let ins: Vec<String> =
                            op.in_regs[..arity].iter().map(|r| self.reg_name(*r)).collect();
                        format!("{out} <- {kind}({})", ins.join(", "))
                    }
                };
                if matches!(op.kind, OpKind::FireReset) {
                    format!("{line}  [spike]")
                } else {
                    line
                }
            })
            .collect()
    }
}

#[pymethods]
impl PyNeuronBuilder {
    /// Создать билдер с именем модели.
    #[new]
    fn new(name: String) -> Self {
        Self {
            name,
            states: Vec::new(),
            params: Vec::new(),
            program: Vec::new(),
            next_temp: TEMP0,
        }
    }

    // ─── Состояния и параметры ───────────────────────────────

    /// Добавить состояние (максимум 4: s0..s3). Возвращает его регистр.
    #[pyo3(signature = (name, default=None))]
    fn state(&mut self, name: String, default: Option<f32>) -> PyResult<PyReg> {
        if self.states.len() >= 4 {
            return Err(value_error("formula: at most 4 states (s0..s3)"));
        }
        if name.is_empty() || self.states.iter().any(|(n, _)| *n == name) {
            return Err(value_error(format!("formula: duplicate/empty state name '{name}'")));
        }
        if is_reserved_name(&name) {
            return Err(value_error(format!(
                "formula: '{name}' — имя занято методом/свойством билдера"
            )));
        }
        if self.params.iter().any(|(n, _)| *n == name) {
            return Err(value_error(format!(
                "formula: state '{name}' уже объявлен как параметр — имена должны быть уникальны"
            )));
        }
        let i = self.states.len();
        self.states.push((name.clone(), default.unwrap_or(0.0)));
        Ok(PyReg { idx: S0 + i, name })
    }

    /// Добавить параметр (максимум 8: p0..p7). Возвращает его регистр.
    ///
    /// Параметр с именем `dt` обязан быть ПЕРВЫМ: движок читает dt из колонки 0.
    #[pyo3(signature = (name, default=None))]
    fn param(&mut self, name: String, default: Option<f32>) -> PyResult<PyReg> {
        if self.params.len() >= 8 {
            return Err(value_error("formula: at most 8 params (p0..p7)"));
        }
        if name.is_empty() || self.params.iter().any(|(n, _)| *n == name) {
            return Err(value_error(format!("formula: duplicate/empty param name '{name}'")));
        }
        if is_reserved_name(&name) {
            return Err(value_error(format!(
                "formula: '{name}' — имя занято методом/свойством билдера"
            )));
        }
        if self.states.iter().any(|(n, _)| *n == name) {
            return Err(value_error(format!(
                "formula: param '{name}' уже объявлен как состояние — имена должны быть уникальны"
            )));
        }
        if name == "dt" && !self.params.is_empty() {
            return Err(value_error(
                "formula: param 'dt' must be declared FIRST — the engine reads dt from param column 0",
            ));
        }
        let i = self.params.len();
        self.params.push((name.clone(), default.unwrap_or(0.0)));
        Ok(PyReg { idx: P0 + i, name })
    }

    /// Регистр входного тока `i_syn` (post-synaptic bus).
    #[getter]
    fn I(&self) -> PyReg {
        self.i_syn_reg()
    }

    /// Уже объявленный параметр по имени (свойство-доступ: `n.inv_tau`).
    fn __getattr__(&self, name: String) -> PyResult<PyReg> {
        if let Some(i) = self.params.iter().position(|(n, _)| *n == name) {
            return Ok(self.param_by_idx(i));
        }
        if let Some(i) = self.states.iter().position(|(n, _)| *n == name) {
            return Ok(self.state_by_idx(i));
        }
        Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(format!(
            "'Neuron' object has no param/state named '{name}'"
        )))
    }

    // ─── Опкоды ──────────────────────────────────────────────

    #[pyo3(signature = (v, i, inv_tau, v_rest, into=None))]
    fn leak(&mut self, v: &PyReg, i: &PyReg, inv_tau: &PyReg, v_rest: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Leak, &[v, i, inv_tau, v_rest], 0.0, 0.0, into)
    }
    #[pyo3(signature = (g, v, e, into=None))]
    fn ohmic(&mut self, g: &PyReg, v: &PyReg, e: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Ohmic, &[g, v, e], 0.0, 0.0, into)
    }
    #[pyo3(signature = (a, b, into=None))]
    fn add(&mut self, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Add, &[a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (a, b, into=None))]
    fn mul(&mut self, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Mul, &[a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (a, b, into=None))]
    fn div(&mut self, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Div, &[a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (a, b, into=None))]
    fn cmp_ge(&mut self, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::CmpGe, &[a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (a, b, into=None))]
    fn cmp_gt(&mut self, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::CmpGt, &[a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (mask, a, b, into=None))]
    fn blend(&mut self, mask: &PyReg, a: &PyReg, b: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Blend, &[mask, a, b], 0.0, 0.0, into)
    }
    #[pyo3(signature = (g, g_inf, inv_tau, into=None))]
    fn gate_step(&mut self, g: &PyReg, g_inf: &PyReg, inv_tau: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::GateStep, &[g, g_inf, inv_tau], 0.0, 0.0, into)
    }
    #[pyo3(signature = (x, v_half, k, into=None))]
    fn sigmoid(&mut self, x: &PyReg, v_half: f32, k: f32, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Sigmoid, &[x], v_half, k, into)
    }
    #[pyo3(signature = (x, into=None))]
    fn exp(&mut self, x: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Exp, &[x], 0.0, 0.0, into)
    }
    /// Спайк: v ≥ thresh → reset. Ставит спайк-флаг, который читает refr_timer.
    #[pyo3(signature = (v, thresh, reset, into=None))]
    fn fire_reset(&mut self, v: &PyReg, thresh: &PyReg, reset: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::FireReset, &[v, thresh, reset], 0.0, 0.0, into)
    }
    #[pyo3(signature = (v, refr, reset, into=None))]
    fn refr_clamp(&mut self, v: &PyReg, refr: &PyReg, reset: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::RefrClamp, &[v, refr, reset], 0.0, 0.0, into)
    }
    /// Таймер рефрактерности: использует спайк-флаг последнего fire_reset.
    #[pyo3(signature = (refr, tau_ref, into=None))]
    fn refr_timer(&mut self, refr: &PyReg, tau_ref: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::RefrTimer, &[refr, tau_ref], 0.0, 0.0, into)
    }
    #[pyo3(signature = (value, into=None))]
    fn const_(&mut self, value: f32, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Const, &[], value, 0.0, into)
    }
    #[pyo3(signature = (into=None))]
    fn stimulus(&mut self, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Stimulus, &[], 0.0, 0.0, into)
    }
    #[pyo3(signature = (x, into=None))]
    fn scope(&mut self, x: &PyReg, into: Option<&PyReg>) -> PyResult<PyReg> {
        self.emit(OpKind::Scope, &[x], 0.0, 0.0, into)
    }

    // ─── Прогон / компиляция / интроспекция ─────────────────

    /// Быстрый прогон одного нейрона (для отладки и графиков).
    ///
    /// Returns a `Trace` with `.v` (траектория состояния, названного "v"),
    /// `.spikes` (число спайков), `.steps`, `.dt`.
    #[pyo3(signature = (steps, i_syn, dt=None, state=None, params=None))]
    fn simulate(
        &self,
        steps: usize,
        i_syn: f32,
        dt: Option<f32>,
        state: Option<HashMap<String, f32>>,
        params: Option<HashMap<String, f32>>,
    ) -> PyResult<PyTrace> {
        let state = state.unwrap_or_default();
        let params = params.unwrap_or_default();
        for k in state.keys() {
            if !self.states.iter().any(|(n, _)| n == k) {
                return Err(value_error(format!(
                    "simulate: unknown state '{k}' — known: {}",
                    self.states.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        for k in params.keys() {
            if !self.params.iter().any(|(n, _)| n == k) {
                return Err(value_error(format!(
                    "simulate: unknown param '{k}' — known: {}",
                    self.params.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        let dt = match dt {
            Some(d) => d,
            None => params
                .get("dt")
                .copied()
                .or_else(|| {
                    self.params
                        .iter()
                        .find(|(n, _)| n == "dt")
                        .map(|(_, d)| *d)
                })
                .ok_or_else(|| {
                    value_error("simulate: dt required — pass dt=... or declare param('dt') first")
                })?,
        };
        let mut regs = [0f32; N_REGS];
        for (i, (name, def)) in self.states.iter().enumerate() {
            regs[S0 + i] = state.get(name).copied().unwrap_or(*def);
        }
        for (i, (name, def)) in self.params.iter().enumerate() {
            regs[P0 + i] = params.get(name).copied().unwrap_or(*def);
        }
        let v_idx = self.states.iter().position(|(n, _)| n == "v").unwrap_or(0);
        let mut v = Vec::with_capacity(steps);
        let mut spikes = 0usize;
        for _ in 0..steps {
            let sp = formula::run_program(&self.program, &mut regs, dt, i_syn);
            v.push(regs[S0 + v_idx]);
            if sp != 0.0 {
                spikes += 1;
            }
        }
        Ok(PyTrace {
            v,
            spikes,
            steps,
            dt,
        })
    }

    /// Зарегистрировать программу в движке → `Formula` (популяция SIMD).
    fn compile(&self) -> PyResult<PyFormula> {
        formula::validate_program(&self.program).map_err(value_error)?;
        if !self.params.iter().any(|(n, _)| n == "dt") {
            return Err(value_error(
                "formula: compile requires param 'dt' declared FIRST — \
                 the engine reads dt from param column 0",
            ));
        }
        let mut id = None;
        for i in 0..16usize {
            if !formula_slot_registered!(i) {
                id = Some(i);
                break;
            }
        }
        let id = id.ok_or_else(|| {
            value_error(
                "formula: all 16 engine slots are in use (MAX_FORMULA_MODELS = 16) — \
                 this process cannot register more compiled models",
            )
        })?;
        formula_slot_register!(id, self.program.clone()).map_err(value_error)?;
        Ok(PyFormula {
            id,
            name: self.name.clone(),
            states: self.states.clone(),
            params: self.params.clone(),
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }
    #[getter]
    fn n_ops(&self) -> usize {
        self.program.len()
    }
    #[getter]
    fn n_states(&self) -> usize {
        self.states.len()
    }
    #[getter]
    fn n_params(&self) -> usize {
        self.params.len()
    }
    /// Список строк-опкодов (то же, что печатает repr).
    #[getter]
    fn ops(&self) -> Vec<String> {
        self.program_lines()
    }

    fn __repr__(&self) -> String {
        let mut out = format!(
            "{} - {} ops, {} states, {} params\n",
            self.name,
            self.program.len(),
            self.states.len(),
            self.params.len()
        );
        for line in self.program_lines() {
            out.push_str(&format!("  {line}\n"));
        }
        out.trim_end().to_string()
    }
}

// ═══════════════════════════════════════════════════════════════
//  Слоты движка: FormulaModel<0..15> (const generic → match)
// ═══════════════════════════════════════════════════════════════
// Макросы formula_slot_registered! / formula_slot_register! объявлены выше,
// рядом с билдером (их использует Neuron.compile).

fn make_pop<M: Model>(
    n: usize,
    param_defaults: &[(String, f32)],
    params: &HashMap<String, f32>,
    state_defaults: &[(String, f32)],
    states: &HashMap<String, f32>,
) -> NeuralPopulation<M> {
    let mut pop = NeuralPopulation::<M>::new(n);
    for (i, (name, def)) in param_defaults.iter().enumerate() {
        pop.fill_param(i, params.get(name).copied().unwrap_or(*def));
    }
    for (i, (name, def)) in state_defaults.iter().enumerate() {
        pop.fill_state(i, states.get(name).copied().unwrap_or(*def));
    }
    pop
}

macro_rules! formula_make_pop {
    ($id:expr, $n:expr, $pd:expr, $po:expr, $sd:expr, $so:expr) => {
        match $id {
            0 => PopInner::V0(make_pop::<FormulaModel<0>>($n, $pd, $po, $sd, $so)),
            1 => PopInner::V1(make_pop::<FormulaModel<1>>($n, $pd, $po, $sd, $so)),
            2 => PopInner::V2(make_pop::<FormulaModel<2>>($n, $pd, $po, $sd, $so)),
            3 => PopInner::V3(make_pop::<FormulaModel<3>>($n, $pd, $po, $sd, $so)),
            4 => PopInner::V4(make_pop::<FormulaModel<4>>($n, $pd, $po, $sd, $so)),
            5 => PopInner::V5(make_pop::<FormulaModel<5>>($n, $pd, $po, $sd, $so)),
            6 => PopInner::V6(make_pop::<FormulaModel<6>>($n, $pd, $po, $sd, $so)),
            7 => PopInner::V7(make_pop::<FormulaModel<7>>($n, $pd, $po, $sd, $so)),
            8 => PopInner::V8(make_pop::<FormulaModel<8>>($n, $pd, $po, $sd, $so)),
            9 => PopInner::V9(make_pop::<FormulaModel<9>>($n, $pd, $po, $sd, $so)),
            10 => PopInner::V10(make_pop::<FormulaModel<10>>($n, $pd, $po, $sd, $so)),
            11 => PopInner::V11(make_pop::<FormulaModel<11>>($n, $pd, $po, $sd, $so)),
            12 => PopInner::V12(make_pop::<FormulaModel<12>>($n, $pd, $po, $sd, $so)),
            13 => PopInner::V13(make_pop::<FormulaModel<13>>($n, $pd, $po, $sd, $so)),
            14 => PopInner::V14(make_pop::<FormulaModel<14>>($n, $pd, $po, $sd, $so)),
            15 => PopInner::V15(make_pop::<FormulaModel<15>>($n, $pd, $po, $sd, $so)),
            _ => unreachable!("formula slot out of range"),
        }
    };
}

macro_rules! pop_match {
    ($inner:expr, |$p:ident| $body:expr) => {
        match $inner {
            PopInner::V0($p) => $body,
            PopInner::V1($p) => $body,
            PopInner::V2($p) => $body,
            PopInner::V3($p) => $body,
            PopInner::V4($p) => $body,
            PopInner::V5($p) => $body,
            PopInner::V6($p) => $body,
            PopInner::V7($p) => $body,
            PopInner::V8($p) => $body,
            PopInner::V9($p) => $body,
            PopInner::V10($p) => $body,
            PopInner::V11($p) => $body,
            PopInner::V12($p) => $body,
            PopInner::V13($p) => $body,
            PopInner::V14($p) => $body,
            PopInner::V15($p) => $body,
        }
    };
}

enum PopInner {
    V0(NeuralPopulation<FormulaModel<0>>),
    V1(NeuralPopulation<FormulaModel<1>>),
    V2(NeuralPopulation<FormulaModel<2>>),
    V3(NeuralPopulation<FormulaModel<3>>),
    V4(NeuralPopulation<FormulaModel<4>>),
    V5(NeuralPopulation<FormulaModel<5>>),
    V6(NeuralPopulation<FormulaModel<6>>),
    V7(NeuralPopulation<FormulaModel<7>>),
    V8(NeuralPopulation<FormulaModel<8>>),
    V9(NeuralPopulation<FormulaModel<9>>),
    V10(NeuralPopulation<FormulaModel<10>>),
    V11(NeuralPopulation<FormulaModel<11>>),
    V12(NeuralPopulation<FormulaModel<12>>),
    V13(NeuralPopulation<FormulaModel<13>>),
    V14(NeuralPopulation<FormulaModel<14>>),
    V15(NeuralPopulation<FormulaModel<15>>),
}

// ═══════════════════════════════════════════════════════════════
//  Formula — скомпилированный нейрон
// ═══════════════════════════════════════════════════════════════

/// Скомпилированная модель: зарегистрирована в слоте движка, умеет создавать
/// SIMD-популяции через `population(n, params=..., state=...)`.
#[pyclass(name = "Formula", module = "taas_checkmate.formula")]
pub struct PyFormula {
    pub id: usize,
    pub name: String,
    pub states: Vec<(String, f32)>,
    pub params: Vec<(String, f32)>,
}

#[pymethods]
impl PyFormula {
    /// Создать SIMD-популяцию на `n` нейронов.
    ///
    /// `params` / `state` — словари имя→значение (скаляр, всем нейронам);
    /// значения по умолчанию берутся из билдера.
    #[pyo3(signature = (n, params=None, state=None))]
    fn population(
        &self,
        n: usize,
        params: Option<HashMap<String, f32>>,
        state: Option<HashMap<String, f32>>,
    ) -> PyResult<PyFormulaPopulation> {
        let params = params.unwrap_or_default();
        let state = state.unwrap_or_default();
        for k in params.keys() {
            if !self.params.iter().any(|(nm, _)| nm == k) {
                return Err(value_error(format!(
                    "FormulaPopulation: unknown param '{k}' — known: {}",
                    self.params.iter().map(|(nm, _)| nm.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        for k in state.keys() {
            if !self.states.iter().any(|(nm, _)| nm == k) {
                return Err(value_error(format!(
                    "FormulaPopulation: unknown state '{k}' — known: {}",
                    self.states.iter().map(|(nm, _)| nm.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        let has_dt = self.params.iter().any(|(nm, _)| nm == "dt") || params.contains_key("dt");
        if !has_dt {
            return Err(value_error(
                "FormulaPopulation: the engine reads dt from param column 0 — \
                 declare param('dt') first (or pass dt=... in params)",
            ));
        }
        let inner = formula_make_pop!(self.id, n, &self.params, &params, &self.states, &state);
        Ok(PyFormulaPopulation {
            inner,
            name: self.name.clone(),
            n,
            state_names: self.states.iter().map(|(nm, _)| nm.clone()).collect(),
            param_names: self.params.iter().map(|(nm, _)| nm.clone()).collect(),
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.params.iter().map(|(nm, _)| nm.clone()).collect()
    }
    #[getter]
    fn state_names(&self) -> Vec<String> {
        self.states.iter().map(|(nm, _)| nm.clone()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Formula<{}>(slot={}, {} params, {} states)",
            self.name,
            self.id,
            self.params.len(),
            self.states.len()
        )
    }
}

// ═══════════════════════════════════════════════════════════════
//  FormulaPopulation — SIMD-популяция своего нейрона
// ═══════════════════════════════════════════════════════════════

/// SIMD-популяция собранного нейрона (`NeuralPopulation<FormulaModel<ID>>`).
#[pyclass(name = "FormulaPopulation", module = "taas_checkmate.formula")]
pub struct PyFormulaPopulation {
    inner: PopInner,
    name: String,
    n: usize,
    state_names: Vec<String>,
    param_names: Vec<String>,
}

impl PyFormulaPopulation {
    fn set_named(&mut self, name: &str, val: f32) -> PyResult<()> {
        if let Some(i) = self.param_names.iter().position(|nm| nm == name) {
            pop_match!(&mut self.inner, |p| p.fill_param(i, val));
            Ok(())
        } else if let Some(i) = self.state_names.iter().position(|nm| nm == name) {
            pop_match!(&mut self.inner, |p| p.fill_state(i, val));
            Ok(())
        } else {
            Err(value_error(format!(
                "FormulaPopulation: unknown name '{name}' — params: {:?}, states: {:?}",
                self.param_names, self.state_names
            )))
        }
    }
}

#[pymethods]
impl PyFormulaPopulation {
    /// Один шаг симуляции всей популяции (zero-copy numpy).
    fn step(
        &mut self,
        i_in: Bound<'_, PyArray1<f32>>,
        spikes_out: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<()> {
        let i_ro = i_in.readonly();
        let mut sp_rw = spikes_out.readwrite();
        let i = i_ro.as_slice()?;
        let sp = sp_rw.as_slice_mut()?;
        let n = self.n;
        if i.len() != n || sp.len() != n {
            return Err(value_error(format!(
                "step: expected i_in and spikes_out of length {n}, got {} and {}",
                i.len(),
                sp.len()
            )));
        }
        pop_match!(&mut self.inner, |p| p.step(i, sp));
        Ok(())
    }

    /// Задать параметры/состояния по имени, например `pop.set(v_rest=-70.0, v=-65.0)`.
    #[pyo3(signature = (**kwargs))]
    fn set(&mut self, kwargs: Option<Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let name: String = k.extract()?;
                let val: f64 = v.extract()?;
                self.set_named(&name, val as f32)?;
            }
        }
        Ok(())
    }

    /// Задать колонку параметра/состояния numpy-массивом (per-neuron).
    fn set_array(&mut self, name: &str, array: Bound<'_, PyArray1<f32>>) -> PyResult<()> {
        let src = array.readonly().as_slice()?.to_vec();
        if let Some(i) = self.param_names.iter().position(|nm| nm == name) {
            pop_match!(&mut self.inner, |p| {
                let dest = p.param_slice_mut(i);
                if src.len() != dest.len() {
                    return Err(value_error(format!(
                        "set_array: expected {} elements (n), got {}",
                        dest.len(),
                        src.len()
                    )));
                }
                dest.copy_from_slice(&src);
                Ok(())
            })
        } else if let Some(i) = self.state_names.iter().position(|nm| nm == name) {
            pop_match!(&mut self.inner, |p| {
                let dest = p.state_slice_mut(i);
                if src.len() != dest.len() {
                    return Err(value_error(format!(
                        "set_array: expected {} elements (n), got {}",
                        dest.len(),
                        src.len()
                    )));
                }
                dest.copy_from_slice(&src);
                Ok(())
            })
        } else {
            Err(value_error(format!(
                "FormulaPopulation: unknown name '{name}' — params: {:?}, states: {:?}",
                self.param_names, self.state_names
            )))
        }
    }

    /// Параметр по имени → numpy-массив (копия).
    fn param<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let i = self
            .param_names
            .iter()
            .position(|nm| nm == name)
            .ok_or_else(|| value_error(format!("FormulaPopulation: no param '{name}'")))?;
        let slice = pop_match!(&self.inner, |p| p.param_slice(i));
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Состояние по имени → numpy-массив (копия).
    fn state<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let i = self
            .state_names
            .iter()
            .position(|nm| nm == name)
            .ok_or_else(|| value_error(format!("FormulaPopulation: no state '{name}'")))?;
        let slice = pop_match!(&self.inner, |p| p.state_slice(i));
        Ok(PyArray1::from_slice(py, slice))
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }
    #[getter]
    fn state_names(&self) -> Vec<String> {
        self.state_names.clone()
    }
    #[getter]
    fn param_names(&self) -> Vec<String> {
        self.param_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FormulaPopulation<{}>(n={}, {} params, {} states)",
            self.name,
            self.n,
            self.param_names.len(),
            self.state_names.len()
        )
    }
}

// ═══════════════════════════════════════════════════════════════
//  Trace — результат simulate()
// ═══════════════════════════════════════════════════════════════

/// Результат `Neuron.simulate()`: траектория v, число спайков, шаги, dt.
#[pyclass(name = "Trace", module = "taas_checkmate.formula")]
pub struct PyTrace {
    pub v: Vec<f32>,
    pub spikes: usize,
    pub steps: usize,
    pub dt: f32,
}

#[pymethods]
impl PyTrace {
    #[getter]
    fn v(&self) -> Vec<f32> {
        self.v.clone()
    }
    #[getter]
    fn spikes(&self) -> usize {
        self.spikes
    }
    #[getter]
    fn steps(&self) -> usize {
        self.steps
    }
    #[getter]
    fn dt(&self) -> f32 {
        self.dt
    }
    fn __repr__(&self) -> String {
        format!("Trace(steps={}, dt={}, spikes={})", self.steps, self.dt, self.spikes)
    }
}

// ═══════════════════════════════════════════════════════════════
//  Пресеты
// ═══════════════════════════════════════════════════════════════

/// Готовый LIF-нейрон (абсолютная рефрактерность, как в зоопарке) —
/// можно сразу шагать, симулировать и расширять новыми ops.
#[pyfunction(name = "lif")]
#[pyo3(signature = (name=None))]
fn py_lif(name: Option<String>) -> PyResult<PyNeuronBuilder> {
    let mut n = PyNeuronBuilder::new(name.unwrap_or_else(|| "MyLIF".into()));
    let v = n.state("v".into(), Some(-70.0))?;
    let refr = n.state("refr".into(), Some(0.0))?;
    let _dt = n.param("dt".into(), Some(0.5))?;
    let inv_tau = n.param("inv_tau".into(), Some(0.1))?;
    let v_rest = n.param("v_rest".into(), Some(-70.0))?;
    let v_thresh = n.param("v_thresh".into(), Some(-54.0))?;
    let v_reset = n.param("v_reset".into(), Some(-70.0))?;
    let tau_ref = n.param("tau_ref".into(), Some(2.0))?;

    let zero = n.const_(0.0, None)?;
    let is_refr = n.cmp_gt(&refr, &zero, None)?;
    let raw = n.leak(&v, &n.i_syn_reg(), &inv_tau, &v_rest, None)?;
    let gated = n.blend(&is_refr, &v, &raw, None)?;
    n.fire_reset(&gated, &v_thresh, &v_reset, Some(&v))?;
    n.refr_timer(&refr, &tau_ref, Some(&refr))?;
    Ok(n)
}

/// Регистрация функций модуля `formula` (`lif`, `run_program`).
///
/// Классы регистрируются в `mod.rs` через `export_class!`/`export_class_facade!`,
/// как у всех подмодулей. Здесь — только функции: `#[pyfunction(name = …)]`
/// порождает приватный модуль, который `wrap_pyfunction!` может взять только
/// в пределах этого файла (поэтому из `mod.rs` их экспортировать нельзя).
pub fn register(
    m: &Bound<'_, PyModule>,
    facade: Option<&Bound<'_, PyModule>>,
    root: Option<&Bound<'_, PyModule>>,
) -> PyResult<()> {
    let lif_fn = pyo3::wrap_pyfunction!(py_lif, m)?;
    m.add_function(lif_fn.clone())?;
    if let Some(f) = facade {
        f.add_function(lif_fn.clone())?;
    }
    if let Some(r) = root {
        r.add_function(lif_fn)?;
    }
    let run_fn = pyo3::wrap_pyfunction!(py_run_program, m)?;
    m.add_function(run_fn.clone())?;
    if let Some(f) = facade {
        f.add_function(run_fn.clone())?;
    }
    if let Some(r) = root {
        r.add_function(run_fn)?;
    }
    Ok(())
}
