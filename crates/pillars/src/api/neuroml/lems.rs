//! Custom LEMS `ComponentType` support for gate time courses / steady states.
//!
//! Real OSB models (e.g. the Poirazi 2003 CA1 pyramidal cell) define gate
//! kinetics as arbitrary LEMS expressions in custom `ComponentType`s
//! (`baseVoltageDepTime` / `baseVoltageDepVariable`) instead of the standard
//! `HHExpRate` / `HHSigmoidVariable` / `HHExpLinearRate` / `fixedTimeCourse`
//! closed forms. The canonical parser rejects those with «unsupported
//! timeCourse type», which blocks whole published models.
//!
//! This module parses such component types and lowers the **exposure**
//! variable to a [`LemsExpr`] tree (see `kernels::hh_channels`):
//!
//! ```xml
//! <ComponentType name="Poirazi_h_tau" extends="baseVoltageDepTime">
//!   <Constant name="TIME_SCALE" dimension="time" value="1 ms"/>
//!   <Dynamics>
//!     <DerivedVariable name="V" value="v / VOLT_SCALE"/>
//!     <DerivedVariable name="tmp" value="2 * (1 / (exp((V+145)/-17.5) + exp((V+16.8)/16.5)) + 5)"/>
//!     <ConditionalDerivedVariable name="t" exposure="t">
//!       <Case condition="V .lt. -30" value="tmp * TIME_SCALE"/>
//!       <Case value="1 * TIME_SCALE"/>
//!     </ConditionalDerivedVariable>
//!   </Dynamics>
//! </ComponentType>
//! ```
//!
//! Supported expression surface: numbers (incl. `1e-3`/`9.648e4`), `v`,
//! named constants/parameters/derived variables, `+ - * /`, unary minus,
//! `exp(...)`, and `ConditionalDerivedVariable` `Case`s with Fortran-style
//! comparisons (`.lt.`/`.le.`/`.gt.`/`.ge.`/`.eq.`). Anything else is a
//! honest parse error (no silent approximation).

use std::collections::HashMap;

use roxmltree::Node;

use super::NeuromlError;
use crate::kernels::hh_channels::{CmpOp, LemsExpr, PiecewiseCase};

// ─── Tokenizer ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Neq,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '.' => {
                // Fortran-style comparison: .lt. .le. .gt. .ge. .eq.
                let rest = &s[i..];
                let end = rest[1..]
                    .find('.')
                    .map(|p| p + 2)
                    .ok_or_else(|| format!("unterminated comparison in `{s}`"))?;
                let word = &rest[..end];
                let op = match word.to_ascii_lowercase().as_str() {
                    ".lt." => Tok::Lt,
                    ".le." => Tok::Le,
                    ".gt." => Tok::Gt,
                    ".ge." => Tok::Ge,
                    ".eq." => Tok::Eq,
                    ".neq." => Tok::Neq,
                    other => return Err(format!("unknown comparison operator `{other}`")),
                };
                out.push(op);
                i += end;
            }
            _ if c.is_ascii_digit() => {
                let start = i;
                while i < b.len() && (b[i] as char).is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] as char == '.' {
                    i += 1;
                    while i < b.len() && (b[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] as char == 'e' || b[i] as char == 'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] as char == '+' || b[j] as char == '-') {
                        j += 1;
                    }
                    if j < b.len() && (b[j] as char).is_ascii_digit() {
                        i = j;
                        while i < b.len() && (b[i] as char).is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let num: f32 = s[start..i]
                    .parse()
                    .map_err(|_| format!("bad number `{}` in `{s}`", &s[start..i]))?;
                out.push(Tok::Num(num));
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_')
                {
                    i += 1;
                }
                out.push(Tok::Ident(s[start..i].to_string()));
            }
            other => {
                return Err(format!(
                    "unexpected character `{other}` in expression `{s}`"
                ))
            }
        }
    }
    Ok(out)
}

// ─── Recursive-descent parser ───────────────────────────────────

/// Symbol table for one `ComponentType` body: name → expression. `v` is the
/// builtin membrane voltage; `<Constant>`s → `Const`; `<Parameter>`s →
/// `Param(index)` placeholders; `<DerivedVariable>`s → substituted trees.
type Symbols = HashMap<String, LemsExpr>;

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(s: &str) -> Result<Self, String> {
        Ok(Self {
            toks: tokenize(s)?,
            pos: 0,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expr(&mut self, syms: &Symbols) -> Result<LemsExpr, String> {
        let mut lhs = self.term(syms)?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.next();
                    let rhs = self.term(syms)?;
                    lhs = LemsExpr::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Minus) => {
                    self.next();
                    let rhs = self.term(syms)?;
                    lhs = LemsExpr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn term(&mut self, syms: &Symbols) -> Result<LemsExpr, String> {
        let mut lhs = self.unary(syms)?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.next();
                    let rhs = self.unary(syms)?;
                    lhs = LemsExpr::Mul(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Slash) => {
                    self.next();
                    let rhs = self.unary(syms)?;
                    lhs = LemsExpr::Div(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self, syms: &Symbols) -> Result<LemsExpr, String> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.next();
                Ok(LemsExpr::Neg(Box::new(self.unary(syms)?)))
            }
            Some(Tok::Plus) => {
                self.next();
                self.unary(syms)
            }
            _ => self.primary(syms),
        }
    }

    fn primary(&mut self, syms: &Symbols) -> Result<LemsExpr, String> {
        match self.next() {
            Some(Tok::Num(n)) => Ok(LemsExpr::Const(n)),
            Some(Tok::LParen) => {
                let e = self.expr(syms)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err("expected `)`".to_string()),
                }
            }
            Some(Tok::Ident(name)) => {
                if self.peek() == Some(&Tok::LParen) {
                    if name == "exp" {
                        self.next();
                        let arg = self.expr(syms)?;
                        match self.next() {
                            Some(Tok::RParen) => Ok(LemsExpr::Exp(Box::new(arg))),
                            _ => Err("expected `)` after exp(...)".to_string()),
                        }
                    } else {
                        Err(format!("unsupported function `{name}` in LEMS expression"))
                    }
                } else {
                    syms.get(&name)
                        .cloned()
                        .ok_or_else(|| format!("unknown symbol `{name}` in LEMS expression"))
                }
            }
            _ => Err("unexpected token in LEMS expression".to_string()),
        }
    }

    /// Parse a `Case` condition: `expr OP expr` (e.g. `V .lt. -30`).
    fn condition(&mut self, syms: &Symbols) -> Result<(CmpOp, LemsExpr, LemsExpr), String> {
        let lhs = self.expr(syms)?;
        let op = match self.next() {
            Some(Tok::Lt) => CmpOp::Lt,
            Some(Tok::Le) => CmpOp::Le,
            Some(Tok::Gt) => CmpOp::Gt,
            Some(Tok::Ge) => CmpOp::Ge,
            Some(Tok::Eq) => CmpOp::Eq,
            Some(Tok::Neq) => CmpOp::Neq,
            _ => return Err("expected a comparison operator in `Case` condition".to_string()),
        };
        let rhs = self.expr(syms)?;
        Ok((op, lhs, rhs))
    }
}

/// Compile a LEMS expression string into a [`LemsExpr`] tree.
fn parse_expr(s: &str, syms: &Symbols) -> Result<LemsExpr, String> {
    let mut p = Parser::new(s)?;
    let e = p.expr(syms)?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens in LEMS expression `{s}`"));
    }
    Ok(e)
}

// ─── ComponentType IR ───────────────────────────────────────────

/// Dimension of a `<Parameter>` — controls unit conversion when binding its
/// value from the gate node's attribute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ParamDim {
    None,
    Voltage,
    Time,
}

/// A parsed `<ComponentType>` (gate time course / steady-state).
pub(crate) struct ComponentType {
    /// Parameter names in declaration order + their dimensions.
    pub params: Vec<(String, ParamDim)>,
    /// The exposure variable, fully substituted (only `Param(i)` placeholders
    /// remain, bound per-gate by [`ComponentType::bind`]).
    pub exposure: LemsExpr,
}

impl ComponentType {
    /// Bind `<Parameter>` values (from the gate node's attributes, in
    /// declaration order) — replaces `Param(i)` placeholders with constants.
    pub fn bind(&self, gate: Node<'_, '_>, what: &str) -> Result<LemsExpr, NeuromlError> {
        let mut values = Vec::with_capacity(self.params.len());
        for (name, dim) in &self.params {
            let raw = gate.attribute(name.as_str()).ok_or_else(|| {
                NeuromlError::parse(format!(
                    "{what}: custom ComponentType parameter `{name}` not supplied on the gate"
                ))
            })?;
            let v = match dim {
                ParamDim::None => super::parse::raw(raw)?,
                ParamDim::Voltage => super::parse::to_mv(raw)?,
                ParamDim::Time => super::parse::to_ms(raw)?,
            };
            values.push(v);
        }
        Ok(self.exposure.bind_params(&values))
    }
}

/// Parse a custom `<ComponentType>` node into its IR.
pub(crate) fn parse_component_type(node: Node<'_, '_>) -> Result<ComponentType, NeuromlError> {
    let mut params: Vec<(String, ParamDim)> = Vec::new();
    let mut syms: Symbols = HashMap::new();
    // `v` — the builtin membrane voltage (LEMS).
    syms.insert("v".to_string(), LemsExpr::V);
    // `caConc` — the compartment's internal Ca²⁺ concentration (mM), the
    // builtin of `baseVoltageConcDep*` ComponentTypes (Ca-dependent gates).
    syms.insert("caConc".to_string(), LemsExpr::CaConc);

    for child in node.children() {
        // Tag names are case-insensitive (real models export `<CONSTANT>`).
        let tag = child.tag_name().name().to_ascii_lowercase();
        match tag.as_str() {
            "constant" => {
                let name = child
                    .attribute("name")
                    .ok_or_else(|| NeuromlError::parse("<Constant> missing `name`"))?;
                let value = child.attribute("value").ok_or_else(|| {
                    NeuromlError::parse(format!("<Constant name=\"{name}\"> missing `value`"))
                })?;
                // `value` may carry a unit suffix ("1 ms", "1 mV", "1 K").
                let num = parse_leading_f32(value).ok_or_else(|| {
                    NeuromlError::parse(format!("<Constant name=\"{name}\"> bad value `{value}`"))
                })?;
                syms.insert(name.to_string(), LemsExpr::Const(num));
            }
            "parameter" => {
                let name = child
                    .attribute("name")
                    .ok_or_else(|| NeuromlError::parse("<Parameter> missing `name`"))?;
                let dim = match child.attribute("dimension") {
                    Some("none") | None => ParamDim::None,
                    Some("voltage") => ParamDim::Voltage,
                    Some("time") => ParamDim::Time,
                    Some(other) => {
                        return Err(NeuromlError::parse(format!(
                            "<Parameter name=\"{name}\"> unsupported dimension `{other}`"
                        )))
                    }
                };
                let idx = params.len();
                params.push((name.to_string(), dim));
                syms.insert(name.to_string(), LemsExpr::Param(idx));
            }
            "requirement" => {
                let name = child.attribute("name").unwrap_or("<unknown>");
                let expr = if name == "rateScale" {
                    // jNeuroML binds the `rateScale` Requirement of a custom
                    // tau from the gate's `q10Settings` (a.k.a. the Q10
                    // multiplier); we bind it in `resolve_channel` via
                    // `bind_rate_scale` (default 1.0 = no Q10).
                    LemsExpr::RateScale
                } else if name == "temperature" {
                    // Simulation temperature (K) — e.g. Poirazi `cagk` rates
                    // `K = z·F/(R·T)`. Bound in `resolve_channel` via
                    // `bind_temperature(celsius + 273.15)`.
                    LemsExpr::Temperature
                } else {
                    // Not bound by the gate node — only reachable if the body
                    // references it, which we reject below with a clear error.
                    LemsExpr::Unresolved(format!("Requirement `{name}`"))
                };
                syms.insert(name.to_string(), expr);
            }
            "dynamics" => {
                for d in child.children() {
                    let dtag = d.tag_name().name().to_ascii_lowercase();
                    match dtag.as_str() {
                        "derivedvariable" => {
                            let name = d.attribute("name").ok_or_else(|| {
                                NeuromlError::parse("<DerivedVariable> missing `name`")
                            })?;
                            let value = d.attribute("value").ok_or_else(|| {
                                NeuromlError::parse(format!(
                                    "<DerivedVariable name=\"{name}\"> missing `value`"
                                ))
                            })?;
                            let expr = parse_expr(value, &syms).map_err(NeuromlError::parse)?;
                            syms.insert(name.to_string(), expr);
                        }
                        "conditionalderivedvariable" => {
                            let name = d.attribute("name").ok_or_else(|| {
                                NeuromlError::parse("<ConditionalDerivedVariable> missing `name`")
                            })?;
                            let mut cases = Vec::new();
                            let mut default = None;
                            for c in d.children() {
                                if !c.tag_name().name().eq_ignore_ascii_case("Case") {
                                    continue;
                                }
                                let value = c.attribute("value").ok_or_else(|| {
                                    NeuromlError::parse(format!(
                                        "<Case> in `{name}` missing `value`"
                                    ))
                                })?;
                                let value_expr =
                                    parse_expr(value, &syms).map_err(NeuromlError::parse)?;
                                match c.attribute("condition") {
                                    Some(cond) => {
                                        let (op, lhs, rhs) = Parser::new(cond)
                                            .and_then(|mut p| p.condition(&syms))
                                            .map_err(NeuromlError::parse)?;
                                        cases.push(PiecewiseCase {
                                            op,
                                            lhs,
                                            rhs,
                                            value: value_expr,
                                        });
                                    }
                                    None => {
                                        if default.is_some() {
                                            return Err(NeuromlError::parse(format!(
                                                "<ConditionalDerivedVariable name=\"{name}\">: multiple default `<Case>` (without condition)"
                                            )));
                                        }
                                        default = Some(value_expr);
                                    }
                                }
                            }
                            // A default `Case` (no condition) is optional: real
                            // models (e.g. Poirazi `cagk` beta) cover the whole
                            // space with `.neq.`/`.eq.` conditions, so the
                            // fallback of 0.0 is never reached.
                            let expr = LemsExpr::Piecewise {
                                cases,
                                default: Box::new(default.unwrap_or(LemsExpr::Const(0.0))),
                            };
                            syms.insert(name.to_string(), expr);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // The output variable is the one with an `exposure` attribute.
    let mut exposure: Option<LemsExpr> = None;
    if let Some(dyn_node) = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "Dynamics")
    {
        for d in dyn_node.children() {
            if d.attribute("exposure").is_some() {
                if let Some(name) = d.attribute("name") {
                    if let Some(e) = syms.get(name) {
                        exposure = Some(e.clone());
                    }
                }
            }
        }
    }

    let exposure = exposure.ok_or_else(|| {
        NeuromlError::parse(format!(
            "custom ComponentType `{}`: no exposure variable found",
            node.attribute("name").unwrap_or("<unnamed>")
        ))
    })?;
    // Reject unresolved `Requirement`s the body actually references.
    if expr_has_unresolved(&exposure) {
        return Err(NeuromlError::parse(format!(
            "custom ComponentType `{}`: expression references a `<Requirement>` (not bound by the gate) — unsupported",
            node.attribute("name").unwrap_or("<unnamed>")
        )));
    }
    Ok(ComponentType { params, exposure })
}

fn expr_has_unresolved(e: &LemsExpr) -> bool {
    match e {
        LemsExpr::Unresolved(_) => true,
        LemsExpr::Neg(a) | LemsExpr::Exp(a) => expr_has_unresolved(a),
        LemsExpr::Add(a, b) | LemsExpr::Sub(a, b) | LemsExpr::Mul(a, b) | LemsExpr::Div(a, b) => {
            expr_has_unresolved(a) || expr_has_unresolved(b)
        }
        LemsExpr::Piecewise { cases, default } => {
            cases.iter().any(|c| {
                expr_has_unresolved(&c.lhs)
                    || expr_has_unresolved(&c.rhs)
                    || expr_has_unresolved(&c.value)
            }) || expr_has_unresolved(default)
        }
        _ => false,
    }
}

/// Parse the leading float of a `<Constant value>` that may carry a unit
/// suffix (`"1 ms"`, `"1 mV"`, `"1 K"`, `"8.315"`).
fn parse_leading_f32(s: &str) -> Option<f32> {
    let toks = tokenize(s).ok()?;
    match toks.as_slice() {
        [Tok::Num(n), ..] => Some(*n),
        _ => None,
    }
}
