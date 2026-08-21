"""Generate NEURON .mod mechanisms from the Poirazi 2003 NeuroML2 channels.

Mini-jNeuroML: reads the NML `ionChannelHH` files (gates + custom LEMS
ComponentTypes) and emits a NEURON `.mod` per channel with the exact same
kinetics, so the pillars engine and the NEURON reference run the same
equations. jNeuroML (Java) is not available on this machine, hence this.

Supported NML forms (all that Poirazi 2003 uses):
  * gateHHrates  — forward/reverse closed forms (HHExpRate/HHSigmoidRate/
    HHExpLinearRate) OR custom ComponentType expressions (cagk: Ca- and
    temperature-dependent).
  * gateHHtauInf — steadyState (HHSigmoidVariable or custom expr) + timeCourse
    (fixedTimeCourse or custom expr).
  * gateHHInstantaneous — algebraic gate, evaluated in BREAKPOINT (no STATE).
  * q10Settings (q10Fixed / q10ExpTemp) — rate multiplier from `celsius`.
  * Ca-dependent gates: `caConc` in the expression maps to NEURON `cai`
    (USEION ca READ cai).
  * `rateScale` / `temperature` Requirements: rateScale = Q10 multiplier,
    temperature = celsius + 273.15 (K).
  * `no_inact` Parameter on a steadyState (na/nadend s-gate).

Channels referenced by `channelDensityGHK2` (cal, cat) are emitted with the
GHK flux current (exact Poirazi cal.mod/cat.mod form); all others are ohmic
(NONSPECIFIC_CURRENT, RANGE erev).
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

NS = {"nml": "http://www.neuroml.org/schema/neuroml2"}


def qname(tag):
    return tag.split("}")[-1]


def attr(el, name):
    return el.attrib.get(name)


# ─── LEMS expression → NEURON expression ────────────────────────────────
# We translate the small expression language used by the Poirazi ComponentTypes:
# numbers, `v`, `caConc`, arithmetic, `exp(...)`, piecewise `Case`s with
# Fortran comparisons, and the `rateScale`/`temperature` requirements.

def lems_to_neuron(expr, ctx):
    """expr: string from a DerivedVariable/TimeDerivative `value` attribute.
    ctx: dict with 'cai' (neuron cai symbol), 'q10' (rateScale value),
         'temp' (temperature K value). Returns a NEURON expression string."""
    expr = expr.strip()
    # Direct symbol substitutions (whole-token).
    def subst(m):
        tok = m.group(0)
        if tok == "v":
            return "v"
        if tok == "caConc":
            return ctx["cai"]
        if tok == "rateScale":
            return ctx["q10"]
        if tok == "temperature":
            return ctx["temp"]
        return tok

    # Protect function names / constants, then substitute known symbols.
    # Simple approach: regex for identifiers not followed by '(' and not in a
    # small known-constant set.
    def ident_subst(m):
        tok = m.group(0)
        if tok in ("exp", "sin", "cos", "log", "fabs", "sqrt"):
            return tok
        if tok == "v":
            return "(v*0.001)"  # LEMS Volts -> NEURON mV
        if tok == "caConc":
            return ctx["cai"]
        if tok == "rateScale":
            return ctx["q10"]
        if tok == "temperature":
            return ctx["temp"]
        # Unknown identifier: leave as-is (constants like TIME_SCALE were
        # already inlined by the caller, or it's a local derived variable).
        return tok

    out = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", ident_subst, expr)
    # Fortran comparisons .lt. .le. .gt. .ge. .eq. .neq.
    out = out.replace(".lt.", "<").replace(".le.", "<=")
    out = out.replace(".gt.", ">").replace(".ge.", ">=")
    out = out.replace(".eq.", "==").replace(".neq.", "!=")
    return out


def make_piecewise(cases, default, ctx):
    """cases: list of (condition_neuron, value_neuron). Build a NEURON
    expression: `(cond1 ? val1 : (cond2 ? val2 : default))` — NEURON's
    `if (cond) { ... } else { ... }` in expression form via nested ternaries
    is not available, so we emit a PROCEDURE/FUNCTION instead. We handle this
    at the caller by generating a FUNCTION that assigns the value."""
    # Represent as a NEURON FUNCTION body: `if (c1) { f = v1 } else if ...`.
    lines = []
    for i, (cond, val) in enumerate(cases):
        kw = "if" if i == 0 else "else if"
        lines.append(f"    {kw} ({cond}) {{ f = {val} }}")
    lines.append(f"    else {{ f = {default} }}")
    return "\n".join(lines)


class LemsExpr:
    """A parsed LEMS expression tree (subset needed by Poirazi)."""

    def __init__(self, s):
        self.s = s.strip()
        self.toks = self._tokenize(self.s)
        self.pos = 0
        self.ast = self._expr()

    # ── tiny recursive-descent parser ──
    def _tokenize(self, s):
        toks = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1
            elif c.isdigit() or (c == "." and i + 1 < len(s) and s[i + 1].isdigit()):
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] in ".eE+-"):
                    # only allow +/- after e
                    if s[j] in "+-" and j > i and s[j - 1] not in "eE":
                        break
                    j += 1
                toks.append(("num", s[i:j]))
                i = j
            elif c.isalpha() or c == "_":
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                toks.append(("id", s[i:j]))
                i = j
            elif c == ".":
                j = i
                while j < len(s) and s[j] == ".":
                    j += 1
                toks.append(("op", s[i:j]))
                i = j
            elif c in "+-*/()":
                toks.append(("op", c))
                i += 1
            else:
                i += 1
        return toks

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _expr(self):
        node = self._term()
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("+", "-"):
                self.pos += 1
                node = ("bin", t[1], node, self._term())
            else:
                return node

    def _term(self):
        node = self._factor()
        while True:
            t = self._peek()
            if t and t[0] == "op" and t[1] in ("*", "/"):
                self.pos += 1
                node = ("bin", t[1], node, self._factor())
            else:
                return node

    def _factor(self):
        t = self._peek()
        if t is None:
            return ("num", "0")
        if t[0] == "op" and t[1] == "-":
            self.pos += 1
            return ("neg", self._factor())
        if t[0] == "op" and t[1] == "(":
            self.pos += 1
            node = self._expr()
            if self._peek() and self._peek()[1] == ")":
                self.pos += 1
            return node
        if t[0] == "num":
            self.pos += 1
            return ("num", t[1])
        if t[0] == "id":
            name = t[1]
            self.pos += 1
            if self._peek() and self._peek()[1] == "(":
                self.pos += 1
                args = []
                # parse comma-separated args
                while True:
                    args.append(self._expr())
                    if self._peek() and self._peek()[1] == ",":
                        self.pos += 1
                        continue
                    break
                if self._peek() and self._peek()[1] == ")":
                    self.pos += 1
                return ("call", name, args)
            return ("id", name)
        return ("num", "0")

    def to_neuron(self, ctx):
        return self._to_neuron(self.ast, ctx)

    def _to_neuron(self, node, ctx):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "id":
            # LEMS `v` is in Volts; NEURON's `v` is in mV.
            return {"v": "(v*0.001)", "caConc": ctx["cai"],
                    "rateScale": ctx["q10"],
                    "temperature": ctx["temp"]}.get(node[1], node[1])
        if kind == "neg":
            return f"(-{self._to_neuron(node[1], ctx)})"
        if kind == "bin":
            op = node[1]
            l = self._to_neuron(node[2], ctx)
            r = self._to_neuron(node[3], ctx)
            if op == "/":
                return f"({l}/({r}))"
            return f"({l}{op}{r})"
        if kind == "call":
            name = node[1]
            args = [self._to_neuron(a, ctx) for a in node[2]]
            if name == "exp" and len(args) == 1:
                return f"exp({args[0]})"
            return f"{name}({', '.join(args)})"
        return "0"


def parse_component_types(root):
    """Return {name: {'params': [(name, dim)], 'exposure': LemsExpr or str,
    'cases': [...], 'default': ...}} from <ComponentType> nodes."""
    cts = {}
    for el in root.iter():
        if qname(el.tag) != "ComponentType":
            continue
        name = attr(el, "name")
        params = []
        constants = {}
        requirements = {}
        exposure_expr = None
        # Walk Dynamics
        dyn = None
        for ch in el:
            tag = qname(ch.tag).lower()
            if tag == "parameter":
                params.append((attr(ch, "name"), attr(ch, "dimension")))
            elif tag == "constant":
                # Some exports use <CONSTANT> (all-caps) — normalise tag case.
                constants[attr(ch, "name")] = to_si(
                    attr(ch, "value"), attr(ch, "dimension"))
            elif tag == "requirement":
                requirements[attr(ch, "name")] = True
            elif tag == "dynamics":
                dyn = ch
        # Derived variables: we need the exposure variable (the one with
        # `exposure` attr) as an expression in terms of params/constants/
        # v/caConc. We inline constants and derived-variable definitions.
        derived = {}
        cases = None
        default = None
        if dyn is not None:
            for d in dyn:
                dname = qname(d.tag)
                if dname == "DerivedVariable":
                    dv_name = attr(d, "name")
                    dv_val = attr(d, "value")
                    # inline constants
                    dv_val = inline_constants(dv_val, constants)
                    derived[dv_name] = LemsExpr(dv_val)
                    if attr(d, "exposure"):
                        exposure_expr = LemsExpr(dv_val)
                elif dname == "ConditionalDerivedVariable":
                    dv_name = attr(d, "name")
                    cs = []
                    for c in d:
                        if qname(c.tag) != "Case":
                            continue
                        cond = attr(c, "condition")
                        val = attr(c, "value")
                        if cond is None:
                            default = inline_constants(val, constants)
                        else:
                            cs.append((cond, inline_constants(val, constants)))
                    cases = cs
                    if attr(d, "exposure"):
                        exposure_expr = ("piecewise", cs, default)
        cts[name] = {
            "params": params,
            "constants": constants,
            "requirements": requirements,
            "derived": derived,
            "exposure": exposure_expr,
            "cases": cases,
            "default": default,
        }
    return cts


def inline_constants(expr, constants):
    """Replace constant names in a LEMS expression string with their values."""
    if expr is None:
        return "0"
    for k, v in constants.items():
        expr = re.sub(rf"\b{k}\b", str(v), expr)
    return expr


def parse_leading_f32(s):
    m = re.match(r"\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def to_si(value_str, dimension):
    """Parse a NeuroML value-with-units string to the SI value used in LEMS
    expressions. NEURON's native units are ms/mM/K, but LEMS `v` is in Volts
    while NEURON's `v` is in mV — so voltage quantities are the one dimension
    that needs a unit conversion (mV -> V) to stay dimensionally consistent.
    """
    num = parse_leading_f32(value_str)
    if num is None:
        return 0.0
    if dimension == "voltage" or (value_str and "mV" in value_str):
        num *= 0.001
    return num


def parse_gate(gate, cts):
    """Return a gate dict: {id, kind, instances, kinetics, q10}."""
    kind = qname(gate.tag)
    gid = attr(gate, "id") or "g"
    instances = int(float(attr(gate, "instances") or "1"))
    q10 = None
    for ch in gate:
        if qname(ch.tag) == "q10Settings":
            t = attr(ch, "type")
            if t == "q10Fixed":
                q10 = ("fixed", parse_leading_f32(attr(ch, "fixedQ10")))
            elif t == "q10ExpTemp":
                q10 = ("exptemp", parse_leading_f32(attr(ch, "q10Factor")),
                       parse_leading_f32(attr(ch, "experimentalTemp")))
    kinetics = None
    if kind == "gateHHrates":
        fwd = rev = None
        for ch in gate:
            if qname(ch.tag) == "forwardRate":
                fwd = parse_rate(ch, cts)
            elif qname(ch.tag) == "reverseRate":
                rev = parse_rate(ch, cts)
        kinetics = ("rates", fwd, rev)
    elif kind == "gateHHtauInf":
        ss = tau = None
        for ch in gate:
            if qname(ch.tag) == "steadyState":
                ss = parse_ss(ch, cts)
            elif qname(ch.tag) == "timeCourse":
                tau = parse_tau(ch, cts)
        kinetics = ("tauinf", ss, tau)
    elif kind == "gateHHInstantaneous":
        ss = None
        for ch in gate:
            if qname(ch.tag) == "steadyState":
                ss = parse_ss(ch, cts)
        kinetics = ("instantaneous", ss)
    return {"id": gid, "kind": kind, "instances": instances, "kinetics": kinetics, "q10": q10}


def parse_rate(node, cts):
    t = attr(node, "type")
    if t in ("HHExpRate", "HHExpVariable"):
        return ("expr", f"{parse_leading_f32(attr(node,'rate'))}*exp((v-{parse_leading_f32(attr(node,'midpoint'))})/{parse_leading_f32(attr(node,'scale'))})")
    if t in ("HHSigmoidRate", "HHSigmoidVariable"):
        return ("expr", f"{parse_leading_f32(attr(node,'rate'))}/(1+exp(-(v-{parse_leading_f32(attr(node,'midpoint'))})/{parse_leading_f32(attr(node,'scale'))}))")
    if t in ("HHExpLinearRate", "HHExpLinearVariable"):
        r = parse_leading_f32(attr(node, "rate"))
        m = parse_leading_f32(attr(node, "midpoint"))
        s = parse_leading_f32(attr(node, "scale"))
        return ("expr", f"({r}/{s})*vtrap(-(v-{m}), {s})")
    # Custom ComponentType (cagk): forward/reverse rate expression.
    ct = cts.get(t)
    if ct is not None:
        return ("ct", t, bind_node_params(ct, node))
    raise SystemExit(f"unknown rate type {t}")


def parse_ss(node, cts):
    t = attr(node, "type")
    if t in ("HHExpRate", "HHExpVariable"):
        return ("expr", f"{parse_leading_f32(attr(node,'rate'))}*exp((v-{parse_leading_f32(attr(node,'midpoint'))})/{parse_leading_f32(attr(node,'scale'))})")
    if t in ("HHSigmoidRate", "HHSigmoidVariable"):
        return ("expr", f"{parse_leading_f32(attr(node,'rate'))}/(1+exp(-(v-{parse_leading_f32(attr(node,'midpoint'))})/{parse_leading_f32(attr(node,'scale'))}))")
    if t in ("HHExpLinearRate", "HHExpLinearVariable"):
        r = parse_leading_f32(attr(node, "rate"))
        m = parse_leading_f32(attr(node, "midpoint"))
        s = parse_leading_f32(attr(node, "scale"))
        return ("expr", f"({r}/{s})*vtrap(-(v-{m}), {s})")
    ct = cts.get(t)
    if ct is not None:
        return ("ct", t, bind_node_params(ct, node))
    raise SystemExit(f"unknown steadyState type {t}")


def parse_tau(node, cts):
    t = attr(node, "type")
    if t == "fixedTimeCourse":
        return ("fixed", parse_leading_f32(attr(node, "tau")))
    ct = cts.get(t)
    if ct is not None:
        return ("ct", t, bind_node_params(ct, node))
    raise SystemExit(f"unknown timeCourse type {t}")


def q10_factor(q10, celsius):
    """Q10 multiplier at `celsius` (matches jNeuroML / pillars)."""
    if q10 is None:
        return 1.0
    if q10[0] == "fixed":
        return q10[1] ** ((celsius - 6.3) / 10.0)
    return q10[1] ** ((celsius - q10[2]) / 10.0)


def inline_derived(expr_str, ct):
    """Inline a ComponentType's <DerivedVariable> definitions into an
    expression string (recursively, with constants already inlined)."""
    s = expr_str
    for _ in range(8):
        changed = False
        for name, de in ct["derived"].items():
            if re.search(rf"\b{name}\b", s):
                s = re.sub(rf"\b{name}\b", f"({inline_constants(de.s, ct['constants'])})", s)
                changed = True
        if not changed:
            break
    return s


def fix_neuron_expr(s):
    """Normalise a generated NEURON expression: `v--30` → `v+30`, etc."""
    if s is None:
        return s
    return s.replace("--", "+ ").replace("+-", "-").replace("++", "+ ")


def ct_exposure_neuron(ct, ctx):
    """Emit a NEURON expression for a ComponentType's exposure variable.
    Handles plain expressions and piecewise (ConditionalDerivedVariable).
    DerivedVariable definitions are inlined first."""
    expo = ct["exposure"]
    if isinstance(expo, LemsExpr):
        return fix_neuron_expr(LemsExpr(inline_derived(expo.s, ct)).to_neuron(ctx))
    # piecewise
    cases = expo[1]
    default = expo[2]
    lines = []
    for i, (cond, val) in enumerate(cases):
        cond_s = inline_constants(cond, ct["constants"])
        cond_s = inline_derived(cond_s, ct)
        val_s = inline_constants(val, ct["constants"])
        val_s = inline_derived(val_s, ct)
        cond_n = fix_neuron_expr(lems_to_neuron(cond_s, ctx))
        val_n = fix_neuron_expr(LemsExpr(val_s).to_neuron(ctx))
        kw = "if" if i == 0 else "else if"
        lines.append(f"    {kw} ({cond_n}) {{ f = {val_n} }}")
    default_s = inline_constants(default, ct["constants"])
    default_s = inline_derived(default_s, ct)
    default_n = fix_neuron_expr(LemsExpr(default_s).to_neuron(ctx))
    lines.append(f"    else {{ f = {default_n} }}")
    return ("piecewise_body", "\n".join(lines))


def bind_ct_params(ct, gate_node):
    """Bind the ComponentType's <Parameter>s from the gate node attributes.
    Returns a dict param_name -> (dim, value)."""
    bound = {}
    for (pname, dim) in ct["params"]:
        raw = gate_node.attrib.get(pname)
        if raw is None:
            # default 0 (some exports omit)
            bound[pname] = 0.0
        else:
            v = parse_leading_f32(raw)
            bound[pname] = v
    return bound


def bind_node_params(ct, node):
    """Return a copy of `ct` with the ComponentType's <Parameter>s bound
    from the gate node attributes (midpoint, scale, ...) and baked into
    every derived/case expression, so emission is a pure function of v.

    Parameters inherited from an LEMS base type (e.g. `no_inact` from
    `baseVoltageDepVariable`) are not declared in the file; jNeuroML defaults
    any unbound parameter to 0, so any identifier that is not a known
    derived/constant/requirement or v/caConc/temperature is baked to 0."""
    ct = dict(ct)
    bp = {}
    for pname, dim in ct["params"]:
        raw = node.attrib.get(pname)
        if raw is not None:
            bp[pname] = to_si(raw, dim)
    ct["bound_params"] = bp

    # Identifiers that are legal in the emitted expression (inlined later or
    # runtime symbols). Everything else is an unbound parameter -> 0.
    known = set(ct["derived"].keys()) | set(ct["constants"].keys()) \
        | set(ct["requirements"].keys()) | {"v", "caConc", "temperature"}

    def zero_unbound(s):
        if s is None:
            return s
        def repl(m):
            return m.group(0) if m.group(0) in known else "0"
        # Keep function calls (full identifier followed by '(') —
        # exp/sin/vtrap/etc.; the (?![A-Za-z0-9_]) stops partial-identifier
        # backtracking ("exp" → "ex"+"p").
        return re.sub(
            r"(?<![\d.])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])(?!\s*\()",
            repl, s)

    def bake(s):
        if s is None:
            return s
        for k, v in bp.items():
            s = re.sub(rf"\b{k}\b", str(v), s)
        return zero_unbound(s)

    ct["derived"] = {n: LemsExpr(bake(d.s)) for n, d in ct["derived"].items()}
    expo = ct["exposure"]
    if isinstance(expo, LemsExpr):
        ct["exposure"] = LemsExpr(bake(expo.s))
    elif isinstance(expo, tuple) and expo[0] == "piecewise":
        cases = [(c, bake(v)) for c, v in expo[1]]
        ct["exposure"] = ("piecewise", cases, bake(expo[2]))
        ct["cases"] = cases
        ct["default"] = bake(expo[2])
    return ct


def channel_have_ghk(ghk_channels, ch_name):
    return ch_name in ghk_channels


def gen_mod(ch_id, ch_name, gates, cts, celsius, ghk=False):
    """Generate the .mod text for one ion channel."""
    ctx_q10 = 1.0
    for g in gates:
        if g["q10"] is not None:
            ctx_q10 = q10_factor(g["q10"], celsius)
            break
    temp_k = celsius + 273.15
    ctx = {"cai": "cai", "q10": f"({ctx_q10})", "temp": f"({temp_k})"}

    # Determine state gates (non-instantaneous) and their exponents.
    states = []  # (name, power)
    inst_exprs = []  # (name, power, expr)
    for g in gates:
        kin = g["kinetics"]
        if kin is None:
            continue
        if kin[0] == "instantaneous":
            ss = kin[1]
            expr = ""
            if ss[0] == "expr":
                expr = ss[1]
            elif ss[0] == "ct":
                ct = ss[2]
                # bind params from... gate node params already resolved in ss? We
                # re-resolve: params were bound when parsing? We store raw; bind here.
                expr = ct_exposure_neuron(ct, ctx)
                if isinstance(expr, tuple):
                    expr = expr  # piecewise body handled specially below
            inst_exprs.append((g["id"], g["instances"], expr))
        else:
            states.append((g["id"], g["instances"]))

    # Build rate functions per state gate.
    lines = []
    state_rates = []  # (name, alpha_expr, beta_expr) or (name, ss_expr, tau_expr)
    for g in gates:
        kin = g["kinetics"]
        if kin is None or kin[0] == "instantaneous":
            continue
        gname = g["id"]
        if kin[0] == "rates":
            fwd = kin[1]
            rev = kin[2]
            fa = fwd[1] if fwd[0] == "expr" else None
            ra = rev[1] if rev[0] == "expr" else None
            if fwd[0] == "ct":
                fa = ct_exposure_neuron(fwd[2], ctx)
            if rev[0] == "ct":
                ra = ct_exposure_neuron(rev[2], ctx)
            state_rates.append((gname, fa, ra))
        elif kin[0] == "tauinf":
            ss = kin[1]
            tau = kin[2]
            ss_expr = ss[1] if ss[0] == "expr" else None
            if ss[0] == "ct":
                ss_expr = ct_exposure_neuron(ss[2], ctx)
            tau_expr = tau[1] if tau[0] == "fixed" else None
            if tau[0] == "ct":
                tau_expr = ct_exposure_neuron(tau[2], ctx)
            state_rates.append((gname, ss_expr, tau_expr))

    # Determine whether the channel needs cai.
    needs_cai = "cai" in " ".join(str(s) for s in state_rates + inst_exprs) or ghk

    # ── emit .mod ──
    useion = "USEION ca READ cai\n    " if needs_cai else ""
    out = []
    out.append(f"TITLE Poirazi 2003 CA1 {ch_name} (generated from NeuroML2)")
    out.append(": Auto-generated from the published NeuroML2 channel by")
    out.append(": gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars")
    out.append(": engine and the NML ComponentTypes exactly.")
    out.append("")
    out.append("NEURON {")
    # SUFFIX is the channel **id** — a single .channel.nml file can define
    # several channels (e.g. nadend_1/0_95/0_4335/0_3 in nadend.channel.nml)
    # with identical gate structure but different kinetics.
    out.append(f"    SUFFIX {ch_id}")
    if ghk:
        out.append("    USEION ca READ cai, cao WRITE ica")
        out.append("    RANGE gbar")
    else:
        out.append(f"    {useion}NONSPECIFIC_CURRENT i")
        out.append("    RANGE gbar, erev")
    out.append("}")
    out.append("")
    out.append("UNITS {")
    out.append("    (mA) = (milliamp)")
    out.append("    (mV) = (millivolt)")
    out.append("    (mM) = (milli/liter)")
    out.append("    FARADAY = 96520 (coul)")
    out.append("    R = 8.3134 (joule/degK)")
    out.append("    KTOMV = .0853 (mV/degC)")
    out.append("}")
    out.append("")
    out.append("PARAMETER {")
    out.append("    gbar = 0 (S/cm2)")
    if not ghk:
        out.append("    erev = 0 (mV)")
        if needs_cai:
            out.append("    cai = 1e-4 (mM)")
    else:
        out.append("    cai = 1e-4 (mM)")
        out.append("    cao = 2 (mM)")
    out.append("}")
    out.append("")
    if states:
        out.append(f"STATE {{ {' '.join(s for s, _ in states)} }}")
        out.append("")
    out.append("ASSIGNED {")
    out.append("    g (mho/cm2)")
    if not ghk:
        out.append("    i (mA/cm2)")
    else:
        out.append("    ica (mA/cm2)")
    for s, _ in states:
        out.append(f"    {s}inf")
        out.append(f"    {s}tau")
    out.append("}")
    out.append("")
    out.append("INITIAL {")
    for s, _, _, _ in []:
        pass
    # rates() sets inf/tau; init states to inf.
    if states:
        out.append("    rates(v)")
        for s, _ in states:
            out.append(f"    {s} = {s}inf")
    out.append("}")
    out.append("")
    out.append("BREAKPOINT {")
    if states:
        out.append("    SOLVE states METHOD cnexp")
    out.append("    g = gbar")
    for s, p in states:
        out.append(f"    g = g * {s}^({p})")
    for (iname, ip, iexpr) in inst_exprs:
        out.append(f"    g = g * ({iexpr})^({ip})")
    if ghk:
        out.append("    ica = g * ghk(v, cai, cao)")
    else:
        out.append("    i = g * (v - erev)")
    out.append("}")
    out.append("")
    if states:
        out.append("DERIVATIVE states {")
        out.append("    rates(v)")
        for s, _ in states:
            out.append(f"    {s}' = ({s}inf - {s})/{s}tau")
        out.append("}")
        out.append("")
    def emit_assign(var, val):
        """Append an assignment for `var`, expanding a piecewise body into
        if/else lines (NEURON has no ternary expression)."""
        if isinstance(val, tuple) and val[0] == "piecewise_body":
            body = val[1]
            # body uses `f` as the result variable — rewrite to `var`.
            body = body.replace("{ f =", f"{{ {var} =").replace("{ f =", f"{{ {var} =")
            for line in body.split("\n"):
                out.append("    " + line)
        else:
            out.append(f"    {var} = {val}")

    # rates() sets inf/tau for each state gate. NMODL requires every LOCAL
    # to be declared at the top of the PROCEDURE, so we first collect which
    # gates need scratch variables, emit one LOCAL line, then the body.
    rate_locals = []  # (gname, needs_a, needs_b) for rates-kind
    tau_locals = []  # (gname, needs_i, needs_t) for tauinf-kind
    for g in gates:
        kin = g["kinetics"]
        if kin is None or kin[0] == "instantaneous":
            continue
        gname = g["id"]
        if kin[0] == "rates":
            fwd, rev = kin[1], kin[2]
            fa = fwd[1] if fwd[0] == "expr" else None
            ra = rev[1] if rev[0] == "expr" else None
            if fwd[0] == "ct":
                fa = ct_exposure_neuron(fwd[2], ctx)
            if rev[0] == "ct":
                ra = ct_exposure_neuron(rev[2], ctx)
            rate_locals.append((gname,
                                isinstance(fa, tuple) and fa[0] == "piecewise_body",
                                isinstance(ra, tuple) and ra[0] == "piecewise_body"))
        elif kin[0] == "tauinf":
            ss = kin[1]
            tau = kin[2]
            ss_expr = ss[1] if ss[0] == "expr" else None
            if ss[0] == "ct":
                ss_expr = ct_exposure_neuron(ss[2], ctx)
            tau_expr = tau[1] if tau[0] == "fixed" else None
            if tau[0] == "ct":
                tau_expr = ct_exposure_neuron(tau[2], ctx)
            tau_locals.append((gname,
                               isinstance(ss_expr, tuple) and ss_expr[0] == "piecewise_body",
                               isinstance(tau_expr, tuple) and tau_expr[0] == "piecewise_body"))

    local_names = []
    for gname, na, nb in rate_locals:
        if na or nb:
            local_names += [f"a{gname}", f"b{gname}"]
    for gname, ni, nt in tau_locals:
        if ni or nt:
            local_names += [f"i{gname}", f"t{gname}"]

    out.append("PROCEDURE rates(v (mV)) {")
    if local_names:
        out.append("    LOCAL " + ", ".join(local_names))
    for g in gates:
        kin = g["kinetics"]
        if kin is None or kin[0] == "instantaneous":
            continue
        gname = g["id"]
        if kin[0] == "rates":
            fwd, rev = kin[1], kin[2]
            fa = fwd[1] if fwd[0] == "expr" else None
            ra = rev[1] if rev[0] == "expr" else None
            if fwd[0] == "ct":
                fa = ct_exposure_neuron(fwd[2], ctx)
            if rev[0] == "ct":
                ra = ct_exposure_neuron(rev[2], ctx)
            q = q10_factor(g["q10"], celsius)
            fa_is_pw = isinstance(fa, tuple) and fa[0] == "piecewise_body"
            ra_is_pw = isinstance(ra, tuple) and ra[0] == "piecewise_body"
            if fa_is_pw or ra_is_pw:
                if fa_is_pw:
                    emit_assign(f"a{gname}", fa)
                else:
                    out.append(f"    a{gname} = {fa}")
                if ra_is_pw:
                    emit_assign(f"b{gname}", ra)
                else:
                    out.append(f"    b{gname} = {ra}")
                out.append(f"    {gname}inf = a{gname}/(a{gname}+b{gname})")
                out.append(f"    {gname}tau = 1/((a{gname}+b{gname})*({q}))")
            else:
                out.append(f"    {gname}inf = {fa}/({fa}+{ra})")
                out.append(f"    {gname}tau = 1/(({fa}+{ra})*({q}))")
        elif kin[0] == "tauinf":
            ss = kin[1]
            tau = kin[2]
            ss_expr = ss[1] if ss[0] == "expr" else None
            if ss[0] == "ct":
                ss_expr = ct_exposure_neuron(ss[2], ctx)
            tau_expr = tau[1] if tau[0] == "fixed" else None
            if tau[0] == "ct":
                tau_expr = ct_exposure_neuron(tau[2], ctx)
            q = q10_factor(g["q10"], celsius)
            ss_is_pw = isinstance(ss_expr, tuple) and ss_expr[0] == "piecewise_body"
            tau_is_pw = isinstance(tau_expr, tuple) and tau_expr[0] == "piecewise_body"
            if ss_is_pw or tau_is_pw:
                if ss_is_pw:
                    emit_assign(f"i{gname}", ss_expr)
                else:
                    out.append(f"    i{gname} = {ss_expr}")
                if tau_is_pw:
                    emit_assign(f"t{gname}", tau_expr)
                else:
                    out.append(f"    t{gname} = {tau_expr}")
                out.append(f"    {gname}inf = i{gname}")
                out.append(f"    {gname}tau = t{gname}/({q})")
            else:
                out.append(f"    {gname}inf = {ss_expr}")
                out.append(f"    {gname}tau = {tau_expr}/({q})")
    out.append("}")
    out.append("")
    if ghk:
        out.append("FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {")
        out.append("    LOCAL nu, f")
        out.append("    f = KTF(celsius)/2")
        out.append("    nu = v/f")
        out.append("    ghk = -f*(1. - (ci/co)*exp(nu))*efun(nu)")
        out.append("}")
        out.append("")
        out.append("FUNCTION KTF(celsius (degC)) (mV) {")
        out.append("    KTF = ((25./293.15)*(celsius + 273.15))")
        out.append("}")
        out.append("")
        out.append("FUNCTION efun(z) {")
        out.append("    if (fabs(z) < 1e-4) {")
        out.append("        efun = 1 - z/2")
        out.append("    } else {")
        out.append("        efun = z/(exp(z) - 1)")
        out.append("    }")
        out.append("}")
        out.append("")
    # vtrap helper for ExpLinear.
    if "vtrap(" in " ".join(str(s) for s in state_rates + inst_exprs):
        out.append("FUNCTION vtrap(x, y) {")
        out.append("    if (fabs(x/y) < 1e-6) {")
        out.append("        vtrap = y*(1 - x/y/2)")
        out.append("    } else {")
        out.append("        vtrap = x/(exp(x/y) - 1)")
        out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out)


def parse_channels(dirpath, celsius, ghk_channels):
    """Parse all channels/*.nml in dirpath. Returns {ch_name: mod_text}."""
    cts = {}
    files = [f for f in os.listdir(os.path.join(dirpath, "channels"))
             if f.endswith(".channel.nml")]
    # First pass: collect ComponentTypes across all files.
    for f in files:
        tree = ET.parse(os.path.join(dirpath, "channels", f))
        cts.update(parse_component_types(tree.getroot()))
    mods = {}
    for f in sorted(files):
        ch_name = f[: -len(".channel.nml")]
        tree = ET.parse(os.path.join(dirpath, "channels", f))
        root = tree.getroot()
        for el in root.iter():
            if qname(el.tag) not in ("ionChannel", "ionChannelHH"):
                continue
            ch_id = attr(el, "id")
            gates = [parse_gate(g, cts) for g in el if qname(g.tag).startswith("gateHH")]
            ghk = ch_id in ghk_channels
            mods[ch_id] = gen_mod(ch_id, ch_name, gates, cts, celsius, ghk=ghk)
    return mods


def write_mods(mods, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, txt in mods.items():
        with open(os.path.join(outdir, f"{name}.mod"), "w", encoding="ascii",
                  errors="replace") as fh:
            fh.write(txt)
    return list(mods.keys())


if __name__ == "__main__":
    # CLI: gen_poirazi_mods.py <cell_dir> <out_dir> <celsius> [ghk channels...]
    cell_dir = sys.argv[1]
    out_dir = sys.argv[2]
    celsius = float(sys.argv[3])
    ghk = sys.argv[4].split(",") if len(sys.argv) > 4 else []
    mods = parse_channels(cell_dir, celsius, ghk)
    names = write_mods(mods, out_dir)
    print("generated:", ", ".join(names))
