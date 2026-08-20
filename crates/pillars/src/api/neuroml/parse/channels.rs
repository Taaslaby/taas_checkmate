use super::*;

use crate::kernels::hh_channels::{
    ClosedForm, FractionalGateSpec, FractionalSubgate, GateKinetics, MarkovGateSpec, MarkovRate,
    MarkovTransition, RateForm, TauForm, VarForm,
};
use std::collections::HashMap;

/// Convert a NeuroML2 HH rate/variable node (`forwardRate`, `reverseRate`,
/// `steadyState`) into a gate steady-state [`VarForm`]. The standard NeuroML
/// `type` attributes map onto the fast [`ClosedForm`]; a custom LEMS
/// `ComponentType` (real OSB models) is lowered to a [`LemsExpr`].
///
/// Mapping (NeuroML → pillars convention `arg = −(v + midpoint)/scale`):
///
/// | NeuroML type                       | NeuroML formula                 | ClosedForm                |
/// |------------------------------------|---------------------------------|---------------------------|
/// | `HHExpRate`/`HHExpVariable`        | `r·exp((v−m)/s)`                | `Exp { r, −m, −s }`       |
/// | `HHSigmoidRate`/`HHSigmoidVariable`| `r/(1+exp(−(v−m)/s))`           | `Sigmoid { r, −m, s }`    |
/// | `HHExpLinearRate`/`…Variable`      | `r·x/(1−exp(−x)), x=(v−m)/s`    | `ExpLinear { r/s, −m, s }`|
fn parse_closed_form(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<VarForm, NeuromlError> {
    let ty = attr(node, "type")?;
    // Custom LEMS ComponentType (e.g. `Poirazi_h_ss`)?
    if !matches!(
        ty,
        "HHExpRate"
            | "HHExpVariable"
            | "HHSigmoidRate"
            | "HHSigmoidVariable"
            | "HHExpLinearRate"
            | "HHExpLinearVariable"
    ) {
        if let Some(ct) = comp_types.get(ty) {
            let expr = ct.bind(node, &format!("steadyState `{ty}`"))?;
            return Ok(VarForm::Expr(expr));
        }
        return Err(NeuromlError::parse(format!(
            "unknown HH rate/variable type `{ty}` (and no matching <ComponentType> in the document)"
        )));
    }
    let is_variable = ty.ends_with("Variable");
    // `rate` is per_time for rate forms, dimensionless for variable forms.
    let rate = if is_variable {
        raw_attr(node, "rate")?
    } else {
        to_per_ms(attr(node, "rate")?)?
    };
    let midpoint = to_mv(attr(node, "midpoint")?)?;
    let scale = to_mv(attr(node, "scale")?)?;
    // `scale` sits in the denominator of every closed form (the exponent for
    // Exp/Sigmoid, both the exponent and the `rate/scale` prefactor for
    // ExpLinear). A zero scale would yield ±inf/NaN silently — reject it here.
    if scale == 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{ty}>: `scale` must be non-zero (got 0)"
        )));
    }
    let cf = match ty {
        "HHExpRate" | "HHExpVariable" => ClosedForm::Exp {
            rate,
            midpoint: -midpoint,
            scale: -scale,
        },
        "HHSigmoidRate" | "HHSigmoidVariable" => ClosedForm::Sigmoid {
            rate,
            midpoint: -midpoint,
            scale,
        },
        "HHExpLinearRate" | "HHExpLinearVariable" => ClosedForm::ExpLinear {
            rate: rate / scale,
            midpoint: -midpoint,
            scale,
        },
        _ => unreachable!("checked above"),
    };
    Ok(VarForm::Closed(cf))
}

/// Parse a `timeCourse` node into a gate [`TauForm`]. `fixedTimeCourse` →
/// fixed tau; a custom LEMS `ComponentType` (real Ih/HCN models) → V-dependent
/// expression.
fn parse_time_course(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<TauForm, NeuromlError> {
    let ty = attr(node, "type")?;
    if ty == "fixedTimeCourse" {
        return Ok(TauForm::Fixed(to_ms(attr(node, "tau")?)?));
    }
    if let Some(ct) = comp_types.get(ty) {
        let expr = ct.bind(node, &format!("timeCourse `{ty}`"))?;
        return Ok(TauForm::Expr(expr));
    }
    Err(NeuromlError::parse(format!(
        "unsupported timeCourse type `{ty}` (only fixedTimeCourse or a custom <ComponentType>)"
    )))
}

/// Parse a `q10Settings` node (temperature scaling of gate time courses).
fn parse_q10(node: roxmltree::Node<'_, '_>) -> Result<IrQ10, NeuromlError> {
    Ok(match attr(node, "type")? {
        "q10Fixed" => IrQ10::Fixed(raw_attr(node, "fixedQ10")?),
        "q10ExpTemp" => IrQ10::ExpTemp {
            factor: raw_attr(node, "q10Factor")?,
            experimental_temp: raw_attr(node, "experimentalTemp")?,
        },
        other => {
            return Err(NeuromlError::parse(format!(
                "unknown q10Settings type `{other}`"
            )))
        }
    })
}

/// Parse a gate child of an ion channel into [`IrGate`]. `gateHHrates` and
/// `gateHHtauInf` map onto the runtime [`GateKinetics`]; the remaining HH gate
/// kinds are recognised but not yet compilable (`kinetics = None`).
fn parse_gate(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrGate, NeuromlError> {
    let tag = node.tag_name().name();
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1). Reject zero/negative/
    // fractional before the `as u32` cast (which would silently truncate or
    // wrap) so a malformed model fails loudly instead of with a wrong exponent.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }
    let instances = instances_f as u32;
    let kind: &'static str = match tag {
        "gateHHrates" => "gateHHrates",
        "gateHHtauInf" => "gateHHtauInf",
        "gateHHInstantaneous" => "gateHHInstantaneous",
        "gateHHratesTau" => "gateHHratesTau",
        "gateHHratesInf" => "gateHHratesInf",
        "gateHHratesTauInf" => "gateHHratesTauInf",
        _ => return Err(NeuromlError::parse(format!("<{tag}> is not an HH gate"))),
    };

    let kinetics = match tag {
        "gateHHrates" => {
            let forward = parse_closed_form(child(node, "forwardRate")?, comp_types)?;
            let reverse = parse_closed_form(child(node, "reverseRate")?, comp_types)?;
            Some(GateKinetics::Rates {
                forward: RateForm::from_var_form(forward),
                reverse: RateForm::from_var_form(reverse),
            })
        }
        "gateHHtauInf" => {
            let steady_state = parse_closed_form(child(node, "steadyState")?, comp_types)?;
            let tau = parse_time_course(child(node, "timeCourse")?, comp_types)?;
            Some(GateKinetics::TauInf { steady_state, tau })
        }
        // Algebraic gate: q = steady_state(v[, caConc]) every step, no ODE.
        // Used by real Ca channels (Poirazi cal/cat `h2 = ki/(ki+caConc)`).
        "gateHHInstantaneous" => {
            let steady_state = parse_closed_form(child(node, "steadyState")?, comp_types)?;
            Some(GateKinetics::Instantaneous { steady_state })
        }
        _ => None, // recognised but not yet compilable
    };

    let q10 = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "q10Settings")
        .map(parse_q10)
        .transpose()?;

    Ok(IrGate {
        id,
        instances,
        kinetics,
        kind,
        q10,
    })
}

/// Parse an `ionChannel` / `ionChannelHH` element.
pub(crate) fn parse_ion_channel(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrChannel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let conductance = attr_opt(node, "conductance").map(to_us).transpose()?;
    let mut gates = Vec::new();
    let mut fractional_gates = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "gateHHrates"
            | "gateHHtauInf"
            | "gateHHInstantaneous"
            | "gateHHratesTau"
            | "gateHHratesInf"
            | "gateHHratesTauInf" => gates.push(parse_gate(child, comp_types)?),
            "gateFractional" => fractional_gates.push(parse_fractional_gate(child, comp_types)?),
            _ => {}
        }
    }
    Ok(IrChannel {
        id,
        conductance,
        gates,
        fractional_gates,
    })
}

/// Parse a `gateFractional` into a [`FractionalGateSpec`]. Each `subGate` is an
/// independent tau-inf gate (`dq/dt = (inf − q)/tau`) weighted by its
/// `fractionalConductance`; the gate's open fraction is `q = Σ frac_i·q_i` and
/// `fcond = q^instances`.
fn parse_fractional_gate(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<FractionalGateSpec, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1) — validate before cast.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<gateFractional id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }
    let mut subgates = Vec::new();
    for sub in node.children() {
        if !sub.is_element() || sub.tag_name().name() != "subGate" {
            continue;
        }
        let fractional_conductance = raw_attr(sub, "fractionalConductance")?;
        let steady_state = parse_closed_form(child(sub, "steadyState")?, comp_types)?;
        let tau = parse_time_course(child(sub, "timeCourse")?, comp_types)?;
        subgates.push(FractionalSubgate {
            steady_state: match steady_state {
                VarForm::Closed(cf) => cf,
                VarForm::Expr(_) => {
                    return Err(NeuromlError::parse(format!(
                        "<gateFractional id=\"{id}\">: subGate steadyState uses a custom ComponentType (only closed forms are supported in fractional gates)"
                    )))
                }
            },
            tau: match tau {
                TauForm::Fixed(t) => t,
                TauForm::Expr(_) => {
                    return Err(NeuromlError::parse(format!(
                        "<gateFractional id=\"{id}\">: subGate timeCourse uses a custom ComponentType (only fixedTimeCourse is supported in fractional gates)"
                    )))
                }
            },
            fractional_conductance,
        });
    }
    if subgates.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<gateFractional id=\"{id}\">: no subGate declared"
        )));
    }
    Ok(FractionalGateSpec {
        subgates,
        power: instances_f as u32,
    })
}

/// Parse a `gateKS` (kinetic scheme) into a [`MarkovGateSpec`]. States are
/// declared as `closedState` (relativeConductance 0) / `openState` (1);
/// transitions are `forwardTransition`+`reverseTransition` (each a directed
/// HH rate, `from`→`to` forward / `to`→`from` reverse) or `tauInfTransition`
/// (forward `inf/tau`, reverse `(1−inf)/tau`).
///
/// State ids are resolved to indices in declaration order; a transition that
/// references an unknown state is a hard parse error (never a silent drop).
fn parse_gate_ks(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<MarkovGateSpec, NeuromlError> {
    let tag = node.tag_name().name();
    let id = attr(node, "id")?.to_string();
    // NeuroML `instances` is a PositiveInteger (≥1) — validate before cast.
    let instances_f = raw_attr(node, "instances")?;
    if instances_f < 1.0 || instances_f.fract() != 0.0 {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: `instances` must be a positive integer, got {instances_f}"
        )));
    }

    // States in declaration order (closed before open, per schema).
    let mut state_ids: Vec<String> = Vec::new();
    let mut open: Vec<f32> = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "closedState" => {
                state_ids.push(attr(child, "id")?.to_string());
                open.push(0.0);
            }
            "openState" => {
                state_ids.push(attr(child, "id")?.to_string());
                open.push(1.0);
            }
            _ => {}
        }
    }
    if state_ids.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: no closedState/openState declared"
        )));
    }
    let index: HashMap<&str, usize> = state_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut transitions = Vec::new();
    for t_node in node.children() {
        match t_node.tag_name().name() {
            "forwardTransition" | "reverseTransition" => {
                let from = resolve_state(&index, attr(t_node, "from")?, &id)?;
                let to = resolve_state(&index, attr(t_node, "to")?, &id)?;
                // `rate` child — любой элемент (xs:any в схеме), но по конвенции
                // `<rate type="HH…Rate" …/>` с параметрами baseHHRate.
                let rate_node = t_node.children().find(|c| c.is_element()).ok_or_else(|| {
                    NeuromlError::parse(format!(
                        "<{} id=\"{id}\">: transition missing rate child",
                        t_node.tag_name().name()
                    ))
                })?;
                let rate = match parse_closed_form(rate_node, comp_types)? {
                    VarForm::Closed(cf) => cf,
                    VarForm::Expr(_) => {
                        return Err(NeuromlError::parse(format!(
                            "<{} id=\"{id}\">: transition rate uses a custom ComponentType (only closed HH rate forms are supported in gateKS)",
                            t_node.tag_name().name()
                        )))
                    }
                };
                let (f, t) = if t_node.tag_name().name() == "forwardTransition" {
                    (from, to)
                } else {
                    // reverseTransition: rate applies to `to`→`from`.
                    (to, from)
                };
                transitions.push(MarkovTransition {
                    from: f,
                    to: t,
                    rate: MarkovRate::Directed { rate },
                });
            }
            "tauInfTransition" => {
                let from = resolve_state(&index, attr(t_node, "from")?, &id)?;
                let to = resolve_state(&index, attr(t_node, "to")?, &id)?;
                let steady_state = parse_closed_form(child(t_node, "steadyState")?, comp_types)?;
                let tau = parse_time_course(child(t_node, "timeCourse")?, comp_types)?;
                transitions.push(MarkovTransition {
                    from,
                    to,
                    rate: MarkovRate::TauInfForward {
                        steady_state: steady_state.clone(),
                        tau: tau.clone(),
                    },
                });
                transitions.push(MarkovTransition {
                    from: to,
                    to: from,
                    rate: MarkovRate::TauInfReverse { steady_state, tau },
                });
            }
            _ => {}
        }
    }
    if transitions.is_empty() {
        return Err(NeuromlError::parse(format!(
            "<{tag} id=\"{id}\">: no transitions declared"
        )));
    }

    Ok(MarkovGateSpec {
        open,
        transitions,
        power: instances_f as u32,
    })
}

/// Resolve a transition `from`/`to` state id to its index, or fail clearly.
fn resolve_state(
    index: &HashMap<&str, usize>,
    state_id: &str,
    gate_id: &str,
) -> Result<usize, NeuromlError> {
    index.get(state_id).copied().ok_or_else(|| {
        NeuromlError::parse(format!(
            "gateKS `{gate_id}`: transition references unknown state `{state_id}`"
        ))
    })
}

/// Parse an `ionChannelKS` element into [`IrKsChannel`].
pub(crate) fn parse_ion_channel_ks(
    node: roxmltree::Node<'_, '_>,
    comp_types: &HashMap<String, ComponentType>,
) -> Result<IrKsChannel, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let conductance = attr_opt(node, "conductance").map(to_us).transpose()?;
    let mut gates = Vec::new();
    for child in node.children() {
        if child.is_element() && child.tag_name().name() == "gateKS" {
            gates.push(parse_gate_ks(child, comp_types)?);
        }
    }
    Ok(IrKsChannel {
        id,
        conductance,
        gates,
    })
}
