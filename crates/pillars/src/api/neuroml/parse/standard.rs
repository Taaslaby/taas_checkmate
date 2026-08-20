use super::*;

use crate::kernels::hh_channels::{
    fractional_gate_to_markov, gate_kinetics_to_markov, GateKinetics, LemsExpr, RateForm, TauForm,
    VarForm,
};

/// Standard HH channel families — the two layouts in
/// [`crate::kernels::hh_channels::standard_channels`] (NaV: m³h, KV: n⁴). A
/// parsed channel must match one of these **bit-exactly** to be lowered onto
/// the canonical `HHModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StandardChannel {
    Nav,
    Kv,
}

/// If a channel's gates (kinetics + exponents) bit-match a standard channel,
/// return which family it belongs to. Bit-exact matching is deliberate: only
/// channels numerically identical to the NEURON-validated NaV/KV layouts may
/// be mapped onto `HHModel`, otherwise their dynamics would silently diverge.
pub(crate) fn standard_role(ch: &IrChannel) -> Option<StandardChannel> {
    // Дробные гейты не вписываются в канонический HHModel (взвешенная сумма
    // субгейтов), поэтому канал с gateFractional никогда не считается стандартным.
    if !ch.fractional_gates.is_empty() {
        return None;
    }
    let std = crate::kernels::hh_channels::standard_channels();
    'outer: for sc in &std {
        if sc.gates.len() != ch.gates.len() {
            continue;
        }
        for (sg, g) in sc.gates.iter().zip(&ch.gates) {
            if sg.power != g.instances {
                continue 'outer;
            }
            match &g.kinetics {
                Some(k) if *k == sg.kinetics => {}
                _ => continue 'outer,
            }
        }
        // All gates bit-match: classify by the channel's canonical name.
        return match sc.name {
            "NaV" => Some(StandardChannel::Nav),
            "KV" => Some(StandardChannel::Kv),
            // A future standard channel we don't recognise as NaV/KV — treat it
            // as non-standard rather than mis-mapping it onto HHModel.
            _ => None,
        };
    }
    None
}

/// Resolve a parsed `ionChannelHH` declaration into a runnable channel.
///
/// Fails with a clear error if any gate is recognised but not yet compilable
/// (e.g. `gateHHInstantaneous`) — a referenced channel must not be silently
/// dropped.
/// Bind the `rateScale` (→ Q10 multiplier) and `temperature` (→ Kelvin)
/// `<Requirement>`s inside a gate's custom LEMS expressions. `rateScale` is 1.0
/// when the gate has no `q10Settings`.
fn bind_kinetics_requirements(kinetics: GateKinetics, q10: f32, temp_k: f32) -> GateKinetics {
    let bind_expr = |e: LemsExpr| e.bind_rate_scale(q10).bind_temperature(temp_k);
    match kinetics {
        GateKinetics::Rates { forward, reverse } => GateKinetics::Rates {
            forward: match forward {
                RateForm::Closed(cf) => RateForm::Closed(cf),
                RateForm::Expr(e) => RateForm::Expr(bind_expr(e)),
            },
            reverse: match reverse {
                RateForm::Closed(cf) => RateForm::Closed(cf),
                RateForm::Expr(e) => RateForm::Expr(bind_expr(e)),
            },
        },
        GateKinetics::TauInf { steady_state, tau } => GateKinetics::TauInf {
            steady_state: match steady_state {
                VarForm::Closed(cf) => VarForm::Closed(cf),
                VarForm::Expr(e) => VarForm::Expr(bind_expr(e)),
            },
            tau: match tau {
                TauForm::Fixed(t) => TauForm::Fixed(t),
                TauForm::Expr(e) => TauForm::Expr(bind_expr(e)),
            },
        },
        GateKinetics::Instantaneous { steady_state } => GateKinetics::Instantaneous {
            steady_state: match steady_state {
                VarForm::Closed(cf) => VarForm::Closed(cf),
                VarForm::Expr(e) => VarForm::Expr(bind_expr(e)),
            },
        },
    }
}

pub(crate) fn resolve_channel(
    ch: &IrChannel,
    e_rev: f32,
    g_max: f32,
    celsius: f32,
) -> Result<ResolvedChannel, NeuromlError> {
    let mut gates = Vec::with_capacity(ch.gates.len());
    let mut q10s = Vec::with_capacity(ch.gates.len());
    for g in &ch.gates {
        let kinetics = g.kinetics.clone().ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel `{}`: gate `{}` ({}) not yet supported by the runtime",
                ch.id, g.id, g.kind
            ))
        })?;
        // Custom taus with a `rateScale` Requirement bind it to the gate's Q10
        // multiplier (jNeuroML semantics; 1.0 when no q10Settings); custom
        // rates/steady-states with a `temperature` Requirement bind the
        // simulation temperature in Kelvin.
        let q10 = g.q10.map(|q| q.factor_at(celsius)).unwrap_or(1.0);
        let temp_k = celsius + 273.15;
        let kinetics = bind_kinetics_requirements(kinetics, q10, temp_k);
        gates.push((kinetics, g.instances));
        q10s.push(q10);
    }
    Ok(ResolvedChannel {
        gates,
        q10s,
        e_rev,
        g_max,
    })
}

/// Resolve a parsed `ionChannelHH` whose gates include `gateFractional` into a
/// runnable kinetic-scheme channel: HH gates drop to 2-state Markov, fractional
/// gates expand to their product-space Markov scheme. Runs on
/// [`crate::api::MarkovPopulation`] (scalar) — the SIMD HH engine cannot carry
/// a weighted-sum gate.
pub(crate) fn resolve_markov_channel(
    ch: &IrChannel,
    e_rev: f32,
    g_max: f32,
    celsius: f32,
) -> Result<ResolvedMarkovChannel, NeuromlError> {
    let mut gates = Vec::with_capacity(ch.gates.len() + ch.fractional_gates.len());
    let mut q10s = Vec::with_capacity(ch.gates.len() + ch.fractional_gates.len());
    for g in &ch.gates {
        let kinetics = g.kinetics.clone().ok_or_else(|| {
            NeuromlError::parse(format!(
                "channel `{}`: gate `{}` ({}) not yet supported by the runtime",
                ch.id, g.id, g.kind
            ))
        })?;
        // Same `rateScale` → Q10 / `temperature` → K binding as
        // `resolve_channel`.
        let q10 = g.q10.map(|q| q.factor_at(celsius)).unwrap_or(1.0);
        let temp_k = celsius + 273.15;
        let kinetics = bind_kinetics_requirements(kinetics, q10, temp_k);
        gates.push(gate_kinetics_to_markov(kinetics, g.instances));
        q10s.push(q10);
    }
    for fg in &ch.fractional_gates {
        gates.push(fractional_gate_to_markov(fg));
        // gateFractional has no q10Settings in the NeuroML spec.
        q10s.push(1.0);
    }
    Ok(ResolvedMarkovChannel {
        gates,
        q10s,
        e_rev,
        g_max,
    })
}

/// The standard NaV/KV gate layout, expressed as a runnable channel for a
/// mixed cell that must run on the generic engine (reversal/density come from
/// the `channelDensity`, not the hardcoded standard defaults).
pub(crate) fn standard_resolved(role: StandardChannel, e_rev: f32, g_max: f32) -> ResolvedChannel {
    let idx = match role {
        StandardChannel::Nav => 0,
        StandardChannel::Kv => 1,
    };
    let sc = &crate::kernels::hh_channels::standard_channels()[idx];
    ResolvedChannel {
        gates: sc
            .gates
            .iter()
            .map(|g| (g.kinetics.clone(), g.power))
            .collect(),
        // Standard zookeeper channels carry no q10Settings — temperature
        // scaling stays on the engine's global Q10 (celsius → q10).
        q10s: vec![1.0; sc.gates.len()],
        e_rev,
        g_max,
    }
}
