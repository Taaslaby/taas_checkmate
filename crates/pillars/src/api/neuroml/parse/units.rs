// ─── Quantity parsing ──────────────────────────────────────────

use super::*;

/// Milliseconds per second — unit factor for `s`→`ms` and `per_s`→`per_ms`.
const MS_PER_S: f32 = 1000.0;
/// Millivolts per volt — unit factor for `V`→`mV`.
const MV_PER_V: f32 = 1000.0;

/// Split `"200ms"` → `(200.0, "ms")`; a bare number → `(n, "")`.
fn split_num_unit(s: &str) -> Result<(f32, &str), NeuromlError> {
    let s = s.trim();
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(s.len());
    let num: f32 = s[..end]
        .parse()
        .map_err(|_| NeuromlError::parse(format!("cannot parse quantity `{s}`")))?;
    Ok((num, s[end..].trim()))
}

/// Parse a **time** quantity to milliseconds.
pub(crate) fn to_ms(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "s" => n * MS_PER_S,
        _ => n, // "ms" or bare → already ms
    })
}

/// Parse a **voltage** quantity to millivolts.
pub(crate) fn to_mv(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "V" => n * MV_PER_V,
        _ => n, // "mV" or bare → already mV
    })
}

/// Parse a **rate** quantity (`per_time`) to per-millisecond. NeuroML rates are
/// commonly `per_ms` (e.g. `1per_ms`), but the schema allows `per_s` too.
pub(crate) fn to_per_ms(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "per_s" => n * MS_PER_S,
        _ => n, // "per_ms" or bare → already per ms
    })
}

/// Parse a **firing rate** quantity to Hz (`per_s`). `averageRate` on
/// `poissonFiringSynapse` is canonical `per_s` (Hz); accept explicit `Hz`,
/// `kHz` and bare (Hz) too.
pub(crate) fn to_hz(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "Hz" | "per_s" | "" => n,
        "kHz" => n * 1000.0,
        "per_ms" => n * MS_PER_S,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported rate unit `{other}` (expected Hz/per_s)"
            )))
        }
    })
}

/// Parse a **dimensionless / raw** quantity — the unit suffix is discarded.
pub(crate) fn raw(s: &str) -> Result<f32, NeuromlError> {
    Ok(split_num_unit(s)?.0)
}

/// Parse a **resistance** to MΩ: `mV/MΩ = nA` gives the pillars raw current
/// units directly (nA → per-population scale applied by the compiler).
pub(crate) fn to_mohm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "Mohm" | "" => n,
        "kohm" => n * 1e-3,
        "ohm" => n * 1e-6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported resistance unit `{other}` (expected ohm/kohm/Mohm)"
            )))
        }
    })
}

/// Parse an **axial resistivity** (`intracellularProperties/resistivity`) to
/// Ω·cm. NeuroML2 writes `kohm_cm` (e.g. Poirazi `0.05kohm_cm` = 50 Ω·cm); a
/// bare number falls back to Ω·cm.
pub(crate) fn to_ohm_cm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "ohm_cm" | "" => n,
        "kohm_cm" => n * 1e3,
        "Mohm_cm" => n * 1e6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported resistivity unit `{other}` (expected ohm_cm/kohm_cm)"
            )))
        }
    })
}

/// Parse a **conductance density** (`condDensity`) to S/cm². NeuroML2's
/// canonical unit is `S_per_m2`; legacy models (and the HH `channelDensity`
/// path) write `mS_per_cm2`; a bare number falls back to the legacy `mS/cm²`
/// convention. `rm (Ω·cm²) = 1/g (S/cm²)` follows directly.
pub(crate) fn to_s_per_cm2(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "S_per_m2" => n * 1e-4,
        "S_per_cm2" => n,
        "mS_per_cm2" => n * 1e-3,
        "mS_per_m2" => n * 1e-7,
        "uS_per_cm2" => n * 1e-6,
        "" => n * 1e-3, // legacy: bare = mS/cm²
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported conductance density unit `{other}`"
            )))
        }
    })
}

/// Parse a **single-channel conductance** (`conductance`) to µS. NeuroML2's
/// canonical unit is `S`; real channel conductances are usually written in
/// `pS`/`nS`/`uS`; a bare number falls back to µS (the codebase's legacy raw
/// convention). The cable solver's absolute-conductance contract is µS, so the
/// value comes out directly in µS (`mV/MΩ = nA`, `g[µS]·(E−V)[mV] = nA`).
pub(crate) fn to_us(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "S" => n * 1e6,
        "mS" => n * 1e3,
        "uS" => n,
        "nS" => n * 1e-3,
        "pS" => n * 1e-6,
        "" => n, // legacy: bare = µS
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported conductance unit `{other}` (expected S/mS/uS/nS/pS)"
            )))
        }
    })
}

/// Parse a **specific capacitance** (`specificCapacitance`) to µF/cm².
/// NeuroML2's canonical unit is `F_per_m2` (1 F/m² = 100 µF/cm²); legacy models
/// write `uF_per_cm2`; a bare number falls back to µF/cm² (default 1.0).
pub(crate) fn to_uf_per_cm2(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "F_per_m2" => n * 100.0,
        "F_per_cm2" => n * 1e6,
        "mF_per_cm2" => n * 1000.0,
        "uF_per_cm2" => n,
        "uF_per_m2" => n * 1e-4,
        "" => n, // legacy: bare = µF/cm²
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported specific capacitance unit `{other}`"
            )))
        }
    })
}

/// Parse an ion **concentration** to **mM** (= mol/m³, the LEMS base unit for
/// the `concentration` dimension). NeuroML writes `mM`/`M`/`uM`/`nM` (and the
/// SI `mol_per_m3`/`mol_per_cm3`); a bare number falls back to mM.
pub(crate) fn to_mm(s: &str) -> Result<f32, NeuromlError> {
    let (n, u) = split_num_unit(s)?;
    Ok(match u {
        "mM" | "mol_per_m3" | "" => n,
        "M" => n * 1000.0,
        "uM" => n * 1e-3,
        "nM" => n * 1e-6,
        "mol_per_cm3" => n * 1e6,
        other => {
            return Err(NeuromlError::parse(format!(
                "unsupported concentration unit `{other}` (expected mM/M/uM/nM)"
            )))
        }
    })
}

/// Attribute lookup on a roxmltree node that reports a clear parse error.
pub(crate) fn attr<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Result<&'a str, NeuromlError> {
    node.attribute(name).ok_or_else(|| {
        NeuromlError::parse(format!("<{}> missing `{name}`", node.tag_name().name()))
    })
}

pub(crate) fn attr_opt<'a>(node: roxmltree::Node<'a, 'a>, name: &str) -> Option<&'a str> {
    node.attribute(name)
}

pub(crate) fn time_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    to_ms(attr(node, name)?)
}

pub(crate) fn voltage_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    to_mv(attr(node, name)?)
}

pub(crate) fn raw_attr(node: roxmltree::Node<'_, '_>, name: &str) -> Result<f32, NeuromlError> {
    raw(attr(node, name)?)
}

pub(crate) fn raw_attr_opt(
    node: roxmltree::Node<'_, '_>,
    name: &str,
) -> Result<Option<f32>, NeuromlError> {
    attr_opt(node, name).map(raw).transpose()
}

// ─── Ion-channel / gate parsing ────────────────────────────────

/// Direct element child with the given tag name (error if absent).
pub(crate) fn child<'a>(
    node: roxmltree::Node<'a, 'a>,
    name: &str,
) -> Result<roxmltree::Node<'a, 'a>, NeuromlError> {
    node.children()
        .find(|c| c.is_element() && c.tag_name().name() == name)
        .ok_or_else(|| {
            NeuromlError::parse(format!(
                "<{}> missing child <{name}>",
                node.tag_name().name()
            ))
        })
}
