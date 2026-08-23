//! -------------------------------------------------------------------------
//! Filename:    src/neuroml/loader.rs
//! Class:       i/o
//! Purpose:
//! Description:
//!
//! NeuroML: `load*`-функции, resolution `<include>`
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use std::path::Path;

use super::{
    compile, parse, swc, LoadedNetwork, NeuromlError, CONDUCTANCE_CELSIUS, DEFAULT_DT_MS,
    DEFAULT_DURATION_MS,
};

/// Parse and compile a NeuroML2 document.
///
/// `dt`/`duration` come from a LEMS `<Simulation>` if present, otherwise from
/// the Izhikevich defaults (`0.5 ms` / `200 ms`).
pub fn load(xml: &str) -> Result<LoadedNetwork, NeuromlError> {
    load_at(xml, DEFAULT_DT_MS, DEFAULT_DURATION_MS, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document with an explicit `dt`/`duration` (ms)
/// and simulation temperature `celsius` (°C, Q10 anchor).
pub fn load_at(
    xml: &str,
    dt_ms: f32,
    duration_ms: f32,
    celsius: f32,
) -> Result<LoadedNetwork, NeuromlError> {
    let ir = parse::parse(xml)?;
    compile::compile(ir, dt_ms, duration_ms, celsius)
}

/// Parse and compile a NeuroML2 document with an explicit `dt`/`duration` (ms).
/// Temperature defaults to `CONDUCTANCE_CELSIUS` (6.3 °C, Q10 = 1).
pub fn load_with(xml: &str, dt_ms: f32, duration_ms: f32) -> Result<LoadedNetwork, NeuromlError> {
    load_at(xml, dt_ms, duration_ms, CONDUCTANCE_CELSIUS)
}

/// Load an SWC morphology file (`.swc`) as a single **passive** cable
/// population — the same `MorphoIR` the NeuroML `<morphology>` path
/// produces, so the cable engine does not care where the tree came from.
///
/// Without a biophysics block the cable uses default passive parameters
/// (leak −65 mV). Active channels require the NeuroML `<cell>` path.
pub fn load_swc(path: &Path, dt_ms: f32, duration_ms: f32) -> Result<LoadedNetwork, NeuromlError> {
    let morpho = swc::load_swc_file(path)?;
    compile::compile_swc(morpho, dt_ms, duration_ms, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document **read from disk**, resolving LEMS
/// `<include href="…"/>` elements relative to the file's directory
/// (recursively, depth-limited). This lets split/multi-file models load — the
/// common case for real NeuroML files that pull in shared component types.
///
/// `dt`/`duration` come from a LEMS `<Simulation>` if present, otherwise from
/// the Izhikevich defaults (`0.5 ms` / `200 ms`).
pub fn load_file(path: &Path) -> Result<LoadedNetwork, NeuromlError> {
    load_file_at(path, CONDUCTANCE_CELSIUS)
}

/// Parse and compile a NeuroML2 document **read from disk**, at the given
/// simulation temperature `celsius` (°C). See [`load_file`] for the include
/// resolution semantics.
pub fn load_file_at(path: &Path, celsius: f32) -> Result<LoadedNetwork, NeuromlError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| NeuromlError::parse(format!("cannot read {}: {e}", path.display())))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let merged = resolve_includes(&xml, base)?;
    let ir = parse::parse(&merged)?;
    let (dt, duration) = match &ir.sim {
        Some(s) => (s.dt_ms, s.duration_ms),
        None => (DEFAULT_DT_MS, DEFAULT_DURATION_MS),
    };
    compile::compile(ir, dt, duration, celsius)
}

/// Inline LEMS `<include href="…"/>` elements, resolving each `href` relative
/// to `base` and recursing into the included file. XML declarations in
/// included files are stripped so concatenation stays valid.
///
/// Cycle detection: tracks canonicalized paths of visited files — a cyclic
/// include (A→B→A) errors out instead of infinite recursion.
fn resolve_includes(xml: &str, base: &Path) -> Result<String, NeuromlError> {
    let mut visited = std::collections::HashSet::new();
    resolve_includes_impl(xml, base, &mut visited, 0)
}

fn resolve_includes_impl(
    xml: &str,
    base: &Path,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
    depth: usize,
) -> Result<String, NeuromlError> {
    if depth > 16 {
        return Err(NeuromlError::parse(
            "LEMS include depth exceeded 16 (possible cycle)",
        ));
    }

    // Parse XML with roxmltree — handles edge cases like `<include` inside
    // comments/CDATA, multi-line attributes, single-quoted href, etc.
    let doc = roxmltree::Document::parse(xml).map_err(|e| {
        NeuromlError::parse(format!("XML parse error during include resolution: {e}"))
    })?;

    // Collect include hrefs (file includes only, not segmentGroup inheritance).
    let mut includes: Vec<String> = Vec::new();
    for node in doc.root_element().descendants() {
        if node.tag_name().name().eq_ignore_ascii_case("include") {
            if let Some(href) = node.attribute("href") {
                includes.push(href.to_string());
            }
        }
    }

    if includes.is_empty() {
        return Ok(xml.to_string());
    }

    // Reconstruct XML with includes inlined.
    // We split by `<include` and rebuild — roxmltree doesn't give us source
    // spans, so we do a targeted replacement.
    let mut out = String::with_capacity(xml.len() * 2);
    let mut rest = xml;

    while let Some(start) = find_ascii_case_insensitive(rest, b"<include", 0) {
        out.push_str(&rest[..start]);

        let after = &rest[start..];
        let end = after
            .find('>')
            .ok_or_else(|| NeuromlError::parse("unterminated <include> element"))?;
        let elem = &after[..=end];

        // Extract href (supports both double and single quotes).
        let Some(href) = extract_href(elem) else {
            // `<include segmentGroup="…"/>` — morphology inheritance, not a file.
            out.push_str(elem);
            rest = &after[end + 1..];
            continue;
        };

        // jNeuroML writes absolute paths like `/channels/foo.nml` — strip
        // leading slash so `base.join` resolves relative to the model dir.
        let href_clean = href.trim_start_matches(['/', '\\']);
        let full = base.join(href_clean);

        // Cycle detection: canonicalize path, check if already visited.
        let canonical = full.canonicalize().unwrap_or_else(|_| full.clone());
        if !visited.insert(canonical.clone()) {
            return Err(NeuromlError::parse(format!(
                "cyclic include detected: {} was already included",
                full.display()
            )));
        }

        let included = std::fs::read_to_string(&full).map_err(|e| {
            NeuromlError::parse(format!("cannot read include {}: {e}", full.display()))
        })?;

        let inc_base = full.parent().unwrap_or(base);
        let inc_merged = resolve_includes_impl(&included, inc_base, visited, depth + 1)?;

        // Strip XML declaration from included file (concatenation must stay valid).
        let inc_merged = strip_xml_decl(&inc_merged);
        out.push_str(&inc_merged);
        out.push('\n');

        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Find `needle` (ASCII) in `haystack` starting at `from`, case-insensitively,
/// returning a **byte index into the original** `haystack` (no re-encoding).
fn find_ascii_case_insensitive(haystack: &str, needle: &[u8], from: usize) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.len();
    if n == 0 || from > h.len() {
        return None;
    }
    let mut i = from;
    while i + n <= h.len() {
        if h[i..i + n].eq_ignore_ascii_case(needle) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Pull `href="…"` or `href='…'` out of an `<include …/>` element.
fn extract_href(elem: &str) -> Option<&str> {
    // Try double quotes first.
    if let Some(s) = elem.find("href=\"") {
        let s = s + "href=\"".len();
        let e = elem[s..].find('"')? + s;
        return Some(&elem[s..e]);
    }
    // Fall back to single quotes (valid XML).
    if let Some(s) = elem.find("href='") {
        let s = s + "href='".len();
        let e = elem[s..].find('\'')? + s;
        return Some(&elem[s..e]);
    }
    None
}

/// Drop a leading `<?xml …?>` declaration if present.
fn strip_xml_decl(xml: &str) -> String {
    let t = xml.trim_start();
    if let Some(rest) = t.strip_prefix("<?xml") {
        if let Some(end) = rest.find("?>") {
            return rest[end + 2..].trim_start().to_string();
        }
    }
    xml.to_string()
}
