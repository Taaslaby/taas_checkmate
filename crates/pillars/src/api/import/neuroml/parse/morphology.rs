// ─── Morphology parsing ────────────────────────────────────────

use super::*;

/// Parse a `<point>` child (`<proximal>` or `<distal>`) into `Point3d`.
/// Units are NeuroML SI (`1e-6 m` = μm).
fn parse_point(node: roxmltree::Node<'_, '_>) -> Result<Point3d, NeuromlError> {
    Ok(Point3d {
        x: raw(node.attribute("x").unwrap_or("0"))?,
        y: raw(node.attribute("y").unwrap_or("0"))?,
        z: raw(node.attribute("z").unwrap_or("0"))?,
        diameter: raw(node.attribute("diameter").unwrap_or("0"))?,
    })
}

/// Parse a `<segment>` element.
fn parse_segment(node: roxmltree::Node<'_, '_>) -> Result<Segment, NeuromlError> {
    let id: usize = raw_attr(node, "id")? as usize;
    let name = attr_opt(node, "name").map(str::to_string);
    let proximal = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "proximal")
        .map(parse_point)
        .transpose()?;
    let distal = node
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "distal")
        .map(parse_point)
        .ok_or_else(|| NeuromlError::parse(format!("<segment id={id}>: missing <distal>")))??;
    // NeuroML2: `<parent segment="…" fractionAlong="…"/>`. `segmentParent`/
    // `fractionalLength` accepted as a lenient alias (older drafts / our
    // historical fixtures) — the canonical names are `parent`/`fractionAlong`.
    // The fraction is validated as 0..=1 but not stored (the cable lowering
    // attaches children at the parent's distal end).
    let parent = node
        .children()
        .find(|c| c.is_element() && matches!(c.tag_name().name(), "parent" | "segmentParent"))
        .map(|sp| {
            let segment: usize = raw_attr(sp, "segment")? as usize;
            if let Some(frac) = attr_opt(sp, "fractionAlong")
                .or_else(|| attr_opt(sp, "fractionalLength"))
                .map(raw)
                .transpose()?
            {
                if !(0.0..=1.0).contains(&frac) {
                    return Err(NeuromlError::parse(format!(
                        "segment parent `fractionAlong` {frac} out of [0, 1]"
                    )));
                }
            }
            Ok(SegmentParent { segment })
        })
        .transpose()?;
    Ok(Segment {
        id,
        name,
        proximal,
        distal,
        parent,
    })
}

/// Parse a `<segmentGroup>` element.
fn parse_segment_group(node: roxmltree::Node<'_, '_>) -> Result<SegmentGroup, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let mut members = Vec::new();
    let mut includes = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "member" => {
                if let Some(seg) = attr_opt(child, "segment") {
                    members.push(seg.parse::<usize>().map_err(|_| {
                        NeuromlError::parse(format!(
                            "<segmentGroup id=\"{id}\">: bad member segment `{seg}`"
                        ))
                    })?);
                }
            }
            "include" => {
                if let Some(seg_group) = attr_opt(child, "segmentGroup") {
                    includes.push(seg_group.to_string());
                }
            }
            _ => {}
        }
    }
    Ok(SegmentGroup {
        id,
        members,
        includes,
    })
}

/// Parse a `<morphology>` element.
pub(crate) fn parse_morphology(node: roxmltree::Node<'_, '_>) -> Result<MorphoIR, NeuromlError> {
    let id = attr(node, "id")?.to_string();
    let mut segments = Vec::new();
    let mut segment_groups = Vec::new();
    for child in node.children() {
        match child.tag_name().name() {
            "segment" => segments.push(parse_segment(child)?),
            "segmentGroup" => segment_groups.push(parse_segment_group(child)?),
            _ => {}
        }
    }
    Ok(MorphoIR {
        id,
        segments,
        segment_groups,
    })
}
