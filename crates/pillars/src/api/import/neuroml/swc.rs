//! # SWC morphology import
//!
//! Reads the SWC point-list format — one `n T x y z R P` record per line — and
//! lowers it to the **same** [`MorphoIR`](super::parse::MorphoIR) that NeuroML
//! `<morphology>` produces. A single morphology representation means the later
//! cable-equation phase (5d) does not care whether a tree came from NeuroML or
//! SWC.
//!
//! ## Format
//!
//! ```text
//! # comment (lines starting with `#`)
//! n  T  x  y  z  R  P
//! ```
//!
//! * `n` — integer sample id (unique, may be 0/1-based or gapped);
//! * `T` — sample type: 1 soma, 2 axon, 3 basal dendrite, 4 apical dendrite,
//!   5+ custom;
//! * `x y z` — coordinates (µm);
//! * `R` — radius (µm) → converted to NeuroML point `diameter = 2R`;
//! * `P` — parent sample id (`-1` = root).
//!
//! Each non-root record becomes a [`Segment`](super::parse::Segment) whose
//! `proximal` point is the parent's point and whose `parent` carries the parent
//! id (fractional length 1.0). The root (`P = -1`) has no parent and no
//! `proximal`. Segment groups are auto-derived per sample type.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use super::parse::{MorphoIR, Point3d, Segment, SegmentGroup, SegmentParent};
use super::NeuromlError;

/// Canonical group names per SWC sample type, in the deterministic order they
/// are emitted (only non-empty groups are kept).
const GROUP_TYPES: [(i32, &str); 4] = [
    (1, "soma_group"),
    (2, "axon_group"),
    (3, "dendrite_group"),
    (4, "apical_group"),
];

/// Segment `name` label per SWC sample type (used only for readability; the
/// `id` is the unique key).
fn type_label(t: i32) -> Option<&'static str> {
    match t {
        1 => Some("soma"),
        2 => Some("axon"),
        3 => Some("dendrite"),
        4 => Some("apical"),
        _ => None,
    }
}

/// Parse an SWC document into a [`MorphoIR`] named `id`.
///
/// Fails with a clear line-numbered error on malformed records, duplicate
/// sample ids, parent references to unknown samples, or an empty document.
pub(crate) fn parse_swc(text: &str, id: &str) -> Result<MorphoIR, NeuromlError> {
    struct Raw {
        n: usize,
        t: i32,
        point: Point3d,
        parent: i64,
    }

    let mut points: Vec<Raw> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 {
            return Err(NeuromlError::parse(format!(
                "swc line {}: expected `n T x y z R P`, got `{line}`",
                lineno + 1
            )));
        }
        let bad = |what: &str| {
            NeuromlError::parse(format!(
                "swc line {}: cannot parse {what} in `{line}`",
                lineno + 1
            ))
        };
        let n: usize = fields[0].parse().map_err(|_| bad("sample id"))?;
        let t: i32 = fields[1].parse().map_err(|_| bad("type"))?;
        let x: f32 = fields[2].parse().map_err(|_| bad("x"))?;
        let y: f32 = fields[3].parse().map_err(|_| bad("y"))?;
        let z: f32 = fields[4].parse().map_err(|_| bad("z"))?;
        let r: f32 = fields[5].parse().map_err(|_| bad("radius"))?;
        let parent: i64 = fields[6].parse().map_err(|_| bad("parent"))?;
        points.push(Raw {
            n,
            t,
            // SWC radius → NeuroML diameter.
            point: Point3d {
                x,
                y,
                z,
                diameter: 2.0 * r,
            },
            parent,
        });
    }

    if points.is_empty() {
        return Err(NeuromlError::parse("swc: no samples"));
    }

    // Sample id → index into `points` (duplicates are a hard error).
    let mut index: HashMap<usize, usize> = HashMap::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        if index.insert(p.n, i).is_some() {
            return Err(NeuromlError::parse(format!(
                "swc: duplicate sample id {}",
                p.n
            )));
        }
    }

    // Group members by type (deterministic order via `GROUP_TYPES`).
    let mut group_members: HashMap<&'static str, Vec<usize>> = HashMap::new();
    for (t, name) in GROUP_TYPES {
        let members: Vec<usize> = points.iter().filter(|p| p.t == t).map(|p| p.n).collect();
        if !members.is_empty() {
            group_members.insert(name, members);
        }
    }

    let mut segments = Vec::with_capacity(points.len());
    for p in &points {
        // Resolve and validate the parent reference (validity check), then
        // carry the parent id into `SegmentParent`; `proximal` reuses the
        // parent's point (same as its `distal`).
        let parent = if p.parent < 0 {
            None
        } else {
            let pid = p.parent as usize;
            index.get(&pid).ok_or_else(|| {
                NeuromlError::parse(format!(
                    "swc: sample {} references unknown parent {}",
                    p.n, pid
                ))
            })?;
            Some(SegmentParent { segment: pid })
        };
        // `Point3d` is `Copy`, so a parent's point is copied verbatim.
        let proximal = parent.as_ref().map(|sp| points[index[&sp.segment]].point);
        segments.push(Segment {
            id: p.n,
            name: type_label(p.t).map(str::to_string),
            proximal,
            distal: p.point,
            parent,
        });
    }

    let segment_groups = GROUP_TYPES
        .iter()
        .filter_map(|(_, name)| {
            group_members.remove(name).map(|members| SegmentGroup {
                id: (*name).to_string(),
                members,
                includes: Vec::new(),
            })
        })
        .collect();

    Ok(MorphoIR {
        id: id.to_string(),
        segments,
        segment_groups,
    })
}

/// Read and parse an SWC file, naming the morphology after the file stem
/// (e.g. `cell.swc` → `cell`).
pub(crate) fn load_swc_file(path: &Path) -> Result<MorphoIR, NeuromlError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| NeuromlError::parse(format!("cannot read {}: {e}", path.display())))?;
    let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("swc");
    parse_swc(&text, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Сома (root) + два дендритных сегмента: root → d1 → d2.
    const SIMPLE_SWC: &str = "\
# a soma and two dendrites
1 1 0 0 0 10 -1
2 3 0 10 0 1 1
3 3 0 20 0 1 2
";

    #[test]
    fn swc_tree_parses_to_segments_and_groups() {
        let m = parse_swc(SIMPLE_SWC, "test").unwrap();
        assert_eq!(m.id, "test");
        assert_eq!(m.segments.len(), 3);

        // Корень: сома, parent = None, proximal = None, radius→diameter.
        let soma = &m.segments[0];
        assert_eq!(soma.id, 1);
        assert_eq!(soma.name.as_deref(), Some("soma"));
        assert!(soma.parent.is_none());
        assert!(soma.proximal.is_none());
        assert_eq!(soma.distal.diameter, 20.0); // R=10 → diameter=20

        // d1: parent 1, proximal = сома-точка (0,0,0, d=20).
        let d1 = &m.segments[1];
        assert_eq!(d1.id, 2);
        assert_eq!(d1.name.as_deref(), Some("dendrite"));
        let p1 = d1.parent.as_ref().expect("d1 parent");
        assert_eq!(p1.segment, 1);
        let prox = d1.proximal.as_ref().expect("d1 proximal");
        assert_eq!(
            (prox.x, prox.y, prox.z, prox.diameter),
            (0.0, 0.0, 0.0, 20.0)
        );
        assert_eq!(d1.distal.diameter, 2.0); // R=1 → diameter=2

        // d2: parent 2.
        let d2 = &m.segments[2];
        assert_eq!(d2.id, 3);
        assert_eq!(d2.parent.as_ref().expect("d2 parent").segment, 2);

        // Группы: soma_group = {1}, dendrite_group = {2, 3} (тип 3). Нет axon/apical.
        assert_eq!(m.segment_groups.len(), 2);
        assert_eq!(m.segment_groups[0].id, "soma_group");
        assert_eq!(m.segment_groups[0].members, vec![1]);
        assert_eq!(m.segment_groups[1].id, "dendrite_group");
        assert_eq!(m.segment_groups[1].members, vec![2, 3]);
    }

    /// Комментарии и пустые строки игнорируются; тип 4 → apical_group.
    #[test]
    fn swc_ignores_comments_and_groups_apical() {
        let swc = "\
# header comment

1 1 0 0 0 5 -1
2 4 0 1 0 0.5 1

";
        let m = parse_swc(swc, "t").unwrap();
        assert_eq!(m.segments.len(), 2);
        let g = &m.segment_groups[1];
        assert_eq!(g.id, "apical_group");
        assert_eq!(g.members, vec![2]);
    }

    /// Неизвестный родитель — честная ошибка, а не молчаливый дроп связи.
    #[test]
    fn swc_unknown_parent_errors() {
        let swc = "1 1 0 0 0 5 -1\n2 3 0 1 0 1 99\n";
        assert!(
            parse_swc(swc, "t").is_err(),
            "parent 99 должен давать ошибку"
        );
    }

    /// Дубликат sample id — ошибка.
    #[test]
    fn swc_duplicate_id_errors() {
        let swc = "1 1 0 0 0 5 -1\n1 3 0 1 0 1 1\n";
        assert!(
            parse_swc(swc, "t").is_err(),
            "дубликат id должен давать ошибку"
        );
    }

    /// Малформат (меньше 7 полей) и пустой документ — ошибки.
    #[test]
    fn swc_malformed_and_empty_error() {
        assert!(
            parse_swc("1 1 0 0 0\n", "t").is_err(),
            "мало полей → ошибка"
        );
        assert!(
            parse_swc("# только комментарий\n", "t").is_err(),
            "пустой → ошибка"
        );
    }

    /// Сквозной путь: SWC-файл → load_swc → пассивная кабельная сеть,
    /// которая реально интегрируется (V конечен, без warnings).
    #[test]
    fn swc_file_loads_into_runnable_passive_cable() {
        let path = std::env::temp_dir().join("pillars_swc_it.swc");
        std::fs::write(&path, SIMPLE_SWC).unwrap();
        let mut ln = super::super::load_swc(&path, 0.1, 10.0).unwrap();
        assert!(ln.warnings.is_empty(), "{:?}", ln.warnings);
        let info = ln.cable_info(0).expect("SWC — морфологическая клетка");
        assert!(info.compartments.len() >= 3, "сома + 2 дендрита");
        ln.run();
        let v = ln.network.state(0, 0).to_vec();
        assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
        let _ = std::fs::remove_file(&path);
    }
}
