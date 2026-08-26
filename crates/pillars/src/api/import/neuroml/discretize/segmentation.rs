use super::*;

use std::collections::HashMap;

use crate::api::neuroml::parse::{MorphoIR, Point3d, Segment};
use crate::api::neuroml::NeuromlError;

/// DC space constant (µm) of a cylinder of diameter `d_um`.
fn space_constant_um(d_um: f32, rm_ohm_cm2: f32, ra_ohm_cm: f32) -> f32 {
    // λ(cm) = √(d(cm)·Rm/(4·Ra)); ×1e4 → µm. Folds to `√(d·Rm/(4·Ra))·1e2`.
    (d_um * rm_ohm_cm2 / (4.0 * ra_ohm_cm)).sqrt() * 100.0
}

/// Euclidean distance between two points (µm).
fn dist(p: &Point3d, q: &Point3d) -> f32 {
    let dx = p.x - q.x;
    let dy = p.y - q.y;
    let dz = p.z - q.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Geometry (length + average diameter + axis endpoints) of one morphology
/// segment, resolving an omitted `proximal` against the parent's `distal`.
struct SegmentGeom {
    length_um: f32,
    diameter_um: f32,
    prox_x: f32,
    prox_y: f32,
    prox_z: f32,
    distal_x: f32,
    distal_y: f32,
    distal_z: f32,
}

/// Discretize a morphology into a compartment tree using the d_λ rule.
///
/// # Errors
/// Fails if the morphology has no root (`parent = None` segment), more than one
/// root, a segment referencing an unknown parent, or a zero-diameter segment.
pub(crate) fn discretize(
    morpho: &MorphoIR,
    params: &CableParams,
) -> Result<CompartmentTree, NeuromlError> {
    let segs = &morpho.segments;
    if segs.is_empty() {
        return Err(NeuromlError::parse(
            "discretize: morphology has no segments",
        ));
    }

    // Segment id → index into `segs`.
    let mut index: HashMap<usize, usize> = HashMap::with_capacity(segs.len());
    for (i, s) in segs.iter().enumerate() {
        if index.insert(s.id, i).is_some() {
            return Err(NeuromlError::parse(format!(
                "discretize: duplicate segment id {}",
                s.id
            )));
        }
    }

    // Root(s): segments with `parent = None`. A neuron is a single tree — one
    // root. Validate every parent reference and count roots.
    let mut root: Option<usize> = None;
    for s in segs {
        match &s.parent {
            None => {
                if root.replace(s.id).is_some() {
                    return Err(NeuromlError::parse(
                        "discretize: morphology has multiple roots (expected one soma)",
                    ));
                }
            }
            Some(p) => {
                if !index.contains_key(&p.segment) {
                    return Err(NeuromlError::parse(format!(
                        "discretize: segment {} references unknown parent {}",
                        s.id, p.segment
                    )));
                }
            }
        }
    }
    let root_id = root.ok_or_else(|| NeuromlError::parse("discretize: no root segment"))?;

    // Children map: parent segment id → child segment ids (declaration order).
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in segs {
        if let Some(p) = &s.parent {
            children.entry(p.segment).or_default().push(s.id);
        }
    }

    // Per-segment geometry (needs parent's distal for omitted proximal).
    let geom_of = |s: &Segment| -> Result<SegmentGeom, NeuromlError> {
        // `Point3d` is `Copy` — the parent's distal / the segment's own distal
        // is copied verbatim (no manual field rebuild).
        let distal = &s.distal;
        let proximal: Point3d = match &s.proximal {
            Some(p) => *p,
            None => match &s.parent {
                Some(p) => segs[index[&p.segment]].distal,
                // Root with no proximal (SWC single point): zero length → use
                // the distal point itself (effective length = diameter below).
                None => *distal,
            },
        };
        let mut length_um = dist(&proximal, distal);
        let diameter_um = (proximal.diameter + distal.diameter) / 2.0;
        if diameter_um <= 0.0 {
            return Err(NeuromlError::parse(format!(
                "discretize: segment {} has non-positive diameter",
                s.id
            )));
        }
        if length_um <= 0.0 {
            // Single-point soma → treat as a cylinder of length = diameter.
            length_um = diameter_um;
        }
        Ok(SegmentGeom {
            length_um,
            diameter_um,
            prox_x: proximal.x,
            prox_y: proximal.y,
            prox_z: proximal.z,
            distal_x: distal.x,
            distal_y: distal.y,
            distal_z: distal.z,
        })
    };

    // Breadth-first traversal from the root, splitting each segment.
    let mut compartments: Vec<Compartment> = Vec::new();
    // segment id → global index of its **last** compartment.
    let mut last_comp: HashMap<usize, usize> = HashMap::new();

    let mut queue: Vec<usize> = vec![root_id];
    let mut head = 0usize;
    while head < queue.len() {
        let seg_id = queue[head];
        head += 1;
        let seg = &segs[index[&seg_id]];
        let geom = geom_of(seg)?;

        let parent_comp = seg.parent.as_ref().map(|p| {
            // Parent segment must already be processed (BFS from root).
            last_comp[&p.segment]
        });

        let lambda = space_constant_um(geom.diameter_um, params.rm_ohm_cm2, params.ra_ohm_cm);
        let target = params.d_lambda * lambda;
        let n = ((geom.length_um / target).ceil() as usize).max(1);

        let comp_len = geom.length_um / n as f32;
        let d = geom.diameter_um;
        let area = PI * d * comp_len;
        // area(µm²) → cm²: ×1e-8.
        let area_cm2 = area * 1e-8;
        // C[nF] = cm[µF/cm²]·A[cm²]·1e3, g[µS] = A/Rm[Ω·cm²]·1e6.
        let cm_nf = params.cm_uf_per_cm2 * area_cm2 * 1e3;
        let g_leak_us = area_cm2 / params.rm_ohm_cm2 * 1e6;
        // R[MΩ] = 4·Ra[Ω·cm]·L[µm]/(π·d²[µm²]) · 1e-2.
        let r_axial_mohm = 4.0 * params.ra_ohm_cm * comp_len / (PI * d * d) * 1e-2;

        // Centre position of compartment j: `proximal + (j+0.5)/n·(distal−proximal)`.
        let axis = (
            geom.distal_x - geom.prox_x,
            geom.distal_y - geom.prox_y,
            geom.distal_z - geom.prox_z,
        );

        for j in 0..n {
            let parent = if j == 0 {
                parent_comp
            } else {
                Some(compartments.len() - 1)
            };
            let f = (j as f32 + 0.5) / n as f32;
            let (cx, cy, cz) = (
                geom.prox_x + f * axis.0,
                geom.prox_y + f * axis.1,
                geom.prox_z + f * axis.2,
            );
            // Path distance to the centre: parent centre + half parent +
            // half own length; within a chain the step is exactly comp_len.
            let path_um = match parent {
                Some(p) => {
                    let pc = &compartments[p];
                    pc.path_um + pc.length_um * 0.5 + comp_len * 0.5
                }
                None => comp_len * 0.5,
            };
            compartments.push(Compartment {
                parent,
                segment_id: seg_id,
                length_um: comp_len,
                diameter_um: d,
                area_um2: area,
                cm_nf,
                g_leak_us,
                r_axial_mohm,
                x: cx,
                y: cy,
                z: cz,
                path_um,
            });
        }
        last_comp.insert(seg_id, compartments.len() - 1);

        // Branch point: if this segment feeds more than one child, insert a
        // zero-volume node at its distal end. This keeps the parent's distal
        // half-resistance a **single shared** resistor (each child connects to
        // the branch node via its own proximal half), exactly as NEURON models
        // a `connect(child, parent(1.0))` branch point. Flattening two children
        // straight onto the parent's last compartment would duplicate the
        // parent half and over-couple the children (a real ~1% error).
        if children.get(&seg_id).is_some_and(|kids| kids.len() >= 2) {
            let bp = compartments.len();
            let parent_comp_idx = last_comp[&seg_id];
            let parent_c = &compartments[parent_comp_idx];
            compartments.push(Compartment {
                parent: Some(parent_comp_idx),
                segment_id: seg_id,
                length_um: 0.0,
                diameter_um: geom.diameter_um,
                area_um2: 0.0,
                cm_nf: 0.0,
                g_leak_us: 0.0,
                r_axial_mohm: 0.0,
                // Branch node sits at the segment's distal end.
                x: geom.distal_x,
                y: geom.distal_y,
                z: geom.distal_z,
                path_um: parent_c.path_um + parent_c.length_um * 0.5,
            });
            last_comp.insert(seg_id, bp);
        }

        if let Some(kids) = children.get(&seg_id) {
            queue.extend_from_slice(kids);
        }
    }

    // BFS starts with the root segment, so its first compartment is index 0.
    Ok(CompartmentTree {
        compartments,
        root: 0,
    })
}
