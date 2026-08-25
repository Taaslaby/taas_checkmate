//! BioDynaMo `core/exporter.h` port.
//! Export + compact growth-stream timelapse (the `.pvd` analogue: one small
//! delta per captured frame instead of a full snapshot each time).

use crate::World;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Serialize the grown network into the `taas-checkmate` `ConnectomeData`
/// input format (a BioDynaMo-style CSR connectome): row_ptr/col_indices/
/// values, per-neuron neurotransmitter type (+1 excitatory / −1 inhibitory)
/// and named exc/inh groups. Feed it to `taas_checkmate.connectome_brain`:
/// `ConnectomeData(row_ptr, col_indices, values, nt_types, groups)` →
/// `BrainRunner` (LIF/Izhikevich/HH) → spikes.
pub fn connectome_json(world: &World) -> String {
    let g = world.to_graph();
    let n = g.nodes.len();

    // CSR row layout: edges grouped by pre-neuron, stable order by (pre, post).
    let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n];
    for e in &g.edges {
        per[e.pre].push((e.post, e.weight as f32));
    }
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::with_capacity(g.edges.len());
    let mut values = Vec::with_capacity(g.edges.len());
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = col_indices.len() as i32;
        for &(post, w) in list {
            col_indices.push(post as i32);
            values.push(w);
        }
    }
    row_ptr[n] = col_indices.len() as i32;

    // tanda-checkmate sign convention: nt_types[pre] = source NT
    // (+1 excitatory, −1 inhibitory), values stay positive.
    let nt_types: Vec<f32> = g
        .nodes
        .iter()
        .map(|nd| match nd.cell_type {
            crate::CellType::Inhibitory => -1.0,
            _ => 1.0,
        })
        .collect();

    let mut exc = Vec::new();
    let mut inh = Vec::new();
    for nd in &g.nodes {
        match nd.cell_type {
            crate::CellType::Inhibitory => inh.push(nd.id),
            crate::CellType::Excitatory => exc.push(nd.id),
            crate::CellType::Glia => {}
        }
    }
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    if !exc.is_empty() {
        groups.push(("exc".to_string(), exc));
    }
    if !inh.is_empty() {
        groups.push(("inh".to_string(), inh));
    }

    serde_json::json!({
        "n_neurons": n,
        "n_synapses": g.edges.len(),
        "row_ptr": row_ptr,
        "col_indices": col_indices,
        "values": values,
        "nt_types": nt_types,
        "groups": groups,
        "neuron_pos": g
            .nodes
            .iter()
            .map(|nd| [nd.pos.x, nd.pos.y, nd.pos.z])
            .collect::<Vec<_>>(),
    })
    .to_string()
}

pub fn export_vtk(world: &World, path: &str) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "# vtk DataFile Version 3.0")?;
    writeln!(f, "PROMETHEUS grown network")?;
    writeln!(f, "ASCII")?;
    writeln!(f, "DATASET POLYDATA")?;
    writeln!(f, "POINTS {} float", world.cells.len())?;
    for c in &world.cells {
        writeln!(f, "{} {} {}", c.soma.pos.x, c.soma.pos.y, c.soma.pos.z)?;
    }
    writeln!(
        f,
        "LINES {} {}\n",
        world.synapses.len(),
        world.synapses.len() * 3
    )?;
    for s in &world.synapses {
        writeln!(f, "2 {} {}", s.pre, s.post)?;
    }
    Ok(())
}

// ─── shared JSON helpers ────────────────────────────────────────────────

/// Compact float formatting: integers without decimals, floats trimmed.
fn fmt_num(x: f64) -> String {
    if x == x.round() && x.abs() < 1e15 {
        format!("{:.0}", x)
    } else {
        format!("{:.4}", x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn vec3_str(p: &crate::Vec3) -> String {
    format!("[{},{},{}]", fmt_num(p.x), fmt_num(p.y), fmt_num(p.z))
}

fn cell_type_str(c: crate::CellType) -> &'static str {
    match c {
        crate::CellType::Excitatory => "Excitatory",
        crate::CellType::Inhibitory => "Inhibitory",
        crate::CellType::Glia => "Glia",
    }
}

fn synapse_kind_str(k: crate::SynapseKind) -> &'static str {
    match k {
        crate::SynapseKind::Excitatory => "Excitatory",
        crate::SynapseKind::Inhibitory => "Inhibitory",
    }
}

fn excrescence_kind_str(k: crate::ExcrescenceKind) -> &'static str {
    match k {
        crate::ExcrescenceKind::Bouton => "Bouton",
        crate::ExcrescenceKind::Spine => "Spine",
    }
}

fn points_json(points: &[crate::Vec3]) -> String {
    let mut s = String::from("[");
    for (j, p) in points.iter().enumerate() {
        if j > 0 {
            s.push(',');
        }
        s.push_str(&vec3_str(p));
    }
    s.push(']');
    s
}

/// Serialize the cells (soma spheres) as a JSON array.
pub fn cells_json(world: &World) -> String {
    let mut out = String::from("[");
    for (i, c) in world.cells.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"id\":{},\"type\":\"{}\",\"tag\":\"{}\",\"pos\":{},\"radius\":{}}}",
            c.id,
            cell_type_str(c.cell_type),
            c.tag,
            vec3_str(&c.soma.pos),
            fmt_num(c.soma.radius)
        ));
    }
    out.push(']');
    out
}

/// Export the full grown scene as JSON for the WebGL visualizer:
/// somas (with type/tag), neurite polylines (with radius, branch order,
/// axon/dendrite, terminal flag), synaptic excrescences and synapses.
///
/// ```json
/// {"cells":[{...}],"neurites":[{...}],"excrescences":[{...}],"synapses":[{...}]}
/// ```
pub fn export_scene_json(world: &World, path: &str) -> std::io::Result<()> {
    std::fs::write(path, scene_json(world))
}

/// Serialize the grown scene to a JSON string (see [`export_scene_json`]
/// for the schema).
pub fn scene_json(world: &World) -> String {
    format!(
        "{{\"cells\":{},\"neurites\":{},\"excrescences\":{},\"synapses\":{}}}",
        cells_json(world),
        neurites_json(world),
        excrescences_json(world),
        synapses_json(world)
    )
}

fn neurites_json(world: &World) -> String {
    let mut out = String::from("[");
    let mut first = true;
    for (ci, c) in world.cells.iter().enumerate() {
        for n in &c.neurites {
            if !first {
                out.push(',');
            }
            first = false;
            let parent = match n.parent {
                Some(p) => p.to_string(),
                None => "null".to_string(),
            };
            out.push_str(&format!(
                "{{\"cell\":{},\"axon\":{},\"order\":{},\"parent\":{},\"terminal\":{},\"diameter\":{},\"points\":{}}}",
                ci,
                n.is_axon,
                n.branch_order,
                parent,
                n.is_terminal,
                fmt_num(n.diameter),
                points_json(&n.points)
            ));
        }
    }
    out.push(']');
    out
}

fn excrescences_json(world: &World) -> String {
    let mut out = String::from("[");
    let mut first = true;
    for (ci, c) in world.cells.iter().enumerate() {
        for n in &c.neurites {
            for e in &n.excrescences {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&format!(
                    "{{\"cell\":{},\"kind\":\"{}\",\"pos\":{}}}",
                    ci,
                    excrescence_kind_str(e.kind),
                    vec3_str(&e.pos)
                ));
            }
        }
    }
    out.push(']');
    out
}

fn synapses_json(world: &World) -> String {
    let mut out = String::from("[");
    for (i, s) in world.synapses.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"pre\":{},\"post\":{},\"kind\":\"{}\",\"weight\":{}}}",
            s.pre,
            s.post,
            synapse_kind_str(s.kind),
            fmt_num(s.weight)
        ));
    }
    out.push(']');
    out
}

// ─── growth-stream timelapse (the BioDynaMo `.pvd` analogue) ────────────

/// Captures a *compact growth stream* for the timelapse viewer: cells are
/// serialized once, then every [`Timelapse::capture`] records only the
/// *deltas* since the previous capture — newly born neurites, tip points
/// appended to existing neurites, new excrescences and new synapses.
///
/// This mirrors BioDynaMo's per-iteration export, but instead of a full
/// snapshot per iteration it ships just the growth events, so a 100-frame
/// movie costs about as much as one full scene.
pub struct Timelapse {
    prev_neurites_per_cell: Vec<usize>,
    prev_points: HashMap<(usize, usize), usize>,
    prev_exc: HashMap<(usize, usize), usize>,
    prev_syn: usize,
    frames: Vec<String>,
}

impl Timelapse {
    /// Start a stream. The first [`Timelapse::capture`] emits *everything that
    /// exists at that moment* as frame 0 (somas are serialized separately; here
    /// it is the initial neurites/excrescences/synapses); every later capture
    /// ships only the deltas since the previous one.
    pub fn new(world: &World) -> Self {
        let prev_neurites_per_cell = vec![0; world.cells.len()];
        Timelapse {
            prev_neurites_per_cell,
            prev_points: HashMap::new(),
            prev_exc: HashMap::new(),
            prev_syn: 0,
            frames: vec![],
        }
    }

    /// Record the current state as the next frame. Returns the frame JSON
    /// (also appended to the internal list).
    pub fn capture(&mut self, world: &World) -> String {
        let mut add = String::from("[");
        let mut grow = String::from("[");
        let mut exc = String::from("[");
        let mut first_add = true;
        let mut first_grow = true;
        let mut first_exc = true;

        for (ci, c) in world.cells.iter().enumerate() {
            let born_from = self.prev_neurites_per_cell[ci];
            for (ni, n) in c.neurites.iter().enumerate() {
                if ni >= born_from {
                    // Newly born neurite: ship it whole.
                    if !first_add {
                        add.push(',');
                    }
                    first_add = false;
                    let parent = match n.parent {
                        Some(p) => p.to_string(),
                        None => "null".to_string(),
                    };
                    add.push_str(&format!(
                        "{{\"cell\":{},\"idx\":{},\"axon\":{},\"order\":{},\"parent\":{},\"terminal\":{},\"diameter\":{},\"points\":{}}}",
                        ci,
                        ni,
                        n.is_axon,
                        n.branch_order,
                        parent,
                        n.is_terminal,
                        fmt_num(n.diameter),
                        points_json(&n.points)
                    ));
                } else if n.points.len() > self.prev_points[&(ci, ni)] {
                    // Existing neurite: only the newly appended tip points.
                    if !first_grow {
                        grow.push(',');
                    }
                    first_grow = false;
                    let prev = self.prev_points[&(ci, ni)];
                    grow.push_str(&format!(
                        "{{\"cell\":{},\"idx\":{},\"points\":{}}}",
                        ci,
                        ni,
                        points_json(&n.points[prev..])
                    ));
                }
                // Newly born neurites have no bookkeeping yet — start at 0.
                let prev_exc_n = if ni >= born_from {
                    0
                } else {
                    self.prev_exc[&(ci, ni)]
                };
                for e in n.excrescences.iter().skip(prev_exc_n) {
                    if !first_exc {
                        exc.push(',');
                    }
                    first_exc = false;
                    exc.push_str(&format!(
                        "{{\"cell\":{},\"kind\":\"{}\",\"pos\":{}}}",
                        ci,
                        excrescence_kind_str(e.kind),
                        vec3_str(&e.pos)
                    ));
                }
            }
        }
        add.push(']');
        grow.push(']');
        exc.push(']');

        // New synapses (flat list, never removed).
        let mut syn = String::from("[");
        for (i, s) in world.synapses.iter().enumerate().skip(self.prev_syn) {
            if i > self.prev_syn {
                syn.push(',');
            }
            syn.push_str(&format!(
                "{{\"pre\":{},\"post\":{},\"kind\":\"{}\",\"weight\":{}}}",
                s.pre,
                s.post,
                synapse_kind_str(s.kind),
                fmt_num(s.weight)
            ));
        }
        syn.push(']');

        // Bookkeeping.
        self.prev_neurites_per_cell = world.cells.iter().map(|c| c.neurites.len()).collect();
        self.prev_points.clear();
        self.prev_exc.clear();
        for (ci, c) in world.cells.iter().enumerate() {
            for (ni, n) in c.neurites.iter().enumerate() {
                self.prev_points.insert((ci, ni), n.points.len());
                self.prev_exc.insert((ci, ni), n.excrescences.len());
            }
        }
        self.prev_syn = world.synapses.len();

        let frame = format!(
            "{{\"add_neurites\":{},\"grow\":{},\"excrescences\":{},\"synapses\":{}}}",
            add, grow, exc, syn
        );
        self.frames.push(frame.clone());
        frame
    }

    /// Build the full timelapse document: cells once + all captured frames.
    pub fn finalize(&self, world: &World) -> String {
        format!(
            "{{\"cells\":{},\"frames\":[{}]}}",
            cells_json(world),
            self.frames.join(",")
        )
    }
}
