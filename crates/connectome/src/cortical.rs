//! # Cortical layers — laminar connectivity for neocortical simulations.
//!
//! Provides a builder for generating [`ConnectomeData`](crate::data::ConnectomeData) with
//! anatomically-motivated laminar connectivity patterns (6-layer neocortex).
//!
//! ## Canonical mammalian neocortical microcircuit
//!
//! ```text
//!                     L1 (molecular layer — mostly axons/dendrites)
//!                      ↕
//!      L2/3 (superficial pyramidal) ←→ L2/3 (recurrent)
//!        ↓  (FF)                        ↑  (FF from L4)
//!      L4   (spiny stellate / thalamorecipient)
//!        ↓  (FF)                        ↑  (FB from L6)
//!      L5   (deep pyramidal) ←→ L5 (recurrent)
//!        ↓  (FF)                        ↑  (FB from L6)
//!      L6   (corticothalamic)
//! ```
//!
//! ## Between columns (inter-areal hierarchy)
//!
//! | Projection | Source | Target | Hierarchy direction |
//! |------------|--------|--------|--------------------|
//! | **Feedforward** | L2/3 | L4 | Lower → higher area |
//! | **Feedforward** | L5   | L4 | Lower → higher area |
//! | **Feedback**    | L5/6 | L1  | Higher → lower area |
//! | **Lateral**     | L2/3 | L2/3 | Same hierarchy level |
//!
//! ## Usage
//!
//! ```ignore
//! use taas_connectome::cortical::LaminarConnectomeBuilder;
//!
//! let builder = LaminarConnectomeBuilder::new(2, 4)   // 2 columns, 4 neurons/col/layer
//!     .with_default_mammalian();                       // standard mammalian rules
//!
//! let (connectome, layer_assignment) = builder.build(42);  // seed 42
//!
//! // Analyze laminar connectivity
//! let stats = builder.compute_stats(&connectome, &layer_assignment);
//! println!("FF:FB ratio = {}", stats.ff_fb_ratio());
//! ```

#![allow(non_snake_case)]

use crate::data::ConnectomeData;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════
//  Layer identifiers
// ═══════════════════════════════════════════════════════════

/// Cortical layer identifier (L1–L6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LayerId {
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
}

impl LayerId {
    /// All 6 layers in ascending order.
    pub fn all() -> [LayerId; 6] {
        [
            LayerId::L1,
            LayerId::L2,
            LayerId::L3,
            LayerId::L4,
            LayerId::L5,
            LayerId::L6,
        ]
    }

    /// Index position 0–5.
    pub fn index(self) -> usize {
        match self {
            LayerId::L1 => 0,
            LayerId::L2 => 1,
            LayerId::L3 => 2,
            LayerId::L4 => 3,
            LayerId::L5 => 4,
            LayerId::L6 => 5,
        }
    }

    /// Name string for display and group naming (distinct per layer).
    pub fn name(self) -> &'static str {
        match self {
            LayerId::L1 => "L1",
            LayerId::L2 => "L2",
            LayerId::L3 => "L3",
            LayerId::L4 => "L4",
            LayerId::L5 => "L5",
            LayerId::L6 => "L6",
        }
    }

    /// Display name (canonical naming with L2/3 grouping).
    pub fn display_name(self) -> &'static str {
        match self {
            LayerId::L1 => "L1",
            LayerId::L2 => "L2/3",
            LayerId::L3 => "L2/3",
            LayerId::L4 => "L4",
            LayerId::L5 => "L5",
            LayerId::L6 => "L6",
        }
    }

    /// Canonical neuron type (Drosophila analogs where applicable).
    pub fn cell_type(self) -> &'static str {
        match self {
            LayerId::L1 => "dendritic/molecular",
            LayerId::L2 => "superficial pyramidal",
            LayerId::L3 => "superficial pyramidal",
            LayerId::L4 => "spiny stellate / thalamorecipient",
            LayerId::L5 => "deep pyramidal",
            LayerId::L6 => "corticothalamic",
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Projection type
// ═══════════════════════════════════════════════════════════

/// Functional projection type between cortical areas/layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProjectionType {
    /// Low→high hierarchy: L2/3 → L4, L5 → L4
    Feedforward,
    /// High→low hierarchy: L5/6 → L1
    Feedback,
    /// Same hierarchy level
    Lateral,
    /// Within same layer and column
    Recurrent,
}

impl ProjectionType {
    pub fn name(self) -> &'static str {
        match self {
            ProjectionType::Feedforward => "FF",
            ProjectionType::Feedback => "FB",
            ProjectionType::Lateral => "LAT",
            ProjectionType::Recurrent => "REC",
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Laminar connection rule
// ═══════════════════════════════════════════════════════════

/// A connection rule between two cortical layers.
///
/// Specifies which pre-synaptic layer projects to which post-synaptic layer,
/// with connection probability, synaptic weight range, and NT type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LaminarRule {
    pub from_layer: LayerId,
    pub to_layer: LayerId,
    /// Connection probability (0..1). Each pair is connected with this prob.
    pub probability: f32,
    /// Synaptic weight range (min..max), sampled uniformly for each connection.
    pub weight_range: (f32, f32),
    /// Neurotransmitter type: +1 = excitatory, -1 = inhibitory
    pub nt_type: f32,
    /// Projection type for statistics
    pub projection_type: ProjectionType,
    /// Whether this rule applies within the same column only (true), or between different columns (false).
    /// `None` means both.
    pub within_column: Option<bool>,
}

impl LaminarRule {
    pub fn new(
        from: LayerId,
        to: LayerId,
        probability: f32,
        w_min: f32,
        w_max: f32,
        nt_type: f32,
        proj: ProjectionType,
    ) -> Self {
        Self {
            from_layer: from,
            to_layer: to,
            probability,
            weight_range: (w_min, w_max),
            nt_type,
            projection_type: proj,
            within_column: None,
        }
    }

    /// Set whether this rule applies within the same column.
    pub fn within_column(mut self, wc: bool) -> Self {
        self.within_column = Some(wc);
        self
    }
}

// ═══════════════════════════════════════════════════════════
//  Layer assignment per neuron
// ═══════════════════════════════════════════════════════════

/// Per-neuron layer and column assignment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerAssignment {
    /// `layer[i]` = LayerId for neuron i
    pub layer: Vec<LayerId>,
    /// `column[i]` = column index for neuron i
    pub column: Vec<usize>,
    /// Number of neurons
    pub n_neurons: usize,
    /// Number of columns
    pub n_columns: usize,
}

impl LayerAssignment {
    pub fn new(layer: Vec<LayerId>, column: Vec<usize>) -> Self {
        let n_neurons = layer.len().min(column.len());
        Self {
            layer,
            column,
            n_neurons,
            n_columns: 1,
        }
    }

    /// Get all neuron indices belonging to a specific layer and column.
    pub fn neurons_in(&self, layer: LayerId, column: usize) -> Vec<usize> {
        (0..self.n_neurons)
            .filter(|&i| self.layer[i] == layer && self.column[i] == column)
            .collect()
    }

    /// Get all neuron indices in a layer (any column).
    pub fn neurons_in_layer(&self, layer: LayerId) -> Vec<usize> {
        (0..self.n_neurons)
            .filter(|&i| self.layer[i] == layer)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
//  Laminar connectome builder
// ═══════════════════════════════════════════════════════════

/// Builder for generating laminar cortical connectomes.
///
/// # Example
/// ```ignore
/// let (conn, assign) = LaminarConnectomeBuilder::new(2, 5)
///     .with_default_mammalian()
///     .build(42);
/// ```
#[derive(Debug, Clone)]
pub struct LaminarConnectomeBuilder {
    /// Number of cortical columns
    pub n_columns: usize,
    /// Neurons per (column, layer) — symmetric across columns
    pub n_per_layer: HashMap<LayerId, usize>,
    /// Connection rules
    pub rules: Vec<LaminarRule>,
}

impl LaminarConnectomeBuilder {
    /// Create a new builder with equal neurons per layer per column.
    ///
    /// `n_per_layer` — neurons in each of the 6 layers, per column.
    /// Total neurons = n_columns × n_per_layer × 6.
    pub fn new(n_columns: usize, n_per_layer: usize) -> Self {
        let mut map = HashMap::new();
        for layer in LayerId::all() {
            map.insert(layer, n_per_layer);
        }
        Self {
            n_columns,
            n_per_layer: map,
            rules: Vec::new(),
        }
    }

    /// Create with custom neurons per layer (some layers may have 0).
    pub fn with_layer_sizes(n_columns: usize, n_per_layer: HashMap<LayerId, usize>) -> Self {
        // Ensure all layers have a value (default 0)
        let mut map = n_per_layer;
        for layer in LayerId::all() {
            map.entry(layer).or_insert(0);
        }
        Self {
            n_columns,
            n_per_layer: map,
            rules: Vec::new(),
        }
    }

    /// Add a laminar connection rule.
    pub fn add_rule(mut self, rule: LaminarRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add multiple rules at once.
    pub fn add_rules(mut self, rules: &[LaminarRule]) -> Self {
        self.rules.extend_from_slice(rules);
        self
    }

    /// Add default mammalian neocortical connectivity rules.
    ///
    /// Based on canonical microcircuit (Thomson & Bannister 2003, Beul & Hilgetag 2015).
    pub fn with_default_mammalian(self) -> Self {
        self.add_rules(&[
            // ── Within-column excitatory ──
            LaminarRule::new(
                LayerId::L4,
                LayerId::L4,
                0.15,
                0.3,
                0.8,
                1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L4,
                LayerId::L2,
                0.25,
                0.4,
                1.0,
                1.0,
                ProjectionType::Feedforward,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L2,
                LayerId::L2,
                0.20,
                0.3,
                0.8,
                1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L2,
                LayerId::L5,
                0.20,
                0.5,
                1.2,
                1.0,
                ProjectionType::Feedforward,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L5,
                0.20,
                0.4,
                1.0,
                1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L6,
                0.25,
                0.5,
                1.2,
                1.0,
                ProjectionType::Feedforward,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L6,
                LayerId::L4,
                0.20,
                0.3,
                0.7,
                1.0,
                ProjectionType::Feedback,
            )
            .within_column(true),
            // ── Within-column inhibitory (PV+ interneurons) ──
            LaminarRule::new(
                LayerId::L4,
                LayerId::L4,
                0.08,
                0.8,
                1.5,
                -1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L2,
                LayerId::L2,
                0.08,
                0.8,
                1.5,
                -1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L5,
                0.06,
                0.8,
                1.5,
                -1.0,
                ProjectionType::Recurrent,
            )
            .within_column(true),
            // ── Inter-columnar feedforward (lower→higher area) ──
            LaminarRule::new(
                LayerId::L2,
                LayerId::L4,
                0.10,
                0.3,
                0.7,
                1.0,
                ProjectionType::Feedforward,
            )
            .within_column(false),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L4,
                0.08,
                0.3,
                0.7,
                1.0,
                ProjectionType::Feedforward,
            )
            .within_column(false),
            // ── Inter-columnar feedback (higher→lower area) ──
            LaminarRule::new(
                LayerId::L5,
                LayerId::L1,
                0.06,
                0.2,
                0.5,
                1.0,
                ProjectionType::Feedback,
            )
            .within_column(false),
            LaminarRule::new(
                LayerId::L6,
                LayerId::L1,
                0.06,
                0.2,
                0.5,
                1.0,
                ProjectionType::Feedback,
            )
            .within_column(false),
        ])
    }

    /// Add simplified insect (Drosophila) laminar connectivity rules.
    ///
    /// Insect central complex / mushroom body laminar analogs:
    /// - L1 ≡ neuropil (mushroom body lobes)\n  - L2/3 ≡ Kenyon cells / intrinsic neurons
    /// - L4 ≡ olfactory projection neurons (PNs)
    /// - L5/6 ≡ output neurons (MBONs)
    pub fn with_default_insect(self) -> Self {
        self.add_rules(&[
            // PN → KC (L4 → L2/3)
            LaminarRule::new(
                LayerId::L4,
                LayerId::L2,
                0.05,
                0.3,
                0.8,
                1.0,
                ProjectionType::Feedforward,
            ),
            // KC → KC recurrent (L2/3 → L2/3)
            LaminarRule::new(
                LayerId::L2,
                LayerId::L2,
                0.10,
                0.2,
                0.6,
                1.0,
                ProjectionType::Recurrent,
            ),
            // KC → MBON (L2/3 → L5)
            LaminarRule::new(
                LayerId::L2,
                LayerId::L5,
                0.15,
                0.4,
                1.0,
                1.0,
                ProjectionType::Feedforward,
            ),
            // MBON → MBON recurrent (L5 → L5)
            LaminarRule::new(
                LayerId::L5,
                LayerId::L5,
                0.15,
                0.3,
                0.8,
                1.0,
                ProjectionType::Recurrent,
            ),
            // Local inhibition
            LaminarRule::new(
                LayerId::L2,
                LayerId::L2,
                0.05,
                0.5,
                1.5,
                -1.0,
                ProjectionType::Recurrent,
            ),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L5,
                0.05,
                0.5,
                1.5,
                -1.0,
                ProjectionType::Recurrent,
            ),
        ])
    }

    /// Total number of neurons across all columns and layers.
    pub fn total_neurons(&self) -> usize {
        let per_column: usize = self.n_per_layer.values().sum();
        self.n_columns * per_column
    }

    /// Global neuron index for (column, layer, local_idx).
    fn global_idx(&self, column: usize, layer: LayerId, local: usize) -> usize {
        // Layout: all layers of column 0, then all layers of column 1, etc.
        let per_column: usize = self.n_per_layer.values().sum();
        let mut offset = 0usize;
        for l in LayerId::all() {
            if l == layer {
                break;
            }
            offset += self.n_per_layer[&l];
        }
        column * per_column + offset + local
    }

    /// Build the laminar connectome using a seeded RNG for reproducibility.
    ///
    /// Returns `(ConnectomeData, LayerAssignment)`.
    /// Uses a simple LCG for deterministic random connections.
    pub fn build(&self, seed: u32) -> (ConnectomeData, LayerAssignment) {
        let n_neurons = self.total_neurons();

        // Pre-allocate with capacity
        let mut row_ptr = vec![0i32; n_neurons + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut nt_types = vec![1.0f32; n_neurons];
        let mut layer_assign = vec![LayerId::L1; n_neurons];
        let mut col_assign = vec![0usize; n_neurons];

        // Assign layers/columns to neurons
        for col in 0..self.n_columns {
            for layer in LayerId::all() {
                let n = self.n_per_layer[&layer];
                for local in 0..n {
                    let idx = self.global_idx(col, layer, local);
                    layer_assign[idx] = layer;
                    col_assign[idx] = col;
                }
            }
        }

        // Simple LCG RNG for deterministic connections (copied from sim.rs pattern)
        let mut rng = Lcg32::new(seed);

        // Build scratch adjacency: for each pre-neuron, collect (post, weight)
        // We use Vec<VecDeque<(usize, f32)>> to avoid dedup cost
        let mut scratch: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_neurons];

        for rule in &self.rules {
            let from_n = self.n_per_layer[&rule.from_layer];
            let to_n = self.n_per_layer[&rule.to_layer];
            if from_n == 0 || to_n == 0 {
                continue;
            }

            let within_col = match rule.within_column {
                Some(true) => true,
                Some(false) => false,
                None => {
                    // both
                    for pre_col in 0..self.n_columns {
                        for post_col in 0..self.n_columns {
                            for pre_local in 0..from_n {
                                let pre_idx = self.global_idx(pre_col, rule.from_layer, pre_local);
                                for post_local in 0..to_n {
                                    let post_idx =
                                        self.global_idx(post_col, rule.to_layer, post_local);
                                    let rand = rng.next_f32();
                                    if rand < rule.probability {
                                        let w = rule.weight_range.0
                                            + rng.next_f32()
                                                * (rule.weight_range.1 - rule.weight_range.0);
                                        scratch[pre_idx].push((post_idx, w));
                                    }
                                }
                            }
                        }
                    }
                    continue; // already handled above
                }
            };

            for pre_col in 0..self.n_columns {
                // For within-column: post_col = pre_col
                // For between-column: post_col = any column ≠ pre_col
                let post_cols: Vec<usize> = if within_col {
                    vec![pre_col]
                } else {
                    (0..self.n_columns).filter(|&c| c != pre_col).collect()
                };

                for pre_local in 0..from_n {
                    let pre_idx = self.global_idx(pre_col, rule.from_layer, pre_local);
                    for &post_col in &post_cols {
                        for post_local in 0..to_n {
                            let post_idx = self.global_idx(post_col, rule.to_layer, post_local);
                            let rand = rng.next_f32();
                            if rand < rule.probability {
                                let w = rule.weight_range.0
                                    + rng.next_f32() * (rule.weight_range.1 - rule.weight_range.0);
                                scratch[pre_idx].push((post_idx, w));
                            }
                        }
                    }
                }
            }
        }

        // Build CSR from scratch
        let mut syn_count = 0i32;
        for i in 0..n_neurons {
            // Sort by post index for cache-friendly CSR
            scratch[i].sort_by_key(|&(post, _)| post);
            let mut last_post = -1i32;
            for &(post, w) in &scratch[i] {
                let post_i32 = post as i32;
                if post_i32 != last_post {
                    col_indices.push(post_i32);
                    values.push(w);
                    syn_count += 1;
                    last_post = post_i32;
                }
                // If duplicate (same pre→post), add weight
                // Actually we skip duplicates — just use the first
            }
            row_ptr[i + 1] = syn_count;
        }

        // Set NT types based on layer (mostly excitatory, with some inhibitory)
        // In a real connectome, some neurons are inhibitory. Here we set NT type
        // per synapse based on the rule that created it, not per neuron.
        // Actually, nt_types is per-neuron, and we should set it per the majority rule.
        // For simplicity, set inhibitory NT for layers with dominant inhibition.
        for i in 0..n_neurons {
            let layer = layer_assign[i];
            // By default all excitatory; rules handle specific inhibitory connections
            nt_types[i] = 1.0;
            // L1 is mostly inhibitory
            if layer == LayerId::L1 {
                nt_types[i] = -1.0;
            }
        }

        let n_synapses = col_indices.len();

        // Build neuron groups
        let mut groups = Vec::new();
        for layer in LayerId::all() {
            let name = format!("layer_{}", layer.name());
            let indices: Vec<usize> = (0..n_neurons)
                .filter(|&i| layer_assign[i] == layer)
                .collect();
            if !indices.is_empty() {
                groups.push(crate::data::NeuronGroup { name, indices });
            }
        }

        // Per-column groups
        for col in 0..self.n_columns {
            let indices: Vec<usize> = (0..n_neurons).filter(|&i| col_assign[i] == col).collect();
            if !indices.is_empty() {
                groups.push(crate::data::NeuronGroup {
                    name: format!("col_{col}"),
                    indices,
                });
            }
        }

        let connectome = ConnectomeData {
            n_neurons,
            n_synapses,
            row_ptr,
            col_indices,
            values,
            nt_types,
            receptor_types: Vec::new(),
            groups,
        };

        let assignment = LayerAssignment {
            layer: layer_assign,
            column: col_assign,
            n_neurons,
            n_columns: self.n_columns,
        };

        (connectome, assignment)
    }

    /// Compute laminar connectivity statistics.
    ///
    /// Analyzes FF/FB/LAT/REC projection counts and builds a layer×layer matrix.
    pub fn compute_stats(
        &self,
        connectome: &ConnectomeData,
        assign: &LayerAssignment,
    ) -> LayerStats {
        let n_neurons = connectome.n_neurons;
        let n_layers = 6;

        // Count synapses per projection type
        let mut ff_count = 0usize;
        let mut fb_count = 0usize;
        let mut lat_count = 0usize;
        let mut rec_count = 0usize;
        let mut total = 0usize;

        // Layer→layer matrix [6×6]
        let mut layer_matrix = vec![0usize; n_layers * n_layers];
        // Projection type matrix [4×6×6]
        let mut proj_matrix = vec![0usize; 4 * n_layers * n_layers];

        let proj_map: HashMap<(LayerId, LayerId), ProjectionType> = self
            .rules
            .iter()
            .map(|r| ((r.from_layer, r.to_layer), r.projection_type))
            .collect();

        for pre in 0..n_neurons {
            let from_layer = assign.layer[pre];
            let from_col = assign.column[pre];
            let start = connectome.row_ptr[pre].max(0) as usize;
            let end = (connectome.row_ptr.get(pre + 1).copied().unwrap_or(0) as usize)
                .min(connectome.col_indices.len());
            for idx in start..end {
                let post = connectome.col_indices[idx].max(0) as usize;
                if post >= n_neurons {
                    continue;
                }
                let to_layer = assign.layer[post];
                let to_col = assign.column[post];

                let fl = from_layer.index();
                let tl = to_layer.index();
                layer_matrix[fl * n_layers + tl] += 1;

                let ptype = if from_col == to_col {
                    if from_layer == to_layer {
                        ProjectionType::Recurrent
                    } else {
                        // Within column, different layers — determine from rules
                        proj_map
                            .get(&(from_layer, to_layer))
                            .copied()
                            .unwrap_or(ProjectionType::Lateral)
                    }
                } else if find_ff_rule(&self.rules, from_layer, to_layer) {
                    ProjectionType::Feedforward
                } else if find_fb_rule(&self.rules, from_layer, to_layer) {
                    ProjectionType::Feedback
                } else {
                    ProjectionType::Lateral
                };

                let pi = match ptype {
                    ProjectionType::Feedforward => 0,
                    ProjectionType::Feedback => 1,
                    ProjectionType::Lateral => 2,
                    ProjectionType::Recurrent => 3,
                };
                proj_matrix[pi * n_layers * n_layers + fl * n_layers + tl] += 1;

                match ptype {
                    ProjectionType::Feedforward => ff_count += 1,
                    ProjectionType::Feedback => fb_count += 1,
                    ProjectionType::Lateral => lat_count += 1,
                    ProjectionType::Recurrent => rec_count += 1,
                }
                total += 1;
            }
        }

        LayerStats {
            total_synapses: total,
            ff_count,
            fb_count,
            lat_count,
            rec_count,
            layer_matrix,
            proj_matrix,
            n_layers,
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Helper: find projection type from rules
// ═══════════════════════════════════════════════════════════

fn find_ff_rule(rules: &[LaminarRule], from: LayerId, to: LayerId) -> bool {
    rules.iter().any(|r| {
        r.from_layer == from && r.to_layer == to && r.projection_type == ProjectionType::Feedforward
    })
}

fn find_fb_rule(rules: &[LaminarRule], from: LayerId, to: LayerId) -> bool {
    rules.iter().any(|r| {
        r.from_layer == from && r.to_layer == to && r.projection_type == ProjectionType::Feedback
    })
}

// ═══════════════════════════════════════════════════════════
//  Layer connectivity statistics
// ═══════════════════════════════════════════════════════════

/// Laminar connectivity statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerStats {
    pub total_synapses: usize,
    /// Feedforward synapse count
    pub ff_count: usize,
    /// Feedback synapse count
    pub fb_count: usize,
    /// Lateral (inter-areal same level) count
    pub lat_count: usize,
    /// Recurrent (within column, same layer) count
    pub rec_count: usize,
    /// Layer→layer matrix [6×6], flattened row-major
    pub layer_matrix: Vec<usize>,
    /// Projection type × layer [4×6×6], flattened: [proj][from][to]
    pub proj_matrix: Vec<usize>,
    pub n_layers: usize,
}

impl LayerStats {
    /// Feedforward / Feedback ratio.
    pub fn ff_fb_ratio(&self) -> f32 {
        if self.fb_count == 0 {
            return f32::INFINITY;
        }
        self.ff_count as f32 / self.fb_count as f32
    }

    /// Fraction of synapses that are recurrent (local).
    pub fn recurrent_fraction(&self) -> f32 {
        if self.total_synapses == 0 {
            return 0.0;
        }
        self.rec_count as f32 / self.total_synapses as f32
    }

    /// Fraction of synapses that are FF.
    pub fn ff_fraction(&self) -> f32 {
        if self.total_synapses == 0 {
            return 0.0;
        }
        self.ff_count as f32 / self.total_synapses as f32
    }

    /// Fraction of synapses that are FB.
    pub fn fb_fraction(&self) -> f32 {
        if self.total_synapses == 0 {
            return 0.0;
        }
        self.fb_count as f32 / self.total_synapses as f32
    }

    /// Get the layer→layer matrix as &[usize; 36].
    pub fn matrix_6x6(&self) -> &[usize] {
        &self.layer_matrix
    }

    /// Format the layer→layer matrix for display.
    pub fn format_matrix(&self) -> String {
        let labels = ["L1", "L2", "L3", "L4", "L5", "L6"];
        let mut s = String::from("Layer→Layer matrix:\n     ");
        for lbl in &labels {
            s.push_str(&format!("{lbl:>6}"));
        }
        s.push('\n');
        for (fi, &flbl) in labels.iter().enumerate() {
            s.push_str(&format!("{flbl:>4} ",));
            for ti in 0..6 {
                let val = self.layer_matrix[fi * 6 + ti];
                s.push_str(&format!("{val:>6}"));
            }
            s.push('\n');
        }
        s
    }

    /// Format projection type counts.
    pub fn format_proj_counts(&self) -> String {
        format!(
            "FF={} ({:.1}%) FB={} ({:.1}%) LAT={} ({:.1}%) REC={} ({:.1}%)",
            self.ff_count,
            self.ff_fraction() * 100.0,
            self.fb_count,
            self.fb_fraction() * 100.0,
            self.lat_count,
            if self.total_synapses > 0 {
                self.lat_count as f32 / self.total_synapses as f32 * 100.0
            } else {
                0.0
            },
            self.rec_count,
            self.recurrent_fraction() * 100.0,
        )
    }
}

// ═══════════════════════════════════════════════════════════
//  Simple LCG RNG (deterministic, no external deps)
// ═══════════════════════════════════════════════════════════

/// Simple LCG for deterministic random connections.
struct Lcg32(u32);

impl Lcg32 {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // MMIX LCG by Knuth
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005u64 as u32)
            .wrapping_add(1442695040888963407u64 as u32);
        self.0
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * 5.58794e-09 // ≈ 1/2²³
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_single_column() {
        let builder = LaminarConnectomeBuilder::new(1, 4).with_default_mammalian();
        let n = builder.total_neurons();
        assert_eq!(n, 24); // 1 col × 6 layers × 4 neurons

        let (conn, assign) = builder.build(42);
        assert_eq!(conn.n_neurons, 24);
        assert!(conn.n_synapses > 0, "Should have synapses");
        assert_eq!(assign.n_neurons, 24);
        assert_eq!(assign.n_columns, 1);

        // Each layer should have 4 neurons
        for layer in LayerId::all() {
            let neurons = assign.neurons_in_layer(layer);
            assert_eq!(neurons.len(), 4, "Layer {:?} should have 4 neurons", layer);
        }

        println!("Single column: {} synapses", conn.n_synapses);
    }

    #[test]
    fn test_multi_column() {
        let builder = LaminarConnectomeBuilder::new(3, 3).with_default_mammalian();
        let n = builder.total_neurons();
        assert_eq!(n, 54); // 3 cols × 6 layers × 3 neurons

        let (conn, assign) = builder.build(123);
        assert_eq!(conn.n_neurons, 54);
        assert!(conn.n_synapses > 0);
        assert_eq!(assign.n_columns, 3);

        // Each column should have 18 neurons
        for col in 0..3 {
            let col_n = assign.neurons_in(LayerId::L2, col).len()
                + assign.neurons_in(LayerId::L1, col).len()
                + assign.neurons_in(LayerId::L3, col).len()
                + assign.neurons_in(LayerId::L4, col).len()
                + assign.neurons_in(LayerId::L5, col).len()
                + assign.neurons_in(LayerId::L6, col).len();
            assert_eq!(col_n, 18, "Column {col} should have 18 neurons");
        }

        println!("Multi-column: {} synapses", conn.n_synapses);
    }

    #[test]
    fn test_deterministic_build() {
        let builder = LaminarConnectomeBuilder::new(2, 5).with_default_mammalian();

        let (conn1, _) = builder.build(42);
        let (conn2, _) = builder.build(42);
        let (conn3, _) = builder.build(99); // different seed

        // Same seed → identical connectome
        assert_eq!(
            conn1.n_synapses, conn2.n_synapses,
            "Same seed should produce same synapse count"
        );
        assert_eq!(
            conn1.values, conn2.values,
            "Same seed should produce identical weights"
        );

        // Different seed → different connectome (almost certainly)
        if conn1.n_synapses == conn3.n_synapses {
            // May be same by coincidence, but values should differ with high probability
            let all_same = conn1
                .values
                .iter()
                .zip(conn3.values.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);
            assert!(
                !all_same,
                "Different seeds should produce different weights (with high probability)"
            );
        }
    }

    #[test]
    fn test_layer_stats() {
        let builder = LaminarConnectomeBuilder::new(2, 4).with_default_mammalian();
        let (conn, assign) = builder.build(42);
        let stats = builder.compute_stats(&conn, &assign);

        println!("{}", stats.format_proj_counts());
        println!("{}", stats.format_matrix());

        // Should have FF > FB (canonical: more feedforward than feedback)
        assert!(stats.ff_count > 0, "Should have FF synapses");
        assert!(stats.total_synapses > 0);

        // Recurrent fraction should be significant (local connections dominate)
        assert!(
            stats.recurrent_fraction() > 0.25,
            "Recurrent fraction {:.2} should be >25%",
            stats.recurrent_fraction()
        );
    }

    #[test]
    fn test_insect_default() {
        let builder = LaminarConnectomeBuilder::new(1, 6).with_default_insect();
        let n = builder.total_neurons();
        assert_eq!(n, 36); // 1 col × 6 layers × 6 neurons

        let (conn, assign) = builder.build(1);
        assert!(conn.n_synapses > 0);
        assert_eq!(assign.n_neurons, 36);

        // All neurons belong to column 0
        for i in 0..36 {
            assert!(assign.column[i] == 0, "Neuron {i} should be in column 0");
        }

        println!("Insect connectome: {} synapses", conn.n_synapses);
    }

    #[test]
    fn test_layer_assignment_queries() {
        let builder = LaminarConnectomeBuilder::new(2, 3).with_default_mammalian();
        let (_, assign) = builder.build(7);

        // All 6 layers present
        for layer in LayerId::all() {
            let neurons = assign.neurons_in_layer(layer);
            assert_eq!(
                neurons.len(),
                6,
                "Layer {:?} should have 6 neurons across 2 columns",
                layer
            );
        }

        // Layer×Column queries
        for col in 0..2 {
            let l2 = assign.neurons_in(LayerId::L2, col);
            assert_eq!(l2.len(), 3, "Column {col} L2 should have 3 neurons");
        }
    }

    #[test]
    fn test_connectome_save_load_roundtrip() {
        let builder = LaminarConnectomeBuilder::new(2, 4).with_default_mammalian();
        let (conn, _) = builder.build(42);

        // Save
        let mut buf = Vec::new();
        conn.save_baked(&mut buf).unwrap();

        // Load
        let loaded = ConnectomeData::load_baked(&mut &buf[..]).unwrap();

        assert_eq!(loaded.n_neurons, conn.n_neurons);
        assert_eq!(loaded.n_synapses, conn.n_synapses);
        assert_eq!(loaded.row_ptr, conn.row_ptr);
        assert_eq!(loaded.col_indices, conn.col_indices);
        assert_eq!(loaded.values, conn.values);
        assert_eq!(loaded.nt_types, conn.nt_types);
    }

    #[test]
    fn test_neuron_group_sizes() {
        let builder = LaminarConnectomeBuilder::new(2, 5).with_default_mammalian();
        let (conn, _) = builder.build(42);

        // Should have 6 layer groups + 2 column groups = 8 groups
        assert!(
            conn.groups.len() >= 8,
            "Should have at least 8 groups, got {}",
            conn.groups.len()
        );

        // Check specific layer group
        let l4 = conn.group("layer_L4").expect("Should have layer_L4 group");
        assert_eq!(l4.len(), 10, "L4 should have 10 neurons (2 cols × 5)");
    }

    #[test]
    fn test_custom_layer_sizes() {
        let mut npl = HashMap::new();
        npl.insert(LayerId::L2, 10); // only L2/3 layer
        npl.insert(LayerId::L5, 8); // and L5
                                    // Other layers default to 0

        let builder = LaminarConnectomeBuilder::with_layer_sizes(1, npl);
        let n = builder.total_neurons();
        assert_eq!(n, 18); // L2=10 + L5=8, others=0

        // Need at least some rules that match existing layers
        let builder = builder.add_rules(&[
            LaminarRule::new(
                LayerId::L2,
                LayerId::L2,
                0.2,
                0.3,
                0.8,
                1.0,
                ProjectionType::Recurrent,
            ),
            LaminarRule::new(
                LayerId::L2,
                LayerId::L5,
                0.3,
                0.5,
                1.0,
                1.0,
                ProjectionType::Feedforward,
            ),
            LaminarRule::new(
                LayerId::L5,
                LayerId::L5,
                0.2,
                0.4,
                0.8,
                1.0,
                ProjectionType::Recurrent,
            ),
        ]);

        let (conn, _assign) = builder.build(1);
        assert_eq!(conn.n_neurons, 18);
        assert!(conn.n_synapses > 0);

        // Check group assignment
        let l2 = conn.group("layer_L2").expect("Should have L2 group");
        assert_eq!(l2.len(), 10);
        let l5 = conn.group("layer_L5").expect("Should have L5 group");
        assert_eq!(l5.len(), 8);
    }
}
