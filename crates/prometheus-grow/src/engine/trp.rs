//! TRP attention for PROMETHEUS: T·R^T × DistanceDecay → synapses
//! (semantic synaptogenesis through attention).
//!
//! Instead of "distance + cell type" (NETMORPH) — semantic growth:
//! a neuron seeks not the nearest cell but the one whose Receptor matches
//! its Tropism. During learning the Tropism mutates → old connections die
//! → new ones grow. Literally neurogenesis + pruning with semantics.
//!
//!   score(i→j) = |Tᵢ · Rⱼ| / √d  ×  exp(−dist² / 2σ²)
//!
//! Where:
//!   Tᵢ — Tropism of cell i ("which pattern I seek")
//!   Rⱼ — Receptor of cell j ("which pattern I broadcast")
//!   d  — embedding dimension (8)
//!   σ  — spatial decay radius
//!
//! A synapse forms if score > threshold (prune_threshold).
//! When T changes (learning) — old low-score synapses die.
//!
//! Integration with the CX3D engine:
//!   - Embeddings stored in HashMap<cell_id, [f32; 8]>
//!   - Spatial queries through the existing UniformGrid
//!   - Synapses appended to World.synapses

use std::collections::HashMap;

use crate::cell::CellType;
use crate::uniform_grid::UniformGrid;
use crate::world::{Synapse, SynapseKind};
use crate::Vec3;

/// Embedding dimension (Receptor / Tropism).
pub const TRP_DIM: usize = 8;

/// TRP embedding for a single cell.
#[derive(Clone, Debug)]
pub struct TrpEmbedding {
    /// Cell "passport": what it broadcasts (static).
    pub receptor: [f32; TRP_DIM],
    /// Axon "intention": what it seeks (mutates during learning).
    pub tropism: [f32; TRP_DIM],
    /// Accumulated reward (trophic signal).
    pub troph: f32,
}

impl TrpEmbedding {
    /// Pseudo-random normalized embedding from a seed.
    pub fn random(seed: u64) -> Self {
        let mut emb = [0.0f32; TRP_DIM];
        for (i, v) in emb.iter_mut().enumerate() {
            let s = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64);
            // Spread upper 32 bits into -1..1 range.
            *v = ((s >> 32) as i32) as f32 / (i32::MAX as f32);
        }
        let mag: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(0.001);
        for v in emb.iter_mut() {
            *v /= mag;
        }
        Self {
            receptor: emb,
            tropism: emb,
            troph: 0.0,
        }
    }

    /// Deterministic embedding from cell type + position.
    pub fn from_cell(cell_type: CellType, pos: Vec3) -> Self {
        // Hash position + type → embedding.
        let h = |s: f64| -> u32 { (s.abs() * 1_000_000.0) as u32 };
        let seed = h(pos.x)
            ^ h(pos.y).wrapping_mul(3)
            ^ h(pos.z).wrapping_mul(7)
            ^ (cell_type.as_usize() as u32).wrapping_mul(13);
        let mut emb = [0.0f32; TRP_DIM];
        for (i, v) in emb.iter_mut().enumerate() {
            let s = seed.wrapping_mul((i as u32).wrapping_add(1).wrapping_mul(0x9E37_79B9));
            *v = (((s >> 16) as i16) as f32) / 32768.0;
        }
        let mag: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(0.001);
        for v in emb.iter_mut() {
            *v /= mag;
        }
        Self {
            receptor: emb,
            tropism: emb,
            troph: 0.0,
        }
    }

    /// Nudge the tropism toward a target (reward).
    pub fn nudge_tropism(&mut self, target: &[f32; TRP_DIM], rate: f32) {
        for (t, r) in self.tropism.iter_mut().zip(target.iter()) {
            *t = *t * (1.0 - rate) + *r * rate;
        }
        let mag: f32 = self
            .tropism
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(0.001);
        for v in self.tropism.iter_mut() {
            *v /= mag;
        }
    }
}

/// TRP synaptogenesis parameters.
#[derive(Clone, Debug)]
pub struct TrpParams {
    /// Spatial decay sigma.
    pub dist_sigma: f64,
    /// √d (dot-product normalization).
    pub sqrt_d: f32,
    /// Prune threshold: synapses below this score are not formed.
    pub prune_threshold: f32,
    /// Maximum candidate search distance.
    pub max_dist: f64,
    /// Top-K synapses per cell (0 = unlimited).
    pub top_k: usize,
}

impl Default for TrpParams {
    fn default() -> Self {
        Self {
            dist_sigma: 0.3,
            sqrt_d: (TRP_DIM as f32).sqrt(),
            prune_threshold: 0.01,
            max_dist: 1.5,
            top_k: 16,
        }
    }
}

/// TRP score for a pair of cells.
fn trp_score(t: &[f32; TRP_DIM], r: &[f32; TRP_DIM], dist_sq: f64, params: &TrpParams) -> f32 {
    let dot: f32 = t.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
    let semantic = dot.abs() / params.sqrt_d;
    let spatial = (-dist_sq / (2.0 * params.dist_sigma * params.dist_sigma)).exp() as f32;
    semantic * spatial
}

/// Form synapses through TRP attention.
///
/// `cell_positions` — soma position of each cell (index-aligned).
/// `cell_types` — type of each cell.
/// `embeddings` — TRP embeddings (key = cell.id).
/// `existing_syns` — already existing (pre, post) pairs, to avoid duplicates.
///
/// Returns the newly formed synapses.
pub fn form_synapses_trp(
    cell_positions: &[Vec3],
    _cell_types: &[CellType],
    cell_ids: &[usize],
    embeddings: &HashMap<usize, TrpEmbedding>,
    existing_syns: &[(usize, usize)],
    grid: &UniformGrid,
    params: &TrpParams,
) -> Vec<Synapse> {
    let n = cell_positions.len();
    let mut new_syns = Vec::new();
    let mut seen: HashMap<(usize, usize), bool> =
        existing_syns.iter().map(|&p| (p, true)).collect();

    // Number of already existing outgoing synapses per cell (for top_k).
    let mut existing_out: HashMap<usize, usize> = HashMap::new();
    for &(pre, _post) in existing_syns {
        *existing_out.entry(pre).or_insert(0) += 1;
    }

    for i in 0..n {
        let pre_id = cell_ids[i];
        let Some(pre_emb) = embeddings.get(&pre_id) else {
            continue;
        };
        let pos = cell_positions[i];

        // Spatial query through the UniformGrid.
        let candidates = grid.query(pos, params.max_dist);

        // Score each candidate with the TRP-score.
        let mut scored: Vec<(f32, usize, f32)> = Vec::new(); // (score, post_idx, dist_sq)
        for &j in &candidates {
            if j == i {
                continue;
            }
            let post_id = cell_ids[j];
            if seen.contains_key(&(pre_id, post_id)) {
                continue;
            }
            let Some(post_emb) = embeddings.get(&post_id) else {
                continue;
            };

            let dist_sq = (pos - cell_positions[j]).len().powi(2);
            let score = trp_score(&pre_emb.tropism, &post_emb.receptor, dist_sq, params);

            if score >= params.prune_threshold {
                scored.push((score, j, dist_sq as f32));
            }
        }

        // Top-K: account for this cell's already existing outgoing synapses.
        if params.top_k > 0 {
            let used = existing_out.get(&pre_id).copied().unwrap_or(0);
            let budget = params.top_k.saturating_sub(used);
            if scored.len() > budget {
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(budget);
            }
        }

        for (score, post_idx, _) in scored {
            let post_id = cell_ids[post_idx];
            // Synapse sign: dot > 0 → excitatory, otherwise inhibitory.
            // (Biological rationale: T·R match = excitation, mismatch = inhibition.)
            let post_emb = embeddings.get(&post_id).unwrap();
            let raw_dot: f32 = pre_emb
                .tropism
                .iter()
                .zip(post_emb.receptor.iter())
                .map(|(a, b)| a * b)
                .sum();
            let kind = if raw_dot >= 0.0 {
                SynapseKind::Excitatory
            } else {
                SynapseKind::Inhibitory
            };
            let weight = (score as f64).clamp(0.01, 2.0);

            seen.insert((pre_id, post_id), true);
            new_syns.push(Synapse {
                pre: pre_id,
                post: post_id,
                kind,
                weight,
            });
        }
    }

    new_syns
}

/// Prune existing synapses: recompute the score (T changed) and drop those
/// below the threshold. Returns (survivors, pruned_count).
///
/// `positions_by_id` — cell position by id (key = cell.id).
pub fn prune_synapses_trp(
    synapses: &[Synapse],
    positions_by_id: &HashMap<usize, Vec3>,
    embeddings: &HashMap<usize, TrpEmbedding>,
    params: &TrpParams,
) -> (Vec<Synapse>, usize) {
    let mut alive = Vec::with_capacity(synapses.len());
    let mut pruned = 0usize;

    for syn in synapses {
        let (Some(pre_emb), Some(post_emb)) = (embeddings.get(&syn.pre), embeddings.get(&syn.post))
        else {
            // No embedding — keep the synapse as is.
            alive.push(*syn);
            continue;
        };
        let (Some(ppre), Some(ppost)) = (
            positions_by_id.get(&syn.pre),
            positions_by_id.get(&syn.post),
        ) else {
            alive.push(*syn);
            continue;
        };
        let dist_sq = (*ppre - *ppost).len().powi(2);
        let score = trp_score(&pre_emb.tropism, &post_emb.receptor, dist_sq, params);
        if score >= params.prune_threshold {
            alive.push(*syn);
        } else {
            pruned += 1;
        }
    }

    (alive, pruned)
}

/// Initialize TRP embeddings for all cells.
pub fn init_embeddings(
    cell_ids: &[usize],
    cell_types: &[CellType],
    positions: &[Vec3],
) -> HashMap<usize, TrpEmbedding> {
    let mut map = HashMap::new();
    for i in 0..cell_ids.len() {
        let emb = TrpEmbedding::from_cell(cell_types[i], positions[i]);
        map.insert(cell_ids[i], emb);
    }
    map
}
