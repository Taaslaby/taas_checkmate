// ═══════════════════════════════════════════════════════════════════════════
// trp_attention.wgsl — TRP pairwise attention: T·R^T × distance decay
// ═══════════════════════════════════════════════════════════════════════════
//
// Один поток = одна пара (i, j). Без тайлинга — D=8 слишком мал для него.
//
// score(i→j) = |T_i · R_j| / sqrt_d  ×  exp(−dist² / 2σ²)
//
// Only scores > prune_threshold → atomic append to sparse output.

struct Uniforms {
    n: u32,
    d: u32,
    pad0: u32,
    pad1: u32,
    sqrt_d: f32,          // 1/√d (precomputed inverse)
    inv_2sigma2: f32,     // 1/(2σ²)
    prune_threshold: f32,
    max_dist: f32,
}

@group(0) @binding(0) var<storage, read> tropisms: array<f32>;       // N × D
@group(0) @binding(1) var<storage, read> receptors: array<f32>;      // N × D
@group(0) @binding(2) var<storage, read> positions: array<f32>;      // N × 3 (x,y,z)
@group(0) @binding(3) var<storage, read_write> syn_out: array<f32>;  // triplets: pre,post,score
@group(0) @binding(4) var<storage, read_write> syn_count: atomic<u32>;
@group(0) @binding(5) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;

    if i >= u.n || j >= u.n { return; }

    // ── Distance check ──
    let p_i = vec3<f32>(positions[i*3u], positions[i*3u+1u], positions[i*3u+2u]);
    let p_j = vec3<f32>(positions[j*3u], positions[j*3u+1u], positions[j*3u+2u]);
    let dist2 = (p_i.x-p_j.x)*(p_i.x-p_j.x)
              + (p_i.y-p_j.y)*(p_i.y-p_j.y)
              + (p_i.z-p_j.z)*(p_i.z-p_j.z);

    if dist2 > u.max_dist * u.max_dist { return; }

    // ── Dot product: Σ T[i][k] · R[j][k] ──
    var dot = 0.0f;
    let base_i = i * u.d;
    let base_j = j * u.d;
    // Unroll for D=8 — WGSL compiler handles this well.
    for (var k = 0u; k < u.d; k++) {
        dot += tropisms[base_i + k] * receptors[base_j + k];
    }

    let raw = abs(dot) * u.sqrt_d;
    let decay = exp(-dist2 * u.inv_2sigma2);
    let score = raw * decay;

    if score <= u.prune_threshold { return; }

    // ── Atomic append ──
    let idx = atomicAdd(&syn_count, 1u);
    let max_syns = u.n * 64u;
    if idx < max_syns {
        syn_out[idx * 3u + 0u] = f32(i);
        syn_out[idx * 3u + 1u] = f32(j);
        syn_out[idx * 3u + 2u] = score;
    }
}