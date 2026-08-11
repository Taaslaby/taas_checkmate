//! # Bench: SpikeGPT-shaped forward on CPU (rayon SIMD)
//!
//! Measures the *compute core* of the real SpikeGPT-216M forward pass on its
//! exact dimensions (read from the checkpoint): d_model=768, n_layer=18,
//! ffn hidden=3072, vocab=50277, binary spike activation after every matmul.
//!
//! Per token: 18 blocks × (att key/value/receptance/output [768²] +
//! ffn key/value [768×3072] + ffn receptance [768²]) + head [768×50277]
//! = 128 matmuls.
//!
//! ```
//! cargo bench --bench bench_spikegpt_forward
//! ```

use std::time::Instant;

use rayon::prelude::*;

const D: usize = 768; // d_model (real SpikeGPT-216M)
const FFN: usize = 3072; // ffn hidden = 4×d_model
const LAYERS: usize = 18; // n_layer (real)
const VOCAB: usize = 50277; // vocab size (real)
const DENSE_PER_LAYER: usize = 5; // att×4 + ffn receptance
const FFN_PER_LAYER: usize = 2; // ffn key + ffn value
const TOKEN_MATMULS: usize = LAYERS * (DENSE_PER_LAYER + FFN_PER_LAYER) + 1; // 127 + head

/// out[m][n] = spike(a[m][k] @ w[k][n]) — SpikeGPT binary spiking activation.
fn matmul_spike(a: &[f32], w: &[f32], out: &mut [f32], _m: usize, n: usize, k: usize) {
    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        let arow = &a[i * k..(i + 1) * k];
        for (j, o) in orow.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += arow[t] * w[j * k + t];
            }
            *o = if acc >= 0.0 { 1.0 } else { 0.0 };
        }
    });
}

/// Real per-token FLOPs: layers×(5·D² + 2·D·FFN) + head(D·VOCAB).
fn per_token_flops() -> f64 {
    let dense = DENSE_PER_LAYER as f64 * D as f64 * D as f64;
    let ffn_ = FFN_PER_LAYER as f64 * D as f64 * FFN as f64;
    let head = D as f64 * VOCAB as f64;
    2.0 * (LAYERS as f64 * (dense + ffn_) + head)
}

fn main() {
    println!(
        "SpikeGPT-216M-shaped forward (real dims): D={D}, layers={LAYERS}, ffn={FFN}, vocab={VOCAB}"
    );
    println!(
        "per token: {TOKEN_MATMULS} matmuls, {:.2} MFLOPs/token",
        per_token_flops() / 1e6
    );
    let w_d = vec![0.01f32; D * D];
    let w_f = vec![0.01f32; D * FFN];
    let w_h = vec![0.01f32; D * VOCAB];
    let mut x = vec![0.1f32; D];
    let mut out = vec![0.0f32; D];

    // ── Autoregressive pattern: m=1, sequential matmuls (realistic) ──
    let t0 = Instant::now();
    let runs = 10;
    for _ in 0..runs {
        for _ in 0..LAYERS {
            for _ in 0..DENSE_PER_LAYER {
                matmul_spike(&x, &w_d, &mut out, 1, D, D);
                x.copy_from_slice(&out);
            }
            for _ in 0..FFN_PER_LAYER {
                matmul_spike(&x, &w_f, &mut out, 1, FFN, D);
                x.copy_from_slice(&out);
            }
        }
        matmul_spike(&x, &w_h, &mut out, 1, VOCAB, D); // head → logits
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    println!(
        "per-token (m=1): {:.3} ms/token  =>  {:.1} tokens/s  ({:.2} GFLOPS)",
        elapsed * 1e3,
        1.0 / elapsed,
        per_token_flops() / 1e9
    );

    // ── Batched pattern: m=64 (parallel across tokens) ──
    let m = 64usize;
    let mut xb = vec![0.1f32; m * D];
    let mut ob = vec![0.0f32; m * D];
    let mut head_out = vec![0.0f32; m * VOCAB];
    let t1 = Instant::now();
    for _ in 0..50 {
        for _ in 0..LAYERS {
            for _ in 0..DENSE_PER_LAYER {
                matmul_spike(&xb, &w_d, &mut ob, m, D, D);
                xb.copy_from_slice(&ob);
            }
            for _ in 0..FFN_PER_LAYER {
                matmul_spike(&xb, &w_f, &mut ob, m, FFN, D);
                xb.copy_from_slice(&ob);
            }
        }
        matmul_spike(&xb, &w_h, &mut head_out, m, VOCAB, D);
    }
    let b_elapsed = t1.elapsed().as_secs_f64() / 50.0;
    println!(
        "batch (m={m}): {:.2} ms/step  =>  {:.0} tokens/s  ({:.2} GFLOPS)",
        b_elapsed * 1e3,
        m as f64 / b_elapsed,
        m as f64 * per_token_flops() / b_elapsed / 1e9
    );
}
