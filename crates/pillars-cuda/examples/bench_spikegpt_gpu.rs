//! # Bench: SpikeGPT-shaped forward on GPU via `CudaKernel`
//!
//! Same workload as `bench_spikegpt_forward.rs` (CPU), executed through the
//! generic NVRTC launcher: one `linear_spike` kernel per matmul with the real
//! host↔device round-trip (the current `CudaKernel` contract).
//!
//! Real SpikeGPT-216M dimensions: D=768, layers=18, ffn=3072, vocab=50277.
//!
//! ```
//! cargo run -p pillars-cuda --example bench_spikegpt_gpu --release
//! ```

use std::time::Instant;

use pillars_cuda::{BufferArg, CudaKernel, LaunchConfig, ScalarArg};

const D: usize = 768; // d_model
const FFN: usize = 3072; // ffn hidden
const LAYERS: usize = 18; // n_layer
const VOCAB: usize = 50277; // vocab
const DENSE_PER_LAYER: usize = 5; // att×4 + ffn receptance
const FFN_PER_LAYER: usize = 2; // ffn key + ffn value

const KERNEL_SRC: &str = r#"
extern "C" __global__ void linear_spike(
    const float* x, const float* w, float* out, int m, int n, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    int i = idx / n;
    int j = idx % n;
    float acc = 0.0f;
    for (int t = 0; t < k; ++t) acc += x[i * k + t] * w[j * k + t];
    out[idx] = acc >= 0.0f ? 1.0f : 0.0f;
}
"#;

fn per_token_flops() -> f64 {
    let dense = DENSE_PER_LAYER as f64 * D as f64 * D as f64;
    let ffn_ = FFN_PER_LAYER as f64 * D as f64 * FFN as f64;
    let head = D as f64 * VOCAB as f64;
    2.0 * (LAYERS as f64 * (dense + ffn_) + head)
}

fn main() {
    println!("SpikeGPT-216M-shaped forward on GPU (CudaKernel), real dims: D={D}, layers={LAYERS}");
    println!("per token: {:.2} MFLOPs", per_token_flops() / 1e6);
    let kernel = CudaKernel::compile(KERNEL_SRC, "linear_spike").expect("CUDA device required");

    let mut w_d = vec![0.01f32; D * D];
    let mut w_f = vec![0.01f32; D * FFN];
    let mut w_h = vec![0.01f32; D * VOCAB];
    let mut x = vec![0.1f32; D];
    let mut out = vec![0.0f32; D];
    let config = LaunchConfig::for_len(D).unwrap();

    // Warmup sanity: output must be binary.
    kernel
        .launch(
            &mut [
                BufferArg::F32(&mut x),
                BufferArg::F32(&mut w_d),
                BufferArg::F32(&mut out),
            ],
            &[
                ScalarArg::I32(1),
                ScalarArg::I32(D as i32),
                ScalarArg::I32(D as i32),
            ],
            config,
        )
        .unwrap();
    assert!(out.iter().all(|&v| v == 0.0 || v == 1.0));

    // ── Autoregressive pattern: m=1, one launch per matmul ──
    let t0 = Instant::now();
    let runs = 5;
    for _ in 0..runs {
        for _ in 0..LAYERS {
            for _ in 0..DENSE_PER_LAYER {
                kernel
                    .launch(
                        &mut [
                            BufferArg::F32(&mut x),
                            BufferArg::F32(&mut w_d),
                            BufferArg::F32(&mut out),
                        ],
                        &[
                            ScalarArg::I32(1),
                            ScalarArg::I32(D as i32),
                            ScalarArg::I32(D as i32),
                        ],
                        config,
                    )
                    .unwrap();
                x.copy_from_slice(&out);
            }
            for _ in 0..FFN_PER_LAYER {
                kernel
                    .launch(
                        &mut [
                            BufferArg::F32(&mut x),
                            BufferArg::F32(&mut w_f),
                            BufferArg::F32(&mut out),
                        ],
                        &[
                            ScalarArg::I32(1),
                            ScalarArg::I32(FFN as i32),
                            ScalarArg::I32(D as i32),
                        ],
                        LaunchConfig::for_len(FFN).unwrap(),
                    )
                    .unwrap();
                x.copy_from_slice(&out);
            }
        }
        kernel
            .launch(
                &mut [
                    BufferArg::F32(&mut x),
                    BufferArg::F32(&mut w_h),
                    BufferArg::F32(&mut out),
                ],
                &[
                    ScalarArg::I32(1),
                    ScalarArg::I32(VOCAB as i32),
                    ScalarArg::I32(D as i32),
                ],
                LaunchConfig::for_len(VOCAB).unwrap(),
            )
            .unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    println!(
        "per-token (m=1): {:.3} ms/token  =>  {:.1} tokens/s  ({:.2} GFLOPS)",
        elapsed * 1e3,
        1.0 / elapsed,
        per_token_flops() / 1e9
    );

    // ── Batched pattern: m=64, one launch per matmul ──
    let b = 64usize;
    let mut xb = vec![0.1f32; b * D];
    let mut ob = vec![0.0f32; b * D];
    let mut head_out = vec![0.0f32; b * VOCAB];
    let t1 = Instant::now();
    for _ in 0..20 {
        for _ in 0..LAYERS {
            for _ in 0..DENSE_PER_LAYER {
                kernel
                    .launch(
                        &mut [
                            BufferArg::F32(&mut xb),
                            BufferArg::F32(&mut w_d),
                            BufferArg::F32(&mut ob),
                        ],
                        &[
                            ScalarArg::I32(b as i32),
                            ScalarArg::I32(D as i32),
                            ScalarArg::I32(D as i32),
                        ],
                        LaunchConfig::for_len(b * D).unwrap(),
                    )
                    .unwrap();
                xb.copy_from_slice(&ob);
            }
            for _ in 0..FFN_PER_LAYER {
                kernel
                    .launch(
                        &mut [
                            BufferArg::F32(&mut xb),
                            BufferArg::F32(&mut w_f),
                            BufferArg::F32(&mut ob),
                        ],
                        &[
                            ScalarArg::I32(b as i32),
                            ScalarArg::I32(FFN as i32),
                            ScalarArg::I32(D as i32),
                        ],
                        LaunchConfig::for_len(b * FFN).unwrap(),
                    )
                    .unwrap();
                xb.copy_from_slice(&ob);
            }
        }
        kernel
            .launch(
                &mut [
                    BufferArg::F32(&mut xb),
                    BufferArg::F32(&mut w_h),
                    BufferArg::F32(&mut head_out),
                ],
                &[
                    ScalarArg::I32(b as i32),
                    ScalarArg::I32(VOCAB as i32),
                    ScalarArg::I32(D as i32),
                ],
                LaunchConfig::for_len(b * VOCAB).unwrap(),
            )
            .unwrap();
    }
    let b_elapsed = t1.elapsed().as_secs_f64() / 20.0;
    println!(
        "batch (m={b}): {:.2} ms/step  =>  {:.0} tokens/s  ({:.2} GFLOPS)",
        b_elapsed * 1e3,
        b as f64 / b_elapsed,
        b as f64 * per_token_flops() / b_elapsed / 1e9
    );
}
