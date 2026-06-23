//! CUDA T3 Music Encoding/Decoding Benchmark.
//! Runs as integration test via `cargo test --test bench_t3_music`

use std::time::Instant;

fn main() {
    println!("\n═══ CUDA T3 Music Benchmark ═══");

    let gpus = cuda_kernels::get_device_count().unwrap_or(0);
    if gpus == 0 { eprintln!("⚠ No GPU — skipping"); return; }
    let _ = cuda_kernels::set_device(0);
    println!("  GPU: sm_89 + --use_fast_math");

    // ─── Config ────────────────────────────────────────
    let n_osc = 512;
    let frame_size = 256;

    // Load audio (pre-processed by Python)
    let data = std::fs::read("/tmp/frank_sinatra.f32").unwrap_or_else(|_| {
        eprintln!("⚠ No audio file at /tmp/frank_sinatra.f32 — generating test data");
        vec![0.0f32; 220500].iter().flat_map(|&x| x.to_le_bytes()).collect()
    });
    let audio: Vec<f32> = data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]]))
        .take(220500).collect();
    let n_frames = audio.len() / frame_size;
    println!("  Audio: {}s, {} frames @ {} samples/frame",
        audio.len() / 22050, n_frames.min(1000), frame_size);

    // ─── Init oscillators ──────────────────────────────
    let mut phases: Vec<f32> = (0..n_osc).map(|i| (i as f32 * 0.001 + 0.5) % 6.2832).collect();
    let omegas: Vec<f32> = (0..n_osc).map(|i| 0.5 + (i as f32 / n_osc as f32)).collect();
    let mut drives = vec![0.0f32; n_osc];
    let mut cos_out = vec![0.0f32; n_osc];
    let mut sin_out = vec![0.0f32; n_osc];
    let mut w = vec![0.0f32; n_osc];
    let mut b = vec![0.0f32; 1];
    // deterministic init: approx sin/cos over n_osc
    for i in 0..n_osc { w[i] = (i as f32 * 0.137).sin() * 0.01; }

    // Warmup
    for _ in 0..3 {
        cuda_kernels::t3_drive_encoder(&mut drives, &vec![audio[0]; n_osc], 20.0).unwrap();
        cuda_kernels::kuramoto_prepare(&phases, &mut cos_out, &mut sin_out).unwrap();
        cuda_kernels::kuramoto_update(&mut phases, &omegas, &drives, 0.001).unwrap();
    }

    let n_iter = 5;
    let max_frames = n_frames.min(500);

    // ─── 1. T3 Step ────────────────────────────────────
    println!("\n─── 1. T3 Step (drive + prepare + update) ───");
    let t0 = Instant::now();
    for _ in 0..n_iter {
        for fi in 0..max_frames {
            let inp = audio[fi * frame_size % audio.len()];
            cuda_kernels::t3_drive_encoder(&mut drives, &vec![inp; n_osc], 20.0).unwrap();
            cuda_kernels::kuramoto_prepare(&phases, &mut cos_out, &mut sin_out).unwrap();
            cuda_kernels::kuramoto_update(&mut phases, &omegas, &drives, 0.001).unwrap();
        }
    }
    let total = n_iter * max_frames;
    let ms_t3 = t0.elapsed().as_secs_f64() / total as f64 * 1000.0;
    println!("  {:>20}: {:.4}ms/frame  ({:.0} fps)", "3 CUDA kernels", ms_t3, 1000.0/ms_t3);

    // ─── 2. NLMS Decoder ───────────────────────────────
    println!("\n─── 2. NLMS Decoder Training ───");
    let t0 = Instant::now();
    for _ in 0..n_iter {
        for fi in 0..max_frames {
            let target = audio[fi * frame_size % audio.len()];
            cuda_kernels::nlms_train(&mut w, &mut b, &cos_out, target, 0.01, 1e-8).unwrap();
        }
    }
    let ms_nlms = t0.elapsed().as_secs_f64() / total as f64 * 1000.0;
    println!("  {:>20}: {:.4}ms/frame  ({:.0} fps)", "NLMS train", ms_nlms, 1000.0/ms_nlms);
    println!("  {:>20}: first weight = {:.6}, bias = {:.6}", "Decoder state", w[0], b[0]);

    // ─── 3. Full Pipeline ──────────────────────────────
    println!("\n─── 3. Full Pipeline (3 + NLMS) ───");
    let t0 = Instant::now();
    for _ in 0..n_iter.min(3) {
        for fi in 0..max_frames {
            let inp = audio[fi * frame_size % audio.len()];
            cuda_kernels::t3_drive_encoder(&mut drives, &vec![inp; n_osc], 20.0).unwrap();
            cuda_kernels::kuramoto_prepare(&phases, &mut cos_out, &mut sin_out).unwrap();
            cuda_kernels::kuramoto_update(&mut phases, &omegas, &drives, 0.001).unwrap();
            cuda_kernels::nlms_train(&mut w, &mut b, &cos_out, inp, 0.01, 1e-8).unwrap();
        }
    }
    let total2 = n_iter.min(3) * max_frames;
    let ms_full = t0.elapsed().as_secs_f64() / total2 as f64 * 1000.0;
    println!("  {:>20}: {:.4}ms/frame  ({:.0} fps)", "4 CUDA kernels", ms_full, 1000.0/ms_full);

    // ─── Realtime Factor ───────────────────────────────
    // audio seconds per frame = frame_size / sr
    // processing seconds per frame = ms / 1000
    // realtime = audio_sec / proc_sec = (frame_size/sr) / (ms/1000) = frame_size * 1000 / (sr * ms)
    let sr = 22050.0;
    let rt_t3 = frame_size as f64 * 1000.0 / (sr * ms_t3);
    let rt_full = frame_size as f64 * 1000.0 / (sr * ms_full);
    println!("\n─── Realtime Performance ───");
    println!("  {:>20}: {:.0}× realtime", "T3 step only", rt_t3);
    println!("  {:>20}: {:.0}× realtime", "Full pipeline", rt_full);
    println!("  {:>20}: {} oscillators, {} samples/frame", "Config", n_osc, frame_size);
    println!("\n═══ CUDA T3 Benchmark Complete ═══");
}

#[test]
fn bench_t3_music_cuda() { main(); }
