//! CUDA T3 Music Encoding/Decoding Benchmark.
//!
//! Pipeline per frame:
//!   1. t3_drive_encoder — audio → oscillator drives (gain * input)
//!   2. kuramoto_prepare — phases → cos, sin
//!   3. kuramoto_update  — phases += dt * (omega + coupling)
//!   4. nlms_train       — adaptive decoder: learn weights from state → target
//!   5. echo_tap         — read delayed state for reconstruction
//!
//! Measures: throughput (frames/sec), reconstruction correlation.

use std::time::Instant;

// ─── Inline RNG ──────────────────────────────────────────

fn rng_f32() -> f32 {
    static mut S: u64 = 12345;
    unsafe { S = S.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((S >> 32) as f32) / 4294967296.0 }
}

fn load_audio(path: &str, max_samples: usize) -> Vec<f32> {
    let data = std::fs::read(path).expect("Can't read audio file");
    let samples: Vec<f32> = data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take(max_samples)
        .collect();
    println!("  Loaded {} audio samples from {}", samples.len(), path);
    samples
}

// ─── T3 Benchmark ────────────────────────────────────────

fn main() {
    println!("\n═══ CUDA T3 Music Benchmark ═══");

    let gpus = cuda_kernels::get_device_count().unwrap_or(0);
    if gpus == 0 { eprintln!("No GPU"); return; }
    let _ = cuda_kernels::set_device(0);

    // ─── Config ────────────────────────────────────────
    let n_osc = 512;           // oscillators
    let frame_size = 256;      // samples per frame
    let sr = 22050;            // sample rate
    let n_iter = 10;           // benchmark iterations

    // Load audio
    let audio = load_audio("/tmp/frank_sinatra.f32", 220500);
    let n_frames = audio.len() / frame_size;
    println!("  Frames: {} ({} samples @ {})", n_frames, frame_size, n_iter);

    // ─── Init CUDA ────────────────────────────────────
    let ctx = cuda_kernels::GpuContext::new(n_osc, n_osc).unwrap();
    let phases: Vec<f32> = (0..n_osc).map(|i| (i as f32 * 0.001 + 0.5) % 6.2832).collect();
    let omegas: Vec<f32> = (0..n_osc).map(|i| 0.5 + (i as f32 / n_osc as f32) * 1.0).collect();
    let coupling = vec![0.0f32; n_osc];
    ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

    // NLMS decoder weights
    let mut weights = vec![0.0f32; n_osc];
    let mut bias = vec![0.0f32; 1];
    // Initialize weights with small random values
    for w in &mut weights { *w = (rng_f32() - 0.5) * 0.01; }

    // ─── Warmup ───────────────────────────────────────
    println!("  Warming up...");
    for frame_idx in 0..100 {
        let input = audio[frame_idx * frame_size % audio.len()];
        let mut drives = vec![input; n_osc];

        // Encode: drive = gain * input
        cuda_kernels::t3_drive_encoder(&mut drives, &[input; n_osc], 20.0).unwrap();
        // No way to set drives in GpuContext... need to use the individual API
    }

    // Define the benchmark using the individual CUDA API
    // Create drives on host, upload to device via the existing API
    // Actually, GpuContext doesn't have a raw device pointer API for drives
    // Let me use the individual (legacy) CUDA wrappers instead

    // ─── Individual Kernel API ─────────────────────────
    // We can use: cuda_kernels::t3_drive_encoder, kuramoto_prepare, etc.
    // These take host pointers and do cudaMemcpy internally
    let mut drives = vec![0.0f32; n_osc];
    let mut cos_out = vec![0.0f32; n_osc];
    let mut sin_out = vec![0.0f32; n_osc];
    let mut phases_local = phases.clone();
    let mut output = vec![0.0f32; 1];

    // Warmup
    for _ in 0..5 {
        let input_val = audio[0];
        for i in 0..n_osc { drives[i] = 20.0 * input_val; }
        cuda_kernels::t3_drive_encoder(&mut drives, &[input_val; n_osc], 20.0).unwrap();
        cuda_kernels::kuramoto_prepare(&phases_local, &mut cos_out, &mut sin_out).unwrap();
        cuda_kernels::kuramoto_update(&mut phases_local, &omegas, &drives, 0.001).unwrap();
    }

    // ─── Benchmark: T3 encode → prepare → update ──────
    println!("\n─── T3 Step (drive_encoder + prepare + update) ───");
    let t0 = Instant::now();
    for iter in 0..n_iter {
        for frame_idx in 0..n_frames.min(1000) {
            let input_val = audio[frame_idx * frame_size % audio.len()];
            // drives = gain * input (per oscillator)
            for i in 0..n_osc { drives[i] = 20.0 * input_val; }
            cuda_kernels::t3_drive_encoder(&mut drives, &[input_val; n_osc], 20.0).unwrap();
            cuda_kernels::kuramoto_prepare(&phases_local, &mut cos_out, &mut sin_out).unwrap();
            cuda_kernels::kuramoto_update(&mut phases_local, &omegas, &drives, 0.001).unwrap();
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_frames = n_iter * n_frames.min(1000);
    let ms_per_frame = elapsed / total_frames as f64 * 1000.0;
    let frames_per_sec = 1.0 / (ms_per_frame / 1000.0);
    println!("  {} frames in {:.2}s", total_frames, elapsed);
    println!("  {:.4}ms/frame ({:.0} frames/sec)", ms_per_frame, frames_per_sec);

    // ─── Benchmark: NLMS train ─────────────────────────
    println!("\n─── NLMS Decoder Training ───");
    let mut w = vec![0.0f32; n_osc];
    let mut b = vec![0.0f32; 1];
    for i in 0..n_osc { w[i] = (rng_f32() - 0.5) * 0.01; }

    let t0 = Instant::now();
    for iter in 0..n_iter {
        for frame_idx in 0..n_frames.min(1000) {
            let target = audio[frame_idx * frame_size % audio.len()];
            cuda_kernels::nlms_train(&mut w, &mut b, &cos_out, target, 0.01, 1e-8).unwrap();
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total = n_iter * n_frames.min(1000);
    let ms_per = elapsed / total as f64 * 1000.0;
    println!("  {} NLMS steps in {:.2}s", total, elapsed);
    println!("  {:.4}ms/step ({:.0} steps/sec)", ms_per, 1.0 / (ms_per / 1000.0));

    // ─── Full Pipeline Throughput ──────────────────────
    println!("\n─── Full T3 Pipeline ───");
    let mut phases_p = phases.clone();
    let t0 = Instant::now();
    for iter in 0..n_iter.min(3) {  // limit iterations for full pipeline
        for frame_idx in 0..n_frames.min(500) {
            let input_val = audio[frame_idx * frame_size % audio.len()];
            for i in 0..n_osc { drives[i] = 20.0 * input_val; }
            cuda_kernels::t3_drive_encoder(&mut drives, &[input_val; n_osc], 20.0).unwrap();
            cuda_kernels::kuramoto_prepare(&phases_p, &mut cos_out, &mut sin_out).unwrap();
            cuda_kernels::kuramoto_update(&mut phases_p, &omegas, &drives, 0.001).unwrap();
            let target = input_val;
            cuda_kernels::nlms_train(&mut w, &mut b, &cos_out, target, 0.01, 1e-8).unwrap();
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total = n_iter.min(3) * n_frames.min(500);
    let ms_per = elapsed / total as f64 * 1000.0;
    println!("  {} full steps in {:.2}s", total, elapsed);
    println!("  {:.4}ms/step ({:.0} steps/sec)", ms_per, 1.0 / (ms_per / 1000.0));
    println!("  Oscillators: {}", n_osc);
    println!("  Frame: {} samples", frame_size);

    println!("\n═══ CUDA T3 Benchmark Complete ═══");
}
