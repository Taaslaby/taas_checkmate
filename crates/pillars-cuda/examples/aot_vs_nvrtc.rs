//! # AOT (`kernels/*.cu`, nvcc) vs NVRTC (`#[dual_neuron]`) — LIF head-to-head
//!
//! Same model (LIF), same size, same workload: the question behind the
//! pillars-cuda dual-layer design is whether the handwritten AOT kernels are
//! actually faster than the macro-generated NVRTC path, or whether the
//! duplicate CUDA math is pure debt.
//!
//! Both paths run `STEPS` steps on `N` neurons with state/params resident in
//! VRAM (persistent path); spikes are downloaded once at the end. The
//! per-step loop is what's timed.
//!
//! ```text
//! cargo run -p pillars-cuda --example aot_vs_nvrtc --release
//! ```
//!
//! Requires a CUDA device at runtime (like the other pillars-cuda examples).

use std::time::Instant;

use pillars_cuda::gpu::{DevicePtr, GpuContext};
use pillars_cuda::{CompiledModel, LIF_CUDA_SOURCE};

const N: usize = 262_144; // 256k neurons
const DT: f32 = 0.1; // ms
const STEPS: usize = 1_000;

type LifParams = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn make_inputs() -> LifParams {
    // LIF params, same values for both paths.
    let tau_m = vec![10.0f32; N];
    let v_rest = vec![-70.0f32; N];
    let v_thresh = vec![-54.0f32; N];
    let v_reset = vec![-70.0f32; N];
    let i_in: Vec<f32> = (0..N).map(|i| if i % 2 == 0 { 30.0 } else { 0.0 }).collect();
    (tau_m, v_rest, v_thresh, v_reset, i_in)
}

fn main() {
    println!("═══ LIF {N}: AOT (nvcc .cu) vs NVRTC (dual_neuron) ═══");
    println!("  steps = {STEPS}, dt = {DT} ms\n");

    let (tau_m, v_rest, v_thresh, v_reset, i_in) = make_inputs();

    // ── AOT path: GpuContext (kernels/izhikevich.cu + gpu_context.cu) ─────
    let ctx = GpuContext::new(N, 0).expect("GpuContext::new failed");
    ctx.upload_lif_params(DT, &tau_m, &v_rest, &v_thresh, &v_reset)
        .expect("upload_lif_params");
    let v0 = vec![-70.0f32; N];
    ctx.upload_v_only(&v0).expect("upload_v");

    let mut dev_spikes = DevicePtr::<i32>::new(STEPS * N).expect("spike buf alloc");
    let mut host_spikes = vec![0i32; STEPS * N];
    let mut host_v = vec![0.0f32; N];

    let t0 = Instant::now();
    ctx        .batch_lif_steps(&i_in, dev_spikes.as_mut_ptr(), &mut host_spikes, &mut host_v, STEPS)
        .expect("batch lif");
    let aot_ms = t0.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── NVRTC path: CompiledModel<LIFModel> (dual_neuron → NVRTC → PTX) ───
    let model = CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N)
        .expect("NVRTC compile failed");
    // states: [v, refr] columns; params: [dt, inv_tau_m, v_rest, v_thresh, v_reset, …]
    let mut states = vec![0.0f32; 2 * N];
    states[..N].fill(-70.0); // v
    // states[N..] refr = 0 already
    let mut params = vec![0.0f32; 6 * N];
    let pvals = [DT, 0.1, -70.0, -54.0, -70.0, 2.0];
    for (c, &v) in pvals.iter().enumerate() {
        params[c * N..(c + 1) * N].fill(v);
    }
    model.upload(&states, &params, &i_in).expect("upload");

    let t1 = Instant::now();
    for _ in 0..STEPS {
        model.step_device(DT).expect("persistent step");
    }
    let nvrtc_ms = t1.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // Final spike dump of the last step (validates the NVRTC path produced
    // activity — compact_spikes only sees the most recent step).
    let mut last = vec![0i32; N];
    model.download_spikes(&mut last).expect("dl last spikes");
    let nvrtc_total: i64 = last.iter().map(|&s| s as i64).sum();
    let mut v_final = vec![0.0f32; 2 * N];
    model.download_states(&mut v_final).expect("dl states");
    let v_min = v_final[..N].iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = v_final[..N].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  NVRTC final v in [{v_min:.4}, {v_max:.4}] (rest -70.0)");

    let aot_total: i64 = host_spikes.iter().map(|&s| s as i64).sum();

    println!("  AOT  (GpuContext batch_lif):  {aot_ms:8.3} ms/step  (spikes {aot_total})");
    println!("  NVRTC(CompiledModel):         {nvrtc_ms:8.3} ms/step  (last-step spikes {nvrtc_total})");
    println!();
    let ratio = nvrtc_ms / aot_ms.max(1e-9);
    let winner = if ratio < 1.0 { "NVRTC faster" } else { "AOT faster" };
    println!("  ratio nvrtc/aot = {ratio:.3} → {winner}");

    println!();
    println!(
        "  note: AOT batch_lif also does device-side spike counting + I reset +"
    );
    println!(
        "  note: CSR propagate (0 synapses here) per step — NVRTC path is the"
    );
    println!(
        "  note: raw neuron step. Parity of the math is a separate test."
    );
    println!("\n(parity across AOT/NVRTC is a separate test; timing is the target here)");
}