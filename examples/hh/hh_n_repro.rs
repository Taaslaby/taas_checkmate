/// HH reproduction test: does n=1 work but n>=2 fail?
///
/// Run: cargo run --features cuda --example hh_n_repro --release
use cuda_kernels::GpuContext;

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 5.0;
const WEIGHT_SCALE: f32 = 0.01;

/// Run HH for `n_neurons` with a step current to neuron 0 only.
/// Returns (n_spikes_total, n_spikes_neuron0, v_final[]).
fn run_hh(n_neurons: usize, n_steps: usize) -> (i32, i32, Vec<f32>) {
    // Minimal ring connectome: i -> (i+1)%n. 1 synapse per neuron.
    let n = n_neurons;
    let n_syn = n; // ring: neuron i -> i+1
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = vec![0i32; n_syn];
    let mut values = vec![1.0f32; n_syn];
    for i in 0..n {
        row_ptr[i + 1] = (i + 1) as i32;
        col_indices[i] = ((i + 1) % n) as i32;
    }

    let ctx = GpuContext::new(n, n_syn).unwrap();
    ctx.upload_connectome(&row_ptr, &col_indices, &values, &vec![1.0; n], &vec![0.0; n]).unwrap();

    let ng = 4;
    let v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * ng];
    for i in 0..n { gates[i*ng..i*ng+4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]); }
    let Ii = vec![0.0f32; n];

    ctx.hh_upload_state(&v_init, &gates, &Ii,
        &vec![0.3f32; n], &vec![-65.0f32; n], &vec![1.0f32; n], ng).unwrap();
    ctx.hh_upload_channels(&vec![50.0, -77.0, 120.0], &vec![120.0, 36.0, 0.5],
        &vec![3u32, 1, 4, 0], &vec![2, 1, 1], ng, 3).unwrap();

    // Inject strong current into neuron 0 only
    let mut external_I = vec![0.0f32; n];
    external_I[0] = 20.0;

    let mut spike_buf = vec![0i32; n_steps * n];
    let mut host_v = vec![0.0f32; n];
    let mut host_isyn = vec![0.0f32; n];

    ctx.batch_hh_steps(&external_I, &mut spike_buf, &mut host_v, &mut host_isyn,
        n_steps, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();

    let total: i32 = spike_buf.iter().sum();
    let n0: i32 = (0..n_steps).map(|t| spike_buf[t * n]).sum();
    (total, n0, host_v)
}

fn main() {
    println!("╔════ HH n-repro: does n>=2 work? ════╗");
    println!("  Inject 20 µA into neuron 0, 1000 steps (10ms), expect neuron 0 to spike.\n");

    for &n in &[1usize, 2, 4, 8] {
        let (total, n0, v) = run_hh(n, 1000);
        let has_nan = v.iter().any(|x| x.is_nan());
        let has_inf = v.iter().any(|x| x.is_infinite());
        let v_str = v.iter().take(4).map(|x| format!("{x:7.2}")).collect::<Vec<_>>().join(" ");
        println!("n={n:>2} | total_spikes={total:>5} | neuron0_spikes={n0:>4} | v_final=[{v_str}...] | NaN={has_nan} Inf={has_inf}");
    }

    println!("\nExpected: neuron 0 should spike (n0 > 0) at ALL n. If n>=2 gives 0 spikes or NaN, bug reproduced.");
}
