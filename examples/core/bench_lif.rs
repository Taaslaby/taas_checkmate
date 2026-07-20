/// LIF throughput benchmark: Rust SIMD vs conceptual peak.
use std::time::Instant;
use taas_kernels::lif_step;

fn main() {
    let dt = 0.1e-3;
    let dur = 1.0;
    let n_steps = (dur / dt) as usize;
    let tau_m = 10e-3;
    let v_rest = -65e-3;
    let v_thresh = -50e-3;
    let v_reset = -65e-3;
    let tau_ref = 2e-3;

    // Neuron counts to test
    let ns = [1usize, 10, 100, 1000, 10000, 100000, 1_000_000];

    println!("{:>10}  {:>16}  {:>18}  {:>12}", "Neurons", "Rust SIMD (s)", "Throughput (n·step/s)", "ns/step");
    println!("{}", "-".repeat(65));

    for &n in &ns {
        let n_pad = (n + 7) & !7;
        let mut v = vec![v_rest; n_pad];
        let mut ii = vec![2e-9; n_pad];  // 2 nA
        let mut spikes = vec![0i32; n_pad];
        // params: [tau_m, v_rest, v_thresh, v_reset, trace_decay]
        let mut params = Vec::with_capacity(n_pad * 5);
        for _ in 0..n_pad {
            params.push(tau_m);
            params.push(v_rest);
            params.push(v_thresh);
            params.push(v_reset);
            params.push(0.0); // trace_decay (unused)
        }
        let mut refractory = vec![0.0; n_pad];

        let t0 = Instant::now();
        for _ in 0..n_steps {
            lif_step(&mut v, &mut ii, &mut spikes, &params, &mut refractory, tau_ref, None);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let throughput = n as f64 * n_steps as f64 / elapsed;
        let ns_per_step = elapsed / n_steps as f64 * 1e9;

        println!("{:>10}  {:>16.4e}  {:>18.1e}  {:>10.0}", n, elapsed, throughput, ns_per_step);
    }
}
