/// CUDA LIF benchmark — после ленивых аллокаций.
use std::time::Instant;
use cuda_kernels::GpuContext;

fn main() {
    let dt = 0.1e-3;
    let tau_m = 10e-3;
    let v_rest = -65e-3;
    let v_thresh = -50e-3;
    let v_reset = -65e-3;
    let n_steps = 1000;

    let ns = [100_000, 1_000_000, 10_000_000, 50_000_000, 80_000_000, 100_000_000];

    println!("{:>12}  {:>12}  {:>18}  {:>10}",
             "Neurons", "Time (s)", "Throughput (n/s)", "us/step");
    println!("{}", "-".repeat(56));

    for &n in &ns {
        let ctx = match GpuContext::new(n, 0) {
            Ok(c) => c,
            Err(e) => { println!("{n:>12}  OOM: {e}"); continue; }
        };

        ctx.upload_lif_params(dt, &vec![tau_m; n], &vec![v_rest; n],
                              &vec![v_thresh; n], &vec![v_reset; n]).unwrap();
        ctx.upload_I(&vec![2e-9; n]).unwrap();

        ctx.lif_step_async().unwrap();
        ctx.sync().unwrap();

        let t0 = Instant::now();
        for _ in 0..n_steps {
            ctx.lif_step_async().unwrap();
        }
        ctx.sync().unwrap();
        let t = t0.elapsed().as_secs_f64();

        let thr = n as f64 * n_steps as f64 / t;
        let us = t / n_steps as f64 * 1e6;

        println!("{n:>12}  {t:>12.6}  {thr:>18.1e}  {us:>10.1}");
    }
}
