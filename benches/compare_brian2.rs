//! Generate LIF traces for Brian2 comparison.
//! Run: cargo run --release --example compare_lif
use std::fs::File;
use std::io::Write;

fn main() {
    let n = 1;
    let dt = 0.1e-3;
    let n_steps = 500;
    let mut v = vec![-65.0f32; n];
    let mut refr = vec![0.0f32; n];
    let tau_m = 10.0e-3;
    let v_rest = -65.0;
    let v_thresh = -50.0;
    let v_reset = -65.0;
    let tau_ref = 2.0e-3;
    let i_ext = 2.0; // nA

    let mut trace = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        trace.push(v[0]);

        if refr[0] > 0.0 {
            refr[0] -= dt;
            continue;
        }

        // Exact exponential Euler
        let decay = (-dt / tau_m).exp();
        v[0] = v_rest + (v[0] - v_rest) * decay + i_ext * tau_m * (1.0 - decay);

        if v[0] >= v_thresh {
            v[0] = v_reset;
            refr[0] = tau_ref;
        }
    }

    let mut f = File::create("/tmp/lif_trace_rust.txt").unwrap();
    for (i, val) in trace.iter().enumerate() {
        writeln!(f, "{},{}", i, val).unwrap();
    }
    eprintln!("Rust trace written to /tmp/lif_trace_rust.txt");
}
