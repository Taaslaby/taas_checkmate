//! AOT biophysics wrapper parity: CPU pillars kernels versus `pillars-cuda`.
//!
//! These tests exercise the host-slice ABI, not only CUDA symbol linkage. They
//! require the configured CUDA device, like the other pillars-cuda parity tests.

use pillars::core::population::run_neuron;
use pillars::zoo::neurons::MorrisLecarModel;
use pillars::zoo::synapses::{nmda_mg_block_step, tsodyks_markram_step};
use pillars_cuda::gpu::{step_morris_lecar, step_nmda_mg_block, step_tsodyks_markram};

const N: usize = 257; // exercises a non-full final CUDA block

#[test]
fn morris_lecar_wrapper_matches_pillars() {
    let mut params = vec![0.0f32; 14 * N];
    let values = [
        1.0, 0.05, 2.0, -60.0, 4.4, 120.0, 8.0, -84.0, -1.2, 18.0, 2.0, 30.0, 0.04, 0.0,
    ];
    for (c, &value) in values.iter().enumerate() {
        params[c * N..(c + 1) * N].fill(value);
    }

    let input = vec![100.0f32; N];
    let mut cpu_v = vec![-60.0f32; N];
    let mut cpu_w = vec![0.0f32; N];
    let mut cpu_below = vec![1.0f32; N];
    let mut cpu_dummy = vec![0.0f32; N];
    let mut cpu_spikes = vec![0i32; N];
    unsafe {
        run_neuron::<MorrisLecarModel>(
            cpu_v.as_mut_ptr(),
            cpu_w.as_mut_ptr(),
            cpu_below.as_mut_ptr(),
            cpu_dummy.as_mut_ptr(),
            cpu_dummy.as_mut_ptr(),
            cpu_dummy.as_mut_ptr(),
            input.as_ptr(),
            cpu_spikes.as_mut_ptr(),
            params.as_ptr(),
            N,
        );
    }

    let mut gpu_v = vec![-60.0f32; N];
    let mut gpu_w = vec![0.0f32; N];
    let mut gpu_below = vec![1.0f32; N];
    let mut gpu_spikes = vec![0i32; N];
    step_morris_lecar(
        &mut gpu_v,
        &mut gpu_w,
        &mut gpu_below,
        &input,
        &mut gpu_spikes,
        &params,
        1.0,
    )
    .expect("Morris-Lecar CUDA wrapper");

    for i in 0..N {
        assert_eq!(cpu_spikes[i], gpu_spikes[i], "Morris spike {i}");
        assert!((cpu_v[i] - gpu_v[i]).abs() < 5e-3, "Morris v {i}");
        assert!((cpu_w[i] - gpu_w[i]).abs() < 5e-4, "Morris w {i}");
        assert_eq!(
            cpu_below[i].to_bits(),
            gpu_below[i].to_bits(),
            "Morris edge state {i}"
        );
    }
}

#[test]
fn nmda_wrapper_matches_pillars() {
    let v: Vec<f32> = (0..N).map(|i| -65.0 + (i % 32) as f32 * 3.0).collect();
    let g_act: Vec<f32> = (0..N).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let mut params = vec![0.0f32; 4 * N];
    for i in 0..N {
        params[4 * i] = g_act[i];
        params[4 * i + 1] = 0.0;
        params[4 * i + 2] = 1.0;
    }

    let mut cpu = vec![0.25f32; N];
    unsafe {
        nmda_mg_block_step(
            v.as_ptr(),
            g_act.as_ptr(),
            cpu.as_mut_ptr(),
            params.as_ptr(),
            N,
        );
    }
    let mut gpu = vec![0.25f32; N];
    step_nmda_mg_block(&v, &g_act, &mut gpu, &params).expect("NMDA CUDA wrapper");

    for i in 0..N {
        assert!(
            (cpu[i] - gpu[i]).abs() < 2e-5,
            "NMDA current {i}: {} vs {}",
            cpu[i],
            gpu[i]
        );
    }
}

#[test]
fn tsodyks_markram_wrapper_matches_pillars() {
    let mut x: Vec<f32> = (0..N).map(|i| 0.6 + (i % 5) as f32 * 0.05).collect();
    let mut u: Vec<f32> = (0..N).map(|i| 0.1 + (i % 4) as f32 * 0.04).collect();
    let spikes: Vec<i32> = (0..N).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let mut params = vec![0.0f32; 4 * N];
    for i in 0..N {
        params[4 * i] = 0.2;
        params[4 * i + 1] = 100.0;
        params[4 * i + 2] = 50.0;
        params[4 * i + 3] = 0.8;
    }

    let mut cpu_x = x.clone();
    let mut cpu_u = u.clone();
    let mut cpu_w = vec![0.0f32; N];
    unsafe {
        tsodyks_markram_step(
            cpu_x.as_mut_ptr(),
            cpu_u.as_mut_ptr(),
            spikes.as_ptr(),
            cpu_w.as_mut_ptr(),
            params.as_ptr(),
            N,
            1.0,
        );
    }

    let mut gpu_x = std::mem::take(&mut x);
    let mut gpu_u = std::mem::take(&mut u);
    let mut gpu_w = vec![0.0f32; N];
    step_tsodyks_markram(&mut gpu_x, &mut gpu_u, &spikes, &mut gpu_w, &params, 1.0)
        .expect("Tsodyks-Markram CUDA wrapper");

    for i in 0..N {
        assert!((cpu_x[i] - gpu_x[i]).abs() < 2e-5, "TM x {i}");
        assert!((cpu_u[i] - gpu_u[i]).abs() < 2e-5, "TM u {i}");
        assert!((cpu_w[i] - gpu_w[i]).abs() < 2e-5, "TM weight {i}");
    }
}
