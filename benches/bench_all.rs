//! Comprehensive benchmarks for ALL kernels — covers gaps in bench.rs.
//!
//! Each benchmark compares scalar (n=1) vs SIMD (n=8) throughput.
//! Some kernels also include n=1024 batch size.
//!
//! Run: cargo bench --bench bench_all

#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars::kernels;
use pillars::core::lane::SimdLane;
use pillars::core::runner::run_neuron;
use pillars::core::util::{load8, load8_i32, store8};
use pillars::ops::plasticity::stp as pillars_stp;
use pillars::zoo::neurons::FHNModel;
use pillars::zoo::Model;
use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  signal kernels
// ═══════════════════════════════════════════════════════════

fn bench_sigmoid_gate(c: &mut Criterion) {
    let mut g = c.benchmark_group("sigmoid_gate");
    let input1 = vec![2.0f32];
    let mut out1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::sigmoid_gate(&mut out1, &input1, 1.0, 0.0);
            black_box(());
        })
    });
    let input8 = vec![2.0f32; 8];
    let mut out8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::sigmoid_gate(&mut out8, &input8, 1.0, 0.0);
            black_box(());
        })
    });
    let input1024 = vec![2.0f32; 1024];
    let mut out1024 = vec![0.0f32; 1024];
    g.throughput(Throughput::Elements(1024));
    g.bench_function("simd_n1024", |b| {
        b.iter(|| {
            kernels::sigmoid_gate(&mut out1024, &input1024, 1.0, 0.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_thalamic_gate(c: &mut Criterion) {
    let mut g = c.benchmark_group("thalamic_gate");
    let src1 = vec![1.0f32];
    let gate1 = vec![5.0f32];
    let mut out1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::thalamic_gate(&mut out1, &src1, &gate1, 0.0, 1.0);
            black_box(());
        })
    });
    let src8 = vec![1.0f32; 8];
    let gate8 = vec![5.0f32; 8];
    let mut out8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::thalamic_gate(&mut out8, &src8, &gate8, 0.0, 1.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_linear_decoder(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear_decoder");
    // n_in=8, n_out=2
    let states1 = vec![1.0f32; 8];
    let w1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let biases1 = vec![0.5f32; 2];
    let mut out1 = vec![0.0f32; 2];
    g.throughput(Throughput::Elements(8));
    g.bench_function("scalar_n8in", |b| {
        b.iter(|| {
            kernels::linear_decoder(&mut out1, &states1, &w1, &biases1);
            black_box(());
        })
    });
    // n_in=64, n_out=8
    let states8 = vec![1.0f32; 64];
    let w8: Vec<f32> = (0..64 * 8).map(|i| i as f32 * 0.01).collect();
    let biases8 = vec![0.5f32; 8];
    let mut out8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(64));
    g.bench_function("simd_n64in_x8out", |b| {
        b.iter(|| {
            kernels::linear_decoder(&mut out8, &states8, &w8, &biases8);
            black_box(());
        })
    });
    g.finish();
}

fn bench_phase_gate(c: &mut Criterion) {
    let mut g = c.benchmark_group("phase_gate");
    let input1 = vec![1.0f32];
    let r1 = vec![2.0f32];
    let mut out1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::phase_gate_apply(&mut out1, &input1, &r1, 4.0, 1.0);
            black_box(());
        })
    });
    let input8 = vec![1.0f32; 8];
    let r8 = vec![2.0f32; 8];
    let mut out8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::phase_gate_apply(&mut out8, &input8, &r8, 4.0, 1.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_relu_simd(c: &mut Criterion) {
    let mut g = c.benchmark_group("relu_simd");
    let mut d1 = vec![-3.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::relu_simd(&mut d1);
            black_box(());
        })
    });
    let mut d8 = vec![-3.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::relu_simd(&mut d8);
            black_box(());
        })
    });
    let mut d1024 = vec![-3.0f32; 1024];
    g.throughput(Throughput::Elements(1024));
    g.bench_function("simd_n1024", |b| {
        b.iter(|| {
            kernels::relu_simd(&mut d1024);
            black_box(());
        })
    });
    g.finish();
}

fn bench_add_simd(c: &mut Criterion) {
    let mut g = c.benchmark_group("add_simd");
    let mut a1 = vec![1.0f32];
    let b1 = vec![0.5f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::add_simd(&mut a1, &b1);
            black_box(());
        })
    });
    let mut a8 = vec![1.0f32; 8];
    let b8 = vec![0.5f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::add_simd(&mut a8, &b8);
            black_box(());
        })
    });
    g.finish();
}

fn bench_sgd_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("sgd_step");
    let mut vel1 = vec![0.0f32];
    let mut w1 = vec![0.5f32];
    let g1 = vec![0.1f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::sgd_step_simd(&mut vel1, &mut w1, &g1, 0.01, 0.9);
            black_box(());
        })
    });
    let mut vel8 = vec![0.0f32; 8];
    let mut w8 = vec![0.5f32; 8];
    let g8 = vec![0.1f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::sgd_step_simd(&mut vel8, &mut w8, &g8, 0.01, 0.9);
            black_box(());
        })
    });
    g.finish();
}

fn bench_t3_drive_encoder(c: &mut Criterion) {
    let mut g = c.benchmark_group("t3_drive_encoder");
    let mut drives1 = vec![0.0f32];
    let input1 = vec![1.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::t3_drive_encoder(&mut drives1, &input1, 10.0);
            black_box(());
        })
    });
    let mut drives8 = vec![0.0f32; 8];
    let input8 = vec![1.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::t3_drive_encoder(&mut drives8, &input8, 10.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_stochastic_encoder(c: &mut Criterion) {
    let mut g = c.benchmark_group("stochastic_encoder");
    let mut i1 = vec![5.0f32];
    let in1 = vec![3.0f32];
    let n1 = vec![0.1f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::stochastic_encoder(&mut i1, &in1, &n1, 0.5, 1.0);
            black_box(());
        })
    });
    let mut i8 = vec![5.0f32; 8];
    let in8 = vec![3.0f32; 8];
    let n8 = vec![0.1f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::stochastic_encoder(&mut i8, &in8, &n8, 0.5, 1.0);
            black_box(());
        })
    });
    g.finish();
}

// ═══════════════════════════════════════════════════════════
//  phase kernels
// ═══════════════════════════════════════════════════════════

fn bench_kuramoto_prepare(c: &mut Criterion) {
    let mut g = c.benchmark_group("kuramoto_prepare");
    let mut ph1 = vec![1.0f32];
    let mut c1 = vec![0.0f32];
    let mut s1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::phase::kuramoto_prepare(&mut ph1, &mut c1, &mut s1);
            black_box(());
        })
    });
    let mut ph8 = vec![1.0f32; 8];
    let mut c8 = vec![0.0f32; 8];
    let mut s8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::phase::kuramoto_prepare(&mut ph8, &mut c8, &mut s8);
            black_box(());
        })
    });
    g.finish();
}

// ═══════════════════════════════════════════════════════════
//  measure kernels
// ═══════════════════════════════════════════════════════════

fn bench_interference(c: &mut Criterion) {
    let mut g = c.benchmark_group("interference");
    let pa = vec![1.0f32; 8];
    let pb = vec![1.0f32; 8];
    let mut out = vec![0.0f32; 8 * 8];
    g.throughput(Throughput::Elements(64));
    g.bench_function("simd_8x8", |b| {
        b.iter(|| {
            kernels::interference(&mut out, &pa, &pb);
            black_box(());
        })
    });
    g.finish();
}

fn bench_phase_coherence(c: &mut Criterion) {
    let mut g = c.benchmark_group("phase_coherence");
    let sin_buf = vec![0.5f32; 8];
    let cos_buf = vec![0.5f32; 8];
    let amps = vec![1.0f32; 8];
    let mut out = 0.0f32;
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::phase_coherence(&mut out, &sin_buf, &cos_buf, &amps);
            black_box(());
        })
    });
    g.finish();
}

fn bench_spike_entropy(c: &mut Criterion) {
    let mut g = c.benchmark_group("spike_entropy");
    let counts = vec![10.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            let (h, hp) = kernels::spike_entropy(&counts, 80.0, 3.0);
            black_box((h, hp));
        })
    });
    g.finish();
}

// ═══════════════════════════════════════════════════════════
//  decode kernels
// ═══════════════════════════════════════════════════════════

fn bench_ring_synthesis(c: &mut Criterion) {
    let mut g = c.benchmark_group("ring_synthesis");
    let amps1 = vec![2.0f32];
    let trig1 = vec![0.5f32];
    let w1 = vec![1.0f32];
    let mut out1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::ring_synthesis(&mut out1, &amps1, &trig1, &w1);
            black_box(());
        })
    });
    let amps8 = vec![2.0f32; 8];
    let trig8 = vec![0.5f32; 8];
    let w8 = vec![1.0f32; 8];
    let mut out8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::ring_synthesis(&mut out8, &amps8, &trig8, &w8);
            black_box(());
        })
    });
    g.finish();
}

// ═══════════════════════════════════════════════════════════
//  attention kernels
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
//  neuron kernels
// ═══════════════════════════════════════════════════════════

fn bench_stp_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("stp_step");
    let mut x1 = vec![1.0f32];
    let mut u1 = vec![0.2f32];
    let pre1 = vec![1i32];
    let td1 = vec![200.0f32];
    let tf1 = vec![500.0f32];
    let u01 = vec![0.2f32];
    let a1 = vec![1.0f32];
    let mut i1 = vec![0.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            stp_run(&mut x1, &mut u1, &pre1, &td1, &tf1, &u01, &a1, &mut i1, 1.0);
            black_box(());
        })
    });
    let mut x8 = vec![1.0f32; 8];
    let mut u8 = vec![0.2f32; 8];
    let pre8 = vec![1i32; 8];
    let td8 = vec![200.0f32; 8];
    let tf8 = vec![500.0f32; 8];
    let u08 = vec![0.2f32; 8];
    let a8 = vec![1.0f32; 8];
    let mut i8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            stp_run(&mut x8, &mut u8, &pre8, &td8, &tf8, &u08, &a8, &mut i8, 1.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_dopamine_traces(c: &mut Criterion) {
    let mut g = c.benchmark_group("dopamine_traces");
    let mut tr_pre1 = vec![0.0f32];
    let mut tr_post1 = vec![0.0f32];
    let mut e1 = vec![0.0f32];
    let pre_s1 = vec![1i32];
    let post_s1 = vec![1i32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::dopamine_traces_step(
                &mut tr_pre1,
                &mut tr_post1,
                &mut e1,
                &pre_s1,
                &post_s1,
                1.0,
                20.0,
                20.0,
                200.0,
                0.01,
                0.01,
            );
            black_box(());
        })
    });
    let mut tr_pre8 = vec![0.0f32; 8];
    let mut tr_post8 = vec![0.0f32; 8];
    let mut e8 = vec![0.0f32; 8];
    let pre_s8 = vec![1i32; 8];
    let post_s8 = vec![1i32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::dopamine_traces_step(
                &mut tr_pre8,
                &mut tr_post8,
                &mut e8,
                &pre_s8,
                &post_s8,
                1.0,
                20.0,
                20.0,
                200.0,
                0.01,
                0.01,
            );
            black_box(());
        })
    });
    g.finish();
}

fn bench_fhn_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("fhn_step");
    let (mut v1, mut w1, i1, mut sp1, p1) = (
        vec![-65.0f32],
        vec![0.0f32],
        vec![10.0f32],
        vec![0i32],
        pack_fhn(&[0.5, 10.0, 0.7, 0.8]),
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            fhn_run(&mut v1, &mut w1, &i1, &mut sp1, &p1);
            black_box(());
        })
    });
    let (mut v8, mut w8, i8, mut sp8, p8) = (
        vec![-65.0f32; 8],
        vec![0.0f32; 8],
        vec![10.0f32; 8],
        vec![0i32; 8],
        pack_fhn_n(&[0.5, 10.0, 0.7, 0.8], 8),
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            fhn_run(&mut v8, &mut w8, &i8, &mut sp8, &p8);
            black_box(());
        })
    });
    g.finish();
}

fn bench_wc_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("wc_step");
    let p_wc = pack_wc(&[0.5, 10.0, 10.0, 5.0, 2.0, 3.0, 2.0, 0.5]);
    let (mut e1, mut i1, ie1, ii1, mut se1, mut si1) = (
        vec![0.3f32],
        vec![0.3f32],
        vec![1.0f32],
        vec![1.0f32],
        vec![0.0f32],
        vec![0.0f32],
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::wc_step(&mut e1, &mut i1, &ie1, &ii1, &mut se1, &mut si1, &p_wc);
            black_box(());
        })
    });
    let p_wc8 = pack_wc_n(&[0.5, 10.0, 10.0, 5.0, 2.0, 3.0, 2.0, 0.5], 8);
    let (mut e8, mut i8, ie8, ii8, mut se8, mut si8) = (
        vec![0.3f32; 8],
        vec![0.3f32; 8],
        vec![1.0f32; 8],
        vec![1.0f32; 8],
        vec![0.0f32; 8],
        vec![0.0f32; 8],
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::wc_step(&mut e8, &mut i8, &ie8, &ii8, &mut se8, &mut si8, &p_wc8);
            black_box(());
        })
    });
    g.finish();
}

fn bench_delay_buffer(c: &mut Criterion) {
    let mut g = c.benchmark_group("delay_buffer");
    let buf1 = vec![1.0f32; 8];
    let mut out1 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("circular_n8", |b| {
        b.iter(|| {
            kernels::delay_buffer_circular(&mut out1, &buf1, 3);
            black_box(());
        })
    });
    let buf_multi = vec![1.0f32; 16];
    let delays = vec![0i32, 3, 7, 11, 15];
    let mut out_multi = vec![0.0f32; 5];
    g.throughput(Throughput::Elements(5));
    g.bench_function("multi_tap_5taps", |b| {
        b.iter(|| {
            kernels::delay_buffer_multi_tap(&mut out_multi, &buf_multi, &delays);
            black_box(());
        })
    });
    g.finish();
}

// ═══════════════════════════════════════════════════════════
//  HH (Hodgkin-Huxley) — fused vs original benchmark
// ═══════════════════════════════════════════════════════════

fn bench_hh_fused_vs_original(c: &mut Criterion) {
    use pillars::kernels::hh_channels as hh;

    let channels = hh::standard_channels();
    let channel_meta = hh::build_channel_meta(&channels);
    let n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();

    let n = 1024;
    let dt = 0.01;

    let v = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * n_gates];
    let init_gates = hh::steady_state_gates(-65.0, &channels);
    for gi in 0..n_gates {
        for ni in 0..n {
            gates[gi * n + ni] = init_gates[gi];
        }
    }
    let Ii = vec![15.0f32; n];
    let g_leak = vec![0.3f32; n];
    let E_leak = vec![-65.0f32; n];
    let C_m = vec![1.0f32; n];
    let n_channels = channel_meta.len();
    // Per-neuron g_max: column-major [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
    let mut g_max_per_neuron = vec![0.0; n_channels * n];
    for (ch_i, ch) in channels.iter().enumerate() {
        for ni in 0..n {
            g_max_per_neuron[ch_i * n + ni] = ch.g_max;
        }
    }

    let mut group = c.benchmark_group("hh_poly_vs_fused");
    group.throughput(Throughput::Elements(n as u64));

    // Old: hh_poly_step + manual spike detection (v_prev copy + loop)
    group.bench_function("original_old_way", |b| {
        let mut v_copy = v.clone();
        let mut gates_copy = gates.clone();
        let mut v_prev = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];
        b.iter(|| {
            v_prev.copy_from_slice(&v_copy);
            unsafe {
                hh::hh_poly_step(
                    v_copy.as_mut_ptr(),
                    gates_copy.as_mut_ptr(),
                    Ii.as_ptr(),
                    g_leak.as_ptr(),
                    E_leak.as_ptr(),
                    C_m.as_ptr(),
                    n,
                    &channel_meta,
                    g_max_per_neuron.as_ptr(),
                    dt,
                    6.3,
                );
            }
            for i in 0..n {
                spikes[i] = if v_prev[i] >= 0.0 && v_copy[i] < 0.0 {
                    1
                } else {
                    0
                };
            }
            black_box(&spikes);
        });
    });

    // New: fused_hh_poly_step (one SIMD pass, no copy, common subexpression)
    group.bench_function("fused_new_way", |b| {
        let mut v_copy = v.clone();
        let mut gates_copy = gates.clone();
        let mut spikes = vec![0i32; n];
        b.iter(|| {
            unsafe {
                hh::fused_hh_poly_step(
                    v_copy.as_mut_ptr(),
                    gates_copy.as_mut_ptr(),
                    Ii.as_ptr(),
                    g_leak.as_ptr(),
                    E_leak.as_ptr(),
                    C_m.as_ptr(),
                    n,
                    &channel_meta,
                    g_max_per_neuron.as_ptr(),
                    dt,
                    6.3,
                    spikes.as_mut_ptr(),
                );
            }
            black_box(&spikes);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════

/// Pillars-канон (транш 3e): FHNModel, legacy 4-col → 5-col repacking.
/// params: [dt, tau_w, a, b] × n; v_thresh = 0.5 (legacy).
fn fhn_run(v: &mut [f32], w: &mut [f32], i: &[f32], sp: &mut [i32], p: &[f32]) {
    let n = v.len();
    let mut params = vec![0.0f32; n * FHNModel::N_PARAMS];
    let mut was_below = vec![0.0f32; n];
    for i in 0..n {
        params[i] = p[i];
        params[n + i] = if p[n + i] > 0.0 { 1.0 / p[n + i] } else { 0.0 };
        params[2 * n + i] = p[2 * n + i];
        params[3 * n + i] = p[3 * n + i];
        params[4 * n + i] = 0.5;
        was_below[i] = if v[i] < 0.5 { 1.0 } else { 0.0 };
    }
    unsafe {
        run_neuron::<FHNModel>(
            v.as_mut_ptr(),
            w.as_mut_ptr(),
            was_below.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            i.as_ptr(),
            sp.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
}

/// Pillars-канон (транш 3e): pillars::ops::plasticity::stp (lane-функция),
/// SIMD-цикл + scalar tail.
fn stp_run(
    x: &mut [f32],
    u: &mut [f32],
    pre: &[i32],
    td: &[f32],
    tf: &[f32],
    u0: &[f32],
    a: &[f32],
    i_out: &mut [f32],
    dt: f32,
) {
    let n = x.len();
    let n8 = n & !7;
    let dt_s = f32x8::splat(dt);
    let zero_s = f32x8::splat(0.0);
    let one_s = f32x8::splat(1.0);
    for off in (0..n8).step_by(8) {
        unsafe {
            let uu = load8(u.as_ptr(), off);
            let xx = load8(x.as_ptr(), off);
            let sp = load8_i32(pre.as_ptr(), off);
            let tdv = load8(td.as_ptr(), off).max(f32x8::splat(1e-10));
            let tfv = load8(tf.as_ptr(), off).max(f32x8::splat(1e-10));
            let u0v = load8(u0.as_ptr(), off);
            let av = load8(a.as_ptr(), off);
            let (un, xn) = pillars_stp::stp_step(uu, xx, sp, u0v, one_s / tdv, one_s / tfv, dt_s);
            let s = sp.cmp_gt(zero_s).blend(one_s, zero_s);
            let mut io = load8(i_out.as_ptr(), off);
            io += s * av * un * xn;
            store8(u.as_mut_ptr(), off, un);
            store8(x.as_mut_ptr(), off, xn);
            store8(i_out.as_mut_ptr(), off, io);
        }
    }
    for i in n8..n {
        let s = if pre[i] != 0 { 1.0f32 } else { 0.0f32 };
        let (un, xn) = pillars_stp::stp_step(
            u[i],
            x[i],
            s,
            u0[i],
            1.0 / td[i].max(1e-10),
            1.0 / tf[i].max(1e-10),
            dt,
        );
        u[i] = un;
        x[i] = xn;
        i_out[i] += s * a[i] * un * xn;
    }
}

fn pack_fhn(vals: &[f32]) -> Vec<f32> {
    let mut p = Vec::with_capacity(4);
    for &v in vals {
        p.push(v);
    }
    p
}

fn pack_fhn_n(vals: &[f32], n: usize) -> Vec<f32> {
    let mut p = Vec::with_capacity(vals.len() * n);
    for &v in vals {
        for _ in 0..n {
            p.push(v);
        }
    }
    p
}

fn pack_wc(vals: &[f32]) -> Vec<f32> {
    pack_fhn_n(vals, 1)
}

fn pack_wc_n(vals: &[f32], n: usize) -> Vec<f32> {
    pack_fhn_n(vals, n)
}

criterion_group!(
    benches_all,
    // signal
    bench_sigmoid_gate,
    bench_thalamic_gate,
    bench_linear_decoder,
    bench_phase_gate,
    bench_relu_simd,
    bench_add_simd,
    bench_sgd_step,
    bench_t3_drive_encoder,
    bench_stochastic_encoder,
    // phase
    bench_kuramoto_prepare,
    // measure
    bench_interference,
    bench_phase_coherence,
    bench_spike_entropy,
    // decode
    bench_ring_synthesis,
    // attention

    // neuron
    bench_stp_step,
    bench_dopamine_traces,
    bench_fhn_step,
    bench_wc_step,
    bench_delay_buffer,
    // hh
    bench_hh_fused_vs_original,
);
criterion_main!(benches_all);
