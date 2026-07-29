#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use taas_checkmate::kernels;

fn bench_izhikevich(c: &mut Criterion) {
    let mut g = c.benchmark_group("izhikevich");
    let (mut v1, mut u1, mut tr1) = (vec![-65.0f32], vec![-13.0f32], vec![0.0f32]);
    let (a1, b1) = (vec![0.02f32], vec![0.2f32]);
    let (mut I1, mut sp1) = (vec![10.0f32], vec![0i32]);
    let (c1, d1) = (vec![-65.0f32], vec![8.0f32]);
    let mut refr1 = vec![0.0f32; 1];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::izhikevich_step(
                &mut v1, &mut u1, &mut tr1, &a1, &b1, &mut I1, 0.5, 0.01, &mut sp1, &c1, &d1,
                &mut refr1, 3.0, None,
            );
            black_box(());
        })
    });
    let (mut v8, mut u8, mut tr8) = (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8]);
    let (a8, b8) = (vec![0.02f32; 8], vec![0.2f32; 8]);
    let (mut I8, mut sp8) = (vec![10.0f32; 8], vec![0i32; 8]);
    let (c8, d8) = (vec![-65.0f32; 8], vec![8.0f32; 8]);
    let mut refr8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::izhikevich_step(
                &mut v8, &mut u8, &mut tr8, &a8, &b8, &mut I8, 0.5, 0.01, &mut sp8, &c8, &d8,
                &mut refr8, 3.0, None,
            );
            black_box(());
        })
    });
    g.finish();
}

fn bench_lif(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif");
    let mut refr1_l = vec![0.0f32; 1];
    let (mut v1, mut I1, mut sp1, p1) = (
        vec![-65.0f32],
        vec![15.0f32],
        vec![0i32],
        pack1(&[0.5, 10.0, -65.0, -50.0, -70.0]),
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::lif_step(&mut v1, &mut I1, &mut sp1, &p1, &mut refr1_l, 3.0, None);
            black_box(());
        })
    });
    let mut refr8_l = vec![0.0f32; 8];
    let (mut v8, mut I8, mut sp8, p8) = (
        vec![-65.0f32; 8],
        vec![15.0f32; 8],
        vec![0i32; 8],
        pack8(&[0.5, 10.0, -65.0, -50.0, -70.0]),
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::lif_step(&mut v8, &mut I8, &mut sp8, &p8, &mut refr8_l, 3.0, None);
            black_box(());
        })
    });
    g.finish();
}

fn bench_lif_at(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif_at");
    let mut refr1_la = vec![0.0f32; 1];
    let (mut v1, mut th1, mut I1, mut sp1, p1) = (
        vec![-65.0f32],
        vec![0.0f32],
        vec![15.0f32],
        vec![0i32],
        pack1(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]),
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::lif_at_step(
                &mut v1,
                &mut th1,
                &mut I1,
                &mut sp1,
                &p1,
                &mut refr1_la,
                3.0,
                None,
            );
            black_box(());
        })
    });
    let mut refr8_la = vec![0.0f32; 8];
    let (mut v8, mut th8, mut I8, mut sp8, p8) = (
        vec![-65.0f32; 8],
        vec![0.0f32; 8],
        vec![15.0f32; 8],
        vec![0i32; 8],
        pack8(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]),
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::lif_at_step(
                &mut v8,
                &mut th8,
                &mut I8,
                &mut sp8,
                &p8,
                &mut refr8_la,
                3.0,
                None,
            );
            black_box(());
        })
    });
    g.finish();
}

fn bench_adex(c: &mut Criterion) {
    let mut g = c.benchmark_group("adex");
    let mut refr1_a = vec![0.0f32; 1];
    let (mut v1, mut u1, mut I1, mut sp1, p1) = (
        vec![-65.0f32],
        vec![0.0f32],
        vec![15.0f32],
        vec![0i32],
        pack1(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]),
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::adex_step(
                &mut v1,
                &mut u1,
                &mut I1,
                &mut sp1,
                &p1,
                &mut refr1_a,
                3.0,
                None,
            );
            black_box(());
        })
    });
    let mut refr8_a = vec![0.0f32; 8];
    let (mut v8, mut u8, mut I8, mut sp8, p8) = (
        vec![-65.0f32; 8],
        vec![0.0f32; 8],
        vec![15.0f32; 8],
        vec![0i32; 8],
        pack8(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]),
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::adex_step(
                &mut v8,
                &mut u8,
                &mut I8,
                &mut sp8,
                &p8,
                &mut refr8_a,
                3.0,
                None,
            );
            black_box(());
        })
    });
    g.finish();
}

fn bench_saw(c: &mut Criterion) {
    let mut g = c.benchmark_group("saw");
    let mut refr1_s = vec![0.0f32; 1];
    let (mut v1, mut u1, mut tr1, a1, b1, I1, mut sp1) = (
        vec![-65.0f32],
        vec![-13.0f32],
        vec![0.0f32],
        vec![0.02f32],
        vec![0.2f32],
        vec![10.0f32],
        vec![0i32],
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::saw_step(
                &mut v1,
                &mut u1,
                &mut tr1,
                &a1,
                &b1,
                &I1,
                0.5,
                0.01,
                0.1,
                &mut sp1,
                &mut refr1_s,
                3.0,
                None,
            );
            black_box(());
        })
    });
    let mut refr8_s = vec![0.0f32; 8];
    let (mut v8, mut u8, mut tr8, a8, b8, I8, mut sp8) = (
        vec![-65.0f32; 8],
        vec![-13.0f32; 8],
        vec![0.0f32; 8],
        vec![0.02f32; 8],
        vec![0.2f32; 8],
        vec![10.0f32; 8],
        vec![0i32; 8],
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::saw_step(
                &mut v8,
                &mut u8,
                &mut tr8,
                &a8,
                &b8,
                &I8,
                0.5,
                0.01,
                0.1,
                &mut sp8,
                &mut refr8_s,
                3.0,
                None,
            );
            black_box(());
        })
    });
    g.finish();
}


fn bench_stdp(c: &mut Criterion) {
    let mut g = c.benchmark_group("stdp");
    let (mut w1, pre1, post1) = (vec![0.5f32], vec![0.3f32], vec![0.1f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::stdp_update(&mut w1, &pre1, &post1, 0.01, 0.005, 0.0, 2.0);
            black_box(());
        })
    });
    let (mut w8, pre8, post8) = (vec![0.5f32; 8], vec![0.3f32; 8], vec![0.1f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::stdp_update(&mut w8, &pre8, &post8, 0.01, 0.005, 0.0, 2.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_kuramoto(c: &mut Criterion) {
    let mut g = c.benchmark_group("kuramoto");
    let (mut p1, o1, c1_) = (vec![1.0f32], vec![2.0f32], vec![0.5f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::kuramoto_update(&mut p1, &o1, &c1_, 0.5);
            black_box(());
        })
    });
    let (mut p8, o8, c8_) = (vec![1.0f32; 8], vec![2.0f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::kuramoto_update(&mut p8, &o8, &c8_, 0.5);
            black_box(());
        })
    });
    g.finish();
}

fn bench_t3(c: &mut Criterion) {
    let mut g = c.benchmark_group("t3");
    let (mut ph1, mut a1, o1, c1_, d1) = (
        vec![1.0f32],
        vec![0.5f32],
        vec![2.0f32],
        vec![0.5f32],
        vec![0.3f32],
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::t3_update(&mut ph1, &mut a1, &o1, &c1_, &d1, 0.5, 0.01, 1.0, 10.0);
            black_box(());
        })
    });
    let (mut ph8, mut a8, o8, c8_, d8) = (
        vec![1.0f32; 8],
        vec![0.5f32; 8],
        vec![2.0f32; 8],
        vec![0.5f32; 8],
        vec![0.3f32; 8],
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::t3_update(&mut ph8, &mut a8, &o8, &c8_, &d8, 0.5, 0.01, 1.0, 10.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_clamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("safety_clamp");
    let mut d1 = vec![50.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::safety_clamp(&mut d1, -10.0, 10.0);
            black_box(());
        })
    });
    let mut d8 = vec![50.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::safety_clamp(&mut d8, -10.0, 10.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_hebbian(c: &mut Criterion) {
    let mut g = c.benchmark_group("hebbian");
    let (mut w1, cd1) = (vec![0.5f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::hebbian_update(&mut w1, &cd1, 0.01, 0.001, 0.5);
            black_box(());
        })
    });
    let (mut w8, cd8) = (vec![0.5f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::hebbian_update(&mut w8, &cd8, 0.01, 0.001, 0.5);
            black_box(());
        })
    });
    g.finish();
}

fn bench_nlms(c: &mut Criterion) {
    let mut g = c.benchmark_group("nlms");
    let (mut w1, mut bias1, s1) = (vec![0.5f32], vec![0.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::nlms_train(&mut w1, &mut bias1, &s1, 1.0, 0.01, 1e-8);
            black_box(());
        })
    });
    let (mut w8, mut bias8, s8) = (vec![0.5f32; 8], vec![0.0f32], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::nlms_train(&mut w8, &mut bias8, &s8, 1.0, 0.01, 1e-8);
            black_box(());
        })
    });
    g.finish();
}

fn bench_kalman(c: &mut Criterion) {
    let mut g = c.benchmark_group("kalman");
    let (mut w1, mut cov1, s1) = (vec![0.5f32], vec![1.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            kernels::kalman_update(&mut w1, &mut cov1, &s1, 1.0, 2.0);
            black_box(());
        })
    });
    let (mut w8, mut cov8, s8) = (vec![0.5f32; 8], vec![1.0f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::kalman_update(&mut w8, &mut cov8, &s8, 1.0, 2.0);
            black_box(());
        })
    });
    g.finish();
}

fn bench_hh_poly(c: &mut Criterion) {
    use taas_checkmate::kernels::neuron::hh::{build_channel_meta, standard_channels};
    let channels = standard_channels();
    let meta = build_channel_meta(&channels);
    let n_gates: usize = meta.iter().map(|ch| ch.n_gates).sum();
    // Single neuron
    let (mut v1, mut g1, I1, gl1, El1, cm1) = (
        vec![-65.0f32],
        vec![0.5f32; n_gates],
        vec![10.0f32],
        vec![0.3f32],
        vec![-65.0f32],
        vec![1.0f32],
    );
    let gmax1 = vec![120.0f32, 36.0]; // per-channel g_max for 1 neuron
    c.benchmark_group("hh_poly")
        .throughput(Throughput::Elements(1))
        .bench_function("scalar_n1", |b| {
            b.iter(|| {
                taas_checkmate::kernels::hh_poly_step(
                    &mut v1, &mut g1, &I1, &gl1, &El1, &cm1, &meta, &gmax1, 0.01, 6.3, None,
                );
                black_box(());
            })
        });
    let (mut v8, mut g8, I8, gl8, El8, cm8) = (
        vec![-65.0f32; 8],
        vec![0.5f32; 8 * n_gates],
        vec![10.0f32; 8],
        vec![0.3f32; 8],
        vec![-65.0f32; 8],
        vec![1.0f32; 8],
    );
    let gmax8 = vec![120.0f32; 16]; // 2 channels × 8 neurons
    c.benchmark_group("hh_poly")
        .throughput(Throughput::Elements(8))
        .bench_function("simd_n8", |b| {
            b.iter(|| {
                taas_checkmate::kernels::hh_poly_step(
                    &mut v8, &mut g8, &I8, &gl8, &El8, &cm8, &meta, &gmax8, 0.01, 6.3, None,
                );
                black_box(());
            })
        });
    let (mut v1024, mut g1024, I1024, gl1024, El1024, cm1024) = (
        vec![-65.0f32; 1024],
        vec![0.5f32; 1024 * n_gates],
        vec![10.0f32; 1024],
        vec![0.3f32; 1024],
        vec![-65.0f32; 1024],
        vec![1.0f32; 1024],
    );
    let gmax1024 = vec![120.0f32; 2048]; // 2 channels × 1024 neurons
    c.benchmark_group("hh_poly")
        .throughput(Throughput::Elements(1024))
        .bench_function("simd_n1024", |b| {
            b.iter(|| {
                taas_checkmate::kernels::hh_poly_step(
                    &mut v1024, &mut g1024, &I1024, &gl1024, &El1024, &cm1024, &meta, &gmax1024,
                    0.01, 6.3, None,
                );
                black_box(());
            })
        });
}

fn bench_hh_lut(c: &mut Criterion) {
    use taas_checkmate::kernels::neuron::hh::{build_luts, standard_channels};
    let channels = standard_channels();
    let luts = build_luts(&channels, 0.01, -100.0, 50.0, 1024);
    let n_gates: usize = channels.iter().map(|ch| ch.gates.len()).sum();
    // Build flat arrays
    let n_bins = 1024;
    let v_min = -100.0;
    let v_max = 50.0;
    let mut m_inf_luts_ptrs: Vec<*const f32> = Vec::new();
    let mut decay_luts_ptrs: Vec<*const f32> = Vec::new();
    let mut e_rev_arr: Vec<f32> = Vec::new();
    let mut g_max_arr: Vec<f32> = Vec::new();
    let mut powers: Vec<u32> = Vec::new();
    let mut n_gates_per_ch: Vec<usize> = Vec::new();
    for (ch_idx, ch) in channels.iter().enumerate() {
        for (gate_idx, gate) in ch.gates.iter().enumerate() {
            m_inf_luts_ptrs.push(luts[ch_idx][gate_idx].m_inf.as_ptr());
            decay_luts_ptrs.push(luts[ch_idx][gate_idx].decay.as_ptr());
            powers.push(gate.power);
        }
        e_rev_arr.push(ch.E_rev);
        g_max_arr.push(ch.g_max);
        n_gates_per_ch.push(ch.gates.len());
    }
    // Single neuron
    let (mut v1, mut g1, I1, gl1, El1, cm1) = (
        vec![-65.0f32],
        vec![0.5f32; n_gates],
        vec![10.0f32],
        vec![0.3f32],
        vec![-65.0f32],
        vec![1.0f32],
    );
    c.benchmark_group("hh_lut")
        .throughput(Throughput::Elements(1))
        .bench_function("scalar_n1", |b| {
            b.iter(|| {
                unsafe {
                    taas_checkmate::kernels::neuron::hh::hh_lut_step(
                        v1.as_mut_ptr(),
                        g1.as_mut_ptr(),
                        I1.as_ptr(),
                        gl1.as_ptr(),
                        El1.as_ptr(),
                        cm1.as_ptr(),
                        1,
                        n_gates,
                        m_inf_luts_ptrs.as_ptr(),
                        decay_luts_ptrs.as_ptr(),
                        v_min,
                        v_max,
                        n_bins as f32,
                        e_rev_arr.as_ptr(),
                        g_max_arr.as_ptr(),
                        powers.as_ptr(),
                        n_gates_per_ch.as_ptr(),
                        channels.len(),
                        0.01,
                    );
                }
                black_box(());
            })
        });
    let (mut v8, mut g8, I8, gl8, El8, cm8) = (
        vec![-65.0f32; 8],
        vec![0.5f32; 8 * n_gates],
        vec![10.0f32; 8],
        vec![0.3f32; 8],
        vec![-65.0f32; 8],
        vec![1.0f32; 8],
    );
    c.benchmark_group("hh_lut")
        .throughput(Throughput::Elements(8))
        .bench_function("simd_n8", |b| {
            b.iter(|| {
                unsafe {
                    taas_checkmate::kernels::neuron::hh::hh_lut_step(
                        v8.as_mut_ptr(),
                        g8.as_mut_ptr(),
                        I8.as_ptr(),
                        gl8.as_ptr(),
                        El8.as_ptr(),
                        cm8.as_ptr(),
                        8,
                        n_gates,
                        m_inf_luts_ptrs.as_ptr(),
                        decay_luts_ptrs.as_ptr(),
                        v_min,
                        v_max,
                        n_bins as f32,
                        e_rev_arr.as_ptr(),
                        g_max_arr.as_ptr(),
                        powers.as_ptr(),
                        n_gates_per_ch.as_ptr(),
                        channels.len(),
                        0.01,
                    );
                }
                black_box(());
            })
        });
    let (mut v1024, mut g1024, I1024, gl1024, El1024, cm1024) = (
        vec![-65.0f32; 1024],
        vec![0.5f32; 1024 * n_gates],
        vec![10.0f32; 1024],
        vec![0.3f32; 1024],
        vec![-65.0f32; 1024],
        vec![1.0f32; 1024],
    );
    c.benchmark_group("hh_lut")
        .throughput(Throughput::Elements(1024))
        .bench_function("simd_n1024", |b| {
            b.iter(|| {
                unsafe {
                    taas_checkmate::kernels::neuron::hh::hh_lut_step(
                        v1024.as_mut_ptr(),
                        g1024.as_mut_ptr(),
                        I1024.as_ptr(),
                        gl1024.as_ptr(),
                        El1024.as_ptr(),
                        cm1024.as_ptr(),
                        1024,
                        n_gates,
                        m_inf_luts_ptrs.as_ptr(),
                        decay_luts_ptrs.as_ptr(),
                        v_min,
                        v_max,
                        n_bins as f32,
                        e_rev_arr.as_ptr(),
                        g_max_arr.as_ptr(),
                        powers.as_ptr(),
                        n_gates_per_ch.as_ptr(),
                        channels.len(),
                        0.01,
                    );
                }
                black_box(());
            })
        });
}

fn pack1(vals: &[f32]) -> Vec<f32> {
    vals.to_vec()
}
fn pack8(vals: &[f32]) -> Vec<f32> {
    let mut p = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        for _ in 0..8 {
            p.push(v);
        }
    }
    p
}

criterion_group!(
    benches,
    bench_izhikevich,
    bench_lif,
    bench_lif_at,
    bench_adex,
    bench_saw,
    bench_stdp,
    bench_kuramoto,
    bench_t3,
    bench_clamp,
    bench_hebbian,
    bench_nlms,
    bench_kalman,
    bench_hh_poly,
    bench_hh_lut
);
criterion_main!(benches);
