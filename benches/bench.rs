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
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::izhikevich_step(
            &mut v1, &mut u1, &mut tr1,
            &a1, &b1, &I1,
            0.5, 0.01, &mut sp1, &c1, &d1,
            &mut refr1, 3.0,
        );
        black_box(());
    }));
    let (mut v8, mut u8, mut tr8) = (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8]);
    let (a8, b8) = (vec![0.02f32; 8], vec![0.2f32; 8]);
    let (mut I8, mut sp8) = (vec![10.0f32; 8], vec![0i32; 8]);
    let (c8, d8) = (vec![-65.0f32; 8], vec![8.0f32; 8]);
    let mut refr8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::izhikevich_step(
            &mut v8, &mut u8, &mut tr8,
            &a8, &b8, &I8,
            0.5, 0.01, &mut sp8, &c8, &d8,
            &mut refr8, 3.0,
        );
        black_box(());
    }));
    g.finish();
}

fn bench_lif(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif");
    let mut refr1_l = vec![0.0f32; 1];
    let (mut v1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, -65.0, -50.0, -70.0]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::lif_step(&mut v1, &mut I1, &mut sp1, &p1, &mut refr1_l, 3.0);
        black_box(());
    }));
    let mut refr8_l = vec![0.0f32; 8];
    let (mut v8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, -65.0, -50.0, -70.0]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::lif_step(&mut v8, &mut I8, &mut sp8, &p8, &mut refr8_l, 3.0);
        black_box(());
    }));
    g.finish();
}

fn bench_lif_at(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif_at");
    let mut refr1_la = vec![0.0f32; 1];
    let (mut v1, mut th1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![0.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::lif_at_step(&mut v1, &mut th1, &mut I1, &mut sp1, &p1, &mut refr1_la, 3.0);
        black_box(());
    }));
    let mut refr8_la = vec![0.0f32; 8];
    let (mut v8, mut th8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![0.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::lif_at_step(&mut v8, &mut th8, &mut I8, &mut sp8, &p8, &mut refr8_la, 3.0);
        black_box(());
    }));
    g.finish();
}

fn bench_adex(c: &mut Criterion) {
    let mut g = c.benchmark_group("adex");
    let mut refr1_a = vec![0.0f32; 1];
    let (mut v1, mut u1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![0.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::adex_step(&mut v1, &mut u1, &mut I1, &mut sp1, &p1, &mut refr1_a, 3.0);
        black_box(());
    }));
    let mut refr8_a = vec![0.0f32; 8];
    let (mut v8, mut u8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![0.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::adex_step(&mut v8, &mut u8, &mut I8, &mut sp8, &p8, &mut refr8_a, 3.0);
        black_box(());
    }));
    g.finish();
}

fn bench_saw(c: &mut Criterion) {
    let mut g = c.benchmark_group("saw");
    let mut refr1_s = vec![0.0f32; 1];
    let (mut v1, mut u1, mut tr1, a1, b1, I1, mut sp1) =
        (vec![-65.0f32], vec![-13.0f32], vec![0.0f32], vec![0.02f32], vec![0.2f32], vec![10.0f32], vec![0i32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::saw_step(&mut v1, &mut u1, &mut tr1, &a1, &b1, &I1, 0.5, 0.01, 0.1, &mut sp1, &mut refr1_s, 3.0);
        black_box(());
    }));
    let mut refr8_s = vec![0.0f32; 8];
    let (mut v8, mut u8, mut tr8, a8, b8, I8, mut sp8) =
        (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8], vec![0.02f32; 8], vec![0.2f32; 8], vec![10.0f32; 8], vec![0i32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::saw_step(&mut v8, &mut u8, &mut tr8, &a8, &b8, &I8, 0.5, 0.01, 0.1, &mut sp8, &mut refr8_s, 3.0);
        black_box(());
    }));
    g.finish();
}

fn bench_psi(c: &mut Criterion) {
    let mut g = c.benchmark_group("psi_wave");
    let (mut v1, mut u1, mut hr1, mut hi1, mut ie1, mut sp1) =
        (vec![1.0f32], vec![0.0f32], vec![0.5f32], vec![0.5f32], vec![0.1f32], vec![0.5f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::psi_wave_step(&mut v1, &mut u1, &mut hr1, &mut hi1, &mut ie1, &mut sp1, 0.5, 1.0);
        black_box(());
    }));
    let (mut v8, mut u8, mut hr8, mut hi8, mut ie8, mut sp8) =
        (vec![1.0f32; 8], vec![0.0f32; 8], vec![0.5f32; 8], vec![0.5f32; 8], vec![0.1f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::psi_wave_step(&mut v8, &mut u8, &mut hr8, &mut hi8, &mut ie8, &mut sp8, 0.5, 1.0);
        black_box(());
    }));
    g.finish();
}

fn bench_stdp(c: &mut Criterion) {
    let mut g = c.benchmark_group("stdp");
    let (mut w1, pre1, post1) = (vec![0.5f32], vec![0.3f32], vec![0.1f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::stdp_update(&mut w1, &pre1, &post1, 0.01, 0.005, 0.0, 2.0);
        black_box(());
    }));
    let (mut w8, pre8, post8) = (vec![0.5f32; 8], vec![0.3f32; 8], vec![0.1f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::stdp_update(&mut w8, &pre8, &post8, 0.01, 0.005, 0.0, 2.0);
        black_box(());
    }));
    g.finish();
}

fn bench_kuramoto(c: &mut Criterion) {
    let mut g = c.benchmark_group("kuramoto");
    let (mut p1, o1, c1_) = (vec![1.0f32], vec![2.0f32], vec![0.5f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::kuramoto_update(&mut p1, &o1, &c1_, 0.5);
        black_box(());
    }));
    let (mut p8, o8, c8_) = (vec![1.0f32; 8], vec![2.0f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::kuramoto_update(&mut p8, &o8, &c8_, 0.5);
        black_box(());
    }));
    g.finish();
}

fn bench_t3(c: &mut Criterion) {
    let mut g = c.benchmark_group("t3");
    let (mut ph1, mut a1, o1, c1_, d1) = (vec![1.0f32], vec![0.5f32], vec![2.0f32], vec![0.5f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::t3_update(&mut ph1, &mut a1, &o1, &c1_, &d1, 0.5, 0.01, 1.0, 10.0);
        black_box(());
    }));
    let (mut ph8, mut a8, o8, c8_, d8) = (vec![1.0f32; 8], vec![0.5f32; 8], vec![2.0f32; 8], vec![0.5f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::t3_update(&mut ph8, &mut a8, &o8, &c8_, &d8, 0.5, 0.01, 1.0, 10.0);
        black_box(());
    }));
    g.finish();
}

fn bench_clamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("safety_clamp");
    let mut d1 = vec![50.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| { kernels::safety_clamp(&mut d1, -10.0, 10.0); black_box(()); }));
    let mut d8 = vec![50.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| { kernels::safety_clamp(&mut d8, -10.0, 10.0); black_box(()); }));
    g.finish();
}

fn bench_hebbian(c: &mut Criterion) {
    let mut g = c.benchmark_group("hebbian");
    let (mut w1, cd1) = (vec![0.5f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::hebbian_update(&mut w1, &cd1, 0.01, 0.001, 0.5);
        black_box(());
    }));
    let (mut w8, cd8) = (vec![0.5f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::hebbian_update(&mut w8, &cd8, 0.01, 0.001, 0.5);
        black_box(());
    }));
    g.finish();
}

fn bench_nlms(c: &mut Criterion) {
    let mut g = c.benchmark_group("nlms");
    let (mut w1, mut bias1, s1) = (vec![0.5f32], vec![0.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::nlms_train(&mut w1, &mut bias1, &s1, 1.0, 0.01, 1e-8);
        black_box(());
    }));
    let (mut w8, mut bias8, s8) = (vec![0.5f32; 8], vec![0.0f32], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::nlms_train(&mut w8, &mut bias8, &s8, 1.0, 0.01, 1e-8);
        black_box(());
    }));
    g.finish();
}

fn bench_kalman(c: &mut Criterion) {
    let mut g = c.benchmark_group("kalman");
    let (mut w1, mut cov1, s1) = (vec![0.5f32], vec![1.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| {
        kernels::kalman_update(&mut w1, &mut cov1, &s1, 1.0, 2.0);
        black_box(());
    }));
    let (mut w8, mut cov8, s8) = (vec![0.5f32; 8], vec![1.0f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| {
        kernels::kalman_update(&mut w8, &mut cov8, &s8, 1.0, 2.0);
        black_box(());
    }));
    g.finish();
}

fn pack1(vals: &[f32]) -> Vec<f32> { vals.to_vec() }
fn pack8(vals: &[f32]) -> Vec<f32> {
    let mut p = Vec::with_capacity(vals.len() * 8);
    for &v in vals { for _ in 0..8 { p.push(v); } }
    p
}

criterion_group!(benches, bench_izhikevich, bench_lif, bench_lif_at, bench_adex, bench_saw, bench_psi, bench_stdp, bench_kuramoto, bench_t3, bench_clamp, bench_hebbian, bench_nlms, bench_kalman);
criterion_main!(benches);
