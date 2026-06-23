//! Benchmarks: scalar (n=1) vs SIMD (n=8) for all kernel functions.
//!
//! n=1 → forces scalar tail path
//! n=8 → processes via f32x8 SIMD
//! Throughput::Elements normalizes per-element timing.

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
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::izhikevich_step(
            black_box(v1.as_mut_ptr()), black_box(u1.as_mut_ptr()), black_box(tr1.as_mut_ptr()),
            black_box(a1.as_ptr()), black_box(b1.as_ptr()), black_box(I1.as_mut_ptr()),
            0.5, 0.01, black_box(sp1.as_mut_ptr()), black_box(c1.as_ptr()), black_box(d1.as_ptr()),
            black_box(refr1.as_mut_ptr()), 3.0, 1,
        );
    }));
    let (mut v8, mut u8, mut tr8) = (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8]);
    let (a8, b8) = (vec![0.02f32; 8], vec![0.2f32; 8]);
    let (mut I8, mut sp8) = (vec![10.0f32; 8], vec![0i32; 8]);
    let (c8, d8) = (vec![-65.0f32; 8], vec![8.0f32; 8]);
    let mut refr8 = vec![0.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::izhikevich_step(
            black_box(v8.as_mut_ptr()), black_box(u8.as_mut_ptr()), black_box(tr8.as_mut_ptr()),
            black_box(a8.as_ptr()), black_box(b8.as_ptr()), black_box(I8.as_mut_ptr()),
            0.5, 0.01, black_box(sp8.as_mut_ptr()), black_box(c8.as_ptr()), black_box(d8.as_ptr()),
            black_box(refr8.as_mut_ptr()), 3.0, 8,
        );
    }));
    g.finish();
}

fn bench_lif(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif");
    let (mut refr1_l) = (vec![0.0f32; 1]);
    let (mut v1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, -65.0, -50.0, -70.0]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::lif_step(black_box(v1.as_mut_ptr()), black_box(I1.as_mut_ptr()), black_box(sp1.as_mut_ptr()), black_box(p1.as_ptr()), black_box(refr1_l.as_mut_ptr()), 3.0, 1);
    }));
    let (mut refr8_l) = (vec![0.0f32; 8]);
    let (mut v8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, -65.0, -50.0, -70.0]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::lif_step(black_box(v8.as_mut_ptr()), black_box(I8.as_mut_ptr()), black_box(sp8.as_mut_ptr()), black_box(p8.as_ptr()), black_box(refr8_l.as_mut_ptr()), 3.0, 8);
    }));
    g.finish();
}

fn bench_lif_at(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif_at");
    let mut refr1_la = vec![0.0f32; 1];
    let (mut v1, mut th1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![0.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::lif_at_step(black_box(v1.as_mut_ptr()), black_box(th1.as_mut_ptr()), black_box(I1.as_mut_ptr()), black_box(sp1.as_mut_ptr()), black_box(p1.as_ptr()), black_box(refr1_la.as_mut_ptr()), 3.0, 1);
    }));
    let mut refr8_la = vec![0.0f32; 8];
    let (mut v8, mut th8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![0.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, -65.0, -50.0, -70.0, 20.0, 0.1]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::lif_at_step(black_box(v8.as_mut_ptr()), black_box(th8.as_mut_ptr()), black_box(I8.as_mut_ptr()), black_box(sp8.as_mut_ptr()), black_box(p8.as_ptr()), black_box(refr8_la.as_mut_ptr()), 3.0, 8);
    }));
    g.finish();
}

fn bench_adex(c: &mut Criterion) {
    let mut g = c.benchmark_group("adex");
    let mut refr1_a = vec![0.0f32; 1];
    let (mut v1, mut u1, mut I1, mut sp1, p1) = (vec![-65.0f32], vec![0.0f32], vec![15.0f32], vec![0i32], pack1(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]));
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::adex_step(black_box(v1.as_mut_ptr()), black_box(u1.as_mut_ptr()), black_box(I1.as_mut_ptr()), black_box(sp1.as_mut_ptr()), black_box(p1.as_ptr()), black_box(refr1_a.as_mut_ptr()), 3.0, 1);
    }));
    let mut refr8_a = vec![0.0f32; 8];
    let (mut v8, mut u8, mut I8, mut sp8, p8) = (vec![-65.0f32; 8], vec![0.0f32; 8], vec![15.0f32; 8], vec![0i32; 8], pack8(&[0.5, 10.0, 30.0, -65.0, -50.0, -70.0, 2.0, 0.03, 0.2]));
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::adex_step(black_box(v8.as_mut_ptr()), black_box(u8.as_mut_ptr()), black_box(I8.as_mut_ptr()), black_box(sp8.as_mut_ptr()), black_box(p8.as_ptr()), black_box(refr8_a.as_mut_ptr()), 3.0, 8);
    }));
    g.finish();
}

fn bench_saw(c: &mut Criterion) {
    let mut g = c.benchmark_group("saw");
    let mut refr1_s = vec![0.0f32; 1];
    let (mut v1, mut u1, mut tr1, a1, b1, I1, mut sp1) =
        (vec![-65.0f32], vec![-13.0f32], vec![0.0f32], vec![0.02f32], vec![0.2f32], vec![10.0f32], vec![0i32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::saw_step(black_box(v1.as_mut_ptr()), black_box(u1.as_mut_ptr()), black_box(tr1.as_mut_ptr()),
            black_box(a1.as_ptr()), black_box(b1.as_ptr()), black_box(I1.as_ptr()), 0.5, 0.01, 0.1, black_box(sp1.as_mut_ptr()), black_box(refr1_s.as_mut_ptr()), 3.0, 1);
    }));
    let mut refr8_s = vec![0.0f32; 8];
    let (mut v8, mut u8, mut tr8, a8, b8, I8, mut sp8) =
        (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8], vec![0.02f32; 8], vec![0.2f32; 8], vec![10.0f32; 8], vec![0i32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::saw_step(black_box(v8.as_mut_ptr()), black_box(u8.as_mut_ptr()), black_box(tr8.as_mut_ptr()),
            black_box(a8.as_ptr()), black_box(b8.as_ptr()), black_box(I8.as_ptr()), 0.5, 0.01, 0.1, black_box(sp8.as_mut_ptr()), black_box(refr8_s.as_mut_ptr()), 3.0, 8);
    }));
    g.finish();
}

fn bench_psi(c: &mut Criterion) {
    let mut g = c.benchmark_group("psi_wave");
    let (mut v1, mut u1, mut hr1, mut hi1, mut ie1, mut sp1) =
        (vec![1.0f32], vec![0.0f32], vec![0.5f32], vec![0.5f32], vec![0.1f32], vec![0.5f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::psi_wave_step(black_box(v1.as_mut_ptr()), black_box(u1.as_mut_ptr()),
            black_box(hr1.as_mut_ptr()), black_box(hi1.as_mut_ptr()), black_box(ie1.as_mut_ptr()), black_box(sp1.as_mut_ptr()), 0.5, 1.0, 1);
    }));
    let (mut v8, mut u8, mut hr8, mut hi8, mut ie8, mut sp8) =
        (vec![1.0f32; 8], vec![0.0f32; 8], vec![0.5f32; 8], vec![0.5f32; 8], vec![0.1f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::psi_wave_step(black_box(v8.as_mut_ptr()), black_box(u8.as_mut_ptr()),
            black_box(hr8.as_mut_ptr()), black_box(hi8.as_mut_ptr()), black_box(ie8.as_mut_ptr()), black_box(sp8.as_mut_ptr()), 0.5, 1.0, 8);
    }));
    g.finish();
}

fn bench_stdp(c: &mut Criterion) {
    let mut g = c.benchmark_group("stdp");
    let (mut w1, pre1, post1) = (vec![0.5f32], vec![0.3f32], vec![0.1f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::stdp_update(black_box(w1.as_mut_ptr()), black_box(pre1.as_ptr()), black_box(post1.as_ptr()), 0.01, 0.005, 0.0, 2.0, 1);
    }));
    let (mut w8, pre8, post8) = (vec![0.5f32; 8], vec![0.3f32; 8], vec![0.1f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::stdp_update(black_box(w8.as_mut_ptr()), black_box(pre8.as_ptr()), black_box(post8.as_ptr()), 0.01, 0.005, 0.0, 2.0, 8);
    }));
    g.finish();
}

fn bench_kuramoto(c: &mut Criterion) {
    let mut g = c.benchmark_group("kuramoto");
    let (mut p1, o1, c1_) = (vec![1.0f32], vec![2.0f32], vec![0.5f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::kuramoto_update(black_box(p1.as_mut_ptr()), black_box(o1.as_ptr()), black_box(c1_.as_ptr()), 0.5, 1);
    }));
    let (mut p8, o8, c8_) = (vec![1.0f32; 8], vec![2.0f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::kuramoto_update(black_box(p8.as_mut_ptr()), black_box(o8.as_ptr()), black_box(c8_.as_ptr()), 0.5, 8);
    }));
    g.finish();
}

fn bench_t3(c: &mut Criterion) {
    let mut g = c.benchmark_group("t3");
    let (mut ph1, mut a1, o1, c1_, d1) = (vec![1.0f32], vec![0.5f32], vec![2.0f32], vec![0.5f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::t3_update(black_box(ph1.as_mut_ptr()), black_box(a1.as_mut_ptr()), black_box(o1.as_ptr()),
            black_box(c1_.as_ptr()), black_box(d1.as_ptr()), 0.5, 0.01, 1.0, 10.0, 1);
    }));
    let (mut ph8, mut a8, o8, c8_, d8) = (vec![1.0f32; 8], vec![0.5f32; 8], vec![2.0f32; 8], vec![0.5f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::t3_update(black_box(ph8.as_mut_ptr()), black_box(a8.as_mut_ptr()), black_box(o8.as_ptr()),
            black_box(c8_.as_ptr()), black_box(d8.as_ptr()), 0.5, 0.01, 1.0, 10.0, 8);
    }));
    g.finish();
}

fn bench_clamp(c: &mut Criterion) {
    let mut g = c.benchmark_group("safety_clamp");
    let mut d1 = vec![50.0f32];
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe { kernels::safety_clamp(black_box(d1.as_mut_ptr()), 1, -10.0, 10.0); }));
    let mut d8 = vec![50.0f32; 8];
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe { kernels::safety_clamp(black_box(d8.as_mut_ptr()), 8, -10.0, 10.0); }));
    g.finish();
}

fn bench_hebbian(c: &mut Criterion) {
    let mut g = c.benchmark_group("hebbian");
    let (mut w1, cd1) = (vec![0.5f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::hebbian_update(black_box(w1.as_mut_ptr()), black_box(cd1.as_ptr()), 0.01, 0.001, 0.5, 1);
    }));
    let (mut w8, cd8) = (vec![0.5f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::hebbian_update(black_box(w8.as_mut_ptr()), black_box(cd8.as_ptr()), 0.01, 0.001, 0.5, 8);
    }));
    g.finish();
}

fn bench_nlms(c: &mut Criterion) {
    let mut g = c.benchmark_group("nlms");
    let (mut w1, mut bias1, s1) = (vec![0.5f32], vec![0.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::nlms_train(black_box(w1.as_mut_ptr()), black_box(bias1.as_mut_ptr()), black_box(s1.as_ptr()), 1.0, 0.01, 1e-8, 1);
    }));
    let (mut w8, mut bias8, s8) = (vec![0.5f32; 8], vec![0.0f32], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::nlms_train(black_box(w8.as_mut_ptr()), black_box(bias8.as_mut_ptr()), black_box(s8.as_ptr()), 1.0, 0.01, 1e-8, 8);
    }));
    g.finish();
}

fn bench_kalman(c: &mut Criterion) {
    let mut g = c.benchmark_group("kalman");
    let (mut w1, mut cov1, s1) = (vec![0.5f32], vec![1.0f32], vec![0.3f32]);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| b.iter(|| unsafe {
        kernels::kalman_update(black_box(w1.as_mut_ptr()), black_box(cov1.as_mut_ptr()), black_box(s1.as_ptr()), 1.0, 2.0, 1);
    }));
    let (mut w8, mut cov8, s8) = (vec![0.5f32; 8], vec![1.0f32; 8], vec![0.3f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| b.iter(|| unsafe {
        kernels::kalman_update(black_box(w8.as_mut_ptr()), black_box(cov8.as_mut_ptr()), black_box(s8.as_ptr()), 1.0, 2.0, 8);
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
