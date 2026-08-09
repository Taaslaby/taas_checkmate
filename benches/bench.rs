#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::sync::OnceLock;
use taas_checkmate::kernels;
use taas_checkmate::pillars::core::runner::run_neuron;
use taas_checkmate::pillars::ops::triangulated::TriangLUT;
use taas_checkmate::pillars::zoo::neurons::{
    AdExModel, HHModel, IzhikevichModel, LIFAtModel, LIFModel, SAWModel,
};
use taas_checkmate::pillars::zoo::Model;

static HH_LUT: OnceLock<TriangLUT> = OnceLock::new();

fn bench_izhikevich(c: &mut Criterion) {
    let mut g = c.benchmark_group("izhikevich");
    let (mut v1, mut u1, mut tr1) = (vec![-65.0f32], vec![-13.0f32], vec![0.0f32]);
    let (I1, mut sp1) = (vec![10.0f32], vec![0i32]);
    let mut refr1 = vec![0.0f32; 1];
    let p1 = pack_n(&[0.5, 0.02, 0.2, -65.0, 8.0, 3.0, 30.0], 7);
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            izh_run(&mut v1, &mut u1, &mut refr1, &I1, &mut sp1, &p1);
            // legacy STDP side-output (trace) не участвует в pillars-каноне
            for i in 0..v1.len() {
                if sp1[i] != 0 {
                    tr1[i] = tr1[i] * 0.01 + (1.0 - tr1[i]);
                } else {
                    tr1[i] *= 1.0 - 0.01;
                }
            }
            black_box(());
        })
    });
    let (mut v8, mut u8, mut tr8) = (vec![-65.0f32; 8], vec![-13.0f32; 8], vec![0.0f32; 8]);
    let (I8, mut sp8) = (vec![10.0f32; 8], vec![0i32; 8]);
    let mut refr8 = vec![0.0f32; 8];
    let p8 = pack_n(&[0.5, 0.02, 0.2, -65.0, 8.0, 3.0, 30.0], 7);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            izh_run(&mut v8, &mut u8, &mut refr8, &I8, &mut sp8, &p8);
            for i in 0..v8.len() {
                if sp8[i] != 0 {
                    tr8[i] = tr8[i] * 0.01 + (1.0 - tr8[i]);
                } else {
                    tr8[i] *= 1.0 - 0.01;
                }
            }
            black_box(());
        })
    });
    g.finish();
}

fn bench_lif(c: &mut Criterion) {
    let mut g = c.benchmark_group("lif");
    let mut refr1_l = vec![0.0f32; 1];
    let (mut v1, I1, mut sp1, p1) = (
        vec![-65.0f32],
        vec![15.0f32],
        vec![0i32],
        pack_n(&[0.5, 0.1, -65.0, -50.0, -70.0, 3.0], 6), // [dt, 1/τ_m, v_rest, v_thresh, v_reset, tau_ref]
    );
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_n1", |b| {
        b.iter(|| {
            lif_run(&mut v1, &mut refr1_l, &I1, &mut sp1, &p1);
            black_box(());
        })
    });
    let mut refr8_l = vec![0.0f32; 8];
    let (mut v8, I8, mut sp8, p8) = (
        vec![-65.0f32; 8],
        vec![15.0f32; 8],
        vec![0i32; 8],
        pack_n(&[0.5, 0.1, -65.0, -50.0, -70.0, 3.0], 6), // [dt, 1/τ_m, v_rest, v_thresh, v_reset, tau_ref]
    );
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            lif_run(&mut v8, &mut refr8_l, &I8, &mut sp8, &p8);
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
            lif_at_run(
                &mut v1,
                &mut th1,
                &mut I1,
                &mut sp1,
                &p1,
                &mut refr1_la,
                3.0,
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
            lif_at_run(
                &mut v8,
                &mut th8,
                &mut I8,
                &mut sp8,
                &p8,
                &mut refr8_la,
                3.0,
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
            adex_run(&mut v1, &mut u1, &mut I1, &mut sp1, &p1, &mut refr1_a, 3.0);
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
            adex_run(&mut v8, &mut u8, &mut I8, &mut sp8, &p8, &mut refr8_a, 3.0);
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
            saw_run(
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
            saw_run(
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
            kernels::phase::kuramoto_update(&mut p1, &o1, &c1_, 0.5);
            black_box(());
        })
    });
    let (mut p8, o8, c8_) = (vec![1.0f32; 8], vec![2.0f32; 8], vec![0.5f32; 8]);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_n8", |b| {
        b.iter(|| {
            kernels::phase::kuramoto_update(&mut p8, &o8, &c8_, 0.5);
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
                hh_poly_run(
                    &mut v1, &mut g1, &I1, &gl1, &El1, &cm1, &meta, &gmax1, 0.01, 6.3,
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
                hh_poly_run(
                    &mut v8, &mut g8, &I8, &gl8, &El8, &cm8, &meta, &gmax8, 0.01, 6.3,
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
                hh_poly_run(
                    &mut v1024, &mut g1024, &I1024, &gl1024, &El1024, &cm1024, &meta, &gmax1024,
                    0.01, 6.3,
                );
                black_box(());
            })
        });
}

fn bench_hh_lut(c: &mut Criterion) {
    // Canonical LUT = pillars `TriangLUT` (calibrated on the full HHModel,
    // 4096×8B = 32KB, L1-resident) + `TriangulatedModel` inference step.
    // Legacy `kernels::neuron::hh::{build_luts, hh_lut_step, GateLut}` was
    // removed in the pillars consolidation (see DUPLICATION_MAP.md).
    use taas_checkmate::pillars::ops::triangulated::{LutBuilder, LutConfig};
    use taas_checkmate::pillars::zoo::neurons::TriangulatedModel;
    use taas_checkmate::pillars::zoo::HHModel;
    use taas_checkmate::pillars::NeuralPopulation;

    let lut = HH_LUT.get_or_init(|| {
        let params = hh_lut_params();
        let state_init = [-65.0, 0.05, 0.6, 0.32, 1.0];
        let config = LutConfig {
            dt: 0.01,
            input_modes: vec![(1.0, 50.0, 1)],
            v_min: -100.0,
            v_max: 60.0,
            n_steps: 2_000_000,
            seed: 42,
        };
        let lut = LutBuilder::new(config).build::<HHModel>(&params, &state_init);
        eprintln!("TriangLUT (HH-calibrated) ready — 32KB, L1-resident");
        lut
    });
    TriangulatedModel::set_lut(lut.clone());

    // Step a TriangulatedModel population (V + 4-bit input history).
    let bench_n = |c: &mut Criterion, n: usize, name: &str| {
        let mut pop = NeuralPopulation::<TriangulatedModel>::new(n);
        pop.fill_param(0, 0.01); // dt
        pop.fill_param(1, 0.0); // v_thresh (HH uses 0.0)
        pop.fill_param(2, 0.5); // input_thresh
        pop.fill_state(0, -65.0); // v
        pop.fill_state(1, 0.0); // history
                                // Alternating 50/0 like pillars `make_current` — exercises both LUT
                                // history states (input present / absent) instead of saturating one.
        let ii: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 50.0 } else { 0.0 })
            .collect();
        let mut spikes = vec![0i32; n];
        c.benchmark_group("hh_lut")
            .throughput(Throughput::Elements(n as u64))
            .bench_function(name, |b| {
                b.iter(|| {
                    pop.fill_state(0, -65.0);
                    pop.fill_state(1, 0.0);
                    for _ in 0..100 {
                        pop.step(&ii, &mut spikes);
                    }
                    black_box(&spikes);
                })
            });
    };
    bench_n(c, 1, "triang_n1");
    bench_n(c, 8, "triang_n8");
    bench_n(c, 1024, "triang_n1024");
}

fn hh_lut_params() -> Vec<f32> {
    use taas_checkmate::pillars::zoo::Model;
    // HHModel::N_PARAMS = 12 (incl. CaV-T g_CaT / E_CaT columns).
    // Single-neuron SoA calibration buffer (n = 1).
    let n = 1;
    let mut p = vec![0.0f32; n * taas_checkmate::pillars::zoo::HHModel::N_PARAMS];
    p[0] = 0.01; // dt
    p[n] = 120.0; // g_Na
    p[2 * n] = 50.0; // E_Na
    p[3 * n] = 36.0; // g_K
    p[4 * n] = -77.0; // E_K
    p[5 * n] = 0.3; // g_Leak
    p[6 * n] = -54.387; // E_Leak
    p[7 * n] = 1.0; // q10
    p[8 * n] = 0.0; // v_thresh
    p[9 * n] = 1.0; // inv_cm
    p[10 * n] = 0.0; // g_CaT — 0 = standard squid-axon HH
    p[11 * n] = 120.0; // E_CaT
    p
}

fn pack1(vals: &[f32]) -> Vec<f32> {
    vals.to_vec()
}
fn pack_n(vals: &[f32], cols: usize) -> Vec<f32> {
    let mut p = Vec::with_capacity(vals.len() * cols);
    for &v in vals {
        for _ in 0..cols {
            p.push(v);
        }
    }
    p
}
fn pack8(vals: &[f32]) -> Vec<f32> {
    pack_n(vals, 8)
}

/// Pillars-канон (транш 3b): IzhikevichModel 7-col, state [v, u, refr].
fn izh_run(v: &mut [f32], u: &mut [f32], refr: &mut [f32], I: &[f32], sp: &mut [i32], p: &[f32]) {
    unsafe {
        run_neuron::<IzhikevichModel>(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            refr.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            sp.as_mut_ptr(),
            p.as_ptr(),
            v.len(),
        );
    }
}

/// Pillars-канон (транш 3b): LIFModel 6-col, state [v, refr].
fn lif_run(v: &mut [f32], refr: &mut [f32], I: &[f32], sp: &mut [i32], p: &[f32]) {
    unsafe {
        run_neuron::<LIFModel>(
            v.as_mut_ptr(),
            refr.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            sp.as_mut_ptr(),
            p.as_ptr(),
            v.len(),
        );
    }
}

/// Pillars-канон (транш 3e): LIFAtModel, legacy 7-col → 8-col repacking.
/// params: [dt, tau_m, v_rest, v_thresh, v_reset, tau_theta, beta] × n.
fn lif_at_run(
    v: &mut [f32],
    th: &mut [f32],
    I: &mut [f32],
    sp: &mut [i32],
    p: &[f32],
    refr: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    let mut params = vec![0.0f32; n * LIFAtModel::N_PARAMS];
    for i in 0..n {
        params[i] = p[i];
        let tm = p[n + i];
        params[n + i] = if tm > 0.0 { 1.0 / tm } else { 0.0 };
        params[2 * n + i] = p[2 * n + i];
        params[3 * n + i] = p[3 * n + i];
        params[4 * n + i] = p[4 * n + i];
        let tt = p[5 * n + i];
        params[5 * n + i] = if tt > 0.0 { 1.0 / tt } else { 1e9 };
        params[6 * n + i] = p[6 * n + i];
        params[7 * n + i] = tau_ref;
    }
    unsafe {
        run_neuron::<LIFAtModel>(
            v.as_mut_ptr(),
            th.as_mut_ptr(),
            refr.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            sp.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
}

/// Pillars-канон (транш 3e): AdExModel, legacy 9-col → 12-col repacking.
/// params: [dt, tau_m, tau_w, v_rest, v_thresh, v_reset, delta_t, a, b] × n.
fn adex_run(
    v: &mut [f32],
    u: &mut [f32],
    I: &mut [f32],
    sp: &mut [i32],
    p: &[f32],
    refr: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    let mut params = vec![0.0f32; n * AdExModel::N_PARAMS];
    for i in 0..n {
        params[i] = p[i];
        params[n + i] = 1.0 / p[n + i].max(1e-6);
        params[2 * n + i] = 1.0 / p[2 * n + i].max(1e-6);
        params[3 * n + i] = p[3 * n + i];
        params[4 * n + i] = p[4 * n + i];
        params[5 * n + i] = p[5 * n + i];
        params[6 * n + i] = p[6 * n + i];
        params[7 * n + i] = 1.0 / p[6 * n + i].max(1e-6);
        params[8 * n + i] = p[7 * n + i];
        params[9 * n + i] = p[8 * n + i];
        params[10 * n + i] = tau_ref;
        params[11 * n + i] = p[4 * n + i]; // v_peak := v_thresh
    }
    unsafe {
        run_neuron::<AdExModel>(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            refr.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            sp.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
}

/// Pillars-канон (транш 3e): SAWModel 8-col + noise injection.
fn saw_run(
    v: &mut [f32],
    u: &mut [f32],
    tr: &mut [f32],
    a: &[f32],
    b: &[f32],
    I: &[f32],
    dt: f32,
    trace_decay: f32,
    _noise_amp: f32,
    sp: &mut [i32],
    refr: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    let mut params = vec![0.0f32; n * SAWModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = a[i];
        params[2 * n + i] = b[i];
        params[3 * n + i] = trace_decay;
        params[4 * n + i] = 0.0; // noise_amp — модель детерминирована
        params[5 * n + i] = tau_ref;
        params[6 * n + i] = 30.0;
        params[7 * n + i] = -65.0;
    }
    unsafe {
        run_neuron::<SAWModel>(
            v.as_mut_ptr(),
            u.as_mut_ptr(),
            tr.as_mut_ptr(),
            refr.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            I.as_ptr(),
            sp.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
}

/// Pillars-канон (транш 3e): HHModel, standard-layout repacking (no spike det.).
/// g_max: per-channel [n_channels × n]; gates: [n_gates × n] (m, h, n order).
fn hh_poly_run(
    v: &mut [f32],
    gates: &mut [f32],
    I: &[f32],
    g_leak: &[f32],
    e_leak: &[f32],
    c_m: &[f32],
    meta: &[taas_checkmate::kernels::neuron::hh::ChannelMeta],
    g_max: &[f32],
    dt: f32,
    celsius: f32,
) {
    let n = v.len();
    // Бенч хардкодит 2-канальный лейаут (standard, без CaV-T) — как в safe.rs
    // adapter'е при meta.len() == 2. Для 3-канального (pacemaking) тут не бенчим.
    let q10 = 3.0f32.powf((celsius - 6.3) / 10.0);
    let mut params = vec![0.0f32; n * HHModel::N_PARAMS];
    for i in 0..n {
        params[i] = dt;
        params[n + i] = g_max[i];
        params[2 * n + i] = meta[0].e_rev;
        params[3 * n + i] = g_max[n + i];
        params[4 * n + i] = meta[1].e_rev;
        params[5 * n + i] = g_leak[i];
        params[6 * n + i] = e_leak[i];
        params[7 * n + i] = q10;
        params[8 * n + i] = 0.0;
        params[9 * n + i] = 1.0 / c_m[i].max(1e-6);
        params[10 * n + i] = 0.0;
        params[11 * n + i] = 120.0;
    }
    let mut spikes = vec![0i32; n];
    let mut was_below = vec![0.0f32; n];
    unsafe {
        run_neuron::<HHModel>(
            v.as_mut_ptr(),
            gates.as_mut_ptr(),
            gates.as_mut_ptr().add(n),
            gates.as_mut_ptr().add(2 * n),
            std::ptr::null_mut(),
            was_below.as_mut_ptr(),
            I.as_ptr(),
            spikes.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
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
