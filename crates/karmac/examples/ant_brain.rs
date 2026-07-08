//! Ant RL brain benchmark — GNW + HH LUT vs GNW + tanh throughput.
//!
//! Uses rayon for parallel W @ r and GNW operations.

use std::time::Instant;
use rand::Rng;
use rayon::prelude::*;
use karmac::gnw::GlobalWorkspace;

fn bench_tanh() {
    let (n, ws, internal, sc, mc, steps) = (512, 32, 5, 27, 8, 200_000);
    let mut rng = rand::thread_rng();

    let gnw = GlobalWorkspace::new(n, ws);
    let sensor_w: Vec<f32> = (0..sc * 30).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
    let motor_w: Vec<f32> = (0..mc * n).map(|_| rng.gen::<f32>() * 0.1).collect();
    let w: Vec<f32> = (0..n * n).map(|_| rng.gen::<f32>() * 0.1 - 0.05).collect();

    let mut r = vec![0.0f32; n];
    let mut obs = vec![0.0f32; sc];
    let mut motor = vec![0.0f32; mc];
    let mut env_steps: u64 = 0;

    let start = Instant::now();
    for _ in 0..steps {
        for o in obs.iter_mut() { *o = rng.gen::<f32>() * 2.0 - 1.0; }

        for _ in 0..internal {
            // W @ r — параллельно
            let w_mul: Vec<f32> = w.par_chunks_exact(n)
                .map(|row| row.iter().zip(r.iter()).map(|(wj, rj)| wj * rj).sum())
                .collect();
            r.copy_from_slice(&w_mul);

            // Сенсорная инъекция
            for s in 0..30.min(sc) {
                let inj: f32 = (0..sc).map(|oi| sensor_w[oi * 30 + s] * obs[oi]).sum();
                r[s] += inj;
            }

            // tanh — поэлементно
            for ri in r.iter_mut() { *ri = ri.tanh(); }

            // GNW
            gnw.forward(&mut r);
            for ri in r.iter_mut() { *ri = ri.clamp(-5.0, 5.0); }
        }

        // Мотор — параллельно
        motor.par_iter_mut().enumerate().for_each(|(m, out)| {
            let sum: f32 = (0..n).map(|i| motor_w[m * n + i] * r[i]).sum();
            *out = sum.tanh();
        });
        env_steps += 1;
    }
    let elapsed = start.elapsed();
    let sps = env_steps as f64 / elapsed.as_secs_f64();

    println!("═══ GNW + tanh (rayon) ═══");
    println!("  Env steps:   {}", env_steps);
    println!("  Time:        {:.3}s", elapsed.as_secs_f64());
    println!("  Env steps/s: {:.0}", sps);
    println!("  Speedup vs Python (~480/s): {:.1}×", sps / 480.0);
    println!("  Motor[0]:    {:.4}", motor[0]);
    println!("  2M steps in: {:.1}s", 2_000_000.0 / sps);
    println!();
}

fn bench_hh() {
    let (n, ws, internal, sub, sc, mc, steps) = (512, 32, 5, 10, 27, 8, 50_000);
    let mut rng = rand::thread_rng();

    let channels = taas_kernels::neuron::hh::pacemaking_channels();
    let n_ch = channels.len();
    let n_g: usize = channels.iter().map(|ch| ch.gates.len()).sum();
    let dt = 0.1f32;
    let luts = taas_kernels::neuron::hh::build_luts(&channels, dt, -100.0, 60.0, 1024);

    let mut m_inf = Vec::with_capacity(n_g);
    let mut decay = Vec::with_capacity(n_g);
    let mut powers = Vec::with_capacity(n_g);
    let mut e_rev = Vec::with_capacity(n_ch);
    let mut g_max = Vec::with_capacity(n_ch);
    let mut ngpc = Vec::with_capacity(n_ch);

    for (ci, ch) in channels.iter().enumerate() {
        e_rev.push(ch.E_rev); g_max.push(ch.g_max); ngpc.push(ch.gates.len());
        for (gi, gate) in ch.gates.iter().enumerate() {
            m_inf.push(luts[ci][gi].m_inf.clone());
            decay.push(luts[ci][gi].decay.clone());
            powers.push(gate.power);
        }
    }

    let mut v = vec![-65.0f32; n];
    let mut gates = vec![0.5f32; n * n_g];
    let mut ii = vec![0.0f32; n];
    let g_leak = vec![0.3f32; n];
    let e_leak = vec![-65.0f32; n];
    let c_m = vec![1.0f32; n];
    let m_inf_p: Vec<*const f32> = m_inf.iter().map(|v| v.as_ptr()).collect();
    let decay_p: Vec<*const f32> = decay.iter().map(|v| v.as_ptr()).collect();

    let gnw = GlobalWorkspace::new(n, ws);
    let sensor_w: Vec<f32> = (0..sc * 30).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect();
    let motor_w: Vec<f32> = (0..mc * n).map(|_| rng.gen::<f32>() * 0.1).collect();

    let mut state = vec![0.0f32; n];
    let mut obs = vec![0.0f32; sc];
    let mut motor = vec![0.0f32; mc];
    let mut env_steps: u64 = 0;

    let start = Instant::now();
    for _ in 0..steps {
        for o in obs.iter_mut() { *o = rng.gen::<f32>() * 2.0 - 1.0; }

        for _ in 0..internal {
            for s in 0..30.min(sc) {
                let inj: f32 = (0..sc).map(|oi| sensor_w[oi * 30 + s] * obs[oi]).sum();
                ii[s] = inj;
            }

            for _ in 0..sub {
                unsafe {
                    taas_kernels::neuron::hh::hh_lut_step(
                        v.as_mut_ptr(), gates.as_mut_ptr(), ii.as_ptr(),
                        g_leak.as_ptr(), e_leak.as_ptr(), c_m.as_ptr(),
                        n, n_g, m_inf_p.as_ptr(), decay_p.as_ptr(),
                        -100.0, 60.0, 1024.0,
                        e_rev.as_ptr(), g_max.as_ptr(), powers.as_ptr(),
                        ngpc.as_ptr(), n_ch, dt,
                    );
                }
            }

            // Нормализация V — параллельно
            state.par_iter_mut().zip(v.par_iter())
                .for_each(|(s, &vi)| *s = (vi.clamp(-100.0, 60.0) / 30.0).tanh());
            gnw.forward(&mut state);
            for s in state.iter_mut() { *s = s.clamp(-5.0, 5.0); }
        }

        motor.par_iter_mut().enumerate().for_each(|(m, out)| {
            let sum: f32 = (0..n).map(|i| motor_w[m * n + i] * state[i]).sum();
            *out = sum.tanh();
        });
        env_steps += 1;
    }
    let elapsed = start.elapsed();
    let sps = env_steps as f64 / elapsed.as_secs_f64();

    println!("═══ GNW + HH LUT (CaV-T spiking, rayon) ═══");
    println!("  Env steps:   {}", env_steps);
    println!("  Time:        {:.3}s", elapsed.as_secs_f64());
    println!("  Env steps/s: {:.0}", sps);
    println!("  HH steps/s:  {:.0}", sps as f64 * internal as f64 * sub as f64);
    println!("  Motor[0]:    {:.4}", motor[0]);
    println!();
}

fn main() {
    println!();
    bench_tanh();
    bench_hh();
    println!("══════════════════════════════════════════");
}
