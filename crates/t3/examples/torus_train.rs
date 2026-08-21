//! 🌊 Волновой таргет: обучаем K поднять мощность конкретной фурье-моды.
//!
//! Задача: loss = (F(kx,ky) − target)² на финальном шаге.
//! Аналитический градиент: ∂F/∂θⱼ (fourier_mode_grad) → BPTT → dL/dK.
//! Это доказывает: тор управляем через градиент — система сама подбирает
//! coupling, чтобы породить нужную волну.
//!
//! Запуск: `cargo run -p taas-t3 --features serde --example torus_train --release`

use taas_t3::torus::{TorusGrid, TorusTrajectory};

const W: usize = 16;
const H: usize = 8;
const DT: f32 = 0.1;
const RADIUS: usize = 1;
const DRIVE_GAIN: f32 = 10.0;
const ALPHA: f32 = 1.0;
const EPS: f32 = 0.1;
const N_STEPS: usize = 8;
const EPOCHS: usize = 300;
const LR: f32 = 0.02;

// Волновой таргет: хотим поднять моду (kx=1, ky=0) — бегущую волну вправо.
const TARGET_KX: i32 = 1;
const TARGET_KY: i32 = 0;
const TARGET_POWER: f32 = 0.9;

fn main() {
    // Одинаковые частоты + разброс фаз = способность к волнообразованию
    let init_tg = TorusGrid::new(W, H, 1.0, 1.0, 42);
    let drives = vec![0.0f32; W * H];

    let mut k = 0.3f32; // слабая связь — волны не держатся

    // ── Состояние ДО обучения ──
    let (f_before, r_before) = {
        let mut g = init_tg.clone();
        for _ in 0..N_STEPS {
            g.step(DT, k, RADIUS, &drives, DRIVE_GAIN, ALPHA, EPS);
        }
        let f = g.fourier_mode(TARGET_KX, TARGET_KY);
        let r = g.order_parameter();
        (f, r)
    };

    println!("🌊 Wave-target BPTT on {W}×{H} torus\n");
    println!("   target:  F({TARGET_KX},{TARGET_KY}) = {TARGET_POWER}");
    println!("   before:  F = {f_before:.5}, R = {r_before:.5}, K = {k:.3}\n");

    // ── Обучение ──
    let tg = init_tg.clone();
    for epoch in 0..=EPOCHS {
        // Forward + record trajectory
        let mut grid = tg.clone();
        let mut traj = TorusTrajectory::new(N_STEPS, W * H);
        for step in 0..N_STEPS {
            grid.record_step(&mut traj, step, &drives);
            grid.step(DT, k, RADIUS, &drives, DRIVE_GAIN, ALPHA, EPS);
        }

        // Градиент от волнового таргета
        let n = W * H;
        let (f, grad_f) = grid.fourier_mode_grad(TARGET_KX, TARGET_KY);
        // dL/dθ = 2·(F − target)·∂F/∂θ
        let dth_final: Vec<f32> = grad_f
            .iter()
            .map(|&df| 2.0 * (f - TARGET_POWER) * df)
            .collect();
        let dr_final = vec![0.0f32; n];

        let (dk, _, _, _) = grid.bptt(
            &traj,
            &dth_final,
            &dr_final,
            &vec![0.0f32; n],
            k,
            RADIUS,
            DT,
            DRIVE_GAIN,
            ALPHA,
            EPS,
        );

        // SGD
        k -= LR * dk;
        k = k.clamp(0.01, 20.0);

        if epoch % 60 == 0 || epoch == EPOCHS {
            // Эвалюация на свежем старте
            let mut g = init_tg.clone();
            for _ in 0..N_STEPS {
                g.step(DT, k, RADIUS, &drives, DRIVE_GAIN, ALPHA, EPS);
            }
            let f_val = g.fourier_mode(TARGET_KX, TARGET_KY);
            let r_val = g.order_parameter();
            let spec = g.wave_spectrum(3);
            let top = &spec[0];
            let target_in_top = spec
                .iter()
                .take(5)
                .any(|(kx, ky, _)| *kx == TARGET_KX && *ky == TARGET_KY);
            let star = if target_in_top { " ← target!" } else { "" };
            println!(
                "   epoch {epoch:>3}  K={k:.4}  F({TARGET_KX},{TARGET_KY})={f_val:.5}  R={r_val:.5}  top=(k={},{})={:.4}{star}",
                top.0, top.1, top.2
            );
        }
    }

    // ── Финальная оценка ──
    let f_after = {
        let mut g = init_tg.clone();
        for _ in 0..N_STEPS {
            g.step(DT, k, RADIUS, &drives, DRIVE_GAIN, ALPHA, EPS);
        }
        g.fourier_mode(TARGET_KX, TARGET_KY)
    };

    println!("\n   🎯 F({TARGET_KX},{TARGET_KY}): {f_before:.5} → {f_after:.5}");
    println!("   🎯 K: 0.300 → {k:.4}");

    if f_after > 0.4 && f_after > f_before * 2.0 {
        let gain = f_after / f_before.max(1e-5);
        println!("   ✅ Тор управляем через волновой таргет: мода выросла в {gain:.1}x!");
    } else if f_after > f_before * 1.5 {
        println!(
            "   ✅ Тор управляем: мода выросла в {:.1}x!",
            f_after / f_before
        );
    } else {
        println!("   ⚠️  Мода не поднялась — возможно, нужна более богатая топология (T³)");
    }
}
