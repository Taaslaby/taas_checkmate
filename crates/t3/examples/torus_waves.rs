//!  Kuramoto на пончике: визуализация волн/спиралей.
//!
//! Запуск: `cargo run -p taas-t3 --features serde --example torus_waves --release`
//!
//!  1. Локальная связь на торе → бегущие волны
//!  2. Глобальный R и локальные R по патчам — мера кооперативности
//!  3. Цветная фазовая карта (ANSI) в реальном времени
//!
//! Параметры подобраны для наглядной динамики.

use std::io::{self, Write};
use std::thread;
use std::time::Duration;
use taas_t3::torus::TorusGrid;

const W: usize = 60;
const H: usize = 30;
const DT: f32 = 0.1;
const COUPLING: f32 = 4.0;
const RADIUS: usize = 2; // 24 соседа
const DRIVE_GAIN: f32 = 10.0;
const ALPHA: f32 = 1.0;
const EPS: f32 = 0.1;
const RING_STEPS: usize = 150;

/// Создать драйв: кольцо на торе (как внешний стимул — запускает волны).
fn ring_drive(tg: &TorusGrid, cx: usize, cy: usize, ring_r: usize, val: f32) -> Vec<f32> {
    let mut d = tg.drives_vec();
    for y in 0..tg.h {
        for x in 0..tg.w {
            let dx = (x as isize - cx as isize)
                .abs()
                .min((tg.w as isize - (x as isize - cx as isize).abs()).abs());
            let dy = (y as isize - cy as isize)
                .abs()
                .min((tg.h as isize - (y as isize - cy as isize).abs()).abs());
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist <= ring_r as f32 {
                d[tg.idx(x, y)] = val * (1.0 - dist / ring_r as f32);
            }
        }
    }
    d
}

/// ANSI-цвет по фазе (24-бит HSV).
fn ansify_phase(phase: f32) -> String {
    let hue = phase.rem_euclid(2.0 * std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let rgb = hsv_to_rgb(hue, 0.9, 1.0);
    format!("\x1b[48;2;{};{};{}m \x1b[0m", rgb.0, rgb.1, rgb.2)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = (v * (1.0 - s) * 255.0) as u8;
    let q = (v * (1.0 - f * s) * 255.0) as u8;
    let t = (v * (1.0 - (1.0 - f) * s) * 255.0) as u8;
    let vv = (v * 255.0) as u8;
    match i % 6 {
        0 => (vv, t, p),
        1 => (q, vv, p),
        2 => (p, vv, t),
        3 => (p, q, vv),
        4 => (t, p, vv),
        _ => (vv, p, q),
    }
}

fn main() {
    println!("\n\n🍩 Kuramoto Torus Waves   ({W}×{H}, K={COUPLING}, radius={RADIUS})");
    println!("   Каждая точка — фаза осциллятора (цвет = угол θ)");
    println!("   R — глобальный порядок, r_loc — локальный (патч 6×6)");
    println!();

    let mut tg = TorusGrid::new(W, H, 1.0, 1.0, 13); // одинаковые частоты!
    let total = 500;

    for step in 0..total {
        // Прикладываем кольцевой стимул первые RING_STEPS шагов — запускает волну
        let drives = if step < RING_STEPS {
            ring_drive(&tg, W / 2, H / 2, 10, 0.8)
        } else {
            let mut d = tg.drives_vec();
            TorusGrid::fill_drives(&mut d, 0.0);
            d
        };

        tg.step(DT, COUPLING, RADIUS, &drives, DRIVE_GAIN, ALPHA, EPS);

        // Вывод каждые 2 шага
        if step % 2 != 0 {
            continue;
        }

        let r = tg.order_parameter();
        let spectrum = tg.wave_spectrum(3);

        // Очистить и перерисовать
        print!("\x1b[{}A\x1b[J", H + 8);
        println!("   step {step:>4}  ║  R = {r:.4}");
        println!("   интерференция (фурье-моды |k|≤3):");
        for &(kx, ky, power) in spectrum.iter().take(5) {
            let bar = "█".repeat((power * 30.0) as usize);
            println!("     (k={kx:>2},{ky:>2})  {power:.4}  {bar}");
        }
        let field = tg.phase_field();
        for y in 0..H {
            print!("   ");
            for x in 0..W {
                print!("{}", ansify_phase(field[y * W + x]));
            }
            println!();
        }
        println!();

        let _ = io::stdout().flush();
        // В релизе без паузы — полная скорость; в дебаге лёгкая пауза
        if cfg!(debug_assertions) {
            thread::sleep(Duration::from_millis(10));
        }
    }

    let r = tg.order_parameter();
    println!("\n── финал ──");
    println!("   R = {r:.4}");
    println!(
        "   Фазы: min={:.2} max={:.2}",
        tg.phases.iter().fold(f32::INFINITY, |a, &p| a.min(p)),
        tg.phases.iter().fold(f32::NEG_INFINITY, |a, &p| a.max(p))
    );
    println!();
}
