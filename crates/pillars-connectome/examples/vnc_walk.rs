//! vnc_walk.rs — реальный MANC VNC: всплывает ли ходьба сама?
//!
//! Грузит `manc_vnc_signed.taas` (реальные веса = synapse counts + знак из
//! predictedNt), стимулирует нисходящие нейроны (DN = команда ходьбы из мозга),
//! гоняет HAAP (M-ток адаптация + h-ток rebound — half-center модель), и меряет
//! антифазу левых/правых моторных нейронов по каждой ноге (T1/T2/T3).
//!
//! Запуск:
//! ```bash
//! cargo run -p pillars-connectome --release --example vnc_walk -- \
//!     resources/data/manc_vnc_signed.taas [dn_drive] [ms]
//! ```

use std::io::BufReader;
use std::time::Instant;

use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

include!("support/vnc_indices_inc.rs");

/// Pearson correlation: <0 → антифаза (левая/правая ноги в противофазе).
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "resources/data/manc_vnc_signed.taas".into());
    let dn_drive: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(8.0);
    let ms: f32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(500.0);

    // ── Загрузка ──
    let f = std::fs::File::open(&path).expect("cannot open .taas");
    let conn = ConnectomeData::load_baked(&mut BufReader::new(f)).expect("bad .taas");
    let n = conn.n_neurons;
    println!("[vnc] n_neurons={n} n_synapses={}", conn.n_synapses);

    // ── HAAP + проводимость не нужна: current-based с реальным знаком ──
    let dt = 0.05f32;
    let tau_syn = 8.0f32; // персистентное торможение → rebound
    let mut runner = BrainRunner::new(conn, NeuronConfig::HaapSynaptic { dt_ms: dt, tau_syn })
        .expect("runner init");

    // ── Стимуляция: нисходящие (DN) = команда «идти» ──
    let mut ext = vec![0.0f32; n];
    for &i in DN {
        ext[i] = dn_drive;
    }
    println!("[stim] {dn_drive} на {} DN-нейронов", DN.len());

    let n_steps = (ms / dt) as usize;
    let chunk = 100usize; // 5 мс на бин
    let n_bins = n_steps / chunk;

    let legs: [(&str, &[usize], &[usize]); 3] =
        [("T1", T1_L, T1_R), ("T2", T2_L, T2_R), ("T3", T3_L, T3_R)];

    // ── Прогон чанками, сбор L/R серий ──
    let mut series: Vec<(Vec<f32>, Vec<f32>)> =
        vec![(Vec::with_capacity(n_bins), Vec::with_capacity(n_bins)); 3];
    let t0 = Instant::now();
    for _b in 0..n_bins {
        let buf = runner.run(chunk, &ext).to_vec();
        for (li, (_name, l_idx, r_idx)) in legs.iter().enumerate() {
            let (mut l, mut r) = (0.0f32, 0.0f32);
            for step in 0..chunk {
                let off = step * n;
                for &i in *l_idx {
                    if buf[off + i] != 0 {
                        l += 1.0;
                    }
                }
                for &i in *r_idx {
                    if buf[off + i] != 0 {
                        r += 1.0;
                    }
                }
            }
            series[li].0.push(l);
            series[li].1.push(r);
        }
    }
    let wall = t0.elapsed();

    // ── Отчёт ──
    let bin_s = chunk as f32 * dt / 1000.0;
    let bin_ms = bin_s * 1000.0;
    println!("\n── {ms} мс мозга, {n_bins} бинов по {bin_ms:.0} мс ──");
    for (li, (name, l_idx, _r_idx)) in legs.iter().enumerate() {
        let l: f32 = series[li].0.iter().sum();
        let r: f32 = series[li].1.iter().sum();
        let corr = pearson(&series[li].0, &series[li].1);
        let l_rate = l / (n_bins as f32 * bin_s) / l_idx.len() as f32;
        let r_rate = r / (n_bins as f32 * bin_s) / legs[li].2.len() as f32;
        let phase = if corr < -0.3 {
            "АНТИФАЗА"
        } else if corr > 0.3 {
            "синфазно"
        } else {
            "нет корреляции"
        };
        println!("  {name}: L={l_rate:.1} Hz  R={r_rate:.1} Hz  corr(L,R)={corr:+.3}  → {phase}");
    }
    println!(
        "\n[wall] {:.2} с на {} мс мозга = {:.3}x realtime",
        wall.as_secs_f64(),
        ms,
        ms as f64 / 1000.0 / wall.as_secs_f64()
    );
}
