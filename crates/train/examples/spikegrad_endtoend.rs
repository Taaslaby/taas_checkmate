//! End-to-end SPIGRAD: обучаемый спайковый источник + мост, один градиент.
//!
//! До этой сессии SPIGRAD обрывал градиент на входе моста:
//! `spikegrad_embodied.rs` честно писал «VSN weights are fixed — this is not
//! end-to-end gradient learning». Теперь [`SpikeGradBridge::backward_full`]
//! возвращает `dL/d(spikes)`, и градиент течёт вверх — в дифференцируемый
//! источник [`TrainableSpikeSource`] (математика TransformerModel).
//!
//! ```text
//! features → source(w,v) → spikes → bridge(W,b) → T3 → PST → loss
//!      ▲                                              │
//!      └──── dL/dw,dL/dv ◀── dL/dspikes ◀── BPTT ─────┘
//! ```
//!
//! Задача: teacher-цепочка (фикс. источник + фикс. мост) генерирует целевой
//! PST из периодических признаков. Студент учится его воспроизводить.
//! Сравниваются две конфигурации с одинаковым стартом:
//!
//! 1. **bridge-only** — источник заморожен (w=v=1), учится только мост.
//! 2. **end-to-end** — учится источник И мост совместно.
//!
//! Run: `cargo run -p taas-train --features t3 --example spikegrad_endtoend --release`

#![cfg(feature = "t3")]

use taas_train::spike::t3spike_compute_pst_matrix;
use taas_train::spike_grad::SpikeGradBridge;
use taas_train::spike_source::TrainableSpikeSource;
use taas_train::t3::pst_linear_dim;

const STEPS: usize = 48;
const SPIKE_DIM: usize = 4;
const N_OSC: usize = 2;
const TRAIN_SEQUENCES: usize = 6;
const VALIDATION_SEQUENCES: usize = 4;
const EPOCHS: usize = 150;
const BRIDGE_DT: f32 = 0.05;
const DRIVE_GAIN: f32 = 10.0;
const ALPHA: f32 = 1.0;
const FREQ_MIN: f32 = 0.5;
const FREQ_MAX: f32 = 3.0;
const W12: f32 = 0.3;
const W13: f32 = 0.15;
const W23: f32 = 0.08;
const ADAM_LR_BRIDGE: f32 = 0.01;
const ADAM_LR_SOURCE: f32 = 0.02;
const GRAD_CLIP: f32 = 0.25;

// Источник: dt, decay, gain, inv_tau, k_scale (как в mnist_grad/движке)
const SRC_DT: f32 = 0.5;
const SRC_DECAY: f32 = 0.9;
const SRC_GAIN: f32 = 3.0;
const SRC_INV_TAU: f32 = 0.2;
const SRC_K_SCALE: f32 = 2.0;

struct Sequence {
    features: Vec<f32>, // [STEPS × SPIKE_DIM]
    target_pst: Vec<f32>,
}

fn teacher_bridge() -> SpikeGradBridge {
    let drive_dim = 3 * N_OSC;
    let mut weights = Vec::with_capacity(SPIKE_DIM * drive_dim);
    for i in 0..SPIKE_DIM {
        for j in 0..drive_dim {
            weights.push(((i * drive_dim + j + 1) as f32 * 0.37).sin() * 0.08);
        }
    }
    let bias: Vec<f32> = (0..drive_dim)
        .map(|j| ((j + 1) as f32 * 0.41).cos() * 0.015)
        .collect();
    SpikeGradBridge::from_weights(
        SPIKE_DIM, N_OSC, BRIDGE_DT, DRIVE_GAIN, ALPHA, FREQ_MIN, FREQ_MAX, W12, W13, W23,
        &weights, &bias,
    )
}

fn features_for(phase_base: f32) -> Vec<f32> {
    let mut f = vec![0.0f32; STEPS * SPIKE_DIM];
    for t in 0..STEPS {
        for i in 0..SPIKE_DIM {
            let phase = phase_base + t as f32 * (0.12 + i as f32 * 0.047);
            f[t * SPIKE_DIM + i] = 0.3 + 0.7 * (phase.sin() * 0.5 + 0.5);
        }
    }
    f
}

fn make_sequence(
    teacher_src: &TrainableSpikeSource,
    teacher_bridge: &SpikeGradBridge,
    phase_base: f32,
) -> Sequence {
    let features = features_for(phase_base);
    let (spikes, _) = teacher_src.forward(&features, STEPS);
    let (_, traj) = teacher_bridge.forward(&spikes, STEPS);
    let target_pst = t3spike_compute_pst_matrix(&traj);
    assert!(target_pst.iter().all(|x| x.is_finite()));
    Sequence {
        features,
        target_pst,
    }
}

fn pst_mse(actual: &[f32], target: &[f32]) -> f32 {
    assert_eq!(actual.len(), target.len());
    actual
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / actual.len() as f32
}

fn dataset_loss(
    source: &TrainableSpikeSource,
    bridge: &SpikeGradBridge,
    dataset: &[Sequence],
) -> f32 {
    dataset
        .iter()
        .map(|seq| {
            let (spikes, _) = source.forward(&seq.features, STEPS);
            let (_, traj) = bridge.forward(&spikes, STEPS);
            let pst = t3spike_compute_pst_matrix(&traj);
            pst_mse(&pst, &seq.target_pst)
        })
        .sum::<f32>()
        / dataset.len() as f32
}

/// Одна эпоха. `train_source`/`train_bridge` выбирают обучаемые части:
/// `(false, true)` — bridge-only, `(true, false)` — source-only (изоляция
/// вклада источника), `(true, true)` — end-to-end.
#[allow(clippy::too_many_arguments)]
fn train_epoch(
    source: &mut TrainableSpikeSource,
    bridge: &mut SpikeGradBridge,
    dataset: &[Sequence],
    pst_dim: usize,
    train_source: bool,
    train_bridge: bool,
    adam_step: u32,
    m_w: &mut [f32],
    v_w: &mut [f32],
    m_b: &mut [f32],
    v_b: &mut [f32],
    m_src_w: &mut [f32],
    v_src_w: &mut [f32],
    m_src_v: &mut [f32],
    v_src_v: &mut [f32],
) -> f32 {
    let mut grad_w = vec![0.0f32; bridge.spike_dim * bridge.drive_dim];
    let mut grad_b = vec![0.0f32; bridge.drive_dim];
    let mut grad_src_w = vec![0.0f32; source.w.len()];
    let mut grad_src_v = vec![0.0f32; source.v.len()];
    let mut loss_sum = 0.0f32;

    for seq in dataset {
        let (spikes, recs) = source.forward(&seq.features, STEPS);
        let (_, traj) = bridge.forward(&spikes, STEPS);
        let pst = t3spike_compute_pst_matrix(&traj);
        loss_sum += pst_mse(&pst, &seq.target_pst);

        let scale = 2.0 / pst.len() as f32 / dataset.len() as f32;
        let per_step: Vec<f32> = pst
            .iter()
            .zip(&seq.target_pst)
            .map(|(a, b)| scale * (a - b))
            .collect();
        let final_grad = vec![0.0f32; pst_dim];

        let (gw, gb, gspikes) = bridge.backward_full(&final_grad, &traj, &spikes, Some(&per_step));
        assert!(
            gspikes.iter().all(|x| x.is_finite()),
            "spike grad non-finite"
        );

        for (sum, x) in grad_w.iter_mut().zip(gw) {
            *sum += x;
        }
        for (sum, x) in grad_b.iter_mut().zip(gb) {
            *sum += x;
        }

        if train_source {
            let (dw, dv) = source.backward(&gspikes, STEPS, &recs);
            assert!(
                dw.iter().chain(&dv).all(|x| x.is_finite()),
                "source grad non-finite"
            );
            for (sum, x) in grad_src_w.iter_mut().zip(dw) {
                *sum += x;
            }
            for (sum, x) in grad_src_v.iter_mut().zip(dv) {
                *sum += x;
            }
        }
    }

    if train_bridge {
        for x in grad_w.iter_mut().chain(grad_b.iter_mut()) {
            *x = x.clamp(-GRAD_CLIP, GRAD_CLIP);
        }
        bridge.update_adam(
            &grad_w,
            &grad_b,
            ADAM_LR_BRIDGE,
            0.9,
            0.999,
            1e-8,
            adam_step,
            m_w,
            v_w,
            m_b,
            v_b,
        );
    }

    if train_source {
        source.update_adam(
            &grad_src_w,
            &grad_src_v,
            ADAM_LR_SOURCE,
            0.9,
            0.999,
            1e-8,
            adam_step,
            m_src_w,
            v_src_w,
            m_src_v,
            v_src_v,
        );
    }

    loss_sum / dataset.len() as f32
}

/// Конфигурация эксперимента.
#[derive(Clone, Copy, PartialEq)]
enum Config {
    /// Источник заморожен (w=v=1), мост с нуля — старая возможность.
    BridgeOnly,
    /// Мост = копия УЧИТЕЛЯ (заморожен), источник с нуля — изоляция вклада
    /// источника. Нулевой мост невозможен: W=0 ⇒ grad_spikes = Wᵀ·grad_drives = 0.
    SourceOnly,
    /// Источник + мост совместно с нуля — новая возможность.
    EndToEnd,
}

fn run_config(cfg: Config) -> (f32, f32, f32) {
    let (train_source, train_bridge, bridge_teacher_init) = match cfg {
        Config::BridgeOnly => (false, true, false),
        Config::SourceOnly => (true, false, true),
        Config::EndToEnd => (true, true, false),
    };
    let pst_dim = pst_linear_dim(N_OSC);
    let teacher_bridge = teacher_bridge();
    let teacher_src = TrainableSpikeSource::new_random(
        SPIKE_DIM,
        SRC_DT,
        SRC_DECAY,
        SRC_GAIN,
        SRC_INV_TAU,
        SRC_K_SCALE,
        7,
    );

    let train_set: Vec<Sequence> = (0..TRAIN_SEQUENCES)
        .map(|i| make_sequence(&teacher_src, &teacher_bridge, i as f32 * 0.61))
        .collect();
    let valid_set: Vec<Sequence> = (0..VALIDATION_SEQUENCES)
        .map(|i| make_sequence(&teacher_src, &teacher_bridge, 4.0 + i as f32 * 0.91))
        .collect();

    let mut source = TrainableSpikeSource::new(
        SPIKE_DIM,
        SRC_DT,
        SRC_DECAY,
        SRC_GAIN,
        SRC_INV_TAU,
        SRC_K_SCALE,
    );
    let mut bridge = if bridge_teacher_init {
        SpikeGradBridge::from_weights(
            SPIKE_DIM,
            N_OSC,
            BRIDGE_DT,
            DRIVE_GAIN,
            ALPHA,
            FREQ_MIN,
            FREQ_MAX,
            W12,
            W13,
            W23,
            &teacher_bridge.w,
            &teacher_bridge.b,
        )
    } else {
        SpikeGradBridge::new_zero(
            SPIKE_DIM, N_OSC, BRIDGE_DT, DRIVE_GAIN, ALPHA, FREQ_MIN, FREQ_MAX, W12, W13, W23,
        )
    };

    let mut m_w = vec![0.0; bridge.w.len()];
    let mut v_w = vec![0.0; bridge.w.len()];
    let mut m_b = vec![0.0; bridge.b.len()];
    let mut v_b = vec![0.0; bridge.b.len()];
    let mut m_src_w = vec![0.0; source.w.len()];
    let mut v_src_w = vec![0.0; source.w.len()];
    let mut m_src_v = vec![0.0; source.v.len()];
    let mut v_src_v = vec![0.0; source.v.len()];

    let baseline = dataset_loss(&source, &bridge, &valid_set);
    let mut last_valid = baseline;
    for epoch in 1..=EPOCHS {
        let train_loss = train_epoch(
            &mut source,
            &mut bridge,
            &train_set,
            pst_dim,
            train_source,
            train_bridge,
            epoch as u32,
            &mut m_w,
            &mut v_w,
            &mut m_b,
            &mut v_b,
            &mut m_src_w,
            &mut v_src_w,
            &mut m_src_v,
            &mut v_src_v,
        );
        if epoch % 30 == 0 || epoch == EPOCHS {
            last_valid = dataset_loss(&source, &bridge, &valid_set);
            println!(
                "   epoch={:>3} train_mse={:.8} validation_mse={:.8}",
                epoch, train_loss, last_valid
            );
        }
    }
    (
        baseline,
        last_valid,
        dataset_loss(&source, &bridge, &train_set),
    )
}

fn main() {
    println!("spikegrad_endtoend");
    println!(
        "source=TrainableSpikeSource(TransformerModel-math) n={SPIKE_DIM} steps={STEPS} osc={N_OSC} epochs={EPOCHS}"
    );
    println!("teacher: random source + fixed bridge → target PST");
    println!("три конфигурации: bridge-only / source-only / end-to-end");

    println!("\n── bridge-only (источник заморожен — старая возможность) ──");
    let (b0, v0, t0) = run_config(Config::BridgeOnly);
    println!("\n── source-only (мост = учитель, заморожен — вклад источника) ──");
    let (b2, v2, t2) = run_config(Config::SourceOnly);
    println!("\n── end-to-end (источник + мост, один градиент — новая) ──");
    let (b1, v1, t1) = run_config(Config::EndToEnd);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  bridge-only:  val {:.8} → {:.8}  (train {:.8})",
        b0, v0, t0
    );
    println!(
        "║  source-only:  val {:.8} → {:.8}  (train {:.8})",
        b2, v2, t2
    );
    println!(
        "║  end-to-end:   val {:.8} → {:.8}  (train {:.8})",
        b1, v1, t1
    );
    let gain = if v0 > 0.0 {
        100.0 * (v0 - v1) / v0
    } else {
        0.0
    };
    println!("║  улучшение end-to-end над bridge-only: {gain:.1}% на валидации");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("⚠️  Честная оговорка: конфигурации имеют разную ёмкость параметров");
    println!("   (bridge-only 30, source-only 64, end-to-end 94) — часть выигрыша от");
    println!("   добавленной ёмкости. source-only (мост-учитель) изолирует вклад");
    println!("   обучаемого источника: сколько может один источник при рабочем мосте.");
}
