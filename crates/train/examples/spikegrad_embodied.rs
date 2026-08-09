//! Multi-sequence bridge-level SPIGRAD benchmark with a real pillars VSN source.
//!
//! The benchmark separates three claims:
//!
//! 1. `SpikeGradBridge` performs true BPTT through differentiable T3 dynamics.
//!    The loss is computed over the complete per-step PST matrix.
//! 2. The bridge is trained on several deterministic VSN spike sequences and
//!    evaluated on held-out sequences, so the result is not one-trace fitting.
//! 3. The same unseen hard VSN spikes can drive Atlasov in forward mode.
//!    Atlasov is not used by the backward pass because `FlyEmbodiedState` has
//!    no differentiable backward API yet.
//!
//! VSN weights are fixed. This is not end-to-end gradient learning through the
//! discrete VSN spike function.
//!
//! Run with:
//! ```text
//! cargo run -p taas-train --features t3 --example spikegrad_embodied --release
//! ```

#![cfg(feature = "t3")]

use pillars::api::embodied::FlyEmbodiedState;
use pillars::prelude::VSNBuilder;
use taas_train::spike::t3spike_compute_pst_matrix;
use taas_train::spike_grad::SpikeGradBridge;
use taas_train::t3::pst_linear_dim;

const STEPS: usize = 64;
const SPIKE_DIM: usize = 4;
const N_OSC: usize = 2;
const TRAIN_SEQUENCES: usize = 6;
const VALIDATION_SEQUENCES: usize = 4;
const EPOCHS: usize = 120;
const BRIDGE_DT: f32 = 0.05;
const DRIVE_GAIN: f32 = 10.0;
const ALPHA: f32 = 1.0;
const FREQ_MIN: f32 = 0.5;
const FREQ_MAX: f32 = 3.0;
const W12: f32 = 0.3;
const W13: f32 = 0.15;
const W23: f32 = 0.08;
const ADAM_LR: f32 = 0.01;
const GRAD_CLIP: f32 = 0.25;

struct Sequence {
    spikes: Vec<f32>,
    target_pst: Vec<f32>,
    spike_count: usize,
}

fn teacher_weights() -> (Vec<f32>, Vec<f32>) {
    let drive_dim = 3 * N_OSC;
    let mut weights = Vec::with_capacity(SPIKE_DIM * drive_dim);
    for i in 0..SPIKE_DIM {
        for j in 0..drive_dim {
            weights.push(((i * drive_dim + j + 1) as f32 * 0.37).sin() * 0.08);
        }
    }
    let bias = (0..drive_dim)
        .map(|j| ((j + 1) as f32 * 0.41).cos() * 0.015)
        .collect();
    (weights, bias)
}

fn hard_vsn_spikes(phase_offset: f32, rate_scale: f32) -> Vec<f32> {
    let mut population = VSNBuilder::new(SPIKE_DIM)
        .dt(0.5)
        .tau_m(10.0)
        .v_rest(-70.0)
        .v_thresh(-54.0)
        .v_reset(-70.0)
        .tau_ref(2.0)
        .beta(0.9)
        .surrogate_gain(10.0)
        .build();
    let mut currents = vec![0.0f32; SPIKE_DIM];
    let mut spikes = vec![0i32; SPIKE_DIM];
    let mut sequence = Vec::with_capacity(STEPS * SPIKE_DIM);

    for t in 0..STEPS {
        for (i, current) in currents.iter_mut().enumerate() {
            let phase = phase_offset + t as f32 * (0.11 + i as f32 * 0.037) * rate_scale;
            *current = 35.0 + 78.0 * (phase.sin() * 0.5 + 0.5);
        }
        population.step(&currents, &mut spikes);
        sequence.extend(spikes.iter().map(|&spike| spike as f32));
    }

    let total_spikes = sequence.iter().filter(|&&spike| spike > 0.5).count();
    assert!(total_spikes > 0, "VSN source produced no hard spikes");
    assert!(sequence.iter().all(|value| *value == 0.0 || *value == 1.0));
    sequence
}

fn make_sequence(teacher: &SpikeGradBridge, phase_offset: f32, rate_scale: f32) -> Sequence {
    let spikes = hard_vsn_spikes(phase_offset, rate_scale);
    let (_, trajectory) = teacher.forward(&spikes, STEPS);
    let target_pst = t3spike_compute_pst_matrix(&trajectory);
    assert!(target_pst.iter().all(|value| value.is_finite()));
    Sequence {
        spike_count: spikes.iter().filter(|&&spike| spike > 0.5).count(),
        spikes,
        target_pst,
    }
}

fn pst_mse(actual: &[f32], target: &[f32]) -> f32 {
    assert_eq!(actual.len(), target.len());
    actual
        .iter()
        .zip(target)
        .map(|(actual, target)| (actual - target).powi(2))
        .sum::<f32>()
        / actual.len() as f32
}

fn zero_baseline(student: &SpikeGradBridge, dataset: &[Sequence]) -> f32 {
    let losses: Vec<f32> = dataset
        .iter()
        .map(|sequence| {
            let (_, trajectory) = student.forward(&sequence.spikes, STEPS);
            let pst = t3spike_compute_pst_matrix(&trajectory);
            pst_mse(&pst, &sequence.target_pst)
        })
        .collect();
    losses.iter().sum::<f32>() / losses.len() as f32
}

fn dataset_loss(student: &SpikeGradBridge, dataset: &[Sequence]) -> f32 {
    zero_baseline(student, dataset)
}

fn train_epoch(
    student: &mut SpikeGradBridge,
    dataset: &[Sequence],
    pst_dim: usize,
    m_w: &mut [f32],
    v_w: &mut [f32],
    m_b: &mut [f32],
    v_b: &mut [f32],
    adam_step: u32,
) -> f32 {
    let mut grad_w = vec![0.0f32; student.spike_dim * student.drive_dim];
    let mut grad_b = vec![0.0f32; student.drive_dim];
    let mut loss_sum = 0.0;
    let mut gradient_norm_sq = 0.0;

    for sequence in dataset {
        let (_, trajectory) = student.forward(&sequence.spikes, STEPS);
        let student_pst = t3spike_compute_pst_matrix(&trajectory);
        loss_sum += pst_mse(&student_pst, &sequence.target_pst);

        let scale = 2.0 / student_pst.len() as f32 / dataset.len() as f32;
        let per_step_grad: Vec<f32> = student_pst
            .iter()
            .zip(&sequence.target_pst)
            .map(|(actual, target)| scale * (actual - target))
            .collect();
        let final_grad = vec![0.0f32; pst_dim];
        let (sequence_grad_w, sequence_grad_b) = student.backward(
            &final_grad,
            &trajectory,
            &sequence.spikes,
            Some(&per_step_grad),
        );

        for (sum, value) in grad_w.iter_mut().zip(sequence_grad_w) {
            *sum += value;
        }
        for (sum, value) in grad_b.iter_mut().zip(sequence_grad_b) {
            *sum += value;
        }
    }

    assert!(grad_w.iter().all(|value| value.is_finite()));
    assert!(grad_b.iter().all(|value| value.is_finite()));
    let gradient_has_signal = grad_w
        .iter()
        .chain(grad_b.iter())
        .any(|value| value.abs() > 1.0e-10);
    if !gradient_has_signal {
        eprintln!("warning: current batch produced an identically zero gradient");
    }
    for value in grad_w.iter_mut().chain(grad_b.iter_mut()) {
        *value = value.clamp(-GRAD_CLIP, GRAD_CLIP);
        gradient_norm_sq += *value * *value;
    }

    student.update_adam(
        &grad_w, &grad_b, ADAM_LR, 0.9, 0.999, 1.0e-8, adam_step, m_w, v_w, m_b, v_b,
    );

    assert!(gradient_norm_sq.is_finite());
    loss_sum / dataset.len() as f32
}

fn atlasov_forward_sanity(sequence: &Sequence) {
    let mut body = FlyEmbodiedState::new(2, 0.001);
    let resets = vec![0i32; body.n_muscles];
    let mut stiffness_sum = 0.0;
    let mut damping_sum = 0.0;
    let mut active_steps = 0usize;

    for frame in sequence.spikes.chunks_exact(SPIKE_DIM) {
        let mut motor_spikes = vec![0i32; body.n_muscles];
        for (dst, &source) in motor_spikes.iter_mut().take(SPIKE_DIM).zip(frame) {
            *dst = source as i32;
        }
        let output = body.step(&motor_spikes, &resets);
        stiffness_sum += output.avg_joint_stiffness;
        damping_sum += output.avg_joint_damping;
        active_steps += usize::from(output.active_muscles > 0);
    }

    assert!(body
        .simulated_joint_qpos
        .iter()
        .all(|value| value.is_finite()));
    assert!(stiffness_sum.is_finite() && damping_sum.is_finite());
    assert!(
        stiffness_sum > 0.0,
        "Atlasov received no effective activation"
    );
    println!(
        "atlasov_validation hard_spikes={} active_steps={}/{} mean_stiffness={:.5} mean_damping={:.5} q=[{:.5}, {:.5}]",
        sequence.spike_count,
        active_steps,
        STEPS,
        stiffness_sum / STEPS as f32,
        damping_sum / STEPS as f32,
        body.simulated_joint_qpos[0],
        body.simulated_joint_qpos[1]
    );
}

fn main() {
    let pst_dim = pst_linear_dim(N_OSC);
    let (teacher_w, teacher_b) = teacher_weights();
    let teacher = SpikeGradBridge::from_weights(
        SPIKE_DIM, N_OSC, BRIDGE_DT, DRIVE_GAIN, ALPHA, FREQ_MIN, FREQ_MAX, W12, W13, W23,
        &teacher_w, &teacher_b,
    );

    let train_dataset: Vec<Sequence> = (0..TRAIN_SEQUENCES)
        .map(|index| make_sequence(&teacher, index as f32 * 0.61, 0.97 + index as f32 * 0.012))
        .collect();
    // Held-out regime: deliberately disjoint phase/rate ranges from training.
    // Because the sensor drive is periodic, this is a held-out parameter
    // regime, not a claim of globally novel behavior.
    let validation_dataset: Vec<Sequence> = (0..VALIDATION_SEQUENCES)
        .map(|index| {
            make_sequence(
                &teacher,
                4.0 + index as f32 * 0.91,
                1.08 + index as f32 * 0.017,
            )
        })
        .collect();

    println!("spikegrad_generalization_benchmark");
    println!(
        "source=VSN train_sequences={} validation_sequences={} steps={} pst_dim={}",
        train_dataset.len(),
        validation_dataset.len(),
        STEPS,
        pst_dim
    );
    println!(
        "train_hard_spikes={} validation_hard_spikes={}",
        train_dataset
            .iter()
            .map(|sequence| sequence.spike_count)
            .sum::<usize>(),
        validation_dataset
            .iter()
            .map(|sequence| sequence.spike_count)
            .sum::<usize>()
    );

    let mut student = SpikeGradBridge::new_zero(
        SPIKE_DIM, N_OSC, BRIDGE_DT, DRIVE_GAIN, ALPHA, FREQ_MIN, FREQ_MAX, W12, W13, W23,
    );
    let train_baseline = zero_baseline(&student, &train_dataset);
    let validation_baseline = zero_baseline(&student, &validation_dataset);
    println!(
        "BASELINE train_pst_mse={:.8} validation_pst_mse={:.8}",
        train_baseline, validation_baseline
    );

    let mut m_w = vec![0.0f32; student.w.len()];
    let mut v_w = vec![0.0f32; student.w.len()];
    let mut m_b = vec![0.0f32; student.b.len()];
    let mut v_b = vec![0.0f32; student.b.len()];
    for epoch in 1..=EPOCHS {
        let train_loss = train_epoch(
            &mut student,
            &train_dataset,
            pst_dim,
            &mut m_w,
            &mut v_w,
            &mut m_b,
            &mut v_b,
            epoch as u32,
        );
        if epoch == 1 || epoch % 20 == 0 || epoch == EPOCHS {
            let validation_loss = dataset_loss(&student, &validation_dataset);
            println!(
                "epoch={} train_pst_mse={:.8} validation_pst_mse={:.8}",
                epoch, train_loss, validation_loss
            );
        }
    }

    let train_final = dataset_loss(&student, &train_dataset);
    let validation_final = dataset_loss(&student, &validation_dataset);
    let validation_improvement =
        100.0 * (validation_baseline - validation_final) / validation_baseline.max(1.0e-12);
    let gap = validation_final - train_final;
    assert!(train_final.is_finite() && validation_final.is_finite());
    assert!(student
        .w
        .iter()
        .chain(student.b.iter())
        .all(|value| value.is_finite()));
    println!(
        "RESULT train_initial={:.8} train_final={:.8} validation_initial={:.8} validation_final={:.8} validation_improvement={:.2}% generalization_gap={:.8}",
        train_baseline,
        train_final,
        validation_baseline,
        validation_final,
        validation_improvement,
        gap
    );
    atlasov_forward_sanity(&validation_dataset[0]);
}
