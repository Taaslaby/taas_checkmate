//! Stateful SNN developmental control experiment.
//!
//! This is the SNN counterpart of `developmental_arm.rs`:
//!
//! ```text
//! proprioception → VSN hard spikes → Atlasov body → joint feedback
//! ```
//!
//! The four VSN neurons are the two agonist/antagonist motor pairs. Their
//! `surrogate` state is used only by a local surrogate learning rule (not full
//! BPTT/e-prop); the forward path sends hard spikes to the body whenever teacher
//! forcing is inactive. Three paired regimes separate pure
//! teacher forcing, scheduled teacher forcing, and scheduled teacher forcing
//! with an early co-contraction phase.
//!
//! Run:
//! ```text
//! cargo run -p pillars --example snn_developmental_arm --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;
use pillars::prelude::{VSNBuilder, VSNModel};

const BODY_DT: f32 = 0.001;
const NEURON_DT: f32 = 0.5;
const STEPS: usize = 240;
const TRAIN_EPISODES: usize = 80;
const EVAL_EPISODES: usize = 40;
const FEATURES: usize = 7;
const MOTORS: usize = 4;
const SEEDS: [u64; 3] = [0xD1CE, 0xA11CE, 0xC0FFEE];
// VSN needs a small tonic current to keep all motor neurons in the
// learnable threshold neighborhood. The sensor-dependent term remains
// trainable; this is analogous to developmental background drive, not a
// teacher action.
const TONIC_CURRENT: f32 = 35.0;
const SENSOR_GAIN: f32 = 2.5;

#[derive(Clone, Copy)]
enum Regime {
    Direct,
    AnnealedNoCocontraction,
    Developmental,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::AnnealedNoCocontraction => "annealed_no_cocontraction",
            Self::Developmental => "developmental",
        }
    }

    fn cocontraction(self) -> bool {
        matches!(self, Self::Developmental)
    }

    fn teacher_probability(self, episode: usize, train: bool) -> f32 {
        if !train {
            0.0
        } else if matches!(self, Self::Direct) {
            1.0
        } else {
            0.95 - 0.85 * episode as f32 / (TRAIN_EPISODES - 1) as f32
        }
    }
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.next() as f64 / u64::MAX as f64;
        lo + (hi - lo) * u as f32
    }
}

struct SnnPolicy {
    weights: [[f32; FEATURES]; MOTORS],
}

impl SnnPolicy {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weights = [[0.0; FEATURES]; MOTORS];
        for row in &mut weights {
            for w in row {
                *w = rng.uniform(-18.0, 18.0);
            }
        }
        Self { weights }
    }

    fn make_population(&self) -> pillars::NeuralPopulation<VSNModel> {
        VSNBuilder::new(MOTORS)
            .dt(NEURON_DT)
            .tau_m(10.0)
            .v_rest(-70.0)
            .v_thresh(-54.0)
            .v_reset(-70.0)
            .beta(0.9)
            .surrogate_gain(10.0)
            .build()
    }

    fn currents(&self, features: [f32; FEATURES]) -> [f32; MOTORS] {
        let mut currents = [0.0; MOTORS];
        for (i, row) in self.weights.iter().enumerate() {
            currents[i] = TONIC_CURRENT
                + SENSOR_GAIN * row.iter().zip(features).map(|(w, x)| w * x).sum::<f32>();
        }
        currents
    }

    /// Local surrogate update. The derivative is the VSN state's stored
    /// surrogate value from this timestep; forward output remains binary.
    fn learn(
        &mut self,
        features: [f32; FEATURES],
        target: [i32; MOTORS],
        spikes: &[i32],
        surrogate: &[f32],
    ) {
        for motor in 0..MOTORS {
            let error = target[motor] as f32 - spikes[motor] as f32;
            let local_grad = error * surrogate[motor].clamp(0.0, 1.0);
            for (weight, &feature) in self.weights[motor].iter_mut().zip(features.iter()) {
                *weight += 0.12 * local_grad * feature;
            }
            for w in &mut self.weights[motor] {
                *w = w.clamp(-80.0, 80.0);
            }
        }
    }
}

fn target_angle(t: usize, joint: usize, phase: f32) -> f32 {
    let x = 2.0 * std::f32::consts::PI * t as f32 / STEPS as f32 + phase;
    if joint == 0 {
        0.42 * x.sin()
    } else {
        0.32 * (x + 0.7).cos()
    }
}

fn target_velocity(t: usize, joint: usize, phase: f32) -> f32 {
    (target_angle(t + 1, joint, phase) - target_angle(t, joint, phase)) / BODY_DT
}

fn teacher_action(
    body: &FlyEmbodiedState,
    t: usize,
    phase: f32,
    cocontraction: bool,
    episode: usize,
) -> [i32; MOTORS] {
    let mut action = [0; MOTORS];
    for joint in 0..2 {
        let error = target_angle(t, joint, phase) - body.simulated_joint_qpos[joint];
        let velocity_error = target_velocity(t, joint, phase) - body.simulated_joint_qvel[joint];
        let torque = (28.0 * error + 0.8 * velocity_error).clamp(-1.0, 1.0);
        if torque > 0.05 {
            action[2 * joint + 1] = 1;
        } else if torque < -0.05 {
            action[2 * joint] = 1;
        }
        if cocontraction && episode < TRAIN_EPISODES / 3 && error.abs() < 0.18 {
            action[2 * joint] = 1;
            action[2 * joint + 1] = 1;
        }
    }
    action
}

fn mix(t: usize, episode: usize, probability: f32) -> bool {
    let x = ((t as u64 + 17 * episode as u64 + 13) * 1_103_515_245 + 12_345) % 1_000;
    x as f32 / 1_000.0 < probability
}

fn episode(
    policy: &mut SnnPolicy,
    regime: Regime,
    episode_index: usize,
    train: bool,
    initial_q: [f32; 2],
    phase: f32,
) -> (f32, f32, f32, f32, bool) {
    let mut body = FlyEmbodiedState::new(2, BODY_DT);
    body.simulated_joint_qpos.copy_from_slice(&initial_q);
    let mut neurons = policy.make_population();
    let mut spikes = vec![0i32; MOTORS];
    let resets = vec![0i32; MOTORS];
    let mut loss = 0.0;
    let mut stiffness = 0.0;
    let mut damping = 0.0;
    let mut spike_count = 0.0;

    for t in 0..STEPS {
        let target0 = target_angle(t, 0, phase);
        let target1 = target_angle(t, 1, phase);
        let features = [
            1.0,
            (target0 - body.simulated_joint_qpos[0]).clamp(-1.0, 1.0),
            body.simulated_joint_qvel[0].clamp(-2.0, 2.0) * 0.5,
            target0,
            (target1 - body.simulated_joint_qpos[1]).clamp(-1.0, 1.0),
            body.simulated_joint_qvel[1].clamp(-2.0, 2.0) * 0.5,
            target1,
        ];
        let currents = policy.currents(features);
        neurons.step(&currents, &mut spikes);
        let teacher = teacher_action(&body, t, phase, regime.cocontraction(), episode_index);
        let action = if mix(
            t,
            episode_index,
            regime.teacher_probability(episode_index, train),
        ) {
            teacher
        } else {
            [spikes[0], spikes[1], spikes[2], spikes[3]]
        };

        if train {
            let surrogate = neurons.state_slice(2).to_vec();
            policy.learn(features, teacher, &spikes, &surrogate);
        }
        body.step(&action, &resets);
        spike_count += spikes.iter().map(|&s| s as f32).sum::<f32>();

        for joint in 0..2 {
            let e = target_angle(t + 1, joint, phase) - body.simulated_joint_qpos[joint];
            loss += 0.5 * e * e;
        }
        stiffness += body.atlasov_state.joint_stiffness[..2].iter().sum::<f32>() * 0.5;
        damping += body.atlasov_state.joint_damping[..2].iter().sum::<f32>() * 0.5;
    }

    let final_error = (0..2)
        .map(|j| (target_angle(STEPS, j, phase) - body.simulated_joint_qpos[j]).powi(2))
        .sum::<f32>()
        .sqrt();
    (
        (loss / STEPS as f32).sqrt(),
        spike_count / STEPS as f32,
        stiffness / STEPS as f32,
        damping / STEPS as f32,
        final_error < 0.16,
    )
}

fn train_eval(regime: Regime, seed: u64) -> (f32, f32, f32, f32, f32) {
    let mut policy = SnnPolicy::new(seed);
    for episode_index in 0..TRAIN_EPISODES {
        let initial_q = [
            0.12 * (episode_index as f32 * 0.37).sin(),
            -0.10 * (episode_index as f32 * 0.23).cos(),
        ];
        let train_metrics = episode(
            &mut policy,
            regime,
            episode_index,
            true,
            initial_q,
            0.17 * episode_index as f32,
        );
        if episode_index + 1 == TRAIN_EPISODES {
            println!(
                "train regime={} rmse={:.5} spikes={:.2} stiffness={:.5} damping={:.5}",
                regime.name(),
                train_metrics.0,
                train_metrics.1,
                train_metrics.2,
                train_metrics.3
            );
        }
    }

    let mut sums = [0.0f32; 4];
    let mut successes = 0usize;
    for i in 0..EVAL_EPISODES {
        let result = episode(
            &mut policy,
            regime,
            TRAIN_EPISODES + i,
            false,
            [0.35 * (i as f32 * 1.7).sin(), 0.30 * (i as f32 * 1.1).cos()],
            0.31 * i as f32,
        );
        let (rmse, avg_spikes, avg_stiffness, avg_damping, success) = result;
        for (sum, value) in sums
            .iter_mut()
            .zip([rmse, avg_spikes, avg_stiffness, avg_damping])
        {
            *sum += value;
        }
        successes += usize::from(success);
    }
    let n = EVAL_EPISODES as f32;
    let result = (
        sums[0] / n,
        successes as f32 / n,
        sums[1] / n,
        sums[2] / n,
        sums[3] / n,
    );
    println!(
        "RESULT regime={} seed={} rmse={:.5} success={:.3} avg_spikes={:.2} avg_stiffness={:.5} avg_damping={:.5}",
        regime.name(), seed, result.0, result.1, result.2, result.3, result.4
    );
    result
}

fn main() {
    println!("stateful_snn_developmental_arm");
    println!("neuron=VSN states=[v,refr,surrogate] body=Atlasov joints=2");
    for regime in [
        Regime::Direct,
        Regime::AnnealedNoCocontraction,
        Regime::Developmental,
    ] {
        let all: Vec<_> = SEEDS.iter().map(|&seed| train_eval(regime, seed)).collect();
        let n = all.len() as f32;
        let means = [
            all.iter().map(|r| r.0).sum::<f32>() / n,
            all.iter().map(|r| r.1).sum::<f32>() / n,
            all.iter().map(|r| r.2).sum::<f32>() / n,
            all.iter().map(|r| r.3).sum::<f32>() / n,
            all.iter().map(|r| r.4).sum::<f32>() / n,
        ];
        assert!(means.iter().all(|value| value.is_finite()));
        assert!(
            means[2] > 0.0,
            "SNN evaluation produced no hard spikes; do not interpret this run"
        );
        let rmse_sd = (all.iter().map(|r| (r.0 - means[0]).powi(2)).sum::<f32>() / n).sqrt();
        let success_sd = (all.iter().map(|r| (r.1 - means[1]).powi(2)).sum::<f32>() / n).sqrt();
        println!(
            "SUMMARY regime={} rmse_mean={:.5} rmse_sd={:.5} success_mean={:.3} success_sd={:.3} avg_spikes={:.2} avg_stiffness={:.5} avg_damping={:.5}",
            regime.name(), means[0], rmse_sd, means[1], success_sd, means[2], means[3], means[4]
        );
    }
}
