//! Developmental embodied control experiment.
//!
//! Compares two controllers with the same online imitation budget:
//!
//! - `direct`: pure expert forcing during training;
//! - `annealed_no_cocontraction`: scheduled teacher forcing without co-contraction;
//! - `developmental`: the same schedule plus early antagonist co-contraction.
//!
//! This is a body/control harness, not yet a surrogate-gradient or SPIGRAD
//! benchmark: the policy is intentionally a stateless hard-threshold model
//! with an online delta rule. Results must not be interpreted as evidence that
//! developmental learning beats SNN backpropagation.
//!
//! The policy is deliberately small and interpretable: a four-channel
//! threshold spiking motor layer (two agonist/antagonist pairs) with a
//! trainable linear readout. The body is the canonical
//! `pillars::api::embodied::FlyEmbodiedState`, including Atlasov calcium,
//! force, stiffness, damping, and joint integration.
//!
//! Run:
//! ```text
//! cargo run -p pillars --example developmental_arm --release
//! ```

use pillars::api::embodied::FlyEmbodiedState;

const DT: f32 = 0.001;
const STEPS: usize = 240;
const TRAIN_EPISODES: usize = 80;
const EVAL_EPISODES: usize = 40;
const SEEDS: [u64; 3] = [0xD1CE, 0xA11CE, 0xC0FFEE];
const FEATURES: usize = 7; // bias + (error, velocity, target) for each joint
const MOTORS: usize = 4; // negative torque, positive torque for joints 0 and 1

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
}

/// Tiny deterministic RNG so the benchmark has no external data dependency.
#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() as f64 / u64::MAX as f64) as f32;
        lo + (hi - lo) * unit
    }
}

#[derive(Clone)]
struct SpikingPolicy {
    weights: [[f32; FEATURES]; MOTORS],
}

impl SpikingPolicy {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weights = [[0.0; FEATURES]; MOTORS];
        for row in &mut weights {
            for w in row {
                *w = rng.uniform(-0.08, 0.08);
            }
        }
        Self { weights }
    }

    fn sigmoid(x: f32) -> f32 {
        // The experiment keeps logits in a safe range; this clamp avoids a
        // numerical corner from dominating a long developmental run.
        (1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())).clamp(1e-5, 1.0 - 1e-5)
    }

    fn predict(&self, features: [f32; FEATURES]) -> [i32; MOTORS] {
        let mut spikes = [0; MOTORS];
        for (motor, row) in self.weights.iter().enumerate() {
            let logit = row.iter().zip(features).map(|(w, x)| w * x).sum::<f32>();
            spikes[motor] = i32::from(Self::sigmoid(logit) >= 0.5);
        }
        spikes
    }

    /// Online delta rule on the expert's motor target. This is a local,
    /// rate-coded learning rule; the emitted action remains hard spikes.
    fn learn(&mut self, features: [f32; FEATURES], target: [i32; MOTORS], lr: f32) {
        for (motor, row) in self.weights.iter_mut().enumerate() {
            let logit = row.iter().zip(features).map(|(w, x)| *w * x).sum::<f32>();
            let probability = Self::sigmoid(logit);
            let error = target[motor] as f32 - probability;
            for (w, x) in row.iter_mut().zip(features) {
                *w += lr * error * x;
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
    let h = 1;
    (target_angle(t + h, joint, phase) - target_angle(t, joint, phase)) / DT
}

/// Expert torque is a bounded PD teacher. Atlasov's joint convention is
/// `delta_f = extensor_tension - flexor_tension`, hence motor 1/3 are the
/// positive-torque channels and motor 0/2 are the negative channels.
fn expert_action(
    body: &FlyEmbodiedState,
    t: usize,
    phase: f32,
    cocontraction: bool,
    episode: usize,
) -> [i32; MOTORS] {
    let mut action = [0; MOTORS];
    for j in 0..2 {
        let target = target_angle(t, j, phase);
        let target_v = target_velocity(t, j, phase);
        let error = target - body.simulated_joint_qpos[j];
        let velocity_error = target_v - body.simulated_joint_qvel[j];
        let torque = (28.0 * error + 0.8 * velocity_error).clamp(-1.0, 1.0);
        if torque > 0.05 {
            action[2 * j + 1] = 1;
        } else if torque < -0.05 {
            action[2 * j] = 1;
        }

        // During the first developmental phase, simultaneous antagonist
        // activity teaches the body to build stiffness before fine control.
        if cocontraction && episode < TRAIN_EPISODES / 3 && error.abs() < 0.18 {
            action[2 * j] = 1;
            action[2 * j + 1] = 1;
        }
    }
    action
}

fn run_episode(
    policy: &mut SpikingPolicy,
    regime: Regime,
    episode: usize,
    train: bool,
    initial_q: [f32; 2],
    phase: f32,
) -> (f32, f32, f32, usize, bool) {
    let cocontraction = matches!(regime, Regime::Developmental);
    let mut body = FlyEmbodiedState::new(2, DT);
    body.simulated_joint_qpos.copy_from_slice(&initial_q);
    body.simulated_joint_qvel.fill(0.0);

    let mut squared_error = 0.0f32;
    let mut stiffness_sum = 0.0f32;
    let mut damping_sum = 0.0f32;
    let mut spikes_total = 0usize;
    let mut resets = [0i32; 4];

    for t in 0..STEPS {
        let teacher = expert_action(&body, t, phase, cocontraction, episode);
        // Keep joint identity explicit. A shared aggregate error would make
        // the two-joint task underdetermined and would invalidate the motor
        // comparison.
        let policy_features = [
            1.0,
            (target_angle(t, 0, phase) - body.simulated_joint_qpos[0]).clamp(-1.0, 1.0),
            body.simulated_joint_qvel[0].clamp(-2.0, 2.0) * 0.5,
            target_angle(t, 0, phase),
            (target_angle(t, 1, phase) - body.simulated_joint_qpos[1]).clamp(-1.0, 1.0),
            body.simulated_joint_qvel[1].clamp(-2.0, 2.0) * 0.5,
            target_angle(t, 1, phase),
        ];
        let predicted = policy.predict(policy_features);

        // Three-way control: direct is pure teacher forcing; the two
        // annealed regimes share the same scheduled-sampling schedule. This
        // separates schedule effects from antagonist co-contraction effects.
        let teacher_prob = if !train {
            0.0
        } else if matches!(regime, Regime::Direct) {
            1.0
        } else {
            0.95 - 0.85 * (episode as f32 / (TRAIN_EPISODES - 1) as f32)
        };
        let use_teacher = train && deterministic_mix(t, episode, teacher_prob);
        let action = if use_teacher { teacher } else { predicted };

        if train {
            policy.learn(policy_features, teacher, 0.04);
        }

        body.step(&action, &resets);
        resets.fill(0);
        spikes_total += action.iter().map(|&s| s as usize).sum::<usize>();
        let mut step_error = 0.0;
        for j in 0..2 {
            let e = target_angle(t + 1, j, phase) - body.simulated_joint_qpos[j];
            step_error += e * e;
        }
        squared_error += step_error * 0.5;
        stiffness_sum += body.atlasov_state.joint_stiffness[..2].iter().sum::<f32>() * 0.5;
        damping_sum += body.atlasov_state.joint_damping[..2].iter().sum::<f32>() * 0.5;
    }

    let rmse = (squared_error / STEPS as f32).sqrt();
    let avg_stiffness = stiffness_sum / STEPS as f32;
    let avg_damping = damping_sum / STEPS as f32;
    let final_error = (0..2)
        .map(|j| (target_angle(STEPS, j, phase) - body.simulated_joint_qpos[j]).powi(2))
        .sum::<f32>()
        .sqrt();
    (
        rmse,
        avg_stiffness,
        avg_damping,
        spikes_total,
        final_error < 0.16,
    )
}

/// Deterministic Bernoulli-like schedule without RNG state in the hot loop.
fn deterministic_mix(t: usize, episode: usize, probability: f32) -> bool {
    let x = ((t as u64 + 17 * episode as u64 + 13) * 1_103_515_245 + 12_345) % 1_000;
    x as f32 / 1_000.0 < probability
}

fn train_and_evaluate(regime: Regime, seed: u64) -> (f32, f32, f32, f32, f32) {
    let mut policy = SpikingPolicy::new(seed);
    for episode in 0..TRAIN_EPISODES {
        let phase = 0.17 * episode as f32;
        let initial_q = [
            0.12 * (episode as f32 * 0.37).sin(),
            -0.10 * (episode as f32 * 0.23).cos(),
        ];
        let metrics = run_episode(&mut policy, regime, episode, true, initial_q, phase);
        if episode + 1 == TRAIN_EPISODES {
            println!(
                "train regime={} rmse={:.5} spikes={} stiffness={:.5} damping={:.5}",
                regime.name(),
                metrics.0,
                metrics.3,
                metrics.1,
                metrics.2
            );
        }
    }

    let mut rmse = 0.0;
    let mut stiffness = 0.0;
    let mut damping = 0.0;
    let mut success = 0usize;
    let mut spikes = 0usize;
    for episode in 0..EVAL_EPISODES {
        let initial_q = [
            0.35 * (episode as f32 * 1.7).sin(),
            0.30 * (episode as f32 * 1.1).cos(),
        ];
        let result = run_episode(
            &mut policy,
            regime,
            TRAIN_EPISODES + episode,
            false,
            initial_q,
            0.31 * episode as f32,
        );
        rmse += result.0;
        stiffness += result.1;
        damping += result.2;
        spikes += result.3;
        success += usize::from(result.4);
    }
    let n = EVAL_EPISODES as f32;
    let eval_rmse = rmse / n;
    let success_rate = success as f32 / n;
    let avg_stiffness = stiffness / n;
    let avg_damping = damping / n;
    println!(
        "RESULT regime={} seed={} eval_rmse={:.5} success={}/{} success_rate={:.3} avg_spikes={:.1} avg_stiffness={:.5} avg_damping={:.5}",
        regime.name(),
        seed,
        eval_rmse,
        success,
        EVAL_EPISODES,
        success_rate,
        spikes as f32 / n,
        avg_stiffness,
        avg_damping
    );
    (
        eval_rmse,
        success_rate,
        spikes as f32 / n,
        avg_stiffness,
        avg_damping,
    )
}

fn main() {
    println!("developmental_embodied_arm");
    println!("body=pillars::api::embodied::FlyEmbodiedState atlasov=true joints=2 dt={DT}");
    println!("train_episodes={TRAIN_EPISODES} eval_episodes={EVAL_EPISODES} steps={STEPS}");
    // Same seeds are used for paired runs; aggregate across seeds below.
    // Direct is the pure-teacher reference, while the two annealed regimes
    // are the causal comparison for curriculum/co-contraction.
    for regime in [
        Regime::Direct,
        Regime::AnnealedNoCocontraction,
        Regime::Developmental,
    ] {
        let mut metrics = Vec::new();
        for &seed in &SEEDS {
            metrics.push(train_and_evaluate(regime, seed));
        }
        let n = metrics.len() as f32;
        let mean_rmse = metrics.iter().map(|m| m.0).sum::<f32>() / n;
        let mean_success = metrics.iter().map(|m| m.1).sum::<f32>() / n;
        let mean_spikes = metrics.iter().map(|m| m.2).sum::<f32>() / n;
        let mean_stiffness = metrics.iter().map(|m| m.3).sum::<f32>() / n;
        let mean_damping = metrics.iter().map(|m| m.4).sum::<f32>() / n;
        let sd_rmse = (metrics
            .iter()
            .map(|m| (m.0 - mean_rmse).powi(2))
            .sum::<f32>()
            / n)
            .sqrt();
        let sd_success = (metrics
            .iter()
            .map(|m| (m.1 - mean_success).powi(2))
            .sum::<f32>()
            / n)
            .sqrt();
        println!(
            "SUMMARY regime={} rmse_mean={:.5} rmse_sd={:.5} success_mean={:.3} success_sd={:.3} avg_spikes={:.1} avg_stiffness={:.5} avg_damping={:.5}",
            regime.name(),
            mean_rmse,
            sd_rmse,
            mean_success,
            sd_success,
            mean_spikes,
            mean_stiffness,
            mean_damping
        );
    }
}
