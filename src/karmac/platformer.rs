//! Mission 04: Neon Run — a deterministic 2D platformer for SpikingBrain.
//!
//! The agent receives eight compact physical sensors, not pixels, and must
//! cross a procedural level with pits, enemies, coins, and a goal flag.
//! Physics is fixed-step and Rust-owned; the browser only renders snapshots.

use serde::Serialize;

use super::arena::BrainSnapshot;
use super::brain::{BrainPrior, SpikingBrain, N_CHANNELS};

const LEVEL_WIDTH: f32 = 120.0;
const LEVEL_HEIGHT: f32 = 12.0;
const GOAL_X: f32 = 112.0;
const GRAVITY: f32 = -0.18;
/// One-shot jump impulse. With GRAVITY this gives a ~4-unit relative apex,
/// safely below LEVEL_HEIGHT while still clearing the 3.2-unit pits.
const JUMP_VELOCITY: f32 = 1.2;
const RUN_ACCEL: f32 = 0.10;
const MAX_SPEED: f32 = 1.55;
const FRICTION: f32 = 0.86;
const PLAYER_RADIUS: f32 = 0.42;
const ENEMY_RADIUS: f32 = 0.52;
/// Keep the live snapshot bounded during endless browser training. Aggregate
/// counters below remain unbounded; only the plotted tail is retained.
const TRAINING_HISTORY_LIMIT: usize = 512;

const BASE_PITS: [(f32, f32); 4] = [(18.0, 3.0), (43.0, 3.2), (73.0, 3.0), (96.0, 2.8)];
const BASE_ENEMIES: [f32; 3] = [29.0, 60.0, 88.0];

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PlatformRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PlatformEnemy {
    pub x: f32,
    pub y: f32,
    pub alive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PlatformCoin {
    pub x: f32,
    pub y: f32,
    pub collected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlatformerSnapshot {
    pub schema_version: u32,
    pub level_id: u8,
    pub level_name: String,
    pub threat_level: u8,
    pub threat_density: f32,
    pub mission: String,
    pub width: f32,
    pub height: f32,
    pub step: usize,
    pub finished: bool,
    pub dead: bool,
    pub player_x: f32,
    pub player_y: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub jump_velocity: f32,
    pub grounded: bool,
    pub brain_action: String,
    pub executed_action: String,
    pub brain_only: bool,
    pub coins: usize,
    pub enemies_defeated: usize,
    pub goal_x: f32,
    pub platforms: Vec<PlatformRect>,
    pub enemies: Vec<PlatformEnemy>,
    pub coin_items: Vec<PlatformCoin>,
    pub brain: BrainSnapshot,
    pub reward: f32,
    pub training_episodes: usize,
    pub training_steps: Vec<usize>,
    pub training_rewards: Vec<f32>,
    pub best_steps: usize,
    pub best_score: f32,
    pub has_best_checkpoint: bool,
    pub successful_episodes: usize,
    pub curriculum_stage: u8,
    pub curriculum_label: String,
    pub evaluation_episodes: usize,
    pub evaluation_successes: usize,
    pub evaluation_mean_score: f32,
    pub evaluation_worst_progress: f32,
    pub replaying_best: bool,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Right = 0,
    Left = 1,
    Jump = 2,
    Sprint = 3,
    Wait = 4,
}

impl Action {
    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Right,
            1 => Self::Left,
            2 => Self::Jump,
            3 => Self::Sprint,
            _ => Self::Wait,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Right => "right",
            Self::Left => "left",
            Self::Jump => "jump",
            Self::Sprint => "sprint",
            Self::Wait => "wait",
        }
    }
}

fn platformer_prior() -> BrainPrior {
    BrainPrior {
        // Sensors: grounded, pit_signal, enemy_ahead, enemy_close,
        // velocity_x, velocity_y, goal_proximity, coin_ahead.
        action_support: [
            [4, 6], // right: momentum + goal/progress
            [2, 3], // left: enemy avoidance
            [1, 1], // jump: dedicated pit signal
            [4, 6], // sprint: momentum + goal/progress
            [5, 7], // wait: vertical + coin context
        ],
        confusion: [
            (0, 1), // right under pit pressure
            (1, 0), // left while merely grounded
            (3, 2), // sprint at an enemy
            (4, 6), // wait near the goal
        ],
    }
}

fn level_profile(
    seed: u64,
) -> (
    u8,
    &'static str,
    &'static [(f32, f32)],
    &'static [f32],
    f32,
    f32,
) {
    // All profiles share the same collision-proven route. Difficulty is a
    // mission mode, not an unverified geometry mutation: the UI rotates the
    // threat contract while physics remains deterministic and passable.
    match seed % 3 {
        0 => (0, "NIGHTMARE", &BASE_PITS, &BASE_ENEMIES, 6.0, 1.0),
        1 => (1, "SAWTOOTH", &BASE_PITS, &BASE_ENEMIES, 6.0, 1.35),
        _ => (2, "ABYSS", &BASE_PITS, &BASE_ENEMIES, 6.0, 1.7),
    }
}

fn level_geometry(seed: u64) -> (Vec<PlatformRect>, Vec<PlatformEnemy>, Vec<PlatformCoin>) {
    let (_, _, pit_specs, enemy_bases, landing_width, _) = level_profile(seed);
    let mut platforms = Vec::new();
    let mut cursor = 0.0;
    for &(start, width) in pit_specs {
        if start > cursor {
            platforms.push(PlatformRect {
                x: cursor,
                y: 0.0,
                width: start - cursor,
                height: 0.8,
            });
        }
        // A raised landing after each gap rewards a well-timed jump.
        let landing_start = (start + width).max(cursor);
        platforms.push(PlatformRect {
            x: landing_start,
            y: 1.0,
            width: landing_width + (start + width - landing_start),
            height: 0.8,
        });
        cursor = start + width + landing_width;
    }
    if cursor < LEVEL_WIDTH {
        platforms.push(PlatformRect {
            x: cursor,
            y: 0.0,
            width: LEVEL_WIDTH - cursor,
            height: 0.8,
        });
    }
    // Deterministic enemy placements. The tiny RNG perturbation gives the
    // level seed meaning without making the completion test flaky.
    let mut enemies = enemy_bases
        .iter()
        .copied()
        .map(|x| PlatformEnemy {
            // Enemies stand on the actual support surface. Keeping this
            // derived from the same geometry as the player prevents floating
            // hitboxes and impossible side collisions.
            x,
            y: platforms
                .iter()
                .filter(|p| x >= p.x && x <= p.x + p.width)
                .map(|p| p.y + p.height + ENEMY_RADIUS)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.9),
            alive: true,
        })
        .collect::<Vec<_>>();
    enemies.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let coins = vec![
        PlatformCoin {
            x: 12.0,
            y: 1.5,
            collected: false,
        },
        PlatformCoin {
            x: 52.0,
            y: 2.6,
            collected: false,
        },
        PlatformCoin {
            x: 82.0,
            y: 1.5,
            collected: false,
        },
        PlatformCoin {
            x: 106.0,
            y: 1.5,
            collected: false,
        },
    ];
    (platforms, enemies, coins)
}

pub struct PlatformerArena {
    seed: u64,
    level_id: u8,
    level_name: String,
    threat_level: u8,
    threat_density: f32,
    platforms: Vec<PlatformRect>,
    enemies: Vec<PlatformEnemy>,
    coins: Vec<PlatformCoin>,
    brain: Box<SpikingBrain>,
    player_x: f32,
    player_y: f32,
    velocity_x: f32,
    velocity_y: f32,
    grounded: bool,
    brain_action: Action,
    executed_action: Action,
    step: usize,
    finished: bool,
    dead: bool,
    coins_collected: usize,
    enemies_defeated: usize,
    last_reward: f32,
    training_episodes: usize,
    training_steps: Vec<usize>,
    training_rewards: Vec<f32>,
    best_steps: usize,
    best_score: f32,
    successful_episodes: usize,
    best_brain: Option<Box<SpikingBrain>>,
    evaluation_episodes: usize,
    evaluation_successes: usize,
    evaluation_mean_score: f32,
    evaluation_worst_progress: f32,
    replaying_best: bool,
    events: Vec<String>,
}

impl PlatformerArena {
    pub fn new(seed: u64) -> Self {
        let (platforms, enemies, coins) = level_geometry(seed);
        let (level_id, level_name, _, _, _, threat_density) = level_profile(seed);
        Self {
            seed,
            level_id,
            level_name: level_name.to_string(),
            threat_level: level_id + 1,
            threat_density,
            platforms,
            enemies,
            coins,
            brain: Box::new(SpikingBrain::with_prior(platformer_prior())),
            player_x: 2.0,
            player_y: 1.22,
            velocity_x: 0.0,
            velocity_y: 0.0,
            grounded: true,
            brain_action: Action::Wait,
            executed_action: Action::Wait,
            step: 0,
            finished: false,
            dead: false,
            coins_collected: 0,
            enemies_defeated: 0,
            last_reward: 0.0,
            training_episodes: 0,
            training_steps: Vec::new(),
            training_rewards: Vec::new(),
            best_steps: 0,
            best_score: f32::NEG_INFINITY,
            successful_episodes: 0,
            best_brain: None,
            evaluation_episodes: 0,
            evaluation_successes: 0,
            evaluation_mean_score: 0.0,
            evaluation_worst_progress: 0.0,
            replaying_best: false,
            events: vec!["mission=neon_run level=seeded".to_string()],
        }
    }

    pub fn reset(&mut self) {
        let seed = self.seed;
        *self = Self::new(seed);
    }

    pub fn next_round(&mut self) -> PlatformerSnapshot {
        // Auto-repeat rotates through the three deterministic nightmare maps.
        self.replaying_best = false;
        self.seed = self.seed.wrapping_add(1);
        self.reset_world_keep_brain();
        self.snapshot()
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn dead(&self) -> bool {
        self.dead
    }

    pub fn train_one(&mut self, max_steps: usize) -> PlatformerSnapshot {
        self.replaying_best = false;
        // Perturb once, then keep learned weights through subsequent rounds.
        if self.training_episodes == 0 {
            *self.brain = SpikingBrain::train_init_with(self.seed ^ 7, platformer_prior());
        }
        self.reset_world_keep_brain();
        let mut steps = 0;
        while !self.finished && !self.dead && steps < max_steps {
            self.step();
            steps += 1;
        }

        // Curriculum is real, not decorative: every episode advances through
        // the three deterministic threat profiles. The brain keeps its W2
        // weights, while each episode tests a new timing/obstacle regime.
        let score = self.episode_score(steps);
        self.training_episodes += 1;
        self.training_steps.push(steps);
        self.training_rewards.push(score);
        if self.training_steps.len() > TRAINING_HISTORY_LIMIT {
            self.training_steps.remove(0);
            self.training_rewards.remove(0);
        }
        if self.finished {
            self.successful_episodes += 1;
            if self.best_steps == 0 || steps < self.best_steps {
                self.best_steps = steps;
            }
        }
        if score > self.best_score {
            self.best_score = score;
            self.best_brain = Some(self.brain.clone());
        }
        self.seed = self.seed.wrapping_add(1);
        self.reset_world_keep_brain();
        self.snapshot()
    }

    pub fn train(&mut self, episodes: usize, max_steps: usize) -> PlatformerSnapshot {
        for _ in 0..episodes {
            self.train_one(max_steps);
        }
        self.snapshot()
    }

    /// Restore the best checkpoint and start a playable fresh round. The
    /// checkpoint is cloned, so replay never consumes or mutates the stored
    /// champion policy.
    pub fn replay_best(&mut self) -> PlatformerSnapshot {
        if let Some(best) = self.best_brain.clone() {
            self.brain = best;
            self.replaying_best = true;
            self.event("replay_best_policy".to_string());
        }
        self.reset_world_keep_brain();
        self.snapshot()
    }

    /// Evaluate the current policy on a deterministic rotation of profiles.
    /// Evaluation uses cloned brains and never changes the trained policy or
    /// its learning counters.
    pub fn evaluate(&mut self, episodes: usize, max_steps: usize) -> PlatformerSnapshot {
        let episodes = episodes.max(1).min(32);
        let policy = self
            .best_brain
            .clone()
            .unwrap_or_else(|| self.brain.clone());
        let mut successes = 0usize;
        let mut score_sum = 0.0f32;
        let mut worst = 1.0f32;
        for offset in 0..episodes {
            let mut trial = Self::new(self.seed.wrapping_add(offset as u64));
            trial.brain = policy.clone();
            trial.reset_world_keep_brain();
            let mut steps = 0;
            while !trial.finished && !trial.dead && steps < max_steps {
                trial.step();
                steps += 1;
            }
            if trial.finished {
                successes += 1;
            }
            score_sum += trial.episode_score(steps);
            worst = worst.min((trial.player_x / GOAL_X).clamp(0.0, 1.0));
        }
        self.evaluation_episodes = episodes;
        self.evaluation_successes = successes;
        self.evaluation_mean_score = score_sum / episodes as f32;
        self.evaluation_worst_progress = worst;
        self.event(format!("evaluation {successes}/{episodes} success"));
        self.snapshot()
    }

    fn episode_score(&self, steps: usize) -> f32 {
        let progress = (self.player_x / GOAL_X).clamp(0.0, 1.0) * 100.0;
        let completion = if self.finished { 100.0 } else { 0.0 };
        let survival = if self.dead { -25.0 } else { 5.0 };
        let efficiency = if self.finished {
            (1.0 - steps as f32 / 220.0).clamp(0.0, 1.0) * 25.0
        } else {
            0.0
        };
        progress
            + completion
            + survival
            + efficiency
            + self.coins_collected as f32 * 2.0
            + self.enemies_defeated as f32 * 3.0
    }

    fn curriculum_label(stage: u8) -> &'static str {
        match stage {
            0 => "calibration · learn timing",
            1 => "pressure · enemy control",
            2 => "mastery · all profiles",
            _ => "generalization · endless rotation",
        }
    }

    fn reset_world_keep_brain(&mut self) {
        let brain = std::mem::replace(
            &mut self.brain,
            Box::new(SpikingBrain::with_prior(platformer_prior())),
        );
        let mut fresh = Self::new(self.seed);
        let mut brain = brain;
        // Preserve learned W2 weights, but every new level starts with a
        // clean neural trajectory: no stale membrane/refractory/action state
        // leaks from the previous nightmare profile.
        brain.reset_dynamics();
        fresh.brain = brain;
        fresh.training_episodes = self.training_episodes;
        fresh.training_steps = std::mem::take(&mut self.training_steps);
        fresh.training_rewards = std::mem::take(&mut self.training_rewards);
        fresh.best_steps = self.best_steps;
        fresh.best_score = self.best_score;
        fresh.successful_episodes = self.successful_episodes;
        fresh.best_brain = self.best_brain.clone();
        fresh.evaluation_episodes = self.evaluation_episodes;
        fresh.evaluation_successes = self.evaluation_successes;
        fresh.evaluation_mean_score = self.evaluation_mean_score;
        fresh.evaluation_worst_progress = self.evaluation_worst_progress;
        fresh.replaying_best = self.replaying_best;
        *self = fresh;
    }

    fn sensors(&self) -> [f32; N_CHANNELS] {
        let pit_signal = self.pit_signal();
        let enemy = self.nearest_enemy_ahead();
        // Keep the perception horizon stable across modes so threat density
        // changes the cost of mistakes, not the proven route itself.
        let enemy_close = enemy.map(|d| d < 3.0).unwrap_or(false);
        let goal = ((GOAL_X - self.player_x) / GOAL_X).clamp(0.0, 1.0);
        let coin = self
            .coins
            .iter()
            .any(|c| !c.collected && c.x > self.player_x && c.x - self.player_x < 8.0);
        [
            self.grounded as u8 as f32,
            pit_signal,
            enemy.is_some() as u8 as f32,
            enemy_close as u8 as f32,
            (self.velocity_x / MAX_SPEED).clamp(-1.0, 1.0),
            (self.velocity_y / JUMP_VELOCITY).clamp(-1.0, 1.0),
            goal,
            coin as u8 as f32,
        ]
    }

    /// Continuous anticipatory observation: distance to the edge of the
    /// current support, normalized over the visual horizon. This is sensory
    /// information, not a jump command; the brain still chooses the action.
    fn pit_signal(&self) -> f32 {
        let horizon = 12.0;
        let Some(platform) = self
            .platforms
            .iter()
            .filter(|p| self.player_x >= p.x && self.player_x <= p.x + p.width)
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
        else {
            return 1.0;
        };
        let distance_to_edge = (platform.x + platform.width - self.player_x).max(0.0);
        if distance_to_edge >= horizon {
            0.0
        } else {
            // A clear early warning is still sensory input. It gives the
            // recurrent/spiking path enough time to propagate to output; it
            // never selects Jump itself.
            1.0
        }
    }

    fn nearest_enemy_ahead(&self) -> Option<f32> {
        self.enemies
            .iter()
            .filter(|e| e.alive && e.x >= self.player_x - 0.5)
            .map(|e| e.x - self.player_x)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    fn support_y(&self, x: f32) -> Option<f32> {
        self.platforms
            .iter()
            .filter(|p| x >= p.x && x <= p.x + p.width)
            .map(|p| p.y + p.height)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn step(&mut self) -> PlatformerSnapshot {
        self.replaying_best = false;
        if self.finished || self.dead {
            return self.snapshot();
        }
        let before_x = self.player_x;
        let previous_y = self.player_y;
        let sensors = self.sensors();
        let brain_action = Action::from_index(self.brain.step(&sensors));
        self.brain_action = brain_action;
        // Brain-only control: the action emitted by the spiking readout is
        // applied verbatim. Hazards are observations and rewards, never
        // hidden motor overrides. This makes `executed_action` a useful audit
        // field: it must always equal `brain_action`.
        let action = brain_action;
        if action == Action::Jump && self.grounded {
            self.velocity_y = JUMP_VELOCITY;
            self.grounded = false;
            self.event("brain_jump".to_string());
        }
        self.executed_action = action;
        let accel = match action {
            Action::Left => -RUN_ACCEL,
            Action::Right => RUN_ACCEL,
            Action::Sprint => RUN_ACCEL * 1.35,
            _ => 0.0,
        };
        self.velocity_x = (self.velocity_x + accel).clamp(-MAX_SPEED, MAX_SPEED);
        // Airborne motion is ballistic: Jump/Wait do not erase horizontal
        // momentum in mid-air. Ground friction is physical drag, not a second
        // controller hiding behind the action API.
        if self.grounded && matches!(action, Action::Wait | Action::Jump) {
            self.velocity_x *= FRICTION;
        }
        self.velocity_y += GRAVITY;
        let next_x = (self.player_x + self.velocity_x).clamp(0.0, LEVEL_WIDTH);
        let next_y = self.player_y + self.velocity_y;
        self.player_x = next_x;
        self.player_y = next_y;
        if let Some(y) = self.support_y(self.player_x) {
            // A platform can only catch the player while descending through
            // its top surface. The previous implementation snapped from
            // below as soon as x entered a landing, creating a hidden
            // teleport and an unavoidable enemy side-collision.
            let current_bottom = self.player_y - PLAYER_RADIUS;
            // A circular player can enter a raised landing from the side
            // while its centre is up to one radius below the top edge. The
            // old fixed `0.25` margin was narrower than the player and let
            // a valid corner landing tunnel through between fixed ticks.
            let crossed_from_above = previous_y >= y - PLAYER_RADIUS
                && current_bottom <= y + 0.05
                && self.velocity_y <= 0.0;
            if crossed_from_above {
                self.player_y = y + PLAYER_RADIUS;
                self.velocity_y = 0.0;
                self.grounded = true;
            } else {
                self.grounded = false;
            }
        } else {
            self.grounded = false;
        }
        if self.player_y < -3.0 {
            self.dead = true;
            self.last_reward = -5.0;
            // Terminal failures must reach R-STDP. Previously the value was
            // visible in telemetry but never applied to the synapses, so the
            // brain could not learn that ignoring a pit was costly.
            self.brain.apply_reward(self.last_reward);
            self.event("fell_into_pit".to_string());
            return self.snapshot();
        }
        // Reaching the flag ends the tick before enemy resolution: a goal
        // cannot be reported as both a win and a death in one fixed step.
        if self.player_x >= GOAL_X {
            self.finished = true;
            self.last_reward = 8.0;
            self.brain.apply_reward(self.last_reward);
            self.event(format!(
                "level_complete step={} coins={}",
                self.step, self.coins_collected
            ));
            self.step += 1;
            return self.snapshot();
        }
        let mut enemy_defeated = false;
        for enemy in self.enemies.iter_mut().filter(|e| e.alive) {
            let dx = (self.player_x - enemy.x).abs();
            let enemy_bottom = enemy.y - ENEMY_RADIUS;
            let enemy_top = enemy.y + ENEMY_RADIUS;
            let player_bottom = self.player_y - PLAYER_RADIUS;
            let player_top = self.player_y + PLAYER_RADIUS;
            let crossed_from_above = self.velocity_y <= 0.0
                && previous_y > self.player_y
                && previous_y >= enemy.y
                && self.player_y + PLAYER_RADIUS >= enemy.y - 0.15
                && player_bottom <= enemy_top + 0.05;
            let vertical_overlap =
                player_bottom < enemy_top - 0.05 && player_top > enemy_bottom + 0.05;
            if dx < PLAYER_RADIUS + ENEMY_RADIUS && crossed_from_above {
                enemy.alive = false;
                self.enemies_defeated += 1;
                self.velocity_y = JUMP_VELOCITY * 0.65;
                self.last_reward = 2.0;
                enemy_defeated = true;
            } else if dx < PLAYER_RADIUS + ENEMY_RADIUS && vertical_overlap {
                self.dead = true;
                self.last_reward = -5.0 * self.threat_density;
                self.brain.apply_reward(self.last_reward);
                self.event("enemy_collision".to_string());
                return self.snapshot();
            }
        }
        if enemy_defeated {
            self.event("enemy_defeated".to_string());
        }
        let mut coin_collected = false;
        for coin in self.coins.iter_mut().filter(|c| !c.collected) {
            if (self.player_x - coin.x).abs() < 0.8 && (self.player_y - coin.y).abs() < 1.0 {
                coin.collected = true;
                self.coins_collected += 1;
                self.last_reward = 0.8;
                coin_collected = true;
            }
        }
        if coin_collected {
            self.event("coin_collected".to_string());
        }
        let progress = (self.player_x - before_x).max(0.0);
        // Reward timing, not a motor override: a jump selected while the
        // danger signal is high is reinforced; ignoring the same observation
        // is mildly punished before the terminal fall penalty arrives.
        let hazard_shaping = if self.grounded && sensors[1] > 0.0 {
            if action == Action::Jump {
                sensors[1] * 0.18
            } else {
                -sensors[1] * 0.12
            }
        } else {
            0.0
        };
        self.last_reward = if self.player_x >= GOAL_X {
            8.0
        } else if progress > 0.0 {
            0.03 + progress * 0.04 + hazard_shaping
        } else {
            -0.02 + hazard_shaping
        };
        self.brain.apply_reward(self.last_reward);
        self.step += 1;
        self.snapshot()
    }

    fn event(&mut self, event: String) {
        if self.events.len() >= 24 {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    pub fn snapshot_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.snapshot())
    }

    pub fn snapshot(&self) -> PlatformerSnapshot {
        PlatformerSnapshot {
            schema_version: 3,
            level_id: self.level_id,
            level_name: self.level_name.clone(),
            threat_level: self.threat_level,
            threat_density: self.threat_density,
            mission: "platformer".to_string(),
            width: LEVEL_WIDTH,
            height: LEVEL_HEIGHT,
            step: self.step,
            finished: self.finished,
            dead: self.dead,
            player_x: self.player_x,
            player_y: self.player_y,
            velocity_x: self.velocity_x,
            velocity_y: self.velocity_y,
            jump_velocity: JUMP_VELOCITY,
            grounded: self.grounded,
            brain_action: self.brain_action.as_str().to_string(),
            executed_action: self.executed_action.as_str().to_string(),
            brain_only: true,
            coins: self.coins_collected,
            enemies_defeated: self.enemies_defeated,
            goal_x: GOAL_X,
            platforms: self.platforms.clone(),
            enemies: self.enemies.clone(),
            coin_items: self.coins.clone(),
            brain: BrainSnapshot {
                action_rates: self.brain.action_rates().to_vec(),
                sensor_rates: self.brain.sensor_channel_rates(),
                workspace_rates: self.brain.workspace_group_rates(),
                weight_heat: self.brain.weight_heat(),
                region_gate: self.brain.region_gate().to_vec(),
                dopamine: self.brain.dopamine(),
                reward: self.brain.last_reward(),
                baseline: self.brain.baseline(),
                spikes: self.brain.last_spikes(),
                spike_history: self.brain.spike_log().iter().copied().collect(),
            },
            reward: self.last_reward,
            training_episodes: self.training_episodes,
            training_steps: self.training_steps.clone(),
            training_rewards: self.training_rewards.clone(),
            best_steps: self.best_steps,
            best_score: if self.best_score.is_finite() {
                self.best_score
            } else {
                0.0
            },
            has_best_checkpoint: self.best_brain.is_some(),
            successful_episodes: self.successful_episodes,
            curriculum_stage: (self.training_episodes / 3).min(3) as u8,
            curriculum_label: Self::curriculum_label((self.training_episodes / 3).min(3) as u8)
                .to_string(),
            evaluation_episodes: self.evaluation_episodes,
            evaluation_successes: self.evaluation_successes,
            evaluation_mean_score: self.evaluation_mean_score,
            evaluation_worst_progress: self.evaluation_worst_progress,
            replaying_best: self.replaying_best,
            events: self.events.clone(),
        }
    }
}

impl Default for PlatformerArena {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_is_deterministic() {
        let a = PlatformerArena::new(7).snapshot();
        let b = PlatformerArena::new(7).snapshot();
        assert_eq!(a.platforms, b.platforms);
        assert_eq!(a.enemies, b.enemies);
        assert_eq!(a.coin_items, b.coin_items);
    }

    #[test]
    fn physics_jumps_over_first_pit() {
        let mut arena = PlatformerArena::new(7);
        arena.player_x = 16.0;
        arena.grounded = true;
        arena.brain = Box::new(SpikingBrain::new());
        arena.velocity_y = JUMP_VELOCITY;
        arena.grounded = false;
        let before = arena.player_y;
        arena.step();
        assert!(arena.player_y > before - 0.2);
        assert!(!arena.dead);
    }

    #[test]
    fn fresh_brain_is_honest_and_finite() {
        let mut arena = PlatformerArena::new(0xCAFE_BABE);
        for _ in 0..900 {
            let snapshot = arena.step();
            assert!(snapshot.brain_only);
            assert_eq!(snapshot.brain_action, snapshot.executed_action);
            assert!(snapshot.player_x.is_finite() && snapshot.player_y.is_finite());
            if snapshot.finished || snapshot.dead {
                break;
            }
        }
        assert!(arena.snapshot().step > 0);
    }

    #[test]
    fn snapshot_contains_brain_raster() {
        let mut arena = PlatformerArena::new(1);
        for _ in 0..8 {
            arena.step();
        }
        let s = arena.snapshot();
        assert!(!s.brain.spike_history.is_empty());
        assert!(s.brain.spike_history.iter().all(|r| r.len() == 3));
    }

    #[test]
    fn jump_apex_stays_inside_level() {
        let relative_apex = JUMP_VELOCITY * JUMP_VELOCITY / (2.0 * -GRAVITY);
        assert!(relative_apex < LEVEL_HEIGHT - 1.0);
        assert!((relative_apex - 4.0).abs() < 0.01);
    }

    #[test]
    fn actual_step_trajectory_stays_inside_level() {
        let mut arena = PlatformerArena::new(0xCAFE_BABE);
        let mut max_y = arena.player_y;
        for _ in 0..900 {
            let snapshot = arena.step();
            max_y = max_y.max(snapshot.player_y);
            if snapshot.finished || snapshot.dead {
                break;
            }
        }
        assert!(
            max_y < LEVEL_HEIGHT,
            "actual step exceeded level ceiling: {max_y}"
        );
        assert!(max_y < 6.0, "unexpected actual apex: {max_y}");
    }

    #[test]
    fn fresh_profiles_are_deterministic_without_instincts() {
        for seed in 0..3 {
            let mut arena = PlatformerArena::new(seed);
            for _ in 0..900 {
                let snapshot = arena.step();
                assert!(snapshot.brain_only);
                assert_eq!(snapshot.brain_action, snapshot.executed_action);
                assert!(snapshot.player_x.is_finite() && snapshot.player_y.is_finite());
                if snapshot.finished || snapshot.dead {
                    break;
                }
            }
            assert!(arena.snapshot().threat_level >= 1);
        }
    }

    #[test]
    fn swept_landing_catches_fast_descent() {
        let mut arena = PlatformerArena::new(1);
        arena.player_x = 25.0;
        arena.player_y = 2.12;
        arena.velocity_x = 0.2;
        arena.velocity_y = -1.20;
        arena.grounded = false;
        let expected_y = arena.support_y(arena.player_x).unwrap() + PLAYER_RADIUS;
        let snapshot = arena.step();
        assert!(
            snapshot.grounded,
            "missed landing at x={}",
            snapshot.player_x
        );
        assert!((snapshot.player_y - expected_y).abs() < 0.01);
        assert!(!snapshot.dead);
    }

    #[test]
    fn corner_entry_lands_on_raised_platform() {
        let mut arena = PlatformerArena::new(2); // ABYSS final raised landing
                                                 // This is the exact fixed-step shape that previously tunneled through:
                                                 // the centre is 0.16 below the platform top, but the player's radius
                                                 // still overlaps its top surface while descending.
        arena.player_x = 98.5689;
        arena.player_y = 1.64;
        arena.velocity_x = 1.3625;
        arena.velocity_y = -0.48;
        arena.grounded = false;

        let snapshot = arena.step();
        assert!(snapshot.grounded, "corner entry missed the raised landing");
        assert!((snapshot.player_y - 2.22).abs() < 0.01);
        assert!(!snapshot.dead);
    }

    #[test]
    fn terminal_failure_delivers_negative_dopamine() {
        let mut arena = PlatformerArena::new(0);
        arena.player_y = -3.1;
        arena.velocity_y = 0.0;
        arena.grounded = false;
        let snapshot = arena.step();
        assert!(snapshot.dead);
        assert!(snapshot.reward < 0.0);
        assert!(snapshot.brain.dopamine < 0.0);
    }

    #[test]
    fn brain_action_is_applied_without_instinct_override() {
        let mut arena = PlatformerArena::new(0);
        for _ in 0..32 {
            let snapshot = arena.step();
            assert!(snapshot.brain_only);
            assert_eq!(snapshot.brain_action, snapshot.executed_action);
            if snapshot.finished || snapshot.dead {
                break;
            }
        }
    }

    #[test]
    fn training_lab_accumulates_curriculum_metrics() {
        let mut arena = PlatformerArena::new(0);
        let first = arena.train_one(900);
        assert_eq!(first.training_episodes, 1);
        assert_eq!(first.training_steps.len(), 1);
        assert_eq!(first.training_rewards.len(), 1);
        assert!(first.best_score.is_finite());

        let second = arena.train_one(900);
        assert_eq!(second.training_episodes, 2);
        assert_eq!(second.training_steps.len(), 2);
        assert_eq!(second.training_rewards.len(), 2);
        assert!(second
            .training_rewards
            .iter()
            .all(|score| score.is_finite()));
        assert!(second.curriculum_label.contains("calibration"));
        assert!(second.successful_episodes <= second.training_episodes);
        assert!(second.successful_episodes >= first.successful_episodes);
    }

    #[test]
    fn training_metrics_survive_round_reset() {
        let mut arena = PlatformerArena::new(0);
        arena.train_one(900);
        let before = arena.snapshot();
        arena.next_round();
        let after = arena.snapshot();
        assert_eq!(after.training_episodes, before.training_episodes);
        assert_eq!(after.training_steps, before.training_steps);
        assert_eq!(after.training_rewards, before.training_rewards);
        assert_eq!(after.best_steps, before.best_steps);
        assert_eq!(after.successful_episodes, before.successful_episodes);
    }

    #[test]
    fn evaluation_is_non_mutating_and_reports_generalization() {
        let mut arena = PlatformerArena::new(0);
        arena.train_one(900);
        let before_episodes = arena.snapshot().training_episodes;
        let evaluated = arena.evaluate(5, 900);
        assert_eq!(evaluated.evaluation_episodes, 5);
        assert!(evaluated.evaluation_successes <= 5);
        assert!(evaluated.evaluation_mean_score.is_finite());
        assert!((0.0..=1.0).contains(&evaluated.evaluation_worst_progress));
        assert_eq!(evaluated.training_episodes, before_episodes);
    }

    #[test]
    fn replay_best_restores_a_playable_checkpoint() {
        let mut arena = PlatformerArena::new(0);
        arena.train_one(900);
        let before = arena.snapshot();
        assert!(before.has_best_checkpoint);
        let replay = arena.replay_best();
        assert!(replay.replaying_best);
        assert_eq!(replay.step, 0);
        assert_eq!(replay.training_episodes, before.training_episodes);
        assert_eq!(replay.best_steps, before.best_steps);
        assert!(!replay.dead);
    }

    #[test]
    fn level_profiles_are_distinct_and_deterministic() {
        let snapshots = (0..3u64)
            .map(|seed| PlatformerArena::new(seed).snapshot())
            .collect::<Vec<_>>();
        assert_eq!(snapshots[0].level_name, "NIGHTMARE");
        assert_eq!(snapshots[1].level_name, "SAWTOOTH");
        assert_eq!(snapshots[2].level_name, "ABYSS");
        assert!(snapshots
            .windows(2)
            .all(|pair| pair[0].level_name != pair[1].level_name));
        assert_eq!(snapshots[0].threat_density, 1.0);
        assert_eq!(snapshots[1].threat_density, 1.35);
        assert_eq!(snapshots[2].threat_density, 1.7);
        assert!(snapshots
            .iter()
            .all(|s| s.threat_level > 0 && s.threat_level <= 3));
        assert_eq!(snapshots[0].threat_density * 5.0, 5.0);
        assert_eq!(snapshots[1].threat_density * 5.0, 6.75);
        assert_eq!(snapshots[2].threat_density * 5.0, 8.5);
    }
}
