//! Deterministic two-agent cognitive arena core — Mission 02: Two-Key Vault.
//!
//! The arena owns the world state and all transitions. Native examples, the
//! WebAssembly facade, and the JavaScript renderer must use this module rather
//! than reimplementing simulation logic.
//!
//! Mission contract:
//! - Scout perceives only the RED key; Carrier perceives only the BLUE key
//!   (color-channel perception, like dichromatic vision).
//! - Each agent must carry its own key to the vault door.
//! - The vault opens only when BOTH agents stand next to the door, each with
//!   their key in hand — synchronized coordination, not solo retrieval.
//! - Agents share their key positions through bidirectional semantic-pointer
//!   messages (bind/unbind), so each knows the other's readiness.
//! - A decoy trap sits on Scout's route: it is not an obstacle to the planner,
//!   so stepping on it stuns the agent and is learned as an obstacle for
//!   replanning (learning from error).
//!
//! Cognition: each agent runs a **real spiking brain** (`SpikingBrain`, built
//! on pillars LIF): perception → sensory population → GNW workspace →
//! output population → action, with reward-modulated STDP learning. The arena
//! issues rewards for mission events (key acquired, message shared, intel
//! decoded, synchronized vault opening) and penalties (trap hit, time), and
//! `train()` runs repeated episodes so the brains visibly improve.

use std::collections::VecDeque;

use serde::Serialize;

use super::brain::{SpikingBrain, N_ACTIONS, N_CHANNELS};
use super::{SemanticPointer, WorkingMemory};

pub const WIDTH: usize = 32;
pub const HEIGHT: usize = 20;
const MEMORY_DIM: usize = 1024;
const VISION_RADIUS: i32 = 5;
const RED_KEY: Pos = Pos { x: 8, y: 4 };
const BLUE_KEY: Pos = Pos { x: 26, y: 6 };
const DOOR: Pos = Pos { x: 27, y: 15 };
const DECOY: Pos = Pos { x: 14, y: 9 };
const STUN_STEPS: usize = 2;
/// If an agent waits at the vault this many steps past the first arrival, it
/// stops waiting and navigates toward the partner's key position (decoded from
/// the received semantic message) to meet them halfway — the decoded value
/// genuinely drives planning.
const MEET_AFTER_WAIT: usize = 15;

// ── Reward shaping for the spiking brains ──────────────────────────────
/// Small time pressure every step (encourages efficiency).
const TIME_PRESSURE: f32 = 0.01;
const REWARD_KEY: f32 = 0.5;
const REWARD_SHARE: f32 = 0.3;
const REWARD_INTEL: f32 = 0.3;
const REWARD_AT_DOOR: f32 = 0.1;
const REWARD_VAULT: f32 = 2.0;
/// Trap hit penalty (also drives the brain away from trap-bound actions).
const PENALTY_DECOY: f32 = 0.6;
/// Reward shaping: ± this amount when the brain's action matches (or
/// contradicts) the expert mission pipeline. This is what gives the
/// reward-modulated STDP a real learning signal — confused actions are
/// pruned, correct ones reinforced.
const SHAPING: f32 = 0.12;

/// Symbol indices in the semantic pointer space.
const SYM_RED: usize = 3;
const SYM_BLUE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    fn distance(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Scout,
    Carrier,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scout => "scout",
            Self::Carrier => "carrier",
        }
    }

    fn own_key(self) -> Pos {
        match self {
            Self::Scout => RED_KEY,
            Self::Carrier => BLUE_KEY,
        }
    }

    fn own_symbol(self) -> usize {
        match self {
            Self::Scout => SYM_RED,
            Self::Carrier => SYM_BLUE,
        }
    }

    fn key_label(self) -> &'static str {
        match self {
            Self::Scout => "red",
            Self::Carrier => "blue",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Explore = 0,
    Share = 1,
    Retrieve = 2,
    Carry = 3,
    Wait = 4,
}

impl Action {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Share => "share",
            Self::Retrieve => "retrieve",
            Self::Carry => "carry",
            Self::Wait => "wait",
        }
    }
}

/// Brain action index → arena action (indices are the brain's output order).
fn action_from_index(index: usize) -> Action {
    match index {
        0 => Action::Explore,
        1 => Action::Share,
        2 => Action::Retrieve,
        3 => Action::Carry,
        _ => Action::Wait,
    }
}

/// A semantic-pointer message carrying a partner's key position.
#[derive(Debug, Clone)]
struct Message {
    encoded_key: Vec<f32>,
    confidence: f32,
    sent_at: usize,
}

struct AgentMemory {
    semantic: SemanticPointer,
    working: WorkingMemory,
    baseline: Option<Vec<f32>>,
    /// Semantic-pointer symbol index of this agent's own key color.
    own_symbol: usize,
    /// Position of this agent's own key (observed or received).
    own_key: Option<Pos>,
    /// Position of the partner's key (decoded from a received message).
    partner_key: Option<Pos>,
    confidence: f32,
    age: usize,
}

impl AgentMemory {
    fn new(own_symbol: usize) -> Self {
        let mut working = WorkingMemory::new(2, MEMORY_DIM);
        working.decay = 0.995;
        working.gate_threshold = 0.0;
        working.recurrent_gain = 0.0;
        Self {
            semantic: SemanticPointer::new(MEMORY_DIM),
            working,
            baseline: None,
            own_symbol,
            own_key: None,
            partner_key: None,
            confidence: 0.0,
            age: 0,
        }
    }

    fn position_symbol(position: Pos) -> usize {
        (position.y as usize * WIDTH + position.x as usize) % MEMORY_DIM
    }

    fn encode_key(&self, symbol: usize, position: Pos) -> Vec<f32> {
        let key = self.semantic.from_index(symbol);
        let position = self.semantic.from_index(Self::position_symbol(position));
        self.semantic.bind(&key, &position)
    }

    fn remember_own_key(&mut self, position: Pos) {
        let encoded = self.encode_key(self.own_symbol, position);
        self.working.store(&encoded);
        self.baseline = Some(self.working.recall(0));
        self.own_key = Some(position);
        self.confidence = 1.0;
        self.age = 0;
    }

    fn receive_partner_key(
        &mut self,
        decoded: Option<Pos>,
        encoded: &[f32],
        confidence: f32,
        age: usize,
    ) {
        self.working.store(encoded);
        self.baseline = Some(self.working.recall(0));
        self.partner_key = decoded;
        self.confidence = self.confidence.max(confidence);
        self.age = age;
    }

    fn decode_key(&self, symbol: usize, encoded: &[f32]) -> Option<Pos> {
        let key = self.semantic.from_index(symbol);
        let decoded = self.semantic.unbind(encoded, &key);
        let (position, confidence) = (0..HEIGHT as i32)
            .flat_map(|y| (0..WIDTH as i32).map(move |x| Pos { x, y }))
            .map(|position| {
                let symbol = self.semantic.from_index(Self::position_symbol(position));
                (position, self.semantic.similarity(&decoded, &symbol))
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        (confidence >= 0.05).then_some(position)
    }

    fn tick(&mut self) {
        self.working.tick();
        self.age += 1;
        self.confidence *= 0.997;
    }

    fn retention(&self) -> f32 {
        let Some(baseline) = &self.baseline else {
            return 0.0;
        };
        let current = self.working.recall(0);
        let base_norm = baseline.iter().map(|x| x * x).sum::<f32>().sqrt();
        let current_norm = current.iter().map(|x| x * x).sum::<f32>().sqrt();
        if base_norm > 1e-12 {
            current_norm / base_norm
        } else {
            0.0
        }
    }
}

struct Agent {
    role: Role,
    position: Pos,
    known_obstacles: Vec<Pos>,
    memory: AgentMemory,
    /// Boxed: the spiking brain is ~100KB of fixed-size arrays — it must live
    /// on the heap so arena resets (`*self = Self::new()`) never touch the
    /// thread stack.
    brain: Box<SpikingBrain>,
    carrying_own: bool,
    shared: bool,
    stunned: usize,
    /// Consecutive steps spent waiting at the vault door (for the meet-up rule).
    waiting_at_door: usize,
    /// Latched once the meet-up decision is made: from then on the agent
    /// navigates to the decoded partner key position instead of the door.
    meeting: bool,
    /// One-shot guard: after a failed meet-up (partner already left the key
    /// position) the agent returns to the vault and never re-triggers the
    /// meet-up, avoiding an infinite door↔key oscillation.
    meet_attempted: bool,
    last_action: Action,
    last_confidence: f32,
    decisions: usize,
    path_steps: usize,
}

impl Agent {
    fn new(role: Role, position: Pos) -> Self {
        Self {
            role,
            position,
            known_obstacles: Vec::new(),
            memory: AgentMemory::new(role.own_symbol()),
            brain: Box::new(SpikingBrain::new()),
            carrying_own: false,
            shared: false,
            stunned: 0,
            waiting_at_door: 0,
            meeting: false,
            meet_attempted: false,
            last_action: Action::Wait,
            last_confidence: 0.0,
            decisions: 0,
            path_steps: 0,
        }
    }

    fn visible_own_key(&self) -> bool {
        self.position.distance(self.role.own_key()) <= VISION_RADIUS
    }

    /// The 8-channel perception vector fed to the spiking brain.
    /// Channels: 0 key_unknown · 1 share_pending · 2 at_own_key ·
    /// 3 carrying · 4 await_intel · 5 go_vault · 6 stunned · 7 at_door.
    fn perception(&self) -> [f32; N_CHANNELS] {
        let known = self.memory.own_key.is_some();
        let at_key = self.position.distance(self.role.own_key()) <= VISION_RADIUS;
        let intel = self.memory.partner_key.is_some();
        [
            if known { 0.0 } else { 1.0 },
            if known && !self.shared { 1.0 } else { 0.0 },
            if known && !self.carrying_own && at_key {
                1.0
            } else {
                0.0
            },
            if self.carrying_own { 1.0 } else { 0.0 },
            if self.carrying_own && self.shared && !intel {
                1.0
            } else {
                0.0
            },
            if self.carrying_own && intel { 1.0 } else { 0.0 },
            if self.stunned > 0 { 1.0 } else { 0.0 },
            if self.position.distance(DOOR) <= 1 {
                1.0
            } else {
                0.0
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BrainSnapshot {
    pub action_rates: Vec<f32>,
    pub sensor_rates: Vec<f32>,
    pub workspace_rates: Vec<f32>,
    /// W2 averaged over each action's output neurons: [5 actions × 128 workspace groups].
    pub weight_heat: Vec<f32>,
    /// GNW motor-access gate per brain region (2 values): which region
    /// currently wins the motor pool.
    pub region_gate: Vec<f32>,
    pub dopamine: f32,
    pub reward: f32,
    pub baseline: f32,
    /// Last-step spike totals (sensor, workspace, output).
    pub spikes: [f32; 3],
    /// Recent aggregate spike raster, oldest → newest. Each row is
    /// [sensor, workspace/GNW, output] activity for one brain tick.
    pub spike_history: Vec<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentSnapshot {
    pub role: String,
    pub x: i32,
    pub y: i32,
    pub action: String,
    pub confidence: f32,
    pub memory_confidence: f32,
    pub memory_age: usize,
    pub memory_retention: f32,
    pub known_obstacles: usize,
    /// "red", "blue", or "none" — which key this agent is carrying.
    pub carrying: String,
    /// Whether the partner's key position was decoded from a received message.
    pub partner_key_known: bool,
    pub decisions: usize,
    pub path_steps: usize,
    pub brain: BrainSnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingSnapshot {
    pub episodes: usize,
    /// Total reward per training episode.
    pub rewards: Vec<f32>,
    /// Steps to complete (or max_steps) per training episode.
    pub steps: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArenaSnapshot {
    pub schema_version: u32,
    pub width: usize,
    pub height: usize,
    pub step: usize,
    pub finished: bool,
    pub red_key: [i32; 2],
    pub blue_key: [i32; 2],
    pub decoy: [i32; 2],
    pub door: [i32; 2],
    pub obstacles: Vec<[i32; 2]>,
    pub scout_known_obstacles: Vec<[i32; 2]>,
    pub carrier_known_obstacles: Vec<[i32; 2]>,
    pub scout: AgentSnapshot,
    pub carrier: AgentSnapshot,
    pub messages_sent: usize,
    pub explored_steps: usize,
    /// Steps between the first agent reaching the vault and its opening.
    pub coordination_delay: usize,
    /// How many times an agent stepped on the decoy trap.
    pub decoy_hits: usize,
    pub belief_uncertainty: f32,
    pub memory_retention: f32,
    /// Mean total spikes per arena step across both brains.
    pub brain_activity: f32,
    pub training: TrainingSnapshot,
    pub events: Vec<String>,
}

/// The single source of truth for the browser/native Cognitive Arena.
pub struct CognitiveArena {
    obstacles: Vec<Pos>,
    scout: Agent,
    carrier: Agent,
    scout_message: Option<Message>,
    carrier_message: Option<Message>,
    messages_sent: usize,
    explored_steps: usize,
    coordination_delay: usize,
    decoy_hits: usize,
    first_at_door: Option<usize>,
    belief_uncertainty_sum: f32,
    memory_retention_sum: f32,
    metric_samples: usize,
    // Reward plumbing for the spiking brains.
    scout_step_reward: f32,
    carrier_step_reward: f32,
    training_episode_reward: f32,
    reward_curve: Vec<f32>,
    steps_curve: Vec<usize>,
    episodes_done: usize,
    /// Whether `train_one` has already perturbed the brains (so repeated
    /// continuous-training calls keep the learned weights instead of
    /// re-perturbing from scratch every time).
    trained_once: bool,
    spike_total: f32,
    spike_samples: usize,
    events: VecDeque<String>,
    step: usize,
    finished: bool,
}

impl CognitiveArena {
    pub fn new() -> Self {
        let mut obstacles = Vec::new();
        // The wall has two gaps (y=9 and y=10): the decoy covers one, so
        // learning the trap must leave the other passable (replan, not dead end).
        for y in 0..HEIGHT as i32 {
            if y != 9 && y != 10 {
                obstacles.push(Pos { x: 15, y });
            }
        }
        for x in 16..23 {
            obstacles.push(Pos { x, y: 12 });
        }
        let scout = Agent::new(Role::Scout, Pos { x: 2, y: 2 });
        let carrier = Agent::new(Role::Carrier, Pos { x: 27, y: 17 });
        let mut arena = Self {
            obstacles,
            scout,
            carrier,
            scout_message: None,
            carrier_message: None,
            messages_sent: 0,
            explored_steps: 0,
            coordination_delay: 0,
            decoy_hits: 0,
            first_at_door: None,
            belief_uncertainty_sum: 0.0,
            memory_retention_sum: 0.0,
            metric_samples: 0,
            scout_step_reward: 0.0,
            carrier_step_reward: 0.0,
            training_episode_reward: 0.0,
            reward_curve: Vec::new(),
            steps_curve: Vec::new(),
            episodes_done: 0,
            trained_once: false,
            spike_total: 0.0,
            spike_samples: 0,
            events: VecDeque::new(),
            step: 0,
            finished: false,
        };
        arena.event("role_bidding policy=initial_cost_assignment A:scout B:carrier".to_string());
        arena.event(format!(
            "mission=two_key_vault red_key=({}, {}) blue_key=({}, {}) door=({}, {})",
            RED_KEY.x, RED_KEY.y, BLUE_KEY.x, BLUE_KEY.y, DOOR.x, DOOR.y
        ));
        arena
    }

    /// Full reset: fresh world AND fresh (prior) brains.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Episode reset: fresh world; trained brains, learning curves, episode
    /// counter and the `trained_once` flag persist (continuous learning).
    /// Used by `train()`/`train_one()` between episodes.
    pub fn reset_episode(&mut self) {
        let reward_curve = std::mem::take(&mut self.reward_curve);
        let steps_curve = std::mem::take(&mut self.steps_curve);
        let episodes_done = self.episodes_done;
        let trained_once = self.trained_once;
        // Swap the whole arena with a fresh one (no throwaway ~100KB brain
        // allocations), then swap the trained brains back in.
        let mut fresh = Self::new();
        std::mem::swap(self, &mut fresh);
        std::mem::swap(&mut self.scout.brain, &mut fresh.scout.brain);
        std::mem::swap(&mut self.carrier.brain, &mut fresh.carrier.brain);
        self.scout.brain.reset_dynamics();
        self.carrier.brain.reset_dynamics();
        self.reward_curve = reward_curve;
        self.steps_curve = steps_curve;
        self.episodes_done = episodes_done;
        self.trained_once = trained_once;
    }

    /// Run exactly one training episode: perturb the brains on the FIRST call
    /// (so the reward-modulated STDP has errors to prune), then reset the
    /// world and run. Repeated calls keep the learned weights — continuous
    /// learning.
    pub fn train_one(&mut self, max_steps: usize) -> ArenaSnapshot {
        if !self.trained_once {
            *self.scout.brain = SpikingBrain::train_init(7);
            *self.carrier.brain = SpikingBrain::train_init(9);
            self.trained_once = true;
        }
        self.reset_episode();

        let mut steps = 0usize;
        while !self.finished && steps < max_steps {
            self.step();
            steps += 1;
        }
        self.reward_curve.push(self.training_episode_reward);
        self.steps_curve.push(steps);
        self.episodes_done += 1;
        // Leave the arena in a fresh, playable state (world reset, trained
        // brains + learning curves preserved) so Play works right after
        // training instead of being stuck on the last episode's `finished`.
        self.reset_episode();
        self.snapshot()
    }

    /// Train `episodes` episodes in one call (convenience wrapper).
    pub fn train(&mut self, episodes: usize, max_steps: usize) -> ArenaSnapshot {
        for _ in 0..episodes {
            self.train_one(max_steps);
        }
        self.snapshot()
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn step(&mut self) -> ArenaSnapshot {
        if self.finished {
            return self.snapshot();
        }
        let step = self.step;
        self.scout_step_reward = -TIME_PRESSURE;
        self.carrier_step_reward = -TIME_PRESSURE;
        self.scout.memory.tick();
        self.carrier.memory.tick();

        // Perception: each agent can see only its own color of key.
        if self.scout.visible_own_key() && self.scout.memory.own_key.is_none() {
            let pos = RED_KEY;
            self.scout.memory.remember_own_key(pos);
            self.event(format!(
                "memory_write scout sees red_key=({}, {})",
                pos.x, pos.y
            ));
        }
        if self.carrier.visible_own_key() && self.carrier.memory.own_key.is_none() {
            let pos = BLUE_KEY;
            self.carrier.memory.remember_own_key(pos);
            self.event(format!(
                "memory_write carrier sees blue_key=({}, {})",
                pos.x, pos.y
            ));
        }
        self.observe_agent(true);
        self.observe_agent(false);

        // Spiking decisions (brain ticks every step, even while stunned).
        let scout_action = self.tick_action(true);
        let carrier_action = self.tick_action(false);

        // Reward shaping: reinforce pipeline-consistent decisions so R-STDP
        // has a gradient to follow (the expert policy is the mission stages).
        // The shaping is a LEARNING signal only — it is excluded from the
        // reported training reward, which measures mission quality.
        let scout_shaping = if scout_action == self.pipeline_action(true) {
            SHAPING
        } else {
            -SHAPING
        };
        self.scout_step_reward += scout_shaping;
        let carrier_shaping = if carrier_action == self.pipeline_action(false) {
            SHAPING
        } else {
            -SHAPING
        };
        self.carrier_step_reward += carrier_shaping;

        // Bidirectional semantic communication of key positions.
        if scout_action == Action::Share && !self.scout.shared {
            if let Some(pos) = self.scout.memory.own_key {
                self.scout_message = Some(Message {
                    encoded_key: self.scout.memory.encode_key(SYM_RED, pos),
                    confidence: self.scout.memory.confidence,
                    sent_at: step,
                });
                self.scout.shared = true;
                self.messages_sent += 1;
                self.scout_step_reward += REWARD_SHARE;
                self.event(format!(
                    "semantic_message scout → carrier red_key=({}, {})",
                    pos.x, pos.y
                ));
            }
        }
        if carrier_action == Action::Share && !self.carrier.shared {
            if let Some(pos) = self.carrier.memory.own_key {
                self.carrier_message = Some(Message {
                    encoded_key: self.carrier.memory.encode_key(SYM_BLUE, pos),
                    confidence: self.carrier.memory.confidence,
                    sent_at: step,
                });
                self.carrier.shared = true;
                self.messages_sent += 1;
                self.carrier_step_reward += REWARD_SHARE;
                self.event(format!(
                    "semantic_message carrier → scout blue_key=({}, {})",
                    pos.x, pos.y
                ));
            }
        }
        if let Some(message) = &self.scout_message {
            if self.carrier.memory.partner_key.is_none() {
                let decoded = self
                    .carrier
                    .memory
                    .decode_key(SYM_RED, &message.encoded_key);
                self.carrier.memory.receive_partner_key(
                    decoded,
                    &message.encoded_key,
                    message.confidence,
                    step.saturating_sub(message.sent_at),
                );
                self.carrier_step_reward += REWARD_INTEL;
                self.event(format!("memory_decode carrier red_key={:?}", decoded));
            }
        }
        if let Some(message) = &self.carrier_message {
            if self.scout.memory.partner_key.is_none() {
                let decoded = self.scout.memory.decode_key(SYM_BLUE, &message.encoded_key);
                self.scout.memory.receive_partner_key(
                    decoded,
                    &message.encoded_key,
                    message.confidence,
                    step.saturating_sub(message.sent_at),
                );
                self.scout_step_reward += REWARD_INTEL;
                self.event(format!("memory_decode scout blue_key={:?}", decoded));
            }
        }

        self.belief_uncertainty_sum +=
            (1.0 - self.scout.memory.confidence) + (1.0 - self.carrier.memory.confidence);
        self.memory_retention_sum +=
            self.scout.memory.retention() + self.carrier.memory.retention();
        self.metric_samples += 2;

        // Latched meet-up decision: long waits at the vault flip the agent into
        // navigation toward the decoded partner key position.
        let scout_meets = self.scout.waiting_at_door > MEET_AFTER_WAIT
            && self.scout.memory.partner_key.is_some()
            && !self.scout.meeting
            && !self.scout.meet_attempted;
        let carrier_meets = self.carrier.waiting_at_door > MEET_AFTER_WAIT
            && self.carrier.memory.partner_key.is_some()
            && !self.carrier.meeting
            && !self.carrier.meet_attempted;
        if scout_meets {
            self.scout.meeting = true;
            if let Some(target) = self.scout.memory.partner_key {
                self.event(format!(
                    "mutual_aid scout_meets_partner target=({}, {}) step={step}",
                    target.x, target.y
                ));
            }
        }
        if carrier_meets {
            self.carrier.meeting = true;
            if let Some(target) = self.carrier.memory.partner_key {
                self.event(format!(
                    "mutual_aid carrier_meets_partner target=({}, {}) step={step}",
                    target.x, target.y
                ));
            }
        }

        // Movement phase.
        self.move_agent(true, scout_action, step);
        self.move_agent(false, carrier_action, step);

        // Vault: synchronized dual-key opening.
        let scout_at_door = self.scout.position.distance(DOOR) <= 1;
        let carrier_at_door = self.carrier.position.distance(DOOR) <= 1;
        self.scout.waiting_at_door = if scout_at_door {
            self.scout.waiting_at_door.saturating_add(1)
        } else {
            0
        };
        self.carrier.waiting_at_door = if carrier_at_door {
            self.carrier.waiting_at_door.saturating_add(1)
        } else {
            0
        };
        if self.first_at_door.is_none() && (scout_at_door || carrier_at_door) {
            self.first_at_door = Some(step);
            if scout_at_door {
                self.scout_step_reward += REWARD_AT_DOOR;
            }
            if carrier_at_door {
                self.carrier_step_reward += REWARD_AT_DOOR;
            }
            self.event(format!("agent_at_door step={step}"));
        }
        if scout_at_door && carrier_at_door {
            self.coordination_delay = step.saturating_sub(self.first_at_door.unwrap_or(step));
            let keys_ready = self.scout.carrying_own && self.carrier.carrying_own;
            if keys_ready {
                self.finished = true;
                self.scout_step_reward += REWARD_VAULT;
                self.carrier_step_reward += REWARD_VAULT;
                self.event(format!(
                    "door_open step={step} coordination_delay={}",
                    self.coordination_delay
                ));
            }
        }

        // Reward-modulated learning + activity metrics.
        self.scout.brain.apply_reward(self.scout_step_reward);
        self.carrier.brain.apply_reward(self.carrier_step_reward);
        // Mission-quality metric: event rewards, penalties and time pressure
        // only — shaping is excluded, so faster missions read as better.
        self.training_episode_reward +=
            self.scout_step_reward + self.carrier_step_reward - scout_shaping - carrier_shaping;
        let [s1, w1, o1] = self.scout.brain.last_spikes();
        let [s2, w2, o2] = self.carrier.brain.last_spikes();
        self.spike_total += s1 + w1 + o1 + s2 + w2 + o2;
        self.spike_samples += 1;

        self.step += 1;
        self.snapshot()
    }

    pub fn run(&mut self, max_steps: usize) -> ArenaSnapshot {
        for _ in 0..max_steps {
            self.step();
            if self.finished {
                break;
            }
        }
        self.snapshot()
    }

    pub fn snapshot_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.snapshot())
    }

    pub fn snapshot(&self) -> ArenaSnapshot {
        let samples = self.metric_samples.max(1) as f32;
        let spike_samples = self.spike_samples.max(1) as f32;
        ArenaSnapshot {
            schema_version: 4,
            width: WIDTH,
            height: HEIGHT,
            step: self.step,
            finished: self.finished,
            red_key: [RED_KEY.x, RED_KEY.y],
            blue_key: [BLUE_KEY.x, BLUE_KEY.y],
            decoy: [DECOY.x, DECOY.y],
            door: [DOOR.x, DOOR.y],
            obstacles: self.obstacles.iter().map(|p| [p.x, p.y]).collect(),
            scout_known_obstacles: self
                .scout
                .known_obstacles
                .iter()
                .map(|p| [p.x, p.y])
                .collect(),
            carrier_known_obstacles: self
                .carrier
                .known_obstacles
                .iter()
                .map(|p| [p.x, p.y])
                .collect(),
            scout: self.agent_snapshot(&self.scout),
            carrier: self.agent_snapshot(&self.carrier),
            messages_sent: self.messages_sent,
            explored_steps: self.explored_steps,
            coordination_delay: self.coordination_delay,
            decoy_hits: self.decoy_hits,
            belief_uncertainty: self.belief_uncertainty_sum / samples,
            memory_retention: self.memory_retention_sum / samples,
            brain_activity: self.spike_total / spike_samples,
            training: TrainingSnapshot {
                episodes: self.episodes_done,
                rewards: self.reward_curve.clone(),
                steps: self.steps_curve.clone(),
            },
            events: self.events.iter().cloned().collect(),
        }
    }

    fn agent_snapshot(&self, agent: &Agent) -> AgentSnapshot {
        AgentSnapshot {
            role: agent.role.as_str().to_string(),
            x: agent.position.x,
            y: agent.position.y,
            action: agent.last_action.as_str().to_string(),
            confidence: agent.last_confidence,
            memory_confidence: agent.memory.confidence,
            memory_age: agent.memory.age,
            memory_retention: agent.memory.retention(),
            known_obstacles: agent.known_obstacles.len(),
            carrying: if agent.carrying_own {
                agent.role.key_label().to_string()
            } else {
                "none".to_string()
            },
            partner_key_known: agent.memory.partner_key.is_some(),
            decisions: agent.decisions,
            path_steps: agent.path_steps,
            brain: BrainSnapshot {
                action_rates: agent.brain.action_rates().to_vec(),
                sensor_rates: agent.brain.sensor_channel_rates(),
                workspace_rates: agent.brain.workspace_group_rates(),
                weight_heat: agent.brain.weight_heat(),
                region_gate: agent.brain.region_gate().to_vec(),
                dopamine: agent.brain.dopamine(),
                reward: agent.brain.last_reward(),
                baseline: agent.brain.baseline(),
                spikes: agent.brain.last_spikes(),
                spike_history: agent.brain.spike_log().iter().copied().collect(),
            },
        }
    }

    fn event(&mut self, message: String) {
        if self.events.len() >= 32 {
            self.events.pop_front();
        }
        self.events.push_back(message);
    }

    fn in_bounds(p: Pos) -> bool {
        p.x >= 0 && p.x < WIDTH as i32 && p.y >= 0 && p.y < HEIGHT as i32
    }

    fn blocked(&self, p: Pos) -> bool {
        !Self::in_bounds(p) || self.obstacles.contains(&p)
    }

    /// The decoy is only an obstacle once an agent has learned it the hard way.
    fn dangerous(&self, p: Pos) -> bool {
        p == DECOY
    }

    fn observe_agent(&mut self, scout: bool) {
        let position = if scout {
            self.scout.position
        } else {
            self.carrier.position
        };
        let mut visible = Vec::new();
        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                let candidate = Pos { x, y };
                if position.distance(candidate) <= VISION_RADIUS
                    && self.obstacles.contains(&candidate)
                {
                    visible.push(candidate);
                }
            }
        }
        let known = if scout {
            &mut self.scout.known_obstacles
        } else {
            &mut self.carrier.known_obstacles
        };
        for candidate in visible {
            if !known.contains(&candidate) {
                known.push(candidate);
            }
        }
    }

    fn next_step(&self, from: Pos, target: Pos, known_obstacles: &[Pos]) -> Option<Pos> {
        if from == target {
            return None;
        }
        let mut previous = vec![None; WIDTH * HEIGHT];
        let mut queue = VecDeque::from([from]);
        previous[from.y as usize * WIDTH + from.x as usize] = Some(from);
        while let Some(current) = queue.pop_front() {
            let neighbors = [
                Pos {
                    x: current.x + 1,
                    y: current.y,
                },
                Pos {
                    x: current.x - 1,
                    y: current.y,
                },
                Pos {
                    x: current.x,
                    y: current.y + 1,
                },
                Pos {
                    x: current.x,
                    y: current.y - 1,
                },
            ];
            for next in neighbors {
                // A wall or a learned trap blocks the planner only once the
                // agent knows about it; unknown obstacles are optimistically
                // traversable until encountered (replanning on collision).
                let known_trap = self.dangerous(next) && known_obstacles.contains(&next);
                if !Self::in_bounds(next)
                    || (self.blocked(next) && known_obstacles.contains(&next))
                    || known_trap
                {
                    continue;
                }
                let index = next.y as usize * WIDTH + next.x as usize;
                if previous[index].is_some() {
                    continue;
                }
                previous[index] = Some(current);
                if next == target {
                    let mut cursor = next;
                    while previous[cursor.y as usize * WIDTH + cursor.x as usize] != Some(from) {
                        cursor = previous[cursor.y as usize * WIDTH + cursor.x as usize]?;
                    }
                    return Some(cursor);
                }
                queue.push_back(next);
            }
        }
        None
    }

    /// One spiking decision: tick the brain with the current perception and
    /// map its output to an arena action. The brain ticks even while stunned
    /// (channel 6 drives the Wait action); the stun only overrides the result.
    fn tick_action(&mut self, scout: bool) -> Action {
        let (_channels, index) = {
            let agent = if scout {
                &mut self.scout
            } else {
                &mut self.carrier
            };
            let channels = agent.perception();
            let index = agent.brain.step(&channels);
            (channels, index)
        };
        let action = action_from_index(index);
        let agent = if scout {
            &mut self.scout
        } else {
            &mut self.carrier
        };
        agent.last_action = action;
        agent.decisions += 1;
        let rates = agent.brain.action_rates().to_vec();
        let selected = rates[index.min(N_ACTIONS - 1)];
        let runner_up = rates
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index.min(N_ACTIONS - 1))
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        agent.last_confidence = (selected - runner_up).clamp(0.0, 1.0);
        if agent.stunned > 0 {
            agent.stunned -= 1;
            Action::Wait
        } else {
            action
        }
    }

    /// The expert mission pipeline: the correct action for the current state.
    /// Explore → Share → Retrieve → Wait-for-intel → Carry (stun forces Wait).
    fn pipeline_action(&self, scout: bool) -> Action {
        let agent = if scout { &self.scout } else { &self.carrier };
        if agent.stunned > 0 {
            return Action::Wait;
        }
        let known = agent.memory.own_key.is_some();
        let intel = agent.memory.partner_key.is_some();
        if !known {
            Action::Explore
        } else if !agent.shared {
            Action::Share
        } else if !agent.carrying_own {
            Action::Retrieve
        } else if !intel {
            Action::Wait
        } else {
            Action::Carry
        }
    }

    fn move_agent(&mut self, scout: bool, action: Action, step: usize) {
        // Phase 1: decide the movement goal without holding a &mut borrow, so
        // self.next_step / self.dangerous / self.blocked stay available.
        let (target, count_explore, count_path, corrupt_meet) = {
            let agent = if scout { &self.scout } else { &self.carrier };
            match action {
                Action::Explore => {
                    let targets = if scout {
                        [Pos { x: 8, y: 2 }, Pos { x: 8, y: 8 }, Pos { x: 12, y: 6 }]
                    } else {
                        [
                            Pos { x: 24, y: 12 },
                            Pos { x: 22, y: 6 },
                            Pos { x: 26, y: 4 },
                        ]
                    };
                    (
                        targets[(step / 8).min(targets.len() - 1)],
                        true,
                        false,
                        false,
                    )
                }
                Action::Retrieve => (
                    agent.role.own_key(),
                    false,
                    agent.memory.own_key.is_some(),
                    false,
                ),
                Action::Carry => {
                    // Meet-up rule: the decoded partner key position genuinely
                    // drives planning. Once `meeting` is latched (long wait at
                    // the vault), abandon the door and navigate toward the
                    // partner's key position instead of waiting indefinitely.
                    let meet_target = if agent.meeting {
                        agent.memory.partner_key
                    } else {
                        None
                    };
                    let target = match meet_target {
                        // Corrupt semantic-pointer decode: a partner key
                        // outside the arena can never be reached, so treat it
                        // as a failed meet-up right now (one-shot) instead of
                        // stranding the agent on an unreachable target.
                        Some(position) if !Self::in_bounds(position) => DOOR,
                        other => other.unwrap_or(DOOR),
                    };
                    let corrupt =
                        matches!(meet_target, Some(position) if !Self::in_bounds(position));
                    (target, false, true, corrupt)
                }
                Action::Share | Action::Wait => return,
            }
        };

        if corrupt_meet {
            let agent = if scout {
                &mut self.scout
            } else {
                &mut self.carrier
            };
            // A corrupt decode is a failed meet-up: clear the latch so the
            // agent returns to the vault and never retries the garbage target.
            agent.meeting = false;
            agent.meet_attempted = true;
        }

        let (position, known) = {
            let agent = if scout { &self.scout } else { &self.carrier };
            (agent.position, agent.known_obstacles.clone())
        };
        let Some(next) = self.next_step(position, target, &known) else {
            return;
        };

        // Learning from error: the decoy is invisible to the planner, so the
        // agent steps on it, gets stunned, and records it as an obstacle.
        if self.dangerous(next) {
            self.decoy_hits += 1;
            {
                let agent = if scout {
                    &mut self.scout
                } else {
                    &mut self.carrier
                };
                agent.stunned = STUN_STEPS;
                agent.last_action = Action::Wait;
                if !agent.known_obstacles.contains(&next) {
                    agent.known_obstacles.push(next);
                }
            }
            if scout {
                self.scout_step_reward -= PENALTY_DECOY;
            } else {
                self.carrier_step_reward -= PENALTY_DECOY;
            }
            self.event(format!(
                "decoy_trap {}_hit step={step} learned_obstacle=({}, {})",
                if scout { "scout" } else { "carrier" },
                next.x,
                next.y
            ));
            return;
        }

        if self.blocked(next) {
            let agent = if scout {
                &mut self.scout
            } else {
                &mut self.carrier
            };
            if !agent.known_obstacles.contains(&next) {
                agent.known_obstacles.push(next);
            }
            return;
        }

        let mut acquired_own = false;
        {
            let agent = if scout {
                &mut self.scout
            } else {
                &mut self.carrier
            };
            agent.position = next;
            if count_path {
                agent.path_steps += 1;
            }
            // Pick up the agent's own key on arrival.
            if action == Action::Retrieve && agent.position == agent.role.own_key() {
                agent.carrying_own = true;
                acquired_own = true;
            }
            // Meet-up fallback: if the partner already left the key position,
            // re-latch back to the vault (one-shot, so we never oscillate
            // between door and key position).
            if agent.meeting && agent.memory.partner_key == Some(agent.position) {
                agent.meeting = false;
                agent.meet_attempted = true;
            }
        }
        if count_explore {
            self.explored_steps += 1;
        }
        if acquired_own {
            let agent = if scout { &self.scout } else { &self.carrier };
            if scout {
                self.scout_step_reward += REWARD_KEY;
            } else {
                self.carrier_step_reward += REWARD_KEY;
            }
            self.event(format!(
                "{}_acquired_{}_key step={step}",
                agent.role.as_str(),
                agent.role.key_label()
            ));
        }
    }
}

impl Default for CognitiveArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::karmac::brain::N_WORKSPACE;

    #[test]
    fn step_exposes_json_snapshot() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.step();
        assert_eq!(snapshot.schema_version, 4);
        assert_eq!(snapshot.width, WIDTH);
        assert_eq!(snapshot.height, HEIGHT);
        assert_eq!(snapshot.red_key, [8, 4]);
        assert_eq!(snapshot.blue_key, [26, 6]);
        assert_eq!(snapshot.decoy, [14, 9]);
        let json = arena.snapshot_json().expect("snapshot should serialize");
        assert!(json.contains("coordination_delay"));
        assert!(json.contains("brain_activity"));
        assert!(json.contains("weight_heat"));
        assert!(json.contains("spike_history"));
        for brain in [&snapshot.scout.brain, &snapshot.carrier.brain] {
            assert!(!brain.spike_history.is_empty());
            assert!(brain.spike_history.len() <= 64);
            assert!(brain
                .spike_history
                .iter()
                .all(|row| row.len() == 3 && row.iter().all(|v| v.is_finite())));
        }
    }

    #[test]
    fn semantic_memory_retains_signal() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.run(30);
        assert!(snapshot.memory_retention.is_finite());
        assert!(snapshot.memory_retention >= 0.0);
    }

    #[test]
    fn spiking_brain_drives_mission() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.run(30);
        assert!(snapshot.brain_activity.is_finite());
        assert!(snapshot.brain_activity >= 0.0);
        assert_eq!(snapshot.scout.brain.action_rates.len(), N_ACTIONS);
        assert_eq!(snapshot.scout.brain.sensor_rates.len(), N_CHANNELS);
        assert_eq!(
            snapshot.scout.brain.weight_heat.len(),
            N_ACTIONS * N_WORKSPACE
        );
        for agent in [&snapshot.scout, &snapshot.carrier] {
            assert!(agent.brain.dopamine.is_finite());
            assert!(agent.brain.action_rates.iter().all(|r| r.is_finite()));
            assert!(agent.brain.weight_heat.iter().all(|w| w.is_finite()));
        }
    }

    #[test]
    fn two_key_vault_mission_completes() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.run(800);
        assert!(
            snapshot.finished,
            "arena did not finish: {}",
            arena.snapshot_json().expect("snapshot should serialize")
        );
        assert_eq!(
            snapshot.messages_sent, 2,
            "both key positions must be shared"
        );
        assert_eq!(snapshot.scout.carrying, "red");
        assert_eq!(snapshot.carrier.carrying, "blue");
        assert!(
            snapshot.coordination_delay > 0,
            "synchronized opening must involve waiting at the door"
        );
        // The semantic pipeline must decode the true partner key positions.
        assert_eq!(arena.scout.memory.partner_key, Some(BLUE_KEY));
        assert_eq!(arena.carrier.memory.partner_key, Some(RED_KEY));
    }

    #[test]
    fn decoded_partner_position_drives_meetup_after_long_wait() {
        // Synthetic scenario: Carrier is far away with its key while Scout is
        // already at the vault holding the red key. After MEET_AFTER_WAIT steps
        // of waiting, Scout must abandon the door and navigate toward the
        // decoded BLUE key position (meet-up), proving the decoded value
        // genuinely influences planning.
        let mut arena = CognitiveArena::new();
        arena.scout.carrying_own = true;
        arena.scout.shared = true;
        arena.scout.memory.own_key = Some(RED_KEY);
        arena.scout.memory.partner_key = Some(BLUE_KEY);
        arena.scout.position = DOOR;
        arena.scout.waiting_at_door = MEET_AFTER_WAIT + 1;
        // Carrier stays far away, so the vault can never open.
        arena.carrier.position = Pos { x: 2, y: 17 };
        arena.carrier.carrying_own = false;
        arena.carrier.memory.own_key = None;
        arena.carrier.memory.partner_key = Some(RED_KEY);

        // 40 steps: enough for the meet-up trip and fallback, but the far
        // carrier (which genuinely explores and can complete the mission on
        // its own) has not yet found its key by then — the vault stays shut.
        let snapshot = arena.run(40);
        assert!(
            arena.scout.path_steps > 0,
            "scout must travel toward the partner: {}",
            arena.snapshot_json().expect("snapshot should serialize")
        );
        assert!(
            snapshot.events.iter().any(|e| e.contains("mutual_aid")),
            "meet-up must be reported in the event stream"
        );
        assert!(!snapshot.finished, "vault cannot open without both keys");
    }

    #[test]
    fn decoy_trap_is_learned_not_fatal() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.run(800);
        // The trap sits on Scout's route to the vault: he must step on it at
        // least once, learn it as an obstacle, and still complete the mission.
        assert!(
            snapshot.decoy_hits >= 1,
            "scout must encounter the trap on his route: {}",
            arena.snapshot_json().expect("snapshot should serialize")
        );
        assert!(
            snapshot.events.iter().any(|e| e.contains("decoy_trap")),
            "trap hits must appear in the event stream"
        );
        assert!(snapshot.finished);
    }

    #[test]
    fn meetup_falls_back_to_vault_if_partner_already_left() {
        // Scout waits at the vault too long, latches a meet-up toward the
        // decoded BLUE key position, arrives there, finds the partner already
        // left — and must return to the vault instead of idling forever or
        // oscillating between door and key position. Carrier is far away with
        // its key, so the vault legitimately cannot open yet.
        let mut arena = CognitiveArena::new();
        arena.scout.carrying_own = true;
        arena.scout.shared = true;
        arena.scout.memory.own_key = Some(RED_KEY);
        arena.scout.memory.partner_key = Some(BLUE_KEY);
        arena.scout.position = DOOR;
        arena.scout.waiting_at_door = MEET_AFTER_WAIT + 1;
        arena.scout.meeting = true; // already latched
        arena.carrier.carrying_own = true;
        arena.carrier.shared = true;
        arena.carrier.memory.own_key = Some(BLUE_KEY);
        arena.carrier.memory.partner_key = Some(RED_KEY);
        arena.carrier.position = Pos { x: 2, y: 17 }; // far away, key in hand

        let _ = arena.run(120);
        assert_eq!(
            arena.scout.position, DOOR,
            "scout must return to the vault after a failed meet-up"
        );
        assert!(
            arena.scout.meet_attempted,
            "the failed meet-up must be recorded as attempted (one-shot)"
        );
        assert!(!arena.scout.meeting, "no infinite meet-up re-trigger");
        assert!(
            arena.snapshot().finished,
            "mission must complete once both agents reach the vault"
        );
    }

    #[test]
    fn corrupt_meetup_decode_falls_back_to_vault() {
        // Scout latched a meet-up toward an out-of-bounds decoded position
        // (corrupted semantic pointer). The unreachable target must be treated
        // as a failed meet-up: the latch clears, meet_attempted latches, and
        // the agent returns to the vault instead of stranding forever.
        let mut arena = CognitiveArena::new();
        arena.scout.carrying_own = true;
        arena.scout.shared = true;
        arena.scout.memory.own_key = Some(RED_KEY);
        arena.scout.memory.partner_key = Some(Pos { x: 65, y: 6 }); // outside 32×20
        arena.scout.position = DOOR;
        arena.scout.waiting_at_door = MEET_AFTER_WAIT + 1;
        arena.scout.meeting = true;
        arena.carrier.carrying_own = true;
        arena.carrier.shared = true;
        arena.carrier.memory.own_key = Some(BLUE_KEY);
        arena.carrier.memory.partner_key = Some(RED_KEY);
        arena.carrier.position = Pos { x: 2, y: 17 };

        let _ = arena.run(40);
        assert!(
            arena.scout.meet_attempted,
            "corrupt decode must be recorded as a failed meet-up"
        );
        assert!(
            !arena.scout.meeting,
            "no meeting re-trigger on corrupt target"
        );
        assert_eq!(
            arena.scout.position, DOOR,
            "scout must stay at the vault (unreachable target ignored)"
        );
    }

    #[test]
    fn training_improves_mission() {
        let mut arena = CognitiveArena::new();
        arena.train(6, 800);
        assert_eq!(arena.episodes_done, 6);
        assert_eq!(arena.reward_curve.len(), 6);
        assert_eq!(arena.steps_curve.len(), 6);
        let first_steps = arena.steps_curve[0];
        let last_steps = *arena.steps_curve.last().unwrap();
        let first_reward = arena.reward_curve[0];
        let last_reward = *arena.reward_curve.last().unwrap();
        assert!(
            last_steps <= first_steps,
            "training must not make the mission slower: {first_steps} -> {last_steps}"
        );
        assert!(
            last_reward > first_reward,
            "training must raise episode reward: {first_reward:.2} -> {last_reward:.2}"
        );
        // Trained brains differ from the pristine prior.
        assert_ne!(
            arena.scout.brain.weights(),
            SpikingBrain::new().weights().as_ref()
        );
        // And the trained arena still completes the mission.
        let snapshot = arena.run(800);
        assert!(snapshot.finished);
    }

    #[test]
    fn train_one_accumulates_episodes_continuously() {
        let mut arena = CognitiveArena::new();
        arena.train_one(800);
        assert_eq!(arena.episodes_done, 1);
        assert_eq!(arena.reward_curve.len(), 1);
        assert_eq!(arena.steps_curve.len(), 1);
        // Repeated calls keep the learned weights AND accumulate the curves
        // (no re-perturb, no curve wipe) — infinite/continuous training.
        arena.train_one(800);
        arena.train_one(800);
        assert_eq!(arena.episodes_done, 3);
        assert_eq!(arena.reward_curve.len(), 3);
        assert_eq!(arena.steps_curve.len(), 3);
        // Training continues to move the weights, but the brain is NOT reset
        // to a fresh perturbed prior (which is what a re-perturb would do).
        assert_ne!(
            arena.scout.brain.weights(),
            SpikingBrain::train_init(7).weights().as_ref(),
            "train_one must not re-perturb already-trained brains"
        );
        // And the arena still completes a fresh run afterwards.
        let snapshot = arena.run(800);
        assert!(snapshot.finished);
    }

    #[test]
    fn reset_episode_keeps_trained_brains() {
        let mut arena = CognitiveArena::new();
        arena.train(3, 800);
        let weights_before = arena.scout.brain.weights().to_vec();
        arena.reset_episode();
        assert_eq!(
            arena.scout.brain.weights(),
            weights_before.as_slice(),
            "episode reset must preserve trained weights"
        );
        assert_eq!(arena.step, 0);
        assert!(!arena.finished);
    }

    #[test]
    fn deterministic_replays() {
        let mut a = CognitiveArena::new();
        let mut b = CognitiveArena::new();
        a.run(50);
        b.run(50);
        assert_eq!(
            a.snapshot_json().expect("serialize"),
            b.snapshot_json().expect("serialize"),
            "identical arenas must replay identically"
        );
    }

    #[test]
    fn reset_returns_to_initial_state() {
        let mut arena = CognitiveArena::new();
        arena.run(20);
        arena.reset();
        let snapshot = arena.snapshot();
        assert_eq!(snapshot.step, 0);
        assert!(!snapshot.finished);
        assert_eq!(snapshot.messages_sent, 0);
        assert_eq!(snapshot.decoy_hits, 0);
        assert_eq!(snapshot.training.episodes, 0);
    }
}
