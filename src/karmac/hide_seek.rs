//! Mission 03: Hide & Seek — an OpenAI-style emergent game.
//!
//! Two **hiders** (blue) must survive a round against four **seekers**
//! (red). The map is a large procedurally generated complex (rooms, dead
//! ends, doorways) with movable **boxes**. Hiders can push boxes into
//! doorways to barricade them; seekers must find the hiders and can push
//! boxes back out of the way. All six agents are controlled by real
//! spiking brains (`SpikingBrain` on pillars LIF) with reward-modulated
//! STDP.
//!
//! Vision is honest **line-of-sight**: walls and boxes block sight, so a
//! barricaded doorway is real cover and a hider behind a wall is invisible.
//! Teams share **pack intel** — any seeker that spots a hider publishes a
//! fading trail the whole pack converges on; any hider that spots a seeker
//! publishes danger intel that scatters the team. No agent sees the map.
//!
//! Unlike Mission 02, there is **no expert-pipeline reward shaping**: the
//! reward signal is genuinely emergent —
//! * hiders: survival per tick, barricading a doorway that lengthens the
//!   seekers' paths, surviving the round, fleeing that increases the
//!   distance to the nearest threat;
//! * seekers: chasing that decreases the distance to the nearest hider,
//!   tagging a hider, unblocking a barricaded doorway.
//!
//! Learning is measurable: the training curve reports mean hider survival
//! (steps) and total tags per episode. A trained hider team should survive
//! longer; a trained seeker team should tag sooner.

use std::collections::{HashSet, VecDeque};

use serde::Serialize;

use super::arena::BrainSnapshot;
use super::brain::{BrainPrior, Rng, SpikingBrain, N_CHANNELS};

pub const WIDTH: usize = 48;
pub const HEIGHT: usize = 36;
const N_HIDERS: usize = 2;
/// Four seekers: the pack must combine line-of-sight hunting with shared
/// intel to pin down the hiders on the larger map.
const N_SEEKERS: usize = 4;
const N_BOXES: usize = 5;
const N_SHELTERS: usize = 8;
const SIGHT_HIDER: i32 = 7;
const SIGHT_SEEKER: i32 = 13;
const TAG_RADIUS: i32 = 1;
/// The arena cycles between dim and bright phases without RNG. Visibility
/// never reaches zero: agents can still make short-range decisions in dusk.
const VISIBILITY_CYCLE: usize = 96;
const MIN_VISIBILITY: f32 = 0.55;
/// Movement is audible through walls, but attenuates with distance. Noise is
/// deliberately weaker than a visual sighting and carries no SPOT reward.
const NOISE_RADIUS: i32 = 8;
/// Per-agent stamina. A full bar supports several pursuit bursts; waiting or
/// hiding restores it, so the optimal policy must budget movement.
const MAX_ENERGY: f32 = 1.0;
const ENERGY_EXPLORE: f32 = 0.045;
/// Flee is a burst: seekers may take a second step, so it costs more than
/// ordinary exploration even when the second step is blocked.
const ENERGY_FLEE: f32 = 0.18;
const ENERGY_PUSH: f32 = 0.16;
const ENERGY_HIDE_RECOVERY: f32 = 0.035;
const ENERGY_WAIT_RECOVERY: f32 = 0.075;
const ENERGY_MIN_ACTION: f32 = 0.01;
const ROUND_STEPS: usize = 300;
/// Team-intel freshness window (steps): a sighting published by any teammate
/// stays a fresh trail for this long, then fades to zero. Long enough for
/// the pack to converge across the map once a hider is spotted.
const INTEL_WINDOW: usize = 30;
/// Sound is shorter-lived and lower-confidence than visual team intel.
const SOUND_WINDOW: usize = 12;
const SOUND_CONFIDENCE: f32 = 0.45;

// ── Honest (emergent) rewards ─────────────────────────────────────────
const SURVIVE_TICK: f32 = 0.02;
const FLEE_GAIN: f32 = 0.25;
const CHASE_GAIN: f32 = 0.25;
/// Emergent reward for LAYING EYES on a hider (a fresh pack trail): the whole
/// live seeker team is reinforced for finding the hunt. Honest — it never
/// prescribes which action, only that spotting prey is good.
const SPOT: f32 = 0.4;
const TAG: f32 = 1.0;
const TAGGED: f32 = -1.0;
const BARRICADE: f32 = 0.6;
const UNBLOCK: f32 = 0.3;
const ROUND_SURVIVED: f32 = 0.5;
const TIME_PRESSURE: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Pos {
    fn manhattan(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Team {
    Hider,
    Seeker,
}

impl Team {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hider => "hider",
            Self::Seeker => "seeker",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HsAction {
    Explore = 0,
    Hide = 1,
    Push = 2,
    Flee = 3,
    Wait = 4,
}

impl HsAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Hide => "hide",
            Self::Push => "push",
            Self::Flee => "flee",
            Self::Wait => "wait",
        }
    }
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Explore,
            1 => Self::Hide,
            2 => Self::Push,
            3 => Self::Flee,
            _ => Self::Wait,
        }
    }
}

/// The generated world: walls, doorways (walkable gaps in walls), shelter
/// cells and box start positions.
pub struct Map {
    walls: Vec<bool>, // [y * WIDTH + x]
    doorways: Vec<Pos>,
    shelters: Vec<Pos>,
    box_starts: Vec<Pos>,
}

impl Map {
    fn wall(&self, p: Pos) -> bool {
        !Self::in_bounds(p) || self.walls[p.y as usize * WIDTH + p.x as usize]
    }

    fn in_bounds(p: Pos) -> bool {
        p.x >= 0 && p.x < WIDTH as i32 && p.y >= 0 && p.y < HEIGHT as i32
    }
}

/// Recursive subdivision: split a region with a wall that has a doorway gap,
/// then recurse. Produces a large, connected, roomy complex.
fn generate_map(seed: u64) -> Map {
    let mut rng = Rng::new(seed ^ 0x51ED_270B_51ED_270B);
    let mut walls = vec![false; WIDTH * HEIGHT];
    // Border walls.
    for x in 0..WIDTH {
        walls[x] = true;
        walls[(HEIGHT - 1) * WIDTH + x] = true;
    }
    for y in 0..HEIGHT {
        walls[y * WIDTH] = true;
        walls[y * WIDTH + WIDTH - 1] = true;
    }
    let mut doorways = Vec::new();

    fn split(
        rng: &mut Rng,
        walls: &mut [bool],
        doorways: &mut Vec<Pos>,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        depth: usize,
    ) {
        let w = x1 - x0;
        let h = y1 - y0;
        if depth >= 6 || w < 10 || h < 8 {
            return;
        }
        let vertical = w >= h;
        if vertical {
            let sx = x0 + 4 + rng.below((w - 8).max(1) as u64);
            let gy = y0 + 1 + rng.below((h - 2).max(1) as u64);
            for y in y0..y1 {
                let p = Pos {
                    x: sx as i32,
                    y: y as i32,
                };
                if y == gy {
                    doorways.push(p);
                } else {
                    walls[p.y as usize * WIDTH + p.x as usize] = true;
                }
            }
            split(rng, walls, doorways, x0, y0, sx, y1, depth + 1);
            split(rng, walls, doorways, sx + 1, y0, x1, y1, depth + 1);
        } else {
            let sy = y0 + 3 + rng.below((h - 6).max(1) as u64);
            let gx = x0 + 1 + rng.below((w - 2).max(1) as u64);
            for x in x0..x1 {
                let p = Pos {
                    x: x as i32,
                    y: sy as i32,
                };
                if x == gx {
                    doorways.push(p);
                } else {
                    walls[p.y as usize * WIDTH + p.x as usize] = true;
                }
            }
            split(rng, walls, doorways, x0, y0, x1, sy, depth + 1);
            split(rng, walls, doorways, x0, sy + 1, x1, y1, depth + 1);
        }
    }
    split(
        &mut rng,
        &mut walls,
        &mut doorways,
        1,
        1,
        WIDTH - 1,
        HEIGHT - 1,
        0,
    );

    // Shelters: walkable cells far from every doorway (dead-end hiding spots).
    let mut candidates: Vec<(i32, Pos)> = Vec::new();
    for y in 1..HEIGHT - 1 {
        for x in 1..WIDTH - 1 {
            let p = Pos {
                x: x as i32,
                y: y as i32,
            };
            if walls[y * WIDTH + x] {
                continue;
            }
            let min_door_dist = doorways
                .iter()
                .map(|d| p.manhattan(*d))
                .min()
                .unwrap_or(i32::MAX);
            if min_door_dist >= 8 {
                candidates.push((min_door_dist, p));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.x.cmp(&b.1.x))
            .then(a.1.y.cmp(&b.1.y))
    });
    let shelters: Vec<Pos> = candidates
        .into_iter()
        .take(N_SHELTERS)
        .map(|(_, p)| p)
        .collect();

    // Boxes: floor cells adjacent to a doorway (easy to push into the gap).
    let mut box_starts = Vec::new();
    let mut seen = HashSet::new();
    let mut door_shuffled: Vec<Pos> = doorways.clone();
    for i in (1..door_shuffled.len()).rev() {
        let j = rng.below((i + 1) as u64);
        door_shuffled.swap(i, j);
    }
    for door in door_shuffled {
        if box_starts.len() >= N_BOXES {
            break;
        }
        for p in [
            Pos {
                x: door.x + 1,
                y: door.y,
            },
            Pos {
                x: door.x - 1,
                y: door.y,
            },
            Pos {
                x: door.x,
                y: door.y + 1,
            },
            Pos {
                x: door.x,
                y: door.y - 1,
            },
        ] {
            if box_starts.len() >= N_BOXES {
                break;
            }
            if Map::in_bounds(p)
                && !walls[p.y as usize * WIDTH + p.x as usize]
                && !seen.contains(&(p.x, p.y))
                && !doorways.contains(&p)
            {
                seen.insert((p.x, p.y));
                box_starts.push(p);
            }
        }
    }

    Map {
        walls,
        doorways,
        shelters,
        box_starts,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HsAgentSnapshot {
    pub id: usize,
    pub team: String,
    pub x: i32,
    pub y: i32,
    pub action: String,
    pub alive: bool,
    /// Stamina in [0, 1]. Exposed so the UI can explain why an agent waits.
    pub energy: f32,
    /// Recent movement noise in [0, 1], decaying after each tick.
    pub noise: f32,
    pub brain: BrainSnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct HsTrainingSnapshot {
    pub episodes: usize,
    /// Mean hider survival in steps per episode.
    pub survival: Vec<f32>,
    /// Total tags per episode.
    pub tags: Vec<usize>,
    /// Mean total reward per episode (hider team minus seeker team).
    pub rewards: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HideSeekSnapshot {
    /// Snapshot wire contract. Version 6 adds deterministic `visibility` and
    /// per-agent `energy`/`noise`; consumers must reject older Hide & Seek
    /// snapshots rather than silently assuming the old fields are present.
    pub schema_version: u32,
    pub mission: String,
    pub width: usize,
    pub height: usize,
    /// Current deterministic light level, 0..1, used to scale sight ranges.
    pub visibility: f32,
    pub step: usize,
    pub finished: bool,
    /// Mean hider survival in steps.
    pub survival: f32,
    pub tags: usize,
    pub walls: Vec<[i32; 2]>,
    pub doorways: Vec<[i32; 2]>,
    pub boxes: Vec<[i32; 2]>,
    pub shelters: Vec<[i32; 2]>,
    pub hiders: Vec<HsAgentSnapshot>,
    pub seekers: Vec<HsAgentSnapshot>,
    /// Mean total spikes per arena step across all five brains.
    pub brain_activity: f32,
    pub training: HsTrainingSnapshot,
    pub events: Vec<String>,
}

#[derive(Clone)]
struct HsAgent {
    id: usize,
    position: Pos,
    /// Boxed: the spiking brain is ~100KB of fixed-size arrays — it must live
    /// on the heap so arena resets never touch the thread stack.
    brain: Box<SpikingBrain>,
    alive: bool,
    last_action: HsAction,
    /// BFS distance to the nearest opposing agent, computed every step.
    threat_dist: i32,
    /// Previous-step threat distance (for flee/chase potential rewards).
    prev_threat_dist: i32,
    /// For seekers: BFS distance to the pack's CURRENT chase objective — the
    /// fresh trail, or the nearest hider when the trail is cold. Measured
    /// separately from `threat_dist` so the chase reward teaches the pack to
    /// converge on the trail (moving toward the last sighting is rewarded)
    /// instead of punishing it for not closing on a hider that escaped.
    chase_dist: i32,
    /// Previous-step chase-objective distance.
    prev_chase_dist: i32,
    /// The chase objective used for the LAST reward computation. If the
    /// objective switches (a tag cleared the trail, or a sighting teleported
    /// it), the distance delta carries no signal — rewarding or punishing it
    /// would be noise, so the reward goes neutral for that step.
    last_chase_target: Option<Pos>,
    /// Stamina budget for movement and box pushing.
    energy: f32,
    /// Acoustic signature emitted by the last action; decays every tick.
    noise: f32,
}

impl HsAgent {
    fn new(id: usize, position: Pos, brain: Box<SpikingBrain>) -> Self {
        Self {
            id,
            position,
            brain,
            alive: true,
            last_action: HsAction::Wait,
            threat_dist: i32::MAX,
            prev_threat_dist: i32::MAX,
            chase_dist: i32::MAX,
            prev_chase_dist: i32::MAX,
            last_chase_target: None,
            energy: MAX_ENERGY,
            noise: 0.0,
        }
    }
}

/// The Hide & Seek arena. State is Rust-owned and fully deterministic.
pub struct HideSeekArena {
    map: Map,
    boxes: Vec<Option<Pos>>,
    hiders: Vec<HsAgent>,
    seekers: Vec<HsAgent>,
    step: usize,
    finished: bool,
    tags: usize,
    survival_sum: usize,
    survival_samples: usize,
    episode_reward: f32,
    training: HsTrainingSnapshot,
    spike_total: f32,
    spike_samples: usize,
    events: VecDeque<String>,
    rng: Rng,
    /// Round counter for auto-repeat: every `next_round` advances it, so
    /// repeated playthroughs get fresh spawn layouts instead of replaying the
    /// byte-identical round (the map itself stays fixed per seed).
    round: u64,
    /// Team intel — the last hider position any seeker laid eyes on (the
    /// pack's fading trail). Published by the whole seeker team.
    hider_last_seen: Option<(Pos, usize)>,
    /// Team intel — the last seeker position any hider spotted (danger
    /// intel). Makes the whole hider team scatter, not just the one seen.
    seeker_last_seen: Option<(Pos, usize)>,
    /// Arena step of the last pack-trail SPOT reward / intel event.
    /// Rate-limits the hunt-reinforcement to genuinely NEW finds (the trail
    /// went cold first) instead of firing on every re-sighting during an
    /// active chase.
    last_spot_step: usize,
    /// Arena step of the last danger-intel event (hiders spotting a seeker).
    /// Rate-limited independently so a pack find doesn't silence danger intel.
    last_danger_step: usize,
    /// Visual evidence seen during the current tick. Acoustic traces may
    /// refresh a cold trail, but never overwrite same-tick visual evidence.
    visual_hider_step: Option<usize>,
    visual_seeker_step: Option<usize>,
    /// Coarse, short-lived acoustic trails. They never overwrite visual GPS.
    hider_sound_last: Option<(Pos, usize)>,
    seeker_sound_last: Option<(Pos, usize)>,
}

impl HideSeekArena {
    pub fn new(seed: u64) -> Self {
        let map = generate_map(seed);
        let mut arena = Self {
            map,
            boxes: Vec::new(),
            hiders: Vec::new(),
            seekers: Vec::new(),
            step: 0,
            finished: false,
            tags: 0,
            survival_sum: 0,
            survival_samples: 0,
            episode_reward: 0.0,
            training: HsTrainingSnapshot {
                episodes: 0,
                survival: Vec::new(),
                tags: Vec::new(),
                rewards: Vec::new(),
            },
            spike_total: 0.0,
            spike_samples: 0,
            events: VecDeque::new(),
            rng: Rng::new(seed ^ 0xD1B5_4A32_D192_ED03),
            round: 0,
            hider_last_seen: None,
            seeker_last_seen: None,
            last_spot_step: 0,
            last_danger_step: 0,
            visual_hider_step: None,
            visual_seeker_step: None,
            hider_sound_last: None,
            seeker_sound_last: None,
        };
        arena.reset_episode();
        arena
    }

    pub fn reset(&mut self) {
        let seed = self.rng.next();
        *self = Self::new(seed ^ 0x5555_5555_5555_5555);
    }

    /// Start a fresh round with the trained brains preserved (auto-repeat).
    /// Advances the round counter so each repeat gets fresh spawn layouts.
    pub fn next_round(&mut self) -> HideSeekSnapshot {
        self.round = self.round.wrapping_add(1);
        self.reset_episode();
        self.snapshot()
    }

    /// Fresh world; trained brains persist (weights + learning state).
    pub fn reset_episode(&mut self) {
        let hider_brains: Vec<SpikingBrain> =
            self.hiders.iter().map(|a| (*a.brain).clone()).collect();
        let seeker_brains: Vec<SpikingBrain> =
            self.seekers.iter().map(|a| (*a.brain).clone()).collect();
        self.hiders.clear();
        self.seekers.clear();
        self.boxes.clear();
        self.step = 0;
        self.finished = false;
        self.tags = 0;
        self.survival_sum = 0;
        self.survival_samples = 0;
        self.episode_reward = 0.0;
        self.spike_total = 0.0;
        self.spike_samples = 0;
        self.events.clear();
        self.hider_last_seen = None;
        self.seeker_last_seen = None;
        self.last_spot_step = 0;
        self.last_danger_step = 0;
        self.visual_hider_step = None;
        self.visual_seeker_step = None;
        self.hider_sound_last = None;
        self.seeker_sound_last = None;

        self.boxes = self.map.box_starts.iter().map(|p| Some(*p)).collect();

        // Hiders start on open floor far from any doorway — the shelter is a
        // place they must *learn* to reach. Spawns are seeded from the episode
        // counter: the first episode (episodes == 0) shares the layout with
        // an untrained baseline run, and every later episode gets a fresh but
        // reproducible layout — so the curve mixes learning with healthy
        // layout variety without being pure spawn luck.
        let mut spawn_rng = Rng::new(
            (self.training.episodes as u64)
                .wrapping_add(self.round.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ 0xC0FF_EE00_C0FF_EE00,
        );
        let mut hider_spawns: Vec<Pos> = Vec::new();
        for y in 1..HEIGHT - 1 {
            for x in 1..WIDTH - 1 {
                let p = Pos {
                    x: x as i32,
                    y: y as i32,
                };
                if self.map.wall(p) {
                    continue;
                }
                let near_door = self.map.doorways.iter().any(|d| p.manhattan(*d) < 6);
                if !near_door && !self.boxes.contains(&Some(p)) && !hider_spawns.contains(&p) {
                    hider_spawns.push(p);
                }
            }
        }
        let mut used = HashSet::new();
        for i in 0..N_HIDERS {
            let pos = if !hider_spawns.is_empty() {
                // Unique spawns: never place two hiders on the same cell.
                let mut pick = hider_spawns[spawn_rng.below(hider_spawns.len() as u64)];
                for _ in 0..8 {
                    if used.insert(pick) {
                        break;
                    }
                    pick = hider_spawns[spawn_rng.below(hider_spawns.len() as u64)];
                }
                used.insert(pick);
                pick
            } else {
                Pos { x: 4, y: 4 }
            };
            let mut brain = hider_brains.get(i).cloned().unwrap_or_else(|| {
                SpikingBrain::train_init_with(0x1111 + i as u64, BrainPrior::hide_seek_hider())
            });
            brain.reset_dynamics();
            self.hiders.push(HsAgent::new(i, pos, Box::new(brain)));
        }
        // Seekers spawn near doorways at the map edges (same fixed seed).
        let mut spawns = Vec::new();
        for door in self.map.doorways.iter() {
            for delta in [
                Pos { x: 1, y: 0 },
                Pos { x: -1, y: 0 },
                Pos { x: 0, y: 1 },
                Pos { x: 0, y: -1 },
            ] {
                let p = Pos {
                    x: door.x + delta.x,
                    y: door.y + delta.y,
                };
                if Map::in_bounds(p)
                    && !self.map.wall(p)
                    && !self.boxes.contains(&Some(p))
                    && !spawns.contains(&p)
                {
                    spawns.push(p);
                }
            }
        }
        for i in 0..N_SEEKERS {
            let pos = if i < spawns.len() {
                spawns[spawn_rng.below(spawns.len() as u64)]
            } else {
                Pos {
                    x: 1 + (i as i32) * 3,
                    y: 1,
                }
            };
            let mut brain = seeker_brains.get(i).cloned().unwrap_or_else(|| {
                SpikingBrain::train_init_with(0x2222 + i as u64, BrainPrior::hide_seek_seeker())
            });
            brain.reset_dynamics();
            self.seekers.push(HsAgent::new(i, pos, Box::new(brain)));
        }
        self.event("mission=hide_seek round_start".to_string());
    }

    /// Run exactly one training episode (fresh world, trained brains persist)
    /// and return the snapshot. Call repeatedly for continuous learning.
    pub fn train_one(&mut self, max_steps: usize) -> HideSeekSnapshot {
        self.reset_episode();
        let mut steps = 0usize;
        while !self.finished && steps < max_steps {
            self.step();
            steps += 1;
        }
        let survivors = self.hiders.iter().filter(|h| h.alive).count();
        let survival = steps as f32 + survivors as f32 * 0.5;
        self.training.survival.push(survival);
        self.training.tags.push(self.tags);
        self.training.rewards.push(self.episode_reward);
        // Advance the episode counter BEFORE the next reset so spawns
        // rotate through layouts deterministically.
        self.training.episodes += 1;
        // Leave the arena in a fresh, playable state (trained brains persist)
        // so Play works right after training instead of being stuck on the
        // last episode's `finished` round.
        self.reset_episode();
        self.snapshot()
    }

    /// Train `episodes` episodes in one call (convenience wrapper).
    pub fn train(&mut self, episodes: usize, max_steps: usize) -> HideSeekSnapshot {
        for _ in 0..episodes {
            self.train_one(max_steps);
        }
        self.snapshot()
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Training curves accumulated across `train()` episodes.
    pub fn training(&self) -> &HsTrainingSnapshot {
        &self.training
    }

    /// Run up to `max_steps` ticks (or until the round ends) and return the
    /// final snapshot. Public convenience for examples/tests.
    pub fn run_for_demo(&mut self, max_steps: usize) -> HideSeekSnapshot {
        for _ in 0..max_steps {
            if self.finished {
                break;
            }
            self.step();
        }
        self.snapshot()
    }

    pub fn step(&mut self) -> HideSeekSnapshot {
        if self.finished {
            return self.snapshot();
        }
        let step = self.step;

        // ── 1. Distances: a BFS field per seeker, plus the chase objective. ─
        let fields = self.distance_fields();
        let chase_targets: Vec<Option<Pos>> = self
            .seekers
            .iter()
            .map(|s| self.chase_target(s.position))
            .collect();
        for h in self.hiders.iter_mut() {
            h.prev_threat_dist = h.threat_dist;
            let idx = h.position.y as usize * WIDTH + h.position.x as usize;
            // Only LIVE seekers threaten: a dead hunter's BFS field must not
            // grant a spurious (non-)threat to a hider.
            h.threat_dist = fields
                .iter()
                .enumerate()
                .filter(|(i, f)| self.seekers[*i].alive && f[idx] != i32::MAX)
                .map(|(_, f)| f[idx])
                .min()
                .unwrap_or(i32::MAX);
        }
        for (i, s) in self.seekers.iter_mut().enumerate() {
            s.prev_threat_dist = s.threat_dist;
            s.threat_dist = self
                .hiders
                .iter()
                .filter(|h| h.alive)
                .map(|h| fields[i][h.position.y as usize * WIDTH + h.position.x as usize])
                .min()
                .unwrap_or(i32::MAX);
            s.prev_chase_dist = s.chase_dist;
            s.chase_dist = match chase_targets[i] {
                Some(t) => fields[i][t.y as usize * WIDTH + t.x as usize],
                None => i32::MAX,
            };
        }

        // ── 2. Team intel: teammates publish sightings. ─────────────
        self.publish_sightings();

        // ── 3. Perceptions + brain ticks + rewards. ──────────────────
        let hider_channels: Vec<[f32; N_CHANNELS]> = (0..self.hiders.len())
            .map(|i| self.hider_perception(i))
            .collect();
        let mut hider_actions = Vec::with_capacity(self.hiders.len());
        for (i, channels) in hider_channels.into_iter().enumerate() {
            let index = self.hiders[i].brain.step(&channels);
            let requested = HsAction::from_index(index);
            let action = self.regulate_action(i, true, requested);
            let reward = if self.hiders[i].alive {
                SURVIVE_TICK
            } else {
                0.0
            } + self.flee_potential(i);
            self.hiders[i].brain.apply_reward(reward);
            hider_actions.push(action);
        }

        let seeker_channels: Vec<[f32; N_CHANNELS]> = (0..self.seekers.len())
            .map(|i| self.seeker_perception(i))
            .collect();
        let mut seeker_actions = Vec::with_capacity(self.seekers.len());
        for (i, channels) in seeker_channels.into_iter().enumerate() {
            let index = self.seekers[i].brain.step(&channels);
            let requested = HsAction::from_index(index);
            let action = self.regulate_action(i, false, requested);
            let reward = if self.seekers[i].alive {
                -TIME_PRESSURE
            } else {
                0.0
            } + self.chase_potential(i);
            self.seekers[i].brain.apply_reward(reward);
            // Remember which objective this reward was measured against.
            self.seekers[i].last_chase_target = chase_targets[i];
            seeker_actions.push(action);
        }

        // ── 4. Movement. ─────────────────────────────────────────────
        for i in 0..self.hiders.len() {
            let action = hider_actions[i];
            self.hiders[i].last_action = action;
            self.move_agent(i, true, action, step);
        }
        for i in 0..self.seekers.len() {
            let action = seeker_actions[i];
            self.seekers[i].last_action = action;
            self.move_agent(i, false, action, step);
        }

        // ── 5. Acoustic traces after movement. ──────────────────────
        self.publish_noise();

        // ── 6. Tags: seeker adjacent to a live hider. ────────────────
        let mut tagged = Vec::new();
        for s in self.seekers.iter() {
            if !s.alive {
                continue;
            }
            for (hi, h) in self.hiders.iter().enumerate() {
                if h.alive && s.position.manhattan(h.position) <= TAG_RADIUS {
                    tagged.push(hi);
                }
            }
        }
        for hi in tagged {
            if self.hiders[hi].alive {
                let pos = self.hiders[hi].position;
                self.hiders[hi].alive = false;
                self.tags += 1;
                self.episode_reward += TAG;
                self.hiders[hi].brain.apply_reward(TAGGED);
                // The pack trail dead-ends at the caught hider: clear it so
                // the pack stops sweeping an empty spot and re-commits to
                // searching for the survivor.
                if self.hider_last_seen.is_some_and(|(p, _)| p == pos) {
                    self.hider_last_seen = None;
                }
                if self
                    .hider_sound_last
                    .is_some_and(|(p, _)| p == self.coarse_noise_position(pos))
                {
                    self.hider_sound_last = None;
                }
                self.event(format!("tag hider=({}, {}) step={step}", pos.x, pos.y));
            }
        }

        // ── 7. Round end + metrics. ──────────────────────────────────
        let all_caught = self.hiders.iter().all(|h| !h.alive);
        if all_caught || self.step >= ROUND_STEPS {
            self.finished = true;
            if !all_caught {
                self.episode_reward += ROUND_SURVIVED;
                self.event(format!(
                    "round_end step={step} hiders_survived={}",
                    self.hiders.iter().filter(|h| h.alive).count()
                ));
            } else {
                self.event(format!("round_end step={step} all_hiders_tagged"));
            }
        }
        self.survival_sum += self.hiders.iter().filter(|h| h.alive).count();
        self.survival_samples += 1;

        for a in self.hiders.iter_mut().chain(self.seekers.iter_mut()) {
            a.noise *= 0.55;
        }
        for a in self.hiders.iter().chain(self.seekers.iter()) {
            let [x, y, z] = a.brain.last_spikes();
            self.spike_total += x + y + z;
        }
        self.spike_samples += 1;

        self.step += 1;
        self.snapshot()
    }

    /// One BFS distance field per seeker (indexed by position, walls and
    /// boxes blocked). Index `i` corresponds to `self.seekers[i]` regardless
    /// of liveness, so per-seeker chase targets can be looked up directly.
    fn distance_fields(&self) -> Vec<Vec<i32>> {
        self.seekers
            .iter()
            .map(|s| {
                let mut dist = vec![i32::MAX; WIDTH * HEIGHT];
                let mut queue = VecDeque::from([s.position]);
                dist[s.position.y as usize * WIDTH + s.position.x as usize] = 0;
                while let Some(current) = queue.pop_front() {
                    let d = dist[current.y as usize * WIDTH + current.x as usize];
                    for next in [
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
                    ] {
                        if !Map::in_bounds(next)
                            || self.map.wall(next)
                            || self.box_at(next)
                            || dist[next.y as usize * WIDTH + next.x as usize] != i32::MAX
                        {
                            continue;
                        }
                        dist[next.y as usize * WIDTH + next.x as usize] = d + 1;
                        queue.push_back(next);
                    }
                }
                dist
            })
            .collect()
    }

    fn box_at(&self, p: Pos) -> bool {
        self.boxes.contains(&Some(p))
    }

    /// Deterministic day/night-like visibility. Brightness peaks halfway
    /// through each cycle and bottoms out at the cycle boundary.
    fn visibility(&self) -> f32 {
        let phase = self.step % VISIBILITY_CYCLE;
        let half = VISIBILITY_CYCLE / 2;
        let distance = phase.min(VISIBILITY_CYCLE - phase);
        MIN_VISIBILITY + (1.0 - MIN_VISIBILITY) * distance as f32 / half as f32
    }

    fn sight_range(&self, hider: bool) -> i32 {
        let base = if hider { SIGHT_HIDER } else { SIGHT_SEEKER };
        (base as f32 * self.visibility()).round().max(2.0) as i32
    }

    /// Check stamina before movement. Energy is committed only after a
    /// requested action actually moves a body or pushes a box; failed no-ops
    /// are not punished.
    fn regulate_action(&mut self, i: usize, hider: bool, requested: HsAction) -> HsAction {
        let agent = if hider {
            &mut self.hiders[i]
        } else {
            &mut self.seekers[i]
        };
        if !agent.alive {
            return HsAction::Wait;
        }
        let cost = match requested {
            HsAction::Explore => ENERGY_EXPLORE,
            HsAction::Flee => ENERGY_FLEE,
            HsAction::Push => ENERGY_PUSH,
            HsAction::Hide | HsAction::Wait => 0.0,
        };
        if cost > agent.energy {
            agent.energy = (agent.energy + ENERGY_WAIT_RECOVERY).min(MAX_ENERGY);
            return HsAction::Wait;
        }
        if matches!(requested, HsAction::Hide) {
            agent.energy = (agent.energy + ENERGY_HIDE_RECOVERY).min(MAX_ENERGY);
        } else if matches!(requested, HsAction::Wait) {
            agent.energy = (agent.energy + ENERGY_WAIT_RECOVERY).min(MAX_ENERGY);
        }
        requested
    }

    fn spend_energy(&mut self, i: usize, hider: bool, action: HsAction) {
        let cost = match action {
            HsAction::Explore => ENERGY_EXPLORE,
            HsAction::Flee => ENERGY_FLEE,
            HsAction::Push => ENERGY_PUSH,
            HsAction::Hide | HsAction::Wait => 0.0,
        };
        let agent = if hider {
            &mut self.hiders[i]
        } else {
            &mut self.seekers[i]
        };
        agent.energy = (agent.energy - cost).max(0.0);
    }

    /// Movement noise is a short-lived acoustic trace. It is intentionally
    /// not encoded as a new brain channel: it feeds the existing team trail,
    /// so the same GNW memory machinery handles sight and sound. Loudness
    /// attenuates linearly with Manhattan distance and only successful moves
    /// emit a trace. Sound is quantized to a coarse cell and decays faster
    /// than visual intel, so it is a search cue rather than perfect GPS.
    fn publish_noise(&mut self) {
        let hider_noise = self
            .hiders
            .iter()
            .filter(|a| a.alive && a.noise >= ENERGY_MIN_ACTION)
            .filter_map(|h| {
                let distance = self
                    .seekers
                    .iter()
                    .filter(|s| s.alive)
                    .map(|s| h.position.manhattan(s.position))
                    .min()?;
                let loudness = h.noise * (1.0 - distance as f32 / NOISE_RADIUS as f32);
                (loudness >= ENERGY_MIN_ACTION).then_some((loudness, h.position))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((_, p)) = hider_noise {
            // Visual sightings are more precise and were published earlier in
            // this tick; sound is only a fallback when no visual sighting was
            // made in this tick. A stale visual trail must not suppress new
            // sound, but sound gets its own coarse/short-lived memory.
            if self.visual_hider_step != Some(self.step) {
                let coarse = self.coarse_noise_position(p);
                self.event(format!(
                    "noise hider@({}, {}) step={}",
                    coarse.x, coarse.y, self.step
                ));
                self.hider_sound_last = Some((coarse, self.step));
            }
        }
        let seeker_noise = self
            .seekers
            .iter()
            .filter(|a| a.alive && a.noise >= ENERGY_MIN_ACTION)
            .filter_map(|s| {
                let distance = self
                    .hiders
                    .iter()
                    .filter(|h| h.alive)
                    .map(|h| s.position.manhattan(h.position))
                    .min()?;
                let loudness = s.noise * (1.0 - distance as f32 / NOISE_RADIUS as f32);
                (loudness >= ENERGY_MIN_ACTION).then_some((loudness, s.position))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((_, p)) = seeker_noise {
            if self.visual_seeker_step != Some(self.step) {
                let coarse = self.coarse_noise_position(p);
                self.event(format!(
                    "noise seeker@({}, {}) step={}",
                    coarse.x, coarse.y, self.step
                ));
                self.seeker_sound_last = Some((coarse, self.step));
            }
        }
    }

    fn coarse_noise_position(&self, p: Pos) -> Pos {
        Pos {
            x: (p.x / 3) * 3 + 1,
            y: (p.y / 3) * 3 + 1,
        }
    }

    /// hider perception: 8 channels.
    /// 0 threat_visible · 1 threat_close · 2 stamina/explore_budget ·
    /// 3 box_adjacent · 4 doorway_adjacent · 5 barricade_useful ·
    /// 6 teammate_visible · 7 intel_fresh.
    /// Visibility (0/1/6) requires a clear LINE OF SIGHT: walls and boxes are
    /// real cover — a seeker behind a wall is simply not seen. Channel 7 is
    /// team danger intel (graded freshness): a teammate was spotted recently,
    /// so the whole team scatters, not just the one seen.
    fn hider_perception(&self, i: usize) -> [f32; N_CHANNELS] {
        let h = &self.hiders[i];
        let sight = self.sight_range(true);
        let threat_d = self
            .nearest_agent_in_sight(h.position, false, sight)
            .map(|p| h.position.manhattan(p))
            .unwrap_or(i32::MAX);
        let box_adj = self
            .boxes
            .iter()
            .any(|b| b.is_some() && h.position.manhattan(b.unwrap()) == 1);
        let door_adj = self
            .map
            .doorways
            .iter()
            .any(|d| h.position.manhattan(*d) == 1);
        let barricade = self.map.doorways.iter().any(|d| self.box_at(*d));
        let teammate = self.hiders.iter().any(|o| {
            o.alive
                && o.id != h.id
                && h.position.manhattan(o.position) <= sight
                && self.has_los(h.position, o.position)
        });
        [
            if threat_d <= sight { 1.0 } else { 0.0 },
            if threat_d <= 3 { 1.0 } else { 0.0 },
            self.hiders[i].energy, // stamina-aware shelter/explore bias
            if box_adj { 1.0 } else { 0.0 },
            if door_adj { 1.0 } else { 0.0 },
            if barricade { 1.0 } else { 0.0 },
            if teammate { 1.0 } else { 0.0 },
            self.intel_fresh(true),
        ]
    }

    /// seeker perception: 8 channels.
    /// 0 hider_visible · 1 hider_close · 2 stamina/patrol_budget ·
    /// 3 box_adjacent · 4 doorway_adjacent · 5 barricade_blocking ·
    /// 6 intel_memory · 7 intel_fresh.
    /// Visibility requires LINE OF SIGHT. Channel 6 is the pack's ROUND
    /// memory: a hider has been sighted at least once, so the pack sweeps the
    /// hideouts near the last spotted area. Channel 7 is the fading hot trail
    /// — a recent sighting, so the whole pack converges on the last position.
    fn seeker_perception(&self, i: usize) -> [f32; N_CHANNELS] {
        let s = &self.seekers[i];
        let sight = self.sight_range(false);
        let target_d = self
            .nearest_agent_in_sight(s.position, true, sight)
            .map(|p| s.position.manhattan(p))
            .unwrap_or(i32::MAX);
        let box_adj = self
            .boxes
            .iter()
            .any(|b| b.is_some() && s.position.manhattan(b.unwrap()) == 1);
        let door_adj = self
            .map
            .doorways
            .iter()
            .any(|d| s.position.manhattan(*d) == 1);
        let blocked = self.map.doorways.iter().any(|d| self.box_at(*d));
        [
            if target_d <= sight { 1.0 } else { 0.0 },
            if target_d <= 3 { 1.0 } else { 0.0 },
            self.seekers[i].energy, // stamina-aware patrol/explore bias
            if box_adj { 1.0 } else { 0.0 },
            if door_adj { 1.0 } else { 0.0 },
            if blocked { 1.0 } else { 0.0 },
            // Memory is graded against the trail: while the trail is hot the
            // pack CHASES (Flee wins); as it cools the pack SWEEPS the
            // hideouts (Hide wins). Never both at full strength, so the
            // pack isn't torn between chasing and spreading.
            if self.hider_last_seen.is_some() {
                1.0 - self.intel_fresh(false)
            } else {
                0.0
            },
            self.intel_fresh(false),
        ]
    }

    /// Nearest opposing agent within `sight` cells with an unobstructed line
    /// of sight from `p`. `None` if nobody is visible.
    fn nearest_agent_in_sight(&self, p: Pos, hider: bool, sight: i32) -> Option<Pos> {
        let agents: Box<dyn Iterator<Item = &HsAgent> + '_> = if hider {
            Box::new(self.hiders.iter().filter(|a| a.alive))
        } else {
            Box::new(self.seekers.iter().filter(|a| a.alive))
        };
        agents
            .filter(|a| p.manhattan(a.position) <= sight && self.has_los(p, a.position))
            .map(|a| (a.position, p.manhattan(a.position)))
            .min_by_key(|&(_, d)| d)
            .map(|(p, _)| p)
    }

    /// Bresenham line-of-sight: `true` if the straight line from `from` to
    /// `to` crosses only open floor. Walls AND boxes block sight — a box
    /// pushed into a doorway is genuine cover, and a hider behind a wall is
    /// invisible.
    fn has_los(&self, from: Pos, to: Pos) -> bool {
        if from == to {
            return true;
        }
        let dx = (to.x - from.x).abs();
        let dy = -(to.y - from.y).abs();
        let sx = if from.x < to.x { 1 } else { -1 };
        let sy = if from.y < to.y { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (from.x, from.y);
        loop {
            if x == to.x && y == to.y {
                return true;
            }
            let p = Pos { x, y };
            if p != from && (self.map.wall(p) || self.box_at(p)) {
                return false;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Graded freshness of the team's last sighting, in 0..1: 1.0 right
    /// after the spot, decaying linearly to 0.0 over `INTEL_WINDOW` steps.
    /// `hider_team` selects the trail: `false` = the pack's hider trail
    /// (seekers), `true` = danger intel (hiders).
    fn visual_fresh(&self, hider_team: bool) -> f32 {
        let visual = if hider_team {
            self.seeker_last_seen
        } else {
            self.hider_last_seen
        };
        visual
            .map(|(_, seen_step)| {
                let age = self.step.saturating_sub(seen_step);
                if age >= INTEL_WINDOW {
                    0.0
                } else {
                    1.0 - age as f32 / INTEL_WINDOW as f32
                }
            })
            .unwrap_or(0.0)
    }

    fn intel_fresh(&self, hider_team: bool) -> f32 {
        let visual_fresh = self.visual_fresh(hider_team);
        let sound = if hider_team {
            self.seeker_sound_last
        } else {
            self.hider_sound_last
        };
        let sound_fresh = sound
            .map(|(_, seen_step)| {
                let age = self.step.saturating_sub(seen_step);
                if age >= SOUND_WINDOW {
                    0.0
                } else {
                    SOUND_CONFIDENCE * (1.0 - age as f32 / SOUND_WINDOW as f32)
                }
            })
            .unwrap_or(0.0);
        visual_fresh.max(sound_fresh)
    }

    fn acoustic_target(&self, hider_team: bool) -> Option<Pos> {
        let trail = if hider_team {
            self.seeker_sound_last
        } else {
            self.hider_sound_last
        }?;
        let age = self.step.saturating_sub(trail.1);
        (age < SOUND_WINDOW).then_some(trail.0)
    }

    /// Team intel: any seeker with line of sight to a live hider publishes
    /// its position to the pack (fading trail); any hider with sight of a
    /// live seeker publishes danger intel. A changed sighting emits an event.
    fn publish_sightings(&mut self) {
        let mut hider_spotted: Option<Pos> = None;
        'pack: for s in self.seekers.iter().filter(|s| s.alive) {
            for h in self.hiders.iter().filter(|h| h.alive) {
                if s.position.manhattan(h.position) <= self.sight_range(false)
                    && self.has_los(s.position, h.position)
                {
                    hider_spotted = Some(h.position);
                    break 'pack;
                }
            }
        }
        let mut seeker_spotted: Option<Pos> = None;
        'danger: for h in self.hiders.iter().filter(|h| h.alive) {
            for s in self.seekers.iter().filter(|s| s.alive) {
                if h.position.manhattan(s.position) <= self.sight_range(true)
                    && self.has_los(h.position, s.position)
                {
                    seeker_spotted = Some(s.position);
                    break 'danger;
                }
            }
        }
        if let Some(p) = hider_spotted {
            // A GENUINELY new find: the trail went cold (or this is the first
            // sighting) AND the position moved. Re-sightings during an active
            // chase refresh the trail (channel 7 stays hot) but do not re-trigger
            // SPOT — otherwise the reward fires every tick of a chase and floods
            // the brains. The same gate rate-limits intel events so the event
            // feed isn't drowned by chase noise (tag/barricade events stay visible).
            let cold_find = self.last_spot_step == 0
                || self.step.saturating_sub(self.last_spot_step) > INTEL_WINDOW;
            let moved = self.hider_last_seen.is_none_or(|(old, _)| old != p);
            if moved && cold_find {
                self.last_spot_step = self.step;
                self.event(format!("intel hider@({}, {}) step={}", p.x, p.y, self.step));
                // Spotting prey reinforces the whole live pack: the hunt is on.
                self.episode_reward += SPOT;
                for s in self.seekers.iter_mut() {
                    if s.alive {
                        s.brain.apply_reward(SPOT);
                    }
                }
            }
            self.hider_last_seen = Some((p, self.step));
            self.visual_hider_step = Some(self.step);
        }
        if let Some(p) = seeker_spotted {
            // Danger intel is rate-limited independently of the pack trail:
            // only a genuinely new sighting hits the event feed.
            let cold_find = self.last_danger_step == 0
                || self.step.saturating_sub(self.last_danger_step) > INTEL_WINDOW;
            let moved = self.seeker_last_seen.is_none_or(|(old, _)| old != p);
            if moved && cold_find {
                self.last_danger_step = self.step;
                self.event(format!(
                    "intel seeker@({}, {}) step={}",
                    p.x, p.y, self.step
                ));
            }
            self.seeker_last_seen = Some((p, self.step));
            self.visual_seeker_step = Some(self.step);
        }
    }

    fn nearest_agent(&self, p: Pos, hider: bool) -> Option<Pos> {
        let agents: Box<dyn Iterator<Item = &HsAgent> + '_> = if hider {
            Box::new(self.hiders.iter().filter(|a| a.alive))
        } else {
            Box::new(self.seekers.iter().filter(|a| a.alive))
        };
        agents
            .map(|a| (a.position, p.manhattan(a.position)))
            .min_by_key(|&(_, d)| d)
            .map(|(p, _)| p)
    }

    /// Potential-based flee reward: positive when the hider moved away from
    /// the nearest threat, negative when it walked closer. Honest — it never
    /// prescribes *which* action, only that safety is good. Escaping beyond
    /// reach (threat became unreachable, e.g. a barricade) is the maximum
    /// safety gain.
    fn flee_potential(&self, i: usize) -> f32 {
        let h = &self.hiders[i];
        if h.threat_dist == i32::MAX && h.prev_threat_dist != i32::MAX {
            // The threat is no longer reachable: escape worked completely.
            return (FLEE_GAIN * 4.0).clamp(-0.5, 0.5);
        }
        if h.threat_dist == i32::MAX {
            return 0.0;
        }
        let delta = h.threat_dist.saturating_sub(h.prev_threat_dist) as f32;
        (delta * FLEE_GAIN).clamp(-0.5, 0.5)
    }

    /// Potential-based chase reward: positive when the seeker closed in on
    /// the pack's CURRENT objective (the fresh trail, or the nearest hider).
    /// The objective becoming unreachable (barricaded away) is the worst
    /// outcome. Measured against the objective rather than the hider itself
    /// so converging on the last sighting is rewarded, not punished.
    fn chase_potential(&self, i: usize) -> f32 {
        let s = &self.seekers[i];
        // The objective switched since the last reward (tag cleared the
        // trail, or a sighting teleported it): the distance delta carries no
        // signal, so stay neutral instead of punishing the pack for nothing.
        if s.last_chase_target != self.chase_target(s.position) {
            return 0.0;
        }
        if s.chase_dist == i32::MAX && s.prev_chase_dist != i32::MAX {
            return -(CHASE_GAIN * 4.0).clamp(-0.5, 0.5);
        }
        if s.chase_dist == i32::MAX {
            return 0.0;
        }
        let delta = s.prev_chase_dist.saturating_sub(s.chase_dist) as f32;
        (delta * CHASE_GAIN).clamp(-0.5, 0.5)
    }

    /// BFS next step from `from` toward `target`, avoiding walls and boxes.
    fn next_step(&self, from: Pos, target: Pos) -> Option<Pos> {
        if from == target {
            return None;
        }
        let mut previous = vec![None; WIDTH * HEIGHT];
        let mut queue = VecDeque::from([from]);
        previous[from.y as usize * WIDTH + from.x as usize] = Some(from);
        while let Some(current) = queue.pop_front() {
            for next in [
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
            ] {
                if !Map::in_bounds(next) || self.map.wall(next) || self.box_at(next) {
                    continue;
                }
                let idx = next.y as usize * WIDTH + next.x as usize;
                if previous[idx].is_some() {
                    continue;
                }
                previous[idx] = Some(current);
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

    fn move_agent(&mut self, i: usize, hider: bool, action: HsAction, step: usize) {
        let occupied = |p: Pos, self_idx: usize| {
            self.hiders
                .iter()
                .any(|a| a.alive && a.id != self_idx && a.position == p)
                || self
                    .seekers
                    .iter()
                    .any(|a| a.alive && a.id != self_idx && a.position == p)
        };

        // ── Push: shove an adjacent box one cell away from the agent. ──
        if action == HsAction::Push {
            let agent_pos = if hider {
                self.hiders[i].position
            } else {
                self.seekers[i].position
            };
            let box_idx = self
                .boxes
                .iter()
                .position(|b| b.is_some() && agent_pos.manhattan(b.unwrap()) == 1);
            if let Some(bi) = box_idx {
                let box_pos = self.boxes[bi].unwrap();
                let dir = Pos {
                    x: (box_pos.x - agent_pos.x).signum(),
                    y: (box_pos.y - agent_pos.y).signum(),
                };
                let dest = Pos {
                    x: box_pos.x + dir.x,
                    y: box_pos.y + dir.y,
                };
                let was_blocked = self.map.doorways.iter().any(|d| self.box_at(*d));
                let ok = Map::in_bounds(dest)
                    && !self.map.wall(dest)
                    && !self.box_at(dest)
                    && !occupied(dest, i);
                if ok {
                    self.boxes[bi] = Some(dest);
                    self.spend_energy(i, hider, action);
                    let noise = if hider {
                        self.hiders[i].energy
                    } else {
                        self.seekers[i].energy
                    };
                    if hider {
                        self.hiders[i].noise = 0.65 * noise;
                    } else {
                        self.seekers[i].noise = 0.65 * noise;
                    }
                    let now_blocked = self.map.doorways.iter().any(|d| self.box_at(*d));
                    if !was_blocked && now_blocked {
                        if hider {
                            self.episode_reward += BARRICADE;
                            self.hiders[i].brain.apply_reward(BARRICADE);
                            self.event(format!(
                                "barricade hider{} door=({}, {}) step={step}",
                                i, dest.x, dest.y
                            ));
                        } else {
                            self.episode_reward -= BARRICADE;
                        }
                    } else if was_blocked && !now_blocked && !hider {
                        self.episode_reward += UNBLOCK;
                        self.seekers[i].brain.apply_reward(UNBLOCK);
                        self.event(format!(
                            "unblock seeker{} door=({}, {}) step={step}",
                            i, dest.x, dest.y
                        ));
                    }
                }
            }
            return;
        }

        let from = if hider {
            self.hiders[i].position
        } else {
            self.seekers[i].position
        };
        let target = if hider {
            match action {
                HsAction::Explore => self.explore_waypoint(from),
                HsAction::Hide => self
                    .map
                    .shelters
                    .iter()
                    .copied()
                    .min_by_key(|s| from.manhattan(*s)),
                HsAction::Flee => self.flee_target(from),
                _ => None,
            }
        } else {
            match action {
                HsAction::Explore => self.explore_waypoint(from),
                HsAction::Hide => self.patrol_target(from, i),
                HsAction::Flee => self.chase_target(from),
                _ => None,
            }
        };
        let Some(target) = target else { return };
        let Some(next) = self.next_step(from, target) else {
            return;
        };
        if occupied(next, i) {
            return;
        }
        // Hunters are faster than hiders: a seeker in active pursuit (Flee)
        // takes a second step along the path, so a chase can actually resolve
        // instead of trailing at equal speed forever.
        let mut final_pos = next;
        if !hider && action == HsAction::Flee {
            if let Some(second) = self.next_step(next, target) {
                if second != from && !occupied(second, i) {
                    final_pos = second;
                }
            }
        }
        let moved = final_pos != from;
        if moved {
            self.spend_energy(i, hider, action);
        }
        if hider {
            self.hiders[i].position = final_pos;
            self.hiders[i].noise = if moved {
                let base = match action {
                    HsAction::Flee => 0.9,
                    HsAction::Explore => 0.45,
                    _ => 0.25,
                };
                base * (0.35 + 0.65 * self.hiders[i].energy)
            } else {
                0.0
            };
        } else {
            self.seekers[i].position = final_pos;
            self.seekers[i].noise = if moved {
                let base = match action {
                    HsAction::Flee => 0.9,
                    HsAction::Explore => 0.45,
                    _ => 0.25,
                };
                base * (0.35 + 0.65 * self.seekers[i].energy)
            } else {
                0.0
            };
        }
    }

    fn explore_waypoint(&self, from: Pos) -> Option<Pos> {
        self.map
            .shelters
            .iter()
            .copied()
            .max_by_key(|s| from.manhattan(*s))
    }

    /// A flee target: the shelter that maximizes distance from the threat.
    /// With no visible threat but fresh danger intel, flee from the last
    /// spotted seeker position instead — the team scatters away from the hunt.
    fn flee_target(&self, from: Pos) -> Option<Pos> {
        let threat = self.nearest_agent(from, false).or_else(|| {
            if self.visual_fresh(true) > 0.0 {
                self.seeker_last_seen
                    .map(|(p, _)| p)
                    .or_else(|| self.acoustic_target(true))
            } else {
                self.acoustic_target(true)
            }
        })?;
        self.map
            .shelters
            .iter()
            .copied()
            .max_by_key(|s| s.manhattan(threat))
    }

    /// A seeker chase target: the pack's freshest trail (the last hider
    /// position any teammate spotted), or the nearest known hider if the
    /// trail has gone cold. Lets the pack converge on a hider that broke
    /// line of sight.
    fn chase_target(&self, from: Pos) -> Option<Pos> {
        if self.visual_fresh(false) > 0.0 {
            if let Some(target) = self.hider_last_seen.map(|(p, _)| p) {
                return Some(target);
            }
        }
        if let Some(target) = self.acoustic_target(false) {
            return Some(target);
        }
        self.nearest_agent(from, true)
    }

    /// A seeker patrol target (Hide action): sweep the hideouts around the
    /// last sighting. Once the pack has a memory of where a hider vanished,
    /// each hunter fans out to a *different* shelter ranked by distance to
    /// that spot — a natural pack search, not a scripted formation. With no
    /// memory yet, patrol the nearest shelter.
    fn patrol_target(&self, from: Pos, i: usize) -> Option<Pos> {
        let last = if self.visual_fresh(false) > 0.0 {
            self.hider_last_seen.map(|(p, _)| p)
        } else {
            self.acoustic_target(false)
        };
        match last {
            Some(anchor) => {
                let mut ranked: Vec<Pos> = self.map.shelters.clone();
                ranked.sort_by_key(|s| s.manhattan(anchor));
                Some(ranked[i % ranked.len()])
            }
            None => self
                .map
                .shelters
                .iter()
                .copied()
                .min_by_key(|s| from.manhattan(*s)),
        }
    }

    fn event(&mut self, message: String) {
        if self.events.len() >= 24 {
            self.events.pop_front();
        }
        self.events.push_back(message);
    }

    pub fn snapshot_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.snapshot())
    }

    pub fn snapshot(&self) -> HideSeekSnapshot {
        let samples = self.survival_samples.max(1) as f32;
        let spike_samples = self.spike_samples.max(1) as f32;
        HideSeekSnapshot {
            schema_version: 6,
            visibility: self.visibility(),
            mission: "hide_seek".to_string(),
            width: WIDTH,
            height: HEIGHT,
            step: self.step,
            finished: self.finished,
            survival: self.survival_sum as f32 / samples,
            tags: self.tags,
            walls: self
                .map
                .walls
                .iter()
                .enumerate()
                .filter(|(_, w)| **w)
                .map(|(i, _)| [(i % WIDTH) as i32, (i / WIDTH) as i32])
                .collect(),
            doorways: self.map.doorways.iter().map(|p| [p.x, p.y]).collect(),
            boxes: self
                .boxes
                .iter()
                .filter_map(|b| b.map(|p| [p.x, p.y]))
                .collect(),
            shelters: self.map.shelters.iter().map(|p| [p.x, p.y]).collect(),
            hiders: self
                .hiders
                .iter()
                .map(|h| HsAgentSnapshot {
                    id: h.id,
                    team: Team::Hider.as_str().to_string(),
                    x: h.position.x,
                    y: h.position.y,
                    action: h.last_action.as_str().to_string(),
                    alive: h.alive,
                    energy: h.energy,
                    noise: h.noise,
                    brain: self.brain_snapshot(h.brain.as_ref()),
                })
                .collect(),
            seekers: self
                .seekers
                .iter()
                .map(|s| HsAgentSnapshot {
                    id: s.id,
                    team: Team::Seeker.as_str().to_string(),
                    x: s.position.x,
                    y: s.position.y,
                    action: s.last_action.as_str().to_string(),
                    alive: s.alive,
                    energy: s.energy,
                    noise: s.noise,
                    brain: self.brain_snapshot(s.brain.as_ref()),
                })
                .collect(),
            brain_activity: self.spike_total / spike_samples,
            training: self.training.clone(),
            events: self.events.iter().cloned().collect(),
        }
    }

    fn brain_snapshot(&self, brain: &SpikingBrain) -> BrainSnapshot {
        BrainSnapshot {
            action_rates: brain.action_rates().to_vec(),
            sensor_rates: brain.sensor_channel_rates(),
            workspace_rates: brain.workspace_group_rates(),
            weight_heat: brain.weight_heat(),
            region_gate: brain.region_gate().to_vec(),
            dopamine: brain.dopamine(),
            reward: brain.last_reward(),
            baseline: brain.baseline(),
            spikes: brain.last_spikes(),
            spike_history: brain.spike_log().iter().copied().collect(),
        }
    }
}

impl Default for HideSeekArena {
    fn default() -> Self {
        Self::new(0x5EED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_connected_and_has_features() {
        let map = generate_map(0x5EED);
        assert!(map.wall(Pos { x: 0, y: 0 }));
        assert!(map.wall(Pos {
            x: 5,
            y: HEIGHT as i32 - 1
        }));
        assert!(!map.doorways.is_empty());
        assert!(map.shelters.len() >= N_SHELTERS);
        for s in &map.shelters {
            let min_door = map.doorways.iter().map(|d| s.manhattan(*d)).min().unwrap();
            assert!(
                min_door >= 8,
                "shelter {s:?} too close to a door: {min_door}"
            );
        }
        assert_eq!(map.box_starts.len(), N_BOXES);
    }

    #[test]
    fn barricade_blocks_path() {
        let mut arena = HideSeekArena::new(0x5EED);
        // Flatten the map into a single room with one central wall + doorway.
        arena.map.walls.fill(false);
        for x in 0..WIDTH {
            arena.map.walls[x] = true;
            arena.map.walls[(HEIGHT - 1) * WIDTH + x] = true;
        }
        for y in 0..HEIGHT {
            arena.map.walls[y * WIDTH] = true;
            arena.map.walls[y * WIDTH + WIDTH - 1] = true;
        }
        for y in 1..HEIGHT - 1 {
            arena.map.walls[y * WIDTH + 24] = true;
        }
        arena.map.doorways.clear();
        let door = Pos { x: 24, y: 18 };
        arena.map.doorways.push(door);
        arena.map.walls[18 * WIDTH + 24] = false;

        let hider_pos = Pos { x: 22, y: 18 }; // adjacent to the box at (23, 18)
        let seeker_pos = Pos { x: 42, y: 18 };
        arena.hiders = vec![HsAgent::new(
            0,
            hider_pos,
            Box::new(SpikingBrain::train_init_with(
                1,
                BrainPrior::hide_seek_hider(),
            )),
        )];
        arena.seekers = vec![HsAgent::new(
            0,
            seeker_pos,
            Box::new(SpikingBrain::train_init_with(
                2,
                BrainPrior::hide_seek_seeker(),
            )),
        )];

        let before = arena.distance_fields();
        let d_before = before[0][hider_pos.y as usize * WIDTH + hider_pos.x as usize];
        assert!(d_before < i32::MAX, "path must exist through the doorway"); // Push a box into the doorway (box sits between hider and the gap).
        arena.boxes = vec![Some(Pos { x: 23, y: 18 })];
        assert_eq!(arena.hiders[0].position.manhattan(Pos { x: 23, y: 18 }), 1);
        arena.hiders[0].last_action = HsAction::Push;
        arena.move_agent(0, true, HsAction::Push, 0);
        assert_eq!(
            arena.boxes[0],
            Some(door),
            "box must be pushed into the doorway"
        );

        let after = arena.distance_fields();
        let d_after = after[0][hider_pos.y as usize * WIDTH + hider_pos.x as usize];
        assert_eq!(d_after, i32::MAX, "barricaded doorway must block the path");
        assert_ne!(d_before, d_after);
    }

    #[test]
    fn los_blocked_by_walls_and_boxes() {
        let mut arena = HideSeekArena::new(0x5EED);
        // Flatten into an open field (borders stay walls).
        arena.map.walls.fill(false);
        for x in 0..WIDTH {
            arena.map.walls[x] = true;
            arena.map.walls[(HEIGHT - 1) * WIDTH + x] = true;
        }
        for y in 0..HEIGHT {
            arena.map.walls[y * WIDTH] = true;
            arena.map.walls[y * WIDTH + WIDTH - 1] = true;
        }
        let a = Pos { x: 5, y: 5 };
        let b = Pos { x: 12, y: 5 };
        assert!(arena.has_los(a, b), "open floor must give line of sight");

        // A wall on the line blocks sight.
        arena.map.walls[5 * WIDTH + 8] = true;
        assert!(!arena.has_los(a, b), "wall must block line of sight");
        assert!(
            arena.has_los(a, Pos { x: 7, y: 5 }),
            "near side of the wall stays visible"
        );
        arena.map.walls[5 * WIDTH + 8] = false;

        // A box on the line blocks sight too (barricades are real cover).
        arena.boxes = vec![Some(Pos { x: 8, y: 5 })];
        assert!(!arena.has_los(a, b), "box must block line of sight");
        arena.boxes.clear();
        assert!(arena.has_los(a, b), "removing the box restores sight");
    }

    #[test]
    fn pack_intel_shares_sightings() {
        let mut arena = HideSeekArena::new(0x5EED);
        // Open field.
        arena.map.walls.fill(false);
        for x in 0..WIDTH {
            arena.map.walls[x] = true;
            arena.map.walls[(HEIGHT - 1) * WIDTH + x] = true;
        }
        for y in 0..HEIGHT {
            arena.map.walls[y * WIDTH] = true;
            arena.map.walls[y * WIDTH + WIDTH - 1] = true;
        }
        let hider_pos = Pos { x: 10, y: 10 };
        arena.hiders = vec![HsAgent::new(
            0,
            hider_pos,
            Box::new(SpikingBrain::train_init_with(
                1,
                BrainPrior::hide_seek_hider(),
            )),
        )];
        // Seeker A sees the hider; seeker B is far away with no line of sight.
        arena.seekers = vec![
            HsAgent::new(
                0,
                Pos { x: 11, y: 10 },
                Box::new(SpikingBrain::train_init_with(
                    2,
                    BrainPrior::hide_seek_seeker(),
                )),
            ),
            HsAgent::new(
                1,
                Pos { x: 40, y: 20 },
                Box::new(SpikingBrain::train_init_with(
                    3,
                    BrainPrior::hide_seek_seeker(),
                )),
            ),
        ];

        arena.publish_sightings();
        assert_eq!(arena.hider_last_seen, Some((hider_pos, 0)));
        // The far seeker still gets the pack trail even without sight.
        assert!(arena.intel_fresh(false) > 0.0, "pack intel must propagate");
        let perc = arena.seeker_perception(1);
        assert!(
            perc[0] == 0.0 && perc[7] > 0.0,
            "far seeker sees no hider but feels the fresh trail: {perc:?}"
        );
        // Chase converges on the trail, not on the (invisible) hider's side
        // of the map: B's target is exactly the last seen position.
        assert_eq!(arena.chase_target(Pos { x: 40, y: 20 }), Some(hider_pos));

        // Hiders get danger intel the same way.
        assert!(arena.intel_fresh(true) > 0.0);
        assert_eq!(arena.seeker_last_seen, Some((Pos { x: 11, y: 10 }, 0)));
        let h_perc = arena.hider_perception(0);
        assert!(
            h_perc[0] == 1.0 && h_perc[7] > 0.0,
            "hider sees the adjacent seeker AND feels danger intel: {h_perc:?}"
        );
    }

    #[test]
    fn snapshot_exposes_dynamic_telemetry() {
        let arena = HideSeekArena::new(0xCAFE);
        let value: serde_json::Value =
            serde_json::from_str(&arena.snapshot_json().unwrap()).unwrap();
        assert_eq!(value["schema_version"], 6);
        assert!(value["visibility"].as_f64().unwrap().is_finite());
        assert!(value["hiders"][0]["energy"].as_f64().unwrap().is_finite());
        assert!(value["hiders"][0]["noise"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn successful_movement_emits_energy_scaled_noise() {
        let mut arena = HideSeekArena::new(0xCAFE);
        arena.map.walls.fill(false);
        arena.map.shelters = vec![Pos { x: 20, y: 10 }];
        arena.hiders[0].position = Pos { x: 10, y: 10 };
        arena.hiders[0].energy = 0.8;
        arena.move_agent(0, true, HsAction::Explore, 0);
        assert_ne!(arena.hiders[0].position, Pos { x: 10, y: 10 });
        assert!(arena.hiders[0].noise > 0.0);
        assert!(arena.hiders[0].noise <= 1.0);

        let visual = Pos { x: 12, y: 12 };
        arena.hider_last_seen = Some((visual, 0));
        arena.visual_hider_step = Some(0);
        arena.hiders[0].noise = 0.9;
        arena.publish_noise();
        assert_eq!(arena.hider_last_seen, Some((visual, 0)));
    }

    #[test]
    fn deterministic_replays() {
        let mut a = HideSeekArena::new(0xABCD);
        let mut b = HideSeekArena::new(0xABCD);
        a.run_for_test(60);
        b.run_for_test(60);
        assert_eq!(
            a.snapshot_json().expect("serialize"),
            b.snapshot_json().expect("serialize"),
            "identical seeds must replay identically"
        );
    }

    #[test]
    fn all_brains_finite_and_round_ends() {
        let mut arena = HideSeekArena::new(0x1234);
        arena.run_for_test(400);
        assert!(arena.finished, "round must end by tag or timeout");
        let s = arena.snapshot();
        for h in &s.hiders {
            assert!(h.brain.action_rates.iter().all(|r| r.is_finite()));
            assert!(h.brain.weight_heat.iter().all(|w| w.is_finite()));
            assert!(!h.brain.spike_history.is_empty());
            assert!(h.brain.spike_history.len() <= 64);
            assert!(h
                .brain
                .spike_history
                .iter()
                .all(|row| row.len() == 3 && row.iter().all(|v| v.is_finite())));
        }
        for k in &s.seekers {
            assert!(k.brain.action_rates.iter().all(|r| r.is_finite()));
            assert!(!k.brain.spike_history.is_empty());
        }
        assert!(s.brain_activity.is_finite());
    }

    #[test]
    fn hiders_survive_when_hiding() {
        let mut arena = HideSeekArena::new(0x7777);
        arena.seekers.clear();
        arena.run_for_test(ROUND_STEPS + 5);
        assert!(arena.finished);
        assert_eq!(arena.snapshot().tags, 0);
        let s = arena.snapshot();
        assert!(s.survival > 0.0);
        assert_eq!(s.hiders.iter().filter(|h| h.alive).count(), N_HIDERS);
    }

    #[test]
    fn training_produces_curve() {
        let mut arena = HideSeekArena::new(0xFEED);
        arena.train(3, 200);
        let t = &arena.training;
        assert_eq!(t.episodes, 3);
        assert_eq!(t.survival.len(), 3);
        assert_eq!(t.tags.len(), 3);
        assert!(t.rewards.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn visibility_cycle_is_deterministic_and_bounded() {
        let mut arena = HideSeekArena::new(0xCAFE);
        arena.step = 0;
        let dusk = arena.visibility();
        arena.step = VISIBILITY_CYCLE / 2;
        let noon = arena.visibility();
        arena.step = VISIBILITY_CYCLE;
        let next_dusk = arena.visibility();
        assert!(dusk >= MIN_VISIBILITY && dusk < noon);
        assert!((dusk - next_dusk).abs() < f32::EPSILON);
        assert!((noon - 1.0).abs() < f32::EPSILON);
        assert!(arena.sight_range(true) >= 2);
        assert!(arena.sight_range(false) >= 2);
    }

    #[test]
    fn exhausted_agent_is_forced_to_wait_and_recovers() {
        let mut arena = HideSeekArena::new(0xCAFE);
        arena.hiders[0].energy = 0.02;
        let action = arena.regulate_action(0, true, HsAction::Flee);
        assert_eq!(action, HsAction::Wait);
        assert!(arena.hiders[0].energy > 0.02);
        assert!(arena.hiders[0].energy <= MAX_ENERGY);
    }

    #[test]
    fn movement_noise_is_fallback_team_intel() {
        let mut arena = HideSeekArena::new(0xCAFE);
        arena.map.walls.fill(false);
        let hider = Pos { x: 10, y: 10 };
        let seeker = Pos { x: 16, y: 10 };
        arena.hiders[0].position = hider;
        arena.seekers[0].position = seeker;
        arena.hider_last_seen = None;
        arena.hiders[0].noise = 0.9;
        arena.publish_noise();
        assert_eq!(
            arena.hider_sound_last,
            Some((arena.coarse_noise_position(hider), 0))
        );
        assert!(arena.hider_last_seen.is_none());
        let coarse = arena.coarse_noise_position(hider);
        assert!(arena
            .events
            .iter()
            .any(|e| { e == &format!("noise hider@({}, {}) step=0", coarse.x, coarse.y) }));

        // A fresh visual trail wins over a noisy position.
        let visual = Pos { x: 11, y: 11 };
        arena.step = 1;
        arena.hider_last_seen = Some((visual, 0));
        arena.visual_hider_step = Some(arena.step);
        arena.hiders[0].position = hider;
        arena.hiders[0].noise = 0.9;
        arena.publish_noise();
        assert_eq!(arena.hider_last_seen, Some((visual, 0)));
    }

    #[test]
    fn continuous_training_accumulates_episodes() {
        // Continuous learning: many train_one calls accumulate the curve and
        // keep the learned weights (no re-perturb, no reset of progress).
        let mut arena = HideSeekArena::new(0xBEEF);
        for _ in 0..5 {
            arena.train_one(150);
        }
        let t = &arena.training;
        assert_eq!(t.episodes, 5);
        assert_eq!(t.survival.len(), 5);
        assert_eq!(t.tags.len(), 5);
        assert!(t.rewards.iter().all(|r| r.is_finite()));
        // Brains learned: weights differ from a fresh perturbed prior.
        let fresh = SpikingBrain::train_init_with(0x1111, BrainPrior::hide_seek_hider());
        assert_ne!(arena.hiders[0].brain.weights(), fresh.weights());
    }

    impl HideSeekArena {
        fn run_for_test(&mut self, max_steps: usize) {
            for _ in 0..max_steps {
                if self.finished {
                    break;
                }
                self.step();
            }
        }
    }
}
