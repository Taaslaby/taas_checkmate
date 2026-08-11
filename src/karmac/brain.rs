//! SpikingBrain — a real spiking controller for arena agents.
//!
//! Built on the canonical pillars LIF kernel (`pillars::api::neurons::lif::step_lif`),
//! this is not a rate-based FSM in disguise:
//!
//! ```text
//!  perception (8 channels)                  regions (2 × 4gr × 16)     actions (5)
//!        │                                       │   ┌──────────┐          ▲
//!        ▼                                       │   │ EVALUATOR │  gate   │
//!  ┌──────────┐  W1 (fixed)   ┌──────────────┐   └──▶│ 2 × 8 LIF │ softmax│
//!  │  SENSOR  │ ────────────▶ │  WORKSPACE   │ ──▶   └──────────┘ ───────┤
//!  │ 8ch × 16 │   block-diag  │ 2 REGIONS    │      GNW arbitrates which │
//!  │ LIF      │               │ 4gr × 16 each│      region wins the motor│
//!  └──────────┘               └──────────────┘      pool (W2, R-STDP)    │
//!        ▲                          │                                    │
//!        │        GNW broadcast     │  gate: softmax over group rates,    │
//!        └──────────────────────────┘  winner group re-excited, losers    │
//!                                      suppressed (ignition / competition)│
//! ```
//!
//! * **Sensory** (128 LIF): 8 perception channels × 16 neurons. Each channel
//!   injects current only into its own 16 neurons.
//! * **Workspace** (128 LIF): the Global Neuronal Workspace, split into **two
//!   regions** of 4 channel-groups × 16. `W1` is fixed and block-diagonal
//!   (workspace group *c* reads only sensory channel *c*), so each group is a
//!   feature detector and each region is a coalition of feature detectors.
//!   Every tick a softmax "gate" over the 8 group firing rates computes a
//!   broadcast bias that re-excites the winning group and suppresses the rest
//!   — genuine GNW ignition/competition dynamics on a spiking substrate.
//! * **Evaluator** (16 LIF): the GNW's arbitration population. Each region's
//!   workspace activity drives its own 8 evaluator neurons; a softmax over the
//!   evaluator firing rates becomes the **motor-access gate**. The dominant
//!   region wins the motor pool, the loser is suppressed — brain regions
//!   literally compete for access to the motors.
//! * **Output** (80 LIF): 5 actions × 16 neurons. `W2` is **learnable** via
//!   three-factor reward-modulated STDP (eligibility trace × dopamine) and is
//!   region-partitioned: the motor input from region *r* is weighted by
//!   `gate[r]`, so only the winning region's synapses actually drive the
//!   readout — and therefore only the winning region's eligibility grows
//!   (win → learn, lose → stay quiet). The readout integrates each action
//!   channel's firing rate with a leaky accumulator plus hysteresis.
//!
//! The prior (`new`) embeds the reflex pipeline as a weak diagonal bias;
//! `train_init` perturbs it (halved diagonal + confusion weights) so the
//! reward-modulated STDP has something genuine to fix across training
//! episodes. All randomness is seeded and deterministic.
//!
//! Learning rule (3-factor R-STDP on `W2`):
//! ```text
//! e[k,j] = λ·e[k,j] + r_w[j]·post_k            eligibility trace
//! d      = reward − baseline                   dopamine
//! W2[k,j] += lr · d · e[k,j]                   weight update (clamped)
//! ```

use std::collections::VecDeque;

use pillars::api::neurons::lif::step_lif;

pub const N_CHANNELS: usize = 8;
pub const N_ACTIONS: usize = 5;
/// Neurons per channel/group. Must be a multiple of 8: the pillars `step_lif`
/// kernel is SIMD over 8 lanes. 16 keeps the whole brain a multiple of 8
/// (sensor 128, workspace 128, output 80, evaluator 32).
const NEURONS_PER_GROUP: usize = 16;
/// Two brain regions compete for the motor pool (see module docs).
pub const N_REGIONS: usize = 2;
/// Workspace channel-groups per region (8 channels / 2 regions).
const GROUPS_PER_REGION: usize = N_CHANNELS / N_REGIONS;
/// Evaluator neurons per region.
const EVAL_PER_REGION: usize = 8;

/// Sensory population size (8 channels × 16).
pub const N_SENSOR: usize = N_CHANNELS * NEURONS_PER_GROUP;
/// Workspace population size (2 regions × 4 groups × 16).
pub const N_WORKSPACE: usize = N_REGIONS * GROUPS_PER_REGION * NEURONS_PER_GROUP;
/// Output population size (5 actions × 16).
pub const N_OUTPUT: usize = N_ACTIONS * NEURONS_PER_GROUP;
/// GNW evaluator population size (2 regions × 8).
pub const N_EVAL: usize = N_REGIONS * EVAL_PER_REGION;

// LIF parameters (mV / tick; arena step = one brain tick).
// Time constants are short so the whole sensor→workspace→output chain
// responds within a handful of ticks (the arena world itself is slow).
const DT: f32 = 1.0;
const TAU_S: f32 = 3.0;
const TAU_W: f32 = 2.0;
const TAU_O: f32 = 2.0;
const TAU_EV: f32 = 2.0;
const V_REST: f32 = -70.0;
const V_THRESH: f32 = -54.0;
/// Workspace threshold: the GNW workspace is the most excitable layer
/// (NMDA-rich pyramidal neurons) — it ignites first, then broadcasts.
/// Equilibrium v = v_rest + I ≈ −56 sits above −64, so the workspace fires
/// densely and sustains the output population.
const V_THRESH_W: f32 = -64.0;
/// Evaluator threshold: slightly above the workspace, so a region must be
/// genuinely active before it can claim motor access.
const V_THRESH_EV: f32 = -60.0;
const V_RESET: f32 = -70.0;
const TAU_REF: f32 = 2.0;

/// Sensory channel input current (per channel).
const I_SENS: f32 = 50.0;
/// Workspace exemplar gains ∈ [4.0, 6.0].
const W1_MIN: f32 = 4.0;
const W1_MAX: f32 = 6.0;
/// Synaptic low-pass on the workspace input (real synapses smooth the
/// pre-synaptic pulse train into a sustained post-synaptic current).
const SYNAPSE_ALPHA: f32 = 0.3;
/// GNW broadcast gain.
const GNW_BETA: f32 = 0.45;
/// Evaluator gain: maps region activity to evaluator drive. With rest −70
/// and thresh −60, a region activity ≈ 0.25 → I ≈ 10 → v ≈ −60 sits just at
/// threshold; higher activity ignites the evaluator and wins the gate.
const EVAL_GAIN: f32 = 40.0;
/// Evaluator softmax temperature: lower = sharper winner-take-all.
/// Deliberately soft: the losing region must keep a real voice at the
/// motors, or R-STDP credit assignment starves it — its weights get
/// renormalized down every step without ever growing, and the brain
/// permanently locks onto one region.
const GNW_EVAL_TEMP: f32 = 1.2;
/// Rate leak (EMA alpha): rate = (1−α)·rate + α·spikes.
const RATE_ALPHA: f32 = 0.4;
/// Readout hysteresis: only switch actions if the challenger beats the
/// incumbent's rate by this margin (or the incumbent died out).
const HYSTERESIS: f32 = 1.15;
const RATE_DEAD: f32 = 0.02;

// R-STDP hyper-parameters. LR is deliberately modest: the arena applies a
// shaped reward EVERY step, so a large LR lets weights over-accumulate by
// the end of a long training run and the brain starts locking onto one
// action (the last episodes regress).
const LR: f32 = 0.25;
const BASELINE_ALPHA: f32 = 0.02;
const ELIG_DECAY: f32 = 0.92;
/// Dopamine is clamped so single reward events cannot push weights across
/// the firing threshold in one step (keeps learning stable).
const DOPAMINE_CLAMP: f32 = 1.0;
/// Prior diagonal strength for the reflex pipeline.
/// (Output equilibrium v = v_rest + I must sit above threshold −54.
/// With 16 neurons per group the supported current is roughly
/// I ≈ 16 neurons × rate 0.3 × diag × 2 supported groups / 8 ≈ 8.0·diag,
/// so diag 6.0 → I ≈ 48 → v ≈ −22, well above threshold.)
const PRIOR_DIAG: f32 = 6.0;
/// Prior off-diagonal noise.
const PRIOR_NOISE: f32 = 0.04;
/// Training perturbation: diagonal scale, confusion weight, noise.
/// The confusion weights are strong enough to actually fire (equilibrium
/// v = v_rest + I must clear −54: weight ≥ 16 / (16 neurons × rate 0.35
/// × 2 supported groups / 8) ≈ 2.9), so the untrained brain genuinely takes
/// wrong actions; R-STDP prunes them with the shaped reward.
const TRAIN_DIAG: f32 = 0.75;
const TRAIN_CONFUSION: f32 = 1.1;
const TRAIN_NOISE: f32 = 0.35;
const WEIGHT_CLAMP: f32 = 10.0;

/// A mission-specific reflex prior: which workspace channel-groups each
/// action reads, plus which (action, wrong-group) confusions `train_init`
/// injects so reward-modulated STDP has genuine errors to prune.
///
/// The channel-groups naturally partition into the two regions: groups 0–3
/// live in region 0, groups 4–7 in region 1. Actions whose support lies in
/// region 0 ("approach / acquire") compete against region-1 actions
/// ("withhold / execute") through the evaluator gate.
#[derive(Clone)]
pub struct BrainPrior {
    pub action_support: [[usize; 2]; N_ACTIONS],
    pub confusion: [(usize, usize); 4],
}

impl BrainPrior {
    /// Two-Key Vault semantics (default).
    /// Action order: 0 Explore, 1 Share, 2 Retrieve, 3 Carry, 4 Wait.
    /// Region 0 = acquire pipeline (Explore/Share/Retrieve, channels 0–3);
    /// region 1 = execution (Carry/Wait, channels 4–7).
    pub fn vault() -> Self {
        Self {
            action_support: [
                [0, 0], // Explore  ← c0 key_unknown
                [1, 1], // Share    ← c1 share_pending
                [2, 2], // Retrieve ← c2 at_own_key
                [5, 7], // Carry    ← c5 go_vault, c7 at_door
                [4, 6], // Wait     ← c4 await_intel, c6 stunned
            ],
            confusion: [
                (1, 0), // Share driven by key_unknown (shares nothing)
                (0, 1), // Explore driven by share_pending (keeps wandering)
                (3, 4), // Carry driven by await_intel (runs off early)
                (4, 5), // Wait driven by go_vault (freezes at the vault)
            ],
        }
    }

    /// Hide & Seek — hider semantics. Channel c2 carries normalized stamina,
    /// preserving the shelter/explore bias while teaching the brain when a
    /// movement burst is affordable. Every other action fires on a real
    /// threat/shelter signal. Vision is line-of-sight:
    /// c0 threat_visible only when the seeker is actually seen, c7 intel_fresh
    /// when a TEAMMATE was spotted — so the whole team scatters, not just the
    /// one seen.
    /// Action order: 0 Explore, 1 Hide, 2 Push, 3 Flee, 4 Wait.
    pub fn hide_seek_hider() -> Self {
        Self {
            action_support: [
                [2, 2], // Explore ← c2 stamina (baseline wander budget)
                [1, 1], // Hide    ← c1 threat_close (duck out of sight)
                [3, 4], // Push    ← c3 box_adjacent, c4 doorway_adjacent
                [0, 7], // Flee    ← c0 threat_visible, c7 intel_fresh (scatter)
                [5, 6], // Wait    ← c5 barricade_useful, c6 teammate_visible
            ],
            confusion: [
                (0, 1), // Explore driven by threat_close (walks into danger)
                (3, 1), // Push driven by threat_close (pushes in a panic)
                (1, 4), // Hide driven by doorway_adjacent (freezes at the gap)
                (4, 2), // Wait driven by stamina (rests instead of wandering)
            ],
        }
    }

    /// Hide & Seek — seeker semantics. Channel c2 carries normalized stamina
    /// for patrol/explore budgeting. Action order: 0 Explore, 1 Hide
    /// (patrol shelters), 2 Push, 3 Flee (chase), 4 Wait.
    /// Vision is line-of-sight: c0 hider_visible only on a real sighting,
    /// c7 intel_fresh is the pack's fading trail — any teammate recently saw
    /// a hider, so the whole pack chases the last spotted position even after
    /// the hider breaks line of sight. c6 intel_memory persists for the whole
    /// round after the first sighting and drives the pack's sweep of the
    /// hideouts near the last spotted area.
    pub fn hide_seek_seeker() -> Self {
        Self {
            action_support: [
                [2, 2], // Explore ← c2 stamina (baseline roam budget)
                [6, 6], // Hide    ← c6 intel_memory (pack sweep of hideouts)
                [3, 4], // Push    ← c3 box_adjacent, c4 doorway_adjacent
                [0, 7], // Flee    ← c0 hider_visible, c7 intel_fresh (chase trail)
                [5, 5], // Wait    ← c5 barricade_blocking
            ],
            confusion: [
                (1, 3), // Patrol driven by box_adjacent (paws at boxes)
                (3, 5), // Chase driven by barricade_blocking (stares at wall)
                (2, 0), // Push driven by hider_visible (drops the box)
                (4, 1), // Wait driven by hider_close (freezes on the trail)
            ],
        }
    }
}

/// Deterministic xorshift64* source of small perturbations. `pub(crate)` so
/// arena modules (e.g. hide_seek) can share one implementation.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, n).
    pub(crate) fn below(&mut self, n: u64) -> usize {
        (self.next() % n.max(1)) as usize
    }

    /// Uniform in [-1, 1].
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Spiking brain for one arena agent. All populations are fixed-size arrays
/// (no per-step allocation); `W2` is the only learnable structure.
#[derive(Clone)]
pub struct SpikingBrain {
    // ── LIF state (SoA) ─────────────────────────────────────────────
    v_s: [f32; N_SENSOR],
    refr_s: [f32; N_SENSOR],
    v_w: [f32; N_WORKSPACE],
    refr_w: [f32; N_WORKSPACE],
    v_o: [f32; N_OUTPUT],
    refr_o: [f32; N_OUTPUT],
    v_ev: [f32; N_EVAL],
    refr_ev: [f32; N_EVAL],
    i_s: [f32; N_SENSOR],
    i_w: [f32; N_WORKSPACE],
    i_o: [f32; N_OUTPUT],
    i_ev: [f32; N_EVAL],
    sp_s: [i32; N_SENSOR],
    sp_w: [i32; N_WORKSPACE],
    sp_o: [i32; N_OUTPUT],
    sp_ev: [i32; N_EVAL],

    // ── Fixed LIF parameters (pre-built, never re-allocated) ────────
    tau_s: [f32; N_SENSOR],
    rest_s: [f32; N_SENSOR],
    thresh_s: [f32; N_SENSOR],
    reset_s: [f32; N_SENSOR],
    tau_w: [f32; N_WORKSPACE],
    rest_w: [f32; N_WORKSPACE],
    thresh_w: [f32; N_WORKSPACE],
    reset_w: [f32; N_WORKSPACE],
    tau_o: [f32; N_OUTPUT],
    rest_o: [f32; N_OUTPUT],
    thresh_o: [f32; N_OUTPUT],
    reset_o: [f32; N_OUTPUT],
    tau_ev: [f32; N_EVAL],
    rest_ev: [f32; N_EVAL],
    thresh_ev: [f32; N_EVAL],
    reset_ev: [f32; N_EVAL],

    // ── Firing rates (leaky EMA) ────────────────────────────────────
    rate_s: [f32; N_SENSOR],
    rate_w: [f32; N_WORKSPACE],
    rate_o: [f32; N_OUTPUT],
    rate_ev: [f32; N_EVAL],
    /// Synaptic low-pass filter on the workspace input current.
    syn_w: [f32; N_WORKSPACE],

    // ── Mission prior ───────────────────────────────────────────────
    /// Which channels each action reads + the confusions to inject.
    prior: BrainPrior,

    // ── Weights ─────────────────────────────────────────────────────
    /// [W × 10] block-diagonal sensor → workspace, fixed after construction.
    w1: [f32; N_WORKSPACE * NEURONS_PER_GROUP],
    /// [A × W] workspace → output, learnable via R-STDP. Region-partitioned:
    /// columns of region *r* are gated by `region_gate[r]` at the output.
    w2: [f32; N_OUTPUT * N_WORKSPACE],
    /// [A × W] eligibility traces.
    elig: [f32; N_OUTPUT * N_WORKSPACE],
    /// Per-action L2 norms of `w2` (the total drive budget per action).
    /// R-STDP redistributes weight within each action's fixed budget, which
    /// keeps the output population stable (no threshold-runaway).
    norm_target: [f32; N_ACTIONS],

    // ── GNW ──────────────────────────────────────────────────────────
    /// Per-workspace-neuron broadcast bias (accumulated from the gate).
    gnw_bias: [f32; N_WORKSPACE],
    /// Motor-access gate: softmax over evaluator firing rates, one value per
    /// region. The dominant region wins the motor pool.
    region_gate: [f32; N_REGIONS],

    // ── Readout / learning state ─────────────────────────────────────
    prev_action: usize,
    action_rates: [f32; N_ACTIONS],
    last_reward: f32,
    dopamine: f32,
    baseline: f32,

    /// Recent per-population spike totals for the GUI (sensor, workspace, output).
    spike_log: VecDeque<[f32; 3]>,
}

impl SpikingBrain {
    /// Build a brain with the default (Two-Key Vault) reflex prior.
    pub fn new() -> Self {
        Self::with_prior(BrainPrior::vault())
    }

    /// Build a brain with a mission-specific reflex prior.
    pub fn with_prior(prior: BrainPrior) -> Self {
        Self::build_prior(prior)
    }

    /// Build a brain with the prior perturbed for training: the diagonal is
    /// halved and confusion weights are added, so reward-modulated STDP has
    /// genuine slack to recover. Deterministic per seed.
    pub fn train_init(seed: u64) -> Self {
        Self::train_init_with(seed, BrainPrior::vault())
    }

    /// Training-perturbed brain with a mission-specific prior.
    pub fn train_init_with(seed: u64, prior: BrainPrior) -> Self {
        let mut brain = Self::build_prior(prior);
        brain.perturb(seed);
        brain.update_norm_target();
        brain
    }

    fn build_prior(prior: BrainPrior) -> Self {
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);

        // W1: workspace group c reads sensory channel c with exemplar gains.
        let mut w1 = [0.0f32; N_WORKSPACE * NEURONS_PER_GROUP];
        for group in 0..N_CHANNELS {
            for e in 0..NEURONS_PER_GROUP {
                let row = (group * NEURONS_PER_GROUP + e) * NEURONS_PER_GROUP;
                for j in 0..NEURONS_PER_GROUP {
                    let gain = W1_MIN + (W1_MAX - W1_MIN) * (rng.unit() * 0.5 + 0.5);
                    w1[row + j] = gain;
                }
            }
        }

        // W2: each action's 10 output neurons read its supporting groups.
        let mut w2 = [0.0f32; N_OUTPUT * N_WORKSPACE];
        for action in 0..N_ACTIONS {
            let [g0, g1] = prior.action_support[action];
            for k in 0..NEURONS_PER_GROUP {
                let out = action * NEURONS_PER_GROUP + k;
                for group in 0..N_CHANNELS {
                    for j in 0..NEURONS_PER_GROUP {
                        let supported = group == g0 || group == g1;
                        let weight = if supported {
                            PRIOR_DIAG + PRIOR_NOISE * rng.unit()
                        } else {
                            PRIOR_NOISE * rng.unit()
                        };
                        w2[out * N_WORKSPACE + group * NEURONS_PER_GROUP + j] = weight;
                    }
                }
            }
        }

        let rest = [V_REST; N_SENSOR];
        let thresh = [V_THRESH; N_SENSOR];
        let reset = [V_RESET; N_SENSOR];
        let rest_w = [V_REST; N_WORKSPACE];
        let thresh_w = [V_THRESH_W; N_WORKSPACE];
        let reset_w = [V_RESET; N_WORKSPACE];
        let rest_o = [V_REST; N_OUTPUT];
        let thresh_o = [V_THRESH; N_OUTPUT];
        let reset_o = [V_RESET; N_OUTPUT];
        let rest_ev = [V_REST; N_EVAL];
        let thresh_ev = [V_THRESH_EV; N_EVAL];
        let reset_ev = [V_RESET; N_EVAL];

        let mut brain = Self {
            v_s: [V_REST; N_SENSOR],
            refr_s: [0.0; N_SENSOR],
            v_w: [V_REST; N_WORKSPACE],
            refr_w: [0.0; N_WORKSPACE],
            v_o: [V_REST; N_OUTPUT],
            refr_o: [0.0; N_OUTPUT],
            v_ev: [V_REST; N_EVAL],
            refr_ev: [0.0; N_EVAL],
            i_s: [0.0; N_SENSOR],
            i_w: [0.0; N_WORKSPACE],
            i_o: [0.0; N_OUTPUT],
            i_ev: [0.0; N_EVAL],
            sp_s: [0; N_SENSOR],
            sp_w: [0; N_WORKSPACE],
            sp_o: [0; N_OUTPUT],
            sp_ev: [0; N_EVAL],
            tau_s: [TAU_S; N_SENSOR],
            rest_s: rest,
            thresh_s: thresh,
            reset_s: reset,
            tau_w: [TAU_W; N_WORKSPACE],
            rest_w,
            thresh_w,
            reset_w,
            tau_o: [TAU_O; N_OUTPUT],
            rest_o,
            thresh_o,
            reset_o,
            tau_ev: [TAU_EV; N_EVAL],
            rest_ev,
            thresh_ev,
            reset_ev,
            rate_s: [0.0; N_SENSOR],
            rate_w: [0.0; N_WORKSPACE],
            rate_o: [0.0; N_OUTPUT],
            rate_ev: [0.0; N_EVAL],
            syn_w: [0.0; N_WORKSPACE],
            prior,
            w1,
            w2,
            elig: [0.0; N_OUTPUT * N_WORKSPACE],
            norm_target: [0.0; N_ACTIONS],
            gnw_bias: [0.0; N_WORKSPACE],
            region_gate: [0.5; N_REGIONS],
            prev_action: 4,
            action_rates: [0.0; N_ACTIONS],
            last_reward: 0.0,
            dopamine: 0.0,
            baseline: 0.0,
            spike_log: VecDeque::with_capacity(64),
        };
        brain.update_norm_target();
        brain
    }

    /// Recompute the per-action L2 weight norms after construction or
    /// perturbation (each brain normalizes around its own initial scale).
    fn update_norm_target(&mut self) {
        for action in 0..N_ACTIONS {
            let mut sum = 0.0f32;
            for k in 0..NEURONS_PER_GROUP {
                let out = action * NEURONS_PER_GROUP + k;
                for j in 0..N_WORKSPACE {
                    let w = self.w2[out * N_WORKSPACE + j];
                    sum += w * w;
                }
            }
            self.norm_target[action] = sum.sqrt().max(1e-6);
        }
    }

    /// Perturb the prior for training episodes: halve the diagonal and add
    /// confusion weights on the most dangerous cross-connections.
    fn perturb(&mut self, seed: u64) {
        let mut rng = Rng::new(seed.wrapping_mul(0x5851_F42D_4C95_7F2D));
        for action in 0..N_ACTIONS {
            let [g0, g1] = self.prior.action_support[action];
            for k in 0..NEURONS_PER_GROUP {
                let out = action * NEURONS_PER_GROUP + k;
                for group in 0..N_CHANNELS {
                    for j in 0..NEURONS_PER_GROUP {
                        let idx = out * N_WORKSPACE + group * NEURONS_PER_GROUP + j;
                        let supported = group == g0 || group == g1;
                        let confused = self
                            .prior
                            .confusion
                            .iter()
                            .any(|&(a, g)| a == action && g == group);
                        let scale = if supported {
                            TRAIN_DIAG
                        } else if confused {
                            TRAIN_CONFUSION
                        } else {
                            0.0
                        };
                        self.w2[idx] = (self.w2[idx] * scale + TRAIN_NOISE * rng.unit())
                            .clamp(-WEIGHT_CLAMP, WEIGHT_CLAMP);
                    }
                }
            }
        }
    }

    /// One brain tick. `channels` is the 8-element perception vector; returns
    /// the selected action index (0 Explore, 1 Share, 2 Retrieve, 3 Carry, 4 Wait).
    pub fn step(&mut self, channels: &[f32; N_CHANNELS]) -> usize {
        // ── Sensory: inject channel currents ─────────────────────────
        self.i_s.fill(0.0);
        for c in 0..N_CHANNELS {
            let current = channels[c] * I_SENS;
            for e in 0..NEURONS_PER_GROUP {
                self.i_s[c * NEURONS_PER_GROUP + e] = current;
            }
        }
        step_lif(
            &mut self.v_s,
            &mut self.i_s,
            &mut self.sp_s,
            DT,
            &self.tau_s,
            &self.rest_s,
            &self.thresh_s,
            &self.reset_s,
            &mut self.refr_s,
            TAU_REF,
        );
        for (r, &s) in self.rate_s.iter_mut().zip(self.sp_s.iter()) {
            *r = (1.0 - RATE_ALPHA) * *r + RATE_ALPHA * s as f32;
        }

        // ── Workspace: W1 input + GNW broadcast bias, low-passed ─────
        self.i_w.fill(0.0);
        for j in 0..N_WORKSPACE {
            let group = j / NEURONS_PER_GROUP;
            let mut cur = self.gnw_bias[j];
            let row = j * NEURONS_PER_GROUP;
            for e in 0..NEURONS_PER_GROUP {
                cur += self.w1[row + e] * self.rate_s[group * NEURONS_PER_GROUP + e];
            }
            // Synaptic filtering: the pulse train becomes a sustained
            // post-synaptic current, so the workspace integrates reliably.
            self.syn_w[j] += SYNAPSE_ALPHA * (cur - self.syn_w[j]);
            self.i_w[j] = self.syn_w[j];
        }
        step_lif(
            &mut self.v_w,
            &mut self.i_w,
            &mut self.sp_w,
            DT,
            &self.tau_w,
            &self.rest_w,
            &self.thresh_w,
            &self.reset_w,
            &mut self.refr_w,
            TAU_REF,
        );
        for (r, &s) in self.rate_w.iter_mut().zip(self.sp_w.iter()) {
            *r = (1.0 - RATE_ALPHA) * *r + RATE_ALPHA * s as f32;
        }

        // ── GNW gate: softmax over group rates → broadcast bias ──────
        let mut group_rate = [0.0f32; N_CHANNELS];
        for group in 0..N_CHANNELS {
            group_rate[group] = self.rate_w
                [group * NEURONS_PER_GROUP..(group + 1) * NEURONS_PER_GROUP]
                .iter()
                .sum::<f32>()
                / NEURONS_PER_GROUP as f32;
        }
        let max_g = group_rate.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut soft = [0.0f32; N_CHANNELS];
        let mut sum = 0.0f32;
        if max_g.is_finite() {
            // Empty workspace: no gate, no bias. (Without this guard,
            // exp(−(−inf)) = inf → NaN poisons the workspace.)
            for (s, &g) in soft.iter_mut().zip(group_rate.iter()) {
                *s = (g - max_g).exp();
                sum += *s;
            }
        }
        let inv = if sum > 1e-9 { 1.0 / sum } else { 0.0 };
        let mean_g = group_rate.iter().sum::<f32>() / N_CHANNELS as f32;
        for group in 0..N_CHANNELS {
            // Winning group re-excited, losers suppressed (global broadcast).
            let bias = GNW_BETA * (group_rate[group] - mean_g) + GNW_BETA * soft[group] * inv;
            for e in 0..NEURONS_PER_GROUP {
                self.gnw_bias[group * NEURONS_PER_GROUP + e] = bias;
            }
        }

        // ── GNW evaluator: regions compete for motor access ──────────
        // Each region's workspace activity drives its own evaluator neurons;
        // a softmax over evaluator firing becomes the motor-access gate.
        // A region's voice is its TOTAL excitation (sum over its groups, not
        // the mean): one strongly active group must be enough to ignite the
        // evaluator and claim motor access.
        let mut region_activity = [0.0f32; N_REGIONS];
        for region in 0..N_REGIONS {
            let start = region * GROUPS_PER_REGION;
            region_activity[region] = group_rate[start..start + GROUPS_PER_REGION]
                .iter()
                .sum::<f32>();
        }
        self.i_ev.fill(0.0);
        for region in 0..N_REGIONS {
            let current = EVAL_GAIN * region_activity[region];
            for e in 0..EVAL_PER_REGION {
                self.i_ev[region * EVAL_PER_REGION + e] = current;
            }
        }
        step_lif(
            &mut self.v_ev,
            &mut self.i_ev,
            &mut self.sp_ev,
            DT,
            &self.tau_ev,
            &self.rest_ev,
            &self.thresh_ev,
            &self.reset_ev,
            &mut self.refr_ev,
            TAU_REF,
        );
        for (r, &s) in self.rate_ev.iter_mut().zip(self.sp_ev.iter()) {
            *r = (1.0 - RATE_ALPHA) * *r + RATE_ALPHA * s as f32;
        }
        let mut ev_region = [0.0f32; N_REGIONS];
        for region in 0..N_REGIONS {
            ev_region[region] = self.rate_ev
                [region * EVAL_PER_REGION..(region + 1) * EVAL_PER_REGION]
                .iter()
                .sum::<f32>()
                / EVAL_PER_REGION as f32;
        }
        let max_ev = ev_region.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut gate = [0.0f32; N_REGIONS];
        let mut gate_sum = 0.0f32;
        if max_ev.is_finite() {
            for (g, &e) in gate.iter_mut().zip(ev_region.iter()) {
                *g = ((e - max_ev) / GNW_EVAL_TEMP).exp();
                gate_sum += *g;
            }
        }
        let inv_g = if gate_sum > 1e-9 { 1.0 / gate_sum } else { 0.0 };
        for g in gate.iter_mut() {
            *g *= inv_g;
        }
        self.region_gate = gate;

        // ── Output: gated W2 input + readout ─────────────────────────
        // The gate REDISTRIBUTES drive rather than damping it: at the tie
        // (0.5/0.5) each region keeps its full share, while a decisive winner
        // is amplified and the loser is weakened but never silenced — the
        // losing region keeps enough motor voice to stay in the R-STDP
        // credit-assignment loop.
        self.i_o.fill(0.0);
        for out in 0..N_OUTPUT {
            let row = out * N_WORKSPACE;
            let mut cur = 0.0f32;
            for region in 0..N_REGIONS {
                let start = region * GROUPS_PER_REGION * NEURONS_PER_GROUP;
                let end = start + GROUPS_PER_REGION * NEURONS_PER_GROUP;
                let mut part = 0.0f32;
                for j in start..end {
                    part += self.w2[row + j] * self.rate_w[j];
                }
                let gain = (gate[region] * N_REGIONS as f32).clamp(0.2, 1.8);
                cur += gain * part;
            }
            self.i_o[out] = cur;
        }
        step_lif(
            &mut self.v_o,
            &mut self.i_o,
            &mut self.sp_o,
            DT,
            &self.tau_o,
            &self.rest_o,
            &self.thresh_o,
            &self.reset_o,
            &mut self.refr_o,
            TAU_REF,
        );
        for (r, &s) in self.rate_o.iter_mut().zip(self.sp_o.iter()) {
            *r = (1.0 - RATE_ALPHA) * *r + RATE_ALPHA * s as f32;
        }

        // Action-channel rates = mean over each action's 10 neurons.
        for action in 0..N_ACTIONS {
            self.action_rates[action] = self.rate_o
                [action * NEURONS_PER_GROUP..(action + 1) * NEURONS_PER_GROUP]
                .iter()
                .sum::<f32>()
                / NEURONS_PER_GROUP as f32;
        }
        let winner = self
            .action_rates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(4);
        // Hysteresis: prefer the incumbent unless clearly out-voted.
        let prev_rate = self.action_rates[self.prev_action];
        let winner_rate = self.action_rates[winner];
        let action = if winner != self.prev_action && winner_rate < prev_rate * HYSTERESIS {
            self.prev_action
        } else if winner_rate < RATE_DEAD {
            4 // no evidence → Wait
        } else {
            winner
        };
        self.prev_action = action;

        // ── Eligibility traces (pre-activity × post-spike) ───────────
        // Only the region that won the gate fires its output neurons, so its
        // eligibility grows and it learns; the suppressed region stays quiet.
        for action in 0..N_ACTIONS {
            let post = self.sp_o[action * NEURONS_PER_GROUP..(action + 1) * NEURONS_PER_GROUP]
                .iter()
                .any(|&s| s > 0);
            let post_f = post as u8 as f32;
            for j in 0..N_WORKSPACE {
                self.elig[action * N_WORKSPACE + j] =
                    ELIG_DECAY * self.elig[action * N_WORKSPACE + j] + self.rate_w[j] * post_f;
            }
        }

        self.spike_log.push_back([
            self.sp_s.iter().sum::<i32>() as f32,
            self.sp_w.iter().sum::<i32>() as f32,
            self.sp_o.iter().sum::<i32>() as f32,
        ]);
        if self.spike_log.len() > 64 {
            self.spike_log.pop_front();
        }

        action
    }

    /// Three-factor reward-modulated STDP: dopamine = reward − baseline,
    /// `W2 += lr · dopamine · eligibility`.
    pub fn apply_reward(&mut self, reward: f32) {
        self.last_reward = reward;
        self.dopamine = (reward - self.baseline).clamp(-DOPAMINE_CLAMP, DOPAMINE_CLAMP);
        self.baseline += BASELINE_ALPHA * (reward - self.baseline);
        let delta = LR * self.dopamine;
        for k in 0..N_OUTPUT {
            // Eligibility traces live per ACTION channel (shared by the 10
            // output neurons of that action) — index by action, not neuron.
            let action = k / NEURONS_PER_GROUP;
            for j in 0..N_WORKSPACE {
                let idx = k * N_WORKSPACE + j;
                self.w2[idx] = (self.w2[idx] + delta * self.elig[action * N_WORKSPACE + j])
                    .clamp(-WEIGHT_CLAMP, WEIGHT_CLAMP);
            }
        }
        // Renormalize each action's weights to its fixed L2 budget: the total
        // drive per action stays bounded, so R-STDP redistributes weight
        // (diag grows, confusion shrinks) without threshold-runaway.
        for action in 0..N_ACTIONS {
            let mut sum = 0.0f32;
            for k in 0..NEURONS_PER_GROUP {
                let out = action * NEURONS_PER_GROUP + k;
                for j in 0..N_WORKSPACE {
                    let w = self.w2[out * N_WORKSPACE + j];
                    sum += w * w;
                }
            }
            let scale = self.norm_target[action] / sum.sqrt().max(1e-6);
            for k in 0..NEURONS_PER_GROUP {
                let out = action * NEURONS_PER_GROUP + k;
                for j in 0..N_WORKSPACE {
                    let idx = out * N_WORKSPACE + j;
                    self.w2[idx] *= scale;
                }
            }
        }
    }

    /// Reset neural dynamics (voltages, rates, eligibility, bias) — keeps weights.
    pub fn reset_dynamics(&mut self) {
        self.v_s = [V_REST; N_SENSOR];
        self.refr_s = [0.0; N_SENSOR];
        self.v_w = [V_REST; N_WORKSPACE];
        self.refr_w = [0.0; N_WORKSPACE];
        self.v_o = [V_REST; N_OUTPUT];
        self.refr_o = [0.0; N_OUTPUT];
        self.v_ev = [V_REST; N_EVAL];
        self.refr_ev = [0.0; N_EVAL];
        self.rate_s = [0.0; N_SENSOR];
        self.rate_w = [0.0; N_WORKSPACE];
        self.rate_o = [0.0; N_OUTPUT];
        self.rate_ev = [0.0; N_EVAL];
        self.syn_w = [0.0; N_WORKSPACE];
        self.elig = [0.0; N_OUTPUT * N_WORKSPACE];
        self.gnw_bias = [0.0; N_WORKSPACE];
        self.region_gate = [0.5; N_REGIONS];
        self.prev_action = 4;
        self.action_rates = [0.0; N_ACTIONS];
        self.dopamine = 0.0;
        self.baseline = 0.0;
        self.last_reward = 0.0;
        self.spike_log.clear();
    }

    /// Average `W2` over each action's 16 output neurons → [5 × 128] heatmap.
    pub fn weight_heat(&self) -> Vec<f32> {
        let mut heat = Vec::with_capacity(N_ACTIONS * N_WORKSPACE);
        for action in 0..N_ACTIONS {
            for j in 0..N_WORKSPACE {
                let sum = (0..NEURONS_PER_GROUP)
                    .map(|k| self.w2[(action * NEURONS_PER_GROUP + k) * N_WORKSPACE + j])
                    .sum::<f32>();
                heat.push(sum / NEURONS_PER_GROUP as f32);
            }
        }
        heat
    }

    pub fn action_rates(&self) -> &[f32] {
        &self.action_rates
    }

    /// Motor-access gate per region: which region currently wins the motors.
    pub fn region_gate(&self) -> &[f32] {
        &self.region_gate
    }

    /// Mean firing rate per sensory channel (8 values).
    pub fn sensor_channel_rates(&self) -> Vec<f32> {
        (0..N_CHANNELS)
            .map(|c| {
                self.rate_s[c * NEURONS_PER_GROUP..(c + 1) * NEURONS_PER_GROUP]
                    .iter()
                    .sum::<f32>()
                    / NEURONS_PER_GROUP as f32
            })
            .collect()
    }

    /// Mean firing rate per workspace group (8 values).
    pub fn workspace_group_rates(&self) -> Vec<f32> {
        (0..N_CHANNELS)
            .map(|c| {
                self.rate_w[c * NEURONS_PER_GROUP..(c + 1) * NEURONS_PER_GROUP]
                    .iter()
                    .sum::<f32>()
                    / NEURONS_PER_GROUP as f32
            })
            .collect()
    }

    /// Recent per-population spike totals: (sensor, workspace, output) per step.
    pub fn spike_log(&self) -> &VecDeque<[f32; 3]> {
        &self.spike_log
    }

    /// Last step spike totals (sensor, workspace, output).
    pub fn last_spikes(&self) -> [f32; 3] {
        self.spike_log.back().copied().unwrap_or([0.0; 3])
    }

    pub fn dopamine(&self) -> f32 {
        self.dopamine
    }
    pub fn last_reward(&self) -> f32 {
        self.last_reward
    }
    pub fn baseline(&self) -> f32 {
        self.baseline
    }
    pub fn weights(&self) -> &[f32] {
        &self.w2
    }
}

impl Default for SpikingBrain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brain_selects_explore_when_key_unknown() {
        let mut brain = SpikingBrain::new();
        let mut channels = [0.0f32; N_CHANNELS];
        channels[0] = 1.0;
        // The sensor→workspace→output chain ignites within ~10 ticks.
        let mut actions = [0usize; 20];
        for a in actions.iter_mut() {
            *a = brain.step(&channels);
        }
        let majority = actions.iter().filter(|&&a| a == 0).count();
        assert!(
            majority >= actions.len() / 2,
            "expected Explore(0) to dominate, got {actions:?}"
        );
        assert!(brain.action_rates().iter().all(|r| r.is_finite()));
        assert!(brain.region_gate().iter().all(|g| g.is_finite()));
        assert!((brain.region_gate().iter().sum::<f32>() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn brain_selects_carry_when_ready() {
        let mut brain = SpikingBrain::new();
        let mut channels = [0.0f32; N_CHANNELS];
        channels[5] = 1.0; // go_vault
        channels[7] = 1.0; // at_door
        let mut actions = [0usize; 20];
        for a in actions.iter_mut() {
            *a = brain.step(&channels);
        }
        let majority = actions.iter().filter(|&&a| a == 3).count();
        assert!(
            majority >= actions.len() / 2,
            "expected Carry(3) to dominate, got {actions:?}"
        );
    }

    #[test]
    fn brain_selects_wait_when_stunned() {
        let mut brain = SpikingBrain::new();
        let mut channels = [0.0f32; N_CHANNELS];
        channels[6] = 1.0; // stunned
        let mut actions = [0usize; 20];
        for a in actions.iter_mut() {
            *a = brain.step(&channels);
        }
        assert!(
            actions.iter().all(|&a| a == 4),
            "expected Wait(4) while stunned, got {actions:?}"
        );
    }

    #[test]
    fn active_region_wins_motor_gate() {
        // Region 1 (channels 4–7) active → gate shifts toward region 1,
        // proving the GNW evaluator arbitrates motor access.
        let mut brain = SpikingBrain::new();
        let mut channels = [0.0f32; N_CHANNELS];
        channels[5] = 1.0; // go_vault → region 1
        for _ in 0..15 {
            brain.step(&channels);
        }
        assert!(
            brain.region_gate()[1] > brain.region_gate()[0],
            "region 1 must win the motor gate: {:?}",
            brain.region_gate()
        );
    }

    fn mean_supported(w: &[f32], action: usize, group: usize) -> f32 {
        (0..NEURONS_PER_GROUP)
            .map(|k| w[(action * NEURONS_PER_GROUP + k) * N_WORKSPACE + group * NEURONS_PER_GROUP])
            .sum::<f32>()
            / NEURONS_PER_GROUP as f32
    }

    #[test]
    fn reward_strengthens_supported_weights() {
        let before = SpikingBrain::train_init(1);
        let mut brain = SpikingBrain::train_init(1);
        let mut channels = [0.0f32; N_CHANNELS];
        channels[5] = 1.0; // go_vault
        for _ in 0..12 {
            brain.step(&channels);
        }
        brain.apply_reward(2.0); // door-open scale reward
        let m_before = mean_supported(before.weights(), 3, 5);
        let m_after = mean_supported(brain.weights(), 3, 5);
        assert!(
            m_after > m_before,
            "supported weight must grow: {m_before:.3} -> {m_after:.3}"
        );
    }

    #[test]
    fn negative_reward_weakens_wrong_weights() {
        let before = SpikingBrain::train_init(1);
        let mut brain = SpikingBrain::train_init(1);
        let mut channels = [0.0f32; N_CHANNELS];
        channels[4] = 1.0; // await_intel
        for _ in 0..12 {
            brain.step(&channels);
        }
        brain.apply_reward(-0.6); // decoy hit scale
        let m_before = mean_supported(before.weights(), 4, 4);
        let m_after = mean_supported(brain.weights(), 4, 4);
        assert!(
            m_after < m_before,
            "negative dopamine must weaken active weights: {m_before:.3} -> {m_after:.3}"
        );
    }

    #[test]
    fn reset_dynamics_keeps_weights() {
        let mut brain = SpikingBrain::new();
        let mut channels = [0.0f32; N_CHANNELS];
        channels[0] = 1.0;
        for _ in 0..10 {
            brain.step(&channels);
        }
        brain.apply_reward(1.0);
        let w_before = brain.weights().to_vec();
        brain.reset_dynamics();
        assert_eq!(brain.weights(), w_before.as_slice());
        assert!(brain.action_rates().iter().all(|&r| r == 0.0));
        assert_eq!(brain.last_spikes(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn train_init_differs_but_valid() {
        let a = SpikingBrain::train_init(1);
        let b = SpikingBrain::train_init(2);
        assert_ne!(a.weights(), b.weights());
        assert!(a
            .weights()
            .iter()
            .all(|w| w.is_finite() && w.abs() <= WEIGHT_CLAMP + 1e-3));
    }
}
