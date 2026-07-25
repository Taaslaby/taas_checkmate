//! Actor-Critic RL agent with dopamine-modulated STDP (3-factor plasticity).
//!
//! ## Architecture
//!
//! ```text
//! State ─→ PoissonEncoder ─→ InputSpikes
//!                               │
//!                    ┌──────────┼──────────┐
//!                    │          │          │
//!               w_ih[i,j]  w_ic[i]    STDP traces
//!                    │          │          │
//!               Izhikevich    LIF      eligibility e
//!               (hidden)   (critic)   de/dt = f(pre,post)
//!                    │          │
//!               w_ho[j,k]    V(s)
//!                    │
//!               LIF (output) ──→ firing rates ──→ softmax ──→ action
//!
//! Learning: TD error δ = r + γ·V(s') - V(s) → dopamine
//!           Δw = η · δ · e  (dopamine-gated by δ)
//! ```
//!
//! Based on Izhikevich (2007) "Solving the distal reward problem through
//! linkage of STDP and dopamine signaling" + Actor-Critic (Sutton & Barto).

#![allow(non_snake_case)]

use crate::{core::neuron, kernels};

const RNG_SCALE_24: f32 = 1.0 / 16777216.0;
const RNG_SCALE_16: f32 = 1.0 / 65536.0;
const PADDED: fn(usize) -> usize = |n| n.div_ceil(8) * 8;

/// RL agent with spiking Actor-Critic + Dopamine-STDP learning.
///
/// Forward pass (`act`):
/// 1. State → Poisson spike rates
/// 2. N_STEPS internal sim steps: Poisson → Izhikevich → LIF → softmax
/// 3. Per-step eligibility accumulation for dopamine-gated learning
///
/// Learning (`learn`):
///   TD error δ = r + γ·V(s') - V(s) → Δw = η · δ · e
pub struct ActorCriticAgent {
    // ─── Dimensions ────────────────────────────────────
    pub n_input: usize,
    pub n_hidden: usize,       // padded to ×8
    pub n_actions_real: usize, // real number of actions (unpadded)
    n_actions: usize,          // padded to ×8
    pub n_steps: usize,

    // ─── Poisson encoder ────────────────────────────────
    encoder: neuron::PoissonGenerator,
    input_spikes_pool: Vec<i32>,

    // ─── Hidden Izhikevich layer ───────────────────────
    v_h: Vec<f32>,
    u_h: Vec<f32>,
    trace_h: Vec<f32>,
    sp_h: Vec<i32>,
    refr_h: Vec<f32>,
    a_h: Vec<f32>,
    b_h: Vec<f32>,
    c_h: Vec<f32>,
    d_h: Vec<f32>,

    // ─── Output LIF readout layer ──────────────────────
    v_o: Vec<f32>,
    sp_o: Vec<i32>,
    trace_o: Vec<f32>,
    refr_o: Vec<f32>,
    act_counts: Vec<f32>,
    act_tau_m: Vec<f32>,
    act_v_rest: Vec<f32>,
    act_v_th: Vec<f32>,
    act_v_reset: Vec<f32>,

    // ─── Critic LIF ────────────────────────────────────
    v_c: Vec<f32>,
    sp_c: Vec<i32>,
    trace_c: Vec<f32>,
    refr_c: Vec<f32>,
    crit_tau_m: Vec<f32>,
    crit_v_rest: Vec<f32>,
    crit_v_th: Vec<f32>,
    crit_v_reset: Vec<f32>,
    critic_count: f32,

    // ─── Weights ───────────────────────────────────────
    pub w_ih: Vec<f32>, // [n_input × n_hidden] column-major
    pub w_ho: Vec<f32>, // [n_hidden × n_actions] column-major
    pub w_ic: Vec<f32>, // [n_input]

    // ─── Eligibility traces ────────────────────────────
    pub elig_ih: Vec<f32>,
    pub elig_ho: Vec<f32>,
    pub elig_ic: Vec<f32>,

    // ─── Pre/post traces for eligibility ────────────────
    pub trace_input: Vec<f32>,
    pub trace_hidden: Vec<f32>,
    pub trace_output: Vec<f32>,
    pub trace_critic_val: Vec<f32>,

    // ─── Pre-allocated current buffer (critic only, 1 neuron) ─
    I_c_buf: Vec<f32>, // [8]

    // ─── Pre-allocated scratch buffers (zero-alloc hot path) ─
    rates_buf: Vec<f32>,     // [PADDED(n_input)] for Poisson rates
    rates_out_buf: Vec<f32>, // [n_actions_real] for softmax
    active_buf: Vec<usize>,  // pre-allocated active list (avoids per-step Vec creation)

    // ─── Last value for TD ─────────────────────────────
    pub last_value: f32,

    // ─── Parameters ────────────────────────────────────
    pub dt: f32,
    pub gamma: f32,
    pub lr_actor: f32,
    pub lr_critic: f32,
    pub tau_trace: f32,
    pub tau_e: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub tau_ref: f32,
}

impl ActorCriticAgent {
    pub fn new(
        n_input: usize,
        n_hidden: usize,
        n_actions: usize,
        n_steps: usize,
        dt: f32,
        seed: u32,
    ) -> Self {
        let nh = PADDED(n_hidden);
        let na = PADDED(n_actions);
        let ni = n_input;
        let dt_s = dt;

        let mut w_ih = vec![0.5; ni * nh];
        let mut w_ho = vec![0.5; nh * na];
        let mut w_ic = vec![0.5; ni];
        let mut rng = LcgRng::new(seed.wrapping_add(42));
        for w in w_ih.iter_mut() {
            *w = 0.3 + rng.next_f32() * 0.4;
        }
        for w in w_ho.iter_mut() {
            *w = 0.3 + rng.next_f32() * 0.4;
        }
        for w in w_ic.iter_mut() {
            *w = 0.3 + rng.next_f32() * 0.4;
        }

        Self {
            n_input: ni,
            n_hidden: nh,
            n_actions_real: n_actions,
            n_actions: na,
            n_steps,
            encoder: neuron::PoissonGenerator::new(ni, dt_s / 1000.0, seed),
            input_spikes_pool: vec![0; PADDED(ni)],
            // Hidden Izhikevich
            v_h: vec![-65.0; nh],
            u_h: vec![-13.0; nh],
            trace_h: vec![0.0; nh],
            sp_h: vec![0; nh],
            refr_h: vec![0.0; nh],
            a_h: vec![0.02; nh],
            b_h: vec![0.2; nh],
            c_h: vec![-65.0; nh],
            d_h: vec![8.0; nh],
            // Output LIF
            v_o: vec![-70.0; na],
            sp_o: vec![0; na],
            trace_o: vec![0.0; na],
            refr_o: vec![0.0; na],
            act_counts: vec![0.0; na],
            act_tau_m: vec![10.0; na],
            act_v_rest: vec![-70.0; na],
            act_v_th: vec![-55.0; na],
            act_v_reset: vec![-75.0; na],
            // Critic LIF
            v_c: vec![-70.0; 8],
            sp_c: vec![0; 8],
            trace_c: vec![0.0; 8],
            refr_c: vec![0.0; 8],
            crit_tau_m: vec![10.0; 8],
            crit_v_rest: vec![-70.0; 8],
            crit_v_th: vec![-55.0; 8],
            crit_v_reset: vec![-75.0; 8],
            critic_count: 0.0,
            // Weights
            w_ih,
            w_ho,
            w_ic,
            // Eligibility
            elig_ih: vec![0.0; ni * nh],
            elig_ho: vec![0.0; nh * na],
            elig_ic: vec![0.0; ni],
            // Traces
            trace_input: vec![0.0; ni],
            trace_hidden: vec![0.0; nh],
            trace_output: vec![0.0; na],
            trace_critic_val: vec![0.0; 8],
            // Pre-allocated scratch (critic + zero-alloc act + fused kernel active list)
            I_c_buf: vec![0.0; 8],
            rates_buf: vec![0.0; PADDED(ni)],
            rates_out_buf: vec![0.0; n_actions],
            active_buf: Vec::with_capacity(ni.min(64)),
            // TD
            last_value: 0.0,
            // Params
            dt: dt_s,
            gamma: 0.95,
            lr_actor: 0.01,
            lr_critic: 0.01,
            tau_trace: 20.0,
            tau_e: 500.0,
            a_ltp: 0.1,
            a_ltd: 0.12,
            w_min: 0.0,
            w_max: 2.0,
            tau_ref: 3.0,
        }
    }

    /// Select action for current state. Returns (action_index, state_value).
    ///
    /// Runs internal simulation with ZERO heap allocations per step
    /// (all buffers pre-allocated in struct).
    pub fn act(&mut self, state: &[f32]) -> (usize, f32) {
        let ni = self.n_input;
        let nh = self.n_hidden;
        let na = self.n_actions;
        let n_steps = self.n_steps;
        let dt = self.dt;
        let tau_ref = self.tau_ref;
        let tau_t = self.tau_trace;
        let tau_e = self.tau_e;
        let a_ltp = self.a_ltp;
        let a_ltd = self.a_ltd;

        // 1. Set Poisson rates from state (reuse buffer, zero alloc)
        let n = ni.min(state.len());
        self.input_spikes_pool.fill(0);
        self.rates_buf.fill(0.0);
        if n > 0 {
            for i in 0..n {
                let val = state[i].clamp(-1.0, 1.0) * 0.5 + 0.5;
                self.rates_buf[i] = 10.0 + val * 190.0;
            }
            self.encoder.set_rates(&self.rates_buf);
        }

        // 2. Reset neuron states
        self.reset_neurons();

        // 3. Internal simulation loop (0 heap allocations)
        for _step in 0..n_steps {
            // a. Poisson → weights → Izhikevich (fused: RNG → SIMD accum → dynamics)
            kernels::poisson_fused_hidden_step(
                &mut self.v_h,
                &mut self.u_h,
                &mut self.trace_h,
                &mut self.sp_h,
                &mut self.refr_h,
                &self.a_h,
                &self.b_h,
                &self.c_h,
                &self.d_h,
                &self.w_ih,
                &mut self.encoder.rng_states,
                &self.encoder.rates,
                &mut self.input_spikes_pool,
                dt,
                0.5,
                tau_ref,
                ni,
                &mut self.active_buf,
            );

            // b. Fused kernel: hidden spikes → LIF (one SIMD pass, no I_o buffer)
            kernels::fused_output_lif_step(
                &mut self.v_o,
                &mut self.sp_o,
                &mut self.refr_o,
                &self.act_tau_m,
                &self.act_v_rest,
                &self.act_v_th,
                &self.act_v_reset,
                &self.w_ho,
                &self.sp_h,
                dt,
                tau_ref,
                nh,
            );

            // d. Critic current (scalar accumulate)
            let mut sum_ic = 0.0f32;
            for i in 0..ni {
                if self.input_spikes_pool[i] != 0 {
                    sum_ic += self.w_ic[i];
                }
            }
            self.I_c_buf[0] = sum_ic * 5.0;

            // Step critic LIF (own params, size 8 — independent of na)
            neuron::step_lif(
                &mut self.v_c,
                &mut self.I_c_buf,
                &mut self.sp_c,
                dt,
                &self.crit_tau_m,
                &self.crit_v_rest,
                &self.crit_v_th,
                &self.crit_v_reset,
                &mut self.refr_c,
                tau_ref,
            );

            // e. Update STDP traces + eligibility
            self.update_eligibility_ih(ni, nh, dt, tau_t, tau_e, a_ltp, a_ltd);
            self.update_eligibility_ho(nh, na, dt, tau_t, tau_e, a_ltp, a_ltd);
            self.update_eligibility_ic(ni, dt, tau_t, tau_e, a_ltp, a_ltd);

            // f. Accumulate action counts + critic
            for k in 0..na {
                if self.sp_o[k] != 0 {
                    self.act_counts[k] += 1.0;
                }
            }
            if self.sp_c[0] != 0 {
                self.critic_count += 1.0;
            }
        }

        // 4. Softmax over firing rates (reuse buffer, zero alloc)
        let n_act = self.n_actions_real;
        let duration_s = (n_steps as f32) * self.dt / 1000.0;
        self.rates_out_buf.fill(0.0);
        let mut max_rate = 0.0f32;
        for k in 0..n_act {
            self.rates_out_buf[k] = if duration_s > 0.0 {
                self.act_counts[k] / duration_s
            } else {
                0.0
            };
            if self.rates_out_buf[k] > max_rate {
                max_rate = self.rates_out_buf[k];
            }
        }

        let temp = 1.0;
        let mut exp_sum = 0.0f32;
        for k in 0..n_act {
            self.rates_out_buf[k] = kernels::fast_exp((self.rates_out_buf[k] - max_rate) / temp);
            exp_sum += self.rates_out_buf[k];
        }
        if exp_sum > 1e-10 {
            for k in 0..n_act {
                self.rates_out_buf[k] /= exp_sum;
            }
        } else {
            for k in 0..n_act {
                self.rates_out_buf[k] = 1.0 / n_act as f32;
            }
        }

        let action = sample_from_softmax(&self.rates_out_buf[..n_act]);
        let value = if duration_s > 0.0 {
            self.critic_count / duration_s / 100.0
        } else {
            0.0
        };
        self.last_value = value.clamp(0.0, 1.0);
        debug_assert!(!self.last_value.is_nan(), "critic value should not be NaN");

        (action, self.last_value)
    }

    /// Learn from reward: Δw = η · δ · e
    pub fn learn(&mut self, reward: f32, next_value: f32, done: bool) -> f32 {
        let td_error =
            reward + self.gamma * next_value * (if done { 0.0 } else { 1.0 }) - self.last_value;

        if td_error.abs() > 1e-10 {
            // Actor: input→hidden  (SIMD dopamine_weight_update)
            kernels::dopamine_weight_update(
                &mut self.w_ih,
                &self.elig_ih,
                td_error,
                self.lr_actor,
                self.w_min,
                self.w_max,
            );

            // Actor: hidden→output (SIMD)
            kernels::dopamine_weight_update(
                &mut self.w_ho,
                &self.elig_ho,
                td_error,
                self.lr_actor,
                self.w_min,
                self.w_max,
            );

            // Critic: input→critic (SIMD, size ni might not be ×8 — has scalar tail)
            kernels::dopamine_weight_update(
                &mut self.w_ic,
                &self.elig_ic,
                td_error,
                self.lr_critic,
                self.w_min,
                self.w_max,
            );
        }

        self.clear_eligibility();
        td_error
    }

    /// Reset for a new episode.
    pub fn reset_episode(&mut self) {
        self.reset_neurons();
        self.clear_eligibility();
        self.last_value = 0.0;
        self.encoder = neuron::PoissonGenerator::new(self.n_input, self.dt / 1000.0, 42);
    }

    // ─── Private helpers ─────────────────────────────────

    fn reset_neurons(&mut self) {
        self.v_h.fill(-65.0);
        self.u_h.fill(-13.0);
        self.trace_h.fill(0.0);
        self.sp_h.fill(0);
        self.refr_h.fill(0.0);
        self.v_o.fill(-70.0);
        self.sp_o.fill(0);
        self.trace_o.fill(0.0);
        self.refr_o.fill(0.0);
        self.act_counts.fill(0.0);
        self.v_c.fill(-70.0);
        self.sp_c.fill(0);
        self.trace_c.fill(0.0);
        self.refr_c.fill(0.0);
        self.critic_count = 0.0;
    }

    fn clear_eligibility(&mut self) {
        self.elig_ih.fill(0.0);
        self.elig_ho.fill(0.0);
        self.elig_ic.fill(0.0);
        self.trace_input.fill(0.0);
        self.trace_hidden.fill(0.0);
        self.trace_output.fill(0.0);
        self.trace_critic_val.fill(0.0);
    }

    fn update_eligibility_ih(
        &mut self,
        ni: usize,
        nh: usize,
        dt: f32,
        tau_t: f32,
        tau_e: f32,
        a_ltp: f32,
        a_ltd: f32,
    ) {
        // NOTE: uses Forward Euler (1 - dt/τ) NOT exp(-dt/τ).
        // At dt/τ=0.1, diff is ~0.5% — intentional speed choice.
        // See crates/core/src/neuron/stdp.rs for details.
        let decay_t = 1.0 - dt / tau_t.max(1.0);
        let decay_e = 1.0 - dt / tau_e.max(1.0);
        for i in 0..ni {
            self.trace_input[i] *= decay_t;
        }
        for j in 0..nh {
            self.trace_hidden[j] *= decay_t;
        }
        for i in 0..ni {
            if self.input_spikes_pool[i] != 0 {
                self.trace_input[i] += 1.0;
            }
        }
        for j in 0..nh {
            if self.sp_h[j] != 0 {
                self.trace_hidden[j] += 1.0;
            }
        }
        for i in 0..ni {
            let zi = self.input_spikes_pool[i];
            let tr_in = self.trace_input[i];
            for j in 0..nh {
                let idx = i * nh + j;
                let ltp = if self.sp_h[j] != 0 {
                    a_ltp * tr_in
                } else {
                    0.0
                };
                let ltd = if zi != 0 {
                    a_ltd * self.trace_hidden[j]
                } else {
                    0.0
                };
                self.elig_ih[idx] = (self.elig_ih[idx] + ltp - ltd) * decay_e;
            }
        }
    }

    fn update_eligibility_ho(
        &mut self,
        nh: usize,
        na: usize,
        dt: f32,
        tau_t: f32,
        tau_e: f32,
        a_ltp: f32,
        a_ltd: f32,
    ) {
        let decay_t = 1.0 - dt / tau_t.max(1.0);
        let decay_e = 1.0 - dt / tau_e.max(1.0);
        for k in 0..na {
            self.trace_output[k] *= decay_t;
        }
        for k in 0..na {
            if self.sp_o[k] != 0 {
                self.trace_output[k] += 1.0;
            }
        }
        for j in 0..nh {
            let zj = self.sp_h[j];
            let tr_h = self.trace_hidden[j];
            for k in 0..na {
                let idx = j * na + k;
                let ltp = if self.sp_o[k] != 0 { a_ltp * tr_h } else { 0.0 };
                let ltd = if zj != 0 {
                    a_ltd * self.trace_output[k]
                } else {
                    0.0
                };
                self.elig_ho[idx] = (self.elig_ho[idx] + ltp - ltd) * decay_e;
            }
        }
    }

    fn update_eligibility_ic(
        &mut self,
        ni: usize,
        dt: f32,
        tau_t: f32,
        tau_e: f32,
        a_ltp: f32,
        a_ltd: f32,
    ) {
        let decay_t = 1.0 - dt / tau_t.max(1.0);
        let decay_e = 1.0 - dt / tau_e.max(1.0);
        self.trace_critic_val[0] *= decay_t;
        if self.sp_c[0] != 0 {
            self.trace_critic_val[0] += 1.0;
        }
        for i in 0..ni {
            let zi = self.input_spikes_pool[i];
            let ltp = if self.sp_c[0] != 0 {
                a_ltp * self.trace_input[i]
            } else {
                0.0
            };
            let ltd = if zi != 0 {
                a_ltd * self.trace_critic_val[0]
            } else {
                0.0
            };
            self.elig_ic[i] += ltp - ltd;
            self.elig_ic[i] *= decay_e;
        }
    }
}

/// Fast LCG for init.
struct LcgRng {
    state: u32,
}
impl LcgRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 * RNG_SCALE_24
    }
}

/// Atomic-counter RNG for softmax sampling.
fn fast_rng_f32() -> f32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static SEED: AtomicU32 = AtomicU32::new(1);
    let mut s = SEED.fetch_add(1103515245, Ordering::Relaxed);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    SEED.store(s, Ordering::Relaxed);
    (s >> 16) as f32 * RNG_SCALE_16
}

fn sample_from_softmax(probs: &[f32]) -> usize {
    let r = fast_rng_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Run a single MDP episode for testing. Returns total reward.
pub fn test_mdp_episode(agent: &mut ActorCriticAgent, max_steps: usize) -> f32 {
    let mut total_reward = 0.0f32;
    let mut state = 0usize;
    for _step in 0..max_steps {
        let mut obs = vec![0.0f32; agent.n_input];
        if state < agent.n_input {
            obs[state] = 1.0;
        }
        let (action, _value) = agent.act(&obs);
        let reward = if (state == 0 && action == 0) || (state == 1 && action == 1) {
            1.0
        } else {
            -0.5
        };
        let next_state = if state == 0 { 1 } else { 0 };
        let mut next_obs = vec![0.0f32; agent.n_input];
        if next_state < agent.n_input {
            next_obs[next_state] = 1.0;
        }
        let (_, next_value) = agent.act(&next_obs);
        let td = agent.learn(reward, next_value, false);
        total_reward += reward;
        state = next_state;
        if td.abs() > 10.0 {
            break;
        }
    }
    total_reward
}
