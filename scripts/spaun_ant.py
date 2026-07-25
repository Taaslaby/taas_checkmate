#!/usr/bin/env python3
"""🧠 256-Neuron Spaun-HH → MuJoCo Ant, обучение T2 SPIGRAD (REINFORCE)

Архитектура:
  Sensory[27] → SemanticPointer (SP, dim=64) → Reservoir[221] (HH spiking)
       ↓ (connectome: sensory→reservoir→motor, structured веса)
  HH heterogeneous dynamics (g_leak_i, C_m_i, g_max_i все разные — нет синхронизации)
       ↓ (motor spikes buffer over N_STEPS=40)
  SpikeGradBridge (spike_dim=8 → n_osc=8 → 3*8=24 drives)
       ↓ T3 forward → PST features [pst_dim]
  Linear decoder → action mean [8]
       ↓ + noise → Ant step → reward
  REINFORCE: advantage → grad_pst → Bridge.backward → update W_drive
                                              ↓
                                        Decoder: W_decode += lr * PST * (a-μ)/σ² * advantage

Отличия от ant_spigrad.py (LIF random ER 100K):
  ✅ Маленький мозг (256 нейронов вместо 100K) — быстрее!
  ✅ Spaun-структурированный коннектом (sensory→reservoir→motor)
  ✅ HH heterogeneous — разброс параметров, нет синхронизации!
  ✅ Semantic pointer encoding — информативный вход вместо raw current
"""

import sys, os, time, argparse
import numpy as np
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import gymnasium as gym
import taas_checkmate as tcm

# ═══ CONFIG ═══════════════════════════════════════════════
N = 256                       # Маленький мозг — быстро!
DT_MS = 0.5
SP_DIM = 64                   # Semantic pointer dimensionality
N_SENSORY = 27
N_MOTOR = 8
N_RESERVOIR = N - N_SENSORY - N_MOTOR  # 221
N_OSC = 8
N_STEPS = 40                  # HH integration window
SPIKE_DIM = 8

# HH heterogeneous ranges (равномерный разброс ±20% от номинала)
G_LEAK_MEAN = 0.3
C_M_MEAN = 1.0
G_MAX_NA_MEAN = 120.0
G_MAX_K_MEAN = 36.0
HETERO_VAR = 0.2              # ±20% разброс
TONIC_HH = 20.0               # HH background tonic

# T2 SPIGRAD
DRIVE_GAIN = 10.0
ALPHA = 1.0
FREQ_MIN = 0.5
FREQ_MAX = 3.0
W12, W13, W23 = 0.3, 0.15, 0.08

# Training
WARMUP_STEPS = 50
SIGMA_INIT = 0.3
SIGMA_FINAL = 0.05
LR_BRIDGE = 0.001
LR_DECODE = 0.01
LR_BASELINE = 0.01
EPISODES = 5
ANT_STEPS = 200


# ═══ BUILD CONNECTOME ══════════════════════════════════

def build_spaun_connectome(n, sensory_idx, motor_idx, rng=None):
    """Build a connectome with Spaun-like structure.

    Connectivity:
      - Sensory → Reservoir: random (each sensory → 3 random reservoir)
      - Reservoir → Reservoir: sparse random (avg_deg=8)
      - Reservoir → Motor: random (each 3 random reservoir → 1 motor)
      - Motor → Motor: self-inhibition (diagonal -0.5)
      - Motor → Reservoir: random feedback (each motor → 2 reservoir)

    Returns:
      row_ptr, col_indices, values, nt_types, groups
    """
    if rng is None:
        rng = np.random.RandomState(42)

    res_idx = [i for i in range(n) if i not in sensory_idx and i not in motor_idx]
    all_idx = list(range(n))

    sources, targets = [], []

    # 1. Sensory → Reservoir: each sensory → 3 random reservoir neurons
    for s in sensory_idx:
        for _ in range(3):
            t = int(rng.choice(res_idx))
            sources.append(s); targets.append(t)

    # 2. Reservoir → Reservoir: avg_deg=8 per reservoir neuron
    for s in res_idx:
        deg = rng.randint(4, 12)
        for _ in range(deg):
            t = int(rng.choice(res_idx))
            if t != s:
                sources.append(s); targets.append(t)

    # 3. Reservoir → Motor: each 3 random reservoir → 1 motor
    for m in motor_idx:
        for _ in range(3):
            s = int(rng.choice(res_idx))
            sources.append(s); targets.append(m)

    # 4. Motor → Motor: self-inhibition
    for m in motor_idx:
        sources.append(m); targets.append(m)

    # 5. Motor → Reservoir: feedback, each motor → 2 reservoir
    for m in motor_idx:
        for _ in range(2):
            t = int(rng.choice(res_idx))
            sources.append(m); targets.append(t)

    sources = np.array(sources, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    # Build CSR
    order = np.argsort(sources, kind='stable')
    sources = sources[order]
    targets = targets[order]

    # Remove duplicates
    dup = (np.diff(sources, prepend=-1) == 0) & (np.diff(targets, prepend=-1) == 0)
    sources = sources[~dup]
    targets = targets[~dup]

    counts = np.bincount(sources, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts).astype(np.int32)

    # Weights: small random + structured excitation/inhibition
    r = rng.randn(len(targets))
    values = np.clip(r * 0.3, -1.0, 1.0).astype(np.float32)

    # Inhibitory self-connections for motor neurons
    for i, (s, t) in enumerate(zip(sources, targets)):
        if s == t and s in motor_idx:
            values[i] = -0.5

    # NT types: +1 for excitatory, -1 for inhibitory
    nt_types = np.where(rng.rand(n) < 0.2, -1.0, 1.0).astype(np.float32)
    # Motor neurons are inhibitory
    for m in motor_idx:
        nt_types[m] = -1.0
    # Sensory neurons are excitatory
    for s in sensory_idx:
        nt_types[s] = 1.0

    groups = [
        ('sensory', list(sensory_idx)),
        ('motor', list(motor_idx)),
        ('reservoir', res_idx),
    ]

    print(f"  🔗 {len(targets)} synapses for N={n}")
    print(f"  👁️  Sensory: {len(sensory_idx)}  🧠 Reservoir: {len(res_idx)}  🦾 Motor: {len(motor_idx)}")

    conn = tcm.ConnectomeData(row_ptr, targets, values, nt_types, groups)
    return conn


# ═══ SEMANTIC POINTER ENCODING ═══════════════════════

class SemanticEncoder:
    """Spaun-style semantic pointer encoding.

    Maps sensory observations (Ant obs_dim=27) → SP (dim=64)
    using random projection, then normalizes.
    """
    def __init__(self, input_dim, sp_dim, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        scale = (2.0 / (input_dim + sp_dim)) ** 0.5
        self.W = (rng.randn(sp_dim, input_dim) * scale).astype(np.float32)
        self.sp_dim = sp_dim

    def encode(self, x):
        """Encode input vector → semantic pointer (normalized)."""
        sp = self.W @ x
        norm = np.linalg.norm(sp) + 1e-10
        return sp / norm


# ═══ HETEROGENEOUS HH SETUP ═════════════════════════

def setup_heterogeneous_hh(sim, n, rng=None):
    """Set heterogeneous HH parameters for each neuron.

    Each neuron gets unique g_leak, C_m, g_max_Na, g_max_K.
    Values are uniform random in [mean * (1 - var), mean * (1 + var)].
    """
    if rng is None:
        rng = np.random.RandomState(42)

    def uniform_range(mean, var, n):
        half = mean * var
        return rng.uniform(mean - half, mean + half, n).astype(np.float32)

    # Per-neuron leak conductance and reversal
    g_leak = uniform_range(G_LEAK_MEAN, HETERO_VAR, n)
    E_leak = np.full(n, -65.0, dtype=np.float32)

    # Per-neuron membrane capacitance
    C_m = uniform_range(C_M_MEAN, HETERO_VAR, n)

    # Per-channel g_max: column-major [ch0_n0..ch0_nN, ch1_n0..ch1_nN, ...]
    # ch0 = NaV (120 mS/cm²), ch1 = KV (36 mS/cm²)
    g_max_na = uniform_range(G_MAX_NA_MEAN, HETERO_VAR, n)
    g_max_k = uniform_range(G_MAX_K_MEAN, HETERO_VAR, n)
    g_max_arr = np.concatenate([g_max_na, g_max_k]).astype(np.float32)

    sim.set_hh_params(
        g_max_arr.astype(np.float32),
        np.float32([50.0, -77.0]),  # E_rev: Na=+50, K=-77
        g_leak.astype(np.float32),
        E_leak.astype(np.float32),
    )
    sim.set_C_m(C_m.astype(np.float32))

    print(f"  🔧 Heterogeneous HH: g_leak∈[{g_leak.min():.3f}, {g_leak.max():.3f}]")
    print(f"                    C_m∈[{C_m.min():.3f}, {C_m.max():.3f}]")
    print(f"                    g_Na∈[{g_max_na.min():.1f}, {g_max_na.max():.1f}]")
    print(f"                    g_K∈[{g_max_k.min():.1f}, {g_max_k.max():.1f}]")

    return g_leak, C_m


# ═══ SPAUN-ANT PIPELINE ═════════════════════════════

class SpaunAnt:
    """Full pipeline: SP encoding → HH reservoir → T2 SPIGRAD → REINFORCE."""

    def __init__(self, conn, sim, encoder, n_neurons, sensory_idx, motor_idx, res_idx):
        self.conn = conn
        self.sim = sim
        self.encoder = encoder
        self.n = n_neurons
        self.sensory_idx = sensory_idx
        self.motor_idx = motor_idx
        self.res_idx = res_idx

        # SP → sensory projection (random, модулирует sensory нейроны)
        self.W_sp_sensory = np.random.randn(N_SENSORY, SP_DIM).astype(np.float32) * 0.5

    def encode_observation(self, obs):
        """Encode Ant observation → SP → sensory neuron currents."""
        sp = self.encoder.encode(obs)
        # Project SP to sensory neuron currents
        I_sensory = self.W_sp_sensory @ sp * 50.0  # scale to HH
        return I_sensory

    def run_reservoir(self, obs, n_steps=N_STEPS):
        """Run HH reservoir for N_STEPS with SP-encoded sensory input.

        Returns:
            motor_buffer: [N_STEPS, N_MOTOR] spike buffer for T3 bridge
        """
        I_sensory = self.encode_observation(obs)

        ext_I = np.zeros(self.n, dtype=np.float32)
        motor_buffer = np.zeros((n_steps, N_MOTOR), dtype=np.float32)

        for step in range(n_steps):
            ext_I.fill(0.0)
            # Inject SP-encoded sensory input into sensory neurons
            for j, idx in enumerate(self.sensory_idx):
                ext_I[idx] = I_sensory[j]
            # Background tonic — все нейроны получают одинаковый фон
            ext_I += TONIC_HH

            spike_buf = self.sim.step_batch(
                self.conn, 1, dt_ms=DT_MS, external_I=ext_I)
            sb = np.asarray(spike_buf).flatten()
            motor_buffer[step] = sb[self.motor_idx].astype(np.float32)

        return motor_buffer


# ═══ MAIN ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--ant-steps", type=int, default=ANT_STEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr-bridge", type=float, default=LR_BRIDGE)
    parser.add_argument("--lr-decode", type=float, default=LR_DECODE)
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🧠 Spaun-HH (256N) → Ant, T2 SPIGRAD              ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # ── 1. Build Spaun connectome ──
    print("[1/4] Building Spaun connectome ...")
    t0 = time.time()
    sensory_idx = list(range(N_SENSORY))
    motor_idx = list(range(N - N_MOTOR, N))
    res_idx = [i for i in range(N) if i not in sensory_idx and i not in motor_idx]

    conn = build_spaun_connectome(N, sensory_idx, motor_idx, rng)
    print(f"  ⏱️  {time.time()-t0:.1f}s\n")

    # ── 2. HH with heterogeneous params ──
    print("[2/4] Heterogeneous HH setup ...")
    t0 = time.time()
    sim = tcm.BrainSimulator(conn.n_neurons, "hh")
    setup_heterogeneous_hh(sim, N, rng)

    # Encoder
    encoder = SemanticEncoder(N_SENSORY, SP_DIM, rng)

    # Warmup
    print(f"  🔥 Warming up {WARMUP_STEPS} steps ...")
    ext_I = np.full(N, 15.0, dtype=np.float32)
    ext_I[motor_idx] = 10.0
    for w in range(WARMUP_STEPS):
        sb = sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I)
        sb_arr = np.asarray(sb).flatten()
        if w % 10 == 0:
            motor_fire = int(sb_arr[motor_idx].sum())
            total_fire = int(sb_arr.sum())
            print(f"    Step {w:3d}: {total_fire:>4d} spikes (motor={motor_fire})")

    sb_final = np.asarray(sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I)).flatten()
    motor_ok = int(sb_final[motor_idx].sum()) > 0
    print(f"  {'✅' if motor_ok else '⚠️'} Motor neurons {'firing!' if motor_ok else 'silent!'}")
    print(f"  ⏱️  {time.time()-t0:.1f}s\n")

    # ── 3. T2 SPIGRAD setup ──
    print("[3/4] T2 SPIGRAD setup ...")
    pst_dim = tcm.pst_linear_dim(N_OSC)
    bridge = tcm.SpikeGradBridge(
        SPIKE_DIM, N_OSC,
        dt=0.05, drive_gain=DRIVE_GAIN, alpha=ALPHA,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX,
        coupling_w12=W12, coupling_w13=W13, coupling_w23=W23,
        zero_init=False,
    )
    W_decode = np.random.randn(pst_dim, SPIKE_DIM).astype(np.float32) * 0.01
    b_decode = np.zeros(SPIKE_DIM, dtype=np.float32)
    print(f"  Bridge: spike_dim={SPIKE_DIM} -> n_osc={N_OSC} (drive_dim={bridge.drive_dim()})")
    print(f"  Decoder: PST dim={pst_dim} -> {SPIKE_DIM} actions")
    total = bridge.drive_dim() * SPIKE_DIM + SPIKE_DIM + pst_dim * SPIKE_DIM + SPIKE_DIM
    print(f"  Total params: {total:,}")
    print(f"  ⏱️  {time.time()-t0:.1f}s\n")

    # ── 4. Ant episodes with T2 SPIGRAD ──
    print(f"[4/4] Training {args.episodes} episodes with T2 SPIGRAD (REINFORCE)...\n")
    env = gym.make("Ant-v4", render_mode="human" if args.render else None)

    pipeline = SpaunAnt(conn, sim, encoder, N, sensory_idx, motor_idx, res_idx)
    episode_rewards = []
    reward_baseline = 0.0
    global_step = 0
    action_buf = np.zeros(N_MOTOR, dtype=np.float32)

    for ep in range(args.episodes):
        sigma = max(SIGMA_FINAL, SIGMA_INIT * (1 - ep / max(args.episodes - 1, 1)))

        try:
            obs, info = env.reset(seed=42 + ep)
        except TypeError:
            obs = env.reset()
        obs = np.asarray(obs[:N_SENSORY], dtype=np.float32)

        ep_reward = 0.0
        ep_actions = 0

        while ep_actions < args.ant_steps:
            # Run HH reservoir with SP-encoded sensory input
            motor_buffer = pipeline.run_reservoir(obs, n_steps=N_STEPS)

            # T3 forward: motor spikes -> PST features
            pst_mat, traj = bridge.forward(motor_buffer, N_STEPS)
            pst_feat = pst_mat[-1]

            # Decode PST -> action
            action_mean = pst_feat @ W_decode + b_decode
            action = action_mean + np.random.randn(SPIKE_DIM).astype(np.float32) * sigma
            action = np.clip(action, -1.0, 1.0)
            action_buf = action_buf * 0.7 + action * 0.3

            # Ant step
            obs, reward, terminated, truncated, _ = env.step(action_buf)
            obs = np.asarray(obs[:N_SENSORY], dtype=np.float32)
            ep_reward += float(reward)
            ep_actions += 1
            global_step += 1

            # REINFORCE update
            baseline_update = float(reward) - reward_baseline
            reward_baseline += LR_BASELINE * baseline_update
            advantage = baseline_update

            d_mu = advantage * (action - action_mean) / (sigma**2 + 1e-8)
            d_mu = np.clip(d_mu, -10.0, 10.0)

            # Decoder ASCENT
            W_decode += LR_DECODE * np.outer(pst_feat, d_mu).astype(np.float32)
            b_decode += LR_DECODE * d_mu.astype(np.float32)

            # Bridge DESCENT
            grad_pst = -(W_decode.astype(np.float64) @ d_mu.astype(np.float64))
            grad_pst = grad_pst.astype(np.float32)

            grad_w, grad_b = bridge.backward(grad_pst, traj, motor_buffer)
            grad_w = np.clip(grad_w, -5.0, 5.0)
            grad_b = np.clip(grad_b, -5.0, 5.0)
            bridge.update(grad_w, grad_b, args.lr_bridge)

            if ep_actions % 20 == 0:
                print(f"  ep={ep+1} act={ep_actions:>3d} reward={reward:>+6.2f} sigma={sigma:.2f}")

            if terminated or truncated:
                print(f"  🏁 Episode {ep+1}: ant fell after {ep_actions} steps")
                break

        episode_rewards.append(ep_reward)
        print(f"  {'─'*45}")
        print(f"  EP {ep+1}: reward={ep_reward:>+7.1f}  steps={ep_actions:>3d}")
        print(f"  {'─'*45}")

    env.close()

    print(f"\n{'='*50}")
    print(f"  Training complete! {args.episodes} episodes")
    for i, r in enumerate(episode_rewards):
        print(f"    Ep {i+1}: {r:>+8.1f}")
    print(f"\n  Spaun-HH (256N) + T2 SPIGRAD + Ant - обучение пошло!")


if __name__ == "__main__":
    main()
