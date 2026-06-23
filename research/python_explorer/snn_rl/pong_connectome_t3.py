#!/usr/bin/env python3
"""
Atari Pong × Connectome → T3 hybrid readout.
=========================================
Резервуарный компьютинг: 138K Izhikevich → T3 осцилляторы → PST → NLMS → REINFORCE.

Ни одной матричной операции. Ни одной нейросети. Чистая физика.
"""

import argparse, time, sys
import numpy as np
from PIL import Image

try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
except ImportError:
    print("Need: pip install gymnasium[atari] AutoROM && AutoROM --accept-license")
    raise

from taas_checkmate import (
    step_izhikevich, propagate_spikes,
    PyTriOscillator, PyNlmsDecoder,
    pst_linear_dim, pst_linear_features,
)

# ─── Параметры ──────────────────────────────────────────────

INPUT_SCALE = 10.0
RESIZE = 20                 # 20×20 = 400 пикселей → в нейроны
N_INPUT_NEURONS = 400       # по одному нейрону на пиксель
N_NEURONS = 24_000          # уменьшим с 138K до 24K для скорости
AVG_SYNS = 24
N_OUTPUT = 32
SEED = 42

# T3
N_OSC = 32
N_T3_STEPS = 5
DT = 0.5
FREQ_MIN = 0.3
FREQ_MAX = 2.5
PST_EMA_ALPHA = 0.3

# RL
N_STEPS_PER_CTRL = 80
N_ACTIONS = 6               # Pong actions
GAMMA = 0.99


def pad8(x):
    return (x + 7) // 8 * 8


def preprocess(frame):
    """RGB 210×160 → grayscale 20×20 → [0,1] flattened"""
    img = Image.fromarray(frame).convert("L")
    img = img.resize((RESIZE, RESIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).ravel() / 255.0


def gen_connectome(n, avg_syn, seed):
    rng = np.random.default_rng(seed)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    cols, vals = [], []
    for i in range(n):
        deg = max(1, int(np.ceil(avg_syn * (0.5 + rng.random() * 0.5))))
        row_ptr[i + 1] = row_ptr[i] + deg
        target = rng.integers(N_INPUT_NEURONS, n, deg)  # avoid input neurons
        cols.extend(target.tolist())
        vals.extend((0.5 + rng.random(deg) * 2.0).tolist())
    return (row_ptr, np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float32))


def setup(n_real):
    n = pad8(n_real)
    rng = np.random.default_rng(SEED)
    return dict(
        v=np.full(n, -65.0, dtype=np.float32),
        u=np.full(n, -13.0, dtype=np.float32),
        trace=np.zeros(n, dtype=np.float32),
        a=np.full(n, 0.02, dtype=np.float32),
        b=np.full(n, 0.2, dtype=np.float32),
        c=np.full(n, -65.0, dtype=np.float32),
        d=np.full(n, 2.0, dtype=np.float32),
        I_buf=np.zeros(n, dtype=np.float32),
        spikes=np.zeros(n, dtype=np.int32),
        nt_types=np.ones(n, dtype=np.float32),
        external_I=np.zeros(n, dtype=np.float32),
        n=n,
    )


class PongConnectomeT3:
    def __init__(self):
        n = N_NEURONS
        self.conn = gen_connectome(n, AVG_SYNS, SEED)
        self.state = setup(n)
        self.n = self.state["n"]
        self.n_real = n

        # Input neurons (first N_INPUT_NEURONS = pixels)
        self.input_idxs = np.arange(N_INPUT_NEURONS)
        # Output groups (readout neurons)
        self.out_idxs = [
            np.arange(1000 + i * N_OUTPUT, 1000 + (i + 1) * N_OUTPUT)
            for i in range(N_ACTIONS)
        ]

        # T3 readout
        self.tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
        self.tri.set_coupling(0.3, 0.15, 0.08)

        # Project connectome → T3 drives: use first 3*N_OSC neurons' spikes
        self.t3_input_idxs = np.arange(3 * N_OSC)

        N_PST = pst_linear_dim(N_OSC)
        self.fv = np.zeros(N_PST, dtype=np.float32)
        self.fv_ema = np.zeros(N_PST, dtype=np.float32)
        self.dec = [PyNlmsDecoder(N_PST, 0.02, 1e-8) for _ in range(N_ACTIONS)]

        # Experience buffer for REINFORCE
        self.ep_buf = []

    def reset(self):
        self.state["v"][:] = -65.0
        self.state["u"][:] = -13.0
        self.state["trace"][:] = 0.0
        self.state["I_buf"][:] = 0.0
        self.state["spikes"][:] = 0
        self.tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
        self.tri.set_coupling(0.3, 0.15, 0.08)
        self.ep_buf = []

    def encode_obs(self, obs):
        I = np.zeros(self.n, dtype=np.float32)
        I[:N_INPUT_NEURONS] = obs * INPUT_SCALE
        self.state["external_I"][:] = I

    def step_connectome(self):
        s = self.state
        conn = self.conn

        prev_spiked = np.where(s["spikes"][:self.n_real] != 0)[0]
        s["I_buf"][:] = s["external_I"][:]
        if len(prev_spiked) > 0:
            propagate_spikes(
                conn[0], conn[1], conn[2],
                prev_spiked.astype(np.int32),
                s["nt_types"], s["I_buf"], 1.0,
            )
        step_izhikevich(
            s["v"], s["u"], s["trace"],
            s["a"], s["b"], s["I_buf"],
            DT, 0.99, s["spikes"], s["c"], s["d"],
        )

        t3_act = s["spikes"][self.t3_input_idxs].copy().astype(np.float32)
        act_spikes = np.array([
            s["spikes"][idx].sum()
            for idx in self.out_idxs
        ], dtype=np.float32)
        return t3_act, act_spikes

    def run_episode(self):
        env = gym.make(
            "ALE/Pong-v5",
            obs_type="rgb",
            frameskip=4,
            repeat_action_probability=0.0,
        )
        obs, _ = env.reset()
        self.reset()

        total_reward = 0.0

        for t in range(5000):
            state_vec = preprocess(obs)
            self.encode_obs(state_vec)

            # Run connectome + T3 for N_STEPS_PER_CTRL
            for _ in range(N_STEPS_PER_CTRL):
                t3_act, act_spikes = self.step_connectome()
                self.tri.set_drive(t3_act)
                for _ in range(N_T3_STEPS):
                    self.tri.step(DT)

            # PST features
            pst_linear_features(self.tri, self.fv)
            self.fv_ema *= (1 - PST_EMA_ALPHA)
            self.fv_ema += PST_EMA_ALPHA * self.fv

            # NLMS predict → softmax
            logits = np.array([d.predict(self.fv_ema) for d in self.dec])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)

            action = np.random.choice(N_ACTIONS, p=probs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.ep_buf.append((self.fv_ema.copy(), action, reward))
            total_reward += reward
            if done:
                break

        env.close()
        return total_reward

    def learn(self):
        n = len(self.ep_buf)
        if n < 2:
            return 0.0

        # Monte Carlo returns
        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        for i in reversed(range(n)):
            G = self.ep_buf[i][2] + GAMMA * G
            returns[i] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE update
        for (fv, act, _), Gt in zip(self.ep_buf, returns):
            logits = np.array([d.predict(fv) for d in self.dec])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)
            grad = -probs.copy()
            grad[act] += 1.0
            for a in range(N_ACTIONS):
                self.dec[a].train(fv, logits[a] + Gt * grad[a] * 0.1)

        self.ep_buf = []
        return float(returns.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    args = p.parse_args()

    print("\n" + "█" * 60)
    print("███  A T A R I   P O N G   ×   C O N N E C T O M E   →   T 3")
    print("█" * 60)
    print(f"  Connectome:   {N_NEURONS:,} Izhikevich + {AVG_SYNS} avg synapses")
    print(f"  Input:        {N_INPUT_NEURONS} pixels → Poisson → Izhikevich reservoir")
    print(f"  T3:           3 populations × {N_OSC} Kuramoto-ST oscillators")
    print(f"  Decoder:      PST({pst_linear_dim(N_OSC)} dim) → NLMS × {N_ACTIONS} actions → softmax")
    print(f"  Learning:     REINFORCE (Monte Carlo)")
    print(f"  Episodes:     {args.episodes}")
    print("█" * 60)

    print("\nGenerating connectome...", end=" ", flush=True)
    t0 = time.perf_counter()
    agent = PongConnectomeT3()
    print(f"{time.perf_counter() - t0:.1f}s")

    best = -21.0
    window = []

    for ep in range(1, args.episodes + 1):
        r = agent.run_episode()
        td = agent.learn()
        best = max(best, r)

        window.append(r)
        if len(window) > 50:
            window.pop(0)
        avg50 = sum(window) / len(window)

        # Visual bar: Pong reward -21..+21
        bar_raw = int((r + 21) / 42 * 50)
        bar_len = max(0, min(bar_raw, 50))
        bar = "█" * bar_len + "░" * (50 - bar_len)

        print(
            f"  ep {ep:3d} |{bar}| "
            f"reward {r:+5.1f}  "
            f"best {best:+5.1f}  "
            f"avg50 {avg50:+5.1f}  "
            f"steps {len(agent.ep_buf):3d}"
        )

    total = time.perf_counter() - t0
    print("\n" + "═" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Best reward: {best:+5.1f}")
    print(f"  Avg last 50: {avg50:+5.1f}")
    print(f"  Total time:  {total:.1f}s")
    print(f"  Time/ep:     {total / args.episodes * 1000:.0f}ms")
    print("═" * 60)


if __name__ == "__main__":
    main()
