#!/usr/bin/env python3
"""
CartPole × Connectome → T3 hybrid readout.

Architecture:
  138K Izhikevich connectome (random Erdos-Renyi)
  → spike activity → T3 oscillator drives
  → PST features → NLMS decoder → REINFORCE

The connectome provides chaotic high-dimensional dynamics.
T3 reads out structured oscillatory features from spike patterns.
"""

import sys, time, argparse
import numpy as np
import gymnasium as gym
from taas_checkmate import (
    step_izhikevich, propagate_spikes,
    PyTriOscillator, PyNlmsDecoder,
    pst_linear_dim, pst_linear_features,
)

DT = 0.5
N_STEPS_PER_CTRL = 80
N_NEURONS = 138_000
AVG_SYNS = 38
N_OUTPUT = 32
SEED = 42
INPUT_SCALE = 20.0

# T3 params
N_OSC = 32
N_T3_STEPS = 5    # T3 steps per connectome step (faster than oscillator)
FREQ_MIN = 0.3
FREQ_MAX = 2.5
PST_EMA_ALPHA = 0.3


def pad8(x):
    return (x + 7) // 8 * 8


def gen_connectome(n, avg_syn, seed):
    rng = np.random.default_rng(seed)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    cols, vals = [], []
    for i in range(n):
        deg = int(np.ceil(avg_syn * (0.5 + rng.random() * 0.5)))
        row_ptr[i + 1] = row_ptr[i] + deg
        cols.extend(rng.integers(0, n, deg).tolist())
        vals.extend((0.5 + rng.random(deg) * 2.0).tolist())
    return (row_ptr,
            np.array(cols, dtype=np.int32),
            np.array(vals, dtype=np.float32))


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


class CartPoleConnectomeT3:
    def __init__(self):
        n = N_NEURONS
        self.conn = gen_connectome(n, AVG_SYNS, SEED)
        self.state = setup(n)
        self.n = self.state["n"]
        self.n_real = n

        # Input neurons (first 4) and output groups
        self.input_idxs = np.array([0, 1, 2, 3])
        self.out_idxs_a = np.arange(1000, 1000 + N_OUTPUT)
        self.out_idxs_b = np.arange(2000, 2000 + N_OUTPUT)

        # T3 readout
        self.tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
        self.tri.set_coupling(0.3, 0.15, 0.08)

        rng_t3 = np.random.default_rng(SEED + 1)
        # Project connectome population activity → T3 drives (3 pops × N_OSC)
        # Use first 3*N_OSC neurons' spike activity as features
        self.t3_input_idxs = np.arange(3 * N_OSC)

        N_PST = pst_linear_dim(N_OSC)
        self.fv = np.zeros(N_PST, dtype=np.float32)
        self.fv_ema = np.zeros(N_PST, dtype=np.float32)
        self.dec = [PyNlmsDecoder(N_PST, 0.05, 1e-8),
                    PyNlmsDecoder(N_PST, 0.05, 1e-8)]
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
        for i, val in enumerate(obs):
            I[i] = val * INPUT_SCALE
        self.state["external_I"][:] = I

    def step_connectome(self):
        """One connectome step. Returns spike counts for T3 input neurons."""
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

        # Activity in T3 input neurons (first 96)
        t3_act = s["spikes"][self.t3_input_idxs].copy().astype(np.float32)
        # Output group spike counts
        sp_a = s["spikes"][self.out_idxs_a].sum()
        sp_b = s["spikes"][self.out_idxs_b].sum()
        return t3_act, sp_a, sp_b

    def run_episode(self, render=False):
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        obs, _ = env.reset()
        self.reset()

        total = 0
        for t in range(500):
            self.encode_obs(obs)

            # Run connectome + T3 for N_STEPS_PER_CTRL
            for _ in range(N_STEPS_PER_CTRL):
                t3_act, sp_a, sp_b = self.step_connectome()

                # Feed spike activity into T3 as drive (all 3 pops at once)
                self.tri.set_drive(t3_act)

                for _ in range(N_T3_STEPS):
                    self.tri.step(DT)

            # Read PST features
            pst_linear_features(self.tri, self.fv)
            self.fv_ema *= (1 - PST_EMA_ALPHA)
            self.fv_ema += PST_EMA_ALPHA * self.fv

            # Decode
            p0 = self.dec[0].predict(self.fv_ema)
            p1 = self.dec[1].predict(self.fv_ema)
            logits = np.array([p0, p1])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)

            action = 0 if np.random.random() < probs[0] else 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.ep_buf.append((self.fv_ema.copy(), action, reward))
            total += reward
            if done:
                break

        env.close()
        return total, t + 1

    def learn(self):
        n = len(self.ep_buf)
        if n < 2:
            return 0.0

        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        for i in reversed(range(n)):
            G = self.ep_buf[i][2] + 0.99 * G
            returns[i] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (fv, act, _), Gt in zip(self.ep_buf, returns):
            p0 = self.dec[0].predict(fv)
            p1 = self.dec[1].predict(fv)
            logits = np.array([p0, p1])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)
            grad = -probs.copy()
            grad[act] += 1.0
            self.dec[0].train(fv, p0 + Gt * grad[0] * 0.5)
            self.dec[1].train(fv, p1 + Gt * grad[1] * 0.5)
        return float(returns.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("═" * 60)
    print("CartPole × Connectome → T3 readout")
    print(f"  {N_NEURONS:,} Izhikevich + {N_OSC} T3 oscillators")
    print(f"  {N_STEPS_PER_CTRL} conn steps + {N_T3_STEPS} T3 steps per ctrl")
    print("═" * 60)

    print("\nGenerating connectome...", end=" ", flush=True)
    t0 = time.perf_counter()
    agent = CartPoleConnectomeT3()
    print(f"{time.perf_counter()-t0:.1f}s")

    best = 0.0
    window = []
    for ep in range(args.episodes):
        r, steps = agent.run_episode()
        agent.learn()
        best = max(best, r)
        window.append(r)
        if len(window) > 50:
            window.pop(0)
        avg = sum(window) / len(window) if window else 0
        bar = "█" * min(int(r/2), 50) + "░" * max(50-min(int(r/2), 50), 0)
        print(f"  ep {ep+1:3d} |{bar}| {r:5.1f}  best {best:5.1f}  avg50 {avg:5.1f}")

    print(f"\nTotal: {time.perf_counter()-t0:.1f}s  |  Best reward: {best:.1f}")


if __name__ == "__main__":
    main()
