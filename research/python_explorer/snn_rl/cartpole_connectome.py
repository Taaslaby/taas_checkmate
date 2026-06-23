#!/usr/bin/env python3
"""
CartPole × 138K connectome: SNN reservoir + REINFORCE readout.

Architecture:
  CartPole state (4 floats) → Poisson → 4 input neurons
  → 138K Izhikevich connectome (CSR, fixed weights)
  → 2×32 output neurons → spike count → logits → action
  → REINFORCE on readout weights
"""

import sys, time, argparse
import numpy as np

import gymnasium as gym
from taas_checkmate import step_izhikevich, propagate_spikes

DT = 0.5
N_STEPS_PER_CTRL = 80
N_NEURONS = 138_000
AVG_SYNS = 38
N_OUTPUT = 32
SEED = 42

LR_BIAS = 0.01
GAMMA = 0.99
INPUT_SCALE = 20.0


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


def pad8(x):
    return (x + 7) // 8 * 8


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


class CartPoleConnectome:
    def __init__(self, n_neurons=N_NEURONS, avg_syn=AVG_SYNS):
        self.conn = gen_connectome(n_neurons, avg_syn, SEED)
        self.n_real = n_neurons
        self.state = setup(n_neurons)
        self.n = self.state["n"]

        self.input_idxs = np.array([0, 1, 2, 3])
        self.out_idxs_a = np.arange(1000, 1000 + N_OUTPUT)
        self.out_idxs_b = np.arange(2000, 2000 + N_OUTPUT)

        self.bias_readout = np.zeros(2, dtype=np.float32)
        self.episode_buf = []  # (logit_a, logit_b, action, reward)

    def reset(self):
        n = self.n
        self.state["v"][:] = -65.0
        self.state["u"][:] = -13.0
        self.state["trace"][:] = 0.0
        self.state["I_buf"][:] = 0.0
        self.state["spikes"][:] = 0
        self.episode_buf = []

    def encode_obs(self, obs):
        I = np.zeros(self.n, dtype=np.float32)
        for i, val in enumerate(obs):
            I[i] = val * INPUT_SCALE
        self.state["external_I"][:] = I

    def step(self):
        s = self.state
        conn = self.conn

        # 1. Collect spikes from PREVIOUS step
        prev_spiked = np.where(s["spikes"][:self.n_real] != 0)[0]

        # 2. I_buf = external input + propagated spikes from previous step
        s["I_buf"][:self.n] = s["external_I"][:self.n]
        if len(prev_spiked) > 0:
            propagate_spikes(
                conn[0], conn[1], conn[2],
                prev_spiked.astype(np.int32),
                s["nt_types"], s["I_buf"], 1.0,
            )

        # 3. Izhikevich step (uses I_buf = external + propagated)
        step_izhikevich(
            s["v"], s["u"], s["trace"],
            s["a"], s["b"], s["I_buf"],
            DT, 0.99, s["spikes"], s["c"], s["d"],
        )

        # 4. Readout: spike counts from output groups
        sp_a = s["spikes"][self.out_idxs_a].sum()
        sp_b = s["spikes"][self.out_idxs_b].sum()
        return sp_a, sp_b

    def run_episode(self, render=False):
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        obs, _ = env.reset()
        self.reset()

        total_reward = 0
        t0 = time.perf_counter()

        for t in range(500):
            self.encode_obs(obs)
            spike_a, spike_b = 0.0, 0.0
            for _ in range(N_STEPS_PER_CTRL):
                a, b = self.step()
                spike_a += a
                spike_b += b

            logit_a = spike_a + self.bias_readout[0]
            logit_b = spike_b + self.bias_readout[1]
            logits = np.array([logit_a, logit_b])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)

            action = 0 if np.random.random() < probs[0] else 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.episode_buf.append((logit_a, logit_b, action, reward))
            total_reward += reward

            if done:
                break

        wall = time.perf_counter() - t0
        env.close()
        return total_reward, t + 1, wall

    def learn(self):
        G = 0.0
        returns = np.zeros(len(self.episode_buf), dtype=np.float32)
        for i in reversed(range(len(self.episode_buf))):
            _, _, _, r = self.episode_buf[i]
            G = r + GAMMA * G
            returns[i] = G

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (loga, logb, act, _), Gt in zip(self.episode_buf, returns):
            log_probs = np.array([loga, logb])
            log_probs -= log_probs.max()
            probs = np.exp(log_probs) / (np.exp(log_probs).sum() + 1e-30)

            # REINFORCE: grad = ∇log π(a|s) = ∇(logit_a) - Σ∇(logit)·p
            # Since logit_a = spike_a + bias_a, ∇bias = 1 for chosen action, 0 otherwise
            self.bias_readout[act] += LR_BIAS * Gt

        return float(returns.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("CartPole × 138K Izhikevich Connectome")
    print("=" * 60)

    print(f"\nGenerating connectome: {N_NEURONS:,} neurons, ~{AVG_SYNS} avg syn/neur...")
    t0 = time.perf_counter()
    agent = CartPoleConnectome(N_NEURONS, AVG_SYNS)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    best = 0.0
    for ep in range(args.episodes):
        r, steps, wall = agent.run_episode(render=args.render and ep % 10 == 0)
        advantage = agent.learn()

        best = max(best, r)
        fps = steps / max(wall, 1e-6)
        bar = "█" * int(r / 5) + "░" * (100 - int(r / 5))
        print(f"  ep {ep+1:3d} | reward {r:5.1f} | steps {steps:3d} | "
              f"adv {advantage:+.3f} | best {best:5.1f} | {fps:5.1f} fps")

    print(f"\nBest reward: {best:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
