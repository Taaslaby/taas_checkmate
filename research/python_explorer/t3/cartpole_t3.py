#!/usr/bin/env python3
"""
CartPole × T3 (TriOscillator + PST + NLMS + REINFORCE).

Tuned: 128 osc, entropy bonus, LR decay.
"""

import sys, time, argparse
import numpy as np
import gymnasium as gym
from taas_checkmate import (
    PyTriOscillator, PyNlmsDecoder,
    pst_linear_dim, pst_linear_features,
)

N_OSC = 128
N_PST = pst_linear_dim(N_OSC)
N_STEPS_PER_CTRL = 10
DT = 0.05
FREQ_MIN, FREQ_MAX = 0.3, 2.5
SEED = 42
LR_BASE = 0.1
GAMMA = 0.99
PST_EMA_ALPHA = 0.3
ENTROPY_COEF = 0.01


class CartPoleT3:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.proj = np.array([
            rng.normal(0, 0.5, 4)
            for _ in range(N_OSC)
        ], dtype=np.float32)

        self.tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
        self.tri.set_coupling(0.3, 0.15, 0.08)

        self.fv = np.zeros(N_PST, dtype=np.float32)
        self.fv_ema = np.zeros(N_PST, dtype=np.float32)
        self.ep_buf = []

        self.lr = LR_BASE

    def reset(self):
        self.tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
        self.tri.set_coupling(0.3, 0.15, 0.08)
        self.ep_buf = []

    def act(self, obs):
        drives = (self.proj @ obs).astype(np.float32)
        for _ in range(N_STEPS_PER_CTRL):
            self.tri.set_drive(drives)
            self.tri.step(DT)

        pst_linear_features(self.tri, self.fv)
        self.fv_ema *= (1 - PST_EMA_ALPHA)
        self.fv_ema += PST_EMA_ALPHA * self.fv

        logit_0 = self.dec[0].predict(self.fv_ema)
        logit_1 = self.dec[1].predict(self.fv_ema)

        logits = np.array([logit_0, logit_1])
        logits -= logits.max()
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)

        return logits, probs

    def run_episode(self, render=False):
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        obs, _ = env.reset()
        self.reset()

        total = 0
        for t in range(500):
            logits, probs = self.act(obs)
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
            G = self.ep_buf[i][2] + GAMMA * G
            returns[i] = G

        lr = self.lr
        for (fv, act, _), Gt in zip(self.ep_buf, returns):
            p0 = self.dec[0].predict(fv)
            p1 = self.dec[1].predict(fv)
            logits = np.array([p0, p1])
            logits -= logits.max()
            probs = np.exp(logits) / (np.exp(logits).sum() + 1e-30)

            # Policy gradient
            grad = -probs.copy()
            grad[act] += 1.0

            # Add entropy bonus gradient
            H = -(probs * np.log(probs + 1e-30)).sum()
            grad -= ENTROPY_COEF * probs * (np.log(probs + 1e-30) + H)

            t0 = p0 + Gt * grad[0] * lr
            t1 = p1 + Gt * grad[1] * lr
            self.dec[0].train(fv, t0)
            self.dec[1].train(fv, t1)

        # LR decay
        self.lr = max(0.005, self.lr * 0.995)
        return float(G)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"CartPole × T3 — {N_OSC} osc, {N_PST} PST, {N_STEPS_PER_CTRL} steps")
    print("=" * 60)

    agent = CartPoleT3()
    dec = agent.dec = [PyNlmsDecoder(N_PST, LR_BASE, 1e-8),
                       PyNlmsDecoder(N_PST, LR_BASE, 1e-8)]

    best = 0.0
    t0 = time.perf_counter()
    window = []

    for ep in range(args.episodes):
        r, steps = agent.run_episode(render=args.render and ep % 50 == 0)
        adv = agent.learn()
        best = max(best, r)
        window.append(r)
        if len(window) > 50:
            window.pop(0)
        avg = sum(window) / len(window)

        bar = "█" * min(int(r / 2), 50) + "░" * max(50 - min(int(r / 2), 50), 0)
        print(f"  ep {ep+1:4d} |{bar}| {r:5.1f}  best {best:5.1f}  avg50 {avg:5.1f}  lr {agent.lr:.4f}")

        if best >= 500:
            print(f"\n  SOLVED! Best reward 500 at episode {ep+1}")
            break

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.1f}s  |  Best reward: {best:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
