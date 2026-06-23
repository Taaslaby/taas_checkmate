#!/usr/bin/env python3
"""
Atari Pong × Pure Rust SNN Actor-Critic (Dopamine-STDP)
======================================================
Весь forward/backward — внутри Rust SIMD. Python только для preprocess и env.

State (20×20=400) → Poisson → Izhikevich(256) → LIF(6) → softmax
                                            ↓
                                    TD-error → dopamine-STDP
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

from taas_checkmate import PyActorCriticAgent

N_INPUT = 400    # 20×20 grayscale
N_HIDDEN = 256   # Izhikevich (padded to ×8 by Rust)
N_ACTIONS = 6    # Pong
N_STEPS = 30     # internal SNN steps per decision
DT = 1.0         # ms

LR_ACTOR = 0.01
LR_CRITIC = 0.02
GAMMA = 0.99
TAU_TRACE = 20.0
A_LTP = 0.01
A_LTD = 0.01
W_MAX = 5.0


def preprocess(frame):
    """RGB 210×160 → grayscale 20×20 → flatten [0,1]"""
    img = Image.fromarray(frame).convert("L")
    img = img.resize((20, 20), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).ravel() / 255.0


def make_env(render=False):
    return gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0.0,
        render_mode="human" if render else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    agent = PyActorCriticAgent(N_INPUT, N_HIDDEN, N_ACTIONS, N_STEPS, DT, seed=42)
    agent.lr_actor = LR_ACTOR
    agent.lr_critic = LR_CRITIC
    agent.gamma = GAMMA
    agent.tau_trace = TAU_TRACE
    agent.a_ltp = A_LTP
    agent.a_ltd = A_LTD
    agent.w_max = W_MAX
    agent.w_min = -W_MAX

    print("█" * 60)
    print("█  PONG × SPIKING ACTOR-CRITIC (dopamine-STDP)")
    print("█" * 60)
    print(f"  Architecture:  400 → Poisson → Izhikevich(256) → LIF(6)")
    print(f"  Plasticity:    Δw = η·δ·e  (3-factor dopamine-STDP)")
    print(f"  Backend:       Pure Rust SIMD (wide::f32x8)")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Params:        lr_a={LR_ACTOR} lr_c={LR_CRITIC} γ={GAMMA}")
    print("█" * 60)

    env = make_env(args.render)
    best = -21.0
    rewards = []

    # Reward normalization: Pong gives -1/+1 at point events
    # Scale them so TD error is meaningful with critic in [0,1]
    REWARD_SCALE = 5.0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        state = preprocess(obs)
        agent.reset_episode()

        ep_reward = 0.0
        done = False
        steps = 0
        ep_td = 0.0

        while not done:
            action, value = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            # REINFORCE-like: learn(reward, next_value=0)
            # δ = reward - V(s)  (Monte Carlo return for each step)
            # With reward scaling for critic range mismatch
            td = agent.learn(reward * REWARD_SCALE, 0.0, done)
            ep_td += abs(td)

            if not done:
                state = preprocess(obs)

        rewards.append(ep_reward)
        best = max(best, ep_reward)

        avg20 = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        avg100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)

        bar = "█" * max(0, min(int((ep_reward + 21) / 42 * 50), 50))
        bar += "░" * (50 - len(bar))

        print(
            f"  ep {ep:3d} |{bar}| "
            f"R {ep_reward:+5.1f}  "
            f"best {best:+5.1f}  "
            f"avg20 {avg20:+5.1f}  "
            f"avg100 {avg100:+5.1f}  "
            f"|δ| {ep_td/steps:.2f}"
        )

    env.close()
    total_time = time.perf_counter() - time.perf_counter()  # placeholder

    print("\n" + "═" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Best reward: {best:+5.1f}")
    print(f"  Avg last 20: {np.mean(rewards[-20:]):+5.1f}")
    print(f"  Avg last 100: {np.mean(rewards[-100:]):+5.1f}")
    print("═" * 60)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"  Total time: {time.perf_counter() - t0:.1f}s")
