#!/usr/bin/env python3
"""
Atari Pong × SNN Actor-Critic (Dopamine-STDP)
============================================
Pure Rust SIMD spiking agent plays Pong — no PyTorch, no TF, just SNN.

Architecture:
  Frame (RGB 210×160) → grayscale → resize 20×20 → flatten 400
  → PoissonEncoder → Izhikevich(256) → LIF(6) → softmax → action
  ↓
  TD-error δ = r + γV(s') - V(s) → dopamine gates STDP → Δw = η·δ·e
"""

import argparse
import time
import numpy as np

try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
except ImportError:
    print("Need: pip install gymnasium[atari] AutoROM && AutoROM --accept-license")
    raise

from taas_checkmate import PyActorCriticAgent

# ─── Config ────────────────────────────────────────────────

RESIZE = 20          # 20×20 = 400 inputs
N_HIDDEN = 256       # Izhikevich neurons (padded to ×8 internally)
N_ACTIONS = 6        # Pong: NOOP(0), FIRE(1), RIGHT(2), LEFT(3), RIGHTFIRE(4), LEFTFIRE(5)
N_STEPS = 30         # internal SNN simulation steps per decision
DT = 1.0
LR_ACTOR = 0.005
LR_CRITIC = 0.01
GAMMA = 0.99
TAU_TRACE = 20.0

N_EPISODES = 500
RENDER_EVERY = 50    # render milestone games


from PIL import Image

def preprocess(frame):
    """RGB 210×160 → grayscale → resize 20×20 → flatten → [0,1]"""
    img = Image.fromarray(frame).convert("L")
    img = img.resize((RESIZE, RESIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).ravel() / 255.0


def make_env(render=False):
    env = gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",        # (210, 160, 3) uint8
        frameskip=4,           # skip 4 frames per step
        repeat_action_probability=0.0,
        render_mode="human" if render else None,
    )
    return env


def smooth_rewards(rewards, window=20):
    if len(rewards) < window:
        return rewards
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--render", action="store_true", help="render all episodes")
    args = p.parse_args()

    n_input = RESIZE * RESIZE  # 400

    agent = PyActorCriticAgent(n_input, N_HIDDEN, N_ACTIONS, N_STEPS, DT, seed=42)
    agent.lr_actor = LR_ACTOR
    agent.lr_critic = LR_CRITIC
    agent.gamma = GAMMA
    agent.tau_trace = TAU_TRACE
    agent.w_max = 10.0
    agent.w_min = -10.0

    print("═" * 60)
    print("Atari Pong × SNN Actor-Critic (Dopamine-STDP)")
    print(f"  Input: {n_input} (20×20 grayscale)")
    print(f"  Hidden: {N_HIDDEN} Izhikevich neurons (SIMD fused)")
    print(f"  Output: {N_ACTIONS} actions (LIF readout)")
    print(f"  Learning: δ-TD × dopamine-STDP")
    print(f"  Episodes: {args.episodes}")
    print("═" * 60)

    env = make_env()
    best_reward = -21.0
    rewards_history = []
    times = []
    t0 = time.perf_counter()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        state = preprocess(obs)
        agent.reset_episode()

        ep_reward = 0.0
        step = 0
        done = False

        while not done:
            # SNN forward: Poisson → Izhikevich → LIF → softmax
            action, value = agent.act(state)

            # Environment step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Preprocess next state
            next_state = preprocess(obs)

            # Get next value for TD target
            _, next_value = agent.act(next_state) if not done else (0, 0.0)

            # SNN learn: dopamine-modulated STDP (Δw = η·δ·e)
            td_error = agent.learn(reward, next_value, done)

            state = next_state
            step += 1

            # render handled by render_mode="human" if set at creation

        rewards_history.append(ep_reward)
        best_reward = max(best_reward, ep_reward)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Stats
        avg20 = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
        bar_len = max(0, min(int((ep_reward + 21) / 42 * 50), 50))
        bar = "█" * bar_len + "░" * (50 - bar_len)

        print(
            f"  ep {ep:3d} |{bar}| "
            f"reward {ep_reward:+5.1f}  "
            f"best {best_reward:+5.1f}  "
            f"avg20 {avg20:+5.1f}  "
            f"steps {step:3d}"
        )

    env.close()
    total = time.perf_counter() - t0

    print()
    print("═" * 60)
    print("RESULTS")
    print(f"  Episodes: {args.episodes}")
    print(f"  Best reward: {best_reward:+5.1f}")
    print(f"  Avg last 20: {np.mean(rewards_history[-20:]):+5.1f}")
    print(f"  Avg last 50: {np.mean(rewards_history[-50:]):+5.1f}")
    print(f"  Total time:  {total:.1f}s")
    print(f"  Time/ep:     {total/args.episodes*1000:.0f}ms")
    print("═" * 60)


if __name__ == "__main__":
    main()
