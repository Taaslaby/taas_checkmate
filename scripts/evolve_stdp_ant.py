#!/usr/bin/env python3
"""
Эволюция readout + dopamine-STDP на синапсы для MuJoCo Ant (GPU SPAUN).

Каждый шаг: forward → env.step → каждые 10 шагов apply_stdp(reward).
Поколения: motor readout эволюционирует (CEM), STDP учит синапсы онлайн.
"""

import sys
import numpy as np
import time
import argparse

sys.path.insert(0, "target/release")
from taas_checkmate import SpaunPipelineGPU
import gymnasium as gym


# ═══════════════════════════════════════════════════════════
#  AntController with dopamine-STDP
# ═══════════════════════════════════════════════════════════

class StdpAntController:
    """Ant controller with online dopamine-STDP on connectome synapses."""

    OBS_DIM = 105
    ACT_DIM = 8

    def __init__(self, brain_path: str, seed: int = 42,
                 sp_dim: int = 128, cortical_dim: int = 512,
                 stdp_lr: float = 0.01, stdp_w_min: float = 0.01,
                 stdp_w_max: float = 3.0, stdp_every: int = 10,
                 trace_decay: float = 0.95):
        self.seed = seed
        self.stdp_lr = stdp_lr
        self.stdp_w_min = stdp_w_min
        self.stdp_w_max = stdp_w_max
        self.stdp_every = stdp_every
        self.cortical_dim = cortical_dim

        self.pipe = SpaunPipelineGPU(brain_path, seed=seed)
        self.pipe.set_n_pixels(self.OBS_DIM)
        self.pipe.set_sp_dim(sp_dim)
        self.pipe.set_cortical_dim(cortical_dim)
        self.pipe.set_n_outputs(self.ACT_DIM)

        self.env = gym.make("Ant-v5", render_mode=None)

        self.forward_times = []
        self.reset_traces()

    def reset_traces(self):
        """Reset eligibility traces (called each episode)."""
        self.pipe.reset_traces()

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Brain forward → action."""
        t0 = time.time()
        logits = self.pipe.forward(obs.astype(np.float32), 5)
        self.forward_times.append(time.time() - t0)
        return np.clip(logits, -1.0, 1.0)

    def apply_reward_stdp(self, reward: float):
        """Apply dopamine-STDP with reward as delta signal."""
        delta = float(np.clip(reward / 5.0, -1.0, 1.0))
        self.pipe.apply_stdp(delta=delta, lr=self.stdp_lr,
                             w_min=self.stdp_w_min, w_max=self.stdp_w_max)

    def get_weights(self) -> np.ndarray:
        """Get motor readout weights."""
        return np.array(self.pipe.weights(), dtype=np.float32)

    def set_weights(self, w: np.ndarray):
        """Set motor readout weights."""
        w_flat = w.ravel().astype(np.float32)
        assert w_flat.size == self.ACT_DIM * self.cortical_dim, \
            f"expected {self.ACT_DIM * self.cortical_dim}, got {w_flat.size}"
        self.pipe.set_weights(w_flat)

    def run_episode(self, max_steps: int = 500, render: bool = False) -> float:
        """Run one episode. Returns total reward."""
        self.reset_traces()
        obs, _ = self.env.reset()
        total_reward = 0.0
        n_steps = 0

        # Create render env if needed
        renv = None
        if render:
            renv = gym.make("Ant-v5", render_mode="human")

        env = renv if render else self.env

        for step in range(max_steps):
            action = self.step(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            n_steps += 1

            # Apply dopamine-STDP every N steps
            if step % self.stdp_every == 0 and step > 0:
                self.apply_reward_stdp(reward)

            if render:
                env.render()

            if done or truncated:
                break

        if renv:
            renv.close()

        return total_reward

    def close(self):
        self.env.close()


# ═══════════════════════════════════════════════════════════
#  Evolution + STDP
# ═══════════════════════════════════════════════════════════

def evolve_stdp(args):
    """CEM evolution with online dopamine-STDP."""
    print(f"\n{'='*60}")
    print(f"🧬 Evolution + dopamine-STDP")
    print(f"   Brain: {args.brain}")
    print(f"   Generations: {args.generations}")
    print(f"   Population: {args.population}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   STDP lr: {args.stdp_lr}, every {args.stdp_every} steps")
    print(f"   Noise: {args.noise}, Elite: {args.elite}")
    print(f"{'='*60}\n")

    # Seed for reproducibility
    rng = np.random.RandomState(args.seed)

    # Create base controller to get weight shape
    base = StdpAntController(args.brain, seed=args.seed,
                              stdp_lr=args.stdp_lr,
                              stdp_w_min=args.stdp_w_min,
                              stdp_w_max=args.stdp_w_max,
                              stdp_every=args.stdp_every)
    w_shape = base.get_weights().shape
    n_weights = w_shape[0]
    print(f"Motor readout weights: {n_weights} ({base.ACT_DIM}×{base.cortical_dim})")
    print(f"Connectome: {base.pipe.n_neurons()} neurons, {base.pipe.n_synapses()} synapses")
    base.close()

    # Initialize population center
    center = rng.randn(n_weights).astype(np.float32) * args.noise * 0.1

    best_overall = -float("inf")
    best_w = center.copy()
    t_start = time.time()

    for gen in range(args.generations):
        gen_t0 = time.time()

        # Sample population
        pop_weights = []
        for i in range(args.population):
            w = center + rng.randn(n_weights).astype(np.float32) * args.noise
            pop_weights.append(w)

        # Evaluate population
        rewards = []
        controllers = []
        for i in range(args.population):
            ctrl = StdpAntController(args.brain, seed=args.seed + i + gen * args.population,
                                      stdp_lr=args.stdp_lr,
                                      stdp_w_min=args.stdp_w_min,
                                      stdp_w_max=args.stdp_w_max,
                                      stdp_every=args.stdp_every)
            ctrl.set_weights(pop_weights[i])
            controllers.append(ctrl)

            reward = ctrl.run_episode(max_steps=args.max_steps)
            rewards.append(reward)
            ctrl.close()

        rewards = np.array(rewards)

        # Select elite
        n_elite = max(1, int(args.population * args.elite))
        elite_idxs = np.argsort(rewards)[-n_elite:]
        elite_rewards = rewards[elite_idxs]
        elite_weights = np.mean([pop_weights[i] for i in elite_idxs], axis=0)

        # Update center
        center = elite_weights

        best_gen = rewards.max()
        if best_gen > best_overall:
            best_overall = best_gen
            best_w = pop_weights[rewards.argmax()].copy()

        gen_time = time.time() - gen_t0
        total_time = time.time() - t_start

        print(f"  Gen {gen+1:3d}/{args.generations}: "
              f"best={best_gen:7.1f}  "
              f"pop_max={rewards.max():7.1f}  "
              f"pop_mean={rewards.mean():7.1f}  "
              f"elite_mean={elite_rewards.mean():7.1f}  "
              f"time={gen_time:.1f}s  total={total_time:.0f}s")

    print(f"\n{'='*60}")
    print(f"🏆 Best reward: {best_overall:.1f}")
    print(f"⏱ Total time: {time.time() - t_start:.0f}s")

    # Demo with best weights
    if args.demo_episodes > 0:
        print(f"\n📺 Demo with best weights ({args.demo_episodes} episodes)...")
        ctrl = StdpAntController(args.brain, seed=999,
                                  stdp_lr=args.stdp_lr,
                                  stdp_w_min=args.stdp_w_min,
                                  stdp_w_max=args.stdp_w_max,
                                  stdp_every=args.stdp_every)
        ctrl.set_weights(best_w)
        for ep in range(args.demo_episodes):
            reward = ctrl.run_episode(max_steps=args.max_steps, render=args.render)
            print(f"  Demo ep {ep+1}: reward={reward:.1f}")
        ctrl.close()

    print(f"\n✅ Done!")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve + dopamine-STDP for Ant")
    parser.add_argument("--brain", default="/tmp/brain_100k.taas",
                        help="Path to .taas brain file")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--elite", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stdp-lr", type=float, default=0.01)
    parser.add_argument("--stdp-w-min", type=float, default=0.01)
    parser.add_argument("--stdp-w-max", type=float, default=3.0)
    parser.add_argument("--stdp-every", type=int, default=10)
    parser.add_argument("--demo-episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true",
                        help="Render demo episodes")

    args = parser.parse_args()
    evolve_stdp(args)
