#!/usr/bin/env python3
"""
SPAUN GPU → MuJoCo Ant-v5: 1M нейронов управляют муравьём.

Архитектура:
  obs (105-dim) → set_n_pixels(105) → forward(obs, steps) → 8 действий

  forward() делает:
    obs → encode_pixels → SP (128-dim) → cortex (512) →
    GNW attention broadcast (×steps) → GPU CSR SpMV + tanh (×steps, 1M нейр.) →
    motor_readout → 8 актуаторов [-1, 1]

Режимы:
  demo            — запуск с GPU мозгом, случайные или обученные веса
  evolve          — эволюционное обучение motor readout (hill-climbing)
  benchmark       — замер производительности (ms/forward, steps/s)
  ppo             — PPO с SPAUN features extractor (экспериментально)

Usage:
  python3 scripts/spaun_ant.py demo --render
  python3 scripts/spaun_ant.py evolve --generations 30
  python3 scripts/spaun_ant.py benchmark
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target", "release"))

try:
    from taas_checkmate import SpaunPipelineGPU
except ImportError:
    print("ERROR: Build GPU SPAUN first:")
    print("  cargo build --release --features python,cuda")
    sys.exit(1)

BRAIN_DEFAULT = "/tmp/brain_1m.taas"


# ═══════════════════════════════════════════════════════════════════
#  SPAUN → Ant Bridge
# ═══════════════════════════════════════════════════════════════════

class SpaunAntController:
    """Bridges SpaunPipelineGPU (1M neurons) with MuJoCo Ant-v5.

    Trick: set_n_pixels(105) so forward(obs, steps) does
    obs → random projection → SP → GNW → GPU connectome → motor_readout.
    """

    OBS_DIM = 105
    ACT_DIM = 8

    def __init__(
        self,
        brain_path: str | None = BRAIN_DEFAULT,
        seed: int = 42,
        sp_dim: int = 128,
        cortical_dim: int = 512,
        workspace_size: int = 32,
        internal_steps: int = 5,
    ):
        if brain_path and not os.path.exists(brain_path):
            print(f"ERROR: Connectome not found: {brain_path}")
            print(f"  Generate: cargo run --release --example bake_large_brain -p connectome-gpt -- 1000000 /tmp/brain_1m.taas")
            sys.exit(1)

        self.pipe = SpaunPipelineGPU(brain_path, seed=seed)
        self.n_neurons = self.pipe.n_neurons()
        self.n_synapses = self.pipe.n_synapses()
        print(f"  [SpaunAnt] Brain: {self.n_neurons:,} neurons, {self.n_synapses:,} synapses")

        cd = min(cortical_dim, self.n_neurons)
        self.pipe.set_n_pixels(self.OBS_DIM)
        self.pipe.set_sp_dim(sp_dim)
        self.pipe.set_cortical_dim(cd)
        self.pipe.set_workspace_size(workspace_size)
        self.pipe.set_n_outputs(self.ACT_DIM)
        self.internal_steps = internal_steps
        self.sp_dim = sp_dim
        self.cortical_dim = cd

        print(f"  [SpaunAnt] SP={sp_dim}, Cortex={cd}, WS={workspace_size}, Steps={internal_steps}")

        import gymnasium as gym
        self.env = gym.make("Ant-v5", render_mode=None)
        self._render_env = None

        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards: list[float] = []
        self.forward_times: list[float] = []

    # ── Motor weights ──

    def get_weights(self) -> np.ndarray:
        """Get motor readout weights (8 × cortical_dim)."""
        return np.array(self.pipe.weights(), dtype=np.float32)

    def set_weights(self, w: np.ndarray):
        """Set motor readout weights (8 × cortical_dim)."""
        w_flat = w.ravel().astype(np.float32)
        assert w_flat.size == self.ACT_DIM * self.cortical_dim, \
            f"Expected {self.ACT_DIM * self.cortical_dim} weights, got {w_flat.size}"
        self.pipe.set_weights(w_flat)

    # ── Core control ──

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (105,) → actions (8,) via GPU brain forward()."""
        t0 = time.perf_counter()
        logits = np.array(self.pipe.forward(
            obs.astype(np.float32), self.internal_steps
        ), dtype=np.float32)
        self.forward_times.append(time.perf_counter() - t0)
        return np.clip(logits, -1.0, 1.0)

    def run_episode(self, max_steps: int = 1000, render: bool = False) -> tuple[float, int, float]:
        """Run one episode. Returns (total_reward, steps_taken, time_seconds)."""
        if render:
            if self._render_env is None:
                import gymnasium as gym
                self._render_env = gym.make("Ant-v5", render_mode="human")
            obs, _ = self._render_env.reset()
        else:
            obs, _ = self.env.reset()

        total_reward = 0.0
        ep_start = time.time()

        for step in range(max_steps):
            action = self.step(obs)
            if render:
                obs, reward, done, truncated, _ = self._render_env.step(action)
            else:
                obs, reward, done, truncated, _ = self.env.step(action)

            total_reward += reward
            self.step_count += 1
            if done or truncated:
                break

        ep_time = time.time() - ep_start
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        return total_reward, step + 1, ep_time

    def close(self):
        self.env.close()
        if self._render_env:
            self._render_env.close()

    # ── Stats ──

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0

    @property
    def avg_forward_ms(self) -> float:
        return float(np.mean(self.forward_times[-100:])) * 1000 if self.forward_times else 0.0

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  📊 SPAUN Ant — Summary")
        print(f"{'='*60}")
        print(f"  Brain:    {self.n_neurons:,} neurons, {self.n_synapses:,} synapses")
        print(f"  GPU fwd:  {self.avg_forward_ms:.1f}ms/step ({self.internal_steps} internal)")
        if self.forward_times:
            print(f"  Throughput: {1.0 / np.mean(self.forward_times[-100:]):.0f} fwd/s")
        print(f"  Episodes: {self.episode_count} | Steps: {self.step_count}")
        if self.episode_rewards:
            r = self.episode_rewards
            print(f"  Reward:   last={r[-1]:.1f}  mean={np.mean(r):.1f}  best={max(r):.1f}")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════
#  Evolution training (hill-climbing on motor weights)
# ═══════════════════════════════════════════════════════════════════

def train_evolve(args):
    """Evolutionary training: hill-climb motor readout weights.

    Uses set_weights() to inject candidate weights, runs episode, keeps best.
    """
    print(f"\n{'='*60}")
    print(f"  🧬 Evolution Training — Motor Readout")
    print(f"  Generations: {args.generations}, Noise: {args.noise}, Elite: {args.elite}")
    print(f"{'='*60}\n")

    ctrl = SpaunAntController(
        brain_path=args.brain,
        seed=args.seed,
        sp_dim=args.sp_dim_val,
        cortical_dim=args.cortical_dim_val,
        workspace_size=args.workspace_size_val,
        internal_steps=args.internal_steps_val,
    )

    best_weights = ctrl.get_weights().copy()
    best_reward = float("-inf")

    for gen in range(args.generations):
        # Generate population
        pop_weights = []
        pop_rewards = []

        for i in range(args.population):
            noise = np.random.randn(*best_weights.shape).astype(np.float32) * args.noise
            w = best_weights + noise
            ctrl.set_weights(w)
            reward, n_steps, ep_time = ctrl.run_episode(
                max_steps=args.max_steps, render=False
            )
            pop_weights.append(w)
            pop_rewards.append(reward)

        # Select elite
        elite_count = max(1, int(args.population * args.elite))
        elite_idxs = np.argsort(pop_rewards)[-elite_count:]
        elite_reward = np.mean([pop_rewards[i] for i in elite_idxs])
        elite_weights = np.mean([pop_weights[i] for i in elite_idxs], axis=0)

        if elite_reward > best_reward:
            best_reward = elite_reward
            best_weights = elite_weights.copy()
            ctrl.set_weights(best_weights)

        if (gen + 1) % 5 == 0 or gen == args.generations - 1:
            print(f"  Gen {gen+1:3d}/{args.generations}: "
                  f"best={best_reward:7.1f}  elite={elite_reward:7.1f}  "
                  f"pop_max={max(pop_rewards):7.1f}  pop_mean={np.mean(pop_rewards):7.1f}")

    # Restore best and show demo
    ctrl.set_weights(best_weights)
    print(f"\n  🏆 Best reward: {best_reward:.1f}")
    print(f"  Running demo with best weights...")

    for ep in range(args.demo_episodes):
        reward, n_steps, ep_time = ctrl.run_episode(
            max_steps=args.max_steps,
            render=args.render and ep == args.demo_episodes - 1,
        )
        print(f"  Demo ep {ep+1}: reward={reward:.1f}, {n_steps} steps, {ep_time:.1f}s")

    ctrl.summary()
    ctrl.close()


# ═══════════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════════

def bench_mode(args):
    """Benchmark GPU brain inference speed."""
    print(f"\n{'='*60}")
    print(f"  🏎️  SPAUN GPU Benchmark — MuJoCo Ant")
    print(f"{'='*60}\n")

    scales = [(None, "default"), ("/tmp/brain_100k.taas", "100K")]
    if os.path.exists(BRAIN_DEFAULT):
        scales.append((BRAIN_DEFAULT, "1M"))

    for brain_path, label in scales:
        if brain_path and not os.path.exists(brain_path):
            print(f"  Skipping {label}: {brain_path} not found")
            continue

        print(f"─── {label} ───")
        ctrl = SpaunAntController(
            brain_path=brain_path,
            seed=42,
            sp_dim=args.sp_dim_val,
            cortical_dim=args.cortical_dim_val,
            workspace_size=args.workspace_size_val,
            internal_steps=args.internal_steps_val,
        )

        obs, _ = ctrl.env.reset()
        n_warmup = 10
        n_bench = args.n_steps

        # Warmup
        for _ in range(n_warmup):
            action = ctrl.step(obs)
            obs, _, _, _, _ = ctrl.env.step(action)

        ctrl.forward_times.clear()

        # Benchmark
        t0 = time.time()
        for _ in range(n_bench):
            obs, _, _, _, _ = ctrl.env.step(ctrl.step(obs))
        total = time.time() - t0

        avg_ms = np.mean(ctrl.forward_times) * 1000
        total_env_steps = n_bench

        print(f"  Neurons:   {ctrl.n_neurons:,}")
        print(f"  Synapses:  {ctrl.n_synapses:,}")
        print(f"  Forward:   {avg_ms:.2f}ms avg ({ctrl.internal_steps} internal steps)")
        print(f"  Throughput: {total_env_steps / total:.0f} env steps/s")
        print(f"  Brain speed: {ctrl.n_neurons / (avg_ms / 1000):,.0f} neur/s")
        print()

        ctrl.close()


# ═══════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════

def demo_mode(args):
    """Run Ant simulation with GPU brain."""
    print(f"\n{'='*60}")
    print(f"  🏃 MuJoCo Ant × SPAUN GPU ({args.episodes} episodes)")
    print(f"{'='*60}\n")

    for run in range(args.runs):
        ctrl = SpaunAntController(
            brain_path=args.brain,
            seed=args.seed + run * 100,
            sp_dim=args.sp_dim_val,
            cortical_dim=args.cortical_dim_val,
            workspace_size=args.workspace_size_val,
            internal_steps=args.internal_steps_val,
        )

        print(f"  Run {run+1}/{args.runs} (seed={args.seed + run * 100}):")
        for ep in range(args.episodes):
            reward, n_steps, ep_time = ctrl.run_episode(
                max_steps=args.max_steps,
                render=args.render and run == args.runs - 1,
            )
            sps = n_steps / ep_time if ep_time > 0 else 0
            print(f"  Ep {ep+1:3d}: reward={reward:8.1f}  steps={n_steps:4d}  "
                  f"{ep_time:.1f}s  {sps:.0f} step/s")

        ctrl.summary()
        ctrl.close()


# ═══════════════════════════════════════════════════════════════════
#  PPO (experimental)
# ═══════════════════════════════════════════════════════════════════

def ppo_mode(args):
    """PPO with SPAUN brain as features extractor."""
    print(f"\n{'='*60}")
    print(f"  🏋️ PPO with SPAUN Features Extractor (experimental)")
    print(f"{'='*60}\n")

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch as th
    import torch.nn as nn

    class SpaunFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.Space, features_dim: int = 8):
            super().__init__(observation_space, features_dim)
            self.brain = SpaunPipelineGPU(args.brain, seed=42)
            brain_neurons = self.brain.n_neurons()
            cd = min(args.cortical_dim_val, brain_neurons)
            self.brain.set_n_pixels(105)
            self.brain.set_sp_dim(args.sp_dim_val)
            self.brain.set_cortical_dim(cd)
            self.brain.set_workspace_size(args.workspace_size_val)
            self.brain.set_n_outputs(features_dim)
            self.internal_steps = args.internal_steps_val
            print(f"  [SpaunExtractor] Brain: {brain_neurons:,} neurons")

        def forward(self, observations: th.Tensor) -> th.Tensor:
            obs_np = observations.detach().cpu().numpy().astype(np.float32)
            B = obs_np.shape[0]
            features = np.zeros((B, 8), dtype=np.float32)
            for i in range(B):
                logits = self.brain.forward(obs_np[i], self.internal_steps)
                features[i] = np.array(logits, dtype=np.float32)
            return th.from_numpy(features).to(observations.device)

    env = gym.make("Ant-v5", render_mode="human" if args.render else None)

    policy_kwargs = dict(
        features_extractor_class=SpaunFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=8),
        net_arch=[64, 64],
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./logs/spaun_ant_ppo/",
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save("spaun_ant_ppo")
    print("Model saved to spaun_ant_ppo.zip")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def add_common_args(p):
    """Add shared args to a subparser or parent."""
    p.add_argument("--brain", default=BRAIN_DEFAULT, help="Path to .taas connectome")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sp-dim-val", type=int, default=128, help="Semantic pointer dim")
    p.add_argument("--cortical-dim-val", type=int, default=512, help="GNW cortex dim")
    p.add_argument("--workspace-size-val", type=int, default=32, help="GNW workspace")
    p.add_argument("--internal-steps-val", type=int, default=5, help="Brain steps per fwd")


def main():
    parser = argparse.ArgumentParser(
        description="SPAUN GPU → MuJoCo Ant: 1M neuron brain controls Ant-v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── demo ──
    demo = sub.add_parser("demo", help="Run Ant with GPU brain")
    add_common_args(demo)
    demo.add_argument("--episodes", type=int, default=3)
    demo.add_argument("--runs", type=int, default=1, help="Number of independent seeds")
    demo.add_argument("--max-steps", type=int, default=500)
    demo.add_argument("--render", action="store_true", help="Show MuJoCo GUI")

    # ── evolve ──
    evo = sub.add_parser("evolve", help="Evolutionary training of motor readout")
    add_common_args(evo)
    evo.add_argument("--generations", type=int, default=30)
    evo.add_argument("--population", type=int, default=8)
    evo.add_argument("--noise", type=float, default=0.3, help="Weight perturbation σ")
    evo.add_argument("--elite", type=float, default=0.25, help="Elite fraction")
    evo.add_argument("--max-steps", type=int, default=300)
    evo.add_argument("--demo-episodes", type=int, default=3)
    evo.add_argument("--render", action="store_true")

    # ── benchmark ──
    bench = sub.add_parser("benchmark", help="Benchmark GPU inference speed")
    add_common_args(bench)
    bench.add_argument("--n-steps", type=int, default=100, help="Benchmark iterations")

    # ── ppo ──
    ppo = sub.add_parser("ppo", help="PPO training with SPAUN features extractor")
    add_common_args(ppo)
    ppo.add_argument("--total-timesteps", type=int, default=50000)
    ppo.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode(args)
    elif args.mode == "evolve":
        train_evolve(args)
    elif args.mode == "benchmark":
        bench_mode(args)
    elif args.mode == "ppo":
        ppo_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
