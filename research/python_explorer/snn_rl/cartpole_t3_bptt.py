#!/usr/bin/env python3
"""
CartPole × T2GRAD (differentiable T3 + BPTT).

Архитектура:
  obs → encoder (Linear) → T3 drives (10 steps) → T3 BPTT → PST → logits → action
  advantage-weighted CrossEntropy backward через весь стек.

Ключевое отличие от REINFORCE: градиенты текут через T3 в encoder,
так что проекция obs→drives учится, а не фиксирована.
"""

import sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from taas_checkmate import (
    t2spigrad_forward, t2spigrad_backward,
    pst_linear_dim,
)


class T2SpigradFunction(torch.autograd.Function):
    """Differentiable T3 — forward + backward через Rust BPTT."""

    @staticmethod
    def forward(ctx, drives, n_osc, n_steps, dt,
                drive_gain=10.0, alpha=1.0,
                freq_min=0.5, freq_max=3.0,
                coupling=(0.3, 0.15, 0.08)):
        was_1d = drives.ndim == 2
        if was_1d:
            drives = drives.unsqueeze(0)
        batch_size, seq_len, n_total = drives.shape
        assert seq_len == n_steps
        assert n_total == 3 * n_osc

        pst_dim = pst_linear_dim(n_osc)
        device = drives.device
        drives_np = drives.detach().cpu().numpy()

        all_pst, all_trajs = [], []
        for b in range(batch_size):
            d = np.ascontiguousarray(drives_np[b])
            pst, traj = t2spigrad_forward(
                n_osc, n_steps, dt, drive_gain, alpha,
                freq_min, freq_max, coupling[0], coupling[1], coupling[2], d,
            )
            all_pst.append(pst)
            all_trajs.append(traj)

        pst_out = np.array(np.stack(all_pst, axis=0))
        ctx.trajs = all_trajs
        ctx.was_1d = was_1d
        result = torch.tensor(pst_out, device=device, dtype=drives.dtype)
        if was_1d:
            result = result.squeeze(0)
        return result

    @staticmethod
    def backward(ctx, grad_pst):
        was_1d = ctx.was_1d
        if was_1d:
            grad_pst = grad_pst.unsqueeze(0)
        batch_size = grad_pst.shape[0]
        device = grad_pst.device

        grad_drives_list = []
        for b in range(batch_size):
            gp = grad_pst[b].cpu().contiguous().numpy()
            gd = t2spigrad_backward(gp, ctx.trajs[b])
            grad_drives_list.append(np.ascontiguousarray(gd))

        grad_drives = np.array(np.stack(grad_drives_list, axis=0))
        result = torch.as_tensor(grad_drives, device=device, dtype=grad_pst.dtype)
        if was_1d:
            result = result.squeeze(0)
        return result, None, None, None, None, None, None, None, None

N_OSC = 64
N_STEPS_PER_CTRL = 10
DT = 0.05
FREQ_MIN, FREQ_MAX = 0.3, 2.5
DRIVE_GAIN = 10.0
ALPHA = 1.0
COUPLING = (0.3, 0.15, 0.08)
SEED = 42
LR = 3e-3
GAMMA = 0.99
ENTROPY_COEF = 0.05
LR_DECAY = 0.998
N_PST = pst_linear_dim(N_OSC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T3Encoder(nn.Module):
    """obs (4) → T3 drives (3*N_OSC)"""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3 * N_OSC, bias=False)
        nn.init.normal_(self.net.weight, std=0.5)

    def forward(self, obs):
        return self.net(obs)


class CartPoleT3BPTT:
    def __init__(self):
        self.encoder = T3Encoder().to(device)
        self.classifier = nn.Linear(N_PST, 2).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=LR,
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, LR_DECAY)
        self.ep_buf = []

    def _encode_batch(self, obs_batch):
        """(batch, 4) → T3 BPTT → PST → logits (batch, 2)"""
        batch_size = obs_batch.shape[0]
        drives_raw = self.encoder(obs_batch)  # (batch, 3*N_OSC)
        drives_seq = drives_raw.unsqueeze(1).expand(-1, N_STEPS_PER_CTRL, -1)  # (batch, 10, 3*N_OSC)
        pst = T2SpigradFunction.apply(
            drives_seq, N_OSC, N_STEPS_PER_CTRL, DT,
            DRIVE_GAIN, ALPHA, FREQ_MIN, FREQ_MAX, COUPLING,
        )
        return self.classifier(pst)

    def act(self, obs_np):
        """obs_np: (4,) → action"""
        obs = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)
        with torch.no_grad():
            logits = self._encode_batch(obs)
        probs = torch.softmax(logits, dim=1)
        action = torch.multinomial(probs.squeeze(0), 1).item()
        return action

    def learn(self):
        n = len(self.ep_buf)
        if n < 2:
            return 0.0

        returns = np.zeros(n, dtype=np.float32)
        G = 0.0
        for i in reversed(range(n)):
            G = self.ep_buf[i]["reward"] + GAMMA * G
            returns[i] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        obs_all = torch.from_numpy(np.array([e["obs"] for e in self.ep_buf])).float().to(device)
        actions_all = torch.tensor([e["action"] for e in self.ep_buf], device=device)
        adv_all = torch.tensor(returns, device=device)

        logits = self._encode_batch(obs_all)
        loss = self.loss_fn(logits, actions_all)
        weighted = (loss * adv_all).mean()

        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-30)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        weighted -= ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        weighted.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return weighted.item()

    def run_episode(self, render=False):
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        obs, _ = env.reset()
        self.ep_buf = []

        total = 0
        for t in range(500):
            action = self.act(obs)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.ep_buf.append({
                "obs": obs.copy(),
                "action": action,
                "reward": reward,
                "next_obs": obs_next.copy(),
            })
            total += reward
            obs = obs_next
            if done:
                break

        env.close()
        return total, t + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print(f"CartPole × T2GRAD BPTT — {N_OSC} osc, {N_STEPS_PER_CTRL} steps")
    print(f"  encoder: {4}→{3*N_OSC}, classifier: {N_PST}→2")
    print(f"  optimizer: Adam lr={LR}, device={device}")
    print("=" * 60)

    agent = CartPoleT3BPTT()
    best = 0.0
    t0 = time.perf_counter()
    window = []

    for ep in range(args.episodes):
        r, steps = agent.run_episode(render=args.render and ep % 50 == 0)
        loss = agent.learn()
        best = max(best, r)
        window.append(r)
        if len(window) > 50:
            window.pop(0)
        avg = sum(window) / len(window)

        bar = "█" * min(int(r / 2), 50) + "░" * max(50 - min(int(r / 2), 50), 0)
        print(f"  ep {ep+1:4d} |{bar}| {r:5.1f}  best {best:5.1f}  avg50 {avg:5.1f}  loss {loss:.4f}")

        if best >= 500:
            print(f"\n  SOLVED! Best reward 500 at episode {ep+1}")
            break

    total = time.perf_counter() - t0
    print(f"\n  Total: {total:.1f}s  |  Best reward: {best:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
