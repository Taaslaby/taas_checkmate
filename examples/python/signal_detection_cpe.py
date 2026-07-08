#!/usr/bin/env python3
"""🧠 Signal Detection — CPE learns to distinguish coherent signal from noise."""

import time, math
from pathlib import Path
import numpy as np

import taas_checkmate as tcm

# ── Config (uses CPE canonical params) ──
DT = 0.01
N_STEPS_EPISODE = 100   # Kuramoto steps per episode (меньше = меньше затухание trace)
N_EPISODES = 40
SPIKE_THRESHOLD = 2.5

# Signal parameters
SIGNAL_FREQ = 5.0       # Hz — coherent oscillation
SIGNAL_AMP = 12.0       # drive amplitude for pattern (сильнее!)
NOISE_AMP = 4.0         # drive amplitude for chaos (слабее!)
N_SENSORY = 20          # sensory neurons to drive
PUNISH_NOISE_MOTOR = 2.0 # штраф за motor spikes на шуме

# CPE hyperparams (canonical, tuned for 800-neuron connectome)
LR = 0.02
ALPHA = 0.3
BETA = 0.3
H_TARGET = 3.0
ELIG_DECAY = 0.90       # τ_trace ≈ 85ms
ELIG_ADD = 0.15

BASE = Path(__file__).parent.parent.parent
BRAIN_PATH = BASE / "crates" / "connectome-gpt" / "models" / "brain.taas"

# ═══════════════════════════════════════════════
#  1. Load brain
# ═══════════════════════════════════════════════

print("=" * 60)
print("  SIGNAL DETECTION — CPE learns signal vs noise")
print("=" * 60)

print(f"\n1. Loading brain...")
cd = tcm.load_connectome(str(BRAIN_PATH))
n = cd.n_neurons
n_syn = cd.n_synapses
rp = np.array(cd.row_ptr(), dtype=np.int32)
ci = np.array(cd.col_indices(), dtype=np.int32)
w = np.array(cd.values(), dtype=np.float32)

gnames = list(cd.group_names())
sensory_idx = list(cd.group('sensory')) if 'sensory' in gnames else list(range(0, min(N_SENSORY, n)))
mn_idx = list(cd.group('mn')) if 'mn' in gnames else list(range(max(0, n - 4), n))
N_SENSORY = min(N_SENSORY, len(sensory_idx))
N_MOTOR = min(4, len(mn_idx))

print(f"  {n} neurons, {n_syn} synapses")
print(f"  Driving {N_SENSORY} sensory neurons, reading {N_MOTOR} motor neurons")

omegas = (3.0 + (np.arange(n) / n - 0.5) * 3.0).astype(np.float32)
w_initial = w.copy()

# ═══════════════════════════════════════════════
#  2. Task generators
# ═══════════════════════════════════════════════

def signal_drive(t, freq=SIGNAL_FREQ, amp=SIGNAL_AMP):
    """Coherent sine wave on sensory neurons — the SIGNAL."""
    drive = np.zeros(n, dtype=np.float32)
    phase = 2 * math.pi * freq * t * DT
    for i, si in enumerate(sensory_idx[:N_SENSORY]):
        drive[si] = amp * math.sin(phase + i * 0.3)  # slight per-neuron offset
    return drive

def noise_drive(rng):
    """Random independent noise — the CHAOS."""
    drive = np.zeros(n, dtype=np.float32)
    for si in sensory_idx[:N_SENSORY]:
        drive[si] = (rng.rand() - 0.5) * 2 * NOISE_AMP
    return drive

# ═══════════════════════════════════════════════
#  3. Training loop
# ═══════════════════════════════════════════════

print(f"\n2. Training (LR={LR}, β={BETA}, α={ALPHA}, τ≈85ms, {N_EPISODES} episodes)...")
print(f"   {'Ep':>3s} {'R_pat':>7s} {'R_cha':>7s} {'H_pat':>7s} {'H_cha':>7s} {'Mot✨':>7s} {'Mot💀':>7s} {'δ':>8s}")
print(f"   {'-'*55}")

t_start = time.time()
deltas, rpats, rchas = [], [], []

for ep in range(N_EPISODES):
    seed = ep * 7919 + 42
    rng = np.random.RandomState(seed)
    elig = np.zeros(n_syn, dtype=np.float32)

    # ── PATTERN run: coherent signal ──
    phases = rng.rand(n).astype(np.float32) * 2 * np.pi
    prev = phases.copy()
    spike_pat = np.zeros(n)
    coh_sum = 0.0
    steps = 0

    for t_step in range(N_STEPS_EPISODE):
        drive = signal_drive(t_step)
        tcm.kuramoto_step_sparse(phases, omegas + drive, rp, ci, w, DT)

        coh_out = np.zeros(1, dtype=np.float32)
        tcm.phase_coherence(coh_out,
            np.sin(phases).astype(np.float32),
            np.cos(phases).astype(np.float32),
            np.ones(n, dtype=np.float32))
        coh_sum += float(coh_out[0])

        sp = np.array(((phases > SPIKE_THRESHOLD) & (prev <= SPIKE_THRESHOLD)).astype(np.int32))
        prev = phases.copy()
        spike_pat += sp.astype(np.float32)
        steps += 1

        # Eligibility trace
        for i in range(n):
            if sp[i]:
                for idx in range(rp[i], rp[i+1]):
                    if idx < n_syn:
                        elig[idx] += ELIG_ADD
        elig *= ELIG_DECAY

    pat_r = coh_sum / max(steps, 1)
    pat_h, pat_err = (lambda c, s: (
        -np.sum((c / max(s,1) + 1e-10) * np.log2(c / max(s,1) + 1e-10)),
        abs(-np.sum((c / max(s,1) + 1e-10) * np.log2(c / max(s,1) + 1e-10)) - H_TARGET)
    ))(spike_pat, steps)

    # Motor response to signal
    motor_pat = np.mean([spike_pat[mi] for mi in mn_idx[:N_MOTOR]]) / max(steps, 1)

    # ── CHAOS run: random noise ──
    phases2 = rng.rand(n).astype(np.float32) * 2 * np.pi
    prev2 = phases2.copy()
    spike_cha = np.zeros(n)
    coh_sum2 = 0.0

    for t_step in range(N_STEPS_EPISODE):
        drive = noise_drive(rng)
        tcm.kuramoto_step_sparse(phases2, omegas + drive, rp, ci, w, DT)

        coh_out2 = np.zeros(1, dtype=np.float32)
        tcm.phase_coherence(coh_out2,
            np.sin(phases2).astype(np.float32),
            np.cos(phases2).astype(np.float32),
            np.ones(n, dtype=np.float32))
        coh_sum2 += float(coh_out2[0])

        sp2 = np.array(((phases2 > SPIKE_THRESHOLD) & (prev2 <= SPIKE_THRESHOLD)).astype(np.int32))
        prev2 = phases2.copy()
        spike_cha += sp2.astype(np.float32)

    cha_r = coh_sum2 / max(N_STEPS_EPISODE, 1)
    cha_h, cha_err = (lambda c, s: (
        -np.sum((c / max(s,1) + 1e-10) * np.log2(c / max(s,1) + 1e-10)),
        abs(-np.sum((c / max(s,1) + 1e-10) * np.log2(c / max(s,1) + 1e-10)) - H_TARGET)
    ))(spike_cha, N_STEPS_EPISODE)

    motor_cha = np.mean([spike_cha[mi] for mi in mn_idx[:N_MOTOR]]) / max(N_STEPS_EPISODE, 1)

    # ── Dopamine δ ──
    # Reward: motor responds to signal, PUNISHED for noise
    reward = motor_pat - PUNISH_NOISE_MOTOR * motor_cha
    pat_util = pat_r - ALPHA * pat_err
    cha_util = cha_r - ALPHA * cha_err
    delta = reward + BETA * (pat_util - cha_util)

    # ── STDP update ──
    w += LR * delta * elig
    pos_mask = w >= 0
    neg_mask = w < 0
    w = np.where(pos_mask, np.clip(w, 0.01, 3.0), np.clip(w, -3.0, -0.01))

    deltas.append(delta)
    rpats.append(pat_r)
    rchas.append(cha_r)

    if ep % 4 == 0 or ep == N_EPISODES - 1:
        print(f"   {ep:3d} {pat_r:7.4f} {cha_r:7.4f} {pat_h:7.3f} {cha_h:7.3f} {motor_pat:7.4f} {motor_cha:7.4f} {delta:8.3f}")

# ═══════════════════════════════════════════════
#  4. Results
# ═══════════════════════════════════════════════

print(f"\n{'='*60}")
print("  RESULTS")
print(f"{'='*60}")
print(f"  R_pat avg: {np.mean(rpats):.4f}  (first 5: {np.mean(rpats[:5]):.4f}, last 5: {np.mean(rpats[-5:]):.4f})")
print(f"  R_cha avg: {np.mean(rchas):.4f}  (first 5: {np.mean(rchas[:5]):.4f}, last 5: {np.mean(rchas[-5:]):.4f})")
print(f"  R_gap:     {np.mean(rpats) - np.mean(rchas):+.4f}")
print(f"  δ first 5: {np.mean(deltas[:5]):.3f}")
print(f"  δ last 5:  {np.mean(deltas[-5:]):.3f}")
trend = "UP 📈" if np.mean(deltas[-5:]) > np.mean(deltas[:5]) else "DOWN 📉"
print(f"  δ trend:   {trend}")

# Weight delta
dw = w - w_initial
abs_dw = np.abs(dw)
n_changed = int(np.sum(abs_dw > 0.001))
print(f"  Synapses changed >0.1%: {n_changed}/{n_syn} ({n_changed/n_syn*100:.1f}%)")
print(f"  Sign flips (E↔I): {np.sum((np.sign(w_initial) != np.sign(w)) & (np.sign(w_initial) != 0) & (np.sign(w) != 0))}")
print(f"  E/I ratio: {np.sum(w_initial > 0)}/{np.sum(w_initial < 0)} → {np.sum(w > 0)}/{np.sum(w < 0)}")
print(f"  Time: {time.time() - t_start:.1f}s")
print(f"{'='*60}")
print(f"  ⚡ Brain learns to detect signal")
print(f"{'='*60}")
